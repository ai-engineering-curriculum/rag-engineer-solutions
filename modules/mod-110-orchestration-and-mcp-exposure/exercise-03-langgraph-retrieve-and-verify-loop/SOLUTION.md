# Exercise 03 — Solution: LangGraph Retrieve-and-Verify Loop

This is a reference solution for
`exercise-03-langgraph-retrieve-and-verify-loop.md`. It is an
**implementation** exercise. The learner builds a stateful LangGraph
graph with a retrieve node, a verify node, a rewrite node, and a
conditional back-edge that re-queries when the verify step is not
satisfied. The graph terminates on either a verified result or a
budget exhaustion.

The teaching point: a retrieve-and-verify loop is the textbook case
for a graph runtime. Written as a `while` loop in Python, the state,
the budget, the retrieval history, and the termination predicate
scatter across local variables; the trace is lost; a human reviewer
cannot resume the run from the middle. Written as a LangGraph
`StateGraph`, the state is a typed dict that every node reads and
writes, the budget counter lives in state, the trace is produced
automatically, and a checkpointer can pause/resume the run on a
human-in-the-loop interrupt. The graph *shape* is what makes the
behaviour debuggable.

## 1. Solution overview

The deliverable has five parts:

1. **`src/state.py`** — the `State` `TypedDict`: query, history of
   retrieval rounds, candidate documents, verify verdict, rewrite
   counter, budget cap, final answer.
2. **`src/nodes.py`** — four pure functions mapping `State → State
   delta`: `retrieve`, `verify`, `rewrite`, `synthesize`. Each
   node reads a slice of `State` and returns a dict of fields to
   update.
3. **`src/graph.py`** — `StateGraph` wiring: nodes, entry point,
   conditional edges from `verify` (to `synthesize` on pass, to
   `rewrite` on fail-and-budget-remaining, to `synthesize` on
   fail-but-budget-exhausted with a `partial=True` annotation),
   a `MemorySaver` checkpointer, and an optional
   `interrupt_before=["rewrite"]` for human-in-the-loop review.
4. **`src/demo.py`** — a driver that runs three canonical queries
   (one that passes first retrieval, one that needs a rewrite,
   one that exhausts the budget) and prints the state deltas per
   node.
5. **`tests/test_loop.py`** — tests with stubbed retriever and
   verifier that assert termination semantics: pass-on-first,
   pass-after-N-rewrites, and terminate-on-budget.

The exercise is scoped to *the graph wiring and termination
contract*; the retriever behind `retrieve` and the verifier model
behind `verify` are stubbed. If the graph terminates correctly on
each path and the final `State` shows the full trace, the exercise
passes.

## 2. Worked answer or implementation

### 2.1 Loop semantics in one paragraph

Each loop iteration is: `retrieve(query)` → `verify(query, docs)` →
if the verdict is `pass`, go to `synthesize` and return. If the
verdict is `fail` and the rewrite budget is not exhausted, go to
`rewrite(query, history)` and loop back to `retrieve` with the new
query. If the budget is exhausted, go to `synthesize` with a
`partial=True` flag so the caller knows the answer is best-effort.
The budget is enforced in the conditional edge, not inside the
`rewrite` node, so the graph has a single termination decision
point.

### 2.2 `src/state.py` — the graph state

```python
"""Typed state for the retrieve-and-verify loop.

Every node reads from State and returns a partial dict of fields to
update. LangGraph merges the partial dict into the running state."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from operator import add


class RetrievalRound(TypedDict):
    query: str
    doc_ids: list[str]
    verdict: Literal["pass", "fail", "pending"]
    verdict_reason: str


class State(TypedDict):
    # Immutable-after-entry
    original_query: str
    max_rewrites: int                     # budget cap (e.g., 3)

    # Rewritten on each loop iteration
    current_query: str

    # Appended on each loop iteration. Annotated[..., add] tells LangGraph
    # to merge list fields by concatenation, not replacement.
    history: Annotated[list[RetrievalRound], add]

    # Scratch between nodes
    candidates: list[dict]                # [{doc_id, text, score}, ...]
    verdict: Literal["pass", "fail", "pending"]
    verdict_reason: str

    # Set by synthesize
    answer: str
    partial: bool                         # True iff budget exhausted
```

**Choices and why:**

- **`Annotated[list[...], add]` for `history`.** Without the
  reducer, LangGraph replaces the field on each node write and
  the earlier retrieval rounds are lost. The `add` reducer (from
  `operator.add`) tells the graph to concatenate; every node
  writes just the rounds it added.
- **`original_query` and `max_rewrites` are set at entry and
  never touched.** Keeping input constants in state (rather than
  closing over them in the nodes) means the checkpointer can
  resume a run with no implicit context.
- **`candidates` is scratch, not history.** Each retrieval
  overwrites `candidates`; the `history` list keeps enough to
  reconstruct what was tried.
- **`partial` is explicit.** A caller that gets an answer with
  `partial=True` can decide to display it with a disclaimer; a
  silent `answer` would look identical to a verified one.

### 2.3 `src/nodes.py` — the four nodes

```python
"""Nodes for the retrieve-and-verify graph.

Each node is a pure function of State. External dependencies
(retriever, verifier model, rewriter model, LLM synth) are injected
so the nodes are deterministic in tests."""
from __future__ import annotations

from typing import Callable

from state import RetrievalRound, State


# -------- dependencies (typed protocols; stubs in tests) --------

Retriever = Callable[[str], list[dict]]                    # query -> [{doc_id, text, score}]
Verifier = Callable[[str, list[dict]], tuple[str, str]]    # (query, docs) -> ("pass"|"fail", reason)
Rewriter = Callable[[str, list[RetrievalRound]], str]      # (original, history) -> new_query
Synth = Callable[[str, list[dict]], str]                   # (query, docs) -> answer


def make_retrieve_node(retriever: Retriever) -> Callable[[State], dict]:
    def retrieve(state: State) -> dict:
        docs = retriever(state["current_query"])
        return {
            "candidates": docs,
            "verdict": "pending",
            "verdict_reason": "",
        }
    return retrieve


def make_verify_node(verifier: Verifier) -> Callable[[State], dict]:
    def verify(state: State) -> dict:
        verdict, reason = verifier(state["current_query"], state["candidates"])
        round_ = RetrievalRound(
            query=state["current_query"],
            doc_ids=[d["doc_id"] for d in state["candidates"]],
            verdict=verdict,
            verdict_reason=reason,
        )
        return {
            "verdict": verdict,
            "verdict_reason": reason,
            "history": [round_],  # reducer-concat into state["history"]
        }
    return verify


def make_rewrite_node(rewriter: Rewriter) -> Callable[[State], dict]:
    def rewrite(state: State) -> dict:
        new_q = rewriter(state["original_query"], state["history"])
        return {
            "current_query": new_q,
            # Clear scratch; the next retrieve will repopulate it.
            "candidates": [],
            "verdict": "pending",
            "verdict_reason": "",
        }
    return rewrite


def make_synthesize_node(synth: Synth) -> Callable[[State], dict]:
    def synthesize(state: State) -> dict:
        # On "pass", use the latest candidates. On budget-exhaust, synth
        # from the best round in history (here: last-seen; a more
        # thoughtful implementation would pick the highest-scored round).
        docs = state["candidates"]
        answer = synth(state["original_query"], docs)
        return {
            "answer": answer,
            "partial": state["verdict"] != "pass",
        }
    return synthesize
```

**Choices and why:**

- **Factory functions, not module-level handlers.** Each node is
  built with its external dependency injected. This keeps the
  graph testable: the test suite builds the graph with stub
  callables; production builds it with the real retriever and
  LLM.
- **`verify` writes the `history` round.** Putting the write in
  `verify` (not `retrieve`) means the appended `RetrievalRound`
  contains the verdict and reason; a round in history always has
  a verdict.
- **`rewrite` clears `candidates` and `verdict`.** The next
  retrieve will overwrite them, but clearing eagerly avoids
  stale scratch bleeding into the next verify if a node is
  re-run during replay.

### 2.4 `src/graph.py` — the `StateGraph`

```python
"""Wire the retrieve-and-verify graph.

Topology:
    START → retrieve → verify ─┬── pass    ───────────→ synthesize → END
                               └── fail
                                   ├─ budget_remaining → rewrite → retrieve
                                   └─ budget_exhausted → synthesize (partial)

The budget is checked in the conditional edge from `verify`, not in
the `rewrite` node. One termination point keeps the trace clean."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    make_retrieve_node,
    make_verify_node,
    make_rewrite_node,
    make_synthesize_node,
)
from state import State


def _route_from_verify(state: State) -> str:
    if state["verdict"] == "pass":
        return "synthesize"
    # Count attempts from history; rewrite nodes do not run before verify
    # so len(history) == number of attempts so far.
    attempts = len(state["history"])
    if attempts >= state["max_rewrites"] + 1:  # +1: the initial attempt
        return "synthesize"  # partial=True set by synthesize()
    return "rewrite"


def build_graph(retriever, verifier, rewriter, synth, *, use_checkpointer=True):
    g = StateGraph(State)
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("verify", make_verify_node(verifier))
    g.add_node("rewrite", make_rewrite_node(rewriter))
    g.add_node("synthesize", make_synthesize_node(synth))

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "verify")
    g.add_conditional_edges(
        "verify",
        _route_from_verify,
        {"synthesize": "synthesize", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("synthesize", END)

    checkpointer = MemorySaver() if use_checkpointer else None
    return g.compile(
        checkpointer=checkpointer,
        # Uncomment for HITL review before a rewrite fires:
        # interrupt_before=["rewrite"],
    )
```

**Choices and why:**

- **One termination decision in `_route_from_verify`.** Every
  decision — pass / fail-and-retry / fail-and-exhausted — lives
  in one function. If termination is wrong, there is one place
  to look.
- **Attempts are counted from `history`, not a separate
  counter.** The history grows by one on every verify; the
  number of attempts is `len(history)`. A separate counter would
  be a second source of truth that drifts under replay.
- **`+1` for the initial attempt.** `max_rewrites=3` means three
  rewrites are allowed, which means four total attempts
  (initial + three rewrites). Being explicit about the off-by-one
  in the condition avoids the classic "loops one more time than
  advertised" bug.
- **`MemorySaver` for the demo**, Redis/SQLite for prod.
  Checkpointer makes the graph resumable and inspectable. Adding
  `interrupt_before=["rewrite"]` lets a human approve each
  rewrite — exercise this once by uncommenting the kwarg.

### 2.5 `src/demo.py` — three canonical runs

```python
"""Drive the graph against three canonical queries with scripted
retriever and verifier behaviour, so the three termination paths
are each exercised."""
from __future__ import annotations

from graph import build_graph


def _retriever_factory(script):
    """Return a retriever that pops the next scripted result per call."""
    calls = iter(script)

    def retrieve(_query):
        return next(calls)
    return retrieve


def _verifier_factory(verdicts):
    calls = iter(verdicts)

    def verify(_query, _docs):
        return next(calls)
    return verify


def _rewriter_that_appends_word(word):
    def rewrite(original, history):
        return f"{original} {word}{len(history)}"
    return rewrite


def _echo_synth(query, docs):
    ids = ",".join(d["doc_id"] for d in docs)
    return f"ANSWER(query={query!r}, docs=[{ids}])"


def run_scenario(name, query, max_rewrites, retriever, verifier, rewriter):
    print(f"\n=== {name} ===")
    graph = build_graph(retriever, verifier, rewriter, _echo_synth)
    state_in = {
        "original_query": query,
        "current_query": query,
        "max_rewrites": max_rewrites,
        "history": [],
        "candidates": [],
        "verdict": "pending",
        "verdict_reason": "",
        "answer": "",
        "partial": False,
    }
    # Config carries the checkpointer thread id.
    config = {"configurable": {"thread_id": name}}
    for step in graph.stream(state_in, config=config):
        for node, delta in step.items():
            print(f"  {node}: {sorted(delta)}")
    final = graph.get_state(config).values
    print(f"  FINAL: partial={final['partial']}, answer={final['answer']!r}")


if __name__ == "__main__":
    # Scenario 1: pass on first attempt.
    run_scenario(
        "pass_first",
        "retention policy",
        max_rewrites=3,
        retriever=_retriever_factory([[{"doc_id": "d1", "text": "...", "score": 0.9}]]),
        verifier=_verifier_factory([("pass", "doc d1 answers the question")]),
        rewriter=_rewriter_that_appends_word("clarify"),
    )

    # Scenario 2: pass after one rewrite.
    run_scenario(
        "pass_after_rewrite",
        "retention",
        max_rewrites=3,
        retriever=_retriever_factory([
            [{"doc_id": "d2", "text": "...", "score": 0.4}],
            [{"doc_id": "d1", "text": "...", "score": 0.9}],
        ]),
        verifier=_verifier_factory([
            ("fail", "doc d2 is on wrong topic"),
            ("pass", "doc d1 answers the clarified question"),
        ]),
        rewriter=_rewriter_that_appends_word("clarify"),
    )

    # Scenario 3: budget exhausted; terminate partial.
    run_scenario(
        "budget_exhausted",
        "retention",
        max_rewrites=1,  # one rewrite => two attempts total
        retriever=_retriever_factory([
            [{"doc_id": "d2", "text": "...", "score": 0.4}],
            [{"doc_id": "d3", "text": "...", "score": 0.5}],
        ]),
        verifier=_verifier_factory([
            ("fail", "doc d2 off-topic"),
            ("fail", "doc d3 off-topic"),
        ]),
        rewriter=_rewriter_that_appends_word("clarify"),
    )
```

### 2.6 `tests/test_loop.py` — termination semantics

```python
"""Assert that the three termination paths behave as advertised.

The nodes are injected with stubs so the tests are deterministic. The
assertions are on `partial`, `len(history)`, and the final answer
shape — not on LLM text."""
from __future__ import annotations

import pytest

from graph import build_graph


def _state(q, max_rewrites=3):
    return {
        "original_query": q,
        "current_query": q,
        "max_rewrites": max_rewrites,
        "history": [],
        "candidates": [],
        "verdict": "pending",
        "verdict_reason": "",
        "answer": "",
        "partial": False,
    }


def _run(graph, state, thread):
    config = {"configurable": {"thread_id": thread}}
    for _ in graph.stream(state, config=config):
        pass
    return graph.get_state(config).values


def test_pass_on_first_attempt():
    graph = build_graph(
        retriever=lambda q: [{"doc_id": "d1", "text": "x", "score": 0.9}],
        verifier=lambda q, docs: ("pass", "ok"),
        rewriter=lambda orig, hist: pytest.fail("rewrite must not run on pass"),
        synth=lambda q, docs: "answer",
    )
    final = _run(graph, _state("q"), "t1")
    assert final["partial"] is False
    assert len(final["history"]) == 1
    assert final["answer"] == "answer"


def test_pass_after_one_rewrite():
    verdicts = iter([("fail", "no"), ("pass", "ok")])
    docs = iter([[{"doc_id": "a", "text": "", "score": 0.1}],
                 [{"doc_id": "b", "text": "", "score": 0.9}]])
    graph = build_graph(
        retriever=lambda q: next(docs),
        verifier=lambda q, _d: next(verdicts),
        rewriter=lambda orig, hist: orig + "?",
        synth=lambda q, d: "answer",
    )
    final = _run(graph, _state("q"), "t2")
    assert final["partial"] is False
    assert len(final["history"]) == 2


def test_budget_exhausted_terminates_partial():
    graph = build_graph(
        retriever=lambda q: [{"doc_id": "x", "text": "", "score": 0.2}],
        verifier=lambda q, _d: ("fail", "no"),
        rewriter=lambda orig, hist: orig + "?",
        synth=lambda q, d: "best-effort answer",
    )
    final = _run(graph, _state("q", max_rewrites=1), "t3")
    assert final["partial"] is True
    # max_rewrites=1 means 2 attempts (initial + one rewrite).
    assert len(final["history"]) == 2
```

## 3. Validation steps

1. `python -m src.demo` prints three traces. The `pass_first`
   trace shows exactly one `retrieve` and one `verify` before
   `synthesize`; `pass_after_rewrite` shows two `retrieve`s and
   one `rewrite`; `budget_exhausted` shows two of each and
   terminates with `partial=True`.
2. `pytest -q tests/test_loop.py` passes all three tests.
3. `graph.get_state(config).values["history"]` on a terminated
   run contains one `RetrievalRound` per attempt, each with a
   verdict and reason.
4. Setting `max_rewrites=0` causes every query to terminate after
   the first attempt (pass or partial); no `rewrite` node runs.
5. With `interrupt_before=["rewrite"]` enabled, a failing verify
   pauses the graph and `graph.get_state` returns a `next` of
   `("rewrite",)`; calling `graph.invoke(None, config)` resumes
   execution.
6. The graph can be visualised: `graph.get_graph().draw_mermaid()`
   emits a diagram with the expected nodes and the conditional
   back-edge from `verify` to `rewrite`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| State typing | `State` is a `TypedDict` or Pydantic model; `history` uses a reducer (`Annotated[list[...], add]`) so rounds concat instead of replace. |
| Node purity | Each node returns a partial `State` dict; no side effects beyond the injected dependencies. |
| Single termination point | The decision to synth-or-rewrite lives in exactly one conditional edge function. |
| Budget enforcement | Off-by-one in the budget check is correct: `max_rewrites=N` → `N+1` total attempts. |
| Partial vs verified | `partial=True` is set on budget exhaustion; a verified run has `partial=False`. Callers can distinguish them. |
| Trace completeness | `history` contains one round per attempt with query, doc ids, verdict, and reason. |
| Checkpointer | A `MemorySaver` (or equivalent) is attached; `graph.get_state(config)` returns inspectable state. |
| Deterministic tests | Tests inject stubs for retriever, verifier, rewriter, synth and assert on state shape, not LLM text. |
| HITL path exercised | At least once, the graph is run with `interrupt_before=["rewrite"]` and resumed; the demo or tests show the pause/resume contract. |
| Observability | Node-level progress visible via `graph.stream(...)` or LangSmith callbacks. |

Fail-fast smells:

- `history` is replaced on each write (no reducer) and the final
  state has only the last round. The audit trail is lost.
- The termination predicate is split across the `rewrite` node
  and the conditional edge. A change to the budget behaviour now
  requires editing both places.
- `rewrite` writes directly to a module-level `attempts_so_far`
  counter. Replay breaks silently; checkpointer-driven resume
  skips the counter increment.
- `partial=True` is not surfaced in the returned answer object
  and the caller cannot tell a budget-exhausted answer from a
  verified one.
- The graph has no checkpointer. A production run cannot be
  paused, resumed, or replayed; HITL is impossible.

## 5. Common mistakes

- **Writing the loop as a `while` in Python and only using
  LangGraph as a wrapper around it.** The whole point of the
  graph runtime is that the loop is modelled; collapsing it back
  into a single Python loop discards state inspection, replay,
  and HITL. If the graph has one node containing a `while`, it
  is not actually a graph.
- **No reducer on `history`.** Writes replace instead of
  concatenate, and the final state has only the last round. The
  classic LangGraph footgun.
- **Counting attempts with a separate counter field.** The
  history grows by one per attempt; use `len(state["history"])`
  as the counter. A duplicate counter is a second source of
  truth that will drift under replay.
- **Enforcing the budget inside `rewrite`.** `rewrite` runs *because*
  the edge chose to route there. Enforcing the budget inside
  `rewrite` means the budget check runs *after* the rewrite was
  already decided; the rewrite either fires anyway or the node
  no-ops. Enforce in the conditional edge, where the decision is
  being made.
- **Confusing `max_rewrites` with `max_attempts`.** A rewrite
  budget of 3 means up to 3 rewrites fire, which is up to 4
  total retrieval attempts. Name the config field and the
  comparison in the edge consistently so the off-by-one is
  explicit.
- **No `partial` flag on the answer.** A budget-exhausted answer
  is best-effort; collapsing it to a `string` identical in shape
  to a verified answer deprives downstream code (UI, eval,
  logs) of the information they need to treat it differently.
- **Verifier calls the main LLM at high temperature.** A
  non-deterministic verifier makes the pass/fail boundary
  unstable; the same retrieval produces different verdicts on
  re-run. Set temperature 0, or use a classifier-style prompt
  with a forced JSON schema.
- **Rewriter prompted without the retrieval history.** Without
  seeing what was tried and why it failed, the rewriter produces
  cosmetic query edits that re-fail. Pass the history into the
  rewriter prompt and keep rewrites meaningfully different.
- **No `thread_id` in the checkpointer config.** All runs share
  one checkpoint and overwrite each other; resume behaves
  randomly. Pass a unique `thread_id` per logical run.

## 6. References

- LangGraph docs — [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/).
  Relevant topics: `StateGraph`, conditional edges, reducers
  (`Annotated[list, add]`), `MemorySaver`,
  `interrupt_before`/`interrupt_after`, `graph.stream`,
  `graph.get_state`.
- LangGraph conceptual guide on state and reducers —
  [langchain-ai.github.io/langgraph/concepts/low_level/](https://langchain-ai.github.io/langgraph/concepts/low_level/).
- LangGraph checkpointer and persistence docs —
  [langchain-ai.github.io/langgraph/concepts/persistence/](https://langchain-ai.github.io/langgraph/concepts/persistence/).
- LangGraph human-in-the-loop —
  [langchain-ai.github.io/langgraph/concepts/human_in_the_loop/](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/).
- Upstream modules consumed by this exercise:
  - mod-106 hybrid retrieval — the retriever behind the
    `retrieve` node can be hybrid.
  - mod-108 retrieval-specific prompting — the verifier prompt
    is a specialisation of the refusal/evidence-grading contract
    from exercise 108-02.
- Risk framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant: ML04 Membership Inference / ML09 Output Integrity
    — the verify step is a structural guard against shipping
    unverified output.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    *Evade ML Model* motivates logging every rewrite — attacker
    queries that trigger the rewrite loop are a signal.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    *Govern* and *Measure* functions both require pause-and-
    review paths; the HITL `interrupt_before` edge is the
    artifact.
