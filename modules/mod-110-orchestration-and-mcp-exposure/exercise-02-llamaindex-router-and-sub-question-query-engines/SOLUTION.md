# Exercise 02 — Solution: LlamaIndex `RouterQueryEngine` and `SubQuestionQueryEngine`

This is a reference solution for
`exercise-02-llamaindex-router-and-sub-question-query-engines.md`.
It is an **implementation** exercise. The learner wires two LlamaIndex
composition primitives against two corpora: a *router* that picks
which retriever to call based on query intent, and a *sub-question
engine* that decomposes a compound question into per-retriever
sub-questions and synthesises an answer across the results.

The teaching point: LlamaIndex's named primitives — `RouterQueryEngine`
and `SubQuestionQueryEngine` — map cleanly onto two different retrieval
graph shapes that production systems routinely hand-roll badly. The
router is "pick one of N retrievers," the sub-question engine is "ask
all of N retrievers and compose the answer." Picking the wrong one is
the common failure mode: routing a compound question collapses it to
one retriever and misses half the answer; sub-questioning a simple
question wastes tokens and inflates latency.

## 1. Solution overview

The deliverable has five parts:

1. **`src/indexes.py`** — build two `VectorStoreIndex` instances
   over two local corpora (e.g., "product docs" and "engineering
   runbooks"), each wrapped as a `QueryEngineTool` with a
   `description` the selector will key on.
2. **`src/router.py`** — a `RouterQueryEngine` with an LLM
   selector that picks one tool per query. Includes the selector
   prompt and a deterministic fallback.
3. **`src/subq.py`** — a `SubQuestionQueryEngine` that decomposes
   a compound query, dispatches per-tool, and synthesises one
   answer with citations to the sub-answers.
4. **`src/demo.py`** — a short driver that runs five canonical
   queries through both engines and prints the per-tool trace.
5. **`tests/test_engines.py`** — assertion-level tests that the
   router selects the intended tool on single-intent queries and
   the sub-question engine dispatches to *both* tools on compound
   queries. Tests mock the LLM so they are deterministic.

The exercise is scoped to *wiring and routing behaviour*; the
underlying retriever quality is covered by mod-106 and mod-107. If
the router consistently picks the correct tool and the sub-question
engine consistently fans out on compound queries, the exercise
passes.

## 2. Worked answer or implementation

### 2.1 The two primitives in one paragraph each

- **`RouterQueryEngine`** takes a list of `QueryEngineTool`s and a
  **selector** (LLM-based or embedding-based). On `.query(q)` it
  asks the selector which tool to call, calls exactly that tool
  (or the top-N if multi-select is enabled), and returns the
  tool's response. The selector keys off each tool's human-written
  `description`. Pick this when queries map to disjoint intents
  that live in different corpora. Reference:
  LlamaIndex `RouterQueryEngine` module docs,
  [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- **`SubQuestionQueryEngine`** takes the same list of tools plus
  a **question generator** (LLM). On `.query(q)` it prompts the
  generator to emit one or more `SubQuestion(sub_question, tool_name)`
  entries, dispatches each to the named tool, collects the
  per-tool answers, and synthesises them. Pick this when a
  single user query spans multiple corpora and needs all of them
  to answer correctly. Reference: LlamaIndex
  `SubQuestionQueryEngine` module docs.

### 2.2 `src/indexes.py` — tools over two corpora

```python
"""Build QueryEngineTools over two local corpora.

Assumes two directories under ./data/:
    data/product_docs/   — product documentation (marketing, PRDs, API ref)
    data/runbooks/       — ops runbooks, incident reports, oncall playbooks

The split is intentional: queries about *what the product does* belong
in product_docs; queries about *how we operate it* belong in runbooks.
That disjointness is what the router selector keys on."""
from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def _load_index(corpus_dir: Path) -> VectorStoreIndex:
    docs = SimpleDirectoryReader(input_dir=str(corpus_dir)).load_data()
    return VectorStoreIndex.from_documents(docs)


def build_tools(data_root: Path) -> list[QueryEngineTool]:
    product_idx = _load_index(data_root / "product_docs")
    runbook_idx = _load_index(data_root / "runbooks")

    product_tool = QueryEngineTool(
        query_engine=product_idx.as_query_engine(similarity_top_k=5),
        metadata=ToolMetadata(
            name="product_docs",
            description=(
                "Answers questions about PRODUCT CAPABILITIES, API surface, "
                "pricing, feature flags, and roadmap. Use this when the "
                "user is asking WHAT the product does or HOW A USER uses it."
            ),
        ),
    )
    runbook_tool = QueryEngineTool(
        query_engine=runbook_idx.as_query_engine(similarity_top_k=5),
        metadata=ToolMetadata(
            name="runbooks",
            description=(
                "Answers questions about OPERATIONAL PROCEDURES, incident "
                "response, oncall rotations, deployment steps, and known "
                "failure modes. Use this when the user is asking HOW WE "
                "OPERATE the product or WHAT TO DO during an incident."
            ),
        ),
    )
    return [product_tool, runbook_tool]
```

**Choices and why:**

- **The `description` is the router's input.** It must name the
  kind of query the tool is for, not the kind of documents it
  contains. "Answers questions about X" beats "contains X" — the
  selector is matching intent, not inventory.
- **Explicit disjointness in the descriptions.** "What the product
  does" vs. "how we operate it" is the boundary. Boundaries that
  are obvious to a human are the ones the selector gets right.
- **Each tool wraps its own `as_query_engine`.** The router does
  not retrieve; it dispatches. All retrieval tuning (top-k,
  reranker, hybrid) lives inside each tool's engine.

### 2.3 `src/router.py` — the `RouterQueryEngine`

```python
"""RouterQueryEngine with an LLM selector.

The selector is given each tool's description and the incoming query
and asked which tool to call. The LLM returns a single index; the
router dispatches to that tool's engine and returns the response."""
from __future__ import annotations

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool


def build_router(tools: list[QueryEngineTool]) -> RouterQueryEngine:
    # LLMSingleSelector: choose exactly one tool. For queries that span
    # both corpora, prefer SubQuestionQueryEngine (see subq.py); do not
    # paper over compound queries with LLMMultiSelector here.
    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
        verbose=True,  # emits the selector's choice + reasoning on each call
    )
```

**Choices and why:**

- **`LLMSingleSelector`, not `LLMMultiSelector`.** Multi-select
  dispatches to every tool the selector picks and concatenates
  responses — that is a worse implementation of the sub-question
  engine. Keep the router single-select; fan-out belongs to
  `SubQuestionQueryEngine`.
- **`verbose=True`.** The selector's reasoning is where routing
  bugs show up. Logging it is cheap at dev time; silence it in
  prod via a callback handler instead of by removing it.
- **No embedding-based fallback here.** `EmbeddingSelector` is
  cheaper and faster but keys on description-vs-query similarity,
  which collapses when descriptions use the same technical
  vocabulary. The LLM selector reasons about *intent*; use the
  embedding selector only when tool descriptions are highly
  distinct lexically.

### 2.4 `src/subq.py` — the `SubQuestionQueryEngine`

```python
"""SubQuestionQueryEngine that decomposes a compound query and fans
out to the same two tools the router uses."""
from __future__ import annotations

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool


def build_subq(tools: list[QueryEngineTool]) -> SubQuestionQueryEngine:
    # LLMQuestionGenerator prompts the LLM to decompose the query into
    # (sub_question, tool_name) tuples. Each sub-question is dispatched
    # to the named tool; results are synthesised into one answer.
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=LLMQuestionGenerator.from_defaults(),
        verbose=True,
        use_async=False,  # set True in prod; False keeps the demo trace readable
    )
```

**Choices and why:**

- **Same tool list as the router.** Exposing the same tools to
  both primitives is deliberate: the router and the sub-question
  engine live side-by-side and pick different shapes of the
  same retrieval graph. If the tool lists diverge, the sub-question
  engine can route to tools the router cannot, and the two
  engines answer the "simple" and "compound" halves of the
  product differently.
- **`use_async=False` in the demo**, `True` in prod. The demo is
  easier to read with sequential sub-question calls; the prod
  path should fan out concurrently (sub-questions are
  independent).
- **No multi-step reasoning loop here.** `SubQuestionQueryEngine`
  does a single fan-out and synth. If the task needs "answer,
  critique, re-query," that is the LangGraph shape from
  exercise-03, not the sub-question engine.

### 2.5 `src/demo.py` — driver

```python
"""Run five canonical queries through both engines and print the
per-tool trace. Use this to verify that routing and decomposition
behave as intended before adding retrievers to the mix."""
from __future__ import annotations

from pathlib import Path

from indexes import build_tools
from router import build_router
from subq import build_subq

QUERIES = [
    # Single-intent: should route to product_docs
    "What authentication methods does the product support?",
    # Single-intent: should route to runbooks
    "What is the oncall procedure when the ingest queue backs up?",
    # Compound: should fan out to both via SubQuestionQueryEngine
    "What retention policy does the product advertise, and what is the "
    "runbook step to purge data beyond that retention?",
    # Edge: ambiguous — selector should pick one and surface its reasoning
    "How do we handle data deletion?",
    # Edge: out-of-corpus — neither tool should confidently answer
    "What is the capital of France?",
]


def main() -> None:
    tools = build_tools(Path("./data"))
    router = build_router(tools)
    subq = build_subq(tools)

    print("=" * 60)
    print("ROUTER (LLMSingleSelector)")
    print("=" * 60)
    for q in QUERIES:
        print(f"\nQ: {q}")
        resp = router.query(q)
        print(f"A: {resp}")

    print("\n" + "=" * 60)
    print("SUBQUESTION (LLMQuestionGenerator)")
    print("=" * 60)
    for q in QUERIES:
        print(f"\nQ: {q}")
        resp = subq.query(q)
        print(f"A: {resp}")


if __name__ == "__main__":
    main()
```

### 2.6 `tests/test_engines.py` — behavioural tests

```python
"""Deterministic tests for the router and sub-question engine.

We mock the LLM selector and question generator so the tests assert on
dispatch behaviour, not on model output. The underlying retriever
responses are stubbed. These tests are intentionally small — they
protect the *wiring*, not the retrieval quality (that belongs to
mod-106/107/109)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_router_picks_runbooks_for_oncall_question(router_with_fakes):
    router, selector, tools = router_with_fakes
    selector.select.return_value = _single_select(tool_index=1)  # runbooks

    _ = router.query("What is the oncall procedure for queue backups?")

    _, called_tool = _find_called_tool(tools)
    assert called_tool.metadata.name == "runbooks"


def test_router_picks_product_for_capability_question(router_with_fakes):
    router, selector, tools = router_with_fakes
    selector.select.return_value = _single_select(tool_index=0)  # product_docs

    _ = router.query("What authentication methods does the product support?")

    _, called_tool = _find_called_tool(tools)
    assert called_tool.metadata.name == "product_docs"


def test_subq_fans_out_to_both_tools_on_compound(subq_with_fakes):
    subq, question_gen, tools = subq_with_fakes
    question_gen.generate.return_value = [
        _sub_question("What retention does the product advertise?", "product_docs"),
        _sub_question("How do we purge data in the runbook?", "runbooks"),
    ]

    _ = subq.query(
        "What retention policy does the product advertise, and what is "
        "the runbook step to purge data beyond that retention?"
    )

    called = {t.metadata.name for t in tools if t.query_engine.query.called}
    assert called == {"product_docs", "runbooks"}


# -------- fixtures & helpers (abbreviated) --------

@pytest.fixture
def router_with_fakes(fake_tools):
    """Yield a RouterQueryEngine whose selector is a MagicMock."""
    from llama_index.core.query_engine import RouterQueryEngine
    selector = MagicMock()
    router = RouterQueryEngine(selector=selector, query_engine_tools=fake_tools)
    return router, selector, fake_tools


@pytest.fixture
def subq_with_fakes(fake_tools):
    """Yield a SubQuestionQueryEngine whose question_gen is a MagicMock."""
    from llama_index.core.query_engine import SubQuestionQueryEngine
    question_gen = MagicMock()
    subq = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=fake_tools, question_gen=question_gen
    )
    return subq, question_gen, fake_tools


# _single_select, _sub_question, _find_called_tool, and the fake_tools
# fixture construct the LlamaIndex-shaped objects the engines expect.
# See the LlamaIndex test fixtures for the exact shapes.
```

**Choices and why:**

- **Mock the selector and the question generator; let the engines
  dispatch for real.** The thing we are testing is the *dispatch
  wiring*, not whether the LLM made the right call. Testing the
  LLM's decisions belongs to mod-109 (eval), not here.
- **Assert on tool names, not on text.** Response text is
  unstable; the tool the engine chose to call is a stable
  observable.
- **Golden compound-query test.** The one test that catches
  "accidentally used the router for a compound query" — the most
  common misuse of these primitives.

## 3. Validation steps

1. `python -m src.demo` runs end-to-end and prints both engines'
   traces without exceptions.
2. On the single-intent product query, the router's `verbose`
   log shows `product_docs` selected.
3. On the single-intent runbooks query, the router's `verbose`
   log shows `runbooks` selected.
4. On the compound query, the sub-question engine's `verbose` log
   shows at least one sub-question per tool and the final synth
   references both.
5. `pytest -q tests/` passes with mocked selectors.
6. Tool descriptions pass a simple sanity audit: a reviewer who
   has never seen the data can predict which tool each sample
   query will route to.
7. No tool has `description == ""`. A missing description is the
   most common cause of silent mis-routing.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Tool descriptions | Each `QueryEngineTool` has a description that names the *kind of query* it answers, not the *kind of documents* it contains. Boundary with sibling tools is explicit. |
| Router selector | `LLMSingleSelector` (or `EmbeddingSelector` with justified lexical distinctness). `verbose=True` or a logged callback captures the choice. |
| Router semantics | On single-intent queries the selected tool is predictable and matches a human's expected routing. |
| Sub-question generator | Compound queries decompose into `(sub_question, tool_name)` tuples that reference the real tool names; no hallucinated tools. |
| Sub-question synth | The final response mentions evidence from each dispatched tool, not just one. |
| Separation of concerns | Retrieval tuning (top-k, reranker, hybrid) lives inside each tool's engine, not at the router / sub-question layer. |
| Tests | Deterministic tests mock the selector and generator and assert on dispatched tool names, not on LLM text. |
| Observability | Either `verbose=True` or a LlamaIndex callback manager captures per-query routing choices for review. |
| Fallback | For out-of-corpus queries, both engines degrade gracefully (empty answer or explicit "I don't have that in corpus"); they do not confidently synthesise a wrong answer. |
| Use case fit | Router is used for disjoint-intent queries; sub-question engine is used for queries that genuinely span tools. Not the reverse. |

Fail-fast smells:

- Router dispatches to the wrong tool on >20% of a small gold set.
  The descriptions are too similar; rewrite them to name the
  disjoint intent explicitly.
- Sub-question engine emits one sub-question per query, always,
  for every input. The question generator is prompted too
  conservatively; check the generator template.
- Router is used on compound queries and quietly returns only the
  product-docs half. Switch to the sub-question engine for those
  queries, or add an upstream classifier that routes between the
  two engines.
- Tool descriptions mention file paths, team names, or internal
  jargon. The selector cannot use those; it reasons from the
  user's query text, which will not contain them.

## 5. Common mistakes

- **Writing descriptions by what the corpus *contains* instead of
  what the tool *answers*.** "Contains PRDs" is an inventory
  statement; "answers questions about product capabilities" is
  the selector's actual input. The selector matches query intent
  against the latter, not the former.
- **Reaching for `LLMMultiSelector` to handle compound queries.**
  Multi-select dispatches all the selected tools and concatenates
  their responses; it does not decompose the query. The result is
  that both tools answer the full query instead of their part of
  it, producing a blob that mixes irrelevant context. Use
  `SubQuestionQueryEngine` for compound queries; keep the router
  single-select.
- **Using the sub-question engine for everything.** Every call
  pays the question-generation latency and tokens; for simple
  single-intent queries the overhead is pure waste. Put a cheap
  router in front of the sub-question engine, or feature-flag the
  decomposition per intent.
- **Sharing a reranker across tools at the router layer.** The
  router dispatches; reranking the dispatched response is a
  category error. If both tools benefit from the same reranker,
  wrap it inside each tool's `as_query_engine(..., node_postprocessors=[...])`.
- **No `verbose` logging and no callback manager.** The selector's
  choice is the single most useful debug observable; shipping
  without it means routing bugs show up as "the answer is wrong"
  without a trail.
- **Letting the tool list drift between router and sub-question
  engine.** If the router has tools A/B and the sub-question
  engine has tools A/C, compound queries get dispatched to C and
  the router never sees C. Keep a single `tools` list and pass it
  to both engines.
- **Testing the LLM's decisions, not the wiring.** Tests that call
  the real LLM with `verbose` logging are integration tests, not
  unit tests — they flake on model updates. Mock the selector in
  unit tests and keep the integration tests on a gold set in
  mod-109.

## 6. References

- LlamaIndex `RouterQueryEngine` —
  [docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/).
  Selectors (`LLMSingleSelector`, `LLMMultiSelector`,
  `EmbeddingSelector`) and tool metadata contracts.
- LlamaIndex `SubQuestionQueryEngine` —
  [docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/).
  Question-generation prompts and sub-question dispatch.
- LlamaIndex `QueryEngineTool` and `ToolMetadata` —
  [docs.llamaindex.ai](https://docs.llamaindex.ai/en/stable/api_reference/tools/).
- Upstream modules consumed by this exercise:
  - mod-106 hybrid retrieval / fusion — the tool's retriever can be
    a hybrid retriever; the router does not care.
  - mod-107 reranking and context assembly — reranking lives
    inside each tool's query engine as a `node_postprocessor`.
- Risk framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML05 Model Theft / ML06 Supply Chain — the
    router's selector prompt is a trust boundary; mis-descriptions
    leak intent across tools.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    *LLM Plugin Compromise* motivates treating each tool as a
    distinct trust boundary even inside one application.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure* function requires observability of the routing
    decisions; `verbose=True` or a callback handler is the
    minimum.
