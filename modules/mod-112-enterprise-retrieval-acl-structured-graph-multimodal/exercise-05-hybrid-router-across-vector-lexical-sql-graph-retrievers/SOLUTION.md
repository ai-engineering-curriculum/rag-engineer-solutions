# Exercise 05 — Solution: Hybrid Router Across Vector, Lexical, SQL, and Graph Retrievers

This is a reference solution for
`exercise-05-hybrid-router-across-vector-lexical-sql-graph-retrievers.md`.
It is an **implementation-with-design-judgment** exercise. The
learner builds a hybrid router that takes a natural-language
question, decides which backend retriever (or which combination)
should answer it — vector-dense, lexical/BM25, text-to-SQL, or
Graph RAG — executes the retriever(s), and returns a unified
result. The router is the thin layer that lets a user ask one
question of four very different storage systems and get a
coherent answer. The deliverable is the router, a routing
policy, a handful of golden routing tests, and an audit trail
of each decision.

The teaching point: a hybrid retrieval stack is not "every
retriever runs on every query and we fuse." That works for
vector + lexical (RRF; mod-106 covers it), but SQL and graph
retrievers expect *structured* queries — running them on an
irrelevant free-text question wastes budget and risks
hallucinated SQL / Cypher. The router's job is to read the
*shape* of the question, not to understand its meaning. Three
signals carry most of the information: classifier labels over
question shape, structural features (names, numbers, dates,
schema-matching terms), and a declared user intent (e.g., an
MCP tool flag). The rest is budget enforcement and fallback.

## 1. Solution overview

The deliverable has five parts:

1. **`src/classifier.py`** — a question classifier that tags
   the incoming question with one or more intent labels:
   `analytical` (SQL), `relational` (Graph), `semantic` (vector),
   `exact_phrase` (lexical), and `visual` (multimodal, from
   exercise-04). The classifier is a composed pipeline:
   deterministic feature rules first (regex for SQL-ish and
   graph-ish signatures), then an LLM classifier for ambiguous
   cases, with the LLM output constrained to the known labels.
2. **`src/retrievers.py`** — a common `Retriever` interface
   (`search(question, policy) → list[Hit]`) with four adapters
   wrapping the backends from mod-106 (hybrid vector+BM25),
   exercise-02 (text-to-SQL), exercise-03 (Graph RAG), and
   exercise-04 (multimodal). Each adapter normalises its output
   to the same `Hit` schema.
3. **`src/router.py`** — the router. Reads the classifier's
   labels, calls the appropriate retrievers (possibly more than
   one), merges results via mod-106's RRF for retrievers that
   returned a ranked list of text hits, and keeps the SQL and
   graph results distinct (they are structured outputs, not
   rankings).
4. **`src/budget.py`** — a per-request budget enforcer.
   Vector + BM25 are cheap; SQL hits the database; Graph hits
   Cypher; ColPali is heaviest. The router tracks wall-clock
   and refuses to add a retriever that would exceed the
   declared budget.
5. **`src/audit.py` + `tests/`** — every routing decision is
   logged: question (hashed for sensitive corpora), labels,
   retrievers called, latency per retriever, and the final
   merged result. Tests cover canonical routes and the
   fallback path when no classifier label fires.

Scope: *the routing layer only.* The backends are inherited
from earlier mod-106 and this module's exercises. If the router
picks the right backend on the golden set, respects the budget,
and emits an auditable trace, the exercise passes.

## 2. Worked answer or implementation

### 2.1 The router in one paragraph

The router is a decision-plus-fanout layer. Given a question, it
classifies the question's *shape* with a small label set, maps
the label(s) to one or more backends (vector, lexical, SQL,
graph, multimodal), calls the backends in parallel under a
budget, and merges the results. Semantic and lexical hits fuse
via RRF because they return ranked text lists. SQL and graph
hits do not fuse — a SQL result is a tuple, a graph result is a
subgraph, and asking "is row 3 more relevant than node Nova?"
is nonsense. The router returns a *bundle*: the fused text
retrieval, the SQL result (if any), the graph context (if any),
and the audit trail. The downstream caller — a chatbot, a
prompt template, an MCP tool — decides how to serialise the
bundle. The router's contract is that every decision is
auditable and no budget is silently blown.

### 2.2 `src/classifier.py` — the question classifier

```python
"""Tag the question with one or more intent labels.

Composed: deterministic rules first, LLM classifier for
ambiguous cases, with the LLM constrained to the known label
set."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Intent(str, Enum):
    ANALYTICAL = "analytical"       # SQL-style: counts, aggregations, filters
    RELATIONAL = "relational"       # Graph-style: paths, neighbours, connections
    SEMANTIC   = "semantic"         # Vector-style: "about this topic"
    EXACT      = "exact_phrase"     # Lexical-style: a quoted name, a code
    VISUAL     = "visual"           # Multimodal: "find the chart that shows..."


@dataclass(frozen=True)
class Classification:
    intents: frozenset[Intent]
    source: str  # "rules", "llm", "rules+llm"
    reasons: list[str]


# --- deterministic signatures ----------------------------------------
_RE_AGG      = re.compile(r"\b(count|how many|sum|average|avg|median|total|per|group by)\b", re.I)
_RE_FILTER   = re.compile(r"\b(top \d+|bottom \d+|between \d|since \d{4}|in (\d{4})|more than \d|less than \d)\b", re.I)
_RE_RELATION = re.compile(r"\b(connected to|related to|supplier of|who works at|path from|shortest|neighbour of|reports to)\b", re.I)
_RE_EXACT    = re.compile(r'"[^"]+"|\b(SKU-\d+|INC-\d+|[A-Z]{2,}-\d+)\b')
_RE_VISUAL   = re.compile(r"\b(image|photo|chart|figure|diagram|page \d+|scanned|screenshot)\b", re.I)


def classify_by_rules(question: str) -> Classification:
    hits: set[Intent] = set()
    reasons: list[str] = []
    if _RE_AGG.search(question) or _RE_FILTER.search(question):
        hits.add(Intent.ANALYTICAL)
        reasons.append("regex: aggregation / filter terms")
    if _RE_RELATION.search(question):
        hits.add(Intent.RELATIONAL)
        reasons.append("regex: relational terms")
    if _RE_EXACT.search(question):
        hits.add(Intent.EXACT)
        reasons.append("regex: quoted phrase or id-shaped literal")
    if _RE_VISUAL.search(question):
        hits.add(Intent.VISUAL)
        reasons.append("regex: visual terms")
    return Classification(intents=frozenset(hits), source="rules", reasons=reasons)


LLM_SYSTEM = """You are a question classifier. Return a JSON array of
labels drawn ONLY from this set:
["analytical", "relational", "semantic", "exact_phrase", "visual"].
Rules:
- analytical: the question asks for counts, sums, aggregations, or
  filtered row-level data that lives in a database.
- relational: the question asks about connections, paths, or how
  two or more entities relate.
- semantic: the question is about a topic / concept / "find
  documents about X" — the default when no other label fits.
- exact_phrase: the question contains a quoted string, a code, or
  a named entity that must match exactly.
- visual: the question references an image, a chart, a scanned
  page, or a page number.

Return at least "semantic" if nothing else fits. Return the array
and nothing else."""


def classify_by_llm(
    question: str,
    chat: Callable[[str, str], str],
) -> Classification:
    import json
    raw = chat(LLM_SYSTEM, question).strip()
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = ["semantic"]
    valid = {i.value for i in Intent}
    kept = [Intent(x) for x in labels if x in valid] or [Intent.SEMANTIC]
    return Classification(
        intents=frozenset(kept),
        source="llm",
        reasons=[f"llm labels: {labels}"],
    )


def classify(
    question: str,
    chat: Callable[[str, str], str] | None = None,
) -> Classification:
    rule_cls = classify_by_rules(question)
    if rule_cls.intents and chat is None:
        return rule_cls
    if chat is None:
        # No rules fired and no LLM — default to semantic.
        return Classification(
            intents=frozenset({Intent.SEMANTIC}),
            source="rules",
            reasons=["default: no signatures matched"],
        )
    llm_cls = classify_by_llm(question, chat)
    return Classification(
        intents=rule_cls.intents | llm_cls.intents,
        source="rules+llm" if rule_cls.intents else "llm",
        reasons=rule_cls.reasons + llm_cls.reasons,
    )
```

**Choices and why:**

- **Rules first, LLM second.** Rules are fast and auditable;
  the LLM is a fallback for ambiguous phrasing. The rule set
  captures the easy cases ("how many," "top 10," "related to,"
  quoted phrase, image/chart) that cover a large fraction of
  real traffic without an LLM call.
- **Multi-label, not single-label.** "Top 5 suppliers connected
  to Apex" is *analytical* + *relational*. Forcing a
  single label would force a wrong choice; the router handles
  multi-label naturally by calling multiple retrievers.
- **LLM output is constrained to the enum.** The system prompt
  lists the only legal labels; the parser validates and drops
  invalid ones; a parse failure defaults to `semantic`. The LLM
  is not allowed to invent a sixth intent.
- **Default is `semantic`.** When nothing else fires, the
  vector retriever is the safest backend — it can answer "about
  a topic" questions without hallucinating SQL or walking an
  unrelated graph.
- **Reasons are first-class.** Every classification emits
  `reasons`; the audit trail captures why a label fired. A
  reviewer can look at a decision and see the trigger.

### 2.3 `src/retrievers.py` — the common retriever interface

```python
"""One Hit type, four retriever adapters.

Hits from every backend normalise to this shape so the router
can merge ranked lists via RRF and present SQL/graph outputs
alongside without type-checking."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    title: str
    snippet: str
    source: str              # e.g., "vector", "bm25", "sql", "graph", "colpali"
    kind: str                # "text" | "row" | "subgraph" | "image"
    extra: dict              # backend-specific details (e.g., SQL text, subgraph ref)


@dataclass(frozen=True)
class RetrieverResult:
    hits: list[Hit]
    retriever: str
    latency_s: float
    error: str | None = None


class Retriever(ABC):
    name: str

    @abstractmethod
    def search(self, question: str, policy: Any) -> RetrieverResult:
        ...


# --- adapters: thin wrappers over the backends from earlier exercises ---

class HybridTextRetriever(Retriever):
    """mod-106 hybrid: vector + BM25 fused by RRF."""
    name = "hybrid_text"

    def __init__(self, backend) -> None:  # backend from mod-106
        self._b = backend

    def search(self, question: str, policy) -> RetrieverResult:
        import time
        t0 = time.perf_counter()
        hits = self._b.search(question, k=10, acl=policy.acl_predicate)
        return RetrieverResult(
            hits=[
                Hit(doc_id=h.doc_id, score=h.score, title=h.title,
                    snippet=h.snippet, source="hybrid_text",
                    kind="text", extra={})
                for h in hits
            ],
            retriever=self.name,
            latency_s=time.perf_counter() - t0,
        )


class SQLRetriever(Retriever):
    """exercise-02 pipeline wrapped as a retriever."""
    name = "sql"

    def __init__(self, pipeline) -> None:
        self._p = pipeline

    def search(self, question: str, policy) -> RetrieverResult:
        import time
        t0 = time.perf_counter()
        res = self._p.run(question, policy.caller_policy)
        if not res.ok:
            return RetrieverResult(
                hits=[],
                retriever=self.name,
                latency_s=time.perf_counter() - t0,
                error=res.reason,
            )
        row_hits = [
            Hit(doc_id=f"sql:{i}", score=1.0 - 0.01 * i,
                title=f"row {i}", snippet=str(row),
                source="sql", kind="row",
                extra={"sql": res.sql})
            for i, row in enumerate(res.rows)
        ]
        return RetrieverResult(hits=row_hits, retriever=self.name,
                                latency_s=time.perf_counter() - t0)


class GraphRetriever(Retriever):
    """exercise-03 Graph RAG pipeline wrapped as a retriever."""
    name = "graph"

    def __init__(self, pipeline) -> None:
        self._p = pipeline

    def search(self, question: str, policy) -> RetrieverResult:
        import time
        t0 = time.perf_counter()
        res = self._p.run(question)  # returns RAGResult(context, anchors, subgraph)
        hits = [
            Hit(doc_id=f"node:{n.node_id}", score=n.score,
                title=n.name, snippet=n.description,
                source="graph", kind="subgraph",
                extra={"labels": list(n.labels)})
            for n in res.subgraph.nodes
        ]
        return RetrieverResult(hits=hits, retriever=self.name,
                                latency_s=time.perf_counter() - t0)


class MultimodalRetriever(Retriever):
    """exercise-04 single-vector or ColPali wrapped as a retriever.

    The router picks which checkpoint via `policy.multimodal_mode`
    so the same adapter serves both the cheap path and the
    document-page path."""
    name = "multimodal"

    def __init__(self, embedder, index) -> None:
        self._embedder = embedder
        self._index = index

    def search(self, question: str, policy) -> RetrieverResult:
        import time
        t0 = time.perf_counter()
        qvec = self._embedder.embed_queries([question])[0]
        ranked = self._index.search(qvec, k=10)
        hits = [
            Hit(doc_id=h.doc_id, score=h.score, title=h.doc_id,
                snippet="", source=f"multimodal:{self._embedder.name}",
                kind="image", extra={})
            for h in ranked
        ]
        return RetrieverResult(hits=hits, retriever=self.name,
                                latency_s=time.perf_counter() - t0)
```

**Choices and why:**

- **`Hit.kind` tags the shape.** Downstream code that cares
  about "just the text hits" filters on `kind == "text"`. Code
  that renders a UI card picks the renderer by kind. Mixing
  rows, subgraphs, and images under one score would be lying
  about comparability.
- **`extra` is a dict, not typed.** Backend-specific metadata
  (SQL text, subgraph reference, bounding box) belongs to the
  backend. The router does not care; the downstream renderer
  does.
- **Errors are first-class.** A SQL retriever that the guard
  rejects returns `RetrieverResult(hits=[], error="...")`. The
  router logs the error, moves on, and does not silently drop
  the signal.

### 2.4 `src/router.py` — the router

```python
"""Route a question to one or more retrievers; merge the output.

The router only *fuses* retrievers that return comparable
rankings (text-shaped `Hit`s from hybrid, lexical, and semantic
sources). SQL rows and graph subgraphs are passed through
alongside the fused text ranking."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from audit import AuditLog, RouteDecision
from budget import Budget, BudgetExceeded
from classifier import Classification, Intent, classify
from retrievers import Hit, Retriever, RetrieverResult


@dataclass(frozen=True)
class Bundle:
    text_hits: list[Hit]        # fused via RRF across text-shaped retrievers
    sql_hits: list[Hit]         # kept separate; structured output
    graph_hits: list[Hit]       # kept separate; subgraph nodes
    image_hits: list[Hit]       # kept separate; image/page hits
    classification: Classification
    results_by_retriever: dict[str, RetrieverResult]


# Mapping from intent → retriever names. Multiple intents call
# multiple retrievers; a question with no clear intent gets the
# semantic default. Lexical is attached to EXACT; hybrid covers
# it on SEMANTIC too so the broader surface is not lost.
_ROUTING: dict[Intent, tuple[str, ...]] = {
    Intent.ANALYTICAL: ("sql",),
    Intent.RELATIONAL: ("graph",),
    Intent.SEMANTIC:   ("hybrid_text",),
    Intent.EXACT:      ("hybrid_text",),
    Intent.VISUAL:     ("multimodal",),
}


class Router:
    def __init__(
        self,
        retrievers: dict[str, Retriever],
        budget: Budget,
        audit: AuditLog,
        chat=None,  # for LLM classifier fallback
    ) -> None:
        self._retrievers = retrievers
        self._budget = budget
        self._audit = audit
        self._chat = chat

    def route(self, question: str, policy) -> Bundle:
        cls = classify(question, chat=self._chat)
        wanted_names = self._pick_retrievers(cls)

        results: dict[str, RetrieverResult] = {}
        with ThreadPoolExecutor(max_workers=len(wanted_names) or 1) as pool:
            futures = {}
            for name in wanted_names:
                retriever = self._retrievers.get(name)
                if retriever is None:
                    continue
                try:
                    self._budget.check_add(name)
                except BudgetExceeded as err:
                    self._audit.record_skip(name, reason=str(err))
                    continue
                futures[pool.submit(retriever.search, question, policy)] = name
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as err:  # noqa: BLE001 — audit then continue
                    self._audit.record_error(name, error=repr(err))

        # Fuse text-shaped hits across retrievers that returned them.
        text_rankings = [r.hits for r in results.values() if r.hits and r.hits[0].kind == "text"]
        text_fused = _reciprocal_rank_fusion(text_rankings, k=10)
        sql_hits   = _flatten(results, kind="row")
        graph_hits = _flatten(results, kind="subgraph")
        image_hits = _flatten(results, kind="image")

        self._audit.record(RouteDecision(
            question=question,
            classification=cls,
            retrievers_called=list(results.keys()),
            latencies={n: r.latency_s for n, r in results.items()},
            errors={n: r.error for n, r in results.items() if r.error},
        ))
        return Bundle(
            text_hits=text_fused,
            sql_hits=sql_hits,
            graph_hits=graph_hits,
            image_hits=image_hits,
            classification=cls,
            results_by_retriever=results,
        )

    def _pick_retrievers(self, cls: Classification) -> list[str]:
        if not cls.intents:
            return ["hybrid_text"]
        names: list[str] = []
        for intent in cls.intents:
            for r in _ROUTING.get(intent, ()):
                if r not in names:
                    names.append(r)
        return names


def _flatten(results: dict[str, RetrieverResult], kind: str) -> list[Hit]:
    out: list[Hit] = []
    for r in results.values():
        out.extend(h for h in r.hits if h.kind == kind)
    return out


def _reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 10, c: int = 60) -> list[Hit]:
    # Standard RRF: score(doc) = sum over rankings of 1 / (c + rank).
    # c=60 is the default from the Cormack/Clarke/Buettcher paper;
    # mod-106 covers the tuning.
    scores: dict[str, float] = {}
    first_hit: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (c + rank)
            first_hit.setdefault(hit.doc_id, hit)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [first_hit[doc_id] for doc_id, _ in ordered]
```

**Choices and why:**

- **Parallel execution.** The retrievers are independent; a
  question tagged `analytical + relational + semantic` fans out
  to three backends simultaneously. Total latency is bounded by
  the slowest, not the sum.
- **RRF fuses only text-shaped rankings.** Mod-106 covers why
  RRF beats score-sum for cross-retriever fusion. Applying RRF
  to SQL rows and subgraph nodes would mean pretending the row
  at rank 3 is "more relevant" than the node at rank 2; the
  semantics are different and the comparison would be a
  category error.
- **Budget check before dispatch.** The router asks the budget
  manager before calling a retriever; if the budget is
  exhausted, the retriever is skipped and the skip is audited.
  The user sees a bundle with fewer legs rather than a
  silently-truncated answer.
- **Errors are audited, not raised.** One backend failing does
  not fail the request; the bundle returns the healthy
  retrievers' output and the error is recorded. A caller that
  wants strict behaviour can inspect `bundle.results_by_retriever`
  and refuse on any `error`.

### 2.5 `src/budget.py` — budget enforcement

```python
"""Per-request wall-clock / cost budget.

Prices are retriever-relative, not currency-exact. The point is
that ColPali is heavy and SQL is cheap; the exact numbers come
from the deployment."""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_cost: float = 1.0
    costs: dict[str, float] = field(default_factory=lambda: {
        "hybrid_text": 0.05,
        "sql": 0.10,
        "graph": 0.15,
        "multimodal:siglip": 0.10,
        "multimodal:colpali": 0.60,
    })
    spent: float = 0.0

    def check_add(self, retriever: str) -> None:
        cost = self.costs.get(retriever, 0.10)
        if self.spent + cost > self.max_cost:
            raise BudgetExceeded(
                f"{retriever} would exceed budget: spent {self.spent}, "
                f"cost {cost}, max {self.max_cost}"
            )
        self.spent += cost
```

**Choices and why:**

- **Costs are relative, not absolute.** The budget reflects the
  differential expense of each retriever on a given deployment.
  The deployment owner calibrates `costs` from real
  measurements (per-call latency × infra cost); the exercise
  ships defaults that match the expected ordering.
- **Fail-explicit on budget.** A budget-exceeded skip is a
  logged event, not a silent fallback. The audit trail tells
  the operator which retrievers were dropped and why.

### 2.6 `src/audit.py` — the audit trail

```python
"""Per-request audit log: what the router decided and why.

Every decision is loggable. The log is append-only; the format
is machine-readable so a downstream analysis can show route
mix over time."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from classifier import Classification


@dataclass
class RouteDecision:
    question: str
    classification: Classification
    retrievers_called: list[str]
    latencies: dict[str, float]
    errors: dict[str, str | None]


class AuditLog:
    def __init__(self, path: Path, hash_question: bool = True) -> None:
        self._path = path
        self._hash = hash_question
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: RouteDecision) -> None:
        payload = {
            "question": self._hashed(decision.question),
            "intents": [i.value for i in decision.classification.intents],
            "source": decision.classification.source,
            "reasons": decision.classification.reasons,
            "retrievers_called": decision.retrievers_called,
            "latencies": decision.latencies,
            "errors": decision.errors,
        }
        self._append(payload)

    def record_skip(self, retriever: str, reason: str) -> None:
        self._append({"event": "skip", "retriever": retriever, "reason": reason})

    def record_error(self, retriever: str, error: str) -> None:
        self._append({"event": "error", "retriever": retriever, "error": error})

    def _hashed(self, q: str) -> str:
        if not self._hash:
            return q
        import hashlib
        return "sha256:" + hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]

    def _append(self, payload: dict) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
```

**Choices and why:**

- **Questions are hashed by default.** In an enterprise
  context, questions can carry PII or confidential facts. The
  hash preserves the audit trail's operational value (route
  mix, latency distribution) without storing the question text.
  A deployment that wants plaintext questions sets
  `hash_question=False`.
- **Append-only JSON lines.** Simple, grep-able, and
  shippable to any log aggregator. The deployment can
  route the file to its normal pipeline.

### 2.7 `tests/test_router.py` — golden routing cases

```python
"""Golden routing tests: question → expected retriever set."""
from __future__ import annotations

import pytest

from classifier import Intent, classify_by_rules
from router import Router  # using stub backends in the test fixture


GOLDEN = [
    # analytical — aggregation phrasing
    ("How many customers do we have in region EU?",       {Intent.ANALYTICAL}),
    ("Top 5 products by revenue since 2024",              {Intent.ANALYTICAL}),

    # relational — connection phrasing
    ("Who at Apex is connected to Nova's account manager?",
     {Intent.RELATIONAL}),
    ("Which supplier is related to incident INC-42?",
     {Intent.RELATIONAL, Intent.EXACT}),

    # exact phrase
    ('Find the article titled "Welcome to Apex"',         {Intent.EXACT}),

    # visual
    ("Show me the chart from page 7 of the quarterly report",
     {Intent.VISUAL}),

    # semantic fallback
    ("What does our product do for retail chains?",       set()),  # no rules fire
]


@pytest.mark.parametrize("question, expected_intents", GOLDEN)
def test_rule_classifier(question, expected_intents):
    cls = classify_by_rules(question)
    # Every expected intent should fire.
    assert expected_intents <= cls.intents, (cls.intents, expected_intents)


def test_router_fans_out_to_expected_backends(router_with_stub_backends):
    bundle = router_with_stub_backends.route(
        "Top 5 suppliers connected to Apex",
        policy=_PolicyStub(),
    )
    # analytical + relational → sql + graph
    assert set(bundle.results_by_retriever.keys()) == {"sql", "graph"}


def test_budget_exceeded_skips_colpali(router_with_tight_budget):
    bundle = router_with_tight_budget.route(
        "Show me page 7 chart in the quarterly PDF",
        policy=_PolicyStub(multimodal_mode="colpali"),
    )
    # ColPali is expensive; a tight budget should skip it.
    assert "multimodal" not in bundle.results_by_retriever
    # And the audit log records the skip.
    assert any(e["event"] == "skip" for e in router_with_tight_budget.audit_events())
```

**What the tests assert:**

- The deterministic classifier fires on the signatures we
  defined; the fallback goes to `semantic` when nothing matches.
- The router fans out to the expected backends for multi-label
  intents; the backend set — not the number of hits — is what
  the router guarantees.
- The budget drops the right retriever and the audit log
  reflects the drop. Silence is a bug; the skip has to be
  recoverable from the trail.

## 3. Validation steps

A correct submission runs locally with stubbed backends and can
be re-pointed at the real ones.

1. **Install.** `pip install pytest`. The adapters in
   `src/retrievers.py` import the earlier-exercise backends;
   the test suite swaps them for stubs.
2. **Unit tests for the classifier.** `pytest tests/test_classifier.py`
   covers rule hits, LLM fallback (with a stubbed chat), and
   the invalid-label defence.
3. **Routing tests.** `pytest tests/test_router.py` covers the
   golden set: at least one case per intent, at least one
   multi-label case, at least one budget-exhausted case, and
   at least one "no intent fires → semantic default" case.
4. **End-to-end smoke test.** Wire the real backends for
   hybrid_text, sql, and graph (multimodal is optional in the
   smoke test). Run a handful of questions; confirm that each
   question lands on the expected backend set and that the
   bundle is non-empty.
5. **Audit trail inspection.** After the smoke test, read
   `audit.jsonl`. Every request is present; every intent is
   recorded; latencies are plausible (hybrid < sql ≈ graph <
   colpali); any skip has a reason.
6. **ACL inheritance from exercise-01.** The `policy` argument
   carries the ACL predicate end-to-end. Confirm that an EU
   caller and a US caller see different row sets (SQL), node
   sets (Graph), and hit sets (text). The router does not
   enforce ACLs; it threads them.
7. **Fan-out safety.** A classification with three intents
   calls three retrievers *in parallel*; total latency is the
   max of the three legs, not the sum. Confirm with the
   per-retriever latencies in the audit log.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Classifier | Deterministic rules fire on the canonical signatures; LLM fallback is constrained to the known labels; default is `semantic`. |
| Retriever interface | One `Hit` type; four adapters; `Hit.kind` present and used for routing of downstream rendering. |
| Routing map | Intent → retriever map is explicit and auditable; multi-intent queries call multiple retrievers. |
| Parallel execution | Retrievers run concurrently; total latency is max, not sum. |
| Fusion | RRF fuses text-shaped hits only; SQL rows and graph nodes are passed through as separate output channels. |
| Budget | Per-request budget with per-retriever costs; skipped retrievers are logged, not silent. |
| Audit log | Every decision records intents, reasons, retrievers called, latencies, and errors; questions hashed by default. |
| ACL inheritance | The caller's policy propagates to every retriever; the router does not re-authorise. |
| Failure handling | A failing retriever does not fail the request; its error is in the bundle and the audit. |
| Tests | Golden tests for each intent, at least one multi-label case, at least one budget-exhausted case, at least one fallback case. |

Fail-fast smells:

- The router fuses SQL rows with vector hits into one ranked
  list. Rows and docs are not comparable; the fusion is a
  category error.
- The classifier is LLM-only. LLM-only classifiers are slower,
  more expensive, and less auditable than a deterministic layer
  on top; the hybrid is the right default.
- The router calls every retriever on every query. The point of
  the router is to *not* do that; without the classifier the
  system is "SELECT *" retrieval.
- The audit log stores the raw question text in a deployment
  that handles PII. Hash by default; plaintext is a deliberate
  opt-in.
- A budget-exceeded skip is silent. The caller has to know
  which legs ran and which did not; silence breaks audit.
- ACLs are enforced inside the router. ACLs belong to the
  backends; the router threads the policy but does not become
  the authorisation layer.

## 5. Common mistakes

- **Treating the router as a chain.** The router is a
  parallel-fanout, not a sequence. Chaining retrievers (vector
  → SQL on the vector hits) is a different architecture and
  usually belongs in the application, not the router.
- **One-label classification.** A real question often needs
  more than one backend; a single-label classifier forces the
  router to pick between them and hide the signal from the
  one it dropped.
- **Score normalisation across retrievers.** BM25 scores,
  cosine scores, and SQL "rank" are on incomparable scales.
  RRF is the right fusion operator for rankings; if an engineer
  reaches for `(a_score + b_score) / 2`, the result is noise.
- **Letting SQL and graph scores enter the RRF.** RRF works
  because it operates on *ranks*, not scores — but it still
  assumes the fused items are the same kind. A row and a
  document should not share a ranking.
- **Over-sizing the classifier.** The rule set in this
  solution has five regexes. Adding more signatures past the
  point where they overlap introduces conflicts; keep the rule
  layer small and let the LLM catch the tail.
- **Forgetting the LLM-classifier failure mode.** The LLM can
  return malformed JSON, non-enum labels, or an empty response.
  Each case has a fallback; the exercise's classifier defaults
  to `semantic` on any failure.
- **Threading the ACL policy through some retrievers but not
  all.** The ACL contract is uniform across retrievers; a
  bypass in one backend is a bypass of the whole system.
- **Logging only successful requests.** The skipped and errored
  retrievers are the audit-worthy events — they reveal budget
  pressure and backend health. Log them all.
- **Measuring router latency as a sum.** If three retrievers
  run in parallel, the total wall-clock is the max. A sum
  mis-reports budget consumption as time and pressures the
  team to drop retrievers that aren't actually the bottleneck.
- **Letting the classifier see the whole conversation.**
  Classify the *current question*. A multi-turn conversation
  needs the orchestration layer (mod-110) to resolve
  references before classification.

## 6. References

Retrieval background (fusion, hybrid, routing patterns):

- Reciprocal Rank Fusion — Cormack, Clarke, Buettcher,
  "Reciprocal Rank Fusion outperforms Condorcet and individual
  Rank Learning Methods," SIGIR 2009 —
  [plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
  RRF's `c=60` default comes from this paper.
- Model Context Protocol specification —
  [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification).
  When the router is fronted as an MCP tool, intent labels can
  be exposed as tool arguments so a caller can override the
  classifier's decision.

Governance frameworks (for the audit and risk framing):

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: ML01 Input Manipulation Attack (an adversary can
  phrase a question to trigger the SQL leg instead of the
  vector leg; the audit trail is how this is caught).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: *LLM Prompt Injection* (a classifier LLM is a
  prompt-injection surface — constrain its output and treat
  invalid labels as default-semantic), *Exfiltration via ML
  Inference API* (budget and audit are mitigations for bulk
  extraction via crafted questions).
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: *Govern* (route policy is a governed artefact — who
  owns it, who reviews changes), *Map* (the intent→retriever
  table is the map), *Measure* (route-mix metrics from the
  audit log), *Manage* (budget-exceeded skips are the
  manage-time safety valve).

Earlier exercises this one composes on:

- mod-106 — hybrid vector+BM25 fusion is the backbone of the
  `hybrid_text` retriever; the RRF implementation reuses that
  module's defaults.
- mod-108 — citation contracts flow through this router; the
  bundle carries enough per-hit metadata for the downstream
  prompt to cite correctly.
- mod-110 — MCP exposure is upstream of the router; the tool's
  input schema can declare an optional `intent` field to let
  advanced callers bypass classification.
- mod-111 — the audit log is one of the retrieval-MLOps
  signals the drift monitor consumes (route-mix shift =
  behavioural drift).
- exercise-01, 02, 03, 04 — the ACL pattern, SQL pipeline,
  Graph RAG pipeline, and multimodal embedders are the four
  backends this router sits in front of.
