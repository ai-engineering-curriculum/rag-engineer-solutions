# Exercise 05 — Solution: Filter-Heavy Hybrid Composition Drill

This is a reference solution for
`exercise-05-filter-heavy-hybrid-composition-drill.md`. It is a
**design and implementation** exercise. The learner composes a hybrid retriever
(BM25 + dense + RRF, from the prior exercises) with *hard filters* —
tenant scopes, ACLs, time windows, structured facets — and defends the
ordering of filter vs. rank-vs-fuse decisions against three failure
scenarios.

The teaching point is that **a filter is not a boost**. Filters are
hard preconditions that must apply before, during, or after the
retriever in ways that keep recall honest and ACLs correct. Learners
who treat filters as "another score contribution" ship systems that
leak ACL-restricted documents into recall or silently drop all
relevant documents because the dense query returned 10 candidates and
the filter accepted none of them. The exercise is about picking the
right composition for the right filter type and documenting why.

## 1. Solution overview

The deliverable has six parts:

1. **`design/composition.md`** — a short decision tree that classifies
   every filter the system supports into one of four categories: hard
   precondition (ACL, tenant), temporal window, structured facet, and
   boost (soft preference). Each category has a composition rule.
2. **`src/retriever.py`** — a reference implementation of the
   filter-aware hybrid retriever: BM25 + dense with per-leg filter
   application, post-fuse ACL re-check, and configurable top-K
   backpressure.
3. **`queries/`** — three representative query shapes in each engine's
   DSL (OpenSearch is the reference, Elasticsearch and Vespa variants
   included): tenant-scoped RRF, time-windowed hybrid, facet-filtered
   hybrid.
4. **`eval/scenarios.md`** — three failure scenarios with labeled
   expected behavior: (A) tenant leak, (B) empty-result after filter,
   (C) over-filtered dense recall.
5. **`eval/metrics.md`** — per-scenario pass/fail results plus
   retrieval metrics on a 25-query filter-heavy eval set.
6. **`notes/acl.md`** — a brief threat-model note on how filters
   interact with retrieval and why post-fuse re-checks are
   non-negotiable for ACL enforcement.

## 2. Worked answer or implementation

### 2.1 The four filter categories

| Category | Example | Composition rule |
|---|---|---|
| **Hard precondition** | ACL (`user_can_read == true`), tenant (`tenant_id == "acme"`), deletion state (`is_deleted == false`). | Apply **pre-retrieval** on both legs (BM25 and dense), **and** re-check post-fuse. Never express as a boost. |
| **Temporal window** | "Only docs from the last 90 days." | Apply pre-retrieval as a range filter on both legs. Prefer `filter` context (score-neutral) over `must` (score-affecting). |
| **Structured facet** | `language == "en"`, `product == "kafka"`. | Pre-retrieval filter on both legs. If the facet is *optional* ("prefer English but accept others"), it is a boost, not a filter — move to the boost category. |
| **Boost** | "Prefer docs with `priority == "high"`," "recency decay." | Apply as a score modifier on each leg or as a reranker feature. **Never gate recall on it.** |

A filter is a hard precondition *iff removing matching documents from
the corpus would still be correct behavior* for this request. If a
document must not appear in the result, it is a hard precondition. If
a document would be nice to rank lower but still visible, it is a
boost. Mixing those up is the single most common filter bug.

### 2.2 `design/composition.md` — the decision tree

Given a filter specification, walk this tree:

```
1. Does the filter affect correctness / safety / tenancy?
   YES → Hard precondition. Apply pre-retrieval on BOTH legs.
         Re-check the document set post-fuse. Go to 4.
   NO  → Continue.
2. Is the filter a time or numeric range?
   YES → Temporal / range. Apply pre-retrieval on both legs.
         Use filter context (score-neutral). Go to 4.
   NO  → Continue.
3. Is the filter a facet (exact/keyword match) that MUST hold?
   YES → Structured facet. Pre-retrieval filter on both legs. Go to 4.
   NO  → Boost. Express as score modifier or rerank feature.
         Recall is NOT gated. Stop.
4. Compose with the hybrid retriever:
   - BM25 leg: `query.bool.filter` (score-neutral).
   - Dense leg: `knn.filter` (if engine supports pre-filtering) OR
     oversample-then-post-filter with top_k_search >> top_k_final.
   - Fuse the filtered lists with RRF (or weighted-sum per exercise-02).
   - Post-fuse: iterate the fused top-K and re-check hard preconditions;
     drop violators before returning.
```

Two notes worth their own paragraph:

**Pre-filter vs. post-filter for the dense leg.** HNSW + filter is not a
clean operation. If a filter is highly selective (e.g., "docs for user
acme" when acme owns 0.1% of the corpus), a strict post-filter against
the dense top-K may return an empty set — ANN found 100 candidates, the
filter accepted zero. Engines handle this in different ways:

- **Elasticsearch / OpenSearch (Lucene HNSW).** `knn.filter` pre-filters
  during graph traversal; if the selectivity is extreme, the engine
  may fall back to exact kNN. Still safe on recall when the fallback
  triggers; opaque on performance.
- **OpenSearch k-NN plugin (nmslib/Faiss backend).** Supports
  `efficient_filter` or post-filter depending on backend. Post-filter
  with a highly selective predicate is the classic "empty dense
  results" trap.
- **Vespa.** Native filter integrated into the ANN traversal; honest
  per-query behavior.
- **Fallback recipe when pre-filter is not available.** Oversample
  (`top_k = 500`, say) and post-filter. Cost: 5–10× the ANN work. Only
  acceptable for small corpora or low QPS.

**Post-fuse re-check for ACLs.** Even if both retrieval legs filter on
ACL pre-retrieval, a stale cache in one leg or a race between an index
update and an ACL change can leak. The post-fuse re-check is a cheap
belt-and-braces guarantee: walk the fused top-K, call the ACL service
(or re-check the field), drop violators. This also catches engine bugs
that would otherwise expose documents to unauthorized users. For ACL,
pay the re-check cost; it is the only honest answer.

### 2.3 `src/retriever.py`

```python
"""Filter-aware hybrid retriever. Applies hard preconditions, temporal
windows, and facets per the composition rules in design/composition.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from fusion import rrf  # from exercise-02


@dataclass
class HardFilter:
    """A hard precondition. Expressed as a Lucene-style filter clause."""
    clauses: list[dict]  # e.g., [{"term": {"tenant_id": "acme"}}]


@dataclass
class RangeFilter:
    """A temporal/numeric range filter."""
    field: str
    gte: str | int | None = None
    lte: str | int | None = None


@dataclass
class FacetFilter:
    """An exact-match facet filter."""
    field: str
    values: list[str]


@dataclass
class FilterSet:
    hard: list[HardFilter] = field(default_factory=list)
    ranges: list[RangeFilter] = field(default_factory=list)
    facets: list[FacetFilter] = field(default_factory=list)


@dataclass
class FilterAwareRetriever:
    client: OpenSearch
    index: str
    embedder: SentenceTransformer
    acl_recheck: Callable[[str, list[str]], list[str]]
    top_k: int = 100
    oversample_dense: int = 500  # only used when post-filtering

    def _filter_clauses(self, fs: FilterSet) -> list[dict]:
        clauses: list[dict] = []
        for h in fs.hard:
            clauses.extend(h.clauses)
        for r in fs.ranges:
            rng = {k: v for k, v in (("gte", r.gte), ("lte", r.lte)) if v is not None}
            clauses.append({"range": {r.field: rng}})
        for f in fs.facets:
            clauses.append({"terms": {f.field: f.values}})
        return clauses

    def bm25(self, query: str, fs: FilterSet) -> list[str]:
        filter_clauses = self._filter_clauses(fs)
        body = {
            "size": self.top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"body": query}}],
                    "filter": filter_clauses,  # score-neutral
                }
            },
        }
        resp = self.client.search(index=self.index, body=body)
        return [h["_id"] for h in resp["hits"]["hits"]]

    def dense(self, query: str, fs: FilterSet) -> list[str]:
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        filter_clauses = self._filter_clauses(fs)
        body = {
            "size": self.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vec,
                        "k": self.top_k,
                        "filter": {"bool": {"filter": filter_clauses}},
                    }
                }
            },
        }
        resp = self.client.search(index=self.index, body=body)
        return [h["_id"] for h in resp["hits"]["hits"]]

    def search(
        self,
        query: str,
        fs: FilterSet,
        *,
        user_id: str,
    ) -> list[tuple[str, float]]:
        lex = self.bm25(query, fs)
        dns = self.dense(query, fs)
        fused = rrf([lex, dns], k=60)
        candidate_ids = [doc for doc, _ in fused]
        # Post-fuse ACL re-check. Belt and braces.
        allowed = set(self.acl_recheck(user_id, candidate_ids))
        return [(doc, score) for doc, score in fused if doc in allowed]
```

Key properties of this implementation:

- **Filters apply pre-retrieval on both legs.** BM25's `bool.filter`
  and `knn.filter` run before scoring on each side. A document that
  fails the filter never appears in a per-leg top-K, so RRF never
  sees it.
- **ACL re-check post-fuse.** Even with pre-filter on both legs, the
  final pass through `acl_recheck` ensures no leaked document is
  returned to the user. For a strict system, call the authoritative
  ACL service here (not the index copy).
- **Range filters go in `bool.filter`, not `bool.must`.** `must` is
  score-affecting; range filters are preconditions and must be
  score-neutral so they don't poison the fused ranking.
- **Oversampling is not shown but documented.** `oversample_dense`
  exists for engines that post-filter; when active, the dense leg
  searches top-500 and the client-side post-filter keeps only matches
  before passing to RRF.

### 2.4 `queries/` — reference DSLs

**Tenant-scoped hybrid (OpenSearch).** ACL + facet enforced
pre-retrieval on both legs; separate `_search` requests are fused
client-side here, but engines that support `rank: rrf` can push the
fuse down.

```json
POST /mod106_chunks/_search
{
  "size": 100,
  "query": {
    "bool": {
      "must":   [{"match": {"body": "consumer group rebalance stuck"}}],
      "filter": [
        {"term":  {"tenant_id": "acme"}},
        {"term":  {"is_deleted": false}},
        {"term":  {"language": "en"}}
      ]
    }
  }
}
```

```json
POST /mod106_chunks/_search
{
  "size": 100,
  "query": {
    "knn": {
      "embedding": {
        "vector": [ ... ],
        "k": 100,
        "filter": {
          "bool": {
            "filter": [
              {"term": {"tenant_id": "acme"}},
              {"term": {"is_deleted": false}},
              {"term": {"language": "en"}}
            ]
          }
        }
      }
    }
  }
}
```

**Time-windowed hybrid.** Same shape; the filter clause is a `range`.

```json
"filter": [
  {"range": {"created_at": {"gte": "now-90d/d", "lte": "now/d"}}}
]
```

**Boost (NOT a filter).** For "prefer recent" without hard-cutting:

```json
{
  "function_score": {
    "query": {"match": {"body": "..."}},
    "functions": [{
      "gauss": {"created_at": {"origin": "now", "scale": "30d", "decay": 0.5}}
    }],
    "score_mode": "multiply",
    "boost_mode": "multiply"
  }
}
```

Distinguishing marker: the `function_score` surface is for boosts.
Filters never go through it.

### 2.5 `eval/scenarios.md` — failure scenarios the implementation must handle

**Scenario A — Tenant leak.**

- Setup: User from tenant `beta` runs a query. One document in the
  index has `tenant_id = acme` but is relevant to the query terms.
  An engine bug (hypothetical) causes BM25 to ignore the filter on
  one request.
- Expected behavior: the post-fuse ACL re-check must drop the acme
  document before returning to the user.
- Pass criterion: in automated tests, deliberately inject an
  unfiltered BM25 response and confirm the re-check catches it.
- Observability: log a counter whenever the post-fuse check drops a
  document. A non-zero counter in production is a sev-1 investigation.

**Scenario B — Empty result after highly selective filter.**

- Setup: Tenant `startup-xyz` owns 50 docs out of a 5M-doc corpus
  (1 in 100,000). A dense search returns 100 candidates; none match
  the tenant.
- Expected behavior: the retriever must still return relevant docs
  for `startup-xyz`. Options: (a) pre-filter in the engine (the
  right option when the engine supports it); (b) oversample dense
  to 5,000 and post-filter; (c) if both legs return zero candidates,
  explicitly return an empty result rather than falling back silently
  to a larger candidate pool.
- Pass criterion: for a small-tenant query with a known answer, the
  retriever returns the answer in top-10, not an empty list.
- Fail smell: an "oversample to infinity" fix that works but
  silently 10× the ANN cost on *every* query. The oversample factor
  should scale with the tenant's selectivity, not be set to a
  worst-case constant.

**Scenario C — Over-filtered dense recall (post-filter trap).**

- Setup: A system uses post-filter on the dense leg and a strict
  facet (`language == "en"` where 30% of docs are `en`). The dense
  top-100 is sampled randomly from the whole corpus; 70% fail the
  facet; only 30 docs survive.
- Expected behavior: the dense leg should return 100 *English* docs,
  not 30. Fix: switch to pre-filter or oversample to 333.
- Pass criterion: per-leg recall measured against the English-only
  gold set should be stable regardless of facet selectivity.

### 2.6 `eval/metrics.md`

Build a 25-query filter-heavy eval set that covers:

- 10 tenant-scoped queries (each with ground truth docs inside the
  tenant).
- 10 time-windowed queries (each with ground truth docs within the
  window).
- 5 facet-filtered queries (each with ground truth under a specific
  `product` facet).

Representative (teaching-run) results:

| Configuration | NDCG@10 (filtered) | Recall@10 (filtered) | Tenant leak rate |
|---|---|---|---|
| Baseline (filters as boosts) | 0.44 | 0.51 | **0.08** (leaked on 2/25 queries) |
| Filters in `must` (score-affecting) | 0.53 | 0.62 | 0.00 |
| Filters in `bool.filter` + dense pre-filter | **0.61** | **0.74** | 0.00 |
| Filters in `bool.filter` + dense post-filter | 0.52 | 0.60 | 0.00 |
| Full composition (pre-filter + post-fuse ACL re-check) | **0.61** | **0.74** | **0.00** |

The headline: a mis-composed filter not only loses NDCG, it can leak
data. The full composition is the right answer; "post-filter" is a
retrieval-quality downgrade; "filter as boost" is both a retrieval and
a security downgrade.

### 2.7 `notes/acl.md` — why post-fuse re-check is non-negotiable

Pre-retrieval filtering is necessary but not sufficient for ACL. A
partial list of ways pre-filter can fail:

- **Stale ACL field in the index.** If permissions changed between
  the last indexing run and now, the index's `allowed_users` field
  may still include the former owner. Pre-filter against the index
  passes; post-fuse re-check against the authoritative ACL service
  catches it.
- **Engine bug / partial filter application.** Known-rare but
  possible. If the engine silently fails to apply a filter on one
  shard, documents leak.
- **Operator mistake in query construction.** A missing `filter`
  array because of templating is easy to do and hard to spot in code
  review. A post-fuse re-check is a defense in depth.
- **ACLs that depend on query-time context.** Row-level security
  predicates sometimes need the current session's role, group, or
  time-bounded token. These are hard to pre-filter cleanly; the
  post-fuse re-check is the natural place to enforce them.

For non-ACL filters (facets, time windows), post-fuse re-checks are
less load-bearing. A time-window range filter is unlikely to be
bypassed by a stale cache. Reserve the re-check cost for filters that
guard confidentiality or correctness.

## 3. Validation steps

1. **Filter-as-filter, not boost.** Static-analyze the queries emitted
   by the retriever. Every ACL/tenant/deletion clause must appear in
   `bool.filter` or `knn.filter`, never in `bool.must` or
   `function_score`. A linter can enforce this.
2. **Pre-retrieval filter on both legs.** For each filter in the
   catalog, confirm it appears in the BM25 request body *and* the
   knn request body. A missing clause on either side is a correctness
   bug.
3. **Post-fuse ACL re-check runs on every query.** Instrument with a
   counter. In production, drop count should be ~0; a non-zero value
   is an incident.
4. **Empty-result sanity.** For a known small-tenant query, confirm
   the retriever returns a non-empty top-10. If empty, inspect the
   per-leg candidate pools and fix the selectivity handling
   (pre-filter preferred; oversample-and-post-filter as fallback).
5. **Scenario tests are automated.** The three scenarios in
   `eval/scenarios.md` must have unit/integration tests that fail
   when a developer "simplifies" the composition (e.g., moves ACL out
   of `knn.filter` into a client-side post-filter only).
6. **Metrics are per-tenant, not only global.** A global NDCG@10 of
   0.60 can hide a per-tenant NDCG@10 of 0.10 for small tenants.
   Break out small-tenant / large-tenant retrieval quality
   separately.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Classification | Every filter in the project is classified into one of the four categories, with written justification. |
| Pre-retrieval filter | ACL, tenant, deletion, time, facet filters applied on **both** retrieval legs pre-retrieval. |
| Score-neutrality | Filters are in `filter` context, not `must` or `function_score`. |
| Post-fuse ACL re-check | Present and instrumented with a drop counter. |
| Small-tenant handling | Pre-filter preferred; documented fallback to oversample-and-post-filter when engine forces it. |
| Boost vs. filter | Soft preferences (recency, priority) are implemented as boosts; recall is not gated on them. |
| Scenario tests | A, B, C automated and in CI. |
| Per-tenant metrics | Reported alongside global metrics. |
| Threat model | `notes/acl.md` enumerates at least three pre-filter failure modes and justifies the re-check cost. |

Fail-fast smells:

- An ACL filter implemented as `function_score` boost. Documents will
  rank lower when ACL fails, but still appear. Hard recall break for
  confidentiality.
- Dense retrieval that takes the raw `knn` top-100 and *then*
  client-side filters. Works on dense, silently degrades on
  high-selectivity tenants.
- `bool.must` on a tenant filter. Changes the score, does not change
  the result set — but now the fused ranking depends on tenant
  cardinality, which is nonsense as a ranking signal.
- No post-fuse re-check, with the justification that "the engine
  filter is reliable." Fine for facets; not fine for ACLs. The cost
  is a few microseconds per candidate.
- Scenario tests that assert "retrieval returns at least one result."
  A tenant with zero relevant documents should return zero, not a
  fallback. An "at least one" assertion hides the empty-tenant bug.

## 5. Common mistakes

- **Filter-as-boost on hard preconditions.** The classic bug: ACL
  implemented as a `function_score` multiplier. Documents with the
  wrong ACL still appear in results, just lower. Hard preconditions
  must gate the result set.
- **Post-filter on the dense leg without oversample.** When the
  filter is selective, the top-100 from dense survives a post-filter
  as a top-5 or top-0. Either pre-filter or oversample until the
  survivor count is meaningful.
- **Score-affecting range filters.** `created_at` in `bool.must`
  makes recency rank, not filter. Use `bool.filter` and apply a
  separate `gauss` boost if you want recency to also move the
  ranking.
- **Different filters on BM25 vs. dense legs.** Easy slip in
  hand-maintained query templates. Any filter that is a precondition
  must be on both legs. A test that walks both bodies and compares
  the filter arrays catches this.
- **Coalescing ACL + facet into one `terms` clause.** An ACL and a
  product facet have different semantics (one is confidentiality,
  one is user preference) and different failure consequences. Keep
  them as separate clauses so they can be audited independently.
- **Logging only the filter spec, not the applied clauses.** The
  spec can be correct; the applied query body can be stale (cached,
  templated from a different session). Log the serialized query
  body alongside the spec.
- **Measuring retrieval quality only globally.** Retrieval over a
  filter-heavy corpus has very different quality for small tenants
  vs. large. A single NDCG number can be dominated by the large
  tenants and mask severe small-tenant problems.
- **Ignoring threat-model cross-references.** ACL leaks and
  tenant-boundary crossings map directly to several NIST AI RMF
  governance controls, OWASP ML-top-10 categories (privacy &
  access control), and MITRE ATLAS tactics (information disclosure
  via inference). Named linkage makes security review fast; absent
  linkage makes it slow.

## 6. References

- **Elasticsearch — filter context and `knn.filter`.**
  - Query and filter context —
    [elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html).
  - `knn` filter —
    [elastic.co/guide/en/elasticsearch/reference/current/knn-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
- **OpenSearch — k-NN with filters.**
  - k-NN filters —
    [opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/).
  - Hybrid search —
    [opensearch.org/docs/latest/search-plugins/hybrid-search/](https://opensearch.org/docs/latest/search-plugins/hybrid-search/).
- **Vespa — filter and ANN composition.**
  - Approximate nearest neighbor (HNSW) + filter —
    [docs.vespa.ai/en/approximate-nn-hnsw.html](https://docs.vespa.ai/en/approximate-nn-hnsw.html).
  - Attribute filters —
    [docs.vespa.ai/en/attributes.html](https://docs.vespa.ai/en/attributes.html).
- **RRF reference (used for fusion).**
  Cormack, Clarke, Büttcher. SIGIR 2009.
  DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- Risk / governance framings consulted while drafting:
  - OWASP Machine Learning Security Top 10 — relevant categories:
    ML05 (Model Theft), ML07 (Transfer Learning Attacks) are
    secondary; the primary concern here is access control and tenant
    isolation, which maps to NIST SP 800-53 access-control controls
    rather than an OWASP ML category. Still consulted:
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — information disclosure and inference techniques —
    [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — access-control and data-
    governance functions —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
