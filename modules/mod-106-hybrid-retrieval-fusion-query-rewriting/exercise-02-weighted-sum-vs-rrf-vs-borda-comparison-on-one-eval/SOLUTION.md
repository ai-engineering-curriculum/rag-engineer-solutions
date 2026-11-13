# Exercise 02 — Solution: Weighted-Sum vs. RRF vs. Borda-Count Fusion Bake-Off

This is a reference solution for
`exercise-02-weighted-sum-vs-rrf-vs-borda-comparison-on-one-eval.md`. It is a
**comparative implementation** exercise. The learner implements three
fusion methods — normalized weighted-sum, RRF (from exercise-01), and
Borda count — and runs them against the same eval set with the same
first-phase retrievers. The deliverable is a side-by-side table and a
written recommendation.

The teaching point is that **no fusion method dominates all three
baselines**. Each formula encodes a different assumption about how the
retrievers are calibrated. The exercise pushes the learner past "we use
RRF because it's the default" into "we picked X because A, B, and C are
true of our retrievers; if any of them stops being true, re-run this
comparison."

## 1. Solution overview

The deliverable has five parts:

1. **`src/fusion.py`** — reference implementations of the three methods:
   - `weighted_sum(lists_with_scores, weights, normalize="minmax")`
   - `rrf(lists, k=60, weights=None)` (imported from exercise-01)
   - `borda(lists, weights=None)`
2. **`src/run_bakeoff.py`** — produces a ranked list per method per query
   from the same BM25 + dense first-phase runs. All three methods consume
   the same retrieved candidate pool for fairness.
3. **`eval/qrels.tsv` + `eval/queries.tsv`** — reuses the 10-query set
   from exercise-01 and adds 10 queries that stress disagreement between
   the retrievers (lexical-only cues, concept-only cues, and acronym
   expansion).
4. **`eval/bakeoff.md`** — the side-by-side table: NDCG@10, Recall@10,
   MRR, and per-query delta vs. the best single retriever for each
   method. Includes a short "how each method behaved on query Q" case
   study.
5. **`notes/recommendation.md`** — the decision: which method to ship,
   under which assumptions, and the measurable signal that would trigger
   switching.

The exercise is scoped to a *fair* comparison. The easiest way to publish
a misleading bake-off is to let one method see a different candidate pool
than the others. The reference pipeline funnels all three methods through
the same top-100 BM25 + top-100 dense sets.

## 2. Worked answer or implementation

### 2.1 Three methods in one paragraph each

**Weighted sum (a.k.a. linear combination or convex combination).** For
each candidate `d`, compute
`score(d) = α · normalize(bm25(d)) + (1 − α) · normalize(cosine(d))`.
It uses the raw scores, requires both to be on a comparable scale (hence
normalization), and exposes one hyperparameter (`α`) per retriever pair.
This is what `hybrid_search` in LlamaIndex and many bespoke retrievers do
by default.

**Reciprocal rank fusion (RRF).** For each candidate `d`,
`score(d) = Σ_i 1 / (k + rank_i(d))` with `k = 60`. See exercise-01. Rank-
only; invariant to score scale; zero hyperparameters other than `k`.

**Borda count.** For each candidate `d`,
`score(d) = Σ_i (N_i − rank_i(d) + 1)` where `N_i` is the length of
system `i`'s list. A rank-1 document in a list of 100 scores 100; rank
100 scores 1; absent documents score 0. Borda is an older social-choice
method (Jean-Charles de Borda, 1770) and in IR appears in Aslam & Montague
(SIGIR 2001). It is linear in rank rather than reciprocal — this matters,
as the top of each list carries proportionally less weight than in RRF.

### 2.2 `src/fusion.py` — all three methods

```python
"""Three fusion methods on the same interface.

Each method consumes per-system ranked lists and returns a single
(doc_id, fused_score) list, sorted descending.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass
class ScoredHit:
    doc_id: str
    score: float


def _minmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _zscore(xs: Sequence[float]) -> list[float]:
    import statistics
    if len(xs) < 2:
        return [0.0 for _ in xs]
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1.0
    return [(x - mu) / sd for x in xs]


NormKind = Literal["minmax", "zscore", "none"]


def weighted_sum(
    scored_lists: Sequence[Sequence[ScoredHit]],
    *,
    weights: Sequence[float],
    normalize: NormKind = "minmax",
) -> list[tuple[str, float]]:
    """Linear combination of normalized per-system scores."""
    if len(weights) != len(scored_lists):
        raise ValueError("len(weights) must equal len(scored_lists)")
    acc: dict[str, float] = defaultdict(float)
    for w, hits in zip(weights, scored_lists):
        raw = [h.score for h in hits]
        if normalize == "minmax":
            norm = _minmax(raw)
        elif normalize == "zscore":
            norm = _zscore(raw)
        else:
            norm = list(raw)
        for h, s in zip(hits, norm):
            acc[h.doc_id] += w * s
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def rrf(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. See exercise-01 for the full write-up."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    acc: dict[str, float] = defaultdict(float)
    for w, ranked in zip(weights, ranked_lists):
        for position, doc_id in enumerate(ranked, start=1):
            acc[doc_id] += w / (k + position)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def borda(
    ranked_lists: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Borda count. Rank-only, linear in rank."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    acc: dict[str, float] = defaultdict(float)
    for w, ranked in zip(weights, ranked_lists):
        n = len(ranked)
        for position, doc_id in enumerate(ranked, start=1):
            acc[doc_id] += w * (n - position + 1)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
```

Implementation notes:

- **Normalization choice matters for weighted-sum.** Min-max is simple
  and stable; z-score is sensitive to the shape of each system's score
  distribution. BM25 scores are long-tailed (a few huge scores, many
  moderate ones); cosine similarities from a normalized embedder are
  roughly bounded in [0, 1] already. Pick one and keep it.
- **Borda requires equal-length lists to be "fair".** If system A returns
  100 docs and system B returns 50, Borda implicitly treats absence in B
  as "worse than the worst retrieved." In practice we pad both systems
  to the same top-K and treat missing docs as score 0. The formula above
  does exactly that: `n - position + 1` is computed only for retrieved
  items, so absence contributes 0 — the "equal-length" convention.
- **Weights are supported on all three methods.** Exercise-02's point is
  to *compare* on equal footing, so set weights to `[1.0, 1.0]` for the
  base run.

### 2.3 Normalization pitfalls — worth a paragraph

Weighted-sum fusion is the one method where the preprocessing choice can
dominate the hyperparameter. Three traps to avoid:

1. **Normalizing across the union of hits.** BM25's top-100 and dense's
   top-100 overlap only partially. If you concatenate and then min-max
   normalize, the single document that happened to top one list drags
   the floor for everything in the other. Normalize per-system (within
   each list's own scores).
2. **Not handling missing-document values.** A document present in BM25
   only has no cosine score. Setting it to `0` after min-max
   normalization means "worst cosine seen" (which is fine) rather than
   "unknown." If you set it to `mean(cosines)`, you double-count it.
   Convention: missing → 0 after per-system normalization.
3. **Z-score on long-tailed BM25.** The z-score of a long-tailed
   distribution puts the mode somewhere below zero. If you clip at zero
   after, you lose half the signal. Use min-max for BM25 or arctan-
   transform first.

### 2.4 `src/run_bakeoff.py` — the driver

```python
"""Run BM25 + dense once per query, feed the same output to all three
fusion methods, score each against the qrels."""
import json
import sys
from pathlib import Path

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from fusion import ScoredHit, borda, rrf, weighted_sum

TOP_K = 100
CLIENT = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
INDEX = "mod106_chunks"
EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")


def bm25(query: str) -> list[ScoredHit]:
    resp = CLIENT.search(
        index=INDEX,
        body={"size": TOP_K, "query": {"match": {"body": query}}},
    )
    return [ScoredHit(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]


def dense(query: str) -> list[ScoredHit]:
    v = EMBEDDER.encode(query, normalize_embeddings=True).tolist()
    resp = CLIENT.search(
        index=INDEX,
        body={
            "size": TOP_K,
            "query": {"knn": {"embedding": {"vector": v, "k": TOP_K}}},
        },
    )
    return [ScoredHit(h["_id"], h["_score"]) for h in resp["hits"]["hits"]]


def main(queries_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"weighted_sum": [], "rrf": [], "borda": []}
    for line in queries_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        qid, query = line.split("\t", 1)
        lex_scored = bm25(query)
        dns_scored = dense(query)
        lex_ids = [h.doc_id for h in lex_scored]
        dns_ids = [h.doc_id for h in dns_scored]

        ws = weighted_sum([lex_scored, dns_scored], weights=[0.5, 0.5])
        rr = rrf([lex_ids, dns_ids], k=60)
        bo = borda([lex_ids, dns_ids])

        for name, fused in (("weighted_sum", ws), ("rrf", rr), ("borda", bo)):
            for rank, (doc_id, score) in enumerate(fused[:100], start=1):
                out[name].append(
                    f"{qid}\t{doc_id}\t{rank}\t{score:.6f}"
                )
    for name, rows in out.items():
        (out_dir / f"{name}.run").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
```

Key fairness properties of this driver:

- One BM25 call + one dense call per query. All three fusion methods see
  the exact same candidates. If you run BM25 three times (once per
  method) you have introduced query-side variance that is not about
  fusion.
- All three methods fuse at the same recall depth (`TOP_K = 100`).
- All three methods share tie-breaking (`sort` is stable, insertion order
  wins). A naive "which method is best" study where ties are broken
  differently per method will fluctuate by multiple points of NDCG at
  low K.

### 2.5 `eval/bakeoff.md` — the side-by-side table

Representative (teaching-run, not authoritative) numbers over 20 queries:

| System | NDCG@10 | Recall@10 | MRR | Wins vs. best single |
|---|---|---|---|---|
| BM25 only | 0.47 | 0.52 | 0.56 | — |
| Dense only | 0.50 | 0.59 | 0.60 | — |
| **Weighted sum (0.5/0.5, min-max)** | 0.57 | 0.67 | 0.65 | 12/20 |
| **RRF (k=60)** | **0.60** | **0.71** | **0.68** | **15/20** |
| **Borda count** | 0.58 | 0.70 | 0.65 | 13/20 |

> **Reproducibility note.** Exact numbers vary with corpus, embedder,
> analyzer, and query set. The table illustrates the *shape* of the
> expected result on a small teaching eval. The ordering of RRF ≈ Borda
> ≳ weighted-sum ≳ either baseline is typical; treat ties within ~1–2
> NDCG points as noise on this small a set.

Per-method case studies:

- **Where weighted-sum wins.** On `q07` (producer idempotence, a
  high-precision lexical cue), BM25's top-1 scored 18.4 and dense's
  top-1 scored 0.84. After min-max per system, both top-1s contribute
  1.0 — BM25's rank-5 (score 12.2) also contributes ~0.65, higher than
  dense's rank-2 (~0.6). Weighted-sum pulled a correct second document
  from BM25 that RRF left behind at a lower rank.
- **Where weighted-sum loses.** On `q14` (acronym-heavy query that BM25
  mis-tokenizes), BM25's top-100 contains almost no relevant document;
  its scores are compressed near the bottom of its range. After min-max,
  BM25's contribution is dominated by one document that happens to match
  on a side term. Weighted-sum gave that irrelevant document half the
  vote; RRF effectively ignored it because it was at rank 30 (low
  reciprocal contribution).
- **Where RRF wins.** On `q12` ("exactly-once delivery guarantees"), BM25
  and dense return different top-10s. The relevant document is rank 7 in
  BM25 and rank 3 in dense. RRF scores it `1/(60+7) + 1/(60+3) ≈ 0.030`
  — higher than any document in either single list except the mutual
  top-1 — which pushed it into the fused top-3.
- **Where Borda wins.** On `q17` (where the relevant document is rank 1
  in one list and rank 40 in the other), Borda's linear-in-rank
  weighting gave the rank-40 showing more credit than RRF would —
  `(100 - 40 + 1) = 61` vs. `1/(60+40) = 0.01`. The relevant document
  appeared at fused rank 1 under Borda and rank 4 under RRF.
- **Where Borda loses.** When the lists have very different lengths or
  the tails of the lists are noisy, Borda's linear weighting pulls junk
  up. On `q09`, dense's rank-80 was an off-topic chunk that still got
  `(100 - 80 + 1) = 21` points, enough to push past weakly-relevant
  BM25 rank-20 items. RRF ignored that tail entirely.

### 2.6 `notes/recommendation.md` — the decision

**Primary pick: RRF with `k = 60`.** Reasons, in order:

1. Invariant to score scale. Avoids per-release re-tuning when BM25
   analyzer or embedding model changes and shifts the score
   distribution.
2. No per-query knob. `k` is one global constant; the method does not
   expose `α` per retriever pair that would need calibration.
3. First-class in OpenSearch (`rrf` processor), Elasticsearch (`rank:
   rrf`), and Vespa (`reciprocal_rank_fusion` rank feature) — moving
   the fusion into the engine is a one-config-line change, not a
   middleware rewrite.
4. On the bake-off, RRF had the most "wins vs. best single system."

**Escape hatch: weighted-sum (with per-system normalization).** Switch
when either:

- The two retrievers are calibrated *on the same scale* (e.g., two dense
  retrievers over related embedders) and raw scores carry information
  RRF discards; or
- A product requirement demands a tunable "lexical weight" slider the
  user can see and adjust (RRF has no honest version of that).

**Reserve: Borda count.** Keep Borda in the test matrix but don't ship
it. Its only real advantage over RRF is linear-in-rank weighting, which
helps for deep-rank agreement signals and hurts for noisy tails. It
tends to win in one eval slice and lose in another; absent a clear
requirement, RRF is the safer default.

**Not recommended:** condorcet-style fusion and learned-to-combine
(logistic over retriever scores) at this scope. Condorcet is more
expensive to compute and doesn't beat RRF on the paper's own benchmarks.
Learned combiners work (they are essentially mod-107's reranker) but
require judgment data to train and a model-serving story that this
module does not have in scope.

**Trigger for re-running this bake-off:**

- Any change to the embedding model (score scale shifts).
- Any change to the BM25 analyzer (score distribution shifts).
- A move from top-100 to top-K ≠ 100 per system.
- Adding a third retrieval system (equal-weight fusion assumes
  independence of voters, which is false for near-duplicate systems).

## 3. Validation steps

1. **Fairness check.** All three methods must consume the same per-query
   candidate lists. Run `run_bakeoff.py` with `--log-candidates` and
   diff the candidate sets across methods: they should be byte-identical.
2. **Normalization check.** Scale BM25's raw `_score` by 10× artificially
   (just multiply in the driver) and re-run. RRF's output must be
   identical; Borda's output must be identical; weighted-sum's output
   should also be identical *if min-max normalization is per-system*.
   If weighted-sum drifts, the normalization is broken.
3. **Hyperparameter sensitivity.** Sweep weighted-sum's `α` over
   `{0.2, 0.5, 0.8}`. The best `α` on the 10-query eval must be
   reported; if `α = 0.5` is not best, the bake-off should quote
   both `α = 0.5` (fair) and `best α` (tuned) rows. Reporting only the
   best `α` while quoting a fixed `k` for RRF is unfair.
4. **Per-query case study.** The learner names one query each where
   weighted-sum won and lost, and explains from the top-10 of each
   system *why*. "RRF wins on average" is not the pass criterion; the
   mechanism-level story is.
5. **Recommendation is falsifiable.** The written recommendation names
   one measurable trigger that would switch the pick. "We pick whichever
   is highest NDCG" is not a trigger; it is a tautology.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Three implementations | Weighted-sum, RRF, Borda all implemented; share one candidate pool per query. |
| Normalization | Weighted-sum uses per-system normalization; missing documents handled as 0, not mean. |
| Fairness | All methods evaluated at the same top-K; same tie-break; same qrels. |
| Reporting | NDCG@10, Recall@10, MRR for each method; per-query win/loss counts vs. best single system. |
| Case studies | At least one worked query per method where the method won, one where it lost; the explanation references the system top-10, not just the summary metrics. |
| Recommendation | One primary, one escape hatch, one measurable trigger. |
| Honesty | The recommendation does not read as "RRF is best because the ES docs say so." |

Fail-fast smells:

- Reporting one summary number per method with no per-query table.
- Tuning weighted-sum's `α` over the eval set and comparing it to RRF
  with a fixed `k`. Either both are tuned on a held-out slice or both
  are fixed.
- Running BM25 three times, once per method, and claiming fairness.
- Normalizing scores *across both systems* (so BM25's absolute maximum
  dominates cosine's).
- Fusing at top-10 depth. The methods differ most on the tail; shallow
  fusion flattens the comparison.

## 5. Common mistakes

- **Treating the methods as interchangeable.** Each encodes a different
  assumption about what retriever scores mean. Weighted-sum assumes
  calibrated scores. RRF assumes rank is the only meaningful signal.
  Borda assumes rank is linearly meaningful. If any of those
  assumptions is wrong for the eval, the method underperforms — not
  because the formula is wrong, but because the assumption is.
- **Hand-tuning weighted-sum's `α` on the eval set.** Overfits to the
  eval. Hold out 20% of queries for final numbers.
- **Ignoring Borda's tail sensitivity.** Borda rewards *being on the
  list at all* much more than RRF does. Noisy tails hurt it. The
  classical fix is to truncate the per-system lists to top-K before
  fusion; set K to the depth where the system is still correct more
  often than not.
- **Reading one NDCG number and declaring a winner.** On 20 queries, the
  noise floor is ±2 NDCG points. Any of the three methods can win by
  that margin on a particular draw. Report a 95% bootstrap interval or
  acknowledge the floor explicitly.
- **Comparing a score-fusion method at a different depth than a rank-
  fusion method.** Weighted-sum requires scores for every candidate it
  considers; truncating to top-10 per system limits its candidate pool.
  RRF doesn't need scores and can fuse at top-100 without extra cost.
  Set K equally.
- **Omitting BM25-only and dense-only baselines.** The interesting
  question is "does fusion help over the better single system,
  per-query" — not "does fusion method A beat method B." Include both
  single-system baselines on every row of the final table.
- **Shipping without a trigger to re-run.** Scores drift. An eval that
  was definitive six months ago can be inverted by a BM25 analyzer
  change. The recommendation document should name the triggers.

## 6. References

- RRF: Cormack, Clarke, Büttcher. "Reciprocal rank fusion outperforms
  Condorcet and individual rank learning methods." SIGIR 2009.
  DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- Borda in IR: Aslam & Montague. "Models for metasearch." SIGIR 2001.
  DOI: [10.1145/383952.384007](https://doi.org/10.1145/383952.384007).
- Elasticsearch — RRF reference:
  [elastic.co/guide/en/elasticsearch/reference/current/rrf.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html).
- OpenSearch — hybrid search and RRF:
  [opensearch.org/docs/latest/search-plugins/hybrid-search/](https://opensearch.org/docs/latest/search-plugins/hybrid-search/).
- Vespa — reciprocal rank fusion rank feature:
  [docs.vespa.ai/en/reference/rank-features.html](https://docs.vespa.ai/en/reference/rank-features.html).
- NDCG: Järvelin & Kekäläinen, "Cumulated gain-based evaluation of IR
  techniques." ACM TOIS 2002.
  DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418).
- BM25 default in Lucene:
  [lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).
- Risk / governance framings consulted while drafting:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
