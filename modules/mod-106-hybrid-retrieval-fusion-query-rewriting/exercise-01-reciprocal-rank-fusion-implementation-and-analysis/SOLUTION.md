# Exercise 01 — Solution: Reciprocal Rank Fusion Implementation and Analysis

This is a reference solution for
`exercise-01-reciprocal-rank-fusion-implementation-and-analysis.md`. It is an
**implementation + analysis** exercise. The learner builds a reciprocal rank
fusion (RRF) combiner for two retrieval lists (lexical BM25 and dense ANN),
runs it on a small eval set, and explains in writing *why* the formula behaves
the way it does when the two lists disagree.

The teaching point is that RRF is a **rank-only** fusion method — it ignores
the raw scores of each system entirely. That is its biggest strength (BM25
scores and cosine similarities live on different scales and don't combine
cleanly) and its biggest weakness (two confidently-right systems and two
unsure systems vote with equal weight). Learners who treat RRF as "the default
hybrid combiner" without understanding what it discards will be surprised
later when a better-calibrated system underperforms a worse one on fused NDCG.

## 1. Solution overview

The deliverable has five parts:

1. **`src/rrf.py`** — a reference implementation of RRF for N ranked lists,
   with the Cormack et al. (2009) form `score(d) = Σ_i 1 / (k + rank_i(d))`.
   The implementation is ~30 lines, including the convention for "document d
   is not in list i."
2. **`src/run_fusion.py`** — a driver that reads the eval set, runs BM25 and
   dense retrieval separately, applies RRF, and writes a fused ranked list per
   query.
3. **`eval/qrels.tsv` + `eval/queries.tsv`** — a tiny 10-query judgment set
   over the mod-106 teaching corpus (chunked articles from mod-102). Labels
   are 0 / 1 / 2 / 3.
4. **`eval/report.md`** — NDCG@10 and Recall@10 for BM25-only, dense-only,
   and RRF, plus a per-query delta table so the analysis is specific rather
   than "RRF helped overall."
5. **`notes/analysis.md`** — the written analysis: when RRF wins, when it
   loses, what the `k` parameter actually does, and why the standard advice
   is `k = 60`.

The reference below works through each part. The code is Python and uses only
the standard library plus `numpy` for NDCG. A real deployment would push the
fusion into the search layer (OpenSearch's `rank` clause, Vespa's rank
profile, or a middleware service) — doing it in Python first makes the math
visible.

## 2. Worked answer or implementation

### 2.1 RRF in one paragraph

Reciprocal Rank Fusion was introduced by Cormack, Clarke, and Büttcher
(SIGIR 2009) as a near-zero-config way to combine ranked lists from
heterogeneous retrieval systems. For each document `d`, the fused score is

```
RRF(d) = Σ over systems i   1 / (k + rank_i(d))
```

where `rank_i(d)` is 1-indexed and documents not returned by system `i`
contribute 0. The constant `k` damps the contribution of low-ranked
documents; the paper uses `k = 60` and it has stuck as the default across
Elasticsearch, OpenSearch, Vespa, LangChain, LlamaIndex, and most practical
implementations. The important property is that `RRF` depends only on the
*rank* a document received, not on the raw score — which eliminates the
need to normalize BM25 scores and cosine similarities onto a shared scale.

### 2.2 `src/rrf.py` — the reference implementation

```python
"""Reciprocal Rank Fusion over N ranked lists.

Reference: Cormack, Clarke, Büttcher. "Reciprocal rank fusion outperforms
Condorcet and individual rank learning methods." SIGIR 2009.
"""
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def rrf(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse N ranked lists of document IDs into one list.

    Args:
      ranked_lists: list of ranked lists; each inner list is doc IDs in rank
        order (best first). A doc missing from a list contributes 0 from that
        list.
      k: the RRF damping constant. Cormack et al. (2009) use k=60.
      weights: optional per-list weights. If None, each list weights 1.0.

    Returns:
      A list of (doc_id, fused_score) tuples, sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("len(weights) must equal len(ranked_lists)")

    scores: dict[str, float] = defaultdict(float)
    for w, ranked in zip(weights, ranked_lists):
        for position, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + position)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

Notes on this implementation:

- **1-indexed ranks.** The original paper uses rank 1 for the top document.
  Using 0-indexed ranks silently halves the top document's contribution
  relative to the paper's formulation and makes `k = 60` behave differently
  than every reference describes.
- **Missing documents contribute zero.** If a document is only returned by
  one system, that system's rank-1 contribution survives; the other systems
  simply don't vote for it. This is why RRF tolerates lists of different
  lengths without truncation.
- **Deterministic tie-breaking.** On exact score ties, Python's sort is
  stable, so insertion order from the `defaultdict` wins. If you need
  reproducibility across language implementations, break ties by `doc_id`
  lexicographically.
- **Weights are allowed but not required.** Pure RRF has no weights. A
  common extension is a per-list weight `w_i`: `Σ_i w_i / (k + rank_i(d))`.
  Elasticsearch and OpenSearch both support `rank` sub-clause weights. The
  teaching version keeps `weights` so exercise-02 can instrument RRF against
  weighted-sum fairly.

### 2.3 `src/run_fusion.py` — the driver

```python
"""Run BM25 + dense retrieval, fuse with RRF, write per-query rankings."""
import json
import sys
from pathlib import Path

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

from rrf import rrf

CLIENT = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
INDEX = "mod106_chunks"
TOP_K = 100          # per-system recall depth
FUSE_K = 60          # RRF damping constant
EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")


def bm25_search(query: str, top_k: int = TOP_K) -> list[str]:
    body = {
        "size": top_k,
        "query": {"match": {"body": query}},
        "_source": False,
    }
    resp = CLIENT.search(index=INDEX, body=body)
    return [hit["_id"] for hit in resp["hits"]["hits"]]


def dense_search(query: str, top_k: int = TOP_K) -> list[str]:
    vec = EMBEDDER.encode(query, normalize_embeddings=True).tolist()
    body = {
        "size": top_k,
        "query": {"knn": {"embedding": {"vector": vec, "k": top_k}}},
        "_source": False,
    }
    resp = CLIENT.search(index=INDEX, body=body)
    return [hit["_id"] for hit in resp["hits"]["hits"]]


def main(queries_path: Path, out_path: Path) -> None:
    out = out_path.open("w")
    for line in queries_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        qid, query = line.split("\t", 1)
        lex = bm25_search(query)
        dns = dense_search(query)
        fused = rrf([lex, dns], k=FUSE_K)
        for rank, (doc_id, score) in enumerate(fused[:100], start=1):
            out.write(f"{qid}\t{doc_id}\t{rank}\t{score:.6f}\n")
    out.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
```

The driver is intentionally plain. In production you would push the fusion
into the retrieval engine itself — OpenSearch 2.12+ exposes an `rrf` processor
and Elasticsearch 8.8+ exposes a `rank: { rrf: {...} }` clause on `_search`.
Doing the fusion in Python first keeps the math inspectable; once the
behavior is understood, move the combine step down the stack so the fused
result is one HTTP round-trip, not two.

### 2.4 `eval/qrels.tsv` and `eval/queries.tsv`

A 10-query judgment set is enough to see the pattern (NDCG stabilizes
around 20–30 queries; this is a teaching sample, not an evaluation
benchmark). Judgments are graded 0–3, following TREC conventions:

```
# qid   query
q01     consumer group rebalance stuck in Kafka
q02     DLQ retry patterns for exactly-once delivery
q03     what does "idempotent producer" mean
q04     how to tune `fetch.min.bytes` for low-latency consumers
q05     ksqlDB vs Kafka Streams for stateful joins
q06     sarama consumer offsets committed twice
q07     producer idempotence enable.idempotence true
q08     Kafka Connect SMT drop null fields
q09     MirrorMaker 2 offset translation
q10     Avro schema evolution breaking change example
```

```
# qid   doc_id        label
q01     ch_003412     3
q01     ch_003411     2
q01     ch_000871     2
q01     ch_004002     1
q01     ch_005118     0
...
```

Keep judgments on documents that **the first-phase retrievers actually
return** (top-100 from BM25, top-100 from dense). Judging documents that
neither system surfaces teaches fusion nothing.

### 2.5 `eval/report.md` — NDCG@10 and Recall@10

Run each of `bm25_only`, `dense_only`, and `rrf` over the 10-query set and
tabulate. A representative (illustrative, not authoritative) teaching-run
shape looks like this:

| System | NDCG@10 | Recall@10 |
|---|---|---|
| BM25 only | 0.48 | 0.55 |
| Dense only | 0.52 | 0.61 |
| RRF (k=60) | **0.61** | **0.72** |

Per-query delta (RRF − better single system):

| qid | Better single | Δ NDCG@10 (RRF vs. better single) | Notes |
|---|---|---|---|
| q01 | Dense (0.60) | +0.10 | Dense found the top-1; BM25 found 2 highly-rel docs dense missed; RRF stacked them. |
| q02 | BM25 (0.45) | +0.18 | "DLQ" is a strong lexical cue; dense widened recall. |
| q03 | Dense (0.70) | −0.05 | Definition query — dense already perfect; BM25 injected a lower-rel doc. |
| q04 | BM25 (0.58) | +0.00 | BM25 already saturated; RRF neither helped nor hurt. |
| q05 | Dense (0.42) | +0.15 | Both systems disagreed strongly; voted top-3 emerged from the overlap. |
| … | … | … | … |

The point of the per-query table is that "RRF helped on average" is the
least-interesting form of the result. The interesting form is **when** it
helped and **when** it hurt. Exercise-02 picks this thread up.

> **Reproducibility note.** The numeric table above is a shape, not a
> guaranteed result. Learners running this against their own corpus and
> embedder will see different absolute NDCG values. The *pattern* — RRF
> usually wins overall, loses on queries where one system is already near
> ceiling — is the generalizable finding.

### 2.6 `notes/analysis.md` — what the formula is doing

A few invariants to argue from, in increasing subtlety:

1. **Top-1 contribution.** A document ranked #1 by system `i` contributes
   `1 / (60 + 1) = 1/61 ≈ 0.0164`. A document ranked #2 contributes
   `1/62 ≈ 0.0161`. The gap between top-1 and top-2 is tiny (~2%). With
   `k = 10` instead, the gap would be `1/11 − 1/12 ≈ 0.0076`, almost 10%
   of the top-1 value. **`k` controls how much RRF cares about the exact
   top of each list.**
2. **Rank 100 vs. rank 10.** At `k = 60`, rank 100 contributes `1/160 ≈
   0.00625` and rank 10 contributes `1/70 ≈ 0.0143`. The ratio is ~2.3×.
   So a document that one system puts at rank 100 and the other puts at
   rank 10 carries ~`0.021` fused — more than a document ranked 15 by one
   system alone (~`0.0133`). **Agreement across systems is the real
   signal.**
3. **A head-vs-body confident system.** If system A is miscalibrated at
   the top (confident but wrong at rank 1, correct at rank 5) and system B
   is well-calibrated (correct at rank 1, redundant thereafter), weighted-
   sum fusion punishes B for being right only once — it has one high score
   and many low. RRF doesn't care: B's correct top-1 and A's correct
   rank-5 each contribute independently.
4. **Why `k = 60` and not something else.** Cormack et al. empirically
   chose `k = 60` on TREC runs; the choice has stuck because the result is
   not very sensitive to `k` in the 20–100 range. Values below ~10
   effectively make the fusion "did the top-1 vote for this" with little
   contribution from the tail. Values above ~200 flatten the rank signal
   and start to behave like "did at least one system return this." A
   defensible answer for a project is: fix `k = 60` and vary it in a
   sensitivity study only if the eval metric looks pinned.

### 2.7 When RRF is the wrong choice (preview of exercise-02)

- The retrieval systems are **calibrated on the same scale** (e.g., two
  dense retrievers over the same embedding space; their cosines are
  comparable). Weighted sum of normalized scores keeps more information
  than rank-only fusion.
- One system is **structured** (e.g., ACL filter + boost) and must act as
  a hard filter, not a soft vote. RRF gives no way to express a hard
  precondition. Filter-then-fuse (exercise-05) is the right pattern.
- The team wants **monotonic score → probability** calibration (temperature
  scaling, Platt scaling). RRF outputs a fused score that is not
  interpretable as a probability.

## 3. Validation steps

1. **Unit test the implementation.** Given
   `ranked_lists = [["a","b","c"], ["b","a","d"]]` and `k = 60`, the fused
   result must be `[("b", 2/61 + 0/… ), ("a", 1/61 + 1/62), …]` calculated
   by hand. Test two cases: exact agreement ("a","b" both at rank 1) and
   total disagreement. Both should produce documented, hand-verifiable
   scores.
2. **Confirm invariance to score scale.** Running the driver with BM25
   scores multiplied by 1000 must produce exactly the same fused output as
   the unscaled run. If the output changes, the implementation is leaking
   raw scores into the fusion (classic bug: sorting by score first and
   then re-ranking).
3. **Confirm the `k` sensitivity.** Run RRF with `k = 10, 60, 200` on the
   same 10 queries. The ordering of the top-10 should change for a
   minority of queries (typically 2–4 out of 10) and stay identical for
   the rest. If the top-10 is identical across all three `k` values on
   all queries, either the two systems agree too much (eval set is too
   easy) or the rank-cutoff for fusion is too shallow.
4. **Reproduce the paper's claim qualitatively.** Cormack et al. argue RRF
   outperforms Condorcet fusion and per-system reranking on average over
   TREC. The learner cannot reproduce TREC, but should be able to
   demonstrate RRF ≥ max(BM25, dense) on NDCG@10 averaged over the 10
   teaching queries. If RRF loses on average, something is wrong in the
   setup: usually the recall depth per system is too shallow, or the
   dense embedder is far enough off-domain that its ranking carries no
   signal.
5. **Document a failure case.** Exercise-02 picks up fusion comparison;
   exercise-01 passes when the learner names at least one query where
   RRF strictly *hurt* and can explain why from the per-system top-10.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Formula | 1-indexed ranks; `k = 60` default; missing-document convention documented and tested. |
| Score invariance | Doubling one system's raw scores does not change the fused output. |
| Per-query analysis | Delta table with at least one win case and at least one loss case explained from the per-system top-10. |
| Recall depth | Fused on at least top-100 per system, not top-10. Top-10 fusion misses the agreement signal. |
| Deterministic tie-break | Documented and implemented; same inputs produce the same ordering. |
| Weighted RRF (optional) | If the learner adds weights, each weight has a stated interpretation and is set to 1.0 in the base run. |
| `k` sensitivity | A short note showing one query whose top-10 changes between `k=10` and `k=200`. |
| Honest numbers | NDCG numbers are reported for the actual run; the learner does not quote the "representative" table from the reference. |

Fail-fast smells:

- Sorting each list by score before fusion. The sort is a no-op if the
  list is already in rank order; if the learner thinks they need to sort,
  they have confused score fusion with rank fusion.
- Normalizing BM25 and cosine scores before RRF. The entire point of RRF
  is that it does not use raw scores.
- Using 0-indexed ranks "because that's more Pythonic." This silently
  changes the formula and breaks comparability with every reference
  implementation.
- Reporting only Recall@10, not NDCG@10. Recall is easy to improve by
  union; the interesting fusion question is whether good documents move
  *up*.

## 5. Common mistakes

- **Treating RRF as score fusion.** Beginners commonly write
  `score = alpha * bm25 + (1-alpha) * cosine`, call it RRF, and sweep
  `alpha`. That is weighted-sum fusion, which is a legitimate method
  (and exercise-02 compares it fairly). It is not RRF and does not have
  RRF's scale-invariance property.
- **Fusing shallow lists.** Taking top-10 from each system and fusing
  gives at most 20 documents. The agreement signal lives in the body of
  each list; `k = 60` is calibrated for lists with a long tail. Use at
  least top-100 per system for the fuse.
- **Judging documents RRF can't retrieve.** The qrels should cover
  documents the retrieval system *actually returned* for at least one of
  the queries. A document in the qrels that no system surfaces contributes
  only to `Recall@∞`, which neither BM25-only nor RRF can improve.
- **Reporting fused NDCG over the union of systems' top-K as the
  denominator.** NDCG's denominator is the ideal ranking of all judged
  documents for that query, independent of what any retriever returned.
  Using the retrieved set as the denominator inflates NDCG for every
  system equally and hides the real gap.
- **Assuming RRF is a replacement for reranking.** RRF is a cheap *fusion*
  step that happens before the reranker in a typical cascade. A learned
  cross-encoder reranker (mod-107) operates on the top-K fused list; RRF
  neither competes with nor replaces that layer.
- **Setting `k` tiny to "make the top-1 count more."** That does move the
  mass up, but it also makes the fusion unstable at the top — a single
  rank-1 vote dominates. Teams that hard-code `k = 1` usually end up
  effectively running per-query A/B between the two systems, which is
  worse than any principled fusion.
- **Fusing more than two systems without weighting.** Equal-weight RRF
  across 5 retrieval systems implicitly says "each voter is equally
  calibrated." If three of them are near-duplicates (e.g., three dense
  retrievers over similar embedding models), the ensemble silently
  overweights the duplicate axis.

## 6. References

- **Original RRF paper.** Cormack, Clarke, Büttcher. "Reciprocal rank
  fusion outperforms Condorcet and individual rank learning methods."
  SIGIR 2009. DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- Elasticsearch — reciprocal rank fusion (RRF):
  [elastic.co/guide/en/elasticsearch/reference/current/rrf.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html).
- OpenSearch — hybrid search and RRF:
  [opensearch.org/docs/latest/search-plugins/hybrid-search/](https://opensearch.org/docs/latest/search-plugins/hybrid-search/).
- Vespa — rank fusion recipes and `reciprocal_rank_fusion`:
  [docs.vespa.ai/en/reference/rank-features.html](https://docs.vespa.ai/en/reference/rank-features.html).
- BM25 similarity default in Lucene:
  [lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).
- NDCG definition and usage:
  Järvelin & Kekäläinen, "Cumulated gain-based evaluation of IR
  techniques." ACM TOIS 2002. DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418).
- Risk / governance framings consulted while drafting:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
