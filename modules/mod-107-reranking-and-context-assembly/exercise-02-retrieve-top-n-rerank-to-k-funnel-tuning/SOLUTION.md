# Exercise 02 — Solution: Retrieve-Top-N → Rerank-to-K Funnel Tuning

This is a reference solution for
`exercise-02-retrieve-top-n-rerank-to-k-funnel-tuning.md`. It is an
**end-to-end implementation** exercise. The learner builds a sweep harness
over the two knobs of a reranking pipeline — `N` (first-phase recall depth)
and `K` (final list size handed to the prompt builder) — and finds the
budget-constrained sweet spot on the mod-109-shaped eval set.

The teaching point: these two numbers are not independent. `N` controls
*recall* (how much of the relevant pool is in play for the reranker) and
`K` controls *precision* (how much of the pool the prompt actually sees).
Tuning them separately is a mistake; tuning only one of them while the
other is fixed usually surfaces a false optimum.

## 1. Solution overview

The deliverable has five parts:

1. **`src/funnel.py`** — the funnel runner: given `(N, K)` and a query,
   call first-phase retrieval, call the reranker, take the top `K`,
   return the result plus timing. Pluggable retriever and reranker.
2. **`src/sweep.py`** — the sweep harness. Runs the funnel across a
   Cartesian grid of `(N, K)` and collects per-query metrics:
   `recall@K`, `nDCG@K`, `MRR@K`, `p50/p95 latency`, and a derived
   `ΔnDCG vs. N=K baseline` to isolate the reranker's contribution.
3. **`judgments/qrels.tsv`** — a small graded judgment set in TREC
   format, reused from mod-109's eval scaffold (≥20 queries × ≥20
   graded docs).
4. **`notes/results.md`** — the heat-map table of `(N, K) → metric`
   for each metric, the identified operating point, and the ablation
   that justifies it.
5. **`configs/funnel.yaml`** — the chosen operating point, exported in
   the shape the downstream prompt builder (exercise-04) reads.

The exercise is scoped to a *working sweep + defensible recommendation*,
not state-of-the-art NDCG. If the learner can show that the chosen
`(N, K)` beats the `N=K` no-reranker baseline on a held-out slice and
stays inside the latency budget, the exercise passes.

## 2. Worked answer or implementation

### 2.1 The funnel in one paragraph

A retrieval-then-rerank pipeline has two independent gates. The retriever
returns `N` candidates; the reranker scores and re-sorts them; the
pipeline emits the top `K`. Recall is bounded by whether the gold doc
is in the retriever's top `N` — if it isn't, no reranker can recover
it. Precision is bounded by the reranker's accuracy on the top-K cut.
If the reranker is perfect and the retriever gives you the gold doc
in the top-N, the gold doc ends up in the top-K. So `N` is bought with
*retrieval latency*; `K` is bought with *prompt tokens*. You are
optimising on a joint budget.

### 2.2 `src/funnel.py` — the funnel runner

```python
"""Retrieve-top-N then rerank-to-K pipeline."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Sequence

@dataclass
class FunnelResult:
    docs: list[str]           # ordered doc ids, length = K
    scores: list[float]       # reranker scores aligned with docs
    first_phase_ids: list[str]  # the length-N first-phase set, in order
    t_retrieve_ms: float
    t_rerank_ms: float

Retriever = Callable[[str, int], list[tuple[str, float]]]  # (docid, score)
Reranker  = Callable[[str, list[str]], list[float]]        # per-doc scores

def run_funnel(
    query: str,
    N: int,
    K: int,
    retriever: Retriever,
    reranker: Reranker,
) -> FunnelResult:
    assert 1 <= K <= N, "K must be in [1, N]"
    t0 = time.perf_counter()
    first_phase = retriever(query, N)
    t1 = time.perf_counter()
    ids = [d for d, _ in first_phase]
    if not ids:
        return FunnelResult([], [], [], (t1 - t0) * 1000, 0.0)

    scores = reranker(query, ids)
    t2 = time.perf_counter()

    paired = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:K]
    top_ids   = [d for d, _ in paired]
    top_scores = [s for _, s in paired]

    return FunnelResult(
        docs=top_ids,
        scores=top_scores,
        first_phase_ids=ids,
        t_retrieve_ms=(t1 - t0) * 1000,
        t_rerank_ms  =(t2 - t1) * 1000,
    )
```

**Design choices and why:**

- The retriever returns `(docid, score)` tuples so the ablation can
  compare "rerank" against "retriever score only" without a second
  call.
- The reranker is a *pure* `(q, docs) → scores` function. This lets
  the sweep plug a cross-encoder, a hosted API, or a no-op (identity)
  reranker without changing the harness.
- Times are measured *around* the two steps; total pipeline latency
  is the sum. The harness does **not** fold in LLM generation —
  that's exercise-04/05 territory.
- `K ≤ N` is asserted. A `K > N` configuration is nonsense and the
  assertion fails fast at config-load time.

### 2.3 `src/sweep.py` — the sweep harness

```python
"""Sweep (N, K) grid and emit per-query metrics + aggregates."""
from __future__ import annotations
import csv, json, math, pathlib
from statistics import median
from typing import Iterable

from funnel import FunnelResult, Retriever, Reranker, run_funnel

def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    rel = {d for d, lab in qrels.items() if lab > 0}
    if not rel:
        return 0.0
    hits = len(set(ranked_ids[:k]) & rel)
    return hits / len(rel)

def mrr_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked_ids[:k], start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for qid, doc, label in csv.reader(open(path), delimiter="\t"):
        qrels.setdefault(qid, {})[doc] = int(label)
    return qrels

def load_queries(path: str) -> dict[str, str]:
    return { qid: text for qid, text in csv.reader(open(path), delimiter="\t") }

def sweep(
    queries: dict[str, str],
    qrels:   dict[str, dict[str, int]],
    grid:    list[tuple[int, int]],
    retriever: Retriever,
    reranker:  Reranker,
    out_path:  str,
) -> None:
    rows: list[dict] = []
    for N, K in grid:
        per_q = []
        for qid, qtext in queries.items():
            r: FunnelResult = run_funnel(qtext, N, K, retriever, reranker)
            q_rels = qrels.get(qid, {})
            per_q.append({
                "qid": qid, "N": N, "K": K,
                "ndcg": ndcg_at_k(r.docs, q_rels, K),
                "recall_at_K": recall_at_k(r.docs, q_rels, K),
                "mrr_at_K":    mrr_at_k(r.docs, q_rels, K),
                "recall_at_N_firstphase": recall_at_k(r.first_phase_ids, q_rels, N),
                "t_retrieve_ms": r.t_retrieve_ms,
                "t_rerank_ms":   r.t_rerank_ms,
                "t_total_ms":    r.t_retrieve_ms + r.t_rerank_ms,
            })
        agg = {
            "N": N, "K": K,
            "mean_ndcg":    sum(x["ndcg"] for x in per_q) / len(per_q),
            "mean_recall":  sum(x["recall_at_K"] for x in per_q) / len(per_q),
            "firstphase_recall": sum(x["recall_at_N_firstphase"] for x in per_q) / len(per_q),
            "mean_mrr":     sum(x["mrr_at_K"] for x in per_q) / len(per_q),
            "p50_total_ms": median(x["t_total_ms"] for x in per_q),
            "p95_total_ms": sorted(x["t_total_ms"] for x in per_q)[int(len(per_q) * 0.95)],
            "per_q": per_q,
        }
        rows.append(agg)
        print(f"N={N:3d} K={K:2d}  nDCG@K={agg['mean_ndcg']:.4f}  "
              f"recall@K={agg['mean_recall']:.4f}  "
              f"first-phase recall@N={agg['firstphase_recall']:.4f}  "
              f"p50={agg['p50_total_ms']:.0f}ms  p95={agg['p95_total_ms']:.0f}ms")

    pathlib.Path(out_path).write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    import os
    queries = load_queries(os.environ.get("QUERIES",  "judgments/queries.tsv"))
    qrels   = load_qrels  (os.environ.get("QRELS",    "judgments/qrels.tsv"))
    grid = [
        (N, K)
        for N in (10, 25, 50, 100, 200, 400)
        for K in (1, 3, 5, 10)
        if K <= N
    ]
    # retriever / reranker wired in from mod-106 / exercise-01.
    from mod106.hybrid import hybrid_retriever as retriever
    from reranker import cross_encoder_rerank as reranker
    sweep(queries, qrels, grid, retriever, reranker, "notes/sweep.json")
```

**Choices and why:**

- `firstphase_recall` is reported **separately**. This isolates
  whether a drop in nDCG@K comes from the first phase missing the
  gold doc (reranker cannot help) or the reranker picking wrong
  within the candidate set (reranker needs tuning).
- Grid uses geometric spacing (10, 25, 50, 100, 200, 400). Linear
  spacing wastes effort around points where the metric has
  saturated.
- `K` stops at 10. For most RAG prompts, context budget caps `K` at
  5–8 chunks; `K > 10` is only relevant for very short chunks or
  long-context models (exercise-05).
- The sweep persists per-query metrics to JSON so downstream
  analysis (histograms, bootstraps) does not re-run the pipeline.

### 2.4 `judgments/qrels.tsv` — the judgment set

TREC-style graded judgments, same shape as mod-105:

```
# qid   doc_id              label
q01     chunk_004521        3
q01     chunk_004522        2
q01     chunk_008810        1
q01     chunk_000010        0
q02     chunk_014377        3
...
```

Keep ≥20 queries, each with ≥20 judged documents drawn from the
retriever's **top-200** (the deepest `N` in the sweep grid). Judging
beyond the sweep's deepest `N` is wasted; judging below it means the
reranker never sees the judged document.

Hold out 20–30% of the queries for the reported numbers. Running the
sweep on the training slice is fine for choosing the operating point;
the number that lands in `results.md` is from the held-out slice.

### 2.5 `notes/results.md` — reporting shape

Three artefacts land in results:

**(a) Heat-map for `nDCG@K` on the held-out slice:**

```
          K=1     K=3     K=5     K=10
N=10     0.312   0.455   0.512     —
N=25     0.340   0.491   0.564   0.625
N=50     0.358   0.517   0.591   0.651
N=100    0.361   0.525   0.603   0.662     ← chosen
N=200    0.361   0.527   0.604   0.663
N=400    0.360   0.527   0.604   0.663     ← saturation
```

Reading: nDCG@K improves with `N` up to a point then flattens. The
operating point should sit on the shoulder, not past it.

**(b) Latency table:**

```
          p50 (ms)                    p95 (ms)
         K=1  K=3  K=5  K=10         K=1  K=3  K=5  K=10
N=10     55   56   58    —          110  115  118    —
N=25     72   74   76   80          130  135  140  150
N=50    105  110  115  125          200  210  220  240
N=100   180  195  210  230          360  380  400  430    ← chosen
N=200   310  330  360  400          620  650  700  760
N=400   560  600  650  720         1100 1150 1240 1360
```

Reading: p95 latency roughly doubles with each doubling of `N` once
the GPU is saturated (reranker time dominates beyond N≈50 here).

**(c) Reranker ablation (`nDCG@K` with and without the reranker):**

```
Config               nDCG@K    Δ vs no-rerank
N=K=10 no-rerank     0.512            —
N=100, K=10 + rerank 0.662         +0.150
```

The ablation proves that the reranker — not just taking more
candidates — is what produced the quality gain. A learner who shows
only "the sweep found a good config" and no ablation has not
isolated the effect.

### 2.6 `configs/funnel.yaml` — the chosen operating point

```yaml
# Chosen operating point for the mod-107 baseline funnel.
# Trigger a re-sweep when (a) retriever changes, (b) reranker model changes,
# or (c) the latency budget changes by more than 20%.
funnel:
  N: 100
  K: 5
  retriever: hybrid_rrf_v2       # mod-106 export
  reranker:  bge-reranker-v2-m3  # exercise-01 pick
  latency_budget_ms:
    p50: 300
    p95: 600
  eval_anchor:
    held_out_qrels: judgments/holdout_qrels.tsv
    mean_ndcg_at_K: 0.603
    firstphase_recall_at_N: 0.978
```

Encoding the eval anchor in the config matters: when the next
engineer swaps the reranker, re-running the sweep should at minimum
reproduce the anchor number on the held-out slice. A drop is a
regression and a build-breaking event.

### 2.7 How to read the sweep: a worked decision

Ranking the candidate rows by `mean nDCG@K` alone, `N=200/K=10` ties
`N=100/K=10`. Within the p95 budget (600 ms), `N=100/K=5` is the
tightest row that passes both metric and latency gates:

- `first-phase recall@N=100 = 0.978` — the retriever surfaces the
  gold doc for ~98% of queries; going to N=200 would only recover
  ~2% of queries at the cost of doubling rerank latency.
- The nDCG@5 → nDCG@10 gap (0.603 → 0.662) is ~0.06. For a
  prompt-token-budgeted system (exercise-04), 5 chunks is more
  commonly the operating point than 10; the gap informs a secondary
  decision about context budget in exercise-05.
- Going from K=5 to K=1 costs ~0.24 nDCG. The chunk at position 3,
  4, 5 contributes real quality; truncating to top-1 is a mistake.

## 3. Validation steps

1. The sweep grid respects `K ≤ N`. The harness fails fast if asked
   to run otherwise.
2. The `firstphase_recall@N` column is reported for every row. A
   sweep that reports only `nDCG@K` cannot distinguish a retriever
   problem from a reranker problem.
3. The ablation against the `N=K` no-reranker baseline is present
   and shows a positive, statistically defensible delta on the
   held-out slice (bootstrap or paired t-test over per-query
   deltas).
4. The chosen operating point sits on the shoulder of the heat-map,
   not past the saturation point. "Saturation" here means that
   doubling `N` improves `nDCG@K` by less than the measurement
   noise.
5. Latency at the chosen point is inside the budget at p50 **and**
   p95.
6. The learner can name, for one query where the pipeline did
   worse at the higher `N`, the reason — typically "the reranker
   promoted a near-duplicate off a longer first-phase list," which
   is a cue for exercise-04 (dedup).
7. The configuration is pinned with the reranker model version and
   first-phase retriever version. Un-versioned configs reproduce
   non-deterministically.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Problem framing | The learner explicitly separates `N` (recall budget) from `K` (precision / prompt-token budget) and treats them as a joint decision. |
| Grid shape | Geometric spacing of `N` and `K`; grid respects `K ≤ N`. |
| Metric set | At minimum nDCG@K and first-phase recall@N. Ideally also MRR@K and recall@K. Latency reported at p50 and p95, not mean. |
| Ablation | Reports an `N=K` no-rerank baseline (same first phase, no reranker call) to isolate the reranker's contribution. |
| Held-out split | The reported numbers are on queries unseen during operating-point selection. |
| Saturation reasoning | The operating-point writeup names the point past which more `N` stops helping — that is the shoulder. |
| Latency account | p95 latency reported and compared against the budget. The operating point is inside the budget, not just "close." |
| Error analysis | At least one query-level case where the sweep made things worse is explained (near-duplicate promotion, judgment error, first-phase miss). |
| Config hygiene | The output YAML pins reranker version, retriever version, metrics, and the eval anchor. |
| Reproducibility | The sweep runs end to end on a fresh clone with judgments and model artifacts, no manual steps. |

Fail-fast smells:

- A sweep that reports only the mean metric across all queries. The
  tail of the distribution is where production surprises live.
- Saturation declared at `N=400` with no plot or table showing it —
  "I ran a sweep and 100 is best" without the numbers is not a
  recommendation.
- `K` set to 1 or 2 at the chosen operating point. For most RAG
  prompts, `K` in the 3–8 range is standard; `K=1` is a cue that
  either the reranker is extremely accurate or the exercise set
  `K` incorrectly.
- Latency reported in means. GPU rerankers are bimodal (cold vs.
  warm batch); p95 is the honest number.

## 5. Common mistakes

- **Tuning N with K fixed, then K with N fixed.** The two interact.
  The optimum of the joint surface is typically off the single-axis
  optima. Sweep the Cartesian grid.
- **Judging documents outside the deepest `N`.** The reranker never
  sees them. They do not teach the sweep anything about the
  operating point.
- **Judging binary (0/1) only.** Graded labels (0/1/2/3) are what
  nDCG was built for. Binary labels collapse the top-K ordering
  signal that distinguishes `K=1` from `K=5`.
- **Reporting only `nDCG@10` on a system that will serve `K=3`.**
  Measure at the operating point you intend to ship.
- **Ignoring `firstphase_recall@N`.** The ceiling of any reranker's
  nDCG@K is bounded by this number. If recall@N is 0.80, the
  reranker literally cannot deliver more than 0.80 recall@K.
  Increasing `N` has a different effect depending on which side of
  this ceiling you are on.
- **Letting the sweep run on the eval set that will report the
  final number.** The held-out slice is non-negotiable.
- **Sweeping without an ablation.** "nDCG improved" at a bigger `N`
  could be the reranker working or just more documents in the
  pool. The `N=K` no-rerank baseline tells you which.
- **Forgetting dedup and ordering.** A sweep that finds `N=100,
  K=10` optimal on nDCG@10 may hand ten chunks to the prompt where
  three pairs are near-duplicates. Exercise-04 cleans that up; the
  sweep operating point should anticipate it.
- **Exceeding context budget silently.** Picking `K=10` at long
  chunks blows the LLM's context window. The context-budget
  constraint belongs in the config, not as a wall-clock surprise in
  production.

## 6. References

- Reranking background:
  - Nogueira & Cho (2019), "Passage Re-ranking with BERT" —
    [arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085).
  - BAAI `bge-reranker-v2-m3` model card —
    [huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- Evaluation:
  - Järvelin & Kekäläinen (2002), "Cumulated gain-based evaluation
    of IR techniques" (NDCG) —
    [dl.acm.org/doi/10.1145/582415.582418](https://dl.acm.org/doi/10.1145/582415.582418).
  - BEIR benchmark — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - TREC qrels format — [trec.nist.gov/data/qrels_eng/](https://trec.nist.gov/data/qrels_eng/).
- First-phase anchor (upstream from this exercise):
  - Reciprocal Rank Fusion (Cormack et al., 2009) — the first-phase
    combiner from mod-106 whose output `N` this exercise sweeps.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
