# Exercise 04 — Solution: Metadata filter pre- vs. post-filter correctness drill

This is a reference solution for
`exercise-04-metadata-filter-pre-vs-post-filter-correctness-drill.md`. It is an
**implementation + correctness** exercise. The learner builds a tiny but adversarial
dataset where naive post-filtering of ANN results silently drops valid matches,
runs the same query against three filter strategies (post-filter, pre-filter, and
filter-aware ANN walk), and shows in a table where each strategy returns the
wrong answer, the slow answer, or the right answer.

The teaching point: "metadata filter" is not a single feature. It is at least
three strategies with different correctness guarantees, and you must know which
one your vector DB is doing before you trust the result.

## 1. Solution overview

The deliverable has four parts:

1. **`src/corpus/gen_skewed_corpus.py`** — a synthetic 10k-vector corpus with a
   skewed metadata distribution: one small tenant (`tenant_id="minority"`, 1% of
   the corpus) whose vectors are pushed into a cluster that is close to the
   query vector. The majority tenants are also near the query. This is the
   adversarial shape that exposes post-filter failure.
2. **`src/filter/strategies.py`** — three strategies against the same corpus:
   `post_filter`, `pre_filter`, `filter_aware`. The last one uses the vector
   store's native filter-pushdown; for the reference we show it on Qdrant with
   a payload index, and the equivalent pgvector pattern (btree + HNSW on the
   filter column).
3. **`src/eval/run_correctness_drill.py`** — runs each strategy on a query set
   with mixed tenants, measures (i) Recall@10 against a brute-force filtered
   top-10 (the correctness denominator), (ii) p95 latency, (iii) the number of
   ANN candidates considered per query.
4. **`results/drill.csv` + `docs/drill.md`** — the table showing that
   `post_filter` has sub-1.0 recall under adversarial skew, `pre_filter` has
   1.0 recall but can be slow, and `filter_aware` has 1.0 recall at reasonable
   cost when the filter has a payload index.

The point is the correctness gap, not the latency gap. A fast wrong answer is
still wrong.

## 2. Worked answer or implementation

### 2.1 Why the three strategies exist

Three ways a vector DB can combine metadata predicates with ANN search:

1. **Post-filter.** Ask the ANN for top-K, then discard the hits that fail the
   filter. If 95 of your 100 top candidates belong to the majority tenant and
   only 5 belong to the minority tenant, you return 5 — even if the *true*
   top-10 for the minority tenant were elsewhere. You never saw them.
2. **Pre-filter.** Apply the filter first, then run exact (or re-indexed) top-K
   over the filtered subset. Correct, but with no ANN pushdown the retrieval
   becomes O(filtered-corpus), which is acceptable at small cardinalities and
   painful at large ones.
3. **Filter-aware ANN walk.** The vector DB knows the filter at index-walk time
   and prunes non-matching nodes during the HNSW traversal, often using a
   payload index to skip edges to non-matching nodes. Correct and fast as long
   as the filter is selective enough to keep the walk connected.

The exercise's teaching value is that strategy #1 is the default in several
popular stacks when the user does not know to ask for anything else — and it
is wrong on skewed data.

### 2.2 The adversarial corpus — `src/corpus/gen_skewed_corpus.py`

```python
# src/corpus/gen_skewed_corpus.py
"""Generate a 10k-vector corpus where naive post-filter will silently miss
minority-tenant hits.

Strategy:
  - 99% of vectors belong to `tenant_id in {"A","B","C","D"}`
  - 1% belong to `tenant_id="minority"`
  - Minority vectors are generated *closer to the query cluster* than most
    majority vectors — so the top-K ANN list will be dominated by the
    minority tenant IF you look wide enough, but at top-10 it's a mix.
  - We then ask: filter to `tenant_id="minority"` and return top-10.
    The TRUE top-10 are all minority.
"""
from __future__ import annotations
import pathlib, numpy as np, pyarrow as pa, pyarrow.parquet as pq

RNG = np.random.default_rng(1234)
D = 128
N_MAJ = 9900
N_MIN = 100
N = N_MAJ + N_MIN
TENANTS = ["A", "B", "C", "D"]

def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _sample_near(q: np.ndarray, n: int, jitter: float) -> np.ndarray:
    x = np.repeat(q[None, :], n, axis=0) + RNG.normal(scale=jitter, size=(n, D))
    return _unit(x.astype(np.float32))

def generate(out: pathlib.Path) -> None:
    q = _unit(RNG.normal(size=(D,)).astype(np.float32))
    # Minority cluster is tightly packed near the query (jitter = 0.4).
    xb_min = _sample_near(q, N_MIN, jitter=0.4)
    # Majority is more diffuse but still substantially near (jitter = 0.8).
    xb_maj = _sample_near(q, N_MAJ, jitter=0.8)
    xb = np.vstack([xb_min, xb_maj])

    tenant = (["minority"] * N_MIN) + [TENANTS[i % 4] for i in range(N_MAJ)]
    chunk_id = [f"c{i:05d}" for i in range(N)]
    doc_id   = [f"d{(i // 10):05d}" for i in range(N)]

    table = pa.table({
        "chunk_id":  pa.array(chunk_id),
        "doc_id":    pa.array(doc_id),
        "tenant_id": pa.array(tenant),
        "embedding": pa.array(xb.tolist(), type=pa.list_(pa.float32(), D)),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    np.save(out.with_suffix(".query.npy"), q)
```

**Why it works as a drill.** A single ANN call with top_K=10 and no filter
returns a mix of minority and majority points (majority outnumbers minority
99:1, even if minority is nearer on average). Post-filter to `minority` and
you keep only the ~1–3 minority hits that happened to show up; the other
~97 true minority top-10 points are not in the candidate set.

### 2.3 Three strategies — `src/filter/strategies.py`

For clarity the reference uses an in-memory FAISS index as the "ANN" primitive;
the same strategies translate directly to pgvector, Qdrant, and Milvus. Each
strategy returns the exact same shape — a list of `(chunk_id, score)` — so the
harness can compare apples to apples.

```python
# src/filter/strategies.py
"""Three filter-combination strategies against the same vector + metadata store."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, faiss

@dataclass
class Index:
    xb: np.ndarray            # (N, d) unit-norm
    chunk_ids: list[str]
    tenants: list[str]
    hnsw: faiss.Index         # HNSW over xb

    @classmethod
    def build(cls, xb, chunk_ids, tenants, M=16, efc=200) -> "Index":
        d = xb.shape[1]
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efc
        idx.add(xb.astype(np.float32, copy=False))
        return cls(xb, chunk_ids, tenants, idx)


def post_filter(ix: Index, q: np.ndarray, *, tenant: str, k: int = 10,
                ef_search: int = 32) -> tuple[list[tuple[str, float]], int]:
    """Ask ANN for top-K (same K as requested), then drop non-matching rows.

    Candidates considered: `k`. Famous failure mode.
    """
    ix.hnsw.hnsw.efSearch = ef_search
    _, ids = ix.hnsw.search(q.astype(np.float32).reshape(1, -1), k)
    hits = []
    for i in ids[0].tolist():
        if i < 0: continue
        if ix.tenants[i] == tenant:
            hits.append((ix.chunk_ids[i], float(ix.xb[i] @ q)))
    return hits, k


def post_filter_over_fetch(ix: Index, q: np.ndarray, *, tenant: str, k: int = 10,
                           overfetch: int = 200,
                           ef_search: int = 256) -> tuple[list[tuple[str, float]], int]:
    """Still post-filter, but fetch `overfetch` candidates then filter down.

    This is the common 'fix' that masks the correctness gap at moderate skew
    but does not close it. Recall is a function of `overfetch`; at extreme
    minority ratios you still miss hits.
    """
    ix.hnsw.hnsw.efSearch = ef_search
    _, ids = ix.hnsw.search(q.astype(np.float32).reshape(1, -1), overfetch)
    hits = []
    for i in ids[0].tolist():
        if i < 0: continue
        if ix.tenants[i] == tenant:
            hits.append((ix.chunk_ids[i], float(ix.xb[i] @ q)))
            if len(hits) == k: break
    return hits, overfetch


def pre_filter(ix: Index, q: np.ndarray, *, tenant: str, k: int = 10) \
        -> tuple[list[tuple[str, float]], int]:
    """Filter the base set, then run EXACT top-K over the survivors.

    Correct by construction. Cost = O(filtered-size). Fine when the filtered
    subset is small; painful when it's not.
    """
    mask = np.array([t == tenant for t in ix.tenants], dtype=bool)
    sub = ix.xb[mask]
    if sub.shape[0] == 0: return [], 0
    scores = sub @ q
    top = np.argsort(-scores)[:k]
    sub_ids = [cid for cid, ok in zip(ix.chunk_ids, mask) if ok]
    return [(sub_ids[i], float(scores[i])) for i in top.tolist()], int(mask.sum())


def filter_aware_ann(ix: Index, q: np.ndarray, *, tenant: str, k: int = 10,
                     ef_search: int = 128) -> tuple[list[tuple[str, float]], int]:
    """Filter-aware HNSW walk — reference-equivalent of Qdrant payload-filtered
    search or pgvector's planner choosing a filtered scan.

    The reference emulates it by restricting the HNSW's traversal to matching
    ids via FAISS's IDSelector. In production, this is done by:
      - Qdrant: payload index + filter, native pushdown during graph walk;
      - Milvus: scalar index + `expr=`, bitmap-pruned walk;
      - pgvector: HNSW index + `WHERE tenant_id=...`, planner selects
        Index Scan with a filter pass OR IndexOnly + recheck depending on
        selectivity.
    """
    ix.hnsw.hnsw.efSearch = ef_search
    allowed = np.array([i for i, t in enumerate(ix.tenants) if t == tenant],
                       dtype=np.int64)
    sel = faiss.IDSelectorArray(allowed.size, faiss.swig_ptr(allowed))
    params = faiss.SearchParametersHNSW()
    params.sel = sel
    params.efSearch = ef_search
    _, ids = ix.hnsw.search(q.astype(np.float32).reshape(1, -1), k, params=params)
    hits = []
    for i in ids[0].tolist():
        if i < 0: continue
        hits.append((ix.chunk_ids[i], float(ix.xb[i] @ q)))
    # 'candidates considered' is approximately ef_search for a filter-aware walk,
    # not the full filtered subset — this is why it's cheap.
    return hits, ef_search
```

A note on the reference's use of `faiss.IDSelectorArray`: FAISS exposes an
`SearchParametersHNSW.sel` hook that approximates the "filter-aware HNSW
walk" that Qdrant and Milvus implement natively with payload / scalar
indexes. For pgvector the equivalent is a `WHERE` clause combined with a
Postgres planner choice — the planner may decide that for a very selective
filter it is cheaper to scan the filter's btree and brute-force-score the
survivors than to use the HNSW index at all. That is still correct behavior;
the HNSW index is just not the right tool at that selectivity.

### 2.4 Harness — `src/eval/run_correctness_drill.py`

```python
# src/eval/run_correctness_drill.py
"""Run the three strategies on a query set; emit recall and candidate counts.

Ground truth = exact top-10 over the FILTERED subset. That is the only correct
denominator for a filtered query.
"""
from __future__ import annotations
import argparse, csv, pathlib, time, statistics
import numpy as np, pyarrow.parquet as pq

from src.filter.strategies import Index, post_filter, post_filter_over_fetch, pre_filter, filter_aware_ann


def ground_truth(ix: Index, q, *, tenant, k=10) -> list[str]:
    hits, _ = pre_filter(ix, q, tenant=tenant, k=k)
    return [cid for cid, _ in hits]


def recall(got: list[str], truth: list[str], k: int) -> float:
    t = set(truth[:k]); g = set(got[:k])
    return len(t & g) / max(1, len(t))


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True)   # parquet with corpus
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--tenant", default="minority")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--overfetch", type=int, default=200)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/drill"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    tb = pq.read_table(args.data).to_pandas()
    xb = np.array(tb["embedding"].tolist(), dtype=np.float32)
    chunk_ids = tb["chunk_id"].tolist()
    tenants = tb["tenant_id"].tolist()
    q_base = np.load(args.data.with_suffix(".query.npy")).astype(np.float32)

    # Generate `args.queries` nearby query variants to average over.
    rs = np.random.default_rng(42)
    Q = q_base + 0.05 * rs.normal(size=(args.queries, q_base.shape[0])).astype(np.float32)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    ix = Index.build(xb, chunk_ids, tenants)

    rows = []
    for strategy_name, strategy in [
        ("post_filter",            lambda q: post_filter(ix, q, tenant=args.tenant, k=args.k)),
        ("post_filter_overfetch",  lambda q: post_filter_over_fetch(ix, q, tenant=args.tenant, k=args.k, overfetch=args.overfetch)),
        ("filter_aware",           lambda q: filter_aware_ann(ix, q, tenant=args.tenant, k=args.k)),
        ("pre_filter",             lambda q: pre_filter(ix, q, tenant=args.tenant, k=args.k)),
    ]:
        rec, cand, lat = [], [], []
        for q in Q:
            truth = ground_truth(ix, q, tenant=args.tenant, k=args.k)
            t0 = time.perf_counter_ns()
            hits, n_cand = strategy(q)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            cand.append(n_cand)
            rec.append(recall([c for c, _ in hits], truth, args.k))
        rows.append({
            "strategy": strategy_name,
            "recall_at_10": statistics.mean(rec),
            "candidates_p50": int(statistics.median(cand)),
            "p50_ms": round(statistics.median(lat), 3),
            "p95_ms": round(sorted(lat)[int(0.95 * len(lat)) - 1], 3),
        })

    with (args.out / "drill.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(rows)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main(sys.argv[1:]))
```

### 2.5 Reporting — `docs/drill.md`

```markdown
# Metadata filter correctness drill — minority tenant (1% of corpus)

Date: <YYYY-MM-DD>
Corpus: 10,000 synthetic 128-dim unit vectors. 100 vectors have
`tenant_id="minority"`; 9,900 have `tenant_id ∈ {A,B,C,D}`.
Minority vectors are more tightly clustered near the query than the majority.

Queries: 100 small variants of a fixed query vector.
Filter: `tenant_id = "minority"`, K=10, HNSW M=16, efConstruction=200.
Ground truth: exact top-10 over the filtered (minority) subset.

## Results (expected shape)

| Strategy | Recall@10 | Candidates / query | p50 ms | p95 ms |
|---|---|---|---|---|
| `post_filter` (fetch 10, drop) | ~0.05–0.15 | 10 | ... | ... |
| `post_filter_overfetch` (fetch 200) | ~0.6–0.9 | 200 | ... | ... |
| `filter_aware` (HNSW walk with id selector / payload index) | ~1.0 | ≈ ef_search | ... | ... |
| `pre_filter` (exact top-K over minority subset) | 1.0 | 100 (filter size) | ... | ... |

## Observation

Post-filter is **not just slower** at this skew — it returns the wrong answer.
Over-fetching narrows the gap but does not close it; the only safe strategies
on adversarial skew are pre-filter (when the filtered set is small) and
filter-aware walk (when the vector DB supports it).

## Recommendation

For queries where the filter cardinality is low (minority tenants, rare
metadata, small date ranges), either:
- use **pre-filter** (SQL `WHERE` + brute-force top-K on the survivors) when
  the filtered set is small enough for brute force, or
- use **filter-aware ANN** (payload / scalar index + filter pushdown) when
  the filtered set is large but still a small fraction of the total.
Post-filter with over-fetch is acceptable **only** when the filter is close
to the entire corpus (>50% selectivity) and the recall contract tolerates
occasional misses.
```

### 2.6 Store-by-store filter strategy map

| Store | What the strategy names map to |
|---|---|
| **pgvector** | `WHERE col = 'x' ORDER BY embedding <=> $1 LIMIT 10` lets the planner pick: (1) HNSW scan then recheck (post-filter in effect) when the filter is non-selective, or (2) btree scan then exact top-K when the filter is very selective. Setting `hnsw.ef_search` higher helps the first path; adding a btree on the filter column helps the planner choose the right path. There is no third "filter-aware HNSW walk" in pgvector today. |
| **Qdrant** | Payload indexes (`create_payload_index(...)`) enable the filter-aware HNSW walk. Without them, Qdrant falls back to a post-filter over the ANN result. Always create payload indexes on your filter keys. |
| **Milvus** | Scalar indexes plus a filter `expr=` push the predicate into the walk. Without a scalar index Milvus will still filter correctly but may do a bitmap pre-filter + scan — slower but correct. |
| **Elasticsearch / OpenSearch** | The `knn` query clause combined with a `bool filter` applies the filter during the vector walk when the field is indexed. Historically ES had a separate "pre-filter" vs. "post-filter" option; recent versions do filter-aware kNN. |

The learner's job is to know which strategy the DB they picked is actually
running — and to pay for the payload / scalar / btree index that turns
post-filter into filter-aware.

## 3. Validation steps

1. `uv run python -m src.corpus.gen_skewed_corpus --out data/skew.parquet` writes
   `data/skew.parquet` and `data/skew.query.npy`.
2. `uv run python -m src.eval.run_correctness_drill --data data/skew.parquet` prints
   four rows. `post_filter`'s `recall_at_10` is noticeably < 1.0 (typical shape:
   0.05–0.15). `pre_filter`'s `recall_at_10` is exactly 1.0. `filter_aware`'s
   `recall_at_10` is ~1.0 at `ef_search=128` or higher.
3. The `candidates_p50` column is monotone across strategies: 10 (post_filter)
   < ef_search (filter_aware) < filter_size (pre_filter) — unless the filter
   set is smaller than ef_search, in which case filter_aware ≈ filter_size.
4. Running the same drill against pgvector with a `WHERE tenant_id='minority'`
   query reproduces the ordering (post-filter recall gap closes only when the
   planner switches to btree-scan + brute-force top-K — observable with
   `EXPLAIN (ANALYZE, BUFFERS)`).
5. Running the drill against Qdrant with and without `create_payload_index` on
   `tenant_id` reproduces the "filter-aware vs. post-filter" distinction
   observable from the vendor's defaults.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Adversarial corpus | Minority tenant is a small fraction (1–5%) and clustered near the query. Without this shape, post-filter looks fine. |
| Three strategies | `post_filter`, (optional) `post_filter_overfetch`, `filter_aware`, `pre_filter` all implemented and share a signature. |
| Ground truth | Truth is the exact top-K over the filtered subset, not the unfiltered top-K. |
| Correctness gap | The report shows `post_filter` Recall@10 < 1.0 at an adversarial skew. If the gap is zero, the corpus is not adversarial enough. |
| Over-fetch is honest | `post_filter_overfetch` is reported alongside the naive version; the write-up acknowledges it is a mitigation, not a fix. |
| Store mapping | The learner names which of their store's features corresponds to each strategy (pgvector planner / Qdrant payload index / Milvus scalar index / ES `knn` filter). |
| Candidate counts | Each strategy reports how many points it considered per query. Reviewers can tell the latency/correctness trade is structural, not accidental. |
| Recommendation | Explicitly names when to prefer pre-filter over filter-aware ANN (small filter set, exact top-K) and vice versa. |

Fail-fast smells:

- All four strategies report Recall@10 = 1.0. The corpus is not adversarial;
  either the minority cluster is far from the query or the majority ratio is
  too low.
- `post_filter_overfetch` Recall@10 = 1.0 at `overfetch=200`. If the minority
  subset size is 100, fetching 200 unfiltered candidates can include all of
  them — but only because the minority is so dense near the query that the
  over-fetch exhausts it. On a less adversarial configuration the mitigation
  will not look this good.
- `filter_aware` Recall@10 is below `pre_filter`. The emulation is wrong or
  `ef_search` is too low. Raise it to 256 and re-check.
- The drill is run against unit-norm vectors with FAISS `METRIC_L2` instead of
  `METRIC_INNER_PRODUCT`. Scores and rankings diverge subtly from a
  cosine store and the recall numbers become implementation noise.
- The learner reports only latency, not Recall@10. Without the correctness
  column the whole point of the exercise is lost.

## 5. Common mistakes

- **Using the unfiltered top-K as ground truth.** Recall@K for a *filtered* query
  is measured against the exact top-K of the filtered subset, not against the
  unfiltered top-K. Mixing those denominators makes post-filter look correct.
- **Treating post-filter + over-fetch as "the same" as filter-aware.** It is not.
  Over-fetch narrows the gap at moderate selectivity, then re-opens it at low
  selectivity. Filter-aware walks have no such drop-off.
- **Forgetting the payload / scalar index.** In Qdrant and Milvus, filter-aware
  walks require the DB to know ahead of time which points match which filter
  value. That requires a payload or scalar index on the filter column.
  Without it, the "filter-aware" call path quietly falls back to post-filter.
- **Confusing "pgvector supports filtering" with "pgvector has filter-aware
  HNSW."** Pgvector supports SQL `WHERE`. Depending on selectivity the planner
  picks HNSW+recheck (post-filter) or btree+brute-force-score (pre-filter).
  There is no graph-level filter walk today.
- **Testing only one filter cardinality.** The correctness vs. latency trade
  flips as selectivity changes. The reference tests at ~1% selectivity; the
  learner should also test at 50% and 100% to see how the trade moves.
- **Using the same query for ground truth and for the strategy call.** They
  must share the query. But *also* share the HNSW config (M, ef_construction,
  ef_search) — otherwise filter_aware runs at a different `ef_search` than
  post_filter and the comparison is confounded.
- **Reading one row as a universal verdict.** "Post-filter is wrong" is only
  honest at adversarial selectivity. At 100% selectivity post-filter is
  correct and cheap. The drill is not an indictment of the strategy; it is a
  demonstration of where the strategy fails.

## 6. References

- Pgvector filtering docs — [github.com/pgvector/pgvector#querying](https://github.com/pgvector/pgvector#querying) (note: planner behavior changes with selectivity; read `EXPLAIN` on your data).
- Qdrant filtering with payload index — [qdrant.tech/documentation/concepts/filtering/](https://qdrant.tech/documentation/concepts/filtering/).
- Milvus scalar indexes and `expr=` filtering — [milvus.io/docs/scalar_index.md](https://milvus.io/docs/scalar_index.md).
- Elasticsearch `knn` with filter — [elastic.co/guide/en/elasticsearch/reference/current/knn-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
- OpenSearch k-NN with pre/post filter — [opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/](https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/).
- FAISS `SearchParametersHNSW` and `IDSelector` — [github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization](https://github.com/facebookresearch/faiss/wiki) (filter-aware search hooks).
- Risk / governance framings consulted while drafting the recommendation:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
