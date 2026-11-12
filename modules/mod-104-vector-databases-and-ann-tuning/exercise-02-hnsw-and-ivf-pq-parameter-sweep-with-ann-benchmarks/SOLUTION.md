# Exercise 02 — Solution: HNSW and IVF-PQ parameter sweep with `ann-benchmarks` methodology

This is a reference solution for
`exercise-02-hnsw-and-ivf-pq-parameter-sweep-with-ann-benchmarks.md`. It is an
**implementation + measurement** exercise. The learner sweeps HNSW and IVF-PQ
parameters on a reproducible corpus, measures each configuration on the
`ann-benchmarks` axes — Recall@10 vs. queries-per-second — and draws a Pareto frontier
with a one-paragraph recommendation.

The numbers printed in the reference are **illustrative shapes**, not benchmark
publications. Exact QPS depends on CPU, BLAS build, batch size, and dataset. The
correctness of the exercise is in (i) using the published methodology, (ii) picking
a ground-truth recall baseline the right way, and (iii) drawing the frontier instead
of cherry-picking one point.

## 1. Solution overview

The deliverable has five parts:

1. **`src/indexes/`** — thin builders around FAISS `IndexHNSWFlat` and
   `IndexIVFPQ`, with reproducible seeds.
2. **`src/sweep/run_sweep.py`** — builds a Cartesian grid of configurations, runs
   each with a warmup, records Recall@10, Recall@100, build time, index-memory
   footprint, and p50/p95 latency plus QPS at batch=1.
3. **`src/sweep/ground_truth.py`** — exact brute-force top-K (over the full
   query set) using `IndexFlat*`. This is the recall denominator; without it the
   numbers are meaningless.
4. **`results/sweep.csv` + `docs/frontier.md`** — the raw rows plus the
   recall-vs-QPS scatter with the Pareto frontier marked and a
   one-paragraph pick.
5. **`notes/parameters.md`** — the parameter dictionary: one paragraph per knob
   explaining what it does and which axis of the frontier it moves.

The teaching point is that each family has two kinds of knob: **build-time** knobs
that are expensive to change (you have to rebuild the index) and **query-time**
knobs that are free to change per request. The sweep must vary both; a report that
only sweeps `ef_search` or only `nprobe` misses the shape.

## 2. Worked answer or implementation

### 2.1 Dataset choice and ground truth

Use one of the published `ann-benchmarks` datasets (SIFT-1M, GloVe-100-angular, or
MSMARCO-passage-dense) so the methodology is reproducible. SIFT-1M is the simplest
to start on because it is Euclidean, 128-dim, and 1M vectors. For a RAG-shaped run,
use a 1M-vector slice of your own corpus embedded with the mod-103 winner.

Ground truth is **exact** top-K from `IndexFlatIP` (for inner product / cosine with
pre-normalized vectors) or `IndexFlatL2`. You compute it once. Every ANN config's
Recall@K is measured against this same top-K list. If the ground truth is itself
approximate, the whole report is approximate.

```python
# src/sweep/ground_truth.py
"""Exact top-K baseline for a normalized (cosine) corpus."""
from __future__ import annotations
import pathlib
import numpy as np
import faiss

def build_flat(xb: np.ndarray, metric: str = "ip") -> faiss.Index:
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    index.add(xb.astype(np.float32, copy=False))
    return index

def compute_ground_truth(xb: np.ndarray, xq: np.ndarray, k: int = 100,
                         metric: str = "ip") -> np.ndarray:
    index = build_flat(xb, metric)
    _, I = index.search(xq.astype(np.float32, copy=False), k)
    return I  # (nq, k) int64

def save(path: pathlib.Path, I: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, I)
```

### 2.2 Index builders — HNSW and IVF-PQ

```python
# src/indexes/hnsw.py
"""faiss.IndexHNSWFlat — HNSW over raw fp32 vectors."""
from __future__ import annotations
import faiss, numpy as np

def build_hnsw(xb: np.ndarray, *, M: int, ef_construction: int,
               metric: str = "ip", seed: int = 1234) -> faiss.Index:
    d = xb.shape[1]
    metric_flag = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(d, M, metric_flag)
    # FAISS exposes HNSW parameters on `index.hnsw`.
    index.hnsw.efConstruction = ef_construction
    # Determinism: pin FAISS's thread count where it matters.
    faiss.omp_set_num_threads(1)  # single-thread build for reproducibility
    index.add(xb.astype(np.float32, copy=False))
    return index

def set_ef_search(index: faiss.Index, ef_search: int) -> None:
    index.hnsw.efSearch = ef_search
```

```python
# src/indexes/ivfpq.py
"""faiss.IndexIVFPQ — IVF partition + per-partition PQ codes."""
from __future__ import annotations
import faiss, numpy as np

def build_ivfpq(xb: np.ndarray, *, nlist: int, m_pq: int, nbits: int = 8,
                metric: str = "ip", train_size: int = 262_144,
                seed: int = 1234) -> faiss.Index:
    d = xb.shape[1]
    assert d % m_pq == 0, (d, m_pq)  # PQ requires d divisible by m_pq
    metric_flag = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, nbits, metric_flag)
    rs = np.random.default_rng(seed)
    train_idx = rs.choice(xb.shape[0], size=min(train_size, xb.shape[0]), replace=False)
    xt = xb[train_idx].astype(np.float32, copy=False)
    faiss.omp_set_num_threads(1)
    index.train(xt)
    index.add(xb.astype(np.float32, copy=False))
    return index

def set_nprobe(index: faiss.Index, nprobe: int) -> None:
    index.nprobe = nprobe
```

### 2.3 Sweep harness — `src/sweep/run_sweep.py`

```python
# src/sweep/run_sweep.py
"""Sweep HNSW and IVF-PQ configs. Emit recall / QPS / memory / build time per row.

Methodology follows ann-benchmarks:
  - fixed query set, K=10 (also report K=100 for shape)
  - single-thread search for comparability (FAISS OMP_NUM_THREADS=1)
  - warmup pass before timing (JIT caches / page in)
  - no per-run reseeding; ground truth computed once
"""
from __future__ import annotations
import argparse, csv, itertools, pathlib, statistics, time
import numpy as np, faiss
from src.indexes.hnsw import build_hnsw, set_ef_search
from src.indexes.ivfpq import build_ivfpq, set_nprobe
from src.sweep.ground_truth import compute_ground_truth

HNSW_M_GRID = (8, 16, 32, 48)
HNSW_EFC_GRID = (64, 128, 200, 400)
HNSW_EFS_GRID = (16, 32, 64, 128, 256)

IVFPQ_NLIST_GRID = (256, 1024, 4096)          # rule-of-thumb ~sqrt(N) … 4*sqrt(N)
IVFPQ_MPQ_GRID = (8, 16, 32)                  # must divide d
IVFPQ_NBITS = 8
IVFPQ_NPROBE_GRID = (1, 4, 16, 64, 256)

K = 10
K_WIDE = 100


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    nq = truth.shape[0]
    hits = 0
    for i in range(nq):
        t = set(truth[i, :k].tolist())
        p = set(pred[i, :k].tolist())
        hits += len(t & p)
    return hits / (nq * k)


def time_search(index: faiss.Index, xq: np.ndarray, k: int,
                warmup: int = 50, batch: int = 1) -> tuple[float, float, float, np.ndarray]:
    faiss.omp_set_num_threads(1)
    n = xq.shape[0]
    _ = index.search(xq[:warmup].astype(np.float32, copy=False), k)  # warmup
    lat_ms = []
    all_I = np.empty((n, k), dtype=np.int64)
    for i in range(0, n, batch):
        q = xq[i:i + batch].astype(np.float32, copy=False)
        t0 = time.perf_counter_ns()
        _, I = index.search(q, k)
        lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
        all_I[i:i + batch] = I
    p50 = float(statistics.median(lat_ms))
    p95 = float(sorted(lat_ms)[int(0.95 * len(lat_ms)) - 1])
    qps = 1000.0 / (sum(lat_ms) / len(lat_ms)) * batch
    return p50, p95, qps, all_I


def mem_bytes(index: faiss.Index) -> int:
    """Approx in-memory footprint. FAISS exposes `index.sa_code_size()` for codes.
    We serialize to a bytearray to get a deterministic size independent of allocator."""
    return len(faiss.serialize_index(index))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True,
                   help="Directory with xb.npy, xq.npy (float32, normalized if cosine).")
    p.add_argument("--metric", choices=("ip", "l2"), default="ip")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/sweep"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    xb = np.load(args.data / "xb.npy")
    xq = np.load(args.data / "xq.npy")
    print(f"xb={xb.shape}, xq={xq.shape}, metric={args.metric}")

    gt_path = args.out / "ground_truth.npy"
    if not gt_path.exists():
        print("computing exact top-100 ground truth ...")
        gt = compute_ground_truth(xb, xq, k=K_WIDE, metric=args.metric)
        np.save(gt_path, gt)
    gt = np.load(gt_path)

    rows = []

    # ---- HNSW sweep ----
    for M, efc in itertools.product(HNSW_M_GRID, HNSW_EFC_GRID):
        t0 = time.perf_counter()
        index = build_hnsw(xb, M=M, ef_construction=efc, metric=args.metric)
        build_s = time.perf_counter() - t0
        size_bytes = mem_bytes(index)
        for efs in HNSW_EFS_GRID:
            set_ef_search(index, efs)
            p50, p95, qps, I = time_search(index, xq, K_WIDE)
            rows.append({
                "family": "hnsw", "M": M, "efConstruction": efc, "efSearch": efs,
                "nlist": None, "m_pq": None, "nprobe": None,
                "recall_at_10": recall_at_k(I, gt, K),
                "recall_at_100": recall_at_k(I, gt, K_WIDE),
                "qps": qps, "p50_ms": p50, "p95_ms": p95,
                "build_s": build_s, "size_bytes": size_bytes,
            })

    # ---- IVF-PQ sweep ----
    for nlist, m_pq in itertools.product(IVFPQ_NLIST_GRID, IVFPQ_MPQ_GRID):
        if xb.shape[1] % m_pq != 0:
            continue
        t0 = time.perf_counter()
        index = build_ivfpq(xb, nlist=nlist, m_pq=m_pq, nbits=IVFPQ_NBITS,
                            metric=args.metric)
        build_s = time.perf_counter() - t0
        size_bytes = mem_bytes(index)
        for nprobe in IVFPQ_NPROBE_GRID:
            if nprobe > nlist:
                continue
            set_nprobe(index, nprobe)
            p50, p95, qps, I = time_search(index, xq, K_WIDE)
            rows.append({
                "family": "ivfpq", "M": None, "efConstruction": None, "efSearch": None,
                "nlist": nlist, "m_pq": m_pq, "nprobe": nprobe,
                "recall_at_10": recall_at_k(I, gt, K),
                "recall_at_100": recall_at_k(I, gt, K_WIDE),
                "qps": qps, "p50_ms": p50, "p95_ms": p95,
                "build_s": build_s, "size_bytes": size_bytes,
            })

    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out/'results.csv'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
```

### 2.4 Parameter dictionary — `notes/parameters.md`

**HNSW.**

- `M` (build time, costly to change): the number of bidirectional neighbors each
  graph node keeps. Larger `M` → denser graph → higher recall at a given
  `efSearch`, at the cost of more memory per vector and slower builds.
  Typical range: 8–48. Rough memory multiplier at 32-bit vectors: each neighbor
  is an 8-byte link, so a graph with `M=32` and `M*2` connections at layer 0
  adds ≈ `M*8 + M*2*8` ≈ 24–50 bytes per vector on top of the raw vector.
- `efConstruction` (build time, costly to change): the size of the dynamic
  candidate list during insertion. Larger `efConstruction` → better-quality
  graph edges at build time → higher asymptotic recall at a given `efSearch`,
  at the cost of much longer builds. Typical range: 64–400. There are
  diminishing returns past ~200.
- `efSearch` (query time, free to change): the size of the dynamic candidate
  list during search. Larger `efSearch` → more candidates considered → higher
  recall and higher latency. This is the main query-time knob.

**IVF-PQ.**

- `nlist` (build time, costly to change): number of coarse quantizer clusters.
  Rule of thumb from the FAISS wiki: start at `nlist ≈ 4 * sqrt(N)` for an
  N-vector corpus. More clusters → smaller per-cluster lists → cheaper probing
  per cluster, but you must probe more to maintain recall.
- `m_pq` (build time, costly to change): number of product-quantization
  subquantizers. Each subquantizer encodes `d / m_pq` dimensions in `nbits`
  bits. Must divide `d` exactly. Larger `m_pq` → less quantization loss →
  higher recall, at the cost of more storage per vector.
- `nbits` (build time): bits per subquantizer code. 8 is the standard. 4
  halves storage and loses non-trivial recall; 12 is rarely worth it.
  Storage per vector ≈ `m_pq * nbits / 8` bytes.
- `nprobe` (query time, free to change): how many of the `nlist` clusters to
  visit per query. The query-time knob equivalent to HNSW's `efSearch`. Larger
  `nprobe` → more candidates examined → higher recall, higher latency.

### 2.5 Reporting the frontier — `docs/frontier.md` template

```markdown
# Recall / QPS frontier — <corpus>, <metric>, K=10

Date: <YYYY-MM-DD>
Rows swept: <HNSW: 4×4×5 = 80, IVF-PQ: 3×3×5 = 45 (less skips)>
CPU / build: <lscpu one-liner + FAISS build info>
Threads: 1 (single-thread search for comparability)
Warmup: 50 queries per config
Ground truth: IndexFlatIP exact top-100 over the same query set

## Pareto frontier at Recall@10 ≥ 0.95

| Family | Config | Recall@10 | QPS (b=1) | p95 ms | Size (MB) | Build (s) |
|---|---|---|---|---|---|---|
| hnsw | M=16, efC=200, efS=64 | ... | ... | ... | ... | ... |
| hnsw | M=32, efC=200, efS=32 | ... | ... | ... | ... | ... |
| ivfpq | nlist=1024, m_pq=16, nprobe=16 | ... | ... | ... | ... | ... |
| ivfpq | nlist=4096, m_pq=32, nprobe=32 | ... | ... | ... | ... | ... |

## Recommendation

For the <corpus>, Recall@10 ≥ 0.95 target, single-node single-thread query path,
we pick `<family, exact config>`. The deciding factor was `<one of: index size,
build time, p95 latency, recall headroom under growth>`. Nearest rival is
`<row>`; we would switch if `<measured condition>`.
```

**Calibrated expectations** (so the reviewer can tell when a run went off the
rails). These are *shape* expectations, not prescribed numbers — the signs of
each effect, not the magnitudes.

- On HNSW: at fixed `M`, Recall@10 is monotone-nondecreasing in both
  `efConstruction` and `efSearch`. If increasing `efSearch` *lowers* recall, the
  run is broken (bad distance metric on normalized vectors is the usual
  culprit — use `IndexFlatIP` with unit-norm vectors, not `IndexFlatL2`).
- On HNSW: raising `M` from 8 to 48 roughly doubles the index memory footprint
  per vector. If the learner's table shows the same size at `M=8` and `M=48`,
  the serializer is lying or the sweep is measuring the wrong thing.
- On IVF-PQ: at fixed `nlist`, Recall@10 is monotone-nondecreasing in `nprobe`
  (up to brute-force when `nprobe = nlist`). Memory is dominated by `m_pq *
  nbits / 8` bytes per vector plus the PQ codebook.
- Between families: HNSW typically reaches higher *maximum* recall on cosine
  bi-encoder data but uses more memory per vector; IVF-PQ reaches lower
  recall ceilings but at much smaller memory (useful at 100M+ vectors).
- The frontier is not a single number. If the learner reports "HNSW is better"
  without naming the operating point, they have skipped the exercise.

## 3. Validation steps

1. `uv run python -m src.sweep.run_sweep --data data/sift1m --metric ip
   --out results/sweep` completes and writes `ground_truth.npy`,
   `results.csv`.
2. Every row's `recall_at_10 ∈ [0, 1]`, `recall_at_100 ≥ recall_at_10`, `qps > 0`,
   `p95_ms ≥ p50_ms`. If any row violates these, the sweep is broken.
3. For HNSW, fixing `M` and `efConstruction` and sweeping `efSearch` yields a
   monotone-nondecreasing recall column (small ties allowed).
4. For IVF-PQ, fixing `nlist, m_pq` and sweeping `nprobe` yields a
   monotone-nondecreasing recall column.
5. `nprobe = nlist` approaches brute-force recall (within ≈1% of 1.0) on IVF-Flat;
   IVF-PQ caps lower because PQ codes are lossy. Reporting "`nprobe=nlist` =
   100% recall" on IVF-PQ is wrong.
6. The Pareto frontier (points not dominated by any other) is a small subset of
   the rows — typically 6–12 points out of 100+. If every row is on the
   frontier, the sweep is too coarse.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Ground truth | Exact top-K from `IndexFlat*` over the same query set, saved to disk and reused. No approximate "proxy" baselines. |
| Sweep coverage | At least four `M` values × four `efConstruction` values × five `efSearch` values for HNSW; at least three `nlist` × three `m_pq` × five `nprobe` for IVF-PQ. |
| Methodology | Single-thread search, warmup pass, deterministic seeds, same query set for every config. One table per family plus a merged scatter. |
| Monotonicity | HNSW recall is monotone in `efSearch` at fixed `M, efConstruction`. IVF-PQ recall is monotone in `nprobe` at fixed `nlist, m_pq`. Deviations are explained (not swept under the rug). |
| Memory / build | Index memory (serialized size) and build time reported per config, not estimated. |
| Pareto frontier | Explicitly marked on the scatter or in a small table. Dominated rows called out so a reviewer can see what was discarded. |
| Recommendation | One row named, one deciding factor named, one escape condition named. The recommendation does not contradict the frontier. |
| Reproducibility | Hardware, FAISS version, thread count, batch size all recorded in `docs/frontier.md`. The next engineer can re-run and reproduce within noise. |

Fail-fast smells:

- `Recall@10` of 1.0 across most rows. This usually means the query set was a
  subset of the base set — leakage inflates recall. `ann-benchmarks` splits
  queries from base; mirror that.
- `efSearch` swept but `efConstruction` fixed at 40 (FAISS default). The learner
  has swept half the knob space.
- IVF-PQ without a training step or with training on the entire base set as a
  single sample. The IVF k-means must be trained on a *subset* and then `add()`
  the full set.
- HNSW parameters assumed symmetric between families ("same `M` as IVF's
  `nprobe`"). They are not comparable; the parameter dictionary should prevent
  this.
- Reporting QPS without naming batch size and thread count. The number is
  meaningless without both.

## 5. Common mistakes

- **Comparing families at different recall targets.** HNSW at Recall@10 = 0.98
  vs. IVF-PQ at Recall@10 = 0.85 is not a comparison. The frontier makes this
  honest: pick a recall target, then compare QPS at that target.
- **Measuring QPS at batch=100 for one family and batch=1 for the other.**
  FAISS batches very differently across families; HNSW scales near-linearly
  with batch, IVF-PQ less so. Pin the batch size.
- **Using `IndexFlatL2` for pre-normalized cosine data.** Mathematically equivalent
  but the FAISS kernel path is different and the metric that recall is measured
  against must match. Normalized vectors → `IndexFlatIP`.
- **Treating `efConstruction` as a free knob.** It is set at index build time.
  If you want to compare multiple `efConstruction` values, you build multiple
  indexes; you cannot flip it at query time.
- **Sweeping only one query-time knob.** `efSearch` and `nprobe` both move along
  the recall axis, but they move it against very different memory baselines.
  A recommendation that only knows one knob cannot pick a family.
- **Computing ground truth per config.** Ground truth is a property of the
  *corpus + query set*, not of the ANN config. Compute it once, save it, reuse
  it. Otherwise the recall denominators drift and columns are not comparable.
- **Reporting the mean of latency without p95.** p95 is the number the serving
  SLO is written against. Mean latency hides the tail where ANN gets expensive.
- **Picking a frontier point without a recall floor.** "This config has the
  highest QPS" is a trick answer at Recall@10 = 0.2. Set a recall floor first
  (0.9 or 0.95 is standard for first-stage retrieval), then optimize.
- **Ignoring build time.** A config that builds in 20 minutes vs. one that
  builds in 20 seconds matters for CI, for cutover windows (exercise-06), and
  for cost. The CSV must record it.

## 6. References

- Methodology:
  - `ann-benchmarks` project — [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks); results table — [ann-benchmarks.com](https://ann-benchmarks.com/).
  - Aumüller, Bernhardsson, Faithfull (2017), "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms" — [arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614).
- Algorithms:
  - Malkov & Yashunin (2016), "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (HNSW) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
  - Jégou, Douze, Schmid (2011), "Product Quantization for Nearest Neighbor Search" (IVF-PQ) — [ieeexplore.ieee.org/document/5432202](https://ieeexplore.ieee.org/document/5432202).
- Tooling:
  - FAISS repository and wiki — [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss), [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki).
  - FAISS "Guidelines to choose an index" — [github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).
  - SIFT-1M dataset — [corpus-texmex.irisa.fr/](http://corpus-texmex.irisa.fr/).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
