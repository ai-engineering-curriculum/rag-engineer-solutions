# Exercise 05 — Solution: Index sizing — memory / latency budget worksheet

This is a reference solution for
`exercise-05-index-sizing-memory-latency-budget-worksheet.md`. It is a
**design + calculation** exercise. The learner fills a budget worksheet for an
HNSW and an IVF-PQ index at a given corpus size, dimension, and recall target,
estimates memory footprint and query latency from first principles, and compares
the estimate to a measurement taken from the exercise-02 sweep.

The teaching point: you should be able to predict, inside an order of magnitude,
how much RAM an index will need and how long a query will take **before** you
provision anything. The DB vendor's sizing calculator is a good cross-check, not
the primary source of truth.

## 1. Solution overview

The deliverable is a short worksheet plus a backing spreadsheet (or a tiny Python
script), not a full application. The learner fills:

1. **`notes/inputs.md`** — the inputs: N (corpus size), d (vector dimension),
   metric, recall target, QPS target, number of replicas, memory budget.
2. **`notes/hnsw-sizing.md`** — first-principles HNSW memory and latency
   estimate for the inputs; compares to pgvector's observed overhead and
   to exercise-02's measured numbers.
3. **`notes/ivfpq-sizing.md`** — first-principles IVF-PQ memory and latency
   estimate for the same inputs; call out which recall band IVF-PQ can hit
   and when it becomes the right choice.
4. **`notes/budget.md`** — the single-page summary: estimated RAM per node,
   estimated p95 latency, estimated build time, cost line for one month of
   steady-state (RAM-hours), and the verdict ("fits / does not fit").
5. **`src/sizing/calc.py`** — a tiny Python module that reproduces the
   estimates from a config dict. One function per formula.

The point is to *work the formulas by hand* once, then put them in a script. A
worksheet with only numbers in it is memorization; a worksheet with named
quantities and sources is engineering.

## 2. Worked answer or implementation

### 2.1 Inputs — `notes/inputs.md`

A concrete reference baseline:

- **N** = 10,000,000 vectors
- **d** = 384 dimensions (mod-103 winner is `BAAI/bge-small-en-v1.5` or similar)
- **Metric** = cosine (equivalently, inner product on unit vectors)
- **Scalar type** = float32 at rest; evaluate halving to float16 (pgvector
  `halfvec`) and int8 scalar quantization later.
- **Recall target** = Recall@10 ≥ 0.95 at first stage.
- **QPS target** = 20 QPS sustained at p95 < 50 ms on a single replica.
- **Replicas** = 2 (one active, one for cutover; see exercise-06).
- **Memory budget per node** = 32 GB.

Any change in these inputs changes the sizing. The worksheet lives or dies on
the inputs being explicit.

### 2.2 HNSW sizing — `notes/hnsw-sizing.md`

#### Vector storage

Raw vectors at float32 cost `N * d * 4` bytes.

```
N * d * 4 = 10_000_000 * 384 * 4 bytes
          = 15,360,000,000 bytes
          ≈ 15.36 GB
```

With float16 (`halfvec` in pgvector, scalar-fp16 in Qdrant/Milvus):
`N * d * 2 = 7.68 GB`.

With int8 scalar quantization: `N * d * 1 ≈ 3.84 GB` plus negligible overhead
for per-dim scale/zero.

#### Graph storage

An HNSW graph has one layer-0 adjacency list per node plus a thin upper layer.
Layer 0 has up to `2 * M` edges per node; upper layers add a geometrically
decaying amount. A standard approximation used in FAISS and hnswlib docs is:

```
bytes_per_node ≈ (2 * M + M * p) * 8
```

where `p` is the probability a node also has upper-layer edges (≈ 1/ln(M) in the
original HNSW formulation). At `M=16, p ≈ 0.36`, that comes out to:

```
(32 + 5.8) * 8 ≈ 303 bytes / node
```

So the graph for `M=16, N=10M` is about:

```
10_000_000 * 303 ≈ 3.03 GB
```

A reasonable rule of thumb: **HNSW graph adds 20–50% on top of the raw vector
bytes at M in `{16, 32}`.** For M=32 the graph roughly doubles (≈ 6 GB) and the
total matches what you see in practice.

#### Total HNSW memory

```
HNSW (fp32, M=16)  ≈ 15.36 (vecs) + 3.03 (graph) ≈ 18.4 GB
HNSW (fp16, M=16)  ≈ 7.68 + 3.03                  ≈ 10.7 GB
HNSW (fp32, M=32)  ≈ 15.36 + 6.0                  ≈ 21.4 GB
HNSW (int8, M=16)  ≈ 3.84 + 3.03                  ≈ 6.9 GB
```

Vector DBs add a bookkeeping overhead on top (payload storage, ids,
per-tenant metadata, caches). Observed in practice:

- **pgvector**: HNSW index memory ≈ graph bytes (above) + the vectors kept in
  the table (also roughly `N * d * 4`, since the index does not duplicate them).
  Toast/alignment overhead pushes the table bytes 10–20% higher than the raw
  calculation.
- **Qdrant**: segment files on disk include vectors + graph + payload index;
  working-set RAM at steady-state is usually similar to the computed total
  plus the payload indexes.
- **Milvus**: `IndexHNSWFlat` on a loaded collection consumes roughly the
  same total; load-time RAM can be higher due to a staged load buffer.

#### Query latency

HNSW query latency at batch=1 is dominated by:

1. One descent through the upper layers (very small, ~log N vector ops).
2. Layer-0 traversal with `efSearch` candidates in the priority queue.

Each candidate costs one distance computation (`d` floating-point multiply-adds)
plus a handful of pointer / queue operations. A reasonable bound:

```
distances_per_query ≈ efSearch * (2 * M + small-constant)
flops_per_query     ≈ distances_per_query * 2 * d       (dot product)
```

For `efSearch=64, M=16, d=384`:

```
distances_per_query ≈ 64 * 36 ≈ 2300
flops_per_query     ≈ 2300 * 768 ≈ 1.77 million FLOPs
```

A modern x86 core at ~20 GFLOPS single-thread for this access pattern runs
this in ~90 microseconds *excluding* memory-bound effects. Real-world p95 for
HNSW on this shape is dominated by cache misses and lands in the
0.5–2 ms range at batch=1 on a single core. Exercise-02's sweep provides the
measured number for a specific CPU; the estimate above sets the expectation.

#### Build time

HNSW build time scales roughly as `N * log(N) * efConstruction / cores`. A
rule of thumb from `ann-benchmarks` reports on SIFT-1M at `M=16,
efConstruction=200` is ~minutes on a single core, ~tens of seconds on an 8-core
machine. For `N=10M` and the same parameters, expect ~10–30 minutes on 8 cores.
pgvector's build is single-session CPU-bound and often slower than FAISS's
build at the same parameters because of Postgres-side bookkeeping; expect
tens of minutes to a few hours at 10M × 384-dim on commodity hardware.

### 2.3 IVF-PQ sizing — `notes/ivfpq-sizing.md`

IVF-PQ gives up some recall for a much smaller memory footprint. The sweet
spot is in the 100M–1B vector regime; at 10M the memory saving is already
noticeable but HNSW is also fine.

#### Memory

```
nlist     : number of coarse clusters         (build-time)
m_pq      : number of PQ subquantizers        (build-time)
nbits     : bits per subquantizer code        (build-time; default 8)
```

- Coarse quantizer (IVF centroids): `nlist * d * 4` bytes. For
  `nlist=4096, d=384`: 4096 × 384 × 4 ≈ 6 MB. Negligible.
- PQ codebook: `m_pq * 2^nbits * (d / m_pq) * 4` bytes. For
  `m_pq=32, nbits=8, d=384`: 32 × 256 × 12 × 4 ≈ 393 KB. Negligible.
- PQ codes per vector: `m_pq * nbits / 8` bytes. For `m_pq=32, nbits=8`:
  32 bytes per vector.

```
IVF-PQ total ≈ N * (m_pq * nbits / 8) + nlist * d * 4 + codebook
             ≈ 10_000_000 * 32 + 6 MB + ~400 KB
             ≈ 320 MB
```

That is ~50× smaller than HNSW fp32. The cost is recall: IVF-PQ at the
example parameters typically caps around Recall@10 = 0.90–0.95 on cosine
bi-encoder data without a rescore stage. With a float32 rescore stage over
the top-200 candidates, recall rises to within 1 point of brute force at
some additional latency and the storage of the full fp32 vectors (or fp16)
alongside the codes — cancelling part of the memory saving.

#### Query latency

- Look up query against coarse centroids: `d * 4 * nlist` dot products to pick
  top `nprobe` lists (actually less — most libraries use a short-list / tree).
- For each probed list: compute an asymmetric distance from the query to each
  code. Fast (table-lookup + SIMD), roughly `(N / nlist) * nprobe` candidate
  distances per query.

For `N=10M, nlist=4096, nprobe=32`, that's about `10M / 4096 * 32 ≈ 78_000`
candidate distances per query. At modern SIMD table-lookup rates (hundreds of
millions of PQ-code distances per core per second), this runs in roughly
0.3–1 ms at batch=1. Observed numbers on `ann-benchmarks` IVF-PQ rows are
in that range for 128-dim data; for 384-dim expect the upper end.

#### Build time

IVF-PQ needs a k-means pass on a *training subset* (usually `max(256 * nlist,
~100K)` vectors), then one add pass over the full corpus. The k-means
dominates at large `nlist`. For `nlist=4096` the training set is ~1M vectors
and training takes minutes on a single CPU; the add pass is fast.

### 2.4 Replicas and QPS

Both HNSW and IVF-PQ scale replicated reads linearly with the number of
replicas. At `QPS_target = 20, replicas = 2`, each replica needs to sustain
10 QPS at p95 < 50 ms on a single query path. At the latencies estimated
above (0.5–2 ms for HNSW, 0.3–1 ms for IVF-PQ), you have enormous margin at
this QPS — the budget is dominated by memory, not CPU.

### 2.5 `notes/budget.md` — the single-page summary

```markdown
# Index sizing budget — mod-104 baseline

Inputs: N=10M, d=384, cosine, Recall@10 ≥ 0.95, QPS=20, 2 replicas, 32 GB/node.

## HNSW (M=16, efConstruction=200, efSearch=64)
- Raw vectors (fp32):    15.36 GB
- Raw vectors (fp16):     7.68 GB
- HNSW graph:             3.03 GB
- Total (fp32):          18.4 GB   <- fits in 32 GB node with headroom
- Total (fp16):          10.7 GB   <- comfortable for a 32 GB node
- Estimated p95 latency: ~1 ms at batch=1 (measure with exercise-02 sweep)
- Build time (8 cores):  ~10–30 min FAISS; ~hours for pgvector

## IVF-PQ (nlist=4096, m_pq=32, nbits=8, nprobe=32)
- PQ codes:               0.32 GB
- Centroids+codebook:     ~6.4 MB
- Total:                  ~0.33 GB   <- essentially free
- + fp32 rescore vectors: 15.36 GB   (optional, for Recall@10 > 0.92)
- Estimated p95 latency: ~0.5–1 ms at batch=1, plus rescore time if used
- Expected recall ceiling: ~0.90–0.95 without rescore; ~0.98 with rescore
- Build time (8 cores):  training ~minutes, add ~minutes

## Verdict
- Primary recommendation: HNSW fp16 (10.7 GB) on each of the 2 replicas.
  Fits in 32 GB with ample headroom for payloads, caches, and another
  index. Build time is within the operational cutover window.
- IVF-PQ becomes the right default at N ≥ ~100M where fp32 vectors
  no longer fit. For this 10M baseline it is a cost optimization at the
  price of recall; not needed yet.
- Trigger for switching to IVF-PQ: corpus grows past ~50M at 384-dim
  (fp32 crosses the single-node RAM budget) OR we need to serve from
  disk (DiskANN / Milvus-DiskANN territory).
```

### 2.6 `src/sizing/calc.py` — the formulas in code

```python
# src/sizing/calc.py
"""First-principles sizing. One function per formula. No vendor sdk needed."""
from __future__ import annotations
import math
from dataclasses import dataclass

BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


@dataclass
class HnswInputs:
    N: int
    d: int
    M: int
    scalar: str = "fp32"       # fp32 | fp16 | int8
    ef_construction: int = 200


def hnsw_vector_bytes(h: HnswInputs) -> int:
    return h.N * h.d * BYTES[h.scalar]


def hnsw_graph_bytes(h: HnswInputs, link_bytes: int = 8) -> int:
    # Layer-0 adjacency plus upper-layer adjacency approximation.
    p_upper = 1.0 / math.log(max(h.M, 2))
    per_node = (2 * h.M + h.M * p_upper) * link_bytes
    return int(h.N * per_node)


def hnsw_total_bytes(h: HnswInputs) -> int:
    return hnsw_vector_bytes(h) + hnsw_graph_bytes(h)


@dataclass
class IvfPqInputs:
    N: int
    d: int
    nlist: int
    m_pq: int
    nbits: int = 8


def ivfpq_code_bytes(ip: IvfPqInputs) -> int:
    return ip.N * (ip.m_pq * ip.nbits // 8)


def ivfpq_centroid_bytes(ip: IvfPqInputs) -> int:
    return ip.nlist * ip.d * 4  # centroids stay fp32 in most implementations


def ivfpq_codebook_bytes(ip: IvfPqInputs) -> int:
    assert ip.d % ip.m_pq == 0
    sub_d = ip.d // ip.m_pq
    return ip.m_pq * (1 << ip.nbits) * sub_d * 4


def ivfpq_total_bytes(ip: IvfPqInputs) -> int:
    return ivfpq_code_bytes(ip) + ivfpq_centroid_bytes(ip) + ivfpq_codebook_bytes(ip)


def gb(n_bytes: int) -> float:
    return n_bytes / (1024 ** 3)


def summarize(hnsw_conf: HnswInputs, ivf_conf: IvfPqInputs) -> dict:
    return {
        "hnsw_vectors_gb":  round(gb(hnsw_vector_bytes(hnsw_conf)), 2),
        "hnsw_graph_gb":    round(gb(hnsw_graph_bytes(hnsw_conf)), 2),
        "hnsw_total_gb":    round(gb(hnsw_total_bytes(hnsw_conf)), 2),
        "ivfpq_codes_gb":   round(gb(ivfpq_code_bytes(ivf_conf)), 3),
        "ivfpq_overhead_mb": round(
            (ivfpq_centroid_bytes(ivf_conf) + ivfpq_codebook_bytes(ivf_conf)) / (1024 ** 2), 2),
        "ivfpq_total_gb":   round(gb(ivfpq_total_bytes(ivf_conf)), 3),
    }


if __name__ == "__main__":
    import json
    hnsw = HnswInputs(N=10_000_000, d=384, M=16, scalar="fp32")
    ivf  = IvfPqInputs(N=10_000_000, d=384, nlist=4096, m_pq=32, nbits=8)
    print(json.dumps(summarize(hnsw, ivf), indent=2))
```

## 3. Validation steps

1. `uv run python -m src.sizing.calc` prints a JSON blob with the fields
   above. The numbers match the values in `notes/hnsw-sizing.md` and
   `notes/ivfpq-sizing.md` to two decimal places in GB.
2. Switching `scalar="fp16"` roughly halves `hnsw_total_gb`. Switching to
   `int8` roughly quarters the vector bytes (graph bytes unchanged).
3. Doubling `M` roughly doubles `hnsw_graph_gb` and leaves
   `hnsw_vectors_gb` unchanged.
4. Doubling `m_pq` roughly doubles `ivfpq_codes_gb` and leaves the centroid /
   codebook overhead negligible.
5. The estimate matches the exercise-02 measured serialized-index size to
   within ±20% on the corresponding row. If it does not, either the measurement
   is wrong or the formulas have omitted a field (payload, id table, etc.) —
   investigate before shipping.
6. The estimate matches pgvector's observed index size (`\di+` in psql) on a
   real build to within ±20%. (Pgvector's HNSW index does not duplicate the
   vector column; the `CREATE INDEX` memory is close to `hnsw_graph_bytes`.)

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Inputs declared | Corpus size, dim, metric, recall target, QPS, replicas, RAM budget — all written down explicitly. No "assume..." without a value. |
| HNSW memory | Vector bytes + graph bytes computed separately; graph bytes scale with `M`; the total is within ±20% of a measured serialized index for the same `(N, d, M)`. |
| Scalar types | At least three scalar-type rows (fp32, fp16, int8). Learner names what each costs on recall as a constraint for the compression trade-off (cross-link to mod-103 exercise-05). |
| IVF-PQ memory | Code bytes dominate; centroids and codebook accounted for separately; the total matches a built FAISS index's serialized size. |
| Latency estimate | Distances-per-query order-of-magnitude estimate is written down for both families and compared to exercise-02 measurements. |
| Build time | Order-of-magnitude build-time estimate for both families (not just "a while"). |
| QPS × replica math | Per-replica QPS target is derived from the global target and the replica count. |
| Verdict | One sentence per family naming "fits / does not fit" and the trigger to re-evaluate. |

Fail-fast smells:

- HNSW total RAM is reported as `N * d * 4` with no graph line. The learner has
  under-counted by 20–50%; big index builds get OOM-killed exactly here.
- IVF-PQ total is reported in GB when the right unit is MB. Off-by-factor from
  forgetting that `m_pq * nbits / 8 = ~16–64` bytes per vector, not `d * 4`.
- Latency estimate is "fast enough" with no number. A reviewer cannot verify.
- Build time is estimated as a single number rather than a range. HNSW build
  time is highly sensitive to `efConstruction`; the worksheet needs to say
  which value was assumed.
- No mention of float16 / int8 compression. On a 32 GB node with a 10M × 384-dim
  corpus, fp32 vs. fp16 is the difference between "comfortable" and "tight";
  omitting it misses the main lever.

## 5. Common mistakes

- **Under-counting HNSW graph memory.** `N * d * 4` is the vector memory only.
  The graph is *not* a free side-effect; it adds 20–50% at common `M` values.
- **Over-counting IVF-PQ memory by storing full vectors alongside PQ codes.**
  You only need the full vectors if you plan to run a rescore stage; if you do,
  say so explicitly, because the "cheap PQ" line now carries a 15 GB
  attachment.
- **Forgetting the payload / metadata column.** `tenant_id`, `doc_type`,
  `source_uri`, `created_at`, and the chunk text itself are non-trivial at 10M
  rows. A text column averaging 500 bytes adds 5 GB, often more than the
  graph. Budget it.
- **Assuming "the DB manages this for me."** Vector DBs do not magically shrink
  data. Their on-disk size matches the formulas above with a small overhead
  factor. If the vendor's sizing calculator says something very different,
  compare the inputs — usually they are using a different compression or a
  different metric assumption.
- **Treating latency as a build-time property.** HNSW latency depends on
  `ef_search`, which is query-time. The budget must state the operating point
  (e.g., `ef_search = 64`) because the latency estimate is only meaningful at
  a specific point on the recall/QPS frontier.
- **Confusing RAM with disk.** Some stores keep the full index in RAM
  (pgvector HNSW when `maintenance_work_mem` + shared buffers can hold it;
  Qdrant with `on_disk=false`); others memory-map and let the OS page. A
  budget for "memory" without naming which semantics is used is ambiguous.
- **Ignoring replicas in the cost math.** Two replicas is 2× RAM and 2×
  licensing / instance cost. If the budget is written per-replica but the
  invoice is for the whole fleet, the final number comes in 2× (or 3×) over
  what was presented.

## 6. References

- HNSW memory and the layer-0 / upper-layer approximation:
  - Malkov & Yashunin (2016), "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
  - hnswlib README on index storage layout — [github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib).
- IVF-PQ memory and codebook layout:
  - Jégou, Douze, Schmid (2011), "Product Quantization for Nearest Neighbor Search" — [ieeexplore.ieee.org/document/5432202](https://ieeexplore.ieee.org/document/5432202).
  - FAISS wiki — "Guidelines to choose an index" and "The index factory" —
    [github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index),
    [github.com/facebookresearch/faiss/wiki/The-index-factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory).
- Store-specific sizing notes (cross-checks):
  - pgvector index notes — [github.com/pgvector/pgvector#hnsw](https://github.com/pgvector/pgvector#hnsw).
  - Qdrant "capacity planning" — [qdrant.tech/documentation/guides/capacity-planning/](https://qdrant.tech/documentation/guides/capacity-planning/).
  - Milvus sizing tool — [milvus.io/tools/sizing](https://milvus.io/tools/sizing/).
- `ann-benchmarks` results page for measured latency / recall shapes —
  [ann-benchmarks.com](https://ann-benchmarks.com/).
- Risk / governance framings consulted while drafting the sizing worksheet:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
