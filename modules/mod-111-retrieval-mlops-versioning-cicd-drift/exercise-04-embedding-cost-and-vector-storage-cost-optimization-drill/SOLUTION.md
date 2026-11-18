# Exercise 04 — Solution: Embedding Cost and Vector-Storage Cost Optimization Drill

This is a reference solution for
`exercise-04-embedding-cost-and-vector-storage-cost-optimization-drill.md`.
It is a **design** exercise with runnable numerical worked examples.
The learner builds a cost model for a retrieval stack, identifies
which knob costs the most, and proposes a bundle of changes that
cuts cost by a target amount without crossing the exercise-02 eval
budget.

The teaching point: *RAG cost is dominated by two axes — embeddings
(one-time at build + per-query at serve) and vector storage (RAM
for HNSW, disk for IVF/PQ, both for most engines)*. Most
"optimizations" people reach for first (chunking tweaks, batch-size
tuning) move the smaller numbers. The levers that actually bend the
cost curve are **embedding-model choice**, **vector
dimensionality**, **quantization**, and **cache hit rate** (see
exercise-05 for the cache tier). Everything else is noise around
those four.

## 1. Solution overview

The deliverable has four parts:

1. **`cost/model.py`** — a concrete cost model that takes the
   stack's parameters and emits a $/month breakdown by component.
   Deterministic; no hidden constants; every unit price is a named
   variable with a cited source (model pricing page, cloud-storage
   price sheet).
2. **`cost/scenarios.md`** — three worked scenarios: small
   (100k chunks, 1k QPD), medium (10M chunks, 100k QPD), large
   (1B chunks, 10M QPD). For each: a baseline cost, a bottleneck
   analysis, and a proposed optimization bundle with the expected
   cost and quality impact.
3. **`cost/selection_matrix.md`** — the decision matrix for
   embedding-model dimensionality and vector-index kind (flat vs
   IVF vs HNSW vs IVFPQ vs product-quantized HNSW), with the
   quality-vs-cost tradeoff for each cell.
4. **`cost/drill_runbook.md`** — the five-step drill: measure
   baseline, rank by cost, propose cuts, validate against the
   exercise-02 eval gate, decide. Ends with a signed-off
   decision memo template.

Scope: *cost design and tradeoff*. Measurement of actual quality
impact lives with exercise-02. Caching lives with exercise-05.
Index automation lives with exercise-03.

## 2. Worked answer or implementation

### 2.1 Where the money goes

A retrieval stack has five cost components at any scale:

1. **One-time embedding (build).** `N_chunks × tokens_per_chunk ×
   price_per_token` for the embedding provider.
2. **Query-time embedding (serve).** `QPD × 30 × avg_query_tokens
   × price_per_token`, reduced by the embedding cache hit rate
   (exercise-05).
3. **Vector storage.** Bytes on the hot path (RAM or SSD, engine-
   dependent) × price-per-GB-month.
4. **ANN compute.** CPU-seconds for the query path if self-hosted;
   per-query charges if managed (Pinecone, Vertex AI Vector
   Search, Azure AI Search).
5. **Metadata + text storage.** The doc text + metadata the
   retriever returns. Usually object storage; usually a rounding
   error until the corpus is huge.

Rules of thumb before the model:

- **Embeddings dominate at build time for large corpora.**
  10M chunks × 300 tokens × $0.00013/1k tokens ≈ $390 one-time —
  fine. 1B chunks × 300 tokens × $0.00013/1k tokens ≈ $39,000 —
  a line item.
- **Storage dominates at serve time for large indices.** 1B × 1536
  dims × 4 bytes/dim (`float32`) = 6 TB on disk, more in RAM.
  Even at $0.023/GB-month for S3 that is $138/month for *cold*
  storage; HNSW in RAM at an AWS r6a.8xlarge class is $500–$800
  per month per node, and you need several.
- **Query-time embedding dominates at serve time for small
  corpora and high QPS.** 100k chunks is "small"; 1M queries/day
  on 3-large embeddings at full price is ~$150/month on the
  embedding bill — a bigger number than the index.
- **ANN compute is rarely the bottleneck for managed services
  until QPS is very high.** For self-hosted, it is the number of
  replicas you need to make p95 — see §2.4.

The point is that the dominant cost moves with the shape of the
deployment. A one-size-fits-all optimization bundle (e.g. "use
MRL with d=256 and PQ-32") is the wrong answer to half the
scenarios in §2.3.

### 2.2 `cost/model.py` — the concrete cost model

```python
"""Deterministic cost model for a retrieval stack. All prices cited
to published public price pages; update by editing the constants.
No hidden assumptions, no 'ops overhead' fudge factor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---- Price constants (edit these when provider prices move) -----------------

# OpenAI embedding model pricing (per 1k input tokens), from
# https://openai.com/api/pricing/.
PRICE_PER_1K_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.00010,
}

# Dimensions per model. 3-small / 3-large support MRL truncation
# https://openai.com/index/new-embedding-models-and-api-updates/
DEFAULT_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Storage unit prices (USD / GB-month), update to your region.
PRICE_PER_GB_MONTH = {
    "ram_r6a_amortized": 4.0,     # amortized RAM on r6a.8xlarge class
    "nvme_local":        0.10,    # instance-attached NVMe
    "s3_standard":       0.023,
    "gcs_standard":      0.020,
    "ebs_gp3":           0.08,
}

# Bytes per vector dimension by element type.
BYTES_PER_ELEMENT: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "int8":    1.0,          # scalar quantization
    "pq8":     0.125,         # one byte per 8 dimensions after product quantization
}


@dataclass
class Stack:
    # Corpus shape
    n_chunks: int                 # e.g. 10_000_000
    avg_chunk_tokens: int = 300

    # Query shape (per day)
    qpd: int = 100_000
    avg_query_tokens: int = 20
    days_per_month: int = 30

    # Embedding model
    model: str = "text-embedding-3-large"
    dim: int | None = None              # None → DEFAULT_DIM[model]
    dim_truncation: int | None = None   # MRL: project to this dim before indexing
    emb_cache_hit_rate: float = 0.0     # exercise-05

    # Index
    index_kind: Literal["flat", "hnsw", "ivfflat", "ivfpq", "hnsw_pq"] = "hnsw"
    element: str = "float32"            # float32|float16|int8|pq8
    hnsw_m: int = 32                    # ~ graph degree
    ivf_nlist: int | None = None        # IVF partitions
    pq_m:    int | None = None          # PQ sub-vectors (bytes/vector when pq8)

    # Hosting
    storage_tier: str = "ram_r6a_amortized"  # ram_r6a_amortized|nvme_local|...

    # Managed? If True, override compute cost with per-query provider price.
    managed: bool = False
    managed_price_per_query: float = 0.00040  # illustrative; set to your provider

    # ---- Derived cost model ------------------------------------------------

    def effective_dim(self) -> int:
        d = self.dim or DEFAULT_DIM[self.model]
        if self.dim_truncation and self.dim_truncation < d:
            d = self.dim_truncation
        return d

    def bytes_per_vector(self) -> float:
        d = self.effective_dim()
        if self.element == "pq8":
            m = self.pq_m or max(1, d // 8)
            return float(m)  # one byte per sub-vector
        return d * BYTES_PER_ELEMENT[self.element]

    def hnsw_overhead(self) -> float:
        """HNSW graph overhead: ~M + M0 pointers/vector × 4 bytes/pointer.
        Rough rule-of-thumb per the HNSW paper's memory discussion
        and FAISS docs. Treat as a lower bound; real-world varies."""
        if self.index_kind in ("hnsw", "hnsw_pq"):
            return (self.hnsw_m + self.hnsw_m * 2) * 4
        return 0.0

    def ivf_overhead(self) -> float:
        return 0.0  # negligible vs vector data

    def storage_bytes(self) -> float:
        return self.n_chunks * (self.bytes_per_vector() + self.hnsw_overhead() + self.ivf_overhead())

    def storage_gb(self) -> float:
        return self.storage_bytes() / (1024 ** 3)

    def monthly_storage_cost(self) -> float:
        return self.storage_gb() * PRICE_PER_GB_MONTH[self.storage_tier]

    def one_time_build_cost(self) -> float:
        tokens = self.n_chunks * self.avg_chunk_tokens
        return tokens / 1000 * PRICE_PER_1K_TOKENS[self.model]

    def monthly_query_embed_cost(self) -> float:
        queries = self.qpd * self.days_per_month * (1 - self.emb_cache_hit_rate)
        tokens  = queries * self.avg_query_tokens
        return tokens / 1000 * PRICE_PER_1K_TOKENS[self.model]

    def monthly_compute_cost(self) -> float:
        if self.managed:
            return self.qpd * self.days_per_month * self.managed_price_per_query
        # Self-hosted: pays via the storage tier (RAM nodes serve queries).
        return 0.0

    def report(self) -> dict:
        return {
            "one_time_build":     round(self.one_time_build_cost(), 2),
            "monthly_query_embed": round(self.monthly_query_embed_cost(), 2),
            "monthly_storage":    round(self.monthly_storage_cost(), 2),
            "monthly_compute":    round(self.monthly_compute_cost(), 2),
            "monthly_total":      round(
                self.monthly_query_embed_cost()
                + self.monthly_storage_cost()
                + self.monthly_compute_cost(), 2,
            ),
            "storage_gb":         round(self.storage_gb(), 2),
            "effective_dim":      self.effective_dim(),
            "bytes_per_vector":   self.bytes_per_vector(),
        }
```

The model has three virtues and one limitation:

- **Deterministic.** Same inputs → same cost. Useful in a review:
  the reviewer can rerun `python -m cost.model` and verify.
- **Transparent.** Every constant is named; the price page is in
  the docstring.
- **Composable.** `Stack` instances can be swapped for the
  scenarios in §2.3 without touching the cost math.
- **(Limit)** The HNSW overhead formula is a rule of thumb; real
  numbers vary by engine (pgvector vs Qdrant vs Milvus).
  Replace with measured `top` / `statm` numbers from a running
  instance when the drill matters.

### 2.3 `cost/scenarios.md` — three worked scenarios

For each scenario the drill is: **baseline cost → bottleneck →
candidate bundle → projected cost → quality guardrail
(exercise-02)**.

#### Scenario A — Small (100k chunks, 1k QPD, 20-tok queries)

```
Baseline Stack:
  model=text-embedding-3-large, dim=3072, index=hnsw(float32),
  storage=ram_r6a_amortized, hit_rate=0
```

Cost report:

| Component | $ |
|---|---:|
| One-time build (100k × 300 tokens × $0.00013/1k) | **3.90** |
| Monthly query-embedding (30k × 20 tokens × $0.00013/1k) | 0.08 |
| Monthly storage (100k × 3072 × 4 B + HNSW overhead = 1.19 GB, @ $4/GB) | 4.76 |
| Monthly compute (self-hosted) | 0 |
| **Monthly total** | **~$4.84 + $3.90 one-time** |

Bottleneck: *no bottleneck*. At this scale the dominant cost is
whatever rounding error is biggest. Optimization bundle: **do
nothing**. A dollar of engineering time is worth more than a
dollar of this stack.

If this grows to 100k QPD, re-run the model — the query-embedding
line will go to ~$8/month and the drill will have a point.

#### Scenario B — Medium (10M chunks, 100k QPD)

```
Baseline Stack:
  model=text-embedding-3-large, dim=3072, index=hnsw(float32),
  storage=ram_r6a_amortized, hit_rate=0
```

Cost report:

| Component | $ |
|---|---:|
| One-time build | 390.00 |
| Monthly query-embedding | 7.80 |
| Monthly storage (10M × 3072 × 4 B + overhead = 118.6 GB × $4) | **474.46** |
| Monthly compute | 0 |
| **Monthly total** | **~$482** |

Bottleneck: **storage**. 474 / 482 = 98% of monthly cost. Query-
embed is a rounding error; one-time build is paid down over a
few months.

Candidate bundle:

| Lever | Change | Effect on cost | Effect on quality |
|---|---|---|---|
| **MRL truncation** | dim 3072 → 1024 (Matryoshka representation learning, [MRL paper](https://arxiv.org/abs/2205.13147); OpenAI's `-3-large` supports `dimensions` [param](https://platform.openai.com/docs/guides/embeddings/)) | −66% on storage ≈ $316/mo saved | Published MRL benchmarks show ~1–3% nDCG delta to 1024, slightly more to 512 (model-dependent; verify with exercise-02) |
| **Scalar quantization** | float32 → int8 (8× smaller) | −75% on vector bytes ≈ $358/mo saved vs baseline | ~1–2% recall delta typical (verify) |
| **HNSW_PQ** | PQ-32 inside HNSW | −12× on vector bytes | 3–5% recall delta typical — more aggressive |

Projected bundle (MRL 1024 + int8): storage ≈ $474 × (1024/3072)
× (1/4) ≈ **$39/mo**, savings ≈ **$435/mo** (−91%). Quality
guardrail: run exercise-02 against both candidates (MRL-only and
MRL+int8). Ship whichever passes the gate with the most headroom.

#### Scenario C — Large (1B chunks, 10M QPD)

```
Baseline Stack:
  model=text-embedding-3-large, dim=3072, index=hnsw(float32),
  storage=ram_r6a_amortized, hit_rate=0
```

Cost report:

| Component | $ |
|---|---:|
| One-time build | 39,000.00 |
| Monthly query-embedding | 780.00 |
| Monthly storage (1B × 3072 × 4 B + overhead = 11.9 TB × $4) | **48,657** |
| Monthly compute | 0 |
| **Monthly total** | **~$49,437** |

RAM-resident HNSW over 11.9 TB is architecturally infeasible on
a reasonable node count. The baseline itself is wrong. At this
scale the stack has to shift to either:

- **IVFPQ or HNSW-PQ on NVMe-backed shards** (FAISS, Milvus).
  Storage drops ~32×, lands on NVMe at $0.10/GB-month, total
  storage ~$37/mo. Compute dominates at this tier (more replicas
  for p95).
- **Managed service** (Pinecone, Vertex AI Vector Search). Shift
  from storage-dominated to per-query-dominated: 10M × 30 ×
  $0.00040 ≈ **$120,000/mo**. Worse than the self-hosted
  optimized stack, but may be the right pick if there is no
  team to operate a sharded Milvus cluster.

Candidate bundle (self-hosted): MRL 1024 + HNSW_PQ(16) on NVMe.
Vector bytes: 16/vector; storage ≈ 15 GB × $0.10 ≈ $1.50/mo (the
index graph overhead dominates; realistic figure is 50–200 GB and
~$5–$20/mo). Quality guardrail: expect 3–5% recall delta;
validate with the full production qrels, not the CI shard.

#### Drill numbers across the three

| Scenario | Baseline $/mo | Bottleneck | Target | Bundle |
|---|---:|---|---:|---|
| Small | $4.84 | none | — | no change |
| Medium | $482 | storage | ≤ $100/mo (−80%) | MRL 1024 + int8, keep HNSW |
| Large | $49,437 | storage (infeasible) | ≤ $5,000/mo (−90%) | IVFPQ or HNSW_PQ on NVMe; consider sharding |

### 2.4 `cost/selection_matrix.md` — the decision matrix

The dimension-vs-index-kind matrix. Each cell shows the *quality
delta vs the full float32/dense baseline* (typical ranges; verify
with exercise-02) and the *storage multiplier*.

| | Flat (exact) | HNSW | IVFFlat | IVFPQ | HNSW_PQ |
|---|---|---|---|---|---|
| **float32, full dim** | 1.00× storage, 0% delta | 1.0–1.05× storage, 0–1% recall delta | 1.00× storage, 0–1% recall delta (needs training) | — | — |
| **float32, MRL-1024** | 0.33× storage, 1–3% delta | 0.33× storage + overhead, 1–3% delta | 0.33× storage, 1–3% delta | — | — |
| **int8, MRL-1024** | 0.08× storage, 2–4% delta | 0.08× storage + overhead, 2–4% delta | 0.08× storage, 2–4% delta | — | — |
| **PQ, MRL-1024** | — | — | — | ~0.02× storage, 3–6% delta | ~0.02× storage, 3–6% delta |

Rules the matrix encodes:

- **Flat is the baseline for correctness.** Use it on small
  corpora, or as the ground truth to measure approximate indexes
  against during tuning.
- **HNSW is the default** for RAM-resident, high-recall
  workloads. The tunable is `M` / `efConstruction`; higher = more
  recall, more RAM. See the HNSW paper (Malkov & Yashunin, 2018)
  and FAISS docs: [faiss.ai](https://faiss.ai/).
- **IVF is the default for disk-resident or very-large indexes.**
  Shard-friendly; `nlist` set ~sqrt(N). Requires training vectors.
- **Quantization composes with ANN.** `int8` is near-free quality-
  wise and 4× smaller than `float16`, 8× smaller than `float32`;
  `PQ` is more aggressive, 32–64× smaller but with a bigger
  recall hit. See FAISS's PQ docs.
- **MRL truncation composes with quantization and ANN.** A model
  trained with MRL (e.g. OpenAI `-3-large`, Nomic Embed) can
  project down to d=256 or d=512 with a quality curve you can
  measure. See the MRL paper and OpenAI's post on
  `-3-small`/`-3-large` dimensions
  ([new-embedding-models-and-api-updates](https://openai.com/index/new-embedding-models-and-api-updates/)).
- **Model-family matters.** E5-mistral-7b, BGE-large, Nomic Embed
  v1.5, Cohere Embed v3 have different cost curves; pricing and
  quality are not fungible across providers. Benchmark on your
  qrels.

### 2.5 `cost/drill_runbook.md` — the drill

Five steps. Each step produces a signed-off artifact.

**Step 1 — Measure the baseline (1 day).**

- Read the current manifest (exercise-01) for `dim`, `element`,
  `index_kind`, `n_chunks`.
- Measure actual memory and disk via `prometheus`/`top`/engine
  stats. Replace the theoretical `storage_gb` with the measured
  one.
- Pull the embedding spend for the last 30 days from the provider
  billing export.
- Produce `cost/baseline.json` from `cost/model.py`.

**Step 2 — Rank components by cost (30 minutes).**

- Sort the baseline report descending. The top line item is the
  only one worth optimizing first. Skip the bottom three line
  items unless they sum to more than the top.

**Step 3 — Propose cuts (half a day).**

- Pick ≤ 3 levers to change per cycle. Changing five things at
  once makes attribution impossible.
- Project the new cost by rerunning `cost/model.py` with the
  candidate `Stack`.
- Cross-reference the selection matrix to estimate the quality
  delta range.

**Step 4 — Validate against exercise-02's eval gate (1 day).**

- Build a candidate index from the candidate config using
  exercise-03's DAG (manual trigger with config override).
- Run the retrieval eval (exercise-02) against the candidate.
- Accept only if the gate passes with the current budget; if it
  does not, either tighten the lever (e.g., MRL 1024 instead of
  512) or drop the lever.

**Step 5 — Decision memo (half a day).**

- Produce a one-page memo with: baseline cost, candidate bundle,
  projected cost, measured quality delta, residual risk, and the
  approver.
- File it with the exercise-01 manifest history so the next
  reviewer sees *why* the index parameters are what they are.

Template memo:

```markdown
# Retrieval cost-optimization decision — YYYY-MM-DD

**Baseline:** $X/month (component breakdown attached).
**Bottleneck:** `<component>`.
**Bundle:** `<lever 1, lever 2, lever 3>`.
**Projected cost:** $Y/month (savings $X−Y, −Z%).
**Measured quality:** <metric> baseline <a> → candidate <b>, delta <d>
  (95% CI [<lo>, <hi>]; see eval run <id>).
**Residual risk:** <what could still go wrong; triggers for revisit>.
**Approver:** <name>.
```

### 2.6 Lock-in considerations (the non-cost axis)

The matrix is money, but the drill must also note:

- **Managed service lock-in.** Pinecone / Vertex AI Vector Search
  / Azure AI Search have their own index formats and alias
  semantics (exercise-01). Egress costs for vector data are
  non-trivial — plan a migration cost in the "residual risk"
  line.
- **Provider pricing volatility.** Embedding prices drop
  periodically and unpredictably. Pin a quarterly review of the
  constants in `cost/model.py` to catch these without an
  incident.
- **Model-specific MRL support.** Not every model supports MRL;
  truncating a non-MRL model loses meaningful information. Check
  the provider's documentation for the specific model before
  assuming `dim_truncation` is safe.

## 3. Validation steps

A correct submission can be validated with these checks:

1. **Model reproducibility.** `python -m cost.model` against the
   baseline JSON produces the same numbers as the submission's
   worked example, to the penny.
2. **Price-page citations.** Every price constant in
   `cost/model.py` has a comment linking to the published
   price page. Reviewer can open the URL and confirm.
3. **Scenario coverage.** All three scenarios (small/medium/large)
   are addressed and the "small → do nothing" answer is explicit;
   no scenario says "optimize" without a cost delta that justifies
   the engineering time.
4. **Lever evidence.** Each proposed lever (MRL, int8, PQ) is
   linked to the provider's or library's documentation — not to a
   blog post.
5. **Quality guardrail.** The drill explicitly routes candidates
   through exercise-02; a candidate that fails the eval is not
   shipped regardless of cost impact.
6. **No silent wins.** A bundle that saves money but claims
   "no quality impact" without an eval number is rejected.
7. **Approved memo.** The one-page memo template is filled out
   for at least one scenario end-to-end.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Cost model is deterministic | Same inputs produce the same numbers; constants are named; sources are linked. |
| All five components covered | One-time build, query-embed, storage, compute, metadata are each represented. |
| Bottleneck identified | The scenario's optimization bundle targets the largest component, not the easiest one. |
| ≤ 3 levers per bundle | More than three levers in one change makes attribution impossible. |
| Lever has a doc reference | MRL, quantization, HNSW_PQ each link to the originating paper or library doc. |
| Quality guardrail | Every cost cut is gated by exercise-02's eval; a cut that fails the gate is rejected. |
| Lock-in disclosed | Managed-service vs self-hosted lock-in is called out in the residual-risk line. |
| Scenario triage | "Small scenario → do nothing" is a valid and correct answer, not a dodge. |
| Decision memo | At least one scenario has a filled-out decision memo with approver. |
| Cost is monthly + one-time separately | One-time build cost is not amortized into the monthly number without saying so. |

Fail-fast smells:

- The cost model has a magic 1.5× "ops overhead" factor. Opaque,
  untestable; replace with the specific line item.
- Every scenario's bundle is "use MRL + int8". Scenarios differ
  because their bottlenecks differ; a one-size-fits-all bundle
  tells you the author did not do the bottleneck step.
- Cost is reduced by 95% and quality impact is "negligible"
  without a cited eval number. 95% storage reductions typically
  cost some quality; the question is how much, measured.
- The savings number is bigger than the baseline (negative cost).
  Price constant is wrong; re-cite.
- The drill proposes an index-engine swap without a migration-cost
  line. Vector-data egress is often the hidden cost.

## 5. Common mistakes

- **Optimizing the wrong knob.** Spending a sprint shrinking
  chunk size when the bottleneck is storage. Rank by cost first.
- **Ignoring the one-time build cost.** For a 1B-chunk corpus,
  the one-time embedding spend is comparable to a month of
  storage. It is not free; it is a line item.
- **Assuming cost scales linearly with chunks.** HNSW graph
  overhead and IVF `nlist` do not; the storage model must account
  for it. Measure a real instance for the actual curve.
- **Confusing float16 with int8.** `float16` is 2 bytes; `int8`
  is 1 byte with scaling factors per sub-vector; not
  interchangeable in quality terms.
- **Applying MRL to a non-MRL model.** Truncating the vector of
  a model that was *not* trained for Matryoshka representations
  destroys information. Check the model card.
- **Changing five things at once.** Ship a −60% cost bundle, see
  a −5% nDCG; which of the five caused it? Attribution impossible.
- **Skipping the eval gate.** "The cost model says it saves $5k"
  without an eval number is not an optimization; it is a
  guess. See exercise-02.
- **Treating managed per-query pricing as free compute.** At
  high QPS it is often the biggest cost; model it.
- **Forgetting cache hit rate.** A 60% embedding-cache hit rate
  (exercise-05) cuts the query-embedding line by 60%. That might
  be the whole optimization.

## 6. References

Embedding-provider pricing (primary; re-check each time):

- OpenAI — [openai.com/api/pricing/](https://openai.com/api/pricing/).
  Per-1k-token pricing for `text-embedding-3-small`,
  `text-embedding-3-large`, `text-embedding-ada-002`.
- OpenAI — [new embedding models and API updates](https://openai.com/index/new-embedding-models-and-api-updates/).
  The `dimensions` parameter and MRL support.
- Cohere — [cohere.com/pricing](https://cohere.com/pricing/).
- Anthropic — Anthropic does not publish a standalone embedding
  model. If this is relevant, treat as a non-option.
- Nomic — [atlas.nomic.ai/embedding](https://atlas.nomic.ai/embedding).
- Hugging Face — [huggingface.co/docs/inference-endpoints](https://huggingface.co/docs/inference-endpoints/).
  Self-hosted via Inference Endpoints or your own GPUs.

Index / ANN libraries:

- FAISS — [faiss.ai](https://faiss.ai/).
  Reference for IVF, PQ, HNSW, IVFPQ; tuning guide for
  memory/recall tradeoffs.
- HNSW paper — Malkov & Yashunin, 2018,
  [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
- Matryoshka Representation Learning — Kusupati et al., 2022,
  [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147).
- Qdrant quantization —
  [qdrant.tech/documentation/guides/quantization](https://qdrant.tech/documentation/guides/quantization/).
- Milvus index types —
  [milvus.io/docs/index.md](https://milvus.io/docs/index.md).
- pgvector — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
- Pinecone pricing — [pinecone.io/pricing](https://www.pinecone.io/pricing/).
- Vertex AI Vector Search pricing —
  [cloud.google.com/vertex-ai/pricing](https://cloud.google.com/vertex-ai/pricing#vector-search).
- Azure AI Search pricing —
  [azure.microsoft.com/en-us/pricing/details/search/](https://azure.microsoft.com/en-us/pricing/details/search/).

Cloud-storage price sheets (used in §2.2 constants):

- AWS S3 — [aws.amazon.com/s3/pricing](https://aws.amazon.com/s3/pricing/).
- GCP Cloud Storage — [cloud.google.com/storage/pricing](https://cloud.google.com/storage/pricing).
- Azure Blob — [azure.microsoft.com/en-us/pricing/details/storage/blobs/](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/).

Governance framings:

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: **MEASURE 2.6** (performance-cost tradeoff is part of
  post-deployment measurement), **MANAGE 4.3** (tradeoff decisions
  and residual-risk disclosure).

Related exercises:

- exercise-01 — the manifest provides the `dim`, `element`,
  `index_kind` the model reads.
- exercise-02 — the eval gate every candidate bundle passes
  through.
- exercise-03 — the DAG that materializes the candidate index.
- exercise-05 — the cache tier whose hit rate reduces the
  query-embedding line.
