# Exercise 03 — Solution: Hosted vs. self-hosted embeddings — cost, latency, and quality per dollar

This is a reference solution for
`exercise-03-hosted-vs-self-hosted-embedding-cost-latency-quality-sweep.md`. It is a
**design + light implementation** exercise: the learner builds a sheet that decides
whether to call a hosted embedding API (OpenAI / Cohere / Voyage) or self-host a model
(bge, e5, nomic) for a given corpus and QPS profile. The reference below provides a
worked spreadsheet, a formula sheet, a decision flow, and a rubric.

## 1. Solution overview

The deliverable has four parts:

1. **`notes/model-shortlist.md`** — the shortlist of embedders being compared, with
   their current price-per-1M-tokens for hosted options and GPU-hours required for
   self-hosted options. Prices and model lists shift; the learner's job is to pin
   the pull date and the citation URL.
2. **`notes/formula-sheet.md`** — the five formulas that drive the decision: ingest
   cost, re-embed cost, query-time cost per 1k queries, storage cost per 1M vectors
   at each dimensionality, and quality-per-dollar normalized by local NDCG@10.
3. **`sheets/sweep.csv`** — a filled sweep with a corpus of 10 M chunks, 50-token
   average query length, and three QPS targets (0.1, 10, 100). Every cell is derivable
   from the formula sheet.
4. **`notes/decision.md`** — the recommendation that falls out of the sweep. The
   reference states which band each option wins in — not "pick one and move on."

The goal is a defensible decision with every input traceable to a cited source price
and a locally measured quality number (from exercise-02). No hand-waving: every dollar
figure has a formula behind it.

## 2. Worked answer or implementation

### 2.1 `notes/model-shortlist.md`

Pin the pull date — these prices and model catalogs change often. The reference
shortlist for comparison:

**Hosted embeddings** — price = $/1M input tokens; verify from the current pricing
page on the day of the run.
- OpenAI `text-embedding-3-small` (1536-dim, with Matryoshka dim-reduction support).
- OpenAI `text-embedding-3-large` (3072-dim, Matryoshka).
- Cohere `embed-v3` family (`english-v3.0`, `multilingual-v3.0`, `light-v3.0`).
- Voyage `voyage-3` / `voyage-3-lite`.

For each row, record:
- Model id.
- Max tokens per input.
- Output dimensionality (and dim-reduction support if any).
- Per-1M-token price for input embedding calls.
- Rate limits (RPM / TPM).
- Language coverage (English-only vs multilingual).
- Terms-of-service constraints (e.g., training-data usage, PII retention,
  region of processing).

`<!-- pin-at-runtime: pin exact per-1M-token prices from the provider's current pricing page on the day of the run; prices change between model revisions and I will not fabricate numbers. -->`

**Self-hosted candidates** — measurable locally, deterministic to price.
- `BAAI/bge-small-en-v1.5` (384-dim, ~33M params).
- `BAAI/bge-base-en-v1.5` (768-dim, ~109M params).
- `BAAI/bge-large-en-v1.5` (1024-dim, ~335M params).
- `intfloat/e5-base-v2` / `intfloat/e5-large-v2`.
- `nomic-ai/nomic-embed-text-v1.5` (768-dim, Matryoshka-capable, up to 8192 tokens).

For each, record:
- Output dimensionality.
- Token-sequence budget.
- Throughput (tokens/sec) measured locally on your target GPU (or CPU).
- VRAM footprint at the chosen batch size.
- License (MIT / Apache-2.0).
- Required inference prefix (from exercise-02).

### 2.2 `notes/formula-sheet.md`

Five formulas. Every cell in the sweep uses one of them.

1. **Ingest cost (hosted).**
   ```
   ingest_cost = (total_tokens / 1_000_000) * price_per_M_input
   ```
   where `total_tokens` counts all tokens embedded during the initial corpus build.

2. **Re-embed cost (hosted) when a corpus revision is required.**
   ```
   reembed_cost = (changed_fraction * total_tokens / 1_000_000) * price_per_M_input
   ```
   This is the number that bites. A model deprecation, a chunking-policy change, or a
   switch of embedder re-embeds everything. The learner must show the "full re-embed"
   cost in the sweep, not just the first-time cost.

3. **Query-time cost per 1k queries.**
   - Hosted:
     ```
     query_cost_1k = 1000 * (avg_query_tokens / 1_000_000) * price_per_M_input
     ```
     (Most hosted embedders bill all input tokens at one flat rate; some charge the
     same rate for queries and corpus; double-check per-provider.)
   - Self-hosted:
     ```
     query_cost_1k = 1000 * avg_query_latency_seconds * gpu_hourly_rate / 3600
     ```
     at the measured per-query latency (batched or unbatched, name which).

4. **Storage cost per 1M vectors.**
   ```
   bytes_per_vector = dim * bytes_per_scalar  # fp32=4, fp16=2, int8=1, binary=1/8
   storage_bytes_1M = 1_000_000 * bytes_per_vector
   storage_cost_month = storage_bytes_1M * price_per_gb_month / (1024**3)
   ```
   This is the raw vector size; indexes add overhead (HNSW adds graph neighbors,
   quantized indexes carry centroid tables). Record the raw number and the actual
   on-disk size from the vector DB (mod-104).

5. **Quality-per-dollar (normalized).**
   ```
   qpd = ndcg_at_10_on_local_corpus / cost_per_1M_tokens_over_year
   ```
   The "per-year" denominator is deliberate: it lets models with high recurring query
   cost (hosted at high QPS) be fairly compared to models with a one-time GPU buy
   (self-hosted). Name the assumed year's query volume.

### 2.3 `sheets/sweep.csv` — the filled grid

Columns: `model`, `deployment` (hosted / self-hosted), `dim`, `price_per_M_tokens_USD`,
`gpu_hourly_USD`, `measured_throughput_tokens_per_sec`, `ingest_tokens_total`,
`ingest_cost_USD`, `reembed_cost_USD`, `queries_per_day`, `query_cost_per_day_USD`,
`storage_bytes_per_vector`, `storage_cost_month_USD_per_1M_vecs`, `local_ndcg_at_10`,
`qpd`.

Worked example (one row per permutation of model × QPS tier). The spreadsheet is the
same shape as the template below. Each column has units written into the header so
nobody mixes `price per 1k` with `price per 1M`.

| model | deploy | dim | $/1M tok | $/GPU-hr | tok/s | ingest tokens | ingest $ | re-embed $ | q/day | query $/day | B/vec | storage $/mo/1M vec | local NDCG@10 | QPD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai/text-embedding-3-small | hosted | 1536 | *pin* | n/a | n/a | 5e9 | *compute* | *compute* | 8.6e5 | *compute* | 6144 | *compute* | *measure* | *compute* |
| openai/text-embedding-3-small-512 (Matryoshka cut) | hosted | 512 | *pin* | n/a | n/a | 5e9 | *compute* | *compute* | 8.6e5 | *compute* | 2048 | *compute* | *measure* | *compute* |
| cohere/embed-v3-english | hosted | 1024 | *pin* | n/a | n/a | 5e9 | *compute* | *compute* | 8.6e5 | *compute* | 4096 | *compute* | *measure* | *compute* |
| bge-small-en-v1.5 (A10G, fp16) | self | 384 | 0 | *pin provider* | *measure* | 5e9 | 0 | 0 | 8.6e5 | *compute* | 768 | *compute* | *measure* | *compute* |
| bge-large-en-v1.5 (A10G, fp16) | self | 1024 | 0 | *pin provider* | *measure* | 5e9 | 0 | 0 | 8.6e5 | *compute* | 2048 | *compute* | *measure* | *compute* |
| nomic-embed-text-v1.5 (A10G, fp16, 768) | self | 768 | 0 | *pin provider* | *measure* | 5e9 | 0 | 0 | 8.6e5 | *compute* | 1536 | *compute* | *measure* | *compute* |

`<!-- pin-at-runtime: GPU-hour prices vary by cloud provider (AWS/GCP/Azure/Lambda/RunPod) and region; pin the exact SKU and provider price URL at run time. -->`

The sweep is finished when every "compute" cell has a number derived from the formula
sheet and every "measure" cell has a number from a local run.

### 2.4 `notes/decision.md` — decision flow

Rather than "pick model X," the reference decision is structured as bands. A real
engineering team reads the sweep and places itself in one band:

- **Band A — low QPS, small corpus, variable workload.** Hosted embedders win. The
  ingest bill is small, there is no GPU sitting idle, and the operational cost of
  running a GPU for ~1 call/minute dominates. OpenAI `text-embedding-3-small` with a
  Matryoshka dim cut to 512 or 1024 is a defensible default here.
- **Band B — steady mid QPS (10–100 QPS), mid corpus (10–100 M chunks).** The
  break-even flips to self-hosted. Running `bge-small-en-v1.5` or
  `bge-base-en-v1.5` on an A10G or L4 fully-loaded is cheaper than the hosted
  per-token meter at this QPS, and the quality floor is in the same ballpark as
  the smaller hosted models on English retrieval.
- **Band C — bursty high QPS, strict p99 latency, multiple regions.** Mixed is often
  the honest answer: self-host the hot corpus (bge or nomic on dedicated GPUs) and
  keep hosted as an overflow / fallback. Do not let a single-provider outage take
  the stack down.
- **Band D — data-residency or PII constraints.** Hosted may be ruled out before any
  cost analysis — data cannot leave the environment. The decision collapses to the
  self-hosted shortlist.
- **Re-embed risk.** Every hosted embedder has been deprecated at least once in the
  industry's short history. Add a row to the sweep for a full re-embed in year two
  and look at the total cost. If re-embed exceeds ingest by a comfortable margin,
  that is a reason to self-host even in Band A.

Example (illustrative — plug in your own numbers):

- A 10 M-chunk corpus with ~500 tokens per chunk is ~5 × 10⁹ tokens of ingest. At
  a hosted rate of $0.02 per 1M input tokens (OpenAI's published small-model rate at
  the time of writing), that is $100 for one-time ingest. A full re-embed is another
  $100. 100 queries per day for a year at ~100 tokens per query is
  `365 * 100 * 100 / 1e6 * $0.02 ≈ $0.07`. Band A fits a hosted embedder cleanly.
- The same corpus at 100 QPS year-round is `100 * 86400 * 365 * 100 / 1e6 ≈
  3.15e8 tokens` per year — order $6.3k on OpenAI small; a self-hosted L4 at
  `~$0.60/hour on-demand` runs ~$5.3k/year *fully burdened at 100% utilization* and
  usually less reserved. Band B territory; self-hosted usually wins.

`<!-- pin-at-runtime: the two dollar values above use "$0.02 per 1M tokens" and "~$0.60/hour L4" as illustrative placeholders only. The learner must substitute their current provider price pulls before signing the recommendation. -->`

## 3. Validation steps

1. Every row in `sheets/sweep.csv` has a numeric value in every column; no cell says
   "TBD."
2. Hosted prices and GPU-hour prices in the sweep match the URL and pull date noted in
   `notes/model-shortlist.md`. (The reviewer re-opens the cited URL and checks.)
3. Self-hosted throughput (`tokens/s`) is measured locally, not taken from a blog
   post. The learner can produce the script that measured it.
4. The local NDCG@10 column matches the numbers produced in exercise-02 (same
   task, same date). If the quality column is empty, the quality-per-dollar column
   should be empty too — do not invent a number.
5. `notes/decision.md` places the team into one of the bands with a one-sentence
   justification. The reviewer can swap the QPS target and watch the recommendation
   change — the answer is sensitive to the inputs, not fixed.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Shortlist | At least three hosted and three self-hosted candidates, with current prices and pull dates cited. |
| Formulas | Five formulas appear; every cell in the sweep is derived from one of them. |
| Units | Every column names its units; nobody has mixed `$/1k` with `$/1M`. |
| Local throughput | Self-hosted rows carry a *measured* throughput, not an advertised one. |
| Quality column | Each row has a quality number from exercise-02 (or is explicitly flagged as "not yet measured" and left blank). |
| Re-embed row | A full-re-embed cost appears for hosted options; that number is in the recommendation, not hidden. |
| Decision | Falls out of the sweep, not from a preferred vendor. States which band the team is in and names one input change that would flip the answer. |
| Compliance band | Data-residency / PII constraints are explicitly considered before cost, not after. |
| Honesty | `<!-- pin-at-runtime: ... -->` markers appear anywhere a price was not actually pinned. |

## 5. Common mistakes

- **Comparing self-hosted at 100% utilization to hosted at 1 QPS.** The hosted bill at
  low QPS is tiny; the self-hosted GPU is spinning idle. Normalize both to the same
  QPS tier.
- **Forgetting the re-embed cost.** Models deprecate, chunking policies change,
  prefixes change. A sweep that only has the first ingest cost under-counts hosted
  options; a sweep that assumes a self-hosted model will never be upgraded
  under-counts self-hosted options. Put both re-embed costs in.
- **Using advertised tokens/sec instead of measured.** Throughput depends on batch
  size, sequence length, precision, and the GPU SKU. A single number from a blog
  post is not a substitute for a measurement on the target cluster.
- **Dropping the dimensionality column.** Storage cost is linear in dim; a 3072-dim
  embedder costs 3× a 1024-dim embedder in storage at the same corpus size. The
  Matryoshka cut rows exist to capture that cost explicitly.
- **Treating "quality" as a single number.** NDCG@10 on SciFact is a signal, not a
  decision. For a domain-specific corpus the quality column belongs to exercise-04.
  Until that number lands, the decision is provisional.
- **Fabricating prices.** If the price is not actually pinned to a URL on the pull
  date, write `<!-- pin-at-runtime: ... -->`. Inventing a vendor price number and
  signing a recommendation on it is the specific failure mode the honesty rules of
  this workpacket forbid.
- **Burying terms-of-service constraints in a footnote.** Data processing region,
  retention, and whether the provider uses your data for training are first-row
  concerns on the decision doc, not appendix items.

## 6. References

- Benchmark + measurement path (used to fill the "quality" column):
  - MTEB — [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb), leaderboard [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  - BEIR — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
- Hosted embedding providers (pin price and model list on the day of the run):
  - OpenAI embeddings docs — [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings).
  - Cohere embed docs — [docs.cohere.com/docs/embeddings](https://docs.cohere.com/docs/embeddings).
  - Voyage AI embedding docs — [docs.voyageai.com/docs/embeddings](https://docs.voyageai.com/docs/embeddings).
- Self-hosted model families:
  - BGE — [huggingface.co/BAAI](https://huggingface.co/BAAI).
  - E5 — [huggingface.co/intfloat](https://huggingface.co/intfloat).
  - Nomic Embed — [huggingface.co/nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).
  - Sentence-Transformers — [sbert.net](https://sbert.net/).
- Serving stacks (used to measure throughput for the self-hosted rows):
  - Hugging Face Text Embeddings Inference — [github.com/huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference).
  - vLLM serving docs (for models supported in vLLM) — [docs.vllm.ai](https://docs.vllm.ai/).
- Security + risk framings (cost sweep intersects with data-handling and model-supply-chain concerns):
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
