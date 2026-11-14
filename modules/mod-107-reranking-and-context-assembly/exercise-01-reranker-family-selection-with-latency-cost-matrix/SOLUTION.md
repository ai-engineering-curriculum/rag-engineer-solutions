# Exercise 01 — Solution: Reranker Family Selection With a Latency + Cost Matrix

This is a reference solution for
`exercise-01-reranker-family-selection-with-latency-cost-matrix.md`. It is a
**design** exercise. The learner surveys the reranker families that sit
between first-phase retrieval (mod-104/105/106) and prompt assembly
(exercise-04), builds a latency + cost matrix across realistic candidate-list
sizes, and writes a one-page recommendation for the mod-107 baseline.

The teaching point: there is no single "best" reranker. A reranker is a
function `(query, candidate) → score` that runs *after* a cheap first phase,
so its budget is **per-candidate inference latency × number of candidates**.
A matrix that does not multiply those two numbers is not yet a matrix.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/family-matrix.md`** — a table that compares the five reranker
   families (cross-encoders, late-interaction / ColBERT, listwise LLM
   rerankers, pointwise LLM-as-a-judge rerankers, hosted API rerankers)
   across eight dimensions: scoring shape, model size, typical per-pair
   latency, batching behaviour, quality on short-passage benchmarks,
   deployment footprint, failure mode, and cost profile.
2. **`notes/latency-cost-matrix.md`** — a numeric table of **end-to-end
   rerank latency and $/million-reranks** at three candidate-list sizes
   (N=20, N=100, N=500), with each family's dominant cost factor called out
   (GPU time, API calls, token count).
3. **`notes/decision.md`** — a one-page recommendation for the mod-107
   baseline (RAG over the mod-102 chunked corpus at ~10 QPS, interactive
   UX with a p95 < 800 ms budget, on-prem GPU available but constrained),
   with one primary pick, one escape hatch, and a measurable trigger to
   switch families.

The matrix must be *comparative*, not a feature catalog. A reviewer should
read one row and name in a sentence why that family exists as its own entry.

## 2. Worked answer or implementation

### 2.1 The five families in one paragraph each

- **Cross-encoders.** A BERT-family encoder (typically 100M–560M params)
  that takes `[CLS] query [SEP] passage [SEP]` and emits a single
  relevance logit. ms-marco-MiniLM-L-6-v2 (22M) and bge-reranker-v2-m3
  (568M) are the common anchors. Scored pointwise, one pair at a time,
  but batched to tens or hundreds of pairs per forward pass. The
  workhorse: strong quality on short passages, latency scales linearly in
  candidate count, trivially self-hosted on CPU (small) or GPU (base/large).
- **Late-interaction (ColBERT / ColBERTv2 / PLAID).** Query and document
  are each tokenised and encoded to a *bag of token-level vectors*. At
  score time the model does per-token MaxSim matching. Precomputes
  document token embeddings offline (index cost), scores at a fraction of
  cross-encoder latency because there is no joint forward pass. Shipped
  as a retrieval system in its own right (PLAID is the production
  server), but is routinely used as a reranker over a lexical first phase.
- **Listwise LLM rerankers (RankGPT, RankVicuna, RankZephyr).** A single
  LLM call receives the query plus the candidate list and returns a
  permutation (e.g., "[3] > [1] > [7] > …"). One call reranks N
  candidates. Quality is strong on judgments-sparse domains, but latency
  is dominated by the output-token count (the permutation) and by prompt
  size (N × passage tokens). Context-window pressure is real past N ≈ 20
  on 8k-context models.
- **Pointwise LLM-as-a-judge rerankers.** Score each candidate with its
  own LLM call ("rate the relevance of this passage to the query on a
  1–10 scale and return JSON"). N calls per query. Flexible — the prompt
  can incorporate business rules, freshness criteria, taxonomy — but
  the most expensive option at both latency and $/query. Suited to
  offline or low-QPS high-value flows (compliance review, enterprise
  search for executives), not to open-ended consumer QA.
- **Hosted API rerankers (Cohere Rerank, Jina Reranker, Voyage Rerank).**
  A paid endpoint takes `(query, documents[])` and returns relevance
  scores. Internally these are typically cross-encoder family models,
  but the operator never sees the weights. Trade self-hosting cost (GPU
  purchase, model loading, scaling) for a per-request bill and a network
  hop. Latency floor is the round-trip time, not the model forward pass.

### 2.2 `notes/family-matrix.md` — the comparison table

| Dimension | Cross-encoder (self-hosted) | Late-interaction (ColBERTv2) | Listwise LLM reranker | Pointwise LLM-as-judge | Hosted API reranker |
|---|---|---|---|---|---|
| **Scoring shape** | Pointwise; one logit per `(q, p)` pair. | Per-token MaxSim between query-token and passage-token vectors. | Listwise; one call returns a permutation over N candidates. | Pointwise; one LLM call per candidate returning a numeric score. | Pointwise under the hood; N scores returned in one HTTP call. |
| **Typical model size** | 22M (MiniLM) – 568M (bge-reranker-v2-m3). | ~110M (ColBERTv2 base). | 7B–70B+ (open) or proprietary (GPT-4-class). | Same as listwise. | Opaque; typically cross-encoder family. |
| **Dominant cost factor** | GPU forward-pass time, linear in N. | Index storage (token-level vectors) + MaxSim compute. | Prompt+completion tokens; one call per query. | Prompt+completion tokens; **N calls per query**. | Per-request + per-document API charge; linear in N. |
| **Batching** | Natural; one forward pass over a batch of pairs. | Natural at MaxSim scoring step. | Not applicable at a single-query level; N is the batch. | Each call independent; concurrent HTTP. | Vendor batches internally; N documents per call. |
| **Hardware** | CPU ok for MiniLM; GPU for base/large; one model, one process. | GPU preferred for encoding; MaxSim is cheap. | Dedicated GPU inference server (vLLM, TGI) or API. | Same as listwise. | None — API only. |
| **Quality anchor** | Strong on MS MARCO/BEIR; bge-reranker-v2-m3 is state-of-the-art class for its size. | Competitive with cross-encoders on BEIR at lower per-query latency. | Strong on sparse-judgment domains; permutation stability varies. | Flexible on custom criteria; expensive to validate at scale. | Vendor-reported benchmarks; verify on your own eval. |
| **Failure mode** | Degrades on passages longer than the max sequence length (often 512 tokens). | Index bloat; one vector per token blows storage past dense bi-encoder budgets. | Hallucinated ids / dropped candidates; permutation does not include all N. | Score drift across calls; same `(q, p)` can get 7 then 9. | Vendor outage; SLO is not yours. |
| **Transparency** | Open weights; you can fine-tune. | Open weights; you can retrain. | Open or closed depending on choice. | Open or closed depending on choice. | Black box. |
| **Governance** | On-prem data; no third-party call. | On-prem data. | On-prem if open model served locally. | On-prem if open model served locally. | Data leaves the perimeter; needs DPA + review. |

### 2.3 `notes/latency-cost-matrix.md` — the numeric matrix

Numbers below are **order-of-magnitude anchors** for planning, not vendor
benchmarks. Each column should be replaced by a locally measured number
before any go/no-go decision. The matrix shape is what matters: the
*relative* cost of the families at a given N.

Assumptions: short passage (~200 tokens), 1x A10/A100-class GPU for
self-hosted inference, standard batching, warm model. Latency is p50 end
to end on the rerank step only (excludes first-phase retrieval).

| Family @ N=20 | p50 latency | $/M-reranks (self-hosted, amortised) | $/M-reranks (if API) |
|---|---|---|---|
| Cross-encoder MiniLM-L-6-v2 | ~5–15 ms | ~$0.5 (shared GPU time) | n/a |
| Cross-encoder bge-reranker-v2-m3 | ~30–80 ms | ~$3 | n/a |
| ColBERTv2 | ~5–15 ms (plus index) | ~$1 + storage | n/a |
| Listwise LLM (7B open) | ~400–1200 ms | GPU-bound; ~$5–15 | n/a |
| Pointwise LLM-as-judge (7B open) | ~400–1500 ms (parallel) | ~$10–30 | n/a |
| Hosted API reranker | ~100–250 ms (round-trip) | n/a | ~$1–3 per 1k queries at N=20 |

| Family @ N=100 | p50 latency | $/M-reranks | $/M-reranks (API) |
|---|---|---|---|
| Cross-encoder MiniLM-L-6-v2 | ~20–60 ms | ~$2 | n/a |
| Cross-encoder bge-reranker-v2-m3 | ~150–400 ms | ~$15 | n/a |
| ColBERTv2 | ~15–40 ms | ~$3 | n/a |
| Listwise LLM (7B open) | ~2000–6000 ms (prompt bloat) | ~$30–100 | n/a |
| Pointwise LLM-as-judge (7B open) | ~2000–8000 ms (5× N) | ~$60–150 | n/a |
| Hosted API reranker | ~200–500 ms | n/a | ~$5–15 per 1k queries |

| Family @ N=500 | p50 latency | $/M-reranks | Viable? |
|---|---|---|---|
| Cross-encoder MiniLM-L-6-v2 | ~100–300 ms | ~$10 | Yes (interactive ok). |
| Cross-encoder bge-reranker-v2-m3 | ~700 ms–2 s | ~$80 | Borderline — needs GPU batching. |
| ColBERTv2 | ~60–150 ms | ~$15 | Yes. |
| Listwise LLM (7B open) | Fails context budget on 8k models. | n/a | **No** without chunked listwise. |
| Pointwise LLM-as-judge (7B open) | Parallelism-bound; seconds. | ~$300+ | **Offline only.** |
| Hosted API reranker | ~1–3 s | n/a | ~$25–75 per 1k queries. |

**Reading the matrix:**

- Self-hosted cross-encoders (small) are the only family that scales
  linearly to N=500 inside an 800 ms interactive budget.
- Listwise LLM rerankers hit a prompt-size wall before latency becomes
  the binding constraint.
- Pointwise LLM-as-judge is the most expensive on every row and only
  defensible when the judging prompt is doing work a cheaper model
  cannot (policy enforcement, multi-aspect scoring).
- Hosted APIs are competitive at N=20–100 and let a team skip the GPU
  purchasing decision, but the network round-trip is the floor.

### 2.4 `notes/decision.md` — recommendation for the mod-107 baseline

**Baseline assumptions.** Interactive RAG over the mod-102 chunked corpus;
first-phase hybrid retrieval (mod-106) returns 50–100 candidates per
query; target QPS ~10; p95 end-to-end budget 800 ms of which ~300 ms is
available for reranking (the remainder spans retrieval, context assembly,
and LLM generation); single-region on-prem deployment with one A10-class
GPU dedicated to retrieval.

**Primary pick: self-hosted cross-encoder (bge-reranker-v2-m3) at N=50.**
Reasons:

1. 50 candidates × ~5 ms per pair batched = ~200–250 ms, inside the 300 ms
   budget with headroom.
2. Open weights, fine-tunable on judgments from mod-109 later.
3. Multilingual and strong on BEIR / MIRACL — matches a mod-105/107 corpus
   that is unlikely to be English-only in production.
4. Deploys as a Transformers / sentence-transformers service with no new
   vendor to onboard.
5. Does not send query text out of the perimeter.

If GPU budget forbids a 568M model, downgrade to
**ms-marco-MiniLM-L-6-v2** at N=100 and accept slightly weaker quality
(and still inside budget). The *shape* of the decision does not change.

**Escape hatch: hosted API reranker (Cohere Rerank / Voyage / Jina).**
Switch when any of the following becomes true:

- QPS grows past what one GPU can serve (~50 QPS at N=50 for a 568M
  cross-encoder) and horizontal scaling adds ops cost that an API bill
  avoids.
- Latency budget shrinks (e.g., mobile client) and the hosted vendor's
  edge POPs beat on-prem p95.
- Team needs multi-tenant data isolation that a single on-prem model
  cannot provide without a bigger ops investment.

**Escape hatch: ColBERTv2.** Switch when index storage is not the
binding constraint *and* the team wants the latency floor of
late-interaction without an API dependency. The migration cost is a
reindex (token-level vectors) and a new serving stack (PLAID or a
custom MaxSim implementation).

**Not recommended for this baseline:**

- **Listwise LLM rerankers.** The p95 budget (800 ms) does not
  accommodate a 7B+ LLM call in the rerank step, and prompt-size
  pressure at N=100 is real.
- **Pointwise LLM-as-judge.** The per-candidate LLM call cost is
  prohibitive at interactive QPS. Reserve it for offline judgment
  generation (mod-109) or for low-QPS, high-value queries (compliance,
  legal).

**Measurable trigger to re-evaluate.** Rerun the matrix when any of these
numbers moves by more than 2×: (a) first-phase candidate count per
query, (b) per-query budget, (c) per-GPU $/hour, (d) hosted-API
$/1k-queries, or (e) corpus language distribution.

### 2.5 Second-order considerations

- **Model drift vs. API drift.** A self-hosted cross-encoder is frozen
  at the version you deployed. A hosted reranker may silently change
  (Cohere, Voyage, and Jina have each versioned their rerankers;
  "latest" aliases can shift). Pin a dated version in production or
  record the version header per request for audit.
- **Evaluation contract.** Whichever family wins, measure under the
  same query mix that mod-109 will evaluate on. A reranker that
  improves NDCG@10 on short-answer queries but regresses on
  comparison-style queries is not necessarily an improvement.
- **Observability.** Log the first-phase rank and the reranked rank of
  the final document. The rank delta is what a reranker *buys* you;
  without it, "the reranker helped" is unfalsifiable. mod-109 covers
  the eval harness, but the logging must be live from day one.

## 3. Validation steps

1. The learner can name all five families **without the matrix**, and
   state in one sentence why each one exists as a separate entry (not a
   variant of another).
2. The latency column is computed as `N × per-pair latency` for pointwise
   families and `one call` for listwise/API rerankers. A matrix that
   quotes a flat "p50 = 200 ms" for every N has not been audited.
3. The cost column distinguishes **$/M-rerank** (what the team pays) from
   **$/1k-queries** (what the user-facing cost model sees). These
   collapse only when N is fixed; the matrix must carry at least one of
   each.
4. The decision document names one *condition* under which the choice
   changes, not "it depends." The condition is measurable (QPS,
   latency budget, $/GPU-hour, API $/1k) rather than aesthetic.
5. The learner can trace each row to a source: Cohere/Voyage/Jina
   pricing pages for the API column, open-weight model cards for the
   self-hosted columns, published latency benchmarks for ColBERTv2.
6. The matrix uses the **first-phase recall set size** as its N axis,
   not an arbitrary range. 20 / 100 / 500 are defensible anchors; 7 or
   13 are suspicious.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Family taxonomy | All five families appear. Distinct scoring shapes identified: pointwise logit, late-interaction MaxSim, listwise permutation, LLM-as-judge pointwise, API pointwise. |
| Cost model | Latency is written as a function of **N** (candidate count), not a scalar. $/M-rerank and $/1k-queries are both present. |
| Listwise LLM reality check | The document explicitly names the prompt-size failure mode at N ≥ 100 on 8k-context models — it is not treated as free. |
| LLM-as-judge framing | Called out as *offline / low-QPS high-value only* at interactive QPS. |
| Hosted-API framing | Round-trip latency and vendor-drift risk are both mentioned. Data-exfiltration and DPA concerns are mentioned. |
| Cross-encoder realism | Distinguishes small (MiniLM-class) from base/large (bge-reranker, Jina Reranker v2) with different latency/quality trade-offs. |
| ColBERT realism | Called out as having an **index cost** (one vector per token). If the matrix lists ColBERT at zero storage overhead, it is wrong. |
| Recommendation | Picks one family, names one escape hatch (preferably two), names one measurable trigger for switching, and does not default to "whichever is cheapest on paper." |
| Observability | Mentions logging the first-phase vs. reranked position so the impact is measurable in production. |
| Governance | Mentions data-residency / third-party-processor consideration for the hosted-API option. |

Fail-fast smells:

- A matrix that lists listwise LLM rerankers at N=500 as "fast." It is
  not — most open 7B models fail the context budget, and the ones that
  do not pay for it in output-token latency.
- A recommendation that picks an API reranker without naming the
  round-trip floor.
- Treating Cohere / Voyage / Jina as interchangeable. They publish
  different model sizes, different language coverage, different
  pricing tiers. The matrix should name which API option the
  recommendation assumes.
- A decision written as "cross-encoders are best" without the
  per-candidate latency math. "Best" is a function of N; if N doubles,
  the decision may flip.

## 5. Common mistakes

- **Flat latency numbers.** Quoting "the reranker is ~100 ms" without
  the N it was measured at. Every pointwise reranker's latency scales
  in N; a 100 ms measurement at N=20 is 500 ms at N=100 unless
  batching saturates the GPU first.
- **Omitting index cost for ColBERT.** ColBERTv2 stores one vector per
  token (reduced via centroids + residuals in the ColBERTv2 recipe,
  but still orders of magnitude bigger than a bi-encoder index). A
  matrix that ignores this misprices the family.
- **Conflating listwise and pointwise LLM rerankers.** They share
  "LLM" in the name but have different cost shapes: listwise is one
  call per query; pointwise is N calls per query. Treating them as
  a single row collapses a 20× cost gap.
- **Using vendor benchmarks as the quality anchor.** Every vendor
  publishes numbers where their model wins. Use at least one
  independent benchmark (BEIR, MIRACL, LoTTE) or internal eval.
- **Forgetting the first-phase gate.** A reranker can only promote
  what retrieval surfaces. Judging a reranker on queries where the
  first phase missed the gold doc tells you nothing about the
  reranker.
- **Treating the API round-trip as free.** The vendor's model may be
  fast (10 ms on their GPU), but the round-trip adds 50–200 ms every
  request. On an 800 ms budget, that is a quarter to a third of the
  budget gone to network.
- **Ignoring data governance.** Sending query text and candidate
  passages to a third-party reranker is a cross-border data flow in
  many regulated settings (HIPAA, GDPR, national-security). The
  matrix should mark hosted-API rows as "data leaves perimeter."
- **Picking by quality alone.** Reranker selection is a joint
  decision over quality **and** latency **and** cost **and**
  governance. A model that wins NDCG by 1 point but costs 10× the
  previous one is not obviously the right choice.

## 6. References

- Cross-encoders:
  - Nogueira & Cho (2019), "Passage Re-ranking with BERT" —
    [arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085).
  - Sentence-Transformers cross-encoder guide —
    [sbert.net/examples/applications/cross-encoder/README.html](https://sbert.net/examples/applications/cross-encoder/README.html).
  - BAAI `bge-reranker-v2-m3` model card —
    [huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).
  - `ms-marco-MiniLM-L-6-v2` model card —
    [huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
- Late interaction:
  - Khattab & Zaharia (2020), "ColBERT: Efficient and Effective Passage
    Search via Contextualized Late Interaction over BERT" —
    [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832).
  - Santhanam et al. (2022), "ColBERTv2: Effective and Efficient Retrieval
    via Lightweight Late Interaction" —
    [arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488).
  - Santhanam et al. (2022), "PLAID: An Efficient Engine for Late
    Interaction Retrieval" —
    [arxiv.org/abs/2205.09707](https://arxiv.org/abs/2205.09707).
- LLM-based rerankers:
  - Sun et al. (2023), "Is ChatGPT Good at Search? Investigating Large
    Language Models as Re-Ranking Agent" (RankGPT) —
    [arxiv.org/abs/2304.09542](https://arxiv.org/abs/2304.09542).
  - Pradeep et al. (2023), "RankVicuna: Zero-Shot Listwise Document
    Reranking with Open-Source Large Language Models" —
    [arxiv.org/abs/2309.15088](https://arxiv.org/abs/2309.15088).
  - Pradeep et al. (2023), "RankZephyr: Effective and Robust Zero-Shot
    Listwise Reranking is a Breeze!" —
    [arxiv.org/abs/2312.02724](https://arxiv.org/abs/2312.02724).
- Hosted API rerankers (pricing and model families):
  - Cohere Rerank — [docs.cohere.com/docs/rerank](https://docs.cohere.com/docs/rerank).
  - Voyage Rerank — [docs.voyageai.com/docs/reranker](https://docs.voyageai.com/docs/reranker).
  - Jina Reranker — [jina.ai/reranker/](https://jina.ai/reranker/).
- Benchmarks:
  - BEIR — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - MIRACL — [project-miracl.github.io/](https://project-miracl.github.io/).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
