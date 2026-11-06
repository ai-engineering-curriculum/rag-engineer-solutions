# Exercise 01 — Solution: Map the RAG pipeline and its decision points

This is a reference solution for `exercise-01-map-the-rag-pipeline-and-decision-points.md`. It
exists so a learner can self-grade and so a reviewer has a rubric to score against. The exercise
is a *design* exercise — correctness is defended by reasoning, not by test output. The three
deliverables (A, B, C) are worked below.

## 1. Solution overview

The exercise asks for three artifacts tied to chapter 1 of the module:

- **Deliverable A** — a pipeline diagram with the seven production stages (ingest → chunk → embed
  → retrieve → rerank → assemble → generate), each annotated with five items (purpose, offline/
  online, the engineering decision, two tunable knobs, the owning module), plus two cross-cutting
  boxes for evaluation and MLOps.
- **Deliverable B** — a table that classifies eight failure scenarios by stage, first knob to turn,
  and ownership.
- **Deliverable C** — a one-page pipeline stanza suitable for a real project's `README.md`,
  committing to a concrete default per stage.

The chapter the exercise draws from is the module's `01-end-to-end-rag-pipeline-and-decisions.md`
lesson. Numbers for the HR parental-leave scenarios are illustrative constants supplied by the
exercise itself; this solution does not invent additional metrics.

## 2. Worked answer or implementation

### Deliverable A — Pipeline map (worked annotation table)

A learner's diagram is scored against the table below, which re-presents chapter 1's seven stages
with the five required annotations plus the two cross-cutting boxes. Any arrangement (left-to-right
flow, top-to-bottom, radial) is acceptable as long as every row is covered.

| Stage | Purpose (one sentence) | Mode | The engineering decision | Knob 1 | Knob 2 | Depth module |
|---|---|---|---|---|---|---|
| **Ingest** | Turn source bytes into structured passages with metadata. | Offline | Which parser per format and which structural + governance metadata to preserve. | Parser-per-format selection (e.g. PyMuPDF vs. Docling vs. OCR fallback) | Which governance fields to attach (`permission_scope`, `updated_at`, `source_url`) | mod-102 |
| **Chunk** | Split documents into retrievable passages. | Offline | The boundary policy and how it interacts with embedding context, reranker cost, and generator budget. | Chunk size (tokens or characters) | Overlap (tokens) | mod-102 |
| **Embed** | Map each passage to a comparable vector. | Offline | Which embedding family and model, defended by a corpus-fit benchmark, at what dimensionality / quantization. | Embedding model ID (e.g. `BAAI/bge-small-en-v1.5` vs. a hosted API) | Dimensionality / quantization regime (full-precision vs. int8 vs. binary vs. Matryoshka truncation) | mod-103 |
| **Retrieve** | Turn a query into a candidate list. | Online | Index choice, ANN parameters, lexical/dense/hybrid shape, and metadata-filter composition. | `efSearch` (HNSW) or `nprobe` (IVF) | Top-*k* (candidate pool size handed to the reranker) | mod-104, mod-105, mod-106 |
| **Rerank** | Rescore candidates to a shorter ordered list. | Online | The reranker type, and the *N* → *K* funnel shape against a latency and cost budget. | Reranker model (e.g. `bge-reranker-large` vs. a hosted reranker) | *N* in (candidates) and *K* out (short-list) | mod-107 |
| **Assemble** | Turn the short-list into a prompt-ready context block. | Online | Dedup / ordering / citation anchoring / compression and the reserved answer budget. | Deduplication policy (by `doc_id`, by content-hash, by near-duplicate threshold) | Per-chunk compression ratio or max-tokens-per-chunk | mod-107 |
| **Generate** | Turn assembled context into a grounded, cited answer. | Online | The citation-forcing prompt shape, refusal policy, contradiction policy, and the post-generation citation-verification check. | Prompt shape (inline `[cite: ...]` vs. structured JSON citations) | Refusal threshold / policy when context is insufficient | mod-108 |

Cross-cutting boxes that must appear **around** the seven-stage flow, not inside it:

- **Evaluation** — judgment set, retrieval-only metrics (`recall@k`, `precision@k`, `MRR`, `nDCG@k`),
  paired significance testing. Depth: mod-109.
- **MLOps / orchestration** — index versioned on
  `(embedding-model, chunker-version, schema-version, corpus-snapshot)`, retriever CI on PRs that
  touch chunker / embedding / retriever / reranker code, ingestion DAG with backfill and lag SLOs,
  drift alerts, MCP exposure of the retriever as a tool. Depth: mod-110 and mod-111.

### Deliverable B — Eight failure cases, classified

Two to three lines of reasoning per row. "Earliest broken stage" is the tiebreaker when more than
one stage is defensible.

| # | Scenario (summary) | Stage | First knob to turn | Who owns it | Reasoning |
|---|---|---|---|---|---|
| 1 | Correct HR-14 chunk scored 0.31 vs. a wrong chunk at 0.44; BM25 is not used. | Retrieve (and possibly Chunk) | Add a lexical leg (BM25) and fuse with RRF; if query→title mismatch persists, inspect the chunk's section heading and consider a chunker change that keeps the form number in-chunk. | rag-engineer (this role) | Dense-only retrieval misses rare-term exact matches like `HR-14`. Adding BM25 is the cheapest fix; hybrid is depth-in mod-106. |
| 2 | Top-3 contains the right chunk; generator fabricates `hr@company.com`. | Generate | Rewrite the prompt to force per-claim `[cite: chunk_id]` anchors, refuse when context is missing, and add a post-generation citation-verification check. | rag-engineer owns the retrieval-specific prompt slice (mod-108); the baseline product-surface prompt is `applied-ai-engineer`'s. | The retrieved chunk was present, so retrieval is fine; the fabrication is a generation-grounding bug (mod-108). |
| 3 | Correct chunk + a superseded 2019 draft both retrieved; LLM quotes the stale one. | Ingest (earliest broken stage) | Attach and filter on `updated_at` + a `supersedes`/`superseded_by` relationship; drop or down-rank superseded docs at ingest. | rag-engineer | Defensible alternative: Assemble (ordering could prefer newer). Ingest is earlier and the real root — stale corpus is a governance-metadata gap. |
| 4 | PDF parser drops table cells; salary-band answer is incomplete. | Ingest | Swap to a table-aware parser (Docling, Unstructured.io, Azure Document Intelligence) or add a fallback for table regions; preserve cell coordinates. | rag-engineer | No knob downstream can recover text that never entered the index. Chunk/retrieve/rerank all look innocent in isolation. |
| 5 | p95 latency jumped after 500 K → 5 M chunks; recall unchanged; vector-DB CPU not saturated. | Retrieve | Lower `efSearch` toward the recall/latency frontier; profile ANN build parameters (`M`, `efConstruction`); check whether the bottleneck is actually the embedding of the incoming query, not the ANN scan. | rag-engineer; escalate to `ai-infra-senior-engineer` if platform-scale serving changes are required. | CPU-unsaturated latency growth with stable recall points at the ANN scan or the online embed call, not the model. Unchanged recall rules out a correctness regression. |
| 6 | Tenant A sees a Tenant B chunk; ACL check happens *after* retrieval. | Retrieve | Move ACL enforcement *before* scoring: index `tenant_id` as a filter field and use a pre-filter so forbidden chunks are never candidates. | rag-engineer owns the mechanics (mod-112); `ai-risk-engineer` owns the methodology; Legal/Compliance sign off. | Post-filter after top-*k* is a classical correctness trap — pre-filter composition is the fix. |
| 7 | Recall@10 went 0.71 → 0.79 after embedding swap; complaints about "subtly wrong" answers rose. | Rerank or Assemble, then Evaluation | Measure faithfulness / answer-relevance at the application layer; inspect assemble-stage ordering and compression. | `ai-eval-engineer` owns application-layer eval (RAG-triad); rag-engineer owns the retrieval-only recall metric. | Classic recall-up / faithfulness-down divergence: better coverage + unchanged ordering/compression can surface subtly off-topic chunks. The evaluation box is the one that caught (or should catch) this. |
| 8 | Every third redeploy fails with `vector dimension 384 does not match index dimension 768`. | MLOps (cross-cutting) | Enforce index versioning keyed on `(embedding-model, chunker-version, schema-version, corpus-snapshot)` so the service refuses to start against a mismatched index. | rag-engineer (mod-111); peers at `ai-infra-mlops` share the CI/CD pipeline. | This is not a stage bug — it is an operational-contract bug. An index dimension is a versioned artifact; CI should reject a model/index mismatch before deploy. |

Of the eight, cases 1, 2, 4, 5, 6, 8 have an unambiguous correct stage. Cases 3 and 7 are the ones
where a defensible alternative exists (Assemble vs. Ingest for case 3; Retrieve, Rerank, or
Evaluation for case 7); either answer with reasoning passes acceptance.

### Deliverable C — Pipeline stanza (pastable into a `README.md`)

Target scenario stated in the exercise: a new project, 5 M chunks, mixed HTML + PDF, no ACL
requirements yet.

```markdown
## Retrieval pipeline — initial design

We ship a seven-stage retrieval pipeline. Defaults below are starting points for a
corpus of ~5 M chunks mixed HTML + PDF with no per-tenant ACL requirement yet;
each default will be defended against alternatives in the module that owns the stage.

1. **Ingest.** Docling for PDF with a Tesseract OCR fallback; `unstructured.io`
   for HTML. One JSON row per section with `{doc_id, section_path, page,
   updated_at, source_url, text}`.
2. **Chunk.** Section-aware primary boundary; sections longer than 512 tokens
   are split at paragraph boundaries with 64-token overlap. Tables are chunked
   whole with a one-sentence summary attached.
3. **Embed.** `BAAI/bge-small-en-v1.5` (384-dim) as the local baseline; a
   domain-fit benchmark will replace it in mod-103.
4. **Retrieve.** `pgvector` with an HNSW index (`vector_cosine_ops`, `M = 16`,
   `efConstruction = 64`, `efSearch` tuned to the recall/latency target in
   mod-104). Lexical leg: BM25 on an Elasticsearch index over the same chunks;
   the two legs are fused with reciprocal-rank fusion.
5. **Rerank.** `BAAI/bge-reranker-base` as a cross-encoder rescoring the top
   100 candidates to the top 8.
6. **Assemble.** Dedup by `doc_id`, order by rerank score, each chunk carries
   a `[cite: doc_id#section]` anchor. Answer budget is reserved at
   context-window-minus-25%.
7. **Generate.** A citation-forcing prompt; the model is instructed to answer
   only from the provided context, cite every claim with `[cite: ...]`, and
   refuse when the context is insufficient. A post-generation check verifies
   every emitted citation anchor matches a chunk that was actually in the
   assembled context.

**Known gaps (in this initial design).**
- Retrieval-only evaluation (judgment set, recall@k, MRR, nDCG@k, paired significance)
  is not yet built; mod-109 owns it and we will not sign off on a launch without it.
- Retrieval MLOps — index versioning on
  `(embedding-model, chunker-version, schema-version, corpus-snapshot)`, retriever
  CI on PRs that touch retrieval code, ingestion DAG with backfill, drift alerts —
  is not yet built; mod-111 owns it and we will not run on-call without it.
```

Concrete defaults, not "a vector DB." Both known-gaps items are the two cross-cutting boxes from
chapter 1 — evaluation and MLOps — rather than filler.

## 3. Validation steps

This is a design exercise; validation is review-based.

1. Check that the diagram in Deliverable A contains all seven stages and that each has every row
   of the required five annotations present (purpose / mode / decision / two knobs / module). A
   stage missing any one annotation fails that row.
2. Check that the two cross-cutting boxes (Evaluation and MLOps) are drawn **outside** the
   seven-stage flow, not treated as an eighth and ninth stage inline.
3. Re-run Deliverable B through the Who-owns-it column against chapter 4's ownership contract. Any
   case assigned purely to a peer role (e.g. "the eval team owns it") without naming the retrieval
   engineer's role in the response is a miss — the retrieval engineer always has *something* to do,
   even when the fix is largely a peer's.
4. Open Deliverable C and confirm it names a concrete technology per stage (vendor or library name)
   rather than a category. The two known-gaps items must be genuine cross-cutting boxes, not
   placeholder text.

## 4. Rubric or review checklist

Pass thresholds:

- **Deliverable A** — all 7 stages × 5 annotations = 35 cells filled, plus 2 cross-cutting boxes.
  A single missing cell is a comment-and-return. Three or more missing cells fails.
- **Deliverable B** — 6 of 8 cases identify the correct stage; the remaining 2 are allowed a
  defensible alternative (cases 3 and 7 from the table above are the natural candidates). Every
  row must state the first knob to turn and the owning role.
- **Deliverable C** — every stage has a concrete default (not "a vector DB"); the two known-gaps
  items are the real cross-cutting boxes (evaluation and MLOps) and not filler.

Reviewer prompts for narrative-style submissions:

- *Does this diagram survive translation to a whiteboard?* If a learner can redraw it in two
  minutes from memory with the same five annotations per stage, the map has become theirs.
- *Does the Deliverable B reasoning point at a knob, not a rewrite?* "Swap the embedding model"
  for case 1 before trying hybrid retrieval is a smell — the first-knob discipline is part of the
  exercise.
- *Does Deliverable C read like a design-doc stanza, or a bullet list of defaults?* The commitment
  is "we would default to X here"; a stanza that hedges every default fails the discipline.

## 5. Common mistakes

- **Collapsing rerank into retrieve.** The level-20 shape in `llm-application-developer` does not
  have a rerank stage; learners carrying that shape forward often draw six stages, not seven. The
  level-30 production shape explicitly separates rerank from retrieve because the trade-offs
  (latency budget, model choice, *N* → *K* funnel) differ.
- **Collapsing assemble into generate.** "Concatenate the top-*K* and call the LLM" is the shape
  that silently loses citation anchors and silently trips the lost-in-the-middle effect
  (Liu et al. 2024, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)). Assemble is its own
  stage because the assembly decisions are their own trade-offs.
- **Drawing evaluation as a stage.** Evaluation is not in the serving path; it runs offline over
  judgment sets. It belongs outside the seven-stage flow.
- **Case-1 misdiagnosis.** Writing "swap the embedding model" as the first knob is the most common
  single-item miss. The cheaper, defensible first move when exact tokens like `HR-14` matter is to
  add a BM25 leg and fuse — see the mod-106 hybrid-retrieval story.
- **Over-claiming ownership.** Writing "we own this" for every row in Deliverable B's `who owns
  it` column is a smell — case 7 is where the retrieval engineer hands application-layer eval
  concerns to `ai-eval-engineer`.
- **Concrete-defaults wash-out in Deliverable C.** Writing "a modern cross-encoder" instead of a
  model ID defeats the purpose. The chapter's `bge-reranker` / `ms-marco-MiniLM-*` naming is the
  level of specificity being asked for.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-101-retrieval-engineering-foundations/01-end-to-end-rag-pipeline-and-decisions.md`
    — the canonical seven-stage picture used in Deliverable A.
  - `lessons/mod-101-retrieval-engineering-foundations/03-retrieval-as-feature-vs-engineered-system.md`
    — the feature-shape vs. engineered-system-shape split used in Deliverable B ownership calls.
  - `lessons/mod-101-retrieval-engineering-foundations/04-role-position-on-the-ladder.md` — the
    ownership contract used for the "who owns it" column.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.*
  v5. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL
  2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — cited for the lost-in-the-middle
  effect in the Common Mistakes section.
- pgvector project — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
- Elasticsearch install docs — [elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
