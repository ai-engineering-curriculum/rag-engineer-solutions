# Exercise 04 — Solution: Read Lewis 2020 and Gao 2024 and annotate them

This is a reference solution for `exercise-04-read-lewis-2020-and-gao-2024-and-annotate.md`. The
three deliverables (A, B, C) are reference annotations against the templates the exercise
provides. The point of the exercise is to produce annotations the *learner* will come back to
later; the reference below is a model, not the only correct answer.

Where a specific figure number, section number, or experimental detail cannot be verified without
re-opening the paper at solution-writing time, the solution inlines a `<!-- verify-against-source: ... -->`
marker. These markers are the contract: a learner producing their own annotation would fill them
in by reading, and a reviewer scoring the solution would check that the learner's filled-in
version is consistent with the paper.

## 1. Solution overview

Chapter 5 of this module (*The Two Papers Every Retrieval Engineer Should Have Read*) is the
reading guide; the exercise asks the learner to turn the guide into two structured annotations
plus a one-page synthesis placing a specific project on Gao 2024's Naïve / Advanced / Modular
taxonomy. We work through all three deliverables.

Paper identifiers used below (fixed; verify against the current arXiv copy):

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.*
  v5, March 2024. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).

## 2. Worked answer or implementation

### Deliverable A — `annotation-lewis-2020.md`

```markdown
# Annotation — Lewis et al. 2020, RAG for Knowledge-Intensive NLP

Contribution in one sentence:
  A dense passage retriever (DPR-style) is combined with a pretrained seq2seq
  generator (BART) at inference, with the retrieved documents treated as latent
  variables marginalized over the generator's output distribution — naming and
  operationalizing the pattern now called retrieval-augmented generation.

RAG-Sequence vs. RAG-Token in one paragraph each:
  RAG-Sequence: the retriever's top-K documents are each conditioned on
    independently; for each retrieved document z, the generator computes the full
    output sequence's probability, and the final probability is the marginal
    across the K documents (p(y|x) = Σ_z p_retr(z|x) · p_gen(y|x, z)). One
    document influences the whole output sequence. This shape fits tasks where
    one retrieved passage is plausibly sufficient to ground the entire answer —
    open-domain QA with short answers is the canonical case.
  RAG-Token: the marginalization is per output token. At each generation step,
    the model can "re-weight" which retrieved document is most responsible for
    the next token (p(y|x) = Π_i Σ_z p_retr(z|x) · p_gen(y_i|x, z, y_<i})).
    Different tokens can be attributed to different retrieved documents. This
    shape fits tasks where an answer draws from several sources — long-form
    generation and some abstractive-summarization setups.

The one figure I would put in a design doc:
  Figure 1 — the architecture diagram (query encoder + retriever + seq2seq
  generator + marginalization). Why: it is the single clearest picture of the
  decoupled retriever-generator interface that is still the assumed shape
  across the industry.

The one experimental result I would reproduce (or defend against reproducing today):
  The paper's open-domain QA results on Natural Questions / TriviaQA /
  WebQuestions / CuratedTREC showing that RAG beats a parametric-only seq2seq
  baseline of comparable parameter count and that updating the non-parametric
  index without retraining shifts the model's behavior appropriately.
  <!-- verify-against-source: cite exact table and metric deltas from Section 4 -->
  Why: the demonstration that the index is independently updatable is the
  property most often misunderstood by product stakeholders today, and a tiny
  reproducible version of that experiment is worth a half-day of design-review
  hand-waving avoided.

The one limitation I would call out in a review:
  The paper's retriever is trained once and frozen at inference; the dense-vs-
  lexical retrieval story and the hybrid-retrieval story are not addressed. In
  2020 that was reasonable; in 2026 it is a visible gap — BM25 remains a strong
  baseline that fuses well with dense retrieval (addressed in mod-106), and this
  paper's framing does not acknowledge that trade-off.

The one open question I would follow up on:
  How sensitive is the quality claim to the retriever quality? If the DPR
  retriever recall@10 drops by X points, how much of the downstream answer
  quality is recovered by RAG-Token's per-token re-weighting versus RAG-Sequence?
  This question motivates the rerank-stage decisions this track covers in mod-107.

Where each of the seven pipeline stages (from chapter 1) appears in the paper — cite section or figure:
  Ingest:    out of scope in the paper; the corpus is Wikipedia, already
             pre-processed and chunked by the DPR pipeline.
  Chunk:     out of scope; Wikipedia chunks are 100-word passages following DPR
             (Karpukhin et al. 2020). The paper inherits DPR's chunking and does
             not revisit the decision.
             <!-- verify-against-source: confirm chunk size from DPR section or
                  appendix reference -->
  Embed:     Section 2 (Methods) — the retriever uses DPR's bi-encoder; the
             query encoder is a frozen BERT-base, the document encoder is
             pre-trained and the document embeddings are precomputed once.
  Retrieve:  Section 2 — dense inner-product search over the precomputed
             document embeddings using FAISS (MIPS, maximum inner-product search).
  Rerank:    NOT COVERED in this paper. In 2020, the strong cross-encoder /
             LLM-reranker literature was earlier (MonoBERT, MonoT5) but was not
             combined with the RAG shape yet. This is depth-in mod-107.
  Assemble:  implicit — the retrieved K documents are concatenated with the
             query before being passed to the generator. The paper does not
             treat dedup, ordering, or compression as design knobs. In 2020 this
             was defensible because K ≤ 10 and context windows were short.
  Generate:  Section 2 — a BART-Large generator (around 400M params) conditions
             on the (query, retrieved-doc) pair. The citation-forcing / refusal
             / contradiction-handling policies that mod-108 treats as first-
             class concerns are not part of the paper's prompt shape.

  Which of the seven stages does the paper NOT cover, and why does that make sense
  given the paper's year?

    Rerank and Assemble are the two stages the paper does not cover in the depth
    chapter 1 treats them. In 2020, context windows were small (BART's effective
    input was on the order of a few thousand tokens; GPT-3 was contemporaneous
    and had its own constraints), K was small, and "lost in the middle" was
    years away from being documented. Treating rerank and assemble as separate
    stages would have been over-engineering in that environment. In 2026 both
    are first-class — rerank because candidate-pool sizes have grown past the
    generator's attention budget, and assemble because ordering, dedup, and
    citation anchors are what make the generator's output trustworthy. This
    is exactly why mod-107 is a separate module.
```

### Deliverable B — `annotation-gao-2024.md`

```markdown
# Annotation — Gao et al. 2024, RAG for LLMs: A Survey

The three-era taxonomy in one line each:
  Naïve RAG:    one-shot retrieve-then-generate with no rewriting and no
                reranking — the demo-stage shape.
  Advanced RAG: retrieve-then-generate with pre-retrieval (query rewriting,
                chunk optimization, metadata attachment) and post-retrieval
                (reranking, prompt compression, context re-ordering) steps.
  Modular RAG:  swappable modules — routers, memory, adapters, fusion — with
                iterative / recursive / adaptive control flow; the shape
                agentic systems consume.

The single figure I would drop into a design doc:
  Figure 3 — the Naïve / Advanced / Modular pipeline comparison diagram.
  <!-- verify-against-source: confirm figure number against arXiv v5; the chapter's
       chapter 5 refers to "Section 2" top-level pipeline diagram -->
  Why: in one picture it lets a non-RAG reviewer locate any proposal on the
  spectrum without having to read a paragraph.

Two subsections I read in depth (name and section number):
  1. Section 3.2 (Indexing structures) — because the module I am about to work
     on (mod-104) is the vector-database-and-ANN-tuning module, and the survey's
     taxonomy of indexing structures (hierarchical, graph-based, HNSW, IVF-PQ)
     is the shortest path to the vocabulary I will reach for there.
     <!-- verify-against-source: confirm Section 3.2 subheading is "Indexing structures"
          in arXiv v5 -->
     What I took away in one paragraph:
       The survey frames index choice as a joint decision over corpus scale,
       query QPS, update latency, and memory budget — not as "pick the one that
       benchmarks fastest on your laptop." The mapping of HNSW parameters
       (M, efConstruction, efSearch) to the recall/latency frontier the ann-
       benchmarks.com project publishes is the industry-standard mental model,
       and this module adopts it.
  2. Section 5 (Augmentation Process) — because this is the vocabulary the
     agentic-AI engineer role (level 40) will use to describe iterative
     retrieval, and because my ownership contract with Support Copilot (exercise
     03) needs me to know what "iterative" means operationally.
     What I took away in one paragraph:
       "Iterative," "recursive," and "adaptive" retrieval are three named
       strategies in the survey. Iterative retrieves, generates a partial
       answer, uses it to re-query, and keeps going until a stop condition.
       Recursive breaks the query into sub-queries and composes the answers.
       Adaptive dynamically decides whether to retrieve at all for a given
       token. All three sit above the retriever's MCP interface, which is why
       mod-110 (my track) exposes the retriever as a tool but does not
       implement the agent loop that calls it multiple times.

The paper's split of evaluation into two layers, in your own words:
  Section 7 distinguishes retrieval-only evaluation (recall / MRR / nDCG on
  judgment sets) from end-to-end system evaluation (faithfulness, answer
  relevance, context precision/recall — the RAG-triad). The survey treats both
  as required; most production stacks do only the second layer, and that is a
  measurable gap. The boundary from this track's chapter 4: retrieval-only
  evaluation is owned by this role (mod-109); application-layer RAG-triad is
  owned by the peer `ai-eval-engineer`. Drawing the boundary cleanly is what
  lets a retrieval regression attribute to the retriever and a faithfulness
  regression attribute to the generator or the assembly, rather than ambiguously
  to "the RAG system."

The one claim in the paper I would push back on:
  The survey's framing of "Advanced RAG obsoletes Naïve RAG" — where it appears
  to be made — understates how often the right shape for a given problem is
  still Naïve. Naïve RAG is the right default for a hand-picked corpus and a
  single consumer; the shape-shift to Advanced is forced by the requirements
  chapter 3 of this module names, not by ambition.
  <!-- verify-against-source: confirm whether the survey explicitly claims obsolescence
       or whether this is a learner-interpretation artifact — Gao 2024 Section
       2 seems more descriptive than prescriptive -->

Three papers cited by Gao 2024 that I would read next:
  1. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain
     Question Answering.* EMNLP 2020. [arXiv:2004.04906] — because it is the
     direct predecessor to Lewis 2020's retriever and the model-selection
     module (mod-103) reaches for its bi-encoder vocabulary.
  2. Izacard, G., & Grave, E. (2021). *Leveraging Passage Retrieval with
     Generative Models for Open Domain Question Answering (FiD).*
     [arXiv:2007.01282] — because it is the clearest contrast with Lewis 2020
     on how retrieval and generation compose (encoder-level fusion vs.
     document-level marginalization).
  3. Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use
     Long Contexts.* TACL 2024. [arXiv:2307.03172] — because it is the
     canonical reference for why the assemble stage (mod-107) is its own
     stage with its own trade-offs.
```

### Deliverable C — `synthesis.md`

The reference synthesis places the exercise-03 scenario on the taxonomy. A learner may pick a
different project.

```markdown
# Synthesis — placing the mid-size B2B SaaS retriever on Gao 2024's taxonomy

## 1. The project

A mid-size B2B SaaS company has four retrieval corpora: public product docs
(~50k pages), internal Confluence (~200k pages, per-page ACL), internal Google
Drive (~100k documents, per-file ACL), and SFDC tickets (~2M rows, per-account
ACL). Two product teams consume retrieval — Ask Docs (a documentation
question-answering surface) and Support Copilot (an agent that retrieves from
tickets to help write support replies). The retrieval engineer's job is to
consolidate these corpora into a permission-aware retriever with a stable
interface and expose it to the agent as an MCP tool.

## 2. Where on the Naïve / Advanced / Modular spectrum is the project *today*?

**Today: Naïve RAG, for Ask Docs on the public-docs corpus only.** The current
system wires LangChain + OpenAI embeddings + pgvector top-K + a generator
prompt. There is no reranker, no query rewriting, no hybrid-with-BM25 leg, no
metadata filtering, no per-tenant isolation, and no retrieval-only evaluation.
The other three corpora (Confluence, Drive, SFDC) are not yet in the system.
The Support Copilot agent does not yet have a retriever to call.

## 3. Where does it need to sit in twelve months?

**Twelve months: Modular RAG, for all four corpora, consumed by Ask Docs and
Support Copilot.** The requirements driving the shift (chapter 3 of this
module's framing):

  - Corpus grows past a few thousand documents → yes, 2.3M+ chunks.
  - Two tenants must not see each other's chunks → yes, per-tenant ACL on
    Confluence, Drive, and SFDC.
  - A chunker / embedding change must ship without a several-day outage → yes.
  - The retriever is reused by more than one product surface → yes.
  - A launch decision has to be defended with numbers → yes.
  - The product has to answer "where did you get this?" with citations → yes.

Six of six forcing requirements are present. The shape is Advanced, trending
Modular as soon as the Support Copilot agent is live.

The Advanced components to add, in order, and why in that order:

  1. **Hybrid retrieval (BM25 + dense, fused with reciprocal-rank fusion)** —
     mod-106. Highest-leverage single change for the public-docs corpus because
     it recovers rare-term queries (form numbers, product SKUs, error codes)
     that dense-only retrieval misses.
  2. **Reranking with a cross-encoder** (`bge-reranker-base` or hosted) —
     mod-107. Cuts false-positive rate on the top-K so the generator stops
     seeing visually-plausible-but-wrong chunks.
  3. **Retrieval-only evaluation (judgment set + recall@k / MRR / nDCG + paired
     significance)** — mod-109. Without this, none of the earlier changes can
     be defended; with it, every subsequent change is auditable.
  4. **Query rewriting (HyDE for Ask Docs, step-back for SFDC)** — mod-106.
     Lower priority than evaluation because without the eval you cannot tell
     whether rewriting helped.
  5. **Context assembly with citation anchors + the citation-verification
     check** — mod-107 + mod-108. Required for Ask Docs's "answer with
     citations" UX and for Support Copilot's "quote the ticket you found."

The Modular shift happens when the Support Copilot agent consumes the
retriever via MCP and calls it multiple times per turn (mod-110). At that
point the retriever is a tool, not a function, and the control flow above the
retriever is owned by `agentic-ai-engineer`.

## 4. Which Lewis 2020 design property is load-bearing?

**The updatable non-parametric memory property is load-bearing.** Three of the
four corpora (Confluence, Drive, SFDC) are updated continuously by human
authors. Any design that bakes those documents into model weights (through
fine-tuning on the corpus) would be unworkable — the authors change the
corpus faster than any fine-tuning cadence could keep up with. The retriever
has to be a *swappable index*, not a *fine-tuned model*. Lewis 2020's
decoupled retriever-generator shape is the architectural foundation that
makes "re-index Confluence overnight and the system is current tomorrow"
feasible without model retraining. This is the exact property legal, product,
and support teams will implicitly demand; naming it is the design-doc defense
against the periodic "should we fine-tune the model on our docs" question.
The retriever-generator decoupling is the secondary load-bearing property —
it is what lets us pin the retriever as its own artifact with its own
quality / latency / cost claim (the ownership contract from exercise 03)
while leaving the generator choice to the product teams.
```

## 3. Validation steps

This is a reading / annotation exercise. Validation is review-based.

1. Open Deliverable A and tick off every section of the template. Missing any field, including
   the "where each stage appears" block with per-stage citations, fails.
2. Open Deliverable B and tick off every section of the template. The two deep-read subsections
   must be named and numbered; the one-paragraph takeaways must be concrete (not "it discusses X").
3. Confirm Deliverable A's "which stages are not covered" paragraph names at least two (rerank
   and assemble are the canonical pair, matching chapter 1's seven-stage shape being a superset
   of what Lewis 2020 formalized).
4. Confirm Deliverable B's three-era one-liners each fit on one line and are in the learner's own
   words. A direct quote from the survey is a miss.
5. Open Deliverable C and check that it names the project's taxonomy position with a stated
   *why*, a target position with a *why*, and at least one specific Advanced/Modular component
   to add with a reason for its ordering.
6. For every `<!-- verify-against-source: ... -->` marker the learner leaves in their own annotation,
   confirm they went back to the paper and resolved it before submitting.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Deliverable A template coverage | Every field in the Lewis-2020 template filled, including per-stage mapping |
| Deliverable A stages-not-covered | At least two stages the paper does not cover, with reason tied to paper year |
| Deliverable B three-era one-liners | Each ≤ one line, learner's own words, Naïve/Advanced/Modular all distinct |
| Deliverable B deep-read subsections | Two named subsections with section numbers; each with a concrete one-paragraph takeaway |
| Deliverable B two-layer evaluation | Explicit mention of the retrieval-only vs. application-layer split and the `ai-eval-engineer` boundary |
| Deliverable C project description | One paragraph; corpus, users, retriever's job |
| Deliverable C placement today | On the taxonomy, with a stated *why* |
| Deliverable C target in 12 months | On the taxonomy, with a stated *why* and at least one component to add |
| Deliverable C Lewis property | Decoupling or updatable non-parametric memory explicitly load-bearing, with reasoning |

Fail-fast smells (any one is a revise-and-resubmit):

- The "one figure I would put in a design doc" field is empty or says "the architecture diagram"
  without a figure number.
- The "one open question I would follow up on" is a generic question that does not depend on the
  paper's specific content.
- The project synthesis claims a taxonomy position without naming the forcing requirements.
- The Lewis-property reflection names "better accuracy" or similar — those are not Lewis 2020's
  load-bearing *design properties* (decoupling and updatability are).

## 5. Common mistakes

- **Reading Lewis 2020 as "the paper that proved dense retrieval beats BM25."** It does not make
  that claim. Mod-106 later shows hybrid tends to win. A learner who writes this in Deliverable A
  has misread the paper and should re-read Sections 1 and 4.
- **Reading Gao 2024 front-to-back.** The paper is a survey; the chapter 5 reading guide says
  to use the section index. Learners who read every word get to the quiz having absorbed less
  than learners who did the mapped read.
- **Collapsing the three-era taxonomy to two eras ("basic" vs. "advanced").** The Modular shape
  is the one that matters for agentic consumption, which is where the Support Copilot scenario
  in exercise 03 lives. Collapsing it away makes mod-110 harder to motivate.
- **Attributing faithfulness measurement to the retrieval engineer in Deliverable B's
  two-layer-evaluation reflection.** This is the same scope-discipline failure as exercise 03.
- **Picking RAG-Sequence as the "modern production shape" in Deliverable A.** Most production
  systems today retrieve once and condition the generator on the top-K jointly — this is closer
  to a *simplification* of RAG-Sequence (no per-document marginalization) than to RAG-Sequence
  itself. Naming it accurately is part of the exercise.
- **Treating the verify-against-source markers as optional.** They are not — the exercise expects the
  learner to go back to the paper and resolve them. A submitted annotation with a markers left
  in is unfinished.
- **Choosing a project in Deliverable C that is already cleanly Naïve and does not need to change
  shape.** The synthesis exercise is more productive on a project where the taxonomy position
  today and in twelve months are different, because that forces the "what Advanced/Modular
  components to add" reflection to have teeth.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-101-retrieval-engineering-foundations/05-canonical-framing-papers.md` — the
    reading guide the exercise's templates are derived from.
  - `lessons/mod-101-retrieval-engineering-foundations/01-end-to-end-rag-pipeline-and-decisions.md`
    — the seven-stage pipeline the Lewis-2020 annotation maps against.
  - `lessons/mod-101-retrieval-engineering-foundations/04-role-position-on-the-ladder.md` — the
    retrieval-only vs. application-layer evaluation boundary referenced in Deliverable B.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.*
  v5. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
  EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906).
- Izacard, G., & Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open
  Domain Question Answering (FiD).* EACL 2021. [arXiv:2007.01282](https://arxiv.org/abs/2007.01282).
- Liu, N. F., et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL
  2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).
