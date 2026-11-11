# Exercise 01 — Solution: Bi-encoder, Cross-encoder, ColBERT, SPLADE — one query, four shapes

This is a reference solution for
`exercise-01-bi-cross-colbert-splade-family-map.md`. It is a **design** exercise. The
learner builds a family map that places four different retrieval model shapes
(bi-encoder, cross-encoder, ColBERT late-interaction, SPLADE learned-sparse) on the same
query and explains where each one belongs in a real retrieval stack. The reference below
is a worked example, decision rationale, and grading rubric — not a generic survey paper.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/family-map.md`** — a table that answers, for each family, five questions:
   query representation, document representation, scoring function, indexable? (yes/no),
   and typical stack role (first-stage recall / re-ranker / both).
2. **`notes/one-query-trace.md`** — the same query (`"How do I rotate a KMS key without
   downtime?"`) walked through each of the four shapes, with concrete references to the
   artifacts each family produces (one vector vs. per-token vectors vs. a sparse
   dictionary) and what the scoring step actually computes.
3. **`notes/decision.md`** — a one-page recommendation that says *for this stack*
   (chapter 7's single-corpus RAG baseline) which combinations we would actually deploy,
   which we would not, and why.

The exercise is not asking the learner to pick a winner. Each family is strong somewhere;
the point is to be able to name that *somewhere* in one sentence and back it with the
underlying mechanic. This solution does that explicitly so a reviewer can see when a
learner has swapped two families' strengths by accident.

## 2. Worked answer or implementation

### 2.1 `notes/family-map.md`

| Family | Query representation | Document representation | Scoring | Indexable with ANN? | Typical role |
|---|---|---|---|---|---|
| **Bi-encoder (dense)** | One fixed-size vector per query, produced by the same encoder that encoded the corpus. | One fixed-size vector per passage, stored once at index time. | Cosine (or dot product) between two vectors. | Yes — any standard ANN index (HNSW, IVF, ScaNN). | First-stage recall. Cheap at query time; the whole index is a single vector table. |
| **Cross-encoder** | The query is concatenated with each candidate passage and fed through a transformer that outputs a single relevance score (`[CLS]` → scalar). | None cached. The passage is re-read at query time. | Scalar score from a joint transformer forward pass. | No — no document-side representation exists outside the model. | Re-ranker over 50–200 candidates produced by a cheaper first stage. |
| **ColBERT (late interaction)** | One contextual vector *per token* of the query. | One contextual vector *per token* of the passage, stored at index time. | For each query token, take the max dot product against any document token; sum over query tokens (MaxSim). | Yes, but needs a token-level index (PLAID in ColBERTv2) and ~10–100× more disk than a single-vector index on the same corpus. | Either strong first-stage recall *or* a mid-stage re-ranker; expensive to store, cheaper to score than a cross-encoder. |
| **SPLADE (learned sparse)** | A sparse weighted vector over the model's WordPiece/BPE vocabulary (typical sizes: tens to hundreds of non-zero terms per query). | A sparse weighted vector over the same vocabulary; often larger expansion than the query side. | Inner product between two sparse vectors, exactly like BM25 arithmetic — can run on an inverted index. | Yes — on an inverted index (Lucene/Elasticsearch/OpenSearch, or Pyserini's `impact` postings). | First-stage recall with term-level explainability; natural partner to BM25 in hybrid fusion. |

### 2.2 `notes/one-query-trace.md` — one query, four shapes

**Query.** `"How do I rotate a KMS key without downtime?"`

**Bi-encoder (e.g., `BAAI/bge-small-en-v1.5` or `intfloat/e5-base-v2`).**
- Query → one 384-dim (bge-small) or 768-dim (e5-base) unit vector.
- Each passage was embedded once, offline, into the same space.
- At query time: ANN search returns the top-k passages by cosine. The question
  "without downtime" and the passage "zero-downtime rotation requires alias updates" live
  close in the embedding space because the encoder was trained on a semantic-similarity
  objective — the lexical overlap is weak but the semantic one is strong.
- Failure mode: the encoder has not seen your in-house phrasing. "key rotation" and
  "KMS alias update" may not be adjacent in the embedding space if the pre-training
  corpus never co-occurred them.

**Cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`).**
- The query and *each* candidate passage are tokenized together as
  `[CLS] query [SEP] passage [SEP]`.
- The transformer attends across both strings; the pooled `[CLS]` output becomes a
  scalar relevance score.
- There is no "document index." You cannot ask a cross-encoder "what are the top-k docs"
  without comparing against every one — that's O(N) transformer forward passes. You run
  it over the first stage's shortlist (50–200 items) and re-rank.
- Why it works: the joint attention sees `"rotate"` in the query line up against
  `"alias update"` in the passage even when their standalone vector representations are
  far apart. The dense bi-encoder cannot do this; its two encoders never see each other.

**ColBERT / ColBERTv2.**
- Query → N query-token vectors (`[Q] rotate kms key without downtime` → ~6 vectors
  after WordPiece tokenization).
- Each passage → M passage-token vectors, stored at index time. Typical passage counts
  are 100–200 vectors per passage; each vector is 128-dim in ColBERTv2.
- Scoring: for every query-token vector `q_i`, find the single passage-token vector
  `d_j` with the highest dot product. Sum those max dot products over `i`. This is
  "late interaction" — you preserve the token-level signal that a bi-encoder's pool
  throws away, but you still do not run a full transformer at query time.
- Index: PLAID (ColBERTv2) centroid-partitions the token vectors and only expands the
  nearest centroids at query time. Storage is larger than a single-vector index by a
  factor roughly equal to the average tokens-per-passage divided by any compression
  (ColBERTv2 uses 2-bit residual compression to make this tractable).

**SPLADE / SPLADEv2.**
- Query → a sparse vector over the vocabulary. The non-zeros are tokens the model
  learned to associate with the query — including tokens that never appeared in the
  query string. For our query, SPLADE will likely light up tokens like
  `rotation`, `aws`, `alias`, `envelope`, `downtime`, `rotate`, `kms` with a
  learned weight on each.
- Each passage → a sparse vector over the same vocabulary, with similar learned
  expansion.
- Scoring: inner product of the two sparse vectors. On an inverted index this is
  identical in shape to BM25 arithmetic — just with different per-term weights.
- Why it matters in hybrid retrieval: SPLADE gives you lexical-style recall
  (the matched tokens are inspectable) with learned-expansion ability. It is the
  natural learned-sparse partner for BM25 in a fusion recipe (mod-106).

### 2.3 `notes/decision.md` — what we actually deploy

**For the mod-101 single-corpus baseline:**

- **First stage:** bi-encoder dense (BGE-small or E5-base) for cheap, broad recall.
  One vector per passage, one ANN index, cheap at query time. This is the default.
- **Re-ranker (optional, when the baseline leaves measurable headroom on NDCG@10):**
  BGE-reranker-base or `ms-marco-MiniLM-L-6-v2` cross-encoder over the top 100
  from the first stage. Expect +3–7 points NDCG@10 at the cost of 20–80 ms per
  query depending on batch size and candidate count. The exact numbers are
  measured, not assumed.
- **Hybrid variant** (when BM25 beats the dense first stage on specific head queries —
  product IDs, serial numbers, APIs): add BM25 (or SPLADE) and fuse with RRF
  (mod-106). SPLADE is the right choice when the team is willing to maintain a
  learned-sparse index; BM25 is the right choice when they are not.
- **ColBERT** is *not* a default for this stack. It is strong on long-tail recall and
  makes short-chunk retrieval robust, but the token-level index is ~10× the disk of a
  single-vector index on the same corpus and demands a separate serving path
  (RAGatouille / PLAID). We would consider it when the corpus is small enough that
  the storage multiplier is not painful (≤ ~1M passages) *and* the queries are short
  conversational prompts where bi-encoder pooling demonstrably loses recall.

**The order of these choices matters.** Picking ColBERT or a cross-encoder first-stage
without a bi-encoder baseline to beat guarantees over-engineering. Start with one
dense family, measure, then add the second stage only when you have a measured gap.

## 3. Validation steps

1. The learner can state the scoring formula for each family in one sentence without
   looking it up. Ask them to write down, from memory, what MaxSim computes and what a
   cross-encoder's `[CLS]` output represents. If they call a cross-encoder "a bi-encoder
   with a joint pool" they have missed the key mechanic.
2. The learner can point at the index object each family produces. Dense bi-encoder → a
   `(N, d)` float matrix. ColBERT → a per-token vector store plus a centroid partition.
   SPLADE → an inverted index keyed by vocabulary token. Cross-encoder → *none*.
3. The learner can name the latency floor for each family at query time. Bi-encoder is
   ~one ANN probe. Cross-encoder is `k` transformer forward passes for `k` candidates.
   ColBERT is dominated by centroid lookup + MaxSim over expanded candidates. SPLADE is
   inverted-index posting-list traversal.
4. The one-query trace cites specific, inspectable artifacts per family: a vector dump
   for the dense case, a sparse-dictionary print for SPLADE, a top-5 re-ranker score
   list for the cross-encoder, and (if the learner went deep) the MaxSim contribution
   of each query token for ColBERT.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Representation | For all four families, the learner correctly names what is produced for the query and what is stored for a document. |
| Scoring | For each family, the learner writes the scoring function in one line (cosine, scalar logit, MaxSim, sparse inner product) and names the operation (ANN probe, transformer forward, token-level late interaction, inverted-index posting traversal). |
| Indexability | The learner correctly says that a cross-encoder has no document-side index, and correctly names the index shape for the other three (dense ANN, token-vector + partition, inverted index). |
| Role in stack | Each family is placed as either first-stage, re-ranker, or hybrid-partner, with one sentence of justification per placement. |
| Query trace | The sample query produces artifacts of the right shape per family. A reviewer can look at the trace and see that ColBERT produced `n_query_tokens * n_passage_tokens` interaction entries, not a single scalar. |
| Decision doc | The recommendation picks a first stage and a re-ranker, names at least one condition under which the choice would change, and does not default to "pick all four." |
| Honesty about costs | The learner states ColBERT's storage multiplier and the cross-encoder's per-candidate transformer cost rather than waving "ColBERT is strong" / "re-rankers are a bit slower." |

Fail-fast smells:

- Calling SPLADE "a dense model with sparse output" — it is a learned-sparse model whose
  output lives on an inverted index, and conflating that with dense embeddings misses
  why it composes with BM25 in hybrid.
- Claiming the cross-encoder "has the document vector cached." It does not — that is the
  entire reason it cannot be a first-stage retriever.
- Picking ColBERT as the default first stage without naming the storage cost or the
  separate serving path. That is a serving decision, not an accuracy choice.

## 5. Common mistakes

- **Swapping cross-encoder with bi-encoder in the family map.** Bi-encoders have two
  encoders that *never see each other* and a cheap cosine step. Cross-encoders are a
  single transformer that reads both strings together. If the learner describes a
  "bi-encoder" that "attends across query and document" they have described a
  cross-encoder.
- **Saying ColBERT and SPLADE "do the same thing."** They both preserve token-level
  signal that a dense pool throws away, but they end up on different data structures:
  ColBERT on a dense per-token store with centroid routing, SPLADE on a classical
  inverted index. That difference drives everything downstream — cost, hybrid
  composition, debuggability.
- **Treating "late interaction" as a scoring variant of a dense model.** The index shape
  changes. If the learner's decision doc proposes ColBERT with "we'll just reuse the
  HNSW index," that will not work.
- **Picking a re-ranker as the first stage.** A cross-encoder is O(N) in corpus size; it
  is not deployable as a first stage without the cheaper candidate generator in front.
  If the recommendation reads "we'll just run the cross-encoder over all 1M passages,"
  that is the wrong family in the wrong slot.
- **Claiming SPLADE "is BM25 with embeddings."** It runs on an inverted index like BM25,
  but the weights come from a learned expansion — the model adds terms that never
  appeared in the surface text. That expansion is what makes it a different family,
  not a tuned BM25.
- **Not naming the query latency floor.** Every production stack has a latency budget.
  A family map that cannot answer "and what is this going to cost at p99?" is a
  classroom answer.

## 6. References

- Family primaries:
  - Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" — bi-encoder / cross-encoder distinction, [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084).
  - Khattab & Zaharia (2020), "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" — [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832).
  - Santhanam, Khattab et al. (2022), "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" (PLAID) — [arxiv.org/abs/2112.01488](https://arxiv.org/abs/2112.01488).
  - Formal, Piwowarski, Clinchant (2021), "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" — [arxiv.org/abs/2107.05720](https://arxiv.org/abs/2107.05720). SPLADEv2 — [arxiv.org/abs/2109.10086](https://arxiv.org/abs/2109.10086).
- Practitioner implementations:
  - Sentence-Transformers library (bi- and cross-encoders) — [sbert.net](https://sbert.net/).
  - Cross-encoder catalog (`cross-encoder/ms-marco-*`) — [sbert.net/docs/pretrained-models/ce-msmarco.html](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html).
  - RAGatouille (ColBERT serving in Python) — [github.com/bclavie/RAGatouille](https://github.com/bclavie/RAGatouille).
  - Official ColBERT repo — [github.com/stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT).
  - SPLADE reference implementation — [github.com/naver/splade](https://github.com/naver/splade).
  - Pyserini impact indexes (SPLADE/uniCOIL) — [github.com/castorini/pyserini](https://github.com/castorini/pyserini).
  - BGE (bi-encoder + reranker families), BAAI — [huggingface.co/BAAI](https://huggingface.co/BAAI).
- Security framings consulted while drafting the decision rationale (threat surface of
  swapping embedding families in production):
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
