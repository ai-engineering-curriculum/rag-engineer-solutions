# Exercise 03 — Solution: HyDE and Multi-Query Fanout — Rewriting Workshop

This is a reference solution for
`exercise-03-hyde-and-multi-query-fanout-rewriting-workshop.md`. It is an
**implementation + evaluation** exercise. The learner builds two query-side
rewriting pipelines — Hypothetical Document Embeddings (HyDE) and
multi-query fanout — plugs them into the mod-106 hybrid retriever, and
reports before/after retrieval metrics plus a short note on cost and
latency.

The teaching point is that *rewriting is retrieval*. A well-chosen rewrite
increases recall by changing what embeddings or lexical tokens the
retriever sees, not by changing the retriever itself. The failure mode is
treating rewriting as a free upgrade: HyDE and multi-query both add LLM
calls per query, both drift off-topic when the base model is wrong, and
both need deduplication and fusion to avoid flooding the reranker with
near-identical candidates.

## 1. Solution overview

The deliverable has six parts:

1. **`src/hyde.py`** — a HyDE rewriter: ask an LLM for a short hypothetical
   answer to the user query, embed that answer, and search with the
   resulting vector (instead of or in addition to the raw query's vector).
2. **`src/multiquery.py`** — a multi-query fanout rewriter: ask an LLM for
   N distinct reformulations of the query, run each against the retriever
   in parallel, and fuse with RRF.
3. **`src/pipeline.py`** — the integrated retriever that can be configured
   as `baseline`, `hyde`, `multiquery`, or `hyde+multiquery`.
4. **`eval/report.md`** — NDCG@10, Recall@10, MRR, mean latency, and mean
   cost per query for each configuration on the 20-query eval set from
   exercise-02.
5. **`prompts/`** — the exact prompts used, versioned. The prompts are
   part of the artifact; swapping a prompt silently invalidates the eval.
6. **`notes/failures.md`** — a short analysis of queries where rewriting
   *hurt* and why, with concrete examples of the generated hypothetical
   documents and reformulations.

## 2. Worked answer or implementation

### 2.1 HyDE in one paragraph

Hypothetical Document Embeddings (Gao, Ma, Lin, Callan — "Precise Zero-Shot
Dense Retrieval without Relevance Labels," ACL 2023) is a zero-shot dense
retrieval technique. Given a query, prompt an LLM for a plausible answer
*as if it were a retrieved document*; embed that hypothetical answer
instead of (or alongside) the original query; search the dense index with
the resulting vector. Intuition: a query like "why did the consumer group
rebalance stall?" is short and under-specified for a bi-encoder; the
generated answer looks more like the documents in the index, so cosine
similarity between it and the true answer is higher. HyDE is an
embedding-space trick, not a reranking trick — it only helps dense
retrieval.

### 2.2 Multi-query fanout in one paragraph

Multi-query fanout (sometimes called query expansion or "sub-query
generation") asks an LLM for N paraphrased or decomposed reformulations
of the query, runs each through the retriever independently, and fuses
the N ranked lists with RRF. Intuition: a single query embedding occupies
one point in embedding space; N reformulations occupy a small cloud, which
is more robust to the embedder's blind spots. It benefits both lexical
(BM25 vocabulary varies across paraphrases) and dense retrieval.
LangChain's `MultiQueryRetriever` and LlamaIndex's `QueryDecomposer` are
reference implementations.

### 2.3 `prompts/hyde.txt`

```
You will be given a user question. Write a short (3–5 sentences) passage
that plausibly answers the question, in the style of a technical
reference document. Do not include hedging, caveats, or "I don't know."
Do not refuse. If the question is ambiguous, pick the most common
interpretation and answer that. Your output will be used as a search
query, not shown to the user.

Question: {question}

Passage:
```

Design notes:

- **"In the style of a reference document."** The point of HyDE is to
  push the embedding toward the indexed document space; writing in the
  user's casual style defeats this. A sentence in the shape of a Kafka
  documentation paragraph embeds closer to indexed Kafka docs than the
  same information phrased as a Reddit comment.
- **"3–5 sentences."** Too short and the embedding degenerates toward
  the query. Too long and it dilutes the specific terms the retriever
  should match on.
- **"Do not refuse."** Base LLMs occasionally refuse an ambiguous or
  edgy query; HyDE needs text. If the model refuses, the pipeline
  should fall back to the raw query embedding rather than embed the
  refusal.

### 2.4 `prompts/multiquery.txt`

```
You will be given a user question. Write exactly {n} reformulations of
the question that a search engine could use to retrieve supporting
documents. The reformulations should:
- Differ in surface form and vocabulary (synonyms, acronym expansions,
  related terms).
- Preserve the intent of the original question — do not change the
  entity, the quantity, or the domain.
- Vary in specificity: at least one should be broader (concept-level)
  and at least one should be more specific (term-level) than the
  original.

Return one reformulation per line, no numbering, no commentary.

Original question: {question}

Reformulations:
```

Design notes:

- **Explicit count.** Telling the model `{n}` reformulations keeps the
  output budget predictable. A model given "a few" will return
  anywhere from 2 to 20.
- **"Preserve intent."** The most common multi-query failure is that one
  of the reformulations changes the entity or the time scope. Guard
  against it in the prompt and then dedupe/validate downstream.
- **Specificity variance.** A broader reformulation improves recall; a
  more specific one improves precision. Asking for at least one of
  each prevents the model from returning N near-duplicate paraphrases.

### 2.5 `src/hyde.py`

```python
"""HyDE query rewriter — generate a hypothetical passage, embed it,
retrieve by vector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sentence_transformers import SentenceTransformer


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 256) -> str: ...


# Phrases that indicate the LLM refused instead of answering. A nonempty
# refusal (e.g. "I cannot help with that request." or "Unfortunately, I
# will not answer.") would otherwise be embedded as if it were a
# hypothetical passage and used for dense retrieval.
_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i will not",
    "i won't",
    "i am unable",
    "i'm unable",
    "i am sorry",
    "i'm sorry",
    "i refuse",
    "cannot help",
    "can't help",
    "cannot assist",
    "can't assist",
    "unable to help",
    "unable to assist",
    "as an ai",
    "as a language model",
)


def _looks_like_refusal(text: str) -> bool:
    # Scan the first sentence or so (not the full passage) so a legitimate
    # mention of "cannot" deep inside a real answer does not trigger the
    # fallback. Substring match (not just prefix) catches lead-ins like
    # "Unfortunately, I cannot help..." that a prefix check would miss.
    head = text.lstrip().lower()[:200]
    return any(phrase in head for phrase in _REFUSAL_PHRASES)


@dataclass
class HydeRewriter:
    llm: LLMClient
    embedder: SentenceTransformer
    prompt_template: str  # loaded from prompts/hyde.txt
    max_tokens: int = 256

    def embed_query(self, question: str) -> list[float]:
        prompt = self.prompt_template.format(question=question)
        passage = self.llm.complete(prompt, max_tokens=self.max_tokens).strip()
        if not passage or _looks_like_refusal(passage):
            # Fallback: empty generation or a nonempty refusal. Embed the
            # raw query instead so the refusal text does not drive dense
            # retrieval.
            passage = question
        vec = self.embedder.encode(passage, normalize_embeddings=True).tolist()
        return vec
```

The HyDE rewriter returns a vector, not a query string. The retriever
layer uses it in place of the embedding of the raw query. A common
extension — recommended in the paper — is to average the HyDE embedding
with the raw-query embedding; this hedges against bad generations.

### 2.6 `src/multiquery.py`

```python
"""Multi-query fanout rewriter — generate N reformulations, run each,
fuse with RRF."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol

from fusion import rrf  # from exercise-02


class LLMClient(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 256) -> str: ...


Retriever = Callable[[str], list[str]]  # query -> ranked doc IDs


@dataclass
class MultiQueryRewriter:
    llm: LLMClient
    prompt_template: str  # loaded from prompts/multiquery.txt
    n: int = 4

    def reformulate(self, question: str) -> list[str]:
        prompt = self.prompt_template.format(question=question, n=self.n)
        raw = self.llm.complete(prompt, max_tokens=256).strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Dedupe case-insensitively; always include the original.
        seen = {question.strip().lower(): question.strip()}
        for ln in lines:
            key = ln.lower()
            if key not in seen:
                seen[key] = ln
        return list(seen.values())[: self.n + 1]  # original + up to n new

    def retrieve(self, question: str, retriever: Retriever) -> list[tuple[str, float]]:
        queries = self.reformulate(question)
        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as ex:
            ranked_lists = list(ex.map(retriever, queries))
        return rrf(ranked_lists, k=60)
```

Implementation notes:

- **The original query is always part of the fanout.** If the model
  produces junk reformulations, the original ensures at least one
  solid retrieval. This is the single highest-leverage robustness
  check in a multi-query pipeline.
- **Deduplicate case-insensitively.** Models frequently return
  near-duplicate reformulations ("How do consumer groups rebalance?"
  vs. "How does a consumer group rebalance?"). Running them both wastes
  budget and over-weights the cluster in the fused ranking.
- **Parallel fanout.** Each reformulation hits the retriever
  independently. Latency is dominated by the slowest call, not the sum.
- **Fuse with RRF, not weighted-sum.** The reformulations have no
  calibrated scores against each other; rank-only fusion is the natural
  combiner (and matches the paper's recipe).

### 2.7 `src/pipeline.py` — the integrated retriever

```python
"""Composable retrieval pipeline. Each configuration shares the same
BM25 + dense retrievers and only changes the query-side rewriting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fusion import rrf  # from exercise-02
from hyde import HydeRewriter
from multiquery import MultiQueryRewriter

Mode = Literal["baseline", "hyde", "multiquery", "hyde+multiquery"]


@dataclass
class HybridRetriever:
    bm25_search: callable
    dense_search_by_vec: callable
    dense_search_by_text: callable
    hyde: HydeRewriter
    mq: MultiQueryRewriter
    mode: Mode = "baseline"

    def _dense_hits(self, query: str) -> list[str]:
        if self.mode in ("hyde", "hyde+multiquery"):
            vec = self.hyde.embed_query(query)
            return self.dense_search_by_vec(vec)
        return self.dense_search_by_text(query)

    def search(self, query: str) -> list[tuple[str, float]]:
        if self.mode in ("multiquery", "hyde+multiquery"):
            def retrieve_one(q: str) -> list[str]:
                lex = self.bm25_search(q)
                dns = self._dense_hits(q) if q == query else self.dense_search_by_text(q)
                return [doc for doc, _ in rrf([lex, dns], k=60)]
            return self.mq.retrieve(query, retrieve_one)
        lex = self.bm25_search(query)
        dns = self._dense_hits(query)
        return rrf([lex, dns], k=60)
```

Key compositional choices:

- **HyDE only touches the dense leg.** HyDE is an embedding-space
  intervention; applying it to BM25 is a category error. The BM25 leg
  always runs on the raw query (or the raw reformulation in the
  multiquery case).
- **HyDE + multiquery runs HyDE only for the original query.** Running
  HyDE on every reformulation multiplies LLM calls by `n+1` for no
  measurable retrieval gain in the reference run. The original query
  gets HyDE, the reformulations hit the dense index with their own
  text embedding. This is a cost choice, not a correctness one.
- **Each rewrite mode produces an RRF-fused list in the same shape.**
  The downstream reranker (mod-107) sees the same artifact regardless
  of mode.

### 2.8 `eval/report.md`

Representative (teaching-run) results over the 20-query eval set from
exercise-02:

| Configuration | NDCG@10 | Recall@10 | MRR | Mean latency | LLM calls / query |
|---|---|---|---|---|---|
| baseline (BM25 + dense + RRF) | 0.60 | 0.71 | 0.68 | 85 ms | 0 |
| hyde | 0.63 | 0.75 | 0.70 | 420 ms | 1 |
| multiquery (n=4) | 0.65 | 0.79 | 0.72 | 480 ms | 1 |
| hyde + multiquery (n=4) | **0.66** | **0.81** | 0.72 | 760 ms | 2 |

> **Reproducibility note.** Exact numbers depend on model, corpus, and
> query set. The pattern — HyDE and multi-query each give a few points
> of NDCG, stacking gives slightly more at substantial latency cost — is
> typical.

Per-query behavior worth flagging:

- **`q04` ("how to tune `fetch.min.bytes` for low-latency consumers").**
  Baseline NDCG@10 was 0.58; HyDE dropped to 0.49 because the generated
  passage drifted into "general consumer tuning" without the specific
  parameter name, and the embedding followed. The lexical leg still
  found the right doc via BM25; the fused rank was lower because HyDE
  moved the dense top-5 toward off-topic content. **Fix: average HyDE
  embedding with raw-query embedding (the paper's recommended
  variant).** Re-run: NDCG@10 = 0.60.
- **`q09` ("MirrorMaker 2 offset translation").** Multi-query produced
  one reformulation that dropped "MirrorMaker" and asked about offset
  translation generally. That reformulation retrieved high-scoring
  off-topic docs. RRF diluted them because the other reformulations
  didn't vote for the same off-topic docs. Net effect on NDCG: +0.03.
- **`q14` (acronym-heavy query that BM25 mis-tokenizes).** Multi-query's
  "at least one specific" reformulation expanded the acronym, which
  BM25 matched cleanly. Baseline NDCG 0.31 → multi-query NDCG 0.52.
  This is the headline multi-query win pattern.

### 2.9 Cost model, briefly

- **HyDE adds one LLM call per query.** With a small cheap model (e.g.,
  Haiku 4.5 class), 1–2 cents per thousand queries at typical prompt
  lengths. The call can be cached by `hash(query)` for repeated
  queries, which matters more than most teams expect — in support/FAQ
  workloads, the top 1% of queries make up >50% of volume.
- **Multi-query with `n = 4` adds one LLM call + 3 extra retriever
  calls per query.** The retriever calls are cheap (same index, same
  cluster) but they do multiply the shard-level work. Watch tail
  latencies, not just the mean.
- **Both are evictable.** A production latency budget of 150 ms rules
  out both at this configuration. In that regime, either (a) cache
  rewrites aggressively, (b) precompute rewrites offline for frequent
  queries, or (c) accept the baseline's scores. Do not deploy "HyDE +
  multi-query + reranker" blindly under a strict SLO.

## 3. Validation steps

1. **Prompt determinism.** Set the LLM temperature to 0 and run the
   eval twice. Metrics should be byte-identical. If they aren't, the
   LLM provider is non-deterministic at `temperature=0` (common with
   batched inference) — switch to logging generated passages and
   reformulations so the noise is at least auditable.
2. **HyDE fallback test.** Feed the rewriter a query the LLM refuses
   (e.g., a blatantly off-topic or harmful prompt). The pipeline must
   fall back to the raw query's embedding and not embed the refusal
   text. Confirm by dumping the vector of the refusal response and the
   vector produced by the fallback and showing they differ.
3. **Multi-query always includes the original.** Dump the fanout list
   for every query in the eval. The original query must appear in every
   list. If it doesn't, the dedup is dropping it incorrectly.
4. **Per-query comparison, not aggregate only.** For each configuration,
   produce a per-query NDCG@10 column. Flag queries where HyDE or
   multi-query *hurt* and keep them in the report — they are the real
   fodder for `notes/failures.md`.
5. **Latency and cost are reported.** A rewriting pipeline that improves
   NDCG by 1 point and triples latency is a worse pipeline for most
   SLOs. Report the trade-off, not just the win.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| HyDE prompt | Explicit style target (reference-doc); length target (3–5 sentences); no-refuse handling. |
| HyDE fallback | If the LLM returns empty or a refusal, the pipeline uses the raw-query embedding. |
| Multi-query prompt | Explicit N; "preserve intent"; at least one broader + one more specific reformulation. |
| Multi-query always keeps the original | Confirmed by dumped fanout lists. |
| Fusion | RRF across reformulations, not weighted-sum. |
| Lexical leg | HyDE is **not** applied to BM25; multi-query hits BM25 with each reformulation's text. |
| Metrics | NDCG@10, Recall@10, MRR, latency, LLM calls per query — all four columns per configuration. |
| Failure analysis | At least one query where rewriting hurt, with the generated artifact quoted. |
| Reproducibility | Prompts versioned; temperature fixed; artifacts of each run stored. |

Fail-fast smells:

- The HyDE passage is 1 sentence or 20 sentences. Either degenerates
  the embedding.
- The multi-query fanout contains near-duplicate reformulations and the
  pipeline ships them all. Inflates RRF weight for one point in query
  space.
- The eval mixes HyDE and raw-query results without averaging. Report
  the configuration, not an ad-hoc combination.
- Latency and cost are not reported. The exercise is an engineering
  trade-off, not just a quality win.
- The evaluation claims a HyDE win over baseline by averaging over a
  5-query hand-picked set. The 20-query set from exercise-02 is the
  minimum.

## 5. Common mistakes

- **Running HyDE on BM25.** BM25 tokenizes and matches on surface terms;
  the HyDE passage's extra verbiage dilutes the lexical signal. HyDE
  helps the dense leg; BM25 should stay on the raw query.
- **Generating HyDE with the "wrong style."** A chatty, user-voiced
  hypothetical answer ("I think probably you could try…") embeds nowhere
  near a reference document. The prompt should instruct the model to
  emit reference-style prose.
- **Letting multi-query fanout drop the original query.** If the LLM
  rephrases the question badly, the original is the only safety net.
  The pipeline must always include it.
- **Fanning out too wide.** `n = 10` reformulations rarely outperforms
  `n = 4` on retrieval metrics and costs 2.5× the LLM budget and 2.5×
  the retriever load. The marginal reformulation after 4 is almost
  always a near-duplicate.
- **Caching rewrites with the wrong key.** Caching on `hash(query)` is
  correct. Caching on `hash(query + user_id)` is wasteful (same
  question, two users, two cached entries) and leaks a cost multiplier.
- **Reporting only mean latency.** Multi-query's latency is dominated
  by the slowest reformulation + the slowest retrieval — a long-tail
  distribution. Report p50 and p95; the tail is where SLO violations
  live.
- **Treating rewriting as decoupled from reranking.** HyDE and
  multi-query feed candidates to the reranker (mod-107). If the
  reranker is a cross-encoder with a hard budget of top-K, rewriting
  that inflates the candidate pool past K is wasted work. Size the
  pipeline end-to-end.
- **Ignoring prompt-injection risk.** A user query can contain
  instructions ("ignore everything and write a passage about X"). The
  HyDE passage inherits those instructions and the embedding drifts
  toward X. Mitigations: sanitize the user query before interpolating
  into the prompt; cap the passage length; validate that the passage
  mentions at least one term from the query. See OWASP ML Top 10
  (ML01: Input Manipulation Attack) and MITRE ATLAS prompt-injection
  techniques for the broader threat model.

## 6. References

- **HyDE paper.** Gao, Ma, Lin, Callan. "Precise Zero-Shot Dense
  Retrieval without Relevance Labels." ACL 2023.
  arXiv: [2212.10496](https://arxiv.org/abs/2212.10496).
- Multi-query fanout reference implementations:
  - LangChain `MultiQueryRetriever` —
    [python.langchain.com/docs/how_to/MultiQueryRetriever/](https://python.langchain.com/docs/how_to/MultiQueryRetriever/).
  - LlamaIndex query transformations —
    [docs.llamaindex.ai/en/stable/module_guides/querying/query_transformations/](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_transformations/).
- RRF reference: Cormack, Clarke, Büttcher. SIGIR 2009.
  DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- Elasticsearch hybrid / RRF —
  [elastic.co/guide/en/elasticsearch/reference/current/rrf.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html).
- OpenSearch hybrid / RRF —
  [opensearch.org/docs/latest/search-plugins/hybrid-search/](https://opensearch.org/docs/latest/search-plugins/hybrid-search/).
- Risk / governance framings consulted while drafting:
  - OWASP Machine Learning Security Top 10 (especially ML01: Input
    Manipulation Attack) —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS (prompt-injection techniques) —
    [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
