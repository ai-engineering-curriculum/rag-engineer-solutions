# Exercise 03 — Solution: Contextual Compression Workshop

This is a reference solution for
`exercise-03-contextual-compression-workshop.md`. It is an
**implementation** exercise. The learner builds three compression stages
that sit between the reranker (exercise-02) and the prompt builder
(exercise-04), measures the token-count reduction, the groundedness
impact, and the added latency, and recommends when each stage earns its
keep.

The teaching point: compression is a **lossy** transform on retrieval
output. The question is not "does it reduce tokens" (every method does)
but "does it reduce tokens *without removing the span the generator
needs*." A compressor that saves 40% of tokens while dropping the
answer sentence 5% of the time is worse than no compression.

## 1. Solution overview

The deliverable has six parts:

1. **`src/compressor_base.py`** — the `Compressor` protocol every stage
   implements: `compress(query, chunks) → compressed_chunks` returning
   objects with `text`, `source_id`, and a per-chunk metadata dict.
2. **`src/compressor_embedding.py`** — embedding-similarity filter.
   Drop chunks whose embedding cosine against the query falls below a
   threshold. Cheap, deterministic, no LLM call.
3. **`src/compressor_sentence.py`** — sentence-level extractive
   compressor. Score each sentence against the query (cross-encoder or
   embedding) and keep the top-M per chunk, preserving document order.
4. **`src/compressor_llm.py`** — LLM-based extractor. A small LLM is
   prompted to return the spans most relevant to the query, verbatim.
5. **`src/bench.py`** — the evaluation harness: runs each compressor
   over the same reranker output, records compression ratio, token
   count, p50/p95 latency, and **answer-span survival** against a
   gold-span annotation set.
6. **`notes/results.md`** — the numeric table and recommendation:
   which compressor to use on which query shape, with a measurable
   trigger to re-evaluate.

The exercise is scoped to a working pipeline plus defensible numbers,
not state-of-the-art compression. If the chosen compressor reduces
prompt tokens by a meaningful fraction while answer-span survival
stays above a threshold (the exercise names 95%), the exercise passes.

## 2. Worked answer or implementation

### 2.1 Contextual compression in one paragraph

A retrieval pipeline that reranks N candidates to K chunks still hands
the prompt builder several thousand tokens of raw passage text.
Contextual compression trims that payload *conditional on the current
query* so the prompt builder sees fewer, shorter, or more focused
chunks. The three dominant strategies are: (a) drop whole chunks
below a relevance threshold, (b) keep only the sentences in each
chunk that match the query, (c) ask an LLM to return the quoted spans
that are relevant. They differ in cost, determinism, and failure
mode, and compose — a cheap filter can gate an expensive extractor.

### 2.2 `src/compressor_base.py` — the protocol

```python
"""Compressor protocol shared by all compression stages."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass
class Chunk:
    text: str
    source_id: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

@runtime_checkable
class Compressor(Protocol):
    name: str
    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...
```

A compressor **must preserve `source_id`** on every surviving chunk.
Exercise-04 relies on this to attach citation anchors; a compressor
that strips it is unshippable.

### 2.3 `src/compressor_embedding.py` — embedding filter

```python
"""Drop chunks whose embedding cosine against the query falls below tau."""
from __future__ import annotations
import numpy as np
from compressor_base import Chunk

class EmbeddingFilter:
    name = "embedding_filter"
    def __init__(self, encode, tau: float = 0.55):
        self.encode = encode     # (text) -> np.ndarray, L2-normalised
        self.tau = tau

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        q = self.encode(query)
        cs = np.stack([self.encode(c.text) for c in chunks])
        sims = cs @ q
        out: list[Chunk] = []
        for chunk, s in zip(chunks, sims):
            if s >= self.tau:
                md = dict(chunk.metadata, compression="kept", sim=float(s))
                out.append(Chunk(chunk.text, chunk.source_id, chunk.score, md))
        return out
```

**Choices and why:**

- One embedding per chunk, one per query. If chunk embeddings are
  already cached from mod-103/104, the filter is nearly free.
- `tau` is a hyperparameter; set it on a dev slice so that
  `~95%` of gold-labelled chunks survive. A threshold copied from a
  blog post will either drop too many or keep everything.
- The filter emits whole chunks unchanged; it reduces *count*, not
  *length*. Pair with a sentence-level stage when per-chunk
  verbosity is the binding constraint.

### 2.4 `src/compressor_sentence.py` — sentence extractor

```python
"""Keep top-M sentences per chunk by (query, sentence) score."""
from __future__ import annotations
import re
from compressor_base import Chunk

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

def split_sentences(text: str) -> list[str]:
    s = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return s or [text]

class SentenceExtractor:
    name = "sentence_extractor"
    def __init__(self, score_pair, keep_m: int = 3, min_score: float = -5.0):
        # score_pair(q, [sents]) -> list[float]; cross-encoder or embedding
        self.score_pair = score_pair
        self.keep_m = keep_m
        self.min_score = min_score

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk in chunks:
            sents = split_sentences(chunk.text)
            scores = self.score_pair(query, sents)
            # Pick top-M sentence indices, then emit in document order
            ranked = sorted(
                range(len(sents)),
                key=lambda i: scores[i],
                reverse=True,
            )[: self.keep_m]
            kept = sorted(
                i for i in ranked
                if scores[i] >= self.min_score
            )
            if not kept:
                continue
            new_text = " ".join(sents[i] for i in kept)
            md = dict(
                chunk.metadata,
                compression="sentence_top_m",
                original_chars=len(chunk.text),
                kept_chars=len(new_text),
                kept_sent_idxs=kept,
            )
            out.append(Chunk(new_text, chunk.source_id, chunk.score, md))
        return out
```

**Choices and why:**

- Sentences are emitted in **document order**, not score order.
  Preserving order keeps coreference and discourse chains intact
  ("Then she returned to the meeting." breaks if "she" was in a
  dropped earlier sentence; keep the earlier sentence or drop the
  later one).
- `keep_m` is per-chunk. Allowing a chunk to contribute zero
  sentences (via `min_score`) lets the stage also drop uninformative
  chunks.
- `kept_sent_idxs` is stored on metadata so exercise-04 can anchor
  citations to a sub-chunk span, not just the chunk id.
- A naive regex sentence splitter is fine for English teaching
  corpora. For multilingual production, swap in `pysbd` or a
  language-aware splitter.

### 2.5 `src/compressor_llm.py` — LLM extractor

```python
"""Extract quoted relevant spans with a small LLM."""
from __future__ import annotations
import json
from compressor_base import Chunk

PROMPT = """You extract the shortest contiguous spans from a passage that
are directly relevant to the user's question. You must:
- Return JSON matching: {"spans": ["...", "..."]}.
- Each span is a verbatim substring of the passage. No paraphrase.
- If no part of the passage is relevant, return {"spans": []}.
- Do not add commentary.

Question:
{query}

Passage:
{passage}
"""

class LLMSpanExtractor:
    name = "llm_span_extractor"
    def __init__(self, llm_complete, max_tokens: int = 256):
        self.llm_complete = llm_complete
        self.max_tokens = max_tokens

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk in chunks:
            raw = self.llm_complete(
                PROMPT.format(query=query, passage=chunk.text),
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                spans = json.loads(raw).get("spans", [])
            except json.JSONDecodeError:
                spans = []
            kept = [s for s in spans if isinstance(s, str) and s in chunk.text]
            if not kept:
                continue
            new_text = " … ".join(kept)
            md = dict(
                chunk.metadata,
                compression="llm_span",
                n_spans=len(kept),
                original_chars=len(chunk.text),
                kept_chars=len(new_text),
            )
            out.append(Chunk(new_text, chunk.source_id, chunk.score, md))
        return out
```

**Choices and why:**

- **Verbatim-only** output. The extractor filters out any string
  the LLM returned that is not a substring of the passage. This is
  non-negotiable: if the extractor paraphrases, the system has
  fabricated content and groundedness is no longer defensible.
  Running `s in chunk.text` is a cheap guard.
- JSON mode is used where the model supports it; a text-mode
  fallback with a parser is acceptable but noisier.
- `max_tokens=256` caps worst-case latency per chunk. Set per
  deployment; the right number is "what you see at p95 on a
  representative query."
- A join with `" … "` makes it visually explicit that non-consecutive
  spans have been elided — this matters when the LLM assembler
  reads the span later.

### 2.6 `src/bench.py` — the evaluation harness

```python
"""Measure each compressor on compression ratio, latency, and
answer-span survival on an annotated eval slice."""
from __future__ import annotations
import csv, json, time, pathlib
from statistics import median
from compressor_base import Chunk, Compressor

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

def answer_span_survives(compressed: list[Chunk], gold_span: str) -> bool:
    for c in compressed:
        if gold_span in c.text:
            return True
    return False

def bench(
    queries: list[dict],      # [{qid, text, retrieved_chunks, gold_span, gold_source_id}]
    compressors: list[Compressor],
    enc,                      # tokenizer
    out_path: str,
) -> None:
    rows: list[dict] = []
    for comp in compressors:
        per_q = []
        for q in queries:
            in_chunks = [Chunk(**c) for c in q["retrieved_chunks"]]
            in_tokens = sum(count_tokens(c.text, enc) for c in in_chunks)
            t0 = time.perf_counter()
            out_chunks = comp.compress(q["text"], in_chunks)
            t1 = time.perf_counter()
            out_tokens = sum(count_tokens(c.text, enc) for c in out_chunks)
            per_q.append({
                "qid": q["qid"],
                "in_tokens":   in_tokens,
                "out_tokens":  out_tokens,
                "ratio":       (out_tokens / in_tokens) if in_tokens else 1.0,
                "latency_ms":  (t1 - t0) * 1000,
                "survived":    answer_span_survives(out_chunks, q["gold_span"]),
                "gold_source_kept": any(
                    c.source_id == q["gold_source_id"] for c in out_chunks
                ),
            })
        agg = {
            "compressor":   comp.name,
            "mean_ratio":   sum(x["ratio"] for x in per_q) / len(per_q),
            "p50_latency":  median(x["latency_ms"] for x in per_q),
            "p95_latency":  sorted(x["latency_ms"] for x in per_q)[int(len(per_q) * 0.95)],
            "span_survival": sum(x["survived"] for x in per_q) / len(per_q),
            "source_kept":  sum(x["gold_source_kept"] for x in per_q) / len(per_q),
            "per_q": per_q,
        }
        rows.append(agg)
        print(f"{comp.name:22s}  "
              f"ratio={agg['mean_ratio']:.3f}  "
              f"survival={agg['span_survival']:.3f}  "
              f"p50={agg['p50_latency']:.0f}ms  "
              f"p95={agg['p95_latency']:.0f}ms")
    pathlib.Path(out_path).write_text(json.dumps(rows, indent=2))
```

**Choices and why:**

- `answer_span_survives` uses **substring match** on the compressed
  text. It intentionally does not reward paraphrase — the goal is
  to measure whether the grounded span is still in the context
  when the generator reads it.
- `gold_source_kept` separates "the right document is still in the
  context" from "the right sentence is still in the context." A
  compressor that drops the wrong document is worse than one that
  trims the right document too aggressively.
- No groundedness-via-LLM-judge in this harness. That is
  mod-109's territory; using it here conflates the compression
  metric with the generator metric.
- The per-query records are persisted so downstream analysis
  (e.g., which query shapes the LLM extractor regresses on) does
  not require re-running compression.

### 2.7 `notes/results.md` — reporting shape

Example table (illustrative, reproduce on the mod-109 eval slice):

```
Compressor             ratio    span_survival   source_kept   p50      p95
--------------------  -------  ---------------  -----------  -------  -------
none (baseline)       1.000        1.000          1.000         0 ms    0 ms
embedding_filter      0.72         0.994          0.995        18 ms   32 ms
sentence_extractor    0.38         0.971          0.988        85 ms  160 ms
llm_span_extractor    0.19         0.952          0.984      1200 ms 2800 ms
embedding_filter
  → sentence_extr.    0.34         0.972          0.988        98 ms  190 ms
```

**Reading the table:**

- `embedding_filter` is nearly free, keeps ~99.4% of answer spans,
  and still cuts prompt tokens by ~30%. Worth running on every
  query.
- `sentence_extractor` adds meaningful latency (~85 ms p50) and
  compresses to ~38% of the original. Span survival drops ~2.3
  points — within the exercise's 95% threshold, but the margin is
  real and depends on `keep_m`.
- `llm_span_extractor` compresses the hardest (~19%) but the
  latency and cost are high and span survival drops a further
  ~2 points, mostly because the LLM sometimes extracts a shorter
  span than the gold span.
- The pipelined combination `embedding_filter → sentence_extr.`
  compresses a little more than `sentence_extr.` alone because
  the filter has already dropped some chunks, and latency is
  essentially the sentence-extractor's latency.

**Recommendation:**

- Default stage: `embedding_filter` at `tau=0.55` (tuned on the dev
  slice). Nearly free, strictly improves token economics.
- When prompt tokens are the binding budget: append
  `sentence_extractor` with `keep_m=3`. Re-measure span survival
  after every retriever / reranker change.
- `llm_span_extractor`: reserve for offline pipelines (precomputed
  summaries for long documents, batch QA), for high-value queries
  where the latency buys back prompt budget for a stronger
  generator, or when the chunk size is large (>1k tokens) and
  dropping to a few spans is the only way to fit the budget.

A **trigger to re-evaluate** the choice: when any of these moves by
more than 20% — the reranker's top-K chunk-length distribution, the
LLM's context window, the p95 budget, or the eval slice's span
survival.

## 3. Validation steps

1. Every compressor preserves `source_id` on surviving chunks. A
   compressor that returns raw strings or drops metadata breaks
   exercise-04 and must be fixed, not patched downstream.
2. The LLM extractor outputs are verified to be substrings of the
   input chunk. The harness drops any string that isn't — the
   fraction of dropped strings is a drift signal worth logging.
3. The eval harness reports ratio, p50, p95, span survival, and
   source survival. Any missing column is a shortcut.
4. `span_survival ≥ 0.95` for the chosen operating point on the
   held-out slice. If below, the compressor is dropping too much;
   loosen the thresholds or add a fallback.
5. The embedding-filter threshold is tuned on a dev slice, not
   taken from a blog post. The tuning script should land in the
   repo.
6. Latency is reported at p95, not mean. The LLM extractor has a
   long tail (slow responses, retries); p95 is the budget-relevant
   number.
7. The report explicitly names when each compressor is the wrong
   choice (not just which is "best").

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Role of compression | The learner frames compression as **lossy** and measures the loss (span survival), not just the token ratio. |
| Three stages | At least one cheap filter, one extractive stage, one LLM-based stage. They are not conflated into a single "compressor." |
| Source-id preservation | `source_id` survives every compressor. Verified in a test, not just assumed. |
| LLM verbatim guard | The LLM extractor filters out strings that are not substrings of the input passage. No paraphrasing. |
| Threshold hygiene | Embedding threshold and `keep_m` are tuned on a dev slice; values are checked in with the tuning script. |
| Eval metrics | Compression ratio, p50/p95 latency, answer-span survival, source-kept rate, all reported per compressor. |
| Composition | A pipelined combination (cheap-filter → sentence-extractor) is measured, not just the single-stage variants. |
| Operating point | Chosen compressor sits above the 95% span-survival floor and inside the latency budget. |
| Tail analysis | At least one query class where a compressor regresses is called out (e.g., comparison questions where the LLM extractor drops the comparator). |
| Trigger | A measurable condition under which the operating point should be re-evaluated. |

Fail-fast smells:

- Reporting "compression ratio = 0.3" with no span survival number.
  That is not a win; it is a loss function that has not been
  computed.
- An LLM extractor that paraphrases or summarises. By the time the
  extractor's output reaches the generator, the context is
  ungrounded.
- A compressor that drops `source_id` and emits plain strings.
  Exercise-04 has no way to attach citation anchors to those
  strings.
- Treating `llm_span_extractor` as the baseline. It is the most
  expensive option; make it the fallback, not the default.
- Measuring span survival with a fuzzy / embedding similarity
  check. Fuzzy survival hides paraphrase.

## 5. Common mistakes

- **Letting the LLM paraphrase.** A span extractor that returns
  "the policy was changed in 2023" instead of the actual sentence
  has fabricated text. Verify as substring.
- **Dropping `source_id`.** Compressors are often implemented as
  `str → str` and lose the attribution path. Enforce `Chunk → Chunk`
  with `source_id` preserved; otherwise citation anchors are
  unrecoverable.
- **Re-ordering sentences by score.** Sentence 3 is scored highest,
  sentence 1 second highest; emitting `[3, 1]` breaks discourse
  structure. Preserve document order.
- **Running the LLM extractor on every chunk.** At N × K = 100 × 5
  = 500 extractor calls per query, latency explodes. Gate with a
  cheap filter first so the LLM only sees chunks that survived.
- **Tuning the embedding threshold on the eval slice.** The number
  is overfit. Use a held-out slice for the reported numbers.
- **Measuring span survival with fuzzy match.** If the gold sentence
  is paraphrased in the compressed output, groundedness is lost
  but the metric reports "survived." Substring match is the honest
  check.
- **Compressing already-small chunks.** For 150-token chunks, an
  embedding filter is value-add but a sentence extractor is probably
  overkill. The right compressor depends on the chunk size the
  retriever is emitting.
- **Ignoring the generator's robustness.** Some generators handle
  elision markers ("… ") well; some go off-rails. Validate against
  the generator you are actually shipping, not against a toy prompt.
- **No fallback path.** When the LLM extractor returns `{"spans":
  []}`, the code should not drop the chunk silently. Either pass
  through the original chunk (safer) or route to the
  sentence-extractor (cheaper-but-loud); never hand an empty chunk
  to the prompt builder.
- **Treating compression as a quality win.** Compression trades
  prompt tokens (cost) for span-dropping risk (quality). The right
  metric is Pareto: more tokens cut per unit survival dropped. A
  compressor that cuts 60% and drops 10% survival is worse than a
  compressor that cuts 40% and drops 1%.

## 6. References

- Embedding-filter and sentence-extractor concepts in the LangChain /
  LlamaIndex contextual-compression retrievers:
  - LangChain contextual compression —
    [python.langchain.com/docs/concepts/retrievers/#contextual-compression](https://python.langchain.com/docs/concepts/retrievers/#contextual-compression).
  - LlamaIndex node postprocessors (sentence-window, auto-merging,
    similarity-filter) —
    [docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root/](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root/).
- Prompt compression:
  - Jiang et al. (2023), "LLMLingua: Compressing Prompts for Accelerated
    Inference of Large Language Models" —
    [arxiv.org/abs/2310.05736](https://arxiv.org/abs/2310.05736).
  - Jiang et al. (2024), "LongLLMLingua: Accelerating and Enhancing LLMs
    in Long Context Scenarios via Prompt Compression" —
    [arxiv.org/abs/2310.06839](https://arxiv.org/abs/2310.06839).
  - Microsoft LLMLingua project —
    [github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua).
- Positional bias motivating "do not just dump everything":
  - Liu et al. (2023), "Lost in the Middle: How Language Models Use Long
    Contexts" — [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
