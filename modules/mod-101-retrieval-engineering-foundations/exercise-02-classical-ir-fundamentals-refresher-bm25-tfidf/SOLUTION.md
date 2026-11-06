# Exercise 02 — Solution: Classical IR fundamentals refresher (inverted index, tf-idf, BM25)

This is a reference solution for `exercise-02-classical-ir-fundamentals-refresher-bm25-tfidf.md`.
Deliverables A–D are worked below, with the arithmetic shown so a learner can diff their own
numbers cell-by-cell. A runnable Python script (`bm25_smoke.py`) is included in Deliverable D.

## 1. Solution overview

The exercise's goal is to make chapter 2's mental model sticky, not to produce an implementation.
We tokenize the five-document mini-corpus with the exercise's prescribed analyzer (lowercase +
whitespace split), build a hand inverted index, compute tf-idf and BM25 scores for the query
`"lexical retrieval"` on each document, run a sensitivity check at `b = 0`, run a saturation check
after duplicating `"retrieval"` five extra times in D1, then reproduce the ranking in Python via
`rank_bm25.BM25Okapi`.

Formula conventions used here, pulled from chapter 2:

- `idf_tf_idf(t) = ln(N / df(t))` (classical tf-idf).
- `idf_bm25(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))` (the Lucene/Elasticsearch smoothed
  form; this is what the Elasticsearch *Practical BM25* blog post documents and what Elasticsearch
  returns). Using this variant keeps all idfs non-negative, which is important here because
  `"retrieval"` appears in every document (`df = N = 5`). The classical unsmoothed BM25 idf would
  go negative for `"retrieval"`.
- `k1 = 1.2`, `b = 0.75` (BM25 defaults from chapter 2).
- Natural log throughout. Changing log base multiplies every score by a constant; rankings are
  base-invariant.

Learners who follow the chapter 2 formula with `ln(N/df)` instead of the Lucene idf will see a
`0` contribution from `"retrieval"` everywhere, which is also fine — the top-2 (D3 and D4) is
unchanged.

## 2. Worked answer or implementation

### The mini-corpus and tokenization

Tokenization = lowercase + split-on-whitespace (as specified).

| Doc | Tokens | `|d|` |
|---|---|---|
| D1 | `vector databases store dense embeddings for retrieval` | 7 |
| D2 | `inverted indexes store terms mapped to postings for retrieval` | 9 |
| D3 | `reciprocal rank fusion combines lexical and dense retrieval` | 8 |
| D4 | `bm25 is the classical scoring function for lexical retrieval` | 9 |
| D5 | `cross encoder rerankers improve retrieval quality` | 6 |

Total tokens = 39. `avgdl = 39 / 5 = 7.8`. `N = 5`.

### Deliverable A — Inverted index

Each row is a term with its document-frequency and posting list `[doc_id, tf]`. The unique
vocabulary has 32 terms (every token appears exactly once in every document that contains it,
so every `tf = 1` in this hand corpus).

| Term | df | Posting list |
|---|---|---|
| and | 1 | `[(D3, 1)]` |
| bm25 | 1 | `[(D4, 1)]` |
| classical | 1 | `[(D4, 1)]` |
| combines | 1 | `[(D3, 1)]` |
| cross | 1 | `[(D5, 1)]` |
| databases | 1 | `[(D1, 1)]` |
| dense | 2 | `[(D1, 1), (D3, 1)]` |
| embeddings | 1 | `[(D1, 1)]` |
| encoder | 1 | `[(D5, 1)]` |
| for | 3 | `[(D1, 1), (D2, 1), (D4, 1)]` |
| function | 1 | `[(D4, 1)]` |
| fusion | 1 | `[(D3, 1)]` |
| improve | 1 | `[(D5, 1)]` |
| indexes | 1 | `[(D2, 1)]` |
| inverted | 1 | `[(D2, 1)]` |
| is | 1 | `[(D4, 1)]` |
| **lexical** | **2** | `[(D3, 1), (D4, 1)]` |
| mapped | 1 | `[(D2, 1)]` |
| postings | 1 | `[(D2, 1)]` |
| quality | 1 | `[(D5, 1)]` |
| rank | 1 | `[(D3, 1)]` |
| reciprocal | 1 | `[(D3, 1)]` |
| rerankers | 1 | `[(D5, 1)]` |
| **retrieval** | **5** | `[(D1, 1), (D2, 1), (D3, 1), (D4, 1), (D5, 1)]` |
| scoring | 1 | `[(D4, 1)]` |
| store | 2 | `[(D1, 1), (D2, 1)]` |
| terms | 1 | `[(D2, 1)]` |
| the | 1 | `[(D4, 1)]` |
| to | 1 | `[(D2, 1)]` |
| vector | 1 | `[(D1, 1)]` |

Self-check: summing `tf` across the posting list for `retrieval` → 5, matches `"retrieval"`
appearing once in each of the five documents. Summing for `for` → 3, matches `"for"` appearing in
D1, D2, D4. Passes.

### Deliverable B — tf-idf for Q = `"lexical retrieval"` per document

`idf(t) = ln(N / df(t))` with `N = 5`.

- `idf(lexical) = ln(5 / 2) = ln(2.5) ≈ 0.9163`
- `idf(retrieval) = ln(5 / 5) = ln(1) = 0`

Per-document score `= Σ_{t ∈ Q ∩ d} tf(t, d) · idf(t)`:

| Doc | `lexical ∈ d?` | `retrieval ∈ d?` | Score | Rank |
|---|---|---|---|---|
| D1 | no | yes (tf=1) | `1 · 0 = 0.0000` | 3 (tied) |
| D2 | no | yes (tf=1) | `0.0000` | 3 (tied) |
| D3 | yes (tf=1) | yes (tf=1) | `1 · 0.9163 + 1 · 0 = 0.9163` | 1 (tied) |
| D4 | yes (tf=1) | yes (tf=1) | `0.9163` | 1 (tied) |
| D5 | no | yes (tf=1) | `0.0000` | 3 (tied) |

Commentary: with the classical `ln(N/df)` idf, `"retrieval"` carries zero information because it
appears in every document. The only differentiating signal in Q is `"lexical"`. Tf-idf cannot
distinguish D3 and D4 because the tokenization gives both a `tf = 1` for `"lexical"` and their
length is not taken into account. That blind spot is exactly what BM25 fixes with its length
normalization.

### Deliverable C — BM25

Formula from chapter 2:

```
score(q, d) = Σ_{t ∈ q ∩ d} idf(t) · (k1 + 1) · tf(t, d)
                              / ( k1 · (1 − b + b · |d|/avgdl) + tf(t, d) )
```

Using the Lucene-smoothed idf:

- `idf(lexical)   = ln(1 + (5 − 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.4)      = ln(2.4)    ≈ 0.8755`
- `idf(retrieval) = ln(1 + (5 − 5 + 0.5) / (5 + 0.5)) = ln(1 + 0.0909)   = ln(1.0909) ≈ 0.0870`

Length-normalization term `B(d) = k1 · (1 − b + b · |d| / avgdl)` with `k1 = 1.2`, `b = 0.75`,
`avgdl = 7.8`:

| Doc | `|d|` | `B(d) = 1.2 · (0.25 + 0.75 · |d| / 7.8)` |
|---|---|---|
| D1 | 7 | `1.1077` |
| D2 | 9 | `1.3385` |
| D3 | 8 | `1.2231` |
| D4 | 9 | `1.3385` |
| D5 | 6 | `0.9923` |

Per-term, per-document contribution with `tf = 1` (every match in this corpus is `tf = 1`):
`contrib(t, d) = idf(t) · (k1 + 1) / (B(d) + 1) = idf(t) · 2.2 / (B(d) + 1)`.

#### Variant 1 — default `k1 = 1.2`, `b = 0.75`

| Doc | Lexical contrib | Retrieval contrib | Score | Rank |
|---|---|---|---|---|
| D1 | — | `0.0870 · 2.2 / 2.1077 = 0.0908` | **0.0908** | 4 |
| D2 | — | `0.0870 · 2.2 / 2.3385 = 0.0818` | **0.0818** | 5 |
| D3 | `0.8755 · 2.2 / 2.2231 = 0.8664` | `0.0870 · 2.2 / 2.2231 = 0.0861` | **0.9525** | 1 |
| D4 | `0.8755 · 2.2 / 2.3385 = 0.8236` | `0.0870 · 2.2 / 2.3385 = 0.0818` | **0.9054** | 2 |
| D5 | — | `0.0870 · 2.2 / 1.9923 = 0.0961` | **0.0961** | 3 |

Default ranking: **D3 > D4 > D5 > D1 > D2**.

Commentary: D3 beats D4 despite both containing the same query terms with the same `tf`, because
D3 is shorter (`|d| = 8` vs. 9) and BM25's `b = 0.75` penalizes the extra length in D4. Among the
no-"lexical" documents, D5 edges D1 edges D2 for the same length-normalization reason — D5 is the
shortest, D2 the longest.

#### Variant 2 — sensitivity check, `b = 0` (no length normalization)

With `b = 0`, `B(d) = k1 · 1 = 1.2` for every document. Then
`contrib(t, d) = idf(t) · 2.2 / (1.2 + 1) = idf(t)` exactly, which makes the arithmetic trivial:

| Doc | Score |
|---|---|
| D1 | `0.0870` |
| D2 | `0.0870` |
| D3 | `0.8755 + 0.0870 = 0.9625` |
| D4 | `0.8755 + 0.0870 = 0.9625` |
| D5 | `0.0870` |

Ranking at `b = 0`: **D3 ≡ D4 > D1 ≡ D2 ≡ D5**.

Commentary: turning off length normalization collapses D3 vs. D4 to a tie (they contain identical
query-term tf) and collapses D1, D2, D5 to a tie. **D5 drops from rank 3 to a tie at rank 3–5**,
losing the small boost its short length gave it under default BM25. This matches chapter 2's
description of `b` as the length-penalty knob.

#### Variant 3 — saturation check, `tf("retrieval", D1) = 6`

Append five extra occurrences of `"retrieval"` to D1. `df("retrieval")` is unchanged (still 5),
so idfs are unchanged. But `|D1|` grows from 7 to 12, and `avgdl` grows from 7.8 to `(39 + 5)/5
= 8.8`. Recompute `B(d)` with the new `avgdl`:

| Doc | `|d|` | `B(d) = 1.2 · (0.25 + 0.75 · |d| / 8.8)` |
|---|---|---|
| D1 | 12 | `1.5273` |
| D2 | 9 | `1.2205` |
| D3 | 8 | `1.1182` |
| D4 | 9 | `1.2205` |
| D5 | 6 | `0.9136` |

| Doc | Lexical contrib | Retrieval contrib | Score | Rank |
|---|---|---|---|---|
| D1 | — | `0.0870 · (2.2 · 6) / (1.5273 + 6) = 0.1526` | **0.1526** | 3 |
| D2 | — | `0.0870 · 2.2 / 2.2205 = 0.0862` | **0.0862** | 5 |
| D3 | `0.8755 · 2.2 / 2.1182 = 0.9093` | `0.0870 · 2.2 / 2.1182 = 0.0904` | **0.9997** | 1 |
| D4 | `0.8755 · 2.2 / 2.2205 = 0.8676` | `0.0870 · 2.2 / 2.2205 = 0.0862` | **0.9538** | 2 |
| D5 | — | `0.0870 · 2.2 / 1.9136 = 0.1000` | **0.1000** | 4 |

Ranking with D1 spammed: **D3 > D4 > D1 > D5 > D2**.

Commentary: D1's score grows from 0.0908 → 0.1526 — a `6×` increase in raw `tf("retrieval")`
yields only a `~1.68×` increase in score. **D1 rises from rank 4 to rank 3 but does not overtake
D3 or D4.** This is the point of BM25: the saturation term (`k1`) prevents term repetition from
scaling the score linearly, and the length normalization (`b`) further penalizes the now-longer
D1. D5 drops from rank 3 to rank 4 because `avgdl` moved up — a document that is short *relative
to the new average* gets a smaller length-normalization boost than it used to.

### Deliverable D — Python script `bm25_smoke.py`

The script below (≤ 30 lines of real code, within the exercise cap) tokenizes the mini-corpus,
scores the query with `rank_bm25.BM25Okapi`, and prints the ranked list.

```python
# bm25_smoke.py — reproduce Part C's BM25 ranking with rank_bm25
# pip install rank_bm25
from rank_bm25 import BM25Okapi

docs = [
    "vector databases store dense embeddings for retrieval",            # D1
    "inverted indexes store terms mapped to postings for retrieval",    # D2
    "reciprocal rank fusion combines lexical and dense retrieval",      # D3
    "bm25 is the classical scoring function for lexical retrieval",     # D4
    "cross encoder rerankers improve retrieval quality",                # D5
]
ids = ["D1", "D2", "D3", "D4", "D5"]

tokenized = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized, k1=1.2, b=0.75)

query = "lexical retrieval".split()
scores = bm25.get_scores(query)

ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
for doc_id, score in ranked:
    print(f"{doc_id}  {score:.4f}")
```

Expected output shape (exact numbers will differ from the hand-computed scores because
`rank_bm25.BM25Okapi` uses the classical BM25 idf `log((N − df + 0.5) / (df + 0.5))` and clamps
the negative `"retrieval"` idf to `epsilon · average_idf`; see commentary below):

```
D3  <positive>
D4  <positive>
D5  <lower>
D1  <lower>
D2  <lower>
```

Commentary vs. hand ranking: the top-2 is stable at **D3 > D4** across the Lucene-smoothed
hand computation and `rank_bm25`. The classical BM25 idf used by `rank_bm25` differs from the
Lucene-smoothed idf used in the hand computation, and this shows up most visibly on `"retrieval"`
(which appears in every document). `rank_bm25` computes `idf(retrieval) = log(0.5/5.5) ≈ -2.398`,
notices it is negative, and overwrites it with `epsilon · average_idf` (`epsilon` defaults to
`0.25`) — in this pathological five-document corpus the average idf is itself negative, so
`"retrieval"` ends up with a small negative idf, which makes the contrast between D3/D4 and the
rest even sharper than in the hand computation. Lucene / Elasticsearch, by contrast, use the
smoothed `log(1 + (N − df + 0.5)/(df + 0.5))` form that keeps all idfs non-negative — this is the
`k1`/`b` / idf story documented in Elastic's *Practical BM25* post. See the Common Mistakes
section for which library choice makes which explanation.

## 3. Validation steps

1. Confirm Deliverable A's posting-list sums match the corpus word counts (shown in the
   self-check row under Deliverable A). This catches off-by-one tokenization errors early.
2. Confirm tf-idf values by hand for one row (D3 is a good one) — the answer should be
   `1 · ln(2.5) + 1 · ln(1) ≈ 0.9163`.
3. Confirm `avgdl = 39 / 5 = 7.8` before computing any `B(d)`. One of the most common hand-
   computation errors is using `|D1| + ... + |D5|` divided by a wrong `N`.
4. Confirm the `b = 0` ranking collapses D3 and D4 to a tie and D1/D2/D5 to a tie; any other
   shape indicates arithmetic drift.
5. Run `python bm25_smoke.py` and confirm the top-2 is `D3, D4` in that order. The exact scores
   will differ from the hand computation — that is expected and discussed in the commentary.

## 4. Rubric or review checklist

- **Deliverable A** — all 30 terms covered with the correct df and `tf = 1` posting lists;
  self-check row (summing posting-list tf) matches total corpus occurrence.
- **Deliverable B** — per-document tf-idf scores match the table above; rankings D3 ≡ D4 at top
  and D1 ≡ D2 ≡ D5 at bottom are explicitly called out.
- **Deliverable C** — three ranked tables (default, `b = 0`, D1-spammed) exist. Default ranking
  matches **D3 > D4 > D5 > D1 > D2** (top-2 is the strict requirement; the exact ordering of the
  no-lexical tail is less important and differs by log base in absolute-score terms but not in
  rank order). Each variant table has a two-sentence commentary that uses the formula to explain
  the shape — not just the plot.
- **Deliverable D** — `bm25_smoke.py` exists, is ≤ 30 lines of real code (comments and blank
  lines do not count), runs, and prints D3 > D4 as the top-2. A commentary paragraph names which
  idf formula the library uses and whether that explains any discrepancy.

Acceptance of a *rank-only* match: the exercise explicitly says scores do not need to match to
four decimal places — only the ranking. A learner whose hand ranking is `D3 > D4` and whose
library ranking is also `D3 > D4` has met the acceptance criterion.

## 5. Common mistakes

- **Wrong idf base.** Using `log10` or `log2` instead of `ln` changes every score by a constant
  factor. Rankings are unaffected, but the learner's numbers will not match library output. Note
  the base chosen and stay consistent.
- **Using `ln(N/df)` for BM25 idf.** Chapter 2's tf-idf uses `ln(N/df)`; BM25 does *not*. Robertson
  & Zaragoza (2009) and real Lucene both use the smoothed `log(1 + (N − df + 0.5)/(df + 0.5))`
  form. Mixing the two produces scores that look wrong until the learner realizes `"retrieval"`
  contributes 0 (not `≈ 0.087`) and the gap between D3 and D5 widens.
- **Forgetting to recompute `avgdl` in the saturation check.** Adding five tokens to D1 changes
  the corpus mean. Leaving `avgdl = 7.8` from Part C gives a plausibly-wrong D5 score and a
  plausibly-wrong D1 score; the ranking might even be correct, hiding the bug.
- **Clamping negative idf without noticing.** `rank_bm25.BM25Okapi` silently replaces negative
  idfs with `epsilon · average_idf`. A learner who hand-computes with the classical formula and
  does not clamp will see library-vs-hand discrepancies concentrated on `"retrieval"` — this is
  the discrepancy the exercise asks the learner to trace to a formula difference, not a bug.
- **Picking Elasticsearch as the comparison without noting its different idf form.** Elastic's
  *Practical BM25* post documents the Lucene smoothed-idf form; `rank_bm25` does not. Learners
  who picked Elasticsearch should get approximately the same *rank* but different absolute scores
  from both the hand computation (if they used `ln(N/df)`) and `rank_bm25`.
- **Claiming D5 is the correct top answer to `"lexical retrieval"`.** It is not — "lexical" is
  not in D5. This usually indicates a copy-paste tokenization error (the learner accidentally
  added "lexical" to D5 in a worksheet).

## 6. References

- Lesson sources in this module:
  - `lessons/mod-101-retrieval-engineering-foundations/02-classical-ir-fundamentals.md` — the
    chapter this exercise refreshes.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
  Foundations and Trends in Information Retrieval 3(4).
  [DOI:10.1561/1500000019](https://dl.acm.org/doi/10.1561/1500000019).
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.*
  Cambridge University Press. Free HTML edition: [nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/).
  Chapters 1 (Boolean retrieval), 6 (scoring and vector space), 11 (probabilistic retrieval / BM25).
- Elastic. *Practical BM25 — Part 2: The BM25 Algorithm and its Variables.*
  [elastic.co/blog/practical-bm25-part-2](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).
  Source for the smoothed idf form and the role of `k1` and `b` in production stacks.
- `rank_bm25` library: [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25).
  Documents the classical idf form used by `BM25Okapi`.
