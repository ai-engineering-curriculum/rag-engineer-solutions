# Exercise 02 — Solution: Compute recall@k, MRR, and nDCG with `ir_measures`

This is a reference solution for
`exercise-02-compute-recall-mrr-ndcg-with-ir-measures.md`. It is an
**implementation** exercise. The learner wires exercise-01's qrels
into the `ir_measures` library, scores one or more retrieval runs,
and produces a metrics table that downstream exercises (03
significance, 04 ablation) build on.

The teaching point: retrieval metrics have crisp definitions, but
every production codebase reinvents them slightly wrong (off-by-one
in the cutoff, binarising a graded qrels, wrong log base on nDCG,
averaging across queries with missing results). Use the standard
library. The exercise is therefore less about *implementing* the
metrics and more about *using the library correctly* and
understanding what each number tells you.

## 1. Solution overview

The deliverable has four parts:

1. **`src/score_run.py`** — the scorer. Loads qrels (TREC format)
   and a run (TREC format), computes recall@{10,20,50,100},
   MRR@{10,100}, and nDCG@{10,20,100} with `ir_measures`, emits
   both an aggregate table and a per-query CSV.
2. **`src/formats.py`** — thin loaders / validators for the TREC
   qrels and run formats, with the specific edge cases
   (duplicate `(qid, doc_id)`, graded vs binary, missing qids in
   the run) called out.
3. **`reports/metrics_table.md`** — the aggregate report (mean,
   min, max, n-queries-with-no-hits) rendered from the per-query
   CSV.
4. **`notes/metric_choices.md`** — a one-page memo on when each
   metric is the right number to show in a review, and the
   specific traps for each (recall@k vs precision@k, MRR under
   graded qrels, nDCG with missing cutoffs).

The exercise is scoped to *scoring and reporting*, not to running
retrievals. The input runs come from mod-105 (BM25), mod-106
(hybrid), and mod-107 (rerank); exercise-02 is agnostic to how
they were produced.

## 2. Worked answer or implementation

### 2.1 The three metrics in one paragraph

**Recall@k** is the fraction of relevant documents (per qrels)
that appear in the top-k of the run, averaged across queries.
It cares only about presence/absence in top-k, not about position
within it. **MRR** (Mean Reciprocal Rank) is 1/rank-of-first-
relevant, averaged across queries; it cares about the position of
the *first* relevant document and ignores everything below. **nDCG@k**
(Järvelin & Kekäläinen, 2002) sums graded gains discounted by
`log2(rank+1)` over the top-k, then divides by the ideal ranking's
sum; it cares about both position and grade and is the right
number for *graded* qrels like the one from exercise-01. Use
recall@k to measure "did the right doc even make it into the
candidate set," MRR to measure "did the first good answer come
up fast," and nDCG to measure "is the ordering correct."

### 2.2 `src/formats.py` — TREC qrels and run formats

```python
"""Loaders and validators for TREC-format qrels and runs.
See https://trec.nist.gov/trec_eval/ for the canonical format."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class QrelsLine:
    query_id: str
    doc_id: str
    grade: int

@dataclass(frozen=True)
class RunLine:
    query_id: str
    doc_id: str
    rank: int          # 1-indexed; ir_measures also accepts scores and recomputes rank
    score: float
    tag: str           # free-text system identifier; conventionally a short slug

def load_qrels(path: Path) -> list[QrelsLine]:
    """Parse a TREC qrels file: `qid 0 doc_id grade`.
    Blank lines and lines starting with `#` are ignored."""
    out: list[QrelsLine] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 columns, got {len(parts)}"
                )
            qid, _iter, doc_id, grade_s = parts
            key = (qid, doc_id)
            if key in seen:
                raise ValueError(
                    f"{path}:{lineno}: duplicate (query_id, doc_id) = {key}"
                )
            seen.add(key)
            out.append(QrelsLine(qid, doc_id, int(grade_s)))
    return out

def load_run(path: Path) -> list[RunLine]:
    """Parse a TREC run file:
      `qid Q0 doc_id rank score tag`
    """
    out: list[RunLine] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{path}:{lineno}: expected 6 columns, got {len(parts)}"
                )
            qid, _q0, doc_id, rank_s, score_s, tag = parts
            out.append(
                RunLine(qid, doc_id, int(rank_s), float(score_s), tag)
            )
    return out

def coverage_report(
    qrels: list[QrelsLine], run: list[RunLine]
) -> dict:
    qids_q = {q.query_id for q in qrels}
    qids_r = {r.query_id for r in run}
    missing_from_run = sorted(qids_q - qids_r)
    extra_in_run = sorted(qids_r - qids_q)
    by_q: dict[str, int] = defaultdict(int)
    for r in run:
        by_q[r.query_id] += 1
    short_qids = sorted(q for q, n in by_q.items() if n < 10)
    return {
        "qrels_qids": len(qids_q),
        "run_qids": len(qids_r),
        "missing_from_run": missing_from_run,
        "extra_in_run": extra_in_run,
        "qids_with_fewer_than_10_results": short_qids,
    }
```

**Choices and why:**

- **Duplicate detection on qrels.** The TREC spec treats the last
  duplicate as authoritative; most scorers silently take the
  first. We refuse both by raising — if your qrels builder emits
  duplicates, you have a bug in exercise-01, not an edge case to
  swallow.
- **Coverage report is a first-class output.** Half of "my
  metrics look wrong" is "the run has no results for 7 queries,
  which were silently treated as recall=0 and dragged the mean
  down." Make it visible before running the scorer.

### 2.3 `src/score_run.py` — the scorer

```python
"""Score a TREC run against TREC qrels using ir_measures.
Emits an aggregate table and a per-query CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import ir_measures
from ir_measures import Recall, nDCG, RR

from .formats import load_qrels, load_run

AGG_METRICS = [
    Recall @ 10, Recall @ 20, Recall @ 50, Recall @ 100,
    RR @ 10, RR @ 100,
    nDCG @ 10, nDCG @ 20, nDCG @ 100,
]

def _as_ir_tuples(qrels, run):
    """ir_measures accepts iterables of (query_id, doc_id, relevance)
    and (query_id, doc_id, score)."""
    qrels_iter = (
        ir_measures.Qrel(q.query_id, q.doc_id, q.grade) for q in qrels
    )
    run_iter = (
        ir_measures.ScoredDoc(r.query_id, r.doc_id, r.score) for r in run
    )
    return qrels_iter, run_iter

def score_aggregate(qrels_path: Path, run_path: Path) -> dict:
    qrels = load_qrels(qrels_path)
    run = load_run(run_path)
    q_iter, r_iter = _as_ir_tuples(qrels, run)
    return ir_measures.calc_aggregate(AGG_METRICS, q_iter, r_iter)

def score_per_query(qrels_path: Path, run_path: Path) -> list[dict]:
    qrels = load_qrels(qrels_path)
    run = load_run(run_path)
    q_iter, r_iter = _as_ir_tuples(qrels, run)
    rows: dict[str, dict] = {}
    for m in ir_measures.iter_calc(AGG_METRICS, q_iter, r_iter):
        row = rows.setdefault(m.query_id, {"query_id": m.query_id})
        row[str(m.measure)] = m.value
    return [rows[qid] for qid in sorted(rows)]

def write_per_query_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fields = ["query_id"] + [k for k in rows[0] if k != "query_id"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def render_markdown_table(agg: dict) -> str:
    rows = ["| Metric | Mean |", "|---|---|"]
    for m in AGG_METRICS:
        rows.append(f"| {m} | {agg[m]:.4f} |")
    return "\n".join(rows) + "\n"
```

**Choices and why:**

- **`ir_measures.calc_aggregate` and `iter_calc` for per-query.**
  Rolling your own averaging is the usual bug source — queries
  with no relevant docs in qrels are divide-by-zero on recall,
  queries with no results in the run should count as 0 (not be
  skipped). The library gets these edges right; the instruction
  is "use it."
- **Score column, not rank column.** `ir_measures` will re-sort
  from the score to break ties deterministically. Feeding ranks
  instead is legal but shifts tie-breaking to the run file,
  which production pipelines frequently get wrong.
- **nDCG cutoffs 10, 20, 100.** 10 for user-visible top-of-page
  quality, 20 as the typical "page 2" cutoff, 100 for ceiling
  under heavy rerank. Pick the one your product surface
  actually renders; shipping nDCG@3 when the UI shows 20 is a
  vanity metric.
- **Recall @ 50 / @ 100.** These are *candidate-set quality*
  metrics; they tell you whether rerank (mod-107) has anything
  to work with. Recall@10 without recall@50 is incomplete —
  you cannot tell a retrieval problem from a rerank problem.

### 2.4 Example run — scoring BM25 vs hybrid vs rerank

With exercise-01's qrels and three runs (`bm25.trec`,
`hybrid.trec`, `rerank.trec`):

```
$ python -m rag_eval.score_run \
    --qrels qrels/judgment_set.trec \
    --run runs/bm25.trec \
    --out reports/bm25/

[coverage] qrels_qids=100 run_qids=100 missing=0 short=0
[write]    reports/bm25/per_query.csv
[write]    reports/bm25/metrics_table.md
```

`reports/bm25/metrics_table.md`:

```markdown
| Metric | Mean |
|---|---|
| R@10 | 0.6212 |
| R@20 | 0.7418 |
| R@50 | 0.8630 |
| R@100 | 0.9204 |
| RR@10 | 0.5841 |
| RR@100 | 0.5912 |
| nDCG@10 | 0.5327 |
| nDCG@20 | 0.5706 |
| nDCG@100 | 0.6133 |
```

`reports/hybrid/metrics_table.md`:

```markdown
| Metric | Mean |
|---|---|
| R@10 | 0.7405 |
| R@20 | 0.8311 |
| R@50 | 0.9013 |
| R@100 | 0.9418 |
| RR@10 | 0.6922 |
| RR@100 | 0.6971 |
| nDCG@10 | 0.6471 |
| nDCG@20 | 0.6803 |
| nDCG@100 | 0.7108 |
```

`reports/rerank/metrics_table.md`:

```markdown
| Metric | Mean |
|---|---|
| R@10 | 0.7818 |
| R@20 | 0.8511 |
| R@50 | 0.9013 |
| R@100 | 0.9418 |
| RR@10 | 0.7563 |
| RR@100 | 0.7604 |
| nDCG@10 | 0.7205 |
| nDCG@20 | 0.7412 |
| nDCG@100 | 0.7489 |
```

(These numbers are *illustrative*. The real pipeline pins its
qrels version and run tags before reporting.)

**What the table shows:**

- Hybrid beats BM25 on every metric. Expected.
- Rerank beats hybrid on nDCG@10 and MRR but is **identical** on
  recall@50/100 — mechanically correct, because rerank cannot
  add documents, only reorder them. Reporting rerank's
  recall@100 as a win would be a bug.
- Headline for a review: nDCG@10 and recall@50. They together
  cover "ordering of the user-visible slice" and "candidate-set
  quality for downstream rerank." Everything else is diagnostic.

### 2.5 `notes/metric_choices.md` — when to show which number

| Question | Right metric | Trap |
|---|---|---|
| "Is the correct document in the top-10?" | Recall @ 10 (binarised graded qrels) | Binarisation choice matters: `grade≥2` is a common default, but if "grade 1" is your refusal-flag case, treat it carefully. |
| "How quickly does the user see a good answer?" | MRR @ 10 | MRR ignores the second relevant doc; a system that puts one great and nine garbage results at the top beats a system that puts five decent results at the top. If the UI surfaces 10 items, nDCG@10 is a better product metric. |
| "Is the ordering of the user-visible slice correct?" | nDCG @ 10 | Uses log2 discounting by default. Report the log base if you deviate from the standard. |
| "Does the rerank candidate set have the right documents?" | Recall @ 50 / @ 100 | Measures the *first-stage retriever*, not the reranker. A rerank-only improvement here is impossible by construction. |
| "Is the system strictly better than the baseline?" | Per-query deltas → exercise-03 | A mean improvement is not a reliable improvement. Paired tests live in the next exercise. |

## 3. Validation steps

1. `score_run.py` runs without error against exercise-01's qrels
   and at least one TREC-format run.
2. The coverage report flags any query in the qrels that is
   missing from the run, and any query in the run with fewer
   than the maximum cutoff worth of results.
3. The aggregate output uses `ir_measures.calc_aggregate` —
   no hand-rolled averaging.
4. nDCG uses the library default log2 discount and ideal-ranking
   normalisation; if the learner deviates (e.g., "exp" gain
   formulation), the choice is documented in
   `notes/metric_choices.md`.
5. The per-query CSV has one row per query_id in the qrels, with
   cells for every (metric, cutoff) in `AGG_METRICS`.
6. Rerank does not beat first-stage retrieval on recall@K when
   K ≥ the rerank input depth. If it does, the run file is wrong
   (the rerank tag is leaking in documents the first stage did
   not produce).
7. `score_run.py` is deterministic across re-runs on the same
   qrels / run (no RNG, no network).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Library usage | `ir_measures` is used; metrics are constructed as `Recall @ 10` / `nDCG @ 10` style, not hand-rolled. |
| Cutoffs | Recall at ≥ one small cutoff (10 or 20) **and** ≥ one large (50 or 100). nDCG at the UI cutoff. MRR at a sensible cap. |
| Graded vs binary | Graded qrels drive nDCG directly; recall uses a named binarisation threshold. |
| Per-query output | CSV with one row per query_id, one column per (metric, cutoff). Downstream exercises consume this directly. |
| Coverage report | Flags missing qids, short runs, and extra qids in the run not in qrels. |
| Reports | Aggregate markdown table includes mean, with min/max optional. |
| Memo | `notes/metric_choices.md` names the headline metrics and the common traps. |
| Reproducibility | Re-running on the same inputs produces byte-identical outputs. |

Fail-fast smells:

- Averaging recall "only over queries with at least one hit"
  silently makes a system that fails half the queries look
  great. `ir_measures` treats no-hit as recall=0 by default —
  do not undo this.
- Reporting rerank's recall@100 as an improvement. Rerank
  cannot change recall at depths beyond its input; the gain is
  a bug in your run file or in your scoring harness.
- nDCG with a non-standard log base undocumented. The number is
  not comparable to anything in the literature.
- Scoring with binarised qrels (`grade≥1`) under nDCG. You lose
  the graded signal the metric was designed to reward.
- Comparing nDCG@10 between runs where one run has fewer than
  10 results per query. The missing positions are silent zeros;
  the delta is not what you think it is.

## 5. Common mistakes

- **Hand-rolling the metric.** Every codebase has a
  `def ndcg_at_k(ranks, grades, k): ...` that is off by one
  somewhere. Use `ir_measures`; the library is small, fast, and
  correct. (Writing it once as a homework exercise to
  understand the formula is fine — shipping it is not.)
- **Mixing grade scales.** Qrels on 0–3, hard-coded `grade>0`
  for recall, and `grade` passed through for nDCG. Decide on
  one grade scale and one binarisation and document both.
- **Reading rank, not score.** Ranks in the run file depend on
  the producing system's tie-break. Scores are the source of
  truth; let `ir_measures` compute the rank.
- **Ignoring short runs.** A run with 7 results per query will
  score recall@50 as "the 7 that were there / all relevant" —
  mechanically fine, but shipping it as a win vs. a 50-result
  run is a bad-faith comparison. The coverage report exists to
  stop you.
- **Reporting the mean without per-query data.** Exercise-03
  needs per-query values to run paired tests. Emit them in
  this exercise; do not re-score in exercise-03.
- **Comparing across different qrels versions.** Qrels v1 and
  qrels v2 numbers are not comparable. Pin the qrels version
  in the metrics table header.
- **Averaging across systems.** "Mean nDCG across BM25,
  hybrid, rerank" is not a meaningful number. Report each
  system, then compare.

## 6. References

- Metric definitions:
  - Järvelin, K. & Kekäläinen, J. (2002), "Cumulated gain-based
    evaluation of IR techniques," *ACM TOIS* — canonical nDCG.
  - Voorhees, E. (1999), "The TREC-8 Question Answering Track
    Report" — introduces MRR for QA-style evaluation.
  - Manning, C., Raghavan, P., Schütze, H. (2008),
    *Introduction to Information Retrieval*, Cambridge —
    chapter 8 covers recall/precision at k.
- Tooling:
  - `ir_measures` library and CLI —
    [ir-measur.es](https://ir-measur.es/). Thin wrapper around
    `trec_eval` and its reference metric implementations.
  - NIST `trec_eval` —
    [trec.nist.gov/trec_eval/](https://trec.nist.gov/trec_eval/).
    Reference implementation; the format definitions come
    from here.
- Formats:
  - TREC qrels format — see `trec_eval` README.
  - TREC run file (TREC "trec_submit" / "whitespace" format) —
    six-column: `qid Q0 doc_id rank score tag`.
- Risk and governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML06 Output Integrity — a metric reported
    from an unvalidated scoring harness is a trusted-output
    surface the attacker does not even need to touch.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around "Evaluation" motivate pinning qrels and
    tooling versions in the metrics table header.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure* function requires repeatable metrics with
    named uncertainty — exercise-03 adds the uncertainty; this
    exercise contributes repeatability.
