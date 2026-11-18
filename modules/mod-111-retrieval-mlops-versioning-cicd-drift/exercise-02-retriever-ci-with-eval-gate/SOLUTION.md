# Exercise 02 — Solution: Retriever CI with an Eval Gate

This is a reference solution for
`exercise-02-retriever-ci-with-eval-gate.md`. It is an
**implementation** exercise. The learner wires an offline retrieval
evaluation (the metrics from mod-109 exercises 02–03 and the judgment
set from mod-109 exercise-01) into a pull-request gate so that a
change to the retriever, the chunker, the embedding model, or the
index config cannot merge if it regresses measured retrieval quality
beyond a budget.

The teaching point: *retriever quality has no unit test — it has an
offline eval*. You cannot assert `retriever.search("...") == [...]`
because the "right" answer is a ranked set over human-judged
relevance, not a literal equality. What you *can* assert is that a
small, versioned judgment set still produces metrics within a budget.
The eval gate is the CI job that makes that assertion. Without it,
every retrieval change is merged on vibes and the only feedback is
user complaints in production.

## 1. Solution overview

The deliverable has five parts:

1. **`eval/qrels/`** — the versioned judgment set from mod-109
   exercise-01, checked into the repo (small: ~100–500 queries,
   graded relevance, TREC format). Pinned to a `qrels_sha` so the
   gate's inputs are reproducible.
2. **`eval/run_eval.py`** — the scorer. Given a retriever
   implementation (imported as a Python entry point), a corpus
   snapshot, and the qrels, produces `metrics.json`, `per_query.csv`,
   and `latency.json`. Uses `ir_measures` for metrics (mod-109
   exercise-02) and a stopwatch around each query for latency.
3. **`eval/gate.py`** — the gate itself. Compares the current
   branch's `metrics.json` against a baseline (`main` branch's
   metrics checked into `eval/baseline/metrics.json`). Applies
   the budget. Exits 0 (pass) or non-zero (fail) with a diffed
   report.
4. **`.github/workflows/retriever-ci.yml`** — the GitHub Actions
   workflow that runs the eval on every PR that touches retrieval
   code, reports the diff as a PR comment, and blocks merge if
   the gate fails.
5. **`eval/BUDGET.md`** — the written contract: which metrics
   gate the build, what the budget is for each, how overrides
   work, what triggers a baseline refresh.

Scope: *the CI gate only*. The qrels construction lives in
mod-109 exercise-01; metric semantics live in mod-109 exercise-02;
significance testing lives in mod-109 exercise-03. This exercise
wires those together behind a `main`-branch gate.

## 2. Worked answer or implementation

### 2.1 Why a gate and not a dashboard

A dashboard shows regressions *after* they ship; a gate prevents
them *before*. For retriever work the gate matters more than for
most ML changes because:

- A retriever regression does not usually crash — answers just get
  worse. Error monitors miss it entirely.
- The surface for regression is wide: prompt changes, chunker
  changes, filter changes, re-ranker weight changes, ANN-param
  changes, embedding-model swaps, and silent data drift in the
  corpus. All of them land as PRs; all of them should hit the same
  gate.
- Offline-eval runtime is seconds-to-minutes for a judgment set of a
  few hundred queries. There is no latency reason to defer the
  gate to a nightly job.

NIST AI RMF **MEASURE 2.3** (evaluation of AI systems against
documented metrics) and **MANAGE 2.3** (continuous monitoring of
system performance) explicitly call for metric-based release gates
on AI components — the eval gate is the implementation of those
controls at CI time.

### 2.2 `eval/run_eval.py` — the scorer

```python
"""Run a retriever against the qrels and emit metrics + per-query + latency."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from statistics import mean, median, quantiles

import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG, R, RR


def load_qrels(path: Path) -> list[Qrel]:
    """TREC qrels format: `qid 0 doc_id grade`."""
    out: list[Qrel] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        qid, _, doc_id, grade = line.split()
        out.append(Qrel(query_id=qid, doc_id=doc_id, relevance=int(grade)))
    return out


def load_queries(path: Path) -> dict[str, str]:
    """TSV: `qid\\tquery_text`."""
    return dict(
        line.split("\t", 1)
        for line in path.read_text().splitlines()
        if line.strip()
    )


def resolve_retriever(entry: str):
    """`module.path:callable` returning a Retriever instance."""
    mod_name, obj_name = entry.split(":")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, obj_name)
    return factory()


def run(retriever, queries: dict[str, str], k: int) -> tuple[list[ScoredDoc], list[float]]:
    runs: list[ScoredDoc] = []
    latencies_ms: list[float] = []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        hits = retriever.search(text, k=k)  # List[(doc_id, score)]
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        for doc_id, score in hits:
            runs.append(ScoredDoc(query_id=qid, doc_id=doc_id, score=float(score)))
    return runs, latencies_ms


def summarise_latency(latencies_ms: list[float]) -> dict[str, float]:
    p50, p95, p99 = quantiles(latencies_ms, n=100)[49], quantiles(latencies_ms, n=100)[94], quantiles(latencies_ms, n=100)[98]
    return {
        "mean_ms":   round(mean(latencies_ms), 2),
        "median_ms": round(median(latencies_ms), 2),
        "p50_ms":    round(p50, 2),
        "p95_ms":    round(p95, 2),
        "p99_ms":    round(p99, 2),
        "n":         len(latencies_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True, help="module.path:factory")
    ap.add_argument("--qrels",     required=True, type=Path)
    ap.add_argument("--queries",   required=True, type=Path)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--k",         type=int, default=100)
    args = ap.parse_args()

    qrels   = load_qrels(args.qrels)
    queries = load_queries(args.queries)
    retr    = resolve_retriever(args.retriever)

    runs, latencies_ms = run(retr, queries, args.k)

    measures = [R @ 10, R @ 20, R @ 100, nDCG @ 10, nDCG @ 20, RR @ 10]
    aggregate = ir_measures.calc_aggregate(measures, qrels, runs)
    per_query = {
        str(m): {qid: v for qid, v in ir_measures.iter_calc([m], qrels, runs).items()}
        for m in measures
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(
        {str(k): round(v, 4) for k, v in aggregate.items()}, indent=2, sort_keys=True,
    ))
    (args.out / "per_query.json").write_text(json.dumps(per_query, indent=2, sort_keys=True))
    (args.out / "latency.json").write_text(json.dumps(summarise_latency(latencies_ms), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Notes:

- The retriever is passed as an import path, not as code. That
  means the gate runs against *whatever retriever implementation
  the branch under test ships* — no bespoke CI retriever that drifts
  from production.
- `ir_measures` is the standard library for IR metrics
  ([Macdonald et al. 2022](https://ir-measur.es/)); handles graded
  qrels correctly (nDCG with `log2(rank+1)` discount, standard
  binarisation for recall). Using the library avoids the common
  per-codebase re-implementations that disagree at the fourth
  decimal.
- Latency is captured per query, aggregated to p50/p95/p99.
  Retrieval quality and latency regress together often enough that
  the gate must cover both.

### 2.3 `eval/gate.py` — the gate

```python
"""Compare current metrics against baseline; apply the budget; exit 0/1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BUDGET = {
    # metric -> (max_relative_regression, max_absolute_regression)
    # The gate uses whichever is tighter at runtime.
    "R@10":    (0.02, 0.01),   # ≤ 2% relative drop OR ≤ 1pp absolute drop
    "R@20":    (0.02, 0.01),
    "R@100":   (0.02, 0.01),
    "nDCG@10": (0.03, 0.015),  # ≤ 3% relative drop OR ≤ 1.5pp absolute drop
    "nDCG@20": (0.03, 0.015),
    "RR@10":   (0.03, 0.015),
}

LATENCY_BUDGET = {
    "p95_ms": 1.15,  # ≤ +15% vs baseline
    "p99_ms": 1.25,  # ≤ +25% vs baseline
}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def evaluate_metric(metric: str, baseline: float, current: float) -> tuple[bool, str]:
    rel_budget, abs_budget = BUDGET[metric]
    delta_abs = current - baseline
    delta_rel = delta_abs / baseline if baseline else 0.0
    rel_ok = delta_rel >= -rel_budget
    abs_ok = delta_abs >= -abs_budget
    ok = rel_ok or abs_ok  # pass if EITHER tolerance holds (small bases)
    sign = "+" if delta_abs >= 0 else ""
    msg = (f"{metric:>8}: {baseline:.4f} -> {current:.4f} "
           f"({sign}{delta_abs:+.4f}, {sign}{delta_rel * 100:+.2f}%)")
    return ok, msg


def evaluate_latency(name: str, baseline: float, current: float) -> tuple[bool, str]:
    ratio = current / baseline if baseline else float("inf")
    ok = ratio <= LATENCY_BUDGET[name]
    msg = (f"{name:>8}: {baseline:.2f}ms -> {current:.2f}ms "
           f"(x{ratio:.2f}, budget x{LATENCY_BUDGET[name]:.2f})")
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--current",  required=True, type=Path)
    ap.add_argument("--override", action="store_true",
                    help="Advisory mode: report but do not fail. "
                         "Requires BUDGET_OVERRIDE_REASON env var.")
    args = ap.parse_args()

    base_m = load(args.baseline / "metrics.json")
    cur_m  = load(args.current / "metrics.json")
    base_l = load(args.baseline / "latency.json")
    cur_l  = load(args.current / "latency.json")

    failures: list[str] = []
    report:   list[str] = []

    report.append("## Quality")
    for metric in BUDGET:
        if metric not in base_m or metric not in cur_m:
            failures.append(f"missing metric {metric}")
            continue
        ok, msg = evaluate_metric(metric, base_m[metric], cur_m[metric])
        report.append(("  OK " if ok else "FAIL ") + msg)
        if not ok:
            failures.append(msg)

    report.append("\n## Latency")
    for name in LATENCY_BUDGET:
        ok, msg = evaluate_latency(name, base_l[name], cur_l[name])
        report.append(("  OK " if ok else "FAIL ") + msg)
        if not ok:
            failures.append(msg)

    print("\n".join(report))

    if failures and not args.override:
        print(f"\nGate FAILED ({len(failures)} regressions).")
        return 1
    if failures and args.override:
        import os
        reason = os.environ.get("BUDGET_OVERRIDE_REASON", "").strip()
        if not reason:
            print("\nOverride requested but BUDGET_OVERRIDE_REASON is empty.")
            return 1
        print(f"\nGate overridden with reason: {reason!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Why "either relative OR absolute":

- A metric with a small base (e.g. `nDCG@10 = 0.08`) will fail a
  2%-relative gate on noise (0.08 → 0.078 is −2.5%). The absolute
  tolerance (0.015) shields against that noise.
- A metric with a large base (e.g. `R@100 = 0.92`) will pass any
  absolute-only tolerance trivially. The relative tolerance holds
  the gate meaningful at the high end.
- Taking the *more generous* of the two means the gate fails only
  when both views agree the drop is real. Add exercise-03's
  bootstrap CI if you want statistical significance in the gate
  itself.

### 2.4 Paired bootstrap significance in the gate (optional)

For a stricter gate, replace the "relative or absolute" tolerance
with a paired bootstrap confidence interval on the per-query delta
(from mod-109 exercise-03). Pseudocode:

```python
# per-query delta for one metric; qids present in both runs
deltas = [cur_pq[q] - base_pq[q] for q in sorted(base_pq) if q in cur_pq]
# 10_000 bootstrap resamples with replacement
import random
random.seed(42)
rng = random.Random(42)
means = []
for _ in range(10_000):
    sample = [rng.choice(deltas) for _ in range(len(deltas))]
    means.append(sum(sample) / len(sample))
means.sort()
lo, hi = means[int(0.025 * len(means))], means[int(0.975 * len(means))]
# FAIL if the 95% CI is entirely below -tolerance, i.e., the regression is real.
ok = hi >= -ABS_TOL  # some overlap with 'not-regressed' zone
```

This adds ~5 seconds of CI time per metric for a 500-query set and
eliminates the noisy-run false alarms that make a strict gate
unpopular.

### 2.5 `.github/workflows/retriever-ci.yml` — the CI job

```yaml
name: retriever-ci
on:
  pull_request:
    paths:
      - "src/retriever/**"
      - "src/chunker/**"
      - "src/embeddings/**"
      - "src/index/**"
      - "eval/**"
      - "config/retrieval/**"

permissions:
  contents: read
  pull-requests: write  # for the PR-comment

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      PYTHONHASHSEED: "0"
      TOKENIZERS_PARALLELISM: "false"
    steps:
      - name: Checkout PR
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install deps
        run: pip install -r requirements.txt -r requirements-eval.txt

      - name: Restore corpus + index artifacts
        uses: actions/cache@v4
        with:
          key: corpus-${{ hashFiles('eval/corpus_lock.json') }}
          path: .cache/corpus/
        # Corpus is a tiny evaluation shard (~1k docs). Full
        # production corpus is not fetched in CI.

      - name: Run eval
        run: |
          python -m eval.run_eval \
            --retriever src.retriever.factory:build \
            --qrels   eval/qrels/judgments.trec \
            --queries eval/qrels/queries.tsv \
            --out     eval/out/pr

      - name: Fetch baseline from main
        run: |
          mkdir -p eval/out/baseline
          curl -fsSL \
            "https://raw.githubusercontent.com/${{ github.repository }}/main/eval/baseline/metrics.json" \
            > eval/out/baseline/metrics.json
          curl -fsSL \
            "https://raw.githubusercontent.com/${{ github.repository }}/main/eval/baseline/latency.json" \
            > eval/out/baseline/latency.json

      - name: Apply gate
        id: gate
        run: |
          python -m eval.gate \
            --baseline eval/out/baseline \
            --current  eval/out/pr \
            | tee eval/out/report.md
        continue-on-error: true  # we want to post the comment even on fail

      - name: Post PR comment
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: retriever-eval
          path: eval/out/report.md

      - name: Fail job on gate failure
        if: steps.gate.outcome == 'failure'
        run: exit 1

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-results
          path: eval/out/
```

Notes:

- `paths:` filter keeps the gate off PRs that do not touch
  retrieval — docs, unrelated backend, frontend. If a change moves
  retrieval-adjacent code (e.g., prompt templates), add the path.
- `PYTHONHASHSEED: "0"` and the bootstrap's fixed seed keep metric
  results reproducible across CI runs.
- Corpus for the gate is a *shard* — a few hundred to a few
  thousand documents that cover the qrels. The full production
  corpus (millions of docs) is not downloaded in CI; the eval
  evaluates relative quality on a stable sample. See mod-109
  exercise-01 for how the sample is sized.
- The baseline is pulled from `main`. Any merge to `main` updates
  `eval/baseline/metrics.json` via a separate workflow (not shown)
  that runs the eval on `main`'s HEAD and pushes the result.
- `continue-on-error` + separate `exit 1` step is the pattern that
  lets the PR get its comment *and* turn red on failure. Without
  it, a failing gate skips the comment.
- `sticky-pull-request-comment@v2` is the common action for
  comment-upsert behavior; alternatives:
  [peter-evans/create-or-update-comment](https://github.com/peter-evans/create-or-update-comment),
  or a two-step `gh pr comment`.

### 2.6 `eval/BUDGET.md` — the written contract

```markdown
# Retriever eval-gate budget

## Metrics in the gate

| Metric    | Tolerance (relative) | Tolerance (absolute) |
|-----------|---------------------:|---------------------:|
| R@10      | 2%                   | 0.01                 |
| R@20      | 2%                   | 0.01                 |
| R@100     | 2%                   | 0.01                 |
| nDCG@10   | 3%                   | 0.015                |
| nDCG@20   | 3%                   | 0.015                |
| RR@10     | 3%                   | 0.015                |

The gate fails if, for any metric, both tolerances are breached.
Rationale: see `src/eval/gate.py` docstring.

## Latency

| p-tile   | Budget                                |
|----------|---------------------------------------|
| p95      | ≤ +15% vs baseline                    |
| p99      | ≤ +25% vs baseline                    |

## Override procedure

A gate failure may be overridden only by someone outside the PR
author's team, and only if:

1. The `BUDGET_OVERRIDE_REASON` env var is set to an explanation
   of what the trade is (e.g., "swapping to text-embedding-3-large,
   nDCG@10 +5pp justifies R@100 −3pp").
2. The override is recorded in the PR body with the baseline and
   current numbers inline.
3. The PR updates `eval/baseline/` on merge so subsequent PRs
   measure against the new baseline, not the old one.

Un-reviewed overrides are the top cause of silent regression; the
gate is only as strong as the review discipline around its
overrides.

## When to refresh the baseline

- An embedding model swap, an index-engine swap, or a chunker
  change that is *intended* to move quality. The PR that lands the
  change is the one that refreshes the baseline.
- A corpus-shard refresh (the small CI corpus grows with the real
  corpus). Check in the new shard, re-score, update baseline.

Never refresh the baseline "to make the gate pass." That is the
override procedure's job and should leave a visible record.

## When to expand the qrels

The qrels are versioned in `eval/qrels/` with a `qrels_sha` in
each metric record. Expand when:

- A production slice is under-represented (see exercise-06 drift
  reports).
- A new feature adds a query shape not in the qrels (e.g., code
  retrieval added on top of text retrieval).

Follow mod-109 exercise-01 for labeling. Pin the new `qrels_sha`
in the baseline.
```

### 2.7 Worked example of a passing and failing run

Passing PR (reranker weight tune):

```
## Quality
  OK    R@10: 0.6420 -> 0.6511 (+0.0091, +1.42%)
  OK    R@20: 0.7714 -> 0.7733 (+0.0019, +0.25%)
  OK   R@100: 0.9018 -> 0.9018 (+0.0000, +0.00%)
  OK  nDCG@10: 0.5102 -> 0.5244 (+0.0142, +2.78%)
  OK  nDCG@20: 0.5331 -> 0.5445 (+0.0114, +2.14%)
  OK   RR@10: 0.6301 -> 0.6442 (+0.0141, +2.24%)

## Latency
  OK   p95_ms: 48.20ms -> 51.80ms (x1.07, budget x1.15)
  OK   p99_ms: 85.40ms -> 92.10ms (x1.08, budget x1.25)
```

Failing PR (changed chunker overlap from 128 → 0 to "save tokens"):

```
## Quality
FAIL  R@10: 0.6420 -> 0.6220 (-0.0200, -3.11%)   <-- rel breach, abs breach (0.02 > 0.01)
  OK  R@20: 0.7714 -> 0.7640 (-0.0074, -0.96%)
  OK R@100: 0.9018 -> 0.8990 (-0.0028, -0.31%)
FAIL nDCG@10: 0.5102 -> 0.4910 (-0.0192, -3.76%)  <-- rel breach (3.76% > 3%)
  OK nDCG@20: 0.5331 -> 0.5250 (-0.0081, -1.52%)
  OK  RR@10: 0.6301 -> 0.6180 (-0.0121, -1.92%)

## Latency
  OK   p95_ms: 48.20ms -> 46.80ms (x0.97, budget x1.15)

Gate FAILED (2 regressions).
```

The first regression is the gate working as intended; the PR
author now has to either (a) justify and override with a reason,
or (b) revert the change.

## 3. Validation steps

A correct submission can be validated with these runnable checks:

1. **Golden pass.** On `main` (unchanged retriever), the gate run
   compares `baseline/` to `pr/` and prints "Gate PASSED" with
   zero regressions (deltas are at most sub-noise).
2. **Injected regression.** Edit the retriever to return `hits[:5]`
   on a `k=10` call. Re-run — recall@10 drops ~5pp, nDCG@10 drops
   similarly. Gate must fail on at least `R@10` and `nDCG@10`.
3. **Injected noise regression.** Shuffle the top-5 of each hit
   list. Gate should still *pass* in most cases if the ordering
   change is within the metric budget (sanity check: the budget is
   not so tight that noise fails it).
4. **Override path.** Run the gate on the injected-regression
   branch with `--override` and `BUDGET_OVERRIDE_REASON="test
   override — see RET-999"`. Exit 0, report printed, reason echoed
   back. Without the env var the override must *still* fail.
5. **Latency budget.** Wrap the retriever with a `time.sleep(0.1)`
   per call. Re-run — p95 goes from ~50 ms to ~150 ms, well over
   the 1.15× budget. Gate must fail on `p95_ms`.
6. **Determinism.** Run the eval twice on the same branch without
   any code change. `metrics.json` diffs are zero. If not, the
   retriever has unseeded randomness; fix with explicit seeds in
   the factory.
7. **CI dry-run.** Push a branch with a trivially regressing
   change to a feature branch and open a PR against a staging
   fork. The workflow runs, the PR comment appears, the merge
   button turns red. Confirm the comment is upserted (not appended)
   on a second push.
8. **Baseline freshness.** `jq .ran_at eval/baseline/metrics.json`
   is ≤ 7 days old (or whatever the staleness budget is). An older
   baseline usually means a stuck baseline-refresh workflow on
   `main`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Scorer uses standard library | `ir_measures` for metrics; no bespoke recall/nDCG implementations in the repo. |
| Qrels versioned | `qrels_sha` is captured in `metrics.json`; a change to qrels forces a baseline refresh. |
| Retriever under test is production | Retriever is imported via `module:factory`, not reimplemented for CI. |
| Latency in the gate | At least p95 is gated; tolerance is explicit. |
| Gate semantics documented | `BUDGET.md` names metrics, tolerances, override procedure, baseline-refresh procedure. |
| Override has evidence | Override requires an explicit reason captured in a `BUDGET_OVERRIDE_REASON` env or PR body. |
| CI runs on relevant PRs | `paths:` filter covers retriever, chunker, embeddings, index, eval. |
| PR comment is upserted | Rerunning the workflow replaces the comment, does not append. |
| Comment posts on failure | PR comment appears even when the gate fails. |
| Reproducibility | `PYTHONHASHSEED`, bootstrap seed, model revisions are pinned; two consecutive runs on the same code yield byte-equal metrics. |
| Baseline refresh on `main` | A separate workflow writes `eval/baseline/` on merges to `main`; baselines do not stale indefinitely. |
| Significance option | Paired bootstrap CI gate is offered as the stricter alternative for mature repos. |

Fail-fast smells:

- The gate's budget is "nDCG@10 can drop up to 10%." That is not a
  gate; that is a dashboard. Tighten to ≤ 3%.
- The gate is advisory only (comments, never fails). Everyone
  ignores it within a quarter.
- The qrels have 20 queries. Variance dominates every signal.
- The baseline is a static file that has not updated in months.
  Any genuine improvement fails the gate until the baseline is
  manually updated — the gate becomes a tax on good work.
- Latency is gated but quality is not, or vice versa. Both regress
  together; both must be in the gate.
- Overrides require no evidence. Every regression gets overridden
  "temporarily" and never reverted.

## 5. Common mistakes

- **Re-implementing nDCG.** Every team has an nDCG utility and
  every team's nDCG utility disagrees at the fourth decimal with
  everyone else's. Use `ir_measures` ([ir-measur.es](https://ir-measur.es/))
  or `pytrec_eval`; both are the standard. If you must write
  your own, check it against `ir_measures` on the same qrels
  before trusting it.
- **Gating on a single metric.** Recall only, or nDCG only, or MRR
  only. Changes that improve one metric and regress another slide
  through. Gate on all three plus latency.
- **Running against a retriever mock.** The CI retriever must be
  the same code path as production. A mock retriever that returns
  canned data turns the gate into a tautology.
- **Fetching the full corpus in CI.** The CI job must be minutes,
  not hours. A sampled shard sized to the qrels is sufficient;
  the full corpus is for exercise-03's nightly build + eval.
- **No baseline-refresh job.** The baseline must be refreshed on
  every merge to `main`, or a merged regression (approved via
  override) will keep tripping subsequent PRs.
- **Tight relative + no absolute tolerance.** `R@k` on small
  qrels has wide-enough variance that a 2% relative budget trips
  on noise. Pair with an absolute floor (1 pp).
- **Override with no reason.** `--override` without
  `BUDGET_OVERRIDE_REASON` must fail; otherwise overrides become
  the default path.
- **Posting one comment per push (not upsert).** The PR comment
  stream fills with 15 identical comments on a 15-push PR.
  Use a sticky-comment action keyed by a header string.
- **Letting qrels drift silently.** A qrels edit that is not
  paired with a baseline refresh breaks every subsequent gate.
  Pin `qrels_sha` into `metrics.json` and fail the gate if the
  baseline's `qrels_sha` differs from the PR's.

## 6. References

Metric libraries (primary tools):

- `ir_measures` — [ir-measur.es](https://ir-measur.es/).
  Standard-library IR metrics (recall, nDCG, MRR, MAP, ERR, etc.)
  with graded-qrels handling.
- `pytrec_eval` — [github.com/cvangysel/pytrec_eval](https://github.com/cvangysel/pytrec_eval).
  Thin Python binding to `trec_eval`; identical semantics.

CI primitives:

- GitHub Actions — [docs.github.com/actions](https://docs.github.com/en/actions).
  `workflow_dispatch`, `pull_request`, caching, sticky comments.
- `marocchino/sticky-pull-request-comment` — maintained action for
  upsert-style PR comments.
- `actions/upload-artifact@v4` — for preserving `metrics.json`,
  `per_query.json`, `latency.json`, and `report.md` for later
  inspection.

Governance framings:

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant functions: **MEASURE 2.3** (evaluation against
  documented metrics), **MANAGE 2.3** (continuous monitoring),
  **GOVERN 1.5** (independent review of overrides).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: **ML02 Data Poisoning Attack** (the gate catches
  quality drift that would otherwise be invisible until user
  complaints).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant techniques: *Evade ML Model*, *Poison Training Data* —
  the eval gate is the first line of defense that surfaces either
  as a measured regression.

Related exercises:

- mod-109 exercise-01 — how to build the qrels used here.
- mod-109 exercise-02 — the metrics this exercise scores with.
- mod-109 exercise-03 — paired bootstrap significance (§2.4).
- mod-111 exercise-01 — the cutover drill that consumes this
  gate's verdict (Phase 4 of the runbook).
