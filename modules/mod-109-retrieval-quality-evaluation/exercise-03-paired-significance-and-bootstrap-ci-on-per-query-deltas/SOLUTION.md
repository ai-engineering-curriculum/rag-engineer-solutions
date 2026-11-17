# Exercise 03 — Solution: Paired Significance and Bootstrap CI on Per-Query Deltas

This is a reference solution for
`exercise-03-paired-significance-and-bootstrap-ci-on-per-query-deltas.md`.
It is an **implementation + analysis** exercise. The learner takes
the per-query metrics CSV from exercise-02, runs paired statistical
tests on the per-query *delta* between two systems, builds a
bootstrap confidence interval, and writes a short memo that
interprets the result for a release decision.

The teaching point: a mean nDCG@10 improvement from 0.712 to 0.721
is *not* evidence that the new system is better. It might be, but
the variance across queries is enormous — a handful of lucky
queries can move the mean by that much. Paired tests and
confidence intervals exist so you can say "the 95% CI on the delta
is [+0.002, +0.016] and the paired Wilcoxon p-value is 0.003" —
which is a decision, not a headline.

## 1. Solution overview

The deliverable has five parts:

1. **`src/paired_stats.py`** — the stats. Loads two per-query
   CSVs, aligns on `query_id`, computes paired t-test, Wilcoxon
   signed-rank, and permutation p-values on the delta; builds a
   bias-corrected-accelerated (BCa) bootstrap CI around the mean
   delta.
2. **`src/report.py`** — turns the stats dict into a one-page
   markdown report with the headline sentence written for you.
3. **`notes/decision.md`** — the memo template: how to read the
   three p-values together, what the CI means, and when "mean
   went up" is not sufficient to ship.
4. **`notes/multiple_comparisons.md`** — a short addendum on how
   to adjust p-values when you are comparing N systems to one
   baseline or N metrics on one system (Bonferroni; Benjamini-
   Hochberg FDR).
5. **`reports/example_bm25_vs_hybrid.md`** — a worked example
   showing the full pipeline on a realistic per-query CSV.

The exercise is scoped to *comparing two systems on one metric*.
Multiple-system or multiple-metric comparisons are handled by the
multiple-comparisons addendum and by exercise-04's ablation grid.

## 2. Worked answer or implementation

### 2.1 Paired testing in one paragraph

Two retrieval systems scored on the same 100 queries are **paired**:
each query produces one metric per system, so the natural unit of
analysis is the per-query *delta* `δ_i = metric(sys_b, q_i) −
metric(sys_a, q_i)`. Paired tests ask "is the mean of {δ_i}
different from zero?" and exploit the pairing to cancel
query-level noise that unpaired tests have to absorb. Three
complementary tests: the **paired t-test** (parametric, sensitive
to outliers, assumes approximately-normal δs), **Wilcoxon
signed-rank** (non-parametric, robust, uses only signs and ranks
of δs), and the **permutation test** (distribution-free, swaps
system labels within pair and recomputes mean delta under the
null). On ≥ 50 queries they usually agree; when they disagree the
permutation is the least-wrong and the t is the most-wrong.

### 2.2 `src/paired_stats.py` — the stats

```python
"""Paired significance tests and bootstrap CI on per-query metric deltas.

Inputs: two per-query CSVs with columns `query_id` + one column per metric,
as emitted by exercise-02's score_run.py.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

RNG_SEED = 0x9EE3

@dataclass
class PairedResult:
    metric: str
    system_a: str
    system_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    ci95: tuple[float, float]        # BCa 95% CI on mean_delta
    p_ttest: float                   # paired two-sided t-test
    p_wilcoxon: float                # Wilcoxon signed-rank, two-sided
    p_permutation: float             # permutation two-sided, 10k iters
    n_a_wins: int                    # queries where δ > 0
    n_b_wins: int                    # queries where δ < 0
    n_ties: int                      # queries where δ == 0 exactly

def _load(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            qid = r["query_id"]
            rows[qid] = {
                k: float(v) for k, v in r.items() if k != "query_id"
            }
    return rows

def _align(a: dict, b: dict, metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    qids = sorted(set(a) & set(b))
    av = np.array([a[q][metric] for q in qids], dtype=float)
    bv = np.array([b[q][metric] for q in qids], dtype=float)
    return av, bv, qids

def _permutation_pvalue(delta: np.ndarray, n_iter: int = 10_000) -> float:
    """Two-sided: under H0 the sign of each delta is 50/50."""
    rng = np.random.default_rng(RNG_SEED)
    obs = float(np.mean(delta))
    signs = rng.choice([-1.0, 1.0], size=(n_iter, len(delta)))
    sampled = (signs * delta).mean(axis=1)
    # two-sided: fraction of |sampled| >= |obs|
    extreme = np.mean(np.abs(sampled) >= abs(obs))
    # standard +1 small-sample correction
    return float((extreme * n_iter + 1) / (n_iter + 1))

def _bca_ci(x: np.ndarray, alpha: float = 0.05, n_iter: int = 10_000) -> tuple[float, float]:
    """Bias-corrected-accelerated bootstrap CI on the mean of x.
    Reference: Efron (1987) 'Better bootstrap confidence intervals'."""
    rng = np.random.default_rng(RNG_SEED + 1)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    theta = np.mean(x)
    # resample means
    idx = rng.integers(0, n, size=(n_iter, n))
    boot = x[idx].mean(axis=1)
    # bias correction z0
    p0 = np.mean(boot < theta)
    p0 = min(max(p0, 1e-9), 1 - 1e-9)   # avoid ±inf at the extremes
    z0 = stats.norm.ppf(p0)
    # acceleration via jackknife
    jk = np.array([
        np.mean(np.delete(x, i)) for i in range(n)
    ])
    jk_mean = jk.mean()
    num = np.sum((jk_mean - jk) ** 3)
    den = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    a_hat = 0.0 if den == 0 else num / den
    # adjusted percentiles
    za = stats.norm.ppf(alpha / 2)
    zb = stats.norm.ppf(1 - alpha / 2)
    lo = stats.norm.cdf(z0 + (z0 + za) / (1 - a_hat * (z0 + za)))
    hi = stats.norm.cdf(z0 + (z0 + zb) / (1 - a_hat * (z0 + zb)))
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))

def paired_compare(
    csv_a: Path, csv_b: Path, metric: str,
    system_a: str, system_b: str
) -> PairedResult:
    a = _load(csv_a)
    b = _load(csv_b)
    av, bv, qids = _align(a, b, metric)
    if len(qids) < 10:
        raise ValueError(
            f"Only {len(qids)} aligned queries for metric={metric!r}. "
            "Paired tests below this size are not informative; use ≥ 30."
        )
    delta = bv - av
    # paired t-test
    t_stat, p_t = stats.ttest_rel(bv, av)
    # Wilcoxon (two-sided); skip zero deltas per convention
    nonzero = delta[delta != 0.0]
    if len(nonzero) == 0:
        p_w = 1.0
    else:
        _, p_w = stats.wilcoxon(nonzero, zero_method="wilcox")
    p_p = _permutation_pvalue(delta)
    ci_lo, ci_hi = _bca_ci(delta)
    return PairedResult(
        metric=metric,
        system_a=system_a, system_b=system_b,
        n=len(qids),
        mean_a=float(av.mean()), mean_b=float(bv.mean()),
        mean_delta=float(delta.mean()),
        ci95=(ci_lo, ci_hi),
        p_ttest=float(p_t), p_wilcoxon=float(p_w),
        p_permutation=float(p_p),
        n_a_wins=int((delta < 0).sum()),
        n_b_wins=int((delta > 0).sum()),
        n_ties=int((delta == 0).sum()),
    )
```

**Choices and why:**

- **Three p-values, not one.** The t-test is familiar but
  sensitive to outliers; Wilcoxon is robust but loses magnitude
  information; permutation is the least-assumption test and the
  right fallback when the first two disagree. Reporting all
  three forces you to look at the data when they conflict
  instead of cherry-picking the smallest p.
- **BCa bootstrap for the CI.** The simple percentile bootstrap
  is biased when the sampling distribution is skewed (common on
  bounded metrics like recall/nDCG near 1.0). BCa corrects for
  bias and skewness. Efron's original paper (1987) is the
  reference.
- **10,000 iterations, fixed seed.** Enough to stabilise the
  tails of a 95% CI; the fixed seed is what makes the test
  reproducible across re-runs.
- **n ≥ 10 hard floor with a nudge to n ≥ 30.** Below 10 queries
  no paired test is informative; between 10 and 30 the t-test
  starts leaning on its normality assumption. The error message
  tells the learner the standard working floor.
- **Count query-level wins and ties separately.** A 60/40 win
  record with a tiny mean delta is a different story from a
  90/10 win record with the same mean — the first says "small
  but reliable," the second says "huge on a minority of
  queries, flat on the rest."

### 2.3 `src/report.py` — the one-page report

```python
from .paired_stats import PairedResult

HEADLINE_FMT = (
    "{system_b} vs {system_a} on {metric}: mean delta "
    "{mean_delta:+.4f} (95% CI {ci_lo:+.4f} to {ci_hi:+.4f}); "
    "paired t p={p_ttest:.4g}, Wilcoxon p={p_wilcoxon:.4g}, "
    "permutation p={p_permutation:.4g}; "
    "{n_b_wins}/{n} queries favour {system_b}, "
    "{n_a_wins} favour {system_a}, {n_ties} tied."
)

def render(r: PairedResult) -> str:
    return HEADLINE_FMT.format(
        system_a=r.system_a, system_b=r.system_b, metric=r.metric,
        mean_delta=r.mean_delta, ci_lo=r.ci95[0], ci_hi=r.ci95[1],
        p_ttest=r.p_ttest, p_wilcoxon=r.p_wilcoxon,
        p_permutation=r.p_permutation,
        n=r.n, n_a_wins=r.n_a_wins, n_b_wins=r.n_b_wins,
        n_ties=r.n_ties,
    )
```

The rendered sentence is the headline you paste into a PR
description or the release-review slide. It names the metric,
the sign of the effect, the CI (not just the point estimate),
all three p-values, and the win/tie counts — every piece an
engineer needs to decide "ship or re-run."

### 2.4 Worked example — BM25 → hybrid on nDCG@10

Using exercise-02's per-query CSVs on nDCG@10 (100 queries):

```
hybrid vs bm25 on nDCG@10: mean delta +0.1144
    (95% CI +0.0797 to +0.1483);
    paired t p=5.7e-10, Wilcoxon p=2.3e-11,
    permutation p=1.0e-04;
    76/100 queries favour hybrid, 19 favour bm25, 5 tied.
```

Read: large effect (mean +0.11 on a 0–1 scale), CI excludes 0 by
a wide margin, all three p-values deeply below any reasonable
threshold, and a clear per-query win rate (76/100). **Decision:
ship hybrid.**

Compare to a less convincing imagined result (same shape, smaller
effect):

```
rerank_v2 vs rerank_v1 on nDCG@10: mean delta +0.0038
    (95% CI -0.0021 to +0.0092);
    paired t p=0.18, Wilcoxon p=0.09,
    permutation p=0.17;
    52/100 queries favour rerank_v2, 44 favour rerank_v1, 4 tied.
```

Read: tiny effect, CI crosses zero, all three p-values above 0.05,
near-50/50 per-query. **Decision: do not ship on the strength of
mean nDCG@10 alone.** The honest write-up is "no significant
difference at n=100; either grow the eval set or find a secondary
signal."

### 2.5 `notes/decision.md` — how to read the three p-values

The four recognisable cases:

1. **All three p's below 0.05, CI excludes 0, ≥ 60/100 query-
   level wins.** Clean ship. Headline the CI, mention the
   p-values, move on.
2. **All three p's above 0.05, CI straddles 0.** Clean "no
   evidence of difference." Do not spin a 0.07 p-value as a
   trend; re-run on a bigger eval set or on a more sensitive
   metric (per-query-delta on a sharper cutoff, or a metric
   the product surface actually uses).
3. **t-test significant, Wilcoxon and permutation are not.** A
   handful of queries have enormous deltas that the t-test's
   normal approximation amplifies. Inspect the top-10 |δ|
   queries; usually one of them is a label bug or a query that
   fell into a failure mode specific to one system. Fix or
   exclude, re-run.
4. **Wilcoxon and permutation significant, t is not.** The
   effect is real but long-tailed: many small positive deltas,
   a few large negatives that drag the mean. The rank-based
   tests see the direction; the mean does not. This is a case
   for *reporting the median delta alongside the mean* and
   making a judgment call.

The ship rule: CI excluding zero *and* at least two of three p's
< 0.05 *and* per-query win rate ≥ 55%. Point estimate alone is
never enough.

### 2.6 `notes/multiple_comparisons.md` — correcting for N tests

When you compare N systems to one baseline or report one system
across N metrics, you are running N tests and p-hacking becomes
a real risk. Two corrections:

- **Bonferroni.** Multiply every p-value by N (or equivalently,
  compare each to α/N). Conservative; use when you can only
  tolerate one false positive across the whole table.
- **Benjamini-Hochberg FDR.** Order p-values ascending; find
  the largest rank k such that `p_(k) ≤ (k/N)·α`; declare
  tests 1..k significant. Less conservative; use when you
  accept a controlled false-discovery rate (e.g., 5% of the
  "significant" results are expected to be spurious).

Default: BH FDR at α=0.05 when the comparison table has > 3
rows. Report both raw and adjusted p-values so the reader can
see the correction's impact.

## 3. Validation steps

1. `paired_compare` loads two per-query CSVs, aligns by
   `query_id`, and reports the number of aligned queries (which
   must be ≥ 10, with 30+ recommended).
2. The t-test and Wilcoxon are computed via SciPy
   (`stats.ttest_rel`, `stats.wilcoxon`), not hand-rolled.
3. The permutation test is two-sided, uses ≥ 10,000 iterations,
   and applies the standard `(k+1)/(n+1)` small-sample
   correction.
4. The bootstrap CI is BCa (not naive percentile) with a fixed
   seed and ≥ 10,000 resamples.
5. Re-running on the same inputs produces identical p-values and
   CIs (deterministic given the seed).
6. Report sentence names the CI, not just the mean; names all
   three p-values; and names win / tie counts.
7. For a system that is known-equivalent (e.g., scoring a run
   against itself), the pipeline returns `mean_delta ≈ 0`, a CI
   containing 0, and p-values near 1.0.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Paired alignment | Comparison is per-query-paired on `query_id`, not on indexes or on aggregate numbers. |
| Multiple tests | Paired t, Wilcoxon signed-rank, *and* permutation reported — not just one. |
| Confidence interval | BCa bootstrap with fixed seed and ≥ 10k resamples. Naive percentile is explicitly rejected in the memo. |
| Win/tie counts | Query-level win and tie counts reported. |
| Minimum n | Floor at 10 queries; 30+ recommended; the harness enforces the floor. |
| Reproducibility | Fixed RNG seed; two runs on the same inputs are bit-identical. |
| Multiple comparisons | Addendum names Bonferroni and BH FDR with a default choice. |
| Memo | `notes/decision.md` names the ship rule and the four p-value cases. |

Fail-fast smells:

- "Significant at p < 0.05" with no CI. The p-value is one
  coordinate; the CI is the other.
- Unpaired t-test on paired data. Loses the query-level cancel-
  out and inflates the p-value — a test that will not reject
  real improvements.
- Hand-rolled bootstrap sampling with a fresh seed each run.
  The number moves between re-runs and the memo loses
  credibility.
- One p-value cherry-picked from several. If you ran t and
  Wilcoxon, report both. If one is below 0.05 and one is not,
  the memo has to name that.
- Reporting mean with no median. Long-tailed metrics need both.
- Running tests across different n (metric-A had 100 queries,
  metric-B had 97). Align first; if the gap is large enough
  that the CIs cross, the alignment bug is the finding.

## 5. Common mistakes

- **Treating "p < 0.05" as the ship criterion.** A p-value of
  0.04 with a CI of [-0.001, +0.012] is not a shipping result;
  the lower bound crosses zero. The CI is the thing to argue
  about.
- **Picking the smallest p-value.** Running a dozen metric ×
  cutoff combinations and reporting the one that comes in
  below 0.05 is p-hacking. Pre-register the metric (exercise-02
  memo).
- **Unpaired test on paired data.** Statistically valid but
  underpowered. Use `ttest_rel`, not `ttest_ind`.
- **Naive percentile bootstrap.** Biased on bounded metrics
  (close to 0 or 1). BCa corrects this; cost is five lines of
  code.
- **Too small n.** Below ~30 queries the paired t-test's
  normality assumption is shaky and every test is low-power.
  Grow the eval set or grow the metric sensitivity before
  concluding.
- **Reporting the mean delta without the sign of a per-query
  win rate.** A mean of +0.01 with 85/100 wins is a different
  story from a mean of +0.01 with 52/100 wins — the latter is
  one or two queries doing all the work.
- **Running N system comparisons and not correcting for
  multiple testing.** 20 comparisons at α=0.05 expect one
  false positive by chance. Apply BH FDR before celebrating.
- **Non-deterministic permutation / bootstrap.** Fix the seed.

## 6. References

- Paired testing in IR:
  - Smucker, M. et al. (2007), "A comparison of statistical
    significance tests for information retrieval evaluation,"
    *CIKM 2007*. The standard recommendation for pairing
    permutation, paired t, and Wilcoxon on IR deltas.
  - Sakai, T. (2006), "Evaluating evaluation metrics based on
    the bootstrap," *SIGIR 2006*. Bootstrap CIs on IR metrics.
- Statistical tests:
  - Wilcoxon, F. (1945), "Individual Comparisons by Ranking
    Methods," *Biometrics Bulletin*.
  - Student (1908), "The Probable Error of a Mean," *Biometrika*
    — the paired t-test.
  - Fisher, R. A. (1935), *The Design of Experiments*
    (permutation test).
- Bootstrap:
  - Efron, B. (1987), "Better Bootstrap Confidence Intervals,"
    *JASA* — the BCa interval.
  - Efron, B. & Tibshirani, R. (1993), *An Introduction to the
    Bootstrap*, Chapman & Hall.
- Multiple comparisons:
  - Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False
    Discovery Rate," *JRSS-B*.
- Tooling:
  - SciPy `scipy.stats.ttest_rel`, `scipy.stats.wilcoxon`,
    `scipy.stats.permutation_test` —
    [docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html).
- Upstream:
  - exercise-02 per-query CSV —
    `../exercise-02-compute-recall-mrr-ndcg-with-ir-measures/SOLUTION.md`.
- Risk and governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML06 Output Integrity — shipping a "better"
    system on an uncorrected p-value is an integrity failure
    on the release-review surface.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around *Evaluation* motivate pre-registered
    metrics and documented significance thresholds.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure* function explicitly calls for uncertainty
    quantification; CI + p-values are the artifact.
