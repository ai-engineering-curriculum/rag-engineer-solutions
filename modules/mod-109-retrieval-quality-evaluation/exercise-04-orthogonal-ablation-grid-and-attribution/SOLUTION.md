# Exercise 04 — Solution: Orthogonal Ablation Grid and Attribution

This is a reference solution for
`exercise-04-orthogonal-ablation-grid-and-attribution.md`. It is a
mixed **design + implementation** exercise. The learner builds a
small factorial grid over the knobs in the retrieval pipeline
(embedder, BM25 presence, fusion, rerank, cutoff), runs every cell,
and attributes the final metric gain to each knob — not by "what
changed last" storytelling but by a factorial analysis that
separates **main effects** from **interactions**.

The teaching point: the common failure in retrieval evaluation is
"we turned on hybrid *and* rerank *and* query rewriting, nDCG@10
went up 8 points, ship it." That number is real, but it does not
tell you which knob did the work. An orthogonal ablation grid
answers that question. It also reveals *interactions* — knobs that
only work together, or knobs that cancel each other — which plain
A/B testing of each change in isolation cannot see.

## 1. Solution overview

The deliverable has six parts:

1. **`grid/factors.yaml`** — the grid definition. Named factors
   with 2–3 levels each. Full factorial when the cell count is
   tractable; a fractional factorial (Taguchi-style) when it is
   not.
2. **`src/run_grid.py`** — the orchestrator. For each cell in the
   grid, builds the retrieval pipeline from the factor levels,
   runs it on the exercise-01 qrels, scores with exercise-02's
   scorer, and writes a long-format CSV: one row per
   `(cell_id, query_id, metric)`.
3. **`src/attribution.py`** — the analysis. Fits a linear model
   on per-query metric values with one coefficient per factor
   level and one per pairwise interaction, reports effect sizes,
   confidence intervals, and a ranked attribution table.
4. **`reports/effects_table.md`** — the headline: each factor's
   main effect, the top interactions, and the "residual" (how
   much of the observed gain is not explained by the modelled
   effects).
5. **`reports/plots/`** — effects plot (sorted by |effect|),
   interaction plot (two-factor cells where the interaction is
   > its main effect), and a Pareto chart of effect contributions.
6. **`notes/attribution.md`** — the one-page memo: which knobs
   deserve the credit, which combinations to ship together,
   which to drop.

The exercise is scoped to *attribution*, not to adding new
pipeline features. The knobs in the grid are already implemented
in mod-104/105/106/107 solutions; this exercise answers "which
ones mattered" after the fact.

## 2. Worked answer or implementation

### 2.1 Factorial design in one paragraph

A **full factorial** grid over k binary factors has 2^k cells; k=5
is 32 cells, k=6 is 64, k=7 is 128. Each cell is a complete
retrieval run over the qrels. Full factorial lets you estimate
every main effect and every k-way interaction cleanly; the
estimates are **orthogonal** (uncorrelated) by construction, which
is why the analysis attributes credit without the usual regression
headaches. A **fractional factorial** halves (or quarters) the
cell count by deliberately aliasing high-order interactions with
lower-order ones — fine when you believe three-way interactions
are near-zero, which is almost always true for retrieval knobs.
Taguchi L8 and L16 arrays are the common fractional designs; the
scipy / statsmodels ecosystems do not generate them, so we either
enumerate the grid ourselves or use a dedicated package.

### 2.2 `grid/factors.yaml` — the grid

```yaml
# Factors and levels for the mod-109 ablation grid.
# Keep factor count ≤ 6 for a full factorial (2^6 = 64 cells).
# Add noise factors to the "noise" block; they are blocked, not crossed.

factors:
  embedder:
    - name: baseline-dense
      description: mod-103 baseline embedding model
    - name: tuned-dense
      description: mod-103 tuned embedding (if available) or alt model
  bm25:
    - name: off
      description: dense-only retrieval
    - name: on
      description: mod-105 BM25 branch enabled, fused via RRF (mod-106)
  fusion:
    - name: rrf
      description: reciprocal rank fusion (mod-106 default)
    - name: linear
      description: score normalisation + weighted linear combine (mod-106 alt)
  rerank:
    - name: off
      description: no reranker; use first-stage ranking
    - name: cross-encoder
      description: mod-107 cross-encoder rerank over top-N
  query_rewrite:
    - name: off
      description: user query passed through unchanged
    - name: multi-query
      description: mod-106 multi-query expansion, 3 variants
  topn_to_rerank:
    - name: '50'
      description: feed top-50 into the reranker
    - name: '200'
      description: feed top-200 into the reranker

response:
  - metric: nDCG@10
    primary: true
  - metric: Recall@50
  - metric: RR@10

noise:
  qrels_slice: full       # exercise-01 qrels v1.0.0
  seed: 0xA1B2
```

**Choices and why:**

- **Six factors ⇒ 64 cells.** On 100 queries with a 50 ms-per-
  query retrieval step, the whole grid runs in minutes. If a
  factor is expensive (a different embedder requires
  re-indexing the corpus), either budget for it or move it to
  a separate study.
- **Levels are concrete, not vibes.** `bm25: on/off`, not
  `bm25: 0.0 / 0.3 / 0.7`. Continuous knobs with three or more
  levels blow up the cell count fast. If a continuous knob
  matters, run a two-level factorial to find the direction of
  the effect, then a one-factor sweep for calibration.
- **Response variables declared up front.** The primary metric
  (nDCG@10) is the one the attribution ranks against; the
  others are tracked but do not drive the decision. This is
  exercise-03's "pre-register the metric" rule applied to a
  grid.
- **Interaction with `topn_to_rerank` is deliberate.**
  Reranker benefit scales with input-set size; crossing those
  two factors tests the interaction rather than guessing.

### 2.3 `src/run_grid.py` — the orchestrator

```python
"""Run every cell of the factorial grid; emit long-format per-query metrics.

Each cell is a dict of {factor: level}. Each cell produces a TREC
run file; exercise-02's scorer is called per cell."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .score_run import score_per_query, write_per_query_csv

@dataclass(frozen=True)
class Cell:
    cell_id: str                    # stable 8-char hash of the level dict
    levels: dict[str, str]

def _enumerate(factors: dict) -> list[Cell]:
    names = list(factors)
    level_lists = [[lv["name"] for lv in factors[n]] for n in names]
    cells = []
    for combo in itertools.product(*level_lists):
        levels = dict(zip(names, combo))
        key = json.dumps(levels, sort_keys=True).encode("utf-8")
        cid = hashlib.sha256(key).hexdigest()[:8]
        cells.append(Cell(cid, levels))
    return cells

def run_grid(
    factors_path: Path,
    qrels_path: Path,
    run_cell: Callable[[Cell], Path],    # returns path to TREC run file
    out_root: Path,
) -> list[dict]:
    with factors_path.open() as fh:
        spec = yaml.safe_load(fh)
    cells = _enumerate(spec["factors"])
    rows: list[dict] = []
    for cell in cells:
        run_path = run_cell(cell)
        per_query = score_per_query(qrels_path, run_path)
        out_dir = out_root / cell.cell_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_per_query_csv(per_query, out_dir / "per_query.csv")
        (out_dir / "levels.json").write_text(json.dumps(cell.levels))
        for r in per_query:
            for metric_name, value in r.items():
                if metric_name == "query_id":
                    continue
                rows.append({
                    "cell_id": cell.cell_id,
                    "query_id": r["query_id"],
                    "metric": metric_name,
                    "value": value,
                    **cell.levels,
                })
    return rows

def write_long_csv(rows: list[dict], out: Path) -> None:
    import csv
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    cols = ["cell_id", "query_id", "metric", "value"] + [
        k for k in rows[0] if k not in {"cell_id", "query_id", "metric", "value"}
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
```

**Choices and why:**

- **`cell_id` is a stable hash of the level dict.** Means that
  adding or removing a cell does not re-number the others and
  caching across re-runs is correct. A monotonic integer id
  becomes a nightmare when the grid is edited.
- **Long-format output (one row per `(cell, query, metric)`).**
  The attribution model wants tidy data. Wide CSVs (one column
  per metric) force a reshape inside the analysis code and
  invite bugs on the join.
- **Factor levels are columns in the long CSV.** Keeps the
  attribution model a one-liner
  (`value ~ C(bm25) + C(rerank) + ...`) without re-joining
  against the levels file.
- **`run_cell` is injected.** The orchestrator does not care
  how the pipeline is assembled; it only cares about
  deterministic per-cell runs. This keeps the ablation grid
  decoupled from whichever module owns the pipeline builder.

### 2.4 `src/attribution.py` — the analysis

```python
"""Attribute per-query metric values to factor main effects and pairwise
interactions. Uses OLS on the long-format CSV."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

@dataclass
class Effect:
    name: str                # e.g., "rerank[T.cross-encoder]"
    estimate: float
    ci95: tuple[float, float]
    p_value: float
    is_interaction: bool

def fit_effects(
    long_csv: Path, metric: str, factors: list[str]
) -> list[Effect]:
    df = pd.read_csv(long_csv)
    df = df[df["metric"] == metric].copy()
    if df.empty:
        raise ValueError(f"No rows for metric={metric!r} in {long_csv}")
    # Sum (deviation) coding so each main-effect coefficient is the
    # factor's *marginal* effect — averaged over the other factors'
    # levels — not the effect-at-reference that patsy's default
    # treatment coding produces. Treatment coding silently reports a
    # conditional slice whenever factors interact, which is exactly
    # the scenario an ablation grid is built to surface.
    terms = [f"C({f}, Sum)" for f in factors]
    for a, b in combinations(factors, 2):
        terms.append(f"C({a}, Sum):C({b}, Sum)")
    formula = f"value ~ {' + '.join(terms)}"
    model = smf.ols(formula=formula, data=df).fit()
    effects: list[Effect] = []
    for name, est in model.params.items():
        if name == "Intercept":
            continue
        ci_lo, ci_hi = model.conf_int().loc[name].tolist()
        p = float(model.pvalues[name])
        is_int = ":" in name
        effects.append(Effect(name, float(est), (ci_lo, ci_hi), p, is_int))
    effects.sort(key=lambda e: abs(e.estimate), reverse=True)
    return effects

def render_effects_table(effects: list[Effect]) -> str:
    out = [
        "| Effect | Estimate | 95% CI | p-value | Kind |",
        "|---|---|---|---|---|",
    ]
    for e in effects:
        kind = "interaction" if e.is_interaction else "main"
        out.append(
            f"| `{e.name}` | {e.estimate:+.4f} | "
            f"({e.ci95[0]:+.4f}, {e.ci95[1]:+.4f}) | "
            f"{e.p_value:.3g} | {kind} |"
        )
    return "\n".join(out) + "\n"

def pareto(effects: list[Effect]) -> list[tuple[str, float]]:
    total = sum(abs(e.estimate) for e in effects) or 1.0
    return [(e.name, abs(e.estimate) / total) for e in effects]
```

**Choices and why:**

- **OLS with Sum-coded categorical factors** is the shortest
  path to clean main-effect and interaction estimates in a
  balanced factorial. Sum coding places each level symmetric
  around the grand mean, so the main-effect coefficient of a
  factor is its *marginal* effect across the other factors'
  levels — the quantity you want to attribute. (Patsy's
  default `C(f)` is treatment coding, which reports the
  factor's effect only at every other factor's reference
  level; in the presence of interactions the two differ.)
  Classical factorial design further defines the *effect* as
  the between-level difference, which is 2× the Sum-coded
  coefficient on a two-level factor; pick one convention
  (coefficient or effect) and stay with it in the memo.
- **Pairwise interactions only.** Three-way interactions in a
  2^6 design are low-power and rarely interpretable. If the
  memo needs them, re-fit with the extra terms; otherwise the
  model's `residual` captures them.
- **CIs come from the fit.** OLS CIs assume homoscedastic
  residuals; for sanity, re-run with clustered standard
  errors by `query_id` (`.get_robustcov_results('cluster',
  groups=df['query_id'])`) — queries are the repeated unit
  and their metric values are correlated within query.
- **Sort by |estimate|.** The reader wants the biggest knobs
  first. Significance is a filter, not the sort key.

### 2.5 Example output — `reports/effects_table.md`

Running the attribution on an illustrative 64-cell run of the
grid against nDCG@10:

```markdown
| Effect | Estimate | 95% CI | p-value | Kind |
|---|---|---|---|---|
| `C(rerank, Sum)[S.cross-encoder]` | +0.0356 | (+0.0306, +0.0407) | 1.3e-28 | main |
| `C(bm25, Sum)[S.on]` | +0.0215 | (+0.0164, +0.0265) | 2.1e-13 | main |
| `C(query_rewrite, Sum)[S.multi-query]` | +0.0091 | (+0.0041, +0.0142) | 4.8e-04 | main |
| `C(embedder, Sum)[S.tuned-dense]` | +0.0086 | (+0.0035, +0.0136) | 9.3e-04 | main |
| `C(topn_to_rerank, Sum)[S.200]` | +0.0047 | (-0.0004, +0.0098) | 0.067 | main |
| `C(bm25, Sum)[S.on]:C(rerank, Sum)[S.cross-encoder]` | +0.0047 | (+0.0016, +0.0079) | 3.5e-03 | interaction |
| `C(query_rewrite, Sum)[S.multi-query]:C(rerank, Sum)[S.cross-encoder]` | -0.0041 | (-0.0072, -0.0009) | 0.012 | interaction |
| `C(fusion, Sum)[S.linear]` | -0.0011 | (-0.0061, +0.0040) | 0.68 | main |
```

**Reading (effects, i.e. between-level deltas = 2× the Sum
coefficient for a two-level main, 4× for a 2×2 interaction):**

- **Rerank is doing most of the work** (effect +0.071 nDCG@10).
  This is the single biggest knob in the grid.
- **BM25 matters** (effect +0.043) and **interacts positively**
  with rerank (interaction effect +0.019): adding BM25 to the
  candidate set gives the reranker more variety to work with.
  The combined gain of BM25-on + rerank-on is roughly +0.043 +
  +0.071 + +0.019 = +0.133, not +0.114, because of the
  interaction.
- **Query rewrite helps alone** (effect +0.018) but **interacts
  negatively with rerank** (interaction effect -0.016): the
  extra queries hand the reranker more noise, and the reranker
  already eats most of the rewriter's lunch. If you ship
  rerank, shipping query rewrite is roughly neutral.
- **Fusion method does not matter** (p=0.68). RRF and linear
  fusion are interchangeable at this grid size; the memo can
  default to RRF for simplicity.
- **Top-N 50 vs 200 into the reranker is a wash** at 50 —
  the CI crosses zero. Pick 50 for latency.

### 2.6 `notes/attribution.md` — the one-page memo

Which knobs deserve credit, which combinations to ship:

| Decision | Rationale from the grid |
|---|---|
| Ship cross-encoder rerank | Main effect +0.071 on nDCG@10, p ≈ 1e-28. Biggest knob in the grid. |
| Ship BM25 branch + RRF fusion | Main effect +0.043 alone; positive interaction with rerank adds +0.019. Combined with rerank: +0.133. |
| Do not ship query rewrite when rerank is on | Positive main effect (+0.018) but negative interaction with rerank (-0.016). Net ≈ 0. |
| Pick top-N = 50 to rerank | Top-N 200 gives +0.009 with CI crossing zero. The 4× cost is not justified. |
| Pick RRF over linear fusion | Fusion method is indistinguishable at the grid size. RRF has one knob instead of three. |

**Headline for the release note:** *"Hybrid + rerank is the
baseline; each knob was measured against the others, and query
rewriting was dropped because the grid showed it conflicts with
the reranker it replaced."*

**Caveats to name in the memo:**

- Three-way interactions were not modelled; if the top pairwise
  interaction is big, re-fit with three-way terms before
  calling the decision final.
- The grid was run on one qrels version; a corpus-fit drift
  (exercise-05) could change the ranking.
- OLS CIs assume per-query independence; with cluster-robust
  SEs the CIs are a few percent wider but the ranking does not
  change.

## 3. Validation steps

1. The full factorial has `∏ |levels|` cells, every cell ran
   successfully, and the long CSV has `n_cells × n_queries ×
   n_metrics` rows.
2. OLS fit succeeds with all main effects and all pairwise
   interactions in the formula. `model.df_resid` is positive
   (grid not under-identified).
3. Coefficients are orthogonal in the sense expected for a
   balanced factorial: swapping a factor's reference level
   changes signs but not magnitudes of main effects.
4. Cluster-robust standard errors (`groups=query_id`) give
   CIs within ~20% of the naive CIs; if they are much wider,
   the memo must report the robust CIs.
5. The sum of absolute main effects + absolute interactions
   explains ≥ 80% of the between-cell variance in the metric
   (R² of the model ≥ 0.8). Lower R² ⇒ model is missing a
   factor or the metric is dominated by per-query noise.
6. Effects whose CIs cross zero are called out as *not
   significant*, not quietly dropped.
7. The memo names a decision for every factor, not only the
   winners.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Grid | Named factors with discrete levels; cell count ≤ a tractable number (≤ 128 is the working ceiling). |
| Orthogonality | Full factorial OR a documented fractional design (Taguchi L8/L16). No haphazard "we tried these combinations" selections. |
| Per-query data | Long-format CSV: one row per `(cell, query, metric)`. |
| Model | OLS with main effects + pairwise interactions; cluster-robust SEs by `query_id`. |
| Effect table | Sorted by `abs(estimate)`; names main vs interaction; reports CI and p. |
| Attribution memo | A decision for every factor, with the interaction effects explicitly called out. |
| Reproducibility | Same grid + qrels ⇒ same effects table (seed fixed in `noise.seed`). |
| Residuals | R² ≥ 0.8 OR a note explaining why the model misses the metric's variance. |

Fail-fast smells:

- The grid is "the three combinations we remembered to try." No
  attribution is possible without balance; the result is a
  table that looks factorial but is not.
- Interactions are omitted from the model. A "query rewrite
  helps" conclusion collapses as soon as the interaction with
  rerank is modelled.
- Effects and p-values are reported, CIs are not. Exercise-03's
  lesson applies here.
- The grid was rerun with floating seeds for retrieval
  components. Cell-to-cell variance now mixes method
  variance with seed variance; the OLS attribution is wrong.
- The memo concludes "ship everything that was significant."
  Significance is not the ship criterion; the net effect
  (main + interactions) is.

## 5. Common mistakes

- **One-knob-at-a-time rather than a grid.** Looks cheaper; is
  not. Four binary knobs at a time is four runs; the full 2^4
  grid is 16 runs but *answers the attribution question*.
  Four one-at-a-time runs cannot see interactions.
- **Interactions ignored.** In retrieval, hybrid × rerank is
  almost always a non-trivial interaction. A main-effects-only
  model will under- or over-estimate both knobs.
- **Confusing effect with coefficient.** Classical factorial
  design defines the *effect* as the between-level difference;
  that equals 2× a Sum-coded (`C(f, Sum)`, ±1) main coefficient
  and 1× a treatment-coded (`C(f)`, 0/1) coefficient for a
  two-level factor. This solution uses Sum coding so the main
  coefficients are marginal (averaged over other factors) and
  comparable to each other; report *effects* in the memo so
  the numbers live on the response scale.
- **Shipping on main effects alone when the interaction is
  negative.** Query-rewrite main effect +0.018, interaction
  with rerank -0.016, net +0.002. Shipping on the main
  effect delivers no user-visible benefit.
- **Running the grid at a single cutoff.** Rerank gains at top-
  10 but is flat at top-50. The decision is about user-
  visible top-10 — but the memo should report both.
- **No cluster-robust SEs.** Per-query metric values within a
  cell are correlated; naive CIs are too narrow. The
  one-line statsmodels fix is cheap.
- **Fractional factorial without declaring the aliasing
  structure.** A 2^(6-1) design aliases a three-way with a
  main; an attribution memo that doesn't name the aliasing is
  concealing a confound.
- **Adding a factor late and re-running only the new cells.**
  Breaks the orthogonality the OLS relies on. Full re-run or
  exclude the factor.

## 6. References

- Factorial design:
  - Box, G. E. P., Hunter, J. S., Hunter, W. G. (2005),
    *Statistics for Experimenters: Design, Innovation, and
    Discovery*, Wiley — the standard reference for full and
    fractional factorial design.
  - Taguchi, G. (1986), *Introduction to Quality Engineering*,
    Asian Productivity Organization — orthogonal arrays for
    fractional designs.
  - NIST/SEMATECH *e-Handbook of Statistical Methods*,
    section 5.3 — "Choosing a Design" —
    [itl.nist.gov/div898/handbook/pri/section3/pri3.htm](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm).
- Attribution in retrieval:
  - Bendersky, M. et al. (2010), "Learning to effectively
    select a weighted rank fusion" — reports factorial-style
    attribution for fusion components.
- Tooling:
  - `statsmodels.formula.api.ols` — categorical and
    interaction formulas via patsy —
    [statsmodels.org/stable/example_formulas.html](https://www.statsmodels.org/stable/example_formulas.html).
  - `cluster` robust covariance in statsmodels — `get_robustcov_results`.
- Upstream:
  - exercise-02 scorer and per-query CSV —
    `../exercise-02-compute-recall-mrr-ndcg-with-ir-measures/SOLUTION.md`.
  - exercise-03 paired tests complement the attribution with
    per-cell head-to-head significance —
    `../exercise-03-paired-significance-and-bootstrap-ci-on-per-query-deltas/SOLUTION.md`.
- Risk and governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML06 Output Integrity — attributing a gain
    to the wrong knob ships a false provenance claim.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around *Evaluation* motivate running the full
    grid rather than cherry-picking combinations.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Govern* and *Measure* functions require that changes
    be attributable to a specific input; the attribution
    memo is that artifact for the retrieval pipeline.
