# Exercise 06 — Solution: Corpus / Query / Quality Drift Monitor with Alert Thresholds

This is a reference solution for
`exercise-06-corpus-query-quality-drift-monitor-with-alert-thresholds.md`.
It is an **implementation** exercise. The learner designs and
implements three complementary drift monitors — corpus drift
(what is indexed), query drift (what is asked), and quality drift
(how retrieval is scoring) — with explicit alert thresholds,
remediation runbook entries, and traceability back to the eval
gate (exercise-02) and the cutover drill (exercise-01).

The teaching point: *the retriever can regress for three unrelated
reasons* — the corpus moved, the query population moved, or the
retriever itself stopped working well on an unchanged workload. Each
has a different signal, a different threshold, and a different
remediation. A single "retrieval looks worse" dashboard conflates
them; a drift monitor is useful only if its three outputs are
distinguishable in the alert.

## 1. Solution overview

The deliverable has five parts:

1. **`drift/corpus_monitor.py`** — monitors what is indexed:
   doc counts, size, source mix, language mix, chunk-length
   distribution, embedding-norm distribution. Scheduled nightly
   via exercise-03's DAG.
2. **`drift/query_monitor.py`** — monitors what the retriever is
   asked: query length, language, intent/topic distribution (via
   a small classifier or embedding clusters), rare-term ratio,
   hit-rate at the retrieval layer, and empty-result ratio.
3. **`drift/quality_monitor.py`** — monitors how the retriever is
   scoring on a held-in labeled stream: periodic rerun of the
   exercise-02 eval, click/thumbs-based implicit feedback,
   calibration of score → relevance.
4. **`drift/alerts.md`** — the alert thresholds and routing: PSI /
   JS / χ² cutoffs, min-sample windows, severity, named owner,
   suggested remediation.
5. **`drift/runbook.md`** — one-page runbook per alert type: what
   to look at, what metrics to pull, what action resolves it
   (reindex, requalify qrels, retrain, cutover, do nothing).

Scope: *detection + alerting + routing to a remediation path*.
Remediation itself happens via exercise-01 (cutover), exercise-02
(eval gate), exercise-03 (rebuild DAG trigger), or qrels-refresh
in mod-109 exercise-01.

## 2. Worked answer or implementation

### 2.1 The three drifts, and why they are not the same

**Corpus drift** — the indexed content changed, intentionally or
not. Causes: new sources, data pipeline dedup bug, format
converter change, document deletions due to an ACL policy change,
language distribution shifted because a new region came online.
Signal: shifts in the corpus-side distributions (doc count,
source mix, chunk-length distribution, embedding-norm
distribution). Remediation: usually "rebuild" — but sometimes
"restore from backup; the drift is a bug."

**Query drift** — the user population or product surface changed.
Causes: new feature routes new question shapes to retrieval; a
marketing campaign drove new user demographics; seasonal shift
(Q4 returns questions in e-commerce); a bot or integration
started hammering a specific query pattern. Signal: shifts in
query-side distributions (length, intent, language, rare-term
rate, empty-result rate). Remediation: typically *measurement*
(update qrels so the eval reflects the new shape), not rebuild.

**Quality drift** — the retriever is scoring worse on a stable
input distribution. Causes: index became stale without a
scheduled rebuild; embedding provider silently updated a model
("ada" → "ada-002" was the classic case); reranker training
drifted; score calibration broke; a dependency version bump
changed tokenization. Signal: eval metrics regress on the held-
in qrels; implicit-feedback metrics regress; latency rises
without a traffic change. Remediation: open a cutover-drill
ticket; if the index is the suspect, rebuild (exercise-03).

The three monitors must be *separable* in the alert payload
("CORPUS_DRIFT fired" vs "QUERY_DRIFT fired" vs "QUALITY_DRIFT
fired"), because the three remediation paths are different and
mixing them wastes on-call time.

### 2.2 Drift measures (the math, in five lines each)

**Population Stability Index (PSI).**

```
PSI(p, q) = sum over bins i of (q_i - p_i) * ln(q_i / p_i)
```

where `p` is the baseline distribution over bins, `q` is the
current. Rule-of-thumb thresholds commonly cited (originate in
credit-scoring practitioner literature; see
[Ronen Canetti, "Population Stability Index"](https://en.wikipedia.org/wiki/Population_stability_index)
for a concise definition):

- PSI < 0.10 — insignificant change, no action.
- 0.10 ≤ PSI < 0.25 — moderate shift, monitor; investigate if
  persistent.
- PSI ≥ 0.25 — significant shift, action required.

PSI is undefined on zero-count bins; add Laplace smoothing
(`+ε` to every bin) or clip the summation terms. Needs ≥ 500
samples per window for a stable value — below that, the index
jitters.

**Jensen–Shannon Divergence (JS).**

```
JS(p, q) = 1/2 * KL(p || m) + 1/2 * KL(q || m),  where m = (p + q) / 2
```

Symmetric, bounded in `[0, ln(2)]` (or `[0, 1]` using log base
2). Preferred over KL for drift because it is finite even when
`p_i = 0`.

- JS < 0.05 — low drift.
- 0.05 ≤ JS < 0.10 — moderate.
- JS ≥ 0.10 — high.

**Kolmogorov–Smirnov (two-sample).**

```
D = sup_x |F_baseline(x) - F_current(x)|
```

For continuous distributions (chunk length, query length,
embedding norm). The test statistic + a p-value from a reference
distribution. Alert on `p < 0.01` *and* `D > 0.05` — significance
without a practical effect size is noise.

**χ² test (for categorical).**

For low-cardinality categoricals (source, language, intent
label). Standard Pearson χ² on the contingency table. Alert on
`p < 0.01` *and* at least one cell with relative shift > 20%.

The implementations use `scipy.stats` for KS and χ²; PSI and JS
are inlined (five lines each).

### 2.3 `drift/corpus_monitor.py` — corpus-side drift

```python
"""Corpus drift monitor. Runs nightly after the exercise-03
build DAG; compares the just-built index's distributions against
a rolling 14-day baseline window."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


# ---- Divergence helpers ----------------------------------------------------

def _laplace(counts: dict[str, int], bins: list[str], eps: float = 0.5) -> dict[str, float]:
    total = sum(counts.get(b, 0) for b in bins) + eps * len(bins)
    return {b: (counts.get(b, 0) + eps) / total for b in bins}


def psi(p: dict[str, float], q: dict[str, float]) -> float:
    return sum((q[b] - p[b]) * math.log(q[b] / p[b]) for b in p)


def js(p: dict[str, float], q: dict[str, float]) -> float:
    m = {b: 0.5 * (p[b] + q[b]) for b in p}
    def kl(a, b):
        return sum(a[k] * math.log(a[k] / b[k]) for k in a if a[k] > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---- Features extracted from the manifest + manifest.jsonl -----------------

def extract_features(manifest_uri: str) -> dict:
    """Pull the counts the monitor compares. Reads the manifest
    (exercise-01) and the attached corpus manifest.jsonl."""
    # ... I/O omitted for brevity; returns:
    return {
        "doc_count":        int,
        "chunk_count":      int,
        "source_mix":       {"notion": int, "slack": int, "s3": int, ...},
        "language_mix":     {"en": int, "es": int, ...},
        "chunk_len_hist":   [int, int, ...],   # bins: 0-50, 50-100, ... tokens
        "embedding_norms":  [float, ...],     # sampled 10k vectors (if available)
    }


# ---- Monitor entry point ---------------------------------------------------

def run(current_manifest_uri: str, baseline_manifest_uris: list[str]) -> dict:
    cur  = extract_features(current_manifest_uri)
    base = _merge_baseline([extract_features(u) for u in baseline_manifest_uris])

    report = {"current": {}, "baseline": {}, "signals": []}

    # Doc count / chunk count — absolute percent change.
    doc_delta = (cur["doc_count"] - base["doc_count"]) / max(base["doc_count"], 1)
    report["signals"].append({
        "metric": "doc_count_change",
        "value":  doc_delta,
        "severity": _severity_abs(doc_delta, warn=0.10, crit=0.30),
    })

    # Source mix — χ² + PSI on categorical.
    source_bins = sorted(set(cur["source_mix"]) | set(base["source_mix"]))
    p = _laplace(base["source_mix"], source_bins)
    q = _laplace(cur["source_mix"],  source_bins)
    report["signals"].append({
        "metric": "source_mix_psi",
        "value":  psi(p, q),
        "severity": _severity_psi(psi(p, q)),
    })

    # Language mix — same shape.
    lang_bins = sorted(set(cur["language_mix"]) | set(base["language_mix"]))
    p = _laplace(base["language_mix"], lang_bins)
    q = _laplace(cur["language_mix"],  lang_bins)
    report["signals"].append({
        "metric": "language_mix_psi",
        "value":  psi(p, q),
        "severity": _severity_psi(psi(p, q)),
    })

    # Chunk-length histogram — JS.
    cur_hist = _norm_hist(cur["chunk_len_hist"])
    base_hist = _norm_hist(base["chunk_len_hist"])
    report["signals"].append({
        "metric": "chunk_length_js",
        "value":  js(dict(enumerate(base_hist)), dict(enumerate(cur_hist))),
        "severity": _severity_js(js(dict(enumerate(base_hist)), dict(enumerate(cur_hist)))),
    })

    # Embedding-norm distribution — KS on continuous.
    import scipy.stats as st
    ks_stat, ks_p = st.ks_2samp(base["embedding_norms"], cur["embedding_norms"])
    report["signals"].append({
        "metric": "embedding_norm_ks",
        "value":  ks_stat,
        "p":      ks_p,
        "severity": "crit" if ks_p < 0.01 and ks_stat > 0.05 else "warn" if ks_p < 0.05 else "ok",
    })

    report["worst"] = max((s["severity"] for s in report["signals"]), key=_severity_rank)
    return report


# ---- Severity rules --------------------------------------------------------

def _severity_abs(v: float, warn: float, crit: float) -> str:
    av = abs(v)
    if av >= crit: return "crit"
    if av >= warn: return "warn"
    return "ok"

def _severity_psi(v: float) -> str:
    if v >= 0.25: return "crit"
    if v >= 0.10: return "warn"
    return "ok"

def _severity_js(v: float) -> str:
    if v >= 0.10: return "crit"
    if v >= 0.05: return "warn"
    return "ok"

def _severity_rank(s: str) -> int:
    return {"ok": 0, "warn": 1, "crit": 2}[s]
```

Design notes:

- Reads the *manifest* (exercise-01), not the raw index. The
  manifest is the stable, auditable artifact.
- Baseline is a *rolling 14-day average*, not a single-day
  snapshot — a single-day baseline is noisy and creates false
  alerts the morning after any Monday.
- Embedding norms are sampled (10k vectors), not all-pairs —
  cheap and sufficient for a 2-sample KS. For `cosine`-metric
  indexes with L2-normed vectors, all norms are 1.0 by
  construction; skip the KS in that case.

### 2.4 `drift/query_monitor.py` — query-side drift

```python
"""Query drift monitor. Reads the retrieval access log
(1-hour aggregates, 7-day baseline). Aggregation is append-only
in a columnar store (ClickHouse / BigQuery / Snowflake); this
module reads aggregates, not row-level logs — PII stays out of
the monitor process."""
from __future__ import annotations

import math
from collections import Counter

from drift.corpus_monitor import psi, js, _laplace, _severity_psi, _severity_js


def features_for_window(window_from: str, window_to: str) -> dict:
    """Pull per-window aggregates from the warehouse:
        q_length_hist:  log-bucketed query token counts
        language_mix:   detected language per query (langdetect / fastText lid.176)
        intent_mix:     a classifier's label distribution (or embedding-cluster id)
        rare_term_rate: fraction of query tokens absent from the index's vocabulary
        empty_result_rate: fraction of queries returning 0 hits
        hit_at_10_rate: fraction of queries with >= 1 clicked result in top-10
                        (implicit feedback; optional)
    """
    ...


def run(current_window: tuple[str, str], baseline_window: tuple[str, str]) -> dict:
    cur  = features_for_window(*current_window)
    base = features_for_window(*baseline_window)

    report = {"signals": []}

    # Query length distribution: JS on log-bucketed histogram.
    q_len_js = js(
        _laplace(base["q_length_hist"], sorted(set(base["q_length_hist"]) | set(cur["q_length_hist"]))),
        _laplace(cur["q_length_hist"],  sorted(set(base["q_length_hist"]) | set(cur["q_length_hist"]))),
    )
    report["signals"].append({"metric": "query_length_js", "value": q_len_js, "severity": _severity_js(q_len_js)})

    # Language mix: PSI.
    bins = sorted(set(base["language_mix"]) | set(cur["language_mix"]))
    p = _laplace(base["language_mix"], bins)
    q = _laplace(cur["language_mix"],  bins)
    report["signals"].append({"metric": "query_language_psi", "value": psi(p, q), "severity": _severity_psi(psi(p, q))})

    # Intent mix: PSI.
    bins = sorted(set(base["intent_mix"]) | set(cur["intent_mix"]))
    p = _laplace(base["intent_mix"], bins)
    q = _laplace(cur["intent_mix"],  bins)
    report["signals"].append({"metric": "query_intent_psi", "value": psi(p, q), "severity": _severity_psi(psi(p, q))})

    # Rare-term rate: absolute delta. Rises when the corpus and
    # query vocabulary drift apart.
    rare_delta = cur["rare_term_rate"] - base["rare_term_rate"]
    report["signals"].append({
        "metric": "rare_term_rate_delta",
        "value":  rare_delta,
        "severity": "crit" if rare_delta > 0.10 else "warn" if rare_delta > 0.05 else "ok",
    })

    # Empty-result rate: a leading indicator of coverage loss.
    empty_delta = cur["empty_result_rate"] - base["empty_result_rate"]
    report["signals"].append({
        "metric": "empty_result_rate_delta",
        "value":  empty_delta,
        "severity": "crit" if empty_delta > 0.05 else "warn" if empty_delta > 0.02 else "ok",
    })

    return report
```

Notes:

- **Aggregates, not rows.** The monitor reads `COUNT(*) GROUP BY
  bucket` from the warehouse. The warehouse never exposes raw
  query text to this process — no PII travels into the drift
  pipeline. Mitigates OWASP ML Top 10 **ML10 Model Poisoning
  via Monitoring Pipeline** and NIST AI RMF **MANAGE 2.1**
  (privacy of operational data).
- **Intent mix.** A small language-model or embedding-cluster
  classifier assigns each query to one of ~20 intents. Even a
  coarse classifier is useful for drift (shifts stand out); a
  perfect classifier is not needed.
- **Rare-term rate** is the sparse-side canary: the fraction of
  query tokens that are *out-of-vocabulary* vs. the corpus
  (approximate by Bloom-filter membership in a build-time
  vocabulary). A sharp rise indicates either query drift or a
  stale index.
- **Empty-result rate** is the coverage canary: if 10% of
  queries now return zero hits where last week 3% did,
  something broke — probably a filter or an ACL change.

### 2.5 `drift/quality_monitor.py` — quality-side drift

```python
"""Quality drift monitor. Two signals:
   (a) scheduled offline eval (exercise-02) against a stable qrels
       — the authoritative number; weekly.
   (b) implicit feedback (clicks, thumbs, dwell) summarized to
       per-day retriever-win rate — the daily high-frequency signal.
"""
from __future__ import annotations

import statistics


def run_scheduled_eval(build_id: str) -> dict:
    """Invokes exercise-02's gate code path against the live index.

    Returns a diff vs. the manifest-time eval (exercise-01) and vs.
    the baseline recorded on main."""
    ...


def run_implicit(window_from: str, window_to: str) -> dict:
    """Pulls per-day:
        - click-through rate on top-10
        - dwell (median seconds on top-1 clicked result)
        - thumbs-up ratio
        - re-query-within-60s ratio (negative signal: user did not get what they wanted)
    Compares against a 14-day baseline window."""
    cur  = ...
    base = ...
    return {
        "signals": [
            {"metric": "ctr@10_delta",    "value": cur["ctr10"]    - base["ctr10"],
             "severity": "crit" if cur["ctr10"] - base["ctr10"] < -0.02 else "ok"},
            {"metric": "requery60s_delta","value": cur["requery60"] - base["requery60"],
             "severity": "crit" if cur["requery60"] - base["requery60"] > 0.05 else "ok"},
        ]
    }
```

Notes:

- **The scheduled eval is the authoritative signal.** Implicit
  feedback is noisy and biased (position bias; cold-start on new
  features). Use it to *detect* a possible regression; use the
  eval to *decide* whether to act.
- **Pair the signals.** A CTR drop without an eval regression is
  usually a UI or query-mix change, not a retriever regression.
  An eval regression with flat CTR is a retriever regression on
  intents users rarely visit — still worth fixing.
- **Scheduled eval cadence.** Weekly is a good default. Running
  it every build (exercise-03) is already happening via the
  attached manifest's `eval` block; the scheduled rerun catches
  the case where the live index is older than the latest build
  and never got evaluated against updated qrels.

### 2.6 `drift/alerts.md` — threshold table and routing

```markdown
# Drift alert routing

Every alert has: severity, owner, suggested remediation path, and
a short name for the oncall channel.

## Corpus drift

| Signal                     | Warn   | Crit   | Owner     | Remediation |
|----------------------------|-------:|-------:|-----------|-------------|
| doc_count_change (abs)     | 10%    | 30%    | ingest    | Verify source pipeline. If intentional, rebuild. |
| chunk_count_change (abs)   | 10%    | 30%    | ingest    | Same. |
| source_mix_psi             | 0.10   | 0.25   | ingest    | Verify no source dropped; investigate new sources. |
| language_mix_psi           | 0.10   | 0.25   | ingest    | Localization coverage? Rebuild + eval on the new language. |
| chunk_length_js            | 0.05   | 0.10   | ingest    | Chunker-config regression; check cfg_sha matches expected. |
| embedding_norm_ks (D, p)   | D>0.05 p<0.05 | D>0.05 p<0.01 | retrieval | Embedding provider changed? Verify model revision. |

## Query drift

| Signal                     | Warn   | Crit   | Owner      | Remediation |
|----------------------------|-------:|-------:|------------|-------------|
| query_length_js            | 0.05   | 0.10   | product    | New feature routing differently? Expand qrels if the shape is here to stay. |
| query_language_psi         | 0.10   | 0.25   | product    | Localization expansion? Expand qrels; evaluate language-specific recall. |
| query_intent_psi           | 0.10   | 0.25   | product    | Same. |
| rare_term_rate_delta       | +5pp   | +10pp  | retrieval  | Corpus vocab stale? Incremental ingest cadence; rebuild embeddings. |
| empty_result_rate_delta    | +2pp   | +5pp   | retrieval  | Filter regression? Check ACL/filter config changes in the last 24h. |

## Quality drift

| Signal                     | Warn   | Crit   | Owner      | Remediation |
|----------------------------|-------:|-------:|------------|-------------|
| scheduled_eval_R@10_delta  | -1pp   | -2pp   | retrieval  | Cutover-drill (exercise-01) ticket; identify regressing slice. |
| scheduled_eval_nDCG@10_d.  | -1.5pp | -3pp   | retrieval  | Same. |
| ctr@10_delta               | -1pp   | -2pp   | retrieval  | Correlate with cutover / deploys / query-mix shift before paging. |
| requery60s_delta           | +2pp   | +5pp   | retrieval  | Same. |

## Routing

- `warn` → 24h Slack bot message to the owning team channel. No
  pager.
- `crit` → pager to the owning oncall. Ack required within 30 min.
- Two `warn`s on correlated metrics within 4 hours upgrade to
  `crit` (correlation set in `routing.yaml`).
- Any `crit` on quality drift *and* any `warn` on corpus drift
  within 24h: the ticket pre-selects the cutover-drill runbook
  (exercise-01).

## Hysteresis

To prevent alert flapping on borderline PSI values, each alert
requires the metric to cross threshold for **two consecutive
windows** (two hours for query, two nights for corpus, two weeks
for scheduled eval). Implicit-feedback alerts (CTR, requery60s)
use a moving-average window of 7 days to prevent single-spike
pages.
```

### 2.7 `drift/runbook.md` — per-alert runbook entries

```markdown
# Drift runbook

## CORPUS_DOC_COUNT_CHANGE (warn/crit)

1. Compare the new `doc_count` in the manifest against the
   previous three nights' manifests. Is the change abrupt or a
   trend?
2. Check the ingest DAG (exercise-03) for failed source-loader
   runs. A 30% drop is often "one source stopped landing."
3. If the change is intentional (a planned bulk ingest / purge),
   acknowledge the alert; no rebuild needed.
4. If unintentional, open an incident; the index is wrong.
   Decide: roll back the alias to the previous build (exercise-01
   rollback), or push forward if the drop is tolerable.

## CORPUS_EMBEDDING_NORM_KS (crit)

1. Check the manifest's `embedding.revision` vs yesterday's.
   An embedding provider silently swapping behind the same
   `model_id` is the classic cause.
2. If the revision changed: the system followed "latest" instead
   of a pinned revision — fix the config, rebuild, re-cut.
3. If the revision is unchanged: check for a dimensionality or
   normalization config change (MRL truncation, L2-norm toggle).

## QUERY_INTENT_PSI (warn/crit)

1. Pull the top-10 intents from yesterday vs the 14-day baseline.
   Which intent shifted most?
2. Is the shift a product feature (route from a new UI, campaign,
   bot)? Likely yes; it is not a retriever problem.
3. Expand the qrels (mod-109 exercise-01) to cover the new intent
   shape. Trigger a baseline refresh on the next scheduled eval.
4. If no feature explains the shift — bot / scraping? Tag affected
   traffic and exclude from drift calculations going forward.

## QUERY_EMPTY_RESULT_RATE_DELTA (crit)

1. Pull per-filter breakdowns. A specific `customer_id` suddenly
   returning zero hits is usually an ACL regression.
2. Diff the `acl_policy_version` in the current manifest against
   the previous one.
3. If ACL regressed: roll back the ACL cfg, rebuild, cut over.
4. If ACL is unchanged: the index may be missing a slice; check
   the corpus-side ingest report for that slice.

## QUALITY_SCHEDULED_EVAL_REGRESSION (crit)

1. Compare the scheduled-eval result to the gate run from the
   current build's manifest (exercise-01). If the manifest's
   eval already showed the regression, the question is "why did
   this build get aliased" — audit exercise-01 Phase 4.
2. If the manifest's eval was green and the scheduled eval is
   red: either the qrels changed (check `qrels_sha`) or the
   live index drifted (shouldn't happen on an immutable artifact
   — rebuild; investigate the drift as a separate incident).
3. File the cutover-drill runbook (exercise-01 §2.5).

## QUALITY_CTR_DROP (warn/crit)

1. Correlate with recent deploys, UI changes, feature rollouts.
2. Correlate with CORPUS_* and QUERY_* alerts from the same
   window — if any of those fired, treat the CTR drop as the
   downstream symptom of the other.
3. If no correlated driver: pull a slice of queries with lower
   CTR than baseline and send to the qrels team for labeling;
   possible recall/relevance drift the current qrels do not cover.
```

### 2.8 Scheduling and observability

- Corpus monitor: nightly, immediately after exercise-03's DAG
  completes. A gate step inside the DAG itself emits the
  `drift_report` artifact to object storage.
- Query monitor: every hour. Reads warehouse aggregates; cheap.
- Quality monitor:
  - Scheduled eval: weekly (Sunday 02:00). A separate workflow
    invokes exercise-02's gate against the live index.
  - Implicit feedback: daily rollup from the click log.
- Metrics expose: `drift_signal_value{metric="...",severity="..."}`
  as Prometheus gauges. The dashboard renders each metric as its
  own panel; one-glance view of "green/yellow/red" per signal.

### 2.9 Worked example — three alerts in one week

Day 1 (Monday, 02:30): **CORPUS_LANGUAGE_PSI = 0.18 (warn).**

- Investigation: `pt-BR` appeared in the language mix (0.7%) for
  the first time. Marketing expanded to Brazil over the weekend.
- Action: ack the warning. Update the qrels (mod-109 exercise-01)
  to include a handful of `pt-BR` queries. Schedule a quality
  monitor rerun after the next weekly eval.

Day 3 (Wednesday, 10:00): **QUERY_INTENT_PSI = 0.31 (crit).**

- Investigation: "billing" intent went from 8% → 24% of queries.
  A backend bug started routing payment-support tickets into
  the search-first experience instead of a form.
- Action: product team fixes the bug; monitor expected to self-
  resolve within 48 hours. No retriever action.

Day 6 (Saturday, 15:20): **QUALITY_REQUERY60S_DELTA = +6pp
(crit).**

- Investigation: no CORPUS alert, no QUERY alert. The scheduled
  eval ran yesterday and was within budget. But implicit feedback
  says users are re-querying.
- Pull a slice: the "pt-BR" queries from the Monday event are
  underperforming — the retriever has no Portuguese tokenizer
  configured on the sparse side, and BM25 is matching nothing.
- Action: rebuild with the correct analyzer; cut over
  (exercise-01). Note in the memo that the drift monitor caught
  what a single-dimension eval missed because the qrels did not
  yet cover `pt-BR`.

The three-signal structure makes the chain visible: a corpus
drift leads to a query drift leads to a quality drift, each
caught by its own monitor, each with a specific remediation.

## 3. Validation steps

A correct submission can be validated with these checks:

1. **PSI / JS parity.** Hand-compute PSI and JS on two trivial
   distributions (`[0.5, 0.5]` vs `[0.6, 0.4]`) and confirm the
   `drift/*` implementations match to 1e-6.
2. **Baseline smoothing.** Run the corpus monitor on two runs
   with identical inputs — every signal is `ok`. Run with one
   fictitious new source at 15% — `source_mix_psi` is between
   warn and crit, matching hand calculation.
3. **Severity hierarchy.** Threshold tests: PSI 0.09 is `ok`,
   0.15 is `warn`, 0.30 is `crit`. KS p-value tests: `p=0.05`
   but `D=0.03` is `ok` (significance without effect);
   `p=0.001` and `D=0.08` is `crit`.
4. **Hysteresis.** A single-window `crit` does not page; two
   consecutive windows do. Verify with a two-row test fixture.
5. **PII isolation.** The query-monitor code imports no
   raw-log reader. `grep -rE 'raw_query|q.text' drift/` returns
   zero application-code hits.
6. **Routing fires.** Simulate a `crit` on corpus → the alert
   lands in the `retrieval` channel (or paging target), not
   in the general dashboard.
7. **Runbook linkage.** The payload for each alert type links
   back to the specific `drift/runbook.md` section (deep link).
8. **End-to-end replay.** Backfill the monitors over a known
   historical window that included a real incident; verify the
   relevant metric crossed threshold in real time.
9. **Separability.** Hand-crafted scenarios (corpus-only shift,
   query-only shift, quality-only regression) fire *exactly*
   one of the three classes each.
10. **No alert fatigue.** Run the monitors over 30 days of
    normal production with no incidents. Fewer than 2 `warn`
    alerts per week on average; zero `crit`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Three monitors separate | Corpus, query, quality are distinct code paths and distinct alert classes. |
| Standard divergence math | Uses PSI, JS, KS, χ² with cited thresholds; no hand-rolled custom metrics without justification. |
| Baselines are rolling | Each monitor compares against a smoothed window, not a single-day snapshot. |
| Thresholds have severities | Every signal has warn + crit thresholds and an owner. |
| Hysteresis present | Alert requires two consecutive windows over threshold, or an equivalent anti-flap mechanism. |
| Reads aggregates, not rows | Query monitor reads warehouse aggregates; no PII in the drift process. |
| Runbook entries present | Each alert type has a runbook section with specific diagnostic steps and a named remediation. |
| Quality is authoritative | Scheduled eval (exercise-02) is the quality authority; implicit feedback is a leading indicator, not a trigger-of-last-resort. |
| Metrics exposed | Prometheus gauges per signal; dashboard renders a per-signal panel. |
| Separability demonstrated | A hand-crafted drift of one kind fires exactly one alert class, not three. |
| Integration points | Corpus monitor consumes exercise-01 manifests; quality monitor invokes exercise-02 code path; alerts can file an exercise-01 cutover ticket. |
| Alert budget sensible | Normal-week alert volume is bounded (< 2 warn/week, 0 crit). |

Fail-fast smells:

- One monster drift dashboard with 60 panels and no severity.
  Nobody reads it; it is the equivalent of no monitor.
- Single-day baseline. Alerts on Monday morning every Monday.
- "We'll tune thresholds later." Thresholds are a product
  decision; ship with concrete numbers tied to measured
  historical variance, or the monitor has no teeth.
- CTR is the quality trigger. Position bias and traffic mix move
  CTR for reasons unrelated to retrieval; eval is the authority.
- Query-text in the alert payload. PII in Slack. Hash or
  summarize.
- No runbook. The alert fires and the oncall googles for 40
  minutes trying to figure out what to do.

## 5. Common mistakes

- **Conflating the three drifts.** A single "retrieval looks
  worse" dashboard tells you nothing about which lever to
  pull. Three monitors, three alert classes, three runbooks.
- **Picking thresholds from no-data.** PSI=0.25 is a rule of
  thumb; your actual noise floor may be 0.15 or 0.05. Measure
  30 days of baseline variance; set warn at 2σ, crit at 3σ.
- **No hysteresis.** PSI jitters across its threshold on
  statistical noise; without hysteresis the pager rings on
  nothing most weeks.
- **Alerts without owners.** Signals routed to "the team" age
  into folklore. Every signal has an owner.
- **PII in the monitor.** Query text is user data. The monitor
  must read aggregates; a Grafana query that pulls raw query
  text for a drift dashboard is an exfiltration surface.
- **Treating CTR as truth.** CTR drops on UI rollouts, on new
  user cohorts, on seasonal shifts. It is a detective, not a
  judge. The eval gate is the judge.
- **Using a monitor as the trigger for a cutover without going
  through the drill.** The drift monitor opens a ticket; the
  cutover drill (exercise-01) owns the change; skipping the
  drill because "the alert is red" is how a bad rebuild ships.
- **Forgetting correlated causes.** A corpus drift causes a
  query drift (rare-term rate), which causes a quality drift
  (lower nDCG). Routing must deduplicate these so three pagers
  do not fire for one root cause.
- **No historical replay.** A monitor whose thresholds have not
  been replayed against a past incident is theater. Replay at
  least one historical event and confirm the monitor fires on
  it.

## 6. References

Primary methods:

- Population Stability Index — Wikipedia summary with formula and
  thresholds: [en.wikipedia.org/wiki/Population_stability_index](https://en.wikipedia.org/wiki/Population_stability_index).
  Originates in practitioner credit-scoring literature; treat
  thresholds (0.10/0.25) as rules of thumb, not absolutes.
- Jensen–Shannon Divergence — Lin, 1991, "Divergence measures
  based on the Shannon entropy," IEEE Transactions on Information
  Theory 37(1). Standard reference for the symmetric,
  bounded-KL variant.
- Kolmogorov–Smirnov test — `scipy.stats.ks_2samp` docs:
  [docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
- χ² test — `scipy.stats.chi2_contingency` docs:
  [docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html).

Tooling:

- Prometheus — [prometheus.io/docs](https://prometheus.io/docs/).
  Gauges per signal are the recommended exposition shape.
- Evidently — [github.com/evidentlyai/evidently](https://github.com/evidentlyai/evidently).
  Open-source drift library; useful as a reference for standard
  metric implementations even if you keep the hand-rolled
  monitor.
- NannyML — [github.com/NannyML/nannyml](https://github.com/NannyML/nannyml).
  Alternative open-source library focused on post-deployment ML
  monitoring.

Governance framings:

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: **MEASURE 2.6** (post-deployment monitoring; drift
  detection is the baseline requirement), **MANAGE 2.2**
  (continuous monitoring and remediation), **MAP 5.1** (impact
  assessment — triage of drift incidents).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: **ML02 Data Poisoning Attack** (drift monitors catch
  poisoning that would otherwise manifest as slow quality drift),
  **ML10 Model Poisoning** — detection pipeline integrity.
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: **AML.T0020 Poison Training Data**, **AML.T0043
  Craft Adversarial Data** — drift anomalies may be adversarial;
  the monitor is also a security signal.

Related exercises:

- exercise-01 — the manifest the corpus monitor reads; the
  cutover drill that remediates quality drift.
- exercise-02 — the scheduled eval that is the quality
  authority.
- exercise-03 — the DAG that runs the corpus monitor as a
  post-build step and may be triggered by a drift alert.
- exercise-05 — the cache hit rate is a leading indicator of
  query drift (hit rate falls when query distribution moves).
- mod-109 exercise-01 — qrels refresh as remediation for
  query-intent drift.
