# Exercise 05 — Solution: BEIR / MTEB Subset Run and Corpus-Fit Delta Report

This is a reference solution for
`exercise-05-beir-mteb-subset-run-and-corpus-delta-report.md`. It is
an **implementation + analysis** exercise. The learner runs a
retrieval pipeline against a subset of the BEIR tasks (and/or MTEB
retrieval tasks), records per-task metrics, and then computes a
**corpus-fit delta** — the difference between each public task's
nDCG and the same pipeline's nDCG on the in-house qrels from
exercise-01. The report says which public tasks predict in-house
performance and which do not.

The teaching point: public benchmarks are useful signals but they
are not your product. A system that is #1 on BEIR MS-MARCO can be
mediocre on your internal corpus, and vice versa, because "better
retrieval" is corpus-specific. The exercise is about building the
smallest piece of infrastructure that catches the mismatch before
it ships: run a benchmark subset on a cadence, compare the delta
to in-domain, and refuse to generalise from benchmarks alone.

## 1. Solution overview

The deliverable has six parts:

1. **`src/beir_runner.py`** — loads a BEIR task (via the official
   `beir` package), runs a provided retrieval callable over the
   task's corpus and queries, writes a TREC run file, and shells
   out to exercise-02's scorer.
2. **`src/mteb_runner.py`** — the same shape for MTEB retrieval
   tasks via the `mteb` package.
3. **`tasks.yaml`** — the subset. Named BEIR + MTEB tasks with
   a rationale for each pick (why that task was chosen to predict
   in-house performance).
4. **`src/corpus_fit.py`** — joins the per-task public metrics
   with the in-house metric (exercise-02 output on the exercise-
   01 qrels), computes per-task delta and rank correlation
   between public and in-house metric across pipeline variants
   (from exercise-04's grid).
5. **`reports/corpus_fit.md`** — the headline report: per-task
   table of public vs in-house nDCG@10, the overall rank
   correlation, and the "most predictive" and "least predictive"
   benchmark tasks for this corpus.
6. **`notes/go_no_go.md`** — a one-page memo on how to use the
   report in a release gate: which benchmark deltas you allow
   to block a release, which you treat as drift signals only,
   and what the "corpus-fit score" is that you track over time.

The exercise is scoped to *running the benchmark subset and
interpreting the result*, not to competing on public leaderboards.
A full BEIR run is expensive; we target one or two tasks per
"family" (QA, fact-verification, duplicate-question, biomedical)
plus the in-house qrels.

## 2. Worked answer or implementation

### 2.1 BEIR / MTEB in one paragraph

**BEIR** (Thakur et al. 2021) is a heterogeneous retrieval
benchmark that packages 18 datasets into a single evaluation
protocol: each dataset ships a corpus, a set of queries, and
qrels, and the scoring is done with `trec_eval` (recall, nDCG,
MRR at standard cutoffs). **MTEB** (Muennighoff et al. 2022) is
a broader embedding benchmark that re-packages many BEIR tasks
plus classification, clustering, and semantic-similarity tasks;
its *retrieval* section is effectively BEIR with the MTEB
runner. Running "BEIR" or "MTEB retrieval" means picking a
subset of tasks, running your retriever over each task's corpus,
and scoring against that task's qrels. The out-of-domain
behaviour of your retriever is the signal.

### 2.2 `tasks.yaml` — the task subset

```yaml
# BEIR/MTEB subset for the mod-109 corpus-fit study. Pick one
# or two tasks per family so the suite runs in a day on a
# single workstation.

beir:
  - name: scifact
    family: fact-verification
    rationale: >
      Small (~5k docs), scientific claims. Predicts how well the
      retriever handles short factual queries against a
      technical corpus.
  - name: nfcorpus
    family: biomedical
    rationale: >
      Biomedical IR, narrow domain, short queries. Predicts
      sensitivity to domain-specific vocabulary.
  - name: fiqa
    family: finance-qa
    rationale: >
      Long natural-language questions, financial corpus.
      Predicts behaviour on consumer-style questions.
  - name: trec-covid
    family: ad-hoc-biomedical
    rationale: >
      Short topic queries, biomedical. Pair with nfcorpus to
      distinguish "biomedical domain" effects from "query
      style" effects.

mteb_retrieval:
  - name: ArguAna
    family: argument-retrieval
    rationale: >
      Each query is itself a short argument; the retriever has
      to find a counter-argument. Predicts behaviour on
      long-form queries.
  - name: TRECCOVID
    family: ad-hoc-biomedical
    rationale: >
      Overlaps BEIR's trec-covid; included to verify the two
      runners agree on scoring.

inhouse:
  qrels: ../exercise-01-build-a-judgment-set-with-curated-and-llm-assisted-labels/qrels/judgment_set.trec
  description: >
    Exercise-01 qrels. The in-house corpus and queries are the
    thing we actually ship; every public task is a signal
    *about* this one.

metric: nDCG@10
```

**Choices and why:**

- **One or two tasks per family.** BEIR has 18 datasets; running
  all of them takes hours and most of them are redundant for
  signalling *your* corpus. Cover the families that match your
  product surface (fact-verification, long-form QA, biomedical,
  etc.) and stop.
- **Rationale per task.** If you cannot say why a task predicts
  your corpus, you will not know what to do when its delta
  blows up.
- **Overlap between BEIR and MTEB on one task** (TRECCOVID /
  trec-covid) is deliberate: the two runners should produce
  the same nDCG@10 on the same qrels; if they do not, the
  bug is in the runner wiring, not the retriever.

### 2.3 `src/beir_runner.py` — the BEIR runner

```python
"""Run a retrieval callable against a BEIR task, emit a TREC run.

Depends on the official `beir` package for task loading only;
scoring goes through exercise-02's ir_measures-based scorer so
every task is reported on the same metric definitions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from beir import util
from beir.datasets.data_loader import GenericDataLoader

@dataclass
class BeirRun:
    task_name: str
    qrels_path: Path
    run_path: Path

RetrievalFn = Callable[[str, dict[str, str]], dict[str, float]]
# query_id -> query_text, corpus_id -> {title, text}; returns doc_id -> score

def _emit_trec_qrels(qrels_map: dict, out: Path) -> None:
    with out.open("w", encoding="utf-8") as fh:
        for qid, rels in sorted(qrels_map.items()):
            for doc_id, grade in sorted(rels.items()):
                fh.write(f"{qid} 0 {doc_id} {grade}\n")

def _emit_trec_run(
    results: dict[str, dict[str, float]], tag: str, out: Path, top_k: int = 1000
) -> None:
    with out.open("w", encoding="utf-8") as fh:
        for qid in sorted(results):
            ranked = sorted(
                results[qid].items(), key=lambda kv: kv[1], reverse=True
            )[:top_k]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                fh.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")

def run_beir_task(
    task_name: str,
    retrieve_batch: Callable[
        [dict[str, str], dict[str, dict]], dict[str, dict[str, float]]
    ],
    data_root: Path,
    out_root: Path,
    system_tag: str,
    split: str = "test",
) -> BeirRun:
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/"
        f"BEIR/datasets/{task_name}.zip"
    )
    data_path = util.download_and_unzip(url, str(data_root))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    results = retrieve_batch(queries, corpus)
    out = out_root / task_name
    out.mkdir(parents=True, exist_ok=True)
    qrels_path = out / f"{task_name}.qrels.trec"
    run_path = out / f"{task_name}.run.trec"
    _emit_trec_qrels(qrels, qrels_path)
    _emit_trec_run(results, system_tag, run_path)
    (out / "task.json").write_text(json.dumps({
        "task": task_name,
        "split": split,
        "n_queries": len(queries),
        "n_docs": len(corpus),
        "n_qrels_entries": sum(len(v) for v in qrels.values()),
    }))
    return BeirRun(task_name, qrels_path, run_path)
```

**Choices and why:**

- **`retrieve_batch` is injected.** The runner does not care how
  the retrieval is implemented — it just needs the
  `(query_id → doc_id → score)` map. This keeps the BEIR
  runner independent of the pipeline builder and makes
  exercise-04's grid trivial to run across tasks.
- **Scoring goes through exercise-02's `ir_measures` scorer,
  not BEIR's bundled scorer.** BEIR's default is correct, but
  routing every run through one scorer eliminates "we scored
  with two different libraries and got different numbers"
  issues. The in-house and BEIR nDCG@10s are comparable only
  if they come from the same scorer.
- **TREC emission is the handoff format.** Everything else
  (BEIR's dict-of-dicts, MTEB's native format) is converted to
  TREC files on disk. One format, one scorer, one set of
  edge cases.
- **Top-K 1000** in the run file. BEIR's default cutoffs go up
  to 1000 (recall@1000 is a standard reporting cutoff). Clip
  smaller for speed only if you are certain you will not want
  deeper recall numbers later.

### 2.4 `src/mteb_runner.py` — the MTEB runner

```python
"""Run a retrieval callable against an MTEB retrieval task.

MTEB packages BEIR tasks under a different loader; the output
shape is identical to beir_runner.py, so corpus_fit.py can
consume either."""
from __future__ import annotations

from pathlib import Path

import mteb

from .beir_runner import BeirRun, _emit_trec_qrels, _emit_trec_run

def run_mteb_task(
    task_name: str,
    retrieve_batch,
    out_root: Path,
    system_tag: str,
    split: str = "test",
) -> BeirRun:
    task = mteb.get_task(task_name)
    task.load_data()
    corpus = task.corpus[split]
    queries = task.queries[split]
    qrels = task.relevant_docs[split]
    results = retrieve_batch(queries, corpus)
    out = out_root / task_name
    out.mkdir(parents=True, exist_ok=True)
    qrels_path = out / f"{task_name}.qrels.trec"
    run_path = out / f"{task_name}.run.trec"
    _emit_trec_qrels(qrels, qrels_path)
    _emit_trec_run(results, system_tag, run_path)
    return BeirRun(task_name, qrels_path, run_path)
```

**Choices and why:**

- **Share the TREC emit helpers with the BEIR runner.** Two
  entry points, one file format on disk, one scorer. The code
  surface stays small.
- **Rely on MTEB's own task registry** (`mteb.get_task`) for
  task metadata. Do not hand-transcribe BEIR-vs-MTEB task
  name mappings.

### 2.5 `src/corpus_fit.py` — the corpus-fit analysis

```python
"""Join public-task and in-house per-system metrics; compute deltas
and rank correlation between public and in-house rankings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class TaskFit:
    task: str
    public_metric: float
    inhouse_metric: float
    delta: float                    # public - inhouse

def gather_metrics(
    runs_root: Path, scorer,            # scorer: (qrels, run) -> {metric: float}
    tasks: list[str],
    system_tag: str,
    metric: str,
    inhouse_qrels: Path, inhouse_run: Path,
) -> tuple[list[TaskFit], float]:
    fits: list[TaskFit] = []
    for t in tasks:
        qrels_path = runs_root / t / f"{t}.qrels.trec"
        run_path = runs_root / t / f"{t}.run.trec"
        agg = scorer(qrels_path, run_path)
        public_metric = agg[metric]
        in_agg = scorer(inhouse_qrels, inhouse_run)
        inhouse_metric = in_agg[metric]
        fits.append(TaskFit(
            task=t,
            public_metric=public_metric,
            inhouse_metric=inhouse_metric,
            delta=public_metric - inhouse_metric,
        ))
    return fits, _rank_correlation(fits)

def _rank_correlation(fits: list[TaskFit]) -> float:
    """Kendall tau between public and in-house metric across tasks.

    Interpretation note: for one *system*, this tau is computed
    across tasks and answers the question "do easy tasks for the
    system match easy in-house queries?" For multi-system corpus-
    fit analysis, compute tau across systems for one task."""
    if len(fits) < 3:
        return float("nan")
    pub = np.array([f.public_metric for f in fits])
    inh = np.array([f.inhouse_metric for f in fits])
    tau, _ = stats.kendalltau(pub, inh)
    return float(tau)

def multi_system_tau(
    public_by_system: dict[str, dict[str, float]],  # system -> task -> metric
    inhouse_by_system: dict[str, float],            # system -> metric
) -> dict[str, float]:
    """For each public task, Kendall tau between system ranking on
    that task and system ranking on the in-house metric. High tau
    ⇒ task predicts in-house performance."""
    systems = sorted(public_by_system)
    inhouse = np.array([inhouse_by_system[s] for s in systems])
    out: dict[str, float] = {}
    tasks = sorted(next(iter(public_by_system.values())).keys())
    for t in tasks:
        pub = np.array([public_by_system[s][t] for s in systems])
        tau, _ = stats.kendalltau(pub, inhouse)
        out[t] = float(tau)
    return out

def render_corpus_fit(fits: list[TaskFit], metric: str) -> str:
    lines = [
        f"| Task | Public {metric} | In-house {metric} | Delta |",
        "|---|---|---|---|",
    ]
    for f in sorted(fits, key=lambda f: abs(f.delta), reverse=True):
        lines.append(
            f"| {f.task} | {f.public_metric:.4f} | "
            f"{f.inhouse_metric:.4f} | {f.delta:+.4f} |"
        )
    return "\n".join(lines) + "\n"
```

**Choices and why:**

- **Two tau computations, two questions.** Single-system tau
  across tasks answers "which tasks look easy for *this*
  system's shape of competence." Multi-system tau across
  systems for a given task answers "does this task rank
  systems the same way the in-house qrels do." The second is
  the one that determines whether the task is a useful proxy.
- **Kendall tau, not Spearman.** Kendall is more robust on
  small n (we have 4–6 tasks and a handful of systems). For
  larger grids either is fine.
- **Public − in-house, not the reverse.** A positive delta
  means the public task is easier than your corpus (common,
  because BEIR tasks have relatively small corpora and dense
  qrels). The sign is meaningful; keep the convention stable
  so the memo reads correctly over time.
- **Return the fits and the tau as a pair.** The caller
  decides whether to fail the release gate on the tau, on an
  individual delta, or both.

### 2.6 Example output — `reports/corpus_fit.md`

Running the subset against a hybrid+rerank baseline:

```markdown
# Corpus-Fit Report — baseline hybrid+rerank, 2026-11-16

Pipeline: dense (baseline) + BM25 (RRF) + cross-encoder rerank,
top-N = 50, qrels v1.0.0.

## Per-task nDCG@10

| Task | Public nDCG@10 | In-house nDCG@10 | Delta |
|---|---|---|---|
| scifact | 0.7412 | 0.7205 | +0.0207 |
| fiqa | 0.5118 | 0.7205 | -0.2087 |
| nfcorpus | 0.3654 | 0.7205 | -0.3551 |
| trec-covid | 0.8012 | 0.7205 | +0.0807 |
| ArguAna | 0.4902 | 0.7205 | -0.2303 |

## Multi-system rank agreement (Kendall tau, nDCG@10, 8 systems)

| Task | Tau vs in-house |
|---|---|
| scifact | +0.78 |
| fiqa | +0.43 |
| nfcorpus | -0.07 |
| trec-covid | +0.69 |
| ArguAna | +0.21 |

## Reading

- **scifact and trec-covid are the most predictive**: high
  rank agreement and small-to-moderate absolute delta. These
  are the two public tasks worth putting in the release gate.
- **nfcorpus is not predictive**: tau near zero. The in-house
  corpus is not biomedical; a system that does well on
  nfcorpus is uncorrelated with one that does well on the
  in-house qrels. Keep nfcorpus for drift monitoring, do not
  use it in a gate.
- **ArguAna has a low tau on this corpus** because ArguAna's
  retrieval task (find a counter-argument) is structurally
  unlike the in-house workload. Low tau is a feature here —
  it tells you ArguAna is measuring something your product
  does not care about.
```

**Reading cue for the reviewer:** a task with tau ≥ 0.6 on
≥ 8 systems is a candidate for the release gate. Tasks with
lower tau go into the dashboard but not the gate.

### 2.7 `notes/go_no_go.md` — the release-gate memo

Use the corpus-fit report as follows:

1. **Release gate threshold.** Pre-commit to a per-task
   threshold on each "gated" task (tau ≥ 0.6). Example: "hybrid
   rerank must score ≥ 0.70 nDCG@10 on scifact and ≥ 0.76 on
   trec-covid." Below the threshold the release is blocked.
2. **Drift detector.** The non-gated tasks (low tau) are still
   worth running: a large unexpected move on nfcorpus does not
   gate the release but opens a ticket. Catastrophic cross-task
   drops flag embedder or ingestion bugs that the gated tasks
   might miss.
3. **In-house qrels remains the ground truth.** Public tasks
   are signals. If the exercise-01 qrels disagree with the
   gated public tasks, the qrels win. Update the gated-task
   list only when both agree.
4. **Corpus-fit score over time.** Track the median tau across
   gated tasks per release. A monotone decline is the warning
   that your corpus has drifted away from the public tasks
   you picked; refresh the subset.

## 3. Validation steps

1. Each BEIR / MTEB task downloads (or loads from cache), and
   the task's `n_queries`, `n_docs`, `n_qrels_entries` are
   recorded in `task.json` for provenance.
2. Every public-task scoring call uses the same scorer and
   same metric definition as the in-house run (exercise-02's
   `ir_measures` wrapper).
3. The TRECCOVID × trec-covid duplicate (BEIR vs MTEB) scores
   within numerical noise of each other (same qrels, same
   run, same scorer).
4. The per-task delta uses the in-house metric from
   exercise-02's output on exercise-01's qrels, not a separate
   rebuild of either.
5. Kendall tau is computed across ≥ 3 systems and ≥ 3 tasks;
   the memo refuses to report tau for a single system.
6. Release-gate thresholds are declared in `notes/go_no_go.md`
   *before* the release run, not back-fit to the result.
7. A re-run on the same pipeline + same task cache produces the
   same per-task metrics (the retrieval callable must itself be
   deterministic — exercise-04's seed policy applies).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Task subset | Declared in `tasks.yaml` with a rationale per task tied to the in-house workload. |
| Family coverage | Subset covers ≥ 2 families relevant to the product surface. |
| One scorer | All tasks (public + in-house) go through exercise-02's scorer. |
| Metric parity | Same metric and same cutoffs used everywhere. |
| Multi-system tau | Computed across ≥ 3 systems; reported per task. |
| Release gate | At least one gated task with a pre-declared threshold and tau ≥ 0.6. |
| Drift tier | Non-gated tasks included for drift monitoring. |
| Reproducibility | Task caches pinned; retrieval callable deterministic; two runs produce identical numbers. |
| Memo | Names the gate, the drift tier, and a trigger to refresh the subset. |

Fail-fast smells:

- "We shipped because BEIR nDCG@10 went up." Benchmark number
  without the in-house comparison is a marketing artifact.
- One task, one system, "tau". Statistically meaningless.
- Different scorers for public and in-house. The delta number
  is scorer variance mixed with retrieval variance.
- No task-level rationale. The subset is a vibe; drift against
  one of these tasks will not be interpretable.
- Threshold set post-run. Gaming your own gate.
- Running the subset once per quarter. Corpus and model drift
  happen on shorter cycles; the point of a small subset is it
  can run weekly.

## 5. Common mistakes

- **Generalising from a public benchmark to your corpus.** A
  system that wins on MS-MARCO (web search) might be mediocre
  on your enterprise corpus. Thakur et al.'s original BEIR
  paper is explicit about this — their contribution was a
  heterogeneous benchmark precisely because single-task
  wins did not transfer.
- **Picking too many public tasks.** 10 tasks at 2 hours each
  is a week of GPU time. The gain over 4 well-chosen tasks is
  marginal.
- **Picking too few.** One task is one data point. The multi-
  system tau cannot be computed; the drift detector has
  nothing to compare.
- **Scoring with BEIR's default and your scorer and reporting
  different numbers.** Pick one scorer and use it everywhere;
  document the choice.
- **Chasing a benchmark leaderboard.** If an "improvement" to
  your retriever raises BEIR nDCG@10 on one task by +0.05 and
  drops in-house nDCG@10 by -0.02, you have shipped a
  benchmark artifact, not a product improvement.
- **Not caching the task downloads.** BEIR task archives are
  big and slow; re-downloading on every CI run is wasteful
  and unreliable. Mirror them.
- **Forgetting the delta sign.** `public - in-house` should
  be stable across reports; flipping the sign silently
  invalidates every chart you built on top.
- **Treating MTEB-retrieval as a strict superset of BEIR.**
  Most BEIR tasks are in MTEB, but a few are not, and MTEB's
  handling of the splits occasionally differs. Verify on the
  overlap (TRECCOVID / trec-covid) before trusting the
  equivalence.

## 6. References

- Benchmarks:
  - Thakur, N. et al. (2021), "BEIR: A Heterogeneous Benchmark
    for Zero-shot Evaluation of Information Retrieval Models,"
    *NeurIPS Datasets and Benchmarks* —
    [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663).
    Source of the datasets and the evaluation protocol.
  - Muennighoff, N. et al. (2022), "MTEB: Massive Text
    Embedding Benchmark" —
    [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316).
  - MTEB task registry — [huggingface.co/mteb](https://huggingface.co/mteb).
- Tooling:
  - Official `beir` package —
    [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - Official `mteb` package —
    [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb).
  - `ir_measures` scorer used throughout —
    [ir-measur.es](https://ir-measur.es/).
- Rank-correlation:
  - Kendall, M. G. (1938), "A New Measure of Rank Correlation,"
    *Biometrika*.
  - SciPy `scipy.stats.kendalltau` —
    [docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).
- Upstream:
  - exercise-01 in-house qrels —
    `../exercise-01-build-a-judgment-set-with-curated-and-llm-assisted-labels/SOLUTION.md`.
  - exercise-02 scorer —
    `../exercise-02-compute-recall-mrr-ndcg-with-ir-measures/SOLUTION.md`.
  - exercise-04 grid supplies the multi-system data for the
    tau computation —
    `../exercise-04-orthogonal-ablation-grid-and-attribution/SOLUTION.md`.
- Risk and governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML06 Output Integrity — shipping on a
    public-benchmark delta without the corpus-fit check is
    an integrity failure on the release-gate surface.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around *Evaluation* motivate running the
    benchmark subset on a cadence (drift detection).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Manage* function requires that external-benchmark
    signals be mapped to in-production impact; the
    corpus-fit score is that mapping for retrieval.
