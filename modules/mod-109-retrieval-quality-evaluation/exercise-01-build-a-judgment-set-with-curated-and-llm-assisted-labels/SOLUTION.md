# Exercise 01 — Solution: Build a Judgment Set with Curated and LLM-Assisted Labels

This is a reference solution for
`exercise-01-build-a-judgment-set-with-curated-and-llm-assisted-labels.md`.
It is a mixed **design + implementation** exercise. The learner builds
a judgment set (also called *qrels* — query relevance judgments) that
the rest of mod-109 uses to score retrieval runs. The twist is that
the labels come from two sources: a small *curated* set labelled by
humans and a larger *LLM-assisted* set labelled by a judge model,
with the curated set used to audit and calibrate the LLM labels.

The teaching point: a retrieval metric is only as trustworthy as the
qrels it is computed against. Shipping a judgment set without naming
who labelled it, under what rubric, with what agreement, and against
what sampling pool is shipping *vibes with a number*. This exercise
produces the qrels **and** the governance paper trail that lets you
defend them in a review.

## 1. Solution overview

The deliverable has six parts:

1. **`qrels/judgment_set.trec`** — the final qrels in TREC format
   (`query_id 0 doc_id grade`) with graded relevance in `{0,1,2,3}`.
2. **`qrels/metadata.jsonl`** — per-label provenance: labeller
   (`human` / `llm:<model-id>`), timestamp, rubric version, pool
   source (`bm25_top50` / `dense_top50` / `union`), and reconciliation
   outcome.
3. **`labeling/rubric.md`** — the written rubric with grade
   definitions, boundary cases, and a worked example per grade.
4. **`labeling/llm_judge_prompt.md`** — the prompt + JSON Schema the
   LLM judge uses. Deterministic, graded, forced to emit a rationale.
5. **`src/build_qrels.py`** — the pipeline: query sampling, pool
   construction, LLM judging, human curation of a calibration slice,
   agreement computation (Cohen's κ), reconciliation, TREC emission.
6. **`notes/audit.md`** — the audit memo: κ against the curated
   slice, confusion matrix, retained-vs-dropped labels, and the
   decision to ship.

The exercise is scoped to *producing and defending the qrels*, not
to scoring any retrieval system against them. Downstream exercises
(02 metrics, 03 significance, 04 ablation) consume
`qrels/judgment_set.trec` as a stable input.

## 2. Worked answer or implementation

### 2.1 Judgment sets in one paragraph

A judgment set maps `(query_id, doc_id)` pairs to a relevance grade.
TREC-style graded qrels use 0 (not relevant), 1 (marginally
relevant), 2 (relevant), 3 (highly relevant). The TREC `qrels` text
file is four whitespace-separated columns: `query_id`, an unused
iteration column (conventionally `0`), `doc_id`, `grade`. Any pair
not in the file is implicitly 0 — which is only safe when the pool
of judged documents is large enough that unjudged documents are
near-certain negatives. That pooling assumption is why the sampling
step below matters more than the metric choice.

### 2.2 Query sampling plan

| Stratum | Why we sample it | Size (example) |
|---|---|---|
| Head intents — top production queries by volume | These dominate user-visible quality; miss them and the metric does not reflect what users feel. | 40 |
| Torso intents — mid-frequency queries covering distinct intents | Protects against overfitting to the head. | 40 |
| Known-hard: queries that triggered escalations, thumb-downs, or "I don't know" refusals in production | Negative capital — regressions here are expensive. | 10 |
| Adversarial / out-of-corpus: queries whose correct answer is "no evidence in corpus" | Gold for the refusal evaluation and for calibrating LLM judges that over-label. | 10 |
| **Total** | | **100** |

100 queries is the lower bound at which a single-system recall@10
delta of ~3 absolute points begins to be detectable with paired
tests (exercise-03 formalises this; here we just size the pool to
stay above that floor). Smaller sets are legal for internal
smoke-tests but should not back release decisions.

### 2.3 Pooling — which documents do we judge?

Unjudged ⇒ not-relevant is the pooling assumption. To keep it
valid, the judged pool must contain every document the systems
under evaluation might rank highly. The TREC convention is:

- Run every system under evaluation on every query.
- For each query, collect the union of the top-K results from
  every system. Call this the **pool**.
- Judge every `(query, doc)` in the pool.

For the exercise we pool from three systems with K=50 each:
production BM25, production dense, and the mod-106 hybrid fusion
baseline. The expected pool size per query is 60–120 after
dedup (systems overlap heavily on the easy items).

### 2.4 `labeling/rubric.md` — the rubric

```markdown
# Relevance Grading Rubric v1 (mod-109)

Grade a `(query, document)` pair on a 0–3 scale. Grade the document
as it stands — do not imagine edits or follow-up retrieval.

- **3 — Highly relevant.** The document directly answers the query
  or contains the exact fact the query asks for. A grounded
  generator could cite this document and answer the query from it
  alone, with no synthesis across other documents.
- **2 — Relevant.** The document is clearly on-topic and would be
  useful context for answering the query, but it does not fully
  answer it on its own (e.g., discusses the correct subject but
  omits the specific fact; answers a sibling question in the same
  topic cluster).
- **1 — Marginally relevant.** The document mentions the entity or
  topic in the query but is not useful for answering it. Think
  "same keyword, different intent."
- **0 — Not relevant.** Off-topic, wrong entity, or correctly
  on-topic but factually in conflict with the known answer (grade
  as 0 and flag `contradiction=true` in metadata).

## Boundary cases

- Document answers the query for a stale time period — grade 1
  unless the query is explicitly historical.
- Document is a correct answer in a related locale or jurisdiction
  that the query did not specify — grade 2 if the user likely
  wants it, 1 otherwise. Leave a note.
- Document is retrieval system output (another LLM's summary) —
  grade 0 for mod-109 qrels; we judge the primary corpus, not
  summaries of it.

## Worked examples

(three examples — one at each of grades 1, 2, 3 — on queries
drawn from the pilot slice, redacted to avoid teaching to the test)
```

**Why this shape:**

- A graded scale lets nDCG distinguish "two relevant docs at the
  top" from "one highly-relevant doc at the top" (exercise-02
  depends on this). A binary scale collapses that signal.
- The refusal grade (0 + contradiction flag) feeds mod-108's
  refusal eval. Without it, a correctly-refusing system looks
  identical to a system that missed the answer.
- Worked examples per grade are the single highest-leverage
  investment. Faggioli et al. (2023) found that most LLM-human
  disagreement collapses when the rubric has concrete examples
  for the hard boundaries.

### 2.5 `labeling/llm_judge_prompt.md` — the LLM judge

````markdown
# System

You are a relevance judge. Score a `(query, document)` pair on the
0–3 rubric below. You MUST respond by calling the
`relevance_judgment` tool with a JSON payload conforming to the
schema. Be strict: when uncertain between two grades, pick the
lower.

Rubric:

- 3 — Highly relevant: directly answers the query.
- 2 — Relevant: on-topic, useful context, not a full answer.
- 1 — Marginally relevant: same entity/topic, wrong intent.
- 0 — Not relevant: off-topic or contradicts the known answer.

Rules:

1. Grade the document as written. Do not guess at missing facts.
2. If the document contradicts a plausible correct answer, grade 0
   and set `contradiction=true`.
3. Output a one-sentence `rationale` that names the specific
   evidence or absence of evidence driving the grade.
4. If the document is not in English and the query is in English
   (or vice versa), grade 0 and set `language_mismatch=true`.

# User

Query: {{query_text}}

Document:
{{doc_text}}
````

Paired JSON Schema (abbreviated):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RelevanceJudgment",
  "type": "object",
  "additionalProperties": false,
  "required": ["grade", "rationale"],
  "properties": {
    "grade": {"type": "integer", "enum": [0, 1, 2, 3]},
    "rationale": {"type": "string", "minLength": 1, "maxLength": 400},
    "contradiction": {"type": "boolean"},
    "language_mismatch": {"type": "boolean"}
  }
}
```

**Why this shape:**

- "Pick the lower grade when uncertain" shifts the LLM's bias from
  "over-label as relevant" (the common failure mode) to
  "under-label as relevant" (safer: forces curators to review
  borderline pairs explicitly).
- A `rationale` is required. It is not trained on, but it
  dramatically speeds the curator's audit — a 3-grade with the
  rationale "document says retention is 180 days, matches query
  about retention period" is defensible in seconds; the same 3
  with no rationale is a 30-second re-read.
- Temperature should be 0 at inference; the LLM judge is being
  used as a classifier, not a brainstormer.

### 2.6 `src/build_qrels.py` — the pipeline

```python
"""Build a graded qrels judgment set with LLM labels audited against
a human-curated calibration slice."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

RNG_SEED = 0x1D9E
CALIBRATION_FRAC = 0.15  # fraction of pairs double-labeled by humans

@dataclass(frozen=True)
class Pair:
    query_id: str
    doc_id: str

@dataclass
class Judgment:
    pair: Pair
    grade: int               # 0..3
    source: str              # "llm:<model-id>" | "human:<labeller-id>"
    rationale: str
    contradiction: bool = False
    language_mismatch: bool = False

def build_pool(
    queries: list[dict], runs: list[dict], top_k: int = 50
) -> list[Pair]:
    """Union pool across all submitted runs at depth top_k."""
    pool: set[Pair] = set()
    for run in runs:
        for qid, ranked in run["results"].items():
            for doc_id, _score in ranked[:top_k]:
                pool.add(Pair(qid, doc_id))
    return sorted(pool, key=lambda p: (p.query_id, p.doc_id))

def sample_calibration(
    pool: list[Pair], frac: float = CALIBRATION_FRAC
) -> set[Pair]:
    rng = random.Random(RNG_SEED)
    n = max(1, int(len(pool) * frac))
    return set(rng.sample(pool, n))

def llm_label(
    pair: Pair,
    get_text: Callable[[str], str],
    get_query: Callable[[str], str],
    judge: Callable[[str, str], dict],   # (query_text, doc_text) -> parsed payload
    judge_model_id: str,
) -> Judgment:
    payload = judge(get_query(pair.query_id), get_text(pair.doc_id))
    return Judgment(
        pair=pair,
        grade=int(payload["grade"]),
        source=f"llm:{judge_model_id}",
        rationale=payload.get("rationale", ""),
        contradiction=bool(payload.get("contradiction", False)),
        language_mismatch=bool(payload.get("language_mismatch", False)),
    )

def cohen_kappa(
    pairs_a: dict[Pair, int], pairs_b: dict[Pair, int]
) -> float:
    """Cohen's kappa on the intersection of labeled pairs. Equal weights.
    For graded disagreement, a weighted kappa variant is more informative;
    see notes/audit.md for the quadratic-weight computation on the real
    slice."""
    shared = set(pairs_a) & set(pairs_b)
    if not shared:
        return float("nan")
    a = [pairs_a[p] for p in shared]
    b = [pairs_b[p] for p in shared]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(shared)
    # expected agreement by chance
    ca = Counter(a); cb = Counter(b); n = len(shared)
    p_e = sum((ca[g] / n) * (cb[g] / n) for g in {0, 1, 2, 3})
    if p_e >= 1.0:
        return 1.0
    return (agree - p_e) / (1.0 - p_e)

def reconcile(
    llm: Judgment, human: Judgment
) -> tuple[Judgment, str]:
    """Return the shipping judgment and a reconciliation tag."""
    if llm.grade == human.grade:
        return human, "agree"
    # human wins for release; keep the LLM grade in metadata.
    return (
        Judgment(
            pair=human.pair,
            grade=human.grade,
            source=human.source,
            rationale=f"human override; LLM said {llm.grade} — {llm.rationale}",
            contradiction=human.contradiction,
            language_mismatch=human.language_mismatch,
        ),
        "human_override",
    )

def write_trec_qrels(path: Path, judgments: Iterable[Judgment]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for j in sorted(
            judgments, key=lambda j: (j.pair.query_id, j.pair.doc_id)
        ):
            fh.write(f"{j.pair.query_id} 0 {j.pair.doc_id} {j.grade}\n")

def write_metadata(path: Path, judgments: Iterable[Judgment]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for j in judgments:
            fh.write(json.dumps({**asdict(j), "pair": asdict(j.pair)}) + "\n")
```

**Choices and why:**

- **Pool union across every run under evaluation.** If a document is
  never in any system's top-K, judging it is wasted effort. If a
  document is in *any* system's top-K it must be judged, otherwise
  that system's apparent score is inflated by free top-ranked
  negatives being counted as implicit 0s.
- **Human calibration slice, not full human re-labelling.** 15% is
  a working compromise: enough pairs to compute a stable κ against
  the LLM, few enough that curator effort is realistic. Under 10%
  and κ becomes noisy; over 30% and you have paid the human-label
  cost without using the LLM.
- **Human override on disagreement, LLM grade preserved in
  metadata.** The shipped qrels are the human truth; the LLM grade
  stays visible so the κ can be recomputed later and so the
  audit can name the error rate.
- **Cohen's κ with equal weights** is good enough for a
  release-gate check. For graded disagreement (a 0-vs-3 error is
  worse than 2-vs-3), compute a quadratic-weighted κ as well and
  report both; the function above is intentionally simple to keep
  the solution focused.

### 2.7 Example audit output — `notes/audit.md`

```markdown
# Judgment set audit — v1.0.0

- Pool: 100 queries × avg 71 docs each = 7,114 pairs.
- LLM judge: `claude-sonnet-4-6`, temperature 0, prompt version
  `v1`.
- Calibration slice: 1,067 pairs (15.0%), double-labelled by two
  annotators (A, B) and the LLM.

## Agreement

- Human A vs Human B — Cohen's κ = 0.78 (substantial).
- Human A vs LLM — κ = 0.69 (substantial).
- Human B vs LLM — κ = 0.71 (substantial).
- Human consensus (AB-agree, n=878) vs LLM — κ = 0.76.

## Confusion matrix — LLM vs human consensus

|  | H=0 | H=1 | H=2 | H=3 |
|---|---|---|---|---|
| LLM=0 | 412 | 18 | 2 | 0 |
| LLM=1 | 24 | 97 | 11 | 1 |
| LLM=2 | 3 | 14 | 118 | 9 |
| LLM=3 | 0 | 1 | 15 | 153 |

Interpretation: LLM errors are off-by-one on grade boundaries (1↔2,
2↔3), not catastrophic 0↔3 flips. Safe to ship for mod-109
exercises.

## Shipping decision

- 7,114 LLM labels; of these, 1,067 are human-reconciled.
- 134 pairs flagged as `contradiction=true`; 11 as
  `language_mismatch=true`. Both routed to curator review before
  shipping.
- Qrels v1.0.0 cut; metadata.jsonl retains full provenance.
- Re-audit on the next LLM judge upgrade (any change to model id
  or prompt version invalidates this audit; re-run `build_qrels`
  with a fresh calibration slice).
```

(Numbers above are *illustrative of the shape of the output*, not
measurements from a real run — the learner computes their own
from their calibration slice.)

## 3. Validation steps

1. The qrels file parses as TREC format (`query_id 0 doc_id grade`,
   grade ∈ {0,1,2,3}, no duplicate `(query_id, doc_id)` keys).
2. `metadata.jsonl` has one entry per qrels line; `source` names
   either a human labeller or `llm:<model-id>@<prompt-version>`.
3. Cohen's κ on the human calibration slice (A vs B) ≥ 0.6
   (substantial agreement). Below that, the rubric is ambiguous —
   fix the rubric before trusting any LLM labels.
4. Cohen's κ of LLM vs human consensus is reported and above the
   team's floor for release (a common default: κ ≥ 0.6 for
   graded, or κ ≥ 0.4 for binary mapping; name the floor in the
   audit memo).
5. No `(query_id, doc_id)` appears in both the "shipped" qrels and
   the "excluded due to language/contradiction flag" list.
6. The pool covers every document ranked top-K by every system
   under evaluation on every query (sanity check: for each
   (system, query), `set(top_k) ⊆ set(judged_pool_for_query)`).
7. Downstream (exercise-02) can load the qrels with `ir_measures`
   without a parse error.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Format | TREC qrels, graded 0–3, no dupes, companion `metadata.jsonl`. |
| Provenance | Every label has a labeller id and a rubric/prompt version. |
| Rubric | Written grade definitions + worked example per grade. Boundary cases named. |
| Pooling | Union pool across every system under evaluation at top-K. Pool depth is named in the audit memo. |
| Query sampling | Stratified across head / torso / known-hard / adversarial. Counts per stratum are named. |
| LLM judge | Prompt is versioned, schema-enforced, temperature 0, forced to emit rationale. |
| Curation slice | ≥ 10% double-labelled; κ reported A-vs-B and LLM-vs-consensus. |
| Reconciliation | Human overrides LLM on disagreement; LLM grade preserved in metadata. |
| Refusal / contradiction | 0-grade documents that contradict a plausible correct answer are flagged. Out-of-corpus queries sampled explicitly. |
| Audit memo | Shipping decision with named κ floor and a trigger for re-audit. |

Fail-fast smells:

- A-vs-B κ is below 0.4 and the team still shipped. The rubric is
  ambiguous; fix it before touching metrics.
- The LLM judge is non-deterministic (temperature > 0, no fixed
  seed). Every re-run produces slightly different qrels and
  longitudinal comparisons are meaningless.
- The judgment set was built from a single system's top-K. Pooling
  bias will make that system look artificially good — every doc
  it ranks top-K is judged, every doc the competitor ranks top-K
  but that one missed is counted as implicit 0.
- Metadata omits the prompt version. A silent prompt edit produces
  silent qrels drift.
- Contradictions are graded 0 but not flagged. mod-108's refusal
  eval cannot tell "no answer in corpus" from "correct refusal
  under conflicting evidence" without the flag.

## 5. Common mistakes

- **Shipping qrels built from one system's top-K.** The system you
  built the qrels from will look best because every one of its
  top-K documents is labelled; its competitors score against
  implicit 0s on documents nobody judged.
- **Binary relevance under a graded metric.** Collapsing to
  relevant/not-relevant loses the signal that nDCG exists to
  reward (ordering highly-relevant above merely-relevant).
- **No curator slice.** An LLM-labelled qrels set without a human
  calibration slice is a marketing artifact, not a measurement
  tool. You cannot defend the metric number in a review.
- **LLM judge at temperature > 0.** Each re-run is a new
  judgment set. Deltas across runs mix model variance with
  system changes.
- **No prompt / rubric versioning.** You cannot tell whether a
  κ regression is a rubric edit, a model upgrade, or a system
  change.
- **Over-sampling the head.** All head-intent queries means the
  torso regresses silently.
- **Missing out-of-corpus queries.** The qrels cannot test
  refusal. mod-108's refusal exercise has no ground truth.
- **Pooling too shallow (top-K = 10).** Rare-but-correct
  documents that any reasonable future system would rank 11–50
  are treated as implicit 0, capping the ceiling.
- **Treating `contradiction=true` as noise.** These are the
  highest-value items for the generator pipeline — they stress
  the citation and refusal contract.

## 6. References

- Query relevance judgments / TREC protocol:
  - Voorhees, E. & Harman, D. (eds.), *TREC: Experiment and
    Evaluation in Information Retrieval* — MIT Press, 2005.
    Canonical reference for pool-based evaluation and qrels
    format.
  - NIST TREC overview and tools —
    [trec.nist.gov](https://trec.nist.gov/).
- LLM-assisted relevance labelling:
  - Faggioli, G. et al. (2023), "Perspectives on Large Language
    Models for Relevance Judgment" —
    [arxiv.org/abs/2304.09161](https://arxiv.org/abs/2304.09161).
    Recommends rubric-anchored prompts and human calibration
    slices.
  - Thomas, P. et al. (2024), "Large Language Models Can
    Accurately Predict Searcher Preferences" —
    [arxiv.org/abs/2309.10621](https://arxiv.org/abs/2309.10621).
- Agreement statistics:
  - Cohen, J. (1960), "A coefficient of agreement for nominal
    scales," *Educational and Psychological Measurement*.
  - For graded qrels, Fleiss or quadratic-weighted κ; see
    Fleiss, J. L. (1971), "Measuring nominal scale agreement
    among many raters."
- Downstream consumer (exercise-02):
  - `ir_measures` qrels loader —
    [ir-measur.es](https://ir-measur.es/).
- Risk and governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML02 Data Poisoning — the judgment set *is*
    training/eval data for retrieval; label drift is a
    poisoning surface.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around *Evade ML Model* motivate pooling across
    systems (not just the model under evaluation).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure* function requires documented provenance and
    uncertainty; the audit memo is the artifact that satisfies
    it.
