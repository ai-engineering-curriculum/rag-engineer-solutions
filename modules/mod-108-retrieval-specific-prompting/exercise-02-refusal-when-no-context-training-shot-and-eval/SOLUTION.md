# Exercise 02 — Solution: Refusal When No Context — Training-Shot and Eval

This is a reference solution for
`exercise-02-refusal-when-no-context-training-shot-and-eval.md`.
It is an **implementation** exercise. The learner builds a few-shot
training-shot block that teaches the model to **refuse when the
retrieved context does not support the question**, writes the eval
set that measures the refusal rate against an under-refusal baseline,
and wires the two into the mod-108 prompt pipeline from exercise-01.

The teaching point: *refusal is the first guardrail a RAG system
owes its users*. A model that hallucinates on empty or weak context
is worse than a model that returns no answer — the empty answer
sends the user back to search; the hallucinated answer sends them
to a wrong decision. The refusal contract has three ingredients: a
canonical refusal string (defined in exercise-01), training-shots
that demonstrate the refusal under realistic empty-context
conditions, and an eval harness that measures false-answer rate
and false-refusal rate side by side.

## 1. Solution overview

The deliverable has five parts:

1. **`prompts/refusal_shots.md`** — the few-shot block appended to
   the exercise-01 template. Six shots: three where the context
   supports an answer (and the model answers), three where it
   does not (and the model emits the canonical refusal string).
2. **`eval/refusal_eval_set.jsonl`** — a 50-row eval set with two
   partitions: **`supported`** (query + context that *does* support
   an answer) and **`unsupported`** (query + context that does
   *not*). Each row has a gold answer or a gold refusal.
3. **`eval/run_eval.py`** — the harness that runs the model on
   every row, classifies each response as `answered` or
   `refused` by exact-string match on the refusal, and reports a
   confusion matrix plus the two headline metrics:
   **false-answer rate** (on `unsupported`) and **false-refusal
   rate** (on `supported`).
4. **`notes/eval_results.md`** — a worked run against a reference
   model with both metrics, confusion matrix, and the four
   per-row failure types annotated.
5. **`notes/triage.md`** — a short triage guide that maps each
   failure type to the likely fix (template, shots, retrieval,
   assembler).

The exercise is scoped to the **refusal-vs-hallucination axis
only**. Citation correctness is measured in exercise-04;
contradiction handling in exercise-03.

## 2. Worked answer or implementation

### 2.1 The refusal contract in one paragraph

The mod-108 template (exercise-01) declares one canonical refusal
string: `"I cannot answer this from the retrieved context."` When
the context does not support an answer, the model must emit that
string verbatim and no other text. This contract is what the eval
harness exact-matches. Few-shot examples are how the model learns
the contract without fine-tuning. The eval set is how you measure
whether the model actually honours it.

### 2.2 `prompts/refusal_shots.md` — the training-shot block

This block is appended to the system prompt from exercise-01's
inline template, inserted between the "Rules" section and the
`<context>` block. The *same* six shots are used for both the
inline and structured templates, with the response format adjusted
per template.

````markdown
## Examples

The following examples demonstrate the required behaviour. Follow
the same format in your own answer.

### Example 1 — supported, single source

<context>
[chunk:policy-vacation-v4]
All full-time employees accrue 15 paid vacation days per year.
Accrual begins on the start date and is prorated for the first year.
</context>

Question: How many vacation days do full-time employees accrue per year?

Answer: Full-time employees accrue 15 paid vacation days per year
[chunk:policy-vacation-v4].

### Example 2 — supported, two sources

<context>
[chunk:policy-overtime-v2]
Overtime is paid at 1.5× the base rate for hours worked beyond 40 in
a week.

[chunk:regional-exceptions-ca]
In California, overtime applies after 8 hours in a single day.
</context>

Question: How is overtime calculated for a California employee who
works a 9-hour day?

Answer: Overtime is paid at 1.5× the base rate for hours worked
beyond 40 in a week [chunk:policy-overtime-v2]. In California,
overtime additionally applies after 8 hours in a single day
[chunk:regional-exceptions-ca]. A 9-hour day therefore triggers
one hour of overtime.

### Example 3 — supported, requires inference within context

<context>
[chunk:holiday-schedule-2026]
Company holidays for 2026 include New Year's Day (Jan 1), Memorial
Day (May 25), Independence Day (Jul 4), Labor Day (Sep 7),
Thanksgiving (Nov 26), and Christmas (Dec 25).
</context>

Question: How many company holidays fall in the second half of 2026?

Answer: Four company holidays fall in the second half of 2026:
Independence Day (Jul 4), Labor Day (Sep 7), Thanksgiving (Nov 26),
and Christmas (Dec 25) [chunk:holiday-schedule-2026].

### Example 4 — unsupported, off-topic context

<context>
[chunk:policy-vacation-v4]
All full-time employees accrue 15 paid vacation days per year.
</context>

Question: What is the current employee discount on company
merchandise?

Answer: I cannot answer this from the retrieved context.

### Example 5 — unsupported, adjacent-topic context

<context>
[chunk:policy-overtime-v2]
Overtime is paid at 1.5× the base rate for hours worked beyond 40 in
a week.
</context>

Question: What is the double-time threshold for overtime?

Answer: I cannot answer this from the retrieved context.

### Example 6 — unsupported, empty context

<context>
</context>

Question: What is the fiscal year start date?

Answer: I cannot answer this from the retrieved context.
````

**Why these six shots:**

- **Three supported, three unsupported.** Equal balance teaches
  the model that refusal is a first-class behaviour, not a
  fallback. A shot block that is 5:1 supported-to-unsupported
  trains the model to answer by default.
- **Example 5 is the hardest shot.** Adjacent-topic context —
  overtime context, double-time question — is the most common
  real-world unsupported case. Models trained without this shot
  will "helpfully" extrapolate and hallucinate.
- **Example 6 is empty context.** Teaches the model that
  "nothing retrieved" is a retrieval-pipeline event, not a
  prompt-engineering challenge. Without this shot, models try
  to answer from pretraining knowledge.
- **The refusal string is byte-identical across shots.** This is
  what the eval harness exact-matches. A single stray punctuation
  variation defeats the contract.
- **Examples 1–3 cite verbatim `[chunk:<id>]` markers from the
  context.** This primes the inline citation behaviour from
  exercise-01 in the same shot block. Two contracts, one
  training surface.

### 2.3 `eval/refusal_eval_set.jsonl` — the eval set

50 rows, each one JSON object per line. Fields:

| field | type | description |
|---|---|---|
| `id` | string | stable row id (e.g., `sup-01`, `uns-01`) |
| `partition` | string | `"supported"` or `"unsupported"` |
| `question` | string | user-facing question |
| `context_block` | string | the serialised mod-107 assembler output |
| `legend` | array | `[{citation_id, source_id}, ...]` |
| `gold_answer_substring` | string or null | a substring that must appear in a correct answer (supported rows only) |
| `gold_citation_ids` | array | subset of the legend that a correct answer must cite (supported rows only) |
| `expected_refusal` | boolean | `true` on unsupported rows |

Partitions:

- **`supported`** (25 rows): 15 single-source, 7 multi-source,
  3 requires-light-inference. Drawn from the mod-106 eval corpus
  so the retrieval pipeline is held constant.
- **`unsupported`** (25 rows), four sub-types, five rows each:
  - **off-topic** — retrieved context about an unrelated subject.
  - **adjacent-topic** — same policy domain, different question
    (double-time vs overtime; parental leave vs vacation).
  - **partial** — context mentions the subject but does not
    answer the question (e.g., "retention policy exists" without
    the number).
  - **empty** — `context_block` is the empty string.
  - **outdated** — context contains a prior policy that was
    superseded (gold answer is refusal because the superseding
    chunk was not retrieved).

**Why this partition shape.** The four unsupported sub-types
decompose the "false-answer" metric into its distinct failure
modes. A model that refuses 100% on *empty* but 20% on
*adjacent-topic* has a specific, fixable problem: add
adjacent-topic shots. Aggregate false-answer rate hides that.

**Example rows:**

```jsonl
{"id":"sup-01","partition":"supported","question":"How many vacation days do full-time employees accrue per year?","context_block":"[chunk:policy-vacation-v4]\nAll full-time employees accrue 15 paid vacation days per year.","legend":[{"citation_id":"policy-vacation-v4","source_id":"policy-vacation-v4"}],"gold_answer_substring":"15","gold_citation_ids":["policy-vacation-v4"],"expected_refusal":false}
{"id":"uns-01","partition":"unsupported","question":"What is the current employee discount on company merchandise?","context_block":"[chunk:policy-vacation-v4]\nAll full-time employees accrue 15 paid vacation days per year.","legend":[{"citation_id":"policy-vacation-v4","source_id":"policy-vacation-v4"}],"gold_answer_substring":null,"gold_citation_ids":[],"expected_refusal":true}
{"id":"uns-02","partition":"unsupported","question":"What is the double-time threshold for overtime?","context_block":"[chunk:policy-overtime-v2]\nOvertime is paid at 1.5x the base rate for hours worked beyond 40 in a week.","legend":[{"citation_id":"policy-overtime-v2","source_id":"policy-overtime-v2"}],"gold_answer_substring":null,"gold_citation_ids":[],"expected_refusal":true}
{"id":"uns-16","partition":"unsupported","question":"What is the fiscal year start date?","context_block":"","legend":[],"gold_answer_substring":null,"gold_citation_ids":[],"expected_refusal":true}
```

### 2.4 `eval/run_eval.py` — the harness

```python
"""Refusal eval harness for mod-108 exercise-02.

Loads the eval set, calls a generation function on every row,
classifies the response, and prints the confusion matrix plus the
two headline metrics (false-answer rate, false-refusal rate)."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REFUSAL = "I cannot answer this from the retrieved context."

@dataclass
class EvalRow:
    id: str
    partition: str
    question: str
    context_block: str
    legend: list[dict]
    gold_answer_substring: str | None
    gold_citation_ids: list[str]
    expected_refusal: bool

@dataclass
class Result:
    row: EvalRow
    response: str
    classified: str   # "refused" | "answered"
    verdict: str      # "correct" | "false_answer" | "false_refusal" | "malformed"

def load_eval(path: str) -> list[EvalRow]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(EvalRow(**d))
    return rows

def classify(response: str) -> str:
    """Exact-match refusal string as the classifier.

    We intentionally refuse to be clever here — regex "I cannot" or
    "I don't know" matches masks real template-drift bugs."""
    return "refused" if response.strip() == REFUSAL else "answered"

def judge(row: EvalRow, response: str) -> str:
    classified = classify(response)
    if row.expected_refusal:
        return "correct" if classified == "refused" else "false_answer"
    # supported row
    if classified == "refused":
        return "false_refusal"
    # answered — basic grounding check: gold substring present
    if row.gold_answer_substring and row.gold_answer_substring.lower() not in response.lower():
        return "malformed"   # answered but not grounded
    return "correct"

def run(
    eval_path: str,
    generate: Callable[[str, str], str],
    out_path: str | None = None,
) -> dict:
    """generate(context_block, question) -> model response text."""
    rows = load_eval(eval_path)
    results: list[Result] = []
    for r in rows:
        resp = generate(r.context_block, r.question)
        verdict = judge(r, resp)
        results.append(Result(r, resp, classify(resp), verdict))

    sup = [x for x in results if x.row.partition == "supported"]
    uns = [x for x in results if x.row.partition == "unsupported"]

    false_answer_rate  = sum(1 for x in uns if x.verdict == "false_answer") / max(len(uns), 1)
    false_refusal_rate = sum(1 for x in sup if x.verdict == "false_refusal") / max(len(sup), 1)
    malformed_rate     = sum(1 for x in sup if x.verdict == "malformed") / max(len(sup), 1)

    summary = {
        "n_total":            len(results),
        "n_supported":        len(sup),
        "n_unsupported":      len(uns),
        "false_answer_rate":  round(false_answer_rate, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "malformed_rate":     round(malformed_rate, 4),
        "confusion": {
            "supported":   {"answered_correct": sum(1 for x in sup if x.verdict == "correct"),
                            "answered_malformed": sum(1 for x in sup if x.verdict == "malformed"),
                            "refused_wrong":    sum(1 for x in sup if x.verdict == "false_refusal")},
            "unsupported": {"refused_correct":  sum(1 for x in uns if x.verdict == "correct"),
                            "answered_wrong":   sum(1 for x in uns if x.verdict == "false_answer")},
        },
    }
    if out_path:
        Path(out_path).write_text(
            "\n".join(json.dumps({
                "id": r.row.id, "partition": r.row.partition,
                "verdict": r.verdict, "response": r.response,
            }) for r in results),
            encoding="utf-8",
        )
    return summary

if __name__ == "__main__":
    # Stub generator for self-test — always refuses.
    # Replace with a real client call in production.
    def stub(ctx: str, q: str) -> str:
        return REFUSAL
    print(json.dumps(run("eval/refusal_eval_set.jsonl", stub), indent=2))
```

**Choices and why:**

- **Exact-string match for refusal classification.** A regex that
  accepts "I can't answer", "I don't have enough information",
  etc. will *mask* template drift. The whole point of the
  canonical refusal string from exercise-01 is to be
  byte-matched. If the model emits a variation, that is a bug
  surfaced by the exact match.
- **Two headline metrics side by side.** False-answer rate alone
  invites the model to refuse everything. False-refusal rate
  alone invites the model to hallucinate confidently. Reporting
  both forces the honest trade-off.
- **`malformed` as its own verdict.** A supported-row answer
  that is "answered" but missing the gold substring is neither
  correct nor a refusal failure — it is a grounding failure
  that exercise-04 verifies more deeply. The harness surfaces
  it here so the two failure modes do not get conflated.
- **The per-row result file is JSONL.** A triage review on 50
  rows is small; the artefact is checked in with the run.

### 2.5 `notes/eval_results.md` — a worked run

With the stub generator (always refuses):

| Metric | Value |
|---|---|
| false_answer_rate | 0.00 (25/25 correctly refused) |
| false_refusal_rate | 1.00 (25/25 incorrectly refused) |
| malformed_rate | 0.00 |

The stub trivially passes the "no hallucination" bar and
trivially fails the "answer when supported" bar. This is the
degenerate baseline the eval exists to prevent.

A **reference-quality run** against a modern instruction-tuned
model with the exercise-01 template and exercise-02 shots
should land near the following (rough bands; exact numbers
depend on model, retrieval quality, and eval composition,
and should be re-measured once the project pins a specific
model):

| Metric | Target band |
|---|---|
| false_answer_rate (on unsupported) | ≤ 5% |
| false_refusal_rate (on supported) | ≤ 10% |
| malformed_rate (on supported) | ≤ 5% |

Per unsupported sub-type breakdown is where the triage value lives:

| Sub-type | Likely failure mode if elevated |
|---|---|
| off-topic | Model is using pretraining knowledge — add shot like Example 4; verify system prompt forbids prior knowledge. |
| adjacent-topic | Model is extrapolating — strengthen Example 5; retrieval may be pulling plausible-but-wrong chunks. |
| partial | Model is pattern-matching on topic without checking the question — strengthen rules clause in template. |
| empty | Model is falling back to pretraining — verify the template renders the `<context></context>` block even when empty. |
| outdated | Retrieval is at fault, not generation — surface in mod-106/107 freshness policy, not here. |

### 2.6 `notes/triage.md` — the triage guide

| Observation | Hypothesis | Fix location |
|---|---|---|
| `false_answer_rate` high on **off-topic** | Model relying on pretraining. | Reinforce rule 3 in exercise-01 template; add off-topic shots. |
| `false_answer_rate` high on **adjacent-topic** | Model extrapolating within a topic. | Strengthen Example 5; consider lowering reranker score threshold in mod-107. |
| `false_answer_rate` high on **partial** | Prompt does not distinguish "topic match" from "question match". | Tighten Rule 1 wording: "every factual sentence must be *directly supported*". |
| `false_refusal_rate` high on single-source | Shots skew unsupported; model over-generalises to refuse. | Rebalance shots; verify retrieval is returning the expected chunk. |
| `false_refusal_rate` high on multi-source | Model is confused by multiple chunks. | Confirm mod-107 assembler's ordering is stable; check context-block formatting is unambiguous. |
| `malformed_rate` high | Model answered but missed the gold substring. | Not a refusal bug — hand off to exercise-04 citation verification. |
| `false_refusal_rate` high on **empty context** rows after retrieval change | Retrieval over-filtering. | Mod-106/107, not here. |

## 3. Validation steps

1. The eval set loads cleanly as JSONL: every row parses as
   JSON, every row has a `partition` in `{"supported",
   "unsupported"}`, and partition sizes match (25 each).
2. The refusal string is byte-identical in the template, the
   shots, and the eval harness. A grep across the three files
   must return the same string.
3. The harness's `classify()` returns `"refused"` on the exact
   canonical string and `"answered"` on anything else,
   including minor variations like `"I cannot answer this
   from the retrieved context"` (missing period).
4. The confusion matrix sums to the eval set size.
5. Re-running the eval on the same model twice (with
   deterministic decoding or seeded sampling) produces the
   same metrics to within sampling noise.
6. The four unsupported sub-types are represented: a per-sub-type
   breakdown table is in `notes/eval_results.md`.
7. The shot block is appended to — not a replacement for — the
   exercise-01 rules section.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Canonical refusal string | Exactly one phrase, used identically in template, shots, eval set, and harness. |
| Shot balance | ≥ 3 supported and ≥ 3 unsupported shots; unsupported shots cover ≥ 2 sub-types (adjacent-topic and empty required). |
| Eval set size | ≥ 50 rows; partitions roughly balanced (±10%). |
| Eval sub-types | Unsupported partition has at least 4 sub-types: off-topic, adjacent-topic, partial, empty. |
| Harness classifier | Exact-string match on the refusal, not regex-fuzzy match. |
| Metrics | False-answer rate and false-refusal rate both reported; `malformed` surfaced separately. |
| Per-row artefact | Run produces a per-row JSONL output file suitable for triage. |
| Triage guide | Maps each failure mode to a fix location. |
| Reproducibility | Same model + same decoding → same metrics. |
| No cross-contamination | Citation correctness is not measured here; that lives in exercise-04. |

Fail-fast smells:

- "I cannot answer this" (missing "from the retrieved context")
  used somewhere; the exact-match classifier will mark every
  response as `answered`.
- 45 supported + 5 unsupported rows. The supported bar is
  trivial and the unsupported bar is sample-noise.
- Regex classifier that accepts "I don't know", "I'm not
  sure", "I cannot help with that". Masks template drift.
- No empty-context row in the eval. The commonest
  real-world case is untested.
- The harness returns an aggregate rate with no per-sub-type
  breakdown. Triage becomes guesswork.
- Shots use different `[chunk:<id>]` id formats than the
  mod-107 assembler emits (uppercase, numbered). Model will
  mimic the shots and fail exercise-04.
- The refusal string differs between the inline and structured
  templates. Two eval surfaces for the same behaviour.

## 5. Common mistakes

- **Fuzzy refusal matching.** The whole point of the canonical
  string is byte-match. Fuzzing it hides regressions.
- **Imbalanced partitions.** 45/5 trivially optimises for
  "answer confidently." 5/45 trivially optimises for "refuse
  everything." Keep it near 50/50.
- **Only testing empty context.** Empty context is the easy
  case. Adjacent-topic is where models hallucinate.
- **No adjacency gradient.** All unsupported rows are wildly
  off-topic. In production the hard cases are "same policy,
  different sub-question."
- **Measuring citation correctness here.** Exercise-02 is the
  refuse-or-not axis. Exercise-04 verifies citations
  per-claim.
- **Forgetting the empty-context fixture.** The template has to
  render `<context></context>` without error; models and
  renderers both choke on this when it is untested.
- **One number as the headline.** False-answer rate alone or
  false-refusal rate alone is optimisation bait.
  Report both.
- **Running the eval on a different template than production.**
  If the shots live in a separate file that production does
  not include, the eval measures a non-production artifact.
- **Changing the refusal string between versions without
  versioning the eval.** A silent string change makes every
  historical result incomparable.

## 6. References

- Grounded-answer / abstention:
  - Gao et al. (2023), "Enabling Large Language Models to
    Generate Text with Citations" —
    [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627).
  - Menick et al. (2022), "Teaching language models to support
    answers with verified quotes" (GopherCite) —
    [arxiv.org/abs/2203.11147](https://arxiv.org/abs/2203.11147).
- Few-shot prompting shape:
  - Brown et al. (2020), "Language Models are Few-Shot Learners"
    (GPT-3) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165).
- Upstream handoff:
  - mod-108 exercise-01 (template and canonical refusal string)
    — `../exercise-01-inline-and-structured-citation-instruction-templates/SOLUTION.md`.
  - mod-107 exercise-04 (assembler output and legend) —
    `../../mod-107-reranking-and-context-assembly/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors/SOLUTION.md`.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML09 Output Integrity. Confident hallucination on
    unretrieved facts is an output-integrity failure.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques involving model output manipulation (e.g.,
    AML.T0043) motivate the empty-context / adjacent-topic
    tests here.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure* function calls for measurable, repeatable
    evaluation of risky behaviours; false-answer and
    false-refusal rates are the measurable pair for this one.
