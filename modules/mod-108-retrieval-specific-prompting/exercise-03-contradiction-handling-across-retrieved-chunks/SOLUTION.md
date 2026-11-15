# Exercise 03 — Solution: Contradiction Handling Across Retrieved Chunks

This is a reference solution for
`exercise-03-contradiction-handling-across-retrieved-chunks.md`. It
is a **design + implementation** exercise. The learner extends the
mod-108 template from exercise-01 with explicit contradiction
behaviour, builds an eval set of retrieved-context pairs that
*disagree*, and chooses a resolution policy that fits the mod-108
baseline. Two of the resolution policies are implemented as prompt
behaviours; one is implemented as pre-generation metadata gating in
the assembler layer.

The teaching point: *two chunks that contradict each other are a
retrieval-quality signal, not a bug in the generator*. The model's
job is to **surface** the contradiction, not to silently pick a
winner. Picking a winner — even correctly — hides a retrieval or
freshness failure that will bite again tomorrow. The one case where
the system *should* pick a winner is when it has out-of-band
metadata (timestamp, version, authority tier) that unambiguously
ranks the sources; that resolution belongs in the assembler, not in
the prompt.

## 1. Solution overview

The deliverable has five parts:

1. **`prompts/contradiction_rules.md`** — the contradiction clause
   appended to the exercise-01 template's rules section. Three
   sub-rules: name the contradiction, cite all conflicting sources,
   emit a `contradiction: true` flag (structured template) or a
   leading `[contradiction]` marker (inline template).
2. **`prompts/contradiction_shots.md`** — three few-shot examples
   covering the three contradiction classes: version-vs-version
   (same source, two revisions), source-vs-source (two different
   authors disagree), and fact-vs-exception (general rule vs
   regional / role-based exception that technically does not
   contradict but reads like one).
3. **`eval/contradiction_eval_set.jsonl`** — a 30-row eval with
   gold-labelled contradiction-present vs contradiction-absent
   partitions (with four sub-types on the present side).
4. **`eval/run_eval.py`** — the harness that scores each response
   on three axes: did it name a contradiction, did it cite both
   sources, did it emit the schema flag (structured) or marker
   (inline).
5. **`notes/policy.md`** — the decision memo comparing the three
   resolution policies (**surface**, **pre-generation gate**,
   **model-side resolve**) and recommending which to ship for the
   mod-108 baseline.

The exercise is scoped to the **model's response to contradictory
context**. Resolving contradictions at retrieval time (deduplicating,
re-ranking by recency, ACL scoping) belongs to mod-106/107/111.

## 2. Worked answer or implementation

### 2.1 Three resolution policies in one paragraph each

- **Surface (model-side).** The generator names the contradiction
  and cites both sources; the application decides what to do.
  Default for most consumer and ambiguous-authority flows. Cheap,
  transparent, teaches the user to question stale docs.
- **Pre-generation metadata gate (assembler-side).** The
  assembler compares chunk timestamps / version numbers /
  authority tiers *before* the model sees them; it drops the
  loser and annotates the legend. The model sees one chunk and
  cannot contradict itself. Correct for policy-vs-changelog
  cases where the newer chunk definitionally supersedes.
- **Model-side resolve.** The prompt tells the model to pick the
  "authoritative" source and answer from it. **Do not ship this
  policy** — it moves a governance decision into the model's
  weights, gives no audit trail, and depends on pretraining
  biases about which sources "sound authoritative."

### 2.2 `prompts/contradiction_rules.md` — the clause

Appended to the exercise-01 template's Rules section:

````markdown
## Rules — contradiction handling

If two or more chunks in the context make claims that cannot both
be true (directly conflicting numbers, dates, policies, or facts),
you MUST:

1. **Name the contradiction.** State explicitly that the sources
   disagree, in the sentence where the conflicting fact appears.
   Example: "The retention period is reported inconsistently
   across the retrieved sources."
2. **Cite every conflicting source.** List the `[chunk:<id>]`
   markers (inline template) or all `citation_id`s in a single
   `citation_ids` array (structured template) for every claim
   that is contradicted.
3. **Do NOT pick a winner on your own.** You do not have the
   authority or metadata to decide which source supersedes.
4. **Indicate the contradiction in the response contract:**
   - Inline template: prepend `[contradiction]` to the first
     sentence that names the disagreement.
   - Structured template: set `contradiction: true` at the top
     level of the response object (schema update in §2.5), and
     include each conflicting fact as a single `citations[]`
     entry with `citation_ids` listing ALL conflicting source ids.
5. **If a chunk clearly supersedes another** via an explicit
   marker IN the context (e.g., "Replaces §4.2 as of 2026-06-12"),
   you may state the supersession but still cite both. The
   supersession must be visible in the chunk text, not inferred.

If the chunks agree, do not invoke the contradiction behaviour —
answer as in exercise-01 / exercise-02.
````

**Why this clause:**

- **Rule 1 forces the model to name the conflict.** Without it,
  the model concatenates both claims in adjacent sentences and
  lets the user figure it out. Users miss it; auditors cannot
  grep for it.
- **Rule 2 keeps provenance intact.** The user (or verifier) can
  go look at both sources and judge.
- **Rule 3 is the governance rule.** "Pick the one that sounds
  more authoritative" is hallucination with extra steps.
- **Rule 4 gives downstream code a hook.** A pre-generation
  gate missed something; a monitor needs to see it; an audit
  needs to count it. The flag/marker is what those consumers
  read.
- **Rule 5 permits the narrow supersession case** — the chunk
  *itself* says it supersedes. This is explicit evidence in the
  retrieved context, not model inference.

### 2.3 `prompts/contradiction_shots.md` — the three shots

````markdown
### Shot A — version-vs-version contradiction

<context>
[chunk:policy-retention-v2]
Export log retention is 90 days.

[chunk:policy-retention-v3]
Export log retention is 180 days.
</context>

Question: What is the retention period for export logs?

Answer (inline):
[contradiction] The retrieved sources disagree on the export log
retention period: one source states 90 days [chunk:policy-retention-v2]
while another states 180 days [chunk:policy-retention-v3]. The
context does not indicate which is current.

Answer (structured):
{
  "insufficient_context": false,
  "contradiction": true,
  "answer": "The retrieved sources disagree on the export log retention period: one source states 90 days while another states 180 days. The context does not indicate which is current.",
  "citations": [
    {
      "claim": "The retrieved sources disagree on the export log retention period: one source states 90 days while another states 180 days.",
      "citation_ids": ["policy-retention-v2", "policy-retention-v3"]
    }
  ]
}

### Shot B — source-vs-source contradiction

<context>
[chunk:hr-handbook-2026]
Parental leave is 12 weeks paid for all employees.

[chunk:benefits-site-2026]
Parental leave is 16 weeks paid for salaried employees.
</context>

Question: How much paid parental leave does the company offer?

Answer (inline):
[contradiction] The retrieved sources disagree on paid parental leave:
the HR handbook states 12 weeks for all employees
[chunk:hr-handbook-2026], while the benefits site states 16 weeks
for salaried employees [chunk:benefits-site-2026].

### Shot C — fact-vs-exception (NOT a contradiction)

<context>
[chunk:policy-overtime-v2]
Overtime is paid at 1.5× the base rate for hours worked beyond 40
in a week.

[chunk:regional-exceptions-ca]
In California, overtime additionally applies after 8 hours in a
single day.
</context>

Question: When does overtime apply?

Answer (inline):
Overtime applies at 1.5× the base rate for hours worked beyond 40
in a week [chunk:policy-overtime-v2]. California adds an additional
rule: overtime also applies after 8 hours in a single day
[chunk:regional-exceptions-ca].
````

**Why these three shots:**

- **Shot A — version contradiction**, with no explicit
  supersession marker. This is the most common real case
  (changelog not retrieved) and the one the model is most
  tempted to resolve on its own.
- **Shot B — source contradiction** between two concurrent
  sources (handbook vs benefits site). The model must not pick.
- **Shot C — fact-vs-exception** is the negative shot that
  prevents over-firing. A rule and its region-specific
  exception are not a contradiction. Without this shot the
  model will label every rule-plus-exception as a conflict.

### 2.4 `eval/contradiction_eval_set.jsonl` — the eval set

30 rows. Partitions:

- **`contradiction`** (20 rows), four sub-types:
  - **version** (6 rows): two chunks from the same source at
    different versions; no supersession marker.
  - **source** (6 rows): two different sources with conflicting
    claims on the same question.
  - **superseded** (4 rows): two chunks, one of which contains
    an explicit "supersedes" marker; the model may call this
    out while citing both.
  - **partial** (4 rows): one chunk makes a strong claim; the
    other makes a weaker, qualified claim that *appears* to
    conflict but is a qualification. Hard negative for Rule 5.
- **`no_contradiction`** (10 rows):
  - 5 rule-plus-exception pairs (like Shot C).
  - 5 complementary pairs (two chunks on different aspects of
    the same answer).

Fields:

| field | type | description |
|---|---|---|
| `id` | string | stable id |
| `partition` | string | `"contradiction"` or `"no_contradiction"` |
| `sub_type` | string | one of the sub-types above |
| `question` | string | user-facing question |
| `context_block` | string | mod-107 assembler output |
| `legend` | array | `[{citation_id, source_id}, ...]` |
| `gold_contradiction_ids` | array | the citation_ids involved in the contradiction (empty when `no_contradiction`) |

### 2.5 Structured-template schema update

The exercise-01 schema is extended with a top-level
`contradiction: boolean`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GroundedAnswer",
  "type": "object",
  "additionalProperties": false,
  "required": ["answer", "citations", "insufficient_context", "contradiction"],
  "properties": {
    "answer":               {"type": "string",  "minLength": 1},
    "insufficient_context": {"type": "boolean"},
    "contradiction":        {"type": "boolean"},
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["claim", "citation_ids"],
        "properties": {
          "claim":        {"type": "string", "minLength": 1},
          "citation_ids": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "string",
              "pattern": "^[A-Za-z0-9][A-Za-z0-9._#-]*$"
            }
          }
        }
      }
    }
  },
  "allOf": [
    {
      "if":   {"properties": {"insufficient_context": {"const": true}}},
      "then": {
        "properties": {
          "answer":        {"const": "I cannot answer this from the retrieved context."},
          "citations":     {"maxItems": 0},
          "contradiction": {"const": false}
        }
      }
    },
    {
      "if":   {"properties": {"contradiction": {"const": true}}},
      "then": {
        "properties": {
          "citations": {
            "contains": {
              "properties": {"citation_ids": {"minItems": 2}}
            }
          }
        }
      }
    }
  ]
}
```

**Why this schema shape:**

- `contradiction` is a required boolean at the top level. If a
  contradiction fires, downstream monitors do not have to
  regex the answer.
- When `contradiction: true`, at least one `citations[]` entry
  must have ≥ 2 ids — the conflict is encoded in the citation
  structure, not just the prose.
- `insufficient_context: true` forces `contradiction: false`.
  The two states are mutually exclusive by schema.

### 2.6 `eval/run_eval.py` — the harness

```python
"""Contradiction eval harness for mod-108 exercise-03.

Scores each response on three axes:
- `named`: was a contradiction named in the prose?
- `cited`: were ALL gold contradiction ids cited in the same claim?
- `flagged`: did the response set the structured flag (or inline marker)?
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MARKER = re.compile(r"\[chunk:([A-Za-z0-9][A-Za-z0-9._#-]*)\]")
INLINE_FLAG = re.compile(r"^\s*\[contradiction\]", re.IGNORECASE)
NAMING_KEYWORDS = (
    "disagree", "conflict", "contradict",
    "inconsistent", "different", "differ",
)

@dataclass
class EvalRow:
    id: str
    partition: str
    sub_type: str
    question: str
    context_block: str
    legend: list[dict]
    gold_contradiction_ids: list[str]

def load_eval(path: str) -> list[EvalRow]:
    return [EvalRow(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]

def score_inline(row: EvalRow, response: str) -> dict:
    cited_ids = set(MARKER.findall(response))
    named = any(kw in response.lower() for kw in NAMING_KEYWORDS)
    flagged = bool(INLINE_FLAG.search(response))
    gold = set(row.gold_contradiction_ids)
    cited_gold = gold.issubset(cited_ids) if gold else (not cited_ids & set(x["citation_id"] for x in row.legend if x.get("citation_id") not in gold))
    return {"named": named, "cited": bool(cited_ids & gold) if gold else True, "flagged": flagged, "cited_all_gold": cited_gold}

def score_structured(row: EvalRow, obj: dict) -> dict:
    gold = set(row.gold_contradiction_ids)
    flagged = bool(obj.get("contradiction"))
    cited_all_gold = False
    for c in obj.get("citations", []):
        if gold and gold.issubset(set(c.get("citation_ids", []))):
            cited_all_gold = True
            break
    named = any(kw in obj.get("answer", "").lower() for kw in NAMING_KEYWORDS)
    return {"named": named, "cited_all_gold": cited_all_gold, "flagged": flagged}

def verdict(row: EvalRow, scored: dict) -> str:
    want = row.partition == "contradiction"
    got = scored["flagged"] or scored["named"]
    if want and got and scored.get("cited_all_gold"):
        return "correct"
    if want and not got:
        return "missed_contradiction"
    if not want and got:
        return "false_contradiction"
    if want and got and not scored.get("cited_all_gold"):
        return "named_but_undercited"
    return "correct"

def run_inline(eval_path: str, generate: Callable[[str, str], str]) -> dict:
    rows = load_eval(eval_path)
    per_row = []
    for r in rows:
        resp = generate(r.context_block, r.question)
        s = score_inline(r, resp)
        per_row.append({"id": r.id, "sub_type": r.sub_type, "verdict": verdict(r, s), **s})
    return _summarise(per_row)

def _summarise(per_row):
    totals = {"correct": 0, "missed_contradiction": 0, "false_contradiction": 0, "named_but_undercited": 0}
    for r in per_row:
        totals[r["verdict"]] += 1
    return {"n": len(per_row), **totals, "detail": per_row}
```

**Choices and why:**

- **`named` is keyword-matched over a small list of six words.**
  We deliberately avoid a cleverer classifier: a model that
  emits a correct answer without any of
  `disagree/conflict/contradict/inconsistent/different/differ`
  is not naming the contradiction in a way a human reviewer
  would recognise.
- **`cited_all_gold` requires ALL gold ids in the SAME `claim`**
  (structured) or at minimum all gold ids anywhere in the
  response (inline). Mentioning both ids in separate paragraphs
  is not "surfacing the contradiction" — it is listing two
  facts.
- **`false_contradiction` on `no_contradiction` partition** is
  the critical false-positive metric. A model that labels every
  rule-plus-exception as a contradiction is useless; this metric
  catches it.
- **Four verdicts, not two.** "Named but not cited" and
  "flagged but undercited" are specific failure modes the
  triage guide maps to specific fixes.

### 2.7 `notes/policy.md` — the resolution policy memo

| Axis | Surface (model-side) | Pre-gen gate (assembler) | Model-side resolve |
|---|---|---|---|
| Transparency | Full — user sees both sources. | Full — legend records what was gated. | None — no audit trail. |
| Latency | Nominal — one generation call. | Negligible — assembler compares metadata. | Nominal. |
| Correctness upper bound | Model can still be wrong about *whether* it's a contradiction, but will not silently pick a loser. | Depends on metadata quality. If timestamps wrong, gates wrong. | Depends on pretraining priors. Opaque. |
| Governance posture | Compliant — the model does not resolve policy. | Compliant — the system resolves on documented metadata. | **Non-compliant for regulated flows.** |
| User trust | High — disagreement is visible. | High — one answer, with the losers disclosed. | Low once the first silent wrong pick lands in support. |
| When it works | Default. All open-domain and policy Q&A. | Changelog vs policy; versioned docs with reliable `updated_at`; authority-tiered docs (CEO memo > regional handout). | Never in regulated, compliance, HR, legal, medical, or financial flows. Tolerable in low-stakes, clearly-marked suggestion-only UX. |

**Recommendation for the mod-108 baseline:**

1. **Primary policy: surface.** The model names the
   contradiction and cites both.
2. **Secondary (narrow): pre-generation gate.** When both
   chunks come from the same `source_family` and have a
   monotone version / timestamp field, the assembler keeps
   the newest and annotates the legend's `extra_sources` with
   the dropped version. This reuses the mod-107 assembler
   `extra_sources` field; no new interface.
3. **Do not ship model-side resolve.** It is the wrong place
   for the policy decision.

Trigger to add pre-gen gating: when the exercise-03 eval shows
`false_contradiction` or `named_but_undercited` elevated on the
**version** sub-type and the corpus has a reliable version
field, gate at ingestion rather than teaching the model to
disambiguate.

### 2.8 Example end-to-end run

Query: `"What is the retention period for export logs?"` with
both `policy-retention-v2` and `policy-retention-v3` retrieved.

Response (inline, surface policy):

```
[contradiction] The retrieved sources disagree on the export log
retention period: one source states 90 days [chunk:policy-retention-v2]
while another states 180 days [chunk:policy-retention-v3]. The context
does not indicate which is current.
```

Response (structured, surface policy):

```json
{
  "insufficient_context": false,
  "contradiction": true,
  "answer": "The retrieved sources disagree on the export log retention period: one source states 90 days while another states 180 days. The context does not indicate which is current.",
  "citations": [
    {
      "claim": "The retrieved sources disagree on the export log retention period: one source states 90 days while another states 180 days.",
      "citation_ids": ["policy-retention-v2", "policy-retention-v3"]
    }
  ]
}
```

With the pre-generation gate: the assembler keeps
`policy-retention-v3` and annotates `extra_sources:
["policy-retention-v2"]`. The model sees one chunk and answers
`180 days` with a single citation — no contradiction flag.

## 3. Validation steps

1. The updated JSON Schema validates: the `contradiction: true`
   → "at least one `citations[]` entry has ≥ 2 ids" invariant
   is enforced.
2. `insufficient_context: true` → `contradiction: false` is
   mutually exclusive by schema.
3. The eval set loads as JSONL; partition sizes are 20 /10;
   each sub-type has at least 4 rows.
4. For every `contradiction` row, `gold_contradiction_ids`
   contains ≥ 2 ids and every id appears in the legend.
5. For every `no_contradiction` row, `gold_contradiction_ids`
   is empty.
6. The harness's `false_contradiction` verdict fires on
   rule-plus-exception rows when the model over-labels.
7. The `[contradiction]` inline marker regex finds the marker
   only when it is the first non-whitespace token of the
   response.
8. The pre-gen gate, when enabled, surfaces the dropped version
   in the legend's `extra_sources` field (reusing the mod-107
   assembler contract).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Three policies named | Surface, pre-gen gate, model-side resolve are all present; the memo recommends one with a trigger to switch. |
| Governance stance | Model-side resolve is explicitly **not recommended** for mod-108; the memo names why. |
| Template clause | Contains rules 1–5 above; no "pick the authoritative source" wording. |
| Three shots | Version-vs-version, source-vs-source, and fact-vs-exception (negative). The negative shot is required to prevent over-labelling. |
| Schema update | `contradiction: boolean` is a top-level required field; `true` requires at least one `citations[]` entry with ≥ 2 ids; `insufficient_context` and `contradiction` are mutually exclusive. |
| Eval partitions | 20 / 10 split with four `contradiction` sub-types (version, source, superseded, partial) and two `no_contradiction` sub-types. |
| Harness verdicts | Four verdicts: correct, missed, false, named-but-undercited. Each maps to a fix. |
| Keyword-match naming | Keyword list is explicit and short; no clever classifier that could miss the auditor's grep. |
| Supersession rule | Rule 5 allows naming supersession ONLY when visible in the chunk text, not inferred. |
| Interface reuse | Pre-gen gate reuses mod-107 assembler `extra_sources` field; no new interface. |

Fail-fast smells:

- Template says "if sources disagree, use the most authoritative
  one." Non-compliant and opaque.
- Only two shots (both positive). Model over-labels every
  rule-plus-exception; no negative shot.
- `contradiction` left as an optional JSON field. Models drift
  to omitting it.
- Eval set has no `superseded` or `partial` sub-types.
- Keyword list for `named` is exhaustive ("disagree,
  conflict, contradict, inconsistent, different, differ, vary,
  divergent, …"). Eval becomes fuzzy; triage becomes noisy.
- Pre-gen gate is implemented but the dropped source is not
  recorded in `extra_sources`. Audit trail is broken.
- The memo recommends model-side resolve for "efficiency".

## 5. Common mistakes

- **Telling the model to "pick the authoritative source."**
  Governance decision in model weights, no audit trail, tends
  to follow pretraining biases about which domains "sound
  official."
- **Only version-vs-version shots.** Model handles the trained
  case; falls over on source-vs-source or on fact-vs-exception.
- **No negative shot.** Model labels every rule-plus-exception
  as a contradiction. Produces alert fatigue.
- **Allowing supersession-by-inference.** "Document A sounds
  newer, so it supersedes B" is the model making a decision it
  does not have evidence for. Supersession must be explicit in
  the chunk.
- **Not emitting a machine-readable flag.** Monitors and
  dashboards cannot regex prose reliably.
- **Pre-gen gate without an audit trail.** The dropped version
  must land somewhere the user or auditor can find.
- **`contradiction` optional in the schema.** Models drift to
  `undefined` under load; downstream logic has to default.
- **Confusing contradiction with uncertainty.** A single chunk
  saying "possibly 180 days" is not a contradiction with
  nothing. That is a weak-grounding case, handled by
  exercise-02's refusal contract.
- **Pre-gen gating without trusted metadata.** Timestamps
  without timezones, versions without monotone ordering, or
  authority tiers defined by vibes — all of these produce
  silent miscalls at the assembler.

## 6. References

- Multi-document conflict in grounded QA:
  - Gao et al. (2023), "Enabling Large Language Models to
    Generate Text with Citations" —
    [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627).
- Grounded reasoning over multiple sources:
  - Menick et al. (2022), "Teaching language models to support
    answers with verified quotes" (GopherCite) —
    [arxiv.org/abs/2203.11147](https://arxiv.org/abs/2203.11147).
- JSON Schema:
  - JSON Schema Draft 2020-12 —
    [json-schema.org/specification](https://json-schema.org/specification).
- Upstream handoffs:
  - mod-108 exercise-01 (base template and schema) —
    `../exercise-01-inline-and-structured-citation-instruction-templates/SOLUTION.md`.
  - mod-108 exercise-02 (refusal contract and eval harness
    shape) —
    `../exercise-02-refusal-when-no-context-training-shot-and-eval/SOLUTION.md`.
  - mod-107 exercise-04 (assembler `extra_sources`) —
    `../../mod-107-reranking-and-context-assembly/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors/SOLUTION.md`.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML09 Output Integrity. A silently-chosen
    winner between contradictory sources is an
    output-integrity failure.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques involving output manipulation and
    retrieval-time poisoning motivate explicit
    contradiction-surfacing.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure / Manage* functions call for transparent
    handling of conflicting evidence; surfacing is the
    measurable, manageable default.
