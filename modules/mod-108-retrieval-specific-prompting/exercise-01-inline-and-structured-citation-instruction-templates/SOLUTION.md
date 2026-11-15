# Exercise 01 — Solution: Inline and Structured Citation Instruction Templates

This is a reference solution for
`exercise-01-inline-and-structured-citation-instruction-templates.md`.
It is a **design + implementation** exercise. The learner writes two
prompt templates — one that elicits *inline* citations (`[chunk:id]`
markers interleaved with the prose) and one that elicits *structured*
citations (a machine-parseable object alongside the answer) — then
documents the trade-offs and picks which to ship for the mod-108
baseline. Both templates consume the `context_block` and `legend`
emitted by mod-107's assembler (see
`../../mod-107-reranking-and-context-assembly/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors/SOLUTION.md`).

The teaching point: a citation is a **contract between the retrieval
system, the model, and the reviewer**. It has to survive the model's
rephrasing, be parseable by a verifier (exercise-04), and be
displayable to the user. If any one of those three is missing, the
citation does not do its job. The instruction template is where that
contract is written, and the schema is where it is enforced.

## 1. Solution overview

The deliverable has five parts:

1. **`prompts/inline_citations.md`** — a system-prompt template that
   produces inline `[chunk:<citation_id>]` markers after each
   supported sentence, driven by the assembler's anchors.
2. **`prompts/structured_citations.md`** — a system-prompt template
   that forces the model to emit a JSON object with
   `answer` plus a `citations[]` array that references
   `citation_id`s from the legend. Paired with a JSON Schema the API
   can enforce (OpenAI Structured Outputs / Anthropic tool call
   schemas / local-model JSON-mode).
3. **`schemas/citation_response.schema.json`** — the JSON Schema for
   the structured template. This is what later modules
   (exercise-04, mod-109) consume.
4. **`notes/decision.md`** — a one-page decision memo comparing the
   two templates across six axes (verifier-friendliness, user
   display, token overhead, failure-mode visibility, model
   support, and latency) with a recommendation for the mod-108
   baseline and a trigger to switch.
5. **`tests/test_templates.py`** — golden tests that exercise both
   templates on a minimal context block and assert the output
   shape, not the specific wording.

The exercise is scoped to *writing and validating the templates*,
not to retraining the model. If the templates produce
schema-conformant output on the gold cases, the exercise passes.
Hallucination and refusal eval live in exercise-02; verification
lives in exercise-04.

## 2. Worked answer or implementation

### 2.1 The two templates in one paragraph

Inline citations place `[chunk:<citation_id>]` markers inside the
answer text, one per supported claim, pointing at the `citation_id`
that the mod-107 assembler emitted in the legend. Structured
citations keep the prose clean and emit a separate JSON array of
`{claim, citation_ids}` objects. Inline is cheaper to render in a
chat UI (one block of text) and easier for a human to scan;
structured is cheaper to *verify* in code (exercise-04 can parse
the citations list without regex) and easier to audit at scale.
Both are grounded in the same assembler output; the difference is
where the citation lives in the response, not what it points to.

### 2.2 `prompts/inline_citations.md` — the inline template

````markdown
# System

You are a retrieval-grounded assistant. Answer the user's question
using ONLY the information in the `<context>` block below. The
context is organised as independent chunks, each preceded by a
citation anchor of the form `[chunk:<citation_id>]`.

## Rules

1. Every factual sentence in your answer MUST be followed by one or
   more citation markers of the form `[chunk:<citation_id>]`,
   referencing the chunk(s) that support it. If a single sentence
   is supported by multiple chunks, concatenate markers:
   `... retention is 180 days [chunk:policy-retention-v3]
   [chunk:changelog-2026-06].`
2. Use ONLY the `citation_id`s that appear in the `<context>` block
   verbatim. Do not invent, abbreviate, or renumber them.
3. If the context does not contain enough information to answer
   the question, reply with exactly:
   `I cannot answer this from the retrieved context.`
   Do NOT guess. Do NOT draw on prior knowledge.
4. If two chunks contradict each other, surface the contradiction
   explicitly and cite both. Example: "The policy states 90 days
   [chunk:policy-v2] but the 2026-06 changelog updates this to 180
   days [chunk:changelog-2026-06]."
5. Do not include a "Sources:" footer. The inline markers are the
   only citation surface.

## Context

<context>
{{context_block}}
</context>

## Legend (for your reference; do not quote back)

{{legend_as_markdown}}

# User

{{question}}
````

**Why this shape:**

- **Rule 1 forces one marker per factual sentence.** A prompt that
  says "cite your sources" without a per-claim rule yields a
  single trailing citation block that cannot be verified
  claim-by-claim. The verifier in exercise-04 operates on
  `(sentence, cited_ids)` pairs; the template has to produce
  that shape.
- **Rule 2 anchors the output to the legend.** Instructing the
  model to use ids *verbatim* is what makes exercise-04's exact-
  match verifier work. Without it the model will rewrite
  `policy-retention-v3` as `Policy Retention v3` and every
  citation will fail validation.
- **Rule 3 pre-empts hallucination under empty context.** This
  is the refusal contract exercise-02 trains and evaluates;
  the inline template declares the exact refusal string so
  the eval harness can string-match.
- **Rule 4 names contradiction handling.** Without it, the model
  silently picks one chunk. Exercise-03 depends on this clause
  firing.
- **Rule 5 forbids a trailing "Sources:" block.** Models trained
  on web data routinely append a bibliography. Two citation
  surfaces (inline + footer) create ambiguity about which the
  verifier should trust and double the token cost.

### 2.3 `prompts/structured_citations.md` — the structured template

````markdown
# System

You are a retrieval-grounded assistant. Answer the user's question
using ONLY the information in the `<context>` block below. You MUST
respond by calling the `grounded_answer` tool with a JSON payload
conforming to the schema the client provided. The payload has
three top-level fields:

- `answer` (string): the user-facing prose answer. Do NOT include
  inline `[chunk:...]` markers in `answer`. Keep it readable.
- `citations` (array): one entry per factual claim in `answer`.
  Each entry is `{claim, citation_ids}` where `claim` is a short
  verbatim or near-verbatim quote from your own `answer` and
  `citation_ids` is a non-empty list of `citation_id` strings that
  appear in the `<context>` block.
- `insufficient_context` (boolean): true if the context did not
  support an answer. When true, `answer` MUST be the refusal
  string `"I cannot answer this from the retrieved context."`
  and `citations` MUST be the empty array.

## Rules

1. Every claim in `citations[*].claim` must be a substring (case-
   insensitive, whitespace-normalised) of `answer`. The claim
   identifies which sentence the citation supports.
2. Use ONLY `citation_id` values that appear verbatim in the
   `<context>` block.
3. If the context does not support an answer, set
   `insufficient_context: true` and emit the refusal string.
4. If two chunks contradict each other, emit a single claim that
   names the contradiction and list BOTH `citation_id`s in
   `citation_ids`.
5. Do not invent `citation_id`s. Do not merge them (e.g., do not
   emit `"policy-v2+changelog-2026-06"`). One id per element of
   `citation_ids`.

## Context

<context>
{{context_block}}
</context>

## Legend

{{legend_as_markdown}}

# User

{{question}}
````

**Why this shape:**

- **The tool-call framing (`grounded_answer`) lets the API enforce
  the JSON Schema** (OpenAI Structured Outputs, Anthropic tool-
  use schemas, or local JSON-mode). The failure mode becomes
  "schema violation" at the API, not "verifier trips in
  production."
- **`claim` is a substring of `answer`.** This is the critical
  anchor that exercise-04's verifier uses to pair a citation with
  the sentence it supports. If the model paraphrased `claim`,
  the verifier could not locate the sentence. Rule 1 pushes the
  entire burden onto the model, where it is cheap.
- **`insufficient_context: true` + a refusal string.** Two signals
  for the same state is intentional: the boolean is cheap for
  machines, the string is the user-facing surface. Exercise-02's
  eval only has to check one or the other; the two must agree.
- **`citation_ids` is a non-empty list, not a single id.** A
  single claim can be supported by multiple chunks; forcing a
  list avoids a second schema for that case and makes
  contradictions encodable.

### 2.4 `schemas/citation_response.schema.json` — the JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "GroundedAnswer",
  "type": "object",
  "additionalProperties": false,
  "required": ["answer", "citations", "insufficient_context"],
  "properties": {
    "answer": {
      "type": "string",
      "minLength": 1
    },
    "insufficient_context": {
      "type": "boolean"
    },
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["claim", "citation_ids"],
        "properties": {
          "claim": {
            "type": "string",
            "minLength": 1
          },
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
      "if": {
        "properties": {"insufficient_context": {"const": true}}
      },
      "then": {
        "properties": {
          "answer": {
            "const": "I cannot answer this from the retrieved context."
          },
          "citations": {"maxItems": 0}
        }
      }
    }
  ]
}
```

**Choices and why:**

- **`additionalProperties: false`** at every object level. Lax
  schemas accept misspelled fields silently; strict schemas fail
  loudly at the API boundary, which is cheaper than failing at
  the verifier.
- **`citation_ids` items are regex-constrained.** The pattern
  mirrors the slug format emitted by mod-107's
  `attach_anchors` (slugified `source_id`, optional `#N`
  disambiguator). A stray space or quote in a `citation_id`
  fails validation up front.
- **`allOf` with an `if/then`** encodes the "refusal ⇒ specific
  answer string, empty citations" invariant in the schema
  itself. The API refuses malformed refusals; the verifier does
  not have to double-check.

### 2.5 Example outputs

**Inline template, well-grounded question.**

Question: "What is the retention period for export logs?"

```
Export logs are retained for 180 days for customer-tier accounts
[chunk:policy-retention-v3]. Internal accounts retain for 90 days
unless Policy §4.2 specifies otherwise [chunk:policy-retention-v3].
The 180-day period was introduced in June 2026, up from the
previous 90 days [chunk:changelog-2026-06].
```

**Structured template, same question.**

```json
{
  "insufficient_context": false,
  "answer": "Export logs are retained for 180 days for customer-tier accounts. Internal accounts retain for 90 days unless Policy §4.2 specifies otherwise. The 180-day period was introduced in June 2026, up from the previous 90 days.",
  "citations": [
    {
      "claim": "Export logs are retained for 180 days for customer-tier accounts.",
      "citation_ids": ["policy-retention-v3"]
    },
    {
      "claim": "Internal accounts retain for 90 days unless Policy §4.2 specifies otherwise.",
      "citation_ids": ["policy-retention-v3"]
    },
    {
      "claim": "The 180-day period was introduced in June 2026, up from the previous 90 days.",
      "citation_ids": ["changelog-2026-06"]
    }
  ]
}
```

**Either template, empty-context question.**

Inline:

```
I cannot answer this from the retrieved context.
```

Structured:

```json
{
  "insufficient_context": true,
  "answer": "I cannot answer this from the retrieved context.",
  "citations": []
}
```

### 2.6 `notes/decision.md` — one-page recommendation

| Axis | Inline template | Structured template |
|---|---|---|
| Verifier-friendliness | Regex over the answer text; brittle on multi-marker sentences and nested punctuation. | Deterministic; `citations[]` parsed directly, no regex. |
| User display | Renders cleanly in chat UIs as a single block. Footnote renderer is straightforward. | Requires a second pass to interleave claims and citations for display, or two UI surfaces. |
| Token overhead | ~5–10% over an uncited answer (one marker per sentence). | ~25–50% over an uncited answer (JSON overhead + repeated `claim` substrings). |
| Failure-mode visibility | A dropped citation looks like a plain sentence — easy to miss. | A missing `claim`/`citation_id` is a schema violation — visible at the API boundary. |
| Model support | Every chat model. No special API required. | Requires JSON-mode / Structured Outputs / tool-use on the generator; smaller OSS models struggle. |
| Latency | One generation pass. | One generation pass; token count is higher so TTFT is comparable but total latency is 10–20% higher. |

**Recommendation for the mod-108 baseline:** ship *both*, select by
flow:

- **Interactive chat UI (consumer, support bots):** inline. The
  render path is one block of text, the token overhead is small,
  and exercise-04's verifier can run on the regex-extracted
  markers as a post-generation audit.
- **Compliance-reviewed, enterprise, or evaluated-at-scale
  flows:** structured. The schema is the contract, failures are
  visible at the API boundary, and the citation list is
  trivially joined against the mod-107 legend in SQL.

**Trigger to switch** from inline to structured: when the
exercise-04 verifier's false-negative rate on inline-marker
parsing exceeds the acceptable threshold (owner sets this per
deployment; a common default is 1–2% of responses), the inline
regex is lying to you — move to structured.

### 2.7 `tests/test_templates.py` — the test cases

```python
"""Golden tests for the mod-108 citation templates.

These do not call a live model. Instead they exercise the template
renderer and the schema validator. A full generation eval lives
in exercise-02 (refusal) and exercise-04 (verifier)."""
import json
import re
from pathlib import Path

import jsonschema
import pytest

SCHEMA = json.loads(
    Path("schemas/citation_response.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)

def _validate(payload):
    errors = sorted(VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    return [e.message for e in errors]

def test_structured_valid_grounded_answer():
    payload = {
        "insufficient_context": False,
        "answer": "Export logs are retained for 180 days.",
        "citations": [
            {
                "claim": "Export logs are retained for 180 days.",
                "citation_ids": ["policy-retention-v3"],
            }
        ],
    }
    assert _validate(payload) == []

def test_structured_refusal_requires_exact_string():
    payload = {
        "insufficient_context": True,
        "answer": "Sorry, I don't know.",
        "citations": [],
    }
    errors = _validate(payload)
    assert errors, "refusal with wrong answer string must fail"

def test_structured_refusal_requires_empty_citations():
    payload = {
        "insufficient_context": True,
        "answer": "I cannot answer this from the retrieved context.",
        "citations": [
            {"claim": "x", "citation_ids": ["any-id"]}
        ],
    }
    errors = _validate(payload)
    assert errors, "refusal with non-empty citations must fail"

def test_structured_rejects_empty_citation_ids():
    payload = {
        "insufficient_context": False,
        "answer": "Export logs are retained.",
        "citations": [
            {"claim": "Export logs are retained.", "citation_ids": []}
        ],
    }
    errors = _validate(payload)
    assert errors, "citations entries must have at least one id"

def test_structured_rejects_malformed_citation_id():
    payload = {
        "insufficient_context": False,
        "answer": "x",
        "citations": [
            {"claim": "x", "citation_ids": ["has space"]}
        ],
    }
    errors = _validate(payload)
    assert errors, "citation_ids with spaces must fail regex"

# Inline template is validated by parse-back: every marker extracted
# from the response must refer to an id in the legend.

MARKER = re.compile(r"\[chunk:([A-Za-z0-9][A-Za-z0-9._#-]*)\]")

def extract_markers(text: str) -> list[str]:
    return MARKER.findall(text)

def test_inline_all_markers_resolve_to_legend():
    legend_ids = {"policy-retention-v3", "changelog-2026-06"}
    response = (
        "Export logs are retained for 180 days for customer-tier accounts "
        "[chunk:policy-retention-v3]. The 180-day period was introduced in "
        "June 2026 [chunk:changelog-2026-06]."
    )
    assert set(extract_markers(response)).issubset(legend_ids)

def test_inline_rejects_invented_marker():
    legend_ids = {"policy-retention-v3"}
    response = "Retention is 180 days [chunk:policy-v99]."
    assert not set(extract_markers(response)).issubset(legend_ids)

def test_inline_refusal_string_exact():
    refusal = "I cannot answer this from the retrieved context."
    response = refusal
    assert response == refusal
    assert extract_markers(response) == []
```

Running these should pass against the reference schema and
templates. A learner implementation that fails any test has a
contract break the downstream exercises will trip on.

## 3. Validation steps

1. The JSON Schema loads under Draft 2020-12 and rejects malformed
   payloads (`additionalProperties`, missing fields, empty
   `citation_ids`, non-slug `citation_id`).
2. For the structured template, the `insufficient_context: true`
   invariant (specific `answer` string, empty `citations`) is
   enforced by schema, not by downstream code.
3. For the inline template, every `[chunk:<id>]` marker extracted
   from the response is a member of the legend's `citation_id`s.
4. Both templates produce the exact refusal string
   `"I cannot answer this from the retrieved context."` when
   `context_block` is empty.
5. Neither template appends a trailing "Sources:" or
   "References:" block — one citation surface only.
6. The template renderer interpolates the mod-107 assembler's
   `context_block` and `legend` without escaping artefacts
   (no stray `{{` tokens survive).
7. Golden tests pass (`pytest -q tests/test_templates.py`).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Two templates delivered | Inline and structured are both present and pointed at the same assembler output. |
| Schema-enforced structured | The structured template is backed by a JSON Schema, and the schema enforces the refusal invariant. |
| Legend-grounded ids | Both templates instruct the model to use `citation_id`s **verbatim** from the legend. No reformatting, no abbreviations, no numeric renumbering. |
| Refusal string | A single canonical refusal string is declared in both templates. |
| Contradiction clause | Both templates include an explicit instruction on how to cite two conflicting chunks (both ids, named contradiction). |
| No double-citation surface | Neither template produces both inline markers and a trailing bibliography. |
| Decision memo | The `notes/decision.md` names one pick per flow and names a measurable trigger to switch. |
| Tests | Golden tests cover schema conformance, refusal invariants, and marker-to-legend resolution. |
| Model support realism | Structured template is used only on models that enforce the JSON schema; the memo names this constraint. |

Fail-fast smells:

- The structured template specifies JSON in prose only, with no
  schema. Models drift.
- The refusal string differs between templates. Exercise-02's
  eval cannot string-match across two forms.
- `citation_ids` is a single string in the schema. Contradictions
  are unencodable; the template has to concatenate ids with `+`,
  which breaks the verifier.
- The inline template tells the model to "cite your sources" with
  no per-sentence rule. A trailing bibliography is what the
  model will emit, defeating per-claim verification.
- The templates reference the mod-107 anchor format but not the
  legend. The model invents ids because nothing anchors it.

## 5. Common mistakes

- **"Cite your sources" without a per-claim rule.** The model
  produces one trailing citation block; the verifier cannot pair
  it with individual claims.
- **Letting the model renumber citations.** `[1]`, `[2]` style
  loses provenance — `[1]` in one response is not `[1]` in
  another. Always pass through the assembler's stable
  `citation_id`s.
- **Describing JSON in prose instead of enforcing a schema.**
  Without a schema the model emits "nearly JSON" that trips the
  parser on the 1% of edge cases.
- **Two citation surfaces.** Instructing the model to cite inline
  *and* list sources at the bottom doubles the cost and creates
  disagreement about which surface is canonical.
- **Soft refusal strings.** "I'm not sure" is unparseable.
  Pick a single exact phrase and enforce it.
- **No contradiction clause.** The model silently picks the
  highest-ranked chunk. Exercise-03 will reveal this as a bug,
  but exercise-01 is where it is prevented.
- **Allowing paraphrased `citation_id`s.** `Policy Retention v3`
  ≠ `policy-retention-v3`. The verifier fails; the model is
  "right" in spirit. Instruct verbatim.
- **Over-permissive JSON Schema.** Omitting
  `additionalProperties: false` lets the model add fields that
  look valid but break downstream parsers silently.
- **Ignoring model support.** JSON-mode is not free on every
  model; shipping the structured template to a model that does
  not enforce schemas produces drift under load.

## 6. References

- Grounded-answer / citation generation:
  - Gao et al. (2023), "Enabling Large Language Models to
    Generate Text with Citations" —
    [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627).
  - Menick et al. (2022), "Teaching language models to support
    answers with verified quotes" (GopherCite) —
    [arxiv.org/abs/2203.11147](https://arxiv.org/abs/2203.11147).
- Structured output in production:
  - OpenAI Structured Outputs — API documentation.
  - Anthropic tool-use / tool schemas — API documentation.
  - JSON Schema Draft 2020-12 —
    [json-schema.org/specification](https://json-schema.org/specification).
- Upstream handoff:
  - mod-107 exercise-04 (assembler legend and anchors) —
    `../../mod-107-reranking-and-context-assembly/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors/SOLUTION.md`.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML02 Data Poisoning / retrieval-side
    prompt-injection surfaces, which an unverified citation
    conceals.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques around model output manipulation motivate
    schema-enforced citations.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Govern / Measure* functions call for traceability of
    model outputs to their evidence, which is what the citation
    contract delivers.
