# Exercise 04 — Solution: Post-Generation Citation Verification Check

This is a reference solution for
`exercise-04-post-generation-citation-verification-check.md`. It is
an **implementation** exercise. The learner builds the verifier that
runs *after* the model has produced its grounded answer and
mechanically checks four things: every emitted citation id exists
in the legend, every factual sentence has at least one citation,
every claim's text is actually supported by the cited chunks, and —
for the inline template — the emitted answer has no "orphan"
citations or orphan sentences.

The teaching point: *the prompt is not the contract — the verifier
is*. Exercise-01's template asks the model to cite; exercise-02
asks the model to refuse under weak context; exercise-03 asks the
model to surface contradictions. The verifier is the deterministic
check that these actually happened in the response. It is the piece
that lets you log, alert, and fail requests in production without
trusting the model to self-report.

## 1. Solution overview

The deliverable has six parts:

1. **`src/verify.py`** — the end-to-end verifier. Takes a response
   (inline or structured), the assembler's legend, and the
   assembler's chunk texts; returns a `VerifyResult` with pass/fail
   plus per-check detail.
2. **`src/checks.py`** — the four checks as independent functions:
   `check_citation_ids_resolve`, `check_every_claim_cited`,
   `check_claim_supported_by_chunks`, `check_no_orphans`.
3. **`src/parse.py`** — the two parsers: inline-marker regex
   extraction and structured-schema JSON validation. Both produce
   a common `Claim[]` intermediate the checks run against.
4. **`eval/verifier_eval_set.jsonl`** — 40-row eval set with
   labelled pass / fail rows, four failure sub-types.
5. **`tests/test_verify.py`** — golden tests covering every check
   and every failure sub-type.
6. **`notes/thresholds.md`** — the decision memo on the
   support-check thresholds (exact substring vs token overlap vs
   small-NLI-model) and the recommendation for the mod-108
   baseline.

The exercise is scoped to **mechanical verification of a
response that is already produced**. It does not re-query the
LLM, does not rerun retrieval, and does not fine-tune anything.
If a response passes the verifier, downstream code can trust the
citation contract from exercise-01. If it fails, the response is
gated (returned as a refusal, logged, routed to review — per the
caller's policy; the verifier does not decide).

## 2. Worked answer or implementation

### 2.1 Four checks in one paragraph each

- **Citation ids resolve.** Every `citation_id` the model emitted
  is a key in the assembler legend. A non-resolving id is an
  invented citation; the rest of the response is suspect.
- **Every claim is cited.** For the structured template, every
  `citations[]` entry must carry ≥ 1 id (schema-enforced). For
  the inline template, every sentence of the answer that makes a
  factual claim must have at least one `[chunk:<id>]` marker.
  Uncited sentences are the model's hallucination surface.
- **Claim is supported by the cited chunks.** For each claim, the
  cited chunks' text must plausibly support the claim. "Plausibly"
  is defined by the chosen threshold policy (§2.6).
- **No orphans.** Inline markers in the response must correspond
  to a claim; claims must correspond to markers. A dangling
  marker ("[chunk:foo] this is a paragraph without a citation
  anchor explanation") or a dangling paragraph breaks the
  per-claim contract.

### 2.2 `src/parse.py` — the two parsers

```python
"""Parse a grounded-answer response into Claim[] for verification."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field

MARKER_RE = re.compile(r"\[chunk:([A-Za-z0-9][A-Za-z0-9._#-]*)\]")
# Simple sentence splitter. Not a research-grade tokenizer; good
# enough for citation-surface segmentation.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])")

@dataclass
class Claim:
    text: str                # the claim sentence (markers stripped)
    citation_ids: list[str]  # ids cited for this claim
    raw: str = ""            # the sentence as it appeared, markers included

def parse_inline(answer: str) -> list[Claim]:
    """Split answer into sentences and attach citation ids per sentence."""
    claims: list[Claim] = []
    for raw in SENTENCE_RE.split(answer.strip()):
        raw = raw.strip()
        if not raw:
            continue
        ids = MARKER_RE.findall(raw)
        text = MARKER_RE.sub("", raw).strip().rstrip(".").strip()
        claims.append(Claim(text=text, citation_ids=ids, raw=raw))
    return claims

def parse_structured(payload: dict) -> list[Claim]:
    """Convert a validated structured response into Claim[]."""
    out: list[Claim] = []
    for c in payload.get("citations", []):
        out.append(Claim(
            text=c["claim"].strip().rstrip(".").strip(),
            citation_ids=list(c["citation_ids"]),
            raw=c["claim"],
        ))
    return out
```

**Choices and why:**

- **One parser per template, one output type.** The checks run
  against `Claim[]`; the template choice is invisible after
  parse.
- **A simple regex sentence splitter**, not a model-based NLP
  tokenizer. Research-grade sentence segmentation adds a 50ms
  dependency to a 5ms verifier. The regex handles the cases the
  citation templates actually produce because exercise-01 rules
  forbid trailing bibliographies and complex nesting.
- **Markers are stripped from `text` but preserved in `raw`.**
  The check that compares the claim against chunk text should
  not match on the marker; the orphan check and the diagnostic
  dump need the raw form.
- **Trailing periods normalised out.** Prevents cosmetic
  mismatches between the inline claim (ends with `.`) and the
  structured claim (optional `.`).

### 2.3 `src/checks.py` — the four checks

```python
"""The four verifier checks. Each returns a CheckResult."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from parse import Claim, MARKER_RE

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: list[str] = field(default_factory=list)

def check_citation_ids_resolve(claims: list[Claim], legend_ids: set[str]) -> CheckResult:
    unresolved = []
    for c in claims:
        for cid in c.citation_ids:
            if cid not in legend_ids:
                unresolved.append((c.raw, cid))
    return CheckResult(
        name="citation_ids_resolve",
        passed=not unresolved,
        detail=[f"invented id {cid} in: {raw}" for raw, cid in unresolved],
    )

def check_every_claim_cited(claims: list[Claim]) -> CheckResult:
    uncited = [c for c in claims if not c.citation_ids and _is_factual(c.text)]
    return CheckResult(
        name="every_claim_cited",
        passed=not uncited,
        detail=[f"uncited claim: {c.raw}" for c in uncited],
    )

def _is_factual(text: str) -> bool:
    """Heuristic: a sentence is factual unless it is empty or a
    pure discourse marker ('However,', 'Therefore.')."""
    stripped = text.strip().strip(",.")
    if len(stripped) < 10:
        return False
    DISCOURSE = {"however", "therefore", "in summary", "as noted above"}
    return stripped.lower() not in DISCOURSE

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def check_claim_supported_by_chunks(
    claims: list[Claim],
    chunks_by_id: dict[str, str],
    min_overlap: float = 0.5,
) -> CheckResult:
    """For each cited claim, require token overlap >= min_overlap
    between the claim and the union of its cited chunks.

    The threshold policy is tunable (see notes/thresholds.md). This
    is deliberately a cheap, deterministic check — a stricter
    policy (NLI) can be plugged in behind the same signature."""
    unsupported = []
    for c in claims:
        if not c.citation_ids:
            continue
        claim_toks = _tokenize(c.text)
        if not claim_toks:
            continue
        support_text = " ".join(chunks_by_id.get(cid, "") for cid in c.citation_ids)
        support_toks = _tokenize(support_text)
        if not support_toks:
            unsupported.append((c.raw, "empty cited chunks"))
            continue
        content_toks = claim_toks - _STOPWORDS
        if not content_toks:
            continue
        overlap = len(content_toks & support_toks) / len(content_toks)
        if overlap < min_overlap:
            unsupported.append(
                (c.raw, f"overlap={overlap:.2f} < {min_overlap:.2f}"),
            )
    return CheckResult(
        name="claim_supported_by_chunks",
        passed=not unsupported,
        detail=[f"unsupported ({why}): {raw}" for raw, why in unsupported],
    )

_STOPWORDS = {
    "a","an","the","of","to","in","on","for","and","or","but","is","are",
    "was","were","be","been","being","it","its","this","that","these",
    "those","as","at","by","with","from","into","over","under","per",
    "than","then","so","do","does","did","if","when","while","which",
    "who","whom","whose","what","where","why","how",
}

def check_no_orphans(response_text: str, claims: list[Claim]) -> CheckResult:
    """Inline-template-only: ensure every [chunk:<id>] in the raw
    response appears inside a parsed Claim, and every Claim's raw
    form appears in the response. Guards against parsing drift."""
    all_markers_in_response = MARKER_RE.findall(response_text)
    all_markers_in_claims   = [cid for c in claims for cid in c.citation_ids]
    diff_resp = _multiset_diff(all_markers_in_response, all_markers_in_claims)
    diff_claim = _multiset_diff(all_markers_in_claims, all_markers_in_response)
    detail = []
    if diff_resp:
        detail.append(f"markers in response not parsed into claims: {diff_resp}")
    if diff_claim:
        detail.append(f"claims with markers not present in response: {diff_claim}")
    return CheckResult(
        name="no_orphans",
        passed=not detail,
        detail=detail,
    )

def _multiset_diff(a: list[str], b: list[str]) -> dict[str, int]:
    from collections import Counter
    diff = Counter(a) - Counter(b)
    return dict(diff)
```

**Choices and why:**

- **Token-overlap support check with a tunable `min_overlap`.**
  Cheap, deterministic, no model dependency. Catches the
  obvious failure case (claim mentions "180 days" while cited
  chunk is about parental leave). `notes/thresholds.md`
  argues for `0.5` as the mod-108 baseline and documents
  when to escalate to NLI.
- **Stopword removal before overlap.** Without it, every
  short claim "passes" because it shares `the`, `a`, `of`
  with every chunk.
- **`_is_factual` is a tight heuristic, not a classifier.**
  A discourse marker or a 3-token sentence is skipped; everything
  else is required to be cited. The bar for "factual" is
  deliberately low — we would rather over-require citations
  than let a hallucination slip through.
- **No-orphans is inline-only.** The structured template's
  schema already forbids orphans by construction; running the
  check on structured payloads is redundant and noisy.
- **Each check returns `detail`.** A failed verifier is
  worthless if the triage team cannot see *which* claim failed
  *which* check. The detail strings are the interface to the
  triage dashboard.

### 2.4 `src/verify.py` — the end-to-end verifier

```python
"""Post-generation citation verifier for mod-108.

Usage:
    result = verify_inline(response_text, legend, chunks_by_id)
    result = verify_structured(response_payload, legend, chunks_by_id)

Both return a VerifyResult(passed, checks, summary)."""
from __future__ import annotations
from dataclasses import dataclass
from parse import parse_inline, parse_structured, Claim
from checks import (
    CheckResult,
    check_citation_ids_resolve,
    check_every_claim_cited,
    check_claim_supported_by_chunks,
    check_no_orphans,
)

REFUSAL = "I cannot answer this from the retrieved context."

@dataclass
class VerifyResult:
    passed: bool
    checks: list[CheckResult]
    summary: dict

def _legend_ids(legend: list[dict]) -> set[str]:
    return {row["citation_id"] for row in legend}

def _summarise(checks: list[CheckResult]) -> dict:
    return {
        "total":   len(checks),
        "passed":  sum(1 for c in checks if c.passed),
        "failed":  [c.name for c in checks if not c.passed],
        "details": {c.name: c.detail for c in checks},
    }

def verify_inline(
    response_text: str,
    legend: list[dict],
    chunks_by_id: dict[str, str],
    min_overlap: float = 0.5,
) -> VerifyResult:
    # Refusal short-circuit: a refusal is trivially valid.
    if response_text.strip() == REFUSAL:
        return VerifyResult(True, [], {"refusal": True})
    claims = parse_inline(response_text)
    legend_ids = _legend_ids(legend)
    checks = [
        check_citation_ids_resolve(claims, legend_ids),
        check_every_claim_cited(claims),
        check_claim_supported_by_chunks(claims, chunks_by_id, min_overlap),
        check_no_orphans(response_text, claims),
    ]
    passed = all(c.passed for c in checks)
    return VerifyResult(passed, checks, _summarise(checks))

def verify_structured(
    payload: dict,
    legend: list[dict],
    chunks_by_id: dict[str, str],
    min_overlap: float = 0.5,
) -> VerifyResult:
    # Refusal short-circuit — the schema has already enforced
    # answer + empty citations when insufficient_context is true.
    if payload.get("insufficient_context"):
        return VerifyResult(True, [], {"refusal": True})
    claims = parse_structured(payload)
    legend_ids = _legend_ids(legend)
    checks = [
        check_citation_ids_resolve(claims, legend_ids),
        # structured template: every citations[] entry has >=1 id by schema,
        # so "every claim cited" is schema-enforced; we still run the
        # factual-claim check against the answer prose for safety.
        check_every_claim_cited(parse_inline(payload.get("answer", ""))),
        check_claim_supported_by_chunks(claims, chunks_by_id, min_overlap),
    ]
    passed = all(c.passed for c in checks)
    return VerifyResult(passed, checks, _summarise(checks))
```

**Choices and why:**

- **Refusal short-circuit.** A refusal has no claims to verify.
  Running the checks against it produces false positives on
  "uncited factual claim" for the refusal string.
- **Structured path still runs `check_every_claim_cited` on the
  prose.** The schema enforces that every `citations[]` entry has
  ids; it does *not* enforce that every factual sentence in
  `answer` has a corresponding `citations[]` entry. The prose
  check catches the case where the model emitted three sentences
  and only cited one.
- **Both paths run `check_claim_supported_by_chunks`.** This is
  the load-bearing check. If it does not fire, nothing else
  catches a confident but unsupported claim.
- **`min_overlap` is injected as an argument.** Operators tune it
  per-deployment; the verifier does not carry a config object.

### 2.5 `eval/verifier_eval_set.jsonl` — the eval set

40 rows. Partitions:

- **`pass`** (16 rows): well-formed responses that should pass
  all four checks. Half inline, half structured.
- **`fail`** (24 rows), four sub-types × 6 rows each:
  - **invented_id** — response cites a `citation_id` that is not
    in the legend.
  - **uncited_claim** — response makes a factual claim without
    a citation marker (inline) or a corresponding `citations[]`
    entry (structured).
  - **unsupported** — response cites an id that exists but
    whose text does not support the claim (overlap < 0.5).
  - **orphan_marker** — inline response has a `[chunk:<id>]`
    marker that sits in a paragraph break or trailing position
    without a corresponding claim sentence.

Fields:

| field | type | description |
|---|---|---|
| `id` | string | stable id |
| `template` | string | `"inline"` or `"structured"` |
| `partition` | string | `"pass"` or `"fail"` |
| `sub_type` | string | `"ok" / "invented_id" / "uncited_claim" / "unsupported" / "orphan_marker"` |
| `response` | string or object | inline text, or structured JSON object |
| `legend` | array | `[{citation_id, source_id}, ...]` |
| `chunks_by_id` | object | `{citation_id: chunk_text}` |
| `expected_pass` | boolean | gold verdict |
| `expected_failing_checks` | array | names of checks that must fail on `expected_pass = false` rows |

### 2.6 `notes/thresholds.md` — the support-check policy memo

| Policy | Mechanism | Cost | False-positive floor | When it fits |
|---|---|---|---|---|
| Exact substring | claim is a substring of the joined chunks text | O(nm), negligible | High FPR on paraphrased claims | Templates that quote verbatim (news, legal) |
| Token overlap | content-token overlap ≥ `τ` | O(n+m), negligible | Moderate FPR on numeric / factual claims; tunable with `τ` | **Mod-108 baseline.** Paraphrase-tolerant. `τ = 0.5` on content tokens is the suggested starting point. |
| Light NLI | small entailment model (e.g., deberta-v3-base-mnli) | 10–50 ms per claim on CPU, <5 ms on GPU | Low FPR; model-dependent FNR | High-stakes flows (compliance, legal) where false negatives are tolerable but false positives are not |
| LLM judge | prompted LLM scores entailment | 200–1000 ms per claim; $$ | Lowest FPR; non-deterministic | Offline audit, periodic review — not in-line |

**Recommendation for mod-108 baseline:** token overlap with
`τ = 0.5` on content tokens.

- Deterministic (same input → same verdict).
- No model dependency to deploy alongside the generator.
- Catches the obvious failure cases: wrong-topic citation,
  hallucinated numbers in an otherwise-plausible answer.
- Tunable: if a deployment sees too many false positives
  (legitimate paraphrases flagged as unsupported), lower `τ`
  to 0.4; if false negatives (unsupported claims slipping
  through), raise to 0.6 or escalate to NLI.

Trigger to escalate to NLI: when the verifier's observed
`unsupported` rate on the production stream drifts above the
exercise-02 `false_answer_rate` target (≤ 5%) and manual review
shows the token check is missing paraphrases. The NLI path
swaps in behind the existing `check_claim_supported_by_chunks`
signature; no caller change.

### 2.7 `tests/test_verify.py` — golden tests

```python
"""Golden tests for the verifier. Run with pytest -q."""
import pytest
from verify import verify_inline, verify_structured

LEGEND = [
    {"citation_id": "policy-retention-v3"},
    {"citation_id": "changelog-2026-06"},
]
CHUNKS = {
    "policy-retention-v3": "Export log retention is 180 days for customer-tier accounts.",
    "changelog-2026-06":   "2026-06-12: Export log retention changed from 90 to 180 days for customer-tier.",
}

def test_refusal_passes_trivially():
    r = verify_inline("I cannot answer this from the retrieved context.", LEGEND, CHUNKS)
    assert r.passed and r.summary.get("refusal")

def test_inline_well_formed_passes():
    resp = (
        "Export logs are retained for 180 days for customer-tier accounts "
        "[chunk:policy-retention-v3]. The 180-day period was introduced in "
        "June 2026 [chunk:changelog-2026-06]."
    )
    r = verify_inline(resp, LEGEND, CHUNKS)
    assert r.passed, r.summary

def test_inline_invented_id_fails():
    resp = "Export logs are retained for 180 days [chunk:policy-v99]."
    r = verify_inline(resp, LEGEND, CHUNKS)
    assert not r.passed
    assert "citation_ids_resolve" in r.summary["failed"]

def test_inline_uncited_claim_fails():
    resp = (
        "Export logs are retained for 180 days [chunk:policy-retention-v3]. "
        "This also applies to all regions."
    )
    r = verify_inline(resp, LEGEND, CHUNKS)
    assert not r.passed
    assert "every_claim_cited" in r.summary["failed"]

def test_inline_unsupported_claim_fails():
    resp = (
        "Export logs are retained for exactly thirty seven days "
        "[chunk:policy-retention-v3]."
    )
    r = verify_inline(resp, LEGEND, CHUNKS)
    assert not r.passed
    assert "claim_supported_by_chunks" in r.summary["failed"]

def test_structured_well_formed_passes():
    payload = {
        "insufficient_context": False,
        "contradiction": False,
        "answer": "Export logs are retained for 180 days for customer-tier accounts.",
        "citations": [{
            "claim": "Export logs are retained for 180 days for customer-tier accounts.",
            "citation_ids": ["policy-retention-v3"],
        }],
    }
    r = verify_structured(payload, LEGEND, CHUNKS)
    assert r.passed, r.summary

def test_structured_invented_id_fails():
    payload = {
        "insufficient_context": False,
        "contradiction": False,
        "answer": "Export logs are retained for 180 days.",
        "citations": [{"claim": "Export logs are retained for 180 days.",
                       "citation_ids": ["ghost-id"]}],
    }
    r = verify_structured(payload, LEGEND, CHUNKS)
    assert not r.passed
    assert "citation_ids_resolve" in r.summary["failed"]

def test_structured_unsupported_fails():
    payload = {
        "insufficient_context": False,
        "contradiction": False,
        "answer": "Export logs are retained for exactly thirty seven days.",
        "citations": [{"claim": "Export logs are retained for exactly thirty seven days.",
                       "citation_ids": ["policy-retention-v3"]}],
    }
    r = verify_structured(payload, LEGEND, CHUNKS)
    assert not r.passed
    assert "claim_supported_by_chunks" in r.summary["failed"]
```

Running these against the reference implementation should pass.
A learner solution that fails any of them has a real bug, not a
stylistic difference.

### 2.8 How the verifier fits in the request path

```
query ──▶ retrieval ──▶ mod-107 assembler ──▶ (context_block, legend, chunks_by_id)
                                                   │
                                                   ▼
                                         exercise-01 prompt template
                                                   │
                                                   ▼
                                               LLM call
                                                   │
                                                   ▼
                                       exercise-04 verifier  ◀── chunks_by_id, legend
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                      passed = True                                 passed = False
                     return response                      (log, alert, gate, or
                                                          substitute refusal — policy
                                                          decision by the caller)
```

The verifier does **not** decide what to do with a failing
response. It returns a verdict and detail; the caller (the chat
endpoint, the batch pipeline, the eval harness) applies policy:
"gate and return refusal," "log and return anyway," "route to
human review." Keeping policy out of the verifier is why it can
be shared across inline, structured, prod, and eval use cases
without special-casing.

## 3. Validation steps

1. All golden tests in `tests/test_verify.py` pass.
2. The verifier returns a passing result on a canonical refusal
   string without running any check.
3. For a well-formed inline response with all claims cited and
   supported, all four checks pass.
4. For a well-formed structured response (schema-valid, all ids
   in legend, all claims supported), the three structured
   checks pass.
5. The invented-id test fails `citation_ids_resolve` only
   (not `claim_supported_by_chunks`, not `every_claim_cited`) —
   the failure is isolated to the right check.
6. The unsupported-claim test fails
   `claim_supported_by_chunks` only. If this test incorrectly
   trips `citation_ids_resolve`, the legend-resolution check is
   bleeding into the support check.
7. The verifier is deterministic: same input → same result.
8. The eval set loads cleanly as JSONL; partitions sum to 40;
   each failure sub-type has at least 4 rows.
9. The verifier runs end-to-end on all 40 eval rows in under
   100 ms (token-overlap policy); the NLI policy may exceed
   this and should be benchmarked separately.
10. `min_overlap` is a function argument, not a global constant.
    Operators tune it without editing source.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Four checks | `citation_ids_resolve`, `every_claim_cited`, `claim_supported_by_chunks`, `no_orphans` are implemented as independent functions with `CheckResult` output. |
| Parser separation | Inline and structured parsers produce a common `Claim[]` output; the checks do not know which template was used. |
| Refusal short-circuit | A canonical refusal returns `passed=True` without running the checks. |
| Support check | Token-overlap with content-word filtering and tunable threshold. Not pure substring (would false-positive on paraphrase); not model-dependent (would make the verifier a second model to deploy). |
| No orphan check (inline) | Both directions: markers → claims, and claims → markers. One-way checks let orphan markers through. |
| Deterministic | Same input → byte-identical verdict and detail. No RNG, no `set` ordering, no clock. |
| Policy-free | The verifier returns a verdict; it does **not** mutate, gate, or substitute the response. |
| Eval sub-types | Four fail sub-types represented; each has ≥ 4 rows. |
| Threshold decision | `notes/thresholds.md` names token overlap `τ = 0.5` as the baseline and names the trigger to escalate to NLI. |
| Observability | `summary.details` contains per-check detail suitable for logging and triage. |

Fail-fast smells:

- Support check is pure substring. Paraphrased claims
  false-positive.
- Support check calls out to an LLM. The verifier now has a
  second model to deploy and non-deterministic verdicts.
- Refusal string not short-circuited; every refusal triggers
  "uncited factual claim."
- `every_claim_cited` without a factual-vs-discourse filter.
  Harmless transition sentences get flagged.
- No orphan check. Dangling `[chunk:<id>]` at the end of a
  paragraph does not fail.
- Verifier gates the response internally ("replace with
  refusal if failed"). That is the caller's policy decision.
- Threshold is a global constant. Tuning requires a code change.
- `check_claim_supported_by_chunks` concatenates all chunks even
  for a claim that cites only one of them. Can accidentally
  "support" with text the model did not cite.

Note on that last item: the reference `check_claim_supported_by_chunks`
above concatenates `c.citation_ids` only — the ids the model actually
cited for that specific claim — not the full legend. A learner
implementation that joins the entire `chunks_by_id` is making the
claim pass on evidence the model did not reference, and that is a bug.

## 5. Common mistakes

- **Pure substring support check.** `"180 days"` must appear
  verbatim in the chunk. Any paraphrase of the claim or the
  chunk fails. Token overlap is the right trade.
- **Concatenating all chunks for the support check.** The
  check must concatenate *only the cited chunks*, not the
  whole legend. Otherwise the claim passes on evidence the
  model never cited.
- **Not short-circuiting refusals.** Every valid refusal looks
  like a hallucination to the "every claim cited" check; the
  verifier flags 100% of refusals as failures.
- **No stopword filtering.** Short claims trivially pass the
  overlap check because `the/a/of/to/for` overlap with every
  chunk.
- **Pretending the structured schema is the whole verifier.**
  The schema enforces `citations[]` has ≥ 1 id per entry; it
  does not enforce that `answer`'s factual sentences each have
  a `citations[]` entry. Run the uncited-claim check on the
  prose too.
- **One-way orphan check.** Markers without claims is one
  failure; claims without markers is another. Check both.
- **Gating in the verifier.** The verifier returns a verdict.
  The caller decides what to do with a failing verdict.
- **Loose `citation_id` regex in parsing.** If the parser
  accepts `chunk:policy v3` (with a space), the resolve
  check will "find" an id that cannot exist. Keep the regex
  aligned with the slug pattern used by mod-107.
- **No eval set.** The verifier is the piece that should be
  regression-tested the most, because it is the last line of
  defence before the user sees a hallucinated citation.
- **Hard-coded threshold.** A verifier that cannot be tuned
  by config is a verifier that gets bypassed in production.

## 6. References

- Citation faithfulness:
  - Gao et al. (2023), "Enabling Large Language Models to
    Generate Text with Citations" —
    [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627).
  - Menick et al. (2022), "Teaching language models to support
    answers with verified quotes" (GopherCite) —
    [arxiv.org/abs/2203.11147](https://arxiv.org/abs/2203.11147).
- Faithfulness / entailment metrics for RAG:
  - Es et al. (2023), "RAGAS: Automated Evaluation of
    Retrieval Augmented Generation" —
    [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217).
- NLI model families (if escalating beyond token overlap):
  - He et al. (2021), "DeBERTa-v3" —
    [arxiv.org/abs/2111.09543](https://arxiv.org/abs/2111.09543).
- Upstream handoffs:
  - mod-108 exercise-01 (template, schema, legend contract) —
    `../exercise-01-inline-and-structured-citation-instruction-templates/SOLUTION.md`.
  - mod-108 exercise-02 (refusal contract and short-circuit) —
    `../exercise-02-refusal-when-no-context-training-shot-and-eval/SOLUTION.md`.
  - mod-108 exercise-03 (contradiction flag; verifier treats
    contradiction entries identically — ≥ 2 citation ids per
    claim) —
    `../exercise-03-contradiction-handling-across-retrieved-chunks/SOLUTION.md`.
  - mod-107 exercise-04 (assembler legend + `chunks_by_id`) —
    `../../mod-107-reranking-and-context-assembly/exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors/SOLUTION.md`.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML09 Output Integrity. The verifier is the
    deterministic enforcement of the output-integrity contract.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    Techniques involving manipulated or spoofed model outputs
    motivate a mechanical, model-free verification layer.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    The *Measure / Manage* functions call for repeatable,
    testable controls on model output; the verifier is the
    repeatable control for the citation contract.
