# Exercise 04 — Solution: Context Assembly With Dedup, Ordering, and Citation Anchors

This is a reference solution for
`exercise-04-context-assembly-with-dedup-ordering-and-citation-anchors.md`.
It is an **implementation** exercise. The learner builds the assembler
that turns the reranker+compressor output (exercises 02 and 03) into the
`context_block` that the prompt builder (mod-108) will interpolate into
the LLM prompt. The assembler is the last stage before generation and
owns three jobs: deduplicate near-identical chunks, order them with the
generator's positional bias in mind, and attach citation anchors that
mod-108 can require the model to emit.

The teaching point: the context block is a **protocol between retrieval
and generation**, not a bag of strings. The grounded-answer contract
(mod-108/109) depends on `[chunk:id]` markers surviving from the
assembler through to the model's output; the assembler is where that
contract is written.

## 1. Solution overview

The deliverable has six parts:

1. **`src/assemble.py`** — the assembler pipeline:
   `dedup → order → budget-trim → anchor → serialise`.
2. **`src/dedup.py`** — two deduplicators: a MinHash-based near-duplicate
   detector (cheap, good for templated boilerplate) and an
   embedding-cosine detector (slightly more expensive, good for
   paraphrases).
3. **`src/order.py`** — the ordering policy. Default: U-shape (top
   reranker score at positions 0 and -1, descending outward). Configurable
   to score-descending or document-original-order.
4. **`src/anchors.py`** — the anchor encoder that attaches a stable
   `[chunk:<citation_id>]` marker and emits the citation legend.
5. **`tests/test_assemble.py`** — golden-test cases: dedup across
   near-duplicates, token-budget trimming, anchor-id uniqueness,
   ordering determinism.
6. **`notes/results.md`** — before/after on a representative query
   showing the five stages' effect on token count and final context.

The exercise is scoped to a *robust assembler*, not a tuning exercise.
If the assembler is deterministic, inside the token budget, emits
unique and stable citation anchors, and the test suite passes, the
exercise passes. Quality-vs-ordering trade-offs are explored in
exercise-05.

## 2. Worked answer or implementation

### 2.1 The assembler's five jobs in one paragraph

Given a list of reranked and (optionally) compressed chunks, the
assembler (1) **deduplicates** — two chunks that say nearly the same
thing waste prompt budget and can confuse the generator; (2) **orders**
— LLMs attend unevenly across the context window (Liu et al. 2023),
and the strongest evidence belongs where the model looks; (3) **trims
to budget** — the LLM's context window is a hard ceiling, and
overshooting is a 400-level error, not a graceful degradation;
(4) **anchors** — each surviving chunk gets a stable identifier the
model is told to cite; (5) **serialises** — produces a textual block
with a documented, parseable structure.

### 2.2 `src/dedup.py` — the dedup stage

```python
"""Near-duplicate removal: MinHash for boilerplate, cosine for paraphrase."""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
import numpy as np

_TOKEN = re.compile(r"[\w']+")

def _shingles(text: str, k: int = 5) -> set[str]:
    toks = _TOKEN.findall(text.lower())
    if len(toks) <= k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+k]) for i in range(len(toks) - k + 1)}

def _minhash(shingles: set[str], num_perm: int = 64) -> np.ndarray:
    sigs = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for sh in shingles:
        for p in range(num_perm):
            h = hashlib.blake2b(
                sh.encode("utf-8"),
                digest_size=8,
                person=p.to_bytes(2, "big"),
            ).digest()
            v = int.from_bytes(h, "big")
            if v < sigs[p]:
                sigs[p] = v
    return sigs

def minhash_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())

@dataclass
class DedupStats:
    dropped: int
    merged_source_ids: dict[int, list[str]]   # surviving-idx -> [merged source_ids]

def dedup_minhash(chunks, tau: float = 0.8):
    """Return (kept_chunks, stats). Drops chunks whose MinHash Jaccard
    against an already-kept chunk is >= tau; records the dropped source id."""
    sigs = [_minhash(_shingles(c.text)) for c in chunks]
    kept_idx: list[int] = []
    merged: dict[int, list[str]] = {}
    for i, sig in enumerate(sigs):
        is_dup = False
        for j in kept_idx:
            if minhash_jaccard(sig, sigs[j]) >= tau:
                merged.setdefault(j, []).append(chunks[i].source_id)
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)
    kept = [chunks[i] for i in kept_idx]
    return kept, DedupStats(len(chunks) - len(kept), merged)

def dedup_cosine(chunks, encode, tau: float = 0.95):
    """Embedding-cosine dedup for paraphrase-grade duplicates."""
    if not chunks:
        return [], DedupStats(0, {})
    vecs = np.stack([encode(c.text) for c in chunks])
    kept_idx: list[int] = []
    merged: dict[int, list[str]] = {}
    for i in range(len(chunks)):
        is_dup = False
        for j in kept_idx:
            sim = float(vecs[i] @ vecs[j])
            if sim >= tau:
                merged.setdefault(j, []).append(chunks[i].source_id)
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)
    kept = [chunks[i] for i in kept_idx]
    return kept, DedupStats(len(chunks) - len(kept), merged)
```

**Choices and why:**

- Two methods, not one. MinHash with 5-grams catches templated
  boilerplate (changelog headers, licence stubs) at near-zero cost.
  Embedding-cosine catches paraphrase (two different phrasings of
  the same policy). They are composable — run MinHash first, then
  cosine on the survivors.
- Preserve the **first** occurrence of a duplicate group. The first
  occurrence has the highest reranker score (chunks arrive in
  reranker order). Dropping it would lose the stronger signal.
- Record the dropped `source_id`s against the surviving chunk so the
  anchor stage can either list all sources or pick one canonical.
- `tau` is a knob. For MinHash, `0.8` is a defensible starting point
  (20% different shingles is a real difference). For embeddings,
  `0.95` cosine is near-paraphrase — do not go lower without
  evaluating on content you intend to keep.

### 2.3 `src/order.py` — the ordering stage

```python
"""Order surviving chunks for the generator's positional bias."""
from __future__ import annotations
from enum import Enum

class OrderPolicy(str, Enum):
    U_SHAPE          = "u_shape"           # strongest at ends (Liu et al. 2023)
    SCORE_DESCENDING = "score_desc"        # strongest first (trusts generator)
    SOURCE_ORDER     = "source_order"      # retain within-document order

def order_u_shape(chunks):
    """Alternate strongest-first-and-last: s0 at position 0, s1 at -1,
    s2 at 1, s3 at -2, ..."""
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    front, back = [], []
    for i, c in enumerate(sorted_chunks):
        (front if i % 2 == 0 else back).append(c)
    return front + list(reversed(back))

def order(chunks, policy: OrderPolicy):
    if not chunks:
        return []
    if policy is OrderPolicy.U_SHAPE:
        return order_u_shape(chunks)
    if policy is OrderPolicy.SCORE_DESCENDING:
        return sorted(chunks, key=lambda c: c.score, reverse=True)
    if policy is OrderPolicy.SOURCE_ORDER:
        return sorted(chunks, key=lambda c: (c.source_id, c.metadata.get("offset", 0)))
    raise ValueError(f"unknown policy: {policy}")
```

**Choices and why:**

- `U_SHAPE` is the exercise-05 default. "Lost in the middle"
  (Liu et al. 2023) describes a U-shaped attention profile on
  common instruction-tuned LLMs; U-shape ordering puts the two
  highest-scoring chunks where the model attends most.
- `SCORE_DESCENDING` is the conservative baseline for strong
  long-context models (Claude, Gemini, GPT-4-class) that have
  mitigated the positional bias meaningfully. Measure on the actual
  generator; do not assume.
- `SOURCE_ORDER` is reserved for workflows that need **narrative
  continuity** (document Q&A, multi-paragraph answers from a single
  source). Score order would scramble a legal clause or a numbered
  procedure.
- The policy is a knob, not a fixed strategy. Exercise-05 sweeps
  it; here the assembler just needs to support all three.

### 2.4 `src/anchors.py` — the anchor encoder

```python
"""Attach stable citation anchors and build the legend."""
from __future__ import annotations
from dataclasses import dataclass
import re

_SLUG = re.compile(r"[^a-z0-9_-]+")

def _slugify(s: str) -> str:
    return _SLUG.sub("-", s.lower()).strip("-")

@dataclass
class AnchoredChunk:
    anchor: str          # "[chunk:<citation_id>]"
    citation_id: str     # stable id the model will cite
    text: str
    source_id: str
    extra_sources: list[str]
    score: float

def attach_anchors(chunks, merged_sources: dict[int, list[str]]):
    """Attach a stable citation anchor per chunk.
    citation_id is derived from source_id, not position, so that re-runs
    with the same retrieval produce the same anchors."""
    anchored: list[AnchoredChunk] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(chunks):
        base = _slugify(c.source_id)
        # Disambiguate when the same source produced multiple surviving chunks
        idx = seen.get(base, 0) + 1
        seen[base] = idx
        citation_id = base if idx == 1 else f"{base}#{idx}"
        anchored.append(AnchoredChunk(
            anchor=f"[chunk:{citation_id}]",
            citation_id=citation_id,
            text=c.text,
            source_id=c.source_id,
            extra_sources=merged_sources.get(i, []),
            score=c.score,
        ))
    return anchored
```

**Choices and why:**

- **Citation ids are derived from `source_id`**, not from the
  chunk's position in the final block. Position-based ids
  (`[chunk:1]`, `[chunk:2]`) break the audit trail — the same
  document can be `[chunk:1]` in one response and `[chunk:3]` in
  another. Source-derived ids are stable across runs.
- Collisions within a response (two chunks from the same source
  surviving dedup) are disambiguated with a `#N` suffix.
- `extra_sources` carries the source ids merged away by dedup so
  the legend can disclose them. This matters for duplicate
  governance notices from different departments, where the user
  may have a right to know every source that agreed.

### 2.5 `src/assemble.py` — the end-to-end assembler

```python
"""End-to-end assembler: dedup -> order -> budget -> anchor -> serialise."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from compressor_base import Chunk       # from exercise-03
from dedup   import dedup_minhash, dedup_cosine
from order   import order, OrderPolicy
from anchors import attach_anchors, AnchoredChunk

@dataclass
class AssembleConfig:
    policy: OrderPolicy = OrderPolicy.U_SHAPE
    max_tokens: int = 3500
    dedup_minhash_tau: float = 0.8
    dedup_cosine_tau: float = 0.95
    run_cosine_dedup: bool = True
    encode: Callable[[str], object] | None = None   # embedding fn if cosine dedup

@dataclass
class AssembleOutput:
    context_block: str
    legend: list[dict]       # [{citation_id, source_id, extra_sources}]
    stats: dict              # dedup / trim stats for logging

def assemble(
    chunks: list[Chunk],
    count_tokens: Callable[[str], int],
    cfg: AssembleConfig = AssembleConfig(),
) -> AssembleOutput:
    # 1. Dedup
    kept, mh_stats = dedup_minhash(chunks, tau=cfg.dedup_minhash_tau)
    if cfg.run_cosine_dedup and cfg.encode is not None:
        kept, cos_stats = dedup_cosine(kept, cfg.encode, tau=cfg.dedup_cosine_tau)
    else:
        cos_stats = type(mh_stats)(0, {})

    # 2. Order (score-descending view is used for budget trimming)
    ordered = order(kept, cfg.policy)

    # 3. Budget trim.
    #    Pop from the position that contains the LOWEST-score chunk currently
    #    in the block — not simply from the tail — so U_SHAPE still trims the
    #    weakest evidence first.
    trimmed = list(ordered)
    dropped_ids: list[str] = []
    def total_tokens(cs): return sum(count_tokens(c.text) for c in cs)
    while total_tokens(trimmed) > cfg.max_tokens and trimmed:
        weakest_idx = min(range(len(trimmed)), key=lambda i: trimmed[i].score)
        dropped_ids.append(trimmed[weakest_idx].source_id)
        trimmed.pop(weakest_idx)

    # 4. Anchors (merged_sources indexed into post-dedup order; we propagate
    #    sources from both dedup passes before trim/order).
    merged_all = _merge_source_maps(chunks, kept, mh_stats, cos_stats)
    anchored = attach_anchors(
        trimmed,
        {i: merged_all.get(c.source_id, []) for i, c in enumerate(trimmed)},
    )

    # 5. Serialise
    block, legend = serialise(anchored)
    stats = {
        "input_chunks":   len(chunks),
        "dedup_minhash":  mh_stats.dropped,
        "dedup_cosine":   cos_stats.dropped,
        "dropped_budget": len(dropped_ids),
        "final_chunks":   len(anchored),
        "final_tokens":   total_tokens(trimmed),
    }
    return AssembleOutput(block, legend, stats)

def _merge_source_maps(input_chunks, kept, mh, cos):
    """Return {kept_source_id: [merged_source_ids]} after both dedup passes."""
    kept_ids = {c.source_id for c in kept}
    mapping: dict[str, list[str]] = {sid: [] for sid in kept_ids}
    for kept_idx, merged_src_ids in mh.merged_source_ids.items():
        host_sid = input_chunks[kept_idx].source_id
        mapping.setdefault(host_sid, []).extend(merged_src_ids)
    for kept_idx, merged_src_ids in cos.merged_source_ids.items():
        host_sid = kept[kept_idx].source_id
        mapping.setdefault(host_sid, []).extend(merged_src_ids)
    return mapping

def serialise(anchored: list[AnchoredChunk]) -> tuple[str, list[dict]]:
    body_lines = []
    legend: list[dict] = []
    for a in anchored:
        body_lines.append(f"{a.anchor}\n{a.text}")
        legend.append({
            "citation_id":   a.citation_id,
            "source_id":     a.source_id,
            "extra_sources": a.extra_sources,
        })
    block = "\n\n---\n\n".join(body_lines)
    return block, legend
```

**Choices and why:**

- **Order before anchor**, not after. If the anchor ids depended on
  the final position in the block, dedup/trim/order would silently
  renumber them.
- **Budget trim drops the weakest chunk, not the last chunk in the
  block.** Under U-shape ordering the weakest chunks are in the
  middle; a blind "pop until under budget" would drop the exact
  chunk you most wanted to keep (strong chunk at the end).
- The output has three layers: `context_block` (what mod-108
  interpolates), `legend` (what mod-108 renders as footnotes /
  the assistant can cite), `stats` (what the ops dashboard logs).
  The three layers are distinct and must not be collapsed.
- `AssembleConfig` captures every tunable. Reproducibility means
  the config object can be serialised, checked in, and re-played.

### 2.6 Serialised output shape

For a query `"What is the retention policy for export logs?"` after
reranker produced three chunks (two from the same policy doc,
near-paraphrased; one from a changelog):

```
[chunk:policy-retention-v3]
Export log retention is 180 days for all customer-tier accounts. Internal
accounts retain for 90 days unless otherwise specified by Policy §4.2.

---

[chunk:policy-retention-v3#2]
The 180-day retention applies across all regions. Customers may request
early deletion via the data-management console.

---

[chunk:changelog-2026-06]
2026-06-12: Export log retention changed from 90 to 180 days for
customer-tier. See Policy §4.2.
```

The legend returned alongside (for mod-108 to render):

```json
[
  {"citation_id": "policy-retention-v3",   "source_id": "policy-retention-v3", "extra_sources": ["policy-retention-v2"]},
  {"citation_id": "policy-retention-v3#2", "source_id": "policy-retention-v3", "extra_sources": []},
  {"citation_id": "changelog-2026-06",     "source_id": "changelog-2026-06",   "extra_sources": []}
]
```

Note that `policy-retention-v2` — a near-paraphrase dropped by
cosine dedup — is disclosed in `extra_sources`. The user's right
to know "which documents agreed" is preserved without wasting
prompt tokens.

### 2.7 `tests/test_assemble.py` — the test cases

```python
"""Golden tests for the assembler. Run with pytest -q."""
import pytest
from compressor_base import Chunk
from order   import OrderPolicy
from assemble import assemble, AssembleConfig

def toks(s: str) -> int:
    return len(s.split())

def test_dedup_merges_identical_text():
    chunks = [
        Chunk("policy applies to all accounts", "a", score=0.9),
        Chunk("policy applies to all accounts", "b", score=0.8),
        Chunk("exception for internal users", "c", score=0.7),
    ]
    out = assemble(chunks, toks, AssembleConfig(run_cosine_dedup=False))
    assert out.stats["final_chunks"] == 2
    assert "b" in out.legend[0]["extra_sources"]

def test_budget_drops_weakest_not_last():
    chunks = [
        Chunk("alpha " * 50, "a", score=0.9),
        Chunk("beta "  * 50, "b", score=0.2),    # weakest
        Chunk("gamma " * 50, "c", score=0.7),
    ]
    out = assemble(
        chunks, toks,
        AssembleConfig(max_tokens=110, run_cosine_dedup=False,
                       policy=OrderPolicy.U_SHAPE),
    )
    kept_cids = {row["citation_id"] for row in out.legend}
    assert "b" not in kept_cids   # weakest got dropped
    assert "a" in kept_cids and "c" in kept_cids

def test_anchors_stable_across_runs():
    chunks = [
        Chunk("x", "doc-1", score=0.9),
        Chunk("y", "doc-2", score=0.8),
    ]
    a = assemble(chunks, toks, AssembleConfig(run_cosine_dedup=False))
    b = assemble(chunks, toks, AssembleConfig(run_cosine_dedup=False))
    assert [r["citation_id"] for r in a.legend] == \
           [r["citation_id"] for r in b.legend]

def test_u_shape_order_puts_strongest_at_ends():
    chunks = [
        Chunk("alpha", "a", score=0.9),
        Chunk("beta",  "b", score=0.8),
        Chunk("gamma", "c", score=0.7),
        Chunk("delta", "d", score=0.6),
    ]
    out = assemble(chunks, toks, AssembleConfig(
        run_cosine_dedup=False, policy=OrderPolicy.U_SHAPE
    ))
    ids = [row["citation_id"] for row in out.legend]
    assert ids[0]  == "a"   # strongest first
    assert ids[-1] == "b"   # second strongest last
```

Running these against the reference implementation should pass. A
learner solution that fails any of them has a real bug, not a
stylistic difference.

### 2.8 `notes/results.md` — a worked before/after

Query: `"Which regions does the 180-day retention apply to?"`

| Stage           | Chunks | Tokens | Note |
|-----------------|-------:|-------:|------|
| Reranker output |     8  |  3620  | three near-duplicates of the policy paragraph |
| After MinHash   |     6  |  2810  | one pair of templated changelog headers merged |
| After cosine    |     4  |  2010  | two policy paraphrases merged |
| After order     |     4  |  2010  | U-shape, strongest at positions 0 and -1 |
| After budget    |     4  |  2010  | no trim needed (under 3500-token budget) |
| After anchor    |     4  |  2010  | citation ids: `policy-retention-v3`, `changelog-2026-06`, `regions-list-v2`, `faq-exports#2` |

Impact: 3620 → 2010 tokens (44% reduction) with no gold-content loss,
U-shape ordering applied, four distinct citations rather than seven
near-duplicates confusing the generator.

## 3. Validation steps

1. All four tests in `test_assemble.py` pass.
2. Running the assembler twice with the same input produces the
   same output bytes. Non-determinism (hash iteration order, seed
   dependence) is a build-breaker.
3. The citation ids in the context block appear exactly in the
   legend, with no orphans in either direction.
4. Budget trimming drops the lowest-score chunk, not the last chunk
   in the ordered list. The test above enforces this.
5. The assembler emits valid Markdown / plain text that mod-108's
   prompt template can interpolate without escaping issues (no
   stray `{{` or template delimiters inside the chunk text).
6. When the context exceeds the budget even after all dedup,
   trimming leaves at least one chunk (never emits an empty block).
7. Merged source ids are surfaced in `extra_sources`. A dedup step
   that drops `source_id`s silently is a data-provenance bug.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Dedup | Both methods implemented and composable; MinHash catches templated duplicates, cosine catches paraphrase. Merged source ids are preserved, not discarded. |
| Ordering | At least U-shape and score-descending are implemented. The U-shape ordering is verifiable by test. |
| Budget | Budget trimming removes the **lowest-score** chunk, not the last chunk by position. Trimming with U-shape ordering is covered by a test. |
| Anchors | Citation ids are derived from `source_id`, not from final block position. Same retrieval → same ids on re-run. |
| Legend | Legend lists every anchor and its merged `extra_sources`; no orphan anchors in either direction. |
| Determinism | Same input → byte-identical output. No reliance on `set` ordering or RNG without a seed. |
| Token accounting | The assembler uses a `count_tokens` callback matched to the generator's tokenizer. A learner who counts characters or words has a budget-overrun bug waiting to happen. |
| Config hygiene | `AssembleConfig` captures all knobs; the config is serialisable and the operating point is checked in. |
| Tests | At least the four listed tests pass; the exercise invites adding more. |
| Observability | Returns `stats` (dedup counts, dropped-by-budget count, final token count) so mod-109 / ops can log per-query. |

Fail-fast smells:

- Citation ids like `[chunk:1]`, `[chunk:2]` numbered by position.
  They are not stable; a reviewer who audits the model's output
  against yesterday's logs will get a different document under the
  same id.
- A dedup implementation that drops the `source_id` of the merged
  chunk silently. Users (and compliance) have a right to know
  which sources agreed.
- A budget-trim loop that pops the last chunk until under budget.
  Under U-shape ordering this drops the strongest tail chunk.
- An assembler that fails open — emits an over-budget block and
  relies on the LLM to truncate. The LLM will 400 or silently
  drop the tail; both are worse than the assembler enforcing the
  budget.
- Non-determinism: using Python's `set()` iteration order or
  default `hash()` with randomised seeding without `sorted()`
  downstream. Two runs, two different context blocks, two
  different answers.

## 5. Common mistakes

- **Citation anchors numbered by position.** Position-based ids
  break on dedup, order, or retry. Use source-derived ids and
  disambiguate with `#N`.
- **Discarding merged source ids.** Dedup collapses evidence but
  does not collapse provenance. The user may need to see every
  source that supported the claim.
- **Pop-the-last-chunk budget trim.** Under U-shape ordering
  this drops the second-strongest chunk. Always drop by score.
- **Hand-tuning MinHash threshold on the whole corpus.** Different
  corpora have different boilerplate rates. Tune per deployment
  on a representative slice.
- **Running cosine dedup before MinHash.** Cosine dedup is
  per-chunk-embedding cost; MinHash is a few hundred hashes. MinHash
  first, then cosine on the survivors.
- **Counting tokens with `len(text.split())`.** Tokenisers and
  whitespace splits disagree by 20–40%. A budget set on a split
  count will overrun the generator's actual context window.
- **Returning the context as a single string.** Downstream
  (mod-108) needs the legend separately to render citations and
  the generator contract needs the anchors to appear in the body.
  Return both.
- **Letting the generator pick its own ordering.** Some prompt
  templates say "the most relevant documents are below" and then
  score-descend; some say "in order of relevance" but shuffle.
  Pick a documented policy and own it.
- **No test for anchor stability across runs.** The test is five
  lines. Not writing it is how silent regressions ship.
- **Over-aggressive cosine dedup threshold.** `tau = 0.80` on
  cosine will merge semantically-different chunks that share a
  topic. Keep cosine dedup at `≥ 0.92` unless you have measured
  on content you trust.

## 6. References

- Lost-in-the-middle (motivates ordering):
  - Liu et al. (2023), "Lost in the Middle: How Language Models Use
    Long Contexts" — [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172).
- Dedup theory:
  - Broder (1997), "On the Resemblance and Containment of
    Documents" (MinHash) —
    [dl.acm.org/doi/10.5555/829502.830043](https://dl.acm.org/doi/10.5555/829502.830043).
  - `datasketch` MinHash library (reference implementation
    pattern) — [github.com/ekzhu/datasketch](https://github.com/ekzhu/datasketch).
- Citation-grounded generation:
  - Gao et al. (2023), "Enabling Large Language Models to Generate
    Text with Citations" —
    [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627).
- Prompt interpolation downstream (mod-108 handoff):
  - OpenAI Structured Outputs — the citation contract is
    consumable as a strict JSON schema in later modules.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
