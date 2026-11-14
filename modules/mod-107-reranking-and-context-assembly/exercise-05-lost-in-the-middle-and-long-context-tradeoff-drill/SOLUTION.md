# Exercise 05 — Solution: Lost-in-the-Middle and the Long-Context Trade-off Drill

This is a reference solution for
`exercise-05-lost-in-the-middle-and-long-context-tradeoff-drill.md`. It
is a mixed **design + implementation** exercise. The learner reproduces
the "Lost in the Middle" effect (Liu et al. 2023) on a modern generator,
sweeps the knobs that make it worse or better (context length, gold-doc
position, chunk count, ordering policy), and writes a one-page memo on
what this means for the assembler built in exercise-04.

The teaching point: even long-context models attend unevenly across
their context window. Shoving all 100 retrieved chunks into a 128k-token
prompt is not a free upgrade over tight top-K retrieval. The exercise
measures that trade-off instead of asserting it.

## 1. Solution overview

The deliverable has five parts:

1. **`src/needle_eval.py`** — the harness. Builds synthetic prompts
   with a controlled "needle" document (a gold fact-containing chunk)
   placed at a chosen position in a context of configurable length,
   runs the generator, measures whether the generator retrieved the
   needle fact.
2. **`src/sweep.py`** — the sweep: Cartesian over (context tokens,
   needle position, number of distractors). Logs per-run accuracy and
   p95 latency.
3. **`src/ordering_ablation.py`** — rerun the mod-107 eval slice
   under three ordering policies (score-descending, source-order,
   U-shape from exercise-04) and compare answer accuracy.
4. **`notes/results.md`** — the three plots and the memo:
   - position-vs-accuracy at each context length (the U-curve);
   - total-tokens-vs-accuracy at fixed K (the long-context trade-off);
   - ordering-policy-vs-accuracy on the real eval slice.
5. **`notes/memo.md`** — the one-page recommendation that mod-108's
   prompt builder and the assembler's default ordering policy should
   implement.

The exercise is scoped to *reproducing the phenomenon and recommending
a default*, not to publishing a new result. If the learner can show the
U-curve on their generator, identify where it breaks, and defend an
ordering policy choice, the exercise passes.

## 2. Worked answer or implementation

### 2.1 Lost-in-the-middle in one paragraph

Liu et al. (2023) showed that transformer LLMs retrieve information
from the start and end of their context window substantially more
reliably than from the middle — a U-shaped accuracy profile versus
position. The effect was large on 2023-era 8k-context models (up to
~20 accuracy points between endpoints and midpoint on multi-document
QA). On long-context models (Claude, Gemini, GPT-4-class), the U is
shallower but **not flat**; the practical implication is that filling
the context window is not a substitute for ranking. mod-107's assembler
(exercise-04) orders by score with U-shape placement as the default
precisely because of this.

### 2.2 `src/needle_eval.py` — the needle harness

```python
"""Needle-in-a-haystack-style evaluation: known fact buried in context."""
from __future__ import annotations
import json, random, time
from dataclasses import dataclass
from typing import Callable

@dataclass
class NeedleCase:
    qid: str
    question: str
    gold_answer: str              # ground-truth answer, exact match or synonym set
    gold_synonyms: list[str]
    needle_text: str              # the chunk that contains the answer
    distractors: list[str]        # unrelated-but-plausible chunks
    position_frac: float          # 0.0 = first, 1.0 = last
    target_total_tokens: int      # construct context around this number

def _pad_to_tokens(
    distractor_pool: list[str], target_tokens: int, count_tokens
) -> list[str]:
    out, tok = [], 0
    rng = random.Random(0xB00FA)
    while tok < target_tokens and distractor_pool:
        d = rng.choice(distractor_pool)
        out.append(d)
        tok += count_tokens(d)
    return out

def build_prompt(case: NeedleCase, count_tokens) -> str:
    distractors = _pad_to_tokens(
        case.distractors, case.target_total_tokens, count_tokens
    )
    pos = int(case.position_frac * len(distractors))
    contexts = distractors[:pos] + [case.needle_text] + distractors[pos:]
    body = "\n\n---\n\n".join(contexts)
    return (
        "Answer the question using only the documents below.\n\n"
        f"Question: {case.question}\n\n"
        f"Documents:\n{body}\n\n"
        "Answer:"
    )

def is_correct(answer: str, case: NeedleCase) -> bool:
    low = answer.lower().strip()
    if case.gold_answer.lower() in low:
        return True
    return any(s.lower() in low for s in case.gold_synonyms)

def run(
    cases: list[NeedleCase],
    generator: Callable[[str], str],
    count_tokens,
) -> list[dict]:
    rows = []
    for c in cases:
        prompt = build_prompt(c, count_tokens)
        t0 = time.perf_counter()
        out = generator(prompt)
        t1 = time.perf_counter()
        rows.append({
            "qid": c.qid,
            "position_frac": c.position_frac,
            "target_tokens": c.target_total_tokens,
            "prompt_tokens": count_tokens(prompt),
            "correct": is_correct(out, c),
            "latency_ms": (t1 - t0) * 1000,
            "answer_sample": out[:160],
        })
    return rows
```

**Choices and why:**

- `position_frac` in `[0.0, 1.0]` lets the sweep place the needle at
  any fraction of the context. Integer positions depend on context
  length and are harder to compare across context sizes.
- Gold-synonym lists avoid false negatives on paraphrase without
  having to call another LLM as a judge (that is mod-109's
  territory; here we need a deterministic metric).
- The distractors come from a **fixed pool**. Random distractors
  per run introduce variance that swamps the position effect.
- The prompt template is intentionally minimal. Fancy prompts
  (chain-of-thought, "cite the document") would help accuracy and
  obscure the position effect we are measuring. Measure the floor
  first.

### 2.3 `src/sweep.py` — the sweep driver

```python
"""Sweep (target_tokens, position_frac) and log the U-curve."""
from __future__ import annotations
import json, pathlib
from needle_eval import NeedleCase, run

def grid(cases_pool: list[dict]) -> list[NeedleCase]:
    cases = []
    for c in cases_pool:
        for tokens in (2_000, 8_000, 32_000, 128_000):
            for pos in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
                cases.append(NeedleCase(
                    qid=f"{c['qid']}__t{tokens}__p{pos}",
                    question=c["question"],
                    gold_answer=c["gold_answer"],
                    gold_synonyms=c["gold_synonyms"],
                    needle_text=c["needle_text"],
                    distractors=c["distractors"],
                    position_frac=pos,
                    target_total_tokens=tokens,
                ))
    return cases

def main(cases_path: str, out_path: str, generator, count_tokens):
    cases_pool = json.load(open(cases_path))
    rows = run(grid(cases_pool), generator, count_tokens)
    pathlib.Path(out_path).write_text(json.dumps(rows, indent=2))
```

Typical grid, 20 base cases × 4 context sizes × 7 positions = 560
runs. Scale up or down as compute allows; the U-curve is visible
at ~200 runs if the base cases are balanced.

### 2.4 `src/ordering_ablation.py` — ordering on real data

Where `needle_eval.py` measures the *phenomenon* on synthetic
contexts, this script measures the *fix* on real data. It reuses the
assembler from exercise-04 and runs the mod-109 eval slice under
each ordering policy:

```python
"""Compare ordering policies on the real eval slice."""
from __future__ import annotations
import json, pathlib
from assemble import assemble, AssembleConfig
from order    import OrderPolicy

def ablate(eval_cases, retrieve_fn, rerank_fn, count_tokens,
           generator, policies=(OrderPolicy.SCORE_DESCENDING,
                                OrderPolicy.SOURCE_ORDER,
                                OrderPolicy.U_SHAPE)):
    out = []
    for policy in policies:
        correct, total = 0, 0
        for case in eval_cases:
            candidates = retrieve_fn(case["question"], N=50)
            reranked   = rerank_fn(case["question"], candidates, K=10)
            assembled  = assemble(
                reranked, count_tokens,
                AssembleConfig(policy=policy, max_tokens=4000,
                               run_cosine_dedup=False),
            )
            answer = generator(build_prompt(case["question"], assembled.context_block))
            if any(s.lower() in answer.lower() for s in case["gold_synonyms"]):
                correct += 1
            total += 1
        out.append({"policy": policy.value, "accuracy": correct / total})
    return out

def build_prompt(question: str, context_block: str) -> str:
    return (
        "Answer the question using only the documents below. Cite the "
        "most relevant [chunk:...] marker inline after the answer.\n\n"
        f"Question: {question}\n\nDocuments:\n{context_block}\n\nAnswer:"
    )
```

### 2.5 `notes/results.md` — the three plots

Numbers below are **illustrative**, on the shape and size one typically
sees. Reproduce on the actual generator; absolute values are
model-dependent.

**(a) Needle position vs. accuracy (at 32k target tokens):**

```
position_frac   accuracy
0.00              0.94
0.10              0.90
0.25              0.82
0.50              0.73   ← trough
0.75              0.80
0.90              0.89
1.00              0.95
```

The U-curve is visible. The trough is around position 0.5; endpoints
are ~0.21 accuracy points higher than the midpoint on this
illustrative row.

**(b) Total tokens vs. midpoint accuracy (needle at 0.5):**

```
target_tokens   midpoint_accuracy   p95_latency
2 000              0.88                 ~900 ms
8 000              0.82                ~1600 ms
32 000             0.73                ~3800 ms
128 000            0.55                ~9200 ms
```

Reading: midpoint accuracy drops as the context grows. Latency grows
roughly linearly to super-linearly in context tokens. "More context
is always better" is false on both axes — accuracy *and* latency.

**(c) Ordering policy on the mod-109 eval slice (K=10, no dedup variation):**

```
policy               accuracy    Δ vs score_desc
score_descending       0.74            —
source_order           0.70         -0.04
u_shape                0.77         +0.03
```

U-shape beats score-descending by a small but defensible margin on
this generator. `source_order` lags both: narrative continuity does
not compensate for burying the strongest chunk in the middle, except
on queries where the chunks were from a single document and the
answer required reading them in sequence (see tail analysis).

### 2.6 `notes/memo.md` — one-page recommendation

**Audience.** The prompt-builder (mod-108) and the assembler
(exercise-04) owners.

**Recommendation.** Default assembler ordering is `U_SHAPE`. Default
`K` is bounded by the chosen generator's context window **and** by
the per-query token budget set in mod-108 (not by what the model
*can* ingest).

**Three concrete policies:**

1. **Do not fill the context window with distractors.** The
   measured accuracy drop between 8k and 128k tokens on this
   generator (illustrative row: 0.82 → 0.55 at midpoint) exceeds
   what any ordering policy recovers. Compress upstream
   (exercise-03) and keep `K` small.
2. **Place the strongest chunks at the ends.** On generators where
   the U-curve is measurable, U-shape ordering buys a few accuracy
   points for free (no additional retrieval call, no additional
   tokens). On generators where the curve is flat, it does not
   hurt.
3. **Reserve `SOURCE_ORDER` for narrative-continuity queries.**
   Legal clauses, numbered procedures, multi-paragraph policy
   excerpts: these lose coherence under score-order. Expose the
   policy as a per-query override, not as a global default.

**Non-recommendation.** Do not switch to a longer-context generator
as a response to retrieval misses. A longer context masks a
retrieval problem for a while and makes the system slower and more
expensive; the retrieval problem returns the day the corpus doubles.

**Trigger to re-evaluate.** Rerun the needle sweep when: (a) the
generator changes (even a minor-version bump), (b) the chunk-size
distribution changes materially (mod-102), (c) the assembler's
max-tokens budget changes, or (d) a new ordering policy is proposed
upstream.

### 2.7 How to read a flat U-curve

Some modern long-context models show a *nearly* flat U-curve on
simple needle tests at ≤32k tokens. Two warnings:

- **Simple needle tests over-report robustness.** A single fact
  embedded in random text is the easiest case. Multi-fact questions,
  negations, and conflicting evidence show the U-curve clearly
  even on generators that look flat on single-needle tests. The
  exercise's base cases should include at least one case per class.
- **Flat at short context ≠ flat at long context.** The U-curve
  reappears as context grows. Design and measure at the context
  length you will actually ship.

If the generator is actually flat on this harness at the deployment
context length, the recommendation collapses to "use
`SCORE_DESCENDING` by default" — one less thing to reason about.
Write the memo either way.

## 3. Validation steps

1. The U-curve is visible on at least one measured row (context
   size × model). A learner who reports "no U-curve" without the
   per-position accuracy plot has either a bug in the harness or a
   generator that genuinely has the effect under control — the
   report needs the plot either way.
2. The sweep uses a **fixed distractor pool and seed**. Two runs on
   the same grid produce the same prompts.
3. Accuracy is measured with gold + synonyms, not an LLM judge.
   Judge-based accuracy conflates the phenomenon with the judge's
   bias.
4. The context-size axis extends to the generator's actual context
   window, not stops at 8k. Measuring only short contexts and
   writing a memo about long contexts is unsound.
5. The ordering-policy ablation is on the real eval slice from
   mod-109, not on the needle set. The needle set shows the
   phenomenon; the real slice tells you whether the fix moves the
   business metric.
6. The memo recommends one policy and one trigger for re-evaluation.
   An "it depends" memo is not a decision.
7. Latency is reported alongside accuracy. A policy that improves
   accuracy by 1 point at 10× latency is a different trade-off
   than a free gain.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Phenomenon | The U-curve is demonstrated on the deployed generator at the deployed context length, not asserted from a 2023 paper. |
| Grid | Context size × position grid covers at least 3 context sizes and 5 positions, with the deployed context length included. |
| Metric | Exact match plus synonym list. No LLM judge in the loop at this stage. |
| Ablation | Ordering-policy ablation is on the real eval slice (mod-109), not only on synthetic needles. |
| Memo | Names one default ordering policy, one exception, and one trigger for re-evaluation. |
| Latency | p95 latency is reported alongside accuracy for each context size. |
| Honesty about flat curves | If the U-curve is flat on this harness at the deployment context, the memo says so and simplifies the recommendation. |
| Governance | The memo is explicit that "use a longer context" is not a substitute for better retrieval. |
| Reproducibility | Fixed seed and distractor pool. Two runs produce identical prompts (and, up to generator nondeterminism, nearly identical answers). |
| Test depth | At least one multi-fact / negation / conflicting-evidence case in the base pool, not only single-needle cases. |

Fail-fast smells:

- Measuring only at 8k tokens and writing a memo about 128k.
- Reporting "accuracy = 82%" at one position and declaring the
  effect gone, without the full position sweep.
- Changing distractors between runs. The variance will drown the
  position effect and the sweep will look flat for the wrong
  reason.
- A memo that recommends "use the longest-context model." That is
  a budget conversation, not a conclusion from this exercise.
- An ordering ablation on the needle set only. The needle set is
  the diagnostic; the fix lives on the real queries.

## 5. Common mistakes

- **Treating context length as a free upgrade.** A 128k-token
  context is not 16× better than an 8k-token context; often it is
  worse on accuracy, slower, and more expensive.
- **Using single-needle tests as the only phenomenon measurement.**
  Single-needle is the easy case. Include multi-fact and
  conflicting-evidence cases.
- **Random distractors per run.** Needle position is one variable
  the sweep *controls*; distractors must be fixed so the position
  signal is not drowned.
- **LLM-judge accuracy at the phenomenon stage.** The judge bias
  interacts with position bias (the judge also uses a context
  window). Use deterministic matching here; save LLM judging for
  mod-109.
- **Measuring only accuracy, ignoring latency.** Context length
  trades generator latency; a memo that ignores latency makes a
  partial recommendation.
- **Reading one row's flat curve as "the effect doesn't apply."**
  The effect reappears at longer context lengths and on harder
  question classes. Measure across both axes.
- **Deciding ordering by the needle set.** The needle set is
  synthetic and uniform. Decide on the real slice.
- **Shipping a long-context model as a retrieval fix.** A corpus
  that outgrew the first retriever will outgrow the longer context
  too. Fix retrieval.
- **Unfixed seeds.** `random.choice(distractors)` without a seed
  means the sweep is irreproducible. Audit will catch this.
- **No memo, just plots.** Plots describe. The memo is the
  decision; without it, the exercise taught the phenomenon but did
  not change what the team ships.

## 6. References

- The phenomenon:
  - Liu et al. (2023), "Lost in the Middle: How Language Models Use
    Long Contexts" —
    [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172).
- Long-context evaluation methodology:
  - Kamradt (2023), "Needle In A Haystack" harness pattern —
    [github.com/gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack).
  - Hsieh et al. (2024), "RULER: What's the Real Context Size of
    Your Long-Context Language Models?" —
    [arxiv.org/abs/2404.06654](https://arxiv.org/abs/2404.06654).
  - Zhang et al. (2024), "∞Bench: Extending Long Context Evaluation
    Beyond 100K Tokens" —
    [arxiv.org/abs/2402.13718](https://arxiv.org/abs/2402.13718).
- Upstream modules' handoff:
  - mod-107 exercise-04 assembler — citation anchors and ordering
    policies this exercise ablates on.
  - mod-108 — prompt building that consumes the assembler's
    context block.
  - mod-109 — retrieval-quality evaluation harness that this
    exercise's real-slice numbers come from.
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
