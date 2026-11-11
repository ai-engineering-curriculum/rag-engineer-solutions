# Exercise 02 — Solution: MTEB / BEIR critical read + local task run

This is a reference solution for
`exercise-02-mteb-and-beir-critical-read-and-local-task-run.md`. It is a **mixed**
exercise: the learner reads two well-known benchmark leaderboards (MTEB and BEIR) with
a skeptical eye, then runs one MTEB retrieval task locally against two embedding models
so the numbers on their own machine can be compared against the published numbers. The
reference below walks the critical-read with the specific traps to look for, then shows
the minimum viable local run.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/leaderboard-critical-read.md`** — a short written critique of the top
   handful of models on MTEB / BEIR that answers five questions per model: parameter
   count, embedding dimensionality, training-data overlap with the eval task, license,
   and known caveats (e.g., prefix / instruction templates that must be applied at
   inference time).
2. **`scripts/run_mteb_task.py`** — a runnable script that evaluates two user-chosen
   models on one MTEB retrieval task (default: `SciFact`, because it is small enough
   to finish on a laptop CPU in minutes and is one of the original BEIR datasets).
3. **`docs/mteb-local-vs-published.md`** — the local NDCG@10 numbers compared against
   the published numbers on the MTEB leaderboard, with a one-paragraph explanation for
   any gap > 1.5 NDCG points.

The learner's goal is not to match the leaderboard exactly — it is to be able to
*explain* any gap. Gaps usually come from four places: forgetting the required
query/passage prefix, using a different tokenizer, running at a shorter max length than
the leaderboard config, or using fp16 on a model that was evaluated in fp32.

## 2. Worked answer or implementation

### 2.1 `notes/leaderboard-critical-read.md` — what to look for

For every candidate from the leaderboard, record five columns:

| Model | Params (M) | Output dim | Training-data overlap with target task | License | Required inference template |
|---|---|---|---|---|---|
| `BAAI/bge-large-en-v1.5` | ~335 | 1024 | Mixed public corpora; contrastive SFT on MS MARCO & other QA pairs. Known to be trained on data overlapping BEIR tasks. | MIT | For retrieval: prepend `"Represent this sentence for searching relevant passages: "` to queries; passages passed as-is. |
| `BAAI/bge-small-en-v1.5` | ~33 | 384 | Same as bge-large. | MIT | Same prefix as bge-large. |
| `intfloat/e5-base-v2` | ~110 | 768 | Contrastive pre-training on web pairs + SFT on MS MARCO / NQ / TriviaQA. | MIT | Query: prefix with `"query: "`. Passage: prefix with `"passage: "`. |
| `intfloat/multilingual-e5-base` | ~278 | 768 | Multilingual mC4 pairs + SFT. | MIT | Same `"query: " / "passage: "` prefix. |
| `sentence-transformers/all-MiniLM-L6-v2` | ~22 | 384 | 1B sentence pairs from public corpora. | Apache-2.0 | No prefix required. |
| `sentence-transformers/all-mpnet-base-v2` | ~110 | 768 | Same corpus family as MiniLM-L6-v2. | Apache-2.0 | No prefix required. |

The five traps to call out explicitly in the critique:

1. **Training-set / eval-set overlap.** Several of the top MTEB rows are models whose
   contrastive fine-tune included MS MARCO, HotpotQA, NQ, TriviaQA, or FiQA — which
   are also MTEB retrieval tasks. The leaderboard numbers are closer to "fit" than
   "generalization" on those tasks. BEIR (the retrieval subset of MTEB) was designed
   to test *out-of-domain* generalization; models that were trained on BEIR's training
   splits are not evaluating zero-shot. Always read the model card for a declared
   training-corpus list before citing its leaderboard rank.
2. **Required prefix / instruction templates.** E5 requires `"query: " / "passage: "`;
   BGE requires a query-side prefix for retrieval; several instruction-tuned embedders
   (Instructor, GTE-instruct, SFR-Embedding-Mistral, nomic-embed-text-v1.5)
   require a task-specific instruction string. Forgetting the template can cost 5–20
   NDCG@10 points — and the model will not throw an error, it will silently return
   a worse number. The critique must call out the template per model.
3. **Dimensionality is a cost choice, not an accuracy choice.** `bge-large` and
   `bge-small` land in similar ballparks on many retrieval tasks; the large model is
   typically a few NDCG points better and ~10× the storage. The decision belongs in
   the hosted-vs-self-hosted sheet (exercise-03), not in a leaderboard ranking.
4. **Max-length truncation.** Most of these models are trained with 512 tokens.
   `nomic-embed-text-v1.5`, `jina-embeddings-v2`, and several newer models advertise
   8192 tokens. If the local run uses 512 while the leaderboard config used 8192, the
   local numbers on long-passage tasks (e.g., MTEB `ArguAna`) will look worse.
5. **Precision and pooling.** Some leaderboards use fp32; fp16 is close but not
   identical. Some use CLS pooling, some use mean pooling with attention-mask; the
   library default can silently flip. For Sentence-Transformers models, call
   `SentenceTransformer(...).encode(...)` and let the library do the right pooling;
   do not roll your own pool.

### 2.2 `scripts/run_mteb_task.py` — minimum viable local run

```python
# scripts/run_mteb_task.py
"""
Run one MTEB retrieval task against two Sentence-Transformers models and print
NDCG@10, Recall@10, MAP@10 side by side. Default task: SciFact (small, well-known).

Usage:
    uv run python -m scripts.run_mteb_task \
        --models BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 \
        --task SciFact \
        --output-dir results/mteb
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import mteb
from sentence_transformers import SentenceTransformer


# Known prefix / template rules we need to honor at inference.
# MTEB handles this for models that implement `encode_queries` / `encode_corpus`
# correctly, but we document the templates here so a learner can see them.
_TEMPLATES = {
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "BAAI/bge-large-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "intfloat/e5-base-v2": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "intfloat/e5-small-v2": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query_prefix": "",
        "passage_prefix": "",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "query_prefix": "",
        "passage_prefix": "",
    },
}


class PrefixedST:
    """Thin adapter that applies the correct template at encode time.

    MTEB's evaluator calls `encode_queries` and `encode_corpus` separately, which
    is exactly where the model-specific prefix must go.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        template = _TEMPLATES.get(model_name, {"query_prefix": "", "passage_prefix": ""})
        self.query_prefix = template["query_prefix"]
        self.passage_prefix = template["passage_prefix"]

    def encode_queries(self, queries: Iterable[str], batch_size: int = 32, **kw):
        texts = [f"{self.query_prefix}{q}" for q in queries]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kw)

    def encode_corpus(self, corpus, batch_size: int = 32, **kw):
        # MTEB corpus items are dicts: {"title": ..., "text": ...}
        texts = [f"{self.passage_prefix}{(d.get('title','') + ' ' + d.get('text','')).strip()}"
                 for d in corpus]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kw)


def run(models: list[str], task_name: str, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    task = mteb.get_tasks(tasks=[task_name])
    results: dict[str, dict] = {}
    for name in models:
        evaluator = mteb.MTEB(tasks=task)
        model = PrefixedST(name)
        model_out = out_dir / name.replace("/", "__")
        model_out.mkdir(parents=True, exist_ok=True)
        r = evaluator.run(model, output_folder=str(model_out), eval_splits=["test"])
        # MTEB returns a list of per-task dicts; collapse to a single row.
        for entry in r:
            scores = entry.scores.get("test", [])
            if not scores:
                continue
            row = scores[0]
            results[name] = {
                "ndcg_at_10": row.get("ndcg_at_10"),
                "recall_at_10": row.get("recall_at_10"),
                "map_at_10": row.get("map_at_10"),
            }
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--task", default="SciFact")
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/mteb"))
    args = p.parse_args(argv)
    r = run(args.models, args.task, args.output_dir)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

Pinned dependencies (use the current versions at run time; pins drift fast):

```
uv add mteb sentence-transformers torch datasets
```

Expected wall-clock on a modern laptop CPU for `SciFact`:
- `all-MiniLM-L6-v2`: a few minutes.
- `bge-small-en-v1.5`: a few minutes.
- `bge-large-en-v1.5`: tens of minutes to an hour (not recommended on CPU unless the
  learner wants to see the latency cliff).

### 2.3 `docs/mteb-local-vs-published.md` — the write-up template

```markdown
# Local MTEB run vs. published numbers

Task: SciFact (BEIR / MTEB retrieval)
Date: <YYYY-MM-DD>
Hardware: <CPU model / GPU / fp16 / fp32>
MTEB version: <paste pip show mteb Version>

## Results

| Model | Local NDCG@10 | Published NDCG@10 | Gap | Likely cause of gap |
|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | <local> | <from leaderboard at run date> | <delta> | <see below> |
| sentence-transformers/all-MiniLM-L6-v2 | <local> | <from leaderboard at run date> | <delta> | <see below> |

## Explanation of each non-zero gap

1. If gap ≤ 1.5 NDCG points: likely numerical / library-version noise. No action.
2. If gap is 2–5 points: check the prefix template; check normalization
   (`normalize_embeddings=True`); check the max-length cap in the tokenizer.
3. If gap is > 5 points and we used the correct template: check whether the published
   number is on `test` or `dev`; MTEB publishes test numbers but some blog-posted
   leaderboards quote dev splits.
```

A plausible, calibrated expectation for `SciFact`: NDCG@10 ≈ 0.69–0.74 for
`bge-small-en-v1.5` and ≈ 0.63–0.68 for `all-MiniLM-L6-v2`, in the right
ballpark. The learner's job is to report whatever they measure, not to match a
specific target — but a result in a completely different range (e.g., 0.20) means the
prefix was wrong or the corpus side was embedded without `title` concatenation.

## 3. Validation steps

1. `uv run python -m scripts.run_mteb_task --models BAAI/bge-small-en-v1.5
   sentence-transformers/all-MiniLM-L6-v2 --task SciFact`. Both models complete without
   raising.
2. `results/mteb/summary.json` exists and contains `ndcg_at_10`, `recall_at_10`, and
   `map_at_10` for each model.
3. The critical-read note in `notes/leaderboard-critical-read.md` names, for each of at
   least four leaderboard models, (a) the required inference template, (b) the
   training-set overlap flag, and (c) the license.
4. The gap write-up in `docs/mteb-local-vs-published.md` lists the local number, the
   published number, and a one-sentence cause when the gap > 1.5 points.
5. Re-running the script on the same machine produces the same numbers to within
   fp16/fp32 noise — if numbers swing by > 1 NDCG point on re-run, pooling or
   normalization is non-deterministic.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Critical read | For every cited leaderboard model, the template, training-set-overlap flag, and license are present; no more than one cell says "unknown." |
| Trap awareness | At least three of the five traps (training-data overlap, prefix templates, dimensionality cost, max-length truncation, pooling/precision) are explicitly discussed. |
| Local run script | Applies the correct per-model prefix via `encode_queries` / `encode_corpus`; uses `normalize_embeddings=True`; task name parameterized. |
| Numbers landed | `summary.json` has non-null NDCG@10 for both models; MiniLM is lower than BGE on SciFact (sanity sign — the opposite is a template bug). |
| Gap explained | Each > 1.5-point gap vs. published leaderboard has a cited cause (template, precision, max-length, or split). No gap-explanation is "we're not sure." |
| Honesty about CPU vs. GPU | The write-up records which model was run on CPU; nobody claims `bge-large` numbers on a laptop in 90 seconds without GPU. |
| Reproducibility | MTEB version and library versions pinned; re-running the same script on the same hardware reproduces the numbers. |

## 5. Common mistakes

- **Running E5 without `"query: "` / `"passage: "`.** The model will not throw. It will
  return an embedding. That embedding will be worse than if the prefix had been
  applied. The learner may see NDCG@10 ≈ 0.40 instead of ≈ 0.70 on SciFact. There is no
  warning — the only way to catch this is to know the template and apply it.
- **Running BGE without the query-side prefix.** Same silent-degradation pattern as E5,
  usually a 5–15 NDCG-point hit depending on the task.
- **Embedding the corpus side as `text` only, ignoring `title`.** MTEB / BEIR corpora
  are `{"title", "text"}` dicts; title is a strong signal in `SciFact`, `NFCorpus`,
  `TRECCOVID`, and others. The reference encoder in this exercise concatenates
  `title + " " + text`, matching the MTEB evaluator's default behavior.
- **Blaming the model for a gap caused by the library default.** If a learner's local
  number is well below the published number, the first thing to check is the template
  and normalization, not the model. "This model is bad" is almost never the right
  answer to a 10-point gap.
- **Trusting a MTEB ranking without reading the training-set list.** Many top-ranked
  retrieval models have seen their eval tasks during training. If the critique says
  "model X is #1 so we pick it," without a note on whether the training corpus
  overlaps our target task, the exercise has been skipped.
- **Picking a task that cannot complete on local hardware.** Full MTEB takes many hours
  on a workstation. The exercise picks one small retrieval task on purpose; swapping
  to `MSMARCO` or `HotpotQA` on a laptop will not finish, and the write-up will have
  no numbers.
- **Comparing against stale leaderboard numbers.** The MTEB leaderboard updates
  continuously. Pin the date you pulled the published number and record the MTEB
  version. "Matches leaderboard" without a date is uncheckable.

## 6. References

- Benchmarks:
  - MTEB — Muennighoff et al., "MTEB: Massive Text Embedding Benchmark" — [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316); library and leaderboard — [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb); leaderboard — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  - BEIR — Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" — [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663); repo — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - MS MARCO passage ranking (training data for many of the top models) — [microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/).
- Model cards (the single source of truth for inference templates):
  - [huggingface.co/BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
  - [huggingface.co/intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2), [huggingface.co/intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base).
  - [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
- Supporting libraries:
  - Sentence-Transformers — [sbert.net](https://sbert.net/).
- Security framings for benchmark-driven model selection (training-data leakage is a
  real supply-chain risk):
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
