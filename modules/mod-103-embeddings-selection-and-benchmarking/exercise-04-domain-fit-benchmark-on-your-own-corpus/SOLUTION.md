# Exercise 04 — Solution: Domain-fit benchmark on your own corpus

This is a reference solution for
`exercise-04-domain-fit-benchmark-on-your-own-corpus.md`. It is an **implementation**
exercise. The learner builds a small, labeled evaluation set from their own corpus
(the one assembled in mod-101 / mod-102) and runs it against three or more candidate
embedders to decide which one actually fits the domain — because the MTEB leaderboard
numbers from exercise-02 are *not* a decision, only a starting shortlist. The
reference below gives runnable code to build the eval set, score models, and report.

## 1. Solution overview

The deliverable has four parts:

1. **`data/evalset/queries.jsonl` + `data/evalset/qrels.tsv`** — a BEIR-style labeled
   evaluation set over the learner's own corpus. 30–100 queries with 1–5 known
   relevant chunk ids per query. Labels come from the learner (or a stand-in SME),
   not an LLM alone.
2. **`src/eval/run_domain_benchmark.py`** — a script that evaluates a configurable list
   of embedders against that eval set and reports NDCG@10, Recall@10, MAP@10, and
   mean query latency.
3. **`results/domain-fit.csv` and `docs/domain-fit.md`** — the output table and a
   written analysis that names the winning model for this corpus and explicitly
   compares its *domain* number with its *MTEB* number.
4. **`notes/calibration.md`** — a short note on label quality: how many queries, how
   labels were produced, inter-annotator agreement (even a rough two-pass check), and
   which queries were thrown out for being too easy or too ambiguous.

The exercise's teaching point is that the public leaderboard is a poor proxy for
"which model will work on *my* text." A 60-query domain set with handmade labels is
worth more than a ranking of 50 generic datasets.

## 2. Worked answer or implementation

### 2.1 Eval set construction

**Target size.** Use 30–100 queries. Fewer than 30 and the confidence interval on
NDCG@10 is too wide to distinguish models. Many more than 100 and the SME labor
dominates and the exercise becomes about labeling, not retrieval.

**Query style.** Mix three shapes in roughly equal proportions:
- *Keyword-flavored:* short, lexical ("KMS key rotation").
- *Natural question:* how/what/why sentences ("how do I rotate a KMS key without
  downtime?").
- *Multi-hop or multi-aspect:* questions requiring two chunks ("compare key rotation
  in KMS vs. HashiCorp Vault").

**Relevance labels.** Binary (relevant / not-relevant) is sufficient for NDCG@10 on
small sets. Graded labels (0/1/2) are better for re-ranker exercises (mod-107) but
overkill here. For each query, the learner records the chunk ids that are *actually*
relevant. Non-exhaustive labels are the norm — most IR benchmarks are non-exhaustive
— but the labeling note must say so.

**LLM-assist with human review.** It is acceptable to seed queries and candidate
relevant chunks with an LLM (GPT-4-class) over the chunked corpus, but every label
must be reviewed by a human before entering the gold set. Pure LLM labels are a
confounder; the point of the exercise is to measure against *your* judgment, not the
LLM's.

### 2.2 File formats (BEIR-compatible, deliberately)

`data/evalset/corpus.jsonl` — one line per chunk:

```json
{"_id": "chunk-00001", "title": "KMS key rotation runbook", "text": "..."}
```

`data/evalset/queries.jsonl` — one line per query:

```json
{"_id": "q-001", "text": "How do I rotate a KMS key without downtime?"}
```

`data/evalset/qrels.tsv` — TSV with header: `query-id`, `corpus-id`, `score`:

```
query-id	corpus-id	score
q-001	chunk-00043	1
q-001	chunk-00044	1
q-002	chunk-00010	1
```

Using BEIR's schema means the same script that runs MTEB tasks in exercise-02 can run
*this* task — the learner is not learning a new file format.

### 2.3 `src/eval/run_domain_benchmark.py`

```python
# src/eval/run_domain_benchmark.py
"""
Evaluate candidate embedders on the learner's domain corpus.

Reads a BEIR-style corpus/queries/qrels triple and reports NDCG@10, Recall@10,
MAP@10, plus p50 / p95 query latency. Writes a CSV and a Markdown table.

Usage:
    uv run python -m src.eval.run_domain_benchmark \
        --data data/evalset \
        --models BAAI/bge-small-en-v1.5 intfloat/e5-base-v2 \
                 sentence-transformers/all-MiniLM-L6-v2 \
        --out results/domain-fit
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import time
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse the templates from exercise-02 so we don't silently run E5 without "query: ".
from scripts.run_mteb_task import PrefixedST, _TEMPLATES  # noqa: F401


def _read_corpus(p: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_qrels(p: pathlib.Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with p.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def _ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    gains = [relevant.get(doc, 0) for doc in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return 0.0 if idcg == 0 else dcg / idcg


def _recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    n_rel = sum(1 for v in relevant.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hit = sum(1 for doc in ranked[:k] if relevant.get(doc, 0) > 0)
    return hit / n_rel


def _map_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    n_rel = sum(1 for v in relevant.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, doc in enumerate(ranked[:k], start=1):
        if relevant.get(doc, 0) > 0:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / n_rel


def evaluate(model_name: str, corpus: list[dict], queries: list[dict],
             qrels: dict[str, dict[str, int]], k: int = 10) -> dict:
    model = PrefixedST(model_name)

    # 1. Embed corpus once.
    corpus_vecs = model.encode_corpus(corpus, batch_size=64)
    corpus_vecs = np.asarray(corpus_vecs, dtype=np.float32)
    corpus_ids = [d["_id"] for d in corpus]

    # 2. Embed queries and score each one; record per-query latency.
    ndcgs, recalls, maps = [], [], []
    latencies: list[float] = []

    query_texts = [q["text"] for q in queries]
    for q, qtext in zip(queries, query_texts):
        t0 = time.perf_counter()
        qvec = model.encode_queries([qtext], batch_size=1)[0].astype(np.float32)
        scores = corpus_vecs @ qvec
        order = np.argsort(-scores)[: max(k, 50)]
        latencies.append(time.perf_counter() - t0)

        ranked = [corpus_ids[i] for i in order.tolist()]
        rel = qrels.get(q["_id"], {})
        ndcgs.append(_ndcg_at_k(ranked, rel, k))
        recalls.append(_recall_at_k(ranked, rel, k))
        maps.append(_map_at_k(ranked, rel, k))

    return {
        "model": model_name,
        "ndcg_at_10": statistics.mean(ndcgs),
        "recall_at_10": statistics.mean(recalls),
        "map_at_10": statistics.mean(maps),
        "query_latency_p50_ms": 1000 * statistics.median(latencies),
        "query_latency_p95_ms": 1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "n_queries": len(queries),
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/domain-fit"))
    args = p.parse_args(argv)

    corpus = _read_corpus(args.data / "corpus.jsonl")
    queries = _read_corpus(args.data / "queries.jsonl")
    qrels = _read_qrels(args.data / "qrels.tsv")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(m, corpus, queries, qrels) for m in args.models]

    # CSV
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md = ["| model | n_queries | NDCG@10 | Recall@10 | MAP@10 | p50 ms | p95 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['n_queries']} | {r['ndcg_at_10']:.3f} | "
                  f"{r['recall_at_10']:.3f} | {r['map_at_10']:.3f} | "
                  f"{r['query_latency_p50_ms']:.1f} | {r['query_latency_p95_ms']:.1f} |")
    (args.out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
```

### 2.4 `docs/domain-fit.md` — reporting template

```markdown
# Domain-fit benchmark — <YOUR CORPUS NAME>

Date: <YYYY-MM-DD>
Corpus size: <N chunks, X MB>
Eval set size: <M queries, Y labels>
Labeling notes: see notes/calibration.md

## Results

| model | NDCG@10 (domain) | NDCG@10 (MTEB avg) | gap | p95 latency | notes |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | ... | ... (from leaderboard) | ... | ... | ... |
| e5-base-v2 | ... | ... | ... | ... | ... |
| all-MiniLM-L6-v2 | ... | ... | ... | ... | ... |

## Interpretation

- The winning model on *this* corpus is X, with NDCG@10 = ...
- The MTEB ranking order was (A, B, C); the domain ranking order was ...
- Why the orders differ: ...
- Would this result change the decision in exercise-03? ...
```

### 2.5 `notes/calibration.md` — label quality

The learner must document, with honesty about bias:

- **Number of queries**, by shape mix (keyword / NL / multi-hop).
- **Who labeled**, and how (single annotator, two-pass, or LLM-seeded + human
  reviewed). Record the LLM model used for seeding if applicable.
- **Rejected queries.** A query whose "relevant chunk" turns out not to actually
  cover the answer should be removed from `qrels.tsv` — write down how many and
  why.
- **Inter-annotator agreement** if two people labeled the same subset. Even a
  rough Cohen's κ on a 20-query sample is better than none.
- **Non-exhaustive labels disclaimer.** Explicitly say that the qrels are
  non-exhaustive; models may retrieve relevant chunks that nobody labeled — this
  shows up as understated Recall@k. This is the same behavior that BEIR and MS
  MARCO carry; naming it keeps the numbers honest.

## 3. Validation steps

1. `data/evalset/{corpus.jsonl,queries.jsonl,qrels.tsv}` all parse without error.
2. `uv run python -m src.eval.run_domain_benchmark --data data/evalset
   --models bge-small-en-v1.5 intfloat/e5-base-v2 all-MiniLM-L6-v2 --out
   results/domain-fit` finishes and produces both `results.csv` and `results.md`.
3. The table shows at least three models. All NDCG@10 values are in `[0, 1]` and
   not all identical — if two models give identical NDCG@10 to three decimal places,
   the embedding step was almost certainly skipped for one of them.
4. For at least one model, the domain NDCG@10 differs from the MTEB-leaderboard
   NDCG@10 by > 0.05 (five NDCG points). This is the actual finding of the exercise:
   leaderboards and your corpus disagree.
5. `notes/calibration.md` names a label count, a labeling method, and a
   non-exhaustive-labels disclaimer.
6. Re-running the script on the same machine with the same models reproduces NDCG@10
   to three decimal places.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Eval set size | 30 ≤ `n_queries` ≤ 100; query shapes mixed (not all keyword or all NL). |
| Eval set format | BEIR-style; `qrels.tsv` is tab-separated with a header. |
| Templates | Per-model prefixes applied via `encode_queries` / `encode_corpus` — no silent template drop. |
| Metrics | NDCG@10, Recall@10, MAP@10 all reported with 3-decimal precision. |
| Latency | p50 and p95 latency per query reported, measured locally. |
| Ranking vs. MTEB | The domain-ranking order is stated and compared to MTEB; at least one model has a non-trivial gap (> 5 NDCG points). |
| Calibration note | Labels, labelers, agreement check, and non-exhaustive disclaimer all present. |
| Honesty | If an LLM was used for label seeding, the LLM name and prompt are logged; nobody signs off on a leaderboard where an LLM also judged its own retrievals. |

## 5. Common mistakes

- **Letting an LLM both label and judge.** If GPT-4 generated the queries and also
  generated the relevance labels for those queries, the benchmark is measuring the
  LLM's consistency with itself, not the retriever's quality. A human has to touch
  the qrels.
- **A corpus with too many near-duplicates.** NDCG@10 saturates (every model gets
  ~0.95) and ranking becomes meaningless. De-dupe aggressively before building the
  eval set — this is the chapter-7 content-hash story paying off.
- **Query / label skew toward the training corpus.** If the queries all say "what is
  X" and the corpus happens to have sections titled "What is X", a lexical model will
  dominate for the wrong reason. Mix the three query shapes in `queries.jsonl`.
- **Measuring latency without warm-up.** The first query includes model-load and
  framework-import time — tens of seconds in some stacks. Discard the first query
  from the latency percentiles or run a warm-up call before the timed loop.
- **Reusing queries from the fine-tune set.** If a candidate model was fine-tuned on
  the organization's own ticket archive, the eval queries cannot come from that
  archive without being split (train / dev / test). This exercise does not fine-tune,
  but the discipline matters once fine-tuning is on the table (see exercise-06).
- **Confusing per-query score with corpus-wide mean.** NDCG@10 is a per-query score;
  the reported value is a mean over queries, not over documents. Reporting the max
  NDCG@10 ("some queries got 1.0!") is not a benchmark.
- **Hiding non-exhaustive labels.** Claiming Recall@10 = 0.72 as a tight number
  without the "labels are non-exhaustive" line misleads downstream readers. A
  non-exhaustive Recall is an under-estimate; name that.

## 6. References

- Benchmark formats and conventions:
  - BEIR — Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models" — [arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663); repo and file formats — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - MTEB — [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb), leaderboard [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  - MS MARCO — [microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/) (the canonical "non-exhaustive labels" benchmark; good prior art on how to talk about missing qrels).
- Metric definitions (standard IR textbooks and references):
  - Manning, Raghavan, Schütze — "Introduction to Information Retrieval" chapters on NDCG, MAP, Recall, Precision — [nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/).
- Tooling:
  - Sentence-Transformers — [sbert.net](https://sbert.net/).
  - Hugging Face `datasets` (for loading eval files as a dataset if the learner prefers that API) — [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets).
- Risk / governance framings for domain-fit decisions:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
