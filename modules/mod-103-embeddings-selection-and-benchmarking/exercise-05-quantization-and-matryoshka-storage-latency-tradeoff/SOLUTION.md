# Exercise 05 — Solution: Quantization and Matryoshka — storage / latency / quality trade-off

This is a reference solution for
`exercise-05-quantization-and-matryoshka-storage-latency-tradeoff.md`. It is an
**implementation** exercise. The learner takes the winning embedder from exercise-04,
measures NDCG@10, storage, and query latency under four compression regimes
(full fp32, fp16, int8 scalar quantization, binary quantization) and two Matryoshka
truncations (full dim and half dim), then recommends a serving setting with the
trade-off explicit in writing.

## 1. Solution overview

The deliverable has four parts:

1. **`src/compression/`** — small utilities for fp16 / int8 / binary conversion of a
   `(N, d)` float32 matrix and the matching dot-product / Hamming-distance scoring.
2. **`src/eval/run_compression_sweep.py`** — runs the exercise-04 eval set under each
   compression regime × each Matryoshka dim and emits a CSV.
3. **`results/compression-sweep.csv` + `docs/compression-sweep.md`** — the table and
   the written recommendation.
4. **`notes/matryoshka-only-applies-when.md`** — a short note explaining which
   embedders actually support Matryoshka truncation and which do not; truncating the
   wrong embedder's output degrades quality much faster than a Matryoshka-trained
   model's.

The teaching point: compression and dim reduction have different quality cliffs. fp16
is almost free. int8 scalar quantization usually loses < 1 NDCG point on retrieval
tasks. Binary quantization is lossy but *recoverable* with a float-rescoring second
stage. Matryoshka truncation is only "free" on models that were explicitly trained
with Matryoshka Representation Learning; naive truncation of a non-Matryoshka model
loses quality fast.

## 2. Worked answer or implementation

### 2.1 Matryoshka support — not every embedder qualifies

**Matryoshka-trained (truncation is safe to ~half dim):**
- OpenAI `text-embedding-3-small` and `text-embedding-3-large` — the API exposes a
  `dimensions` parameter that cuts the output while preserving quality (the trained
  prefix dimensions are the informative ones).
- `nomic-ai/nomic-embed-text-v1.5` — the model card documents Matryoshka training and
  supports truncation to 512 / 256 / 128.
- Several newer families advertise Matryoshka training; always confirm against the
  model card before cutting.

**Not Matryoshka-trained (truncation is naive dim reduction; expect quality loss):**
- Most bge-* and e5-* checkpoints (unless the model card explicitly says otherwise).
- `sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`.

The sweep in this exercise reports two truncations: full dim (the baseline) and half
dim. On a Matryoshka-trained model the half-dim column should be within ~1 NDCG point
of the full-dim column. On a non-Matryoshka model, expect a larger drop — that is the
observation the exercise is asking the learner to see.

### 2.2 Compression utilities — `src/compression/`

```python
# src/compression/quantize.py
"""fp16 / int8 / binary (1-bit) conversion of a (N, d) float32 matrix.

Everything is numpy — no vector-DB dependencies. The scoring helpers show what
the DB's int8 / binary modes are doing under the hood.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def to_fp16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16, copy=False)


@dataclass
class Int8Pack:
    data: np.ndarray       # (N, d) int8
    scale: np.ndarray      # (d,)   float32 — per-dimension scale
    zero: np.ndarray       # (d,)   float32 — per-dimension zero point


def to_int8_per_dim(x: np.ndarray) -> Int8Pack:
    """Scalar quantization with a per-dimension min/max calibration.

    Works well for already-normalized embeddings, where per-dim ranges are narrow.
    """
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    scale = (mx - mn) / 255.0
    scale[scale == 0] = 1.0           # guard against dead dimensions
    zero = mn
    q = np.round((x - zero) / scale).astype(np.int16)
    q = np.clip(q, 0, 255) - 128      # store as signed int8 centered at 0
    return Int8Pack(data=q.astype(np.int8), scale=scale.astype(np.float32),
                    zero=zero.astype(np.float32))


def int8_dot(q: Int8Pack, v: np.ndarray) -> np.ndarray:
    """Score a batch of fp32 queries against an int8 corpus.

    Reconstructs on the fly; a real serving stack would keep the queries in fp32 and
    the corpus in int8 and use a BLAS int8 path. Here we optimize for clarity.
    """
    # (q.data + 128) * scale + zero  == original value
    deq = (q.data.astype(np.float32) + 128.0) * q.scale + q.zero
    return deq @ v


def to_binary(x: np.ndarray) -> np.ndarray:
    """1-bit sign quantization. Returns a (N, d/8) uint8 packed matrix."""
    bits = (x > 0).astype(np.uint8)
    return np.packbits(bits, axis=1)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance between a (N, d/8) packed matrix and one (d/8,) query."""
    xor = np.bitwise_xor(a, b)
    return np.unpackbits(xor, axis=1).sum(axis=1)
```

Matryoshka cut — this is "take the first `k` dims and renormalize." For a
Matryoshka-trained model, this preserves quality; for a non-Matryoshka model it does
not. The utility does the same math either way — the model-card check is what
determines whether it is safe:

```python
# src/compression/matryoshka.py
import numpy as np

def cut_and_renormalize(x: np.ndarray, k: int) -> np.ndarray:
    """Keep the first k dimensions and renormalize to unit length."""
    assert 0 < k <= x.shape[1], (k, x.shape)
    y = x[:, :k].astype(np.float32, copy=True)
    norm = np.linalg.norm(y, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return y / norm
```

### 2.3 `src/eval/run_compression_sweep.py`

```python
# src/eval/run_compression_sweep.py
"""Sweep compression × Matryoshka dim on the exercise-04 eval set.

For each (model, dim, compression) run NDCG@10, Recall@10, measure storage bytes
per vector, and measure p50/p95 query latency. Write a CSV.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import time
from typing import Iterable

import numpy as np

from src.eval.run_domain_benchmark import (
    _read_corpus, _read_qrels, _ndcg_at_k, _recall_at_k,
)
from src.compression.quantize import (
    to_fp16, to_int8_per_dim, int8_dot, to_binary, hamming,
)
from src.compression.matryoshka import cut_and_renormalize

# Reuse exercise-02's PrefixedST so templates stay honest.
from scripts.run_mteb_task import PrefixedST


COMPRESSIONS = ("fp32", "fp16", "int8", "binary")
DIM_FRACTIONS = (1.0, 0.5)


def _score_fp(corpus_vecs: np.ndarray, qvec: np.ndarray) -> np.ndarray:
    return corpus_vecs @ qvec


def _rescore(top_n_ids: list[int], corpus_fp32: np.ndarray, qvec_fp32: np.ndarray,
             top_k: int) -> list[int]:
    """Second-stage fp32 rescore of a shortlist — the standard binary recovery."""
    sub = corpus_fp32[top_n_ids]
    scores = sub @ qvec_fp32
    order = np.argsort(-scores)[:top_k]
    return [top_n_ids[i] for i in order.tolist()]


def _storage_bytes(dim: int, compression: str) -> int:
    if compression == "fp32":
        return 4 * dim
    if compression == "fp16":
        return 2 * dim
    if compression == "int8":
        return 1 * dim + 8  # +8 is the per-vector scale/zero amortized; approx
    if compression == "binary":
        return max(1, dim // 8)
    raise ValueError(compression)


def evaluate_config(model_name: str, corpus: list[dict], queries: list[dict],
                    qrels: dict[str, dict[str, int]], dim_fraction: float,
                    compression: str, rescore_shortlist: int = 100,
                    k: int = 10) -> dict:
    model = PrefixedST(model_name)

    # 1. Encode fp32 corpus + apply Matryoshka cut.
    full = np.asarray(model.encode_corpus(corpus, batch_size=64), dtype=np.float32)
    d = int(full.shape[1] * dim_fraction)
    corpus_fp32 = cut_and_renormalize(full, d)

    # 2. Compress corpus side.
    if compression == "fp16":
        corpus_cmp = to_fp16(corpus_fp32)
    elif compression == "int8":
        corpus_cmp = to_int8_per_dim(corpus_fp32)
    elif compression == "binary":
        corpus_cmp = to_binary(corpus_fp32)
    else:
        corpus_cmp = corpus_fp32

    corpus_ids = [doc["_id"] for doc in corpus]
    ndcgs, recalls = [], []
    latencies: list[float] = []

    for q in queries:
        qtext = q["text"]
        qvec_full = np.asarray(model.encode_queries([qtext]))[0].astype(np.float32)
        qvec = cut_and_renormalize(qvec_full.reshape(1, -1), d).reshape(-1)

        t0 = time.perf_counter()
        if compression == "fp16":
            scores = corpus_cmp.astype(np.float32, copy=False) @ qvec
            order = np.argsort(-scores)[: max(k, 50)]
            final_ids = order.tolist()
        elif compression == "int8":
            scores = int8_dot(corpus_cmp, qvec)
            order = np.argsort(-scores)[: max(k, 50)]
            final_ids = order.tolist()
        elif compression == "binary":
            qbin = to_binary(qvec.reshape(1, -1))[0]
            dist = hamming(corpus_cmp, qbin)
            short = np.argsort(dist)[:rescore_shortlist].tolist()
            final_ids = _rescore(short, corpus_fp32, qvec, k)
        else:
            scores = _score_fp(corpus_fp32, qvec)
            final_ids = np.argsort(-scores)[: max(k, 50)].tolist()
        latencies.append(time.perf_counter() - t0)

        ranked = [corpus_ids[i] for i in final_ids]
        rel = qrels.get(q["_id"], {})
        ndcgs.append(_ndcg_at_k(ranked, rel, k))
        recalls.append(_recall_at_k(ranked, rel, k))

    return {
        "model": model_name,
        "dim": d,
        "compression": compression,
        "ndcg_at_10": statistics.mean(ndcgs),
        "recall_at_10": statistics.mean(recalls),
        "storage_bytes_per_vec": _storage_bytes(d, compression),
        "latency_p50_ms": 1000 * statistics.median(latencies),
        "latency_p95_ms": 1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "rescore_shortlist": rescore_shortlist if compression == "binary" else None,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/compression"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    corpus = _read_corpus(args.data / "corpus.jsonl")
    queries = _read_corpus(args.data / "queries.jsonl")
    qrels = _read_qrels(args.data / "qrels.tsv")

    rows = []
    for frac in DIM_FRACTIONS:
        for cmp_name in COMPRESSIONS:
            rows.append(evaluate_config(args.model, corpus, queries, qrels,
                                        dim_fraction=frac, compression=cmp_name))

    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(rows)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
```

### 2.4 `docs/compression-sweep.md` — reporting template and expected shape

```markdown
# Compression sweep — <model name>

Date: <YYYY-MM-DD>
Corpus: <same corpus as exercise-04>
Baseline (fp32, full dim) NDCG@10: <X.XXX>

## Results

| dim | compression | NDCG@10 | ΔNDCG vs baseline | bytes/vec | storage ratio | p95 ms |
|---|---|---|---|---|---|---|
| full | fp32 | <baseline> | 0.000 | 4*d | 1.0× | ... |
| full | fp16 | ... | ... | 2*d | 0.5× | ... |
| full | int8 | ... | ... | d | 0.25× | ... |
| full | binary | ... (with fp32 rescore of top-100) | ... | d/8 | 0.031× | ... |
| half | fp32 | ... | ... | 2*d | 0.5× | ... |
| half | fp16 | ... | ... | d | 0.25× | ... |
| half | int8 | ... | ... | d/2 | 0.125× | ... |
| half | binary | ... (rescore) | ... | d/16 | 0.016× | ... |

## Recommendation

For this corpus and QPS tier, we recommend <...> because <...>. The dominant
factor in the decision is <storage / latency / p99 quality>.
```

**Calibrated expectation**, so a reviewer can tell when a run went off the rails:

- `fp16` should be within ~0.001 NDCG of `fp32`; a larger drop means the per-query
  renorm is mishandled.
- `int8` scalar quantization should cost < 1 NDCG point on retrieval; a drop > 3
  points means the calibration range is wrong (likely a non-normalized embedder or
  dead-dim scale).
- `binary + fp32 rescore of top-100` is well-studied and typically lands within
  1–2 NDCG points of `fp32` while cutting storage by ~32× (fp32 → 1 bit per dim).
  Without the rescore stage, binary alone drops multiple NDCG points; call that
  out in the write-up if the learner ran it both ways.
- **Half-dim behavior is model-dependent.** Matryoshka models (nomic-embed-text-v1.5,
  OpenAI `text-embedding-3-*`) should be close to full-dim. Non-Matryoshka models can
  lose several NDCG points at half dim. That contrast *is* the finding — the
  rubric below asks for it explicitly.

## 3. Validation steps

1. `uv run python -m src.eval.run_compression_sweep --data data/evalset --model
   <winning-model-from-exercise-04> --out results/compression` completes.
2. `results/compression/results.csv` has eight rows (4 compressions × 2 dim
   fractions).
3. Every NDCG@10 is in `[0, 1]`. `fp16@full` ≈ `fp32@full` to three decimal places.
4. Binary row with `rescore_shortlist=100` is substantially better than binary
   row without rescore (the learner should run both and show both, even though the
   reference script defaults to the rescored shape).
5. Storage column values are deterministic from dim and compression — a reviewer
   can recompute them from the formulas.
6. Recommendation in `docs/compression-sweep.md` names a specific regime and
   references one row from the table as the justifying number.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Baseline | fp32 full-dim NDCG@10 matches the exercise-04 number for the same model to three decimals. |
| fp16 | Quality indistinguishable from fp32 at this scale (< 0.002 NDCG). |
| int8 | < 1 NDCG-point drop; per-dim scale/zero are stored, not a single global scale. |
| Binary | Hamming + fp32 rescore of the top-N is implemented and used; raw binary (no rescore) is also reported for honesty. |
| Matryoshka | The learner has checked whether their model is Matryoshka-trained; `notes/matryoshka-only-applies-when.md` explicitly says so, and the half-dim numbers are interpreted in that light. |
| Storage | Bytes-per-vector column is computed, not guessed; matches dim × bytes-per-scalar + int8 scale overhead. |
| Latency | p50 and p95 reported for every regime; binary + rescore row carries the *total* latency including the rescore step. |
| Recommendation | Names one regime, one compression ratio, one quality cost — in one sentence. No "it depends" without a condition attached. |

## 5. Common mistakes

- **Treating half-dim as free for any embedder.** Only Matryoshka-trained models can
  be truncated to half dim without noticeable quality loss. If the sweep shows a
  big drop at half-dim and the learner concludes "Matryoshka doesn't work," the
  diagnosis is actually "this model is not Matryoshka — don't truncate it."
- **Binary quantization without rescoring.** Hamming-only retrieval is cheap but
  lossy. The industry pattern is binary first-stage recall over the whole corpus,
  then fp32 rescoring of the top 100–200. Reporting only the lossy number makes
  binary look much worse than it is in practice.
- **A single global int8 scale.** One scale/zero per *dimension* is standard for
  embedding quantization; one scale for the whole matrix throws away too much
  range. The reference uses per-dim scale and the sweep numbers reflect that.
- **Forgetting the rescore stage's own latency.** Binary + rescore is binary's
  latency *plus* an fp32 dot product over the shortlist. Report the total; the
  recommendation changes if the shortlist is 1000 and the rescore dominates.
- **Mis-measuring storage.** Vector DBs carry graph or inverted-list overhead on top
  of the raw vector bytes. The reference reports raw bytes-per-vector and notes
  that the on-disk size will be larger — see mod-104 for the measured index
  overhead.
- **Running the sweep against fewer than two dim fractions.** The whole point is the
  interaction between compression and dim; one dim collapses the sweep into a
  compression-only table.
- **Running on a corpus so small that NDCG@10 saturates.** If every regime returns
  NDCG@10 ≈ 1.0, the sweep is uninformative. Reuse the exercise-04 eval set
  (30–100 queries) at minimum.

## 6. References

- Methods:
  - Matryoshka Representation Learning — Kusupati et al., "Matryoshka Representation Learning" — [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147).
  - OpenAI embedding v3 (Matryoshka-capable `dimensions` parameter) — [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings).
  - Nomic Embed v1.5 model card (Matryoshka training, 8192-token max) — [huggingface.co/nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).
  - Cohere "Introducing int8 and binary embeddings" (binary + rescoring pattern) — [cohere.com/blog/int8-binary-embeddings](https://cohere.com/blog/int8-binary-embeddings).
- Vector-DB quantization docs (implementations of the same ideas in production):
  - Qdrant quantization guide — [qdrant.tech/documentation/guides/quantization/](https://qdrant.tech/documentation/guides/quantization/).
  - Weaviate PQ / BQ docs — [weaviate.io/developers/weaviate/configuration/compression](https://weaviate.io/developers/weaviate/configuration/compression).
  - Milvus vector quantization docs — [milvus.io/docs](https://milvus.io/docs).
- Tooling:
  - Sentence-Transformers — [sbert.net](https://sbert.net/).
  - NumPy `packbits`/`unpackbits` reference — [numpy.org/doc/stable/reference/generated/numpy.packbits.html](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
