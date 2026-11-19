# Exercise 04 — Solution: Multimodal Retrieval Survey — CLIP, SigLIP, and ColPali on the Same Corpus

This is a reference solution for
`exercise-04-multimodal-retrieval-survey-clip-siglip-colpali.md`.
It is a **design-plus-benchmark** exercise. The learner runs three
multimodal retrievers — CLIP, SigLIP, and ColPali — against the
same seeded corpus of images and PDF pages, records recall at
k and latency, and defends a pick (or a layered pick) for two
concrete scenarios: natural-image search and complex-document
retrieval. The deliverable is a benchmark harness, a results
table, and a per-scenario recommendation grounded in the
strengths each model was designed for.

The teaching point: "multimodal retrieval" is a bucket with
three different retrieval shapes inside it. CLIP and SigLIP are
**dual-encoder, single-vector** retrievers: one embedding per
image, one per text, cosine (or sigmoid) similarity. ColPali is
a **late-interaction, multi-vector** retriever designed for
document pages with mixed text, tables, and figures. The axes
that decide the pick are *what each pixel actually represents*
(a photograph? a page screenshot?), *whether page layout and
text content matter*, and *how big the index has to be*. A
practitioner who treats all three as interchangeable
"image embedders" is going to pick wrongly for at least one of
the two scenarios.

## 1. Solution overview

The deliverable has six parts:

1. **`corpus/`** — two small corpora with ground-truth
   relevance judgments:
   - `corpus/photos/` — ≥ 50 natural images with captions and
     a small set of text queries (e.g., "a cat on a windowsill")
     with known-relevant image IDs.
   - `corpus/pages/` — ≥ 20 PDF page screenshots (a mix of
     text, tables, and figure-heavy pages) with questions
     whose answers are on specific pages.
2. **`src/embedders.py`** — three embedders behind one common
   interface: `embed_queries`, `embed_docs`, and `score`. The
   first two return vectors (or vector sets for ColPali); the
   third hides the similarity function (cosine for CLIP, sigmoid
   for SigLIP, late-interaction MaxSim for ColPali).
3. **`src/benchmark.py`** — the harness. Computes recall@k and
   mean reciprocal rank on the labelled queries, measures
   per-query latency, and writes a results table.
4. **`src/index.py`** — the storage strategy per embedder.
   Single-vector models land in a standard vector DB (the
   reference uses an in-process FAISS flat index for small
   corpora). ColPali's multi-vector output needs either a
   late-interaction-aware store or an in-memory implementation
   for the exercise scale.
5. **`results/`** — the output artefacts: `results.csv` with
   per-query scores, `summary.md` with the aggregate table and
   the per-scenario recommendation.
6. **`notes/model_cards.md`** — one-paragraph *model-card*
   summaries sourced from the official releases, documenting
   training corpus, licence, intended use, and limitations.
   This is the artefact that supports the governance framing
   in §6.

Scope: *comparative evaluation, not state-of-the-art chasing.*
The exercise uses a small corpus, pinned model checkpoints, and
pinned library versions. If the harness runs end-to-end and the
numbers reproduce across runs with the same seed, the exercise
passes.

## 2. Worked answer or implementation

### 2.1 The three models in one paragraph each

- **CLIP (OpenAI).** A dual-encoder model trained on
  web-scraped image–text pairs with a symmetric
  contrastive objective (InfoNCE-style). One image → one
  embedding; one text caption → one embedding; similarity is
  cosine. Sweet spot: zero-shot natural-image retrieval and
  classification. Weakest on fine-grained text inside images
  (a page of prose, a dense table) because the single vector
  compresses the whole image. Official project:
  [github.com/openai/CLIP](https://github.com/openai/CLIP).
  Paper: Radford et al., "Learning Transferable Visual Models
  From Natural Language Supervision," 2021 —
  [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020).
- **SigLIP (Google).** A CLIP-shaped dual encoder trained with
  a *sigmoid* loss instead of softmax contrastive. The practical
  consequence is that training does not require the whole batch
  as a normalising set, which stabilises training on web-scale
  data and often improves accuracy at matched compute. Same
  retrieval shape as CLIP: one vector per image and per text,
  score is a learned bias-shifted dot product (pass through a
  sigmoid for a probability, or use the pre-sigmoid logit as a
  similarity score). Paper: Zhai et al., "Sigmoid Loss for
  Language Image Pre-Training," 2023 —
  [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343).
  Checkpoints via Hugging Face:
  [huggingface.co/google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224).
- **ColPali (ILLUIN Technology).** Built on PaliGemma; emits
  *per-patch* visual embeddings per document page and scores
  with a late-interaction MaxSim operator borrowed from
  ColBERT. Designed to retrieve *document pages* (screenshots)
  from natural-language questions; outperforms text-only
  retrieval over OCR'd text because page layout, figures, and
  mixed modalities are visible to the model. Paper: Faysse et
  al., "ColPali: Efficient Document Retrieval with Vision
  Language Models," 2024 —
  [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449).
  Checkpoints via Hugging Face:
  [huggingface.co/vidore/colpali](https://huggingface.co/vidore/colpali).
  Benchmark: ViDoRe —
  [huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard).

### 2.2 Why the three models land in different places

A quick mental model, before the code:

- **Shape of the output.** CLIP and SigLIP are *single-vector*
  per image. ColPali is *multi-vector* per image — one vector
  per visual patch, so a 448×448 image at patch size 14 gives
  ≈ 1024 vectors. This changes the storage footprint by two
  to three orders of magnitude; it is the single most important
  operational fact.
- **Scoring function.** CLIP uses cosine. SigLIP uses a shifted
  dot product passed through a sigmoid (for a probability) or
  used raw as a similarity logit. ColPali scores a query–page
  pair as `sum_i max_j (q_i · p_j)` — for each query token, take
  the max over all page patches, then sum. This is the
  late-interaction operator from ColBERT, adapted to vision.
- **What the training data biases each model toward.** CLIP and
  SigLIP were trained on image–caption pairs scraped from the
  web; they are great at the "photo + short caption" shape.
  ColPali was trained with a focus on real documents (ViDoRe
  benchmark). On a page of a financial report, CLIP sees a
  grey rectangle; ColPali sees the table.

### 2.3 `src/embedders.py` — a common interface

```python
"""Three embedders behind one interface.

Single-vector embedders return `np.ndarray` of shape (D,). The
multi-vector embedder returns `np.ndarray` of shape (N_patches, D).
Scoring is model-specific; the caller does not need to know the
operator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DocEmbedding:
    doc_id: str
    vectors: np.ndarray  # (D,) for single-vector, (N, D) for multi-vector


class MultimodalEmbedder(ABC):
    name: str

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_docs(self, image_paths: list[str]) -> list[DocEmbedding]:
        ...

    @abstractmethod
    def score(self, query_vec: np.ndarray, doc: DocEmbedding) -> float:
        ...


class CLIPEmbedder(MultimodalEmbedder):
    name = "clip"

    def __init__(self, checkpoint: str = "openai/clip-vit-base-patch32") -> None:
        # Lazy import so the module can be read without heavy deps.
        from transformers import CLIPModel, CLIPProcessor
        self._model = CLIPModel.from_pretrained(checkpoint).eval()
        self._processor = CLIPProcessor.from_pretrained(checkpoint)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def embed_docs(self, image_paths: list[str]) -> list[DocEmbedding]:
        import torch
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return [
            DocEmbedding(doc_id=str(p), vectors=v)
            for p, v in zip(image_paths, feats.cpu().numpy())
        ]

    def score(self, query_vec: np.ndarray, doc: DocEmbedding) -> float:
        return float(np.dot(query_vec, doc.vectors))


class SigLIPEmbedder(MultimodalEmbedder):
    name = "siglip"

    def __init__(self, checkpoint: str = "google/siglip-base-patch16-224") -> None:
        from transformers import AutoModel, AutoProcessor
        self._model = AutoModel.from_pretrained(checkpoint).eval()
        self._processor = AutoProcessor.from_pretrained(checkpoint)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        import torch
        inputs = self._processor(text=texts, return_tensors="pt", padding="max_length")
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def embed_docs(self, image_paths: list[str]) -> list[DocEmbedding]:
        import torch
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return [
            DocEmbedding(doc_id=str(p), vectors=v)
            for p, v in zip(image_paths, feats.cpu().numpy())
        ]

    def score(self, query_vec: np.ndarray, doc: DocEmbedding) -> float:
        # Raw dot-product as similarity logit. The sigmoid probability
        # would require the model's learned bias/scale; the logit
        # ranks the same way and keeps the harness checkpoint-agnostic.
        return float(np.dot(query_vec, doc.vectors))


class ColPaliEmbedder(MultimodalEmbedder):
    name = "colpali"

    def __init__(self, checkpoint: str = "vidore/colpali") -> None:
        # The official usage pattern is covered in the model card
        # on Hugging Face; imports are kept lazy because the model
        # is substantially heavier than CLIP/SigLIP.
        from colpali_engine.models import ColPali, ColPaliProcessor
        self._model = ColPali.from_pretrained(checkpoint).eval()
        self._processor = ColPaliProcessor.from_pretrained(checkpoint)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        import torch
        batch = self._processor.process_queries(texts)
        with torch.no_grad():
            q = self._model(**batch)
        # (B, N_q, D)
        return q.cpu().numpy()

    def embed_docs(self, image_paths: list[str]) -> list[DocEmbedding]:
        import torch
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]
        batch = self._processor.process_images(images)
        with torch.no_grad():
            feats = self._model(**batch)
        # (B, N_patches, D) — one DocEmbedding per image, with its
        # full patch set.
        return [
            DocEmbedding(doc_id=str(p), vectors=np.asarray(v))
            for p, v in zip(image_paths, feats.cpu().numpy())
        ]

    def score(self, query_vec: np.ndarray, doc: DocEmbedding) -> float:
        # Late-interaction MaxSim: for each query token, take the
        # max similarity over all document patches, then sum.
        # query_vec: (N_q, D); doc.vectors: (N_p, D)
        sim = query_vec @ doc.vectors.T   # (N_q, N_p)
        max_per_q = sim.max(axis=-1)      # (N_q,)
        return float(max_per_q.sum())
```

**Choices and why:**

- **Lazy imports.** Each embedder depends on heavy libraries;
  importing the module for tests should not require all three.
- **Normalisation only for the single-vector models.** CLIP and
  SigLIP's conventional retrieval use is cosine over
  L2-normalised embeddings. ColPali does not normalise; the
  late-interaction operator is its own scoring function.
- **`score` on the embedder, not on the index.** The index uses
  whatever vector math the embedder prescribes; the harness
  does not pretend that one similarity function fits all three.
- **SigLIP uses the raw logit.** The sigmoid probability
  depends on the model's learned `logit_scale` and
  `logit_bias`; the ranking is identical for the raw logit,
  which keeps the exercise less coupled to specific checkpoint
  internals. A learner who wants calibrated probabilities
  should apply the learned bias/scale from the config.

### 2.4 `src/benchmark.py` — the harness

```python
"""Compute recall@k, MRR, and per-query latency for each embedder.

Deterministic: fixed random seeds, pinned checkpoints, pinned
corpus. Reruns produce identical numbers (modulo GPU
non-determinism; use `torch.use_deterministic_algorithms` for
exact reproduction)."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embedders import MultimodalEmbedder, DocEmbedding


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class BenchmarkRow:
    embedder: str
    qid: str
    latency_s: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float


def load_queries(path: Path) -> list[Query]:
    rows = json.loads(path.read_text())
    return [
        Query(
            qid=r["qid"],
            text=r["text"],
            relevant_doc_ids=frozenset(r["relevant_doc_ids"]),
        )
        for r in rows
    ]


def run_embedder(
    embedder: MultimodalEmbedder,
    queries: list[Query],
    docs: list[DocEmbedding],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    q_vecs = embedder.embed_queries([q.text for q in queries])
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        scores = np.array([embedder.score(q_vecs[i], d) for d in docs])
        order = np.argsort(-scores)
        latency = time.perf_counter() - t0
        ranked_ids = [docs[j].doc_id for j in order]
        recalls = {kk: _recall_at_k(ranked_ids, q.relevant_doc_ids, kk) for kk in k_values}
        rr = _reciprocal_rank(ranked_ids, q.relevant_doc_ids)
        rows.append(BenchmarkRow(
            embedder=embedder.name,
            qid=q.qid,
            latency_s=latency,
            recall_at_1=recalls[1],
            recall_at_5=recalls[5],
            recall_at_10=recalls[10],
            reciprocal_rank=rr,
        ))
    return rows


def _recall_at_k(ranked: list[str], gold: frozenset[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(ranked[:k])
    return len(top & gold) / len(gold)


def _reciprocal_rank(ranked: list[str], gold: frozenset[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def write_csv(rows: list[BenchmarkRow], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "embedder", "qid", "latency_s",
            "recall_at_1", "recall_at_5", "recall_at_10", "reciprocal_rank",
        ])
        for r in rows:
            w.writerow([
                r.embedder, r.qid, f"{r.latency_s:.4f}",
                f"{r.recall_at_1:.3f}", f"{r.recall_at_5:.3f}",
                f"{r.recall_at_10:.3f}", f"{r.reciprocal_rank:.3f}",
            ])
```

**Choices and why:**

- **Linear scan over documents.** On ≥ 50 and ≥ 20 documents,
  an ANN index adds no value and introduces an extra variable;
  the harness measures *retrieval quality*, not *index speed*.
  For a production scale deployment, replace the linear scan
  with a vector DB — the single-vector models fit, and
  late-interaction stores (e.g., native ColBERT-style
  databases) handle ColPali.
- **Recall@k set-wise.** A query with multiple relevant
  documents counts recall as the fraction found, not whether
  any were found. This is the standard definition; the
  exercise is explicit about it so the numbers can be
  compared across runs.
- **Latency per query.** Single-vector embedders have a cheap
  score (one dot product per doc); ColPali's score is
  `N_q × N_p` dot products per doc. The numbers surface the
  cost directly.

### 2.5 Expected shape of the results

The absolute numbers depend on the chosen corpus, checkpoints,
and hardware; the exercise does not claim a specific value. The
*shape* of the results is what the learner is expected to
observe and defend:

- On `corpus/photos/` (natural images, short captions): CLIP
  and SigLIP are competitive. SigLIP typically matches or edges
  out CLIP at matched compute on standard image–text retrieval
  benchmarks (per the SigLIP paper), but on a 50-image
  bespoke corpus the difference is dominated by noise. ColPali
  is *not* the right model for this corpus — it was trained
  for document pages, and the per-patch vectors are overkill
  for a 300×300 photograph.
- On `corpus/pages/` (PDF screenshots, mixed modality,
  question-style queries): ColPali's recall is substantially
  higher than CLIP's or SigLIP's. The ViDoRe benchmark results
  demonstrate this pattern across standard document-retrieval
  datasets (see the Hugging Face leaderboard). Latency is also
  substantially higher; the trade-off is explicit.
- Storage: single-vector models produce (D,) per document;
  ColPali produces (N_patches, D). For the pages corpus this is
  roughly a 1000× index-size multiplier at the reference patch
  size, which is the operational line between "add ColPali to
  the stack" and "rebuild the index tier." Numbers for the
  exact corpus, checkpoint, and patch config land in
  `results/summary.md`.

### 2.6 `results/summary.md` — the recommendation

The deliverable's `summary.md` collects the tables and the two
scenario picks.

#### Scenario A — Natural-image search for a stock-photo app

User types a caption; app returns images.

- **Pick: SigLIP** (CLIP is adequate; the choice is driven by
  training objective quality at matched compute, per the SigLIP
  paper's improvements over CLIP on image–text retrieval).
- **Why not ColPali:** single-vector is cheaper to index and
  serve by 2–3 orders of magnitude; the natural-image corpus
  does not need per-patch interaction.
- **Kill switch:** if the product adds "search inside my
  scanned documents," the pick re-opens and ColPali comes in.

#### Scenario B — Enterprise document search over a mixed PDF corpus

Users ask questions like "what was the EBITDA margin in Q3?"
and need the right *page* of the right PDF.

- **Pick: ColPali** as the primary retriever; keep SigLIP as a
  cheap first-pass for pages that are predominantly natural
  images (figures, photos).
- **Why not CLIP / SigLIP alone:** the single-vector models
  compress a full page into one vector; fine-grained text
  content inside the image is attenuated. The ViDoRe
  leaderboard shows this gap directly — late-interaction
  multi-vector models outperform single-vector ones on
  document-retrieval benchmarks.
- **Why not OCR → text retriever alone:** OCR degrades on
  tables and multi-column layouts, and loses figures entirely.
  ColPali reads the page as it is.
- **Residual risk:** storage multiplier. If the corpus reaches
  millions of pages, late-interaction storage requires either
  a specialised index (e.g., a compression-aware
  multi-vector store) or a two-stage pipeline (cheap
  single-vector first pass, ColPali re-rank on top-N).

### 2.7 `notes/model_cards.md` — a note on provenance

Each embedder has a published model card the learner should
read before deploying:

- CLIP — see the CLIP paper
  ([arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020))
  and the OpenAI model card that ships with the GitHub release
  ([github.com/openai/CLIP/blob/main/model-card.md](https://github.com/openai/CLIP/blob/main/model-card.md)).
  Training data: a 400M-image web-scraped dataset. Known
  limitations the card calls out: bias in web-scale training
  data, and degraded performance on non-natural images (text
  documents, medical images) and on languages other than
  English.
- SigLIP — see the SigLIP paper
  ([arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343))
  and the model cards on each Hugging Face checkpoint. Trained
  on WebLI with the sigmoid objective. Known limitations: like
  CLIP, biased by the training corpus; multilingual variants
  exist but the base checkpoint is English-centric.
- ColPali — see the ColPali paper
  ([arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449))
  and the Hugging Face model card
  ([huggingface.co/vidore/colpali](https://huggingface.co/vidore/colpali)).
  Built on PaliGemma. Intended use: document retrieval from
  natural-language questions. Licence: review the checkpoint
  page for the current licence terms before deploying.

The point of the model-card step is twofold. First, it supports
the NIST AI RMF *Govern* and *Map* functions: who built the
model, what it was trained on, and what the authors say it is
not for. Second, it surfaces supply-chain concerns (OWASP ML06,
MITRE ATLAS *ML Supply Chain Compromise*) — a model is as
trustworthy as its training data and its distribution channel.

## 3. Validation steps

A correct submission is reproducible given the checked-in
corpus, pinned checkpoints, and pinned library versions.

1. **Install.** `pip install -r requirements.txt` where
   `requirements.txt` pins `transformers`, `torch`, `pillow`,
   `numpy`, and (for ColPali) `colpali-engine`. Pin versions.
2. **Seed the corpora.** The repository ships the images and
   the queries JSON; check them in verbatim and do not
   regenerate.
3. **Run the harness per embedder.** `python -m src.benchmark
   --embedder clip` → writes `results/clip.csv`. Same for
   `siglip` and `colpali`.
4. **Aggregate.** `python -m src.benchmark --aggregate` →
   writes `results/summary.md` with the mean recall@k and
   mean latency per embedder per corpus.
5. **Sanity invariants.** On the photos corpus, CLIP and SigLIP
   recall@5 should be positive; ColPali should at least not
   crash. On the pages corpus, ColPali recall@5 should beat
   both CLIP and SigLIP by a visible margin. If this inversion
   is not reproduced, either the corpus is too small to
   discriminate or the ColPali embedder is misconfigured.
6. **Determinism.** With `torch.manual_seed(0)` and (where
   applicable) `torch.use_deterministic_algorithms(True)`, two
   runs produce the same ranked lists and therefore the same
   recall numbers.
7. **Latency recorded.** `results/summary.md` reports mean
   per-query latency per embedder, so the "ColPali is slower"
   observation is backed by numbers, not by vibe.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Three embedders | CLIP, SigLIP, and ColPali each behind the same interface; score function per model, not shared. |
| Two corpora | Natural-image corpus with captioned ground truth; PDF-page corpus with question-style ground truth. |
| Metrics | Recall@1/5/10 and MRR per query, aggregated per embedder; per-query latency. |
| Reproducibility | Seeds set; checkpoints pinned; library versions pinned; two runs produce the same numbers. |
| Results narrative | `summary.md` reports the numbers *and* explains the win conditions — not just a leaderboard. |
| Scenario picks | Two scenarios (natural-image, enterprise document) each with a pick, a why pointing at the numbers, and a kill switch. |
| Model cards cited | Each embedder's official card is cited, with training data, intended use, and limitations. |
| Supply chain acknowledged | Supply-chain risk (ML06 / ATLAS ML Supply Chain Compromise) is addressed in the recommendation. |
| Storage trade-off named | The single-vector vs. multi-vector index-size difference is quantified for the exercise corpus. |
| No cherry-picked numbers | The results table shows every query, not only the ones that make a pick look good. |

Fail-fast smells:

- The report compares only aggregate recall and omits latency
  and storage. The multi-vector cost is the trade-off; hiding
  it hides the whole point.
- ColPali is the pick on the natural-image corpus. Possible,
  but very unlikely; look for a bug in the single-vector
  embedders if the result holds.
- CLIP or SigLIP is the pick on the PDF corpus. Possible if
  the pages are mostly natural-image figures, but the memo
  should defend *why* the single-vector model wins — the
  document-retrieval benchmarks (ViDoRe) generally favour
  late-interaction.
- The results depend on an unpinned checkpoint ("the latest on
  HF"). Governance breaks when the model behind a benchmark
  moves.
- No model cards are cited. The pick cannot be defended against
  "what is this model, where did it come from" without them.

## 5. Common mistakes

- **Treating the three models as interchangeable "image
  embedders."** They are three retrieval shapes. Picking one
  for a scenario it was not designed for is the top failure
  mode.
- **L2-normalising ColPali patch vectors.** ColPali's late-
  interaction scoring assumes the trained-in geometry of the
  patch vectors. Extra normalisation changes the operator's
  behaviour; use the model as published.
- **Benchmarking with batched cosine similarity against
  ColPali.** The score is not cosine; it is MaxSim-sum. A
  benchmark that calls `cosine(query, doc)` with ColPali
  vectors is running a score the model was not trained for.
- **Measuring on 10 queries.** Small-corpus results are
  dominated by noise; the exercise's 20–50 scale is a floor,
  not a ceiling. For a production decision, use a benchmark
  of ≥ 1000 queries from the target domain.
- **Treating SigLIP and CLIP as "the same but newer."** They
  are similar in retrieval shape but have different losses and
  different training corpora; a drop-in swap can improve or
  regress a specific retrieval task. Measure before deploying.
- **Deploying the ColPali multi-vector index into a standard
  vector DB without a plan.** Standard vector databases store
  one vector per document by default; ColPali needs either a
  specialised store or an application-side implementation of
  MaxSim. The exercise's in-memory implementation is for
  measurement, not production.
- **Omitting per-query numbers.** Aggregates hide failure cases
  ("one bad query dragged the mean"). The results CSV must
  include per-query scores so a reviewer can look at the
  failure modes directly.
- **Picking a model without reading its model card.** The card
  states training data, intended use, and known limitations.
  Shipping a model without reading its card is the governance
  equivalent of running unsigned code.
- **Ignoring the storage multiplier.** A ~1000× index-size
  increase is a tier-change in the index infrastructure, not a
  configuration flip. The scenario memo has to say what
  changes operationally.

## 6. References

Model documentation (primary sources for the scoring functions
and the model cards):

- CLIP paper — Radford et al., 2021 —
  [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020).
  OpenAI CLIP repository and model card —
  [github.com/openai/CLIP](https://github.com/openai/CLIP).
- SigLIP paper — Zhai et al., 2023 —
  [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343).
  Checkpoint and card —
  [huggingface.co/google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224).
- ColPali paper — Faysse et al., 2024 —
  [arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449).
  Checkpoint and card —
  [huggingface.co/vidore/colpali](https://huggingface.co/vidore/colpali).
- ViDoRe leaderboard (document-retrieval benchmark) —
  [huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard).
- PaliGemma (the base vision-language model under ColPali) —
  [ai.google.dev/gemma/docs/paligemma](https://ai.google.dev/gemma/docs/paligemma).
- Hugging Face `transformers` documentation —
  [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers).

Governance frameworks:

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: ML06 AI Supply Chain Attacks — pinning model
  checkpoints and verifying the publishing organisation are
  part of the retrieval system's trust boundary.
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: *ML Supply Chain Compromise* (compromised
  weights), *Adversarial ML Attack on Image Classifier*
  (adversarial inputs that look like legitimate images).
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: *Govern* (pin checkpoints, vet licence terms),
  *Map* (declare which model handles which scenario and why),
  *Measure* (the recall/latency/storage benchmark is the
  measurement), *Manage* (document the kill-switch conditions
  that flip the pick).

Downstream consumers:

- exercise-01 (ACL patterns) — a multimodal retriever still
  needs an ACL field on each image / page; the embedder is
  authorisation-agnostic.
- exercise-05 (hybrid router) uses the scenario picks: the
  router dispatches "describe-this-photo" questions to the
  single-vector path and "find-this-in-a-PDF" questions to
  the ColPali path. The router does not know embeddings; it
  knows the question class.
