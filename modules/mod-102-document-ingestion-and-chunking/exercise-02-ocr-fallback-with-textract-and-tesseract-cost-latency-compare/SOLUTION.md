# Exercise 02 — Solution: OCR fallback, Tesseract vs. Textract cost/latency compare

This is a reference solution for `exercise-02-ocr-fallback-with-textract-and-tesseract-cost-latency-compare.md`.
It is an **implementation + measurement** exercise. The deliverable is a per-page routing probe,
two OCR back-ends behind one adapter, a benchmark script that produces four numbers per
back-end, and a defensible per-corpus policy. Pricing and vendor latency numbers in this
reference are **placeholders**; the learner must pull current prices from each vendor's pricing
page on the date they run the benchmark. No invented numbers are cited as fact.

## 1. Solution overview

The exercise rewrites the PDF branch of exercise-01's router to decide *per page* whether to
stay on the native-text path or fall back to OCR, using the two-signal probe from chapter 3
(`chars_per_page < 100` OR `ratio_alpha < 0.5`). Two OCR back-ends hide behind one `OcrBackend`
protocol: local Tesseract (free, CPU-bound, no table structure) and a hosted service
(this reference uses AWS Textract; Azure Document Intelligence and GCP Document AI are
documented as drop-ins). A 50-page benchmark script measures cost per 1,000 pages, p50/p95
per-page wall-clock latency, character error rate (CER) against five hand-transcribed
ground-truth pages, and table-cell recall on two table-containing pages. The chosen policy —
static, confidence-escalation, or feature-triggered — is wired into the router and exercised
by a test.

Important scoping notes:

- **This is a cost exercise.** Keep the hosted-service bill under $5 by capping to 50 pages
  and tearing down any S3 buckets created for Textract async jobs. Export the console cost
  line into `docs/ocr-benchmark.md` as proof.
- **Don't invent pricing or latency numbers.** Each vendor publishes current per-page pricing;
  cite the pricing page with the dated line `Prices as of YYYY-MM-DD` in the benchmark doc.
  Do not hard-code "$1.50 per 1,000 pages" from memory.
- **"Textract" in the reference implementation = detect-text API.** The pricing and latency
  profile is different for `AnalyzeDocument` (tables/forms) vs. `DetectDocumentText`; the
  table-recall metric is the one that should drive you toward `AnalyzeDocument`'s `TABLES`
  feature, and that choice is a benchmark output, not an input.

## 2. Worked answer or implementation

### 2.1 `src/ingest/router.py` — per-page routing probe

```python
# src/ingest/router.py (relevant additions)
from __future__ import annotations
import pathlib
from typing import Literal

import pypdf

_MIN_CHARS_PER_PAGE = 100
_MIN_ALPHA_RATIO = 0.5

def _ratio_alpha(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))

def route_pdf(path: pathlib.Path) -> list[Literal["native", "ocr"]]:
    """Return one branch per page using the two-signal probe (chapter 3)."""
    branches: list[Literal["native", "ocr"]] = []
    reader = pypdf.PdfReader(str(path))
    for page in reader.pages:
        text = page.extract_text() or ""
        if len(text.strip()) < _MIN_CHARS_PER_PAGE or _ratio_alpha(text) < _MIN_ALPHA_RATIO:
            branches.append("ocr")
        else:
            branches.append("native")
    return branches
```

The rasterization helper uses PyMuPDF so the two OCR back-ends can share one PNG pipeline:

```python
# src/ingest/_raster.py
import fitz  # pymupdf

def rasterize_page(path: str, page_index: int, dpi: int = 300) -> bytes:
    doc = fitz.open(path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()
```

Routing test:

```python
# tests/test_route_pdf.py
import pathlib

from ingest.router import route_pdf

def test_native_corpus_stays_native():
    branches = route_pdf(pathlib.Path("data/corpus/handbook.pdf"))
    # The clean native-text PDF from exercise-01 has every page on the native branch.
    assert branches and all(b == "native" for b in branches), branches

def test_scanned_corpus_routes_to_ocr():
    pdfs = sorted(pathlib.Path("data/ocr-corpus").glob("*.pdf"))
    assert pdfs, "scanned corpus is missing"
    for pdf in pdfs:
        branches = route_pdf(pdf)
        assert branches and all(b == "ocr" for b in branches), (pdf, branches)
```

### 2.2 `src/ingest/ocr.py` — adapter + two back-ends

```python
# src/ingest/ocr.py
from __future__ import annotations
import io
import time
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image
import pytesseract

class OcrError(Exception):
    """Typed error for every OCR path. Never raise a bare Exception."""

@dataclass
class OcrToken:
    text: str
    bbox: tuple[float, float, float, float]  # (x, y, w, h), image-pixel space
    confidence: float  # 0..1

@dataclass
class OcrPage:
    text: str
    tokens: list[OcrToken] = field(default_factory=list)
    mean_confidence: float = 0.0

class OcrBackend(Protocol):
    name: str
    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> OcrPage: ...

# ---------- Tesseract ----------

class TesseractBackend:
    name = "tesseract"

    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> OcrPage:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # image_to_data gives per-word boxes + confidence.
            data = pytesseract.image_to_data(
                image, lang=hint_lang, output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError as e:
            raise OcrError(f"tesseract failed: {e}") from e

        tokens: list[OcrToken] = []
        texts: list[str] = []
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            conf = float(data["conf"][i])
            if conf < 0:  # -1 is Tesseract's "no confidence"
                continue
            tokens.append(OcrToken(
                text=word,
                bbox=(data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
                confidence=conf / 100.0,
            ))
            texts.append(word)
            confs.append(conf / 100.0)
        mean = sum(confs) / len(confs) if confs else 0.0
        return OcrPage(text=" ".join(texts), tokens=tokens, mean_confidence=mean)

# ---------- Textract (sync, detect-text; see note on async for 100+ pages) ----------

class TextractBackend:
    name = "textract"

    def __init__(self, client=None):
        import boto3
        self.client = client or boto3.client("textract")

    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> OcrPage:
        from botocore.exceptions import ClientError
        try:
            resp = self.client.detect_document_text(Document={"Bytes": image_bytes})
        except ClientError as e:
            raise OcrError(f"textract failed: {e.response['Error']['Message']}") from e

        tokens: list[OcrToken] = []
        texts: list[str] = []
        confs: list[float] = []
        for block in resp.get("Blocks", []):
            if block.get("BlockType") != "WORD":
                continue
            bbox = block["Geometry"]["BoundingBox"]
            confidence = float(block.get("Confidence", 0.0)) / 100.0
            tokens.append(OcrToken(
                text=block["Text"],
                bbox=(bbox["Left"], bbox["Top"], bbox["Width"], bbox["Height"]),
                confidence=confidence,
            ))
            texts.append(block["Text"])
            confs.append(confidence)
        mean = sum(confs) / len(confs) if confs else 0.0
        return OcrPage(text=" ".join(texts), tokens=tokens, mean_confidence=mean)
```

Azure and GCP drop-ins (documented in `src/ingest/README.md`, enabled via `OCR_BACKEND` env var):

```python
# DocIntelligenceBackend — azure-ai-documentintelligence
#   client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
#   poller = client.begin_analyze_document("prebuilt-read", image_bytes)
#   result = poller.result()
#   iterate result.pages[0].words for text + confidence
#
# DocumentAIBackend — google-cloud-documentai
#   client = documentai.DocumentProcessorServiceClient()
#   name = client.processor_path(project, location, processor_id)
#   result = client.process_document(request={"name": name,
#            "raw_document": {"content": image_bytes, "mime_type": "image/png"}})
#   iterate result.document.pages[0].tokens
```

Router wiring:

```python
# src/ingest/router.py (OCR branch)
from .ocr import OcrBackend, TesseractBackend
from ._raster import rasterize_page
from .schema import Chunk
from ._ids import stable_chunk_id, content_hash, now_utc
from ._tokens import count_tokens

_CHUNKER = "ocr.per_page"
_CHUNKER_VERSION = "1.0.0"

def ingest_pdf_with_ocr(
    path: pathlib.Path,
    *,
    backend: OcrBackend | None = None,
    permission_scope: str = "public",
) -> list[Chunk]:
    backend = backend or TesseractBackend()
    branches = route_pdf(path)
    doc_id = path.stem
    chunks: list[Chunk] = []
    ordinal = 0
    for page_index, branch in enumerate(branches):
        if branch != "ocr":
            continue  # native path handled by ingest_pdf_* implementations
        img = rasterize_page(str(path), page_index)
        page = backend.ocr_page(img)
        chunks.append(Chunk(
            chunk_id=stable_chunk_id(doc_id, _CHUNKER_VERSION, ordinal),
            doc_id=doc_id,
            source_uri=str(path.resolve()),
            ordinal=ordinal,
            text=page.text,
            element_type="paragraph",
            tokens=count_tokens(page.text),
            page=page_index + 1,
            permission_scope=permission_scope,
            content_hash=content_hash(page.text),
            parser=backend.name,
            parser_version=_backend_version(backend),
            chunker=_CHUNKER,
            chunker_version=_CHUNKER_VERSION,
            ingested_at=now_utc(),
            extras={
                "ocr_used": True,
                "ocr_mean_confidence": page.mean_confidence,
            },
        ))
        ordinal += 1
    return chunks
```

(The `Chunk` schema from exercise-01 already carries `extras`; the policy of using
`extras["ocr_used"]` vs. adding a top-level `ocr_used` column is defensible either way. If
your downstream retriever needs to filter on "OCR vs. native" efficiently, promote it to a
first-class column — chapter 7's pre-filter discipline.)

### 2.3 `scripts/ocr_benchmark.py`

```python
# scripts/ocr_benchmark.py
from __future__ import annotations
import pathlib
import statistics
import time
from datetime import date

import jiwer

from ingest.ocr import OcrBackend, TesseractBackend, TextractBackend
from ingest._raster import rasterize_page

CORPUS = pathlib.Path("data/ocr-corpus")
GT = CORPUS / "ground-truth"

# Pricing MUST be filled in from the vendor's current pricing page on the run date.
# Example shape — DO NOT COMMIT A NUMBER THAT WAS NOT READ FROM THE VENDOR TODAY.
PRICING_USD_PER_PAGE = {
    "tesseract": 0.0,
    "textract":  None,  # fill from the AWS Textract pricing page (DetectDocumentText tier)
}

def _per_page_latency_and_text(backend: OcrBackend, pdf: pathlib.Path):
    import fitz
    doc = fitz.open(str(pdf))
    rows = []
    try:
        for i in range(len(doc)):
            img = rasterize_page(str(pdf), i)
            t0 = time.perf_counter()
            result = backend.ocr_page(img)
            wall = time.perf_counter() - t0
            rows.append({
                "pdf": pdf.name, "page": i + 1, "wall_s": wall,
                "text": result.text, "conf": result.mean_confidence,
            })
    finally:
        doc.close()
    return rows

def _cer(hyp: str, ref: str) -> float:
    return jiwer.cer(ref, hyp)  # jiwer: cer(reference, hypothesis)

def _run(backend: OcrBackend) -> dict:
    rows: list = []
    for pdf in sorted(CORPUS.glob("*.pdf")):
        rows.extend(_per_page_latency_and_text(backend, pdf))

    # Latency.
    latencies = sorted(r["wall_s"] for r in rows)
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies)) - 1] if len(latencies) >= 20 else latencies[-1]

    # Cost.
    unit = PRICING_USD_PER_PAGE[backend.name]
    cost_per_1000 = "n/a" if unit is None else f"${unit * 1000:.2f}"

    # CER on ground-truth pages.
    cers: list[float] = []
    for gt_file in GT.glob("*.txt"):
        stem = gt_file.stem  # "docA--p3"
        pdf_name, page = stem.split("--p")
        hyp = next(
            (r["text"] for r in rows if r["pdf"] == f"{pdf_name}.pdf" and r["page"] == int(page)),
            None,
        )
        if hyp is None:
            continue
        cers.append(_cer(hyp, gt_file.read_text(encoding="utf-8")))

    return {
        "name": backend.name,
        "cost_per_1000": cost_per_1000,
        "p50_s": p50, "p95_s": p95,
        "cer_mean": statistics.mean(cers) if cers else float("nan"),
        "cer_min": min(cers) if cers else float("nan"),
        "cer_max": max(cers) if cers else float("nan"),
    }

if __name__ == "__main__":
    out = []
    for backend in (TesseractBackend(), TextractBackend()):
        out.append(_run(backend))

    today = date.today().isoformat()
    print(f"# OCR benchmark — {today}\n")
    print(f"Prices as of {today}.\n")
    print("| back-end  | $/1000 pages | p50 latency (s) | p95 latency (s) | CER mean | CER range |")
    print("|-----------|--------------|-----------------|-----------------|----------|-----------|")
    for r in out:
        print(f"| {r['name']:<9} | {r['cost_per_1000']:>12} | {r['p50_s']:>15.2f} |"
              f" {r['p95_s']:>15.2f} | {r['cer_mean']:>8.3f} |"
              f" {r['cer_min']:.3f}–{r['cer_max']:.3f} |")
```

Table-recall column is produced separately — pick the two pages with tables, hand-count
the ground-truth cells, parse each back-end's output into `(row, col, text)` triples, and
report `recovered_cells / ground_truth_cells`. Tesseract rarely preserves row/column
indexing on tables; Textract's `AnalyzeDocument(FeatureTypes=["TABLES"])` does.

### 2.4 `docs/ocr-benchmark.md` — template

```markdown
# OCR benchmark — corpus: {name}, sampled 50 pages on {date}

Prices as of {date}. Sources:
- Tesseract: self-hosted; cost is CPU time only. [Tesseract install](https://tesseract-ocr.github.io/tessdoc/Installation.html).
- {hosted}: see the vendor's pricing page.

| back-end   | $/1000 pages | p50 latency | p95 latency | CER (mean / range) | table recall |
|------------|--------------|-------------|-------------|--------------------|--------------|
| tesseract  | $0.00        | <t>         | <t>         | <mean> / <lo>–<hi> | <recovered>/<gt> |
| {hosted}   | $<p×1000>    | <t>         | <t>         | <mean> / <lo>–<hi> | <recovered>/<gt> |

**AWS / Azure / GCP cost line.** See `docs/ocr-cost-screenshot.png`.

## Recommendation

<One paragraph naming the policy. Example shapes below — do not copy, write your own.>

- *Static.* "For this 50-page scanned-contracts corpus the CER gap between Tesseract and
  Textract was <0.03 absolute and the table recall difference was <5%. We default to
  Tesseract and revisit if the corpus grows past 10k pages."
- *Confidence-escalation.* "Tesseract-first. When `ocr_mean_confidence < 0.75` on a page,
  re-run through Textract and prefer its output. The 75% threshold was chosen from the
  per-page scatter plot where CER crosses 0.10."
- *Feature-triggered.* "Any page whose layout probe detects a table region goes directly
  to Textract's AnalyzeDocument. All other pages go to Tesseract. Triggered on the two
  table-heavy pages in the sample and spent $<n> on those two calls."
```

### 2.5 `src/ingest/policy.py` — wiring the chosen policy

```python
# src/ingest/policy.py — confidence-escalation shape (example)
from __future__ import annotations
from .ocr import OcrBackend, TesseractBackend, TextractBackend, OcrPage

class EscalatingBackend:
    name = "tesseract->textract"

    def __init__(
        self,
        primary: OcrBackend | None = None,
        secondary: OcrBackend | None = None,
        *,
        threshold: float = 0.75,
    ):
        self.primary = primary or TesseractBackend()
        self.secondary = secondary or TextractBackend()
        self.threshold = threshold

    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> OcrPage:
        first = self.primary.ocr_page(image_bytes, hint_lang=hint_lang)
        if first.mean_confidence >= self.threshold:
            return first
        return self.secondary.ocr_page(image_bytes, hint_lang=hint_lang)
```

Policy test (with the hosted call mocked so the test doesn't hit AWS on CI):

```python
# tests/test_policy.py
from unittest.mock import MagicMock

from ingest.policy import EscalatingBackend
from ingest.ocr import OcrPage

def _fake_backend(name: str, text: str, conf: float):
    b = MagicMock()
    b.name = name
    b.ocr_page.return_value = OcrPage(text=text, tokens=[], mean_confidence=conf)
    return b

def test_escalates_below_threshold():
    primary = _fake_backend("tesseract", "garbled", 0.4)
    secondary = _fake_backend("textract", "clean", 0.98)
    be = EscalatingBackend(primary, secondary, threshold=0.75)
    out = be.ocr_page(b"\x89PNG...")
    assert out.text == "clean"
    secondary.ocr_page.assert_called_once()

def test_keeps_primary_above_threshold():
    primary = _fake_backend("tesseract", "ok", 0.9)
    secondary = _fake_backend("textract", "never-called", 0.99)
    be = EscalatingBackend(primary, secondary, threshold=0.75)
    out = be.ocr_page(b"\x89PNG...")
    assert out.text == "ok"
    secondary.ocr_page.assert_not_called()
```

## 3. Validation steps

1. Dependencies: `uv add pytesseract pillow jiwer pymupdf boto3` (and the Azure or GCP SDK
   if you picked that hosted service).
2. System packages: Tesseract installed (`tesseract --version` prints something).
3. Credentials: `aws configure` (or Azure/GCP equivalents) under a low-limit IAM user.
   Verify from the shell: `aws textract detect-document-text --document '{"Bytes":"iVBORw0KGgo..."}'`
   succeeds on a tiny PNG.
4. Corpus: 10 scanned PDFs × ~5 pages under `data/ocr-corpus/`, plus 5 hand-transcribed
   pages under `data/ocr-corpus/ground-truth/{stem}--p{n}.txt`.
5. `uv run pytest -q tests/test_route_pdf.py tests/test_policy.py` — both tests pass.
6. `uv run python -m scripts.ocr_benchmark > docs/ocr-benchmark.md` — fills in the table
   (cost column reports `n/a` for any backend whose `PRICING_USD_PER_PAGE` was left as
   `None`; the learner must edit the dict with the vendor's current per-page price before
   re-running).
7. Open AWS/Azure/GCP console → Cost Explorer → filter to the current month, screenshot
   the Textract line, save to `docs/ocr-cost-screenshot.png`, assert total < $5.
8. Open `docs/ocr-benchmark.md` and confirm the Recommendation paragraph makes a decision
   (names a backend or names a threshold), not "it depends."

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Per-page probe | `route_pdf` returns `["native"]` for every page of exercise-01's `handbook.pdf`, and `["ocr"]` for every page of the scanned corpus. Both covered by tests. |
| Rasterization | 300 DPI PNG bytes via PyMuPDF; one helper reused by both backends. |
| Adapter | `OcrBackend` protocol with `name` + `ocr_page`; both `TesseractBackend` and the hosted backend conform; typed `OcrError` raised on network or parse failure (never silent `""`). |
| Output chunks | Each page produces one `Chunk` with `element_type="paragraph"`, `page >= 1`, `parser=backend.name`, `extras["ocr_used"]=True`, `extras["ocr_mean_confidence"]` populated. |
| Benchmark coverage | Table has rows for both backends with cost/1000, p50, p95, CER mean + range, and table recall. Prices-as-of line present. |
| Ground truth | 5 hand-transcribed pages committed under `data/ocr-corpus/ground-truth/`; CER computed with `jiwer.cer(reference, hypothesis)` (argument order matters). |
| Policy | One of static / confidence-escalation / feature-triggered implemented and exercised by a test. |
| Cost discipline | Total hosted spend < $5, screenshot committed. |
| Discipline | No invented pricing numbers. `PRICING_USD_PER_PAGE` either filled from the vendor's pricing page today or returns `"n/a"` in the output. |

Fail-fast smells:

- A silent `return OcrPage(text="")` on backend failure. The adapter contract says raise a
  typed error; the router decides whether to escalate or skip.
- `jiwer.cer(hypothesis, reference)` swapped argument order — jiwer's signature is
  `cer(reference, hypothesis)`.
- Benchmarking Tesseract with a 72 DPI rasterization. Tesseract's CER degrades sharply
  below 300 DPI; the comparison is only meaningful when the input resolution is fair.
- A Recommendation paragraph that reads "both are fine." The exercise asks for a decision.
- A test that calls the real Textract API in CI. Mock the hosted backend in policy tests
  (see `_fake_backend`); the only place the real API is called is `scripts/ocr_benchmark.py`.

## 5. Common mistakes

- **Routing by document, not by page.** The point of the two-signal probe in chapter 3 is
  that real corpora contain mixed PDFs — a scanned cover page plus native-text body text,
  a born-digital report with one scanned-figure page. Per-document routing undersells OCR
  on the native path and overpays on it on the OCR path.
- **Forgetting the async wait for Textract's `StartDocumentTextDetection`.** The
  per-page `DetectDocumentText` used in this reference is synchronous; if you switch to
  the async job API for cost or multi-page reasons, time *including the poll loop* or the
  measurement is dishonest.
- **Benchmarking CER without ground truth.** Running the benchmark against the corpus's
  native text (i.e. using the native-text extraction as "ground truth") trains the metric
  on the parser's own bias. Hand-transcribe five pages; it takes ten minutes. The point of
  the exercise is that doing this once gives you a defensible number.
- **Using `extras["ocr_mean_confidence"]` to filter at retrieval time.** An opaque `extras`
  dict is not an indexable field in most vector DBs (chapter 7). If you plan to filter on
  it, promote `ocr_mean_confidence` to a first-class column; otherwise record it only for
  debugging.
- **Reading pricing from training data instead of the vendor page.** Pricing changes.
  Textract's `DetectDocumentText` tier price, Document Intelligence's `prebuilt-read`
  tier price, and Document AI's `OCR` processor price all have public pricing pages; cite
  them with the dated line.
- **Over-fitting the confidence threshold to the five ground-truth pages.** Five pages
  is a *signal*, not a decision boundary. The recommendation should name the threshold
  you chose *and* say "to be re-evaluated on a 500-page sample in the next iteration."
- **Running Tesseract without the `--oem 1` LSTM engine or without `--psm 6` for
  single-block pages.** The defaults are usually fine, but if your CER is surprisingly
  bad on a block of body text, try `psm=4` (single column) or `psm=6` (uniform block)
  before escalating to the hosted service.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-102-document-ingestion-and-chunking/03-ocr-fallbacks-tesseract-textract-doc-intelligence-doc-ai.md`
    — the two-signal probe and the hosted-vs-local trade-off.
  - `lessons/mod-102-document-ingestion-and-chunking/07-metadata-as-first-class-retrieval-fields.md`
    — the `ocr_used` / `ocr_mean_confidence` schema decision.
- Tesseract — [tesseract-ocr.github.io](https://tesseract-ocr.github.io/) and the
  install docs [tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html).
- `pytesseract` — [github.com/madmaze/pytesseract](https://github.com/madmaze/pytesseract).
- AWS Textract — service page [aws.amazon.com/textract/](https://aws.amazon.com/textract/)
  and `DetectDocumentText` API docs
  [docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html](https://docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html).
  Pricing page: [aws.amazon.com/textract/pricing/](https://aws.amazon.com/textract/pricing/)
  — **check the current per-page price on your run date**.
- Azure Document Intelligence —
  [learn.microsoft.com/azure/ai-services/document-intelligence/](https://learn.microsoft.com/azure/ai-services/document-intelligence/).
- GCP Document AI —
  [cloud.google.com/document-ai/docs](https://cloud.google.com/document-ai/docs).
- `jiwer` — [github.com/jitsi/jiwer](https://github.com/jitsi/jiwer) (the `cer` function's
  signature is `cer(reference, hypothesis)`).
- `PyMuPDF` (rasterization) — [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/).
