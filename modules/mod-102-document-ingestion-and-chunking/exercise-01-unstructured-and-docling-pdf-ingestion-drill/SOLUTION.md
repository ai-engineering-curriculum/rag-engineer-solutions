# Exercise 01 — Solution: unstructured and Docling PDF ingestion drill

This is a reference solution for `exercise-01-unstructured-and-docling-pdf-ingestion-drill.md`.
It is an **implementation** exercise. The learner ships a format router plus two PDF back-ends
and passes four round-trip tests. The reference below gives runnable artifacts at the shapes
the exercise asks for — version pins and parser API surfaces will drift, so always confirm against
the current upstream docs linked in References before copying verbatim.

## 1. Solution overview

The exercise asks for one format router (`src/ingest/router.py`) that dispatches by MIME to
three paths — PDF, HTML, and Markdown — each producing `list[Chunk]` conforming to the
chapter-7 schema. The PDF branch is implemented twice (unstructured + Docling) and compared
side-by-side in `scripts/compare_pdf_parsers.py`. Four `pytest` tests close the contract:
no silent content loss, metadata completeness, deterministic `chunk_id`, and PDF page numbers
present. The reference uses:

- `unstructured` with `partition_pdf(strategy="hi_res", infer_table_structure=True)` + `chunk_by_title`.
- `docling` with `DocumentConverter().convert(...).document` and `iterate_items()`.
- `trafilatura` → Markdown + `MarkdownHeaderTextSplitter` for HTML (documented in the per-format README).
- `MarkdownHeaderTextSplitter` + a 800-token recursive split for Markdown.

Parser APIs have been changing fairly quickly across 2024-2025; the shapes below are the ones
documented by the upstream projects at the time of writing. If an API has moved, the correct
answer is "update to the current API and keep the `Chunk` schema stable," not "rewrite the
schema." The ingest layer is the stable contract — parsers are swappable behind it.

## 2. Worked answer or implementation

### 2.1 `src/ingest/schema.py` — the `Chunk` dataclass

The chapter-7 `Chunk` dataclass, with a thin `extras: dict` for parser-specific side-channels
(Docling table HTML, `unstructured` coordinates, OCR confidences from exercise-02). Fields
`parent_id` / `children_ids` / `window_ids` are stubbed for exercise-04; `summary` is stubbed
for exercise-05.

```python
# src/ingest/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class Chunk:
    # identity
    chunk_id: str
    doc_id: str
    source_uri: str
    ordinal: int
    # content
    text: str
    element_type: str  # heading | paragraph | list_item | table | code | figure_caption | ...
    tokens: int
    # structural
    section_path: list[str] = field(default_factory=list)
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    # provenance / governance
    updated_at: datetime | None = None
    published_at: datetime | None = None
    author: str | None = None
    permission_scope: str = "public"
    language: str = "en"
    content_hash: str = ""
    # versioning
    parser: str = ""
    parser_version: str = ""
    chunker: str = ""
    chunker_version: str = ""
    ingested_at: datetime | None = None
    # hierarchical linkage (stubbed here; populated in exercise-04)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    window_ids: list[str] | None = None
    # retrieval assist
    keywords: list[str] = field(default_factory=list)
    summary: str | None = None
    # parser-specific side-channel; never a substitute for a real column
    extras: dict[str, Any] = field(default_factory=dict)
```

Helpers used across the router:

```python
# src/ingest/_ids.py
import hashlib
from datetime import datetime, timezone

def stable_chunk_id(doc_id: str, chunker_version: str, ordinal: int) -> str:
    return hashlib.sha256(
        f"{doc_id}::{chunker_version}::{ordinal}".encode("utf-8")
    ).hexdigest()[:24]

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)
```

Token counting uses the embedding model's tokenizer (chapter 7's discipline — if a chunk
is going to `BAAI/bge-small-en-v1.5`, count tokens with that tokenizer):

```python
# src/ingest/_tokens.py
from functools import lru_cache
from transformers import AutoTokenizer

@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))
```

### 2.2 `src/ingest/router.py` — format router

```python
# src/ingest/router.py
from __future__ import annotations
import pathlib

try:
    import magic  # python-magic
    _HAS_MAGIC = True
except ImportError:
    _HAS_MAGIC = False

from .schema import Chunk
from .pdf_unstructured import ingest_pdf_unstructured
from .pdf_docling import ingest_pdf_docling
from .html_path import ingest_html
from .md_path import ingest_markdown

_EXT_FALLBACK = {
    ".pdf":  "application/pdf",
    ".html": "text/html",
    ".htm":  "text/html",
    ".md":   "text/markdown",
    ".markdown": "text/markdown",
}

def _detect_mime(path: pathlib.Path) -> str:
    if _HAS_MAGIC:
        return magic.from_file(str(path), mime=True)
    return _EXT_FALLBACK.get(path.suffix.lower(), "application/octet-stream")

def ingest(
    path: pathlib.Path,
    *,
    permission_scope: str = "public",
    pdf_backend: str = "unstructured",   # or "docling"
) -> list[Chunk]:
    """Route a file to the right parser and return normalized Chunks."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    mime = _detect_mime(path)
    if mime == "application/pdf":
        impl = ingest_pdf_unstructured if pdf_backend == "unstructured" else ingest_pdf_docling
        return impl(path, permission_scope=permission_scope)
    if mime == "text/html":
        return ingest_html(path, permission_scope=permission_scope)
    if mime == "text/markdown":
        return ingest_markdown(path, permission_scope=permission_scope)
    raise ValueError(f"unsupported mime {mime!r} for {path}")
```

### 2.3 `src/ingest/pdf_unstructured.py`

```python
# src/ingest/pdf_unstructured.py
from __future__ import annotations
import pathlib
from importlib.metadata import version

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from .schema import Chunk
from ._ids import stable_chunk_id, content_hash, now_utc
from ._tokens import count_tokens

_PARSER = "unstructured.partition_pdf"
_PARSER_VERSION = version("unstructured")
_CHUNKER = "unstructured.chunk_by_title"
_CHUNKER_VERSION = "1.0.0"  # bump when the max_characters / new_after_n_chars defaults change

_ELEMENT_MAP = {
    "Title": "heading",
    "Header": "heading",
    "NarrativeText": "paragraph",
    "UncategorizedText": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "Image": "figure_caption",
    "Formula": "formula",
    "FigureCaption": "figure_caption",
    "CodeSnippet": "code",
}

def _section_path_from_metadata(parents: list) -> list[str]:
    # chunk_by_title emits CompositeElement whose .metadata.parent_id points at a Title;
    # we walk the parent chain and emit titles top-down.
    out = []
    for p in parents:
        if getattr(p, "category", None) in ("Title", "Header"):
            out.append(p.text.strip())
    return out

def ingest_pdf_unstructured(
    path: pathlib.Path,
    *,
    permission_scope: str = "public",
) -> list[Chunk]:
    doc_id = path.stem
    elements = partition_pdf(
        filename=str(path),
        strategy="hi_res",
        infer_table_structure=True,
        languages=["eng"],
    )
    # Index elements by element_id so we can walk parent_id back to Title elements.
    by_id = {getattr(e, "id", None) or getattr(e.metadata, "element_id", None): e
             for e in elements}

    composites = chunk_by_title(
        elements,
        max_characters=2000,
        new_after_n_chars=1500,
        combine_text_under_n_chars=250,
    )

    chunks: list[Chunk] = []
    for ordinal, c in enumerate(composites):
        parent_chain = []
        cursor = getattr(c.metadata, "parent_id", None)
        while cursor and cursor in by_id:
            parent_chain.append(by_id[cursor])
            cursor = getattr(by_id[cursor].metadata, "parent_id", None)
        parent_chain.reverse()
        section_path = _section_path_from_metadata(parent_chain)

        category = getattr(c, "category", "CompositeElement")
        element_type = _ELEMENT_MAP.get(category, "paragraph")
        text = c.text or ""
        extras: dict = {}
        if element_type == "table":
            table_html = getattr(c.metadata, "text_as_html", None)
            if table_html:
                extras["text_as_html"] = table_html

        chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, ordinal)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_uri=str(path.resolve()),
            ordinal=ordinal,
            text=text,
            element_type=element_type,
            tokens=count_tokens(text),
            section_path=section_path,
            page=getattr(c.metadata, "page_number", None),
            bbox=None,  # unstructured provides metadata.coordinates; left as stretch
            permission_scope=permission_scope,
            content_hash=content_hash(text),
            parser=_PARSER,
            parser_version=_PARSER_VERSION,
            chunker=_CHUNKER,
            chunker_version=_CHUNKER_VERSION,
            ingested_at=now_utc(),
            extras=extras,
        ))
    return chunks
```

### 2.4 `src/ingest/pdf_docling.py`

```python
# src/ingest/pdf_docling.py
from __future__ import annotations
import pathlib
from importlib.metadata import version

from docling.document_converter import DocumentConverter

from .schema import Chunk
from ._ids import stable_chunk_id, content_hash, now_utc
from ._tokens import count_tokens

_PARSER = "docling.DocumentConverter"
_PARSER_VERSION = version("docling")
_CHUNKER = "docling.iterate_items"
_CHUNKER_VERSION = "1.0.0"

def _is_heading(item) -> int | None:
    """Return the heading level if this is a heading, else None."""
    label = (getattr(item, "label", "") or "").lower()
    if label in {"title", "section_header", "sectionheader"}:
        return int(getattr(item, "level", 1) or 1)
    return None

def _element_type(item) -> str:
    label = (getattr(item, "label", "") or "").lower()
    if "title" in label or "header" in label:
        return "heading"
    if "table" in label:
        return "table"
    if "figure" in label or "picture" in label:
        return "figure_caption"
    if "list" in label:
        return "list_item"
    if "code" in label:
        return "code"
    return "paragraph"

def _bbox(item) -> tuple[float, float, float, float] | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return None
    return (bbox.l, bbox.t, bbox.r, bbox.b)

def _page(item) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    return getattr(first, "page_no", None) or getattr(first, "page", None)

def ingest_pdf_docling(
    path: pathlib.Path,
    *,
    permission_scope: str = "public",
) -> list[Chunk]:
    doc_id = path.stem
    converter = DocumentConverter()
    doc = converter.convert(str(path)).document

    chunks: list[Chunk] = []
    section_stack: list[tuple[int, str]] = []  # (level, text)

    for ordinal, (item, _level) in enumerate(doc.iterate_items()):
        level = _is_heading(item)
        if level is not None:
            # Pop headings at same-or-deeper level, then push.
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, (item.text or "").strip()))

        element_type = _element_type(item)
        if element_type == "table":
            # Docling exposes table_rep.export_to_markdown() on table items.
            text = item.export_to_markdown() if hasattr(item, "export_to_markdown") else (item.text or "")
            extras = {"text_as_markdown": text}
        else:
            text = item.text or ""
            extras = {}

        section_path = [s for _, s in section_stack]

        chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, ordinal)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source_uri=str(path.resolve()),
            ordinal=ordinal,
            text=text,
            element_type=element_type,
            tokens=count_tokens(text),
            section_path=section_path,
            page=_page(item),
            bbox=_bbox(item),
            permission_scope=permission_scope,
            content_hash=content_hash(text),
            parser=_PARSER,
            parser_version=_PARSER_VERSION,
            chunker=_CHUNKER,
            chunker_version=_CHUNKER_VERSION,
            ingested_at=now_utc(),
            extras=extras,
        ))
    return chunks
```

### 2.5 `src/ingest/md_path.py` and `src/ingest/html_path.py`

```python
# src/ingest/md_path.py
from __future__ import annotations
import pathlib
from importlib.metadata import version

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .schema import Chunk
from ._ids import stable_chunk_id, content_hash, now_utc
from ._tokens import count_tokens

_PARSER = "langchain_text_splitters.MarkdownHeaderTextSplitter"
_PARSER_VERSION = version("langchain-text-splitters")
_CHUNKER = "MarkdownHeaderTextSplitter+Recursive"
_CHUNKER_VERSION = "1.0.0"
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

def ingest_markdown(
    path: pathlib.Path,
    *,
    permission_scope: str = "public",
) -> list[Chunk]:
    doc_id = path.stem
    raw = path.read_text(encoding="utf-8")
    md_splits = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(raw)

    recur = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, length_function=count_tokens
    )

    chunks: list[Chunk] = []
    ordinal = 0
    for doc in md_splits:
        section_path = [v for k, v in doc.metadata.items() if k in {"h1", "h2", "h3"} and v]
        element_type = "code" if doc.page_content.strip().startswith("```") else "paragraph"
        # Recursive split any section that exceeds the 800-token budget.
        parts = recur.split_text(doc.page_content) if count_tokens(doc.page_content) > 800 \
                else [doc.page_content]
        for part in parts:
            chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, ordinal)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_uri=str(path.resolve()),
                ordinal=ordinal,
                text=part,
                element_type=element_type,
                tokens=count_tokens(part),
                section_path=section_path,
                permission_scope=permission_scope,
                content_hash=content_hash(part),
                parser=_PARSER,
                parser_version=_PARSER_VERSION,
                chunker=_CHUNKER,
                chunker_version=_CHUNKER_VERSION,
                ingested_at=now_utc(),
            ))
            ordinal += 1
    return chunks
```

```python
# src/ingest/html_path.py
from __future__ import annotations
import pathlib
import trafilatura

from .md_path import ingest_markdown as _ingest_markdown

def ingest_html(path: pathlib.Path, *, permission_scope: str = "public"):
    """Clean HTML → Markdown via trafilatura, then feed through the Markdown path."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = trafilatura.extract(raw, output_format="markdown",
                                  include_tables=True, include_links=False) or ""
    tmp = path.with_suffix(".cleaned.md")
    tmp.write_text(cleaned, encoding="utf-8")
    try:
        chunks = _ingest_markdown(tmp, permission_scope=permission_scope)
    finally:
        tmp.unlink(missing_ok=True)
    # Rewrite source_uri back to the original HTML; keep doc_id from the HTML stem.
    for c in chunks:
        c.source_uri = str(path.resolve())
        c.doc_id = path.stem
        c.parser = "trafilatura+" + c.parser
    return chunks
```

### 2.6 `scripts/compare_pdf_parsers.py`

```python
# scripts/compare_pdf_parsers.py
from __future__ import annotations
import pathlib
import statistics
import sys
import time

from ingest.pdf_unstructured import ingest_pdf_unstructured
from ingest.pdf_docling import ingest_pdf_docling

def _row(name: str, chunks, wall_s: float) -> str:
    tokens = [c.tokens for c in chunks]
    tables = sum(1 for c in chunks if c.element_type == "table")
    headings = sum(1 for c in chunks if c.element_type == "heading")
    depths = [len(c.section_path) for c in chunks if c.section_path]
    mean_depth = statistics.mean(depths) if depths else 0.0
    avg_tokens = statistics.mean(tokens) if tokens else 0
    return (f"{name:<13} | {len(chunks):>8} | {avg_tokens:>10.1f} | "
            f"{tables:>8} | {headings:>10} | {mean_depth:>22.2f} | {wall_s:>11.2f}")

def main(pdf: str):
    path = pathlib.Path(pdf)
    print("parser        | n_chunks | avg_tokens | n_tables | n_headings | mean_section_path_depth | wall_clock_s")
    print("-" * 110)

    t0 = time.perf_counter()
    u = ingest_pdf_unstructured(path)
    t_u = time.perf_counter() - t0
    print(_row("unstructured", u, t_u))

    t0 = time.perf_counter()
    d = ingest_pdf_docling(path)
    t_d = time.perf_counter() - t0
    print(_row("docling", d, t_d))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/corpus/handbook.pdf")
```

Commit the captured output to `docs/parser-comparison.txt` so a reviewer can see the shape
without running the script.

### 2.7 `tests/test_ingest.py`

```python
# tests/test_ingest.py
from __future__ import annotations
import pathlib

import pytest

from ingest.router import ingest

CORPUS = pathlib.Path("data/corpus")
FILES = [CORPUS / "handbook.pdf", CORPUS / "notes.md", CORPUS / "article.html"]

_REQUIRED = [
    "doc_id", "source_uri", "ordinal", "text", "element_type",
    "tokens", "content_hash", "parser", "chunker_version", "ingested_at",
]

@pytest.fixture(scope="module")
def all_chunks():
    return {f.name: ingest(f) for f in FILES if f.exists()}

def _raw_text(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pypdf
        return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages)
    if path.suffix.lower() in {".html", ".htm"}:
        import trafilatura
        return trafilatura.extract(path.read_text(encoding="utf-8", errors="replace")) or ""
    return path.read_text(encoding="utf-8")

def test_no_content_silently_lost(all_chunks):
    for name, chunks in all_chunks.items():
        source = CORPUS / name
        raw_len = max(1, len(_raw_text(source)))
        kept = sum(len(c.text) for c in chunks)
        assert kept >= 0.90 * raw_len, (name, kept, raw_len)

def test_every_chunk_has_required_metadata(all_chunks):
    for name, chunks in all_chunks.items():
        assert chunks, name
        for c in chunks:
            for field in _REQUIRED:
                val = getattr(c, field)
                assert val not in (None, "", []), (name, c.chunk_id, field)
        # section_path discipline: PDF + MD non-empty; HTML tolerated flat.
        if name.endswith((".pdf", ".md")):
            assert all(c.section_path for c in chunks if c.element_type != "heading"), name

def test_stable_chunk_ids_across_runs():
    pdf = CORPUS / "handbook.pdf"
    a = [c.chunk_id for c in ingest(pdf)]
    b = [c.chunk_id for c in ingest(pdf)]
    assert a == b
    assert len(a) == len(set(a))  # chunk_ids are unique within a run

def test_pdf_chunks_carry_page_numbers(all_chunks):
    for c in all_chunks["handbook.pdf"]:
        assert c.page is not None and c.page >= 1, c.chunk_id
```

### 2.8 `src/ingest/README.md` — per-format parser-of-record

```markdown
# ingest/ — parser of record per format

| Format   | MIME              | Default parser                                 | Rationale (one sentence) |
|----------|-------------------|------------------------------------------------|---------------------------|
| PDF      | `application/pdf` | `unstructured.partition_pdf` + `chunk_by_title`| Hi-res layout with table extraction is strong out of the box; Docling is wired as an alternate for structural-chunk ablations. |
| HTML     | `text/html`       | `trafilatura` → Markdown → Markdown path       | `trafilatura` beats naive `BeautifulSoup` boilerplate removal; the downstream Markdown splitter then uses the article's heading structure. |
| Markdown | `text/markdown`   | `MarkdownHeaderTextSplitter` + recursive split | Heading-aware primary split; recursive split handles any section that overflows 800 tokens. |

Swap the PDF backend by passing `pdf_backend="docling"` to `ingest(...)`. The two backends
emit the same `Chunk` schema; the point of the ablation is to pick a default for *this* corpus
based on the comparison script output in `docs/parser-comparison.txt`.
```

## 3. Validation steps

1. Dependencies installed: `uv add "unstructured[pdf,html,md,docx]" docling pypdf python-magic
   trafilatura langchain-text-splitters transformers` (plus the mod-101 toolchain from
   exercise-05).
2. Corpus assembled under `data/corpus/` (three files).
3. `uv run python -m scripts.compare_pdf_parsers data/corpus/handbook.pdf > docs/parser-comparison.txt`
   — table has both rows with non-zero `n_chunks`, at least one `n_tables > 0` on the PDF
   containing a table, and non-zero `mean_section_path_depth`.
4. `uv run pytest -q tests/test_ingest.py` — all four tests pass.
5. Spot-check one PDF chunk: `chunks[0].section_path` is a non-empty list of heading strings,
   `page` is an integer ≥ 1, `content_hash` is a 64-char hex, `parser` and `chunker_version`
   are non-empty strings, `chunk_id` is 24 hex chars.
6. Re-run `pytest` immediately. `test_stable_chunk_ids_across_runs` still passes — it will fail
   loudly if any `chunk_id` depended on wall-clock time, PID, or a Python `uuid4`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Router | MIME dispatch covers PDF, HTML, Markdown; raises on unknown MIME. |
| PDF unstructured | Returns `list[Chunk]` with non-empty `section_path`, `page ≥ 1`, at least one `element_type="table"` chunk on a PDF with a table. |
| PDF Docling | Same `Chunk` shape as unstructured; heading propagation populates `section_path`; `bbox` populated when Docling provides it. |
| Markdown | Heading-aware splitting; code blocks emitted as `element_type="code"`; recursive split triggers above 800 tokens. |
| HTML | `trafilatura.extract` ran first; chunks have `source_uri` pointing at the HTML path (not the tmp Markdown). |
| Comparison script | Outputs the required columns including wall-clock; committed to `docs/parser-comparison.txt`. |
| Tests | `test_no_content_silently_lost`, `test_every_chunk_has_required_metadata`, `test_stable_chunk_ids_across_runs`, `test_pdf_chunks_carry_page_numbers` all pass. |
| Discipline | Every chunk has `content_hash`; no `dict[str, Any]` fields bypass the schema (the `extras` escape hatch is used only for parser-specific side-channels, never for `doc_id` or `page`). |

Fail-fast smells:

- `chunk_id = uuid4()`. Non-deterministic IDs fail `test_stable_chunk_ids_across_runs`.
- `page = 0` or `page = -1` for Markdown chunks. The right value is `None`; the schema is
  explicitly nullable.
- `section_path = ""` instead of `[]` for the empty case. List, not string.
- HTML chunks carrying the temp `*.cleaned.md` path in `source_uri`. The source URI is the
  *original* HTML file; the Markdown intermediary is an implementation detail.
- A Docling chunk whose `section_path` includes a heading from a later section (pop discipline
  failed). Headings pop at same-or-deeper level, push *after* popping.

## 5. Common mistakes

- **Treating the PDF backends as "pick one and never look back."** The exercise asks you to run
  both because production pipelines usually end up routing by document shape: Docling for
  table-heavy reports, `unstructured` for text-heavy policy PDFs. Running the comparison once
  is cheap insurance against the wrong default.
- **Mixing table HTML into the top-level `text` field.** Keep the Markdown or plain-text
  rendering in `text` (what will be embedded) and the full HTML or cell matrix in `extras`
  (what the assembler can use for presentation). Embedding HTML tokens into a dense vector
  wastes the model's budget on `<td>` boilerplate.
- **Building `chunk_id` from file bytes.** The chunk ID must be stable across re-ingests of
  the same `(doc_id, chunker_version, ordinal)`; hashing the file bytes makes every re-ingest
  change every ID and torches the incremental-reindex story.
- **Running `partition_pdf(strategy="fast")` by default.** Fast mode drops table structure and
  layout info — that's fine for a text-only prose PDF but silently breaks the table test. The
  exercise explicitly calls for `strategy="hi_res"`.
- **Calling `trafilatura.extract` without `include_tables=True`.** Default behavior drops table
  content; this trips the 90% round-trip test on HTML with substantial tabular content.
- **Threading `section_path` via a mutable global in the Docling pass.** Iterate items in order,
  maintain a stack, pop on same-or-deeper level — treat the stack like you would an HTML DOM
  cursor. A global state leaks across documents.
- **Counting tokens with `len(text.split())`.** The retriever will embed under the embedder's
  tokenizer; counting tokens with whitespace splits produces numbers that have no relationship
  to the real vector-store budget. Use the embedder's tokenizer (chapter 7).

## 6. References

- Lesson sources in this module:
  - `lessons/mod-102-document-ingestion-and-chunking/01-ingestion-overview-and-format-matrix.md`
    — the format-matrix the router implements.
  - `lessons/mod-102-document-ingestion-and-chunking/02-parser-toolchain-unstructured-pymupdf-docling-nougat.md`
    — side-by-side on the two PDF backends.
  - `lessons/mod-102-document-ingestion-and-chunking/07-metadata-as-first-class-retrieval-fields.md`
    — the `Chunk` schema used throughout.
- `unstructured` — [docs.unstructured.io](https://docs.unstructured.io/), and the Python
  library on GitHub: [github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured).
- Docling — [github.com/docling-project/docling](https://github.com/docling-project/docling)
  and its docs site [docling-project.github.io/docling/](https://docling-project.github.io/docling/).
- `pypdf` — [pypdf.readthedocs.io](https://pypdf.readthedocs.io/).
- `trafilatura` — [trafilatura.readthedocs.io](https://trafilatura.readthedocs.io/).
- LangChain `MarkdownHeaderTextSplitter` —
  [python.langchain.com/docs/how_to/markdown_header_metadata_splitter/](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/).
- `python-magic` — [github.com/ahupp/python-magic](https://github.com/ahupp/python-magic).
- `BAAI/bge-small-en-v1.5` — [huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
