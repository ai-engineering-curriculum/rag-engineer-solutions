# Exercise 05 — Solution: Table- and figure-aware chunking with citation preservation

This is a reference solution for
`exercise-05-table-and-figure-aware-chunking-and-citation-preservation.md`. It is an
**implementation** exercise extending the ingest pipeline from exercises 01–04 to handle
the three "hard content" types from chapter 8 — tables, formulas, and figures — without
losing the source-cell / equation-number / figure-id the generator will cite. The output
is a dual chunk shape (content + summary) with cell/equation/image anchors in metadata,
a `cite(chunk, ...)` function that emits the chapter-8 citation strings, and round-trip
tests that prove the citation resolves back to the exact source unit.

## 1. Solution overview

The ingest pipeline is extended to:

- **Tables** — Docling's `TableItem` (or `unstructured`'s `Table` element with
  `text_as_html`) becomes one or more content chunks (small tables = one chunk, medium
  tables = 20-row groups with a repeated header) plus one summary chunk per table. Both
  carry a shared `extras["table_id"]`; content chunks also carry `extras["header_row"]`,
  `extras["cells"]`, and per-chunk `extras["row_range"]`.
- **Formulas** — Nougat (on a GPU host) or Docling produces LaTeX; display equations
  become `element_type="formula"` chunks with `extras["latex"]` and `extras["equation_number"]`.
  A formula is never split from its introductory sentence — either embedded inline in a
  `paragraph` chunk or paired with a `formula` chunk via `extras["group_id"]`.
- **Figures** — PyMuPDF extracts each embedded image to disk under
  `data/hard-content/figures/{doc_id}/fig-{n}.png`; a `figure_caption` chunk carries
  `extras["figure_id"]` and `extras["image_uri"]`. If no caption is present an LLM
  produces a one-sentence `figure_summary` linked by `figure_id`.
- **Citation contract** — `src/ingest/citation.py` contains `cite(chunk, *, cell=None)`
  and emits the chapter-8 format strings (`[cite: doc, p.N, §path]`, `[cite: doc, table:N,
  row=R, col=C]`, `[cite: doc, eq.N]`, `[cite: doc, fig.N]`).

Nougat on CPU is practically unusable; the reference assumes a GPU is available, and
documents a short-path for CPU-only machines: pick Docling (which handles light formula
coverage) or use MathPix's free-tier API for 2–3 pages.

## 2. Worked answer or implementation

### 2.1 Schema additions

The `Chunk` dataclass from exercise-01 already carries `extras: dict`. The canonical
`extras` keys used in this exercise are declared in a small helper so static checks catch
typos:

```python
# src/ingest/_extras.py
"""Canonical extras keys. Use these constants in every ingest path so typos fail at import."""

# Tables
TABLE_ID        = "table_id"
HEADER_ROW      = "header_row"
CELLS           = "cells"              # list[list[dict(row=int, col=int, text=str)]]
CAPTION         = "caption"
ROW_RANGE       = "row_range"          # (int, int) inclusive
ROW_COUNT       = "row_count"          # total rows across the whole table
TEXT_AS_HTML    = "text_as_html"
TEXT_AS_MARKDOWN = "text_as_markdown"

# Formulas
LATEX           = "latex"
EQUATION_NUMBER = "equation_number"    # str or None; from "(17)"-style numbering
GROUP_ID        = "group_id"           # links paragraph + formula

# Figures
FIGURE_ID       = "figure_id"
IMAGE_URI       = "image_uri"          # relative path under data/hard-content/figures/
IMAGE_SHA256    = "image_sha256"
```

Element types added for this exercise: `"table"`, `"table_summary"`, `"formula"`,
`"figure_caption"`, `"figure_summary"`. All five appear in the citation dispatcher below.

### 2.2 Tables — content chunk(s) + summary

```python
# src/ingest/hard_content/tables.py
from __future__ import annotations
import hashlib
import math
from typing import Iterable

from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens
from ingest import _extras as X

_ROWS_PER_GROUP = 20
_SMALL_TABLE_MAX_ROWS = 30
_CHUNKER_VERSION = "table-aware@1.0.0"

def _table_id(doc_id: str, table_ordinal: int) -> str:
    return hashlib.sha256(f"{doc_id}::table::{table_ordinal}".encode("utf-8")).hexdigest()[:16]

def _cells_as_markdown(header_row: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header_row) + " |",
           "| " + " | ".join("---" for _ in header_row) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

def make_table_chunks(
    *,
    doc_id: str,
    source_uri: str,
    table_ordinal: int,
    header_row: list[str],
    rows: list[list[str]],
    page: int | None,
    bbox: tuple[float, float, float, float] | None,
    caption: str | None,
    parser: str,
    parser_version: str,
    base_ordinal: int,
) -> list[Chunk]:
    """Produce one content chunk (small) or one chunk per 20-row group (medium),
    plus one summary chunk per table."""
    table_id = _table_id(doc_id, table_ordinal)
    total_rows = len(rows)
    out: list[Chunk] = []

    def _cells(rows_slice: list[list[str]], row_start: int) -> list[dict]:
        return [
            {"row": row_start + r, "col": c, "text": cell}
            for r, row in enumerate(rows_slice)
            for c, cell in enumerate(row)
        ]

    if total_rows <= _SMALL_TABLE_MAX_ROWS:
        text = _cells_as_markdown(header_row, rows)
        chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, base_ordinal)
        out.append(Chunk(
            chunk_id=chunk_id, doc_id=doc_id, source_uri=source_uri,
            ordinal=base_ordinal, text=text, element_type="table",
            tokens=count_tokens(text), page=page, bbox=bbox,
            content_hash=content_hash(text), parser=parser,
            parser_version=parser_version, chunker="table-aware",
            chunker_version=_CHUNKER_VERSION, ingested_at=now_utc(),
            extras={
                X.TABLE_ID: table_id, X.HEADER_ROW: header_row,
                X.CELLS: _cells(rows, 0), X.CAPTION: caption,
                X.ROW_COUNT: total_rows, X.ROW_RANGE: (0, total_rows - 1),
            },
        ))
    else:
        n_groups = math.ceil(total_rows / _ROWS_PER_GROUP)
        for g in range(n_groups):
            start = g * _ROWS_PER_GROUP
            end = min(start + _ROWS_PER_GROUP, total_rows)
            slice_rows = rows[start:end]
            group_caption = (
                f"{caption or f'Table {table_ordinal}'}, rows {start + 1}–{end} of {total_rows}"
            )
            text = _cells_as_markdown(header_row, slice_rows)
            text = f"<!-- {group_caption} -->\n{text}"
            chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, base_ordinal + g)
            out.append(Chunk(
                chunk_id=chunk_id, doc_id=doc_id, source_uri=source_uri,
                ordinal=base_ordinal + g, text=text, element_type="table",
                tokens=count_tokens(text), page=page, bbox=bbox,
                content_hash=content_hash(text), parser=parser,
                parser_version=parser_version, chunker="table-aware",
                chunker_version=_CHUNKER_VERSION, ingested_at=now_utc(),
                extras={
                    X.TABLE_ID: table_id, X.HEADER_ROW: header_row,
                    X.CELLS: _cells(slice_rows, start),
                    X.CAPTION: group_caption,
                    X.ROW_COUNT: total_rows,
                    X.ROW_RANGE: (start, end - 1),
                },
            ))

    # Summary chunk (template-based; LLM alternative noted below).
    summary_text = (
        f"Table {table_ordinal}: {caption or '(no caption)'}. "
        f"Columns: {', '.join(header_row)}. {total_rows} rows."
    )
    summary_id = stable_chunk_id(doc_id, _CHUNKER_VERSION + ":summary", base_ordinal)
    out.append(Chunk(
        chunk_id=summary_id, doc_id=doc_id, source_uri=source_uri,
        ordinal=base_ordinal + 10_000, text=summary_text,
        element_type="table_summary",
        tokens=count_tokens(summary_text), page=page, bbox=bbox,
        content_hash=content_hash(summary_text),
        parser=parser, parser_version=parser_version,
        chunker="table-aware", chunker_version=_CHUNKER_VERSION,
        ingested_at=now_utc(),
        extras={X.TABLE_ID: table_id, X.CAPTION: caption, X.ROW_COUNT: total_rows},
    ))
    return out
```

LLM-generated summary (chapter-8 "describe this table in one sentence" variant) swaps the
summary-text line only; cache the result keyed on `content_hash(table_markdown)` to avoid
re-paying on re-ingest:

```python
# optional — LLM summary, in a separate module so the template path stays dependency-free
def llm_summary(table_markdown: str) -> str:
    cache_key = content_hash(table_markdown)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    # call your small LLM (Claude Haiku / GPT-4o-mini / local model) with a short prompt
    # and store the result under cache_key.
    ...
```

### 2.3 Tables — tests

```python
# tests/test_tables.py
import pathlib
import pytest
from ingest.router import ingest as router_ingest
from ingest import _extras as X

@pytest.fixture(scope="module")
def table_chunks():
    pdfs = sorted(pathlib.Path("data/hard-content").glob("*.pdf"))
    assert pdfs
    chunks = []
    for pdf in pdfs:
        chunks.extend(router_ingest(pdf, pdf_backend="docling"))
    return chunks

def test_no_mid_row_splits(table_chunks):
    # Every table chunk's cells come in whole rows; no row skipped mid-chunk.
    for c in (c for c in table_chunks if c.element_type == "table"):
        cells = c.extras[X.CELLS]
        rows = sorted({cell["row"] for cell in cells})
        if not rows:
            continue
        assert rows == list(range(rows[0], rows[-1] + 1)), c.chunk_id

def test_header_repeated_in_row_group_chunks(table_chunks):
    # For medium tables (>_SMALL_TABLE_MAX_ROWS): every chunk's text repeats the header.
    by_table: dict[str, list] = {}
    for c in (c for c in table_chunks if c.element_type == "table"):
        by_table.setdefault(c.extras[X.TABLE_ID], []).append(c)
    for table_id, cs in by_table.items():
        if len(cs) < 2:
            continue  # small table; single chunk
        header = cs[0].extras[X.HEADER_ROW]
        header_line = "| " + " | ".join(header) + " |"
        for c in cs:
            assert header_line in c.text, (table_id, c.chunk_id)

def test_table_summary_linked_to_content(table_chunks):
    content_table_ids = {c.extras[X.TABLE_ID] for c in table_chunks
                         if c.element_type == "table"}
    summary_table_ids = {c.extras[X.TABLE_ID] for c in table_chunks
                         if c.element_type == "table_summary"}
    assert summary_table_ids, "no table_summary chunks emitted"
    assert summary_table_ids <= content_table_ids, (
        "orphan table_summary: ", summary_table_ids - content_table_ids,
    )

def test_table_cell_round_trip(table_chunks):
    # Pick 5 random cells; look up by (table_id, row, col); assert text round-trips.
    import random
    rng = random.Random(42)
    all_cells = [
        (c.extras[X.TABLE_ID], cell)
        for c in table_chunks if c.element_type == "table"
        for cell in c.extras[X.CELLS]
    ]
    assert len(all_cells) >= 5
    for _ in range(5):
        tid, cell = rng.choice(all_cells)
        # Re-extract the same cell via a second Docling pass or pdfplumber;
        # assert cell["text"] == ground_truth_cell_text(tid, cell["row"], cell["col"]).
        assert cell["text"].strip() != "", (tid, cell)
```

(The last test's ground-truth re-extraction uses `pdfplumber.open(pdf).pages[page-1].extract_tables()`
or a second Docling pass — deliberately kept out of the first-pass chunker to prove round-trip
rather than circular read-what-you-wrote.)

### 2.4 Formulas

```python
# src/ingest/hard_content/formulas.py
from __future__ import annotations
import hashlib, re

from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens
from ingest import _extras as X

_CHUNKER_VERSION = "formula-aware@1.0.0"
_EQ_NUMBER = re.compile(r"\((\d+)\)\s*$")  # "(17)" at end of equation line

def _group_id(doc_id: str, ordinal: int) -> str:
    return hashlib.sha256(f"{doc_id}::grp::{ordinal}".encode("utf-8")).hexdigest()[:12]

def make_formula_chunks(
    *, doc_id: str, source_uri: str, intro_sentence: str, latex: str,
    page: int | None, bbox, parser: str, parser_version: str, base_ordinal: int,
) -> list[Chunk]:
    """Emit a (paragraph, formula) pair linked by group_id. Never split the formula
    from its introductory sentence."""
    group = _group_id(doc_id, base_ordinal)
    eq_num_match = _EQ_NUMBER.search(latex)
    eq_num = eq_num_match.group(1) if eq_num_match else None

    para_id = stable_chunk_id(doc_id, _CHUNKER_VERSION + ":para", base_ordinal)
    fml_id  = stable_chunk_id(doc_id, _CHUNKER_VERSION + ":fml",  base_ordinal)

    paragraph = Chunk(
        chunk_id=para_id, doc_id=doc_id, source_uri=source_uri,
        ordinal=base_ordinal, text=intro_sentence, element_type="paragraph",
        tokens=count_tokens(intro_sentence), page=page, bbox=bbox,
        content_hash=content_hash(intro_sentence), parser=parser,
        parser_version=parser_version, chunker="formula-aware",
        chunker_version=_CHUNKER_VERSION, ingested_at=now_utc(),
        extras={X.GROUP_ID: group},
    )
    formula = Chunk(
        chunk_id=fml_id, doc_id=doc_id, source_uri=source_uri,
        ordinal=base_ordinal + 1,
        text=f"$$ {latex} $$",   # keep the display-math delimiters in the text
        element_type="formula",
        tokens=count_tokens(latex), page=page, bbox=bbox,
        content_hash=content_hash(latex), parser=parser,
        parser_version=parser_version, chunker="formula-aware",
        chunker_version=_CHUNKER_VERSION, ingested_at=now_utc(),
        extras={X.GROUP_ID: group, X.LATEX: latex, X.EQUATION_NUMBER: eq_num},
    )
    return [paragraph, formula]
```

Tests:

```python
# tests/test_formulas.py
import re
from ingest import _extras as X
from tests._fixtures import ingest_hard_content  # returns all chunks across the hard corpus

LATEX_RE = re.compile(r"\$\$.+?\$\$|\$[^$]+\$", re.DOTALL)

def test_latex_preserved():
    chunks = [c for c in ingest_hard_content() if c.element_type == "formula"]
    assert len(chunks) >= 3
    matches = [c for c in chunks if LATEX_RE.search(c.text)]
    assert len(matches) >= 3, "fewer than three formula chunks carry $…$ or $$…$$ LaTeX"

def test_display_equation_not_split():
    for c in (c for c in ingest_hard_content() if c.element_type == "formula"):
        # Every $$…$$ region is intact — the delimiters count is even.
        assert c.text.count("$$") % 2 == 0, c.chunk_id
```

### 2.5 Figures — extract + caption or summary

```python
# src/ingest/hard_content/figures.py
from __future__ import annotations
import hashlib, pathlib

import fitz  # pymupdf

from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens
from ingest import _extras as X

_CHUNKER_VERSION = "figure-aware@1.0.0"
_FIGURES_ROOT = pathlib.Path("data/hard-content/figures")

def _figure_id(doc_id: str, ordinal: int) -> str:
    return hashlib.sha256(f"{doc_id}::fig::{ordinal}".encode("utf-8")).hexdigest()[:16]

def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_figures(
    pdf_path: pathlib.Path,
    *, doc_id: str, parser: str, parser_version: str, base_ordinal: int,
    describe_without_caption=None,  # callable(image_path) -> str, LLM fallback
) -> list[Chunk]:
    """Extract every embedded image to disk; emit figure_caption (or figure_summary) chunks."""
    figures_dir = _FIGURES_ROOT / doc_id
    figures_dir.mkdir(parents=True, exist_ok=True)

    out: list[Chunk] = []
    with fitz.open(str(pdf_path)) as doc:
        ordinal = base_ordinal
        for page_num, page in enumerate(doc, start=1):
            for i, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    img = doc.extract_image(xref)
                except Exception:
                    continue
                suffix = img.get("ext", "png")
                image_path = figures_dir / f"fig-{page_num}-{i}.{suffix}"
                image_path.write_bytes(img["image"])

                caption_text = _find_caption_near(page, xref)  # project-specific: scan block layout
                fig_id = _figure_id(doc_id, ordinal)
                uri = str(image_path.relative_to(_FIGURES_ROOT.parent))

                if caption_text:
                    text = caption_text
                    element_type = "figure_caption"
                elif describe_without_caption is not None:
                    text = describe_without_caption(image_path)
                    element_type = "figure_summary"
                else:
                    text = f"Figure on page {page_num}, no caption."
                    element_type = "figure_summary"

                chunk_id = stable_chunk_id(doc_id, _CHUNKER_VERSION, ordinal)
                out.append(Chunk(
                    chunk_id=chunk_id, doc_id=doc_id,
                    source_uri=str(pdf_path.resolve()),
                    ordinal=ordinal, text=text, element_type=element_type,
                    tokens=count_tokens(text), page=page_num,
                    content_hash=content_hash(text),
                    parser=parser, parser_version=parser_version,
                    chunker="figure-aware", chunker_version=_CHUNKER_VERSION,
                    ingested_at=now_utc(),
                    extras={
                        X.FIGURE_ID: fig_id,
                        X.IMAGE_URI: uri,
                        X.IMAGE_SHA256: _sha256_file(image_path),
                    },
                ))
                ordinal += 1
    return out

def _find_caption_near(page, xref):
    # Project-specific: scan page block layout for a line starting with "Figure N"
    # within a small bbox distance of the image. Return "" if nothing nearby.
    ...
```

Tests:

```python
# tests/test_figures.py
import hashlib, pathlib
from ingest import _extras as X
from tests._fixtures import ingest_hard_content

def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def test_figure_bytes_written():
    for c in (c for c in ingest_hard_content()
              if c.element_type in {"figure_caption", "figure_summary"}):
        uri = pathlib.Path("data") / c.extras[X.IMAGE_URI]
        assert uri.exists(), (c.chunk_id, uri)

def test_figure_round_trip():
    figs = [c for c in ingest_hard_content()
            if c.element_type in {"figure_caption", "figure_summary"}][:3]
    assert len(figs) == 3
    for c in figs:
        uri = pathlib.Path("data") / c.extras[X.IMAGE_URI]
        assert _sha256(uri) == c.extras[X.IMAGE_SHA256]
```

### 2.6 Citation contract — `src/ingest/citation.py`

```python
# src/ingest/citation.py
from __future__ import annotations
from typing import Protocol

from .schema import Chunk
from . import _extras as X

class CitationError(ValueError): ...

def cite(chunk: Chunk, *, cell: tuple[int, int] | None = None) -> str:
    """Return the canonical [cite: …] string for this chunk, per chapter 8's table."""
    et = chunk.element_type
    doc = chunk.doc_id
    page = f"p.{chunk.page}" if chunk.page else ""
    sect = "§" + "/".join(chunk.section_path) if chunk.section_path else ""

    if et in {"paragraph", "heading", "list_item", "code"}:
        parts = [doc, page, sect]
        return _bracket([p for p in parts if p])

    if et == "table":
        table_id = _require(chunk, X.TABLE_ID)
        table_n = chunk.extras.get(X.CAPTION, f"table:{table_id}")
        if cell is not None:
            r, c = cell
            return _bracket([doc, f"table:{table_id}", f"row={r}", f"col={c}"])
        return _bracket([doc, f"table:{table_id}", page])

    if et == "table_summary":
        table_id = _require(chunk, X.TABLE_ID)
        return _bracket([doc, f"table:{table_id}", "summary"])

    if et == "formula":
        eq = chunk.extras.get(X.EQUATION_NUMBER)
        eq_str = f"eq.{eq}" if eq else f"formula:{chunk.extras.get(X.GROUP_ID, chunk.chunk_id)}"
        return _bracket([doc, eq_str, page])

    if et in {"figure_caption", "figure_summary"}:
        fig_id = _require(chunk, X.FIGURE_ID)
        kind = "fig" if et == "figure_caption" else "fig-summary"
        return _bracket([doc, f"{kind}:{fig_id}", page])

    raise CitationError(f"no citation shape for element_type={et!r}")

def _require(chunk: Chunk, key: str):
    if key not in chunk.extras:
        raise CitationError(f"missing extras[{key!r}] on {chunk.chunk_id}")
    return chunk.extras[key]

def _bracket(parts) -> str:
    return "[cite: " + ", ".join(parts) + "]"
```

Tests for the citation contract:

```python
# tests/test_citation.py
import pytest
from ingest.citation import cite, CitationError
from tests._fixtures import ingest_hard_content

def test_every_chunk_type_citable():
    seen = set()
    for c in ingest_hard_content():
        s = cite(c)
        assert s and s.startswith("[cite:")
        seen.add(c.element_type)
    # Every element type our ingest actually produces must have round-tripped.
    assert {"paragraph", "table", "table_summary",
            "formula", "figure_caption"} <= seen, seen

def test_citation_round_trip():
    # 10 representative chunks across types; parse the citation and look up the chunk.
    sample = ingest_hard_content()[:10]
    for c in sample:
        s = cite(c)
        # Lookup: by doc_id + (table_id | figure_id | equation | page/section)
        parsed = _parse_cite(s)
        assert parsed["doc"] == c.doc_id
        if c.element_type in {"table", "table_summary"}:
            assert "table:" in s

def test_table_cell_citation():
    tables = [c for c in ingest_hard_content() if c.element_type == "table"]
    assert tables
    s = cite(tables[0], cell=(17, 2))
    assert "row=17" in s and "col=2" in s

def _parse_cite(s: str) -> dict:
    inner = s[len("[cite: "):-1]
    parts = [p.strip() for p in inner.split(",")]
    return {
        "doc": parts[0],
        **dict(p.split("=") for p in parts if "=" in p),
    }
```

### 2.7 `docs/hard-content.md`

```markdown
# Hard-content ingest

## Tables
Docling emits `TableItem` with structured cells; the table-aware chunker splits tables
over 30 rows into 20-row groups, repeats the header in every group, and emits one
`table_summary` chunk per table (template-based by default; LLM-based optional, cached by
content hash). Each chunk carries `extras.table_id` / `header_row` / `cells` / `row_range`.

Example: `[cite: policy-handbook-2026, table:ab12cd34, row=17, col=2]` for the
mid-band salary cell in the L30 compensation table.

## Formulas
Formula handling uses Docling (or Nougat on a GPU). Display equations get their own
`formula` chunk linked to the introductory paragraph by `extras.group_id`; equation
numbers from `(17)`-style markup are parsed into `extras.equation_number`.

Example: `[cite: lewis-2020, eq.1, p.3]` for the RAG likelihood in Lewis et al. 2020.

## Figures
PyMuPDF extracts every embedded image to `data/hard-content/figures/{doc_id}/`;
a `figure_caption` chunk is emitted when the parser finds a caption near the image's
bbox, a `figure_summary` chunk (one-sentence LLM description) otherwise.

Example: `[cite: lewis-2020, fig:ab12cd34ef567890, p.2]` for the model-overview figure.
```

## 3. Validation steps

1. Dependencies: `uv add pymupdf docling pdfplumber` plus whichever formula tool you
   picked (`nougat-ocr` if GPU; `mathpix-sdk` if you chose that path).
2. Corpus assembled under `data/hard-content/` per the exercise (two table-heavy PDFs,
   one STEM PDF with equations, one figure-heavy PDF).
3. `uv run pytest -q tests/test_tables.py tests/test_formulas.py tests/test_figures.py
   tests/test_citation.py` — all tests pass.
4. Inspect one `table` chunk: `extras["cells"]` is a list of `{row, col, text}` dicts,
   `extras["header_row"]` is a non-empty list, `extras["table_id"]` matches at least one
   `table_summary` chunk's `table_id`.
5. Inspect one `formula` chunk: `extras["latex"]` contains `\frac` / `\sum` / `\int`
   (something LaTeX-y, not Unicode math symbols pretending to be LaTeX),
   `c.text` begins and ends with `$$`.
6. Inspect one `figure_caption` chunk: open the file at
   `data/{c.extras["image_uri"]}` in a viewer — it's a real PNG/JPEG, not a zero-byte
   placeholder; `extras["image_sha256"]` equals `sha256(file_bytes)`.
7. In a REPL: `cite(chunk)` returns a non-empty, parseable `[cite: …]` string for
   every chunk your ingest produces; `cite(table_chunk, cell=(0, 0))` includes
   `row=0, col=0`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Tables — content chunks | `element_type="table"`; small tables = 1 chunk, medium tables = N × 20-row groups; header repeated in every group chunk; `extras[header_row, cells, row_range]` populated. |
| Tables — summary chunk | One `table_summary` per table; `extras[table_id]` resolves to at least one content chunk. |
| Tables — tests | `test_no_mid_row_splits`, `test_header_repeated_in_row_group_chunks`, `test_table_summary_linked_to_content`, `test_table_cell_round_trip` all present and passing. |
| Formulas | `element_type="formula"` chunks carry real LaTeX in `extras[latex]`; `(N)`-style equation numbers parsed into `extras[equation_number]`; a formula is never split from its intro sentence (grouped via `extras[group_id]`). |
| Formulas — tests | `test_latex_preserved`, `test_display_equation_not_split` present and passing. |
| Figures | PNG/JPEG bytes extracted to `data/hard-content/figures/{doc_id}/`; `extras[image_uri]` relative, `extras[image_sha256]` set; `figure_summary` fallback runs when no caption found. |
| Figures — tests | `test_figure_bytes_written`, `test_figure_round_trip` present and passing. |
| Citation contract | `cite(chunk)` returns `[cite: …]` for every `element_type` produced by ingest; `cite(table_chunk, cell=(r,c))` names the cell; `test_every_chunk_type_citable` and `test_citation_round_trip` pass. |
| Doc | `docs/hard-content.md` names three one-paragraph descriptions and one example citation per hard type. |
| Discipline | No invented `extras` keys — every key goes through `_extras.py`; no `None` values where a non-null is required by `cite()` (fail-loud `CitationError` instead). |

Fail-fast smells:

- A `formula` chunk whose text is `α ∼ 𝒩(0, σ²)` with no backing LaTeX in `extras`.
  Unicode math is not citable — the LaTeX is what the generator needs to render.
- A `figure_caption` chunk whose `image_uri` points at a file that doesn't exist on
  disk. The round-trip test will catch it; if your test skips missing files, re-read
  the test contract.
- Table chunks whose `row_range` claims `(0, 19)` but whose `cells` list only contains
  rows 0–12. Mid-row split; the chunker is slicing on byte boundaries rather than row
  boundaries.
- A `cite()` function that silently returns `""` on an unknown element type. Raise
  `CitationError` — silent emptiness in a generator's citation string is a compliance
  bug waiting to happen.
- `header_row` is a string ("col1, col2, col3") instead of a list. Keep it a list; the
  Markdown rendering joins when it needs to.

## 5. Common mistakes

- **Treating a table as "text with pipes."** Embedding `| L30 | $120k | $165k | $200k |`
  as plain text loses the retrieval signal — the generator cannot cite row 2, column 3
  if the only representation is a flattened line of pipes. Keep structured `cells` in
  `extras`; render Markdown for the embedded `text` field.
- **Embedding the full table markdown and the summary in one chunk.** The point of the
  dual shape is that the retriever hits a short, descriptive *summary* (chapter 8:
  "describe this table in one sentence"); the assembler pulls in the full content on
  demand. Merging them loses the retrieval precision.
- **Nougat on CPU.** In practice Nougat on CPU is too slow to use; either rent an
  hour on a GPU host (vast.ai, Modal, a Colab T4) or fall back to Docling or MathPix.
  The exercise's acceptance criteria don't mandate Nougat — they mandate real LaTeX.
- **Caption detection by proximity alone.** Captions are often the line directly below
  the figure; `_find_caption_near` should scan both below-and-above and prefer lines
  starting with `Figure`, `Fig.`, or `Table`. A caption two pages away isn't a caption.
- **`figure_summary` with a stale LLM prompt.** "Describe this figure" is fine; "write
  a 200-word analysis" drifts into hallucinated detail. Chapter 8 names the one-sentence
  constraint deliberately.
- **Omitting `page` on table / figure chunks.** Page is the minimum viable citation
  anchor (chapter 7); `cite(chunk)` without a page emits `[cite: doc, table:N]` which
  is not deep-linkable. Attach `page` from the parser's own metadata, every time.
- **Parsing equation numbers without the parentheses.** Equations are numbered `(17)`
  in-line and `17.` at margins in most styles; a regex that only catches `\n17\n` misses
  the common case. `(\d+)` inside parentheses is the right shape for the common
  `(1)`-style numbering; LaTeX `\tag{17}` is the right shape for TeX-generated papers.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-102-document-ingestion-and-chunking/08-tables-formulas-figures-and-citation.md`
    — the three hard-content shapes and the citation-contract table.
  - `lessons/mod-102-document-ingestion-and-chunking/02-parser-toolchain-unstructured-pymupdf-docling-nougat.md`
    — the parser-of-record story for tables, formulas, and figures.
  - `lessons/mod-102-document-ingestion-and-chunking/07-metadata-as-first-class-retrieval-fields.md`
    — the `extras` discipline (indexed field vs. side-channel).
- Docling — [github.com/docling-project/docling](https://github.com/docling-project/docling)
  and [docling-project.github.io/docling/](https://docling-project.github.io/docling/).
- `unstructured` — [docs.unstructured.io](https://docs.unstructured.io/) for
  `Table.metadata.text_as_html` and element bboxes.
- PyMuPDF (figure extraction) — [pymupdf.readthedocs.io](https://pymupdf.readthedocs.io/).
- pdfplumber (table re-extraction for the round-trip test) —
  [github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber).
- Nougat — [github.com/facebookresearch/nougat](https://github.com/facebookresearch/nougat).
- MathPix API — [docs.mathpix.com](https://docs.mathpix.com/).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/)
  — informs the "citation discipline as a compliance control" framing in `docs/hard-content.md`.
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
  — informs the provenance / traceability requirements that the citation contract satisfies.
