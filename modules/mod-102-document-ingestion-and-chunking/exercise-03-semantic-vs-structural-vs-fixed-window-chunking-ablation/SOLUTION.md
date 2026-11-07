# Exercise 03 — Solution: Semantic vs. structural vs. fixed-window chunking ablation

This is a reference solution for
`exercise-03-semantic-vs-structural-vs-fixed-window-chunking-ablation.md`. It is an
**implementation + measurement** exercise. The deliverable is five chunker variants, five
pgvector tables, a judgment set of 30+ queries in TREC format, a `scripts/eval_chunkers.py`
harness that produces `Recall@10 / nDCG@10 / MRR@10` per variant with `ir_measures`, and a
`docs/chunker-ablation.md` with per-query error analysis and a one-variant recommendation.

The measured numbers in the reference are **illustrative placeholders**. "Winner" always
depends on your corpus and judgment set. The correctness of the exercise is in the method,
not in matching a specific number. Do not copy the numbers below as evidence.

## 1. Solution overview

The ablation holds three axes constant (embedding model, retriever, corpus) and varies one
(chunker). The reference variants:

1. `fixed-800-100` — `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`.
2. `fixed-400-40` — same splitter, smaller chunk size. Isolates chunk-size effect.
3. `structural-markdown` — `MarkdownHeaderTextSplitter` + recursive split at 800 tokens.
4. `semantic-95` — LlamaIndex `SemanticSplitterNodeParser(breakpoint_percentile_threshold=95)`
   with `BAAI/bge-small-en-v1.5` as the splitter's embed model (same family as the retrieval
   embedder; this is intentional).
5. `title-structural` — `unstructured.chunk_by_title(max_characters=1200, new_after_n_chars=800)`
   on the exercise-01 `unstructured` parser output.

Embedding model is `BAAI/bge-small-en-v1.5` (384-dim, normalized, cosine). Retriever is
pgvector with HNSW (`M=16, efConstruction=64, efSearch=40`), `top_k=20`. Judgment set is the
learner's own 30+ queries over the mod-102 corpus, labeled either coarse (doc-level) or fine
(span-level).

The output is one table in `docs/chunker-ablation.md`, two paragraphs of per-query error
analysis, and a one-paragraph `## Recommendation` that names a single winner plus a
`chunker.yaml` block that could drop straight into a design doc.

## 2. Worked answer or implementation

### 2.1 Five chunker variants

```python
# src/chunkers/__init__.py
from .fixed      import chunk_fixed_800_100, chunk_fixed_400_40
from .structural import chunk_structural_markdown, chunk_title_structural
from .semantic   import chunk_semantic_95

VARIANTS = {
    "fixed-800-100":       (chunk_fixed_800_100,       "chunks_fixed_800_100"),
    "fixed-400-40":        (chunk_fixed_400_40,        "chunks_fixed_400_40"),
    "structural-markdown": (chunk_structural_markdown, "chunks_structural_markdown"),
    "semantic-95":         (chunk_semantic_95,         "chunks_semantic_95"),
    "title-structural":    (chunk_title_structural,    "chunks_title_structural"),
}
```

```python
# src/chunkers/fixed.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens

def _recursive(passages, *, chunk_size: int, chunk_overlap: int,
               chunker_version: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=count_tokens,
    )
    out, ordinal = [], 0
    for p in passages:  # p is the normalized paragraph/passage from exercise-01
        for text in splitter.split_text(p.text):
            out.append(Chunk(
                chunk_id=stable_chunk_id(p.doc_id, chunker_version, ordinal),
                doc_id=p.doc_id, source_uri=p.source_uri, ordinal=ordinal,
                text=text, element_type="paragraph",
                tokens=count_tokens(text),
                section_path=p.section_path, page=p.page,
                content_hash=content_hash(text),
                parser=p.parser, parser_version=p.parser_version,
                chunker="RecursiveCharacterTextSplitter",
                chunker_version=chunker_version,
                ingested_at=now_utc(),
            ))
            ordinal += 1
    return out

def chunk_fixed_800_100(passages):
    return _recursive(passages, chunk_size=800, chunk_overlap=100,
                      chunker_version="fixed-800-100@1.0.0")

def chunk_fixed_400_40(passages):
    return _recursive(passages, chunk_size=400, chunk_overlap=40,
                      chunker_version="fixed-400-40@1.0.0")
```

```python
# src/chunkers/structural.py
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

def chunk_structural_markdown(passages):
    """Structural markdown: split at headings, then recurse any section > 800 tokens."""
    recur = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, length_function=count_tokens,
    )
    hdr = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    out, ordinal = [], 0
    for p in passages:
        splits = hdr.split_text(p.text)
        for s in splits:
            section_path = [v for k, v in s.metadata.items() if k in {"h1", "h2", "h3"} and v]
            parts = recur.split_text(s.page_content) if count_tokens(s.page_content) > 800 \
                    else [s.page_content]
            for part in parts:
                out.append(Chunk(
                    chunk_id=stable_chunk_id(p.doc_id, "structural-md@1.0.0", ordinal),
                    doc_id=p.doc_id, source_uri=p.source_uri, ordinal=ordinal,
                    text=part, element_type="paragraph",
                    tokens=count_tokens(part),
                    section_path=section_path, page=p.page,
                    content_hash=content_hash(part),
                    parser=p.parser, parser_version=p.parser_version,
                    chunker="MarkdownHeaderTextSplitter",
                    chunker_version="structural-md@1.0.0",
                    ingested_at=now_utc(),
                ))
                ordinal += 1
    return out

def chunk_title_structural(pdf_paths: list[str]):
    """unstructured.chunk_by_title at the exercise-01 shape, size-capped."""
    out, ordinal = [], 0
    for pdf in pdf_paths:
        elements = partition_pdf(filename=pdf, strategy="hi_res", infer_table_structure=True)
        for c in chunk_by_title(elements, max_characters=1200, new_after_n_chars=800):
            text = c.text or ""
            out.append(Chunk(
                chunk_id=stable_chunk_id(pdf, "title-structural@1.0.0", ordinal),
                doc_id=pdf, source_uri=pdf, ordinal=ordinal,
                text=text, element_type="paragraph",
                tokens=count_tokens(text),
                page=getattr(c.metadata, "page_number", None),
                content_hash=content_hash(text),
                parser="unstructured.partition_pdf", parser_version="",
                chunker="unstructured.chunk_by_title",
                chunker_version="title-structural@1.0.0",
                ingested_at=now_utc(),
            ))
            ordinal += 1
    return out
```

```python
# src/chunkers/semantic.py
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document

from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens

_EMBED = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def chunk_semantic_95(passages):
    parser = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=_EMBED,
    )
    out, ordinal = [], 0
    for p in passages:
        docs = [Document(text=p.text, metadata={"doc_id": p.doc_id})]
        for node in parser.get_nodes_from_documents(docs):
            text = node.get_content()
            out.append(Chunk(
                chunk_id=stable_chunk_id(p.doc_id, "semantic-95@1.0.0", ordinal),
                doc_id=p.doc_id, source_uri=p.source_uri, ordinal=ordinal,
                text=text, element_type="paragraph",
                tokens=count_tokens(text),
                section_path=p.section_path, page=p.page,
                content_hash=content_hash(text),
                parser=p.parser, parser_version=p.parser_version,
                chunker="SemanticSplitterNodeParser",
                chunker_version="semantic-95@1.0.0",
                ingested_at=now_utc(),
            ))
            ordinal += 1
    return out
```

### 2.2 One pgvector table per variant

Schema (ported from the mod-101 toolchain, with one fresh table per variant so the ablation
is a clean reindex):

```sql
-- One table per variant; swap the name.
CREATE TABLE chunks_fixed_800_100 (
    chunk_id       text PRIMARY KEY,
    doc_id         text NOT NULL,
    source_uri     text NOT NULL,
    ordinal        integer NOT NULL,
    text           text NOT NULL,
    element_type   text NOT NULL,
    tokens         integer NOT NULL,
    section_path   text[],
    page           integer,
    content_hash   text NOT NULL,
    parser         text NOT NULL,
    chunker        text NOT NULL,
    chunker_version text NOT NULL,
    embedding      vector(384) NOT NULL
);
CREATE INDEX chunks_fixed_800_100_hnsw
    ON chunks_fixed_800_100 USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Ingest helper (same for every variant; the only thing that varies is the chunker fn + table):

```python
# scripts/ingest_variant.py
from __future__ import annotations
import time, pathlib
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from chunkers import VARIANTS

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PG_DSN = "postgresql://rag:rag@localhost:5432/rag"

def ingest(variant: str, passages):
    fn, table = VARIANTS[variant]
    model = SentenceTransformer(EMBED_MODEL)
    t0 = time.perf_counter()
    chunks = fn(passages)
    vecs = model.encode([c.text for c in chunks], convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table};")
            cur.execute(f"""
                CREATE TABLE {table} (
                    chunk_id text PRIMARY KEY, doc_id text NOT NULL,
                    source_uri text NOT NULL, ordinal integer NOT NULL,
                    text text NOT NULL, element_type text NOT NULL,
                    tokens integer NOT NULL, section_path text[],
                    page integer, content_hash text NOT NULL,
                    parser text NOT NULL, chunker text NOT NULL,
                    chunker_version text NOT NULL, embedding vector(384) NOT NULL
                );
            """)
            for c, v in zip(chunks, vecs):
                cur.execute(
                    f"INSERT INTO {table} (chunk_id, doc_id, source_uri, ordinal, text, "
                    f"element_type, tokens, section_path, page, content_hash, parser, "
                    f"chunker, chunker_version, embedding) VALUES "
                    f"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (c.chunk_id, c.doc_id, c.source_uri, c.ordinal, c.text,
                     c.element_type, c.tokens, c.section_path, c.page, c.content_hash,
                     c.parser, c.chunker, c.chunker_version, v),
                )
            cur.execute(
                f"CREATE INDEX {table}_hnsw ON {table} USING hnsw "
                f"(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
            )
    wall = time.perf_counter() - t0
    return {"variant": variant, "n_chunks": len(chunks), "wall_s": wall,
            "mean_tokens": float(np.mean([c.tokens for c in chunks])),
            "p95_tokens": float(np.percentile([c.tokens for c in chunks], 95))}
```

### 2.3 Judgment set format

```
# data/judgments/queries.tsv  (tab-separated, no header)
q01     what are the PTO accrual rates for new hires
q02     can I roll over unused parental leave to next year
q03     who approves a mid-cycle salary adjustment
...     (at least 30)

# data/judgments/qrels.tsv  (TREC: qid 0 doc_id rel, tab-separated, no header)
q01     0       handbook#benefits/pto    1
q02     0       handbook#benefits/parental-leave   1
q02     0       handbook#benefits/parental-leave/rollover  1
...
```

If you label coarsely (doc-level), `doc_id` is the full document ID. If you label finely
(span-level), the third column is a span ID; chunks whose `section_path` matches the span's
section path are counted as relevant.

### 2.4 `scripts/eval_chunkers.py`

```python
# scripts/eval_chunkers.py
from __future__ import annotations
import csv
import pathlib

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import ir_measures
from ir_measures import Recall, nDCG, RR

from chunkers import VARIANTS

PG_DSN = "postgresql://rag:rag@localhost:5432/rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EF_SEARCH = 40
TOP_K = 20

def _load_tsv(path: pathlib.Path):
    with open(path) as f:
        return [tuple(row) for row in csv.reader(f, delimiter="\t") if row]

def _coarse_label(hit_doc_id: str) -> str:
    return hit_doc_id  # qrels are doc-level

def _dedup_by_doc(hits):
    seen, out = set(), []
    for doc_id, score in hits:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append((doc_id, score))
    return out

def _retrieve(cur, table: str, q_vec, top_k: int):
    cur.execute(f"SET hnsw.ef_search = {EF_SEARCH};")
    cur.execute(
        f"SELECT doc_id, 1 - (embedding <=> %s) AS score "
        f"FROM {table} ORDER BY embedding <=> %s LIMIT %s",
        (q_vec, q_vec, top_k * 3),  # over-fetch, then dedup
    )
    return _dedup_by_doc([(d, s) for d, s in cur.fetchall()])[:top_k]

def main():
    queries = {qid: text for qid, text in _load_tsv(pathlib.Path("data/judgments/queries.tsv"))}
    qrels = [
        ir_measures.Qrel(query_id=qid, doc_id=doc_id, relevance=int(rel))
        for qid, _, doc_id, rel in _load_tsv(pathlib.Path("data/judgments/qrels.tsv"))
    ]

    model = SentenceTransformer(EMBED_MODEL)
    q_vecs = model.encode(list(queries.values()), convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
    qid_to_vec = dict(zip(queries.keys(), q_vecs))

    rows: list[dict] = []
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for variant, (_, table) in VARIANTS.items():
                run = []
                for qid, text in queries.items():
                    hits = _retrieve(cur, table, qid_to_vec[qid], TOP_K)
                    for doc_id, score in hits:
                        run.append(ir_measures.ScoredDoc(
                            query_id=qid, doc_id=_coarse_label(doc_id), score=float(score),
                        ))
                metrics = ir_measures.calc_aggregate(
                    [Recall @ 10, nDCG @ 10, RR @ 10], qrels, run,
                )
                rows.append({
                    "variant": variant,
                    "recall_at_10": metrics[Recall @ 10],
                    "ndcg_at_10":   metrics[nDCG @ 10],
                    "mrr_at_10":    metrics[RR @ 10],
                })
                print(f"{variant:<22} recall@10={metrics[Recall @ 10]:.3f}  "
                      f"nDCG@10={metrics[nDCG @ 10]:.3f}  MRR@10={metrics[RR @ 10]:.3f}")

    with open("docs/chunker-ablation-row.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
```

Note on `Recall@10`: when the qrels are coarse (doc-level), deduplicate hits by `doc_id`
before scoring. Otherwise 10 chunks from the same winning document can falsely inflate
`Recall@10`. (`_dedup_by_doc` + `TOP_K * 3` over-fetch handles this.)

### 2.5 `docs/chunker-ablation.md` — filled template

```markdown
# Chunker ablation — corpus: mod-102-handbook, judgment set: 30 queries, 2026-11-07

**Held constant.** Embedding model: `BAAI/bge-small-en-v1.5` (384-dim, normalized, cosine).
Retriever: pgvector with HNSW (`M=16, ef_construction=64, ef_search=40`), `top_k=20`,
`dedup by doc_id`. Corpus: 18 documents / ~210 pages (`data/corpus/`).

| chunker              | n_chunks | mean_tokens | p95_tokens | recall@10 | nDCG@10 | MRR@10 | ingest_s | index_mb |
|----------------------|---------:|------------:|-----------:|----------:|--------:|-------:|---------:|---------:|
| fixed-800-100        |      450 |         620 |        790 | <learner> |<learner>|<learner>| <learner>|<learner> |
| fixed-400-40         |      890 |         340 |        395 | <learner> |<learner>|<learner>| <learner>|<learner> |
| structural-markdown  |      380 |         580 |      1,120 | <learner> |<learner>|<learner>| <learner>|<learner> |
| semantic-95          |      410 |         540 |        880 | <learner> |<learner>|<learner>| <learner>|<learner> |
| title-structural     |      360 |         810 |      1,180 | <learner> |<learner>|<learner>| <learner>|<learner> |

## Per-query analysis

**Where `structural-markdown` wins by ≥ 0.5 recall@10.** <List five queries.> These are
queries whose answer sits inside a single section whose heading the user partially
quotes ("PTO accrual rates for new hires" → `Benefits → PTO → Accrual`). The structural
chunk tightly contains both the heading and the answer; the fixed-window variants cut
across the heading boundary and lose the cue.

**Where `fixed-400-40` wins by ≥ 0.5 recall@10.** <List five queries.> These are
queries looking for a single concrete fact ("maximum contribution for 401k in 2026") —
a small chunk centered on the fact gets a stronger score than a paragraph-length
structural chunk that also contains several unrelated sentences.

**Queries where all variants scored 0.** <List any.> These are judgment bugs. The
right `doc_id` is missing from qrels; fix the qrels and re-run, not the chunker.

## Recommendation

For this corpus, deploy `structural-markdown`. It wins on nDCG@10 by <delta> at <nearly
the same / lower / higher> ingest cost than `semantic-95`, and loses to `fixed-400-40` only
on fact-lookup queries where a hybrid lexical leg (mod-106) will likely close the gap.
Next iteration: add a BM25 leg and re-run; and try `structural-markdown` with the first
sentence of each section repeated into any recurse-split child (a form of small-to-big
that mod-102 exercise-04 explores).

```yaml
chunker: langchain_text_splitters.MarkdownHeaderTextSplitter+RecursiveCharacterTextSplitter
version: structural-md@1.0.0
primary_boundary: [h1, h2, h3]
recurse_above_tokens: 800
recurse_overlap_tokens: 100
notes: |
  Winner of exercise-03 ablation on 2026-11-07, n=30 queries on mod-102 corpus.
  recall@10=<v> (vs. <v> fixed-800-100, <v> semantic-95). Ingest cost ~<t>x cheaper
  than semantic. Known weakness: fact-lookup queries — plan hybrid retrieval in mod-106.
```
```

## 3. Validation steps

1. Dependencies: `uv add ir_measures sentence-transformers llama-index
   llama-index-embeddings-huggingface langchain-text-splitters unstructured`.
2. pgvector service from mod-101's compose file running: `pg_isready -U rag -d rag`
   returns success; `SELECT * FROM pg_extension WHERE extname='vector';` returns a row.
3. `data/judgments/queries.tsv` has ≥ 30 rows; `qrels.tsv` is in TREC format (4 columns:
   `qid 0 doc_id rel`), tab-separated.
4. `uv run python -m scripts.ingest_variant --variant fixed-800-100` (repeat for all
   five variants) — each prints `n_chunks`, `mean_tokens`, `p95_tokens`, `wall_s`. Rows
   accumulate into `docs/ingest-stats.md`.
5. `uv run python -m scripts.eval_chunkers` prints one line per variant with
   `recall@10`, `nDCG@10`, `MRR@10`, and writes `docs/chunker-ablation-row.csv`.
6. Open `docs/chunker-ablation.md`: the table has five rows all populated; the
   recommendation names one variant; a `chunker.yaml` block follows the format in
   chapter 5.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Five variants | All five functions in `src/chunkers/` produce non-empty `list[Chunk]` with valid schema. |
| Isolation | Embedding model, retriever parameters, and `top_k` are documented at the top of `chunker-ablation.md` and identical across variants. |
| Judgment set | ≥ 30 queries, TREC-format qrels, spans at least two granularity bands (fact-lookup + section-summary). |
| Harness | `scripts/eval_chunkers.py` uses `ir_measures.calc_aggregate` with `Recall@10`, `nDCG@10`, `RR@10`; dedups hits by `doc_id` before scoring when qrels are coarse. |
| Table | Nine columns populated per variant (n_chunks, mean_tokens, p95_tokens, recall@10, nDCG@10, MRR@10, ingest_s, index_mb — plus variant name). |
| Analysis | ≥ 5 queries per winning pattern are cited by `qid` with a one-sentence explanation; all-zero queries diagnosed as judgment bugs or genuine misses. |
| Recommendation | One variant named; a `chunker.yaml` block in the chapter-5 shape; one "next iteration" paragraph. |
| Discipline | No invented metrics — every number in the table came from a run; the per-query delta threshold is explicit (0.5 in the template). |

Fail-fast smells:

- A "winner" paragraph that names two variants ("go with A or B"). The exercise asks
  for a decision. If the numbers are genuinely tied, the tie-breaker is the ingest-cost
  column.
- A `semantic-95` chunker whose splitter embed model is `text-embedding-3-small`
  while the retriever embedder is `bge-small-en-v1.5`. This mixes two embedding spaces
  and makes the "same-family splitter" argument of chapter 4 silently wrong.
- `top_k=5` in the harness, `top_k=20` in the ingest doc. Keep *every* retriever
  parameter identical across variants — the ablation only means something if the
  chunker is the one thing that moved.
- `Recall@10` computed on 10 chunks all from the same relevant document (`doc_id`).
  Dedup first; the TREC convention is that `Recall@k` answers "did I find a relevant
  *document* in my top-k", not "did I find the relevant *chunk* k times."
- A per-query analysis section that reads "fixed-800-100 wins sometimes, loses sometimes."
  Every bullet should name a pattern (fact-lookup, policy-summary, acronym-heavy).

## 5. Common mistakes

- **Varying two axes at once.** Running `semantic-95` with `BAAI/bge-small-en-v1.5` as the
  retriever and `text-embedding-3-small` as the splitter, or `fixed-400-40` with
  `top_k=20` and `semantic-95` with `top_k=50`. You cannot attribute any delta.
- **Scoring chunks when the qrels are documents.** TREC qrels are document-level unless
  you explicitly build span-level IDs. Deduplicate retrieval hits by `doc_id` and score
  the dedup'd list.
- **Picking only fact-lookup queries for the judgment set.** The dense/structural/semantic
  families have different strengths; a judgment set that has only one query shape will
  pick a winner that generalizes poorly.
- **Not re-indexing between variants.** A table that still contains chunks from
  `fixed-400-40` when you intended to measure `fixed-800-100` silently scores the wrong
  thing. The ingest helper explicitly `DROP TABLE IF EXISTS` on every ingest.
- **Reporting a single CER / nDCG mean without the distribution.** Two variants with
  the same mean nDCG@10 but very different variance are not interchangeable. Include
  min/max or a boxplot in a stretch write-up.
- **Using `ef_search=100` for one variant's retrieval run and `ef_search=40` for
  another.** `ef_search` is a quality knob with a latency cost; it must be held
  constant across variants just like `top_k`.
- **Writing the recommendation before seeing the per-query analysis.** The aggregate
  table is the headline; the per-query delta is where chapter 4's strategy-family
  claims become concrete. If the aggregate winner has three obvious failure patterns,
  the recommendation should name them, not paper over them.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-102-document-ingestion-and-chunking/04-chunking-strategies-fixed-structural-semantic.md`
    — the five-strategy taxonomy.
  - `lessons/mod-102-document-ingestion-and-chunking/05-chunk-size-recall-precision-tradeoff.md`
    — the ablation discipline and `chunker.yaml` shape.
  - `lessons/mod-102-document-ingestion-and-chunking/07-metadata-as-first-class-retrieval-fields.md`
    — the `Chunk` schema across every variant.
- `ir_measures` — [ir-measur.es/en/latest/getting-started.html](https://ir-measur.es/en/latest/getting-started.html)
  (TREC qrel conventions and `calc_aggregate` API).
- LlamaIndex `SemanticSplitterNodeParser` —
  [docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#semanticsplitternodeparser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#semanticsplitternodeparser).
- LangChain `MarkdownHeaderTextSplitter` —
  [python.langchain.com/docs/how_to/markdown_header_metadata_splitter/](https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/).
- LangChain `RecursiveCharacterTextSplitter` —
  [python.langchain.com/docs/how_to/recursive_text_splitter/](https://python.langchain.com/docs/how_to/recursive_text_splitter/).
- `unstructured.chunking.title.chunk_by_title` —
  [docs.unstructured.io/open-source/core-functionality/chunking](https://docs.unstructured.io/open-source/core-functionality/chunking).
- pgvector — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector),
  [HNSW index docs](https://github.com/pgvector/pgvector#hnsw).
- sentence-transformers / `BAAI/bge-small-en-v1.5` — [sbert.net](https://www.sbert.net/)
  and [huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
