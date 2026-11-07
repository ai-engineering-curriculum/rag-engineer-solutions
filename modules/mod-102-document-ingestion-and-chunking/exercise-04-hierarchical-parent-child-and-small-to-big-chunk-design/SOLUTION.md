# Exercise 04 — Solution: Hierarchical, parent-child, and small-to-big chunk design

This is a reference solution for
`exercise-04-hierarchical-parent-child-and-small-to-big-chunk-design.md`. It is an
**implementation + measurement** exercise building on exercise-03's winning chunker. The
learner picks one of three hierarchical patterns, implements it end-to-end, instruments it
with three counters, A/Bs against the exercise-03 baseline on the same 30+ query judgment
set, and writes a decision. The reference walks the **parent-child (small-to-big)** path
in detail — the other two options are sketched with pointers to the upstream docs.

Numbers in this reference are **illustrative placeholders**. The whole point of the
exercise is that the learner's corpus answers "does hierarchical pay here?" differently
from another corpus. Do not copy these numbers as evidence.

## 1. Solution overview

The reference implements **parent-child** with LlamaIndex: parents of ~2,000 tokens, children
of ~400 tokens, embed children, retrieve on children, resolve to parents via `parent_id`,
dedup, return parent text. Three instrumentation hooks (`n_children_hit`,
`n_unique_parents_returned`, `children_per_parent_max`) are logged per query to a
`docs/hierarchical-instrumentation.jsonl` file so the mean/max can be computed for the A/B
section. The baseline is exercise-03's winner (`structural-markdown` in the worked example);
both retrievers are evaluated on the same qrels with `ir_measures`. The deliverable closes
with a decision, not a hedge.

Options 2 (auto-merging) and 3 (sentence-window) are strictly valid for this exercise. The
rubric scores the *shape* (instrumentation + A/B + decision) rather than the specific
pattern.

## 2. Worked answer or implementation

### 2.1 `docs/hierarchical-choice.md`

```markdown
# Hierarchical pattern choice

We chose **parent-child (small-to-big)** because our judgment set has a mix of fact-lookup
queries (where small children win on precision) and policy-summary queries (where the
parent's surrounding paragraphs are what the user actually wanted). Parent-child lets us
index on precision and return on coverage, and we preferred it over auto-merging because
our corpus's section structure from exercise-03 gave us a natural two-level cut without
needing the three-level `HierarchicalNodeParser` hierarchy.
```

### 2.2 `src/retrievers/hierarchical.py`

```python
# src/retrievers/hierarchical.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter

from ingest.schema import Chunk
from ingest._ids import stable_chunk_id, content_hash, now_utc
from ingest._tokens import count_tokens

_PARENT_CHUNKER = "sentence-splitter-2000-200"
_CHILD_CHUNKER  = "sentence-splitter-400-40"
_CHUNKER_VERSION = "parent-child@1.0.0"

@dataclass
class ParentChildBundle:
    parents: list[Chunk]
    children: list[Chunk]  # child.parent_id -> parent.chunk_id

def build_parent_child(passages: Iterable) -> ParentChildBundle:
    """Split each document twice: parents (~2k tokens), children (~400 tokens).

    `passages` are the normalized Passage records from exercise-01 ingest.
    """
    parent_splitter = SentenceSplitter(chunk_size=2000, chunk_overlap=200,
                                       tokenizer=count_tokens)
    child_splitter  = SentenceSplitter(chunk_size=400,  chunk_overlap=40,
                                       tokenizer=count_tokens)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    parent_ordinal = 0

    for p in passages:
        doc = Document(text=p.text, metadata={
            "doc_id": p.doc_id, "source_uri": p.source_uri,
            "section_path": p.section_path, "page": p.page,
        })
        parent_nodes = parent_splitter.get_nodes_from_documents([doc])
        for parent_node in parent_nodes:
            parent_cid = stable_chunk_id(p.doc_id, f"{_CHUNKER_VERSION}:parent", parent_ordinal)
            parent_text = parent_node.get_content()
            parents.append(Chunk(
                chunk_id=parent_cid,
                doc_id=p.doc_id, source_uri=p.source_uri, ordinal=parent_ordinal,
                text=parent_text, element_type="parent",
                tokens=count_tokens(parent_text),
                section_path=p.section_path, page=p.page,
                content_hash=content_hash(parent_text),
                parser=p.parser, parser_version=p.parser_version,
                chunker=_PARENT_CHUNKER, chunker_version=_CHUNKER_VERSION,
                ingested_at=now_utc(),
            ))

            # Re-split this parent into children.
            child_nodes = child_splitter.get_nodes_from_documents(
                [Document(text=parent_text, metadata=doc.metadata)]
            )
            for i, child_node in enumerate(child_nodes):
                child_cid = stable_chunk_id(
                    p.doc_id, f"{_CHUNKER_VERSION}:child", parent_ordinal * 1000 + i,
                )
                child_text = child_node.get_content()
                children.append(Chunk(
                    chunk_id=child_cid,
                    doc_id=p.doc_id, source_uri=p.source_uri,
                    ordinal=parent_ordinal * 1000 + i,
                    text=child_text, element_type="child",
                    tokens=count_tokens(child_text),
                    section_path=p.section_path, page=p.page,
                    content_hash=content_hash(child_text),
                    parser=p.parser, parser_version=p.parser_version,
                    chunker=_CHILD_CHUNKER, chunker_version=_CHUNKER_VERSION,
                    ingested_at=now_utc(),
                    parent_id=parent_cid,
                ))
            parent_ordinal += 1
    return ParentChildBundle(parents=parents, children=children)
```

### 2.3 Index layout — two pgvector tables

```sql
CREATE TABLE parents (
    chunk_id text PRIMARY KEY, doc_id text NOT NULL, text text NOT NULL,
    section_path text[], page integer, tokens integer NOT NULL
);

CREATE TABLE children (
    chunk_id  text PRIMARY KEY, doc_id text NOT NULL, text text NOT NULL,
    parent_id text NOT NULL REFERENCES parents(chunk_id),
    section_path text[], page integer, tokens integer NOT NULL,
    embedding vector(384) NOT NULL
);
CREATE INDEX children_hnsw
    ON children USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX children_parent_fkey ON children (parent_id);
```

Only the children are embedded. Parents are text-only; they are fetched by `parent_id`
at assembly time.

### 2.4 Retriever — hit children, dedup to parents, cap per-parent concentration

```python
# src/retrievers/hierarchical_retrieve.py
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

MAX_CHILDREN_PER_PARENT = 3   # over-fetch guard; chapter 6's warning
CHILDREN_TOP_K = 30            # pool from which we dedup
PARENTS_TOP_K  = 10

@dataclass
class ParentHit:
    parent_id: str
    text: str
    best_child_score: float
    n_children_hit: int

@dataclass
class QueryInstrumentation:
    qid: str
    n_children_hit: int
    n_unique_parents_returned: int
    children_per_parent_max: int

def retrieve_parents(
    cur, q_vec,
    *,
    children_top_k: int = CHILDREN_TOP_K,
    parents_top_k:  int = PARENTS_TOP_K,
    max_children_per_parent: int = MAX_CHILDREN_PER_PARENT,
) -> tuple[list[ParentHit], QueryInstrumentation]:
    cur.execute("SET hnsw.ef_search = 60;")
    cur.execute(
        "SELECT chunk_id, parent_id, 1 - (embedding <=> %s) AS score "
        "FROM children ORDER BY embedding <=> %s LIMIT %s",
        (q_vec, q_vec, children_top_k),
    )
    child_hits = cur.fetchall()

    # Accumulate best-score + count per parent, respecting the per-parent cap.
    by_parent: dict[str, dict] = {}
    for chunk_id, parent_id, score in child_hits:
        entry = by_parent.setdefault(parent_id, {"best": 0.0, "n": 0})
        entry["n"] += 1
        if score > entry["best"]:
            entry["best"] = score

    children_per_parent_max = max(e["n"] for e in by_parent.values()) if by_parent else 0

    # Cap: a document that fills the children top-k by itself doesn't dominate.
    for pid, entry in list(by_parent.items()):
        if entry["n"] > max_children_per_parent:
            entry["n"] = max_children_per_parent

    ranked = sorted(by_parent.items(), key=lambda kv: kv[1]["best"], reverse=True)
    top_parents = ranked[:parents_top_k]

    # Resolve parent text in one query.
    pids = [pid for pid, _ in top_parents]
    cur.execute("SELECT chunk_id, text FROM parents WHERE chunk_id = ANY(%s)", (pids,))
    text_by_pid = {pid: text for pid, text in cur.fetchall()}

    hits = [
        ParentHit(parent_id=pid, text=text_by_pid[pid],
                  best_child_score=entry["best"], n_children_hit=entry["n"])
        for pid, entry in top_parents
    ]
    inst = QueryInstrumentation(
        qid="",
        n_children_hit=len(child_hits),
        n_unique_parents_returned=len(top_parents),
        children_per_parent_max=children_per_parent_max,
    )
    return hits, inst
```

### 2.5 A/B harness

```python
# scripts/hierarchical_ab.py
from __future__ import annotations
import csv, json, pathlib

import ir_measures
from ir_measures import Recall, nDCG, RR
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from retrievers.hierarchical_retrieve import retrieve_parents

PG_DSN = "postgresql://rag:rag@localhost:5432/rag"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BASELINE_TABLE = "chunks_structural_markdown"  # exercise-03 winner
TOP_K = 10

def _load_judgments():
    queries = {qid: text for qid, text in (
        row for row in csv.reader(open("data/judgments/queries.tsv"), delimiter="\t") if row)}
    qrels = [
        ir_measures.Qrel(query_id=qid, doc_id=doc_id, relevance=int(rel))
        for qid, _, doc_id, rel in (
            row for row in csv.reader(open("data/judgments/qrels.tsv"), delimiter="\t") if row)
    ]
    return queries, qrels

def _run_baseline(cur, q_vec):
    cur.execute("SET hnsw.ef_search = 40;")
    cur.execute(
        f"SELECT doc_id, 1 - (embedding <=> %s) AS score "
        f"FROM {BASELINE_TABLE} ORDER BY embedding <=> %s LIMIT %s",
        (q_vec, q_vec, TOP_K * 3),
    )
    # Dedup by doc_id so Recall@10 is document-level.
    seen, out = set(), []
    for doc_id, score in cur.fetchall():
        if doc_id in seen: continue
        seen.add(doc_id); out.append((doc_id, score))
    return out[:TOP_K]

def main():
    queries, qrels = _load_judgments()
    model = SentenceTransformer(EMBED_MODEL)
    q_vecs = model.encode(list(queries.values()), convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
    qid_to_vec = dict(zip(queries.keys(), q_vecs))

    baseline_run, hier_run = [], []
    instrumentation: list[dict] = []
    tokens_baseline, tokens_hier = [], []

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for qid, _ in queries.items():
                q_vec = qid_to_vec[qid]
                # baseline
                base_hits = _run_baseline(cur, q_vec)
                for doc_id, score in base_hits:
                    baseline_run.append(ir_measures.ScoredDoc(
                        query_id=qid, doc_id=doc_id, score=float(score)))
                # we approximate tokens-returned per query from the top-k rows
                tokens_baseline.append(sum(400 for _ in base_hits))  # or JOIN to get real tokens

                # hierarchical
                hits, inst = retrieve_parents(cur, q_vec)
                inst.qid = qid
                instrumentation.append(inst.__dict__)
                for h in hits:
                    # Resolve parent_id -> doc_id for Recall@10 on the same qrels.
                    cur.execute("SELECT doc_id FROM parents WHERE chunk_id=%s", (h.parent_id,))
                    doc_id = cur.fetchone()[0]
                    hier_run.append(ir_measures.ScoredDoc(
                        query_id=qid, doc_id=doc_id, score=float(h.best_child_score)))
                tokens_hier.append(sum(len(h.text.split()) for h in hits))

    def _metrics(run):
        return ir_measures.calc_aggregate([Recall @ 10, nDCG @ 10, RR @ 10], qrels, run)
    b = _metrics(baseline_run); h = _metrics(hier_run)

    with open("docs/hierarchical-instrumentation.jsonl", "w") as f:
        for row in instrumentation:
            f.write(json.dumps(row) + "\n")

    rows = [
        ("Recall@10",                  b[Recall @ 10], h[Recall @ 10]),
        ("nDCG@10",                    b[nDCG @ 10],   h[nDCG @ 10]),
        ("MRR@10",                     b[RR @ 10],     h[RR @ 10]),
        ("avg tokens returned / query",
         sum(tokens_baseline) / len(tokens_baseline),
         sum(tokens_hier) / len(tokens_hier)),
        ("avg n_children_hit / query",
         "n/a",
         sum(i["n_children_hit"] for i in instrumentation) / len(instrumentation)),
        ("max children_per_parent",
         "n/a",
         max(i["children_per_parent_max"] for i in instrumentation)),
    ]
    print("| Metric | baseline | hierarchical | Δ |")
    print("|---|---|---|---|")
    for name, b_v, h_v in rows:
        if isinstance(b_v, (int, float)) and isinstance(h_v, (int, float)):
            print(f"| {name} | {b_v:.3f} | {h_v:.3f} | {h_v - b_v:+.3f} |")
        else:
            print(f"| {name} | {b_v} | {h_v} | n/a |")

if __name__ == "__main__":
    main()
```

### 2.6 Test — "hits are children, returned text is parent's"

```python
# tests/test_hierarchical.py
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from retrievers.hierarchical_retrieve import retrieve_parents

def test_returned_text_is_parent_not_child():
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    q_vec = model.encode(["sample policy question"],
                         convert_to_numpy=True, normalize_embeddings=True)[0]
    with psycopg.connect("postgresql://rag:rag@localhost:5432/rag",
                         autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            hits, inst = retrieve_parents(cur, q_vec)
            assert hits, "expected at least one parent hit"
            # The returned text should be a parent, not any child:
            for h in hits:
                cur.execute("SELECT text FROM parents WHERE chunk_id=%s", (h.parent_id,))
                (parent_text,) = cur.fetchone()
                assert h.text == parent_text
                # Instrumentation counters are populated.
                assert inst.n_children_hit > 0
                assert inst.n_unique_parents_returned > 0
```

### 2.7 `docs/hierarchical-ab.md` — filled template

```markdown
# A/B — parent-child hierarchical vs. structural-markdown (exercise-03 winner)

Corpus: mod-102-handbook. Judgment set: 30 queries (same as exercise-03).
Held constant: embedding model `BAAI/bge-small-en-v1.5`, HNSW `(M=16, ef_construction=64)`,
`ef_search=60` for children (vs. 40 for baseline — justified: wider over-fetch pool because
children dedup to parents), `top_k=10` in both cases.

| Metric | exercise-03 winner | hierarchical variant | Δ |
|---|---|---|---|
| Recall@10 | <learner> | <learner> | <+/−> |
| nDCG@10   | <learner> | <learner> | <+/−> |
| MRR@10    | <learner> | <learner> | <+/−> |
| avg. tokens returned per query | <learner> | <learner> | <+/−> |
| avg. n_chunks (parents) returned per query | 10 | <≤10 after dedup> | <+/−> |
| index size (MB) | <baseline> | <hier (children only)> | <+/−> |
| ingest wall-clock (s) | <baseline> | <hier> | <+/−> |

## Per-query analysis

**Where the hierarchical variant wins by ≥ 0.3 Recall@10.** <List queries by qid.>
These are policy-summary queries whose answer spans 3–4 sentences; the 400-token child
returned a strong score, and the 2k-token parent contained the full surrounding context
the small chunk cut off.

**Where the hierarchical variant loses by ≥ 0.3 Recall@10.** <List queries by qid.>
These are queries where several relevant documents each contributed a child hit, but
the cap `MAX_CHILDREN_PER_PARENT=3` plus the top-30 pool left the fourth relevant doc
off the parents short-list. Raising `CHILDREN_TOP_K` to 60 closed two of these.

**Assembler dedup pressure.** `children_per_parent_max` across the 30 queries:
mean=<n>, max=<n>. The cap at 3 fired on <k> queries — a sign that without the cap,
one over-concentrated document would have crowded out other relevant documents.

## Recommendation

Deploy parent-child for this corpus. The recall@10 gain is +<delta> absolute at a
<cost-multiplier>× ingest cost and <token-multiplier>× tokens-returned. Downstream
cost implication: the generator budget sees ~<n>× more context per query; mod-107
will introduce a compression pass to pay it back.

Alternative — **only-for-certain-queries**: if the recall gain concentrates on
policy-summary queries (len > 10 tokens, verb-heavy), a two-retriever router keyed on
query length / shape may deliver most of the gain at 1× cost. Prototype in mod-110.
```

## 3. Validation steps

1. Dependencies: `uv add llama-index llama-index-embeddings-huggingface` (parent-child
   references the LlamaIndex SentenceSplitter).
2. Baseline table from exercise-03 (`chunks_structural_markdown`) still present and
   indexed; run `SELECT COUNT(*) FROM chunks_structural_markdown;` to confirm.
3. `uv run python -m scripts.ingest_hierarchical` (not shown — a near-copy of
   `scripts.ingest_variant` that builds `parents` + `children`): verify
   `SELECT COUNT(*) FROM parents; SELECT COUNT(*) FROM children;` both > 0 and
   `SELECT COUNT(DISTINCT parent_id) FROM children;` equals `COUNT(*) FROM parents`.
4. `uv run pytest -q tests/test_hierarchical.py` passes.
5. `uv run python -m scripts.hierarchical_ab > docs/hierarchical-ab-row.md` — the
   table has every row filled in; `docs/hierarchical-instrumentation.jsonl` has one
   JSON line per query with the three counters.
6. Open `docs/hierarchical-ab.md` — the per-query analysis cites qids; the
   recommendation makes a decision (deploy / don't deploy / route conditionally).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Pattern choice | `docs/hierarchical-choice.md` names one of {parent-child, auto-merging, sentence-window} with ≥ 2 sentences of why. |
| Implementation | Uses an upstream pattern (LlamaIndex `ParentDocumentRetriever` / `AutoMergingRetriever` / `SentenceWindowNodeParser`, or LangChain equivalent) — not a hand-rolled sketch. |
| Instrumentation | All three counters (`n_children_hit`, `n_unique_parents_returned`, `children_per_parent_max`) populated per query and persisted to a file. |
| A/B | Same 30+ qrels + qids as exercise-03; same embedding model; same dedup rule. |
| Required columns | Metric table has Recall@10, nDCG@10, MRR@10, avg tokens returned per query, avg parents returned per query, index size (MB), ingest wall-clock (s). |
| Dedup math | For parent-child: parents returned ≤ children top-k / avg children_per_parent (sanity check on the dedup logic). |
| Per-query | ≥ 5 queries cited per "wins" and "loses" direction, each with a one-sentence pattern identification. |
| Recommendation | Decision stated (deploy / don't / route conditionally); if "route conditionally", the routing condition is named. |
| Token-budget note | The recommendation addresses the "hierarchical returns more tokens per query" downstream cost. |
| Test | `test_returned_text_is_parent_not_child` (or the equivalent for your pattern) passes. |

Fail-fast smells:

- `children_per_parent_max` is always 1. Either the cap is being applied before the
  counter (bug), or the child pool is too small to observe over-concentration (raise
  `CHILDREN_TOP_K`).
- No parent-dedup — each query returns 10 child hits that all point at the same parent.
  Dedup by `parent_id` *before* taking top-10.
- The baseline retriever uses `ef_search=40` and the hierarchical retriever uses
  `ef_search=200` with no justification — a quality-knob difference is a different
  experiment.
- `avg tokens returned per query` column empty. Hierarchical's main *cost* is that it
  returns more tokens; the recommendation must weigh the recall gain against that.
- Recommendation reads "it depends." The exercise explicitly asks for a decision.

## 5. Common mistakes

- **"Hierarchical always wins."** It routinely *loses* on fact-lookup queries where a
  tight 400-token child already has everything the answer needs. If the entire judgment
  set is fact-lookup, the baseline may be the right deploy choice; the per-query
  analysis is where you see this.
- **Not capping children-per-parent.** One document whose boilerplate matches 15 child
  chunks consumes the top-30 pool by itself; the other relevant documents never surface.
  `MAX_CHILDREN_PER_PARENT=3` is a starting point; tune from the instrumentation.
- **Embedding parents.** Only children need embeddings. Parents are fetched by ID. Teams
  that accidentally embed both pay 2–3× the embedding cost for no retrieval benefit.
- **Judging hierarchical recall@10 on child-level qrels.** If your qrels are document-level,
  resolve each child hit to its parent → doc_id, then compute Recall@10 over doc_ids.
  If your qrels are span-level (`section_path`), map parents to their section_path and
  score accordingly — but then the baseline needs the same remap, or the A/B is unfair.
- **Measuring index size on the children table only.** The parents table is overhead
  the baseline doesn't have — count it in the "index size" column (text column size
  plus any FK index).
- **Treating "returns more tokens per query" as free.** Mod-107 is where context
  compression lives; if your recommendation commits to parent-child, the next-iteration
  paragraph should name the compression plan (`LLMChainExtractor`, bge-reranker + truncate,
  sentence-window overlay, etc.).
- **Picking auto-merging without reading chapter 6 carefully.** `AutoMergingRetriever`
  merges leaf hits upward *when a merging threshold is crossed*; if that threshold is
  wrong for your corpus you get either no merges (acting like plain leaf retrieval) or
  all merges (acting like plain parent retrieval). Tune on a small sample before scoring.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-102-document-ingestion-and-chunking/06-hierarchical-parent-child-and-small-to-big.md`
    — the three patterns and their trade-offs.
  - `lessons/mod-102-document-ingestion-and-chunking/07-metadata-as-first-class-retrieval-fields.md`
    — the `parent_id` / `children_ids` / `window_ids` fields in the `Chunk` schema.
- LlamaIndex `ParentDocumentRetriever` + `SentenceSplitter` —
  [docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/).
- LlamaIndex `HierarchicalNodeParser` + `AutoMergingRetriever` —
  [docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/](https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/).
- LlamaIndex `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` —
  [docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/).
- LangChain `ParentDocumentRetriever` —
  [python.langchain.com/docs/how_to/parent_document_retriever/](https://python.langchain.com/docs/how_to/parent_document_retriever/).
- `ir_measures` — [ir-measur.es/en/latest/getting-started.html](https://ir-measur.es/en/latest/getting-started.html).
- pgvector HNSW — [github.com/pgvector/pgvector#hnsw](https://github.com/pgvector/pgvector#hnsw).
