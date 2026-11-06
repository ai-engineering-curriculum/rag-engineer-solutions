# Exercise 05 — Solution: Bootstrap the local toolchain and pass five smoke tests

This is a reference solution for `exercise-05-local-toolchain-bootstrap-and-smoke-test.md`.
Deliverables A–D are worked below with runnable artifacts (`pyproject.toml`, `docker-compose.yml`,
`tests/test_smoke.py`, `TOOLCHAIN.md`). Every artifact is a reference shape — version pins move;
always confirm current supported versions against the primary docs linked in the References
section rather than hard-coding the numbers below.

## 1. Solution overview

The exercise's outcome is a green `pytest -q` against a five-test smoke suite: imports, embedding
round-trip, pgvector round-trip, BM25 round-trip (Elasticsearch or OpenSearch), and `ir_measures`
round-trip. The reference solution:

- Pins the dependencies from chapter 6 in a `pyproject.toml` the learner can copy and adapt.
- Provides a `docker-compose.yml` for Postgres (with pgvector) and Elasticsearch with
  **local-only, security-disabled** settings — the chapter's warning that these flags do not
  travel to any deployment is reproduced in the TOOLCHAIN file.
- Provides the full `tests/test_smoke.py` implementing all five tests exactly to the exercise's
  acceptance criteria.
- Provides a `TOOLCHAIN.md` that records Python version, OS/hardware, container tags, and the
  model revision, so a colleague on a similar machine can reproduce the setup.

The reference uses the stack the chapter recommends: `uv` as the project manager,
Elasticsearch 8.x as the Lucene-family engine (OpenSearch is noted as a drop-in with the
documented client swap), and `BAAI/bge-small-en-v1.5` as the embedding model.

## 2. Worked answer or implementation

### Deliverable A — `pyproject.toml`

```toml
[project]
name = "rag-engineer-lab"
version = "0.1.0"
description = "mod-101 toolchain bootstrap"
requires-python = ">=3.11"

dependencies = [
  "llama-index>=0.11",         # orchestration reference; mod-102+ depend on it
  "sentence-transformers>=3.0", # bi-encoder + cross-encoder interface
  "torch>=2.2",                # backs sentence-transformers / HF
  "psycopg[binary]>=3.2",      # Postgres client
  "pgvector>=0.3",             # pgvector-python bindings
  "elasticsearch>=8.14,<9",    # pick one and stick with it; see docker-compose.yml
  "ir-measures>=0.3",          # retrieval-only evaluation
  "pytest>=8.0",               # for the smoke tests
]
```

Lock with `uv lock` (or `uv sync`) and commit the resulting `uv.lock`. The exact resolved versions
should be captured by the lockfile, not by this file. Learners using Poetry run `poetry lock`
instead; the dependency set is identical.

### Deliverable B — `docker-compose.yml`

```yaml
# docker-compose.yml — local-only development stack for mod-101 toolchain.
# Security is intentionally disabled on both services for local work. Do NOT
# carry `xpack.security.enabled=false` or `plugins.security.disabled=true`
# into any environment that is reachable by anything other than localhost.
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag"]
      interval: 5s
      timeout: 5s
      retries: 10

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.14.3
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:9200 >/dev/null"]
      interval: 5s
      timeout: 5s
      retries: 20
```

OpenSearch substitution (choose one and stay with it through mod-105; the exercise is explicit
about not mixing):

```yaml
# Swap this block in place of the `elasticsearch` service to use OpenSearch.
  opensearch:
    image: opensearchproject/opensearch:2.15.0
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
      - OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
```

Connectivity checks that must both succeed before running `pytest`:

```bash
# Postgres
psql -h localhost -U rag -d rag -c "SELECT 1;"
# → returns 1

# Elasticsearch (or OpenSearch — same URL)
curl -s http://localhost:9200 | head -20
# → returns engine welcome JSON (cluster name, version, tagline)
```

### Deliverable C — `tests/test_smoke.py`

```python
"""mod-101 toolchain smoke tests.

Every test fails loudly if the stack is not correctly wired. Requires:
- Local Postgres with pgvector on localhost:5432 (user=rag, password=rag, db=rag)
- Local Elasticsearch on localhost:9200 (single-node, security disabled) — or
  OpenSearch with the client import swap documented in test_bm25_round_trip.
- `uv sync` (or equivalent) has installed everything in pyproject.toml.
"""
from __future__ import annotations

import os
import time

import pytest

PG_DSN = os.environ.get(
    "PG_DSN", "postgresql://rag:rag@localhost:5432/rag"
)
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, CPU-friendly, permissively licensed


# ---------- Smoke test 1 — imports ----------

def test_imports():
    import llama_index               # noqa: F401
    import sentence_transformers     # noqa: F401
    import torch                     # noqa: F401
    import psycopg                   # noqa: F401
    import pgvector                  # noqa: F401
    import elasticsearch             # noqa: F401
    import ir_measures               # noqa: F401


# ---------- Smoke test 2 — embedding round-trip ----------

@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def test_embedding_round_trip(embedder):
    import numpy as np

    vectors = embedder.encode(
        ["hello world", "retrieval is fun"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    assert vectors.shape == (2, 384)
    # Embeddings are not identical.
    assert not np.allclose(vectors[0], vectors[1])

    # Ranking check: cosine("greeting", "hello world") > cosine("greeting", "retrieval is fun").
    q = embedder.encode(
        ["greeting"], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    # Normalized embeddings → dot product == cosine similarity.
    sim_hello = float(q @ vectors[0])
    sim_retr = float(q @ vectors[1])
    assert sim_hello > sim_retr, (sim_hello, sim_retr)


# ---------- Smoke test 3 — pgvector round-trip ----------

def test_pgvector_round_trip(embedder):
    import numpy as np
    import psycopg
    from pgvector.psycopg import register_vector

    docs = [
        "a greeting message",
        "an introduction to retrieval",
        "notes on dense vector search",
    ]
    query = "say hello"

    doc_vecs = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Semantic expectation: the greeting sentence is closest to "say hello".
    target_idx = int(np.argmax(doc_vecs @ q_vec))
    target_text = docs[target_idx]

    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS smoke_chunks;")
            cur.execute(
                "CREATE TABLE smoke_chunks ("
                "  id serial primary key,"
                "  text text,"
                "  embedding vector(384)"
                ");"
            )
            for text, vec in zip(docs, doc_vecs):
                cur.execute(
                    "INSERT INTO smoke_chunks (text, embedding) VALUES (%s, %s);",
                    (text, vec),
                )
            # HNSW index with cosine distance operator class.
            cur.execute(
                "CREATE INDEX smoke_chunks_hnsw "
                "ON smoke_chunks USING hnsw (embedding vector_cosine_ops);"
            )
            cur.execute(
                "SELECT text FROM smoke_chunks "
                "ORDER BY embedding <=> %s LIMIT 1;",
                (q_vec,),
            )
            (nearest_text,) = cur.fetchone()

    assert nearest_text == target_text, (nearest_text, target_text)


# ---------- Smoke test 4 — BM25 round-trip (Elasticsearch) ----------
# To swap to OpenSearch: replace `from elasticsearch import Elasticsearch` with
# `from opensearchpy import OpenSearch`; the basic index/match API shape is the
# same.

def test_bm25_round_trip():
    from elasticsearch import Elasticsearch

    es = Elasticsearch(ES_URL, request_timeout=30)
    # Wait up to a few seconds for the single-node cluster to be ready.
    for _ in range(20):
        if es.ping():
            break
        time.sleep(0.5)
    assert es.ping()

    index = "smoke-bm25"
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(
        index=index,
        mappings={"properties": {"text": {"type": "text"}}},
    )

    docs = [
        {"id": "1", "text": "vector databases store dense embeddings for retrieval"},
        {"id": "2", "text": "bm25 is the classical scoring function for lexical retrieval"},
        {"id": "3", "text": "cross encoder rerankers improve retrieval quality"},
    ]
    for d in docs:
        es.index(index=index, id=d["id"], document={"text": d["text"]})
    es.indices.refresh(index=index)

    # "bm25" appears in exactly one document.
    result = es.search(index=index, query={"match": {"text": "bm25"}})
    hits = result["hits"]["hits"]
    assert hits, "expected at least one BM25 hit"
    assert hits[0]["_id"] == "2"
    assert hits[0]["_score"] > 0.0


# ---------- Smoke test 5 — ir_measures round-trip ----------

def test_ir_measures_round_trip():
    import ir_measures
    from ir_measures import nDCG, R

    qrels = [
        ir_measures.Qrel(query_id="q1", doc_id="d1", relevance=1),
        ir_measures.Qrel(query_id="q2", doc_id="d3", relevance=1),
    ]
    # Run A ranks the ground-truth document first.
    run_good = [
        ir_measures.ScoredDoc(query_id="q1", doc_id="d1", score=3.0),
        ir_measures.ScoredDoc(query_id="q1", doc_id="d2", score=2.0),
        ir_measures.ScoredDoc(query_id="q1", doc_id="d3", score=1.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d3", score=3.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d2", score=2.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d1", score=1.0),
    ]
    # Run B ranks the ground-truth document second.
    run_worse = [
        ir_measures.ScoredDoc(query_id="q1", doc_id="d2", score=3.0),
        ir_measures.ScoredDoc(query_id="q1", doc_id="d1", score=2.0),
        ir_measures.ScoredDoc(query_id="q1", doc_id="d3", score=1.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d2", score=3.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d3", score=2.0),
        ir_measures.ScoredDoc(query_id="q2", doc_id="d1", score=1.0),
    ]

    good = ir_measures.calc_aggregate([nDCG @ 10, R @ 10], qrels, run_good)
    worse = ir_measures.calc_aggregate([nDCG @ 10, R @ 10], qrels, run_worse)

    good_ndcg = good[nDCG @ 10]
    good_recall = good[R @ 10]
    worse_ndcg = worse[nDCG @ 10]

    assert 0.0 <= good_ndcg <= 1.0
    assert 0.0 <= good_recall <= 1.0
    # Recall@10 is the same for both runs (ground truth is in top-10 either way).
    # nDCG@10 is strictly higher for the run that ranks ground truth first.
    assert good_ndcg > worse_ndcg, (good_ndcg, worse_ndcg)
```

Expected `pytest -q` output:

```text
.....                                                                    [100%]
5 passed in <t>s
```

### Deliverable D — `TOOLCHAIN.md`

```markdown
# TOOLCHAIN.md — mod-101 local toolchain reproducibility note

## Environment

- Host OS: <e.g. Ubuntu 22.04 / macOS 14.5 / Windows 11 + WSL2>
- Hardware: <e.g. Apple M2, 16 GB; or AMD 7950X + 64 GB + RTX 4090>
- Python: `<python --version>` — target: 3.11 or 3.12.
- Project manager: `uv` version `<uv --version>` (or Poetry if swapped).

## Container images (pinned)

- `pgvector/pgvector:pg16` — Postgres 16 with pgvector preinstalled.
- `docker.elastic.co/elasticsearch/elasticsearch:8.14.3` — single-node dev stack
  with `xpack.security.enabled=false`.
  (If using OpenSearch: `opensearchproject/opensearch:2.15.0` with
  `plugins.security.disabled=true`.)

**Security note.** Both engines ran with security *disabled*. This is
appropriate only for a local-development stack bound to `localhost`. These
flags must not be carried into any environment reachable by anything other than
localhost, including shared dev boxes. The Elasticsearch install docs and the
OpenSearch install docs document the production setup.

## Python package versions

Captured in `uv.lock` (committed). The top-level dependency list is in
`pyproject.toml`. Notable pins at bootstrap time:

- `llama-index == <lock-resolved>`
- `sentence-transformers == <lock-resolved>`
- `torch == <lock-resolved>` (CPU build unless CUDA detected — see PyTorch
  install matrix at https://pytorch.org/get-started/locally/)
- `psycopg[binary] == <lock-resolved>`
- `pgvector == <lock-resolved>`
- `elasticsearch == <lock-resolved>`
- `ir-measures == <lock-resolved>`
- `pytest == <lock-resolved>`

## Models

- Embedding model: `BAAI/bge-small-en-v1.5`
- Model revision (Git SHA on Hugging Face Hub): `<pin via sentence_transformers
  config; copy the revision from ~/.cache/huggingface after first load>`
  <!-- verify-against-source: paste the actual revision SHA after running test 2
       locally; the Hub page shows the most recent commit and `SentenceTransformer`
       records it in the local cache -->
- Model cache: `~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/`
  (set `HF_HOME` to relocate).

## Deviations from chapter 6

- Chapter 6 recommends Elasticsearch *or* OpenSearch. This project uses:
  `<fill in: Elasticsearch | OpenSearch>`.
- Chapter 6 recommends `uv` or `poetry`. This project uses: `<uv | poetry>`.
- If any dependency was pinned below a chapter-6 recommended version because
  of a transitive conflict, record it here with a one-line reason.

## Reproduction

```bash
docker compose up -d
docker compose ps        # confirm both services healthy
psql -h localhost -U rag -d rag -c "SELECT 1;"
curl -s http://localhost:9200 | head -20
uv sync
uv run pytest -q         # expect `5 passed`
```
```

## 3. Validation steps

1. `docker compose up -d` and `docker compose ps` — both services must report `healthy` (the
   healthchecks in the compose file are what the exercise asks for in a slightly implicit way).
2. The two connectivity checks from Deliverable B must both succeed before running pytest.
3. `uv sync` (or `poetry install`) completes without error and commits a lockfile.
4. `uv run pytest -q` (or `poetry run pytest -q`) returns `5 passed`.
5. Re-run `uv run pytest -q` immediately — it should still pass. The HF model is cached after
   the first download, Postgres state is dropped-and-recreated per test, and the Elasticsearch
   index is dropped-and-recreated per test. Intermittent flakes usually point at the
   Elasticsearch single-node cluster not being ready — the test's `ping()` loop handles up to
   ~10 seconds of warmup.
6. Open `TOOLCHAIN.md` and confirm Python version, OS/hardware, container tags, and model
   revision are all filled in. A placeholder-left-in TOOLCHAIN file is unfinished.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Deliverable A — pyproject | Exists, pins all eight dependencies, lockfile committed |
| Deliverable B — compose | Exists, both services declared healthy via healthchecks; connectivity checks succeed |
| Deliverable C — test 1 (imports) | All seven top-level modules import without error |
| Deliverable C — test 2 (embedding) | Shape `(2, 384)`, non-equal embeddings, correct cosine ranking on the greeting/retrieval pair |
| Deliverable C — test 3 (pgvector) | Extension created, table created with `vector(384)`, HNSW index on `vector_cosine_ops`, nearest-neighbor matches the semantically-closest document |
| Deliverable C — test 4 (BM25) | Index created, three documents indexed, `match` query returns the correct top-1 with a non-zero score |
| Deliverable C — test 5 (ir_measures) | Both metrics in `[0, 1]`, nDCG@10 strictly higher for the run that ranks ground truth first |
| Deliverable D — TOOLCHAIN.md | Python version, OS/hardware, container tags, model revision all filled in; deviations from chapter 6 recorded |
| Full-suite outcome | `pytest -q` prints `5 passed` |

Fail-fast smells (any one is a revise-and-resubmit):

- `xpack.security.enabled=false` without the TOOLCHAIN security note.
- `pyproject.toml` with no lockfile committed.
- A test that uses `assert True` or `pytest.skip` as a stub — the exercise explicitly says every
  test must fail loudly if the stack is wrong.
- `TOOLCHAIN.md` with the model revision left as `TBD`.

## 5. Common mistakes

- **Running `pytest` before `docker compose ps` says healthy.** Elasticsearch's single-node
  cluster takes a few seconds to come up. The test's `ping()` loop absorbs ~10 seconds; longer
  warmups require either waiting manually or adjusting the loop. A green test run that depends
  on the container being up for exactly the right duration is a fragile signal.
- **Forgetting `register_vector(conn)` in the pgvector test.** Without it, `psycopg` sends the
  Python list as a string literal to Postgres and the query does not use the HNSW index. The
  test might still pass by accident on a three-row table, but the failure mode is instructive to
  learn about now rather than in mod-104.
- **Mixing Elasticsearch and OpenSearch clients.** The exercise explicitly says: pick one and
  stay with it. Mixing the two produces type-error noise that eats an afternoon.
- **Normalizing embeddings in the embedding test but not in the pgvector test.** With `pgvector`,
  cosine distance is the right operator for normalized embeddings; the `vector_cosine_ops`
  operator class and the `<=>` operator in the SQL both encode that choice. Learners who
  normalize in one place but not the other get surprising top-1 matches.
- **Using `R@10` to distinguish the two ir_measures runs.** Both runs have the ground-truth
  document in the top-10, so recall@10 is identical. The ordering sensitivity shows up in nDCG@10,
  not in recall@10 — picking the right metric for the invariant being tested is part of the
  exercise.
- **Treating the TOOLCHAIN.md deviation note as optional.** Chapter 6 says every deviation from
  the recommended setup should be recorded. In practice the reasons deviations get made (a
  dependency resolver conflict, a GPU/CPU mismatch, a port already in use) are the ones a
  colleague would want to know about when they try to reproduce the setup.
- **Not committing the lockfile.** `pyproject.toml` lists loose constraints; `uv.lock` /
  `poetry.lock` records the resolved versions. Without the lockfile, the next clean install can
  pull a different resolved version of `sentence-transformers` or `torch` and the embedding test
  may change behavior subtly.

## 6. References

- Lesson sources in this module:
  - `lessons/mod-101-retrieval-engineering-foundations/06-local-toolchain-setup.md` — the
    chapter this exercise implements.
- `uv`: [docs.astral.sh/uv](https://docs.astral.sh/uv/).
- Poetry: [python-poetry.org/docs](https://python-poetry.org/docs/).
- PyTorch install matrix: [pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
- LlamaIndex: [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- sentence-transformers: [sbert.net](https://www.sbert.net/) and the model page
  [huggingface.co/BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
- pgvector: [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) and
  [github.com/pgvector/pgvector-python](https://github.com/pgvector/pgvector-python) (which the
  pgvector round-trip test uses).
- Elasticsearch install docs: [elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
- OpenSearch install docs: [opensearch.org/docs/latest/install-and-configure/](https://opensearch.org/docs/latest/install-and-configure/).
- `ir_measures` quickstart: [ir-measur.es/en/latest/getting-started.html](https://ir-measur.es/en/latest/getting-started.html).
