# Exercise 03 — Solution: pgvector vs. Qdrant vs. Milvus — hands-on index build and comparison

This is a reference solution for
`exercise-03-pgvector-vs-qdrant-vs-milvus-hands-on-index-build.md`. It is an
**implementation** exercise. The learner stands up all three vector stores locally
via Docker Compose, ingests the same corpus with the same embeddings, builds an
HNSW index with matched parameters, runs the same query set, and writes a one-page
comparison with Recall@10, p95 latency, build time, and on-disk size.

The numbers in the reference are **illustrative shapes**. The exercise is not a
vendor benchmark — it is a hands-on demonstration that three stores that all
"have HNSW" differ sharply in their APIs, their index-build ergonomics, and their
operational surface. The deliverable is reproducible scripts, not a chart that
claims one is "fastest."

## 1. Solution overview

The deliverable has six parts:

1. **`compose.yaml`** — Docker Compose that brings up all three stores locally
   on known ports (pgvector on 5432, Qdrant on 6333/6334, Milvus on 19530 with
   its etcd + minio sidecars).
2. **`src/ingest/`** — one ingest script per store, all consuming the same
   `data/embeddings.parquet` (pre-computed with the mod-103 winning embedder).
3. **`src/query/`** — one query client per store, all exposing the same
   `search(query_vec, top_k, filter=None) -> list[(doc_id, score)]` signature.
4. **`src/bench/`** — a harness that drives the three clients with the same
   query set, same `top_k`, same filters, times each call, and records
   Recall@10 against a brute-force baseline from exercise-02.
5. **`results/compare.csv` + `docs/compare.md`** — the table and the
   one-paragraph pick.
6. **`notes/api-differences.md`** — the thing the exercise is really teaching:
   where the three APIs disagree (filter syntax, upsert semantics, vector
   schema definition, metric naming) and how to think about those disagreements.

The teaching point: at HNSW with matched `M`, `ef_construction`, `ef_search`,
Recall@10 lands within noise across all three stores for a given query set.
The *reason* you pick one over another is the surrounding shape — filter
language, transactions, deployment, ops — not the recall number.

## 2. Worked answer or implementation

### 2.1 `compose.yaml` — the local stack

```yaml
# compose.yaml — local-only; do NOT deploy as-is.
services:
  pgvector:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: localdev
      POSTGRES_DB: rag
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro

  qdrant:
    image: qdrant/qdrant:v1.11.0
    ports: ["6333:6333", "6334:6334"]
    volumes:
      - qdrant-data:/qdrant/storage

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls=http://0.0.0.0:2379 -listen-client-urls=http://0.0.0.0:2379

  minio:
    image: minio/minio:RELEASE.2024-08-29T01-40-52Z
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: minio server /minio_data

  milvus:
    image: milvusdb/milvus:v2.4.6
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports: ["19530:19530"]
    depends_on: [etcd, minio]

volumes:
  pgdata:
  qdrant-data:
```

`sql/init.sql` is the pgvector setup:

```sql
-- sql/init.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
  chunk_id  text PRIMARY KEY,
  doc_id    text NOT NULL,
  tenant_id text NOT NULL,
  doc_type  text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  text      text NOT NULL,
  embedding vector(384) NOT NULL   -- match the mod-103 winner's dim
);

-- HNSW index. Match exercise-02 parameters for a fair cross-store compare.
CREATE INDEX chunks_embedding_hnsw
  ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);

CREATE INDEX chunks_tenant_doc_type_idx ON chunks (tenant_id, doc_type);
CREATE INDEX chunks_created_at_idx ON chunks (created_at);
```

### 2.2 Shared ingest input

All three ingests read the same `data/embeddings.parquet` with the schema:

```
chunk_id: str
doc_id: str
tenant_id: str
doc_type: str
created_at: int64 (unix seconds)
text: str
embedding: list<float32> (length = 384, L2-normalized for cosine)
```

Embeddings are L2-normalized once at build time. Cosine distance = 1 − inner
product on unit vectors, so inner product is the metric to pick in each store.

### 2.3 Per-store ingest — `src/ingest/`

**pgvector** — use `psycopg` and `COPY` for bulk load; index is created
separately so the HNSW build happens once.

```python
# src/ingest/pg.py
from __future__ import annotations
import io, json, pathlib
import psycopg, pyarrow.parquet as pq

def _row_to_copy(row: dict) -> str:
    emb = "[" + ",".join(f"{x:.6f}" for x in row["embedding"]) + "]"
    # COPY ... FORMAT text — tab-separated, escape nothing exotic.
    return "\t".join([
        row["chunk_id"], row["doc_id"], row["tenant_id"], row["doc_type"],
        str(row["created_at"]), row["text"].replace("\t", " ").replace("\n", " "),
        emb,
    ]) + "\n"

def ingest(parquet_path: pathlib.Path, dsn: str) -> None:
    table = pq.read_table(parquet_path)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        with cur.copy(
            "COPY chunks (chunk_id, doc_id, tenant_id, doc_type, created_at, text, embedding) "
            "FROM STDIN WITH (FORMAT text)"
        ) as copy:
            for batch in table.to_batches(max_chunksize=4096):
                for row in batch.to_pylist():
                    copy.write(_row_to_copy(row))
        conn.commit()
```

**Qdrant** — use the official `qdrant-client`. Create a collection with
matched HNSW parameters; upsert in batches; set payload indexes so filter
pushdown works.

```python
# src/ingest/qdrant.py
from __future__ import annotations
import pathlib, pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION = "chunks"

def _client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    return QdrantClient(host=host, port=port)

def setup(client: QdrantClient, dim: int = 384) -> None:
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=200),
    )
    # Payload indexes let the filter walk HNSW with pruning rather than post-filter.
    client.create_payload_index(COLLECTION, "tenant_id", qm.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "doc_type",  qm.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "created_at", qm.PayloadSchemaType.INTEGER)

def ingest(parquet_path: pathlib.Path) -> None:
    client = _client()
    setup(client)
    table = pq.read_table(parquet_path)
    points: list[qm.PointStruct] = []
    for batch in table.to_batches(max_chunksize=1024):
        for row in batch.to_pylist():
            points.append(qm.PointStruct(
                id=row["chunk_id"],
                vector=row["embedding"],
                payload={
                    "doc_id": row["doc_id"],
                    "tenant_id": row["tenant_id"],
                    "doc_type": row["doc_type"],
                    "created_at": row["created_at"],
                    "text": row["text"],
                },
            ))
            if len(points) >= 1024:
                client.upsert(COLLECTION, points=points, wait=False)
                points = []
    if points:
        client.upsert(COLLECTION, points=points, wait=True)
```

**Milvus** — use `pymilvus`. Define a schema explicitly (Milvus is schema-first),
create the collection, insert in batches, create the HNSW index after insert,
and load the collection into memory.

```python
# src/ingest/milvus.py
from __future__ import annotations
import pathlib, pyarrow.parquet as pq
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility,
)

COLLECTION = "chunks"

def setup(dim: int = 384) -> Collection:
    connections.connect(alias="default", host="localhost", port="19530")
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)
    schema = CollectionSchema(fields=[
        FieldSchema("chunk_id",  DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema("doc_id",    DataType.VARCHAR, max_length=128),
        FieldSchema("tenant_id", DataType.VARCHAR, max_length=64),
        FieldSchema("doc_type",  DataType.VARCHAR, max_length=32),
        FieldSchema("created_at",DataType.INT64),
        FieldSchema("text",      DataType.VARCHAR, max_length=8192),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])
    coll = Collection(name=COLLECTION, schema=schema, consistency_level="Strong")
    return coll

def ingest(parquet_path: pathlib.Path) -> None:
    coll = setup()
    table = pq.read_table(parquet_path)
    for batch in table.to_batches(max_chunksize=1024):
        rows = batch.to_pylist()
        coll.insert([
            [r["chunk_id"]  for r in rows],
            [r["doc_id"]    for r in rows],
            [r["tenant_id"] for r in rows],
            [r["doc_type"]  for r in rows],
            [r["created_at"]for r in rows],
            [r["text"]      for r in rows],
            [r["embedding"] for r in rows],
        ])
    coll.flush()
    coll.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "IP",   # cosine via pre-normalized vectors
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    coll.load()
```

### 2.4 Per-store query clients — `src/query/`

```python
# src/query/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    score: float

class SearchClient(Protocol):
    name: str
    def search(self, qvec: list[float], top_k: int,
               tenant_id: str | None = None,
               doc_types: list[str] | None = None,
               min_created_at: int | None = None) -> list[Hit]: ...
```

```python
# src/query/pg.py
import psycopg
from src.query.base import Hit

class PgClient:
    name = "pgvector"
    def __init__(self, dsn: str): self.dsn = dsn

    def search(self, qvec, top_k, tenant_id=None, doc_types=None, min_created_at=None):
        wheres, params = ["TRUE"], []
        if tenant_id is not None:
            wheres.append("tenant_id = %s"); params.append(tenant_id)
        if doc_types:
            wheres.append("doc_type = ANY(%s)"); params.append(doc_types)
        if min_created_at is not None:
            wheres.append("created_at >= to_timestamp(%s)"); params.append(min_created_at)
        where_sql = " AND ".join(wheres)
        # `<=>` is pgvector's cosine-distance operator; smaller is better.
        sql = (f"SELECT chunk_id, doc_id, (embedding <=> %s) AS dist "
               f"FROM chunks WHERE {where_sql} ORDER BY dist LIMIT %s")
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 100")
            cur.execute(sql, [qvec, *params, top_k])
            return [Hit(chunk_id=r[0], doc_id=r[1], score=-float(r[2])) for r in cur.fetchall()]
```

```python
# src/query/qdrant_client_wrap.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.query.base import Hit

class QdrantWrap:
    name = "qdrant"
    def __init__(self): self._c = QdrantClient(host="localhost", port=6333)

    def search(self, qvec, top_k, tenant_id=None, doc_types=None, min_created_at=None):
        must = []
        if tenant_id is not None:
            must.append(qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)))
        if doc_types:
            must.append(qm.FieldCondition(key="doc_type",  match=qm.MatchAny(any=doc_types)))
        if min_created_at is not None:
            must.append(qm.FieldCondition(key="created_at",
                                          range=qm.Range(gte=min_created_at)))
        filt = qm.Filter(must=must) if must else None
        res = self._c.search(
            collection_name="chunks", query_vector=qvec, limit=top_k,
            query_filter=filt, search_params=qm.SearchParams(hnsw_ef=100),
        )
        return [Hit(chunk_id=str(r.id), doc_id=r.payload["doc_id"], score=float(r.score)) for r in res]
```

```python
# src/query/milvus_client_wrap.py
from pymilvus import Collection, connections
from src.query.base import Hit

class MilvusWrap:
    name = "milvus"
    def __init__(self):
        connections.connect(alias="default", host="localhost", port="19530")
        self._c = Collection("chunks")

    def search(self, qvec, top_k, tenant_id=None, doc_types=None, min_created_at=None):
        parts = []
        if tenant_id is not None:
            parts.append(f'tenant_id == "{tenant_id}"')
        if doc_types:
            lst = ", ".join(f'"{t}"' for t in doc_types)
            parts.append(f"doc_type in [{lst}]")
        if min_created_at is not None:
            parts.append(f"created_at >= {int(min_created_at)}")
        expr = " and ".join(parts) if parts else None
        res = self._c.search(
            data=[qvec], anns_field="embedding", limit=top_k,
            param={"metric_type": "IP", "params": {"ef": 100}},
            expr=expr, output_fields=["doc_id"],
        )
        hits = res[0]
        return [Hit(chunk_id=str(h.id), doc_id=h.entity.get("doc_id"), score=float(h.distance))
                for h in hits]
```

### 2.5 Bench harness — `src/bench/run_compare.py`

```python
# src/bench/run_compare.py
"""Drive the three clients with the same queries; emit results.csv."""
from __future__ import annotations
import argparse, csv, pathlib, statistics, time
import numpy as np

from src.query.pg import PgClient
from src.query.qdrant_client_wrap import QdrantWrap
from src.query.milvus_client_wrap import MilvusWrap


def recall_at_k(got: list[str], truth: list[str], k: int) -> float:
    t = set(truth[:k])
    g = set(got[:k])
    return len(t & g) / max(1, len(t))


def _time_search(client, qvec, top_k, filters):
    t0 = time.perf_counter_ns()
    hits = client.search(qvec, top_k=top_k, **filters)
    return (time.perf_counter_ns() - t0) / 1e6, [h.chunk_id for h in hits]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=pathlib.Path, required=True,
                   help="jsonl with {id, text, embedding, filters, truth}")
    p.add_argument("--dsn", required=True)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/compare"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import json
    qs = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    clients = [PgClient(args.dsn), QdrantWrap(), MilvusWrap()]

    rows = []
    for c in clients:
        # Warmup.
        for q in qs[:16]:
            c.search(q["embedding"], top_k=10, **q.get("filters", {}))
        lats, recs = [], []
        for q in qs:
            ms, got_ids = _time_search(c, q["embedding"], 10, q.get("filters", {}))
            lats.append(ms)
            recs.append(recall_at_k(got_ids, q["truth"], 10))
        rows.append({
            "store": c.name,
            "n_queries": len(qs),
            "recall_at_10": statistics.mean(recs),
            "p50_ms": statistics.median(lats),
            "p95_ms": sorted(lats)[int(0.95 * len(lats)) - 1],
            "qps_single": 1000.0 / statistics.mean(lats),
        })

    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(rows)
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main(sys.argv[1:]))
```

### 2.6 `notes/api-differences.md` — the non-numerical finding

| Dimension | pgvector | Qdrant | Milvus |
|---|---|---|---|
| **Schema** | SQL columns. Add/rename with `ALTER TABLE`. | Collection has a vector config + free-form JSON payload. Add payload keys at upsert. | Schema-first: every field declared at `CollectionSchema` time. Adding a field needs a new collection. |
| **Distance / metric** | `vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops` as operator classes on the index. Query uses `<=>`, `<->`, `<#>`. | `qm.Distance.COSINE / EUCLID / DOT` at collection creation. Query is agnostic — one `search` call. | `metric_type` string at index creation (`IP`, `L2`, `COSINE`). Filter DSL is a Python-expression string, not structured. |
| **Index build time** | `CREATE INDEX ... USING hnsw`. Blocking unless `CONCURRENTLY`. Build is single-session and CPU-heavy. | `recreate_collection` with `HnswConfigDiff` — background indexing; `wait=True` on upserts forces catchup. | Build with `create_index`, then `load()` the collection. Index-build and load are separate steps; forget either and queries fail or run on brute force. |
| **Upsert semantics** | `INSERT ... ON CONFLICT ... DO UPDATE`. Transactional. | `upsert` replaces by point id; soft-delete + compaction under the hood. | `upsert` on PK; schema evolution is weak. |
| **Filter syntax** | SQL `WHERE`. | Structured `Filter(must=[...], should=[...], must_not=[...])`. | Python-expression string: `'doc_type == "faq" and tenant_id == "t1"'`. |
| **Filter-aware index** | SQL predicate evaluated outside the HNSW walk; `btree`/`hash` indexes on payload do help the planner. | Payload indexes (`create_payload_index`) let Qdrant prune during the HNSW walk. | Scalar indexes on filter fields reduce candidates before search. |
| **Query-time recall knob** | `SET hnsw.ef_search = N;` per session, or globally. | `SearchParams(hnsw_ef=N)` per call. | `param={"params": {"ef": N}}` per call. |
| **Consistency** | Postgres MVCC. Transactions over vectors + payload are one unit. | Durable WAL + snapshots; writes are eventually consistent across replicas. | `consistency_level` knob (`Strong`, `Session`, `Bounded`, `Eventually`). Default per-op. |
| **Deletes** | `DELETE FROM chunks WHERE ...` and the HNSW index tracks it. | `delete(points=...)` or filter-based delete; tombstones with online compaction. | `delete(expr=...)`; compaction is a background op. |
| **Backups** | `pg_basebackup`, PITR via WAL — same tooling as any other Postgres DB. | Snapshot + restore APIs; replicas for HA. | Snapshot via `utility.create_backup` / rely on object-store side (MinIO/S3). |

The point of the table: **the HNSW recall numbers in `results/compare.csv` will
be within noise across the three stores at matched `M=16, ef_construction=200,
ef_search=100`.** That is the correct observation. What differs is the surface
area: schema rigidity, filter language, consistency defaults, and the operational
burden. The learner's recommendation should be driven by those differences, not
by a 2ms p95 gap.

### 2.7 `docs/compare.md` — reporting template

```markdown
# pgvector / Qdrant / Milvus — matched-HNSW comparison

Date: <YYYY-MM-DD>
Corpus: <N> chunks, 384-dim, L2-normalized (cosine via IP).
Index params (all three): HNSW M=16, ef_construction=200, ef_search=100.
Queries: <nq> queries from <source>, each with a (doc_id-level) truth list.
Hardware: <lscpu / docker host>.

## Results

| Store | Recall@10 | p50 ms | p95 ms | Build time | On-disk size | Ingest throughput |
|---|---|---|---|---|---|---|
| pgvector | ... | ... | ... | ... | `\dt+ chunks; \di+ chunks_embedding_hnsw` | ... |
| Qdrant | ... | ... | ... | ... | `/qdrant/storage` du -sh | ... |
| Milvus | ... | ... | ... | ... | MinIO bucket size | ... |

## Observations

- Recall@10 is within <X.XX> across all three — matched HNSW parameters produce
  matched recall, as expected.
- Latency differences are dominated by <network round-trip / planner overhead /
  Python client framing>, not by the index algorithm.
- Build time differs because <per-store: pgvector single-session CPU build;
  Qdrant background indexing; Milvus explicit index + load>.

## Recommendation

For the mod-104 baseline we pick `<store>` because `<one-sentence driver
from notes/api-differences.md>`. We would switch to `<alternative>` if
`<measured trigger>`.
```

**Calibrated expectation.** Recall@10 across the three stores at matched HNSW
parameters should be identical to ~1%. If one store's Recall@10 is noticeably
lower, the most common culprit is a metric mismatch (one store is actually
running L2 while the others run IP) or a filter that is silently returning
different candidate sets.

## 3. Validation steps

1. `docker compose up -d` brings up all five services (pgvector, Qdrant, etcd,
   MinIO, Milvus). All report healthy within 60s.
2. `uv run python -m src.ingest.pg --parquet data/embeddings.parquet` runs and
   `SELECT count(*) FROM chunks;` matches the parquet row count. Then
   `CREATE INDEX ... USING hnsw` completes and `EXPLAIN` on a vector query shows
   `Index Scan using chunks_embedding_hnsw`.
3. `uv run python -m src.ingest.qdrant` completes; `GET http://localhost:6333/collections/chunks`
   reports `points_count` equal to the parquet row count and
   `vectors_count == points_count`.
4. `uv run python -m src.ingest.milvus` completes; `coll.num_entities` equals
   the parquet row count and `utility.index_building_progress("chunks")` reports
   100% before the bench starts.
5. `uv run python -m src.bench.run_compare` prints three rows with
   `recall_at_10 ∈ [0, 1]` and Recall@10 within ~1% across stores.
6. The same query vector, same filters, same `top_k=10` returns the same
   top-1 result across all three stores on at least 95% of queries. (Minor
   permutation at ranks 2–10 is normal.)

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Matched parameters | All three stores use HNSW M=16, ef_construction=200, ef_search=100, cosine (via IP on unit vectors). The learner can point at the line that sets each. |
| Same inputs | All three ingests consume the same `data/embeddings.parquet`. No per-store re-embedding. |
| Same queries | The bench harness feeds identical query vectors and identical filters to every client. |
| Recall truth | Recall@10 is measured against a shared ground-truth list (a doc_id-level truth or a brute-force top-K from exercise-02). |
| Filters | At least one filtered-query set (`tenant_id = 't1' AND doc_type IN ('faq')`) is included; the three stores' filter languages are shown side by side in `notes/api-differences.md`. |
| Build-time reporting | Build time is measured for each store with the same N and same dim. Milvus's `create_index` + `load()` are both counted, pgvector's single `CREATE INDEX` is counted, Qdrant's `wait=True` final upsert is counted. |
| On-disk size | The learner measured, not estimated, the on-disk footprint for each store. |
| Recommendation | Picks one store and names one API-shape driver from the table (not just "lowest p95"). |
| Reproducibility | `compose.yaml` runs clean on a fresh machine; `uv run` scripts are deterministic given fixed embeddings. |

Fail-fast smells:

- Recall@10 differs by more than ~5% between stores on the same inputs. Almost
  always a metric mismatch (one is on L2, the others on IP) or a filter
  returning a different candidate set.
- Milvus numbers include `load()` time in the "search" latency. `load()` is
  cold-start; it belongs in a separate column.
- pgvector query uses `ORDER BY embedding <=> $1` without `SET hnsw.ef_search`.
  The default `ef_search` is low; the sweep is unfair.
- Qdrant uses `search()` with no `SearchParams(hnsw_ef=...)`. Same issue as
  above — default `ef_search` is per-collection and may not match the matched
  parameter target.
- The three ingests use different embedders ("bge-small" in one, "all-MiniLM"
  in another). The exercise is comparing stores, not embedders; fix the
  embedder before touching the stores.

## 5. Common mistakes

- **Comparing stores on different HNSW configs.** The point is: at matched
  parameters, recall is matched. If the learner reports Qdrant M=32 vs.
  pgvector M=16 and concludes "Qdrant has better recall," they have compared
  configurations, not stores.
- **Running queries through three different network paths.** Local Postgres
  vs. local Qdrant HTTP vs. local Milvus gRPC have different per-call
  overhead. Report framing cost separately or run all queries through the
  same wrapper.
- **Ignoring the `load()` step in Milvus.** A Milvus collection that is not
  loaded serves queries at brute-force recall but with much higher latency —
  easy to mis-attribute to "Milvus is slow."
- **Treating "cosine" the same across stores.** pgvector exposes
  `vector_cosine_ops` (distance `<=>` in `[0, 2]`). Qdrant exposes
  `Distance.COSINE`. Milvus asks you to pre-normalize and use `IP`. All three
  land at the same ranking on unit vectors; they do not land at the same
  numeric score.
- **Not pre-normalizing vectors.** On cosine-with-IP stores (Milvus, Qdrant
  `DOT`), unit-norm is required. On non-normalized inputs, IP ranks by the
  product of the magnitudes and the retrieval is dominated by long vectors.
- **Measuring p95 at batch=100 on one store and batch=1 on another.** Batch
  shape dominates latency; pin it.
- **Forgetting that pgvector is Postgres.** The usual Postgres caveats apply:
  cold `shared_buffers` → first queries are I/O-bound; `ANALYZE` after bulk
  load; `autovacuum` settings for churn. A naked default-tuned Postgres will
  underperform both competitors; make the comparison fair.
- **Over-generalizing from one corpus.** The three stores all run HNSW; at
  matched parameters they will produce similar recall on *any* corpus. The
  recommendation should be about the surface area (schema, filters,
  consistency), not about recall from a single run.

## 6. References

- Store documentation (authoritative for APIs):
  - pgvector — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
  - pgvector HNSW index docs — [github.com/pgvector/pgvector#hnsw](https://github.com/pgvector/pgvector#hnsw).
  - Qdrant docs — [qdrant.tech/documentation/](https://qdrant.tech/documentation/).
  - Qdrant HNSW parameters — [qdrant.tech/documentation/concepts/indexing/](https://qdrant.tech/documentation/concepts/indexing/).
  - Qdrant filtering and payload indexes — [qdrant.tech/documentation/concepts/filtering/](https://qdrant.tech/documentation/concepts/filtering/).
  - Milvus docs — [milvus.io/docs](https://milvus.io/docs).
  - Milvus HNSW index — [milvus.io/docs/index.md#HNSW](https://milvus.io/docs/index.md).
  - `pymilvus` — [github.com/milvus-io/pymilvus](https://github.com/milvus-io/pymilvus).
- Methodology:
  - `ann-benchmarks` — [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks).
  - HNSW original paper — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
- Risk / governance framings consulted while drafting the comparison:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
