# Exercise 01 — Solution: Vector Database Family Comparison Matrix

This is a reference solution for
`exercise-01-vector-db-family-comparison-matrix.md`. It is a **design** exercise. The
learner builds a comparison matrix across the main vector-database families — a
library (FAISS), a Postgres extension (pgvector), a standalone open-source server
(Qdrant / Weaviate / Milvus), a search-engine integration (Elasticsearch / OpenSearch /
Vespa), and a hosted/managed service (Pinecone) — and writes a one-page recommendation
for the mod-104 single-corpus RAG baseline. The reference below is a worked matrix,
decision rationale, and grading rubric — not a vendor leaderboard.

The exercise is not asking for "which is best." Each family lives on a different point
in the space of (control vs. ops burden) × (richness of filtering) × (scale). The
grading rubric checks whether the learner can name that point in one sentence per
family and defend the recommendation for *this* stack.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/family-matrix.md`** — a table that answers, for each family, eight
   questions: deployment shape, index algorithms supported, metadata filtering
   strategy, consistency / durability model, horizontal scaling story, query-time
   knobs exposed to the application, operational burden, and licensing / vendor
   lock-in profile.
2. **`notes/decision.md`** — a one-page recommendation for the mod-104 single-corpus
   baseline (≤10M vectors, mixed metadata filters, single-region deploy) that names
   one primary pick and one escape hatch, with the trigger for switching written
   down.
3. **`notes/disqualifiers.md`** — a short list of factual "do-not-ship" conditions
   that eliminate a family outright for this stack (e.g., "no delete API" rules out
   a library-embedded index for corpora that get updated).

The matrix is *comparative*, not a feature inventory. A reviewer should be able to
read one column and know in a sentence why this family exists as a distinct category.

## 2. Worked answer or implementation

### 2.1 Reference families

The matrix fixes five families. The specific products named inside each family are
representative; swapping one product for another in the same family does not change
the matrix row.

| Family | Representative products | What makes it its own category |
|---|---|---|
| **A. Library / embedded** | FAISS, hnswlib, ScaNN, Annoy | You link a library into your process. No server, no network round trip, you own persistence. |
| **B. Postgres extension** | pgvector, pgvecto.rs | Vectors live as a column in your existing OLTP database. You get SQL, transactions, joins, and whatever ops skill the team already has for Postgres. |
| **C. Standalone OSS server** | Qdrant, Weaviate, Milvus, Chroma | A purpose-built vector service you run (or pay someone to run). Native vector-first APIs, rich payload filtering, replication/sharding out of the box. |
| **D. Search-engine integration** | Elasticsearch, OpenSearch, Vespa, Redis Stack | A general-purpose search / document store that grew a dense-vector index. Hybrid lexical + vector in one query, mature cluster story, operations are search-engine operations. |
| **E. Hosted / managed service** | Pinecone (also the hosted tiers of C) | SaaS. You get an API; you do not run the storage layer. Fastest to ship, hardest to inspect at the bytes level. |

### 2.2 `notes/family-matrix.md` — the comparison table

| Dimension | A. Library | B. pgvector | C. Standalone OSS | D. Search-engine | E. Hosted |
|---|---|---|---|---|---|
| **Deployment shape** | In-process library; you own persistence and recovery. | A Postgres extension; shares the Postgres cluster's deploy, backups, replication. | A separate service you run (binary / container / helm chart). | A separate service you run, usually already deployed for logs / text search. | SaaS API. No local state. |
| **Index algorithms** | FAISS: Flat, IVF, IVF-PQ, HNSW. hnswlib: HNSW only. ScaNN: tree + AH. Annoy: random projections. | HNSW and IVF-Flat as of pgvector ≥ 0.5; `vector` and `halfvec` types; optional `sparsevec`. | HNSW universally; Milvus adds IVF / IVF-PQ / DiskANN variants; Weaviate exposes `flat` for small corpora. | Lucene HNSW (Elasticsearch / OpenSearch); Vespa HNSW; Redis HNSW / FLAT. | HNSW-family under the hood; the exact algorithm is a product detail, not a user knob. |
| **Metadata filtering** | None in the library itself; you keep a parallel store and filter *outside* the index. | SQL `WHERE` clauses on any column. Pre-filter and post-filter both possible; see exercise-04 for the correctness drill. | Native payload filters with boolean / range / geo operators; in-index filter integration (Qdrant payload index, Milvus scalar index). | Full lexical query language (DSL / YQL) combined with `knn`/`hybrid` — richest filter surface in practice. | Metadata filters as first-class JSON on each vector; filter expression surface varies by product. |
| **Consistency & durability** | Whatever you implement. The library gives you a memory structure; snapshotting is your problem. | Postgres MVCC + WAL + PITR. Single-writer; standard Postgres replication. | Write-ahead log + snapshotting per product; typically eventually consistent reads on replicas. | Search-engine semantics: refresh-interval eventually consistent, snapshot-restore for durability. | Opaque; vendor's SLA is the contract. |
| **Horizontal scale** | Sharding is your code. | Postgres sharding (Citus, partitioning, per-tenant DBs) with real care; vertical scale is the easy path. | Native sharding / replication (Milvus sharded cluster; Qdrant collections with shards + replicas; Weaviate multi-node). | Native sharding and replication — the main reason the family exists. | Pay-per-pod or pay-per-serverless; the cluster is the vendor's problem. |
| **Query-time knobs** | Full access — you call the library directly. `efSearch`, `nprobe`, `k`, every reranking step is yours. | SQL + GUC settings (`hnsw.ef_search`, `ivfflat.probes`). Knobs are set per-session or per-index. | Collection / index parameters at create time; per-query parameters (`hnsw_ef`, `nprobe`) on each search. | Index-time parameters (`m`, `ef_construction`); per-query `k`, `num_candidates`, filter. | Fewer knobs. Pod size / replica count; occasionally per-query `top_k` and `filter`. |
| **Operational burden** | You run a process that happens to have an index in it. No separate service, but also no built-in snapshot / replica tooling. | Adds one extension to an existing Postgres footprint. The ops skill you already have carries over. | A new service to run. Helm charts exist; so do new failure modes (OOM at index load, cold-start, snapshot-restore). | If you already run Elasticsearch or OpenSearch, incremental. If you do not, the cluster itself is the bulk of the work. | Lowest day-to-day burden; highest *control* cost when something goes wrong. |
| **Licensing / lock-in** | FAISS: MIT. hnswlib: Apache-2.0. ScaNN: Apache-2.0. Full control of your bytes. | PostgreSQL license (pgvector). Portable because the data model is SQL. | Apache-2.0 / BSL / Elastic-License varies by product — read before adopting. Qdrant Apache-2.0; Weaviate BSD-3; Milvus Apache-2.0. | Mixed: Elasticsearch moved to ELv2/SSPL (fork: OpenSearch Apache-2.0); Vespa Apache-2.0; Redis Stack had license changes historically. | Vendor contract. Lowest portability. |

### 2.3 Capability / fit summary, in plain English

- **A — Library.** Lowest latency per query because there is no process boundary.
  Hardest to scale past "one box" and hardest to keep in sync with updates. Use when
  the corpus is small, updates are batch, and your workload is a single service.
  FAISS is the reference implementation of most ANN research; if you want to
  understand what a vector DB is doing under the hood, this is where you read it.
- **B — pgvector.** Lowest *incremental* cost when the team already runs Postgres.
  Transactional guarantees over vectors, SQL filtering, joins with the authoritative
  row — all free. The ceiling is single-writer throughput and index-build time at
  high scale; HNSW builds in Postgres are CPU-bound and index memory is a regular
  shared-buffers concern.
- **C — Standalone OSS.** The purpose-built shape. Richest native filtering, native
  sharding/replication, native snapshot/restore, HTTP + gRPC APIs. The cost is
  running another service with its own ops profile — and the fact that your
  canonical data still lives somewhere else (Postgres, S3, a warehouse), so you
  take on a sync problem.
- **D — Search-engine integration.** Best when hybrid lexical + vector is required
  from day one (product search, logs, enterprise search). Vector is one query
  clause among many; the filter DSL is unmatched. Index-build and recall knobs are
  available but the vector path inherits Lucene's segment model, which affects
  how you tune `ef_construction` and how long force-merges take.
- **E — Hosted.** Pay for someone else to run the cluster. Right default when the
  team is small, the roadmap is "ship RAG this quarter," and the corpus is ≤ a few
  million vectors. Trade-offs appear when you need deterministic behavior for
  debugging, on-prem data residency, or very high QPS at low cost.

### 2.4 `notes/decision.md` — recommendation for the mod-104 baseline

**Baseline assumptions.** Corpus ≤ 10M passages after chunking (mod-102), dense
bi-encoder embeddings (mod-103), single region, < 100 QPS steady-state, metadata
filters on `tenant_id`, `doc_type`, `created_at`, `source`. Updates are
hourly-to-daily. The team already runs Postgres.

**Primary pick: pgvector (family B).** Reasons, in order:

1. The team already operates Postgres. The *incremental* ops surface is one
   extension and a pair of GUC settings.
2. Metadata filtering is SQL. `WHERE tenant_id = $1 AND doc_type IN (...) AND
   created_at >= $2` is a predicate the team already reads fluently.
3. The "canonical row + its embedding" live in the same transaction. There is no
   separate sync pipeline for the mod-104 scale tier.
4. HNSW is supported since pgvector 0.5; `halfvec` (2×-cheaper storage) since 0.7.
   These are the knobs the exercise-02 sweep targets.

**Escape hatch: Qdrant (family C).** Switch when any of the following becomes true:

- Corpus size crosses ~20–50M vectors and index-build time in Postgres becomes an
  operational blocker. Qdrant's segment model and payload indexes are built for
  that regime.
- Filter latency starts to dominate because filtered recall requires post-filtering
  very large shortlists. Qdrant's payload-indexed filtering integrates with the
  HNSW traversal; pgvector's filter is a SQL predicate applied outside the index
  (see exercise-04).
- Multi-tenant isolation requires per-tenant shards / replicas rather than a
  single shared table with a `tenant_id` column.

**Not recommended for this baseline:**

- **FAISS / hnswlib (A)** — no delete semantics and no filter pushdown means every
  update day becomes "rebuild the shard." Fine as the kernel inside a service; not
  as the user-facing database.
- **Elasticsearch / OpenSearch (D)** — correct if lexical hybrid is required from
  day one, but mod-104 is a dense-first baseline; the lexical path arrives in
  mod-106. Introducing a cluster solely for vectors is operational cost without a
  hybrid-search payoff yet.
- **Hosted / Pinecone (E)** — correct if the team cannot own a stateful service.
  For this baseline, the Postgres cluster already is that service; adding a
  separate billed system is cost without a corresponding operational saving.

### 2.5 `notes/disqualifiers.md` — do-not-ship conditions

Each family has one or two conditions that take it off the table before cost even
enters the conversation.

- **Family A disqualifier**: corpus receives per-document updates or deletes and
  there is no custom rebuild/compaction story. HNSW library deletes are soft
  ("tombstones"); over time recall degrades without a rebuild. Ship only when you
  own the rebuild cadence.
- **Family B disqualifier**: workload needs sub-20ms p95 filtered search over
  100M+ vectors with changing payload. pgvector will do it, but at a materially
  bigger Postgres instance than the team is likely paying for — the budget breaks
  before the technology does.
- **Family C disqualifier**: no owner for the service and no runbook. Running
  Qdrant / Weaviate / Milvus is cheap compared to Elasticsearch, but it is not
  zero; "we spun it up and forgot about snapshots" is the common failure.
- **Family D disqualifier**: the team is not already on the search engine. Adding
  an Elasticsearch / OpenSearch / Vespa cluster *only* for vector search is
  usually the wrong tool.
- **Family E disqualifier**: data residency, on-prem, or air-gap requirements.
  Hosted is out when the corpus cannot leave the VPC.

## 3. Validation steps

1. The learner can map each row of the matrix to the concrete mechanism that makes
   the row true. For example, for "metadata filtering" the pgvector cell should
   trace to "SQL `WHERE` evaluated by the Postgres planner, possibly before or
   after the vector operator depending on selectivity." If the learner cannot
   name the mechanism, the row is a guess.
2. The learner can state — without the matrix in hand — which family has no
   document-side delete semantics (A, by default), which has transactional
   consistency (B), which has the richest filter DSL (D), and which has the
   fewest user-facing knobs (E).
3. The decision doc names one *condition* under which the choice changes, not
   "it depends." The condition should be measurable (corpus size, filter
   latency, tenancy count) rather than aesthetic.
4. The disqualifier list is factual. Each disqualifier is attributable to a
   limitation that is true today, not a vendor pitch ("Pinecone is closed
   source" is not a disqualifier; "data cannot leave the VPC" is).
5. The matrix cells about licensing are dated. Elasticsearch / Redis Stack have
   changed licenses in recent years; if the learner's cell reads "Apache-2.0"
   for Elasticsearch, they have not re-checked.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Family taxonomy | The matrix uses a taxonomy by *shape* (library / RDBMS extension / standalone service / search-engine / SaaS), not by product name. A matrix of "Pinecone vs. Weaviate vs. Qdrant" has lost the taxonomy. |
| Cell accuracy | Each cell names a mechanism or a documented feature. "Fast" and "slow" are not cells; "HNSW with `ef_search` tuned per query" is a cell. |
| Filtering | The filtering row distinguishes (i) no native filter (A), (ii) SQL `WHERE` outside the index (B), (iii) native payload + index-integrated filtering (C), (iv) full DSL with hybrid lexical (D), (v) JSON metadata filters (E). |
| Consistency | B is called out as transactional (MVCC + WAL). D is called out as refresh-eventually-consistent. The other families get the right semantics. |
| Scale | The scaling row distinguishes "vertical only" from "native shard+replica" from "vendor's problem." |
| Licensing | Each cell cites the actual current license — Elasticsearch ≠ Apache-2.0; OpenSearch = Apache-2.0; Qdrant Apache-2.0; Weaviate BSD-3; Milvus Apache-2.0; pgvector PostgreSQL License; FAISS MIT; hnswlib Apache-2.0. |
| Recommendation | Picks one family, names one escape hatch, names one measurable trigger for switching, and does *not* default to "use all five." |
| Disqualifiers | At least one per-family do-not-ship condition. Each condition is testable (a yes/no fact about the workload or the team), not a vibe. |
| Honesty about ops | Family E is not described as "free" and family A is not described as "cheap at scale." Both are called out at their actual cost shape. |

Fail-fast smells:

- Calling pgvector "a search engine." It is a Postgres extension; the Postgres planner
  does the filtering. Naming this wrong leads to wrong expectations about filter
  latency (see exercise-04).
- Confusing FAISS with a database. FAISS is a library; "delete a vector" is not an
  API call on an HNSW index (it's a tombstone + rebuild cycle).
- Treating "hosted" as a separate family *from* its underlying open-source product.
  If the learner lists "Pinecone" and "Qdrant Cloud" as different families, they
  have confused deployment mode with product family — all of C is available hosted.
- Omitting the "search-engine with vectors" family entirely. Elasticsearch,
  OpenSearch, and Vespa are often the *right* answer for product-search
  workloads; a matrix that leaves them out mis-frames the whole space.

## 5. Common mistakes

- **Comparing products within one family and calling it "the matrix."** Qdrant vs.
  Weaviate vs. Milvus vs. Chroma is useful, but it is a *within-family* comparison.
  Without pgvector, FAISS, and Elasticsearch on the same page, the learner cannot
  defend the architectural decision.
- **Picking "whatever has HNSW."** Nearly every vector DB has HNSW. HNSW is not a
  differentiator; the index algorithm is table stakes. The differentiators are
  filtering shape, consistency model, and ops surface.
- **Treating latency numbers from a vendor benchmark as portable.** A p95 published
  under cosine distance with a specific HNSW config on a specific corpus does not
  move to your corpus unchanged. The matrix should describe the *shape* of the
  performance story, not pretend the numbers are universal.
- **Ignoring the delete and update story.** Vector DBs differ sharply on
  updates/deletes (pgvector: regular DML; Qdrant: tombstones with online compaction;
  Elasticsearch: segment merges; FAISS library: user-supplied). Any matrix that
  does not address deletes will produce a wrong recommendation for corpora that
  change.
- **Hand-waving licensing.** Licensing is not a footnote. It determines whether the
  team can ship into a customer's environment. A matrix that says "open source"
  for Elasticsearch after 2021 is wrong on a point that materially constrains
  deployment choices.
- **Picking the hosted family by default "because ops is hard."** Hosted saves ops,
  but at mod-104 scale (≤10M vectors) the Postgres the team *already* runs is
  cheaper than a managed vector-DB tier that adds a second vendor contract.
- **Forgetting that the recommendation must survive mod-106 (hybrid retrieval).**
  If the primary pick cannot do BM25, the plan to add hybrid in mod-106 needs
  either (i) a second store and a fusion step at the application layer (mod-106
  default) or (ii) a family-D migration. Name this explicitly rather than
  discover it later.

## 6. References

- Index algorithms (what each family actually implements):
  - Malkov & Yashunin (2016), "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (HNSW) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
  - Jégou, Douze, Schmid (2011), "Product Quantization for Nearest Neighbor Search" (IVF-PQ) — [ieeexplore.ieee.org/document/5432202](https://ieeexplore.ieee.org/document/5432202).
  - Guo et al. (2020), "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ScaNN) — [arxiv.org/abs/1908.10396](https://arxiv.org/abs/1908.10396).
  - Jayaram Subramanya et al. (2019), "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" — [papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node).
- Family A — libraries:
  - FAISS — [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss), wiki — [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki).
  - hnswlib — [github.com/nmslib/hnswlib](https://github.com/nmslib/hnswlib).
  - ScaNN — [github.com/google-research/google-research/tree/master/scann](https://github.com/google-research/google-research/tree/master/scann).
  - Annoy — [github.com/spotify/annoy](https://github.com/spotify/annoy).
- Family B — Postgres extensions:
  - pgvector — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
  - pgvecto.rs — [github.com/tensorchord/pgvecto.rs](https://github.com/tensorchord/pgvecto.rs).
- Family C — standalone OSS servers:
  - Qdrant docs — [qdrant.tech/documentation/](https://qdrant.tech/documentation/).
  - Weaviate docs — [weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate).
  - Milvus docs — [milvus.io/docs](https://milvus.io/docs).
  - Chroma docs — [docs.trychroma.com/](https://docs.trychroma.com/).
- Family D — search-engine integrations:
  - Elasticsearch dense vector field and `knn` search — [elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
  - OpenSearch k-NN plugin — [opensearch.org/docs/latest/search-plugins/knn/](https://opensearch.org/docs/latest/search-plugins/knn/).
  - Vespa ANN — [docs.vespa.ai/en/approximate-nn-hnsw.html](https://docs.vespa.ai/en/approximate-nn-hnsw.html).
  - Redis vector search — [redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/](https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/).
- Family E — hosted:
  - Pinecone docs — [docs.pinecone.io](https://docs.pinecone.io/).
- Risk / governance framings consulted while drafting the recommendation:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
