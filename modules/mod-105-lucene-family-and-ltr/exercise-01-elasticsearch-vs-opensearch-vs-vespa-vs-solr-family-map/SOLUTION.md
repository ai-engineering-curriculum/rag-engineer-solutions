# Exercise 01 — Solution: Elasticsearch vs. OpenSearch vs. Vespa vs. Solr — The Family Map

This is a reference solution for
`exercise-01-elasticsearch-vs-opensearch-vs-vespa-vs-solr-family-map.md`. It is a
**design** exercise. The learner draws a family map of the four search engines that
dominate lexical and lexical+vector retrieval, and writes a one-page recommendation
for the mod-105 teaching corpus. The reference below is a worked matrix, decision
rationale, and grading rubric — not a vendor leaderboard.

A key teaching point: three of the four engines (Elasticsearch, OpenSearch, Solr) are
Apache Lucene libraries wrapped in a distributed service; **Vespa is not Lucene-based**
at all. A matrix that does not call this out has already lost the family taxonomy.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/family-matrix.md`** — a table that answers, for each engine, nine
   questions: lineage (who forked whom), core index library, license, cluster
   topology, vector-search support, learning-to-rank support, query DSL,
   operational shape, and governance.
2. **`notes/decision.md`** — a one-page recommendation for the mod-105 baseline
   (a single-tenant corpus of a few million passages that needs BM25 + dense
   vectors + LTR, run on-prem) with one primary pick, one escape hatch, and the
   measurable trigger for switching.
3. **`notes/disqualifiers.md`** — a short list of factual "do-not-ship"
   conditions that eliminate an engine outright for this stack (e.g., "license
   forbids offering the engine as a competing SaaS" rules Elasticsearch out for
   that class of product).

The matrix is *comparative*, not a feature catalog. A reviewer should be able to
read one column and say in a sentence why this engine exists as a distinct entry
rather than as a variant of another.

## 2. Worked answer or implementation

### 2.1 Lineage in one paragraph

Apache Lucene is a Java indexing library. Apache Solr (2004) and Elasticsearch
(2010) are both distributed-server wrappers around Lucene — different product
shape, same storage engine. In 2021 Elastic relicensed Elasticsearch from
Apache-2.0 to the Elastic License 2.0 / SSPL dual license. AWS forked the last
Apache-2.0 Elasticsearch release and renamed it **OpenSearch**. Both continue to
use Lucene and have diverged in feature set since. **Vespa** is a separate
codebase, Apache-2.0, open-sourced by Yahoo in 2017 and now maintained by Vespa.ai.
Vespa has its own inverted-index implementation, its own ANN library, and its own
query language (YQL); it is Lucene-adjacent in use cases but not in code.

### 2.2 `notes/family-matrix.md` — the comparison table

| Dimension | Elasticsearch | OpenSearch | Vespa | Solr |
|---|---|---|---|---|
| **Lineage** | Elastic NV; grew from Shay Banon's Compass (2010). | AWS fork of the last Apache-2.0 Elasticsearch release (2021); community-governed since. | Yahoo-origin; open-sourced 2017; now Vespa.ai. | Apache Software Foundation; CNET donation 2006. |
| **Core index library** | Apache Lucene (Java). | Apache Lucene (Java). | Proprietary C++ inverted index + ANN; no Lucene. | Apache Lucene (Java). |
| **License** | Elastic License 2.0 **and** SSPL v1 (dual); source-available, not OSI-approved open source. | Apache-2.0. | Apache-2.0. | Apache-2.0. |
| **Distribution model** | Master/data-node cluster; replication via primary + replica shards. | Same as Elasticsearch (fork). | Content nodes + container (stateless) nodes + config servers; separates query serving from indexing. | SolrCloud mode: ZooKeeper-coordinated collection/shard model. |
| **Query DSL** | JSON Query DSL (`match`, `bool`, `knn`, `rank_feature`, …). | JSON Query DSL (matches Elasticsearch 7.10 surface; has diverged since). | YQL — SQL-like: `select … from sources where … order by …`. Ranking is a separate "rank profile" artifact. | JSON Query DSL **and** Lucene query syntax; function queries; `/select` HTTP endpoint. |
| **Dense-vector / ANN** | `dense_vector` field with `index: true`; Lucene HNSW backing; `knn` query clause; supports filter, pre- or post-filter. | `knn_vector` field via the k-NN plugin; backends include Lucene HNSW, nmslib HNSW, Faiss (IVF/HNSW/PQ). | Native `tensor` field type + HNSW; first-class first-phase, second-phase, and global-phase ranking. | `dense_vector` field + `knn` query; Lucene HNSW backing (since Solr 9). |
| **Learning-to-rank** | Elasticsearch Learning to Rank plugin (originally OpenSource Connections, now maintained community + Elastic). Native `rank_feature` and `rank_features` fields. | Same plugin family available as an OpenSearch plugin; OpenSearch also ships a Neural Search plugin and reranker. | LTR is first-class: define features in a rank profile, train with the Vespa feature logger, serve XGBoost / LightGBM / ONNX models at query time. | Solr Learning-to-Rank module ships *in-tree* (`solr-ltr` module) — stores feature definitions and model in managed resources. |
| **Operational burden** | Mature; most third-party tooling targets this stack. Cluster is stateful; rolling upgrades and shard rebalance are the daily grind. | Same operational shape as Elasticsearch. Fewer managed-service options outside AWS historically; the gap is closing. | Highest conceptual bar — rank profiles, tensor framework, container/content split — but fewest surprise behaviors once in production because of the layered design. | Oldest of the four. SolrCloud + ZooKeeper is well-understood; the admin UI and config-set workflow have their own idioms. |
| **Governance** | Elastic NV (public company). | Linux Foundation (OpenSearch Software Foundation, 2024). | Vespa.ai company; Apache-2.0 codebase. | Apache Software Foundation; multi-committer governance. |

### 2.3 Capability / fit summary, in plain English

- **Elasticsearch.** The default reference stack for ELK-shaped workloads (logs,
  APM, observability) and for app-search where the JSON DSL and ecosystem
  (Kibana, Beats, agents, SDKs) are themselves the product. Vector and LTR are
  added surface rather than the raison d'être. The license change in 2021 is the
  single biggest consideration before adopting; for internal workloads it is a
  non-issue, for products that would "offer the engine as a service" it is a
  blocker.
- **OpenSearch.** Elasticsearch's Apache-2.0 doppelgänger for teams that need
  pure-OSS licensing. API-compatible with Elasticsearch 7.10 and has diverged
  since; the k-NN plugin exposes more ANN backends (nmslib, Faiss) than
  Elasticsearch's Lucene-only path. The gap in ecosystem depth versus
  Elasticsearch is real but shrinking. Preferred when (i) license matters and
  (ii) you want nmslib or Faiss ANN without an external service.
- **Vespa.** Different shape. Query path is a two-layer system (stateless
  container → stateful content). Ranking is a first-class pipeline — first
  phase (cheap), second phase (expensive, Top-K only), optional global phase
  (reranking across the whole result set). Native tensor types mean you can
  express BM25 + dense + sparse + LTR in one ranking expression without plugins.
  The cost is a steeper onboarding curve and a smaller tooling ecosystem.
- **Solr.** The oldest of the four. Strong in traditional enterprise search,
  library catalogs, e-commerce. LTR has shipped in the main distribution since
  Solr 6, which is years earlier than Elasticsearch's plugin equivalent. Dense
  vectors landed in Solr 9 with Lucene HNSW. The operational model is
  SolrCloud + ZooKeeper; it is well-trod ground but slightly off the hot path
  in modern RAG tooling.

### 2.4 `notes/decision.md` — recommendation for the mod-105 baseline

**Baseline assumptions.** A few million passages after chunking (mod-102), dense
bi-encoder embeddings (mod-103), lexical BM25 required from day one (this
module), LTR anticipated in the next iteration, on-prem deployment, single-region,
no SaaS resale of the engine. The team has no pre-existing search-engine footprint.

**Primary pick: OpenSearch.** Reasons, in order:

1. Apache-2.0 removes the license question entirely — no legal review before
   moving to production, no restriction on embedding the engine in a product.
2. BM25 is Lucene's default similarity since Lucene 6, so it is identical to
   Elasticsearch and Solr out of the box.
3. The k-NN plugin exposes Lucene HNSW (identical to Elasticsearch) **and**
   nmslib HNSW / Faiss IVF+PQ, which is useful when the embedding dimension or
   corpus size pushes past what Lucene HNSW handles economically.
4. The Learning-to-Rank plugin family is available and compatible with the
   LambdaMART workflow in exercise-04.
5. The JSON DSL surface is well-trodden and the ecosystem of clients, SDKs,
   and Kibana-equivalent dashboards (OpenSearch Dashboards) exists.

**Escape hatch: Vespa.** Switch when any of the following becomes true:

- The ranking stack grows beyond two phases — e.g., a learned first-phase
  retrieval scoring function **and** a neural second-phase reranker **and** a
  global cross-encoder — because Vespa's rank-profile model expresses this
  natively rather than as glue between plugins.
- Latency budgets fall below the Lucene segment-model floor. Vespa's content-node
  model is purpose-built for low-latency serving at high QPS.
- Multi-tenant hard isolation per content group becomes a requirement that the
  shard model struggles with.

**Not recommended for this baseline:**

- **Elasticsearch.** Same engine feel as OpenSearch, but the license question
  is non-zero overhead that this baseline does not need. Pick Elasticsearch
  when the team already runs Elastic Stack and the x-pack features (alerting,
  machine learning, Fleet, ES|QL) are load-bearing.
- **Solr.** Solid choice; loses to OpenSearch here only on ecosystem gravity
  for new RAG work and on the smaller gallery of SDKs and reference stacks in
  recent LLM tutorials. If the team already runs Solr, stay on Solr —
  reconfiguring is more expensive than migrating.

### 2.5 `notes/disqualifiers.md` — do-not-ship conditions

Each engine has a condition that takes it off the table before cost enters.

- **Elasticsearch disqualifier**: the product plans to offer the engine *itself*
  as a hosted service, or to resell it. The Elastic License 2.0 / SSPL
  explicitly restricts that use case.
- **OpenSearch disqualifier**: the team depends on Elastic-specific features
  that have not been reimplemented in OpenSearch (e.g., certain Elastic Stack
  integrations, ES|QL, Elastic Machine Learning). These features are not part
  of the fork.
- **Vespa disqualifier**: no engineer owns the stack and the team expects to
  copy-paste JSON from blog posts to get going. Vespa rewards learning its
  model and punishes cargo-culted configuration.
- **Solr disqualifier**: the team has no ZooKeeper experience and no appetite
  to run an additional coordination service. Single-node Solr exists but is
  not the recommended production shape.

## 3. Validation steps

1. The learner can state, without the matrix in hand, which of the four engines
   is **not** Lucene-based (Vespa) and why that matters (its ranking and ANN
   paths were designed together rather than bolted on to a lexical index).
2. The learner can name the licensing status of each engine as of 2026
   (Elasticsearch: Elastic License 2.0 + SSPL; OpenSearch: Apache-2.0; Vespa:
   Apache-2.0; Solr: Apache-2.0) and explain what the Elasticsearch license
   restricts.
3. The learner can map each row of the matrix to a concrete mechanism. For
   example, "Vector field" for Elasticsearch should trace to "`dense_vector`
   field with `index: true`, backed by Lucene HNSW; the `knn` query clause
   runs approximate kNN; filters may be applied pre- or post-`knn`." If the
   learner cannot name the mechanism, the row is a guess.
4. The decision document names one *condition* under which the choice changes,
   not "it depends." The condition should be measurable (QPS, latency budget,
   LTR phase count, license requirement) rather than aesthetic.
5. The matrix cells about LTR are accurate: Solr has LTR in-tree since Solr 6;
   Elasticsearch and OpenSearch have LTR via plugin; Vespa has LTR as a
   first-class rank-profile concept.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Family taxonomy | Correctly identifies the three Lucene-based engines (Elasticsearch, OpenSearch, Solr) and separates Vespa as a distinct codebase. |
| Lineage | Explains the Elasticsearch → OpenSearch fork and the license change that drove it. |
| License accuracy | Each cell cites the actual current license — Elasticsearch is **not** Apache-2.0 since 2021; OpenSearch, Vespa, and Solr are Apache-2.0. |
| Vector / ANN | For each engine, names the underlying ANN implementation (Lucene HNSW for ES/OS/Solr; nmslib/Faiss additionally for OpenSearch k-NN plugin; native C++ HNSW for Vespa). |
| LTR | Correctly identifies that Solr ships LTR in-tree, Elasticsearch/OpenSearch use a plugin, Vespa expresses LTR as a rank profile rather than as a separate plugin. |
| Cluster shape | Distinguishes master/data node (ES/OS), content/container/config (Vespa), and SolrCloud/ZooKeeper (Solr). |
| Query DSL | Notes that Vespa's query language (YQL) is SQL-like while the others are JSON DSLs; notes that Solr has both Lucene syntax and a JSON DSL. |
| Recommendation | Picks one engine, names one escape hatch, names one measurable trigger for switching, and does *not* default to "use whichever the cloud provider pushes." |
| Disqualifiers | At least one per-engine do-not-ship condition. Each condition is testable (a yes/no fact about the workload, team, or license), not a vibe. |
| Honesty about ops | Vespa is not described as "simpler"; Solr is not described as "legacy." Each is called out at its actual cost shape. |

Fail-fast smells:

- Calling Vespa "another Elasticsearch." It shares none of the Elasticsearch
  codebase. Naming this wrong leads to wrong expectations about plugins and
  JSON surface compatibility.
- Describing OpenSearch as "exactly Elasticsearch." It was a fork of the 7.10
  release; both have diverged. ES|QL, for example, exists only on the
  Elasticsearch side.
- Treating the Elasticsearch license as "basically open source." The Elastic
  License 2.0 restricts resale and specific hosted-service offerings; this
  materially affects some deployment plans.
- Omitting Solr. Solr is older than Elasticsearch, has LTR in-tree before any
  other engine on this list, and is the default for many library and
  e-commerce stacks. A matrix that leaves it out mis-frames the space.

## 5. Common mistakes

- **Treating Elasticsearch and OpenSearch as "the same thing" for this
  decision.** The API-level surface overlaps substantially, but license,
  governance, available features, and bundled plugins diverge. Document the
  divergence that actually affects this project (license here; k-NN backends
  for exercise-03; LTR path for exercise-04).
- **Framing Vespa as a vector database.** Vespa is a general-purpose
  serving engine that happens to support vectors as first-class tensors. It
  predates the current ANN wave by years and was built around multi-phase
  ranking, not pure similarity search.
- **Omitting the Elastic License 2.0 / SSPL change.** The license change is a
  factual event (2021), affects customer agreements, and is often the reason
  teams migrated to OpenSearch. A matrix that says "Apache-2.0" for
  Elasticsearch has not been re-checked since 2020.
- **Assuming LTR is a plugin everywhere.** Solr ships LTR in-tree since Solr 6
  (`solr-ltr` module). Vespa expresses LTR as a rank profile with native
  support for XGBoost / LightGBM / ONNX. Only Elasticsearch and OpenSearch use
  the plugin model for LambdaMART-style LTR.
- **Picking the engine by SDK availability only.** The SDK layer is the
  cheapest layer to replace; licensing, ranking model, and ANN backend are
  not. Decide on the layers that are hard to change.
- **Forgetting that BM25 is the default similarity.** Since Lucene 6
  (2016), BM25 is the default. Solr, Elasticsearch, and OpenSearch all use it
  unchanged unless you configure otherwise. A learner who treats "BM25
  support" as a differentiator has misread the taxonomy.
- **Ignoring the mod-106 (hybrid) handoff.** If the primary pick cannot do
  BM25 **and** dense ANN in the same query, exercise-01 produces a
  recommendation that exercise-03 and mod-106 will have to walk back. Vespa,
  Elasticsearch, OpenSearch, and Solr all do hybrid; the detail is which
  combiner each exposes (RRF, linear combination, custom script, rank profile).

## 6. References

- Lucene and Lucene-based engines:
  - Apache Lucene — [lucene.apache.org/core/](https://lucene.apache.org/core/).
  - BM25Similarity (Lucene default since 6.0) — [lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).
- Elasticsearch:
  - Elasticsearch reference — [elastic.co/guide/en/elasticsearch/reference/current/index.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html).
  - `dense_vector` field and `knn` search — [elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
  - Elastic License 2.0 — [elastic.co/licensing/elastic-license](https://www.elastic.co/licensing/elastic-license).
  - Elasticsearch Learning to Rank plugin — [github.com/o19s/elasticsearch-learning-to-rank](https://github.com/o19s/elasticsearch-learning-to-rank).
- OpenSearch:
  - OpenSearch documentation — [opensearch.org/docs/latest/](https://opensearch.org/docs/latest/).
  - k-NN plugin — [opensearch.org/docs/latest/search-plugins/knn/](https://opensearch.org/docs/latest/search-plugins/knn/).
  - OpenSearch Learning to Rank — [github.com/opensearch-project/opensearch-learning-to-rank-base](https://github.com/opensearch-project/opensearch-learning-to-rank-base).
  - OpenSearch Software Foundation — [opensearch.org/foundation/](https://opensearch.org/foundation/).
- Vespa:
  - Vespa documentation — [docs.vespa.ai/](https://docs.vespa.ai/).
  - Ranking overview and phases — [docs.vespa.ai/en/ranking.html](https://docs.vespa.ai/en/ranking.html).
  - Approximate nearest neighbor (HNSW) — [docs.vespa.ai/en/approximate-nn-hnsw.html](https://docs.vespa.ai/en/approximate-nn-hnsw.html).
  - YQL — [docs.vespa.ai/en/query-language.html](https://docs.vespa.ai/en/query-language.html).
- Solr:
  - Apache Solr reference guide — [solr.apache.org/guide/](https://solr.apache.org/guide/).
  - Solr Learning to Rank module — [solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html](https://solr.apache.org/guide/solr/latest/query-guide/learning-to-rank.html).
  - Solr dense vector search — [solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html](https://solr.apache.org/guide/solr/latest/query-guide/dense-vector-search.html).
- Risk / governance framings consulted while drafting the recommendation:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
