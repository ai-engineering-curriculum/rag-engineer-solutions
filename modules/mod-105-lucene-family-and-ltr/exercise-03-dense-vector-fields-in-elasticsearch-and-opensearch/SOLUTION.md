# Exercise 03 — Solution: Dense-Vector Fields in Elasticsearch and OpenSearch

This is a reference solution for
`exercise-03-dense-vector-fields-in-elasticsearch-and-opensearch.md`. It is an
**implementation + comparison** exercise. The learner adds a dense-vector field
to the mod-105 articles index on both Elasticsearch and OpenSearch, ingests
embeddings from the mod-103 bi-encoder, and runs equivalent `knn` queries on
each engine. The deliverable is a running pair of indices, a query harness
that exercises the two, and a short write-up of where the engines agree and
disagree.

The teaching point is that **both engines ship Lucene HNSW**, but the field
definitions, the `knn` query shape, and the filter semantics differ enough to
bite a port between them. OpenSearch's k-NN plugin additionally exposes nmslib
HNSW and Faiss (IVF, HNSW, PQ) backends — the exercise should touch the Lucene
backend first and demonstrate how to swap to Faiss for one configuration.

## 1. Solution overview

The deliverable has five parts:

1. **`mappings/articles_es.json`** — Elasticsearch mapping extending
   exercise-02 with a `dense_vector` field.
2. **`mappings/articles_os_lucene.json`** — OpenSearch mapping with
   `knn_vector` field backed by the Lucene engine (same ANN as Elasticsearch).
3. **`mappings/articles_os_faiss.json`** — OpenSearch mapping with
   `knn_vector` field backed by Faiss (HNSW + optional PQ).
4. **`src/ingest.py`** — the small Python ingester that reads the mod-103
   embedding file (`.npy` + `jsonl` metadata) and bulk-indexes to both
   engines using their respective clients.
5. **`src/query.py` + `notes/comparison.md`** — the probing queries (plain
   kNN, filtered kNN, hybrid BM25 + kNN) run against both engines, with a
   short write-up of the behavioral diffs (scoring normalization, filter
   pushdown, `k` vs. `num_candidates` semantics).

The exercise is not building a benchmark. It is proving the learner can read
both engines' documentation, write mappings that use the same embedding shape,
and recognize where a query has to be rewritten rather than just translated.

## 2. Worked answer or implementation

### 2.1 Embedding shape

Pick one embedding model from mod-103 and write down its shape in the exercise
notes: dimension `d`, metric (cosine vs. dot-product vs. L2), and whether
embeddings are pre-normalized. The mod-105 reference assumes:

- `d = 768`
- metric: cosine, with embeddings **pre-normalized to unit length** at
  ingestion time. (Pre-normalized cosine search can be served exactly by
  `dot_product` — this matters because both engines treat `dot_product` as
  the fastest path.)

If the exercise's embedding is 1024-dim, or not normalized, the mappings below
need `dims` and `space_type` / `similarity` adjusted accordingly.

### 2.2 `mappings/articles_es.json` — Elasticsearch dense_vector

```json
{
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "id":    { "type": "keyword" },
      "title": { "type": "text", "analyzer": "prose_en" },
      "body":  { "type": "text", "analyzer": "prose_en" },
      "tags":  { "type": "keyword", "normalizer": "exact_lower" },

      "embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "dot_product",
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 128
        }
      }
    }
  }
}
```

**Choices and why:**

- `index: true` turns the field from storage-only into an ANN-searchable
  field backed by Lucene HNSW.
- `similarity: "dot_product"` requires pre-normalized vectors. Elasticsearch
  validates norms on ingest for this similarity; if a vector arrives
  non-normalized, the request fails. That is a feature — it keeps the
  "pre-normalize at ingest" contract honest.
- `m: 16, ef_construction: 128` are the Lucene HNSW defaults as of recent
  releases and are a sane starting point. Tune in exercise-04 of mod-104 if
  this were a performance exercise.

### 2.3 `mappings/articles_os_lucene.json` — OpenSearch knn_vector, Lucene engine

```json
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "id":    { "type": "keyword" },
      "title": { "type": "text", "analyzer": "prose_en" },
      "body":  { "type": "text", "analyzer": "prose_en" },
      "tags":  { "type": "keyword", "normalizer": "exact_lower" },

      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": {
            "m": 16,
            "ef_construction": 128
          }
        }
      }
    }
  }
}
```

**Diffs vs. Elasticsearch worth internalizing:**

| Thing | Elasticsearch | OpenSearch |
|---|---|---|
| Field type | `dense_vector` | `knn_vector` |
| Dimension key | `dims` | `dimension` |
| Enable ANN | `"index": true` on the field | `"index.knn": true` on the index + `method` on the field |
| Metric | `similarity: "cosine" / "dot_product" / "l2_norm" / "max_inner_product"` | `space_type: "cosinesimil" / "innerproduct" / "l2" / "linf"` |
| HNSW tuning | `index_options.type: "hnsw"`, `m`, `ef_construction` | `method.name: "hnsw"`, `parameters.m`, `parameters.ef_construction` |

Same algorithm, different field surface. The port is mechanical but easy to
get wrong.

### 2.4 `mappings/articles_os_faiss.json` — OpenSearch knn_vector, Faiss engine

```json
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "id":    { "type": "keyword" },
      "title": { "type": "text", "analyzer": "prose_en" },
      "body":  { "type": "text", "analyzer": "prose_en" },
      "tags":  { "type": "keyword", "normalizer": "exact_lower" },

      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "innerproduct",
          "engine": "faiss",
          "parameters": {
            "m": 16,
            "ef_construction": 128,
            "ef_search": 128
          }
        }
      }
    }
  }
}
```

The Faiss engine exposes `ef_search` as an index-time parameter in addition to
`m` and `ef_construction`. It also unlocks IVF and PQ variants under the same
`knn_vector` field type — useful when the corpus is too large to keep HNSW
entirely in RAM. The Lucene engine stores HNSW graphs in segment files and
mmaps them, which has very different memory behavior.

Elasticsearch does **not** expose Faiss or nmslib. If an exercise learner's
plan calls for Faiss IVF+PQ on a Lucene-family engine, OpenSearch is the
answer — this is one of the concrete reasons exercise-01 lists in favor of
OpenSearch for this baseline.

### 2.5 `src/ingest.py` — bulk ingestion

```python
"""Ingest mod-103 embeddings into both engines.

Inputs:
  data/embeddings.npy        # shape (N, 768), float32, pre-normalized
  data/metadata.jsonl        # per-row: {id, title, body, tags}
"""
from __future__ import annotations
import json, pathlib, numpy as np
from elasticsearch import Elasticsearch, helpers as es_helpers
from opensearchpy import OpenSearch, helpers as os_helpers

EMBED_PATH = pathlib.Path("data/embeddings.npy")
META_PATH  = pathlib.Path("data/metadata.jsonl")

def load():
    vecs = np.load(EMBED_PATH).astype("float32")
    # Safety check — dot_product / innerproduct assume unit vectors.
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings must be pre-normalized"
    with META_PATH.open() as f:
        meta = [json.loads(line) for line in f]
    assert len(meta) == vecs.shape[0]
    return vecs, meta

def actions(index: str, vecs, meta, embed_key: str):
    for i, row in enumerate(meta):
        yield {
            "_index": index,
            "_id":    row["id"],
            "_source": {**row, embed_key: vecs[i].tolist()},
        }

def main():
    vecs, meta = load()

    es = Elasticsearch("http://localhost:9200")
    es_helpers.bulk(es, actions("articles_es", vecs, meta, "embedding"),
                    chunk_size=500)

    os_ = OpenSearch(hosts=[{"host": "localhost", "port": 9201}])
    os_helpers.bulk(os_, actions("articles_os_lucene", vecs, meta, "embedding"),
                    chunk_size=500)
    os_helpers.bulk(os_, actions("articles_os_faiss",  vecs, meta, "embedding"),
                    chunk_size=500)

if __name__ == "__main__":
    main()
```

The embedding field name is identical in both engines (`embedding`), so the
same source JSON works for both. The *field definition* differs by engine;
the *document* does not.

### 2.6 `src/query.py` — three query shapes on each engine

#### 2.6.1 Plain kNN

Elasticsearch:

```json
POST articles_es/_search
{
  "knn": {
    "field": "embedding",
    "query_vector": [...768 floats...],
    "k": 10,
    "num_candidates": 100
  },
  "_source": ["id", "title"]
}
```

OpenSearch (Lucene or Faiss backend):

```json
POST articles_os_lucene/_search
{
  "size": 10,
  "query": {
    "knn": {
      "embedding": {
        "vector": [...768 floats...],
        "k": 10
      }
    }
  },
  "_source": ["id", "title"]
}
```

**Diff:** Elasticsearch hoists `knn` to a top-level clause with explicit
`num_candidates` (the HNSW shortlist before filter/sort); OpenSearch treats
`knn` as a query clause. In OpenSearch's Lucene engine the equivalent of
`num_candidates` is passed through the plugin's `ef_search` request
parameter or via the engine's method parameters.

#### 2.6.2 Filtered kNN — `tenant_id` or `tags` restricted

Elasticsearch:

```json
POST articles_es/_search
{
  "knn": {
    "field": "embedding",
    "query_vector": [...],
    "k": 10,
    "num_candidates": 100,
    "filter": {
      "bool": {
        "must": [
          { "term":  { "tags":  "kafka" } },
          { "range": { "created_at": { "gte": "2024-01-01" } } }
        ]
      }
    }
  }
}
```

OpenSearch (Lucene engine, which supports the same filter surface):

```json
POST articles_os_lucene/_search
{
  "size": 10,
  "query": {
    "knn": {
      "embedding": {
        "vector": [...],
        "k": 10,
        "filter": {
          "bool": {
            "must": [
              { "term":  { "tags":  "kafka" } },
              { "range": { "created_at": { "gte": "2024-01-01" } } }
            ]
          }
        }
      }
    }
  }
}
```

**Important semantics diff.** Both engines support filtering, but *when* the
filter runs differs by engine and backend:

- Elasticsearch `knn` with `filter`: filter is applied **during** HNSW
  traversal (Lucene's "filtered HNSW" path). A very selective filter can
  degrade recall if the shortlist runs out — raise `num_candidates`.
- OpenSearch Lucene engine `knn` with `filter`: same filtered-HNSW path; same
  caveat.
- OpenSearch Faiss engine `knn` with `filter`: filtering is applied as a
  post-filter on the ANN result set in older versions; newer versions have
  moved toward integration. Verify against the version in use before
  depending on in-traversal filter behavior.

Exercise-04 of mod-104 (filter correctness drill) covers the pre- vs.
post-filter failure modes in depth; that exercise is prerequisite reading.

#### 2.6.3 Hybrid: BM25 + kNN

Elasticsearch supports a top-level `query` **and** a top-level `knn`; by
default it uses **Reciprocal Rank Fusion (RRF)** when `rank: { rrf: {} }` is
set (in newer releases) or a sum of scores with weights when not. The
simplest portable form:

```json
POST articles_es/_search
{
  "size": 10,
  "query": {
    "match": { "body": "consumer rebalance" }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [...],
    "k": 50,
    "num_candidates": 200,
    "boost": 0.6
  }
}
```

OpenSearch's canonical hybrid path is the **Hybrid query** via a search
pipeline that applies `normalization-processor` to combine BM25 and kNN
scores on comparable scales:

```json
POST _search/pipeline/hybrid_norm_rrf
{
  "description": "normalize and rrf-combine lexical + knn",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination": { "technique": "arithmetic_mean" }
      }
    }
  ]
}

POST articles_os_lucene/_search?search_pipeline=hybrid_norm_rrf
{
  "size": 10,
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "body": "consumer rebalance" } },
        { "knn":   { "embedding": { "vector": [...], "k": 50 } } }
      ]
    }
  }
}
```

OpenSearch exposes `hybrid` as a first-class query type requiring a search
pipeline that specifies how to normalize and combine the two score lists.
Elasticsearch's path is closer to a `knn` clause with `boost` plus an
optional `rank.rrf` combiner. mod-106 covers fusion in detail; here, the
teaching point is that the *shapes of the hybrid query differ* across the
two engines even though both can do the same thing.

### 2.7 `notes/comparison.md` — where the engines agree and disagree

| Dimension | Elasticsearch | OpenSearch |
|---|---|---|
| Field type name | `dense_vector` | `knn_vector` |
| Dimension key | `dims` | `dimension` |
| Enable ANN | `index: true` per field | `index.knn: true` (index-level) + `method` (field-level) |
| Metric keywords | `cosine`, `dot_product`, `l2_norm`, `max_inner_product` | `cosinesimil`, `innerproduct`, `l2`, `linf` |
| Underlying ANN library | Lucene HNSW only | Lucene HNSW, nmslib HNSW, Faiss (HNSW/IVF/PQ) |
| Query shape | `knn` is a sibling of `query` at the top level | `knn` is a query type nested inside `query` |
| "Shortlist size" knob | `num_candidates` per request | `ef_search` per request (method/plugin parameter) |
| Filter + kNN | In-traversal filter (Lucene engine) | In-traversal (Lucene engine); engine-version-dependent (Faiss) |
| Hybrid combiner | `knn.boost`, `rank.rrf`, linear sum | `hybrid` query + search pipeline with normalization-processor |
| LTR integration | Elasticsearch LTR plugin (`sltr` query) | OpenSearch LTR plugin (community port) |

Three "gotchas" worth writing down:

1. **Pre-normalize once, choose the metric once.** If the exercise's embedding
   is unit-length, choose `dot_product` on Elasticsearch and `innerproduct` on
   OpenSearch. Do not oscillate between `cosine` and `dot_product` across
   indices — each index's distances will shift and ranked lists will diverge.

2. **`k` and `num_candidates`/`ef_search` are not the same number.** `k` is
   how many results are returned. `num_candidates` (ES) and `ef_search` (OS)
   bound the HNSW visit set. Setting them equal degrades recall sharply.

3. **Filter semantics are the hazardous part of the port.** Elasticsearch's
   filtered HNSW can produce recall <1.0 if the filter is extremely selective
   and the shortlist exhausts. OpenSearch Faiss's post-filter can return
   *fewer than `k` results* if post-filtering removes too many. Both are
   expected behavior; both show up as "missing results" in a dashboard that
   doesn't log the shortlist size.

## 3. Validation steps

1. Create all three indices successfully (`articles_es`, `articles_os_lucene`,
   `articles_os_faiss`). `PUT /index` returns `"acknowledged": true`.
2. Ingest the same embedding set. Document counts must match across indices:
   `GET articles_*/_count` returns identical counts.
3. Run plain kNN on each index with the same query vector and `k=10`. The
   two Lucene-backed indices (Elasticsearch and OpenSearch) should return
   substantially overlapping top-10 lists (same algorithm, same metric). The
   Faiss-backed index may diverge slightly because of implementation
   differences in HNSW traversal and vector quantization paths.
4. Run filtered kNN with a tag filter that keeps ~1% of the corpus. Check
   that the result count is `min(k, matching_docs)` on each engine. Record
   where Faiss's post-filter returns fewer than `k`.
5. Run hybrid BM25+kNN on both engines and confirm the top-10 is a *merger*
   of what plain BM25 and plain kNN returned separately — not identical to
   either.
6. Use `_explain` on a hybrid result in Elasticsearch to see each clause's
   contribution; use the search-pipeline processor output in OpenSearch to
   see the normalized-and-combined score.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Field type | Elasticsearch uses `dense_vector` with `index: true`; OpenSearch uses `knn_vector` with `method` set. Both have the correct dimension. |
| Metric correctness | The chosen similarity matches the embedding: pre-normalized → `dot_product` (ES) / `innerproduct` (OS); un-normalized cosine → `cosine` (ES) / `cosinesimil` (OS). |
| HNSW parameters | `m` and `ef_construction` are set explicitly and defended at a one-line level. Shortlist size (`num_candidates` / `ef_search`) is named in the query, not left at default. |
| Both engines running | Three indices (ES + OS Lucene + OS Faiss) ingested from the *same* embedding file. Not three different corpora. |
| Query translations | Each query form (plain kNN, filtered kNN, hybrid) is shown in both engines' idioms. The learner does not use ES syntax against OS or vice versa. |
| Filter semantics | The write-up names in-traversal vs. post-filter and attributes each to the correct engine/backend. |
| Hybrid combiner | The learner explains the ES vs. OS hybrid paths and does not claim one is "just like the other." |
| Faiss delta | The OpenSearch Faiss mapping is included **and** the learner explains what Faiss unlocks (nmslib/HNSW was the historical option; Faiss adds IVF/PQ for larger-than-RAM corpora). |
| Reproducibility | Ingest script loads the same `.npy` file into both engines. The embedding dimension and norm are asserted at load time. |

Fail-fast smells:

- Using `knn_vector` with no `method`. OpenSearch will take it, but no ANN
  index is built — queries fall back to brute force.
- Using `cosine` on pre-normalized vectors and `dot_product` on the same
  vectors in two different indices. The *scores* will differ in magnitude
  and the ranked list *can* differ because floating-point rounding is not
  identical between the two code paths.
- Treating the hybrid query shapes as interchangeable. They are not.
- Writing `knn` under `query` on Elasticsearch (as of this writing, that
  path is deprecated in favor of top-level `knn`).

## 5. Common mistakes

- **Not pre-normalizing and then picking `dot_product`.** `dot_product`
  requires unit vectors; the engine will refuse non-normalized input. This
  shows up as `illegal_argument_exception` at index time if the learner is
  lucky, and as silently wrong scores if the engine does not validate.
- **Confusing `dense_vector.dims` with `knn_vector.dimension`.** Mapping
  errors here are loud (HTTP 400). But a *copied* mapping from one engine to
  the other that passes validation because the field type is permissive is
  worse — it silently uses the wrong size.
- **Setting `num_candidates` equal to `k`.** The HNSW shortlist is the
  mechanism that trades latency for recall. Setting them equal removes
  recall.
- **Expecting Elasticsearch to have Faiss.** Faiss is only an option on
  OpenSearch's k-NN plugin. If an exercise plan calls for `IVF`+`PQ` on
  Elasticsearch, the plan is wrong — the option does not exist.
- **Shipping hybrid without a normalization strategy.** BM25 scores are
  unbounded; `dot_product` cosine scores are in [-1, 1]; combining them
  without normalization amounts to "the BM25 side always wins on long docs."
  Both engines provide normalization machinery; use it.
- **Reindexing to change the metric.** Metric is baked into the field
  definition. Changing it is a reindex. Choose once, pre-normalize, don't
  revisit without a reason.
- **Forgetting that `index.knn: true` is index-level in OpenSearch.** If
  the setting is missed, the `knn_vector` field exists but no HNSW graph is
  built — queries silently run as brute force and latency is 100× what the
  learner expects.
- **Benchmarking an un-warm JVM.** HNSW graphs are mmapped; first-query
  latency is dominated by page faults. Run a warmup pass before trusting the
  p95. (The deeper benchmark methodology lives in mod-104 exercise-02; this
  exercise just notes the pitfall.)

## 6. References

- Lucene HNSW backing (shared by both engines):
  - Lucene `HnswGraph` — [lucene.apache.org/core/9_10_0/core/org/apache/lucene/util/hnsw/HnswGraph.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/util/hnsw/HnswGraph.html).
  - Malkov & Yashunin (2016), HNSW original paper — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320).
- Elasticsearch dense vectors:
  - `dense_vector` field — [elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html).
  - kNN search — [elastic.co/guide/en/elasticsearch/reference/current/knn-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html).
  - Semantic search guide — [elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html).
- OpenSearch k-NN:
  - k-NN plugin overview — [opensearch.org/docs/latest/search-plugins/knn/](https://opensearch.org/docs/latest/search-plugins/knn/).
  - `knn_vector` field type — [opensearch.org/docs/latest/field-types/supported-field-types/knn-vector/](https://opensearch.org/docs/latest/field-types/supported-field-types/knn-vector/).
  - Approximate k-NN with Faiss / nmslib / Lucene — [opensearch.org/docs/latest/search-plugins/knn/approximate-knn/](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/).
  - Hybrid query and search pipeline — [opensearch.org/docs/latest/search-plugins/hybrid-search/](https://opensearch.org/docs/latest/search-plugins/hybrid-search/).
- Faiss (OpenSearch k-NN backend):
  - FAISS project — [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
