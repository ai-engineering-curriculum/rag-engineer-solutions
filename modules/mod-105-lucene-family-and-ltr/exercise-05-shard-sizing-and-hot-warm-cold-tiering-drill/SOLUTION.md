# Exercise 05 — Solution: Shard Sizing and Hot-Warm-Cold Tiering Drill

This is a reference solution for
`exercise-05-shard-sizing-and-hot-warm-cold-tiering-drill.md`. It is a
**capacity-planning** exercise. The learner sizes shards for the mod-105
corpus, writes an Index Lifecycle Management (ILM / ISM) policy that moves
indices through hot → warm → cold tiers, defines a search-tier → data-tier
routing strategy, and defends the numbers against the official guidance from
Elastic and OpenSearch.

The teaching point is that a shard is both a *unit of parallelism* and a
*unit of overhead*. Too few shards and one node is the bottleneck; too many
and the cluster metadata overhead dominates. Tiering adds a second axis:
hot shards need fast storage and high CPU, cold shards need only cheap
storage. Getting both right is the difference between a cluster that costs
$X/month and one that costs 10X.

## 1. Solution overview

The deliverable has five parts:

1. **`notes/shard-sizing.md`** — the sizing worksheet. Starts from the
   corpus parameters (document count, average size, retention, QPS) and
   derives primary shard count, replica count, and the shard target size.
   Compares the derivation to Elastic's guidance (≤ 50 GB per shard; ≤ 20
   shards per GB of heap).
2. **`policies/ilm-logs-90d.json`** — an Index Lifecycle Management policy
   for a logs-shaped index that rolls over at 50 GB or 7 days, moves to
   warm at 7 days, cold at 30 days, and deletes at 90 days.
3. **`policies/ism-articles-perpetual.json`** — an OpenSearch Index State
   Management policy for the mod-105 articles index. Articles are not
   log-shaped; they have the opposite pattern (small corpus, searched
   continuously), so the policy focuses on forcemerge and replica count
   rather than rollover.
4. **`templates/articles-template.json`** — an index template that pins
   `index.routing.allocation.include._tier_preference` to
   `data_hot,data_warm,data_cold` so new indices get the right tier
   automatically.
5. **`notes/retrieval-impact.md`** — a short note on how tiering affects
   first-query latency in a RAG pipeline (cold tier wake-ups) and how to
   keep the retrieval path warm.

The exercise is scoped to *defensible* numbers, not optimal ones. Any
two reviewers starting from the same corpus parameters should land within
a factor of two of each other and should be able to articulate every
assumption they made.

## 2. Worked answer or implementation

### 2.1 Corpus parameters

Make the assumptions explicit before deriving anything. Change the numbers
for the learner's actual corpus; the shape of the arithmetic stays the same.

| Parameter | Value for the worked example |
|---|---|
| Logical documents (passages after mod-102 chunking) | 10,000,000 |
| Average source document size (serialized JSON) | ~8 KB |
| Average passage size (after chunking) | ~1.5 KB |
| Dense embedding dimension | 768 |
| Embedding precision | float32 (4 bytes/dim) = 3 KB/doc |
| Expected steady-state QPS (search) | 50 |
| Expected indexing throughput at steady state | ~50 docs/s (batchable) |
| Retention (articles) | perpetual (no delete) |
| Retention (operational logs) | 90 days (separate index pattern) |
| Replica count | 1 (cost baseline) |
| Node heap | 30 GB per data node (JVM safe ceiling ~31 GB) |

**Derived per-doc storage.** A passage with its embedding is roughly

```
raw text        ~1.5 KB
inverted index  ~0.5 KB (depends on analyzer, stopwords, stored fields)
embedding float32  3.0 KB
HNSW overhead   ~0.5 KB (per-vector graph edges, dependent on `m`)
doc values etc. ~0.3 KB
--------------------------------
total           ~6 KB per doc
```

At 10M docs: ~60 GB primary. With one replica: 120 GB cluster-wide storage.

### 2.2 `notes/shard-sizing.md` — primary shard count

Elastic's guidance, stable across recent versions:

- Aim for shard size between a few GB and **≤ 50 GB** for search workloads.
- Keep shard count per GB of heap **≤ 20** (ideally ≤ 10 for comfortably
  below the ceiling).
- Smaller shards: more parallelism per query, but more cluster metadata
  and more per-shard overhead.
- Larger shards: less metadata, but less parallelism and slower recovery.

For 60 GB of primary data and a 50 GB ceiling: **2 primary shards** is
the floor; **3 primary shards** (20 GB each) is the middle; **6 primary
shards** (10 GB each) is the upper practical bound for this size.

Given 50 QPS and the mod-105 "a single node is fine" scale, **3 primary
shards × 1 replica** is the recommended setting. That yields:

- 6 shards total (3 primary + 3 replica) on at least 2 data nodes.
- Each query fans out to 3 shards per replica group; parallelism is modest
  but adequate for 50 QPS.
- Shard size of ~20 GB leaves plenty of headroom under the 50 GB ceiling
  as the corpus grows.

Per-node shard budget check:

- Heap 30 GB → soft ceiling 300 shards per node (at ≤ 10 shards per GB of
  heap). 6 shards is well under.
- Elasticsearch/OpenSearch defaults (`cluster.max_shards_per_node = 1000`)
  are intentionally lax; the operational ceiling is lower.

**Replica count.** 1 replica is the minimum for availability and is the
right baseline. Scale replicas to serve read throughput:

- ≥ 100 QPS steady state → consider 2 replicas.
- < 20 QPS → 1 replica is fine.

### 2.3 Shard count for logs (90-day retention)

Logs are a time-series workload with a very different shape:

- Write-heavy (typically 10×–1000× higher than search QPS).
- Oldest data searched rarely, newest data searched constantly.
- Deletion by index (drop the oldest index), not by document.

Use **rollover** rather than a fixed shard count. The policy rolls over
when:

- Primary shard size reaches **50 GB**, or
- Age reaches **7 days** (pick a time cap so slow days don't accumulate
  into stale indices).

A typical settings file for the logs template:

```json
{
  "index": {
    "number_of_shards":   1,
    "number_of_replicas": 1,
    "lifecycle": {
      "name": "logs-90d",
      "rollover_alias": "logs"
    }
  }
}
```

One primary shard per index is normal for rolled-over time-series indices.
Parallelism comes from querying *across* multiple rolled-over indices at
once, not from splitting a single index.

### 2.4 `policies/ilm-logs-90d.json` — Elasticsearch ILM policy

```json
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_primary_shard_size": "50gb",
            "max_age": "7d"
          },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "allocate": { "include": { "_tier_preference": "data_warm" } },
          "forcemerge": { "max_num_segments": 1 },
          "set_priority": { "priority": 50 }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "allocate": { "include": { "_tier_preference": "data_cold" } },
          "freeze": {},
          "set_priority": { "priority": 10 }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": { "delete": {} }
      }
    }
  }
}
```

**Choices and why:**

- **Hot:** write-active, replicated (1 replica), on fast SSD-backed nodes
  (`data_hot` tier). Rollover is 50 GB or 7 days, whichever first.
- **Warm:** 7–30 days. Forcemerged to 1 segment per shard (improves search
  latency; irreversible — never forcemerge a hot index receiving writes).
  Lower priority, moved to warm nodes which are usually cheaper per TB.
- **Cold:** 30–90 days. On cold-tier nodes (large HDD or snapshot-backed).
  Frozen indices use the searchable-snapshot restore-on-read path; queries
  are slower but storage is 10× cheaper.
- **Delete:** 90 days. The log's retention promise, evaluated at index
  level — the whole rolled-over index drops in one atomic operation.

### 2.5 `policies/ism-articles-perpetual.json` — OpenSearch ISM for articles

OpenSearch calls the same mechanism Index State Management (ISM). The
article index is **not** rolled over — it is a single logical index
updated incrementally. The policy's job is to force-merge after bulk
loads and keep the replica count sensible.

```json
{
  "policy": {
    "description": "mod-105 articles: no rollover, periodic merge, replicas scale with traffic",
    "default_state": "active",
    "states": [
      {
        "name": "active",
        "actions": [
          {
            "retry": { "count": 3, "backoff": "exponential", "delay": "1m" },
            "replica_count": { "number_of_replicas": 1 }
          }
        ],
        "transitions": [
          { "state_name": "optimized",
            "conditions": { "cron": { "cron_expression": "0 2 * * *",
                                       "timezone": "UTC" } } }
        ]
      },
      {
        "name": "optimized",
        "actions": [
          { "force_merge": { "max_num_segments": 5 } }
        ],
        "transitions": [
          { "state_name": "active", "conditions": {} }
        ]
      }
    ]
  }
}
```

**Choices and why:**

- `force_merge` to 5 segments (not 1) because the index is still being
  written. Merging to 1 segment on a live write index wastes IO and
  immediately produces new segments. 5 is a compromise: fewer than Lucene
  merge policy's natural equilibrium, enough to keep segment-level
  overhead low.
- Replica count stays at 1 through the active state. If QPS grows, bump
  the replica count manually or add a condition-based transition.

### 2.6 `templates/articles-template.json` — tier pinning

```json
{
  "index_patterns": ["articles-*"],
  "template": {
    "settings": {
      "index": {
        "number_of_shards":   3,
        "number_of_replicas": 1,
        "routing": {
          "allocation": {
            "include": { "_tier_preference": "data_hot,data_warm" }
          }
        }
      }
    }
  },
  "priority": 100
}
```

For the articles corpus, `data_hot` for the active shards; `data_warm`
only if a snapshot restore places the index on a warm node. The articles
index is never cold.

### 2.7 `notes/retrieval-impact.md` — what tiering costs at query time

Three facts a RAG engineer should internalize:

1. **Cold-tier queries wake up a searchable snapshot.** First query after
   an idle period can take seconds, not milliseconds. If a RAG system
   serves user queries from a cold index, the user sees the cold-start.
2. **Hot/warm/cold split works for logs, not for lookup indices.** Logs
   have a monotonic recency bias; articles don't. Do not put the main
   articles index on a warm tier to "save money" unless the retrieval
   path tolerates the slower storage.
3. **Forcemerge is not free.** Merging to one segment is an exclusive
   operation on the shard; it uses a lot of IO and CPU. Schedule it off
   the write path. For a mod-105 articles index that is updated
   continuously, merge to 5 segments, not 1.

Mitigations for RAG latency on tiered storage:

- **Hot cache.** Pin the top-K hot queries' documents in OS page cache by
  running a scheduled warmer.
- **Searchable snapshots with partial restore.** Keep the hot-shard HNSW
  graph in memory even when the raw doc values are on object storage —
  trade storage cost for query latency.
- **Dedicated search tier.** Elastic Cloud and OpenSearch Dashboards
  separate ingest tier (`data_content` / `data_hot`) from search tier.
  For a pure RAG workload with heavy reads, the search tier is where
  replicas live.

## 3. Validation steps

1. The learner's shard sizing arithmetic matches the Elastic / OpenSearch
   guidance: shard size ≤ 50 GB, shard count per GB of heap ≤ 20 (soft);
   the cluster has at least two data nodes so a replica can exist on a
   different node.
2. `PUT _ilm/policy/logs-90d` succeeds and `GET _ilm/policy/logs-90d`
   returns the policy unchanged. ISM equivalent for OpenSearch.
3. Create a test index with `logs-000001` and the rollover alias `logs`;
   push enough data to trigger rollover; `GET logs/_ilm/explain` (or the
   OpenSearch ISM explain API) shows the phase transition.
4. The articles index template applies to a new `articles-2026-11`
   index — `GET articles-2026-11/_settings` shows the template-provided
   shard count and tier preference.
5. Verify that forcemerge on the articles index completes without
   impacting write throughput materially during a 10-minute load test
   (or the learner notes that forcemerge is scheduled off-peak).
6. On a fresh cluster with `data_hot` and `data_warm` tiers, issue a
   query to a warmed hot index and a query to a cold-tier index after a
   restart; record the first-query latency difference. Write it down so
   mod-107 (reranking) can budget around it.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Assumptions written | Corpus size, average doc size, retention, QPS, replica policy, and node heap are explicit numbers at the top of the sizing note. Reviewing the solution does not require guessing inputs. |
| Shard size target | Per-shard size is within the ≤ 50 GB guidance and above ~1 GB (very small shards waste metadata). |
| Shard count rationale | The number of primary shards is derived from (corpus size / target shard size), not picked out of a hat. |
| Replicas | Replica count is set as a function of availability (minimum 1) and read throughput. The learner can name the QPS at which they would add a second replica. |
| Rollover shape (logs) | Logs index template uses rollover with both a size cap (≤ 50 GB) and an age cap (≤ a week). Not one or the other. |
| ILM/ISM phases | Hot → warm → cold → delete transitions are named, with `min_age` values in days, and each transition has a reason (allocate, forcemerge, freeze, delete). |
| Forcemerge discipline | `max_num_segments: 1` is scheduled only on indices no longer receiving writes (warm or later). The articles index policy uses a larger `max_num_segments` to avoid contention with ongoing writes. |
| Tier allocation | Index templates pin `_tier_preference` explicitly. Reliance on cluster-wide defaults is called out as a known gap, not an oversight. |
| RAG latency note | The solution acknowledges that cold-tier storage adds first-query latency and explains how the retrieval path keeps the hot index warm. |
| Cost tradeoff | The policy describes the cost shape: hot (SSD) more expensive, cold (snapshot) cheaper. The exercise is not implicitly asking for the fastest or the cheapest; it is asking for the right curve. |

Fail-fast smells:

- "Use the default 5 primary shards." The default dropped to 1 in
  Elasticsearch 7.0+. Learners quoting "5 shards" are usually recalling
  stale defaults.
- Forcemerge on a hot index. This is a classic mistake — it saturates
  the write node and produces new segments immediately.
- Deleting data by document on a time-series index. The idiomatic
  shape is to delete the *index*, not individual documents.
- Mixing article and log workloads under one ILM policy. They have
  opposite shapes — one is perpetual with slow updates, the other is
  write-heavy with drop-oldest. Two policies, two templates.

## 5. Common mistakes

- **Oversharding a small corpus.** 10M documents across 30 primary shards
  is seductive ("parallelism!") and is wrong — each shard is now ~2 GB,
  per-query coordination cost dominates, and the heap pays for metadata
  overhead on every node. Shard count is a last resort; replica count
  scales reads at far lower overhead.
- **Underfilling shards.** The inverse: 2 shards of 500 GB each because
  "it's one big index." Recovery of a 500 GB shard takes hours and
  blocks relocation. Keep shard size closer to 20–40 GB in production.
- **Replica count = 0 "to save money."** Zero replicas means a single
  node loss drops the index. Replica count 1 is the floor for anything
  resembling production.
- **Using `max_age` without a size cap on rollover.** A quiet day or a
  Kafka-outage day produces a tiny index; a busy day produces a shard
  past 50 GB. Always set both `max_primary_shard_size` *and* `max_age`.
- **Freezing indices that are still queried.** Frozen / cold indices are
  cheap to store and expensive to serve. If the RAG retrieval path
  touches the frozen index, every query pays a wake-up cost. Freeze
  only indices that are rarely read.
- **Treating the cold tier as free.** Searchable snapshots are cheaper
  per GB *stored* but more expensive per GB *searched* — the cost moves
  from storage to compute at query time. Model the change in the right
  column.
- **Forgetting the embedding vectors.** The storage worksheet is often
  calibrated on the raw text size, not the dense vector overhead. A
  768-dim float32 vector is 3 KB per doc; at 10M docs that's 30 GB of
  pure vector storage before HNSW. Halve by using `halfvec` / fp16 if
  the engine supports it.
- **Not testing the policy on a disposable index first.** ILM / ISM
  transitions have subtle conditions (`min_age` is measured from
  rollover, not from index creation; `delete` fires regardless of what
  else is scheduled). Smoke-test the policy on `test-000001` before
  pointing it at `logs-000001`.
- **Running one data node and calling it a cluster.** A single-node
  cluster cannot host a primary and a replica — they would be on the
  same node. The replica stays unassigned, the cluster is yellow, and
  loss of the node is loss of the data.

## 6. References

- Elasticsearch sizing and lifecycle:
  - Size your shards — [elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/size-your-shards.html).
  - Index Lifecycle Management overview — [elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html).
  - Data tiers — [elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html).
  - Rollover API — [elastic.co/guide/en/elasticsearch/reference/current/indices-rollover-index.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-rollover-index.html).
  - Forcemerge — [elastic.co/guide/en/elasticsearch/reference/current/indices-forcemerge.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-forcemerge.html).
  - Searchable snapshots — [elastic.co/guide/en/elasticsearch/reference/current/searchable-snapshots.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/searchable-snapshots.html).
- OpenSearch sizing and lifecycle:
  - Index State Management — [opensearch.org/docs/latest/im-plugin/ism/index/](https://opensearch.org/docs/latest/im-plugin/ism/index/).
  - Sizing recommendations — [opensearch.org/docs/latest/tuning-your-cluster/cluster/](https://opensearch.org/docs/latest/tuning-your-cluster/cluster/).
  - Data tiers and node roles — [opensearch.org/docs/latest/tuning-your-cluster/cluster/](https://opensearch.org/docs/latest/tuning-your-cluster/cluster/).
- Lucene segment model (background for the forcemerge discussion):
  - Lucene index file formats — [lucene.apache.org/core/9_10_0/core/org/apache/lucene/codecs/lucene99/package-summary.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/codecs/lucene99/package-summary.html).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
