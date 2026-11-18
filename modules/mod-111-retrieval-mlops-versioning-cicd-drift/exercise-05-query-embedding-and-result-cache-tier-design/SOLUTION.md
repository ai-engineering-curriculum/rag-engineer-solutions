# Exercise 05 — Solution: Query-Embedding and Result Cache Tier Design

This is a reference solution for
`exercise-05-query-embedding-and-result-cache-tier-design.md`. It is
a **design + implementation** exercise. The learner designs a
two-layer cache (query → embedding; query → retrieval result) in
front of the retriever, writes the keying/invalidation contract,
and implements a reference Redis-backed version with correct TTL,
versioning, and metrics.

The teaching point: *the query embedding and the retrieval result are
cacheable along different lifetimes*. The query embedding is pure in
the embedding model and the query text — invalidate it when the
model changes. The retrieval result is impure in the index (a
different build id returns different top-k) — invalidate it on
every cutover. Collapsing both into a single cache key (as "just
cache the top-k by query text") is the mistake that makes the cache
serve wrong answers after a cutover, and makes it miss after every
minor query reword.

## 1. Solution overview

The deliverable has four parts:

1. **`cache/spec.md`** — the keying contract, TTL table, and
   invalidation triggers for both cache layers.
2. **`cache/impl.py`** — a reference Redis-backed implementation
   with explicit `get_or_compute` for each layer, structured
   metrics (hits, misses, stale-served), and a background
   warm-up hook.
3. **`cache/hit_rate_model.md`** — a back-of-the-envelope model
   for what hit rate is plausible given the query-rate curve, and
   the `$/month` reduction in the exercise-04 cost model.
4. **`cache/review_memo.md`** — the one-page decision memo
   template: "here is the cache topology we picked, here is the
   measured hit rate on a 7-day replay, here are the
   invalidation triggers."

Scope: *the cache tier only*. Measurement of quality impact
(cache cannot degrade recall if invalidation is correct) uses
exercise-02; cost accounting uses exercise-04; index versioning
uses exercise-01.

## 2. Worked answer or implementation

### 2.1 Two layers, two keying schemes

**Layer 1 — Query-embedding cache (`qemb`)**

- Key: `qemb:{embedding_model}:{model_revision}:sha256(normalized(query_text))`.
- Value: the embedding vector (binary little-endian floats or
  int8s, length `dim`). Add a small header: `{dim, dtype,
  model, revision}`.
- Lifetime: **long**. The embedding of `"reset password"` under
  `text-embedding-3-large@20240125` is the same today and in six
  months. TTL exists only to put a bound on storage growth, not to
  guarantee freshness.
- Invalidation: on embedding-model swap. Done cheaply by
  *changing the key prefix* (bump `model_revision`) — old
  entries age out via TTL; zero-downtime cutover.

**Layer 2 — Retrieval-result cache (`result`)**

- Key: `result:{build_id}:{filter_sha}:sha256(normalized(query_text))`.
- Value: a list of `{doc_id, score, rank, metadata}` tuples for
  top-K. Include a header: `{build_id, k, filter_sha, retrieved_at,
  retrieval_params_sha}`.
- Lifetime: **short-to-medium**. The top-k is sensitive to the
  index state (exercise-01), the filter the user supplied, and
  the reranker weights. TTL bounds *inconsistency* after
  incremental ingest or hot-reranker tunes.
- Invalidation: by prefix on every alias flip (exercise-01). The
  `build_id` is in the key, so the old-build results stay
  resident but will *never* be hit under the new alias. They age
  out via TTL.

Why the two layers must not collapse:

- Collapsing to one key means a cutover has to purge the entire
  cache. A full purge loses the query-embedding hits too, which
  are the cheapest to earn.
- Collapsing means every reword ("reset my password" vs "reset
  password") misses both layers. Split: the `qemb` cache misses
  on text-normalization edge cases, but the `result` cache can
  still hit on the embedding-space neighborhood (via the second
  layer's semantic key — see §2.5 for a semantic-cache
  variant).

### 2.2 Keying details

**Query normalization (both layers).**

Normalization is the strongest hit-rate lever and the strongest
correctness footgun. The contract:

```python
def normalize(q: str) -> str:
    """Deterministic, lossless-enough normalization for caching."""
    import unicodedata, re
    q = unicodedata.normalize("NFKC", q)
    q = q.strip()
    q = re.sub(r"\s+", " ", q)           # collapse whitespace
    q = q.casefold()                      # unicode-aware lowercasing
    return q
```

What the normalization does **not** do:

- No stemming / lemmatization / stopword removal. These are the
  retriever's business, not the cache's. Stemming at cache time
  breaks the contract that "same cache key → same user-visible
  answer."
- No punctuation stripping. `"Q4?"` and `"Q4."` are different
  questions; the cache should respect that.
- No language-specific heuristics. The same normalization runs for
  every language. Language-aware normalization belongs inside the
  retriever.

**Model-revision and build-id in the prefix (not the suffix).**

The two layer prefixes are:

```
qemb:openai-text-embedding-3-large:20240125:<sha256(normalized(q))>
result:docs-support-prod_corpus-7f3a..._...:<filter_sha>:<sha256(normalized(q))>
```

Two reasons for prefix placement:

1. **Prefix scan.** Redis `SCAN` with a prefix lets you count,
   inspect, and (if needed) delete all keys for a given model or
   build id without rebuilding the key.
2. **Cutover purge pattern.** `DEL` by prefix is slow; the
   intended pattern is to let the old-prefix keys age out via TTL
   while writes go only to the new prefix. Prefix placement
   makes that obvious at a glance.

**Filter sha.** The `result` cache has one more field in the key:
a hash of the request filter (`{customer_id, product_line,
language, date_from, date_to, acl_subject_sha}`). Two users with
the same text but different customer scopes must *never* share a
result cache entry. ACL leaks by cache key are a real incident
class (MITRE ATLAS **AML.T0024 Exfiltration via ML Inference API**;
NIST AI RMF **MANAGE 4.1** calls this out under "unintended
access").

### 2.3 TTL table

| Layer | TTL | Rationale |
|---|---|---|
| `qemb` (embedding) | 30 days | Purely a storage bound; values are correct until model revision changes. |
| `result` (dynamic corpus, batch ingest hourly) | 10 minutes | Shorter than the ingest cadence; the longest window during which a stale result can be served. |
| `result` (quasi-static corpus, batch ingest nightly) | 2 hours | Longer TTL; still bounded by the ingest window. |
| `result` (static reference corpus, never ingests) | 24 hours | Longer still; the only invalidation path is a cutover. |

Rule: **TTL ≤ ingest cadence.** A TTL longer than the ingest
cadence means the cache will serve answers that miss newly
ingested documents. If the product requires "answers reflect the
corpus within 1 minute of a doc landing," the TTL is 1 minute and
the cache is nearly useless — use a real-time invalidation feed
(§2.6).

### 2.4 `cache/impl.py` — a reference Redis-backed implementation

```python
"""Two-layer cache with explicit keying, metrics, and invalidation.

Dependencies: redis-py async client, prometheus_client.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import time
import unicodedata
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

import redis.asyncio as redis
from prometheus_client import Counter, Histogram


# ---- Metrics (Prometheus) --------------------------------------------------

CACHE_LOOKUPS = Counter(
    "retrieval_cache_lookups_total",
    "Cache lookups",
    ["layer", "outcome"],   # qemb|result  x  hit|miss|stale
)
CACHE_LATENCY = Histogram(
    "retrieval_cache_lookup_seconds",
    "Cache lookup wall-clock",
    ["layer"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050),
)


# ---- Normalization + keying ------------------------------------------------

def _normalize(q: str) -> str:
    import re
    q = unicodedata.normalize("NFKC", q)
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q.casefold()


def _qhash(q: str) -> str:
    return hashlib.sha256(_normalize(q).encode("utf-8")).hexdigest()


def _filter_hash(f: dict | None) -> str:
    body = json.dumps(f or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:16]


def qemb_key(model: str, revision: str, q: str) -> str:
    return f"qemb:{model}:{revision}:{_qhash(q)}"


def result_key(build_id: str, filters: dict | None, q: str, k: int) -> str:
    return f"result:{build_id}:{_filter_hash(filters)}:{k}:{_qhash(q)}"


# ---- Binary codec for vectors ----------------------------------------------

def _encode_vec(v: list[float]) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def _decode_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ---- The cache -------------------------------------------------------------

@dataclass
class CacheCfg:
    qemb_ttl_s:   int = 30 * 24 * 3600   # 30 days
    result_ttl_s: int = 10 * 60          # 10 minutes (override per-deployment)
    allow_stale_on_miss_window_s: int = 0  # 0 = never serve stale


class RetrievalCache:
    def __init__(self, client: redis.Redis, cfg: CacheCfg = CacheCfg()) -> None:
        self.r   = client
        self.cfg = cfg

    # ---- Layer 1: query embedding -----------------------------------------

    async def get_or_compute_qemb(
        self,
        q: str,
        model: str,
        revision: str,
        compute: Callable[[str], Awaitable[list[float]]],
    ) -> list[float]:
        key = qemb_key(model, revision, q)
        t0 = time.perf_counter()
        blob = await self.r.get(key)
        CACHE_LATENCY.labels("qemb").observe(time.perf_counter() - t0)
        if blob is not None:
            CACHE_LOOKUPS.labels("qemb", "hit").inc()
            return _decode_vec(blob)
        CACHE_LOOKUPS.labels("qemb", "miss").inc()
        vec = await compute(q)
        await self.r.set(key, _encode_vec(vec), ex=self.cfg.qemb_ttl_s)
        return vec

    # ---- Layer 2: retrieval result ----------------------------------------

    async def get_or_compute_result(
        self,
        q: str,
        build_id: str,
        k: int,
        filters: dict | None,
        compute: Callable[[], Awaitable[list[dict]]],
    ) -> list[dict]:
        key = result_key(build_id, filters, q, k)
        t0 = time.perf_counter()
        raw = await self.r.get(key)
        CACHE_LATENCY.labels("result").observe(time.perf_counter() - t0)
        if raw is not None:
            CACHE_LOOKUPS.labels("result", "hit").inc()
            return json.loads(raw)
        CACHE_LOOKUPS.labels("result", "miss").inc()
        hits = await compute()
        payload = json.dumps(hits).encode("utf-8")
        await self.r.set(key, payload, ex=self.cfg.result_ttl_s)
        return hits

    # ---- Invalidation -----------------------------------------------------

    async def invalidate_build(self, build_id: str) -> int:
        """Prefix-scan and delete results for a given build id.

        Intended for operator-driven cutover purge *when the old
        build is retired* (exercise-01 Phase 6 cleanup). During a
        normal cutover you do not need this: the new alias writes a
        new build_id prefix, and the old entries TTL out.
        """
        pattern = f"result:{build_id}:*"
        deleted = 0
        async for key in self.r.scan_iter(match=pattern, count=1000):
            deleted += await self.r.delete(key)
        return deleted

    async def warm(self, items: Iterable[tuple[str, int, dict | None, str]]) -> None:
        """Best-effort warm: precompute both layers for a set of
        known-hot queries. Items are (query, k, filters, build_id)."""
        for q, k, filters, build_id in items:
            key = result_key(build_id, filters, q, k)
            exists = await self.r.exists(key)
            if exists:
                continue
            # The caller supplies an `embed` and `retrieve` to warm;
            # omitted here. The point is that warm does the full path
            # through the two layers with the same code used online.
```

Design notes:

- **Explicit layers.** `get_or_compute_qemb` and
  `get_or_compute_result` are the only public entry points. A caller
  cannot accidentally use a result key with a query-embedding value.
- **Binary vectors.** `qemb` values are packed floats (~6 KB for a
  1536-dim `float32`); `float16` or `int8` halves/quarters that if
  the embedding provider's precision allows.
- **Prometheus labels keep cardinality bounded.** `layer x outcome`
  produces six time series total (`qemb/result × hit/miss/stale`),
  not one per query. Per-query hit counts are not useful.
- **Stale-while-revalidate (optional).** If
  `allow_stale_on_miss_window_s > 0`, the cache keeps the old key
  with a longer TTL in a parallel namespace and, on a true miss
  during a burst, serves the stale value while triggering a
  background refresh. Not shown in the skeleton; wire in only after
  measuring that the straight path can't make p95.

### 2.5 Semantic cache: the common wrong turn

A tempting third layer: *cache by embedding neighborhood*. "If the
new query's embedding is within distance ε of an old query's
embedding, return the cached result for the old query." Libraries
sell this as "semantic caching" (e.g. GPTCache, Redis
[semantic caching docs](https://redis.io/docs/latest/develop/ai/cache/semantic-cache/)).

Why it is dangerous as the only key:

- The *answer* to "reset password" is almost always a different
  document from the answer to "my account is locked" even though
  the vectors cluster. Collapsing them under semantic ε caches a
  *similar query's answer*, not *this query's answer*. Recall
  ceiling drops.
- An ε set too tight loses most hits (semantic hashing is low
  cardinality by design). An ε set too loose serves wrong
  answers.
- For personalised retrieval (ACL), the semantic layer must
  include the filter hash in the neighborhood — at which point
  you have a per-ACL cluster, i.e., the result cache from §2.1
  with more steps.

Reasonable use: as a *shadow* layer whose hits are logged but not
served, to measure how much you could hypothetically gain. If the
log says 15% of semantic-near-hits would have returned the exact
same top-10 as the exact-match cache, maybe it is worth building.
Most teams find the number smaller than the demo claims; in that
case, skip it.

### 2.6 Freshness and invalidation on the write path

Two patterns, in order of preference:

**Pattern A — TTL + alias prefix (default).** The cache knows the
current `build_id`; after a cutover the new `build_id` is a new
key prefix; old entries age out via TTL. This is the pattern baked
into §2.1–§2.4. *No active invalidation* is needed for a cutover.
This is the big win: cutovers are lazy on the cache.

**Pattern B — Document-level invalidation (for real-time
corpora).** If the product requires "a newly added doc appears in
results within T seconds" and T < TTL, there must be an active
invalidation feed:

1. The ingest DAG (exercise-03) emits events on a message bus
   (NATS / Kafka / Pub/Sub) when a doc is added/updated/removed.
2. A cache-invalidation consumer subscribes. For each event, it
   enumerates the `result:*:*:*` keys whose top-k *could have
   contained* the affected doc. The naive way is to keep a
   reverse index `doc_id → [result_keys]` the cache populates on
   write.
3. The consumer deletes those keys.

The reverse-index approach keeps `DEL` cost proportional to the
number of affected keys, not to the cache size, but doubles cache
memory. Acceptable at small scale; considered prohibitive at
very large scale, which is why **Pattern A with a short TTL is
almost always the right answer** unless the product truly requires
sub-TTL freshness.

### 2.7 Hit-rate model

Query traffic typically follows a long-tail distribution — a
handful of queries account for most of the volume. Per Zipf's law
on access distributions in web search ("Access patterns in web
search," classic observation; see Silverstein et al., 1999, in
`SIGIR Forum`), the hit-rate curve climbs quickly at small cache
sizes and flattens:

| Cache size (unique queries kept warm) | Expected hit rate (Zipfian traffic) |
|---:|---:|
| 1,000 | 15–25% |
| 10,000 | 40–60% |
| 100,000 | 65–80% |
| 1,000,000 | 80–90% |

Caveats and reality checks:

- These numbers assume the traffic is actually Zipfian. Chatbot
  traffic often is; enterprise-search traffic sometimes isn't.
  *Measure the real distribution* before trusting the table.
- Hit rate above 90% for retrieval is suspicious unless there is a
  bot or an integration hammering the same query — if present,
  check whether the cache is masking an application-level
  retry loop.
- The **embedding cache hit rate is usually higher than the
  result cache hit rate**, because the embedding cache ignores
  filters. A 70% embedding-cache hit rate and a 30% result-cache
  hit rate is a plausible mature configuration.

Cost impact (feeds exercise-04):

```python
# 100k QPD, 20 tokens avg, text-embedding-3-large @ $0.00013/1k tok
baseline_month = 100_000 * 30 * 20 / 1000 * 0.00013   # ≈ $7.80/mo (small corpus from exercise-04 scenario B)
# At 70% qemb hit rate:
new_month = baseline_month * (1 - 0.70)                # ≈ $2.34/mo — 70% cut
```

For medium/large scenarios with higher QPD, these percentages
translate into much bigger absolute savings (see §2.3 of
exercise-04). The cache tier is cheap infrastructure (Redis is
RAM) but high-leverage.

### 2.8 Cache sizing and eviction

Redis defaults to `noeviction` — at full memory, writes fail.
Switch to `allkeys-lru` for this workload:

```
maxmemory 8gb
maxmemory-policy allkeys-lru
```

Sizing:

- `qemb` entries are `dim × dtype_bytes + ~100 bytes overhead`. A
  1536-dim `float32` entry is ~6.3 KB. 1M distinct queries at 7 KB
  each = ~7 GB.
- `result` entries are roughly `k × (doc_id + score + metadata)`.
  For `k=10` with doc_ids of 32 bytes and ~128 bytes of metadata
  each, ~1.5–2 KB per entry. 1M entries ≈ 2 GB.
- Add 20% headroom for fragmentation; set `maxmemory` to the
  budget less that.

`allkeys-lru` is the right policy because: both layers are pure
functions of their key; a cold key is cheap to recompute (an
embedding call for `qemb`, an index call for `result`). `volatile-
lru` only evicts keys with explicit TTL — fine too, but
`allkeys-lru` tolerates a late-added entry without TTL (bug
protection).

### 2.9 Topology and placement

- **Deploy Redis as a side-car or a dedicated service**, not as a
  library-level in-process cache. In-process caches:
  - do not share across replicas, so the hit rate per replica is
    lower
  - complicate invalidation (must broadcast)
  - lose their warm state on every rolling deploy
- **One Redis per retrieval region.** Cross-region latency
  (typically tens of ms) on cache-lookup is often *slower* than
  recomputing the embedding. Replicate across regions only if
  the ingest/invalidation path replicates too.
- **TLS + auth at the Redis boundary.** The cache stores embedded
  user queries and result sets that often include text excerpts;
  it is a PII surface. NIST AI RMF **MANAGE 2.1** (data-privacy
  controls on AI pipeline components) applies.

### 2.10 Worked example — stack with cache (exercise-04 Scenario B)

Baseline from exercise-04 Scenario B: $482/mo, storage-dominated.

Insert Layer 1 + Layer 2 cache with 70% qemb hit rate, 40%
result-cache hit rate:

- `monthly_query_embed` line in `cost/model.py` becomes
  `qpd × days × avg_tokens × price × (1 - 0.70)` = $2.34/mo vs
  $7.80/mo (−70%).
- Compute line (if managed): `qpd × days × price × (1 - 0.40)` =
  60% of what it was.
- Storage: unaffected; the cache is a separate RAM tier of its
  own.
- Cache cost: ~$30–60/mo for a 10 GB Redis instance in a managed
  tier (ElastiCache, Memorystore, Redis Cloud); check your
  provider's current pricing.
- Net impact at Scenario B scale: $5.46/mo saved on query-embed,
  $30+/mo added on Redis — the cache is a *correctness and
  latency* improvement at this scale, not a pure cost win. The
  cost win kicks in at Scenario C scale.

The two numbers (hit-rate improvement, dollar savings) must be
measured on a 7-day replay from real traffic logs before the
cache is shipped; see the review memo template in §2.11.

### 2.11 `cache/review_memo.md` — the decision memo

```markdown
# Retrieval cache decision — YYYY-MM-DD

**Workload:** <QPD, avg query tokens, top-k, filter cardinality>.
**Topology:** <single Redis / per-region / managed service>.
**TTLs:** qemb=<n days>, result=<n minutes>.
**Keying:** qemb=`qemb:{model}:{revision}:{sha(norm(q))}`;
           result=`result:{build_id}:{filter_sha}:{k}:{sha(norm(q))}`.

**Measured hit rate on 7-day replay:**
  qemb   <x>% (p50 / p95 latency on hit <y>ms / <z>ms)
  result <w>% (p50 / p95 latency on hit <y>ms / <z>ms)

**Cost delta (exercise-04 model):**
  query-embed line   $<baseline> → $<new>
  managed compute    $<baseline> → $<new>
  redis              $<new>
  **net monthly**    $<delta>

**Latency delta on retrieval p95:** <ms before> → <ms after>.
**Quality delta (exercise-02 eval, pre- and post-cache):** none expected;
  measured: <metric> <v> vs <v>.

**Invalidation:**
  cutover — prefix shift on alias flip (Pattern A, §2.6);
  real-time ingest — <wired/not wired>; cadence <T>.

**Residual risks / kill switches:**
  - hit rate drops below <threshold>% for >24h → investigate (traffic
    pattern shift; see exercise-06 drift monitor).
  - cache miss rate spike correlated with cutover → check alias flip
    did not purge a prefix it shouldn't have.
  - PII in result payload → cache is not an export surface; enforce
    TLS + auth; see NIST AI RMF MANAGE 2.1.

**Approver:** <name>.
```

## 3. Validation steps

A correct submission can be validated with these checks:

1. **Correctness under cutover.** Simulate a cutover by changing
   the `build_id` passed to `get_or_compute_result`. The next
   `get_or_compute_result` on an old-build-cached query must
   MISS (not hit against the old `build_id` prefix). Confirm
   `CACHE_LOOKUPS{layer="result", outcome="miss"}` increments
   and the compute lambda is called.
2. **Correctness under model swap.** Same, but change
   `(model, revision)` passed to `get_or_compute_qemb`. Old
   embeddings must miss.
3. **Filter isolation.** Two calls with the same query and
   different `filters` dicts must miss each other and each
   produce independent cache entries. Confirm by
   `redis-cli KEYS "result:*" | wc -l` → 2.
4. **Normalization effect.** `"Reset password"` and
   `"reset  password  "` (double space) and
   `"RESET PASSWORD"` all collapse to the same cache key. Confirm
   on three calls that lookups count: hits = 2, misses = 1.
5. **Prefix purge.** `invalidate_build(old_build_id)` deletes
   only entries with that prefix. Count before, count after, and
   confirm other-build entries are untouched.
6. **TTL enforcement.** Set `result_ttl_s=2`, write an entry,
   wait 3 seconds, confirm the read misses.
7. **Metrics shape.** Prometheus exposition shows exactly six
   counter time series (`qemb × 3 × result × 3` for the outcomes)
   after a mixed workload; no per-query labels; cardinality
   bounded.
8. **LRU eviction.** Fill Redis past `maxmemory`; confirm the
   oldest-accessed `qemb` keys evict first (verify by querying
   evicted key counts via `INFO stats`).
9. **Latency.** p95 on the `qemb` lookup path is < 2 ms when
   Redis is on the same network. If not, the Redis is in a
   different region; move it or scrap Pattern A.
10. **Replay-based hit-rate measurement.** On a 7-day replay from
    production traffic logs, hit rates are within the table in
    §2.7 (a 10-order-of-magnitude deviation is a warning sign the
    traffic is not Zipfian; recompute sizing).
11. **ACL leak check.** Two users with different `acl_subject_sha`
    issuing the same query produce distinct cache keys; confirm
    by inspecting the Redis key list.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Two layers separate | Query-embedding and result cache are distinct key spaces with distinct TTLs. |
| Keys contain versioning | `qemb` key includes model + revision; `result` key includes `build_id`. |
| Cutover is lazy | Normal cutover does not require active cache purge; relies on prefix + TTL. |
| Filter in result key | ACL/filter is in the result cache key; users do not share results across scopes. |
| Normalization is lossless | Normalization does NOT stem, drop stopwords, or alter punctuation. |
| TTL ≤ ingest cadence | Default TTL matches or is shorter than the ingest cadence; documented. |
| Eviction policy set | Redis configured with `allkeys-lru` or `volatile-lru`, with explicit `maxmemory`. |
| Metrics bounded | Hit/miss/stale counters have low, fixed label cardinality. |
| Semantic cache is advisory | Semantic cache (if present) is a shadow/measurement layer, not a load-bearing cache. |
| Replay-based sizing | Cache size is justified by a replay of real traffic, not a round number. |
| Topology is out-of-process | Cache is a service/side-car, not a Python dict in the retriever. |
| Security | TLS + auth on the cache; PII concerns addressed. |
| Memo filled | Review memo includes measured hit rate, cost delta, invalidation plan, and approver. |

Fail-fast smells:

- One cache key for both layers. The two concerns have different
  lifetimes; one key forces the worse of them on both.
- TTL = 24 hours on a corpus that ingests every 5 minutes. The
  cache will serve 24-hour-old answers; user complaints incoming.
- The cache key is `query.lower()` without SHA. Long or
  non-ASCII queries break the key; URL-unsafe characters break
  the metrics labels; bounds on key length matter.
- Normalization strips stopwords. The cache now conflates "what is
  X" with "X" — different retrievals, same cache key.
- Invalidation on cutover is "delete everything." Takes the
  qemb cache down with it; recovery time measured in hours.
- Hit rate claimed 95%+ without a replay. Usually a bug in the
  metric or the test stream.
- Cache stores plaintext user queries in a bucket with no auth.
  PII surface; move to encrypted Redis with AUTH.

## 5. Common mistakes

- **Collapsing the two layers.** The most common footgun. A
  single `cache[query] = top_k` makes cutover a nuclear purge
  and reword-robustness worthless.
- **Omitting `build_id` from the result key.** Post-cutover, the
  cache serves answers from the old index. Correctness incident,
  not a performance incident.
- **Omitting the filter hash.** Two users see each other's
  results. Security incident. MITRE ATLAS
  **AML.T0024** is the threat framing.
- **Over-normalizing.** Stemming or stopword-removal at cache
  time shifts the retrieval contract inside the cache layer;
  users get different top-10s depending on cache state.
- **TTL longer than ingest cadence.** Serves stale answers;
  product-level SLO is silently violated; nobody notices until a
  specific question "doesn't find a doc we added yesterday."
- **In-process cache that resets on deploy.** Rolling deploys
  cost the whole cache; hit rate drops post-deploy; latency p95
  spikes; the team adds a "deploy slowly" rule to paper over it.
  Move the cache out of process.
- **No eviction policy.** `noeviction` + full Redis = writes
  start failing silently; miss rate climbs; retrieval latency
  degrades; cache looks "cold" even though it is just stuck.
- **Semantic cache as the only layer.** Serves similar-query
  answers under a correctness contract the user did not opt into.
  Great as a shadow measurement; dangerous as the production key.
- **No metric on stale/miss ratio.** Debugging a cache without
  `hit/miss` counters per layer is detective work via logs.
- **Hot-path blocking on Redis.** If the cache lookup is
  synchronous and Redis has a bad minute, retrieval p95 explodes.
  Wrap with a short timeout + fall back to the retriever; never
  let a cold cache page on-call.

## 6. References

Primary tool documentation:

- Redis — [redis.io/docs](https://redis.io/docs/latest/).
  Relevant: eviction policies, `SCAN`, TTL semantics, persistence
  modes. Semantic caching with Redis Vector:
  [semantic-cache docs](https://redis.io/docs/latest/develop/ai/cache/semantic-cache/)
  (shadow-layer use only, per §2.5).
- Memcached — [memcached.org](https://memcached.org/).
  Simpler alternative when values are < 1 MB and persistence is
  not required.
- `redis-py` — [redis.readthedocs.io](https://redis.readthedocs.io/).
  Async client used in `cache/impl.py`.
- Prometheus client — [prometheus.github.io/client_python](https://prometheus.github.io/client_python/).

Patterns and background:

- GPTCache — [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache).
  A reference implementation of semantic caching; useful as an
  example but inherits the pitfalls in §2.5.
- Zipf's law in web search — Silverstein et al., 1999,
  "Analysis of a very large web search engine query log," SIGIR
  Forum 33(1).

Governance framings:

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: **MANAGE 2.1** (privacy controls on pipeline
  components — the cache is one), **MEASURE 2.6** (performance
  measurement, including cache hit rate).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: **ML02 Data Poisoning Attack** (a cache that serves
  wrong answers due to stale invalidation is an effective
  functional-poisoning amplifier).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: **AML.T0024 Exfiltration via ML Inference API**
  (ACL-leaking cache keys are a path to exfiltration).

Related exercises:

- exercise-01 — the `build_id` that keys the result cache.
- exercise-02 — the eval gate that confirms the cache does not
  degrade quality.
- exercise-03 — the ingest DAG that may emit invalidation
  events for Pattern B.
- exercise-04 — the cost model whose `emb_cache_hit_rate` is the
  parameter this exercise measures.
- exercise-06 — the drift monitor that watches for query-mix
  shifts which can drop the hit rate.
