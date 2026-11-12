# Exercise 06 — Solution: Dual-index cutover and versioning convention

This is a reference solution for
`exercise-06-dual-index-cutover-and-versioning-convention.md`. It is a
**design + runbook** exercise. The learner writes the versioning convention that
lets two indexes live side by side (an "active" one serving reads and a "shadow"
one being rebuilt), designs the cutover runbook that flips traffic between them
without downtime, and provides the rollback path. The teaching point is that the
risky changes in RAG — a new embedding model, a new chunker, a new index
algorithm, a schema migration — all route through the same mechanism.

## 1. Solution overview

The deliverable has five parts:

1. **`notes/versioning-convention.md`** — the naming scheme that lets two
   indexes coexist. One line: `index_name = "{corpus}__e{embed_version}__c{chunker_version}__i{index_version}"`.
2. **`notes/active-alias.md`** — how the *application* resolves which index to
   query: a single pointer (database alias, environment variable, config
   map) that names the active index. Flipping the pointer is the cutover.
3. **`runbooks/cutover.md`** — the step-by-step runbook with explicit
   verification gates (parity queries, recall check, shadow traffic, cutover,
   soak, old-index retirement).
4. **`runbooks/rollback.md`** — the fast path back to the prior active
   version when the cutover is observed to be worse.
5. **`src/dual_index/`** — a reference router that reads the active alias
   from a store (config, DB row, or env) and routes `search()` to the
   named index. Switching the alias flips traffic; no code deploy.

The point is that **cutover is the normal operation** of a RAG system that
evolves. Any change to the embedding model, chunking strategy, filter payload,
or ANN algorithm forces a reindex, which forces a dual-index period, which
must be a routine and automatable procedure — not a heroic migration.

## 2. Worked answer or implementation

### 2.1 What changes force a cutover (and what does not)

| Change | Forces reindex? | Forces cutover? | Why |
|---|---|---|---|
| Add a payload column (e.g., `doc_type`) | No | No | Populate in place; the vectors are unchanged. |
| Add a payload index on existing column | No | No | Online. |
| Add new chunks from new documents | No | No | Routine upsert. |
| Change HNSW `ef_search` | No | No | Query-time knob. |
| Change HNSW `M` or `ef_construction` | **Yes** | **Yes** | Index rebuild from scratch. |
| Change embedding model | **Yes** | **Yes** | All vectors must be re-embedded; dim may also change. |
| Change distance metric | **Yes** | **Yes** | Index is metric-specific. |
| Change chunking strategy | **Yes** | **Yes** | Every chunk's text, id, and embedding changes. |
| Change vector scalar type (fp32 → fp16 → int8) | **Yes** | **Yes** | Index must be rebuilt in the new dtype. |
| Migrate vector store (pgvector → Qdrant) | **Yes** | **Yes** | Different service entirely. |

The convention below handles every "Yes" row through the same
version-bump-then-alias-flip procedure.

### 2.2 Versioning convention — `notes/versioning-convention.md`

**Rule.** An index is named by the combination of inputs that could change its
contents. Two indexes whose names differ in any of these inputs are not
interchangeable — the router must see them as separate names.

```
index_name = "{corpus}__e{embed_version}__c{chunker_version}__i{index_version}"
```

- `corpus` — logical corpus ID. `faq`, `policy`, `kb_global`, etc. One index
  serves one corpus.
- `embed_version` — a compact version tag for the embedding model that
  produced the vectors. Not just the model name, because a model can be re-run
  with different prompt templates or normalization.
    - Format: `{model-short}-{prompt-template-hash-6}`
    - Example: `bge-small-7e3a9c`
- `chunker_version` — the chunker identity and its parameters, hashed.
    - Format: `{chunker-name}-{params-hash-6}`
    - Example: `recursive-800-100-0ab12e`, `semantic-95-9c44aa`
- `index_version` — the index algorithm and parameters.
    - Format: `{algo}-{params-hash-6}`
    - Example: `hnsw-m16-efc200-3d9f71`

Three concrete examples:

```
kb_global__ebge-small-7e3a9c__crecursive-800-100-0ab12e__ihnsw-m16-efc200-3d9f71
kb_global__ebge-small-7e3a9c__crecursive-800-100-0ab12e__ihnsw-m32-efc200-aa9220
kb_global__ee5-base-9ff017__crecursive-800-100-0ab12e__ihnsw-m16-efc200-3d9f71
```

The first two differ only in `index_version` (an HNSW `M` sweep). The first
and third differ only in `embed_version` (an embedder swap). The three can
co-exist as three separate physical indexes.

**The names are stable and reproducible.** Given the same inputs, the hash is
the same; given different inputs, it is different. Reproducibility matters
because the cutover runbook depends on the ability to re-build the "old"
index from its name if a roll-forward goes sideways.

The version metadata itself lives in a per-index `manifest.json`:

```json
{
  "name": "kb_global__ebge-small-7e3a9c__crecursive-800-100-0ab12e__ihnsw-m16-efc200-3d9f71",
  "corpus": "kb_global",
  "embed": {
    "model": "BAAI/bge-small-en-v1.5",
    "template": "passage: {text}",
    "template_hash": "7e3a9c",
    "dim": 384,
    "normalized": true
  },
  "chunker": {
    "name": "recursive",
    "params": {"chunk_size": 800, "chunk_overlap": 100},
    "params_hash": "0ab12e"
  },
  "index": {
    "algo": "hnsw",
    "metric": "cosine",
    "params": {"M": 16, "efConstruction": 200},
    "params_hash": "3d9f71"
  },
  "built_at": "2026-02-14T18:22:10Z",
  "built_from_snapshot": "s3://corpora/kb_global/2026-02-14/manifest.json",
  "row_count": 10234567
}
```

### 2.3 Active alias — `notes/active-alias.md`

The application never hard-codes an index name. It reads an *alias* that maps
a logical name (`kb_global.active`) to a physical name (one of the versioned
indexes above). Cutover is a single atomic update of this alias.

Four reasonable ways to store the alias, all equivalent in semantics:

- A Postgres row in a `search_routes` table (`corpus, alias, target_index`),
  updated with a single `UPDATE ... RETURNING *` transaction.
- A Consul / etcd key.
- A ConfigMap in Kubernetes updated by the deploy pipeline.
- An environment variable on each replica, re-read at hot-reload.

For pgvector, the alias can be *the table itself*: use a view
(`kb_global_active`) backed by the physical table name, and `CREATE OR REPLACE
VIEW` to flip. Views are atomic; no stale cursor sees the old table mid-flip.

```sql
-- One atomic flip of the active view.
CREATE OR REPLACE VIEW kb_global_active AS
  SELECT * FROM kb_global__ebge_small_7e3a9c__crecursive_800_100_0ab12e__ihnsw_m16_efc200_3d9f71;
```

For Qdrant, use the **collection alias** API:

```python
client.update_collection_aliases(change_aliases_operations=[
    qm.ChangeAliasesOperationsAnyOf(
        actions=[qm.CreateAlias(
            alias_name="kb_global_active",
            collection_name="kb_global__ebge-small-7e3a9c__...__ihnsw-m16-efc200-3d9f71",
        )]
    )
])
```

For Milvus, aliases are first-class: `utility.create_alias(collection_name,
alias)` and `utility.alter_alias(collection_name, alias)`.

The application always targets the alias, not the physical name. **The physical
name appears in exactly three places: the index manifest, the alias
definition, and the build script. Nowhere else.**

### 2.4 The cutover runbook — `runbooks/cutover.md`

The runbook has six stages. Each stage has a go/no-go gate; you do not advance
without the gate.

```markdown
# Cutover runbook: `kb_global` from <old-index> to <new-index>

Ticket: <link>
Approver: <name>
Scheduled: <YYYY-MM-DD HH:MM TZ>

## Pre-flight (24 hours before)

- [ ] `manifest.json` for the new index is written and reviewed.
- [ ] New index has been built and `row_count` matches the snapshot the
      old index was built from (± new-docs-since-snapshot).
- [ ] New index has passed the parity eval:
      Recall@10 on the shared eval set within −2 points of the current
      active index (or improved, if that is the point of the cutover).
      Measured with `src/eval/run_parity.py`.
- [ ] Rollback plan is written (`runbooks/rollback.md`) and names the
      specific physical index to flip back to.

## Stage 1 — Shadow traffic (1 hour)

Mirror 100% of read traffic to the new index. Do NOT serve its results to
users; log the top-10 result ids alongside the active's top-10 result ids.

Gate: at least 1000 shadow queries have been mirrored and the
`src/eval/compare_shadow.py` report shows:
- Overlap@10 ≥ 0.80 between old and new on a labeled sample (unless the
  cutover is specifically trying to change results — then skip this and
  rely on the Recall@10 parity from pre-flight).
- No per-tenant drops of > 10 points in Overlap@10 (surfaces per-cohort
  regressions that aggregate metrics hide).
- Shadow p95 latency on the new index ≤ 1.5× active p95. Beyond that,
  pause and investigate.

## Stage 2 — Canary flip (5% for 30 minutes)

Flip the alias for 5% of traffic to the new index (either by hashing
`tenant_id % 20 == 0` at the router or by using a canary alias). Watch:

- Error rate on the new index path < baseline + 0.5 absolute points.
- p95 latency < baseline + 10 ms.
- Qualitative thumbs-up/thumbs-down from in-app feedback does not
  degrade.

Gate: 30-minute soak with the above thresholds met. Abort → stage 6
(rollback).

## Stage 3 — Full flip

Atomic `UPDATE search_routes SET target_index = '<new>' WHERE alias =
'kb_global_active';` (or equivalent alias API call). Confirm with
`SELECT * FROM search_routes WHERE alias = 'kb_global_active';` returning
the new name.

## Stage 4 — Soak (24 hours)

- Error rate, p95 latency, Recall@10 on a pinned eval set, and
  user-feedback metrics are all tracked for 24 hours.
- Any sustained regression ≥ the stage-2 thresholds for > 10 minutes
  triggers rollback.

## Stage 5 — Retire the old index

After 7 days of clean operation, the old index can be dropped:

- Confirm no application code references the physical name (grep).
- Snapshot the old index (`pg_dump` partial / Qdrant snapshot /
  Milvus backup) to the archive bucket under
  `s3://backups/kb_global/retired/<physical-name>/`.
- Drop the physical index. Record the drop in the cutover ticket.

## Stage 6 — Rollback (only if an earlier gate fails)

See `runbooks/rollback.md`.
```

### 2.5 The rollback runbook — `runbooks/rollback.md`

```markdown
# Rollback: `kb_global` from <new-index> back to <old-index>

Trigger: any stage-2 or stage-4 gate in `cutover.md` fails.

## Procedure

1. Flip the alias back. One statement:
   `UPDATE search_routes SET target_index = '<old-physical-name>'
    WHERE alias = 'kb_global_active';`
   (pgvector view: `CREATE OR REPLACE VIEW ...`; Qdrant: alias API;
   Milvus: `alter_alias`.)
2. Confirm by issuing one test query through the application path and
   checking the response's `index_name` field (see 2.6).
3. Do not drop the new index yet. It is still useful for post-incident
   analysis.
4. Open a post-cutover ticket with the specific regression observed,
   the shadow-traffic diff that caught it (if any), and a proposed fix.

## Timing

A clean rollback from a failing stage-4 soak should complete within 60
seconds end-to-end: one alias update + one verification query. If the
rollback takes longer, the versioning convention has been violated
somewhere (physical name leaked into code, alias is not atomic, etc.).
```

### 2.6 Router — `src/dual_index/`

The router is the thin shim between the application and the physical index.
It reads the alias, forwards the query, and stamps the response with the
*physical* index name for observability.

```python
# src/dual_index/router.py
"""Resolve `logical_name` -> `physical_name`, forward the query.

Alias resolution is a cached lookup with a short TTL; the cache lets a hot
loop avoid a round-trip to the alias store, and the TTL means a flip
propagates within TTL seconds.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Protocol

@dataclass
class SearchResult:
    hits: list
    index_name: str   # stamped physical name
    served_from_alias: str


class AliasStore(Protocol):
    def resolve(self, logical: str) -> str: ...


class EnvAliasStore:
    """Dev-mode alias store: read from env vars, no network round trip."""
    def resolve(self, logical: str) -> str:
        key = f"RAG_INDEX_{logical.upper().replace('.', '_')}"
        name = os.environ.get(key)
        if not name:
            raise KeyError(f"no alias configured for {logical} (env {key})")
        return name


class CachedAlias:
    def __init__(self, inner: AliasStore, ttl_s: float = 5.0):
        self.inner = inner
        self.ttl = ttl_s
        self._cache: dict[str, tuple[float, str]] = {}

    def resolve(self, logical: str) -> str:
        now = time.monotonic()
        hit = self._cache.get(logical)
        if hit and now < hit[0]:
            return hit[1]
        name = self.inner.resolve(logical)
        self._cache[logical] = (now + self.ttl, name)
        return name


class Router:
    """One router per (store, logical-name). Picks the physical index per call."""
    def __init__(self, store, aliases: CachedAlias, logical_name: str):
        self.store = store
        self.aliases = aliases
        self.logical_name = logical_name

    def search(self, qvec, top_k: int = 10, **filters) -> SearchResult:
        physical = self.aliases.resolve(self.logical_name)
        hits = self.store.search_on(physical, qvec, top_k=top_k, **filters)
        return SearchResult(
            hits=hits, index_name=physical, served_from_alias=self.logical_name,
        )
```

The `store.search_on(physical_name, ...)` method is a thin per-store wrapper
whose only job is to map a physical name to the store's native "search the
collection X" call. For pgvector this is a query against the table (or view);
for Qdrant and Milvus it is the collection name parameter to `search`.

The `index_name` field on every response is the thing that makes rollback
verifiable: when traffic should be on the new index, the response includes
the new physical name; when rolled back, it includes the old one. No log
line, no deploy tag, just the data itself.

### 2.7 Verification queries

Two small verifications are the backbone of the runbook. Both are standalone
scripts the learner should produce.

**Parity (pre-flight).** Run the shared eval set through both indexes.
Compute Recall@10 against the shared ground truth. The new index must be
within −2 points of the old one *or* the cutover is explicitly meant to
improve retrieval (in which case the gate is that it does, not that it is
the same).

**Shadow overlap (stage 1).** For a replayed production query log, compute
`Overlap@10` (fraction of the top-10 ids in common) between the two
indexes. A full overlap is not expected after a cutover — different
embeddings give different results. What matters is that the overlap is in
the range you *expected*. A larger-than-expected delta means something
other than the planned change moved, and is a reason to pause.

## 3. Validation steps

1. The versioning convention produces a stable hash: feeding the same
   `{model, template}` through the hash function produces `7e3a9c` every time;
   changing the template changes the hash.
2. Two indexes with different `index_version` can be built into the same store
   simultaneously; nothing in the names collides.
3. The alias flip is atomic: a concurrent reader either sees the old physical
   name or the new one, never both and never neither. For pgvector this is
   verified by `CREATE OR REPLACE VIEW` under an in-flight query. For Qdrant
   / Milvus, the alias API is documented atomic.
4. `src/dual_index/router.py` with `EnvAliasStore` and `CachedAlias` returns
   the configured name within `ttl_s` seconds after the env is changed.
5. The rollback runbook, executed on the local Docker stack, flips the alias
   back and the next query's `index_name` reflects the old physical name.
   Total wall-time from "trigger rollback" to "alias confirmed" is under
   60 seconds.
6. The cutover ticket template records every gate's measured value — not
   "passed" or "failed" but the actual Recall@10, p95, Overlap@10 that
   justified the decision. Reviewers can audit the decision later.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Convention | Index name encodes corpus, embedder, chunker, index-algo + hash. Two indexes that differ in *any* of these are named distinctly. |
| Manifest | `manifest.json` next to each index records the exact inputs; no "we think this is bge-small" guesses. |
| Alias | Application reads a logical alias, never a physical index name. The alias lives in exactly one authoritative store. |
| Atomic flip | The cutover is one operation. `CREATE OR REPLACE VIEW` / `update_collection_aliases` / `alter_alias`. No multi-step choreography. |
| Pre-flight | Parity eval runs on the shared eval set with the same ground truth as the active index. Recall@10 delta is reported, not asserted. |
| Shadow stage | Shadow traffic is logged but NOT served; overlap and latency gates fire before any user sees the new results. |
| Canary | Small, time-boxed canary with explicit error-rate and latency thresholds. |
| Soak | 24-hour soak with per-cohort tracking — aggregate metrics hide tenant regressions. |
| Rollback | A single flip reverses it. Time-to-rollback is a stated SLO (< 60 s). |
| Retirement | Old index is retained for ≥ 7 days post-cutover, then snapshotted before drop. |
| Observability | Every response carries the physical index name; dashboards can slice by it. |

Fail-fast smells:

- The alias is "the latest index" by convention (highest timestamp). Easy to
  write; impossible to rollback atomically. Use a *named* alias.
- The physical index name is in application code. Then "deploy a code change"
  becomes part of the cutover, destroying atomicity.
- The cutover runbook has no numeric gates — just "look at it and decide." You
  cannot repeat that; you cannot audit it; you cannot automate it.
- The rollback runbook assumes the old index is still around but there is no
  retention policy that enforces it. First time something goes wrong, the old
  index was dropped yesterday.
- Shadow traffic is served to users ("we'll just route a few queries to the
  new one to see"). That is the canary stage, not the shadow stage. Do not
  conflate them; shadow is users-cannot-see, canary is some-users-can-see.

## 5. Common mistakes

- **Treating cutover as a one-off migration.** The right mental model is that
  every embedder bump, chunker tweak, or `M` sweep is a cutover. If the
  procedure is a 3-day heroic effort, the system is not shippable past the
  first retrieval evaluation.
- **Naming the index after what it "is" rather than how it was built.**
  "`kb_global_v2`" tells you nothing about which embedder, which chunker,
  which parameters. The grading checks that the name is reconstructable
  from the inputs.
- **Hardcoding the physical index name in application code.** Any reference
  other than through the alias is a bug waiting for the next cutover.
- **Skipping the shadow stage because "the eval set already passed."** The
  eval set is not representative of all tenants. Shadow catches
  tenant-specific regressions that the eval set doesn't.
- **Dropping the old index too early.** The soak is 24 hours; the retention
  is 7 days. Dropping on day 2 because "the new one works" means a late-
  arriving regression has nowhere to roll back to.
- **Using feature flags instead of an alias.** Feature flags are per-user;
  aliases are system-level. For a cutover, the system-level flip is what
  gives you a clean rollback. Feature flags are the right tool for canary
  (stage 2) but not for the cutover itself.
- **Letting `manifest.json` drift from reality.** The manifest records what
  was *intended*. If the build script was interrupted, retried, or hand-
  edited, the manifest may not match. Validate on build: a manifest whose
  `row_count` disagrees with the index's actual count is a failed build,
  not a shippable artifact.
- **Forgetting that embedding-model versions change *under you*.** "bge-
  small-en-v1.5" is a Hugging Face checkpoint; the uploader can update the
  underlying weights without changing the tag in some repositories. Pin
  model hashes in the manifest (`sha256` of the config + weights), not just
  the model name.
- **Not stamping the response with the physical name.** Then rollback
  verification is "check a config file" rather than "check the response."
  The former is error-prone; the latter is self-evident.

## 6. References

- Alias / view mechanisms (authoritative docs):
  - PostgreSQL `CREATE OR REPLACE VIEW` — [postgresql.org/docs/current/sql-createview.html](https://www.postgresql.org/docs/current/sql-createview.html).
  - Qdrant collection aliases — [qdrant.tech/documentation/concepts/collections/#collection-aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases).
  - Milvus collection alias API — [milvus.io/docs/manage_collection.md#Alias](https://milvus.io/docs/manage_collection.md).
  - Elasticsearch / OpenSearch index aliases — [elastic.co/guide/en/elasticsearch/reference/current/aliases.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html).
- Operational patterns:
  - "Blue-green deployment" (Humble & Farley, *Continuous Delivery*) — the
    cutover model the runbook adapts to vector indexes.
  - Google SRE book, "Release Engineering" chapter — [sre.google/sre-book/release-engineering/](https://sre.google/sre-book/release-engineering/).
- Risk / governance framings (the framings that make retention,
  reproducibility, and rollback non-negotiable):
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
