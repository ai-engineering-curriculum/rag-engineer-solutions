# Exercise 01 — Solution: Index Versioning and Dual-Index Cutover Drill

This is a reference solution for
`exercise-01-index-versioning-and-dual-index-cutover-drill.md`. It is a
**design + implementation** exercise. The learner picks an index-naming
scheme that captures every input that can change retrieval behavior,
runs a dual-index (blue/green) cutover drill against a live retriever,
and writes the rollback contract so a bad cutover is reversible in
minutes, not hours.

The teaching point: *the vector index is a build artifact, not a live
database*. Treat it like a container image — immutable, content-
addressed, built from pinned inputs, promoted through environments by
alias/pointer flips. Mutating an index in place to "update it" is the
same mistake as mutating a running container's filesystem to "patch it":
you lose reproducibility, you lose the ability to roll back, and the
next audit will not be able to reconstruct what retrieval looked like on
the day a given answer was served.

## 1. Solution overview

The deliverable has four parts:

1. **`versioning/naming_spec.md`** — the index-naming contract. Every
   field that can change the retrieval surface (corpus snapshot,
   chunking config, embedding model, vector index parameters, ACL
   policy version) is captured in either the index name, an attached
   manifest, or both. The naming scheme is deterministic: identical
   inputs produce identical names.
2. **`versioning/manifest.schema.json`** — a JSON Schema for the
   per-index manifest. One manifest per build, co-located with the
   index artifact, signed or hashed so tampering is detectable.
3. **`cutover/runbook.md`** — the step-by-step cutover drill (build →
   dual-write → dual-read + compare → eval gate → alias flip → bake →
   decommission) with a named abort condition at every step.
4. **`cutover/rollback.md`** — the rollback contract: which signal
   fires the rollback, who approves it, what the alias flip command
   is, how long the bake on the old index must be before the old
   alias is reused.

Scope: *the version/cutover plane only*. The eval gate itself is
exercise-02; the index-build automation (so that "rebuild" is a DAG
run, not a laptop script) is exercise-03; the drift monitor that may
trigger a rebuild is exercise-06. This exercise ensures that when any
of those fires, the cutover is safe and reversible.

## 2. Worked answer or implementation

### 2.1 What an index version must capture

A retrieval system's behavior changes when any of the following
changes. The version must therefore cover all of them:

| Input | What changes if this moves |
|---|---|
| Corpus snapshot | Which documents can be retrieved at all. |
| Chunker config (splitter, chunk size, overlap, metadata extracted) | Which spans of text are addressable. |
| Embedding model (provider + model id + version/revision) | Vector space is different; old queries do not project to the same neighborhood. |
| Vector-index parameters (HNSW `M`, `efConstruction`; IVF `nlist`; PQ codebook) | Recall/latency curve shifts; nearest-neighbor set changes even on the same vectors. |
| Normalization (L2-norm, cosine vs dot, distance metric) | Score scales differ; score-based reranking downstream breaks. |
| Field schema (metadata fields indexed, filter-able fields, ACL fields) | Which filters work, which facets exist, which ACL enforcement is possible. |
| Analyzer / tokenizer (for the sparse side of hybrid) | Token set and BM25 statistics differ. |

A version that does not capture one of these is a trap: a "no-op
rebuild" that silently shifts recall by a few points, with no way to
attribute the delta.

### 2.2 `versioning/naming_spec.md` — the naming contract

The name has two parts: a **collection family** (the stable public
name an application code aliases against) and a **build id** (the
content-addressed build artifact). Applications only ever bind to
the family via an alias; the build id rotates on every rebuild.

Collection-family examples (stable, human-readable, lowercase, no
semver):

```
docs-support-prod
docs-support-staging
tickets-eng-prod
```

Build id format:

```
<family>_<corpus_sha>_<cfg_sha>_<emb>_<built_at>
```

Where:

- `corpus_sha` — SHA-256 of the sorted list of
  `(source_uri, content_sha256, mtime)` tuples that constitute the
  corpus snapshot. Deterministic across machines; cached in object
  store under `corpora/<corpus_sha>/manifest.jsonl`.
- `cfg_sha` — SHA-256 of a canonical-JSON serialization of
  `{chunker_cfg, index_cfg, analyzer_cfg, field_schema, acl_cfg}`.
  The *canonical* qualifier matters: sort keys, strip whitespace,
  fix number formatting; otherwise the same config produces different
  shas on different machines.
- `emb` — `<provider>-<model_id>-<revision>`, e.g.
  `openai-text-embedding-3-large-20240125` or
  `sbert-all-mpnet-base-v2-20220818`. The revision is the model
  weights' publish date or the Hugging Face commit, never "latest."
- `built_at` — ISO-8601 UTC timestamp, minute granularity, included
  for human readability only (not for identity — two builds with the
  same inputs will have the same first three shas and should be
  treated as equivalent).

Example full build id:

```
docs-support-prod_corpus-7f3a..9e_cfg-bd12..01_openai-text-embedding-3-large-20240125_20260311T1402Z
```

Rules the name enforces:

- **Content-addressed identity.** If the first three shas match a
  prior build, the index is byte-equivalent in terms of retrieval
  behavior; the system can reuse the old artifact instead of
  rebuilding.
- **No semver.** `v1`, `v2` encode nothing a reader can verify.
  Content hashes encode everything a reader can verify.
- **No in-place mutation.** `*_patched`, `*_fixed`, `*_v2a` are
  forbidden. A change in any input is a new build id.

### 2.3 `versioning/manifest.schema.json` — the attached manifest

Every build writes a `manifest.json` next to the index data. Fields:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": [
    "family", "build_id", "corpus",
    "chunker", "embedding", "index", "analyzer",
    "schema", "acl_policy_version",
    "built_at", "built_by", "built_with_git_sha"
  ],
  "properties": {
    "family":   {"type": "string"},
    "build_id": {"type": "string"},
    "corpus": {
      "type": "object",
      "required": ["corpus_sha", "doc_count", "token_count", "manifest_uri"],
      "properties": {
        "corpus_sha":   {"type": "string"},
        "doc_count":    {"type": "integer"},
        "token_count":  {"type": "integer"},
        "manifest_uri": {"type": "string"}
      }
    },
    "chunker": {
      "type": "object",
      "required": ["splitter", "chunk_size", "chunk_overlap", "cfg_sha"],
      "properties": {
        "splitter":      {"type": "string"},
        "chunk_size":    {"type": "integer"},
        "chunk_overlap": {"type": "integer"},
        "cfg_sha":       {"type": "string"}
      }
    },
    "embedding": {
      "type": "object",
      "required": ["provider", "model_id", "revision", "dim", "normalize"],
      "properties": {
        "provider":  {"type": "string"},
        "model_id":  {"type": "string"},
        "revision":  {"type": "string"},
        "dim":       {"type": "integer"},
        "normalize": {"type": "boolean"}
      }
    },
    "index": {
      "type": "object",
      "required": ["engine", "kind", "params", "metric"],
      "properties": {
        "engine": {"type": "string", "description": "pgvector|qdrant|opensearch|weaviate|milvus|pinecone"},
        "kind":   {"type": "string", "description": "hnsw|ivfflat|flat|pq|ivfpq"},
        "params": {"type": "object"},
        "metric": {"type": "string", "description": "cosine|dot|l2"}
      }
    },
    "analyzer":           {"type": "object"},
    "schema":             {"type": "object"},
    "acl_policy_version": {"type": "string"},
    "built_at":           {"type": "string", "format": "date-time"},
    "built_by":           {"type": "string"},
    "built_with_git_sha": {"type": "string"},
    "eval": {
      "type": "object",
      "description": "Optional: offline eval numbers if an eval gate ran (exercise-02).",
      "properties": {
        "qrels_sha":  {"type": "string"},
        "recall_at_10": {"type": "number"},
        "ndcg_at_10":   {"type": "number"},
        "mrr_at_10":    {"type": "number"}
      }
    }
  }
}
```

Operational rules:

- The manifest is written atomically alongside the index artifact
  and is immutable after the build completes.
- Ship it to object storage with a content hash in the key
  (`manifests/<build_id>.json`) and reference it from any
  observability record that mentions this index.
- Signing the manifest with the build system's key (Sigstore /
  cosign over object-storage URLs, or in-toto attestations) raises
  the bar for supply-chain tampering. OWASP ML Top 10 **ML06 AI
  Supply Chain Attacks** names the artifact chain as a top risk;
  a signed manifest is how you detect substitution.

### 2.4 Alias pattern across common engines

Applications bind to an **alias** (or logical name), never to a
build id. Cutover is an alias flip, not an application redeploy.

| Engine | Alias primitive | Doc |
|---|---|---|
| OpenSearch / Elasticsearch | Index aliases (`PUT /_aliases` atomic add/remove). Supports filtered aliases and routing. | OpenSearch docs: [Index aliases](https://opensearch.org/docs/latest/im-plugin/index-alias/). |
| Qdrant | Collection aliases (`create_alias`, `delete_alias`, atomic switch via `change_aliases_operations`). | Qdrant docs: [Aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases). |
| Pinecone | Namespace per build within one index, or distinct indexes with a routing layer at the client side. | Pinecone docs: [Namespaces](https://docs.pinecone.io/guides/indexes/manage-namespaces). |
| Weaviate | Separate classes per build + an application-level pointer; Weaviate has no first-class alias as of this writing. | Weaviate docs: [Collections](https://weaviate.io/developers/weaviate/manage-data/collections). |
| pgvector | Separate tables per build + a view or a config row the app reads. | pgvector repo: [pgvector/pgvector](https://github.com/pgvector/pgvector). |
| Milvus | Collection aliases (`CreateAlias`, `AlterAlias`, `DropAlias`, atomic swap). | Milvus docs: [Aliases](https://milvus.io/docs/manage-aliases.md). |

Where the engine does not provide a first-class alias (Weaviate,
pgvector, Pinecone-single-index), the alias is a row in a
`retrieval_bindings` table the app reads at startup and watches for
change (SIGHUP or a 60-second refresh). The *rule* is that the app
code never names a build id directly; only the binding layer does.

### 2.5 `cutover/runbook.md` — the drill, step by step

The cutover has six phases. Each phase has an abort condition that,
if met, stops the cutover and preserves the current live index.

**Phase 1 — Build (offline).**

1. Snapshot the corpus. Write
   `corpora/<corpus_sha>/manifest.jsonl` with one line per source
   doc: `{source_uri, content_sha256, mtime, size}`.
2. Resolve `cfg_sha` by hashing the canonical JSON of all configs
   that matter to retrieval (see §2.1).
3. Materialize the embedding model's revision id. For OpenAI,
   that is the model name + a dated revision captured from the
   API response; for Hugging Face, the commit sha of the model
   repo. Never "latest."
4. Compute the `build_id`. If an index with that `build_id`
   already exists in object storage, short-circuit — the build is
   a no-op.
5. Run the ingest + embed + index DAG (exercise-03) with
   `build_id` as the write target.
6. Write `manifest.json` next to the index data; sign it.

*Abort:* corpus manifest hash does not match a known-good upstream
(provenance check fails), or embedding provider returns a different
`revision` field than expected.

**Phase 2 — Shadow write (optional, skippable if ingestion is
batch).**

If the system does online ingestion, start dual-writing new
documents to both the current live index (alias `prod`) and the
new build id for a window (24–72 h depending on ingest rate). This
ensures the new index is not stale on the day of cutover.

*Abort:* dual-write error rate on the new index > 0.1% of ops, or
dual-write latency on the new index > 2× the live index p95.

**Phase 3 — Dual-read compare (traffic mirror).**

Mirror live production queries to the new index. Record both
result sets. *Do not use the new index to answer users yet.*
Compare:

- **Top-k overlap.** Fraction of queries where `|new ∩ live| / k ≥
  0.8` for k ∈ {10, 50}. Expect 70–90% on a healthy embedding-model
  change; expect ~100% on a pure index-param change that should not
  move recall.
- **Score-delta distribution.** For queries where the top-1 is the
  same, the score distribution of top-1 on the new index. A large
  shift here is a normalization bug.
- **Per-slice sanity.** Break the mirror corpus into slices
  (short query / long query / rare-term query / recent docs / old
  docs). A regression hiding in one slice is the common failure
  mode.

Run for ≥ 1000 queries and ≥ 24 hours (so diurnal traffic is
covered).

*Abort:* overlap < 50% on any slice unless the change was
*intended* to move recall (embedding model swap), in which case
proceed to Phase 4 only if the eval gate is green.

**Phase 4 — Eval gate (exercise-02).**

Run the retrieval eval suite against both indexes with the shared
qrels (exercise-01 of mod-109). Compute recall@10, nDCG@10, MRR@10
and the paired bootstrap confidence interval on the delta
(mod-109 exercise-03). Promote only if:

- No metric drops by more than the exercise-02 budget (default
  −2% relative on recall@10, −3% on nDCG@10, with 95% CI not
  spanning zero in the wrong direction).
- Latency p95 is within +15% of the current live index.

Write the eval numbers into the manifest's `eval` block and
re-sign the manifest.

*Abort:* any metric fails the gate. Delete the new build (or move
to `retired/` in object storage), leave the alias pointing at the
live index, open a ticket with the regressing slice called out.

**Phase 5 — Alias flip (the cutover).**

A single atomic operation, scripted:

```bash
# OpenSearch example — one POST, aliases flip together:
curl -X POST "$OS/_aliases" -H 'content-type: application/json' -d'
{
  "actions": [
    {"remove": {"index": "docs-support-prod_corpus-OLD..._...",  "alias": "docs-support-prod"}},
    {"add":    {"index": "docs-support-prod_corpus-NEW..._...",  "alias": "docs-support-prod"}}
  ]
}'
```

Qdrant equivalent:

```python
client.update_collection_aliases(
    change_aliases_operations=[
        DeleteAliasOperation(delete_alias=DeleteAlias(alias_name="docs-support-prod")),
        CreateAliasOperation(create_alias=CreateAlias(
            collection_name="docs-support-prod_...NEW...",
            alias_name="docs-support-prod")),
    ],
)
```

Right after the flip, record the `(timestamp, old_build_id,
new_build_id, operator, ticket)` tuple in an append-only audit log
(`cutovers.log` in object storage, WORM bucket if available).
NIST AI RMF **MANAGE 4.1** calls for traceable change records on
AI components; this log is the artifact that satisfies that.

*Abort:* the alias-flip API returns non-2xx. Treat it as a
cutover abort: do not retry blindly — verify the current alias
state and re-plan. Stuck in half-aliased state is the worst
possible outcome.

**Phase 6 — Bake and decommission.**

- Keep the *old* build id alive, with no alias pointing to it,
  for a bake window (default: 7 days, longer if the change is
  risky or the business cycle is weekly). This is the rollback
  reservoir.
- Monitor live p50/p95 latency, retriever hit-rate (fraction of
  queries returning at least one result), and the top-level
  application metric (answer rate, click-through, thumbs-up).
  Compare to pre-cutover baseline.
- After the bake window and a clean post-cutover eval (see
  exercise-02's scheduled runs), delete the old index and move
  its manifest to `retired/`.

*Abort (= rollback):* see §2.6.

### 2.6 `cutover/rollback.md` — the rollback contract

The rollback is a *reverse alias flip* — one command, same
operator permissions as cutover, no code deploy. It must be
fast and boring; a rollback is not where you want to debug.

**Rollback triggers (any one is enough):**

- Retriever hit-rate drops by > 5% absolute within 30 minutes of
  cutover.
- Application-level metric (answer rate, resolution rate, CTR)
  drops by > 10% relative within 1 hour of cutover.
- Latency p95 on the retrieval span exceeds 1.5× the pre-cutover
  p95 for > 10 minutes.
- Error rate on retrieval API > 1% for > 5 minutes.
- A security/compliance alert that would not fire on the old index
  (e.g., ACL regression — a filter dropped a field the policy
  requires).

**Rollback steps (≤ 2 minutes):**

1. Flip the alias back to the previous build id. Same API call as
   Phase 5, arguments reversed. Confirm the alias now points at
   the old build.
2. Append `{timestamp, trigger, operator, new_build_id,
   old_build_id}` to `cutovers.log` with `kind: "rollback"`.
3. Open an incident ticket referencing the cutover record and the
   trigger metric's dashboard.
4. Do *not* delete the new build immediately — preserve it for
   post-mortem analysis. Move it to `incident/<ticket>/` in object
   storage.

**Decisions that belong in the runbook, not the moment:**

- Who can press the button: the on-call retrieval engineer, no
  additional approvals required. Rollback is reversible; the cost
  of asking is higher than the cost of doing.
- How long before the old index is "cold enough" to re-cutover:
  wait for the triggering metric to recover for 2× the triggering
  window, then rebuild from a known-good config and re-run Phases
  1–5.
- What does *not* trigger a rollback: a small drop in top-k
  overlap at the mirror step after the cutover (already measured
  in Phase 3). Drops in metrics that are noisy at short time
  horizons (daily unique users, long-tail query-mix). Use
  dashboards to verify the signal is sustained before rolling.

### 2.7 Worked example — swapping embedding models

A concrete example of the full drill:

- **Before:** `docs-support-prod` → build id with
  `openai-text-embedding-ada-002-20221215`, HNSW `M=16`,
  `ef=200`, cosine.
- **Goal:** cut over to
  `openai-text-embedding-3-large-20240125`, dim 3072, HNSW
  `M=32`, `ef=256`.
- **Why risky:** different vector space, different dimensionality,
  different latency profile. Top-k overlap is expected to be
  modest (the vectors literally do not live in the same geometry);
  the eval gate is the authority, not the overlap metric.

Phase 1: build the new index under a new build id. Verify the
manifest has `embedding.dim: 3072`.

Phase 3: mirror ≥ 2000 production queries. Top-10 overlap comes
back at 42% — low, but *expected* for a model swap. The slice
check shows no slice below 30%.

Phase 4: eval gate on the mod-109 qrels. recall@10: 0.71 → 0.76
(+5pp, CI [+3pp, +7pp]); nDCG@10: 0.49 → 0.54 (+5pp, CI [+3pp,
+7pp]); MRR@10: 0.58 → 0.61 (+3pp, CI [+2pp, +5pp]). p95 latency:
+8%. All within budget; gate is green.

Phase 5: alias flip. Audit record:

```
2026-03-11T14:12Z CUTOVER
  alias=docs-support-prod
  from=docs-support-prod_corpus-7f3a..._cfg-bd12..._openai-text-embedding-ada-002-20221215_...
  to=docs-support-prod_corpus-7f3a..._cfg-ab41..._openai-text-embedding-3-large-20240125_...
  operator=alice ticket=RET-482
```

Phase 6: bake 7 days. No rollback trigger fires. Retire the old
build; manifest moves to `retired/`.

## 3. Validation steps

A correct submission can be validated against these concrete
checks, most of which are runnable:

1. **Build-id determinism.** Rebuild the same corpus + config +
   model on a different machine. The first three shas in the
   build id must match bit-for-bit. If they don't, the config
   serialization is not canonical.
2. **Alias indirection.** The application code must contain zero
   references to a build id. `grep -rE 'docs-support-prod_corpus'
   src/` returns no application-code hits. Only the binding layer
   and cutover scripts know the full build id.
3. **Atomic cutover.** Simulate a cutover and, *during* the alias
   flip, run `while true; do ./health_probe.sh; done` against the
   retrieval API. There must be no window where the probe sees
   "alias not pointing at any index" or "two indexes aliased to
   the same name." (OpenSearch's `POST /_aliases` is documented
   as atomic; the test is confirming the client wiring respects
   that.)
4. **Rollback rehearsal.** Script the full rollback from a healthy
   state, timed. Total wall-clock from "trigger fires" to "alias
   restored + audit record written" must be ≤ 5 minutes. (Target:
   ≤ 2 minutes; 5 minutes is the fail threshold.)
5. **Manifest completeness.** Every build in object storage has a
   manifest that validates against `manifest.schema.json`. A CI
   job over the `manifests/` prefix runs
   `jsonschema -i manifest.json manifest.schema.json` on every
   file nightly; a failure blocks new cutovers.
6. **No in-place mutation.** `aws s3api list-object-versions
   --bucket ... --prefix indexes/<build_id>/` shows exactly one
   version of each object. Versioning + an object-lock policy
   on the bucket makes this enforceable, not aspirational.
7. **Bake-window compliance.** A linter (or a manual review step)
   confirms that `retire-index` scripts refuse to run on a build
   id whose `cutovers.log` record is younger than the bake window.
8. **Supply-chain evidence.** The manifest is signed and the
   signature verifies against the build system's public key.
   OWASP ML Top 10 **ML06** names this as a supply-chain control;
   without it, an attacker who can write to object storage can
   substitute the index.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Captures every input | Naming scheme + manifest covers corpus, chunker, embedding, index params, analyzer, schema, ACL — none missing. |
| Content-addressed | Build id is a function of the inputs; no semver, no human-chosen suffixes. |
| Determinism | Build id is reproducible across machines; config serialization is canonical (sorted keys, fixed number format). |
| Alias indirection | Application code binds to the family alias, never to a build id. |
| Dual-read before cutover | Mirror + compare step is present; overlap and per-slice checks are defined. |
| Eval gate wired in | Phase 4 gates the cutover on exercise-02's offline metrics with explicit thresholds and a paired-CI test on the delta. |
| Atomic alias flip | Flip is a single API call with a documented atomicity guarantee for the chosen engine; verified by a concurrent probe. |
| Audit log | Append-only record of each cutover and rollback with `old_build_id`, `new_build_id`, operator, timestamp. |
| Rollback contract | Named triggers, named operator role, ≤ 5-min rehearsed drill, bake window on the old index preserved. |
| No in-place mutation | Bucket policy or convention prevents overwriting an index after its manifest is sealed. |
| Supply-chain hygiene | Manifest is signed or hashed with an integrity check that surfaces in CI. |
| Reversibility | At any step before the alias flip, the drill can be aborted with no live-traffic impact; after the flip, rollback is a single reverse flip. |

Fail-fast smells:

- The naming scheme uses `v1`, `v2`, `-new`, `-fixed`. Encodes
  nothing; cannot detect a silent input change.
- The runbook has no abort condition on the mirror step. "If
  overlap is weird, think about it" is not a condition.
- The rollback trigger is "we decide in the moment." The moment is
  the worst place to decide.
- No bake window — the old index is deleted the moment the new one
  is aliased. Rollback becomes a *rebuild*, which takes hours.
- ACL / filter fields are not in the manifest. A filter-field
  rename in a cutover will silently return documents that violate
  policy; without the manifest, the auditor cannot even see which
  policy version was active.

## 5. Common mistakes

- **Mutating an index in place.** "Just reindex the new docs into
  the old collection" collapses the version plane. The alias now
  points at a moving target; recall at 10:00 and recall at 14:00
  are not the same thing. Fix: every ingest batch targets a new
  build id *or* the manifest is updated with the new corpus_sha
  and a new build id is produced and aliased.
- **Using semver in the index name.** `index-v3` tells a human
  that it's the third one; it tells no one what inputs produced
  it. Use content-addressed names for identity and let humans read
  the manifest.
- **Pointing application code at a build id directly.** The
  cutover becomes a code deploy instead of an alias flip; rollback
  is now a code rollback; coordination cost blows up. Always
  alias.
- **No bake window.** Deleting the old index immediately after the
  flip makes rollback a *rebuild*. Rebuild takes hours; a
  rollback has to be minutes.
- **Flipping the alias with a non-atomic API.** On engines where
  `remove` and `add` are separate calls, a client failure between
  them leaves the alias pointing nowhere. Use the engine's atomic
  `actions` API (OpenSearch), the batched change-aliases API
  (Qdrant, Milvus), or a two-phase commit in the binding layer
  (Weaviate, pgvector).
- **Treating the mirror step as the eval.** A mirror measures
  overlap on *live* traffic, which is not labeled. A high overlap
  on unlabeled queries does not prove recall is good; a low
  overlap does not prove recall got worse. The eval gate uses
  qrels.
- **Rolling back on a noisy metric.** Rollback on a 5-minute dip
  in a daily-cyclical metric is how you oscillate between two
  indexes. Rollback triggers must be *sustained signals*, not
  spikes.
- **Omitting the ACL / schema versions from the manifest.** An
  ACL-policy rename inside a cutover can return data that
  violates policy, and the manifest is the only record of which
  policy was live when a given answer was served. MITRE ATLAS
  **AML.T0049 Exploit Public-Facing Application** and
  **AML.T0065 LLM Plugin Compromise** are the threat framings
  that pay attention to policy-version drift.
- **No signed manifest.** Any write to object storage silently
  overwrites the retrieval surface. Signed manifests turn
  "retrieval behavior changed under me" from an audit mystery
  into a signature-verification failure.

## 6. References

Engine documentation (for the alias/version primitives cited in
§2.4):

- OpenSearch — [Index aliases](https://opensearch.org/docs/latest/im-plugin/index-alias/).
  Atomic `POST /_aliases` actions.
- Elasticsearch — [Aliases](https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html).
  Same semantics; use the atomic `actions` body.
- Qdrant — [Aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases).
  Batched `change_aliases_operations`.
- Milvus — [Manage aliases](https://milvus.io/docs/manage-aliases.md).
- Pinecone — [Manage namespaces](https://docs.pinecone.io/guides/indexes/manage-namespaces).
- Weaviate — [Collections](https://weaviate.io/developers/weaviate/manage-data/collections).
- pgvector — [pgvector repository](https://github.com/pgvector/pgvector).

Standards and governance framings (for the manifest, audit log,
and rollback contract):

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant functions: **GOVERN 1.2** (policies, accountability),
  **MANAGE 4.1** (traceable change records), **MEASURE 2.6**
  (post-deployment monitoring).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant items: **ML06 AI Supply Chain Attacks** (signed
  manifests, immutable artifacts), **ML05 Model Theft** (manifest
  references model revision).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: **AML.T0010 ML Supply Chain Compromise** motivates
  the signed-manifest + immutable-build pattern.

Related exercises in this module:

- exercise-02 — the eval gate that Phase 4 depends on.
- exercise-03 — the DAG that executes Phase 1 reproducibly.
- exercise-06 — the drift monitor whose alerts frequently trigger
  a rebuild + cutover drill.
