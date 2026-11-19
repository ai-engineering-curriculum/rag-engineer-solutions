# Exercise 01 — Solution: Permission-Aware Retrieval — Query-Time vs. Index-Time Drill

This is a reference solution for
`exercise-01-permission-aware-retrieval-query-time-vs-index-time-drill.md`.
It is a **design** exercise. The learner picks between query-time ACL
enforcement and index-time ACL enforcement for an enterprise retriever,
defends the pick for three concrete scenarios, and documents the
failure modes each pattern introduces. The deliverable is a comparison
matrix, three scenario memos, a decision tree, and a threat-model
mapping to the official frameworks the course uses for governance.

The teaching point: in enterprise retrieval, the *index* is a copy of
documents the user may or may not be allowed to see. The ACL decision
is therefore not a feature — it is the trust boundary. Where the
decision runs (query-time filter vs. index-time projection vs. a
tenant-isolated index) controls which failure modes are even possible.
Query-time filtering can over-fetch into forbidden content and then
drop it; index-time projection prevents the fetch entirely but drifts
if ACLs change in the source system faster than the index refreshes.
The honest answer is almost always "both, at different layers," and
this exercise makes the learner say *which* layer carries *which*
guarantee.

## 1. Solution overview

The deliverable has four parts:

1. **`matrix/acl_matrix.md`** — the comparison table. Rows are the
   evaluation axes (correctness, latency, freshness, blast radius,
   leakage mode, operational cost, auditability, compatibility with
   hybrid retrievers). Columns are the three canonical patterns
   (query-time post-filter, query-time pre-filter, index-time
   projection / separate-index-per-tenant). Each cell is scored
   `strong` / `adequate` / `weak` with a one-sentence justification.
2. **`matrix/scenarios.md`** — three scenarios, each with a pick, a
   *why*, and the failure mode the pick leaves on the table. The
   scenarios are drawn from the learning file and span SMB
   (single-tenant, per-document ACL), mid-market (multi-tenant,
   per-folder ACL), and regulated (customer-managed keys,
   tenant-isolated indexes).
3. **`matrix/decision_tree.md`** — a 5–7 node decision tree routing
   from scenario characteristics (tenant count, ACL churn rate,
   compliance regime, query latency budget) to a recommended
   pattern.
4. **`notes/threat_model.md`** — a one-page threat model mapping
   each pattern's residual risks to OWASP ML Top 10, MITRE ATLAS,
   and the four NIST AI RMF functions (Govern, Map, Measure,
   Manage). The point is to make the risks nameable in a review,
   not to catalogue every CVE.

Scope: *the enforcement decision only.* The downstream consumers —
exercise-02 (text-to-SQL), exercise-03 (Graph RAG), exercise-04
(multimodal), exercise-05 (hybrid router) — inherit whichever
pattern this exercise picks. The goal here is to make the pick
defensible against a security review, not to write the retriever.

## 2. Worked answer or implementation

### 2.1 The three patterns in one paragraph each

- **Query-time post-filter.** The retriever returns the top-k
  candidates by relevance *without* ACL awareness; a filter layer
  then drops documents the user cannot access and backfills from
  the next-best candidates. Sweet spot: small corpora where the
  ACL check is cheap and the k* (over-fetch) factor stays small.
  Weakest when the ACL predicate is highly selective — the system
  burns the retrieval budget on candidates nobody will see, and
  sometimes returns fewer than k hits because the backfill pool
  runs out.
- **Query-time pre-filter.** The retriever receives the ACL
  predicate *as part of the query* (e.g., a Lucene `filter`
  clause, a vector-DB `where` clause, or a SQL `WHERE`
  constraint) and only ranks candidates the caller is allowed to
  see. Sweet spot: engines whose filter path is efficient and
  whose ACL fields are low-cardinality (e.g., tenant_id,
  classification label). Weakest when the ACL is per-document and
  the allow-set is large — the filter either explodes the query
  plan or degrades recall (ANN pre-filter accuracy drops as
  filters get selective; see mod-104).
- **Index-time projection / separate index per tenant.** ACL
  information is materialized into the index at write time —
  either as a filterable field on every document, or by
  partitioning documents into separate indexes per tenant /
  classification. Sweet spot: strict tenant isolation
  requirements (regulated workloads, CMEK, residency), or
  workloads where the ACL rarely changes. Weakest when ACLs
  change frequently — the index is a stale cache of the
  authorisation system until the next refresh, and a revoked
  permission remains effective at query time until re-indexed.

### 2.2 `matrix/acl_matrix.md` — the comparison table

Scoring legend: **S** = strong / first-class mechanism in the
pattern, **A** = adequate / achievable with modest extra
engineering, **W** = weak / requires significant scaffolding or
sits outside what the pattern is intended to do.

| Axis | Post-filter | Pre-filter | Index-time projection / per-tenant |
|---|---|---|---|
| **Correctness (no forbidden doc ever returned)** | A — correct only if the filter is defensively applied to every code path; common bugs: filter bypass on streaming, on cache, on "explain" endpoints. | S — the forbidden doc is never ranked. A bug in the filter clause fails closed to an empty result, not to a leak. | S — the forbidden doc is not present in the queried index at all. |
| **Recall under selective ACLs** | W — fixed top-k before filter; selective ACL ⇒ frequent short results. | A — modern ANN engines support filtered search with varying recall degradation; lexical filters are first-class. | S — the caller only ever sees their own index, so there is no recall penalty from filtering. |
| **Latency** | S — no extra retrieval work; the filter is cheap list membership. | A — filter cost depends on the engine; ANN pre-filter is often 1.1×–3× over unfiltered, lexical filter is near-free. | S — per-tenant index is small and fast. Multi-tenant single-index with a filterable field matches the pre-filter numbers. |
| **Freshness vs. ACL source of truth** | S — the filter reads the live ACL at query time. | S — the query embeds the live ACL predicate. | W — ACLs are a cached projection at index-write time; revocation is not effective until re-index. |
| **Blast radius of a misconfiguration** | W — a missing filter leaks the entire corpus to one user. | A — a missing filter leaks the corpus but typically on a narrower surface (one query path). | A — a cross-tenant join (wrong index name, wrong alias) leaks one tenant's corpus to another; isolation reduces but does not eliminate this. |
| **Auditability** | A — the ACL evaluation lives in the application layer; logs can be comprehensive but are also easy to omit. | S — the query contains the ACL predicate; query logs capture the authorisation state at call time. | S — per-tenant isolation gives an index-level audit boundary; access to the tenant's index is itself the audit event. |
| **Operational cost (index storage + build)** | S — one index for everything; no duplication. | S — one index, with ACL fields added. | W — storage multiplies with tenants; build pipelines run per-tenant; ANN parameters must be retuned per tenant corpus (mod-104). |
| **Operational cost (ACL change propagation)** | S — instantaneous; nothing to re-index. | S — instantaneous; next query uses the new predicate. | W — requires a re-index or a mutable filterable field; revocation latency = refresh cadence. |
| **Compatibility with hybrid retrievers (BM25 + dense + rerank)** | A — the filter applies to the final fused list; works, but the BM25 and dense pipes each still over-fetched. | S — both the BM25 engine (Lucene `filter`) and the vector DB (`where`) accept the predicate independently. | S — each pipe queries the tenant's index; fusion runs over already-authorised candidates. |
| **Fit with mod-110 MCP exposure** | A — the server must inject the caller's identity into the filter; easy to forget. | S — the MCP tool's input schema can take the ACL predicate explicitly, auditable at the protocol boundary. | S — the MCP server routes the caller to the tenant index at session init; one less per-request decision. |
| **Fit with regulated / CMEK / residency** | W — a single index cannot hold data with different encryption keys or residency domains. | W — same reason; the index itself is the regulated unit. | S — per-tenant or per-region indexes are the canonical way to meet CMEK, residency, and data-isolation controls. |

Notes on reading the matrix:

- No pattern is strong on every axis. Picking honestly means
  trading freshness for isolation (projection), or isolation for
  freshness (query-time), or storage cost for simplicity
  (per-tenant indexes).
- "S" on **correctness** for pre-filter and projection is
  structural — the forbidden document never reaches the
  application layer, so a bug in the application cannot leak it.
  Post-filter's A is a reminder that the forbidden document
  passes through the server; a bug in the filter is a leak, not
  an empty response.
- The W cells on **freshness** for projection are the single most
  important operational fact in this exercise. If ACLs change
  often (contractor off-boarding, a document being reclassified
  mid-day), the projection pattern must be paired with either a
  near-real-time re-index path or a thin query-time veto layer.

### 2.3 `matrix/scenarios.md` — the three scenarios

#### Scenario A — SMB help-center (single tenant, public + customer-only docs)

A startup's support corpus has two classes of documents: public
marketing / FAQ pages, and customer-only articles that require a
signed-in user with an active subscription. ACLs change rarely
(when a customer churns). The retrieval service is a single
Python app in front of OpenSearch and a vector DB.

- **Pick: query-time pre-filter.**
- **Why:** ACL cardinality is tiny (two values: `public`,
  `customer`), the filter field slots directly into OpenSearch's
  `filter` clause and the vector DB's `where` clause, and the
  low ACL churn means a projection pattern would buy nothing.
  Post-filter would work but is strictly worse — it over-fetches
  and the risk of filter-bypass (e.g., on a "related articles"
  endpoint) is real for a small team. Per-tenant indexes would
  be operationally expensive for a two-value ACL.
- **Residual risk:** a bug in the filter expression (wrong field
  name, wrong caller identity) leaks customer docs to signed-out
  users. The mitigation is an end-to-end test asserting that
  anonymous callers *never* see a document with
  `visibility=customer`, and a periodic sweep that samples a
  signed-out query and asserts the invariant.

#### Scenario B — Mid-market knowledge base (per-user and per-group ACLs)

An HR / IT internal KB with per-folder permissions from the
source system (Confluence or SharePoint). A user has an identity,
a set of groups, and a set of explicit document grants. ACLs
change several times a day (joiners, movers, leavers). The
corpus is single-tenant.

- **Pick: query-time pre-filter with ACL predicate as a set-term
  query; keep a minimal index-time projection for the
  tenant/region partitioning.**
- **Why:** the allow-set (groups ∪ explicit grants) is small per
  user (typically < 200 terms) and fits in a `terms` filter on
  both the lexical and vector engines. Index-time projection is
  the wrong primary mechanism because ACL churn is faster than
  any reasonable reindex cadence — a revoked folder permission
  must be effective by the user's next query, not by the next
  index build. Post-filter is wrong because selective ACLs here
  shrink the authorised candidate set below k before the filter
  runs; the user sees fewer hits than they would have under
  pre-filter.
- **Residual risk:** stale ACL cache in the retrieval service —
  the service pulls the user's group memberships from the
  identity provider (Entra ID, Okta). If the cache TTL is
  longer than the revocation window the business expects,
  revocation is effectively lazy. Mitigation: bound the cache
  TTL to the business revocation window, and read the ACL at
  query time (not at session start) for high-sensitivity
  classes.

#### Scenario C — Regulated multi-tenant platform (CMEK, residency, strict isolation)

A multi-tenant retrieval platform where each customer runs under
customer-managed keys, data residency is pinned per tenant, and
contractual terms require that no tenant's data ever touches
another tenant's infrastructure. ACLs within a tenant are also
in play, but the first-level isolation is tenant.

- **Pick: per-tenant indexes (index-time projection) + per-tenant
  query-time filter for intra-tenant ACLs.**
- **Why:** CMEK, residency, and the contractual obligation are
  all index-level concerns. One index containing data from
  multiple tenants cannot be encrypted under different CMEKs or
  pinned to different regions. Per-tenant indexes make the
  isolation a routing decision at session init: the caller's
  tenant selects the index; a cross-tenant query is impossible
  without a routing bug. Within the tenant, per-user ACLs ride
  the Scenario B pattern.
- **Residual risk:** routing bugs. A misconfigured index alias
  or a missing tenant-id in the request context can route a
  caller to the wrong index. Mitigation: enforce
  tenant-id-to-index mapping at the ingress (not inside the
  retriever), and add an invariant test that cross-tenant
  queries fail closed with a structured error, not a silent
  empty result.

### 2.4 `matrix/decision_tree.md` — route scenario → pattern

```text
Q1. Is this a regulated / multi-tenant workload where the
    contract or compliance regime (CMEK, residency, sector-
    specific isolation) treats cross-tenant access as a
    first-order risk?
    └── Yes → per-tenant indexes (projection). Stop.
    └── No  → Q2.

Q2. Do ACLs change frequently (more than once per typical
    index-refresh interval)?
    └── Yes → query-time pre-filter (never post-filter when ACLs
              change fast — the filter is the ACL system of record
              at query time).
    └── No  → Q3.

Q3. Is the ACL field low-cardinality (e.g., a label or
    classification set with ≤ ~100 values) and does the
    engine's filter path support it efficiently?
    └── Yes → query-time pre-filter.
    └── No  → Q4.

Q4. Is the corpus small enough and the ACL check cheap enough
    that over-fetching and filtering locally is acceptable?
    └── Yes → query-time post-filter, *with* a fail-closed
              invariant test and an end-to-end leak canary.
    └── No  → index-time projection with a query-time veto
              overlay for revocations in the refresh window.

Default: when in doubt, prefer query-time pre-filter. It has the
fewest latent failure modes and is the easiest to reason about
in a review.
```

The tree collapses the matrix into the single question each
pattern answers best. The default fallback is deliberately
pre-filter, not post-filter — post-filter is strictly less safe
and should only be used when the operational constraints prevent
pre-filter from being implemented.

### 2.5 `notes/threat_model.md` — threat-model mapping

Each pattern leaves residual risks. The mapping below names them
in the vocabulary of the three frameworks the course uses, so a
reviewer can connect the engineering decision to the governance
regime.

**Residual risks by pattern**

| Pattern | Primary residual risk | Secondary residual risk |
|---|---|---|
| Post-filter | Filter bypass in a code path that forgot to call the filter (e.g., streaming endpoint, "related items" widget, cache-warm job). | Short-result degradation when ACLs are selective — the user sees fewer hits than exist and may conclude "nothing is there." |
| Pre-filter | ACL predicate injection — if the filter is built from a user-controlled string, the user can widen their own authorisation. | Stale group-membership cache if the service caches identity-provider lookups. |
| Projection / per-tenant | ACL staleness between source-of-truth revocation and next re-index. | Routing bug: wrong tenant-id resolves to wrong index. |

**Mapping to OWASP Machine Learning Security Top 10**
[owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/)

The OWASP ML Top 10 is organised around ML-specific failure
modes, not classical access-control; its direct coverage of ACL
enforcement in retrieval is partial. The two items most
applicable here are:

- **ML02 Data Poisoning** — in an enterprise retriever, the
  "training data" equivalent is the indexed corpus. A broken
  ingestion path that writes a document with the wrong ACL field
  (e.g., `visibility=public` where it should be
  `visibility=confidential`) is effectively poisoning the index
  against the authorisation policy. All three patterns are
  affected; pre-filter and projection make the blast larger
  because the index itself now vouches for the wrong ACL.
- **ML06 AI Supply Chain Attacks** — a compromised ingestion
  connector or a compromised identity-provider token can inject
  incorrect ACLs into the corpus or the user's session. The
  post-filter pattern depends on the live identity-provider
  response, so a compromised IdP leaks the corpus; the projection
  pattern depends on the ingestion pipeline, so a compromised
  connector persists the leak.

**Mapping to MITRE ATLAS**
[atlas.mitre.org/](https://atlas.mitre.org/)

MITRE ATLAS catalogues adversary tactics and techniques against
ML systems. The techniques most applicable to permission-aware
retrieval are the data-exfiltration and model-access families:

- *ML Model Access* — an unauthenticated or under-privileged
  caller probes the retriever to enumerate documents it should
  not see. Post-filter is vulnerable to timing side channels
  (did the call return in 20ms or 200ms? the gap may reveal
  whether a forbidden candidate was present). Pre-filter is
  structurally safer. Projection makes the probe meaningless —
  the index does not contain the forbidden documents.
- *Exfiltration via ML Inference API* — a legitimate caller
  abuses the retriever to exfiltrate authorised but
  classification-mismatched data (e.g., downloads every
  "customer" document one page at a time). All three patterns
  share this risk; the mitigation is not in the ACL pattern but
  in rate-limiting and audit logging, which is why the matrix
  scores **auditability** as a first-class axis.

**Mapping to NIST AI Risk Management Framework**
[nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)

The NIST AI RMF organises risk work under four functions:
*Govern, Map, Measure, Manage*. Each pattern lands differently
across the functions:

- **Govern** — who signs off on the ACL policy. In all three
  patterns, the governance boundary is the same (the policy
  owner in the source system). The pattern choice only affects
  *where the policy is enforced*, not *who sets it*.
- **Map** — what systems the retriever depends on for the ACL
  decision. Post-filter depends on the live IdP + a correct
  application filter; pre-filter depends on the live IdP + the
  engine's filter clause; projection depends on the ingestion
  pipeline + the index-to-tenant routing layer. A clean dependency
  map is part of the deliverable for a review.
- **Measure** — how leaks are detected. A canary of forbidden
  queries, an end-to-end test for "signed-out caller never sees
  internal doc," and a reconciliation of index ACL fields
  against the source of truth (for projection) are the three
  measurements the review should expect.
- **Manage** — how a revocation propagates. In all three
  patterns, the manage story needs an explicit revocation SLO:
  "from policy change to effective at query time, max N
  minutes." Pre-filter can hit sub-minute. Projection is bounded
  by the re-index cadence; a projection-only system cannot meet a
  one-minute SLO without an auxiliary query-time veto.

**Hybrid note.** The honest production answer is almost always
*hybrid*: per-tenant indexes for the isolation boundary +
query-time pre-filter for per-user ACLs within the tenant + a
thin veto layer for high-sensitivity revocations between
re-indexes. The matrix and the decision tree are tools for
picking the *primary* enforcement layer, not for excluding the
other layers from the design.

## 3. Validation steps

A correct submission can be validated by a reviewer without
running retriever code. The reviewer should be able to answer
"yes" to each of the following:

1. The matrix has three columns (the three patterns), ≥ ten
   rows, and every cell is scored with a one-sentence
   justification that cites either a concrete engine capability
   (e.g., Lucene `filter`, vector-DB `where`, tenant-scoped
   index) or a documented failure mode — not vendor marketing.
2. No cell scores **S** on an axis where the pattern is known to
   be weak (e.g., projection strong on freshness would be wrong;
   projection is structurally stale by its refresh cadence).
3. All three scenarios have a pick, a *why* that points at the
   matrix, and a named residual risk with a mitigation. The *why*
   survives the question "what would a reviewer say to challenge
   this pick?"
4. The decision tree routes each of the three scenarios to the
   pattern the scenario memo picks. If the tree disagrees with
   the memo, one of them is wrong.
5. The threat model names *specific* residual risks (filter
   bypass, injection, staleness, routing bug) and maps each to
   OWASP ML Top 10, MITRE ATLAS, and the NIST AI RMF functions.
   Mapping each risk to all three frameworks is the signal that
   the author has read the frameworks, not just cited them.
6. The hybrid note is present. A submission that claims one
   pattern wins outright on all three scenarios is probably
   wrong; enterprise retrieval almost always layers them.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Matrix completeness | ≥ 10 axes, 3 patterns, every cell scored with a one-line justification tied to a concrete mechanism. |
| Scenario coverage | All three scenarios (SMB, mid-market, regulated) are addressed with pick, why, and residual risk. |
| Consistency | The decision tree routes each scenario to the pattern the memo picks. |
| Correctness vs. recall trade-off | The matrix discusses recall under selective ACLs explicitly; post-filter's short-result failure mode is called out. |
| Freshness discussion | Projection's staleness and its mitigation (query-time veto, revocation SLO) are named. |
| Hybrid design acknowledged | The deliverable notes that the three patterns can be layered; the memo does not claim one pattern wins all scenarios. |
| Threat-model mapping | Residual risks are mapped to OWASP ML Top 10, MITRE ATLAS, and NIST AI RMF (at least one named item per framework). |
| No pattern tribalism | A reviewer cannot tell from the memo whether the author started with a preferred pattern. |
| Sources | Primary sources are the three listed governance frameworks and the engines' own documentation (vector DB, Lucene / Elasticsearch / OpenSearch). |
| Invariant tests named | The deliverable names at least one end-to-end invariant test per pattern (e.g., "anonymous caller never sees a document with `visibility=customer`"). |

Fail-fast smells:

- The memo picks post-filter for the regulated scenario. Post-
  filter cannot carry CMEK or residency; this is a disqualifying
  error.
- The memo claims projection has no staleness problem. All
  projection patterns have a non-zero revocation lag; if the
  author claims zero, they are not thinking about the write path.
- The matrix has no W cells. Every pattern has at least two
  weaknesses; an all-S matrix is dishonest.
- The threat model lists OWASP and MITRE but maps no specific
  items. Citing a framework is not using it; the mapping has
  to name techniques or items by identifier.
- The decision tree has 15 nodes. A decision tree that long will
  not be used; collapse it.

## 5. Common mistakes

- **Treating post-filter as "the safe default."** Post-filter is
  the *default* in many retrieval libraries because it is the
  easiest to implement, but it is strictly less safe than
  pre-filter — a missing filter call is a leak, not a bug. For
  new designs, pre-filter is the safer default.
- **Assuming the ANN pre-filter is free.** On HNSW and IVF
  indexes, filtered search degrades recall as the filter gets
  more selective. mod-104 covers the trade-off; the retriever's
  recall@k is not the same as its pre-filtered recall@k. The
  design must specify which it reports.
- **Encoding a per-user allow-set as a long `terms` list in every
  query.** Workable up to a few hundred terms; past that the
  query plan explodes. For high-cardinality per-user ACLs, move
  to an identity-indexed structure (e.g., a per-document
  `allowed_principal` set plus a terms-set query), or precompute
  group expansions.
- **Using a shared index for CMEK tenants.** Key management
  operates at the storage layer; a shared index cannot be
  encrypted under multiple keys. Per-tenant indexes are the
  canonical pattern.
- **Treating ACL fields as regular metadata.** ACL fields are
  the *trust boundary*. They should have the highest-level
  schema validation on write, should be immutable in the
  retriever's view (updates re-write the doc), and should be
  included in audit logs by default.
- **Reading the ACL at session start and caching for the session.**
  A one-hour session with a 10-minute revocation SLO leaks for
  50 minutes. Read the ACL at query time, or cap the cache TTL
  to the revocation SLO.
- **Forgetting the empty-result failure mode.** If a filter is
  broken "fail-closed," the user sees nothing and may think the
  retriever is broken, not that they are unauthorized. A
  reviewable system returns a structured "access denied" marker
  distinguishable from "no hits," so the audit log is honest.
- **Omitting the hybrid-retriever story.** A BM25 pipe and a
  dense pipe that each over-fetch and then fuse-then-filter is
  the worst of both worlds. The filter has to live in every
  pipe, or at the fusion input, or at the index boundary — and
  the design must say which.
- **Confusing tenancy with authorisation.** Tenancy is a
  coarse, rarely-changing partition; authorisation is a fine,
  often-changing predicate. They are enforced at different
  layers. A design that conflates them tends to pick projection
  for authorisation and then rebuild the index hourly.

## 6. References

Governance frameworks (primary sources for the threat-model
mapping):

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant items: ML02 Data Poisoning (misclassified ACL fields
  at ingest), ML06 AI Supply Chain Attacks (compromised
  connectors or IdP tokens).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant techniques: *ML Model Access* (probing / enumeration),
  *Exfiltration via ML Inference API* (bulk extraction by an
  authorised caller).
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant functions: *Govern* (who owns the policy),
  *Map* (what systems the retriever trusts for the ACL),
  *Measure* (canary + reconciliation), *Manage* (revocation SLO).

Engine documentation (used to populate the matrix cells on
filter paths and tenant isolation):

- Elasticsearch filter and `terms` queries —
  [elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html).
- OpenSearch search filtering —
  [opensearch.org/docs/latest/query-dsl/query-filter-context/](https://opensearch.org/docs/latest/query-dsl/query-filter-context/).
- Apache Lucene — official site
  [lucene.apache.org](https://lucene.apache.org/). Relevant:
  `BooleanQuery` filter clauses, `IndexSearcher#search(Query, Filter)` style
  filtering semantics.
- Model Context Protocol specification —
  [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification).
  Tool input schemas are the canonical boundary where ACL
  predicates should be declared for MCP-exposed retrievers.

Downstream consumers:

- exercise-02 (text-to-SQL) inherits the ACL pattern — the SQL
  guard in that exercise enforces the row-level security that
  this exercise authorises.
- exercise-03 (Graph RAG) inherits per-node and per-edge ACLs;
  the projection vs. pre-filter trade-off re-emerges for graph
  traversals.
- exercise-04 (multimodal) inherits the ACL field on documents;
  image / page hits carry the same ACL predicate as text hits.
- exercise-05 (hybrid router) depends on every underlying
  retriever applying the same ACL pattern — the router does not
  add authorisation; it assumes it.
