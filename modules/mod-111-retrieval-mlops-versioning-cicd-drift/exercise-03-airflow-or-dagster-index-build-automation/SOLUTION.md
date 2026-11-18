# Exercise 03 — Solution: Index-Build Automation with Airflow or Dagster

This is a reference solution for
`exercise-03-airflow-or-dagster-index-build-automation.md`. It is an
**implementation** exercise. The learner picks an orchestrator
(Airflow or Dagster) and wires the retrieval index build as a
scheduled, retriable, observable pipeline: ingest → chunk → embed →
index → eval → publish. The output is a new content-addressed
build id (exercise-01) that the cutover drill can promote.

The teaching point: *the index build is a data pipeline, not a
notebook*. Running it on a laptop makes four problems unavoidable —
non-reproducibility, no retries, no backfills, no provenance. An
orchestrator supplies all four, essentially for free, provided the
pipeline is written as a DAG of pure-ish tasks that take hashable
inputs and emit hashable outputs. The hard part is not the DAG
syntax; it is designing the tasks to be *idempotent* and
*content-addressed*, so that a rerun never duplicates work and a
backfill always converges.

## 1. Solution overview

The deliverable has five parts:

1. **`dags/build_index_airflow.py`** — the Airflow DAG (if Airflow
   is chosen). Tasks: `snapshot_corpus`, `plan_batches`,
   `embed_batch` (dynamic task mapping), `assemble_vectors`,
   `build_index`, `write_manifest`, `offline_eval`,
   `register_build`. Scheduled nightly with a manual-trigger path.
2. **`dagster/build_index_assets.py`** — the Dagster alternative:
   five software-defined assets (`corpus_snapshot`,
   `embedding_shards`, `vector_index`, `manifest`, `eval_report`),
   with a daily schedule and a sensor on new corpus commits.
3. **`src/tasks/*.py`** — the task implementations themselves,
   written as plain functions that both orchestrators call. This
   keeps the business logic out of orchestrator-specific abstract
   classes and makes the code testable with `pytest` directly.
4. **`infra/README.md`** — the deployment story: where the
   orchestrator runs, how secrets are mounted, how worker pools are
   sized, how the DAG consumes exercise-01's naming scheme.
5. **`eval/BUDGET.md`** — the cross-reference to exercise-02's
   gate so that `offline_eval` is the exact same code path as the
   CI gate (no "staging" vs "CI" eval divergence).

Scope: *the automation plane* — scheduling, retries, dynamic
fan-out, dependency tracking, observability. The index-naming
contract is exercise-01, the eval gate is exercise-02, the drift
monitor that may *trigger* a rebuild is exercise-06.

## 2. Worked answer or implementation

### 2.1 Picking Airflow vs. Dagster

Short, honest matrix for retrieval index builds specifically:

| Axis | Airflow | Dagster |
|---|---|---|
| **Mental model** | DAG of tasks; outputs are implicit side effects. | DAG of **assets**; outputs are first-class. Fits "an index is a thing we produce" cleanly. |
| **Dynamic fan-out** | Dynamic task mapping (`.expand`) since 2.3. Works. | Partitioned assets + dynamic partitions. Ergonomic. |
| **Backfill** | Catchup from a `start_date`, or `airflow dags backfill`. | First-class partitioned backfills with a UI selector. |
| **Lineage** | XCom + logs; lineage is possible via OpenLineage integration. | Lineage is the default view (assets → upstreams). |
| **Running cost** | Scheduler + webserver + workers; `LocalExecutor` works for small shops, Celery/K8s for bigger. | Daemon + web server + code locations; lighter for small shops. |
| **Ecosystem** | Larger, older; most on-prem ops teams already run it. | Smaller, newer, cleaner API. First-class asset materialization semantics. |
| **Fit for this exercise** | Fine. Dynamic mapping covers the embedding fan-out. | Slightly better fit — the "asset" model matches the "content-addressed index" model exactly. |

The solution below shows **both**. In practice pick the one your
org already runs. If this is greenfield and the pipeline is *only*
RAG/ML artifacts, Dagster's asset model pays for itself. If the
pipeline shares a data platform with an existing Airflow install,
Airflow is the right default.

### 2.2 Task design (orchestrator-agnostic)

All business logic lives in `src/tasks/`. Each function:

- Takes **hashable, serializable** inputs (strings, ints, small
  dicts). No in-memory objects crossing task boundaries.
- Returns a **content-addressed artifact URI**, not the artifact
  itself. The artifact lives in object storage keyed by its
  content hash; the URI is what the next task reads.
- Is **idempotent**: calling it twice with the same inputs is a
  no-op on the second call. Implemented by checking "does
  `artifact_uri` already exist in object storage?" before
  producing it.

```python
# src/tasks/snapshot_corpus.py
"""Walk the source and write a corpus manifest; return the manifest URI."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.io import object_store  # local-filesystem or S3 wrapper


@dataclass(frozen=True)
class CorpusSnapshot:
    corpus_sha: str
    manifest_uri: str
    doc_count: int


def _sha256_stream(it) -> str:
    h = hashlib.sha256()
    for chunk in it:
        h.update(chunk)
    return h.hexdigest()


def snapshot_corpus(source_uri: str, bucket: str) -> CorpusSnapshot:
    """Enumerate docs under `source_uri`, hash each, write manifest.

    Deterministic: same source + same bytes → same corpus_sha.
    """
    rows = []
    for doc in object_store.walk(source_uri):
        with object_store.open(doc.uri, "rb") as fh:
            content_sha = _sha256_stream(iter(lambda: fh.read(8192), b""))
        rows.append({
            "source_uri":    doc.uri,
            "content_sha256": content_sha,
            "mtime":          doc.mtime.isoformat(),
            "size":           doc.size,
        })
    rows.sort(key=lambda r: (r["source_uri"], r["content_sha256"]))
    manifest_bytes = ("\n".join(json.dumps(r, sort_keys=True) for r in rows)).encode()
    corpus_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_uri = f"s3://{bucket}/corpora/{corpus_sha}/manifest.jsonl"
    if not object_store.exists(manifest_uri):
        object_store.write_bytes(manifest_uri, manifest_bytes)
    return CorpusSnapshot(corpus_sha, manifest_uri, len(rows))
```

```python
# src/tasks/plan_batches.py
"""Split the corpus manifest into fixed-size embedding batches."""
from __future__ import annotations

import json

from src.io import object_store


def plan_batches(manifest_uri: str, batch_size: int = 512) -> list[str]:
    """Return a list of batch-manifest URIs, one per embedding batch.

    Deterministic batching: we sort by `content_sha256` so the same
    corpus always produces the same batches, which is what makes
    `embed_batch` idempotent.
    """
    lines = object_store.read_text(manifest_uri).splitlines()
    rows = [json.loads(line) for line in lines]
    rows.sort(key=lambda r: r["content_sha256"])

    batches: list[str] = []
    bucket = manifest_uri.rsplit("/corpora/", 1)[0]
    corpus_sha = manifest_uri.rsplit("/corpora/", 1)[1].split("/", 1)[0]

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        batch_body = "\n".join(json.dumps(r, sort_keys=True) for r in batch).encode()
        batch_sha  = __import__("hashlib").sha256(batch_body).hexdigest()
        uri = f"{bucket}/corpora/{corpus_sha}/batches/{batch_sha}.jsonl"
        if not object_store.exists(uri):
            object_store.write_bytes(uri, batch_body)
        batches.append(uri)
    return batches
```

```python
# src/tasks/embed_batch.py
"""Embed one batch of documents; write vector rows keyed by (batch_sha, model_rev).

Idempotent: re-running with the same batch + same model returns the
cached vector file without any provider calls.
"""
from __future__ import annotations

import json

from src.embeddings import load_provider
from src.io import object_store
from src.chunker import load as load_chunker


def embed_batch(
    batch_uri: str,
    model_id: str,
    model_revision: str,
    chunker_cfg_sha: str,
    index_bucket: str,
) -> str:
    out_uri = f"{index_bucket}/embeddings/{model_id}-{model_revision}/{chunker_cfg_sha}/{__basename(batch_uri)}.vectors.jsonl"
    if object_store.exists(out_uri):
        return out_uri  # idempotent short-circuit

    provider = load_provider(model_id, revision=model_revision)
    chunker  = load_chunker(chunker_cfg_sha)

    rows = []
    for raw_line in object_store.read_text(batch_uri).splitlines():
        doc  = json.loads(raw_line)
        text = object_store.read_text(doc["source_uri"])
        for chunk in chunker.split(text, source_uri=doc["source_uri"]):
            rows.append({
                "doc_id":    f"{doc['content_sha256']}:{chunk.ordinal}",
                "vector":    provider.embed(chunk.text),
                "metadata": chunk.metadata,
            })

    body = "\n".join(json.dumps(r) for r in rows).encode()
    object_store.write_bytes(out_uri, body)
    return out_uri


def __basename(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
```

```python
# src/tasks/build_index.py
"""Build the vector index from the embedding shards; write the artifact."""
from __future__ import annotations

from src.indexers import build  # dispatches on engine (qdrant, pgvector, opensearch, ...)
from src.io import object_store


def build_index(
    vector_uris: list[str],
    build_id: str,
    engine: str,
    index_cfg: dict,
    index_bucket: str,
) -> str:
    """Materialize the index under `index_bucket/indexes/<build_id>/` and
    return its URI. Idempotent: if the target already exists, skip."""
    out_uri = f"{index_bucket}/indexes/{build_id}/"
    if object_store.exists(out_uri + "SUCCESS"):
        return out_uri
    build(
        engine=engine,
        index_cfg=index_cfg,
        inputs=vector_uris,
        output_uri=out_uri,
    )
    object_store.write_bytes(out_uri + "SUCCESS", b"ok")
    return out_uri
```

The `write_manifest`, `offline_eval`, and `register_build` tasks
follow the same shape: small, idempotent, content-addressed.
`offline_eval` imports exactly the same code as the exercise-02 CI
gate — no "staging eval" divergence. `register_build` appends a row
to a `builds` table (DynamoDB / Postgres) that exercise-01's
cutover drill reads to find candidate indexes.

### 2.3 `dags/build_index_airflow.py` — Airflow DAG

```python
"""Airflow DAG: nightly index build with dynamic embedding fan-out."""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator

from src.tasks.snapshot_corpus import snapshot_corpus
from src.tasks.plan_batches   import plan_batches
from src.tasks.embed_batch    import embed_batch
from src.tasks.build_index    import build_index
from src.tasks.write_manifest import write_manifest
from src.tasks.offline_eval   import offline_eval
from src.tasks.register_build import register_build
from src.config               import load_retrieval_config

CFG = load_retrieval_config()  # read-only; config_sha computed here

default_args = {
    "owner": "retrieval-platform",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=60),
    "email_on_retry": False,
    "email_on_failure": True,
    "sla": timedelta(hours=4),
}

with DAG(
    dag_id="build_retrieval_index",
    description="Snapshot corpus → embed → index → eval → publish build.",
    schedule="0 2 * * *",                # 02:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,                   # no overlapping runs
    tags=["retrieval", "mlops"],
    default_args=default_args,
) as dag:

    @task(task_id="snapshot_corpus")
    def _snapshot() -> dict:
        snap = snapshot_corpus(CFG.source_uri, CFG.index_bucket)
        return snap.__dict__

    @task(task_id="plan_batches")
    def _plan(snap: dict) -> list[str]:
        return plan_batches(snap["manifest_uri"], batch_size=CFG.embed_batch_size)

    @task(task_id="embed_batch", retries=5, retry_delay=timedelta(minutes=2))
    def _embed(batch_uri: str) -> str:
        return embed_batch(
            batch_uri,
            model_id=CFG.embedding.model_id,
            model_revision=CFG.embedding.revision,
            chunker_cfg_sha=CFG.chunker.cfg_sha,
            index_bucket=CFG.index_bucket,
        )

    @task(task_id="build_index", execution_timeout=timedelta(hours=2))
    def _build(vector_uris: list[str], snap: dict) -> dict:
        build_id = CFG.compose_build_id(snap["corpus_sha"])
        index_uri = build_index(
            vector_uris=vector_uris,
            build_id=build_id,
            engine=CFG.index.engine,
            index_cfg=CFG.index.params,
            index_bucket=CFG.index_bucket,
        )
        return {"build_id": build_id, "index_uri": index_uri, "corpus_snapshot": snap}

    @task(task_id="write_manifest")
    def _manifest(built: dict) -> str:
        return write_manifest(
            build_id=built["build_id"],
            index_uri=built["index_uri"],
            corpus_snapshot=built["corpus_snapshot"],
            cfg=CFG,
        )

    @task(task_id="offline_eval")
    def _eval(built: dict) -> dict:
        return offline_eval(
            build_id=built["build_id"],
            index_uri=built["index_uri"],
            qrels_uri=CFG.eval.qrels_uri,
            queries_uri=CFG.eval.queries_uri,
        )

    @task(task_id="register_build")
    def _register(built: dict, manifest_uri: str, eval_report: dict) -> None:
        register_build(
            build_id=built["build_id"],
            manifest_uri=manifest_uri,
            eval_report=eval_report,
        )

    snap        = _snapshot()
    batch_uris  = _plan(snap)
    vector_uris = _embed.expand(batch_uri=batch_uris)   # <-- dynamic fan-out
    built       = _build(vector_uris, snap)
    manifest    = _manifest(built)
    eval_report = _eval(built)
    done        = _register(built, manifest, eval_report)

    EmptyOperator(task_id="start") >> snap
    done >> EmptyOperator(task_id="finish")
```

Notes on design:

- `.expand(batch_uri=batch_uris)` is dynamic task mapping (Airflow
  ≥ 2.3; docs:
  [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html)).
  Each batch gets its own task instance, its own log, its own
  retry history. A single poison-pill document fails one task,
  not the whole run.
- `max_active_runs=1` guarantees no overlapping index builds. Two
  concurrent builds against the same corpus are wasted work and
  can race on the manifest write.
- `retries=5` on `embed_batch` because the provider (OpenAI, Azure
  OpenAI, Cohere) is the most common transient-failure source;
  `retries=3` everywhere else.
- `sla=timedelta(hours=4)` so a stuck run is surfaced rather than
  running indefinitely. The SLA callback is wired to the on-call
  channel.
- The DAG passes dicts (not Python objects) across tasks so the
  XCom serialization is JSON-safe.

### 2.4 `dagster/build_index_assets.py` — Dagster alternative

```python
"""Dagster: five assets whose materialization produces a new build."""
from __future__ import annotations

import json

from dagster import (
    AssetKey,
    AssetExecutionContext,
    AssetOut,
    DailyPartitionsDefinition,
    Definitions,
    MetadataValue,
    MultiPartitionsDefinition,
    ScheduleDefinition,
    StaticPartitionsDefinition,
    asset,
    multi_asset,
    DynamicPartitionsDefinition,
    define_asset_job,
)

from src.tasks.snapshot_corpus import snapshot_corpus
from src.tasks.plan_batches   import plan_batches
from src.tasks.embed_batch    import embed_batch
from src.tasks.build_index    import build_index
from src.tasks.write_manifest import write_manifest
from src.tasks.offline_eval   import offline_eval
from src.config               import load_retrieval_config

CFG = load_retrieval_config()

daily = DailyPartitionsDefinition(start_date="2026-01-01")
batches_partitions = DynamicPartitionsDefinition(name="embed_batch_shards")


@asset(partitions_def=daily, compute_kind="python")
def corpus_snapshot(context: AssetExecutionContext) -> dict:
    snap = snapshot_corpus(CFG.source_uri, CFG.index_bucket)
    context.add_output_metadata({
        "corpus_sha":   MetadataValue.text(snap.corpus_sha),
        "manifest_uri": MetadataValue.url(snap.manifest_uri),
        "doc_count":    MetadataValue.int(snap.doc_count),
    })
    return snap.__dict__


@asset(partitions_def=daily, compute_kind="python", deps=[corpus_snapshot])
def embedding_shards(
    context: AssetExecutionContext, corpus_snapshot: dict,
) -> list[str]:
    batch_uris  = plan_batches(corpus_snapshot["manifest_uri"], batch_size=CFG.embed_batch_size)
    vector_uris: list[str] = []
    for i, bu in enumerate(batch_uris):
        with context.op_execution_context.log.span(f"embed_batch_{i}"):
            vector_uris.append(embed_batch(
                bu,
                model_id=CFG.embedding.model_id,
                model_revision=CFG.embedding.revision,
                chunker_cfg_sha=CFG.chunker.cfg_sha,
                index_bucket=CFG.index_bucket,
            ))
    context.add_output_metadata({
        "n_shards": len(vector_uris),
        "model":    f"{CFG.embedding.model_id}@{CFG.embedding.revision}",
    })
    return vector_uris


@asset(partitions_def=daily, compute_kind="python")
def vector_index(
    context: AssetExecutionContext, corpus_snapshot: dict, embedding_shards: list[str],
) -> dict:
    build_id = CFG.compose_build_id(corpus_snapshot["corpus_sha"])
    index_uri = build_index(
        vector_uris=embedding_shards,
        build_id=build_id,
        engine=CFG.index.engine,
        index_cfg=CFG.index.params,
        index_bucket=CFG.index_bucket,
    )
    context.add_output_metadata({
        "build_id": build_id, "index_uri": MetadataValue.url(index_uri),
    })
    return {"build_id": build_id, "index_uri": index_uri,
            "corpus_snapshot": corpus_snapshot}


@asset(partitions_def=daily, compute_kind="python")
def manifest(context: AssetExecutionContext, vector_index: dict) -> str:
    uri = write_manifest(**vector_index, cfg=CFG)
    context.add_output_metadata({"manifest_uri": MetadataValue.url(uri)})
    return uri


@asset(partitions_def=daily, compute_kind="python")
def eval_report(context: AssetExecutionContext, vector_index: dict) -> dict:
    rep = offline_eval(
        build_id=vector_index["build_id"],
        index_uri=vector_index["index_uri"],
        qrels_uri=CFG.eval.qrels_uri,
        queries_uri=CFG.eval.queries_uri,
    )
    context.add_output_metadata({
        "recall_at_10": MetadataValue.float(rep["metrics"]["R@10"]),
        "ndcg_at_10":   MetadataValue.float(rep["metrics"]["nDCG@10"]),
    })
    return rep


index_build_job = define_asset_job(
    "index_build",
    selection=[corpus_snapshot, embedding_shards, vector_index, manifest, eval_report],
    partitions_def=daily,
)

defs = Definitions(
    assets=[corpus_snapshot, embedding_shards, vector_index, manifest, eval_report],
    jobs=[index_build_job],
    schedules=[ScheduleDefinition(job=index_build_job, cron_schedule="0 2 * * *")],
)
```

Notes:

- Daily partitions give a free backfill UI: "re-materialize
  `vector_index` for 2026-03-01" is a click, not a code change.
- Metadata on each materialization (`build_id`, `recall_at_10`,
  etc.) surfaces in the Dagster UI and in OpenLineage emissions.
- The same `src/tasks/*.py` functions run under both orchestrators
  — the orchestration code is thin.

### 2.5 `infra/README.md` — deployment

Required minimums:

- **Secrets.** Embedding-provider API keys mounted via the
  orchestrator's secret backend (Airflow Variables backed by AWS
  Secrets Manager / GCP Secret Manager; Dagster
  `EnvVar`/resources with the same backend). Never in plaintext
  in the DAG.
- **Workers.** Two pools: `retriever-cpu` for the orchestrator
  bookkeeping tasks and `retriever-embed` for `embed_batch` (the
  only task that spends real money and real wall-clock). Size the
  embed pool to the embedding-provider's rate limit, not the
  CPU count.
- **Permissions.** Workers have read access to the corpus bucket,
  write access to the index bucket, and no access to anything
  else. OWASP ML Top 10 **ML06 AI Supply Chain Attacks** names
  build-time credential scope as a core control.
- **Observability.** Export task duration, failure, and retry
  counts to the metrics backend. Alert on: DAG SLA miss,
  `embed_batch` failure rate > 5%, `offline_eval` metrics drop
  (same thresholds as exercise-02).
- **Backfill policy.** Catchup off by default (`catchup=False` /
  Dagster default). Explicit backfills are a human decision:
  "rebuild 2026-02-14 through 2026-02-21 to pick up the
  corpus-dedup fix in that window."
- **Concurrency.** `max_active_runs=1` (Airflow) /
  `max_concurrent=1` on the job (Dagster). Overlapping index
  builds race on the manifest and waste provider spend.

### 2.6 Trigger surface

The DAG must be triggerable from three sources:

1. **Schedule** — nightly at 02:00 UTC. Catches routine corpus
   updates.
2. **Manual** — operator-initiated for emergency rebuilds (e.g.,
   after a corpus-dedup fix). Both orchestrators ship this in the
   UI.
3. **Event** — a webhook or sensor from the corpus side. In
   Airflow, an `ExternalTaskSensor` or a `TriggerDagRunOperator`
   from an upstream ingestion DAG; in Dagster, a sensor on the
   corpus object-store prefix:

```python
from dagster import sensor, SensorEvaluationContext, RunRequest, SkipReason
from src.io import object_store

@sensor(job=index_build_job, minimum_interval_seconds=300)
def corpus_changed_sensor(context: SensorEvaluationContext):
    cursor = context.cursor or "0"
    latest = str(object_store.last_modified(CFG.source_uri).timestamp())
    if latest == cursor:
        yield SkipReason("no new corpus commits")
        return
    context.update_cursor(latest)
    yield RunRequest(run_key=f"corpus-{latest}")
```

The event path is what makes the system react to *why* a rebuild
is needed (new docs, drift alert from exercise-06) rather than to
the clock.

### 2.7 How this feeds the cutover drill

`register_build` writes a row like:

```json
{
  "build_id":     "docs-support-prod_corpus-7f3a..._cfg-ab41..._openai-text-embedding-3-large-20240125_20260311T0204Z",
  "manifest_uri": "s3://.../manifests/docs-support-prod_corpus-7f3a..._.json",
  "index_uri":    "s3://.../indexes/docs-support-prod_corpus-7f3a..._.../",
  "eval":         {"R@10": 0.76, "nDCG@10": 0.54, "RR@10": 0.61},
  "status":       "built"
}
```

Exercise-01's cutover drill reads `builds` where `status = 'built'`
and selects the newest build whose eval passes the gate. The
operator then kicks off Phases 3–6 (mirror → gate → alias flip →
bake).

## 3. Validation steps

A correct submission can be validated with these checks:

1. **Idempotency on rerun.** Run the DAG once. Clear the
   `offline_eval` task only and rerun. `embed_batch` must be
   skipped (either shown as "not re-run" by the orchestrator, or
   short-circuit inside the function based on object-store
   existence). Provider spend on the second run is $0.
2. **Idempotency on backfill.** Backfill a range that includes
   today's partition. The system does not produce a new build id
   if inputs are identical; the existing build is reused.
3. **Partial failure recovery.** Inject a 500 response on the
   first `embed_batch` task. Retry 1 succeeds. The DAG completes.
   Only one task instance's logs show the retry; no upstream
   tasks rerun.
4. **Dynamic fan-out size.** With a 10k-doc corpus and
   `batch_size=512`, the `embed_batch` task group shows ~20
   mapped instances. Reducing `batch_size` to 128 bumps it to
   ~80. (Airflow dynamic-mapping scales per docs:
   [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html).)
5. **Manifest completeness.** After a successful run, the
   manifest at `manifests/<build_id>.json` validates against
   exercise-01's schema (`jsonschema -i ...`). Every field is
   populated; `eval` sub-object is present with the numbers.
6. **Lineage.** The orchestrator UI shows the full
   `corpus → batches → vectors → index → manifest → eval` lineage.
   Dagster asset graph renders five nodes; Airflow DAG view
   renders the task graph with dynamic-mapped `embed_batch`
   flattened appropriately.
7. **Determinism.** Two runs against the same corpus bytes + the
   same configs produce the same `build_id`. Compare: `aws s3 ls
   s3://.../indexes/` shows one directory for that build id, not
   two.
8. **Rate-limit respect.** With a 500-req/min provider limit, the
   worker pool is sized such that concurrent `embed_batch`
   instances × internal batch size never exceeds that limit.
   Observe no 429 responses during a full run.
9. **Trigger surface.** Manual trigger from the UI works;
   schedule fires at 02:00 UTC; the event sensor fires within
   5 minutes of a new corpus commit.
10. **Scoped permissions.** Attempt a `aws s3 rm
    s3://<corpus-bucket>/<key>` from a worker pod — denied by
    IAM. Attempt a write to a bucket outside the index bucket —
    denied. Build credentials are scoped.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Task granularity | ingest, chunk/embed, index, manifest, eval, register are separate tasks/assets — not one monolithic op. |
| Idempotent tasks | Every task short-circuits when its output URI already exists in object storage. |
| Content-addressed artifacts | Corpus manifest, embedding shards, index, manifest are keyed by content hash — not by run id or timestamp. |
| Dynamic fan-out | Embedding fan-out uses the orchestrator's dynamic-mapping / partition primitive, not a single-task loop. |
| Retries + backoff | Transient-failure tasks (API calls) have retries with exponential backoff; terminal failure escalates. |
| Concurrency bound | `max_active_runs=1` or equivalent prevents overlapping runs. |
| Rate-limit respect | Worker pool sizing accounts for the embedding provider's rate limit. |
| Secrets via backend | No plaintext keys in the DAG; backend is one of AWS/GCP/Vault. |
| Scoped IAM | Build credentials have only the access they need; no `s3:*`-on-all-buckets grants. |
| Observability | Metadata on each materialization (counts, sha, eval numbers) is attached; the orchestrator UI surfaces it. |
| SLA / alerting | SLA miss or task failure pages on-call; eval-regression fires the exercise-02 budget. |
| Trigger diversity | Schedule + manual + event trigger all work; event trigger is the preferred path for drift-driven rebuilds. |
| Lineage visible | Upstream → downstream dependency is a first-class view, not inferable from logs. |
| Uses exercise-01 naming | Build id matches the content-addressed scheme; no `v3`/`fixed`/`new` suffixes anywhere. |
| Uses exercise-02 eval | `offline_eval` imports the exact code path the CI gate imports; no divergence. |

Fail-fast smells:

- One `BashOperator` that runs `python build_everything.py`.
  Not a DAG — an orchestration-shaped shell wrapper. No retries,
  no lineage, no fan-out.
- Embedding fan-out written as a `for` loop inside a single task.
  One flaky document fails the whole task; retries re-embed the
  other 9,999.
- Secrets passed as environment variables at DAG parse time.
  Leaks into the webserver logs; rotates become a redeploy.
- Build id includes a wall-clock timestamp in its *identity* field.
  Two runs on the same inputs produce different build ids; the
  cutover can't tell they are equivalent.
- Eval is a notebook the author runs after the DAG finishes.
  Not in the pipeline; drift in eval vs CI gate guaranteed.
- `catchup=True` on the first DAG parse. The scheduler launches
  one run per day since `start_date` — months of accidental
  provider spend.

## 5. Common mistakes

- **Treating the DAG as a script runner.** The whole point of an
  orchestrator is dependency tracking + retries + lineage. If
  every task is `bash run_script.sh` with its own retries and
  logging inside, you have a scheduler running shell scripts, not
  a DAG.
- **Non-idempotent tasks.** An `embed_batch` that re-embeds on
  rerun wastes real money and prevents backfills. The idempotency
  check is five lines (`if exists(uri): return uri`).
- **Hashing on run id instead of content.** Using `run_id` or
  `logical_date` in artifact paths means every run produces new
  artifacts even when the inputs are identical. Can't short-
  circuit; can't deduplicate across backfills.
- **Dynamic fan-out as a for-loop.** One task, iterating, burns
  the dynamic-mapping benefit (per-item retries, per-item logs)
  and typically hits a task-level timeout on large corpora.
- **No rate-limit awareness.** Fanning out to 500 concurrent
  `embed_batch` tasks on a 100-req/min API fails almost every
  task. Worker pool size × per-task concurrency must be under
  the provider's limit.
- **No SLA.** A hung embedding call can wedge the DAG for hours
  before anyone notices. Set `execution_timeout` on long tasks
  and `sla` on the DAG.
- **Plaintext credentials.** OWASP ML Top 10 **ML06** explicitly
  calls out credential scope as a supply-chain control.
- **Mixing eval code paths.** An "orchestrator eval" that is
  different from the "CI eval" guarantees that the gate and the
  pipeline disagree. Share the import path; share the qrels.
- **Rebuilding on wall-clock schedule only.** A nightly rebuild
  will miss urgent corpus changes and will waste money on nights
  when nothing changed. Add the event trigger.
- **Letting the webserver run as root in the worker.** The build
  worker should be the least-privileged component. A compromised
  embedding call should not be able to list the whole bucket.

## 6. References

Orchestrator documentation (primary):

- Airflow — [airflow.apache.org/docs](https://airflow.apache.org/docs/).
  Relevant topics: TaskFlow API, Dynamic Task Mapping, SLAs,
  Executors, Secrets Backends. Dynamic Task Mapping: see
  [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/dynamic-task-mapping.html).
- Dagster — [docs.dagster.io](https://docs.dagster.io/).
  Relevant topics: Assets, Partitions, Sensors, Schedules,
  Dynamic Partitions, IO Managers. Assets:
  [Software-defined assets](https://docs.dagster.io/concepts/assets/software-defined-assets).

Supporting tools:

- `ir_measures` — [ir-measur.es](https://ir-measur.es/).
  Used by the `offline_eval` task to score the build (shared with
  exercise-02).
- OpenLineage — [openlineage.io](https://openlineage.io/).
  Lineage emission from Airflow and Dagster is standardized here;
  turns task runs into cross-system lineage events.

Governance framings:

- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: **MANAGE 4.1** (provenance records), **GOVERN 1.5**
  (access-control policies on AI components), **MEASURE 2.6**
  (post-deployment evaluation).
- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: **ML06 AI Supply Chain Attacks** — least-privilege
  build credentials, signed manifests from exercise-01.
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: **AML.T0010 ML Supply Chain Compromise** — the
  automated build pipeline is part of that chain.

Related exercises:

- exercise-01 — the build-id contract and cutover drill this DAG
  feeds.
- exercise-02 — the eval gate the `offline_eval` task reuses.
- exercise-06 — the drift monitor whose alerts trigger ad-hoc
  rebuilds through the event path in §2.6.
