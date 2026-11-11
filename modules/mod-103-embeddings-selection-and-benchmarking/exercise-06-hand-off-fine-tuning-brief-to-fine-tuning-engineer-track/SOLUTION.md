# Exercise 06 — Solution: Hand-off — the fine-tuning brief to the fine-tuning engineer

This is a reference solution for
`exercise-06-hand-off-fine-tuning-brief-to-fine-tuning-engineer-track.md`. It is a
**design + documentation** exercise. The retrieval engineer has run exercises 02–05,
has domain-fit numbers and a compression recommendation, and now must decide whether
to escalate to fine-tuning — and if so, hand a crisp brief to the fine-tuning track so
they do not have to re-derive the same facts. The reference below is a worked brief
template, a decision rationale for *when* to fine-tune at all, and a rubric.

## 1. Solution overview

The deliverable has three parts:

1. **`notes/to-finetune-or-not.md`** — the decision to escalate. Fine-tuning an
   embedder is an expensive commitment (training data labor, ongoing re-training
   cadence, a second eval harness); it should happen only when the measured gap
   between the best off-the-shelf model (exercise-04) and the quality target cannot
   be closed by hybrid retrieval (mod-106) or re-ranking (mod-107). This note states
   that gap and names the quality target.
2. **`briefs/finetune-brief-<domain>.md`** — the hand-off itself. A fixed section
   schema the fine-tuning engineer can read in five minutes: objective, data
   contract, eval contract, serving contract, risk contract.
3. **`briefs/training-pairs.jsonl.schema.json`** — the JSON Schema for the training
   pairs file the retrieval team will hand over. Nailing the schema down before the
   fine-tuning engineer is on the hook is how the hand-off avoids a six-week
   round-trip over "we expected `positive_passage`, not `positive`."

The exercise is not "write a fine-tuning recipe." That belongs to the fine-tuning
track. The exercise is "write the brief that lets the fine-tuning track do their job
without guessing at the retrieval stack's requirements."

## 2. Worked answer or implementation

### 2.1 `notes/to-finetune-or-not.md` — the decision rationale

Fine-tune an embedder when **all four** of the following hold:

1. **A measured domain gap.** Exercise-04 shows the best off-the-shelf model is below
   the retrieval quality target by a margin that matters — typically ≥ 5 NDCG@10
   points on the domain eval set. Name the target and the gap, with numbers.
2. **Hybrid + reranker are not enough.** Before approving fine-tuning, the team has
   layered BM25 + dense hybrid fusion (mod-106) and a cross-encoder re-ranker
   (mod-107) over the best off-the-shelf model. If those together close the gap,
   fine-tuning is the wrong tool — operate the simpler stack.
3. **Labeled or labelable pairs exist.** Fine-tuning a bi-encoder with contrastive
   learning needs query–positive pairs (and optionally hard negatives). If the
   organization has no path to generating or extracting these from logs / tickets /
   the SME, the project will stall at week two. Confirm the data path before the
   training compute path.
4. **Someone owns re-tuning.** Embedder fine-tunes drift as the corpus and query
   distribution drift. If no one on the roadmap owns the re-tune cadence, you are
   creating a model that will silently degrade.

If any of the four is missing, do **not** escalate. Operate the hybrid + reranker
stack and revisit the gap next quarter.

The decision doc records the numbers that make the escalation defensible:

- Off-the-shelf winner from exercise-04: *<model name>*, NDCG@10 = *<x.xxx>* on
  *<eval-set-size>* queries.
- Measured gap vs. quality target: *<y.yyy>* NDCG@10 points.
- Hybrid + reranker attempt: NDCG@10 = *<z.zzz>* — closed/did-not-close the gap.
- Data path: *<N>* candidate query–positive pairs available, source *<...>*.

### 2.2 `briefs/finetune-brief-<domain>.md` — the hand-off brief

```markdown
# Fine-tuning brief: <domain>

Owner (retrieval side): <name>, <role>
Intended owner (fine-tuning side): <team or name>
Hand-off date: <YYYY-MM-DD>
Status: draft | approved | in-flight | delivered

## 1. Objective

Produce a bi-encoder embedder fine-tuned on `<domain>` that improves NDCG@10 on the
`<domain>` eval set from `<baseline>` to at least `<target>` while preserving ≥ 90 %
of the best off-the-shelf model's NDCG@10 on the MTEB retrieval subset we care about
(so the fine-tuned model is not catastrophically specialized).

## 2. Data contract — what retrieval ships

### 2.1 Training pairs

- Source: `briefs/training-pairs.jsonl.gz` (schema in
  `briefs/training-pairs.jsonl.schema.json`).
- Count: `<N>` query–positive pairs, `<M>` of them with at least one hard negative.
- Provenance: `<link to the extraction pipeline / notebook>`. Must include the SHA256
  of the input logs or the dataset snapshot id.
- PII handling: `<policy>` — fields redacted / hashed. Names of the redaction tool and
  its version are in the brief, not a comment.
- License / data governance: `<explicit allow-list for training use>`. If any pair
  originated in a dataset with a non-commercial license, it is not in this file.

### 2.2 Eval set

- Source: `data/evalset/` (same BEIR-style triple from exercise-04).
- Size: `<Q>` queries, `<L>` relevance judgments.
- Split: `<train/dev/test proportion>`. The eval set used for the quality target in
  section 1 is the *test* split, and the fine-tuning engineer must not touch it until
  the final evaluation.

### 2.3 MTEB-subset we care about

- List of MTEB retrieval tasks that must *not* regress by more than 10 % NDCG@10
  relative to the baseline model. The retrieval team names these; the fine-tuning
  team evaluates against them.
- Example list (project-specific): `SciFact`, `NFCorpus`, `FiQA2018` — pick the
  tasks whose domains resemble the organization's corpus.

## 3. Eval contract — how we judge success

- Primary metric: NDCG@10 on the `<domain>` test split, with the eval set frozen as
  of the hand-off date.
- Secondary metrics: Recall@10 on the same test split, and NDCG@10 on each
  "must-not-regress" MTEB task.
- Statistical honesty: paired bootstrap confidence interval over queries for the
  NDCG@10 delta between baseline and fine-tuned models. The brief states that a
  delta without an interval is not a result.
- Acceptance thresholds:
  - Primary: `NDCG@10(fine-tuned) - NDCG@10(baseline) ≥ <target delta>` on the
    `<domain>` test split, 95 % CI lower bound above zero.
  - Secondary: no MTEB-subset task regresses by more than 10 % NDCG@10.

## 4. Serving contract — what the fine-tuned model must look like

- Architecture: bi-encoder (single-vector dense embedder). The retrieval stack is
  built around a vector index; the fine-tune must not switch families. If a learner
  wants ColBERT or SPLADE instead, that is a separate exercise (not a fine-tune).
- Output dimensionality: ≤ `<d_max>` (the current serving dim from exercise-05).
  Fine-tuning to 1024 dim when the serving stack is built for 768 forces an index
  rebuild.
- Normalization: L2-normalized outputs (cosine ≡ dot product).
- Inference template: documented in the model card the fine-tuning track produces
  (the "exercise-02 prefix" entry must exist). If a new prefix is introduced,
  the hand-off brief says so in bold.
- Max token length: `<= 512>` unless the brief explicitly requests a longer-context
  variant and the chunker in mod-102 is updated.
- Target throughput: `<X tokens/s on <GPU>>`, matching the exercise-03 serving
  envelope.
- License: the base model must allow fine-tuning + redistribution for the
  organization's intended use. If the base is Mistral-derived or under a
  non-commercial license, call it out here, not after training.

## 5. Risk contract — what can go wrong, and who owns it

- **Catastrophic forgetting.** The fine-tuned model may improve on `<domain>` but
  regress on the MTEB subset. The brief names this as a watched failure mode;
  the eval contract (section 3) is the gate.
- **Overfit to the training pairs' query style.** If training pairs are all
  "how do I X?" style, the model will lose on keyword-shaped queries. The brief
  requires the training-pair distribution to match the query-shape mix from the
  exercise-04 eval set (keyword / NL / multi-hop).
- **Hard-negative mining bias.** Hard negatives mined with the baseline model
  reinforce that model's blindspots. The brief asks the fine-tuning team to
  re-mine negatives with the new model at the halfway point of training
  (or to use a published mining pipeline that does this — e.g., GPL / BEIR
  recipes).
- **Data governance.** PII in training pairs becomes PII in model weights. The
  redaction step in section 2 is non-negotiable.
- **Supply chain.** The base model's SHA and the training-container digest are
  recorded in the fine-tune's artifact manifest. This threads into the
  OWASP ML Top 10 and MITRE ATLAS threat surfaces listed in References.
- **Rollback.** The fine-tuned model ships as a new revision alongside the
  baseline; the retrieval stack can roll back by switching embedder version
  without re-embedding (impossible — a new embedder implies re-embedding;
  rollback means keeping both embeddings materialized during migration).

## 6. Timeline and ownership

- Week 1 — fine-tuning engineer validates the data contract: can they load the
  training pairs and reach the stated NDCG baseline on `dev` with a trivial
  contrastive fit?
- Week 2–3 — training runs; checkpoint evaluated on `dev`; not on `test`.
- Week 4 — final run on `test`; paired bootstrap CI; MTEB-subset regression
  check; model card produced.
- Hand-back — retrieval team runs the fine-tuned model through exercise-05
  (compression + Matryoshka) and exercise-03 (serving-cost sweep) before
  promoting it to prod.

## 7. Open questions

Use this section to list unanswered questions *in writing*. Hand-offs die on the
unspoken assumption; naming questions is the cheapest way to surface them.
```

### 2.3 `briefs/training-pairs.jsonl.schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "training-pair",
  "type": "object",
  "required": ["pair_id", "query", "positive", "source", "redaction"],
  "properties": {
    "pair_id":    {"type": "string", "description": "stable identifier; sha256(query||positive) truncated to 24 hex chars"},
    "query":      {"type": "string", "minLength": 1},
    "positive":   {"type": "string", "minLength": 1, "description": "a passage known to answer the query"},
    "hard_negatives": {
      "type": "array",
      "items": {"type": "string", "minLength": 1},
      "description": "zero or more mined hard negatives — passages that are lexically or semantically close but do not answer the query"
    },
    "weight":     {"type": "number", "minimum": 0, "maximum": 1, "default": 1.0},
    "source":     {
      "type": "object",
      "required": ["kind", "ref"],
      "properties": {
        "kind": {"enum": ["support-ticket", "faq", "sme-annotation", "llm-synth-reviewed", "web-crawl-licensed"]},
        "ref":  {"type": "string", "description": "pointer back to the originating record"}
      }
    },
    "redaction":  {
      "type": "object",
      "required": ["tool", "version", "applied"],
      "properties": {
        "tool":    {"type": "string"},
        "version": {"type": "string"},
        "applied": {"type": "array", "items": {"enum": ["pii", "secret", "customer_id", "none"]}}
      }
    }
  }
}
```

Three worked rows for the brief:

```jsonl
{"pair_id": "a7c1e3f8d2b4a90c1e5f7a9c", "query": "how do I rotate a KMS key without downtime?", "positive": "Rotating a KMS key with aliases avoids downtime because clients reference the alias, not the key material. To rotate: create a new KMS key, update the alias to point at the new key id, then schedule the old key for deletion after the retention window.", "hard_negatives": ["To rotate a KMS key, call the console's \"Rotate\" button — the UI handles everything."], "weight": 1.0, "source": {"kind": "faq", "ref": "faq/kms-rotation#zero-downtime"}, "redaction": {"tool": "scrub", "version": "0.4.1", "applied": ["none"]}}
{"pair_id": "b9f2e7a1c5d6b03a2f8e4b7d", "query": "ssh into prod bastion fails with permission denied", "positive": "Permission denied on the prod bastion usually means the engineer's SSH certificate expired. Re-issue the short-lived certificate via the issuance service and try again; the issuance service is at <internal-url>.", "hard_negatives": ["SSH permission-denied on local VMs is usually a public-key-authentication mismatch — add the key to ~/.ssh/authorized_keys."], "weight": 1.0, "source": {"kind": "support-ticket", "ref": "tickets/42193"}, "redaction": {"tool": "scrub", "version": "0.4.1", "applied": ["customer_id", "pii"]}}
{"pair_id": "c4d8e1a2b7f9c3e4d5a6b7c8", "query": "retry backoff for the billing webhook", "positive": "Billing webhooks retry with exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s — then dead-letter. Duplicates are expected; the consumer must be idempotent keyed on event_id.", "hard_negatives": [], "weight": 1.0, "source": {"kind": "sme-annotation", "ref": "runbook/billing#retries"}, "redaction": {"tool": "scrub", "version": "0.4.1", "applied": ["none"]}}
```

## 3. Validation steps

1. `notes/to-finetune-or-not.md` records all four criteria (measured gap, hybrid +
   reranker attempted, labelable pairs exist, owner for re-tuning). Each criterion
   has a number or a named owner — none are "to be determined."
2. `briefs/finetune-brief-<domain>.md` is filled with no placeholder square-bracket
   tokens; a reviewer reading in cold can state what the fine-tuning team is being
   asked to deliver in one sentence.
3. `briefs/training-pairs.jsonl.schema.json` parses with a JSON Schema validator.
   The reference pairs above validate against it.
4. Every acceptance threshold in the eval contract is a *number* with a *comparison*
   — the brief does not say "better than baseline" without specifying by how much
   and on which split.
5. The "must-not-regress MTEB subset" names concrete tasks; the fine-tuning team
   does not have to guess.
6. A risk the brief names ("catastrophic forgetting," "hard-negative mining bias,"
   "PII in weights") maps to a specific section of the eval or data contract
   that would catch it — the risk list is not decorative.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Escalation rationale | Measured domain gap stated with numbers; hybrid + reranker attempt stated with a number; data path is named with a count and a source. |
| Objective | A single primary metric and a quality floor are stated. "Make it better" is not an objective. |
| Data contract | Training pairs file path + schema file path; PII redaction tool and version; license statement for training data. |
| Eval contract | Primary metric + statistical test (paired bootstrap CI) + both acceptance thresholds. A no-regression clause on a named MTEB subset. |
| Serving contract | Architecture, output dim, normalization, template, max tokens, throughput target, base-model license. No surprise constraints. |
| Risk contract | Catastrophic forgetting, over-fit to query style, hard-negative mining bias, PII in weights, supply-chain, rollback — each named with an owner and a mitigation. |
| Schema | The JSON Schema file validates and the example pairs validate against it. |
| Honesty | `<!-- pin-at-runtime: ... -->` markers appear anywhere a fact was not pinned (pricing, base-model license, redaction tool version). |

## 5. Common mistakes

- **Escalating to fine-tuning when hybrid retrieval would close the gap.** Fine-tuning
  is an expensive commitment. If BM25 + dense fusion (mod-106) and a
  `ms-marco-MiniLM` cross-encoder reranker (mod-107) together close the measured gap,
  the escalation is premature.
- **Shipping the brief without a data-path confirmation.** "We will fine-tune on
  support tickets" is a sentence, not a plan. Confirm that extraction, PII
  redaction, and training-pair generation actually produce `N` pairs before the
  fine-tuning engineer is on the hook.
- **No regression clause.** A fine-tuned model that improves on your corpus and
  collapses on MTEB SciFact is still a bad model if the stack is ever going to
  serve queries outside the training distribution. The eval contract must contain
  a named MTEB subset that must not regress.
- **Changing architecture mid-brief.** "Could we fine-tune ColBERTv2 instead?" is a
  reasonable question, but it changes the serving contract completely — index
  shape, storage, latency, serving framework. The brief says explicitly that the
  deliverable is a bi-encoder; a different family is a new project.
- **Ignoring base-model licensing.** Several popular embedder bases are under
  non-commercial or research-only terms. Fine-tuning does not sanitize that
  license. The brief must state the base-model license explicitly, pulled from
  the current model card.
- **Dropping the schema.** Without a schema, the fine-tuning team spends two weeks
  reverse-engineering what each column means. The schema is cheaper to write than
  that round trip.
- **Implying the fine-tuned model replaces the baseline atomically.** Changing the
  embedder requires re-embedding the entire corpus (mod-111 covers the migration
  pattern). The brief flags that this is a serving-side change, not just a model
  swap.
- **Omitting the re-tune cadence.** A fine-tuned embedder on a drifting corpus
  silently rots. Naming the owner for the re-tune is as important as naming the
  owner for the first tune.

## 6. References

- Fine-tuning for embeddings — recipes and open implementations:
  - Sentence-Transformers "Training Overview" — [sbert.net/docs/training/overview.html](https://sbert.net/docs/training/overview.html).
  - Sentence-Transformers Matryoshka training example — [sbert.net/examples/training/matryoshka](https://sbert.net/examples/training/matryoshka/).
  - BEIR GPL (Generative Pseudo-Labeling) — Wang et al., "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval" — [arxiv.org/abs/2112.07577](https://arxiv.org/abs/2112.07577); repo — [github.com/UKPLab/gpl](https://github.com/UKPLab/gpl).
  - LlamaIndex finetuning guide (for retrieval-focused embedder fine-tunes) — [docs.llamaindex.ai/en/stable/examples/finetuning](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/).
- Evaluation and reporting discipline:
  - MTEB — [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) and leaderboard — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
  - BEIR — [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
  - Paired bootstrap confidence intervals for IR deltas — Manning, Raghavan, Schütze, "Introduction to Information Retrieval" — [nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/).
- Governance / threat surface this hand-off creates:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
