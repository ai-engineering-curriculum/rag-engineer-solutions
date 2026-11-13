# Exercise 04 — Solution: LambdaMART LTR with the Elasticsearch LTR Plugin

This is a reference solution for
`exercise-04-lambdamart-ltr-with-elasticsearch-ltr-plugin.md`. It is an
**end-to-end implementation** exercise. The learner trains a LambdaMART model
on a small judgment set from the mod-105 articles corpus, uploads the model
to an Elasticsearch (or OpenSearch) cluster via the Learning-to-Rank plugin,
rescans a first-phase BM25 recall set at query time, and reports before/after
NDCG@10 on a held-out slice.

The teaching point is that LTR is a **rescore layer**, not a replacement for
BM25 or ANN. The first phase is a cheap recall query; the LTR model reranks
the top ~100. Shipping an LTR model without a disciplined feature store, a
versioned model artifact, and a held-out evaluation is how teams "deploy LTR"
and then can't reproduce the numbers that convinced them to deploy it.

## 1. Solution overview

The deliverable has six parts:

1. **`judgments/qrels.tsv`** — a small judgment set in TREC-style format
   (`qid, docid, label`). ~20 queries × ~20 documents each is enough to see
   the method work; production LTR wants orders of magnitude more.
2. **`features/featureset.json`** — the feature definitions registered with
   the LTR plugin: BM25 on `title` and `body`, the kNN cosine score from
   exercise-03, recency, tag-match, and author-match features. Each is a
   templated Elasticsearch query.
3. **`src/extract_features.py`** — the feature-logging script that runs each
   `qid` + `docid` through the LTR plugin's `logging_only` mode and dumps
   training rows in LibSVM / RankLib / XGBoost ranker format.
4. **`src/train_lambdamart.py`** — trains a LambdaMART model with XGBoost
   `rank:ndcg` (or RankLib's LambdaMART) and saves it in the format the LTR
   plugin accepts.
5. **`src/upload_model.py`** — uploads the trained model artifact to the
   LTR plugin's model registry and tags it with a version.
6. **`src/rescore.py` + `notes/results.md`** — the query shape that runs
   BM25 first-phase retrieval then LTR rescore on the top-100, plus the
   held-out NDCG@10 before/after and a short error analysis.

The exercise is scoped to a *working pipeline*, not state-of-the-art NDCG.
If the model improves NDCG@10 on the held-out slice by any meaningful margin
and the artifact is reproducible, the exercise passes.

## 2. Worked answer or implementation

### 2.1 LambdaMART in one paragraph

LambdaMART is a pairwise/listwise learning-to-rank algorithm that trains a
gradient-boosted tree ensemble (MART — Multiple Additive Regression Trees)
whose gradients come from LambdaRank's listwise loss. In plain terms: it
learns to rank items in a *list* given per-item features, using the gain
you lose by swapping two neighbors (weighted by `|ΔNDCG|`) as the training
signal. The model takes a fixed-length feature vector per `(query, doc)`
pair and outputs a score; results are sorted by that score. In the
Elasticsearch / OpenSearch LTR plugin, the model is uploaded as a JSON
representation of the tree ensemble and the `sltr` query evaluates it at
rescore time.

### 2.2 `judgments/qrels.tsv` — the judgment set

TREC-style `qrels`:

```
# qid   doc_id                          label
q01     article_000123                  3
q01     article_000455                  2
q01     article_000911                  1
q01     article_001042                  0
q01     article_000777                  0
q02     article_002101                  3
q02     article_002220                  2
...
```

- `label` is a graded relevance score (0 = irrelevant, 3 = highly relevant).
- Collect judgments on the **first-phase recall set** (top-K BM25 and / or
  top-K kNN). Judging documents the first phase does not retrieve teaches
  the LTR model nothing that it can use, because at inference time those
  documents are not in the rescore candidate pool.
- Keep a disjoint **held-out** slice (10–20% of queries) that never
  participates in training.

A separate `queries.tsv` holds the query text per `qid`:

```
# qid   query
q01     consumer rebalance stuck
q02     dlq retry patterns
...
```

### 2.3 `features/featureset.json` — the feature definitions

Features are Elasticsearch queries templated with `{{query}}` and (when
needed) `{{keywords}}` or `{{embedding}}` parameters. The LTR plugin
evaluates each as a scoring query against the candidate doc and returns the
resulting `_score` as a feature value.

```json
{
  "featureset": {
    "name": "mod105_articles_v1",
    "features": [
      {
        "name": "bm25_title",
        "params": ["query"],
        "template": { "match": { "title": "{{query}}" } }
      },
      {
        "name": "bm25_body",
        "params": ["query"],
        "template": { "match": { "body": "{{query}}" } }
      },
      {
        "name": "bm25_title_phrase",
        "params": ["query"],
        "template": { "match_phrase": { "title": "{{query}}" } }
      },
      {
        "name": "knn_cosine",
        "params": ["embedding"],
        "template": {
          "script_score": {
            "query": { "match_all": {} },
            "script": {
              "source": "dotProduct(params.q, 'embedding')",
              "params": { "q": "{{embedding}}" }
            }
          }
        }
      },
      {
        "name": "tag_match_count",
        "params": ["tags"],
        "template": {
          "function_score": {
            "query": { "terms": { "tags": {{#toJson}}tags{{/toJson}} } },
            "script_score": { "script": { "source": "_score" } }
          }
        }
      },
      {
        "name": "recency_days",
        "params": [],
        "template": {
          "function_score": {
            "query": { "match_all": {} },
            "gauss": {
              "created_at": {
                "origin": "now",
                "scale":  "365d",
                "decay":  0.5
              }
            }
          }
        }
      },
      {
        "name": "author_match",
        "params": ["query"],
        "template": { "match": { "author": "{{query}}" } }
      }
    ]
  }
}
```

**Choices and why:**

- Two BM25 features on `title` and `body` capture field-level lexical
  signal independently — a doc that matches in the title tells the model
  something a body match doesn't.
- `bm25_title_phrase` separates "words appear together" from "words appear
  anywhere."
- `knn_cosine` injects the dense-vector score as a feature. The model
  learns when kNN is informative and when it is not, rather than fusing
  lexical + dense with a hand-picked weight (mod-106 does the hand-picked
  version; here the fusion is learned).
- `tag_match_count` lets the model use controlled vocabulary overlap.
- `recency_days` is a Gaussian decay feature; relevance decays with
  distance from "now."
- `author_match` is a cheap signal for query-is-a-name failures.

Keep the feature list *small* (7–15 features) for a first pass. LambdaMART
can handle hundreds of features, but each feature adds a per-document
query at rescore time; an LTR latency budget blows up fast if the feature
set is lazy.

Register the feature set:

```bash
curl -XPUT 'http://localhost:9200/_ltr/_featureset/mod105_articles_v1' \
  -H 'Content-Type: application/json' \
  -d @features/featureset.json
```

### 2.4 `src/extract_features.py` — logging feature vectors

The plugin exposes a "log" mode: run a normal query but wrap it so the
feature values for the matched documents are returned alongside each hit.
The script runs this for every `(qid, judged_doc)` pair in `qrels.tsv`:

```python
"""Produce RankLib / XGBoost ranker training rows from judgments."""
from __future__ import annotations
import csv, json, pathlib
import numpy as np
from elasticsearch import Elasticsearch

INDEX = "articles"
FEATURESET = "mod105_articles_v1"
LOG_NAME   = "log_features"

es = Elasticsearch("http://localhost:9200")

def log_body(query_text: str, docids: list[str], query_vec: list[float]):
    return {
        "query": {
            "bool": {
                "filter": [ { "terms": { "_id": docids } } ],
                "must":   [ {
                    "sltr": {
                        "_name": LOG_NAME,
                        "featureset": FEATURESET,
                        "params": {
                            "query":     query_text,
                            "embedding": query_vec,
                            "tags":      []
                        }
                    }
                } ]
            }
        },
        "ext": {
            "ltr_log": {
                "log_specs": {
                    "name": LOG_NAME,
                    "named_query": LOG_NAME
                }
            }
        },
        "size": len(docids),
        "_source": False
    }

def main():
    qrels = list(csv.reader(open("judgments/qrels.tsv"), delimiter="\t"))
    queries = { q: text for q, text in csv.reader(
        open("judgments/queries.tsv"), delimiter="\t") }

    # Group qrels by qid
    by_qid: dict[str, list[tuple[str, int]]] = {}
    for qid, doc, label in qrels:
        by_qid.setdefault(qid, []).append((doc, int(label)))

    rows: list[str] = []
    for qid, items in by_qid.items():
        q_text = queries[qid]
        q_vec  = embed(q_text).tolist()        # mod-103 encoder
        docids = [d for d, _ in items]
        resp = es.search(index=INDEX, body=log_body(q_text, docids, q_vec))

        feat_vals = {
            hit["_id"]: hit["fields"]["_ltrlog"][0]["log_entries"]
            for hit in resp["hits"]["hits"]
        }
        for doc, label in items:
            fs = { e["name"]: e.get("value", 0.0) for e in feat_vals[doc] }
            # LibSVM-ranker format: <label> qid:<qid> 1:v1 2:v2 ... # docid
            ordered = [
                fs.get("bm25_title", 0.0),
                fs.get("bm25_body",  0.0),
                fs.get("bm25_title_phrase", 0.0),
                fs.get("knn_cosine", 0.0),
                fs.get("tag_match_count", 0.0),
                fs.get("recency_days", 0.0),
                fs.get("author_match", 0.0),
            ]
            line = f"{label} qid:{qid} " + " ".join(
                f"{i+1}:{v}" for i, v in enumerate(ordered)) + f" # {doc}"
            rows.append(line)

    pathlib.Path("features/train.txt").write_text("\n".join(rows))

def embed(text: str):
    # Load the mod-103 bi-encoder here.
    raise NotImplementedError("wire to mod-103 encoder")

if __name__ == "__main__":
    main()
```

The output file (`features/train.txt`) is in the standard RankLib / XGBoost
ranker input format: one line per `(qid, doc)` pair with a graded label,
a `qid` group tag, and numbered feature columns.

### 2.5 `src/train_lambdamart.py` — XGBoost LambdaMART

```python
"""Train a LambdaMART ranker with XGBoost and export in the LTR plugin format."""
from __future__ import annotations
import json, pathlib
import numpy as np
import xgboost as xgb

TRAIN  = "features/train.txt"
HELD   = "features/eval.txt"
MODEL  = "models/lambdamart_v1.xgb.json"

def load_ranklib(path: str):
    """Minimal LibSVM-ranker parser → (X, y, qid_group_sizes)."""
    y, groups, rows = [], {}, []
    cur_qid, cur_count = None, 0
    nfeat = 0
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            label = int(parts[0])
            qid = parts[1].split(":")[1]
            feats = {int(p.split(":")[0]): float(p.split(":")[1]) for p in parts[2:]}
            nfeat = max(nfeat, max(feats))
            rows.append(feats); y.append(label)
            if qid != cur_qid:
                if cur_qid is not None:
                    groups[cur_qid] = cur_count
                cur_qid, cur_count = qid, 0
            cur_count += 1
        groups[cur_qid] = cur_count
    X = np.zeros((len(rows), nfeat), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, v in r.items():
            X[i, j-1] = v
    return X, np.array(y), list(groups.values())

def main():
    Xt, yt, gt = load_ranklib(TRAIN)
    Xe, ye, ge = load_ranklib(HELD)

    dtrain = xgb.DMatrix(Xt, label=yt); dtrain.set_group(gt)
    deval  = xgb.DMatrix(Xe, label=ye); deval.set_group(ge)

    params = {
        "objective":       "rank:ndcg",
        "eval_metric":     ["ndcg@10", "ndcg@5"],
        "learning_rate":   0.1,
        "max_depth":       4,
        "min_child_weight": 1,
        "gamma":           0.0,
        "subsample":       0.8,
        "colsample_bytree": 1.0,
        "tree_method":     "hist",
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 8,
    }

    booster = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=25,
        verbose_eval=10,
    )

    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL)
    # The LTR plugin accepts XGBoost's JSON model directly for the
    # `model/model-xgboost` format.
    print("saved", MODEL)

if __name__ == "__main__":
    main()
```

Notes:

- `rank:ndcg` is XGBoost's LambdaRank-style objective. `rank:pairwise`
  (RankNet-style) is cheaper but typically worse on graded judgments.
- Early stopping on `ndcg@10` prevents overfitting on a small judgment set.
- `max_depth` of 4–6 is typical; deeper trees overfit on judgment sets of
  this size.

### 2.6 `src/upload_model.py` — model registration

```bash
# Upload the model artifact under the feature set it was trained against.
curl -XPUT 'http://localhost:9200/_ltr/_featureset/mod105_articles_v1/_createmodel' \
  -H 'Content-Type: application/json' \
  -d @models/upload_body.json
```

Where `models/upload_body.json` wraps the XGBoost booster JSON:

```json
{
  "model": {
    "name": "lambdamart_v1_2026-11-12",
    "model": {
      "type":       "model/xgboost+json",
      "definition": "<escaped JSON from booster.save_model>"
    }
  }
}
```

The model name must be versioned — never overwrite a deployed model's name
with a retrained artifact. `lambdamart_v1_2026-11-12` encodes the feature-set
version, the model version, and the date; the next retrain increments the
date and the serving query rewrites to the new name.

### 2.7 `src/rescore.py` — BM25 first phase, LTR rescore

```json
POST articles/_search
{
  "query": {
    "bool": {
      "must": { "match": { "body": "{{query}}" } },
      "should": [
        { "match": { "title": { "query": "{{query}}", "boost": 2.0 } } }
      ]
    }
  },
  "rescore": {
    "window_size": 100,
    "query": {
      "rescore_query": {
        "sltr": {
          "params": {
            "query":     "{{query}}",
            "embedding": "{{embedding}}",
            "tags":      []
          },
          "model": "lambdamart_v1_2026-11-12"
        }
      },
      "rescore_query_weight": 2.0,
      "query_weight": 0.0,
      "score_mode": "total"
    }
  }
}
```

**Why `window_size: 100`.** The LTR rescore runs the model's feature
queries over the first-phase top-N. 100 is a defensible default on a
small corpus and is cheap at interactive QPS. Too small a window and the
first phase gates out documents the model would have promoted; too large
and p95 inflates linearly. mod-107 (reranking) covers budgets in detail.

**Why `query_weight: 0.0`.** The first-phase BM25 score is already used
by the LTR model via the `bm25_title`/`bm25_body` features; summing it
again at rescore time double-counts the signal. Keep the first phase for
recall, let the model do all the scoring at the rescore step.

### 2.8 `notes/results.md` — reporting shape

```
System                          NDCG@10 (held-out, N queries)
------------------------------  ----------------------------
BM25 baseline (no LTR)                   0.xx
BM25 + LTR v1 (LambdaMART)               0.yy
Delta                                    +0.zz  (p-value via per-query bootstrap)
```

Also report:

- **Per-query delta histogram.** A system that helps the mean but *hurts
  the tail* is not a system you want to ship. The histogram of per-query
  NDCG differences should be visibly shifted to the right with few deep
  negative outliers.
- **Feature importance.** XGBoost's `get_score(importance_type="gain")`
  gives the per-feature gain. If one feature dominates (e.g.,
  `bm25_title` with 80% of gain), the exercise learned a weight, not a
  ranking model. Add features or revisit the judgment set.
- **Error analysis on five worst queries.** For the queries where LTR
  made things worse, inspect the top-10: often the judgment labels are
  wrong, or the first-phase missed the gold doc entirely and no amount
  of rescoring can save it.

## 3. Validation steps

1. The feature set registers successfully (`PUT /_ltr/_featureset/...`
   returns 201/200). Re-registering under the same name is a no-op; use a
   new name when features change.
2. `extract_features.py` produces a `train.txt` with the expected number
   of lines (sum of docs per qid) and no `NaN`/`Infinity` values. A single
   missing feature value in the ranker input will break XGBoost training
   silently.
3. XGBoost training converges — early-stopping fires before the hard
   iteration cap, and `eval_ndcg@10` on the held-out slice *beats* the
   first-phase BM25 baseline. If not, feature set or judgments are the
   problem; no training trick saves a mis-specified setup.
4. The model upload returns 200. Querying `GET /_ltr/_model/<name>` returns
   the registered model.
5. The rescore query runs within the project's p95 latency budget at the
   target QPS. Record the absolute numbers.
6. Held-out NDCG@10 improves over BM25-only by a statistically defensible
   margin (bootstrap or paired t-test, not a single-decimal difference
   presented without variance).
7. The learner can name which feature gained most of the model's
   importance and give a sentence about *why* it is informative on this
   corpus.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Role of LTR | The learner explicitly treats LTR as a **rescore** on a first-phase recall set, not as a replacement for first-phase retrieval. |
| Judgment hygiene | A held-out slice is kept, judgments are graded (0–3, not binary), and the judgment set covers the *first-phase recall set* rather than arbitrary documents. |
| Feature set | Small (≤ ~15) and purposeful — each feature is defensible in one sentence, not "we have BM25 on every field." |
| Feature logging | Uses the plugin's `sltr` logging mode (the `ext.ltr_log` block). Does **not** hand-run each feature query and stitch the results — stitching is error-prone and does not match what rescore will see. |
| Training setup | LambdaMART / `rank:ndcg` objective. Group sizes per `qid` set on the DMatrix (XGBoost requires this). Early stopping on a held-out metric. |
| Reproducibility | Model artifact is versioned by name (`lambdamart_v1_2026-11-12` or similar). The upload JSON lives in the repo. The training script is deterministic given the random seed and the input files. |
| Rescore query shape | Uses `rescore` with `window_size` set and `query_weight: 0.0`. The exercise learner notes why both choices. |
| Latency account | The exercise reports a p50/p95 latency delta of adding the rescore, not just the NDCG delta. An LTR system with +3% NDCG and +80 ms p95 is a trade-off; the number has to be on the page. |
| Evaluation | NDCG@10 (and ideally NDCG@5 / MRR@10) reported before and after on the **held-out** slice. The exercise is explicit about which queries were held out. |
| Error analysis | Per-query delta histogram and a close read of the worst-regressed queries. |

Fail-fast smells:

- "Our training NDCG improved by 0.12." Training NDCG is not an evaluation
  metric; it only tells you the model is fitting.
- A feature set that contains BM25 on every analyzable field — the model
  will learn redundancy and the latency goes up without an NDCG payoff.
- Uploading the model under the name `latest`. Models get rolled back; a
  name like `latest` makes "which model actually served last Tuesday"
  unanswerable.
- A rescore that evaluates the LTR model on the whole corpus instead of
  the window. The plugin will let you; the latency will not.

## 5. Common mistakes

- **Judging documents that aren't in the first-phase recall set.** The
  LTR model only sees documents the first phase surfaces. Judgments on
  unreachable documents do not contribute training signal; they just
  waste judge time.
- **Shipping binary labels.** LambdaMART's listwise loss is most
  informative on graded labels. Binary (relevant / not) collapses the
  loss landscape; the model trains, but shallowly.
- **No held-out split.** Reporting NDCG on the training set is a lie
  by omission. Hold out *entire queries*, not individual documents within
  a query.
- **Training on features drawn against a stale index.** Features must be
  logged against the exact analyzer/mapping that will serve. If the
  mapping changes between logging and serving, the model is evaluating
  on different numbers than it trained on.
- **Overweighting the first-phase score at rescore.** Setting
  `query_weight: 1.0` and `rescore_query_weight: 1.0` sums the BM25 score
  with the model score — the BM25 signal, which is already an input
  feature, is counted twice. Prefer `query_weight: 0.0` and let the
  model own scoring on the window.
- **Unversioned models.** "The model we deployed last quarter" is a
  commitment the team cannot keep without versioned names. Encode the
  feature-set version, the model version, and the training date in the
  name.
- **Big feature sets early.** 60+ features on 100 judged queries is
  classic overfit. Start with a small set, verify it beats BM25, then
  add features one at a time while measuring per-query deltas.
- **Treating latency as free.** Every feature in the set is a query at
  rescore time. If the feature set has ten heavy function-score features
  and the window is 100, the engine does 1000 scoring query evaluations
  per user query. Measure.
- **Forgetting the feature-store problem.** The feature set is as much
  of the model as the booster is. Check it in. Version it. A retrain
  against a *different* feature set and the *same* model name is a
  silent data corruption.

## 6. References

- LambdaMART and LambdaRank:
  - Burges (2010), "From RankNet to LambdaRank to LambdaMART: An
    Overview" (Microsoft Research TR) —
    [microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/).
  - Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting
    System" — [arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754).
  - XGBoost Learning to Rank docs —
    [xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
- LTR plugin and plumbing:
  - Elasticsearch Learning to Rank plugin —
    [github.com/o19s/elasticsearch-learning-to-rank](https://github.com/o19s/elasticsearch-learning-to-rank).
  - OpenSearch LTR plugin —
    [github.com/opensearch-project/opensearch-learning-to-rank-base](https://github.com/opensearch-project/opensearch-learning-to-rank-base).
  - Elasticsearch rescore API —
    [elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#rescore](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#rescore).
- Evaluation:
  - Järvelin & Kekäläinen (2002), "Cumulated gain-based evaluation
    of IR techniques" (NDCG) —
    [dl.acm.org/doi/10.1145/582415.582418](https://dl.acm.org/doi/10.1145/582415.582418).
  - TREC qrels format —
    [trec.nist.gov/data/qrels_eng/](https://trec.nist.gov/data/qrels_eng/).
- Risk / governance framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
