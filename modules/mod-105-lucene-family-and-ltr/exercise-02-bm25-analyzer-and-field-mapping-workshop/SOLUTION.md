# Exercise 02 — Solution: BM25 Analyzer and Field-Mapping Workshop

This is a reference solution for
`exercise-02-bm25-analyzer-and-field-mapping-workshop.md`. It is an
**implementation + explanation** exercise. The learner builds an index mapping
for a Lucene-family engine (OpenSearch is used here as the reference, since
exercise-01 recommends it), picks analyzers and tokenizers deliberately, tunes
the BM25 `k1`/`b` parameters, and shows each choice's effect with a few probing
queries.

The teaching point is that an index mapping is *the search algorithm*. Choosing
`keyword` instead of `text` on a field changes the recall shape completely;
leaving `english` analyzer on an identifier field makes exact-match queries
fail silently. The exercise is about making those choices on purpose and being
able to defend them.

## 1. Solution overview

The deliverable has four parts:

1. **`mappings/articles.json`** — an index mapping for a "technical articles"
   corpus (title, body, tags, author, urls, created_at). Every field choice
   has a one-line justification above it.
2. **`settings/analyzers.json`** — custom analyzers: a `code_ngram` analyzer
   for code identifiers, a `exact_lower` normalizer for keyword fields, and a
   default English analyzer for prose. These are referenced from the mapping.
3. **`notes/bm25-tuning.md`** — a short walk-through of `k1` and `b` with
   concrete query/document pairs that demonstrate how each knob moves recall
   and ranking. Plus the two `similarity: BM25` custom similarities the index
   defines (prose vs. short-title).
4. **`validate/queries.http`** — a `.http` file (works with `httpie`, VS Code
   REST Client, curl) of probing queries that each test exactly one choice:
   stemming, stopwords, keyword-vs-text, phrase matching, length normalization.
   Each query has a one-sentence "what this proves."

The learner is not writing a toy — they are writing an index mapping that a
colleague should be able to read as prose and accept.

## 2. Worked answer or implementation

### 2.1 Corpus shape and field inventory

A representative technical-articles corpus has fields with sharply different
search semantics:

| Field | Shape | Searcher's intent |
|---|---|---|
| `title` | Short prose, 3–12 tokens. | Match on words in the title; title matches should rank higher per unit of text than body matches. |
| `body` | Long prose, 200–3000 tokens. | Classic full-text match; stemming and stopword removal expected. |
| `tags` | A small set of controlled values (`["kafka", "retry-queue"]`). | Exact-match filter (faceting). No stemming. |
| `author` | A person's name, 2–5 tokens. | Exact match on a typed name; also match on partial names. |
| `code_identifiers` | Snippets like `OpenAIResponse`, `token_count`. | CamelCase and `snake_case` should be splittable; substring match useful. |
| `url` | A URL. | Exact filter and domain-aware match. |
| `created_at` | Timestamp. | Range filter and date math. |

Each of these needs a different analyzer.

### 2.2 `settings/analyzers.json` — custom analyzers and similarities

```json
{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "number_of_replicas": 1,
      "similarity": {
        "bm25_prose": {
          "type": "BM25",
          "k1": 1.2,
          "b": 0.75
        },
        "bm25_short": {
          "type": "BM25",
          "k1": 1.0,
          "b": 0.3
        }
      },
      "analysis": {
        "normalizer": {
          "exact_lower": {
            "type": "custom",
            "filter": ["lowercase", "asciifolding"]
          }
        },
        "tokenizer": {
          "code_tok": {
            "type": "pattern",
            "pattern": "[\\s/\\.,\\(\\)\\[\\]{}]+|(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\\d)"
          }
        },
        "filter": {
          "en_stop": {
            "type": "stop",
            "stopwords": "_english_"
          },
          "en_stem": {
            "type": "stemmer",
            "language": "light_english"
          },
          "edge_ngram_2_15": {
            "type": "edge_ngram",
            "min_gram": 2,
            "max_gram": 15
          }
        },
        "analyzer": {
          "prose_en": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding", "en_stop", "en_stem"]
          },
          "code_analyzer": {
            "type": "custom",
            "tokenizer": "code_tok",
            "filter": ["lowercase"]
          },
          "code_search": {
            "type": "custom",
            "tokenizer": "code_tok",
            "filter": ["lowercase"]
          },
          "author_edge": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding", "edge_ngram_2_15"]
          },
          "author_search": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding"]
          }
        }
      }
    }
  }
}
```

**Choices and why:**

- **`bm25_short` for titles.** Short-text scoring suffers from BM25's default
  `b=0.75` length normalization because every title is "short." Lowering `b`
  to ~0.3 flattens the length penalty so a title that happens to be slightly
  longer is not unfairly demoted. Dropping `k1` to 1.0 reduces the term-frequency
  saturation effect (titles rarely repeat terms).
- **Light stemmer, not full Porter.** `light_english` reduces plurals and common
  suffixes (`-s`, `-es`, `-ing`) without the aggressive stem-collapsing of
  full Porter. It costs little recall and buys back precision on identifier-like
  words.
- **`code_tok` pattern tokenizer.** Splits on whitespace *and* on CamelCase
  boundaries *and* on digits-after-letters. `OpenAIResponse` becomes
  `open`, `ai`, `response`; `token_count_v2` becomes `token`, `count`, `v`,
  `2`. A standard tokenizer leaves `OpenAIResponse` as a single token, which
  makes every "open ai" query miss.
- **Author `edge_ngram` at index time, standard analyzer at search time.**
  Prefix search on names ("jor…" → "jordan hughes") without requiring the
  user to type a wildcard. Separate search analyzer prevents the user's query
  from being expanded into all of its prefixes too.
- **`exact_lower` normalizer for keyword fields.** For tags, lowercase and
  asciifold without tokenizing. A tag is one value, not a bag of words.

### 2.3 `mappings/articles.json` — the index mapping

```json
{
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "id": { "type": "keyword" },

      "title": {
        "type": "text",
        "analyzer": "prose_en",
        "similarity": "bm25_short",
        "fields": {
          "raw": {
            "type": "keyword",
            "normalizer": "exact_lower",
            "ignore_above": 512
          }
        }
      },

      "body": {
        "type": "text",
        "analyzer": "prose_en",
        "similarity": "bm25_prose"
      },

      "tags": {
        "type": "keyword",
        "normalizer": "exact_lower"
      },

      "author": {
        "type": "text",
        "analyzer": "author_edge",
        "search_analyzer": "author_search",
        "fields": {
          "raw": {
            "type": "keyword",
            "normalizer": "exact_lower"
          }
        }
      },

      "code_identifiers": {
        "type": "text",
        "analyzer": "code_analyzer",
        "search_analyzer": "code_search"
      },

      "url": {
        "type": "keyword",
        "fields": {
          "domain": {
            "type": "keyword",
            "normalizer": "exact_lower"
          }
        }
      },

      "created_at": { "type": "date", "format": "strict_date_optional_time" }
    }
  }
}
```

**Choices and why:**

- `dynamic: "strict"`. Mapping errors must fail loudly, not silently create a
  `string` field with default analyzer when a typo lands.
- `title` is `text` **and** has a `raw` keyword sub-field. Prose search on
  the `text` side, exact lookups and sorting on the `raw` side.
- `body` uses `bm25_prose` (default BM25 tuning) because bodies vary from a
  short note to a long article — the length normalization is doing useful work.
- `tags` is `keyword`. Faceting and `terms` filters need exact matches; a
  `text` tag would stem `authentication` → `authent` and miss filter equality.
- `author` is a `text` field with `author_edge` at index time and
  `author_search` at search time, plus a `raw` keyword for exact-string
  faceting.
- `code_identifiers` has the same analyzer at index and search time so
  CamelCase/snake_case tokenization is applied symmetrically.
- `url` is `keyword` plus a `domain` sub-field for filtering on domain.
- `created_at` is a strict date; permissive date parsing in production has
  burned too many pipelines.

### 2.4 `notes/bm25-tuning.md` — what `k1` and `b` actually do

BM25's score for a term `t` in document `d` is

```
score(d, t) = IDF(t) * ( f(t, d) * (k1 + 1) ) / ( f(t, d) + k1 * (1 - b + b * |d| / avgdl) )
```

where `f(t, d)` is the term frequency in `d`, `|d|` is the document length,
and `avgdl` is the average document length in the index.

- **`k1` controls term-frequency saturation.** As `k1 → 0`, the first
  occurrence of `t` in `d` dominates and additional occurrences are nearly
  free. As `k1 → ∞`, the score grows almost linearly with `f(t, d)`.
  Lucene's default is `k1 = 1.2`. Shorter documents benefit from lower `k1`
  because repetition is less meaningful when the text is small.

- **`b` controls length normalization.** `b = 0` means no length penalty —
  a 10,000-word document matching "kafka" once is scored the same as a
  10-word document matching "kafka" once. `b = 1` means the full BM25
  length normalization. Lucene's default is `b = 0.75`. Lower `b` on fields
  where every document is short (titles, tags, product names) because the
  "length relative to average" ratio is noisy; higher `b` on fields where
  long documents dilute term meaning (bodies, long transcripts).

A second Lucene subtlety: `avgdl` is **computed per field per segment**. That
is why forcing a larger merge (`_forcemerge`) can shift BM25 scores slightly —
the average-doc-length statistic recomputes as segments combine.

### 2.5 `validate/queries.http` — probing queries

```http
# 1) Stemming works on body: "retries" should match a doc that only has "retry".
# Why: prose_en has light_english stemmer.
POST articles/_search
Content-Type: application/json

{ "query": { "match": { "body": "retries" } } }

###
# 2) Exact-match on tags: {"tags": "retry-queue"} must match literally.
# Why: tags is keyword with exact_lower normalizer; no tokenization.
POST articles/_search
Content-Type: application/json

{ "query": { "term": { "tags": "retry-queue" } } }

###
# 3) Title scoring: a 4-word title matching one query term should beat
# a 12-word title matching the same term because bm25_short is configured.
# Why: bm25_short uses b=0.3, softening length normalization for short texts.
POST articles/_search
Content-Type: application/json

{
  "query": {
    "match": { "title": "backoff" }
  },
  "explain": true
}

###
# 4) CamelCase code search: "OpenAIResponse" indexed, query "open ai response"
# should match because code_analyzer splits CamelCase boundaries.
POST articles/_search
Content-Type: application/json

{ "query": { "match": { "code_identifiers": "open ai response" } } }

###
# 5) Stopword removal on body: query "the kafka" and "kafka" should score
# identically on bodies that contain "kafka" because "the" is in en_stop.
POST articles/_search
Content-Type: application/json

{ "query": { "match": { "body": "the kafka" } } }

###
# 6) Author prefix: typing "jor" should surface "Jordan Hughes".
# Why: author indexed with author_edge (edge_ngram), searched with
# author_search (plain lowercase). No wildcard needed.
POST articles/_search
Content-Type: application/json

{ "query": { "match": { "author": "jor" } } }

###
# 7) Sort by created_at desc, filter tag, match body. Catches date format drift.
POST articles/_search
Content-Type: application/json

{
  "query": {
    "bool": {
      "filter": [
        { "term": { "tags": "kafka" } },
        { "range": { "created_at": { "gte": "2024-01-01" } } }
      ],
      "must":   [ { "match": { "body": "consumer rebalance" } } ]
    }
  },
  "sort": [ { "created_at": "desc" } ]
}

###
# 8) Dynamic strict catch: posting an unknown field must 400, not silently index.
# Expected: 400 strict_dynamic_mapping_exception.
POST articles/_doc
Content-Type: application/json

{ "title": "Oops", "typoField": "value" }
```

Each query maps to one choice in the mapping. If a query's expected behavior
breaks, the fix is in `mappings/articles.json` or `settings/analyzers.json`.

### 2.6 How to confirm an analyzer does what it says

OpenSearch and Elasticsearch both expose an `_analyze` endpoint that returns
the actual token stream for a given analyzer and input. Use this before you
ship the mapping — it is cheaper than discovering a tokenization bug in
production.

```http
POST articles/_analyze
Content-Type: application/json

{ "analyzer": "code_analyzer", "text": "OpenAIResponse.token_count_v2" }

# Expected tokens: open, ai, response, token, count, v, 2
```

```http
POST articles/_analyze
Content-Type: application/json

{ "analyzer": "prose_en", "text": "The retries were exponential." }

# Expected tokens: retri (or retry depending on stemmer), exponenti, exponential
# "the", "were" removed by en_stop.
```

## 3. Validation steps

1. Create the index with the settings and mapping above. The create call must
   succeed with `"acknowledged": true`. If it fails, the error message names
   the offending field — fix before continuing.
2. Run every query in `validate/queries.http`. Each query has a stated
   expectation; `explain: true` reveals which term and which similarity are
   contributing to the score.
3. Use `_analyze` to confirm the analyzers behave as documented in §2.6.
   `code_analyzer` on `OpenAIResponse` must produce three tokens, not one.
4. Delete one tag, reingest, and confirm the exact-match tag query no longer
   returns the deleted document. If `tags` had been `text`, the test would
   accidentally pass via stemming.
5. Index two documents with identical `body` text but different `title` lengths.
   Confirm the one with the shorter title ranks higher for a `title` query,
   demonstrating the `bm25_short` similarity's lower `b`.
6. Try to index `{"typoField": "value"}`. The call must return
   `strict_dynamic_mapping_exception`.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Field-shape taxonomy | The mapping distinguishes `text`, `keyword`, numeric, date, and `text`-with-`.raw` sub-field. No field is miscategorized (e.g., `author` as pure keyword, or `tags` as `text`). |
| Analyzer choice per field | Every `text` field has a named analyzer. No field silently falls back to the `standard` analyzer "because it works in the console." |
| Stemming choice | The learner chose `light_english` or justified `english` / `porter`. A fuller stemmer is acceptable with reason. |
| Stopword list | A language-appropriate stop list is in place. Removing stopwords is a decision, not a default; the choice is written down. |
| BM25 tuning | Two or more similarities are defined — at least one for short fields and one for long fields — and the choice of `k1`/`b` is defended in prose. |
| Keyword normalizer | Keyword fields have `lowercase` + `asciifolding` via a normalizer. A raw `keyword` field with no normalizer is called out as a decision, not an oversight. |
| Dynamic mapping | The mapping uses `dynamic: "strict"` or the learner writes a one-line defense of `dynamic: "true"` in this corpus. |
| Validation harness | `validate/queries.http` contains at least one query per mapping choice; each query names the behavior it is testing. |
| Explainability | `_explain` output is shown for the title-ranking case. The learner can point at the term-frequency and length-normalization factors on the page. |
| Pitfalls listed | Common mistakes (§5) are acknowledged and defended against. |

Fail-fast smells:

- A `text` field with no `analyzer` specified. The default is the `standard`
  analyzer — acceptable, but silence is not a decision.
- A `keyword` field with `lowercase: true` written at the field level. That
  option does not exist in modern mappings; the right mechanism is a
  `normalizer`.
- A `text` field on `tags` with a comment "to allow searching." The field
  should be `keyword`; "search" of tags is faceting.
- A `date` field with `format: "epoch_millis||strict_date_optional_time||yyyy-MM-dd"`.
  Permissive date parsing hides upstream bugs.

## 5. Common mistakes

- **Leaving the default analyzer on every field.** The standard analyzer is
  fine for prose and wrong for identifiers, URLs, codes, and anything with
  CamelCase or digits-adjacent-to-letters. Pick per field.
- **Using `text` where a `keyword` is correct.** Tags, enums, URLs, IDs,
  statuses — anything with a controlled vocabulary — are `keyword`. Using
  `text` for these fields breaks exact-equality filters in ways that are
  hard to debug (it "sometimes" matches because of stemming).
- **Reversing the index/search analyzer for edge-ngrams.** Index the field
  with the edge-ngram analyzer; search with a plain analyzer. Running the
  edge-ngram analyzer on the query too would expand the query into every
  prefix and distort scoring.
- **Treating BM25 defaults as universal.** Lucene's `k1=1.2, b=0.75` is a
  reasonable default for prose of mixed length. It is not optimal for
  titles, product names, ID fields, or very long documents. The exercise
  requires naming at least one case where a non-default similarity is
  deliberate.
- **Forgetting `avgdl` is per-field-per-segment.** A learner who merges
  segments to "make queries faster" and then observes BM25 scores shift is
  seeing the length-normalization statistic recompute. Document this
  behavior; do not treat it as a bug.
- **Confusing `normalizer` with `analyzer`.** A normalizer produces one
  token (no tokenization). An analyzer produces many tokens. Keyword fields
  take normalizers; text fields take analyzers.
- **Not testing the mapping with `_analyze`.** Shipping without running
  `_analyze` against the actual inputs is how a `code_identifiers` field
  ends up tokenized on whitespace only, and every CamelCase search misses
  silently.
- **Making the mapping clever before it is correct.** Multi-analyzer
  fields, `copy_to`, and `search_as_you_type` all have their place. They
  are not the first mapping you write. Start with per-field analyzers and
  a defensible similarity; add complexity when a validation query proves
  you need it.

## 6. References

- Lucene BM25 similarity:
  - `BM25Similarity` javadoc —
    [lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).
  - Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework:
    BM25 and Beyond" — [nowpublishers.com/article/Details/INR-019](https://www.nowpublishers.com/article/Details/INR-019).
- OpenSearch mapping and analysis:
  - Index mappings — [opensearch.org/docs/latest/field-types/](https://opensearch.org/docs/latest/field-types/).
  - Text analysis — [opensearch.org/docs/latest/analyzers/](https://opensearch.org/docs/latest/analyzers/).
  - Tune BM25 (similarity) — [opensearch.org/docs/latest/query-dsl/full-text/match/](https://opensearch.org/docs/latest/query-dsl/full-text/match/).
- Elasticsearch equivalents (API-compatible with OpenSearch at the level the
  exercise uses):
  - `dense_vector` and text fields — [elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html).
  - Similarity module and BM25 parameters — [elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html).
  - `_analyze` API — [elastic.co/guide/en/elasticsearch/reference/current/indices-analyze.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-analyze.html).
- Solr equivalents (for learners choosing Solr):
  - Analyzers, tokenizers, filters — [solr.apache.org/guide/solr/latest/indexing-guide/analyzers.html](https://solr.apache.org/guide/solr/latest/indexing-guide/analyzers.html).
  - Similarities — [solr.apache.org/guide/solr/latest/indexing-guide/schema-elements.html](https://solr.apache.org/guide/solr/latest/indexing-guide/schema-elements.html).
- Risk / governance framings consulted while drafting this reference:
  - OWASP Machine Learning Security Top 10 — [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  - NIST AI Risk Management Framework — [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
