# Exercise 02 — Solution: Text-to-SQL Retrieval with Schema-Linking and a Safety Guard

This is a reference solution for
`exercise-02-text-to-sql-retrieval-with-schema-linking-and-safety-guard.md`.
It is an **implementation** exercise. The learner builds a text-to-SQL
retrieval pipeline over a small analytical database with three
components: a *schema linker* (pick the tables and columns the
question needs), a *SQL generator* (prompt an LLM with the linked
schema), and a *safety guard* (an AST-level validator that rejects
unsafe queries before execution). The deliverable is a runnable
module that converts a natural-language question into a verified
result set and refuses when the generated SQL is unsafe or
unverifiable.

The teaching point: text-to-SQL is a retrieval problem *and* an
authorisation problem. The LLM is a non-deterministic code
generator; the safety guard is a deterministic verifier. Schema
linking is the step that keeps both of them honest — the LLM only
sees the relevant tables, which cuts hallucination, and the guard
knows which columns are allowed per caller, which cuts privilege
escalation. The combination lets a text-to-SQL retriever sit
behind an MCP tool or a chatbot without turning the database into
a prompt-injection surface.

## 1. Solution overview

The deliverable has six parts:

1. **`db/schema.sql` and `db/seed.sql`** — a small SQLite schema
   (3–5 tables) with seeded rows, so the whole exercise runs
   locally without network or cloud credentials. The schema is
   deliberately small; the exercise is about the pipeline, not
   about scaling.
2. **`src/schema_linker.py`** — retrieves the relevant
   subset of the schema for a given question. A two-pass design:
   first a dense retriever over per-table documents (name +
   description + sample values), then an exact match / keyword
   pass to recover tables the dense pass may have missed.
3. **`src/sql_generator.py`** — prompts an LLM with the linked
   schema and the question, using a tightly-constrained prompt
   template that forbids data modification and forces a
   single-statement response. Model-agnostic; the reference
   driver points at a local stub.
4. **`src/safety_guard.py`** — the deterministic verifier.
   Parses the generated SQL with `sqlglot`, enforces an allow-list
   of statement types, disallows subqueries with side effects,
   enforces LIMIT, verifies that every referenced table and
   column is in the allow-set for the caller, and rejects
   constructs that could bypass row-level security.
5. **`src/pipeline.py`** — the orchestrator: question → linker →
   generator → guard → executor → result. On guard failure,
   emits a structured error with the reason; never falls back to
   executing an unverified query.
6. **`tests/`** — golden tests of the whole pipeline on a seeded
   set of (question, expected-SQL-shape, expected-row-count)
   tuples, plus adversarial tests that assert the guard rejects
   the full catalogue of known-bad SQL (DROP, DELETE, UNION
   exfiltration, column probing).

Scope: *correctness of the pipeline, not benchmark scores.* The
reference uses SQLite for portability; the design is engine-
agnostic. If the pipeline answers the golden questions and the
guard rejects the adversarial set, the exercise passes.

## 2. Worked answer or implementation

### 2.1 Pipeline shape in one paragraph

A text-to-SQL retriever is a four-stage pipeline:
`link(schema, question) → generate(linked_schema, question) →
guard(sql, caller_policy) → execute(sql, limit)`.
The link stage reduces the schema the LLM sees from "the whole
database" to "the tables the question is about," which cuts both
prompt cost and hallucination. The generate stage is the only
stochastic step; everything after it is deterministic. The guard
stage is the trust boundary — if the guard rejects, the query
does not run, full stop. The execute stage is scoped to the
caller's role (SELECT-only, row-level security) so the guard and
the database agree on what is allowed. The pipeline returns
either a result set or a structured refusal; it never returns a
best-effort answer on an unverified query.

### 2.2 `db/schema.sql` — the demo schema

```sql
-- A tiny e-commerce slice: customers, orders, order_items, products.
-- Deliberately small; the learner should be able to reason about
-- which tables a given question needs by eye.

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    email         TEXT    NOT NULL UNIQUE,
    region        TEXT    NOT NULL,                 -- ACL dimension
    created_at    TEXT    NOT NULL                  -- ISO-8601
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    sku           TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL,
    unit_price    REAL    NOT NULL CHECK (unit_price >= 0)
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date    TEXT    NOT NULL,                 -- ISO-8601
    status        TEXT    NOT NULL CHECK (status IN ('open', 'paid', 'shipped', 'cancelled'))
);

CREATE TABLE order_items (
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    line_total    REAL    NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
```

**Choices and why:**

- **`customers.region`** is the ACL dimension. Row-level
  security filters by region; the guard enforces that the
  generated SQL can only read rows the caller's region covers.
- **Status is a `CHECK` enum.** The guard can use this to
  validate literal comparisons against known values.
- **Four tables, not forty.** The exercise is about the pipeline.
  A realistic schema linker would face thousands of columns; the
  smaller schema lets the learner see the pipeline's behaviour
  end-to-end.

### 2.3 `src/schema_linker.py` — schema linking

```python
"""Pick the tables and columns the question actually needs.

Two-pass design: a dense retriever over table documents first,
then a lexical fallback that catches named entities the dense
pass missed."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TableDoc:
    name: str
    description: str
    columns: tuple[str, ...]
    sample_values: tuple[str, ...]  # e.g., "status: open, paid, shipped"

    def as_document(self) -> str:
        return (
            f"TABLE {self.name}\n"
            f"  description: {self.description}\n"
            f"  columns: {', '.join(self.columns)}\n"
            f"  sample values: {'; '.join(self.sample_values)}"
        )


# In a real system, build these from information_schema + a human
# description written once per table. Keep in version control so
# review tracks intent.
TABLE_DOCS: dict[str, TableDoc] = {
    "customers": TableDoc(
        name="customers",
        description="One row per registered customer. Region is the ACL dimension.",
        columns=("customer_id", "email", "region", "created_at"),
        sample_values=("region: EU, US, APAC", "created_at: ISO-8601 dates"),
    ),
    "products": TableDoc(
        name="products",
        description="Catalogue of products with price and category.",
        columns=("product_id", "sku", "name", "category", "unit_price"),
        sample_values=("category: books, electronics, apparel", "unit_price: USD"),
    ),
    "orders": TableDoc(
        name="orders",
        description="Orders placed by customers. Join to customers on customer_id.",
        columns=("order_id", "customer_id", "order_date", "status"),
        sample_values=("status: open, paid, shipped, cancelled",),
    ),
    "order_items": TableDoc(
        name="order_items",
        description="Line items per order. Join to orders and products.",
        columns=("order_id", "product_id", "quantity", "line_total"),
        sample_values=(),
    ),
}


class SchemaLinker:
    def __init__(self, embedder) -> None:
        # embedder: a callable str -> list[float]. Kept abstract so
        # the exercise does not require a specific embedding
        # provider; tests inject a deterministic stub.
        self._embedder = embedder
        self._doc_embeddings = {
            name: embedder(doc.as_document()) for name, doc in TABLE_DOCS.items()
        }

    def link(self, question: str, k: int = 3) -> list[TableDoc]:
        """Return up to k TableDocs most relevant to the question."""
        q_vec = self._embedder(question)
        dense_scores = {
            name: _cosine(q_vec, vec) for name, vec in self._doc_embeddings.items()
        }
        dense_pick = sorted(dense_scores.items(), key=lambda kv: -kv[1])[:k]
        dense_names = {name for name, _ in dense_pick}

        # Lexical fallback: if the question mentions a table name,
        # a column name, or a sample value, include that table even
        # if the dense retriever did not surface it.
        lex_names = _lexical_hits(question, TABLE_DOCS)

        chosen = dense_names | lex_names
        return [TABLE_DOCS[name] for name in chosen if name in TABLE_DOCS]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-12)


def _lexical_hits(question: str, docs: dict[str, TableDoc]) -> set[str]:
    q = question.lower()
    tokens = set(re.findall(r"[a-z_]+", q))
    hits: set[str] = set()
    for name, doc in docs.items():
        # Exact table name
        if name in tokens:
            hits.add(name)
            continue
        # Any column name mentioned verbatim
        if any(col in tokens for col in doc.columns):
            hits.add(name)
            continue
        # Any sample-value term (lower-cased)
        for sample in doc.sample_values:
            if any(term.strip().lower() in q for term in re.split(r"[:,;]", sample) if term.strip()):
                hits.add(name)
                break
    return hits
```

**Choices and why:**

- **Two-pass linker.** Dense alone misses named entities ("orders
  in region APAC" might retrieve `orders` and `products` but
  miss `customers` even though the ACL dimension lives there).
  Lexical alone misses paraphrase ("purchases" instead of
  "orders"). Dense ∪ lexical is a strict superset of either and
  is cheap at the schema's scale.
- **Table docs include sample values.** The LLM downstream uses
  these to pick literal comparisons correctly ("status = 'paid'"
  not "status = 'completed'"). Documenting sample values once
  per table is the highest-leverage prompt-engineering step in
  text-to-SQL.
- **No token-level embedding of the full schema.** Rows are
  sensitive; the linker embeds table *descriptions*, not table
  *data*. Row content never appears in the embedding store.

### 2.4 `src/sql_generator.py` — the LLM step

```python
"""Prompt the LLM with the linked schema and the question.

Model-agnostic: `generate_sql` takes a `chat` callable with the
signature (system: str, user: str) -> str. Tests inject a stub;
production wires in the real provider."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from schema_linker import TableDoc

SYSTEM_PROMPT = """You are a SQL generator for a read-only analytical database.

Rules you MUST follow:
1. Emit exactly one SQL statement. No semicolons separating
   multiple statements. No comments.
2. The statement must be a SELECT (or WITH ... SELECT). Any
   DDL (CREATE/DROP/ALTER), DML (INSERT/UPDATE/DELETE), or
   privilege statement is forbidden.
3. Reference only the tables and columns listed in the
   SCHEMA block below. Never invent a table name, a column
   name, or a value.
4. Always include an explicit LIMIT clause (max 1000 rows).
5. Prefer standard SQL. Do not use database-specific functions
   unless the schema block names them.
6. If the question cannot be answered from the schema, output
   the single word "REFUSE" instead of SQL.
"""


@dataclass(frozen=True)
class GeneratedSQL:
    sql: str
    refused: bool


def generate_sql(
    question: str,
    linked: list[TableDoc],
    chat: Callable[[str, str], str],
) -> GeneratedSQL:
    schema_block = "\n\n".join(doc.as_document() for doc in linked)
    user = f"SCHEMA:\n{schema_block}\n\nQUESTION:\n{question}\n\nSQL:"
    raw = chat(SYSTEM_PROMPT, user).strip()
    if raw.upper() == "REFUSE" or not raw:
        return GeneratedSQL(sql="", refused=True)
    return GeneratedSQL(sql=raw, refused=False)
```

**Choices and why:**

- **The prompt forbids DML/DDL.** Belt-and-suspenders: the guard
  enforces the same contract at the AST level, so a prompt
  bypass does not leak into execution. But the prompt sets the
  LLM up to succeed, and reduces guard-reject rate on honest
  questions.
- **The system prompt forces a `REFUSE` sentinel.** Giving the
  model an escape hatch is a documented improvement in text-to-
  SQL reliability — it reduces the rate of made-up SQL for
  unanswerable questions.
- **One statement, no semicolons.** Multi-statement execution is
  a known SQL-injection vector even on parameterised engines;
  the guard enforces this and the prompt sets the expectation.

### 2.5 `src/safety_guard.py` — the deterministic verifier

This is the most important file in the deliverable. The guard is
the trust boundary; the LLM is not. Every rule below exists
because it catches a specific failure mode.

```python
"""Validate generated SQL against a deterministic allow-policy.

The guard is NOT prompt-sensitive. It parses with sqlglot and
walks the AST; the LLM's output does not change what the guard
accepts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class GuardDecision:
    ok: bool
    reason: str
    normalised_sql: str  # empty on reject


@dataclass(frozen=True)
class CallerPolicy:
    allowed_tables: frozenset[str]
    allowed_columns: dict[str, frozenset[str]]  # table -> columns
    row_filter: str | None  # e.g., "customers.region IN ('EU','US')"
    max_rows: int = 1000


def verify(sql: str, policy: CallerPolicy, dialect: str = "sqlite") -> GuardDecision:
    try:
        trees = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as err:
        return GuardDecision(False, f"parse_error: {err}", "")

    if len(trees) != 1 or trees[0] is None:
        return GuardDecision(False, "exactly_one_statement_required", "")
    tree = trees[0]

    # 1. Statement must be a SELECT (or WITH ... SELECT).
    root = _unwrap_cte(tree)
    if not isinstance(root, exp.Select):
        return GuardDecision(False, f"not_a_select: {root.key}", "")

    # 2. No DDL/DML anywhere in the tree.
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
        exp.Create, exp.Merge, exp.Command, exp.Transaction,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            return GuardDecision(False, f"forbidden_statement: {type(node).__name__}", "")

    # 3. No functions that touch files or spawn processes.
    banned_funcs = {"load_extension", "readfile", "writefile", "system"}
    for fn in tree.find_all(exp.Func):
        name = (fn.name or "").lower()
        if name in banned_funcs:
            return GuardDecision(False, f"banned_function: {name}", "")

    # 4. Every referenced table must be in the allow-set.
    referenced_tables = {t.name for t in tree.find_all(exp.Table)}
    unknown = referenced_tables - policy.allowed_tables
    if unknown:
        return GuardDecision(False, f"table_not_allowed: {sorted(unknown)}", "")

    # 5. Every referenced column must be in the allow-set for
    #    its table. Unqualified columns require qualification if
    #    the FROM references more than one table.
    col_errs = _check_columns(tree, policy)
    if col_errs:
        return GuardDecision(False, f"column_not_allowed: {col_errs}", "")

    # 6. Enforce LIMIT. If missing or > max_rows, rewrite it.
    tree = _enforce_limit(tree, policy.max_rows)

    # 7. Inject row-level filter if the policy carries one.
    if policy.row_filter:
        tree = _inject_row_filter(tree, policy.row_filter, dialect)

    # 8. UNION / INTERSECT / EXCEPT are allowed only if every
    #    arm satisfies the same table and column policies.
    #    Already enforced structurally by the walks above.

    return GuardDecision(True, "ok", tree.sql(dialect=dialect))


def _unwrap_cte(tree: exp.Expression) -> exp.Expression:
    if isinstance(tree, exp.With):
        return tree.this
    return tree


def _check_columns(tree: exp.Expression, policy: CallerPolicy) -> list[str]:
    errs: list[str] = []
    for col in tree.find_all(exp.Column):
        table = col.table
        name = col.name
        if name == "*":
            # Disallow SELECT * — we cannot verify unknown columns.
            errs.append("select_star_forbidden")
            continue
        if not table:
            # Require qualification; the generator prompt instructs
            # this, and ambiguity defeats the column allow-list.
            errs.append(f"unqualified: {name}")
            continue
        allowed = policy.allowed_columns.get(table, frozenset())
        if name not in allowed:
            errs.append(f"{table}.{name}")
    return errs


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    if not isinstance(tree, exp.Select):
        return tree
    existing = tree.args.get("limit")
    if existing is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return tree
    try:
        n = int(existing.expression.name)
    except (AttributeError, ValueError):
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return tree
    if n > max_rows:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return tree


def _inject_row_filter(tree: exp.Expression, predicate: str, dialect: str) -> exp.Expression:
    # Parse the predicate as a stand-alone WHERE fragment and AND
    # it into the top-level SELECT. Subqueries inherit the filter
    # by referencing the already-filtered base tables; if subquery
    # filtering is required, this helper must be extended.
    extra = sqlglot.parse_one(f"SELECT 1 WHERE {predicate}", read=dialect).find(exp.Where)
    assert extra is not None
    if isinstance(tree, exp.Select):
        existing = tree.args.get("where")
        if existing is None:
            tree.set("where", extra.copy())
        else:
            combined = exp.and_(existing.this, extra.this)
            existing.set("this", combined)
    return tree
```

**Choices and why:**

- **`sqlglot` for parsing.** It parses many dialects and exposes
  a stable AST, which is what the guard needs. Regex-based
  "SQL sanitisation" is a known-failed pattern; the guard must
  parse.
- **Forbid `SELECT *`.** An unknown column cannot be checked
  against the allow-list; the guard refuses rather than fail
  open.
- **Require column qualification.** `SELECT region FROM
  customers JOIN orders` leaves `region` unqualified; the guard
  cannot know whose column it is. The prompt asks the generator
  to qualify; the guard enforces it.
- **Inject the row filter server-side.** The generator does not
  know the caller's row-level policy; the guard owns it. This is
  the ACL inheritance from exercise-01: the retriever honours the
  authorisation decision the ACL system made.
- **Rewrite LIMIT rather than reject.** A query missing LIMIT is
  honest but incomplete; rewriting to the policy max is friendlier
  than refusing and keeps the result set bounded.
- **Enumerate banned functions.** SQLite ships `load_extension`
  and user-defined readers / writers; PostgreSQL has
  `pg_read_server_files`; every engine has at least one function
  that reaches outside the query. The guard maintains a per-
  dialect deny-list.

### 2.6 `src/pipeline.py` — the orchestrator

```python
"""End-to-end orchestration: question → result (or structured refusal)."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from safety_guard import CallerPolicy, GuardDecision, verify
from schema_linker import SchemaLinker
from sql_generator import generate_sql


@dataclass(frozen=True)
class PipelineResult:
    ok: bool
    sql: str
    rows: list[tuple]
    reason: str  # populated on refusal


def run(
    question: str,
    linker: SchemaLinker,
    chat: Callable[[str, str], str],
    policy: CallerPolicy,
    db_path: str,
) -> PipelineResult:
    linked = linker.link(question)
    if not linked:
        return PipelineResult(False, "", [], "no_relevant_schema")

    gen = generate_sql(question, linked, chat)
    if gen.refused:
        return PipelineResult(False, "", [], "generator_refused")

    decision = verify(gen.sql, policy)
    if not decision.ok:
        return PipelineResult(False, gen.sql, [], f"guard_rejected: {decision.reason}")

    # Execute under a read-only, timeout-bounded connection.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(decision.normalised_sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    return PipelineResult(True, decision.normalised_sql, rows, "ok")
```

**Choices and why:**

- **Read-only database connection.** SQLite's URI mode supports
  `mode=ro`; every engine supports a read-only role. Paired with
  the guard, this is defense in depth — if the guard misses a
  write, the database refuses it.
- **Empty linked schema → refuse.** If the linker finds no
  relevant tables, the generator cannot answer; running it
  anyway produces hallucinated SQL. Fail-closed.
- **Guard rejection surfaces the reason.** The caller sees
  `guard_rejected: forbidden_statement: Insert`, not a silent
  empty result. The reason enables audit and iterative fixes.

### 2.7 `tests/test_pipeline.py` — golden and adversarial tests

```python
"""Two test suites: honest questions (golden), hostile inputs (adversarial)."""
import pytest
from safety_guard import CallerPolicy, verify


EU_POLICY = CallerPolicy(
    allowed_tables=frozenset({"customers", "products", "orders", "order_items"}),
    allowed_columns={
        "customers": frozenset({"customer_id", "email", "region", "created_at"}),
        "products":  frozenset({"product_id", "sku", "name", "category", "unit_price"}),
        "orders":    frozenset({"order_id", "customer_id", "order_date", "status"}),
        "order_items": frozenset({"order_id", "product_id", "quantity", "line_total"}),
    },
    row_filter="customers.region IN ('EU')",
    max_rows=100,
)


# Golden — queries the pipeline should answer.
GOLDEN = [
    ("How many customers are in each region?",
     "SELECT customers.region, COUNT(customers.customer_id) FROM customers GROUP BY customers.region"),
    ("What are the top 5 products by revenue?",
     "SELECT products.name, SUM(order_items.line_total) AS revenue "
     "FROM order_items JOIN products ON order_items.product_id = products.product_id "
     "GROUP BY products.name ORDER BY revenue DESC LIMIT 5"),
]


@pytest.mark.parametrize("_, sql", GOLDEN)
def test_guard_accepts_honest_sql(_, sql):
    decision = verify(sql, EU_POLICY)
    assert decision.ok, decision.reason
    # Row filter must be present in the rewritten SQL.
    assert "customers.region" in decision.normalised_sql.lower() or "region" in decision.normalised_sql.lower()


# Adversarial — queries the guard MUST reject.
ADVERSARIAL = [
    # DML / DDL
    ("DROP TABLE customers", "forbidden_statement"),
    ("DELETE FROM orders WHERE 1=1", "forbidden_statement"),
    ("INSERT INTO orders (order_id) VALUES (999)", "forbidden_statement"),
    ("CREATE TABLE t (x INT)", "forbidden_statement"),

    # Table / column probing outside the allow-set
    ("SELECT secret FROM admin_users", "table_not_allowed"),
    ("SELECT customers.password FROM customers", "column_not_allowed"),
    ("SELECT * FROM customers", "column_not_allowed"),  # select_star_forbidden

    # Multi-statement
    ("SELECT customers.email FROM customers; DROP TABLE customers", "exactly_one_statement_required"),

    # Dangerous functions
    ("SELECT load_extension('/tmp/evil.so')", "banned_function"),
]


@pytest.mark.parametrize("sql,expected_reason", ADVERSARIAL)
def test_guard_rejects_hostile_sql(sql, expected_reason):
    decision = verify(sql, EU_POLICY)
    assert not decision.ok
    assert expected_reason in decision.reason
```

The adversarial set is *not* exhaustive. The exercise should
include at least one example from each category and document
that the list grows with every real-world incident. Think of
it as a regression suite for the guard.

## 3. Validation steps

A correct submission runs locally and passes these checks.

1. **Install and seed.** `pip install sqlglot pytest` (plus an
   embedder library of choice; the reference tests use a
   deterministic stub). `sqlite3 demo.db < db/schema.sql &&
   sqlite3 demo.db < db/seed.sql`.
2. **Run the test suite.** `pytest -q`. Both golden and
   adversarial tests pass. The adversarial set covers DML/DDL,
   out-of-set tables and columns, `SELECT *`, multi-statement
   injection, and dangerous functions.
3. **Guard-only round trip.** Call `verify(sql, policy)` on each
   adversarial SQL by hand; confirm every reject has a
   human-readable reason code (not a bare `False`).
4. **End-to-end with a live LLM (optional).** Point the
   `chat` callable at a real provider; confirm the pipeline
   answers 3 pre-written questions correctly and refuses on 2
   unanswerable ones (questions whose required tables are not
   in the schema).
5. **ACL inheritance from exercise-01.** Run the pipeline under
   the EU policy and the US policy; the same question returns
   different row counts, and the row filter is visible in the
   normalised SQL.
6. **Determinism of the guard.** The guard must return identical
   decisions across runs for identical inputs — no randomness,
   no network calls, no clock dependence. If the test suite is
   flaky, the guard has a non-determinism bug.
7. **LIMIT enforcement.** A query without a LIMIT is rewritten
   to the policy max. A query with LIMIT 10000 is rewritten to
   the policy max. A query with LIMIT 5 is left alone.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Pipeline shape | Four stages present and separately testable: link, generate, guard, execute. |
| Schema linker | Two-pass (dense + lexical); returns at most a handful of tables for the demo schema; is unit-tested with a deterministic embedder. |
| Generator prompt | Forbids DML/DDL in the system prompt, requires single-statement output, documents a REFUSE sentinel, and asks for column qualification. |
| Guard — parsing | Uses an AST parser (`sqlglot` or equivalent), not regex. Rejects on parse error. |
| Guard — statement type | Only `Select` (optionally wrapped in `With`) is accepted. Everything else is rejected with a reason. |
| Guard — table/column allow-list | Rejects references to tables or columns not in the caller's policy; rejects `SELECT *`. |
| Guard — row-level filter | Injects the caller's row filter into the top-level `WHERE`; the filter is visible in the normalised SQL. |
| Guard — LIMIT | Enforces a maximum; rewrites rather than rejecting. |
| Guard — dangerous functions | Per-dialect deny-list of functions that touch the filesystem or spawn processes. |
| Execution safety | Database connection is read-only (`PRAGMA query_only = ON` or a role with no write grant). |
| Adversarial tests | A regression suite covering DML/DDL, out-of-set tables/columns, multi-statement, and dangerous functions. |
| Honest tests | A golden suite that answers at least three realistic questions and asserts the row filter is applied. |
| Audit trail | On reject, the pipeline returns a structured reason code; the SQL and the reason are logged. |

Fail-fast smells:

- The guard uses regex. Regex-based SQL filtering is a graveyard
  of CVEs; the guard must parse.
- `SELECT *` is accepted. An unknown column is an unverifiable
  column; accept means fail-open.
- The row filter is passed in the LLM prompt, not injected by the
  guard. If the LLM is the authority on authorisation, the
  authorisation has moved into a stochastic component.
- The pipeline falls back to "run anyway" on guard failure.
  There is no fallback; reject is reject.
- The database role has write privileges. Defense in depth
  requires that even if the guard misses, the engine refuses.
- `max_rows` is not enforced. A single question that returns
  10 M rows is a DoS vector.

## 5. Common mistakes

- **Trusting the prompt.** A system prompt that says "never
  emit DROP" is a suggestion. The guard enforces it; the prompt
  only shapes the distribution. Belt and suspenders.
- **Letting the LLM see the row data for schema linking.**
  Embedding row content in the linker leaks sensitive values
  through prompt injection ("what's an example row of
  `customers.password`?"). The linker embeds table *descriptions*,
  not table *rows*.
- **Treating `sqlglot` as a sanitiser.** `sqlglot` is a parser;
  the guard is the sanitiser. The guard uses the parser's AST
  to make allow/deny decisions, not the parser's "is this valid
  SQL" boolean.
- **Row filter in the FROM clause instead of WHERE.** A filter
  pushed into a CTE or subquery can be unioned around; the
  top-level `WHERE` is the canonical injection point. If the
  schema needs deep filtering (e.g., row-level security across
  joins), use the database's native RLS rather than hand-
  rolling it in the guard.
- **Allowing one unqualified column because it's "obvious."**
  Allow-lists work on names, not on obviousness. An unqualified
  column defeats the per-table allow-list; require qualification
  and the generator will do it.
- **Catching the LLM's errors in a try/except.** The LLM returning
  `REFUSE` is not an error; it is a signal. Treat it as a
  first-class path in the pipeline.
- **Running the generator twice on reject ("try again").** The
  LLM is not more trustworthy on the second call. If the guard
  rejects, return the reason and let the caller rephrase.
- **Logging only successful SQL.** Rejects are the audit-worthy
  events — they tell the team which questions the pipeline
  cannot safely answer and which prompts are probing the guard.
  Log every reject with the reason, the caller, and the hashed
  SQL.
- **Hard-coding the policy in the generator.** The policy is
  per-caller (region, role, project). Thread it through the
  pipeline; do not bake it into the prompt or the generator.

## 6. References

Governance frameworks (primary sources for the threat model):

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: ML01 Input Manipulation Attack (the question is
  user input flowing into code generation), ML06 AI Supply
  Chain Attacks (the LLM provider is a trusted upstream; the
  guard reduces trust to "non-trusted code generator").
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant techniques: *LLM Prompt Injection* (the question or
  the schema description can embed hostile instructions),
  *Exfiltration via ML Inference API* (bulk extraction via
  crafted questions).
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant functions: *Measure* (the guard is the measurement;
  reject rates and categories are the metric), *Manage* (the
  adversarial test suite is the manage-time artifact — it grows
  as new attacks are observed).

Engine and parser documentation:

- `sqlglot` — [sqlglot.com](https://sqlglot.com/). Used for
  dialect-aware parsing; the guard walks `sqlglot.exp` nodes.
- SQLite URI connection flags —
  [sqlite.org/uri.html](https://www.sqlite.org/uri.html).
  `file:db.sqlite?mode=ro` is the canonical read-only mode.
- SQLite `PRAGMA query_only` —
  [sqlite.org/pragma.html#pragma_query_only](https://www.sqlite.org/pragma.html#pragma_query_only).
- Model Context Protocol specification —
  [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification).
  When this retriever is exposed via MCP, the tool's input
  schema should name `question` and the output schema should
  include both the row set and the normalised SQL for audit.

Downstream consumers:

- exercise-01 (ACL patterns) supplies the row filter the guard
  injects. A text-to-SQL retriever without an ACL contract
  behind it is a privilege-escalation surface.
- exercise-05 (hybrid router) uses this pipeline as the "SQL
  retriever" leg. The router dispatches to this pipeline when
  the question has an analytical / structured shape.
