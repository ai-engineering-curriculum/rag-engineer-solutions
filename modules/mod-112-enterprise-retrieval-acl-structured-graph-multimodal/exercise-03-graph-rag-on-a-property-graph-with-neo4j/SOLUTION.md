# Exercise 03 — Solution: Graph RAG on a Property Graph with Neo4j

This is a reference solution for
`exercise-03-graph-rag-on-a-property-graph-with-neo4j.md`. It is
an **implementation** exercise. The learner builds a Graph RAG
pipeline over a Neo4j property graph with four components: a
seeded graph of entities and relationships, a hybrid *anchor*
retriever (vector or lexical over node text) that finds entry
points, a *traversal* step that expands the anchor set into a
subgraph relevant to the question, and a *context assembly* step
that packages nodes and edges into a prompt. The deliverable is
a working pipeline that answers multi-hop questions a flat
retriever would miss.

The teaching point: Graph RAG is not "retrieval, but on a
graph." The vector or lexical layer finds *where to start*;
the traversal does the retrieval work a flat index cannot —
following named relationships across hops, aggregating evidence
across connected entities, and surfacing the shape of the
connection, not just the endpoints. The win case for Graph RAG
is multi-hop questions ("which supplier do our EU customers
share?"). The lose case is single-hop facts ("what is
customer X's email?") — on those, a flat retriever is faster
and simpler.

## 1. Solution overview

The deliverable has six parts:

1. **`graph/schema.cypher`** — the property-graph schema:
   node labels (`Company`, `Person`, `Product`, `Incident`),
   relationship types (`WORKS_AT`, `SUPPLIES`, `AFFECTED_BY`),
   uniqueness constraints, and indexes (including a vector
   index on node `embedding` fields where available).
2. **`graph/seed.cypher`** — a seeded graph small enough to
   reason about by eye (≈ 30 nodes, ≈ 60 relationships) with
   known multi-hop answers.
3. **`src/anchor.py`** — anchor retrieval. Two modes: vector
   anchors via Neo4j's native vector index if the instance
   supports it, or lexical anchors via `CONTAINS` / full-text
   index as a fallback. Returns the top-k entry nodes with a
   score and a snippet.
4. **`src/traversal.py`** — the graph expansion. Starts from
   the anchors, follows named relationships up to a bounded
   depth, collects a *relevance subgraph*, and ranks the
   collected nodes by a combination of anchor similarity and
   graph-distance decay.
5. **`src/assemble.py`** — packaging the subgraph into prompt
   context. Emits a structured section with entities first,
   then an edge-list section ("A `WORKS_AT` B") and a per-node
   snippet section, under a token budget.
6. **`src/pipeline.py` + `tests/`** — the end-to-end
   orchestrator and a test suite with (question, expected
   anchor, expected traversal nodes) tuples. Tests use a
   Testcontainers-managed Neo4j, or a stub driver when a Neo4j
   instance is not available.

Scope: *the pipeline shape and the traversal logic.* The
exercise does not require a production-grade KG — the seeded
graph is enough. If the pipeline answers the multi-hop test
questions correctly, the exercise passes.

## 2. Worked answer or implementation

### 2.1 Graph RAG in one paragraph

The graph is a materialised view of how facts connect. Where a
flat vector index answers "find chunks like this," a graph
answers "find chunks connected to this one in a named way." The
pipeline has three steps. **Anchor** turns the question into
one or more node IDs via a flat retriever — vector, full-text,
or both. **Traverse** walks the graph from the anchors along
named edges up to a bounded depth, collecting a subgraph. The
walk is bounded by edge-type allow-lists (not all edges are
relevant to every question), by depth (usually 1–2 hops), and by
a decay factor (nodes farther from an anchor count less).
**Assemble** serialises the subgraph into prompt context: the
entities, the edges that connect them, and short snippets
capturing the facts.

### 2.2 `graph/schema.cypher` — the property graph

```cypher
// --- constraints ---
CREATE CONSTRAINT company_name_uq IF NOT EXISTS
  FOR (c:Company) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT person_email_uq IF NOT EXISTS
  FOR (p:Person) REQUIRE p.email IS UNIQUE;
CREATE CONSTRAINT product_sku_uq IF NOT EXISTS
  FOR (pr:Product) REQUIRE pr.sku IS UNIQUE;
CREATE CONSTRAINT incident_id_uq IF NOT EXISTS
  FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE;

// --- full-text index (falls back gracefully across community / enterprise) ---
CREATE FULLTEXT INDEX node_text IF NOT EXISTS
  FOR (n:Company|Person|Product|Incident)
  ON EACH [n.name, n.description];

// --- vector index on node.embedding (Neo4j >= 5.11 supports
//     native vector indexes; see refs). On older versions, fall
//     back to the full-text index above and skip this statement. ---
CREATE VECTOR INDEX node_vec IF NOT EXISTS
  FOR (n:Company|Person|Product|Incident)
  ON n.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }};
```

**Choices and why:**

- **Uniqueness constraints on name / email / SKU.** Entity
  resolution is the single most important correctness property
  of a knowledge graph; a duplicated company node routes the
  traversal around the real connections.
- **One full-text index across the labels we traverse.** Keeps
  anchor retrieval simple — one call returns candidates of any
  relevant type.
- **Vector index is optional.** Neo4j added native vector
  indexes starting in version 5.11, and the syntax and options
  have evolved across point releases. The pipeline detects
  the index at startup and falls back to full-text anchors if
  it is not present. See the Neo4j vector-index documentation
  for the current syntax
  ([neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)).

### 2.3 `graph/seed.cypher` — the seeded graph (abbreviated)

```cypher
// Companies
MERGE (apex:Company {name: "Apex Robotics",     region: "EU"})
  SET apex.description = "Industrial robotics vendor based in Berlin.";
MERGE (nova:Company {name: "Nova Components",   region: "EU"})
  SET nova.description = "Precision-component supplier; sells to robotics OEMs.";
MERGE (orio:Company {name: "Orion Logistics",   region: "US"})
  SET orio.description = "Third-party logistics partner used by Apex for US delivery.";

// People
MERGE (ada:Person {email: "ada@apex.example"})
  SET ada.name = "Ada Byrne", ada.title = "Head of Procurement";
MERGE (bo:Person {email: "bo@nova.example"})
  SET bo.name = "Bo Rivera", bo.title = "Account Manager";

// Products
MERGE (actA:Product {sku: "ACT-100"})
  SET actA.name = "Actuator 100", actA.description = "Linear actuator used in Apex arm series.";
MERGE (actB:Product {sku: "ACT-200"})
  SET actB.name = "Actuator 200", actB.description = "High-torque actuator; second-sourced from Nova.";

// Incidents
MERGE (iA:Incident {incident_id: "INC-42"})
  SET iA.description = "Component recall affecting ACT-100 lot 2026Q3.",
      iA.severity    = "high";

// Relationships
MERGE (ada)-[:WORKS_AT]->(apex);
MERGE (bo)-[:WORKS_AT]->(nova);
MERGE (nova)-[:SUPPLIES]->(apex);
MERGE (orio)-[:SERVES]->(apex);
MERGE (actA)-[:MADE_BY]->(apex);
MERGE (actB)-[:SECOND_SOURCED_FROM]->(nova);
MERGE (iA)-[:AFFECTS]->(actA);
```

The seeded graph answers at least these multi-hop questions:

- "Who should Ada contact at Apex's main supplier?"
  (`Ada -WORKS_AT-> Apex <-SUPPLIES- Nova <-WORKS_AT- Bo`)
- "Which products are affected by incident INC-42 and who made
  them?"
  (`INC-42 -AFFECTS-> ACT-100 -MADE_BY-> Apex`)
- "Which companies supply EU-region customers?"
  (filter on `Company.region = 'EU'` after supplier traversal)

### 2.4 `src/anchor.py` — anchor retrieval

```python
"""Anchor retrieval: question → list of entry nodes with scores.

Uses native vector index if present; falls back to full-text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from neo4j import Driver


@dataclass(frozen=True)
class Anchor:
    node_id: int          # internal Neo4j id
    labels: list[str]
    name: str             # canonical name field
    score: float
    snippet: str


class AnchorRetriever:
    def __init__(
        self,
        driver: Driver,
        embedder: Callable[[str], list[float]] | None = None,
        use_vector_index: bool = False,
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._use_vector = use_vector_index and embedder is not None

    def retrieve(self, question: str, k: int = 5) -> list[Anchor]:
        if self._use_vector:
            return self._vector_anchors(question, k)
        return self._lexical_anchors(question, k)

    def _vector_anchors(self, question: str, k: int) -> list[Anchor]:
        qvec = self._embedder(question)  # type: ignore[misc]
        cypher = """
        CALL db.index.vector.queryNodes('node_vec', $k, $qvec)
          YIELD node, score
        RETURN id(node) AS node_id,
               labels(node) AS labels,
               coalesce(node.name, node.incident_id) AS name,
               coalesce(node.description, '') AS snippet,
               score
        """
        with self._driver.session() as session:
            result = session.run(cypher, qvec=qvec, k=k)
            return [
                Anchor(r["node_id"], r["labels"], r["name"], r["score"], r["snippet"])
                for r in result
            ]

    def _lexical_anchors(self, question: str, k: int) -> list[Anchor]:
        cypher = """
        CALL db.index.fulltext.queryNodes('node_text', $q)
          YIELD node, score
        RETURN id(node) AS node_id,
               labels(node) AS labels,
               coalesce(node.name, node.incident_id) AS name,
               coalesce(node.description, '') AS snippet,
               score
        LIMIT $k
        """
        with self._driver.session() as session:
            result = session.run(cypher, q=question, k=k)
            return [
                Anchor(r["node_id"], r["labels"], r["name"], r["score"], r["snippet"])
                for r in result
            ]
```

**Choices and why:**

- **Vector or lexical, not both yet.** A hybrid anchor (RRF
  across vector and lexical) is a legitimate improvement, but
  mod-106 covers fusion in depth; this exercise is about the
  graph half. Pick one anchor retriever and keep the pipeline
  honest about the trade-off.
- **`coalesce(node.name, node.incident_id)`** — different labels
  carry different canonical name fields; coalesce lets one
  Cypher statement cover all labels in the index.
- **No projection of edges in anchor retrieval.** The anchor
  step returns nodes, not subgraphs. Traversal is a separate
  step so the two can be tuned and tested independently.

### 2.5 `src/traversal.py` — bounded graph expansion

```python
"""From anchors, expand along allow-listed edge types with
bounded depth and per-hop decay.

The point is a *small*, *named* subgraph — not every reachable
node."""
from __future__ import annotations

from dataclasses import dataclass, field

from neo4j import Driver

from anchor import Anchor


@dataclass(frozen=True)
class SubgraphNode:
    node_id: int
    labels: tuple[str, ...]
    name: str
    description: str
    score: float  # anchor-similarity * depth-decay


@dataclass(frozen=True)
class SubgraphEdge:
    src_id: int
    dst_id: int
    type: str


@dataclass(frozen=True)
class Subgraph:
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


class Traverser:
    def __init__(
        self,
        driver: Driver,
        allowed_rel_types: tuple[str, ...] = (
            "WORKS_AT", "SUPPLIES", "SERVES", "MADE_BY",
            "SECOND_SOURCED_FROM", "AFFECTS",
        ),
        max_depth: int = 2,
        max_nodes: int = 25,
        decay: float = 0.6,  # per-hop multiplicative decay
    ) -> None:
        self._driver = driver
        self._rels = allowed_rel_types
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._decay = decay

    def expand(self, anchors: list[Anchor]) -> Subgraph:
        if not anchors:
            return Subgraph(nodes=[], edges=[])

        anchor_ids = [a.node_id for a in anchors]
        anchor_scores = {a.node_id: a.score for a in anchors}
        rel_filter = "|".join(self._rels)

        # Variable-length path up to max_depth with an allow-list on
        # relationship types. Returning per-hop information lets us
        # apply the decay and cap.
        cypher = f"""
        UNWIND $anchor_ids AS aid
        MATCH p = (a)-[rels:{rel_filter}*1..{self._max_depth}]-(n)
        WHERE id(a) = aid
        RETURN id(a) AS anchor_id,
               [r IN relationships(p) | type(r)] AS rel_types,
               [node IN nodes(p) | {{
                   id: id(node),
                   labels: labels(node),
                   name: coalesce(node.name, node.incident_id, toString(id(node))),
                   description: coalesce(node.description, '')
               }}] AS path_nodes
        """
        rows = []
        with self._driver.session() as session:
            for rec in session.run(cypher, anchor_ids=anchor_ids):
                rows.append((rec["anchor_id"], rec["rel_types"], rec["path_nodes"]))

        nodes: dict[int, SubgraphNode] = {}
        edges: set[tuple[int, int, str]] = set()

        # Seed with the anchors themselves (score = anchor similarity).
        for a in anchors:
            nodes[a.node_id] = SubgraphNode(
                node_id=a.node_id,
                labels=tuple(a.labels),
                name=a.name,
                description=a.snippet,
                score=a.score,
            )

        for anchor_id, rel_types, path_nodes in rows:
            base_score = anchor_scores[anchor_id]
            for depth, node in enumerate(path_nodes):
                decayed = base_score * (self._decay ** depth)
                existing = nodes.get(node["id"])
                if existing is None or decayed > existing.score:
                    nodes[node["id"]] = SubgraphNode(
                        node_id=node["id"],
                        labels=tuple(node["labels"]),
                        name=node["name"],
                        description=node["description"],
                        score=decayed,
                    )
            for i, rt in enumerate(rel_types):
                src = path_nodes[i]["id"]
                dst = path_nodes[i + 1]["id"]
                edges.add((src, dst, rt))

        # Cap to top-N nodes by score; keep edges whose endpoints survive.
        ranked = sorted(nodes.values(), key=lambda n: -n.score)[: self._max_nodes]
        kept_ids = {n.node_id for n in ranked}
        kept_edges = [SubgraphEdge(s, d, t) for (s, d, t) in edges if s in kept_ids and d in kept_ids]

        return Subgraph(nodes=ranked, edges=kept_edges)
```

**Choices and why:**

- **Allow-list of edge types, not deny-list.** A relationship
  like `VIEWED_BY` would create a star from any busy node to
  every unrelated viewer. Allow-listing forces the pipeline to
  declare which edges the question is about.
- **`*1..max_depth`** with max_depth = 2. Graph fanout grows
  exponentially; depth-3 traversals on real graphs balloon into
  thousands of nodes. The exercise defaults to 2 and lets the
  caller set it higher explicitly, as a reminder.
- **Per-hop multiplicative decay.** The anchor is the ground
  truth; a node two hops away is weaker evidence. A 0.6 decay
  means a 2-hop node contributes 0.36× its anchor's score. Any
  smooth decay works; the point is that distance has to matter.
- **Undirected traversal** (`-[*]-`). Property graphs have
  directions, but for retrieval the reverse-direction fact is
  usually just as useful ("Nova supplies Apex" = "Apex is
  supplied by Nova"). The pipeline keeps both and lets the
  assembler verbalise the edge direction correctly.
- **Cap by top-N score, not top-N by depth.** If one anchor's
  1-hop neighbourhood has fewer high-score nodes than another's
  2-hop neighbourhood, prefer the latter. Score is the truth.

### 2.6 `src/assemble.py` — pack subgraph into prompt context

```python
"""Serialise the subgraph into a prompt-ready block.

Sections:
  1. Entities (one line per node with label)
  2. Relationships (edge list; direction preserved)
  3. Snippets (descriptions, in score order, under a token budget)
"""
from __future__ import annotations

from traversal import Subgraph


def assemble_context(subgraph: Subgraph, max_chars: int = 4000) -> str:
    if not subgraph.nodes:
        return "CONTEXT: (no related entities found)"

    entities = "\n".join(
        f"- {':'.join(n.labels)} \"{n.name}\" (score={n.score:.3f})"
        for n in subgraph.nodes
    )

    name_by_id = {n.node_id: n.name for n in subgraph.nodes}
    edges = "\n".join(
        f"- {name_by_id[e.src_id]} -[{e.type}]-> {name_by_id[e.dst_id]}"
        for e in subgraph.edges
        if e.src_id in name_by_id and e.dst_id in name_by_id
    ) or "(no edges in subgraph)"

    snippets_lines: list[str] = []
    running = 0
    for node in subgraph.nodes:
        if not node.description:
            continue
        line = f"- {node.name}: {node.description}"
        if running + len(line) + 1 > max_chars:
            break
        snippets_lines.append(line)
        running += len(line) + 1

    return (
        "ENTITIES:\n" + entities
        + "\n\nRELATIONSHIPS:\n" + edges
        + "\n\nSNIPPETS:\n" + "\n".join(snippets_lines)
    )
```

**Choices and why:**

- **Three sections, in this order.** The LLM needs the entities
  first (grounding), then the relationships (structure), then
  the snippets (prose evidence). Interleaving them makes the
  model harder to cite accurately — mod-108's citation contract
  depends on a stable context layout.
- **Edge direction preserved.** `A -[SUPPLIES]-> B` is not the
  same as `B -[SUPPLIES]-> A`; the assembler verbalises what the
  graph actually says.
- **Snippet budget is character-based.** A proper token budget
  needs a tokeniser; character count is a reasonable approximation
  for an exercise and is deterministic for testing.

### 2.7 `src/pipeline.py` — end to end

```python
"""Compose anchor → traverse → assemble."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from anchor import AnchorRetriever
from assemble import assemble_context
from traversal import Traverser


@dataclass(frozen=True)
class RAGResult:
    context: str
    anchors: list
    subgraph: object  # Subgraph


def run_graph_rag(
    question: str,
    anchor: AnchorRetriever,
    traverser: Traverser,
    max_context_chars: int = 4000,
) -> RAGResult:
    anchors = anchor.retrieve(question, k=5)
    subgraph = traverser.expand(anchors)
    context = assemble_context(subgraph, max_chars=max_context_chars)
    return RAGResult(context=context, anchors=anchors, subgraph=subgraph)
```

### 2.8 `tests/test_pipeline.py` — golden multi-hop cases

```python
"""Golden tests: question → anchor set → subgraph membership.

These assert *coverage*, not exact scores — scoring is coupled
to the embedder and the engine version."""
from __future__ import annotations

import pytest

# The tests run against a seeded Neo4j (Testcontainers or a
# developer-run instance on bolt://localhost:7687). When Neo4j is
# unavailable, the tests are skipped rather than silently passing.

pytestmark = pytest.mark.integration

MULTI_HOP = [
    (
        "Who should Ada contact at Apex's main supplier?",
        {"Ada Byrne", "Apex Robotics", "Nova Components", "Bo Rivera"},
    ),
    (
        "Which products are affected by incident INC-42 and who made them?",
        {"INC-42", "Actuator 100", "Apex Robotics"},
    ),
]


@pytest.mark.parametrize("question, expected_names", MULTI_HOP)
def test_subgraph_covers_expected_nodes(pipeline, question, expected_names):
    result = pipeline.run(question)
    names = {n.name for n in result.subgraph.nodes}
    missing = expected_names - names
    assert not missing, f"missing from subgraph: {missing}"
```

**What the test asserts:**

- The expected nodes are *present* in the subgraph. A flat
  retriever would find only the anchors (e.g., "Ada") and miss
  "Nova" and "Bo"; the traversal is what completes the answer.
- The test asserts a *set*, not an order. Score ordering is
  coupled to the embedder and is a legitimate thing to tune;
  coverage is what the pipeline guarantees.

## 3. Validation steps

A correct submission runs locally.

1. **Start Neo4j.** `docker run --name neo4j -d -p 7474:7474
   -p 7687:7687 -e NEO4J_AUTH=neo4j/testtest neo4j:5-enterprise`
   (or Community; the pipeline works on both — the vector index
   is optional).
2. **Load schema and seed.** `cypher-shell -u neo4j -p testtest
   -f graph/schema.cypher && cypher-shell -u neo4j -p testtest
   -f graph/seed.cypher`.
3. **Install Python deps.** `pip install neo4j pytest`.
4. **Smoke test.** `python -c "from src.pipeline import run_graph_rag; ..."`
   on one of the multi-hop questions. Inspect the assembled
   context by eye; confirm that Entities + Relationships +
   Snippets are all present and that the edges carry the right
   directions.
5. **Golden tests.** `pytest tests/ -q`. All multi-hop cases
   pass; coverage check reports no missing nodes.
6. **Compare against flat retrieval.** Run the same question
   with only the anchor step (no traversal). Record which nodes
   are missing. The exercise is correct iff the traversal
   recovers the missing nodes.
7. **ACL inheritance from exercise-01.** Add a region filter to
   the Cypher (`WHERE node.region IN $allowed_regions`). Confirm
   that calls from an EU caller never return US-only nodes. The
   row filter rides on the node property, mirroring exercise-01's
   pre-filter pattern.
8. **Fanout sanity.** On the seeded graph, `max_depth=3` returns
   substantially more nodes than `max_depth=2`. Document the
   ratio; it is the "exponential fanout" lesson in a concrete
   number.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Schema | Uniqueness constraints on canonical node keys; at least one full-text index; optional vector index with graceful fallback. |
| Seed data | ≥ 25 nodes, ≥ 2 distinct multi-hop paths, at least one `Incident`-style aggregation node. |
| Anchor step | Returns typed anchors with a score; vector *or* lexical mode selectable at init; the fallback path is tested. |
| Traversal step | Edge-type allow-list; bounded depth (default ≤ 2); per-hop decay; top-N cap on returned nodes. |
| Context assembly | Three distinct sections (Entities, Relationships, Snippets) under a configurable budget. |
| Edge direction | The relationships section preserves direction; verbalisation uses the arrow notation. |
| Tests | Golden multi-hop tests assert *coverage* (set membership), not ordering; a baseline test confirms the flat retriever is insufficient. |
| ACL inheritance | A region or classification field is honoured in the Cypher; the test suite covers at least one ACL case. |
| Determinism | Given the same seed, the same question returns the same subgraph node set. Non-determinism from vector embeddings is isolated to the anchor step. |
| Fail-closed | Empty anchor set returns an empty subgraph and a sentinel context, not a hallucinated answer. |

Fail-fast smells:

- The traversal uses `-[*]-` with no type filter. The edge-type
  allow-list is not optional; without it, the subgraph grows
  with the graph.
- `max_depth` defaults to 3+. Depth-3 traversals on real graphs
  are large; the default needs to be 1 or 2 and the exercise
  should document the fanout.
- The anchor step returns the whole graph when it finds nothing.
  Fail-closed: no anchors → no traversal.
- The context is one flat string of facts. The LLM cannot cite
  a specific edge if the edge is not written as an edge.
- The vector index is required. Neo4j Community without the
  vector index should still run the pipeline on full-text
  anchors.

## 5. Common mistakes

- **Running the pipeline without entity resolution.** Two
  "Apex Robotics" nodes with slightly different spellings
  (`Apex Robotics`, `Apex Robotics, Inc.`) silently split the
  supplier relationship. The uniqueness constraints and a
  dedup step at ingest are not optional.
- **Expanding on every edge type.** Edges have semantics; a
  `VIEWED_BY` edge from a page to a viewer is not "connection"
  in any retrieval sense. Default-deny edge types; the question
  class decides which to include.
- **Treating the graph as a vector DB with hops.** The embedding
  at the node level is one signal; the graph's named structure
  is the other. A pipeline that ranks only by embedding and uses
  the graph only as fanout is doing BFS, not Graph RAG.
- **Serialising the subgraph as JSON for the LLM.** The LLM
  reads prose more reliably than JSON on retrieval context
  (mod-107's context-assembly rules apply here). Keep the three
  sections as prose with a stable layout.
- **Returning hundreds of nodes for one question.** More context
  is not better; it dilutes the signal and wastes tokens. Cap
  the subgraph and let score ordering do the work.
- **Ignoring direction.** The LLM cannot reason about a
  bidirectional edge if the retriever presents one. Preserve
  direction in the assembled context.
- **Writing Cypher with f-string interpolation of user input.**
  Cypher injection is a thing; pass values as `$params`, never
  concatenate.
- **Skipping the flat-retrieval baseline.** The win case for
  Graph RAG is the question where a flat retriever misses an
  answer. Without the baseline, there is no evidence the graph
  earns its cost. Measure the lift.
- **Using the graph for single-hop facts.** "What is Ada's
  email?" is a one-node lookup; there is no graph to walk.
  A good pipeline knows when to short-circuit back to a flat
  retrieval or a direct lookup.

## 6. References

Neo4j documentation (primary sources for schema, indexing, and
Cypher syntax):

- Cypher manual —
  [neo4j.com/docs/cypher-manual/current/](https://neo4j.com/docs/cypher-manual/current/).
  Relevant: `MATCH`, variable-length paths, `MERGE`, constraints.
- Full-text search indexes —
  [neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/).
- Vector indexes —
  [neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).
- Neo4j Python driver —
  [neo4j.com/docs/python-manual/current/](https://neo4j.com/docs/python-manual/current/).

Governance frameworks (for the ACL and risk framing):

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant: ML06 AI Supply Chain Attacks — a graph built by
  scraping third-party sources inherits the trustworthiness of
  those sources; a poisoned node ripples through the traversal.
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: *Exfiltration via ML Inference API* — a Graph RAG
  retriever can walk into neighbourhoods the user is not
  authorised for if ACLs are per-document but traversal is
  per-edge. Enforce ACLs on every hop.
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  Relevant: *Map* (document the relationship types the pipeline
  treats as retrieval edges) and *Measure* (the baseline-vs-
  Graph-RAG comparison is the measure of value).

Downstream consumers:

- exercise-01 (ACL patterns) is the authorisation context; the
  Cypher traversal must respect the same policy. Property-level
  filters on every node are the pre-filter pattern.
- exercise-05 (hybrid router) dispatches to this Graph RAG
  pipeline for questions with a connection shape ("who is
  related to X via Y?"). The router does not know the graph;
  it knows the question class.
