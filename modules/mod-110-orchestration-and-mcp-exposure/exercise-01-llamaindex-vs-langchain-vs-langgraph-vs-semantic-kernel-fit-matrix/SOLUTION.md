# Exercise 01 — Solution: LlamaIndex vs. LangChain vs. LangGraph vs. Semantic Kernel — Fit Matrix

This is a reference solution for
`exercise-01-llamaindex-vs-langchain-vs-langgraph-vs-semantic-kernel-fit-matrix.md`.
It is a **design** exercise. The learner picks a retrieval-orchestration
framework for four concrete scenarios drawn from earlier mod-110 work
and writes a defensible recommendation. The deliverable is the matrix,
the per-scenario call, and the kill-switch that would flip the call
later.

The teaching point: these four frameworks solve overlapping but
non-identical problems, and the pick is driven by the *shape of the
retrieval graph* and the *control-flow guarantees the runtime needs*,
not by vendor affinity or feature count. A framework is the right pick
when it makes the hard parts (sub-question decomposition, loop
termination, state inspection, failover) cheap and the easy parts
(a single retriever call) at worst neutral. It is the wrong pick when
the team would end up reimplementing the framework's missing
primitives on top of it.

## 1. Solution overview

The deliverable has four parts:

1. **`matrix/fit_matrix.md`** — the comparison table. Rows are the
   evaluation axes (control flow, state, retrieval primitives,
   streaming, observability, deployability, lock-in). Columns are
   the four frameworks. Cells are scored (`strong` / `adequate` /
   `weak`) with a one-sentence justification pointing at the
   framework's own documented primitives — no vendor marketing
   claims.
2. **`matrix/scenarios.md`** — four scenarios, each with a
   one-paragraph recommendation and the single-line justification
   that would hold up in a design review.
3. **`matrix/decision_tree.md`** — a decision tree (5–7 nodes) that
   routes from a scenario description to a framework pick. The
   intent is "what question answers which framework wins" not
   "which features does each framework have."
4. **`notes/kill_switch.md`** — for each per-scenario pick, the
   signal that would force a re-evaluation (e.g., "if the loop
   grows a second branch, LangGraph wins over LlamaIndex").

Scope: *framework selection only*. The exercises that follow build
on these choices — exercise-02 implements the LlamaIndex router,
exercise-03 implements the LangGraph loop, exercises 04/05 implement
the MCP and native-tool exposure that is framework-agnostic. The
goal here is to make the pick defensible, not to write the code.

## 2. Worked answer or implementation

### 2.1 The four frameworks in one paragraph each

- **LlamaIndex** is a *retrieval-first* framework. Its core
  abstractions are `Index`, `Retriever`, `QueryEngine`, `NodeParser`,
  and `ResponseSynthesizer`. The sweet spot is: documents in,
  retrieval graph wired declaratively, one `query()` call out. The
  built-in `RouterQueryEngine` and `SubQuestionQueryEngine` cover
  intent routing and decomposition without extra code (see
  exercise-02). It is weakest when the orchestration needs
  cycles, human-in-the-loop interrupts, or long-lived state.
  Reference: LlamaIndex docs,
  [docs.llamaindex.ai](https://docs.llamaindex.ai/).
- **LangChain** is a *component library* for LLM applications:
  chains, agents, retrievers, memory, output parsers. It is
  framework-agnostic about control flow — a `Runnable` composes
  with `|` and executes top-to-bottom. It has the broadest
  integration surface (vector stores, model providers, tool
  wrappers) and the loosest opinion about how to wire them. It is
  weakest when the graph is non-linear or stateful beyond a single
  chain invocation. Reference: LangChain docs,
  [python.langchain.com/docs](https://python.langchain.com/docs/).
- **LangGraph** is a *stateful graph runtime* authored by the
  LangChain team. Nodes are functions, edges are typed transitions,
  state is an explicit `TypedDict` that nodes read and mutate.
  Cycles are first-class; checkpointers persist state across
  interrupts; human-in-the-loop breakpoints are built in. It is
  the right pick when the orchestration is a graph with loops,
  conditional branches, or long-running state. It is weakest when
  the task is a single retrieval call — the graph overhead is
  unnecessary. Reference: LangGraph docs,
  [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/).
- **Semantic Kernel** is Microsoft's *plugin-and-planner* SDK.
  Skills are typed functions, plugins group them, planners
  (Handlebars, Function-calling, Stepwise) select which to call.
  Native C#, Python, and Java bindings; tight Azure OpenAI and
  Azure AI Search integration. The sweet spot is enterprise
  stacks already on .NET or Azure AI — the retrieval primitives
  ride on Azure AI Search, Cosmos DB, or Redis via first-party
  connectors. It is weakest outside that stack, where the ecosystem
  is thinner. Reference: Semantic Kernel docs,
  [learn.microsoft.com/semantic-kernel](https://learn.microsoft.com/en-us/semantic-kernel/).

### 2.2 `matrix/fit_matrix.md` — the comparison table

Scoring legend: **S** = strong / first-class primitive, **A** = adequate /
achievable with light code, **W** = weak / requires significant custom
scaffolding or sits outside the framework's intended use.

| Axis | LlamaIndex | LangChain | LangGraph | Semantic Kernel |
|---|---|---|---|---|
| **Linear retrieve → synth chain** | S — `QueryEngine` is the default unit of composition. | S — `Runnable` pipe is the canonical idiom. | A — works, but the graph overhead has no payoff on a single path. | S — a plugin function or planner invocation covers it. |
| **Intent routing (query → retriever)** | S — `RouterQueryEngine` with LLM or embedding selectors. | A — built via `RunnableBranch` / `RunnableRouter` plus a selector. | A — conditional edges; author writes the selector node. | A — planner selects a skill; or an `if` in Python/C#. |
| **Sub-question decomposition** | S — `SubQuestionQueryEngine` is purpose-built. | A — requires custom chain + aggregator. | S — natural fan-out via multiple edges from a decompose node. | A — Stepwise/Function-calling planner can decompose with prompting. |
| **Cycles / retrieve-and-verify loop** | W — the query-engine abstraction is acyclic by design. | W — a `Runnable` is linear; loops require manual control flow outside it. | S — cycles and conditional back-edges are first-class. | A — a planner can re-plan, but the cycle contract is prompt-driven, not graph-typed. |
| **Explicit, typed state across steps** | W — state lives in the engine's internal response objects. | A — pass dicts through `RunnablePassthrough` + `with_config`. | S — `TypedDict` / Pydantic `State` is the graph contract. | A — plugin inputs/outputs are typed; cross-step state is per-plugin. |
| **Human-in-the-loop interrupts** | W — not a modelled concept. | W — ad-hoc. | S — `interrupt_before` / `interrupt_after` plus checkpointer resumption. | A — a planner step can prompt for input, but there is no built-in checkpoint/resume. |
| **Streaming of intermediate state** | A — `QueryEngine.query` streams tokens; intermediate retrievals are observable via callbacks. | S — `Runnable.astream_events` yields every intermediate event. | S — `graph.astream` yields per-node state deltas. | A — streams tokens; node-level event streaming is less uniform. |
| **Observability hooks** | A — callback manager; integrates with LangSmith / Arize / Weave via adapters. | S — LangSmith is first-party; callbacks on every `Runnable`. | S — LangSmith is first-party; state deltas are inspectable per node. | A — OpenTelemetry integration; Azure Application Insights for Azure deployments. |
| **Retriever ecosystem breadth** | S — large connector set for vector stores, graph stores, SQL, hybrid. | S — the broadest connector surface of the four. | S — inherits LangChain's retrievers directly. | A — first-party connectors for Azure AI Search, Cosmos, Redis; broader ecosystem via community plugins. |
| **Deployment story** | A — library; deploy however you deploy Python. | A — library; deploy however you deploy Python / JS. | A — library + optional LangGraph Platform (hosted) for stateful runs. | S — first-class in .NET + Azure; Python also supported. |
| **Enterprise lock-in surface** | Low — Apache-2.0, vendor-neutral. | Low — MIT, vendor-neutral. | Low — MIT, vendor-neutral (LangGraph Platform is optional). | Moderate — vendor-neutral code, but the opinionated path is Azure OpenAI + Azure AI Search. |

Notes on reading the matrix:

- No framework scores **W** on more than two axes. The W-cells are
  where the framework is intentionally *not* trying to solve that
  problem — LlamaIndex deliberately does not model long-lived state,
  LangChain deliberately does not model cycles. A W in one of those
  rows is a signal to switch frameworks, not to write the missing
  primitive yourself.
- "Strong" does not mean "the only reasonable pick." Multiple
  frameworks score S on streaming; the differentiator is the
  interaction with the surrounding stack, not that axis in isolation.
- Scores are dated to the current framework releases (see §6).
  Frameworks move. The `notes/kill_switch.md` pattern below is how
  you avoid re-running the whole matrix every quarter.

### 2.3 `matrix/scenarios.md` — the four scenarios

#### Scenario A — Single corpus, two retrieval strategies, one answer

A support KB with two indexes: BM25 over long-form articles and
dense over short FAQ snippets. The product requirement is: route the
query to one or both, synthesise one answer, cite sources.

- **Pick: LlamaIndex.**
- **Why:** this is the canonical `RouterQueryEngine` + `ResponseSynthesizer`
  shape. One index per sub-corpus, one selector, one synth; no state
  beyond the call. LangChain is adequate but needs the selector
  hand-rolled; LangGraph is overkill; Semantic Kernel is overkill unless
  the rest of the stack is .NET/Azure.
- **Kill switch:** if a second retrieval round is needed when the
  first result is weak, switch to LangGraph (see Scenario C).

#### Scenario B — Multi-source ingestion pipeline feeding one retriever

Documents land from S3, Notion, Slack, Google Drive. Each has a
different loader + splitter; all converge on one vector index. The
runtime path (query → retrieve → answer) is linear.

- **Pick: LangChain.**
- **Why:** the breadth of document-loader and splitter connectors
  is the deciding factor, and the runtime is a linear
  `Runnable` pipe. LlamaIndex would also work and its ingestion
  pipeline is competitive; the call is close enough that an
  existing LangChain shop should stay on LangChain and vice versa.
  LangGraph and Semantic Kernel are the wrong shape — one is a graph
  runtime, the other is plugin/planner-first.
- **Kill switch:** if the ingestion pipeline grows conditional
  branches with retry/dead-letter semantics, lift it into LangGraph
  so the state is explicit.

#### Scenario C — Retrieve-and-verify loop with a critic model

Retrieve top-k, run a critic model to check whether the retrieved
context plausibly answers the question. If not, rewrite the query,
retrieve again, repeat up to N times. Terminate on sufficient
context or on budget exhaustion. Record the trace for audit.

- **Pick: LangGraph.**
- **Why:** this is the textbook case for a stateful graph with a
  cycle. The termination predicate, the budget counter, and the
  verification score live in `State`. The retrieve node, the
  verify node, and the rewrite node are first-class graph nodes;
  the back-edge is explicit. LlamaIndex and LangChain do not model
  cycles cleanly. Semantic Kernel's planners can approximate this
  with a Stepwise plan but lack the typed, checkpointed state
  contract. (exercise-03 implements this scenario end-to-end.)
- **Kill switch:** if the loop collapses to a one-shot retrieve
  with no re-query path, step back to LlamaIndex or LangChain —
  the graph overhead is dead weight.

#### Scenario D — .NET enterprise app with Azure AI Search and Azure OpenAI

A C# application already deployed on Azure App Service, talking to
Azure OpenAI and indexing content in Azure AI Search. Compliance
requires Entra ID auth, Azure Monitor telemetry, and
customer-managed keys.

- **Pick: Semantic Kernel.**
- **Why:** the first-party .NET story, the Entra ID / Managed
  Identity integration path, and the Azure AI Search connector
  all remove bespoke glue. The other three frameworks have no
  first-class .NET story at this writing (LangChain.NET and
  community LlamaIndex ports exist but are thinner). The Python
  frameworks could be bridged via a sidecar, but that adds a
  deployment surface the team has to own.
- **Kill switch:** if the primary runtime moves off Azure (another
  cloud, on-prem) or off .NET (migration to Python backends), the
  Semantic Kernel advantage evaporates and the pick flips to
  LangGraph (for stateful flows) or LlamaIndex (for retrieve-then-
  answer flows).

### 2.4 `matrix/decision_tree.md` — route scenario → framework

```text
Q1. Does the orchestration need a cycle (retry / verify / re-query loop)
    or long-lived typed state across steps?
    └── Yes → LangGraph.
    └── No → Q2.

Q2. Is the runtime stack primarily .NET or Azure-centric, with
    compliance pressure to use first-party Microsoft components?
    └── Yes → Semantic Kernel.
    └── No → Q3.

Q3. Is the primary engineering concern on the ingest side
    (many sources, many loaders, many splitters)?
    └── Yes → LangChain (breadth of loaders).
    └── No → Q4.

Q4. Is the runtime path a retrieve-then-answer shape with intent
    routing or sub-question decomposition?
    └── Yes → LlamaIndex (RouterQueryEngine / SubQuestionQueryEngine).
    └── No → Q5.

Q5. Does the team already have one of these frameworks deployed and
    the new feature fits inside that framework's sweet spot?
    └── Yes → stay. Framework switching has a cost the matrix
              does not price in.
    └── No → default to LangChain for new greenfield; its component
             surface is the broadest and its exit cost is the lowest.
```

The tree collapses the matrix to the single question each framework
answers best. It is deliberately short — a decision tree with more
than ~7 nodes stops being read and starts being ignored.

### 2.5 `notes/kill_switch.md` — when to revisit

For each per-scenario pick, write down the *signal* that flips it:

- **A → C (LlamaIndex → LangGraph)**: product adds a "retry if
  confidence low" feature or a user-visible "verify sources" step.
- **B → C (LangChain → LangGraph)**: ingest grows retry/DLQ logic
  with cross-step state; or the runtime path grows a loop.
- **C → A (LangGraph → LlamaIndex)**: the verify loop is removed
  or lowered to a single-pass critic; graph overhead becomes dead
  weight.
- **D → C (Semantic Kernel → LangGraph)**: the .NET-only or
  Azure-only assumption is lifted and a Python runtime enters the
  picture.
- **Any → stay-put**: the new feature fits the current framework's
  sweet spot and switching cost exceeds the delta in capability.

The kill switch is deliberately written as a *trigger*, not a
schedule. Frameworks move quickly and a quarterly matrix rebuild is
the wrong cadence — the rebuild is what you do when a kill-switch
signal fires.

## 3. Validation steps

A correct submission can be validated without running any code. The
reviewer should be able to answer "yes" to each of the following:

1. The matrix has four columns (one per framework), ≥ eight rows
   (one per axis), and every cell has a one-sentence justification
   that cites a documented primitive, not a vendor claim.
2. No cell scores **S** on an axis the framework explicitly does
   not try to solve (e.g., LlamaIndex strong on cycles would be
   wrong — check against the framework's own docs).
3. All four scenarios have a pick, a *why* that points at the
   matrix, and an explicit kill-switch condition. The *why*
   survives the question "what would a reviewer say to challenge
   this pick?"
4. The decision tree routes each of the four scenarios to the pick
   the scenario memo makes. If the tree disagrees with the memo,
   one of them is wrong.
5. No scenario is picked on the basis of "framework X is more
   popular" or "the team prefers X." Preference is a tiebreaker,
   not a decider.
6. The references section links to the framework's *own*
   documentation, dated or versioned. Blog posts and vendor
   marketing are not a primary source.

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Matrix completeness | ≥ 8 axes, 4 frameworks, every cell scored with a one-line justification tied to a documented primitive. |
| Scenario coverage | All four scenarios named in the exercise are addressed with a pick, a *why*, and a kill switch. |
| Consistency | The decision tree routes each scenario to the framework the memo picks. |
| Lock-in disclosed | Each framework's lock-in surface (licensing, hosted platforms, cloud affinity) is called out at least once in the matrix. |
| Kill switches present | At least one per-scenario trigger that would flip the pick is written down. |
| No framework-tribalism | A reviewer cannot tell from the memo which framework the author used before writing it. |
| Sources | Primary source per claim is the framework's own documentation (links in §6). No cell is justified from a blog post without the underlying doc link. |
| Threat framing | At least one row or scenario addresses observability, auditability, or supply-chain lock-in — not only "which is fastest." |

Fail-fast smells:

- Every scenario picks the same framework. Either the scenarios do
  not actually differ or the author is reasoning from preference,
  not shape.
- No W cells anywhere. Every framework has a weakness; a matrix
  with no W cells is not honest.
- The decision tree has 15 nodes. A decision tree that long will
  not be used; collapse it.
- The kill switch is a date ("re-evaluate in 6 months"), not a
  signal. Signals are observable; calendars are not.
- The matrix cites a vendor blog as the authority for a
  framework's own capabilities. Use the framework's docs.

## 5. Common mistakes

- **Scoring on feature count, not fit.** A framework can have the
  most features on paper and still be the wrong pick because none
  of those features fit the scenario's shape. The matrix is a
  fit map, not a feature inventory.
- **Picking LangGraph for a single-shot retrieve-then-answer.** The
  graph runtime has a startup cost (state schema, node definitions,
  checkpointer config) that pays back only when there is a cycle,
  an interrupt, or multi-step state. For one-shot flows it adds
  code without adding value.
- **Picking LlamaIndex for a stateful loop.** The `QueryEngine`
  abstraction is acyclic. Forcing a loop on top of it means
  reimplementing LangGraph's state contract inside your app.
- **Picking LangChain for everything because the team knows it.**
  Familiarity is a tiebreaker on close calls. It is not a reason
  to pick the wrong shape; the surrounding code will eventually
  reinvent whatever primitive the chosen framework is missing.
- **Ignoring Semantic Kernel because "Microsoft."** For .NET or
  Azure-first shops, the first-party integration and compliance
  story is the deciding factor. Dismissing SK on vibes leads to
  a Python sidecar the team has to operate.
- **Freezing the pick.** Framework capabilities move quickly. A
  matrix that is not paired with kill-switch signals is a
  decision you will not revisit until the pain is already
  blocking shipping.
- **Not disclosing lock-in.** Each framework has one — licensing,
  hosted services, cloud affinity. Omitting it in the memo
  surprises the next reviewer.
- **Confusing LangChain and LangGraph.** They share a team and some
  telemetry (LangSmith), but they are different runtimes with
  different control-flow models. Picking "LangChain" when you
  needed "LangGraph" (and then bolting loops on top) is the most
  common misstep in this space.

## 6. References

Framework documentation (primary sources for the matrix cells):

- LlamaIndex — [docs.llamaindex.ai](https://docs.llamaindex.ai/).
  Relevant: `RouterQueryEngine`, `SubQuestionQueryEngine`,
  `ResponseSynthesizer`, callback manager.
- LangChain — [python.langchain.com/docs](https://python.langchain.com/docs/).
  Relevant: `Runnable`, `RunnableBranch`, `astream_events`, document
  loaders.
- LangGraph — [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/).
  Relevant: `StateGraph`, conditional edges, checkpointers,
  `interrupt_before` / `interrupt_after`.
- Semantic Kernel —
  [learn.microsoft.com/semantic-kernel](https://learn.microsoft.com/en-us/semantic-kernel/).
  Relevant: plugins/skills, Function-calling and Stepwise planners,
  Azure AI Search and Azure OpenAI connectors.

Risk and governance framings (used to populate the lock-in,
observability, and auditability rows):

- OWASP Machine Learning Security Top 10 —
  [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
  Relevant items: ML06 AI Supply Chain Attacks (framework
  dependencies are part of the supply chain surface).
- MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
  Relevant: *ML Supply Chain Compromise* and *LLM Plugin
  Compromise* techniques motivate the lock-in / observability axes.
- NIST AI Risk Management Framework —
  [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
  The *Govern* and *Measure* functions require documented
  component selection and traceability — the matrix is the
  artifact that supports both.

Downstream consumers:

- exercise-02 (LlamaIndex router + sub-question) builds on the
  Scenario A pick.
- exercise-03 (LangGraph verify loop) builds on the Scenario C
  pick.
- exercises 04 and 05 (MCP and native tool exposure) are
  framework-agnostic — the retriever behind the tool can be
  wired in any of the four frameworks.
