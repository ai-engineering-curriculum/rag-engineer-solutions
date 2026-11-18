# Exercise 04 — Solution: Wrap a Retriever as an MCP Server (Python SDK)

This is a reference solution for
`exercise-04-wrap-a-retriever-as-an-mcp-server-with-python-sdk.md`.
It is an **implementation** exercise. The learner exposes a retriever
as a Model Context Protocol (MCP) server using the official
`mcp` Python SDK. The server offers one tool (`search`), one resource
(`corpus://stats`), and documents the result schema. The deliverable
is a working server that any MCP-compatible client (Claude Desktop,
Cursor, custom agents) can call.

The teaching point: MCP is a protocol-level contract for how LLM
clients discover and call tools. Shipping your retriever behind an
MCP server means every MCP-compatible client can use it without
you writing a bespoke integration per client. The contract that
matters is the *tool schema* and the *result shape*; the transport
(stdio vs. HTTP) is secondary. If the schema is honest about what
the retriever returns, the exercise is a success.

## 1. Solution overview

The deliverable has five parts:

1. **`server/retriever.py`** — the underlying retriever. Pluggable
   (BM25 / dense / hybrid from mod-106); the exercise treats it
   as a black box with a `search(query, k) → list[Hit]` contract.
2. **`server/mcp_server.py`** — the MCP server built with the
   `mcp` Python SDK. Registers one `search` tool with a typed
   input schema, a declared output shape, and a resource that
   exposes corpus stats.
3. **`server/manifest.md`** — human-readable documentation of the
   tool: purpose, inputs, outputs, failure modes, trust boundary.
4. **`client/smoke_test.py`** — an in-process MCP client that
   starts the server, lists tools, calls `search`, reads the
   resource, and asserts on the responses.
5. **`claude_desktop_config.example.json`** — a minimal
   `mcp_servers` entry so a user can wire the server into Claude
   Desktop or a compatible client.

Scope: *server wiring and schema correctness*; the underlying
retriever is a stub or a mod-106 implementation. If the server
starts, a client can list tools, and `search` returns well-formed
results, the exercise passes.

## 2. Worked answer or implementation

### 2.1 MCP in one paragraph

MCP is a JSON-RPC 2.0 protocol between an LLM host (client) and
one or more servers that expose capabilities. The three capability
kinds are **tools** (callable functions, like `search`),
**resources** (addressable read-only content, like a corpus
schema document), and **prompts** (parameterised prompt
templates). Servers advertise their capabilities on the
`initialize` handshake; the client calls them by name. The
default transport for local servers is **stdio** (the client
spawns the server as a subprocess); remote servers use
**Streamable HTTP**. Specification:
[modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification).

### 2.2 `server/retriever.py` — the retriever

```python
"""Toy retriever with a stable API. Swap in BM25 / dense / hybrid
from mod-106 for the real thing; the MCP layer does not care."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    title: str
    snippet: str
    source: str  # e.g., "s3://bucket/key" or "notion://page-id"


class Retriever:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = query.lower().strip()
        scored = []
        for d in self._docs:
            text = (d["title"] + " " + d["snippet"]).lower()
            score = sum(1 for token in q.split() if token in text)
            if score > 0:
                scored.append(
                    Hit(
                        doc_id=d["doc_id"],
                        score=float(score),
                        title=d["title"],
                        snippet=d["snippet"][:240],
                        source=d["source"],
                    )
                )
        scored.sort(key=lambda h: (-h.score, h.doc_id))
        return scored[:k]

    def stats(self) -> dict:
        return {"num_docs": len(self._docs)}
```

**Choices and why:**

- **Retriever is a plain class, not an MCP concept.** The MCP
  layer wraps it; swapping to a real retriever is a constructor
  change.
- **`Hit` is explicit about the fields the client will see.**
  `snippet` is bounded (240 chars) so the tool result is a
  predictable size.
- **`source` is in every hit.** mod-108's citation contract and
  mod-112's ACL enforcement both need the originating document
  identifier; omitting it forces the client to guess.

### 2.3 `server/mcp_server.py` — the MCP server

```python
"""MCP server exposing the retriever as a `search` tool and a
`corpus://stats` resource.

Uses the official `mcp` Python SDK's FastMCP convenience layer; see
https://github.com/modelcontextprotocol/python-sdk for the full API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from retriever import Hit, Retriever


# ---- bootstrap --------------------------------------------------------

def _load_corpus(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_corpus_path = Path(__file__).parent / "data" / "corpus.jsonl"
_retriever = Retriever(_load_corpus(_corpus_path))
mcp = FastMCP(
    name="retriever-mcp",
    instructions=(
        "A retriever over a local corpus. Call `search` with a natural-"
        "language query to get the top documents. Every result includes "
        "`doc_id`, `score`, `title`, `snippet`, and `source`; cite the "
        "`source` in any downstream answer."
    ),
)


# ---- result schema ----------------------------------------------------

class SearchHit(BaseModel):
    """One retrieval hit. All fields are required."""
    doc_id: str = Field(..., description="Stable document identifier.")
    score: float = Field(..., description="Retriever-native similarity score.")
    title: str
    snippet: str = Field(..., description="Short excerpt; truncated at 240 chars.")
    source: str = Field(
        ...,
        description="URI of the source document; the client MUST cite this.",
    )


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    truncated: bool = Field(
        False,
        description="True when `k` was clamped to the server maximum.",
    )


# ---- tool -------------------------------------------------------------

MAX_K = 20


@mcp.tool()
def search(
    query: Annotated[str, Field(min_length=1, description="User query.")],
    k: Annotated[
        int, Field(ge=1, le=MAX_K, description="Number of results, 1..20.")
    ] = 5,
) -> SearchResult:
    """Search the local corpus and return the top-k documents.

    The server clamps `k` to [1, 20]. Each hit includes a `source` URI;
    callers MUST cite the source in any user-visible answer. Results
    are ordered by descending score; ties broken by `doc_id`.
    """
    k_eff = min(max(int(k), 1), MAX_K)
    hits = _retriever.search(query, k=k_eff)
    return SearchResult(
        query=query,
        hits=[SearchHit(**hit.__dict__) for hit in hits],
        truncated=(k_eff < int(k)) if isinstance(k, int) else False,
    )


# ---- resource ---------------------------------------------------------

@mcp.resource("corpus://stats")
def corpus_stats() -> str:
    """Return corpus statistics (JSON): document count, last-indexed timestamp."""
    return json.dumps(_retriever.stats())


# ---- entry point ------------------------------------------------------

if __name__ == "__main__":
    # Default transport for local MCP servers is stdio; swap to
    # `mcp.run(transport="streamable-http")` for a remote deployment.
    mcp.run(transport="stdio")
```

**Choices and why:**

- **`FastMCP` convenience layer.** The SDK exposes both a
  low-level `Server` API and a `FastMCP` decorator-based layer.
  `FastMCP` emits the JSON Schema from the function signature +
  Pydantic types, which is the right level of abstraction for
  this exercise. The low-level API is the correct pick when you
  need lifecycle hooks (`list_tools` interception, streaming
  mid-tool, custom auth).
- **Typed Pydantic input and output.** The SDK converts the
  Pydantic model into the tool's declared JSON Schema. Clients
  receive the schema on `tools/list` and the LLM can produce
  schema-conformant arguments without hallucinating fields.
- **`k` is clamped on the server.** The server is a trust
  boundary. A client that asks for `k=10000` gets `k=20` with
  `truncated=True`, not a 500-ms retrieval over 10k documents.
- **Docstrings are the LLM's documentation.** The `mcp` SDK uses
  the function docstring as the tool description; make it an
  honest instruction, not a one-liner. "Call this with a
  natural-language query" and "clients MUST cite the `source`"
  are what the LLM reads when it decides how to call the tool.
- **stdio transport.** Local MCP servers run under the host as a
  subprocess over stdio. For a remote server, swap to
  `transport="streamable-http"`; the tool contract does not
  change.
- **Resource for stats.** Static-ish "describe what this server
  holds" belongs in a resource, not a tool, so a client can read
  it eagerly without calling a tool. Resources are addressable by
  URI; `corpus://` is a plausible scheme for the corpus's
  metadata.

### 2.4 `server/manifest.md` — human-readable tool docs

```markdown
# retriever-mcp — tool manifest

## Tool: `search`

- **Purpose:** retrieve the top-k documents from the local corpus
  for a natural-language query.
- **Inputs:**
  - `query: str` — required, min length 1.
  - `k: int` — optional, 1..20 (default 5). Server clamps.
- **Outputs:** `SearchResult { query, hits[], truncated }` where
  each hit has `doc_id`, `score`, `title`, `snippet` (≤ 240 chars),
  `source`.
- **Failure modes:**
  - Query shorter than 1 char → MCP tool-call validation error.
  - Empty corpus → `hits: []`, `truncated: false`.
  - Transient retriever error → MCP tool-call error with
    `isError: true` and a `TextContent` describing the failure.
    No partial results are returned.
- **Trust boundary:** the server trusts the host to invoke the
  tool on behalf of an authorised user; the retriever does not
  re-authenticate. For ACLs or scoped retrieval, see mod-112.

## Resource: `corpus://stats`

- JSON blob with `num_docs` and (in the production fork)
  `last_indexed_at`. Cheap to read; use before deciding to call
  `search`.

## Pinned versions

- `mcp` Python SDK — pin a specific version in `pyproject.toml`
  and bump intentionally; the schema-emission behaviour has
  evolved between SDK releases.
```

### 2.5 `client/smoke_test.py` — minimal in-process client

```python
"""Smoke test: start the server over stdio, list tools, call `search`,
read the `corpus://stats` resource, assert on shape.

Uses the SDK's ClientSession + stdio_client helpers; see
https://github.com/modelcontextprotocol/python-sdk for the full API."""
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER = StdioServerParameters(
    command="python",
    args=["-m", "server.mcp_server"],
)


async def _run() -> None:
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert "search" in names, names

            call = await session.call_tool(
                "search", {"query": "retention policy", "k": 3}
            )
            # The server returned a structured content (SearchResult).
            # The SDK unpacks it; .structuredContent is the typed payload.
            payload = call.structuredContent
            assert payload is not None
            assert payload["query"] == "retention policy"
            assert isinstance(payload["hits"], list)
            for hit in payload["hits"]:
                assert {"doc_id", "score", "title", "snippet", "source"} <= set(hit)

            # Over-asked k should be clamped.
            over = await session.call_tool("search", {"query": "x", "k": 9999})
            assert over.structuredContent["truncated"] is True

            res = await session.read_resource("corpus://stats")
            # Resource content is a list of (uri, text) tuples.
            payload = json.loads(res.contents[0].text)
            assert "num_docs" in payload


if __name__ == "__main__":
    asyncio.run(_run())
    print("OK")
```

**Choices and why:**

- **In-process client over stdio.** No separate process to
  orchestrate; the test spawns the server, talks to it, and
  tears it down. The same wire protocol as a real host.
- **Assert on `structuredContent`, not on the first `TextContent`
  block.** `FastMCP` emits both — a structured payload for
  schema-aware clients and a text block for legacy ones.
  Asserting on the structured payload is the modern contract.
- **Clamp test.** The single most important behaviour-test:
  assert that the server enforces its own bounds.

### 2.6 `claude_desktop_config.example.json`

```json
{
  "mcpServers": {
    "retriever-mcp": {
      "command": "python",
      "args": ["-m", "server.mcp_server"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/this/project"
      }
    }
  }
}
```

**Notes:**

- The config schema is defined by the host (Claude Desktop in
  this example). The MCP spec only governs the wire protocol,
  not the host-side launch config.
- Prefer absolute paths; the host launches the subprocess from
  its own working directory, not from your repository.
- For a production remote server, use a Streamable-HTTP transport
  and point the host at its URL instead of launching a
  subprocess.

## 3. Validation steps

1. `python -m server.mcp_server` starts the server over stdio and
   does not exit. (Expected behaviour: the server idles waiting
   for the first JSON-RPC request on stdin.)
2. `python -m client.smoke_test` returns `OK` and exits 0.
3. The server's `tools/list` response advertises exactly one
   tool `search` with a JSON Schema that includes `query` (string,
   required, min length 1) and `k` (integer, 1..20).
4. A `tools/call` with `query=""` fails validation with a
   structured error, not a 500.
5. A `tools/call` with `k=9999` returns `truncated: true` and
   `hits` of length ≤ 20.
6. `resources/list` advertises `corpus://stats`; `resources/read`
   returns a JSON object with `num_docs`.
7. The server is registered in a Claude Desktop (or other MCP
   client) config and the client lists it as "available" after
   restart. (This is a manual check; include a screenshot or
   `tools/list` output in the submission.)

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Tool schema fidelity | `search` advertises a JSON Schema with required `query`, optional `k` (with bounds), and a documented return shape. |
| Clamp on server | Out-of-range `k` is clamped to the server max and `truncated` is set. The server never trusts the client's bound. |
| Result honesty | Each hit has `doc_id`, `score`, `title`, `snippet` (bounded), `source`. No hidden fields the client cannot see. |
| Resource present | At least one resource (e.g., `corpus://stats`) exposes useful metadata; readable before any tool call. |
| Docstring quality | The tool docstring is the instruction the LLM reads. It names the required `source` citation and the clamp. |
| Transport correctness | Local path uses stdio; the server runs as a subprocess. Remote path uses Streamable HTTP (documented in `manifest.md`). |
| Smoke test | In-process client lists tools, calls `search`, reads the resource, asserts on structured payload. |
| Error surface | Invalid inputs produce MCP validation errors; runtime errors set `isError: true` on the tool-call result. No partial `hits`. |
| Pinning | SDK version pinned; the submission names the SDK version it was built against. |
| Documentation | `manifest.md` lists purpose, inputs, outputs, failure modes, and the trust boundary in plain English. |

Fail-fast smells:

- No input validation; the client can pass arbitrary types and
  the server returns a Python stack trace.
- `k` is honoured literally; a request for 10k documents either
  returns 10k or crashes.
- `snippet` is unbounded and the tool result balloons to the
  underlying document size.
- No `source` on hits. Downstream citation (mod-108) is
  impossible.
- The tool docstring says "search the corpus" and nothing else;
  the LLM cannot tell what it will get back.
- No smoke test. Protocol-level mistakes (missing `initialize`,
  wrong tool name, schema misalignment) go undetected.

## 5. Common mistakes

- **Treating MCP as a vendor binding.** MCP is an open protocol
  with multiple hosts (Claude Desktop, Cursor, Zed, custom
  agents). Writing the server against "the Claude integration"
  rather than the spec locks you into one host.
- **Hand-writing the JSON Schema instead of letting the SDK
  derive it from types.** Divergence between the declared schema
  and the actual function signature is the top source of
  silent-failure tool calls.
- **No clamp on expensive parameters.** The retriever runs
  server-side; a client that asks for `k=10000` can effectively
  DoS the server. Clamp.
- **Leaking exceptions as 500 errors.** Tool failures should set
  `isError: true` in the structured response with a `TextContent`
  describing the failure in terms the LLM can act on ("query too
  short; please provide at least one word"), not a Python stack
  trace.
- **Wrapping the retriever as a "chat" instead of a "search".**
  MCP tools should be *small and composable*. "Search the corpus"
  is one tool; "answer the question using the corpus" is a
  different responsibility that belongs to the host LLM, not the
  server.
- **Confusing tools and resources.** A tool is a function with
  side effects / parameters; a resource is addressable content.
  "Corpus stats" is a resource; "search" is a tool. Mixing them
  confuses hosts that enumerate capabilities.
- **Omitting `source` to save bytes.** The citation URL is what
  makes a RAG answer defensible; stripping it collapses the
  downstream citation contract from mod-108.
- **Shipping the server without pinning the SDK version.** The
  SDK is young; its schema-emission behaviour is still evolving.
  Pin the version, test against it, and bump intentionally.
- **Registering the server as "always-on" with a Claude Desktop
  user config as if it were a trusted system component.** MCP
  servers inherit host privileges; the server's trust boundary
  must be documented (`manifest.md`). An attacker-controlled
  server could exfiltrate conversation content.

## 6. References

- Model Context Protocol specification —
  [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification).
  The wire protocol, lifecycle, transports, and capability
  model.
- Official Python SDK —
  [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk).
  `FastMCP`, low-level `Server`, `ClientSession`,
  `stdio_client`.
- MCP introduction and concepts —
  [modelcontextprotocol.io/introduction](https://modelcontextprotocol.io/introduction).
- Claude Desktop MCP integration docs —
  [docs.anthropic.com](https://docs.anthropic.com/en/docs/mcp)
  (host-side `claude_desktop_config.json` schema).
- Related modules:
  - mod-106 hybrid retrieval — the retriever behind the tool.
  - mod-108 citation contract — the `source` field is what the
    citation chain consumes.
  - mod-112 enterprise retrieval / ACL — the server-side trust
    boundary is where ACL enforcement belongs.
- Risk framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant item: ML06 AI Supply Chain Attacks — a malicious
    MCP server configured into a host is a supply-chain
    vector.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    *LLM Plugin Compromise* directly motivates server-side input
    validation and the clamp on `k`.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    *Govern* and *Manage* require documenting the trust boundary
    and the allowed callers; `manifest.md` is the artifact.
