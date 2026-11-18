# Exercise 05 — Solution: Wrap the Same Retriever as OpenAI and Anthropic Tool Endpoints

This is a reference solution for
`exercise-05-wrap-a-retriever-as-openai-and-anthropic-tool-endpoints.md`.
It is an **implementation** exercise. The learner takes the same
retriever from exercise 04 and exposes it two more ways: as a
**function tool** in the OpenAI Responses / Chat Completions API,
and as a **tool** in the Anthropic Messages API. Both wrappers share
one retriever and one result schema; the difference is the wire
protocol by which each model requests the tool.

The teaching point: MCP and native provider tool-calling solve the
same problem — *let a model invoke a function on your server* — at
different layers. MCP decouples server from host; native tool
calling binds the tool to a single provider's API. Shipping both
surfaces side-by-side lets the same retriever ride behind any client
and also behind a provider-native agent. The hard part is not the
wiring; it is converging on a *single result schema* so downstream
code (citation, logging, eval) does not fork.

## 1. Solution overview

The deliverable has six parts:

1. **`shared/retriever.py`** — the same `Retriever` and `Hit` from
   exercise 04, re-used verbatim.
2. **`shared/schema.py`** — the canonical result schema (Pydantic
   `SearchResult` with `hits[]`). Both provider wrappers return
   this shape.
3. **`openai_adapter/tool.py`** — the OpenAI tool definition
   (JSON Schema for arguments), the dispatcher, and an example
   agentic loop that ends on a `stop` response.
4. **`anthropic_adapter/tool.py`** — the Anthropic tool definition
   (`input_schema` + `name` + `description`), the dispatcher, and
   an example agentic loop that ends on `stop_reason == "end_turn"`.
5. **`shared/compat_table.md`** — a one-page table mapping the
   three contracts (MCP, OpenAI, Anthropic) onto the same
   retriever so a reviewer can see the exact field-level
   differences.
6. **`tests/test_adapters.py`** — tests that assert both adapters
   dispatch to the same retriever on the same inputs and return
   the same `SearchResult` payload.

Scope: *tool wiring and schema normalisation*. The retriever is the
exercise-04 stub; the agentic loops are minimal illustrations, not
production agents. Provider SDKs change; the tests pin the SDK
versions used.

## 2. Worked answer or implementation

### 2.1 Provider tool calling in one paragraph each

- **OpenAI** (Chat Completions and Responses APIs). The caller
  passes `tools=[{"type": "function", "function": {name,
  description, parameters}}]` on the request. The model may
  respond with `tool_calls`, each naming a function and a JSON
  `arguments` string. The caller executes the function and sends
  a follow-up message of role `tool` with the result. The loop
  continues until the model returns a normal message with no
  `tool_calls`. Reference:
  [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling).
- **Anthropic** (Messages API). The caller passes
  `tools=[{name, description, input_schema}]`. The model may
  emit content blocks of type `tool_use` with `name` and `input`.
  The caller sends a follow-up user message containing a
  `tool_result` block with the matching `tool_use_id`. The loop
  continues until `stop_reason == "end_turn"`. Reference:
  [docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).

### 2.2 `shared/schema.py` — one result shape

```python
"""Canonical result schema for the retriever. All three surfaces
(MCP, OpenAI, Anthropic) serialise this same shape."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    doc_id: str
    score: float
    title: str
    snippet: str = Field(..., max_length=240)
    source: str = Field(..., description="URI to cite in any downstream answer.")


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    truncated: bool = False


MAX_K = 20


def do_search(retriever, query: str, k: int) -> SearchResult:
    """Clamp-and-dispatch helper shared by both adapters."""
    k_eff = min(max(int(k), 1), MAX_K)
    hits = retriever.search(query, k=k_eff)
    return SearchResult(
        query=query,
        hits=[SearchHit(**h.__dict__) for h in hits],
        truncated=(k_eff < int(k)),
    )
```

**Choices and why:**

- **One schema, two wire formats.** The result is `SearchResult`;
  the adapters just serialise it. Downstream citation, logging,
  and eval code sees the same shape regardless of which provider
  the host used.
- **Clamp in the shared helper.** Clamping in each adapter would
  drift; clamp once.
- **`max_length=240` on snippet.** The same bound as exercise 04;
  provider contexts cost real money, so bounded result sizes are
  a feature, not a limitation.

### 2.3 `openai_adapter/tool.py` — OpenAI function tool

```python
"""OpenAI function-tool wrapper for the retriever.

Tool schema is JSON Schema; the model emits a function call with a
JSON-encoded `arguments` string that the caller executes. The
retriever returns a canonical SearchResult serialised as JSON."""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from shared.retriever import Retriever
from shared.schema import MAX_K, SearchResult, do_search


OPENAI_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the local corpus and return the top-k documents. "
            "Each result includes doc_id, score, title, a bounded snippet, "
            "and a source URI. Cite the source in any user-visible answer."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_K,
                    "default": 5,
                },
            },
        },
    },
}


def dispatch_tool_call(retriever: Retriever, name: str, arguments: str) -> str:
    """Execute a tool call from the model's `tool_calls` list.

    Returns the JSON string to stuff into the follow-up `tool` message's
    `content`. Both errors and success are structured JSON so the model
    can parse and recover."""
    if name != "search":
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid arguments JSON: {e}"})
    q = args.get("query")
    if not q or not isinstance(q, str):
        return json.dumps({"error": "`query` is required and must be a string"})
    k = args.get("k", 5)
    result: SearchResult = do_search(retriever, q, k)
    return result.model_dump_json()


def run_loop(retriever: Retriever, user_prompt: str, *, model="gpt-5-codex") -> str:
    """Minimal agentic loop demonstrating the tool-call protocol.

    This is the simplest correct shape; a production loop would add
    per-call timeouts, token budgets, and error recovery. Note that the
    `tool` message must carry `tool_call_id` matching the model's
    tool_call to correlate request and response."""
    client = OpenAI()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Use the `search` tool to find relevant documents, then "
                "answer the user's question and cite the `source` field "
                "from each document you used."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    for _ in range(5):  # loop budget
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=[OPENAI_SEARCH_TOOL]
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or ""
        messages.append(msg.model_dump())
        for call in msg.tool_calls:
            result_json = dispatch_tool_call(
                retriever, call.function.name, call.function.arguments
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_json,
                }
            )
    return "(budget exhausted)"
```

**Choices and why:**

- **`additionalProperties: false`.** Strict schema means the model
  cannot inject fields the server does not know about; makes the
  tool-call contract auditable.
- **Return JSON on both success and error.** The model is more
  reliable at recovering from a structured error than a raw
  exception string. `{"error": "..."}` is parseable; a Python
  traceback is not.
- **Loop budget of 5.** The loop must terminate. In production,
  replace with a token-budget check (see mod-110 exercise 03's
  budget-exhaustion pattern).
- **`tool_call_id` correlation.** Each follow-up `tool` message
  must reference the model's `tool_call.id`. Skipping this breaks
  the model's internal tracking of which result belongs to which
  call.

### 2.4 `anthropic_adapter/tool.py` — Anthropic tool

```python
"""Anthropic tool-use wrapper for the retriever.

Tool schema is `input_schema` (JSON Schema). The model emits content
blocks of type `tool_use` with `input`. The caller returns a user
message containing a `tool_result` block with the matching
`tool_use_id`."""
from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from shared.retriever import Retriever
from shared.schema import MAX_K, SearchResult, do_search


ANTHROPIC_SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": (
        "Search the local corpus and return the top-k documents. Each "
        "result includes doc_id, score, title, a bounded snippet, and a "
        "source URI. Cite the source in any user-visible answer."
    ),
    "input_schema": {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_K,
                "default": 5,
            },
        },
    },
}


def dispatch_tool_use(retriever: Retriever, block: dict[str, Any]) -> dict[str, Any]:
    """Execute one `tool_use` block and return the matching `tool_result`
    block to send back as the next user message's content.

    On error, the tool_result sets `is_error: true` and carries a
    short diagnostic string. The model handles `is_error` natively."""
    tool_use_id = block["id"]
    name = block["name"]
    inp = block.get("input") or {}

    if name != "search":
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": True,
            "content": f"unknown tool: {name}",
        }
    q = inp.get("query")
    if not q or not isinstance(q, str):
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": True,
            "content": "`query` is required and must be a string",
        }
    k = inp.get("k", 5)
    result: SearchResult = do_search(retriever, q, k)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result.model_dump_json(),
    }


def run_loop(
    retriever: Retriever, user_prompt: str, *, model="claude-opus-4-8"
) -> str:
    """Minimal agentic loop demonstrating the Anthropic tool-use protocol.

    Loop terminates on stop_reason == "end_turn" or when the model emits
    no tool_use blocks. A production loop would add explicit timeouts,
    prompt caching on the tools+system blocks (see claude-api skill), and
    a token-budget check."""
    client = Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]
    system_text = (
        "Use the `search` tool to find relevant documents, then answer "
        "the user's question and cite the `source` field from each "
        "document you used."
    )
    for _ in range(5):
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_text,
            tools=[ANTHROPIC_SEARCH_TOOL],
            messages=messages,
        )
        # Append the assistant turn verbatim (required by the API).
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in resp.content
            if b.type == "tool_use"
        ]
        if resp.stop_reason == "end_turn" or not tool_uses:
            # Extract final text blocks.
            return "".join(b.text for b in resp.content if b.type == "text")
        results = [dispatch_tool_use(retriever, tu) for tu in tool_uses]
        messages.append({"role": "user", "content": results})
    return "(budget exhausted)"
```

**Choices and why:**

- **Append the assistant's full `content` verbatim.** Anthropic's
  Messages API requires the next `user` turn to follow the
  assistant turn exactly; stripping `tool_use` blocks or
  reformatting the list of blocks breaks the correlation.
- **`is_error: true` instead of a Python exception.** The
  Messages API has first-class support for tool errors; the
  model handles `is_error` more reliably than a parsed error
  string.
- **Correlate on `tool_use_id`.** Every `tool_result` must
  include the `tool_use_id` from the matching `tool_use` block.
- **Stop on `end_turn`.** The model can emit a final text block
  after a tool call; terminate on `stop_reason == "end_turn"`,
  not on the presence or absence of tool blocks alone.
- **Prompt caching lives in the mod-110 extension**, not here.
  The `claude-api` skill in this repo covers the cache
  placement; add `cache_control` on the `tools` and `system`
  blocks when the loop gets hot.

### 2.5 `shared/compat_table.md` — three contracts side by side

| Concept | MCP (exercise 04) | OpenAI | Anthropic |
|---|---|---|---|
| Tool declaration | `@mcp.tool()` decorator; schema derived from types | `{"type":"function","function":{"name","description","parameters"}}` | `{"name","description","input_schema"}` |
| Input schema | JSON Schema (emitted by SDK) | JSON Schema in `function.parameters` | JSON Schema in `input_schema` |
| Call shape | JSON-RPC `tools/call` with `arguments` object | `tool_calls[*]` on the assistant message; `arguments` is a JSON **string** | `tool_use` content block with `input` as a JSON **object** |
| Result shape | `CallToolResult` with `structuredContent` and `content[]` | `tool` message with JSON string content | `tool_result` block with `content` and optional `is_error` |
| Error contract | `isError: true` on the result + `TextContent` | caller may either throw (surface as a 500) or embed `{"error":"..."}` in the result | first-class `is_error: true` on the `tool_result` |
| Correlation id | per-call id round-tripped by the SDK | `tool_call.id` → `tool_call_id` on the `tool` message | `tool_use.id` → `tool_use_id` on the `tool_result` |
| Termination signal | no explicit "end"; host decides | assistant message with empty `tool_calls` | `stop_reason == "end_turn"` |
| Discovery | `tools/list` on the server | tools declared per request | tools declared per request |
| Host coupling | host-agnostic (any MCP client) | OpenAI API only | Anthropic API only |
| Trust boundary | server-side (your retriever) | server-side (your dispatcher) | server-side (your dispatcher) |

Pick guidance:

- Use **MCP** when the retriever will be consumed by multiple
  hosts (Claude Desktop, Cursor, local agents, custom clients).
- Use **OpenAI** or **Anthropic** native tooling when the
  retriever is embedded inside one provider's agent runtime and
  the extra MCP hop is overhead.
- Many projects ship all three: MCP as the "any client can use
  this" surface; native tools for performance-critical in-agent
  calls on a known provider. They share `shared/retriever.py`
  and `shared/schema.py`; adapters stay thin.

### 2.6 `tests/test_adapters.py` — parity tests

```python
"""Assert both adapters dispatch to the same retriever and return the
same SearchResult shape. Tests do not call the live LLM APIs; they
test the dispatcher functions directly."""
from __future__ import annotations

import json

from shared.retriever import Retriever
from shared.schema import SearchResult
from openai_adapter.tool import dispatch_tool_call
from anthropic_adapter.tool import dispatch_tool_use


def _retriever():
    docs = [
        {
            "doc_id": "d1",
            "title": "Retention policy",
            "snippet": "We retain data for 90 days.",
            "source": "s3://docs/retention.md",
        },
        {
            "doc_id": "d2",
            "title": "Deletion runbook",
            "snippet": "To purge data past retention, run the sweep job.",
            "source": "s3://runbooks/sweep.md",
        },
    ]
    return Retriever(docs)


def test_openai_dispatch_returns_search_result_json():
    r = _retriever()
    raw = dispatch_tool_call(r, "search", json.dumps({"query": "retention", "k": 2}))
    sr = SearchResult.model_validate_json(raw)
    assert sr.query == "retention"
    assert len(sr.hits) >= 1
    assert sr.hits[0].source.startswith("s3://")


def test_anthropic_dispatch_returns_search_result_json():
    r = _retriever()
    block = {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "search",
        "input": {"query": "retention", "k": 2},
    }
    out = dispatch_tool_use(r, block)
    assert out["type"] == "tool_result"
    assert out["tool_use_id"] == "toolu_123"
    sr = SearchResult.model_validate_json(out["content"])
    assert sr.query == "retention"
    assert sr.hits[0].source.startswith("s3://")


def test_both_adapters_agree_on_same_query():
    r = _retriever()
    openai_raw = dispatch_tool_call(r, "search", json.dumps({"query": "deletion"}))
    anth_raw = dispatch_tool_use(r, {
        "id": "toolu_x", "name": "search", "input": {"query": "deletion"},
    })["content"]
    assert SearchResult.model_validate_json(openai_raw).hits == \
        SearchResult.model_validate_json(anth_raw).hits


def test_openai_rejects_missing_query():
    r = _retriever()
    raw = dispatch_tool_call(r, "search", json.dumps({}))
    assert "error" in json.loads(raw)


def test_anthropic_rejects_missing_query():
    r = _retriever()
    out = dispatch_tool_use(r, {"id": "x", "name": "search", "input": {}})
    assert out.get("is_error") is True


def test_unknown_tool_is_rejected():
    r = _retriever()
    assert "error" in json.loads(
        dispatch_tool_call(r, "delete_everything", "{}")
    )
    out = dispatch_tool_use(r, {"id": "y", "name": "delete_everything", "input": {}})
    assert out.get("is_error") is True
```

## 3. Validation steps

1. `pytest -q tests/test_adapters.py` passes.
2. Both adapters import `shared.retriever.Retriever` and
   `shared.schema.SearchResult`; no retriever or schema logic is
   duplicated in the adapters.
3. Running the OpenAI loop against a configured key produces a
   final assistant message that cites at least one `source` URI
   from the fixture corpus.
4. Running the Anthropic loop against a configured key produces
   a final assistant message that cites at least one `source` URI
   from the fixture corpus.
5. Passing `k=9999` through either adapter returns a
   `SearchResult` with `truncated=True` and `len(hits) <= 20`.
6. Passing a missing `query` through either adapter produces a
   structured error on the respective wire format (`{"error":...}`
   or `is_error: true`), not a Python traceback propagating up.
7. The compatibility table in `shared/compat_table.md` matches
   observed behaviour on each wire format (verified by running
   the loops once with request/response logging and diffing
   against the table).

## 4. Rubric or review checklist

| Axis | Pass criterion |
|---|---|
| Shared retriever | One `Retriever` and one `SearchResult` schema; the adapters do not reimplement either. |
| Correct tool shape | OpenAI tool uses `{"type":"function","function":{...}}`; Anthropic tool uses `{name, description, input_schema}`. |
| Correct call shape | OpenAI `arguments` is parsed as a JSON **string**; Anthropic `input` is used as a JSON **object**. |
| Correct result shape | OpenAI returns a `tool` role message with `tool_call_id`; Anthropic returns a user message containing a `tool_result` block with `tool_use_id`. |
| Error contract | Errors are structured on the wire (`{"error":...}` for OpenAI, `is_error: true` for Anthropic); no raw tracebacks. |
| Loop termination | Both loops terminate: OpenAI on empty `tool_calls`, Anthropic on `stop_reason == "end_turn"`; both honour a budget. |
| Correlation ids | Per-call ids are round-tripped on both surfaces. |
| Clamp | `k` is clamped server-side in `do_search`; neither adapter trusts the model. |
| Compat table | Honest field-level mapping; a reviewer can predict a request shape from the table alone. |
| Tests | Parity tests assert both adapters produce the same `SearchResult` shape for the same query. |

Fail-fast smells:

- `arguments` is passed to Python as-is without `json.loads`.
  OpenAI's `arguments` is a JSON **string** by API contract;
  parsing is mandatory.
- Anthropic `tool_result` omits `tool_use_id`. The model cannot
  match the result to its call; the next turn produces nonsense.
- Errors raised as Python exceptions and bubbled back as 500s.
  Models recover from structured errors; they cannot recover
  from a dropped turn.
- Adapter-local `do_search` reimplementation. The clamp or the
  schema drifts; downstream citation code sees two shapes.
- No loop budget. The model loops indefinitely on a misconfigured
  tool and burns tokens.

## 5. Common mistakes

- **Treating `arguments` as a dict in OpenAI.** The Chat
  Completions API sends `arguments` as a JSON-encoded **string**;
  parsing is the caller's responsibility. Treating it as a dict
  produces `AttributeError` at runtime.
- **Treating `input` as a string in Anthropic.** The Messages
  API sends `input` as a JSON **object** already parsed; calling
  `json.loads` on it fails.
- **Dropping the assistant turn before appending the tool result
  (Anthropic).** The Messages API requires the full assistant
  turn to be present in the conversation before the tool
  response. Reconstructing the assistant turn from just text
  strips the `tool_use` blocks and the API rejects the next call.
- **Mismatched correlation ids.** OpenAI's `tool_call_id` and
  Anthropic's `tool_use_id` are mandatory. Reusing an id across
  calls, or omitting it on the response, breaks the model's
  internal tracking.
- **Divergent schemas between surfaces.** If `search` on OpenAI
  returns `{"results":[...]}` and on Anthropic returns
  `{"hits":[...]}`, downstream code forks. Pick one schema
  (`SearchResult`) and serialise it on both wires.
- **No clamp on `k`.** The model is cheap to the agent author
  but `k=10000` is expensive on your retriever; clamp.
- **Passing the *tool schema* as the *result schema*.** The tool
  schema describes **inputs** (what the model may send). The
  result shape is whatever JSON you stuff in the response; both
  providers treat it as opaque text and schema-validate nothing.
  Define and enforce the result shape yourself.
- **No loop budget.** Tool loops that depend on the model to
  terminate will loop forever when the system prompt or the tool
  description is miswritten.
- **Ignoring provider-specific optimisations.** Both SDKs support
  prompt caching on the `tools` and `system` blocks (see this
  repo's `claude-api` skill for the Anthropic details). On hot
  loops, cache the tools block — it is identical across every
  turn.
- **Hand-writing the JSON Schema in two places.** Keep it in one
  module (e.g., `shared/schema.py`), emit both provider-shaped
  wrappers from it, and the drift problem goes away.

## 6. References

- OpenAI function calling guide —
  [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling).
  Tool definition shape, `tool_calls`, `tool_call_id`,
  `tool` role message, parallel tool calls.
- OpenAI Python SDK —
  [github.com/openai/openai-python](https://github.com/openai/openai-python).
- Anthropic tool use guide —
  [docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).
  `tool_use` / `tool_result` content blocks, `tool_use_id`,
  `is_error`, `stop_reason` values.
- Anthropic Python SDK —
  [github.com/anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python).
- Related exercises:
  - exercise-04 (MCP server) uses the same `Retriever` and the
    same `SearchResult` schema; the MCP wrapper is a third
    surface onto the same code.
  - exercise-03 (LangGraph verify loop) can call either of these
    adapters inside a node; the loop budget pattern from
    exercise-03 is the pattern the `run_loop` budget here would
    scale up to in production.
  - mod-108 citation contract — the `source` field in each hit
    is what the citation verifier downstream consumes.
- Risk framings:
  - OWASP Machine Learning Security Top 10 —
    [owasp.org/www-project-machine-learning-security-top-10/](https://owasp.org/www-project-machine-learning-security-top-10/).
    Relevant: ML06 Supply Chain — provider-native tool contracts
    are a less-portable surface than MCP; be explicit about the
    lock-in.
  - MITRE ATLAS — [atlas.mitre.org/](https://atlas.mitre.org/).
    *LLM Plugin Compromise* motivates the server-side clamp and
    the structured error contract.
  - NIST AI Risk Management Framework —
    [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework).
    *Govern* requires documenting which provider surfaces are
    deployed and who authorises them; the compatibility table is
    the artifact.
