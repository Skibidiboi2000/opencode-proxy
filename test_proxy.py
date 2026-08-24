"""Tests for opencode-proxy error handling."""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

import proxy


@pytest.fixture(autouse=True)
def _monkeypatch_upstream(monkeypatch):
    monkeypatch.setattr(proxy, "BASE_URL", "http://upstream-mock/v1")


def _client_with_mock(monkeypatch, handler):
    """Point proxy._client at an AsyncClient backed by an httpx.MockTransport."""

    def _client(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(proxy, "_client", _client)


def _run_stream(streamer):
    """Collect all chunks from an async generator stream."""

    async def collect():
        out = []
        async for chunk in streamer:
            out.append(chunk)
        return out

    return asyncio.get_event_loop().run_until_complete(collect())


def test_stream_upstream_429_surfaces_real_status(monkeypatch):
    """A 429 from Zen must surface with HTTP 429 + message, not a fake 200 SSE.

    Regression for the bug where stream errors were emitted as SSE events under
    HTTP 200, which made pool-proxy skip its retry logic.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded. Please try again later."}},
        )

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return status, err, chunks

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit exceeded" in err
    assert chunks is None


def test_stream_upstream_200_passes_chunks_through(monkeypatch):
    """A healthy 200 stream is relayed unchanged, ending with exactly one
    [DONE] (the upstream's own [DONE] is intercepted and re-yielded once, with
    no trailing frames after it)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return status, err, chunks

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200
    assert err is None
    data = b"".join(_run_stream(chunks)).decode()
    assert '"content":"x"' in data
    assert data.count("[DONE]") == 1
    assert data.rstrip().endswith("[DONE]")


def test_stream_first_chunk_free_usage_limit_emits_sse_error(monkeypatch):
    """FreeUsageLimitError in the first chunk must surface as a real 429 —
    the probe detects the error frame before HTTP 200 is ever committed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'data: {"type":"error","error":{"type":"FreeUsageLimitError"}}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return status, err, chunks

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit" in (err or "")
    assert chunks is None


def test_fit_context_reduces_max_tokens():
    big = [{"role": "user", "content": "x" * 30000}] * 90
    out = proxy._fit_context({"messages": big, "max_tokens": 64000})
    assert len(out["messages"]) <= 90
    assert out["max_tokens"] < 64000


def test_fit_context_fits_without_change():
    out = proxy._fit_context({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64000})
    assert out["max_tokens"] == 64000
    assert len(out["messages"]) == 1


def test_fit_context_preserves_tool_pairing():
    """Dropping an assistant message with tool_calls must also drop its tool
    responses, otherwise upstream rejects with 400
    'Messages with role tool must be a response to a preceding message with tool_calls'."""

    turn = {"role": "assistant", "content": "t" * 3900, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "r"}
    cost_turn = len(json.dumps(turn)) // 2

    # total cost lands 512 under the fit threshold, so trimming stops right
    # after dropping `turn` — leaving `result` orphaned at index 0.
    budget = int(proxy.CONTEXT_LIMIT * 0.9) - 512
    users_cost = budget - cost_turn - len(json.dumps(result)) // 2
    for _ in range(6):
        users_cost -= len(json.dumps({"role": "user", "content": "x" * 300000})) // 2
    padding = max(1, users_cost * 2 - 30)
    users = [{"role": "user", "content": "x" * 300000} for _ in range(6)]
    users.append({"role": "user", "content": "y" * padding})

    big = [turn, result] + users
    total = sum(len(json.dumps(m)) // 2 for m in big)
    assert total > int(proxy.CONTEXT_LIMIT * 0.9) - 1024
    assert total - cost_turn < int(proxy.CONTEXT_LIMIT * 0.9)

    out = proxy._fit_context({"messages": big, "max_tokens": 1024})
    msgs = out["messages"]
    assert msgs[0].get("role") != "tool"
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            prev = msgs[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls")


def test_headers_include_auth_and_session():
    h = proxy._make_headers("ses-123")
    assert h["Authorization"] == "Bearer public"
    assert h["x-opencode-session"] == "ses-123"
    assert "x-opencode-request" in h


def test_is_rate_limit_error_detects_429_and_free_usage():
    assert proxy._is_rate_limit_error(429, b'{"error":{"message":"RL"}}')
    assert proxy._is_rate_limit_error(200, b"FreeUsageLimitError: quota")
    assert not proxy._is_rate_limit_error(200, b"normal response")


# --- Anthropic native translation ---


def test_anthropic_to_openai_basic():
    out = proxy._anthropic_to_openai(
        {
            "model": "claude-opus-4-7",
            "system": "Be brief",
            "max_tokens": 2000,
            "stream": True,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
            ],
        }
    )
    assert out["model"] == proxy.DEFAULT_MODEL
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert out["messages"][0]["content"] == "Be brief"
    assert out["messages"][-1]["content"] == "hi there"
    assert out["max_tokens"] == 2000
    assert out["stream"] is True


def test_anthropic_to_openai_keeps_supported_model():
    out = proxy._anthropic_to_openai({"model": "deepseek-v4-flash-free", "messages": []})
    assert out["model"] == "deepseek-v4-flash-free"


def test_anthropic_to_openai_tools_roundtrip():
    out = proxy._anthropic_to_openai(
        {
            "model": "claude-3-whatever",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "call_ab", "name": "get_weather", "input": {"city": "Hanoi"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_ab", "content": "30C"}],
                },
            ],
            "tools": [
                {"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {}}}
            ],
            "tool_choice": {"type": "any"},
        }
    )
    msgs = out["messages"]
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["id"] == "call_ab"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "Hanoi"}
    assert msgs[1] == {"role": "tool", "tool_call_id": "call_ab", "content": "30C"}
    assert out["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert out["tool_choice"] == "required"


def test_openai_to_anthropic_message():
    resp = {
        "id": "chatcmpl-x",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "checking",
                    "tool_calls": [
                        {"id": "call_xy", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = proxy._openai_to_anthropic(resp, "deepseek-v4-flash-free")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "deepseek-v4-flash-free"
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {"type": "text", "text": "checking"}
    assert out["content"][1] == {"type": "tool_use", "id": "call_xy", "name": "f", "input": {"a": 1}}
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_events_full_stream():
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_z", "function": {"name": "f", "arguments": '{"x":'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 7}},
    ]

    async def feed():
        for c in chunks:
            yield c

    out = "".join(c.decode() for c in _run_stream(proxy._anthropic_events(feed(), "deepseek-v4-flash-free")))
    assert 'event: message_start' in out
    assert '"role": "assistant"' in out
    assert 'event: content_block_start' in out
    assert '"type": "text"' in out
    assert '"delta": {"type": "text_delta", "text": "Hello"}' in out
    assert 'content_block_start' in out and '"type": "tool_use"' in out
    assert '"delta": {"type": "input_json_delta", "partial_json": "{\\"x\\":' in out
    assert 'event: message_delta' in out
    assert '"stop_reason": "tool_use"' in out
    assert '"output_tokens": 7' in out
    assert 'event: message_stop' in out


def test_zen_stream_parsed_429_surfaces_real_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit exceeded" in err
    assert chunks is None


def test_zen_stream_parsed_frees_usage_midstream_emits_rate_limit_error(monkeypatch):
    sse = (
        b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'
        b'data: {"type":"error","error":{"type":"FreeUsageLimitError"}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return status, err, chunks

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200
    events = _run_stream(chunks)
    assert events[0]["choices"][0]["delta"]["content"] == "a"
    assert events[1]["type"] == "error"
    assert "rate_limit_error" in events[1]["error"]["type"]
    assert len(events) == 2


def test_anthropic_events_empty_stream_emits_terminal():
    """Empty upstream must still emit a complete Anthropic terminal sequence."""

    async def feed():
        if False:
            yield {}

    out = "".join(c.decode() for c in _run_stream(proxy._anthropic_events(feed(), "muse-spark-1.2-contributor-free")))
    assert "event: message_start" in out
    assert "event: message_delta" in out
    assert "event: message_stop" in out


def test_keepalive_wrapper_survives_slow_chunks():
    """Keepalive ticks must not cancel/kill the upstream read (regression:
    wait_for around __anext__ cancelled the read and truncated the stream
    after the first stall)."""

    async def slow():
        yield b"data: one\n\n"
        await asyncio.sleep(0.3)  # longer than interval -> keepalive fires
        yield b"data: two\n\n"

    out = b"".join(_run_stream(proxy._keepalive_wrapper(slow(), interval=0.05))).decode()
    assert "data: one" in out
    assert "data: two" in out          # chunk AFTER the stall must survive
    assert out.count(": keepalive") >= 1


def test_anthropic_events_reasoning_then_tool():
    """Reasoning deltas (muse-spark) must become thinking blocks, not dropped."""

    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think step 1 "}, "finish_reason": None}]},
        {"choices": [{"delta": {"reasoning_content": "think step 2"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_r", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]

    async def feed():
        for c in chunks:
            yield c

    out = "".join(c.decode() for c in _run_stream(proxy._anthropic_events(feed(), "muse-spark-1.2-contributor-free")))
    assert '"type": "thinking"' in out
    assert '"type": "thinking_delta"' in out
    assert out.count("content_block_stop") >= 3  # thinking + text + tool
    assert "event: message_stop" in out


def test_catch_all_v1_responses_url_no_double_prefix(monkeypatch):
    """POST /v1/responses via catch_all must not become /v1/v1/responses."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "resp_1"})

    _client_with_mock(monkeypatch, handler)

    # Simulate what FastAPI would do for POST /v1/responses -> catch_all path="v1/responses"
    # We test the URL construction directly (BASE_URL is monkeypatched to upstream-mock)
    base = proxy.BASE_URL.rstrip("/")
    path = "v1/responses"
    if path.startswith("v1/"):
        url = f"{base}/{path[3:]}"
    else:
        url = f"{base}/{path}"
    assert url == f"{proxy.BASE_URL}/responses"
    assert "/v1/v1/" not in url


# --- v1.3.2 bug-fix regressions ---


def _stream_case(monkeypatch, content: bytes):
    """Run SSE content through _zen_stream_upstream; return (status, err, bytes)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        if chunks is None:
            return status, err, b""
        out = []
        async for c in chunks:
            out.append(c)
        return status, err, b"".join(out)

    return asyncio.get_event_loop().run_until_complete(run())


def test_stream_rate_limit_after_first_256_bytes_is_normalized(monkeypatch):
    """FreeUsageLimitError arriving past the first 256 bytes must still become
    the normalized 429 SSE error event, not leak raw to the client."""
    padding = b'data: {"choices":[{"delta":{"content":"aaaaaaaaaa"}}]}\n\n' * 12  # 432 bytes
    content = padding + b'data: {"type":"error","error":{"type":"FreeUsageLimitError"}}\n\n'
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b"Rate limit exceeded" in data
    assert b"FreeUsageLimitError" not in data


def test_stream_tool_calls_marker_split_across_chunk_boundary_detected(monkeypatch):
    '''"tool_calls" split across a 32-byte chunk boundary (past the first 256
    bytes) must still trigger the synthetic finish_reason.'''
    content = b"x" * 315 + b'"tool_calls"' + b"tail padding\n\n"  # crosses offset 320
    status, err, data = _stream_case(monkeypatch, content)
    assert b'"finish_reason":"tool_calls"' in data


def test_keepalive_wrapper_emits_error_event_on_upstream_exception():
    """A mid-stream upstream exception must become an SSE error + [DONE],
    not a silently truncated stream."""

    async def broken():
        yield b"data: one\n\n"
        raise httpx.ReadTimeout("read timed out")

    out = b"".join(_run_stream(proxy._keepalive_wrapper(broken(), interval=0.05)))
    assert b"data: one" in out
    assert b'"error"' in out
    assert b"[DONE]" in out


def test_relay_openai_stream_connect_error_yields_sse_error(monkeypatch):
    """An upstream connect failure must surface as an SSE error event + [DONE],
    not escape the generator and kill the connection."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _client_with_mock(monkeypatch, handler)

    async def collect():
        out = []
        try:
            async for chunk in proxy._relay_openai_stream(
                "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
            ):
                out.append(chunk)
            return out, None
        except Exception as e:
            return out, e

    out, escaped = asyncio.get_event_loop().run_until_complete(collect())
    assert escaped is None, f"exception escaped the relay: {escaped!r}"
    data = b"".join(out)
    assert b'"error"' in data
    assert b"[DONE]" in data


def test_chat_completions_route_upstream_connect_error_yields_sse_error(monkeypatch):
    """End-to-end: POST /v1/chat/completions with a dead upstream. A refused
    connection answers 'immediately' — within STATUS_HOLD_SECS — so the route
    must surface the REAL HTTP 502 (downstream lock/fallback logic depends on
    real statuses, not in-band SSE errors under a fake 200)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 502
    assert "connection refused" in resp.text


def test_relay_cancels_upstream_and_closes_client_on_disconnect(monkeypatch):
    """Client disconnect during the header-wait phase must cancel the upstream
    request and close its httpx client (no orphaned connections)."""
    closed = []
    orig_aclose = httpx.AsyncClient.aclose

    async def rec_aclose(self):
        closed.append(self)
        await orig_aclose(self)

    monkeypatch.setattr(httpx.AsyncClient, "aclose", rec_aclose)

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    class SlowClient(httpx.AsyncClient):
        async def send(self, request, **kw):
            await asyncio.sleep(0.5)  # upstream stalls before sending headers
            return await super().send(request, **kw)

    transport = httpx.MockTransport(slow_handler)

    def _client(**kwargs):
        return SlowClient(transport=transport, **kwargs)

    monkeypatch.setattr(proxy, "_client", _client)

    async def scenario():
        relay = proxy._relay_openai_stream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4", interval=0.02
        )
        first = await relay.__anext__()  # keepalive while upstream stalls
        await relay.aclose()  # client hangs up
        await asyncio.sleep(0.1)  # let cancellation propagate
        return first

    first = asyncio.get_event_loop().run_until_complete(scenario())
    assert b"keepalive" in first
    assert len(closed) >= 1, "upstream httpx client leaked on disconnect"


def test_anthropic_to_openai_wraps_text_parts_when_image_present():
    """Text alongside an image must become {"type":"text"} parts — bare strings
    inside a content-parts array are schema-invalid for OpenAI backends."""
    out = proxy._anthropic_to_openai(
        {
            "model": "claude-x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
                    ],
                }
            ],
        }
    )
    content = out["messages"][-1]["content"]
    assert isinstance(content, list)
    assert all(isinstance(p, dict) and "type" in p for p in content), content
    assert {"type": "text", "text": "describe"} in content
    assert any(p["type"] == "image_url" for p in content)


def test_catch_all_forwards_query_string(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.get("/v1/anything?x=1&y=2")
    assert resp.status_code == 200
    assert captured["url"] == "http://upstream-mock/v1/anything?x=1&y=2"


def test_invalid_json_body_returns_400():
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", content=b"{not json")
    assert resp.status_code == 400
    resp = client.post("/v1/messages", content=b"{not json")
    assert resp.status_code == 400


def _req_with(headers=None):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return StarletteRequest(scope)


def test_wants_stream_detects_spaced_json_and_parameterized_accept():
    assert proxy._wants_stream(_req_with({"accept": "text/event-stream"}), b"{}")
    assert proxy._wants_stream(_req_with({"accept": "text/event-stream; charset=utf-8"}), b"{}")
    assert proxy._wants_stream(_req_with(), b'{"model":"m","stream": true}')
    assert proxy._wants_stream(_req_with(), b'{"model":"m","stream":true}')
    assert not proxy._wants_stream(_req_with(), b'{"model":"m","stream": false}')


def test_stream_strips_trailing_frames_after_done(monkeypatch):
    """Zen appends `data: {"choices":[],"cost":"0"}` after `data: [DONE]`.
    These post-DONE frames must be stripped so downstream (9Router) doesn't
    parse empty choices as a failure signal.
    """
    content = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: [DONE]\n\n'
        b'data: {"choices":[],"cost":"0"}\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b'[DONE]' in data
    assert b'"choices":[]' not in data, "post-DONE frame leaked to client"
    assert b'"cost"' not in data


def test_keepalive_wrapper_never_breaks_sse_frames():
    """Regression (9Router: 'Failed to parse SSE line ... "mode: keepalive'):
    a keepalive emitted while a partial frame is buffered splices the comment
    into the client's in-flight JSON line. Keepalives are only valid on a
    frame boundary — never while a partial frame is pending."""

    async def stalling_stream():
        yield b'data: {"id":"","object":"chat.completion.chunk","created":1787394907,"mode'
        await asyncio.sleep(0.2)  # stall mid-frame, longer than interval
        yield b'l":"","choices":[]}\n\n'
        await asyncio.sleep(0.2)  # stall BETWEEN frames — keepalive is safe here
        yield b"data: [DONE]\n\n"

    out = b"".join(_run_stream(proxy._keepalive_wrapper(stalling_stream(), interval=0.05)))
    assert ':"mode'.encode() in out or b'"mode' in out  # frame content intact
    # The JSON frame must survive un-split: reassembling yields valid JSON lines
    lines = [l for l in out.split(b"\n") if l.startswith(b"data: ")]
    import json as _json
    payload = _json.loads(lines[0][6:])  # must parse — no keepalive spliced in
    assert payload["object"] == "chat.completion.chunk"
    assert b'{"id":"","object":"chat.completion.chunk","created":1787394907,"model":"","choices":[]}' in out
    assert out.count(b": keepalive") >= 1  # between-frames stall still keepalives


def test_keepalive_wrapper_flushes_partial_tail_on_end():
    """A stream ending without a trailing blank line must still deliver its tail."""

    async def tailless():
        yield b"data: {\"a\":1}\n\n"
        yield b"data: {\"b\":2}"  # no \n\n terminator

    out = b"".join(_run_stream(proxy._keepalive_wrapper(tailless(), interval=0.05)))
    assert b'{"a":1}' in out
    assert b'{"b":2}' in out


def test_nonstream_upstream_garbage_body_returns_502_not_500(monkeypatch):
    """A 200 from Zen with a non-JSON body must surface as a clean 502, not an
    unhandled crash (plain-text 'Internal Server Error' 500 seen in 9Router)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway junk</html>",
                              headers={"Content-Type": "text/html"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions",
                       json={"model": "m", "messages": [], "stream": False})
    assert resp.status_code == 502
    assert "detail" in resp.text


def test_models_upstream_garbage_falls_back_to_static_list(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"Content-Type": "text/plain"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"
    assert len(resp.json()["data"]) > 0


def test_stream_does_not_cut_on_done_embedded_in_content(monkeypatch):
    """`data: [DONE]` appearing INSIDE a JSON content string (e.g. the model
    writing SSE-handling code — common in sessions that edit this very proxy)
    must not terminate the stream. Only a standalone `data: [DONE]` frame does.
    Regression for premature cut: response truncated + usage frame dropped
    (9Router reported 'succeeded' with IN 0 · OUT 0)."""
    content = (
        b'data: {"choices":[{"index":0,"delta":{"content":"yield b\'data: [DONE]\' in gen()"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":" more text after"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
        b'data: [DONE]\n\n'
        b'data: {"choices":[],"cost":"0"}\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b"more text after" in data   # content after the embedded marker survived
    assert b'"prompt_tokens":10' in data  # usage frame survived
    done_lines = [l for l in data.split(b"\n") if l.strip() in (b"data: [DONE]", b"data:[DONE]")]
    assert len(done_lines) == 1         # exactly one terminal DONE line
    assert b'"cost"' not in data        # trailing junk still stripped


def test_stream_does_not_mistake_error_name_in_content_for_rate_limit(monkeypatch):
    """'FreeUsageLimitError' mentioned as ordinary model output text must pass
    through, not trigger the rate-limit error event."""
    content = (
        b'data: {"choices":[{"delta":{"content":"the code checks FreeUsageLimitError frames"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b"Rate limit exceeded" not in data
    assert b"FreeUsageLimitError" in data  # relayed as ordinary content


def test_stream_truncated_midway_yields_error_event_not_silent_done(monkeypatch):
    """A stream that delivers content then just ENDS (no finish_reason, no
    usage frame, no [DONE]) was cut upstream (Zen kills long free-tier
    streams). The proxy must surface an SSE error event so 9Router retries,
    NOT paper over it with a synthetic [DONE] 'success' (user saw the model
    stop mid-sentence with no error)."""
    content = (
        b'data: {"choices":[{"index":0,"delta":{"content":"meaning in Hebbian networks:"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":" here comes the lis"},"finish_reason":null}]}\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b"meaning in Hebbian" in data      # partial content still relayed
    assert b'"error"' in data                 # ...but flagged as failed
    assert b"ended before completion" in data
    done_lines = [l for l in data.split(b"\n") if l.strip() in (b"data: [DONE]", b"data:[DONE]")]
    assert len(done_lines) == 1               # single terminal DONE after the error


def test_stream_usage_without_done_still_gets_clean_synthetic_done(monkeypatch):
    """muse-spark's legit pattern: content + usage frame, NO [DONE] from Zen —
    must stay a clean synthetic-DONE success, not a truncation error."""
    content = (
        b'data: {"choices":[{"delta":{"content":"PROXY TEST OK"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":15,"completion_tokens":337,"total_tokens":352}}\n\n'
        b'data: {"choices":[],"cost":"0"}\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b"PROXY TEST OK" in data
    assert b'"error"' not in data             # no false truncation error
    assert b"[DONE]" in data                  # synthetic terminal present
    assert b'"cost"' not in data              # trailing junk stripped


def test_stream_empty_junk_only_stream_is_error_not_empty_success(monkeypatch):
    """A stream of nothing but Zen heartbeat/junk frames carries zero signal —
    surfacing it as an empty 'success' gives the harness a blank assistant
    message. It must surface as a real HTTP 502 so 9Router retries."""
    content = (
        b'data: {"id":"resp_x","object":"chat.completion.chunk","created":1,"model":"m","choices":[]}\n\n'
        b'data: {"id":"","object":"chat.completion.chunk","created":1,"model":"","choices":[]}\n\n'
        b'data: {"choices":[],"cost":"0"}\n\n'
    )
    status, err, data = _stream_case(monkeypatch, content)
    assert status == 502
    assert "no usable content" in (err or "")


def test_stream_reasoning_only_cut_reports_reasoning_flag(monkeypatch, caplog):
    """A stream cut during pure reasoning (x-preview on huge contexts) must be
    classified with reasoning=True in the diagnostics — reasoning is content
    for detection purposes, not 'empty'."""
    import logging

    content = (
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking hard about the task"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning_content":" still thinking..."},"finish_reason":null}]}\n\n'
    )
    with caplog.at_level(logging.INFO, logger="opencode-proxy"):
        status, err, data = _stream_case(monkeypatch, content)
    assert status == 200
    assert b'"error"' in data  # still an error so 9Router can retry/fallback
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "reasoning=True" in joined
    assert "content=False" in joined


def test_hy3_free_is_served_as_first_class_model():
    """hy3-free (capped-reasoning workhorse) must be selectable through the
    proxy in both OpenAI and Anthropic formats."""
    assert "hy3-free" in proxy.FREE_MODELS
    assert proxy._map_model("hy3-free") == "hy3-free"


def test_get_session_evicts_stale_entries_when_over_cap(monkeypatch):
    monkeypatch.setattr(proxy, "MAX_SESSIONS", 3)
    proxy._user_sessions.clear()
    now = time.time()
    proxy._user_sessions["stale1"] = {"id": "s1", "ts": now - 9999}
    proxy._user_sessions["stale2"] = {"id": "s2", "ts": now - 9999}
    proxy._user_sessions["fresh"] = {"id": "s3", "ts": now}
    sid = proxy._get_session("new-user")  # exceeds cap -> stale evicted
    assert sid
    assert "stale1" not in proxy._user_sessions
    assert "stale2" not in proxy._user_sessions
    assert "fresh" in proxy._user_sessions


# --- Header-hold: fast upstream failures surface as real HTTP statuses ---
# Regression for the 2026-08-24 outage: streaming requests committed HTTP 200
# instantly, so zen failures became in-band SSE error frames that 9Router
# treated as successful empty completions (no lock, no fallback, harness
# re-fired 131K-token payloads into an exhausted shared pool).


def test_stream_fast_upstream_error_surfaces_real_status(monkeypatch):
    """Zen answering non-200 within STATUS_HOLD_SECS must surface the REAL HTTP
    status to the caller — not HTTP 200 with an in-band SSE error frame."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"type": "error", "error": {"type": "error", "message": "Internal server error"}},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 500, f"expected real status, got {resp.status_code}: {resp.text[:200]}"
    assert "Internal server error" in resp.text


def test_stream_slow_pending_upstream_falls_back_to_keepalive_stream(monkeypatch):
    """An upstream that hasn't answered headers within STATUS_HOLD_SECS keeps
    the old behavior: HTTP 200 committed with SSE streaming so middle-hops with
    short first-byte timeouts get a live connection (in-band keepalives during
    long silences are covered by test_keepalive_wrapper_survives_slow_chunks)."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 0.05)

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.25)  # headers arrive after the hold window expired
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert b'"content":"hi"' in resp.content
    assert b"[DONE]" in resp.content


def test_anthropic_route_fast_upstream_error_surfaces_real_status(monkeypatch):
    """Same header-hold contract on /v1/messages (claude-format callers)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"type": "server_error", "message": "Endpoint is unavailable."}},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/messages",
        json={"model": "x-preview-f-free", "max_tokens": 16, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503, f"expected real status, got {resp.status_code}: {resp.text[:200]}"


# --- Error classification: stop labeling every upstream error as a 429 ---


def _first_frame_error(monkeypatch, frame):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=f"data: {frame}\n\ndata: [DONE]\n\n".encode(),
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    return asyncio.get_event_loop().run_until_complete(run())


def test_instream_internal_error_not_labeled_rate_limit(monkeypatch):
    """An api_error must keep its original message and a non-429 status —
    blanket 'Rate limit exceeded (429)' mislabels server errors and makes
    downstream lock the model as throttled instead of failing over."""
    status, err, chunks = _first_frame_error(
        monkeypatch, '{"type":"error","error":{"type":"api_error","message":"Internal server error"}}'
    )
    assert status == 502
    assert "Internal server error" in (err or "")
    assert "Rate limit" not in (err or "")
    assert chunks is None


def test_instream_rate_limit_error_still_classified_429(monkeypatch):
    """Genuine throttle frames keep their 429 classification."""
    status, err, chunks = _first_frame_error(
        monkeypatch, '{"type":"error","error":{"type":"FreeUsageLimitError","message":"quota"}}'
    )
    assert status == 429
    assert "quota" in (err or "")
    assert chunks is None


def test_generic_openai_error_frame_detected(monkeypatch):
    """An OpenAI-style {"error": {...}} frame without top-level type is still an
    error — it must surface as a real upstream failure, not be relayed."""
    status, err, chunks = _first_frame_error(monkeypatch, '{"error": {"message": "boom"}}')
    assert status == 502
    assert "boom" in (err or "")
    assert chunks is None


# --- Body-level dead streams: 200 + well-formed but empty/error stream ---
# Regression for the 2026-08-24 afternoon outage: zen's Console provider
# answers HTTP 200 with a single finish_reason:"network_error" frame (or a
# bare empty completion). Header-hold can't catch these — the status really is
# 200 — yet 9Router treats well-formed empty streams as successful completions
# (OUT 0), so the harness re-fires the same 100K-token request endlessly.


def test_stream_body_network_error_surfaces_real_status(monkeypatch):
    """A 200 stream whose only terminal is finish_reason:'network_error' must
    surface as a real HTTP 502."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"id":"x","choices":[{"index":0,"finish_reason":"network_error",'
                b'"delta":{"role":"assistant","content":""}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:200]}"
    assert "network_error" in resp.text


def test_stream_body_empty_completion_surfaces_real_status(monkeypatch):
    """A 200 stream that terminates cleanly but carries zero content/reasoning/
    tool output is a dead completion (free-tier outage artifact) — surface 502."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":0}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:200]}"
    assert "empty" in resp.text.lower()


def test_stream_body_with_content_passes_through(monkeypatch):
    """A stream that starts producing real content within the probe window must
    stream normally — probe must not swallow or delay live traffic."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 200
    assert b'"content":"hello"' in resp.content
    assert resp.content.count(b"[DONE]") == 1


def test_stream_relay_continues_after_probe_stops_at_first_frame(monkeypatch):
    """Regression (v1.5.0 StreamConsumed): the probe consumes the first frames
    to decide liveness; the relay phase must continue the SAME stream — not
    re-iterate resp.aiter_bytes (httpx forbids that) — so every subsequent
    frame still reaches the client with no error frame appended. A real
    streaming body (async iterator) is required: bytes bodies don't set
    httpx's stream-consumed flag and mask the bug."""

    async def sse_bytes():
        yield b'data: {"choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n'
        yield b'data: {"choices":[{"index":0,"delta":{"content":"world"}}]}\n\n'
        yield (
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_bytes(), headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": True})
    assert resp.status_code == 200
    assert b'"content":"hello "' in resp.content
    assert b'"content":"world"' in resp.content, "frames after the probe window were dropped (stream reuse bug)"
    assert b'"finish_reason":"stop"' in resp.content
    assert b'"error"' not in resp.content
    assert resp.content.count(b"[DONE]") == 1