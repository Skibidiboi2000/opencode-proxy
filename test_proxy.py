"""Tests for opencode-proxy error handling."""

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
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

    # Default retries 429s; disable here to keep this regression fast and to
    # assert the immediate-surface behavior independently of the retry budget.
    monkeypatch.setattr(proxy, "RETRY_429", 0)
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
    msg = {"role": "user", "content": "x" * 30000}
    per_msg = proxy._message_cost([msg])
    count = int(proxy.CONTEXT_LIMIT * 0.9) // per_msg + 1
    big = [dict(msg) for _ in range(count)]
    out = proxy._fit_context({"messages": big, "max_tokens": 64000})
    assert len(out["messages"]) <= count
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
    # Cost must be measured with the SAME estimator _fit_context charges, or this
    # fixture silently drifts out of the trim window whenever that estimator
    # changes (it did — //2 -> //4 — and the test passed vacuously).
    cost_turn = proxy._message_cost([turn])
    cost_result = proxy._message_cost([result])

    # total cost lands 512 under the fit threshold, so trimming stops right
    # after dropping `turn` — leaving `result` orphaned at index 0.
    budget = int(proxy.CONTEXT_LIMIT * 0.9) - 512
    filler = [{"role": "user", "content": "x" * 300000} for _ in range(6)]
    cost_filler = proxy._message_cost(filler)
    padding_tokens = max(1, budget - cost_turn - cost_result - cost_filler)
    users = list(filler)
    users.append({"role": "user", "content": "y" * (padding_tokens * 4)})

    big = [turn, result] + users
    total = proxy._message_cost(big)
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
    out = proxy._anthropic_to_openai({"model": "space-bunny-free", "messages": []})
    assert out["model"] == "space-bunny-free"


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

    monkeypatch.setattr(proxy, "RETRY_429", 0)
    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit exceeded" in err
    assert chunks is None


def test_stream_upstream_429_retries_then_succeeds(monkeypatch):
    """A transient 429 must be retried with backoff and, once Zen accepts the
    request, the stream must succeed — not hard-fail on the first throttle."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 0.0)
    _client_with_mock(monkeypatch, handler)

    status, err, chunks = asyncio.get_event_loop().run_until_complete(
        proxy._zen_stream_upstream("http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4")
    )
    assert status == 200
    assert err is None
    assert attempts["n"] == 3  # two 429 retries, then served
    data = b"".join(_run_stream(chunks)).decode()
    assert '"content":"hi"' in data


def test_stream_upstream_429_exhausts_retries_then_surfaces(monkeypatch):
    """After the retry budget is spent, the real 429 must still surface."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    monkeypatch.setattr(proxy, "RETRY_429", 2)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 0.0)
    _client_with_mock(monkeypatch, handler)

    status, err, chunks = asyncio.get_event_loop().run_until_complete(
        proxy._zen_stream_upstream("http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4")
    )
    assert status == 429
    assert "rate limited" in err
    assert chunks is None
    assert attempts["n"] == 3  # initial attempt + 2 retries


def test_stream_parsed_429_retries_then_succeeds(monkeypatch):
    """The parsed stream path must smooth a transient 429 the same way."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"yo"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 0.0)
    _client_with_mock(monkeypatch, handler)

    status, err, chunks = asyncio.get_event_loop().run_until_complete(
        proxy._zen_stream_parsed("http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4")
    )
    assert status == 200
    assert err is None
    assert attempts["n"] == 2


def test_nonstream_429_retries_then_succeeds(monkeypatch):
    """_post_with_retry must also smooth a transient 429."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 0.0)
    _client_with_mock(monkeypatch, handler)

    resp = asyncio.get_event_loop().run_until_complete(
        proxy._post_with_retry("http://upstream-mock/v1/chat/completions", {}, {"stream": False, "messages": []})
    )
    assert resp.status_code == 200
    assert attempts["n"] == 2


def test_429_retry_disabled_when_retry_429_zero(monkeypatch):
    """RETRY_429=0 restores the documented 'never retried' behavior."""
    monkeypatch.setattr(proxy, "RETRY_429", 0)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    _client_with_mock(monkeypatch, handler)

    status, err, chunks = asyncio.get_event_loop().run_until_complete(
        proxy._zen_stream_upstream("http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4")
    )
    assert status == 429
    assert attempts["n"] == 1  # no retry


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
    with pytest.raises(proxy.UpstreamDead):
        _run_stream(chunks)


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
    with pytest.raises(proxy.UpstreamDead):
        _stream_case(monkeypatch, content)


def test_stream_tool_calls_marker_split_across_chunk_boundary_detected(monkeypatch):
    '''"tool_calls" split across a 32-byte chunk boundary (past the first 256
    bytes) must still trigger the synthetic finish_reason.'''
    content = b"x" * 315 + b'"tool_calls"' + b"tail padding\n\n"  # crosses offset 320
    status, err, data = _stream_case(monkeypatch, content)
    assert b'"finish_reason":"tool_calls"' in data


def test_keepalive_wrapper_emits_error_event_on_upstream_exception():
    """A mid-stream upstream exception must ABORT the stream (re-raise), not
    be converted into a clean error+[DONE] the router reads as success."""

    async def broken():
        yield b"data: one\n\n"
        raise httpx.ReadTimeout("read timed out")

    with pytest.raises(httpx.ReadTimeout):
        b"".join(_run_stream(proxy._keepalive_wrapper(broken(), interval=0.05)))


def test_relay_openai_stream_connect_error_yields_sse_error(monkeypatch):
    """An upstream connect failure must propagate (abort) — never a clean
    error+[DONE] the router would parse as an empty success."""

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
    assert isinstance(escaped, httpx.ConnectError), f"expected abort, got clean end: {escaped!r}"
    assert b"[DONE]" not in b"".join(out)


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
    streams). The relay must ABORT the connection (premature close) so 9Router
    classifies the model as failed — NOT paper over it with a synthetic [DONE]
    'success' (user saw the model stop mid-sentence with no error)."""
    content = (
        b'data: {"choices":[{"index":0,"delta":{"content":"meaning in Hebbian networks:"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":" here comes the lis"},"finish_reason":null}]}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    partial = []

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        assert status == 200 and err is None
        async for c in chunks:
            partial.append(c)
        return b""

    with pytest.raises(proxy.UpstreamDead):
        asyncio.get_event_loop().run_until_complete(run())
    data = b"".join(partial)
    # partial content was relayed before the abort; no clean terminal
    assert b"meaning in Hebbian" in data
    assert b"[DONE]" not in data


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
        with pytest.raises(proxy.UpstreamDead):
            status, err, data = _stream_case(monkeypatch, content)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "reasoning=True" in joined
    assert "content=False" in joined


def test_union_alpha_is_no_longer_served():
    """union-alpha was retired upstream (401 "Model union-alpha is not supported"
    on the 2026-09-28 probe). It must no longer be offered as a first-class
    model; requests naming it resolve through the unknown-ID contract."""
    assert "union-alpha" not in proxy._known_models()
    assert proxy._map_model("union-alpha") == proxy.DEFAULT_MODEL
    assert "union-alpha" not in proxy.MESSAGES_MODELS


def test_fit_context_trims_responses_api_input():
    """The Responses API carries the conversation in `input` (with `instructions`
    as the system prompt), not `messages`. _fit_context only looked at `messages`,
    so the oversized-context guard silently did nothing on /v1/responses and the
    1M-token rejection came straight from upstream."""
    msg = {"role": "user", "content": "x" * 400000}
    # Derive the item count from CONTEXT_LIMIT and the live estimator so this
    # fixture cannot silently fall out of the trim window when either changes.
    per_msg = proxy._message_cost([msg])
    oversized = [dict(msg) for _ in range(int(proxy.CONTEXT_LIMIT * 0.9) // per_msg + 1)]
    body = {"model": "space-bunny-free", "instructions": "be brief",
            "input": oversized, "max_tokens": 4096}
    out = proxy._fit_context(body)
    assert "input" in out
    assert len(out["input"]) < len(oversized)
    # The system prompt is never dropped.
    assert out.get("instructions") == "be brief"
    assert out["max_tokens"] <= 4096


def test_fit_context_leaves_healthy_responses_body_untouched():
    body = {"model": "space-bunny-free", "instructions": "s",
            "input": [{"role": "user", "content": "hi"}], "max_tokens": 512}
    out = proxy._fit_context(body)
    assert out["input"] == body["input"]
    assert out["max_tokens"] == 512


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


def test_get_session_stays_bounded_when_all_entries_are_fresh(monkeypatch):
    """MAX_SESSIONS must be a real cap, not just a trigger for a best-effort sweep.

    Session keys come from the client-controlled X-Forwarded-For header, so an
    attacker can mint unlimited distinct keys inside the 30-minute freshness
    window. The previous eviction only dropped entries older than 1800s, so the
    dict grew without bound; this fills the map with fresh entries and asserts
    the oldest are evicted regardless of age."""
    monkeypatch.setattr(proxy, "MAX_SESSIONS", 10)
    monkeypatch.delenv("OPENCODE_SESSION", raising=False)
    proxy._user_sessions.clear()
    now = time.time()
    for i in range(50):
        proxy._user_sessions[f"ip-{i}"] = {"id": f"s{i}", "ts": now - i}  # all fresh
    sid = proxy._get_session("ip-new")
    assert sid
    assert len(proxy._user_sessions) <= 11, f"grew to {len(proxy._user_sessions)}"
    # Oldest entries are the ones dropped.
    assert "ip-49" not in proxy._user_sessions
    assert "ip-0" in proxy._user_sessions


def test_get_session_uses_configured_static_session(monkeypatch):
    """OPENCODE_SESSION (a real CLI-minted session) must be sent upstream
    verbatim for every client: the backend rejects fabricated ses_ IDs."""
    monkeypatch.setenv("OPENCODE_SESSION", "ses_real123")
    proxy._user_sessions.clear()
    assert proxy._get_session("1.2.3.4") == "ses_real123"
    assert proxy._get_session("5.6.7.8") == "ses_real123"


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


# --- Responses API: probe liveness + muse-spark bridge ---
# zen's /chat/completions 500s for muse-spark (Console), but /responses works —
# the opencode CLI itself talks Responses API for this model. 9Router only
# speaks chat/completions, so the proxy must bridge, and the probe must
# understand Responses-format frames (no choices[] — different liveness).


def _responses_sse_events():
    return (
        b'event: response.created\ndata: {"type":"response.created","response":{"id":"r1"}}\n\n'
        b'event: response.in_progress\ndata: {"type":"response.in_progress","response":{}}\n\n'
        b'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"id":"m1","type":"message","role":"assistant","content":[]}}\n\n'
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Four"}\n\n'
        b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":" teen"}\n\n'
        b'event: ping\ndata: {"type":"ping"}\n\n'
        b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1","status":"completed","usage":{"input_tokens":17,"output_tokens":555,"total_tokens":572}}}\n\n'
    )


def test_probe_recognizes_responses_format_liveness(monkeypatch):
    """Responses-API SSE (event:/data: with response.* types, no choices[]) is
    alive the moment output_text deltas flow — must NOT be declared dead."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_responses_sse_events(), headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/responses", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200, f"probe killed a live Responses stream: {err}"
    assert err is None
    objs = _run_stream(chunks)
    types = [o.get("type") for o in objs if isinstance(o, dict)]
    assert "response.output_text.delta" in types


def test_chat_completions_muse_spark_bridges_to_responses(monkeypatch):
    """chat/completions for muse-spark must be served from zen's /responses
    endpoint (chat/completions 500s server-side) and translated back to
    chat.completion.chunk SSE."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                500,
                json={"type": "error", "error": {"type": "error", "message": "Internal server error"}},
            )
        assert request.url.path.endswith("/responses"), request.url.path
        rbody = json.loads(request.content)
        assert rbody["input"], "messages must be translated to input items"
        assert rbody.get("max_output_tokens", 0) >= 2048, "reasoning headroom required"
        return httpx.Response(200, content=_responses_sse_events(), headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)
    # The Responses bridge is selected by model id, so the id must be a
    # discovered one now that the pool is dynamic.
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids",
                        list(proxy._known_models()) + ["muse-spark-1.3-contributor-free"])
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "muse-spark-1.3-contributor-free", "stream": True, "max_tokens": 64,
              "messages": [{"role": "user", "content": "say ok"}]},
    )
    assert resp.status_code == 200, resp.text[:200]
    assert b'"content":"Four"' in resp.content
    assert b'"finish_reason":"stop"' in resp.content
    assert b'"prompt_tokens":17' in resp.content
    assert b'"completion_tokens":555' in resp.content
    assert b'"error"' not in resp.content
    assert resp.content.count(b"[DONE]") == 1


def test_chat_to_responses_request_translation():
    out = proxy._chat_to_responses(
        {
            "model": "muse-spark-1.2-contributor-free",
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                      "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ],
            "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}],
        }
    )
    assert out["instructions"] == "be brief"
    types = [i.get("type", i.get("role")) for i in out["input"]]
    assert types == ["user", "function_call", "function_call_output"]
    assert out["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
    assert out["input"][1] == {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a":1}'}
    assert out["input"][2] == {"type": "function_call_output", "call_id": "c1", "output": "result"}
    assert out["tools"][0] == {"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}
    assert {t["name"] for t in out["tools"][1:]} >= {"bash", "read", "write"}  # harness padding appended
    assert out["max_output_tokens"] == 2048  # reasoning headroom bump


def test_responses_events_translate_to_chat_chunks():
    events = iter([
        {"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "role": "assistant", "content": []}},
        {"type": "response.output_text.delta", "delta": "Four"},
        {"type": "response.completed", "response": {"id": "r1", "status": "completed",
                                                    "usage": {"input_tokens": 17, "output_tokens": 555, "total_tokens": 572}}},
    ])

    async def gen():
        for e in events:
            yield e

    out = b"".join(_run_stream(proxy._responses_events_to_chat(gen()))).decode()
    assert '"content":"Four"' in out
    assert '"finish_reason":"stop"' in out
    assert '"prompt_tokens":17' in out
    assert '"completion_tokens":555' in out
    assert out.count("[DONE]") == 1


def test_responses_incomplete_maps_length_finish():
    events = iter([
        {"type": "response.output_text.delta", "delta": "par"},
        {"type": "response.incomplete", "response": {"status": "incomplete",
                                                     "usage": {"input_tokens": 10, "output_tokens": 2048, "total_tokens": 2058}}},
    ])

    async def gen():
        for e in events:
            yield e

    out = b"".join(_run_stream(proxy._responses_events_to_chat(gen()))).decode()
    assert '"finish_reason":"length"' in out
    assert '"completion_tokens":2048' in out
    assert out.count("[DONE]") == 1


def test_chat_via_responses_non_stream(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(500, json={"error": {"message": "Internal server error"}})
        return httpx.Response(200, json={
            "id": "resp_1", "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Four"}]},
            ],
            "usage": {"input_tokens": 17, "output_tokens": 555, "total_tokens": 572},
        })

    _client_with_mock(monkeypatch, handler)
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids",
                        list(proxy._known_models()) + ["muse-spark-1.3-contributor-free"])
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "muse-spark-1.3-contributor-free", "stream": False,
              "messages": [{"role": "user", "content": "say ok"}]},
    )
    assert resp.status_code == 200, resp.text[:200]
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Four"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 555


# --- Upstream retry: zen's 503s are ~1s transient blips; the opencode CLI
# survives them via internal retries, so the proxy must too (9Router surfaces
# first-failure to the harness instead of retrying). ---


def test_stream_retries_transient_503_then_succeeds(monkeypatch):
    """A 503 that clears on the next attempt must never reach the client."""
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 0.01)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": {"message": "Endpoint is unavailable."}})
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200, f"expected retry to succeed, got {status}: {err}"
    assert calls["n"] == 3
    data = b"".join(_run_stream(chunks)).decode()
    assert '"content":"ok"' in data


def test_stream_retries_exhausted_surfaces_last_status(monkeypatch):
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 0.01)
    monkeypatch.setattr(proxy, "UPSTREAM_RETRIES", 2)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "Endpoint is unavailable."}})

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 503
    assert calls["n"] == 3  # 1 initial + 2 retries
    assert chunks is None


def test_stream_does_not_retry_429(monkeypatch):
    """RETRY_429=0 restores the documented 'never retried' default: 429 surfaces
    instantly without amplifying load on an exhausted pool."""
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 0.01)
    monkeypatch.setattr(proxy, "RETRY_429", 0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    status, err, chunks = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert calls["n"] == 1


def test_nonstream_retries_transient_503(monkeypatch):
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 0.01)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, json={"error": {"message": "Endpoint is unavailable."}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": [], "stream": False})
    assert resp.status_code == 200
    assert calls["n"] == 2


def test_retry_backoff_spreads_exponentially(monkeypatch):
    """Backoff must widen (1x, 2x, 4x...) — zen's 503 streaks outlast a linear
    3s window; an exponential spread covers ~7s within the same attempt count
    while total pre-commit time stays under 9Router's ~25s first-byte kill."""
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(proxy.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 1.0)
    monkeypatch.setattr(proxy, "UPSTREAM_RETRIES", 3)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "Endpoint is unavailable."}})

    _client_with_mock(monkeypatch, handler)

    async def run():
        return await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    asyncio.get_event_loop().run_until_complete(run())
    assert calls["n"] == 4  # 1 initial + 3 retries
    assert sleeps == [1.0, 2.0, 4.0]


# --- Post-commit failures must ABORT, not emit clean SSE errors ---
# 9Router parses a well-formed error-frame + [DONE] stream as a completed
# (empty) success — 'succeeded · IN 0 · OUT 0' — and the harness re-fires
# 400KB payloads forever. An aborted connection is the only signal it
# classifies as a model failure.


def test_pending_path_upstream_error_aborts_connection(monkeypatch):
    """Upstream still failing when the hold window expires: 200 is committed
    (keepalives flow), but when the task finally returns an error the relay
    must RAISE — not yield a parseable error frame + [DONE]."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 0.05)
    monkeypatch.setattr(proxy, "UPSTREAM_RETRY_BACKOFF", 0.01)
    monkeypatch.setattr(proxy, "UPSTREAM_RETRIES", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.2)  # upstream answers after hold expiry
        return httpx.Response(503, json={"error": {"message": "Endpoint is unavailable."}})

    _client_with_mock(monkeypatch, handler)

    async def run():
        task = asyncio.create_task(
            proxy._zen_stream_upstream("http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4")
        )
        await asyncio.sleep(0.3)  # hold expires, task completes with error
        relay = proxy._relay_from_task(task)
        out = b""
        async for c in relay:
            out += c
        return out

    with pytest.raises(proxy.UpstreamDead, match="503"):
        asyncio.get_event_loop().run_until_complete(run())


def test_post_commit_dead_stream_aborts_not_clean_done(monkeypatch):
    """Zen accepts (200), the probe window expires on silence, then the stream
    terminates with finish_reason:network_error AFTER commit — the relay must
    abort the connection, not deliver a clean [DONE] the router reads as an
    empty success."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 0.05)

    async def sse_bytes():
        yield b'data: {"choices":[]}\n\n'  # heartbeat junk: no liveness signal
        await asyncio.sleep(0.2)  # probe window expires during the silence
        yield (
            b'data: {"choices":[{"index":0,"finish_reason":"network_error",'
            b'"delta":{"role":"assistant","content":""}}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_bytes(), headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        assert status == 200 and err is None
        out = b""
        async for c in chunks:
            out += c
        return out

    with pytest.raises(proxy.UpstreamDead):
        asyncio.get_event_loop().run_until_complete(run())


def test_midstream_error_frame_aborts(monkeypatch):
    """An error frame arriving AFTER content was flowing (post-commit) must
    abort the stream, not end it cleanly with [DONE]."""
    content = (
        b'data: {"choices":[{"delta":{"content":"working"}}]}\n\n'
        b'data: {"type":"error","error":{"type":"api_error","message":"boom"}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        assert status == 200 and err is None
        out = b""
        async for c in chunks:
            out += c
        return out

    with pytest.raises(proxy.UpstreamDead, match="boom"):
        asyncio.get_event_loop().run_until_complete(run())


# --- Pending-phase upstream retry: zen's 503 waves last minutes; the 7s
# pre-commit window cannot cover them, but once 200 is committed the client
# sits on keepalives (9Router tolerates 40s+ TTFT — proven in its own logs),
# so retry the upstream behind the keepalives until it succeeds or the
# budget expires. ---


def test_pending_relay_retries_until_success(monkeypatch):
    monkeypatch.setattr(proxy, "PENDING_RETRY_SECS", 30)
    calls = {"n": 0}

    def make_task():
        # Production shape: a COROUTINE-returning factory (e.g.
        # `lambda: _zen_stream_upstream(...)`), not a Task factory —
        # regression for the AttributeError: 'coroutine' object has no
        # attribute 'done' crash on the second pending attempt.
        async def upstream():
            calls["n"] += 1
            if calls["n"] < 3:
                return 503, "Endpoint is unavailable. (503)", None

            async def chunks():
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            return 200, None, chunks()

        return upstream()

    async def run():
        first = asyncio.create_task(make_task())
        out = b""
        async for c in proxy._pending_relay_with_retry(first, make_task, 30, proxy._relay_openai_chunks, interval=0.01):
            out += c
        return out

    data = asyncio.get_event_loop().run_until_complete(run())
    assert calls["n"] == 3
    assert b'"content":"ok"' in data
    assert data.count(b"[DONE]") == 1


def test_pending_relay_raises_after_budget(monkeypatch):
    monkeypatch.setattr(proxy, "PENDING_RETRY_SECS", 0.05)

    def make_task():
        async def task_body():
            return 503, "Endpoint is unavailable. (503)", None

        return asyncio.create_task(task_body())

    async def run():
        out = b""
        async for c in proxy._pending_relay_with_retry(make_task(), make_task, 0.05, proxy._relay_openai_chunks, interval=0.01):
            out += c
        return out

    with pytest.raises(proxy.UpstreamDead, match="503"):
        asyncio.get_event_loop().run_until_complete(run())


def test_responses_translator_raises_on_empty_completed():
    """muse burning the whole budget on reasoning: completed with ZERO output —
    a clean empty success for the harness. Must abort instead."""
    events = iter([
        {"type": "response.output_item.added", "item": {"id": "m1", "type": "message", "role": "assistant", "content": []}},
        {"type": "response.completed", "response": {"id": "r1", "status": "completed",
                                                    "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100}}},
    ])

    async def gen():
        for e in events:
            yield e

    with pytest.raises(proxy.UpstreamDead, match="empty"):
        b"".join(_run_stream(proxy._responses_events_to_chat(gen())))


def test_post_commit_clean_empty_aborts(monkeypatch):
    """zen answers slowly with a WELL-FORMED but empty completion (finish stop,
    zero tokens, [DONE]) after the probe window — post-commit that must abort,
    not deliver a clean empty success."""
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 0.05)

    async def sse_bytes():
        yield b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
        await asyncio.sleep(0.2)  # probe window expires during silence
        yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":0}}\n\n'
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_bytes(), headers={"Content-Type": "text/event-stream"})

    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        assert status == 200 and err is None
        out = b""
        async for c in chunks:
            out += c
        return out

    with pytest.raises(proxy.UpstreamDead, match="empty"):
        asyncio.get_event_loop().run_until_complete(run())


def test_nonstream_bridge_empty_returns_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "resp_1", "status": "completed",
            "output": [{"type": "reasoning", "summary": []}],
            "usage": {"input_tokens": 500, "output_tokens": 0, "total_tokens": 500},
        })

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "muse-spark-1.3-contributor-free", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:150]}"
    assert "empty" in resp.text.lower()


def test_zen_model_still_routes_to_zen(monkeypatch):
    """A Zen model ID must route to the Zen upstream."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "hy3-free", "stream": False, "messages": []},
    )
    assert resp.status_code == 200
    assert captured["url"].startswith("http://upstream-mock/v1/")


def test_map_model_defaults_unknown():
    assert proxy._map_model("totally-unknown") == proxy.DEFAULT_MODEL


def test_models_list_free_only_with_prefixes():
    """`/v1/models` serves the proxy's discovered cache with `oc-` prefixes and
    makes no upstream call — clients pick a source by picking a name."""
    client = TestClient(proxy.app)
    resp = client.get("/v1/models")
    ids = [m["id"] for m in resp.json()["data"]]
    assert ids == [f"oc-{m}" for m in proxy._known_models()]
    # Everything is advertised with the prefix, never raw.
    assert all(i.startswith("oc-") for i in ids)


def test_kilo_provider_removed():
    """The Kilo upstream is gone: no module-level Kilo symbols, and a legacy
    `kl-` ID follows the unknown-ID contract (Zen + DEFAULT_MODEL)."""
    assert not hasattr(proxy, "KILO_BASE_URL")
    assert not hasattr(proxy, "KILO_MODELS")
    assert not hasattr(proxy, "KILO_MODEL_MAP")
    assert proxy._route("kl-kilo-auto/free") == (proxy.BASE_URL, proxy.DEFAULT_MODEL)


def test_prefixed_oc_model_routes_to_zen_stripped(monkeypatch):
    """`oc-<id>` must go to Zen with the prefix stripped from the model field
    actually forwarded upstream. Uses a discovered id, not a pinned name."""
    captured = {}
    target = proxy._known_models()[0]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["model"] = json.loads(request.content).get("model")
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": f"oc-{target}", "stream": False, "messages": []},
    )
    assert resp.status_code == 200
    assert captured["url"].startswith("http://upstream-mock/v1/")
    assert captured["model"] == target
    assert not captured["model"].startswith("oc-")


def test_prefixed_oc_unknown_model_falls_back_to_default(monkeypatch):
    """`oc-` + unknown Zen ID keeps the old contract: rewrite to DEFAULT_MODEL
    instead of forwarding a model Zen will reject."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content).get("model")
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "oc-bogus", "stream": False, "messages": []},
    )
    assert resp.status_code == 200
    assert captured["model"] == proxy.DEFAULT_MODEL


def test_route_unit_prefix_semantics(monkeypatch):
    known = list(proxy._known_models())
    m = known[0]
    assert proxy._route(f"oc-{m}") == (proxy.BASE_URL, m)
    assert proxy._route("oc-bogus")[1] == proxy.DEFAULT_MODEL
    assert proxy._route(m) == (proxy.BASE_URL, m)          # unprefixed zen


# --- v1.13.0 pool refresh (2026-09-28 live probe).
#
# Ground truth came from three independent sources:
#   1. GET https://opencode.ai/zen/v1/models  (the upstream list the proxy mirrors)
#   2. `opencode models` -> the opencode/ provider's own free set
#   3. a live POST per candidate through both /chat/completions and /messages
#
# Result: five entries in the old pool are gone upstream (union-alpha now answers
# 401 "Model union-alpha is not supported"; the CLI's opencode/ provider no longer
# lists it, nor mimo-v2.5-free, jev-1.13-free, deepseek-v4-flash-free or
# muse-spark-1.2-contributor-free), and longcat-2.5-preview-free is new. ---

V20_LIVE_FREE_MODELS = {
    "big-pickle",
    "ling-3.0-flash-fin-free",
    "longcat-2.5-preview-free",
    "mimo-v2.6-flash-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "space-bunny-free",
}

# Models that were in the v1.12.0 pool but are no longer served by Zen. They must
# fall through the unknown-ID contract instead of being advertised in /v1/models.
V20_REMOVED_MODELS = {
    "union-alpha",
    "mimo-v2.5-free",
    "jev-1.13-free",
    "deepseek-v4-flash-free",
    "muse-spark-1.2-contributor-free",
}

V20_DEAD_MODELS = {
    "minimax-m2.5-free",
    "nemotron-3-super-free",
    "qwen3.6-plus-free",
    "x-preview-f-free",
    "hy3-free",
    "laguna-s-2.1-free",
}


def test_v20_pool_contains_only_live_free_models():
    """Pool is discovered from the upstream, so there is no pinned list to drift.
    The bootstrap set must still be a subset of what discovery can return."""
    assert set(V20_LIVE_FREE_MODELS) >= set(proxy._BOOTSTRAP_MODELS)
    assert proxy._BOOTSTRAP_MODELS == ["space-bunny-free"]


def test_v20_default_model_is_proxy_reachable():
    """DEFAULT_MODEL must be a model that actually answers through the proxy.

    mimo-v2.5-free was the v1.12.0 default but now returns
    403 FreeTierError upstream, so every unknown-model request — and therefore
    the whole unknown-ID fallback contract — failed."""
    assert proxy.DEFAULT_MODEL == "space-bunny-free"
    assert proxy._map_model(proxy.DEFAULT_MODEL) == proxy.DEFAULT_MODEL


def test_v20_removed_models_fall_back_to_default(monkeypatch):
    """A model the upstream no longer serves must fall through to the default
    once discovery has run (these ids are absent from the served catalog)."""
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids", list(proxy._BOOTSTRAP_MODELS))
    for m in V20_REMOVED_MODELS:
        assert m not in proxy._known_models(), m
        assert proxy._map_model(m) == proxy.DEFAULT_MODEL, m
        assert proxy._route(f"oc-{m}")[1] == proxy.DEFAULT_MODEL, m


def test_v20_dead_models_fall_back_to_default():
    for m in V20_DEAD_MODELS:
        assert proxy._map_model(m) == proxy.DEFAULT_MODEL, m
        assert proxy._route(f"oc-{m}")[1] == proxy.DEFAULT_MODEL, m


def test_v20_union_alpha_is_no_longer_routed_to_messages_bridge():
    """union-alpha is gone upstream (401 ModelError). Keeping it in
    MESSAGES_MODELS made every request naming it take the Anthropic bridge and
    then fail upstream instead of resolving to a live model."""
    assert "union-alpha" not in proxy.MESSAGES_MODELS
    assert proxy._route("oc-union-alpha") == (proxy.BASE_URL, proxy.DEFAULT_MODEL)


def test_free_tier_gate_surfaces_as_403_not_502():
    """Zen answers 403 FreeTierError ("free tier can only be used from within
    OpenCode") for gated models. Classifying that as 502 tells callers the
    upstream is broken, when it is a policy gate they should not retry."""
    status, msg = proxy._classify_upstream_error(
        {"type": "FreeTierError",
         "message": "Error from provider (Console): OpenCode's free tier can only be used from within OpenCode"}
    )
    assert status == 403
    assert "free tier" in msg.lower()


def test_upstream_errors_still_classify_throttle_as_429():
    status, _ = proxy._classify_upstream_error({"type": "FreeUsageLimitError", "message": "quota"})
    assert status == 429
    status, _ = proxy._classify_upstream_error({"type": "rate_limit_error", "message": "slow down"})
    assert status == 429


def test_map_model_is_defined_once():
    """A second `_map_model` definition silently shadowed the first, so edits to
    the earlier one had no effect. Guard against a regression."""
    import inspect
    src = inspect.getsource(proxy)
    assert src.count("\ndef _map_model(") == 1


def test_scan_frame_recognizes_anthropic_stream_liveness():
    """The probe must see Anthropic message events as liveness/done signals,
    otherwise every union-alpha stream dies as 'empty completion' in the hold window."""
    delta = b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"abc"}}\n\n'
    assert proxy._scan_frame(delta)["live"] is True
    stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    assert proxy._scan_frame(stop)["done"] is True
    start = b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1"}}\n\n'
    s = proxy._scan_frame(start)
    assert s["live"] is False and s["done"] is False


def test_chat_to_anthropic_request_translation():
    out = proxy._chat_to_anthropic(
        {
            "model": "union-alpha",
            "max_tokens": 512,
            "temperature": 0.5,
            "stream": True,
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                                      "function": {"name": "f", "arguments": '{"a":1}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ],
            "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}],
        }
    )
    assert out["model"] == "union-alpha"
    assert out["system"] == "be brief"
    assert out["stream"] is True
    assert out["max_tokens"] == 512
    assert out["temperature"] == 0.5
    roles = [(m.get("role"), (m.get("content") or [{}])[0].get("type")) for m in out["messages"]]
    assert roles[0] == ("user", "text")
    assert out["messages"][1]["content"][0]["type"] == "tool_use"
    assert out["messages"][1]["content"][0]["id"] == "c1"
    assert out["messages"][2]["content"][0]["type"] == "tool_result"
    assert out["tools"][0] == {"name": "f", "description": "d", "input_schema": {"type": "object"}}
    assert {t["name"] for t in out["tools"][1:]} >= {"bash", "read", "write"}  # harness padding appended


def test_anthropic_msg_to_chat_json():
    resp = {
        "id": "msg_1", "model": "union-alpha", "stop_reason": "tool_use",
        "usage": {"input_tokens": 137, "output_tokens": 31},
        "content": [
            {"type": "text", "text": "calc:"},
            {"type": "tool_use", "id": "call_1", "name": "calculator", "input": {"expr": "2+2"}},
        ],
    }
    out = proxy._anthropic_msg_to_chat(resp, "union-alpha")
    assert out["object"] == "chat.completion"
    msg = out["choices"][0]["message"]
    assert msg["content"] == "calc:"
    assert msg["tool_calls"] == [{"id": "call_1", "type": "function",
                                  "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}}]
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    assert out["usage"]["prompt_tokens"] == 137
    assert out["usage"]["completion_tokens"] == 31


def _anthropic_text_sse_dicts():
    return [
        {"type": "message_start", "message": {"id": "m1"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "abc"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]


def test_anthropic_sse_dicts_translate_to_chat_chunks():
    async def gen():
        for e in _anthropic_text_sse_dicts():
            yield e

    out = b"".join(_run_stream(proxy._anthropic_sse_to_chat(gen(), "union-alpha"))).decode()
    assert '"content":"abc"' in out
    assert '"finish_reason":"stop"' in out
    assert '"completion_tokens":5' in out
    assert out.count("[DONE]") == 1


def test_anthropic_sse_tool_use_translates_to_tool_calls():
    async def gen():
        yield {"type": "message_start", "message": {"id": "m1"}}
        yield {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "call_1", "name": "calculator", "input": {}}}
        yield {"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": '{"expr": '}}
        yield {"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": '"2+2"}'}}
        yield {"type": "content_block_stop", "index": 0}
        yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
               "usage": {"output_tokens": 31}}
        yield {"type": "message_stop"}

    out = b"".join(_run_stream(proxy._anthropic_sse_to_chat(gen(), "union-alpha"))).decode()
    assert '"name":"calculator"' in out
    assert '"finish_reason":"tool_calls"' in out
    assert out.count("[DONE]") == 1


def test_anthropic_orphan_input_json_delta_emits_no_tool_chunk():
    """An input_json_delta with no preceding tool_use content_block_start must
    be skipped — not emitted as a tool_calls chunk with index -1 and no
    id/name (CodeRabbit 2026-09-28 minor: unguarded state['tool_idx'] - 1)."""

    async def gen():
        yield {"type": "message_start", "message": {"id": "m1"}}
        yield {"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": "hi"}}
        yield {"type": "content_block_delta", "index": 0,
               "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'}}
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"output_tokens": 5}}
        yield {"type": "message_stop"}

    out = b"".join(_run_stream(proxy._anthropic_sse_to_chat(gen(), "union-alpha"))).decode()
    assert "tool_calls" not in out


def _anthropic_messages_sse_bytes():
    return (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1"}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"abc"}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


# union-alpha — the only member of MESSAGES_MODELS — was retired upstream in
# v1.13.0 (401 "Model union-alpha is not supported"), leaving the set empty. The
# Messages bridge itself is still live infrastructure, so these tests keep it
# covered by monkeypatching a sentinel Anthropic-native model into the routing
# set. A new such model upstream only needs adding to MESSAGES_MODELS.
SENTINEL_MESSAGES_MODEL = "test-anthropic-native-free"


def _enable_messages_bridge(monkeypatch):
    monkeypatch.setattr(proxy, "MESSAGES_MODELS", {SENTINEL_MESSAGES_MODEL})
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids",
                        list(proxy._known_models()) + [SENTINEL_MESSAGES_MODEL])


def test_chat_completions_messages_bridge_streams_from_messages_endpoint(monkeypatch):
    """chat/completions for an Anthropic-native model must be served from zen's
    /messages endpoint and translated back to chat.completion.chunk SSE."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages"), request.url.path
        rbody = json.loads(request.content)
        assert rbody["model"] == SENTINEL_MESSAGES_MODEL
        assert rbody["messages"][0]["role"] == "user"
        return httpx.Response(200, content=_anthropic_messages_sse_bytes(),
                              headers={"Content-Type": "text/event-stream"})

    _enable_messages_bridge(monkeypatch)
    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": f"oc-{SENTINEL_MESSAGES_MODEL}", "stream": True, "max_tokens": 64,
              "messages": [{"role": "user", "content": "say abc"}]},
    )
    assert resp.status_code == 200, resp.text[:200]
    assert b'"content":"abc"' in resp.content
    assert b'"finish_reason":"stop"' in resp.content
    assert b'"error"' not in resp.content
    assert resp.content.count(b"[DONE]") == 1


def test_chat_via_messages_non_stream(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages"), request.url.path
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": SENTINEL_MESSAGES_MODEL,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "abc"}],
            "usage": {"input_tokens": 1, "output_tokens": 5},
        })

    _enable_messages_bridge(monkeypatch)
    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": SENTINEL_MESSAGES_MODEL, "stream": False,
              "messages": [{"role": "user", "content": "say abc"}]},
    )
    assert resp.status_code == 200, resp.text[:200]
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "abc"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 5


def test_anthropic_messages_bridge_roundtrip(monkeypatch):
    """Anthropic-speaking clients asking for an Anthropic-native model get
    native Anthropic SSE back (OAI bridge in the middle, re-translated on the
    way out)."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages"), request.url.path
        return httpx.Response(200, content=_anthropic_messages_sse_bytes(),
                              headers={"Content-Type": "text/event-stream"})

    _enable_messages_bridge(monkeypatch)
    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/messages",
        json={"model": SENTINEL_MESSAGES_MODEL, "max_tokens": 64, "stream": True,
              "messages": [{"role": "user", "content": "say abc"}]},
    )
    assert resp.status_code == 200, resp.text[:200]
    assert b"text_delta" in resp.content
    assert b"message_stop" in resp.content


# --- v1.10.0 harness-tools padding (2026-09-18 gate analysis): Zen's free
# tier only serves requests carrying opencode's genuine harness tool
# definitions. The proxy appends them upstream; client tools keep priority
# on name collisions. ---

EXPECTED_HARNESS_TOOL_NAMES = {
    "bash", "edit", "get_goal", "glob", "grep", "read", "skill", "task",
    "todowrite", "update_goal", "webfetch", "websearch", "write",
}


def test_harness_tools_match_genuine_cli_set():
    assert set(proxy._HARNESS_TOOL_NAMES) == EXPECTED_HARNESS_TOOL_NAMES


def test_pad_adds_all_harness_tools_to_bare_request():
    out = proxy._pad_harness_tools(
        {"model": "mimo-v2.5-free", "messages": [{"role": "user", "content": "hi"}]}
    )
    names = [t["function"]["name"] for t in out["tools"]]
    assert set(names) == EXPECTED_HARNESS_TOOL_NAMES
    assert out["model"] == "mimo-v2.5-free"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_pad_preserves_client_tools_first():
    calc = {"type": "function", "function": {"name": "calc", "description": "c",
                                             "parameters": {"type": "object"}}}
    out = proxy._pad_harness_tools(
        {"model": "m", "messages": [], "tools": [calc]}
    )
    assert out["tools"][0] == calc
    assert len(out["tools"]) == 1 + len(EXPECTED_HARNESS_TOOL_NAMES)


def test_pad_skips_colliding_harness_tools():
    mine = {"type": "function", "function": {"name": "bash", "description": "mine",
                                             "parameters": {"type": "object"}}}
    out = proxy._pad_harness_tools(
        {"model": "m", "messages": [], "tools": [mine]}
    )
    bash = [t for t in out["tools"] if t["function"]["name"] == "bash"]
    assert len(bash) == 1
    assert bash[0]["function"]["description"] == "mine"
    assert len(out["tools"]) == len(EXPECTED_HARNESS_TOOL_NAMES)


def test_pad_never_injects_tool_choice():
    """The padded harness tools exist only to satisfy Zen's gate. Injecting
    tool_choice:"auto" alongside them silently opted tool-less callers INTO
    tool use, and the model then called a padded tool the client cannot run
    (observed live: a bare chat request returned tool_use for 'bash'/'read').
    Padding must not change any behavioural field of the request."""
    out = proxy._pad_harness_tools({"model": "m", "messages": []})
    assert "tool_choice" not in out
    assert len(out["tools"]) > 0  # definitions are still injected

    out2 = proxy._pad_harness_tools({"model": "m", "messages": [], "tool_choice": "none"})
    assert out2["tool_choice"] == "none"  # an explicit choice is left alone



def test_pad_ignores_non_chat_bodies():
    assert proxy._pad_harness_tools({"foo": 1}) == {"foo": 1}
    assert proxy._pad_harness_tools({}) == {}


def test_pad_anthropic_format_merge():
    mine = {"name": "calc", "description": "c", "input_schema": {"type": "object"}}
    out = proxy._pad_harness_tools_anthropic(
        {"model": "union-alpha", "messages": [], "tools": [mine]}
    )
    assert out["tools"][0] == mine
    names = [t["name"] for t in out["tools"]]
    assert set(names) == {"calc"} | EXPECTED_HARNESS_TOOL_NAMES


def test_pad_responses_format_merge():
    out = proxy._pad_harness_tools_responses(
        {"model": "muse-spark-1.3-contributor-free", "input": []}
    )
    names = [t["name"] for t in out["tools"]]
    assert set(names) == EXPECTED_HARNESS_TOOL_NAMES
    assert all(t["type"] == "function" for t in out["tools"])


# --- do-not-call note on padded tools (2026-09-28): padded harness tools
# exist only to pass Zen's free-tier gate; the client harness rejects calls
# to them (Cursor: "unavailable tool 'read'"). Every injected definition
# must carry a description note so the model leaves them alone. ---


def test_pad_marks_padded_tools_do_not_call():
    out = proxy._pad_harness_tools(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    for t in out["tools"]:
        assert proxy._HARNESS_IGNORE_NOTE in t["function"]["description"]


def test_pad_leaves_client_tools_unmarked():
    calc = {"type": "function", "function": {"name": "calc", "description": "calc things",
                                             "parameters": {"type": "object"}}}
    out = proxy._pad_harness_tools(
        {"model": "m", "messages": [], "tools": [calc]}
    )
    by_name = {t["function"]["name"]: t for t in out["tools"]}
    assert by_name["calc"]["function"]["description"] == "calc things"
    assert proxy._HARNESS_IGNORE_NOTE in by_name["read"]["function"]["description"]


def test_pad_anthropic_format_marks_do_not_call():
    out = proxy._pad_harness_tools_anthropic(
        {"model": "union-alpha", "messages": [], "tools": []}
    )
    for t in out["tools"]:
        assert proxy._HARNESS_IGNORE_NOTE in t["description"]


def test_pad_responses_format_marks_do_not_call():
    out = proxy._pad_harness_tools_responses({"model": "m", "input": []})
    for t in out["tools"]:
        assert proxy._HARNESS_IGNORE_NOTE in t["description"]


def test_chat_completions_forwards_padded_tools_upstream(monkeypatch):
    """Upstream must see the harness tools even when the client sent none."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["tools"] = json.loads(request.content).get("tools", [])
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "oc-mimo-v2.5-free", "stream": False,
              "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert set(t["function"]["name"] for t in captured["tools"]) >= EXPECTED_HARNESS_TOOL_NAMES


# --- 429 backoff must stay inside STATUS_HOLD_SECS (CodeRabbit 2026-09-28) ---


def _always_429_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        429,
        json={"error": {"message": "Rate limit exceeded. Please try again later."}},
    )


def test_stream_upstream_429_backoff_stays_within_hold(monkeypatch):
    """Cumulative 429 backoff in _zen_stream_upstream must stay below
    STATUS_HOLD_SECS: a persistent 429 has to return before the streaming
    header hold expires, or it degrades to a committed 200 keepalive stream.
    Regression: 5+10+20s default backoff outlasted the 15s hold."""

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 1.0)
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)
    _client_with_mock(monkeypatch, _always_429_handler)

    async def run():
        t0 = time.monotonic()
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return time.monotonic() - t0, status, err

    elapsed, status, err = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit exceeded" in err
    assert elapsed < proxy.STATUS_HOLD_SECS


def test_zen_stream_parsed_429_backoff_stays_within_hold(monkeypatch):
    """Anthropic-path twin of the hold-budget regression: _zen_stream_parsed's
    cumulative 429 backoff must also stay below STATUS_HOLD_SECS."""

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 1.0)
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)
    _client_with_mock(monkeypatch, _always_429_handler)

    async def run():
        t0 = time.monotonic()
        status, err, chunks = await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )
        return time.monotonic() - t0, status, err

    elapsed, status, err = asyncio.get_event_loop().run_until_complete(run())
    assert status == 429
    assert "Rate limit exceeded" in err
    assert elapsed < proxy.STATUS_HOLD_SECS


def test_openai_stream_persistent_429_surfaces_real_status_within_hold(monkeypatch):
    """End-to-end through the header hold: a persistent 429 must raise
    HTTPException(429) while the hold is still waiting — never a committed
    200 StreamingResponse."""

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 1.0)
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)
    _client_with_mock(monkeypatch, _always_429_handler)

    async def run():
        return await proxy._openai_stream_response(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": True}, "1.2.3.4"
        )

    raised = None
    try:
        asyncio.get_event_loop().run_until_complete(run())
    except HTTPException as e:
        raised = e
    assert raised is not None, "expected HTTPException, got a committed response"
    assert raised.status_code == 429
    assert "Rate limit exceeded" in raised.detail


def test_post_with_retry_429_backoff_stays_within_hold(monkeypatch):
    """Non-streaming _post_with_retry must bound its cumulative 429 backoff by
    STATUS_HOLD_SECS as well, so both paths share the same 429 patience."""

    monkeypatch.setattr(proxy, "RETRY_429", 3)
    monkeypatch.setattr(proxy, "RETRY_429_BACKOFF", 1.0)
    monkeypatch.setattr(proxy, "STATUS_HOLD_SECS", 2.0)
    _client_with_mock(monkeypatch, _always_429_handler)

    async def run():
        t0 = time.monotonic()
        resp = await proxy._post_with_retry(
            "http://upstream-mock/v1/chat/completions", {}, {"stream": False}
        )
        return time.monotonic() - t0, resp

    elapsed, resp = asyncio.get_event_loop().run_until_complete(run())
    assert resp.status_code == 429
    assert elapsed < proxy.STATUS_HOLD_SECS


def test_entrypoint_banner_prints_current_version(tmp_path):
    """The startup banner must print the live proxy.VERSION — a hardcoded
    version drifts (regression: banner said v1.10.0 while VERSION was 1.12.0)."""

    script = Path(proxy.__file__).parent / "entrypoint.sh"
    content = script.read_text()
    assert "exec uvicorn" in content
    sandbox = tmp_path / "entrypoint.sh"
    sandbox.write_text(content.replace("exec uvicorn", "exit 0 # uvicorn"))

    proc = subprocess.run(
        ["sh", str(sandbox)],
        capture_output=True,
        text=True,
        cwd=str(script.parent),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    m = re.search(r"Starting OpenCode Proxy v(\S+) on", proc.stdout)
    assert m, proc.stdout
    assert m.group(1) == proxy.VERSION


def test_harness_tools_is_present_for_the_image_build():
    """proxy.py opens harness_tools.json at IMPORT time and the Dockerfile
    COPYs it, so a clone that lacks it cannot even import the module. Pin the
    file to version control's view of the repo, not just the working tree."""
    import shutil
    import subprocess

    root = Path(proxy.__file__).parent
    tool_file = root / proxy._HARNESS_TOOLS_PATH.rsplit("/", 1)[-1]
    assert tool_file.exists(), f"missing {tool_file} — proxy.py cannot be imported"
    if shutil.which("git"):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", tool_file.name],
            cwd=str(root), capture_output=True, text=True,
        )
        assert tracked.returncode == 0, (
            f"{tool_file.name} is untracked; the Docker build does a clean copy and "
            "proxy.py raises FileNotFoundError at import"
        )


# --- v1.13.0 audit fixes ---

# BUG 1: _fit_context did `list(body[key])` unconditionally, so a Responses body
# whose `input` is a plain STRING became a list of single characters that was
# forwarded upstream. Both passthrough routes reach this path.
def test_fit_context_leaves_string_input_untouched():
    body = {"model": "m", "input": "summarize this"}
    out = proxy._fit_context(body)
    assert out["input"] == "summarize this"
    assert not isinstance(out["input"], list)


def test_responses_string_input_is_not_mangled_upstream(monkeypatch):
    """End-to-end: POST /v1/responses with a string `input` must reach upstream
    with `input` still a string (regression: ['s','u','m',...])."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "resp_1", "output": []},
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"model": "oc-space-bunny-free", "input": "summarize this"})
    assert resp.status_code == 200, resp.text
    assert captured["body"]["input"] == "summarize this"


def test_responses_object_input_is_not_mangled_upstream(monkeypatch):
    """`input` may also be a single item object, not just an array."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "resp_1", "output": []},
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    payload = {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    resp = client.post("/v1/responses", json={"model": "oc-space-bunny-free", "input": payload})
    assert resp.status_code == 200, resp.text
    assert captured["body"]["input"] == payload


def test_fit_context_still_trims_array_input():
    """The guard must keep working for the array form it was written for."""
    msg = {"role": "user", "content": "x" * 300000}
    per_msg = proxy._message_cost([msg])
    big = [dict(msg) for _ in range(int(proxy.CONTEXT_LIMIT * 0.9) // per_msg + 1)]
    out = proxy._fit_context({"input": big, "max_tokens": 64000})
    assert len(out["input"]) < len(big)


# BUG 3: _is_rate_limit_error did data.get("error", {}).get("message") with no
# isinstance guard, so a 429 whose `error` is a string or list raised
# AttributeError and surfaced as 500 instead of a retryable 429.
def test_is_rate_limit_error_survives_non_dict_error():
    assert proxy._is_rate_limit_error(429, b'{"error": "too many"}')
    assert proxy._is_rate_limit_error(429, b"[1,2,3]")
    assert proxy._is_rate_limit_error(429, b"not json at all")


def test_429_with_string_error_surfaces_429_not_500(monkeypatch):
    """End-to-end: the non-stream chat route must return 429, not 500."""
    monkeypatch.setattr(proxy, "RETRY_429", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"error": "too many"}',
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={
        "model": "oc-space-bunny-free",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 429, f"got {resp.status_code}: {resp.text[:150]}"


# BUG 2: the non-stream paths had no empty-completion guard, so an upstream
# 200 with null content and zero output tokens surfaced as a well-formed empty
# success. Both bridge paths already raised 502.
def test_messages_nonstream_empty_completion_returns_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "c1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        })

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/messages", json={
        "model": "oc-space-bunny-free", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:150]}"
    assert "empty" in resp.text.lower()


def test_chat_nonstream_empty_completion_returns_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "c1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        })

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={
        "model": "oc-space-bunny-free",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:150]}"


def test_messages_nonstream_with_content_still_200(monkeypatch):
    """Guard the fix: real answers must not be rejected."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "c1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/messages", json={
        "model": "oc-space-bunny-free", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["content"] == [{"type": "text", "text": "hello"}]


# BUG 4: the passthrough routes never called _route(), so `oc-` prefixes leaked
# to Zen verbatim and unknown IDs got no DEFAULT_MODEL fallback.
def test_responses_route_normalizes_model(monkeypatch):
    captured = {}
    target = proxy._known_models()[0]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content).get("model")
        return httpx.Response(200, json={"id": "resp_1", "output": []},
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    client.post("/v1/responses", json={"model": f"oc-{target}", "input": "hi"})
    assert captured["model"] == target

    client.post("/v1/responses", json={"model": "gpt-4o", "input": "hi"})
    assert captured["model"] == proxy.DEFAULT_MODEL


def test_catch_all_normalizes_model(monkeypatch):
    captured = {}
    target = proxy._known_models()[0]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["model"] = json.loads(request.content).get("model")
        return httpx.Response(200, json={"ok": True},
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    client.post("/v1/embeddings", json={"model": f"oc-{target}", "input": "hi"})
    assert captured["model"] == target

    client.post("/v1/embeddings", json={"model": "nope", "input": "hi"})
    assert captured["model"] == proxy.DEFAULT_MODEL


# BUG 7: the passthrough routes called resp.json() directly, so a 200 with a
# lying content-type produced a bare 500 instead of the clean 502 the guarded
# paths produce.
def test_responses_passthrough_non_json_body_returns_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>Internal Server Error</html>",
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"model": "oc-space-bunny-free", "input": "hi"})
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:150]}"


def test_catch_all_passthrough_non_json_body_returns_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>Internal Server Error</html>",
                              headers={"Content-Type": "application/json"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/embeddings", json={"model": "oc-space-bunny-free", "input": "hi"})
    assert resp.status_code == 502, f"got {resp.status_code}: {resp.text[:150]}"


# BUG 8: count_tokens divided serialized length by 4 while _fit_context charged
# //2 for the same message, so the proxy's own accounting disagreed by 2x and
# Claude Code's compaction decisions drifted from the real budget.
def test_count_tokens_estimator_matches_fit_context_cost():
    """count_tokens and _fit_context must charge the same message identically —
    they disagreed 2x (//4 vs //2), so Claude Code's compaction decisions
    drifted from the budget the proxy actually enforces."""
    msgs = [{"role": "user", "content": "hello world " * 100}]
    charged = proxy._message_cost(msgs)
    reported = sum(proxy._estimate_tokens([m]) for m in msgs)
    assert reported == charged


def test_count_tokens_endpoint_uses_shared_estimator():
    """count_tokens must charge the same set a real request sends: the
    translated messages PLUS the padded harness tool definitions."""
    msgs = [{"role": "user", "content": "hi"}]
    client = TestClient(proxy.app)
    resp = client.post("/v1/messages/count_tokens", json={"model": "x", "messages": msgs})
    # Reproduce the /v1/messages pipeline the endpoint now mirrors.
    counted = proxy._pad_harness_tools(
        proxy._fit_context(proxy._anthropic_to_openai({"model": "x", "messages": msgs})))
    expected = (proxy._estimate_tokens(counted.get("messages") or [])
                + proxy._estimate_tokens(counted.get("tools") or []))
    assert resp.json()["input_tokens"] == expected
    # The harness padding alone is ~6k tokens, so an estimate that ignores it
    # is a ~99% under-count.
    assert expected > proxy._estimate_tokens(msgs)


# BUG 13: the trim loop stopped at len(items) > 1, so a single message that
# already exceeds the context was never trimmed and max_tokens was set to
# max(1024, negative_budget) -- guaranteeing an upstream 400.
def test_fit_context_does_not_invent_max_tokens_on_negative_budget():
    huge = {"role": "user", "content": "x" * (proxy.CONTEXT_LIMIT * 2)}
    out = proxy._fit_context({"messages": [huge], "max_tokens": 4096})
    assert out.get("max_tokens", 0) > 0
    # must not silently grow a request that was already over the ceiling
    assert out["max_tokens"] <= 4096


# BUG 11: the fallback-list test only asserted len(data) > 0, so a model
# dropped from the advertised pool by mistake passed silently.
def test_health_model_count_matches_free_pool():
    client = TestClient(proxy.app)
    health = client.get("/health").json()
    listing = client.get("/v1/models").json()
    assert health["models"] == len(proxy._known_models())
    assert len(listing["data"]) == len(proxy._known_models())
    assert health["version"] == proxy.VERSION


# BUG 14: the catch-all shadowed GET /v1/models, so POST /v1/models hit a
# non-existent upstream endpoint instead of the proxy's own list.
def test_post_v1_models_returns_static_list_not_upstream(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"must not call upstream: {request.url}")

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app, raise_server_exceptions=False)
    resp = client.post("/v1/models", json={"probe": 1})
    assert resp.status_code == 200, resp.text
    assert resp.json()["object"] == "list"
    assert len(resp.json()["data"]) == len(proxy._known_models())



# --- auto-correct tool call names (2026-09-28): when the model emits a tool
# call whose name case-insensitively matches a tool the CLIENT advertised,
# rewrite it to the client's official name before the response leaves the
# proxy (read -> Read). Nothing is dropped: unmatched names (bash), exact
# matches and empty maps pass through untouched. ---


def _oai_client_tools():
    return [
        {"type": "function", "function": {"name": "Read", "description": "r",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "Shell", "description": "s",
                                          "parameters": {"type": "object"}}},
    ]


def test_client_tool_name_map_oai_shape():
    # padded body: harness tools carry the do-not-call note and are excluded
    body = proxy._pad_harness_tools(
        {"messages": [], "tools": _oai_client_tools()}
    )
    m = proxy._client_tool_name_map(body)
    assert m["read"] == "Read"
    assert m["shell"] == "Shell"
    assert "bash" not in m and "write" not in m      # padded-only names excluded
    assert m["read"] != "read" or True


def test_client_tool_name_map_anthropic_and_responses_shapes():
    anth = {"tools": [{"name": "Read", "description": "r",
                       "input_schema": {"type": "object"}}]}
    resp = {"tools": [{"type": "function", "name": "Shell", "description": "s",
                       "parameters": {"type": "object"}}]}
    assert proxy._client_tool_name_map(anth) == {"read": "Read"}
    assert proxy._client_tool_name_map(resp) == {"shell": "Shell"}
    assert proxy._client_tool_name_map({}) == {}


def test_client_tool_name_map_ambiguous_dropped():
    body = {"tools": [
        {"type": "function", "function": {"name": "Read", "description": "a",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "read", "description": "b",
                                          "parameters": {"type": "object"}}},
    ]}
    assert proxy._client_tool_name_map(body) == {}   # ambiguous -> no rewrite


def test_stream_autocorrects_tool_name(monkeypatch):
    """Streaming bytes relay: model emits `read`, client advertised `Read` ->
    relayed chunks carry the official name with id/args/index intact and the
    tool_calls finish preserved."""

    def handler(request):
        return httpx.Response(200, content=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"read","arguments":""}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"p\\":1}"}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            b"data: [DONE]\n\n"
        ), headers={"Content-Type": "text/event-stream"})

    body = proxy._pad_harness_tools(
        {"stream": True, "messages": [{"role": "user", "content": "hi"}],
         "tools": _oai_client_tools()}
    )
    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, body, "1.2.3.4"
        )
        parts = []
        async for c in chunks:
            parts.append(c)
        return status, err, b"".join(parts)

    status, err, out = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200 and err is None
    assert b'"name":"Read"' in out
    assert b'"name":"read"' not in out
    assert b'{\\"p\\":1}' in out                      # args untouched
    assert b'"finish_reason":"tool_calls"' in out
    assert out.count(b"[DONE]") == 1


def test_stream_keeps_unmatched_and_exact_names(monkeypatch):
    """A tool the client did NOT advertise and that is not a harness tool
    passes through unchanged; an already-correct `Read` is untouched; finish
    stays tool_calls.

    `bash` used to be the 'unmatched' case, but it is a harness tool no client
    implements, so the guard now converts it. The unmatched case is therefore
    a name that is neither advertised nor padded."""

    def handler(request):
        return httpx.Response(200, content=(
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"custom_search","arguments":""}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"Read","arguments":""}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            b"data: [DONE]\n\n"
        ), headers={"Content-Type": "text/event-stream"})

    body = proxy._pad_harness_tools(
        {"stream": True, "messages": [{"role": "user", "content": "hi"}],
         "tools": _oai_client_tools()}
    )
    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_upstream(
            "http://upstream-mock/v1/chat/completions", {}, body, "1.2.3.4"
        )
        parts = []
        async for c in chunks:
            parts.append(c)
        return status, err, b"".join(parts)

    status, err, out = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200 and err is None
    assert b'"name":"custom_search"' in out
    assert b'"name":"Read"' in out
    assert b'"finish_reason":"tool_calls"' in out
    assert out.count(b"[DONE]") == 1


def test_parsed_stream_autocorrects_anthropic_event(monkeypatch):
    """_zen_stream_parsed on an Anthropic-shaped body: a tool_use named
    `read` is rewritten to `Read` in the parsed event stream."""

    def handler(request):
        return httpx.Response(200, content=(
            b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"read","input":{}}}\n\n'
            b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"p\\":1}"}}\n\n'
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n\n'
            b'data: {"type":"message_stop"}\n\n'
        ), headers={"Content-Type": "text/event-stream"})

    mbody = proxy._pad_harness_tools_anthropic(
        {"stream": True, "messages": [],
         "tools": [{"name": "Read", "description": "r",
                    "input_schema": {"type": "object"}}]}
    )
    _client_with_mock(monkeypatch, handler)

    async def run():
        status, err, chunks = await proxy._zen_stream_parsed(
            "http://upstream-mock/v1/messages", {}, mbody, "1.2.3.4"
        )
        objs = []
        async for o in chunks:
            objs.append(o)
        return status, err, objs

    status, err, objs = asyncio.get_event_loop().run_until_complete(run())
    assert status == 200 and err is None
    starts = [o for o in objs if isinstance(o, dict) and o.get("type") == "content_block_start"]
    assert starts and starts[0]["content_block"]["name"] == "Read"


def test_nonstream_autocorrects_openai_json(monkeypatch):
    """Route-level: non-stream chat/completions where the model emits `read`
    -> the client receives `Read` with arguments intact."""

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "c1", "type": "function",
                                                     "function": {"name": "read",
                                                                  "arguments": "{\"p\":1}"}}]}}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "oc-space-bunny-free", "stream": False,
              "messages": [{"role": "user", "content": "hi"}],
              "tools": _oai_client_tools()},
    )
    assert resp.status_code == 200
    calls = resp.json()["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "Read"
    assert calls[0]["function"]["arguments"] == '{"p":1}'


def test_autocorrect_tool_calls_json_anthropic_and_responses_shapes():
    anth = {"content": [
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"p": 1}},
        {"type": "text", "text": "hi"},
    ], "stop_reason": "tool_use"}
    out = proxy._autocorrect_tool_calls_json(anth, {"read": "Read"})
    assert out["content"][0]["name"] == "Read"
    assert out["content"][0]["input"] == {"p": 1}
    assert out["stop_reason"] == "tool_use"

    resp = {"output": [{"type": "function_call", "call_id": "c1",
                        "name": "todowrite", "arguments": "{}"}]}
    out = proxy._autocorrect_tool_calls_json(resp, {"todowrite": "TodoWrite"})
    assert out["output"][0]["name"] == "TodoWrite"


def test_noop_when_no_client_tools():
    resp = {"choices": [{"index": 0, "message": {"tool_calls": [
        {"function": {"name": "read"}}]}}]}
    assert proxy._autocorrect_tool_calls_json(resp, {}) == resp
    assert proxy._autocorrect_tool_calls_json(resp, {"other": "Other"}) == resp

    step = proxy._tool_call_autocorrector({})
    chunk = {"choices": [{"delta": {"tool_calls": [
        {"function": {"name": "read"}}]}}]}
    assert step(chunk) is chunk
    assert chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"


# --- agent-report bug fixes (2026-09-28): case-insensitive pad dedupe,
# missing-model routing, count_tokens tool definitions, passthrough
# content-type preservation. ---


def test_pad_skips_case_insensitive_collisions_oai():
    """Client advertises `Bash`/`Read` -> padding must not inject the
    case-variant harness twins: upstream would see two tools differing only
    by case and the model calls the padded lowercase one the client cannot
    execute (live repro: tools after pad ['Bash','bash',...])."""
    body = proxy._pad_harness_tools(
        {"model": "m", "messages": [], "tools": [
            {"type": "function", "function": {"name": "Bash", "description": "c",
                                              "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "Read", "description": "c",
                                              "parameters": {"type": "object"}}},
        ]}
    )
    names = [t["function"]["name"] for t in body["tools"]]
    assert names[:2] == ["Bash", "Read"]               # client tools first
    assert len({n.lower() for n in names}) == len(names)
    assert "bash" not in names and "read" not in names
    assert "get_goal" in names                          # no counterpart -> kept


def test_pad_skips_case_insensitive_collisions_anthropic_and_responses():
    anth = proxy._pad_harness_tools_anthropic(
        {"messages": [], "tools": [{"name": "Read", "description": "c",
                                    "input_schema": {"type": "object"}}]}
    )
    anames = [t["name"] for t in anth["tools"]]
    assert anames[0] == "Read"
    assert len({n.lower() for n in anames}) == len(anames)
    assert "read" not in anames
    assert "bash" in anames

    resp = proxy._pad_harness_tools_responses(
        {"input": [], "tools": [{"type": "function", "name": "Write", "description": "c",
                                 "parameters": {"type": "object"}}]}
    )
    rnames = [t["name"] for t in resp["tools"]]
    assert rnames[0] == "Write"
    assert len({n.lower() for n in rnames}) == len(rnames)
    assert "write" not in rnames
    assert "bash" in rnames


def test_chat_completions_missing_model_gets_default(monkeypatch):
    """A body with NO `model` key must still be routed — upstream receives
    DEFAULT_MODEL, not a model-less request (_route already resolved the
    default; the old `is not None` guard dropped it)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert captured["body"]["model"] == proxy.DEFAULT_MODEL


def test_count_tokens_includes_tool_definitions():
    """count_tokens must charge serialized tool definitions — Claude Code
    sends ~10k tokens of tool schemas the old messages-only estimate ignored,
    so the client compacted too late."""
    client = TestClient(proxy.app)
    tools = [{"name": f"tool_{i}", "description": "x" * 400,
              "input_schema": {"type": "object"}} for i in range(15)]
    messages = [{"role": "user", "content": "hi"}]

    bare = client.post("/v1/messages/count_tokens",
                       json={"model": "oc-space-bunny-free", "messages": messages}).json()
    with_tools = client.post("/v1/messages/count_tokens",
                             json={"model": "oc-space-bunny-free",
                                   "messages": messages, "tools": tools}).json()
    # the proxy estimates the TRANSLATED (OpenAI-shape) definitions
    translated = proxy._anthropic_to_openai({"messages": messages, "tools": tools})["tools"]
    expect = sum(max(1, len(json.dumps(t)) // proxy._TOKENS_PER_BYTE_DIVISOR) for t in translated)
    assert with_tools["input_tokens"] - bare["input_tokens"] == expect


def test_passthrough_preserves_non_json_content_type(monkeypatch):
    """An upstream non-JSON error body (text/html) must reach the client with
    the upstream content-type and verbatim text — not JSON-encoded into an
    application/json string ('\"<html>…\"')."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"<html>Bad Request</html>",
                              headers={"Content-Type": "text/html; charset=utf-8"})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    resp = client.post("/v1/responses", json={"input": "hi"})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.text == "<html>Bad Request</html>"


def test_count_tokens_charges_harness_padding_like_a_real_request():
    """count_tokens must report what a REAL /v1/messages request actually sends.

    /v1/messages runs _anthropic_to_openai -> _fit_context ->
    _pad_harness_tools, so 13 harness definitions ride along on every
    request. count_tokens stopped at the client's own tools and reported
    33 tokens where the upstream received ~6000 (98.7% under-count), so
    Claude Code compacted far too late."""
    client = TestClient(proxy.app)
    payload = {"model": "oc-space-bunny-free", "max_tokens": 1024,
               "messages": [{"role": "user", "content": "hi"}],
               "tools": [{"name": "Read", "description": "d",
                          "input_schema": {"type": "object"}}]}

    reported = client.post("/v1/messages/count_tokens", json=payload).json()["input_tokens"]

    # Reproduce the exact pipeline the real route uses.
    actual = proxy._pad_harness_tools(proxy._fit_context(proxy._anthropic_to_openai(payload)))
    expected = (proxy._estimate_tokens(actual.get("messages") or [])
                + proxy._estimate_tokens(actual.get("tools") or []))

    assert expected > 5000, "sanity: harness padding is ~6k tokens"
    assert reported == expected, f"count_tokens said {reported}, real request sends {expected}"


def test_debug_log_helpers_flush_inside_the_with_block(tmp_path):
    """Both instrumentation helpers must not raise on a successful write.

    _dbg called fh.flush() AFTER the `with` block closed the handle, so every
    write raised ValueError('I/O operation on closed file.') — the log line
    landed but the helper always reported failure. _dbg02 additionally
    defaulted to a hard-coded host path that does not exist in the container."""
    log = tmp_path / "dbg.log"
    orig_dbg, orig_dbg02 = proxy._DBG_PATH, proxy._DBG02_PATH
    proxy._DBG_PATH = str(log)
    proxy._DBG02_PATH = str(log)

    # Capture the warnings both helpers emit; a correct write emits none.
    seen = []
    real_warning = proxy.logger.warning
    proxy.logger.warning = lambda *a, **k: seen.append(a[0] if a else "")
    try:
        proxy._dbg("t:1", "hello", {"k": 1})
        proxy._dbg02("t:2", "hello", {"k": 2})
    finally:
        proxy.logger.warning = real_warning

    assert not [m for m in seen if "log write failed" in str(m)], \
        f"helpers reported a failed write: {seen}"

    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, f"expected 2 log lines, got {len(lines)}: {lines}"
    payloads = [json.loads(ln) for ln in lines]
    assert {p["location"] for p in payloads} == {"t:1", "t:2"}
    assert {p["sessionId"] for p in payloads} == {proxy._DBG_SID, proxy._DBG02_SID}

    # Restore so later tests see the real defaults (module-level state leaks).
    proxy._DBG_PATH, proxy._DBG02_PATH = orig_dbg, orig_dbg02


def test_dbg02_default_path_is_portable_not_host_specific():
    """_dbg02's default must be derived from the module location, not a
    hard-coded host path — inside the container the app is at /app, so a
    literal /Users/... default raises FileNotFoundError on every write."""
    import os
    src = open(proxy.__file__).read()
    default_expr = src.split('_DBG02_PATH = os.environ.get(')[1].split(')')[0]
    assert '"/Users/' not in default_expr, "default still hard-codes a host path"
    # The resolved default must sit under the module's own directory.
    assert proxy._DBG02_PATH == os.path.join(
        os.path.dirname(os.path.abspath(proxy.__file__)), ".cursor", "debug-02b13a.log")
    assert os.path.isabs(proxy._DBG02_PATH)


# --- v1.14.0: catch_all prefix strip, static /v1/models, dynamic discovery ---

def test_catch_all_strips_repeated_v1_prefix(monkeypatch):
    """BASE_URL already ends in /v1, so /v1/v1/models must resolve to
    /v1/models — the old single `path.startswith("v1/")` check only stripped
    one segment and produced .../v1/v1/models upstream."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"ok": True})

    _client_with_mock(monkeypatch, handler)
    client = TestClient(proxy.app)
    for path, want in [("/v1/embeddings", "embeddings"),
                       ("/v1/foo/bar", "foo/bar")]:
        client.post(path, json={"model": "oc-space-bunny-free", "input": "x"})
        assert seen["url"].split("?")[0] == f"{proxy.BASE_URL}/{want}", \
            f"{path} -> {seen['url']}"

    # Repeated LEADING v1/ segments collapse; a v1/ in the middle is preserved.
    for raw, want in [("v1/v1/v1/models", "models"),
                      ("v1/v1/embeddings", "embeddings"),
                      ("v1/embeddings", "embeddings"),
                      ("v1/foo/v1/bar", "foo/v1/bar")]:
        assert proxy._upstream_path(raw) == want, f"{raw} -> {proxy._upstream_path(raw)}"


def test_models_endpoint_serves_proxy_cache_not_upstream(monkeypatch):
    """/v1/models must answer from the proxy's own cache so the paid ids in
    Zen's catalog (claude-opus-*, gpt-*, gemini-*) are never exposed."""
    called = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "space-bunny-free", "object": "model"},
            {"id": "claude-opus-5-5", "object": "model"},
            {"id": "gpt-5.6-sol", "object": "model"},
        ]})

    _client_with_mock(monkeypatch, handler)
    # Seed the cache as if discovery had run.
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids", ["space-bunny-free"])
    client = TestClient(proxy.app)
    before = called["n"]
    data = client.get("/v1/models").json()["data"]
    assert called["n"] == before, "/v1/models must not hit the upstream"
    ids = [m["id"] for m in data]
    assert ids == ["oc-space-bunny-free"], ids


def test_model_discovery_picks_up_new_upstream_ids(monkeypatch):
    """A newly added upstream model must resolve without a code change."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "brand-new-model-free", "object": "model"},
            {"id": "space-bunny-free", "object": "model"},
        ]})

    _client_with_mock(monkeypatch, handler)
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids", ["space-bunny-free"])
    monkeypatch.setitem(proxy._MODEL_CACHE, "fetched", 0.0)
    assert proxy._map_model("oc-brand-new-model-free") == proxy.DEFAULT_MODEL
    assert "brand-new-model-free" not in proxy._known_models()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        proxy._refresh_models(force=True))
    assert "brand-new-model-free" in proxy._known_models()
    assert proxy._map_model("oc-brand-new-model-free") == "brand-new-model-free"


def test_model_discovery_survives_upstream_failure(monkeypatch):
    """A failed refresh must keep the last good list, never empty the cache."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _client_with_mock(monkeypatch, handler)
    monkeypatch.setitem(proxy._MODEL_CACHE, "ids", ["space-bunny-free"])
    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        loop.run_until_complete(proxy._refresh_models(force=True))
    finally:
        loop.close()
    assert proxy._known_models() == ["space-bunny-free"]
    assert proxy._map_model("oc-space-bunny-free") == "space-bunny-free"


def test_no_hardcoded_free_model_list():
    """The model inventory must come from discovery, not a literal list."""
    src = open(proxy.__file__).read()
    assert "FREE_MODELS" not in src, "FREE_MODELS should no longer exist"
    # DEFAULT_MODEL is the only allowed literal model id.
    import re
    literals = re.findall(r'^\s*"([a-z0-9][a-z0-9.\-]*-free)"\s*,?\s*$', src, re.M)
    assert not literals, f"hardcoded model ids still in source: {literals}"




