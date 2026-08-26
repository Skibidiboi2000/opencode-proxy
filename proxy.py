"""OpenCode Proxy - OpenAI & Anthropic compatible proxy for OpenCode Zen free tier."""

import asyncio
import json
import logging
import os
import secrets
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("opencode-proxy")

VERSION = "1.5.0"
OC_VERSION = "1.18.21"
BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
BROKE_MODE = os.environ.get("OPENCODE_BROKE", "").lower() in ("1", "true", "yes")

# Hold the HTTP response long enough to learn the upstream status before
# committing 200+SSE to the client. Failures arriving faster than this window
# surface as real HTTP statuses (callers like 9Router can then lock/fallback);
# slower upstreams fall back to 200 + keepalive streaming as before. Must stay
# below the shortest downstream first-byte timeout (~25s on 9Router) AFTER the
# retry window: 7s worst-case backoff + 15s hold ≈ 22s.
STATUS_HOLD_SECS = float(os.environ.get("STATUS_HOLD_SECS", "15"))

# zen's 502/503s are ~1s transient blips. The opencode CLI survives them via
# internal retries; 9Router surfaces first-failure to the harness instead.
# Retry transient upstream failures here so both callers see a stable pipe.
# 429 is quota, not a blip — never retried.
UPSTREAM_RETRIES = int(os.environ.get("UPSTREAM_RETRIES", "2"))
UPSTREAM_RETRY_BACKOFF = float(os.environ.get("UPSTREAM_RETRY_BACKOFF", "0.4"))
_RETRYABLE_STATUSES = {502, 503, 504}

# After the 200 is committed the client sits on keepalive comments — and
# 9Router demonstrably tolerates 40s+ TTFT on that connection. zen's 503
# waves last minutes, far longer than any pre-commit window, so once committed
# we keep retrying the upstream behind the keepalives until it succeeds or
# this budget runs out (kept below 9Router's own ~240s fetch timeout).
PENDING_RETRY_SECS = float(os.environ.get("PENDING_RETRY_SECS", "120"))

# Outbound proxy (e.g. socks5://warp:1080). httpx does not auto-read HTTP_PROXY
# env vars for programmatic clients, so we pass it explicitly. Falls back to the
# standard env vars; empty means direct connection.
OUTBOUND_PROXY = (
    os.environ.get("OUTBOUND_PROXY")
    or os.environ.get("HTTPS_PROXY")
    or os.environ.get("HTTP_PROXY")
    or None
)


def _client(**kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=OUTBOUND_PROXY, **kwargs)

FREE_MODELS = [
    "deepseek-v4-flash-free",
    "big-pickle",
    "minimax-m2.5-free",
    "mimo-v2.5-free",
    "nemotron-3-super-free",
    "qwen3.6-plus-free",
    "muse-spark-1.2-contributor-free",
    "x-preview-f-free",
    "hy3-free",
]

# Zen free tier models have a 1M-token context; fit oversized requests so the
# upstream never rejects with a 1048576-token context error.
CONTEXT_LIMIT = 1048576
MIN_COMPLETION = 1024


def _fit_context(body: dict) -> dict:
    if "messages" not in body:
        return body
    messages = list(body.get("messages", []))
    max_tokens = body.get("max_tokens") or 0

    def cost(msgs):
        return sum(max(1, len(json.dumps(m)) // 2) for m in msgs)

    budget = int(CONTEXT_LIMIT * 0.9) - cost(messages)

    while budget < MIN_COMPLETION and len(messages) > 1:
        idx = 0 if messages[0].get("role") != "system" else 1
        messages.pop(idx)
        # A "tool" message is only valid right after an assistant message with
        # tool_calls; once that assistant is dropped the trailing tool messages
        # become orphans and upstream rejects with 400. Drop them in sync.
        while idx < len(messages) and messages[idx].get("role") == "tool":
            messages.pop(idx)
        budget = int(CONTEXT_LIMIT * 0.9) - cost(messages)

    out = dict(body)
    out["messages"] = messages
    if max_tokens > budget:
        out["max_tokens"] = max(MIN_COMPLETION, budget)
    if out["messages"] != body.get("messages") or out.get("max_tokens") != max_tokens:
        logger.info(
            "[FIT] msgs %d->%d max_tokens %s->%s budget=%d",
            len(body.get("messages", [])), len(out["messages"]),
            max_tokens, out.get("max_tokens"), budget,
        )
    return out

MODEL_MAP = {m: {"id": m, "object": "model", "created": 1779000000, "owned_by": "opencode"} for m in FREE_MODELS}

app = FastAPI(title="OpenCode Proxy", description="OpenAI-compatible API proxy for OpenCode Zen", version=VERSION)

# Wildcard origins with credentials is rejected by browsers per the CORS spec.
# This proxy authenticates via a bearer token, not cookies, so credentials off.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# --- Session & ID generation ---
_user_sessions: dict[str, dict] = {}
MAX_SESSIONS = 4096


def _gen_id(prefix: str) -> str:
    ts = hex(int(time.time() * 1000))[2:]
    rnd = secrets.token_urlsafe(12)[:16]
    return f"{prefix}_{ts}{rnd}"


def _get_session(user: str = "default") -> str:
    now = time.time()
    # Keys are client-supplied (X-Forwarded-For) so the dict can be inflated
    # arbitrarily; cap it by evicting expired sessions once over the limit.
    if len(_user_sessions) >= MAX_SESSIONS:
        for stale in [u for u, s in _user_sessions.items() if now - s["ts"] > 1800]:
            del _user_sessions[stale]
    sess = _user_sessions.get(user)
    if not sess or now - sess["ts"] > 1800:
        sess = {"id": _gen_id("ses"), "ts": now}
        _user_sessions[user] = sess
    return sess["id"]


def _make_headers(session_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer public",
        "User-Agent": f"opencode/{OC_VERSION} ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-request": _gen_id("msg"),
        "x-opencode-session": session_id,
    }


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- Rate limit detection ---

def _is_rate_limit_error(status: int, body: bytes) -> Optional[str]:
    if status == 429:
        try:
            data = json.loads(body.decode().strip())
            msg = data.get("error", {}).get("message", "") or data.get("message", "") or "Rate limit exceeded"
            return msg
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "Rate limit exceeded"
    if b"FreeUsageLimitError" in body:
        return "Free usage limit reached"
    return None


def _classify_upstream_error(err: dict, fallback_msg: str = "Upstream error") -> tuple[int, str]:
    """Map an upstream error object to (http_status, message).

    Only genuine throttle conditions get 429; everything else keeps a 5xx so
    callers don't mistake outages for quota exhaustion and lock the model for
    the wrong reason. The original upstream message is preserved verbatim; a
    generic label is synthesized only when the upstream sent none."""
    if not isinstance(err, dict):
        err = {}
    etype = str(err.get("type", "") or "")
    code = err.get("code")
    throttled = (
        "FreeUsageLimitError" in etype
        or "rate_limit" in etype.lower()
        or "rate limit" in str(err.get("message", "")).lower()
        or code == 429
        or str(code) == "429"
    )
    msg = str(err.get("message") or "").strip()
    if throttled:
        return 429, msg or "Rate limit exceeded"
    return 502, msg or fallback_msg


def _frame_error(frame: bytes) -> Optional[dict]:
    """Return the upstream error object carried by an SSE frame, if any.

    Detects Anthropic-style {"type":"error","error":{...}} frames AND plain
    OpenAI-style {"error":{...}} frames. Ordinary content that merely mentions
    'error' never matches: content lives inside choices[].delta, never at the
    JSON top level."""
    for line in frame.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        err = obj.get("error")
        if isinstance(err, dict):
            return err
        if obj.get("type") == "error":
            return {"message": "Upstream error"}
    return None


# --- HTTPX streaming ---

def _oai_error_sse(message: str, status: Optional[int] = None) -> bytes:
    err: dict = {"message": message}
    if status is not None:
        err["status"] = status
    return f"data: {json.dumps({'error': err})}\n\ndata: [DONE]\n\n".encode()


def _is_done_frame(frame: bytes) -> bool:
    # Only a standalone `data: [DONE]` line terminates the stream. The
    # marker inside a JSON content string (model writing SSE-handling
    # code) sits mid-line and must NOT match — JSON escapes newlines,
    # so embedded text can never form its own SSE line.
    return any(l.strip() in (b"data: [DONE]", b"data:[DONE]") for l in frame.split(b"\n"))


def _scan_frame(frame: bytes) -> dict:
    """Extract liveness signals from one raw SSE frame.

    live: frame carries real output (content/reasoning/tool_calls or non-zero
          completion usage). bad: terminal finish_reason contains 'error'
          (Console's in-band failure mode). done: proper [DONE]. err: an error
          object (Anthropic- or OpenAI-style)."""
    info = {"live": False, "bad": False, "done": False, "err": None}
    info["done"] = any(l.strip() in (b"data: [DONE]", b"data:[DONE]") for l in frame.split(b"\n"))
    for line in frame.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        err = obj.get("error")
        if isinstance(err, dict):
            info["err"] = err
        elif obj.get("type") == "error":
            info["err"] = {"message": "Upstream error"}
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                info["live"] = True
            fr = ch.get("finish_reason")
            if fr and "error" in str(fr):
                info["bad"] = True
        usage = obj.get("usage")
        if isinstance(usage, dict) and (usage.get("completion_tokens") or 0) > 0:
            info["live"] = True
        # Responses-API frames (no choices[]): output deltas are live; the
        # lifecycle terminals are done; response.failed is an error.
        rtype = obj.get("type")
        if isinstance(rtype, str) and rtype.startswith("response."):
            if rtype.endswith(".delta") and isinstance(obj.get("delta"), str):
                info["live"] = True
            if rtype in ("response.completed", "response.incomplete"):
                rusage = (obj.get("response") or {}).get("usage") or {}
                if isinstance(rusage, dict) and (rusage.get("output_tokens") or 0) > 0:
                    info["live"] = True
                info["done"] = True
            if rtype == "response.failed":
                info["done"] = True
                info["err"] = {"message": "response failed"}
    # Byte-level fallback mirrors the relay's own marker detection so probes
    # and relays can never disagree about whether a frame carried output.
    if b'"tool_calls"' in frame:
        info["live"] = True
    return info


async def _probe_sse_stream(resp, client, hold_secs: float) -> tuple[Optional[tuple[int, str]], list, Optional[asyncio.Queue]]:
    """Read ahead SSE frames up to hold_secs to learn whether a 200 stream is
    alive before committing HTTP 200 to the caller.

    A single producer task owns the one-and-only resp.aiter_bytes() iteration
    (httpx forbids re-iterating a consumed stream — the v1.5.0 StreamConsumed
    regression) and feeds complete frames into a queue. The probe inspects the
    queue head; the relay phase drains the remainder. Cancellation-safe: the
    producer is never cancelled while the relay will keep consuming.

    Returns (dead, buffered_frames, live_queue):
      dead=None                      → relay: replay buffered, then drain queue
      dead=(status, reason)          → resources released; surface a real error
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        buf = b""
        try:
            async for raw in resp.aiter_bytes(chunk_size=None):
                buf += raw
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    await queue.put(frame)
            if buf:  # trailing partial frame — relay raw like the old tail flush
                await queue.put(buf)
            await queue.put(None)
        except BaseException as e:
            await queue.put(e)

    task = asyncio.create_task(producer())

    buffered: list = []
    st = {"live": False, "bad": False, "done": False, "err": None}

    async def consume():
        while True:
            item = await queue.get()
            if item is None or isinstance(item, BaseException):
                # EOF or producer failure: re-queue for the relay phase
                await queue.put(item)
                return
            buffered.append(item)
            s = _scan_frame(item)
            st["live"] = st["live"] or s["live"]
            st["bad"] = st["bad"] or s["bad"]
            st["done"] = st["done"] or s["done"]
            if s["err"] is not None:
                st["err"] = s["err"]
            if s["live"] or s["bad"] or s["err"] is not None or s["done"]:
                return

    timed_out = False
    try:
        await asyncio.wait_for(consume(), timeout=hold_secs)
    except asyncio.TimeoutError:
        timed_out = True

    if timed_out or st["live"]:
        return None, buffered, queue

    # Dead stream: stop the producer and release the connection.
    task.cancel()
    await resp.aclose()
    await client.aclose()
    if st["err"] is not None:
        rstatus, reason = _classify_upstream_error(st["err"])
    elif st["bad"]:
        rstatus, reason = 502, "no usable content (finish_reason: network_error)"
    else:
        rstatus, reason = 502, "no usable content (empty completion)"
    return (rstatus, reason), buffered, None


async def _zen_stream_upstream(url: str, headers: dict, body: dict, client_ip: str) -> tuple[int, Optional[str], Optional[AsyncIterator[bytes]]]:
    """Open a stream to Zen. Returns (200, None, iterator) on success or
    (status, detail, None) on error — so non-200 responses surface with the
    real HTTP status (pool-proxy then retries 429s properly)."""
    attempt = 0
    while True:
        client = _client(timeout=httpx.Timeout(180.0, connect=15.0))
        try:
            req = client.build_request("POST", url, headers=headers, json=body)
            resp = await client.send(req, stream=True)
        except httpx.TransportError:
            # Connect/read failures before any bytes: transient-safe to retry.
            await client.aclose()
            if attempt < UPSTREAM_RETRIES:
                attempt += 1
                delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning("[RETRY] upstream transport error, attempt %d/%d in %.1fs", attempt, UPSTREAM_RETRIES + 1, delay)
                await asyncio.sleep(delay)
                continue
            raise
        except BaseException:
            # BaseException: CancelledError (client hung up before headers) must
            # also release the connection, not just plain HTTP errors.
            await client.aclose()
            raise
        if resp.status_code in _RETRYABLE_STATUSES and attempt < UPSTREAM_RETRIES:
            await resp.aread()
            await resp.aclose()
            await client.aclose()
            attempt += 1
            delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning("[RETRY] upstream %s, attempt %d/%d in %.1fs", resp.status_code, attempt, UPSTREAM_RETRIES + 1, delay)
            await asyncio.sleep(delay)
            continue
        break
    if resp.status_code != 200:
        raw = await resp.aread()
        await resp.aclose()
        await client.aclose()
        rl = _is_rate_limit_error(resp.status_code, raw)
        if rl:
            detail = f"Rate limit exceeded: {rl} ({resp.status_code})"
        else:
            try:
                obj = json.loads(raw.decode().strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                obj = {}
            err_obj = obj.get("error") if isinstance(obj, dict) and isinstance(obj.get("error"), dict) else {}
            _, emsg = _classify_upstream_error(err_obj or {})
            detail = f"{emsg} ({resp.status_code})"
        logger.warning("[ZEN] stream %s %s body=%s", resp.status_code, url, raw[:300])
        return resp.status_code, detail, None

    dead, buffered, live_q = await _probe_sse_stream(resp, client, STATUS_HOLD_SECS)
    if dead is not None:
        logger.warning("[ZEN] dead 200-stream from %s: %s", client_ip, dead[1])
        return dead[0], f"{dead[1]} (200)", None

    state = {
        "saw_tool": False,
        "saw_finish": False,
        "saw_usage": False,
        "saw_content": False,
        "saw_reasoning": False,
        "done_sent": False,
        "stop": False,
        "relayed": 0,
        "live": False,
    }

    def _has_non_null_finish(data: bytes) -> bool:
        # Detect finish_reason with a non-null string value (tool_calls/stop/length)
        if b'"finish_reason"' not in data:
            return False
        if b'"finish_reason":null' in data or b'"finish_reason": null' in data:
            return b'"finish_reason":"' in data or b'"finish_reason": "' in data
        return True

    def _is_junk_frame(frame: bytes) -> bool:
        # Zen filler: empty-choices heartbeat chunks (muse-spark) and the
        # trailing `{"choices":[],"cost":"0"}`. They carry no content,
        # usage or finish, and 9Router misreads them as a dead model.
        junk = 0
        parsed = 0
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return False
            parsed += 1
            if isinstance(obj, dict) and obj.get("choices") == [] and not obj.get("usage"):
                junk += 1
        return parsed > 0 and junk == parsed

    def _handle(frame: bytes):
        """Process one complete SSE frame; yields relayable bytes and updates
        shared relay state."""
        if _is_done_frame(frame):
            # Zen appends trailing frames (`data: {"choices":[],"cost":"0"}`)
            # after the real [DONE]; relay stops here so 9Router doesn't parse
            # empty choices as a dead model.
            if not state["live"]:
                # Clean [DONE] but zero usable output post-commit — abort so
                # 9Router classifies the model as failed, not empty-success.
                state["stop"] = True
                raise UpstreamDead("no usable content (empty completion)")
            state["done_sent"] = True
            state["stop"] = True
            yield b"data: [DONE]\n\n"
            return
        frame_err = _frame_error(frame)
        if frame_err is not None:
            estatus, emsg = _classify_upstream_error(frame_err)
            logger.warning("[UPSTREAM ERROR FRAME] %s from %s", emsg, client_ip)
            state["stop"] = True
            raise UpstreamDead(f"{emsg} ({estatus})")
        if _is_junk_frame(frame):
            return
        scan = _scan_frame(frame)
        if scan["live"]:
            state["live"] = True
        if scan["bad"]:
            # Console's in-band failure: finish_reason:"network_error" under a
            # well-formed frame. Aborting (no DONE) is what 9Router reads as a
            # model failure; a clean end reads as an empty success.
            state["stop"] = True
            raise UpstreamDead("no usable content (finish_reason: network_error)")
        if b'"tool_calls"' in frame:
            state["saw_tool"] = True
            state["saw_content"] = True
        if _has_non_null_finish(frame):
            state["saw_finish"] = True
        if b'"usage"' in frame and (b'"prompt_tokens"' in frame or b'"completion_tokens"' in frame):
            state["saw_usage"] = True
        if b'"content":"' in frame or b'"content": "' in frame:
            state["saw_content"] = True
        if b'"reasoning_content"' in frame:
            state["saw_reasoning"] = True
        out = frame + b"\n\n"
        state["relayed"] += len(out)
        yield out

    async def gen():
        try:
            for frame in buffered:
                for out in _handle(frame):
                    yield out
                if state["stop"]:
                    return
            if live_q is None:
                return
            while True:
                item = await live_q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                for out in _handle(item):
                    yield out
                if state["stop"]:
                    return
            # Muse Spark via Zen's chat completions often omits the terminal
            # finish_reason + [DONE] for tool calls (it emits usage/cost instead).
            # Without them the harness reports "Tool call ended without a terminal event."
            if not state["done_sent"]:
                logger.info(
                    "[ZEN] stream ended without DONE: finish=%s usage=%s tool=%s content=%s reasoning=%s bytes=%d client=%s",
                    state["saw_finish"], state["saw_usage"], state["saw_tool"], state["saw_content"], state["saw_reasoning"], state["relayed"], client_ip,
                )
                if not state["saw_tool"] and not state["saw_finish"] and not state["saw_usage"]:
                    # Upstream cut the stream before any terminal signal —
                    # mid-response (content was flowing) or all-junk (empty).
                    # Zen kills long free-tier streams this way; ABORT the
                    # connection so 9Router classifies the model as failed
                    # instead of parsing a clean empty stream as success.
                    logger.warning(
                        "[ZEN] stream ended WITHOUT terminal signal (truncated or empty): content=%s bytes=%d client=%s",
                        state["saw_content"], state["relayed"], client_ip,
                    )
                    raise UpstreamDead("Upstream stream ended before completion")
                if state["saw_tool"] and not state["saw_finish"]:
                    yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
                yield b"data: [DONE]\n\n"
        finally:
            await resp.aclose()
            await client.aclose()

    return 200, None, gen()


# --- Anthropic native translation ---

DEFAULT_MODEL = "deepseek-v4-flash-free"

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def _map_model(model: Optional[str]) -> str:
    return model if model in FREE_MODELS else DEFAULT_MODEL


def _anthropic_to_openai(body: dict) -> dict:
    """Translate an Anthropic /v1/messages request into OpenAI chat.completions format.

    Handles: system (str or block list), text blocks, image blocks (as data URIs),
    tool_use blocks -> tool_calls, tool_result blocks -> role "tool" messages, and
    tools with input_schema -> function tools. Tool results are emitted before any
    text of the same user message so they attach to the preceding assistant tool_calls.
    """
    model = _map_model(body.get("model"))
    oai_messages: list[dict] = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            system_text = system
        else:
            system_text = "".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
        if system_text:
            oai_messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64" and src.get("data") and src.get("media_type"):
                    text_parts.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
            elif btype == "tool_use":
                try:
                    args = json.dumps(block.get("input", {}))
                except (TypeError, ValueError):
                    args = "{}"
                tool_calls.append({
                    "id": block.get("id") or _gen_id("call"),
                    "type": "function",
                    "function": {"name": block.get("name") or "unknown", "arguments": args},
                })
            elif btype == "tool_result":
                tc = block.get("content")
                if isinstance(tc, str):
                    text = tc
                else:
                    text = "".join(x.get("text", "") for x in tc or [] if isinstance(x, dict))
                if block.get("is_error"):
                    text = f"Error: {text}"
                tool_results.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": text})

        if role == "assistant":
            if tool_calls:
                oai_messages.append({
                    "role": "assistant",
                    "content": "".join(t for t in text_parts if isinstance(t, str)) or None,
                    "tool_calls": tool_calls,
                })
            elif text_parts:
                oai_messages.append({"role": "assistant", "content": "".join(t for t in text_parts if isinstance(t, str))})
        else:
            oai_messages.extend(tool_results)
            if text_parts:
                if any(isinstance(t, dict) for t in text_parts):
                    # Image present: content must be a parts array, and every
                    # element an object — bare strings inside a content-parts
                    # array are schema-invalid for OpenAI backends.
                    parts = [{"type": "text", "text": t} if isinstance(t, str) else t for t in text_parts]
                    oai_messages.append({"role": role, "content": parts})
                else:
                    oai_messages.append({"role": role, "content": "".join(text_parts)})

    out: dict = {"model": model, "messages": oai_messages}
    for key in ("max_tokens", "temperature", "top_p", "stream"):
        if body.get(key) is not None:
            out[key] = body.get(key)
    if "max_tokens" not in out:
        out["max_tokens"] = 1024

    tools = body.get("tools")
    if tools:
        mapped = []
        for t in tools:
            if isinstance(t, dict) and t.get("input_schema"):
                mapped.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description") or "",
                        "parameters": t.get("input_schema"),
                    },
                })
        if mapped:
            out["tools"] = mapped

    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name", "")}}
        elif ctype == "any":
            out["tool_choice"] = "required"
        else:
            out["tool_choice"] = "auto"
    return out


def _openai_to_anthropic(resp: dict, model: str) -> dict:
    """Translate a non-stream OpenAI chat.completions response into an Anthropic message."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_blocks: list[dict] = []

    text = message.get("content")
    if text:
        if isinstance(text, str):
            text_str = text
        elif isinstance(text, list):
            text_str = "".join(t.get("text", "") for t in text if isinstance(t, dict))
        else:
            text_str = str(text)
        if text_str:
            content_blocks.append({"type": "text", "text": text_str})

    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": call.get("id") or _gen_id("toolu"),
            "name": fn.get("name") or "unknown",
            "input": args,
        })

    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or _gen_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _keepalive_wrapper(gen: AsyncIterator[bytes], interval: float = 10.0) -> AsyncIterator[bytes]:
    """Wrap an SSE byte stream with periodic comment keepalives.

    While upstream is silent (long TTFT on 400k-token prompts) the wrapper
    emits `: keepalive\\n\\n` every `interval` seconds so middle-hops
    (9Router's ~25s first-byte timeout) don't kill the connection.

    Frame-aware: chunks are buffered until a complete `\\n\\n`-terminated SSE
    frame is available, and keepalives are emitted ONLY when no partial frame
    is pending — an SSE comment spliced into a half-delivered JSON line
    corrupts it (9Router: 'Failed to parse SSE line ... "mode: keepalive').

    Uses a producer task + queue: never cancels an in-flight upstream read
    (cancelling __anext__ kills the httpx stream and drops data). If the
    upstream stream fails mid-flight the failure becomes an OpenAI-style
    SSE error event + [DONE] instead of a silently truncated stream.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        try:
            async for chunk in gen:
                await queue.put(chunk)
        except Exception as e:
            logger.exception("[SSE] upstream stream failed")
            await queue.put(e)
            return
        await queue.put(None)

    task = asyncio.create_task(producer())
    buf = b""
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                if not buf:  # frame boundary — a comment here is safe
                    yield b": keepalive\n\n"
                continue
            if item is None:
                break
            if isinstance(item, BaseException):
                if buf:
                    yield buf
                raise item
            buf += item
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                yield frame + b"\n\n"
        if buf:
            yield buf
    finally:
        task.cancel()


_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


class UpstreamDead(Exception):
    """Post-commit upstream failure.

    Raised inside the response generator AFTER HTTP 200 was committed, so the
    connection terminates mid-stream (premature close) instead of delivering a
    well-formed error frame + [DONE] — which protocol-loose callers like
    9Router parse as a completed EMPTY success ('succeeded · IN 0 · OUT 0')
    and never fail over. A transport-level abort is the only signal they
    classify as a model failure."""


async def _relay_openai_chunks(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async for chunk in _keepalive_wrapper(chunks):
        yield chunk


async def _pending_relay_with_retry(
    first_task: asyncio.Task,
    make_task,
    budget: float,
    relay_fn,
    interval: float = 10.0,
) -> AsyncIterator[bytes]:
    """Post-commit pending relay with upstream retries.

    The client already has HTTP 200 + keepalives, so a failed upstream attempt
    no longer needs to abort: start a NEW upstream request and keep feeding
    keepalives until one succeeds or `budget` seconds elapse (then abort via
    UpstreamDead so the caller classifies the model as failed)."""
    deadline = time.monotonic() + budget
    task = first_task
    attempt = 0
    while True:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                break
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
            except Exception:
                break
        try:
            status, err, chunks = task.result()
        except Exception as e:
            err, status, chunks = str(e), 0, None
        if err is None and chunks is not None:
            async for chunk in relay_fn(chunks):
                yield chunk
            return
        attempt += 1
        if time.monotonic() >= deadline:
            logger.warning("[PENDING] giving up after %d attempt(s): %s (%s)", attempt, err, status)
            raise UpstreamDead(f"{err} ({status})")
        logger.warning("[PENDING RETRY] attempt %d failed (%s %s), retrying upstream", attempt, status, err)
        await asyncio.sleep(1.0)
        # make_task() returns a coroutine (e.g. `lambda: _zen_stream_upstream(...)`):
        # wrap it, the loop below needs a Task with .done()/.result().
        task = asyncio.ensure_future(make_task())


async def _relay_from_task(task: asyncio.Task, interval: float = 10.0) -> AsyncIterator[bytes]:
    """Consume an already-started _zen_stream_upstream task: keepalive comments
    while pending; on failure RAISE (abort the committed 200) — never a clean
    error frame + [DONE], which 9Router parses as an empty success."""
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                break
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
            except Exception:
                # Task failed while being awaited; task.result() below
                # re-raises it into the handler instead of escaping here.
                break
        try:
            status, err, chunks = task.result()
        except Exception:
            # Upstream died while the client was already on keepalives: abort
            # (UpstreamDead semantics) instead of a parseable error frame.
            raise
        if err:
            raise UpstreamDead(f"{err} ({status})")
        async for chunk in _keepalive_wrapper(chunks):
            yield chunk
    finally:
        if not task.done():
            task.cancel()


async def _relay_openai_stream(url: str, headers: dict, body: dict, client_ip: str, interval: float = 10.0) -> AsyncIterator[bytes]:
    """Relay an OpenAI-format SSE stream: keepalive comments while upstream
    headers are pending, keepalives during body silence, and an SSE error
    event + [DONE] on any upstream failure so downstream retries (9Router)
    can classify it. Cancels the upstream request if the client hangs up."""
    task = asyncio.create_task(_zen_stream_upstream(url, headers, body, client_ip))
    async for chunk in _relay_from_task(task, interval):
        yield chunk


async def _openai_stream_response(url: str, headers: dict, body: dict, client_ip: str) -> StreamingResponse:
    """Hold the HTTP response for STATUS_HOLD_SECS so upstream failures that
    arrive quickly surface as REAL HTTP statuses (downstream can lock/fallback
    correctly); only genuinely slow upstreams degrade to 200 + keepalive SSE."""
    task = asyncio.create_task(_zen_stream_upstream(url, headers, body, client_ip))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=STATUS_HOLD_SECS)
    except asyncio.TimeoutError:
        return StreamingResponse(
            _pending_relay_with_retry(
                task,
                lambda: _zen_stream_upstream(url, headers, body, client_ip),
                PENDING_RETRY_SECS,
                _relay_openai_chunks,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    except Exception as e:
        if not task.done():
            task.cancel()
        raise HTTPException(502, f"Upstream error: {e}")
    status, err, chunks = task.result()
    if err:
        raise HTTPException(status or 502, err)
    return StreamingResponse(_keepalive_wrapper(chunks), media_type="text/event-stream", headers=_SSE_HEADERS)


async def _anthropic_relay_from_task(task: asyncio.Task, model: str, interval: float = 10.0) -> AsyncIterator[bytes]:
    """Anthropic-format counterpart of _relay_from_task."""
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=interval)
                break
            except asyncio.TimeoutError:
                yield b": keepalive\n\n"
            except Exception:
                break
        try:
            status, err, chunks = task.result()
        except Exception:
            raise
        if err:
            raise UpstreamDead(f"{err} ({status})")
        async for chunk in _anthropic_events(chunks, model):
            yield chunk
    finally:
        if not task.done():
            task.cancel()


async def _anthropic_stream_response(url: str, headers: dict, body: dict, client_ip: str, model: str) -> StreamingResponse:
    """Header-hold counterpart of _openai_stream_response for /v1/messages."""
    task = asyncio.create_task(_zen_stream_parsed(url, headers, body, client_ip))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=STATUS_HOLD_SECS)
    except asyncio.TimeoutError:
        async def relay_anth(chunks):
            async for ev in _anthropic_events(chunks, model):
                yield ev
        return StreamingResponse(
            _pending_relay_with_retry(
                task,
                lambda: _zen_stream_parsed(url, headers, body, client_ip),
                PENDING_RETRY_SECS,
                relay_anth,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    except Exception as e:
        if not task.done():
            task.cancel()
        raise HTTPException(502, f"Upstream error: {e}")
    status, err, chunks = task.result()
    if err:
        raise HTTPException(status or 502, err)
    return StreamingResponse(_anthropic_events(chunks, model), media_type="text/event-stream", headers=_SSE_HEADERS)


async def _anthropic_events(chunks: AsyncIterator[dict], model: str) -> AsyncIterator[bytes]:
    """Translate parsed OpenAI SSE chunks into Anthropic message events.

    Emits message_start, content_block_start/delta/stop (text + thinking + tool_use
    with input_json_delta), message_delta, message_stop. A mid-stream error chunk
    (FreeUsageLimitError) becomes an Anthropic error event and ends the stream.
    Always emits a terminal sequence even on empty/reasoning-only streams so the
    harness never sees 'Tool call ended without a terminal event'.
    """
    # Emit message_start immediately so even empty streams have a terminal
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": _gen_id("msg"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    text_idx: Optional[int] = None
    reasoning_idx: Optional[int] = None
    tool_index: dict[int, int] = {}
    next_block = 0
    output_tokens = 0
    stop_reason = None
    try:
        async for chunk in chunks:
            if chunk.get("type") == "error":
                yield _sse_event("error", {"type": "error", "error": chunk.get("error") or {"type": "rate_limit_error", "message": "Rate limit exceeded (429)"}})
                return
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            # Reasoning/thinking deltas (muse-spark, deepseek, etc.)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("reasoning_text") or delta.get("thought")
            if reasoning:
                if reasoning_idx is None:
                    reasoning_idx = next_block
                    next_block += 1
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": reasoning_idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": reasoning_idx,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                })
            text = delta.get("content")
            if text:
                if text_idx is None:
                    text_idx = next_block
                    next_block += 1
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": text_idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_idx,
                    "delta": {"type": "text_delta", "text": text},
                })
            for call in delta.get("tool_calls") or []:
                call_idx = call.get("index", 0)
                fn = call.get("function") or {}
                if call_idx not in tool_index:
                    block_idx = next_block
                    next_block += 1
                    tool_index[call_idx] = block_idx
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.get("id") or _gen_id("toolu"),
                            "name": fn.get("name") or "unknown",
                            "input": {},
                        },
                    })
                    args = fn.get("arguments")
                    if args:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        })
                else:
                    args = fn.get("arguments")
                    if args:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": tool_index[call_idx],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        })
            if choice.get("finish_reason"):
                stop_reason = _FINISH_TO_STOP.get(choice.get("finish_reason"), "end_turn")
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                output_tokens = usage["completion_tokens"]
        # Always emit terminal sequence
        if text_idx is not None:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_idx})
        if reasoning_idx is not None:
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": reasoning_idx})
        for block_idx in sorted(tool_index.values()):
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        yield _sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield _sse_event("message_stop", {"type": "message_stop"})
    except Exception:
        logger.exception("[ANTH] SSE translation failed")
        raise


# --- HTTPX streaming (parsed variant for Anthropic) ---

async def _zen_stream_parsed(url: str, headers: dict, body: dict, client_ip: str) -> tuple[int, Optional[str], Optional[AsyncIterator[dict]]]:
    """Like _zen_stream_upstream but yields parsed JSON objects per SSE frame.

    Skips [DONE] markers and non-JSON data lines. A FreeUsageLimitError frame
    (even mid-stream) becomes a synthetic rate_limit_error object and ends the
    stream. Returns (200, None, iterator) or (status, detail, None)."""
    attempt = 0
    while True:
        client = _client(timeout=httpx.Timeout(180.0, connect=15.0))
        try:
            req = client.build_request("POST", url, headers=headers, json=body)
            resp = await client.send(req, stream=True)
        except httpx.TransportError:
            await client.aclose()
            if attempt < UPSTREAM_RETRIES:
                attempt += 1
                delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning("[RETRY] upstream transport error, attempt %d/%d in %.1fs", attempt, UPSTREAM_RETRIES + 1, delay)
                await asyncio.sleep(delay)
                continue
            raise
        except BaseException:
            await client.aclose()
            raise
        if resp.status_code in _RETRYABLE_STATUSES and attempt < UPSTREAM_RETRIES:
            await resp.aread()
            await resp.aclose()
            await client.aclose()
            attempt += 1
            delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning("[RETRY] upstream %s, attempt %d/%d in %.1fs", resp.status_code, attempt, UPSTREAM_RETRIES + 1, delay)
            await asyncio.sleep(delay)
            continue
        break
    if resp.status_code != 200:
        raw = await resp.aread()
        await resp.aclose()
        await client.aclose()
        rl = _is_rate_limit_error(resp.status_code, raw)
        if rl:
            detail = f"Rate limit exceeded: {rl} ({resp.status_code})"
        else:
            try:
                obj = json.loads(raw.decode().strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                obj = {}
            err_obj = obj.get("error") if isinstance(obj, dict) and isinstance(obj.get("error"), dict) else {}
            _, emsg = _classify_upstream_error(err_obj or {})
            detail = f"{emsg} ({resp.status_code})"
        logger.warning("[ZEN] stream %s %s body=%s", resp.status_code, url, raw[:300])
        return resp.status_code, detail, None

    dead, buffered, live_q = await _probe_sse_stream(resp, client, STATUS_HOLD_SECS)
    if dead is not None:
        logger.warning("[ZEN] dead 200-stream from %s: %s", client_ip, dead[1])
        return dead[0], f"{dead[1]} (200)", None

    def _parse_objs(frame: bytes):
        """Yield parsed JSON objects from one SSE frame's data lines."""
        for line in frame.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

    def _emit(obj: dict):
        """Raise UpstreamDead on upstream error frames; else None."""
        err_obj = obj.get("error") if isinstance(obj.get("error"), dict) else None
        if err_obj is not None or obj.get("type") == "error":
            estatus, emsg = _classify_upstream_error(err_obj or {})
            logger.warning("[UPSTREAM ERROR FRAME] %s from %s", emsg, client_ip)
            raise UpstreamDead(f"{emsg} ({estatus})")
        return None

    async def gen():
        try:
            for frame in buffered:
                for obj in _parse_objs(frame):
                    _emit(obj)
                    yield obj
            if live_q is None:
                return
            while True:
                item = await live_q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                for obj in _parse_objs(item):
                    _emit(obj)
                    yield obj
        finally:
            await resp.aclose()
            await client.aclose()

    return 200, None, gen()


# --- Responses API bridge (muse-spark) ---
# zen's /chat/completions returns 500 for muse-spark (Console), while
# /responses works — the opencode CLI itself talks the Responses API for this
# model. 9Router only speaks chat/completions, so we translate both ways.

RESPONSES_MODELS = {"muse-spark-1.2-contributor-free"}


def _chat_to_responses(body: dict) -> dict:
    """Translate an OpenAI chat.completions request into a Responses request."""
    instructions: list[str] = []
    input_items: list[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "developer"):
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content or [] if isinstance(b, dict)
            )
            if text:
                instructions.append(text)
            continue
        if role == "tool":
            output = content if isinstance(content, str) else json.dumps(content)
            input_items.append({"type": "function_call_output", "call_id": msg.get("tool_call_id", ""), "output": output})
            continue
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                })
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content or [] if isinstance(b, dict)
            )
            if text:
                input_items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
            continue
        # user
        if isinstance(content, str):
            input_items.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
        else:
            parts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    parts.append({"type": "input_text", "text": b.get("text", "")})
                elif b.get("type") == "image_url":
                    u = (b.get("image_url") or {}).get("url", "")
                    if u.startswith("data:"):
                        parts.append({"type": "input_image", "image_url": u})
                    elif u:
                        parts.append({"type": "input_image", "image_url": u})
            if parts:
                input_items.append({"role": "user", "content": parts})

    tools_out = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function") or {}
            tools_out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })

    out: dict = {"model": body.get("model"), "input": input_items, "stream": bool(body.get("stream"))}
    if instructions:
        out["instructions"] = "\n\n".join(instructions)
    mt = body.get("max_tokens")
    if mt:
        # muse-spark runs reasoning effort=high before answering; a small
        # max_output_tokens is consumed by reasoning alone → response.incomplete
        # with zero text (the CLI never sets one). Give it headroom.
        out["max_output_tokens"] = max(int(mt), 2048)
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if tools_out:
        out["tools"] = tools_out
    tc = body.get("tool_choice")
    if isinstance(tc, str) and tc in ("auto", "required", "none"):
        out["tool_choice"] = tc
    elif isinstance(tc, dict):
        if tc.get("type") == "function":
            out["tool_choice"] = {"type": "function", "name": (tc.get("function") or {}).get("name", "")}
        elif tc.get("type") in ("auto", "required", "none"):
            out["tool_choice"] = tc.get("type")
    return out


def _map_responses_usage(rusage: dict) -> dict:
    return {
        "prompt_tokens": rusage.get("input_tokens", 0),
        "completion_tokens": rusage.get("output_tokens", 0),
        "total_tokens": rusage.get("total_tokens", 0),
    }


async def _responses_events_to_chat(events: AsyncIterator[dict]) -> AsyncIterator[bytes]:
    """Translate parsed Responses-API events into chat.completion.chunk SSE."""
    state = {"role_sent": False, "tool_idx": 0, "saw_call": False, "finish": None, "usage": None, "emitted": False}

    def chunk(delta: Optional[dict] = None, finish: Optional[str] = None, usage: Optional[dict] = None, role: bool = False) -> bytes:
        d: dict = {}
        if role:
            d["role"] = "assistant"
        if delta:
            d.update(delta)
        choice: dict = {"index": 0, "delta": d}
        if finish:
            choice["finish_reason"] = finish
        frame: dict = {
            "id": _gen_id("chatcmpl"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [choice],
        }
        if usage is not None:
            frame["usage"] = usage
        return f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()

    async for ev in events:
        if not isinstance(ev, dict):
            continue
        t = ev.get("type", "")
        if t == "response.output_text.delta":
            state["emitted"] = True
            if not state["role_sent"]:
                state["role_sent"] = True
                yield chunk(role=True)
            yield chunk(delta={"content": ev.get("delta", "")})
        elif t in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            state["emitted"] = True
            if not state["role_sent"]:
                state["role_sent"] = True
                yield chunk(role=True)
            yield chunk(delta={"reasoning_content": ev.get("delta", "")})
        elif t == "response.function_call_arguments.delta":
            yield chunk(delta={"tool_calls": [{"index": state["tool_idx"] - 1, "function": {"arguments": ev.get("delta", "")}}]})
        elif t == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "message" and not state["role_sent"]:
                state["role_sent"] = True
                yield chunk(role=True)
            elif item.get("type") == "function_call":
                state["saw_call"] = True
                state["emitted"] = True
                yield chunk(delta={"tool_calls": [{
                    "index": state["tool_idx"],
                    "id": item.get("call_id") or _gen_id("call"),
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": ""},
                }]})
                state["tool_idx"] += 1
        elif t == "response.completed":
            resp = ev.get("response") or {}
            state["usage"] = _map_responses_usage(resp.get("usage") or {})
            state["finish"] = "tool_calls" if state["saw_call"] else "stop"
        elif t == "response.incomplete":
            resp = ev.get("response") or {}
            state["usage"] = _map_responses_usage(resp.get("usage") or {})
            state["finish"] = "tool_calls" if state["saw_call"] else "length"
        elif t == "response.failed":
            rerr = (ev.get("response") or {}).get("error") or {}
            raise UpstreamDead(str(rerr.get("message") or "Upstream error"))
        elif t == "error":
            err = ev.get("error")
            _, emsg = _classify_upstream_error(err if isinstance(err, dict) else {})
            raise UpstreamDead(emsg)
        # created/in_progress/queued/ping/content_part.*/output_item.done → skip
    if state["finish"] is None and state["usage"] is None:
        # No terminal event — upstream cut the stream mid-flight: abort.
        raise UpstreamDead("Upstream stream ended before completion")
    if not state["emitted"]:
        # Terminal without any output (reasoning-only budget burn): abort.
        raise UpstreamDead("no usable content (empty completion)")
    yield chunk(finish=state["finish"] or "stop", usage=state["usage"])
    yield b"data: [DONE]\n\n"


def _responses_to_chat_json(resp_obj: dict, model: str) -> dict:
    """Translate a non-streaming Responses object into chat.completions JSON."""
    text_parts: list[str] = []
    tool_calls = []
    for item in resp_obj.get("output") or []:
        it = item.get("type")
        if it == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
        elif it == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or _gen_id("call"),
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")},
            })
    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    finish = "tool_calls" if tool_calls else ("length" if resp_obj.get("status") == "incomplete" else "stop")
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": resp_obj.get("id") or _gen_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _map_responses_usage(resp_obj.get("usage") or {}),
    }


async def _chat_via_responses(headers: dict, body: dict, client_ip: str):
    """Serve a chat.completions request for a RESPONSES_MODELS model by calling
    zen's /responses endpoint and translating back."""
    rbody = _chat_to_responses(body)
    url = f"{BASE_URL}/responses"

    if rbody.get("stream"):
        task = asyncio.create_task(_zen_stream_parsed(url, headers, rbody, client_ip))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=STATUS_HOLD_SECS)
        except asyncio.TimeoutError:
            async def relay_rsp(chunks):
                async for out in _responses_events_to_chat(chunks):
                    yield out
            return StreamingResponse(
                _pending_relay_with_retry(
                    task,
                    lambda: _zen_stream_parsed(url, headers, rbody, client_ip),
                    PENDING_RETRY_SECS,
                    relay_rsp,
                ),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            if not task.done():
                task.cancel()
            raise HTTPException(502, f"Upstream error: {e}")
        status, err, chunks = task.result()
        if err:
            raise HTTPException(status or 502, err)
        return StreamingResponse(
            _keepalive_wrapper(_responses_events_to_chat(chunks)),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    async with _client(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        resp = await client.post(url, headers=headers, json=rbody)
    if resp.status_code != 200:
        logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
        rl = _is_rate_limit_error(resp.status_code, resp.content)
        if rl:
            raise HTTPException(429, f"Rate limit: {rl}")
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")
    out = _responses_to_chat_json(_safe_json(resp), body.get("model"))
    choice = out["choices"][0]
    if not choice["message"].get("content") and not choice["message"].get("tool_calls") and out["usage"]["completion_tokens"] == 0:
        raise HTTPException(502, "no usable content (empty completion)")
    return out


# --- Routes ---

async def _json_body(request: Request) -> dict:
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "Invalid JSON body")
    if not isinstance(data, dict):
        raise HTTPException(400, "JSON body must be an object")
    return data


def _wants_stream(request: Request, body: bytes) -> bool:
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        return True
    return b'"stream":true' in body or b'"stream": true' in body


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION, "models": len(FREE_MODELS)}


def _safe_json(resp: httpx.Response):
    """Parse an upstream JSON body; a 200 with garbage must surface as a clean
    502, not an unhandled crash (plain-text 'Internal Server Error' 500)."""
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        logger.warning("[ZEN] non-JSON body (%s): %s", resp.status_code, resp.text[:200])
        raise HTTPException(502, "Upstream returned a non-JSON body")


async def _post_with_retry(url: str, headers: dict, json_body: dict) -> httpx.Response:
    """POST with transient-failure retry (502/503/504/connect errors). The
    response body is fully read before returning, so each attempt's client is
    closed and only the final response escapes."""
    attempt = 0
    while True:
        client = _client(timeout=httpx.Timeout(180.0, connect=15.0))
        try:
            resp = await client.post(url, headers=headers, json=json_body)
            _ = resp.content  # force read so this attempt's client can close
            await client.aclose()
        except httpx.TransportError:
            await client.aclose()
            if attempt < UPSTREAM_RETRIES:
                attempt += 1
                delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.warning("[RETRY] upstream transport error, attempt %d/%d in %.1fs", attempt, UPSTREAM_RETRIES + 1, delay)
                await asyncio.sleep(delay)
                continue
            raise
        if resp.status_code in _RETRYABLE_STATUSES and attempt < UPSTREAM_RETRIES:
            attempt += 1
            delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning("[RETRY] upstream %s, attempt %d/%d in %.1fs", resp.status_code, attempt, UPSTREAM_RETRIES + 1, delay)
            await asyncio.sleep(delay)
            continue
        return resp


@app.get("/v1/models")
async def list_models():
    if BROKE_MODE:
        return {"object": "list", "data": list(MODEL_MAP.values())}

    async with _client(timeout=15.0) as client:
        resp = await client.get(f"{BASE_URL}/models", headers=_make_headers(_gen_id("ses")))
    if resp.status_code != 200:
        return {"object": "list", "data": list(MODEL_MAP.values())}
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {"object": "list", "data": list(MODEL_MAP.values())}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _json_body(request)
    body = _fit_context(body)
    ip = _get_client_ip(request)
    session = _get_session(ip)
    headers = _make_headers(session)
    stream = body.get("stream", False)

    logger.info("[OAI] %s %s stream=%s", ip, body.get("model", "?"), stream)

    url = f"{BASE_URL}/chat/completions"

    if body.get("model") in RESPONSES_MODELS:
        return await _chat_via_responses(headers, body, ip)

    if stream:
        return await _openai_stream_response(url, headers, body, ip)

    resp = await _post_with_retry(url, headers, body)

    if resp.status_code != 200:
        logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
        rl = _is_rate_limit_error(resp.status_code, resp.content)
        if rl:
            raise HTTPException(429, f"Rate limit: {rl}")
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")

    return _safe_json(resp)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await _json_body(request)
    ip = _get_client_ip(request)
    session = _get_session(ip)
    headers = _make_headers(session)
    stream = body.get("stream", False)
    model = _map_model(body.get("model"))

    logger.info("[ANTH] %s %s stream=%s", ip, body.get("model", "?"), stream)

    oai_body = _anthropic_to_openai(body)
    oai_body = _fit_context(oai_body)
    url = f"{BASE_URL}/chat/completions"

    if stream:
        oai_body["stream"] = True
        return await _anthropic_stream_response(url, headers, oai_body, ip, model)

    resp = await _post_with_retry(url, headers, oai_body)

    if resp.status_code != 200:
        logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
        rl = _is_rate_limit_error(resp.status_code, resp.content)
        if rl:
            raise HTTPException(429, f"Rate limit: {rl}")
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")

    return _openai_to_anthropic(_safe_json(resp), model)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    body = await _json_body(request)
    oai_body = _anthropic_to_openai(body)
    total = sum(max(1, len(json.dumps(m)) // 4) for m in oai_body.get("messages", []))
    return {"input_tokens": total}


@app.api_route("/v1/responses", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def responses_proxy(request: Request):
    """Explicit handler for the Responses API (muse-spark via @ai-sdk/openai).

    Zen's Responses endpoint is at /v1/responses, not /v1/chat/completions.
    This avoids the catch_all double-v1 pitfall and ensures streaming
    keepalives for long-reasoning turns.
    """
    ip = _get_client_ip(request)
    session = _get_session(ip)
    headers = _make_headers(session)
    body = await request.body()
    try:
        body_json = json.loads(body) if body else None
    except json.JSONDecodeError:
        body_json = None
    if body_json:
        body_json = _fit_context(body_json)
    is_stream = _wants_stream(request, body)
    url = f"{BASE_URL}/responses"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    if is_stream and request.method == "POST" and body_json:
        return await _openai_stream_response(url, headers, body_json, ip)
    async with _client(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        method = request.method.upper()
        if method == "GET":
            resp = await client.get(url, headers=headers)
        elif method == "POST":
            resp = await client.post(url, headers=headers, json=body_json or {})
        elif method == "PUT":
            resp = await client.put(url, headers=headers, json=body_json or {})
        elif method == "DELETE":
            resp = await client.delete(url, headers=headers)
        elif method == "PATCH":
            resp = await client.patch(url, headers=headers, json=body_json or {})
        else:
            raise HTTPException(405, "Method not allowed")
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    return JSONResponse(content=resp.text, status_code=resp.status_code)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    ip = _get_client_ip(request)
    session = _get_session(ip)
    headers = _make_headers(session)
    body = await request.body()
    try:
        body_json = json.loads(body) if body else None
    except json.JSONDecodeError:
        body_json = None
    if body_json:
        body_json = _fit_context(body_json)

    is_stream = _wants_stream(request, body)

    # BASE_URL already ends with /v1 — strip leading v1/ to avoid double prefix
    if path.startswith("v1/"):
        url = f"{BASE_URL}/{path[3:]}"
    else:
        url = f"{BASE_URL}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    if is_stream and request.method == "POST" and body_json:
        return await _openai_stream_response(url, headers, body_json, ip)

    async with _client(timeout=httpx.Timeout(180.0, connect=15.0)) as client:

        method = request.method.upper()
        if method == "GET":
            resp = await client.get(url, headers=headers)
        elif method == "POST":
            resp = await client.post(url, headers=headers, json=body_json or {})
        elif method == "PUT":
            resp = await client.put(url, headers=headers, json=body_json or {})
        elif method == "DELETE":
            resp = await client.delete(url, headers=headers)
        elif method == "PATCH":
            resp = await client.patch(url, headers=headers, json=body_json or {})
        else:
            raise HTTPException(405, "Method not allowed")

    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    return JSONResponse(content=resp.text, status_code=resp.status_code)
