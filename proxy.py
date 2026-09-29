"""OpenCode Proxy - OpenAI & Anthropic compatible proxy for OpenCode Zen free tier."""

import asyncio
import ipaddress
import json
import logging
import os
import secrets
import socket
import time
import urllib.parse
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# #region agent log
_DBG_PATH = os.environ.get(
    "DBG_LOG_PATH", "/Users/khangdeptrai/docker/opencode-proxy/.cursor/debug-3ef755.log"
)
_DBG_SID = "3ef755"


def _dbg(location: str, message: str, data: dict) -> None:
    import json as _json
    import time as _time

    try:
        payload = {
            "sessionId": _DBG_SID,
            "runId": os.environ.get("DBG_RUN", "pre-fix"),
            "timestamp": int(_time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
        }
        with open(_DBG_PATH, "a") as fh:
            fh.write(_json.dumps(payload, default=str) + "\n")
            fh.flush()
    except Exception as _e:
        # never let instrumentation break the request path, but make a failed
        # write visible instead of silently losing every log line
        logging.getLogger("opencode-proxy").warning(
            "[DBG] log write failed path=%s err=%r", _DBG_PATH, _e
        )
# #endregion

# #region agent log
# Relative to the app dir so the same default works on the host and inside the
# container (where the app lives at /app, not the host path).
_DBG02_PATH = os.environ.get(
    "DBG_LOG_PATH_02B13A",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-02b13a.log"),
)
_DBG02_SID = "02b13a"


def _dbg02(location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": _DBG02_SID,
            "runId": os.environ.get("DBG_RUN", "pre-fix"),
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
        }
        with open(_DBG02_PATH, "a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception as _e:
        logging.getLogger("opencode-proxy").warning(
            "[DBG02] log write failed path=%s err=%r", _DBG02_PATH, _e
        )
# #endregion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("opencode-proxy")

VERSION = "1.14.0"
OC_VERSION = "2.0.11"
BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")

# Server-side requests must be http/https only, and never reach loopback,
# private, or reserved addresses (SSRF guard). Applied to the configured
# upstream base URL at import time.
def _require_safe_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"unsafe upstream scheme: {parsed.scheme!r} ({url})")
    host = (parsed.hostname or "").lower()
    if host in ("", "localhost", "0.0.0.0", "::1") or host.endswith(".local") or host.endswith(".internal"):
        raise RuntimeError(f"unsafe upstream host: {host!r} ({url})")
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_loopback or ip.is_private or ip.is_reserved or ip.is_link_local or ip.is_multicast:
                raise RuntimeError(f"unsafe upstream host resolves to {ip} ({url})")
    except (socket.gaierror, ValueError, OSError):
        pass  # offline / unresolvable here: rely on the hostname heuristic above
    return url

_require_safe_url(BASE_URL)

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
# 429 is handled separately below (RETRY_429), not by this budget.
UPSTREAM_RETRIES = int(os.environ.get("UPSTREAM_RETRIES", "2"))
UPSTREAM_RETRY_BACKOFF = float(os.environ.get("UPSTREAM_RETRY_BACKOFF", "0.4"))
_RETRYABLE_STATUSES = {502, 503, 504}

# Zen's free tier rate-limits the anonymous `Bearer public` identity per server
# egress IP. A hard quota exhaustion never recovers, but a per-minute RATE
# throttle does — so retry 429s with exponential backoff (bounded) to let
# traffic flow through quiet windows instead of hard-failing instantly. Set
# RETRY_429=0 for the pre-v1.13 behavior of never retrying.
RETRY_429 = int(os.environ.get("RETRY_429", "3"))
RETRY_429_BACKOFF = float(os.environ.get("RETRY_429_BACKOFF", "5.0"))

# Headroom subtracted from STATUS_HOLD_SECS for request round-trips so the
# final 429 still lands inside the streaming header hold.
_429_HOLD_MARGIN = 1.0


def _429_retry_delay(next_attempt: int, spent: float) -> Optional[float]:
    """Backoff before the next 429 retry, capped so cumulative 429 sleeping
    can never reach STATUS_HOLD_SECS: a persistent 429 must surface as a real
    HTTP status inside the streaming header hold, not outlast it and degrade
    into a committed 200 keepalive stream (which then re-POSTs the throttled
    upstream via the pending-retry path). None means the budget is spent —
    stop retrying and surface the 429 now; a 0.0 return is a real zero delay
    (RETRY_429_BACKOFF=0) and the retry still happens. Existing retry
    behavior is preserved whenever the backoff fits the budget."""
    remaining = STATUS_HOLD_SECS - _429_HOLD_MARGIN - spent
    if remaining <= 0:
        return None
    return min(RETRY_429_BACKOFF * (2 ** (next_attempt - 1)), remaining)

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

# Model inventory is DISCOVERED from the upstream at runtime instead of being
# hardcoded here, so Zen additions/retirements are picked up without a code
# change. Cache it and refresh in the background; every resolution path falls
# back to a safe value if discovery has not succeeded yet.
#
# NOTE — reachability: Zen gates the free tier by request origin, so most free
# ids answer 403 FreeTierError through this proxy while working inside the real
# opencode CLI. The gate is server-side and model-intrinsic (headers/session do
# not affect it), so discovery deliberately does NOT filter on the `-free`
# suffix or probe each id: a gated id still resolves and surfaces a truthful
# 403 instead of being silently swapped for a different model.
MODELS_TTL_SECS = float(os.environ.get("MODELS_TTL_SECS", "900"))
MODELS_REFRESH_SECS = float(os.environ.get("MODELS_REFRESH_SECS", "300"))

# Offline bootstrap only. A live refresh replaces this within MODELS_TTL_SECS;
# it exists so a cold start with no network still routes known-free ids.
_BOOTSTRAP_MODELS = ["space-bunny-free"]
_MODEL_CACHE: dict = {"ids": list(_BOOTSTRAP_MODELS), "fetched": 0.0}


def _models_expired() -> bool:
    return (time.monotonic() - _MODEL_CACHE["fetched"]) > MODELS_TTL_SECS


def _known_models() -> list:
    return _MODEL_CACHE["ids"]


async def _refresh_models(force: bool = False) -> list:
    """Fetch the upstream model inventory and cache it.

    Zen's /v1/models mixes free and paid ids; both are kept because the caller
    asked for automatic detection, and the free tier is served under the
    anonymous `Bearer public` identity. Any failure keeps the previous cache
    so routing never degrades to an empty list."""
    if not force and not _models_expired():
        return _MODEL_CACHE["ids"]
    try:
        async with _client(timeout=httpx.Timeout(20.0, connect=10.0)) as c:
            resp = await c.get(f"{BASE_URL}/models", headers=_make_headers(_get_session()))
    except Exception as e:
        logger.warning("[MODELS] refresh failed, keeping %d cached: %s", len(_MODEL_CACHE["ids"]), e)
        return _MODEL_CACHE["ids"]
    if resp.status_code != 200:
        logger.warning("[MODELS] refresh got HTTP %s, keeping %d cached", resp.status_code, len(_MODEL_CACHE["ids"]))
        return _MODEL_CACHE["ids"]
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        logger.warning("[MODELS] refresh returned non-JSON, keeping %d cached", len(_MODEL_CACHE["ids"]))
        return _MODEL_CACHE["ids"]
    ids = sorted({m["id"] for m in (data.get("data") or [])
                  if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]})
    if not ids:
        logger.warning("[MODELS] refresh returned no ids, keeping %d cached", len(_MODEL_CACHE["ids"]))
        return _MODEL_CACHE["ids"]
    _MODEL_CACHE["ids"] = ids
    _MODEL_CACHE["fetched"] = time.monotonic()
    logger.info("[MODELS] discovered %d model(s) upstream", len(ids))
    return ids


# Genuine opencode CLI harness tool definitions (captured 2026-09-18 from a
# real `opencode run` wire body). Zen's free tier only serves requests
# carrying these: bodies without them get 403 FreeTierError ("can only be
# used from within OpenCode"), regardless of headers/session/UA/TLS.
# Client tools keep priority on name collisions; harness names skipped then.
_HARNESS_TOOLS_PATH = os.path.join(os.path.dirname(__file__), "harness_tools.json")
with open(_HARNESS_TOOLS_PATH) as _hf:
    _HARNESS_TOOLS: list[dict] = json.load(_hf)
_HARNESS_TOOL_NAMES = [t["function"]["name"] for t in _HARNESS_TOOLS]
# A model may emit a padded harness tool with any casing (observed live:
# "bash" and "read" for tools defined as "Bash"/"Read"). This set is the
# case-insensitive view of the harness names, used by the padding helpers to
# skip a harness definition when the client already advertises the same tool
# under different casing.
_HARNESS_TOOL_NAMES_LOWER = {n.lower() for n in _HARNESS_TOOL_NAMES}

# Padded tools exist only to pass Zen's free-tier gate — the client harness
# rejects calls to them (Cursor: "unavailable tool 'read'"). Every injected
# definition carries this note so the model leaves them alone.
_HARNESS_IGNORE_NOTE = (
    "Do NOT call this tool: it is not available to you and any call to it is "
    "rejected by the client. It is listed only to satisfy the upstream "
    "provider."
)


def _ignore_oai(tool: dict) -> dict:
    """Copy of an OpenAI-format harness tool with the do-not-call note
    prepended to its description."""
    t = dict(tool)
    fn = dict(t["function"])
    fn["description"] = f"{_HARNESS_IGNORE_NOTE} {fn.get('description') or ''}".strip()
    t["function"] = fn
    return t



def _autocorrected_name(name_map: dict, name):
    """Official client name for a model-emitted tool name: case-insensitive
    exact match only; unmatched and already-correct names pass through."""
    if isinstance(name, str):
        official = name_map.get(name.lower())
        if official and official != name:
            return official
    return name


def _client_tool_name_map(body) -> dict:
    """Map of the client's own advertised tool names: {lower_name: official}.

    Client tools are the request's own definitions — the padded harness tools
    are excluded because only injected definitions carry the do-not-call note
    in their description. Two client tools differing only by case are dropped
    entirely so a rewrite can never be ambiguous. Empty/absent tools -> {}."""
    name_map: dict = {}
    ambiguous: set = set()
    tools = body.get("tools") if isinstance(body, dict) else None
    if not isinstance(tools, list):
        return name_map
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            name, desc = fn["name"], fn.get("description") or ""
        else:
            name, desc = t.get("name"), t.get("description") or ""
        if not isinstance(name, str) or not name:
            continue
        if _HARNESS_IGNORE_NOTE in desc:
            continue
        key = name.lower()
        if key in name_map and name_map[key] != name:
            ambiguous.add(key)
        elif key not in name_map:
            name_map[key] = name
    for key in ambiguous:
        name_map.pop(key, None)
    return name_map


def _autocorrect_tool_calls_json(obj, name_map: dict):
    """Rewrite model-emitted tool-call names that case-insensitively match a
    client-advertised tool (read -> Read) before the response reaches the
    client. Handles the relay shapes in one pass — OpenAI
    (choices[].message.tool_calls and stream chunks delta.tool_calls),
    Anthropic (content[].tool_use) and Responses (output[].function_call).
    Unmatched names, exact matches and empty maps pass through untouched;
    nothing is dropped and finish reasons are untouched."""
    if not name_map or not isinstance(obj, dict):
        return obj
    for ch in obj.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        for holder in (ch.get("message"), ch.get("delta")):
            tcs = holder.get("tool_calls") if isinstance(holder, dict) else None
            if isinstance(tcs, list):
                for tc in tcs:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict) and fn.get("name"):
                        fn["name"] = _autocorrected_name(name_map, fn["name"])
    for block in obj.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
            block["name"] = _autocorrected_name(name_map, block["name"])
    for item in obj.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call" and item.get("name"):
            item["name"] = _autocorrected_name(name_map, item["name"])
    return obj


def _tool_call_autocorrector(name_map: dict):
    """Stateful per-stream step filter rewriting tool-call names to the
    client's official names. step(obj) -> obj handles both relay shapes —
    OpenAI chunks (choices[].delta.tool_calls) and Anthropic events
    (content_block_start.tool_use.name). Nothing is dropped: unknown names,
    exact matches and empty maps pass through unchanged."""
    def step(obj):
        if not name_map or not isinstance(obj, dict):
            return obj
        obj = _autocorrect_tool_calls_json(obj, name_map)
        cb = obj.get("content_block")
        if isinstance(cb, dict) and cb.get("type") == "tool_use" and cb.get("name"):
            cb["name"] = _autocorrected_name(name_map, cb["name"])
        return obj
    return step



def _client_tool_names_lower(tools, anthropic: bool) -> set:
    """Lower-cased names of the client's own advertised tools.

    Collision detection is case-INsensitive: the model may emit any casing
    (`bash`, `Read`) and an upper-case client definition plus its lower-case
    harness twin would present upstream with two tools that differ only by
    case, inviting a call the client cannot execute.
    """
    names: set = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name") if anthropic else ((t.get("function") or {}).get("name"))
        if isinstance(name, str) and name:
            names.add(name.lower())
    return names


def _pad_harness_tools(body: dict) -> dict:
    """Append missing harness tool definitions (OpenAI format) to a
    chat.completions body. Client tools come first and win name collisions.

    The padded definitions exist only to satisfy Zen's free-tier gate. This
    function deliberately does NOT touch `tool_choice`: injecting
    "auto" turned a client that advertised no tools into one that asks for
    tool use, and the model duly called a padded harness tool the client
    cannot execute. Definitions are visible; behaviour stays the caller's.
    """
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return body
    have = _client_tool_names_lower(body.get("tools"), anthropic=False)
    missing = [_ignore_oai(t) for t in _HARNESS_TOOLS
               if t["function"]["name"].lower() not in have]
    if not missing:
        return body
    out = dict(body)
    out["tools"] = list(body.get("tools") or []) + missing
    # #region agent log
    _dbg("proxy.py:_pad_harness_tools:exit", "H1 harness pad appended client-shape tools",
         {"hypothesisId": "H1", "injected": len(missing), "total_tools": len(out["tools"]),
          "client_tool_choice": body.get("tool_choice"), "client_tools": sorted(have)[:12]})
    # #endregion
    return out


def _pad_harness_tools_anthropic(mbody: dict) -> dict:
    """Anthropic-format ({name, description, input_schema}) counterpart."""
    if not isinstance(mbody, dict) or not isinstance(mbody.get("messages"), list):
        return mbody
    have = _client_tool_names_lower(mbody.get("tools"), anthropic=True)
    missing = [
        {"name": t["function"]["name"],
         "description": f"{_HARNESS_IGNORE_NOTE} {t['function'].get('description') or ''}".strip(),
         "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
        for t in _HARNESS_TOOLS
        if t["function"]["name"].lower() not in have
    ]
    if not missing:
        return mbody
    out = dict(mbody)
    out["tools"] = list(mbody.get("tools") or []) + missing
    return out


def _pad_harness_tools_responses(rbody: dict) -> dict:
    """Responses-format ({type: function, name, ...}) counterpart."""
    if not isinstance(rbody, dict):
        return rbody
    have = _client_tool_names_lower(rbody.get("tools"), anthropic=True)
    missing = [
        {"type": "function", "name": t["function"]["name"],
         "description": f"{_HARNESS_IGNORE_NOTE} {t['function'].get('description') or ''}".strip(),
         "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}}}
        for t in _HARNESS_TOOLS
        if t["function"]["name"].lower() not in have
    ]
    if not missing:
        return rbody
    out = dict(rbody)
    out["tools"] = list(rbody.get("tools") or []) + missing
    return out


# Zen free tier models have a 1M-token context; fit oversized requests so the
# upstream never rejects with a 1048576-token context error.
CONTEXT_LIMIT = 1048576
MIN_COMPLETION = 1024

# One token ~ 4 bytes of serialized message. Single source of truth: the
# /v1/messages/count_tokens endpoint and the _fit_context budget MUST agree, or
# Claude Code compacts on a number the proxy never enforced (they diverged 2x
# before this existed: //4 vs //2).
_TOKENS_PER_BYTE_DIVISOR = 4


def _estimate_tokens(messages: list) -> int:
    """Approximate token count for a list of chat/responses items."""
    if not isinstance(messages, list):
        return max(1, len(json.dumps(messages)) // _TOKENS_PER_BYTE_DIVISOR)
    return sum(max(1, len(json.dumps(m)) // _TOKENS_PER_BYTE_DIVISOR) for m in messages)


def _message_cost(items: list) -> int:
    """Budget charged by _fit_context. Same estimator as count_tokens so the
    two can never drift apart again."""
    return _estimate_tokens(items)


def _is_trimmable_sequence(value) -> bool:
    """`messages` and `input` are usually arrays, but the Responses API also
    accepts a bare string or a single item object. Only arrays can be trimmed;
    everything else must pass through byte-for-byte."""
    return isinstance(value, list) and not isinstance(value, (str, bytes))


def _fit_context(body: dict) -> dict:
    # Chat Completions carries the conversation in `messages`; the Responses API
    # carries it in `input` with the system prompt hoisted to `instructions`.
    # Both shapes arrive through this guard (see responses_proxy / catch_all).
    key = "messages" if "messages" in body else ("input" if "input" in body else None)
    if key is None:
        return body
    raw = body.get(key)
    if not _is_trimmable_sequence(raw):
        # String / object / absent `input`: list(raw) would shred a string into
        # single characters, so there is nothing this guard can safely trim.
        return body
    items = list(raw)
    max_tokens = body.get("max_tokens") or 0

    budget = int(CONTEXT_LIMIT * 0.9) - _message_cost(items)

    while budget < MIN_COMPLETION and len(items) > 1:
        # Drop the oldest turn. Always index 0: in Responses form the system
        # prompt lives outside the list in `instructions`, and a leading system
        # turn in chat form is the last thing we want to lose anyway.
        idx = 0
        items.pop(idx)
        # A "tool" message is only valid right after an assistant message with
        # tool_calls; once that assistant is dropped the trailing tool messages
        # become orphans and upstream rejects with 400. Drop them in sync.
        # Responses encodes these as function_call_output items.
        while idx < len(items) and (
            items[idx].get("role") == "tool" or items[idx].get("type") == "function_call_output"
        ):
            items.pop(idx)
        budget = int(CONTEXT_LIMIT * 0.9) - _message_cost(items)

    out = dict(body)
    out[key] = items
    # A non-positive budget means the remaining context is already over the
    # ceiling. Handing upstream max_tokens = max(1024, negative) can only
    # 400, so drop the key and let the upstream apply its own default.
    if budget <= 0:
        out.pop("max_tokens", None)
    elif max_tokens > budget:
        out["max_tokens"] = max(MIN_COMPLETION, budget)
    if out[key] != body.get(key) or out.get("max_tokens") != max_tokens:
        logger.info(
            "[FIT] %s %d->%d max_tokens %s->%s budget=%d",
            key, len(raw), len(out[key]),
            max_tokens, out.get("max_tokens"), budget,
        )
    return out

# Model IDs exposed to clients are prefixed with `oc-` (OpenCode Zen).
# Unprefixed IDs still resolve (back-compat with existing 9Router combos).
# Built from the discovered inventory on each call so a refresh is visible
# immediately, without a restart.
def _model_map() -> dict:
    return {f"oc-{m}": {"id": f"oc-{m}", "object": "model",
                        "created": 1779000000, "owned_by": "opencode"}
            for m in _known_models()}


def _route(model: Optional[str]) -> tuple[str, str]:
    """Resolve a client model ID to (upstream_base, upstream_model_id).

    `oc-X` routes to Zen with ID X; unprefixed IDs resolve against the
    discovered inventory. Unknown IDs fall back to DEFAULT_MODEL
    (long-standing contract)."""
    known = _known_models()
    mid = model if isinstance(model, str) else None
    if mid is not None and mid.startswith("oc-"):
        rest = mid[3:]
        return BASE_URL, rest if rest in known else DEFAULT_MODEL
    if mid in known:
        return BASE_URL, mid
    return BASE_URL, DEFAULT_MODEL


def _map_model(model: Optional[str]) -> str:
    return _route(model)[1]


app = FastAPI(title="OpenCode Proxy", description="OpenAI-compatible API proxy for OpenCode Zen", version=VERSION)

# Populate the discovered model inventory on boot, then keep it fresh so Zen
# additions/retirements are picked up without a restart.
@app.on_event("startup")
async def _startup_refresh_models():
    await _refresh_models(force=True)


@app.on_event("startup")
async def _startup_model_loop():
    async def _loop():
        while True:
            await asyncio.sleep(MODELS_REFRESH_SECS)
            await _refresh_models(force=True)
    asyncio.create_task(_loop())


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
    # A real CLI-minted session is used when configured. The README records a
    # 2026-09-28 probe that disproved the earlier claim that a fabricated
    # ses_ id triggers FreeTierError — it behaves identically either way.
    static = os.environ.get("OPENCODE_SESSION", "").strip()
    if static:
        return static
    now = time.time()
    # Keys are client-supplied (X-Forwarded-For), so the map can be inflated with
    # unlimited distinct keys inside the freshness window. Evict expired entries
    # first, then — if the map is still over cap — drop the oldest entries by ts
    # so MAX_SESSIONS is a real bound rather than a best-effort trigger.
    if len(_user_sessions) >= MAX_SESSIONS:
        for stale in [u for u, s in _user_sessions.items() if now - s["ts"] > 1800]:
            del _user_sessions[stale]
        overflow = len(_user_sessions) - MAX_SESSIONS + 1
        if overflow > 0:
            for user_key, _ in sorted(_user_sessions.items(), key=lambda kv: kv[1]["ts"])[:overflow]:
                del _user_sessions[user_key]
    sess = _user_sessions.get(user)
    if not sess or now - sess["ts"] > 1800:
        sess = {"id": _gen_id("ses"), "ts": now}
        _user_sessions[user] = sess
    return sess["id"]


def _make_headers(session_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer public",
        "User-Agent": f"opencode/{OC_VERSION} ai-sdk/provider-utils/4.0.23 runtime/bun/1.4.2",
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
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "Rate limit exceeded"
        if not isinstance(data, dict):
            return "Rate limit exceeded"
        err = data.get("error")
        # `error` is a string, a list or absent in the wild; only a dict carries
        # a message. Unguarded .get() here raised AttributeError and turned a
        # retryable 429 into an opaque 500.
        if isinstance(err, dict):
            msg = err.get("message", "")
        else:
            msg = data.get("message", "")
        return msg or "Rate limit exceeded"
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
    msg = str(err.get("message") or "").strip()
    throttled = (
        "FreeUsageLimitError" in etype
        or "rate_limit" in etype.lower()
        or "rate limit" in msg.lower()
        or code == 429
        or str(code) == "429"
    )
    if throttled:
        return 429, msg or "Rate limit exceeded"
    # Zen gates most free models by request origin and answers 403 FreeTierError
    # ("free tier can only be used from within OpenCode"). That is a policy gate,
    # not an outage: reporting it as 502 makes callers retry and mark the model
    # broken, so surface the real 403.
    if "FreeTierError" in etype or "free tier" in msg.lower():
        return 403, msg or "Free tier restricted to OpenCode"
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
        # Anthropic Messages frames (union-alpha): content deltas are live
        # (text/thinking/tool-input), message_stop is done. message_start and
        # ping frames are neutral — they carry no output.
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
        if rtype == "content_block_delta":
            d = obj.get("delta") or {}
            if d.get("text") or d.get("thinking") or "partial_json" in d:
                info["live"] = True
        elif rtype == "content_block_start":
            cb = obj.get("content_block") or {}
            if cb.get("type") in ("text", "thinking", "tool_use"):
                info["live"] = True
        elif rtype == "message_stop":
            info["done"] = True
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
        # #region agent log
        if buffered:
            _dbg02("proxy.py:_probe_sse_stream:return",
                   "H2/H3 probe buffered terminal frames the relay will not credit",
                   {"hypothesisId": "H2,H3", "timed_out": timed_out, "live": st["live"],
                    "done": st["done"], "err": bool(st["err"]), "bad": st["bad"],
                    "n_buffered": len(buffered),
                    "buffered_first_200": [b[:200] for b in buffered[:3]]})
        # #endregion
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
    attempt_429 = 0
    spent_429 = 0.0
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
        if resp.status_code == 429 and attempt_429 < RETRY_429:
            delay = _429_retry_delay(attempt_429 + 1, spent_429)
            if delay is None:
                break  # hold budget spent: surface the real 429 now
            await resp.aread()
            await resp.aclose()
            await client.aclose()
            attempt_429 += 1
            spent_429 += delay
            logger.warning("[RETRY 429] %s attempt %d/%d in %.1fs", url, attempt_429, RETRY_429, delay)
            await asyncio.sleep(delay)
            continue
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

    name_map = _client_tool_name_map(body)
    autocorrect = _tool_call_autocorrector(name_map)
    # #region agent log
    _dbg02("proxy.py:_zen_stream_upstream:post-probe",
           "H1/H2/H3 state entering the relay after the probe",
           {"hypothesisId": "H1,H2,H3",
            "n_buffered": len(buffered),
            "buffered_first_200": [b[:200] for b in buffered[:3]],
            "n_name_map": len(name_map)})
    # #endregion
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
        if name_map and b'"tool_calls"' in frame:
            parts = []
            replaced = False
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    parts.append(line)
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    parts.append(line)
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    parts.append(line)
                    continue
                parts.append(b"data: " + json.dumps(autocorrect(obj), separators=(",", ":")).encode())
                replaced = True
            if replaced:
                frame = b"\n".join(parts)
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
                # #region agent log
                _dbg02("proxy.py:_zen_stream_upstream:no-done",
                       "H1 raw relay reached EOF without a [DONE] frame",
                       {"hypothesisId": "H1",
                        "n_buffered": len(buffered),
                        "relay_saw_finish": state["saw_finish"],
                        "relay_saw_usage": state["saw_usage"],
                        "relay_saw_tool": state["saw_tool"],
                        "relay_saw_content": state["saw_content"],
                        "relay_bytes": state["relayed"]})
                # #endregion
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

# Fallback for unknown / retired model IDs. Must be a model the proxy can
# actually reach — mimo-v2.5-free was the old default but now returns 403
# FreeTierError, which broke the entire unknown-ID fallback contract.
DEFAULT_MODEL = "space-bunny-free"

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


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
    attempt_429 = 0
    spent_429 = 0.0
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
        if resp.status_code == 429 and attempt_429 < RETRY_429:
            delay = _429_retry_delay(attempt_429 + 1, spent_429)
            if delay is None:
                break  # hold budget spent: surface the real 429 now
            await resp.aread()
            await resp.aclose()
            await client.aclose()
            attempt_429 += 1
            spent_429 += delay
            logger.warning("[RETRY 429] %s attempt %d/%d in %.1fs", url, attempt_429, RETRY_429, delay)
            await asyncio.sleep(delay)
            continue
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

    name_map = _client_tool_name_map(body)
    autocorrect = _tool_call_autocorrector(name_map)

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
                    obj = autocorrect(obj)
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
                    obj = autocorrect(obj)
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
# (muse-spark-1.2-contributor-free was dropped in v1.13.0: retired upstream.)

RESPONSES_MODELS = {"muse-spark-1.3-contributor-free"}

# Models that only speak Anthropic's Messages API upstream, served via the
# _chat_via_messages bridge below (mirroring RESPONSES_MODELS).
# Empty as of v1.13.0: union-alpha was the only entry and was retired upstream
# (401 "Model union-alpha is not supported" on 2026-09-28). The bridge itself is
# kept — a new Anthropic-native free model only needs to be added here.
MESSAGES_MODELS: set[str] = set()


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
    return _pad_harness_tools_responses(out)


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
        # #region agent log
        _dbg02("proxy.py:_responses_events_to_chat:no-terminal",
               "H1b Responses bridge aborted despite a possibly probe-buffered terminal",
               {"hypothesisId": "H1b", "emitted": state["emitted"]})
        # #endregion
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
    out = _responses_to_chat_json(_autocorrect_tool_calls_json(_safe_json(resp), _client_tool_name_map(body)), body.get("model"))
    choice = out["choices"][0]
    if not choice["message"].get("content") and not choice["message"].get("tool_calls") and out["usage"]["completion_tokens"] == 0:
        raise HTTPException(502, "no usable content (empty completion)")
    return out


# --- Messages API bridge (union-alpha) ---
# union-alpha only speaks Anthropic's Messages API upstream (verified
# 2026-09-17: HTTP 500 on /chat/completions and /responses, 200 on
# /messages). Translate OpenAI chat/completions both ways, mirroring the
# RESPONSES_MODELS bridge above.

_ANTH_STOP_TO_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


def _chat_to_anthropic(body: dict) -> dict:
    """Translate an OpenAI chat.completions request into an Anthropic messages request."""
    system_parts: list[str] = []
    messages: list[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "developer"):
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content or [] if isinstance(b, dict)
            )
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            text = content if isinstance(content, str) else json.dumps(content)
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": text},
            ]})
            continue
        if role == "assistant":
            blocks: list[dict] = []
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content or [] if isinstance(b, dict)
            )
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or _gen_id("toolu"),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
        else:
            parts = []
            for b in content or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "image_url":
                    u = (b.get("image_url") or {}).get("url", "")
                    if u.startswith("data:"):
                        try:
                            header, data = u.split(",", 1)
                            media = header.split(";")[0].split(":")[1]
                            parts.append({"type": "image", "source": {
                                "type": "base64", "media_type": media, "data": data}})
                        except (ValueError, IndexError):
                            pass
                    elif u:
                        parts.append({"type": "image", "source": {"type": "url", "url": u}})
            if parts:
                messages.append({"role": "user", "content": parts})

    out: dict = {"model": body.get("model"), "messages": messages}
    if system_parts:
        out["system"] = "\n\n".join(system_parts)
    out["max_tokens"] = body.get("max_tokens") or 1024
    for key in ("temperature", "top_p", "stream"):
        if body.get(key) is not None:
            out[key] = body.get(key)
    tools_out = []
    for t in body.get("tools") or []:
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function") or {}
            tools_out.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
    if tools_out:
        out["tools"] = tools_out
    tc = body.get("tool_choice")
    if tc == "required":
        out["tool_choice"] = {"type": "any"}
    elif tc == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif isinstance(tc, dict):
        if tc.get("type") == "function":
            out["tool_choice"] = {"type": "tool", "name": (tc.get("function") or {}).get("name", "")}
    return _pad_harness_tools_anthropic(out)


def _anthropic_msg_to_chat(resp_obj: dict, model: str) -> dict:
    """Translate a non-streaming Anthropic message into chat.completions JSON."""
    text_parts: list[str] = []
    tool_calls = []
    for block in resp_obj.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") or _gen_id("call"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    stop = resp_obj.get("stop_reason", "end_turn")
    usage = resp_obj.get("usage") or {}
    return {
        "id": resp_obj.get("id") or _gen_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": _ANTH_STOP_TO_FINISH.get(stop, "stop")}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


async def _anthropic_dicts_to_oai_dicts(events: AsyncIterator[dict], model: str) -> AsyncIterator[dict]:
    """Translate parsed Anthropic message events into chat.completion.chunk dicts.

    Feeds both output directions: serialized to SSE for OpenAI clients, and
    into _anthropic_events for Anthropic-speaking clients."""
    state = {"role_sent": False, "tool_idx": 0, "emitted": False, "finish": None, "usage": None}

    def chunk(delta: Optional[dict] = None, finish: Optional[str] = None,
              usage: Optional[dict] = None, role: bool = False) -> dict:
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
            "model": model,
            "choices": [choice],
        }
        if usage is not None:
            frame["usage"] = usage
        return frame

    async for ev in events:
        if not isinstance(ev, dict):
            continue
        t = ev.get("type", "")
        if t == "message_start":
            continue
        elif t == "content_block_start":
            cb = ev.get("content_block") or {}
            if cb.get("type") == "tool_use":
                state["emitted"] = True
                yield chunk(delta={"tool_calls": [{
                    "index": state["tool_idx"],
                    "id": cb.get("id") or _gen_id("call"),
                    "type": "function",
                    "function": {"name": cb.get("name", ""), "arguments": ""},
                }]})
                state["tool_idx"] += 1
            elif cb.get("type") in ("text", "thinking") and not state["role_sent"]:
                state["role_sent"] = True
                yield chunk(role=True)
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            dtype = d.get("type")
            if dtype == "text_delta" and d.get("text"):
                state["emitted"] = True
                if not state["role_sent"]:
                    state["role_sent"] = True
                    yield chunk(role=True)
                yield chunk(delta={"content": d["text"]})
            elif dtype == "input_json_delta" and state["tool_idx"] > 0:
                # Only a tool_use content_block_start advances tool_idx; an
                # orphan delta (malformed upstream) must not emit a tool chunk
                # with index -1 and no id/name.
                yield chunk(delta={"tool_calls": [{
                    "index": state["tool_idx"] - 1,
                    "function": {"arguments": d.get("partial_json", "")},
                }]})
            elif dtype in ("thinking_delta", "signature_delta") and d.get("thinking"):
                state["emitted"] = True
                if not state["role_sent"]:
                    state["role_sent"] = True
                    yield chunk(role=True)
                yield chunk(delta={"reasoning_content": d["thinking"]})
        elif t == "message_delta":
            state["finish"] = _ANTH_STOP_TO_FINISH.get(
                (ev.get("delta") or {}).get("stop_reason"), "stop")
            out_tokens = (ev.get("usage") or {}).get("output_tokens", 0)
            state["usage"] = {"prompt_tokens": 0, "completion_tokens": out_tokens,
                              "total_tokens": out_tokens}
        elif t == "message_stop":
            break
        elif t == "error":
            err = ev.get("error")
            _, emsg = _classify_upstream_error(err if isinstance(err, dict) else {})
            raise UpstreamDead(emsg)
        # ping and other lifecycle events carry no output — skip
    if state["finish"] is None and state["usage"] is None:
        raise UpstreamDead("Upstream stream ended before completion")
    if not state["emitted"]:
        raise UpstreamDead("no usable content (empty completion)")
    yield chunk(finish=state["finish"] or "stop", usage=state["usage"])


async def _anthropic_sse_to_chat(events: AsyncIterator[dict], model: str) -> AsyncIterator[bytes]:
    """Serialize translated Anthropic events into chat.completion.chunk SSE."""
    async for d in _anthropic_dicts_to_oai_dicts(events, model):
        yield f"data: {json.dumps(d, separators=(',', ':'))}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _chat_via_messages(headers: dict, body: dict, client_ip: str):
    """Serve a chat.completions request for a MESSAGES_MODELS model by calling
    zen's /messages endpoint and translating back."""
    mbody = _chat_to_anthropic(body)
    url = f"{BASE_URL}/messages"

    if mbody.get("stream"):
        task = asyncio.create_task(_zen_stream_parsed(url, headers, mbody, client_ip))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=STATUS_HOLD_SECS)
        except asyncio.TimeoutError:
            async def relay_msg(chunks):
                async for out in _anthropic_sse_to_chat(chunks, body.get("model")):
                    yield out
            return StreamingResponse(
                _pending_relay_with_retry(
                    task,
                    lambda: _zen_stream_parsed(url, headers, mbody, client_ip),
                    PENDING_RETRY_SECS,
                    relay_msg,
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
            _keepalive_wrapper(_anthropic_sse_to_chat(chunks, body.get("model"))),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    resp = await _post_with_retry(url, headers, mbody)
    if resp.status_code != 200:
        logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
        rl = _is_rate_limit_error(resp.status_code, resp.content)
        if rl:
            raise HTTPException(429, f"Rate limit: {rl}")
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")
    out = _anthropic_msg_to_chat(_autocorrect_tool_calls_json(_safe_json(resp), _client_tool_name_map(body)), body.get("model"))
    choice = out["choices"][0]
    if not choice["message"].get("content") and not choice["message"].get("tool_calls") and out["usage"]["completion_tokens"] == 0:
        raise HTTPException(502, "no usable content (empty completion)")
    return out


async def _messages_anthropic_stream_response(url: str, headers: dict, mbody: dict,
                                              client_ip: str, model: str) -> StreamingResponse:
    """Header-hold stream for Anthropic-speaking clients on MESSAGES_MODELS:
    bridge through OpenAI dicts, re-translate to Anthropic events on the way out."""
    task = asyncio.create_task(_zen_stream_parsed(url, headers, mbody, client_ip))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=STATUS_HOLD_SECS)
    except asyncio.TimeoutError:
        async def relay_anth(chunks):
            async for ev in _anthropic_events(
                    _anthropic_dicts_to_oai_dicts(chunks, model), model):
                yield ev
        return StreamingResponse(
            _pending_relay_with_retry(
                task,
                lambda: _zen_stream_parsed(url, headers, mbody, client_ip),
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
    return StreamingResponse(
        _anthropic_events(_anthropic_dicts_to_oai_dicts(chunks, model), model),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


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


def _upstream_path(path: str) -> str:
    """Map a client request path onto the upstream path under BASE_URL.

    BASE_URL already ends in /v1, so every LEADING v1/ segment is stripped —
    a single check left /v1/v1/models becoming .../v1/v1/models upstream.
    Segments in the middle are preserved (/v1/foo/v1/bar -> foo/v1/bar)."""
    while path.startswith("v1/"):
        path = path[3:]
    return path


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION, "models": len(_model_map())}


def _safe_json(resp: httpx.Response):
    """Parse an upstream JSON body; a 200 with garbage must surface as a clean
    502, not an unhandled crash (plain-text 'Internal Server Error' 500)."""
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        logger.warning("[ZEN] non-JSON body (%s): %s", resp.status_code, resp.text[:200])
        raise HTTPException(502, "Upstream returned a non-JSON body")


def _reject_empty_completion(choice: dict) -> None:
    """A 200 whose single choice carries no text, no tool_calls and no output
    tokens is an empty completion, not an answer. Both bridge paths already
    raised 502 here; the two native non-stream paths silently returned a
    well-formed empty success, which callers (Claude Code) book as a completed
    turn. Tool-only answers are exempt — they legitimately have null content."""
    message = choice.get("message") or {}
    if message.get("content") or message.get("tool_calls"):
        return
    raise HTTPException(502, "no usable content (empty completion)")


async def _post_with_retry(url: str, headers: dict, json_body: dict) -> httpx.Response:
    """POST with transient-failure retry (502/503/504/connect errors). The
    response body is fully read before returning, so each attempt's client is
    closed and only the final response escapes."""
    attempt = 0
    attempt_429 = 0
    spent_429 = 0.0
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
        if resp.status_code == 429 and attempt_429 < RETRY_429:
            delay = _429_retry_delay(attempt_429 + 1, spent_429)
            if delay is None:
                return resp  # hold budget spent: surface the real 429 now
            attempt_429 += 1
            spent_429 += delay
            logger.warning("[RETRY 429] %s attempt %d/%d in %.1fs", url, attempt_429, RETRY_429, delay)
            await asyncio.sleep(delay)
            continue
        if resp.status_code in _RETRYABLE_STATUSES and attempt < UPSTREAM_RETRIES:
            attempt += 1
            delay = UPSTREAM_RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning("[RETRY] upstream %s, attempt %d/%d in %.1fs", resp.status_code, attempt, UPSTREAM_RETRIES + 1, delay)
            await asyncio.sleep(delay)
            continue
        return resp


@app.get("/v1/models")
@app.api_route("/v1/models", methods=["POST", "PUT", "DELETE", "PATCH"])
async def list_models():
    # Served from the proxy's own cache, never proxied upstream, so this route
    # cannot leak the paid ids in Zen's full catalog. Each entry is tagged with
    # the `oc-` prefix. Registered for non-GET verbs too: the catch-all would
    # otherwise forward POST /v1/models to a non-existent upstream endpoint.
    return {"object": "list", "data": list(_model_map().values())}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await _json_body(request)
    # #region agent log
    _dbg("proxy.py:chat_completions:entry", "H1/H2/H3 request arrived",
         {"hypothesisId": "H1,H2,H3", "stream": body.get("stream"),
          "has_tool_choice": "tool_choice" in body,
          "n_tools": len(body.get("tools") or []),
          "tool_names": [((t.get("function") or {}).get("name") or t.get("name"))
                         for t in (body.get("tools") or []) if isinstance(t, dict)][:20]})
    # #endregion
    body = _fit_context(body)
    body = _pad_harness_tools(body)
    ip = _get_client_ip(request)
    session = _get_session(ip)
    stream = body.get("stream", False)

    logger.info("[OAI] %s %s stream=%s", ip, body.get("model", "?"), stream)

    base, upstream_id = _route(body.get("model"))
    headers = _make_headers(session)
    # Always send an explicit model: _route resolved the default for an absent
    # or unknown id, and the old `is not None` guard dropped it — upstream saw
    # a model-less body and rejected it.
    body["model"] = upstream_id
    url = f"{base}/chat/completions"

    if upstream_id in RESPONSES_MODELS:
        return await _chat_via_responses(headers, body, ip)

    if upstream_id in MESSAGES_MODELS:
        return await _chat_via_messages(headers, body, ip)

    if stream:
        return await _openai_stream_response(url, headers, body, ip)

    resp = await _post_with_retry(url, headers, body)

    if resp.status_code != 200:
        logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
        rl = _is_rate_limit_error(resp.status_code, resp.content)
        if rl:
            raise HTTPException(429, f"Rate limit: {rl}")
        raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")

    out = _autocorrect_tool_calls_json(_safe_json(resp), _client_tool_name_map(body))
    # #region agent log
    _dbg("proxy.py:chat_completions:nonstream-exit", "H2/H3 name map from POST-PAD body",
         {"hypothesisId": "H2,H3", "name_map": _client_tool_name_map(body),
          "n_map": len(_client_tool_name_map(body)),
          "emitted_names": [((tc.get("function") or {}).get("name"))
                            for ch in (out.get("choices") or [])
                            for tc in ((ch.get("message") or {}).get("tool_calls") or [])]})
    # #endregion
    _reject_empty_completion((out.get("choices") or [{}])[0])
    return out


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await _json_body(request)
    # #region agent log
    _dbg("proxy.py:anthropic_messages:entry", "H2/H3 Anthropic request arrived",
         {"hypothesisId": "H2,H3", "stream": body.get("stream"),
          "has_tool_choice": "tool_choice" in body,
          "n_tools": len(body.get("tools") or []),
          "tool_names": [t.get("name") for t in (body.get("tools") or [])
                         if isinstance(t, dict)][:20]})
    # #endregion
    ip = _get_client_ip(request)
    session = _get_session(ip)
    stream = body.get("stream", False)
    base, model = _route(body.get("model"))
    headers = _make_headers(session)

    logger.info("[ANTH] %s %s stream=%s", ip, body.get("model", "?"), stream)

    oai_body = _anthropic_to_openai(body)
    oai_body = _fit_context(oai_body)
    oai_body = _pad_harness_tools(oai_body)
    # #region agent log
    _dbg("proxy.py:anthropic_messages:post-pad", "H2/H3/H5/H7 name map after translation+padding",
         {"hypothesisId": "H2,H3,H5,H7",
          "map_from_raw_body": _client_tool_name_map(body),
          "map_from_padded_oai": _client_tool_name_map(oai_body),
          "n_map_padded": len(_client_tool_name_map(oai_body)),
          "n_tools": len(oai_body.get("tools") or []),
          "tool_choice_sent_upstream": oai_body.get("tool_choice"),
          "tool_choice_in_request": body.get("tool_choice")})
    # #endregion
    url = f"{base}/chat/completions"

    if model in MESSAGES_MODELS:
        # Anthropic-native upstream: bridge OAI back to Messages format and
        # re-translate the response to Anthropic events/JSON for the caller.
        mbody = _chat_to_anthropic(oai_body)
        murl = f"{base}/messages"
        if stream:
            mbody["stream"] = True
            return await _messages_anthropic_stream_response(murl, headers, mbody, ip, model)
        resp = await _post_with_retry(murl, headers, mbody)
        if resp.status_code != 200:
            logger.warning("[ZEN] status=%s body=%s", resp.status_code, resp.text[:300])
            rl = _is_rate_limit_error(resp.status_code, resp.content)
            if rl:
                raise HTTPException(429, f"Rate limit: {rl}")
            raise HTTPException(resp.status_code, f"Upstream error: {resp.text[:200]}")
        return _openai_to_anthropic(_anthropic_msg_to_chat(_autocorrect_tool_calls_json(_safe_json(resp), _client_tool_name_map(body)), model), model)

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

    out = _autocorrect_tool_calls_json(_safe_json(resp), _client_tool_name_map(oai_body))
    # #region agent log
    _dbg("proxy.py:anthropic_messages:nonstream-exit", "H2/H3/H5 native path name map source",
         {"hypothesisId": "H2,H3,H5", "n_map_padded": len(_client_tool_name_map(oai_body)),
          "n_map_raw": len(_client_tool_name_map(body)),
          "emitted_names": [((tc.get("function") or {}).get("name"))
                            for ch in (out.get("choices") or [])
                            for tc in ((ch.get("message") or {}).get("tool_calls") or [])]})
    # #endregion
    _reject_empty_completion((out.get("choices") or [{}])[0])
    return _openai_to_anthropic(out, model)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    body = await _json_body(request)
    oai_body = _anthropic_to_openai(body)
    # Charge exactly what a real /v1/messages request sends: that route runs
    # _fit_context -> _pad_harness_tools, so ~6k tokens of harness definitions
    # ride along on every call. Estimating the client's tools alone under-counted
    # by 98% and made callers compact far too late.
    counted = _pad_harness_tools(_fit_context(oai_body))
    input_tokens = (_estimate_tokens(counted.get("messages") or [])
                    + _estimate_tokens(counted.get("tools") or []))
    # #region agent log
    _dbg02("proxy.py:count_tokens",
           "H5 tool definitions silently dropped by the OpenAI translation",
           {"hypothesisId": "H5",
            "n_tools_in": len(body.get("tools") or []),
            "n_tools_out": len(oai_body.get("tools") or []),
            "n_messages_out": len(oai_body.get("messages") or []),
            "deflated": bool(body.get("tools")) and not oai_body.get("tools"),
            "n_tools_counted": len(counted.get("tools") or []),
            "input_tokens_returned": input_tokens})
    # #endregion
    return {"input_tokens": input_tokens}


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
        body_json = _pad_harness_tools_responses(body_json)
        # #region agent log
        _dbg("proxy.py:responses_proxy:post-pad", "H4 name map after responses padding",
             {"hypothesisId": "H4", "n_map": len(_client_tool_name_map(body_json)),
              "n_tools": len(body_json.get("tools") or []),
              "map": dict(list(_client_tool_name_map(body_json).items())[:8])})
        # #endregion
        # Same model contract as /v1/chat/completions: strip the oc- prefix and
        # fall back to DEFAULT_MODEL for unknown IDs. Without this the prefix
        # reached Zen verbatim (invalid model) and unknown IDs were not mapped.
        if isinstance(body_json.get("model"), str):
            body_json["model"] = _map_model(body_json["model"])
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
        # A lying content-type (200 + <html>Internal Server Error) must be a
        # clean 502 like every other guarded path, not a bare 500.
        try:
            content = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            logger.warning("[ZEN] non-JSON body (%s): %s", resp.status_code, resp.text[:200])
            raise HTTPException(502, "Upstream returned a non-JSON body")
        return JSONResponse(content=_autocorrect_tool_calls_json(content, _client_tool_name_map(body_json)), status_code=resp.status_code)
    # A non-JSON body must reach the client as-is: JSONResponse would relabel it
    # application/json and hand the caller a JSON *string* of the original text.
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=ct or None)


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
        before_pad = body_json
        body_json = _pad_harness_tools(body_json)
        # #region agent log
        _dbg02("proxy.py:catch_all:post-pad",
               "H6/H7 catch_all padding + name map on a possibly non-OpenAI body",
               {"hypothesisId": "H6,H7", "path": path,
                "body_keys": sorted(body_json.keys())[:14],
                "n_tools_in": len(before_pad.get("tools") or []),
                "n_tools_out": len(body_json.get("tools") or []),
                "n_map": len(_client_tool_name_map(body_json)),
                "tools_are_responses_shape": bool(body_json.get("tools"))
                    and isinstance(body_json["tools"][0], dict)
                    and "function" not in body_json["tools"][0],
                "n_map_raw_before_pad": len(_client_tool_name_map(before_pad))})
        # #endregion
        # Same model contract as /v1/chat/completions: strip the oc- prefix and
        # fall back to DEFAULT_MODEL for unknown IDs.
        if isinstance(body_json.get("model"), str):
            body_json["model"] = _map_model(body_json["model"])

    is_stream = _wants_stream(request, body)

    # BASE_URL already ends with /v1 — strip every leading v1/ segment so
    # /v1/v1/models does not become .../v1/v1/models upstream.
    url = f"{BASE_URL}/{_upstream_path(path)}"
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
    # #region agent log
    _dbg02("proxy.py:catch_all:passthrough",
           "H8 passthrough response shape (non-JSON body becomes a JSON string)",
           {"hypothesisId": "H8", "path": path, "status": resp.status_code,
            "content_type": ct[:80], "is_json_ct": "application/json" in ct,
            "body_preview": resp.text[:160] if not ("application/json" in ct) else None})
    # #endregion
    if "application/json" in ct:
        try:
            content = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            logger.warning("[ZEN] non-JSON body (%s): %s", resp.status_code, resp.text[:200])
            raise HTTPException(502, "Upstream returned a non-JSON body")
        return JSONResponse(content=_autocorrect_tool_calls_json(content, _client_tool_name_map(body_json)), status_code=resp.status_code)
    # A non-JSON body must reach the client as-is: JSONResponse would relabel it
    # application/json and hand the caller a JSON *string* of the original text.
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=ct or None)
