# opencode-proxy

OpenAI & Anthropic compatible API proxy for the [OpenCode Zen](https://opencode.ai/docs/zen/) free tier.

One small FastAPI service — point any OpenAI- or Anthropic-speaking tool at it and use Zen's free models
from a single endpoint.

## ⚠️ Free-tier origin gate (read this first)

As of the 2026-09-28 live probe, Zen restricts its free tier by **request origin**. Every free model
except `space-bunny-free` answers:

```json
{"type":"error","error":{"type":"FreeTierError",
 "message":"Error from provider (Console): OpenCode's free tier can only be used from within OpenCode"}}
```

when reached through this proxy — while the *same* model works from inside the real `opencode` CLI.
The gate is server-side and model-intrinsic. It is **not** affected by the spoofed `x-opencode-*`
headers or by `OPENCODE_SESSION`: a fabricated `ses_` id behaves identically to a CLI-minted one
(verified across four sessions, three rounds each, deterministic).

So today **`space-bunny-free` is the only model usable from a third-party client**, and it is the
`DEFAULT_MODEL`. The remaining seven are kept in the pool because they are genuinely free and the
gate is expected to be lifted. A gated model now surfaces a real **403**, not a 502, so callers can
tell a policy gate from an outage.

## Run

```bash
docker compose up -d --build   # from the parent directory containing docker-compose.yml
curl localhost:5381/health     # {"status":"ok","version":"1.13.1","models":8}
```

Endpoints:

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat format (stream + non-stream) |
| `POST /v1/messages` | Anthropic messages format (stream + non-stream), translated both ways |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/responses`, catch-all `/*` | Passthrough with the same stream/error handling |
| `GET /v1/models` | Free models only, `oc-`-prefixed |
| `GET /health` | Liveness + version |

## Use with Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:5381
export ANTHROPIC_AUTH_TOKEN=any          # the proxy ignores the caller's token
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000   # Zen's 1M window; avoids the auto-compact warning
claude --model oc-space-bunny-free
```

`ANTHROPIC_API_KEY` is not required — the proxy authenticates upstream as the anonymous
`Bearer public` free tier. Verified working end-to-end (Claude Code 2.1.236).

## Upstream routing

`/v1/models` lists **only the free models**, each prefixed with `oc-`
(e.g. `oc-mimo-v2.6-flash-free`). Requests accept both forms, on every route
(including the `/v1/responses` and catch-all passthroughs):

| Client model ID | Goes to | Upstream sees |
|---|---|---|
| `oc-X` (X in Zen's free list) | Zen | `X` |
| unprefixed `X` | resolved against the free list | `X` |

Unknown IDs fall back to `DEFAULT_MODEL` (`space-bunny-free`). Zen is an
anonymous free tier (`Bearer public`). The full relay stack — probe-before-commit,
keepalives, truncation detection, error classification — applies to every request.

**Breaking (v1.13.0):** the free pool was refreshed against a live probe. Five entries are gone
upstream and now follow the unknown-ID contract above instead of being advertised:
`union-alpha` (401 *"Model union-alpha is not supported"*), `mimo-v2.5-free`,
`jev-1.13-free`, `deepseek-v4-flash-free`, `muse-spark-1.2-contributor-free`.
`longcat-2.5-preview-free` is new. Because `mimo-v2.5-free` was the old `DEFAULT_MODEL` *and* is
now gated, every unknown-model request failed with a 403; the default is now `space-bunny-free`.

`MESSAGES_MODELS` is empty — `union-alpha` was its only member. The Anthropic-native bridge
(`_chat_via_messages`) is kept and needs only a new model added to that set to be re-enabled.

## v1.13.1 — correctness fixes

Found by an audit of the v1.13.0 diff; the test suite was green throughout, so each fix
ships with the test that was missing:

- A Responses body whose `input` is a bare **string** or a single item object was shredded
  into a list of single characters by the context guard and forwarded upstream. Only arrays
  are trimmed now.
- Non-stream `/v1/messages` and `/v1/chat/completions` returned `200` with an empty
  `content`/`null` on an empty completion; both now raise `502` like every other path.
  Tool-only answers stay exempt.
- A `429` whose `error` is a string (or a non-dict JSON body) raised `AttributeError` and
  surfaced as `500`, defeating the retry chain. It now correctly surfaces as `429`.
- The `/v1/responses` and catch-all passthroughs never applied the model contract: the `oc-`
  prefix reached Zen verbatim and unknown IDs got no `DEFAULT_MODEL` fallback.
- Passthrough routes returned a bare `500` on a lying upstream `content-type`; now a clean `502`.
- `count_tokens` and the fit budget now share one estimator (they disagreed 2x, so callers
  compacted on a number the proxy never enforced).
- A request already over the context ceiling no longer gets an invented `max_tokens`.
- `harness_tools.json` is tracked — the image build needs it and `proxy.py` reads it at import.

## Zen quirks this proxy handles

Hard-won against the real thing — each has a regression test:

- **Post-`[DONE]` trailing frames** — Zen appends `data: {"choices":[],"cost":"0"}` after the real
  `[DONE]`; downstreams parse the empty choices as a dead model. The relay stops at the true terminal frame.
- **Markers inside model output** — `data: [DONE]` or `FreeUsageLimitError` appearing as ordinary
  *content* (e.g. the model writing SSE-handling code) must not terminate or fail the stream. Terminal
  and error detection is frame- and JSON-aware, never byte-matching.
- **Frame-boundary keepalives** — `: keepalive` comments keep middle-hops alive during long silent TTFT,
  but only ever between complete `\n\n` frames; splicing one into a half-buffered JSON line corrupts it.
- **Truncation detection** — Zen kills long free-tier streams mid-response (no finish/usage/DONE).
  Those surface as an SSE error event (`502`) so routers retry/fallback instead of showing a half answer
  as success. Tool-call streams are exempt (muse-spark legitimately omits finish_reason for tools).
- **Probe-before-commit** (`STATUS_HOLD_SECS`, default 15) — a 200 stream that turns out dead within the
  hold window surfaces as a *real* HTTP status instead of 200+SSE, so callers can distinguish outages
  from quota exhaustion and lock models for the right reason.
- **Honest error classification** — genuine throttle becomes `429`, the free-tier origin gate becomes
  `403`, everything else stays `5xx`. In-band `finish_reason:"error"` frames are caught too.
- **Junk-frame drop** — empty-choices heartbeats and cost trailers are dropped (usage-bearing frames kept,
  so token accounting stays truthful).
- **Muse-spark's no-`[DONE]` termination** gets a clean synthetic terminal; reasoning-only streams are
  reported distinctly in diagnostics (`reasoning=True content=False bytes=…`).
- Client disconnects cancel the upstream request; cancellations release connections (`BaseException`
  cleanup); oversized contexts are fitted to Zen's 1M limit without orphaning tool messages.
  `count_tokens` and the fit budget share one estimator, so callers can trust the number
  they are given. A Responses body whose `input` is a bare string or a single item object
  passes through untouched — only arrays are trimmed.
- **Empty completions are errors, not answers** — a 200 carrying no text, no tool calls and
  no output tokens surfaces as a `502` on every path, stream and non-stream, so a caller can
  never mistake a blank turn for a finished one.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/v1` | Zen upstream |
| `OPENCODE_SESSION` | – | Optional session id (`ses_…`) sent as `x-opencode-session`. **Note:** earlier versions documented this as *required*, because a fabricated id was reported to trigger `FreeTierError`. The 2026-09-28 probe disproved that — a made-up `ses_` id behaves identically to a CLI-minted one, for every model including the gated ones. It does not affect reachability. |
| `STATUS_HOLD_SECS` | `15` | Seconds to hold headers probing a 200 stream's health |
| `UPSTREAM_RETRIES` | `2` | Retries transient upstream `502/503/504` and connect errors (`backoff * 2**(attempt-1)`) |
| `UPSTREAM_RETRY_BACKOFF` | `0.4` | Base seconds for the transient-retry backoff |
| `PENDING_RETRY_SECS` | `120` | Budget for post-commit upstream re-requests, served behind keepalives once HTTP 200 is already committed |
| `OUTBOUND_PROXY` | – | Optional egress proxy (`socks5://…`). Falls back to `HTTPS_PROXY`, then `HTTP_PROXY` (httpx does not read those automatically for programmatic clients); empty means direct connection. |
| `RETRY_429` | `3` | Retries a Zen `429` this many times with exponential backoff before surfacing it. Zen's free tier rate-limits the anonymous `Bearer public` identity per server egress IP, so a per-minute throttle recovers on retry while a hard quota does not. Set `0` to never retry (`429` surfaces instantly). |
| `RETRY_429_BACKOFF` | `5.0` | Base seconds for the `429` backoff (`backoff * 2**(attempt-1)`). |

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest test_proxy.py   # no network needed (mocked transport)
```

License: MIT.
