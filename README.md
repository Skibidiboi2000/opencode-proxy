# opencode-proxy

OpenAI & Anthropic compatible API proxy for [OpenCode Zen](https://opencode.ai/docs/zen/) free tier.
One small FastAPI service — point any OpenAI- or Anthropic-speaking tool at it and use Zen's free models
(DeepSeek V4 Flash, big-pickle, MiniMax M2.5, MiMo V2.5, Nemotron, Qwen 3.6 Plus, muse-spark, x-preview, hy3).

## Run

```bash
docker compose up -d --build   # from the parent directory containing docker-compose.yml
curl localhost:5381/health     # {"status":"ok","version":"1.5.0","models":9}
```

Endpoints:

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat format (stream + non-stream) |
| `POST /v1/messages` | Anthropic messages format (stream + non-stream), translated both ways |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/responses`, catch-all `/*` | Passthrough with the same stream/error handling |
| `GET /v1/models` | Live model list from Zen (static fallback on failure) |
| `GET /health` | Liveness + version |

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
- **Probe-before-commit** (`STATUS_HOLD_SECS`, default 20) — a 200 stream that turns out dead within the
  hold window surfaces as a *real* HTTP status instead of 200+SSE, so callers can distinguish outages
  from quota exhaustion and lock models for the right reason.
- **Honest error classification** — only genuine throttle conditions become `429`; everything else stays
  `5xx`. In-band `finish_reason:"error"` frames are caught too.
- **Junk-frame drop** — empty-choices heartbeats and cost trailers are dropped (usage-bearing frames kept,
  so token accounting stays truthful).
- **Muse-spark's no-`[DONE]` termination** gets a clean synthetic terminal; reasoning-only streams are
  reported distinctly in diagnostics (`reasoning=True content=False bytes=…`).
- Client disconnects cancel the upstream request; cancellations release connections (`BaseException`
  cleanup); oversized contexts are fitted to Zen's 1M limit without orphaning tool messages.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/v1` | Upstream |
| `OPENCODE_BROKE` | `false` | Serve static model list without hitting upstream |
| `STATUS_HOLD_SECS` | `20` | Seconds to hold headers probing a 200 stream's health |
| `OUTBOUND_PROXY` | – | Optional egress proxy (`socks5://…`) |

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest test_proxy.py   # 52 tests, no network needed (mocked transport)
```

License: MIT.
