# opencode-proxy

OpenAI & Anthropic compatible API proxy for the [OpenCode Zen](https://opencode.ai/docs/zen/).
## Run

```bash
docker compose up -d --build   # from the parent directory containing docker-compose.yml
curl localhost:5381/health     # {"status":"ok","version":"1.14.0","models":86}
```

The container binds **127.0.0.1 only**. The proxy has no authentication of its own, so publishing
the port would expose an open model relay to the network.

Endpoints:

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat format (stream + non-stream) |
| `POST /v1/messages` | Anthropic messages format (stream + non-stream), translated both ways |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/responses`, catch-all `/*` | Passthrough with the same stream/error handling |
| `GET /v1/models` | Discovered upstream inventory, `oc-`-prefixed (see below) |
| `GET /health` | Liveness + version |

> **Note:** this means `/v1/models` lists the **full discovered catalog, paid ids included** — 86
> entries at time of writing, 13 of which end in `-free`. The `/v1/models` route is served from the
> proxy's own cache and never proxied upstream, but the cache itself is unfiltered. Requesting a
> paid id resolves and then fails upstream on `Bearer public`; it does not yield a working paid
> model. Filter client-side on the `-free` suffix if you need only the free set.

## Upstream routing

Client model IDs are accepted in both forms on every route (including `/v1/responses` and the
catch-all passthroughs):

| Client model ID | Goes to | Upstream sees |
|---|---|---|
| `oc-X` (X in the discovered inventory) | Zen | `X` |
| unprefixed `X` | resolved against the discovered inventory | `X` |

Unknown IDs fall back to `DEFAULT_MODEL` (`space-bunny-free`). Zen is an anonymous free tier
(`Bearer public`). The full relay stack — probe-before-commit, keepalives, truncation detection,
error classification — applies to every request.

`MESSAGES_MODELS` is empty, so the Anthropic-native bridge (`_chat_via_messages`) is inactive; it
needs only a new model added to that set to be re-enabled.

## Harness tool padding

Zen's free tier only serves requests carrying the 13-tool harness set defined in
`harness_tools.json` (`bash`, `edit`, `get_goal`, `glob`, `grep`, `read`, `skill`, `task`,
`todowrite`, `update_goal`, `webfetch`, `websearch`, `write`). Every request is padded up to that
set before it goes upstream, in all three body shapes (OpenAI, Anthropic, Responses).

Client definitions always come first and win an **exact** name collision. Case is significant:
a client advertising `Read` does not suppress the padded `read`, because the free-tier gate counts
specific definitions rather than a tool count, and folding case would leave the request short of the
required set. The case-variant twin is handled on the way back out — `_client_tool_name_map` plus
`_autocorrect_tool_calls_json` rewrite a model-emitted `read` to the client's `Read` before the
response is delivered.

If Zen answers a padded request with `403 FreeTierError`, the proxy retries once with the client's
own tools (log line `[PAD]`). This covers the OpenCode CLI's utility requests such as title
generation, which it sends with no tools: with a fresh session, Zen accepted that
prompt only unpadded and refused the padded copy (live test 2026-10-21).

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `OPENCODE_BASE_URL` | `https://opencode.ai/inference/v1` | Upstream. Validated at import against an SSRF guard (http/https only, never loopback/private/reserved). `zen/v1` is a newer endpoint that requires an account key; `inference/v1` serves the anonymous free tier (`Bearer public`). |
| `OPENCODE_SESSION` | – | Forces one session id (`ses_…`) for ALL traffic. Leave unset: Zen counts the free allowance per session id, so a pinned id makes every client share one allowance that runs out within hours (live test 2026-10-21: pinned id → `429`, fresh id → `200`). Unset, the proxy forwards the client's own `x-opencode-session` / `x-session-id` when it is well formed, else mints one CLI-format id per conversation (client address + first user turn), kept across model switches. Zen rejects a malformed id with `403 FreeTierError`; valid ids look like `ses_` + 12 hex + 14 base62 characters. |
| `SESSION_TTL_SECS` | `43200` | Idle seconds after which a conversation's session id is forgotten (the next request starts a new session). |
| `MODELS_REFRESH_SECS` | `300` | Background discovery refresh interval |
| `MODELS_TTL_SECS` | `900` | Age at which a cached inventory counts as stale |
| `STATUS_HOLD_SECS` | `15` | Seconds to hold headers probing a 200 stream's health |
| `UPSTREAM_RETRIES` | `2` | Retries transient upstream `502/503/504` and connect errors (`backoff * 2**(attempt-1)`) |
| `PENDING_RETRY_SECS` | `120` | Budget for post-commit upstream re-requests, served behind keepalives once HTTP 200 is already committed |
| `OUTBOUND_PROXY` | – | Optional egress proxy (`socks5://…`). Falls back to `HTTPS_PROXY`, then `HTTP_PROXY`; empty means direct connection. |
| `HARD_LIMIT_SECS` | `60` | A `429` whose `Retry-After` exceeds this is an exhausted free allowance, not a throttle. That session's use of that model is blocked: no retries, and its requests fail fast with a real `429` naming the reset time. Other sessions and models are unaffected; `/health` lists blocked models under `limited_models` (`sessions` blocked, soonest `resets_in` seconds). |
| `MODEL_RECHECK_SECS` | `900` | While a session's model is blocked, one request re-probes upstream after this many seconds (or at reset, if sooner). |

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest test_proxy.py   # no network needed (mocked transport)
```
