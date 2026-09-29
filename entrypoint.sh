#!/bin/sh
set -e

VERSION="$(python3 -c 'import proxy; print(proxy.VERSION)' 2>/dev/null || echo dev)"
echo "Starting OpenCode Proxy v${VERSION} on 0.0.0.0:5381"
echo "Backend: $OPENCODE_BASE_URL"
echo "Endpoints:"
echo "  OpenAI:    POST /v1/chat/completions"
echo "  Anthropic: POST /v1/messages"
echo "  Models:    GET  /v1/models"

exec uvicorn proxy:app --host 0.0.0.0 --port 5381 --access-log
