#!/bin/sh
set -e

echo "Starting OpenCode Proxy v1.5.0 on 0.0.0.0:5381"
echo "Backend: $OPENCODE_BASE_URL"
echo "Broke mode: $OPENCODE_BROKE"
echo "Endpoints:"
echo "  OpenAI:    POST /v1/chat/completions"
echo "  Anthropic: POST /v1/messages"
echo "  Models:    GET  /v1/models"

exec uvicorn proxy:app --host 0.0.0.0 --port 5381 --access-log
