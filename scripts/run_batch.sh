#!/usr/bin/env bash
# One-time local batch. No public / Internet-facing inference endpoint.
# Examples: bash scripts/run_batch.sh --limit 2
#           bash scripts/run_batch.sh --input data/my_faq.csv
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL_FILE="${MODEL_FILE:-$ROOT/models/gpt-oss-120b-MXFP4.gguf}"
SERVER_BIN="${SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
PORT="${PORT:-8080}"
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Missing teacher GGUF: $MODEL_FILE. Run scripts/download_model.sh first." >&2
  exit 1
fi
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing llama-server binary: $SERVER_BIN. Run scripts/setup_graviton.sh first." >&2
  exit 1
fi
mkdir -p output logs
"$SERVER_BIN" \
  --model "$MODEL_FILE" \
  --alias gpt-oss-120b \
  --host 127.0.0.1 --port "$PORT" --no-webui \
  --ctx-size 4096 --threads "$(nproc)" --parallel 1 --n-gpu-layers 0 \
  --jinja --reasoning-effort low --reasoning-format auto \
  > logs/llama-server.log 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
READY=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "llama-server exited before readiness; see logs/llama-server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "$READY" != 1 ]]; then
  echo "llama-server not ready; see logs/llama-server.log" >&2
  exit 1
fi
python3 src/generate_teacher.py --base-url "http://127.0.0.1:${PORT}" "$@"
