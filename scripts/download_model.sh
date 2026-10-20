#!/usr/bin/env bash
# Downloads the actual 120B teacher checkpoint (~63.4GB): NOT included in Git.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p models
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
hf download ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf \
  --local-dir models
ls -lh models/gpt-oss-120b-MXFP4.gguf
