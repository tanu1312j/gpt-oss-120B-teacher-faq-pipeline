#!/usr/bin/env bash
# Run on a Graviton ARM64 Amazon Linux 2023 EC2 host.
set -euo pipefail
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This setup is intended for Graviton aarch64 (ARM64)" >&2
  exit 1
fi
sudo dnf install -y git gcc gcc-c++ cmake make libcurl-devel python3 python3-pip curl
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-mcpu=native \
  -DCMAKE_CXX_FLAGS=-mcpu=native -DGGML_NATIVE=ON
cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip 'huggingface_hub>=1.0,<2.0'
echo "Built llama-server at $LLAMA_DIR/build/bin/llama-server"
