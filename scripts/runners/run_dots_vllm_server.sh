#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/venvs/dots_vllm_venv"
PYTHON_BIN="$VENV_DIR/bin/python"
VLLM_BIN="$VENV_DIR/bin/vllm"
MODEL_PATH="${DOTS_MODEL_PATH:-$ROOT_DIR/models/DotsOCR_1_5}"
HOST="${DOTS_VLLM_HOST:-127.0.0.1}"
PORT="${DOTS_VLLM_PORT:-8000}"
GPU_DEVICE="${DOTS_VLLM_GPU:-0}"
GPU_MEMORY_UTILIZATION="${DOTS_VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
SERVED_MODEL_NAME="${DOTS_VLLM_SERVED_MODEL_NAME:-model}"
MAX_MODEL_LEN="${DOTS_VLLM_MAX_MODEL_LEN:-16384}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python in $VENV_DIR" >&2
  exit 1
fi

if [ ! -x "$VLLM_BIN" ]; then
  echo "Missing vllm in $VENV_DIR" >&2
  exit 1
fi

if [ ! -e "$MODEL_PATH" ]; then
  echo "Missing model path: $MODEL_PATH" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface/home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT_DIR/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$ROOT_DIR/.cache/huggingface/modules}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_MODULES_CACHE"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --chat-template-content-format string \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code
