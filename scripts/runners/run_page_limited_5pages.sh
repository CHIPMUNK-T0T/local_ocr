#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/ocr_venv/bin/python"
RUNNER="$ROOT_DIR/model_load_and_infer_test/page_limited_model_test.py"
PAGE_LIMIT=5
DPI=250

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("docling" "glm" "dots" "paddle")
fi

for model in "${MODELS[@]}"; do
  case "$model" in
    glm)
      "$PYTHON_BIN" "$RUNNER" --model glm --page-limit "$PAGE_LIMIT" --dpi "$DPI"
      ;;
    dots)
      "$PYTHON_BIN" "$RUNNER" --model dots --page-limit "$PAGE_LIMIT" --dpi "$DPI" --max-length 16384 --max-new-tokens 2048
      ;;
    docling)
      "$PYTHON_BIN" "$RUNNER" --model docling --page-limit "$PAGE_LIMIT" --dpi "$DPI"
      ;;
    paddle)
      "$PYTHON_BIN" "$RUNNER" --model paddle --page-limit "$PAGE_LIMIT" --dpi "$DPI"
      ;;
    *)
      echo "Unsupported model: $model" >&2
      echo "Usage: $0 [docling] [glm] [dots] [paddle]" >&2
      exit 1
      ;;
  esac
done
