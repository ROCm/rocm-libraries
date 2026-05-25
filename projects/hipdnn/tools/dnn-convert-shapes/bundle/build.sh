#!/usr/bin/env bash
# Build dnn-convert-shapes.pyz using shiv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT="${SCRIPT_DIR}/dnn-convert-shapes.pyz"

pip install shiv --quiet
shiv -c dnn-convert-shapes -o "$OUT" "$PKG_DIR"
echo "Built: $OUT"
