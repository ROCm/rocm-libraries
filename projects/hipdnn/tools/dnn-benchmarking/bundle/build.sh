#!/usr/bin/env bash
# Build dnn-benchmark.pyz using shiv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT="${SCRIPT_DIR}/dnn-benchmark.pyz"

pip install shiv --quiet
shiv -c dnn-benchmark -o "$OUT" "$PKG_DIR"
echo "Built: $OUT"
