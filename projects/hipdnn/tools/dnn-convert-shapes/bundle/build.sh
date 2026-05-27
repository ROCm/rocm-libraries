#!/usr/bin/env bash
# Build dnn-convert-shapes.pyz using shiv.
#
# Output:    <bundle-dir>/dnn-convert-shapes.pyz
# Override:  DNN_BUNDLE_OUT=/path/to/output.pyz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT="${DNN_BUNDLE_OUT:-${SCRIPT_DIR}/dnn-convert-shapes.pyz}"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found on PATH" >&2
  exit 1
fi

"$PYTHON" -m pip install --quiet shiv
"$PYTHON" -m shiv \
  --console-script dnn-convert-shapes \
  --output-file "$OUT" \
  --compressed \
  "$PKG_DIR"

echo "Built: $OUT"
