#!/usr/bin/env bash
# Build dnn-convert-shapes.pyz using shiv in an isolated venv.
# Honors BUNDLE_OUT_DIR for out-of-source builds; defaults to script dir.
# Honors BUNDLE_VENV_DIR for a shared shiv venv; defaults to ${OUT_DIR}/.venv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${BUNDLE_OUT_DIR:-$SCRIPT_DIR}"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/dnn-convert-shapes.pyz"
VENV_DIR="${BUNDLE_VENV_DIR:-${OUT_DIR}/.venv}"
SHIV_VERSION="1.0.8"

PYTHON="${PYTHON:-python3}"

if [[ ! -x "${VENV_DIR}/bin/shiv" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --disable-pip-version-check "shiv==${SHIV_VERSION}"
fi

"${VENV_DIR}/bin/shiv" -c dnn-convert-shapes -o "$OUT" "$PKG_DIR"
echo "Built: $OUT"
