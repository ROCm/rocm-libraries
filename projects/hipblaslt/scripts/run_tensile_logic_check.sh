#!/usr/bin/env bash
# Optional Unix wrapper: run the Python logic check script (use .py on Windows).
# From hipblaslt root: ./scripts/run_tensile_logic_check.sh [LIBLOGIC_PATH]
# Prefers .venv/bin/python if present, otherwise falls back to python3.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIPBLASLT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/run_tensile_logic_check.py"

if [[ -x "${HIPBLASLT_ROOT}/.venv/bin/python" ]]; then
  exec "${HIPBLASLT_ROOT}/.venv/bin/python" "${PY_SCRIPT}" "$@"
else
  exec python3 "${PY_SCRIPT}" "$@"
fi
