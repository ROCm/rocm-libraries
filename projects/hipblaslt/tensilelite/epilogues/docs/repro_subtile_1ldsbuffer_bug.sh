#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Reproducer for the Subtile 1LDSBuffer=1 swap-mask bug.
#
# Runs a plain bf16 GEMM tuning sweep with 1LDSBuffer=1 on the Subtile path
# (UseSubtileImpl=True, DirectToLds=0, SIA=3). Expects FAILED results on
# every kernel — the LDS double-buffer swap mask always points one
# ldsTotalSize past the allocated region, corrupting the MFMA accumulator
# from the second macro-tile onward.
#
# Usage (from the tensilelite root):
#   source ~/.tensile/bin/activate
#   bash epilogues/docs/repro_subtile_1ldsbuffer_bug.sh
#
# To override the output directory:
#   OUT_DIR=/tmp/my_repro bash epilogues/docs/repro_subtile_1ldsbuffer_bug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILELITE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML="${SCRIPT_DIR}/repro_subtile_1ldsbuffer_bug.yaml"
OUT_DIR="${OUT_DIR:-/tmp/repro_subtile_1ldsbuffer_bug}"

# Resolve Python: prefer the project venv, fall back to python3.
_VENV_PYTHON="${TENSILELITE_ROOT}/../../../.tensile/bin/python"
if [[ -x "$_VENV_PYTHON" ]]; then
    PYTHON="${PYTHON:-${_VENV_PYTHON}}"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "==> Subtile 1LDSBuffer=1 swap-mask bug reproducer"
echo "    Tensile root : ${TENSILELITE_ROOT}"
echo "    YAML         : ${YAML}"
echo "    Output dir   : ${OUT_DIR}"
echo "    Python       : ${PYTHON}"
echo ""
echo "    Expected outcome: FAILED on every kernel."
echo "    Root cause: the Subtile GR/LR swap mask is computed as"
echo "      swapMask = base XOR (base + ldsTotalSize)"
echo "    and applied at every macro-tile boundary regardless of 1LDSBuffer."
echo "    With 1LDSBuffer=1, only one LDS slot is allocated, so the swap"
echo "    target is outside the reserved region — the MFMA accumulator"
echo "    fills with data from a neighbouring workgroup's LDS."
echo ""

rm -rf "${OUT_DIR}"
TENSILE_LOG=$(mktemp)
# Tensile exits non-zero when any kernel fails validation (exit code 8).
# That is the expected outcome for this reproducer, so capture and ignore it.
"${PYTHON}" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" "${YAML}" "${OUT_DIR}" \
    >"${TENSILE_LOG}" 2>&1 || true

# The client prints one CSV result line per (solution, problemSize) containing
# either ',PASSED,' or ',FAILED,' as the status field.
PASS=$(grep -c ',PASSED,' "${TENSILE_LOG}" 2>/dev/null || true)
FAIL=$(grep -c ',FAILED,' "${TENSILE_LOG}" 2>/dev/null || true)

echo "==> Benchmark results:"
grep -oP 'MT\d+x\d+x\d+[^,]*(?=,PASSED,)' "${TENSILE_LOG}" 2>/dev/null | \
    while read -r k; do echo "  PASS: $k"; done || true
grep -oP 'MT\d+x\d+x\d+[^,]*(?=,FAILED,)' "${TENSILE_LOG}" 2>/dev/null | \
    while read -r k; do echo "  FAIL: $k"; done || true
rm -f "${TENSILE_LOG}"

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
if [[ $FAIL -gt 0 && $PASS -eq 0 ]]; then
    echo "==> Bug reproduced: all kernels failed as expected."
    exit 0
elif [[ $FAIL -gt 0 ]]; then
    echo "==> Partial reproduction: some kernels failed."
    exit 1
else
    echo "==> Bug NOT reproduced: all kernels passed (unexpected)."
    exit 2
fi
