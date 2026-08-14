#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run a tuning round: build a merged device library covering all epilogue variants.
#
# Variants (in entry order from the benchmark YAML):
#   0 — plain GEMM
#   1 — GEMM + PartialRMS (no residual add)
#   2 — GEMM + PartialRMS + residual add
#
# Usage:
#   tune_all_epilogues.sh [--yaml PATH] [--fast] [--chip CHIP]
#                         [--out-dir DIR] [--library-format yaml|msgpack]
#                         [--client PATH]
#
# Arguments:
#   --yaml PATH          Benchmark YAML (default: tune_all_epilogues.yaml).
#   --fast               Use tune_all_epilogues_fast.yaml (quick CI run).
#   --chip CHIP          GPU architecture (default: gfx950).
#   --out-dir DIR        Output root (default: /tmp/tune_all_epilogues_<chip>).
#   --library-format FMT yaml or msgpack (default: msgpack).
#   --client PATH        tensilelite-client binary; enables smoke tests (GPU required).

set -euo pipefail

TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve Python: prefer venv next to the hipblaslt repo, fall back to python3.
_VENV_PYTHON="${TENSILELITE_ROOT}/../../../.tensile/bin/python"
if [[ -x "$_VENV_PYTHON" ]]; then
    PYTHON="${PYTHON:-${_VENV_PYTHON}}"
else
    PYTHON="${PYTHON:-python3}"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
YAML="${TENSILELITE_ROOT}/epilogues/yaml/tune_all_epilogues.yaml"
CHIP="gfx950"
OUT_DIR=""
LIBRARY_FORMAT="msgpack"
CLIENT=""
FAST=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)            YAML="$2";           shift 2 ;;
        --fast)            FAST=1;              shift ;;
        --chip)            CHIP="$2";           shift 2 ;;
        --out-dir)         OUT_DIR="$2";        shift 2 ;;
        --library-format)  LIBRARY_FORMAT="$2"; shift 2 ;;
        --client)          CLIENT="$2";         shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/tune_all_epilogues_${CHIP}"
fi

# Resolve YAML.
if [[ -n "$FAST" ]]; then
    YAML="${TENSILELITE_ROOT}/epilogues/yaml/tune_all_epilogues_fast.yaml"
fi

LOGIC_DIR="${OUT_DIR}/3_LibraryLogic"
LIB_DIR="${OUT_DIR}/library"

# ── Phase 1+2: benchmark + logic generation ───────────────────────────────────
echo "==> Running Tensile benchmark + logic generation ..."
"$PYTHON" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" "$YAML" "$OUT_DIR"

# Check logic files were produced.
LOGIC_COUNT=$(find "$LOGIC_DIR" -name "*.yaml" 2>/dev/null | wc -l)
if [[ "$LOGIC_COUNT" -lt 3 ]]; then
    echo "error: expected at least 3 LibraryLogic YAMLs, got $LOGIC_COUNT" >&2
    exit 1
fi
echo "==> Generated $LOGIC_COUNT LibraryLogic YAML(s)"

# ── Phase 3: compile all variants into one library ────────────────────────────
rm -rf "$LIB_DIR"
echo "==> Building device library (format: ${LIBRARY_FORMAT}) ..."
"$PYTHON" "${TENSILELITE_ROOT}/Tensile/bin/TensileCreateLibrary" \
    --library-format="${LIBRARY_FORMAT}" \
    --architecture "$CHIP" \
    "$LOGIC_DIR" "$LIB_DIR" HIP

# ── Report artifact paths ─────────────────────────────────────────────────────
ARTIFACT_DIR="${LIB_DIR}/library/${CHIP}"
echo ""
echo "==> Library artifacts in ${ARTIFACT_DIR}:"
find "$ARTIFACT_DIR" \
    \( -name "*.co" -o -name "*.hsaco" -o -name "*.dat" -o -name "*.dat.zlib" \) \
    -printf "  %p\n" 2>/dev/null \
    || ls "$ARTIFACT_DIR" 2>/dev/null \
    || true

# ── Optional: smoke-test with client ─────────────────────────────────────────
if [[ -z "$CLIENT" ]]; then
    exit 0
fi

echo ""
echo "==> Running smoke tests with client: ${CLIENT} ..."

# Locate the non-lazy library file and its code object.
LIB_YAML="$(find "$ARTIFACT_DIR" -maxdepth 1 -name "TensileLibrary_BB_*.yaml" \
                ! -name "*lazy*" 2>/dev/null | head -1 || true)"
if [[ -z "$LIB_YAML" ]]; then
    LIB_YAML="$(find "$ARTIFACT_DIR" -maxdepth 1 -name "TensileLibrary_BB_*.dat" \
                    2>/dev/null | head -1 || true)"
fi
if [[ -z "$LIB_YAML" ]]; then
    echo "warning: could not find non-lazy TensileLibrary_BB_* file; skipping smoke tests" >&2
    exit 0
fi
LIB_CO="${LIB_YAML%.*}.co"
if [[ ! -f "$LIB_CO" ]]; then
    echo "warning: code object not found: $LIB_CO; skipping smoke tests" >&2
    exit 0
fi

# ── Helper ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0

check() {
    local label="$1"
    local output="$2"
    if echo "$output" | grep -q "^0,.*,PASSED,"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        echo "$output" | grep "^0," | head -1 | cut -c1-200 >&2
        FAIL=$((FAIL + 1))
    fi
}

# Common client arguments shared by all variants.
COMMON_ARGS=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type BFloat16 --b-type BFloat16 --c-type BFloat16 --d-type BFloat16
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --num-benchmarks 1
    --num-elements-to-validate 64
    --device-idx 0
)

# Variant 0 — plain GEMM.
OUT="$("$CLIENT" "${COMMON_ARGS[@]}" --problem-size 512,512,1,512 2>&1)" || true
check "GemmPlain (no epilogue)" "$OUT"

# Variant 1 — PartialRMS, no residual add.
OUT="$("$CLIENT" "${COMMON_ARGS[@]}" --problem-size 512,512,1,512 \
        --use-partial-rms 2>&1)" || true
check "PartialRMS (no residual)" "$OUT"

# Variant 2 — PartialRMS + residual add.
OUT="$("$CLIENT" "${COMMON_ARGS[@]}" --problem-size 512,512,1,512 \
        --use-partial-rms --partial-rms-residual-add 2>&1)" || true
check "PartialRMS (residual add)" "$OUT"

echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
