#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Integration test for the tensilelite-client --use-partial-rms path.
#
# Steps:
#   1. Run the Tensile pipeline to generate LibraryLogic YAMLs.
#   2. Compile a device library (YAML format) with TensileCreateLibrary.
#   3. Run the client without residual add, verify PASSED.
#   4. Run the client with residual add, verify PASSED.
#
# Usage:
#   epilogues/scripts/test_client_partialrms.sh [--chip CHIP] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --chip   GPU architecture string (default: gfx950).
#   --client Path to the tensilelite-client binary (required if not on PATH).
#   --out-dir Scratch directory for generated files (default: /tmp/prms_test_<chip>).
#
# Prerequisites:
#   * Python environment with TensileLite installed (or run from tensilelite/ root).
#   * An AMD GPU matching the chip argument must be present and accessible.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CHIP="gfx950"
CLIENT_BIN=""
OUT_DIR=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chip)    CHIP="$2";       shift 2 ;;
        --client)  CLIENT_BIN="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/prms_test_${CHIP}"
fi

LIB_DIR="${OUT_DIR}/library_yaml"

# ── Resolve client binary ─────────────────────────────────────────────────────
if [[ -z "$CLIENT_BIN" ]]; then
    if command -v tensilelite-client &>/dev/null; then
        CLIENT_BIN="$(command -v tensilelite-client)"
    else
        echo "error: tensilelite-client not found; pass --client <path>" >&2
        exit 1
    fi
fi

if [[ ! -x "$CLIENT_BIN" ]]; then
    echo "error: client binary not executable: $CLIENT_BIN" >&2
    exit 1
fi

# ── Helper ────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0

check() {
    local label="$1"
    local output="$2"
    # Look for PASSED in the data rows; INVALID or absence means failure.
    if echo "$output" | grep -q "^0,.*,PASSED,"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        # Show the data row for diagnosis.
        echo "$output" | grep "^0," | head -1 | cut -c1-200 >&2
        FAIL=$((FAIL + 1))
    fi
}

# ── Step 1: Run Tensile pipeline ──────────────────────────────────────────────
echo "==> Running Tensile pipeline for $CHIP ..."
TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${TENSILELITE_ROOT}/epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml" \
    "${OUT_DIR}/tensile_out"

# ── Step 2: Build device library ─────────────────────────────────────────────
echo "==> Building device library (YAML format) ..."
python3 -m Tensile.TensileCreateLibrary \
    --library-format=yaml \
    --architecture "$CHIP" \
    "${OUT_DIR}/tensile_out/3_LibraryLogic" \
    "$LIB_DIR" \
    HIP

# Locate the non-lazy YAML and its code object.
LIB_YAML="$(find "${LIB_DIR}/library/${CHIP}" -maxdepth 1 \
    -name "TensileLibrary_BB_*.yaml" ! -name "*lazy*" | head -1)"
if [[ -z "$LIB_YAML" ]]; then
    echo "error: could not find non-lazy library YAML under ${LIB_DIR}/library/${CHIP}" >&2
    exit 1
fi
LIB_CO="${LIB_YAML%.yaml}.co"
if [[ ! -f "$LIB_CO" ]]; then
    echo "error: code object not found: $LIB_CO" >&2
    exit 1
fi
echo "  Library YAML: $LIB_YAML"
echo "  Code object : $LIB_CO"

# ── Common client arguments ───────────────────────────────────────────────────
COMMON_ARGS=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type BFloat16 --b-type BFloat16 --c-type BFloat16 --d-type BFloat16
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --problem-size 4096,4096,1,4096
    --num-benchmarks 1
    --num-elements-to-validate 128
    --device-idx 0
)

# ── Step 3: PartialRMS without residual add ───────────────────────────────────
echo "==> Running client: --use-partial-rms (no residual) ..."
OUT_NORES="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --use-partial-rms 2>&1)" || true
check "--use-partial-rms (no residual)" "$OUT_NORES"

# ── Step 4: PartialRMS with residual add ──────────────────────────────────────
echo "==> Running client: --use-partial-rms --partial-rms-residual-add ..."
OUT_RES="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --use-partial-rms --partial-rms-residual-add 2>&1)" || true
check "--use-partial-rms --partial-rms-residual-add" "$OUT_RES"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
