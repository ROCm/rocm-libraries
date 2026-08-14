#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Integration test for the combined --use-partial-rms + --dquant-type mxfp8 path.
#
# Steps:
#   1. Run the Tensile pipeline on gemm_partial_rms_mxfp8_quant_k1.yaml.
#   2. Compile a device library (YAML format) with TensileCreateLibrary.
#   3. Run the client for several shapes with both epilogues enabled, verifying
#      the client reference checker reports PASSED for each shape.
#
# Usage:
#   epilogues/scripts/test_client_partialrms_mxfp8.sh [--chip CHIP] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --chip    GPU architecture string (default: gfx950).
#   --client  Path to the tensilelite-client binary (required if not on PATH).
#   --out-dir Scratch directory for generated files (default: /tmp/prms_mx_test_<chip>).
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
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/prms_mx_test_${CHIP}"
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
    if echo "$output" | grep -q "^0,.*,PASSED,"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        echo "$output" | grep "^0," | head -1 | cut -c1-200 >&2 || true
        FAIL=$((FAIL + 1))
    fi
}

# ── Step 1: Run Tensile pipeline ──────────────────────────────────────────────
echo "==> Running Tensile pipeline for $CHIP ..."
TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +e
python3 "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${TENSILELITE_ROOT}/epilogues/yaml/gemm_partial_rms_mxfp8_quant_k1.yaml" \
    "${OUT_DIR}/tensile_out"
TENSILE_EXIT=$?
set -e
if [[ ! -d "${OUT_DIR}/tensile_out/3_LibraryLogic" ]]; then
    echo "error: Tensile pipeline did not produce 3_LibraryLogic (exit ${TENSILE_EXIT})" >&2
    exit 1
fi

# ── Step 2: Build device library ─────────────────────────────────────────────
echo "==> Building device library (YAML format) ..."
python3 -m Tensile.TensileCreateLibrary \
    --library-format=yaml \
    --architecture "$CHIP" \
    "${OUT_DIR}/tensile_out/3_LibraryLogic" \
    "$LIB_DIR" \
    HIP

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
# D is OCP fp8 e4m3; inputs are bf16; compute is f32.
# Both --use-partial-rms and --dquant-type mxfp8 are enabled simultaneously.
# The client reference validates partialBuf (Sigma x^2, pre-gamma), MXScale
# (e8m0 bytes, exact), and D (fp8) against the CPU reference.
# MT0=64, MT1=128 match the solutions in the combined YAML.
COMMON_ARGS=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type BFloat16 --b-type BFloat16 --c-type Float8 --d-type Float8
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --use-partial-rms
    --partial-rms-mt0 64
    --partial-rms-mt1 128
    --dquant-type mxfp8
    --dquant-size-0 1
    --dquant-size-1 32
    --alpha-type Float
    --init-beta  Zero
    --num-benchmarks 1
    --num-elements-to-validate -1
    --device-idx 0
)

# ── Step 3: Aligned shapes with sub-row MX block Q=[1,32] ────────────────────
# Sizes match the Q=[1,32] benchmark group in gemm_partial_rms_mxfp8_quant_k1.yaml
# so that LibraryLogic dispatches the Q=[1,32] kernel.
echo "==> Running client: N=128 M=256 K=64 ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "128,256,1,64" 2>&1)" || true
check "N=128,M=256,K=64" "$OUT"

echo "==> Running client: N=256 M=2048 K=128 ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "256,2048,1,128" 2>&1)" || true
check "N=256,M=2048,K=128" "$OUT"

# ── Step 4: Non-multiple-of-tile M (tokens) ──────────────────────────────────
echo "==> Running client: N=128 M=100 K=64 (non-multiple M) ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "128,100,1,64" 2>&1)" || true
check "N=128,M=100,K=64 non-multiple-M" "$OUT"

# ── Step 5: Larger N_hidden ───────────────────────────────────────────────────
echo "==> Running client: N=256 M=512 K=128 ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "256,512,1,128" 2>&1)" || true
check "N=256,M=512,K=128" "$OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
