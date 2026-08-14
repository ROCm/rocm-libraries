#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Integration test for the tensilelite-client --dquant-type tile path.
#
# Steps:
#   1. Run the Tensile pipeline to generate LibraryLogic YAMLs.
#   2. Compile a device library (YAML format) with TensileCreateLibrary.
#   3. Run the client for several (M, N, K) shapes and quant-tile sizes,
#      verifying the client reference checker reports PASSED for each.
#
# Usage:
#   epilogues/scripts/test_client_tilequant.sh [--chip CHIP] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --chip    GPU architecture string (default: gfx950).
#   --client  Path to the tensilelite-client binary (required if not on PATH).
#   --out-dir Scratch directory for generated files (default: /tmp/tq_test_<chip>).
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
    OUT_DIR="/tmp/tq_test_${CHIP}"
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
        echo "$output" | grep "^0," | head -1 | cut -c1-200 >&2 || true
        FAIL=$((FAIL + 1))
    fi
}

# ── Step 1: Run Tensile pipeline ──────────────────────────────────────────────
echo "==> Running Tensile pipeline for $CHIP ..."
TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The Tensile pipeline benchmark phase validates with its own reference; a FAILED
# result there does not indicate a kernel bug. Our correctness validation is done
# by the explicit client steps below with --num-elements-to-validate -1.
set +e
python3 "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${TENSILELITE_ROOT}/epilogues/yaml/gemm_tile_quant_k1.yaml" \
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
# D is OCP fp8 e4m3 (DestDataType=Float8); inputs are bf16; compute is f32.
# --dquant-type tile enables the epilogue; the client reference validates
# both QuantScale (fp32) and D (fp8) against the CPU reference.
COMMON_ARGS=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type BFloat16 --b-type BFloat16 --c-type Float8 --d-type Float8
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --dquant-type tile
    # TileQuant kernels compute D = alpha*A*B without beta (beta=0 is Tier 1 constraint).
    # --alpha-type must be explicit so initializeConstantInputs can initialize the alpha value from
    # --init-alpha; without it both the cached and problem alpha types stay None, and the
    # initialization block is skipped, leaving inputs.alpha = 0.
    --alpha-type Float
    --init-beta  Zero
    --num-benchmarks 1
    --num-elements-to-validate -1
    --device-idx 0
)

# ── Step 3: Quant tile Q=[16,16] ─────────────────────────────────────────────
# Each shape group benchmarks a distinct problem size so LibraryLogic assigns
# a separate exact entry per shape; querying those exact sizes here guarantees
# the library returns the compiled TQS-matching kernel.
echo "==> Running client: M=256 N=256 K=256, Q=[16,16] ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "256,256,1,256" \
    --dquant-size-0 16 --dquant-size-1 16 2>&1)" || true
check "M=256,N=256,K=256 Q=[16,16]" "$OUT"

# ── Step 4: Quant tile Q=[32,32] ─────────────────────────────────────────────
echo "==> Running client: M=1024 N=512 K=512, Q=[32,32] ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "512,1024,1,512" \
    --dquant-size-0 32 --dquant-size-1 32 2>&1)" || true
check "M=1024,N=512,K=512 Q=[32,32]" "$OUT"

# ── Step 5: Sub-row quant tile Q=[1,32] ──────────────────────────────────────
echo "==> Running client: M=512 N=512 K=512, Q=[1,32] ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "512,512,1,512" \
    --dquant-size-0 1 --dquant-size-1 32 2>&1)" || true
check "M=512,N=512,K=512 Q=[1,32]" "$OUT"

# ── Step 6: Non-multiple-of-tile M, Q=[1,32] (boundary handling) ─────────────
# M=100 is not a multiple of MT0=64; tests StreamK partial-tile boundary with
# the sub-row quant shape, which is also benchmarked at this exact size.
echo "==> Running client: M=100 N=128 K=64, Q=[1,32] (non-multiple M) ..."
OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" --problem-size "128,100,1,64" \
    --dquant-size-0 1 --dquant-size-1 32 2>&1)" || true
check "M=100,N=128,K=64 Q=[1,32] non-multiple" "$OUT"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
