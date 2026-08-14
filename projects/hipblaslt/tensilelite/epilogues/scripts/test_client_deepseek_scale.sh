#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Integration test for the tensilelite-client Deepseek-scale mainloop (PGR=0) paths.
#
# Tests all three modes:
#   A-only: --use-deepseek-scale-a
#   B-only: --use-deepseek-scale-b
#   A+B:    --use-deepseek-scale-a --use-deepseek-scale-b
#
# Steps:
#   1. Run the Tensile pipeline to generate per-mode kernels (one cache per mode).
#   2. Locate the per-mode cache libraries by inspecting kernel names.
#   3. Run the client for each mode and shape against the matching library.
#
# The three BenchmarkProblems sections in gemm_deepseek_scale_client.yaml share the
# same problem-type name, so TensileCreateLibrary would merge them into a single
# kernel entry. To avoid that, this script reads from the per-mode benchmark caches
# that Tensile already built (one cache hash per BenchmarkProblems section), matching
# each cache to its mode by inspecting the kernel name for UDSA1/UDSB1 tokens.
#
# Usage:
#   epilogues/scripts/test_client_deepseek_scale.sh [--chip CHIP] [--client PATH] [--out-dir DIR]
#
# Arguments:
#   --chip    GPU architecture string (default: gfx950).
#   --client  Path to the tensilelite-client binary (required if not on PATH).
#   --out-dir Scratch directory for generated files (default: /tmp/ds_test_<chip>).
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
    OUT_DIR="/tmp/ds_test_${CHIP}"
fi

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

# Find the benchmark cache directory whose TensileLibrary.yaml has a kernel name
# matching the requested UDSA/UDSB combination. Prints the path to the cache's
# library directory on success, or exits with error if none is found.
#   $1: want_udsa — "yes" if UDSA1 must appear in the kernel name, "no" otherwise.
#   $2: want_udsb — "yes" if UDSB1 must appear in the kernel name, "no" otherwise.
find_mode_lib_dir() {
    local want_udsa="$1"
    local want_udsb="$2"
    local label="$3"
    local caches_root="${OUT_DIR}/tensile_out/1_BenchmarkProblems/Cijk_Alik_Bljk_F8SS_BH_UserArgs_00/00_Final/caches"

    for cache_dir in "${caches_root}"/*/; do
        local yaml="${cache_dir}source/library/${CHIP}/TensileLibrary.yaml"
        [[ -f "$yaml" ]] || continue

        local kname
        kname="$(python3 -c "
import yaml
with open('${yaml}') as f:
    d = yaml.safe_load(f)
print(d['solutions'][0]['kernelName'])
" 2>/dev/null)" || continue

        local has_udsa=no has_udsb=no
        echo "$kname" | grep -q 'UDSA1' && has_udsa=yes
        echo "$kname" | grep -q 'UDSB1' && has_udsb=yes

        if [[ "$has_udsa" == "$want_udsa" && "$has_udsb" == "$want_udsb" ]]; then
            echo "${cache_dir}source/library/${CHIP}"
            return 0
        fi
    done

    echo "error: could not find benchmark cache for mode $label (want_udsa=$want_udsa want_udsb=$want_udsb)" >&2
    exit 1
}

# ── Step 1: Run Tensile pipeline ──────────────────────────────────────────────
echo "==> Running Tensile pipeline for $CHIP ..."
TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The benchmark phase validates with its own reference; a FAILED result there
# does not indicate a kernel bug. Correctness is checked by the client below
# with --num-elements-to-validate -1.
set +e
python3 "${TENSILELITE_ROOT}/Tensile/bin/Tensile" \
    "${TENSILELITE_ROOT}/epilogues/yaml/gemm_deepseek_scale_client.yaml" \
    "${OUT_DIR}/tensile_out"
TENSILE_EXIT=$?
set -e
CACHES_ROOT="${OUT_DIR}/tensile_out/1_BenchmarkProblems/Cijk_Alik_Bljk_F8SS_BH_UserArgs_00/00_Final/caches"
if [[ ! -d "$CACHES_ROOT" ]]; then
    echo "error: Tensile pipeline did not produce benchmark caches (exit ${TENSILE_EXIT})" >&2
    exit 1
fi

# ── Step 2: Locate per-mode cache libraries ───────────────────────────────────
echo "==> Locating per-mode cache libraries ..."

AONLY_LIB_DIR="$(find_mode_lib_dir yes no A-only)"
BONLY_LIB_DIR="$(find_mode_lib_dir no  yes B-only)"
AB_LIB_DIR="$(find_mode_lib_dir    yes yes A+B)"

# Each mode's library dir holds TensileLibrary.yaml and TensileLibrary_<chip>.co.
AONLY_YAML="${AONLY_LIB_DIR}/TensileLibrary.yaml"
AONLY_CO="${AONLY_LIB_DIR}/TensileLibrary_${CHIP}.co"
BONLY_YAML="${BONLY_LIB_DIR}/TensileLibrary.yaml"
BONLY_CO="${BONLY_LIB_DIR}/TensileLibrary_${CHIP}.co"
AB_YAML="${AB_LIB_DIR}/TensileLibrary.yaml"
AB_CO="${AB_LIB_DIR}/TensileLibrary_${CHIP}.co"

for f in "$AONLY_YAML" "$AONLY_CO" "$BONLY_YAML" "$BONLY_CO" "$AB_YAML" "$AB_CO"; do
    if [[ ! -f "$f" ]]; then
        echo "error: expected library file not found: $f" >&2
        exit 1
    fi
done

echo "  A-only : $AONLY_LIB_DIR"
echo "  B-only : $BONLY_LIB_DIR"
echo "  A+B    : $AB_LIB_DIR"

# ── Common client arguments (no library — supplied per mode below) ─────────────
# Inputs: fp8 e4m3 (F8); output: f32 (S); compute: f32.
# Transpose: A=TN (TransposeA=True, TransposeB=False).
COMMON_ARGS=(
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type Float8 --b-type Float8 --c-type Float --d-type Float
    --compute-input-type-A Float8 --compute-input-type-B Float8
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --alpha-type Float
    --num-benchmarks 1
    --num-elements-to-validate -1
    --device-idx 0
)

# ── Step 3: A-only mode ───────────────────────────────────────────────────────
echo "==> Testing DeepseekScaleA-only ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="A-only M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" \
        --library-file "$AONLY_YAML" \
        --code-object  "$AONLY_CO" \
        --use-deepseek-scale-a \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Step 4: B-only mode ───────────────────────────────────────────────────────
echo "==> Testing DeepseekScaleB-only ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="B-only M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" \
        --library-file "$BONLY_YAML" \
        --code-object  "$BONLY_CO" \
        --use-deepseek-scale-b \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Step 5: A+B combined mode ─────────────────────────────────────────────────
echo "==> Testing DeepseekScaleA+B ..."

for MN in "128,128" "256,256"; do
    M="${MN%,*}"
    N="${MN#*,}"
    K=128
    label="A+B M=${M} N=${N} K=${K}"
    echo "  ==> $label ..."
    OUT="$("$CLIENT_BIN" "${COMMON_ARGS[@]}" \
        --library-file "$AB_YAML" \
        --code-object  "$AB_CO" \
        --use-deepseek-scale-a \
        --use-deepseek-scale-b \
        --problem-size "${M},${N},1,${K}" 2>&1)" || true
    check "$label" "$OUT"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
