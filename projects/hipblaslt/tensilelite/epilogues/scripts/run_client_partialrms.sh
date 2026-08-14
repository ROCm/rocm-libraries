#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run the tensilelite-client against a pre-built PartialRMS library,
# exercising both modes (plain PartialRMS and PartialRMS + ResidualAdd).
#
# This wraps the integration flow documented in epilogues/docs/TUNING_PIPELINE.md:
#   1. Optionally build the client (invoke build-client).
#   2. Optionally run the Tensile pipeline to generate a library (Tensile + TensileCreateLibrary).
#   3. Run the client for each test case and report PASSED / FAILED.
#
# Usage:
#   epilogues/scripts/run_client_partialrms.sh [OPTIONS]
#
# Options:
#   --client PATH        Path to tensilelite-client binary.
#                        Default: build_tmp/tensilelite/client/tensilelite-client
#   --chip CHIP          GPU architecture (default: auto-detected via 'invoke get-gpu-arch').
#   --lib-yaml PATH      Pre-built library YAML.  If omitted, generates one.
#   --lib-co PATH        Code object (.co) paired with --lib-yaml.  Required if
#                        --lib-yaml is given; otherwise derived automatically.
#   --out-dir DIR        Scratch directory for generated files.
#                        Default: /tmp/prms_client_<chip>
#   --build-client       Re-build the client before running (runs invoke build-client).
#   --problem-size M,N,B,K  Problem size (default: 4096,4096,1,4096).
#   --num-elements N     Elements to validate per run (default: 256).
#   --extra-args ARGS    Extra flags forwarded verbatim to the client binary.
#
# Exit code: 0 if all cases pass, non-zero otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSILE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
CLIENT_BIN="${TENSILE_ROOT}/build_tmp/tensilelite/client/tensilelite-client"
CHIP=""
LIB_YAML=""
LIB_CO=""
OUT_DIR=""
DO_BUILD_CLIENT=0
PROBLEM_SIZE="4096,4096,1,4096"
NUM_ELEMENTS=256
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --client)        CLIENT_BIN="$2";  shift 2 ;;
        --chip)          CHIP="$2";        shift 2 ;;
        --lib-yaml)      LIB_YAML="$2";    shift 2 ;;
        --lib-co)        LIB_CO="$2";      shift 2 ;;
        --out-dir)       OUT_DIR="$2";     shift 2 ;;
        --build-client)  DO_BUILD_CLIENT=1; shift ;;
        --problem-size)  PROBLEM_SIZE="$2"; shift 2 ;;
        --num-elements)  NUM_ELEMENTS="$2"; shift 2 ;;
        --extra-args)    read -ra EXTRA_ARGS <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Detect chip ───────────────────────────────────────────────────────────────
if [[ -z "$CHIP" ]]; then
    CHIP="$(cd "${TENSILE_ROOT}" && invoke get-gpu-arch 2>/dev/null || echo "gfx950")"
    echo "Detected chip: $CHIP"
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/prms_client_${CHIP}"
fi
LIB_DIR="${OUT_DIR}/library_yaml"

# ── Optionally build client ───────────────────────────────────────────────────
if [[ "$DO_BUILD_CLIENT" -eq 1 ]]; then
    echo "==> Building tensilelite-client ..."
    (cd "${TENSILE_ROOT}" && invoke build-client)
fi

if [[ ! -x "$CLIENT_BIN" ]]; then
    echo "error: client binary not found or not executable: $CLIENT_BIN" >&2
    echo "       Run with --build-client to build it, or pass --client <path>." >&2
    exit 1
fi
echo "Client: $CLIENT_BIN"

# ── Generate library if not provided ─────────────────────────────────────────
if [[ -z "$LIB_YAML" ]]; then
    TENSILE_OUT="${OUT_DIR}/tensile_out"

    echo "==> Running Tensile pipeline for $CHIP ..."
    python3 -m Tensile.Tensile \
        "${SCRIPT_DIR}/../yaml/gemm_partial_rms_k1_rowmajor.yaml" \
        "$TENSILE_OUT"

    echo "==> Compiling device library ..."
    rm -rf "$LIB_DIR"
    python3 -m Tensile.TensileCreateLibrary \
        --library-format=yaml \
        --architecture "$CHIP" \
        "${TENSILE_OUT}/3_LibraryLogic" \
        "$LIB_DIR" \
        HIP

    LIB_YAML="$(find "${LIB_DIR}/library/${CHIP}" -maxdepth 1 \
        -name "TensileLibrary_BB_*.yaml" ! -name "*lazy*" | head -1)"
    if [[ -z "$LIB_YAML" ]]; then
        echo "error: could not find non-lazy library YAML under ${LIB_DIR}/library/${CHIP}" >&2
        exit 1
    fi
    LIB_CO="${LIB_YAML%.yaml}.co"
else
    # Explicit --lib-yaml: derive .co from yaml path if not given.
    if [[ -z "$LIB_CO" ]]; then
        LIB_CO="${LIB_YAML%.yaml}.co"
    fi
fi

if [[ ! -f "$LIB_YAML" ]]; then
    echo "error: library YAML not found: $LIB_YAML" >&2; exit 1
fi
if [[ ! -f "$LIB_CO" ]]; then
    echo "error: code object not found: $LIB_CO" >&2; exit 1
fi
echo "Library YAML: $LIB_YAML"
echo "Code object : $LIB_CO"

# ── Shared client arguments ───────────────────────────────────────────────────
COMMON=(
    --library-file      "$LIB_YAML"
    --code-object       "$LIB_CO"
    --problem-identifier "Contraction_l_Alik_Bljk_Cijk_Dijk"
    --a-type BFloat16 --b-type BFloat16 --c-type BFloat16 --d-type BFloat16
    --compute-input-type-A BFloat16 --compute-input-type-B BFloat16
    --high-precision-accumulate
    --f32-xdl-math-op Float
    --problem-size "$PROBLEM_SIZE"
    --num-benchmarks 1
    --num-elements-to-validate "$NUM_ELEMENTS"
    --device-idx 0
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)

# ── Result tracking ───────────────────────────────────────────────────────────
PASS=0
FAIL=0

run_case() {
    local label="$1"
    shift
    echo ""
    echo "-- $label --"
    local out
    out="$("$CLIENT_BIN" "${COMMON[@]}" "$@" 2>&1)" || true
    if echo "$out" | grep -q "^0,.*,PASSED,"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        # Print the first data row for diagnosis.
        echo "$out" | grep "^0," | head -1 | cut -c1-300 >&2
        FAIL=$((FAIL + 1))
    fi
}

# ── Test cases ────────────────────────────────────────────────────────────────
run_case "PartialRMS (no residual)" \
    --use-partial-rms

run_case "PartialRMS + ResidualAdd" \
    --use-partial-rms --partial-rms-residual-add

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo " Results: $PASS passed, $FAIL failed"
echo "=============================="
[[ $FAIL -eq 0 ]] || exit 1
