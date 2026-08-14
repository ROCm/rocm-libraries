#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Build a device library from a benchmark YAML.
# Usage: build_library.sh --yaml PATH [--chip gfx950] [--out-dir DIR]
#        [--library-format yaml|msgpack] [--client PATH]

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
YAML=""
CHIP="gfx950"
OUT_DIR=""
LIBRARY_FORMAT="msgpack"
CLIENT=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yaml)               YAML="$2";               shift 2 ;;
        --chip)               CHIP="$2";               shift 2 ;;
        --out-dir)            OUT_DIR="$2";            shift 2 ;;
        --library-format)     LIBRARY_FORMAT="$2";     shift 2 ;;
        --client)             CLIENT="$2";             shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$YAML" ]]; then
    echo "error: --yaml is required" >&2
    exit 1
fi
if [[ ! -r "$YAML" ]]; then
    echo "error: YAML file not found: $YAML" >&2
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="/tmp/lib_$(basename "$YAML" .yaml)"
fi

LOGIC_DIR="${OUT_DIR}/3_LibraryLogic"
LIB_DIR="${OUT_DIR}/library"

# ── Step 1: Run Tensile benchmark + logic generation ──────────────────────────
echo "==> Running Tensile pipeline for ${CHIP} ..."
"$PYTHON" "${TENSILELITE_ROOT}/Tensile/bin/Tensile" "$YAML" "$OUT_DIR"

# ── Step 2: Compile the device library ───────────────────────────────────────
echo ""
echo "==> Building device library (format: ${LIBRARY_FORMAT}) ..."
rm -rf "$LIB_DIR"
"$PYTHON" "${TENSILELITE_ROOT}/Tensile/bin/TensileCreateLibrary" \
    --library-format="$LIBRARY_FORMAT" \
    --architecture "$CHIP" \
    "$LOGIC_DIR" "$LIB_DIR" HIP

# ── Step 3: Report artifact paths ────────────────────────────────────────────
ARTIFACT_DIR="${LIB_DIR}/library/${CHIP}"
echo ""
echo "==> Library artifacts in ${ARTIFACT_DIR}:"
find "$ARTIFACT_DIR" \
    \( -name "*.co" -o -name "*.hsaco" -o -name "*.dat" -o -name "*.dat.zlib" \) \
    -printf "  %p\n" 2>/dev/null \
    || ls "$ARTIFACT_DIR" 2>/dev/null \
    || true

# ── Step 4 (optional): Smoke-test with client ─────────────────────────────────
if [[ -n "$CLIENT" ]]; then
    echo ""
    echo "==> Running smoke test with client: ${CLIENT} ..."

    # Locate the non-lazy library YAML and its code object.
    LIBRARY_YAML="$(find "$ARTIFACT_DIR" -maxdepth 1 -name "TensileLibrary_BB_*.yaml" \
                        2>/dev/null | head -1 || true)"
    if [[ -z "$LIBRARY_YAML" ]]; then
        # Fall back to first non-lazy .dat.
        LIBRARY_YAML="$(find "$ARTIFACT_DIR" -maxdepth 1 -name "TensileLibrary_BB_*.dat" \
                            2>/dev/null | head -1 || true)"
    fi
    if [[ -z "$LIBRARY_YAML" ]]; then
        echo "warning: could not find non-lazy TensileLibrary_BB_* file; skipping smoke test" >&2
    else
        LIBRARY_CO="$(find "$ARTIFACT_DIR" -maxdepth 1 -name "TensileLibrary_BB_*.co" \
                          2>/dev/null | head -1 || true)"

        SMOKE_ARGS=("$CLIENT"
                    "--library-file" "$LIBRARY_YAML"
                    "--code-object" "$LIBRARY_CO")

        if echo "$YAML" | grep -qi "partialrms"; then
            SMOKE_ARGS+=("--use-partial-rms" "--problem-size" "512,512,1,512")
        else
            SMOKE_ARGS+=("--problem-size" "512,512,1,512")
        fi

        echo "  command: ${SMOKE_ARGS[*]}"
        "${SMOKE_ARGS[@]}" | tee /tmp/build_library_smoke.log
        if grep -qP '^0,.*,PASSED,' /tmp/build_library_smoke.log; then
            echo "==> Smoke test PASSED"
        else
            echo "error: smoke test did not produce a PASSED line" >&2
            exit 1
        fi
    fi
fi
