#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build a device library from a hand-written assembly kernel.
#
# Steps:
#   1. Generate a LibraryLogic YAML via gen_asm_logic.py.
#   2. Compile a device library (msgpack format) with TensileCreateLibrary.
#   3. Report the produced code object and data files.
#
# Usage:
#   epilogues/scripts/build_asm_library.sh <asm_file> [--chip CHIP] [--out-dir DIR]
#       [--kernel-name NAME]
#
# Arguments:
#   asm_file       Path to the hand-written .s assembly file (required).
#   --chip         GPU architecture string (default: gfx950).
#   --out-dir      Scratch directory for generated files (default: /tmp/asm_library).
#   --kernel-name  Override the kernel symbol name used in the library.
#
# Prerequisites:
#   The TensileLite Python environment must be active, or set PYTHON to the
#   path of the venv python (e.g. PYTHON=/path/to/venv/bin/python).

set -euo pipefail

_VENV="${HOME}/.tensile/bin/python"
PYTHON="${PYTHON:-${_VENV}}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi
CHIP="gfx950"
OUT_DIR="/tmp/asm_library"
KERNEL_NAME=""
ASM_FILE=""

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "error: asm_file argument is required" >&2
    exit 1
fi
ASM_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chip)        CHIP="$2";        shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --kernel-name) KERNEL_NAME="$2"; shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -r "$ASM_FILE" ]]; then
    echo "error: asm file not found: $ASM_FILE" >&2
    exit 1
fi

TENSILELITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGIC_DIR="${OUT_DIR}/logic"
TCL_OUT_DIR="${OUT_DIR}/library"

# ── Step 1: Generate the LibraryLogic YAML ───────────────────────────────────
echo "==> Generating LibraryLogic YAML for ${CHIP} ..."
"$PYTHON" "${TENSILELITE_ROOT}/epilogues/gen/gen_asm_logic.py" "$ASM_FILE" \
    --chip "$CHIP" \
    --out-dir "$LOGIC_DIR" \
    ${KERNEL_NAME:+--kernel-name "$KERNEL_NAME"}

# ── Step 2: Sanity-check that the logic YAML was produced ─────────────────────
LOGIC_YAML="$(find "${LOGIC_DIR}/${CHIP}/Equality" -maxdepth 1 -name "*.yaml" 2>/dev/null | head -1)"
if [[ -z "$LOGIC_YAML" ]]; then
    echo "error: no logic YAML found under ${LOGIC_DIR}/${CHIP}/Equality" >&2
    exit 1
fi
echo "  Logic YAML: ${LOGIC_YAML}"

# ── Step 3: Build the device library (msgpack format) ────────────────────────
echo "==> Building device library ..."
rm -rf "$TCL_OUT_DIR"
"$PYTHON" -m Tensile.TensileCreateLibrary \
    --architecture "$CHIP" \
    --cxx-compiler "$(command -v amdclang++)" \
    --jobs "$(nproc)" \
    "$LOGIC_DIR" "$TCL_OUT_DIR" HIP

# ── Step 4: Report produced artifacts ────────────────────────────────────────
ARTIFACT_DIR="${TCL_OUT_DIR}/library/${CHIP}"
echo ""
echo "==> Library artifacts in ${ARTIFACT_DIR}:"
find "$ARTIFACT_DIR" \( -name "*.co" -o -name "*.dat" -o -name "*.dat.zlib" -o -name "*.hsaco" \) \
    -printf "  %f\n" 2>/dev/null || ls "$ARTIFACT_DIR" 2>/dev/null || true
