#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_profile_sweep.sh — Run fp64_emul_accuracy with profiling enabled for
#   N = 1024..65536 (powers of 2), s=16, phi=0.5, all 4 transpose types.
#   Produces one profile CSV per transpose combination.
#
# Usage:
#   ./run_profile_sweep.sh [binary] [base_output] [num_runs]
#
# Defaults:
#   binary      = ./build/fp64_emul_accuracy
#   base_output = fp64_emul_profile_<timestamp>
#                 (produces <base>_NN.csv, <base>_NT.csv, <base>_TN.csv, <base>_TT.csv)
#   num_runs    = 5
#
# The combined plot is generated at <base>.png by calling plot_profile.py
# with all four CSVs.
#
# Note: N=32768 and N=65536 require large workspaces (~28 GB and ~88 GB
# respectively) — ensure sufficient GPU memory before running those sizes.

set -euo pipefail

BINARY="${1:-./build/fp64_emul_accuracy}"
BASE="${2:-fp64_emul_profile_$(date +%Y%m%d_%H%M%S)}"
NUM_RUNS="${3:-5}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    exit 1
fi

echo "=== fp64 emulation profile sweep ===" >&2
echo "Binary   : $BINARY" >&2
echo "Base     : $BASE" >&2
echo "num_runs : $NUM_RUNS" >&2
echo "" >&2

# Profile s=16 (library default), phi=0.5, --no-check (timing only).
# Outer loop: N; inner loop: transpose combinations.
# Each transpose writes to its own CSV (HIPBLASLT_EMULATION_PROFILE is set
# before each binary invocation; the library appends to the file).
TRANS_LIST=(NN NT TN TT)
CSVS=()

# Pre-populate CSVS list and clear all output files before starting.
for TRANS in "${TRANS_LIST[@]}"; do
    OUT="${BASE}_${TRANS}.csv"
    CSVS+=("$OUT")
    rm -f "$OUT"
done

for N in 1024 2048 4096 8192 16384 32768 65536; do
    echo "[$(date +%H:%M:%S)] N=$N ..." >&2
    for TRANS in "${TRANS_LIST[@]}"; do
        OUT="${BASE}_${TRANS}.csv"
        export HIPBLASLT_EMULATION_PROFILE="$OUT"
        "$BINARY" \
            -n "$N" \
            --trans "$TRANS" \
            --num-runs "$NUM_RUNS" \
            --phi-list 0.5 \
            --no-adaptive \
            --min-s 16 \
            --max-s 16 \
            --no-check \
          > /dev/null
    done
    echo "[$(date +%H:%M:%S)] N=$N done (all transposes)." >&2
done
echo "" >&2

echo "Profile data:" >&2
for F in "${CSVS[@]}"; do echo "  $F" >&2; done
echo "" >&2

PLOT_OUT="${BASE}.png"
echo "Generating combined plot: $PLOT_OUT" >&2
python3 "$(dirname "$0")/plot_profile.py" "${CSVS[@]}" "$PLOT_OUT"
echo "Done." >&2
