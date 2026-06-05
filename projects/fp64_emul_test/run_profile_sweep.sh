#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_profile_sweep.sh — Run fp64_emul_accuracy with profiling enabled for
#   N = 1024, 2048, 4096, 8192, 16384 and fixed-s moduli s=2..20.
#
# Usage:
#   ./run_profile_sweep.sh [binary] [output_csv] [num_runs]
#
# Defaults:
#   binary     = ./build/fp64_emul_accuracy
#   output_csv = fp64_emul_profile_<timestamp>.csv
#   num_runs   = 5

set -euo pipefail

BINARY="${1:-./build/fp64_emul_accuracy}"
OUTPUT="${2:-fp64_emul_profile_$(date +%Y%m%d_%H%M%S).csv}"
NUM_RUNS="${3:-5}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0
export HIPBLASLT_EMULATION_PROFILE="$OUTPUT"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    exit 1
fi

rm -f "$OUTPUT"

echo "=== fp64 emulation profile sweep ===" >&2
echo "Binary   : $BINARY" >&2
echo "Profile  : $OUTPUT" >&2
echo "num_runs : $NUM_RUNS" >&2
echo "" >&2

# Profile only s=16 (the library default).
# --no-adaptive: skip the separate adaptive run (which also uses s=16).
# --min-s/--max-s 16: run exactly one fixed configuration.
for N in 1024 2048 4096 8192 16384; do
    echo "[$(date +%H:%M:%S)] N=$N ..." >&2
    "$BINARY" \
        -n "$N" \
        --num-runs "$NUM_RUNS" \
        --phi-list 0.5 \
        --no-adaptive \
        --min-s 16 \
        --max-s 16 \
        --no-check \
      > /dev/null
    echo "[$(date +%H:%M:%S)] N=$N done." >&2
done

echo "" >&2
echo "Profile data: $OUTPUT" >&2
echo "Plot with:    python3 $(dirname "$0")/plot_profile.py $OUTPUT" >&2
