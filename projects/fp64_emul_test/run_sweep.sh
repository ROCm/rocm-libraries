#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_sweep.sh — Run fp64_emul_accuracy for N = 1024..16384 (powers of 2).
#
# Usage:
#   ./run_sweep.sh [binary] [num_runs] [output_csv]
#
# Defaults:
#   binary      = ./build/fp64_emul_accuracy
#   num_runs    = 30
#   output_csv  = results_<timestamp>.csv
#
# Progress and device info are written to stderr; only CSV goes to stdout
# (and then to the output file).

set -euo pipefail

BINARY="${1:-./build/fp64_emul_accuracy}"
NUM_RUNS="${2:-30}"
OUTPUT="${3:-results_$(date +%Y%m%d_%H%M%S).csv}"

PHI_LIST="0.5,1,2,4"

# Pin to device 0 (overridable: HIP_VISIBLE_DEVICES=2 ./run_sweep.sh)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    echo "Build with:  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)" >&2
    exit 1
fi

echo "=== fp64 emulation sweep ===" >&2
echo "Binary   : $BINARY" >&2
echo "phi_list : $PHI_LIST" >&2
echo "num_runs : $NUM_RUNS (warmup = same)" >&2
echo "Output   : $OUTPUT" >&2
echo "" >&2

# Write CSV header once
echo "phi,N,algo,crt_bits,err_max,err_med,ms_per_run,workspace_MiB" > "$OUTPUT"

# Use fewer timing iterations for larger N:
#   N ≤ 2048 → NUM_RUNS        (user-specified, default 30)
#   N = 4096 → max(5, runs/4)
#   N = 8192 → max(3, runs/8)
#   N ≥ 16384→ max(2, runs/16)
# This gives stable timing at each size without wasting hours on large N.
for N in 1024 2048 4096 8192 16384; do
    if   [ "$N" -le 2048 ]; then RUNS="$NUM_RUNS"
    elif [ "$N" -le 4096 ]; then RUNS=$(( NUM_RUNS / 4  < 5 ? 5 : NUM_RUNS / 4  ))
    elif [ "$N" -le 8192 ]; then RUNS=$(( NUM_RUNS / 8  < 3 ? 3 : NUM_RUNS / 8  ))
    else                         RUNS=$(( NUM_RUNS / 16 < 2 ? 2 : NUM_RUNS / 16 ))
    fi
    echo "[$(date +%H:%M:%S)] Running N=$N (runs=$RUNS) ..." >&2
    "$BINARY" \
        -n         "$N"        \
        --num-runs "$RUNS"     \
        --phi-list "$PHI_LIST" \
      | grep -v "^phi,"   \
      >> "$OUTPUT"
    echo "[$(date +%H:%M:%S)] N=$N done." >&2
done

echo "" >&2
echo "Sweep complete.  Results saved to: $OUTPUT" >&2
