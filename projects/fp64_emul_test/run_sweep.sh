#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_sweep.sh — Run fp64_emul_accuracy for N = 1024..65536 (powers of 2),
#   all four transpose combinations (NN, NT, TN, TT).
#
# Usage:
#   ./run_sweep.sh [binary] [num_runs] [output_csv] [extra_flags]
#
# Defaults:
#   binary      = ./build/fp64_emul_accuracy
#   num_runs    = 30
#   output_csv  = results_<timestamp>.csv
#   extra_flags = (none — error check enabled, all 4 transposes run)
#
# Example — timing-only sweep, no DD reference GEMM (fast for large N):
#   ./run_sweep.sh ./build/fp64_emul_accuracy 30 timing.csv --no-check
#
# Example — single transpose only:
#   ./run_sweep.sh ./build/fp64_emul_accuracy 30 results_TN.csv "--trans TN"
#
# Output CSV columns:
#   phi,N,transa,transb,algo,crt_bits,err_max,err_med,ms_per_run,workspace_MiB
#
# Progress and device info are written to stderr; only CSV goes to stdout
# (and then to the output file).

set -euo pipefail

BINARY="${1:-./build/fp64_emul_accuracy}"
NUM_RUNS="${2:-30}"
OUTPUT="${3:-results_$(date +%Y%m%d_%H%M%S).csv}"
EXTRA_FLAGS="${4:-}"

PHI_LIST="0.5,1,2,4"

# Pin to device 0 (overridable: HIP_VISIBLE_DEVICES=2 ./run_sweep.sh)
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# Disable Inf/NaN detection for the emulation — the benchmark uses clean
# synthetic data and the detection adds a per-call device→host sync that
# inflates measured latency, especially for small N.
# (The fp64_emul_accuracy driver also sets this via the handle API, but the
#  env var serves as an explicit process-level safeguard.)
export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binary not found or not executable: $BINARY" >&2
    echo "Build with:  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)" >&2
    exit 1
fi

echo "=== fp64 emulation sweep ===" >&2
echo "Binary     : $BINARY" >&2
echo "phi_list   : $PHI_LIST" >&2
echo "num_runs   : $NUM_RUNS (warmup = same)" >&2
echo "Output     : $OUTPUT" >&2
echo "Extra flags: ${EXTRA_FLAGS:-(none)}" >&2
echo "" >&2

# Write CSV header once
echo "phi,N,transa,transb,algo,crt_bits,err_max,err_med,ms_per_run,workspace_MiB" > "$OUTPUT"

# Use fewer timing iterations for larger N:
#   N ≤  2048 → NUM_RUNS          (user-specified, default 30)
#   N =  4096 → max(5,  runs/4)
#   N =  8192 → max(3,  runs/8)
#   N = 16384 → max(2,  runs/16)
#   N = 32768 → max(1,  runs/32)
#   N = 65536 → 1  (single timed run; each call is many seconds)
# This gives stable timing at each size without wasting hours on large N.
for N in 1024 2048 4096 8192 16384 32768 65536; do
    if   [ "$N" -le  2048 ]; then RUNS="$NUM_RUNS"
    elif [ "$N" -le  4096 ]; then RUNS=$(( NUM_RUNS / 4  < 5 ? 5 : NUM_RUNS / 4  ))
    elif [ "$N" -le  8192 ]; then RUNS=$(( NUM_RUNS / 8  < 3 ? 3 : NUM_RUNS / 8  ))
    elif [ "$N" -le 16384 ]; then RUNS=$(( NUM_RUNS / 16 < 2 ? 2 : NUM_RUNS / 16 ))
    elif [ "$N" -le 32768 ]; then RUNS=$(( NUM_RUNS / 32 < 1 ? 1 : NUM_RUNS / 32 ))
    else                          RUNS=1
    fi
    echo "[$(date +%H:%M:%S)] Running N=$N (runs=$RUNS) ..." >&2
    # shellcheck disable=SC2086
    "$BINARY" \
        -n         "$N"        \
        --num-runs "$RUNS"     \
        --phi-list "$PHI_LIST" \
        $EXTRA_FLAGS           \
      | grep -v "^phi,"   \
      >> "$OUTPUT"
    echo "[$(date +%H:%M:%S)] N=$N done." >&2
done

echo "" >&2
echo "Sweep complete.  Results saved to: $OUTPUT" >&2
