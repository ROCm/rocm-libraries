#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Times comp_tdm_scheduled's tuned instances through the dispatcher (mx_gemm_full_benchmark.py)
# on one gfx1250 GPU, one problem after another. NEEDS A gfx1250 GPU, except with COMPILE_ONLY=1.
#
#   case    problem           configs/comp_tdm_scheduled_<case>_gfx1250.json (example 56 instance)
#   maf     4096x4096x65536   maf     t256x256x512, 2x2 warps, 32x16x128, c4x4, store width 8
#   medium  2048x2048x65536   medium  t128x128x1024, 2x2 warps, 16x16x128, c4x4,
#                                     no SplitBLdsRead, store width 4
#   mab     65536x16x65536    mab256  t256x16x512, 4x1 warps, 16x16x128, c1x4, GL2 distance 0,
#                                     no PrefetchLdsRead or SplitBLdsRead, store width 1
#   Each problem is whole tiles, so no edge code; no K tail; direct epilogue; FP16 E (example 56's
#   instances store FP8; maf and maf_fp16 time the same).
#
# INIT=trig (default) times on example 56's -data=trig inputs (E2M1 codes of sin/cos of the
# element index, scales 2^-3..2^3), so the times compare with example 56's; INIT=grid on the
# driver's default inputs (random exact grid values), which switch more bits and run slower.
#
# Kernels are built once into $WORK_DIR/build and reused by name: remove it after changing the
# pipeline. Each run writes $WORK_DIR/results/<YYYYmmdd_HHMMSS>_gpu<GPU>[_<TAG>]/:
#   run.info                host, GPU, git state, settings
#   <case>.log, <case>.csv  the driver's output and results (kernel, latency_ms, tflops, ...)
#
# Environment:
#   GPU=0  CASES="maf medium mab"  INIT=trig|grid  WARMUP=200  REPEAT=2000
#   WORK_DIR=$PWD/comp_tdm_scheduled_bench  TIMEOUT=600 (s per kernel, input generation
#   included: trig takes about 2 min for mab)  VERIFY=0 (1: numpy reference; slow, and mab's needs
#   tens of GiB)  COMPILE_ONLY=0  WORKERS=8  TAG=  CK_TILE_HIPCC=/opt/rocm/bin/hipcc
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEMM_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG_DIR="$SCRIPT_DIR/configs"

GPU="${GPU:-0}"
CASES="${CASES:-maf medium mab}"
INIT="${INIT:-trig}"
WARMUP="${WARMUP:-200}"
REPEAT="${REPEAT:-2000}"
WORK_DIR="${WORK_DIR:-$PWD/comp_tdm_scheduled_bench}"
TIMEOUT="${TIMEOUT:-600}"
VERIFY="${VERIFY:-0}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
WORKERS="${WORKERS:-8}"
TAG="${TAG:-}"

for c in $CASES; do
    for f in "$CONFIG_DIR/comp_tdm_scheduled_${c}_gfx1250.json" \
        "$CONFIG_DIR/comp_tdm_scheduled_${c}_problems.json"; do
        [[ -f "$f" ]] || { echo "unknown case '$c': no $f" >&2; exit 2; }
    done
done

export CK_TILE_HIPCC="${CK_TILE_HIPCC:-/opt/rocm/bin/hipcc}"
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export CK_TILE_BENCH_WARMUP="$WARMUP"
export CK_TILE_BENCH_REPEAT="$REPEAT"

OUT="$WORK_DIR/results/$(date +%Y%m%d_%H%M%S)_gpu${GPU}${TAG:+_$TAG}"
mkdir -p "$OUT" "$WORK_DIR/build"

{
    echo "date:     $(date -Iseconds)"
    echo "host:     $(hostname)"
    echo "gpu:      $GPU"
    echo "commit:   $(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch:   $(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "dirty:    $(git -C "$SCRIPT_DIR" status --porcelain --untracked-files=no 2>/dev/null |
        wc -l) changed tracked paths"
    echo "cases:    $CASES"
    echo "init:     $INIT"
    echo "warmup:   $WARMUP  repeat: $REPEAT  verify: $VERIFY  compile_only: $COMPILE_ONLY"
} | tee "$OUT/run.info"

cd "$GEMM_DIR" || exit 1
status=0
for c in $CASES; do
    echo "=== $c ==="
    args=("$CONFIG_DIR/comp_tdm_scheduled_${c}_gfx1250.json" --arch gfx1250 --dtype fp4
        --problems "$CONFIG_DIR/comp_tdm_scheduled_${c}_problems.json" --init "$INIT"
        --devices "$GPU," --workers "$WORKERS" --kernel-timeout "$TIMEOUT"
        --output-dir "$WORK_DIR/build" --csv "$OUT/$c.csv")
    [[ "$VERIFY" == 1 ]] && args+=(--verify)
    [[ "$COMPILE_ONLY" == 1 ]] && args+=(--compile-only)
    python3 mx_gemm_full_benchmark.py "${args[@]}" 2>&1 | tee "$OUT/$c.log"
    rc=${PIPESTATUS[0]}
    ((rc != 0)) && status=$rc
done

if [[ "$COMPILE_ONLY" != 1 ]]; then
    echo
    echo "=== summary ==="
    for c in $CASES; do
        grep -h "  OK\|FAIL\|SKIP" "$OUT/$c.log" | sed "s/^/$c /"
    done
fi
echo "results: $OUT"
exit "$status"
