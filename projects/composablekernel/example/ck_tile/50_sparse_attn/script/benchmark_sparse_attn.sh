#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Benchmark for all sparse attention variants.
# Usage: bash benchmark_sparse_attn.sh                     (auto-find exe)
#        bash benchmark_sparse_attn.sh /path/to/build      (search in build dir)
#        EXE=/path/to/exe bash benchmark_sparse_attn.sh    (explicit exe path)

set -euo pipefail

if [ -z "${EXE:-}" ]; then
    SEARCH_DIR="${1:-.}"
    EXE="$(find "$SEARCH_DIR" -name tile_example_sparse_attn_fwd -type f 2>/dev/null | head -n 1)"
fi

if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: Cannot find executable 'tile_example_sparse_attn_fwd'"
    echo "  Run from CK root or build directory, or pass build dir as argument:"
    echo "    bash benchmark_sparse_attn.sh /path/to/build"
    exit 1
fi

VALID=0
WARMUP=10
REPEAT=50

echo "============================================================"
echo "Benchmarking Sparse Attention Variants"
echo "============================================================"
echo "Executable: $EXE"
echo "warmup=$WARMUP, repeat=$REPEAT"
echo ""

for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do

echo ""
echo "=== prec=$prec, perm=$perm ==="

# --- Dense baseline (sparge sparsity=0 = full block selection) ---
echo "--- Dense baseline (for speedup reference) ---"
for s in 4096 8192 16384; do
    "$EXE" -api=sparge -prec=$prec -b=1 -h=16 -d=128 -s=$s -sparsity=0 \
           -iperm=$perm -operm=$perm -kname=1 -v=$VALID -warmup=$WARMUP -repeat=$REPEAT
done

# --- Jenga ---
echo "--- Jenga Sparse Attention ---"
for sparsity in 0.3 0.5 0.7 ; do
for s in 4096 8192 16384 ; do
    "$EXE" -api=jenga -prec=$prec -b=1 -h=16 -d=128 -s=$s -sparsity=$sparsity \
           -iperm=$perm -operm=$perm -kname=1 -v=$VALID -warmup=$WARMUP -repeat=$REPEAT
done
done

# --- VSA ---
echo "--- VSA Sparse Attention ---"
for sparsity in 0.3 0.5 0.7 ; do
for s in 4096 8192 16384 ; do
    "$EXE" -api=vsa -prec=$prec -b=1 -h=16 -d=128 -s=$s -sparsity=$sparsity \
           -iperm=$perm -operm=$perm -kname=1 -v=$VALID -warmup=$WARMUP -repeat=$REPEAT
done
done

# --- Sparge (-print_sparsity=1 makes TFlops/GB/s reflect actual sparsity) ---
echo "--- SpargeAttention ---"
for sparsity in 0.3 0.5 0.7 ; do
for s in 4096 8192 16384 ; do
    "$EXE" -api=sparge -prec=$prec -b=1 -h=16 -d=128 -s=$s -sparsity=$sparsity \
           -iperm=$perm -operm=$perm -kname=1 -v=$VALID -warmup=$WARMUP -repeat=$REPEAT \
           -print_sparsity=1
done
done

# --- SpargeAttention-Sage (quantized; bf16 only, requires gfx950/MI350) ---
if [ "$prec" = "bf16" ] ; then
echo "--- SpargeAttention-Sage (quantized) ---"
for qkdtype in int8 fp8 ; do
for sparsity in 0.3 0.5 0.7 ; do
for s in 4096 8192 16384 ; do
    "$EXE" -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=perwarp \
           -b=1 -h=16 -d=128 -s=$s -sparsity=$sparsity \
           -iperm=$perm -operm=$perm -kname=1 -v=$VALID -warmup=$WARMUP -repeat=$REPEAT \
           -print_sparsity=1
done
done
done
fi

done
done
