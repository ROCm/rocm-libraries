#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Benchmark sparse attention variants against an independent DENSE baseline
# (example 01 tile_example_fmha_fwd, async pipeline, bf16), aligned with the
# official SpargeAttn methodology. Emits one JSONL per variant under OUTDIR;
# run summarize_benchmark.py on them to produce markdown tables.
#
# Usage:
#   bash benchmark_sparse_attn.sh [BUILD_DIR]      (default search dir = .)
#   OUTDIR=/path bash benchmark_sparse_attn.sh BUILD_DIR
#   SPARSE_EXE=... DENSE_EXE=... bash benchmark_sparse_attn.sh
#
# NOTE: OUTDIR's result files (dense/sparge/sage_*.jsonl) are cleared each run.

set -euo pipefail

SEARCH_DIR="${1:-.}"
: "${SPARSE_EXE:=$(find "$SEARCH_DIR" -name tile_example_sparse_attn_fwd -type f 2>/dev/null | head -n1)}"
: "${DENSE_EXE:=$(find "$SEARCH_DIR" -name tile_example_fmha_fwd -type f 2>/dev/null | head -n1)}"
: "${OUTDIR:=./bench_out}"

for e in "$SPARSE_EXE" "$DENSE_EXE"; do
    if [ -z "$e" ] || [ ! -x "$e" ]; then
        echo "ERROR: missing executable (sparse='$SPARSE_EXE' dense='$DENSE_EXE')."
        echo "  Build tile_example_sparse_attn_fwd and tile_example_fmha_fwd, or pass BUILD_DIR."
        exit 1
    fi
done

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/dense.jsonl "$OUTDIR"/sparge.jsonl "$OUTDIR"/sage_int8.jsonl "$OUTDIR"/sage_fp8.jsonl

VALID=0
WARMUP=10
REPEAT=50
B=1; H=16; D=128; PERM=1   # official-style: perm=1 (BHSD), batch=1, h=16, d=128

echo "============================================================"
echo "Sparse Attention Benchmark (dense baseline = example 01 async bf16)"
echo "============================================================"
echo "sparse=$SPARSE_EXE"
echo "dense =$DENSE_EXE"
echo "outdir=$OUTDIR  warmup=$WARMUP repeat=$REPEAT  b=$B h=$H d=$D perm=$PERM"

# mask flag: 0=no-mask, 1=causal (top-left). Used for both binaries.
mask_name() { [ "$1" = "0" ] && echo "no" || echo "causal"; }

# Run dense baseline; parse "<ms> ms" from stdout; append a synthetic JSON line.
run_dense() {
    local s="$1" mask="$2" out
    out="$("$DENSE_EXE" -prec=bf16 -b=$B -h=$H -d=$D -s=$s -mask=$mask \
            -iperm=$PERM -operm=$PERM -v=0 -warmup=$WARMUP -repeat=$REPEAT -kname=1 2>&1)"
    printf '%s\n' "$out"
    local ms
    ms="$(printf '%s\n' "$out" | grep -oE '[0-9]+\.[0-9]+ ms,' | tail -n1 | awk '{print $1}')"
    if [ -z "$ms" ]; then echo "ERROR: could not parse dense ms for s=$s mask=$mask" >&2; exit 2; fi
    printf '{"api":"dense_fmha","prec":"bf16","seqlen_k":%s,"sparsity":0,"mask_type":"%s","latency_ms":%s}\n' \
        "$s" "$(mask_name "$mask")" "$ms" >> "$OUTDIR/dense.jsonl"
}

# Run a sparse variant -> its own JSONL file (so int8/fp8 are distinguishable).
run_sparge()    { local s="$1" sp="$2" mask="$3"; "$SPARSE_EXE" -api=sparge -prec=bf16 -b=$B -h=$H -d=$D -s=$s -sparsity=$sp -mask=$mask -iperm=$PERM -operm=$PERM -v=$VALID -warmup=$WARMUP -repeat=$REPEAT -jsonfile="$OUTDIR/sparge.jsonl"; }
run_sage()      { local qk="$1" s="$2" sp="$3" mask="$4"; "$SPARSE_EXE" -api=sparge_sage -prec=bf16 -qkdtype=$qk -qscale=perwarp -b=$B -h=$H -d=$D -s=$s -sparsity=$sp -mask=$mask -iperm=$PERM -operm=$PERM -v=$VALID -warmup=$WARMUP -repeat=$REPEAT -jsonfile="$OUTDIR/sage_${qk}.jsonl"; }

# ---- Sweep A: long-context matrix ----
echo ""; echo "=== Sweep A: matrix (s x sparsity) ==="
for mask in 0 1 ; do
for s in 8192 16384 32768 ; do
    run_dense "$s" "$mask"
    for sp in 0.3 0.5 0.7 ; do
        run_sparge "$s" "$sp" "$mask"
        run_sage int8 "$s" "$sp" "$mask"
        run_sage fp8  "$s" "$sp" "$mask"
    done
done
done

# ---- Sweep B: official-style curve (fixed s=16384, scan sparsity) ----
echo ""; echo "=== Sweep B: official curve (s=16384, scan sparsity) ==="
for mask in 0 1 ; do
    run_dense 16384 "$mask"   # idempotent re-add; summarizer dedups by (s,mask)
    for sp in 0.1 0.2 0.3 0.4 0.5 0.6 ; do
        run_sparge 16384 "$sp" "$mask"
        run_sage int8 16384 "$sp" "$mask"
        run_sage fp8  16384 "$sp" "$mask"
    done
done

echo ""
echo "Done. JSONL written to $OUTDIR/. Summarize with:"
echo "  python3 $(dirname "$0")/summarize_benchmark.py $OUTDIR"
