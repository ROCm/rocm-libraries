#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Smoke test for all sparse attention variants.
# Usage: bash smoke_test_sparse_attn.sh                     (auto-find exe)
#        bash smoke_test_sparse_attn.sh /path/to/build      (search in build dir)
#        EXE=/path/to/exe bash smoke_test_sparse_attn.sh    (explicit exe path)

set -euo pipefail

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
EXE_NAME=tile_example_sparse_attn_fwd

if [ -z "${EXE:-}" ]; then
    SEARCH_DIR="${1:-.}"
    EXE="$(find "$SEARCH_DIR" -name $EXE_NAME -type f 2>/dev/null | head -n 1)"
fi

if [ -z "$EXE" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: Cannot find executable '$EXE_NAME'"
    echo "  Run from CK root or build directory, or pass build dir as argument:"
    echo "    bash smoke_test_sparse_attn.sh /path/to/build"
    exit 1
fi

GPU_arch=${GPU_arch:-}
if [ -z "$GPU_arch" ] ; then
    GPU_arch=$(rocminfo | grep -E 'Name:\s+gfx' | head -n1 | awk '{print $2}')
fi

CURR_FAILS_FILE=${CURR_FAILS_FILE:-"sparse_attn_fwd_fails_$GPU_arch.txt"}
rm -f $CURR_FAILS_FILE
touch $CURR_FAILS_FILE
KNOWN_FAILS_FILE=${KNOWN_FAILS_FILE:-"$SCRIPT_DIR/sparse_attn_fwd_known_fails_$GPU_arch.txt"}

COMMON_ARGS='-v=1 -warmup=0 -repeat=1'

# Smoke defaults to a fast subset; run_full_test.sh overrides these for the full matrix.
#   PERMS    : iperm/operm directions to sweep (smoke: 1; full: "0 1")
#   SL       : long-seqlen used by validated cases (smoke: 2048; full: 4096)
#   QSCALES  : sage qscale modes (smoke: perwarp; full: all four)
#   VARIANTS : APIs to test (default all); tokens: jenga vsa sparge sparge_sage
PERMS="${PERMS:-1}"
SL="${SL:-2048}"
QSCALES="${QSCALES:-perwarp}"
VARIANTS="${VARIANTS:-jenga vsa sparge sparge_sage}"

# want <token> : true if the variant is in the selected set.
want() { case " $VARIANTS " in *" $1 "*) return 0;; *) return 1;; esac; }

run_exe() {
    # Skip if this invocation's -api=<x> is not in VARIANTS.
    local a api=""
    for a in "$@"; do case "$a" in -api=*) api="${a#-api=}";; esac; done
    if [ -n "$api" ] && ! want "$api"; then return 0; fi
    set +e
    echo ">>> $EXE $*"
    "$EXE" "$@"
    local ret=$?
    if [ $ret -ne 0 ] ; then
        echo "FAILED: $EXE_NAME $*"
        echo "$EXE_NAME $*" >> $CURR_FAILS_FILE
    fi
    set -e
}

echo "============================================================"
echo "Smoke Test: Sparse Attention"
echo "============================================================"
echo "Executable: $EXE"
echo ""

# ============================================================================
# Jenga tests
# ============================================================================
echo ""
echo "=== Jenga Sparse Attention ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=1 -h=4 -d=128 -s=1024  -sparsity=0.5 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=2 -h=4 -d=128 -s=2048  -sparsity=0.3 -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=1 -h=8 -h_k=2 -d=128 -s=2048 -sparsity=0.5 -iperm=$perm -operm=$perm
    # jenga batch ignores -mask; use sparge/vsa for masked workloads.
    # Elementwise bias (rank 0/1, batch + mask=0 only).
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -iperm=$perm -operm=$perm
done
done

# ============================================================================
# Jenga group tests
# ============================================================================
echo ""
echo "=== Jenga group ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=4 -h=4 -d=128 -s=1024 -sparsity=0.3 -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=2 -h=8 -h_k=2 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
    # group + elementwise bias (rank 0/1/2). alibi excluded: known mask-drift under bias amplification.
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=jenga -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:2 -iperm=$perm -operm=$perm
done
done

# ============================================================================
# VSA tests
# ============================================================================
echo ""
echo "=== VSA Sparse Attention ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=1024  -sparsity=0.5 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048  -sparsity=0.3 -iperm=$perm -operm=$perm
    # Top-left causal
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # Bottom-right causal
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=b -iperm=$perm -operm=$perm
    # SWA / generic window (t:l,r / b:l,r / g:y,x / xt:N), same syntax as 01_fmha.
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t:-1,0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t:128,32 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=b:0,32 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=g:128,32 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=xt:256 -iperm=$perm -operm=$perm
    # bias: elementwise rank 0/1 + alibi rank 0/1 (batch only).
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=b -iperm=$perm -operm=$perm
    # elementwise rank=2 (full b*h*sq*sk, no broadcast).
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:2 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:2 -mask=t -iperm=$perm -operm=$perm
done
done

# ============================================================================
# VSA group tests
# ============================================================================
echo ""
echo "=== VSA group ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    # mask=0 (regression guard for the NoMask dispatch fix)
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -mask=0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -mask=b -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=8 -h_k=2 -d=128 -s=2048 -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # group + alibi (rank 0/1, requires causal mask).
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=b -iperm=$perm -operm=$perm
    # group + elementwise rank 0/1/2.
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:2 -iperm=$perm -operm=$perm
done
done

# ============================================================================
# SpargeAttention tests
# ============================================================================
echo ""
echo "=== SpargeAttention ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    # CDF mode, no mask
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -iperm=$perm -operm=$perm
    # Top-left causal
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # Bottom-right causal
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=b -iperm=$perm -operm=$perm
    # SWA / generic window — same expressivity as 01_fmha (mask.hpp parses all 4 forms).
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t:128,32 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=g:128,32 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=xt:256 -iperm=$perm -operm=$perm
    # Attention sink
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t -sink=1 -iperm=$perm -operm=$perm
    # CDF mode (-sparge_mode=cdf; topk is the default mode)
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.6 -sparge_mode=cdf -iperm=$perm -operm=$perm
    # Stage 2 P*V skip threshold
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.6 -pvthreshd=3 -iperm=$perm -operm=$perm
    # K smoothing off
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.6 -smooth_k=0 -iperm=$perm -operm=$perm
    # Sim threshold (K-sim/Q-sim union)
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.6 -simthreshold=0.5 -iperm=$perm -operm=$perm
    # Print sparsity (FLOP/byte rescale path)
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -print_sparsity=1 -iperm=$perm -operm=$perm
    # bias: elementwise rank 0/1 (batch only)
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -mask=t -iperm=$perm -operm=$perm
    # bias: alibi rank 0/1 (requires causal mask)
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a   -mask=b -iperm=$perm -operm=$perm
    # elementwise rank=2 (full b*h*sq*sk, no broadcast).
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:2 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:2 -mask=t -iperm=$perm -operm=$perm
done
done

# ============================================================================
# Sparge group tests (s=1024 keeps CPU mask reference tractable per sub-batch)
# ============================================================================
echo ""
echo "=== Sparge group ==="
for prec in "fp16" "bf16" ; do
for perm in $PERMS ; do
    # CDF mode, no mask
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
    # Top-left causal
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=8 -h_k=2 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
    # group + alibi (rank 0/1, requires causal mask).
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=a   -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=a:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=a   -mask=b -iperm=$perm -operm=$perm
    # group + elementwise rank 0/1/2 (padded [B,h_or_1,max_sq,max_sk]; kernel reads per-batch sub-region).
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:1 -mask=t -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -bias=e:2 -iperm=$perm -operm=$perm
done
done

# ============================================================================
# SpargeAttention-Sage (quantized) tests
# sparge block selection + SageAttention quantization (INT8/FP8 QK + per-channel
# FP8 V). bf16 ONLY (-prec=fp16 is rejected by the wrapper). Requires gfx950/MI350.
# ============================================================================
echo ""
echo "=== SpargeAttention-Sage (quantized) ==="
for qkdtype in int8 fp8 ; do
for qscale in $QSCALES ; do
for perm in $PERMS ; do
    # no mask
    run_exe $COMMON_ARGS -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -iperm=$perm -operm=$perm
    # Top-left causal
    run_exe $COMMON_ARGS -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=1 -h=4 -d=128 -s=$SL -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=1 -h=8 -h_k=2 -d=128 -s=$SL -sparsity=0.5 -iperm=$perm -operm=$perm
    # Elementwise bias rank=1
    run_exe $COMMON_ARGS -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=e:1 -iperm=$perm -operm=$perm
    # ALIBI (requires causal mask)
    run_exe $COMMON_ARGS -api=sparge_sage -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=2 -h=4 -d=128 -s=2048 -sparsity=0.5 -bias=a -mask=t -iperm=$perm -operm=$perm
done
done
done

# ============================================================================
# Sparge-Sage group tests (s=1024 keeps CPU mask reference tractable per sub-batch)
# bf16 ONLY; requires gfx950/MI350.
# ============================================================================
echo ""
echo "=== Sparge-Sage group ==="
for qkdtype in int8 fp8 ; do
for qscale in perwarp pertensor ; do
for perm in $PERMS ; do
    # no mask
    run_exe $COMMON_ARGS -api=sparge_sage -mode=1 -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
    # Top-left causal
    run_exe $COMMON_ARGS -api=sparge_sage -mode=1 -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t -iperm=$perm -operm=$perm
    # GQA
    run_exe $COMMON_ARGS -api=sparge_sage -mode=1 -prec=bf16 -qkdtype=$qkdtype -qscale=$qscale -b=2 -h=8 -h_k=2 -d=128 -s=1024 -sparsity=0.5 -iperm=$perm -operm=$perm
done
done
done

# ============================================================================
# Logits soft cap (Gemma-style; NO_BIAS only). batch + group, all 3 APIs.
# ============================================================================
echo "=== Logits soft cap ==="
for prec in fp16 bf16 ; do
for perm in $PERMS ; do
    # Batch mode + cap=8 (Gemma-2 / Gemma-3 style)
    run_exe $COMMON_ARGS -api=jenga  -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=0          -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa    -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t          -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -prec=$prec -b=2 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t          -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
    # Group mode
    run_exe $COMMON_ARGS -api=jenga  -mode=1 -prec=$prec -b=4 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=0  -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=vsa    -mode=1 -prec=$prec -b=4 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t  -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
    run_exe $COMMON_ARGS -api=sparge -mode=1 -prec=$prec -b=4 -h=4 -d=128 -s=1024 -sparsity=0.5 -mask=t  -logits_soft_cap=8.0 -iperm=$perm -operm=$perm
done
done

echo ""
echo "============================================================"

new_fails_count=0
known_fails_count=0
if [ -f $KNOWN_FAILS_FILE ] ; then
    echo "Comparing current fails ($CURR_FAILS_FILE) against known fails ($KNOWN_FAILS_FILE):"
    while IFS= read -r line; do
        if grep -Fxq "$line" $KNOWN_FAILS_FILE; then
            echo "Known fail: $line"
            known_fails_count=$(($known_fails_count + 1))
        else
            echo "New fail: $line"
            new_fails_count=$(($new_fails_count + 1))
        fi
    done < $CURR_FAILS_FILE
else
    new_fails_count=$(wc -l < $CURR_FAILS_FILE)
    echo "No known fails file, all fails ($new_fails_count) are new:"
    cat $CURR_FAILS_FILE
fi
echo "New fails count: $new_fails_count; Known fails count: $known_fails_count"
echo "============================================================"
exit $(($new_fails_count != 0))
