#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates the comprehensive-tier dense (BATCH) SDPA forward coverage bundles. They
# are graph-only: the harness fills the inputs from each .meta.json recipe and verifies
# the engine against its GPU/CPU reference executor, so there is no golden data to push.
# Each bundle covers a shape, layout, mask or dtype the ASM SDPA forward kernels accept
# that no other bundle exercises. Requires PyTorch (float8_e4m3fnuz for the fp8 bundles).
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/comprehensive/SdpaFwd
#   bash generate_dense_coverage_bundles.sh
#   # then record claims with --write-support-claims (docs/adding-tests.md)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$(cd "$SCRIPT_DIR/../../../reference-data-scripts" && pwd)/generate_sdpa_fwd_golden.py"

SCALE_D128="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(128)))')"
SCALE_D192="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(192)))')"

# gen <layout> <dtype dir> <name> <args...>: writes <layout>/<dtype dir>/<name>/<name>.*
# and an empty support sidecar when none exists.
gen() {
    local layout="$1" dtype="$2" name="$3"
    shift 3
    local base="$SCRIPT_DIR/$layout/$dtype/$name/$name"
    local dtype_args=(--dtype "$dtype")
    if [[ "$dtype" == fp8 ]]; then
        # gfx942 fp8 kernels read FNUZ e4m3 and run fp8 MFMA, which no 16-bit
        # mma_core_mode names, so the engine expects it unset.
        dtype_args=(--dtype fp8 --fp8-format e4m3fnuz --mma-core-mode unset)
    fi
    python3 "$GENERATOR" --base-filename "$base" --layout "$layout" --seed 42 --graph-only \
        "${dtype_args[@]}" "$@"
    if [[ ! -f "$base.support.json" ]]; then
        printf '{\n  "claims": {},\n  "version": 1\n}\n' > "$base.support.json"
    fi
}

# Bottom-right causal with S_q != S_kv. S_q > S_kv fully masks the leading rows (O = 0,
# SdpaSemanticsRegister SEM-010); odd lengths leave partial Q and K/V tiles.
gen bhsd bf16 hd128_causal_br_sq_lt_skv_odd --q-dims 2 4 200 128 --v-dims 2 4 333 128 \
    --causal bottom_right --attn-scale "$SCALE_D128"
gen bhsd bf16 hd128_causal_br_sq_gt_skv --q-dims 2 4 300 128 --v-dims 2 4 160 128 \
    --causal bottom_right --attn-scale "$SCALE_D128"
gen bhsd bf16 hd192_causal_br_sq_lt_skv_odd --q-dims 2 4 77 192 --v-dims 2 4 1000 128 \
    --causal bottom_right --attn-scale "$SCALE_D192"
gen bhsd bf16 hd192_causal_br_sq_gt_skv --q-dims 1 4 260 192 --v-dims 1 4 100 128 \
    --causal bottom_right --attn-scale "$SCALE_D192"

# Decode (S_q = 1), odd lengths, non-power-of-two GQA and MQA.
gen bhsd bf16 hd128_nomask_decode_gqa4 --q-dims 4 8 1 128 --v-dims 4 2 517 128 \
    --attn-scale "$SCALE_D128"
gen bhsd bf16 hd192_nomask_odd_gqa3 --q-dims 2 6 333 192 --v-dims 2 2 257 128 \
    --attn-scale "$SCALE_D192"
gen bhsd bf16 hd192_causal_br_mqa_odd --q-dims 2 12 129 192 --v-dims 2 1 129 128 \
    --causal bottom_right --attn-scale "$SCALE_D192"

# LSE (stats) on the hd192x128 kernel, with odd lengths and GQA, and in BSHD.
gen bhsd bf16 hd192_nomask_stats --q-dims 2 4 200 192 --v-dims 2 4 300 128 --stats \
    --attn-scale "$SCALE_D192"
gen bhsd bf16 hd128_nomask_stats_odd_gqa --q-dims 1 8 77 128 --v-dims 1 2 1031 128 --stats \
    --attn-scale "$SCALE_D128"
gen bshd bf16 hd128_nomask_stats --q-dims 2 4 256 128 --v-dims 2 4 384 128 --stats \
    --attn-scale "$SCALE_D128"

# Masks the engine folds onto the shipped kernels: top-left causal with S_q == S_kv is the
# bottom-right diagonal (AITER get_cfg_mask_type maps it to mask 2), and windows that
# cover the whole span are causal or no mask.
gen bhsd bf16 hd128_causal_tl_sq_eq_skv --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --causal top_left --attn-scale "$SCALE_D128"
gen bhsd bf16 hd128_window_covering_causal_br --q-dims 1 4 512 128 --v-dims 1 4 512 128 \
    --causal bottom_right --left-bound 512 --attn-scale "$SCALE_D128"
gen bhsd bf16 hd128_window_covering_nomask --q-dims 1 4 300 128 --v-dims 1 4 200 128 \
    --left-bound 300 --window-right 199 --attn-scale "$SCALE_D128"

# The 16-bit mma_core_mode values a bf16 graph may carry (cuDNN writes HALF by default).
gen bhsd bf16 hd128_nomask_mma_bfloat16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --mma-core-mode bfloat16 --attn-scale "$SCALE_D128"
gen bhsd bf16 hd128_causal_br_mma_half --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --causal bottom_right --mma-core-mode half --attn-scale "$SCALE_D128"

# Non-packed strides (padded rows, including rows that are not 16-byte aligned) and O in
# a different layout from Q/K/V.
gen bhsd bf16 hd128_nomask_rowpad --q-dims 2 4 256 128 --v-dims 2 4 300 128 \
    --seq-stride-pad 8 --attn-scale "$SCALE_D128"
gen bhsd bf16 hd128_nomask_rowpad_unaligned --q-dims 1 2 256 128 --v-dims 1 2 256 128 \
    --seq-stride-pad 1 --attn-scale "$SCALE_D128"
gen bshd bf16 hd128_causal_br_rowpad --q-dims 2 4 256 128 --v-dims 2 4 300 128 \
    --causal bottom_right --seq-stride-pad 8 --attn-scale "$SCALE_D128"
gen bhsd bf16 hd192_nomask_rowpad --q-dims 2 4 256 192 --v-dims 2 4 300 128 \
    --seq-stride-pad 64 --attn-scale "$SCALE_D192"
gen bshd bf16 hd128_causal_br_o_bhsd --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --causal bottom_right --o-layout bhsd --attn-scale "$SCALE_D128"
gen bhsd bf16 hd192_nomask_o_bshd --q-dims 2 4 256 192 --v-dims 2 4 256 128 \
    --o-layout bshd --attn-scale "$SCALE_D192"
gen bshd bf16 hd192_causal_br --q-dims 3 16 384 192 --v-dims 3 16 384 128 \
    --causal bottom_right --attn-scale "$SCALE_D192"

# Masked graphs with S_q > 16K take the downgraded tune_opt (AITER init_fmha_fwd_v3_args).
gen bhsd bf16 hd128_causal_br_seq16640 --q-dims 1 8 16640 128 --v-dims 1 8 16640 128 \
    --causal bottom_right --attn-scale "$SCALE_D128"

# gfx942 fp8 (FP8_E4M3_FNUZ inputs, per-tensor descales, BF16 O). The causal cases keep
# S_q < S_kv: the kernel rounds P to fp8, which on rows with only a few keys exceeds the
# default bf16 tolerance against the FP32-P references.
gen bhsd fp8 fnuz_hd128_nomask --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --attn-scale "$SCALE_D128"
gen bhsd fp8 fnuz_hd128_causal_br_gqa2_odd --q-dims 2 8 200 128 --v-dims 2 4 333 128 \
    --causal bottom_right --attn-scale "$SCALE_D128"
gen bhsd fp8 fnuz_hd128_nomask_gqa3 --q-dims 1 6 256 128 --v-dims 1 2 256 128 \
    --attn-scale "$SCALE_D128"
gen bhsd fp8 fnuz_hd128_nomask_stats --q-dims 2 4 256 128 --v-dims 2 4 384 128 --stats \
    --attn-scale "$SCALE_D128"
gen bshd fp8 fnuz_hd128_causal_br_sq_lt_skv --q-dims 2 4 200 128 --v-dims 2 4 333 128 \
    --causal bottom_right --attn-scale "$SCALE_D128"
