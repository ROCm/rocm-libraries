#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates SDPA forward golden reference bundles. The `quick`/`standard`
# arguments select which group of bundles to generate; each bundle is written to
# the coverage tier it lives in today (see resolve_tier_outdir).
# Requires PyTorch with ROCm support.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd
#   bash generate_golden_data.sh              # Generate all tiers
#   bash generate_golden_data.sh quick        # Generate quick tier only
#   bash generate_golden_data.sh standard     # Generate standard tier only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$(cd "$GOLDEN_ROOT/../reference-data-scripts" && pwd)/generate_sdpa_fwd_golden.py"
PAGED_GENERATOR="$(dirname "$GENERATOR")/generate_sdpa_fwd_paged_golden.py"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at: $GENERATOR"
    exit 1
fi

TIER="${1:-all}"

# The checked-in bundles use the conventional 1/sqrt(D_qk) scale. The generator's
# default is 1.0 (no scaling, as for an unset attn_scale_value), so state the scale
# for every bundle to regenerate the same data.
SCALE_D128="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(128)))')"
SCALE_D192="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(192)))')"
SCALE_D64="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(64)))')"

# DVC remote that hipDNN SDPA golden bundles live on. Each generated bundle's
# .tensors.dvc pointer is written with a per-output `remote:` key set to this, so
# a bare `dvc pull` (and CI) fetches the data from the right place without any
# `-r` flag or workflow change. See ../../../docs/file-formats.md ("Golden data").
DVC_REMOTE="golden-data"

# Bundles are tiered by coverage (quick/standard/comprehensive/full), not by the
# section of this script that generates them. Regenerate each bundle where it
# lives today: swap the tier segment of "outdir" for the tier that already holds
# "<outdir>/<name>". A bundle that exists in no tier keeps the tier given in
# "outdir".
resolve_tier_outdir() {
    local outdir="$1"
    local name="$2"
    local rel="${outdir#"$GOLDEN_ROOT"/*/}"
    local tier
    for tier in quick standard comprehensive full; do
        if [[ -d "$GOLDEN_ROOT/$tier/$rel/$name" ]]; then
            echo "$GOLDEN_ROOT/$tier/$rel"
            return
        fi
    done
    echo "$outdir"
}

generate_bundle() {
    local outdir="$1"
    local name="$2"
    shift 2
    outdir="$(resolve_tier_outdir "$outdir" "$name")"
    mkdir -p "$outdir/$name"
    python3 "$GENERATOR" --base-filename "$outdir/$name/$name" "$@"

    # Author the per-bundle DVC pointer with the golden-data remote pinned per
    # output. DVC fills in md5/size on `dvc commit`; the `remote:` key persists
    # across regenerations. This is what routes the bundle to the golden-data
    # remote — do not drop it, or CI will fail to pull the data.
    local dvc_file="$outdir/$name/$name.tensors.dvc"
    {
        echo "outs:"
        for bin in "$outdir/$name/$name".tensor*.bin; do
            echo "- path: $(basename "$bin")"
            echo "  remote: $DVC_REMOTE"
        done
    } > "$dvc_file"
}

# Paged bundles: the paged generator writes the complete .tensors.dvc pointer
# (golden-data remote, md5 and size) and an empty .support.json itself.
generate_paged_bundle() {
    local outdir="$1"
    local name="$2"
    shift 2
    outdir="$(resolve_tier_outdir "$outdir" "$name")"
    mkdir -p "$outdir/$name"
    python3 "$PAGED_GENERATOR" --base-filename "$outdir/$name/$name" "$@"
}

# --- quick tier: Small bundles only ---
if [[ "$TIER" == "all" || "$TIER" == "quick" ]]; then
    echo "=== Generating quick tier ==="

    # BF16 non-stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "SmallStats" --stats --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 non-stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "SmallStats" --stats --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 hd192 (D_qk=192, D_v=128), no mask + causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D192"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D192"

    # FP16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp16 --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D192"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --dtype fp16 --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D192"

    echo ""
fi

# --- standard tier: Medium and Gqa bundles ---
if [[ "$TIER" == "all" || "$TIER" == "standard" ]]; then
    echo "=== Generating standard tier ==="

    # BF16 non-stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "Gqa" --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "MediumStats" --stats --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "GqaStats" --stats --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 non-stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "Gqa" --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "MediumStats" --stats --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "GqaStats" --stats --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # FP16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # BF16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D192"
    generate_bundle "$OUTDIR" "Gqa" --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D192"
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D192"
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D192"

    # FP16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --dtype fp16 --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D192"
    generate_bundle "$OUTDIR" "Gqa" --dtype fp16 --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D192"
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --dtype fp16 --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42 --attn-scale "$SCALE_D192"
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --dtype fp16 --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42 --attn-scale "$SCALE_D192"

    echo ""
fi

# --- Tier B: FP8 and GROUP-mode bundles (generator-complete; not yet validated by
# the CPU reference executor, which does not accept FP8 descale or variable-seq-len
# attributes). Generate with GENERATE_TIER_B=1. ---
if [[ "${GENERATE_TIER_B:-0}" == "1" ]]; then
    echo "=== Generating Tier B (fp8 + group) ==="

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42 --attn-scale "$SCALE_D128"

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_group"
    generate_bundle "$OUTDIR" "Small" --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42 --attn-scale "$SCALE_D128"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_causal_group"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42 --attn-scale "$SCALE_D128"

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_causal_group"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --causal bottom_right --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42 --attn-scale "$SCALE_D128"

    echo ""
fi

# --- Discriminating bundles: seq_len_q != seq_len_kv per batch and Sq != Skv, so a
# reference that computes the bottom-right offset from the maximum lengths, or swaps
# the alignments, fails them (the group bundles above have equal lengths and cannot
# tell). Batch 1 of the padded set has Sq > Skv, so bottom-right gives it dead rows
# (O = 0, LSE = -inf). No windows: their left_bound convention is still open
# (SdpaSemanticsRegister.md SEM-001). Generate in isolation with:
#   GENERATE_DISCRIMINATING=1 bash generate_golden_data.sh none ---
if [[ "${GENERATE_DISCRIMINATING:-0}" == "1" ]]; then
    echo "=== Generating discriminating bundles ==="

    PADNEQ=(--q-dims 3 2 64 64 --v-dims 3 2 96 64 --seq-lens-q 17 64 5 --seq-lens-kv 96 40 33
            --stats --seed 7 --attn-scale 0.125)
    for causal in none top_left bottom_right; do
        case "$causal" in
            none) name=padneq_nomask ;;
            top_left) name=padneq_causal_tl ;;
            bottom_right) name=padneq_causal_br ;;
        esac
        generate_bundle "$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/$name" "Small" "${PADNEQ[@]}" --causal "$causal"
    done
    generate_bundle "$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/neq_causal_br_gqa" "Small" --causal bottom_right \
        --q-dims 1 4 64 64 --v-dims 1 2 96 64 --stats --seed 7 --attn-scale 0.125
    generate_bundle "$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/neq_causal_tl_gqa" "Small" --causal top_left \
        --q-dims 1 4 96 64 --v-dims 1 2 64 64 --stats --seed 7 --attn-scale 0.125

    echo ""
fi

# --- Ragged (RFC-0014) bundles (not validated by the reference executors,
# which reject ragged tensors). Generate in isolation with:
#   GENERATE_RAGGED=1 bash generate_golden_data.sh none ---
if [[ "${GENERATE_RAGGED:-0}" == "1" ]]; then
    echo "=== Generating ragged bundles ==="

    # Ragged (packed RFC-0014 BSHD), uniform S_max blocks, no seq-lens
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd128_nomask_ragged"
    generate_bundle "$OUTDIR" "Small" --ragged-offsets --layout bshd \
        --q-dims 3 2 256 128 --v-dims 3 2 256 128 --seed 42 --attn-scale "$SCALE_D128"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd128_causal_ragged"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --ragged-offsets --layout bshd \
        --q-dims 3 2 256 128 --v-dims 3 2 256 128 --seed 42 --attn-scale "$SCALE_D128"

    # hd192x128 ragged: Q and O share offset uid 10 with different
    # ragged_offset_multipliers (H*192 vs H*128).
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd192_nomask_ragged"
    generate_bundle "$OUTDIR" "Small" --ragged-offsets --layout bshd \
        --q-dims 3 2 256 192 --v-dims 3 2 256 128 --seed 42 --attn-scale "$SCALE_D192"
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd192_causal_ragged"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --ragged-offsets --layout bshd \
        --q-dims 3 2 256 192 --v-dims 3 2 256 128 --seed 42 --attn-scale "$SCALE_D192"

    echo ""
fi

# --- Paged K/V (vLLM paged attention) bundles, dense or ragged Q. Generate in
# isolation with:
#   GENERATE_PAGED=1 bash generate_golden_data.sh none ---
if [[ "${GENERATE_PAGED:-0}" == "1" ]]; then
    echo "=== Generating paged bundles ==="

    # Decode: dense Q (Sq=1), GQA 8/2, an empty sequence and partial last blocks.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd128_causal_paged"
    generate_paged_bundle "$OUTDIR" "Decode" --dtype bf16 \
        --num-heads-q 8 --num-heads-kv 2 --head-dim 128 --seq-q 1 \
        --block-size 16 --max-seq-len-kv 128 --seq-lens-kv 37 0 128 16 \
        --causal bottom_right --seed 42 --attn-scale "$SCALE_D128"

    # Speculative decode: dense Q (Sq=4), MQA, one page table bound to K and V.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/fp16/hd64_causal_paged"
    generate_paged_bundle "$OUTDIR" "SpecDecode" --dtype fp16 \
        --num-heads-q 4 --num-heads-kv 1 --head-dim 64 --seq-q 4 \
        --block-size 32 --max-seq-len-kv 96 --seq-lens-kv 96 5 0 \
        --shared-page-table --causal bottom_right --seed 42 --attn-scale "$SCALE_D64"

    # Mixed prefill/decode: ragged Q, interleaved (2*D row pitch) K/V, packed LSE.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd128_causal_ragged_paged"
    generate_paged_bundle "$OUTDIR" "Mixed" --dtype bf16 \
        --num-heads-q 8 --num-heads-kv 2 --head-dim 128 \
        --ragged-q --seq-lens-q 5 1 0 3 \
        --block-size 16 --max-seq-len-kv 64 --seq-lens-kv 20 33 0 3 \
        --interleaved-kv --stats --causal bottom_right --seed 42 --attn-scale "$SCALE_D128"

    # Mixed with a sliding window (left_bound 8); batch 2 has Q rows but no keys.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/fp16/hd64_window_ragged_paged"
    generate_paged_bundle "$OUTDIR" "Mixed" --dtype fp16 \
        --num-heads-q 4 --num-heads-kv 2 --head-dim 64 \
        --ragged-q --seq-lens-q 6 1 4 \
        --block-size 16 --max-seq-len-kv 80 --seq-lens-kv 70 1 0 \
        --causal bottom_right --window-left 8 --seed 42 --attn-scale "$SCALE_D64"

    echo ""
fi

# --- Paged K/V bundles with attention sinks (gpt-oss: one sink logit per query
# head, N(0, 1), in the io dtype as vLLM keeps them). Generate in isolation with:
#   GENERATE_SINKS=1 bash generate_golden_data.sh none ---
if [[ "${GENERATE_SINKS:-0}" == "1" ]]; then
    echo "=== Generating paged bundles with attention sinks ==="

    # Decode: dense Q (Sq=1), GQA 8 (16/2), an empty sequence (O = 0), one page
    # table bound to K and V as vLLM passes it. No LSE, so the CDNA paged engine
    # (which has no LSE output) can serve it too.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd64_causal_paged_sinks"
    generate_paged_bundle "$OUTDIR" "Decode" --dtype bf16 \
        --num-heads-q 16 --num-heads-kv 2 --head-dim 64 --seq-q 1 \
        --block-size 16 --max-seq-len-kv 128 --seq-lens-kv 37 0 128 16 \
        --shared-page-table --sinks --causal bottom_right --seed 42 \
        --attn-scale "$SCALE_D64"

    # The same decode with a FLOAT sink (cuDNN's dtype). Reference-only: the
    # rocKE-backed engines take the sink in the io dtype.
    generate_paged_bundle "$OUTDIR" "FloatSink" --dtype bf16 \
        --num-heads-q 16 --num-heads-kv 2 --head-dim 64 --seq-q 1 \
        --block-size 16 --max-seq-len-kv 128 --seq-lens-kv 37 0 128 16 \
        --sinks --sink-dtype float --stats --causal bottom_right --seed 42 \
        --attn-scale "$SCALE_D64"

    # Mixed prefill/decode with gpt-oss's 128-token sliding window (left bound 127),
    # ragged Q, a sequence with Q rows but no keys, separate K and V containers with
    # a 2*D row pitch (the strides of an interleaved K|V cache) and one shared page
    # table.
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/bf16/hd64_window_ragged_paged_sinks"
    generate_paged_bundle "$OUTDIR" "Mixed" --dtype bf16 \
        --num-heads-q 16 --num-heads-kv 2 --head-dim 64 \
        --ragged-q --seq-lens-q 5 1 2 3 1 \
        --block-size 16 --max-seq-len-kv 256 --seq-lens-kv 200 33 0 3 150 \
        --interleaved-kv --shared-page-table --sinks --causal bottom_right \
        --window-left 127 --seed 42 --attn-scale "$SCALE_D64"

    # fp16 decode with LSE (the sink is part of it; an empty sequence has LSE = sink).
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bshd/fp16/hd64_causal_paged_sinks"
    generate_paged_bundle "$OUTDIR" "Decode" --dtype fp16 \
        --num-heads-q 16 --num-heads-kv 2 --head-dim 64 --seq-q 1 \
        --block-size 16 --max-seq-len-kv 128 --seq-lens-kv 37 0 128 16 \
        --shared-page-table --sinks --stats --causal bottom_right --seed 7 \
        --attn-scale "$SCALE_D64"

    echo ""
fi

echo "=== Done ==="
echo "Generated bundles:"
find "$GOLDEN_ROOT/quick/SdpaFwd" "$GOLDEN_ROOT/standard/SdpaFwd" -name "*.json" ! -name "*.meta.json" 2>/dev/null | sort
