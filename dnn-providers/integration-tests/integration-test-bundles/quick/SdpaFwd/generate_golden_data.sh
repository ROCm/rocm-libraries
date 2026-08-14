#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates SDPA forward golden reference bundles for quick and standard tiers.
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

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at: $GENERATOR"
    exit 1
fi

TIER="${1:-all}"

# DVC remote that hipDNN SDPA golden bundles live on. Each generated bundle's
# .tensors.dvc pointer is written with a per-output `remote:` key set to this, so
# a bare `dvc pull` (and CI) fetches the data from the right place without any
# `-r` flag or workflow change. See ../../README.md ("DVC Remote Layout").
DVC_REMOTE="golden-data"

generate_bundle() {
    local outdir="$1"
    local name="$2"
    shift 2
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

# --- quick tier: Small bundles only ---
if [[ "$TIER" == "all" || "$TIER" == "quick" ]]; then
    echo "=== Generating quick tier ==="

    # BF16 non-stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # BF16 stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "SmallStats" --stats --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # FP16 non-stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # FP16 stats
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "SmallStats" --stats --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # BF16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # FP16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    # BF16 hd192 (D_qk=192, D_v=128), no mask + causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42

    # FP16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp16 --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --dtype fp16 --q-dims 2 4 256 192 --v-dims 2 4 256 128 --seed 42

    echo ""
fi

# --- standard tier: Medium and Gqa bundles ---
if [[ "$TIER" == "all" || "$TIER" == "standard" ]]; then
    echo "=== Generating standard tier ==="

    # BF16 non-stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # BF16 stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "MediumStats" --stats --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "GqaStats" --stats --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # FP16 non-stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # FP16 stats
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"
    generate_bundle "$OUTDIR" "MediumStats" --stats --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "GqaStats" --stats --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # BF16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # FP16 hd128 bottom-right causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --dtype fp16 --q-dims 2 4 512 128 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --dtype fp16 --q-dims 1 8 256 128 --v-dims 1 2 256 128 --seed 42

    # BF16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/bf16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42

    # FP16 hd192, no mask + causal
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd192_nomask_batch"
    generate_bundle "$OUTDIR" "Medium" --dtype fp16 --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --dtype fp16 --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/standard/SdpaFwd/bhsd/fp16/hd192_causal_batch"
    generate_bundle "$OUTDIR" "Medium" --causal bottom_right --dtype fp16 --q-dims 2 4 512 192 --v-dims 2 4 512 128 --seed 42
    generate_bundle "$OUTDIR" "Gqa" --causal bottom_right --dtype fp16 --q-dims 1 8 256 192 --v-dims 1 2 256 128 --seed 42

    echo ""
fi

# --- Tier B: FP8 and GROUP-mode bundles (generator-complete; not yet validated by
# the CPU reference executor, which does not accept FP8 descale or variable-seq-len
# attributes). Generate with GENERATE_TIER_B=1. ---
if [[ "${GENERATE_TIER_B:-0}" == "1" ]]; then
    echo "=== Generating Tier B (fp8 + group) ==="

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_nomask_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_causal_batch"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_nomask_group"
    generate_bundle "$OUTDIR" "Small" --variable-seq-lens --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42
    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/bf16/hd128_causal_group"
    generate_bundle "$OUTDIR" "Small" --causal bottom_right --variable-seq-lens --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42

    OUTDIR="$GOLDEN_ROOT/quick/SdpaFwd/bhsd/fp8/hd128_causal_group"
    generate_bundle "$OUTDIR" "Small" --dtype fp8 --causal bottom_right --variable-seq-lens --seq-lens-q 256 384 512 --seq-lens-kv 256 384 512 --q-dims 3 4 512 128 --v-dims 3 4 512 128 --seed 42

    echo ""
fi

echo "=== Done ==="
echo "Generated bundles:"
find "$GOLDEN_ROOT/quick/SdpaFwd" "$GOLDEN_ROOT/standard/SdpaFwd" -name "*.json" ! -name "*.meta.json" 2>/dev/null | sort
