#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates SDPA forward golden reference bundles for quick and standard tiers.
# Requires PyTorch with ROCm support.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration_test_bundles/quick/SdpaFwd
#   bash generate_golden_data.sh              # Generate all tiers
#   bash generate_golden_data.sh quick        # Generate quick tier only
#   bash generate_golden_data.sh standard     # Generate standard tier only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$(cd "$GOLDEN_ROOT/../reference_data_scripts" && pwd)/generate_sdpa_fwd_golden.py"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at: $GENERATOR"
    exit 1
fi

TIER="${1:-all}"

generate_bundle() {
    local outdir="$1"
    local name="$2"
    shift 2
    mkdir -p "$outdir/$name"
    python3 "$GENERATOR" --base-filename "$outdir/$name/$name" "$@"
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

    echo ""
fi

echo "=== Done ==="
echo "Generated bundles:"
find "$GOLDEN_ROOT/quick/SdpaFwd" "$GOLDEN_ROOT/standard/SdpaFwd" -name "*.json" ! -name "*.meta.json" 2>/dev/null | sort
