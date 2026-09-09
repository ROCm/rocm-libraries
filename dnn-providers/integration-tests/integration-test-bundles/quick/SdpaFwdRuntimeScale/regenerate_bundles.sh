#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates all SdpaFwdRuntimeScale test bundles (quick tier).
#
# These bundles carry no tensor blobs (see generate_sdpa_runtime_scale_test_bundle.py):
# the harness synthesizes inputs and reference-verifies outputs at test time, so
# there is nothing here to pin to a DVC remote, unlike ../SdpaFwd/generate_golden_data.sh.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwdRuntimeScale
#   bash regenerate_bundles.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$(cd "$GOLDEN_ROOT/../reference-data-scripts" && pwd)/generate_sdpa_runtime_scale_test_bundle.py"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at: $GENERATOR"
    exit 1
fi

generate_bundle() {
    local outdir="$1"
    local name="$2"
    shift 2
    mkdir -p "$outdir/$name"
    python3 "$GENERATOR" --base-filename "$outdir/$name/$name" "$@"
}

BHSD="$GOLDEN_ROOT/quick/SdpaFwdRuntimeScale/bhsd"

echo "=== Generating SdpaFwdRuntimeScale quick tier ==="

# --- bf16, hd128 ---
generate_bundle "$BHSD/bf16/hd128_nomask_batch" "Small" \
    --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_nomask_batch_Small

generate_bundle "$BHSD/bf16/hd128_nomask_group" "Small" \
    --variable-seq-lens --q-dims 3 4 512 128 --v-dims 3 4 512 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_nomask_group_Small

generate_bundle "$BHSD/bf16/hd128_nomask_batch_stats" "SmallStats" \
    --stats --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_nomask_batch_stats_SmallStats

generate_bundle "$BHSD/bf16/hd128_causal_batch" "Small" \
    --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_causal_batch_Small

generate_bundle "$BHSD/bf16/hd128_causal_group" "Small" \
    --causal bottom_right --variable-seq-lens \
    --q-dims 3 4 512 128 --v-dims 3 4 512 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_causal_group_Small

# --- bf16, hd192 (D_qk=192, D_v=128) ---
generate_bundle "$BHSD/bf16/hd192_nomask_batch" "Small" \
    --q-dims 2 4 256 192 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd192_nomask_batch_Small

generate_bundle "$BHSD/bf16/hd192_causal_batch" "Small" \
    --causal bottom_right --q-dims 2 4 256 192 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_bf16_hd192_causal_batch_Small

# --- fp16, hd128 ---
generate_bundle "$BHSD/fp16/hd128_nomask_batch" "Small" \
    --dtype fp16 --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp16_hd128_nomask_batch_Small

generate_bundle "$BHSD/fp16/hd128_nomask_batch_stats" "SmallStats" \
    --dtype fp16 --stats --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp16_hd128_nomask_batch_stats_SmallStats

generate_bundle "$BHSD/fp16/hd128_causal_batch" "Small" \
    --dtype fp16 --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp16_hd128_causal_batch_Small

# --- fp16, hd192 (D_qk=192, D_v=128) ---
generate_bundle "$BHSD/fp16/hd192_nomask_batch" "Small" \
    --dtype fp16 --q-dims 2 4 256 192 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp16_hd192_nomask_batch_Small

generate_bundle "$BHSD/fp16/hd192_causal_batch" "Small" \
    --dtype fp16 --causal bottom_right --q-dims 2 4 256 192 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp16_hd192_causal_batch_Small

# --- fp8, hd128 ---
generate_bundle "$BHSD/fp8/hd128_nomask_batch" "Small" \
    --dtype fp8 --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp8_hd128_nomask_batch_Small

generate_bundle "$BHSD/fp8/hd128_causal_batch" "Small" \
    --dtype fp8 --causal bottom_right --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp8_hd128_causal_batch_Small

generate_bundle "$BHSD/fp8/hd128_causal_group" "Small" \
    --dtype fp8 --causal bottom_right --variable-seq-lens \
    --q-dims 3 4 512 128 --v-dims 3 4 512 128 \
    --name SdpaFwdRuntimeScale_bhsd_fp8_hd128_causal_group_Small

echo ""
echo "=== Done ==="
echo "Generated bundles:"
find "$BHSD" -name "*.json" ! -name "*.meta.json" 2>/dev/null | sort
