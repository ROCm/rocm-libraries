#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates SDPA forward golden reference bundles.
# Requires PyTorch with ROCm support.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration_test_bundles/quick/SdpaFwd
#   bash generate_golden_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$(cd "$SCRIPT_DIR" && cd ../../../reference_data_scripts && pwd)/generate_sdpa_fwd_golden.py"

if [[ ! -f "$GENERATOR" ]]; then
    echo "ERROR: Generator script not found at: $GENERATOR"
    exit 1
fi

# --- BF16 hd128, no mask, batch mode (BHSD layout) ---

OUTDIR="$SCRIPT_DIR/bhsd/bf16/hd128_nomask_batch"

echo "=== Generating BF16 hd128 nomask batch bundles ==="

for bundle in Small Medium Gqa; do
    mkdir -p "$OUTDIR/$bundle"
done

python3 "$GENERATOR" \
    --base-filename "$OUTDIR/Small/Small" \
    --q-dims 2 4 256 128 --v-dims 2 4 256 128 \
    --seed 42

python3 "$GENERATOR" \
    --base-filename "$OUTDIR/Medium/Medium" \
    --q-dims 2 4 512 128 --v-dims 2 4 512 128 \
    --seed 42

python3 "$GENERATOR" \
    --base-filename "$OUTDIR/Gqa/Gqa" \
    --q-dims 1 8 256 128 --v-dims 1 2 256 128 \
    --seed 42

echo "=== Done. Golden data written to: $OUTDIR ==="
find "$OUTDIR" -type f | sort
