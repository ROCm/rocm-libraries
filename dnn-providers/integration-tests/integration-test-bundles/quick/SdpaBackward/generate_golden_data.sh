#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates SDPA backward golden reference bundles (RFC 0011). The CPU and GPU
# backward references are otherwise checked mainly against each other, so a
# shared misunderstanding passes those gate tests; these bundles come from
# PyTorch, an independent source, and catch it.
#
# Every bundle is produced by a single pipeline run, which RFC 0011 requires for
# backward operations: Q, K, V and dO are drawn from one seeded RNG stream, the
# forward pass produces O and LSE and is saved as its own bundle, and the
# backward pass runs on exactly that O and LSE. The backward bundle records the
# forward bundle it came from in its `forward_source` metadata, so the pairing
# stays checkable after the fact. That is why each case below writes two
# bundles, and why the forward half must not be regenerated on its own.
#
# Requires PyTorch. A GPU is not needed: the reference runs on the CPU.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/quick/SdpaBackward
#   bash generate_golden_data.sh              # Generate all tiers
#   bash generate_golden_data.sh quick        # Generate quick tier only
#
# After generating, register the tensor payloads with DVC from the repo root:
#   dvc commit -f <bundle>/<Name>.tensors.dvc
#   dvc push -r golden-data
# See ../../../docs/adding-tests.md ("Golden data") for the full workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOLDEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$(cd "$GOLDEN_ROOT/../reference-data-scripts" && pwd)/generate_sdpa_bwd_golden.py"

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

# DVC remote that hipDNN SDPA golden bundles live on. Each generated bundle's
# .tensors.dvc pointer is written with a per-output `remote:` key set to this, so
# a bare `dvc pull` (and CI) fetches the data from the right place without any
# `-r` flag or workflow change. See ../../../docs/file-formats.md ("Golden data").
DVC_REMOTE="golden-data"

# RFC 0011 budgets 2 MiB of committed tensor payload per case and asks for 1 MiB
# or less where possible. A backward bundle carries eight IO-dtype tensors (Q, K,
# V, O, dO and the three gradients) plus FP32 stats, against the forward bundle's
# four, so the shapes here are deliberately smaller than the SdpaFwd driver's:
# B=1, H=1, S=256 holds a hd128 bundle at ~512 KiB and a hd192 one at ~640 KiB.
# Raising any of B, H, S or D_qk pushes a bundle past the warning threshold.
SHAPE_HD128=(--q-dims 1 1 256 128 --v-dims 1 1 256 128 --attn-scale "$SCALE_D128")
SHAPE_HD192=(--q-dims 1 1 256 192 --v-dims 1 1 256 128 --attn-scale "$SCALE_D192")

# Author the per-bundle DVC pointer with the golden-data remote pinned per
# output. DVC fills in md5/size on `dvc commit`; the `remote:` key persists
# across regenerations. This is what routes the bundle to the golden-data
# remote -- do not drop it, or CI will fail to pull the data.
write_dvc_pointer() {
    local bundle_dir="$1"
    local name="$2"
    local bin
    {
        echo "outs:"
        for bin in "$bundle_dir/$name".tensor*.bin; do
            echo "- path: $(basename "$bin")"
            echo "  remote: $DVC_REMOTE"
        done
    } > "$bundle_dir/$name.tensors.dvc"
}

# Generate one forward/backward bundle pair. The forward half is written under
# the matching SdpaFwd tier directory with a `_bwd_pair` suffix, which keeps it
# clearly owned by this script: the SdpaFwd driver must never regenerate it on
# its own, or the backward bundle's O and LSE would stop matching the data its
# gradients were computed from.
generate_pair() {
    local tier="$1"
    local topology="$2"
    local name="$3"
    shift 3

    local bwd_dir="$GOLDEN_ROOT/$tier/SdpaBackward/$topology/$name"
    local fwd_dir="$GOLDEN_ROOT/$tier/SdpaFwd/${topology}_bwd_pair/$name"
    mkdir -p "$bwd_dir" "$fwd_dir"

    python3 "$GENERATOR" \
        --base-filename "$bwd_dir/$name" \
        --forward-base-filename "$fwd_dir/$name" \
        --seed 42 \
        "$@"

    write_dvc_pointer "$bwd_dir" "$name"
    write_dvc_pointer "$fwd_dir" "$name"
}

# --- quick tier ---
#
# The mask configurations mirror the graph-only cases already checked in under
# quick/SdpaBackward/Default/sweep.json, so the golden data covers the shapes the
# sweep already exercises. Features the generator does not support yet (GQA, FP8,
# dropout, bias/dBias, ALiBi and variable length) are deliberately absent; add a
# case here as each reference feature lands.
if [[ "$TIER" == "all" || "$TIER" == "quick" ]]; then
    echo "=== Generating quick tier ==="

    for DTYPE in bf16 fp16; do
        # No mask.
        generate_pair quick "bhsd/$DTYPE/hd128_nomask_batch" Small \
            --dtype "$DTYPE" "${SHAPE_HD128[@]}"

        # Top-left causal.
        generate_pair quick "bhsd/$DTYPE/hd128_causal_top_left_batch" Small \
            --dtype "$DTYPE" --right-bound 0 --diagonal-alignment top_left \
            "${SHAPE_HD128[@]}"

        # Bottom-right causal, the alignment production models use.
        generate_pair quick "bhsd/$DTYPE/hd128_causal_batch" Small \
            --dtype "$DTYPE" --right-bound 0 --diagonal-alignment bottom_right \
            "${SHAPE_HD128[@]}"

        # Sliding window: bounded on both sides of the diagonal. This is the
        # configuration a backward reference is most likely to get wrong, since
        # dQ and dK must drop the same positions the forward did.
        generate_pair quick "bhsd/$DTYPE/hd128_window_batch" Small \
            --dtype "$DTYPE" --left-bound 64 --right-bound 64 \
            --diagonal-alignment top_left "${SHAPE_HD128[@]}"

        # D_qk != D_v, which exercises the separate dQ/dK and dV shapes.
        generate_pair quick "bhsd/$DTYPE/hd192_nomask_batch" Small \
            --dtype "$DTYPE" "${SHAPE_HD192[@]}"
        generate_pair quick "bhsd/$DTYPE/hd192_causal_batch" Small \
            --dtype "$DTYPE" --right-bound 0 --diagonal-alignment bottom_right \
            "${SHAPE_HD192[@]}"
    done

    echo ""
fi

echo "=== Done ==="
echo "Generated bundles:"
find "$GOLDEN_ROOT" \
    \( -path "*/SdpaBackward/*" -o -path "*_bwd_pair/*" \) \
    -name "*.json" ! -name "*.meta.json" ! -name "*.support.json" \
    ! -name "graph.template.json" ! -name "sweep.json" 2>/dev/null | sort
echo ""
echo "Next: 'dvc commit -f' each .tensors.dvc above, then 'dvc push -r $DVC_REMOTE'."
