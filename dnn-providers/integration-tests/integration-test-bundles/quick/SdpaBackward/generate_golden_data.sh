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
# bundles, both under SdpaBackward: the backward bundle, and its forward half in
# a sibling `<topology>_fwd_pair` directory. Keeping the pair together is what
# stops the SdpaFwd driver from regenerating the forward half out from under it.
#
# Requires PyTorch. A GPU is not needed: the reference runs on the CPU.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/quick/SdpaBackward
#   bash generate_golden_data.sh              # Generate all tiers
#   bash generate_golden_data.sh quick        # Generate quick tier only
#   GENERATE_DISCRIMINATING=1 bash generate_golden_data.sh none   # Discriminating bundles only
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

# Order the payloads of a bundle deterministically. Glob expansion sorts by the
# ambient collation, which puts tensor12 before tensor1 in some locales and after
# it in others -- enough to make the committed pointer differ between machines
# even when every md5 matches. Sorting by the numeric UID removes that.
sorted_tensor_bins() {
    local bin uid
    for bin in "$@"; do
        uid="$(basename "$bin")"
        uid="${uid##*tensor}"   # Small.tensor12.bin -> 12.bin
        uid="${uid%%.bin}"
        printf '%s\t%s\n' "$uid" "$bin"
    done | sort -n -k1,1 | cut -f2-
}

# Emit one `outs:` entry for a tensor payload. DVC's hash for a file output is
# the plain MD5 of its contents, so the pointer can be written complete here
# rather than left for `dvc commit` to fill in -- which matters because the
# pointer is what gets committed to git, and every pointer already in the tree
# carries its md5 and size. `dvc commit` then only confirms what is written.
#
# The `remote:` key is what routes the payload to the golden-data remote, so a
# bare `dvc pull` (and CI) fetches it without any `-r` flag or workflow change.
# Do not drop it, or CI will fail to pull the data.
write_dvc_out() {
    local bin="$1"
    echo "- path: $(basename "$bin")"
    echo "  remote: $DVC_REMOTE"
    echo "  md5: $(md5sum "$bin" | cut -d' ' -f1)"
    echo "  size: $(stat -c %s "$bin")"
    echo "  hash: md5"
}

# Author the DVC pointer for a single-graph bundle, whose payloads are named
# <BundleName>.tensor<uid>.bin.
write_dvc_pointer() {
    local bundle_dir="$1"
    local name="$2"
    local bin
    {
        echo "outs:"
        while IFS= read -r bin; do
            write_dvc_out "$bin"
        done < <(sorted_tensor_bins "$bundle_dir/$name".tensor*.bin)
    } > "$bundle_dir/$name.tensors.dvc"
}

# Generate one forward/backward bundle pair. The forward half is written beside
# the backward bundle under SdpaBackward, in a sibling directory with a
# `_fwd_pair` suffix, which keeps it
# clearly owned by this script, and keeps it out of SdpaFwd, whose own driver
# must never regenerate it: were the forward half regenerated on its own, the
# backward bundle's O and LSE would stop matching the data its gradients were
# computed from, and the pairing its forward_source metadata records would be a
# lie that nothing would catch.
generate_pair() {
    local tier="$1"
    local topology="$2"
    local name="$3"
    shift 3

    local bwd_dir="$GOLDEN_ROOT/$tier/SdpaBackward/$topology/$name"
    local fwd_dir="$GOLDEN_ROOT/$tier/SdpaBackward/${topology}_fwd_pair/$name"
    mkdir -p "$bwd_dir" "$fwd_dir"

    python3 "$GENERATOR" \
        --base-filename "$bwd_dir/$name" \
        --forward-base-filename "$fwd_dir/$name" \
        --seed 42 \
        "$@"

    write_dvc_pointer "$bwd_dir" "$name"
    write_dvc_pointer "$fwd_dir" "$name"
}

# The sweep equivalent of write_dvc_pointer. A sweep case's golden directory is
# resolved as the parent of its golden.path, and the tensors inside it are named
# tensor<uid>.bin without a bundle-name prefix, so the pointer lists those names.
write_sweep_dvc_pointer() {
    local golden_dir="$1"
    local bin
    {
        echo "outs:"
        while IFS= read -r bin; do
            write_dvc_out "$bin"
        done < <(sorted_tensor_bins "$golden_dir"/tensor*.bin)
    } > "$golden_dir/tensors.dvc"
}

# Attach golden data to one case of a checked-in template sweep. Unlike
# generate_pair, the graph topology is not written here: it already exists as the
# topology's graph.template.json, and the generator only contributes the case's
# tensor values and the case entry that points at them. That is the whole point
# of the template sweep -- the same topology the graph-only cases exercise is
# re-run against data from an independent source.
#
# The forward half still has to go somewhere, since RFC 0011 requires it to be
# saved; it is written as an ordinary single-graph bundle under
# <topology>_fwd_pair/<case id>/Small, for the same reasons generate_pair does.
# The trailing Small level is what makes the directory name match the graph file
# name, which is how a single-graph bundle is recognised.
generate_sweep_case() {
    local tier="$1"
    local topology="$2"
    local case_id="$3"
    shift 3

    local sweep_dir="$GOLDEN_ROOT/$tier/SdpaBackward/$topology"
    local fwd_dir="$GOLDEN_ROOT/$tier/SdpaBackward/${topology}_fwd_pair/$case_id/Small"

    if [[ ! -f "$sweep_dir/graph.template.json" ]]; then
        echo "ERROR: no graph.template.json in $sweep_dir"
        exit 1
    fi
    mkdir -p "$fwd_dir"

    python3 "$GENERATOR" \
        --sweep-bundle-dir "$sweep_dir" \
        --case-id "$case_id" \
        --forward-base-filename "$fwd_dir/Small" \
        --seed 42 \
        "$@"

    write_sweep_dvc_pointer "$sweep_dir/golden/$case_id"
    write_dvc_pointer "$fwd_dir" Small
}

# --- quick tier ---
#
# The mask configurations mirror the graph-only cases already checked in under
# quick/SdpaBackward/Default/sweep.json, so the golden data covers the shapes the
# sweep already exercises. GQA, padding, bias/dBias and dropout are single-graph only
# and live in the discriminating bundles below; FP8 and ALiBi are not supported by the
# generator yet.
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

# --- quick tier: template-sweep cases ---
#
# These reuse the graph-only topologies under quick/SdpaBackward/ rather than
# adding new ones, so the golden data runs through exactly the graph the sweep's
# captured cases run through. The captured cases are left untouched: they declare
# rank-3 stats, which the CPU backward reference rejects (it requires
# [B, H_q, S_q, 1]), so golden data could not be attached to them as they stand.
#
# Both templates hard-code attn_scale_value = 1/sqrt(128), so every case here is
# D_qk = 128 and passes the matching --attn-scale. S = 128 keeps a case near
# 300 KiB, comfortably inside the RFC 0011 per-case budget even with the mask.
if [[ "$TIER" == "all" || "$TIER" == "quick" ]]; then
    echo "=== Generating quick tier template-sweep cases ==="

    SWEEP_SHAPE=(--q-dims 1 1 128 128 --v-dims 1 1 128 128 --attn-scale "$SCALE_D128")

    for DTYPE in bf16 fp16; do
        generate_sweep_case quick Default "golden_1_1_128_128_${DTYPE}_nomask" \
            --dtype "$DTYPE" "${SWEEP_SHAPE[@]}"

        generate_sweep_case quick Default "golden_1_1_128_128_${DTYPE}_causal" \
            --dtype "$DTYPE" --right-bound 0 --diagonal-alignment bottom_right \
            "${SWEEP_SHAPE[@]}"

        generate_sweep_case quick AdditiveMask \
            "golden_1_1_128_128_${DTYPE}_nomask" \
            --dtype "$DTYPE" --additive-mask "${SWEEP_SHAPE[@]}"

        # An additive bias under a causal band: the bias must not resurrect a
        # position the band drops, which is the ordering the CPU reference fixes
        # and the easiest thing for a backward implementation to invert.
        generate_sweep_case quick AdditiveMask \
            "golden_1_1_128_128_${DTYPE}_causal" \
            --dtype "$DTYPE" --additive-mask --right-bound 0 \
            --diagonal-alignment bottom_right "${SWEEP_SHAPE[@]}"
    done

    echo ""
fi

# --- Discriminating bundles: Sq != Skv everywhere, per-batch lengths that differ
# (batch 1 of the padded set has Sq > Skv, so bottom-right leaves it dead rows:
# O = 0, LSE = -inf, zero gradients), GQA, broadcast bias with dBias, and an
# explicit dropout mask. A reference that takes the bottom-right offset from the
# maximum lengths, forgets to sum a GQA group, reduces dBias over the wrong axes
# or drops dropout_scale_inv fails at least one of them. Each is cross-checked
# against FP64 autograd at generation. Generate in isolation with:
#   GENERATE_DISCRIMINATING=1 bash generate_golden_data.sh none ---
if [[ "${GENERATE_DISCRIMINATING:-0}" == "1" ]]; then
    echo "=== Generating discriminating bundles ==="

    PADNEQ=(--q-dims 3 2 64 64 --v-dims 3 2 96 64 --seq-lens-q 17 64 5
            --seq-lens-kv 96 40 33 --attn-scale 0.125 --dtype bf16)
    NEQ_GQA=(--q-dims 2 4 64 64 --v-dims 2 2 96 64 --attn-scale 0.125 --dtype bf16)
    T=bhsd/bf16

    generate_pair quick "$T/padneq_nomask" Small "${PADNEQ[@]}"
    generate_pair quick "$T/padneq_causal_tl" Small "${PADNEQ[@]}" \
        --right-bound 0 --diagonal-alignment top_left
    generate_pair quick "$T/padneq_causal_br" Small "${PADNEQ[@]}" \
        --right-bound 0 --diagonal-alignment bottom_right
    generate_pair quick "$T/padneq_window_br" Small "${PADNEQ[@]}" \
        --left-bound 16 --right-bound 0 --diagonal-alignment bottom_right
    generate_pair quick "$T/neq_gqa_causal_br" Small "${NEQ_GQA[@]}" \
        --right-bound 0 --diagonal-alignment bottom_right
    generate_pair quick "$T/neq_gqa_dbias_heads" Small "${NEQ_GQA[@]}" \
        --additive-mask --bias-dims 1 4 64 96 --dbias \
        --right-bound 0 --diagonal-alignment bottom_right
    generate_pair quick "$T/padneq_dbias_batch" Small "${PADNEQ[@]}" \
        --additive-mask --bias-dims 3 1 64 96 --dbias
    # Inputs in [0, 1] make D = rowsum(dO * O) as large as dP, so a wrong
    # dropout_scale_inv changes the gradients by O(1) instead of a few percent.
    generate_pair quick "$T/neq_gqa_dropout_causal_br" Small "${NEQ_GQA[@]}" \
        --dropout-p 0.2 --right-bound 0 --diagonal-alignment bottom_right \
        --min 0 --max 1

    echo ""
fi

echo "=== Done ==="
echo "Generated bundles:"
find "$GOLDEN_ROOT" -path "*/SdpaBackward/*" \
    -name "*.json" ! -name "*.meta.json" ! -name "*.support.json" \
    ! -name "graph.template.json" ! -name "sweep.json" 2>/dev/null | sort
echo ""
echo "Next: 'dvc commit -f' each .tensors.dvc above, then 'dvc push -r $DVC_REMOTE'."
