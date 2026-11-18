#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regenerates the comprehensive-tier ragged SDPA forward bundles whose per-batch
# lengths pin how an engine sizes a packed (RFC 0014) batch from the graph's S_max.
# A ragged tensor's dims[2] is an upper bound on every per-batch length (normally the
# maximum), not the packed token total; these bundles fail if an engine launches or
# addresses from anything smaller than every per-batch length, or treats S_max as a
# per-batch row count. Shapes stay under verify_golden_bundles.py's 2 MiB cap.
# Requires PyTorch.
#
# Usage:
#   cd <repo-root>/dnn-providers/integration-tests/integration-test-bundles/comprehensive/SdpaFwd
#   bash generate_ragged_lengths_golden_data.sh
#   # then: dvc commit -f <bundle dirs> && dvc push -r golden-data <bundle dirs>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/bshd/bf16"
GENERATOR="$(cd "$SCRIPT_DIR/../../../reference-data-scripts" && pwd)/generate_sdpa_fwd_ragged_lengths_golden.py"

SCALE_D128="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(128)))')"
SCALE_D192="$(python3 -c 'import math; print(repr(1.0 / math.sqrt(192)))')"

gen() {
    local name="$1"
    shift
    python3 "$GENERATOR" --base-filename "$OUT/$name/$name" "$@"
}

# The longest Q sequence is the LAST batch and equals S_q exactly (600: three 256-row Q
# tiles, the last partial); the longest K/V sequence sits mid-batch. Batch 8 has no Q
# tokens. Fails if the Q-tile grid comes from anything below max_b(len_q) (the tail rows
# of batch 12 stay unwritten), or if a batch stride leaks into the packed addressing.
LONG_LAST_Q=(1 2 3 17 31 64 100 129 0 255 257 150 600)
LONG_LAST_KV=(5 1 300 17 650 64 1 129 7 255 260 33 599)
gen hd128_nomask_ragged_long_last --heads 1 1 --head-dims 128 128 --seq-dims 600 650 \
    --lens-q "${LONG_LAST_Q[@]}" --lens-kv "${LONG_LAST_KV[@]}" \
    --attn-scale "$SCALE_D128" \
    --description "S_q = max len in the last batch; grid under-launch / batch-stride leak"

# Same lengths, bottom-right causal. Batches 1, 6 and 11 have more Q than K/V tokens, so
# their leading rows are fully masked (O = 0); every batch aligns the diagonal on its own
# lengths, which differ from S_kv - S_q of the dims. Batch 12 needs three Q tiles, so the
# causal head/tail tile pairing (two workgroups) must still cover its middle tile.
gen hd128_causal_ragged_long_last --heads 1 1 --head-dims 128 128 --seq-dims 600 650 \
    --lens-q "${LONG_LAST_Q[@]}" --lens-kv "${LONG_LAST_KV[@]}" --causal bottom_right \
    --attn-scale "$SCALE_D128" \
    --description "per-batch bottom-right alignment and causal tile pairing under S_q = max len"

# hd192x128 causal, Q lengths straddling the 128-row tile edges. The gfx942 group kernel
# derives its tile index from s_seq_len (ceil(s_seq_len / 128) - 1 - blockIdx.z), so
# s_seq_len and the grid must both come from the same S_q >= max len. Batch 5 has Q
# tokens but no K/V tokens (O = 0).
gen hd192_causal_ragged_tile_edges --heads 1 1 --head-dims 192 128 --seq-dims 385 500 \
    --lens-q 127 128 129 256 257 1 385 --lens-kv 128 130 1 256 500 0 385 \
    --causal bottom_right --attn-scale "$SCALE_D192" \
    --description "hd192 causal: s_seq_len/grid coupling at tile boundaries; empty K/V batch"

# 96 batches, mostly 0-5 tokens, one long Q batch (index 50) and one long K/V batch
# (index 70): the packed totals exceed S_max, and the offset table is long.
MANY_Q=()
MANY_KV=()
for b in $(seq 0 95); do
    q=(1 2 3 0 4)
    kv=(3 1 0 2 5)
    MANY_Q+=("${q[$((b % 5))]}")
    MANY_KV+=("${kv[$((b % 5))]}")
done
MANY_Q[50]=300
MANY_KV[70]=400
gen hd192_nomask_ragged_many_batches --heads 2 2 --head-dims 192 128 --seq-dims 300 400 \
    --lens-q "${MANY_Q[@]}" --lens-kv "${MANY_KV[@]}" --attn-scale "$SCALE_D192" \
    --description "B=96 mostly tiny/empty batches; total tokens > S_max"

# Declared S_q/S_kv (2048/4096) well above every per-batch length (300/400): S_max is
# only an upper bound, so over-launched tiles must not touch a neighbouring batch's rows.
gen hd128_nomask_ragged_gqa_smax_slack --heads 4 1 --head-dims 128 128 --seq-dims 2048 4096 \
    --lens-q 300 13 0 129 90 1 --lens-kv 400 1 40 256 0 77 --attn-scale "$SCALE_D128" \
    --description "S_max above every length; GQA 4:1; empty Q and empty K/V batches"

# Declared S_q = S_kv = 20000 with at most 400 real tokens per batch (masked, so the
# tune_opt downgrade path runs). Fails if S_max is used as a per-batch row count.
gen hd128_causal_ragged_smax_slack --heads 2 2 --head-dims 128 128 --seq-dims 20000 20000 \
    --lens-q 300 1 257 0 128 --lens-kv 300 2 100 5 400 --causal bottom_right \
    --attn-scale "$SCALE_D128" \
    --description "S_max = 20000 >> lengths with a causal mask"
