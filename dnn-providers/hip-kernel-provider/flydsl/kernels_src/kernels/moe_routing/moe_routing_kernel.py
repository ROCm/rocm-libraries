# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Ported from: https://github.com/ROCm/FlyDSL
#   commit:   226cde8d2a84bc2687c8d69587607fddf0c16f2f  (v0.5.0)
#   path:     kernels/moe/moe_sorting_kernel.py
#   compiler: flydsl 0.3.4 (pip wheel)
#
# What is kept: upstream's algorithm and decomposition -- a counting sort over the
# router's (token, slot) pairs in three phases, histogram -> prefix sum -> scatter,
# with the multiphase path's HBM workspace between kernels, buffer-resource loads and
# stores whose out-of-range offsets the hardware drops, and LDS for each block's
# per-expert state.
#
# Modifications (rewritten against the 0.3.4 API; upstream is written for 0.5.0):
#   1. Output: hipDNN's MoeRouting (moe_routing_attributes.fbs) instead of CK's padded
#      sorted ids. Routed row r gets token_index[r] = s and token_ks[r] = k, and
#      first_token_offset[e] counts the rows with expert < e (e in [0, E]). No
#      unit-size padding, sorted weights, sorted expert ids or moe_buf zeroing.
#   2. Order: rows are ordered by expert, then token, then slot -- a stable counting
#      sort over the row-major (s, k) order -- and one token may pick an expert more
#      than once (each pick is its own row, in slot order). Upstream's expert mesh
#      holds one slot per (expert, token), so it assumes unique experts per token.
#      Here each wave groups its lanes by expert with one ballot per id bit, and the
#      waves of a block take their rows in wave order: deterministic, no atomics.
#   3. Expert parallelism: an id outside [0, E) is not routed (upstream masks experts
#      through a separate expert_mask tensor), and the rows past first_token_offset[E]
#      are not written.
#   4. Workspace: one int32 count per (expert, block) instead of upstream's
#      E x tokens byte mesh, so it does not grow with the token count. The histogram
#      kernel writes each block's per-expert counts, expert-major; the prefix kernel
#      (one block) scans them in place, which makes each entry the block's first row
#      for the expert, and writes first_token_offset; the scatter kernel places each
#      block's pairs from those bases. The block count and each block's span are
#      launch arguments.
#   5. Runtime shapes: E, top_k, the pair count, the block count and the span are
#      kernel arguments; E is bounded by the LDS table (E_MAX). first_token_offset is
#      written as int32 or, on a runtime flag, as int64 (the low word, then a zero
#      high word: every offset is non-negative and below 2^31).
"""MoE routing (hipDNN MoeRouting) as a three-kernel counting sort."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.kernels_common import get_warp_size

# The histogram and scatter kernels' block: one pair per thread per tile.
BLOCK_THREADS = 256
# The prefix kernel's single block.
PREFIX_THREADS = 1024
# The per-expert LDS table's size: the largest E an object serves.
E_MAX = 8192
# The lane masks are 64-bit ballots.
WAVE_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE

# An element offset past every buffer: the load returns 0, the store is dropped.
_OOB = 0x7FFFFFFF
# Workspace entries each prefix thread scans per chunk: one 16-byte load.
PREFIX_ITEMS = 4
PREFIX_CHUNK = PREFIX_THREADS * PREFIX_ITEMS


def _make_tile_storage(e_max: int):
    @fx.struct
    class TileStorage:
        per_expert: fx.Array[fx.Int32, e_max, 16]

    return TileStorage


def _load_i32(rsrc, at, valid):
    """rsrc[at] where `valid`, else 0 (the load is dropped)."""
    offset = valid.select(at, fx.Int32(_OOB))
    return fx.Int32(buffer_ops.buffer_load(rsrc, offset, vec_width=1, dtype=T.i32))


def _same_expert_lanes(key, routed, key_bits: int):
    """The routed lanes of this wave whose key equals this lane's (garbage if unrouted).

    A match-any built from one ballot per key bit: starting from the routed lanes,
    each bit keeps the lanes that agree with this lane on it. The cost is fixed by
    the key width, not by how many distinct experts the wave holds.
    """
    lanes = fx.Int64(rocdl.ballot(T.i64, routed.ir_value()))
    for bit in range_constexpr(key_bits):
        is_set = ((key >> fx.Int32(bit)) & fx.Int32(1)) != fx.Int32(0)
        ones = fx.Int64(rocdl.ballot(T.i64, is_set.ir_value()))
        lanes = lanes & is_set.select(ones, ones ^ fx.Int64(-1))
    return lanes


def build_moe_routing_module(e_max: int = E_MAX):
    """The histogram, prefix and scatter kernels, and a launcher running all three.

    Every tensor is a flat int32 view: TopkIds [n_pairs] (row-major [S, top_k]),
    Counts (the workspace: E * num_blocks entries, expert-major -- block b's count of
    expert e at e * num_blocks + b -- allocated to a multiple of PREFIX_ITEMS),
    FirstTokenOffset [(E + 1) * (2 if int64 else 1)], TokenIndex and TokenKs
    [n_pairs]. Block b covers pairs [b * span, (b + 1) * span).
    """
    if get_warp_size() != WAVE_SIZE:
        raise ValueError(
            f"moe_routing groups lanes with 64-bit ballots; this target's wave is "
            f"{get_warp_size()} lanes"
        )
    TileStorage = _make_tile_storage(e_max)
    # Bits of the largest routed id: what the lane matching compares.
    KEY_BITS = max(1, (e_max - 1).bit_length())
    block_scan = fx.coop.BlockScan[fx.Int32, PREFIX_THREADS]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def moe_routing_histogram_kernel(
        TopkIds: fx.Tensor,
        Counts: fx.Tensor,
        n_pairs: fx.Int32,
        num_experts: fx.Int32,
        span: fx.Int32,
        num_blocks: fx.Int32,
    ):
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(WAVE_SIZE)
        lanes_below = (fx.Int64(1) << fx.Int64(tid % fx.Int32(WAVE_SIZE))) - fx.Int64(1)
        topk_rsrc = buffer_ops.create_buffer_resource(TopkIds, max_size=True)
        counts_rsrc = buffer_ops.create_buffer_resource(Counts, max_size=True)

        lds = fx.SharedAllocator().allocate(TileStorage).peek()
        s_hist = lds.per_expert.view(fx.make_layout(e_max, 1))

        for e in range(tid, num_experts, BLOCK_THREADS):
            fx.memref_store(fx.Int32(0), s_hist, e)
        gpu.barrier()

        start = bid * span
        end = (start + span < n_pairs).select(start + span, n_pairs)
        # Each tile's ids are loaded one tile ahead.
        expert = _load_i32(topk_rsrc, start + tid, start + tid < end)
        for base in range(fx.Int32(0), span, fx.Int32(BLOCK_THREADS)):
            pair = start + base + tid
            ahead = pair + fx.Int32(BLOCK_THREADS)
            next_expert = _load_i32(topk_rsrc, ahead, ahead < end)
            routed = (pair < end) & (expert >= fx.Int32(0)) & (expert < num_experts)
            lanes = _same_expert_lanes(expert, routed, KEY_BITS)
            before = fx.Int32(fx.math.ctpop(lanes & lanes_below))
            count = fx.Int32(fx.math.ctpop(lanes))
            key = routed.select(expert, fx.Int32(0))
            # The last lane of each expert's group adds the group; the waves take
            # turns, so each sum is the same on every launch.
            last = routed & (before + fx.Int32(1) == count)
            for w in range_constexpr(NUM_WAVES):
                if last & (wave == fx.Int32(w)):
                    fx.memref_store(fx.memref_load(s_hist, key) + count, s_hist, key)
                gpu.barrier()
            expert = next_expert

        for e in range(tid, num_experts, BLOCK_THREADS):
            buffer_ops.buffer_store(
                fx.memref_load(s_hist, e), counts_rsrc, e * num_blocks + bid
            )

    @flyc.kernel(known_block_size=[PREFIX_THREADS, 1, 1])
    def moe_routing_prefix_kernel(
        Counts: fx.Tensor,
        FirstTokenOffset: fx.Tensor,
        num_experts: fx.Int32,
        num_blocks: fx.Int32,
        offset_is_i64: fx.Int32,
    ):
        # Expert-major, the workspace's exclusive scan is each (expert, block)'s first
        # row: the rows of lower experts, then this expert's rows in lower blocks.
        # The block scans it a chunk at a time, each thread four consecutive entries
        # (one 16-byte load, the next chunk's issued ahead), carrying the running
        # total from chunk to chunk. A block-0 entry is its expert's
        # first_token_offset. The workspace is padded to whole 16-byte groups.
        tid = fx.Int32(fx.thread_idx.x)
        counts_rsrc = buffer_ops.create_buffer_resource(Counts, max_size=True)
        offset_rsrc = buffer_ops.create_buffer_resource(FirstTokenOffset, max_size=True)
        storage = fx.SharedAllocator().allocate(block_scan.SharedStorage).peek()
        wide = offset_is_i64 != fx.Int32(0)

        entries = num_experts * num_blocks
        mine = tid * fx.Int32(PREFIX_ITEMS)

        def load_group(at):
            return buffer_ops.buffer_load(
                counts_rsrc,
                (at < entries).select(at, fx.Int32(_OOB)),
                vec_width=PREFIX_ITEMS,
                dtype=T.i32,
            )

        group = load_group(mine)
        carry = fx.Int32(0)
        for base in range(fx.Int32(0), entries, fx.Int32(PREFIX_CHUNK)):
            at = base + mine
            next_group = load_group(at + fx.Int32(PREFIX_CHUNK))
            counts = [
                (at + fx.Int32(i) < entries).select(
                    fx.Int32(fx.Vector(group)[i]), fx.Int32(0)
                )
                for i in range_constexpr(PREFIX_ITEMS)
            ]
            firsts, total = block_scan.exclusive_with_aggregate(
                fx.Vector.from_elements(counts, fx.Int32),
                fx.ReductionOp.ADD,
                storage=storage,
                init=carry,
            )
            buffer_ops.buffer_store(
                firsts, counts_rsrc, (at < entries).select(at, fx.Int32(_OOB))
            )
            expert = at // num_blocks
            block = at - expert * num_blocks
            for i in range_constexpr(PREFIX_ITEMS):
                opens = (at + fx.Int32(i) < entries) & (block == fx.Int32(0))
                buffer_ops.buffer_store(
                    firsts[i],
                    offset_rsrc,
                    opens.select(wide.select(expert * 2, expert), fx.Int32(_OOB)),
                )
                buffer_ops.buffer_store(
                    fx.Int32(0),
                    offset_rsrc,
                    (opens & wide).select(expert * 2 + 1, fx.Int32(_OOB)),
                )
                block = block + fx.Int32(1)
                wraps = block == num_blocks
                block = wraps.select(fx.Int32(0), block)
                expert = wraps.select(expert + fx.Int32(1), expert)
            carry = carry + total
            group = next_group
            gpu.barrier()

        # first_token_offset[E]: every routed row.
        top = tid == fx.Int32(0)
        buffer_ops.buffer_store(
            carry,
            offset_rsrc,
            top.select(wide.select(num_experts * 2, num_experts), fx.Int32(_OOB)),
        )
        buffer_ops.buffer_store(
            fx.Int32(0),
            offset_rsrc,
            (top & wide).select(num_experts * 2 + 1, fx.Int32(_OOB)),
        )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def moe_routing_scatter_kernel(
        TopkIds: fx.Tensor,
        Counts: fx.Tensor,
        TokenIndex: fx.Tensor,
        TokenKs: fx.Tensor,
        n_pairs: fx.Int32,
        num_experts: fx.Int32,
        span: fx.Int32,
        num_blocks: fx.Int32,
        top_k: fx.Int32,
    ):
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(WAVE_SIZE)
        lanes_below = (fx.Int64(1) << fx.Int64(tid % fx.Int32(WAVE_SIZE))) - fx.Int64(1)
        topk_rsrc = buffer_ops.create_buffer_resource(TopkIds, max_size=True)
        counts_rsrc = buffer_ops.create_buffer_resource(Counts, max_size=True)
        index_rsrc = buffer_ops.create_buffer_resource(TokenIndex, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(TokenKs, max_size=True)

        lds = fx.SharedAllocator().allocate(TileStorage).peek()
        s_cursor = lds.per_expert.view(fx.make_layout(e_max, 1))

        for e in range(tid, num_experts, BLOCK_THREADS):
            first_row = fx.Int32(
                buffer_ops.buffer_load(
                    counts_rsrc, e * num_blocks + bid, vec_width=1, dtype=T.i32
                )
            )
            fx.memref_store(first_row, s_cursor, e)
        gpu.barrier()

        start = bid * span
        end = (start + span < n_pairs).select(start + span, n_pairs)
        expert = _load_i32(topk_rsrc, start + tid, start + tid < end)
        for base in range(fx.Int32(0), span, fx.Int32(BLOCK_THREADS)):
            pair = start + base + tid
            ahead = pair + fx.Int32(BLOCK_THREADS)
            next_expert = _load_i32(topk_rsrc, ahead, ahead < end)
            routed = (pair < end) & (expert >= fx.Int32(0)) & (expert < num_experts)
            lanes = _same_expert_lanes(expert, routed, KEY_BITS)
            before = fx.Int32(fx.math.ctpop(lanes & lanes_below))
            count = fx.Int32(fx.math.ctpop(lanes))
            key = routed.select(expert, fx.Int32(0))
            # Wave by wave, each expert's group takes the expert's next `count` rows
            # in lane order, and its last lane advances the expert's cursor.
            row = fx.Int32(_OOB)
            for w in range_constexpr(NUM_WAVES):
                if routed & (wave == fx.Int32(w)):
                    cursor = fx.memref_load(s_cursor, key)
                    row = cursor + before
                    if before + fx.Int32(1) == count:
                        fx.memref_store(cursor + count, s_cursor, key)
                gpu.barrier()
            token = pair // top_k
            buffer_ops.buffer_store(token, index_rsrc, row)
            buffer_ops.buffer_store(pair - token * top_k, ks_rsrc, row)
            expert = next_expert

    @flyc.jit
    def launch_moe_routing(
        TopkIds: fx.Tensor,
        Counts: fx.Tensor,
        FirstTokenOffset: fx.Tensor,
        TokenIndex: fx.Tensor,
        TokenKs: fx.Tensor,
        n_pairs: fx.Int32,
        num_experts: fx.Int32,
        top_k: fx.Int32,
        span: fx.Int32,
        num_blocks: fx.Int32,
        offset_is_i64: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        moe_routing_histogram_kernel(
            TopkIds, Counts, n_pairs, num_experts, span, num_blocks
        ).launch(grid=(num_blocks, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
        moe_routing_prefix_kernel(
            Counts, FirstTokenOffset, num_experts, num_blocks, offset_is_i64
        ).launch(grid=(1, 1, 1), block=(PREFIX_THREADS, 1, 1), stream=stream)
        moe_routing_scatter_kernel(
            TopkIds,
            Counts,
            TokenIndex,
            TokenKs,
            n_pairs,
            num_experts,
            span,
            num_blocks,
            top_k,
        ).launch(grid=(num_blocks, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_moe_routing
