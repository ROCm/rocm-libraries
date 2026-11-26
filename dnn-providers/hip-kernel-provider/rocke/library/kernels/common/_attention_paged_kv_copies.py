# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Addressing a tile's bf16/fp16 K/V copies by page, for the tiled 2D kernels.

A copy moves ``halves_per_call`` halves of a K/V tile (8 per lane) into LDS. When a wave's part
of each copy lies in one page, that page is a function of the copy and the wave only, so it is
read once per tile with a wave-uniform block-table index, before the copies: a read per copy made
the compiler wait on it right before its copy, a vmcnt drain of the copies in flight. A copy is
then addressed as its page's base, its offset in the page and the lane's own part. The page base
goes to ``soffset`` or into ``voffset`` (the kernel picks by measurement), or, for a cache that
needs 64-bit offsets, into one buffer resource per page shared by its copies.

The C twin is ``instance_attention_paged_kv_copies_internal.h``; both emit the same ops in the
same order.
"""

from __future__ import annotations

from rocke.core.ir import I64, IRBuilder, Value


class PagedKVCopies:
    """The page addressing of one kernel's K/V copies.

    Geometry (``block_size`` tokens of ``head_size`` halves per page, ``blocks_per_tile`` pages per
    tile) and the kernel's values are fixed at construction, which emits nothing. Call
    ``emit_offsets`` where the kernel builds its per-kernel K/V offsets, then per loader
    ``tile_pages`` once and ``copy`` per copy.
    """

    def __init__(
        self,
        b: IRBuilder,
        *,
        block_size: int,
        head_size: int,
        halves_per_call: int,
        num_warps: int,
        blocks_per_tile: int,
        calls_per_tile: int,
        kv_bytes: int,
        head_stride_b: int,
        lane_half_base: Value,
        wave_id: Value | None,
        kv_head: Value,
        token_stride_b: Value,
        page_stride_b: Value,
        page_bytes: Value,
        zero_soff: Value,
        i64: bool,
        page_base_in_soffset: bool,
    ) -> None:
        self.b = b
        self.block_size = block_size
        self.head_size = head_size
        self.halves_per_call = halves_per_call
        self.blocks_per_tile = blocks_per_tile
        self.calls_per_tile = calls_per_tile
        self.kv_bytes = kv_bytes
        self.head_stride_b = head_stride_b
        self.lane_half_base = lane_half_base
        self.wave_id = wave_id
        self.kv_head = kv_head
        self.token_stride_b = token_stride_b
        self.page_stride_b = page_stride_b
        self.page_bytes = page_bytes
        self.zero_soff = zero_soff
        self.i64 = i64
        self.page_base_in_soffset = page_base_in_soffset
        page_halves = block_size * head_size
        self.wave_halves = halves_per_call // num_warps
        # A wave's part of every copy lies in one page: the condition for page copies.
        self.wave_part_in_page = page_halves % self.wave_halves == 0
        # A copy of whole rows within a page, or of whole pages, puts each lane at the same row
        # of its page in every copy: its offset is built once. A copy that ends mid-row
        # (head_size 192) splits each lane's position per copy.
        self.copy_tokens = halves_per_call // head_size
        self.whole_row_copies = halves_per_call % head_size == 0 and (
            block_size % self.copy_tokens == 0 or self.copy_tokens % block_size == 0
        )
        # Every wave's part of a copy lies in the same page of the tile (else a wave's page
        # depends on the wave).
        self.uniform_blocks = blocks_per_tile == 1 or (
            halves_per_call <= page_halves and page_halves % halves_per_call == 0
        )
        self.lane_within_b: Value | None = None
        self.copy_tok_b: dict[int, Value] = {}
        self.page_stride_i64: Value | None = None

    def emit_offsets(self) -> None:
        """The lane's offset in its page and the copies' token offsets (whole-row copies), and
        the 64-bit page stride. The token stride is a kernarg, so they are built once rather
        than multiplying the token index in every copy of every tile."""
        b = self.b
        if self.whole_row_copies:
            lane_tok = b.div(self.lane_half_base, b.const_i32(self.head_size))
            if self.block_size % self.copy_tokens:
                # A copy covers whole pages: the lane's row within its page.
                lane_tok = b.mod(lane_tok, b.const_i32(self.block_size))
            lane_dim = b.mod(self.lane_half_base, b.const_i32(self.head_size))
            self.lane_within_b = b.add(
                b.add(
                    b.mul(lane_tok, self.token_stride_b),
                    b.mul(self.kv_head, b.const_i32(self.head_stride_b)),
                ),
                b.mul(lane_dim, b.const_i32(self.kv_bytes)),
            )
            for call in range(self.calls_per_tile):
                tok0 = (call * self.copy_tokens) % self.block_size
                if tok0 and tok0 not in self.copy_tok_b:
                    self.copy_tok_b[tok0] = b.mul(
                        self.token_stride_b, b.const_i32(tok0)
                    )
        if self.i64:
            self.page_stride_i64 = b.zext(self.page_stride_b, I64)

    def copy_block(self, call: int):
        """The tile's block that the calling wave's part of copy ``call`` lies in: an int when
        every wave's part lies in the same block, else a wave-uniform value."""
        b = self.b
        page_halves = self.block_size * self.head_size
        if self.uniform_blocks:
            return (call * self.halves_per_call) // page_halves
        assert self.wave_id is not None, "a block from the wave needs several warps"
        wave_first = b.add(
            b.const_i32(call * self.halves_per_call),
            b.to_sgpr_u32(b.mul(self.wave_id, b.const_i32(self.wave_halves))),
        )
        return b.div(wave_first, b.const_i32(page_halves))

    def tile_pages(self, page_id, carried=None):
        """Each copy's block and the page ids of the tile's blocks: ``carried`` (by block, read
        by the loop ahead), or read once here by ``page_id(blk)`` before the copies (a block
        from the wave keyed by its call)."""
        blocks = [self.copy_block(c) for c in range(self.calls_per_tile)]
        if carried is not None:
            return blocks, dict(enumerate(carried))
        pages = {}
        for call, blk in enumerate(blocks):
            key = _key(blk, call)
            if key not in pages:
                pages[key] = page_id(blk)
        return blocks, pages

    def copy(self, call: int, blocks, pages, src: Value, src_rsrc: Value):
        """(pointer, resource, voffset, soffset) of copy ``call``."""
        b = self.b
        blk = blocks[call]
        key = _key(blk, call)
        page = pages[key]
        if self.whole_row_copies:
            voff = self.lane_within_b
            assert voff is not None, "emit_offsets() first"
            tok0 = (call * self.copy_tokens) % self.block_size
            if tok0:
                voff = b.add(voff, self.copy_tok_b[tok0])
        else:
            page_halves = self.block_size * self.head_size
            start = (
                b.const_i32(blk * page_halves)
                if isinstance(blk, int)
                else b.mul(blk, b.const_i32(page_halves))
            )
            half = b.sub(
                b.add(b.const_i32(call * self.halves_per_call), self.lane_half_base),
                start,
            )
            tok = b.div(half, b.const_i32(self.head_size))
            dim = b.mod(half, b.const_i32(self.head_size))
            voff = b.add(
                b.add(
                    b.mul(tok, self.token_stride_b),
                    b.mul(self.kv_head, b.const_i32(self.head_stride_b)),
                ),
                b.mul(dim, b.const_i32(self.kv_bytes)),
            )
        if self.i64:
            # One pointer and resource per page, shared by its copies.
            if ("rsrc", key) not in pages:
                assert self.page_stride_i64 is not None, "emit_offsets() first"
                ptr = b.global_ptr_add(
                    src, b.mul(b.zext(page, I64), self.page_stride_i64)
                )
                pages[("rsrc", key)] = ptr, b.buffer_rsrc(ptr, self.page_bytes)
            ptr, rsrc = pages[("rsrc", key)]
            return ptr, rsrc, voff, self.zero_soff
        page_b = b.mul(page, self.page_stride_b)
        if self.page_base_in_soffset:
            return src, src_rsrc, voff, page_b
        return src, src_rsrc, b.add(voff, page_b), self.zero_soff


def _key(blk, call: int):
    """A page's key in a tile's page table: its block, or the copy for a block from the wave."""
    return blk if isinstance(blk, int) else ("call", call)


__all__ = ["PagedKVCopies"]
