# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tile loaders: global memory -> LDS.

Two strategies are exposed, both with the same authoring surface:

* `CoalescedTileLoader` — the classic two-step pattern. Each lane issues
  a `<load_vec x half>` `buffer_load`/`global_load` from DRAM into a
  register, then a matching `smem_store_vN_f16` writes that register to
  LDS. The compiler hides DRAM latency by interleaving the load
  instructions with whatever follows. This is the "compv3"-grade
  pipeline and is what the dsl/01-08 GEMM/conv ladder uses today.

* `AsyncTileLoader` — the compv4-grade pipeline. Issues
  `raw_ptr_buffer_load_lds` so the DRAM→LDS hop happens in hardware
  without a register intermediate. The intrinsic writes
  *lane-contiguous* bytes: lane `i` of a wave deposits its payload at
  `lds_addr + i * size_bytes`. Each call moves `dwords ∈ {1, 3, 4}`
  dwords per lane (so 4, 12, or 16 bytes per lane). Completion is
  signalled via the VMEM counter, so consumers must drop an
  `s_waitcnt(vmcnt=0)` before reading the LDS.

Both loaders share the same authoring contract:

  - Input: a 2D tile `(tile_rows, tile_cols)` of fp16 elements, a
    block-level thread count, an LDS allocation, and a *descriptor*
    that maps `(row, col) -> (linear_element_offset, valid_predicate)`.
  - Output: the LDS is filled with the tile in the layout
    `LDS[row, col] = global[block_row_off + row, block_col_off + col]`
    for `row ∈ [0, tile_rows), col ∈ [0, tile_cols)`. The MFMA helpers
    consume that layout directly.

The descriptor argument is what makes this layer convolution-aware:
the loader does not care whether `(row, col) -> NHWC linear offset`
comes from a naive `row * K + col` formula, a coordinate-transform DAG,
or a hand-rolled implicit-GEMM mapping. The loader only sees a
callback `descriptor(b, row, col) -> (off_elements, valid_or_None)`.

`AsyncTileLoader` enforces the lane-contiguous LDS contract by
*requiring* a non-swizzled LDS destination — exactly the runbook §6.3
caveat. If you need a swizzled layout you must express the swizzle in
the *address arithmetic of the consumer* (i.e. in the MFMA's LDS
read), not by handing the intrinsic a swizzled destination pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from ..core.ir import I64, IRBuilder, Value


# A descriptor callback maps (row, col) in the tile-local coordinate
# system to (element_offset_in_global_array, valid_predicate). The
# returned offset is in *fp16 elements*; the loader scales by 2 (sizeof
# half) when feeding it to a buffer_load voffset. `valid` is an i1
# Value (or `None` to mean "always in-bounds").
DescriptorFn = Callable[[IRBuilder, Value, Value], Tuple[Value, Optional[Value]]]


@dataclass(frozen=True)
class CoalescedTileLoader:
    """Sync coalesced global -> LDS load plan for a 2D tile.

    The kernel calls `load(b, ...)` per K-iteration. The loader picks a
    `load_vec` (halves per thread per chunk) that distributes the tile
    evenly across `block_size` threads with the widest possible
    natural alignment, then emits the per-thread for-loop that issues
    `vecs_per_thread = (tile_rows * tile_cols) / load_vec / block_size`
    chunks per thread.

    The `descriptor` callback is the only place addressing changes
    across kernels. The same loader handles plain GEMM (`row * K +
    col`) and implicit-GEMM conv (`A_desc.offset(row, col)`).
    """

    tile_rows: int
    tile_cols: int
    block_size: int
    load_vec: int  # halves per thread per chunk
    use_buffer_rsrc: bool = (
        True  # use buffer_load_vN (bounds-checked); False uses raw ptr
    )
    oob_sentinel: int = (1 << 31) - 1  # voffset used when valid=False (clamped to 0)

    @classmethod
    def choose_vec(
        cls,
        *,
        tile_rows: int,
        tile_cols: int,
        block_size: int,
        max_vec: int = 8,
    ) -> int:
        """Pick the widest `load_vec` that distributes evenly.

        Conditions, in order:
          1. `tile_cols % vec == 0` (no partial column at the end).
          2. `(tile_rows * tile_cols) % (vec * block_size) == 0` (every
             thread does the same number of chunks).
          3. `(tile_rows * tile_cols) / vec >= block_size` (at least
             one chunk per thread per phase — otherwise some threads
             would be idle).
        """
        v = max_vec
        while v >= 1:
            if (
                tile_cols % v == 0
                and (tile_rows * tile_cols) // v >= block_size
                and ((tile_rows * tile_cols) // v) % block_size == 0
            ):
                return v
            v //= 2
        raise ValueError(
            f"no usable load_vec for tile {tile_rows}x{tile_cols} "
            f"with block_size {block_size}"
        )

    @classmethod
    def from_tile(
        cls,
        *,
        tile_rows: int,
        tile_cols: int,
        block_size: int,
        max_vec: int = 8,
        use_buffer_rsrc: bool = True,
    ) -> "CoalescedTileLoader":
        vec = cls.choose_vec(
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            block_size=block_size,
            max_vec=max_vec,
        )
        return cls(
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            block_size=block_size,
            load_vec=vec,
            use_buffer_rsrc=use_buffer_rsrc,
        )

    @property
    def vecs_per_thread(self) -> int:
        total_vecs = (self.tile_rows * self.tile_cols) // self.load_vec
        if total_vecs % self.block_size:
            raise ValueError(
                f"tile {self.tile_rows}x{self.tile_cols} / {self.load_vec} = "
                f"{total_vecs} not divisible by block_size {self.block_size}"
            )
        return total_vecs // self.block_size

    @property
    def cols_per_vec(self) -> int:
        return self.tile_cols // self.load_vec

    def load(
        self,
        b: IRBuilder,
        *,
        tid: Value,
        smem_dst: Value,
        descriptor: DescriptorFn,
        rsrc: Optional[Value] = None,
        ptr: Optional[Value] = None,
    ) -> None:
        """Emit the per-thread load loop.

        `descriptor(b, row, col) -> (offset_in_elements, valid)` defines
        the tile -> global mapping. `valid` may be `None` for tiles that
        are fully in-bounds.

        If `use_buffer_rsrc`, `rsrc` must be provided (a buffer
        resource from `b.buffer_rsrc(ptr, num_bytes)`). Otherwise
        `ptr` must be provided (a `ptr<f16, global>`).
        """
        if self.use_buffer_rsrc and rsrc is None:
            raise ValueError("CoalescedTileLoader: use_buffer_rsrc=True requires rsrc")
        if not self.use_buffer_rsrc and ptr is None:
            raise ValueError("CoalescedTileLoader: use_buffer_rsrc=False requires ptr")

        c_threads = b.const_i32(self.block_size)
        c_load_vec = b.const_i32(self.load_vec)
        c_cols_per_vec = b.const_i32(self.cols_per_vec)
        c_half_bytes = b.const_i32(2)
        c0 = b.const_i32(0)
        c_oob = b.const_i32(self.oob_sentinel)

        for e in range(self.vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_cols_per_vec)
            col_v = b.mod(vec_idx, c_cols_per_vec)
            col = b.mul(col_v, c_load_vec) if self.load_vec > 1 else col_v

            off_elems, valid = descriptor(b, row, col)

            if self.use_buffer_rsrc:
                off_bytes = b.mul(off_elems, c_half_bytes)
                if valid is not None:
                    safe = b.select(valid, off_bytes, c_oob)
                else:
                    safe = off_bytes
                if self.load_vec == 1:
                    v = b.buffer_load_f16(rsrc, safe, c0)
                    b.smem_store_f16(smem_dst, [row, col], v)
                else:
                    dwords = self.load_vec // 2
                    v = b.buffer_load_vN_f16(rsrc, safe, c0, dwords)
                    b.smem_store_vN_f16(smem_dst, [row, col], v, self.load_vec)
            else:
                if self.load_vec == 1:
                    v = b.global_load_f16(ptr, off_elems)
                    b.smem_store_f16(smem_dst, [row, col], v)
                else:
                    v = b.global_load_vN_f16(ptr, off_elems, self.load_vec)
                    b.smem_store_vN_f16(smem_dst, [row, col], v, self.load_vec)


# ---------------------------------------------------------------------
# Async DRAM -> LDS loader (runbook §6.3, the compv4 lever)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AsyncTileLoader:
    """Direct DRAM->LDS load via `raw_ptr_buffer_load_lds`.

    Per-lane semantics:
      The intrinsic writes ``dwords * 4`` bytes per lane, *lane
      contiguously* — lane `i` in a wave deposits its payload at
      ``lds_base + i * dwords * 4``. There is no register intermediate.

    Authoring contract (mirror of `CoalescedTileLoader`):
      `LDS[row, col] = global[tile_row_off + row, tile_col_off + col]`
      with `LDS[row, col]` stored as the canonical row-major byte
      layout. This loader picks `load_vec` such that the byte layout
      lane lane writes naturally land at the right place.

    Restrictions vs `CoalescedTileLoader`:
      * `dwords` must be 1, 3, or 4 (the intrinsic does not accept 2 on
        this LLVM target). The corresponding fp16 chunk widths are
        2, 6, and 8 halves per lane.
      * The LDS destination is computed per-wave-and-pass from
        `smem_addr_of(smem) + wave_id * wave_stride + pass_idx * pass_stride`
        and is *uniform within a wave*. The intrinsic does not accept
        an arbitrary per-lane LDS pointer; per-lane swizzles have to
        live in the *consumer*'s read address arithmetic (the MFMA
        helper), not here.
      * Consumers must call `b.s_waitcnt(vmcnt=0)` before reading the
        LDS (the intrinsic uses the VMEM counter, not the LGKM
        counter).

    The `descriptor` callback is the same as
    `CoalescedTileLoader.load`, with one extra constraint: the
    per-thread chunks must map to *contiguous LDS bytes per lane*. The
    default chunk layout already satisfies this: chunk_idx of lane
    `(w, l)` in pass `p` is `w * wave + l + p * threads`, which lands
    at LDS bytes `chunk_idx * dwords * 4` — exactly the lane-contiguous
    pattern the intrinsic writes.

    Usage:

        loader = AsyncTileLoader.from_tile(
            tile_rows=block_m, tile_cols=block_k,
            block_size=block_size, wave_size=64,
        )
        # bind once: computes the LDS base in SSA
        slot = loader.bind(b, smem_dst=A_smem, wave_id=wave_id)
        slot.issue(b, tid=tid, rsrc=a_rsrc, descriptor=a_desc_fn)
        # ... emit MFMAs that need NOT consume A yet ...
        b.s_waitcnt(vmcnt=0)
        b.sync()
        # now safe to read A_smem
    """

    tile_rows: int
    tile_cols: int
    block_size: int
    wave_size: int
    dwords: int  # 1, 3, or 4
    chunks_total: int  # tile_rows * tile_cols / (dwords * 2)
    chunks_per_pass: int  # = block_size
    passes: int  # ceil(chunks_total / block_size)

    @classmethod
    def choose_dwords(
        cls,
        *,
        tile_rows: int,
        tile_cols: int,
        block_size: int,
        max_dwords: int = 4,
    ) -> int:
        """Pick the widest `dwords` value that divides the tile evenly."""
        if max_dwords > 4:
            max_dwords = 4
        for d in (4, 3, 1):
            if d > max_dwords:
                continue
            halves = d * 2
            if tile_cols % halves != 0:
                continue
            chunks = (tile_rows * tile_cols) // halves
            if chunks < block_size:
                continue
            return d
        raise ValueError(
            f"no usable dwords value for tile {tile_rows}x{tile_cols} "
            f"with block_size {block_size}"
        )

    @classmethod
    def from_tile(
        cls,
        *,
        tile_rows: int,
        tile_cols: int,
        block_size: int,
        wave_size: int = 64,
        max_dwords: int = 4,
    ) -> "AsyncTileLoader":
        d = cls.choose_dwords(
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            block_size=block_size,
            max_dwords=max_dwords,
        )
        halves = d * 2
        chunks = (tile_rows * tile_cols) // halves
        passes = (chunks + block_size - 1) // block_size
        return cls(
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            block_size=block_size,
            wave_size=wave_size,
            dwords=d,
            chunks_total=chunks,
            chunks_per_pass=block_size,
            passes=passes,
        )

    @property
    def halves_per_chunk(self) -> int:
        return self.dwords * 2

    @property
    def bytes_per_chunk(self) -> int:
        return self.dwords * 4

    @property
    def cols_per_chunk(self) -> int:
        return self.halves_per_chunk

    @property
    def wave_bytes(self) -> int:
        """Bytes one wave writes per pass."""
        return self.wave_size * self.bytes_per_chunk

    @property
    def pass_bytes(self) -> int:
        """Bytes the whole block writes per pass."""
        return self.block_size * self.bytes_per_chunk

    def bind(
        self,
        b: IRBuilder,
        *,
        smem_dst: Value,
        wave_id: Value,
    ) -> "AsyncTileLoaderSlot":
        """Materialise the SSA values needed by `issue`."""
        lds_base = b.smem_addr_of(smem_dst)  # i64
        wave_byte_off_i32 = b.mul(wave_id, b.const_i32(self.wave_bytes))
        wave_byte_off_i32 = b.readfirstlane(wave_byte_off_i32)
        wave_byte_off_i64 = b.zext(wave_byte_off_i32, I64)
        per_wave_lds = b.smem_ptr_add(lds_base, wave_byte_off_i64)
        return AsyncTileLoaderSlot(
            loader=self,
            smem_dst=smem_dst,
            per_wave_lds_base=per_wave_lds,
        )


@dataclass(frozen=True)
class AsyncTileLoaderSlot:
    """Bound `AsyncTileLoader`: ready to `issue` for a specific K-iter."""

    loader: AsyncTileLoader
    smem_dst: Value
    per_wave_lds_base: Value  # i64; lane 0 of the wave writes here

    def issue(
        self,
        b: IRBuilder,
        *,
        tid: Value,
        rsrc: Value,
        descriptor: DescriptorFn,
        oob_sentinel: int = (1 << 31) - 1,
    ) -> None:
        """Fire all `passes * threads` async loads for this iteration.

        After `issue`, the LDS contents are *in flight*. Consumers must
        place an `s_waitcnt(vmcnt=0)` before reading the LDS.
        """
        L = self.loader
        c_half_bytes = b.const_i32(2)
        c_oob = b.const_i32(oob_sentinel)
        c0 = b.const_i32(0)
        c_cols_per_chunk = b.const_i32(L.cols_per_chunk)

        for p in range(L.passes):
            # Per-pass LDS base = per_wave_lds_base + p * pass_bytes.
            # Each lane in the wave writes at its own lane*bytes_per_chunk
            # offset *from that base*, which the intrinsic supplies.
            pass_byte_off = p * L.pass_bytes
            pass_base = (
                b.smem_ptr_add(
                    self.per_wave_lds_base,
                    b.zext(b.const_i32(pass_byte_off), I64),
                )
                if p > 0
                else self.per_wave_lds_base
            )

            # chunk_idx = tid + p * block_size
            chunk_idx = b.add(tid, b.const_i32(p * L.block_size))
            row = b.div(chunk_idx, c_cols_per_chunk)
            col_v = b.mod(chunk_idx, c_cols_per_chunk)
            col = b.mul(col_v, b.const_i32(L.halves_per_chunk))

            off_elems, valid = descriptor(b, row, col)
            off_bytes = b.mul(off_elems, c_half_bytes)
            # Threads whose chunk_idx >= chunks_total still issue an
            # async load, but the descriptor *must* return valid=0 (or
            # an out-of-range offset) for them so the intrinsic writes
            # zeros to the corresponding LDS slot. The over-allocation
            # in the LDS allocation handles the slack region cleanly.
            in_pass = b.cmp_lt(chunk_idx, b.const_i32(L.chunks_total))
            valid_final = b.land(valid, in_pass) if valid is not None else in_pass
            safe = b.select(valid_final, off_bytes, c_oob)
            b.async_buffer_load_lds_addr(
                rsrc,
                pass_base,
                safe,
                c0,
                L.dwords,
            )

    def required_lds_bytes(self) -> int:
        """How many LDS bytes this loader's allocation needs to reserve.

        Each pass writes `block_size * dwords * 4` bytes, so
        `passes * block_size * dwords * 4` covers every chunk including
        the slack region for threads whose chunk_idx is OOB.
        """
        return self.loader.passes * self.loader.pass_bytes


def lane_contiguous_descriptor(
    *,
    cols_per_chunk: int,
    chunks_per_row: int,
    base_off_fn: Callable[[IRBuilder, Value, Value], Tuple[Value, Optional[Value]]],
) -> DescriptorFn:
    """Adapt a `(tile_row, tile_col_halves) -> (off, valid)` callback to
    the async loader's per-lane chunk-coordinate input.

    The async loader passes `(row, col_halves)` where `col_halves`
    increments by `halves_per_chunk` between adjacent lanes. This
    helper is mostly documentation: most kernels can pass their
    descriptor callback directly because it already takes
    `(row, col_halves)`.
    """
    return base_off_fn
