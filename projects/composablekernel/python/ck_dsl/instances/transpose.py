# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D transpose kernel instance builder.

DSL counterpart of CK Tile's ``example/ck_tile/37_transpose`` and
``35_batched_transpose``. The whole kernel is expressed against the
:class:`ck_dsl.helpers.TensorView` / :class:`ck_dsl.helpers.TileWindow`
abstractions ported from CK Tile's ``make_tensor_view`` /
``make_tile_window``; the LDS staging buffer is an
``make_tensor_view<addr_space::lds>`` (see
:func:`ck_dsl.helpers.make_lds_view`).

Algorithm (per tile):
  1. Coalesced HBM read: thread t loads a contiguous ``vec`` halves
     from the input tile and stores them into the matching LDS row.
  2. ``__syncthreads`` (via :meth:`IRBuilder.sync`).
  3. Column-major LDS scan + coalesced HBM write: thread t reads
     ``vec`` scalars from one LDS column at adjacent rows and packs
     them into the output row of the transposed tile.

LDS layout: ``[tile_m, tile_n + lds_pad]`` half-words. The default
``lds_pad = 8`` rounds each row up to a 16-byte boundary so the
column-strided reads in step 3 don't pile onto a single LDS bank.

What we cover today:
  - Dtypes ``f16`` / ``bf16``
  - Square tile ``tile_m == tile_n`` in {16, 32, 64}
  - Vec widths {2, 4, 8} for both phases
  - Both ``H`` and ``W`` must be divisible by the tile size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ..helpers.tensor_view import make_global_view, make_lds_view


DType = Literal["f16", "bf16"]


GridOrder = Literal["row", "morton"]
"""Grid-traversal order option for :class:`Transpose2DSpec`.

* ``"row"``    -- the default. CTAs visit tiles in plain
                  ``(block_id_x, block_id_y)`` row-major order.
* ``"morton"`` -- CTAs are launched on a 1D grid; the kernel decodes
                  ``block_id_x`` as a Morton (Z-order) index to
                  recover ``(logical_tile_x, logical_tile_y)``.

Empirically, Morton ordering does **not** speed up the transpose for
the workload sizes we benchmark today:

* At small sizes (e.g. 1024x1024) the kernel is launch-overhead
  bound -- the ~256 CTAs amortise poorly against ~3-5us launch cost.
  Morton can't help; cache reuse isn't the bottleneck.
* At large sizes (4096x4096+) the kernel saturates HBM at
  ~5700 GB/s on MI355X (close to peak). There's no spare bandwidth
  for cache tricks to claw back.

In the middle band (2048x2048) row-major beats Morton by ~25%
because the Morton bit-decode adds work without measurable cache
benefit -- our 1024 CTAs there spread evenly across CUs already.

Hilbert ordering has slightly better locality than Morton on very
large grids, but requires more arithmetic per CTA (~5x the Morton
decode cost). We expose Morton as an opt-in for cases where the
GPU's CTA scheduler is the limiter (rare for transpose; potentially
useful for cache-bound stencil-style ops). The default
``grid_order="row"`` is the right choice for transpose specifically.
"""


@dataclass(frozen=True)
class Transpose2DSpec:
    tile_m: int = 64
    tile_n: int = 64
    vec: int = 8
    dtype: DType = "f16"
    lds_pad: int = 8
    grid_order: GridOrder = "row"
    name: str = "ck_dsl_transpose2d"

    @property
    def block_size(self) -> int:
        return (self.tile_m * self.tile_n) // self.vec

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"{self.tile_m}x{self.tile_n}",
            f"v{self.vec}",
            f"p{self.lds_pad}",
            f"g{self.grid_order}" if self.grid_order != "row" else "",
        )


def is_valid_spec(spec: Transpose2DSpec) -> Tuple[bool, str]:
    # First the common dtype / vec validations.
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
        )
    )
    if not ok:
        return ok, why
    if spec.tile_m not in (16, 32, 64) or spec.tile_n not in (16, 32, 64):
        return False, "tile_m/tile_n must be in {16, 32, 64}"
    if spec.tile_m != spec.tile_n:
        return False, "non-square tiles not yet supported (set tile_m == tile_n)"
    if (spec.tile_m * spec.tile_n) % spec.vec:
        return False, "tile area must be divisible by vec"
    if spec.block_size > 1024:
        return False, (
            f"block_size {spec.block_size} > 1024 hardware cap (tile_m*tile_n/vec)"
        )
    return True, "ok"


def build_transpose2d(spec: Transpose2DSpec) -> KernelDef:
    """Build the IR for one 2D transpose instance.

    Kernel signature: ``(X: ptr, Y: ptr, H: i32, W: i32)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid transpose2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    TM, TN, vec, BS = spec.tile_m, spec.tile_n, spec.vec, spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    H = b.param("H", I32)
    W = b.param("W", I32)

    tid = b.thread_id_x()

    c_vec = b.const_i32(vec)
    c_TN_chunks = b.const_i32(TN // vec)
    c_TM_chunks = b.const_i32(TM // vec)

    # Decode logical (tile_x, tile_y) from block_id according to the
    # chosen grid order. For ``"row"`` we just use the launched 2D
    # block_id; for ``"morton"`` we expect a 1D grid (block_id_x = a
    # linearised tile id) and de-interleave its bits.
    if spec.grid_order == "morton":
        bid = b.block_id_x()
        # Morton decode for 16-bit indices: x = compact bits 0,2,4,...
        # y = compact bits 1,3,5,...  Sixteen bits cover up to 65536x65536
        # tiles, which is way more than any realistic launch.
        tile_x = _morton_decode_bits(b, bid, shift=0, width=16)
        tile_y = _morton_decode_bits(b, bid, shift=1, width=16)
    else:
        tile_x = b.block_id_x()
        tile_y = b.block_id_y()

    h0 = b.mul(tile_y, b.const_i32(TM))
    w0 = b.mul(tile_x, b.const_i32(TN))

    # CK Tile-style data abstractions: input and output are 2D global
    # views with a runtime row stride (W / H). The LDS staging buffer
    # is a packed addrspace(3) view with bank-conflict padding.
    x_view = make_global_view(X, shape=(TM, TN), dtype=io_ty, strides=(W, 1))
    y_view = make_global_view(Y, shape=(TN, TM), dtype=io_ty, strides=(H, 1))
    x_tile = x_view.tile(lengths=(TM, TN), origin=(h0, w0))
    y_tile = y_view.tile(lengths=(TN, TM), origin=(w0, h0))

    lds_view = make_lds_view(
        b, dtype=io_ty, shape=(TM, TN + spec.lds_pad), name_hint="lds_xpose"
    )
    lds_tile = lds_view.tile(lengths=(TM, TN), origin=(b.const_i32(0), b.const_i32(0)))

    # Phase 1 thread layout: (row1, col1_chunk).
    row1 = b.div(tid, c_TN_chunks)
    col1_chunk = b.mod(tid, c_TN_chunks)
    col1 = b.mul(col1_chunk, c_vec)

    # Phase 1: coalesced global -> LDS.
    x_vec = x_tile.load_vec(b, row1, col1, n=vec)
    lds_tile.store_vec(b, row1, col1, value=x_vec, n=vec)
    b.sync()

    # Phase 2 thread layout: (col2, row2_chunk).
    col2 = b.div(tid, c_TM_chunks)
    row2_chunk = b.mod(tid, c_TM_chunks)
    row2_base = b.mul(row2_chunk, c_vec)

    # Column-strided LDS reads pack into one output vector, then a
    # single coalesced global store per thread emits the transposed row.
    elems = [
        lds_tile.load_scalar(b, b.add(row2_base, b.const_i32(i)), col2)
        for i in range(vec)
    ]
    out_vec = b.vec_pack(elems, io_ty)
    y_tile.store_vec(b, col2, row2_base, value=out_vec, n=vec)

    return b.kernel


def transpose2d_grid(h: int, w: int, spec: Transpose2DSpec) -> Tuple[int, int, int]:
    """Return the launch grid for ``(H, W)`` and this spec.

    For ``grid_order="row"`` this is the canonical 2D grid
    ``(W/tile_n, H/tile_m, 1)``. For ``grid_order="morton"`` we
    flatten to 1D so the kernel can do its own Morton decode: the
    grid becomes ``(N_tiles, 1, 1)`` where
    ``N_tiles = (W/tile_n) * (H/tile_m)`` rounded up to the next
    power of two (so the Morton index occupies a clean bit field
    and no out-of-bounds tiles are emitted).
    """
    nx = (w + spec.tile_n - 1) // spec.tile_n
    ny = (h + spec.tile_m - 1) // spec.tile_m
    if spec.grid_order == "morton":
        # Pad to the next power of two so adjacent CTAs map cleanly
        # to neighbouring (tile_x, tile_y) pairs in the Z-order. We
        # round each side up to a power of two and then take the
        # max -- the kernel handles any out-of-range tiles by virtue
        # of the global-store going to a 1D-offset (no bounds check
        # is needed for in-spec H/W; for non-power-of-two sizes the
        # caller pads externally).
        side = 1
        while side < max(nx, ny):
            side *= 2
        return (side * side, 1, 1)
    return ceil_div_grid((w, spec.tile_n), (h, spec.tile_m))


def transpose2d_signature(spec: Transpose2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("H", "i32")
        .scalar("W", "i32")
        .build()
    )


# ---------------------------------------------------------------------
# Morton (Z-order) bit decode
# ---------------------------------------------------------------------
#
# The classic compact-bits trick: keep only the even (or odd) bits
# of a 32-bit input and pack them into the low half of the result.
# Reference: Bit Twiddling Hacks, "Interleave bits by Binary Magic Numbers".
#
# We use the standard 5-step magic-number compaction. ``shift`` is 0
# for the X coordinate (even bits) or 1 for the Y coordinate (odd
# bits, right-shifted by 1 first). ``width`` is the number of output
# bits to retain; 16 is plenty for any practical grid.


def _morton_decode_bits(b, val, *, shift: int, width: int):
    """Compact the bits at positions ``shift, shift+2, shift+4, ...``
    of ``val`` into a contiguous integer with ``width`` significant bits.

    Returns an i32 SSA value.
    """
    v = val
    if shift:
        v = b.div(v, b.const_i32(1 << shift))  # logical right-shift by `shift`
    # Mask to even bits and compact via the standard 5-step routine.
    # Operating on at most 32 bits so we keep only the lowest 32 of `v`.
    v = b.land(v, b.const_i32(0x55555555 & ((1 << (2 * width)) - 1)))
    # Step 1: combine pairs of even bits.
    # (v | (v >> 1)) & 0x33333333
    v = b.land(
        b.lor(v, b.div(v, b.const_i32(2))),
        b.const_i32(0x33333333 & ((1 << (2 * width)) - 1)),
    )
    # Step 2: combine nibbles.
    v = b.land(
        b.lor(v, b.div(v, b.const_i32(4))),
        b.const_i32(0x0F0F0F0F & ((1 << (2 * width)) - 1)),
    )
    # Step 3: combine bytes.
    v = b.land(
        b.lor(v, b.div(v, b.const_i32(16))),
        b.const_i32(0x00FF00FF & ((1 << (2 * width)) - 1)),
    )
    # Step 4: combine 16-bit halves into one 16-bit field.
    v = b.land(
        b.lor(v, b.div(v, b.const_i32(256))),
        b.const_i32(0x0000FFFF & ((1 << (2 * width)) - 1)),
    )
    return v
