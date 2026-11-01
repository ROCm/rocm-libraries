# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""gfx11 / gfx12 helpers that `fmha_common_gfx1201` (frozen) spells with `kernels_common._if_then`, which the vendored `kernels.common`
(FlyDSL 0.3.4) does not define. Same names and semantics as the copies in `fmha_bwd_dq_gfx11.py`: the guarded cooperative-load staging, the
padded-head store, and the context manager under both. Imports no module of this repository.
"""

import contextlib

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import Vector as Vec


@contextlib.contextmanager
def _if_then(if_op):
    """Body of an `scf.if` without results: the insertion point of its then block, terminated by `scf.yield`."""
    with ir.InsertionPoint(if_op.then_block):
        yield
        _scf.YieldOp([])


def _guarded_batches(aperture, base_row, block_rows, body):
    """`fmha._over_batches` over `_if_then`: `body(batch, row)` for each cooperative-load batch, under the row guard
    (`aperture.needs_guard`: the ceil() batches can overshoot the tile)."""
    for batch in range_constexpr(aperture.num_batches):
        row = aperture.batch_row(base_row, batch)
        if aperture.needs_guard:
            with _if_then(_scf.IfOp(fx.as_ir_value(row < block_rows))):
                body(batch, row)
        else:
            body(batch, row)


def _stage(aperture, lds_ptr, read, base_row, col, block_rows):
    """`fmha.stage` over `_guarded_batches`: VRAM -> registers -> LDS for this thread's share of one tile."""

    def body(_batch, row):
        aperture.to_lds(
            lds_ptr, aperture.read_vec(read, row, col), row, col, aperture.vec_width
        )

    _guarded_batches(aperture, base_row, block_rows, body)


def _write_v8(aperture, write, row, col, val):
    """`fmha.write_v8` over `_if_then`: one 8-wide chunk, skipped when `col` is past the runtime extent (padded heads)."""
    if not aperture.cols.active:
        write(row, col, val)
        return
    with _if_then(_scf.IfOp(fx.as_ir_value(aperture.cols.valid(col)))):
        write(row, col, val)


def bf16_rne_pack_v8(f32_vals, elem_dtype):
    """Pack 8 f32 values into v8bf16, rounding to nearest even (the `rne=1` knob of the gfx11 bf16 dK/dV and dQ objects).

    `fmha_common_gfx1201.bf16_trunc_pack_v8` truncates (round toward zero), which leaves the bf16 dK/dV/dB error at 1.6-1.7x a round-to-nearest
    reference (f16 is at parity). Here `x + 0x7FFF + ((x >> 16) & 1)` is formed in integer arithmetic before the upper half is taken. For a
    finite value or an infinity the sum cannot carry into the sign bit (the largest magnitude is 0x7F800000), and a carry out of the mantissa into the
    exponent is the correct rounding (up to the next power of two, or to infinity). A NaN (`|x| > 0x7F800000`) would carry into the sign: it keeps its
    upper half with the quiet bit set instead (never rounded to infinity or to a negative zero).
    """
    c16 = fx.Int32(16)
    one = fx.Int32(1)
    half = fx.Int32(0x7FFF)
    absmask = fx.Int32(0x7FFFFFFF)
    inf_bits = fx.Int32(0x7F800000)
    quiet = fx.Int32(0x00400000)
    hi_mask = fx.Int32(0xFFFF0000)
    words = []
    for j in range_constexpr(8):
        b = fx.Float32(f32_vals[j]).bitcast(fx.Int32)
        lsb = arith.shrui(b, c16) & one
        rounded = b + half + lsb
        is_nan = (b & absmask) > inf_bits
        words.append(is_nan.select(b | quiet, rounded))
    pairs = [
        (words[2 * j + 1] & hi_mask) | arith.shrui(words[2 * j], c16)
        for j in range_constexpr(4)
    ]
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


RT_FLAG_AOTRITON_CTR = 4  # `rt_flags` bit 2 (bit 0: padded head, bit 1: runtime dropout; the gfx11 kernels read only this bit)


def philox_plane_base(offset_base, plane, max_seqlen_q, row_stride, rt_flags):
    """First Philox offset of a `(batch, head)` plane: `offset_base + plane * max_seqlen_q * row_stride`.

    Default (`rt_flags` bit 2 clear): the product is formed in 64 bits, as `Philox.grid_plane`, the FlyDSL forward and gfx950 / gfx1201 do.
    `rt_flags` bit 2 set: AOTriton's Triton arithmetic, a 32-bit product (`plane * max_seqlen_q * row_stride` wraps at 2**31) that is
    sign-extended before the 64-bit add. A backward must use the convention of the forward
    that drew the mask: bit 2 set after an AOTriton forward, clear after a FlyDSL forward. The two agree while the product is below 2**31.
    """
    wide = fx.Int64(offset_base) + fx.Int64(plane) * fx.Int64(max_seqlen_q) * fx.Int64(
        row_stride
    )
    narrow = fx.Int64(offset_base) + fx.Int64(
        fx.Int32(plane) * fx.Int32(max_seqlen_q) * fx.Int32(row_stride)
    )
    return (
        (fx.Int32(rt_flags) & fx.Int32(RT_FLAG_AOTRITON_CTR))
        == fx.Int32(RT_FLAG_AOTRITON_CTR)
    ).select(narrow, wide)
