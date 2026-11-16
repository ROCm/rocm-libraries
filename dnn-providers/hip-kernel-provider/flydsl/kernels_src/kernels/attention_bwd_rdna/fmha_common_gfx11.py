# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""gfx11 / gfx12 helpers over `_if_then`, the `scf.if` context manager that the vendored `kernels.common` (FlyDSL 0.3.4) does not define
(`fmha_common_gfx1201` reaches the same helper as `common_utils._if_then` of this package's `flyc_polyfill`): the guarded cooperative-load
staging, the padded-head store, and the context manager under both. Used by `fmha_bwd_dq_gfx11.py`, `fmha_bwd_dq_f32_gfx11.py` and
`fmha_bwd_dkdv_gfx11.py`. Also the ABI-wide operand helpers (`read_operand_parts`, `bf16_pack_parts`,
`reduce_s_across_shards`): the 4-wide (MFMA16) and 8-wide (WMMA) reads, packs and accumulator tiles, so a kernel names no width. Imports
this package's `flyc_polyfill` and `rdna_layout_gfx11`.
"""

from kernels.attention_bwd_rdna.flyc_polyfill import _if_then

import flydsl.expr as fx
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

from kernels.attention_bwd_rdna import rdna_layout_gfx11 as lay


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
    """Pack 8 (MFMA16: 4) f32 values into v8 (v4) bf16, rounding to nearest even (the `rne=1` knob of the gfx11 bf16 dK/dV and dQ objects).

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
    n = len(f32_vals)
    for j in range_constexpr(n):
        b = fx.Float32(f32_vals[j]).bitcast(fx.Int32)
        lsb = arith.shrui(b, c16) & one
        rounded = b + half + lsb
        is_nan = (b & absmask) > inf_bits
        words.append(is_nan.select(b | quiet, rounded))
    pairs = [
        (words[2 * j + 1] & hi_mask) | arith.shrui(words[2 * j], c16)
        for j in range_constexpr(n // 2)
    ]
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


def bf16_pack_parts(f32_vals, elem_dtype, rne, abi):
    """The 16-bit-bf16 operand parts of `f32_vals` (`lay.op_width(abi)` values, an `acc_to_operand` result): `lay.read_count(abi)` vectors of
    `lay.read_width(abi)` values, rounded to nearest even (`rne`) or truncated."""
    pack = bf16_rne_pack_v8 if rne else bf16_trunc_pack
    width = lay.read_width(abi)
    return [
        pack(f32_vals[i * width : (i + 1) * width], elem_dtype)
        for i in range_constexpr(lay.read_count(abi))
    ]


def bf16_trunc_pack(f32_vals, elem_dtype):
    """`fmha.bf16_trunc_pack_v8` for any even count of values (8 on WMMA, 4 on MFMA16): truncation (round toward zero) to bf16."""
    from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha  # noqa: PLC0415

    c16 = fx.Int32(16)
    cmask = fx.Int32(0xFFFF0000)
    pairs = [
        fmha.pack_bf16_pair(f32_vals[2 * j], f32_vals[2 * j + 1], c16, cmask)
        for j in range_constexpr(len(f32_vals) // 2)
    ]
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


def read_vn(aperture, fetch, row, col, row_ok, width):
    """`Aperture.read_v8` for `width` elements (4 on MFMA16: one lane's operand); `fetch(row, col, width)` issues the load."""
    raw = aperture.cols.discard(fetch(row, aperture.cols.safe(col), width), col, width)
    zeros = Vec.filled(width, 0.0, aperture.cols.elem_dtype).ir_value()
    return row_ok.select(raw, zeros)


def read_operand_parts(aperture, fetch, row, col, row_ok, abi):
    """The `lay.read_count(abi)` masked reads of `lay.read_width(abi)` elements that make one operand of the lane, starting at head-dim / q / kv
    index `col` (`lay.op_origin`): two 8-wide reads on gfx11, one 8-wide on gfx12, one 4-wide on MFMA16. `row` / `row_ok` as `Aperture.read_v8`;
    `fetch(row, col, width)` issues one load. The caller concatenates the parts (its own `_cat`).
    """
    width = lay.read_width(abi)
    return [
        read_vn(aperture, fetch, row, col + fx.Index(8 * i), row_ok, width)
        for i in range_constexpr(lay.read_count(abi))
    ]


def reduce_s_across_shards(s_accs, acc, **kw):
    """`fmha.reduce_s_across_shards` with `acc` f32 elements per accumulator tile (8 on WMMA, 4 on MFMA16)."""
    from kernels.attention_bwd_cdna import fmha_common_gfx1201 as fmha  # noqa: PLC0415

    s_flat = [fx.as_ir_value(Vec(a)[r]) for a in s_accs for r in range_constexpr(acc)]
    lds_byte_base, byte0 = kw["lds_byte_base"], kw["byte0"]
    wave_id, lane, shard_id = kw["wave_id"], kw["lane"], kw["shard_id"]
    num_shards, warp_size = kw["num_shards"], kw["warp_size"]
    fastmath = kw["fastmath"]
    own = wave_id * fx.Index(kw["f32_per_wave"])
    for e in range_constexpr(len(s_flat)):
        fmha.lds_f32_store(
            lds_byte_base, byte0, own + fx.Index(e * warp_size) + lane, s_flat[e]
        )
    gpu.barrier()
    base_group = kw["q_tile_in_block"] * fx.Index(num_shards * kw["f32_per_wave"])
    for e in range_constexpr(len(s_flat)):
        v = s_flat[e]
        for k in range_constexpr(num_shards - 1):
            peer = base_group + (
                (shard_id + fx.Index(k + 1)) % fx.Index(num_shards)
            ) * fx.Index(kw["f32_per_wave"])
            v = fastmath.add(
                v,
                fmha.lds_f32_load(
                    lds_byte_base, byte0, peer + fx.Index(e * warp_size) + lane
                ),
            )
        s_flat[e] = v
    gpu.barrier()
    return [
        Vec.from_elements(
            [fx.Float32(s_flat[st * acc + r]) for r in range_constexpr(acc)],
            fx.Float32,
        ).ir_value()
        for st in range_constexpr(len(s_accs))
    ]


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
