# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ping-pong direct dgrad on wave64 MFMA with LDS DMA (gfx950).

A dense NHWC KxK stride-1 dgrad, computed as the forward pass over the
transposed problem -- dY in, dX out, flipped padding -- on the ping-pong
pipeline of :mod:`kernels.common.conv_direct_pp_mfma` (LDS DMA, two wave
ranks, the ping-pong schedule). The kernel reads the original KRSC filter
itself, flipped and transposed, so there is no weight pre-pass. Throughout
this module "C" and "K" are the pass's: C is the reduction (dY channels), K
the outputs (dX channels).

* **In-kernel transposed weights.** The filter is ``W[C][R][S][K]`` in this
  pass's terms, contiguous along the outputs. The weights of one tap are
  staged as ``[64 reduction rows][block_k outputs]`` and the A fragments are
  read with two ``ds_read_b64_tr_b16`` per 16x16x32 operand. Granule ``g`` of
  row ``r`` sits at slot ``g ^ 2 * (r % 4 + 4 * ((r / 8) % 2))``, which puts
  the 32 lanes of a transposing-read phase on distinct 8-byte slots of the
  256-byte bank period (rows of at least 256 bytes: block_k >= 128).

AOT: the forward direct-conv kernarg block over the transposed problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import IRBuilder, KernelDef, Value

from kernels.common.conv_direct_grouped import dgrad_4c_transposed_problem
from kernels.common.conv_direct_pp_mfma import (
    ATOM,
    CK,
    GRAN,
    PARTS_C,
    RANK_WAVES,
    PpCtx,
    PpMfmaSpec,
    PpWeights,
    emit_pp_mfma,
    pp_mfma_reason,
    pp_mfma_tile_efficiency,
)

__all__ = [
    "BLOCK_K",
    "DgradPpMfmaSpec",
    "build_dgrad_pp_mfma",
    "dgrad_pp_mfma_problem",
    "dgrad_pp_mfma_reason",
    "dgrad_pp_mfma_tile_efficiency",
    "is_valid_dgrad_pp_mfma_spec",
]

BLOCK_K = (128, 256)


# The forward problem the dgrad of a problem runs as: the 4c dgrad's
# transposition (dY in, dX out, PAD and an explicit pad_w flipped; the stride
# is kept, so the validator rejects a strided dgrad).
dgrad_pp_mfma_problem = dgrad_4c_transposed_problem


@dataclass(frozen=True)
class DgradPpMfmaSpec(PpMfmaSpec):
    """One configuration of :func:`build_dgrad_pp_mfma`.

    ``problem`` is the forward pass the dgrad runs as
    (:func:`dgrad_pp_mfma_problem`): cpg = the dgrad's K, kpg = its C.
    """

    name: str = "dgrad_pp_mfma"


def dgrad_pp_mfma_tile_efficiency(spec: DgradPpMfmaSpec) -> float:
    """Real work over the padded tile grid (pixels x output channels)."""
    return pp_mfma_tile_efficiency(spec)


def dgrad_pp_mfma_reason(spec: DgradPpMfmaSpec, arch: str = "gfx950") -> "str | None":
    return pp_mfma_reason(
        spec,
        arch,
        family="dgrad_pp_mfma",
        block_ks=BLOCK_K,
        block_k_reason="block_k must be 128 or 256 (256-byte weight rows for the swizzle)",
        needs_ds_read_tr=True,
    )


def is_valid_dgrad_pp_mfma_spec(
    spec: DgradPpMfmaSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    why = dgrad_pp_mfma_reason(spec, arch)
    return (why is None, why or "ok")


def _tr_swz(b: IRBuilder, row: Value) -> Value:
    """Granule XOR of weight row ``row``, 2 * (row % 4 + 4 * ((row // 8) % 2)):
    the 32 lanes of a transposing-read phase (rows 8b + i, b in 0..1, i in
    0..3, column pairs 4j) land on 32 distinct 8-byte slots of a 256-byte bank
    period. Not an entry of ``rocke.helpers.layouts.XOR_SWIZZLE_TABLE``, whose
    byte-offset swizzles cover the listed MFMA tile shapes but not this
    64-row tile read across rows by ``ds_read_b64_tr_b16``; the C++ mirror is
    ``tr_swz`` in ``conv_direct_dgrad_pp_mfma.cpp``."""
    lo = b.mod(row, b.const_i32(4))
    hi = b.mod(b.div(row, b.const_i32(8)), b.const_i32(2))
    return b.mul(b.add(lo, b.mul(hi, b.const_i32(4))), b.const_i32(2))


class _DgradWeights(PpWeights):
    """The flipped KRSC filter staged as ``[64 reduction rows][block_k
    outputs]`` per tap, read back with transposing reads."""

    def dma_geometry(self, ctx: PpCtx) -> None:
        # [64 reduction rows][BK outputs] per tap, 16-byte granule g of row r
        # stored at slot g ^ _tr_swz(r). Piece q covers rows q * RPI ..; lane l
        # writes row q * RPI + l // GPR, slot l % GPR.
        b, ci = ctx.b, ctx.ci
        spec = ctx.spec
        p = spec.problem
        BK = spec.block_k
        w_per_wave = BK * CK * 2 // 1024 // RANK_WAVES
        GPR = BK // GRAN  # weight granules per row
        RPI = 1024 // (BK * 2)  # weight rows per 1 KiB piece
        row_len = b.mul(ctx.K, ci(p.KH * p.KW))  # filter row: TAPS x outputs
        self.w_pieces = []
        for i in range(w_per_wave):
            q = b.add(b.mul(ctx.wave_g, ci(w_per_wave)), ci(i))
            row = b.add(b.mul(q, ci(RPI)), b.div(ctx.lane, ci(GPR)))
            gran = b.xor(b.mod(ctx.lane, ci(GPR)), _tr_swz(b, row))
            cc = b.add(ctx.k0, b.mul(gran, ci(GRAN)))
            self.w_pieces.append(
                (
                    row,
                    b.add(b.mul(row, row_len), cc),
                    b.cmp_lt(cc, ctx.K),
                    b.mul(q, ci(1024)),
                )
            )
        self.row_len = row_len

    def issue(self, ctx: PpCtx, buf: int, tap: int, c_base: Value) -> None:
        """DMA the weights of ``tap`` for reduction rows ``c_base ..`` into
        buffer ``buf``: the flipped filter tap, rows past C zero-filled."""
        b, ci = ctx.b, ctx.ci
        p = ctx.spec.problem
        KH, KW = p.KH, p.KW
        r, s_ = divmod(tap, KW)
        tap_f = (KH - 1 - r) * KW + (KW - 1 - s_)
        base = b.add(b.mul(c_base, self.row_len), b.mul(ci(tap_f), ctx.K))
        for row, off, k_ok, lds_off in self.w_pieces:
            ok = b.land(k_ok, b.cmp_lt(b.add(c_base, row), ctx.C))
            voff = b.select(ok, b.mul(b.add(off, base), ctx.c2), ctx.oob)
            b.async_buffer_load_lds(
                ctx.w_rsrc, ctx.lds_at(ctx.w_addr[buf], lds_off), voff, ctx.c0, 4
            )

    def a_geometry(self, ctx: PpCtx) -> None:
        # A (weights), transposing reads: lane (tb, ti, tj) = (l // 16, (l // 4) %
        # 4, l % 4) reads outputs m0 + 4 tj .. +3 of reduction row 8 tb + ti (+ 4
        # for the second read); the hardware hands lane l the reduction values
        # 8 tb .. 8 tb + 7 of output m0 + l % 16 -- the A operand.
        b, ci = ctx.b, ctx.ci
        spec = ctx.spec
        BK = spec.block_k
        tb = b.div(ctx.lane, ci(16))
        ti = b.mod(b.div(ctx.lane, ci(4)), ci(4))
        tj = b.mod(ctx.lane, ci(4))
        self.a_idx = {}
        for kf in range(spec.k_frags):
            col = b.add(
                b.mul(ctx.wave_k, ci(spec.wave_k)),
                b.add(ci(kf * ATOM), b.mul(tj, ci(4))),
            )
            for part in range(PARTS_C):
                for rd in range(2):
                    row = b.add(b.add(ci(part * 32 + 4 * rd), b.mul(tb, ci(8))), ti)
                    slot = b.xor(b.div(col, ci(GRAN)), _tr_swz(b, row))
                    self.a_idx[(kf, part, rd)] = b.add(
                        b.mul(row, ci(BK)),
                        b.add(b.mul(slot, ci(GRAN)), b.mod(col, ci(GRAN))),
                    )
        ctx.shared["lane_hi"] = tb

    def read_a(self, ctx: PpCtx, buf: int, kf: int, part: int) -> Value:
        b = ctx.b
        lo = b.ds_read_tr16_b64(
            ctx.w_bufs[buf], ctx.c0, self.a_idx[(kf, part, 0)], dtype=ctx.io_type
        )
        hi = b.ds_read_tr16_b64(
            ctx.w_bufs[buf], ctx.c0, self.a_idx[(kf, part, 1)], dtype=ctx.io_type
        )
        return b.vec_concat(lo, hi)


def build_dgrad_pp_mfma(spec: DgradPpMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = dgrad_pp_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DgradPpMfmaSpec for {arch}: {why}")
    return emit_pp_mfma(spec, _DgradWeights())
