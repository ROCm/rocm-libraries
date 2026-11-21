# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ping-pong direct forward convolution on wave64 MFMA with LDS DMA (gfx950).

A dense NHWC KxK stride-1 convolution on the ping-pong pipeline of
:mod:`kernels.common.conv_direct_pp_mfma` (LDS DMA, two wave ranks, the
ping-pong schedule). This module supplies the forward's weight half:

* **KRSC weights as stored.** The filter ``W[K][R][S][C]`` is contiguous
  along the channels, the reduction, so the weights of one tap are staged as
  ``[block_k output rows][64 channels]`` like an input row: each row is 8
  granules of 16 bytes, granule ``g`` of row ``k`` stored at slot
  ``g ^ (k % 8)``. Rank 0 streams the next tap's ``block_k x 64`` tile.
* **Plain A reads.** An A fragment is one 16-byte read of 8 channels of an
  output row, ``block_k`` 64, 128 or 256.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import KernelDef, Value

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
    "FwdPpMfmaSpec",
    "build_fwd_pp_mfma",
    "fwd_pp_mfma_reason",
    "fwd_pp_mfma_tile_efficiency",
    "is_valid_fwd_pp_mfma_spec",
]

BLOCK_K = (64, 128, 256)


@dataclass(frozen=True)
class FwdPpMfmaSpec(PpMfmaSpec):
    """One configuration of :func:`build_fwd_pp_mfma`."""

    name: str = "fwd_pp_mfma"


def fwd_pp_mfma_tile_efficiency(spec: FwdPpMfmaSpec) -> float:
    """Real work over the padded tile grid (pixels x output channels)."""
    return pp_mfma_tile_efficiency(spec)


def fwd_pp_mfma_reason(spec: FwdPpMfmaSpec, arch: str = "gfx950") -> "str | None":
    return pp_mfma_reason(
        spec,
        arch,
        family="fwd_pp_mfma",
        block_ks=BLOCK_K,
        block_k_reason="block_k must be 64, 128 or 256",
    )


def is_valid_fwd_pp_mfma_spec(
    spec: FwdPpMfmaSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    why = fwd_pp_mfma_reason(spec, arch)
    return (why is None, why or "ok")


class _FwdWeights(PpWeights):
    """KRSC weights staged as ``[block_k rows][64 channels]`` per tap."""

    def dma_geometry(self, ctx: PpCtx) -> None:
        # [BK output rows][64 channels] per tap. Piece q = wave_g * W_PER_WAVE + i
        # holds weight rows 8q .. 8q + 7 of the tile; lane l writes row 8q + l / 8,
        # slot l % 8, which holds logical granule (l % 8) ^ (l / 8) (row % 8 ==
        # l / 8).
        b, ci = ctx.b, ctx.ci
        spec = ctx.spec
        p = spec.problem
        w_per_wave = spec.block_k * CK * 2 // 1024 // RANK_WAVES
        l_row = b.div(ctx.lane, ci(8))
        l_gran = b.xor(b.mod(ctx.lane, ci(8)), l_row)
        w_c = b.mul(l_gran, ci(GRAN))
        taps_c = b.mul(ctx.C, ci(p.KH * p.KW))  # filter row length (KRSC)
        self.w_pieces = []
        for i in range(w_per_wave):
            q = b.add(b.mul(ctx.wave_g, ci(w_per_wave)), ci(i))
            k_glob = b.add(ctx.k0, b.add(b.mul(q, ci(8)), l_row))
            off = b.add(b.mul(k_glob, taps_c), w_c)
            self.w_pieces.append((off, b.cmp_lt(k_glob, ctx.K), b.mul(q, ci(1024))))
        self.w_c = w_c
        ctx.shared.update(l_row=l_row, l_gran=l_gran, lane_c=w_c)

    def issue(self, ctx: PpCtx, buf: int, tap: int, c_base: Value) -> None:
        """DMA the weights of ``tap`` for channels ``c_base ..`` into buffer
        ``buf``: rows past K and channels past C zero-filled."""
        b, ci = ctx.b, ctx.ci
        tap_off = b.add(b.mul(ci(tap), ctx.C), c_base)
        c_ok = b.cmp_lt(b.add(c_base, self.w_c), ctx.C)
        for off, k_ok, lds_off in self.w_pieces:
            voff = b.select(
                b.land(k_ok, c_ok), b.mul(b.add(off, tap_off), ctx.c2), ctx.oob
            )
            b.async_buffer_load_lds(
                ctx.w_rsrc, ctx.lds_at(ctx.w_addr[buf], lds_off), voff, ctx.c0, 4
            )

    def a_geometry(self, ctx: PpCtx) -> None:
        # A (weights): row k = wave_k * wave_k + kf * 16 + lane % 16, granule
        # part * 4 + lane / 16 -> slot granule ^ (k % 8).
        b, ci = ctx.b, ctx.ci
        spec = ctx.spec
        l16 = b.mod(ctx.lane, ci(16))
        lhi = b.div(ctx.lane, ci(16))
        self.a_idx = {}
        for kf in range(spec.k_frags):
            k_loc = b.add(b.mul(ctx.wave_k, ci(spec.wave_k)), b.add(ci(kf * ATOM), l16))
            for part in range(PARTS_C):
                g = b.add(ci(part * 4), lhi)
                slot = b.xor(g, b.mod(k_loc, ci(8)))
                self.a_idx[(kf, part)] = b.add(
                    b.mul(k_loc, ci(CK)), b.mul(slot, ci(GRAN))
                )
        ctx.shared.update(lane_hi=lhi, l16=l16)

    def read_a(self, ctx: PpCtx, buf: int, kf: int, part: int) -> Value:
        return ctx.b.smem_load_vN(
            ctx.w_bufs[buf], ctx.c0, self.a_idx[(kf, part)], dtype=ctx.io_type, n=GRAN
        )


def build_fwd_pp_mfma(spec: FwdPpMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_pp_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdPpMfmaSpec for {arch}: {why}")
    return emit_pp_mfma(spec, _FwdWeights())
