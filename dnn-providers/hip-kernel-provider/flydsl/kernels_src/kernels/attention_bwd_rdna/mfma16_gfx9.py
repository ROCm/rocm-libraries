# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The MFMA16 atom of the backward layout layer on gfx90a / gfx942 (ABI `rdna_layout_gfx11.ABI_MFMA16`): `v_mfma_f32_16x16x16{f16,bf16_1k}` as the
kernels' `wmma_acc`.

A and B operands are 4 16-bit values per lane (lane (m = l % 16, g = l // 16) holds K 4g .. 4g+3); the accumulator is 4 f32 per lane (register r of
lane (n, g) is D[4g + r][n])."""

from flydsl._mlir.dialects import rocdl as R
from flydsl.expr.rocdl import _blgp_attr
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec


def mfma_acc(dtype_str, a, b, c):
    """`c + a . b` for one 16x16x16 tile; returns the 4 f32 accumulator vector as an IR value."""
    f32x4 = Vec.make_type(4, fx.Float32)
    if dtype_str == "bf16":
        a = Vec(a).bitcast(fx.Int16)
        b = Vec(b).bitcast(fx.Int16)
        return R.mfma_f32_16x16x16bf16_1k(f32x4, a, b, c, 0, 0, _blgp_attr(0)).result
    return R.mfma_f32_16x16x16f16(f32x4, a, b, c, 0, 0, _blgp_attr(0)).result
