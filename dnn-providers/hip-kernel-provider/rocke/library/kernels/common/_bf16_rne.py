# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scalar-path f32 -> bf16 round-to-nearest-even for finite values.

The vector-path sibling with the same rounding is
:func:`kernels.common._attention_bwd_valu.rne_noguard_bf16`; it packs a whole
``<n x f32>`` fragment through vector ops and so emits different IR. This one
takes four scalar f32 values -- e.g. softmax probabilities already extracted
for a reduction -- and packs them without first re-forming a vector.
"""

from __future__ import annotations

from collections.abc import Sequence

from rocke.core.ir import BF16, I32, IRBuilder, Value, VectorType

__all__ = ["bf16x4_rne_finite"]

# v_perm_b32 selector taking the high halves: [lo = src1, hi = src0].
_HI_HALVES = 0x07060302


def bf16x4_rne_finite(b: IRBuilder, vals: Sequence[Value]) -> Value:
    """Four finite f32 -> ``<4 x bf16>``, round-to-nearest-even, no NaN path.

    Targets without an f32 -> bf16 convert (gfx942) lower ``fptrunc`` to five
    VALU per element because it must preserve NaN. For finite inputs the
    rounding is two integer ops per element and one ``v_perm`` per pair.
    """
    if len(vals) != 4:
        raise ValueError(f"bf16x4_rne_finite packs four values, got {len(vals)}")
    rounded = []
    for v in vals:
        bits = b.bitcast(v, I32)
        lsb = b.land(b.lshr(bits, b.const_i32(16)), b.const_i32(1))
        rounded.append(b.add(b.add(bits, b.const_i32(0x7FFF)), lsb))
    sel = b.const_i32(_HI_HALVES)
    words = [b.perm_b32(rounded[2 * j + 1], rounded[2 * j], sel) for j in range(2)]
    return b.vec_bitcast(b.vec_pack(words, I32), VectorType(BF16, 4))
