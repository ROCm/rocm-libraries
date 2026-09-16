# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 scaled-WMMA operand contracts shared by packing and lowering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E8M0ScalePacking:
    """Pack consecutive K groups into a word, first group in its low byte.

    Each scale is one E8M0 byte. Both A and B use this ordering independently;
    the integer word is only a carrier for the encoded floating-point scales.
    """

    count: int
    block_k: int

    @property
    def element_bits(self) -> int:
        return 8

    @property
    def word_bits(self) -> int:
        return self.count * self.element_bits

    @property
    def llvm_type(self) -> str:
        return f"i{self.word_bits}"


@dataclass(frozen=True)
class ScaledWmmaOp:
    """One supported matrix format and its packed E8M0 scale contract."""

    op_id: str
    matrix_dtype: str
    scales: E8M0ScalePacking

    @property
    def scale16(self) -> bool:
        return self.scales.block_k == 16

    @property
    def matrix_format(self) -> int:
        return {"fp8": 0, "fp4": 4}[self.matrix_dtype]


_SCALE = E8M0ScalePacking(count=4, block_k=32)
_SCALE16 = E8M0ScalePacking(count=8, block_k=16)
_GFX1250_WMMA_SCALE = {
    op_id: ScaledWmmaOp(op_id=op_id, matrix_dtype=dtype, scales=packing)
    for family, packing in (("wmma_scale", _SCALE), ("wmma_scale16", _SCALE16))
    for dtype in ("fp8", "fp4")
    for op_id in (f"{family}_f32_16x16x128_{dtype}_{dtype}",)
}


def gfx1250_scaled_wmma(op_id: str) -> ScaledWmmaOp | None:
    """Resolve a catalog ID or concrete tile op name; reject unknown variants."""
    return _GFX1250_WMMA_SCALE.get(op_id.removeprefix("tile."))
