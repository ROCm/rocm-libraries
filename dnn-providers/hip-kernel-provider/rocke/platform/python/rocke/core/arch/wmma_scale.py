# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 scaled-WMMA operand contracts shared by packing and lowering."""

from __future__ import annotations

from dataclasses import dataclass

MATRIX_FORMATS = {"fp8": 0, "bf8": 1, "fp6": 2, "bf6": 3, "fp4": 4}


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
    """One supported matrix pair and its packed E8M0 scale contract."""

    op_id: str
    matrix_dtype: str
    scales: E8M0ScalePacking
    matrix_dtype_b: str | None = None

    @property
    def scale16(self) -> bool:
        return self.scales.block_k == 16

    @property
    def matrix_format(self) -> int:
        return MATRIX_FORMATS[self.matrix_dtype]

    @property
    def matrix_format_b(self) -> int:
        return MATRIX_FORMATS[self.matrix_dtype_b or self.matrix_dtype]


_SCALE = E8M0ScalePacking(count=4, block_k=32)
_SCALE16 = E8M0ScalePacking(count=8, block_k=16)
_GFX1250_WMMA_SCALE = {
    op_id: ScaledWmmaOp(op_id=op_id, matrix_dtype=a, scales=packing, matrix_dtype_b=b)
    for family, packing in (("wmma_scale", _SCALE), ("wmma_scale16", _SCALE16))
    for a in MATRIX_FORMATS
    for b in MATRIX_FORMATS
    for op_id in (f"{family}_f32_16x16x128_{a}_{b}",)
}


def gfx1250_scaled_wmma(op_id: str) -> ScaledWmmaOp | None:
    """Resolve a catalog ID or concrete tile op name; reject unknown variants."""
    return _GFX1250_WMMA_SCALE.get(op_id.removeprefix("tile."))


SCALED_WMMA_OPS = {
    op_id: (spec.scale16, spec.matrix_format, spec.matrix_format_b)
    for mode in ("wmma_scale", "wmma_scale16")
    for a in MATRIX_FORMATS
    for b in MATRIX_FORMATS
    for op_id in (f"{mode}_f32_16x16x128_{a}_{b}",)
    if (spec := gfx1250_scaled_wmma(op_id)) is not None
}
