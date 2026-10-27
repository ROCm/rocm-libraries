# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 scaled-WMMA operand contracts shared by packing and lowering."""

from __future__ import annotations

from dataclasses import dataclass

from .target import (
    MmaOp,
    MmaScaleDType,
    _gfx1250_scale_combination_valid,
    _mma_op_by_id,
    _op_id_family,
)
from ..dtypes import dtype_info
from ..storage import BitPacking, FragmentPacking, MatrixFragmentLayout


def scaled_matrix_layout(dtype: str, abi_words: int) -> MatrixFragmentLayout:
    """gfx1250 scaled operand layout, independent of atom availability.

    Matrix formats share storage mechanics but retain their atom-selected layout.
    """
    info = dtype_info(dtype)
    chunks = {"fp8e4m3": 16, "bf8e5m2": 16, "fp6e2m3": 32, "fp6e3m2": 32, "fp4e2m1": 32}
    if info.name not in chunks:
        raise ValueError(f"unsupported scaled matrix layout {dtype!r}")
    return MatrixFragmentLayout(
        FragmentPacking(BitPacking(info.encoded_bits), 64, 32, abi_words),
        chunks[info.name],
        2,
        16,
    )


@dataclass(frozen=True)
class ScalePacking:
    """Pack consecutive K groups into a word, first group in its low byte.

    Each encoded scale occupies one byte. A and B use this ordering independently;
    the integer word is only a carrier for the encoded floating-point scales.
    """

    count: int
    block_k: int

    def __post_init__(self) -> None:
        if self.block_k <= 0:
            raise ValueError("scale block_k must be positive")
        if not isinstance(self.count, int) or self.count not in (1, 2, 4, 8):
            raise ValueError("carrier_bits must be 8, 16, 32, or 64")

    @property
    def packing(self) -> BitPacking:
        return BitPacking(self.element_bits)

    @property
    def fragment(self) -> FragmentPacking:
        return FragmentPacking(self.packing, self.count, self.word_bits, 1)

    @property
    def element_bits(self) -> int:
        return 8

    @property
    def word_bits(self) -> int:
        return self.count * self.element_bits


@dataclass(frozen=True)
class ScaledWmmaOp:
    """Backend packing derived from a supported catalog operand contract."""

    atom: MmaOp
    matrix_formats: tuple[int, int]
    scale_formats: tuple[int, int]
    scales: ScalePacking

    def matrix_layout(self, operand: str) -> MatrixFragmentLayout:
        if operand == "a":
            return scaled_matrix_layout(self.atom.a_dtype, self.atom.a_frag_len)
        if operand == "b":
            return scaled_matrix_layout(self.atom.b_dtype, self.atom.b_frag_len)
        raise ValueError("matrix operand must be 'a' or 'b'")

    def scale_packing(self, operand: str) -> ScalePacking:
        if operand == "a":
            count = self.atom.a_scale_frag_len
        elif operand == "b":
            count = self.atom.b_scale_frag_len
        else:
            raise ValueError("scale operand must be 'a' or 'b'")
        return ScalePacking(count, self.atom.scale_block_k)

    @property
    def op_id(self) -> str:
        return self.atom.op_id

    @property
    def scale16(self) -> bool:
        return self.scales.block_k == 16


def gfx1250_scaled_wmma(op_id: str) -> ScaledWmmaOp | None:
    """Resolve a catalog contract; LLVM selectors are backend details.

    The atom comes from whichever arch's catalog lists ``op_id``; whether a
    given target may lower it is the lowerer's question.
    """
    op_id = op_id.removeprefix("tile.")
    if _op_id_family().get(op_id) != "wmma_scaled":
        return None
    atom = _mma_op_by_id(op_id)
    formats = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
    scale_selectors = {
        MmaScaleDType.E8M0: 0,
        MmaScaleDType.E5M3: 1,
        MmaScaleDType.E4M3: 2,
    }
    # Matrix and scale formats are independent selectors for each operand.
    # Validate the backend contract independently of the catalog query model.
    if (
        atom.a_dtype not in formats
        or atom.b_dtype not in formats
        or atom.a_scale_dtype not in scale_selectors
        or atom.b_scale_dtype not in scale_selectors
        or atom.scale_block_k not in (16, 32)
        or atom.c_dtype != "fp32"
        or atom.shape != (16, 16, 128)
    ):
        raise ValueError(f"unsupported scaled WMMA backend contract: {atom.op_id}")
    if not _gfx1250_scale_combination_valid(
        atom.a_dtype, atom.b_dtype, atom.a_scale_dtype, atom.b_scale_dtype
    ):
        raise ValueError("unsupported scaled WMMA matrix/scale format combination")
    counts = (atom.a_scale_frag_len, atom.b_scale_frag_len)
    if any(count != atom.k // atom.scale_block_k for count in counts):
        raise ValueError(f"unsupported scaled WMMA scale fragment lengths: {counts}")
    return ScaledWmmaOp(
        atom=atom,
        matrix_formats=(formats[atom.a_dtype], formats[atom.b_dtype]),
        scale_formats=(
            scale_selectors[atom.a_scale_dtype],
            scale_selectors[atom.b_scale_dtype],
        ),
        scales=ScalePacking(count=atom.a_scale_frag_len, block_k=atom.scale_block_k),
    )
