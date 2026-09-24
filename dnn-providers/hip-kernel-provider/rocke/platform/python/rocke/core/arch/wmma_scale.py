# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 scaled-WMMA operand contracts shared by packing and lowering."""

from __future__ import annotations

from dataclasses import dataclass

from .target import ArchTarget, MmaOp, MmaScaleDType


@dataclass(frozen=True)
class ScalePacking:
    """Pack consecutive K groups into a word, first group in its low byte.

    Each encoded scale occupies one byte. A and B use this ordering independently;
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


@dataclass(frozen=True)
class ScaledWmmaOp:
    """Backend packing derived from a supported catalog operand contract."""

    atom: MmaOp
    matrix_formats: tuple[int, int]
    scale_formats: tuple[int, int]
    scales: ScalePacking

    @property
    def op_id(self) -> str:
        return self.atom.op_id

    @property
    def scale16(self) -> bool:
        return self.scales.block_k == 16


def gfx1250_scaled_wmma(op_id: str) -> ScaledWmmaOp | None:
    """Resolve a catalog contract; LLVM selectors are backend details."""
    atom = ArchTarget.from_gfx("gfx1250").mma.by_op_id(op_id.removeprefix("tile."))
    if atom is None or atom.family != "wmma_scaled":
        return None
    formats = {"fp8e4m3": 0, "bf8e5m2": 1}
    # The current backend supports E8M0 for both inputs and a shared K-group size.
    # Keep these restrictions here, independently of the catalog query model.
    if (
        atom.a_dtype not in formats
        or atom.b_dtype not in formats
        or atom.a_scale_dtype != MmaScaleDType.E8M0
        or atom.b_scale_dtype != MmaScaleDType.E8M0
        or atom.scale_block_k not in (16, 32)
        or atom.c_dtype != "fp32"
        or atom.shape != (16, 16, 128)
    ):
        raise ValueError(f"unsupported scaled WMMA backend contract: {atom.op_id}")
    counts = (atom.a_scale_frag_len, atom.b_scale_frag_len)
    if any(count != atom.k // atom.scale_block_k for count in counts):
        raise ValueError(f"unsupported scaled WMMA scale fragment lengths: {counts}")
    return ScaledWmmaOp(
        atom=atom,
        matrix_formats=(formats[atom.a_dtype], formats[atom.b_dtype]),
        scale_formats=(0, 0),  # E8M0 for each source.
        scales=ScalePacking(count=atom.a_scale_frag_len, block_k=atom.scale_block_k),
    )
