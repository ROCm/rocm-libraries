# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LLVM spelling for the shared gfx1250 scaled-WMMA operand contract."""

from __future__ import annotations

from dataclasses import dataclass

from ..arch.wmma_scale import ScaledWmmaOp


@dataclass(frozen=True)
class ScaledWmmaLLVM:
    """Derive LLVM carriers and intrinsic names from architecture metadata."""

    contract: ScaledWmmaOp

    @property
    def scale_type(self) -> str:
        return f"i{self.contract.scales.word_bits}"

    @property
    def matrix_types(self) -> tuple[str, str]:
        atom = self.contract.atom
        return tuple(f"<{n} x i32>" for n in (atom.a_frag_len, atom.b_frag_len))

    @property
    def intrinsic_suffix(self) -> str:
        atom = self.contract.atom
        return (
            f"f32.{atom.m}x{atom.n}x{atom.k}.f8f6f4.v{atom.c_frag_len}f32."
            f"v{atom.a_frag_len}i32.v{atom.b_frag_len}i32"
        )

    @property
    def intrinsic(self) -> str:
        mode = "scale16" if self.contract.scale16 else "scale"
        return f"llvm.amdgcn.wmma.{mode}.{self.intrinsic_suffix}"

    @property
    def declaration_key(self) -> str:
        block_k = self.contract.scales.block_k
        return f"wmma.scale.block{block_k}.gfx1250.{self.intrinsic_suffix}"
