# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``CanonicalStyle`` -- the default layout style: the operand is loaded MMA-ready straight from global
memory (the atom's native StridedK distribution), no LDS interleave bridge.

Composes the internal warp-encoding calculators (``a/b/c_warp_encoding``); it adds no register reorder,
so the MMA-ready descriptor IS the atom-canonical descriptor. A K-contiguous global source makes this
the natural choice; a free-contiguous source wants :class:`InterleavedStyle` for wide coalesced loads.
"""

from __future__ import annotations

from ...fragments import TileDesc
from ...traits import MmaTraits
from ..warp_encoding import a_warp_encoding, b_warp_encoding, c_warp_encoding
from .base import LayoutStyle


class CanonicalStyle(LayoutStyle):
    name = "canonical"

    def operand_desc(
        self, traits: MmaTraits, *, role: str, free_sub: int, k_sub: int
    ) -> TileDesc:
        if role == "A":
            layout = a_warp_encoding(traits, m_iter=free_sub, k_iter=k_sub)
            return TileDesc((free_sub * traits.m, k_sub * traits.k), layout)
        if role == "B":
            layout = b_warp_encoding(traits, n_iter=free_sub, k_iter=k_sub)
            return TileDesc((free_sub * traits.n, k_sub * traits.k), layout)
        raise ValueError(f"operand role must be 'A' or 'B' -- got {role!r}")

    def accumulator_desc(self, traits: MmaTraits, *, m_sub: int, n_sub: int) -> TileDesc:
        layout = c_warp_encoding(traits, m_iter=m_sub, n_iter=n_sub)
        return TileDesc((m_sub * traits.m, n_sub * traits.n), layout)
