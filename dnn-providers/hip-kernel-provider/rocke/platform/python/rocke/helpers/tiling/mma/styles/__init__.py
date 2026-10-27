# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout styles -- the operand-layout STRATEGY seam (rev-7 Principle 8). One file per style.

``LayoutStyle`` is the extension substrate; ``CanonicalStyle`` (default) and ``InterleavedStyle`` are the
first two profiles. See :mod:`rocke.helpers.tiling.mma.styles.base` and ``docs/tiling_api_contract.md``.
"""

from __future__ import annotations

from .base import AtomNumbers, LayoutStyle
from .canonical import CanonicalStyle
from .interleaved import InterleavedStyle

__all__ = ["LayoutStyle", "AtomNumbers", "CanonicalStyle", "InterleavedStyle"]
