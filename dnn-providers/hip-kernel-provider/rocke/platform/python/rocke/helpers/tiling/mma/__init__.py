# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.mma -- the MMA operation (public) + the warp-encoding calculators (internal).

Public: front-door :class:`TileMma` (target-aware MMA resolution + the atom-grid driver) and its
:class:`Tiling` policy, plus the composed pieces :class:`TileMmaPlan` (design) and
:class:`TileMmaDriver` (iteration) -- toolbox tier, for authors who want to customise (see
``docs/tiling_api_contract.md``). The ``warp_encoding`` calculators (``a/b/c_warp_encoding``) are
INTERNAL machinery -- the plan wraps them; they are not re-exported here. The foundational
``WarpDistributionEncoding`` type and the ``RegisterMapper`` live at the PACKAGE ROOT
(:mod:`rocke.helpers.tiling.encoding`, :mod:`rocke.helpers.tiling.register_mapper`), since they are not
MMA-specific.
"""

from __future__ import annotations

from .driver import TileMmaDriver
from .mma_operation import TileMma
from .plan import Tiling, TileMmaPlan
from .styles import CanonicalStyle, InterleavedStyle, LayoutStyle

__all__ = [
    "TileMma", "Tiling", "TileMmaPlan", "TileMmaDriver",
    "LayoutStyle", "CanonicalStyle", "InterleavedStyle",
]
