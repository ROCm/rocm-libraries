# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.mma -- the MMA operation (public) + the warp-encoding calculators (internal).

Public: :class:`TileMma` (target-aware MMA resolution from logical intent + the atom-grid MMA
driver) and its :class:`Tiling` policy. The ``warp_encoding`` calculators
(``a/b/c_warp_encoding``) are INTERNAL machinery -- ``TileMma`` wraps them; they are not
re-exported here. The foundational ``WarpDistributionEncoding`` type and the ``RegisterMapper``
live at the PACKAGE ROOT (:mod:`rocke.helpers.tiling.encoding`, :mod:`rocke.helpers.tiling.register_mapper`),
since they are not MMA-specific.
"""

from __future__ import annotations

from .mma_operation import TileMma, Tiling

__all__ = ["TileMma", "Tiling"]
