# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ``waves_per_eu`` values the attention spaces sweep.

Both attention spaces sweep the AMDGPU occupancy hint outside the pruned
knob walk and name it in the id stem, ``{variant_id}_wpe{N}@{config_key}``,
through the shared :class:`rocke.dispatch.tuning.WavesPerEuSpace`; the values
each level tries are attention's and live here.
"""

from __future__ import annotations

from typing import Optional, Tuple

# The values each level tries; ``None`` is the kernel's own policy.
FULL_WAVES: Tuple[Optional[int], ...] = (None, 1, 2, 3, 4)
PRODUCTION_WAVES: Tuple[Optional[int], ...] = (None, 2, 4)


def waves_per_eu_sweep_values(default: int, level: str) -> Tuple[int, ...]:
    """Concrete values for ``level`` when the base spec already resolved its
    policy to ``default``; that comes first."""
    axis = PRODUCTION_WAVES if level == "production" else FULL_WAVES
    resolved: list[int] = []
    for value in axis:
        wpe = int(default if value is None else value)
        if wpe not in resolved:
            resolved.append(wpe)
    return tuple(resolved)
