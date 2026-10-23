# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``waves_per_eu`` as the attention spaces' outer knob.

Both attention spaces sweep the AMDGPU occupancy hint outside the pruned
knob walk and name it in the id stem: ``{variant_id}_wpe{N}@{config_key}``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Optional, Tuple

from rocke.dispatch.tuning import KnobSpace

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


@dataclass(frozen=True)
class WavesPerEuSpace(KnobSpace):
    """A :class:`KnobSpace` whose outer knob is ``waves_per_eu``."""

    outer_knob = "waves_per_eu"

    def stem(self, kernel) -> str:
        value = self.field_value(kernel, "waves_per_eu")
        wpe = "none" if value is MISSING or value is None else int(value)
        return f"{self.variant_id}_wpe{wpe}"

    def stem_prefix(self) -> str:
        return f"{self.variant_id}_wpe"
