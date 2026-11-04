# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``waves_per_eu`` as a space's outer knob.

A family that sweeps the AMDGPU occupancy hint does it outside the pruned knob
walk and names it in the id stem, ``{variant_id}_wpe{N}@{config_key}``
(``wpenone`` for a kernel that leaves it unset). One class, so every family
spells that stored identifier the same way; the values each sweep level tries
stay the family's (:meth:`KnobSpace.outer_values`).
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass

from .space import KnobSpace


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
