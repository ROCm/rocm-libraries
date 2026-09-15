# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Dict
from functools import lru_cache

from geko.config_generator.shared_utils import ParamMeta


def _format_range(rng, start_elements=2, end_elements=2):
    """Compact string representation of a valid-values list."""
    if not isinstance(rng, list):
        return str(rng)
    if len(rng) <= start_elements + end_elements:
        return str(rng)
    start = ", ".join(map(str, rng[:start_elements]))
    end = ", ".join(map(str, rng[-end_elements:]))
    return f"[{start}, ..., {end}]"

@lru_cache(maxsize=1)
def load_tensile_metadata() -> Dict[str, ParamMeta]:
    """Pull defaults and valid ranges from Tensile's validParameters
    and defaultBenchmarkCommonParameters.  Returns Dict[str, ParamMeta].
    Loaded lazily and cached for reuse across callers."""
    from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
    from Tensile.Common.ValidParameters import validParameters

    defaults = {}
    for dp in defaultBenchmarkCommonParameters:
        defaults.update(dp)

    meta = {}
    for name in set(validParameters.keys()) & set(defaults.keys()):
        meta[name] = ParamMeta(
            name=name,
            default_value=defaults[name],
            valid_range=_format_range(validParameters[name]),
        )
    return meta
