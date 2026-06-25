################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from typing import Dict

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


def load_tensile_metadata() -> Dict[str, ParamMeta]:
    """Pull defaults and valid ranges from Tensile's validParameters
    and defaultBenchmarkCommonParameters.  Returns Dict[str, ParamMeta].
    Called once per OptimizationParams instance."""
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
