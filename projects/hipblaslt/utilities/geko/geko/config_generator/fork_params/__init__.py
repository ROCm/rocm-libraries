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

from geko.config_generator.fork_params.hw_profiles.gfx950.optimization_param import (
    GFX950Params,
    GFX950GAParams,
)
from geko.config_generator.fork_params.hw_profiles.gfx942.optimization_param import (
    GFX942Params,
    GFX942GAParams,
)
from geko.config_generator.fork_params.hw_profiles.gfx950.post_processor import (
    GFX950PostProcessor,
    GFX950GAPostProcessor,
)
from geko.config_generator.fork_params.hw_profiles.gfx942.post_processor import (
    GFX942PostProcessor,
    GFX942GAPostProcessor,
)

_GFX942_ARCHS = (
    "gfx942",
    "gfx942_80cu",
    "gfx942_38cu",
    "gfx942_20cu",
    "gfx942_228cu",
)

_GFX950_ARCHS = (
    "gfx950",
    "gfx950_128cu",
)

_HEURISTIC_PROFILES = {}
_HEURISTIC_PROFILES.update({a: GFX950Params for a in _GFX950_ARCHS})
_HEURISTIC_PROFILES.update({a: GFX942Params for a in _GFX942_ARCHS})

_GA_PROFILES = {}
_GA_PROFILES.update({a: GFX950GAParams for a in _GFX950_ARCHS})
_GA_PROFILES.update({a: GFX942GAParams for a in _GFX942_ARCHS})

_HEURISTIC_POST_PROCESSORS = {}
_HEURISTIC_POST_PROCESSORS.update({a: GFX950PostProcessor for a in _GFX950_ARCHS})
_HEURISTIC_POST_PROCESSORS.update({a: GFX942PostProcessor for a in _GFX942_ARCHS})

_GA_POST_PROCESSORS = {}
_GA_POST_PROCESSORS.update({a: GFX950GAPostProcessor for a in _GFX950_ARCHS})
_GA_POST_PROCESSORS.update({a: GFX942GAPostProcessor for a in _GFX942_ARCHS})


def get_optimization_params(config):
    """Factory: return the OptimizationParams subclass for ``config['ARCH']``.

    Uses the GA profile when ``config['GA']`` is true, otherwise the heuristic
    profile. Missing ``ARCH`` raises ``KeyError`` from the registry lookup.
    """
    is_ga = config.get("GA", False)
    registry = _GA_PROFILES if is_ga else _HEURISTIC_PROFILES
    return registry[config["ARCH"]](config)


def get_post_processor(config):
    """Factory: return the PostProcessor for ``config['ARCH']``, or ``None``.

    Uses the GA post-processor registry when ``config['GA']`` is true,
    otherwise the heuristic registry. Unknown ``ARCH`` yields ``None``.
    """
    is_ga = config.get("GA", False)
    registry = _GA_POST_PROCESSORS if is_ga else _HEURISTIC_POST_PROCESSORS
    cls = registry.get(config["ARCH"])
    return cls(config) if cls else None
