# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
