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

_GENERIC_PROFILES = {}
_GENERIC_PROFILES.update({a: GFX950GAParams for a in _GFX950_ARCHS})
_GENERIC_PROFILES.update({a: GFX942GAParams for a in _GFX942_ARCHS})

_HEURISTIC_POST_PROCESSORS = {}
_HEURISTIC_POST_PROCESSORS.update({a: GFX950PostProcessor for a in _GFX950_ARCHS})
_HEURISTIC_POST_PROCESSORS.update({a: GFX942PostProcessor for a in _GFX942_ARCHS})

_GENERIC_POST_PROCESSORS = {}
_GENERIC_POST_PROCESSORS.update({a: GFX950GAPostProcessor for a in _GFX950_ARCHS})
_GENERIC_POST_PROCESSORS.update({a: GFX942GAPostProcessor for a in _GFX942_ARCHS})


_SEARCH_SPACE_PROFILES = {
    "generic": _GENERIC_PROFILES,
    "heuristic": _HEURISTIC_PROFILES,
}

_SEARCH_SPACE_POST_PROCESSORS = {
    "generic": _GENERIC_POST_PROCESSORS,
    "heuristic": _HEURISTIC_POST_PROCESSORS,
}


def get_optimization_params(config):
    """Return the OptimizationParams for config ARCH and search_space."""
    ss = config.get("search_space", "heuristic")
    registry = _SEARCH_SPACE_PROFILES.get(ss, _HEURISTIC_PROFILES)
    return registry[config["ARCH"]](config)


def get_post_processor(config):
    """Return the PostProcessor for config ARCH and search_space, or None."""
    ss = config.get("search_space", "heuristic")
    registry = _SEARCH_SPACE_POST_PROCESSORS.get(ss, _HEURISTIC_POST_PROCESSORS)
    cls = registry.get(config["ARCH"])
    return cls(config) if cls else None
