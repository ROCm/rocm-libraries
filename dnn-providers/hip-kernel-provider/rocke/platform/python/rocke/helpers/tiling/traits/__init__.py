# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.traits -- the MMA catalog (dense/sparse/scaled) + its loader/validator.

Public surface: the typed catalog row and its parts, and the loader. The catalog
(``data/mma_traits.json``) is hand-maintained; ``README.md`` beside this file holds its schema and
edit procedure.
"""

from __future__ import annotations

from .mma_traits import (
    DEFAULT_TRAITS_PATH,
    MmaLayoutParams,
    MmaScale,
    MmaTraits,
    MmaTraitsCatalog,
    load_mma_traits,
)

__all__ = [
    "MmaLayoutParams",
    "MmaScale",
    "MmaTraits",
    "MmaTraitsCatalog",
    "load_mma_traits",
    "DEFAULT_TRAITS_PATH",
]
