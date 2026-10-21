# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.traits -- MMA traits SSOT (dense/sparse/scaled) + loader/validator.

Public surface: the typed traits and loader. The traits table (``mma_traits.json``) is the
committed SSOT read at runtime.
"""

from __future__ import annotations

from .mma_traits import (
    DEFAULT_TRAITS_PATH,
    MmaTraits,
    MmaTraitsCatalog,
    load_mma_traits,
)

__all__ = ["MmaTraits", "MmaTraitsCatalog", "load_mma_traits", "DEFAULT_TRAITS_PATH"]
