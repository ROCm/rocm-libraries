# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.traits -- the MMA catalog (dense/sparse/scaled) + its loader/validator.

Public surface: the typed catalog row and its parts, and the loader. The catalog
(``data/mma_traits.json``) is hand-maintained; ``README.md`` beside this file holds its schema and
edit procedure. The backend coverage record (``data/backend_coverage.json``, read by
``load_backend_coverage``) says which catalog instructions rocke's backend issues on each target.
"""

from __future__ import annotations

from .backend_coverage import (
    COVERAGE_CLASSES,
    DEFAULT_COVERAGE_PATH,
    BackendCoverage,
    CoverageEntry,
    load_backend_coverage,
)
from .mma_traits import (
    DEFAULT_TRAITS_PATH,
    MmaLayoutParams,
    MmaScale,
    MmaTraits,
    MmaTraitsCatalog,
    load_mma_traits,
)

__all__ = [
    "COVERAGE_CLASSES",
    "DEFAULT_COVERAGE_PATH",
    "DEFAULT_TRAITS_PATH",
    "BackendCoverage",
    "CoverageEntry",
    "MmaLayoutParams",
    "MmaScale",
    "MmaTraits",
    "MmaTraitsCatalog",
    "load_backend_coverage",
    "load_mma_traits",
]
