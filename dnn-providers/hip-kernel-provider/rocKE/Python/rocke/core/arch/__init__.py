# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ROCKE architecture metadata package (polymorphic-core SSOT).

Exposes :class:`ArchTarget` and the MMA catalog. Hardware facts only — no
pipeline/scheduler vocabulary, no LLVM intrinsic text, no ``dispatcher/`` imports.
"""

from .target import (  # noqa: F401
    ArchTarget,
    LayoutMap,
    MemoryCapabilities,
    MmaCatalog,
    MmaOp,
    ResourceLimits,
    arch_from_isa,
    known_arches,
    normalize_dtype,
)

__all__ = [
    "ArchTarget",
    "LayoutMap",
    "MemoryCapabilities",
    "MmaCatalog",
    "MmaOp",
    "ResourceLimits",
    "arch_from_isa",
    "known_arches",
    "normalize_dtype",
]
