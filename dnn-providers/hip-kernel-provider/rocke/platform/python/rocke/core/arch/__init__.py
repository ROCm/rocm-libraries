# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Architecture metadata and target-name helpers.

Exposes :class:`ArchTarget`, the MMA catalog, and helpers for converting HIP
and COMGR target names. These helpers read the architecture catalog without
querying a GPU or compiler.
"""

from .target import (  # noqa: F401
    ArchTarget,
    LayoutMap,
    MemoryCapabilities,
    MmaCatalog,
    MmaOp,
    ResourceLimits,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_from_target_id,
    known_arches,
    normalize_dtype,
    target_id_from_isa,
    validate_arch,
)

__all__ = [
    "ArchTarget",
    "LayoutMap",
    "MemoryCapabilities",
    "MmaCatalog",
    "MmaOp",
    "ResourceLimits",
    "arch_from_isa",
    "base_arch_from_target_id",
    "compiler_target_from_target_id",
    "known_arches",
    "normalize_dtype",
    "target_id_from_isa",
    "validate_arch",
]
