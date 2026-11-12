# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Architecture metadata and target-name helpers.

Exposes :class:`ArchTarget`, the MMA catalog, and helpers for converting HIP
and COMGR target names. These helpers read the architecture catalog without
querying a GPU or compiler.
"""

from ..dtypes import normalize_dtype
from .target import (  # noqa: F401
    ArchTarget,
    LayoutMap,
    MemoryCapabilities,
    MmaCatalog,
    MmaDst,
    MmaEmulation,
    MmaEmulationRecipe,
    MmaOp,
    MmaScaleOperand,
    MmaSrc,
    MmaScaleBlockK,
    MmaScaleDType,
    ResourceLimits,
    arch_from_isa,
    arches_with_own_facts,
    base_arch_from_target_id,
    compiler_target_for_flavor,
    compiler_target_for_llvm_major,
    compiler_target_from_target_id,
    generic_arch_from_target_id,
    known_arches,
    listed_arch_for,
    parse_target_id,
    processor_from_target_id,
    target_id_from_isa,
    validate_arch,
)

__all__ = [
    "ArchTarget",
    "LayoutMap",
    "MemoryCapabilities",
    "MmaCatalog",
    "MmaDst",
    "MmaEmulation",
    "MmaEmulationRecipe",
    "MmaOp",
    "MmaScaleOperand",
    "MmaSrc",
    "MmaScaleBlockK",
    "MmaScaleDType",
    "ResourceLimits",
    "arch_from_isa",
    "arches_with_own_facts",
    "base_arch_from_target_id",
    "compiler_target_for_flavor",
    "compiler_target_for_llvm_major",
    "compiler_target_from_target_id",
    "generic_arch_from_target_id",
    "known_arches",
    "listed_arch_for",
    "normalize_dtype",
    "parse_target_id",
    "processor_from_target_id",
    "target_id_from_isa",
    "validate_arch",
]
