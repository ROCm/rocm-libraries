# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Static analysis helpers for CK DSL kernels.

The optimisation runbook repeatedly asks for evidence from the lowered
program: how many MFMAs were issued, whether an async DRAM->LDS intrinsic
actually appeared, whether the epilogue lowered to vector stores, how many
barriers/waits are in the hot path, and what resources the code object uses.

This package turns those checks into reusable Python APIs instead of ad hoc
`llvm-objdump | grep` shell fragments.
"""

from __future__ import annotations

from .ir import LlvmIrStats, analyze_llvm_ir
from .isa import (
    HsacoAnalysis,
    IsaStats,
    ResourceInfo,
    analyze_hsaco,
    parse_isa,
    parse_resources,
)
from .report import VariantReport, compare_variant_reports

__all__ = [
    "HsacoAnalysis",
    "IsaStats",
    "LlvmIrStats",
    "ResourceInfo",
    "VariantReport",
    "analyze_hsaco",
    "analyze_llvm_ir",
    "compare_variant_reports",
    "parse_isa",
    "parse_resources",
]
