# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Core IR primitives + lowering passes.

`ck_dsl.core` is the foundation of the DSL: every higher-level helper
(``ck_dsl.helpers``, ``ck_dsl.instances``) builds on the SSA `IRBuilder`
defined here, and every kernel emitted by this DSL is lowered via the
passes in this package.

Modules:

  - ``ir``         : `IRBuilder`, `KernelDef`, `Value`, `Op`, `Region`,
                     plus type system (`F16`, `F32`, `I32`, `I64`,
                     `VectorType`, `PtrType`, `SmemType`).
  - ``ir_print``   : MLIR-style textual dump (`print_ir(kernel)`).
  - ``lower_llvm`` : `lower_kernel_to_llvm(kernel) -> str` AMDGPU LLVM IR.
  - ``lower_hip``  : `lower_kernel_to_hip(kernel) -> str` HIP C++ text
                     (debug-only path; the production pipeline uses
                     LLVM IR).

The top-level package re-exports the most commonly-used names; this
module exists so the layering is explicit.
"""

from __future__ import annotations

from .ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I1,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    Op,
    PtrType,
    Region,
    SmemType,
    Type,
    Value,
    VectorType,
)
from .ir_print import print_ir
from .lower_hip import lower_kernel_to_hip
from .lower_llvm import lower_kernel_to_llvm
from .passes import (
    PassStats,
    canonicalize_region,
    eliminate_dead_pure_ops,
    optimize_kernel,
)

__all__ = [
    "BF16",
    "F16",
    "F32",
    "FP8E4M3",
    "I1",
    "I8",
    "I32",
    "I64",
    "IRBuilder",
    "KernelDef",
    "Op",
    "PtrType",
    "Region",
    "SmemType",
    "Type",
    "Value",
    "VectorType",
    "print_ir",
    "lower_kernel_to_hip",
    "lower_kernel_to_llvm",
    "PassStats",
    "canonicalize_region",
    "eliminate_dead_pure_ops",
    "optimize_kernel",
]
