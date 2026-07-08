# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Core IR primitives + lowering passes.

`rocke.core` is the foundation of the DSL: every higher-level helper
(``rocke.helpers``, ``rocke.instances``) builds on the SSA `IRBuilder`
defined here, and every kernel emitted by this DSL is lowered via the
passes in this package.

Modules:

  - ``ir``         : `IRBuilder`, `KernelDef`, `Value`, `Op`, `Region`,
                     plus type system (`F16`, `F32`, `I32`, `I64`,
                     `VectorType`, `PtrType`, `SmemType`).
  - ``ir_print``   : MLIR-style textual dump (`print_ir(kernel)`); human-only.
  - ``ir_serialize``: round-trippable ``ck.dsl.ir/v1`` machine format
                     (`serialize` / `parse`) with explicit SSA ids, plus
                     `canonicalize` / `canonical_equal` for semantic diff.
  - ``verify``     : LLVM-`verify`-style well-formedness pass
                     (`verify(kernel) -> list[Diagnostic]`).
  - ``lower_llvm`` : `lower_kernel_to_llvm(kernel) -> str` AMDGPU LLVM IR
                     -- the production path, comgr-friendly.
  - ``lower_hip``  : `lower_kernel_to_hip(kernel) -> str` raw HIP C++
                     that mirrors the SSA IR one-to-one (compiles via
                     hipcc; useful for IR inspection and ISA diffs).
  - ``lower_cktile``: `lower_spec_to_cktile(spec) -> str` CK Tile-shaped
                     C++ source that composes the same templates
                     (``GemmKernel<TilePartitioner, GemmPipeline,
                     GemmEpilogue>``, ``GroupedConvolutionForwardKernel<...>``)
                     a hand-written CK Tile kernel uses. Operates on the
                     instance-level spec (e.g. :class:`UniversalGemmSpec`),
                     not the post-IR ``KernelDef``.

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
from .backend import (
    BackendError,
    BackendMismatch,
    lower_universal_gemm,
    resolve_backend,
)
from .ir_print import print_ir
from .ir_serialize import (
    canonical_equal,
    canonicalize,
    parse,
    serialize,
)
from .lower_cktile import (
    lower_implicit_gemm_conv_to_cktile,
    lower_spec_to_cktile,
    lower_universal_gemm_to_cktile,
)
from .lower_hip import lower_kernel_to_hip
from .lower_llvm import lower_kernel_to_llvm
from .passes import (
    PassStats,
    canonicalize_region,
    eliminate_dead_pure_ops,
    optimize_kernel,
)
from .verify import Diagnostic, verify, verify_or_raise

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
    "BackendError",
    "BackendMismatch",
    "lower_universal_gemm",
    "resolve_backend",
    "print_ir",
    "serialize",
    "parse",
    "canonicalize",
    "canonical_equal",
    "Diagnostic",
    "verify",
    "verify_or_raise",
    "lower_implicit_gemm_conv_to_cktile",
    "lower_kernel_to_hip",
    "lower_kernel_to_llvm",
    "lower_spec_to_cktile",
    "lower_universal_gemm_to_cktile",
    "PassStats",
    "canonicalize_region",
    "eliminate_dead_pure_ops",
    "optimize_kernel",
]
