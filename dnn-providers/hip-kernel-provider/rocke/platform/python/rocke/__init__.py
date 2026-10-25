# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Composable Kernel Tile DSL — Python authoring surface.

`rocke` is a Python DSL for writing CK Tile-style GPU kernels that
lower to AMDGPU LLVM IR and then to HSACO through a selectable code-object
compiler. No template metaprogramming or MLIR round-trip is required.

The package is organised as four layers:

  - `rocke.core`      — SSA IR, lowering passes (LLVM IR / HIP C++),
                         and textual IR printer.
  - `rocke.runtime`   — in-process HSACO build + HIP module load +
                         kernel launch (libamd_comgr + libamdhip64 via
                         ctypes).
  - `rocke.helpers`   — high-level kernel-authoring helpers:
                         `MfmaAtom`, `WarpGrid`, `CoalescedTileLoader`,
                         `AsyncTileLoader`, `DirectEpilogue`,
                         `CShuffleEpilogue`, `LdsLayout`,
                         `SchedulePolicy`, `SoftwarePipeline`,
                         `compile_kernel`,
                         `make_{gemm,conv}_manifest`.
  - `rocke.analysis`  — reusable LLVM IR / HSACO instruction and
                         resource inspection.
  - `rocke.benchmark` — repeated-run benchmark summaries with
                         median/spread reporting.
  - `rocke.instances` — parametric kernel builders:
                         `build_universal_gemm`,
                         `build_implicit_gemm_conv`,
                         `build_direct_conv_{16c,4c}`.

For everyday use:

    from rocke import (
        # IR + types
        IRBuilder, F16, F32, I32,
        # helpers
        compile_kernel, MfmaAtom, mfma_atom, WarpGrid,
        CoalescedTileLoader, AsyncTileLoader,
        DirectEpilogue, CShuffleEpilogue,
        make_gemm_manifest, make_conv_manifest, write_artifact,
        # transforms (CK Tile coord-transform DAG)
        TensorDescriptor, unmerge, embed, pad, merge, pass_through,
    )

The submodule paths remain the source of truth; the top-level
re-exports listed below are a convenience.
"""

from __future__ import annotations

# ---- core ----
from .core.codegen_policy import (
    CodegenPolicy,
    SchedulerStrategy,
    apply_codegen_policy,
    codegen_policy_for_kernel,
)
from .core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    F64,
    FP8E4M3,
    I1,
    I4,
    U4,
    U8,
    I8,
    I16,
    I32,
    I64,
    TF32,
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
from .core.ir_print import print_ir
from .core.lower_cktile import (
    lower_implicit_gemm_conv_to_cktile,
    lower_spec_to_cktile,
    lower_universal_gemm_to_cktile,
)
from .core.lower_hip import lower_kernel_to_hip
from .core.lower_llvm import lower_kernel_to_llvm
from .core.passes import PassStats, optimize_kernel

# ---- runtime ----
from .runtime.code_object import (
    CodeObjectCompilerInfo,
    CodeObjectError,
    CodeObjectProcessDied,
    CodeObjectTimings,
    build_hsaco_from_llvm_ir,
    compiler_identity,
    resolve_code_object_backend,
)
from .runtime.comgr import ComgrError, ComgrProcessDied, ComgrTimings
from .runtime.hip_module import HipError, Runtime
from .runtime.launcher import StreamConfig, launch_kernel, make_kernel

# ---- helpers ----
from .helpers import (
    MFMA_F16_ATOMS,
    AsyncTileLoader,
    Attention2DConfig,
    Attention3DConfig,
    AsyncTileLoaderSlot,
    CoalescedTileLoader,
    CodeObjectInput,
    CShuffleEpilogue,
    DirectEpilogue,
    KernelArtifact,
    LdsLayout,
    MfmaAtom,
    OnlineSoftmaxState,
    PagedKvDescriptor,
    apply_softcap_scalar,
    causal_mask,
    SchedulePolicy,
    SoftwarePipeline,
    WarpGrid,
    attention_args_signature,
    compile_gfx1151_wmma_gemm_direct,
    compile_kernel,
    lower_kernel_for_code_object,
    gemm_args_signature,
    make_conv_manifest,
    make_attention_manifest,
    make_gemm_manifest,
    mfma_atom,
    select_2d_config,
    select_3d_config,
    sliding_window_mask,
    use_2d_kernel,
    write_artifact,
)

# ---- analysis / benchmark ----
from .analysis import (
    HsacoAnalysis,
    IsaStats,
    LlvmIrStats,
    ResourceInfo,
    VariantReport,
    analyze_hsaco,
    analyze_llvm_ir,
    compare_variant_reports,
    parse_isa,
    parse_resources,
)
from .benchmark import BenchmarkSummary, benchmark_manifest, summarize_runs

# ---- transforms ----
from .helpers.transforms import (
    CoordVar,
    Indirect,
    PadDynamic,
    TensorDescriptor,
    embed,
    indirect,
    merge,
    pad,
    pad_dynamic,
    pass_through,
    unmerge,
)

__all__ = [
    # core
    "CodegenPolicy",
    "SchedulerStrategy",
    "apply_codegen_policy",
    "codegen_policy_for_kernel",
    "BF8E5M2",
    "BF16",
    "F16",
    "F32",
    "F64",
    "FP8E4M3",
    "I1",
    "I4",
    "U4",
    "U8",
    "I8",
    "I16",
    "I32",
    "I64",
    "TF32",
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
    "lower_implicit_gemm_conv_to_cktile",
    "lower_kernel_to_hip",
    "lower_kernel_to_llvm",
    "lower_spec_to_cktile",
    "lower_universal_gemm_to_cktile",
    "PassStats",
    "optimize_kernel",
    # runtime
    "ComgrError",
    "ComgrProcessDied",
    "ComgrTimings",
    "CodeObjectCompilerInfo",
    "CodeObjectError",
    "CodeObjectProcessDied",
    "CodeObjectTimings",
    "HipError",
    "Runtime",
    "StreamConfig",
    "build_hsaco_from_llvm_ir",
    "compiler_identity",
    "launch_kernel",
    "make_kernel",
    "resolve_code_object_backend",
    # helpers
    "MFMA_F16_ATOMS",
    "Attention2DConfig",
    "Attention3DConfig",
    "AsyncTileLoader",
    "AsyncTileLoaderSlot",
    "CoalescedTileLoader",
    "CodeObjectInput",
    "CShuffleEpilogue",
    "DirectEpilogue",
    "KernelArtifact",
    "LdsLayout",
    "MfmaAtom",
    "OnlineSoftmaxState",
    "PagedKvDescriptor",
    "SchedulePolicy",
    "SoftwarePipeline",
    "WarpGrid",
    "compile_gfx1151_wmma_gemm_direct",
    "compile_kernel",
    "lower_kernel_for_code_object",
    "attention_args_signature",
    "gemm_args_signature",
    "make_attention_manifest",
    "make_conv_manifest",
    "make_gemm_manifest",
    "mfma_atom",
    "apply_softcap_scalar",
    "causal_mask",
    "select_2d_config",
    "select_3d_config",
    "sliding_window_mask",
    "use_2d_kernel",
    "write_artifact",
    # analysis / benchmark
    "BenchmarkSummary",
    "HsacoAnalysis",
    "IsaStats",
    "LlvmIrStats",
    "ResourceInfo",
    "VariantReport",
    "analyze_hsaco",
    "analyze_llvm_ir",
    "compare_variant_reports",
    "benchmark_manifest",
    "parse_isa",
    "parse_resources",
    "summarize_runs",
    # transforms
    "CoordVar",
    "Indirect",
    "PadDynamic",
    "TensorDescriptor",
    "embed",
    "indirect",
    "merge",
    "pad",
    "pad_dynamic",
    "pass_through",
    "unmerge",
]
