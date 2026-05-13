# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Composable Kernel Tile DSL — Python authoring surface.

`ck_dsl` is a Python DSL for writing CK Tile-style GPU kernels that
lower in-process to AMDGPU LLVM IR and then to HSACO via libamd_comgr.
No `hipcc`, no template metaprogramming, no MLIR round-trip.

The package is organised as four layers:

  - `ck_dsl.core`      — SSA IR, lowering passes (LLVM IR / HIP C++),
                         and textual IR printer.
  - `ck_dsl.runtime`   — in-process HSACO build + HIP module load +
                         kernel launch (libamd_comgr + libamdhip64 via
                         ctypes).
  - `ck_dsl.helpers`   — high-level kernel-authoring helpers:
                         `MfmaAtom`, `WarpGrid`, `CoalescedTileLoader`,
                         `AsyncTileLoader`, `DirectEpilogue`,
                         `CShuffleEpilogue`, `LdsLayout`,
                         `SchedulePolicy`, `SoftwarePipeline`,
                         `compile_kernel`,
                         `make_{gemm,conv}_manifest`.
  - `ck_dsl.analysis`  — reusable LLVM IR / HSACO instruction and
                         resource inspection.
  - `ck_dsl.benchmark` — repeated-run benchmark summaries with
                         median/spread reporting.
  - `ck_dsl.instances` — parametric kernel builders:
                         `build_universal_gemm`,
                         `build_implicit_gemm_conv`,
                         `build_direct_conv_{16c,4c}`.

For everyday use:

    from ck_dsl import (
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
from .core.ir import (
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
from .core.ir_print import print_ir
from .core.lower_hip import lower_kernel_to_hip
from .core.lower_llvm import lower_kernel_to_llvm
from .core.passes import PassStats, optimize_kernel

# ---- runtime ----
from .runtime.comgr import ComgrError, ComgrTimings, build_hsaco_from_llvm_ir
from .runtime.hip_module import HipError, Runtime

# ---- helpers ----
from .helpers import (
    MFMA_F16_ATOMS,
    AsyncTileLoader,
    Attention2DConfig,
    Attention3DConfig,
    AsyncTileLoaderSlot,
    CoalescedTileLoader,
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
    compile_kernel,
    conv_args_signature,
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
from .transforms import (
    CoordVar,
    TensorDescriptor,
    embed,
    merge,
    pad,
    pass_through,
    unmerge,
)

__all__ = [
    # core
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
    "optimize_kernel",
    # runtime
    "ComgrError",
    "ComgrTimings",
    "HipError",
    "Runtime",
    "build_hsaco_from_llvm_ir",
    # helpers
    "MFMA_F16_ATOMS",
    "Attention2DConfig",
    "Attention3DConfig",
    "AsyncTileLoader",
    "AsyncTileLoaderSlot",
    "CoalescedTileLoader",
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
    "compile_kernel",
    "attention_args_signature",
    "conv_args_signature",
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
    "TensorDescriptor",
    "embed",
    "merge",
    "pad",
    "pass_through",
    "unmerge",
]
