# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verify every claim made by ck_dsl/dsl_docs against the real package.

Imports are deliberately interleaved with the verification sections so that
each block reads as a self-contained "what does this subpackage export"
spec; Ruff's ``E402`` (module-level imports must be at top) is therefore
disabled for this single file.
"""

# ruff: noqa: E402

from __future__ import annotations

import importlib
import sys
from typing import Any, Callable


fails: list[tuple[str, str]] = []
passes: list[str] = []


def check(name: str, fn: Callable[[], Any]) -> Any:
    try:
        v = fn()
        passes.append(name)
        return v
    except Exception as e:  # noqa: BLE001
        fails.append((name, f"{type(e).__name__}: {e}"))
        return None


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---- Eagerly warm torch's HIP context so ctypes Runtime sees the device ----
import torch

assert torch.cuda.is_available(), "no GPU; this verifier requires HIP/torch.cuda"
_warm = torch.zeros(1, device="cuda")
del _warm


section("imports")


def _imp(mod: str, names: list[str]) -> None:
    def go() -> None:
        m = importlib.import_module(mod)
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            raise AttributeError(f"{mod} missing: {missing}")

    check(f"import {mod} ({len(names)} symbols)", go)


_imp(
    "ck_dsl",
    [
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
        "lower_implicit_gemm_conv_to_cktile",
        "lower_kernel_to_hip",
        "lower_kernel_to_llvm",
        "lower_spec_to_cktile",
        "lower_universal_gemm_to_cktile",
        "PassStats",
        "optimize_kernel",
        "ComgrError",
        "ComgrTimings",
        "build_hsaco_from_llvm_ir",
        "HipError",
        "Runtime",
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
        "make_conv_manifest",
        "make_attention_manifest",
        "make_gemm_manifest",
        "mfma_atom",
        "apply_softcap_scalar",
        "causal_mask",
        "sliding_window_mask",
        "select_2d_config",
        "select_3d_config",
        "use_2d_kernel",
        "write_artifact",
        "BenchmarkSummary",
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
        "benchmark_manifest",
        "summarize_runs",
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
    ],
)

_imp(
    "ck_dsl.helpers",
    [
        "TensorDescriptor",
        "TensorView",
        "TileWindow",
        "TensorCoordinate",
        "BufferResource",
        "make_global_view",
        "make_lds_view",
        "make_buffer_resource",
        "make_buffer_view",
        "make_naive_tensor_descriptor_packed",
        "make_naive_tensor_view_packed",
        "make_tile_window",
        "make_tensor_coordinate",
        "move_tensor_coordinate",
        "view_from_transforms_descriptor",
        "TileDistributionEncoding",
        "TileDistribution",
        "StaticDistributedTensor",
        "LoadStoreTraits",
        "load_tile",
        "store_tile",
        "make_static_tile_distribution",
        "make_load_store_traits",
        "make_static_distributed_tensor",
        "sweep_row_chunks",
        "pass2_row_chunks",
        "RowChunkSweepResult",
        "io_ir_type",
        "load_scalar",
        "load_scalar_as_f32",
        "load_vec",
        "load_vec_as_f32",
        "pack_f32_to",
        "store_scalar",
        "store_scalar_from_f32",
        "store_vec",
        "ReduceCombine",
        "block_lds_reduce",
        "IOSpecRule",
        "validate_io",
        "SignatureBuilder",
        "kernel_name_join",
        "ceil_div_grid",
        "ptr_type_str",
        "sig_param",
        "sig_scalar",
        "MANIFEST_SCHEMA",
        "make_simple_op_manifest",
        "NUM_XCDS_MI300X",
        "NUM_XCDS_MI325X",
        "NUM_XCDS_MI350X",
        "SuperTileSwizzleResult",
        "chiplet_aware_super_tile",
        "chiplet_aware_super_tile_dynamic",
        "chiplet_transform_chunked",
        "chiplet_transform_chunked_dynamic",
        "super_tile_swizzle",
        "super_tile_swizzle_dynamic",
        "Autotuner",
        "AutotuneConfig",
        "AutotuneKey",
        "AutotuneResult",
        "autotune_sweep",
        "make_autotune_key",
        "spec_replace",
        "compile_fn",
        "explain_fn",
        "fuse_matmul_bias_relu",
        "EpilogueOp",
        "BiasAdd",
        "Cast",
        "Clamp",
        "GELU",
        "ReLU",
        "ResidualAdd",
        "ResidualMul",
        "Scale",
        "SiLU",
        "FusedEpilogue",
        "FusionMatchError",
        "FusionPlan",
        "dtype_to_ir",
        "ir_dtype_zero",
        "ir_dtype_const",
        "ir_dtype_global_load",
        "FusionGraph",
        "FusionOp",
        "FusionRegion",
        "FusionTensor",
        "build_graph",
        "FusionLegalizer",
        "LegalResult",
        "BuiltRegion",
        "ElementwiseLowerer",
        "ExplainOnlyLowerer",
        "GemmEpilogueLowerer",
        "LoweringRegistry",
        "ReductionLowerer",
        "default_lowering_registry",
        "WorkspaceAllocation",
        "WorkspacePlanner",
        "materialize_plan",
        "GreedyFusionScheduler",
        "RegionCost",
        "BackendTiming",
        "BenchmarkCase",
        "FusionMatrixRunner",
        "ValidationReport",
        "run_fusion_validation_matrix",
        "LdsLayout",
        "TransposeLdsReader",
        "QDType",
        "QUANT_MAX_ABS",
        "dequantize_scalar_to_f32",
        "ir_to_qdtype",
        "quant_ir_type",
        "quant_max_abs",
        "quantize_scalar_f32",
    ],
)

_imp(
    "ck_dsl.runtime.launcher",
    [
        "DeviceMem",
        "KernelLauncher",
        "LaunchConfig",
        "LaunchSummary",
        "PipelineLauncher",
        "WorkspaceSpec",
        "WorkspacePool",
        "no_fence",
        "release_retained_for_stream",
        "synchronize_and_release",
        "time_launches",
        "wait_stream_and_release",
    ],
)

_imp(
    "ck_dsl.instances",
    [
        "TileSpec",
        "TraitSpec",
        "DataSpec",
        "UniversalGemmSpec",
        "build_universal_gemm",
        "BatchedGemmSpec",
        "build_batched_gemm",
        "GroupedGemmProblem",
        "GroupedGemmSpec",
        "build_grouped_gemm",
        "grouped_gemm_problems",
        "ConvProblem",
        "ImplicitGemmConvSpec",
        "build_implicit_gemm_conv",
        "DirectConvProblem",
        "DirectConv16cSpec",
        "build_direct_conv_16c",
        "DirectConv4cSpec",
        "build_direct_conv_4c",
        "ElementwiseSpec",
        "build_elementwise",
        "Reduce2DSpec",
        "build_reduce2d",
        "LayerNorm2DSpec",
        "build_layernorm2d",
        "RMSNorm2DSpec",
        "build_rmsnorm2d",
        "Transpose2DSpec",
        "build_transpose2d",
        "UnifiedAttentionProblem",
        "UnifiedAttention2DSpec",
        "build_unified_attention_2d",
        "UnifiedAttention3DSpec",
        "build_unified_attention_3d",
        "UnifiedAttentionReduceSpec",
        "build_unified_attention_reduce",
        "run_unified_attention_torch",
        "supports_native_unified_attention",
        "attention_3d_workspace_nbytes",
    ],
)


# =============================================================================
# Phase 2: IR construction + LLVM lowering of every documented op
# =============================================================================

section("IR builder methods")

from ck_dsl.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    PtrType,
    CACHE_ALL,
    CACHE_GLOBAL,
    CACHE_STREAM,
    NON_TEMPORAL,
)
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm
from ck_dsl.core.lower_hip import lower_kernel_to_hip
from ck_dsl.core.passes import optimize_kernel
from ck_dsl.core.ir_print import print_ir


def builder(name: str = "u") -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    return b


check(
    "cache hint constants exist",
    lambda: (CACHE_ALL, CACHE_GLOBAL, CACHE_STREAM, NON_TEMPORAL),
)


def t_arith() -> str:
    b = builder("t_arith")
    a = b.const_i32(1)
    c = b.const_i32(2)
    f = b.const_f32(1.0)
    g = b.const_f32(2.0)
    b.add(a, c)
    b.sub(a, c)
    b.mul(a, c)
    b.div(a, c)
    b.mod(a, c)
    b.fadd(f, g)
    b.fsub(f, g)
    b.fmul(f, g)
    b.fdiv(f, g)
    b.fneg(f)
    b.fmax(f, g)
    b.fmin(f, g)
    b.cmp_lt(a, c)
    b.cmp_le(a, c)
    b.cmp_gt(a, c)
    b.cmp_ge(a, c)
    b.cmp_eq(a, c)
    b.cmp_ne(a, c)
    b.fcmp("olt", f, g)
    b.fcmp("ogt", f, g)
    b.land(a, c)
    b.lor(a, c)
    b.lnot(a)
    b.zext(a, I64)
    b.sext(a, I64)
    b.select(b.cmp_lt(a, c), a, c)
    b.exp2(f)
    b.sqrt(f)
    b.rsqrt(f)
    b.tanh(f)
    b.rcp(f)
    b.clamp_f32(f, b.const_f32(-1.0), b.const_f32(1.0))
    h = b.cast_f32_to(f, F16)
    b.cast_to_f32(h)
    b.trunc_f32_to_f16(f)
    b.sitofp_f32(a)
    return lower_kernel_to_llvm(b.kernel)


check("arith/math/cast methods lower", t_arith)


def t_fp_conv() -> None:
    b = builder("t_fpconv")
    f = b.const_f32(1.5)
    fp8 = b.cvt_f32_to_fp8(f)
    bf8 = b.cvt_f32_to_bf8(f)
    b.cvt_f32_to_i8_sat(f)
    b.cvt_fp8_to_f32(fp8)
    b.cvt_bf8_to_f32(bf8)
    lower_kernel_to_llvm(b.kernel)


check("fp8/bf8/i8 conversion methods lower", t_fp_conv)


def t_wave_ops() -> None:
    b = builder("t_wave")
    p = b.cmp_eq(b.lane_id(), b.const_i32(0))
    b.wave_all(p)
    b.wave_any(p)
    b.wave_ballot(p)
    v = b.const_i32(7)
    rfl = b.readfirstlane(v)
    b.pin_sgpr(rfl)
    b.to_sgpr_u32(v)
    b.ds_bpermute(b.const_i32(0), v)
    f = b.const_f32(1.0)
    b.warp_shuffle_xor(f, 1)
    lower_kernel_to_llvm(b.kernel)


check("wave/cross-lane methods lower", t_wave_ops)


def t_smem() -> None:
    b = builder("t_smem")
    smem = b.smem_alloc(F16, (16, 16))
    base = b.smem_addr_of(smem)
    b.smem_ptr_add(base, b.zext(b.const_i32(32), I64))
    b.smem_store_f16(smem, [b.const_i32(0), b.const_i32(0)], b.fp16_zero())
    b.smem_load_v4_f16(smem, b.const_i32(0), b.const_i32(0))
    z8 = b.zero_vec(F16, 8)
    b.smem_store_vN_f16(smem, [b.const_i32(0), b.const_i32(0)], z8, 8)
    b.smem_load_vN_f16(smem, b.const_i32(0), b.const_i32(0), n=8)
    lower_kernel_to_llvm(b.kernel)


check("LDS scalar+vector methods lower", t_smem)


def t_ds_read_tr16() -> None:
    b = builder("t_tr16")
    smem = b.smem_alloc(F16, (16, 16))
    b.ds_read_tr16_b64(smem, b.const_i32(0), b.const_i32(0))
    lower_kernel_to_llvm(b.kernel)


check("ds_read_tr16_b64 lowers", t_ds_read_tr16)


def t_buffer_rsrc() -> str:
    b = builder("t_rsrc")
    A = b.param("A", PtrType(F16, "global"))
    nb = b.const_i32(1024)
    rsrc = b.buffer_rsrc(A, nb)
    zero = b.const_i32(0)
    b.buffer_load_f16(rsrc, zero, zero)
    b.buffer_load_vN_f16(rsrc, zero, zero, 2)
    b.buffer_load_vN_f16(rsrc, zero, zero, 4)
    b.buffer_store_f16(rsrc, zero, zero, b.fp16_zero())
    z8 = b.zero_vec(F16, 8)
    b.buffer_store_vN_f16(rsrc, zero, zero, z8, 4)
    return lower_kernel_to_llvm(b.kernel)


ll = check("buffer_rsrc + loads/stores lower", t_buffer_rsrc)
if isinstance(ll, str):
    if "i32 159744" in ll:
        passes.append("buffer_rsrc DW3 flag = 0x00027000 (i32 159744) emitted")
    else:
        fails.append(("buffer_rsrc DW3 flag", "expected i32 159744 in LLVM"))


def t_async_lds() -> None:
    b = builder("t_async")
    A = b.param("A", PtrType(F16, "global"))
    rsrc = b.buffer_rsrc(A, b.const_i32(1024))
    smem = b.smem_alloc(F16, (16, 64))
    base = b.smem_addr_of(smem)
    zero = b.const_i32(0)
    b.async_buffer_load_lds(rsrc, smem, zero, zero, dwords=4, coherency=CACHE_STREAM)
    b.async_buffer_load_lds_addr(rsrc, base, zero, zero, 4, coherency=CACHE_ALL)
    b.s_waitcnt(vmcnt=0)
    b.sync()
    lower_kernel_to_llvm(b.kernel)


check("async_buffer_load_lds + waitcnt + sync lower", t_async_lds)


def t_mfma() -> None:
    b = builder("t_mfma")
    a4h = b.zero_vec(F16, 4)
    a8h = b.zero_vec(F16, 8)
    c4f = b.zero_vec(F32, 4)
    c16f = b.zero_vec(F32, 16)
    b.mfma_f32_16x16x16_f16(a4h, a4h, c4f)
    b.mfma_f32_16x16x32_f16(a8h, a8h, c4f)
    b.mfma_f32_32x32x8_f16(a4h, a4h, c16f)
    b.mfma_f32_32x32x16_f16(a8h, a8h, c16f)
    b.mfma_f32_4x4x4_f16(a4h, a4h, c4f)
    a4bf = b.zero_vec(BF16, 4)
    a8bf = b.zero_vec(BF16, 8)
    b.mfma_f32_16x16x16_bf16(a4bf, a4bf, c4f)
    b.mfma_f32_16x16x32_bf16(a8bf, a8bf, c4f)
    lower_kernel_to_llvm(b.kernel)


check("every MFMA atom in docs lowers", t_mfma)


def t_waitcnt_encoding() -> None:
    b = builder("t_wait")
    b.s_waitcnt(vmcnt=16, lgkmcnt=16)
    ll = lower_kernel_to_llvm(b.kernel)
    if "call void @llvm.amdgcn.s.waitcnt(i32 20336)" not in ll:
        raise AssertionError("vmcnt=16 lgkmcnt=16 encoding missing 20336")


check("s_waitcnt extended encoding -> i32 20336", t_waitcnt_encoding)


def t_scf_for_iter() -> None:
    b = builder("t_scffor")
    init = b.zero_vec_f32(4)
    loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(8),
        b.const_i32(1),
        iter_args=[("acc", init)],
    )
    with loop as (iv, iter_vars):
        acc = iter_vars[0]
        new_acc = b.fadd(acc, b.zero_vec_f32(4))
        b.scf_yield(new_acc)
    lower_kernel_to_llvm(b.kernel)


check("scf_for_iter lowers (with iter_args)", t_scf_for_iter)


def t_scf_if() -> None:
    b = builder("t_scfif")
    v = b.const_i32(1)
    with b.scf_if(b.cmp_eq(v, b.const_i32(1))):
        b.const_i32(2)
    lower_kernel_to_llvm(b.kernel)


check("scf_if lowers", t_scf_if)


def t_passes() -> None:
    b = builder("t_pass")
    a = b.const_i32(2)
    c = b.const_i32(3)
    b.add(a, c)
    b.add(a, c)
    stats = optimize_kernel(b.kernel)
    assert stats.common_subexpressions >= 1


check("optimize_kernel CSE", t_passes)


# =============================================================================
# Specs -> CK Tile + HIP
# =============================================================================

section("specs build + lower (LLVM / HIP / CK Tile)")

from ck_dsl.instances import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    DirectConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    build_direct_conv_16c,
    build_direct_conv_4c,
    ElementwiseSpec,
    build_elementwise,
    Reduce2DSpec,
    build_reduce2d,
    LayerNorm2DSpec,
    build_layernorm2d,
    RMSNorm2DSpec,
    build_rmsnorm2d,
    Transpose2DSpec,
    build_transpose2d,
    BatchedGemmSpec,
    build_batched_gemm,
    GroupedGemmSpec,
    build_grouped_gemm,
    UnifiedAttentionProblem,
    UnifiedAttention2DSpec,
    UnifiedAttention3DSpec,
    UnifiedAttentionReduceSpec,
    build_unified_attention_2d,
    build_unified_attention_3d,
    build_unified_attention_reduce,
)
from ck_dsl.core.lower_cktile import (
    lower_spec_to_cktile,
    lower_universal_gemm_to_cktile,
    lower_implicit_gemm_conv_to_cktile,
)


def t_gemm_spec() -> dict:
    spec = UniversalGemmSpec(
        name="vgemm",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
    )
    kernel = build_universal_gemm(spec)
    return {
        "ir": print_ir(kernel),
        "llvm": lower_kernel_to_llvm(kernel),
        "hip": lower_kernel_to_hip(kernel),
        "cktile_via_spec": lower_spec_to_cktile(spec),
        "cktile_direct": lower_universal_gemm_to_cktile(spec),
    }


gemm_out = check(
    "UniversalGemmSpec: build + print_ir + LLVM + HIP + CK Tile", t_gemm_spec
)


def t_conv_spec() -> None:
    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        ),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
    )
    kernel = build_implicit_gemm_conv(spec)
    print_ir(kernel)
    lower_kernel_to_llvm(kernel)
    lower_kernel_to_hip(kernel)
    lower_spec_to_cktile(spec)
    lower_implicit_gemm_conv_to_cktile(spec)


check("ImplicitGemmConvSpec: build + print_ir + LLVM + HIP + CK Tile", t_conv_spec)


def t_direct_conv() -> None:
    # DirectConvProblem(N, H, W, groups, cpg, kpg, KH=3, KW=3, PAD=1, stride=1)
    # The 16c spec's default BLOCK_GROUPS=8; use groups=16 so divisibility holds.
    p16 = DirectConvProblem(
        N=8, H=56, W=56, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
    )
    s16 = DirectConv16cSpec(problem=p16)
    lower_kernel_to_llvm(build_direct_conv_16c(s16))
    p4 = DirectConvProblem(
        N=8, H=56, W=56, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
    )
    s4 = DirectConv4cSpec(problem=p4)
    lower_kernel_to_llvm(build_direct_conv_4c(s4))


check("DirectConv16cSpec + DirectConv4cSpec: build + LLVM", t_direct_conv)


def t_smallops() -> None:
    cases = [
        (ElementwiseSpec(op="add", dtype="f16"), build_elementwise),
        (
            Reduce2DSpec(
                n_per_block=4096, block_size=256, vec=8, dtype="f16", op="sum"
            ),
            build_reduce2d,
        ),
        (
            LayerNorm2DSpec(n_per_block=4096, block_size=256, vec=8, dtype="f16"),
            build_layernorm2d,
        ),
        (
            RMSNorm2DSpec(n_per_block=4096, block_size=256, vec=8, dtype="f16"),
            build_rmsnorm2d,
        ),
        (Transpose2DSpec(tile_m=32, tile_n=32, dtype="f16"), build_transpose2d),
    ]
    for spec, builder_fn in cases:
        lower_kernel_to_llvm(builder_fn(spec))


check("small ops: build + LLVM", t_smallops)


def t_batched_grouped() -> None:
    s = BatchedGemmSpec(
        name="bgemm",
        tile=TileSpec(
            64,
            64,
            32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
    )
    lower_kernel_to_llvm(build_batched_gemm(s))
    g = GroupedGemmSpec(
        name="grpgemm",
        tile=TileSpec(
            64,
            64,
            32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
    )
    lower_kernel_to_llvm(build_grouped_gemm(g))


check("BatchedGemmSpec + GroupedGemmSpec: build + LLVM", t_batched_grouped)


def t_attention() -> None:
    p = UnifiedAttentionProblem(
        total_q=1,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=1024,
        dtype="fp16",
    )
    lower_kernel_to_llvm(build_unified_attention_2d(UnifiedAttention2DSpec(problem=p)))
    lower_kernel_to_llvm(build_unified_attention_3d(UnifiedAttention3DSpec(problem=p)))
    lower_kernel_to_llvm(
        build_unified_attention_reduce(
            UnifiedAttentionReduceSpec(problem=p, num_segments=8)
        )
    )


check("UnifiedAttention 2D + 3D + Reduce: build + LLVM", t_attention)


# =============================================================================
# helpers/quant.py — QDType is a Literal, helpers are string-keyed
# =============================================================================

section("quantization")

from ck_dsl.helpers import (
    QUANT_MAX_ABS,
    quant_ir_type,
    quant_max_abs,
    ir_to_qdtype,
    quantize_scalar_f32,
    dequantize_scalar_to_f32,
)


def t_quant() -> None:
    # The QDType "values" are plain strings.
    assert quant_max_abs("fp8e4m3") == 448.0
    assert quant_max_abs("bf8e5m2") == 57344.0
    assert quant_max_abs("i8") == 127.0
    assert QUANT_MAX_ABS["fp8e4m3"] == 448.0
    assert quant_ir_type("fp8e4m3").name == "fp8e4m3"
    # round-trip via IR. quantize_scalar_f32 expects keyword inv_scale + qdtype.
    b = builder("t_quant")
    f = b.const_f32(0.5)
    inv_scale = b.const_f32(1.0)
    scale = b.const_f32(1.0)
    q = quantize_scalar_f32(b, f, inv_scale=inv_scale, qdtype="fp8e4m3")
    dequantize_scalar_to_f32(b, q, scale=scale)
    lower_kernel_to_llvm(b.kernel)
    assert ir_to_qdtype(quant_ir_type("fp8e4m3")) == "fp8e4m3"


check("QDType/quant_*/quantize round-trip", t_quant)


# =============================================================================
# Chiplet swizzle (real helpers/grid.py)
# =============================================================================

section("chiplet")

from ck_dsl.helpers import (
    NUM_XCDS_MI300X,
    NUM_XCDS_MI325X,
    NUM_XCDS_MI350X,
    chiplet_transform_chunked,
)


def t_chiplet() -> None:
    assert NUM_XCDS_MI300X == 8 == NUM_XCDS_MI325X == NUM_XCDS_MI350X
    b = builder("t_chiplet")
    wgid = b.block_id_x()
    chiplet_transform_chunked(b, wgid, num_wgs=512, num_xcds=8, chunk_size=64)
    lower_kernel_to_llvm(b.kernel)


check("chiplet helpers emit IR", t_chiplet)


# =============================================================================
# Transform DAG
# =============================================================================

section("transform DAG")

from ck_dsl.transforms import (
    TensorDescriptor,
    unmerge,
    embed,
    pad,
    pad_dynamic,
    indirect,
    merge,
)


def t_xfm_conv() -> None:
    desc = TensorDescriptor.naive(
        "A_nhwc", lengths=[8, 58, 58, 64], dtype=F16, coord_names=["n", "hi", "wi", "c"]
    ).transform(
        unmerge("m", into=["n", "ho", "wo"], dims=[8, 56, 56]),
        embed(["ho", "r"], "hi", strides=[1, 1], offset=-1, lo=0, hi=58),
        embed(["wo", "s"], "wi", strides=[1, 1], offset=-1, lo=0, hi=58),
        unmerge("k", into=["r", "s", "c"], dims=[3, 3, 64]),
        pad("r", lo=0, hi=3),
        pad("s", lo=0, hi=3),
    )
    b = builder("t_xfm")
    m = b.param("m", I32)
    k = b.param("k", I32)
    off, valid = desc.offset(b, m=m, k=k)
    b.select(valid, off, b.const_i32(0))
    lower_kernel_to_llvm(b.kernel)


check("transform DAG: pad+embed+unmerge", t_xfm_conv)


def t_xfm_indirect_dynamic_merge() -> None:
    b = builder("t_xfm_ind")
    table = b.param("table", PtrType(I32, "global"))
    base = b.const_i32(0)
    desc = (
        TensorDescriptor.naive(
            "kv_bytes",
            lengths=[1 << 24, 16, 2, 128],
            dtype=F16,
            coord_names=("physical_block", "token", "kv_head", "dim"),
        )
        .transform(
            indirect("logical_block", into="physical_block", table=table, base=base)
        )
        .transform(pad_dynamic("token", lo=b.const_i32(0), hi=b.const_i32(2048)))
    )
    upper = b.param("upper", I32)
    tok = b.param("tok", I32)
    head = b.param("head", I32)
    dim = b.param("dim", I32)
    off, valid = desc.offset(b, logical_block=upper, token=tok, kv_head=head, dim=dim)
    b.select(valid, off, b.const_i32(0))
    lower_kernel_to_llvm(b.kernel)

    # merge flattens N upper coords into one linear lower coord; the
    # *user* still supplies the original uppers (a, b), and the merge
    # computes a flat lower coord behind the scenes. (This is the
    # inverse-direction relative to unmerge.) Verify the call shape.
    md = TensorDescriptor.naive(
        "X", lengths=[4, 8], dtype=F16, coord_names=["a", "b"]
    ).transform(merge(upper=["a", "b"], into="flat", dims=[4, 8]))
    assert "a" in md.upper_names and "b" in md.upper_names


check("transform DAG: indirect + pad_dynamic + merge", t_xfm_indirect_dynamic_merge)


# =============================================================================
# Distribution
# =============================================================================

section("distribution")

from ck_dsl.helpers import (
    TileDistributionEncoding,
    make_static_tile_distribution,
    make_load_store_traits,
    load_tile,
    make_naive_tensor_view_packed,
    make_tile_window,
    make_lds_view,
)


def t_distribution() -> None:
    enc = TileDistributionEncoding(
        Hs=((4, 256, 8),),
        Ps2RHs_major=((1,),),
        Ps2RHs_minor=((1,),),
        Ys2RHs_major=(1, 1),
        Ys2RHs_minor=(0, 2),
    )
    dist = make_static_tile_distribution(enc)
    traits = make_load_store_traits(dist)
    assert traits.scalar_per_vector == 8
    b = builder("t_dist")
    X = b.param("X", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    view = make_naive_tensor_view_packed(X, shape=(8192,), dtype=F16)
    tile = make_tile_window(view, lengths=(8192,), origin=(b.const_i32(0),))
    load_tile(b, tile, distribution=dist, ps=[[b.thread_id_x()]], traits=traits)
    lower_kernel_to_llvm(b.kernel)


check("TileDistribution: encoding + load_tile", t_distribution)


# =============================================================================
# Loaders + epilogues
# =============================================================================

section("loaders + epilogues")

from ck_dsl.helpers import (
    CoalescedTileLoader,
    AsyncTileLoader,
    DirectEpilogue,
    CShuffleEpilogue,
    WarpGrid,
    LdsLayout,
    TransposeLdsReader,
    SchedulePolicy,
    SoftwarePipeline,
    MfmaAtom,
    MFMA_F16_ATOMS,
    mfma_atom,
)


def t_loaders() -> None:
    v = CoalescedTileLoader.choose_vec(tile_rows=64, tile_cols=64, block_size=256)
    assert v in (1, 2, 4, 8)
    loader_c = CoalescedTileLoader.from_tile(tile_rows=64, tile_cols=64, block_size=256)
    assert loader_c.vecs_per_thread >= 1

    d = AsyncTileLoader.choose_dwords(tile_rows=64, tile_cols=64, block_size=256)
    assert d in (1, 3, 4)
    loader_a = AsyncTileLoader.from_tile(
        tile_rows=64, tile_cols=64, block_size=256, wave_size=64
    )
    assert loader_a.dwords in (1, 3, 4)

    b = builder("t_async_load")
    smem = b.smem_alloc(F16, (64, 64))
    A = b.param("A", PtrType(F16, "global"))
    rsrc = b.buffer_rsrc(A, b.const_i32(64 * 64 * 2))
    slot = loader_a.bind(b, smem_dst=smem, wave_id=b.const_i32(0))

    def desc(b, row, col):
        off = b.add(b.mul(row, b.const_i32(64)), col)
        valid = b.cmp_lt(off, b.const_i32(64 * 64))
        return off, valid

    slot.issue(b, tid=b.thread_id_x(), rsrc=rsrc, descriptor=desc)
    b.s_waitcnt(vmcnt=0)
    b.sync()
    lower_kernel_to_llvm(b.kernel)


check("CoalescedTileLoader + AsyncTileLoader.bind+issue", t_loaders)


def t_mfma_atom() -> None:
    a = mfma_atom("f16", 32, 32, 16)
    assert isinstance(a, MfmaAtom)
    assert len(MFMA_F16_ATOMS) == 5
    b = builder("t_atom")
    A = b.zero_vec(F16, 8)
    B = b.zero_vec(F16, 8)
    C = b.zero_vec_f32(16)
    a.emit(b, A, B, C)
    lower_kernel_to_llvm(b.kernel)


check("mfma_atom + MfmaAtom.emit", t_mfma_atom)


def t_layouts_pipeline_schedule() -> None:
    # LdsLayout has logical_cols + k_pad + swizzle + requires_packed_async
    L = LdsLayout.padded_k(logical_cols=32, k_pad=8)
    assert L.logical_cols == 32 and L.k_pad == 8
    L2 = LdsLayout.packed_async(logical_cols=32)
    assert L2.requires_packed_async
    # TransposeLdsReader(K, M=16)
    r = TransposeLdsReader(K=64)
    assert r.M == 16 and r.K == 64 and r.k_lanes == 16
    # SoftwarePipeline takes num_iters
    sp = SoftwarePipeline(num_iters=4)
    _ = sp
    # SchedulePolicy.for_pipeline
    p = SchedulePolicy.for_pipeline("compv4")
    _ = p


check(
    "LdsLayout + TransposeLdsReader + SoftwarePipeline + SchedulePolicy",
    t_layouts_pipeline_schedule,
)


def t_warpgrid_and_epilogues() -> None:
    b = builder("t_wg")
    b.kernel.attrs["max_workgroup_size"] = 256
    # WarpGrid is a frozen dataclass; instantiate with compile-time geometry,
    # then call .bind(b) to emit per-thread SSA and return a fully-populated grid.
    wg0 = WarpGrid(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=64,
    )
    wg = wg0.bind(b)
    assert wg.tile_m == 128 and wg.tid is not None
    atom = mfma_atom("f16", 32, 32, 16)
    de = DirectEpilogue(atom=atom, grid=wg)
    ce = CShuffleEpilogue.from_grid(atom=atom, grid=wg, max_store_vec=8)
    _ = (de, ce)


check(
    "WarpGrid.bind + DirectEpilogue + CShuffleEpilogue.from_grid",
    t_warpgrid_and_epilogues,
)


# =============================================================================
# io / spec / reduction / sweep
# =============================================================================

section("io / spec / reduction / sweep")

from ck_dsl.helpers import (
    io_ir_type,
    load_scalar,
    load_scalar_as_f32,
    load_vec,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar,
    store_scalar_from_f32,
    store_vec,
    block_lds_reduce,
    sweep_row_chunks,
    IOSpecRule,
    validate_io,
    SignatureBuilder,
    kernel_name_join,
    ceil_div_grid,
    ptr_type_str,
    sig_param,
    sig_scalar,
)


def t_io() -> None:
    assert io_ir_type("f16").name == "f16"
    assert io_ir_type("bf16").name == "bf16"
    b = builder("t_io")
    X = b.param("X", PtrType(F16, "global"))
    Y = b.param("Y", PtrType(F16, "global"))
    v = load_scalar(b, X, b.const_i32(0), dtype="f16")
    load_scalar_as_f32(b, X, b.const_i32(0), dtype="f16")
    vec = load_vec(b, X, b.const_i32(0), dtype="f16", n=8)
    load_vec_as_f32(
        b, X, b.const_i32(0), dtype="f16", n=8
    )  # returns list of f32 scalars
    # store_vec takes (ptr, idx, value, *, n) — value must already be in target dtype.
    store_vec(b, Y, b.const_i32(0), vec, n=8)
    store_scalar_from_f32(b, Y, b.const_i32(0), b.const_f32(1.0), dtype="f16")
    store_scalar(b, Y, b.const_i32(0), b.fp16_zero(), dtype="f16")
    _ = v
    lower_kernel_to_llvm(b.kernel)


check("io helpers: load/store/cast", t_io)


def t_pack_f32_to_list() -> None:
    # pack_f32_to documented signature: list of f32 scalars -> vector
    b = builder("t_pack")
    scalars = [b.const_f32(float(i)) for i in range(4)]
    out = pack_f32_to(b, scalars, dtype="f16")
    assert out.type.name.startswith("vec<f16x4>") or "f16x4" in out.type.name
    lower_kernel_to_llvm(b.kernel)


check("pack_f32_to(scalars, *, dtype) returns vec", t_pack_f32_to_list)


def t_block_lds_reduce() -> None:
    b = builder("t_blr")
    b.kernel.attrs["max_workgroup_size"] = 256
    lds = make_lds_view(b, dtype=F32, shape=(256,), name_hint="r").base
    tid = b.thread_id_x()
    v = b.const_f32(1.0)
    block_lds_reduce(b, v, lds, tid, block_size=256, combine="sum")
    block_lds_reduce(b, v, lds, tid, block_size=256, combine="max")
    lower_kernel_to_llvm(b.kernel)


check("block_lds_reduce(sum+max)", t_block_lds_reduce)


def t_spec_helpers() -> None:
    rule = IOSpecRule(dtype="f16", block_size=256, vec=8, n_per_block=4096)
    ok, why = validate_io(rule)
    assert ok, why
    sig = SignatureBuilder().ptr("A", "f16").scalar("M", "i32").build()
    assert sig == [
        {"name": "A", "type": "ptr<f16, global>"},
        {"name": "M", "type": "i32"},
    ]
    assert (
        kernel_name_join("op", "f16", "N4096", flags={"smv": True})
        == "op_f16_N4096_smv"
    )
    assert ptr_type_str("fp16") == "ptr<f16, global>"
    assert sig_param("X", "f16")["type"] == "ptr<f16, global>"
    assert sig_scalar("M", "i32")["type"] == "i32"
    # ceil_div_grid takes pairs: ((total, tile), ...)
    g = ceil_div_grid((1000, 7))
    assert g[0] >= 1 and g[1] == 1 and g[2] == 1


check("validate_io + SignatureBuilder + ceil_div_grid", t_spec_helpers)


def t_sweep() -> None:
    b = builder("t_swp")
    b.kernel.attrs["max_workgroup_size"] = 256
    X = b.param("X", PtrType(F16, "global"))
    tid = b.thread_id_x()
    # sweep_row_chunks expects a 2D TileWindow (rows x cols).
    view = make_naive_tensor_view_packed(X, shape=(16, 4096), dtype=F16)
    tile = make_tile_window(
        view, lengths=(1, 4096), origin=(b.const_i32(0), b.const_i32(0))
    )
    sweep_row_chunks(
        b,
        tile,
        tid=tid,
        block_size=256,
        vec=8,
        elems_per_thread=4096 // 256,
        row=b.const_i32(0),
    )
    lower_kernel_to_llvm(b.kernel)


check("sweep_row_chunks lowers", t_sweep)


# =============================================================================
# Manifest
# =============================================================================

section("manifest")

from ck_dsl.helpers import (
    MANIFEST_SCHEMA,
    attention_args_signature,
    conv_args_signature,
    gemm_args_signature,
    compile_kernel,
)


def t_manifest() -> None:
    assert MANIFEST_SCHEMA == "ck.dsl.example.manifest/v1"
    gs = gemm_args_signature()
    assert any(e["name"] == "M" for e in gs)
    gsb = gemm_args_signature(with_bytes=True)
    assert any(e["name"] == "A_bytes" for e in gsb)
    cs = conv_args_signature()
    assert {e["name"] for e in cs} >= {"A", "B", "D", "A_bytes", "B_bytes", "D_bytes"}
    a2 = attention_args_signature(path="2d")
    ar = attention_args_signature(path="reduce")
    assert len(a2) >= 6 and len(ar) >= 4


check("manifest signatures + schema constant", t_manifest)


# =============================================================================
# COMGR + analysis
# =============================================================================

section("comgr + analysis")

from ck_dsl import analyze_llvm_ir


def t_comgr() -> None:
    art = compile_kernel(build_elementwise(ElementwiseSpec(op="copy", dtype="f16")))
    assert art.hsaco_bytes > 0
    assert "amdgpu_kernel" in art.llvm_text
    for k in (
        "ir_build",
        "ir_lower_llvm",
        "comgr_bc",
        "comgr_relocatable",
        "comgr_executable",
        "total",
    ):
        assert k in art.timings


check("compile_kernel -> HSACO + timings keys", t_comgr)


def t_analyze() -> None:
    if gemm_out is None:
        raise RuntimeError("gemm_out missing")
    stats = analyze_llvm_ir(gemm_out["llvm"])
    d = stats.as_dict()
    assert isinstance(d, dict) and d


check("analyze_llvm_ir.as_dict", t_analyze)


# =============================================================================
# Fusion
# =============================================================================

section("fusion")

from ck_dsl.helpers import (
    BiasAdd,
    ReLU,
    FusedEpilogue,
    explain_fn,
    default_lowering_registry,
    LoweringRegistry,
    FusionLegalizer,
    GreedyFusionScheduler,
    dtype_to_ir,
)


def t_fusion() -> None:
    # BiasAdd(param_name, dtype) — verified signature
    chain = FusedEpilogue(ops=[BiasAdd(param_name="bias", dtype="fp16"), ReLU()])
    assert len(chain.ops) == 2
    assert dtype_to_ir("fp16").name == "f16"

    def fn(a, b, bias):
        return torch.nn.functional.relu(a @ b.t() + bias)

    plan = explain_fn(fn)
    assert plan is not None

    reg = default_lowering_registry()
    assert isinstance(reg, LoweringRegistry)
    FusionLegalizer()
    GreedyFusionScheduler()


check("Fusion: FusedEpilogue + explain_fn + registry + legalizer/scheduler", t_fusion)


# =============================================================================
# Autotuner
# =============================================================================

section("autotuner")

from ck_dsl.helpers import (
    AutotuneConfig,
    AutotuneResult,
    make_autotune_key,
    spec_replace,
)


def t_autotune() -> None:
    spec = UniversalGemmSpec(
        name="hero",
        tile=TileSpec(
            64,
            64,
            32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
    )
    cfg = AutotuneConfig(spec=spec, name="t64x64x32")
    assert cfg.spec is spec
    key = make_autotune_key(graph_hash="g0", shape=(4096, 4096, 4096), dtype="f16")
    assert isinstance(key, tuple)
    spec2 = spec_replace(spec, name="hero2")
    assert spec2.name == "hero2"
    res = AutotuneResult(config_name="x", ms_per_iter=0.01)
    assert res.is_ok


check("Autotuner objects constructible", t_autotune)


# =============================================================================
# Runtime launcher (GPU)
# =============================================================================

section("runtime launcher (GPU)")

from ck_dsl.runtime.launcher import (
    KernelLauncher,
    LaunchConfig,
    PipelineLauncher,
    WorkspaceSpec,
    WorkspacePool,
    DeviceMem,
    time_launches,
    no_fence,
    synchronize_and_release,
    wait_stream_and_release,
    release_retained_for_stream,
)
from ck_dsl.instances import (
    elementwise_signature,
    elementwise_grid,
)


def t_workspacepool() -> None:
    pool = WorkspacePool()
    t = pool.get("buf", shape=(64,), dtype=torch.float32, device="cuda")
    assert tuple(t.shape) == (64,)
    t2 = pool.get("buf", shape=(64,), dtype=torch.float32, device="cuda")
    assert t.data_ptr() == t2.data_ptr()
    t3 = pool.get("buf", shape=(32,), dtype=torch.float32, device="cuda")
    assert t3.data_ptr() == t.data_ptr()
    spec = WorkspaceSpec("buf", (64,), torch.float32, "cuda")
    t4 = pool.get_spec(spec)
    assert t4.data_ptr() == t.data_ptr()
    pool.prepare([spec])
    pool.clear()


check("WorkspacePool: get/get_spec/prepare/clear", t_workspacepool)


def t_devicemem() -> None:
    m = DeviceMem(1024)
    assert m.ptr() != 0 and m.nbytes() == 1024
    m.realloc(2048)
    assert m.nbytes() == 2048
    del m


check("DeviceMem alloc/realloc/free", t_devicemem)


def t_kernel_launcher() -> None:
    spec = ElementwiseSpec(op="copy", dtype="f16")
    art = compile_kernel(build_elementwise(spec))
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=elementwise_signature(spec),
    )
    N = 1024
    A = torch.arange(N, dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    cfg = LaunchConfig(grid=elementwise_grid(N, spec), block=(spec.block_size, 1, 1))
    launcher({"A": A, "C": C, "N": N}, config=cfg)
    assert torch.equal(A, C)


check("KernelLauncher.__call__ launches and matches torch", t_kernel_launcher)


def t_time_launches() -> None:
    spec = ElementwiseSpec(op="copy", dtype="f16")
    art = compile_kernel(build_elementwise(spec))
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=elementwise_signature(spec),
    )
    N = 4096
    A = torch.zeros(N, dtype=torch.float16, device="cuda")
    C = torch.empty_like(A)
    cfg = LaunchConfig(grid=elementwise_grid(N, spec), block=(spec.block_size, 1, 1))
    ms = time_launches(
        lambda: launcher({"A": A, "C": C, "N": N}, config=cfg), warmup=3, iters=10
    )
    assert ms > 0


check("time_launches > 0 ms", t_time_launches)


def t_pipeline_launcher() -> None:
    spec = ElementwiseSpec(op="copy", dtype="f16")
    art1 = compile_kernel(build_elementwise(spec))
    art2 = compile_kernel(build_elementwise(spec))
    s1 = KernelLauncher(
        hsaco=art1.hsaco,
        kernel_name=art1.kernel_name,
        signature=elementwise_signature(spec),
    )
    s2 = KernelLauncher(
        hsaco=art2.hsaco,
        kernel_name=art2.kernel_name,
        signature=elementwise_signature(spec),
    )
    pipeline = PipelineLauncher([s1, s2])
    N = 1024
    A = torch.arange(N, dtype=torch.float16, device="cuda")
    B = torch.empty_like(A)
    C = torch.empty_like(A)
    cfg = LaunchConfig(grid=elementwise_grid(N, spec), block=(spec.block_size, 1, 1))
    pipeline(
        values_per_stage=[{"A": A, "C": B, "N": N}, {"A": B, "C": C, "N": N}],
        configs_per_stage=[cfg, cfg],
        stream=0,
    )
    assert torch.equal(A, C)


check("PipelineLauncher chains two stages", t_pipeline_launcher)


def t_sync_helpers() -> None:
    with no_fence():
        pass
    synchronize_and_release()
    wait_stream_and_release(0)
    release_retained_for_stream(0)


check("no_fence + sync/release helpers callable", t_sync_helpers)


# =============================================================================
# Implicit-GEMM conv end-to-end (GPU)
# =============================================================================

section("implicit-GEMM conv end-to-end")


def t_implicit_conv_e2e() -> None:
    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        ),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
    )
    kernel = build_implicit_gemm_conv(spec)
    art = compile_kernel(kernel)
    assert art.hsaco_bytes > 0
    # Don't allocate / launch (the README/run_manifest path handles that).
    # The static build alone proves the IR + LLVM + comgr path.


check("implicit-GEMM conv: build + LLVM + comgr", t_implicit_conv_e2e)


# =============================================================================
# Report
# =============================================================================

print("\n" + "=" * 72)
print(f"PASS: {len(passes)}    FAIL: {len(fails)}")
if fails:
    print("\nFailures:")
    for n, r in fails:
        print(f"  FAIL  {n}")
        print(f"        {r}")
print("=" * 72)

sys.exit(0 if not fails else 1)
