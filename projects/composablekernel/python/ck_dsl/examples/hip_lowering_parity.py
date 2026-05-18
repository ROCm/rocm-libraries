# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Audit HIP debug lowering parity against production LLVM lowering.

This harness deliberately separates three levels of confidence:

1. Build a representative instance ``KernelDef``.
2. Require production LLVM lowering to succeed, then require HIP debug
   lowering to produce source for the same ``KernelDef``.
3. Optionally compile the generated HIP source to HSACO with ``hipcc
   --genco``. This catches source-level gaps that a string lowerer cannot
   see: missing prologue declarations, unsupported clang builtins, invalid
   vector types, and malformed C++.

The production path remains LLVM -> COMGR. HIP lowering is still a debug /
inspection / extraction path, so performance parity should only be claimed
after correctness and timing against the same input contract. The
``--bench-smoke`` mode currently performs that check for one cheap kernel
(``elementwise.add``) and is intended as the seed for broader per-instance
runtime parity.

Usage::

    python -m ck_dsl.examples.hip_lowering_parity
    python -m ck_dsl.examples.hip_lowering_parity --compile-hip --case gemm
    PYTHONPATH=python python \\
        python/ck_dsl/examples/hip_lowering_parity.py --bench-smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from ck_dsl.core.lower_hip import lower_kernel_to_hip  # noqa: E402
from ck_dsl.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from ck_dsl.helpers import QkScaleSpec  # noqa: E402
from ck_dsl.instances import (  # noqa: E402
    AddRmsnorm2DRdquantSpec,
    BatchedContractionSpec,
    BatchedGemmSpec,
    BatchedTranspose2DSpec,
    BlockScaleGemmSpec,
    ConvProblem,
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConvProblem,
    ElementwiseSpec,
    FlatMMSpec,
    FmhaAppendKvSpec,
    FmhaBwdSpec,
    FmhaCommonSpec,
    FmhaFwdFp8Spec,
    FmhaFwdHeadGroupingSpec,
    FmhaFwdPagedPrefillSpec,
    FmhaFwdSplitKvDecodeSpec,
    FmhaFwdVarlenSpec,
    FmhaShape,
    FusedMoeSpec,
    GemmMultiAbdSpec,
    GemmMultiDSpec,
    Img2ColSpec,
    ImplicitGemmConvSpec,
    JengaSparseSpec,
    LayerNorm2DSpec,
    MfmaGemmSpec,
    MoeSmoothQuantSpec,
    MoeSortingSpec,
    MxGemmSpec,
    PermuteSpec,
    Pooling2DSpec,
    PoolingProblem,
    RMSNorm2DSpec,
    Reduce2DSpec,
    SageAttentionSpec,
    SmoothQuantSpec,
    StreamKGemmSpec,
    TileSpec,
    TopkSoftmaxSpec,
    TraitSpec,
    Transpose2DSpec,
    UnifiedAttention2DTiledSpec,
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    UniversalGemmSpec,
    VsaSparseSpec,
    build_add_rmsnorm2d_rdquant,
    build_batched_contraction,
    build_batched_gemm,
    build_batched_transpose2d,
    build_block_scale_gemm,
    build_direct_conv_16c,
    build_direct_conv_4c,
    build_elementwise,
    build_flatmm,
    build_fmha_bwd,
    build_fmha_fwd_appendkv,
    build_fmha_fwd_fp8,
    build_fmha_fwd_head_grouping,
    build_fmha_fwd_paged_prefill,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    build_fmha_fwd_varlen,
    build_gemm_multi_abd,
    build_gemm_multi_d,
    build_img2col,
    build_implicit_gemm_conv,
    build_jenga_sparse_attention,
    build_layernorm2d,
    build_mfma_gemm,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_smoothquant,
    build_moe_sort_histogram,
    build_moe_sort_scan,
    build_moe_sort_scatter,
    build_moe_topk_weighted_reduce,
    build_mx_gemm,
    build_permute,
    build_pooling2d,
    build_reduce2d,
    build_rmsnorm2d,
    build_sage_attention,
    build_smoothquant,
    build_streamk_gemm,
    build_topk_softmax,
    build_transpose2d,
    build_unified_attention_2d_tiled,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    build_universal_gemm,
    build_vsa_sparse_attention,
)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    build: Callable[[], object]


@dataclass
class AuditResult:
    name: str
    group: str
    llvm_ok: bool = False
    hip_ok: bool = False
    hip_compile_ok: Optional[bool] = None
    hip_chars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        if not (self.llvm_ok and self.hip_ok):
            return False
        return self.hip_compile_ok is not False


def _base_tile() -> TileSpec:
    return TileSpec(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def _base_trait() -> TraitSpec:
    return TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="cshuffle")


def _base_gemm(name: str = "hip_audit_gemm") -> UniversalGemmSpec:
    return UniversalGemmSpec(name=name, tile=_base_tile(), trait=_base_trait())


def _conv_problem() -> ConvProblem:
    return ConvProblem(
        N=1,
        Hi=8,
        Wi=8,
        C=16,
        K=16,
        R=3,
        S=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )


def _fmha_common(dtype: str = "f16") -> FmhaCommonSpec:
    return FmhaCommonSpec(
        FmhaShape(
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            block_size_q=16,
            block_size_k=64,
        ),
        dtype=dtype,
    )


def make_cases(*, include_attention: bool = True) -> List[Case]:
    base = _base_gemm()
    base_tile = _base_tile()
    base_trait = _base_trait()
    convp = _conv_problem()
    common = _fmha_common()
    qks = QkScaleSpec(layout="per_head", stride_batch=4, stride_head=1)

    cases: List[Case] = [
        Case(
            "elementwise.add",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="add")),
        ),
        Case(
            "elementwise.silu",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="silu")),
        ),
        Case(
            "elementwise.gelu_tanh",
            "small",
            lambda: build_elementwise(ElementwiseSpec(op="gelu_tanh")),
        ),
        Case(
            "layernorm2d",
            "small",
            lambda: build_layernorm2d(
                LayerNorm2DSpec(n_per_block=2048, block_size=256, vec=8)
            ),
        ),
        Case(
            "rmsnorm2d",
            "small",
            lambda: build_rmsnorm2d(
                RMSNorm2DSpec(n_per_block=2048, block_size=256, vec=8)
            ),
        ),
        Case(
            "reduce.sum",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(n_per_block=2048, op="sum", block_size=256, vec=8)
            ),
        ),
        Case(
            "reduce.max",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(n_per_block=2048, op="max", block_size=256, vec=8)
            ),
        ),
        Case(
            "transpose2d",
            "small",
            lambda: build_transpose2d(Transpose2DSpec(tile_m=16, tile_n=16, vec=4)),
        ),
        Case(
            "batched_transpose2d",
            "small",
            lambda: build_batched_transpose2d(
                BatchedTranspose2DSpec(tile_m=16, tile_n=16, vec=4)
            ),
        ),
        Case(
            "img2col",
            "small",
            lambda: build_img2col(
                Img2ColSpec(problem=convp, block_tile_m=8, block_tile_k=64)
            ),
        ),
        Case(
            "pooling.max",
            "small",
            lambda: build_pooling2d(
                Pooling2DSpec(
                    problem=PoolingProblem(N=1, H=8, W=8, C=16, Y=2, X=2), op="max"
                )
            ),
        ),
        Case(
            "permute.2d",
            "small",
            lambda: build_permute(PermuteSpec(x_shape=(16, 32), perm=(1, 0))),
        ),
        Case(
            "smoothquant.i8",
            "quant",
            lambda: build_smoothquant(
                SmoothQuantSpec(n_per_block=2048, out_dtype="i8", block_size=256, vec=8)
            ),
        ),
        Case(
            "smoothquant.fp8",
            "quant",
            lambda: build_smoothquant(
                SmoothQuantSpec(
                    n_per_block=2048, out_dtype="fp8e4m3", block_size=256, vec=8
                )
            ),
        ),
        Case(
            "moe_smoothquant",
            "quant",
            lambda: build_moe_smoothquant(
                MoeSmoothQuantSpec(
                    n_per_block=2048,
                    topk=2,
                    experts=4,
                    out_dtype="i8",
                    block_size=256,
                    vec=8,
                )
            ),
        ),
        Case(
            "add_rmsnorm2d_rdquant",
            "quant",
            lambda: build_add_rmsnorm2d_rdquant(
                AddRmsnorm2DRdquantSpec(
                    n_per_block=2048, out_dtype="i8", block_size=256, vec=8
                )
            ),
        ),
        Case(
            "topk_softmax",
            "small",
            lambda: build_topk_softmax(
                TopkSoftmaxSpec(n_per_row=64, k=4, dtype="f16", out_dtype="f16")
            ),
        ),
        Case(
            "moe_sort_histogram",
            "moe",
            lambda: build_moe_sort_histogram(
                MoeSortingSpec(tokens=32, experts=4, topk=2)
            ),
        ),
        Case(
            "moe_sort_scan",
            "moe",
            lambda: build_moe_sort_scan(MoeSortingSpec(tokens=32, experts=4, topk=2)),
        ),
        Case(
            "moe_sort_scatter",
            "moe",
            lambda: build_moe_sort_scatter(
                MoeSortingSpec(tokens=32, experts=4, topk=2)
            ),
        ),
        Case(
            "moe_gather",
            "moe",
            lambda: build_moe_gather(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case(
            "moe_silu_mul",
            "moe",
            lambda: build_moe_silu_mul(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case(
            "moe_topk_weighted_reduce",
            "moe",
            lambda: build_moe_topk_weighted_reduce(
                FusedMoeSpec(
                    tokens=32,
                    experts=4,
                    topk=2,
                    hidden=128,
                    intermediate=256,
                    block_size=64,
                    vec=4,
                )
            ),
        ),
        Case("universal_gemm", "gemm", lambda: build_universal_gemm(base)),
        Case(
            "batched_gemm",
            "gemm",
            lambda: build_batched_gemm(
                BatchedGemmSpec(
                    name="hip_audit_batched_gemm", tile=base_tile, trait=base_trait
                )
            ),
        ),
        Case(
            "flatmm",
            "gemm",
            lambda: build_flatmm(FlatMMSpec(tile=base_tile, trait=base_trait)),
        ),
        Case(
            "batched_contraction",
            "gemm",
            lambda: build_batched_contraction(
                BatchedContractionSpec(
                    tile=base_tile, trait=base_trait, batch_shape=(2, 3)
                )
            ),
        ),
        Case(
            "gemm_multi_d",
            "gemm",
            lambda: build_gemm_multi_d(
                GemmMultiDSpec(base=base, d_operands=(("D0", "add"),))
            ),
        ),
        Case(
            "gemm_multi_abd",
            "gemm",
            lambda: build_gemm_multi_abd(
                GemmMultiAbdSpec(base=base, d_operands=(("D0", "add"),))
            ),
        ),
        Case(
            "mfma_gemm", "gemm", lambda: build_mfma_gemm(MfmaGemmSpec(M=64, N=64, K=32))
        ),
        Case(
            "block_scale_gemm",
            "gemm",
            lambda: build_block_scale_gemm(
                BlockScaleGemmSpec(
                    M=16, N=16, K=128, quant_mode="abquant", mantissa_dtype="fp8e4m3"
                )
            ),
        ),
        Case(
            "mx_gemm",
            "gemm",
            lambda: build_mx_gemm(
                MxGemmSpec(M=16, N=16, K=32, mantissa_dtype="fp8e4m3")
            ),
        ),
        Case(
            "streamk_gemm",
            "gemm",
            lambda: build_streamk_gemm(
                StreamKGemmSpec(
                    M=64, N=64, K=32, tile_m=16, tile_n=16, tile_k=16, num_cus=4
                )
            ),
        ),
        Case(
            "implicit_gemm_conv",
            "conv",
            lambda: build_implicit_gemm_conv(
                ImplicitGemmConvSpec(
                    problem=convp,
                    name="hip_audit_conv",
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
            ),
        ),
        Case(
            "direct_conv_16c",
            "conv",
            lambda: build_direct_conv_16c(
                DirectConv16cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16)
                )
            ),
        ),
        Case(
            "direct_conv_4c",
            "conv",
            lambda: build_direct_conv_4c(
                DirectConv4cSpec(
                    DirectConvProblem(N=1, H=8, W=8, groups=16, cpg=4, kpg=4)
                )
            ),
        ),
    ]

    for dtype in ("f16", "bf16"):
        for op in (
            "copy",
            "neg",
            "abs",
            "relu",
            "quick_gelu",
            "swish",
            "tanh",
            "sigmoid",
            "exp2",
            "sub",
            "mul",
            "max",
            "min",
            "swiglu",
            "geglu",
        ):
            cases.append(
                Case(
                    f"elementwise.{dtype}.{op}",
                    "small",
                    lambda op=op, dtype=dtype: build_elementwise(
                        ElementwiseSpec(op=op, dtype=dtype)
                    ),
                )
            )

    for dtype in ("f16", "bf16"):
        cases.extend(
            [
                Case(
                    f"layernorm2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_layernorm2d(
                        LayerNorm2DSpec(
                            n_per_block=2048, dtype=dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"rmsnorm2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_rmsnorm2d(
                        RMSNorm2DSpec(
                            n_per_block=2048, dtype=dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"transpose2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_transpose2d(
                        Transpose2DSpec(tile_m=16, tile_n=16, vec=4, dtype=dtype)
                    ),
                ),
                Case(
                    f"batched_transpose2d.{dtype}",
                    "small",
                    lambda dtype=dtype: build_batched_transpose2d(
                        BatchedTranspose2DSpec(tile_m=16, tile_n=16, vec=4, dtype=dtype)
                    ),
                ),
            ]
        )

    for op in ("min", "mean", "prod"):
        cases.append(
            Case(
                f"reduce.{op}",
                "small",
                lambda op=op: build_reduce2d(
                    Reduce2DSpec(n_per_block=2048, op=op, block_size=256, vec=8)
                ),
            )
        )
    cases.append(
        Case(
            "reduce.bf16.sum",
            "small",
            lambda: build_reduce2d(
                Reduce2DSpec(
                    n_per_block=2048, op="sum", dtype="bf16", block_size=256, vec=8
                )
            ),
        )
    )

    for op in ("avg", "sum"):
        cases.append(
            Case(
                f"pooling.{op}",
                "small",
                lambda op=op: build_pooling2d(
                    Pooling2DSpec(
                        problem=PoolingProblem(N=1, H=8, W=8, C=16, Y=2, X=2),
                        op=op,
                    )
                ),
            )
        )
    cases.append(
        Case(
            "permute.3d.bf16",
            "small",
            lambda: build_permute(
                PermuteSpec(x_shape=(4, 8, 16), perm=(2, 0, 1), dtype="bf16")
            ),
        )
    )

    for out_dtype in ("bf8e5m2",):
        cases.extend(
            [
                Case(
                    f"smoothquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_smoothquant(
                        SmoothQuantSpec(
                            n_per_block=2048, out_dtype=out_dtype, block_size=256, vec=8
                        )
                    ),
                ),
                Case(
                    f"moe_smoothquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_moe_smoothquant(
                        MoeSmoothQuantSpec(
                            n_per_block=2048,
                            topk=2,
                            experts=4,
                            out_dtype=out_dtype,
                            block_size=256,
                            vec=8,
                        )
                    ),
                ),
                Case(
                    f"add_rmsnorm2d_rdquant.{out_dtype}",
                    "quant",
                    lambda out_dtype=out_dtype: build_add_rmsnorm2d_rdquant(
                        AddRmsnorm2DRdquantSpec(
                            n_per_block=2048, out_dtype=out_dtype, block_size=256, vec=8
                        )
                    ),
                ),
            ]
        )

    for dtype, out_dtype in (("bf16", "bf16"), ("f32", "f32"), ("f16", "f32")):
        cases.append(
            Case(
                f"topk_softmax.{dtype}.{out_dtype}",
                "small",
                lambda dtype=dtype, out_dtype=out_dtype: build_topk_softmax(
                    TopkSoftmaxSpec(
                        n_per_row=64,
                        k=4,
                        dtype=dtype,
                        out_dtype=out_dtype,
                        block_size=64,
                    )
                ),
            )
        )

    for pipeline in ("compv3", "compv4"):
        cases.append(
            Case(
                f"universal_gemm.{pipeline}",
                "gemm",
                lambda pipeline=pipeline: build_universal_gemm(
                    UniversalGemmSpec(
                        name=f"hip_audit_gemm_{pipeline}",
                        tile=base_tile,
                        trait=TraitSpec(
                            pipeline=pipeline,
                            scheduler="intrawave",
                            epilogue="cshuffle",
                        ),
                    )
                ),
            )
        )
    cases.append(
        Case(
            "universal_gemm.default_epilogue",
            "gemm",
            lambda: build_universal_gemm(
                UniversalGemmSpec(
                    name="hip_audit_gemm_default_epilogue",
                    tile=base_tile,
                    trait=TraitSpec(
                        pipeline="mem", scheduler="intrawave", epilogue="default"
                    ),
                )
            ),
        )
    )
    cases.extend(
        [
            Case(
                "block_scale_gemm.bf8",
                "gemm",
                lambda: build_block_scale_gemm(
                    BlockScaleGemmSpec(
                        M=16,
                        N=16,
                        K=128,
                        quant_mode="abquant",
                        mantissa_dtype="bf8e5m2",
                    )
                ),
            ),
            Case(
                "mx_gemm.bf8",
                "gemm",
                lambda: build_mx_gemm(
                    MxGemmSpec(M=16, N=16, K=32, mantissa_dtype="bf8e5m2")
                ),
            ),
        ]
    )

    if not include_attention:
        return cases

    cases.extend(
        [
            Case(
                "fmha_varlen",
                "attention",
                lambda: build_fmha_fwd_varlen(
                    FmhaFwdVarlenSpec(
                        common=common, max_seqlen_q=64, max_seqlen_k=64, batch=2
                    )
                ),
            ),
            Case(
                "fmha_appendkv",
                "attention",
                lambda: build_fmha_fwd_appendkv(
                    FmhaAppendKvSpec(common=common, batch=2)
                ),
            ),
            Case(
                "fmha_paged_prefill",
                "attention",
                lambda: build_fmha_fwd_paged_prefill(
                    FmhaFwdPagedPrefillSpec(
                        common=common, page_block_size=16, max_blocks_per_seq=8, batch=2
                    )
                ),
            ),
            Case(
                "fmha_splitkv_segment",
                "attention",
                lambda: build_fmha_fwd_splitkv_decode_segment(
                    FmhaFwdSplitKvDecodeSpec(common=common, batch=2, num_segments=4)
                ),
            ),
            Case(
                "fmha_splitkv_reduce",
                "attention",
                lambda: build_fmha_fwd_splitkv_decode_reduce(
                    FmhaFwdSplitKvDecodeSpec(common=common, batch=2, num_segments=4)
                ),
            ),
            Case(
                "fmha_head_grouping",
                "attention",
                lambda: build_fmha_fwd_head_grouping(
                    FmhaFwdHeadGroupingSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "fmha_bwd",
                "attention",
                lambda: build_fmha_bwd(
                    FmhaBwdSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "fmha_fwd_fp8",
                "attention",
                lambda: build_fmha_fwd_fp8(
                    FmhaFwdFp8Spec(
                        common=common, kv_dtype="fp8e4m3", seqlen_q=16, seqlen_k=64
                    )
                ),
            ),
            Case(
                "sage_fp16_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="fp16_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "sage_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "jenga_sparse",
                "attention",
                lambda: build_jenga_sparse_attention(
                    JengaSparseSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "vsa_sparse",
                "attention",
                lambda: build_vsa_sparse_attention(
                    VsaSparseSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "uattn_2d_tiled",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_bf16",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                    )
                ),
            ),
            Case(
                "uattn_3d_tiled",
                "attention",
                lambda: build_unified_attention_3d_tiled(
                    UnifiedAttention3DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        num_segments=4,
                    )
                ),
            ),
            Case(
                "uattn_reduce_tiled",
                "attention",
                lambda: build_unified_attention_reduce_tiled(
                    UnifiedAttentionReduceTiledSpec(
                        head_size=128,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        num_segments=4,
                    )
                ),
            ),
        ]
    )
    common_bf16 = _fmha_common("bf16")
    cases.extend(
        [
            Case(
                "fmha_varlen.bf16",
                "attention",
                lambda: build_fmha_fwd_varlen(
                    FmhaFwdVarlenSpec(
                        common=common_bf16, max_seqlen_q=64, max_seqlen_k=64, batch=2
                    )
                ),
            ),
            Case(
                "fmha_fwd_fp8.bf8",
                "attention",
                lambda: build_fmha_fwd_fp8(
                    FmhaFwdFp8Spec(
                        common=common, kv_dtype="bf8e5m2", seqlen_q=16, seqlen_k=64
                    )
                ),
            ),
            Case(
                "sage_i8_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="i8_fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "sage_i4_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="i4_fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_softcap",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=True,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_alibi",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        use_alibi=True,
                    )
                ),
            ),
            Case(
                "uattn_3d_tiled_bf16",
                "attention",
                lambda: build_unified_attention_3d_tiled(
                    UnifiedAttention3DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        num_segments=4,
                    )
                ),
            ),
            Case(
                "uattn_reduce_tiled_bf16",
                "attention",
                lambda: build_unified_attention_reduce_tiled(
                    UnifiedAttentionReduceTiledSpec(
                        head_size=128,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        num_segments=4,
                    )
                ),
            ),
        ]
    )
    return cases


def _selected(cases: Iterable[Case], needle: str) -> List[Case]:
    if needle == "all":
        return list(cases)
    return [c for c in cases if needle in c.name or needle == c.group]


def _compile_hip_source(
    source: str,
    *,
    name: str,
    arch: str,
    output_dir: Path,
    timeout_s: int,
) -> None:
    src = output_dir / f"{name.replace('.', '_')}.hip"
    hsaco = output_dir / f"{name.replace('.', '_')}.hsaco"
    src.write_text(source, encoding="utf-8")
    cmd = [
        "hipcc",
        f"--offload-arch={arch}",
        "--genco",
        str(src),
        "-o",
        str(hsaco),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        msg = "\n".join(proc.stdout.splitlines()[:40])
        raise RuntimeError(f"hipcc failed with rc={proc.returncode}\n{msg}")


def audit_cases(
    cases: Iterable[Case],
    *,
    compile_hip: bool,
    arch: str,
    emit_dir: Optional[Path],
    compile_timeout_s: int,
) -> List[AuditResult]:
    if emit_dir is None:
        tmp = tempfile.TemporaryDirectory()
        output_dir = Path(tmp.name)
    else:
        tmp = None
        output_dir = emit_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[AuditResult] = []
    try:
        for case in cases:
            result = AuditResult(name=case.name, group=case.group)
            try:
                kernel = case.build()
                lower_kernel_to_llvm(kernel)
                result.llvm_ok = True
                hip_source = lower_kernel_to_hip(kernel)
                result.hip_ok = True
                result.hip_chars = len(hip_source)
                if compile_hip:
                    _compile_hip_source(
                        hip_source,
                        name=case.name,
                        arch=arch,
                        output_dir=output_dir,
                        timeout_s=compile_timeout_s,
                    )
                    result.hip_compile_ok = True
            except Exception as exc:  # noqa: BLE001 - audit should classify all failures
                result.error = f"{type(exc).__name__}: {exc}"
                if compile_hip and result.hip_ok and result.hip_compile_ok is None:
                    result.hip_compile_ok = False
            results.append(result)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return results


def bench_elementwise_add(*, arch: str) -> None:
    # This script imports the full instance package for audit coverage, and
    # some instance modules import `runtime.launcher` at module import time.
    # Torch-backed launches need torch to initialize HIP first, so run the
    # smoke benchmark in a fresh child process with a minimal import order.
    code = f"""
import subprocess, sys, tempfile
from pathlib import Path
import torch
torch.cuda.init()
sys.path.insert(0, {str(ROOT / "python")!r})
from ck_dsl.core.lower_hip import lower_kernel_to_hip
from ck_dsl.helpers.compile import compile_kernel
from ck_dsl.instances.elementwise import ElementwiseSpec, build_elementwise, elementwise_grid, elementwise_signature
from ck_dsl.instances.mfma_gemm import MfmaGemmSpec, build_mfma_gemm, mfma_gemm_grid, mfma_gemm_signature
from ck_dsl.instances.reduce import Reduce2DSpec, build_reduce2d, reduce2d_grid, reduce2d_signature
from ck_dsl.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release, time_launches

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)

    def make_launchers(name, kernel, signature):
        llvm_artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--{arch}")
        llvm_launcher = KernelLauncher(
            hsaco=llvm_artifact.hsaco,
            kernel_name=kernel.name,
            signature=signature,
        )
        stem = name.replace(".", "_")
        src = td_path / f"{{stem}}.hip"
        hsaco = td_path / f"{{stem}}.hsaco"
        src.write_text(lower_kernel_to_hip(kernel), encoding="utf-8")
        proc = subprocess.run(
            ["hipcc", "--offload-arch={arch}", "--genco", str(src), "-o", str(hsaco)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            raise SystemExit(proc.returncode)
        hip_launcher = KernelLauncher(
            hsaco=hsaco.read_bytes(),
            kernel_name=kernel.name,
            signature=signature,
        )
        return llvm_launcher, hip_launcher

    def run_pair(name, llvm_launcher, hip_launcher, args_llvm, args_hip, cfg, out_llvm, out_hip, *, iters=50):
        llvm_launcher(args_llvm, config=cfg)
        hip_launcher(args_hip, config=cfg)
        synchronize_and_release()
        max_abs = float((out_llvm.float() - out_hip.float()).abs().max().item())
        llvm_ms = time_launches(lambda: llvm_launcher(args_llvm, config=cfg), warmup=5, iters=iters)
        hip_ms = time_launches(lambda: hip_launcher(args_hip, config=cfg), warmup=5, iters=iters)
        ratio = hip_ms / llvm_ms if llvm_ms else float("inf")
        print(
            f"BENCH {{name}} correct={{max_abs == 0.0}} max_abs={{max_abs:.3g}} "
            f"llvm_ms={{llvm_ms:.6f}} hip_ms={{hip_ms:.6f}} hip_over_llvm={{ratio:.3f}}"
        )

    # Memory/vector path smoke.
    ew_spec = ElementwiseSpec(op="add")
    ew_kernel = build_elementwise(ew_spec)
    ew_llvm, ew_hip = make_launchers("elementwise.add", ew_kernel, elementwise_signature(ew_spec))
    n = 1 << 20
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    c_llvm = torch.empty_like(a)
    c_hip = torch.empty_like(a)
    ew_cfg = LaunchConfig(grid=elementwise_grid(n, ew_spec), block=(ew_spec.block_size, 1, 1))
    run_pair(
        "elementwise.add",
        ew_llvm,
        ew_hip,
        {{"A": a, "B": b, "C": c_llvm, "N": n}},
        {{"A": a, "B": b, "C": c_hip, "N": n}},
        ew_cfg,
        c_llvm,
        c_hip,
    )

    # LDS reduction path smoke.
    red_spec = Reduce2DSpec(n_per_block=2048, op="sum", block_size=256, vec=8)
    red_kernel = build_reduce2d(red_spec)
    red_llvm, red_hip = make_launchers("reduce.sum", red_kernel, reduce2d_signature(red_spec))
    m = 128
    x = torch.randn(m, red_spec.n_per_block, device="cuda", dtype=torch.float16)
    y_llvm = torch.empty(m, device="cuda", dtype=torch.float16)
    y_hip = torch.empty(m, device="cuda", dtype=torch.float16)
    red_cfg = LaunchConfig(grid=reduce2d_grid(m, red_spec), block=(red_spec.block_size, 1, 1))
    run_pair(
        "reduce.sum",
        red_llvm,
        red_hip,
        {{"X": x, "Y": y_llvm, "M": m, "N": red_spec.n_per_block}},
        {{"X": x, "Y": y_hip, "M": m, "N": red_spec.n_per_block}},
        red_cfg,
        y_llvm,
        y_hip,
    )

    # MFMA path smoke.
    gemm_spec = MfmaGemmSpec(M=64, N=64, K=32)
    gemm_kernel = build_mfma_gemm(gemm_spec)
    gemm_llvm, gemm_hip = make_launchers("mfma_gemm", gemm_kernel, mfma_gemm_signature(gemm_spec))
    ga = torch.randn(gemm_spec.M, gemm_spec.K, device="cuda", dtype=torch.float16)
    gb = torch.randn(gemm_spec.K, gemm_spec.N, device="cuda", dtype=torch.float16)
    gc_llvm = torch.empty(gemm_spec.M, gemm_spec.N, device="cuda", dtype=torch.float16)
    gc_hip = torch.empty_like(gc_llvm)
    gemm_cfg = LaunchConfig(grid=mfma_gemm_grid(gemm_spec), block=(gemm_spec.block_size, 1, 1))
    run_pair(
        "mfma_gemm",
        gemm_llvm,
        gemm_hip,
        {{"A": ga, "B": gb, "C": gc_llvm, "M": gemm_spec.M, "N": gemm_spec.N, "K": gemm_spec.K}},
        {{"A": ga, "B": gb, "C": gc_hip, "M": gemm_spec.M, "N": gemm_spec.N, "K": gemm_spec.K}},
        gemm_cfg,
        gc_llvm,
        gc_hip,
        iters=30,
    )
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"bench smoke failed with rc={proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="substring or group filter")
    parser.add_argument(
        "--no-attention", action="store_true", help="skip attention cases"
    )
    parser.add_argument(
        "--compile-hip", action="store_true", help="compile HIP source to HSACO"
    )
    parser.add_argument("--compile-timeout-s", type=int, default=180)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--emit-dir", type=Path, default=None)
    parser.add_argument("--bench-smoke", action="store_true")
    args = parser.parse_args()

    cases = _selected(make_cases(include_attention=not args.no_attention), args.case)
    if not cases:
        print(f"no cases matched {args.case!r}")
        return 2

    results = audit_cases(
        cases,
        compile_hip=args.compile_hip,
        arch=args.arch,
        emit_dir=args.emit_dir,
        compile_timeout_s=args.compile_timeout_s,
    )
    for r in results:
        compile_status = ""
        if args.compile_hip:
            compile_status = f" hipcc={'OK' if r.hip_compile_ok else 'FAIL'}"
        status = "OK" if r.ok else "FAIL"
        print(
            f"{status:4} {r.group:9} {r.name:28} "
            f"llvm={'OK' if r.llvm_ok else 'FAIL'} hip={'OK' if r.hip_ok else 'FAIL'} "
            f"chars={r.hip_chars}{compile_status}"
        )
        if r.error:
            print(f"     {r.error}")

    failures = [r for r in results if not r.ok]
    print(
        f"SUMMARY total={len(results)} ok={len(results) - len(failures)} fail={len(failures)}"
    )

    if args.bench_smoke:
        bench_elementwise_add(arch=args.arch)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
