# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the convolution families.

Hashes the Python-lowered LLVM IR (SHA256) of representative convolution
kernels and compares against a checked-in per-flavor golden fixture,
``golden/conv_representative_ir/`` (one ``<family>.json`` shard per family),
catching any unintended codegen drift. Pure
text lowering -- no GPU / no comgr required. Every flavor in ``LLVM_FLAVORS``
is checked from any host, through the shared ``rocke.core.ir_golden``
comparator, so a newly added flavor fails here until the fixture is
re-blessed.

Covers the representative convolution cases: implicit-GEMM forward
(``conv``, with its split-K Stage-2 ``conv_fwd_split_k_reduce``), wgrad
(``conv_wgrad``, ``conv_wgrad_two_stage``, ``conv_wgrad_reduce``) and dgrad
(``conv_dgrad``); the deep fused conv+pool arch shims (``deep``, including the
gfx11/gfx12 generic targets); the direct conv kernels (``conv_direct*``),
including the gfx1250 WMMA forward, dgrad, small-channel and patch-stem
kernels; and the gfx1250 Winograd families (``conv_winograd_*``).

These cases, their case ids and their recorded hashes were moved verbatim
from the platform's representative-IR harness
(``platform/tests/instances/rocke_ir_parity_harness.py`` and its shards in
``platform/tests/golden/rocke_representative_ir_sha256/``), so a conv codegen
change re-blesses this library-owned fixture only. The shards keep the
harness' schema (``ck.dsl.ir_golden_sha256/v2``) and lane aliases: per
profile flavor the ``cases`` and any ``expected_failures``.

C++ engine byte-identity for these families is gated separately, by
``tools/check_byte_identity.py`` over ``library/tests/parity/``.

Check, or re-bless after a reviewed, expected output change, from
``rocke/library``:

    PYTHONPATH=../platform/python:. python tests/test_conv_ir_golden.py --check
    PYTHONPATH=../platform/python:. python tests/test_conv_ir_golden.py --write
    PYTHONPATH=../platform/python:. python tests/test_conv_ir_golden.py \\
        --write --only conv_wgrad

Re-blessing is a deliberate act, never a way to turn red green: a hash that
moved without a matching, explained emission change is a defect, not a stale
fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pin the ``library/`` root ahead of everything on sys.path. When this file is
# run directly Python puts ``tests/`` on sys.path[0], where ``tests/dispatch/``
# would otherwise shadow the real ``library/dispatch`` package. Harmless under
# pytest.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)

from rocke.core.ir_golden import (  # noqa: E402 -- library root pinned above
    GOLDEN_FLAVORS,
    check_golden_flavors,
    golden_document,
    golden_main,
    lower_golden_case,
    run_golden_cases,
    write_golden_shards,
)
from rocke.core.ir_golden import check_golden as _check_golden  # noqa: E402
from rocke.tests.arch_marks import arch_specific  # noqa: E402

_GOLDEN = Path(__file__).resolve().parent / "golden" / "conv_representative_ir"


# The platform harness's lowering, (record, llvm), so
# platform/tools/check_ir_validity.py --cases can walk this corpus; see
# rocke.core.ir_golden.lower_golden_case for why the lowerer and flavor are
# pinned.
lower_case = lower_golden_case


def _tf32_conv_data():
    """fp32 storage with TF32 MFMA operands (gfx942 native XF32, gfx950
    BF16x3) for the implicit-GEMM conv builders' ``tf32=True``."""
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    return ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")


def build_conv(
    name,
    arch,
    problem_args,
    *,
    wave_size,
    wtm,
    wtn,
    wtk,
    tile_m,
    tile_n,
    tile_k,
    pipeline="mem",
    epilogue="default",
    groups=1,
    vector_size_c=None,
    group_merge=1,
    split_k=1,
    split_k_cluster=0,
    dtype="fp16",
    tf32=False,
):
    def _build():
        from kernels.common.conv_implicit_gemm import (
            ConvDataSpec,
            ConvProblem,
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        p = ConvProblem(*problem_args)
        spec = ImplicitGemmConvSpec(
            problem=p,
            data=(_tf32_conv_data() if tf32 else ConvDataSpec(dtype, dtype, dtype)),
            name=name,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=1 if wave_size == 32 else 2,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
            wave_size=wave_size,
            pipeline=pipeline,
            epilogue=epilogue,
            groups=groups,
            vector_size_c=vector_size_c,
            group_merge=group_merge,
            split_k=split_k,
            split_k_cluster=split_k_cluster,
        )
        return build_implicit_gemm_conv(spec, arch=arch)

    return _build


def build_conv_wgrad(
    name,
    arch,
    problem_args,
    *,
    wave_size,
    wtm,
    wtn,
    wtk,
    tile_m,
    tile_n,
    tile_k,
    pipeline="mem",
    epilogue="default",
    split_k=1,
    dtype_d="fp16",
    two_stage=False,
    streamk="off",
    streamk_reduction="linear",
    streamk_ctas=-1,
    tf32=False,
    **spec_extra,
):
    def _build():
        from kernels.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            build_implicit_gemm_conv_wgrad,
        )
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )

        p = ConvProblem(*problem_args)
        spec = WgradConvSpec(
            problem=p,
            name=name,
            data=(
                _tf32_conv_data()
                if tf32
                else ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d)
            ),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=1 if wave_size == 32 else 2,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
            wave_size=wave_size,
            pipeline=pipeline,
            epilogue=epilogue,
            split_k=split_k,
            two_stage=two_stage,
            streamk=streamk,
            streamk_reduction=streamk_reduction,
            streamk_ctas=streamk_ctas,
            **spec_extra,
        )
        return build_implicit_gemm_conv_wgrad(spec, arch=arch)

    return _build


def build_dgrad(
    name,
    arch,
    problem_args,
    *,
    wave_size,
    wtm,
    wtn,
    wtk,
    tile_m,
    tile_n,
    tile_k,
    split_k=1,
    tap_outer_k=False,
    halo=None,
    tf32=False,
):
    def _build():
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
        )
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )

        p = ConvProblem(*problem_args)
        spec = DgradConvSpec(
            problem=p,
            name=name,
            data=(
                _tf32_conv_data()
                if tf32
                else ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16")
            ),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=1 if wave_size == 32 else 2,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
            wave_size=wave_size,
            split_k=split_k,
            tap_outer_k=tap_outer_k,
            **(halo or {}),
        )
        return build_implicit_gemm_conv_dgrad(spec, arch=arch)

    return _build


def build_conv_wgrad_reduce(
    name, arch, wg_M, wg_N, dtype_d="fp16", tile_m=4, tile_n=64
):
    """Build a workspace-reduce (Stage 2) kernel for the two-stage deterministic wgrad.

    ``wg_M`` = K (output channels), ``wg_N`` = Y*X*C (filter spatial × input channel).
    A minimal ConvProblem with Y=1, X=1, C=wg_N, K=wg_M is used solely to satisfy
    WgradReduceSpec's requirement for a ConvProblem (it only reads wg_M/wg_N from it).
    """

    def _build():
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            build_conv_wgrad_workspace_reduce,
        )
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        # WgradReduceSpec only uses p for wg_M (= K) and wg_N (= Y*X*C).
        # Y=1, X=1 → wg_N = C = wg_N.
        p = ConvProblem(N=1, Hi=1, Wi=1, C=wg_N, K=wg_M, Y=1, X=1)
        spec = WgradReduceSpec(
            problem=p,
            dtype_d=dtype_d,
            tile_m=tile_m,
            tile_n=tile_n,
            name=name,
        )
        return build_conv_wgrad_workspace_reduce(spec, arch=arch)

    return _build


def build_conv_fwd_split_k_reduce(arch, **spec_kw):
    """Build the split-K forward Stage 2 (folds the fp32 slabs into D)."""

    def _build():
        from kernels.common.conv_fwd_split_k_reduce import (
            FwdSplitKReduceSpec,
            build_conv_fwd_split_k_reduce,
        )

        return build_conv_fwd_split_k_reduce(FwdSplitKReduceSpec(**spec_kw), arch=arch)

    return _build


def build_deep(kind, arch, **kw):
    def _build():
        if kind == "common":
            from kernels.common.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        elif kind == "gfx950":
            from kernels.gfx950.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        elif kind == "gfx1201":
            from kernels.gfx1201.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        else:
            from kernels.gfx1151.deep_fused_conv_pool import (
                build_deep_fused_conv_pool,
                make_deep_fused_conv_pool_spec,
            )
        return build_deep_fused_conv_pool(
            make_deep_fused_conv_pool_spec(**kw), arch=arch
        )

    return _build


# ---------------------------------------------------------------------------
# Direct conv builder helpers
# ---------------------------------------------------------------------------
# Each helper returns a zero-argument closure (_build) that constructs the
# spec and calls the appropriate build_direct_conv_* function.  The Python
# lowerer is used exclusively (lower_case pins _lower_kernel_to_llvm_python).
#
# C++ engine parity: direct-conv parity is gated via tools/check_byte_identity.py
# (see library/tests/parity/conv_direct_grouped_emit.* and the
# conv_direct_grouped family in library/tests/parity/golden/llvm_gfx_all.json).
# The existing direct-conv variants occupy configs 0-24; the wgrad variant is
# configs 25-31 (25 mfma_k=32, 26 mfma_k=16, 27 multi-wave K/C/Q, 28-29 the two
# gfx942 rejection paths, 30-31 bf16 at mfma_k=32 / 16).
# If you add new direct-conv variants, add matching configs to both emitters and
# re-bless the golden.
# ---------------------------------------------------------------------------


def build_direct_16c(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    *,
    block_q=16,
    block_groups=8,
    fold_k32=True,
    double_buffer=True,
    dtype="fp16",
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConv16cSpec,
            DirectConvProblem,
            build_direct_conv_16c,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=16,
            kpg=16,
            KH=KH,
            KW=KW,
            PAD=PAD,
            dtype=dtype,
        )
        spec = DirectConv16cSpec(
            problem=p,
            name=name,
            block_q=block_q,
            block_groups=block_groups,
            fold_k32=fold_k32,
            double_buffer=double_buffer,
        )
        return build_direct_conv_16c(spec, arch=arch)

    return _build


def build_direct_4c(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    *,
    block_q=4,
    block_groups=16,
    dtype="fp16",
    dgrad_fused_weights=False,
    dgrad_weights_lds=False,
    stage_rows=False,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConv4cSpec,
            DirectConvProblem,
            build_direct_conv_4c,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=4,
            kpg=4,
            KH=KH,
            KW=KW,
            PAD=PAD,
            dtype=dtype,
        )
        spec = DirectConv4cSpec(
            problem=p,
            name=name,
            block_q=block_q,
            block_groups=block_groups,
            dgrad_fused_weights=dgrad_fused_weights,
            dgrad_weights_lds=dgrad_weights_lds,
            stage_rows=stage_rows,
            waves_q=block_q // 4 if stage_rows else 1,
        )
        return build_direct_conv_4c(spec, arch=arch)

    return _build


def build_direct_generic(name, arch, problem, **spec_kw):
    """The generic forward DirectConvSpec kernel; ``problem`` holds the
    DirectConvProblem fields, ``spec_kw`` further DirectConvSpec fields."""

    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectConvSpec,
            build_direct_conv,
        )

        spec = DirectConvSpec(
            problem=DirectConvProblem(**problem), name=name, **spec_kw
        )
        return build_direct_conv(spec, arch=arch)

    return _build


def build_direct_mfma_dgrad_fused(
    name,
    arch,
    N,
    H,
    W,
    groups,
    cpg,
    kpg,
    *,
    dtype,
    fold_k32,
    weights_lds,
    block_groups=2,
    block_h=8,
    knobs=None,
):
    """Single-kernel generic direct-MFMA dgrad (DirectConvSpec over the
    transposed problem, fused weights, row guard); ``knobs`` adds further
    DirectConvSpec fields."""

    def _build():
        from dataclasses import replace

        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            build_direct_conv,
            make_dgrad_fprop_spec,
        )

        p = DirectConvProblem(
            N=N, H=H, W=W, groups=groups, cpg=cpg, kpg=kpg, dtype=dtype
        )
        spec = make_dgrad_fprop_spec(
            p,
            block_q=16,
            block_groups=block_groups,
            block_h=block_h,
            fold_k32=fold_k32,
            dgrad_fused_weights=True,
            dgrad_weights_lds=weights_lds,
            row_guard=True,
        )
        spec = replace(spec, name=name, **(knobs or {}))
        return build_direct_conv(spec, arch=arch)

    return _build


def build_direct_8c(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    *,
    block_q=16,
    block_groups=8,
    double_buffer=True,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConv8cSpec,
            DirectConvProblem,
            build_direct_conv_8c,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=8,
            kpg=8,
            KH=KH,
            KW=KW,
            PAD=PAD,
        )
        spec = DirectConv8cSpec(
            problem=p,
            name=name,
            block_q=block_q,
            block_groups=block_groups,
            double_buffer=double_buffer,
        )
        return build_direct_conv_8c(spec, arch=arch)

    return _build


def build_direct_32c(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    *,
    block_q=32,
    block_groups=4,
    double_buffer=True,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConv32cSpec,
            DirectConvProblem,
            build_direct_conv_32c,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=32,
            kpg=32,
            KH=KH,
            KW=KW,
            PAD=PAD,
        )
        spec = DirectConv32cSpec(
            problem=p,
            name=name,
            block_q=block_q,
            block_groups=block_groups,
            double_buffer=double_buffer,
        )
        return build_direct_conv_32c(spec, arch=arch)

    return _build


def build_direct_depthwise(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    *,
    block_w=16,
    block_waves=2,
    wave_size=64,
    cpl=1,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseSpec,
            build_direct_depthwise as _build_dw,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KW,
            PAD=PAD,
        )
        spec = DirectDepthwiseSpec(
            problem=p,
            name=name,
            block_w=block_w,
            block_waves=block_waves,
            wave_size=wave_size,
            cpl=cpl,
        )
        return _build_dw(spec, arch=arch)

    return _build


def build_direct_conv_dgrad(
    name,
    arch,
    N,
    H,
    W,
    groups,
    cpg,
    kpg,
    KH=3,
    KW=3,
    PAD=1,
    stride=1,
    *,
    block_q=16,
    block_groups=8,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvDgradSpec,
            DirectConvProblem,
            build_direct_conv_dgrad as _build_dgrad,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=cpg,
            kpg=kpg,
            KH=KH,
            KW=KW,
            PAD=PAD,
            stride=stride,
        )
        spec = DirectConvDgradSpec(
            problem=p,
            name=name,
            block_q=block_q,
            block_groups=block_groups,
        )
        return _build_dgrad(spec, arch=arch)

    return _build


def build_direct_depthwise_dgrad(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    stride=1,
    *,
    block_w=8,
    block_waves=1,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseDgradSpec,
            build_direct_depthwise_dgrad as _build_dw_dgrad,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KW,
            PAD=PAD,
            stride=stride,
        )
        spec = DirectDepthwiseDgradSpec(
            problem=p,
            name=name,
            block_w=block_w,
            block_waves=block_waves,
        )
        return _build_dw_dgrad(spec, arch=arch)

    return _build


def build_direct_depthwise_dgrad_stream(
    name, arch, N, H, W, groups, KH, PAD, stride=1, dtype="fp16", **spec_kw
):
    """Row-streaming depthwise dgrad (DirectDepthwiseDgradStreamSpec);
    ``spec_kw`` overrides its fields (block_w, block_waves, wave_size, cpl).
    Mirrors parity/conv_direct_grouped_emit.* (indices 171+)."""

    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseDgradStreamSpec,
            build_direct_depthwise_dgrad_streaming,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KH,
            PAD=PAD,
            stride=stride,
            dtype=dtype,
        )
        spec = DirectDepthwiseDgradStreamSpec(problem=p, name=name, **spec_kw)
        return build_direct_depthwise_dgrad_streaming(spec, arch=arch)

    return _build


def build_direct_depthwise_dgrad_win(
    name, arch, N, H, W, groups, KH, PAD, dtype, **knobs
):
    """Windowed depthwise dgrad (stride 1); ``knobs`` are the spec's."""

    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseDgradWindowedSpec,
            build_direct_depthwise_dgrad_windowed,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KH,
            PAD=PAD,
            stride=1,
            dtype=dtype,
        )
        spec = DirectDepthwiseDgradWindowedSpec(problem=p, name=name, **knobs)
        return build_direct_depthwise_dgrad_windowed(spec, arch=arch)

    return _build


def build_direct_nongrouped(
    name,
    arch,
    N,
    H,
    W,
    C,
    K,
    KH=3,
    KW=3,
    PAD=1,
    stride=1,
    dtype="bf16",
    dil_h=1,
    dil_w=1,
    **spec_kw,
):
    """Non-grouped (groups == 1) direct conv; ``spec_kw`` overrides
    DirectNongroupedConvSpec fields. Configs mirror parity/conv_direct_grouped_emit.*
    (indices 42+), where the C++ engine is gated byte-identical by
    check_byte_identity.py."""

    def _build():
        from kernels.common.conv_direct_grouped import DirectConvProblem
        from kernels.common.conv_direct_nongrouped import (
            DirectNongroupedConvSpec,
            build_direct_conv_nongrouped,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=1,
            cpg=C,
            kpg=K,
            KH=KH,
            KW=KW,
            PAD=PAD,
            stride=stride,
            dtype=dtype,
            dil_h=dil_h,
            dil_w=dil_w,
        )
        spec = DirectNongroupedConvSpec(problem=p, name=name, **spec_kw)
        return build_direct_conv_nongrouped(spec, arch=arch)

    return _build


def build_direct_wgrad(name, arch, kind, problem, **spec_kw):
    """A direct wgrad kernel: ``kind`` is tiled / dw_lds / dw_cols / gather /
    dw_mfma / grouped_mfma / fold, or the gfx1250 wave32 image (and small-C)
    ``w32`` and depthwise ``dw_valu`` kernels; ``problem`` overrides the
    DirectConvProblem fields (unused by the fold). Configs mirror
    parity/conv_direct_grouped_emit.* (indices 95+ and 201+), where the C++
    engine is gated byte-identical by check_byte_identity.py."""

    def _build():
        from kernels.common import conv_direct_dw_wgrad as dw
        from kernels.common import conv_direct_dw_wgrad_mfma as dwm
        from kernels.common import conv_direct_grouped_wgrad_mfma as gwm
        from kernels.common import conv_direct_wgrad_gather as wg
        from kernels.common import conv_direct_wgrad_tiled as wt
        from kernels.common import conv_direct_grouped as dc
        from kernels.common.conv_direct_grouped import DirectConvProblem

        if kind == "fold":
            return dw.build_dw_wgrad_fold(
                dw.DwWgradFoldSpec(name=name, **spec_kw), arch=arch
            )
        kw = dict(
            N=2,
            H=16,
            W=32,
            groups=1,
            cpg=64,
            kpg=64,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            dtype="bf16",
        )
        kw.update(problem)
        p = DirectConvProblem(**kw)
        spec_cls, build = {
            "tiled": (wt.DirectWgradTiledSpec, wt.build_direct_wgrad_tiled),
            "dw_lds": (dw.DwWgradLdsSpec, dw.build_dw_wgrad_lds),
            "dw_cols": (dw.DwWgradColsSpec, dw.build_dw_wgrad_cols),
            "gather": (wg.WgradGatherSpec, wg.build_wgrad_gather),
            "dw_mfma": (dwm.DwWgradMfmaSpec, dwm.build_dw_wgrad_mfma),
            "grouped_mfma": (
                gwm.GroupedWgradMfmaSpec,
                gwm.build_grouped_wgrad_mfma,
            ),
            "w32": (dc.DirectConvWgradSpec, dc.build_direct_conv_wgrad),
            "dw_valu": (dc.DirectDepthwiseWgradSpec, dc.build_direct_depthwise_wgrad),
        }[kind]
        return build(spec_cls(problem=p, name=name, **spec_kw), arch=arch)

    return _build


def build_direct_depthwise_tiled(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH,
    PAD,
    stride=1,
    dtype="bf16",
    KW=None,
    pad_w=-1,
    **spec_kw,
):
    """Output-stationary depthwise direct conv; ``spec_kw`` overrides
    DirectDepthwiseTiledSpec fields. Configs mirror parity/conv_direct_grouped_emit.*
    (indices 57+). ``KW`` defaults to ``KH``; ``pad_w`` (-1: ``PAD``) is the
    column padding of the rotated-filter dgrad mode (``dgrad=True``)."""

    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseTiledSpec,
            build_direct_depthwise_tiled,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KH if KW is None else KW,
            PAD=PAD,
            stride=stride,
            dtype=dtype,
            pad_w=pad_w,
        )
        spec = DirectDepthwiseTiledSpec(problem=p, name=name, **spec_kw)
        return build_direct_depthwise_tiled(spec, arch=arch)

    return _build


def build_direct_depthwise_col(
    name,
    arch,
    N,
    H,
    W,
    groups,
    KH=3,
    KW=3,
    PAD=1,
    stride=1,
    *,
    block_h=16,
    block_w=4,
    block_waves=1,
    dtype="fp16",
    wave_size=64,
):
    def _build():
        from kernels.common.conv_direct_grouped import (
            DirectConvProblem,
            DirectDepthwiseColSpec,
            build_direct_depthwise_col as _build_dwcol,
        )

        p = DirectConvProblem(
            N=N,
            H=H,
            W=W,
            groups=groups,
            cpg=1,
            kpg=1,
            KH=KH,
            KW=KW,
            PAD=PAD,
            stride=stride,
        )
        spec = DirectDepthwiseColSpec(
            problem=p,
            name=name,
            block_h=block_h,
            block_w=block_w,
            block_waves=block_waves,
            dtype=dtype,
            wave_size=wave_size,
        )
        return _build_dwcol(spec, arch=arch)

    return _build


def cases():
    out = []

    def add(family, case_id, arch, build):
        out.append({"family": family, "case_id": case_id, "arch": arch, "build": build})

    # Conv: problem-shape and arch variants.
    conv1 = (1, 8, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1)
    conv2 = (2, 16, 16, 32, 32, 1, 1, 1, 1, 0, 0, 1, 1)
    add(
        "conv",
        "conv/gfx942/n1h8c16k32r3",
        "gfx942",
        build_conv(
            "irhash_conv_942_a",
            "gfx942",
            conv1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3",
        "gfx950",
        build_conv(
            "irhash_conv_950_a",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="compv4",
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx950/n2h16c32k32r1",
        "gfx950",
        build_conv(
            "irhash_conv_950_b",
            "gfx950",
            conv2,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx1151/n1h8c16k32r3",
        "gfx1151",
        build_conv(
            "irhash_conv_1151_a",
            "gfx1151",
            conv1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx1151/n2h16c32k32r1",
        "gfx1151",
        build_conv(
            "irhash_conv_1151_b",
            "gfx1151",
            conv2,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv",
        "conv/gfx1201/n1h8c16k32r3",
        "gfx1201",
        build_conv(
            "irhash_conv_1201_a",
            "gfx1201",
            conv1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    # gfx90a conv mirrors the gfx942 MFMA path (wave64, 16x16x16 atom). gfx1250
    # has no conv case: WMMA conv requires the 16x16x16 atom, but gfx1250's fp16
    # warp_tile is 16x16x32, so the two constraints are mutually exclusive.
    add(
        "conv",
        "conv/gfx90a/n1h8c16k32r3",
        "gfx90a",
        build_conv(
            "irhash_conv_90a_a",
            "gfx90a",
            conv1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    # pipeline=basic: single-buffer global-read/compute overlap on gfx950.
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3/basic_default",
        "gfx950",
        build_conv(
            "irhash_conv_950_basic_a",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="basic",
            epilogue="default",
            vector_size_c=1,
        ),
    )
    add(
        "conv",
        "conv/gfx950/n1h8c16k32r3/basic_cshuffle",
        "gfx950",
        build_conv(
            "irhash_conv_950_basic_b",
            "gfx950",
            conv1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="basic",
            epilogue="cshuffle",
        ),
    )

    # Conv forward + merged groups (group_merge).
    #
    # Merging folds Gm conv groups into one GEMM so the vector-width picker
    # sees Gm channels per group instead of 1. It is admissible only on
    # depthwise shapes (cpg == kpg == 1), so these cases need their own
    # problem: conv1/conv2 above are dense and would be rejected by the gate.
    #
    # conv_dw is depthwise -- C == K == groups == 64, so cpg == kpg == 1 and
    # M == 2*14*14 == 392. The variants below span the axes that actually
    # change emitted IR under merge, which is what makes the golden a
    # regression gate rather than a single smoke hash:
    #   - dw_unmerged is the group_merge=1 control. Without it a drift in the
    #     merged hashes could not be attributed to the merge path rather than
    #     to the depthwise path, which carried no coverage here at all.
    #   - dw_unmerged_cshuffle pins the Gm=1 cshuffle store width: it derives
    #     from kpg (_vec(1) = 1 here) rather than a fixed 8, which failed the
    #     whole-vector bound and dropped every output. Python twin of parity
    #     config 28.
    #   - gm4/gm8/gm32 vary the shift and mask width (log2 Gm) of the merged
    #     k decode that feeds both the A channel and the B-load diagonal.
    #   - gm64 is Gm == groups: one merged group, but the grouped decode stays
    #     engaged because an AOT binary serves every group count Gm divides.
    #   - dw_gm4_direct pins vector_size_c=1 to reach _emit_direct_epilogue.
    #     Merged kpg otherwise auto-derives vec_c > 1, which the validator
    #     turns into a cshuffle requirement, so without the pin every merged
    #     case here would exercise only the cshuffle epilogue.
    # The gfx942 case checks that merging is arch-independent: it shares the
    # emitter path with gfx950 and differs only in atom and tile shape.
    conv_dw = (2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 64)
    for tag, gm, epi, vec_c in (
        ("dw_unmerged", 1, "default", None),
        ("dw_unmerged_cshuffle", 1, "cshuffle", None),
        ("dw_gm4_direct", 4, "default", 1),
        ("dw_gm8_cshuffle", 8, "cshuffle", None),
        ("dw_gm32_cshuffle", 32, "cshuffle", None),
        ("dw_gm64_cshuffle", 64, "cshuffle", None),
    ):
        add(
            "conv",
            f"conv/gfx950/n2h14c64k64r3g64/{tag}",
            "gfx950",
            build_conv(
                f"irhash_conv_950_{tag}",
                "gfx950",
                conv_dw,
                wave_size=64,
                wtm=32,
                wtn=32,
                wtk=16,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                pipeline="mem",
                epilogue=epi,
                vector_size_c=vec_c,
                group_merge=gm,
            ),
        )
    add(
        "conv",
        "conv/gfx942/n2h14c64k64r3g64/dw_gm8_cshuffle",
        "gfx942",
        build_conv(
            "irhash_conv_942_dw_gm8",
            "gfx942",
            conv_dw,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
            group_merge=8,
        ),
    )
    # gfx1250 WMMA (wave32, 16x16x32): the depthwise problem unmerged
    # (grid-per-group) and merged on both epilogues, plus a dense-grouped
    # problem (groups=4, cpg=kpg=16) unmerged.
    conv_g4 = (2, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 4)
    for tag, prob, gm, epi, vec_c in (
        ("n2h14c64k64r3g64/dw_unmerged", conv_dw, 1, "default", None),
        ("n2h14c64k64r3g64/dw_gm4_direct", conv_dw, 4, "default", 1),
        ("n2h14c64k64r3g64/dw_gm8_cshuffle", conv_dw, 8, "cshuffle", None),
        ("n2h14c64k64r3g4/g4_cshuffle", conv_g4, 1, "cshuffle", None),
    ):
        add(
            "conv",
            f"conv/gfx1250/{tag}",
            "gfx1250",
            build_conv(
                "irhash_conv_1250_" + tag.split("/")[1],
                "gfx1250",
                prob,
                wave_size=32,
                wtm=16,
                wtn=16,
                wtk=32,
                tile_m=32,
                tile_n=32,
                tile_k=32,
                epilogue=epi,
                vector_size_c=vec_c,
                group_merge=gm,
            ),
        )

    # gfx1250 split-K forward (WMMA): the two-stage binary (fp32 workspace
    # slabs, degree as a kernarg) on mem/cshuffle and mem/default, and the
    # single-launch cluster form (S=4) that folds the slabs through a cluster
    # barrier. Small M, deep K: 9*256 reduction.
    conv_small_m = (1, 7, 7, 256, 64, 3, 3, 1, 1, 1, 1, 1, 1)
    for tag, epi, vec_c, skc in (
        ("splitk_cshuffle", "cshuffle", None, 0),
        ("splitk_default", "default", 1, 0),
        ("splitk_cluster4", "cshuffle", None, 4),
    ):
        add(
            "conv",
            f"conv/gfx1250/n1h7c256k64r3/{tag}",
            "gfx1250",
            build_conv(
                "irhash_conv_1250_" + tag,
                "gfx1250",
                conv_small_m,
                wave_size=32,
                wtm=16,
                wtn=16,
                wtk=32,
                tile_m=32,
                tile_n=32,
                tile_k=32,
                epilogue=epi,
                vector_size_c=vec_c,
                split_k=2,
                split_k_cluster=skc,
            ),
        )
    # gfx950 split-K forward (MFMA, wave64): the same two-stage binary on the
    # plain mem / compv3 K loops, one per dtype.
    for tag, pipeline, dtype in (
        ("splitk_mem_fp16", "mem", "fp16"),
        ("splitk_compv3_bf16", "compv3", "bf16"),
    ):
        add(
            "conv",
            f"conv/gfx950/n1h7c256k64r3/{tag}",
            "gfx950",
            build_conv(
                "irhash_conv_950_" + tag,
                "gfx950",
                conv_small_m,
                wave_size=64,
                wtm=32,
                wtn=32,
                wtk=16,
                tile_m=64,
                tile_n=64,
                tile_k=64,
                pipeline=pipeline,
                epilogue="cshuffle",
                split_k=2,
                dtype=dtype,
            ),
        )
    # Its Stage 2: the runtime slab loop and the degree-unrolled fold.
    for tag, kw in (
        ("bf16_v4", dict(dtype_d="bf16", vec=4)),
        ("fp16_v2_s4", dict(dtype_d="fp16", vec=2, degree=4)),
    ):
        add(
            "conv_fwd_split_k_reduce",
            f"conv_fwd_split_k_reduce/gfx1250/{tag}",
            "gfx1250",
            build_conv_fwd_split_k_reduce("gfx1250", **kw),
        )

    # Conv wgrad: implicit-GEMM backward-weight direction (NHWK x NHWC -> KYXC).
    # wgrad1: small 3x3 conv (N8 H8 W8 C16 K32 Y3 X3, stride/pad=defaults).
    # wgrad2: 1x1 conv (N2 H16 W16 C32 K32, no filter spatial).
    wgrad1 = (1, 8, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1)
    wgrad2 = (2, 16, 16, 32, 32, 1, 1, 1, 1, 0, 0, 1, 1)
    # wgrad_g4: grouped 3x3 conv (N1 H8 W8 C64 K64 Y3 X3, groups=4 -> cpg=kpg=16).
    # 14th tuple element sets ConvProblem.groups. Exercises the grouped
    # (grid-per-group) dW IR.
    wgrad_g4 = (1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 4)
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c16k32r3",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_a",
            "gfx942",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_a",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            pipeline="mem",
            epilogue="cshuffle",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n2h16c32k32r1",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_b",
            "gfx950",
            wgrad2,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_spk4",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_spk4",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            split_k=4,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1151/n1h8c16k32r3",
        "gfx1151",
        build_conv_wgrad(
            "irhash_wgrad_1151_a",
            "gfx1151",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            dtype_d="fp32",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1201/n1h8c16k32r3",
        "gfx1201",
        build_conv_wgrad(
            "irhash_wgrad_1201_a",
            "gfx1201",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
            dtype_d="fp32",
        ),
    )
    # gfx90a wgrad mirrors the gfx942 MFMA path.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx90a/n1h8c16k32r3",
        "gfx90a",
        build_conv_wgrad(
            "irhash_wgrad_90a_a",
            "gfx90a",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    # Grouped wgrad, grid-per-group (groups=4, cpg=kpg=16). Guards the per-group
    # dW IR against silent drift. MFMA-only, so gfx942/gfx950.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c64k64r3_g4",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_g4",
            "gfx942",
            wgrad_g4,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            epilogue="cshuffle",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            epilogue="cshuffle",
        ),
    )
    # Stream-K wgrad: the linear and tree flag fixups, the atomic reduction,
    # the persistent launch and the group fold into GEMM-M. MFMA-only.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_sk_linear",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_sk_linear",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            streamk="dp_sk",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_sk_atomic",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_sk_atomic",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            dtype_d="fp32",
            streamk="dp_sk",
            streamk_reduction="atomic",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c64k64r3_g4_sk_tree_pers",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_g4_sk_tree_pers",
            "gfx942",
            wgrad_g4,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            streamk="persistent",
            streamk_reduction="tree",
        ),
    )
    # The remaining {dp_sk, persistent} x {linear, tree, atomic, workspace}
    # combinations, so every fixup is pinned under both launch modes.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_sk_tree",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_sk_tree",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            streamk="dp_sk",
            streamk_reduction="tree",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c64k64r3_g4_sk_workspace",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_g4_sk_workspace",
            "gfx942",
            wgrad_g4,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            streamk="dp_sk",
            streamk_reduction="workspace",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_sk_linear_pers",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_sk_linear_pers",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            streamk="persistent",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c16k32r3_sk_atomic_pers",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_sk_atomic_pers",
            "gfx942",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            dtype_d="fp32",
            streamk="persistent",
            streamk_reduction="atomic",
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c16k32r3_sk_workspace_pers",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_sk_workspace_pers",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            streamk="persistent",
            streamk_reduction="workspace",
        ),
    )
    # Grouped + cshuffle epilogue (MFMA): the LDS-staged store threads the
    # per-group k_out fold (group*kpg) and derives its store-vec from cpg.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_cshuffle",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_cshuffle",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            epilogue="cshuffle",
        ),
    )
    # Grouped + split-K + cshuffle (MFMA): the group and the K-slice share
    # block_id_z (z = groups*split_k); cshuffle required for fp16 paired atomics.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_spk4",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_spk4",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            split_k=4,
            epilogue="cshuffle",
        ),
    )
    # The register-prefetch K loop (prefetch_global_regs): the plain synchronous
    # loop with tile k+1's reads held in VGPRs under tile k's MFMAs, behind
    # LDS-only barriers. K-outer cshuffle without and with the packed fp16
    # atomic (the 16x16x32 atom, compv3), and two-stage into bf16 dW.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_pgr",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_pgr",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            epilogue="cshuffle",
            lds_k_outer=True,
            prefetch_global_regs=True,
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_spk4_pgr",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_spk4_pgr",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            pipeline="compv3",
            split_k=4,
            epilogue="cshuffle",
            lds_k_outer=True,
            prefetch_global_regs=True,
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx950/n1h8c64k64r3_g4_2stage_bf16_pgr",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_g4_2stage_bf16_pgr",
            "gfx950",
            wgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=64,
            pipeline="compv4",
            split_k=4,
            two_stage=True,
            dtype_d="bf16",
            lds_k_outer=True,
            prefetch_global_regs=True,
        ),
    )
    add(
        "conv_wgrad",
        "conv_wgrad/gfx942/n1h8c64k64r3_g4_spk4",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_g4_spk4",
            "gfx942",
            wgrad_g4,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            split_k=4,
            epilogue="cshuffle",
        ),
    )
    # gfx1250 WMMA grouped (wave32, 16x16x32 -- its only fp16/bf16 atom).
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1250/n1h8c64k64r3_g4",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_g4",
            "gfx1250",
            wgrad_g4,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            dtype_d="fp32",
        ),
    )
    # WMMA split-K: the packed bf16 atomic with a zero partner.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1250/n1h8c16k32r3_spk4",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_spk4",
            "gfx1250",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            split_k=4,
            dtype_d="bf16",
        ),
    )
    # WMMA split-K over the gfx1250 async K loop (global_load_async_to_lds).
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1250/n1h8c16k32r3_spk4_async",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_spk4_async",
            "gfx1250",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            split_k=4,
            dtype_d="bf16",
            lds_k_outer=True,
            async_dma=True,
        ),
    )
    # The synchronous K-outer loop with explicit A/B load-width caps (4 and 2
    # under the auto 8/8), which both engines apply to the free-axis loads.
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1250/n1h8c16k32r3_spk4_kouter_vec4x2",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_spk4_kouter_vec4x2",
            "gfx1250",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            split_k=4,
            dtype_d="bf16",
            lds_k_outer=True,
            vector_size_a=4,
            vector_size_b=2,
        ),
    )
    # WMMA split-K through the cshuffle epilogue (paired atomics from LDS).
    add(
        "conv_wgrad",
        "conv_wgrad/gfx1250/n1h8c16k32r3_spk4_cshuffle",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_spk4_cshuffle",
            "gfx1250",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            split_k=4,
            dtype_d="fp16",
            epilogue="cshuffle",
        ),
    )

    # Dgrad: backward-data implicit-GEMM.  Problem args match the forward conv
    # set; atoms chosen to be valid on each arch.  Stride=2 exercises the tiled
    # (multi-sub-GEMM) path; stride=1 exercises the direct-store epilogue.
    #
    # ConvProblem positional args: (N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW)
    dgrad1 = (1, 8, 8, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1)  # stride=1
    dgrad2 = (2, 16, 16, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1)  # stride=2, 4 sub-GEMMs
    # Grouped dgrad (14th tuple element = ConvProblem.groups). C=K=64, groups=4
    # -> cpg=kpg=16; grid-per-group on blockIdx.y (byte-identical to groups=1
    # for the ungrouped path).
    dgrad_g4 = (1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 4)  # stride=1 grouped
    dgrad_g4_s2 = (1, 8, 8, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 4)  # stride=2 grouped tilde
    add(
        "conv_dgrad",
        "conv_dgrad/gfx942/n1h8c16k32r3_s1",
        "gfx942",
        build_dgrad(
            "irhash_dgrad_942_s1",
            "gfx942",
            dgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=8,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    add(
        "conv_dgrad",
        "conv_dgrad/gfx942/n2h16c32k32r3_s2",
        "gfx942",
        build_dgrad(
            "irhash_dgrad_942_s2",
            "gfx942",
            dgrad2,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
        ),
    )
    add(
        "conv_dgrad",
        "conv_dgrad/gfx950/n1h8c16k32r3_s1",
        "gfx950",
        build_dgrad(
            "irhash_dgrad_950_s1",
            "gfx950",
            dgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    add(
        "conv_dgrad",
        "conv_dgrad/gfx950/n2h16c32k32r3_s2",
        "gfx950",
        build_dgrad(
            "irhash_dgrad_950_s2",
            "gfx950",
            dgrad2,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    # gfx1151 WMMA (wave32) -- stride=1 only; WMMA dgrad supports only 16x16x16.
    add(
        "conv_dgrad",
        "conv_dgrad/gfx1151/n1h8c16k32r3_s1",
        "gfx1151",
        build_dgrad(
            "irhash_dgrad_1151_s1",
            "gfx1151",
            dgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    # gfx90a mirrors gfx942 (wave64 MFMA).
    add(
        "conv_dgrad",
        "conv_dgrad/gfx90a/n1h8c16k32r3_s1",
        "gfx90a",
        build_dgrad(
            "irhash_dgrad_90a_s1",
            "gfx90a",
            dgrad1,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=8,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    # Grouped dgrad (groups=4, cpg=kpg=16) -- grid-per-group on blockIdx.y.
    add(
        "conv_dgrad",
        "conv_dgrad/gfx942/n1h8c64k64r3_g4",
        "gfx942",
        build_dgrad(
            "irhash_dgrad_942_g4_s1",
            "gfx942",
            dgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=8,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    add(
        "conv_dgrad",
        "conv_dgrad/gfx942/n1h8c64k64r3_g4_s2",
        "gfx942",
        build_dgrad(
            "irhash_dgrad_942_g4_s2",
            "gfx942",
            dgrad_g4_s2,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=32,
        ),
    )
    add(
        "conv_dgrad",
        "conv_dgrad/gfx950/n1h8c64k64r3_g4",
        "gfx950",
        build_dgrad(
            "irhash_dgrad_950_g4_s1",
            "gfx950",
            dgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=64,
        ),
    )
    # The stride-1 tap-outer K loop (tap_outer_k) of the same kernel.
    add(
        "conv_dgrad",
        "conv_dgrad/gfx950/n1h8c64k64r3_g4_tapk",
        "gfx950",
        build_dgrad(
            "irhash_dgrad_950_g4_s1_tapk",
            "gfx950",
            dgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            tap_outer_k=True,
        ),
    )
    # The dY halo mode of that loop (dy_halo=2), sized for Wo <= 8.
    add(
        "conv_dgrad",
        "conv_dgrad/gfx950/n1h8c64k64r3_g4_halo2",
        "gfx950",
        build_dgrad(
            "irhash_dgrad_950_g4_s1_halo2",
            "gfx950",
            dgrad_g4,
            wave_size=64,
            wtm=32,
            wtn=32,
            wtk=16,
            tile_m=64,
            tile_n=64,
            tile_k=16,
            tap_outer_k=True,
            halo={"dy_halo": 2, "halo_max_wo": 8},
        ),
    )
    # gfx1151 WMMA grouped (wave32, 16x16x16) -- stride=1 only.
    add(
        "conv_dgrad",
        "conv_dgrad/gfx1151/n1h8c64k64r3_g4",
        "gfx1151",
        build_dgrad(
            "irhash_dgrad_1151_g4_s1",
            "gfx1151",
            dgrad_g4,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=32,
            tile_n=32,
            tile_k=16,
        ),
    )
    # gfx1250 WMMA grouped (wave32, 16x16x32 -- its only fp16/bf16 atom).
    add(
        "conv_dgrad",
        "conv_dgrad/gfx1250/n1h8c64k64r3_g4",
        "gfx1250",
        build_dgrad(
            "irhash_dgrad_1250_g4_s1",
            "gfx1250",
            dgrad_g4,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
        ),
    )

    # TF32 compute (fp32 storage, mma_dtype="tf32") on the implicit-GEMM fwd,
    # dgrad and wgrad: gfx942 issues the native XF32 MFMA, gfx950 the BF16x3
    # split (three bf16 MFMAs per logical atom). Between them the cases cover
    # all four TF32 atoms. check_byte_identity.py holds the C++ builders to
    # the same IR (the TF32 configs of the conv_implicit_gemm* parity families);
    # this pins it against drift in both engines.
    for _arch, _fwd_atom, _bwd_atom in (
        ("gfx942", (32, 4), (16, 8)),
        ("gfx950", (16, 32), (32, 16)),
    ):
        _a = _arch[3:]
        _fwd = dict(wtm=_fwd_atom[0], wtn=_fwd_atom[0], wtk=_fwd_atom[1])
        _bwd = dict(wtm=_bwd_atom[0], wtn=_bwd_atom[0], wtk=_bwd_atom[1])
        _tile = dict(wave_size=64, tile_m=64, tile_n=64, tile_k=32, tf32=True)
        add(
            "conv",
            f"conv/{_arch}/n1h8c16k32r3_tf32",
            _arch,
            build_conv(
                f"irhash_conv_{_a}_tf32",
                _arch,
                conv1,
                epilogue="cshuffle",
                **_fwd,
                **_tile,
            ),
        )
        add(
            "conv_dgrad",
            f"conv_dgrad/{_arch}/n1h8c16k32r3_s1_tf32",
            _arch,
            build_dgrad(f"irhash_dgrad_{_a}_s1_tf32", _arch, dgrad1, **_bwd, **_tile),
        )
        add(
            "conv_wgrad",
            f"conv_wgrad/{_arch}/n1h8c16k32r3_tf32",
            _arch,
            build_conv_wgrad(f"irhash_wgrad_{_a}_tf32", _arch, wgrad1, **_bwd, **_tile),
        )

    # Two-stage wgrad Stage 1: workspace-store epilogue instead of atomic-add.
    # split_k=4, two_stage=True — exercises the _emit_wgrad_workspace_store_epilogue
    # path.  The problem/tile params match the existing spk4 case for comparability.
    add(
        "conv_wgrad_two_stage",
        "conv_wgrad_two_stage/gfx950/n1h8c16k32r3_spk4",
        "gfx950",
        build_conv_wgrad(
            "irhash_wgrad_950_twostage_spk4",
            "gfx950",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            split_k=4,
            two_stage=True,
        ),
    )
    add(
        "conv_wgrad_two_stage",
        "conv_wgrad_two_stage/gfx942/n1h8c16k32r3_spk4",
        "gfx942",
        build_conv_wgrad(
            "irhash_wgrad_942_twostage_spk4",
            "gfx942",
            wgrad1,
            wave_size=64,
            wtm=16,
            wtn=16,
            wtk=16,
            tile_m=64,
            tile_n=32,
            tile_k=16,
            split_k=4,
            two_stage=True,
        ),
    )

    # gfx1250 WMMA: the scratch epilogue decodes the WMMA C fragment.
    add(
        "conv_wgrad_two_stage",
        "conv_wgrad_two_stage/gfx1250/n1h8c16k32r3_spk4",
        "gfx1250",
        build_conv_wgrad(
            "irhash_wgrad_1250_twostage_spk4",
            "gfx1250",
            wgrad1,
            wave_size=32,
            wtm=16,
            wtn=16,
            wtk=32,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            split_k=4,
            two_stage=True,
        ),
    )

    # Workspace reduce Stage 2: reads f32 partial sums written by Stage 1 and
    # reduces along split_k, emitting the result as dtype_d.
    # wg_M=32, wg_N=72 (3*3*8): small 3x3 filter footprint, fp16 output.
    add(
        "conv_wgrad_reduce",
        "conv_wgrad_reduce/gfx950/wgM32_wgN72_fp16",
        "gfx950",
        build_conv_wgrad_reduce(
            "irhash_wgrad_reduce_950_m32n72_fp16",
            "gfx950",
            wg_M=32,
            wg_N=72,
            dtype_d="fp16",
        ),
    )
    # wg_M=64, wg_N=576 (3*3*64): larger filter footprint, bf16 output.
    add(
        "conv_wgrad_reduce",
        "conv_wgrad_reduce/gfx950/wgM64_wgN576_bf16",
        "gfx950",
        build_conv_wgrad_reduce(
            "irhash_wgrad_reduce_950_m64n576_bf16",
            "gfx950",
            wg_M=64,
            wg_N=576,
            dtype_d="bf16",
        ),
    )
    add(
        "conv_wgrad_reduce",
        "conv_wgrad_reduce/gfx942/wgM32_wgN72_fp16",
        "gfx942",
        build_conv_wgrad_reduce(
            "irhash_wgrad_reduce_942_m32n72_fp16",
            "gfx942",
            wg_M=32,
            wg_N=72,
            dtype_d="fp16",
        ),
    )

    # conv_direct: parametric direct grouped convolution.
    # Covers all cpg variants (4c, 8c, 16c, 32c, depthwise) on gfx942 and gfx950.
    # The cases mirror the six configs in parity/conv_direct_grouped_emit.py and
    # extend them with 8c, 32c, and depthwise.
    #
    # Problem sizes are kept small (H=W=8 or 32) to compile fast on CI.
    # The gfx950 16c cases exercise fold_k32=True (uses mfma_f32_16x16x32_f16).
    # The gfx942 16c case uses fold_k32=False (mfma_f32_16x16x16_f16 only).
    #
    # C++ engine parity: tracked for conv_direct_dgrad (see cases below);
    # the 16c/4c/8c/32c/depthwise fprop variants have a C++ port but their
    # golden is pinned to the Python emitter only (no separate cpp-vs-python
    # differential gate for the fprop families in this harness).

    # --- 16c, gfx950, fold_k32=True (matches parity emit idx=0 and idx=1) ---
    add(
        "conv_direct",
        "conv_direct/gfx950/16c_n32h8_bg4_fold",
        "gfx950",
        build_direct_16c(
            "irhash_direct16c_950_bg4",
            "gfx950",
            N=32,
            H=8,
            W=8,
            groups=16,
            KH=3,
            KW=3,
            PAD=1,
            block_groups=4,
            fold_k32=True,
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/16c_n32h8_bg8_fold",
        "gfx950",
        build_direct_16c(
            "irhash_direct16c_950_bg8",
            "gfx950",
            N=32,
            H=8,
            W=8,
            groups=16,
            KH=3,
            KW=3,
            PAD=1,
            block_groups=8,
            fold_k32=True,
        ),
    )
    # --- 16c, gfx942, fold_k32=False (matches parity emit idx=4) ---
    add(
        "conv_direct",
        "conv_direct/gfx942/16c_n1h8_fold_false",
        "gfx942",
        build_direct_16c(
            "irhash_direct16c_942_a",
            "gfx942",
            N=1,
            H=8,
            W=8,
            groups=8,
            KH=3,
            KW=3,
            PAD=1,
            block_groups=4,
            fold_k32=False,
        ),
    )
    # --- 4c, gfx950 (matches parity emit idx=2 and idx=3) ---
    add(
        "conv_direct",
        "conv_direct/gfx950/4c_n32h8_bq4",
        "gfx950",
        build_direct_4c(
            "irhash_direct4c_950_bq4",
            "gfx950",
            N=32,
            H=8,
            W=8,
            groups=64,
            KH=3,
            KW=3,
            PAD=1,
            block_q=4,
            block_groups=16,
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/4c_n1h8_small",
        "gfx950",
        build_direct_4c(
            "irhash_direct4c_950_small",
            "gfx950",
            N=1,
            H=8,
            W=8,
            groups=16,
            KH=3,
            KW=3,
            PAD=1,
            block_q=4,
            block_groups=16,
        ),
    )
    # --- 4c bf16 (mfma_f32_4x4x4_bf16), gfx950 and gfx942 ---
    add(
        "conv_direct",
        "conv_direct/gfx950/4c_bf16_n2h13",
        "gfx950",
        build_direct_4c(
            "irhash_direct4c_950_bf16",
            "gfx950",
            N=2,
            H=13,
            W=13,
            groups=32,
            block_q=4,
            block_groups=16,
            dtype="bf16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx942/4c_bf16_n1h8",
        "gfx942",
        build_direct_4c(
            "irhash_direct4c_942_bf16",
            "gfx942",
            N=1,
            H=8,
            W=8,
            groups=16,
            block_q=8,
            block_groups=16,
            dtype="bf16",
        ),
    )
    # --- 8c, gfx950 ---
    add(
        "conv_direct",
        "conv_direct/gfx950/8c_n4h16",
        "gfx950",
        build_direct_8c(
            "irhash_direct8c_950_a",
            "gfx950",
            N=4,
            H=16,
            W=16,
            groups=8,
            KH=3,
            KW=3,
            PAD=1,
            block_q=16,
            block_groups=8,
        ),
    )
    # --- 8c, gfx942 ---
    add(
        "conv_direct",
        "conv_direct/gfx942/8c_n4h16",
        "gfx942",
        build_direct_8c(
            "irhash_direct8c_942_a",
            "gfx942",
            N=4,
            H=16,
            W=16,
            groups=8,
            KH=3,
            KW=3,
            PAD=1,
            block_q=16,
            block_groups=8,
        ),
    )
    # --- 32c, gfx950 ---
    add(
        "conv_direct",
        "conv_direct/gfx950/32c_n2h8",
        "gfx950",
        build_direct_32c(
            "irhash_direct32c_950_a",
            "gfx950",
            N=2,
            H=8,
            W=8,
            groups=4,
            KH=3,
            KW=3,
            PAD=1,
            block_q=32,
            block_groups=4,
        ),
    )
    # --- 32c, gfx942 ---
    add(
        "conv_direct",
        "conv_direct/gfx942/32c_n2h8",
        "gfx942",
        build_direct_32c(
            "irhash_direct32c_942_a",
            "gfx942",
            N=2,
            H=8,
            W=8,
            groups=4,
            KH=3,
            KW=3,
            PAD=1,
            block_q=32,
            block_groups=4,
        ),
    )
    # --- depthwise, gfx950 ---
    # groups=64, block_waves=1 → block_ch=64; 64%64=0.
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_n4h16",
        "gfx950",
        build_direct_depthwise(
            "irhash_directdw_950_a",
            "gfx950",
            N=4,
            H=16,
            W=16,
            groups=64,
            KH=3,
            KW=3,
            PAD=1,
            block_w=16,
            block_waves=1,
        ),
    )
    # --- depthwise, gfx942 ---
    add(
        "conv_direct",
        "conv_direct/gfx942/dw_n4h16",
        "gfx942",
        build_direct_depthwise(
            "irhash_directdw_942_a",
            "gfx942",
            N=4,
            H=16,
            W=16,
            groups=64,
            KH=3,
            KW=3,
            PAD=1,
            block_w=16,
            block_waves=1,
        ),
    )

    # --- column-streamed depthwise (DirectDepthwiseColSpec) ---
    # Mirrors library parity emit indices 32-40.  Both gfx950 and gfx942 are covered.
    #
    # Those emit indices pin the two engines against each other; these hashes do
    # the other half of the job.  Byte-identity says the engines *agree*, while a
    # recorded hash says the emission has not *moved*: a change that edits both
    # engines in lockstep keeps the parity gate GREEN and still lands here, which
    # is what makes a refactor meant to be behaviour-preserving shown to be so
    # rather than argued to be.
    #
    # The stride-1 fp16 cases carry the extra weight.  The stride generalization
    # (n_iters / tap-pruning / accumulator-index formulas parameterized by
    # p.stride) claims to reduce to the original expressions at stride == 1;
    # these hashes are what turns that claim into a check.
    #
    # Each case pins one axis the emitter branches on: the static tap grid
    # (KH, PAD, stride), the accumulator band (block_h x block_w), the channel
    # tiling (block_waves), and the element type.  Review the IR diff before
    # re-blessing any of them.
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_s1_fp16",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_s1_fp16",
            "gfx950",
            N=2,
            H=8,
            W=8,
            groups=128,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            block_w=4,
            block_waves=2,
            dtype="fp16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_s2_bf16_tail",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_s2_bf16",
            "gfx950",
            N=1,
            H=9,
            W=9,
            groups=70,
            KH=3,
            KW=3,
            PAD=1,
            stride=2,
            block_h=2,
            block_w=4,
            block_waves=1,
            dtype="bf16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_k31_fp16",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_k31",
            "gfx950",
            N=1,
            H=8,
            W=8,
            groups=64,
            KH=31,
            KW=31,
            PAD=15,
            stride=1,
            block_h=4,
            block_w=4,
            block_waves=1,
            dtype="fp16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_k1_pad0",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_k1",
            "gfx950",
            N=2,
            H=6,
            W=6,
            groups=3,
            KH=1,
            KW=1,
            PAD=0,
            stride=1,
            block_w=2,
            block_waves=1,
            dtype="fp16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_kh5kw3_bf16",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_kh5kw3",
            "gfx950",
            N=1,
            H=8,
            W=8,
            groups=128,
            KH=5,
            KW=3,
            PAD=2,
            stride=1,
            block_w=4,
            block_waves=2,
            dtype="bf16",
        ),
    )
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_bw1_tail",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_bw1",
            "gfx950",
            N=1,
            H=10,
            W=10,
            groups=100,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            block_w=1,
            block_waves=2,
            dtype="bf16",
        ),
    )
    # stride=3: the only case above stride 2, so the (y - r) % stride tap
    # pruning is pinned at a stride where more than one tap is dropped.
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_s3_k5_bw3",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_s3_k5",
            "gfx950",
            N=1,
            H=12,
            W=12,
            groups=64,
            KH=5,
            KW=5,
            PAD=2,
            stride=3,
            block_w=3,
            block_waves=1,
            dtype="bf16",
        ),
    )
    # PAD > (KH-1)/2: padded-input overhang, where the last row tile's input
    # rows run past H.  Unreachable at the PAD/KH pairs the cases above use.
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_s2_pad2_ovh",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_pad2_ovh",
            "gfx950",
            N=1,
            H=12,
            W=12,
            groups=64,
            KH=3,
            KW=3,
            PAD=2,
            stride=2,
            block_h=2,
            block_w=2,
            block_waves=1,
            dtype="fp16",
        ),
    )
    # KW >> KH: the regime the column-streamed variant exists for -- KW rides
    # the runtime loop, so only KH weights are live regardless of how wide the
    # filter gets.  dw_col_k31 has KH == KW and so cannot show that separation.
    add(
        "conv_direct",
        "conv_direct/gfx950/dw_col_k3x31",
        "gfx950",
        build_direct_depthwise_col(
            "irhash_dwcol_950_k3x31",
            "gfx950",
            N=1,
            H=16,
            W=32,
            groups=64,
            KH=3,
            KW=31,
            PAD=1,
            stride=1,
            block_w=1,
            block_waves=1,
            dtype="fp16",
        ),
    )
    # gfx942 target -- exercises the arch-specific VGPR budget path
    add(
        "conv_direct",
        "conv_direct/gfx942/dw_col_s1_fp16",
        "gfx942",
        build_direct_depthwise_col(
            "irhash_dwcol_942_s1_fp16",
            "gfx942",
            N=2,
            H=8,
            W=8,
            groups=128,
            KH=3,
            KW=3,
            PAD=1,
            stride=1,
            block_w=4,
            block_waves=2,
            dtype="fp16",
        ),
    )

    # --- conv_direct_dgrad: grouped dgrad (scalar FMA, any cpg/kpg) ---
    # Mirrors parity emit indices 12-14.  Both gfx950 and gfx942 are covered.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dgrad_n2h8_cpg16_bg8",
        "gfx950",
        build_direct_conv_dgrad(
            "irhash_dgrad_950_bg8",
            "gfx950",
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            block_q=16,
            block_groups=8,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dgrad_n2h8_cpg32_bg4",
        "gfx950",
        build_direct_conv_dgrad(
            "irhash_dgrad_950_cpg32_bg4",
            "gfx950",
            N=2,
            H=8,
            W=8,
            groups=8,
            cpg=32,
            kpg=32,
            block_q=16,
            block_groups=4,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx942/dgrad_n1h8_cpg16_bg8",
        "gfx942",
        build_direct_conv_dgrad(
            "irhash_dgrad_942_bg8",
            "gfx942",
            N=1,
            H=8,
            W=8,
            groups=8,
            cpg=16,
            kpg=16,
            block_q=16,
            block_groups=8,
        ),
    )
    # --- conv_direct_dgrad: depthwise dgrad (cpg=kpg=1) ---
    # Mirrors parity emit indices 15-16.  stride=2 exercises divisibility checks.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dw_dgrad_n2h14_s1",
        "gfx950",
        build_direct_depthwise_dgrad(
            "irhash_dw_dgrad_950_s1",
            "gfx950",
            N=2,
            H=14,
            W=14,
            groups=64,
            stride=1,
            block_w=8,
            block_waves=1,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dw_dgrad_n2h14_s2",
        "gfx950",
        build_direct_depthwise_dgrad(
            "irhash_dw_dgrad_950_s2",
            "gfx950",
            N=2,
            H=14,
            W=14,
            groups=64,
            stride=2,
            block_w=8,
            block_waves=1,
        ),
    )

    # --- conv_direct_dgrad: 4c dgrad (forward 4c kernel, transposed problem) ---
    # Fused gathered weights (gfx942), LDS-staged weights with transpose reads
    # and the row-staged runtime row stream (gfx950); the branch matrix and
    # the C++ twin live in library/tests/parity/conv_direct_dgrad_ext_emit.*.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx942/4c_dgrad_fw_fp16_n1h8",
        "gfx942",
        build_direct_4c(
            "irhash_direct4c_dgrad_942_fw",
            "gfx942",
            N=1,
            H=8,
            W=8,
            groups=16,
            block_q=8,
            block_groups=16,
            dgrad_fused_weights=True,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/4c_dgrad_fwl_bf16_n2h13",
        "gfx950",
        build_direct_4c(
            "irhash_direct4c_dgrad_950_fwl",
            "gfx950",
            N=2,
            H=13,
            W=13,
            groups=32,
            block_q=4,
            block_groups=16,
            dtype="bf16",
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/4c_dgrad_fwl_sr2_bf16_n2h9",
        "gfx950",
        build_direct_4c(
            "irhash_direct4c_dgrad_950_sr2",
            "gfx950",
            N=2,
            H=9,
            W=21,
            groups=64,
            block_q=8,
            block_groups=32,
            dtype="bf16",
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
            stage_rows=True,
        ),
    )
    # --- conv_direct_dgrad: generic direct-MFMA dgrad, fused weights ---
    # The forward row stream over the transposed problem reading the original
    # weights (gathered, or LDS-staged transpose reads on gfx950), row guard
    # on. Python-only family (no C++ twin of the generic builder yet).
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c16_bf16",
        "gfx950",
        build_direct_mfma_dgrad_fused(
            "irhash_mfma_dgrad_950_fwl_c16",
            "gfx950",
            N=2,
            H=7,
            W=13,
            groups=32,
            cpg=16,
            kpg=16,
            dtype="bf16",
            fold_k32=False,
            weights_lds=True,
            knobs={"waves_per_eu": 4},
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c32_k32_bf16",
        "gfx950",
        build_direct_mfma_dgrad_fused(
            "irhash_mfma_dgrad_950_fwl_c32k32",
            "gfx950",
            N=2,
            H=7,
            W=13,
            groups=16,
            cpg=32,
            kpg=32,
            dtype="bf16",
            fold_k32=True,
            weights_lds=True,
            block_groups=1,
        ),
    )
    # The row-stream knobs: the policy's stack (two-row prefetch, LDS-only
    # row barrier, XCD row-band order, column pad, staged output) on a
    # whole-image tile with the early exit, and the split-M stack with
    # 32-wide atoms.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c16_stream_bf16",
        "gfx950",
        build_direct_mfma_dgrad_fused(
            "irhash_mfma_dgrad_950_fwl_c16_stream",
            "gfx950",
            N=8,
            H=7,
            W=13,
            groups=32,
            cpg=16,
            kpg=16,
            dtype="bf16",
            fold_k32=False,
            weights_lds=True,
            block_h=16,
            knobs={
                "row_exit": True,
                "waves_per_eu": 4,
                "prefetch_rows": 2,
                "lds_only_sync": True,
                "xcd_tiles": True,
                "lds_pad": 8,
                "stage_out": True,
            },
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c32_k32_wm2_bf16",
        "gfx950",
        build_direct_mfma_dgrad_fused(
            "irhash_mfma_dgrad_950_fwl_c32k32_wm2",
            "gfx950",
            N=8,
            H=7,
            W=13,
            groups=16,
            cpg=32,
            kpg=32,
            dtype="bf16",
            fold_k32=True,
            weights_lds=True,
            block_groups=1,
            knobs={
                "prefetch_rows": 2,
                "lds_only_sync": True,
                "waves_m": 2,
                "xcd_tiles": True,
            },
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx942/mfma_dgrad_fw_c12_fp16",
        "gfx942",
        build_direct_mfma_dgrad_fused(
            "irhash_mfma_dgrad_942_fw_c12",
            "gfx942",
            N=1,
            H=9,
            W=11,
            groups=8,
            cpg=12,
            kpg=12,
            dtype="fp16",
            fold_k32=False,
            weights_lds=False,
            block_h=16,
        ),
    )
    # --- conv_direct_dgrad: windowed depthwise dgrad (stride 1) ---
    # The dot2 VALU form and the Toeplitz MFMA form (gfx950) and its wave32
    # WMMA form (gfx1250); the branch
    # matrix and the C++ twin live in library/tests/parity/conv_direct_dgrad_ext_emit.*.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dw_dgrad_win_dot2_k7",
        "gfx950",
        build_direct_depthwise_dgrad_win(
            "irhash_dw_dgrad_win_950_dot2_k7",
            "gfx950",
            N=2,
            H=14,
            W=14,
            groups=64,
            KH=7,
            PAD=3,
            dtype="bf16",
            block_w=7,
            block_h=14,
            dot2=True,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx950/dw_dgrad_win_mfma_k7",
        "gfx950",
        build_direct_depthwise_dgrad_win(
            "irhash_dw_dgrad_win_950_mfma_k7",
            "gfx950",
            N=3,
            H=14,
            W=14,
            groups=64,
            KH=7,
            PAD=3,
            dtype="bf16",
            block_waves=8,
            block_h=14,
            mfma=True,
            w_fold=2,
            prefetch_rows=2,
        ),
    )
    # The Toeplitz form on the wave32 WMMA (gfx1250): a 28-row 7x7 tile and a
    # 5x5 fp16 one.
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx1250/dw_dgrad_win_wmma_k7",
        "gfx1250",
        build_direct_depthwise_dgrad_win(
            "irhash_dw_dgrad_win_1250_wmma_k7",
            "gfx1250",
            N=2,
            H=30,
            W=33,
            groups=64,
            KH=7,
            PAD=3,
            dtype="bf16",
            block_waves=8,
            block_h=28,
            mfma=True,
            w_fold=1,
            prefetch_rows=2,
            wave_size=32,
        ),
    )
    add(
        "conv_direct_dgrad",
        "conv_direct_dgrad/gfx1250/dw_dgrad_win_wmma_k5",
        "gfx1250",
        build_direct_depthwise_dgrad_win(
            "irhash_dw_dgrad_win_1250_wmma_k5",
            "gfx1250",
            N=5,
            H=13,
            W=9,
            groups=40,
            KH=5,
            PAD=2,
            dtype="fp16",
            block_waves=4,
            block_h=8,
            mfma=True,
            w_fold=4,
            prefetch_rows=2,
            wave_size=32,
        ),
    )

    # --- conv_direct_nongrouped: non-grouped (groups == 1) direct conv ---
    # LDS halo-reuse tile, tap-shared activation fragments, fragment-order
    # weights, hoisted staging predication. One case per structural branch; the
    # full branch matrix (and the C++ twin) lives in
    # library/tests/parity/conv_direct_grouped_emit.* (indices 42+).
    _nongrouped_base = dict(
        tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2, iglp=0
    )
    for _case_id, _arch, _shape, _over in (
        ("bf16_t8x32x64_iglp0", "gfx950", dict(N=2, H=16, W=32, C=64, K=128), {}),
        (
            "fp16_db_s2",
            "gfx950",
            dict(N=1, H=32, W=64, C=64, K=64, stride=2, dtype="fp16"),
            dict(ck=16, double_buffer=True),
        ),
        (
            "bf16_a16x16x32_partial_w",
            "gfx950",
            dict(N=2, H=20, W=40, C=64, K=64),
            dict(tile_w=48, tile_k=32, waves_m=1, atom="16x16x32"),
        ),
        (
            "bf16_noswizzle_we3",
            "gfx950",
            dict(N=2, H=16, W=32, C=64, K=128),
            dict(chiplet_swizzle=False, iglp=None, waves_per_eu=3),
        ),
        (
            "fp16_a32x32x8",
            "gfx942",
            dict(N=1, H=16, W=32, C=64, K=64, dtype="fp16"),
            dict(atom="32x32x8", ck=16),
        ),
        # Dilation: a taller / wider staged halo, taps dil rows / columns apart.
        ("bf16_dil3x1", "gfx950", dict(N=2, H=16, W=32, C=64, K=128, dil_h=3), {}),
        (
            "bf16_dil1x4_p4",
            "gfx950",
            dict(N=1, H=32, W=32, C=64, K=64, PAD=4, dil_w=4),
            {},
        ),
    ):
        add(
            "conv_direct_nongrouped",
            f"conv_direct_nongrouped/{_arch}/{_case_id}",
            _arch,
            build_direct_nongrouped(
                f"irhash_direct_nongrouped_{_case_id}",
                _arch,
                **_shape,
                **{**_nongrouped_base, **_over},
            ),
        )

    # fp32 storage on the direct conv: dtype "fp32" runs the exact fp32 atoms,
    # "tf32" the TF32 atoms (gfx942 native XF32, gfx950 BF16x3). Covers the
    # fixed-cpg 16c forward, the generic DirectConvSpec forward and the
    # non-grouped kernel. check_byte_identity.py holds the C++ twins to the same
    # IR (the fp32-storage configs of conv_direct_grouped_emit); this pins it
    # against drift in both engines.
    for _arch in ("gfx942", "gfx950"):
        _a = _arch[3:]
        for _dt in ("fp32", "tf32"):
            add(
                "conv_direct",
                f"conv_direct/{_arch}/16c_n1h8_{_dt}",
                _arch,
                build_direct_16c(
                    f"irhash_direct16c_{_a}_{_dt}",
                    _arch,
                    N=1,
                    H=8,
                    W=8,
                    groups=8,
                    block_groups=4,
                    fold_k32=_arch == "gfx950",
                    dtype=_dt,
                ),
            )
            add(
                "conv_direct_nongrouped",
                f"conv_direct_nongrouped/{_arch}/{_dt}_a32x32x16_t4x32x64",
                _arch,
                build_direct_nongrouped(
                    f"irhash_direct_nongrouped_{_a}_{_dt}",
                    _arch,
                    N=1,
                    H=16,
                    W=32,
                    C=64,
                    K=64,
                    dtype=_dt,
                    tile_h=4,
                    tile_w=32,
                    tile_k=64,
                    ck=16,
                    waves_m=2,
                    waves_n=2,
                    atom="32x32x16",
                    lds_pad=4,
                ),
            )
        add(
            "conv_direct",
            f"conv_direct/{_arch}/gen_c32_n2h14_tf32",
            _arch,
            build_direct_generic(
                f"irhash_direct_gen_{_a}_tf32",
                _arch,
                dict(N=2, H=14, W=14, groups=8, cpg=32, kpg=32, dtype="tf32"),
                block_h=8,
            ),
        )

    # --- conv_dgrad_{patch,smallc}_wmma: wave32 WMMA stem dgrad (gfx1250) ---
    # Patch embeddings as one GEMM with a pixel-shuffle TDM store, and small-C
    # stems as a GEMM over the flattened filter plus an LDS overlap-add. Every
    # case needs llvm23 (TDM), so the other flavors record them as expected
    # failures. The C++ twins are gated by
    # library/tests/parity/conv_dgrad_{patch,smallc}_wmma_emit.*.
    def build_dgrad_stem_wmma_case(kind, dtype, K, S, PAD, C, **spec_kw):
        def _build():
            from kernels.common.conv_direct_grouped import DirectConvProblem

            if kind == "patch":
                from kernels.common.conv_dgrad_patch_wmma import (
                    DgradPatchWmmaSpec as Spec,
                    build_dgrad_patch_wmma as build,
                )
            else:
                from kernels.common.conv_dgrad_smallc_wmma import (
                    DgradSmallcWmmaSpec as Spec,
                    build_dgrad_smallc_wmma as build,
                )
            p = DirectConvProblem(
                N=2,
                H=64,
                W=64,
                groups=1,
                cpg=C,
                kpg=64,
                KH=K,
                KW=K,
                PAD=PAD,
                stride=S,
                dtype=dtype,
            )
            return build(Spec(problem=p, **spec_kw), arch="gfx1250")

        return _build

    for _kind, _case_id, _dtype, _conv, _over in (
        (
            "patch",
            "bf16_p16_rb8",
            "bf16",
            (16, 16, 0, 3),
            dict(waves_c=4, waves_p=2, tiles_c=6, tiles_p=4, tile_w=16, r_blk=8),
        ),
        (
            "patch",
            "fp16_p4_pf",
            "fp16",
            (4, 4, 0, 3),
            dict(
                waves_c=3,
                waves_p=1,
                tiles_c=1,
                tiles_p=8,
                tile_w=16,
                r_blk=4,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
        (
            "smallc",
            "bf16_3x3s1_c4_pf",
            "bf16",
            (3, 1, 1, 4),
            dict(
                waves_c=1,
                waves_p=3,
                tiles_c=3,
                tiles_p=8,
                out_h=8,
                out_w=32,
                ck=64,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
        (
            "smallc",
            "fp16_7x7s2_c3",
            "fp16",
            (7, 2, 3, 3),
            dict(waves_c=2, waves_p=4, tiles_c=5, tiles_p=6, out_h=32, out_w=32),
        ),
    ):
        _fam = f"conv_dgrad_{_kind}_wmma"
        add(
            _fam,
            f"{_fam}/gfx1250/{_case_id}",
            "gfx1250",
            build_dgrad_stem_wmma_case(_kind, _dtype, *_conv, **_over),
        )

    # --- conv_direct_fwd_small{c,k}_wmma: wave32 WMMA forward of tiny channel
    # counts (gfx1250). TDM window / channel-chunk staging; every case needs
    # llvm23 (TDM), so the other flavors record them as expected failures. The
    # C++ twin is gated by library/tests/parity/conv_fwd_small_wmma_emit.*.
    def build_fwd_small_wmma_case(kind, prob_kw, **spec_kw):
        def _build():
            from kernels.common.conv_direct_grouped import DirectConvProblem
            from kernels.common.conv_fwd_small_wmma import (
                FwdSmallcWmmaSpec,
                FwdSmallkWmmaSpec,
                build_fwd_smallc_wmma,
                build_fwd_smallk_wmma,
            )

            p = DirectConvProblem(N=2, H=32, W=40, groups=1, **prob_kw)
            if kind == "smallc":
                return build_fwd_smallc_wmma(
                    FwdSmallcWmmaSpec(problem=p, **spec_kw), arch="gfx1250"
                )
            return build_fwd_smallk_wmma(
                FwdSmallkWmmaSpec(problem=p, **spec_kw), arch="gfx1250"
            )

        return _build

    for _kind, _case_id, _prob, _over in (
        (
            "smallc",
            "bf16_c3_default",
            dict(cpg=3, kpg=64, dtype="bf16"),
            {},
        ),
        (
            "smallc",
            "fp16_c3_s2_tb4_ts",
            dict(cpg=3, kpg=64, stride=2, dtype="fp16"),
            dict(
                waves_p=2,
                tiles_k=4,
                out_h=8,
                out_w=16,
                tiles_per_block=4,
                tdm_store=True,
            ),
        ),
        (
            "smallc",
            "bf16_c3_k4s4_tb4_ts",
            dict(cpg=3, kpg=64, KH=4, KW=4, PAD=0, stride=4, dtype="bf16"),
            dict(
                waves_p=2,
                tiles_k=8,
                out_h=4,
                out_w=32,
                tiles_per_block=4,
                tdm_store=True,
            ),
        ),
        (
            "smallk",
            "bf16_k4_pf",
            dict(cpg=128, kpg=4, dtype="bf16"),
            dict(tiles_c=3, tiles_p=3, out_h=6, out_w=16, ck=64, prefetch=True),
        ),
        (
            "smallk",
            "fp16_k4_st3_pf",
            dict(cpg=128, kpg=4, dtype="fp16"),
            dict(
                waves_c=3,
                waves_p=2,
                tiles_c=1,
                tiles_p=5,
                out_h=10,
                out_w=10,
                ck=64,
                stages=3,
                prefetch=True,
            ),
        ),
        (
            "smallk",
            "bf16_k3_pf",
            dict(cpg=64, kpg=3, dtype="bf16"),
            dict(tiles_c=2, tiles_p=3, out_h=6, out_w=16, ck=32, prefetch=True),
        ),
        (
            "smallk",
            "fp16_1x7_pw3_k1",
            dict(cpg=32, kpg=1, KH=1, KW=7, PAD=0, pad_w=3, dtype="fp16"),
            dict(waves_p=2, tiles_c=1, tiles_p=4, out_h=1, out_w=122, ck=32),
        ),
    ):
        add(
            f"conv_direct_fwd_{_kind}_wmma",
            f"conv_direct_fwd_{_kind}_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_fwd_small_wmma_case(_kind, _prob, **_over),
        )

    # --- conv_direct_fwd_tapouter_wmma: tap-outer forward of large dilations
    # (gfx1250). TDM tap windows + wave32 WMMA; every case needs llvm23 (TDM),
    # so the other flavors record them as expected failures. The C++ twin is
    # gated by library/tests/parity/conv_direct_fwd_tapouter_wmma_emit.*.
    def build_fwd_tapouter_case(dtype, dil, **spec_kw):
        def _build():
            from kernels.common.conv_direct_fwd_tapouter_wmma import (
                DirectFwdTapOuterSpec,
                build_direct_fwd_tapouter_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            p = DirectConvProblem(
                N=2,
                H=33,
                W=33,
                groups=1,
                cpg=64,
                kpg=96,
                KH=3,
                KW=3,
                PAD=dil,
                stride=1,
                dtype=dtype,
                dil_h=dil,
                dil_w=dil,
            )
            return build_direct_fwd_tapouter_wmma(
                DirectFwdTapOuterSpec(problem=p, **spec_kw), arch="gfx1250"
            )

        return _build

    for _case_id, _dtype, _dil, _over in (
        (
            "bf16_dil12",
            "bf16",
            12,
            dict(
                waves_k=2, waves_p=2, tiles_k=4, tiles_p=4, tile_w=16, ck=128, stages=2
            ),
        ),
        ("fp16_dil36_st3_pf", "fp16", 36, dict(prefetch=True)),
    ):
        add(
            "conv_direct_fwd_tapouter_wmma",
            f"conv_direct_fwd_tapouter_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_fwd_tapouter_case(_dtype, _dil, **_over),
        )

    # --- conv_direct_fwd_gstream_wmma: small-group forward walking row bands
    # (gfx1250). TDM halo ring + wave32 WMMA; every case needs llvm23 (TDM), so
    # the other flavors record them as expected failures. The C++ twin is gated
    # by library/tests/parity/conv_direct_fwd_gstream_wmma_emit.*.
    def build_fwd_gstream_case(prob_kw, **spec_kw):
        def _build():
            from kernels.common.conv_direct_fwd_gstream_wmma import (
                FwdGstreamWmmaSpec,
                build_fwd_gstream_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            shape = dict(N=2, H=28, W=28, KH=3, KW=3, PAD=1, stride=1, dtype="bf16")
            shape.update(prob_kw)
            return build_fwd_gstream_wmma(
                FwdGstreamWmmaSpec(problem=DirectConvProblem(**shape), **spec_kw),
                arch="gfx1250",
            )

        return _build

    for _case_id, _prob, _over in (
        (
            "bf16_c16_s1",
            dict(groups=8, cpg=16, kpg=16),
            dict(waves_k=2, waves_p=2, tiles_k=2, tiles_p=2, tile_w=16, walk=4),
        ),
        (
            "fp16_c56_s2_ps",
            dict(groups=4, cpg=56, kpg=56, stride=2, dtype="fp16"),
            dict(
                waves_k=4,
                waves_p=1,
                tiles_k=2,
                tiles_p=4,
                tile_w=16,
                walk=4,
                phase_split=True,
            ),
        ),
    ):
        add(
            "conv_direct_fwd_gstream_wmma",
            f"conv_direct_fwd_gstream_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_fwd_gstream_case(_prob, **_over),
        )

    # --- conv_direct_dw_fwd_wmma: Toeplitz WMMA depthwise forward (gfx1250) ---
    # Buffer loads + LDS staging + wave32 16x16x32 WMMA. The C++ twin is gated
    # by library/tests/parity/conv_direct_dw_fwd_wmma_emit.*.
    def build_dw_fwd_wmma_case(dtype, K, PAD, **spec_kw):
        def _build():
            from kernels.common.conv_direct_dw_fwd_wmma import (
                DwFwdWmmaSpec,
                build_dw_fwd_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            p = DirectConvProblem(
                N=2,
                H=20,
                W=37,
                groups=64,
                cpg=1,
                kpg=1,
                KH=K,
                KW=K,
                PAD=PAD,
                stride=1,
                dtype=dtype,
            )
            return build_dw_fwd_wmma(
                DwFwdWmmaSpec(problem=p, **spec_kw), arch="gfx1250"
            )

        return _build

    for _case_id, _dtype, _K, _PAD, _over in (
        ("bf16_k3_default", "bf16", 3, 1, {}),
        (
            "fp16_k7_wv8_f4_pf4",
            "fp16",
            7,
            3,
            dict(block_waves=8, block_h=4, w_fold=4, prefetch_rows=4),
        ),
    ):
        add(
            "conv_direct_dw_fwd_wmma",
            f"conv_direct_dw_fwd_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_dw_fwd_wmma_case(_dtype, _K, _PAD, **_over),
        )

    # --- conv_direct_dgrad_wmma: wave32 WMMA direct dgrad (gfx1250) ---
    # 3-D TDM halo and weight staging, transpose-on-read weights, TDM epilogue.
    # Every case needs llvm23 (TDM), so the other flavors record them as
    # expected failures. The C++ twin is gated by
    # library/tests/parity/conv_direct_dgrad_wmma_emit.*.
    def build_direct_dgrad_wmma_case(dtype, **spec_kw):
        def _build():
            from kernels.common.conv_direct_dgrad_wmma import (
                DirectDgradWmmaSpec,
                build_direct_dgrad_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            p = DirectConvProblem(
                N=2,
                H=20,
                W=24,
                groups=1,
                cpg=64,
                kpg=96,
                KH=3,
                KW=3,
                PAD=1,
                stride=1,
                dtype=dtype,
            )
            return build_direct_dgrad_wmma(
                DirectDgradWmmaSpec(problem=p, **spec_kw), arch="gfx1250"
            )

        return _build

    for _case_id, _dtype, _over in (
        ("bf16_default", "bf16", {}),
        (
            "fp16_tn2_st3_pf",
            "fp16",
            dict(
                tiles_c=2,
                tiles_p=2,
                tile_w=8,
                tile_n=2,
                stages=3,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
    ):
        add(
            "conv_direct_dgrad_wmma",
            f"conv_direct_dgrad_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_direct_dgrad_wmma_case(_dtype, **_over),
        )

    # --- conv_direct_fwd_wmma: wave32 WMMA direct forward (gfx1250) ---
    # 3-D TDM halo and weight staging, shifted halo reads per tap, TDM
    # epilogue. Every case needs llvm23 (TDM), so the other flavors record
    # them as expected failures. The C++ twin is gated (every knob) by
    # library/tests/parity/conv_direct_fwd_wmma_emit.*.
    def build_direct_fwd_wmma_case(prob_kw, **spec_kw):
        def _build():
            from kernels.common.conv_direct_fwd_wmma import (
                DirectFwdWmmaSpec,
                build_direct_fwd_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            shape = dict(
                N=2,
                H=20,
                W=24,
                groups=1,
                cpg=64,
                kpg=96,
                KH=3,
                KW=3,
                PAD=1,
                stride=1,
                dtype="bf16",
            )
            shape.update(prob_kw)
            return build_direct_fwd_wmma(
                DirectFwdWmmaSpec(problem=DirectConvProblem(**shape), **spec_kw),
                arch="gfx1250",
            )

        return _build

    for _case_id, _prob, _over in (
        ("bf16_default", {}, {}),
        (
            "fp16_tn2_sb_we_pf",
            dict(dtype="fp16"),
            dict(
                waves_k=1,
                tiles_k=4,
                tiles_p=6,
                tile_n=2,
                split_barrier=True,
                wave_epilogue=True,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
        (
            "bf16_s2_ps_pg4",
            dict(stride=2, groups=4, cpg=8, kpg=8),
            dict(
                tiles_k=1,
                tiles_p=2,
                tile_w=32,
                tile_n=2,
                pack_groups=4,
                phase_split=True,
            ),
        ),
        (
            "bf16_pp_pg3",
            {},
            dict(tiles_p=6, pingpong=True, pp_group=3, sched="iterative-ilp"),
        ),
        (
            "bf16_tu_tsh_dil",
            dict(PAD=2, dil_w=2),
            dict(waves_k=1, tiles_k=2, tap_reuse=True, tap_shift=True),
        ),
        (
            "fp16_1x7_dil3_pw9",
            dict(H=1, W=300, KH=1, KW=7, PAD=0, pad_w=9, dil_w=3, dtype="fp16"),
            dict(
                waves_k=2,
                tiles_k=4,
                tiles_p=4,
                tile_w=256,
                stages=3,
                prefetch=True,
                split_barrier=True,
            ),
        ),
        (
            "bf16_7x1_pw0_tn2",
            dict(H=17, W=17, KH=7, KW=1, PAD=3, pad_w=0),
            dict(waves_k=1, tiles_k=4, tiles_p=5, tile_w=20, tile_n=2),
        ),
        (
            "bf16_1x1_s2_sv_fn",
            dict(KH=1, KW=1, PAD=0, stride=2),
            dict(
                waves_p=1,
                ck=64,
                stages=3,
                strided_view=True,
                fold_n=True,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
        (
            "fp16_2x2_s2_sv",
            dict(KH=2, KW=2, PAD=0, stride=2, dtype="fp16"),
            dict(waves_p=2, tile_w=8, ck=64, strided_view=True),
        ),
        (
            "bf16_1x16_s8_pw4_pv",
            dict(H=1, W=512, KH=1, KW=16, PAD=0, pad_w=4, stride=8),
            dict(
                waves_k=1,
                tiles_k=4,
                tiles_p=4,
                tile_w=256,
                ck=64,
                poly_view=True,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
        (
            "bf16_s2_rr_tn4",
            dict(H=16, W=16, kpg=128, stride=2),
            dict(tiles_p=4, tile_w=8, tile_n=4, stages=3, row_ring=True),
        ),
        (
            "fp16_s2_valid_tn2",
            dict(H=35, W=35, PAD=0, stride=2, dtype="fp16"),
            dict(
                waves_p=2,
                tiles_k=2,
                tiles_p=5,
                tile_w=20,
                tile_n=2,
                prefetch=True,
                sched="iterative-ilp",
            ),
        ),
    ):
        add(
            "conv_direct_fwd_wmma",
            f"conv_direct_fwd_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_direct_fwd_wmma_case(_prob, **_over),
        )

    # --- conv_winograd_fwd: Winograd F(m x m_w, 3x3) forward transforms ---
    # K2 input transform (register path, fused bf16 prescale), K4 output
    # transform and the merged K2+K1 launch. The full matrix (and the C++
    # twin) lives in library/tests/parity/conv_winograd_fwd_emit.*.
    def build_winograd_fwd_case(stage, **spec_kw):
        def _build():
            from kernels.common._conv_implicit_gemm_common import ConvProblem
            from kernels.common import conv_winograd_fwd as wf

            p = ConvProblem(N=2, Hi=20, Wi=24, C=64, K=96, Y=3, X=3, pH=1, pW=1)
            spec = wf.WinogradFwdSpec(problem=p, **spec_kw)
            if isinstance(stage, tuple):
                return wf.build_winograd_merged(
                    wf.WinogradMergedSpec(spec, stage), arch="gfx1250"
                )
            build = {
                "K2": wf.build_winograd_input_transform,
                "K4": wf.build_winograd_output_transform,
            }[stage]
            return build(spec, arch="gfx1250")

        return _build

    for _case_id, _stage, _over in (
        ("f4_k2", "K2", {}),
        ("f4_bf16_prescale_k2", "K2", dict(io_dtype="bf16", prescale=True)),
        ("f6x4_k4", "K4", dict(m=6, m_w=4)),
        ("f4_merged_k2_k1", ("K2", "K1"), {}),
    ):
        add(
            "conv_winograd_fwd",
            f"conv_winograd_fwd/gfx1250/{_case_id}",
            "gfx1250",
            build_winograd_fwd_case(_stage, **_over),
        )

    # --- conv_winograd_fwd_rt: shape-generic (AOT) Winograd transforms ---
    # Every extent a kernel argument: K1, K2a, K2 (register path with the fused
    # and the split prescale, LDS path) and K4. The full matrix (and the C++
    # twin) lives in library/tests/parity/conv_winograd_fwd_rt_emit.*.
    def build_winograd_rt_case(cls, **spec_kw):
        def _build():
            from kernels.common import conv_winograd_fwd_rt as wr

            return wr.build_stage(getattr(wr, cls)(**spec_kw), arch="gfx1250")

        return _build

    for _case_id, _cls, _over in (
        ("f4_k1", "RtFilterSpec", dict(filter_rows=2)),
        ("bf16_k2a", "RtPrescaleSpec", dict(io_dtype="bf16")),
        ("f4_k2", "RtInputSpec", dict(vec=4)),
        (
            "f4_bf16_fused_prescale_k2",
            "RtInputSpec",
            dict(io_dtype="bf16", prescale=True, fused=True, vec=4),
        ),
        (
            "f2_lds_k2",
            "RtInputSpec",
            dict(mh=2, mw=2, lds=(4, 8), tiles_per_thread=4),
        ),
        (
            "f6x4_bf16_prescale_k4",
            "RtOutputSpec",
            dict(mh=6, mw=4, io_dtype="bf16", prescale=True, vec=4),
        ),
    ):
        add(
            "conv_winograd_fwd_rt",
            f"conv_winograd_fwd_rt/gfx1250/{_case_id}",
            "gfx1250",
            build_winograd_rt_case(_cls, **_over),
        )

    # --- conv_winograd_gemm_wmma: the Winograd batched GEMM (K3, gfx1250) ---
    # TDM operand ring + WMMA, TDM or per-lane epilogue, split-K slabs and
    # their reduce. The full matrix (and the C++ twin) lives in
    # library/tests/parity/conv_winograd_gemm_wmma_emit.*.
    def build_winograd_gemm_case(**spec_kw):
        def _build():
            from kernels.common.conv_winograd_gemm_wmma import (
                WinogradGemmWmmaSpec,
                build_winograd_gemm_wmma,
            )

            return build_winograd_gemm_wmma(
                WinogradGemmWmmaSpec(**spec_kw), arch="gfx1250"
            )

        return _build

    def build_winograd_gemm_reduce_case(dtype):
        def _build():
            from kernels.common.conv_winograd_gemm_wmma import (
                WinogradGemmReduceSpec,
                build_winograd_gemm_reduce,
            )

            return build_winograd_gemm_reduce(
                WinogradGemmReduceSpec(dtype), arch="gfx1250"
            )

        return _build

    for _case_id, _build_fn in (
        ("fp16_default", build_winograd_gemm_case()),
        (
            "bf16_sk_direct_taps2",
            build_winograd_gemm_case(
                dtype="bf16", split_k=True, epilogue="direct", taps=2, tile_k=64
            ),
        ),
        ("reduce_fp16", build_winograd_gemm_reduce_case("fp16")),
    ):
        add(
            "conv_winograd_gemm_wmma",
            f"conv_winograd_gemm_wmma/gfx1250/{_case_id}",
            "gfx1250",
            _build_fn,
        )

    # --- conv_winograd_gen: generalized Winograd forward transforms ---
    # K1 filter / K2 input / K4 output transforms of a segmented 1-D filter,
    # a grouped 5x5 one and the 1x5 / 5x1 F(4,5). The full matrix (and the
    # C++ twin) lives in library/tests/parity/conv_winograd_gen_emit.*.
    def build_winograd_gen_case(stage, prob, ay, ax, **spec_kw):
        def _build():
            from kernels.common import conv_winograd_gen as G
            from kernels.common._conv_implicit_gemm_common import ConvProblem

            spec = G.WinogradGenSpec(
                problem=ConvProblem(**prob),
                ax_y=G.WinoAxis(*ay),
                ax_x=G.WinoAxis(*ax),
                **spec_kw,
            )
            build = {
                "K1": G.build_filter_transform,
                "K2": G.build_input_transform,
                "K4": G.build_output_transform,
            }[stage]
            return build(spec, arch="gfx1250")

        return _build

    _wg_1x11 = dict(N=2, Hi=1, Wi=200, C=64, K=64, Y=1, X=11, pW=5)
    _wg_5x5 = dict(N=1, Hi=20, Wi=20, C=64, K=64, Y=5, X=5, pH=2, pW=2, groups=2)
    # F(4,5): the selector's only 1x5 / 5x1 variant. K1 / K2 / K4 pin its
    # G / B^T / A^T respectively.
    _wg_1x5 = dict(N=2, Hi=1, Wi=130, C=64, K=64, Y=1, X=5, pW=2)
    _wg_5x1 = dict(N=1, Hi=60, Wi=3, C=32, K=64, Y=5, X=1, pH=4, dH=2)
    for _case_id, _stage, _prob, _ay, _ax, _kw in (
        ("1x11_6s2_k1", "K1", _wg_1x11, (1, 1), (6, 11, 2), {}),
        ("1x11_6s2_k2", "K2", _wg_1x11, (1, 1), (6, 11, 2), {}),
        ("5x5_f2_g2_bf16_k4", "K4", _wg_5x5, (2, 5), (2, 5), dict(io_dtype="bf16")),
        ("1x5_f4_k1", "K1", _wg_1x5, (1, 1), (4, 5), {}),
        ("1x5_f4_k2", "K2", _wg_1x5, (1, 1), (4, 5), {}),
        ("1x5_f4_k4", "K4", _wg_1x5, (1, 1), (4, 5), {}),
        ("5x1_f4_d2_bf16_k2", "K2", _wg_5x1, (4, 5), (1, 1), dict(io_dtype="bf16")),
    ):
        add(
            "conv_winograd_gen",
            f"conv_winograd_gen/gfx1250/{_case_id}",
            "gfx1250",
            build_winograd_gen_case(_stage, _prob, _ay, _ax, **_kw),
        )

    # --- conv_winograd_fused_1d_wmma: fused 1-D Winograd forward (gfx1250) ---
    # Weight-stationary 1 x r Winograd with the WMMA input transform, async
    # prefetch and merged phases (the selector's config), and the VALU
    # transform without prefetch. The C++ twin is gated by
    # library/tests/parity/conv_winograd_fused_1d_wmma_emit.*.
    def build_winograd_fused_1d_case(C, K, ax, **spec_kw):
        def _build():
            from kernels.common._conv_implicit_gemm_common import ConvProblem
            from kernels.common.conv_winograd_fused_1d_wmma import (
                FusedWino1DSpec,
                build_fused_wino_1d,
            )
            from kernels.common.conv_winograd_gen import WinoAxis

            p = ConvProblem(N=1, Hi=1, Wi=600, C=C, K=K, Y=1, X=ax[1], pW=ax[1] // 2)
            return build_fused_wino_1d(
                FusedWino1DSpec(problem=p, ax=WinoAxis(*ax), **spec_kw),
                arch="gfx1250",
            )

        return _build

    for _case_id, _c, _k, _ax, _kw in (
        (
            "1x11_4s3_c128_sel",
            128,
            128,
            (4, 11, 3),
            dict(
                q=2,
                fk=4,
                waves=4,
                xf="wmma",
                xf_batch=16,
                prefetch=True,
                m_f16=True,
                merge=True,
            ),
        ),
        ("1x7_4s2_c64_valu", 64, 64, (4, 7, 2), dict(q=2, fk=2)),
    ):
        add(
            "conv_winograd_fused_1d_wmma",
            f"conv_winograd_fused_1d_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_winograd_fused_1d_case(_c, _k, _ax, **_kw),
        )

    # Stride 2 (3x3 phases and the 2x2 patch merge), dilation, packed small
    # groups (also at stride 2), the split barrier with and without the
    # per-wave epilogue, and tile widths that are not a power of two. Also
    # llvm23-only.
    def build_direct_dgrad_wmma_ext_case(prob_kw, **spec_kw):
        def _build():
            from kernels.common.conv_direct_dgrad_wmma import (
                DirectDgradWmmaSpec,
                build_direct_dgrad_wmma,
            )
            from kernels.common.conv_direct_grouped import DirectConvProblem

            pk = dict(
                N=2,
                H=20,
                W=24,
                groups=1,
                cpg=64,
                kpg=96,
                KH=3,
                PAD=1,
                stride=1,
                dtype="bf16",
            )
            pk.update(prob_kw)
            pk["KW"] = pk["KH"]
            return build_direct_dgrad_wmma(
                DirectDgradWmmaSpec(problem=DirectConvProblem(**pk), **spec_kw),
                arch="gfx1250",
            )

        return _build

    _dgw_pf = dict(prefetch=True, sched="iterative-ilp")
    for _case_id, _prob, _over in (
        (
            "bf16_dil2x2_p2",
            dict(PAD=2, dil_h=2, dil_w=2),
            dict(waves_c=2, waves_p=4, tiles_c=4, tiles_p=8, **_dgw_pf),
        ),
        (
            "bf16_pg4_c8k8",
            dict(groups=8, cpg=8, kpg=8),
            dict(waves_c=1, waves_p=8, tiles_c=2, tiles_p=4, tile_n=2, pack_groups=4),
        ),
        (
            "bf16_s2_pg4_c8k8",
            dict(stride=2, groups=8, cpg=8, kpg=8),
            dict(waves_c=1, waves_p=8, tiles_c=2, tiles_p=4, tile_w=8, pack_groups=4),
        ),
        (
            "fp16_s2_pg8_c4k4",
            dict(dtype="fp16", stride=2, groups=16, cpg=4, kpg=4),
            dict(waves_c=2, waves_p=4, tiles_c=1, tiles_p=4, tile_n=2, pack_groups=8),
        ),
        (
            "bf16_s2_3x3",
            dict(stride=2),
            dict(waves_c=2, waves_p=4, tiles_c=2, tiles_p=4, tile_w=8, tile_n=2, ck=64),
        ),
        (
            "bf16_s2_patch2",
            dict(KH=2, PAD=0, stride=2),
            dict(waves_c=2, waves_p=4, tiles_c=2, tiles_p=4, ck=64),
        ),
        (
            "bf16_sb_tw56",
            {},
            dict(
                waves_c=1,
                waves_p=7,
                tiles_c=4,
                tiles_p=10,
                tile_w=56,
                split_barrier=True,
            ),
        ),
        (
            "bf16_sb_we_tw24",
            {},
            dict(
                waves_c=1,
                waves_p=8,
                tiles_c=4,
                tiles_p=6,
                tile_w=24,
                tile_n=4,
                split_barrier=True,
                wave_epilogue=True,
                **_dgw_pf,
            ),
        ),
    ):
        add(
            "conv_direct_dgrad_wmma",
            f"conv_direct_dgrad_wmma/gfx1250/{_case_id}",
            "gfx1250",
            build_direct_dgrad_wmma_ext_case(_prob, **_over),
        )

    # --- conv_direct_wgrad: the direct wgrad kernels on gfx950 ---
    # Tiled (LDS-DMA row pipeline, swizzled transpose reads, atomics), the
    # depthwise partial-slot kernels and their fold, the small-C gather, and
    # the 4x4x4 MFMA depthwise and 4/8-channel grouped kernels.
    # The full branch matrix (and the C++ twins) lives in
    # library/tests/parity/conv_direct_grouped_emit.* (indices 95+).
    for _case_id, _kind, _prob, _over in (
        ("tiled_bf16_c128", "tiled", dict(cpg=128, kpg=128, H=30), {}),
        (
            "tiled_qg2_pingpong",
            "tiled",
            {},
            dict(
                q_groups=2,
                pingpong=True,
                waves_k=2,
                wave_k16=1,
                rows_per_seg=12,
                prefetch=3,
            ),
        ),
        (
            "tiled_fp16_unfold4",
            "tiled",
            dict(N=4, H=15, W=40, dtype="fp16"),
            dict(unfold_n=4, rows_per_seg=15, row_unroll=3),
        ),
        ("fold_t9", "fold", {}, dict(taps=9)),
        (
            "dw_cols_c3_7x7_s2",
            "dw_cols",
            dict(groups=3, cpg=1, kpg=1, KH=7, KW=7, PAD=3, stride=2, H=20),
            {},
        ),
        ("dw_lds_c64", "dw_lds", dict(groups=64, cpg=1, kpg=1), {}),
        (
            "gather_c3_s2",
            "gather",
            dict(cpg=3, kpg=24, stride=2, H=32, W=64),
            {},
        ),
        (
            "gather_c24_tr",
            "gather",
            dict(cpg=24, kpg=96, stride=2, H=30, W=40),
            dict(wave_k16=1, waves_k=6, rows_per_seg=8, target_blocks=256),
        ),
        (
            "dw_mfma_7x7_c192",
            "dw_mfma",
            dict(groups=192, cpg=1, kpg=1, KH=7, KW=7, PAD=3, H=56, W=56),
            dict(q_tiles=8, rows_per_chunk=28),
        ),
        (
            "dw_mfma_7x7_unfold3",
            "dw_mfma",
            dict(groups=1024, cpg=1, kpg=1, KH=7, KW=7, PAD=3, N=8, H=7, W=7),
            dict(rows_per_chunk=14, unfold_n=3),
        ),
        (
            "grouped_mfma_c4",
            "grouped_mfma",
            dict(groups=32, cpg=4, kpg=4, H=56, W=56),
            dict(wave_q4=2),
        ),
        (
            "grouped_mfma_c8_fp16",
            "grouped_mfma",
            dict(groups=32, cpg=8, kpg=8, H=56, W=56, dtype="fp16"),
            dict(wave_q4=2, prefetch=3),
        ),
    ):
        add(
            "conv_direct_wgrad",
            f"conv_direct_wgrad/gfx950/{_case_id}",
            "gfx950",
            build_direct_wgrad(
                f"irhash_direct_wgrad_{_case_id}", "gfx950", _kind, _prob, **_over
            ),
        )
    # The gfx1250 (wave32) direct wgrad kernels: the image-per-block kernel
    # (register blocking, row split, TDM ring, replicated two-stage scratch,
    # LDS epilogue), the small-C stem form, and the depthwise VALU kernel.
    # Mirrors library/tests/parity/conv_direct_grouped_emit.* (indices 318+).
    for _case_id, _kind, _prob, _over in (
        (
            "image_rps3_rb2_ws8",
            "w32",
            dict(H=12),
            dict(
                waves_k=2,
                waves_c=2,
                image_per_block=True,
                rows_per_sync=3,
                row_blocks=2,
                ws_replicas=8,
                lds_epilogue=True,
            ),
        ),
        (
            "image_tdm3_fp16",
            "w32",
            dict(dtype="fp16"),
            dict(
                waves_k=4,
                waves_c=2,
                image_per_block=True,
                rows_per_sync=3,
                tdm_stages=3,
                ws_replicas=8,
            ),
        ),
        (
            "small_c3_3x3_s2",
            "w32",
            dict(cpg=3, kpg=32, stride=2, H=32, W=64),
            dict(
                waves_k=2,
                waves_c=1,
                wave_tiles_k=1,
                wave_tiles_c=2,
                small_c=True,
                vec_x=4,
                sub_steps=2,
                ws_replicas=4,
                row_blocks=8,
            ),
        ),
        (
            "dw_valu_7x7_c128",
            "dw_valu",
            dict(groups=128, cpg=1, kpg=1, KH=7, KW=7, PAD=3, H=14, W=14),
            dict(chan_per_lane=4, waves=8, rows_per_wave=4, r_split=2),
        ),
    ):
        add(
            "conv_direct_wgrad",
            f"conv_direct_wgrad/gfx1250/{_case_id}",
            "gfx1250",
            build_direct_wgrad(
                f"irhash_direct_wgrad_1250_{_case_id}",
                "gfx1250",
                _kind,
                _prob,
                wave_size=32,
                **({"mfma_k": 32} if _kind == "w32" else {}),
                **_over,
            ),
        )

    # --- conv_direct_depthwise_tiled: output-stationary depthwise direct conv ---
    # Runtime filter-row loop, sliding input-row window, one weight row in
    # registers. Mirrors library/tests/parity/conv_direct_grouped_emit.*
    # (indices 57+).
    for _case_id, _arch, _shape, _over in (
        (
            "bf16_k31_bw8_bh2",
            "gfx950",
            dict(N=1, H=56, W=56, groups=192, KH=31, PAD=15),
            dict(block_w=8, block_h=2),
        ),
        (
            "fp16_k7_bw14_bh7_ur",
            "gfx950",
            dict(N=4, H=14, W=14, groups=128, KH=7, PAD=3, dtype="fp16"),
            dict(block_w=14, block_h=7, unroll_rows=True),
        ),
        (
            "bf16_k5_s2_bwv2",
            "gfx942",
            dict(N=2, H=28, W=28, groups=256, KH=5, PAD=2, stride=2),
            dict(block_w=8, block_h=4, block_waves=2),
        ),
    ):
        add(
            "conv_direct_depthwise_tiled",
            f"conv_direct_depthwise_tiled/{_arch}/{_case_id}",
            _arch,
            build_direct_depthwise_tiled(
                f"irhash_direct_dw_tiled_{_case_id}", _arch, **_shape, **_over
            ),
        )

    # Depthwise forward direct kernels on the wave32 gfx1250, built at its wave
    # size; plus the preloading kernel as a logical wave64 (no cross-lane
    # operation, so it runs as two hardware waves). Mirrors
    # library/tests/parity/conv_direct_grouped_emit.* (indices 114+).
    for _tag, _wave in (("dw_n4h16_w32", 32), ("dw_n4h16_w64", 64)):
        add(
            "conv_direct",
            f"conv_direct/gfx1250/{_tag}",
            "gfx1250",
            build_direct_depthwise(
                f"irhash_directdw_1250_{_tag}",
                "gfx1250",
                N=4,
                H=16,
                W=16,
                groups=64,
                block_w=16,
                block_waves=1,
                wave_size=_wave,
            ),
        )
    add(
        "conv_direct",
        "conv_direct/gfx1250/dw_col_s2_fp16_w32",
        "gfx1250",
        build_direct_depthwise_col(
            "irhash_dwcol_1250_s2_fp16",
            "gfx1250",
            N=2,
            H=12,
            W=12,
            groups=96,
            stride=2,
            block_h=8,
            block_w=2,
            wave_size=32,
        ),
    )
    add(
        "conv_direct_depthwise_tiled",
        "conv_direct_depthwise_tiled/gfx1250/bf16_k5_bwv2_w32",
        "gfx1250",
        build_direct_depthwise_tiled(
            "irhash_direct_dw_tiled_1250_bf16_k5",
            "gfx1250",
            N=2,
            H=16,
            W=16,
            groups=96,
            KH=5,
            PAD=2,
            block_w=8,
            block_h=2,
            block_waves=2,
            wave_size=32,
        ),
    )
    # Channel-packed (cpl > 1) depthwise forward on the wave32 gfx1250:
    # cpl-wide vector loads / stores and packed <2 x f32> FMAs. Mirrors
    # library/tests/parity/conv_direct_grouped_emit.* (indices 171+).
    add(
        "conv_direct",
        "conv_direct/gfx1250/dw_n4h16_cpl2_w32",
        "gfx1250",
        build_direct_depthwise(
            "irhash_directdw_1250_cpl2",
            "gfx1250",
            N=4,
            H=16,
            W=16,
            groups=64,
            block_w=8,
            block_waves=1,
            wave_size=32,
            cpl=2,
        ),
    )
    add(
        "conv_direct_depthwise_tiled",
        "conv_direct_depthwise_tiled/gfx1250/bf16_k3_cpl4_w32",
        "gfx1250",
        build_direct_depthwise_tiled(
            "irhash_direct_dw_tiled_1250_bf16_k3_cpl4",
            "gfx1250",
            N=2,
            H=16,
            W=16,
            groups=96,
            KH=3,
            PAD=1,
            block_w=8,
            block_h=2,
            block_waves=1,
            wave_size=32,
            cpl=4,
        ),
    )
    # Depthwise dgrad on the wave32 gfx1250: the row-streaming kernel, scalar
    # and channel-packed (cpl 2 / 8), and the rotated-filter tiled kernel on a
    # large and a non-square filter with per-axis padding. Mirrors
    # library/tests/parity/conv_direct_grouped_emit.* (indices 179+).
    for _tag, _shape, _over in (
        ("k3", dict(KH=3, PAD=1), dict(block_w=8, block_waves=2)),
        ("k3_cpl2", dict(KH=3, PAD=1), dict(block_w=8, block_waves=2, cpl=2)),
        (
            "k5_s2_bf16_cpl8",
            dict(KH=5, PAD=2, stride=2, dtype="bf16", groups=256),
            dict(block_w=4, block_waves=1, cpl=8),
        ),
    ):
        add(
            "conv_direct",
            f"conv_direct/gfx1250/dw_dgrad_stream_{_tag}_w32",
            "gfx1250",
            build_direct_depthwise_dgrad_stream(
                f"irhash_dw_dgrad_stream_1250_{_tag}",
                "gfx1250",
                **{"N": 2, "H": 14, "W": 14, "groups": 64, **_shape},
                wave_size=32,
                **_over,
            ),
        )
    for _tag, _shape, _over in (
        (
            "bf16_k31",
            dict(H=16, W=16, groups=32, KH=31, PAD=15),
            dict(block_w=4, block_h=1),
        ),
        (
            "fp16_k51x5",
            dict(H=24, W=24, groups=64, KH=51, KW=5, PAD=25, pad_w=2, dtype="fp16"),
            dict(block_w=8, block_h=2),
        ),
    ):
        add(
            "conv_direct_depthwise_tiled",
            f"conv_direct_depthwise_tiled/gfx1250/dgrad_{_tag}_w32",
            "gfx1250",
            build_direct_depthwise_tiled(
                f"irhash_direct_dw_tiled_dgrad_1250_{_tag}",
                "gfx1250",
                N=2,
                **_shape,
                block_waves=1,
                wave_size=32,
                dgrad=True,
                **_over,
            ),
        )

    # Deep fused conv: arch shims and shape/toggle samples.
    add(
        "deep_fused_conv",
        "deep/gfx950/h16w16_k32",
        "gfx950",
        build_deep(
            "gfx950",
            "gfx950",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx950/h24w24_k32",
        "gfx950",
        build_deep(
            "gfx950",
            "gfx950",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/common_gfx950_wt16/h16w16_k32",
        "gfx950",
        build_deep(
            "common",
            "gfx950",
            name="irhash_deep_gfx950_wt16",
            wave_size=64,
            warp_tile_m=16,
            warp_tile_n=16,
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/common_gfx950_wt16/h24w24_k32",
        "gfx950",
        build_deep(
            "common",
            "gfx950",
            name="irhash_deep_gfx950_wt16b",
            wave_size=64,
            warp_tile_m=16,
            warp_tile_n=16,
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1201/h16w16_k32",
        "gfx1201",
        build_deep(
            "gfx1201",
            "gfx1201",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            epilogue="cshuffle",
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1201/h24w24_k32",
        "gfx1201",
        build_deep(
            "gfx1201",
            "gfx1201",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            epilogue="cshuffle",
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx12_generic/h16w16_k32",
        "gfx12-generic",
        build_deep(
            "gfx1201",
            "gfx12-generic",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            epilogue="cshuffle",
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1151/native_h16w16",
        "gfx1151",
        build_deep(
            "gfx1151",
            "gfx1151",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx1151/native_h24w24",
        "gfx1151",
        build_deep(
            "gfx1151",
            "gfx1151",
            h=24,
            w=24,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )
    add(
        "deep_fused_conv",
        "deep/gfx11_generic/native_h16w16",
        "gfx11-generic",
        build_deep(
            "gfx1151",
            "gfx11-generic",
            h=16,
            w=16,
            c=16,
            k0=32,
            k1=32,
            pool_tile_h=4,
            pool_tile_w=4,
            native_int=True,
        ),
    )

    return out


def run(flavor: str, shards: set[str] | None = None):
    """One flavor's golden sub-document, computed as the platform harness
    computes it: a case that raises is recorded as an expected failure."""
    return run_golden_cases(cases(), flavor, lower=lower_case, shards=shards)


def build_golden(shards: set[str] | None = None) -> dict:
    """Run every case (or only those in ``shards``) under each golden profile
    flavor and return the flavor-keyed doc."""
    return golden_document(lambda fl: run(fl, shards))


def write_golden(golden_dir: Path, shards: set[str] | None = None) -> dict:
    """Bless ``golden_dir`` as one ``<shard>.json`` per family, as the platform
    harness does; with ``shards`` only those files are rewritten."""
    doc = build_golden(shards)
    write_golden_shards(golden_dir, doc, shards)
    return doc


def check_golden(golden_path: Path = _GOLDEN, flavor: str | None = None) -> list[str]:
    """Compare a fresh run against the golden sub-document(s). Empty list == OK."""
    return _check_golden(golden_path, lambda fl: run(fl), flavor)


def test_conv_ir_case_ids_are_unique():
    case_ids = [case["case_id"] for case in cases()]
    assert len(case_ids) == len(set(case_ids))


def test_conv_ir_cases_cover_generic_targets():
    """The deep fused conv cases are the representative-IR coverage of the
    gfx11/gfx12 generic targets; keep them here now that the platform harness
    no longer builds them."""
    arches = {case["arch"] for case in cases()}
    assert {"gfx11-generic", "gfx12-generic"} <= arches


@arch_specific(
    ("gfx950",),
    reason="deep_fused_conv has an IR case on gfx950 only; gfx942 needs a new case and golden",
)
def test_deep_fused_conv_ir_case_exists():
    families = {case["family"] for case in cases() if case["arch"] == "gfx950"}
    assert "deep_fused_conv" in families


def pytest_generate_tests(metafunc):
    # One test per LLVM flavor, as the platform golden test does, so each
    # flavor gets its own timeout. A hook rather than a decorator keeps pytest
    # out of the --check / --write script path.
    if "flavor" in metafunc.fixturenames:
        metafunc.parametrize("flavor", GOLDEN_FLAVORS)


def test_conv_ir_golden_holds_every_llvm_flavor():
    """The golden has a document (or alias) for every LLVM flavor and nothing
    for a flavor rocke no longer qualifies."""
    import pytest

    if not _GOLDEN.is_dir():
        pytest.skip("conv golden fixture missing; generate with --write")
    errors = check_golden_flavors(_GOLDEN)
    assert not errors, "conv golden flavor set is wrong:\n  " + "\n  ".join(errors)


def test_conv_ir_matches_golden(flavor):
    import pytest

    if not _GOLDEN.is_dir():
        pytest.skip("conv golden fixture missing; generate with --write")
    drift = check_golden(_GOLDEN, flavor)
    assert not drift, f"conv IR drift vs golden at {flavor}:\n  " + "\n  ".join(drift)


def main():
    golden_main(
        cases=cases,
        write_golden=write_golden,
        check_golden=check_golden,
        default_golden=_GOLDEN,
        label="conv IR parity",
    )


if __name__ == "__main__":
    main()
