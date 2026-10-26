# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python <-> C++ bridge for the convolution families.

Each conv family is built twice: by its Python builder in ``kernels/`` and by
its C++ mirror in ``library/cpp``, bound into the ``rocke_library_engine``
extension. This module turns a Python spec into the flat dict that extension's
``<family>_lower_llvm`` / ``<family>_serialize_ir`` entry points take, and
drives both engines through :func:`rocke.core.backend.lower_family`: the same
``python`` / ``cpp`` / ``both`` selection and differential diff as the engine's
own families.

The dict keys are the contract with ``library/cpp/bindings/
rocke_library_engine_conv.cpp`` (and ``..._img2col.cpp``); a new spec field is
added on both sides in the same change.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from rocke.core.backend import GemmLowerResult, import_engine, lower_family, name_of

#: The extension that binds the conv families' C++ mirrors.
ENGINE_MODULE = "rocke_library_engine"


def _conv_problem_to_dict(p: Any) -> Dict[str, Any]:
    """:class:`ConvProblem` -> dict (shared by conv_implicit_gemm / img2col).

    The filter-window extent is named ``Y``/``X`` in the C engine; on this
    branch the Python ``ConvProblem`` still exposes it as ``R``/``S`` (the
    merge-target rename to ``Y``/``X`` is a separate change). Read whichever the
    Python object exposes and always emit the engine's ``Y``/``X`` keys.
    3-D fields are passed through when present (default 2-D leaves them 0)."""
    y = getattr(p, "Y", None)
    if y is None:
        y = p.R
    x = getattr(p, "X", None)
    if x is None:
        x = p.S
    d = dict(
        N=p.N,
        Hi=p.Hi,
        Wi=p.Wi,
        C=p.C,
        K=p.K,
        Y=y,
        X=x,
        sH=p.sH,
        sW=p.sW,
        pH=p.pH,
        pW=p.pW,
        dH=p.dH,
        dW=p.dW,
        groups=getattr(p, "groups", 1),
    )
    for f in ("Di", "Z", "sD", "pD", "dD"):
        v = getattr(p, f, None)
        if v is not None:
            d[f] = v
            d["is_3d"] = True
    return d


def conv_implicit_gemm_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`ImplicitGemmConvSpec` -> flat dict (problem nested). Optional
    fields (lds_k_pad/waves_per_eu/vector_size_*) are forwarded when set; the
    binding leaves them at the engine default (Python ``None``) otherwise."""
    d = dict(
        problem=_conv_problem_to_dict(spec.problem),
        name=spec.name,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_tile_m=spec.warp_tile_m,
        warp_tile_n=spec.warp_tile_n,
        warp_tile_k=spec.warp_tile_k,
        wave_size=spec.wave_size,
        pipeline=spec.pipeline,
        epilogue=spec.epilogue,
        async_dma=spec.async_dma,
        unroll_k=spec.unroll_k,
        chiplet_swizzle=spec.chiplet_swizzle,
        chiplet_wgm=spec.chiplet_wgm,
        chiplet_num_xcds=spec.chiplet_num_xcds,
        chiplet_chunk_size=spec.chiplet_chunk_size,
        k0_k1_split=spec.k0_k1_split,
        groups=spec.groups,
        # Changes the emitted kernel and its launch grid; dropped, the engine
        # would build an unmerged kernel under a merged spec's name and grid.
        group_merge=getattr(spec, "group_merge", 1),
    )
    # dtype_* and the vector-size/optional knobs are merge-target additions
    # (#8624); forward them only when this branch's spec exposes them so the
    # binding falls back to the engine default otherwise.
    for f in (
        "dtype_a",
        "dtype_b",
        "dtype_d",
        "dtype_acc",
        "lds_k_pad",
        "waves_per_eu",
        "vector_size_a",
        "vector_size_b",
        "vector_size_c",
    ):
        v = getattr(spec, f, None)
        if v is not None:
            d[f] = v
    return d


def conv_direct_grouped_spec_to_dict(spec: Any, kind: str) -> Dict[str, Any]:
    """:class:`DirectConv16cSpec` / :class:`DirectConv4cSpec` /
    :class:`DirectConvSpec` -> flat dict. ``kind`` ("16c"|"4c"|"generic")
    selects the binding's spec path. The problem's ``dtype`` is always
    forwarded; every other kernel field the spec has is forwarded when
    present: the 16c ``double_buffer``/``fold_k32``, the 4c dgrad knobs
    (``dgrad_fused_weights``, ``dgrad_weights_lds``, ``stage_rows``,
    ``waves_q``) and the generic kernel's tiling, persistent-grid,
    runtime-channel, weight-path and row-stream fields."""
    p = spec.problem
    d = dict(
        kind=kind,
        problem=dict(
            N=p.N,
            H=p.H,
            W=p.W,
            groups=p.groups,
            cpg=p.cpg,
            kpg=p.kpg,
            KH=p.KH,
            KW=p.KW,
            PAD=p.PAD,
            stride=p.stride,
            dil_h=p.dil_h,
            dil_w=p.dil_w,
            dtype=p.dtype,
        ),
        name=spec.name,
        block_q=spec.block_q,
        block_groups=spec.block_groups,
        wave_size=spec.wave_size,
    )
    for f in (
        "double_buffer",
        "fold_k32",
        "dgrad_fused_weights",
        "dgrad_weights_lds",
        "stage_rows",
        "waves_q",
        # DirectConvSpec (kind "generic").
        "block_h",
        "waves_k",
        "runtime_k_loop",
        "persistent_grid",
        "persistent_rounds",
        "runtime_channels",
        "k_atoms",
        "m_tiles",
        "preload_weights",
        "waves_per_eu",
        "row_guard",
        "prefetch_rows",
        "lds_only_sync",
        "waves_m",
        "xcd_tiles",
        "lds_pad",
        "stage_out",
        "row_exit",
    ):
        v = getattr(spec, f, None)
        if v is not None:
            d[f] = v
    return d


def conv_direct_dgrad_wmma_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`DirectDgradWmmaSpec` -> flat dict (problem nested, dtype included).
    ``sched`` is forwarded as-is; ``None`` means the backend default."""
    p = spec.problem
    return dict(
        problem=dict(
            N=p.N,
            H=p.H,
            W=p.W,
            groups=p.groups,
            cpg=p.cpg,
            kpg=p.kpg,
            KH=p.KH,
            KW=p.KW,
            PAD=p.PAD,
            stride=p.stride,
            dil_h=p.dil_h,
            dil_w=p.dil_w,
            dtype=p.dtype,
        ),
        name=spec.name,
        waves_c=spec.waves_c,
        waves_p=spec.waves_p,
        tiles_c=spec.tiles_c,
        tiles_p=spec.tiles_p,
        tile_w=spec.tile_w,
        tile_n=spec.tile_n,
        ck=spec.ck,
        stages=spec.stages,
        prefetch=spec.prefetch,
        sched=spec.sched,
        wave_size=spec.wave_size,
    )


def conv_direct_nongrouped_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`DirectNongroupedConvSpec` -> flat dict (problem nested, dtype included).
    ``iglp`` / ``waves_per_eu`` are forwarded as-is; ``None`` means "knob off"."""
    p = spec.problem
    return dict(
        problem=dict(
            N=p.N,
            H=p.H,
            W=p.W,
            groups=p.groups,
            cpg=p.cpg,
            kpg=p.kpg,
            KH=p.KH,
            KW=p.KW,
            PAD=p.PAD,
            stride=p.stride,
            dil_h=p.dil_h,
            dil_w=p.dil_w,
            dtype=p.dtype,
        ),
        name=spec.name,
        tile_h=spec.tile_h,
        tile_w=spec.tile_w,
        tile_k=spec.tile_k,
        ck=spec.ck,
        waves_m=spec.waves_m,
        waves_n=spec.waves_n,
        atom=spec.atom,
        wave_size=spec.wave_size,
        lds_pad=spec.lds_pad,
        chiplet_swizzle=spec.chiplet_swizzle,
        swizzle_wgm=spec.swizzle_wgm,
        chiplet_chunk=spec.chiplet_chunk,
        num_xcds=spec.num_xcds,
        double_buffer=spec.double_buffer,
        iglp=spec.iglp,
        waves_per_eu=spec.waves_per_eu,
        split_c=spec.split_c,
        grouped=spec.grouped,
        group_merge=spec.group_merge,
    )


def img2col_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`Img2ColSpec` -> flat dict (problem nested)."""
    return dict(
        problem=_conv_problem_to_dict(spec.problem),
        dtype=spec.dtype,
        block_tile_m=spec.block_tile_m,
        block_tile_k=spec.block_tile_k,
        vec_k=spec.vec_k,
        name=spec.name,
    )


def deep_fused_conv_pool_spec_to_dict(spec: Any) -> Dict[str, Any]:
    """:class:`DeepFusedConvPoolSpec` -> the factory-argument dict the binding
    feeds to ``rocke_make_deep_fused_conv_pool_spec``. The factory derives the
    same fields both engines do, so the flat conv/pool shape + tiling knobs are
    sufficient."""
    prob = spec.problem
    conv = prob.conv
    r = getattr(conv, "Y", None)
    if r is None:
        r = conv.R
    s = getattr(conv, "X", None)
    if s is None:
        s = conv.S
    return dict(
        n=conv.N,
        h=conv.Hi,
        w=conv.Wi,
        c=conv.C,
        k0=conv.K,
        k1=prob.conv1_k,
        r=r,
        s=s,
        pool_tile_h=spec.pool_tile_h,
        pool_tile_w=spec.pool_tile_w,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        conv1_tile_k=spec.conv1_tile_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_tile_m=spec.warp_tile_m,
        warp_tile_n=spec.warp_tile_n,
        warp_tile_k=spec.warp_tile_k,
        wave_size=spec.wave_size,
        name=spec.name,
        pipeline=spec.pipeline,
        unroll_k=spec.unroll_k,
        async_dma=spec.async_dma,
        cache_input_footprint=spec.cache_input_footprint,
        direct_conv0_from_input_cache=spec.direct_conv0_from_input_cache,
    )


def lower_conv_implicit_gemm(
    spec: Any,
    *,
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower an :class:`ImplicitGemmConvSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_implicit_gemm_conv(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = conv_implicit_gemm_spec_to_dict(spec)
    return lower_family(
        "conv_implicit_gemm",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_implicit_gemm_lower_llvm(sd, arch=arch),
        lambda: eng.conv_implicit_gemm_serialize_ir(sd, arch=arch),
        name_of(spec),
    )


def lower_conv_direct_grouped(
    spec: Any,
    *,
    kind: str = "16c",
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower a :class:`DirectConv16cSpec` / :class:`DirectConv4cSpec` /
    :class:`DirectConvSpec`. ``kind`` ("16c"|"4c"|"generic") selects the
    kernel variant."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.conv_direct_grouped import (
            build_direct_conv,
            build_direct_conv_16c,
            build_direct_conv_4c,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        build = {
            "4c": build_direct_conv_4c,
            "generic": build_direct_conv,
        }.get(kind, build_direct_conv_16c)
        k = build(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = conv_direct_grouped_spec_to_dict(spec, kind)
    return lower_family(
        "conv_direct_grouped",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_direct_grouped_lower_llvm(sd, arch=arch),
        lambda: eng.conv_direct_grouped_serialize_ir(sd, arch=arch),
        name_of(spec),
    )


def lower_conv_direct_nongrouped(
    spec: Any,
    *,
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower a :class:`DirectNongroupedConvSpec` (the ``groups == 1`` direct conv)."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.conv_direct_nongrouped import build_direct_conv_nongrouped
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_direct_conv_nongrouped(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = conv_direct_nongrouped_spec_to_dict(spec)
    return lower_family(
        "conv_direct_nongrouped",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_direct_nongrouped_lower_llvm(sd, arch=arch),
        lambda: eng.conv_direct_nongrouped_serialize_ir(sd, arch=arch),
        name_of(spec),
    )


def lower_conv_direct_dgrad_wmma(
    spec: Any,
    *,
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower a :class:`DirectDgradWmmaSpec` (the gfx1250 WMMA direct dgrad)."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.conv_direct_dgrad_wmma import build_direct_dgrad_wmma
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_direct_dgrad_wmma(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = conv_direct_dgrad_wmma_spec_to_dict(spec)
    return lower_family(
        "conv_direct_dgrad_wmma",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.conv_direct_dgrad_wmma_lower_llvm(sd, arch=arch),
        lambda: eng.conv_direct_dgrad_wmma_serialize_ir(sd, arch=arch),
        name_of(spec),
    )


def lower_img2col(
    spec: Any,
    *,
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower an :class:`Img2ColSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.img2col import build_img2col
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_img2col(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = img2col_spec_to_dict(spec)
    return lower_family(
        "img2col",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.img2col_lower_llvm(sd, arch=arch),
        lambda: eng.img2col_serialize_ir(sd, arch=arch),
        name_of(spec),
    )


def lower_deep_fused_conv_pool(
    spec: Any,
    *,
    arch: str,
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> GemmLowerResult:
    """Lower a :class:`DeepFusedConvPoolSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from kernels.common.deep_fused_conv_pool import (
            build_deep_fused_conv_pool,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_deep_fused_conv_pool(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine(ENGINE_MODULE)
    sd = deep_fused_conv_pool_spec_to_dict(spec)
    return lower_family(
        "deep_fused_conv_pool",
        spec,
        arch,
        backend,
        want_ir,
        py_fn,
        lambda: eng.deep_fused_conv_pool_lower_llvm(sd, arch=arch),
        lambda: eng.deep_fused_conv_pool_serialize_ir(sd, arch=arch),
        name_of(spec),
    )
