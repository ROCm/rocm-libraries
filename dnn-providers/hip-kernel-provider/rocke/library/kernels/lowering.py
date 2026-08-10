# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Backend dispatch for the convolution vertical.

The ``lower_*`` wrappers and the spec->engine-dict adapters for the conv
family used to live in :mod:`rocke.core.backend` alongside every other
family. They moved here with the kernels themselves: the platform SDK must
not import ``kernels`` (one-way layering -- library depends on platform,
never the reverse), and each ``lower_*`` wrapper necessarily reaches its
Python builder.

Nothing about the dispatch semantics changed. These call straight back into
the platform's family-agnostic driver::

    rocke.core.backend.lower_family(family, spec, arch, backend, want_ir,
                                    py_fn, cpp_ll_fn, cpp_ir_fn, spec_name)

so backend selection (``python`` / ``cpp`` / ``both``), the differential
comparison, and the error taxonomy are shared with every in-tree family.
The ``*_spec_to_dict`` adapters are duck-typed (attribute reads only) and
mirror the argument dicts the ``rocke_engine`` conv factories expect.

Import as::

    from kernels.lowering import lower_conv_implicit_gemm
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from rocke.core.backend import (
    GemmLowerResult,
    import_engine,
    lower_family,
    name_of,
)

__all__ = [
    "conv_direct_grouped_spec_to_dict",
    "conv_implicit_gemm_spec_to_dict",
    "deep_fused_conv_pool_spec_to_dict",
    "img2col_spec_to_dict",
    "lower_conv_direct_grouped",
    "lower_conv_implicit_gemm",
    "lower_deep_fused_conv_pool",
    "lower_img2col",
]


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
    """:class:`DirectConv16cSpec` / :class:`DirectConv4cSpec` -> flat dict.
    ``kind`` ("16c"|"4c") selects the binding's spec path. The 16c-only
    ``double_buffer``/``fold_k32`` fields are forwarded when present."""
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
        ),
        name=spec.name,
        block_q=spec.block_q,
        block_groups=spec.block_groups,
        wave_size=spec.wave_size,
    )
    for f in ("double_buffer", "fold_k32"):
        v = getattr(spec, f, None)
        if v is not None:
            d[f] = v
    return d


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
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower an :class:`ImplicitGemmConvSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from .common.conv_implicit_gemm import build_implicit_gemm_conv
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_implicit_gemm_conv(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine()
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
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`DirectConv16cSpec` / :class:`DirectConv4cSpec`.
    ``kind`` ("16c"|"4c") selects the channel-blocking variant."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from .common.conv_direct_grouped import (
            build_direct_conv_16c,
            build_direct_conv_4c,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        build = build_direct_conv_4c if kind == "4c" else build_direct_conv_16c
        k = build(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine()
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


def lower_img2col(
    spec: Any,
    *,
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower an :class:`Img2ColSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from .common.img2col import build_img2col
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = build_img2col(spec, arch=arch)
        ll = lower_kernel_to_llvm(k, arch=arch)
        ir = ""
        if wi:
            from rocke.core.ir_serialize import serialize

            ir = serialize(k)
        return ll, ir

    eng = import_engine()
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
    arch: str = "gfx950",
    backend: Optional[str] = None,
    want_ir: bool = False,
) -> "GemmLowerResult":
    """Lower a :class:`DeepFusedConvPoolSpec`."""

    def py_fn(wi: bool) -> Tuple[str, str]:
        from .common.deep_fused_conv_pool import (
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

    eng = import_engine()
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
