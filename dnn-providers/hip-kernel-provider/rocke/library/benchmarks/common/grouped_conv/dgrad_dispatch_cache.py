# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The AOT cache identity a dgrad dispatch selection names.

``dispatch.grouped_convolution`` picks gfx950 dgrad kernels among the binaries
the AOT caches build: the implicit-GEMM dgrad grid
(:mod:`benchmarks.common.kernel_sweep`) and the direct-conv grid
(:mod:`benchmarks.common.direct_kernel_sweep`). Dispatch sits below the
benchmarks layer and cannot name a ``KernelIdentity`` itself; this module turns
a selection back into the identity, so a run can load exactly that binary and
the tests can check that every selection is an identity the grid enumerates.
"""

from __future__ import annotations

from benchmarks.common.kernel_cache import KernelIdentity


def _igemm_identity(req, spec, llvm_flavor: str | None) -> KernelIdentity:
    from benchmarks.common.kernel_sweep import CACHE_DGRAD_MAX_SUB_GEMMS
    from rocke.core.arch import ArchTarget

    stride1 = 1 if spec.tap_outer_k else 0
    halo_filter = (int(req.Y), int(req.X)) if spec.dy_halo else (0, 0)
    kwargs = {} if llvm_flavor is None else {"llvm_flavor": llvm_flavor}
    return KernelIdentity(
        arch=req.arch,
        direction="dgrad",
        algorithm="implicit_gemm",
        dtype_a=spec.dtype,
        dtype_b=spec.dtype,
        dtype_d=spec.dtype,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        warp_tile_m=spec.warp_tile_mn,
        warp_tile_n=spec.warp_tile_mn,
        warp_tile_k=spec.warp_tile_k,
        pipeline=spec.pipeline,
        epilogue=spec.epilogue,
        wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        vector_size_a=spec.vector_size_a,
        vector_size_b=spec.vector_size_b,
        vector_size_c=spec.vector_size_c,
        lds_k_outer=spec.lds_k_outer,
        split_k=max(1, spec.split_k),
        uniform_k_decode=spec.uniform_k_decode,
        tap_outer_k=spec.tap_outer_k,
        dy_halo=spec.dy_halo,
        dy_halo_setprio=spec.dy_halo_setprio,
        dy_halo_kouter_pad=spec.dy_halo_kouter_pad,
        grouped=True,
        filter_h=halo_filter[0],
        filter_w=halo_filter[1],
        stride_h=stride1,
        stride_w=stride1,
        dilation_h=stride1,
        dilation_w=stride1,
        max_sub_gemms=CACHE_DGRAD_MAX_SUB_GEMMS,
        halo_max_wo=spec.halo_max_wo,
        **kwargs,
    )


def _direct_identity(spec, llvm_flavor: str | None) -> KernelIdentity:
    from dataclasses import replace

    from benchmarks.common import direct_kernel_sweep as dks
    from rocke.core.arch import ArchTarget

    caps = dks.DirectCaps(
        KH=spec.KH, PAD=spec.PAD, stride=spec.stride, cpg=spec.cpg, kpg=spec.kpg
    )
    ident = dks._identity(
        spec.arch,
        ArchTarget.from_gfx(spec.arch).wave_size,
        spec.variant,
        "direct_dgrad",
        caps,
        spec.knob_dict(),
        spec.dtype,
    )
    return ident if llvm_flavor is None else replace(ident, llvm_flavor=llvm_flavor)


def dgrad_cache_identity(req, spec, llvm_flavor: str | None = None) -> KernelIdentity:
    """The cache identity of a gfx950 dgrad selection ``spec`` for ``req``
    (``DispatchResult.request`` / ``.spec``).

    An implicit-GEMM pick (``ConvGroupedSpec``) names a binary of the
    implicit-GEMM cache, a direct pick (``ConvDirectDgradSpec``) one of the
    direct cache. ``llvm_flavor`` defaults to this process's flavor, as a
    cache lookup does.
    """
    from dispatch.grouped_convolution import ConvDirectDgradSpec, ConvGroupedSpec

    if isinstance(spec, ConvDirectDgradSpec):
        return _direct_identity(spec, llvm_flavor)
    if isinstance(spec, ConvGroupedSpec) and spec.direction == "dgrad":
        return _igemm_identity(req, spec, llvm_flavor)
    raise TypeError(f"not a dgrad selection: {type(spec).__name__}")


def dgrad_cache_problem(req, spec):
    """The problem object the cache's ``supports_problem`` / ``plan_for``
    take for ``req``: a ``DirectConvProblem`` for a direct pick, the
    ``ConvProblem`` otherwise."""
    from dispatch.grouped_convolution import (
        ConvDirectDgradSpec,
        _direct_problem,
        _problem,
    )

    if isinstance(spec, ConvDirectDgradSpec):
        return _direct_problem(req)
    return _problem(req)
