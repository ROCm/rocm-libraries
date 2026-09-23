# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 implicit-GEMM forward convolution.

The descriptor spec is flat, while the original builder takes nested problem
and data specs plus optional fusion callbacks. This adapter resolves dispatcher
defaults before packaging and calls the original builder without callbacks.
It does not change the kernel body or introduce a runtime Python dependency.

The initial contract is 2D cross-correlation, dense NHWC input, KYXC filter,
NHWK output, groups=1, uniform fp16/bf16, and fp32 accumulation. Padding is
symmetric. The ABI is ``(A, B, D, A_bytes:i32, B_bytes:i32, D_bytes:i32)``;
pointers must be 16-byte aligned and non-aliasing. Forward needs no workspace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped
from rocke.core.ir import KernelDef
from kernels.common import conv_implicit_gemm as _conv

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class Gfx950ConvFwdSpec:
    """Serializable problem and resolved tuning values for one packaged kernel.

    Use :func:`gfx950_conv_fwd_spec_for_request` to obtain the dispatcher's
    current defaults. Direct construction is useful for descriptor hydration;
    :func:`supports_gfx950_conv_fwd` and the builder validate it before emission.
    All geometry is compiled into the kernel and must match the runtime graph.
    """

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    Y: int
    X: int
    sH: int = 1
    sW: int = 1
    pH: int = 0
    pW: int = 0
    dH: int = 1
    dW: int = 1
    dtype: str = "fp16"
    layout: str = "NHWC"
    groups: int = 1
    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 64
    warp_m: int = 2
    warp_n: int = 2
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16
    wave_size: int = 64
    pipeline: str = "mem"
    epilogue: str = "cshuffle"

    def to_problem(self) -> _conv.ConvProblem:
        """Reconstruct the original problem without dropping geometry fields."""
        return _conv.ConvProblem(
            N=self.N,
            Hi=self.Hi,
            Wi=self.Wi,
            C=self.C,
            K=self.K,
            Y=self.Y,
            X=self.X,
            sH=self.sH,
            sW=self.sW,
            pH=self.pH,
            pW=self.pW,
            dH=self.dH,
            dW=self.dW,
            groups=self.groups,
        )

    def to_instance_spec(self) -> _conv.ImplicitGemmConvSpec:
        """Reconstruct the original builder spec; optional fusion stays disabled."""
        # ConvProblem.short() omits stride, padding and dilation, and the
        # original kernel name omits dtype. Hash every flat field so otherwise
        # identical shape/tile labels cannot collide in the packaged catalog.
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return _conv.ImplicitGemmConvSpec(
            problem=self.to_problem(),
            name=f"hkp_conv_fwd_gfx950_{suffix}",
            data=_conv.ConvDataSpec(
                dtype_a=self.dtype,
                dtype_b=self.dtype,
                dtype_d=self.dtype,
                dtype_acc="fp32",
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            wave_size=self.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            groups=self.groups,
        )

    def kernel_name(self) -> str:
        """The actual launch symbol emitted by the original builder."""
        return self.to_instance_spec().kernel_name()

    def grid(self) -> tuple[int, int, int]:
        """Grid in workgroups, ordered N-tiles, M-tiles, group."""
        return _conv.implicit_gemm_conv_grid(self.to_instance_spec())

    def block(self) -> tuple[int, int, int]:
        return self.warp_m * self.warp_n * self.wave_size, 1, 1


def _problem_error(spec: Gfx950ConvFwdSpec) -> str:
    for name in ("N", "Hi", "Wi", "C", "K", "Y", "X", "sH", "sW", "dH", "dW"):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    for name in ("pH", "pW"):
        value = getattr(spec, name)
        if type(value) is not int or not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    if type(spec.groups) is not int or spec.groups != 1:
        return "the packaged forward contract requires groups=1"
    if spec.layout != "NHWC":
        return "the packaged forward contract requires dense NHWC storage"
    if spec.dtype not in ("fp16", "bf16"):
        return "the packaged forward contract supports fp16 or bf16 only"

    problem = spec.to_problem()
    for label, value in (
        ("padded height", spec.Hi + 2 * spec.pH),
        ("padded width", spec.Wi + 2 * spec.pW),
        ("effective filter height", spec.dH * (spec.Y - 1) + 1),
        ("effective filter width", spec.dW * (spec.X - 1) + 1),
    ):
        if value > _INT32_MAX:
            return f"{label} exceeds signed 32-bit descriptor arithmetic"
    if problem.Ho <= 0 or problem.Wo <= 0:
        return "convolution has nonpositive output spatial dimensions"
    for label, elements in (
        ("input", spec.N * spec.Hi * spec.Wi * spec.C),
        ("filter", spec.K * spec.Y * spec.X * spec.C),
        ("output", spec.N * problem.Ho * problem.Wo * spec.K),
    ):
        if elements * 2 > _INT32_MAX:
            return f"{label} byte count exceeds the signed 32-bit kernel ABI"
    return ""


def supports_gfx950_conv_fwd(
    spec: Gfx950ConvFwdSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Check the integration contract, then rocKE's real problem/arch predicate."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx950ConvFwdSpec):
        return False, "expected Gfx950ConvFwdSpec"
    error = _problem_error(spec)
    if error:
        return False, error
    for name in (
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "wave_size",
    ):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return False, f"{name} must be a positive signed 32-bit integer"
    if spec.pipeline != "mem":
        return False, "the packaged forward contract currently requires pipeline='mem'"
    if spec.epilogue not in ("default", "cshuffle"):
        return False, "epilogue must be 'default' or 'cshuffle'"
    try:
        instance = spec.to_instance_spec()
        instance.validate()
        return _conv.is_valid_spec_for_problem(instance, instance.problem, arch=arch)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)


def gfx950_conv_fwd_spec_for_request(
    request: ConvGroupedRequest, *, tile_k: int | None = None
) -> Gfx950ConvFwdSpec:
    """Resolve the existing dispatcher's defaults and optionally override tile_k.

    The request interface also describes grouped/3D/backward convolution.
    Those requests are rejected here as integration gaps, even when another
    rocKE dispatcher candidate can serve them.
    """
    if not isinstance(request, ConvGroupedRequest):
        raise TypeError("expected ConvGroupedRequest")
    if request.arch != _ARCH:
        raise ValueError(f"this adapter requires arch={_ARCH}")
    if request.direction != "fwd":
        raise ValueError("the packaged contract supports forward convolution only")
    if any(
        value is not None
        for value in (
            request.Di,
            request.Z,
            request.stride_d,
            request.pad_d,
            request.dilation_d,
        )
    ):
        raise ValueError("the packaged forward contract supports 2D convolution only")
    if request.vec_size_c is not None:
        raise ValueError("the packaged contract uses the dispatcher's vector defaults")
    base = Gfx950ConvFwdSpec(
        N=request.N,
        Hi=request.Hi,
        Wi=request.Wi,
        C=request.C,
        K=request.K,
        Y=request.Y,
        X=request.X,
        sH=request.stride_h,
        sW=request.stride_w,
        pH=request.pad_h,
        pW=request.pad_w,
        dH=request.dilation_h,
        dW=request.dilation_w,
        groups=request.G,
        dtype=request.dtype.lower(),
        layout=request.layout.upper(),
    )
    error = _problem_error(base)
    if error:
        raise ValueError(error)
    selected = dispatch_conv_grouped(request).spec
    spec = replace(
        base,
        tile_m=selected.tile_m,
        tile_n=selected.tile_n,
        tile_k=selected.tile_k if tile_k is None else tile_k,
        warp_m=selected.warp_m,
        warp_n=selected.warp_n,
        warp_tile_m=selected.warp_tile_mn,
        warp_tile_n=selected.warp_tile_mn,
        warp_tile_k=selected.warp_tile_k,
        wave_size=selected.to_fwd_spec(base.to_problem()).wave_size,
        pipeline=selected.pipeline,
        epilogue=selected.epilogue,
    )
    ok, reason = supports_gfx950_conv_fwd(spec, arch=request.arch)
    if not ok:
        raise ValueError(reason)
    return spec


def build_gfx950_conv_fwd(spec: Gfx950ConvFwdSpec, *, arch: str = _ARCH) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point consumed by hkp_pack."""
    ok, reason = supports_gfx950_conv_fwd(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged forward convolution: {reason}")
    return _conv.build_implicit_gemm_conv(spec.to_instance_spec(), arch=arch)
