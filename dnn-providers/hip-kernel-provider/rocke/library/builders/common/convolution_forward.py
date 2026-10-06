# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 implicit-GEMM forward convolution.

The descriptor spec is flat, while the original builder takes nested problem
and data specs plus optional fusion callbacks. This adapter resolves dispatcher
defaults before packaging and calls the original builder without callbacks.
It does not change the kernel body or introduce a runtime Python dependency.

The initial contract is 2D cross-correlation, dense NHWC input, KYXC filter,
NHWK output, uniform fp16/bf16, and fp32 accumulation. Padding is symmetric.
Grouped convolution (including depthwise) is supported: the filter is
``[K, Y, X, C/groups]``, each group owns a contiguous ``C/groups`` input and
``K/groups`` output channel slab, and the group rides on grid z. Grouped
pointwise (1x1, stride 1, no padding) is declined because the kernel's flat
pointwise shortcut does not select the group's slabs. Forward needs no
workspace; pointers must be 16-byte aligned and non-aliasing.

The kernels are rocKE's AOT builds: the ABI is ``(A, B, D, A_bytes:i32,
B_bytes:i32, D_bytes:i32)`` followed by the problem block of
``kernels.common.conv_abi`` (extents, strides, magic divisors and tile counts as
i32 kernel arguments). Each packaged kernel is still built for, and matched to,
one exact problem: the builder keys code shape decisions (pointwise, grouping,
load widths) on it. :meth:`Gfx950ConvFwdSpec.launch_signature` and
:meth:`Gfx950ConvFwdSpec.launch_values` give the launch through rocKE's own
``ConvArgs``; the native pack mirrors them.

A second kernel family, rocKE's direct depthwise kernels
(``kernels.common.conv_direct_grouped``), packages through the same flat spec
when ``kernel_family`` is ``KERNEL_FAMILY_DIRECT_DEPTHWISE``. They serve pure
depthwise (``groups == C == K``) with one stride, one padding and no dilation,
take the direct-conv ABI (the same six leading arguments, then a shorter
problem block), use no LDS and no workspace, and launch over their own grid.
Their implicit-GEMM tuning fields hold fixed placeholders (``tile_k`` is 0, so
a forced ``tile_k`` of 64 or 128 never selects one). rocKE itself refuses any
forward direct shape without an odd filter and "same" padding, because the row
stream then writes wrong rows. :func:`_direct_error` additionally requires the
last input column to feed the last output column, which a rectangular filter
can break.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped
from rocke.core.ir import KernelDef
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_implicit_gemm as _conv
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
from kernels.common.conv_args import MUL24_REDUCTION_LIMIT, ConvArgs

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1
_MAX_GRID_DIM_Z = 65535
# rocKE's grid bounds for the direct kernels: every axis at most 65535.
_MAX_GRID_DIM = 65535
_PACKAGED_PIPELINES = ("mem", "compv3", "compv4", "basic")

#: ``kernel_family`` values; the metadata field is also an integer engine knob.
KERNEL_FAMILY_IMPLICIT_GEMM = 0
KERNEL_FAMILY_DIRECT_DEPTHWISE = 1
#: ``direct_variant`` values: implicit GEMM carries "none"; ``std`` is
#: DirectDepthwiseSpec (one channel per lane, block_w output columns per block);
#: ``spatial`` is DirectDepthwiseSpatialSpec (groups < 64 channels and
#: 64 // groups output columns per wave).
DIRECT_VARIANT_NONE = "none"
DIRECT_VARIANT_STD = "std"
DIRECT_VARIANT_SPATIAL = "spatial"
_DIRECT_VARIANTS = (DIRECT_VARIANT_STD, DIRECT_VARIANT_SPATIAL)
_DIRECT_WAVE_SIZE = 64
_DIRECT_MAX_BLOCK_WAVES = 16
#: Implicit-GEMM fields a direct kernel does not read; each holds exactly this.
_DIRECT_PLACEHOLDERS = {
    "tile_m": 0,
    "tile_n": 0,
    "tile_k": 0,
    "warp_m": 0,
    "warp_n": 0,
    "warp_tile_m": 0,
    "warp_tile_n": 0,
    "warp_tile_k": 0,
    "wave_size": _DIRECT_WAVE_SIZE,
    "pipeline": "none",
    "epilogue": "none",
}
#: Family fields appended after the original 27; at these values an
#: implicit-GEMM spec hashes exactly as it did before they existed.
_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    "block_w": 0,
    "block_waves": 0,
}
#: Default direct arm when the caller names none (see
#: :func:`gfx950_conv_fwd_direct_spec_for_request`).
DEFAULT_DIRECT_BLOCK_W = 4
DEFAULT_DIRECT_BLOCK_WAVES = 1


@dataclass(frozen=True)
class Gfx950ConvFwdSpec:
    """Serializable problem and resolved tuning values for one packaged kernel.

    Use :func:`gfx950_conv_fwd_spec_for_request` to obtain the dispatcher's
    current defaults, or :func:`gfx950_conv_fwd_direct_spec_for_request` for a
    direct depthwise kernel. Direct construction is useful for descriptor
    hydration; :func:`supports_gfx950_conv_fwd` and the builder validate it
    before emission. All geometry is compiled into the kernel and must match the
    runtime graph.

    The trailing family fields default to the implicit-GEMM family, so a spec
    serialized before they existed hydrates and hashes unchanged.
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
    kernel_family: int = KERNEL_FAMILY_IMPLICIT_GEMM
    direct_variant: str = DIRECT_VARIANT_NONE
    block_w: int = 0
    block_waves: int = 0

    @property
    def is_direct(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_DIRECT_DEPTHWISE

    def _name_suffix(self) -> str:
        values = asdict(self)
        if not self.is_direct and all(
            values[name] == default for name, default in _FAMILY_DEFAULTS.items()
        ):
            # Existing implicit-GEMM symbols predate the family fields; leaving
            # them out at their defaults keeps every shipped symbol unchanged.
            for name in _FAMILY_DEFAULTS:
                del values[name]
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

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
        if self.is_direct:
            raise ValueError("a direct depthwise spec has no implicit-GEMM instance")
        # ConvProblem.short() omits stride, padding and dilation, and the
        # original kernel name omits dtype. Hash every flat field so otherwise
        # identical shape/tile labels cannot collide in the packaged catalog.
        return _conv.ImplicitGemmConvSpec(
            problem=self.to_problem(),
            name=f"hkp_conv_fwd_gfx950_{self._name_suffix()}",
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

    def to_direct_problem(self) -> _direct.DirectConvProblem:
        """The direct kernels' problem: one stride and one padding for H and W."""
        return _direct.DirectConvProblem(
            N=self.N,
            H=self.Hi,
            W=self.Wi,
            groups=self.groups,
            cpg=self.C // self.groups,
            kpg=self.K // self.groups,
            KH=self.Y,
            KW=self.X,
            PAD=self.pH,
            stride=self.sH,
            dtype=self.dtype,
        )

    def to_direct_spec(
        self,
    ) -> _direct.DirectDepthwiseSpec | _direct.DirectDepthwiseSpatialSpec:
        """Reconstruct the direct depthwise builder spec under a unique name."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM spec has no direct depthwise instance")
        # The direct kernels' own names omit the filter size, padding and
        # stride, so distinct kernels would share a symbol. Hash every field.
        name = f"hkp_conv_fwd_dw_gfx950_{self._name_suffix()}"
        if self.direct_variant == DIRECT_VARIANT_SPATIAL:
            return _direct.DirectDepthwiseSpatialSpec(
                problem=self.to_direct_problem(),
                name=name,
                block_waves=self.block_waves,
                wave_size=self.wave_size,
            )
        return _direct.DirectDepthwiseSpec(
            problem=self.to_direct_problem(),
            name=name,
            block_w=self.block_w,
            block_waves=self.block_waves,
            wave_size=self.wave_size,
        )

    def kernel_name(self) -> str:
        """The actual launch symbol emitted by the original builder."""
        if self.is_direct:
            return self.to_direct_spec().kernel_name()
        return self.to_instance_spec().kernel_name()

    def direct_block_w(self) -> int:
        """Output columns per direct workgroup; spatial derives it from groups."""
        if self.direct_variant == DIRECT_VARIANT_SPATIAL:
            return self.block_waves * (self.wave_size // self.groups)
        return self.block_w

    def grid(self) -> tuple[int, int, int]:
        """Grid in workgroups.

        Implicit GEMM: N-tiles over K/groups, M-tiles, groups. Direct std:
        ``(ceil(Wo/block_w), ceil(groups/(64*block_waves)), N)``. Direct
        spatial: ``(ceil(Wo/(block_waves*(64//groups))), 1, N)``. Both direct
        grids mirror ``benchmarks/common/benchmark_direct_conv.py``.
        """
        if not self.is_direct:
            return _conv.implicit_gemm_conv_grid(self.to_instance_spec())
        wo = self.to_direct_problem().Wo
        q_tiles = -(-wo // self.direct_block_w())
        if self.direct_variant == DIRECT_VARIANT_SPATIAL:
            return q_tiles, 1, self.N
        block_ch = self.block_waves * self.wave_size
        return q_tiles, -(-self.groups // block_ch), self.N

    def block(self) -> tuple[int, int, int]:
        if self.is_direct:
            return self.block_waves * self.wave_size, 1, 1
        return self.warp_m * self.warp_n * self.wave_size, 1, 1

    def _conv_args(self) -> ConvArgs:
        if self.is_direct:
            return ConvArgs.from_problem(self.to_direct_problem())
        return ConvArgs.from_problem(
            self.to_problem(), tile_m=self.tile_m, tile_n=self.tile_n
        )

    def launch_signature(self) -> list[dict]:
        """The kernel's launch signature in ``kernels.common.conv_abi`` order."""
        if self.is_direct:
            return conv_direct_args_signature(self.dtype)
        return conv_args_signature(self.dtype)

    def launch_values(
        self,
        a_ptr: int,
        b_ptr: int,
        d_ptr: int,
        a_bytes: int,
        b_bytes: int,
        d_bytes: int,
    ) -> dict[str, int]:
        """Every kernel argument, keyed by :meth:`launch_signature`'s names.

        Computed by rocKE's ``ConvArgs``, which checks the result against the ABI.
        """
        return self._conv_args().to_launch_values(
            a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes
        )


def _problem_error(spec: Gfx950ConvFwdSpec) -> str:
    for name in ("N", "Hi", "Wi", "C", "K", "Y", "X", "sH", "sW", "dH", "dW"):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    for name in ("pH", "pW"):
        value = getattr(spec, name)
        if type(value) is not int or not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    # Checked before to_problem(): ConvProblem raises on non-divisible channels,
    # and callers expect a declined reason rather than an exception.
    if type(spec.groups) is not int or not 0 < spec.groups <= _MAX_GRID_DIM_Z:
        return f"groups must be an integer in [1, {_MAX_GRID_DIM_Z}] (grid z)"
    if spec.C % spec.groups != 0:
        return f"C={spec.C} is not divisible by groups={spec.groups}"
    if spec.K % spec.groups != 0:
        return f"K={spec.K} is not divisible by groups={spec.groups}"
    if spec.layout != "NHWC":
        return "the packaged forward contract requires dense NHWC storage"
    if spec.dtype not in ("fp16", "bf16"):
        return "the packaged forward contract supports fp16 or bf16 only"
    # Mirrors the 2D ConvProblem.is_pointwise predicate. That flat shortcut
    # indexes A and D as per-group matrices with no group slab offset, so
    # grouped pointwise computes wrong results.
    pointwise = (
        spec.Y == spec.X == 1 and spec.sH == spec.sW == 1 and spec.pH == spec.pW == 0
    )
    if spec.groups > 1 and pointwise:
        return (
            "grouped pointwise convolution (1x1 filter, stride 1, no padding) "
            "is not supported: the kernel's pointwise shortcut ignores groups"
        )

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
        ("filter", spec.K * spec.Y * spec.X * (spec.C // spec.groups)),
        ("output", spec.N * problem.Ho * problem.Wo * spec.K),
    ):
        if elements * 2 > _INT32_MAX:
            return f"{label} byte count exceeds the signed 32-bit kernel ABI"
    return ""


def _direct_rows_covered(
    extent: int, padding: int, filter_size: int, stride: int
) -> bool:
    """Whether the direct kernels' input stream reaches the last output row.

    Both kernels stream ``extent + filter_size - 1`` input rows and flush
    output row ``p // stride`` for input row ``p < extent``, so the last row
    they write is ``(extent - 1) // stride``. Fewer output rows than that
    makes the runtime H loop write into the next image; more leaves rows
    unwritten. The same rule is applied to the width.
    """
    output = (extent + 2 * padding - filter_size) // stride + 1
    return (extent - 1) // stride == output - 1


def _direct_error(spec: Gfx950ConvFwdSpec) -> str:
    """Integration guard for the direct depthwise family; "" when accepted.

    Runs after :func:`_problem_error`, so geometry is already positive and
    byte counts fit the ABI. rocKE's validators, which :func:`_direct_valid`
    runs afterwards, also refuse every shape without an odd filter and
    "same" padding; the coverage rule here still guards the width.
    """
    if not (spec.groups == spec.C == spec.K):
        return (
            "the direct depthwise family requires groups == C == K "
            f"(got groups={spec.groups}, C={spec.C}, K={spec.K})"
        )
    if spec.sH != spec.sW:
        return "the direct depthwise family requires sH == sW"
    if spec.pH != spec.pW:
        return "the direct depthwise family requires pH == pW"
    if spec.dH != 1 or spec.dW != 1:
        return "the direct depthwise family requires dH == dW == 1"
    for label, extent, filter_size in (
        ("height", spec.Hi, spec.Y),
        ("width", spec.Wi, spec.X),
    ):
        if not _direct_rows_covered(extent, spec.pH, filter_size, spec.sH):
            return (
                f"direct depthwise output {label} is not covered by its input "
                "stream: floor((in-1)/stride) must equal out-1 (rocKE writes "
                "wrong rows otherwise)"
            )
    for name, value in _DIRECT_PLACEHOLDERS.items():
        actual = getattr(spec, name)
        if actual != value or type(actual) is not type(value):
            return (
                f"a direct depthwise spec must set {name} to {value!r}, got {actual!r}"
            )
    if spec.direct_variant not in _DIRECT_VARIANTS:
        return (
            f"direct_variant must be one of {_DIRECT_VARIANTS} for the direct "
            f"depthwise family, got {spec.direct_variant!r}"
        )
    if (
        type(spec.block_waves) is not int
        or not 1 <= spec.block_waves <= _DIRECT_MAX_BLOCK_WAVES
    ):
        return f"block_waves must be an integer in [1, {_DIRECT_MAX_BLOCK_WAVES}]"
    if spec.direct_variant == DIRECT_VARIANT_SPATIAL:
        if spec.groups >= _DIRECT_WAVE_SIZE:
            return (
                f"the spatial direct variant requires groups < {_DIRECT_WAVE_SIZE}, "
                f"got {spec.groups}"
            )
        if type(spec.block_w) is not int or spec.block_w != 0:
            return "the spatial direct variant derives block_w; it must be 0"
    elif type(spec.block_w) is not int or not 0 < spec.block_w <= _INT32_MAX:
        return "block_w must be a positive signed 32-bit integer"
    for axis, extent in zip("xyz", spec.grid()):
        if extent > _MAX_GRID_DIM:
            return f"direct grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return ""


def _direct_valid(spec: Gfx950ConvFwdSpec, arch: str) -> tuple[bool, str]:
    """rocKE's own validators for the selected direct kernel."""
    try:
        instance = spec.to_direct_spec()
        instance.validate()
        if spec.direct_variant == DIRECT_VARIANT_SPATIAL:
            return _direct.is_valid_depthwise_spatial_spec(instance, arch=arch)
        return _direct.is_valid_depthwise_spec(instance, arch=arch)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)


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
    if type(spec.kernel_family) is not int or spec.kernel_family not in (
        KERNEL_FAMILY_IMPLICIT_GEMM,
        KERNEL_FAMILY_DIRECT_DEPTHWISE,
    ):
        return False, (
            f"kernel_family must be {KERNEL_FAMILY_IMPLICIT_GEMM} (implicit GEMM) or "
            f"{KERNEL_FAMILY_DIRECT_DEPTHWISE} (direct depthwise), "
            f"got {spec.kernel_family!r}"
        )
    if spec.is_direct:
        error = _direct_error(spec)
        if error:
            return False, error
        return _direct_valid(spec, arch)
    for name, default in _FAMILY_DEFAULTS.items():
        if getattr(spec, name) != default or type(getattr(spec, name)) is not type(
            default
        ):
            return False, (
                f"an implicit-GEMM spec must leave {name} at {default!r}, "
                f"got {getattr(spec, name)!r}"
            )
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
    # Every pipeline here launches warp_m * warp_n * wave_size threads over the same
    # (N-tiles, M-tiles, groups) grid with static LDS, which is the launch the native
    # pack computes. "wavelet" appends load waves to the block and is WMMA-only anyway.
    if spec.pipeline not in _PACKAGED_PIPELINES:
        return False, (
            f"the packaged forward contract supports pipelines {_PACKAGED_PIPELINES}, "
            f"got {spec.pipeline!r}"
        )
    if spec.epilogue not in ("default", "cshuffle"):
        return False, "epilogue must be 'default' or 'cshuffle'"
    # ConvArgs refuses this at launch; refuse it before a kernel is built instead.
    if spec.Y * spec.X * (spec.C // spec.groups) >= MUL24_REDUCTION_LIMIT:
        return False, (
            "the reduction extent Y*X*C/groups must stay below 2**23 for the "
            "kernel's 24-bit address products"
        )
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

    The request interface also describes 3D/backward convolution. Those
    requests, and grouped pointwise, are rejected here as integration gaps, even
    when another rocKE dispatcher candidate can serve them. Grouped 2D forward
    (``G > 1``, including depthwise) is accepted.
    """
    base = _base_spec_for_request(request)
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


def gfx950_conv_fwd_direct_spec_for_request(
    request: ConvGroupedRequest,
    *,
    block_w: int | None = None,
    block_waves: int | None = None,
) -> Gfx950ConvFwdSpec:
    """A direct depthwise spec for ``request``; rocKE has no dispatcher for it.

    ``block_w`` selects the variant: ``None`` applies the default policy,
    ``0`` selects the spatial kernel (which derives its own block width) and a
    positive value selects the standard kernel with that many output columns
    per block. The default policy is the spatial kernel when ``G < 64``,
    otherwise the standard kernel with ``block_w=4``; ``block_waves`` defaults
    to 1. Block widths of 8 and above hit a load-scheduling cliff on the
    standard kernel, so the default stays below it. Raises ``ValueError`` when
    the request is outside the guarded direct contract.
    """
    base = _base_spec_for_request(request)
    if block_w is None:
        block_w = 0 if base.groups < _DIRECT_WAVE_SIZE else DEFAULT_DIRECT_BLOCK_W
    spec = replace(
        base,
        **_DIRECT_PLACEHOLDERS,
        kernel_family=KERNEL_FAMILY_DIRECT_DEPTHWISE,
        direct_variant=DIRECT_VARIANT_SPATIAL if block_w == 0 else DIRECT_VARIANT_STD,
        block_w=block_w,
        block_waves=DEFAULT_DIRECT_BLOCK_WAVES if block_waves is None else block_waves,
    )
    ok, reason = supports_gfx950_conv_fwd(spec, arch=request.arch)
    if not ok:
        raise ValueError(reason)
    return spec


def _base_spec_for_request(request: ConvGroupedRequest) -> Gfx950ConvFwdSpec:
    """The request's problem fields under the packaged contract's request checks."""
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
    return base


def build_gfx950_conv_fwd(spec: Gfx950ConvFwdSpec, *, arch: str = _ARCH) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point consumed by hkp_pack.

    Dispatches on ``kernel_family``. Direct kernels must be lowered with rocKE's
    Python backend (hkp_pack pins it): the C++ lowering is superlinear in the
    unrolled kernel size.
    """
    ok, reason = supports_gfx950_conv_fwd(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged forward convolution: {reason}")
    if spec.is_direct:
        instance = spec.to_direct_spec()
        if spec.direct_variant == DIRECT_VARIANT_SPATIAL:
            return _direct.build_direct_depthwise_spatial(instance, arch=arch)
        return _direct.build_direct_depthwise(instance, arch=arch)
    return _conv.build_implicit_gemm_conv(spec.to_instance_spec(), arch=arch)
