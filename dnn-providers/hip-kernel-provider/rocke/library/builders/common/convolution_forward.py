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
i32 kernel arguments). A packaged kernel is built for one problem, but it
computes every problem its build-time capabilities admit, and the native pack
matches it to all of them. :func:`gfx950_conv_fwd_capabilities` reports what a
binary keeps from its build problem (grouped and pointwise code paths, load and
store widths; for the direct kernels the filter, stride and padding) and
:func:`gfx950_conv_fwd_serves` decides a
runtime problem against it. :meth:`Gfx950ConvFwdSpec.launch_signature` and
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
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace

from dispatch.grouped_convolution import ConvGroupedRequest, dispatch_conv_grouped
from rocke.core.ir import KernelDef
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_direct_nongrouped as _nongrouped
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
    before emission. The geometry is the build problem; the kernel takes the
    runtime problem as AOT arguments (see :func:`gfx950_conv_fwd_serves`).

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


def _problem_error(
    spec: Gfx950ConvFwdSpec, *, decline_grouped_pointwise: bool = True
) -> str:
    """The packaged contract's problem checks; "" when accepted.

    A kernel is never built for a grouped pointwise problem (rocKE's pointwise
    shortcut ignores groups), so that is declined by default. A runtime problem
    passes ``decline_grouped_pointwise=False``: grouped binaries built against
    a non-pointwise problem compute it through their descriptors.
    """
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
    if decline_grouped_pointwise and spec.groups > 1 and pointwise:
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


def _base_spec_for_request(
    request: ConvGroupedRequest, *, decline_grouped_pointwise: bool = True
) -> Gfx950ConvFwdSpec:
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
    error = _problem_error(base, decline_grouped_pointwise=decline_grouped_pointwise)
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


# ---------------------------------------------------------------------------
# What a packaged binary computes at runtime
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950ConvFwdCapabilities:
    """What a packaged binary keeps from the problem it was built for.

    Implicit GEMM: ``load_width`` (A and B, along C/groups), ``store_width``
    (the cshuffle store along K/groups; 1 for the scalar default epilogue),
    and the ``pointwise`` and ``grouped`` code paths. Direct depthwise: the
    compiled filter, stride and padding; batch, image and group count are
    kernel arguments for both variants.
    """

    load_width: int = 0
    store_width: int = 0
    pointwise: bool = False
    grouped: bool = False
    filter_hw: tuple[int, int] = (0, 0)
    stride_hw: tuple[int, int] = (0, 0)
    pad_hw: tuple[int, int] = (0, 0)


def _is_pointwise(spec: Gfx950ConvFwdSpec) -> bool:
    return spec.Y == spec.X == 1 and spec.sH == spec.sW == 1 and spec.pH == spec.pW == 0


def gfx950_conv_fwd_capabilities(spec: Gfx950ConvFwdSpec) -> Gfx950ConvFwdCapabilities:
    """The build-time capabilities of the binary ``spec`` describes.

    The widths are the ones rocKE's builder resolves for the build problem
    (``_sync_load_vecs`` and ``default_vector_sizes``), so they are exactly what
    the binary bakes in. The native pack's ``conv::implicitGemmCapabilities``
    recomputes them from the same metadata.
    """
    if spec.is_direct:
        return Gfx950ConvFwdCapabilities(
            filter_hw=(spec.Y, spec.X),
            stride_hw=(spec.sH, spec.sW),
            pad_hw=(spec.pH, spec.pW),
        )
    instance = spec.to_instance_spec()
    load_width, _ = _conv._sync_load_vecs(instance)
    store_width = 1
    if spec.epilogue == "cshuffle":
        _, _, store_width = _conv.ImplicitGemmConvSpec.default_vector_sizes(
            spec.C // spec.groups, spec.K // spec.groups, spec.dtype
        )
    return Gfx950ConvFwdCapabilities(
        load_width=load_width,
        store_width=store_width,
        pointwise=_is_pointwise(spec),
        grouped=spec.groups > 1,
    )


def gfx950_conv_fwd_serves(
    spec: Gfx950ConvFwdSpec, problem: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """Whether the binary ``spec`` describes computes ``problem``.

    ``problem`` is read for its problem fields only. Implicit GEMM mirrors the
    forward subset of rocKE's ``KernelCache.supports_problem``: the load width
    divides C/groups, the store width K/groups, a grouped problem needs the
    grouped code path (which computes groups == 1 too), and the pointwise GEMM
    serves only pointwise problems. Direct depthwise needs the build's filter,
    stride and padding, and the runtime problem must pass the direct guard. Both families then need a grid
    and kernel arguments that fit.
    """
    error = _problem_error(problem, decline_grouped_pointwise=False)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != kernel dtype {spec.dtype}"
    caps = gfx950_conv_fwd_capabilities(spec)
    if spec.is_direct:
        runtime = replace(
            spec, **{f.name: getattr(problem, f.name) for f in fields(_ProblemFields)}
        )
        return _direct_serves(caps, runtime)
    return _implicit_gemm_serves(caps, problem, spec.tile_m, spec.tile_n)


def _direct_serves(
    caps: Gfx950ConvFwdCapabilities, runtime: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """Whether a direct depthwise binary with ``caps`` computes ``runtime``: the
    runtime problem on the binary's direct tuning, which already passed the problem
    checks. It must pass the direct guard and have the compiled filter, stride and
    padding (the guard already pins dilation to 1).

    Shared by :func:`gfx950_conv_fwd_serves` and :func:`gfx950_conv_fwd_aot_serves`.
    """
    error = _direct_error(runtime)
    if error:
        return False, error
    for label, baked, actual in (
        ("filter", caps.filter_hw, (runtime.Y, runtime.X)),
        ("stride", caps.stride_hw, (runtime.sH, runtime.sW)),
        ("padding", caps.pad_hw, (runtime.pH, runtime.pW)),
    ):
        if baked != actual:
            return (
                False,
                f"direct kernel compiled for {label} {baked}, problem has {actual}",
            )
    return True, "ok"


def _implicit_gemm_serves(
    caps: Gfx950ConvFwdCapabilities,
    problem: Gfx950ConvFwdSpec,
    tile_m: int,
    tile_n: int,
) -> tuple[bool, str]:
    """Whether an implicit-GEMM binary with ``caps`` and this tile computes
    ``problem``, which already passed the problem checks.

    Shared by :func:`gfx950_conv_fwd_serves` and :func:`gfx950_conv_fwd_aot_serves`.
    """
    cpg = problem.C // problem.groups
    kpg = problem.K // problem.groups
    if problem.groups > 1 and not caps.grouped:
        return False, "kernel is ungrouped, problem is grouped"
    if caps.pointwise and not _is_pointwise(problem):
        return False, "kernel is the pointwise GEMM, problem is not pointwise"
    if cpg % caps.load_width:
        return False, f"cpg={cpg} not divisible by the load width {caps.load_width}"
    if kpg % caps.store_width:
        return False, f"kpg={kpg} not divisible by the store width {caps.store_width}"
    if problem.Y * problem.X * cpg >= MUL24_REDUCTION_LIMIT:
        return False, "the reduction extent Y*X*C/groups must stay below 2**23"
    args = ConvArgs.from_problem(problem.to_problem(), tile_m=tile_m, tile_n=tile_n)
    for axis, extent in zip("xyz", args.grid()):
        if extent > _MAX_GRID_DIM:
            return False, f"grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return True, "ok"


@dataclass(frozen=True)
class _ProblemFields:
    """The problem half of Gfx950ConvFwdSpec, for moving a runtime problem onto a
    kernel's tuning."""

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    Y: int
    X: int
    sH: int
    sW: int
    pH: int
    pW: int
    dH: int
    dW: int
    groups: int


# ---------------------------------------------------------------------------
# Shape-generic AOT instances
# ---------------------------------------------------------------------------
#
# rocKE's AOT conv kernels take the problem as kernel arguments, so one binary
# serves every problem its build-time capabilities admit. An instance is one
# forward rocKE ``KernelIdentity`` (``benchmarks/common/kernel_cache.py``):
# tuning and capability fields, no problem extents. Two families are packaged:
# the implicit-GEMM identities of the AOT sweep (``kernel_sweep.py``) and the
# forward identities of the direct sweep (``direct_kernel_sweep.py``), one
# direct variant per direct algorithm (``AOT_DIRECT_ALGORITHMS``).
# ``instance_id`` is the identity's ``stable_hash()``; this layer cannot import
# ``benchmarks``, so the catalog tool computes it and ``library/tests`` checks
# the correspondence.

#: Vector widths rocKE's AOT sweep builds (``CACHE_VECS``).
_AOT_VECTOR_WIDTHS = (1, 2, 4, 8)
#: Pipelines with the packaged launch (see ``_PACKAGED_PIPELINES``).
_AOT_PIPELINES = _PACKAGED_PIPELINES
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_AOT_INSTANCE_ID_LENGTH = 40
#: The implicit-GEMM fields of a direct instance hold exactly what the direct
#: sweep writes into a direct identity (``direct_kernel_sweep._identity``).
_AOT_DIRECT_PLACEHOLDERS = {
    "tile_m": 0,
    "tile_n": 0,
    "tile_k": 0,
    "warp_m": 0,
    "warp_n": 0,
    "warp_tile_m": 0,
    "warp_tile_n": 0,
    "warp_tile_k": 0,
    "pipeline": "direct",
    "epilogue": "direct",
    "vector_size_a": 0,
    "vector_size_b": 0,
    "vector_size_c": 0,
    "unroll_k": 0,
    "async_dma": 0,
    "async_chunk_a": 0,
    "async_chunk_b": 0,
    "grouped": 0,
}

# The direct variants only the AOT engine packages.
#: DirectDepthwiseTiledSpec: an output-stationary block_h x block_w tile per block.
DIRECT_VARIANT_TILED = "tiled"
#: DirectDepthwiseColSpec: the filter-column loop outermost; it bounds its rows by
#: the runtime Ho, so it also serves "valid" padding.
DIRECT_VARIANT_COL = "col"
#: DirectConvSpec: cpg == kpg in {4, 8, 16, 32}, one wave per group.
DIRECT_VARIANT_GROUPED = "grouped"
#: DirectNongroupedConvSpec: groups == 1, C and K runtime.
DIRECT_VARIANT_NONGROUPED = "nongrouped"
#: Each forward algorithm of ``direct_kernel_sweep.DIRECT_CAPABILITIES`` and the
#: ``direct_variant`` an instance of it carries.
AOT_DIRECT_ALGORITHMS = {
    "direct_depthwise": DIRECT_VARIANT_STD,
    "direct_depthwise_spatial": DIRECT_VARIANT_SPATIAL,
    "direct_depthwise_tiled": DIRECT_VARIANT_TILED,
    "direct_depthwise_col": DIRECT_VARIANT_COL,
    "direct_grouped": DIRECT_VARIANT_GROUPED,
    "direct_nongrouped": DIRECT_VARIANT_NONGROUPED,
}
_AOT_DIRECT_VARIANTS = tuple(AOT_DIRECT_ALGORITHMS.values())
#: The block knobs after ``block_w``/``block_waves`` and the value each holds
#: where its variant does not use it (the KMD default). ``cpg``/``kpg`` are the
#: grouped kernel's compiled channel counts, ``block_h`` also its row tile;
#: ``tile_k_out`` is DirectNongroupedConvSpec.tile_k (output channels per
#: block), renamed so it cannot be the implicit-GEMM ``tile_k``. ``iglp`` -1
#: and ``waves_per_eu`` 0 stand for rocKE's ``None``.
_AOT_DIRECT_KNOB_DEFAULTS = {
    "block_h": 0,
    "unroll_rows": 0,
    "block_q": 0,
    "block_groups": 0,
    "double_buffer": 0,
    "cpg": 0,
    "kpg": 0,
    "tile_h": 0,
    "tile_w": 0,
    "tile_k_out": 0,
    "ck": 0,
    "waves_m": 0,
    "waves_n": 0,
    "atom": "none",
    "swizzle_wgm": 0,
    "iglp": -1,
    "waves_per_eu": 0,
}
#: Per variant, the direct identity's knob keys (``KernelIdentity.knobs``) and
#: the instance field each one is carried in.
AOT_DIRECT_KNOBS = {
    DIRECT_VARIANT_STD: {"block_w": "block_w", "block_waves": "block_waves"},
    DIRECT_VARIANT_SPATIAL: {"block_waves": "block_waves"},
    DIRECT_VARIANT_TILED: {
        "block_w": "block_w",
        "block_h": "block_h",
        "block_waves": "block_waves",
        "unroll_rows": "unroll_rows",
    },
    DIRECT_VARIANT_COL: {
        "block_w": "block_w",
        "block_h": "block_h",
        "block_waves": "block_waves",
    },
    DIRECT_VARIANT_GROUPED: {
        "block_q": "block_q",
        "block_groups": "block_groups",
        "double_buffer": "double_buffer",
    },
    DIRECT_VARIANT_NONGROUPED: {
        "tile_h": "tile_h",
        "tile_w": "tile_w",
        "tile_k": "tile_k_out",
        "ck": "ck",
        "waves_m": "waves_m",
        "waves_n": "waves_n",
        "atom": "atom",
        "swizzle_wgm": "swizzle_wgm",
        "iglp": "iglp",
        "waves_per_eu": "waves_per_eu",
    },
}
#: Fields a variant states beyond its knobs: the grouped kernel's compiled
#: channel counts and its row tile, which the direct sweep leaves at
#: DirectConvSpec's default (see :func:`aot_direct_grouped_block_h`).
_AOT_DIRECT_EXTRA_FIELDS = {DIRECT_VARIANT_GROUPED: ("cpg", "kpg", "block_h")}
#: Knobs rocKE spells as a bool, and as None: the descriptor holds an int.
_AOT_DIRECT_BOOL_KNOBS = ("unroll_rows", "double_buffer")
_AOT_DIRECT_NONE_KNOBS = {"iglp": -1, "waves_per_eu": 0}
#: Direct fields appended after the implicit-GEMM ones. An implicit-GEMM
#: instance leaves each at this value (the KMD default), so its descriptor and
#: spec stay as they were before the direct family existed; a direct instance
#: leaves the knobs its variant does not use at it.
_AOT_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    "filter_h": 0,
    "filter_w": 0,
    "stride_h": 0,
    "stride_w": 0,
    "pad_h": 0,
    "pad_w": 0,
    "dilation_h": 0,
    "dilation_w": 0,
    "block_w": 0,
    "block_waves": 0,
    **_AOT_DIRECT_KNOB_DEFAULTS,
}
#: Every axis of a direct grid but the non-grouped kernel's flat one is held to
#: this, as for std and spatial (``_MAX_GRID_DIM``). The non-grouped kernel
#: decodes a flat 1-D block id in i32, and a dispatch's x extent in work-items
#: is 32-bit.
_AOT_NONGROUPED_MAX_GRID_X = _INT32_MAX
_AOT_MAX_DISPATCH_ITEMS = (1 << 32) - 1


def aot_direct_grouped_block_h() -> int:
    """The row tile a grouped direct instance compiles: DirectConvSpec's default,
    since the direct sweep's grouped grid does not vary it."""
    return _direct.DirectConvSpec.__dataclass_fields__["block_h"].default


def aot_direct_fields_from_knobs(variant: str, knobs: dict) -> dict:
    """The instance fields a direct identity's ``knobs`` of ``variant`` set.

    Raises ``ValueError`` when the keys are not exactly the variant's.
    """
    mapping = AOT_DIRECT_KNOBS[variant]
    if set(knobs) != set(mapping):
        raise ValueError(f"knobs {sorted(knobs)} are not {sorted(mapping)}")
    values = {}
    for key, field_name in mapping.items():
        value = knobs[key]
        if field_name in _AOT_DIRECT_NONE_KNOBS and value is None:
            value = _AOT_DIRECT_NONE_KNOBS[field_name]
        elif field_name in _AOT_DIRECT_BOOL_KNOBS:
            if type(value) is not bool:
                raise ValueError(f"knob {key} must be a bool, got {value!r}")
            value = int(value)
        values[field_name] = value
    return values


def aot_direct_instance_fields(identity) -> dict:
    """The direct fields of the instance packaged from a forward direct identity.

    ``identity`` is read duck-typed (a ``KernelIdentity``, which this layer cannot
    import): its ``algorithm``, compiled filter, stride, padding, dilation and
    channel counts, and ``knobs`` JSON. Raises ``ValueError`` for an algorithm
    without a packaged launch or knobs that are not exactly its variant's. The
    caller states the placeholders (``_AOT_DIRECT_PLACEHOLDERS``) and checks the
    result with :func:`supports_gfx950_conv_fwd_aot`.
    """
    variant = AOT_DIRECT_ALGORITHMS.get(identity.algorithm)
    if variant is None:
        raise ValueError(
            f"direct algorithm {identity.algorithm!r} has no packaged launch "
            f"(supported: {sorted(AOT_DIRECT_ALGORITHMS)})"
        )
    fields_ = {
        "kernel_family": KERNEL_FAMILY_DIRECT_DEPTHWISE,
        "direct_variant": variant,
        "filter_h": identity.filter_h,
        "filter_w": identity.filter_w,
        "stride_h": identity.stride_h,
        "stride_w": identity.stride_w,
        "pad_h": identity.pad_h,
        "pad_w": identity.pad_w,
        "dilation_h": identity.dilation_h,
        "dilation_w": identity.dilation_w,
    }
    fields_.update(
        aot_direct_fields_from_knobs(
            variant, json.loads(identity.knobs) if identity.knobs else {}
        )
    )
    if variant == DIRECT_VARIANT_GROUPED:
        fields_.update(
            cpg=identity.cpg, kpg=identity.kpg, block_h=aot_direct_grouped_block_h()
        )
    return fields_


@dataclass(frozen=True)
class Gfx950ConvFwdAotSpec:
    """One shape-generic forward instance.

    Booleans are ints (0/1) so every field is a descriptor ``int`` or
    ``string``. ``async_chunk_a``/``async_chunk_b`` are the async loaders'
    chunk widths, a capability of an ``async_dma`` binary (0 otherwise).

    The trailing fields describe a direct instance (``kernel_family`` is
    ``KERNEL_FAMILY_DIRECT_DEPTHWISE``, which every direct variant shares): the
    compiled filter, stride, padding and dilation, and the variant's block
    knobs (``AOT_DIRECT_KNOBS``; ``block_w`` is 0 for spatial, which derives it
    from the runtime group count). Its implicit-GEMM fields hold
    ``_AOT_DIRECT_PLACEHOLDERS``, and the knobs of other variants their
    ``_AOT_FAMILY_DEFAULTS``. An implicit-GEMM instance leaves every trailing
    field at ``_AOT_FAMILY_DEFAULTS``.
    """

    dtype: str
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int
    wave_size: int
    pipeline: str
    epilogue: str
    vector_size_a: int
    vector_size_b: int
    vector_size_c: int
    unroll_k: int
    async_dma: int
    async_chunk_a: int
    async_chunk_b: int
    grouped: int
    instance_id: str
    layout: str = "NHWC"
    kernel_family: int = KERNEL_FAMILY_IMPLICIT_GEMM
    direct_variant: str = DIRECT_VARIANT_NONE
    filter_h: int = 0
    filter_w: int = 0
    stride_h: int = 0
    stride_w: int = 0
    pad_h: int = 0
    pad_w: int = 0
    dilation_h: int = 0
    dilation_w: int = 0
    block_w: int = 0
    block_waves: int = 0
    block_h: int = 0
    unroll_rows: int = 0
    block_q: int = 0
    block_groups: int = 0
    double_buffer: int = 0
    cpg: int = 0
    kpg: int = 0
    tile_h: int = 0
    tile_w: int = 0
    tile_k_out: int = 0
    ck: int = 0
    waves_m: int = 0
    waves_n: int = 0
    atom: str = "none"
    swizzle_wgm: int = 0
    iglp: int = -1
    waves_per_eu: int = 0

    @property
    def is_direct(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_DIRECT_DEPTHWISE

    def _direct_caps(self) -> tuple[int, int]:
        """``(cpg, kpg)`` the direct kernel compiles: 1 for the depthwise variants,
        the recorded counts for grouped, 0 (runtime) for non-grouped."""
        if self.direct_variant == DIRECT_VARIANT_NONGROUPED:
            return 0, 0
        if self.direct_variant == DIRECT_VARIANT_GROUPED:
            return self.cpg, self.kpg
        return 1, 1

    def probe_problem(self) -> _conv.ConvProblem | _direct.DirectConvProblem:
        """The problem rocKE's sweep builds this instance against.

        Implicit GEMM mirrors ``_probe_problem("fwd", caps)`` in
        ``benchmarks/common/kernel_sweep.py``; a direct instance mirrors
        ``probe_problem(caps, dtype, variant)`` in
        ``benchmarks/common/direct_kernel_sweep.py``. The emitted code does not
        depend on the probe's extents, only on its capability axes, so this is
        what makes the packaged binary the one the sweep benchmarked.
        """
        if self.is_direct:
            cpg, kpg = self._direct_caps()
            if cpg == 0:
                # Runtime channel counts: one group of 256.
                groups, cpg, kpg = 1, 256, 256
            elif self.direct_variant == DIRECT_VARIANT_SPATIAL:
                # The spatial variant takes fewer groups than a wave.
                groups = 8
            else:
                groups = 256 if cpg == 1 else 64
            return _direct.DirectConvProblem(
                N=1,
                H=64,
                W=64,
                groups=groups,
                cpg=cpg,
                kpg=kpg,
                KH=self.filter_h,
                KW=self.filter_w,
                PAD=self.pad_h,
                stride=self.stride_h,
                dtype=self.dtype,
                dil_h=self.dilation_h,
                dil_w=self.dilation_w,
            )
        groups = 2 if self.grouped else 1
        return _conv.ConvProblem(
            N=1,
            Hi=16,
            Wi=16,
            C=64 * groups,
            K=64 * groups,
            Y=3,
            X=3,
            sH=1,
            sW=1,
            dH=1,
            dW=1,
            pH=1,
            pW=1,
            groups=groups,
        )

    def to_instance_spec(self) -> _conv.ImplicitGemmConvSpec:
        """The rocKE builder spec, as ``kernel_sweep.build_kernel`` makes it."""
        if self.is_direct:
            raise ValueError("a direct instance has no implicit-GEMM spec")
        return _conv.ImplicitGemmConvSpec(
            problem=self.probe_problem(),
            name=self.kernel_symbol_prefix(),
            data=_conv.ConvDataSpec(
                dtype_a=self.dtype, dtype_b=self.dtype, dtype_d=self.dtype
            ),
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            wave_size=self.wave_size,
            vector_size_a=self.vector_size_a,
            vector_size_b=self.vector_size_b,
            vector_size_c=self.vector_size_c,
            unroll_k=bool(self.unroll_k),
            async_dma=bool(self.async_dma),
        )

    def direct_knobs(self) -> dict:
        """The direct identity's knobs (``KernelIdentity.knobs``) this instance
        carries, in rocKE's spelling: the inverse of
        :func:`aot_direct_fields_from_knobs`."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct knobs")
        knobs = {}
        for key, field_name in AOT_DIRECT_KNOBS[self.direct_variant].items():
            value = getattr(self, field_name)
            if field_name in _AOT_DIRECT_BOOL_KNOBS:
                value = bool(value)
            elif (
                field_name in _AOT_DIRECT_NONE_KNOBS
                and value == _AOT_DIRECT_NONE_KNOBS[field_name]
            ):
                value = None
            knobs[key] = value
        return knobs

    def rocke_direct_spec(self, problem: _direct.DirectConvProblem):
        """The rocKE kernel spec of this direct instance on ``problem``, as
        ``direct_kernel_sweep.make_spec`` makes it; only the name differs."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        name = self.kernel_symbol_prefix()
        knobs = self.direct_knobs()
        variant = self.direct_variant
        if variant == DIRECT_VARIANT_SPATIAL:
            return _direct.DirectDepthwiseSpatialSpec(
                problem=problem, name=name, wave_size=self.wave_size, **knobs
            )
        if variant == DIRECT_VARIANT_STD:
            return _direct.DirectDepthwiseSpec(
                problem=problem, name=name, wave_size=self.wave_size, **knobs
            )
        if variant == DIRECT_VARIANT_TILED:
            return _direct.DirectDepthwiseTiledSpec(problem=problem, name=name, **knobs)
        if variant == DIRECT_VARIANT_COL:
            return _direct.DirectDepthwiseColSpec(
                problem=problem, name=name, dtype=problem.dtype, **knobs
            )
        if variant == DIRECT_VARIANT_GROUPED:
            return _direct.DirectConvSpec(
                problem=problem, name=name, block_h=self.block_h, **knobs
            )
        if variant == DIRECT_VARIANT_NONGROUPED:
            return _nongrouped.DirectNongroupedConvSpec(
                problem=problem, name=name, **knobs
            )
        raise ValueError(f"unknown direct variant {variant!r}")

    def to_direct_spec(self):
        """The rocKE builder spec, as ``direct_kernel_sweep.make_spec`` makes it
        for the sweep's probe problem."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        return self.rocke_direct_spec(self.probe_problem())

    def kernel_symbol_prefix(self) -> str:
        # rocKE appends the probe problem and tuning to this; the instance id
        # alone already makes the symbol unique.
        if self.is_direct:
            return f"hkp_conv_fwd_dw_aot_gfx950_{self.instance_id[:24]}"
        return f"hkp_conv_fwd_aot_gfx950_{self.instance_id[:24]}"

    def kernel_name(self) -> str:
        if self.is_direct:
            return self.to_direct_spec().kernel_name()
        return self.to_instance_spec().kernel_name()

    def runtime_spec(self, problem: Gfx950ConvFwdSpec) -> Gfx950ConvFwdSpec:
        """The exact-shape direct spec of a std or spatial instance on
        ``problem``'s geometry.

        The exact-shape direct guard, grid and launch values read only the
        problem plus ``direct_variant``/``block_w``/``block_waves``, so a std or
        spatial instance reuses them unchanged through this. The other direct
        variants have no exact-shape spec; see :meth:`runtime_direct_spec`.
        """
        if not self.is_direct or self.direct_variant not in _DIRECT_VARIANTS:
            raise ValueError("only a std or spatial direct instance has a runtime spec")
        return replace(
            problem,
            **_DIRECT_PLACEHOLDERS,
            kernel_family=KERNEL_FAMILY_DIRECT_DEPTHWISE,
            direct_variant=self.direct_variant,
            block_w=self.block_w,
            block_waves=self.block_waves,
        )

    def runtime_direct_spec(self, problem: Gfx950ConvFwdSpec):
        """rocKE's kernel spec of this direct instance on ``problem``: what
        ``direct_kernel_sweep.plan_for`` re-validates and launches."""
        return self.rocke_direct_spec(aot_direct_problem(problem))

    def block(self) -> tuple[int, int, int]:
        if self.is_direct:
            if self.direct_variant in _DIRECT_VARIANTS:
                return self.block_waves * self.wave_size, 1, 1
            return self.to_direct_spec().threads_per_block, 1, 1
        return self.warp_m * self.warp_n * self.wave_size, 1, 1

    def grid(self, problem: Gfx950ConvFwdSpec) -> tuple[int, int, int]:
        """``(K/groups tiles, N*Ho*Wo tiles, groups)`` for ``problem``; a std or
        spatial instance takes :meth:`Gfx950ConvFwdSpec.grid`'s direct grid, any
        other direct instance rocKE's own launch geometry."""
        if self.is_direct:
            if self.direct_variant in _DIRECT_VARIANTS:
                return self.runtime_spec(problem).grid()
            return _aot_direct_grid(self.runtime_direct_spec(problem))
        args = self._conv_args(problem)
        return args.num_pid_n, args.num_pid_m, problem.groups

    def _conv_args(self, problem: Gfx950ConvFwdSpec) -> ConvArgs:
        if self.is_direct:
            return ConvArgs.from_problem(aot_direct_problem(problem))
        return ConvArgs.from_problem(
            problem.to_problem(), tile_m=self.tile_m, tile_n=self.tile_n
        )

    def launch_signature(self) -> list[dict]:
        if self.is_direct:
            return conv_direct_args_signature(self.dtype)
        return conv_args_signature(self.dtype)

    def launch_values(
        self,
        problem: Gfx950ConvFwdSpec,
        a_ptr: int,
        b_ptr: int,
        d_ptr: int,
        a_bytes: int,
        b_bytes: int,
        d_bytes: int,
    ) -> dict[str, int]:
        """Every kernel argument for ``problem``, from rocKE's ``ConvArgs``."""
        return self._conv_args(problem).to_launch_values(
            a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes
        )


def aot_direct_problem(problem: Gfx950ConvFwdSpec) -> _direct.DirectConvProblem:
    """``problem`` as rocKE's direct kernels see it, dilation included. They take
    one stride and one padding, so the caller has checked sH == sW and pH == pW."""
    return _direct.DirectConvProblem(
        N=problem.N,
        H=problem.Hi,
        W=problem.Wi,
        groups=problem.groups,
        cpg=problem.C // problem.groups,
        kpg=problem.K // problem.groups,
        KH=problem.Y,
        KW=problem.X,
        PAD=problem.pH,
        stride=problem.sH,
        dtype=problem.dtype,
        dil_h=problem.dH,
        dil_w=problem.dW,
    )


def _aot_direct_grid(rocke_spec) -> tuple[int, int, int]:
    """rocKE's launch grid for a direct kernel spec (``plan_for``'s ``_conv_step``)."""
    if isinstance(rocke_spec, _nongrouped.DirectNongroupedConvSpec):
        return rocke_spec.grid()
    return _direct.direct_launch_geometry(rocke_spec)[0]


def _aot_direct_validate(variant: str, rocke_spec, arch: str) -> tuple[bool, str]:
    """rocKE's validator for ``rocke_spec``, as ``direct_kernel_sweep.validate_spec``
    runs it."""
    try:
        if variant == DIRECT_VARIANT_NONGROUPED:
            return _nongrouped.is_valid_nongrouped_spec(rocke_spec, arch=arch)
        if variant == DIRECT_VARIANT_GROUPED:
            rocke_spec.validate()
            return _direct.is_valid_spec(rocke_spec, arch=arch)
        if variant == DIRECT_VARIANT_STD:
            return _direct.is_valid_depthwise_spec(rocke_spec, arch=arch)
        if variant == DIRECT_VARIANT_SPATIAL:
            return _direct.is_valid_depthwise_spatial_spec(rocke_spec, arch=arch)
        if variant == DIRECT_VARIANT_TILED:
            return _direct.is_valid_depthwise_tiled_spec(rocke_spec, arch=arch)
        if variant == DIRECT_VARIANT_COL:
            rocke_spec.validate()
            return _direct.is_valid_depthwise_col_spec(rocke_spec, arch=arch)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return False, f"unknown direct variant {variant!r}"


def _aot_async_chunks(spec: Gfx950ConvFwdAotSpec) -> tuple[int, int]:
    """The chunk widths the builder picks for ``spec``'s async loaders."""
    a_loader, b_loader = _conv.async_tile_loaders(spec.to_instance_spec())
    return a_loader.elems_per_chunk, b_loader.elems_per_chunk


def _aot_direct_knob_error(spec: Gfx950ConvFwdAotSpec) -> str:
    """The block knobs of a direct instance other than std or spatial: each used
    one has its type and a sane range (rocKE's validators bound it further), every
    other one holds its ``_AOT_FAMILY_DEFAULTS`` value; "" when accepted."""
    variant = spec.direct_variant
    used = set(AOT_DIRECT_KNOBS[variant].values()) | set(
        _AOT_DIRECT_EXTRA_FIELDS.get(variant, ())
    )
    for name in ("block_w", "block_waves", *_AOT_DIRECT_KNOB_DEFAULTS):
        value = getattr(spec, name)
        default = _AOT_FAMILY_DEFAULTS[name]
        if name not in used:
            if value != default or type(value) is not type(default):
                return (
                    f"the {variant} direct variant does not use {name}; it must be "
                    f"{default!r}, got {value!r}"
                )
            continue
        if name == "atom":
            if value not in _nongrouped._ATOMS:
                return (
                    f"atom must be one of {sorted(_nongrouped._ATOMS)}, got {value!r}"
                )
        elif type(value) is not int:
            return f"{name} must be an integer, got {value!r}"
        elif name in _AOT_DIRECT_BOOL_KNOBS:
            if value not in (0, 1):
                return f"{name} must be 0 or 1"
        elif name == "iglp":
            if not -1 <= value <= _INT32_MAX:
                return "iglp must be -1 (unset) or a nonnegative level"
        elif name == "waves_per_eu":
            if not 0 <= value <= _INT32_MAX:
                return "waves_per_eu must be 0 (unset) or positive"
        elif name in ("block_waves", "block_groups"):
            # One wave per channel block or group: at most 1024 threads.
            if not 1 <= value <= _DIRECT_MAX_BLOCK_WAVES:
                return f"{name} must be an integer in [1, {_DIRECT_MAX_BLOCK_WAVES}]"
        elif not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    if (
        variant == DIRECT_VARIANT_GROUPED
        and spec.block_h != aot_direct_grouped_block_h()
    ):
        return (
            f"a grouped direct instance compiles the direct sweep's row tile "
            f"block_h={aot_direct_grouped_block_h()}, got {spec.block_h}"
        )
    return ""


def _aot_direct_error(spec: Gfx950ConvFwdAotSpec) -> str:
    """The direct instance contract, before rocKE's validators; "" when accepted.

    rocKE's direct kernels compile one stride and one padding (they read
    neither ``sW`` nor ``pW``), and the direct sweep builds square filters
    only; the per-axis fields state that without a convention. Only the
    non-grouped kernel computes a dilated convolution.
    """
    for name, value in _AOT_DIRECT_PLACEHOLDERS.items():
        actual = getattr(spec, name)
        if actual != value or type(actual) is not type(value):
            return f"a direct depthwise instance must set {name} to {value!r}, got {actual!r}"
    if spec.wave_size != _DIRECT_WAVE_SIZE:
        return f"wave_size must be {_DIRECT_WAVE_SIZE} on {_ARCH}"
    if spec.direct_variant not in _AOT_DIRECT_VARIANTS:
        return (
            f"direct_variant must be one of {_AOT_DIRECT_VARIANTS} for a direct "
            f"instance, got {spec.direct_variant!r}"
        )
    for name in (
        "filter_h",
        "filter_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    ):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    for name in ("pad_h", "pad_w"):
        value = getattr(spec, name)
        if type(value) is not int or not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    if spec.direct_variant != DIRECT_VARIANT_NONGROUPED and (
        spec.dilation_h != 1 or spec.dilation_w != 1
    ):
        return "the direct depthwise kernels have no dilation (dilation_h == dilation_w == 1)"
    if spec.stride_h != spec.stride_w or spec.pad_h != spec.pad_w:
        return "the direct depthwise kernels compile one stride and one padding"
    if spec.filter_h != spec.filter_w:
        return "the direct sweep builds square filters only"
    if spec.direct_variant not in _DIRECT_VARIANTS:
        return _aot_direct_knob_error(spec)
    for name, default in _AOT_DIRECT_KNOB_DEFAULTS.items():
        value = getattr(spec, name)
        if value != default or type(value) is not type(default):
            return (
                f"the {spec.direct_variant} direct variant does not use {name}; it "
                f"must be {default!r}, got {value!r}"
            )
    if (
        type(spec.block_waves) is not int
        or not 1 <= spec.block_waves <= _DIRECT_MAX_BLOCK_WAVES
    ):
        return f"block_waves must be an integer in [1, {_DIRECT_MAX_BLOCK_WAVES}]"
    if spec.direct_variant == DIRECT_VARIANT_SPATIAL:
        if type(spec.block_w) is not int or spec.block_w != 0:
            return "the spatial direct variant derives block_w; it must be 0"
    elif type(spec.block_w) is not int or not 0 < spec.block_w <= _INT32_MAX:
        return "block_w must be a positive signed 32-bit integer"
    return ""


def supports_gfx950_conv_fwd_aot(
    spec: Gfx950ConvFwdAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance; rocKE's validators decide last."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx950ConvFwdAotSpec):
        return False, "expected Gfx950ConvFwdAotSpec"
    if spec.dtype not in ("fp16", "bf16"):
        return False, "the packaged forward contract supports fp16 or bf16 only"
    if spec.layout != "NHWC":
        return False, "the packaged forward contract requires dense NHWC storage"
    if (
        type(spec.instance_id) is not str
        or len(spec.instance_id) != _AOT_INSTANCE_ID_LENGTH
        or any(c not in "0123456789abcdef" for c in spec.instance_id)
    ):
        return False, "instance_id must be a KernelIdentity.stable_hash() (40 hex)"
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
        error = _aot_direct_error(spec)
        if error:
            return False, error
        try:
            return _aot_direct_validate(
                spec.direct_variant, spec.to_direct_spec(), arch
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return False, str(exc)
    for name, default in _AOT_FAMILY_DEFAULTS.items():
        value = getattr(spec, name)
        if value != default or type(value) is not type(default):
            return False, (
                f"an implicit-GEMM instance must leave {name} at {default!r}, "
                f"got {value!r}"
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
    ):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return False, f"{name} must be a positive signed 32-bit integer"
    if spec.wave_size != _DIRECT_WAVE_SIZE:
        return False, f"wave_size must be {_DIRECT_WAVE_SIZE} on {_ARCH}"
    if spec.pipeline not in _AOT_PIPELINES:
        return False, (
            f"the packaged forward contract supports pipelines {_AOT_PIPELINES}, "
            f"got {spec.pipeline!r}"
        )
    if spec.epilogue not in ("default", "cshuffle"):
        return False, "epilogue must be 'default' or 'cshuffle'"
    for name in ("vector_size_a", "vector_size_b", "vector_size_c"):
        if getattr(spec, name) not in _AOT_VECTOR_WIDTHS:
            return False, f"{name} must be one of {_AOT_VECTOR_WIDTHS}"
    for name in ("unroll_k", "async_dma", "grouped"):
        value = getattr(spec, name)
        if type(value) is not int or value not in (0, 1):
            return False, f"{name} must be 0 or 1"
    if spec.unroll_k and spec.async_dma:
        return False, "unroll_k and async_dma are separate K-loop drivers"
    try:
        instance = spec.to_instance_spec()
        instance.validate()
        ok, reason = _conv.is_valid_spec_for_problem(
            instance, instance.problem, arch=arch
        )
        if not ok:
            return False, reason
        if spec.async_dma:
            chunks = _aot_async_chunks(spec)
            if chunks != (spec.async_chunk_a, spec.async_chunk_b):
                return False, (
                    f"async chunk widths {chunks} differ from the recorded "
                    f"({spec.async_chunk_a}, {spec.async_chunk_b})"
                )
        elif spec.async_chunk_a or spec.async_chunk_b:
            return False, "async_chunk_a/async_chunk_b are 0 without async_dma"
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, "ok"


def gfx950_conv_fwd_aot_capabilities(
    spec: Gfx950ConvFwdAotSpec,
) -> Gfx950ConvFwdCapabilities:
    """What instance ``spec`` bakes in, as :func:`gfx950_conv_fwd_capabilities` reports it.

    The widths are the identity's own, not derived from a build problem. A and B
    both run along C/groups, so both divide it exactly when their lcm does. The
    probe problem is 3x3, so an instance is never the pointwise GEMM. A direct
    instance reports its compiled filter, stride and padding.
    """
    if spec.is_direct:
        return Gfx950ConvFwdCapabilities(
            filter_hw=(spec.filter_h, spec.filter_w),
            stride_hw=(spec.stride_h, spec.stride_w),
            pad_hw=(spec.pad_h, spec.pad_w),
        )
    return Gfx950ConvFwdCapabilities(
        load_width=math.lcm(spec.vector_size_a, spec.vector_size_b),
        store_width=spec.vector_size_c,
        pointwise=False,
        grouped=bool(spec.grouped),
    )


def _aot_direct_serves(
    spec: Gfx950ConvFwdAotSpec, problem: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """Whether a direct instance other than std or spatial computes ``problem``,
    which already passed the problem checks.

    rocKE's own offer, ``KernelCache.supports_problem`` then ``plan_for``: the
    compiled filter, stride, padding, dilation and channel counts are the
    problem's, and the kernel's validator accepts the spec rebuilt on the real
    problem (for non-grouped, also only the tile widths a sweep of its Wo would
    try). The kernels compile one stride and one padding, so both axes must
    agree. Then the launch has to fit: every grid axis 65535, the non-grouped
    kernel's flat grid a 32-bit dispatch.
    """
    variant = spec.direct_variant
    if problem.sH != problem.sW or problem.pH != problem.pW:
        return False, "the direct kernels compile one stride and one padding"
    for label, baked, actual in (
        ("filter", (spec.filter_h, spec.filter_w), (problem.Y, problem.X)),
        ("stride", (spec.stride_h, spec.stride_w), (problem.sH, problem.sW)),
        ("padding", (spec.pad_h, spec.pad_w), (problem.pH, problem.pW)),
        ("dilation", (spec.dilation_h, spec.dilation_w), (problem.dH, problem.dW)),
    ):
        if baked != actual:
            return (
                False,
                f"direct kernel compiled for {label} {baked}, problem has {actual}",
            )
    cpg, kpg = problem.C // problem.groups, problem.K // problem.groups
    if variant == DIRECT_VARIANT_NONGROUPED:
        if problem.groups != 1:
            return False, "the non-grouped direct kernel serves groups == 1 only"
    elif (cpg, kpg) != spec._direct_caps():
        return False, (
            f"direct kernel compiled for cpg, kpg = {spec._direct_caps()}, "
            f"problem has {(cpg, kpg)}"
        )
    rocke_spec = spec.runtime_direct_spec(problem)
    ok, why = _aot_direct_validate(variant, rocke_spec, _ARCH)
    if not ok:
        return False, why
    grid = _aot_direct_grid(rocke_spec)
    if variant == DIRECT_VARIANT_NONGROUPED:
        widths = _nongrouped.tile_w_candidates(
            rocke_spec.problem.Wo, rocke_spec.atom_tile
        )
        if spec.tile_w not in widths:
            return False, (
                f"tile_w={spec.tile_w} does not fit Wo={rocke_spec.problem.Wo} "
                f"(widths tried for it: {widths})"
            )
        if (
            grid[0] > _AOT_NONGROUPED_MAX_GRID_X
            or grid[0] * rocke_spec.threads_per_block > _AOT_MAX_DISPATCH_ITEMS
        ):
            return False, f"flat direct grid {grid[0]} does not fit one dispatch"
        return True, "ok"
    for axis, extent in zip("xyz", grid):
        if extent > _MAX_GRID_DIM:
            return False, f"direct grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return True, "ok"


def gfx950_conv_fwd_aot_serves(
    spec: Gfx950ConvFwdAotSpec, problem: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """Whether instance ``spec`` computes ``problem`` (only its problem fields).

    Implicit GEMM: the forward subset of ``KernelCache.supports_problem`` plus
    the launch bounds ``ConvArgs`` enforces: the vector widths and async chunks
    divide the contiguous per-group runs (A and B along C/groups, D along
    K/groups), a grouped problem needs a grouped binary, every grid axis fits
    65535, and the reduction extent fits the kernel's 24-bit products. A grouped
    binary was not built as pointwise, so grouped pointwise problems are served.

    Direct std and spatial: the exact-shape engine's direct rule
    (:func:`_direct_serves`) on the instance's compiled filter, stride and
    padding. The other direct variants: rocKE's own offer
    (:func:`_aot_direct_serves`). Batch, image and group count are kernel
    arguments, and for non-grouped the channel counts too.
    """
    error = _problem_error(problem, decline_grouped_pointwise=False)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    caps = gfx950_conv_fwd_aot_capabilities(spec)
    if spec.is_direct:
        if spec.direct_variant in _DIRECT_VARIANTS:
            return _direct_serves(caps, spec.runtime_spec(problem))
        return _aot_direct_serves(spec, problem)
    cpg = problem.C // problem.groups
    if spec.async_dma:
        for chunk in (spec.async_chunk_a, spec.async_chunk_b):
            if cpg % chunk:
                return False, f"async chunk {chunk} does not divide cpg={cpg}"
    return _implicit_gemm_serves(caps, problem, spec.tile_m, spec.tile_n)


def gfx950_conv_fwd_aot_runtime_candidates(
    problem: Gfx950ConvFwdSpec, specs: Sequence[Gfx950ConvFwdAotSpec]
) -> list[Gfx950ConvFwdAotSpec]:
    """The instances of ``specs`` hipkernel:Gfx950ConvFwdAot selects for ``problem``.

    "Serves" (:func:`gfx950_conv_fwd_aot_serves`) is what a binary can compute;
    this is what rocKE's own runtime selection then runs, so it depends on the whole
    catalog ``specs``, not on one instance. It mirrors ``--run-from-cache`` in
    ``benchmarks/common/benchmark_implicit_gemm_conv.py`` with its default policy:
    ``KernelCache.compatible`` then ``_widest_vector_sizes``, which keeps, per kernel
    kind, only the widest vector widths present (greatest ``(a + b + c, a, b, c)``),
    and drops the scalar-store default epilogue when any serving instance stores
    wider. Every packaged instance is forward with the default split-K, two-stage and
    group merge, so the kind is the epilogue. rocKE's direct benchmark runs every
    direct kernel that serves, so direct instances are all kept. The native pack's
    ``conv::widestVectorSelection`` is the same policy; the result keeps the order of
    ``specs``.
    """
    serving = [spec for spec in specs if gfx950_conv_fwd_aot_serves(spec, problem)[0]]
    implicit = [spec for spec in serving if not spec.is_direct]
    widest_store = max((spec.vector_size_c for spec in implicit), default=0)
    implicit = [
        spec
        for spec in implicit
        if not (spec.epilogue == "default" and widest_store > 1)
    ]

    def rank(spec: Gfx950ConvFwdAotSpec) -> tuple[int, int, int, int]:
        a, b, c = spec.vector_size_a, spec.vector_size_b, spec.vector_size_c
        return (a + b + c, a, b, c)

    best: dict[str, tuple[int, int, int, int]] = {}
    for spec in implicit:
        best[spec.epilogue] = max(best.get(spec.epilogue, rank(spec)), rank(spec))
    selected = {id(spec) for spec in implicit if rank(spec) == best[spec.epilogue]}
    return [spec for spec in serving if spec.is_direct or id(spec) in selected]


_AOT_DIRECT_BUILDERS = {
    DIRECT_VARIANT_STD: _direct.build_direct_depthwise,
    DIRECT_VARIANT_SPATIAL: _direct.build_direct_depthwise_spatial,
    DIRECT_VARIANT_TILED: _direct.build_direct_depthwise_tiled,
    DIRECT_VARIANT_COL: _direct.build_direct_depthwise_col,
    DIRECT_VARIANT_GROUPED: _direct.build_direct_conv,
    DIRECT_VARIANT_NONGROUPED: _nongrouped.build_direct_conv_nongrouped,
}


def build_gfx950_conv_fwd_aot(
    spec: Gfx950ConvFwdAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Dispatches on ``kernel_family`` and the direct variant, as
    ``direct_kernel_sweep._build_kernel`` does.
    """
    ok, reason = supports_gfx950_conv_fwd_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT forward instance: {reason}")
    if spec.is_direct:
        return _AOT_DIRECT_BUILDERS[spec.direct_variant](
            spec.to_direct_spec(), arch=arch
        )
    return _conv.build_implicit_gemm_conv(spec.to_instance_spec(), arch=arch)


def gfx950_conv_fwd_aot_problem_for_request(
    request: ConvGroupedRequest,
) -> Gfx950ConvFwdSpec:
    """The problem hipkernel:Gfx950ConvFwdAot would match for ``request``.

    Instances are chosen per problem by capability, so there is no single spec
    to resolve; this is the graph side of that match (the profile's
    ``dispatch``). Raises ``ValueError`` outside the graph contract, which
    unlike the exact-shape one includes grouped pointwise.
    """
    return _base_spec_for_request(request, decline_grouped_pointwise=False)


def supports_gfx950_conv_fwd_aot_problem(
    problem: Gfx950ConvFwdSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """The graph contract of hipkernel:Gfx950ConvFwdAot (the profile's predicate)."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(problem, Gfx950ConvFwdSpec):
        return False, "expected Gfx950ConvFwdSpec"
    error = _problem_error(problem, decline_grouped_pointwise=False)
    return (False, error) if error else (True, "ok")
