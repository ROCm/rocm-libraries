# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 implicit-GEMM forward convolution.

The descriptor spec is flat, while the original builder takes nested problem
and data specs plus optional fusion callbacks. This adapter resolves the
exact-shape engine's fixed defaults (:data:`EXACT_SHAPE_DEFAULTS`) before
packaging and calls the original builder without callbacks.
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

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    ConvInstanceSpec,
    launch_plan_for,
)

# The dispatcher's admission rule for one AOT identity (``_admits``) has no
# public per-identity entry point; it is imported so that "serves" here is the
# dispatcher's own verdict rather than a copy of it.
from dispatch.grouped_convolution import _admits as _dispatcher_admits
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_direct_nongrouped as _nongrouped
from kernels.common import conv_implicit_gemm as _conv
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
from kernels.common.conv_args import MUL24_REDUCTION_LIMIT, ConvArgs
from kernels.conv.aot_jobs import _probe_problem as _aot_probe_problem
from kernels.conv.aot_jobs import job_for_identity
from kernels.conv.direct_aot_jobs import DirectCaps, direct_job_for_identity
from kernels.conv.direct_aot_jobs import probe_problem as _direct_probe_problem
from kernels.conv.direct_variants import (
    DIRECT_GROUPED_HALO,
    DIRECT_SPLIT_REDUCE,
    make_direct_spec,
    select_direct_kernel_builder,
    validate_direct_spec,
)
from kernels.conv.kernel_identity import KernelIdentity
from rocke.core.arch import ArchTarget
from rocke.core.ir import KernelDef
from rocke.core.lower_llvm import LLVM_FLAVORS

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
#: The implicit-GEMM tuning every exact-shape request resolves to (see
#: :func:`gfx950_conv_fwd_spec_for_request`). It is the gfx950 forward
#: candidate rocKE's grouped-convolution dispatcher resolved when the
#: exact-shape catalog was first packaged. rocKE's dispatcher now selects among
#: registered AOT instances (``hipkernel:Gfx950ConvFwdAot`` packages those), so
#: the exact-shape engine keeps this configuration itself: its shipped kernels
#: and symbols stay what they were. The epilogue is not fixed; see
#: :func:`exact_shape_epilogue`.
EXACT_SHAPE_DEFAULTS = {
    "tile_m": 64,
    "tile_n": 64,
    "tile_k": 64,
    "warp_m": 2,
    "warp_n": 2,
    "warp_tile_m": 32,
    "warp_tile_n": 32,
    "warp_tile_k": 16,
    "pipeline": "mem",
}


def exact_shape_epilogue(C: int, K: int, dtype: str) -> str:
    """The exact-shape engine's epilogue for a request: ``cshuffle`` where the
    store width rocKE's builder picks for ``(C, K)`` is wider than one element,
    else the scalar ``default`` one. The full channel counts are used, as when
    the catalog was first packaged."""
    _, _, store = _conv.ImplicitGemmConvSpec.default_vector_sizes(C, K, dtype)
    return "cshuffle" if store > 1 else "default"


@dataclass(frozen=True)
class Gfx950ConvFwdSpec:
    """Serializable problem and resolved tuning values for one packaged kernel.

    Use :func:`gfx950_conv_fwd_spec_for_request` to obtain the exact-shape
    engine's defaults (:data:`EXACT_SHAPE_DEFAULTS`), or :func:`gfx950_conv_fwd_direct_spec_for_request` for a
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
    """Resolve the exact-shape engine's defaults and optionally override tile_k.

    The tuning is :data:`EXACT_SHAPE_DEFAULTS` with the arch's wave size and
    the epilogue of :func:`exact_shape_epilogue`; rocKE's builder validators
    then decide (:func:`supports_gfx950_conv_fwd`). The request interface also
    describes 3D/backward convolution. Those requests, and grouped pointwise,
    are rejected here as integration gaps, even when a rocKE dispatcher
    instance can serve them. Grouped 2D forward (``G > 1``, including
    depthwise) is accepted.
    """
    base = _base_spec_for_request(request)
    spec = replace(
        base,
        **EXACT_SHAPE_DEFAULTS,
        wave_size=ArchTarget.from_gfx(request.arch).wave_size,
        epilogue=exact_shape_epilogue(base.C, base.K, base.dtype),
    )
    if tile_k is not None:
        spec = replace(spec, tile_k=tile_k)
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
    """A direct depthwise spec for ``request``; the engine picks the arm itself.

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
# forward rocKE ``KernelIdentity`` (``kernels/conv/kernel_identity.py``):
# tuning and capability fields, no problem extents. Two families are packaged:
# the implicit-GEMM identities of the AOT sweep (``kernels/conv/aot_jobs.py``)
# and the forward identities of the direct sweep
# (``kernels/conv/direct_aot_jobs.py``), one direct variant per direct
# algorithm (``AOT_DIRECT_ALGORITHMS``). ``instance_id`` is the identity's
# ``stable_hash()``.
#
# Everything rocKE decides is delegated to rocKE: an instance is built from the
# job the sweep builds its identity from (``job_for_identity``,
# ``direct_job_for_identity``), it serves what the grouped-convolution
# dispatcher admits for that identity, and it launches with the dispatcher's
# launch plan. This module adds only the packaged contract: which identities
# the native launch models (:func:`gfx950_conv_fwd_aot_spec_for_identity`
# refuses the others with a reason) and the launch bounds the native pack
# enforces on top of rocKE's.

#: Vector widths rocKE's AOT sweep builds (``CACHE_VECS``).
_AOT_VECTOR_WIDTHS = (1, 2, 4, 8)
#: Pipelines with the packaged launch (see ``_PACKAGED_PIPELINES``).
_AOT_PIPELINES = _PACKAGED_PIPELINES
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_AOT_INSTANCE_ID_LENGTH = 40
#: The flavor an instance's identity is spelled with when no LLVM flavor
#: reproduces its ``instance_id`` (the flavor rocKE's dispatcher lists use).
_AOT_DEFAULT_LLVM_FLAVOR = "llvm22"
#: The implicit-GEMM fields of a direct instance hold exactly what the direct
#: sweep writes into a direct identity (``direct_aot_jobs._identity``).
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
#: Each forward algorithm of ``direct_aot_jobs.DIRECT_CAPABILITIES`` the packaged
#: launch runs, and the ``direct_variant`` an instance of it carries.
AOT_DIRECT_ALGORITHMS = {
    "direct_depthwise": DIRECT_VARIANT_STD,
    "direct_depthwise_spatial": DIRECT_VARIANT_SPATIAL,
    "direct_depthwise_tiled": DIRECT_VARIANT_TILED,
    "direct_depthwise_col": DIRECT_VARIANT_COL,
    "direct_grouped": DIRECT_VARIANT_GROUPED,
    "direct_nongrouped": DIRECT_VARIANT_NONGROUPED,
}
_AOT_DIRECT_VARIANTS = tuple(AOT_DIRECT_ALGORITHMS.values())
_AOT_ALGORITHM_OF_VARIANT = {v: a for a, v in AOT_DIRECT_ALGORITHMS.items()}
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

#: Implicit-GEMM identity fields an instance carries (the identity's own value).
#: Every other field of a forward implicit-GEMM identity must hold its
#: ``KernelIdentity`` default: it selects code the packaged launch does not model.
AOT_IMPLICIT_GEMM_CARRIED = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "wave_size",
    "pipeline",
    "epilogue",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "grouped",
)
#: Identity fields every instance states but no instance field carries.
_AOT_IDENTITY_KEYS = (
    "arch",
    "direction",
    "algorithm",
    "dtype_a",
    "dtype_b",
    "dtype_d",
    "llvm_flavor",
)
#: ``KernelIdentity`` fields with a plain default and that default.
_IDENTITY_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(KernelIdentity)
    if f.default is not dataclasses.MISSING
}

# Identity features of rocKE's forward sweeps that the packaged launch does not
# model. gfx950_conv_fwd_aot_spec_for_identity refuses each with its reason.
#: Implicit GEMM, ``group_merge`` > 1 (``aot_jobs.CACHE_FWD_GROUP_MERGES``).
AOT_REFUSED_GROUP_MERGE = (
    "group_merge={gm} is not packaged: a group-merged depthwise implicit-GEMM "
    "kernel launches groups/{gm} workgroups along z over a merged GEMM-N of "
    "kpg*{gm}, and records the vector widths its builder derives as 0; the "
    "packaged launch (aotInstanceLaunch) runs one group per z and checks baked "
    "widths against C/groups and K/groups"
)
#: Direct ``direct_grouped_halo`` (grouped forward through the LDS-halo kernel).
AOT_REFUSED_GROUPED_HALO = (
    "direct algorithm 'direct_grouped_halo' is not packaged: it runs the "
    "non-grouped LDS-halo kernel over groups/group_merge merged groups (knobs "
    "grouped, group_merge), a grid and channel decode the packaged non-grouped "
    "launch (nongroupedDirectLaunch, groups == 1 only) does not compute"
)
#: Direct ``direct_nongrouped`` with a ``split_c`` knob (split-C).
AOT_REFUSED_SPLIT_C = (
    "split_c={split_c} is not packaged: a split-C non-grouped kernel writes fp32 "
    "partial sums into a [split_c, N, Ho, Wo, K] workspace bound to D over a grid "
    "split_c times wider, and a second kernel (direct_split_reduce) folds them "
    "into the output; the packaged launch runs one kernel and no workspace"
)
#: The split-C fold itself (``direction`` ``direct_fwd_helper``).
AOT_REFUSED_SPLIT_REDUCE = (
    "'direct_split_reduce' is not packaged: it is the second pass of a split-C "
    "kernel, not a convolution, and split-C is not packaged"
)


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


def aot_direct_instance_fields(identity: KernelIdentity) -> dict:
    """The direct fields of the instance packaged from a forward direct identity.

    Reads the identity's ``algorithm``, compiled filter, stride, padding,
    dilation and channel counts, and ``knobs`` JSON. Raises ``ValueError`` for
    an algorithm or knob without a packaged launch (grouped-halo, split-C; see
    ``AOT_REFUSED_*``) or knobs that are not exactly its variant's. The caller
    states the placeholders (``_AOT_DIRECT_PLACEHOLDERS``) and checks the
    result with :func:`supports_gfx950_conv_fwd_aot`.
    """
    if identity.algorithm == DIRECT_GROUPED_HALO:
        raise ValueError(AOT_REFUSED_GROUPED_HALO)
    variant = AOT_DIRECT_ALGORITHMS.get(identity.algorithm)
    if variant is None:
        raise ValueError(
            f"direct algorithm {identity.algorithm!r} has no packaged launch "
            f"(supported: {sorted(AOT_DIRECT_ALGORITHMS)})"
        )
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if variant == DIRECT_VARIANT_NONGROUPED and knobs.get("split_c", 1) != 1:
        raise ValueError(AOT_REFUSED_SPLIT_C.format(split_c=knobs["split_c"]))
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
    fields_.update(aot_direct_fields_from_knobs(variant, knobs))
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

    :meth:`identity` is the rocKE ``KernelIdentity`` the fields spell; a
    packageable instance's ``instance_id`` is its ``stable_hash()``.
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

    @property
    def algorithm(self) -> str:
        """rocKE's ``KernelIdentity.algorithm`` of this instance."""
        if not self.is_direct:
            return "implicit_gemm"
        return _AOT_ALGORITHM_OF_VARIANT[self.direct_variant]

    def identity(self) -> KernelIdentity:
        """The rocKE identity these fields spell.

        Its ``llvm_flavor`` is the one whose ``stable_hash()`` is ``instance_id``;
        when none is, the dispatcher lists' flavor. The fields must already be
        consistent (:func:`supports_gfx950_conv_fwd_aot` checks them first).
        """
        return _aot_identity(self)

    def _direct_caps(self) -> tuple[int, int]:
        """``(cpg, kpg)`` the direct kernel compiles: 1 for the depthwise variants,
        the recorded counts for grouped, 0 (runtime) for non-grouped."""
        if self.direct_variant == DIRECT_VARIANT_NONGROUPED:
            return 0, 0
        if self.direct_variant == DIRECT_VARIANT_GROUPED:
            return self.cpg, self.kpg
        return 1, 1

    def _direct_rocke_caps(self) -> DirectCaps:
        cpg, kpg = self._direct_caps()
        return DirectCaps(
            KH=self.filter_h,
            PAD=self.pad_h,
            stride=self.stride_h,
            cpg=cpg,
            kpg=kpg,
            dil_h=self.dilation_h,
            dil_w=self.dilation_w,
        )

    def probe_problem(self) -> _conv.ConvProblem | _direct.DirectConvProblem:
        """The problem rocKE's sweep builds this instance against.

        rocKE's own probe: ``_probe_problem("fwd", caps)`` of
        ``kernels/conv/aot_jobs.py`` for implicit GEMM, ``probe_problem(caps,
        dtype, variant, knobs)`` of ``kernels/conv/direct_aot_jobs.py`` for a
        direct instance. The emitted code does not depend on the probe's
        extents, only on its capability axes.
        """
        if self.is_direct:
            return _direct_probe_problem(
                self._direct_rocke_caps(),
                self.dtype,
                self.algorithm,
                self.direct_knobs(),
            )
        return _aot_probe_problem("fwd", {"grouped": bool(self.grouped)})

    def _field_instance_spec(self) -> _conv.ImplicitGemmConvSpec:
        """The rocKE builder spec these fields spell, without the sweep lookup
        (to report rocKE's own reason for a spec the sweep does not build)."""
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

    def to_instance_spec(self) -> _conv.ImplicitGemmConvSpec:
        """The rocKE builder spec ``aot_jobs.kernel_for_job`` builds this
        instance's sweep job with: the job's spec kwargs on the job's probe.
        Only the name differs. Raises ``ValueError`` for an instance the sweep
        does not build."""
        if self.is_direct:
            raise ValueError("a direct instance has no implicit-GEMM spec")
        job = _aot_sweep_job(self)
        return _conv.ImplicitGemmConvSpec(
            problem=_aot_probe_problem(job.direction, job.caps),
            data=_conv.ConvDataSpec(
                dtype_a=self.dtype, dtype_b=self.dtype, dtype_d=self.dtype
            ),
            name=self.kernel_symbol_prefix(),
            **job.spec_kwargs,
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
        ``direct_variants.make_direct_spec`` makes it; only the name differs."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        spec = make_direct_spec(self.algorithm, problem, self.direct_knobs())
        return replace(spec, name=self.kernel_symbol_prefix())

    def to_direct_spec(self):
        """The rocKE builder spec ``direct_aot_jobs.direct_kernel_for_job``
        builds this instance's sweep job with: the job's knobs on the job's
        probe. Only the name differs. Raises ``ValueError`` for an instance the
        sweep does not build."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        job = _aot_sweep_job(self)
        probe = _direct_probe_problem(
            DirectCaps(**job.caps), self.dtype, self.algorithm, job.spec_kwargs
        )
        spec = make_direct_spec(self.algorithm, probe, job.spec_kwargs)
        return replace(spec, name=self.kernel_symbol_prefix())

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

        The exact-shape direct guard reads only the problem plus
        ``direct_variant``/``block_w``/``block_waves``, so a std or spatial
        instance reuses it unchanged through this. The other direct variants
        have no exact-shape spec; see :meth:`runtime_direct_spec`.
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
        ``direct_variants.direct_spec_for_problem`` re-validates and launches."""
        return self.rocke_direct_spec(aot_direct_problem(problem))

    def block(self) -> tuple[int, int, int]:
        """Workgroup size; a function of the tuning alone, never of the shape."""
        if self.is_direct:
            spec = self.rocke_direct_spec(self.probe_problem())
            return spec.threads_per_block, 1, 1
        return self.warp_m * self.warp_n * self.wave_size, 1, 1

    def request(self, problem: Gfx950ConvFwdSpec) -> ConvGroupedRequest:
        """``problem`` as a request to rocKE's grouped-convolution dispatcher, for
        this instance's LLVM flavor."""
        return ConvGroupedRequest(
            N=problem.N,
            C=problem.C,
            K=problem.K,
            Hi=problem.Hi,
            Wi=problem.Wi,
            Y=problem.Y,
            X=problem.X,
            arch=_ARCH,
            G=problem.groups,
            stride_h=problem.sH,
            stride_w=problem.sW,
            pad_h=problem.pH,
            pad_w=problem.pW,
            dilation_h=problem.dH,
            dilation_w=problem.dW,
            dtype=problem.dtype,
            layout=problem.layout,
            direction="fwd",
            llvm_flavor=self.identity().llvm_flavor,
        )

    def launch_step(self, problem: Gfx950ConvFwdSpec):
        """The dispatcher's launch (``launch_plan_for``) of this instance on
        ``problem``, which it must serve: its one kernel launch step."""
        plan = launch_plan_for(self.request(problem), _aot_instance(self))
        if len(plan.steps) != 1 or plan.workspaces or plan.num_sub_gemms:
            raise ValueError("the packaged launch runs one kernel and no workspace")
        return plan.steps[0]

    def grid(self, problem: Gfx950ConvFwdSpec) -> tuple[int, int, int]:
        """The dispatcher's launch grid for ``problem``: ``(K/groups tiles,
        N*Ho*Wo tiles, groups)`` for implicit GEMM, rocKE's direct launch
        geometry for a direct instance."""
        return tuple(self.launch_step(problem).grid)

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
        """Every kernel argument for ``problem``: the dispatcher's launch plan,
        computed by rocKE's ``ConvArgs``."""
        step = self.launch_step(problem)
        return step.args.to_launch_values(
            a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes
        )


def _aot_identity_fields(spec: Gfx950ConvFwdAotSpec) -> dict:
    """The ``KernelIdentity`` fields ``spec`` spells, all but ``llvm_flavor``."""
    values = dict(
        arch=_ARCH,
        dtype_a=spec.dtype,
        dtype_b=spec.dtype,
        dtype_d=spec.dtype,
        **{name: getattr(spec, name) for name in AOT_IMPLICIT_GEMM_CARRIED},
    )
    for name in ("unroll_k", "async_dma", "grouped"):
        values[name] = bool(values[name])
    if not spec.is_direct:
        return dict(values, direction="fwd", algorithm="implicit_gemm")
    cpg, kpg = spec._direct_caps()
    return dict(
        values,
        direction="direct_fwd",
        algorithm=spec.algorithm,
        filter_h=spec.filter_h,
        filter_w=spec.filter_w,
        stride_h=spec.stride_h,
        stride_w=spec.stride_w,
        pad_h=spec.pad_h,
        pad_w=spec.pad_w,
        dilation_h=spec.dilation_h,
        dilation_w=spec.dilation_w,
        cpg=cpg,
        kpg=kpg,
        knobs=json.dumps(spec.direct_knobs(), sort_keys=True),
    )


@functools.lru_cache(maxsize=1 << 18)
def _aot_identity(spec: Gfx950ConvFwdAotSpec) -> KernelIdentity:
    values = _aot_identity_fields(spec)
    for flavor in LLVM_FLAVORS:
        identity = KernelIdentity(**values, llvm_flavor=flavor)
        if identity.stable_hash() == spec.instance_id:
            return identity
    return KernelIdentity(**values, llvm_flavor=_AOT_DEFAULT_LLVM_FLAVOR)


@functools.lru_cache(maxsize=1 << 14)
def _aot_sweep_job(spec: Gfx950ConvFwdAotSpec):
    """The job rocKE's sweep builds ``spec``'s identity from; ``ValueError`` when
    the sweep does not build it."""
    identity = spec.identity()
    try:
        if spec.is_direct:
            return direct_job_for_identity(identity)
        return job_for_identity(identity)
    except LookupError as exc:
        raise ValueError(
            f"not an identity of rocKE's {_ARCH} forward AOT sweep: {exc}"
        ) from None


@functools.lru_cache(maxsize=1 << 14)
def _aot_instance(spec: Gfx950ConvFwdAotSpec) -> ConvInstanceSpec:
    """``spec`` as the dispatcher's instance, for its launch plan."""
    identity = spec.identity()
    return ConvInstanceSpec(
        name=identity.label(), identity=identity, block=spec.block()
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


def _aot_sweep_error(spec: Gfx950ConvFwdAotSpec) -> tuple[bool, str]:
    """``spec`` must be an identity of rocKE's sweep, and ``instance_id`` its hash."""
    try:
        _aot_sweep_job(spec)
    except ValueError as exc:
        return False, str(exc)
    if spec.identity().stable_hash() != spec.instance_id:
        return False, (
            f"instance_id {spec.instance_id} is not the stable_hash() of the "
            f"identity the instance's fields spell ({spec.identity().label()})"
        )
    return True, "ok"


def supports_gfx950_conv_fwd_aot(
    spec: Gfx950ConvFwdAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance.

    The packaged contract's field checks, then rocKE's validators on the
    sweep's probe, then rocKE's sweep itself: the fields must spell an identity
    the sweep builds (``job_for_identity`` / ``direct_job_for_identity``), whose
    ``stable_hash()`` is ``instance_id``.
    """
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
            ok, why = validate_direct_spec(
                spec.algorithm, spec.rocke_direct_spec(spec.probe_problem()), arch
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return False, str(exc)
        if not ok:
            return False, why
        return _aot_sweep_error(spec)
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
    if not spec.async_dma and (spec.async_chunk_a or spec.async_chunk_b):
        return False, "async_chunk_a/async_chunk_b are 0 without async_dma"
    try:
        instance = spec._field_instance_spec()
        instance.validate()
        ok, reason = _conv.is_valid_spec_for_problem(
            instance, instance.problem, arch=arch
        )
        if not ok:
            return False, reason
        if spec.async_dma:
            a_loader, b_loader = _conv.async_tile_loaders(instance, arch)
            chunks = (a_loader.elems_per_chunk, b_loader.elems_per_chunk)
            if chunks != (spec.async_chunk_a, spec.async_chunk_b):
                return False, (
                    f"async chunk widths {chunks} differ from the recorded "
                    f"({spec.async_chunk_a}, {spec.async_chunk_b})"
                )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return _aot_sweep_error(spec)


def gfx950_conv_fwd_aot_spec_for_identity(
    identity: KernelIdentity,
) -> tuple[Gfx950ConvFwdAotSpec | None, str]:
    """``(instance, "")`` for a packageable rocKE identity, else ``(None, reason)``.

    Packageable: a gfx950 forward identity in fp16 or bf16 of rocKE's sweeps,
    with no feature the packaged launch does not model. Refused, each with its
    reason:

    * implicit GEMM with ``group_merge`` > 1 (``AOT_REFUSED_GROUP_MERGE``), or any
      other identity field off its ``KernelIdentity`` default that the instance
      does not carry (``AOT_IMPLICIT_GEMM_CARRIED``);
    * the direct algorithms without a packaged launch: ``direct_grouped_halo``
      (``AOT_REFUSED_GROUPED_HALO``) and any other one outside
      ``AOT_DIRECT_ALGORITHMS``;
    * split-C non-grouped kernels (knob ``split_c``; ``AOT_REFUSED_SPLIT_C``) and
      their fold, ``direct_split_reduce`` (``AOT_REFUSED_SPLIT_REDUCE``);
    * anything :func:`supports_gfx950_conv_fwd_aot` declines.
    """
    if not isinstance(identity, KernelIdentity):
        return None, "expected a KernelIdentity"
    kind = f"({identity.arch}, {identity.direction}, {identity.algorithm})"
    if identity.arch != _ARCH:
        return None, f"not a {_ARCH} forward implicit-GEMM or direct identity {kind}"
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    if identity.direction == "direct_fwd_helper":
        if identity.algorithm == DIRECT_SPLIT_REDUCE:
            return None, AOT_REFUSED_SPLIT_REDUCE
        return None, f"forward helper {identity.algorithm!r} is not packaged"
    if identity.direction == "direct_fwd":
        # The packaged variant and knobs first, then the direct sweep's own
        # round trip: every other field must be what the grid writes for these
        # capabilities and knobs, and the kernel's validator must take them.
        try:
            fields_ = aot_direct_instance_fields(identity)
            direct_job_for_identity(identity)
        except (KeyError, LookupError, TypeError, ValueError) as exc:
            return None, str(exc)
        spec = Gfx950ConvFwdAotSpec(
            dtype=identity.dtype_a,
            wave_size=identity.wave_size,
            instance_id=identity.stable_hash(),
            **_AOT_DIRECT_PLACEHOLDERS,
            **fields_,
        )
    elif (identity.direction, identity.algorithm) == ("fwd", "implicit_gemm"):
        if identity.group_merge != 1:
            return None, AOT_REFUSED_GROUP_MERGE.format(gm=identity.group_merge)
        for name, default in _IDENTITY_DEFAULTS.items():
            if name in AOT_IMPLICIT_GEMM_CARRIED or name in _AOT_IDENTITY_KEYS:
                continue
            if getattr(identity, name) != default:
                return None, (
                    f"{name}={getattr(identity, name)!r} is not packaged (only "
                    f"{default!r}): it selects code the packaged launch does not model"
                )
        values = {name: getattr(identity, name) for name in AOT_IMPLICIT_GEMM_CARRIED}
        for name in ("unroll_k", "async_dma", "grouped"):
            values[name] = int(values[name])
        spec = Gfx950ConvFwdAotSpec(
            dtype=identity.dtype_a, instance_id=identity.stable_hash(), **values
        )
    else:
        return None, f"not a {_ARCH} forward implicit-GEMM or direct identity {kind}"
    ok, reason = supports_gfx950_conv_fwd_aot(spec)
    if not ok:
        return None, reason
    if spec.identity() != identity:
        return None, (
            f"the instance spells {spec.identity().label()}, not {identity.label()}"
        )
    return spec, ""


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


def _aot_admits(
    spec: Gfx950ConvFwdAotSpec, problem: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """rocKE's grouped-convolution dispatcher's admission of ``spec``'s identity
    for ``problem``: ``identity_supports_problem``, and for a direct identity the
    per-shape re-validation of its knobs (``direct_spec_for_problem``)."""
    ok, why = _dispatcher_admits(spec.identity(), spec.request(problem))
    return (True, "ok") if ok else (False, why)


def _aot_direct_serves(
    spec: Gfx950ConvFwdAotSpec, problem: Gfx950ConvFwdSpec
) -> tuple[bool, str]:
    """Whether a direct instance other than std or spatial computes ``problem``,
    which already passed the problem checks.

    rocKE's dispatcher admission (:func:`_aot_admits`), after naming the first
    compiled capability the problem does not have; the kernels compile one
    stride and one padding, so both axes must agree. Then the launch has to fit:
    every grid axis 65535, the non-grouped kernel's flat grid a 32-bit dispatch.
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
    ok, why = _aot_admits(spec, problem)
    if not ok:
        return False, why
    step = spec.launch_step(problem)
    grid = step.grid
    if variant == DIRECT_VARIANT_NONGROUPED:
        if (
            grid[0] > _AOT_NONGROUPED_MAX_GRID_X
            or grid[0] * step.block[0] > _AOT_MAX_DISPATCH_ITEMS
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

    Implicit GEMM: rocKE's dispatcher admission of the identity
    (``identity_supports_problem``: the vector widths and async chunks divide
    the contiguous per-group runs, a grouped problem needs a grouped binary,
    every grid axis fits 65535), plus the reduction extent the kernel's 24-bit
    products bound. A grouped binary was not built as pointwise, so grouped
    pointwise problems are served.

    Direct std and spatial: the exact-shape engine's direct rule
    (:func:`_direct_serves`) on the instance's compiled filter, stride and
    padding, which the native pack shares with the exact-shape engine. The
    other direct variants: rocKE's dispatcher admission plus the packaged
    launch bounds (:func:`_aot_direct_serves`). Batch, image and group count
    are kernel arguments, and for non-grouped the channel counts too.
    """
    error = _problem_error(problem, decline_grouped_pointwise=False)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_direct:
        if spec.direct_variant in _DIRECT_VARIANTS:
            caps = gfx950_conv_fwd_aot_capabilities(spec)
            return _direct_serves(caps, spec.runtime_spec(problem))
        return _aot_direct_serves(spec, problem)
    if problem.Y * problem.X * (problem.C // problem.groups) >= MUL24_REDUCTION_LIMIT:
        return False, "the reduction extent Y*X*C/groups must stay below 2**23"
    return _aot_admits(spec, problem)


def gfx950_conv_fwd_aot_runtime_candidates(
    problem: Gfx950ConvFwdSpec, specs: Sequence[Gfx950ConvFwdAotSpec]
) -> list[Gfx950ConvFwdAotSpec]:
    """The instances of ``specs`` hipkernel:Gfx950ConvFwdAot selects for ``problem``.

    "Serves" (:func:`gfx950_conv_fwd_aot_serves`) is what a binary can compute;
    this is what the engine then runs, so it depends on the whole catalog ``specs``,
    not on one instance. The problem's widest combination is the widest load width
    C/groups admits and the widest store width K/groups admits (the largest of 8, 4,
    2 and 1 dividing each). When some serving implicit-GEMM instance bakes exactly
    that combination, only those run, of either epilogue: in a catalog that builds
    every width this is ``--run-from-cache`` in
    ``benchmarks/common/benchmark_implicit_gemm_conv.py`` with its default policy
    (``_widest_vector_sizes``). The shipped catalog builds only some combinations,
    so when none serving bakes the problem's widest, every serving instance runs.
    rocKE's direct benchmark runs every direct kernel that serves, so direct
    instances are all kept. The native pack's ``conv::widestVectorSelection`` is the
    same policy; the result keeps the order of ``specs``.
    """
    serving = [spec for spec in specs if gfx950_conv_fwd_aot_serves(spec, problem)[0]]
    load = _widest_vector_width(problem.C // problem.groups)
    store = _widest_vector_width(problem.K // problem.groups)
    widest = [
        spec
        for spec in serving
        if not spec.is_direct
        and (spec.vector_size_a, spec.vector_size_b, spec.vector_size_c)
        == (load, load, store)
    ]
    if not widest:
        return serving
    selected = {id(spec) for spec in widest}
    return [spec for spec in serving if spec.is_direct or id(spec) in selected]


def _widest_vector_width(channels: int) -> int:
    """The largest of 8, 4, 2 and 1 that divides ``channels``."""
    return next(width for width in (8, 4, 2, 1) if channels % width == 0)


def build_gfx950_conv_fwd_aot(
    spec: Gfx950ConvFwdAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Builds what rocKE's AOT build builds for the identity: ``kernel_for_job`` of
    its sweep job for implicit GEMM, ``direct_kernel_for_job`` (the variant's
    builder, ``select_direct_kernel_builder``) for a direct one. Only the kernel
    symbol differs.
    """
    ok, reason = supports_gfx950_conv_fwd_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT forward instance: {reason}")
    if spec.is_direct:
        return select_direct_kernel_builder(spec.algorithm, spec.to_direct_spec(), arch)
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
