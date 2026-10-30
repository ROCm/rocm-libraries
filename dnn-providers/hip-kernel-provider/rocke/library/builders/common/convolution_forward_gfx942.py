# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx942 shape-generic (AOT) forward convolution.

This is the AOT-only counterpart of ``convolution_forward.py`` (gfx950):
same rocKE kernel families (implicit-GEMM AOT sweep and the direct AOT sweep's
depthwise/grouped/nongrouped/grouped-halo/windowed variants), same packaged
contract, ported for gfx942 (MI300X) because nothing here is arch-specific
beyond the ``_ARCH`` constant and the dispatcher request it builds. It does not
port gfx950's exact-shape engine (``hipkernel:Gfx950ConvFwd``'s
``supports_gfx950_conv_fwd`` / ``gfx950_conv_fwd_spec_for_request`` /
``build_gfx950_conv_fwd`` and friends): nothing in the AOT path here calls
into them, and this integration only ships ``hipkernel:Gfx942ConvFwdAot``. If
an exact-shape gfx942 engine is ever wanted, port that half the same way,
following the gfx950 file as the template.

The kernels are rocKE's AOT builds: the ABI is ``(A, B, D, A_bytes:i32,
B_bytes:i32, D_bytes:i32)`` followed by the problem block of
``kernels.common.conv_abi`` (extents, strides, magic divisors and tile counts as
i32 kernel arguments). A packaged kernel is built for one problem, but it
computes every problem its build-time capabilities admit, and the native pack
matches it to all of them. :func:`gfx942_conv_fwd_aot_capabilities` reports what
an instance bakes in and :func:`gfx942_conv_fwd_aot_serves` decides a runtime
problem against it.

The initial contract is 2D cross-correlation, dense NHWC input, KYXC filter,
NHWK output, uniform fp16/bf16, and fp32 accumulation. Padding is symmetric.
Grouped convolution (including depthwise) is supported: the filter is
``[K, Y, X, C/groups]``, each group owns a contiguous ``C/groups`` input and
``K/groups`` output channel slab, and the group rides on grid z. Grouped
pointwise (1x1, stride 1, no padding) is declined because the kernel's flat
pointwise shortcut does not select the group's slabs. Forward needs no
workspace; pointers must be 16-byte aligned and non-aliasing.
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
from rocke.core.ir import KernelDef
from rocke.core.lower_llvm import LLVM_FLAVORS


_ARCH = "gfx942"


_INT32_MAX = (1 << 31) - 1


_MAX_GRID_DIM_Z = 65535


_MAX_GRID_DIM = 65535


_PACKAGED_PIPELINES = ("mem", "compv3", "compv4", "basic")


KERNEL_FAMILY_IMPLICIT_GEMM = 0


KERNEL_FAMILY_DIRECT_DEPTHWISE = 1


DIRECT_VARIANT_NONE = "none"


DIRECT_VARIANT_STD = "std"


DIRECT_VARIANT_SPATIAL = "spatial"


_DIRECT_VARIANTS = (DIRECT_VARIANT_STD, DIRECT_VARIANT_SPATIAL)


_DIRECT_WAVE_SIZE = 64


_DIRECT_MAX_BLOCK_WAVES = 16


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


_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    "block_w": 0,
    "block_waves": 0,
}


@dataclass(frozen=True)
class Gfx942ConvFwdSpec:
    """Serializable problem and resolved tuning values for one packaged kernel.

    Use :func:`gfx942_conv_fwd_spec_for_request` to obtain the exact-shape
    engine's defaults (:data:`EXACT_SHAPE_DEFAULTS`), or :func:`gfx942_conv_fwd_direct_spec_for_request` for a
    direct depthwise kernel. Direct construction is useful for descriptor
    hydration; :func:`supports_gfx942_conv_fwd` and the builder validate it
    before emission. The geometry is the build problem; the kernel takes the
    runtime problem as AOT arguments (see :func:`gfx942_conv_fwd_serves`).

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
            name=f"hkp_conv_fwd_gfx942_{self._name_suffix()}",
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
        name = f"hkp_conv_fwd_dw_gfx942_{self._name_suffix()}"
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
    spec: Gfx942ConvFwdSpec, *, decline_grouped_pointwise: bool = True
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


def _direct_error(spec: Gfx942ConvFwdSpec) -> str:
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


def _base_spec_for_request(
    request: ConvGroupedRequest, *, decline_grouped_pointwise: bool = True
) -> Gfx942ConvFwdSpec:
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
    base = Gfx942ConvFwdSpec(
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


@dataclass(frozen=True)
class Gfx942ConvFwdCapabilities:
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


def _direct_serves(
    caps: Gfx942ConvFwdCapabilities, runtime: Gfx942ConvFwdSpec
) -> tuple[bool, str]:
    """Whether a direct depthwise binary with ``caps`` computes ``runtime``: the
    runtime problem on the binary's direct tuning, which already passed the problem
    checks. It must pass the direct guard and have the compiled filter, stride and
    padding (the guard already pins dilation to 1).

    Shared by :func:`gfx942_conv_fwd_serves` and :func:`gfx942_conv_fwd_aot_serves`.
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


@dataclass(frozen=True)
class _ProblemFields:
    """The problem half of Gfx942ConvFwdSpec, for moving a runtime problem onto a
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


_AOT_VECTOR_WIDTHS = (1, 2, 4, 8)


_AOT_PIPELINES = _PACKAGED_PIPELINES


_AOT_INSTANCE_ID_LENGTH = 40


_AOT_DEFAULT_LLVM_FLAVOR = "llvm22"


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


DIRECT_VARIANT_TILED = "tiled"


DIRECT_VARIANT_COL = "col"


DIRECT_VARIANT_GROUPED = "grouped"


DIRECT_VARIANT_NONGROUPED = "nongrouped"


DIRECT_VARIANT_GROUPED_HALO = "grouped_halo"


DIRECT_VARIANT_WIN = "win"


AOT_DIRECT_ALGORITHMS = {
    "direct_depthwise": DIRECT_VARIANT_STD,
    "direct_depthwise_spatial": DIRECT_VARIANT_SPATIAL,
    "direct_depthwise_tiled": DIRECT_VARIANT_TILED,
    "direct_depthwise_col": DIRECT_VARIANT_COL,
    "direct_grouped": DIRECT_VARIANT_GROUPED,
    "direct_nongrouped": DIRECT_VARIANT_NONGROUPED,
    DIRECT_GROUPED_HALO: DIRECT_VARIANT_GROUPED_HALO,
    "direct_depthwise_win": DIRECT_VARIANT_WIN,
}


_AOT_HALO_VARIANTS = (DIRECT_VARIANT_NONGROUPED, DIRECT_VARIANT_GROUPED_HALO)


_AOT_DIRECT_VARIANTS = tuple(AOT_DIRECT_ALGORITHMS.values())


_AOT_ALGORITHM_OF_VARIANT = {v: a for a, v in AOT_DIRECT_ALGORITHMS.items()}


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
    "group_merge": 1,
    "split_c": 1,
    "prefetch_rows": 0,
    "w_fold": 0,
}


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
    DIRECT_VARIANT_GROUPED_HALO: {
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
        "group_merge": "group_merge",
    },
    DIRECT_VARIANT_WIN: {
        "block_h": "block_h",
        "block_waves": "block_waves",
        "prefetch_rows": "prefetch_rows",
        "w_fold": "w_fold",
    },
}


_AOT_DIRECT_CONSTANT_KNOBS = {DIRECT_VARIANT_GROUPED_HALO: {"grouped": True}}


_AOT_DIRECT_OPTIONAL_KNOBS = {DIRECT_VARIANT_NONGROUPED: {"split_c": ("split_c", 1)}}


_AOT_DIRECT_EXTRA_FIELDS = {
    DIRECT_VARIANT_GROUPED: ("cpg", "kpg", "block_h"),
    DIRECT_VARIANT_NONGROUPED: ("split_c",),
    DIRECT_VARIANT_WIN: ("cpg", "kpg"),
}


_AOT_DIRECT_BOOL_KNOBS = ("unroll_rows", "double_buffer")


_AOT_DIRECT_NONE_KNOBS = {"iglp": -1, "waves_per_eu": 0}


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


_AOT_NONGROUPED_MAX_GRID_X = _INT32_MAX


_AOT_MAX_DISPATCH_ITEMS = (1 << 32) - 1


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


_AOT_IDENTITY_KEYS = (
    "arch",
    "direction",
    "algorithm",
    "dtype_a",
    "dtype_b",
    "dtype_d",
    "llvm_flavor",
)


_IDENTITY_DEFAULTS = {
    f.name: f.default
    for f in dataclasses.fields(KernelIdentity)
    if f.default is not dataclasses.MISSING
}


AOT_REFUSED_SPLIT_REDUCE = (
    "'direct_split_reduce' is not packaged as an instance: it is the second "
    "launch of a split-C non-grouped instance, not a convolution"
)


def aot_direct_grouped_block_h() -> int:
    """The row tile a grouped direct instance compiles: DirectConvSpec's default,
    since the direct sweep's grouped grid does not vary it."""
    return _direct.DirectConvSpec.__dataclass_fields__["block_h"].default


def aot_direct_fields_from_knobs(variant: str, knobs: dict) -> dict:
    """The instance fields a direct identity's ``knobs`` of ``variant`` set.

    Raises ``ValueError`` when the keys are not exactly the variant's (its
    optional knobs may be absent), or a constant knob holds another value.
    """
    mapping = AOT_DIRECT_KNOBS[variant]
    constants = _AOT_DIRECT_CONSTANT_KNOBS.get(variant, {})
    optional = _AOT_DIRECT_OPTIONAL_KNOBS.get(variant, {})
    keys = set(knobs) - set(optional)
    if keys != set(mapping) | set(constants):
        raise ValueError(
            f"knobs {sorted(knobs)} are not {sorted(set(mapping) | set(constants))}"
            + (f" (optional: {sorted(optional)})" if optional else "")
        )
    for key, value in constants.items():
        if knobs[key] != value or type(knobs[key]) is not type(value):
            raise ValueError(f"knob {key} must be {value!r}, got {knobs[key]!r}")
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
    for key, (field_name, default) in optional.items():
        values[field_name] = knobs.get(key, default)
    return values


def aot_direct_instance_fields(identity: KernelIdentity) -> dict:
    """The direct fields of the instance packaged from a forward direct identity.

    Reads the identity's ``algorithm``, compiled filter, stride, padding,
    dilation and channel counts, and ``knobs`` JSON. Raises ``ValueError`` for
    an algorithm without a packaged launch or knobs that are not exactly its
    variant's. The caller states the placeholders (``_AOT_DIRECT_PLACEHOLDERS``)
    and checks the result with :func:`supports_gfx942_conv_fwd_aot`.
    """
    variant = AOT_DIRECT_ALGORITHMS.get(identity.algorithm)
    if variant is None:
        raise ValueError(
            f"direct algorithm {identity.algorithm!r} has no packaged launch "
            f"(supported: {sorted(AOT_DIRECT_ALGORITHMS)})"
        )
    knobs = json.loads(identity.knobs) if identity.knobs else {}
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
    elif variant == DIRECT_VARIANT_WIN:
        fields_.update(cpg=identity.cpg, kpg=identity.kpg)
    return fields_


@dataclass(frozen=True)
class Gfx942ConvFwdAotSpec:
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
    field at ``_AOT_FAMILY_DEFAULTS``, except ``group_merge``: an implicit-GEMM
    instance with ``group_merge`` > 1 is a depthwise kernel in which one
    workgroup owns that many groups, and records its builder-derived vector
    widths as 0, as rocKE's sweep does.

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
    group_merge: int = 1
    split_c: int = 1
    prefetch_rows: int = 0
    w_fold: int = 0

    @property
    def is_direct(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_DIRECT_DEPTHWISE

    @property
    def is_merged(self) -> bool:
        """A group-merged depthwise implicit-GEMM instance: one workgroup owns
        ``group_merge`` groups, and the builder derives its vector widths from
        the merged problem (recorded as 0)."""
        return not self.is_direct and self.group_merge > 1

    @property
    def is_split_c(self) -> bool:
        """A split-C non-grouped direct instance: two launches, the conv pass
        into an fp32 workspace and ``direct_split_reduce``."""
        return self.is_direct and self.split_c > 1

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
        consistent (:func:`supports_gfx942_conv_fwd_aot` checks them first).
        """
        return _aot_identity(self)

    def _direct_caps(self) -> tuple[int, int]:
        """``(cpg, kpg)`` the direct kernel compiles: 1 for the depthwise variants,
        the recorded counts for grouped, 0 (runtime) for non-grouped and
        grouped-halo."""
        if self.direct_variant in _AOT_HALO_VARIANTS:
            return 0, 0
        if self.direct_variant in (DIRECT_VARIANT_GROUPED, DIRECT_VARIANT_WIN):
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
        caps = {"grouped": bool(self.grouped)}
        if self.is_merged:
            caps["group_merge"] = self.group_merge
        return _aot_probe_problem("fwd", caps)

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
            # A recorded 0 is a width the builder derives (a merged instance's),
            # passed as None, as the sweep's job does.
            vector_size_a=self.vector_size_a or None,
            vector_size_b=self.vector_size_b or None,
            vector_size_c=self.vector_size_c or None,
            unroll_k=bool(self.unroll_k),
            async_dma=bool(self.async_dma),
            group_merge=self.group_merge,
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
        knobs.update(_AOT_DIRECT_CONSTANT_KNOBS.get(self.direct_variant, {}))
        optional = _AOT_DIRECT_OPTIONAL_KNOBS.get(self.direct_variant, {})
        for key, (field_name, default) in optional.items():
            if getattr(self, field_name) != default:
                knobs[key] = getattr(self, field_name)
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
            return f"hkp_conv_fwd_dw_aot_gfx942_{self.instance_id[:24]}"
        return f"hkp_conv_fwd_aot_gfx942_{self.instance_id[:24]}"

    def kernel_name(self) -> str:
        if self.is_direct:
            return self.to_direct_spec().kernel_name()
        return self.to_instance_spec().kernel_name()

    def runtime_spec(self, problem: Gfx942ConvFwdSpec) -> Gfx942ConvFwdSpec:
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

    def runtime_direct_spec(self, problem: Gfx942ConvFwdSpec):
        """rocKE's kernel spec of this direct instance on ``problem``: what
        ``direct_variants.direct_spec_for_problem`` re-validates and launches."""
        return self.rocke_direct_spec(aot_direct_problem(problem))

    def block(self) -> tuple[int, int, int]:
        """Workgroup size; a function of the tuning alone, never of the shape."""
        if self.is_direct:
            spec = self.rocke_direct_spec(self.probe_problem())
            return spec.threads_per_block, 1, 1
        return self.warp_m * self.warp_n * self.wave_size, 1, 1

    def request(self, problem: Gfx942ConvFwdSpec) -> ConvGroupedRequest:
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

    def launch_step(self, problem: Gfx942ConvFwdSpec):
        """The dispatcher's launch (``launch_plan_for``) of this instance's own
        kernel on ``problem``, which it must serve.

        A merged instance tiles the merged GEMM N, ``K/groups * group_merge``,
        and launches ``groups / group_merge`` z slices
        (``implicit_gemm_conv_grid``), with the true problem's kernel
        arguments; the dispatcher's plan does not merge, so the grid is
        replaced here. A split-C instance's step is its conv pass, whose D is
        the fp32 workspace (``ws_split``) that :meth:`split_reduce_step` folds.
        """
        plan = launch_plan_for(self.request(problem), _aot_instance(self))
        if len(plan.steps) != 1 or plan.workspaces or plan.num_sub_gemms:
            raise ValueError("the packaged launch runs one kernel and no workspace")
        step = plan.steps[0]
        if self.is_merged:
            merged_n = (problem.K // problem.groups) * self.group_merge
            step = replace(
                step,
                grid=(
                    -(-merged_n // self.tile_n),
                    step.grid[1],
                    problem.groups // self.group_merge,
                ),
            )
        if self.is_split_c:
            step = replace(step, buffers=(step.buffers[0], step.buffers[1], "ws_split"))
        return step

    def split_reduce_spec(self):
        """The ``direct_split_reduce`` spec a split-C instance launches second:
        rocKE's fold of ``split_c`` fp32 slabs, as the direct sweep builds it for
        this dtype (``_split_reduce_job``); only the name differs."""
        if not self.is_split_c:
            raise ValueError("only a split-C instance has a split reduction")
        spec = make_direct_spec(
            DIRECT_SPLIT_REDUCE,
            self.probe_problem(),
            {"split_c": self.split_c},
        )
        return replace(spec, name=f"hkp_conv_fwd_sr_aot_gfx942_{self.instance_id[:24]}")

    def split_workspace_bytes(self, problem: Gfx942ConvFwdSpec) -> int:
        """Bytes of the fp32 ``[split_c, N, Ho, Wo, K]`` workspace a split-C
        instance's conv pass writes on ``problem``; 0 for any other instance."""
        if not self.is_split_c:
            return 0
        return self.runtime_direct_spec(problem).workspace_bytes

    def split_reduce_step(self, problem: Gfx942ConvFwdSpec):
        """The second launch of a split-C instance on ``problem``: the fold reads
        the workspace (A), ignores B and writes the output (D), with the
        forward direct-conv arguments of the conv pass, as rocKE's direct sweep
        runs it (``direct_kernel_sweep.plan_for``)."""
        conv = self.launch_step(problem)
        reduce_spec = self.split_reduce_spec()
        return replace(
            conv,
            grid=reduce_spec.grid(aot_direct_problem(problem)),
            block=(reduce_spec.threads, 1, 1),
            buffers=("ws_split", conv.buffers[1], "y"),
        )

    def grid(self, problem: Gfx942ConvFwdSpec) -> tuple[int, int, int]:
        """The launch grid of this instance's own kernel for ``problem``:
        ``(K/groups tiles, N*Ho*Wo tiles, groups)`` for implicit GEMM (over the
        merged GEMM N and ``groups / group_merge`` for a merged instance),
        rocKE's direct launch geometry for a direct instance."""
        return tuple(self.launch_step(problem).grid)

    def launch_signature(self) -> list[dict]:
        if self.is_direct:
            return conv_direct_args_signature(self.dtype)
        return conv_args_signature(self.dtype)

    def launch_values(
        self,
        problem: Gfx942ConvFwdSpec,
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


def _aot_identity_fields(spec: Gfx942ConvFwdAotSpec) -> dict:
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
        # A direct kernel's merge degree is a knob (``group_merge`` in its
        # knobs JSON); only implicit GEMM carries it as the identity field.
        return dict(
            values,
            direction="fwd",
            algorithm="implicit_gemm",
            group_merge=spec.group_merge,
        )
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
def _aot_identity(spec: Gfx942ConvFwdAotSpec) -> KernelIdentity:
    values = _aot_identity_fields(spec)
    for flavor in LLVM_FLAVORS:
        identity = KernelIdentity(**values, llvm_flavor=flavor)
        if identity.stable_hash() == spec.instance_id:
            return identity
    return KernelIdentity(**values, llvm_flavor=_AOT_DEFAULT_LLVM_FLAVOR)


@functools.lru_cache(maxsize=1 << 14)
def _aot_sweep_job(spec: Gfx942ConvFwdAotSpec):
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
def _aot_instance(spec: Gfx942ConvFwdAotSpec) -> ConvInstanceSpec:
    """``spec`` as the dispatcher's instance, for its launch plan."""
    identity = spec.identity()
    return ConvInstanceSpec(
        name=identity.label(), identity=identity, block=spec.block()
    )


def aot_direct_problem(problem: Gfx942ConvFwdSpec) -> _direct.DirectConvProblem:
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


def _aot_direct_knob_error(spec: Gfx942ConvFwdAotSpec) -> str:
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


def _aot_direct_error(spec: Gfx942ConvFwdAotSpec) -> str:
    """The direct instance contract, before rocKE's validators; "" when accepted.

    rocKE's direct kernels compile one stride and one padding (they read
    neither ``sW`` nor ``pW``), and the direct sweep builds square filters
    only; the per-axis fields state that without a convention. Only the
    non-grouped and grouped-halo kernels compute a dilated convolution.
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
    if spec.direct_variant not in _AOT_HALO_VARIANTS and (
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


def _aot_sweep_error(spec: Gfx942ConvFwdAotSpec) -> tuple[bool, str]:
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


def supports_gfx942_conv_fwd_aot(
    spec: Gfx942ConvFwdAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance.

    The packaged contract's field checks, then rocKE's validators on the
    sweep's probe, then rocKE's sweep itself: the fields must spell an identity
    the sweep builds (``job_for_identity`` / ``direct_job_for_identity``), whose
    ``stable_hash()`` is ``instance_id``.
    """
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx942ConvFwdAotSpec):
        return False, "expected Gfx942ConvFwdAotSpec"
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
        if name == "group_merge":
            if type(value) is not int or not 1 <= value <= _MAX_GRID_DIM:
                return False, "group_merge must be a positive integer"
            continue
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
    widths = (spec.vector_size_a, spec.vector_size_b, spec.vector_size_c)
    if spec.is_merged:
        # The builder derives a merged instance's widths from the merged problem.
        if widths != (0, 0, 0) or spec.grouped != 1:
            return False, (
                "a merged instance records derived vector widths (0, 0, 0) and is "
                "built on the grouped code path"
            )
    else:
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


def gfx942_conv_fwd_aot_spec_for_identity(
    identity: KernelIdentity,
) -> tuple[Gfx942ConvFwdAotSpec | None, str]:
    """``(instance, "")`` for a packageable rocKE identity, else ``(None, reason)``.

    Packageable: a gfx942 forward identity in fp16 or bf16 of rocKE's sweeps,
    with no feature the packaged launch does not model. Refused, each with its
    reason:

    * implicit GEMM with any identity field off its ``KernelIdentity`` default
      that the instance does not carry (``AOT_IMPLICIT_GEMM_CARRIED`` and
      ``group_merge``);
    * the direct algorithms without a packaged launch: any one outside
      ``AOT_DIRECT_ALGORITHMS``, and ``direct_split_reduce``, which ships as the
      second launch of each split-C instance (``AOT_REFUSED_SPLIT_REDUCE``);
    * anything :func:`supports_gfx942_conv_fwd_aot` declines.
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
        spec = Gfx942ConvFwdAotSpec(
            dtype=identity.dtype_a,
            wave_size=identity.wave_size,
            instance_id=identity.stable_hash(),
            **_AOT_DIRECT_PLACEHOLDERS,
            **fields_,
        )
    elif (identity.direction, identity.algorithm) == ("fwd", "implicit_gemm"):
        for name, default in _IDENTITY_DEFAULTS.items():
            if (
                name in AOT_IMPLICIT_GEMM_CARRIED
                or name in _AOT_IDENTITY_KEYS
                or name == "group_merge"
            ):
                continue
            if getattr(identity, name) != default:
                return None, (
                    f"{name}={getattr(identity, name)!r} is not packaged (only "
                    f"{default!r}): it selects code the packaged launch does not model"
                )
        values = {name: getattr(identity, name) for name in AOT_IMPLICIT_GEMM_CARRIED}
        for name in ("unroll_k", "async_dma", "grouped"):
            values[name] = int(values[name])
        spec = Gfx942ConvFwdAotSpec(
            dtype=identity.dtype_a,
            instance_id=identity.stable_hash(),
            group_merge=identity.group_merge,
            **values,
        )
    else:
        return None, f"not a {_ARCH} forward implicit-GEMM or direct identity {kind}"
    ok, reason = supports_gfx942_conv_fwd_aot(spec)
    if not ok:
        return None, reason
    if spec.identity() != identity:
        return None, (
            f"the instance spells {spec.identity().label()}, not {identity.label()}"
        )
    return spec, ""


def gfx942_conv_fwd_aot_capabilities(
    spec: Gfx942ConvFwdAotSpec,
) -> Gfx942ConvFwdCapabilities:
    """What instance ``spec`` bakes in, as :func:`gfx942_conv_fwd_capabilities` reports it.

    The widths are the identity's own, not derived from a build problem. A and B
    both run along C/groups, so both divide it exactly when their lcm does. The
    probe problem is 3x3, so an instance is never the pointwise GEMM. A direct
    instance reports its compiled filter, stride and padding.
    """
    if spec.is_direct:
        return Gfx942ConvFwdCapabilities(
            filter_hw=(spec.filter_h, spec.filter_w),
            stride_hw=(spec.stride_h, spec.stride_w),
            pad_hw=(spec.pad_h, spec.pad_w),
        )
    if spec.is_merged:
        # Derived by the builder on the merged channel run, which every problem a
        # merged instance serves shares (cpg = kpg = 1): 1 divides every run.
        return Gfx942ConvFwdCapabilities(
            load_width=1, store_width=1, pointwise=False, grouped=True
        )
    return Gfx942ConvFwdCapabilities(
        load_width=math.lcm(spec.vector_size_a, spec.vector_size_b),
        store_width=spec.vector_size_c,
        pointwise=False,
        grouped=bool(spec.grouped),
    )


def _aot_admits(
    spec: Gfx942ConvFwdAotSpec, problem: Gfx942ConvFwdSpec
) -> tuple[bool, str]:
    """rocKE's grouped-convolution dispatcher's admission of ``spec``'s identity
    for ``problem``: ``identity_supports_problem``, and for a direct identity the
    per-shape re-validation of its knobs (``direct_spec_for_problem``)."""
    ok, why = _dispatcher_admits(spec.identity(), spec.request(problem))
    return (True, "ok") if ok else (False, why)


def _aot_direct_serves(
    spec: Gfx942ConvFwdAotSpec, problem: Gfx942ConvFwdSpec
) -> tuple[bool, str]:
    """Whether a direct instance other than std or spatial computes ``problem``,
    which already passed the problem checks.

    rocKE's dispatcher admission (:func:`_aot_admits`), after naming the first
    compiled capability the problem does not have; the kernels compile one
    stride and one padding, so both axes must agree. The non-grouped kernel
    serves groups == 1 only and the grouped-halo kernel groups > 1 only, as
    rocKE's ``direct_plans`` offers them. Then the launch has to fit: every grid
    axis 65535, the flat grid of the non-grouped and grouped-halo kernels (and
    of a split-C instance's fold) a 32-bit dispatch.
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
    elif variant == DIRECT_VARIANT_GROUPED_HALO:
        if problem.groups == 1:
            return False, "the grouped-halo direct kernel serves groups > 1 only"
    elif (cpg, kpg) != spec._direct_caps():
        return False, (
            f"direct kernel compiled for cpg, kpg = {spec._direct_caps()}, "
            f"problem has {(cpg, kpg)}"
        )
    ok, why = _aot_admits(spec, problem)
    if not ok:
        return False, why
    steps = [spec.launch_step(problem)]
    if spec.is_split_c:
        steps.append(spec.split_reduce_step(problem))
    if variant in _AOT_HALO_VARIANTS:
        for step in steps:
            grid = step.grid
            if (
                grid[0] > _AOT_NONGROUPED_MAX_GRID_X
                or grid[0] * step.block[0] > _AOT_MAX_DISPATCH_ITEMS
            ):
                return False, f"flat direct grid {grid[0]} does not fit one dispatch"
        return True, "ok"
    grid = steps[0].grid
    for axis, extent in zip("xyz", grid):
        if extent > _MAX_GRID_DIM:
            return False, f"direct grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return True, "ok"


def gfx942_conv_fwd_aot_batch_chunks(problem: Gfx942ConvFwdSpec) -> list[int] | None:
    """Images per launch of a non-grouped or grouped-halo kernel on ``problem``, or
    None when one launch serves it: rocKE's ``direct_kernel_sweep._batch_chunks``.

    Those kernels address every operand through one 32-bit buffer resource and
    redirect masked accesses past ``_OOB_BASE``, so no operand may reach it. The
    images are independent, so a batch whose activation or output does is split
    into launches whose windows each fit, balanced rather than greedy. None too
    when not even one image fits. The native pack's ``conv::nongroupedBatchChunks``
    is the same rule.
    """
    runtime = problem.to_problem()
    if runtime.Ho <= 0 or runtime.Wo <= 0:
        return None
    in_img = 2 * problem.Hi * problem.Wi * problem.C
    out_img = 2 * runtime.Ho * runtime.Wo * problem.K
    per_launch = _nongrouped._OOB_BASE // max(in_img, out_img)
    if problem.N <= per_launch or per_launch < 1:
        return None
    n_steps = -(-problem.N // per_launch)
    base, extra = divmod(problem.N, n_steps)
    return [base + (i < extra) for i in range(n_steps)]


def gfx942_conv_fwd_aot_serves(
    spec: Gfx942ConvFwdAotSpec, problem: Gfx942ConvFwdSpec
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

    A non-grouped or grouped-halo instance runs a batch rocKE splits in pieces
    (:func:`gfx942_conv_fwd_aot_batch_chunks`) one launch per chunk, as
    ``direct_kernel_sweep._batch_split_plan`` does: it serves when it serves each
    chunk as a problem of its own, and a split-C one, whose workspace would span
    the whole batch, never does. Every other instance needs the whole problem in
    one launch.
    """
    if (
        spec.is_direct
        and spec.direct_variant in _AOT_HALO_VARIANTS
        and problem.dtype == spec.dtype
    ):
        chunks = gfx942_conv_fwd_aot_batch_chunks(problem)
        if chunks is not None:
            if spec.is_split_c:
                return False, "a batch-split launch does not take split-C kernels"
            for images in sorted(set(chunks)):
                ok, why = gfx942_conv_fwd_aot_serves(spec, replace(problem, N=images))
                if not ok:
                    return False, f"batch chunk of {images} image(s): {why}"
            return True, "ok"
    error = _problem_error(problem, decline_grouped_pointwise=False)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_direct:
        if spec.direct_variant in _DIRECT_VARIANTS:
            caps = gfx942_conv_fwd_aot_capabilities(spec)
            return _direct_serves(caps, spec.runtime_spec(problem))
        return _aot_direct_serves(spec, problem)
    if problem.Y * problem.X * (problem.C // problem.groups) >= MUL24_REDUCTION_LIMIT:
        return False, "the reduction extent Y*X*C/groups must stay below 2**23"
    return _aot_admits(spec, problem)


def gfx942_conv_fwd_aot_runtime_candidates(
    problem: Gfx942ConvFwdSpec, specs: Sequence[Gfx942ConvFwdAotSpec]
) -> list[Gfx942ConvFwdAotSpec]:
    """The instances of ``specs`` hipkernel:Gfx942ConvFwdAot selects for ``problem``.

    "Serves" (:func:`gfx942_conv_fwd_aot_serves`) is what a binary can compute;
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
    instances are all kept, and so are group-merged ones: they record derived
    widths (0), so no width comparison applies to them. The native pack's
    ``conv::widestVectorSelection`` is the same policy; the result keeps the order
    of ``specs``.
    """
    serving = [spec for spec in specs if gfx942_conv_fwd_aot_serves(spec, problem)[0]]
    load = _widest_vector_width(problem.C // problem.groups)
    store = _widest_vector_width(problem.K // problem.groups)
    widest = [
        spec
        for spec in serving
        if not spec.is_direct
        and not spec.is_merged
        and (spec.vector_size_a, spec.vector_size_b, spec.vector_size_c)
        == (load, load, store)
    ]
    if not widest:
        return serving
    selected = {id(spec) for spec in widest}
    return [
        spec
        for spec in serving
        if spec.is_direct or spec.is_merged or id(spec) in selected
    ]


def _widest_vector_width(channels: int) -> int:
    """The largest of 8, 4, 2 and 1 that divides ``channels``."""
    return next(width for width in (8, 4, 2, 1) if channels % width == 0)


def build_gfx942_conv_fwd_aot(
    spec: Gfx942ConvFwdAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Builds what rocKE's AOT build builds for the identity: ``kernel_for_job`` of
    its sweep job for implicit GEMM, ``direct_kernel_for_job`` (the variant's
    builder, ``select_direct_kernel_builder``) for a direct one. Only the kernel
    symbol differs.
    """
    ok, reason = supports_gfx942_conv_fwd_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT forward instance: {reason}")
    if spec.is_direct:
        return select_direct_kernel_builder(spec.algorithm, spec.to_direct_spec(), arch)
    return _conv.build_implicit_gemm_conv(spec.to_instance_spec(), arch=arch)


def supports_gfx942_conv_fwd_aot_split_reduce(
    spec: Gfx942ConvFwdAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """hkp_pack's predicate for :func:`build_gfx942_conv_fwd_aot_split_reduce`: a
    packageable split-C instance."""
    ok, reason = supports_gfx942_conv_fwd_aot(spec, arch=arch)
    if not ok:
        return False, reason
    if not spec.is_split_c:
        return False, "only a split-C instance has a split reduction"
    return True, "ok"


def build_gfx942_conv_fwd_aot_split_reduce(
    spec: Gfx942ConvFwdAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The second launch of a split-C instance, the ``reduce`` entry of its
    descriptor's ``sources``: rocKE's ``direct_split_reduce`` fold for the
    instance's dtype and ``split_c`` (:meth:`Gfx942ConvFwdAotSpec.split_reduce_spec`).
    """
    ok, reason = supports_gfx942_conv_fwd_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT forward instance: {reason}")
    if not spec.is_split_c:
        raise ValueError("only a split-C instance has a split reduction")
    return select_direct_kernel_builder(
        DIRECT_SPLIT_REDUCE, spec.split_reduce_spec(), arch
    )


def gfx942_conv_fwd_aot_problem_for_request(
    request: ConvGroupedRequest,
) -> Gfx942ConvFwdSpec:
    """The problem hipkernel:Gfx942ConvFwdAot would match for ``request``.

    Instances are chosen per problem by capability, so there is no single spec
    to resolve; this is the graph side of that match (the profile's
    ``dispatch``). Raises ``ValueError`` outside the graph contract, which
    unlike the exact-shape one includes grouped pointwise.
    """
    return _base_spec_for_request(request, decline_grouped_pointwise=False)


def supports_gfx942_conv_fwd_aot_problem(
    problem: Gfx942ConvFwdSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """The graph contract of hipkernel:Gfx942ConvFwdAot (the profile's predicate)."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(problem, Gfx942ConvFwdSpec):
        return False, "expected Gfx942ConvFwdSpec"
    error = _problem_error(problem, decline_grouped_pointwise=False)
    return (False, error) if error else (True, "ok")
