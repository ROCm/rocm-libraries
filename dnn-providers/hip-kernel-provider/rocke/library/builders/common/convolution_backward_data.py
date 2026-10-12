# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 shape-generic AOT backward-data convolution.

This is the dgrad sibling of the ``Gfx950ConvFwdAot*`` pieces of
:mod:`builders.common.convolution_forward`: one flat descriptor spec
(:class:`Gfx950ConvDgradAotSpec`, every field an ``int`` or ``str``) per rocKE
``KernelIdentity`` of the dgrad AOT sweeps, built exactly as the sweeps build it, so
the packaged IR is the swept kernel's byte for byte apart from its symbol name.

The contract is 2-D cross-correlation, NHWK dY, KYXC W, NHWC dX, uniform fp16/bf16,
fp32 accumulation, symmetric padding. Four families are packaged:

* **Implicit GEMM** (``kernel_family`` 0): ``DgradConvSpec`` of
  ``benchmarks/common/kernel_sweep.py``. One binary serves every stride and dilation:
  the sweep builds the tilde (strided) kernel only, against a stride-2 probe, and the
  kernel reads its sub-GEMM decomposition from a host-packed record table
  (``sub_gemm_buf``, ``pack_sub_gemm_buffer``). The ABI is the 44-argument dgrad
  block of ``kernels.common.conv_abi`` (``dY, W, dX``, byte sizes, extents, GEMM
  dims, strides, the ``Hi``/``Wi`` magic pairs, tile counts, then the
  ``sub_gemm_buf`` pointer and ``num_sub_gemms``). The grid is the flat tile total
  over the sub-GEMMs by the group count. dX rows a strided problem never touches are
  never written, so the caller zero-fills dX first when a stride exceeds 1.
* **Direct depthwise dgrad** (``kernel_family`` 1, ``direct_variant``
  ``depthwise_dgrad``): ``DirectDepthwiseDgradStreamSpec``, baked filter, stride and
  padding, ``cpg == kpg == 1``; the 20-argument direct dgrad ABI.
* **Direct grouped MFMA dgrad** (``kernel_family`` 1, ``direct_variant``
  ``grouped_dgrad_mfma``): a three-step pipeline (weight transpose, an optional
  reorganize, then a *forward* ``DirectConvSpec`` on the transposed problem launched
  with the forward direct ABI), baked 3x3 / pad 1 / stride 1 and exact ``(cpg, kpg)``.
* **MFMA helpers** (``kernel_family`` 2, ``direct_variant`` ``transpose`` or
  ``reorganize``): the weight-transform kernels the MFMA pipeline launches. They are
  packaged so the pipeline's handler can load them, and never serve a problem on
  their own.

:func:`gfx950_conv_dgrad_aot_serves` is rocKE's own offer (``KernelCache
.supports_problem`` plus ``direct_kernel_sweep.plan_for``'s per-shape re-validation)
plus the launch bounds the native pack enforces. The launch mirrors
(:meth:`Gfx950ConvDgradAotSpec.grid`, ``launch_values``,
``sub_gemm_buffer``, :func:`gfx950_conv_dgrad_aot_mfma_plan`) are what the native
pack ports.

``instance_id`` is ``KernelIdentity.stable_hash()``; this layer cannot import
``benchmarks``, so the catalog tool computes it and ``library/tests`` checks it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from dispatch.grouped_convolution import ConvGroupedRequest
from rocke.core.ir import KernelDef
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_implicit_gemm_dgrad as _dgrad
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
from kernels.common.conv_args import ConvArgs

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1
#: Grid y and z (and every axis of the direct compute grids, as for forward).
_MAX_GRID_DIM = 65535
#: A dispatch's x extent in work-items is 32-bit.
_MAX_DISPATCH_ITEMS = (1 << 32) - 1
_WAVE_SIZE = 64
_DTYPES = ("fp16", "bf16")
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_INSTANCE_ID_LENGTH = 40

#: ``kernel_family`` values; the metadata field is also an integer engine knob.
KERNEL_FAMILY_IMPLICIT_GEMM = 0
KERNEL_FAMILY_DIRECT = 1
KERNEL_FAMILY_DGRAD_HELPER = 2
_KERNEL_FAMILIES = (
    KERNEL_FAMILY_IMPLICIT_GEMM,
    KERNEL_FAMILY_DIRECT,
    KERNEL_FAMILY_DGRAD_HELPER,
)

#: ``direct_variant`` values. Implicit GEMM carries "none".
DIRECT_VARIANT_NONE = "none"
DIRECT_VARIANT_DEPTHWISE = "depthwise_dgrad"
DIRECT_VARIANT_MFMA = "grouped_dgrad_mfma"
HELPER_VARIANT_TRANSPOSE = "transpose"
HELPER_VARIANT_REORGANIZE = "reorganize"
_DIRECT_VARIANTS = (DIRECT_VARIANT_DEPTHWISE, DIRECT_VARIANT_MFMA)
_HELPER_VARIANTS = (HELPER_VARIANT_TRANSPOSE, HELPER_VARIANT_REORGANIZE)

#: Each dgrad direct-sweep algorithm (``direct_kernel_sweep.DIRECT_CAPABILITIES`` and
#: its helper variants) and the ``(kernel_family, direct_variant)`` an instance of it
#: carries.
AOT_DGRAD_DIRECT_ALGORITHMS = {
    "direct_depthwise_dgrad": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DEPTHWISE),
    "direct_grouped_dgrad_mfma": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_MFMA),
    "direct_dgrad_transpose": (KERNEL_FAMILY_DGRAD_HELPER, HELPER_VARIANT_TRANSPOSE),
    "direct_dgrad_reorganize": (
        KERNEL_FAMILY_DGRAD_HELPER,
        HELPER_VARIANT_REORGANIZE,
    ),
}
_ALGORITHM_OF_VARIANT = {
    variant: algorithm
    for algorithm, (_, variant) in AOT_DGRAD_DIRECT_ALGORITHMS.items()
}

#: Pipelines the dgrad AOT sweep builds (``basic`` is an alias it skips).
_PIPELINES = ("mem", "compv3", "compv4")
_EPILOGUES = ("default", "cshuffle")
_VECTOR_WIDTHS = (1, 2, 4, 8)

#: The implicit-GEMM fields of a direct or helper instance hold exactly what the
#: direct sweep writes into a direct identity (``direct_kernel_sweep._identity``).
_DIRECT_PLACEHOLDERS = {
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
    "lds_k_outer": 0,
    "max_sub_gemms": 0,
    "grouped": 0,
}
#: The capability fields every direct and helper instance states.
_DIRECT_CAPABILITY_FIELDS = (
    "filter_h",
    "filter_w",
    "stride_h",
    "stride_w",
    "pad_h",
    "pad_w",
    "dilation_h",
    "dilation_w",
    "cpg",
    "kpg",
)
#: Per direct/helper variant, the identity's knob keys (``KernelIdentity.knobs``)
#: and the instance field each one is carried in. The MFMA ``block_h`` is a knob
#: token, not the identity's ``block_h`` field (which the direct sweep leaves 0).
AOT_DGRAD_DIRECT_KNOBS = {
    DIRECT_VARIANT_DEPTHWISE: {"block_w": "block_w", "block_waves": "block_waves"},
    DIRECT_VARIANT_MFMA: {
        "block_groups": "block_groups",
        "block_h": "block_h",
        "block_q": "block_q",
        "fold_k32": "fold_k32",
        "persistent_grid": "persistent_grid",
        "runtime_k_loop": "runtime_k_loop",
        "waves_k": "waves_k",
        "waves_q": "waves_q",
    },
    HELPER_VARIANT_TRANSPOSE: {},
    HELPER_VARIANT_REORGANIZE: {"fold_k32": "fold_k32"},
}
#: Knobs rocKE spells as a bool; the descriptor holds 0/1.
_BOOL_KNOBS = ("fold_k32", "persistent_grid", "runtime_k_loop")
#: The block knobs after the capabilities, with the value each holds where its
#: variant does not use it (the KMD default).
_KNOB_DEFAULTS = {
    "block_w": 0,
    "block_waves": 0,
    "block_h": 0,
    "block_q": 0,
    "block_groups": 0,
    "fold_k32": 0,
    "waves_k": 0,
    "waves_q": 0,
    "persistent_grid": 0,
    "runtime_k_loop": 0,
}
#: Fields appended after the implicit-GEMM ones. An implicit-GEMM instance leaves
#: each at this value (the KMD default); a direct or helper instance leaves the
#: knobs its variant does not use at it.
_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    **{name: 0 for name in _DIRECT_CAPABILITY_FIELDS},
    **_KNOB_DEFAULTS,
}

#: (cpg, kpg) of the dense (groups == 1) MFMA rows; the sweep probes them with one
#: group (``direct_kernel_sweep._DENSE_MFMA_DGRAD_CHANNELS``).
_DENSE_MFMA_CHANNELS = ((4, 320), (24, 24), (48, 192))
#: The direct sweep's probe image (``direct_kernel_sweep.probe_problem``).
_DIRECT_PROBE_HW = 64
#: The weight-workspace offsets of the MFMA pipeline are aligned to this.
MFMA_WORKSPACE_ALIGNMENT = 256

#: Record and magic-table widths of the implicit-GEMM ``sub_gemm_buf``, in i32.
SUB_GEMM_RECORD_FIELDS = 22
SUB_GEMM_MAGIC_FIELDS = 8


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


# ---------------------------------------------------------------------------
# The runtime problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950ConvDgradProblem:
    """One backward-data problem as hipkernel:Gfx950ConvDgradAot matches it.

    ``Hi``/``Wi`` are the dX (forward input) extents, which the graph states: several
    input extents give one output extent under a stride, so they cannot be derived
    from dY. ``C``/``K`` are total channel counts; dY is ``[N, Ho, Wo, K]`` NHWK, W is
    ``[K, Y, X, C/groups]`` KYXC and dX ``[N, Hi, Wi, C]`` NHWC.
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
    groups: int = 1
    dtype: str = "fp16"
    layout: str = "NHWC"

    @property
    def cpg(self) -> int:
        return self.C // self.groups

    @property
    def kpg(self) -> int:
        return self.K // self.groups

    @property
    def Ho(self) -> int:
        return (self.Hi + 2 * self.pH - self.dH * (self.Y - 1) - 1) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.Wi + 2 * self.pW - self.dW * (self.X - 1) - 1) // self.sW + 1

    def to_problem(self) -> ConvProblem:
        """The implicit-GEMM view (``ConvProblem``; its "input" is dX)."""
        return ConvProblem(
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

    def to_direct_problem(self) -> _direct.DirectConvProblem:
        """The direct kernels' view: one stride and one padding (``sH``/``pH``)."""
        return _direct.DirectConvProblem(
            N=self.N,
            H=self.Hi,
            W=self.Wi,
            groups=self.groups,
            cpg=self.cpg,
            kpg=self.kpg,
            KH=self.Y,
            KW=self.X,
            PAD=self.pH,
            stride=self.sH,
            dtype=self.dtype,
            dil_h=self.dH,
            dil_w=self.dW,
        )

    def tensor_bytes(self) -> tuple[int, int, int]:
        """``(dY, W, dX)`` sizes in bytes (16-bit elements)."""
        return (
            self.N * self.Ho * self.Wo * self.K * 2,
            self.K * self.Y * self.X * self.cpg * 2,
            self.N * self.Hi * self.Wi * self.C * 2,
        )


def _problem_error(problem: Gfx950ConvDgradProblem) -> str:
    """The graph contract's problem checks; "" when accepted."""
    for name in ("N", "Hi", "Wi", "C", "K", "Y", "X", "sH", "sW", "dH", "dW"):
        value = getattr(problem, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    for name in ("pH", "pW"):
        value = getattr(problem, name)
        if type(value) is not int or not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    if type(problem.groups) is not int or not 0 < problem.groups <= _MAX_GRID_DIM:
        return f"groups must be an integer in [1, {_MAX_GRID_DIM}] (grid y)"
    if problem.C % problem.groups:
        return f"C={problem.C} is not divisible by groups={problem.groups}"
    if problem.K % problem.groups:
        return f"K={problem.K} is not divisible by groups={problem.groups}"
    if problem.layout != "NHWC":
        return "the packaged dgrad contract requires dense NHWC storage"
    if problem.dtype not in _DTYPES:
        return "the packaged dgrad contract supports fp16 or bf16 only"
    for label, value in (
        ("padded height", problem.Hi + 2 * problem.pH),
        ("padded width", problem.Wi + 2 * problem.pW),
        ("effective filter height", problem.dH * (problem.Y - 1) + 1),
        ("effective filter width", problem.dW * (problem.X - 1) + 1),
    ):
        if value > _INT32_MAX:
            return f"{label} exceeds signed 32-bit descriptor arithmetic"
    if problem.Ho <= 0 or problem.Wo <= 0:
        return "convolution has nonpositive output (dY) spatial dimensions"
    for label, size in zip(("dY", "W", "dX"), problem.tensor_bytes()):
        if size > _INT32_MAX:
            return f"{label} byte count exceeds the signed 32-bit kernel ABI"
    return ""


# ---------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------


def aot_dgrad_direct_fields_from_knobs(variant: str, knobs: dict) -> dict:
    """The instance fields a direct or helper identity's ``knobs`` set.

    Raises ``ValueError`` when the keys are not exactly the variant's, or a bool
    knob is not a bool.
    """
    mapping = AOT_DGRAD_DIRECT_KNOBS[variant]
    if set(knobs) != set(mapping):
        raise ValueError(f"knobs {sorted(knobs)} are not {sorted(mapping)}")
    values = {}
    for key, field_name in mapping.items():
        value = knobs[key]
        if field_name in _BOOL_KNOBS:
            if type(value) is not bool:
                raise ValueError(f"knob {key} must be a bool, got {value!r}")
            value = int(value)
        elif type(value) is not int:
            raise ValueError(f"knob {key} must be an integer, got {value!r}")
        values[field_name] = value
    return values


def aot_dgrad_direct_instance_fields(identity) -> dict:
    """The direct fields of the instance packaged from a dgrad direct or helper identity.

    ``identity`` is read duck-typed (a ``KernelIdentity``, which this layer cannot
    import): its ``algorithm``, compiled filter, stride, padding, dilation, channel
    counts and ``knobs`` JSON. Raises ``ValueError`` for an algorithm without a
    packaged launch or knobs that are not exactly its variant's. The caller states
    the placeholders and checks the result with :func:`supports_gfx950_conv_dgrad_aot`.
    """
    family_variant = AOT_DGRAD_DIRECT_ALGORITHMS.get(identity.algorithm)
    if family_variant is None:
        raise ValueError(
            f"direct algorithm {identity.algorithm!r} has no packaged dgrad launch "
            f"(supported: {sorted(AOT_DGRAD_DIRECT_ALGORITHMS)})"
        )
    family, variant = family_variant
    fields_ = {"kernel_family": family, "direct_variant": variant}
    fields_.update(
        {name: getattr(identity, name) for name in _DIRECT_CAPABILITY_FIELDS}
    )
    fields_.update(
        aot_dgrad_direct_fields_from_knobs(
            variant, json.loads(identity.knobs) if identity.knobs else {}
        )
    )
    return fields_


@dataclass(frozen=True)
class Gfx950ConvDgradAotSpec:
    """One shape-generic backward-data instance.

    Booleans are ints (0/1), so every field is a descriptor ``int`` or ``string``.
    The leading fields are the implicit-GEMM identity's (``DgradConvSpec``'s tuning,
    ``lds_k_outer`` and ``max_sub_gemms``). The trailing ones describe a direct or
    helper instance: ``kernel_family``, ``direct_variant``, the compiled filter,
    stride, padding, dilation and channel counts, and the variant's knobs
    (:data:`AOT_DGRAD_DIRECT_KNOBS`). A direct or helper instance holds the direct
    identity's placeholders in the implicit-GEMM fields; an implicit-GEMM instance
    leaves every trailing field at its KMD default.
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
    lds_k_outer: int
    max_sub_gemms: int
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
    cpg: int = 0
    kpg: int = 0
    block_w: int = 0
    block_waves: int = 0
    block_h: int = 0
    block_q: int = 0
    block_groups: int = 0
    fold_k32: int = 0
    waves_k: int = 0
    waves_q: int = 0
    persistent_grid: int = 0
    runtime_k_loop: int = 0

    # ---- family ----

    @property
    def is_direct(self) -> bool:
        """A direct conv kernel or a helper (no implicit-GEMM tile)."""
        return self.kernel_family != KERNEL_FAMILY_IMPLICIT_GEMM

    @property
    def is_helper(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_DGRAD_HELPER

    @property
    def is_mfma(self) -> bool:
        return (
            self.kernel_family == KERNEL_FAMILY_DIRECT
            and self.direct_variant == DIRECT_VARIANT_MFMA
        )

    @property
    def is_depthwise(self) -> bool:
        return (
            self.kernel_family == KERNEL_FAMILY_DIRECT
            and self.direct_variant == DIRECT_VARIANT_DEPTHWISE
        )

    @property
    def algorithm(self) -> str:
        """The rocKE ``KernelIdentity.algorithm`` this instance was packaged from."""
        if not self.is_direct:
            return "implicit_gemm"
        return _ALGORITHM_OF_VARIANT[self.direct_variant]

    def direct_knobs(self) -> dict:
        """The identity's knobs in rocKE's spelling: the inverse of
        :func:`aot_dgrad_direct_fields_from_knobs`."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct knobs")
        knobs = {}
        for key, field_name in AOT_DGRAD_DIRECT_KNOBS[self.direct_variant].items():
            value = getattr(self, field_name)
            knobs[key] = bool(value) if field_name in _BOOL_KNOBS else value
        return knobs

    def kernel_symbol_prefix(self) -> str:
        # rocKE appends the probe problem and tuning to this (not for the helpers,
        # whose symbol is exactly this); the instance id alone makes it unique.
        if self.is_helper:
            return f"hkp_conv_dgrad_helper_aot_gfx950_{self.instance_id[:24]}"
        if self.is_direct:
            return f"hkp_conv_dgrad_dw_aot_gfx950_{self.instance_id[:24]}"
        return f"hkp_conv_dgrad_aot_gfx950_{self.instance_id[:24]}"

    # ---- the sweep's build ----

    def probe_problem(self) -> ConvProblem | _direct.DirectConvProblem:
        """The problem rocKE's sweep builds this instance against.

        Implicit GEMM mirrors ``kernel_sweep._probe_problem("dgrad", caps)`` with
        ``caps = {grouped, tilde: True}``: 16x16, 3x3, stride 2 (the tilde kernel),
        pad 1, 64 channels per group, two groups when grouped. A direct instance
        mirrors ``direct_kernel_sweep.probe_problem``: 64x64, 256 depthwise groups,
        64 groups for the grouped MFMA rows and 1 for the dense ones. A helper is
        built against the probe of the MFMA row that pulls it in (its identity zeroes
        stride and padding; the row has "same" padding and stride 1). The emitted code
        depends only on the capability axes, so this is what makes the packaged
        binary the one the sweep benchmarked.
        """
        if self.is_direct:
            cpg, kpg = self.cpg, self.kpg
            if (cpg, kpg) in _DENSE_MFMA_CHANNELS:
                groups = 1
            else:
                groups = 256 if cpg == 1 else 64
            pad, stride = self.pad_h, self.stride_h
            if self.is_helper:
                pad, stride = (self.filter_h - 1) // 2, 1
            return _direct.DirectConvProblem(
                N=1,
                H=_DIRECT_PROBE_HW,
                W=_DIRECT_PROBE_HW,
                groups=groups,
                cpg=cpg,
                kpg=kpg,
                KH=self.filter_h,
                KW=self.filter_w,
                PAD=pad,
                stride=stride,
                dtype=self.dtype,
                dil_h=self.dilation_h,
                dil_w=self.dilation_w,
            )
        groups = 2 if self.grouped else 1
        return ConvProblem(
            N=1,
            Hi=16,
            Wi=16,
            C=64 * groups,
            K=64 * groups,
            Y=3,
            X=3,
            sH=2,
            sW=2,
            dH=1,
            dW=1,
            pH=1,
            pW=1,
            groups=groups,
        )

    def to_instance_spec(
        self, problem: ConvProblem | None = None
    ) -> _dgrad.DgradConvSpec:
        """The rocKE builder spec, as ``kernel_sweep.build_kernel`` makes it (on the
        probe problem unless ``problem`` is given)."""
        if self.is_direct:
            raise ValueError("a direct instance has no implicit-GEMM spec")
        return _dgrad.DgradConvSpec(
            problem=self.probe_problem() if problem is None else problem,
            name=self.kernel_symbol_prefix(),
            data=ConvDataSpec(
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
            lds_k_outer=bool(self.lds_k_outer),
            max_sub_gemms=self.max_sub_gemms,
        )

    def rocke_direct_spec(self, problem: _direct.DirectConvProblem):
        """The rocKE kernel spec of this direct or helper instance on ``problem``,
        as ``direct_kernel_sweep.make_spec`` makes it; only the name differs (the
        helpers have no name field: :func:`build_gfx950_conv_dgrad_aot` renames
        their kernel). For the MFMA pipeline this is its compute pass, a forward
        ``DirectConvSpec`` on the transposed problem."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        knobs = self.direct_knobs()
        variant = self.direct_variant
        if variant == DIRECT_VARIANT_DEPTHWISE:
            return _direct.DirectDepthwiseDgradStreamSpec(
                problem=problem, name=self.kernel_symbol_prefix(), **knobs
            )
        if variant == DIRECT_VARIANT_MFMA:
            fold_k32 = knobs.pop("fold_k32")
            spec = _direct.make_dgrad_fprop_spec(problem, **knobs)
            if fold_k32:
                spec = replace(spec, fold_k32=fold_k32)
            return replace(spec, name=self.kernel_symbol_prefix())
        if variant == HELPER_VARIANT_TRANSPOSE:
            return _direct.DirectTransposeWeightsDgradSpec(problem=problem)
        if variant == HELPER_VARIANT_REORGANIZE:
            return _direct.DirectReorganizeWeightsSpec(problem=problem, **knobs)
        raise ValueError(f"unknown direct variant {variant!r}")

    def to_direct_spec(self):
        """The rocKE builder spec for the sweep's probe problem."""
        return self.rocke_direct_spec(self.probe_problem())

    def kernel_name(self) -> str:
        """The launch symbol the packaged kernel carries."""
        if self.is_helper:
            return self.kernel_symbol_prefix()
        if self.is_direct:
            return self.to_direct_spec().kernel_name()
        return self.to_instance_spec().kernel_name()

    # ---- launch mirrors ----

    def block(self) -> tuple[int, int, int]:
        """Workgroup size; for the MFMA pipeline, its compute pass's."""
        if self.is_helper:
            return 64, 1, 1
        if self.is_depthwise:
            return self.block_waves * self.wave_size, 1, 1
        if self.is_mfma:
            return (
                self.block_groups * self.waves_q * self.waves_k * self.wave_size,
                1,
                1,
            )
        return self.warp_m * self.warp_n * self.wave_size, 1, 1

    def sub_gemms(self, problem: Gfx950ConvDgradProblem) -> list:
        """The non-empty tilde sub-GEMMs of ``problem`` for this implicit-GEMM tile."""
        if self.is_direct:
            raise ValueError("only an implicit-GEMM instance has sub-GEMMs")
        p = problem.to_problem()
        return _dgrad.enumerate_sub_gemms(
            p, _dgrad.compute_tilde(p), self.tile_m, self.tile_n, tile_k=self.tile_k
        )

    def sub_gemm_buffer(self, problem: Gfx950ConvDgradProblem) -> list[int]:
        """The ``sub_gemm_buf`` table of ``problem`` (signed i32 values): the
        22-field records, then one 8-field magic entry per record."""
        return _dgrad.pack_sub_gemm_buffer(
            self.sub_gemms(problem), self.tile_m, self.tile_n
        )

    def workspace_bytes(self, problem: Gfx950ConvDgradProblem) -> int:
        """Device scratch the launch needs: the record table for implicit GEMM, the
        transformed weights for the MFMA pipeline, nothing otherwise."""
        if self.is_mfma:
            return gfx950_conv_dgrad_aot_mfma_plan(self, problem)["workspace_bytes"]
        if self.is_direct:
            return 0
        return (
            (SUB_GEMM_RECORD_FIELDS + SUB_GEMM_MAGIC_FIELDS)
            * len(self.sub_gemms(problem))
            * 4
        )

    def grid(self, problem: Gfx950ConvDgradProblem) -> tuple[int, int, int]:
        """Launch grid for ``problem``. Implicit GEMM: ``(flat tiles over the
        sub-GEMMs, groups, 1)``. Depthwise: rocKE's direct launch geometry. MFMA: its
        compute pass's (see :func:`gfx950_conv_dgrad_aot_mfma_plan` for every step)."""
        if self.is_helper:
            raise ValueError("a helper is launched by its MFMA pipeline's plan")
        if self.is_mfma:
            return gfx950_conv_dgrad_aot_mfma_plan(self, problem)["steps"][-1]["grid"]
        if self.is_depthwise:
            spec = self.rocke_direct_spec(problem.to_direct_problem())
            return _direct.direct_launch_geometry(spec)[0]
        flat = sum(
            _ceil_div(sg.gemm_m, self.tile_m) * _ceil_div(sg.gemm_n, self.tile_n)
            for sg in self.sub_gemms(problem)
        )
        return flat, problem.groups, 1

    def launch_signature(self) -> list[dict]:
        """The ABI in ``kernels.common.conv_abi`` order: the 44-argument dgrad block
        for implicit GEMM, the 20-argument direct dgrad block for depthwise, the
        forward direct block for the MFMA compute pass, the 4-argument transform
        block for a helper."""
        if self.is_helper:
            return _transform_signature(self.dtype)
        if self.is_mfma:
            return conv_direct_args_signature(self.dtype, direction="fwd")
        if self.is_depthwise:
            return conv_direct_args_signature(self.dtype, direction="dgrad")
        return conv_args_signature(self.dtype, direction="dgrad")

    def launch_values(
        self,
        problem: Gfx950ConvDgradProblem,
        dy_ptr: int,
        w_ptr: int,
        dx_ptr: int,
        dy_bytes: int,
        w_bytes: int,
        dx_bytes: int,
        sub_gemm_buf: int = 0,
    ) -> dict[str, int]:
        """Every kernel argument of the single-launch families for ``problem``, from
        rocKE's ``ConvArgs``; ``sub_gemm_buf`` is the device address of the packed
        record table (implicit GEMM only). The MFMA pipeline's per-step values come
        from :func:`gfx950_conv_dgrad_aot_mfma_plan`."""
        if self.is_mfma or self.is_helper:
            raise ValueError("the MFMA pipeline launches through its plan")
        if self.is_depthwise:
            return ConvArgs.from_problem(
                problem.to_direct_problem(), direction="dgrad"
            ).to_launch_values(dy_ptr, w_ptr, dx_ptr, dy_bytes, w_bytes, dx_bytes)
        return ConvArgs.from_problem(
            problem.to_problem(),
            direction="dgrad",
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        ).to_launch_values(
            dy_ptr,
            w_ptr,
            dx_ptr,
            dy_bytes,
            w_bytes,
            dx_bytes,
            sub_gemm_buf=sub_gemm_buf,
            num_sub_gemms=len(self.sub_gemms(problem)),
        )

    def needs_zeroed_dx(self, problem: Gfx950ConvDgradProblem) -> bool:
        """Whether dX must be zero-filled before the launch: a strided implicit-GEMM
        problem leaves the dX rows and columns no tap reaches unwritten. The direct
        kernels write every dX element."""
        return not self.is_direct and (problem.sH > 1 or problem.sW > 1)


def _transform_signature(dtype: str) -> list[dict]:
    """The weight transforms' launch signature (``direct_kernel_sweep``'s)."""
    elem = {"fp16": "f16", "bf16": "bf16"}[dtype]
    return [
        {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
        {"name": "A_bytes", "type": "i32", "size_bytes": 4},
        {"name": "D_bytes", "type": "i32", "size_bytes": 4},
    ]


def gfx950_conv_dgrad_aot_helper_fields(
    spec: Gfx950ConvDgradAotSpec,
) -> list[dict]:
    """The helper instances an MFMA instance's pipeline launches, in launch order,
    as the KMD field values that identify each in the catalog (with the instance's
    dtype): the transpose always, the reorganize when ``waves_k > 1`` or
    ``runtime_k_loop`` (keyed on ``fold_k32``)."""
    if not spec.is_mfma:
        raise ValueError("only an MFMA instance has helpers")
    common = {
        "dtype": spec.dtype,
        "kernel_family": KERNEL_FAMILY_DGRAD_HELPER,
        "filter_h": spec.filter_h,
        "filter_w": spec.filter_w,
        "stride_h": 0,
        "stride_w": 0,
        "pad_h": 0,
        "pad_w": 0,
        "dilation_h": spec.dilation_h,
        "dilation_w": spec.dilation_w,
        "cpg": spec.cpg,
        "kpg": spec.kpg,
    }
    helpers = [dict(common, direct_variant=HELPER_VARIANT_TRANSPOSE, fold_k32=0)]
    if _uses_coalesced_weights(spec):
        helpers.append(
            dict(
                common, direct_variant=HELPER_VARIANT_REORGANIZE, fold_k32=spec.fold_k32
            )
        )
    return helpers


def _uses_coalesced_weights(spec: Gfx950ConvDgradAotSpec) -> bool:
    """``direct_kernel_sweep._uses_coalesced_weights``."""
    return spec.waves_k > 1 or bool(spec.runtime_k_loop)


def gfx950_conv_dgrad_aot_mfma_plan(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> dict:
    """The launch plan of an MFMA instance on ``problem``: ``direct_kernel_sweep
    .plan_for``'s steps with the scratch laid out in one workspace.

    Returns ``{"steps": [...], "workspace_bytes": int, "workspaces": {name:
    (offset, bytes)}}``. Each step is ``{"kernel": "transpose" | "reorganize" |
    "compute", "grid", "block", "buffers": (A, B, D), "signature"}`` with buffer
    names ``dy``, ``w``, ``dx``, ``ws_t``, ``ws_coa`` (B is ``None`` for the
    transforms). ``ws_t`` starts at offset 0 and ``ws_coa`` at ``ws_t``'s size
    rounded up to :data:`MFMA_WORKSPACE_ALIGNMENT`. Values come from
    :func:`gfx950_conv_dgrad_aot_mfma_launch_values`.
    """
    if not spec.is_mfma:
        raise ValueError("not an MFMA instance")
    direct = problem.to_direct_problem()
    use_coalesced = _uses_coalesced_weights(spec)
    steps = []
    helpers = [(HELPER_VARIANT_TRANSPOSE, {}, "w", "ws_t")]
    if use_coalesced:
        helpers.append(
            (
                HELPER_VARIANT_REORGANIZE,
                {"fold_k32": bool(spec.fold_k32)},
                "ws_t",
                "ws_coa",
            )
        )
    for variant, knobs, src, dst in helpers:
        if variant == HELPER_VARIANT_TRANSPOSE:
            helper_spec = _direct.DirectTransposeWeightsDgradSpec(problem=direct)
        else:
            helper_spec = _direct.DirectReorganizeWeightsSpec(problem=direct, **knobs)
        grid, block = _direct.direct_launch_geometry(helper_spec)
        steps.append(
            {
                "kernel": variant,
                "grid": grid,
                "block": block,
                "buffers": (src, None, dst),
                "signature": _transform_signature(spec.dtype),
            }
        )
    compute = spec.rocke_direct_spec(direct)
    grid, block = _direct.direct_launch_geometry(compute)
    weights = "ws_coa" if use_coalesced else "ws_t"
    steps.append(
        {
            "kernel": "compute",
            "grid": grid,
            "block": block,
            "buffers": ("dy", weights, "dx"),
            "signature": conv_direct_args_signature(spec.dtype, direction="fwd"),
        }
    )
    ws_t = _direct.direct_dgrad_workspace_bytes(direct)
    workspaces = {"ws_t": (0, ws_t)}
    total = ws_t
    if use_coalesced:
        ws_coa = _direct.direct_dgrad_coalesced_workspace_bytes(
            direct, fold_k32=bool(spec.fold_k32)
        )
        offset = _align_up(ws_t, MFMA_WORKSPACE_ALIGNMENT)
        workspaces["ws_coa"] = (offset, ws_coa)
        total = offset + ws_coa
    return {"steps": steps, "workspace_bytes": total, "workspaces": workspaces}


def gfx950_conv_dgrad_aot_mfma_launch_values(
    spec: Gfx950ConvDgradAotSpec,
    problem: Gfx950ConvDgradProblem,
    step: dict,
    ptrs: dict,
    sizes: dict,
) -> dict[str, int]:
    """Kernel arguments of one plan step, its buffers bound to ``ptrs``/``sizes``
    (``direct_kernel_sweep.launch_values``). The compute pass takes the forward
    direct ABI on the transposed problem (``H = Ho, W = Wo, cpg' = kpg, kpg' = cpg,
    PAD' = KH - 1 - PAD``)."""
    a, b, d = step["buffers"]
    if step["kernel"] != "compute":
        return {"A": ptrs[a], "D": ptrs[d], "A_bytes": sizes[a], "D_bytes": sizes[d]}
    compute = spec.rocke_direct_spec(problem.to_direct_problem())
    return ConvArgs.from_problem(compute.problem, direction="fwd").to_launch_values(
        ptrs[a], ptrs[b], ptrs[d], sizes[a], sizes[b], sizes[d]
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _direct_validate(spec: Gfx950ConvDgradAotSpec, rocke_spec, arch: str):
    """rocKE's validator for ``rocke_spec``, as ``direct_kernel_sweep.validate_spec``
    runs it (the helpers have none)."""
    try:
        if spec.is_depthwise:
            return _direct.is_valid_depthwise_dgrad_stream_spec(rocke_spec, arch=arch)
        if spec.is_mfma:
            rocke_spec.validate()
            return _direct.is_valid_spec(rocke_spec, arch=arch)
        if spec.is_helper:
            return True, "ok"
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return False, f"unknown direct variant {spec.direct_variant!r}"


def _mfma_knobs_reason(spec: Gfx950ConvDgradAotSpec, kpg: int) -> str:
    """``direct_kernel_sweep._mfma_knobs_reason`` for a problem with ``kpg``."""
    if spec.fold_k32:
        if kpg % 32:
            return f"fold_k32 needs kpg % 32 == 0 (got kpg={kpg})"
        n_atoms = kpg // 32
    else:
        n_atoms = (kpg + 15) // 16
    if n_atoms == 0 or n_atoms % spec.waves_k:
        return f"{n_atoms} K-atoms do not split over waves_k={spec.waves_k}"
    return ""


#: The refusal :func:`_coalesced_groups_reason` gives.
COALESCED_GROUPS_REASON = (
    "the reorganize helper mis-addresses groups > 1 "
    "(rocKE build_direct_reorganize_weights), so an MFMA instance with "
    "waves_k > 1 or runtime_k_loop serves groups == 1 only"
)


def _coalesced_groups_reason(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> str:
    """Why an MFMA instance whose compute reads the reorganized (coalesced) weights
    cannot serve ``problem``, or ``""``.

    rocKE's ``build_direct_reorganize_weights`` adds the group offset to the
    per-group K axis of the transposed weights a second time, so the coalesced
    weights are wrong for every group but the first. rocKE's own offer admits these
    problems; this adapter refuses them until the helper is fixed.
    """
    if _uses_coalesced_weights(spec) and problem.groups != 1:
        return COALESCED_GROUPS_REASON
    return ""


def _direct_error(spec: Gfx950ConvDgradAotSpec) -> str:
    """The direct and helper instance contract, before rocKE's validators."""
    for name, value in _DIRECT_PLACEHOLDERS.items():
        actual = getattr(spec, name)
        if actual != value or type(actual) is not type(value):
            return f"a direct instance must set {name} to {value!r}, got {actual!r}"
    if spec.wave_size != _WAVE_SIZE:
        return f"wave_size must be {_WAVE_SIZE} on {_ARCH}"
    expected = _DIRECT_VARIANTS if not spec.is_helper else _HELPER_VARIANTS
    if spec.direct_variant not in expected:
        return (
            f"direct_variant must be one of {expected} for kernel_family "
            f"{spec.kernel_family}, got {spec.direct_variant!r}"
        )
    for name in _DIRECT_CAPABILITY_FIELDS:
        value = getattr(spec, name)
        if type(value) is not int or not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    for name in ("filter_h", "filter_w", "cpg", "kpg"):
        if getattr(spec, name) == 0:
            return f"{name} must be positive"
    if spec.filter_h != spec.filter_w:
        return "the direct sweep builds square filters only"
    if spec.dilation_h != 1 or spec.dilation_w != 1:
        return (
            "the direct dgrad kernels have no dilation (dilation_h == dilation_w == 1)"
        )
    if spec.is_helper:
        # The helper identity zeroes stride and padding (its code reads neither).
        if (spec.stride_h, spec.stride_w, spec.pad_h, spec.pad_w) != (0, 0, 0, 0):
            return "a helper instance records stride and padding as 0"
    else:
        if spec.stride_h == 0 or spec.stride_w == 0:
            return "stride_h/stride_w must be positive"
        if spec.stride_h != spec.stride_w or spec.pad_h != spec.pad_w:
            return "the direct dgrad kernels compile one stride and one padding"
    if spec.is_depthwise and (spec.cpg, spec.kpg) != (1, 1):
        return "a depthwise dgrad instance compiles cpg == kpg == 1"
    used = set(AOT_DGRAD_DIRECT_KNOBS[spec.direct_variant].values())
    for name, default in _KNOB_DEFAULTS.items():
        value = getattr(spec, name)
        if type(value) is not int:
            return f"{name} must be an integer, got {value!r}"
        if name not in used:
            if value != default:
                return (
                    f"the {spec.direct_variant} variant does not use {name}; it must "
                    f"be {default!r}, got {value!r}"
                )
        elif name in _BOOL_KNOBS:
            if value not in (0, 1):
                return f"{name} must be 0 or 1"
        elif not 0 < value <= _INT32_MAX:
            return f"{name} must be a positive signed 32-bit integer"
    if spec.is_mfma:
        why = _mfma_knobs_reason(spec, spec.kpg)
        if why:
            return why
    return ""


def supports_gfx950_conv_dgrad_aot(
    spec: Gfx950ConvDgradAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance; rocKE's validators decide last."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx950ConvDgradAotSpec):
        return False, "expected Gfx950ConvDgradAotSpec"
    if spec.dtype not in _DTYPES:
        return False, "the packaged dgrad contract supports fp16 or bf16 only"
    if spec.layout != "NHWC":
        return False, "the packaged dgrad contract requires dense NHWC storage"
    if (
        type(spec.instance_id) is not str
        or len(spec.instance_id) != _INSTANCE_ID_LENGTH
        or any(c not in "0123456789abcdef" for c in spec.instance_id)
    ):
        return False, "instance_id must be a KernelIdentity.stable_hash() (40 hex)"
    if (
        type(spec.kernel_family) is not int
        or spec.kernel_family not in _KERNEL_FAMILIES
    ):
        return False, (
            f"kernel_family must be one of {_KERNEL_FAMILIES} (implicit GEMM, direct, "
            f"MFMA helper), got {spec.kernel_family!r}"
        )
    if spec.is_direct:
        error = _direct_error(spec)
        if error:
            return False, error
        try:
            return _direct_validate(spec, spec.to_direct_spec(), arch)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            return False, str(exc)
    for name, default in _FAMILY_DEFAULTS.items():
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
        "max_sub_gemms",
    ):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return False, f"{name} must be a positive signed 32-bit integer"
    if spec.wave_size != _WAVE_SIZE:
        return False, f"wave_size must be {_WAVE_SIZE} on {_ARCH}"
    if spec.pipeline not in _PIPELINES:
        return False, (
            f"the packaged dgrad contract supports pipelines {_PIPELINES}, "
            f"got {spec.pipeline!r}"
        )
    if spec.epilogue not in _EPILOGUES:
        return False, f"epilogue must be one of {_EPILOGUES}"
    for name in ("vector_size_a", "vector_size_b", "vector_size_c"):
        if getattr(spec, name) not in _VECTOR_WIDTHS:
            return False, f"{name} must be one of {_VECTOR_WIDTHS}"
    for name in ("lds_k_outer", "grouped"):
        value = getattr(spec, name)
        if type(value) is not int or value not in (0, 1):
            return False, f"{name} must be 0 or 1"
    try:
        instance = spec.to_instance_spec()
        instance.validate()
        ok, reason = _dgrad.is_valid_dgrad_spec(instance, arch=arch)
        if not ok:
            return False, reason
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, "ok"


# ---------------------------------------------------------------------------
# Served problems
# ---------------------------------------------------------------------------


def _implicit_serves(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[bool, str]:
    """The dgrad branch of ``KernelCache.supports_problem`` plus the launch bounds."""
    cpg, kpg = problem.cpg, problem.kpg
    for name, width, extent, label in (
        ("vector_size_a", spec.vector_size_a, kpg, "kpg"),
        ("vector_size_b", spec.vector_size_b, cpg, "cpg"),
        ("vector_size_c", spec.vector_size_c, cpg, "cpg"),
    ):
        if extent % width:
            return False, f"{label}={extent} not divisible by {name}={width}"
    if problem.groups > 1 and not spec.grouped:
        return False, "kernel is ungrouped, problem is grouped"
    n_sub = len(spec.sub_gemms(problem))
    if n_sub == 0:
        return False, "the problem decomposes into no non-empty tilde sub-GEMM"
    if n_sub > spec.max_sub_gemms:
        return False, (
            f"problem needs {n_sub} tilde sub-GEMMs, kernel's record table "
            f"supports {spec.max_sub_gemms} (max_sub_gemms)"
        )
    if problem.groups > 1 and cpg == 1:
        return False, (
            "depthwise dgrad (cpg == 1) is not supported by the implicit-GEMM "
            "grouped path"
        )
    # groups <= 65535 (grid y) is a problem check. Grid x is the flat tile total.
    flat, _, _ = spec.grid(problem)
    if flat > _INT32_MAX or flat * spec.block()[0] > _MAX_DISPATCH_ITEMS:
        return False, f"flat dgrad grid {flat} does not fit one dispatch"
    if problem.N * problem.Hi * problem.Wi >= 1 << 31:
        return False, "gemm_m = N*Hi*Wi must stay below 2^31 (magic division)"
    return True, "ok"


def _direct_serves(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[bool, str]:
    """rocKE's offer for a direct instance (``KernelCache.supports_problem`` then
    ``plan_for``), plus the launch bounds rocKE does not check."""
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
    if (problem.cpg, problem.kpg) != (spec.cpg, spec.kpg):
        return False, (
            f"direct kernel compiled for cpg, kpg = {(spec.cpg, spec.kpg)}, problem "
            f"has {(problem.cpg, problem.kpg)}"
        )
    direct = problem.to_direct_problem()
    if spec.is_mfma:
        why = _mfma_knobs_reason(spec, problem.kpg)
        if why:
            return False, why
        why = _coalesced_groups_reason(spec, problem)
        if why:
            return False, why
    try:
        rocke_spec = spec.rocke_direct_spec(direct)
    except (AssertionError, TypeError, ValueError) as exc:
        return False, str(exc)
    ok, why = _direct_validate(spec, rocke_spec, _ARCH)
    if not ok:
        return False, why
    if spec.is_depthwise:
        grid = _direct.direct_launch_geometry(rocke_spec)[0]
        if grid[0] > _INT32_MAX:
            return False, f"direct grid x = {grid[0]} exceeds {_INT32_MAX}"
        for axis, extent in zip("yz", grid[1:]):
            if extent > _MAX_GRID_DIM:
                return False, f"direct grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
        return True, "ok"
    plan = gfx950_conv_dgrad_aot_mfma_plan(spec, problem)
    for step in plan["steps"]:
        grid = step["grid"]
        # The compute grid is held to 65535 on every axis, as for forward; the
        # transforms' flat x axes to a 32-bit dispatch.
        x_cap = _MAX_GRID_DIM if step["kernel"] == "compute" else _INT32_MAX
        if grid[0] > x_cap or grid[0] * step["block"][0] > _MAX_DISPATCH_ITEMS:
            return False, f"{step['kernel']} grid x = {grid[0]} exceeds {x_cap}"
        for axis, extent in zip("yz", grid[1:]):
            if extent > _MAX_GRID_DIM:
                return False, (
                    f"{step['kernel']} grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
                )
    for name, (_, size) in plan["workspaces"].items():
        if size > _INT32_MAX:
            return False, f"{name} workspace exceeds the signed 32-bit kernel ABI"
    return True, "ok"


def gfx950_conv_dgrad_aot_serves(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[bool, str]:
    """Whether instance ``spec`` computes ``problem``.

    Implicit GEMM: the dgrad subset of ``KernelCache.supports_problem`` -- the A
    (dY) width divides K/groups, the B (W) and D (dX) widths divide C/groups, a
    grouped problem needs a grouped binary, the problem's non-empty tilde sub-GEMMs
    fit ``max_sub_gemms``, depthwise (groups > 1, C/groups == 1) is refused, groups
    fits grid y -- plus the launch bounds: the flat tile grid fits one dispatch and
    ``N*Hi*Wi`` stays below 2^31.

    Direct (depthwise and MFMA): the compiled filter, stride, padding, dilation and
    channel counts are the problem's, the kernel's validator accepts the spec rebuilt
    on the problem (for MFMA also ``_mfma_knobs_reason`` and ``groups %
    block_groups``; an MFMA instance using the reorganize helper, ``waves_k > 1``
    or ``runtime_k_loop``, serves ``groups == 1`` only), and every launch fits: grid y and z 65535 (and the MFMA compute
    pass's x), the transforms' x a 32-bit dispatch, each workspace a 32-bit size.

    A helper serves nothing on its own.
    """
    error = _problem_error(problem)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_helper:
        return False, "a helper kernel is launched only by its MFMA pipeline"
    if spec.is_direct:
        return _direct_serves(spec, problem)
    return _implicit_serves(spec, problem)


def gfx950_conv_dgrad_aot_runtime_candidates(
    problem: Gfx950ConvDgradProblem, specs: Sequence[Gfx950ConvDgradAotSpec]
) -> list[Gfx950ConvDgradAotSpec]:
    """The instances of ``specs`` hipkernel:Gfx950ConvDgradAot offers for
    ``problem``: every serving one, in the order of ``specs``. The catalog is small,
    so there is no widest-vector filter; the ranker chooses."""
    return [spec for spec in specs if gfx950_conv_dgrad_aot_serves(spec, problem)[0]]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


_DIRECT_BUILDERS = {
    DIRECT_VARIANT_DEPTHWISE: _direct.build_direct_depthwise_dgrad_streaming,
    DIRECT_VARIANT_MFMA: _direct.build_direct_conv,
    HELPER_VARIANT_TRANSPOSE: _direct.build_direct_transpose_weights_dgrad,
    HELPER_VARIANT_REORGANIZE: _direct.build_direct_reorganize_weights,
}


def build_gfx950_conv_dgrad_aot(
    spec: Gfx950ConvDgradAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Dispatches on the family as the sweeps do: implicit GEMM through
    ``build_implicit_gemm_conv_dgrad`` (``kernel_sweep.build_kernel``), the direct
    and helper kernels through ``direct_kernel_sweep._build_kernel``'s builders. A
    helper's kernel is renamed to :meth:`Gfx950ConvDgradAotSpec.kernel_symbol_prefix`
    (its builder names it after the probe problem, which fp16 and bf16 share).
    """
    ok, reason = supports_gfx950_conv_dgrad_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT dgrad instance: {reason}")
    if spec.is_direct:
        kernel = _DIRECT_BUILDERS[spec.direct_variant](spec.to_direct_spec(), arch=arch)
        if spec.is_helper:
            kernel = replace(kernel, name=spec.kernel_symbol_prefix())
        return kernel
    return _dgrad.build_implicit_gemm_conv_dgrad(spec.to_instance_spec(), arch=arch)


# ---------------------------------------------------------------------------
# Profile hooks
# ---------------------------------------------------------------------------


def gfx950_conv_dgrad_aot_problem_for_request(
    request: ConvGroupedRequest,
) -> Gfx950ConvDgradProblem:
    """The problem hipkernel:Gfx950ConvDgradAot would match for ``request``.

    Instances are chosen per problem by capability, so there is no single spec to
    resolve; this is the graph side of that match (the profile's ``dispatch``).
    ``Hi``/``Wi`` are the dX extents. Raises ``ValueError`` outside the graph
    contract.
    """
    if not isinstance(request, ConvGroupedRequest):
        raise TypeError("expected ConvGroupedRequest")
    if request.arch != _ARCH:
        raise ValueError(f"this adapter requires arch={_ARCH}")
    if request.direction != "dgrad":
        raise ValueError("this engine serves backward-data convolution only")
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
        raise ValueError("the packaged dgrad contract supports 2D convolution only")
    if request.vec_size_c is not None:
        raise ValueError("the packaged contract takes no vector-width override")
    problem = Gfx950ConvDgradProblem(
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
    error = _problem_error(problem)
    if error:
        raise ValueError(error)
    return problem


def supports_gfx950_conv_dgrad_aot_problem(
    problem: Gfx950ConvDgradProblem, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """The graph contract of hipkernel:Gfx950ConvDgradAot (the profile's predicate)."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(problem, Gfx950ConvDgradProblem):
        return False, "expected Gfx950ConvDgradProblem"
    error = _problem_error(problem)
    return (False, error) if error else (True, "ok")


def sub_gemm_buffer_for(
    problem: Gfx950ConvDgradProblem, tile_m: int, tile_n: int, tile_k: int
) -> list[int]:
    """The ``sub_gemm_buf`` an implicit-GEMM instance of tile ``(tile_m, tile_n,
    tile_k)`` reads for ``problem``: golden vectors for the native packer."""
    p = problem.to_problem()
    sub_gemms = _dgrad.enumerate_sub_gemms(
        p, _dgrad.compute_tilde(p), tile_m, tile_n, tile_k=tile_k
    )
    return _dgrad.pack_sub_gemm_buffer(sub_gemms, tile_m, tile_n)
