# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 shape-generic AOT backward-data convolution.

This is the dgrad sibling of the ``Gfx950ConvFwdAot*`` pieces of
:mod:`builders.common.convolution_forward`: one flat descriptor spec
(:class:`Gfx950ConvDgradAotSpec`, every field an ``int`` or ``str``) per rocKE
``KernelIdentity`` of the dgrad AOT sweeps, built exactly as the sweeps build it, so
the packaged IR is the swept kernel's byte for byte apart from its symbol name.

The contract is 2-D cross-correlation, NHWK dY, KYXC W, NHWC dX, uniform fp16/bf16,
fp32 accumulation, symmetric padding. Three families are packaged:

* **Implicit GEMM** (``kernel_family`` 0): ``DgradConvSpec``. The kernel reads its
  sub-GEMM decomposition from a host-packed record table (``sub_gemm_buf``,
  ``pack_sub_gemm_buffer``): the tilde decomposition, or for a ``patch`` kernel the one
  GEMM over the filter. A binary may bake its filter, stride and dilation, and the
  dY-halo, tap-outer K loop and patch forms serve only the problems they compute. The
  ABI is the 44-argument dgrad block of ``kernels.common.conv_abi``.
* **Direct** (``kernel_family`` 1), every dgrad pipeline of rocKE's direct sweep:
  ``depthwise_dgrad`` and ``depthwise_dgrad_win`` (the direct dgrad ABI),
  ``grouped_dgrad_fused`` and ``grouped_dgrad_4c`` (one forward kernel over the
  transposed problem reading the original weights), and the MFMA pipelines
  ``grouped_dgrad_mfma`` / ``dgrad_mfma_rtc`` (weight transforms into the workspace,
  then a forward kernel over the transposed problem).
* **MFMA helpers** (``kernel_family`` 2): the weight transforms the MFMA pipelines
  launch. They are packaged so the pipeline's handler can load them, and never serve a
  problem on their own.

:func:`gfx950_conv_dgrad_aot_plan` is the launch of one instance on one problem --
rocKE's own offer (``identity_supports_problem``, ``direct_spec_for_problem``) plus the
launch bounds, then its steps, workspace and dX zero-fill -- and is what the native
pack (``Gfx950ConvDgradGeometry.hpp``) ports; the two are compared by
:meth:`Gfx950ConvDgradPlan.digest`.

``instance_id`` is ``KernelIdentity.stable_hash()``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_implicit_gemm_dgrad as _dgrad
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
from kernels.common.conv_args import ConvArgs
from rocke.core.ir import KernelDef

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1
#: Grid y and z (and every axis of the direct compute grids, as for forward).
_MAX_GRID_DIM = 65535
#: A dispatch's x extent in work-items is 32-bit.
_MAX_DISPATCH_ITEMS = (1 << 32) - 1
_MAX_BLOCK_THREADS = 1024
_WAVE_SIZE = 64
_DTYPES = ("fp16", "bf16")
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_INSTANCE_ID_LENGTH = 40
#: The LLVM flavor the catalog's instances were swept, hashed and measured under
#: (every label of ``gfx950_conv_dgrad_aot.labels.txt`` ends in it).
PACKAGED_LLVM_FLAVOR = "llvm22"

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
DIRECT_VARIANT_DEPTHWISE_WIN = "depthwise_dgrad_win"
DIRECT_VARIANT_MFMA = "grouped_dgrad_mfma"
DIRECT_VARIANT_MFMA_RTC = "dgrad_mfma_rtc"
DIRECT_VARIANT_FUSED = "grouped_dgrad_fused"
DIRECT_VARIANT_4C = "grouped_dgrad_4c"
HELPER_VARIANT_TRANSPOSE = "transpose"
HELPER_VARIANT_REORGANIZE = "reorganize"
HELPER_VARIANT_TRANSPOSE_RTC = "transpose_rtc"
HELPER_VARIANT_REORGANIZE_RTC = "reorganize_rtc"
_DIRECT_VARIANTS = (
    DIRECT_VARIANT_DEPTHWISE,
    DIRECT_VARIANT_DEPTHWISE_WIN,
    DIRECT_VARIANT_MFMA,
    DIRECT_VARIANT_MFMA_RTC,
    DIRECT_VARIANT_FUSED,
    DIRECT_VARIANT_4C,
)
_HELPER_VARIANTS = (
    HELPER_VARIANT_TRANSPOSE,
    HELPER_VARIANT_REORGANIZE,
    HELPER_VARIANT_TRANSPOSE_RTC,
    HELPER_VARIANT_REORGANIZE_RTC,
)
#: The MFMA pipelines, which launch weight transforms before their compute pass.
_MFMA_VARIANTS = (DIRECT_VARIANT_MFMA, DIRECT_VARIANT_MFMA_RTC)

#: Each dgrad direct-sweep algorithm (``direct_kernel_sweep.DIRECT_CAPABILITIES`` and
#: its helper variants) and the ``(kernel_family, direct_variant)`` an instance of it
#: carries.
AOT_DGRAD_DIRECT_ALGORITHMS = {
    "direct_depthwise_dgrad": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DEPTHWISE),
    "direct_depthwise_dgrad_win": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DEPTHWISE_WIN),
    "direct_grouped_dgrad_mfma": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_MFMA),
    "direct_dgrad_mfma_rtc": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_MFMA_RTC),
    "direct_grouped_dgrad_fused": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_FUSED),
    "direct_grouped_dgrad_4c": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_4C),
    "direct_dgrad_transpose_rtc": (
        KERNEL_FAMILY_DGRAD_HELPER,
        HELPER_VARIANT_TRANSPOSE_RTC,
    ),
    "direct_dgrad_reorganize_rtc": (
        KERNEL_FAMILY_DGRAD_HELPER,
        HELPER_VARIANT_REORGANIZE_RTC,
    ),
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
_MFMA_KNOBS = {
    "block_groups": "block_groups",
    "block_h": "block_h",
    "block_q": "block_q",
    "fold_k32": "fold_k32",
    "persistent_grid": "persistent_grid",
    "runtime_k_loop": "runtime_k_loop",
    "waves_k": "waves_k",
    "waves_q": "waves_q",
}
AOT_DGRAD_DIRECT_KNOBS = {
    DIRECT_VARIANT_DEPTHWISE: {"block_w": "block_w", "block_waves": "block_waves"},
    DIRECT_VARIANT_DEPTHWISE_WIN: {
        "block_h": "block_h",
        "block_w": "block_w",
        "block_waves": "block_waves",
        "ch_per_lane": "ch_per_lane",
        "dot2": "dot2",
        "mfma": "mfma",
        "prefetch_rows": "prefetch_rows",
        "w_fold": "w_fold",
    },
    DIRECT_VARIANT_MFMA: _MFMA_KNOBS,
    # The runtime-channel pipeline takes the MFMA knobs plus its own K-loop shape.
    DIRECT_VARIANT_MFMA_RTC: {
        **_MFMA_KNOBS,
        "k_atoms": "k_atoms",
        "m_tiles": "m_tiles",
    },
    DIRECT_VARIANT_FUSED: {
        "block_groups": "block_groups",
        "block_h": "block_h",
        "block_q": "block_q",
        "dgrad_weights_lds": "dgrad_weights_lds",
        "fold_k32": "fold_k32",
        "lds_only_sync": "lds_only_sync",
        "lds_pad": "lds_pad",
        "prefetch_rows": "prefetch_rows",
        "row_exit": "row_exit",
        "stage_out": "stage_out",
        "waves_per_eu": "waves_per_eu",
        "xcd_tiles": "xcd_tiles",
    },
    DIRECT_VARIANT_4C: {
        "block_groups": "block_groups",
        "block_q": "block_q",
        "dgrad_weights_lds": "dgrad_weights_lds",
        "stage_rows": "stage_rows",
    },
    HELPER_VARIANT_TRANSPOSE: {},
    HELPER_VARIANT_REORGANIZE: {"fold_k32": "fold_k32"},
    HELPER_VARIANT_TRANSPOSE_RTC: {},
    HELPER_VARIANT_REORGANIZE_RTC: {"fold_k32": "fold_k32"},
}
#: Knobs whose 0 is a value rather than "not stated": no LDS padding, and the
#: unset occupancy hint.
_ZERO_IS_A_VALUE = ("lds_pad", "waves_per_eu")
#: Knobs rocKE spells as a bool; the descriptor holds 0/1.
_BOOL_KNOBS = (
    "fold_k32",
    "persistent_grid",
    "runtime_k_loop",
    "dot2",
    "mfma",
    "dgrad_weights_lds",
    "lds_only_sync",
    "row_exit",
    "stage_out",
    "xcd_tiles",
    "stage_rows",
)
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
    "ch_per_lane": 0,
    "dot2": 0,
    "mfma": 0,
    "prefetch_rows": 0,
    "w_fold": 0,
    "k_atoms": 0,
    "m_tiles": 0,
    "dgrad_weights_lds": 0,
    "lds_only_sync": 0,
    "lds_pad": 0,
    "row_exit": 0,
    "stage_out": 0,
    "waves_per_eu": 0,
    "xcd_tiles": 0,
    "stage_rows": 0,
}
#: Geometry an implicit-GEMM instance may state, because the tilde kernel can be
#: compiled for one filter, stride and dilation instead of reading them from its
#: sub-GEMM records. 0 keeps the runtime behaviour; a stated value is a promise the
#: matcher enforces (_implicit_serves). Padding, cpg and kpg stay runtime-only.
_IMPLICIT_BAKEABLE = (
    "filter_h",
    "filter_w",
    "stride_h",
    "stride_w",
    "dilation_h",
    "dilation_w",
)
#: Fields appended after the implicit-GEMM ones. An implicit-GEMM instance leaves
#: each at this value (the KMD default), apart from _IMPLICIT_BAKEABLE; a direct or
#: helper instance leaves the knobs its variant does not use at it.
_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    **{name: 0 for name in _DIRECT_CAPABILITY_FIELDS if name not in _IMPLICIT_BAKEABLE},
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

    def tensor_bytes(self, images: int | None = None) -> tuple[int, int, int]:
        """``(dY, W, dX)`` sizes in bytes (16-bit elements), for ``images`` images
        (default the batch)."""
        n = self.N if images is None else images
        return (
            n * self.Ho * self.Wo * self.K * 2,
            self.K * self.Y * self.X * self.cpg * 2,
            n * self.Hi * self.Wi * self.C * 2,
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
    # A batch whose dY or dX exceeds the 32-bit ABI runs as batch slices
    # (gfx950_conv_dgrad_aot_batch_chunks), so one image of each, and the filter,
    # must fit; the whole tensors stay addressable by a 64-bit offset.
    for label, size in zip(("dY", "W", "dX"), problem.tensor_bytes(1)):
        if size > _INT32_MAX:
            return (
                f"{label} byte count of one image exceeds the signed 32-bit kernel ABI"
            )
    if max(problem.tensor_bytes()) >= 1 << 62:
        return "a tensor exceeds 2^62 bytes"
    return ""


# ---------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------


def aot_dgrad_direct_fields_from_knobs(variant: str, knobs: dict) -> dict:
    """The instance fields a direct or helper identity's ``knobs`` set.

    A variant's knobs are what it *may* set, not what it must: the windowed
    depthwise pipeline alone has a dot2 and an MFMA mode that take different ones,
    and the sweep states only the mode's. An unlisted knob is still refused -- it
    would otherwise be dropped silently and the instance would be packaged for a
    kernel nobody asked for. An omitted one keeps its _KNOB_DEFAULTS value.

    Raises ``ValueError`` for a knob the variant does not take, or a bool knob that
    is not a bool.
    """
    mapping = AOT_DGRAD_DIRECT_KNOBS[variant]
    unknown = sorted(set(knobs) - set(mapping))
    if unknown:
        raise ValueError(
            f"knobs {unknown} are not {variant}'s (it takes {sorted(mapping)})"
        )
    values = {}
    for key, field_name in mapping.items():
        if key not in knobs:
            continue
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
    # Implicit-GEMM dY-halo and K-decode tuning. Each selects emission, so two
    # instances differing only here are different binaries.
    dy_halo: int = 0
    dy_halo_setprio: int = 0
    dy_halo_kouter_pad: int = 0
    halo_max_wo: int = 0
    patch: int = 0
    tap_outer_k: int = 0
    uniform_k_decode: int = 0
    # Windowed depthwise.
    ch_per_lane: int = 0
    dot2: int = 0
    mfma: int = 0
    prefetch_rows: int = 0
    w_fold: int = 0
    # Runtime-channel MFMA.
    k_atoms: int = 0
    m_tiles: int = 0
    # Fused and 4c forward-on-transposed pipelines.
    dgrad_weights_lds: int = 0
    lds_only_sync: int = 0
    lds_pad: int = 0
    row_exit: int = 0
    stage_out: int = 0
    waves_per_eu: int = 0
    xcd_tiles: int = 0
    stage_rows: int = 0

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
        """An MFMA pipeline: a compute pass preceded by weight transforms."""
        return (
            self.kernel_family == KERNEL_FAMILY_DIRECT
            and self.direct_variant in _MFMA_VARIANTS
        )

    @property
    def is_runtime_channels(self) -> bool:
        """Takes the per-group channel counts as kernargs instead of baking them:
        the runtime-channel MFMA pipeline and the two transforms it launches."""
        return self.direct_variant in (
            DIRECT_VARIANT_MFMA_RTC,
            HELPER_VARIANT_TRANSPOSE_RTC,
            HELPER_VARIANT_REORGANIZE_RTC,
        )

    @property
    def is_depthwise(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_DIRECT and self.direct_variant in (
            DIRECT_VARIANT_DEPTHWISE,
            DIRECT_VARIANT_DEPTHWISE_WIN,
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
            if field_name in _BOOL_KNOBS:
                knobs[key] = bool(value)
            elif value or field_name in _ZERO_IS_A_VALUE:
                knobs[key] = value
            # Otherwise the instance did not state it -- a knob its mode does not
            # apply (the windowed depthwise MFMA form takes no block_w) -- so it is
            # left out and rocKE's own default stands. Passing 0 would be a value,
            # and its spec rejects one.
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

    def to_identity(self):
        """The rocKE ``KernelIdentity`` this instance was packaged from.

        Implicit GEMM: every field the packaging does not carry is at its
        KernelIdentity default, which is exactly what supports_gfx950_conv_dgrad_aot
        enforces, so this reconstructs the identity (its ``stable_hash()`` is the
        ``instance_id``). Direct and helper instances: the identity
        ``direct_aot_jobs`` builds, up to the spelling of the knobs JSON (a bool knob
        left at false may or may not have been stated), so it decides what the
        binary serves but is not hashed.

        ``llvm_flavor`` is the catalog's, :data:`PACKAGED_LLVM_FLAVOR`.
        """
        from kernels.conv.kernel_identity import KernelIdentity

        common = {
            "arch": _ARCH,
            "dtype_a": self.dtype,
            "dtype_b": self.dtype,
            "dtype_d": self.dtype,
            "wave_size": self.wave_size,
            "llvm_flavor": PACKAGED_LLVM_FLAVOR,
        }
        if self.is_direct:
            return KernelIdentity(
                direction="direct_dgrad_helper" if self.is_helper else "direct_dgrad",
                algorithm=self.algorithm,
                tile_m=0,
                tile_n=0,
                tile_k=0,
                warp_m=0,
                warp_n=0,
                warp_tile_m=0,
                warp_tile_n=0,
                warp_tile_k=0,
                pipeline="direct",
                epilogue="direct",
                vector_size_a=0,
                vector_size_b=0,
                vector_size_c=0,
                knobs=json.dumps(self.direct_knobs(), sort_keys=True),
                **{name: getattr(self, name) for name in _DIRECT_CAPABILITY_FIELDS},
                **common,
            )
        carried = (
            "tile_m",
            "tile_n",
            "tile_k",
            "warp_m",
            "warp_n",
            "warp_tile_m",
            "warp_tile_n",
            "warp_tile_k",
            "pipeline",
            "epilogue",
            "vector_size_a",
            "vector_size_b",
            "vector_size_c",
            "max_sub_gemms",
            "filter_h",
            "filter_w",
            "stride_h",
            "stride_w",
            "dilation_h",
            "dilation_w",
            "dy_halo",
            "dy_halo_setprio",
            "dy_halo_kouter_pad",
            "halo_max_wo",
        )
        flags = ("lds_k_outer", "grouped", "patch", "tap_outer_k", "uniform_k_decode")
        return KernelIdentity(
            direction="dgrad",
            algorithm="implicit_gemm",
            **{name: getattr(self, name) for name in carried},
            **{name: bool(getattr(self, name)) for name in flags},
            **common,
        )

    def probe_problem(self) -> ConvProblem | _direct.DirectConvProblem:
        """The problem rocKE's sweep builds this instance against.

        Implicit GEMM asks rocKE for it (see below). A direct instance
        mirrors ``direct_kernel_sweep.probe_problem``: 64x64, 256 depthwise groups,
        64 groups for the grouped MFMA rows and 1 for the dense ones. A helper is
        built against the probe of the MFMA row that pulls it in (its identity zeroes
        stride and padding; the row has "same" padding and stride 1). The emitted code
        depends only on the capability axes, so this is what makes the packaged
        binary the one the sweep benchmarked.
        """
        if self.is_direct:
            # rocKE's own probe, not a mirror: it sizes the image for the knobs (an
            # xcd_tiles remap needs eight row bands), takes runtime channel counts
            # (cpg == 0) and derives the runtime-channel MFMA probe from k_atoms and
            # m_tiles. A probe that is merely "big enough" is not good enough -- it
            # has to be the one the sweep built the binary against, or the packaged
            # kernel is not the kernel that was benchmarked.
            from kernels.conv.direct_aot_jobs import DirectCaps, probe_problem

            pad, stride = self.pad_h, self.stride_h
            if self.is_helper:
                pad, stride = (self.filter_h - 1) // 2, 1
            caps = DirectCaps(
                KH=self.filter_h,
                PAD=pad,
                stride=stride,
                cpg=self.cpg,
                kpg=self.kpg,
                dil_h=self.dilation_h,
                dil_w=self.dilation_w,
            )
            return probe_problem(
                caps,
                dtype=self.dtype,
                variant=self.algorithm,
                knobs=self.direct_knobs(),
            )
        # The probe carries the geometry the instance baked, where it baked any: a
        # kernel compiled for stride 1 must be built against a stride-1 problem, or
        # rocKE validates the emission it was never going to run (tap_outer_k is only
        # legal at stride 1 and dilation 1). 0 keeps the runtime probe, whose stride 2
        # is what selects the tilde code path.
        # rocKE's own probe for this identity, not a mirror of it. The probe is not
        # cosmetic: the emitted IR depends on the capability axes, so a probe that
        # does not match them silently builds a binary the cache then offers for
        # shapes it cannot compute. Each axis (tilde stride, patch embedding, a dY
        # halo sized for halo_max_wo) has its own shape, and restating them here is
        # what let this adapter drift behind rocKE in the first place.
        from kernels.conv.aot_jobs import _probe_problem, job_for_identity

        return _probe_problem("dgrad", job_for_identity(self.to_identity()).caps)

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
            dy_halo=self.dy_halo,
            dy_halo_setprio=self.dy_halo_setprio,
            dy_halo_kouter_pad=self.dy_halo_kouter_pad,
            halo_max_wo=self.halo_max_wo,
            patch=bool(self.patch),
            tap_outer_k=bool(self.tap_outer_k),
            uniform_k_decode=bool(self.uniform_k_decode),
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
        if not self.is_helper:
            # rocKE's own factory, the one the direct sweep builds through, rather
            # than a per-variant mirror here: a pipeline that gains a knob or changes
            # its spec stays packageable without a change in this file.
            from kernels.conv.direct_variants import make_direct_spec

            spec = make_direct_spec(
                self.algorithm, problem, knobs, wave_size=self.wave_size
            )
            return replace(spec, name=self.kernel_symbol_prefix())
        # The _rtc transforms are the same kernels reading their channel counts from
        # kernargs, so one binary per filter serves every channel count.
        if variant in (HELPER_VARIANT_TRANSPOSE, HELPER_VARIANT_TRANSPOSE_RTC):
            return _direct.DirectTransposeWeightsDgradSpec(
                problem=problem,
                runtime_channels=variant == HELPER_VARIANT_TRANSPOSE_RTC,
            )
        if variant in (HELPER_VARIANT_REORGANIZE, HELPER_VARIANT_REORGANIZE_RTC):
            return _direct.DirectReorganizeWeightsSpec(
                problem=problem,
                runtime_channels=variant == HELPER_VARIANT_REORGANIZE_RTC,
                **knobs,
            )
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
    # The runtime-channel pipeline launches the _rtc transforms: same passes, reading
    # their channel counts from kernargs.
    transpose, reorganize = (
        (HELPER_VARIANT_TRANSPOSE_RTC, HELPER_VARIANT_REORGANIZE_RTC)
        if spec.is_runtime_channels
        else (HELPER_VARIANT_TRANSPOSE, HELPER_VARIANT_REORGANIZE)
    )
    helpers = [dict(common, direct_variant=transpose, fold_k32=0)]
    if _uses_coalesced_weights(spec):
        helpers.append(dict(common, direct_variant=reorganize, fold_k32=spec.fold_k32))
    return helpers


def _uses_coalesced_weights(spec: Gfx950ConvDgradAotSpec) -> bool:
    """``kernels.conv.direct_variants.direct_uses_coalesced_weights``."""
    return spec.waves_k > 1 or bool(spec.runtime_k_loop)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _direct_validate(spec: Gfx950ConvDgradAotSpec, rocke_spec, arch: str):
    """rocKE's validator for ``rocke_spec``, as ``direct_kernel_sweep.validate_spec``
    runs it (the helpers have none)."""
    if spec.is_helper:
        return True, "ok"
    # rocKE's own validator, keyed on the algorithm, rather than a per-variant
    # mirror: a pipeline whose validity rule changes stays packageable unchanged.
    from kernels.conv.direct_variants import validate_direct_spec

    try:
        return validate_direct_spec(spec.algorithm, rocke_spec, arch)
    except (TypeError, ValueError, ZeroDivisionError, KeyError) as exc:
        return False, str(exc)


def _mfma_knobs_reason(spec: Gfx950ConvDgradAotSpec, kpg: int) -> str:
    """``direct_kernel_sweep._mfma_knobs_reason`` for a problem with ``kpg``.

    The runtime-channel pipeline bakes its K-atom count rather than deriving it
    from a channel count it does not know, so k_atoms answers directly.
    """
    if spec.is_runtime_channels:
        if spec.k_atoms == 0 or spec.k_atoms % spec.waves_k:
            return f"k_atoms={spec.k_atoms} does not split over waves_k={spec.waves_k}"
        return ""
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


def _persistent_cells_reason(rocke_spec) -> str:
    """Why a persistent-grid compute pass cannot cover ``rocke_spec.problem``, or
    ``""``. Its blocks walk at most ``persistent_rounds`` cells each, a bound baked
    into the kernel, and a problem with more cells would lose its tail."""
    if not getattr(rocke_spec, "persistent_grid", False):
        return ""
    cells = 1
    for extent in _direct.direct_conv_grid(rocke_spec):
        cells *= extent
    budget = rocke_spec.persistent_rounds * _direct.DIRECT_PERSISTENT_BLOCKS
    if cells > budget:
        return f"{cells} persistent cells exceed {budget} (persistent_rounds * blocks)"
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
    for name in ("filter_h", "filter_w"):
        if getattr(spec, name) == 0:
            return f"{name} must be positive"
    # cpg == kpg == 0 is rocKE's "runtime channel counts" (DirectCaps): the
    # runtime-channel MFMA pipeline takes them as kernargs and bakes only its
    # channel *tile* counts, so one binary per filter serves every channel count.
    # Either both are runtime or neither is.
    if (spec.cpg == 0) != (spec.kpg == 0):
        return "cpg and kpg are both runtime (0) or both baked"
    if spec.cpg == 0 and not spec.is_runtime_channels:
        return (
            f"only the runtime-channel pipeline takes runtime channel counts; "
            f"{spec.direct_variant} bakes them"
        )
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
        elif not 0 <= value <= _INT32_MAX:
            # 0 is a knob the variant takes and this instance left out, or one whose
            # own zero means something (lds_pad, waves_per_eu). Whether the value
            # makes sense for the kernel is rocKE's validator to say, not this shape
            # check's.
            return f"{name} must be a nonnegative signed 32-bit integer"
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
# The launch plan
# ---------------------------------------------------------------------------

#: The kernel a plan step launches: the instance itself, or one of the helpers of
#: its pack (named by their ``direct_variant``).
KERNEL_SELF = "self"
#: The ABI a plan step launches with.
#: ``implicit_dgrad``: the 44-argument dgrad implicit-GEMM block (``dY, W, dX``, byte
#: sizes, 36 problem values, ``sub_gemm_buf``, ``num_sub_gemms``).
ABI_IMPLICIT = "implicit_dgrad"
#: ``direct_dgrad``: the 20-argument direct dgrad block (the depthwise kernels).
ABI_DIRECT_DGRAD = "direct_dgrad"
#: ``direct_fwd``: the forward direct block, on the transposed problem (the fused,
#: 4c and MFMA compute passes).
ABI_DIRECT_FWD = "direct_fwd"
#: ``transform`` / ``transform_rtc``: a weight transform's ``A, D, A_bytes, D_bytes``,
#: then for the runtime-channel ones the original problem's ``p_cpg, p_kpg``.
ABI_TRANSFORM = "transform"
ABI_TRANSFORM_RTC = "transform_rtc"
#: Buffers a step binds: the three graph tensors and the workspace regions.
BUFFER_DY = "dy"
BUFFER_W = "w"
BUFFER_DX = "dx"
#: The implicit-GEMM record table, uploaded into the workspace before the launch.
BUFFER_SUB_GEMM = "sub_gemm"
#: The MFMA pipelines' transposed and reorganized weights.
BUFFER_WS_T = "ws_t"
BUFFER_WS_COA = "ws_coa"

#: The integer codes :meth:`Gfx950ConvDgradPlan.serialize` writes; the native pack
#: (Gfx950ConvDgradGeometry.hpp) serializes its plan with the same ones, so the two
#: can be compared by digest.
PLAN_KERNEL_CODES = {
    KERNEL_SELF: 0,
    HELPER_VARIANT_TRANSPOSE: 1,
    HELPER_VARIANT_REORGANIZE: 2,
    HELPER_VARIANT_TRANSPOSE_RTC: 3,
    HELPER_VARIANT_REORGANIZE_RTC: 4,
}
PLAN_ABI_CODES = {
    ABI_IMPLICIT: 0,
    ABI_DIRECT_DGRAD: 1,
    ABI_DIRECT_FWD: 2,
    ABI_TRANSFORM: 3,
    ABI_TRANSFORM_RTC: 4,
}
PLAN_BUFFER_CODES = {
    None: -1,
    BUFFER_DY: 0,
    BUFFER_W: 1,
    BUFFER_DX: 2,
    BUFFER_SUB_GEMM: 3,
    BUFFER_WS_T: 4,
    BUFFER_WS_COA: 5,
}

#: The direct compute passes that are ``DirectConvSpec`` forward kernels: their
#: grid is held to 65535 on every axis, as forward's direct kernels are.
_DIRECT_CONV_SPEC_ALGORITHMS = (
    "direct_grouped_dgrad_mfma",
    "direct_dgrad_mfma_rtc",
    "direct_grouped_dgrad_fused",
)
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class Gfx950ConvDgradStep:
    """One kernel launch of a plan.

    ``kernel`` is :data:`KERNEL_SELF` or a helper's ``direct_variant``; ``buffers``
    the ``A``, ``B`` and ``D`` operands (``B`` is ``None`` for a weight transform).
    ``values`` are the by-value i32 arguments after the operands and their byte
    sizes, in ABI order: for implicit GEMM the 36 problem values then
    ``num_sub_gemms``. The workgroup is ``(block, 1, 1)``.

    ``views`` is, per operand, the ``(byte offset, byte size)`` window of its buffer
    the step binds, or ``None`` for the whole buffer: a batch-split launch sees only
    its images. An implicit-GEMM step's ``sub_gemm_buf`` is its ``table``, uploaded
    at ``table_offset`` bytes into the :data:`BUFFER_SUB_GEMM` region.
    """

    kernel: str
    abi: str
    grid: tuple[int, int, int]
    block: int
    buffers: tuple[str, str | None, str]
    values: tuple[tuple[str, int], ...]
    views: tuple[tuple[int, int] | None, ...] = (None, None, None)
    table: tuple[int, ...] = ()
    table_offset: int = 0


@dataclass(frozen=True)
class Gfx950ConvDgradPlan:
    """Everything hipkernel:Gfx950ConvDgradAot does to run one instance on one
    problem, in order: zero dX when ``zero_dx``, upload every step's record table,
    then launch ``steps``.

    ``launched`` is the problem the kernels see: the graph's own, or for implicit
    GEMM its width folded into the channels (``fold_width``), which is the same
    bytes in all three tensors. ``workspaces`` are ``(name, offset, bytes)`` regions
    of the one workspace.
    """

    launched: Gfx950ConvDgradProblem
    steps: tuple[Gfx950ConvDgradStep, ...]
    workspaces: tuple[tuple[str, int, int], ...] = ()
    zero_dx: bool = False

    @property
    def workspace_bytes(self) -> int:
        return max((offset + size for _, offset, size in self.workspaces), default=0)

    def buffer_bytes(self, name: str) -> int:
        """The byte size of buffer ``name`` (a step with a view passes the view's)."""
        tensors = dict(
            zip((BUFFER_DY, BUFFER_W, BUFFER_DX), self.launched.tensor_bytes())
        )
        if name in tensors:
            return tensors[name]
        for region, _, size in self.workspaces:
            if region == name:
                return size
        raise KeyError(f"the plan has no buffer {name!r}")

    def serialize(self) -> list[int]:
        """The plan as one integer sequence, the native pack's format."""
        out = [int(self.zero_dx), self.workspace_bytes, len(self.workspaces)]
        for name, offset, size in self.workspaces:
            out += [PLAN_BUFFER_CODES[name], offset, size]
        out.append(len(self.steps))
        for step in self.steps:
            out += [PLAN_KERNEL_CODES[step.kernel], PLAN_ABI_CODES[step.abi]]
            out += list(step.grid) + [step.block]
            out += [PLAN_BUFFER_CODES[b] for b in step.buffers]
            for view in step.views:
                out += list(view) if view is not None else [-1, -1]
            out.append(len(step.values))
            out += [value for _, value in step.values]
            out += [step.table_offset, len(step.table)]
            out += list(step.table)
        return out

    def digest(self) -> int:
        """FNV-1a 64 over :meth:`serialize`, each value as 8 little-endian bytes."""
        h = _FNV_OFFSET
        for value in self.serialize():
            for byte in (value & _U64).to_bytes(8, "little"):
                h = ((h ^ byte) * _FNV_PRIME) & _U64
        return h


def gfx950_conv_dgrad_aot_step_signature(abi: str, dtype: str) -> list[dict]:
    """The launch signature of a step of ABI ``abi`` (``kernels.common.conv_abi``)."""
    from kernels.common.conv_abi import conv_direct_transform_signature

    if abi == ABI_IMPLICIT:
        return conv_args_signature(dtype, direction="dgrad")
    if abi == ABI_DIRECT_DGRAD:
        return conv_direct_args_signature(dtype, direction="dgrad")
    if abi == ABI_DIRECT_FWD:
        return conv_direct_args_signature(dtype, direction="fwd")
    if abi in (ABI_TRANSFORM, ABI_TRANSFORM_RTC):
        return conv_direct_transform_signature(dtype, abi == ABI_TRANSFORM_RTC)
    raise ValueError(f"unknown step ABI {abi!r}")


def gfx950_conv_dgrad_aot_launch_problem(
    problem: Gfx950ConvDgradProblem,
) -> Gfx950ConvDgradProblem:
    """The problem the implicit-GEMM kernels run ``problem`` as.

    A dgrad whose width folds into its channels (``fold_width``: ``f`` divides the
    filter width, the W stride, the input width and the W padding, dilation 1,
    ungrouped) runs folded -- the same bytes in all three tensors, in fewer and wider
    sub-GEMMs -- exactly as rocKE's dispatch and its sweep launch it. Every other
    problem launches as it is.
    """
    p = problem.to_problem()
    f = _dgrad.width_fold(p)
    if f == 1:
        return problem
    folded = _dgrad.fold_width(p, f)
    return replace(
        problem, Wi=folded.Wi, C=folded.C, X=folded.X, sW=folded.sW, pW=folded.pW
    )


def _fits_i32(values) -> bool:
    return all(-_INT32_MAX - 1 <= v <= _INT32_MAX for v in values)


def _launch_values(conv_args, bytes_: tuple[int, int, int], **extra) -> tuple:
    """``ConvArgs.to_launch_values`` past the operands and their sizes, as
    ``(name, value)`` pairs in ABI order; the record-table pointer is left out (the
    plan binds it as a buffer)."""
    values = conv_args.to_launch_values(0, 0, 0, *bytes_, **extra)
    names = [name for name, _ in conv_args.arg_names()]
    return tuple((name, values[name]) for name in names[6:] if name != "sub_gemm_buf")


def _covered(extent: int, runs) -> bool:
    """Whether the ``(offset, stride, count, width)`` runs -- ``width`` consecutive
    positions from each of ``offset + stride * t``, ``t < count`` -- cover every
    position in ``[0, extent)``."""
    hit = bytearray(extent)
    for offset, stride, count, width in runs:
        for t in range(count):
            start = offset + stride * t
            for pos in range(max(start, 0), min(start + width, extent)):
                hit[pos] = 1
    return all(hit)


def _dx_fully_written(problem: ConvProblem, sub_gemms, patch: bool) -> bool:
    """Whether the implicit-GEMM kernel stores every dX element of ``problem``.

    A tilde record stores dX rows ``d_h_offset + d_h_stride * t`` (``t <
    h_tilde_slice``) by columns ``d_w_offset + d_w_stride * t`` (``t <
    w_tilde_slice``), every pixel of that grid -- a pixel whose taps all fall outside
    dY still stores the zero it sums to. Which records exist depends on each axis'
    tilde residue alone, and every record of one residue has the same row (column)
    set, so the records cover the rectangle exactly when the union of their rows
    covers ``[0, Hi)`` and the union of their columns ``[0, Wi)``. A strided problem
    whose filter misses a residue (a 1x1 filter at stride 2) leaves rows unwritten.

    A patch kernel's one record stores a whole ``Y x X`` patch from each of its
    pixels: it covers dX when the stride equals the filter and divides the extent.
    """
    width_h, width_w = (problem.Y, problem.X) if patch else (1, 1)
    rows = [
        (sg.d_h_offset, sg.d_h_stride, sg.h_tilde_slice, width_h) for sg in sub_gemms
    ]
    cols = [
        (sg.d_w_offset, sg.d_w_stride, sg.w_tilde_slice, width_w) for sg in sub_gemms
    ]
    return _covered(problem.Hi, rows) and _covered(problem.Wi, cols)


def _grid_reason(grid, block: int, x_cap: int) -> str:
    """Why a ``grid`` of ``block``-thread workgroups cannot launch, or ``""``."""
    if not 1 <= block <= _MAX_BLOCK_THREADS:
        return f"block of {block} threads exceeds {_MAX_BLOCK_THREADS}"
    if min(grid) < 1:
        return f"empty grid {grid}"
    if grid[0] > x_cap or grid[0] * block > _MAX_DISPATCH_ITEMS:
        return f"grid x = {grid[0]} exceeds {x_cap}"
    for axis, extent in zip("yz", grid[1:]):
        if extent > _MAX_GRID_DIM:
            return f"grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return ""


def gfx950_conv_dgrad_aot_batch_chunks(
    problem: Gfx950ConvDgradProblem,
) -> list[tuple[int, int]]:
    """``[(n0, n), ...]``: the batch slices one implicit-GEMM launch each covers
    (the benchmark's ``_batch_chunks``).

    A kernel takes its operand byte sizes as i32 kernargs and addresses each operand
    through a buffer resource whose sentinel is the i32 maximum, so no operand it is
    launched on may exceed 2^31 - 1 bytes. dgrad computes every image on its own and
    N is a kernarg: a batch whose dY or dX is larger runs as consecutive launches of
    the same binary over equal slices of N, at offset pointers. Raises ``ValueError``
    when one image is already too large.
    """
    dy_image, _, dx_image = problem.tensor_bytes(1)
    per_image = max(dy_image, dx_image)
    if per_image * problem.N <= _INT32_MAX:
        return [(0, problem.N)]
    if per_image > _INT32_MAX:
        raise ValueError(
            f"one image ({per_image} bytes) exceeds the kernels' 2^31-byte operand "
            f"addressing"
        )
    n_chunks = _ceil_div(problem.N, _INT32_MAX // per_image)
    step = _ceil_div(problem.N, n_chunks)
    return [(n0, min(step, problem.N - n0)) for n0 in range(0, problem.N, step)]


def _implicit_plan(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[Gfx950ConvDgradPlan | None, str]:
    """rocKE's offer for an implicit-GEMM binary (``identity_supports_problem`` on
    the launched problem) and the launch of its benchmark (``_extras_and_grid``,
    ``_cached_bind_batch``), plus the bounds the 32-bit launch needs."""
    from kernels.conv.kernel_identity import dgrad_sub_gemms, identity_supports_problem

    launched = gfx950_conv_dgrad_aot_launch_problem(problem)
    identity = spec.to_identity()
    threads = spec.warp_m * spec.warp_n * spec.wave_size
    chunks = gfx950_conv_dgrad_aot_batch_chunks(launched)
    dy_image, _, dx_image = launched.tensor_bytes(1)
    steps = []
    table_bytes = 0
    zero_dx = False
    for n0, n in chunks:
        chunk = replace(launched, N=n)
        p = chunk.to_problem()
        ok, why = identity_supports_problem(
            identity, p, llvm_flavor=identity.llvm_flavor
        )
        if not ok:
            return None, why
        sub_gemms = dgrad_sub_gemms(identity, p)
        if not sub_gemms:
            return None, "the problem decomposes into no non-empty tilde sub-GEMM"
        flat = sum(
            _ceil_div(sg.gemm_m, spec.tile_m) * _ceil_div(sg.gemm_n, spec.tile_n)
            for sg in sub_gemms
        )
        grid = (flat, chunk.groups, 1)
        why = _grid_reason(grid, threads, _INT32_MAX)
        if why:
            return None, f"dgrad tile grid: {why}"
        if chunk.N * chunk.Hi * chunk.Wi >= 1 << 31:
            return None, "gemm_m = N*Hi*Wi must stay below 2^31 (magic division)"
        table = tuple(_dgrad.pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n))
        values = _launch_values(
            ConvArgs.from_problem(
                p, direction="dgrad", tile_m=spec.tile_m, tile_n=spec.tile_n
            ),
            chunk.tensor_bytes(),
            sub_gemm_buf=0,
            num_sub_gemms=len(sub_gemms),
        )
        if not _fits_i32(table) or not _fits_i32(v for _, v in values):
            return None, "a launch value leaves the signed 32-bit kernel ABI"
        views = (None, None, None)
        if len(chunks) > 1:
            views = ((n0 * dy_image, n * dy_image), None, (n0 * dx_image, n * dx_image))
        steps.append(
            Gfx950ConvDgradStep(
                KERNEL_SELF,
                ABI_IMPLICIT,
                grid,
                threads,
                (BUFFER_DY, BUFFER_W, BUFFER_DX),
                values,
                views=views,
                table=table,
                table_offset=table_bytes,
            )
        )
        table_bytes += 4 * len(table)
        zero_dx = zero_dx or not _dx_fully_written(p, sub_gemms, bool(spec.patch))
    return (
        Gfx950ConvDgradPlan(
            launched=launched,
            steps=tuple(steps),
            workspaces=((BUFFER_SUB_GEMM, 0, table_bytes),),
            zero_dx=zero_dx,
        ),
        "ok",
    )


def _direct_plan(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[Gfx950ConvDgradPlan | None, str]:
    """``direct_kernel_sweep.plan_for`` for a dgrad direct instance: rocKE's offer
    (``identity_supports_problem``, then ``direct_spec_for_problem``'s per-shape
    re-validation), its steps and scratch, plus the launch bounds rocKE does not
    check and :func:`_coalesced_groups_reason`."""
    from kernels.conv.direct_variants import (
        DIRECT_DGRAD_FPROP_SINGLE,
        direct_dgrad_weight_transforms,
        direct_dgrad_workspaces,
        direct_spec_for_problem,
        make_direct_spec,
    )
    from kernels.conv.kernel_identity import identity_supports_problem

    if problem.sH != problem.sW or problem.pH != problem.pW:
        return None, "the direct kernels compile one stride and one padding"
    direct = problem.to_direct_problem()
    identity = spec.to_identity()
    ok, why = identity_supports_problem(
        identity, direct, llvm_flavor=identity.llvm_flavor
    )
    if not ok:
        return None, why
    knobs = spec.direct_knobs()
    rocke_spec, why = direct_spec_for_problem(
        spec.algorithm, direct, knobs, _ARCH, spec.wave_size
    )
    if rocke_spec is None:
        return None, why
    if spec.is_mfma:
        why = _coalesced_groups_reason(spec, problem) or _persistent_cells_reason(
            rocke_spec
        )
        if why:
            return None, why
    tensors = problem.tensor_bytes()
    if max(tensors) > _INT32_MAX:
        return None, "a tensor exceeds the direct kernels' 2^31-byte operand addressing"
    x_cap = (
        _MAX_GRID_DIM if spec.algorithm in _DIRECT_CONV_SPEC_ALGORITHMS else _INT32_MAX
    )
    if not spec.is_mfma:
        grid, block = _direct.direct_launch_geometry(rocke_spec)
        if spec.algorithm in DIRECT_DGRAD_FPROP_SINGLE:
            # A forward kernel over the transposed problem that reads the original
            # weights itself: the forward ABI, transposed extents.
            abi = ABI_DIRECT_FWD
            args = ConvArgs.from_problem(rocke_spec.problem, direction="fwd")
        else:
            abi = ABI_DIRECT_DGRAD
            args = ConvArgs.from_problem(direct, direction="dgrad")
        why = _grid_reason(grid, block[0], x_cap)
        if why:
            return None, why
        step = Gfx950ConvDgradStep(
            KERNEL_SELF,
            abi,
            grid,
            block[0],
            (BUFFER_DY, BUFFER_W, BUFFER_DX),
            _launch_values(args, tensors),
        )
        return Gfx950ConvDgradPlan(launched=problem, steps=(step,)), "ok"

    # MFMA pipeline: W -> ws_t (transpose), optionally ws_t -> ws_coa (reorganize),
    # then a forward pass over the transposed problem.
    runtime_channels = spec.is_runtime_channels
    scalars = (
        (("p_cpg", problem.cpg), ("p_kpg", problem.kpg)) if runtime_channels else ()
    )
    sizes = direct_dgrad_workspaces(direct, knobs)
    workspaces = [(BUFFER_WS_T, 0, sizes[BUFFER_WS_T])]
    if BUFFER_WS_COA in sizes:
        offset = _align_up(sizes[BUFFER_WS_T], MFMA_WORKSPACE_ALIGNMENT)
        workspaces.append((BUFFER_WS_COA, offset, sizes[BUFFER_WS_COA]))
    if any(size > _INT32_MAX for _, _, size in workspaces):
        return None, "an MFMA workspace exceeds the signed 32-bit kernel ABI"
    steps = []
    for helper, helper_knobs, src, dst in direct_dgrad_weight_transforms(
        knobs, spec.algorithm
    ):
        helper_spec = make_direct_spec(helper, direct, helper_knobs)
        grid, block = _direct.direct_launch_geometry(helper_spec)
        why = _grid_reason(grid, block[0], _INT32_MAX)
        if why:
            return None, f"{helper}: {why}"
        steps.append(
            Gfx950ConvDgradStep(
                AOT_DGRAD_DIRECT_ALGORITHMS[helper][1],
                ABI_TRANSFORM_RTC if runtime_channels else ABI_TRANSFORM,
                grid,
                block[0],
                (src, None, dst),
                scalars,
            )
        )
    weights = steps[-1].buffers[2]
    grid, block = _direct.direct_launch_geometry(rocke_spec)
    why = _grid_reason(grid, block[0], x_cap)
    if why:
        return None, f"compute pass: {why}"
    steps.append(
        Gfx950ConvDgradStep(
            KERNEL_SELF,
            ABI_DIRECT_FWD,
            grid,
            block[0],
            (BUFFER_DY, weights, BUFFER_DX),
            _launch_values(
                ConvArgs.from_problem(rocke_spec.problem, direction="fwd"),
                tensors,
            ),
        )
    )
    return (
        Gfx950ConvDgradPlan(
            launched=problem, steps=tuple(steps), workspaces=tuple(workspaces)
        ),
        "ok",
    )


def gfx950_conv_dgrad_aot_plan(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[Gfx950ConvDgradPlan | None, str]:
    """The launch of instance ``spec`` on ``problem``, or ``(None, why)`` when the
    instance does not compute it.

    The served predicate is rocKE's own offer, as its sweeps make it: implicit GEMM
    through ``identity_supports_problem`` on the problem it launches (folded, see
    :func:`gfx950_conv_dgrad_aot_launch_problem`), which holds the dY-halo, tap-outer
    K loop and patch kernels to the problems they compute; the direct pipelines
    through ``identity_supports_problem`` and ``direct_spec_for_problem``. On top come
    the launch bounds of the 32-bit dispatch and :func:`_coalesced_groups_reason`.
    The plan is what the native pack runs; it ports this function.
    """
    error = _problem_error(problem)
    if error:
        return None, error
    if problem.dtype != spec.dtype:
        return None, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_helper:
        return None, "a helper kernel is launched only by its MFMA pipeline"
    try:
        if spec.is_direct:
            return _direct_plan(spec, problem)
        return _implicit_plan(spec, problem)
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def gfx950_conv_dgrad_aot_serves(
    spec: Gfx950ConvDgradAotSpec, problem: Gfx950ConvDgradProblem
) -> tuple[bool, str]:
    """Whether instance ``spec`` computes ``problem`` (a plan exists)."""
    plan, why = gfx950_conv_dgrad_aot_plan(spec, problem)
    return plan is not None, why


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


def _flavor_rank(flavor: str) -> int:
    return int(flavor.removeprefix("llvm"))


def _require_packaged_flavor() -> None:
    """Refuse to package under an LLVM older than the catalog's.

    The catalog's instances are the kernels the rocKE sweep measured, lowered for
    :data:`PACKAGED_LLVM_FLAVOR` and compiled by a COMGR of that LLVM or newer. A
    pack step that resolves an older COMGR -- the ROCm the build found, rather than
    the one the sweep ran -- lowers for its flavor and gets that LLVM's code
    generation: the same configuration, but compiled to more registers and slower
    code, shipped without a word. Point ``HIPKERNELPROVIDER_ROCKE_COMGR_LIB`` at a
    COMGR of LLVM 22 or newer (and ``HIPKERNELPROVIDER_ROCKE_LLVM_FLAVOR`` at
    ``llvm22`` to lower exactly as the sweep did).
    """
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    flavor = _resolve_llvm_flavor()
    if _flavor_rank(flavor) < _flavor_rank(PACKAGED_LLVM_FLAVOR):
        raise ValueError(
            f"hipkernel:Gfx950ConvDgradAot packages {PACKAGED_LLVM_FLAVOR} kernels, "
            f"but this process lowers to {flavor} (an older COMGR): set "
            f"HIPKERNELPROVIDER_ROCKE_COMGR_LIB to a COMGR of LLVM 22 or newer"
        )


def build_gfx950_conv_dgrad_aot(
    spec: Gfx950ConvDgradAotSpec, *, arch: str = _ARCH
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Dispatches on the family as the sweeps do: implicit GEMM through
    ``build_implicit_gemm_conv_dgrad`` (``kernel_sweep.build_kernel``), the direct
    and helper kernels through ``direct_kernel_sweep._build_kernel``'s builders. A
    helper's kernel is renamed to :meth:`Gfx950ConvDgradAotSpec.kernel_symbol_prefix`
    (its builder names it after the probe problem, which fp16 and bf16 share).

    The direct builder is rocKE's own dispatch over its sweep algorithms, as the
    spec factory above is: a per-variant table here is a mirror, and a mirror goes
    stale -- it is what left the windowed depthwise unbuildable while the catalog
    already packaged it.
    """
    ok, reason = supports_gfx950_conv_dgrad_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT dgrad instance: {reason}")
    _require_packaged_flavor()
    if not spec.is_direct and spec.to_identity().stable_hash() != spec.instance_id:
        raise ValueError(
            f"instance {spec.instance_id} is not the identity its fields describe "
            f"(hash {spec.to_identity().stable_hash()}): the catalog and this adapter "
            f"disagree on a field"
        )
    if spec.is_direct:
        from kernels.conv.direct_variants import select_direct_kernel_builder

        kernel = select_direct_kernel_builder(
            spec.algorithm, spec.to_direct_spec(), arch
        )
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
    if getattr(request, "vec_size_c", None) is not None:
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
