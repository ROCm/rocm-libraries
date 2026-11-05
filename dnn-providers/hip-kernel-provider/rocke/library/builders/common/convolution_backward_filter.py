# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 shape-generic AOT backward-weight convolution.

This is the wgrad sibling of :mod:`builders.common.convolution_backward_data`
(the dgrad AOT adapter), mirrored structurally: one flat descriptor spec
(:class:`Gfx950ConvWgradAotSpec`, every field an ``int`` or ``str``) per rocKE
``KernelIdentity`` of the wgrad AOT catalog, built exactly as the sweep that
produced ``grouped_convolution_instances/gfx950_wgrad.json`` built it.

The contract is 2-D cross-correlation, NHWK dY, KYXC dW, NHWC X, uniform
fp16/bf16 activations, symmetric padding. Eight families are packaged:

* **Implicit GEMM** (``kernel_family`` 0): ``WgradConvSpec``. Fully
  shape-generic -- unlike dgrad's tilde decomposition, a wgrad binary bakes no
  filter, stride, padding or channel count at all, so it carries no sub-GEMM
  record table and runs the problem exactly as the graph states it (no
  ``fold_width``/patch embedding, unlike fwd/dgrad). Its split-K *launch*
  degree is never baked -- the real degree is resolved per request by the
  same cost model :func:`dispatch.grouped_convolution._wgrad_split_k` calls
  (:func:`kernels.common.conv_split_k.resolve_conv_split_k`) -- but the
  catalog's own ``split_k`` field is not a placeholder: ``split_k == 1`` vs
  ``> 1`` is whether the packaged binary was built with an atomic-accumulate
  epilogue at all (``WgradConvSpec``'s ``_needs_atomic``), so a ``split_k ==
  1`` instance is held to degree 1 regardless of what the cost model would
  otherwise pick. A two-stage instance (``two_stage``) splits K into an f32
  scratch of ``ws_replicas`` dW-sized slabs per group instead of
  atomic-adding a packed 16-bit pair into dW, so it carries no dW-row parity
  rule: it serves every problem its binary takes on which K splits at least
  two ways and the scratch fits the i32 ABI -- exactly
  ``dispatch.grouped_convolution._admits``'s two-stage offer -- and its plan
  zeroes the scratch, launches the instance, then the Stage 2 fold.
* **Direct** (``kernel_family`` 1), every one of rocKE's seven wgrad direct
  pipelines (``direct_wgrad``, ``direct_wgrad_tiled``, ``direct_wgrad_dw``,
  ``direct_wgrad_dw_cols``, ``direct_wgrad_dw_mfma``, ``direct_wgrad_gather``,
  ``direct_wgrad_grouped_mfma``). Every one writes an fp32 ``dW``
  (``kernels.common.conv_abi.conv_direct_arg_names``): two
  (``direct_wgrad``, ``direct_wgrad_tiled``) atomic-add into it directly, so
  the caller must zero it first; the other five instead store one partial
  slot per partition into a scratch workspace and need a second kernel,
  ``direct_wgrad_fold``, to sum the partitions into ``dW`` -- which it writes
  whole, so it needs no zeroing. Reuses
  :mod:`kernels.conv.direct_variants`'s ``(variant, problem, knobs)`` table
  rather than re-deriving any of the seven families' shape rules.
* **Helpers** (``kernel_family`` 2), never serving a problem on their own:
  ``direct_wgrad_fold``, packaged so the five folded pipelines' handler can
  load it, and ``direct_wgrad_ws_reduce``, the two-stage implicit GEMM's
  Stage 2 (one per dtype and ``ws_replicas``, which it records in that field;
  every extent is a kernarg).

:func:`gfx950_conv_wgrad_aot_plan` is the launch of one instance on one
problem -- rocKE's own offer (``identity_supports_problem``,
``direct_spec_for_problem``) plus the launch bounds, then its steps and
workspace -- and is what a later native geometry pack ports; the two are
compared by :meth:`Gfx950ConvWgradPlan.digest`.

``instance_id`` is ``KernelIdentity.stable_hash()``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common import conv_direct_grouped as _direct
from kernels.common import conv_implicit_gemm_wgrad as _wgrad
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature
from kernels.common.conv_args import ConvArgs
from rocke.core.ir import KernelDef

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1
#: Grid y and z, and every axis of the direct compute grids.
_MAX_GRID_DIM = 65535
_MAX_BLOCK_THREADS = 1024
_WAVE_SIZE = 64
_DTYPES = ("fp16", "bf16")
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_INSTANCE_ID_LENGTH = 40
#: The LLVM flavor the catalog's instances were swept, hashed and measured
#: under (``gfx950_wgrad.json``'s instances all end in it).
PACKAGED_LLVM_FLAVOR = "llvm22"

#: ``kernel_family`` values; the metadata field is also an integer engine knob.
KERNEL_FAMILY_IMPLICIT_GEMM = 0
KERNEL_FAMILY_DIRECT = 1
KERNEL_FAMILY_WGRAD_HELPER = 2
_KERNEL_FAMILIES = (
    KERNEL_FAMILY_IMPLICIT_GEMM,
    KERNEL_FAMILY_DIRECT,
    KERNEL_FAMILY_WGRAD_HELPER,
)

#: ``direct_variant`` values. Implicit GEMM carries "none". Unlike dgrad, a
#: wgrad direct identity's ``algorithm`` already *is* its variant name --
#: there is no separate rocKE-internal spelling to translate.
DIRECT_VARIANT_NONE = "none"
DIRECT_VARIANT_WGRAD = "direct_wgrad"
DIRECT_VARIANT_WGRAD_TILED = "direct_wgrad_tiled"
DIRECT_VARIANT_WGRAD_DW = "direct_wgrad_dw"
DIRECT_VARIANT_WGRAD_DW_COLS = "direct_wgrad_dw_cols"
DIRECT_VARIANT_WGRAD_DW_MFMA = "direct_wgrad_dw_mfma"
DIRECT_VARIANT_WGRAD_GATHER = "direct_wgrad_gather"
DIRECT_VARIANT_WGRAD_GROUPED_MFMA = "direct_wgrad_grouped_mfma"
HELPER_VARIANT_FOLD = "direct_wgrad_fold"
HELPER_VARIANT_WS_REDUCE = "direct_wgrad_ws_reduce"
_DIRECT_VARIANTS = (
    DIRECT_VARIANT_WGRAD,
    DIRECT_VARIANT_WGRAD_TILED,
    DIRECT_VARIANT_WGRAD_DW,
    DIRECT_VARIANT_WGRAD_DW_COLS,
    DIRECT_VARIANT_WGRAD_DW_MFMA,
    DIRECT_VARIANT_WGRAD_GATHER,
    DIRECT_VARIANT_WGRAD_GROUPED_MFMA,
)
_HELPER_VARIANTS = (HELPER_VARIANT_FOLD, HELPER_VARIANT_WS_REDUCE)
#: The five families that store partial slots and need the fold helper
#: (``kernels.conv.direct_variants.DIRECT_WGRAD_FOLDED``).
_FOLDED_VARIANTS = (
    DIRECT_VARIANT_WGRAD_DW,
    DIRECT_VARIANT_WGRAD_DW_COLS,
    DIRECT_VARIANT_WGRAD_DW_MFMA,
    DIRECT_VARIANT_WGRAD_GATHER,
    DIRECT_VARIANT_WGRAD_GROUPED_MFMA,
)
#: Families whose fold is per output channel (``KH*KW*cpg`` taps) rather than
#: per depthwise group (``KH*KW``) -- ``direct_wgrad_fold_knobs``'s
#: ``per_channel``.
_PER_CHANNEL_FOLD_VARIANTS = (DIRECT_VARIANT_WGRAD_GATHER, DIRECT_VARIANT_WGRAD_GROUPED_MFMA)

#: Each wgrad direct-sweep algorithm and the ``(kernel_family, direct_variant)``
#: an instance of it carries. A wgrad identity's ``algorithm`` already spells
#: its ``direct_variant`` (no rocKE-internal rename, unlike dgrad).
AOT_WGRAD_DIRECT_ALGORITHMS = {
    variant: (KERNEL_FAMILY_DIRECT, variant) for variant in _DIRECT_VARIANTS
}
for _helper in _HELPER_VARIANTS:
    AOT_WGRAD_DIRECT_ALGORITHMS[_helper] = (KERNEL_FAMILY_WGRAD_HELPER, _helper)

#: The implicit-GEMM fields of a direct or helper instance hold exactly what
#: a direct identity leaves them at (mirrors dgrad's ``_DIRECT_PLACEHOLDERS``).
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
    "group_merge": 1,
    "ws_replicas": 0,
    "prefetch_addr": 0,
    "async_dma": 0,
    "unroll_k": 0,
    "lds_k_pad": 0,
    "lds_layout": "default",
    "split_k": -1,
    "two_stage": 0,
    "grouped": 0,
}
#: The capability fields every direct and helper instance states. Unlike
#: dgrad, a wgrad implicit-GEMM instance bakes *none* of these (the kernel is
#: fully shape-generic); they are a direct/helper-only concept here.
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
#: Per direct/helper variant, the knob names its catalog rows use and the
#: instance field each one is carried in -- here the identity. No knob is
#: renamed or spelled as a bool (the wgrad direct sweep's knobs are all
#: plain non-negative integers).
AOT_WGRAD_DIRECT_KNOBS = {
    DIRECT_VARIANT_WGRAD: {
        "ho_per_block": "ho_per_block",
        "mfma_k": "mfma_k",
        "waves_c": "waves_c",
        "waves_k": "waves_k",
        "waves_q": "waves_q",
    },
    DIRECT_VARIANT_WGRAD_TILED: {
        "prefetch": "prefetch",
        "row_unroll": "row_unroll",
        "rows_per_seg": "rows_per_seg",
        "target_blocks": "target_blocks",
        "unfold_n": "unfold_n",
        "wave_c16": "wave_c16",
        "wave_k16": "wave_k16",
        "waves_c": "waves_c",
        "waves_k": "waves_k",
    },
    DIRECT_VARIANT_WGRAD_DW: {
        "lanes_c": "lanes_c",
        "lanes_w": "lanes_w",
        "prefetch": "prefetch",
        "rows_per_seg": "rows_per_seg",
        "target_blocks": "target_blocks",
    },
    DIRECT_VARIANT_WGRAD_DW_COLS: {
        "channels": "channels",
        "strip": "strip",
        "target_blocks": "target_blocks",
        "threads": "threads",
    },
    DIRECT_VARIANT_WGRAD_DW_MFMA: {
        "prefetch": "prefetch",
        "q_tiles": "q_tiles",
        "rows_per_chunk": "rows_per_chunk",
        "target_blocks": "target_blocks",
        "unfold_n": "unfold_n",
    },
    DIRECT_VARIANT_WGRAD_GATHER: {
        "prefetch": "prefetch",
        "rows_per_seg": "rows_per_seg",
        "target_blocks": "target_blocks",
        "wave_k16": "wave_k16",
        "waves_k": "waves_k",
    },
    DIRECT_VARIANT_WGRAD_GROUPED_MFMA: {
        "prefetch": "prefetch",
        "rows_per_chunk": "rows_per_chunk",
        "target_blocks": "target_blocks",
        "wave_q4": "wave_q4",
        "waves_q": "waves_q",
    },
    HELPER_VARIANT_FOLD: {"taps": "taps"},
    HELPER_VARIANT_WS_REDUCE: {"ws_replicas": "ws_replicas"},
}
#: Knobs carried in an implicit-GEMM field rather than a knob field of their
#: own: the Stage 2 fold's slab count is the instance's ``ws_replicas``.
_KNOBS_IN_IMPLICIT_FIELDS = ("ws_replicas",)
#: Every knob field name across every variant, each defaulting to 0 (no wgrad
#: direct knob is a bool or has a nonzero "unset" meaning, unlike dgrad's
#: ``lds_pad``/``waves_per_eu``).
_KNOB_DEFAULTS = {
    name: 0
    for mapping in AOT_WGRAD_DIRECT_KNOBS.values()
    for name in mapping.values()
    if name not in _KNOBS_IN_IMPLICIT_FIELDS
}
#: Fields appended after the implicit-GEMM ones. An implicit-GEMM instance
#: leaves each at this value; a direct or helper instance leaves the knobs
#: its variant does not use at it.
_FAMILY_DEFAULTS = {
    "kernel_family": KERNEL_FAMILY_IMPLICIT_GEMM,
    "direct_variant": DIRECT_VARIANT_NONE,
    **{name: 0 for name in _DIRECT_CAPABILITY_FIELDS},
    **_KNOB_DEFAULTS,
}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# The runtime problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950ConvWgradProblem:
    """One backward-weight problem as hipkernel:Gfx950ConvWgradAot matches it.

    dY is ``[N, Ho, Wo, K]`` NHWK, X is ``[N, Hi, Wi, C]`` NHWC and dW is
    ``[K, Y, X, C/groups]`` KYXC (``A`` = dY, ``B`` = X, ``D`` = dW).
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
        """The implicit-GEMM view (``ConvProblem``). Wgrad runs it unfolded --
        no ``fold_width`` patch-embedding trick, unlike fwd/dgrad."""
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

    def tensor_bytes(self, dw_bytes: int = 2) -> tuple[int, int, int]:
        """``(dY, X, dW)`` sizes in bytes: dY/X are 16-bit elements over the
        whole batch (wgrad's reduction spans ``N``, so there is no per-image
        batch slice to shrink them the way dgrad's dX/dY are); ``dW`` is
        filter-shaped, independent of ``N``, in ``dw_bytes``-byte elements
        (2 for an implicit-GEMM instance's native dtype, 4 for a direct
        instance's fp32 accumulator -- see
        :func:`kernels.common.conv_abi.conv_direct_arg_names`)."""
        return (
            self.N * self.Ho * self.Wo * self.K * 2,
            self.N * self.Hi * self.Wi * self.C * 2,
            self.K * self.Y * self.X * self.cpg * dw_bytes,
        )


def _problem_error(problem: Gfx950ConvWgradProblem) -> str:
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
        return f"groups must be an integer in [1, {_MAX_GRID_DIM}] (grid z)"
    if problem.C % problem.groups:
        return f"C={problem.C} is not divisible by groups={problem.groups}"
    if problem.K % problem.groups:
        return f"K={problem.K} is not divisible by groups={problem.groups}"
    if problem.layout != "NHWC":
        return "the packaged wgrad contract requires dense NHWC storage"
    if problem.dtype not in _DTYPES:
        return "the packaged wgrad contract supports fp16 or bf16 only"
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
    # Unlike dgrad, wgrad's reduction (GEMM-K) spans the whole batch N, so
    # there is no batch-slice escape valve: dY and X (the tensors that scale
    # with N) must each fit the 32-bit kernel ABI as a whole, not per image.
    # dW does not scale with N or with spatial extent at all, but is checked
    # too (conservatively, at the widest -- fp32 -- element width a packaged
    # family can bake), since nothing shrinks it either.
    for label, size in zip(("dY", "X", "dW"), problem.tensor_bytes(dw_bytes=4)):
        if size > _INT32_MAX:
            return f"{label} byte count exceeds the signed 32-bit kernel ABI"
    return ""


# ---------------------------------------------------------------------------
# The instance
# ---------------------------------------------------------------------------


def aot_wgrad_direct_fields_from_knobs(variant: str, knobs: dict) -> dict:
    """The instance fields a direct or helper identity's ``knobs`` set.

    Raises ``ValueError`` for a knob the variant does not take, or a
    non-integer value. An omitted knob keeps its :data:`_KNOB_DEFAULTS` value
    (0), which for every wgrad direct knob is also "not used".
    """
    mapping = AOT_WGRAD_DIRECT_KNOBS[variant]
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
        if type(value) is not int:
            raise ValueError(f"knob {key} must be an integer, got {value!r}")
        values[field_name] = value
    return values


def aot_wgrad_direct_instance_fields(identity) -> dict:
    """The direct fields of the instance packaged from a wgrad direct or
    helper identity.

    ``identity`` is read duck-typed (a ``KernelIdentity``, which this layer
    cannot import): its ``algorithm``, compiled filter, stride, padding,
    dilation, channel counts and ``knobs`` JSON. Raises ``ValueError`` for an
    algorithm without a packaged launch or knobs that are not exactly its
    variant's.
    """
    family_variant = AOT_WGRAD_DIRECT_ALGORITHMS.get(identity.algorithm)
    if family_variant is None:
        raise ValueError(
            f"direct algorithm {identity.algorithm!r} has no packaged wgrad launch "
            f"(supported: {sorted(AOT_WGRAD_DIRECT_ALGORITHMS)})"
        )
    family, variant = family_variant
    fields_ = {"kernel_family": family, "direct_variant": variant}
    fields_.update(
        {name: getattr(identity, name) for name in _DIRECT_CAPABILITY_FIELDS}
    )
    fields_.update(
        aot_wgrad_direct_fields_from_knobs(
            variant, json.loads(identity.knobs) if identity.knobs else {}
        )
    )
    return fields_


@dataclass(frozen=True)
class Gfx950ConvWgradAotSpec:
    """One shape-generic backward-weight instance.

    Booleans are ints (0/1), so every field is a descriptor ``int`` or
    ``string``. The leading fields are the implicit-GEMM identity's
    (``WgradConvSpec``'s tuning, ``group_merge``, ``ws_replicas``,
    ``split_k``, ``two_stage``). The trailing ones describe a direct or
    helper instance: ``kernel_family``, ``direct_variant``, the compiled
    filter, stride, padding, dilation and channel counts and the variant's
    knobs (:data:`AOT_WGRAD_DIRECT_KNOBS`). A direct or helper instance holds
    the direct identity's placeholders in the implicit-GEMM fields (every one
    of them, since a wgrad implicit-GEMM instance bakes no capability at
    all); an implicit-GEMM instance leaves every trailing field at its KMD
    default.

    ``split_k`` defaults to ``-1`` on the dataclass, but the catalog bakes a
    concrete value (1, or > 1 recorded as 2): never the degree actually
    launched -- :func:`gfx950_conv_wgrad_aot_split_k` re-resolves the launch
    degree from the real problem, exactly as
    ``dispatch.grouped_convolution._wgrad_split_k`` does -- but it *is* read
    for one thing: ``split_k == 1`` means no atomic epilogue was built into
    this binary at all, so the resolved degree is held to 1 regardless of
    what the problem shape would otherwise call for. It otherwise exists so
    the reconstructed identity (:meth:`to_identity`) hashes back to
    ``instance_id``. ``two_stage`` selects the f32-scratch reduction: its
    plan launches the ``direct_wgrad_ws_reduce`` helper of the same dtype and
    ``ws_replicas`` after the instance.

    A two-stage *implicit-GEMM* instance is the only one; the direct and
    helper families have no two-stage form at all (the catalog never bakes
    one), so ``two_stage`` is always 0 on those. ``ws_replicas`` is 0 on them
    too, except on the ``direct_wgrad_ws_reduce`` helper, which records its
    slab count there.
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
    grouped: int
    instance_id: str
    layout: str = "NHWC"
    async_dma: int = 0
    unroll_k: int = 0
    lds_k_pad: int = 0
    lds_layout: str = "default"
    group_merge: int = 1
    ws_replicas: int = 0
    prefetch_addr: int = 0
    split_k: int = -1
    two_stage: int = 0
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
    # direct_wgrad
    ho_per_block: int = 0
    mfma_k: int = 0
    waves_c: int = 0
    waves_k: int = 0
    waves_q: int = 0
    # direct_wgrad_dw / _dw_mfma / _tiled
    lanes_c: int = 0
    lanes_w: int = 0
    prefetch: int = 0
    rows_per_seg: int = 0
    target_blocks: int = 0
    q_tiles: int = 0
    rows_per_chunk: int = 0
    unfold_n: int = 0
    row_unroll: int = 0
    wave_c16: int = 0
    wave_k16: int = 0
    # direct_wgrad_dw_cols
    channels: int = 0
    strip: int = 0
    threads: int = 0
    # direct_wgrad_grouped_mfma
    wave_q4: int = 0
    # direct_wgrad_fold (helper)
    taps: int = 0

    # ---- family ----

    @property
    def is_direct(self) -> bool:
        """A direct conv kernel or a helper (no implicit-GEMM tile)."""
        return self.kernel_family != KERNEL_FAMILY_IMPLICIT_GEMM

    @property
    def is_helper(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_WGRAD_HELPER

    @property
    def is_ws_reduce(self) -> bool:
        """The two-stage implicit GEMM's Stage 2 fold."""
        return self.direct_variant == HELPER_VARIANT_WS_REDUCE

    @property
    def is_folded(self) -> bool:
        """A direct instance whose pipeline needs the fold helper after it."""
        return self.direct_variant in _FOLDED_VARIANTS

    @property
    def algorithm(self) -> str:
        """The rocKE ``KernelIdentity.algorithm`` this instance was packaged
        from. A wgrad direct/helper identity's algorithm already is its
        ``direct_variant``."""
        if not self.is_direct:
            return "implicit_gemm"
        return self.direct_variant

    def direct_knobs(self) -> dict:
        """The identity's knobs: the inverse of
        :func:`aot_wgrad_direct_fields_from_knobs`."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct knobs")
        knobs = {}
        for key, field_name in AOT_WGRAD_DIRECT_KNOBS[self.direct_variant].items():
            value = getattr(self, field_name)
            if value:
                knobs[key] = value
            # Otherwise the instance did not state it; every wgrad direct
            # knob's 0 is "not used", so leaving it out is exact.
        return knobs

    def kernel_symbol_prefix(self) -> str:
        if self.is_helper:
            return f"hkp_conv_wgrad_helper_aot_gfx950_{self.instance_id[:24]}"
        if self.is_direct:
            return f"hkp_conv_wgrad_dw_aot_gfx950_{self.instance_id[:24]}"
        return f"hkp_conv_wgrad_aot_gfx950_{self.instance_id[:24]}"

    # ---- the catalog's build ----

    def to_identity(self):
        """The rocKE ``KernelIdentity`` this instance was packaged from.

        Implicit GEMM: every field the packaging does not carry is at its
        KernelIdentity default, which is exactly what
        ``supports_gfx950_conv_wgrad_aot`` enforces, so this reconstructs the
        identity (its ``stable_hash()`` is the ``instance_id``). Direct and
        helper instances: the identity the catalog's instance list spells, up
        to the spelling of the knobs JSON.

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
        if self.is_ws_reduce:
            from dataclasses import replace as _replace

            from kernels.conv.direct_aot_jobs import _ws_reduce_identity

            return _replace(
                _ws_reduce_identity(_ARCH, self.wave_size, self.ws_replicas, self.dtype),
                llvm_flavor=PACKAGED_LLVM_FLAVOR,
            )
        if self.is_direct:
            return KernelIdentity(
                direction="direct_wgrad_helper" if self.is_helper else "direct_wgrad",
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
            "group_merge",
            "ws_replicas",
            "split_k",
            "lds_k_pad",
            "lds_layout",
        )
        flags = (
            "lds_k_outer",
            "grouped",
            "prefetch_addr",
            "two_stage",
            "async_dma",
            "unroll_k",
        )
        return KernelIdentity(
            direction="wgrad",
            algorithm="implicit_gemm",
            **{name: getattr(self, name) for name in carried},
            **{name: bool(getattr(self, name)) for name in flags},
            **common,
        )

    def to_instance_spec(
        self, problem: ConvProblem | None = None
    ) -> _wgrad.WgradConvSpec:
        """The rocKE builder spec, built on the real ``problem`` (an
        implicit-GEMM wgrad binary bakes no geometry, so it has no probe of
        its own the way dgrad's tilde/patch kernels do -- it is validated
        against whatever problem it is asked to run)."""
        if self.is_direct:
            raise ValueError("a direct instance has no implicit-GEMM spec")
        if problem is None:
            raise ValueError("an implicit-GEMM wgrad instance needs a real problem")
        return _wgrad.WgradConvSpec(
            problem=problem,
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
            wave_size=self.wave_size,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            async_dma=bool(self.async_dma),
            unroll_k=bool(self.unroll_k),
            lds_k_pad=self.lds_k_pad or None,
            vector_size_a=self.vector_size_a or None,
            vector_size_b=self.vector_size_b or None,
            vector_size_c=self.vector_size_c or None,
            lds_k_outer=bool(self.lds_k_outer),
            group_merge=self.group_merge,
            prefetch_addr=bool(self.prefetch_addr),
            split_k=self.split_k,
            two_stage=bool(self.two_stage),
            ws_replicas=self.ws_replicas or _wgrad._DEFAULT_WS_REPLICAS,
            streamk="off",
        )

    def rocke_direct_spec(self, problem: _direct.DirectConvProblem):
        """The rocKE kernel spec of this direct or helper instance on
        ``problem``, as :func:`kernels.conv.direct_variants.make_direct_spec`
        makes it; only the name differs for a non-helper (the fold helper has
        no name field: :func:`build_gfx950_conv_wgrad_aot` renames its
        kernel)."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        from kernels.conv.direct_variants import make_direct_spec

        if self.is_helper:
            return make_direct_spec(self.algorithm, problem, self.direct_knobs())
        spec = make_direct_spec(
            self.algorithm, problem, self.direct_knobs(), wave_size=self.wave_size
        )
        return replace(spec, name=self.kernel_symbol_prefix())

    def probe_problem(self) -> _direct.DirectConvProblem:
        """rocKE's own probe for this direct/helper instance's capability
        axes (``kernels.conv.direct_aot_jobs.probe_problem``), not a mirror of
        it -- the emitted IR depends on the capability axes, so a probe that
        does not match them silently builds a binary the cache then offers
        for shapes it cannot compute."""
        if not self.is_direct:
            raise ValueError("only a direct/helper instance has a capability probe")
        if self.is_ws_reduce:
            raise ValueError(
                "the two-stage Stage 2 fold bakes no capability; it has no probe"
            )
        from kernels.conv.direct_aot_jobs import DirectCaps, probe_problem

        caps = DirectCaps(
            KH=self.filter_h,
            PAD=self.pad_h,
            stride=self.stride_h,
            cpg=self.cpg,
            kpg=self.kpg,
            dil_h=self.dilation_h,
            dil_w=self.dilation_w,
            kw=0 if self.filter_w == self.filter_h else self.filter_w,
            pad_w=-1 if self.pad_w == self.pad_h else self.pad_w,
        )
        return probe_problem(
            caps, dtype=self.dtype, variant=self.algorithm, knobs=self.direct_knobs()
        )

    def to_direct_spec(self):
        """The rocKE builder spec for the capability probe problem (the
        shape-independent spec for the Stage 2 fold)."""
        if self.is_ws_reduce:
            from kernels.common.conv_wgrad_workspace_reduce import (
                wgrad_reduce_aot_spec,
            )

            return wgrad_reduce_aot_spec(self.dtype, self.ws_replicas)
        return self.rocke_direct_spec(self.probe_problem())

    def kernel_name(self) -> str:
        """The launch symbol the packaged kernel carries. Only meaningful for
        a direct/helper instance, or for an implicit-GEMM one given a real
        problem (it bakes no probe of its own)."""
        if self.is_direct:
            return self.to_direct_spec().kernel_name()
        raise ValueError(
            "an implicit-GEMM instance's kernel name depends on the real "
            "problem (to_instance_spec(problem).kernel_name())"
        )


def gfx950_conv_wgrad_aot_fold_fields(spec: Gfx950ConvWgradAotSpec) -> dict:
    """The fold-helper instance the five folded wgrad families pull in, as the
    KMD field values that identify it in the catalog (with the instance's
    dtype): ``taps`` is ``KH*KW`` for the depthwise families, ``KH*KW*cpg``
    for the dense gather and grouped-MFMA ones whose fold is per channel
    (``kernels.conv.direct_variants.direct_wgrad_fold_knobs``)."""
    if not spec.is_folded:
        raise ValueError(f"{spec.direct_variant} does not fold partial slots")
    per_channel = spec.direct_variant in _PER_CHANNEL_FOLD_VARIANTS
    taps = spec.filter_h * spec.filter_w * (spec.cpg if per_channel else 1)
    return {
        "dtype": spec.dtype,
        "kernel_family": KERNEL_FAMILY_WGRAD_HELPER,
        "direct_variant": HELPER_VARIANT_FOLD,
        "filter_h": 0,
        "filter_w": 0,
        "stride_h": 0,
        "stride_w": 0,
        "pad_h": 0,
        "pad_w": 0,
        "dilation_h": 0,
        "dilation_w": 0,
        "cpg": 0,
        "kpg": 0,
        "taps": taps,
    }


def gfx950_conv_wgrad_aot_ws_reduce_fields(spec: Gfx950ConvWgradAotSpec) -> dict:
    """The Stage 2 fold a two-stage implicit-GEMM instance pulls in, as the
    KMD field values that identify it in the catalog: its dtype and slab
    count (``dispatch.grouped_convolution._instance_helpers``)."""
    if spec.is_direct or not spec.two_stage:
        raise ValueError("only a two-stage implicit-GEMM instance has a Stage 2")
    return {
        "dtype": spec.dtype,
        "kernel_family": KERNEL_FAMILY_WGRAD_HELPER,
        "direct_variant": HELPER_VARIANT_WS_REDUCE,
        "ws_replicas": spec.ws_replicas or _wgrad._DEFAULT_WS_REPLICAS,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _direct_validate(spec: Gfx950ConvWgradAotSpec, rocke_spec, arch: str):
    """rocKE's validator for ``rocke_spec`` (the fold helper has none)."""
    if spec.is_helper:
        return True, "ok"
    from kernels.conv.direct_variants import validate_direct_spec

    try:
        return validate_direct_spec(spec.algorithm, rocke_spec, arch)
    except (TypeError, ValueError, ZeroDivisionError, KeyError) as exc:
        return False, str(exc)


def _direct_error(spec: Gfx950ConvWgradAotSpec) -> str:
    """The direct and helper instance contract, before rocKE's validators."""
    for name, value in _DIRECT_PLACEHOLDERS.items():
        actual = getattr(spec, name)
        if name == "ws_replicas" and spec.is_ws_reduce:
            if type(actual) is not int or not 1 <= actual <= _INT32_MAX:
                return "the Stage 2 fold records its ws_replicas, a positive count"
            continue
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
    if spec.is_helper:
        for name in _DIRECT_CAPABILITY_FIELDS:
            if getattr(spec, name) != 0:
                return f"a helper instance records {name} as 0"
    else:
        for name in _DIRECT_CAPABILITY_FIELDS:
            value = getattr(spec, name)
            if type(value) is not int or not 0 <= value <= _INT32_MAX:
                return f"{name} must be a nonnegative signed 32-bit integer"
        for name in ("filter_h", "filter_w"):
            if getattr(spec, name) == 0:
                return f"{name} must be positive"
        if spec.filter_h != spec.filter_w:
            return "the direct wgrad sweep builds square filters only"
        if spec.dilation_h != 1 or spec.dilation_w != 1:
            return (
                "the direct wgrad kernels have no dilation "
                "(dilation_h == dilation_w == 1)"
            )
        if spec.stride_h == 0 or spec.stride_w == 0:
            return "stride_h/stride_w must be positive"
        if spec.stride_h != spec.stride_w or spec.pad_h != spec.pad_w:
            return "the direct wgrad kernels compile one stride and one padding"
        # cpg == 0 (and kpg == 0) is rocKE's "runtime channel count" (DirectCaps):
        # direct_wgrad and direct_wgrad_tiled always take both as kernargs;
        # direct_wgrad_gather bakes cpg but takes kpg at runtime; every other
        # family bakes both.
        if spec.direct_variant in (DIRECT_VARIANT_WGRAD, DIRECT_VARIANT_WGRAD_TILED):
            if spec.cpg or spec.kpg:
                return (
                    f"{spec.direct_variant} takes cpg and kpg as kernargs; both "
                    f"must be 0"
                )
        elif spec.direct_variant == DIRECT_VARIANT_WGRAD_GATHER:
            if spec.cpg == 0:
                return "direct_wgrad_gather bakes cpg; it must be positive"
            if spec.kpg:
                return "direct_wgrad_gather takes kpg as a kernarg; it must be 0"
        else:
            if spec.cpg == 0 or spec.kpg == 0:
                return f"{spec.direct_variant} bakes cpg and kpg; both must be positive"
    used = set(AOT_WGRAD_DIRECT_KNOBS[spec.direct_variant].values())
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
        elif not 0 <= value <= _INT32_MAX:
            return f"{name} must be a nonnegative signed 32-bit integer"
    return ""


def supports_gfx950_conv_wgrad_aot(
    spec: Gfx950ConvWgradAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance; rocKE's validators decide
    last."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx950ConvWgradAotSpec):
        return False, "expected Gfx950ConvWgradAotSpec"
    if spec.dtype not in _DTYPES:
        return False, "the packaged wgrad contract supports fp16 or bf16 only"
    if spec.layout != "NHWC":
        return False, "the packaged wgrad contract requires dense NHWC storage"
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
            f"wgrad-fold helper), got {spec.kernel_family!r}"
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
    ):
        value = getattr(spec, name)
        if type(value) is not int or not 0 < value <= _INT32_MAX:
            return False, f"{name} must be a positive signed 32-bit integer"
    if spec.wave_size != _WAVE_SIZE:
        return False, f"wave_size must be {_WAVE_SIZE} on {_ARCH}"
    for name in (
        "lds_k_outer",
        "grouped",
        "prefetch_addr",
        "async_dma",
        "unroll_k",
        "two_stage",
    ):
        value = getattr(spec, name)
        if type(value) is not int or value not in (0, 1):
            return False, f"{name} must be 0 or 1"
    if spec.split_k != -1 and spec.split_k < 1:
        return False, f"split_k must be -1 (auto), 1, or >1 (got {spec.split_k})"
    if spec.group_merge < 1:
        return False, f"group_merge must be >= 1 (got {spec.group_merge})"
    if spec.two_stage and spec.split_k == 1:
        return False, "two_stage=1 requires split_k != 1 (-1 auto or > 1)"
    identity = spec.to_identity()
    if identity.stable_hash() != spec.instance_id:
        return False, (
            f"instance {spec.instance_id} is not the identity its fields "
            f"describe (hash {identity.stable_hash()})"
        )
    # A spec needs *some* concrete problem to validate an implicit-GEMM wgrad
    # instance's own WgradConvSpec.validate()/is_valid_wgrad_spec rules
    # (block size, LDS budget, MFMA atom availability, group_merge's "fits one
    # tile" gate, ...), because the binary bakes no geometry of its own to
    # probe. group_merge is depthwise-only and must divide groups, so the
    # probe is tailored to it rather than "merely big enough".
    if spec.group_merge > 1:
        g = spec.group_merge
        probe = ConvProblem(N=1, Hi=16, Wi=16, C=g, K=g, Y=3, X=3, pH=1, pW=1, groups=g)
    else:
        probe = ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    try:
        instance = spec.to_instance_spec(probe)
        instance.validate()
        ok, reason = _wgrad.is_valid_wgrad_spec(instance, arch=arch)
        if not ok:
            return False, reason
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, "ok"


# ---------------------------------------------------------------------------
# The launch plan
# ---------------------------------------------------------------------------

#: The kernel a plan step launches: the instance itself, or a helper.
KERNEL_SELF = "self"
#: The ABI a plan step launches with.
#: ``implicit_wgrad``: the implicit-GEMM wgrad block (``dY, X, dW``, byte
#: sizes, the problem values, ``split_k``).
ABI_IMPLICIT = "implicit_wgrad"
#: ``direct_wgrad``: the direct wgrad block (``dY, X, dW`` -- fp32 ``dW``).
ABI_DIRECT_WGRAD = "direct_wgrad"
#: ``fold``: the fold helper's ``parts, channels`` scalars after ``A, D``.
ABI_FOLD = "fold"
#: ``implicit_wgrad_two_stage``: the implicit-GEMM block with the scratch
#: pair (``ws_ptr``, ``ws_bytes``) before ``ks``/``ks_count``. Its values
#: are every i32 after the operands' byte sizes; the scratch pointer is the
#: ``ws`` workspace.
ABI_IMPLICIT_TWO_STAGE = "implicit_wgrad_two_stage"
#: ``ws_reduce``: the Stage 2 fold's ``ws_ptr, dw_ptr`` then ``wg_M, wg_N,
#: ws_bytes, dw_bytes, groups``.
ABI_WS_REDUCE = "ws_reduce"
#: Buffers a step binds: the three graph tensors, and the scratch.
BUFFER_DY = "dy"
BUFFER_X = "x"
BUFFER_DW = "dw"
BUFFER_WS = "ws"

#: The integer codes :meth:`Gfx950ConvWgradPlan.serialize` writes.
PLAN_KERNEL_CODES = {KERNEL_SELF: 0, HELPER_VARIANT_FOLD: 1, HELPER_VARIANT_WS_REDUCE: 2}
PLAN_ABI_CODES = {
    ABI_IMPLICIT: 0,
    ABI_DIRECT_WGRAD: 1,
    ABI_FOLD: 2,
    ABI_IMPLICIT_TWO_STAGE: 3,
    ABI_WS_REDUCE: 4,
}
PLAN_BUFFER_CODES = {None: -1, BUFFER_DY: 0, BUFFER_X: 1, BUFFER_DW: 2, BUFFER_WS: 3}

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class Gfx950ConvWgradStep:
    """One kernel launch of a plan.

    ``kernel`` is :data:`KERNEL_SELF` or a helper variant; ``buffers`` the
    ``A``, ``B`` and ``D`` operands (a helper's ``B`` is ``None``). ``values`` are the by-value i32 arguments after the operands
    and their byte sizes, in ABI order; for implicit GEMM that is the problem
    values and, when ``split_k != 1``, ``split_k`` itself. The workgroup is
    ``(block, 1, 1)``.
    """

    kernel: str
    abi: str
    grid: tuple[int, int, int]
    block: int
    buffers: tuple[str, str | None, str]
    values: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Gfx950ConvWgradPlan:
    """Everything hipkernel:Gfx950ConvWgradAot does to run one instance on one
    problem, in order: zero dW when ``zero_dw`` and the ``ws`` workspace when
    ``zero_ws``, then launch ``steps``.

    ``launched`` is the problem the kernels see -- unfolded, the graph's own
    (wgrad's reduction spans the whole batch, so there is no batch-slicing
    view the way dgrad's is). ``workspaces`` are ``(name, offset, bytes)``
    regions of the one workspace (the fold scratch, when the instance needs
    one).
    """

    launched: Gfx950ConvWgradProblem
    steps: tuple[Gfx950ConvWgradStep, ...]
    workspaces: tuple[tuple[str, int, int], ...] = ()
    zero_dw: bool = False
    #: The resolved split-K degree (a runtime value, never baked into an
    #: instance's binary identity, so it rides on the plan/step rather than
    #: the spec -- see :func:`gfx950_conv_wgrad_aot_split_k`). 1 everywhere a
    #: launch does not split.
    split_k: int = 1
    zero_ws: bool = False

    @property
    def workspace_bytes(self) -> int:
        return max((offset + size for _, offset, size in self.workspaces), default=0)

    def buffer_bytes(self, name: str, dw_bytes: int = 2) -> int:
        """The byte size of buffer ``name``."""
        tensors = dict(
            zip(
                (BUFFER_DY, BUFFER_X, BUFFER_DW),
                self.launched.tensor_bytes(dw_bytes=dw_bytes),
            )
        )
        if name in tensors:
            return tensors[name]
        for region, _, size in self.workspaces:
            if region == name:
                return size
        raise KeyError(f"the plan has no buffer {name!r}")

    def serialize(self) -> list[int]:
        """The plan as one integer sequence, the native pack's format."""
        out = [
            int(self.zero_dw),
            int(self.zero_ws),
            self.split_k,
            self.workspace_bytes,
            len(self.workspaces),
        ]
        for name, offset, size in self.workspaces:
            out += [PLAN_BUFFER_CODES[name], offset, size]
        out.append(len(self.steps))
        for step in self.steps:
            out += [PLAN_KERNEL_CODES[step.kernel], PLAN_ABI_CODES[step.abi]]
            out += list(step.grid) + [step.block]
            out += [PLAN_BUFFER_CODES[b] for b in step.buffers]
            out.append(len(step.values))
            out += [value for _, value in step.values]
        return out

    def digest(self) -> int:
        """FNV-1a 64 over :meth:`serialize`, each value as 8 little-endian bytes."""
        h = _FNV_OFFSET
        for value in self.serialize():
            for byte in (value & _U64).to_bytes(8, "little"):
                h = ((h ^ byte) * _FNV_PRIME) & _U64
        return h


def gfx950_conv_wgrad_aot_step_signature(abi: str, dtype: str) -> list[dict]:
    """The launch signature of a step of ABI ``abi`` (``kernels.common.conv_abi``)."""
    if abi == ABI_IMPLICIT:
        return conv_args_signature(dtype, direction="wgrad")
    if abi == ABI_DIRECT_WGRAD:
        return conv_direct_args_signature(dtype, direction="wgrad")
    if abi == ABI_FOLD:
        from kernels.common.conv_direct_dw_wgrad import (
            DwWgradFoldSpec,
            dw_wgrad_fold_signature,
        )

        return dw_wgrad_fold_signature(DwWgradFoldSpec(taps=1))
    if abi == ABI_IMPLICIT_TWO_STAGE:
        return conv_args_signature(dtype, direction="wgrad", two_stage=True)
    if abi == ABI_WS_REDUCE:
        from kernels.common.conv_wgrad_workspace_reduce import (
            wgrad_reduce_aot_spec,
            wgrad_reduce_signature,
        )

        return wgrad_reduce_signature(wgrad_reduce_aot_spec(dtype, 1))
    raise ValueError(f"unknown step ABI {abi!r}")


def _fits_i32(values) -> bool:
    return all(-_INT32_MAX - 1 <= v <= _INT32_MAX for v in values)


def _launch_values(conv_args, bytes_: tuple[int, int, int], **extra) -> tuple:
    """``ConvArgs.to_launch_values`` past the operands and their sizes, as
    ``(name, value)`` pairs in ABI order -- the by-value ones: a two-stage
    launch's ``ws_ptr`` is the ``ws`` workspace, bound at launch."""
    values = conv_args.to_launch_values(0, 0, 0, *bytes_, **extra)
    names = [
        name
        for name, _ in conv_args.arg_names(two_stage="ws_ptr" in extra)
        if name != "ws_ptr"
    ]
    return tuple((name, values[name]) for name in names[6:])


def _grid_reason(grid, block: int, x_cap: int) -> str:
    """Why a ``grid`` of ``block``-thread workgroups cannot launch, or ``""``."""
    if not 1 <= block <= _MAX_BLOCK_THREADS:
        return f"block of {block} threads exceeds {_MAX_BLOCK_THREADS}"
    if min(grid) < 1:
        return f"empty grid {grid}"
    if grid[0] > x_cap:
        return f"grid x = {grid[0]} exceeds {x_cap}"
    for axis, extent in zip("yz", grid[1:]):
        if extent > _MAX_GRID_DIM:
            return f"grid {axis} = {extent} exceeds {_MAX_GRID_DIM}"
    return ""


def gfx950_conv_wgrad_aot_split_k(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> int:
    """The split-K degree an implicit-GEMM wgrad instance runs ``problem`` at.

    Exactly ``dispatch.grouped_convolution._wgrad_split_k``'s call, with the
    same argument derivation, so the two cannot drift apart. One field of the
    catalog's baked ``spec.split_k`` *is* read, despite the rest of this
    adapter treating it as a placeholder (see :class:`Gfx950ConvWgradAotSpec`'s
    docstring): ``split_k == 1`` means the packaged binary was built with
    ``WgradConvSpec._needs_atomic`` false, i.e. a plain direct-store epilogue
    with no atomic-accumulate code at all (``kernels.conv.aot_jobs``: "split_k=1
    (direct store, no atomics) is a structurally different kernel"). Handing
    such a binary a resolved degree > 1 makes every grid-z block overwrite
    ``dW`` instead of accumulating -- the same rule
    ``dispatch.grouped_convolution._wgrad_split_k`` applies.
    """
    if spec.split_k == 1:
        return 1
    # Raises ValueError for a two-stage instance on a problem it cannot
    # launch (one K tile, no gridDim.z budget, or a scratch past i32): it has
    # no unsplit form to fall back to.
    from kernels.common.conv_split_k import resolve_conv_split_k

    resolved = resolve_conv_split_k(
        "wgrad",
        problem.to_problem(),
        arch=_ARCH,
        dtype=spec.dtype,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
        block_size=spec.warp_m * spec.warp_n * spec.wave_size,
        group_merge=spec.group_merge,
        ws_replicas=spec.ws_replicas or _wgrad._DEFAULT_WS_REPLICAS,
        reduction="workspace" if spec.two_stage else "atomic",
        epilogue=None if spec.two_stage else spec.epilogue,
        allow_unsplit=not spec.two_stage,
    )
    return resolved.split_k


def _implicit_plan(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> tuple[Gfx950ConvWgradPlan | None, str]:
    """rocKE's offer for an implicit-GEMM wgrad binary
    (``identity_supports_problem``; for a two-stage one also a split-K degree
    its scratch can carry) plus the launch bounds the 32-bit launch needs,
    mirroring ``dispatch.grouped_convolution._grid``'s wgrad branch."""
    from kernels.conv.kernel_identity import identity_supports_problem

    identity = spec.to_identity()
    p = problem.to_problem()
    ok, why = identity_supports_problem(identity, p, llvm_flavor=identity.llvm_flavor)
    if not ok:
        return None, why
    threads = spec.warp_m * spec.warp_n * spec.wave_size
    try:
        split_k = gfx950_conv_wgrad_aot_split_k(spec, problem)
    except ValueError as exc:
        return None, f"two-stage wgrad: {exc}"
    args = ConvArgs.from_problem(
        p, direction="wgrad", tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=spec.tile_k
    )
    # Mirrors dispatch._grid's wgrad branch exactly: gx/gy come from the
    # unmerged problem (ConvArgs.grid()'s own z does not know about
    # group_merge, so z is recomputed here instead).
    gx, gy, _ = args.grid()
    grid_groups = max(1, problem.groups) // max(1, spec.group_merge)
    grid = (gx, gy, grid_groups * split_k)
    why = _grid_reason(grid, threads, _INT32_MAX)
    if why:
        return None, f"wgrad tile grid: {why}"
    tensors = problem.tensor_bytes(dw_bytes=2)
    if not spec.two_stage:
        values = _launch_values(args, tensors, split_k=split_k)
        if not _fits_i32(v for _, v in values):
            return None, "a launch value leaves the signed 32-bit kernel ABI"
        step = Gfx950ConvWgradStep(
            KERNEL_SELF,
            ABI_IMPLICIT,
            grid,
            threads,
            (BUFFER_DY, BUFFER_X, BUFFER_DW),
            values,
        )
        return (
            Gfx950ConvWgradPlan(
                launched=problem, steps=(step,), zero_dw=split_k > 1, split_k=split_k
            ),
            "ok",
        )

    # Two-stage: Stage 1 f32-atomic-adds its slices into the zeroed scratch,
    # Stage 2 folds the scratch's slabs into dW, written whole.
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        wgrad_workspace_nbytes,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        wgrad_reduce_grid,
    )

    replicas = spec.ws_replicas or _wgrad._DEFAULT_WS_REPLICAS
    ws_bytes = wgrad_workspace_nbytes(p, replicas)
    values = _launch_values(args, tensors, split_k=split_k, ws_ptr=0, ws_bytes=ws_bytes)
    if not _fits_i32(v for _, v in values):
        return None, "a launch value leaves the signed 32-bit kernel ABI"
    stage1 = Gfx950ConvWgradStep(
        KERNEL_SELF,
        ABI_IMPLICIT_TWO_STAGE,
        grid,
        threads,
        (BUFFER_DY, BUFFER_X, BUFFER_DW),
        values,
    )
    reduce_spec = WgradReduceSpec(
        problem=p, dtype_d=spec.dtype, groups=p.groups, ws_replicas=replicas
    )
    reduce_grid = wgrad_reduce_grid(reduce_spec)
    why = _grid_reason(reduce_grid, reduce_spec.block_size, _INT32_MAX)
    if why:
        return None, f"Stage 2: {why}"
    stage2 = Gfx950ConvWgradStep(
        HELPER_VARIANT_WS_REDUCE,
        ABI_WS_REDUCE,
        reduce_grid,
        reduce_spec.block_size,
        (BUFFER_WS, None, BUFFER_DW),
        (
            ("wg_M", reduce_spec.wg_M),
            ("wg_N", reduce_spec.wg_N),
            ("ws_bytes", ws_bytes),
            ("dw_bytes", tensors[2]),
            ("groups", p.groups),
        ),
    )
    return (
        Gfx950ConvWgradPlan(
            launched=problem,
            steps=(stage1, stage2),
            workspaces=((BUFFER_WS, 0, ws_bytes),),
            split_k=split_k,
            zero_ws=True,
        ),
        "ok",
    )


def _direct_geometry(rocke_spec) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """``(grid, block)`` of a built direct spec: the six families that carry
    ``launch_geometry()`` (as for ``DwWgradFoldSpec``), or ``DirectConvWgradSpec``
    (``direct_wgrad``), which ``conv_direct_grouped.direct_launch_geometry``
    handles by isinstance -- mirrors ``dispatch.grouped_convolution._direct_geometry``
    for the subset of spec classes wgrad actually has."""
    if hasattr(rocke_spec, "launch_geometry"):
        return rocke_spec.launch_geometry()
    return _direct.direct_launch_geometry(rocke_spec)


def _direct_plan(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> tuple[Gfx950ConvWgradPlan | None, str]:
    """rocKE's offer for a wgrad direct instance (``identity_supports_problem``,
    then ``direct_spec_for_problem``'s per-shape re-validation), its steps and
    scratch, plus the launch bounds rocKE does not check."""
    from kernels.conv.direct_variants import direct_spec_for_problem
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
    tensors = problem.tensor_bytes(dw_bytes=4)
    if max(tensors) > _INT32_MAX:
        return None, "a tensor exceeds the direct kernels' 2^31-byte operand addressing"
    grid, block = _direct_geometry(rocke_spec)
    why = _grid_reason(grid, block[0], _INT32_MAX)
    if why:
        return None, why
    args = ConvArgs.from_problem(direct, direction="wgrad")

    if not spec.is_folded:
        step = Gfx950ConvWgradStep(
            KERNEL_SELF,
            ABI_DIRECT_WGRAD,
            grid,
            block[0],
            (BUFFER_DY, BUFFER_X, BUFFER_DW),
            _launch_values(args, tensors),
        )
        return (
            Gfx950ConvWgradPlan(launched=problem, steps=(step,), zero_dw=True),
            "ok",
        )

    # The five folded families: the compute pass stores one partial slot per
    # partition into the scratch ``ws``; the fold sums them into dW (written
    # whole, no zeroing).
    from kernels.common.conv_direct_dw_wgrad import DW_WGRAD_FOLD_ARGS, DwWgradFoldSpec
    from kernels.conv.direct_variants import direct_wgrad_fold_knobs

    workspace_bytes = rocke_spec.workspace_bytes()
    conv_step = Gfx950ConvWgradStep(
        KERNEL_SELF,
        ABI_DIRECT_WGRAD,
        grid,
        block[0],
        (BUFFER_DY, BUFFER_X, BUFFER_WS),
        _launch_values(args, (tensors[0], tensors[1], workspace_bytes)),
    )
    fold_knobs = direct_wgrad_fold_knobs(spec.algorithm, direct.KH, direct.KW, direct.cpg)
    taps = fold_knobs["taps"]
    channels = (
        rocke_spec.fold_channels() if hasattr(rocke_spec, "fold_channels") else direct.groups
    )
    n_partitions = rocke_spec.n_partitions()
    fold_spec = DwWgradFoldSpec(taps=taps)
    fold_grid, fold_block = fold_spec.launch_geometry(channels)
    why = _grid_reason(fold_grid, fold_block[0], _INT32_MAX)
    if why:
        return None, f"fold: {why}"
    fold_step = Gfx950ConvWgradStep(
        HELPER_VARIANT_FOLD,
        ABI_FOLD,
        fold_grid,
        fold_block[0],
        (BUFFER_WS, None, BUFFER_DW),
        tuple(zip(DW_WGRAD_FOLD_ARGS, (n_partitions, channels))),
    )
    return (
        Gfx950ConvWgradPlan(
            launched=problem,
            steps=(conv_step, fold_step),
            workspaces=((BUFFER_WS, 0, workspace_bytes),),
            zero_dw=False,
        ),
        "ok",
    )


def gfx950_conv_wgrad_aot_plan(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> tuple[Gfx950ConvWgradPlan | None, str]:
    """The launch of instance ``spec`` on ``problem``, or ``(None, why)`` when
    the instance does not compute it.

    The served predicate is rocKE's own offer, as its dispatcher and sweeps
    make it: implicit GEMM through ``identity_supports_problem`` on the
    problem (unfolded -- wgrad has no patch/tilde form), the direct pipelines
    through ``identity_supports_problem`` and ``direct_spec_for_problem``. On
    top come the launch bounds of the 32-bit dispatch. Every instance that
    ``supports_gfx950_conv_wgrad_aot`` accepts, helpers aside, either plans
    here or is refused with a reason naming rocKE's own rule.
    """
    error = _problem_error(problem)
    if error:
        return None, error
    if problem.dtype != spec.dtype:
        return None, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_helper:
        return None, "a helper is launched only by its pipeline"
    try:
        if spec.is_direct:
            return _direct_plan(spec, problem)
        return _implicit_plan(spec, problem)
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def gfx950_conv_wgrad_aot_serves(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> tuple[bool, str]:
    """Whether instance ``spec`` computes ``problem`` (a plan exists)."""
    plan, why = gfx950_conv_wgrad_aot_plan(spec, problem)
    return plan is not None, why


def gfx950_conv_wgrad_aot_runtime_candidates(
    problem: Gfx950ConvWgradProblem, specs: Sequence[Gfx950ConvWgradAotSpec]
) -> list[Gfx950ConvWgradAotSpec]:
    """The instances of ``specs`` hipkernel:Gfx950ConvWgradAot offers for
    ``problem``: every serving one, in the order of ``specs``."""
    return [spec for spec in specs if gfx950_conv_wgrad_aot_serves(spec, problem)[0]]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _flavor_rank(flavor: str) -> int:
    return int(flavor.removeprefix("llvm"))


def _require_packaged_flavor() -> None:
    """Refuse to package under an LLVM older than the catalog's -- see
    ``builders.common.convolution_backward_data._require_packaged_flavor``,
    whose rationale applies unchanged."""
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    flavor = _resolve_llvm_flavor()
    if _flavor_rank(flavor) < _flavor_rank(PACKAGED_LLVM_FLAVOR):
        raise ValueError(
            f"hipkernel:Gfx950ConvWgradAot packages {PACKAGED_LLVM_FLAVOR} kernels, "
            f"but this process lowers to {flavor} (an older COMGR): set "
            f"HIPKERNELPROVIDER_ROCKE_COMGR_LIB to a COMGR of LLVM 22 or newer"
        )


def build_gfx950_conv_wgrad_aot(
    spec: Gfx950ConvWgradAotSpec,
    *,
    arch: str = _ARCH,
    problem: ConvProblem | None = None,
) -> KernelDef:
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an
    instance.

    Dispatches on the family: implicit GEMM through
    ``build_implicit_gemm_conv_wgrad`` (needs ``problem`` -- a wgrad
    implicit-GEMM binary bakes no geometry of its own to build against,
    unlike dgrad's tilde/patch probe), the direct and helper kernels through
    ``kernels.conv.direct_variants.select_direct_kernel_builder`` -- rocKE's
    own dispatch over its algorithms, rather than a per-variant mirror here.
    """
    ok, reason = supports_gfx950_conv_wgrad_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT wgrad instance: {reason}")
    _require_packaged_flavor()
    if spec.is_direct:
        from kernels.conv.direct_variants import select_direct_kernel_builder

        kernel = select_direct_kernel_builder(
            spec.algorithm, spec.to_direct_spec(), arch
        )
        if spec.is_helper:
            kernel = replace(kernel, name=spec.kernel_symbol_prefix())
        return kernel
    if spec.to_identity().stable_hash() != spec.instance_id:
        raise ValueError(
            f"instance {spec.instance_id} is not the identity its fields describe "
            f"(hash {spec.to_identity().stable_hash()}): the catalog and this adapter "
            f"disagree on a field"
        )
    if problem is None:
        raise ValueError(
            "building an implicit-GEMM wgrad instance needs a real problem: it "
            "bakes no geometry of its own"
        )
    return _wgrad.build_implicit_gemm_conv_wgrad(
        spec.to_instance_spec(problem), arch=arch
    )


# ---------------------------------------------------------------------------
# Profile hooks
# ---------------------------------------------------------------------------


def gfx950_conv_wgrad_aot_problem_for_request(
    request: ConvGroupedRequest,
) -> Gfx950ConvWgradProblem:
    """The problem hipkernel:Gfx950ConvWgradAot would match for ``request``.

    Instances are chosen per problem by capability, so there is no single spec
    to resolve; this is the graph side of that match (the profile's
    ``dispatch``). Raises ``ValueError`` outside the graph contract.
    """
    if not isinstance(request, ConvGroupedRequest):
        raise TypeError("expected ConvGroupedRequest")
    if request.arch != _ARCH:
        raise ValueError(f"this adapter requires arch={_ARCH}")
    if request.direction != "wgrad":
        raise ValueError("this engine serves backward-weight convolution only")
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
        raise ValueError("the packaged wgrad contract supports 2D convolution only")
    if getattr(request, "vec_size_c", None) is not None:
        raise ValueError("the packaged contract takes no vector-width override")
    problem = Gfx950ConvWgradProblem(
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


def supports_gfx950_conv_wgrad_aot_problem(
    problem: Gfx950ConvWgradProblem, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """The graph contract of hipkernel:Gfx950ConvWgradAot (the profile's predicate)."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(problem, Gfx950ConvWgradProblem):
        return False, "expected Gfx950ConvWgradProblem"
    error = _problem_error(problem)
    return (False, error) if error else (True, "ok")
