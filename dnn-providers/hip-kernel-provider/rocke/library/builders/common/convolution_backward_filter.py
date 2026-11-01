# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packaging adapter for the gfx950 shape-generic AOT backward-filter (wgrad)
convolution.

This is the wgrad sibling of :mod:`builders.common.convolution_backward_data`
(dgrad): one flat descriptor spec (:class:`Gfx950ConvWgradAotSpec`, every field
an ``int`` or ``str``) per rocKE ``KernelIdentity`` of the wgrad AOT instance
list (``instances_wgrad.txt``), built by dispatching to the exact same rocKE
entry points the instance list's own sweeps built it with -- so the packaged IR
is the swept kernel's bytes apart from its symbol name.

The contract is 2-D cross-correlation, NHWK dY, NHWC X, KYXC dW, uniform
fp16/bf16, fp32 accumulation, symmetric padding. Two families are packaged,
read off ``instances_wgrad.txt``'s family prefixes:

* **Implicit GEMM** (``kernel_family`` 0, prefix ``implicit_gemm``):
  :class:`~kernels.common.conv_implicit_gemm_wgrad.WgradConvSpec`, built by
  :func:`~kernels.common.conv_implicit_gemm_wgrad.build_implicit_gemm_conv_wgrad`.
* **Direct** (``kernel_family`` 1), every direct wgrad pipeline rocKE's shared
  :mod:`kernels.conv.direct_variants` dispatch already serves for every
  direction: ``direct_wgrad`` (:class:`DirectConvWgradSpec`, runtime channel
  counts), ``direct_wgrad_tiled``, ``direct_wgrad_dw`` (the LDS-ring
  depthwise form), ``direct_wgrad_dw_cols``, ``direct_wgrad_dw_mfma``,
  ``direct_wgrad_grouped_mfma``, ``direct_wgrad_gather`` and the
  ``direct_wgrad_fold`` helper (never serves a problem on its own).

Unlike dgrad, there is no third "MFMA weight-transform helper" family: wgrad's
MFMA direct pipelines compute dW directly, they do not transpose into a
forward kernel's problem. The two-stage implicit-GEMM pipeline's workspace
reduce (:mod:`kernels.common.conv_wgrad_workspace_reduce`) plays the
analogous "launched by its own pipeline, never served standalone" role and is
packaged as ``kernel_family`` 2 (``KERNEL_FAMILY_WGRAD_HELPER``) alongside the
fold helper, mirroring dgrad's helper family.

``instance_id`` is ``KernelIdentity.stable_hash()`` for an implicit-GEMM
instance; a direct or helper instance's identity is the one
:mod:`kernels.conv.direct_variants`/``direct_aot_jobs`` builds for it, as for
dgrad.

Scope note (first pass, see ``WGRAD_AOT_PROGRESS.md``): :func:`probe_problem`
derives the probe directly from the instance's stated geometry rather than
reconstructing rocKE's exact sweep probe through ``aot_jobs``/
``direct_aot_jobs`` internals (the dgrad adapter's ``probe_problem`` does the
latter). This is close enough to drive :func:`build_gfx950_conv_wgrad_aot` and
the capability/plan checks below, but exact sweep-probe parity (and therefore
byte-identity against a from-scratch rebuild) is deferred to Phase 3, when the
native C++ pack exists to cross-check against.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common import conv_implicit_gemm_wgrad as _wgrad
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.direct_aot_jobs import DirectCaps
from kernels.conv.direct_aot_jobs import probe_problem as _direct_probe_problem
from kernels.conv.direct_variants import (
    direct_spec_for_problem,
    select_direct_kernel_builder,
    validate_direct_spec,
)
from kernels.conv.kernel_identity import KernelIdentity, identity_supports_problem

_ARCH = "gfx950"
_INT32_MAX = (1 << 31) - 1
_WAVE_SIZE = 64
_DTYPES = ("fp16", "bf16")
#: ``KernelIdentity.stable_hash()`` is a SHA-1 hex digest.
_INSTANCE_ID_LENGTH = 40
#: The LLVM flavor the catalog's instances were swept, hashed and measured
#: under (every label of ``instances_wgrad.txt`` ends in it).
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

#: ``direct_variant`` values. Implicit GEMM carries "none".
DIRECT_VARIANT_NONE = "none"
DIRECT_VARIANT_PLAIN = "direct_wgrad"
DIRECT_VARIANT_TILED = "direct_wgrad_tiled"
DIRECT_VARIANT_DW_LDS = "direct_wgrad_dw"
DIRECT_VARIANT_DW_COLS = "direct_wgrad_dw_cols"
DIRECT_VARIANT_DW_MFMA = "direct_wgrad_dw_mfma"
DIRECT_VARIANT_GROUPED_MFMA = "direct_wgrad_grouped_mfma"
DIRECT_VARIANT_GATHER = "direct_wgrad_gather"
HELPER_VARIANT_FOLD = "direct_wgrad_fold"
HELPER_VARIANT_CAST = "direct_wgrad_cast"
HELPER_VARIANT_WORKSPACE_REDUCE = "wgrad_workspace_reduce"
_DIRECT_VARIANTS = (
    DIRECT_VARIANT_PLAIN,
    DIRECT_VARIANT_TILED,
    DIRECT_VARIANT_DW_LDS,
    DIRECT_VARIANT_DW_COLS,
    DIRECT_VARIANT_DW_MFMA,
    DIRECT_VARIANT_GROUPED_MFMA,
    DIRECT_VARIANT_GATHER,
)
_HELPER_VARIANTS = (HELPER_VARIANT_FOLD, HELPER_VARIANT_CAST, HELPER_VARIANT_WORKSPACE_REDUCE)

#: Each wgrad direct-family instance-name prefix (``instances_wgrad.txt``) and
#: the ``(kernel_family, direct_variant)`` it carries.
AOT_WGRAD_DIRECT_PREFIXES = {
    "direct_wgrad": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_PLAIN),
    "direct_wgrad_tiled": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_TILED),
    "direct_wgrad_dw": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DW_LDS),
    "direct_wgrad_dw_cols": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DW_COLS),
    "direct_wgrad_dw_mfma": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_DW_MFMA),
    "direct_wgrad_grouped_mfma": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_GROUPED_MFMA),
    "direct_wgrad_gather": (KERNEL_FAMILY_DIRECT, DIRECT_VARIANT_GATHER),
}

#: Helper-family instance-name prefixes (``wgrad_helper-<family>-...``, derived by
#: the sweep rather than listed in ``instances_wgrad.txt``), each with the
#: ``(kernel_family, direct_variant)`` it carries. Kept separate from
#: ``AOT_WGRAD_DIRECT_PREFIXES`` so ``packaged_direct_algorithms()`` does not pick
#: up a helper as a listed direct algorithm.
AOT_WGRAD_HELPER_PREFIXES = {
    HELPER_VARIANT_FOLD: (KERNEL_FAMILY_WGRAD_HELPER, HELPER_VARIANT_FOLD),
    HELPER_VARIANT_CAST: (KERNEL_FAMILY_WGRAD_HELPER, HELPER_VARIANT_CAST),
    HELPER_VARIANT_WORKSPACE_REDUCE: (KERNEL_FAMILY_WGRAD_HELPER, HELPER_VARIANT_WORKSPACE_REDUCE),
}

#: Per direct/helper variant, the knob names a parsed ``key=value`` instance
#: token carries, in the order the instance string states them. These are
#: exactly the dataclass fields of each variant's rocKE spec
#: (``DirectConvWgradSpec``, ``DirectWgradTiledSpec``, ``DwWgradLdsSpec``,
#: ``DwWgradColsSpec``, ``DwWgradMfmaSpec``, ``GroupedWgradMfmaSpec``,
#: ``WgradGatherSpec``).
AOT_WGRAD_DIRECT_KNOBS = {
    DIRECT_VARIANT_PLAIN: ("waves_c", "waves_k", "waves_q", "ho_per_block", "mfma_k"),
    DIRECT_VARIANT_TILED: (
        "waves_k",
        "waves_c",
        "wave_k16",
        "wave_c16",
        "rows_per_seg",
        "prefetch",
        "target_blocks",
        "row_unroll",
        "unfold_n",
        "q_groups",
        "pingpong",
    ),
    DIRECT_VARIANT_DW_LDS: (
        "lanes_c",
        "lanes_w",
        "cols_per_lane",
        "prefetch",
        "rows_per_seg",
        "target_blocks",
    ),
    DIRECT_VARIANT_DW_COLS: ("channels", "strip", "threads", "target_blocks"),
    DIRECT_VARIANT_DW_MFMA: (
        "q_tiles",
        "rows_per_chunk",
        "prefetch",
        "target_blocks",
        "unfold_n",
    ),
    DIRECT_VARIANT_GROUPED_MFMA: (
        "waves_q",
        "wave_q4",
        "rows_per_chunk",
        "prefetch",
        "target_blocks",
    ),
    DIRECT_VARIANT_GATHER: ("waves_k", "wave_k16", "prefetch", "rows_per_seg", "target_blocks"),
}
#: Union of every direct variant's knob fields, each defaulted to 0 (the KMD
#: default a variant that does not use a field leaves it at).
_KNOB_DEFAULTS = {
    name: 0
    for names in AOT_WGRAD_DIRECT_KNOBS.values()
    for name in names
}
#: The implicit-GEMM fields of a direct or helper instance hold exactly this
#: (mirrors dgrad's ``_DIRECT_PLACEHOLDERS``).
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
    "group_merge": 0,
    "prefetch_addr": 0,
    "split_k": 0,
}
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
#: Knobs rocKE spells as a bool; the descriptor holds 0/1.
_BOOL_KNOBS = ("pingpong",)
_PIPELINES = ("mem", "compv3", "compv4")
_EPILOGUES = ("default", "cshuffle")
_VECTOR_WIDTHS = (1, 2, 4, 8)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# The runtime problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950ConvWgradProblem:
    """One backward-filter problem as hipkernel:Gfx950ConvWgradAot matches it.

    dY is ``[N, Ho, Wo, K]`` NHWK, X is ``[N, Hi, Wi, C]`` NHWC and dW is
    ``[K, Y, X, C/groups]`` KYXC.
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
        """The implicit-GEMM view (``ConvProblem``)."""
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

    def to_direct_problem(self) -> DirectConvProblem:
        """The direct kernels' view: one stride and one padding (``sH``/``pH``)."""
        return DirectConvProblem(
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
        )


def _problem_error(problem: Gfx950ConvWgradProblem) -> str:
    if problem.dtype not in _DTYPES:
        return f"dtype must be one of {_DTYPES}, got {problem.dtype!r}"
    if problem.layout != "NHWC":
        return "the packaged wgrad contract requires dense NHWC storage"
    if problem.sH != problem.sW or problem.pH != problem.pW or problem.dH != problem.dW:
        return "the packaged wgrad contract compiles one stride/pad/dilation pair"
    if problem.C % problem.groups or problem.K % problem.groups:
        return "C and K must each divide evenly by groups"
    return ""


# ---------------------------------------------------------------------------
# The instance spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gfx950ConvWgradAotSpec:
    """One shape-generic backward-filter instance.

    Booleans are ints (0/1), so every field is a descriptor ``int`` or
    ``string``. The leading fields are the implicit-GEMM identity's
    (:class:`WgradConvSpec`'s tuning). The trailing ones describe a direct or
    helper instance: ``kernel_family``, ``direct_variant``, the compiled
    filter/stride/pad/dilation/channel counts and the variant's knobs
    (:data:`AOT_WGRAD_DIRECT_KNOBS`). A direct or helper instance holds the
    direct identity's placeholders in the implicit-GEMM fields; an
    implicit-GEMM instance leaves every trailing field at its KMD default.
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
    group_merge: int
    prefetch_addr: int
    split_k: int
    instance_id: str
    two_stage_flag: int = 0
    ws_replicas: int = 1
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
    waves_c: int = 0
    waves_k: int = 0
    waves_q: int = 0
    #: ``direct_wgrad_tiled``/``direct_wgrad_gather``-only; 0 means "omitted
    #: from the instance name, use the rocKE spec's own default" (2 for both),
    #: the same convention ``kernel_label.py`` uses to skip a token at its
    #: baseline.
    wave_k16: int = 0
    wave_c16: int = 0
    #: ``direct_wgrad_grouped_mfma``-only; 0 means the same (rocKE default 1).
    wave_q4: int = 0
    ho_per_block: int = 0
    mfma_k: int = 0
    rows_per_seg: int = 0
    rows_per_chunk: int = 0
    prefetch: int = 0
    target_blocks: int = 0
    row_unroll: int = 0
    unfold_n: int = 0
    q_groups: int = 0
    lanes_c: int = 0
    lanes_w: int = 0
    cols_per_lane: int = 0
    channels: int = 0
    strip: int = 0
    threads: int = 0
    q_tiles: int = 0
    pingpong: int = 0
    #: ``direct_wgrad_fold``-only: ``KH*KW*(cpg if the folded producer is
    #: per-channel else 1)``, read verbatim off the instance name's ``taps=N``
    #: token (``with_helpers()``/``direct_wgrad_fold_knobs()`` already computed
    #: it with the producer's real ``cpg``, before ``_helper_identity`` zeroes
    #: the helper's own ``cpg``/``kpg`` -- recomputing it here from this spec's
    #: ``filter_h*filter_w`` would silently drop that factor for the two
    #: per-channel variants, ``direct_wgrad_gather``/``direct_wgrad_grouped_mfma``).
    #: 0 for every other instance.
    taps: int = 0

    # ---- family ----

    @property
    def is_direct(self) -> bool:
        return self.kernel_family != KERNEL_FAMILY_IMPLICIT_GEMM

    @property
    def is_helper(self) -> bool:
        return self.kernel_family == KERNEL_FAMILY_WGRAD_HELPER

    @property
    def effective_group_merge(self) -> int:
        """The merged-group count (``Gm``) this instance was baked for.

        Read straight off the instance name's ``gm<N>`` token (``group_merge``
        is the real, lossless value ``kernel_label.py`` encodes it as -- see
        ``parse_wgrad_instance_name``). ``grp`` is a different field entirely
        (``KernelIdentity.grouped``, "this binary serves more than one
        group"), not a group-merge marker; conflating the two was the Phase 1
        bug documented in ``WGRAD_AOT_PROGRESS.md`` and fixed here.
        """
        return max(1, self.group_merge)

    @property
    def is_two_stage(self) -> bool:
        """Whether this instance's identity carries ``two_stage=True``.

        Read straight off the instance name's ``2stage`` token (see
        ``parse_wgrad_instance_name``), not derived from ``group_merge`` and
        ``split_k`` -- a plain (non-merged) split-K wgrad instance can also be
        two-stage. Not yet packaged by this adapter
        (``build_implicit_gemm_conv_wgrad_two_stage``, see
        ``WGRAD_AOT_PROGRESS.md``)."""
        return not self.is_direct and bool(self.two_stage_flag)

    @property
    def algorithm(self) -> str:
        """The rocKE ``direct_variants`` variant string (or ``implicit_gemm``)."""
        if not self.is_direct:
            return "implicit_gemm"
        return self.direct_variant

    def direct_knobs(self) -> dict:
        if not self.is_direct or self.is_helper:
            raise ValueError("only a non-helper direct instance has direct knobs")
        knobs = {}
        for name in AOT_WGRAD_DIRECT_KNOBS[self.direct_variant]:
            value = getattr(self, name)
            if value:
                knobs[name] = bool(value) if name in _BOOL_KNOBS else value
        return knobs

    def kernel_symbol_prefix(self) -> str:
        if self.is_helper:
            return f"hkp_conv_wgrad_helper_aot_gfx950_{self.instance_id[:24]}"
        if self.is_direct:
            return f"hkp_conv_wgrad_dw_aot_gfx950_{self.instance_id[:24]}"
        return f"hkp_conv_wgrad_aot_gfx950_{self.instance_id[:24]}"

    # ---- the sweep's build ----

    def to_identity(self) -> KernelIdentity:
        """The rocKE ``KernelIdentity`` this instance was packaged from
        (implicit-GEMM only; a direct/helper instance has no single stable
        hash across its knob dict)."""
        if self.is_direct:
            raise ValueError("a direct/helper instance has no implicit-GEMM identity")
        return KernelIdentity(
            direction="wgrad",
            algorithm="implicit_gemm",
            arch=_ARCH,
            dtype_a=self.dtype,
            dtype_b=self.dtype,
            dtype_d=self.dtype,
            wave_size=self.wave_size,
            llvm_flavor=PACKAGED_LLVM_FLAVOR,
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
            vector_size_a=self.vector_size_a,
            vector_size_b=self.vector_size_b,
            vector_size_c=self.vector_size_c,
        )

    def probe_problem(self):
        """The problem this instance is built/validated against.

        Direct/helper instances go through rocKE's own
        :func:`kernels.conv.direct_aot_jobs.probe_problem` -- the same probe
        the real AOT sweep and dispatcher (:mod:`dispatch.grouped_convolution`,
        via :func:`kernels.conv.direct_aot_jobs.direct_job_for_identity`) use
        -- rather than a synthetic stand-in, so a family with a narrow probe
        width requirement (e.g. ``direct_wgrad_dw_mfma``'s packed-image tile)
        gets the real one instead of failing it.
        """
        if self.is_direct:
            caps = DirectCaps(
                KH=self.filter_h or 3,
                PAD=self.pad_h,
                stride=self.stride_h or 1,
                cpg=self.cpg,
                kpg=self.kpg,
                dil_h=self.dilation_h or 1,
                dil_w=self.dilation_w or 1,
            )
            return _direct_probe_problem(
                caps, self.dtype, self.direct_variant, self.direct_knobs()
            )
        # kernels.conv.aot_jobs.CACHE_GROUPED = (True,): "there is no separate
        # ungrouped binary" -- every implicit-GEMM wgrad job the real sweep
        # builds has caps["grouped"] = True (_wgrad_jobs), and its own
        # _probe_problem() probes those with groups=2 unconditionally (not
        # only the group_merge>1 jobs). That is what bakes the per-group
        # address fold and the block_id_z group/slice decode into the IR
        # (conv_implicit_gemm_wgrad.py: ``_grouped = p_load.groups > 1 or
        # spec.group_merge > 1``). Probing here with groups=1 for every
        # non-group_merge instance -- the bug this comment replaces -- baked
        # _grouped=False into the whole catalog apart from the group_merge
        # jobs: every compiled binary silently ignored groups>1 at runtime
        # (every block_id_z CTA addressed group 0), which is exactly why a
        # groups=1 smoke-test graph passed and a groups=32 one did not.
        groups = 2
        channels = 64
        if self.group_merge > 1:
            # Depthwise (cpg == kpg == 1); groups a generous multiple of the
            # effective merge count so it divides cleanly.
            merge = self.effective_group_merge
            groups = merge * 8
            channels = groups
        problem = Gfx950ConvWgradProblem(
            N=2,
            Hi=56,
            Wi=56,
            C=channels,
            K=channels,
            Y=self.filter_h or 3,
            X=self.filter_w or 3,
            sH=self.stride_h or 1,
            sW=self.stride_w or 1,
            pH=self.pad_h,
            pW=self.pad_w,
            dH=self.dilation_h or 1,
            dW=self.dilation_w or 1,
            groups=groups,
            dtype=self.dtype,
        )
        return problem.to_problem()

    def to_instance_spec(self, problem: ConvProblem | None = None) -> _wgrad.WgradConvSpec:
        if self.is_direct:
            raise ValueError("a direct instance has no implicit-GEMM spec")
        return _wgrad.WgradConvSpec(
            problem=self.probe_problem() if problem is None else problem,
            name=self.kernel_symbol_prefix(),
            data=ConvDataSpec(dtype_a=self.dtype, dtype_b=self.dtype, dtype_d=self.dtype),
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
            vector_size_a=self.vector_size_a or None,
            vector_size_b=self.vector_size_b or None,
            vector_size_c=self.vector_size_c or None,
            lds_k_outer=bool(self.lds_k_outer),
            group_merge=self.effective_group_merge,
            prefetch_addr=bool(self.prefetch_addr),
            split_k=self.split_k or 1,
            two_stage=self.is_two_stage,
            ws_replicas=self.ws_replicas,
        )

    def to_direct_spec(self, problem: DirectConvProblem | None = None):
        """The rocKE builder spec for this direct/helper instance, as
        :func:`kernels.conv.direct_variants.make_direct_spec` makes it."""
        if not self.is_direct:
            raise ValueError("an implicit-GEMM instance has no direct spec")
        if self.direct_variant == HELPER_VARIANT_FOLD:
            from kernels.conv.direct_variants import make_direct_spec

            return make_direct_spec(
                HELPER_VARIANT_FOLD,
                None,
                {"taps": self.taps, "dtype_d": self.dtype},
            )
        if self.direct_variant == HELPER_VARIANT_CAST:
            from kernels.conv.direct_variants import make_direct_spec

            return make_direct_spec(
                HELPER_VARIANT_CAST, None, {"dtype_d": self.dtype}
            )
        if self.direct_variant == HELPER_VARIANT_WORKSPACE_REDUCE:
            raise ValueError(
                "the workspace-reduce helper is built directly, not through "
                "direct_variants.make_direct_spec"
            )
        from kernels.conv.direct_variants import make_direct_spec

        spec = make_direct_spec(
            self.direct_variant,
            problem if problem is not None else self.probe_problem(),
            self.direct_knobs(),
            wave_size=self.wave_size,
        )
        from dataclasses import replace as _replace

        return _replace(spec, name=self.kernel_symbol_prefix())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _direct_error(spec: Gfx950ConvWgradAotSpec) -> str:
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
    if spec.direct_variant == HELPER_VARIANT_FOLD:
        if spec.taps <= 0:
            return f"a {HELPER_VARIANT_FOLD} instance must set taps > 0, got {spec.taps!r}"
    elif spec.taps != 0:
        return f"only a {HELPER_VARIANT_FOLD} instance sets taps, got {spec.taps!r}"
    if spec.is_helper:
        return ""
    used = set(AOT_WGRAD_DIRECT_KNOBS.get(spec.direct_variant, ()))
    for name, default in _KNOB_DEFAULTS.items():
        value = getattr(spec, name)
        if type(value) is not int:
            return f"{name} must be an integer, got {value!r}"
        if name not in used and value != default:
            return (
                f"the {spec.direct_variant} variant does not use {name}; it must "
                f"be {default!r}, got {value!r}"
            )
    return ""


def supports_gfx950_conv_wgrad_aot(
    spec: Gfx950ConvWgradAotSpec, *, arch: str = _ARCH
) -> tuple[bool, str]:
    """Whether ``spec`` is a packageable instance; rocKE's validators decide last."""
    if arch != _ARCH:
        return False, f"this adapter requires arch={_ARCH}, got {arch!r}"
    if not isinstance(spec, Gfx950ConvWgradAotSpec):
        return False, "expected Gfx950ConvWgradAotSpec"
    if spec.dtype not in _DTYPES:
        return False, "the packaged wgrad contract supports fp16 or bf16 only"
    if spec.layout != "NHWC":
        return False, "the packaged wgrad contract requires dense NHWC storage"
    if (
        type(spec.kernel_family) is not int
        or spec.kernel_family not in _KERNEL_FAMILIES
    ):
        return False, (
            f"kernel_family must be one of {_KERNEL_FAMILIES} (implicit GEMM, "
            f"direct, wgrad helper), got {spec.kernel_family!r}"
        )
    if spec.is_direct:
        error = _direct_error(spec)
        if error:
            return False, error
        if spec.direct_variant == HELPER_VARIANT_WORKSPACE_REDUCE:
            return True, "ok"
        try:
            rocke_spec = spec.to_direct_spec()
            return validate_direct_spec(spec.algorithm, rocke_spec, arch)
        except (TypeError, ValueError, ZeroDivisionError, KeyError) as exc:
            return False, str(exc)
    if spec.pipeline not in _PIPELINES:
        return False, f"the packaged wgrad contract supports pipelines {_PIPELINES}"
    if spec.epilogue not in _EPILOGUES:
        return False, f"epilogue must be one of {_EPILOGUES}"
    for name in ("vector_size_a", "vector_size_b", "vector_size_c"):
        value = getattr(spec, name)
        if value and value not in _VECTOR_WIDTHS:
            return False, f"{name} must be 0 (unset) or one of {_VECTOR_WIDTHS}"
    try:
        instance = spec.to_instance_spec()
        instance.validate()
        ok, reason = _wgrad.is_valid_wgrad_spec(instance, arch=arch)
        if not ok:
            return False, reason
    except (TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
        return False, str(exc)
    return True, "ok"


# ---------------------------------------------------------------------------
# Serving and planning
# ---------------------------------------------------------------------------


def gfx950_conv_wgrad_aot_serves(
    spec: Gfx950ConvWgradAotSpec, problem: Gfx950ConvWgradProblem
) -> tuple[bool, str]:
    """Whether instance ``spec`` computes ``problem``.

    Implicit GEMM: rocKE's own offer, ``identity_supports_problem``. Direct:
    ``direct_spec_for_problem``, rocKE's shared capability match for every
    direct conv direction. A helper never serves a problem on its own.
    """
    error = _problem_error(problem)
    if error:
        return False, error
    if problem.dtype != spec.dtype:
        return False, f"dtype {problem.dtype} != instance dtype {spec.dtype}"
    if spec.is_helper:
        return False, "a helper kernel is launched only by its own pipeline"
    try:
        if spec.is_direct:
            _, why = direct_spec_for_problem(
                spec.direct_variant,
                problem.to_direct_problem(),
                spec.direct_knobs(),
                _ARCH,
                wave_size=spec.wave_size,
            )
            return why == "ok", why
        ok, why = identity_supports_problem(spec.to_identity(), problem.to_problem())
        return ok, why
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


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
    """Refuse to package under an LLVM older than the catalog's; see the dgrad
    adapter's identical check (``build_gfx950_conv_dgrad_aot``) for the why."""
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    flavor = _resolve_llvm_flavor()
    if _flavor_rank(flavor) < _flavor_rank(PACKAGED_LLVM_FLAVOR):
        raise ValueError(
            f"hipkernel:Gfx950ConvWgradAot packages {PACKAGED_LLVM_FLAVOR} kernels, "
            f"but this process lowers to {flavor} (an older COMGR): set "
            f"HIPKERNELPROVIDER_ROCKE_COMGR_LIB to a COMGR of LLVM 22 or newer"
        )


def build_gfx950_conv_wgrad_aot(spec: Gfx950ConvWgradAotSpec, *, arch: str = _ARCH):
    """The typed ``(spec, *, arch)`` entry point hkp_pack calls for an instance.

    Dispatches on the family: implicit GEMM through
    ``build_implicit_gemm_conv_wgrad``, every direct variant through rocKE's
    shared ``select_direct_kernel_builder`` (the same dispatch table serving
    fwd/dgrad/wgrad), the workspace-reduce helper through
    ``build_conv_wgrad_workspace_reduce`` directly.
    """
    ok, reason = supports_gfx950_conv_wgrad_aot(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid packaged AOT wgrad instance: {reason}")
    _require_packaged_flavor()
    if spec.is_direct:
        from dataclasses import replace as _replace

        if spec.direct_variant == HELPER_VARIANT_WORKSPACE_REDUCE:
            from kernels.common.conv_wgrad_workspace_reduce import (
                WgradReduceSpec,
                build_conv_wgrad_workspace_reduce,
            )

            reduce_spec = WgradReduceSpec(
                dtype=spec.dtype,
                name=spec.kernel_symbol_prefix(),
                **{k: v for k, v in spec.direct_knobs().items()},
            )
            return build_conv_wgrad_workspace_reduce(reduce_spec, arch=arch)
        rocke_spec = spec.to_direct_spec()
        kernel = select_direct_kernel_builder(spec.algorithm, rocke_spec, arch)
        if spec.direct_variant in (HELPER_VARIANT_FOLD, HELPER_VARIANT_CAST):
            kernel = _replace(kernel, name=spec.kernel_symbol_prefix())
        return kernel
    # A two-stage instance's Stage 1 (the f32-scratch-atomic partial-sums GEMM)
    # is an ordinary single-launch KernelDef -- the pipeline wrapper in
    # conv_implicit_gemm_wgrad_two_stage.py exists only to also *compile and
    # launch* Stage 2 for the runtime JIT path, which AOT packaging does not
    # need: each launch of a multi-launch KDP is packaged by its own call to
    # this adapter (see build_gfx950_conv_wgrad_aot_stage2 for Stage 2).
    return _wgrad.build_implicit_gemm_conv_wgrad(spec.to_instance_spec(), arch=arch)


def supports_gfx950_conv_wgrad_aot_stage2(spec: Gfx950ConvWgradAotSpec, *, arch: str = _ARCH):
    """Stage 2 shares Stage 1's validation: both launches describe the same
    two-stage instance, so whatever makes Stage 1 buildable makes Stage 2
    buildable too."""
    return supports_gfx950_conv_wgrad_aot(spec, arch=arch)


def build_gfx950_conv_wgrad_aot_stage2(spec: Gfx950ConvWgradAotSpec, *, arch: str = _ARCH):
    """Stage 2 of a two-stage wgrad instance: fold the ``ws_replicas`` f32
    scratch slabs Stage 1 wrote into the final ``dW`` (see
    ``conv_wgrad_workspace_reduce.build_conv_wgrad_workspace_reduce``).

    Takes the same ``spec`` as Stage 1 (``build_gfx950_conv_wgrad_aot``) --
    the packaged KDP's two launches share one descriptor spec, so Stage 2
    always folds exactly the problem/``ws_replicas`` Stage 1 was built for.
    """
    if not spec.is_two_stage:
        raise ValueError("spec is not a two-stage instance")
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
    )

    problem = spec.probe_problem()
    reduce_spec = WgradReduceSpec(
        problem=problem,
        dtype_d=spec.dtype,
        groups=problem.groups,
        ws_replicas=spec.ws_replicas,
        name=f"{spec.kernel_symbol_prefix()}_reduce",
    )
    return build_conv_wgrad_workspace_reduce(reduce_spec, arch=arch)


# ---------------------------------------------------------------------------
# Profile hooks
# ---------------------------------------------------------------------------


def gfx950_conv_wgrad_aot_problem_for_request(
    request: ConvGroupedRequest,
) -> Gfx950ConvWgradProblem:
    """The problem hipkernel:Gfx950ConvWgradAot would match for ``request``."""
    if not isinstance(request, ConvGroupedRequest):
        raise TypeError("expected ConvGroupedRequest")
    if request.arch != _ARCH:
        raise ValueError(f"this adapter requires arch={_ARCH}")
    if request.direction != "wgrad":
        raise ValueError("this engine serves backward-filter convolution only")
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


# ---------------------------------------------------------------------------
# instances_wgrad.txt parsing
# ---------------------------------------------------------------------------

_SHAPE_RE_FILTER = "f"
_WAVE_TOKEN = "wave"


def _parse_dims(token: str) -> tuple[int, ...]:
    return tuple(int(part) for part in token.split("x"))


def parse_wgrad_instance_name(name: str) -> Gfx950ConvWgradAotSpec:
    """Parse one line of ``instances_wgrad.txt`` into a
    :class:`Gfx950ConvWgradAotSpec`.

    Shape: ``wgrad-<family>-<dtype>-<dims>d-f<Y>x<X>-p<pH>x<pW>-s<sH>x<sW>-``
    ``d<dH>x<dW>-c<cpg>k<kpg>-<family-specific key=value tokens>-wave<N>-``
    ``<arch>-<llvm flavor>``, or for implicit_gemm
    ``wgrad-implicit_gemm-<dtype>-<dims>d-t<M>x<N>x<K>-w<Wm>x<Wn>-``
    ``a<Am>x<An>x<Ak>-v<Va>x<Vb>x<Vc>-<pipeline>-<epilogue>-[grp]-[kouter]-``
    ``[sk<N>]-[pfa]-wave<N>-<arch>-<llvm flavor>``.

    A derived helper (not listed in ``instances_wgrad.txt``, pulled in by
    ``with_helpers()``) is named the same way with ``wgrad_helper`` in place of
    ``wgrad``. Its ``AOT_WGRAD_DIRECT_KNOBS`` knob tokens are shape-generic and
    are not carried onto the spec -- a helper states no knobs -- except
    ``direct_wgrad_fold``'s ``taps=N``, which is carried onto the dedicated
    ``taps`` field (the fold's only real build-time parameter).
    """
    parts = name.split("-")
    if parts[0] not in ("wgrad", "wgrad_helper"):
        raise ValueError(f"not a wgrad instance name: {name!r}")
    is_helper_name = parts[0] == "wgrad_helper"
    llvm_flavor = parts[-1]
    arch = parts[-2]
    if arch != _ARCH:
        raise ValueError(f"instance {name!r} is not for {_ARCH}")
    wave_token = parts[-3]
    if not wave_token.startswith(_WAVE_TOKEN):
        raise ValueError(f"instance {name!r} has no wave-size token")
    wave_size = int(wave_token.removeprefix(_WAVE_TOKEN))
    dtype = parts[2]
    body = parts[3:-3]
    if not body or not body[0].endswith("d"):
        raise ValueError(f"instance {name!r} has no dimension token")
    ndims = int(body[0][:-1])
    if ndims != 2:
        raise ValueError(f"only 2D wgrad instances are supported, got {name!r}")
    rest = body[1:]

    if parts[1] == "implicit_gemm":
        tile_m, tile_n, tile_k = _parse_dims(rest[0][1:])
        warp_m, warp_n = _parse_dims(rest[1][1:])
        warp_tile_m, warp_tile_n, warp_tile_k = _parse_dims(rest[2][1:])
        vector_size_a, vector_size_b, vector_size_c = _parse_dims(rest[3][1:])
        pipeline = rest[4]
        epilogue = rest[5]
        flags = set(rest[6:])
        split_k = 1
        group_merge = 1
        ws_replicas = 1
        for flag in flags:
            if flag.startswith("sk"):
                split_k = int(flag[2:])
            elif flag.startswith("gm"):
                group_merge = int(flag[2:])
            elif flag.startswith("wsr"):
                ws_replicas = int(flag[3:])
        two_stage_flag = int("2stage" in flags)
        return Gfx950ConvWgradAotSpec(
            dtype=dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            wave_size=wave_size,
            pipeline=pipeline,
            epilogue=epilogue,
            vector_size_a=vector_size_a,
            vector_size_b=vector_size_b,
            vector_size_c=vector_size_c,
            lds_k_outer=int("kouter" in flags),
            group_merge=group_merge,
            prefetch_addr=int("pfa" in flags),
            split_k=split_k,
            instance_id="",
            two_stage_flag=two_stage_flag,
            ws_replicas=ws_replicas,
        )

    family = parts[1]
    prefixes = AOT_WGRAD_HELPER_PREFIXES if is_helper_name else AOT_WGRAD_DIRECT_PREFIXES
    if family not in prefixes:
        kind = "helper" if is_helper_name else "direct"
        raise ValueError(f"unknown wgrad {kind} family {family!r} in {name!r}")
    kernel_family, direct_variant = prefixes[family]
    filter_h = filter_w = 0
    stride_h = stride_w = 0
    pad_h = pad_w = 0
    dilation_h = dilation_w = 1
    cpg = kpg = 0
    knobs: dict[str, int] = {}
    for token in rest:
        if token.startswith("f") and "x" in token and token[1:].replace("x", "").isdigit():
            filter_h, filter_w = _parse_dims(token[1:])
        elif token.startswith("p") and "x" in token and token[1:].replace("x", "").isdigit():
            pad_h, pad_w = _parse_dims(token[1:])
        elif token.startswith("s") and "x" in token and token[1:].replace("x", "").isdigit():
            stride_h, stride_w = _parse_dims(token[1:])
        elif token.startswith("d") and "x" in token and token[1:].replace("x", "").isdigit():
            dilation_h, dilation_w = _parse_dims(token[1:])
        elif token[0] == "c" and "k" in token[1:]:
            c_part, k_part = token[1:].split("k", 1)
            cpg, kpg = int(c_part), int(k_part)
        elif token.startswith("dtype_d="):
            # direct_wgrad_cast's only real parameter -- already carried by this
            # instance's own dtype (parts[2]); the token exists so the derived
            # helper identity's label round-trips, not because the spec needs a
            # second copy of it.
            token_dtype_d = token.split("=", 1)[1]
            if token_dtype_d != dtype:
                raise ValueError(
                    f"dtype_d token {token!r} disagrees with dtype {dtype!r} in {name!r}"
                )
        elif "=" in token:
            key, value = token.split("=", 1)
            if value == "true":
                knobs[key] = 1
            elif value == "false":
                knobs[key] = 0
            else:
                knobs[key] = int(value)
        else:
            raise ValueError(f"unrecognised token {token!r} in instance {name!r}")
    return Gfx950ConvWgradAotSpec(
        dtype=dtype,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        wave_size=wave_size,
        pipeline="direct",
        epilogue="direct",
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        lds_k_outer=0,
        group_merge=0,
        prefetch_addr=0,
        split_k=0,
        instance_id="",
        kernel_family=kernel_family,
        direct_variant=direct_variant,
        filter_h=filter_h,
        filter_w=filter_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        cpg=cpg,
        kpg=kpg,
        taps=knobs.get("taps", 0) if is_helper_name else 0,
        **{k: v for k, v in knobs.items() if k in _KNOB_DEFAULTS},
    )


def load_wgrad_instances(path: str) -> list[Gfx950ConvWgradAotSpec]:
    """Parse every non-blank, non-comment line of an ``instances_wgrad.txt``-shaped
    file into specs. ``instance_id`` is left blank (set once the native pack
    assigns catalog ids in Phase 3)."""
    specs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(parse_wgrad_instance_name(line))
    return specs


def build_gfx950_conv_wgrad_aot_catalog(path: str, *, arch: str = _ARCH):
    """Parse and build every instance of an ``instances_wgrad.txt``-shaped file,
    returning ``(spec, kernel_def_or_None, error_or_None)`` triples."""
    results = []
    for spec in load_wgrad_instances(path):
        try:
            kernel = build_gfx950_conv_wgrad_aot(spec, arch=arch)
            if spec.is_two_stage:
                # Validate both launches of the packaged KDP, not just Stage 1.
                build_gfx950_conv_wgrad_aot_stage2(spec, arch=arch)
            results.append((spec, kernel, None))
        except (ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as exc:
            results.append((spec, None, f"{type(exc).__name__}: {exc}"))
    return results
