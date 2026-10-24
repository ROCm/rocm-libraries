# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One table over the direct convolution kernel variants.

A direct-conv AOT identity names its kernel by ``algorithm`` (the variant)
plus a JSON of tuning knobs. This module is the single place such a
``(variant, problem, knobs)`` triple turns into a kernel spec, is validated,
and is built -- used by the AOT cache build and run side
(``benchmarks.common.direct_kernel_sweep``) and by the grouped-convolution
dispatcher (``dispatch.grouped_convolution``), so the two cannot disagree on
which shapes a direct instance serves.
"""

from __future__ import annotations

from dataclasses import replace

__all__ = [
    "DIRECT_WGRAD_FOLD",
    "DIRECT_WGRAD_FOLDED",
    "DIRECT_DGRAD_REORGANIZE",
    "DIRECT_DGRAD_REORGANIZE_RTC",
    "DIRECT_DGRAD_TRANSPOSE",
    "DIRECT_DGRAD_TRANSPOSE_RTC",
    "DIRECT_DGRAD_WMMA",
    "DIRECT_GROUPED_HALO",
    "DIRECT_HALO_VARIANTS",
    "DIRECT_MFMA_DGRAD",
    "DIRECT_MFMA_HELPERS",
    "DIRECT_SPLIT_REDUCE",
    "direct_dgrad_weight_transforms",
    "direct_dgrad_workspaces",
    "direct_mfma_knobs_reason",
    "direct_spec_for_problem",
    "direct_wgrad_fold_knobs",
    "direct_uses_coalesced_weights",
    "make_direct_spec",
    "select_direct_kernel_builder",
    "validate_direct_spec",
]

# Weight-transform kernels of the MFMA dgrad pipeline, cached as their own
# entries and found by exact identity. The _RTC pair takes the channel counts
# as kernargs (one binary per filter and fold_k32).
DIRECT_DGRAD_TRANSPOSE = "direct_dgrad_transpose"
DIRECT_DGRAD_REORGANIZE = "direct_dgrad_reorganize"
DIRECT_DGRAD_TRANSPOSE_RTC = "direct_dgrad_transpose_rtc"
DIRECT_DGRAD_REORGANIZE_RTC = "direct_dgrad_reorganize_rtc"
DIRECT_MFMA_DGRAD = ("direct_grouped_dgrad_mfma", "direct_dgrad_mfma_rtc")
# Wave32 WMMA direct dgrad (stride 1, dense and grouped, channels runtime).
DIRECT_DGRAD_WMMA = "direct_dgrad_wmma"
# Single-kernel dgrad variants that run a forward kernel on the transposed
# problem (fused weights, no pre-pass): one step with the forward ABI.
DIRECT_DGRAD_FPROP_SINGLE = ("direct_grouped_dgrad_4c", "direct_grouped_dgrad_fused")
# Grouped forward through the non-grouped kernel's LDS-halo pipeline (runtime
# groups/channels, small cpg merged into one MFMA problem).
DIRECT_GROUPED_HALO = "direct_grouped_halo"
# Variants built by conv_direct_nongrouped (dense, and grouped via merging).
DIRECT_HALO_VARIANTS = ("direct_nongrouped", DIRECT_GROUPED_HALO)
# Second pass of a split-C non-grouped forward kernel: folds the fp32 partial
# sums into the output. Shape-independent (one binary per split_c and dtype),
# cached under its own direction and found by exact identity.
DIRECT_SPLIT_REDUCE = "direct_split_reduce"
# variant -> (transpose helper, reorganize helper)
DIRECT_MFMA_HELPERS = {
    "direct_grouped_dgrad_mfma": (DIRECT_DGRAD_TRANSPOSE, DIRECT_DGRAD_REORGANIZE),
    "direct_dgrad_mfma_rtc": (DIRECT_DGRAD_TRANSPOSE_RTC, DIRECT_DGRAD_REORGANIZE_RTC),
}
# Partial-slot fold of the wgrad kernels that store one slot per partition
# instead of atomically adding into dW; cached as its own entry, keyed by the
# number of dW elements per output channel it folds (``taps``).
DIRECT_WGRAD_FOLD = "direct_wgrad_fold"
DIRECT_WGRAD_FOLDED = (
    "direct_wgrad_dw",
    "direct_wgrad_dw_cols",
    "direct_wgrad_gather",
)
_HELPERS = (
    DIRECT_DGRAD_TRANSPOSE,
    DIRECT_DGRAD_REORGANIZE,
    DIRECT_DGRAD_TRANSPOSE_RTC,
    DIRECT_DGRAD_REORGANIZE_RTC,
    DIRECT_WGRAD_FOLD,
)
# Register budget for the MFMA dgrad fprop pass's preloaded weights and
# accumulators (see direct_mfma_knobs_reason): three quarters of the 512-VGPR
# file, leaving the rest for the LDS fragments and addressing.
_MFMA_PRELOAD_VGPR_BUDGET = 384
# The same estimate's cut for a runtime-channel preloading binary whose wave
# covers two 16-column Q subtiles (block_q 32): above it the binary computed a
# wrong dX on gfx950 (see direct_mfma_knobs_reason).
_RTC_WIDE_Q_PRELOAD_VGPR_BUDGET = 216


# Variants whose kernel is written for any wave size (scalar FMA, one lane per
# channel or per channel and column): their spec takes the target's wave size,
# so the binary matches the wave size the identity records (and the tiled and
# column-streamed validators require it). Every other variant is a wave64 MFMA
# kernel and keeps its spec default, which its validator checks against the
# target.
DIRECT_WAVE_PARAMETRIC = frozenset(
    {
        "direct_depthwise",
        "direct_depthwise_spatial",
        "direct_depthwise_tiled",
        "direct_depthwise_col",
    }
)


def make_direct_spec(variant: str, problem, knobs: dict, wave_size: int = 64):
    """The spec ``variant`` builds for ``problem``.

    For the MFMA dgrad pipeline this is the spec of its compute pass (a
    forward kernel on the transposed problem); the weight transforms are
    derived from the original problem separately. ``wave_size`` is the
    target's; only the :data:`DIRECT_WAVE_PARAMETRIC` variants take it.
    """
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    if variant in DIRECT_WAVE_PARAMETRIC:
        if "wave_size" in knobs and knobs["wave_size"] != wave_size:
            raise ValueError(
                f"knob wave_size {knobs['wave_size']!r} conflicts with wave_size "
                f"{wave_size!r}"
            )
        knobs = dict(knobs, wave_size=wave_size)

    if variant == "direct_grouped":
        return dc.DirectConvSpec(problem=problem, name="rocke_direct_conv", **knobs)
    if variant == DIRECT_DGRAD_WMMA:
        from kernels.common.conv_direct_dgrad_wmma import DirectDgradWmmaSpec

        return DirectDgradWmmaSpec(
            problem=problem, name="rocke_direct_dgrad_wmma", **knobs
        )
    if variant == "direct_nongrouped":
        return dn.DirectNongroupedConvSpec(
            problem=problem, name="rocke_direct_conv_nongrouped", **knobs
        )
    if variant == DIRECT_GROUPED_HALO:
        return dn.DirectNongroupedConvSpec(
            problem=problem, name="rocke_direct_conv_grouped_halo", **knobs
        )
    if variant == "direct_depthwise":
        return dc.DirectDepthwiseSpec(
            problem=problem, name="rocke_direct_depthwise", **knobs
        )
    if variant == "direct_depthwise_spatial":
        return dc.DirectDepthwiseSpatialSpec(
            problem=problem, name="rocke_direct_depthwise_spatial", **knobs
        )
    if variant == "direct_depthwise_tiled":
        return dc.DirectDepthwiseTiledSpec(
            problem=problem, name="rocke_direct_depthwise_tiled", **knobs
        )
    if variant == "direct_depthwise_col":
        # The col spec carries its own element type; a binary serves only the
        # dtype it was built for, so it follows the problem's.
        return dc.DirectDepthwiseColSpec(
            problem=problem,
            name="rocke_direct_depthwise_col",
            dtype=problem.dtype,
            **knobs,
        )
    if variant == "direct_depthwise_dgrad":
        return dc.DirectDepthwiseDgradStreamSpec(
            problem=problem, name="rocke_direct_dw_dgrad", **knobs
        )
    if variant == "direct_depthwise_dgrad_win":
        return dc.DirectDepthwiseDgradWindowedSpec(
            problem=problem, name="rocke_direct_dw_dgrad_win", **knobs
        )
    if variant == "direct_grouped_dgrad":
        return dc.DirectConvDgradSpec(
            problem=problem, name="rocke_direct_dgrad", **knobs
        )
    if variant == "direct_grouped_dgrad_4c":
        # The forward 4c kernel on the transposed problem (no problem check
        # here: plan_for re-validates the spec on the real shape).
        stage_rows = knobs["stage_rows"]
        return dc.DirectConv4cSpec(
            problem=dc.dgrad_4c_transposed_problem(problem),
            name="rocke_direct_dgrad_4c",
            block_q=knobs["block_q"],
            block_groups=knobs["block_groups"],
            dgrad_fused_weights=True,
            dgrad_weights_lds=knobs["dgrad_weights_lds"],
            stage_rows=stage_rows,
            waves_q=knobs["block_q"] // 4 if stage_rows else 1,
        )
    if variant == "direct_grouped_dgrad_fused":
        # The generic forward kernel on the transposed problem, reading the
        # original weights (no problem check here: plan_for re-validates the
        # spec on the real shape). Every tile carries the row guard: a tile
        # may reach past the image on any problem whose height it does not
        # divide.
        spec = dc.make_dgrad_fprop_spec(
            problem,
            dgrad_fused_weights=True,
            row_guard=True,
            **knobs,
        )
        return replace(spec, name="rocke_direct_dgrad_fused")
    if variant == "direct_grouped_dgrad_mfma":
        fold_k32 = knobs["fold_k32"]
        spec = dc.make_dgrad_fprop_spec(
            problem, **{k: v for k, v in knobs.items() if k != "fold_k32"}
        )
        return replace(spec, fold_k32=fold_k32) if fold_k32 else spec
    if variant == "direct_dgrad_mfma_rtc":
        fold_k32 = knobs["fold_k32"]
        spec = dc.make_dgrad_fprop_spec(
            problem,
            runtime_channels=True,
            **{k: v for k, v in knobs.items() if k != "fold_k32"},
        )
        return replace(spec, fold_k32=fold_k32) if fold_k32 else spec
    if variant == "direct_wgrad":
        return dc.DirectConvWgradSpec(
            problem=problem,
            name="rocke_direct_conv_wgrad",
            runtime_channels=True,
            **knobs,
        )
    if variant == "direct_wgrad_tiled":
        from kernels.common.conv_direct_wgrad_tiled import DirectWgradTiledSpec

        return DirectWgradTiledSpec(
            problem=problem, name="rocke_direct_wgrad_tiled", **knobs
        )
    if variant == "direct_wgrad_dw":
        from kernels.common.conv_direct_dw_wgrad import DwWgradLdsSpec

        return DwWgradLdsSpec(problem=problem, name="rocke_direct_wgrad_dw", **knobs)
    if variant == "direct_wgrad_dw_cols":
        from kernels.common.conv_direct_dw_wgrad import DwWgradColsSpec

        # ``channels`` names the C the binary bakes; the spec reads it off
        # the problem, and direct_spec_for_problem holds the two equal.
        return DwWgradColsSpec(
            problem=problem,
            name="rocke_direct_wgrad_dw_cols",
            **{k: v for k, v in knobs.items() if k != "channels"},
        )
    if variant == "direct_wgrad_gather":
        from kernels.common.conv_direct_wgrad_gather import WgradGatherSpec

        return WgradGatherSpec(
            problem=problem, name="rocke_direct_wgrad_gather", **knobs
        )
    if variant == DIRECT_WGRAD_FOLD:
        from kernels.common.conv_direct_dw_wgrad import DwWgradFoldSpec

        return DwWgradFoldSpec(name="rocke_direct_wgrad_fold", **knobs)
    if variant == DIRECT_SPLIT_REDUCE:
        return dn.DirectSplitReduceSpec(dtype=problem.dtype, **knobs)
    if variant == DIRECT_DGRAD_TRANSPOSE:
        return dc.DirectTransposeWeightsDgradSpec(problem=problem)
    if variant == DIRECT_DGRAD_REORGANIZE:
        return dc.DirectReorganizeWeightsSpec(problem=problem, **knobs)
    if variant == DIRECT_DGRAD_TRANSPOSE_RTC:
        return dc.DirectTransposeWeightsDgradSpec(
            problem=problem, runtime_channels=True
        )
    if variant == DIRECT_DGRAD_REORGANIZE_RTC:
        return dc.DirectReorganizeWeightsSpec(
            problem=problem, runtime_channels=True, **knobs
        )
    raise ValueError(f"unknown direct conv variant {variant!r}")


def dgrad_rtc_vgprs(knobs: dict, KH: int, KW: int) -> int:
    """Registers per lane the compute pass of a runtime-channel dgrad binary
    holds in its weights (preloading splits only) and accumulators."""
    load_vec = 8 if knobs["fold_k32"] else 4
    acc = (knobs["block_q"] // knobs["waves_q"] // 16) * knobs["m_tiles"] * KH * 4
    if knobs["waves_k"] == 1:
        return acc
    per_wave = knobs["k_atoms"] // knobs["waves_k"]
    return acc + KH * KW * per_wave * knobs["m_tiles"] * (load_vec // 2)


def direct_mfma_knobs_reason(problem, knobs: dict) -> str | None:
    """Constraints of the MFMA dgrad pipeline that its spec cannot see.

    The fprop spec validates K-atoms as ``ceil(cpg/16)``; with ``fold_k32``
    the pass actually runs ``kpg/32`` atoms of 32, and each wave along K
    takes an equal slice, so the split has to divide that count instead.

    With ``waves_k > 1`` the fprop pass preloads its whole weight slice into
    registers and keeps ``KH`` accumulator slots per (Q-subtile, M-tile) on
    top. Past :data:`_MFMA_PRELOAD_VGPR_BUDGET` the kernel sits at the
    512-VGPR ceiling and spills, and comgr's VGPR-to-AGPR spilling there has
    produced a kernel that silently computes the wrong dX (correct at -O1
    and with ``-amdgpu-spill-vgpr-to-agpr=0``). No spec under the budget
    spills, so the cut costs no kernel that runs out of registers cleanly.

    The runtime-channel pipeline (knobs ``k_atoms`` / ``m_tiles``) has a
    narrower cut where a wave covers two Q subtiles (block_q 32): a
    verification of every preloading binary of its gfx950 grid, both dtypes,
    on the dense problem of its tile counts found wrong dX only there, on the
    binaries above :data:`_RTC_WIDE_Q_PRELOAD_VGPR_BUDGET` -- deterministic,
    whether or not the binary spills, and gone at -O1 for most of them, so a
    code generation fault rather than a wrong emission. The block_q 16 binary
    of the same tile counts was correct everywhere and serves the same
    problems.
    """
    if knobs["fold_k32"]:
        if problem.kpg % 32:
            return f"fold_k32 needs kpg % 32 == 0 (got kpg={problem.kpg})"
        n_atoms = problem.kpg // 32
    else:
        n_atoms = (problem.kpg + 15) // 16
    if n_atoms == 0 or n_atoms % knobs["waves_k"]:
        return f"{n_atoms} K-atoms do not split over waves_k={knobs['waves_k']}"
    if knobs["waves_k"] > 1:
        # The fprop pass runs on the transposed problem: it reduces over this
        # kpg and its M-tiles cover this cpg.
        n_m_tiles = (problem.cpg + 15) // 16
        frag_vgprs = 4 if knobs["fold_k32"] else 2
        weights = (
            problem.KH
            * problem.KW
            * (n_atoms // knobs["waves_k"])
            * n_m_tiles
            * frag_vgprs
        )
        q_subtiles = knobs["block_q"] // knobs.get("waves_q", 1) // 16
        accumulators = q_subtiles * n_m_tiles * problem.KH * 4
        if weights + accumulators > _MFMA_PRELOAD_VGPR_BUDGET:
            return (
                f"preloaded weights ({weights}) + accumulators ({accumulators}) "
                f"VGPRs > {_MFMA_PRELOAD_VGPR_BUDGET}: the kernel would spill"
            )
        if "k_atoms" in knobs and q_subtiles > 1:
            vgprs = dgrad_rtc_vgprs(knobs, problem.KH, problem.KW)
            if vgprs > _RTC_WIDE_Q_PRELOAD_VGPR_BUDGET:
                return (
                    f"runtime-channel preload of {vgprs} VGPRs over "
                    f"{q_subtiles} Q subtiles > "
                    f"{_RTC_WIDE_Q_PRELOAD_VGPR_BUDGET}: wrong dX on gfx950 "
                    "(take block_q 16)"
                )
    return None


def direct_uses_coalesced_weights(knobs: dict) -> bool:
    """Does the MFMA dgrad's fprop pass read the reorganized weight layout?

    Only when K is split over waves or walked by the runtime K loop; otherwise
    it reads the plain transposed weights and the reorganize pass is skipped.
    """
    return knobs["waves_k"] > 1 or knobs["runtime_k_loop"]


def direct_dgrad_weight_transforms(
    knobs: dict, variant: str = "direct_grouped_dgrad_mfma"
) -> tuple[tuple[str, dict, str, str], ...]:
    """The weight transforms the MFMA dgrad ``variant`` runs before its fprop pass.

    ``(variant, knobs, source, destination)`` per kernel, in launch order, over
    the buffers ``w`` (the KYXC filter), ``ws_t`` and ``ws_coa`` (the scratch
    of :func:`direct_dgrad_workspaces`). The fprop pass reads the last
    destination.
    """
    transpose, reorganize = DIRECT_MFMA_HELPERS[variant]
    passes = [(transpose, {}, "w", "ws_t")]
    if direct_uses_coalesced_weights(knobs):
        passes.append((reorganize, {"fold_k32": knobs["fold_k32"]}, "ws_t", "ws_coa"))
    return tuple(passes)


def direct_dgrad_workspaces(problem, knobs: dict) -> dict[str, int]:
    """Scratch buffer -> bytes of the MFMA dgrad's weight transforms."""
    from kernels.common import conv_direct_grouped as dc

    workspaces = {"ws_t": dc.direct_dgrad_workspace_bytes(problem)}
    if direct_uses_coalesced_weights(knobs):
        workspaces["ws_coa"] = dc.direct_dgrad_coalesced_workspace_bytes(
            problem, fold_k32=knobs["fold_k32"]
        )
    return workspaces


def validate_direct_spec(variant: str, spec, arch: str) -> tuple[bool, str]:
    """``(ok, reason)`` for ``spec`` on ``arch``, from the kernel's own validator."""
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    try:
        if variant in DIRECT_HALO_VARIANTS:
            return dn.is_valid_nongrouped_spec(spec, arch=arch)
        if (
            variant
            in ("direct_grouped", "direct_grouped_dgrad_fused") + DIRECT_MFMA_DGRAD
        ):
            spec.validate()
            return dc.is_valid_spec(spec, arch=arch)
        if variant == "direct_depthwise":
            return dc.is_valid_depthwise_spec(spec, arch=arch)
        if variant == "direct_depthwise_spatial":
            return dc.is_valid_depthwise_spatial_spec(spec, arch=arch)
        if variant == "direct_depthwise_tiled":
            return dc.is_valid_depthwise_tiled_spec(spec, arch=arch)
        if variant == "direct_depthwise_col":
            spec.validate()
            return dc.is_valid_depthwise_col_spec(spec, arch=arch)
        if variant == "direct_depthwise_dgrad":
            return dc.is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
        if variant == "direct_depthwise_dgrad_win":
            return dc.is_valid_depthwise_dgrad_win_spec(spec, arch=arch)
        if variant == "direct_grouped_dgrad":
            return dc.is_valid_dgrad_spec(spec, arch=arch)
        if variant == DIRECT_DGRAD_WMMA:
            from kernels.common.conv_direct_dgrad_wmma import is_valid_dgrad_wmma_spec

            return is_valid_dgrad_wmma_spec(spec, arch=arch)
        if variant == "direct_grouped_dgrad_4c":
            spec.validate()
            return dc.is_valid_spec_4c(spec, arch=arch)
        if variant == "direct_wgrad":
            spec.validate()
            return dc.is_valid_wgrad_spec(spec, arch=arch)
        if variant == "direct_wgrad_tiled":
            from kernels.common.conv_direct_wgrad_tiled import is_valid_wgrad_tiled_spec

            return is_valid_wgrad_tiled_spec(spec, arch=arch)
        if variant == "direct_wgrad_dw":
            from kernels.common.conv_direct_dw_wgrad import is_valid_dw_wgrad_lds_spec

            return is_valid_dw_wgrad_lds_spec(spec, arch=arch)
        if variant == "direct_wgrad_dw_cols":
            from kernels.common.conv_direct_dw_wgrad import is_valid_dw_wgrad_cols_spec

            return is_valid_dw_wgrad_cols_spec(spec, arch=arch)
        if variant == "direct_wgrad_gather":
            from kernels.common.conv_direct_wgrad_gather import (
                is_valid_wgrad_gather_spec,
            )

            return is_valid_wgrad_gather_spec(spec, arch=arch)
        if variant in _HELPERS:
            return True, "ok"
        if variant == DIRECT_SPLIT_REDUCE:
            spec.validate()
            return True, "ok"
    except ValueError as e:
        return False, str(e)
    raise ValueError(f"unknown direct conv variant {variant!r}")


def select_direct_kernel_builder(variant: str, spec, arch: str):
    """The ``KernelDef`` of ``variant`` for ``spec`` (validated by the caller).

    Named away from the ``build_*`` prefix on purpose: this is a dispatch
    helper over the real per-variant builders below, not a builder itself --
    ``test_builder_signature_contract.py`` scans every ``build_*`` top-level
    function for the ``(spec, arch)`` builder signature, and this one takes
    an extra ``variant``.
    """
    from kernels.common import conv_direct_dgrad_wmma as dwm
    from kernels.common import conv_direct_dw_wgrad as dw
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn
    from kernels.common import conv_direct_wgrad_gather as wg
    from kernels.common import conv_direct_wgrad_tiled as wt

    builders = {
        DIRECT_DGRAD_WMMA: dwm.build_direct_dgrad_wmma,
        "direct_wgrad_tiled": wt.build_direct_wgrad_tiled,
        "direct_wgrad_dw": dw.build_dw_wgrad_lds,
        "direct_wgrad_dw_cols": dw.build_dw_wgrad_cols,
        "direct_wgrad_gather": wg.build_wgrad_gather,
        DIRECT_WGRAD_FOLD: dw.build_dw_wgrad_fold,
        "direct_grouped": dc.build_direct_conv,
        "direct_nongrouped": dn.build_direct_conv_nongrouped,
        DIRECT_GROUPED_HALO: dn.build_direct_conv_nongrouped,
        "direct_grouped_dgrad_mfma": dc.build_direct_conv,
        "direct_dgrad_mfma_rtc": dc.build_direct_conv,
        "direct_depthwise": dc.build_direct_depthwise,
        "direct_depthwise_spatial": dc.build_direct_depthwise_spatial,
        "direct_depthwise_tiled": dc.build_direct_depthwise_tiled,
        "direct_depthwise_col": dc.build_direct_depthwise_col,
        "direct_depthwise_dgrad": dc.build_direct_depthwise_dgrad_streaming,
        "direct_depthwise_dgrad_win": dc.build_direct_depthwise_dgrad_windowed,
        "direct_grouped_dgrad": dc.build_direct_conv_dgrad,
        "direct_grouped_dgrad_4c": dc.build_direct_conv_4c,
        "direct_grouped_dgrad_fused": dc.build_direct_conv,
        "direct_wgrad": dc.build_direct_conv_wgrad,
        DIRECT_DGRAD_TRANSPOSE: dc.build_direct_transpose_weights_dgrad,
        DIRECT_DGRAD_REORGANIZE: dc.build_direct_reorganize_weights,
        DIRECT_DGRAD_TRANSPOSE_RTC: dc.build_direct_transpose_weights_dgrad,
        DIRECT_DGRAD_REORGANIZE_RTC: dc.build_direct_reorganize_weights,
        DIRECT_SPLIT_REDUCE: dn.build_direct_conv_split_reduce,
    }
    return builders[variant](spec, arch=arch)


def direct_entry_wave_reason(
    variant: str, identity_wave_size: int, meta: dict
) -> str | None:
    """Why a cached binary of ``variant`` cannot launch at its identity's wave
    size, or ``None`` when it can.

    The launch geometry of a :data:`DIRECT_WAVE_PARAMETRIC` kernel follows the
    spec's ``wave_size`` (block size, channels per block), so it has to be the
    one the binary was built with. The build records it as ``spec_wave_size``;
    an entry without it predates the record and was built at the wave64 spec
    default, whatever wave size its identity says.
    """
    if variant not in DIRECT_WAVE_PARAMETRIC:
        return None
    built = meta.get("spec_wave_size", 64)
    if built != identity_wave_size:
        return (
            f"cache entry was built at wave_size {built}, its identity says "
            f"{identity_wave_size} (rebuild the cache with --compile-all)"
        )
    return None


def direct_spec_for_problem(
    variant: str, problem, knobs: dict, arch: str, wave_size: int = 64
) -> tuple[object | None, str]:
    """``(spec, "ok")`` when a binary of ``variant``/``knobs`` serves ``problem``.

    The capability match
    (:func:`kernels.conv.kernel_identity.identity_supports_problem`)
    is necessary but not sufficient: tuning knobs have shape rules of their
    own (groups divisible by block_groups, a persistent grid's cell budget,
    ...). The spec is rebuilt on the real problem and run through the kernel's
    own validator, so a binary is only offered where a fresh build would accept
    the shape too. The returned spec is the one its launch geometry comes from.
    ``wave_size`` is the binary's (its identity's). Returns ``(None, why)``
    otherwise.
    """
    if variant in DIRECT_MFMA_DGRAD:
        why = direct_mfma_knobs_reason(problem, knobs)
        if why:
            return None, why
    if variant == "direct_wgrad_dw_cols" and knobs["channels"] != problem.groups:
        return None, (
            f"binary bakes C={knobs['channels']}, problem has C={problem.groups}"
        )
    spec = make_direct_spec(variant, problem, knobs, wave_size)
    ok, why = validate_direct_spec(variant, spec, arch)
    if not ok:
        return None, why
    if variant == "direct_wgrad":
        # As the JIT sweep: no C tile wider than the problem's channels, so no
        # wave stages and multiplies only channels the epilogue then drops.
        c_tiles = -(-problem.cpg // spec.wave_tile_c)
        if spec.waves_c > c_tiles:
            return None, (
                f"waves_c={spec.waves_c} exceeds the {c_tiles} channel tile(s) "
                f"of cpg={problem.cpg}"
            )
    if variant in DIRECT_HALO_VARIANTS:
        # A binary serves every width, but only the widths a sweep for this
        # shape would try are offered, so no binary spends MFMA work past the
        # image edge when one that does not exists.
        from kernels.common.conv_direct_nongrouped import tile_w_candidates

        widths = tile_w_candidates(problem.Wo, spec.atom_tile)
        if spec.tile_w not in widths:
            return None, (
                f"tile_w={spec.tile_w} does not fit Wo={problem.Wo} "
                f"(widths tried for it: {widths})"
            )
    if variant == DIRECT_DGRAD_WMMA:
        # A binary serves every extent and channel count, but is only offered
        # where at least half of its padded tile grid is real work.
        from kernels.common.conv_direct_dgrad_wmma import dgrad_wmma_tile_efficiency

        eff = dgrad_wmma_tile_efficiency(spec)
        if eff < 0.5:
            return None, f"tile efficiency {eff:.2f} < 0.5 on this shape"
    if variant == DIRECT_GROUPED_HALO and spec.tile_k > spec.merged_k:
        return None, (
            f"tile_k={spec.tile_k} exceeds the {spec.merged_k} output channels "
            f"of a merged block"
        )
    return spec, "ok"


def direct_wgrad_fold_knobs(variant: str, KH: int, KW: int, cpg: int) -> dict:
    """Knobs of the fold a partial-slot wgrad ``variant`` runs after its pass.

    ``taps`` is the dW extent per output channel: ``KH * KW`` for depthwise,
    ``KH * KW * C`` for the dense gather kernel (cpg baked).
    """
    if variant not in DIRECT_WGRAD_FOLDED:
        raise ValueError(f"{variant!r} does not fold partial slots")
    taps = KH * KW * (cpg if variant == "direct_wgrad_gather" else 1)
    return {"taps": taps}
