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
    "DIRECT_DGRAD_REORGANIZE",
    "DIRECT_DGRAD_REORGANIZE_RTC",
    "DIRECT_DGRAD_TRANSPOSE",
    "DIRECT_DGRAD_TRANSPOSE_RTC",
    "DIRECT_MFMA_DGRAD",
    "DIRECT_MFMA_HELPERS",
    "direct_dgrad_weight_transforms",
    "direct_dgrad_workspaces",
    "direct_mfma_knobs_reason",
    "direct_spec_for_problem",
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
# variant -> (transpose helper, reorganize helper)
DIRECT_MFMA_HELPERS = {
    "direct_grouped_dgrad_mfma": (DIRECT_DGRAD_TRANSPOSE, DIRECT_DGRAD_REORGANIZE),
    "direct_dgrad_mfma_rtc": (DIRECT_DGRAD_TRANSPOSE_RTC, DIRECT_DGRAD_REORGANIZE_RTC),
}
_HELPERS = (
    DIRECT_DGRAD_TRANSPOSE,
    DIRECT_DGRAD_REORGANIZE,
    DIRECT_DGRAD_TRANSPOSE_RTC,
    DIRECT_DGRAD_REORGANIZE_RTC,
)
# Register budget for the MFMA dgrad fprop pass's preloaded weights and
# accumulators (see direct_mfma_knobs_reason): three quarters of the 512-VGPR
# file, leaving the rest for the LDS fragments and addressing.
_MFMA_PRELOAD_VGPR_BUDGET = 384


def make_direct_spec(variant: str, problem, knobs: dict):
    """The spec ``variant`` builds for ``problem``.

    For the MFMA dgrad pipeline this is the spec of its compute pass (a
    forward kernel on the transposed problem); the weight transforms are
    derived from the original problem separately.
    """
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    if variant == "direct_grouped":
        return dc.DirectConvSpec(problem=problem, name="rocke_direct_conv", **knobs)
    if variant == "direct_nongrouped":
        return dn.DirectNongroupedConvSpec(
            problem=problem, name="rocke_direct_conv_nongrouped", **knobs
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
    if variant == "direct_grouped_dgrad":
        return dc.DirectConvDgradSpec(
            problem=problem, name="rocke_direct_dgrad", **knobs
        )
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
        if variant == "direct_nongrouped":
            return dn.is_valid_nongrouped_spec(spec, arch=arch)
        if variant in ("direct_grouped",) + DIRECT_MFMA_DGRAD:
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
        if variant == "direct_grouped_dgrad":
            return dc.is_valid_dgrad_spec(spec, arch=arch)
        if variant == "direct_wgrad":
            spec.validate()
            return dc.is_valid_wgrad_spec(spec, arch=arch)
        if variant in _HELPERS:
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
    from kernels.common import conv_direct_grouped as dc
    from kernels.common import conv_direct_nongrouped as dn

    builders = {
        "direct_grouped": dc.build_direct_conv,
        "direct_nongrouped": dn.build_direct_conv_nongrouped,
        "direct_grouped_dgrad_mfma": dc.build_direct_conv,
        "direct_dgrad_mfma_rtc": dc.build_direct_conv,
        "direct_depthwise": dc.build_direct_depthwise,
        "direct_depthwise_spatial": dc.build_direct_depthwise_spatial,
        "direct_depthwise_tiled": dc.build_direct_depthwise_tiled,
        "direct_depthwise_col": dc.build_direct_depthwise_col,
        "direct_depthwise_dgrad": dc.build_direct_depthwise_dgrad_streaming,
        "direct_grouped_dgrad": dc.build_direct_conv_dgrad,
        "direct_wgrad": dc.build_direct_conv_wgrad,
        DIRECT_DGRAD_TRANSPOSE: dc.build_direct_transpose_weights_dgrad,
        DIRECT_DGRAD_REORGANIZE: dc.build_direct_reorganize_weights,
        DIRECT_DGRAD_TRANSPOSE_RTC: dc.build_direct_transpose_weights_dgrad,
        DIRECT_DGRAD_REORGANIZE_RTC: dc.build_direct_reorganize_weights,
    }
    return builders[variant](spec, arch=arch)


def direct_spec_for_problem(
    variant: str, problem, knobs: dict, arch: str
) -> tuple[object | None, str]:
    """``(spec, "ok")`` when a binary of ``variant``/``knobs`` serves ``problem``.

    The capability match
    (:func:`kernels.conv.kernel_identity.identity_supports_problem`)
    is necessary but not sufficient: tuning knobs have shape rules of their
    own (groups divisible by block_groups, a persistent grid's cell budget,
    ...). The spec is rebuilt on the real problem and run through the kernel's
    own validator, so a binary is only offered where a fresh build would accept
    the shape too. The returned spec is the one its launch geometry comes from.
    Returns ``(None, why)`` otherwise.
    """
    if variant in DIRECT_MFMA_DGRAD:
        why = direct_mfma_knobs_reason(problem, knobs)
        if why:
            return None, why
    spec = make_direct_spec(variant, problem, knobs)
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
    if variant == "direct_nongrouped":
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
    return spec, "ok"
