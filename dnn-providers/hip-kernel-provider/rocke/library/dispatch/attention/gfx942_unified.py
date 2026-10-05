# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 unified-attention candidates: the fp16 ``dense_pipe`` flash path and
the explicit, sweep-only geometry catalog.

Dense-kernel candidates for gfx942 live in :mod:`.gfx942_dense`.
"""

from __future__ import annotations

from typing import Tuple

from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _problem,
    _request_errors,
    _selector_matches,
)
from .candidate import make_tuning_candidate
from .unified_rules import AttentionGeometryVariant


# Absolute T=64/128 points are already represented by one of these multipliers
# for every supported block_size {16,32,64}; omit duplicate candidates.
_TILES = ("1x", "2x", "4x", "8x")


def _variants():
    # The generic gfx942 body supports one or two 16-row atoms per warp.
    for tile in _TILES:
        for nw in (1, 2, 4, 8):
            for mw in (16, 32):
                yield AttentionGeometryVariant(
                    arch="gfx942",
                    path="2d",
                    codepath="narrow",
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=mw,
                )
    # CDNA3 wide paths use the selectable 32x32x8 atom.
    for codepath in ("wide32x8", "transposed_x8"):
        for tile in _TILES:
            for nw in (1, 2, 4, 8):
                yield AttentionGeometryVariant(
                    arch="gfx942",
                    path="2d",
                    codepath=codepath,
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=32,
                )
    # This builder owns real BLOCK_M=128 / 256-thread geometry internally; its
    # spec is a discriminator and not another free tiling axis.
    yield AttentionGeometryVariant(
        arch="gfx942",
        path="2d",
        codepath="gfx942_4warp",
        builder_kind="gfx942_4warp_gqa",
        tile_policy="64",
        num_warps=1,
        block_m_per_warp=32,
    )
    for tile in ("1x", "half"):
        # ``half`` is translated below by the explicit builder through a
        # concrete token; use per-block variants because it is only legal for
        # block sizes >=32.
        for segments in (8, 16, 32, 64, 128):
            yield AttentionGeometryVariant(
                arch="gfx942",
                path="3d",
                codepath="splitkv",
                builder_kind="tiled_3d",
                tile_policy=tile,
                num_segments=segments,
            )


GFX942_TUNING_VARIANTS = tuple(_variants())


def _make_gfx942_dense_pipe_candidate() -> KernelCandidate:
    """Fast gfx942 fp16 prefill kernel — transposed-x8 flash with ring-sliced K.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) whenever both would match the same gfx942 fp16 2D problem.
    The registry sorts ascending (lower = higher precedence).
    Callers can also force this path explicitly via algorithm="dense_pipe".

    GEOMETRY OWNERSHIP: this engine owns the per-engine spec builder
    ``builders.common.attention_spec_builder._spec_gfx942_fp16_flash`` -- the
    GEMM-style ``spec_fn`` for this cohort. Geometry lives in the builder layer
    (not here): the dispatcher's identity stays ``(path, head_size, block_size)``
    and its C++ parity contract is unchanged. Both this candidate and the
    ``_tiled_spec_from_problem`` cascade route the cohort through that one
    function (single source).
    """
    spec_id = "gfx942_dense_pipe"
    name = "attention_gfx942_dense_pipe"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem, arch=req.arch)
        if not ok:
            return False, why
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        from kernels.common.attention_unified import _enable_gfx942_fp16_flash

        if not _enable_gfx942_fp16_flash(problem):
            return False, "gfx942 fp16 flash not eligible for this shape"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx942_dense_pipe",
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="dense_pipe",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16",),
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            # ``_enable_gfx942_fp16_flash`` is the real narrowing; fp8 is
            # unsupported, but the unified body already shifts causal masking.
            supports_features=ATTENTION_FEATURES - {"fp8"},
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    """The dense_pipe path routes only; each tuning geometry goes on both."""
    route.register(_make_gfx942_dense_pipe_candidate())
    for variant in GFX942_TUNING_VARIANTS:
        candidate = make_tuning_candidate(variant)
        route.register(candidate)
        execution.register(candidate)
