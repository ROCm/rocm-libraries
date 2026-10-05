# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 unified-attention candidates: the D256 prefill fast path and the
explicit, sweep-only geometry catalog.

Dense-kernel candidates for gfx950 live in :mod:`.gfx950_dense`.
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
    UNIFIED_BLOCK_SIZES,
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
    # Narrow 16x16 path: one atom per warp at any CTA width, or two stacked
    # atoms per warp (M=32), which the 1024-thread cap limits to nw<=4.
    for backend in ("llvm", "hipcc"):
        for tile in _TILES:
            for nw, mw in [(nw, 16) for nw in (1, 2, 4, 8)] + [
                (nw, 32) for nw in (1, 2, 4)
            ]:
                yield AttentionGeometryVariant(
                    arch="gfx950",
                    path="2d",
                    codepath="narrow",
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=mw,
                    compile_backend=backend,
                )
    # gfx950 32x32 paths require M-per-warp=32 and reject nw=8.
    for codepath in ("wide32", "transposed32"):
        for backend in ("llvm", "hipcc"):
            for tile in _TILES:
                for nw in (1, 2, 4):
                    yield AttentionGeometryVariant(
                        arch="gfx950",
                        path="2d",
                        codepath=codepath,
                        builder_kind="tiled",
                        tile_policy=tile,
                        num_warps=nw,
                        block_m_per_warp=32,
                        compile_backend=backend,
                    )
    # gfx950 3D has fixed BLOCK_M=16 and T=block_size; segment count is the
    # load-balancing geometry exposed by the kernel.
    for segments in (8, 16, 32, 64, 128):
        yield AttentionGeometryVariant(
            arch="gfx950",
            path="3d",
            codepath="splitkv",
            builder_kind="tiled_3d",
            tile_policy="1x",
            num_segments=segments,
        )


GFX950_TUNING_VARIANTS = tuple(_variants())


def _make_gfx950_d256_candidate() -> KernelCandidate:
    """Fast gfx950 bf16 head_size-256 prefill kernel — 32x32 transposed stack
    with FA3-style softmax<->MFMA interleave (mode2/g4) + slab-padded K_lds.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) for the gfx950 bf16 D256 prefill cohort. The registry sorts
    ascending (lower = higher precedence); gfx950-only, so it never competes
    with the gfx942 dense_pipe candidate. Callers can also force this path
    explicitly via algorithm="d256_gfx950".

    The cohort is the single source of truth
    ``kernels.common.attention_unified._d256_gfx950_cohort`` — the same predicate
    the orchestrator's ``_d256_gfx950_fast`` override uses — so dispatch selection
    and the built spec cannot drift. Only the arch gate differs (request arch
    here vs resolved device arch there).
    """
    spec_id = "gfx950_d256"
    name = "attention_gfx950_d256"

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
        from kernels.common.attention_unified import _d256_gfx950_cohort

        if not _d256_gfx950_cohort(problem):
            return False, "not the gfx950 bf16 D256 prefill fast-path cohort"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        from kernels.common.attention_unified import _d256_gfx950_spec_overrides

        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx950_d256",
            tiled_overrides=tuple(sorted(_d256_gfx950_spec_overrides().items())),
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="d256_gfx950",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("bf16",),
            shapes=(
                ShapeRange("hdim_q", allowed=(256,)),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            supports_features=frozenset({"causal", "causal_bottom_right"}),
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
    """The D256 fast path routes only; each tuning geometry goes on both."""
    route.register(_make_gfx950_d256_candidate())
    for variant in GFX950_TUNING_VARIANTS:
        candidate = make_tuning_candidate(variant)
        route.register(candidate)
        execution.register(candidate)
