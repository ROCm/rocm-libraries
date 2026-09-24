# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP16 RCR UniversalGemm dispatcher case."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Optional, Sequence, Tuple

from ...core.arch import ArchTarget
from ...helpers.manifest import gemm_args_signature
from ...helpers.spec import ceil_div_grid
from ...instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from ..core import (
    CandidateRegistry,
    Capability,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    stable_json_hash,
)
from .binding import gemm_rcr_binding
from .common import (
    GEMM_DIM_VOCABULARY,
    GemmRequest,
    apply_split_k,
    rcr_request_errors,
    selector_matches,
)
from .support import (
    gemm_config_supported,
    request_shape_supported,
    support_query_from_universal_spec,
)

_FAMILY = "gemm_fp16_rcr"
_ALGORITHM = "universal_gemm"
GEMM_FP16_RCR_ABI_VERSION = "hipkg-gemm-fp16-rcr/v1"


def _request_errors(req: OperatorRequest) -> list[str]:
    return rcr_request_errors(req, dtype="fp16")


def _make_spec(
    *,
    name: str,
    arch: str,
    tile: TileSpec,
    trait: TraitSpec,
) -> UniversalGemmSpec:
    wave = ArchTarget.from_gfx(arch).wave_size
    return UniversalGemmSpec(
        name=name,
        tile=tile,
        trait=trait,
        data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", layout="RCR"),
        wave_size=wave,
    )


def _spec_cdna_cshuffle(req: GemmRequest, name: str) -> UniversalGemmSpec:
    if req.arch == "gfx942":
        return _make_spec(
            name=name,
            arch=req.arch,
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=8,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
        )
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"),
    )


def _spec_cdna_mem(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=64,
            tile_n=128,
            tile_k=16 if req.arch == "gfx942" else 32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16 if req.arch == "gfx942" else 32,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


def _spec_rdna_wmma(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=64,
            tile_n=32,
            tile_k=16,
            warp_m=2,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


def _spec_rdna_wmma_small(req: GemmRequest, name: str) -> UniversalGemmSpec:
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=32,
            tile_n=32,
            tile_k=16,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue="default"),
    )


def _spec_gfx1250_wmma(req: GemmRequest, name: str) -> UniversalGemmSpec:
    """gfx1250 fp16 RCR: wave32 WMMA on the K=32 ``16x16x32`` atom.

    Geometry and traits are the winner of an exhaustive step-0 sweep over every
    lever the gfx1250 WMMA path leaves reachable (432 geometries x the live
    trait product), correctness-gated against the fp32 reference and confirmed
    over fresh-process repeats at 2048/4096/8192 cubed.

    Notes on the fixed fields, all forced rather than chosen:

    * ``epilogue="default"`` and the absent DTL / preshuffle / chiplet flags are
      the only legal settings on this path (``gemm_universal.py`` WMMA gate).
    * ``lds_swizzle`` stays **off**: on gfx1250 it produces numerically wrong
      results on every geometry tested, and nothing in the validator rejects it.
    * ``pad_*`` stays off; it costs VGPRs and buys nothing on aligned shapes.
    * ``pipeline="mem"``: ``wmma_v1`` is roughly competitive at ``tile_k=32``
      but collapses to about a quarter of the throughput at ``tile_k=64`` --
      its schedule does not scale to the larger K step.

    ``tile_k=64`` + ``lds_k_pad=8`` are a **pair**, and the order they were
    found in matters. A first sweep ranked geometry at ``lds_k_pad=0``, where
    every top config had ``tile_k=32``. Re-ranking the same geometries at
    ``lds_k_pad=8`` inverted that completely -- the whole leaderboard became
    ``tile_k=64``. The pad is not a small correction here: at ``tile_k=64`` the
    unpadded variant runs ~2.9x slower, and the response across pad values
    {0, 8, 16, 24, 32} is non-monotonic (a bank-aliasing signature), so the pad
    cannot be tuned by hill-climbing from 0. ``kpad=24`` measures within noise
    of 8; 8 is chosen for the smaller LDS footprint and tighter run-to-run
    spread.
    """
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="default",
            lds_k_pad=8,
            # Tensor-DMA fill at this tile measures ~1.09x the cooperative copy.
            # Depth stays at 2: depth 3 *regresses* here (~0.97x) -- the third
            # buffer only pays once there is enough compute per tile to hide the
            # extra transfers behind, which this tile does not have. See
            # ``_spec_gfx1250_wmma_tdm`` for the tile that does.
            tdm_lds=True,
            tdm_prefetch=True,
            tdm_prefetch_depth=2,
        ),
    )


def _spec_gfx1250_wmma_tdm(req: GemmRequest, name: str) -> UniversalGemmSpec:
    """gfx1250 fp16 RCR, large square shapes: 256x256 tile on tensor-DMA.

    Admits only shapes divisible by its 256x256 tile (``pad_*`` is off);
    everything else falls through to the candidates below.

    **Registered geometry is not the fastest one measured.** The fastest
    configuration on this part is ``w4x2`` at ``tdm_prefetch_depth=3``
    (**~1.40x** the previous tuned candidate, ``_spec_gfx1250_wmma``), and the
    lever analysis below describes *that* configuration. It is not what this
    function returns, because it does not currently reproduce: built fresh from
    this tree it faults at launch with an HSA aperture violation, on every
    8-wave (``w4x2``) TDM geometry tried and at ``w2x2`` depth 3, while the
    stored artifact from the tuning run still runs correctly. Same sources,
    same three ROCm installs, deterministic -- the cause is not yet identified,
    so the difference is real but unexplained.

    Registering a candidate the dispatcher cannot build is worse than
    registering a slower one, so this returns the nearest configuration that
    was *verified to build and validate from a fresh tree*: ``w2x2`` at depth
    2. By the component ratios below that costs roughly ``0.98x`` for the warp
    grid and ~10% for the depth. The direct A/B re-measurement is still
    outstanding (the tuning host was unavailable), so treat the cost as
    estimated, not measured.

    Restore ``warp_m=4`` / ``tdm_prefetch_depth=3`` once the fault is
    understood; the analysis below is the record of why those are the target.

    Three levers compound here, and the order they were found in matters
    because each one invalidated the reasoning behind the previous stopping
    point:

    * ``tdm_lds`` replaces the cooperative global->LDS copy with one
      wave-uniform descriptor per operand. **Alone it is a small net loss**
      (~0.99x): the descriptor is issued and immediately waited on, so the
      transfer overlaps nothing. It must be paired with the prefetch.
    * ``tdm_prefetch`` double-buffers AB and issues tile N+1 before computing
      tile N, which is what turns the loss into ~1.09x at the old 128x128 tile.
    * The geometry then had to be re-swept, because the fill cost had changed:
      ``256x256`` was previously *rejected* by a double-buffer affordability
      test that both ignored ``lds_k_pad`` and demanded headroom for a second
      workgroup per CU that VGPR pressure (253/256) already prevents.

    ``tdm_prefetch_depth=3`` keeps two tiles in flight. Depth 4 measures
    *identically* at this warp grid, so 3 is chosen for the smaller footprint
    (221 KB vs 295 KB of LDS for the same throughput); depth 2 costs ~10% and
    depth 5 exceeds the 320 KiB per-workgroup LDS ceiling. Ablating the wait
    entirely (numerically invalid, diagnostic only) measures ~1.10x above this
    point, so the residual DMA stall is small and not reachable by deeper
    pipelining.

    ``warp_n=2`` rather than 4 is register blocking: a 64x128 per-warp tile is
    a 4x8 grid of atoms, which drops LDS reads per WMMA from 1.00 to 0.75 by
    reusing each hoisted A fragment across twice as many matrix ops. It is
    worth ~1.02x and it does **not** generalise in either direction:

    * Pushing further *loses*. 8x8 (``w2x2``, ratio 0.50) needs 512 accumulator
      registers and measures 0.98x; ``w4x1`` (ratio 0.625) measures 0.96x. The
      ratio keeps improving while throughput turns over, because the >255-index
      VGPR latency (§21.4) and the lost occupancy overtake it.
    * **Orientation is not symmetric.** ``w2x4`` has the identical atom count,
      accumulator size and ratio, and measures **0.93x** -- a ~10% gap the
      ratio formula cannot see. Spills track it (154 vs 48).

    Do not tune this by spill count. ``w4x2`` at depth 2 is the only nearby
    configuration that spills *nothing* (452 VGPR, 0 spills) and it is ~10%
    slower than this one, which spills 48.

    ``lds_k_pad=8`` is unchanged and still a sharp optimum -- it survived the
    re-sweep across seven geometries and four depths. TDM makes the pad free to
    *apply* (a descriptor field, not instructions), which is not the same as
    making it unnecessary: its job is bank-conflict avoidance on ``ds_read``,
    which TDM does not touch, and ``ds_read`` is now the dominant traffic.
    """
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=256,
            tile_n=256,
            tile_k=64,
            # w2x2 / depth 2, not the measured-best w4x2 / depth 3: see the
            # note at the top of this docstring. Every 8-wave TDM geometry
            # faults at launch when built fresh from this tree.
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="default",
            lds_k_pad=8,
            tdm_lds=True,
            tdm_prefetch=True,
            tdm_prefetch_depth=2,
        ),
    )


def _spec_gfx1250_wmma_small(req: GemmRequest, name: str) -> UniversalGemmSpec:
    """gfx1250 fp16 RCR coverage candidate for shapes the 128x128 tile can't take.

    The tuned candidate above has ``pad_*`` off, so it only admits shapes
    divisible by its 128x128 tile. Without this companion, every skinny /
    decode shape would fall off the end of the registry and
    ``dispatch_gemm_fp16`` would raise -- the same two-candidate split the CDNA
    and RDNA entries already use.

    This is a **coverage** choice, not a tuned one: the step-0 sweep behind
    ``_spec_gfx1250_wmma`` covered square-large shapes only. Of five padded
    small-tile geometries verified correct on decode shapes (M = 1, 2, 8), this
    was the fastest, but it has not been swept as a decode optimum. Re-tune it
    against a real decode shape set before treating its geometry as settled.
    """
    return _make_spec(
        name=name,
        arch=req.arch,
        tile=TileSpec(
            tile_m=64,
            tile_n=128,
            tile_k=32,
            warp_m=1,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            lds_k_pad=8,
        ),
    )


# Largest M/N at which the gfx1250 128x128 candidate was verified to produce
# correct results. Above it the emitted kernel aborts the queue with
# HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
_GFX1250_WMMA_128_MAX_MN = 2048


def _gfx1250_wmma_128_shape_guard(req: GemmRequest) -> Tuple[bool, str]:
    """Decline shapes where the 128x128 gfx1250 kernel is known to fault.

    Verified on hardware: correct at 1920x1920 (grid 15x15) and 2048x2048
    (16x16), faults at 2176x2176 (17x17) and every larger square tried, and at
    2048x4096 / 4096x2048 -- i.e. whenever **either** grid dimension exceeds 16.
    The 256x256 candidate does not share the limit (clean to 16384x16384,
    grid 64x64), so this is specific to this kernel, not to the grid size.

    The root cause is not yet identified, so the bound is empirical rather than
    derived; it is deliberately the last *verified-good* value rather than the
    first failing one. Declining here makes the request fall through, which is
    the behaviour that existed before this candidate was added -- dispatching a
    kernel that aborts the queue is strictly worse than reporting no support.
    """
    if req.M > _GFX1250_WMMA_128_MAX_MN or req.N > _GFX1250_WMMA_128_MAX_MN:
        return False, (
            f"gfx1250 128x128 candidate is verified only to "
            f"M,N <= {_GFX1250_WMMA_128_MAX_MN} (got M={req.M}, N={req.N}); "
            "larger shapes fault at launch"
        )
    return True, "ok"


def _make_candidate(
    *,
    name: str,
    spec_id: str,
    priority: int,
    spec_fn: Callable[[GemmRequest, str], UniversalGemmSpec],
    arches: Tuple[str, ...],
    extra_support: Optional[Callable[[GemmRequest], Tuple[bool, str]]] = None,
) -> KernelCandidate:
    """Register one candidate.

    ``extra_support`` is an optional per-candidate veto, applied after the
    shared checks. It exists for a candidate whose *emitted kernel* is known
    to be wrong outside some range: the shared predicates only see the spec,
    so they cannot tell. Declining is strictly better than dispatching a
    kernel that faults -- the request falls through to the next candidate, or
    reports no support, which is what happened before the candidate existed.
    """

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if extra_support is not None:
            ok, why = extra_support(req)
            if not ok:
                return False, why
        spec = spec_fn(req, name)
        ok, why = gemm_config_supported(
            support_query_from_universal_spec(spec, arch=req.arch)
        )
        if not ok:
            return False, why
        return request_shape_supported(req, spec)

    def select(req: OperatorRequest) -> UniversalGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        # Engage split-K for skinny/tall-N decode shapes that leave the device
        # idle; a no-op (returns the spec unchanged) for shapes that already
        # fill the device, keeping the default / square path byte-identical.
        return apply_split_k(req, spec_fn(req, name))

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=_ALGORITHM,
        spec_id=spec_id,
        abi_version=GEMM_FP16_RCR_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=arches, dtypes=("fp16",), layouts=("RCR",)),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: gemm_args_signature(),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build_universal_gemm,
        bind=lambda result, verify: gemm_rcr_binding(result, verify, dtype="fp16"),
    )
    return candidate


def _grid(spec: UniversalGemmSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    t = spec.tile
    assert isinstance(req, GemmRequest)
    # Split-K adds a Z dimension of ``split_k`` K-slice CTAs per (m,n) tile;
    # split_k == 1 (default) collapses to the canonical 2D grid.
    return ceil_div_grid((req.N, t.tile_n), (req.M, t.tile_m), (spec.trait.split_k, 1))


# Explicit gfx targets rather than a cdna/rdna family label. Family does not
# imply wave size -- gfx1250 is cdna at wave32 -- so a family gate would admit a
# wave32 target into these wave64 MFMA candidates. An arch absent from a list is
# a target the candidate was never built or run against.
_CDNA_MFMA_FP16 = ("gfx942", "gfx950")
_RDNA_WMMA = ("gfx11-generic", "gfx1151", "gfx1201")
# gfx1250 gets its own list: cdna family, wave32, WMMA path, and a K=32 atom
# (16x16x32) that none of the above share.
_GFX1250_WMMA = ("gfx1250",)

GEMM_FP16_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=GEMM_DIM_VOCABULARY,
    require_build=True,
    require_binding=True,
)
GEMM_FP16_REGISTRY.extend(
    (
        _make_candidate(
            name="universal_gemm_fp16_cdna_cshuffle",
            spec_id="cdna_cshuffle_default",
            priority=10,
            spec_fn=_spec_cdna_cshuffle,
            arches=_CDNA_MFMA_FP16,
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma",
            spec_id="rdna_wmma_default",
            priority=10,
            spec_fn=_spec_rdna_wmma,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_fp16_cdna_mem",
            spec_id="cdna_mem_64x128",
            priority=20,
            spec_fn=_spec_cdna_mem,
            arches=_CDNA_MFMA_FP16,
        ),
        _make_candidate(
            name="universal_gemm_fp16_rdna_wmma_small",
            spec_id="rdna_wmma_32x32",
            priority=20,
            spec_fn=_spec_rdna_wmma_small,
            arches=_RDNA_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1250_wmma_tdm",
            spec_id="gfx1250_wmma_tdm_256x256x64_d4",
            priority=5,
            spec_fn=_spec_gfx1250_wmma_tdm,
            arches=_GFX1250_WMMA,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1250_wmma",
            spec_id="gfx1250_wmma_128x128x64",
            priority=10,
            spec_fn=_spec_gfx1250_wmma,
            arches=_GFX1250_WMMA,
            extra_support=_gfx1250_wmma_128_shape_guard,
        ),
        _make_candidate(
            name="universal_gemm_fp16_gfx1250_wmma_small",
            spec_id="gfx1250_wmma_64x128x32_padded",
            priority=20,
            spec_fn=_spec_gfx1250_wmma_small,
            arches=_GFX1250_WMMA,
        ),
    )
)


def gemm_fp16_candidates() -> Tuple[KernelCandidate, ...]:
    return GEMM_FP16_REGISTRY.candidates()


def _kernel_id(
    req: GemmRequest, candidate: KernelCandidate, spec: UniversalGemmSpec
) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(asdict(spec), n=16)
    return KernelId(
        op="gemm",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def build_kernel(result: DispatchResult):
    """Deprecated in favour of ``result.build()``.

    Kept because benchmark harnesses and examples import it by name; it now
    delegates so there is one definition of how a selection becomes IR.
    """
    return result.build()


def gemm_fp16_sweep_space(req: OperatorRequest) -> Sequence[UniversalGemmSpec]:
    """Bounded sweep space from all registered FP16 RCR candidates."""
    if _request_errors(req):
        return ()
    specs: list[UniversalGemmSpec] = []
    seen = set()
    for candidate in GEMM_FP16_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        spec_hash = stable_json_hash(asdict(spec), n=16)
        if spec_hash not in seen:
            seen.add(spec_hash)
            specs.append(spec)
    return tuple(specs)


def dispatch_gemm_fp16(
    req: GemmRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select a registered FP16 RCR UniversalGemm candidate for ``req``."""
    candidate = GEMM_FP16_REGISTRY.select(req, ranker=ranker)
    spec = candidate.select_spec(req)
    kid = _kernel_id(req, candidate, spec)
    return DispatchResult(
        request=req,
        candidate=candidate,
        spec=spec,
        kernel_id=kid,
        grid=candidate.grid(spec, req),
        block=candidate.block(spec),
        signature=tuple(candidate.signature(spec)),
        explanation=(
            f"selected {candidate.name} for fp16 RCR GEMM on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        ),
    )
