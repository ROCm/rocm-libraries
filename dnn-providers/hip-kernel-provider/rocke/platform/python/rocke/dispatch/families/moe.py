# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused MoE dispatcher family (single-launch mega-kernel).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`rocke.instances.common.moe_fused_mega` (f16/bf16) and
:mod:`rocke.instances.common.moe_fused_mega_fp8` (fp8 e4m3 block-scale).

SCOPE -- what this dispatcher decides
-------------------------------------
The fused-MoE mega-kernel has a STATIC tile geometry (locked by BUILD_SPEC:
``tile_m=16, tile_n_inter=256, tile_k_gu=32``); the MoE problem dims
(num_tokens / hidden / intermediate / num_experts / top_k) are RUNTIME kernel
args, not selection knobs. The load-bearing dispatch decision is therefore the
*element path*: the f16/bf16 mega-kernel vs the fp8 block-scale mega-kernel.

The candidate set is two element-path kernels, selected by request dtype:

* ``mega_f16``  : f16/bf16 mega-kernel (atom 16x16x32),
* ``mega_fp8``  : fp8 e4m3 block-scale mega-kernel (hero atom 16x16x128).

Arch coverage: MoE is CDNA-only (the mega-kernel atoms are MFMA), and both
shipped configs are gfx950-tuned -- the f16 path's 16x16x32 atom does not exist
on gfx942, and the fp8 path's 16x16x128 hero atom is a gfx950-only scaled
intrinsic (it is not even in the generic MMA catalog; the instance builder skips
the catalog guard for ``atom.k==128``). So the support predicate gates to the
CDNA family AND to gfx950, and validates the f16 atom against the per-arch MMA
catalog where it is expressible.

Beside the two mega-kernel element paths this family also registers the
token-banded fp8 cohort (below) and the activation gather/rescale prologue
(:data:`PROLOGUE_SPEC_ID`), which produces the A matrix and scatter metadata
the mega-kernel consumes.

ROUTING NOTE -- the fused/split boundary at 8|9
-----------------------------------------------
The boundary is where the measurements put it: the single launch wins outright
to T=8 and is behind from T=16 up. It is worth naming because callers written
before the bands existed expect every decode token count to run the single
fused kernel. That is a stale expectation rather than a second measurement --
moving the boundary back to cover T=32 would also hand T=16 and T=64 to the
kernel that measured slower at both. Re-measuring is the way to move it, not
agreeing with whichever caller complains first.

DEFERRED -- the MoE component pipeline
--------------------------------------
The non-fused MoE component kernels (``moe_sorting``, ``moe_gemm_fused``,
``moe_smoothquant``) and the multi-launch ``fused_moe`` path are separate
algorithms; only the single-launch mega-kernel is dispatched here. Adding them
is a candidate-registration follow-on (same recipe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Sequence, Tuple

from ...core.arch import ArchTarget
from ...instances.common.moe_fused_mega import (
    FusedMegaKernelSpec,
    build_moe_fused_mega_gemm,
    moe_fused_mega_grid,
    moe_fused_mega_signature,
)
from ...instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
    build_moe_split_down_fp8,
    moe_fused_mega_fp8_grid,
    moe_fused_mega_fp8_signature,
    moe_split_down_fp8_grid,
    moe_split_down_fp8_signature,
)
from ...instances.common.moe_fused_mega_fp8_tuned import (
    BAND_GEOMETRY,
    BAND_RANGE,
    BASE_KNOBS,
    COOP_LEVERS,
    FUSED_BAND,
    FUSED_LDS_PAD,
    MAX_TUNED_TOKENS,
    TUNED_SHAPE,
    band_for,
    matches_tuned_shape,
)
from ...instances.common.moe_gather_rescale_a import (
    GROUP_K,
    MoeGatherRescaleSpec,
    build_moe_gather_rescale_a,
    moe_gather_rescale_a_grid,
    moe_gather_rescale_a_signature,
)
from ..core import (
    Capability,
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    Ranker,
    ShapeRange,
    stable_json_hash,
)

_FAMILY = "moe_fused_mega"
MOE_ABI_VERSION = "hipkg-moe-fused-mega/v1"

# Both mega configs are gfx950-tuned (see module docstring).
_SUPPORTED_ARCHES = ("gfx950",)


@dataclass(frozen=True)
class MoeRequest(OperatorRequest):
    """Normalized fused mixture-of-experts request."""

    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    arch: str
    op: str = "moe"
    dtype: str = "fp16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    #: Block-scale group width, on both weight axes and the activation. On the
    #: request rather than assumed because a different group is a different
    #: kernel, not a different tile size, and a caller whose weights were
    #: quantized at another width must be declined rather than mis-served.
    group_k: int = GROUP_K
    #: SiLU is the only activation wired into the fused epilogue. Carried so
    #: that a request for another one is refused by name instead of silently
    #: receiving a SiLU kernel.
    activation: str = "silu"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = _moe_dtype(self.dtype)
        d["activation"] = self.activation.lower()
        return d

    def dims(self) -> dict[str, int]:
        return {
            "num_tokens": int(self.num_tokens),
            "hidden": int(self.hidden),
            "intermediate": int(self.intermediate),
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
            "group_k": int(self.group_k),
        }


MOE_DIM_VOCABULARY = (
    "num_tokens",
    "hidden",
    "intermediate",
    "num_experts",
    "top_k",
    "group_k",
)

#: SiLU is the only activation wired into the fused epilogue.
MOE_ACTIVATIONS = ("silu",)


def _moe_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    if d in ("fp8", "f8", "fp8e4m3", "e4m3"):
        return "fp8e4m3"
    return d


_F16_DTYPES = ("fp16", "bf16")
_FP8_DTYPES = ("fp8e4m3",)

# ---------------------------------------------------------------------------
# Token-banded fp8 cohort
# ---------------------------------------------------------------------------
#
# The shape the sweep was run on, the band boundaries, and the knob sets those
# measurements produced are stated once, in
# :mod:`rocke.instances.common.moe_fused_mega_fp8_tuned`, and imported here.
# This module is one of that record's two consumers -- the ``fused_mega_moe``
# benchmark is the other -- so restating any of it would create a copy that can
# disagree, and the disagreement would be silent: a boundary moved in one place
# routes production traffic the other place never measured.
#
# Registering the cohort makes ``num_tokens`` a selection knob. It has to be:
# the measured winner changes four times between decode and prefill, because
# ``tile_m`` trades weight traffic against row padding and the two cross over.

def _in_tuned_cohort(req: MoeRequest) -> Tuple[bool, str]:
    """Whether this request is the shape and token range the bands cover.

    The shape half is :func:`matches_tuned_shape`; the token bound is applied
    here because it is a statement about how far the sweep ran, and the bands
    themselves already carry it.
    """
    ok, why = matches_tuned_shape(
        hidden=req.hidden,
        intermediate=req.intermediate,
        num_experts=req.num_experts,
        top_k=req.top_k,
    )
    if not ok:
        return False, why
    tokens = int(req.num_tokens)
    if tokens > MAX_TUNED_TOKENS:
        return (
            False,
            f"num_tokens={tokens} exceeds the measured bound {MAX_TUNED_TOKENS}",
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Performance knobs the dispatcher pins
# ---------------------------------------------------------------------------
#
# These used to be read from environment variables inside the instance, which
# meant the dispatched spec did not describe the kernel that got built: the
# same request produced different ISA on two machines, and a spec_hash minted
# on one of them replayed the wrong binary on the other. They are spec fields
# now, and the dispatcher states them, because "which schedule wins" is a
# routing decision and the instance's default is only a default.
#
# The values are the measured-best settings and they reproduce what the env
# default did (``ROCKE_FP8_SCHED=iglp1``), so pinning them is not a codegen
# change -- it is the same kernel, now stated.
#
# Only names the spec actually declares belong here. The lookup drops unknown
# ones so a stale entry cannot raise, which means a knob that stops existing
# goes on being "pinned" to no effect; ``TestPinnedPerfKnobs`` fails on that.
#
# ``sched_cadence`` earns its place on the untuned fallback spec, where the
# three cadences emit three different modules. The tuned bands emit the
# same bytes for all three, so pinning it there states the setting rather than
# changing it -- which is the point, since the band would otherwise inherit
# whatever the instance default happened to be.
_PINNED_PERF_KNOBS: dict[str, object] = {
    "sched_cadence": "iglp1",
    "coop_alias": False,
}


def _fp8_spec_fields() -> frozenset[str]:
    return frozenset(f.name for f in dataclass_fields(FusedMegaKernelSpecFp8))


def _pinned_perf_knobs() -> dict:
    """The knobs above, restricted to the ones the spec actually declares.

    The instance is landing these fields on a different branch, so the set is
    resolved against the dataclass instead of assumed. A knob the spec has not
    grown yet is reported by :func:`_deferred_perf_knobs` rather than dropped
    quietly -- the whole point is that nothing about the built kernel is left
    to a default nobody named.
    """
    declared = _fp8_spec_fields()
    return {k: v for k, v in _PINNED_PERF_KNOBS.items() if k in declared}


def _deferred_perf_knobs() -> Tuple[str, ...]:
    declared = _fp8_spec_fields()
    return tuple(k for k in _PINNED_PERF_KNOBS if k not in declared)


def _stage1_spec(req: MoeRequest, *, spec_id: str) -> FusedMegaKernelSpecFp8:
    """Stage-1 (or fused) spec for one band."""
    tile_m, warp_m = BAND_GEOMETRY[spec_id]
    fields = dict(BASE_KNOBS)
    fields["tile_m"] = tile_m
    fields["warp_m"] = warp_m
    if spec_id == FUSED_BAND:
        # The single-launch kernel keeps the padded staging it was tuned with;
        # it has no shared B tile competing for the LDS budget.
        fields["lds_pad"] = FUSED_LDS_PAD
    else:
        fields.update(COOP_LEVERS)
        # Stage 1 publishes the intermediate to HBM, so it needs to know how
        # wide that buffer is. Request-derived, not a tuning knob.
        fields["split_inter_max"] = int(req.intermediate)
    # Re-derive from warp_m * warp_n * wave_size rather than stating it twice.
    fields["block_size"] = 0
    fields.update(_pinned_perf_knobs())
    return FusedMegaKernelSpecFp8(name=f"moe_{spec_id}", **fields)


def _stage2_spec(req: MoeRequest, *, spec_id: str) -> FusedMegaKernelSpecFp8:
    """Stage-2 (split down GEMM) spec for one band.

    Stage 2 is a separate launch and is tuned separately: it has about a
    quarter of stage 1's register pressure and wants ``warp_n=4`` where stage 1
    wants 1, which is worth 11 us of the split's margin. Only ``tile_m`` has to
    agree with stage 1. ``warp_n=4`` caps ``warp_m`` at 4 (1024 threads), so a
    wide stage-1 tile takes more M atoms per warp here rather than more warps.
    """
    stage1 = _stage1_spec(req, spec_id=spec_id)
    fields = dict(BASE_KNOBS)
    fields.update(COOP_LEVERS)
    fields["tile_m"] = stage1.tile_m
    fields["warp_m"] = min(4, max(1, stage1.tile_m // 16))
    fields["warp_n"] = 4
    fields["tile_n_down"] = 128
    fields["lds_pad"] = 0
    fields["split_inter_max"] = int(req.intermediate)
    fields["block_size"] = 0
    fields.update(_pinned_perf_knobs())
    return FusedMegaKernelSpecFp8(name=f"moe_{spec_id}_dn", **fields)


#: Stage-2 candidates are reachable only by explicit ``spec_id``. They are
#: registered because they are real, compilable, launchable kernels that the
#: dispatcher owns the geometry of -- but an ``auto`` request means "the MoE
#: layer", and answering that with a down GEMM alone would be wrong. Use
#: :func:`dispatch_moe_plan` to get both stages.
_STAGE2_SUFFIX = "_stage2"


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, MoeRequest):
        return [f"expected MoeRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "moe":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("num_tokens", "hidden", "intermediate", "num_experts", "top_k"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if int(req.top_k) > int(req.num_experts):
        errors.append("top_k must be <= num_experts")
    if int(req.group_k) != GROUP_K:
        errors.append(
            f"group_k={req.group_k} unsupported; the block-scale dequant is "
            f"hard-coded to a {GROUP_K}-wide group"
        )
    if req.activation.lower() not in MOE_ACTIVATIONS:
        errors.append(
            f"unsupported activation {req.activation!r}; "
            f"one of {'/'.join(MOE_ACTIVATIONS)}"
        )
    dt = _moe_dtype(req.dtype)
    if dt not in _F16_DTYPES + _FP8_DTYPES:
        errors.append(f"unsupported dtype {req.dtype!r}; one of fp16/bf16/fp8")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _selector_matches(req: MoeRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


def _spec_f16(req: MoeRequest):
    dt = _moe_dtype(req.dtype)
    return FusedMegaKernelSpec(name=f"moe_{dt}", dtype=dt)


def _spec_fp8(req: MoeRequest):
    return FusedMegaKernelSpecFp8(name="moe_fp8", **_pinned_perf_knobs())


def _selected_tile_m(req: MoeRequest) -> int:
    """The ``tile_m`` the rest of this family will run for ``req``.

    The prologue's row blocking is not free to choose: the activation scale it
    publishes is uniform over exactly ``tile_m`` rows, and the gate/up fold
    applies one per-lane scalar across accumulator slots from four different
    output rows, so a prologue built at a different ``tile_m`` than the GEMM it
    feeds is silently wrong -- legal addresses, wrong row's scale, no error.
    Deriving it here rather than defaulting it is what stops the two drifting.
    """
    band = band_for(req.num_tokens) if _in_tuned_cohort(req)[0] else None
    if band is None:
        return int(_spec_fp8(req).tile_m)
    return int(BAND_GEOMETRY[band][0])


def _prologue_spec(req: MoeRequest) -> MoeGatherRescaleSpec:
    # ``max_n_hb`` sizes LDS scratch statically; ``hidden`` itself stays a
    # runtime argument, so this is a ceiling, not a shape claim.
    n_hb = -(-int(req.hidden) // GROUP_K)
    return MoeGatherRescaleSpec(tile_m=_selected_tile_m(req), max_n_hb=n_hb)


def _num_m_blocks(req: MoeRequest, tile_m: int) -> int:
    """Worst-case bound on the sorted token blocks a launch must cover.

    The exact count is ``sum(ceil(n_e / tile_m))`` over the routing histogram,
    which a dispatch decision does not have and must not wait for. Spreading
    the slots evenly across the experts and rounding that share up computes the
    *average*, not a bound: a real histogram is lopsided, and every expert
    whose share crosses a tile boundary adds a block the even split never
    counted. Bounding instead of averaging, ``ceil(n_e / t) <= n_e // t + 1``
    summed over active experts gives ``slots // tile_m + active``, and at most
    ``slots`` experts can be active, which keeps the bound tight when the
    tokens cannot reach every expert.
    """
    slots = int(req.num_tokens) * int(req.top_k)
    active = min(int(req.num_experts), slots)
    return max(1, active + slots // max(1, int(tile_m)))


def _build(spec, arch: str):
    """Route to the builder that matches the spec ``select_spec`` produced.

    The family carries two spec types, so the fp8/f16 split that ``_struct``
    already makes for identity has to be made here too. Keyed on the spec type
    rather than on the request dtype, so the two can never disagree.
    """
    if isinstance(spec, FusedMegaKernelSpecFp8):
        return build_moe_fused_mega_gemm_fp8(spec, arch)
    return build_moe_fused_mega_gemm(spec, arch)


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------
#
# Every candidate reports a real grid and a real argument signature. They used
# to report ``(0, 0, 0)`` and ``()`` on the grounds that the grid is a function
# of runtime dims -- but those dims are on the request, which is exactly what
# ``grid(spec, req)`` is handed. The placeholder made a DispatchResult look
# complete while being unlaunchable, so a caller had to re-derive the geometry
# next to the dispatcher and could disagree with it.
#
# One dimension resists that: the count of sorted token blocks depends on the
# routing histogram, which does not exist until after the dispatch decision.
# The reported grid therefore carries the worst case, and a caller that has
# routed should ask :func:`moe_launch_grid` for the exact one.

#: ``candidate name -> which launch of the algorithm it is``. Part of the
#: compile identity: two of these launches can hold specs that agree on every
#: field and still be different kernels, because a different builder consumes
#: them.
_LAUNCH_KIND: dict[str, str] = {}


def moe_launch_kind(result: DispatchResult) -> str:
    """Which launch of the algorithm ``result`` is.

    A split plan is a sequence of these, and reading the stage off the position
    in the plan is the assumption that breaks the first time a plan grows a
    stage. Public because a caller has no other way to ask.
    """
    return _LAUNCH_KIND[result.candidate.name]


def moe_launch_grid(result: DispatchResult, num_m_blocks: int) -> Tuple[int, int, int]:
    """The exact grid for ``result`` once the real block count is known.

    :attr:`DispatchResult.grid` can only carry :func:`_num_m_blocks`, a bound
    computed before routing exists. Launching that bound is not a safe
    shortcut: only the fused kernel skips a block whose ``BlockExpertIds``
    entry is the ``-1`` empty marker, while ``build_moe_split_down_fp8``
    rebases its weight pointers off that entry unguarded, so surplus blocks
    read past the end of the metadata. A caller that has run the block
    alignment step passes its block count here and launches the result.
    """
    return _grid_for_blocks(
        result.spec, result.request, int(num_m_blocks), launch=moe_launch_kind(result)
    )


def _grid(spec, req: OperatorRequest, *, launch: str) -> Tuple[int, int, int]:
    assert isinstance(req, MoeRequest)
    return _grid_for_blocks(spec, req, _num_m_blocks(req, spec.tile_m), launch=launch)


def _grid_for_blocks(
    spec, req: OperatorRequest, blocks: int, *, launch: str
) -> Tuple[int, int, int]:
    assert isinstance(req, MoeRequest)
    if launch == "prologue":
        return tuple(moe_gather_rescale_a_grid(blocks, spec))
    if launch == "down":
        # Stage 2 tiles over H_out rather than the intermediate -- that is the
        # reason it is a separate launch at all.
        return tuple(moe_split_down_fp8_grid(blocks, int(req.hidden), spec))
    if launch == "fused_f16":
        return tuple(moe_fused_mega_grid(blocks, int(req.intermediate), spec))
    return tuple(moe_fused_mega_fp8_grid(blocks, int(req.intermediate), spec))


def _signature(spec, *, launch: str) -> Sequence[dict]:
    if launch == "prologue":
        return tuple(moe_gather_rescale_a_signature(spec))
    if launch == "down":
        return tuple(moe_split_down_fp8_signature(spec))
    if launch == "fused_f16":
        return tuple(moe_fused_mega_signature(spec))
    return tuple(
        moe_fused_mega_fp8_signature(spec, split_gateup=launch == "gate_up")
    )


def _make_candidate(
    *, name, spec_id, dtypes, spec_fn, priority, launch
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # f16 path: validate the 16x16x32 atom against the per-arch catalog
        # (gfx942 lacks it -> rejected even though arch-family is CDNA).
        if _moe_dtype(req.dtype) in _F16_DTYPES:
            target = ArchTarget.from_gfx(req.arch)
            dt = _moe_dtype(req.dtype)
            if not target.mma.has_shape(
                family="mma",
                a_dtype=dt,
                b_dtype=dt,
                c_dtype="fp32",
                m=16,
                n=16,
                k=32,
            ):
                return False, f"unsupported {dt} 16x16x32 MoE atom on {req.arch}"
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MoeRequest)
        return spec_fn(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=spec_id,
        spec_id=spec_id,
        abi_version=MOE_ABI_VERSION,
        priority=priority,
        # Both mega configs are gfx950-tuned: the 16x16x32 / 16x16x128 atoms
        # they are built around are gfx950-specific.
        capability=Capability(arches=_SUPPORTED_ARCHES, dtypes=dtypes),
        _supports=support,
        select_spec=select,
        signature=lambda spec: _signature(spec, launch=launch),
        grid=lambda spec, req: _grid(spec, req, launch=launch),
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    _LAUNCH_KIND[name] = launch
    return candidate


def _make_banded_candidate(*, spec_id: str, stage2: bool) -> KernelCandidate:
    """A token-banded fp8 candidate from the tuned cohort.

    Separate from :func:`_make_candidate` because these gate on ``num_tokens``
    and on the shape cohort, and because each one owns a distinct builder --
    fused, split stage 1, or split stage 2 -- rather than routing on spec type.
    """
    name = f"moe_{spec_id}" + (_STAGE2_SUFFIX if stage2 else "")
    lo, hi = BAND_RANGE[spec_id]
    if stage2:
        launch = "down"
    elif spec_id == FUSED_BAND:
        launch = "fused"
    else:
        launch = "gate_up"

    def build(spec, arch: str):
        if launch == "down":
            return build_moe_split_down_fp8(spec, arch)
        return build_moe_fused_mega_gemm_fp8(
            spec, arch, split_gateup=launch == "gate_up"
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        ok, why = _in_tuned_cohort(req)
        if not ok:
            return False, why
        if stage2 and req.spec_id.strip().lower() == "auto":
            # Never an answer to "auto": see _STAGE2_SUFFIX.
            return False, (
                "stage 2 is not selectable on its own; "
                "use dispatch_moe_plan for the split pair"
            )
        # Checked for stage 2 as well, so naming a stage-2 candidate cannot
        # pair a down GEMM with a stage 1 from a different band -- the two
        # would then disagree about the row blocking of the intermediate one
        # writes and the other reads, which is a wrong answer, not a slow one.
        band = band_for(req.num_tokens)
        if band != spec_id:
            return False, (
                f"num_tokens={req.num_tokens} is served by {band!r}, "
                f"not {spec_id!r}"
            )
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MoeRequest)
        fn = _stage2_spec if stage2 else _stage1_spec
        return fn(req, spec_id=spec_id)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=spec_id,
        spec_id=name if stage2 else spec_id,
        abi_version=MOE_ABI_VERSION,
        # Ranked ahead of the generic fp8 pair (``candidates()`` sorts on
        # priority ascending, so lower wins). Within its band and cohort this
        # candidate is measured against Triton; the generic default is tuned
        # for a different, much wider intermediate.
        priority=5,
        capability=Capability(
            arches=_SUPPORTED_ARCHES,
            dtypes=_FP8_DTYPES,
            # The cohort and the band as data, so ``coverage()`` can answer
            # "which MoE shapes and token counts does gfx950 serve, with
            # what?" without probing the predicate with a request. It also
            # rejects an out-of-cohort shape before the predicate runs.
            shapes=(
                ShapeRange("hidden", TUNED_SHAPE.hidden, TUNED_SHAPE.hidden),
                ShapeRange(
                    "intermediate",
                    TUNED_SHAPE.intermediate,
                    TUNED_SHAPE.intermediate,
                ),
                ShapeRange(
                    "num_experts", TUNED_SHAPE.num_experts, TUNED_SHAPE.num_experts
                ),
                ShapeRange("top_k", TUNED_SHAPE.top_k, TUNED_SHAPE.top_k),
                ShapeRange("num_tokens", lo, hi),
            ),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: _signature(spec, launch=launch),
        grid=lambda spec, req: _grid(spec, req, launch=launch),
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=build,
    )
    _LAUNCH_KIND[name] = launch
    return candidate


#: The activation prologue's identity. Like stage 2 it is never an answer to
#: ``auto`` -- it produces the mega-kernel's A matrix and scatter metadata, not
#: a layer output -- so it is reachable by naming it, or by asking
#: :func:`dispatch_moe_plan` for it.
PROLOGUE_SPEC_ID = "moe_gather_rescale_a"


def _make_prologue_candidate() -> KernelCandidate:
    """The activation gather/rescale prologue, as a dispatchable candidate.

    It was launchable only by an adapter holding the instance module directly,
    which put the one thing dispatch must own -- ``tile_m``, shared with the
    GEMM that consumes the gathered A -- outside dispatch. A mismatch there is
    not a slow kernel, it is the wrong row's activation scale applied to a
    legal address, so the coupling belongs where the GEMM's ``tile_m`` is
    already decided. See :func:`_selected_tile_m`.

    Unlike the mega-kernel candidates this one is not cohort-gated: it is plain
    vector memory work with ``hidden``, ``topk`` and the group count as runtime
    arguments, so it has no tuned geometry to overclaim. The one shape it does
    constrain is ``hidden``, which the kernel requires to be a whole number of
    scale groups -- a partial group would leave the tail of a vector load with
    no rescale ratio, and the kernel emits no tail path.
    """
    name = PROLOGUE_SPEC_ID

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MoeRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        if req.spec_id.strip().lower() == "auto":
            return False, (
                "the activation prologue is not the MoE layer; name it as "
                "spec_id, or use dispatch_moe_plan(..., with_prologue=True)"
            )
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MoeRequest)
        return _prologue_spec(req)

    candidate = KernelCandidate(
        name=name,
        family=_FAMILY,
        algorithm=PROLOGUE_SPEC_ID,
        spec_id=PROLOGUE_SPEC_ID,
        abi_version=MOE_ABI_VERSION,
        # Never competes: it declines ``auto``, so priority only orders it
        # against an explicit request for it, where it is alone.
        priority=5,
        capability=Capability(
            arches=_SUPPORTED_ARCHES,
            dtypes=_FP8_DTYPES,
            shapes=(ShapeRange("hidden", multiple_of=GROUP_K),),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: _signature(spec, launch="prologue"),
        grid=lambda spec, req: _grid(spec, req, launch="prologue"),
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: build_moe_gather_rescale_a(spec, arch=arch),
    )
    _LAUNCH_KIND[name] = "prologue"
    return candidate


MOE_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=MOE_DIM_VOCABULARY, require_build=True
)
MOE_REGISTRY.extend(
    (
        _make_candidate(
            name="moe_fused_mega_f16",
            spec_id="mega_f16",
            dtypes=_F16_DTYPES,
            spec_fn=_spec_f16,
            priority=10,
            launch="fused_f16",
        ),
        _make_candidate(
            name="moe_fused_mega_fp8",
            spec_id="mega_fp8",
            dtypes=_FP8_DTYPES,
            spec_fn=_spec_fp8,
            priority=10,
            launch="fused",
        ),
    )
)
MOE_REGISTRY.extend(
    tuple(
        _make_banded_candidate(spec_id=spec_id, stage2=False)
        for spec_id in BAND_GEOMETRY
    )
    + tuple(
        # Only the split bands have a stage 2; the fused band is one launch.
        _make_banded_candidate(spec_id=spec_id, stage2=True)
        for spec_id in BAND_GEOMETRY
        if spec_id != FUSED_BAND
    )
)
MOE_REGISTRY.register(_make_prologue_candidate())


def moe_candidates() -> Tuple[KernelCandidate, ...]:
    return MOE_REGISTRY.candidates()


def _spec_path(spec) -> str:
    if isinstance(spec, FusedMegaKernelSpecFp8):
        return "fp8"
    if isinstance(spec, MoeGatherRescaleSpec):
        return "prologue"
    return "f16"


def _struct(spec, *, launch: str) -> dict:
    """The compile identity of a selection: everything that changes the binary.

    Derived from the spec's own fields rather than an allowlist. The allowlist
    this replaces had to be extended by hand for every knob added to the spec,
    and omitting one did not fail: two different kernels quietly shared a
    ``compile_key``, so the second selection replayed the first one's HSACO.
    Pinning the scheduling knobs (see :data:`_PINNED_PERF_KNOBS`) put four more
    codegen inputs on the spec and would have widened that gap again.

    ``launch`` is part of the identity and not derivable from the spec: stage 1
    of a split band and its stage 2 can hold specs agreeing on every field and
    still be different kernels, because a different builder consumes them.
    ``name`` is excluded -- it labels the identity rather than participating in
    it, so two bands that agreed on every knob would correctly be one binary.
    """
    payload = asdict(spec)
    payload.pop("name", None)
    struct: dict = {"path": _spec_path(spec), "launch": launch}
    struct.update(_json_safe(payload))
    return struct


def _json_safe(value):
    """Coerce a spec field into something ``stable_json_hash`` can hash.

    The f16 spec carries a nested trait dataclass, which ``asdict`` flattens,
    and enums or sets would not survive ``json.dumps`` at all. Falling back to
    ``repr`` keeps an unhashable field *in* the identity, which is the property
    that matters: a knob that cannot be serialized must not become a knob that
    is silently ignored.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (int, float, str)):
        return value
    return repr(value)


def _launch_of(candidate: KernelCandidate) -> str:
    return _LAUNCH_KIND[candidate.name]


def _kernel_id(req: MoeRequest, candidate: KernelCandidate, spec) -> KernelId:
    request_hash = stable_json_hash(req.normalized(), n=16)
    spec_hash = stable_json_hash(_struct(spec, launch=_launch_of(candidate)), n=16)
    return KernelId(
        op="moe",
        family=_FAMILY,
        candidate=candidate.name,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        arch=req.arch,
        abi_version=candidate.abi_version,
        request_hash=request_hash,
        spec_hash=spec_hash,
    )


def moe_sweep_space(req: OperatorRequest) -> Sequence[object]:
    if _request_errors(req):
        return ()
    specs = []
    seen = set()
    for candidate in MOE_REGISTRY.supported(req):
        spec = candidate.select_spec(req)
        h = stable_json_hash(_struct(spec, launch=_launch_of(candidate)), n=16)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return tuple(specs)


def _preconditions(spec) -> Tuple[str, ...]:
    """Host-side obligations this selection carries, as explanation lines.

    A spec is not always a complete answer: ``static_inter_scale`` moves work
    out of the kernel and onto the caller, and a caller who does not know that
    gets a kernel that runs at full speed and writes NaN -- no launch failure,
    no wrong-looking latency, just a silently poisoned result. Carrying it in
    the explanation keeps the obligation attached to the selection that
    incurred it rather than to documentation someone has to already be looking
    for.

    The deferred-knob line is the same idea one level down: while a pinned knob
    is not yet a field on the spec, the built kernel takes it from the
    instance's own default and the dispatcher is not in full control of it. The
    selection should say so rather than imply a completeness it does not have.
    """
    lines: list[str] = []
    if getattr(spec, "static_inter_scale", False):
        lines += [
            "REQUIRES: InterScale must hold a scale calibrated for this data "
            "before launch -- this spec reads the intermediate's fp8 scale "
            "instead of deriving it from the tile amax. Populate it by running "
            "the same spec with static_inter_scale=False once, or from an "
            "offline calibration. An uninitialised buffer yields NaN at full "
            "speed.",
            "NOTE: this is a numerics change, not only a scheduling one; the "
            "intermediate is quantized against a supplied divisor rather than "
            "its own amax.",
        ]
    if isinstance(spec, FusedMegaKernelSpecFp8):
        deferred = _deferred_perf_knobs()
        if deferred:
            lines.append(
                "NOTE: this build still inherits "
                + ", ".join(deferred)
                + " from the instance default; the dispatcher pins them once "
                "the spec declares them."
            )
    return tuple(lines)


def dispatch_moe_plan(
    req: MoeRequest,
    *,
    ranker: Ranker | None = None,
    with_prologue: bool = False,
) -> Tuple[DispatchResult, ...]:
    """Every launch the selected MoE algorithm needs, in execution order.

    One element for the single-launch mega-kernel, two for the partial-fusion
    pair (gate/up + requantize, then the down GEMM over H_out). Callers that
    execute a selection should use this rather than :func:`dispatch_moe`: the
    registry is single-kernel by design -- section 3.4 of ARCHITECTURE.md, so
    that a selection stays reproducible from a request alone -- which means a
    two-launch algorithm cannot be one candidate, and launching only the
    stage 1 of a split pair produces an intermediate and no output.

    Stage 2 is dispatched by explicit ``spec_id`` because it is not an answer
    to ``auto``; the band is still chosen once, by stage 1, so the two stages
    cannot disagree about geometry.

    ``with_prologue`` prepends the activation gather/rescale kernel. Opt-in
    rather than always-on because whether it is needed is a property of the
    caller's activations, not of the shape: a caller that already holds A
    gathered into expert-block order under a block-uniform scale does not need
    it, and one holding per-token-quantized activations does. Its ``tile_m``
    comes from the same band decision as the GEMM's, so the pair cannot
    disagree about the rows the activation scale is uniform over.
    """
    first = dispatch_moe(req, ranker=ranker)
    plan = [first]
    stage2_id = f"{first.candidate.name}{_STAGE2_SUFFIX}"
    if stage2_id in {c.name for c in MOE_REGISTRY.candidates()}:
        from dataclasses import replace as _replace

        plan.append(dispatch_moe(_replace(req, spec_id=stage2_id), ranker=ranker))
    if with_prologue:
        from dataclasses import replace as _replace

        prologue = dispatch_moe(
            _replace(req, algorithm="auto", spec_id=PROLOGUE_SPEC_ID), ranker=ranker
        )
        plan.insert(0, prologue)
    return tuple(plan)


def dispatch_moe(req: MoeRequest, *, ranker: Ranker | None = None) -> DispatchResult:
    """Select one fused-MoE kernel for ``req``.

    For the split bands this returns stage 1 only. Use
    :func:`dispatch_moe_plan` to get the whole algorithm.
    """
    candidate = MOE_REGISTRY.select(req, ranker=ranker)
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
            f"selected {candidate.name} for {req.dtype} fused MoE on {req.arch}",
            f"algorithm={candidate.algorithm}",
            f"spec_id={candidate.spec_id}",
            f"spec_hash={kid.spec_hash}",
            f"request_hash={kid.request_hash}",
        )
        + _preconditions(spec),
    )
