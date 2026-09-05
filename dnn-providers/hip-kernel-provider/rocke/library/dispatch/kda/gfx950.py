# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 chunkwise KDA candidates (CDNA4, bf16, K-packed bf16 MFMA).

``kernels/gfx950/kda_chunkwise.py`` has carried specs, validators, builders,
grid helpers and signatures since it landed, and nothing referenced them from
dispatch -- the family was compilable but unreachable. This module is the
registration that makes it selectable.

Three candidates, one per emitted kernel, because each is separately
compilable with its own grid and ABI:

* ``kda_gfx950_chunk_fused`` -- the whole prefill in one kernel. Default.
* ``kda_gfx950_chunk_prep``  -- split path, phase 1 (per-chunk tiles to HBM).
* ``kda_gfx950_chunk_scan``  -- split path, phase 2 (state scan over tiles).

The split halves are opt-in and must be launched in that order; see
:mod:`.common` for why there is no automatic fused/split routing yet.

The two paths do not share a tile schedule. The fused kernel carries its own
tuned 512-thread panel mapping; the split scan uses the kernel-owned measured
geometry selector to add value-band workgroups when ``batch * heads`` leaves
gfx950 underfilled. Every scan geometry remains capped so at least two
workgroups fit per CU.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

from kernels.gfx950.kda_chunkwise import (
    KDA_CHUNK_SIZES,
    KDA_DTYPES,
    KdaChunkFusedSpec,
    KdaChunkPrepSpec,
    KdaChunkScanSpec,
    KdaTileSpec,
    build_kda_chunk_fused,
    build_kda_chunk_prep,
    build_kda_chunk_scan,
    is_valid_fused_spec,
    is_valid_scan_spec,
    is_valid_spec,
    kda_chunk_fused_grid,
    kda_chunk_fused_signature,
    kda_chunk_prep_grid,
    kda_chunk_prep_signature,
    kda_chunk_scan_grid,
    kda_chunk_scan_signature,
    tuned_kda_chunk_scan_spec,
)
from rocke.dispatch.core import (
    CandidateRegistry,
    Capability,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    FAMILY,
    KDA_ABI_VERSION,
    KdaRequest,
    _request_errors,
    _selector_matches,
)

# Declared coverage, restating the validators' *data* gates so a request can be
# filtered without constructing a spec, and taking the numbers from the kernel
# rather than transcribing them. Everything the validators compute -- the LDS
# budget, the cumsum row-group divisibility, the wave/panel cover, the scan
# partition -- stays in the residual predicate, because it is a function of the
# spec rather than a property of the request.
#
# ``num_chunks >= 1`` is how a seqlen that does not tile the chunk is rejected:
# KdaRequest.num_chunks reports 0 for a ragged length, and this family has no
# varlen path. There is no head_v relation to declare: what makes a value width
# legal is that the workgroup's waves cover it exactly at the scan atom's M
# extent, and both of those come from the spec's tile.
_SHARED_SHAPES = (
    ShapeRange("chunk_size", allowed=KDA_CHUNK_SIZES),
    ShapeRange("head_k", multiple_of=8),
    ShapeRange("head_v", multiple_of=8),
    ShapeRange("num_chunks", min=1),
    ShapeRange("batch", min=1),
    ShapeRange("num_heads", min=1),
)
# Both state flags describe the *problem*, so every candidate that can take
# part in serving it declares them -- including the tile builder, which is
# state-independent and simply leaves them to the scan half. Withholding them
# there would make the split path unable to serve any problem that wants a
# final state, which is the opposite of true: the split path is how you get one
# without the fused kernel.
_STATE_FEATURES = frozenset({"initial_state", "final_state"})

# The kernel's own schedules, read rather than copied, so a retune of either
# path moves the dispatcher with it.
_FUSED_TILE = KdaChunkFusedSpec().tile
_SPLIT_TILE = KdaTileSpec()


def _fused_tile(req: KdaRequest) -> KdaTileSpec:
    return dataclasses.replace(_FUSED_TILE, chunk=req.effective_chunk_size)


def _split_tile(req: KdaRequest) -> KdaTileSpec:
    return dataclasses.replace(_SPLIT_TILE, chunk=req.effective_chunk_size)


def _fused_spec(req: OperatorRequest) -> KdaChunkFusedSpec:
    assert isinstance(req, KdaRequest)
    return KdaChunkFusedSpec(
        head_k=int(req.head_k),
        head_v=int(req.head_v),
        dtype=req.dtype.lower(),
        tile=_fused_tile(req),
        has_initial_state=bool(req.has_initial_state),
        store_final_state=bool(req.store_final_state),
    )


def _prep_spec(req: OperatorRequest) -> KdaChunkPrepSpec:
    assert isinstance(req, KdaRequest)
    return KdaChunkPrepSpec(
        head_k=int(req.head_k),
        head_v=int(req.head_v),
        dtype=req.dtype.lower(),
        tile=_split_tile(req),
    )


def _scan_spec(req: OperatorRequest) -> KdaChunkScanSpec:
    """Select the pipelined scan and its measured value-band geometry."""
    assert isinstance(req, KdaRequest)
    return tuned_kda_chunk_scan_spec(
        req.workgroups,
        head_k=int(req.head_k),
        head_v=int(req.head_v),
        dtype=req.dtype.lower(),
        chunk=req.effective_chunk_size,
        has_initial_state=bool(req.has_initial_state),
        store_final_state=bool(req.store_final_state),
    )


def _scan_validator(spec: KdaChunkScanSpec, arch: str) -> Tuple[bool, str]:
    """A scan is selectable only if the tile builder that feeds it is too.

    The scan validates the staging copies and the state partition; the tile
    builder additionally validates the grouped cumulative sum, whose row-group
    rules the scan has no equivalent of. So a spec can pass here and leave the
    pair unusable -- ``head_k=32`` at the shipped split schedule is one -- and
    admitting that would move the failure to whoever tried to launch the two
    halves in order.
    """
    ok, why = is_valid_scan_spec(spec, arch=arch)
    if not ok:
        return False, why
    ok, why = is_valid_spec(spec.prep, arch=arch)
    if not ok:
        return False, f"tile builder for this scan is unbuildable: {why}"
    return True, "ok"


def _make_candidate(
    *,
    name: str,
    algorithm: str,
    spec_id: str,
    priority: int,
    capability: Capability,
    spec_for,
    validator,
    builder,
    grid_for,
    signature_for,
    opt_in_reason: str = "",
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, KdaRequest)
        if (
            opt_in_reason
            and req.algorithm.strip().lower() != algorithm
            and (req.spec_id.strip().lower() != spec_id)
        ):
            return False, opt_in_reason
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # Capability already cleared arch, dtype and chunk, so the spec
        # construction below cannot raise. Only the residual, spec-computed
        # gates run here.
        return validator(spec_for(req), arch=req.arch)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return spec_for(req)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=KDA_ABI_VERSION,
        priority=priority,
        capability=capability,
        _supports=support,
        select_spec=select,
        build=builder,
        grid=grid_for,
        block=lambda spec: (spec.tile.block_size, 1, 1),
        signature=signature_for,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _fused_grid(spec: KdaChunkFusedSpec, req: OperatorRequest):
    assert isinstance(req, KdaRequest)
    return kda_chunk_fused_grid(spec, req.workgroups)


def _scan_grid(spec: KdaChunkScanSpec, req: OperatorRequest):
    assert isinstance(req, KdaRequest)
    # The helper applies the spec's own ``value_splits`` v-band multiplier, so
    # this stays the count of (batch, head) recurrences.
    return kda_chunk_scan_grid(spec, req.workgroups)


def _prep_grid(spec: KdaChunkPrepSpec, req: OperatorRequest):
    assert isinstance(req, KdaRequest)
    # One workgroup per chunk -- the same ``BH * NC`` tile count the host
    # builder allocates and packs.
    return kda_chunk_prep_grid(spec, req.workgroups * req.num_chunks)


_SPLIT_OPT_IN = (
    "the gfx950 KDA split path is opt-in (algorithm='chunk_prep' / "
    "'chunk_scan'); the fused/split crossover has been measured but not at "
    "enough shapes to encode a threshold, so default routing stays on the "
    "fused kernel"
)


def _capability() -> Capability:
    return Capability(
        arches=("gfx950",),
        dtypes=KDA_DTYPES,
        shapes=_SHARED_SHAPES,
        supports_features=_STATE_FEATURES,
    )


def _fused_candidate() -> KernelCandidate:
    """The default gfx950 prefill path.

    Default rather than opt-in because this is the family's first routing
    decision, not a re-route of a benchmarked path: nothing dispatched KDA
    before. It keeps the six per-chunk tiles in LDS, so the only HBM traffic is
    q/k/g/beta in and o plus the final state out, and it is one launch rather
    than two -- which is also why it is the half that can carry a ``bind``
    when the launch seam moves.
    """
    return _make_candidate(
        name="kda_gfx950_chunk_fused",
        algorithm="chunk_fused",
        spec_id="gfx950_chunk_fused",
        priority=10,
        capability=_capability(),
        spec_for=_fused_spec,
        validator=is_valid_fused_spec,
        builder=build_kda_chunk_fused,
        grid_for=_fused_grid,
        signature_for=kda_chunk_fused_signature,
    )


def _prep_candidate() -> KernelCandidate:
    """Split path phase 1: the state-independent per-chunk tile builder.

    Fully parallel over the sequence, one workgroup per chunk, which is where
    the split path's parallelism advantage at small ``B*H`` comes from.
    ``KdaChunkPrepSpec`` has no state fields at all, so the request's state
    flags do not reach this kernel -- the scan half applies them. It still
    declares the state features, for the reason given at ``_STATE_FEATURES``.
    """
    return _make_candidate(
        name="kda_gfx950_chunk_prep",
        algorithm="chunk_prep",
        spec_id="gfx950_chunk_prep",
        priority=20,
        capability=_capability(),
        spec_for=_prep_spec,
        validator=is_valid_spec,
        builder=build_kda_chunk_prep,
        grid_for=_prep_grid,
        signature_for=kda_chunk_prep_signature,
        opt_in_reason=_SPLIT_OPT_IN,
    )


def _scan_candidate() -> KernelCandidate:
    """Split path phase 2: the state scan over materialized tiles.

    Consumes what ``kda_gfx950_chunk_prep`` wrote, so it is only meaningful
    after that kernel has run on the same problem. ``_scan_spec`` derives the
    adaptive value split from the request's recurrence-stream count, while the
    builder enables the register-held next-chunk pipeline for eligible C32
    schedules.
    """
    return _make_candidate(
        name="kda_gfx950_chunk_scan",
        algorithm="chunk_scan",
        spec_id="gfx950_chunk_scan",
        priority=20,
        capability=_capability(),
        spec_for=_scan_spec,
        validator=_scan_validator,
        builder=build_kda_chunk_scan,
        grid_for=_scan_grid,
        signature_for=kda_chunk_scan_signature,
        opt_in_reason=_SPLIT_OPT_IN,
    )


def register(registry: CandidateRegistry) -> None:
    registry.register(_fused_candidate())
    registry.register(_prep_candidate())
    registry.register(_scan_candidate())
