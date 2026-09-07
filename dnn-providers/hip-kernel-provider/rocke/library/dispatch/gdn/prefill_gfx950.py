# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidates and tuned ``value_splits`` selection for GDN prefill.

The two split halves of the chunkwise KDA kernel, run in GDN mode:

* ``chunk_prep`` -- the state-independent per-chunk tile builder, with the raw
  token-major GDN gate fused in (``gate_kind="gdn"``, ``kv_group`` GQA gather,
  q/k L2-norm + softplus gate + beta sigmoid all fused).
* ``chunk_scan`` -- the serial state scan over the materialized tiles, whose
  ``value_splits`` fans the value dimension across workgroups to manufacture
  parallelism at low ``BH``.

Both are selected by an explicit ``algorithm`` pin; there is no fused GDN
kernel, so ``auto`` does not resolve here (see :mod:`.prefill_common`).

Tuned ``value_splits`` table
----------------------------
Measured on gfx950 (correctness-gated vs the fp64 oracle): the
serial scan is parallelism-starved at small ``BH`` and ``value_splits`` fills
the GPU. The optimum is per-``BH``; each split also fixes the scan tile's block
size (and the ``vs=8`` atom), because the scan block must cover the split V
extent. Raw prep always keeps the 256-thread builder regardless of the split.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

from kernels.gfx950.kda_chunkwise import (
    KDA_DTYPES,
    KdaChunkPrepSpec,
    KdaChunkScanSpec,
    KdaTileSpec,
    build_kda_chunk_prep,
    build_kda_chunk_scan,
    is_valid_scan_spec,
    is_valid_spec,
    kda_chunk_prep_grid,
    kda_chunk_prep_signature,
    kda_chunk_scan_grid,
    kda_chunk_scan_signature,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
)

from .common import normalize_dtype
from .prefill_common import (
    FAMILY_PREFILL,
    GDN_PREFILL_ABI_VERSION,
    GdnPrefillRequest,
    prefill_request_errors,
    prefill_selector_matches,
)

ARCH = "gfx950"

# (max_batch_heads, value_splits). BH = batch * num_v_heads. The final band is
# open-ended (value_splits=1, natural parallelism fills the GPU). Measured
# anchors: BH<=64 -> value_splits=8, BH<=128 -> 2. Ratios in the internal perf repo.
# Snapshot: regenerate with
# ``python -m benchmarks.gfx950.gdn.sweep_prefill_value_splits`` after any
# kernel, compiler, or shape change -- a baked table drifts otherwise.
_VALUE_SPLIT_BANDS = (
    (64, 8),
    (128, 2),
)
_DEFAULT_VALUE_SPLITS = 1

# Each value_splits fixes the scan tile: the scan block must cover the split V
# extent, and vs=8 additionally needs the M16 scan atom. Mirrors the builder's
# ``aligned_split_specs``. Raw prep overrides block_size back to 256. Only the
# splits the bands actually select (8/2/1) are listed; the builder also defines
# vs=4, but no band picks it here so it is intentionally absent.
_SPLIT_TILE = {
    8: dict(block_size=64, scan_atom_m=16),
    2: dict(block_size=128),
    1: dict(block_size=256),
}


def value_splits_for(batch_heads: int) -> int:
    """Tuned ``value_splits`` for ``BH = batch * num_v_heads``."""
    for max_bh, splits in _VALUE_SPLIT_BANDS:
        if batch_heads <= max_bh:
            return splits
    return _DEFAULT_VALUE_SPLITS


def _scan_tile(req: GdnPrefillRequest, value_splits: int) -> KdaTileSpec:
    return KdaTileSpec(chunk=req.effective_chunk_size, **_SPLIT_TILE[value_splits])


def _scan_spec(req: OperatorRequest) -> KdaChunkScanSpec:
    assert isinstance(req, GdnPrefillRequest)
    value_splits = value_splits_for(req.batch_heads)
    return KdaChunkScanSpec(
        head_k=int(req.head_k_dim),
        head_v=int(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        tile=_scan_tile(req, value_splits),
        value_splits=value_splits,
        token_major_io=True,
        has_initial_state=bool(req.has_initial_state),
        store_final_state=bool(req.store_final_state),
    )


def _prep_spec(req: OperatorRequest) -> KdaChunkPrepSpec:
    assert isinstance(req, GdnPrefillRequest)
    # The raw GDN prep is derived from the scan's tile, but keeps the 256-thread
    # builder (block_size=256) even when the scan uses a narrow block for a
    # value split -- exactly the builder's ``prep_spec_of(scan, raw=True)``.
    scan = _scan_spec(req)
    prep_tile = dataclasses.replace(scan.tile, block_size=256)
    return KdaChunkPrepSpec(
        head_k=int(req.head_k_dim),
        head_v=int(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        tile=prep_tile,
        raw_inputs=True,
        fuse_qk_l2norm=True,
        fuse_gate=True,
        fuse_beta_sigmoid=True,
        has_dt_bias=True,
        lower_bound=-5.0,
        gate_kind="gdn",
        kv_group=int(req.kv_group),
    )


def _scan_validator(spec: KdaChunkScanSpec, arch: str) -> Tuple[bool, str]:
    """A scan is selectable only if its standalone spec is valid on ``arch``.

    The paired raw prep's validity (GDN flags, GQA group) is the ``chunk_prep``
    candidate's own gate; both halves must admit for the split path to serve a
    request, so the scan need not re-validate the prep here.
    """
    return is_valid_scan_spec(spec, arch=arch)


def _prep_grid(spec: KdaChunkPrepSpec, req: OperatorRequest):
    assert isinstance(req, GdnPrefillRequest)
    # One workgroup per (batch, value head, chunk).
    return kda_chunk_prep_grid(spec, req.workgroups * req.num_chunks)


def _scan_grid(spec: KdaChunkScanSpec, req: OperatorRequest):
    assert isinstance(req, GdnPrefillRequest)
    # One recurrence stream per (batch, value head); the grid helper fans each
    # out by ``spec.value_splits``.
    return kda_chunk_scan_grid(spec, req.workgroups)


def _capability() -> Capability:
    # Both state flags describe the problem; the scan half applies them and the
    # prep half is state-independent, so both candidates declare them (see KDA).
    return Capability(
        arches=(ARCH,),
        dtypes=KDA_DTYPES,
        supports_features=frozenset({"initial_state", "final_state"}),
    )


_SPLIT_ONLY = (
    "GDN prefill is a two-phase split path; pin algorithm='chunk_prep' then "
    "'chunk_scan'. There is no fused GDN kernel -- the fused path is packed-only "
    "and cannot emit the in-kernel GDN gate."
)


def _make_candidate(
    *,
    name: str,
    algorithm: str,
    spec_id: str,
    priority: int,
    spec_for,
    validator,
    builder,
    grid_for,
    signature_for,
) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = prefill_request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnPrefillRequest)
        if req.arch != ARCH:
            return False, f"candidate arch {ARCH} != request arch {req.arch!r}"
        # Opt-in by algorithm: there is no fused default, so a bare "auto"
        # request matches neither half rather than silently picking one.
        if req.algorithm.strip().lower() not in (algorithm, ""):
            if req.spec_id.strip().lower() != spec_id:
                return False, _SPLIT_ONLY
        ok, why = prefill_selector_matches(req, candidate)
        if not ok:
            return False, why
        return validator(spec_for(req), arch=req.arch)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return spec_for(req)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY_PREFILL,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=GDN_PREFILL_ABI_VERSION,
        priority=priority,
        capability=_capability(),
        _supports=support,
        select_spec=select,
        build=builder,
        grid=grid_for,
        block=lambda spec: (spec.tile.block_size, 1, 1),
        signature=signature_for,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _prep_candidate() -> KernelCandidate:
    """Split path phase 1: the raw GDN per-chunk tile builder."""
    return _make_candidate(
        name="gdn_prefill_gfx950_chunk_prep",
        algorithm="chunk_prep",
        spec_id="gfx950_gdn_chunk_prep",
        priority=10,
        spec_for=_prep_spec,
        validator=is_valid_spec,
        builder=build_kda_chunk_prep,
        grid_for=_prep_grid,
        signature_for=kda_chunk_prep_signature,
    )


def _scan_candidate() -> KernelCandidate:
    """Split path phase 2: the serial state scan, ``value_splits``-tuned."""
    return _make_candidate(
        name="gdn_prefill_gfx950_chunk_scan",
        algorithm="chunk_scan",
        spec_id="gfx950_gdn_chunk_scan",
        priority=20,
        spec_for=_scan_spec,
        validator=_scan_validator,
        builder=build_kda_chunk_scan,
        grid_for=_scan_grid,
        signature_for=kda_chunk_scan_signature,
    )


def candidates() -> Tuple[KernelCandidate, ...]:
    return (_prep_candidate(), _scan_candidate())


def register(registry: CandidateRegistry) -> None:
    registry.extend(candidates())
