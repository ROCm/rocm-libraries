# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate registration for GDN decode.

The registry owns the configured tile space. The kernel validator remains the
only authority that decides which configured tiles are legal for a request.
Production ``auto`` uses one static priority order and never consults batch or
runtime measurements.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product
from typing import Tuple

from kernels.gfx950.gdn_decode import (
    GDN_DTYPES,
    GdnDecodeSpec,
    build_gdn_decode,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.dispatch.core import Capability, KernelCandidate, OperatorRequest

from .common import (
    FAMILY,
    GDN_ABI_VERSION,
    GdnDecodeRequest,
    normalize_dtype,
    request_errors,
    selector_matches,
)

ARCH = "gfx950"

# KDA keeps its measured work-keyed table. GDN candidate registration below
# owns the full configured tile space and has no GDN batch-winner table.
#
# (max_work, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
_TUNED_TILES_KDA = (
    (128, (4, 16, 4), "kda_w128"),
    (512, (1, 16, 4), "kda_w512"),
    (None, (2, 16, 1), "kda_w_large"),
)

NUM_WARPS = (1, 2, 4, 8, 16)
WARP_THREADS_K = (1, 2, 4, 8, 16, 32)
BLOCKS_PER_V_DIM = (1, 2, 4, 8, 16, 32)
# GDN ``auto`` is one static tile on purpose; batch only changes the grid.
# This replaced a batch-keyed table that picked (4,16,8) / (2,8,2) / (1,8,1) /
# (8,16,1) for batch <=4 / <=32 / <=128 / larger. No single legal tile matches
# all four of those winners. (2,16,8) was measured on gfx950 against that table
# at batch 1/16/64/256: batch 256 was neutral, batch 64 is slower, and batch 1
# and 16 read slower but were too noisy to call. The batch-64 cost was accepted
# in exchange for one deterministic default. Across every legal tile at those
# batches, (2,16,8) has the lowest geomean and worst-case slowdown against each
# batch's fastest tile, though it is not the fastest at any single batch. To
# revisit, run ``tune.py --gate-kind gdn``; it reports this default's rank and
# its ratio to the fastest legal tile per batch.
DEFAULT_TILE = (2, 16, 8)

_LEXICOGRAPHIC_TILES = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))
CONFIGURED_TILES = (DEFAULT_TILE,) + tuple(
    tile for tile in _LEXICOGRAPHIC_TILES if tile != DEFAULT_TILE
)

TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES_KDA)


def work_for(batch: int, num_v_heads: int) -> int:
    """KDA's measured selection quantity."""
    return int(batch) * int(num_v_heads)


def tile_for_work(work: int, gate_kind: str = "kda") -> Tuple[int, int, int]:
    if gate_kind != "kda":
        raise ValueError(f"no work-keyed table for gate kind {gate_kind!r}")
    for max_work, tile, _ in _TUNED_TILES_KDA:
        if max_work is None or work <= max_work:
            return tile
    raise AssertionError("unreachable: KDA table has an open-ended final band")


def spec_id_for_work(work: int) -> str:
    for max_work, _, spec_id in _TUNED_TILES_KDA:
        if max_work is None or work <= max_work:
            return spec_id
    raise AssertionError("unreachable: KDA table has an open-ended final band")


def _tile_for_kda_spec_id(spec_id: str) -> Tuple[int, int, int]:
    for _, tile, candidate_spec_id in _TUNED_TILES_KDA:
        if candidate_spec_id == spec_id:
            return tile
    raise KeyError(spec_id)


def _gate_kind_for_spec_id(spec_id: str) -> str:
    """Which gate kind's table a spec id belongs to."""
    if any(sid == spec_id for _, _, sid in _TUNED_TILES_KDA):
        return "kda"
    return "gdn"


def make_spec(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    return dc.replace(
        GdnDecodeSpec(),
        num_k_heads=int(req.num_k_heads),
        num_v_heads=int(req.num_v_heads),
        head_k_dim=int(req.head_k_dim),
        head_v_dim=int(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        state_dtype=normalize_dtype(req.state_dtype),
        use_qk_l2norm=bool(req.use_qk_l2norm),
        gate_kind=str(req.gate_kind),
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
    )


def _grid(spec: GdnDecodeSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GdnDecodeRequest)
    return gdn_decode_grid(int(req.batch), spec)


def _build(spec: GdnDecodeSpec, arch: str):
    return build_gdn_decode(spec, arch=arch)


def _make_candidate(
    *,
    tile: Tuple[int, int, int],
    priority: int,
    spec_id: str | None = None,
):
    spec_id = spec_id or f"nw{tile[0]}_wtk{tile[1]}_bpv{tile[2]}"
    name = f"gdn_decode_{ARCH}_{spec_id}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GdnDecodeRequest)
        if req.arch != ARCH:
            return False, f"candidate arch {ARCH} != request arch {req.arch!r}"
        # A candidate belongs to exactly one gate kind's table. Serving the
        # other kind would hand the request a tile tuned for a different
        # kernel, which is the failure the split table exists to prevent.
        if req.gate_kind != _gate_kind_for_spec_id(spec_id):
            return False, (
                f"candidate {spec_id!r} is tuned for the "
                f"{_gate_kind_for_spec_id(spec_id)!r} gate, request asks for "
                f"{req.gate_kind!r}"
            )
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if req.spec_id.strip().lower() == "auto" and req.gate_kind == "gdn":
            default_is_legal = is_valid_spec(
                make_spec(req, DEFAULT_TILE), arch=req.arch
            )[0]
            if default_is_legal and tile != DEFAULT_TILE:
                return False, (
                    f"static GDN auto tile is {DEFAULT_TILE!r}, not {tile!r}"
                )
        if req.spec_id.strip().lower() == "auto" and req.gate_kind == "kda":
            wanted = spec_id_for_work(work_for(req.batch, req.num_v_heads))
            if (
                wanted != spec_id
                and is_valid_spec(
                    make_spec(req, _tile_for_kda_spec_id(wanted)), arch=req.arch
                )[0]
            ):
                return False, (
                    f"tuned KDA tile for work {work_for(req.batch, req.num_v_heads)} "
                    f"is {wanted!r}, not {spec_id!r}"
                )
        # Final authority is the kernel's own validator.
        return is_valid_spec(make_spec(req, tile), arch=req.arch)

    def select(req: OperatorRequest) -> GdnDecodeSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GdnDecodeRequest)
        return make_spec(req, tile)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="warp_tiled",
        spec_id=spec_id,
        abi_version=GDN_ABI_VERSION,
        priority=priority,
        capability=Capability(arches=(ARCH,), dtypes=GDN_DTYPES),
        _supports=support,
        select_spec=select,
        signature=lambda spec: gdn_decode_signature(spec),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
        # No `bind`: this family is selectable but not launchable through the
        # generic runner, so today EVERY launch goes through the driver's
        # `prepare()` and therefore through `_validate_decode_inputs`. That is
        # the only thing standing between a mis-shaped tensor and an
        # out-of-bounds access -- the kernel emits no buffer descriptor, so
        # there is no `num_records` to clamp one. Whoever adds `bind` must
        # route it through that validator, or the checks stop covering the
        # path callers actually use.
    )
    return candidate


def candidates() -> Tuple[KernelCandidate, ...]:
    """GDN configured candidates followed by KDA measured candidates."""
    gdn = tuple(
        _make_candidate(tile=tile, priority=10 + i)
        for i, tile in enumerate(CONFIGURED_TILES)
    )
    kda = tuple(
        _make_candidate(
            tile=tile,
            priority=10 + len(gdn) + i,
            spec_id=spec_id,
        )
        for i, (_, tile, spec_id) in enumerate(_TUNED_TILES_KDA)
    )
    return gdn + kda


def register(registry) -> None:
    registry.extend(candidates())
