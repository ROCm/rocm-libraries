# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate registration for GDN decode.

The registry owns the configured tile space. The kernel validator remains the
only authority that decides which configured tiles are legal for a request.
Production ``auto`` prefers one tile per request (``auto_tile``: GDN's static
default, KDA's work-keyed table, or the shared fused default) and never
consults runtime measurements.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product
from numbers import Integral
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
# Both this table and DEFAULT_TILE below were measured on the interleaved
# load order, before the warp-tiled emitter issued every load first
# (ALGORITHM.md §4.3); re-run tune.py before relying on their ranking.
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
# batch's fastest tile, though it is not the fastest at any single batch (on
# the interleaved load order; see above). To revisit, run
# ``tune.py --gate-kind gdn``; it reports this default's rank and its ratio to
# the fastest legal tile per batch.
DEFAULT_TILE = (2, 16, 8)

_LEXICOGRAPHIC_TILES = tuple(product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM))
# Known limitation: when DEFAULT_TILE is illegal (e.g. head_k_dim 64/192), the
# fallback is the first legal tile in product order, (1,1,1), which is
# register-heavy. Needs a footprint-based fallback order.
CONFIGURED_TILES = (DEFAULT_TILE,) + tuple(
    tile for tile in _LEXICOGRAPHIC_TILES if tile != DEFAULT_TILE
)

# Fused (fuse_conv / fuse_out_norm) requests need blocks_per_v_dim == 1, which
# GDN's DEFAULT_TILE is not. Interim static fused default for both gate kinds
# and every state width (ALGORITHM.md §4.6): the tile the fused numeric suite is
# built around, not a measured winner. It needs head_k_dim % 128 == 0 and
# head_v_dim % 16 == 0; otherwise (e.g. GDN head_k_dim 64) a fused request still
# falls back to (1,1,1), as for DEFAULT_TILE above. KDA registers it as one
# extra candidate (below).
FUSED_DEFAULT_TILE = (4, 16, 1)
assert FUSED_DEFAULT_TILE in CONFIGURED_TILES and FUSED_DEFAULT_TILE[2] == 1


def spec_id_for(tile: Tuple[int, int, int]) -> str:
    """The spec id a configured tile registers under."""
    return f"nw{tile[0]}_wtk{tile[1]}_bpv{tile[2]}"


KDA_FUSED_SPEC_ID = "kda_" + spec_id_for(FUSED_DEFAULT_TILE)

TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES_KDA)


def work_for(batch: int, num_v_heads: int) -> int:
    """KDA's measured selection quantity."""
    return batch * num_v_heads


def tile_for_work(work: int, gate_kind: str = "kda") -> Tuple[int, int, int]:
    if gate_kind != "kda":
        raise ValueError(f"no work-keyed table for gate kind {gate_kind!r}")
    for max_work, tile, _ in _TUNED_TILES_KDA:
        if max_work is None or work <= max_work:
            return tile
    raise AssertionError("unreachable: KDA table has an open-ended final band")


def _gate_kind_for_spec_id(spec_id: str) -> str:
    """Which gate kind's table a spec id belongs to."""
    if spec_id == KDA_FUSED_SPEC_ID or any(
        sid == spec_id for _, _, sid in _TUNED_TILES_KDA
    ):
        return "kda"
    return "gdn"


def is_fused(req: GdnDecodeRequest) -> bool:
    return req.fuse_conv or req.fuse_out_norm


def auto_tile(req: GdnDecodeRequest) -> Tuple[int, int, int]:
    """The tile ``auto`` prefers for ``req`` whenever it is legal.

    Fused requests: ``FUSED_DEFAULT_TILE`` for both gate kinds. Unfused: GDN's
    static ``DEFAULT_TILE``, KDA's work-keyed table.
    """
    if is_fused(req):
        return FUSED_DEFAULT_TILE
    if req.gate_kind == "kda":
        return tile_for_work(work_for(req.batch, req.num_v_heads), "kda")
    return DEFAULT_TILE


def make_spec(req: GdnDecodeRequest, tile: Tuple[int, int, int]) -> GdnDecodeSpec:
    """Map a request plus a chosen tile onto a concrete kernel spec.

    Fields that ``is_valid_spec`` type-checks pass through unconverted, so a
    malformed request (``use_qk_l2norm="false"``, ``num_k_heads=True``) is
    rejected by that validator instead of being coerced into a different valid
    spec. ``normalize_dtype`` stays: it maps documented dtype aliases. A copied
    dim is cast only when it is an Integral that is not a bool -- a numpy
    integer canonicalizes to plain ``int`` for the JSON-hashed identity
    payload, while a malformed one (``True``, ``4.0``) passes through raw so
    ``is_valid_spec`` still rejects it.
    """
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    cast_dim = lambda v: (
        int(v) if isinstance(v, Integral) and not isinstance(v, bool) else v
    )
    return dc.replace(
        GdnDecodeSpec(),
        num_k_heads=cast_dim(req.num_k_heads),
        num_v_heads=cast_dim(req.num_v_heads),
        head_k_dim=cast_dim(req.head_k_dim),
        head_v_dim=cast_dim(req.head_v_dim),
        dtype=normalize_dtype(req.dtype),
        state_dtype=normalize_dtype(req.state_dtype),
        use_qk_l2norm=req.use_qk_l2norm,
        gate_kind=req.gate_kind,
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
        fuse_conv=req.fuse_conv,
        fuse_out_norm=req.fuse_out_norm,
    )


def _grid(spec: GdnDecodeSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GdnDecodeRequest)
    return gdn_decode_grid(req.batch, spec)


def _build(spec: GdnDecodeSpec, arch: str):
    return build_gdn_decode(spec, arch=arch)


def _make_candidate(
    *,
    tile: Tuple[int, int, int],
    priority: int,
    spec_id: str | None = None,
):
    spec_id = spec_id or spec_id_for(tile)
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
        if req.spec_id.strip().lower() == "auto":
            # ``auto`` is one preferred tile per request mode (auto_tile);
            # when that tile is illegal for the request, every legal
            # candidate stays admitted and priority order picks the first.
            wanted = auto_tile(req)
            if (
                tile != wanted
                and is_valid_spec(make_spec(req, wanted), arch=req.arch)[0]
            ):
                return False, (
                    f"static {req.gate_kind.upper()} auto tile for this request "
                    f"is {wanted!r}, not {tile!r}"
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
    """GDN configured candidates, KDA measured candidates, then KDA's fused
    default."""
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
    # The KDA fused default lives outside the work-keyed table, so an unfused
    # KDA auto reaches it only as a fallback, after every band tile.
    kda_fused = (
        _make_candidate(
            tile=FUSED_DEFAULT_TILE,
            priority=10 + len(gdn) + len(kda),
            spec_id=KDA_FUSED_SPEC_ID,
        ),
    )
    return gdn + kda + kda_fused


def register(registry) -> None:
    registry.extend(candidates())
