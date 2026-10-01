# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 candidate and tuned tile selection for GDN decode.

This module owns the two arch-specific decisions: what the chip can serve
(``Capability`` plus the residual predicate, which ends in the kernel's own
``is_valid_spec``) and how a request is turned into a concrete spec.

The tile selection is the interesting part. ``blocks_per_v_dim`` splits each
value head's V dimension across several workgroups purely to manufacture
parallelism; it costs redundant work per split. At small batch there are too
few sequences to fill the machine, so paying that cost buys occupancy. As the
batch grows the launch already has ample parallelism and the split becomes
overhead, so the tuned tile collapses to one workgroup per head and instead
widens the workgroup. That trend, not any individual measurement, is what this
table encodes.
"""

from __future__ import annotations

import dataclasses as dc
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

# GDN keeps its original batch-keyed table. KDA is keyed on
# WORK = batch * num_v_heads because that table was measured across head-count
# geometries. The distinction is deliberate: changing GDN to work-keying
# reroutes already-supported sharded-head requests without GDN measurements,
# which is independent performance work and does not belong in this PR.
#
# (max_batch, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
_TUNED_TILES_GDN = (
    (4, (4, 16, 8), "b4"),
    (32, (2, 8, 2), "b32"),
    (128, (1, 8, 1), "b128"),
    (None, (8, 16, 1), "b_large"),
)

# KDA: measured on gfx950 with the per-channel gate. All 54 legal tiles were
# enumerated through is_valid_spec and correctness-gated against the fp32
# reference before timing; survivors were timed with a device clock (replayed
# HIP graph, so host submission is off the critical path).
#
# Measured: work in {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}, realised
# across three head geometries (Hk/Hv of 16/32, 8/16, 4/8) x eight batches --
# 24 cells, 54 configurations each. Band edges BETWEEN those anchors are
# interpolation, not measured crossovers.
#
# Three bands, not more. Four bands improve the geomean by 0.75pp and five by a
# further 0.37pp, both inside the 0.3-2.2% spread measured between adjacent
# configurations at the same anchor -- and the extra edges land at work 32/64,
# exactly where the sweep shows tile choice is already inside run-to-run
# variation. More bands here would be fitting noise.
#
# What this table claims: only that the chosen tile is close to the best LEGAL
# ROCKE tile at the measured anchors -- geomean 1.016, worst 1.042 against the
# per-anchor optimum, against 1.088 / 1.304 for the best single universal tile.
# It is a claim against our own tile space and nothing else.
#
# Validated out of sample on the MHA shape that ships (Hk=Hv=32), which was
# NOT in the fit, and this banding scores geomean 1.015, worst 1.053 there.
_TUNED_TILES_KDA = (
    (128, (4, 16, 4), "kda_w128"),
    (512, (1, 16, 4), "kda_w512"),
    (None, (2, 16, 1), "kda_w_large"),
)


def _tuned_tiles(gate_kind: str):
    return _TUNED_TILES_KDA if gate_kind == "kda" else _TUNED_TILES_GDN


# Every tile the tables can produce, for tuners and for the sweep space.
TUNED_SPEC_IDS = tuple(e[2] for e in _TUNED_TILES_GDN + _TUNED_TILES_KDA)


def tile_for_batch(batch: int) -> Tuple[int, int, int]:
    """Original GDN tile selection, keyed on batch."""
    for max_batch, tile, _ in _TUNED_TILES_GDN:
        if max_batch is None or batch <= max_batch:
            return tile
    raise AssertionError("unreachable: table has an open-ended final band")


def spec_id_for_batch(batch: int) -> str:
    for max_batch, _, spec_id in _TUNED_TILES_GDN:
        if max_batch is None or batch <= max_batch:
            return spec_id
    raise AssertionError("unreachable: table has an open-ended final band")


def work_for(batch: int, num_v_heads: int) -> int:
    """The quantity the tile tables are keyed on."""
    return int(batch) * int(num_v_heads)


def tile_for_work(work: int, gate_kind: str = "gdn") -> Tuple[int, int, int]:
    """Tuned ``(num_warps, warp_threads_k, blocks_per_v_dim)`` for ``work``."""
    for max_work, tile, _ in _tuned_tiles(gate_kind):
        if max_work is None or work <= max_work:
            return tile
    raise AssertionError("unreachable: table has an open-ended final band")


def spec_id_for_work(work: int, gate_kind: str = "gdn") -> str:
    for max_work, _, spec_id in _tuned_tiles(gate_kind):
        if max_work is None or work <= max_work:
            return spec_id
    raise AssertionError("unreachable: table has an open-ended final band")


def _tile_for_spec_id(spec_id: str) -> Tuple[int, int, int]:
    for _, tile, sid in _TUNED_TILES_GDN + _TUNED_TILES_KDA:
        if sid == spec_id:
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


def _make_candidate(*, tile: Tuple[int, int, int], spec_id: str, priority: int):
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
        # Under ``auto`` only the candidate the tuning table names may serve the
        # request, so selection is decided by measurement rather than by
        # registration order. An explicit ``spec_id`` pin bypasses this, which
        # is what makes a tuning sweep able to force a non-default tile.
        if req.spec_id.strip().lower() == "auto":
            if req.gate_kind == "kda":
                wanted = spec_id_for_work(work_for(req.batch, req.num_v_heads), "kda")
            else:
                wanted = spec_id_for_batch(req.batch)
            # Prefer the tuned tile, but only when it is valid for this geometry.
            # If it is not, fall through so any valid candidate may serve (the
            # registry picks by priority) rather than failing a kernel-supported
            # request.
            if (
                wanted != spec_id
                and is_valid_spec(
                    make_spec(req, _tile_for_spec_id(wanted)), arch=req.arch
                )[0]
            ):
                return False, (
                    f"tuned tile for work {work_for(req.batch, req.num_v_heads)} "
                    f"(batch {req.batch} x {req.num_v_heads} heads) is {wanted!r}, "
                    f"not {spec_id!r}"
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
    """One candidate per tuned tile, GDN's table then KDA's, in table order.

    Both kinds are registered together; each candidate's ``support`` admits
    only its own gate kind, so the tables cannot cross-serve.
    """
    entries = _TUNED_TILES_GDN + _TUNED_TILES_KDA
    return tuple(
        _make_candidate(tile=tile, spec_id=spec_id, priority=10 + i)
        for i, (_, tile, spec_id) in enumerate(entries)
    )


def register(registry) -> None:
    registry.extend(candidates())
