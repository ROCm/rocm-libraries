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

# (max_batch, (num_warps, warp_threads_k, blocks_per_v_dim), spec_id)
#
# Device-time optimum per batch, measured on gfx950 by an exhaustive sweep of
# all 54 legal tile configurations, correctness-gated against the fp32
# reference at every point. Only the four batch anchors 1 / 16 / 64 / 256 were
# measured; the band edges between them are interpolation, chosen to place each
# anchor comfortably inside its own band rather than at a boundary.
#
# The bands are deliberately coarse. Adjacent configurations sit within a few
# percent of each other, which is the same order as run-to-run variation, so a
# finer table would be encoding noise. The gain this table is actually claiming
# is against the single universal default, which the sweep ranked around
# 30th of 54 at the larger batches.
_TUNED_TILES = (
    (4, (4, 16, 8), "b4"),
    (32, (2, 8, 2), "b32"),
    (128, (1, 8, 1), "b128"),
    (None, (8, 16, 1), "b_large"),
)

# Every tile the table can produce, for tuners and for the sweep space.
TUNED_SPEC_IDS = tuple(entry[2] for entry in _TUNED_TILES)


def tile_for_batch(batch: int) -> Tuple[int, int, int]:
    """Tuned ``(num_warps, warp_threads_k, blocks_per_v_dim)`` for ``batch``."""
    for max_batch, tile, _ in _TUNED_TILES:
        if max_batch is None or batch <= max_batch:
            return tile
    raise AssertionError("unreachable: table has an open-ended final band")


def spec_id_for_batch(batch: int) -> str:
    for max_batch, _, spec_id in _TUNED_TILES:
        if max_batch is None or batch <= max_batch:
            return spec_id
    raise AssertionError("unreachable: table has an open-ended final band")


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
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        # Under ``auto`` only the candidate the tuning table names may serve the
        # request, so selection is decided by measurement rather than by
        # registration order. An explicit ``spec_id`` pin bypasses this, which
        # is what makes a tuning sweep able to force a non-default tile.
        if req.spec_id.strip().lower() == "auto":
            wanted = spec_id_for_batch(int(req.batch))
            if wanted != spec_id:
                return False, (
                    f"tuned tile for batch {req.batch} is {wanted!r}, not {spec_id!r}"
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
        capability=Capability(arches=(ARCH,), dtypes=("bf16", "f16")),
        _supports=support,
        select_spec=select,
        signature=lambda spec: gdn_decode_signature(spec),
        grid=_grid,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=_build,
    )
    return candidate


def candidates() -> Tuple[KernelCandidate, ...]:
    """One candidate per tuned tile, in table order."""
    return tuple(
        _make_candidate(tile=tile, spec_id=spec_id, priority=10 + i)
        for i, (_, tile, spec_id) in enumerate(_TUNED_TILES)
    )


def register(registry) -> None:
    registry.extend(candidates())
