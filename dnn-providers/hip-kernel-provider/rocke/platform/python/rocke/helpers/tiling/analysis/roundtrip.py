# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Addressing round-trip (gate 1) -- PURE calc, no matplotlib.

Per LDS buffer-half, keyed by smem identity + the RESOLVED half: every read address a cooperative
store never wrote is a leak. Orthogonal to the MMA-soundness gate (``analysis/soundness.py``).
"""

from __future__ import annotations

from typing import Any

from .geometry import _arch_wave, _lane_span, resolve_origin


class RoundTripError(RuntimeError):
    """Raised when a read touches an LDS address the cooperative store never wrote (per half)."""


def verify_lds_roundtrip(pipeline: Any, space_id: int, *, tile_k: int) -> list[int]:
    """Addressing round-trip for one double-buffered LDS space, PER buffer-half.

    Store and read descriptors are both ``_transpose_desc``'d into the same ``(K, free)`` frame, so the
    element ADDRESS and the ``(K, free)`` label coincide -- the gate reduces to: every read address (over
    all waves, at half ``h``) was written by the cooperative store to half ``h``, and the store is a
    bijection with no leak across the half boundary. Origins are resolved via :func:`resolve_origin`;
    the half is keyed on the RESOLVED free-origin, never the raw symbolic ``oth`` expression (the store
    writes ``oth`` while the same-iteration read reads ``cur`` -- pairing those would be a silent bug).

    Rank-2 ``(K, free)``: axis 1 is the free axis, and its length spans every buffer half; invoke once
    per batch slice for N-D. Each read element is checked by its index and reported by its
    carried ``(K, free)`` positions.
    ``tile_k`` pins the K-loop iteration for the resolver (from the render context). Returns the list of
    verified halves; raises :class:`RoundTripError` naming the first offending read.
    """
    from ..lds_conflict import addr_map

    _arch, wave_size = _arch_wave(pipeline)  # DERIVED from the recording, not defaulted
    txns = [t for t in pipeline.transactions if t.space_id == space_id]
    stores = [t for t in txns if t.kind == "store"]
    reads = [t for t in txns if t.kind == "load"]
    if not stores or not reads:
        return []

    store_desc, read_desc = stores[0].tile_desc, reads[0].tile_desc
    # Each side is replayed with its own descriptor: (K, free), the free axis spanning every half.
    store_tensor, read_tensor = stores[0].tensor, reads[0].tensor
    store_lanes = _lane_span(
        store_desc.layout
    )  # cooperative: all waves' threads (e.g. 256)
    n_waves = store_lanes // wave_size
    read_dag = reads[0].origin

    # Halves = the distinct resolved store free-origins across the K-tiles (prologue writes half 0; the
    # in-loop store writes `oth` -> the other half). tile_m = the per-half free extent.
    store_frees = {
        resolve_origin(t.origin, {"k": kb * tile_k, "tid": 0})[1]
        for t in stores
        for kb in range(2)
    }
    bufs = len(store_frees)
    tile_m = store_tensor.lengths[1] // bufs

    name = pipeline.spaces[space_id]

    def elements(desc, tensor, origin, n_lanes, swizzle):
        """Every element the access touches, as ``(index, (K, free) positions)``."""
        acc, _ = addr_map(
            desc,
            tensor,
            origin=origin,
            n_lanes=n_lanes,
            lds_swizzle=swizzle,
        )
        for a in acc:
            for i in range(a["vw"]):
                yield a["base"] + i, a["element_positions"][i]

    verified: list[int] = []
    for h in range(bufs):
        lo, hi = h * tile_m, (h + 1) * tile_m
        # (a) confinement -- the store's DATA (pre-swizzle) stays within half h; a swizzle then permutes
        # WITHIN the buffer, so this leak check is a pre-swizzle property (post-swizzle freely crosses lo/hi).
        for _addr, (_k, free) in elements(
            store_desc, store_tensor, (0, h * tile_m), store_lanes, False
        ):
            if not (lo <= free < hi):
                raise RoundTripError(
                    f"{name}: store data leaks half {h} at free {free}"
                )
        # (b) coverage -- every read address (all waves, at the REAL swizzle) was written by the coop store.
        written: set[int] = set()
        for addr, _positions in elements(
            store_desc, store_tensor, (0, h * tile_m), store_lanes, stores[0].swizzle
        ):
            if addr in written:
                raise RoundTripError(f"{name}: store collision half {h} addr {addr}")
            written.add(addr)
        for w in range(n_waves):
            r_origin = resolve_origin(read_dag, {"k": h * tile_k, "tid": w * wave_size})
            if r_origin[1] // tile_m != h:
                raise RoundTripError(
                    f"{name}: read wave {w} resolves to half {r_origin[1] // tile_m}, expected {h}"
                )
            for addr, (k, free) in elements(
                read_desc, read_tensor, r_origin, wave_size, reads[0].swizzle
            ):
                if addr not in written:
                    raise RoundTripError(
                        f"{name} half {h} wave {w}: read -> addr {addr} "
                        f"(K={k}, free={free}) never written"
                    )
        verified.append(h)
    return verified
