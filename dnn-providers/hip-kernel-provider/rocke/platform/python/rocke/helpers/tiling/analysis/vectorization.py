# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Vectorization model -- the b128-capped hardware-transaction pattern of a memory access. PURE calc (no
matplotlib): the shared backend that visualization, skills, and agents all ground on. GENERIC -- a function of
a distribution's ``{(lane,reg)->coord}`` map + an ``addr_fn`` (from the tensor's real strides), never a fixed
descriptor or an assumed row/col-major order."""

from __future__ import annotations

__all__ = ["vector_transactions", "addr_fn_from_strides"]


def addr_fn_from_strides(strides):
    """Return ``addr(*coord) -> ELEMENT address`` = ``sum(coord_i * strides_i)``. The strides come from the
    memory tensor descriptor, so which axis is stride-1 (the contiguous axis) is decided by the DATA, never
    assumed."""
    s = tuple(int(x) for x in strides)
    return lambda *coord: sum(int(coord[i]) * s[i] for i in range(len(s)))


def vector_transactions(mp, addr_fn, dtype_bits, order_by="reg", max_bits=128):
    """Cover each lane's cells with a TIMED sequence of hardware transactions, where the transaction WIDTH is
    derived from the STRIDES (not a recorded emit ``vw``). A transaction = the widest LEGAL, ALIGNED power-of-2
    access (``b8..b{max_bits}``, default b128 ceiling) over a stride-1 contiguous run: at each position take the
    largest width ``w`` (in elements) such that (a) ``w`` consecutive elements are present at unit element
    stride and (b) the run's base BYTE address is aligned to ``w*elem_bytes``. This is why a 32-contiguous f16
    run is 4x b128 (four transactions), NOT one; and a strided axis falls to b16/b8 singletons. ``addr_fn(r,
    c)`` returns the ELEMENT address of a coord; ``dtype_bits`` sets elem bytes and the legal-width ladder.
    ``order_by`` sets the time order within a lane: ``"reg"`` (fill / register order -- loads) or ``"addr"``
    (memory order -- stores). ``objdump`` VALIDATES this best guess; it is never the source. Returns
    ``({(lane,reg): tstep}, max_tsteps)``."""
    ebytes = max(1, dtype_bits // 8)                       # bytes per element (f16 -> 2, f32 -> 4)
    widths = [b // dtype_bits for b in                     # legal widths in ELEMENTS, widest first (>=1)
              (max_bits >> i for i in range(0, max_bits.bit_length())) if b >= dtype_bits]
    by_lane = {}
    for (lane, reg), (r, c) in mp.items():
        by_lane.setdefault(lane, []).append((addr_fn(r, c), reg))
    ts, maxt = {}, 1
    for lane, items in by_lane.items():
        items.sort()                                       # by element address
        runs, i, n = [], 0, len(items)
        while i < n:
            base_addr = items[i][0]
            chosen = 1                                     # b(dtype) singleton fallback
            for w in widths:                               # widest legal aligned vector first
                if w == 1:
                    break
                if (base_addr * ebytes) % (w * ebytes):    # base must be w*elem_bytes-aligned
                    continue
                if i + w <= n and all(items[i + j][0] == base_addr + j for j in range(w)):
                    chosen = w
                    break
            runs.append(items[i:i + chosen])
            i += chosen
        key = ((lambda run: min(a for a, _reg in run)) if order_by == "addr"
               else (lambda run: min(reg for _a, reg in run)))
        order = sorted(range(len(runs)), key=lambda idx: key(runs[idx]))
        rank = {idx: k for k, idx in enumerate(order)}
        for idx, run in enumerate(runs):
            for _a, reg in run:
                ts[(lane, reg)] = rank[idx]
        maxt = max(maxt, len(runs))
    return ts, maxt
