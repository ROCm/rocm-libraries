# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Vectorization model -- the b128-capped hardware-transaction pattern of a memory access. PURE calc (no
matplotlib): the shared backend that visualization, skills, and agents all ground on. GENERIC -- a function of
a distribution's ``{(lane,reg)->coord}`` map + an ``addr_fn`` (from the tensor's real strides), never a fixed
descriptor or an assumed memory order."""

from __future__ import annotations

from collections.abc import Callable, Sequence

__all__ = [
    "MEMORY_SPACES",
    "addr_fn_from_strides",
    "check_memory_space",
    "vector_transactions",
    "widest_vector_units",
]

# The memory spaces a transaction model distinguishes: an LDS access must be aligned to its width, a global
# access need not be.
MEMORY_SPACES = ("global", "lds")

# One element's (address, register) inside a lane, and a lane's access: the elements it moves at once.
_Item = tuple[int, int]
_Access = list[_Item]


def check_memory_space(caller: str, space: str) -> None:
    """Refuse a `space` outside :data:`MEMORY_SPACES`, naming `caller`."""
    if space not in MEMORY_SPACES:
        raise ValueError(
            f"{caller} failed - space={space!r}, expected one of {MEMORY_SPACES}; pass the memory "
            "space the access targets"
        )


def _max_vector_units(unit_bits: int, max_bits: int) -> int:
    """The largest power-of-two count of `unit_bits` units at most `max_bits` wide (at least 1)."""
    units = 1
    while units * 2 * unit_bits <= max_bits:
        units *= 2
    return units


def widest_vector_units(
    run_units: int,
    unit_bits: int,
    *,
    max_bits: int = 128,
    other_strides: Sequence[int] = (),
) -> int:
    """The widest vector access over a contiguous run of `run_units` units of `unit_bits` each: the
    largest power-of-two count of at least 2 units, at most `max_bits` wide (default 128 bits, one
    ``dwordx4``), that divides the run and every one of `other_strides`; else 1. A longer run takes
    several such accesses. A global access needs no alignment, so this is the width a lane's global
    run reaches: the one rule both the emitter
    (:func:`rocke.helpers.tiling.emit.planned_global_access_units`) and the transaction model
    (:func:`vector_transactions` for ``space="global"``) use."""
    if not 0 < unit_bits <= max_bits:
        raise ValueError(
            f"widest_vector_units failed - unit_bits={unit_bits}, max_bits={max_bits}, expected "
            "0 < unit_bits <= max_bits"
        )
    units = _max_vector_units(unit_bits, max_bits)
    while units >= 2:
        if run_units % units == 0 and all(
            stride % units == 0 for stride in other_strides
        ):
            return units
        units //= 2
    return 1


def addr_fn_from_strides(strides):
    """Return ``addr(*coord) -> ELEMENT address`` = ``sum(coord_i * strides_i)``. The strides come from the
    memory tensor descriptor, so which axis is stride-1 (the contiguous axis) is decided by the DATA, never
    assumed."""
    s = tuple(int(x) for x in strides)
    return lambda *coord: sum(int(coord[i]) * s[i] for i in range(len(s)))


def vector_transactions(
    mp: dict[tuple[int, int], tuple[int, int]],
    addr_fn: Callable[[int, int], int],
    dtype_bits: int,
    order_by: str = "reg",
    max_bits: int = 128,
    *,
    space: str,
) -> tuple[dict[tuple[int, int], int], int]:
    """Cover each lane's cells with a TIMED sequence of hardware transactions, where the transaction WIDTH is
    derived from the STRIDES (not a recorded emit ``vw``), over each stride-1 contiguous run, capped at
    ``b{max_bits}`` (default b128). This is why a 32-contiguous f16 run is 4x b128 (four transactions), NOT one;
    and a strided axis falls to b16/b8 singletons. ``space`` (required, one of :data:`MEMORY_SPACES`) sets the
    alignment rule:

    - ``"lds"``: at each position take the widest LEGAL, ALIGNED power-of-2 access -- the largest width ``w``
      (in elements) such that (a) ``w`` consecutive elements are present at unit element stride and (b) the
      run's base BYTE address is aligned to ``w*elem_bytes``.
    - ``"global"``: a global access needs no alignment, so each lane's maximal address-contiguous run is covered
      by equal accesses of :func:`widest_vector_units`. A run that starts mid-vector (an odd row stride) is as
      wide as one that does not. When the lane's run is one register run of the emitter, this is the width the
      emitter plans for it.

    ``addr_fn(r, c)`` returns the ELEMENT address of a coord; ``dtype_bits`` sets elem bytes and the legal-width
    ladder. ``order_by`` sets the time order within a lane: ``"reg"`` (fill / register order -- loads) or
    ``"addr"`` (memory order -- stores). ``objdump`` VALIDATES this best guess; it is never the source. Returns
    ``({(lane,reg): tstep}, max_tsteps)``."""
    check_memory_space("vector_transactions", space)
    by_lane: dict[int, list[_Item]] = {}
    for (lane, reg), (r, c) in mp.items():
        by_lane.setdefault(lane, []).append((addr_fn(r, c), reg))
    ts, maxt = {}, 1
    for lane, items in by_lane.items():
        items.sort()  # by element address
        if space == "global":
            runs = _unaligned_accesses(items, dtype_bits, max_bits)
        else:
            runs = _aligned_accesses(items, dtype_bits, max_bits)
        key = (
            (lambda run: min(a for a, _reg in run))
            if order_by == "addr"
            else (lambda run: min(reg for _a, reg in run))
        )
        order = sorted(range(len(runs)), key=lambda idx: key(runs[idx]))
        rank = {idx: k for k, idx in enumerate(order)}
        for idx, run in enumerate(runs):
            for _a, reg in run:
                ts[(lane, reg)] = rank[idx]
        maxt = max(maxt, len(runs))
    return ts, maxt


def _aligned_accesses(
    items: list[_Item], dtype_bits: int, max_bits: int
) -> list[_Access]:
    """Split a lane's address-sorted ``(element address, reg)`` items greedily into the widest aligned
    accesses (see :func:`vector_transactions`, ``space="lds"``)."""
    ebytes = max(1, dtype_bits // 8)  # bytes per element (f16 -> 2, f32 -> 4)
    widths = [
        b // dtype_bits
        for b in (  # legal widths in ELEMENTS, widest first (>=1)
            max_bits >> i for i in range(max_bits.bit_length())
        )
        if b >= dtype_bits
    ]
    accesses, i, n = [], 0, len(items)
    while i < n:
        base_addr = items[i][0]
        chosen = 1  # b(dtype) singleton fallback
        for w in widths:  # widest legal aligned vector first
            if w == 1:
                break
            if (base_addr * ebytes) % (w * ebytes):  # base must be w*elem_bytes-aligned
                continue
            if i + w <= n and all(items[i + j][0] == base_addr + j for j in range(w)):
                chosen = w
                break
        accesses.append(items[i : i + chosen])
        i += chosen
    return accesses


def _unaligned_accesses(
    items: list[_Item], dtype_bits: int, max_bits: int
) -> list[_Access]:
    """Split a lane's address-sorted ``(element address, reg)`` items into its maximal contiguous runs,
    each covered by equal accesses of :func:`widest_vector_units` (see :func:`vector_transactions`,
    ``space="global"``)."""
    accesses, start = [], 0
    for end in range(1, len(items) + 1):
        if end < len(items) and items[end][0] == items[end - 1][0] + 1:
            continue
        width = widest_vector_units(end - start, dtype_bits, max_bits=max_bits)
        accesses.extend(items[at : at + width] for at in range(start, end, width))
        start = end
    return accesses
