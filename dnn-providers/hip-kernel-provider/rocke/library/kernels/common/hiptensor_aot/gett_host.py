# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side plans of the contraction kernels that read and write the tensors in place.

hipTensor's rocKE backend computes these in C++ (``rocke_plan.cpp``); this module is the Python
statement of the same rules, used by the numeric tests to launch the kernels:

* :func:`k_plan` -- how a kernel walks the K modes with its K tile: in-tile digits (a per-lane,
  loop-invariant offset), tile digits (one scalar offset per K tile), and the padding of the K
  space when its extents do not split at the K tile;
* :func:`dgett_plan` -- arguments and grid of a :mod:`.dgett` launch.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

from .common import as_i32_bits, magic_u31

I32_MAX = (1 << 31) - 1

#: A contraction mode: ``(extent, {tensor: stride})`` with the tensors ``"A", "B", "C", "D"`` that
#: have the mode (strides in elements; complex elements for complex data).
Mode = Tuple[int, Dict[str, int]]

#: K modes up to which every order of them is tried for a split at the K tile.
K_ORDER_SEARCH = 6


def _magic(vals: Dict[str, int], m: str, s: str, d: int) -> None:
    mm, ss = magic_u31(d)
    vals[m] = as_i32_bits(mm)
    vals[s] = ss


def merge_modes(modes: Sequence[Mode], tensors: str) -> List[Mode]:
    """Consecutive modes (innermost first) contiguous in every tensor of ``tensors`` merge into
    one digit (while the merged extent stays below 2^31)."""
    out: List[Mode] = []
    for e, st in modes:
        if out:
            le, ls = out[-1]
            if all(st[t] == ls[t] * le for t in tensors) and le * e <= I32_MAX:
                out[-1] = (le * e, ls)
                continue
        out.append((e, dict(st)))
    return out


def k_split(exts: Sequence[int], tk: int):
    """In-tile and tile digits of K modes with extents ``exts`` (innermost first) at the K tile
    ``tk`` as ``(radix, mode index, multiplier)`` triples, or None when the K space does not split
    at ``tk``: ``tk`` must be the product of the innermost extents, the last of them possibly
    split evenly (``tk = e_1 ... e_{j-1} f`` with ``f`` dividing ``e_j``)."""
    rem = tk
    local, tile = [], []
    for i, e in enumerate(exts):
        if rem == 1:
            tile.append((e, i, 1))
        elif e <= rem:
            if rem % e:
                return None
            local.append((e, i, 1))
            rem //= e
        else:
            if e % rem:
                return None
            local.append((rem, i, 1))
            tile.append((e // rem, i, rem))
            rem = 1
    if rem != 1:
        return None
    return local, tile


def k_split_padded(exts: Sequence[int], tk: int):
    """:func:`k_split` with one mode padded to the smallest extent that splits, the padded mode
    entirely in the tile (an in-tile digit; its padded indices read zeros). Returns ``(local,
    tile, (mode index, real extent, padded extent))`` with the smallest padded K, or None.
    """
    best = None
    for j in range(len(exts)):
        rem = tk
        for x in exts[:j]:
            if x > rem or rem % x:
                rem = 0
                break
            rem //= x
        if not rem or exts[j] >= rem:
            continue
        d = next(v for v in range(exts[j] + 1, rem + 1) if rem % v == 0)
        padded = [d if i == j else e for i, e in enumerate(exts)]
        sp = k_split(padded, tk)
        if sp is None or not any(i == j and r == d for r, i, _ in sp[0]):
            continue
        cost = math.prod(padded)
        if best is None or cost < best[0]:
            best = (cost, sp, (j, exts[j], d))
    if best is None:
        return None
    return best[1][0], best[1][1], best[2]


def k_plan(kmodes: Sequence[Mode], tk: int, slots_local: int, slots_tile: int) -> Dict:
    """How a kernel walks the K modes ``kmodes`` (strides of A and B) with K tile ``tk``.

    Returns ``dict(local=[(radix, sA, sB)], tile=[(extent, sA, sB)], pad=(slot, limit) or
    None, krl=limit or None, nk=K tiles)``. The K offset of in-tile index ``kl`` of tile ``kt``
    is the mixed-radix offset of ``kl`` over ``local`` plus that of ``kt`` over ``tile``.
    ``pad``: in-tile digit ``slot`` must stay below ``limit``; ``krl``: ``kl + tk * (first tile
    digit)`` must stay below it (the innermost mode padded to a multiple of ``tk``).

    Orders tried: every order of at most :data:`K_ORDER_SEARCH` modes (else A's and B's memory
    orders). An exact split first (smallest strides innermost), then one padded in-tile mode
    (least padded K, then smallest strides), else the innermost mode of A's memory order in
    tiles of ``tk``, the last one cut at its extent."""
    ks = [m for m in kmodes if m[0] != 1]
    if not ks:
        ks = [(1, {"A": 0, "B": 0})]
    by_a = merge_modes(sorted(ks, key=lambda m: abs(m[1]["A"])), "AB")
    if len(by_a) <= K_ORDER_SEARCH:
        orders = [list(p) for p in itertools.permutations(by_a)]
    else:
        orders = [by_a, merge_modes(sorted(ks, key=lambda m: abs(m[1]["B"])), "AB")]

    def key(order):
        return tuple(min(abs(st["A"]), abs(st["B"])) for _e, st in order)

    def digits(order, triples):
        return [
            (r, mult * order[i][1]["A"], mult * order[i][1]["B"])
            for r, i, mult in triples
        ]

    found = None
    for order in orders:
        sp = k_split([e for e, _ in order], tk)
        if sp is None or len(sp[0]) > slots_local or len(sp[1]) > slots_tile:
            continue
        k = key(order)
        if found is None or k < found[0]:
            found = (k, order, sp, None)
    if found is None:
        for order in orders:
            sp = k_split_padded([e for e, _ in order], tk)
            if sp is None or len(sp[0]) > slots_local or len(sp[1]) > slots_tile:
                continue
            j, _real, padded = sp[2]
            vol = math.prod(padded if i == j else e for i, (e, _) in enumerate(order))
            k = (vol,) + key(order)
            if found is None or k < found[0]:
                found = (k, order, (sp[0], sp[1]), sp[2])
    if found is not None:
        _k, order, (local, tile), pad = found
        pad_slot = None
        if pad is not None:
            j, real, _padded = pad
            pad_slot = (next(s for s, (_r, i, _m) in enumerate(local) if i == j), real)
        ti = digits(order, tile)
        return dict(
            local=digits(order, local),
            tile=ti,
            pad=pad_slot,
            krl=None,
            nk=math.prod(e for e, _a, _b in ti),
        )
    e0, st0 = by_a[0]
    lo = [(tk, st0["A"], st0["B"])]
    ti = [(-(-e0 // tk), tk * st0["A"], tk * st0["B"])]
    ti += [(e, st["A"], st["B"]) for e, st in by_a[1:]]
    if len(ti) > slots_tile:
        raise ValueError(f"{len(ti)} K tile digits, {slots_tile} slots")
    return dict(
        local=lo, tile=ti, pad=None, krl=e0, nk=math.prod(e for e, _a, _b in ti)
    )


def fill_digits(
    vals: Dict[str, int], prefix: str, digits: Sequence[Mode], tensors: str, slots: int
) -> None:
    """Kernel arguments of a digit space (``<prefix>e<i>``, magic pairs ``<prefix>m<i>,
    <prefix>s<i>`` of every digit but the last, strides ``<prefix><tensor><i>``): ``digits``
    innermost first, the slots past them extent 1 and stride 0."""
    if len(digits) > slots:
        raise ValueError(f"{prefix}: {len(digits)} digits, {slots} slots")
    for i in range(slots):
        e, st = digits[i] if i < len(digits) else (1, {})
        vals[f"{prefix}e{i}"] = e
        if i < slots - 1:
            _magic(vals, f"{prefix}m{i}", f"{prefix}s{i}", e)
        for t in tensors:
            vals[f"{prefix}{t}{i}"] = st.get(t, 0) if e != 1 else 0


def span_elems(modes: Sequence[Mode], t: str) -> int:
    """Elements a tensor's offsets reach: ``1 + sum (extent - 1) * |stride|``."""
    return 1 + sum((e - 1) * abs(st[t]) for e, st in modes if t in st)


def dgett_plan(
    groups: Dict[str, Sequence[Mode]],
    es: Dict[str, int],
    tk: int,
    tm: int,
    tn: int,
) -> Dict:
    """Arguments (sizes, digits, magic numbers, spans) and launch of a contraction on a
    :mod:`.dgett` kernel with tile ``tm x tn x tk``.

    ``groups``: ``"M"`` (modes of A and D), ``"N"`` (B and D), ``"K"`` (A and B), ``"L"`` (A, B
    and D); C has D's modes (give C the strides of D when the problem has no C). ``es``:
    element bytes of each tensor. Returns ``dict(vals, grid, swapped, a_fast, b_fast, idx64)``:
    ``swapped`` means the kernel's A is the problem's B (M and N trade places, so that D's
    stride-1 mode is an N mode, along which the lanes of the epilogue run); ``a_fast`` /
    ``b_fast`` the load direction of the kernel's operands (``"k"`` when the operand's stride-1
    mode is a K mode, ``"r"`` when it is a row mode)."""
    from .dgett import SLOTS_KL, SLOTS_KT, SLOTS_L, SLOTS_MN

    def clean(ms):
        return [(e, dict(st)) for e, st in ms if e != 1]

    M, N = clean(groups.get("M", [])), clean(groups.get("N", []))
    K, L = clean(groups.get("K", [])), clean(groups.get("L", []))
    es = dict(es)
    swapped = any(st["D"] == 1 for _e, st in M)
    if swapped:
        sw = {"A": "B", "B": "A", "C": "C", "D": "D"}

        def flip(ms):
            return [(e, {sw[t]: s for t, s in st.items()}) for e, st in ms]

        M, N, K, L = flip(N), flip(M), flip(K), flip(L)
        es = {sw[t]: v for t, v in es.items()}

    def fast(rows, t):
        if any(st[t] == 1 for _e, st in K):
            return "k"
        if any(st[t] == 1 for _e, st in rows):
            return "r"
        return "k"

    a_fast, b_fast = fast(M, "A"), fast(N, "B")
    M = merge_modes(
        sorted(M, key=lambda m: abs(m[1]["A" if a_fast == "r" else "D"])), "ACD"
    )
    N = merge_modes(
        sorted(N, key=lambda m: abs(m[1]["B" if b_fast == "r" else "D"])), "BCD"
    )
    L = merge_modes(sorted(L, key=lambda m: abs(m[1]["D"])), "ABCD")
    kp = k_plan(K, tk, SLOTS_KL, SLOTS_KT)
    m_ext = math.prod(e for e, _ in M)
    n_ext = math.prod(e for e, _ in N)
    l_ext = math.prod(e for e, _ in L)
    vals: Dict[str, int] = {}
    fill_digits(vals, "m", M, "ACD", SLOTS_MN)
    fill_digits(vals, "n", N, "BCD", SLOTS_MN)
    fill_digits(vals, "l", L, "ABCD", SLOTS_L)
    fill_digits(
        vals, "kl", [(r, {"A": a, "B": b}) for r, a, b in kp["local"]], "AB", SLOTS_KL
    )
    fill_digits(
        vals, "kt", [(e, {"A": a, "B": b}) for e, a, b in kp["tile"]], "AB", SLOTS_KT
    )
    vals["kps"], vals["kpl"] = kp["pad"] if kp["pad"] else (-1, 0)
    vals["krl"] = kp["krl"] if kp["krl"] is not None else 1 << 30
    tiles_m, tiles_n = -(-m_ext // tm), -(-n_ext // tn)
    vals.update(M=m_ext, N=n_ext, nk=kp["nk"], tiles_n=tiles_n)
    _magic(vals, "tn_m", "tn_s", tiles_n)
    allm = list(M) + list(N) + list(K) + list(L)
    spans = {t: span_elems(allm, t) * es[t] for t in "ABCD"}
    idx64 = any(v > I32_MAX for v in spans.values())
    for t in "ABCD":
        vals[f"{t.lower()}_bytes"] = min(spans[t], I32_MAX)
    for v in (m_ext, n_ext, l_ext, kp["nk"] * tk, tiles_m * tiles_n):
        if v > I32_MAX:
            raise ValueError("an index space past 2^31")
    return dict(
        vals=vals,
        grid=(tiles_m * tiles_n, 1, l_ext),
        swapped=swapped,
        a_fast=a_fast,
        b_fast=b_fast,
        idx64=idx64,
        kplan=kp,
    )


def contraction_groups(
    modes: Dict[str, Sequence[int]],
    extents: Dict[int, int],
    strides: Dict[str, Sequence[int]],
) -> Dict[str, List[Mode]]:
    """Groups of a contraction given hipTensor-style mode lists and strides per tensor
    (``"A", "B", "D"`` and optionally ``"C"``, which has D's modes)."""
    out: Dict[str, List[Mode]] = {"M": [], "N": [], "K": [], "L": []}
    st = {t: dict(zip(modes[t], strides[t])) for t in modes}
    if "C" not in st:
        st["C"] = dict(st["D"])
    for m, e in extents.items():
        ina, inb, ind = m in st["A"], m in st["B"], m in st["D"]
        s = {t: st[t][m] for t in "ABCD" if m in st[t]}
        if ina and inb and ind:
            out["L"].append((e, s))
        elif ina and ind:
            out["M"].append((e, s))
        elif inb and ind:
            out["N"].append((e, s))
        elif ina and inb:
            out["K"].append((e, s))
        else:
            raise ValueError(f"mode {m} is in only one tensor")
    return out


# ------------------------------------------------------------------------------------------------
# The ring GETT classes (rgett)
# ------------------------------------------------------------------------------------------------

#: Largest operand byte span the DTL buffer offsets of the ring body address.
RGETT_MAX_OPERAND_BYTES = 0x7FFF0000
#: Largest D (and C) byte span of the ring body's epilogue (32-bit offsets with the out-of-range
#: row / column sentinel).
RGETT_MAX_D_BYTES = 0x7FFFFFF0


def _vec(digits: Sequence[Mode], t: str, others: Sequence[int], cap: int) -> int:
    """Widest power of two ``<= cap`` of consecutive elements that are contiguous and aligned
    in ``t``: the innermost digit has stride 1 and an extent multiple of it, every other stride
    is a multiple of it."""
    e0, st0 = digits[0]
    if st0[t] != 1:
        return 1
    v = cap
    while v > 1:
        if (
            e0 % v == 0
            and all(st[t] % v == 0 for _e, st in digits[1:])
            and all(s % v == 0 for s in others)
        ):
            return v
        v //= 2
    return 1


def rgett_plan(
    groups: Dict[str, Sequence[Mode]], es: int, bilinear: bool, tk: int = 32
) -> Dict:
    """The ring GETT class and geometry of a contraction, the rules of
    ``rocke.instances.common.tensor.gett_plan.plan_gett`` with runtime digits. ``groups`` as
    for :func:`dgett_plan` (16-bit data: ``es`` 2). Raises ValueError when the problem is not
    one the ring body serves: batch modes, an operand whose fastest free run is not even, K
    extents that do not split at ``tk`` (one padded in-tile mode allowed), a negative stride,
    C strides other than D's, too many digits, spans past the buffer ranges.

    Returns ``dict(vals, swapped, vec_a, vec_b, vec_d, vec_d8, grid, M, N, K)``: ``swapped``
    means the kernel's A operand is the problem's B (the kernel's A owns D's fastest mode).
    """
    from .rgett import SLOTS_KL, SLOTS_KT, SLOTS_ROW

    if any(e != 1 for e, _ in groups.get("L", [])):
        raise ValueError("batch modes")
    M = [(e, dict(st)) for e, st in groups.get("M", []) if e != 1]
    N = [(e, dict(st)) for e, st in groups.get("N", []) if e != 1]
    K = [(e, dict(st)) for e, st in groups.get("K", []) if e != 1]
    if not M or not N or not K:
        raise ValueError("an empty M, N or K space")
    for _e, st in M + N + K:
        if any(v < 0 for v in st.values()):
            raise ValueError("a negative stride")
    if bilinear and any(st["C"] != st["D"] for _e, st in M + N):
        raise ValueError("C strides other than D's")
    d_fast = min(M + N, key=lambda m: m[1]["D"])
    swapped = any(m is d_fast for m in N)
    ga, gb = ("B", "A") if swapped else ("A", "B")
    rows = {"A": N if swapped else M, "B": M if swapped else N}
    ops = {}
    for role, t in (("A", ga), ("B", gb)):
        order = sorted(rows[role], key=lambda m: m[1][t])  # innermost first
        dig = merge_modes(order, t)
        v = _vec(dig, t, [st[t] for _e, st in K], 8)
        if v < 2:
            raise ValueError(f"operand {t}: fastest free run not a multiple of 2")
        ops[role] = dict(order=order, digits=dig, vec=8 if v >= 8 else 2, t=t)
    kmodes = [(e, {"A": st[ga], "B": st[gb]}) for e, st in K]
    kp = k_plan(kmodes, tk, SLOTS_KL, SLOTS_KT)
    if kp["krl"] is not None:
        raise ValueError("K extents that do not split at the K tile")
    d_row = merge_modes(ops["A"]["order"], "D")
    d_col = merge_modes(ops["B"]["order"], "D")
    vd = 1
    if d_row[0][1]["D"] == 1:
        for v in (4, 2):
            if (
                d_row[0][0] % v == 0
                and all(st["D"] % v == 0 for _e, st in d_row[1:])
                and all(st["D"] % v == 0 for _e, st in d_col)
            ):
                vd = v
                break
    vd8 = (
        d_row[0][1]["D"] == 1
        and d_row[0][0] % 8 == 0
        and all(st["D"] % 8 == 0 for _e, st in d_row[1:])
        and all(st["D"] % 8 == 0 for _e, st in d_col)
    )
    vals: Dict[str, int] = {}
    fill_digits(
        vals, "ra", [(e, {"S": st[ga]}) for e, st in ops["A"]["digits"]], "S", SLOTS_ROW
    )
    fill_digits(
        vals, "rb", [(e, {"S": st[gb]}) for e, st in ops["B"]["digits"]], "S", SLOTS_ROW
    )
    fill_digits(vals, "dr", [(e, {"S": st["D"]}) for e, st in d_row], "S", SLOTS_ROW)
    fill_digits(vals, "dc", [(e, {"S": st["D"]}) for e, st in d_col], "S", SLOTS_ROW)
    fill_digits(
        vals, "kl", [(r, {"A": a, "B": b}) for r, a, b in kp["local"]], "AB", SLOTS_KL
    )
    fill_digits(
        vals, "kt", [(e, {"A": a, "B": b}) for e, a, b in kp["tile"]], "AB", SLOTS_KT
    )
    vals["kps"], vals["kpl"] = kp["pad"] if kp["pad"] else (-1, 0)
    m_ext = math.prod(e for e, _ in rows["A"])
    n_ext = math.prod(e for e, _ in rows["B"])
    k_pad = kp["nk"] * tk
    allm = M + N + K
    for t in "AB":
        if span_elems(allm, t) * es > RGETT_MAX_OPERAND_BYTES:
            raise ValueError(f"operand {t} past the DMA buffer range")
    for t in "CD" if bilinear else "D":
        if span_elems(allm, t) * es > RGETT_MAX_D_BYTES:
            raise ValueError(f"{t} past 2^31 bytes")
    for v in (m_ext, n_ext, k_pad):
        if v > I32_MAX:
            raise ValueError("an index space past 2^31")
    if -(-m_ext // 256) > 65535:
        raise ValueError("more than 65535 rows of tiles")
    vals.update(M=m_ext, N=n_ext, K=k_pad)
    digits = max(
        len(ops["A"]["digits"]),
        len(ops["B"]["digits"]),
        len(d_row),
        len(d_col),
        len(kp["local"]),
        len(kp["tile"]),
    )
    return dict(
        vals=vals,
        digits=digits,
        swapped=swapped,
        vec_a=ops["A"]["vec"],
        vec_b=ops["B"]["vec"],
        vec_d=vd,
        vec_d8=vd8,
        grid=(-(-n_ext // 256), -(-m_ext // 256), 1),
        kplan=kp,
    )


__all__ = [
    "K_ORDER_SEARCH",
    "RGETT_MAX_D_BYTES",
    "RGETT_MAX_OPERAND_BYTES",
    "Mode",
    "contraction_groups",
    "dgett_plan",
    "fill_digits",
    "k_plan",
    "k_split",
    "k_split_padded",
    "merge_modes",
    "rgett_plan",
    "span_elems",
]
