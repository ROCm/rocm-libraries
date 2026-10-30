# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side planning of a native GETT contraction (no permute pass).

A hipTensor contraction ``D[modes_d] = alpha * sum_k A[modes_a] B[modes_b]
(+ beta * C[modes_d])`` with arbitrary mode orders is computed as ONE GEMM
whose operands are read IN PLACE:

* GEMM rows of operand X (= hipTensor A or B) are X's free modes in X's own
  memory order (outer -> inner), so 8 consecutive GEMM rows are 8 contiguous
  elements whenever X's fastest mode is a free mode (MN-contiguous operand,
  the hipTensor column-major default). Every row offset is a mixed-radix
  function of the row index (consecutive modes that are contiguous in X merge
  into one radix digit).
* GEMM "A" (= the rows that run along the GEMM's M) is the operand that owns
  D's fastest mode, so each MFMA lane's 4 consecutive accumulator rows are
  contiguous in D (vector stores of the permuted output).
* The K modes get one shared order (any order is legal, both operands are
  read through offsets) chosen so that a K tile of ``tile_k`` elements is
  SEPARABLE: ``koff(kt * T + kl) = ktile(kt) + klocal(kl)``. The tile part is
  one scalar per K tile (buffer soffset), the in-tile part a per-lane
  constant. This holds iff the inner K modes' extents multiply into T with
  the last one split evenly (``T = e1 * ... * e_{j-1} * f``, ``f | e_j``).
* D (and C) offsets use the same row digits with D's strides.

Everything here is plain integer bookkeeping; the device side lives in
:mod:`.gett`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def packed_strides(modes: Sequence[str], ext: Dict[str, int]) -> Dict[str, int]:
    """Row-major (last mode fastest) element strides."""
    st, s = {}, 1
    for m in reversed(list(modes)):
        st[m] = s
        s *= ext[m]
    return st


def merge_digits(
    order: Sequence[str], ext: Dict[str, int], strides: Dict[str, int]
) -> List[Tuple[int, int]]:
    """Radix digits (extent, stride), INNER first, of the index space ``order``
    (outer -> inner) under ``strides``; consecutive modes that are contiguous
    (stride[outer] == ext[inner] * stride[inner]) merge into one digit."""
    dig: List[List[int]] = []
    for m in reversed(list(order)):
        e, s = ext[m], strides[m]
        if e == 1:
            continue
        if dig and s == dig[-1][0] * dig[-1][1]:
            dig[-1][0] *= e
        else:
            dig.append([e, s])
    return [(e, s) for e, s in dig] or [(1, 0)]


def mixed_offset(idx: int, digits: Sequence[Tuple[int, int]]) -> int:
    off = 0
    for e, s in digits:
        off += (idx % e) * s
        idx //= e
    return off


def k_split(order_inner_first: Sequence[str], ext: Dict[str, int], T: int):
    """(local digits, tile digits) of the K space with tile T, as mode-level
    (radix, mode, mult) triples, or None when T is not separable."""
    rem = T
    local: List[Tuple[int, str, int]] = []
    tile: List[Tuple[int, str, int]] = []
    for m in order_inner_first:
        e = ext[m]
        if rem == 1:
            tile.append((e, m, 1))
        elif e <= rem:
            if rem % e:
                return None
            local.append((e, m, 1))
            rem //= e
        else:
            if e % rem:
                return None
            local.append((rem, m, 1))
            tile.append((e // rem, m, rem))
            rem = 1
    if rem != 1:
        return None
    return local, tile


@dataclass
class OpGeom:
    name: str  # hipTensor operand ('A' / 'B')
    rows_order: List[str]  # GEMM row modes, outer -> inner
    rows: int
    row_digits: List[Tuple[int, int]]  # (ext, stride) inner first (operand strides)
    vec: int  # elements per DTL lane (8 -> 16-byte, 2 -> 4-byte DTL)
    strides: Dict[str, int]
    nbytes: int


@dataclass
class GettPlan:
    modes_a: List[str]
    modes_b: List[str]
    modes_d: List[str]
    ext: Dict[str, int]
    swapped: bool  # GEMM A = hipTensor B
    M: int
    N: int
    K: int
    tile_k: int
    k_order: List[str]  # inner first
    k_local: List[Tuple[int, str, int]]
    k_tile: List[Tuple[int, str, int]]
    ops: Dict[str, OpGeom]  # by GEMM role 'A' / 'B'
    d_row_digits: List[Tuple[int, int]]  # GEMM A rows -> D offsets
    d_col_digits: List[Tuple[int, int]]  # GEMM B rows -> D offsets
    vec_d: int
    strides_d: Dict[str, int] = field(default_factory=dict)
    vec_d8: bool = (
        False  # 8 consecutive GEMM rows are 8 contiguous, 8-aligned D elements (16-byte stores)
    )
    #: (mode, real extent, padded extent) of the one K mode padded to make the
    #: K tile separable (an in-tile digit; padded rows read zeros), or None.
    #: ``K`` and the K digits use the padded extent.
    k_pad: Optional[Tuple[str, int, int]] = None
    #: K-contiguous packed operands (``kc``): the GEMM reads A as [M, K] and B
    #: as [N, K] row-major copies (rows in ``ops[*].rows_order``, K in
    #: ``k_order`` reversed = outer first); no K split, no operand runs.
    kc: bool = False
    #: 8 consecutive GEMM columns are 8 contiguous, 8-aligned D elements (the
    #: row-interleave vector epilogue: 8 columns per lane)
    vec_dc8: bool = False

    def describe(self) -> str:
        kp = "" if self.k_pad is None else f" kpad={self.k_pad}"
        return kp + (
            f"M{self.M} N{self.N} K{self.K} T{self.tile_k} swap={int(self.swapped)} "
            f"vecA={self.ops['A'].vec} vecB={self.ops['B'].vec} vecD={self.vec_d} "
            f"kord={self.k_order} rowsA={self.ops['A'].rows_order} rowsB={self.ops['B'].rows_order} "
            f"digA={self.ops['A'].row_digits} digB={self.ops['B'].row_digits} "
            f"dR={self.d_row_digits} dC={self.d_col_digits}"
        )

    def klocal(self, kl: int, role: str) -> int:
        st = self.ops[role].strides
        off = 0
        for r, m, mult in self.k_local:
            off += (kl % r) * mult * st[m]
            kl //= r
        return off

    def ktile(self, kt: int, role: str) -> int:
        st = self.ops[role].strides
        off = 0
        for r, m, mult in self.k_tile:
            off += (kt % r) * mult * st[m]
            kt //= r
        return off


def k_split_padded(order_inner_first: Sequence[str], ext: Dict[str, int], T: int):
    """``k_split`` with ONE K mode padded up to the smallest extent that makes
    the tile separable, the padded mode being entirely in-tile (its digit is
    local, so the padded rows are a per-lane mask). Returns ``(local, tile,
    (mode, real, padded))`` with the least padded K, or None."""
    best = None
    for j, m in enumerate(order_inner_first):
        rem = T
        for x in order_inner_first[:j]:
            e = ext[x]
            if e > rem or rem % e:
                rem = 0
                break
            rem //= e
        if not rem or ext[m] >= rem:
            continue
        d = next(v for v in range(ext[m] + 1, rem + 1) if rem % v == 0)
        e2 = dict(ext)
        e2[m] = d
        sp = k_split(order_inner_first, e2, T)
        if sp is None or not any(mm == m and r == d for r, mm, _ in sp[0]):
            continue
        cost = 1
        for x in order_inner_first:
            cost *= e2[x]
        if best is None or cost < best[0]:
            best = (cost, sp, (m, ext[m], d))
    if best is None:
        return None
    return best[1][0], best[1][1], best[2]


def _vec(
    digits: Sequence[Tuple[int, int]], other_strides: Sequence[int], cap: int = 8
) -> int:
    e0, s0 = digits[0]
    if s0 != 1:
        return 1
    v = cap
    while v > 1:
        if (
            e0 % v == 0
            and all(s % v == 0 for (_e, s) in digits[1:])
            and all(s % v == 0 for s in other_strides)
        ):
            return v
        v //= 2
    return 1


def plan_gett(
    modes_a: Sequence[str],
    modes_b: Sequence[str],
    modes_d: Sequence[str],
    ext: Dict[str, int],
    *,
    strides_a: Optional[Dict[str, int]] = None,
    strides_b: Optional[Dict[str, int]] = None,
    strides_d: Optional[Dict[str, int]] = None,
    tile_k: Optional[int] = None,
    tile_mn: int = 256,
    kc: bool = False,
    d_fast_role: str = "A",
) -> GettPlan:
    """Mode lists are ROW-major (last mode fastest), i.e. a hipTensor
    column-major descriptor's mode list reversed.

    ``kc``: plan for K-contiguous packed operand copies (no K split, no MN run
    requirement). ``d_fast_role``: the GEMM operand (``"A"``: rows, ``"B"``:
    columns) that owns D's fastest mode."""
    if d_fast_role not in ("A", "B"):
        raise ValueError(f"d_fast_role {d_fast_role!r} not in ('A', 'B')")
    modes_a, modes_b, modes_d = list(modes_a), list(modes_b), list(modes_d)
    sa = strides_a or packed_strides(modes_a, ext)
    sb = strides_b or packed_strides(modes_b, ext)
    sd = strides_d or packed_strides(modes_d, ext)
    m_modes = [x for x in modes_d if x in modes_a and x not in modes_b]
    n_modes = [x for x in modes_d if x in modes_b and x not in modes_a]
    k_modes = [x for x in modes_a if x in modes_b and x not in modes_d]
    # every mode of D is an M or N mode, and every mode of A (of B) an M (N) or K mode: a batch /
    # Hadamard mode (in A, B and D) or a mode of one operand only (summed over by no kernel) is
    # refused, not dropped
    if (
        set(m_modes) | set(n_modes) != set(modes_d)
        or set(m_modes) | set(k_modes) != set(modes_a)
        or set(n_modes) | set(k_modes) != set(modes_b)
        or not k_modes
        or not m_modes
        or not n_modes
    ):
        raise ValueError(
            "not a plain bilinear contraction (batch / Hadamard modes and modes of one "
            "operand only unsupported)"
        )
    d_fast = min(modes_d, key=lambda x: sd[x])
    swapped = (d_fast in n_modes) == (d_fast_role == "A")
    roles = (
        {"A": ("B", modes_b, sb, n_modes), "B": ("A", modes_a, sa, m_modes)}
        if swapped
        else {"A": ("A", modes_a, sa, m_modes), "B": ("B", modes_b, sb, n_modes)}
    )
    # K order (inner first): try every order, prefer separable with the
    # largest tile, then small A strides innermost
    tks = [tile_k] if tile_k else [64, 32]
    best = None
    if kc:
        # packed K-contiguous copies: any shared K order (A's memory order)
        order = sorted(k_modes, key=lambda x: sa[x])
        best = (tks[0], order, ([], []))
    for T in tks if best is None else ():
        cands = []
        for perm in itertools.permutations(k_modes):
            sp = k_split(perm, ext, T)
            if sp is None:
                continue
            key = tuple(min(sa[m], sb[m]) for m in perm)
            cands.append((key, list(perm), sp))
        if cands:
            cands.sort(key=lambda c: c[0])
            best = (T,) + cands[0][1:]
            break
    k_pad = None
    if best is None and not kc:
        # no exact split: pad one in-tile K mode (padded rows read zeros)
        for T in tks:
            cands = []
            for perm in itertools.permutations(k_modes):
                sp = k_split_padded(perm, ext, T)
                if sp is None:
                    continue
                kv = 1
                for m in k_modes:
                    kv *= ext[m] if m != sp[2][0] else sp[2][2]
                key = (kv,) + tuple(min(sa[m], sb[m]) for m in perm)
                cands.append((key, list(perm), sp))
            if cands:
                cands.sort(key=lambda c: c[0])
                key, perm, sp = cands[0]
                best = (T, perm, (sp[0], sp[1]))
                k_pad = sp[2]
                break
    if best is None:
        raise ValueError(
            f"K modes {[(m, ext[m]) for m in k_modes]} not separable for tiles {tks}"
        )
    T, k_order, (k_local, k_tile) = best
    ops = {}
    for role, (nm, modes, st, free) in roles.items():
        rows_order = sorted(free, key=lambda x: -st[x])  # memory order, outer -> inner
        dig = merge_digits(rows_order, ext, st)
        rows = 1
        for x in free:
            rows *= ext[x]
        v = _vec(dig, [st[m] for m in k_modes])
        if kc:
            v = 0  # read from the packed K-contiguous copy
        elif v < 2:
            raise ValueError(
                f"operand {nm} is not MN-contiguous (vec {v}); GETT needs the K-outer path"
            )
        if not kc:
            v = 8 if v >= 8 else 2  # DTL widths: 16-byte or 4-byte per lane
        nbytes = 2 * (1 + sum((ext[m] - 1) * st[m] for m in modes))
        ops[role] = OpGeom(nm, rows_order, rows, dig, v, st, nbytes)
    d_row = merge_digits(ops["A"].rows_order, ext, sd)
    d_col = merge_digits(ops["B"].rows_order, ext, sd)
    vd = 4
    while vd > 1:
        if (
            d_row[0][1] == 1
            and d_row[0][0] % vd == 0
            and all(s % vd == 0 for (_e, s) in d_row[1:])
            and all(s % vd == 0 for (_e, s) in d_col)
        ):
            break
        vd //= 2
    vd8 = (
        d_row[0][1] == 1
        and d_row[0][0] % 8 == 0
        and all(s % 8 == 0 for (_e, s) in d_row[1:])
        and all(s % 8 == 0 for (_e, s) in d_col)
    )
    vdc8 = (
        d_col[0][1] == 1
        and d_col[0][0] % 8 == 0
        and all(s % 8 == 0 for (_e, s) in d_col[1:])
        and all(s % 8 == 0 for (_e, s) in d_row)
    )
    K = 1
    for m in k_modes:
        K *= ext[m] if k_pad is None or m != k_pad[0] else k_pad[2]
    return GettPlan(
        modes_a,
        modes_b,
        modes_d,
        dict(ext),
        swapped,
        ops["A"].rows,
        ops["B"].rows,
        K,
        T,
        k_order,
        k_local,
        k_tile,
        ops,
        d_row,
        d_col,
        vd,
        sd,
        vd8,
        k_pad,
        kc,
        vdc8,
    )
