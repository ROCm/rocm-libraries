# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side planning of the persistent hipTensor contraction monokernel.

A hipTensor contraction ``D[modes_d] = alpha * sum_k A[modes_a] B[modes_b]
(+ beta * C[modes_d])`` with arbitrary mode orders is mapped onto ONE
canonical GEMM ``D'[m, n] = A'[m, k] B'[n, k]^T`` (RCR, K contiguous):

* M modes = modes of D and A (not B), N modes = modes of D and B, K modes =
  modes of A and B. If D's innermost mode is an M mode the operands swap
  roles (the GEMM computes D^T), so the GEMM's N index always runs along
  D's contiguous mode.
* Canonical orders: N follows D's order (D's innermost mode last, so 8
  consecutive GEMM columns are contiguous in D whenever that mode's extent
  allows); M follows A's order; K follows A's order when A is already
  canonical, else B's when B is, else A's.
* An operand whose modes already read ``rows + K`` in the canonical order
  (packed, row-major) is read in place. Any other operand is PACKED by
  producer tasks of the same launch into a canonical ``[rows, K]``
  workspace through two offset tables (``rowoff[r] + koff[k]``: every
  packed tensor offset is separable in the row and K indices).
* The output and C use the same separable form: ``off(m, n) = drow[m] +
  dcol[n]``.

Everything here is plain integer bookkeeping; the device
side lives in :mod:`.persistent_contraction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def packed_strides(modes: Sequence[str], ext: Dict[str, int]) -> Dict[str, int]:
    """Row-major (last mode fastest) element strides."""
    st, s = {}, 1
    for m in reversed(list(modes)):
        st[m] = s
        s *= ext[m]
    return st


def offsets(
    order: Sequence[str], ext: Dict[str, int], strides: Dict[str, int]
) -> List[int]:
    """Element offset of every index of the mixed-radix space ``order``
    (last mode fastest) under ``strides``."""
    out = [0]
    for m in order:  # outer -> inner: out = out x ext[m]
        e, s = ext[m], strides[m]
        out = [o + i * s for o in out for i in range(e)]
    return out


def contiguous_run(
    order: Sequence[str], ext: Dict[str, int], strides: Dict[str, int]
) -> int:
    """Length of the maximal run of consecutive indices of ``order`` (last
    mode fastest) that are consecutive elements: the innermost modes merge
    while ``stride[outer] == ext[inner] * stride[inner]``; 1 if the innermost
    mode is not unit-stride."""
    if not order or strides[order[-1]] != 1:
        return 1
    run = ext[order[-1]]
    for i in range(len(order) - 2, -1, -1):
        if strides[order[i]] == run:
            run *= ext[order[i]]
        else:
            break
    return run


def _run_modes(order, ext, strides):
    run = contiguous_run(order, ext, strides)
    inner, out = 1, set()
    if run == 1:
        return out
    for m in reversed(list(order)):
        if inner >= run:
            break
        inner *= ext[m]
        out.add(m)
    return out


def _vec_width(
    order: Sequence[str], ext: Dict[str, int], strides: Dict[str, int], cap: int = 8
) -> int:
    """Largest v in {8, 4, 2, 1} (<= cap) such that every aligned group of v
    consecutive indices of ``order`` is v contiguous, v-aligned elements
    (v divides the merged contiguous run and every stride outside it)."""
    run = contiguous_run(order, ext, strides)
    if run == 1:
        return 1
    # modes outside the merged run: their strides must keep v-alignment
    inner, n_in = 1, 0
    for m in reversed(list(order)):
        if inner >= run:
            break
        inner *= ext[m]
        n_in += 1
    outer = list(order)[: len(order) - n_in]
    v = cap
    while v > 1:
        if run % v == 0 and all(strides[m] % v == 0 for m in outer):
            return v
        v //= 2
    return 1


@dataclass
class OperandPlan:
    name: str  # hipTensor operand ('A' or 'B')
    gemm_role: str  # 'A' (rows = M) or 'B' (rows = N)
    modes: List[str]
    rows_order: List[str]
    rows: int
    canonical: bool  # read in place
    vec: int  # source vector width of a pack along K (elements)
    vec_rows: int = 1  # source vector width along the rows (transposing pack)
    mode: str = (
        "k"  # "k": gather along K (no LDS); "t": LDS-transposing pack (row-contiguous source)
    )
    rowoff: List[int] = field(default_factory=list)
    koff: List[int] = field(default_factory=list)


@dataclass
class ContractionPlan:
    modes_a: List[str]
    modes_b: List[str]
    modes_d: List[str]
    ext: Dict[str, int]
    swapped: bool
    m_order: List[str]
    n_order: List[str]
    k_order: List[str]
    M: int
    N: int
    K: int
    ops: Dict[str, OperandPlan]  # by GEMM role 'A' / 'B'
    drow: List[int]
    dcol: List[int]
    vec_d: int
    # row-reduction output (max over every N mode): R[m] in canonical M order
    red_modes: List[str] = field(default_factory=list)

    @property
    def packs(self) -> List[OperandPlan]:
        return [self.ops[r] for r in ("A", "B") if not self.ops[r].canonical]

    def describe(self) -> str:
        pk = (
            ",".join(
                f"{p.name}->{p.gemm_role}({p.mode}{p.vec_rows if p.mode == 't' else p.vec})"
                for p in self.packs
            )
            or "none"
        )
        return (
            f"M{self.M} N{self.N} K{self.K} swap={int(self.swapped)} "
            f"m={self.m_order} n={self.n_order} k={self.k_order} packs={pk} vecD={self.vec_d}"
        )


def plan_contraction(
    modes_a: Sequence[str],
    modes_b: Sequence[str],
    modes_d: Sequence[str],
    ext: Dict[str, int],
    *,
    strides_a: Optional[Dict[str, int]] = None,
    strides_b: Optional[Dict[str, int]] = None,
    strides_d: Optional[Dict[str, int]] = None,
    with_tables: bool = True,
) -> ContractionPlan:
    modes_a, modes_b, modes_d = list(modes_a), list(modes_b), list(modes_d)
    sa = strides_a or packed_strides(modes_a, ext)
    sb = strides_b or packed_strides(modes_b, ext)
    sd = strides_d or packed_strides(modes_d, ext)
    m_modes = [x for x in modes_d if x in modes_a and x not in modes_b]
    n_modes = [x for x in modes_d if x in modes_b and x not in modes_a]
    k_modes = [x for x in modes_a if x in modes_b and x not in modes_d]
    if set(m_modes) | set(n_modes) != set(modes_d) or len(k_modes) == 0:
        raise ValueError(
            "not a plain bilinear contraction (batch / Hadamard modes unsupported here)"
        )
    # the GEMM's N must run along D's contiguous mode
    swapped = modes_d[-1] in m_modes and modes_d[-1] not in n_modes
    if swapped:
        ga_modes, gb_modes, sga, sgb = modes_b, modes_a, sb, sa
        rm, rn = n_modes, m_modes
        names = ("B", "A")
    else:
        ga_modes, gb_modes, sga, sgb = modes_a, modes_b, sa, sb
        rm, rn = m_modes, n_modes
        names = ("A", "B")
    n_order = [x for x in modes_d if x in rn]
    m_order = [x for x in ga_modes if x in rm]
    ka = [x for x in ga_modes if x in k_modes]
    kb = [x for x in gb_modes if x in k_modes]

    def _canon(modes, rows_order, korder, strides):
        return modes == list(rows_order) + list(korder) and strides == packed_strides(
            modes, ext
        )

    if _canon(ga_modes, m_order, ka, sga):
        k_order = ka
    elif _canon(gb_modes, n_order, kb, sgb):
        k_order = kb
    else:
        k_order = ka
    M = _prod(ext[x] for x in m_order)
    N = _prod(ext[x] for x in n_order)
    K = _prod(ext[x] for x in k_order)
    ops = {}
    for role, nm, modes, strides, rows_order in (
        ("A", names[0], ga_modes, sga, m_order),
        ("B", names[1], gb_modes, sgb, n_order),
    ):
        canon = _canon(modes, rows_order, k_order, strides)
        vec = _vec_width(k_order, ext, strides)
        # rows also have to keep v-alignment
        while vec > 1 and any(
            strides[x] % vec
            for x in rows_order
            if x not in _run_modes(rows_order, ext, strides)
        ):
            vec //= 2
        vr = _vec_width(rows_order, ext, strides)
        while vr > 1 and any(
            strides[x] % vr
            for x in k_order
            if x not in _run_modes(k_order, ext, strides)
        ):
            vr //= 2
        op = OperandPlan(
            name=nm,
            gemm_role=role,
            modes=modes,
            rows_order=list(rows_order),
            rows=_prod(ext[x] for x in rows_order),
            canonical=canon,
            vec=vec,
            vec_rows=vr,
            mode="t" if vr > vec else "k",
        )
        if with_tables and not canon:
            op.rowoff = offsets(rows_order, ext, strides)
            op.koff = offsets(k_order, ext, strides)
        ops[role] = op
    drow = offsets(m_order, ext, sd) if with_tables else []
    dcol = offsets(n_order, ext, sd) if with_tables else []
    vec_d = _vec_width(n_order, ext, sd)
    while vec_d > 1 and any(
        sd[x] % vec_d for x in m_order if x not in _run_modes(m_order, ext, sd)
    ):
        vec_d //= 2
    return ContractionPlan(
        modes_a=modes_a,
        modes_b=modes_b,
        modes_d=modes_d,
        ext=dict(ext),
        swapped=swapped,
        m_order=m_order,
        n_order=n_order,
        k_order=k_order,
        M=M,
        N=N,
        K=K,
        ops=ops,
        drow=drow,
        dcol=dcol,
        vec_d=vec_d,
        red_modes=list(m_order),
    )


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p
