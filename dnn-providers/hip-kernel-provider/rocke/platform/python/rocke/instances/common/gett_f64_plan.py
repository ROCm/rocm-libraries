# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host planning: a tensor contraction with arbitrary mode orders -> GETT launches.

``D[modes_d] = alpha * sum A[modes_a] * B[modes_b] + beta * C[modes_d]``
(hipTensor / cuTENSOR contraction; C has D's modes and extents and its own
strides). Every mode is classified:

* **M** modes: in A and D only; **N** modes: in B and D only;
* **K** modes: in A and B only (summed); **L** (batch) modes: in all three.

Each index space gets ONE logical mode order, shared by every tensor that
carries it, and every tensor gets its own digits for it (consecutive modes
that are contiguous in that tensor merge into one digit). No data moves: the
kernel reads A / B / C and writes D through these digits.

Choices (all overridable):

* **Roles.** The f64 MFMA accumulator holds 16 consecutive GEMM columns per
  row group, so D stores coalesce along N. The operand owning D's fastest
  free mode becomes GEMM-B (N side); if that mode belongs to hipTensor A, the
  roles swap (``swapped``).
* **Orders.** M modes follow GEMM-A's stride order, K modes follow the
  operand that is contiguous along K (else GEMM-A), N modes follow D's
  stride order, L modes follow D. Fastest first.
* **Launch splitting.** The kernel moves 64-bit base pointers by the batch
  and K-tile offsets, and addresses the rest (rows and one K tile of A / B;
  rows and columns of C / D) with 32-bit byte offsets below 2^31. A
  contraction whose addressed spans exceed that is cut along M / N / L modes
  into several launches (base offsets per launch); K is never cut. The batch
  runs on grid y, so batch modes are also cut until each launch has at most
  :data:`~.gett_f64.MAX_GRID_Y` batch entries (chunks of one size plus a
  remainder chunk).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .gett_f64 import MAX_GRID_Y, Digits, GettF64Problem, _MAX_SPAN

Modes = Sequence[str]


def merge_digits(
    order: Sequence[str], ext: Dict[str, int], strides: Dict[str, int]
) -> Digits:
    """Digits (fastest first) of the index space ``order`` (fastest first)."""
    dig: List[List[int]] = []
    for m in order:
        e, s = ext[m], strides[m]
        if e == 1:
            continue
        if dig and s == dig[-1][0] * dig[-1][1]:
            dig[-1][0] *= e
        else:
            dig.append([e, s])
    return tuple((e, s) for e, s in dig) or ((1, 0),)


def packed_strides(modes: Modes, ext: Dict[str, int]) -> Dict[str, int]:
    """hipTensor's default (NULL strides): generalized column-major, first mode fastest."""
    st, s = {}, 1
    for m in modes:
        st[m] = s
        s *= ext[m]
    return st


@dataclass(frozen=True)
class GettLaunch:
    problem: GettF64Problem
    a_offset: int  # element offsets of the launch's base pointers
    b_offset: int
    c_offset: int
    d_offset: int


@dataclass
class GettPlan:
    swapped: bool  # GEMM-A is hipTensor B
    m_order: List[str]
    n_order: List[str]
    k_order: List[str]
    l_order: List[str]
    ext: Dict[str, int]
    launches: List[GettLaunch] = field(default_factory=list)

    @property
    def M(self) -> int:
        return math.prod(self.ext[m] for m in self.m_order)

    @property
    def N(self) -> int:
        return math.prod(self.ext[m] for m in self.n_order)

    @property
    def K(self) -> int:
        return math.prod(self.ext[m] for m in self.k_order)

    @property
    def L(self) -> int:
        return math.prod(self.ext[m] for m in self.l_order)


def _by_stride(modes: Sequence[str], strides: Dict[str, int], ext: Dict[str, int]):
    return sorted(modes, key=lambda m: (strides[m], -ext[m]))


def plan_contraction(
    modes_a: Modes,
    strides_a: Dict[str, int],
    modes_b: Modes,
    strides_b: Dict[str, int],
    modes_d: Modes,
    strides_d: Dict[str, int],
    ext: Dict[str, int],
    *,
    strides_c: Optional[Dict[str, int]] = None,
    complex_: bool = False,
    conj_a: bool = False,
    conj_b: bool = False,
    n_order_from: str = "D",
    max_span: int = _MAX_SPAN,
    max_batch: int = MAX_GRID_Y,
) -> GettPlan:
    """Plan a contraction. Strides are in elements (complex elements for cf64)."""
    sa, sb, sd = set(modes_a), set(modes_b), set(modes_d)
    for name, ms in (("A", modes_a), ("B", modes_b), ("D", modes_d)):
        if len(set(ms)) != len(ms):
            raise ValueError(f"{name} repeats a mode: {list(ms)}")
    m_modes = [m for m in modes_a if m in sd and m not in sb]
    n_modes = [m for m in modes_b if m in sd and m not in sa]
    k_modes = [m for m in modes_a if m in sb and m not in sd]
    l_modes = [m for m in modes_d if m in sa and m in sb]
    lonely = (sa | sb | sd) - set(m_modes) - set(n_modes) - set(k_modes) - set(l_modes)
    if lonely:
        raise ValueError(
            f"modes in only one tensor are not a contraction: {sorted(lonely)}"
        )
    strides_c = strides_c or strides_d

    # Roles: GEMM-B owns D's fastest free mode.
    free = [m for m in modes_d if ext[m] > 1 and m not in l_modes]
    swapped = False
    if free:
        fastest = min(free, key=lambda m: strides_d[m])
        swapped = fastest in m_modes
    if swapped:
        modes_a, modes_b = modes_b, modes_a
        strides_a, strides_b = strides_b, strides_a
        m_modes, n_modes = n_modes, m_modes
        conj_a, conj_b = conj_b, conj_a

    m_order = _by_stride(m_modes, strides_a, ext)
    n_src = strides_d if n_order_from == "D" else strides_b
    n_order = _by_stride(n_modes, n_src, ext)
    l_order = _by_stride(l_modes, strides_d, ext)
    k_inner_a = min(k_modes, key=lambda m: strides_a[m]) if k_modes else None
    k_inner_b = min(k_modes, key=lambda m: strides_b[m]) if k_modes else None
    if k_modes and strides_b[k_inner_b] == 1 and strides_a[k_inner_a] != 1:
        k_order = _by_stride(k_modes, strides_b, ext)
    else:
        k_order = _by_stride(k_modes, strides_a, ext)
    k_order = _separable_k_order(k_order, ext, strides_a, strides_b)

    plan = GettPlan(swapped, m_order, n_order, k_order, l_order, dict(ext))
    plan.launches = _split(
        plan,
        strides_a,
        strides_b,
        strides_c,
        strides_d,
        complex_,
        conj_a,
        conj_b,
        max_span,
        max_batch,
    )
    return plan


def _separable_k_order(order, ext, sa, sb, tile_ks=(8, 16)):
    """``order`` if both operands' K digits split at every K tile, else the
    first permutation (keeping the fastest mode first when possible) that does.

    Any K order is legal -- both operands are read through offsets -- and a
    split K tile keeps the K offset in the 64-bit base instead of each lane.
    """
    from itertools import permutations

    from .gett_f64 import k_split

    def ok(o):
        # Both operands split at every K tile, with a small in-tile span.
        for st in (sa, sb):
            for t in tile_ks:
                split = k_split(merge_digits(o, ext, st), t)
                if split is None or sum((e - 1) * s for e, s in split[0]) >= (1 << 24):
                    return False
        return True

    if len(order) <= 1 or ok(order):
        return list(order)
    rest = [list(p) for p in permutations(order) if list(p) != list(order)]
    rest.sort(key=lambda o: (o[0] != order[0], o))
    for cand in rest[:720]:
        if ok(cand):
            return cand
    return list(order)


def _problem(plan, ext, sa, sb, sc, sd, cx, ca, cb) -> GettF64Problem:
    mo, no, ko, lo = plan.m_order, plan.n_order, plan.k_order, plan.l_order
    prod = lambda ms: math.prod(ext[m] for m in ms)  # noqa: E731
    return GettF64Problem(
        M=prod(mo),
        N=prod(no),
        K=prod(ko),
        L=prod(lo),
        a_m=merge_digits(mo, ext, sa),
        a_k=merge_digits(ko, ext, sa),
        a_l=merge_digits(lo, ext, sa),
        b_n=merge_digits(no, ext, sb),
        b_k=merge_digits(ko, ext, sb),
        b_l=merge_digits(lo, ext, sb),
        d_m=merge_digits(mo, ext, sd),
        d_n=merge_digits(no, ext, sd),
        d_l=merge_digits(lo, ext, sd),
        c_m=merge_digits(mo, ext, sc),
        c_n=merge_digits(no, ext, sc),
        c_l=merge_digits(lo, ext, sc),
        complex_=cx,
        conj_a=ca,
        conj_b=cb,
    )


def _split(
    plan, sa, sb, sc, sd, cx, ca, cb, max_span, max_batch=MAX_GRID_Y
) -> List[GettLaunch]:
    """Cut M / N / L modes until every tensor of one launch spans < 2^31 bytes
    and the batch of one launch fits the grid.

    Greedy: while a tensor is too large, shrink the cuttable mode with the
    largest span contribution ``(extent - 1) * stride`` among the offending
    tensors to its next smaller divisor. That cuts the outermost strides first
    and keeps the launch count low. K is never cut. Then, while the batch
    extent exceeds ``max_batch``, the largest batch mode is cut into chunks of
    the largest size that fits, plus a remainder chunk.
    """

    def problem(e):
        return _problem(plan, e, sa, sb, sc, sd, cx, ca, cb)

    # Addressed spans of the K tiles the per-arch specs may pick.
    from .gett_f64 import TILE_K_CANDIDATES

    tile_ks = TILE_K_CANDIDATES
    tensors = (
        ("A", sa, plan.m_order + plan.k_order),
        ("B", sb, plan.n_order + plan.k_order),
        ("C", sc, plan.m_order + plan.n_order),
        ("D", sd, plan.m_order + plan.n_order),
    )
    cuttable = set(plan.m_order) | set(plan.n_order) | set(plan.l_order)
    ext = dict(plan.ext)
    work = dict(ext)
    while True:
        p = problem(work)
        # A launch fits when some K tile keeps every addressed span in budget.
        best_t = min(
            tile_ks, key=lambda t: max(p.addr_span_bytes(w, t) for w in "ABCD")
        )
        over = [
            (w, st, ms)
            for w, st, ms in tensors
            if p.addr_span_bytes(w, best_t) > max_span
        ]
        if not over:
            break
        best = None
        for _, st, ms in over:
            for m in ms:
                if m in cuttable and work[m] > 1:
                    contrib = (work[m] - 1) * st.get(m, 0)
                    if best is None or contrib > best[0]:
                        best = (contrib, m)
        if best is None or best[0] == 0:
            names = ", ".join(w for w, _, _ in over)
            raise ValueError(
                f"{names} address more than 2^31 bytes even with every M / N mode "
                "cut to 1 (the K digits do not split at the K tile and span too much)"
            )
        m = best[1]
        work[m] = max(d for d in range(1, work[m]) if ext[m] % d == 0)
    while math.prod(work[m] for m in plan.l_order) > max_batch:
        m = max((x for x in plan.l_order if work[x] > 1), key=lambda x: work[x])
        others = math.prod(work[x] for x in plan.l_order if x != m)
        work[m] = max(1, max_batch // others)
    cut = {m: work[m] for m in ext if work[m] != ext[m]}
    modes = list(cut)
    # (start, size) chunks per cut mode; M / N cuts are divisors (equal
    # chunks), batch cuts may end in a smaller remainder chunk.
    chunks = [
        [(s, min(cut[m], ext[m] - s)) for s in range(0, ext[m], cut[m])] for m in modes
    ]
    launches: List[GettLaunch] = []
    for combo in itertools.product(*reversed(chunks)):
        combo = combo[::-1]  # the first cut mode varies fastest
        offs = [0, 0, 0, 0]
        e = dict(work)
        for m, (start, size) in zip(modes, combo):
            e[m] = size
            for t, st in enumerate((sa, sb, sc, sd)):
                offs[t] += start * st.get(m, 0)
        p = _problem(plan, e, sa, sb, sc, sd, cx, ca, cb)
        launches.append(GettLaunch(p, *offs))
    return launches


__all__ = [
    "GettLaunch",
    "GettPlan",
    "merge_digits",
    "packed_strides",
    "plan_contraction",
]
