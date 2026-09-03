# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout optimizer -- ``evaluate_transform`` / ``optimize_layout`` as a min-cost transform search.

The transform between two register STATES is never a formula to look up; it is the delta between two
constructed layouts, discovered by rocKE's own solver (:func:`classify_transform`) and priced by a cost
ladder. :func:`evaluate_transform` answers "is there a valid path from a source layout to a target, and what
does it cost". :func:`optimize_layout` is the minimization: given a source and candidate target distributions
(supply your own, or sweep them with :func:`enumerate_stripings`), return the CHEAPEST valid one -- trying a
free symmetry / an intra-lane reorder / a different striping / an LDS reposition before ever falling to
cross-lane (the last resort, whose cost grows with tile size).

Cost ladder (cheapest -> last resort):
  0    relabel / symmetry        register-identity axis rename (A<->B M<->N, col<->row) -- zero movement
  1    reorder, dword-aligned    whole-VGPR register MOVs
  ~pk  reorder, sub-dword        unpack/move/repack, cost ~ dtype pack factor (f16 2x, f8 4x)
  rt   reposition via LDS        the cheaper alternative to a register CROSS-LANE when the data already transits
                                 LDS -- a re-ownership via LDS re-addressing. NOT free, and NOT below a register
                                 reorder: it is a full round-trip = store + read at the throughput FLOOR + a
                                 BARRIER (paid even conflict-free) + the NEW access's bank conflicts (TWO
                                 patterns, store AND read, each EMPIRICAL under its own port rule via
                                 `/bank-conflict`) + any LDS capacity/occupancy cost. Routing through LDS can
                                 INTRODUCE conflicts absent from the register path -> measure the new pattern per
                                 case. The floor **scales with the #registers moved** (LDS is BANDWIDTH-bound).
                                 Pass ``lds_conflict_cost=(store_bc, read_bc)``; without it the edge is a flagged
                                 LOWER BOUND (barrier + bandwidth only).
  hi   cross_lane                DPP / ds_bpermute -- LAST RESORT, cost scales with **(ops-per-register) x
                                 #registers**: a 1-op/reg permute is cheap, flip&zip / grouped interleaves do
                                 MORE ops/reg -> dearer. Worse as the tile grows.

**Equal shots at the empirical frontier.** ``cross_lane`` and ``reposition_lds`` are flagged ``empirical`` --
their heuristic costs are estimates, NOT trustworthy enough to eliminate one another. So when the cheapest
valid path comes down to LDS-vs-cross-lane, :func:`recommend` returns BOTH as co-contenders to settle by
TESTING (`/bank-conflict` + a wall-time sweep) -- neither falls out on these numbers alone. Deterministic
edges (relabel / reorder / identity) that beat both are trustworthy winners and DO decide outright.

Validity is rocKE's own gates: :func:`operand_soundness` (per-operand sound MMA operand), optionally the
pairwise :func:`diagnose_k_match` (``A.K-dist == B.K-dist`` -- pass ``k_partner``), and the tier from
:func:`classify_transform`. No reference tables, no hand formulas -- read off the actual layouts, so it is
correct for any context (atom / wave / #atoms / tile shape / dtype).
"""

from __future__ import annotations

from dataclasses import dataclass

from .layouts.tile_distribution import make_tile_desc
from .transforms import (as_forward_map, classify_transform, describe_edge, diagnose_k_match,
                         name_permutation, operand_soundness)

__all__ = ["Edge", "Assessment", "evaluate_transform", "optimize_layout", "enumerate_stripings", "recommend"]

_DWORD_BITS = 32
# BOTH heavy movers scale with the NUMBER OF REGISTERS moved (the data volume) -- the tile only gets more
# expensive as it grows. cross-lane moves each register individually (DPP/ds_bpermute), so it scales HARD; an
# LDS reposition moves them in bulk but is fundamentally **bandwidth-bound** (store + read of every register).
# These slopes are ranking heuristics, not measured cycles.
_LDS_BARRIER = 1.0             # fixed sync for a reposition round-trip
_LDS_BW_PER_REG = 0.25        # LDS round-trip traffic per register -- LDS BANDWIDTH is the binding resource
_CROSS_LANE_BASE = 8.0        # per-op setup for a cross-lane instruction (DPP / ds_bpermute)
_CROSS_LANE_CYC_PER_OP_REG = 0.5   # ~cost per (cross-lane op x register): a 1-op/reg permute is cheap; flip&zip
                                   # / grouped interleaves do MORE ops per register -> proportionally dearer


@dataclass(frozen=True)
class Edge:
    """One priced transform edge. ``kind`` in identity/relabel/reposition_lds/reorder/cross_lane/invalid.
    ``empirical`` is True for the heavy movers (``cross_lane``, ``reposition_lds``) whose cost is a HEURISTIC
    estimate -- their relative ordering is NOT trustworthy and must be settled by measurement, so when they are
    the last contenders neither may fall out on cost alone (see :func:`recommend`)."""

    kind: str
    cost: float
    detail: str
    empirical: bool = False


@dataclass(frozen=True)
class Assessment:
    """The result of assessing one source->target transform: ``works`` (a valid path exists), the per-operand
    ``sound`` severity, the pairwise ``k_match`` severity (``"n/a"`` when no ``k_partner`` given), the priced
    ``edge`` (cheapest way to reach the target, used for ranking), a plain ``reason``, and ``contenders`` --
    every priced way to reach the target. When a re-ownership can go either register-cross-lane OR LDS
    reposition, BOTH are here (both ``empirical``); :func:`recommend` refuses to pick between them on cost."""

    works: bool
    sound: str
    k_match: str
    edge: Edge
    reason: str
    contenders: tuple = ()


def _reorder_grade(perm: tuple[int, ...], per_dword: int) -> tuple[float, str]:
    """A register reorder is **dword-aligned** (cheap, ~1) when each packed-dword group of elements moves as a
    unit; otherwise it is **sub-dword** (unpack/move/repack, cost ~ the pack factor)."""
    if per_dword <= 1:
        return 1.0, "dword-aligned"
    for i in range(len(perm)):
        base = i - (i % per_dword)
        if perm[i] - perm[base] != (i - base):          # the dword group did not move as a unit
            return float(per_dword), "sub-dword"
    return 1.0, "dword-aligned"


def _num_lanes(fwd: dict) -> int:
    return len({l for l, _ in fwd})


def _price(plan, *, dtype_bits: int, through_lds: bool, tile_regs: int, lds_conflict_cost,
           cross_lane_ops_per_reg: float = 1.0) -> Edge:
    """Price the classified delta against the cost ladder. ``cross_lane_ops_per_reg`` scales the cross-lane cost
    by how many ops the mechanism runs per register (1 for a simple permute; more for flip&zip / grouped)."""
    if plan.tier == "reorder":
        if plan.permutation == tuple(range(len(plan.permutation))):
            return Edge("identity", 0.0, "no movement")
        cost, grade = _reorder_grade(plan.permutation, max(1, _DWORD_BITS // dtype_bits))
        return Edge("reorder", cost, f"{grade} register reorder = {name_permutation(plan.permutation)}")
    # cross-lane in registers is the last resort -- BUT if the data already transits LDS, the same re-ownership
    # can be a reposition (re-addressing). Its cost is a full round-trip: the throughput FLOOR + a BARRIER (paid
    # even conflict-free, scaling with the registers moved -> BANDWIDTH) PLUS the NEW access's bank conflicts,
    # which are TWO patterns (store + read, each under its own port rule) and EMPIRICAL (/bank-conflict). Routing
    # through LDS can INTRODUCE conflicts the register path never had. Without measured costs this is only a
    # LOWER BOUND (barrier + bandwidth), flagged so no one reads it as free.
    if through_lds:
        floor = _LDS_BARRIER + _LDS_BW_PER_REG * tile_regs      # bandwidth: scales with registers moved
        if lds_conflict_cost is None:
            return Edge("reposition_lds", floor,
                        f"LDS reposition LOWER BOUND (barrier + bandwidth for {tile_regs} regs = {floor:g}) -- "
                        "the NEW store AND read bank-conflict patterns are NOT evaluated (two patterns, two port "
                        "rules) and LDS capacity/occupancy is not counted; route via /bank-conflict, never free",
                        empirical=True)
        store_bc, read_bc = (lds_conflict_cost if isinstance(lds_conflict_cost, tuple)
                             else (float(lds_conflict_cost), 0.0))
        return Edge("reposition_lds", floor + float(store_bc) + float(read_bc),
                    f"LDS reposition = barrier+bandwidth ({floor:g}, {tile_regs} regs) + measured store BC "
                    f"{float(store_bc):g} + read BC {float(read_bc):g}", empirical=True)
    cost = _CROSS_LANE_BASE + _CROSS_LANE_CYC_PER_OP_REG * cross_lane_ops_per_reg * tile_regs
    return Edge("cross_lane", cost,
                f"cross-lane (DPP/ds_bpermute) {cross_lane_ops_per_reg:g} op/reg over {tile_regs} regs -- "
                f"{plan.reason}", empirical=True)


def evaluate_transform(source, target, *, canon=None, k_partner=None, dtype_bits: int = 16,
                       through_lds: bool = False, lds_conflict_cost=None,
                       cross_lane_ops_per_reg: float = 1.0) -> Assessment:
    """Is there a VALID path from ``source`` to ``target``, and what is the cheapest edge that reaches it?

    ``source``/``target`` are ``WarpDistributionEncoding`` or forward maps. Validity gates (all optional, all
    rocKE's own solvers):
    - ``canon`` -- the atom's canonical operand ref -> **per-operand** soundness (:func:`operand_soundness`).
    - ``k_partner`` -- the OTHER operand's target layout -> the **pairwise** ``A.K-dist == B.K-dist`` half of
      the sound MAC (:func:`diagnose_k_match`). Without it, that half is the caller's responsibility.
    Cost knobs: ``through_lds`` allows an LDS reposition (pass ``lds_conflict_cost=(store_bc, read_bc)`` for its
    true cost; unset = flagged lower bound). With NO ``canon`` and NO ``k_partner`` the result is
    transform-cost-ONLY and does not imply a valid MMA.
    """
    src, tgt = as_forward_map(source), as_forward_map(target)
    sound = operand_soundness(tgt, canon).severity if canon is not None else "n/a"
    k_match = diagnose_k_match(tgt, k_partner).severity if k_partner is not None else "n/a"
    works = (canon is None or sound == "ok") and (k_partner is None or k_match == "ok")
    valid_note = f"sound={sound}, K-match={k_match}"

    def _assess(edge: Edge, how: str, contenders=None) -> Assessment:
        return Assessment(works, sound, k_match, edge, f"{valid_note}; {how}",
                          tuple(contenders) if contenders else (edge,))

    # TOP OF THE LADDER: a free relabel / symmetry -- a pure axis-permutation at register identity (transpose /
    # col<->row / A<->B M<->N rename), zero movement. Detect it BEFORE pricing a delta, else classify_transform
    # reads a same-slot axis-swap as a move and mis-prices the biggest lever (the crossed symmetry bridge, §8).
    if src != tgt:
        try:
            _kind, why = describe_edge(src, tgt, relabel=True)
            return _assess(Edge("relabel", 0.0, why), "free relabel/symmetry")
        except ValueError:
            pass
    try:
        plan = classify_transform(src, tgt)
    except ValueError as exc:
        return Assessment(False, sound, k_match, Edge("invalid", float("inf"), str(exc)),
                          "source and target hold different elements -- no transform exists")
    tile_regs = len(tgt) // max(1, _num_lanes(tgt))
    reg = _price(plan, dtype_bits=dtype_bits, through_lds=False, tile_regs=tile_regs, lds_conflict_cost=None,
                 cross_lane_ops_per_reg=cross_lane_ops_per_reg)
    gate = "" if (canon is not None or k_partner is not None) else " [no validity gate -- cost only]"
    if reg.kind != "cross_lane":                          # deterministic (identity / reorder) -- decides on cost
        return _assess(reg, f"reach via {reg.kind} (cost {reg.cost:g}){gate}")
    # A cross-lane re-ownership. The register cross-lane op is one path; if the data transits LDS, an LDS
    # reposition is the OTHER -- both empirical, so carry BOTH as contenders (equal shots; see recommend()).
    contenders = [reg]
    if through_lds:
        contenders.append(_price(plan, dtype_bits=dtype_bits, through_lds=True, tile_regs=tile_regs,
                                 lds_conflict_cost=lds_conflict_cost,
                                 cross_lane_ops_per_reg=cross_lane_ops_per_reg))
    best = min(contenders, key=lambda e: e.cost)
    how = (f"heavy mover ({' vs '.join(e.kind for e in contenders)}) -- EQUAL SHOTS, settle by testing"
           if len(contenders) > 1 else f"reach via cross_lane (cost {best.cost:g}){gate}")
    return _assess(best, how, contenders)


def optimize_layout(source, candidates: dict, *, canon=None, k_partner=None, dtype_bits: int = 16,
                    through_lds: bool = False, lds_conflict_cost=None,
                    cross_lane_ops_per_reg: float = 1.0) -> list[tuple[str, Assessment]]:
    """MINIMIZE: assess every candidate target distribution with :func:`evaluate_transform` and return them
    ordered cheapest-VALID first. The head is the recommended distribution -- but read it via :func:`recommend`,
    which refuses to crown a winner among the ``empirical`` heavy movers (LDS reposition vs cross-lane) on
    heuristic cost alone.

    ``candidates`` maps a name -> a target layout (a striping / ownership / register order). Build the ones
    worth trying yourself, or sweep them with :func:`enumerate_stripings`.
    """
    scored = [(name, evaluate_transform(source, tgt, canon=canon, k_partner=k_partner, dtype_bits=dtype_bits,
                                        through_lds=through_lds, lds_conflict_cost=lds_conflict_cost,
                                        cross_lane_ops_per_reg=cross_lane_ops_per_reg))
              for name, tgt in candidates.items()]
    scored.sort(key=lambda na: (not na[1].works, na[1].edge.cost))
    return scored


def recommend(ranked: list[tuple[str, Assessment]]) -> tuple[str, list[tuple[str, Assessment]], tuple]:
    """Read the outcome of :func:`optimize_layout` HONESTLY. Returns ``(status, picks, contenders)``:

    - ``("none", [], ())``               -- no valid candidate.
    - ``("decided", [(name, a)], ())``   -- the cheapest valid path is a DETERMINISTIC edge (relabel / reorder
                                            / identity); its cost is trustworthy, so it wins outright.
    - ``("measure", [(name, a)], edges)``-- the cheapest valid path is a heavy mover. ``edges`` are its
                                            ``empirical`` contenders (register cross-lane and/or LDS
                                            reposition) -- NOT separable on heuristic cost, so each gets an
                                            EQUAL SHOT: pick by testing (`/bank-conflict` + a wall-time sweep),
                                            never on these estimates alone.
    """
    valid = [(n, a) for n, a in ranked if a.works]
    if not valid:
        return ("none", [], ())
    name, top = valid[0]
    if not top.edge.empirical:
        return ("decided", [(name, top)], ())
    return ("measure", [(name, top)], tuple(e for e in top.contenders if e.empirical))


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_stripings(shape, wave_size: int) -> dict:
    """Sweep the candidate lane STRIPINGS of a 2-D ``shape`` across ``wave_size`` lanes: every way to split the
    two axes over the lanes (``lanes0·lanes1 == wave_size``, each dividing its axis) x the lane axis-order.
    Returns ``{name -> WarpDistributionEncoding}`` -- the raw material :func:`optimize_layout` ranks. This is
    the "try a different distribution" lever: rocKE constructs each candidate, the optimizer prices it."""
    d0, d1 = int(shape[0]), int(shape[1])
    out: dict = {}
    for lanes0 in _divisors(d0):
        if wave_size % lanes0:
            continue
        lanes1 = wave_size // lanes0
        if lanes1 < 1 or d1 % lanes1:
            continue
        t0, t1 = d0 // lanes0, d1 // lanes1
        for order in ([0, 1], [1, 0]):
            try:
                desc = make_tile_desc(shape=[d0, d1], thread_tile=[t0, t1], thread_dist=[lanes0, lanes1],
                                      thread_order=order, block_repeat=[1, 1], wave_dist=[1, 1],
                                      wave_size=wave_size)
            except (ValueError, Exception):
                continue
            name = f"lanes({lanes0}x{lanes1}) tile({t0}x{t1}) order{order[0]}{order[1]}"
            out.setdefault(name, desc.layout)
    return out
