# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The MOVEMENT COVERAGE: which tiles one instruction carries, decided on the emit plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """What `leaves.emitLdsReadTile` must do with one read act.

    `emit` -- issue an instruction (False = this act's payload rides a leader's instruction).
    `narrowed` -- the group was not mergeable, so the coverage was cut to one tile and this act emits
                 its OWN narrower load.  The leaf reads this to pick the narrowed read context.
    """
    emit: bool = True
    narrowed: bool = False
    reg_slot: int = None


#: NO DECISION.  An act whose operand theta gave no merge gets `None`, never a permissive
#: `Decision`, so a consumer can tell "theta said every act issues" from "theta said nothing".
WHOLE = None


def _act_generation(at) -> tuple:
    """The GENERATION a read act belongs to -- what every coordinate of one coverage must agree on."""
    return (at.get("reg_buf"), at.get("group"))


class CoveragePlan:
    """Per-act `Decision`s for one emitted block, keyed by the act's position in that block's plan.

    A result API rather than a bare dict so callers never key on `id(act)` and never re-derive the
    grouping rule; `violations` is the same computation read as a report."""

    def __init__(self, decisions, violations):
        self._d = list(decisions)
        self.violations = list(violations)

    def decision(self, index: int):
        """The `Decision` for the act at `index`, or None when theta supplied no merge for it.

        None means "theta said nothing", so the caller keeps its own behaviour; it does NOT mean
        "every act issues".  See `WHOLE`."""
        return self._d[index] if 0 <= index < len(self._d) else None

    @property
    def merged(self) -> int:
        """How many acts theta's merge governs -- the count worth logging."""
        return sum(1 for d in self._d if d is not None)


def carrier_tile_of(q, carrier_value, extent):
    """Any tile whose carrier is `carrier_value` -- the handle `CoverageMap.group` needs.

    `carrier_of` maps tile -> INSTRUCTION INDEX, so the two spaces must not be confused; this is
    the one place that crosses back."""
    for u in range(max(1, extent)):
        if q.carrier_of(u) == carrier_value:
            return u
    return 0


def _check_carrier_groups(acts, groups, decisions, violations, coverage_of, extent_of, folded_of):
    """For each carrier group: does one instruction serve it, and is it one generation?"""
    for (tc, _region, _k, _k_flat, carrier), idxs in groups.items():
        q = coverage_of(tc)
        n = (extent_of(tc) if extent_of else
             1 + max(acts[i].at.get("tile_flat", acts[i].at["tile"]) for i in idxs))
        grp = q.group(carrier_tile_of(q, carrier, n), n)
        want = len(grp)
        lead = min(grp) if grp else None
        for i in idxs:
            tile = acts[i].at.get("tile_flat", acts[i].at["tile"])
            decisions[i] = Decision(emit=(tile == lead), narrowed=False,
                                    reg_slot=q.carrier_of(tile))
        present = {acts[i].at.get("tile_flat", acts[i].at["tile"]) for i in idxs}
        # ONE ACT PER CARRIER GROUP, AT THE LEADER -- the invariant since theta models the coverage itself.
        #
        folded_ok = bool(folded_of and folded_of(tc)) and len(idxs) == 1
        if len(idxs) != want and not folded_ok:
            # TWO DIFFERENT FAULTS ARE REPORTED BY THIS FUNCTION AND THEY ARE NOT THE SAME ONE.
            # This branch is an INCOMPLETE CARRIER GROUP: the acts a merged instruction would serve
            violations.append(
                "operand %s: INCOMPLETE CARRIER GROUP -- the instruction at carrier %d serves %d tiles "
                "but only %d of their acts are in this block (group %s, issuing act tile %s, acts "
                "at %s), so the wide instruction would write a register whose act lives "
                "elsewhere.  A group is "
                "complete two ways: every member has its own act (theta did not fold), or exactly ONE "
                "act sits at the leader and carries the rest in `Ref.covers` (theta folded it via the "
                "read's `axis %% q == 0` first-touch guard).  Neither holds here.  This is "
                "an EMISSION fact (which acts landed), NOT a generation straddle -- see the "
                "REGISTER-generation check for that."
                % (tc, carrier, want, len(idxs), sorted(grp), lead, sorted(present)))
        gens = {_act_generation(acts[i].at) for i in idxs}
        if len(gens) > 1:
            violations.append(
                "operand %s: GENERATION STRADDLE -- the instruction "
                "at tile %d spans %d REGISTER generations %s; one instruction cannot write two "
                "`Valu*_X*` bases, so this merge has no encoding.  THIS is the hardware hazard, and "
                "an incomplete carrier group above is a different, "
                "weaker fault.  The LDS half of a straddle is StinkyTofu's, resolved from the "
                "frame contract rather than named here."
                % (tc, carrier, len(gens), sorted(gens, key=repr)))


def plan_coverage(acts, coverage_of, extent_of=None, folded_of=None) -> CoveragePlan:
    """Decide, for every read act, whether it emits -- from theta's SUPPLIED merge, not from a rule here.

 `coverage_of(operand) -> ir.CoverageMap | None` is theta's own field, and `None` (the default) is the
 identity merge: one instruction per act, which is every A/B read. `extent_of(operand) -> int`
 bounds the tile scan for a carrier preimage; it defaults to the acts actually present.
    """
    decisions = [None] * len(acts)
    violations = []
    groups = {}

    for i, act in enumerate(acts):
        if act.kind != "read":
            continue
        at = act.at
        q = coverage_of(at["tc"])
        if q is None:
            continue
        # theta'S CLAMP OVERRIDES THE SUPPLIED MAP.  The `CoverageMap` is a TARGET fact; whether a coverage
        # is LEGAL on this operand's axis is theta's, by the regularity condition ("`q` must divide
        if folded_of is not None and not folded_of(at["tc"]):
            continue
        tile = at.get("tile_flat", at["tile"])
        key = (at["tc"], at.get("region", 0), at["k"], at.get("k_flat", at["k"]),
               q.carrier_of(tile))
        groups.setdefault(key, []).append(i)

    _check_carrier_groups(acts, groups, decisions, violations, coverage_of, extent_of, folded_of)
    return CoveragePlan(decisions, violations)
