# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The search for one MI (``classify``) and the DepthUs a buildable MI builds at (``buildable_depthus``)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from geko.config_generator.buildability.codegen import CodegenChecker, TuningSpace
from geko.config_generator.buildability.constants import MULTI_STARTS, OFAT_PASSES, REPAIRS
from geko.config_generator.buildability.search import SettingSearch, _key, repair
from geko.config_generator.mi_candidates import MI


@dataclass
class Verdict:
    mi: MI
    buildable: bool
    stage: str  # preset / core / repair / one-at-a-time / "" (unbuildable)
    checks: int
    witness: Dict[str, Any] = field(default_factory=dict)
    overflow: float = math.inf  # closest kernel's Measurement.overflow() (unbuildable MIs)
    reason: str = ""
    depthu: List[int] = field(default_factory=list)  # DepthUs it builds at (buildable_depthus)
    validations: int = 0  # validation-only checks (repair), besides the ``checks`` code generations
    unmodelled: Dict[int, int] = field(default_factory=dict)  # failures with every resource fitting, per error code
    # Resources over their cap in the kernel that built: TensileLite no longer enforces the limit as modelled
    # (the early exit and the overflow measure would then misjudge kernels).
    over_cap: List[str] = field(default_factory=list)


def _starts(failures: Sequence[Tuple[float, str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Up to MULTI_STARTS failed kernels to start the one-at-a-time stage from: the closest to fitting
    for each over-used resource, then the next closest overall. Refusals with every resource under its
    cap (negative overflow) rank last: they never led to a build. Refused settings are never starts."""
    ranked = sorted((f for f in failures if f[0] != math.inf), key=lambda f: (f[0] < 0, f[0]))
    starts, resources = [], set()
    for ov, res, s in ranked:
        if ov >= 0 and res not in resources:
            resources.add(res)
            starts.append(s)
    for ov, res, s in ranked:
        if len(starts) >= MULTI_STARTS:
            break
        if not any(s is t for t in starts):
            starts.append(s)
    starts.sort(key=lambda s: next((f[0] < 0, f[0]) for f in ranked if f[2] is s))
    return starts[:MULTI_STARTS]


def classify(checker: CodegenChecker, search: SettingSearch, mi: MI) -> Verdict:
    """Try kernels for one MI until one builds."""
    entries = search.mi_entries(mi)
    if not entries:
        return Verdict(mi, False, "", 0, reason="MI Designer emits no entry for this MI")
    checks, best, best_m, seen = 0, None, None, set()
    failures: List[Tuple[float, str, Dict[str, Any]]] = []  # (overflow, resource, setting)
    refused: List[Tuple[Dict[str, Any], str]] = []  # (setting, reason): failures without resource counts
    accepted: Dict[str, bool] = {}  # the repair stage's validation verdicts
    unmodelled: Dict[int, int] = {}

    def attempt(perms: Iterable[Dict[str, Any]], stage: str) -> Optional[Verdict]:
        nonlocal checks, best, best_m
        for s in perms:
            key = _key(s)
            if key in seen:
                continue
            seen.add(key)
            m = checker.measure(s)
            checks += 1
            if m.err == 0:
                v = Verdict(mi, True, stage, checks, s)
                if m.vgpr is not None:
                    v.over_cap = sorted(r for r, u in m._usage().items() if u > 1)
                return v
            failures.append((m.overflow(), m.resource(), s))
            if m.overflow() == math.inf:
                refused.append((s, m.reason))
            if best_m is None or m.overflow() < best_m.overflow():
                best, best_m = s, m
            if m.vgpr is not None and m.overflow() <= 0:
                unmodelled[m.err] = unmodelled.get(m.err, 0) + 1
        return None

    def stages() -> Optional[Verdict]:
        for stage, perms in (
            ("preset", lambda e: search.presets(mi, e)),
            ("core", lambda e: search.core(mi, e)),
        ):
            for entry in entries:
                v = attempt(perms(entry), stage)
                if v:
                    return v
        # Refused settings give stage 3 no distance: repair one per (MI entry, DepthU, rule), numbers masked
        # so one rule broken with different values counts once.
        repaired, before_repair = set(), len(failures)
        for s, reason in list(refused):
            group = (_key(s[search.space.mi_group]), s.get("DepthU"), re.sub(r"\d+", "#", reason))
            if group in repaired:
                continue
            if len(repaired) == REPAIRS:
                break
            repaired.add(group)
            v = attempt(repair(checker, search, s, reason, entries, accepted), "repair")
            if v:
                return v
        # The failures stages 1-2 measured come first, so repair-generated kernels never crowd out a start
        # the search would have used without them; repair-generated ones fill the remaining slots.
        starts = _starts(failures[:before_repair])
        starts += [s for s in _starts(failures[before_repair:]) if not any(s is t for t in starts)]
        for start in starts[:MULTI_STARTS]:
            for _ in range(OFAT_PASSES):
                n = len(failures)
                v = attempt(search.one_at_a_time(start, entries), "one-at-a-time")
                if v:
                    return v
                closer = min(failures[n:], key=lambda f: f[0], default=None)
                start_ov = next(f[0] for f in failures if f[2] is start)
                if closer is None or closer[0] >= start_ov:
                    break
                start = closer[2]
        return None

    v = stages() or Verdict(
        mi,
        False,
        "",
        checks,
        best or {},
        best_m.overflow() if best_m else math.inf,
        best_m.reason if best_m else "",
    )
    v.validations, v.unmodelled = len(accepted), unmodelled
    return v


def searched_depthu(space: TuningSpace) -> List[int]:
    """The DepthUs buildable_depthus tries: every integer DepthU of the space, ascending."""
    return sorted(d for d in space.fork.get("DepthU", []) if isinstance(d, int))


def buildable_depthus(
    checker: CodegenChecker, search: SettingSearch, mi: MI, witness: Dict[str, Any]
) -> List[int]:
    """Every DepthU of the space at which ``mi`` builds, for an MI whose ``witness`` setting built.

    Upward from the witness's DepthU, the search stops at the first DepthU where classify finds no kernel:
    a larger DepthU stages more of A and B in LDS and registers, so resource use only grows. That held for
    all 900 gfx950 MIs checked at every DepthU (300 each of BBS NN, F8BS NT, SSS NN), and 0 of 481 sampled
    upward stops built under a much deeper search (3,000 attempts each). Each stop is a full search that
    fails, which roughly doubles a list's generation time.

    Every DepthU below the witness is searched: there, validity rules decide as much as resources, and they
    need not be monotonic in DepthU. The witness is usually at or near the smallest legal DepthU (the
    presets and factorial start there), and a DepthU below MFMA K x local split U has no legal MI entry, so
    classify rejects it without code generation.
    """
    dus = searched_depthu(search.space)
    w = witness.get("DepthU")
    if w not in dus:
        return []
    i = dus.index(w)
    found = [w]
    for d in dus[i + 1 :]:
        if not classify(checker, search.pinned(d), mi).buildable:
            break
        found.append(d)
    below = [d for d in dus[:i] if classify(checker, search.pinned(d), mi).buildable]
    return below + found
