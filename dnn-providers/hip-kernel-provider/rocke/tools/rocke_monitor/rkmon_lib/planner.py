# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The planner, an optional on-prem model role (rkmon_lib/roles.py): its proposal
IS the merge plan when it keeps every rule of the plan as a hard constraint; otherwise the toolkit's own plan stands,
and says why.

The planner sees the plan as waves that land in order, wave 1 what can go in now, and nothing of how the toolkit builds
its own: the toolkit states the facts and its rules in these terms (mergeplan._planner_facts) and turns a proposal that
passed into its plan (mergeplan._proposed_layout). As plain data (waves counted from 0):
- `prs`: the PRs to place, with their readiness class, weight and size (the PRs that conflict with develop and are no
  top priority are not the planner's to place: the toolkit ranks them last, as in its own plan);
- `edges`: the conflicts, of either kind: textual on today's develop, two claims in an ABI lane that orders, a pair not
  checked yet, a pair the union check joined, or a file or ABI lane shared with a top priority that must rebase first.
  No wave holds two PRs that conflict, but for two members of a pinned group (`groups`, each together in one wave, in
  the user's order);
- `prio`: the priorities, highest first. Each lands in the earliest wave with no higher priority it conflicts with,
  and no PR that is not a priority lands in an earlier wave than a priority it conflicts with;
- `rebase`: the top priorities that must rebase first; each ends its wave, after its rebase, and the union check
  leaves it out (it has no merge with develop to check);
- `hubs`: they land after every PR but the priorities;
- `admitted`: the wave the toolkit's rules place each PR in. The proposal places it there or earlier, or names in
  `kept_out` the PR of that wave, in the proposal, that the two conflict with: a conflict the facts hold (the user's
  first-group rule of f2c3866 and 3068d06, for every wave: what can go in, goes in);
- `union` (the caller's check, when it has one): each wave's members, landed one after another, merge clean on develop.
The planner proposes the waves and their order: the priorities first, what can go in now in wave 1, then the least
rebase toil. A proposal is for one set of facts (`fp`); a later develop that leaves them alone keeps it.

The planner never arms, merges or posts. `models.planner.act` is reserved for a future switch that would let a plan
it made act on its own; it stays off, and nothing reads it yet.
"""

from __future__ import annotations

import hashlib
import json

from . import roles

ROLE = "planner"
SYSTEM = (
    "You plan the landing order of a repository's open pull requests. Answer with one JSON object only, no "
    "prose around it."
)
SET_ASIDE = "the planner's proposal was set aside: "


def payload(
    prs: list[dict],
    edges: dict,
    prio: list[int],
    groups: list[list[int]],
    hubs,
    admitted: dict,
    baseline: list[list[int]],
    rebase=(),
    care: dict | None = None,
    notes: dict | None = None,
    union: bool = False,
) -> dict:
    """The facts and hard constraints of one plan (module docstring), JSON-ready; waves count from 0. `prs` [{"pr",
    "cls", "weight", "size"}], `edges` {(a, b): kind}, `admitted` {PR: wave}, `care` {(a, b): [reasons]} (to know, no
    constraint), `notes` {what: [PR, ...]}: more facts to show."""
    p = {
        "prs": sorted(prs, key=lambda x: x["pr"]),
        "edges": sorted([a, b, k] for (a, b), k in edges.items()),
        "prio": list(prio),
        "groups": [list(x) for x in groups],
        "rebase": sorted(rebase),
        "hubs": sorted(hubs),
        "admitted": {str(m): w for m, w in sorted(admitted.items())},
        "baseline": [list(w) for w in baseline],
        "care": sorted([a, b, "; ".join(r)] for (a, b), r in (care or {}).items()),
        "notes": notes or {},
        "union": union,
    }
    # What a proposal answers to: the PRs and their classes, the conflicting pairs, and the rules' data (not the sizes,
    # the conflicts' files or the reasons, which change with no new answer due).
    key = [
        [[x["pr"], x["cls"]] for x in p["prs"]],
        [e[:2] for e in p["edges"]],
        p["prio"],
        p["groups"],
        p["rebase"],
        p["hubs"],
        p["admitted"],
        union,
    ]
    p["fp"] = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]
    return p


def messages(p: dict) -> list[dict]:
    """The planner's request: the facts, the rules it must keep, the toolkit's own waves, and the answer's form (waves
    count from 1 here)."""
    prs = "; ".join(
        f"#{x['pr']} {x['cls']} weight {x['weight']} size {x['size']}" for x in p["prs"]
    )
    lines = [
        "Place each of these pull requests once, in waves that land in order: wave 1 is what can go in now, then "
        "wave 2, and so on. The PRs of a wave land one after another, in your order, with no rebase among them.",
        f"PRs to place (readiness class, weight: higher is readier, lines changed): {prs}.",
        f"Conflicts (two PRs listed together never land in one wave, but for two members of a pinned group): "
        f"{json.dumps(p['edges'])}.",
    ]
    if p["groups"]:
        lines.append(
            f"Pinned groups, each together in one wave, in this order: {json.dumps(p['groups'])}."
        )
    if p["prio"]:
        lines.append(
            f"Priorities, highest first: {json.dumps(p['prio'])}. Each lands in the earliest wave with "
            "no higher priority it conflicts with; a PR that is not a priority never lands in an earlier wave "
            "than a priority it conflicts with."
        )
    if p["rebase"]:
        lines.append(
            f"Top priorities that must rebase first: {json.dumps(p['rebase'])}. Each ends its wave, after its "
            "rebase."
        )
    if p["hubs"]:
        lines.append(
            f"Hubs (they conflict with many): {json.dumps(p['hubs'])}. A hub lands after every PR that is "
            "neither a hub nor a priority."
        )
    if p["admitted"]:
        adm = {m: w + 1 for m, w in p["admitted"].items()}
        lines.append(
            f"The rules place each PR in a wave: {json.dumps(adm)}. Place each there or earlier; to place one "
            "later, name in kept_out the PR of that wave, in your waves, it conflicts with (one of the "
            "conflicts above)."
        )
    if p["union"]:
        lines.append(
            "Each wave's PRs, landed one after another in your order, must merge clean on develop together "
            "(the toolkit checks it)."
        )
    if p["care"]:
        lines.append(
            "Care, no conflict (the second to land may have to re-bless or renumber): "
            f"{json.dumps(p['care'])}."
        )
    lines += [f"{what}: {json.dumps(ns)}." for what, ns in p["notes"].items() if ns]
    lines += [
        "Goal: the priorities first, everything that can go in now in wave 1, then the least rebase toil: in a "
        "wave, the PRs whose landing makes no later PR rebase first; readier (higher weight) PRs earlier.",
        f"The toolkit's own waves, which you may keep or improve: {json.dumps(p['baseline'])}.",
        'Answer: {"waves": [[PR, ...], ...], "kept_out": {"PR": {"conflicts_with": PR, "evidence": "the '
        'conflict"}}, "why": "one sentence"}',
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _pr(v) -> int | None:
    """A PR number as a model may write it: 123 or "#123"."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    s = v.strip().lstrip("#") if isinstance(v, str) else ""
    return int(s) if s.isdigit() else None


def _prs(ns) -> str:
    return ", ".join(f"#{x}" for x in ns)


def _waves(v) -> list[list[int]] | None:
    """The proposal's waves as PR numbers; None when they are not lists of PR numbers."""
    if not isinstance(v, list) or not v or not all(isinstance(w, list) for w in v):
        return None
    out = [[_pr(x) for x in w] for w in v]
    return None if any(x is None for w in out for x in w) else out


def check(p: dict, ans: dict, union=None) -> list[str]:
    """Every hard constraint the proposal breaks, in words (waves counted from 1); empty when it keeps them all (module
    docstring). `union(wave)`: the caller's check of a wave's union on develop: why it fails, or "".
    """
    waves, kept = _waves(ans.get("waves")), ans.get("kept_out") or {}
    if waves is None:
        return ["its waves are not lists of PR numbers"]
    if not isinstance(kept, dict):
        return ["its kept_out is not an object"]
    kept = {str(_pr(k)): v for k, v in kept.items() if _pr(k) is not None}
    placed = [x for w in waves for x in w]
    twice = sorted({x for x in placed if placed.count(x) > 1})
    if twice:
        return [f"it places {_prs(twice)} twice"]
    out = ["it has an empty wave"] if any(not w for w in waves) else []
    want = {x["pr"] for x in p["prs"]}
    missing, extra = sorted(want - set(placed)), sorted(set(placed) - want)
    if missing:
        out.append(f"it leaves {_prs(missing)} unplaced")
    if extra:
        out.append(
            f"it places {_prs(extra)}, which are not to be placed (they conflict with develop, or are out of "
            "scope)"
        )
    at = {x: i for i, w in enumerate(waves) for x in w}
    unit = {x: tuple(grp) for grp in p["groups"] for x in grp}
    for grp in p["groups"]:
        w = waves[at[grp[0]]] if grp[0] in at else []
        if not w or w[w.index(grp[0]) : w.index(grp[0]) + len(grp)] != grp:
            out.append(
                f"the pinned group {_prs(grp)} is not together in one wave, in its order"
            )
    edges = {(min(a, b), max(a, b)): k for a, b, k in p["edges"]}
    near: dict[int, set] = {}
    for a, b in edges:
        if unit.get(a, (a,)) != unit.get(
            b, (b,)
        ):  # a pinned group's own conflicts stay: the user pinned it
            near.setdefault(a, set()).add(b)
            near.setdefault(b, set()).add(a)
    for i, w in enumerate(waves, 1):
        for j, a in enumerate(w):
            for b in w[j + 1 :]:
                if b in near.get(a, ()):
                    out.append(
                        f"its wave {i} has #{a} and #{b}, which conflict ({edges[(min(a, b), max(a, b))]})"
                    )
    rank = {x: i for i, x in enumerate(p["prio"])}
    for x in p["prio"]:
        if x not in at:
            continue
        ahead = sorted(
            y for y in near.get(x, ()) if y in at and at[y] < at[x] and y not in rank
        )
        if ahead:
            out.append(
                f"{_prs(ahead)} land{'s' if len(ahead) == 1 else ''} ahead of the priority #{x}, which "
                f"{'it conflicts' if len(ahead) == 1 else 'they conflict'} with"
            )
        free = next(
            (
                w
                for w in range(at[x])
                if not any(
                    y in near.get(z, ())
                    for z in unit.get(x, (x,))
                    for y in waves[w]
                    if rank.get(y, len(rank)) < rank[x]
                )
            ),
            None,
        )
        if free is not None:
            out.append(
                f"the priority #{x} lands in wave {at[x] + 1}, though wave {free + 1} has no higher priority "
                "it conflicts with"
            )
    rebase = set(p["rebase"])
    for i, w in enumerate(waves, 1):
        first = next((j for j, x in enumerate(w) if x in rebase), len(w))
        late = [x for x in w[first:] if x not in rebase]
        if late:
            out.append(
                f"{_prs(late)} land{'s' if len(late) == 1 else ''} after #{w[first]} in wave {i}, which must "
                "rebase first and ends its wave"
            )
    hubs, prio = set(p["hubs"]), set(p["prio"])
    first_hub = min((at[h] for h in hubs if h in at), default=None)
    if first_hub is not None:
        late = sorted(
            x for x in placed if x not in hubs and x not in prio and at[x] >= first_hub
        )
        if late:
            out.append(
                f"{_prs(late)} land{'s' if len(late) == 1 else ''} with or after a hub (wave {first_hub + 1}); "
                "hubs land after every PR but the priorities"
            )
    for m, w in p["admitted"].items():
        x = int(m)
        if at.get(x, w) <= w:
            continue
        why = kept.get(m) if isinstance(kept.get(m), dict) else {}
        y = _pr(why.get("conflicts_with"))
        if not (
            y is not None and w < len(waves) and y in waves[w] and y in near.get(x, ())
        ):
            out.append(
                f"#{x}, which the rules place in wave {w + 1}, lands in wave {at[x] + 1} without a conflict "
                f"the facts hold with a PR of wave {w + 1}"
            )
    if union and not out:
        for i, w in enumerate(waves, 1):
            clean = [x for x in w if x not in rebase]
            why = union(clean) if len(clean) > 1 else ""
            if why:
                out.append(
                    f"its wave {i} does not merge clean on develop as one: {why}"
                )
    return out


def use(st: dict, p: dict, union=None) -> tuple[list[list[int]] | None, str]:
    """(the planner's waves, how) when its last answer is for these facts and keeps every rule; else (None, why the
    toolkit's own plan stands, the planner its subject)."""
    r = roles.state(st, ROLE)
    last = r.get("last") or {}
    if last.get("fp") != p["fp"]:
        return None, (
            "the planner's proposal for these facts is on its way"
            if r.get("job") and r.get("fp") == p["fp"]
            else "the planner has no proposal for these facts yet"
        )
    if not last.get("ok"):
        return None, f"the planner failed: {str(last.get('error'))[:160]}"
    problems = check(p, last.get("answer") or {}, union)
    if problems:
        more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        return None, f"{SET_ASIDE}{problems[0]}{more}"
    return (
        _waves(last["answer"]["waves"]),
        f"its proposal of {last.get('at')} ({last.get('model')}) keeps every rule of the plan",
    )


def start(cfg: dict, st: dict, store, p: dict | None) -> str:
    """Ask for a proposal when the facts have none yet (one job at a time); the line to show, or ""."""
    if not p or not roles.on(cfg, ROLE):
        return ""
    r = roles.state(st, ROLE)
    if (r.get("last") or {}).get("fp") == p["fp"] or (
        r.get("job") and r.get("fp") == p["fp"]
    ):
        return ""
    return roles.start(cfg, st, store, ROLE, p["fp"], messages(p))
