# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The merge plan: which near-merge PRs can land together without a conflict, in what order, and who rebases when,
so the dashboard recommends the merge order that saves the most rebases. Only real textual conflicts count: a shared
hot spot alone (platform/CMakeLists.txt, for example) is no conflict, and most PR pairs that share one merge clean.

**Scope.** The near-merge PRs, about 30: non-draft PRs into develop without a hold label, with our 🟢/🟡 (at head or
before a later push) or at least one approval; develop fixes and the user's priorities always.

**Conflicts, on today's develop.** Each PR is merged with develop once (`git merge-tree --write-tree develop head`),
cached by (head, develop). Two PRs whose changed files do not overlap cannot conflict; for the others the two merge
trees are merged on develop (`--merge-base=develop T_A T_B`), cached by (heads, develop's blobs of the shared files),
so a develop move elsewhere costs no pair merge. A PR that conflicts with develop gets no pairs: it rebases first, and
the plan names the PRs it shares the most files with. A tick runs at most `max_merges` merges in `budget_s` seconds
(`merge_plan` in config.json), the readiest PRs first; a pair not checked yet counts as a conflict and marks the plan
partial. No GitHub call. The pairs land in st["pair_cache"], which batches and the hold rule read: only textual
conflicts on develop batch or hold anything.

**ABI conflicts** (the ABI differences help determine the merge order). Two PRs with claims on one ABI counter (the binary lane,
ROCKE_ABI_VERSION, or the wire lane, ROCKE_RECIPE_ABI; a stack or PRs with an identical bump hunk are one claim, as
abi.lane_claims groups them) conflict even when git merges them cleanly: the later to land renumbers to develop + 1,
and an abi.h bump moves its portable_ir mirror too (BINARY_ABI, RECIPE_ABI). They order the plan as textual conflicts
do (never two in one wave; the earlier in plan order first, and the later waits for it), but they never hold a PR or
make a hub, and those of a PR the hold rule keeps waiting (held, or waiting for a held PR: queue.hold_bound) are shown,
not ordered: the rule exempts ABI overlaps (the next ready lane member may take the number),
so such a PR renumbers after the rest of its lane. The queue reads the claims before the plan, so who is in a lane
never depends on the order; the lanes then follow the plan (7. below).

**Care** (no conflict, but the second to land may have to re-bless, renumber or re-check): the same
representative-golden shard, differential-golden family or ABI family string (`<FAMILY>_ABI_VERSION`), the same CMake
list or test-category file, or overlapping new parity config indices (C emitters' `case N:`; the first to land keeps
the index). Shown and weighted, never used to order, batch or hold.

**Order.** Each row, or wave, is a group of PRs that can be merged together in the given order without a conflict:
what can go in right now without causing issues in the same wave. A wave ideally ends with a PR that causes conflicts,
and prioritized PRs that conflict with each other go in different waves. A conflict is
either kind, or a pair not checked yet. Readiness weights: top priority (develop fixes, the user's priorities) and
ready 8 (R0; the ABI lane turn aside, as the plan orders the lanes), only approvals or threads (or a CI run) left 4
(R1), approved but fixes or our follow-up left 2 (R3), the rest 1 (R4).
1. A wave is a set of PRs with no conflict between any two: they land in its order, with no rebase among them. Wave 1
   is what can go in now. The waves are built one at a time from the PRs left, each as 2. and 3. say.
2. The priorities, in order: pinned groups (`rk group`, one node each, in the user's order), the top priorities (a
   develop fix CI sees, then the user's priorities, then other develop fixes), held PRs (they keep their place), and
   ready PRs the plan has kept waiting `STARVE_H` hours. Each priority left joins the wave unless it conflicts with one
   already in it, so conflicting priorities split across waves, the higher one first.
3. Then the wave takes in the maximum-weight set of the other PRs left that conflict with none of its members, solved
   exactly (ties: more ready PRs, then older ones). Left out: a PR that conflicts with a priority still waiting (it
   lands after that priority, so no PR makes a priority rebase), and hubs (5.). A wave is never empty: as a guard, it
   takes the smallest PR left.
4. A top priority that conflicts with develop is a priority like the others, placed by the evidence there is: the
   matrix merges no pair with it, so a changed file in common counts as a conflict, as does another claim in one of its
   ABI lanes when the hold rule keeps neither waiting (top priorities that each conflict with develop in one file can
   share many files with each other, and must not all end one wave). So it joins
   the earliest wave where no member placed before it shares a file with it, and the PRs that do land after it (3.). It
   ends its wave, "after its rebase": never `go`, and out of the union check (no develop merge to check). In wave 1 it
   rebases now; in a later wave, once the wave before it has landed, so the landings ahead make it rebase once. The
   other PRs that conflict with develop rank last ("after a rebase").
5. Hubs last: a PR with `HUB_DEGREE` or more textual conflicts with the other PRs outside the priorities joins the
   waves once no other PR is left. An ABI conflict, or a textual one only on the ABI counters, does not count: every
   lane member has one with every other, and the lane serializes them anyway.
6. A wave ends with the PRs that cause conflicts: first the members whose landing makes no PR in a later wave rebase,
   then the others (a conflict of either kind with a PR in a later wave), then the top priorities that rebase (4.). In
   each of the first two parts the priorities in their order, then the readiest, then the lowest weighted rebase cost
   (the weights of the later PRs its landing makes rebase or renumber), then the smaller PR, then the older.
7. The ABI lanes follow the plan: the lane head is the lane member in the earliest wave (queue.abi_lanes reads
   `plan_rank`), and each later claim renumbers once the claim ahead of it has landed (the rebase advice says when).
**Union check** (does the whole stack A + B + C + D still merge clean?). The pairs say nothing of three PRs together, so each wave's members are merged on develop one after another, in its landing
order (`unions`): a chain per set of members linked by shared changed files, as members with no file in common cannot
conflict, so most waves cost 0-2 merges. A member that conflicts with the union of those before it (each alone merges
clean) conflicts from then on with each earlier member that changes one of the files (every earlier one when none
does), and the waves are laid out again, until every wave's union merges clean; each round adds an edge, so it ends.
The steps are cached by develop and the chain's heads so far (`unions` in the cache), and share the tick's merge budget
after the pairs: a wave whose chain is not checked to its end keeps the plan partial.
**The planner** (rkmon_lib/planner.py, an optional model role, off by default) sees the plan as waves and the rules above as hard constraints (_planner_facts): no conflict inside a wave (either kind,
a pair not checked yet, one the union check found, or a file or ABI lane shared with a top priority that must rebase
first), each wave's union merges clean on develop, each priority in the earliest wave with no higher priority it
conflicts with and no PR that is not a priority in an earlier wave than a priority it conflicts with (2.-4.), a top
priority that must rebase first at the end of its wave (4.), hubs after every PR but the priorities (5.), and each PR in
the wave the rules place it in, or earlier, unless the proposal names the conflict that keeps it out: one the facts
hold, with a PR of that wave. A proposal for these facts that keeps them all is the plan (_proposed_layout); otherwise
the waves above stand. plan["made_by"] says which, plan["planner"] why. Its union checks share the tick's merge budget,
after the toolkit's own.
A PR may land now (`plan_go`, which feeds the batches) when it is in wave 1, merges clean with develop, and no PR ahead
of it conflicts (either kind, or the union): every PR in wave 1 but a top priority that must rebase first and a pinned
group's member that conflicts with an earlier one.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time

from . import abi, breakage, gitops, idle, paths, registry, roles
from .conflicts import short, within
from .queue import (
    FAIL,
    GOOD_VERDICTS,
    PASS,
    USER_PRIORITY,
    abi_only,
    hold_bound,
    hold_labels,
    hot_paths,
    path_hits,
    pr_hotspots,
)
from .store import minutes_since, now

WEIGHT = {"T0": 8, "R0": 8, "R1": 4, "R3": 2, "R4": 1}
HUB_DEGREE = 5  # conflicts (the ABI lane aside) with this many other remaining PRs make a hub: it lands after them
STARVE_H = 24  # a ready PR the plan has kept waiting this long lands anyway
DEFAULTS = registry.package_section("merge_plan")
CACHE = "merge_plan_cache"
EMITTER = re.compile(
    r"/(?:library/tests/parity|platform/tests/instances/parity)/(\w+)_emit\.c$"
)
CASE = re.compile(r"^\s*case\s+(\d+)\s*:", re.M)
CATEGORIES = re.compile(r"_test_categories[^/]*\.yaml$")
MAX_EXACT = 40  # a conflict component larger than this is solved greedily (real ones stay near 20)
# The planner's facts of the last plan built (rkmon_lib/planner.py): the tick asks for a proposal on them. Kept here,
# never in the state.
PLANNER: dict = {"payload": None}


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("merge_plan") or {}))


def _base(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _span(nums: list[int]) -> str:
    """[18, 19, 20] -> "18–20"; [18, 20] -> "18, 20"."""
    if len(nums) > 1 and nums[-1] - nums[0] == len(nums) - 1:
        return f"{nums[0]}–{nums[-1]}"
    return ", ".join(str(x) for x in nums)


def some(items: list, most: int = 3) -> str:
    return ", ".join(str(x) for x in items[:most]) + (
        f" (+{len(items) - most})" if len(items) > most else ""
    )


def _hot(cfg: dict, key: str) -> list[str]:
    return hot_paths(cfg, key) if key in (cfg.get("hot_spots") or {}) else []


# --- scope and readiness -------------------------------------------------------------------------------------------


def scope(cfg: dict, st: dict, rows: list[dict]) -> list[dict]:
    """The near-merge PRs among the queue's rows (module docstring). A freeze leaves them in: it stops arming, not the
    order."""
    fixes = st.get("develop_fixes") or {}
    out = []
    for r in rows:
        rec = st["prs"][str(r["n"])]
        near = (
            rec.get("base") == "develop"
            and not hold_labels(cfg, rec)
            and (
                (rec.get("ours") or {}).get("verdict") in GOOD_VERDICTS
                or any(x.get("state") == "APPROVED" for x in rec.get("reviews") or [])
            )
        )
        if near or str(r["n"]) in fixes:
            out.append(r)
    return out


def readiness(st: dict, r: dict) -> str:
    """A planned PR's class (module docstring) from its queue row: the gates before G9, and its hold. G4 does not
    count: the plan merges every PR with develop itself."""
    if str(r["n"]) in (st.get("develop_fixes") or {}):
        return "T0"
    if r.get("held"):
        return "R4"
    g = {k: r["gates"][k][0] for k in ("G1", "G2", "G3")}
    if g["G1"] == PASS and g["G2"] == PASS and g["G3"] == PASS:
        return "R0"
    if g["G1"] == PASS and FAIL not in (g["G2"], g["G3"]):
        return "R1"
    if g["G2"] == PASS and not r["gates"]["G1"][1].startswith(
        "our verdict \U0001f534"
    ):  # an open 🔴 is further out
        return "R3"
    return "R4"


# --- the conflict matrix ---------------------------------------------------------------------------------------------


def merger(cfg: dict) -> tuple:
    """The tick's merges: (merge(a, b, base=None) -> (tree, conflicts) of one merge, or None once the tick's merge
    budget is spent; the budget, {"done": merges so far, "start": monotonic seconds}). The matrix merges first; the
    waves' union chains take what is left (module docstring, Union check)."""
    c, budget = settings(cfg), {"done": 0, "start": time.monotonic()}

    def merge(a: str, b: str, base: str | None = None) -> tuple | None:
        if (
            budget["done"] >= c["max_merges"]
            or time.monotonic() - budget["start"] >= c["budget_s"]
        ):
            return None
        budget["done"] += 1
        try:
            return gitops.merge_tree(
                cfg["paths"]["repo"], a, b, base=base, timeout=c["timeout_s"]
            )
        except subprocess.TimeoutExpired:
            return None, None

    return merge, budget


def matrix(cfg: dict, st: dict, rows: list[dict], develop: str) -> dict | None:
    """Every near-merge PR merged with develop, and every pair of clean ones that shares a file merged on develop
    (module docstring). None when git could not merge a single one with develop. Its merges (`merge`, `budget`) go on
    to the waves' union chains (build)."""
    nodes = scope(cfg, st, rows)
    cache = st.setdefault(CACHE, {})
    old_trees, old_pairs = cache.get("trees") or {}, cache.get("pairs") or {}
    trees, kept = {}, {}
    merge, budget = merger(cfg)
    prio = {r["n"]: WEIGHT[readiness(st, r)] for r in nodes}
    heads, files, clean, devc, unknown = {}, {}, {}, {}, []
    for r in sorted(nodes, key=lambda r: (-prio[r["n"]], r["n"])):
        n, rec = r["n"], st["prs"][str(r["n"])]
        heads[n] = rec["head"]
        try:
            changed = pr_hotspots(cfg, rec).get("files")
        except (gitops.GitError, OSError):
            changed = None
        if changed is None:
            unknown.append(n)
            continue
        files[n] = set(changed)
        key = f"{rec['head'][:12]}:{develop[:12]}"
        t = old_trees.get(key)
        if t is None:
            got = merge(develop, rec["head"])
            if not got or got[1] is None:
                unknown.append(n)
                continue
            t = {"tree": got[0], "conflicts": got[1]}
        trees[key] = t
        if t["conflicts"]:
            devc[n] = t["conflicts"]
        else:
            clean[n] = t["tree"]
    if nodes and not clean and not devc:
        return None
    ns = sorted(clean, key=lambda n: (-prio[n], n))
    todo = [
        (a, b, files[a] & files[b])
        for i, a in enumerate(ns)
        for b in ns[i + 1 :]
        if files[a] & files[b]
    ]
    special = {f for n in clean for f in files[n] if _special(cfg, f)}
    want = sorted(set().union(*(s for _, _, s in todo), special))
    blobs = gitops.tree_blobs(cfg["paths"]["repo"], develop, want) if want else {}
    pairs = {}
    for a, b, shared in sorted(
        todo, key=lambda t: (-(prio[t[0]] + prio[t[1]]), t[0], t[1])
    ):
        digest = hashlib.sha1(
            "".join(f"{p} {blobs.get(p, '-')}\n" for p in sorted(shared)).encode()
        ).hexdigest()[:12]
        x, y = sorted((heads[a], heads[b]))
        key = f"{x[:12]}:{y[:12]}:{digest}"
        res = old_pairs.get(key)
        if res is None:
            got = merge(clean[a], clean[b], base=develop)
            res = {"x": got[1]} if got and got[1] is not None else None
        if res is not None:
            kept[key] = res
        pairs[(min(a, b), max(a, b))] = None if res is None else res["x"]
    cache["trees"], cache["pairs"] = trees, kept
    return {
        "develop": develop,
        "scope": [r["n"] for r in nodes],
        "heads": heads,
        "files": files,
        "clean": clean,
        "dev_conflicts": devc,
        "unknown": sorted(unknown),
        "pairs": pairs,
        "blobs": blobs,
        "merges": budget["done"],
        "secs": round(time.monotonic() - budget["start"], 2),
        "merge": merge,
        "budget": budget,
    }


def _special(cfg: dict, f: str) -> bool:
    """A file whose content the care relation reads: the differential golden, a parity C emitter."""
    return path_hits(_hot(cfg, "DG"), f) or bool(EMITTER.search(f))


def conflicts(M: dict, a: int, b: int) -> list[str] | None:
    """The files a and b conflict on, on develop: [] = clean (or no shared file), None = not checked yet."""
    return M["pairs"].get((min(a, b), max(a, b)), [])


def write_pair_cache(st: dict, M: dict) -> None:
    """The pairs on today's develop, in the pair cache batches and the hold rule read (key: the two heads)."""
    pc, ns = {}, sorted(M["clean"])
    for i, a in enumerate(ns):
        for b in ns[i + 1 :]:
            x = conflicts(M, a, b)
            pc[f"{M['heads'][a][:12]}:{M['heads'][b][:12]}"] = {
                "conflicts": (x or [])[:6],
                "undecided": x is None,
            }
    st["pair_cache"] = pc


def conflict_score(M: dict) -> dict[int, list]:
    """The hold rule's input: for each clean PR, the PRs it conflicts with on develop (a pair not checked yet counts)."""
    score = {n: [] for n in M["clean"]}
    for (a, b), x in M["pairs"].items():
        if x is None or x:
            score[a].append((b, ["files"]))
            score[b].append((a, ["files"]))
    return score


# --- care --------------------------------------------------------------------------------------------------------


def dg_families(before: str, after: str) -> set[str]:
    """The differential-golden families whose entries differ (configs added, changed or removed, any section)."""

    def flat(text: str) -> dict:
        out, stack = {}, [((), json.loads(text) if text else {})]
        while stack:
            path, v = stack.pop()
            if isinstance(v, dict):
                stack.extend(
                    (path + (k,), x) for k, x in v.items() if path or k != "aliases"
                )
            elif (
                len(path) >= 3
            ):  # ir_canonical/<family>/<idx>, ll/<flavor>/<family>/<idx>
                out[path] = v
        return out

    a, b = flat(before), flat(after)
    return {k[-2] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def new_cases(before: str | None, after: str | None) -> list[int]:
    """Config indices (`case N:`) a parity C emitter has after and not before."""
    return sorted(
        {int(x) for x in CASE.findall(after or "")}
        - {int(x) for x in CASE.findall(before or "")}
    )


def facts(cfg: dict, st: dict, M: dict) -> dict[int, dict]:
    """What each clean PR's merge touches that care compares: representative-golden shards, differential-golden
    families, ABI family strings, CMake lists, test-category files and new parity config indices. The two read from
    file contents (golden families, emitter indices) compare its merge tree with develop, cached by (head, develop
    blob). The ABI counters are no care: two claims on one are an ABI conflict (abi_conflicts).
    """
    rg, dg = _hot(cfg, "RG"), _hot(cfg, "DG")
    cache = st.setdefault(CACHE, {})
    old, kept, read, out = cache.get("facts") or {}, {}, [], {}
    for n, tree in M["clean"].items():
        rec, info = st["prs"][str(n)], {
            "rg": set(),
            "dg": set(),
            "abi": set(),
            "cmk": set(),
            "cat": set(),
            "idx": {},
        }
        for f in sorted(M["files"][n]):
            if path_hits(rg, f):
                info["rg"].add(_base(f).removesuffix(".json"))
            if f.endswith(("CMakeLists.txt", ".cmake")):
                info["cmk"].add(f)
            if CATEGORIES.search(f):
                info["cat"].add(f)
            if _special(cfg, f):
                read.append(
                    (
                        n,
                        f,
                        f"{M['heads'][n][:12]}:{M['blobs'].get(f, '-')[:12]}:{f}",
                        f"{tree}:{f}",
                    )
                )
        c = rec.get("abi_cache") or {}
        if (
            c.get("head") == rec["head"]
        ):  # this head's claims (the queue refreshes them before the plan)
            info["abi"] = {
                x["name"] for x in c.get("families") or [] if x.get("kind") == "changed"
            }
        out[n] = info
    miss = [(f, key, spec) for _, f, key, spec in read if key not in old]
    texts = gitops.cat_blobs(
        cfg["paths"]["repo"],
        sorted({s for f, _, spec in miss for s in (spec, f"{M['develop']}:{f}")}),
    )
    for f, key, spec in miss:
        before, after = texts.get(f"{M['develop']}:{f}"), texts.get(spec)
        try:
            old[key] = (
                sorted(dg_families(before or "", after or ""))
                if path_hits(dg, f)
                else new_cases(before, after)
            )
        except ValueError:  # a golden that does not parse: no family known
            old[key] = []
    for n, f, key, _ in read:
        kept[key] = old[key]
        if path_hits(dg, f):
            out[n]["dg"] |= set(old[key])
        elif old[key]:
            out[n]["idx"][EMITTER.search(f).group(1)] = old[key]
    cache["facts"] = kept
    return out


def care_reasons(fa: dict, fb: dict, rocke_path: str) -> list[str]:
    """Why two PRs with no textual conflict still interact (module docstring)."""
    out = [f"golden shard {s}" for s in sorted(fa["rg"] & fb["rg"])]
    fams = sorted(fa["dg"] & fb["dg"])
    if fams:
        out.append(f"DG {some(fams, 2)}")
    out += [f"ABI {lane}" for lane in sorted(fa["abi"] & fb["abi"])]
    out += [f"CMake {within(f, rocke_path)}" for f in sorted(fa["cmk"] & fb["cmk"])]
    out += [f"test categories {_base(f)}" for f in sorted(fa["cat"] & fb["cat"])]
    for fam in sorted(fa["idx"].keys() & fb["idx"].keys()):
        both = sorted(set(fa["idx"][fam]) & set(fb["idx"][fam]))
        if both:
            out.append(f"parity {fam} idx {_span(both)} (the first to land keeps it)")
    return out


# --- ABI conflicts ---------------------------------------------------------------------------------------------------

ABI_LABEL = {
    "binary": "both bump ROCKE_ABI_VERSION: the later renumbers to develop + 1",
    "recipe": "both bump ROCKE_RECIPE_ABI: the later renumbers to develop + 1",
}


def abi_conflicts(st: dict, clean) -> tuple[dict, dict]:
    """The ABI conflicts between the clean PRs, {(a, b): [lanes]}: two claims in one counter's lane (module docstring),
    and every lane's claims among the open PRs whose claims are read at their head (abi.lane_claims).
    """
    recs = [
        r
        for r in st["prs"].values()
        if r.get("status") == "open"
        and not r.get("draft")
        and (r.get("abi_cache") or {}).get("head") == r.get("head")
    ]
    lanes, cs, out = abi.lane_claims(recs), set(clean), {}
    for lane, claims in lanes.items():
        for i, ca in enumerate(claims):
            for cb in claims[i + 1 :]:
                for a in ca["members"]:
                    for b in cb["members"]:
                        if a in cs and b in cs:
                            out.setdefault((min(a, b), max(a, b)), []).append(lane)
    return out, lanes


def abi_step(plan: dict, n: int) -> tuple[str, bool]:
    """The ABI part of PR n's rebase advice, lane by lane in plan order, and whether a
    claim ahead of it makes it wait: the first claim takes develop + 1 (as G9 asks of the lane head); each later one
    renumbers to develop + 1 once the claim ahead of it has landed, and moves its portable_ir mirror with it.
    """
    steps, waits = [], False
    for lane, L in (plan.get("abi") or {}).items():
        i = next((i for i, c in enumerate(L["claims"]) if n in c["members"]), None)
        if i is None:
            continue
        name, mirror, d, c = (
            abi.DEFINES[lane],
            abi.MIRRORS[lane],
            L.get("develop"),
            L["claims"][i],
        )
        if i:
            waits = True
            steps.append(
                f"after {tagged(plan, L['claims'][i - 1]['members'][0])} lands: renumber {name} to "
                f"develop + 1 (and {mirror})"
            )
            continue
        taken = c.get("taken") or []
        keep = (
            f"; keep develop's history line{'s' if len(taken) > 1 else ''} {_span(taken)}"
            if taken
            else ""
        )
        if not isinstance(d, int):
            steps.append(f"first in the {name} lane: it takes develop + 1{keep}")
        elif c["to"] == d + 1:
            steps.append(f"first in the {name} lane: it takes {d + 1}{keep}")
        else:
            steps.append(
                f"first in the {name} lane: renumber {c['to']} to {d + 1} = develop + 1 (and {mirror}){keep}"
            )
    return "; ".join(steps), waits


# --- waves -----------------------------------------------------------------------------------------------------------


def mwis(nodes, weight: dict, adj: dict) -> set:
    """A maximum-weight set of `nodes` with no edge between any two, exact: branch and bound on each connected component.
    The same graph always gives the same set (fixed search order; the first optimum found stays).
    """
    left, out = set(nodes), set()
    while left:
        comp, stack = set(), [min(left)]
        while stack:
            v = stack.pop()
            if v not in comp:
                comp.add(v)
                stack.extend((adj[v] & left) - comp)
        left -= comp
        nb = {v: frozenset(adj[v] & comp) for v in comp}
        if len(comp) > MAX_EXACT:
            out |= _greedy(comp, weight, nb)
            continue
        best = [-1, frozenset()]

        def rec(cand: frozenset, picked: frozenset, total: int) -> None:
            if total + sum(weight[v] for v in cand) <= best[0]:
                return
            free = frozenset(v for v in cand if not nb[v] & cand)
            if free:  # nothing left to trade them against
                cand, picked, total = (
                    cand - free,
                    picked | free,
                    total + sum(weight[v] for v in free),
                )
            if not cand:
                if total > best[0]:
                    best[:] = [total, picked]
                return
            v = max(cand, key=lambda x: (len(nb[x] & cand), weight[x], -x))
            rec(cand - {v} - nb[v], picked | {v}, total + weight[v])
            rec(cand - {v}, picked, total)

        rec(frozenset(comp), frozenset(), 0)
        out |= best[1]
    return out


def _greedy(comp: set, weight: dict, nb: dict) -> set:
    picked, cand = set(), set(comp)
    while cand:
        v = max(cand, key=lambda x: (weight[x] / (1 + len(nb[x] & cand)), -x))
        picked.add(v)
        cand -= {v} | nb[v]
    return picked


def waves(
    nodes, weight: dict, adj: dict, hubs: set, prio=(), size: dict | None = None
) -> list[list]:
    """Conflict-free waves over `nodes`, built one at a time from the nodes left (module docstring, Order 1.-5.): the
    priorities `prio` left, in their order, each unless it conflicts with one already in the wave; then the
    maximum-weight set of the other nodes left that conflict with no member and with no priority still waiting, hubs
    only once no other node is left. A wave holds one node at least (a guard: the smallest node left by `size`, then
    number). Members in the order they joined: the priorities first."""
    ns, size = set(nodes), size or {}
    waiting, left, out = [p for p in prio if p in ns], ns - set(prio), []
    while waiting or left:
        wave = []
        for p in waiting:
            if not adj[p] & set(wave):
                wave.append(p)
        waiting = [p for p in waiting if p not in wave]
        inside = set(wave)
        # Out of the fill: a node that conflicts with a priority still waiting (it lands after it), and hubs while
        # another node is left.
        skip = set().union(*(adj[p] for p in waiting)) | (
            hubs if left - hubs else set()
        )
        fill = mwis(
            [n for n in left if n not in skip and not adj[n] & inside], weight, adj
        )
        if not wave and not fill:
            fill = {min(left, key=lambda n: (size.get(n, 0), n))}
        left -= fill
        out.append(wave + sorted(fill))
    return out


def _linked(wave: list[int], files: dict) -> list[list[int]]:
    """The wave's members in sets linked by shared changed files, each in landing order; the sets by their first."""
    sets = []
    for n in wave:
        hit = [s for s in sets if any(files[n] & files[m] for m in s)]
        sets = [s for s in sets if s not in hit] + [[m for s in hit for m in s] + [n]]
    return sorted(
        (sorted(s, key=wave.index) for s in sets), key=lambda s: wave.index(s[0])
    )


def unions(
    M: dict, waves_: list[list[int]], rep_of: dict, old: dict, kept: dict
) -> tuple[list[dict], int]:
    """Each wave's union on develop (module docstring, Union check): per set of members linked by shared files, the
    members' develop merges merged one after another in landing order. Returns the conflicts, one per chain at most,
    {"pr": the member that conflicts with the union of those before it, "with": the earlier ones that change a file it
    conflicts on (all of them when none does), "files": ...}, and the number of waves not checked to the end (the merge
    budget ran out). A pinned group's own conflicts stay: the user pinned it. Steps are read from `old` and kept in
    `kept`, by develop and the chain's heads so far."""
    develop, files, found, short = M["develop"], M["files"], [], 0
    for wave in waves_:
        whole = True
        for chain in _linked(wave, files):
            tree, seen = M["clean"][chain[0]], [M["heads"][chain[0]][:12]]
            for i, k in enumerate(chain[1:], 1):
                seen.append(M["heads"][k][:12])
                key = f"{develop[:12]}:{hashlib.sha1(':'.join(seen).encode()).hexdigest()[:16]}"
                res = old.get(key) or kept.get(key)
                if res is None:
                    got = M["merge"](tree, M["clean"][k], develop)
                    if (
                        not got or got[1] is None
                    ):  # the budget is spent, or git could not decide
                        whole = False
                        break
                    res = {"tree": got[0], "x": got[1]}
                kept[key] = res
                if not res["x"]:
                    tree = res["tree"]
                    continue
                hit = [e for e in chain[:i] if files[e] & set(res["x"])] or chain[:i]
                if all(rep_of[e] == rep_of[k] for e in hit):
                    continue
                found.append({"pr": k, "with": hit, "files": sorted(res["x"])})
                break
        short += not whole
    return found, short


# --- the plan --------------------------------------------------------------------------------------------------------


def _priority_order(cfg: dict, st: dict, ns: list[int], size: dict) -> list[int]:
    """The top priorities in landing order: a develop fix CI sees (a CI-red issue, an incident) first, then the user's
    priorities (they block other work), then other develop fixes;
    smaller first."""
    fixes = st.get("develop_fixes") or {}
    red = {  # a CI-red issue's fix, by the issue its reason names ("<label> #N")
        f"#{i['n']}"
        for i in (st.get("post_merge") or {}).get("open") or []
        if breakage.entry_kind(cfg, i) == breakage.RED
    }

    def ci_sees(w: str) -> bool:
        return w.startswith("incident") or (
            not w.startswith(USER_PRIORITY) and w.rsplit(" ", 1)[-1] in red
        )

    def key(n: int) -> tuple:
        why = fixes.get(str(n)) or []
        return (
            (
                0
                if any(ci_sees(w) for w in why)
                else 1 if any(w.startswith(USER_PRIORITY) for w in why) else 2
            ),
            size.get(n, 0),
            n,
        )

    return sorted(ns, key=key)


def why_first(st: dict, n: int) -> str:
    why = (st.get("develop_fixes") or {}).get(str(n)) or []
    return (
        "your priority"
        if any(w.startswith(USER_PRIORITY) for w in why)
        else "fixes develop" if why else ""
    )


def _layout(g: dict, prio: list) -> dict:
    """Landing order for the node graph `g` (module docstring, Order): its waves (waves(), with the priorities `prio`
    in their order), each in the order of Order 6.; ranks per PR, each node's wave (1 first), the hubs, and what may
    land now. `waves` holds the PRs that merge clean with develop; a top priority that must rebase first (a node of
    g["rebase"]) has a wave in `pos` and a rank, and ends its wave."""
    members, adj, radj, gw, gcls, size = (
        g["members"],
        g["adj"],
        g["radj"],
        g["gw"],
        g["gcls"],
        g["size"],
    )
    virtual = g["rebase"]
    pi = {n: i for i, n in enumerate(prio)}
    rest = {n for n in members if n not in pi}
    hubs = {n for n in rest if len(radj[n] & rest) >= HUB_DEGREE}
    # The exact solve's weights: readiness first, then the number of ready PRs, then older PRs (stable ties).
    age = {n: i for i, n in enumerate(sorted(members, reverse=True))}
    W = {
        n: gw[n] * 2**24 + (2**16 if gcls[n] in ("T0", "R0") else 0) + min(age[n], 255)
        for n in members
    }
    big = {n: sum(size[m] for m in members[n]) for n in members}
    wv = waves(members, W, adj, hubs, prio, big)
    pos = {n: i for i, w in enumerate(wv, 1) for n in w}
    cost = {n: sum(gw[m] for m in adj[n] if pos[m] > pos[n]) for n in members}

    def key(
        n: int,
    ) -> (
        tuple
    ):  # Order 6.: no PR in a later wave to rebase first, then priorities, readiest, cheapest
        if n in virtual:  # after its rebase: last, in priority order
            return (1, pi[n])
        return (
            0,
            cost[n] > 0,
            n not in pi,
            pi.get(n, 0),
            -WEIGHT[gcls[n]],
            cost[n],
            big[n],
            n,
        )

    wv = [sorted(w, key=key) for w in wv]
    rank = {m: i for i, m in enumerate(m for w in wv for n in w for m in members[n])}
    xadj = g["xadj"]
    go = {
        m
        for m, i in rank.items()
        if pos[g["rep_of"][m]] == 1
        and m not in virtual
        and not any(rank[x] < i for x in xadj[m])
    }
    return {
        "waves": [[m for n in w if n not in virtual for m in members[n]] for w in wv],
        "hubs": hubs,
        "pos": pos,
        "rank": rank,
        "go": go,
    }


def _rivals(M: dict, lanes: dict, bound: set, p: int, ns) -> set:
    """The PRs among `ns` that conflict with p, a top priority that must rebase first, by the evidence there is (Order
    4.): they change a file it changes (the matrix merges no pair with a PR that conflicts with develop, so a shared
    file is the best evidence, and counts), or hold another claim in one of its ABI lanes (`lanes`: abi.lane_claims);
    an ABI lane only when the hold rule keeps neither of the two waiting, as for the ABI conflicts that order the
    plan."""
    rivals = set()
    if p not in bound:
        for claims in lanes.values():
            if any(p in c["members"] for c in claims):
                rivals |= {
                    m for c in claims if p not in c["members"] for m in c["members"]
                }
    return {
        n
        for n in ns
        if n != p and (M["files"][n] & M["files"][p] or n in rivals - bound)
    }


def _edges(
    M: dict, xadj: dict, xabi: dict, bound: set, joint: list, rebase: set
) -> dict:
    """The conflicts that order the plan, in words, for the planner: {(a, b): "textual: f1, f2" | "not checked yet" |
    "ABI binary" | "together with the PRs before it in a wave: f1" | "shares f1 with a PR that must rebase first" ...}
    for each pair of PRs joined in `xadj`: either kind, the pairs the union check joined (`joint`), and the files and
    ABI lanes a top priority that must rebase first shares (`rebase`, Order 4.)."""
    together = {}
    for u in joint:
        for e in u["with"]:
            together.setdefault((min(e, u["pr"]), max(e, u["pr"])), u["files"])
    out = {}
    for a, near in xadj.items():
        for b in near:
            if a >= b:
                continue
            kinds, x = [], M["pairs"].get((a, b), [])
            if {a, b} & rebase:
                shared = sorted(M["files"][a] & M["files"][b])
                kinds.append(
                    (
                        "shares " + ", ".join(_base(f) for f in shared[:3])
                        if shared
                        else "an ABI lane"
                    )
                    + " with a PR that must rebase first"
                )
            elif x is None:
                kinds.append("not checked yet")
            elif x:
                kinds.append(
                    "textual: "
                    + ", ".join(_base(f) for f in x[:3])
                    + (" ..." if len(x) > 3 else "")
                )
            if (a, b) in xabi and not {a, b} & bound:
                kinds.append("ABI " + ", ".join(xabi[(a, b)]))
            if (a, b) in together:
                kinds.append(
                    "together with the PRs before it in a wave: "
                    + ", ".join(_base(f) for f in together[(a, b)][:3])
                )
            out[(a, b)] = "; ".join(kinds) or "conflict"
    return out


def _planner_facts(
    g: dict,
    cls: dict,
    lay: dict,
    prio: list,
    edges: dict,
    care: dict,
    others: list,
    union: bool,
) -> dict:
    """The plan's facts, and its rules as the planner's hard constraints (rkmon_lib/planner.py; waves counted from 0):
    the priorities in their order (2.), the top priorities that must rebase first, each at the end of its wave (4.), the
    hubs (5.), and each PR admitted to the wave the rules place it in. A wave lists its PRs that merge clean with
    develop, then the ones that rebase first."""
    from . import planner

    rep_of, members, size, virtual = g["rep_of"], g["members"], g["size"], g["rebase"]
    ms = sorted(rep_of)
    waves = [
        w + sorted((p for p in virtual if lay["pos"][p] == i), key=lay["rank"].get)
        for i, w in enumerate(lay["waves"], 1)
    ]
    return planner.payload(
        prs=[
            {"pr": m, "cls": cls[m], "weight": WEIGHT[cls[m]], "size": size.get(m, 0)}
            for m in ms
        ],
        edges=edges,
        prio=[m for n in prio for m in members[n]],
        groups=[x for x in members.values() if len(x) > 1],
        hubs=[m for m in ms if rep_of[m] in lay["hubs"]],
        admitted={m: i for i, w in enumerate(waves) for m in w},
        baseline=waves,
        rebase=sorted(virtual),
        care=care,
        union=union,
        notes={
            "PRs that conflict with develop (they land last, after a rebase; not yours to place)": others
        },
    )


def _proposed_layout(g: dict, lay: dict, waves: list[list[int]]) -> dict:
    """A planner proposal that passed its check (rkmon_lib/planner.py), as _layout's result: its waves in its order
    (the PRs that merge clean with develop), each node's wave (1 first), the ranks, and what may land now; the hubs stay
    the toolkit's (a fact, not a choice)."""
    rep_of, xadj, virtual = g["rep_of"], g["xadj"], g["rebase"]
    pos = {rep_of[m]: i for i, w in enumerate(waves, 1) for m in w}
    rank = {m: i for i, m in enumerate(m for w in waves for m in w)}
    go = {
        m
        for m, i in rank.items()
        if pos[rep_of[m]] == 1
        and m not in virtual
        and not any(rank[x] < i for x in xadj[m])
    }
    return {
        "waves": [[m for m in w if m not in virtual] for w in waves],
        "hubs": lay["hubs"],
        "pos": pos,
        "rank": rank,
        "go": go,
    }


def build(
    cfg: dict, st: dict, rows: list[dict], M: dict, at: str | None = None
) -> dict:
    """The plan from the matrix (module docstring). Sets each row's `plan_rank` (landing order; None when unplanned),
    `plan_go` (may land now: feeds the batches), and, for planned PRs, `disrupts` (✗ conflict, Ⓐ ABI conflict, ◆ care)
    and `disruption` (its weighted rebase cost)."""
    at = at or now()
    folder = paths.repository_folder(cfg)
    old = st.get("merge_plan") or {}
    rowmap = {r["n"]: r for r in rows}
    size = {n: rowmap[n]["size"] for n in M["scope"]}
    cls = {n: readiness(st, rowmap[n]) for n in M["scope"]}
    clean = sorted(M["clean"])
    padj = {n: set() for n in clean}  # PR to PR: textual conflicts on develop
    for (a, b), x in M["pairs"].items():
        if x is None or x:
            padj[a].add(b)
            padj[b].add(a)
    # PR to PR: two claims in one ABI lane (aadj, shown), and the ones that order (oadj): a PR the hold rule keeps
    # waiting never blocks its lane, as the rule exempts ABI overlaps (queue.hold_bound: the next ready lane member may
    # take the number), so it renumbers after the others.
    bound = {n for n in M["scope"] if hold_bound(rowmap[n])}
    xabi, claims = abi_conflicts(st, clean)
    aadj, oadj = {n: set() for n in clean}, {n: set() for n in clean}
    for a, b in xabi:
        aadj[a].add(b)
        aadj[b].add(a)
        if not {a, b} & bound:
            oadj[a].add(b)
            oadj[b].add(a)
    xadj = {
        n: padj[n] | oadj[n] for n in clean
    }  # either kind: they order the plan and what may land now

    def real(
        a: int, b: int
    ) -> bool:  # a known conflict that is not only on the ABI counters (makes hubs)
        x = conflicts(M, a, b)
        return bool(x) and not abi_only(cfg, x)

    # A pinned group (`rk group`) is one node: the group's first member stands for it, in the user's order.
    members, adj = {n: [n] for n in clean}, {n: set(v) for n, v in xadj.items()}
    for grp in st.get("groups") or []:
        live = [n for n in grp if n in members and members[n] == [n]]
        if len(live) < 2:
            continue
        rep = live[0]
        for m in live[1:]:
            for x in adj.pop(m):
                adj[x].discard(m)
                if x not in live:
                    adj[x].add(rep)
                    adj[rep].add(x)
            members[rep] += members.pop(m)
        adj[rep] -= set(live)
    # A top priority that must rebase first is a node too (Order 4.): it conflicts with the PRs that share a changed
    # file with it or hold another claim in one of its ABI lanes, the others that rebase first included.
    rebase = [n for n in M["dev_conflicts"] if cls[n] == "T0"]
    for p in rebase:
        members[p], xadj[p], adj[p] = [p], set(), set()
    rep_of = {m: rep for rep, ms in members.items() for m in ms}
    for p in rebase:
        for x in _rivals(M, claims, bound, p, clean + rebase):
            xadj[p].add(x)
            xadj[x].add(p)
            adj[p].add(rep_of[x])
            adj[rep_of[x]].add(p)
    g = {
        "members": members,
        "adj": adj,
        "xadj": xadj,
        "size": size,
        "rep_of": rep_of,
        "rebase": set(rebase),
        "radj": {
            n: {
                m
                for m in adj[n]
                if any(real(a, b) for a in members[n] for b in members[m])
            }
            for n in members
        },
        "gw": {rep: sum(WEIGHT[cls[m]] for m in ms) for rep, ms in members.items()},
        "gcls": {
            rep: min((cls[m] for m in ms), key=list(WEIGHT).index)
            for rep, ms in members.items()
        },
    }
    groups = [rep for rep, ms in members.items() if len(ms) > 1]
    tier0 = _priority_order(
        cfg, st, [n for n in members if g["gcls"][n] == "T0" and n not in groups], size
    )
    held = [
        n
        for n in members
        if n not in groups + tier0 and any(rowmap[m].get("held") for m in members[n])
    ]
    forced = groups + tier0 + held
    first_rebase = [n for n in tier0 if n in g["rebase"]]  # in priority order
    # The union check (module docstring): with the matrix's merges at hand (a matrix built without git has none).
    cache, t0 = st.setdefault(CACHE, {}), time.monotonic()
    uold, ukept, joint, short_waves = cache.get("unions") or {}, {}, [], [0]

    def lay_out(prio: list) -> dict:
        """_layout, until every wave's union merges clean: a member that conflicts with the union of those before it
        conflicts with the ones that change those files from then on (a new edge each time, so it ends).
        """
        while True:
            lay = _layout(g, prio)
            found, short_waves[0] = (
                unions(M, lay["waves"], g["rep_of"], uold, ukept)
                if M.get("merge")
                else ([], 0)
            )
            if not found:
                return lay
            for u in found:
                joint.append(u)
                for e in u["with"]:
                    xadj[u["pr"]].add(e)
                    xadj[e].add(u["pr"])
                    a, b = g["rep_of"][u["pr"]], g["rep_of"][e]
                    if a != b:
                        adj[a].add(b)
                        adj[b].add(a)

    lay = lay_out(forced)
    # The starvation cap: a ready PR the plan keeps waiting lands anyway after STARVE_H hours (at its head).
    waiting, prev = {}, old.get("held_back") or {}
    for n in members:
        if (
            g["gcls"][n] == "R0"
            and n not in forced
            and not all(m in lay["go"] for m in members[n])
        ):
            p = prev.get(str(n))
            head = M["heads"][n][:12]
            waiting[str(n)] = (
                p if p and p.get("head") == head else {"head": head, "since": at}
            )
    starving = sorted(
        (
            int(k)
            for k, v in waiting.items()
            if minutes_since(v["since"], at) >= STARVE_H * 60
        ),
        key=lambda n: waiting[str(n)]["since"],
    )
    if starving:
        lay = lay_out(forced + starving)
    F = M.get("facts") or {}
    care = {}
    for i, a in enumerate(clean):
        for b in clean[i + 1 :]:
            reasons = care_reasons(F[a], F[b], folder) if a in F and b in F else []
            if reasons:
                care[(a, b)] = reasons
    # The planner (rkmon_lib/planner.py, an optional model role): its proposal for
    # these facts is the plan when it keeps every rule above (module docstring); otherwise the waves above stand, and
    # the plan says why. The facts go to the tick, which asks for the next proposal.
    made_by, planner_use, PLANNER["payload"] = "toolkit", "", None
    if roles.on(cfg, "planner"):
        try:
            from . import planner

            others = sorted(
                (n for n in M["dev_conflicts"] if n not in first_rebase),
                key=lambda n: (-WEIGHT[cls[n]], n),
            )
            PLANNER["payload"] = pf = _planner_facts(
                g,
                cls,
                lay,
                forced + starving,
                _edges(M, xadj, xabi, bound, joint, g["rebase"]),
                care,
                others,
                bool(M.get("merge")),
            )

            def union(
                wave: list[int],
            ) -> str:  # the proposal's waves, as the union check checks the toolkit's
                found, short_ = unions(M, [wave], g["rep_of"], uold, ukept)
                if found:
                    u = found[0]
                    return (
                        f"#{u['pr']} conflicts with {', '.join(f'#{x}' for x in u['with'])} together in "
                        f"{some(sorted({_base(f) for f in u['files']}))}"
                    )
                return (
                    "not checked to the end yet: the merge budget ran out"
                    if short_
                    else ""
                )

            proposal, planner_use = planner.use(
                st, pf, union if M.get("merge") else None
            )
            if proposal:
                lay, made_by = _proposed_layout(g, lay, proposal), "planner"
        except (
            Exception
        ) as exc:  # noqa: BLE001 - the planner never stops the plan: the toolkit's own stands
            planner_use = f"the planner's proposal was not used: {str(exc)[:160]}"
    if M.get("merge"):
        cache["unions"] = ukept
    # A top priority that conflicts with develop ranks at the end of its wave, so its ABI lane, which follows the plan,
    # puts it where it lands (never behind wave PRs in its lane). The
    # other PRs that conflict with develop rank behind everything that merges clean now.
    rank = dict(lay["rank"])
    for m in sorted(
        (n for n in M["dev_conflicts"] if n not in first_rebase),
        key=lambda n: (-WEIGHT[cls[n]], n),
    ):
        rank[m] = len(rank)  # after a rebase: behind everything that merges clean now
    why = {
        str(m): (
            "pinned group"
            if n in groups
            else (
                why_first(st, m)
                if n in tier0
                else (
                    "held: it keeps its place"
                    if n in held
                    else "ready, kept waiting a day"
                )
            )
        )
        for n in forced + starving
        for m in members[n]
    }
    why.update({str(n): why_first(st, n) for n in first_rebase})
    # `waves`: every wave's PRs that merge clean with develop, wave 1 first; `first_rebase`: the top priorities that
    # rebase first, in priority order, each at the end of its wave (`wave` in dev_conflicts).
    # `unions`: the PRs whose union conflicts (each alone merges clean), and the files (the union check).
    plan = {
        "v": 2,
        "at": at,
        "develop": M["develop"],
        "scope": M["scope"],
        "waves": lay["waves"],
        "first_rebase": first_rebase,
        "why": why,
        "hubs": sorted((m for h in lay["hubs"] for m in members[h]), key=rank.get),
        "groups": {str(rep): ms for rep, ms in members.items() if len(ms) > 1},
        "unions": [
            {
                "prs": sorted(u["with"] + [u["pr"]], key=rank.get),
                "files": [short(f, folder) for f in u["files"]],
            }
            for u in joint
        ],
        "nodes": {},
        "edges": {},
        "advice": {},
        "dev_conflicts": {},
        "unknown": M["unknown"],
        "partial": {
            "pairs": sum(1 for x in M["pairs"].values() if x is None),
            "prs": len(M["unknown"]),
            "waves": 0 if made_by == "planner" else short_waves[0],
        },
        "merges": (M.get("budget") or {}).get("done", M["merges"]),
        "secs": round(M["secs"] + (time.monotonic() - t0 if M.get("merge") else 0), 2),
        "held_back": waiting,
        "jump_told": old.get("jump_told") or {},
        "ready_told": old.get("ready_told") or {},
        "made_by": made_by,
        "planner": planner_use,
    }
    if not any(plan["partial"].values()):
        plan["partial"] = None
    for a, b in sorted(
        set(care) | set(xabi) | {k for k, x in M["pairs"].items() if x is None or x}
    ):
        e, x = {}, M["pairs"].get((a, b), [])
        if x is None or x:  # for the eye: rocke/... paths (None: not checked yet)
            e["x"] = None if x is None else [short(f, folder) for f in x]
        if (a, b) in xabi:
            e["abi"] = xabi[(a, b)]
        if (a, b) in care:
            e["care"] = care[(a, b)]
        plan["edges"][f"{a}-{b}"] = e
    for m in clean:
        rep = g["rep_of"][m]
        care_only = sorted(
            (
                x
                for x in clean
                if (min(m, x), max(m, x)) in care and x not in padj[m] | aadj[m]
            ),
            key=rank.get,
        )
        later = sorted((x for x in xadj[m] if rank[x] > rank[m]), key=rank.get)
        plan["nodes"][str(m)] = {
            "cls": cls[m],
            "w": WEIGHT[cls[m]],
            "wave": lay["pos"][rep],
            "rank": rank[m],
            "go": m in lay["go"],
            "conflicts": sorted(padj[m], key=rank.get),
            "abi": sorted(aadj[m], key=rank.get),
            "care": care_only,
            "forces": later,
            "cost": sum(WEIGHT[cls[x]] for x in later),
            "care_cost": sum(WEIGHT[cls[x]] for x in care_only),
            "hub": rep in lay["hubs"],
            "hub_degree": len(g["radj"][rep]),
            # the top priorities ahead of it that rebase first, and it waits for (Order 4.)
            "rebase_ahead": sorted(
                (x for x in xadj[m] if x in g["rebase"] and rank[x] < rank[m]),
                key=rank.get,
            ),
        }
    for m in M["dev_conflicts"]:
        shares = sorted(
            (
                (x, len(M["files"][x] & M["files"][m]))
                for x in M["files"]
                if x != m and M["files"][x] & M["files"][m]
            ),
            key=lambda t: (-t[1], rank.get(t[0], 10**6)),
        )
        plan["dev_conflicts"][str(m)] = {
            "cls": cls[m],
            "files": [_base(f) for f in M["dev_conflicts"][m]],
            "shares": [list(t) for t in shares[:6]],
            "rank": rank[m],
            "wave": lay["pos"][m] if m in g["rebase"] else len(lay["waves"]) + 1,
        }
    # Each ABI lane's planned claims in plan order, the ones the hold rule keeps waiting last, as the lanes follow it
    # (queue.abi_lanes), with develop's number.
    dev, plan["abi"] = st["develop"].get("abi") or {}, {}
    for lane, cl in claims.items():
        cl = [
            dict(
                c, members=sorted((m for m in c["members"] if m in rank), key=rank.get)
            )
            for c in cl
        ]
        cl = sorted(
            (c for c in cl if c["members"]),
            key=lambda c: (
                all(m in bound for m in c["members"]),
                rank[c["members"][0]],
            ),
        )
        if cl:
            plan["abi"][lane] = {
                "develop": dev.get(lane) if dev.get("sha") == M["develop"] else None,
                "claims": cl,
            }
    for n in M["scope"]:
        plan["advice"][str(n)] = advice(plan, n, M)
    for r in rows:
        r["plan_rank"], r["plan_go"] = rank.get(r["n"]), r["n"] in lay["go"]
        node = plan["nodes"].get(str(r["n"]))
        if node:
            r["disrupts"] = (
                [f"#{x}✗" for x in node["conflicts"]]
                + [f"#{x}Ⓐ" for x in node["abi"]]
                + [f"#{x}◆" for x in node["care"]]
            )
            r["disruption"] = node["cost"]
    return plan


def tagged(plan: dict, n: int) -> str:
    why = plan["why"].get(str(n))
    return f"#{n} ({why.split(':')[0]})" if why else f"#{n}"


def advice(plan: dict, n: int, M: dict) -> str:
    """Rebase advice for PR n, as dashboard text (text only, no notice), with its ABI
    step (abi_step) when it claims an ABI number."""
    k = str(n)
    step, abi_wait = abi_step(plan, n)
    if k in plan["dev_conflicts"]:
        d = plan["dev_conflicts"][k]
        where = f"conflicts with develop in {some(d['files'])}"
        if n in (
            plan.get("first_rebase") or []
        ):  # Order 4.: in a later wave, once the waves before it have landed
            w = d.get("wave", 1)
            text = (
                f"rebase now: it ends wave 1 ({where})"
                if w == 1
                else f"rebase after wave {w - 1} lands: it ends wave {w} ({where})"
            )
        else:
            text = (
                f"rebase now: you're next ({where})"
                if d["cls"] in ("R0", "R1")
                else f"rebase now ({where})"
            )
        most = ", ".join(f"#{x} ({c})" for x, c in d["shares"][:3])
        return (
            text
            + (f"; after rebase it shares the most files with {most}" if most else "")
            + (f"; {step}" if step else "")
        )
    node = plan["nodes"].get(k)
    if not node:
        return (
            "not planned yet: its changed files are not known"
            if n not in M["files"]
            else "not planned yet: its merge with develop is not checked yet (partial plan)"
        )
    lead = f"{plan['why'][k]}: " if plan["why"].get(k) else ""  # a priority (Order 2.)
    ahead = [
        x for x in node["conflicts"] if plan["nodes"][str(x)]["rank"] < node["rank"]
    ]
    shared = [
        x
        for x in node.get("rebase_ahead") or []
        if M["files"].get(x, set()) & M["files"].get(n, set())
    ]
    rank = {
        x: plan["nodes"][str(x)]["rank"]
        for u in plan.get("unions") or []
        for x in u["prs"]
    }
    joint = [
        u
        for u in plan.get("unions") or []
        if n in u["prs"] and max(u["prs"], key=rank.get) == n
    ]
    together = ""
    if (
        joint
    ):  # the union check moved it: the PRs ahead of it conflict with it together, each alone merges clean
        others = [x for x in joint[0]["prs"] if x != n]
        together = (
            f"conflicts with {' + '.join(tagged(plan, x) for x in others)} together in "
            f"{some(sorted({_base(f) for f in joint[0]['files']}))} (each alone merges clean)"
        )
    if ahead:
        files = sorted({_base(f) for x in ahead[:2] for f in conflicts(M, n, x) or []})
        why = (
            f"conflicts in {some(files)}"
            if files
            else "not checked yet: counted as a conflict"
        )
        names = ", ".join(tagged(plan, x) for x in ahead[:2]) + (
            f" (+{len(ahead) - 2})" if len(ahead) > 2 else ""
        )
        text = (
            f"wait: {names} land{'s' if len(ahead) == 1 else ''} first ({why}), then rebase"
            + (f"; {together}" if together else "")
        )
    elif together:
        after = (
            max(plan["nodes"][str(x)]["wave"] for x in joint[0]["prs"] if x != n) + 1
        )
        text = together + (
            ": next wave" if node["wave"] == after else f": wave {node['wave']}"
        )
    elif (
        shared
    ):  # it shares files with a top priority ahead that rebases first (Order 4.)
        files = sorted(
            {_base(f) for x in shared[:2] for f in M["files"][n] & M["files"][x]}
        )
        names = ", ".join(tagged(plan, x) for x in shared[:2]) + (
            f" (+{len(shared) - 2})" if len(shared) > 2 else ""
        )
        one = len(shared) == 1
        text = (
            f"wait: {names} land{'s' if one else ''} first, after {'its' if one else 'their'} rebase (shares "
            f"{some(files)} with {'it' if one else 'them'})"
        )
    elif abi_wait:
        text = ""  # only an ABI claim ahead: the step says what it waits for
    elif node["wave"] == 1:
        text = (
            "no rebase needed"
            if lead
            else (
                "next: no rebase needed (merges clean with develop and every PR ahead)"
                if node["cls"] == "R0"
                else "no rebase needed: merges clean with develop and every PR ahead"
            )
        )
    else:
        text = f"no rebase needed: nothing ahead of it conflicts (wave {node['wave']})"
    if step:
        text = f"{text}; {step}" if text else step
    if node["forces"] and (
        lead or plan["waves"][node["wave"] - 1][-1] == n
    ):  # a priority, or the end of its wave
        text += (
            f"; its landing makes {some(['#%d' % x for x in node['forces']], 4)} rebase"
        )
    if node["hub"]:
        text += f"; hub ({node['hub_degree']} conflicts outside the ABI lane): it lands after the others"
    return lead + text


def prepare(cfg: dict, st: dict, rows: list[dict], develop: str) -> dict | None:
    """The matrix and its care facts, for build_queue; None when git could not merge a single near-merge PR with
    develop here, or the plan failed. The queue then orders and batches as before (head-to-head pairs, `d`), and the
    board says why. The plan is advice: no failure of it may stop the queue."""
    try:
        M = matrix(cfg, st, rows, develop)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        unavailable(st, f"the merge plan failed: {exc}")
        return None
    if M is None:
        unavailable(st, "no near-merge PR could be merged with develop here")
        return None
    try:
        M["facts"] = facts(cfg, st, M)
    except Exception as exc:  # noqa: BLE001 - care is shown, never acted on
        M["facts"], M["facts_error"] = {}, str(exc)[:160]
    write_pair_cache(st, M)
    return M


def make(cfg: dict, st: dict, rows: list[dict], M: dict) -> dict | None:
    """build(), or None (with the reason on the board) when it fails: the queue then orders as before."""
    try:
        return build(cfg, st, rows, M)
    except Exception as exc:  # noqa: BLE001 - see prepare()
        unavailable(st, f"the merge plan failed: {exc}")
        for r in rows:
            r.pop("plan_rank", None)
            r.pop("plan_go", None)
        return None


def finish(cfg: dict, st: dict, plan: dict, ready=frozenset()) -> None:
    """After the ABI lanes: what auto-merge may land early, the top priorities ready before the waves ahead of them
    (`ready`: the PRs at the Ready stage; ready_early), then the fingerprint; stores st["merge_plan"].
    """
    try:
        plan["automerge"] = automerge_notes(cfg, st, plan)
    except Exception as exc:  # noqa: BLE001 - see prepare()
        plan["automerge"], plan["automerge_error"] = {}, str(exc)[:160]
    try:
        plan["ready_early"] = ready_early(st, plan, set(ready))
    except Exception as exc:  # noqa: BLE001 - see prepare()
        plan["ready_early"], plan["ready_early_error"] = {}, str(exc)[:160]
    plan["fp"] = hashlib.sha1(
        json.dumps(
            [
                plan["waves"],
                plan["first_rebase"],
                plan["hubs"],
                sorted(plan["advice"].items()),
                sorted((k, a["note"]) for k, a in plan["automerge"].items()),
                bool(plan["partial"]),
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()[:12]
    st["merge_plan"] = plan


def unavailable(st: dict, why: str) -> None:
    """No plan this tick: the queue orders and batches as before (head-to-head pairs, `d`)."""
    old = st.get("merge_plan") or {}
    st["merge_plan"] = {
        "v": 2,
        "at": now(),
        "unavailable": why[:200],
        "held_back": old.get("held_back") or {},
        "jump_told": old.get("jump_told") or {},
        "ready_told": old.get("ready_told") or {},
    }


# --- auto-merge in the plan ------------------------------------------------------------------------------------------


def automerge_notes(cfg: dict, st: dict, plan: dict) -> dict:
    """Auto-merge someone enabled on a planned PR: it may land when CI turns green,
    whatever the plan says, so the plan names the rebases it forces and the ABI number it takes. An alert when it would
    take an ABI number ahead of an earlier lane member or make an R0-R1 PR rebase; a top priority or a pinned group is
    expected first (its rebases and ABI number are still named)."""
    armed = ((st.get("queue_state") or {}).get("armed") or {}).get("pr")
    out = {}
    for n in plan["scope"]:
        k, rec = str(n), st["prs"].get(str(n)) or {}
        am = rec.get("auto_merge")
        if not am or n == armed:
            continue
        by = am.get("by") or "?"
        if k in plan["dev_conflicts"]:
            out[k] = {
                "note": f"auto-merge on ({by}), but it conflicts with develop: GitHub merges it only after a rebase",
                "alert": False,
                "key": "",
            }
            continue
        node = plan["nodes"].get(k)
        if not node:
            continue
        claims = []
        c = rec.get("abi_cache") or {}
        if c.get("head") == rec.get("head"):
            claims = [
                f"{lane} ABI {cl['to']}"
                for lane, cl in sorted((c.get("lanes") or {}).items())
            ] + [
                f"{f['name']} {f.get('new')}"
                for f in c.get("families") or []
                if f.get("kind") == "changed"
            ]
        jumps = []
        for lane, L in sorted((st.get("abi_lanes") or {}).items()):
            pos = next(
                (i for i, grp in enumerate(L["groups"]) if n in grp["members"]), None
            )
            if pos:  # an earlier lane member exists: the plan gives it the number first
                jumps.append(f"#{L['groups'][0]['root']} in the {lane} lane")
        near = [x for x in node["conflicts"] if plan_cls(plan, x) in ("T0", "R0", "R1")]
        expected = plan["why"].get(k) in (
            "your priority",
            "fixes develop",
            "pinned group",
        )
        when = (
            "as soon as GitHub allows"
            if rec.get("ci_rollup") == "SUCCESS"
            else "when CI turns green"
        )
        note = (
            f"auto-merge on ({by}): may land {when}"
            + (
                f"; forces rebases on {some([f'#{x} ({plan_cls(plan, x)})' for x in node['conflicts']], 6)}"
                if node["conflicts"]
                else "; forces no rebase"
            )
            + (f"; takes {', '.join(claims)}" if claims else "")
            + (f" ahead of {', '.join(jumps)}" if jumps and claims else "")
            + (f"; expected first ({plan['why'][k]})" if expected else "")
        )
        out[k] = {
            "note": note,
            "alert": not expected and bool(jumps or near),
            "key": f"{rec.get('head', '')[:12]}:{near}:{jumps}",
        }
    return out


def plan_cls(plan: dict, n: int) -> str:
    return (plan["nodes"].get(str(n)) or plan["dev_conflicts"].get(str(n)) or {}).get(
        "cls", "?"
    )


def plan_rank(plan: dict, n: int) -> int:
    """PR n's landing rank, a PR that conflicts with develop included (a top priority that rebases first can be in a
    PR's `forces`)."""
    return (plan["nodes"].get(str(n)) or plan["dev_conflicts"].get(str(n)) or {}).get(
        "rank", 10**6
    )


def automerge_lines(st: dict) -> list[str]:
    """AUTOMERGE_JUMP lines for the tick, once per PR head and finding: auto-merge would land a PR ahead of the plan."""
    plan = st.get("merge_plan") or {}
    notes = plan.get("automerge") or {}
    told = plan.setdefault("jump_told", {})
    lines = []
    for k, a in sorted(notes.items(), key=lambda kv: int(kv[0])):
        if a.get("alert") and told.get(k) != a["key"]:
            told[k] = a["key"]
            lines.append(
                f"AUTOMERGE_JUMP #{k}: {a['note']}. It would land ahead of the merge plan: tell the user "
                "(their call: ask the author to wait, or let it land)"
            )
    for k in [k for k in told if not (notes.get(k) or {}).get("alert")]:
        del told[k]
    return lines


# --- a top priority ready before the waves ahead of it ----------------------------------------------------------------


def ready_early(st: dict, plan: dict, ready: set) -> dict:
    """The top priorities (the user's or develop fixes; on their own or in a pinned group) ready to merge now in a wave
    after the first while a PR in an earlier wave is not: a heads-up, so the user may merge them at their discretion and
    the plan is computed again. `ready`: the PRs whose
    queue row passes every readiness gate (G1-G4, G8, G9: the Ready stage); T0 hides that in `cls`. A top priority that
    must rebase first is never ready. {str(n): {"wave", "why", "waits" (the earlier PRs not ready), "rebase" (the
    earlier PRs that conflict with it, or share a file with it and rebase first: landing it now makes them rebase),
    "renumber" (the earlier ABI claims it would take the number from), "key": "<head>:<wave>"}}. It changes no order
    and does nothing outward: the next tick plans again after the user merges it."""
    fixes, rebasing = st.get("develop_fixes") or {}, set(plan.get("first_rebase") or [])
    wave = {int(k): v["wave"] for k, v in plan["nodes"].items()} | {
        n: _wave(plan, n) for n in rebasing
    }
    out = {}
    for k, node in plan["nodes"].items():
        n, w = int(k), node["wave"]
        if w < 2 or not fixes.get(k) or n not in ready:
            continue
        waits = sorted(
            (m for m, x in wave.items() if x < w and (m in rebasing or m not in ready)),
            key=lambda m: plan_rank(plan, m),
        )
        if not waits:
            continue
        rebase = sorted(
            {
                m
                for m in node["conflicts"] + (node.get("rebase_ahead") or [])
                if wave.get(m, 99) < w
            },
            key=lambda m: plan_rank(plan, m),
        )
        out[k] = {
            "wave": w,
            "why": why_first(st, n),
            "waits": waits,
            "rebase": rebase,
            "renumber": [
                m for m in node["abi"] if wave.get(m, 99) < w and m not in rebase
            ],
            "key": f"{(st['prs'].get(k) or {}).get('head', '')[:12]}:{w}",
        }
    return out


def ready_effect(a: dict) -> str:
    """What landing a ready_early PR now does to the earlier waves, in words: "makes #x, #y rebase", "makes #x rebase
    and #z renumber its ABI number", "forces no rebase"."""
    rebase = some([f"#{m}" for m in a["rebase"]], 4)
    renumber = some([f"#{m}" for m in a["renumber"]], 4)
    if not rebase and not renumber:
        return "forces no rebase"
    return "makes " + " and ".join(
        ([f"{rebase} rebase"] if rebase else [])
        + (
            [
                f"{renumber} renumber {'its' if len(a['renumber']) == 1 else 'their'} ABI "
                f"number{'s' if len(a['renumber']) > 1 else ''}"
            ]
            if renumber
            else []
        )
    )


def ready_line(k: str, a: dict) -> str:
    w = a["wave"]
    return (
        f"PRIORITY_READY_EARLY #{k} (wave {w}, {a['why']}): ready now, but "
        + ("wave 1 still waits" if w == 2 else f"waves 1..{w - 1} still wait")
        + f" on {some(['#%d' % m for m in a['waits']])}; landing it now {ready_effect(a)}; your call"
    )


def ready_need(k: str, a: dict) -> str:
    """The board's Needs-you item while a ready_early PR waits (the HTML page shows the same list)."""
    return (
        f"#{k} ({a['why']}) is ready to merge now but waits in wave {a['wave']} behind "
        f"{some(['#%d' % m for m in a['waits']])}. Landing it now {ready_effect(a)}. Your call: merge it now, or "
        "let the waves run."
    )


def ready_lines(st: dict) -> list[str]:
    """PRIORITY_READY_EARLY lines for the tick, in prototype and live alike, once per PR, head and wave (`ready_told`, as
    automerge_lines keeps `jump_told`): again after the head or the wave changes, or once it stops being ready and is
    ready again. Nothing while the plan is not computed: what was told stays told."""
    plan = st.get("merge_plan") or {}
    if plan.get("unavailable") or "nodes" not in plan:
        return []
    notes = plan.get("ready_early") or {}
    told = plan.setdefault("ready_told", {})
    lines = []
    for k, a in sorted(notes.items(), key=lambda kv: int(kv[0])):
        if told.get(k) != a["key"]:
            told[k] = a["key"]
            lines.append(ready_line(k, a))
    for k in [k for k in told if k not in notes]:
        del told[k]
    return lines


# --- for the board and the dashboards ----------------------------------------------------------------------------------

# The order in plain words, for both dashboards (one home: the module docstring's Order says it in full).
RULE = (
    "Each wave is a set of PRs with no conflict between any two: they land in its order with no rebase among "
    "them, and wave 1 is what can go in now. The priorities (pinned groups, top priorities, held PRs, ready PRs "
    "kept waiting a day) come first: each joins the earliest wave with no higher priority it conflicts with. Each "
    "wave then takes in the best set of the other PRs that conflict with none of its members and with no priority "
    "still waiting (they land after it); hubs last. A wave opens with the PRs whose landing makes no later PR "
    "rebase and ends with those that do. A top priority that conflicts with develop is placed the same way, by "
    "the files and ABI lanes it shares (the PRs that share them land after it), and ends its wave after its "
    "rebase: in wave 1 it rebases now, in a later wave once the wave before it has landed."
)


def _wave(mp: dict, n: int) -> int:
    node = (mp.get("nodes") or {}).get(str(n))
    if node:
        return node["wave"]
    d = (mp.get("dev_conflicts") or {}).get(str(n))
    if (
        d
    ):  # a plan stored before the top priorities that rebase first had waves of their own put them in wave 1
        return (
            d.get("wave", 1)
            if n in (mp.get("first_rebase") or [])
            else len(mp.get("waves") or []) + 1
        )
    return 99


def wave_index(st: dict, n: int) -> int:
    """PR n's wave in the plan, 1 first. A PR that conflicts with develop: its wave for a top priority (it ends that
    wave after its rebase), else one past the last wave. 99 when unplanned."""
    return _wave(st.get("merge_plan") or {}, n)


def place(st: dict, n: int) -> str:
    """PR n's place in the plan, in words: "wave 2", "wave 3, after its rebase" (a top priority that conflicts with
    develop), "after its rebase", or "" when unplanned."""
    mp = st.get("merge_plan") or {}
    w = wave_index(st, n)
    if str(n) in (mp.get("dev_conflicts") or {}):
        return (
            f"wave {w}, after its rebase"
            if n in (mp.get("first_rebase") or [])
            else "after its rebase"
        )
    return "" if w >= 99 else f"wave {w}"


def wave_rows(mp: dict) -> list[tuple[str, list[int], list[int]]]:
    """The dashboards' rows, wave by wave: the label ("wave 2"; "wave 5 (hubs)" when only hubs are in it), its PRs that
    merge clean with develop in landing order, and the top priorities that end it after their rebase.
    """
    nodes, rb = mp.get("nodes") or {}, mp.get("first_rebase") or []
    at = {n: _wave(mp, n) for n in rb}
    ws = list(mp.get("waves") or [])
    ws += [[] for _ in range(max(at.values(), default=0) - len(ws))]
    return [
        (
            f"wave {i}"
            + (
                " (hubs)"
                if w and all((nodes.get(str(n)) or {}).get("hub") for n in w)
                else ""
            ),
            w,
            [n for n in rb if at[n] == i],
        )
        for i, w in enumerate(ws, 1)
    ]


def rebase_last(mp: dict) -> list[int]:
    """The PRs that conflict with develop and rank last ("after a rebase"), in plan order: all but the top priorities
    that end a wave."""
    rb = mp.get("first_rebase") or []
    return [
        int(k)
        for k, _ in sorted(
            (mp.get("dev_conflicts") or {}).items(), key=lambda kv: kv[1]["rank"]
        )
        if int(k) not in rb
    ]


def unchecked(mp: dict) -> str:
    """The waves whose union is not checked to the end yet (the merge budget ran out), in words: "2 waves not
    union-checked"; "" when none."""
    w = (mp.get("partial") or {}).get("waves") or 0
    return f"{w} wave{'s' if w != 1 else ''} not union-checked" if w else ""


def abi_turns(mp: dict) -> str:
    """The ABI lanes the plan serializes, e.g. "ABI in turn: binary ×7" ("" when no lane has two planned claims)."""
    lanes = [
        f"{lane} ×{len(L['claims'])}"
        for lane, L in (mp.get("abi") or {}).items()
        if len(L["claims"]) > 1
    ]
    return f"ABI in turn: {', '.join(lanes)}" if lanes else ""


def made_by_line(mp: dict) -> str:
    """Who made the plan, when the planner is on (rkmon_lib/planner.py): "Made by the planner: its proposal of ...",
    or "Made by the toolkit: <why the planner's proposal is not used>". "" with the planner off: the toolkit's plan, as
    always."""
    use = mp.get("planner") or ""
    return f"Made by the {mp.get('made_by') or 'toolkit'}: {use}." if use else ""


def summary(mp: dict) -> str:
    """One line: wave 1, the top priorities that end a wave after their rebase, the number of waves, the hubs, the ABI
    lanes in turn, who rebases first, and the planner when it made the plan (the tick summary's MERGE PLAN line).
    """
    if mp.get("unavailable"):
        return f"not computed: {mp['unavailable']}"
    rows = wave_rows(mp)
    w1 = rows[0][1] if rows else []
    parts = (
        (
            [
                f"wave 1 {' '.join('#%d' % n for n in w1[:8])}"
                + (" …" if len(w1) > 8 else "")
            ]
            if w1
            else []
        )
        + [
            f"after its rebase{f' in wave {i}' if i > 1 else ''} {' '.join('#%d' % n for n in rb)}"
            for i, (_, _, rb) in enumerate(rows, 1)
            if rb
        ]
        + [f"{len(rows)} wave{'s' if len(rows) != 1 else ''}"]
        + (
            [f"hubs last {' '.join('#%d' % n for n in mp['hubs'])}"]
            if mp.get("hubs")
            else []
        )
        + ([abi_turns(mp)] if abi_turns(mp) else [])
        + (
            [f"{len(mp['dev_conflicts'])} rebase first"]
            if mp.get("dev_conflicts")
            else []
        )
        + (["partial"] if mp.get("partial") else [])
        + (["made by the planner"] if mp.get("made_by") == "planner" else [])
    )
    return "; ".join(parts)


@idle.timer("starvation cap")
def _starve_due(cfg: dict, st: dict, store):
    """When each ready PR the plan keeps waiting reaches STARVE_H hours and lands anyway."""
    for k, v in ((st.get("merge_plan") or {}).get("held_back") or {}).items():
        if v.get("since"):
            yield f"starvation cap #{k}", idle.plus(v["since"], STARVE_H * 60)
