# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Assign test cases to levels (quick, standard, comprehensive, full).

Reads the per-value labels, `level_labels` and the `levels` recipe from model.yaml and writes one
CSV with a `level` column plus a summary. Levels are cumulative: each level runs every case of
the levels before it, and a case belongs to the first level that needs it.

Per level, in recipe order:
  1. Restrict every axis to the values whose labels the level accepts (the shared base value is
     always accepted); with `answerable`, also forbid the `answer_key_needs` combinations.
  2. Pairwise, warm-started: rows from lower levels already cover some of the level's pairs; new
     rows are generated only for the pairs still uncovered. With `max_changes` the rows are
     low-change (lowchange.py), counted from the shared base. `max_changes: N` caps every row at N
     changes; a pair that no such row can hold is not asked for. `max_changes: {start: N,
     escalate: true}` covers the pairs N changes can hold with rows of at most N changes, then
     the pairs that need N + 1 with rows of at most N + 1, and so on until every pair is covered;
     each pair is covered by a row with no more changes than the pair needs. At each step only
     rows within that step's limit count as covering. Without `max_changes` the rows are packed
     (modes.pairwise). With `super_setting: {id, axes}` the listed axes form one setting: each
     complete combination of their values is one value of it, counts as one change, and is paired
     with every value of every other setting; pairs between the listed axes are not asked for.
  3. One-at-a-time rows from the shared base, targeted cases and real-world cases, when the recipe
     asks for them. Targeted and real-world cases skip the level's labels. Seeds (`seeds`) are
     completed with the fewest changes; one whose row the references can check joins the
     real-world cases, any other one runs in the last level. Seeds are written as origin
     real_world, detail seed:<id>.
  4. Every row is expanded over the level's shape tiers, subject to the level's per-case ceiling.
     A level can restrict its own new cases (the cases it does not keep move up one level):
       larger_tiers: size_sensitive   a case at a tier other than S is kept only if its row changes,
                                      from the shared base, a setting whose `size_sensitive` label
                                      is "yes".
       tiers_per_row: one             each row keeps exactly one of the tiers left to it. Rows are
                                      taken in a fixed order (row level, origin, smallest case key);
                                      each gets the tier given out least so far, ties broken by
                                      cycling on from the tier given last.
       tiers_per_value: [T, ...]      afterwards, every value that some kept-or-allowed case of the
                                      level or a lower level holds at tier T still has a case at T
                                      (cumulatively); a missing one is restored by switching a row's
                                      kept tier, when that loses no other value its last case at a
                                      listed tier.
  5. When a level's cumulative time exceeds its budget: with `level_labels.over_budget: move` its
     non-S cases move up one level, largest tier and cost first; with `report` nothing moves and
     the summary says what would have moved.
  6. Fill: each budgeted level takes cases from higher levels, in `level_labels.fill.priority`
     order, until its cumulative time plus the per-run overhead reaches headroom x budget. A
     filled case obeys the level's `larger_tiers` and `tiers_per_row` too.

A level's `size_sweep` labels are checked against SIZE_SWEEP_LABELS. This generator emits no
size-sweep rows, and it does not read the `numeric_domains` and `size_sweep` sections.

Run: python3 -m sdpa_case_flow_poc.levels --model-dir DIR --out-dir DIR [--seed N]
         [--timings ARCH=CSV ...]
A timings CSV has columns case_key,seconds,source (source: measured, measured-stops-early or
modelled). `level_labels.cost.basis` names the arch the budgets use; every arch in the cost
model gets a cost column. Cases a file does
not name are costed by the arch's model in `level_labels.cost`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import modes
from .lowchange import LowChange, Projection, change_count, lowchange_pairwise
from .model import Model, ModelError, Rule, load_model, load_yaml
from .pairs import candidate_pairs, feasible_pairs
from .solver import Solver

BASE = "shared"
TIER_ORDER = ("S", "H", "M", "L", "X")
ORIGIN_ORDER = ("low-change", "pairwise", "one-at-a-time", "targeted", "real_world")
HEAD_DIM = {"d1": 1, "d64": 64, "d128": 128, "d192": 192, "d256": 256}
SIZE_SWEEP_LABELS = (
    "typical",
    "corner_plain",
    "corner_crossed",
    "corner_crossed_at_L",
    "stress",
)
OUT_CSV = "levels.csv"
OUT_SUMMARY = "levels-summary.md"


@dataclass
class SettingRow:
    row: dict
    origin: str
    detail: str
    level: int
    # Targeted, real-world and stress rows run at one tier only; other rows take the tiers of each level.
    fixed_tier: str | None = None
    sizes: dict | None = None
    # A targeted row that is also a later level's pairwise or one-at-a-time row takes that
    # origin's tiers from that level on.
    expand_as: str | None = None
    expand_from: int | None = None


@dataclass
class Case:
    srow: SettingRow
    tier: str
    level: int
    mflop: float
    seconds: float
    cost_source: str
    sizes: dict
    tensor_mb: float = 0.0
    arch_seconds: dict = field(default_factory=dict)
    pushed_from: list = field(default_factory=list)
    filled_from: str = ""


@dataclass
class Recipe:
    levels: list
    labels: dict
    groups: dict
    bridges: list
    answer_rules: list
    tiers: dict
    targeted: list
    real_world: list
    cost: dict
    fill: dict | None
    alibi_heads: dict = field(default_factory=dict)
    sensitive: dict = field(default_factory=dict)
    over_budget: str = "move"
    seeds: list = field(default_factory=list)


def _as_list(x):
    return x if isinstance(x, list) else [x]


def super_groups(model: Model, spec: dict) -> dict:
    """{group id: axes} from a level's `super_setting`, or {}."""
    sup = spec.get("super_setting")
    if not sup:
        return {}
    if set(sup) != {"id", "axes"} or not sup["axes"]:
        raise ModelError(
            f"level {spec['name']}: super_setting needs exactly `id` and a non-empty `axes`"
        )
    for a in sup["axes"]:
        if a not in model.values:
            raise ModelError(
                f"level {spec['name']}: super_setting names unknown axis {a}"
            )
    if sup["id"] in model.values:
        raise ModelError(
            f"level {spec['name']}: super_setting id {sup['id']} is an axis id"
        )
    return {sup["id"]: list(sup["axes"])}


def link_rules(groups: dict) -> list:
    """Rules that forbid nothing but put each group's axes in one solver component."""
    return [
        Rule(
            f"LINK-{g}",
            "level",
            "link",
            ({a: frozenset() for a in axes},),
            "model.yaml:levels",
        )
        for g, axes in groups.items()
    ]


def change_limit(spec: dict):
    """(start, escalate) from a level's `max_changes`, or None for packed pairwise."""
    lim = spec.get("max_changes")
    if lim is None:
        return None
    if isinstance(lim, dict):
        unknown = set(lim) - {"start", "escalate"}
        if unknown or "start" not in lim:
            raise ModelError(
                f"level {spec['name']}: max_changes needs `start` and allows only `escalate`"
            )
        return int(lim["start"]), bool(lim.get("escalate", False))
    return int(lim), False


def load_recipe(model: Model, model_dir: Path) -> Recipe:
    doc = load_yaml(Path(model_dir) / "model.yaml")
    if "levels" not in doc or "level_labels" not in doc:
        raise ModelError("model.yaml has no `levels` recipe or no `level_labels`")
    labels, groups, sensitive = {}, {}, {}
    for a in doc["axes"]:
        lab = a.get("labels")
        if lab is None or "group" not in a:
            raise ModelError(f"axis {a['id']} has no labels or no group")
        ss = a.get("size_sensitive")
        if not (
            isinstance(ss, list)
            and len(ss) == 2
            and str(ss[0]) in ("yes", "no")
            and str(ss[1]).strip()
        ):
            raise ModelError(
                f"axis {a['id']}: size_sensitive must be [\"yes\" or \"no\", reason]"
            )
        sensitive[a["id"]] = str(ss[0]) == "yes"
        missing = set(model.values[a["id"]]) - set(lab)
        if missing:
            raise ModelError(f"axis {a['id']}: no labels for {sorted(missing)}")
        labels[a["id"]] = {v: tuple(str(x) for x in t) for v, t in lab.items()}
        groups[a["id"]] = a["group"]
    ll = doc["level_labels"]
    for b in ll["bridges"]:
        for ax in b["a"] + b["b"]:
            if ax not in model.values:
                raise ModelError(f"bridge {b['id']}: unknown axis {ax}")
    answer_rules = []
    for r in ll.get("answer_key_needs") or []:
        conjs = ([r["forbid"]] if "forbid" in r else []) + list(
            r.get("forbid_any") or []
        )
        parsed = []
        for c in conjs:
            for ax, vs in c.items():
                if ax not in model.values or any(v not in model.values[ax] for v in vs):
                    raise ModelError(
                        f"answer_key_needs {r['id']}: unknown axis or value in {c}"
                    )
            parsed.append({ax: frozenset(vs) for ax, vs in c.items()})
        answer_rules.append(
            Rule(
                r["id"],
                "answer_key",
                "exclusion",
                tuple(parsed),
                "model.yaml:level_labels",
            )
        )
    for kind in ("targeted", "real_world", "seeds"):
        for t in ll.get(kind) or []:
            for ax, v in t["changes"].items():
                if ax not in model.values or v not in model.values[ax]:
                    raise ModelError(f"{kind} {t['id']}: unknown {ax}={v}")
    for spec in doc["levels"]:
        super_groups(model, spec)
        if spec.get("tiers_per_row") not in (None, "one"):
            raise ModelError(f"level {spec['name']}: tiers_per_row must be `one`")
        if spec.get("larger_tiers") not in (None, "size_sensitive"):
            raise ModelError(
                f"level {spec['name']}: larger_tiers must be `size_sensitive`"
            )
        if any(t not in spec["shapes"] for t in spec.get("tiers_per_value") or []):
            raise ModelError(
                f"level {spec['name']}: tiers_per_value names a tier outside its shapes"
            )
        sweep = spec.get("size_sweep", [])
        if not isinstance(sweep, list) or any(
            x not in SIZE_SWEEP_LABELS for x in sweep
        ):
            raise ModelError(
                f"level {spec['name']}: size_sweep must list labels from {SIZE_SWEEP_LABELS}"
            )
    over_budget = ll.get("over_budget", "move")
    if over_budget not in ("move", "report"):
        raise ModelError("level_labels.over_budget must be `move` or `report`")
    ctv = doc.get("class_to_value") or {}
    alibi = {}
    if "alibi_head_count" in model.values:
        replaced = ctv["alibi_head_count"]["not_power_of_two"]["H_q_by_power_of_two"]
        table = ctv["head_counts"]["with_alibi_not_power_of_two"]
        for src, hq in replaced.items():
            alibi[int(src)] = {
                "H_q": int(hq),
                "mha": int(table["mha"][hq]),
                "gqa": int(table["gqa"][hq]),
                "mqa": int(table["mqa"][hq]),
                "differ": (
                    int(table["k_and_v_differ"][hq]["H_k"]),
                    int(table["k_and_v_differ"][hq]["H_v"]),
                ),
            }
    return Recipe(
        levels=doc["levels"],
        labels=labels,
        groups=groups,
        bridges=ll["bridges"],
        answer_rules=answer_rules,
        tiers=ll["shape_tiers"],
        targeted=ll["targeted"],
        real_world=ll.get("real_world") or [],
        cost=ll["cost"],
        fill=ll.get("fill"),
        alibi_heads=alibi,
        sensitive=sensitive,
        over_budget=over_budget,
        seeds=ll.get("seeds") or [],
    )


def accepts(spec: dict, label: tuple, ignore=()) -> bool:
    usage, numbers, key = label
    for name, got in (
        ("usage", usage),
        ("changes_numbers", numbers),
        ("answer_key", key),
    ):
        want = spec.get(name, "any")
        if (
            name not in ignore
            and want != "any"
            and got not in [str(x) for x in _as_list(want)]
        ):
            return False
    return True


def accepted_values(model: Model, recipe: Recipe, spec: dict) -> dict:
    """Map each axis to the values the level's labels accept (the shared base value always).

    A child setting must be stated whenever its parent switches it on, so when no value of a
    child passes `changes_numbers`, that label is not applied to the child: a required
    declaration that cannot change the numbers never holds its parent back.
    """
    base = model.base_cases[BASE]
    out = {}
    for a in model.axes:
        ok = {
            v
            for v in model.values[a]
            if v == base[a] or accepts(spec, recipe.labels[a][v])
        }
        if a in model.parent and not ok:
            ok = {
                v
                for v in model.values[a]
                if accepts(spec, recipe.labels[a][v], ignore=("changes_numbers",))
            }
        out[a] = ok
    return out


def level_rules(model: Model, recipe: Recipe, spec: dict) -> tuple:
    """Return (extra exclusion rules, disallowed features) for one level."""
    base = model.base_cases[BASE]
    rules, banned = [], set()
    ok = accepted_values(model, recipe, spec)
    for a in model.axes:
        banned.update((a, v) for v in model.values[a] if v not in ok[a])
    # A parent value that would activate a child with no accepted value is held back too; left to
    # the solver, proving such a value impossible is an exponential search.
    changed = True
    while changed:
        changed = False
        for child, (pax, pvals) in model.parent.items():
            if all((child, v) in banned for v in model.values[child]):
                new = {(pax, p) for p in pvals if p != base[pax]} - banned
                if new:
                    banned |= new
                    changed = True
    for a in model.axes:
        bad = [v for v in model.values[a] if (a, v) in banned]
        if bad:
            rules.append(
                Rule(
                    f"LEVEL-{spec['name']}-{a}",
                    "level",
                    "exclusion",
                    ({a: frozenset(bad)},),
                    "model.yaml:levels",
                )
            )
    if spec.get("answerable"):
        rules.extend(recipe.answer_rules)
    rules.extend(link_rules(super_groups(model, spec)))
    return rules, banned


def in_scope_fn(recipe: Recipe, scope: str):
    if scope == "all":
        return lambda a, b: True
    if scope != "groups_and_bridges":
        raise ModelError(f"unknown pairs scope {scope}")
    linked = set()
    for br in recipe.bridges:
        for x in br["a"]:
            for y in br["b"]:
                if x != y:
                    linked.add(frozenset((x, y)))
    g = recipe.groups
    return lambda a, b: g[a] == g[b] or frozenset((a, b)) in linked


def level_pairs(
    model: Model, solver: Solver, recipe: Recipe, spec: dict, banned: set
) -> list:
    scope = in_scope_fn(recipe, spec["pairs"])
    groups = super_groups(model, spec)
    if groups:
        return projected_pairs(model, solver, recipe, spec, banned, groups, scope)
    cands = [
        (f, g)
        for f, g in candidate_pairs(model)
        if f not in banned and g not in banned and scope(f[0], g[0])
    ]
    return feasible_pairs(solver, cands)


def group_values(model: Model, solver: Solver, banned: set, axes: list) -> list:
    """Every combination of the axes' values (None for an inactive child) that some valid row holds."""
    doms = []
    for a in axes:
        d = [v for v in model.values[a] if (a, v) not in banned]
        if a in model.parent:
            d.append(None)
        doms.append(d)
    return [
        combo
        for combo in itertools.product(*doms)
        if solver.feasible(dict(zip(axes, combo)))
    ]


def projected_pairs(
    model: Model,
    solver: Solver,
    recipe: Recipe,
    spec: dict,
    banned: set,
    groups: dict,
    scope,
) -> list:
    """Valid pairs of projected features: plain axes outside the groups, and each group as one."""
    if spec["pairs"] != "all":
        raise ModelError(f"level {spec['name']}: super_setting needs `pairs: all`")
    proj = Projection(model, groups)
    feats = []
    for key in proj.keys:
        if key in groups:
            feats += [
                (key, combo)
                for combo in group_values(model, solver, banned, groups[key])
            ]
        else:
            feats += [(key, v) for v in model.values[key] if (key, v) not in banned]
    single = {}

    def ok(f):
        if f not in single:
            single[f] = solver.feasible(proj.assignment(f))
        return single[f]

    out = []
    for i, f in enumerate(feats):
        if not ok(f):
            continue
        for g in feats[i + 1 :]:
            if g[0] == f[0] or not ok(g):
                continue
            if solver.feasible({**proj.assignment(f), **proj.assignment(g)}):
                out.append(proj.make_pair(f, g))
    return out


def row_pairs(model: Model, row: dict):
    feats = [(a, row[a]) for a in model.axes if row[a] is not None]
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            yield feats[i], feats[j]


def row_key(model: Model, row: dict) -> tuple:
    return tuple(row[a] for a in model.axes)


# ------------------------------------------------------------------ sizes and cost
def concrete_sizes(row: dict, tier: dict, alibi_heads: dict | None = None) -> dict:
    many_b, many_h, s, s_short = (
        tier["B"],
        tier["H_q"],
        tier["S"],
        tier.get("S_short", tier["S"] // 2),
    )
    b = 1 if row["batch_size"] == "one" else many_b
    hq = 1 if row["q_heads"] == "one" else many_h
    alibi = None
    if row.get("alibi_head_count") == "not_power_of_two":
        alibi = (alibi_heads or {}).get(hq)
        if alibi is None:
            raise ModelError(
                f"alibi_head_count not_power_of_two: no head count replaces H_q {hq}"
            )
        hq = alibi["H_q"]
    q_one, kv_one = row["seq_q"] == "one", row["seq_kv"] == "one"
    rel = row["seq_q_vs_seq_kv"]
    if q_one and kv_one:
        sq, skv = 1, 1
    elif q_one:
        sq, skv = 1, s
    elif kv_one:
        sq, skv = s, 1
    else:
        sq, skv = {"shorter": (s_short, s), "equal": (s, s), "longer": (s, s_short)}[
            rel
        ]

    def heads(sharing):
        return {"mha": hq, "gqa": max(2, hq // 4), "mqa": 1}.get(sharing, hq)

    hk, hv = heads(row["k_head_sharing"]), heads(row["v_head_sharing"])
    if (
        row["k_heads_vs_v_heads"] == "differ"
        and row["k_head_sharing"] == "gqa"
        and row["v_head_sharing"] == "gqa"
    ):
        hv = 2 * hk
    if alibi is not None:
        hk, hv = alibi[row["k_head_sharing"]], alibi[row["v_head_sharing"]]
        if (
            row["k_heads_vs_v_heads"] == "differ"
            and row["k_head_sharing"] == "gqa"
            and row["v_head_sharing"] == "gqa"
        ):
            hk, hv = alibi["differ"]
    return {
        "B": b,
        "H_q": hq,
        "H_k": hk,
        "H_v": hv,
        "S_q": sq,
        "S_kv": skv,
        "D_qk": HEAD_DIM[row["head_dim_qk"]],
        "D_v": HEAD_DIM[row["head_dim_v"]],
    }


def work_flops(z: dict) -> float:
    return 2.0 * z["B"] * z["H_q"] * z["S_q"] * z["S_kv"] * (z["D_qk"] + z["D_v"])


DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def tensor_mb(row: dict, z: dict) -> float:
    """Approximate bytes of Q, K, V, O, Stats and bias, in MiB."""
    q = DTYPE_BYTES.get(row["dtype_q"], 1)
    o = DTYPE_BYTES.get(row["dtype_o"], q)
    n = z["B"] * (
        z["H_q"] * z["S_q"] * z["D_qk"] * q
        + z["H_k"] * z["S_kv"] * z["D_qk"] * q
        + z["H_v"] * z["S_kv"] * z["D_v"] * q
        + z["H_q"] * z["S_q"] * (z["D_v"] * o + 4)
    )
    if row["bias"] == "present":
        n += z["B"] * z["H_q"] * z["S_q"] * z["S_kv"] * 4
    return n / 2**20


def modelled_seconds(cost: dict, arch: str, mflop: float, mb: float) -> float:
    m = cost["model"][arch]
    return max(float(m["a"]) + float(m["b"]) * mflop + float(m["c"]) * mb, 0.001)


def in_fitted_range(cost: dict, c) -> bool:
    r = cost["fitted_range"]
    return c.mflop <= float(r["max_mflop"]) and c.tensor_mb <= float(r["max_tensor_mb"])


def model_cost(cost: dict, c) -> None:
    """Cost every arch of a case from the fitted model."""
    for arch in cost["model"]:
        c.arch_seconds[arch] = modelled_seconds(cost, arch, c.mflop, c.tensor_mb)
    c.seconds = c.arch_seconds[cost["basis"]]
    c.cost_source = "modelled" if in_fitted_range(cost, c) else "UNMEASURED"


def fmt_sizes(z: dict) -> str:
    return (
        f"B{z['B']} Hq{z['H_q']} Hk{z['H_k']} Hv{z['H_v']} Sq{z['S_q']} Skv{z['S_kv']} "
        f"Dqk{z['D_qk']} Dv{z['D_v']}"
    )


def seed_completions(model: Model, recipe: Recipe) -> list:
    """Return (id, row, checkable) per seed: its fewest-change row under the rules, the answer-key rules
    and the labels every answerable level shares, else under the rules alone (checkable False).
    """
    if not recipe.seeds:
        return []
    base = model.base_cases[BASE]
    answerable = [s for s in recipe.levels if s.get("answerable")]

    def union(name):
        wants = [s.get(name, "any") for s in answerable]
        return (
            "any"
            if "any" in wants
            else sorted({str(x) for w in wants for x in _as_list(w)})
        )

    spec = {
        "name": "seeds",
        "answerable": True,
        "usage": union("usage"),
        "answer_key": union("answer_key"),
    }
    extra, _ = level_rules(model, recipe, spec)
    checked = LowChange(
        model, Solver(model, rules=list(model.rules) + extra), base, len(model.axes)
    )
    plain = LowChange(model, Solver(model), base, len(model.axes))
    out = []
    for t in recipe.seeds:
        fixed = dict(t["changes"])
        got, checkable = checked.complete(fixed), True
        if got is None:
            got, checkable = plain.complete(fixed), False
        if got is None:
            raise ModelError(f"seed {t['id']}: no valid row holds its changes")
        out.append((t["id"], got[0], checkable))
    return out


# ------------------------------------------------------------------ assignment
def assign(model: Model, recipe: Recipe, seed: int, candidates: int = 2):
    base_solver = Solver(model)
    base = model.base_cases[BASE]
    srows: list = []
    by_key: dict = {}
    covered: set = set()
    stats = []
    notes = []
    targeted_done = real_world_done = False
    seed_rows = seed_completions(model, recipe)

    def add_srow(row, origin, detail, li, fixed_tier=None, sizes=None):
        k = (
            row_key(model, row),
            fixed_tier if fixed_tier == "X" else None,
            None if sizes is None else tuple(sorted(sizes.items())),
        )
        if k in by_key:
            prev = by_key[k]
            note = f"{origin}:{detail}" if detail else origin
            if note != prev.origin and note not in prev.detail.split("; "):
                prev.detail = "; ".join(x for x in (prev.detail, note) if x)
            if (
                prev.fixed_tier == "S"
                and prev.expand_from is None
                and fixed_tier is None
                and sizes is None
            ):
                prev.expand_as, prev.expand_from = origin, li
            return False
        sr = SettingRow(dict(row), origin, detail, li, fixed_tier, sizes)
        srows.append(sr)
        by_key[k] = sr
        covered.update(row_pairs(model, row))
        return True

    for li, spec in enumerate(recipe.levels):
        extra, banned = level_rules(model, recipe, spec)
        solver = Solver(model, rules=list(model.rules) + extra)
        pairs = level_pairs(model, solver, recipe, spec, banned)
        groups = super_groups(model, spec)
        proj = Projection(model, groups)
        rng = random.Random(f"{seed}:{spec['name']}")
        limit = change_limit(spec)
        beyond = 0
        steps = []
        if limit is None:
            remaining = [p for p in pairs if p not in covered]
            pw = (
                modes.pairwise(model, solver, remaining, rng, candidates=candidates)
                if remaining
                else []
            )
            n_pw = sum(add_srow(r, "pairwise", "", li) for _, r in pw)
            have = covered
            most = 0
        else:
            start, escalate = limit
            lc = LowChange(model, solver, base, start, groups)
            pending = list(pairs)
            asked, have, remaining, pw = [], set(), [], []
            n_pw = most = 0
            k = start
            while pending and k <= len(model.axes):
                lc.max_changes = k
                step_pairs = [
                    p
                    for p in pending
                    if lc.feasible({**proj.assignment(p[0]), **proj.assignment(p[1])})
                ]
                step_set = set(step_pairs)
                pending = [p for p in pending if p not in step_set]
                within = set()
                for sr in srows:
                    if proj.count(base, sr.row) <= k:
                        within.update(proj.pairs(sr.row))
                todo = [p for p in step_pairs if p not in within]
                have.update(p for p in step_pairs if p in within)
                rows = (
                    lowchange_pairwise(model, lc, todo, rng, projection=proj)
                    if todo
                    else []
                )
                new_rows = 0
                for meta, r in rows:
                    if add_srow(r, "low-change", f"{meta['changes']} changes", li):
                        new_rows += 1
                        most = max(most, meta["changes"])
                        have.update(p for p in proj.pairs(r) if p in step_set)
                if step_pairs:
                    steps.append(
                        {
                            "changes": k,
                            "pairs": len(step_pairs),
                            "warm_covered": len(step_pairs) - len(todo),
                            "new_rows": new_rows,
                        }
                    )
                asked += step_pairs
                remaining += todo
                pw += rows
                n_pw += new_rows
                if not escalate:
                    break
                k += 1
            beyond = len(pending)
            if escalate and pending:
                notes.append(
                    f"{spec['name']}: {len(pending)} valid pairs fit in no row within {len(model.axes)} changes"
                )
            pairs = asked
        n_oat = 0
        if spec.get("one_at_a_time"):
            rows, _ = modes.one_at_a_time(model, solver, [BASE])
            for meta, r in rows:
                n_oat += add_srow(r, "one-at-a-time", meta["varied"], li)
        n_tg = 0
        if spec.get("targeted") and not targeted_done:
            targeted_done = True
            for t in recipe.targeted:
                r = base_solver.complete(dict(t["changes"]), prefs=base)
                if r is None:
                    notes.append(
                        f"targeted {t['id']}: no valid row holds its changes; skipped"
                    )
                    continue
                n_tg += add_srow(
                    r, "targeted", t["id"], li, t.get("tier", "S"), t.get("sizes")
                )
        n_rw = 0
        if spec.get("real_world") and not real_world_done:
            real_world_done = True
            for t in recipe.real_world:
                r = base_solver.complete(dict(t["changes"]), prefs=base)
                if r is None:
                    notes.append(
                        f"real_world {t['id']}: no valid row holds its changes; skipped"
                    )
                    continue
                if any(r[a] != v for a, v in t["changes"].items()):
                    raise AssertionError(
                        f"real_world {t['id']}: the completed row changed a requested value"
                    )
                n_rw += add_srow(r, "real_world", t["id"], li, "S")
            for sid, r, checkable in seed_rows:
                if checkable:
                    n_rw += add_srow(r, "real_world", f"seed:{sid}", li, "S")
        if li == len(recipe.levels) - 1:
            for sid, r, checkable in seed_rows:
                if not checkable:
                    n_rw += add_srow(r, "real_world", f"seed:{sid}", li, "S")
        if spec.get("stress_tier") and "base" in spec.get("stress_rows", []):
            add_srow(base, "one-at-a-time", "- (stress)", li, spec["stress_tier"])
        missing = [p for p in pairs if p not in have]
        if missing:
            raise AssertionError(
                f"level {spec['name']}: {len(missing)} pairs uncovered"
            )
        stats.append(
            {
                "level": spec["name"],
                "pairs": len(pairs),
                "warm_covered": len(pairs) - len(remaining),
                "super_values": {
                    g: len({f[1] for p in pairs for f in p if f[0] == g})
                    for g in groups
                },
                "max_changes": limit,
                "pairs_beyond_limit": beyond,
                "steps": steps,
                "max_changes_used": most,
                "new_pairwise_rows": n_pw,
                "new_oat_rows": n_oat,
                "new_targeted_rows": n_tg,
                "new_real_world_rows": n_rw,
                "banned_values": len(banned),
            }
        )

    cases = []
    seen = set()
    for li, spec in enumerate(recipe.levels):
        ceiling = spec.get("ceiling_mflop")
        for sr in srows:
            if sr.level > li:
                continue
            kind = (
                sr.expand_as
                if sr.expand_from is not None and li >= sr.expand_from
                else sr.origin
            )
            if sr.fixed_tier is not None and kind == sr.origin:
                tiers = [sr.fixed_tier]
            elif kind == "one-at-a-time":
                tiers = [
                    t
                    for t in spec.get("oat_shapes", spec["shapes"])
                    if t in spec["shapes"]
                ]
            else:
                tiers = list(spec["shapes"])
            for tier in tiers:
                if (id(sr), tier) in seen:
                    continue
                tdef = dict(recipe.tiers[tier])
                if sr.sizes:
                    tdef.update(sr.sizes)
                z = concrete_sizes(sr.row, tdef, recipe.alibi_heads)
                mflop = work_flops(z) / 1e6
                if tier != "S" and ceiling is not None and mflop > ceiling:
                    continue
                seen.add((id(sr), tier))
                c = Case(sr, tier, li, mflop, 0.0, "", z, tensor_mb(sr.row, z))
                model_cost(recipe.cost, c)
                cases.append(c)
    for sr in srows:
        if sr.fixed_tier is not None and (id(sr), sr.fixed_tier) not in seen:
            raise AssertionError(
                f"{sr.origin} {sr.detail}: its tier {sr.fixed_tier} fits no level"
            )
    for li, trim in restrict_tiers(model, recipe, cases).items():
        stats[li]["tier_trim"] = trim
    return srows, cases, stats, notes


def size_sensitive(model: Model, recipe: Recipe, row: dict) -> bool:
    """True when the row changes, from the shared base, a setting labelled size_sensitive yes."""
    base = model.base_cases[BASE]
    return any(
        recipe.sensitive[a]
        for a in model.axes
        if row[a] is not None and row[a] != base.get(a)
    )


def tier_allowed(model: Model, recipe: Recipe, spec: dict, c: Case) -> bool:
    return (
        c.tier == "S"
        or spec.get("larger_tiers") != "size_sensitive"
        or size_sensitive(model, recipe, c.srow.row)
    )


def row_values(model: Model, row: dict) -> set:
    return {(a, row[a]) for a in model.axes if row[a] is not None}


def restrict_tiers(model: Model, recipe: Recipe, cases: list) -> dict:
    """Apply each level's larger_tiers, tiers_per_row and tiers_per_value; return {level: report}."""
    out = {}
    for li, spec in enumerate(recipe.levels):
        if not (spec.get("tiers_per_row") or spec.get("larger_tiers")):
            continue
        if li + 1 >= len(recipe.levels):
            raise ModelError(
                f"level {spec['name']}: the last level cannot hand cases up"
            )
        name = spec["name"]
        mine = [c for c in cases if c.level == li]
        allowed = [c for c in mine if tier_allowed(model, recipe, spec, c)]
        by_row: dict = {}
        for c in mine:
            by_row.setdefault(id(c.srow), []).append(c)
        if spec.get("tiers_per_row") == "one":
            tiers = [t for t in TIER_ORDER if t in spec["shapes"]]
            keys = {id(c): case_key(model, c) for c in mine}
            order = sorted(
                by_row,
                key=lambda r: (
                    by_row[r][0].srow.level,
                    ORIGIN_ORDER.index(by_row[r][0].srow.origin),
                    min(keys[id(c)] for c in by_row[r]),
                ),
            )
            ok = {id(c) for c in allowed}
            given, last, kept = Counter(), -1, {}
            for r in order:
                avail = {c.tier: c for c in by_row[r] if id(c) in ok}
                if not avail:
                    continue
                t = min(
                    avail,
                    key=lambda t: (given[t], (tiers.index(t) - last - 1) % len(tiers)),
                )
                given[t] += 1
                last = tiers.index(t)
                kept[r] = avail[t]
        else:
            kept = None
        switched, unresolved = [], []
        if kept is not None and spec.get("tiers_per_value"):
            need = list(spec["tiers_per_value"])
            lower = [c for c in cases if c.level < li]
            held = Counter()
            for c in lower + list(kept.values()):
                if c.tier in need:
                    held.update((v, c.tier) for v in row_values(model, c.srow.row))
            for t in need:
                wanted = set()
                for c in lower + allowed:
                    if c.tier == t:
                        wanted |= row_values(model, c.srow.row)
                for v in sorted(wanted):
                    if held[(v, t)]:
                        continue
                    cands = sorted(
                        (
                            c
                            for c in allowed
                            if c.tier == t and v in row_values(model, c.srow.row)
                        ),
                        key=lambda c: case_key(model, c),
                    )
                    for c in cands:
                        old = kept.get(id(c.srow))
                        if (
                            old is not None
                            and old.tier in need
                            and any(
                                held[(w, old.tier)] < 2
                                for w in row_values(model, old.srow.row)
                            )
                        ):
                            continue
                        if old is not None and old.tier in need:
                            held.subtract(
                                (w, old.tier) for w in row_values(model, old.srow.row)
                            )
                        held.update((w, t) for w in row_values(model, c.srow.row))
                        kept[id(c.srow)] = c
                        switched.append((v, t, None if old is None else old.tier))
                        break
                    else:
                        unresolved.append((v, t))
        keep_ids = {id(c) for c in (kept.values() if kept is not None else allowed)}
        allowed_ids = {id(c) for c in allowed}
        moved = Counter()
        for c in mine:
            if id(c) in keep_ids:
                continue
            reason = (
                "one tier per row" if id(c) in allowed_ids else "not size-sensitive"
            )
            moved[(reason, c.tier)] += 1
            c.pushed_from.append(f"{name}: {reason}")
            c.level = li + 1
        out[li] = {
            "rows": len(by_row),
            "rows_kept": len({id(c.srow) for c in mine if id(c) in keep_ids}),
            "kept_tiers": dict(Counter(c.tier for c in mine if id(c) in keep_ids)),
            "moved": dict(moved),
            "switched": switched,
            "unresolved": unresolved,
        }
    return out


def case_key(model: Model, c: Case) -> str:
    """Content key of a case (tier, sizes and settings); stable when case ids shift."""
    text = "|".join(
        [
            c.tier,
            fmt_sizes(c.sizes),
            *("-" if c.srow.row[a] is None else c.srow.row[a] for a in model.axes),
        ]
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def apply_timings(model: Model, recipe: Recipe, cases, files: list) -> dict:
    """Take each arch's seconds from its timing file; return per-arch counts of cases found."""
    cost = recipe.cost
    by_key = {case_key(model, c): c for c in cases}
    found = {}
    for arch, path in files:
        if arch not in cost["model"]:
            raise ModelError(
                f"--timings {arch}: no cost model for this arch in level_labels.cost"
            )
        n = 0
        with open(path, encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                c = by_key.get(rec["case_key"])
                if c is None:
                    continue
                n += 1
                src, sec = rec.get("source", "measured"), float(rec["seconds"])
                if (
                    src == "measured-stops-early"
                    and cost.get("stops_early_budget") == "modelled"
                ):
                    sec = max(sec, modelled_seconds(cost, arch, c.mflop, c.tensor_mb))
                c.arch_seconds[arch] = sec
                if arch == cost["basis"]:
                    c.seconds = sec
                    if src == "modelled":
                        c.cost_source = (
                            "modelled" if in_fitted_range(cost, c) else "UNMEASURED"
                        )
                    else:
                        c.cost_source = src
        found[arch] = n
    return found


def fill_budgets(recipe: Recipe, cases: list) -> list:
    """Move non-S cases up one level while a level's cumulative time exceeds its budget."""
    report = []
    n = len(recipe.levels)
    for li, spec in enumerate(recipe.levels):
        budget = spec.get("budget_min")
        total = sum(c.seconds for c in cases if c.level <= li)
        entry = {
            "level": spec["name"],
            "budget_s": None if budget is None else budget * 60.0,
            "moved_up": 0,
            "over_after": 0.0,
        }
        if budget is not None and total > budget * 60.0 and li + 1 < n:
            movable = [c for c in cases if c.level == li and c.tier != "S"]
            movable.sort(key=lambda c: (-TIER_ORDER.index(c.tier), -c.seconds))
            report_only = recipe.over_budget == "report"
            entry["over_before"] = total - budget * 60.0
            would, left = [], total
            for c in movable:
                if left <= budget * 60.0:
                    break
                would.append(c)
                left -= c.seconds
            entry["would_move"] = {
                "cases": len(would),
                "seconds": sum(c.seconds for c in would),
                "tiers": dict(Counter(c.tier for c in would)),
                "fits": left <= budget * 60.0,
            }
            if not report_only:
                for c in would:
                    c.pushed_from.append(spec["name"])
                    c.level = li + 1
                    entry["moved_up"] += 1
                total = left
        entry["total_s"] = total
        if budget is not None and total > budget * 60.0:
            entry["over_after"] = total - budget * 60.0
        report.append(entry)
    return report


def priority_key(model: Model, recipe: Recipe, c: Case) -> tuple:
    base = model.base_cases[BASE]
    rare = no_numbers = worst_key = 0
    for a in model.axes:
        v = c.srow.row[a]
        if v is None or v == base[a]:
            continue
        usage, numbers, key = recipe.labels[a][v]
        rare += usage == "rare"
        no_numbers += numbers == "no"
        worst_key = max(worst_key, {"yes": 0, "partial": 1}.get(key, 2))
    if any(
        all(c.srow.row[a] in vs for a, vs in conj.items())
        for r in recipe.answer_rules
        for conj in r.conjs
    ):
        worst_key = 2
    keys = {
        "fewest_rare": rare,
        "best_answer_key": worst_key,
        "fewest_no_numbers": no_numbers,
        "cheapest": c.seconds,
    }
    return tuple(keys[p] for p in recipe.fill["priority"]) + (
        TIER_ORDER.index(c.tier),
        case_key(model, c),
    )


def labels_accept(model: Model, recipe: Recipe, spec: dict, row: dict) -> bool:
    ok = accepted_values(model, recipe, spec)
    for a in model.axes:
        v = row[a]
        if v is not None and v not in ok[a]:
            return False
    if spec.get("answerable"):
        for r in recipe.answer_rules:
            if any(all(row[a] in vs for a, vs in conj.items()) for conj in r.conjs):
                return False
    return True


def fill_down(model: Model, recipe: Recipe, cases: list) -> list:
    """Pull cases from higher levels into each budgeted level, up to its fill target."""
    fill = recipe.fill
    report = []
    if not fill:
        return report
    overhead, headroom = float(fill["overhead_s"]), float(fill["headroom"])
    for li, spec in enumerate(recipe.levels):
        budget = spec.get("budget_min")
        if budget is None:
            continue
        target = headroom * budget * 60.0
        used = overhead + sum(c.seconds for c in cases if c.level <= li)
        ceiling = spec.get("ceiling_mflop")
        one = spec.get("tiers_per_row") == "one"
        rows_here = {id(c.srow) for c in cases if c.level == li}
        pool = [
            c
            for c in cases
            if c.level > li
            and c.tier in spec["shapes"]
            and c.cost_source != "UNMEASURED"
            and (ceiling is None or c.mflop <= ceiling)
            and tier_allowed(model, recipe, spec, c)
            and not (one and id(c.srow) in rows_here)
            and (
                not fill.get("respect_labels")
                or labels_accept(model, recipe, spec, c.srow.row)
            )
            and (
                change_limit(spec) is None
                or change_count(
                    model, model.base_cases[BASE], c.srow.row, super_groups(model, spec)
                )
                <= change_limit(spec)[0]
            )
        ]
        pool.sort(key=lambda c: priority_key(model, recipe, c))
        taken = 0
        for c in pool:
            if used + c.seconds > target or (one and id(c.srow) in rows_here):
                continue
            rows_here.add(id(c.srow))
            c.filled_from = c.filled_from or recipe.levels[c.level]["name"]
            c.level = li
            used += c.seconds
            taken += 1
        report.append(
            {
                "level": spec["name"],
                "target_s": target,
                "taken": taken,
                "pool": len(pool),
            }
        )
    return report


def order_cases(cases):
    return sorted(
        cases,
        key=lambda c: (
            c.level,
            bool(c.filled_from),
            ORIGIN_ORDER.index(c.srow.origin),
            c.srow.level,
            TIER_ORDER.index(c.tier),
        ),
    )


# ------------------------------------------------------------------ output
META = [
    "level",
    "shape_tier",
    "origin",
    "detail",
    "filled_from",
    "est_seconds",
    "cost_source",
    "work_mflop",
    "tensor_mb",
    "sizes",
    "case_key",
]


def alt_archs(recipe: Recipe) -> list:
    return [a for a in recipe.cost["model"] if a != recipe.cost["basis"]]


def render(model: Model, recipe: Recipe, cases) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    alts = alt_archs(recipe)
    w.writerow(["case_id", *META, *(f"est_seconds_{a}" for a in alts), *model.axes])
    for i, c in enumerate(cases, start=1):
        errs = model.structure_errors(c.srow.row)
        broken = model.violated_rules(c.srow.row)
        if errs or broken:
            raise AssertionError(f"case {i}: structure {errs} rules {broken}")
        w.writerow(
            [
                f"lv-{i:04d}",
                recipe.levels[c.level]["name"],
                c.tier,
                c.srow.origin,
                c.srow.detail,
                c.filled_from,
                f"{c.seconds:.4f}",
                c.cost_source,
                f"{c.mflop:.1f}",
                f"{c.tensor_mb:.3f}",
                fmt_sizes(c.sizes),
                case_key(model, c),
                *(f"{c.arch_seconds[a]:.4f}" for a in alts),
                *("-" if c.srow.row[a] is None else c.srow.row[a] for a in model.axes),
            ]
        )
    return buf.getvalue().encode("utf-8")


def summary(
    recipe: Recipe,
    cases,
    stats,
    budget_report,
    fill_report,
    notes,
    seed: int,
    csv_sha: str,
    found: dict,
) -> str:
    names = [s["name"] for s in recipe.levels]
    cost, fill = recipe.cost, recipe.fill or {}
    basis, alts = cost["basis"], alt_archs(recipe)
    archs = [basis, *alts]
    overhead = float(fill.get("overhead_s", 0.0))
    headroom = float(fill.get("headroom", 1.0))
    timing_text = (
        ", ".join(f"{a}: {n} cases found in its timing file" for a, n in found.items())
        or "no timing file"
    )
    out = [
        "# Level assignment summary — DRAFT",
        "",
        f"Generated by `python3 -m sdpa_case_flow_poc.levels`, seed {seed}. `levels.csv` sha256 `{csv_sha}`.",
        f"Costs: {timing_text}; budget basis {basis}. Every other case is costed by the fitted model "
        f"(`level_labels.cost`). Per level run, {overhead:g} s overhead is added; a level is filled to "
        f"{headroom:.0%} of its budget.",
        "",
        "## Cases and time per level (cumulative time includes the per-run overhead)",
        "",
        "| Level | New cases | Of which filled from higher levels | Cumulative cases | "
        + " | ".join(f"Cumulative s ({a})" for a in archs)
        + " | Budget (min) | Budget use ("
        + basis
        + ") | Fill target use ("
        + basis
        + ") |",
        "|---|---|---|---|" + "---|" * len(archs) + "---|---|---|",
    ]
    cum = 0
    for li, name in enumerate(names):
        mine = [c for c in cases if c.level == li]
        cum += len(mine)
        tot = {
            a: overhead + sum(c.arch_seconds[a] for c in cases if c.level <= li)
            for a in archs
        }
        budget = recipe.levels[li].get("budget_min")
        use = "—" if budget is None else f"{tot[basis] / (budget * 60.0):.1%}"
        fuse = (
            "—" if budget is None else f"{tot[basis] / (headroom * budget * 60.0):.1%}"
        )
        out.append(
            f"| {name} | {len(mine)} | {sum(1 for c in mine if c.filled_from)} | {cum} | "
            + " | ".join(f"{tot[a]:.1f}" for a in archs)
            + f" | {'none' if budget is None else budget} | {use} | {fuse} |"
        )
    sources = ("measured", "measured-stops-early", "modelled", "UNMEASURED")
    out += [
        "",
        f"## Where each case's cost comes from ({basis})",
        "",
        "| Level | " + " | ".join(sources) + " |",
        "|---|" + "---|" * len(sources),
    ]
    for li, name in enumerate(names):
        mine = [c for c in cases if c.level == li]
        out.append(
            f"| {name} | "
            + " | ".join(
                str(sum(1 for c in mine if c.cost_source == x)) for x in sources
            )
            + " |"
        )
    out.append(
        f"| all | "
        + " | ".join(str(sum(1 for c in cases if c.cost_source == x)) for x in sources)
        + " |"
    )
    out += [
        "",
        "## Filled cases, by the level the recipe gave them",
        "",
        "| Level | Pool that fits shapes and ceiling | Taken | "
        + " | ".join(f"from {n}" for n in names)
        + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for fr in fill_report:
        li = names.index(fr["level"])
        mine = [c for c in cases if c.level == li and c.filled_from]
        out.append(
            f"| {fr['level']} | {fr['pool']} | {fr['taken']} | "
            + " | ".join(str(sum(1 for c in mine if c.filled_from == n)) for n in names)
            + " |"
        )
    out += [
        "",
        "## New cases per level, by origin and shape tier",
        "",
        "| Level | " + " | ".join(ORIGIN_ORDER) + " | " + " | ".join(TIER_ORDER) + " |",
        "|---|" + "---|" * (len(ORIGIN_ORDER) + len(TIER_ORDER)),
    ]
    for li, name in enumerate(names):
        mine = [c for c in cases if c.level == li]
        o = [sum(1 for c in mine if c.srow.origin == x) for x in ORIGIN_ORDER]
        t = [sum(1 for c in mine if c.tier == x) for x in TIER_ORDER]
        out.append(
            f"| {name} | "
            + " | ".join(map(str, o))
            + " | "
            + " | ".join(map(str, t))
            + " |"
        )
    out += [
        "",
        "## Pairs per level (setting rows, before shape tiers and before filling)",
        "",
        "| Level | Max changes (limit / most in a new row) | Pairs in scope | Valid pairs beyond the limit (not asked) | Already covered by lower levels | New pairwise rows (low-change or packed) | New one-at-a-time rows | New targeted rows | New real-world rows | Values held back |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        if s["max_changes"] is None:
            lim = "packed"
        else:
            start, escalate = s["max_changes"]
            lim = (
                f"{start}{', escalating' if escalate else ''} / {s['max_changes_used']}"
            )
        out.append(
            f"| {s['level']} | {lim} | {s['pairs']} | {s['pairs_beyond_limit']} | {s['warm_covered']} | {s['new_pairwise_rows']} | "
            f"{s['new_oat_rows']} | {s['new_targeted_rows']} | {s['new_real_world_rows']} | {s['banned_values']} |"
        )
    for s in stats:
        for g, n in s.get("super_values", {}).items():
            out.append("")
            out.append(
                f"- {s['level']}: `{g}` is one setting; {n} of its value combinations are paired with "
                "every other value the level allows (each combination counts as one change)."
            )
    if any(s["steps"] for s in stats):
        out += [
            "",
            "## Low-change steps: pairs by the fewest changes a row needs to hold them",
            "",
            "A pair is asked for at the first step whose limit some row can meet; it is covered by a row within "
            "that limit. Counts are per level (pairs in the level's scope), new rows are rows the level adds.",
            "",
            "| Level | Changes | Pairs that need this many | Already covered | New rows |",
            "|---|---|---|---|---|",
        ]
        for s in stats:
            for st in s["steps"]:
                out.append(
                    f"| {s['level']} | {st['changes']} | {st['pairs']} | {st['warm_covered']} | {st['new_rows']} |"
                )
    s_over = [
        c
        for c in cases
        if c.tier == "S"
        and recipe.levels[0].get("ceiling_mflop")
        and c.mflop > recipe.levels[0]["ceiling_mflop"]
        and not c.filled_from
    ]
    unmeasured = [c for c in cases if c.cost_source == "UNMEASURED"]
    out += [
        "",
        "## Notes",
        "",
        f"- S-tier cases above the quick per-case ceiling ({recipe.levels[0].get('ceiling_mflop')} MFLOP), kept because S "
        f"is never removed by a ceiling: {len(s_over)}. Filled cases always respect the ceiling.",
        "- Targeted cases are placed at the first level whose recipe has `targeted: true` and whose ceiling admits "
        "their tier; they use values the level's labels would otherwise hold back.",
        "- Real-world cases run at tier S in the first level whose recipe has `real_world: true`, whatever "
        "that level's labels and ceiling. So do seeds (detail `seed:<id>`) whose row the references can check; "
        "the other seeds run at tier S in the last level.",
        f"- UNMEASURED (outside the fitted range of the cost model): "
        + (
            ", ".join(
                f"{c.tier} {c.srow.origin} {c.srow.detail or ''} {c.mflop:.0f} MFLOP {c.tensor_mb:.0f} MB".replace(
                    "  ", " "
                )
                for c in unmeasured
            )
            or "none"
        )
        + ".",
    ]
    for b in budget_report:
        if b["moved_up"]:
            out.append(
                f"- {b['level']}: {b['moved_up']} cases moved up because the level was over budget."
            )
        elif "would_move" in b:
            w = b["would_move"]
            tiers = ", ".join(
                f"{t} {n}"
                for t, n in sorted(
                    w["tiers"].items(), key=lambda x: TIER_ORDER.index(x[0])
                )
            )
            out.append(
                f"- {b['level']}: OVER BUDGET by {b['over_before']:.1f} s; nothing was moved (`over_budget: report`). "
                f"Moving would take {w['cases']} cases ({tiers}; {w['seconds']:.1f} s) up one level"
                + ("." if w["fits"] else ", and the level would still be over budget.")
            )
    for li, s in enumerate(stats):
        tr = s.get("tier_trim")
        if not tr:
            continue
        kt = ", ".join(
            f"{t} {n}"
            for t, n in sorted(
                tr["kept_tiers"].items(), key=lambda x: TIER_ORDER.index(x[0])
            )
        )
        mv = "; ".join(
            f"{r} {t}: {n}"
            for (r, t), n in sorted(
                tr["moved"].items(), key=lambda x: (x[0][0], TIER_ORDER.index(x[0][1]))
            )
        )
        out.append(
            f"- {s['level']}: tier restriction kept {tr['rows_kept']} of {tr['rows']} rows, one case each ({kt}); "
            f"moved up one level by reason and tier: {mv or 'none'}; rows switched to keep a value at a tier: "
            f"{len(tr['switched'])}; values left without a case at a tier: "
            + (", ".join(f"{a}={v} at {t}" for (a, v), t in tr["unresolved"]) or "none")
            + "."
        )
    out += [f"- {n}" for n in notes]
    return "\n".join(out) + "\n"


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="sdpa_case_flow_poc.levels",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2893)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument(
        "--timings",
        action="append",
        default=[],
        metavar="ARCH=CSV",
        help="per-case seconds for one arch: CSV with columns case_key,seconds,source",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model = load_model(args.model_dir)
    recipe = load_recipe(model, args.model_dir)
    srows, cases, stats, notes = assign(
        model, recipe, args.seed, candidates=args.candidates
    )
    files = []
    for t in args.timings:
        arch, sep, path = t.partition("=")
        if not sep:
            raise ModelError(f"--timings {t}: expected ARCH=CSV")
        files.append((arch, Path(path)))
    found = apply_timings(model, recipe, cases, files)
    budget_report = fill_budgets(recipe, cases)
    fill_report = fill_down(model, recipe, cases)
    cases = order_cases(cases)
    data = render(model, recipe, cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / OUT_CSV).write_bytes(data)
    text = summary(
        recipe,
        cases,
        stats,
        budget_report,
        fill_report,
        notes,
        args.seed,
        hashlib.sha256(data).hexdigest(),
        found,
    )
    (args.out_dir / OUT_SUMMARY).write_text(text, encoding="utf-8")
    for li, s in enumerate(recipe.levels):
        n = sum(1 for c in cases if c.level == li)
        print(
            f"{s['name']}: {n} new cases, {sum(1 for c in cases if c.level <= li)} cumulative"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
