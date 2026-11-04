#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Level checker for levels.csv. It does not import the generator.

For each level of the `levels` recipe in model.yaml:
  values    every pairwise and one-at-a-time case of the level uses only values the level's labels
            accept (the shared base value is always accepted; a child setting none of whose values
            passes `changes_numbers` is not held to that label); targeted cases, real-world cases
            and cases filled down from a higher level (`filled_from` set) are exempt.
  changes   with `max_changes: N`, every case of the level except targeted, real-world,
            size-sweep and coverage cases changes at most N settings from the shared base (a
            setting counts when it is set in both rows with different values). With
            `max_changes: {start: N, escalate: true}`, a filled case changes at most N settings,
            and every other case with more than N changes (those four origins aside, and rows a
            lower level already holds) holds a pair of the level whose fewest possible changes
            equal the case's changes.
  fill      a filled case runs at one of the level's shapes, is within its per-case ceiling and is
            not UNMEASURED; when the level holds filled cases, its cumulative est_seconds plus
            `level_labels.fill.overhead_s` is within headroom x budget.
  coverage  every pair the level asks for (in scope, both values accepted, and some row that
            breaks no rule, no level restriction and, with `answerable`, no answer_key_needs
            combination holds it; with `max_changes: N`, some such row within N changes) appears
            in a case of this level or a lower one (with `max_changes: N`, in a case within N
            changes). With escalation, every such pair is asked for, and the case with the fewest
            changes that holds it changes at most max(N, the fewest changes any valid row needs
            for the pair); the checker proves that bound by its own search.
            With `super_setting: {id, axes}`, the listed axes are one setting: a pair is a value of
            another setting with one complete combination of the listed axes' values, the
            combination counts as one change, and pairs between the listed axes are not asked for.
  tiers     with `larger_tiers: size_sensitive`, every case of the level at a tier other than S
            changes from the shared base a setting whose `size_sensitive` label is "yes". With
            `tiers_per_row: one`, every settings row with a case at the level has exactly one case
            there (size-sweep cases aside: each has its own sizes). With
            `tiers_per_value: [T, ...]`, every value held by a case up to the level has
            a case at T up to the level, unless no case in the file at T could be one of the level's
            own: within its shapes and ceiling, every value accepted by its labels (and, with
            `answerable`, no answer_key_needs combination), and size-sensitive under `larger_tiers`.
Checked once for the file: every axis has `size_sensitive: [yes | no, reason]`; a case holding a
value whose usage label is `fringe` is at a level whose `usage` accepts fringe.
Exit status 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_common import Spec, change_count, read_cases, row_pairs  # noqa: E402


def level_groups(lv: dict) -> dict:
    sup = lv.get("super_setting")
    return {sup["id"]: list(sup["axes"])} if sup else {}


MAX_LISTED = 20
# Origins whose rows the change limit does not apply to.
NOT_PAIRWISE = ("targeted", "real_world", "size-sweep", "coverage")
BASE = "shared"


def change_limit(lv: dict):
    """(start, escalate) from a level's `max_changes`, or None."""
    lim = lv.get("max_changes")
    if lim is None:
        return None
    if isinstance(lim, dict):
        return int(lim["start"]), bool(lim.get("escalate", False))
    return int(lim), False


def wanted(spec: dict, name: str, got: str) -> bool:
    want = spec.get(name, "any")
    if want == "any":
        return True
    want = want if isinstance(want, list) else [want]
    return got in [str(x) for x in want]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("csv", type=Path)
    args = ap.parse_args(argv)

    spec = Spec(args.model_dir)
    with open(args.model_dir / "model.yaml", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    labels = {
        a["id"]: {v: [str(x) for x in t] for v, t in a["labels"].items()}
        for a in doc["axes"]
    }
    group = {a["id"]: a["group"] for a in doc["axes"]}
    ll = doc["level_labels"]
    links = {
        frozenset((x, y))
        for b in ll["bridges"]
        for x in b["a"]
        for y in b["b"]
        if x != y
    }
    answer = []
    for r in ll.get("answer_key_needs") or []:
        for c in ([r["forbid"]] if "forbid" in r else []) + list(
            r.get("forbid_any") or []
        ):
            answer.append((r["id"], {ax: set(vs) for ax, vs in c.items()}))
    base = spec.base_row(BASE)
    names = [lv["name"] for lv in doc["levels"]]

    cases = list(read_cases(args.csv, spec))
    model_ids = set(spec.rule_ids)
    failed = False

    general = []
    sensitive = {}
    for a in doc["axes"]:
        ss = a.get("size_sensitive")
        if not (
            isinstance(ss, list)
            and len(ss) == 2
            and str(ss[0]) in ("yes", "no")
            and str(ss[1]).strip()
        ):
            general.append(f"axis {a['id']}: size_sensitive is not [yes | no, reason]")
            continue
        sensitive[a["id"]] = str(ss[0]) == "yes"
    fringe = {(a, v) for a in labels for v, t in labels[a].items() if t[0] == "fringe"}
    takes_fringe = [lv["name"] for lv in doc["levels"] if wanted(lv, "usage", "fringe")]
    fringe_cases = 0
    for cid, meta, row in cases:
        held = [f"{a}={row[a]}" for a in spec.axes if (a, row[a]) in fringe]
        if held:
            fringe_cases += 1
            if meta["level"] not in takes_fringe:
                general.append(
                    f"{cid}: fringe value {', '.join(held)} at {meta['level']}"
                )
    print(
        ("PASS" if not general else "FAIL")
        + f" labels: {len(sensitive)} of {len(doc['axes'])} axes have size_sensitive "
        f"({sum(sensitive.values())} yes); {len(fringe)} fringe values, in {fringe_cases} cases, "
        f"accepted only at {', '.join(takes_fringe) or 'no level'}"
    )
    for p in general[:MAX_LISTED]:
        print(f"  {p}")
    failed |= bool(general)

    def is_sensitive(row):
        return any(
            sensitive.get(a, True)
            for a in spec.axes
            if row[a] is not None and row[a] != base[a]
        )

    for li, lv in enumerate(doc["levels"]):

        def label_ok(a, v, lv=lv, skip=()):
            return all(
                wanted(lv, n, x)
                for n, x in zip(
                    ("usage", "changes_numbers", "answer_key"), labels[a][v]
                )
                if n not in skip
            )

        relaxed = {
            a
            for a in spec.axes
            if a in spec.parent and not any(label_ok(a, v) for v in spec.values[a])
        }

        def accepted(a, v, relaxed=relaxed, label_ok=label_ok):
            if v == base[a]:
                return True
            return label_ok(a, v, skip=("changes_numbers",) if a in relaxed else ())

        groups = level_groups(lv)
        mine = [
            (cid, meta, row)
            for cid, meta, row in cases
            if names.index(meta["level"]) <= li
        ]
        problems = []
        fill = ll.get("fill") or {}
        filled = [
            (cid, meta)
            for cid, meta, _ in cases
            if meta["level"] == lv["name"] and meta.get("filled_from")
        ]
        for cid, meta in filled:
            if meta["shape_tier"] not in lv["shapes"]:
                problems.append(
                    f"{cid}: filled at tier {meta['shape_tier']}, not a {lv['name']} shape"
                )
            if (
                lv.get("ceiling_mflop") is not None
                and float(meta["work_mflop"]) > lv["ceiling_mflop"]
            ):
                problems.append(
                    f"{cid}: filled with {meta['work_mflop']} MFLOP, above the {lv['name']} ceiling"
                )
            if meta["cost_source"] == "UNMEASURED":
                problems.append(f"{cid}: filled although its cost is UNMEASURED")
        if filled and lv.get("budget_min") is not None:
            total = float(fill.get("overhead_s", 0.0)) + sum(
                float(m["est_seconds"]) for _, m, _ in mine
            )
            target = float(fill.get("headroom", 1.0)) * lv["budget_min"] * 60.0
            if total > target:
                problems.append(
                    f"{lv['name']}: {total:.1f} s after filling, above the fill target {target:.1f} s"
                )
        lim = change_limit(lv)
        limit, escalate = (None, False) if lim is None else lim
        if limit is not None:
            for cid, meta, row in cases:
                if meta["level"] == lv["name"] and meta["origin"] not in NOT_PAIRWISE:
                    n = change_count(spec, base, row, groups)
                    if n > limit and (not escalate or meta.get("filled_from")):
                        problems.append(
                            f"{cid}: changes {n} settings, above the {lv['name']} limit {limit}"
                        )
        for cid, meta, row in cases:
            if (
                meta["level"] != lv["name"]
                or meta["origin"] in ("targeted", "real_world")
                or meta.get("filled_from")
            ):
                continue
            for a in spec.axes:
                if row[a] is not None and not accepted(a, row[a]):
                    problems.append(f"{cid}: {a}={row[a]} is held back at {lv['name']}")

        tier_note = ""
        here = [
            (cid, meta, row) for cid, meta, row in cases if meta["level"] == lv["name"]
        ]
        if lv.get("larger_tiers") == "size_sensitive":
            bad = [
                cid
                for cid, meta, row in here
                if meta["shape_tier"] != "S" and not is_sensitive(row)
            ]
            problems += [
                f"{cid}: tier other than S, but its row changes no size-sensitive setting"
                for cid in bad
            ]
            tier_note += (
                f"; {sum(1 for _, m, _ in here if m['shape_tier'] != 'S')} cases above S, all size-sensitive"
                if not bad
                else ""
            )
        if lv.get("tiers_per_row") == "one":
            per_row = {}
            for cid, meta, row in here:
                if meta["origin"] == "size-sweep":
                    continue
                per_row.setdefault(tuple(row[a] for a in spec.axes), []).append(cid)
            many = [ids for ids in per_row.values() if len(ids) > 1]
            problems += [
                f"{', '.join(ids)}: one settings row with {len(ids)} cases at {lv['name']}"
                for ids in many
            ]
            tier_note += f"; {len(per_row)} rows, one case each" if not many else ""
        if lv.get("tiers_per_value"):

            def admissible(meta, row, lv=lv, accepted=accepted):
                if meta["shape_tier"] not in lv["shapes"]:
                    return False
                if (
                    lv.get("ceiling_mflop") is not None
                    and float(meta["work_mflop"]) > lv["ceiling_mflop"]
                ):
                    return False
                if any(v is not None and not accepted(a, v) for a, v in row.items()):
                    return False
                if lv.get("answerable") and any(
                    all(row[a] in vs for a, vs in c.items()) for _, c in answer
                ):
                    return False
                return (
                    lv.get("larger_tiers") != "size_sensitive"
                    or meta["shape_tier"] == "S"
                    or is_sensitive(row)
                )

            held_any = {
                (a, row[a])
                for _, _, row in mine
                for a in spec.axes
                if row[a] is not None
            }
            for t in lv["tiers_per_value"]:
                have = {
                    (a, row[a])
                    for _, m, row in mine
                    if m["shape_tier"] == t
                    for a in spec.axes
                    if row[a] is not None
                }
                could = {
                    (a, row[a])
                    for _, m, row in cases
                    if m["shape_tier"] == t and admissible(m, row)
                    for a in spec.axes
                    if row[a] is not None
                }
                lost = sorted(held_any & could - have)
                exempt = sorted(held_any - could - have)
                problems += [
                    f"{a}={v}: no case at {t} up to {lv['name']}" for a, v in lost
                ]
                tier_note += (
                    f"; {t}: {len(held_any & have)} of {len(held_any)} values have a case"
                    + (
                        f" ({len(exempt)} exempt, no admissible {t} case: "
                        + ", ".join(f"{a}={v}" for a, v in exempt)
                        + ")"
                        if exempt
                        else ""
                    )
                )

        level_nogoods = []
        for a in spec.axes:
            bad = {v for v in spec.values[a] if not accepted(a, v)}
            if bad:
                level_nogoods.append((f"LEVEL-{a}", {a: bad}))
        if lv.get("answerable"):
            level_nogoods.extend(answer)
        saved = spec.nogoods
        spec.nogoods = saved + level_nogoods
        ids = model_ids | {rid for rid, _ in level_nogoods}

        covered = {}
        most = 0
        for _, meta, row in mine:
            n = change_count(spec, base, row, groups)
            if limit is None or escalate or n <= limit:
                for p in row_pairs(spec, row, groups):
                    if n < covered.get(p, n + 1):
                        covered[p] = n
            if limit is not None and meta["origin"] not in NOT_PAIRWISE:
                most = max(most, n)
        combos = ""
        for gid in groups:
            seen = {f[1] for p in covered for f in p if f[0] == gid}
            combos += f"; {gid}: {len(seen)} combinations in cases"
        scope_all = lv["pairs"] == "all"
        asked = missing = 0
        need = {}

        def assign(feat, groups=groups):
            key, value = feat
            return dict(zip(groups[key], value)) if key in groups else {key: value}

        def feature_ok(feat, assign=assign, accepted=accepted):
            return all(v is None or accepted(a, v) for a, v in assign(feat).items())

        if groups:
            if not scope_all:
                raise ValueError(f"level {lv['name']}: super_setting needs pairs: all")
            grouped = {a for axes in groups.values() for a in axes}
            feats = []
            for a in spec.axes:
                for gid, axes in groups.items():
                    if a == min(axes, key=spec.axes.index):
                        doms = [
                            list(spec.values[x]) + ([None] if x in spec.parent else [])
                            for x in axes
                        ]
                        feats += [(gid, combo) for combo in itertools.product(*doms)]
                if a not in grouped:
                    feats += [(a, v) for v in spec.values[a]]
            feats = [f for f in feats if feature_ok(f)]
            candidates = [
                (f, g)
                for i, f in enumerate(feats)
                for g in feats[i + 1 :]
                if f[0] != g[0]
            ]
        else:
            candidates = spec.candidate_pairs()
        for f, g in candidates:
            if not (feature_ok(f) and feature_ok(g)):
                continue
            if (
                not groups
                and not scope_all
                and group[f[0]] != group[g[0]]
                and frozenset((f[0], g[0])) not in links
            ):
                continue
            fixed = {**assign(f), **assign(g)}
            if (f, g) in covered:
                asked += 1
                n = covered[(f, g)]
                if escalate and n > limit:
                    if spec.satisfiable(
                        fixed, ids, base=base, max_changes=n - 1, groups=groups
                    ):
                        problems.append(
                            f"pair {f[0]}={f[1]} + {g[0]}={g[1]}: its case with the fewest changes "
                            f"changes {n}, but a valid row with {n - 1} holds it"
                        )
                    else:
                        need[(f, g)] = n
                continue
            if spec.satisfiable(
                fixed,
                ids,
                base=base,
                max_changes=None if escalate else limit,
                groups=groups,
            ):
                asked += 1
                missing += 1
                problems.append(
                    f"pair {f[0]}={f[1]} + {g[0]}={g[1]} is asked for but in no case up to {lv['name']}"
                )
        spec.nogoods = saved

        extra = ""
        if escalate:
            # A settings row a lower level already holds is that level's row at another shape tier.
            lower = {
                tuple(r[a] for a in spec.axes)
                for _, m, r in cases
                if names.index(m["level"]) < li
            }
            over = [
                (cid, meta, row)
                for cid, meta, row in cases
                if meta["level"] == lv["name"]
                and meta["origin"] not in NOT_PAIRWISE
                and tuple(row[a] for a in spec.axes) not in lower
                and change_count(spec, base, row, groups) > limit
            ]
            only = 0
            changes_of, holders = {}, {}
            for _, _, r in mine:
                k = tuple(r[a] for a in spec.axes)
                if k in changes_of:
                    continue
                changes_of[k] = change_count(spec, base, r, groups)
                for p in row_pairs(spec, r, groups):
                    if p in need:
                        holders.setdefault(p, []).append(k)
            for cid, meta, row in over:
                n = change_count(spec, base, row, groups)
                held = [p for p in row_pairs(spec, row, groups) if need.get(p) == n]
                if not held:
                    problems.append(
                        f"{cid}: changes {n} settings, but holds no pair of {lv['name']} that needs {n}"
                    )
                    continue
                key = tuple(row[a] for a in spec.axes)
                if any(
                    all(k == key or changes_of[k] > n for k in holders.get(p, ()))
                    for p in held
                ):
                    only += 1
            by_n = {}
            for n in need.values():
                by_n[n] = by_n.get(n, 0) + 1
            rows_n = {}
            for _, meta, row in over:
                n = change_count(spec, base, row, groups)
                rows_n[n] = rows_n.get(n, 0) + 1
            extra = (
                f"; pairs needing more than {limit} changes: "
                + (", ".join(f"{n}: {by_n[n]}" for n in sorted(by_n)) or "none")
                + f"; cases of this level above {limit}: "
                + (", ".join(f"{n}: {rows_n[n]}" for n in sorted(rows_n)) or "none")
                + f" ({only} of {len(over)} hold a pair that no case with other settings and as few changes holds)"
            )

        status = "PASS" if not problems else "FAIL"
        if limit is None:
            within = ""
        elif escalate:
            within = (
                f", at most {most} changes per case (start {limit}, escalating){extra}"
            )
        else:
            within = f", at most {most} changes per case (limit {limit})"
        print(
            f"{status} level {lv['name']}: {len(mine)} cases up to this level, "
            f"{asked} pairs asked for, {missing} missing{within}{combos}{tier_note}"
        )
        for p in problems[:MAX_LISTED]:
            print(f"  {p}")
        if len(problems) > MAX_LISTED:
            print(f"  ... {len(problems) - MAX_LISTED} more")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
