#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Select the bundle folders of each tier that reach the tier's pair-coverage target.

Reads levels.csv, the output folder of select_rows.py (selected.csv, exclusions.csv,
summary.json) and model.yaml + rules.yaml. It changes none of them.

Pairs. A row holds the valid value pairs (sdpa_case_flow_poc.pairs: candidate_pairs, then
feasible_pairs on the model's rules) of its active values. The reachable pairs of a tier are
the pairs of its bundle-expressible rows (the rows of selected.csv).

Folders. The folder of a row is its tier and the superset template of its strict signature
(topology.superset_map over the tier's bundle-expressible rows). The report also gives the
figures for exact templates, where the folder is the topology of the signature itself.

Selection, per tier (every tie goes to the smaller folder name, then the smaller case_key):
  1. need = ceil(target x reachable pairs / 100); the target is 100 % for quick, standard and
     comprehensive and 90 % for full.
  2. Every folder that holds a hand-picked row (origin targeted, real_world or one-at-a-time)
     is selected ("forced"). A hand-picked row that select_rows.py excludes is reported with
     its reasons.
  3. While the covered pairs are fewer than need, a lazy greedy step selects the folder that
     adds the most pairs ("greedy"). A folder that adds no pair is never selected.
Whole mode installs every row of each selected folder. Subset mode is a reported figure only:
per selected folder, in selection order, it keeps every hand-picked row, then each row that
adds a pair not yet covered (most new pairs first). Both modes cover the same pairs.

Outputs in --out-dir:
  selected-by-target.csv  every row of every selected folder: the levels.csv line plus the
                          columns selection_folder, selection_why (forced, greedy or ship_all) and
                          selection_subset_kept (yes or no); input for make_bundles.py --selected
  selection-report.json   per tier and in total: pairs, folders, cases, hand-picked rows, the
                          estimated lane times against the budgets, and the same figures with
                          exact templates
  selection-report.md     the same figures as tables

The lane time of a tier is the summed est_seconds of the whole selection of that tier and of
every tier below it. A lane time over its working target is reported only. After it writes the
outputs, the script exits with status 3 when a lane time is over its budget, else with status 1
when a tier misses its target or a bundle-expressible hand-picked row is not selected.

--ship-all-tiers T1,T2 selects every folder of the named tiers instead of applying the
target; the other tiers keep the target rule.
"""
import argparse
import csv
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent.parent))
import select_rows  # noqa: E402
import topology  # noqa: E402
from sdpa_case_flow_poc.model import load_model  # noqa: E402
from sdpa_case_flow_poc.pairs import (  # noqa: E402
    axis_pos,
    candidate_pairs,
    feasible_pairs,
)
from sdpa_case_flow_poc.solver import Solver  # noqa: E402

LEVELS = select_rows.LEVELS
HAND_PICKED = ("targeted", "real_world", "one-at-a-time")
TARGET_PERCENT = {"quick": 100, "standard": 100, "comprehensive": 100, "full": 90}
# Lane time limits in seconds; a lane runs its tier and every tier below it.
LANE_BUDGET = {"quick": 300, "standard": 1800, "comprehensive": 3600}
LANE_WORKING_TARGET = {"quick": 150, "standard": 900, "comprehensive": 1800}
EXTRA_COLUMNS = ("selection_folder", "selection_why", "selection_subset_kept")
EXIT_OVER_BUDGET = 3


def popcount(mask):
    return bin(mask).count("1")


def lazy_greedy(gain_sets, covered, keys, stop=None):
    """[(key, gain)] in greedy order: most new bits first, ties by the smaller key.

    gain_sets maps a key to a bit mask. covered is the mask that counts as held already.
    stop(covered) ends the loop when it returns True. A key that adds nothing is never taken.
    """
    heap = [(-popcount(gain_sets[k] & ~covered), k) for k in keys]
    heapq.heapify(heap)
    order = []
    while heap and not (stop and stop(covered)):
        _, k = heapq.heappop(heap)
        gain = popcount(gain_sets[k] & ~covered)
        if gain == 0:
            continue
        if heap and (-gain, k) > heap[0]:
            heapq.heappush(heap, (-gain, k))
            continue
        covered |= gain_sets[k]
        order.append((k, gain))
    return order, covered


def select_tier(rows, target_percent, ship_all=False):
    """Selection of one tier.

    rows: the tier's bundle-expressible rows; each has case_key, origin, est_seconds,
    _pairs (bit mask) and _folder. With ship_all, every folder of the tier is selected.
    Returns a dict with the selected folders and the figures.
    """
    folders = defaultdict(list)
    for r in rows:
        folders[r["_folder"]].append(r)
    for name in folders:
        folders[name].sort(key=lambda r: r["case_key"])
    union = {}
    for name, members in folders.items():
        mask = 0
        for r in members:
            mask |= r["_pairs"]
        union[name] = mask
    ceiling = 0
    for mask in union.values():
        ceiling |= mask
    reachable = popcount(ceiling)
    need = math.ceil(target_percent * reachable / 100)

    forced = sorted(
        name
        for name, members in folders.items()
        if any(r["origin"] in HAND_PICKED for r in members)
    )
    covered = 0
    for name in forced:
        covered |= union[name]
    forced_set = set(forced)
    rest = sorted(n for n in folders if n not in forced_set)
    if ship_all:
        greedy = []
        for name in rest:
            covered |= union[name]
        shipped = rest
    else:
        greedy, covered = lazy_greedy(
            union, covered, rest, stop=lambda c: popcount(c) >= need
        )
        shipped = []
    shortfall = max(0, need - popcount(covered))

    why = {name: "forced" for name in forced}
    why.update({name: "greedy" for name, _ in greedy})
    why.update({name: "ship_all" for name in shipped})
    order = forced + [name for name, _ in greedy] + shipped

    kept = set()
    sub_covered = 0
    for name in order:
        members = folders[name]
        for r in members:
            if r["origin"] in HAND_PICKED:
                kept.add(r["case_key"])
                sub_covered |= r["_pairs"]
        others = {
            r["case_key"]: r["_pairs"] for r in members if r["case_key"] not in kept
        }
        picked, sub_covered = lazy_greedy(others, sub_covered, sorted(others))
        kept.update(k for k, _ in picked)
    if sub_covered != covered:
        raise AssertionError("subset mode and whole mode cover different pairs")

    selected_rows = [r for name in order for r in folders[name]]
    subset_count = Counter(
        name for name in order for r in folders[name] if r["case_key"] in kept
    )
    return {
        "folders": folders,
        "order": order,
        "why": why,
        "greedy_gain": dict(greedy),
        "kept": kept,
        "covered": covered,
        "ceiling": ceiling,
        "figures": {
            "target_percent": target_percent,
            "reachable_pairs": reachable,
            "need": need,
            "pairs_reached": popcount(covered),
            "percent_reached": (
                round(100 * popcount(covered) / reachable, 3) if reachable else 100.0
            ),
            "shortfall": shortfall,
            "folders_available": len(folders),
            "folders_selected": len(order),
            "folders_forced": len(forced),
            "folders_greedy": len(greedy),
            "folders_ship_all": len(shipped),
            "cases_available": len(rows),
            "cases_whole": len(selected_rows),
            "cases_subset": len(kept),
            "pairs_whole": popcount(covered),
            "pairs_subset": popcount(sub_covered),
            "one_case_folders_whole": sum(1 for n in order if len(folders[n]) == 1),
            "folders_keeping_one_case_subset": sum(
                1 for n in order if subset_count[n] == 1
            ),
            "est_seconds_whole_gfx942": round(
                sum(float(r["est_seconds"] or 0) for r in selected_rows), 1
            ),
            "est_seconds_whole_gfx950": round(
                sum(float(r.get("est_seconds_gfx950") or 0) for r in selected_rows), 1
            ),
            "est_seconds_subset_gfx942": round(
                sum(
                    float(r["est_seconds"] or 0)
                    for r in selected_rows
                    if r["case_key"] in kept
                ),
                1,
            ),
        },
    }


def assign_folders(rows, model, exact=False):
    """Set _signature and _folder on every row; supersets are computed per tier."""
    for r in rows:
        r["_signature"] = model.tensors(r, True)
    if exact:
        for r in rows:
            r["_folder"] = topology.topology_name(r["_signature"])
        return
    names = topology.superset_topologies([(r["level"], r["_signature"]) for r in rows])
    for r in rows:
        r["_folder"] = names[(r["level"], r["_signature"])]


def check_regroup(selected_rows):
    """Exit unless the superset map over only the selected rows keeps every row's folder."""
    names = topology.superset_topologies(
        [(r["level"], r["_signature"]) for r in selected_rows]
    )
    moved = [
        r["case_key"]
        for r in selected_rows
        if names[(r["level"], r["_signature"])] != r["_folder"]
    ]
    if moved:
        sys.exit(
            f"select_by_target: {len(moved)} selected rows change folder when only the"
            f" selected rows are grouped, e.g. {moved[:5]}"
        )


def select_all(rows, hand_picked_out, ship_all_tiers=()):
    """Per-tier selections of the bundle-expressible rows (folders already assigned).

    Every folder of a tier in ship_all_tiers is selected.
    """
    out = {}
    for lv in LEVELS:
        tier_rows = [r for r in rows if r["level"] == lv]
        if not tier_rows and not hand_picked_out.get(lv):
            continue
        sel = select_tier(tier_rows, TARGET_PERCENT[lv], ship_all=lv in ship_all_tiers)
        sel["hand_picked"] = hand_picked_figures(
            tier_rows, sel, hand_picked_out.get(lv, [])
        )
        out[lv] = sel
    return out


def hand_picked_figures(tier_rows, sel, out_rows):
    selected = {r["case_key"] for n in sel["order"] for r in sel["folders"][n]}
    per_origin = {}
    for o in HAND_PICKED:
        mine = [r for r in tier_rows if r["origin"] == o]
        per_origin[o] = {
            "bundle_expressible": len(mine),
            "in_whole": sum(1 for r in mine if r["case_key"] in selected),
            "kept_subset": sum(1 for r in mine if r["case_key"] in sel["kept"]),
            "not_bundle_expressible": sum(1 for r in out_rows if r["origin"] == o),
        }
    missing = sorted(
        r["case_key"]
        for r in tier_rows
        if r["origin"] in HAND_PICKED
        and (r["case_key"] not in selected or r["case_key"] not in sel["kept"])
    )
    return {
        "per_origin": per_origin,
        "expressible_not_selected": missing,
        "not_bundle_expressible": [
            {
                "case_key": r["case_key"],
                "origin": r["origin"],
                "detail": r["detail"],
                "reasons": r["_reasons"],
            }
            for r in sorted(out_rows, key=lambda r: r["case_key"])
        ],
    }


def lane_figures(selections):
    lanes = {}
    total = {"gfx942": 0.0, "gfx950": 0.0}
    for lv in LEVELS:
        if lv not in selections:
            continue
        f = selections[lv]["figures"]
        total["gfx942"] += f["est_seconds_whole_gfx942"]
        total["gfx950"] += f["est_seconds_whole_gfx950"]
        lane = {
            "lane_seconds_gfx942": round(total["gfx942"], 1),
            "lane_seconds_gfx950": round(total["gfx950"], 1),
        }
        worst = max(total.values())
        if lv in LANE_BUDGET:
            lane["budget_seconds"] = LANE_BUDGET[lv]
            lane["working_target_seconds"] = LANE_WORKING_TARGET[lv]
            if worst > LANE_BUDGET[lv]:
                lane["status"] = "over_budget"
            elif worst > LANE_WORKING_TARGET[lv]:
                lane["status"] = "over_working_target"
            else:
                lane["status"] = "within_working_target"
        else:
            lane["status"] = "no_budget"
        lanes[lv] = lane
    return lanes


class PairIndex:
    """Bit positions of the model's valid pairs, and the pair mask of a levels.csv row."""

    def __init__(self, model_dir):
        model = load_model(model_dir)
        pos = axis_pos(model)
        valid = feasible_pairs(Solver(model), candidate_pairs(model))
        self.axes = list(model.axes)
        feature_id = {}
        for a in self.axes:
            for v in model.values[a]:
                feature_id[(a, str(v))] = len(feature_id)
        self.feature_id = feature_id
        bit = {}
        for f, g in sorted(
            valid,
            key=lambda p: (pos[p[0][0]], str(p[0][1]), pos[p[1][0]], str(p[1][1])),
        ):
            i = feature_id[(f[0], str(f[1]))]
            j = feature_id[(g[0], str(g[1]))]
            bit[(min(i, j), max(i, j))] = len(bit)
        self.partners = defaultdict(dict)
        for (i, j), b in bit.items():
            self.partners[i][j] = b
        self.valid_pairs = len(bit)

    def row_mask(self, row):
        ids = sorted(
            self.feature_id[(a, row[a])] for a in self.axes if row[a] not in ("", "-")
        )
        bits = []
        for n, i in enumerate(ids):
            partners = self.partners.get(i)
            if partners:
                bits.extend(partners[j] for j in ids[n + 1 :] if j in partners)
        data = bytearray((self.valid_pairs + 7) // 8)
        for b in bits:
            data[b >> 3] |= 1 << (b & 7)
        return int.from_bytes(data, "little")


def read_select_dir(select_dir, input_path, lines, rows):
    """{case_key: reasons} of the excluded rows; checks select_rows.py ran on this input."""
    for name in ("summary.json", "selected.csv", "exclusions.csv"):
        if not (select_dir / name).is_file():
            sys.exit(f"select_by_target: {select_dir} has no {name}")
    summary = json.loads((select_dir / "summary.json").read_text())
    digest = select_rows.sha256(input_path)
    if summary.get("input_sha256") != digest:
        sys.exit(
            f"select_by_target: {select_dir}/summary.json is for input"
            f" {summary.get('input_sha256')}, not {digest}"
        )
    _, sel_lines, _ = select_rows.read_rows(select_dir / "selected.csv")
    excluded = {}
    with open(select_dir / "exclusions.csv", newline="") as f:
        for rec in csv.DictReader(f):
            excluded[rec["case_key"]] = [x for x in rec["reasons"].split(";") if x]
    keys = {r["case_key"] for r in rows}
    if len(keys) != len(rows):
        sys.exit("select_by_target: case keys of the input are not unique")
    by_line = {line: r["case_key"] for r, line in zip(rows, lines)}
    unknown = [line for line in sel_lines if line not in by_line]
    if unknown:
        sys.exit(
            f"select_by_target: {len(unknown)} lines of selected.csv are not lines of the input"
        )
    selected = {by_line[line] for line in sel_lines}
    stray = set(excluded) - keys
    if stray or len(set(sel_lines)) != len(sel_lines):
        sys.exit(
            f"select_by_target: {len(stray)} excluded keys are not in the input,"
            f" {len(sel_lines) - len(set(sel_lines))} lines of selected.csv repeat"
        )
    both = selected & set(excluded)
    neither = keys - selected - set(excluded)
    if both or neither:
        sys.exit(
            f"select_by_target: {len(both)} rows are both selected and excluded,"
            f" {len(neither)} rows are neither"
        )
    return selected, excluded, summary


def write_selected(path, header, selections):
    rows = []
    for lv in LEVELS:
        sel = selections.get(lv)
        if not sel:
            continue
        for name in sel["order"]:
            for r in sel["folders"][name]:
                rows.append(
                    (
                        LEVELS.index(lv),
                        name,
                        r["case_key"],
                        ",".join(
                            (
                                r["_line"],
                                name,
                                sel["why"][name],
                                "yes" if r["case_key"] in sel["kept"] else "no",
                            )
                        ),
                    )
                )
    rows.sort(key=lambda t: t[:3])
    with open(path, "w", newline="") as f:
        f.write(",".join([header] + list(EXTRA_COLUMNS)) + "\n")
        for t in rows:
            f.write(t[3] + "\n")


def tier_report(sel):
    return {
        **sel["figures"],
        "hand_picked": sel["hand_picked"],
        "folders": [
            {
                "folder": name,
                "why": sel["why"][name],
                "cases_whole": len(sel["folders"][name]),
                "cases_subset": sum(
                    1 for r in sel["folders"][name] if r["case_key"] in sel["kept"]
                ),
                "greedy_new_pairs": sel["greedy_gain"].get(name),
            }
            for name in sel["order"]
        ],
    }


TOTAL_KEYS = (
    "reachable_pairs",
    "pairs_reached",
    "folders_available",
    "folders_selected",
    "folders_forced",
    "folders_greedy",
    "folders_ship_all",
    "cases_available",
    "cases_whole",
    "cases_subset",
    "one_case_folders_whole",
    "folders_keeping_one_case_subset",
    "est_seconds_whole_gfx942",
    "est_seconds_whole_gfx950",
    "est_seconds_subset_gfx942",
)


def totals(selections):
    return {
        k: round(sum(s["figures"][k] for s in selections.values()), 1)
        for k in TOTAL_KEYS
    }


def cumulative_reachable(selections):
    out, acc = {}, 0
    for lv in LEVELS:
        if lv in selections:
            acc |= selections[lv]["ceiling"]
            out[lv] = popcount(acc)
    return out


def markdown(report):
    lines = ["# SDPA forward bundle selection by pair-coverage target", ""]
    lines.append(f"Input `{report['input']}` sha256 `{report['input_sha256']}`.")
    lines.append(
        f"select_rows.py output `{report['select_dir']}`; export outcomes:"
        f" {report['select_parameters'].get('export_outcomes') or 'none'}."
    )
    lines.append(f"Valid pairs of the model: {report['valid_pairs']}.")
    lines.append(
        "Tiers that select every folder: "
        + (", ".join(report["ship_all_tiers"]) or "none")
        + "."
    )
    lines.append("")
    for mode in ("superset", "exact"):
        tiers = report["modes"][mode]["tiers"]
        lines.append(
            f"## {'Superset templates (installed)' if mode == 'superset' else 'Exact templates (comparison only)'}"
        )
        lines.append("")
        lines.append(
            "| tier | reachable pairs | target % | reached whole / subset | % | folders forced + greedy + ship-all = selected (of) |"
            " cases whole / subset (of) | one-case folders whole / subset | est s whole gfx942 / gfx950 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for lv, t in tiers.items():
            lines.append(
                f"| {lv} | {t['reachable_pairs']} | {t['target_percent']} | {t['pairs_whole']} /"
                f" {t['pairs_subset']} |"
                f" {t['percent_reached']} | {t['folders_forced']} + {t['folders_greedy']} +"
                f" {t['folders_ship_all']} ="
                f" {t['folders_selected']} ({t['folders_available']}) | {t['cases_whole']} /"
                f" {t['cases_subset']} ({t['cases_available']}) | {t['one_case_folders_whole']} /"
                f" {t['folders_keeping_one_case_subset']} | {t['est_seconds_whole_gfx942']} /"
                f" {t['est_seconds_whole_gfx950']} |"
            )
        tot = report["modes"][mode]["total"]
        lines.append(
            f"| sum over tiers | {tot['reachable_pairs']} | | {tot['pairs_reached']} | |"
            f" {tot['folders_forced']} + {tot['folders_greedy']} + {tot['folders_ship_all']}"
            f" = {tot['folders_selected']}"
            f" ({tot['folders_available']}) | {tot['cases_whole']} / {tot['cases_subset']}"
            f" ({tot['cases_available']}) | {tot['one_case_folders_whole']} /"
            f" {tot['folders_keeping_one_case_subset']} | {tot['est_seconds_whole_gfx942']} /"
            f" {tot['est_seconds_whole_gfx950']} |"
        )
        lines.append("")
        lines.append(
            "| lane | seconds gfx942 / gfx950 | working target | budget | status |"
        )
        lines.append("|---|---|---|---|---|")
        for lv, lane in report["modes"][mode]["lanes"].items():
            lines.append(
                f"| {lv} | {lane['lane_seconds_gfx942']} / {lane['lane_seconds_gfx950']} |"
                f" {lane.get('working_target_seconds', '-')} | {lane.get('budget_seconds', '-')} |"
                f" {lane['status']} |"
            )
        lines.append("")
    lines.append("## Hand-picked rows (superset templates)")
    lines.append("")
    lines.append(
        "| tier | origin | bundle-expressible | in whole | kept in subset | not bundle-expressible |"
    )
    lines.append("|---|---|---|---|---|---|")
    for lv, t in report["modes"]["superset"]["tiers"].items():
        for o, c in t["hand_picked"]["per_origin"].items():
            lines.append(
                f"| {lv} | {o} | {c['bundle_expressible']} | {c['in_whole']} |"
                f" {c['kept_subset']} | {c['not_bundle_expressible']} |"
            )
    lines.append("")
    out_rows = [
        (lv, x)
        for lv, t in report["modes"]["superset"]["tiers"].items()
        for x in t["hand_picked"]["not_bundle_expressible"]
    ]
    if out_rows:
        lines.append(
            f"Hand-picked rows that select_rows.py excludes ({len(out_rows)});"
            " no folder holds them:"
        )
        lines.append("")
        lines.append("| tier | case_key | origin | detail | reasons |")
        lines.append("|---|---|---|---|---|")
        for lv, x in out_rows:
            lines.append(
                f"| {lv} | {x['case_key']} | {x['origin']} | {x['detail']} |"
                f" {';'.join(x['reasons'])} |"
            )
        lines.append("")
    lines.append(
        "Reachable pairs, cumulative over the tiers: "
        + ", ".join(
            f"{lv} {n}" for lv, n in report["reachable_pairs_cumulative"].items()
        )
        + "."
    )
    lines.append("")
    return "\n".join(lines)


def run(args):
    header, lines, rows = select_rows.read_rows(args.input)
    selected, excluded, summary = read_select_dir(
        args.select_dir, args.input, lines, rows
    )
    model = select_rows.Model(args.class_values)
    index = PairIndex(args.model_dir)

    expressible = []
    hand_picked_out = defaultdict(list)
    for r, line in zip(rows, lines):
        r["_line"] = line
        if r["case_key"] in selected:
            r["_pairs"] = index.row_mask(r)
            expressible.append(r)
        elif r["origin"] in HAND_PICKED:
            r["_reasons"] = excluded[r["case_key"]]
            hand_picked_out[r["level"]].append(r)

    modes = {}
    for mode in ("exact", "superset"):
        assign_folders(expressible, model, exact=(mode == "exact"))
        selections = select_all(expressible, hand_picked_out, args.ship_all_tiers)
        modes[mode] = {
            "selections": selections,
            "tiers": {lv: tier_report(s) for lv, s in selections.items()},
            "total": totals(selections),
            "lanes": lane_figures(selections),
        }
    superset = modes["superset"]["selections"]
    check_regroup(
        [r for s in superset.values() for n in s["order"] for r in s["folders"][n]]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "selected-by-target.csv"
    write_selected(csv_path, header, superset)
    report = {
        "input": str(args.input),
        "input_sha256": select_rows.sha256(args.input),
        "select_dir": str(args.select_dir),
        "select_parameters": summary.get("parameters", {}),
        "ship_all_tiers": list(args.ship_all_tiers),
        "class_values_sha256": select_rows.sha256(args.class_values),
        "model_sha256": select_rows.sha256(args.model_dir / "model.yaml"),
        "rules_sha256": select_rows.sha256(args.model_dir / "rules.yaml"),
        "valid_pairs": index.valid_pairs,
        "reachable_pairs_cumulative": cumulative_reachable(superset),
        "selected_by_target_sha256": select_rows.sha256(csv_path),
        "modes": {
            m: {k: v for k, v in d.items() if k != "selections"}
            for m, d in modes.items()
        },
    }
    (args.out_dir / "selection-report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n"
    )
    (args.out_dir / "selection-report.md").write_text(markdown(report))

    t = modes["superset"]["total"]
    print(
        f"select_by_target: {t['folders_selected']} folders, {t['cases_whole']} cases"
        f" (subset {t['cases_subset']}); wrote {args.out_dir}",
        file=sys.stderr,
    )
    status = 0
    over_budget = False
    for lv, s in superset.items():
        if s["figures"]["shortfall"]:
            print(
                f"select_by_target: {lv} misses its target by {s['figures']['shortfall']} pairs",
                file=sys.stderr,
            )
            status = 1
        if s["hand_picked"]["expressible_not_selected"]:
            print(
                f"select_by_target: {lv} hand-picked rows not selected:"
                f" {s['hand_picked']['expressible_not_selected']}",
                file=sys.stderr,
            )
            status = 1
    for lv, lane in modes["superset"]["lanes"].items():
        if lane["status"] == "over_budget":
            print(
                f"select_by_target: lane {lv} is over its budget"
                f" ({lane['lane_seconds_gfx942']} s gfx942, {lane['lane_seconds_gfx950']} s"
                f" gfx950, budget {lane['budget_seconds']} s)",
                file=sys.stderr,
            )
            over_budget = True
        elif lane["status"] == "over_working_target":
            print(
                f"select_by_target: lane {lv} is over its working target"
                f" ({lane['lane_seconds_gfx942']} s gfx942, target"
                f" {lane['working_target_seconds']} s)",
                file=sys.stderr,
            )
    return EXIT_OVER_BUDGET if over_budget else status


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, required=True, help="levels.csv")
    ap.add_argument(
        "--select-dir",
        type=Path,
        required=True,
        help="output folder of select_rows.py for the same levels.csv",
    )
    ap.add_argument(
        "--class-values", type=Path, required=True, help="class-to-value.json"
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=select_rows.PLAN_DIR,
        help="folder with model.yaml and rules.yaml",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--ship-all-tiers",
        default="",
        help="comma-separated tiers whose folders are all selected (no target)",
    )
    args = ap.parse_args()
    args.ship_all_tiers = tuple(t for t in args.ship_all_tiers.split(",") if t)
    bad = [t for t in args.ship_all_tiers if t not in LEVELS]
    if bad:
        sys.exit(f"select_by_target: unknown tier(s) {bad}")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
