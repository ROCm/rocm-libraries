#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Import exported SDPA forward graphs into a bundle tree as grouped sweeps.

Rows are grouped into superset templates (topology.superset_map, one map per tier): the
sweep of a superset holds every row whose signature it can hold, and a row with fewer
tensors runs as a case that names its missing tensors absent. --exact-templates keeps one
sweep per signature instead. For every selected row, sorted by tier, topology, signature
size (largest first, so the superset's own rows create the template), signature and
case_key, the script runs the suite's import_graph.py in grouped mode on the row's exported
graph, into an empty work tree W that holds only <tier>/<OP>/ (so each import scans only
these sweeps):

  --graph <export>/<case_key>/<case_key>.json --bundle-dir W --tier T --op-dir OP
  --topology NAME --template-attributes --template-alignment --skeleton-value-form
  --only-op-dir --require-marked-sweep [--superset-templates]
  --case-id <dtype_q>_<case_key> --strict

Afterwards it checks that every case of one topology name lands in one sweep per tier
(<tier>/<OP>/<NAME>/), that no sweep holds two topology names, that the op folder holds no
other sweep, and that every case re-expands to its exported graph. With --install-into TREE
it then copies each <tier>/<OP>/ folder of W into TREE; the copy refuses any path that exists
in TREE and checks every copied file byte for byte. It writes:

  placements.csv  case_key, tier, topology, case_id, bundle_path, gtest_name
  report.json     per-import times, sweep.json bytes per case, failures, projections, and
                  per tier the sweeps and one-case sweeps without and with superset
                  templates
  timings.csv     per import: tier, topology, cases already in the sweep, seconds

--only-tier, --only-topology and --limit import a subset (for example about 300 rows of the
largest topology) to measure the time per import. The report fits seconds = a + b * k, with
k the cases already in the sweep, and projects the full-run time per tier from the sweep
sizes of all selected rows.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import select_rows  # noqa: E402
import topology  # noqa: E402

MIGRATION = select_rows.MIGRATION_DIR
GROUPED_FLAGS = [
    "--template-attributes",
    "--template-alignment",
    "--skeleton-value-form",
    "--only-op-dir",
    "--require-marked-sweep",
]
TIER_ORDER = {lv: i for i, lv in enumerate(select_rows.LEVELS)}


def case_id(row):
    return f"{row['dtype_q']}_{row['case_key']}"


def sanitize_gtest(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def gtest_name(tier, op_dir, topo, cid):
    return f"{sanitize_gtest(tier)}_{sanitize_gtest(op_dir)}_{sanitize_gtest(topo)}.{sanitize_gtest(cid)}"


def load_rows(selected, class_values):
    """Selected rows with their signature (_signature), topology (_topology) and case id
    (_case_id)."""
    _, _, rows = select_rows.read_rows(selected)
    model = select_rows.Model(class_values)
    topology.check_one_to_one(model, rows)
    for r in rows:
        r["_signature"] = model.tensors(r, True)
        r["_topology"] = topology.topology_name(r["_signature"])
        r["_case_id"] = case_id(r)
    if len({r["_case_id"] for r in rows}) != len(rows):
        sys.exit("make_bundles: case ids are not unique")
    return rows


def sample_rows(rows, per_tier):
    """A deterministic sample that covers every tier and several topologies per tier.

    Per tier: two rows of the largest topology (so one import appends), one row of a
    runtime-scale topology when the tier has one, then one row of each next-largest topology.
    """
    out = []
    for lv in select_rows.LEVELS:
        tier_rows = sorted(
            (r for r in rows if r["level"] == lv), key=lambda r: r["case_key"]
        )
        by_topo = defaultdict(list)
        for r in tier_rows:
            by_topo[r["_topology"]].append(r)
        order = sorted(by_topo, key=lambda t: (-len(by_topo[t]), t))
        if not order:
            continue
        picked = by_topo[order[0]][:2]
        runtime = [t for t in order[1:] if "ScRt" in t.split("_")]
        rest = runtime[:1] + [t for t in order[1:] if t not in runtime[:1]]
        for t in rest:
            if len(picked) >= per_tier:
                break
            picked.append(by_topo[t][0])
        out.extend(picked[:per_tier])
    return out


def assign_supersets(rows):
    """Set each row's _topology to the name of its superset template (per tier)."""
    names = topology.superset_topologies([(r["level"], r["_signature"]) for r in rows])
    for r in rows:
        r["_topology"] = names[(r["level"], r["_signature"])]


def import_order(row):
    return (
        TIER_ORDER[row["level"]],
        row["_topology"],
        -len(row["_signature"]),
        row["_signature"],
        row["case_key"],
    )


def run_import(import_graph, graph, bundle_dir, row, op_dir, env, flags):
    cmd = [
        sys.executable,
        str(import_graph),
        "--graph",
        str(graph),
        "--bundle-dir",
        str(bundle_dir),
        "--tier",
        row["level"],
        "--op-dir",
        op_dir,
        "--topology",
        row["_topology"],
        *flags,
        "--case-id",
        row["_case_id"],
        "--strict",
    ]
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc, time.monotonic() - start, cmd


def check_tree(bundle_dir, op_dir, placed, export_dir):
    """Problems found in the written tree; empty when every check holds."""
    sys.path.insert(0, str(MIGRATION))
    from bundle_utils import (
        ABSENT_TENSORS_KEY,
        absent_case_problems,
        aliases_by_name,
        canon,
        expand_case,
        inputs_by_name,
        name_by_uid,
        remap_graph,
        uid_map_onto,
    )  # noqa: E402

    def uids_by_name(meta, graph, section):
        """Entries of a uid-carrying metadata section with every uid named, sorted."""
        names = name_by_uid(graph)
        return sorted(
            json.dumps(
                {
                    k: names.get(v, v) if k in ("uid", "lengths", "like") else v
                    for k, v in e.items()
                },
                sort_keys=True,
            )
            for e in meta.get(section, [])
        )

    problems = []
    expected = defaultdict(dict)
    for row in placed:
        expected[(row["level"], row["_topology"])][row["_case_id"]] = row
    tiers = sorted({t for t, _ in expected}, key=TIER_ORDER.get)
    for tier in tiers:
        op_path = bundle_dir / tier / op_dir
        present = (
            sorted(p.name for p in op_path.iterdir() if p.is_dir())
            if op_path.is_dir()
            else []
        )
        wanted = sorted(topo for t, topo in expected if t == tier)
        extra = sorted(set(present) - set(wanted))
        if extra:
            problems.append(
                f"{tier}/{op_dir} holds sweeps no selected row names: {extra}"
            )
    for (tier, topo), cases in sorted(
        expected.items(), key=lambda kv: (TIER_ORDER[kv[0][0]], kv[0][1])
    ):
        sweep_dir = bundle_dir / tier / op_dir / topo
        sweep_path = sweep_dir / "sweep.json"
        tpl_path = sweep_dir / "graph.template.json"
        if not sweep_path.is_file() or not tpl_path.is_file():
            problems.append(f"{sweep_dir}: no sweep.json or graph.template.json")
            continue
        sweep = json.loads(sweep_path.read_text())
        template = json.loads(tpl_path.read_text())
        if "import_mode" not in sweep:
            problems.append(f"{sweep_path}: no import_mode marker")
        ids = [c["id"] for c in sweep.get("cases", [])]
        if sorted(ids) != sorted(cases):
            problems.append(
                f"{sweep_path}: case ids {len(ids)} differ from the {len(cases)} expected"
                f" (missing {sorted(set(cases) - set(ids))[:5]}, extra {sorted(set(ids) - set(cases))[:5]})"
            )
        for c in sweep.get("cases", []):
            row = cases.get(c["id"])
            if row is None:
                continue
            case_dir = export_dir / row["case_key"]
            graph = json.loads((case_dir / f"{row['case_key']}.json").read_text())
            meta = json.loads((case_dir / f"{row['case_key']}.meta.json").read_text())
            uid_map = uid_map_onto(graph, template)
            if uid_map is None:
                problems.append(
                    f"{sweep_path}:{c['id']} has a tensor name the template lacks"
                )
                continue
            graph_t = remap_graph(graph, uid_map)
            used = {t["uid"] for t in graph_t.get("tensors", []) if "uid" in t}
            if used & set(c.get(ABSENT_TENSORS_KEY) or []):
                problems.append(
                    f"{sweep_path}:{c['id']} marks absent a tensor its graph has"
                )
            for p in absent_case_problems(sweep, template, c):
                problems.append(f"{sweep_path}:{c['id']} {p}")
            try:
                expanded = expand_case(template, c)
            except ValueError as e:
                problems.append(f"{sweep_path}:{c['id']} does not expand: {e}")
                continue
            if canon(expanded) != canon(graph_t):
                problems.append(
                    f"{sweep_path}:{c['id']} does not re-expand to its exported graph"
                )
            if inputs_by_name(c.get("metadata", {}), expanded) != inputs_by_name(
                meta, graph
            ):
                problems.append(
                    f"{sweep_path}:{c['id']} inputs differ from its exported metadata"
                )
            placed_meta = c.get("metadata", {})
            if aliases_by_name(placed_meta, expanded) != aliases_by_name(meta, graph):
                problems.append(
                    f"{sweep_path}:{c['id']} aliases differ from its exported metadata"
                )
            for section in ("start_offsets", "live_rows"):
                if uids_by_name(placed_meta, expanded, section) != uids_by_name(
                    meta, graph, section
                ):
                    problems.append(
                        f"{sweep_path}:{c['id']} {section} differ from its exported metadata"
                    )
    return problems


def prepare_work_dir(work_dir):
    if work_dir.exists() and any(work_dir.iterdir()):
        sys.exit(f"make_bundles: work tree {work_dir} is not empty")
    work_dir.mkdir(parents=True, exist_ok=True)


def install(work_dir, tree, op_dir, tiers):
    """Copy <tier>/<op_dir>/ of the work tree into tree. Refuses existing paths; checks bytes."""
    plan = []
    for tier in tiers:
        src = work_dir / tier / op_dir
        if not src.is_dir():
            continue
        dst = tree / tier / op_dir
        if not (tree / tier).is_dir():
            sys.exit(f"make_bundles: {tree / tier} does not exist")
        if dst.exists():
            sys.exit(f"make_bundles: {dst} exists; install refuses to overwrite")
        plan.append((src, dst))
    copied = 0
    for src, dst in plan:
        shutil.copytree(src, dst)
        for f in src.rglob("*"):
            if f.is_file():
                g = dst / f.relative_to(src)
                if not g.is_file() or g.read_bytes() != f.read_bytes():
                    sys.exit(f"make_bundles: copy of {f} to {g} differs")
                copied += 1
    return [str(d) for _, d in plan], copied


def fit_line(points):
    """Least-squares (a, b) of seconds = a + b * k over (k, seconds) points."""
    n = len(points)
    if n == 0:
        return 0.0, 0.0
    mk = sum(k for k, _ in points) / n
    ms = sum(s for _, s in points) / n
    var = sum((k - mk) ** 2 for k, _ in points)
    b = sum((k - mk) * (s - ms) for k, s in points) / var if var else 0.0
    return ms - b * mk, b


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--selected", type=Path, required=True, help="selected.csv from select_rows.py"
    )
    ap.add_argument(
        "--class-values", type=Path, required=True, help="class-to-value.json"
    )
    ap.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="runner --export-graphs directory",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="empty work tree the imports write into",
    )
    ap.add_argument(
        "--install-into",
        type=Path,
        help="bundle tree to copy the finished op folders into",
    )
    ap.add_argument("--op-dir", default="SdpaFwdPlan")
    ap.add_argument("--import-graph", type=Path, default=MIGRATION / "import_graph.py")
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="where placements.csv and report.json go",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="import only N rows per tier (a covering sample)",
    )
    ap.add_argument("--only-tier", help="import only rows of this tier")
    ap.add_argument(
        "--only-topology",
        help="import only rows whose own signature has this topology name",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="import only the first N rows after the filters",
    )
    ap.add_argument(
        "--keep-going", action="store_true", help="continue after a failed import"
    )
    ap.add_argument(
        "--exact-templates",
        action="store_true",
        help="one sweep per signature (no superset templates, no absent tensors)",
    )
    args = ap.parse_args()
    subset = bool(args.sample or args.only_tier or args.only_topology or args.limit)
    if args.install_into and subset:
        sys.exit(
            "make_bundles: --install-into needs the full selection (no --sample/--only-*/--limit)"
        )

    all_rows = load_rows(args.selected, args.class_values)
    rows = list(all_rows)
    if args.sample:
        rows = sample_rows(rows, args.sample)
    if args.only_tier:
        rows = [r for r in rows if r["level"] == args.only_tier]
    if args.only_topology:
        rows = [r for r in rows if r["_topology"] == args.only_topology]
    if not args.exact_templates:
        assign_supersets(rows)
    flags = GROUPED_FLAGS + ([] if args.exact_templates else ["--superset-templates"])
    rows.sort(key=import_order)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("make_bundles: no rows to import")

    missing = [
        r["case_key"]
        for r in rows
        if not (args.export_dir / r["case_key"] / f"{r['case_key']}.json").is_file()
    ]
    if missing:
        sys.exit(
            f"make_bundles: {len(missing)} selected rows have no exported graph, e.g. {missing[:5]}"
        )

    prepare_work_dir(args.work_dir)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.out_dir / "imports.log", "w")
    timing = open(args.out_dir / "timings.csv", "w")
    timing.write("tier,topology,cases_before,seconds,rc\n")
    times, points, failures, placed = [], [], [], []
    in_sweep = Counter()
    started = time.monotonic()
    for n, r in enumerate(rows, 1):
        graph = args.export_dir / r["case_key"] / f"{r['case_key']}.json"
        proc, seconds, cmd = run_import(
            args.import_graph, graph, args.work_dir, r, args.op_dir, env, flags
        )
        key = (r["level"], r["_topology"])
        times.append(seconds)
        timing.write(
            f"{r['level']},{r['_topology']},{in_sweep[key]},{seconds:.4f},{proc.returncode}\n"
        )
        log.write(
            f"### {n}/{len(rows)} rc={proc.returncode} {seconds:.2f}s {' '.join(cmd)}\n{proc.stderr}\n"
        )
        if proc.returncode != 0:
            failures.append(
                {
                    "case_key": r["case_key"],
                    "rc": proc.returncode,
                    "stderr": proc.stderr[-1500:],
                }
            )
            print(
                f"make_bundles: import failed for {r['case_key']} (rc {proc.returncode})",
                file=sys.stderr,
            )
            if not args.keep_going:
                break
            continue
        points.append((in_sweep[key], seconds))
        in_sweep[key] += 1
        placed.append(r)
        if n % 250 == 0:
            print(
                f"make_bundles: {n}/{len(rows)} imported, {time.monotonic() - started:.0f}s",
                file=sys.stderr,
            )
    log.close()
    timing.close()

    problems = (
        check_tree(args.work_dir, args.op_dir, placed, args.export_dir)
        if not failures
        else []
    )
    installed, copied = [], 0
    if args.install_into and not failures and not problems:
        installed, copied = install(
            args.work_dir, args.install_into, args.op_dir, select_rows.LEVELS
        )

    with open(args.out_dir / "placements.csv", "w", newline="") as f:
        f.write("case_key,tier,topology,case_id,bundle_path,gtest_name\n")
        for r in sorted(placed, key=lambda r: r["case_key"]):
            path = f"{r['level']}/{args.op_dir}/{r['_topology']}"
            f.write(
                ",".join(
                    [
                        r["case_key"],
                        r["level"],
                        r["_topology"],
                        r["_case_id"],
                        path,
                        gtest_name(
                            r["level"], args.op_dir, r["_topology"], r["_case_id"]
                        ),
                    ]
                )
                + "\n"
            )

    sweeps = []
    for (tier, topo), n_cases in sorted(
        Counter((r["level"], r["_topology"]) for r in placed).items()
    ):
        p = args.work_dir / tier / args.op_dir / topo / "sweep.json"
        size = p.stat().st_size if p.is_file() else 0
        sweeps.append(
            {
                "sweep": f"{tier}/{args.op_dir}/{topo}",
                "cases": n_cases,
                "bytes": size,
                "bytes_per_case": round(size / n_cases, 1) if n_cases else None,
            }
        )
    total_bytes = sum(s["bytes"] for s in sweeps)
    total_cases = sum(s["cases"] for s in sweeps)
    per_case = total_bytes / total_cases if total_cases else 0.0
    mean_time = sum(times) / len(times) if times else 0.0

    a, b = fit_line(points)
    full_names = (
        {}
        if args.exact_templates
        else topology.superset_topologies(
            [(r["level"], r["_signature"]) for r in all_rows]
        )
    )
    full_sizes = Counter(
        (
            r["level"],
            full_names.get((r["level"], r["_signature"]), r["_topology"]),
        )
        for r in all_rows
    )
    projected = {lv: 0.0 for lv in select_rows.LEVELS}
    for (tier, _), n_cases in full_sizes.items():
        projected[tier] += (
            n_cases * max(a, 0.0) + max(b, 0.0) * n_cases * (n_cases - 1) / 2
        )
    report = {
        "rows": len(rows),
        "imported": len(placed),
        "failures": failures,
        "problems": problems,
        "installed": installed,
        "installed_files": copied,
        "seconds_total": round(time.monotonic() - started, 1),
        "seconds_per_import_mean": round(mean_time, 3),
        "seconds_per_import_max": round(max(times), 3) if times else 0.0,
        "fit_seconds_per_import": {
            "a": round(a, 4),
            "b_per_case_in_sweep": round(b, 6),
            "points": len(points),
            "max_cases_before": max((k for k, _ in points), default=0),
        },
        "projected_hours_per_tier": {
            lv: round(s / 3600, 2) for lv, s in projected.items()
        },
        "projected_hours_total": round(sum(projected.values()) / 3600, 2),
        "projection_note": "fit from this run only; the projection holds only up to the sweep sizes it measured",
        "sweep_json_bytes_total": total_bytes,
        "sweep_json_bytes_per_case": round(per_case, 1),
        "projected_largest_sweep_mb": round(
            max(full_sizes.values()) * per_case / 1e6, 2
        ),
        "projected_all_selected_cases_mb": round(len(all_rows) * per_case / 1e6, 1),
        "sweeps": sweeps,
        "superset_templates": not args.exact_templates,
        "grouping_of_imported_rows": topology.grouping_summary(
            [(r["level"], r["_signature"]) for r in rows]
        ),
    }
    with open(args.out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    for p in problems:
        print("make_bundles: CHECK FAILED:", p, file=sys.stderr)
    ok = not failures and not problems
    print(
        f"make_bundles: {len(placed)}/{len(rows)} imported into {len(sweeps)} sweeps;"
        f" {mean_time:.2f}s per import (fit {a:.3f} + {b:.5f}*k);"
        f" projected {report['projected_hours_total']} h for all selected rows;"
        f" {per_case:.0f} sweep.json bytes per case; installed {len(installed)} folders;"
        f" {'OK' if ok else 'FAILED'}",
        file=sys.stderr,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
