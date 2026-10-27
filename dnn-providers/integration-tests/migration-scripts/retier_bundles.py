#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Move existing bundle cases between tiers (quick/standard/comprehensive/full).

Re-tiering changes only *where* a case lives. No case is added, removed or
modified: the expanded graph, metadata, tensor_patches, golden pointer and
per-engine support claims of every case are identical before and after, and
the tool proves it by re-reading the tree from disk after writing.

An assignment file maps every existing case to the tier it should live in::

    [{"current_tier": "quick",
      "key": "ConvolutionFwd/Default/1_16_16_16_bfp16_nchw_dil1x1_postpad0x0_prepad0x0",
      "proposed_tier": "standard"}, ...]

``key`` is the tier-independent case identity:

  * template-sweep case:  ``<Operation>/<Topology>/<case id>``
  * single-graph bundle:  ``<Operation>/<path from the operation dir to the
    bundle dir>``, e.g. ``SdpaFwd/bhsd/bf16/hd128_causal_batch/Small``

Every case must be assigned exactly once: a case without an assignment would
be silently dropped, so it is an error. ``--write-identity`` emits an
assignment file that leaves everything where it is, as a starting point.

How a move is applied:

  * Sweep cases move between the ``sweep.json`` files of the same
    ``<Operation>/<Topology>`` in different tier directories (created when
    missing). ``golden/<case id>/`` and the case's per-engine claims in
    ``support.json`` travel with it.
  * Single-graph bundles move as whole directories.
  * A topology whose ``graph.template.json`` differs between tiers is unified
    to the template with the most ``${case.*}`` placeholders. A case that came
    from a template with a literal where the unified template has a
    placeholder gets that literal as an explicit ``attributes`` value, so its
    expanded graph does not change.
  * Two cases with the same id may not land in one sweep.

Usage::

    retier_bundles.py --bundle-dir integration-test-bundles/ \\
        --assignments assignments/ [--apply]

Without ``--apply`` the plan is built and checked but nothing is written.
``--assignments`` is one ``*.json`` file or a directory of ``*.json`` files.
"""

import argparse
import collections
import copy
import json
import os
import shutil
import sys
from pathlib import Path

from bundle_utils import canon, expand

# The sidecar renderer lives with the tools that verify and rewrite sidecars.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from support_sidecar import Claim, render  # noqa: E402

TIERS = ["quick", "standard", "comprehensive", "full"]
# Single-graph bundle JSON that is not the graph itself.
_NOT_A_GRAPH = (".support.json", "sweep.json", "support.json", "meta.json")
_ATTR_PREFIX = "${case.attributes."


def _read_json(path: Path):
    with open(path, newline="") as f:
        return json.loads(f.read().replace("\r\n", "\n"))


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(json.dumps(obj, indent=2) + "\n")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class Sweep:
    """One ``<tier>/<operation>/<topology>`` template-sweep directory."""

    def __init__(self, root: Path, tier: str, operation: str, topology: str):
        self.tier, self.operation, self.topology = tier, operation, topology
        self.dir = root / tier / operation / topology
        self.template = _read_json(self.dir / "graph.template.json")
        self.sweep = _read_json(self.dir / "sweep.json")
        support = self.dir / "support.json"
        self.support = _read_json(support) if support.is_file() else None

    def claims_of(self, case_id: str) -> dict:
        """{engine: support map} for the claims that name this case."""
        out = {}
        for engine, groups in (self.support or {}).get("claims", {}).items():
            for group in groups:
                if case_id in group["cases"]:
                    out[engine] = group["support"]
        return out


def load_tree(root: Path):
    """Return ({(tier, op, topology): Sweep}, {(tier, key): single bundle dir})."""
    sweeps, singles = {}, {}
    for tier in TIERS:
        for op_dir in sorted(p for p in (root / tier).glob("*") if p.is_dir()):
            for dirpath, dirnames, filenames in os.walk(op_dir):
                here = Path(dirpath)
                if "graph.template.json" in filenames and "sweep.json" in filenames:
                    topology = here.relative_to(op_dir).as_posix()
                    sweeps[(tier, op_dir.name, topology)] = Sweep(
                        root, tier, op_dir.name, topology
                    )
                    dirnames[:] = []
                    continue
                graphs = [
                    f
                    for f in filenames
                    if f.endswith(".json") and not f.endswith(_NOT_A_GRAPH)
                ]
                if graphs:
                    rel = here.relative_to(op_dir).as_posix()
                    singles[(tier, f"{op_dir.name}/{rel}")] = here
                    dirnames[:] = []
    return sweeps, singles


def snapshot(root: Path) -> dict:
    """Per-case content fingerprint keyed by (tier, key), for before/after proof."""
    sweeps, singles = load_tree(root)
    snap = {}
    for (tier, op, topology), sw in sweeps.items():
        for case in sw.sweep["cases"]:
            key = f"{op}/{topology}/{case['id']}"
            snap[(tier, key)] = {
                "graph": canon(expand(sw.template, case["values"])),
                "metadata": canon(case.get("metadata")),
                "tensor_patches": canon(case.get("tensor_patches")),
                "golden": canon(case.get("golden")),
                "claims": canon(sw.claims_of(case["id"])),
            }
    for (tier, key), bundle_dir in singles.items():
        files = {}
        for f in sorted(bundle_dir.iterdir()):
            if f.is_file():
                files[f.name] = f.read_bytes().replace(b"\r\n", b"\n")
        snap[(tier, key)] = {"files": files}
    return snap


# --------------------------------------------------------------------------
# Assignments
# --------------------------------------------------------------------------


def load_assignments(path: Path) -> dict:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    assignments = {}
    for f in files:
        for a in _read_json(f):
            k = (a["current_tier"], a["key"])
            if k in assignments:
                raise SystemExit(f"error: {k} is assigned twice (second in {f})")
            if a["proposed_tier"] not in TIERS:
                raise SystemExit(f"error: {k}: bad proposed_tier {a['proposed_tier']}")
            assignments[k] = a["proposed_tier"]
    return assignments


def write_identity(root: Path, out: Path):
    snap = snapshot(root)
    _write_json(
        out,
        [
            {"current_tier": t, "key": k, "proposed_tier": t}
            for (t, k) in sorted(snap, key=lambda x: (TIERS.index(x[0]), x[1]))
        ],
    )
    print(f"wrote {len(snap)} identity assignments to {out}")


# --------------------------------------------------------------------------
# Template unification
# --------------------------------------------------------------------------


def _placeholder_count(template) -> int:
    return canon(template).count("${case.")


def _attr_key(node):
    if isinstance(node, str) and node.startswith(_ATTR_PREFIX) and node.endswith("}"):
        return node[len(_ATTR_PREFIX) : -1]
    return None


def unify_values(values: dict, src_template: dict, unified: dict) -> dict:
    """Re-express ``values`` so ``expand(unified, v) == expand(src_template, values)``.

    Only ``attributes`` placeholders may differ between the two templates: where
    ``unified`` has a placeholder and the source template has a literal (or a
    differently named placeholder), the literal / source attribute value becomes
    the new attribute.
    """
    out = copy.deepcopy(values)

    def walk(want, have, path):
        key = _attr_key(want)
        if key is not None:
            if have == want:
                return
            src_key = _attr_key(have)
            value = values["attributes"][src_key] if src_key else have
            attrs = out.setdefault("attributes", {})
            if key in attrs and attrs[key] != value:
                raise SystemExit(f"error: conflicting attribute {key!r} at {path}")
            attrs[key] = value
        elif isinstance(want, dict) and isinstance(have, dict):
            for k in want:
                if k in have:
                    walk(want[k], have[k], f"{path}/{k}")
        elif (
            isinstance(want, list) and isinstance(have, list) and len(want) == len(have)
        ):
            for i, (w, h) in enumerate(zip(want, have)):
                walk(w, h, f"{path}[{i}]")

    walk(unified, src_template, "")
    return out


# --------------------------------------------------------------------------
# Plan + apply
# --------------------------------------------------------------------------


def build_plan(root: Path, assignments: dict):
    sweeps, singles = load_tree(root)
    remaining = dict(assignments)

    by_topology = collections.defaultdict(dict)
    for (tier, op, topology), sw in sweeps.items():
        by_topology[(op, topology)][tier] = sw
    unified = {
        ot: max(
            d.values(),
            key=lambda s: (_placeholder_count(s.template), -TIERS.index(s.tier)),
        ).template
        for ot, d in by_topology.items()
    }

    dest = collections.defaultdict(
        list
    )  # (tier, op, topology) -> [(Sweep, case, order)]
    for (tier, op, topology), sw in sweeps.items():
        for i, case in enumerate(sw.sweep["cases"]):
            key = f"{op}/{topology}/{case['id']}"
            if (tier, key) not in remaining:
                raise SystemExit(f"error: no assignment for {tier}/{key}")
            new_tier = remaining.pop((tier, key))
            # Cases that stay keep their position; arrivals follow in source order.
            order = (new_tier != tier, TIERS.index(tier), i)
            dest[(new_tier, op, topology)].append((sw, case, order))

    single_moves = []
    for (tier, key), bundle_dir in singles.items():
        if (tier, key) not in remaining:
            raise SystemExit(f"error: no assignment for {tier}/{key}")
        single_moves.append((tier, remaining.pop((tier, key)), key, bundle_dir))

    if remaining:
        raise SystemExit(
            f"error: {len(remaining)} assignments match no case, e.g. {next(iter(remaining))}"
        )

    out = {}
    for (new_tier, op, topology), items in dest.items():
        items.sort(key=lambda x: x[2])
        template = unified[(op, topology)]
        cases, claims, goldens, seen = [], set(), [], set()
        for sw, case, _ in items:
            if case["id"] in seen:
                raise SystemExit(
                    f"error: id {case['id']!r} would appear twice in "
                    f"{new_tier}/{op}/{topology}; assign one of the copies elsewhere"
                )
            seen.add(case["id"])
            moved = copy.deepcopy(case)
            if canon(sw.template) != canon(template):
                moved["values"] = unify_values(case["values"], sw.template, template)
            cases.append(moved)
            for engine, support in sw.claims_of(case["id"]).items():
                claims.update(
                    Claim(case["id"], engine, arch, platform)
                    for arch, platforms in support.items()
                    for platform in platforms
                )
            if case.get("golden"):
                goldens.append((sw, case))
        # Canonical sidecar text, so the checked-in file round-trips through
        # support_sidecar.py whatever order the cases arrived in.
        support_text = render(Path("support.json"), claims) if claims else None
        existing = sweeps.get((new_tier, op, topology))
        unchanged = (
            existing is not None
            and canon(existing.template) == canon(template)
            and canon(existing.sweep["cases"]) == canon(cases)
        )
        first_sweep = items[0][0].sweep
        out[(new_tier, op, topology)] = {
            "template": template,
            "sweep": {
                **{k: v for k, v in first_sweep.items() if k != "cases"},
                "cases": cases,
            },
            "support": support_text,
            "goldens": goldens,
            "unchanged": unchanged,
        }
    return sweeps, out, single_moves


def _prune_empty_parents(path: Path, root: Path):
    parent = path.parent
    while parent != root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def apply_plan(root: Path, sweeps: dict, out: dict, single_moves: list):
    stage = root / ".retier_stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    try:
        # 1. Stage everything that is read from locations about to be rewritten.
        staged_golden = {}
        for (tier, op, topology), o in out.items():
            for sw, case in o["goldens"]:
                golden_dir = (sw.dir / case["golden"]["path"]).parent
                if golden_dir.name != case["id"]:
                    raise SystemExit(
                        f"error: golden path of {case['id']!r} is not golden/<case id>/…"
                    )
                target = stage / "golden" / tier / op / topology / case["id"]
                shutil.copytree(golden_dir, target)
                staged_golden[(tier, op, topology, case["id"])] = target
        for tier, new_tier, key, bundle_dir in single_moves:
            if tier != new_tier:
                shutil.copytree(bundle_dir, stage / "single" / new_tier / key)

        # 2. Remove what moves away.
        for tier, new_tier, key, bundle_dir in single_moves:
            if tier != new_tier:
                shutil.rmtree(bundle_dir)
                _prune_empty_parents(bundle_dir, root)
        for (tier, op, topology), sw in sweeps.items():
            plan = out.get((tier, op, topology))
            if plan is None or not plan["unchanged"]:
                shutil.rmtree(sw.dir)
                _prune_empty_parents(sw.dir, root)

        # 3. Write sweeps and place single bundles.
        for (tier, op, topology), plan in out.items():
            if plan["unchanged"]:
                continue
            d = root / tier / op / topology
            _write_json(d / "graph.template.json", plan["template"])
            _write_json(d / "sweep.json", plan["sweep"])
            if plan["support"]:
                (d / "support.json").write_bytes(plan["support"].encode("utf-8"))
            for (g_tier, g_op, g_topology, case_id), src in staged_golden.items():
                if (g_tier, g_op, g_topology) == (tier, op, topology):
                    shutil.copytree(src, d / "golden" / case_id)
        for tier, new_tier, key, _ in single_moves:
            if tier != new_tier:
                op, _, rel = key.partition("/")
                target = root / new_tier / op / rel
                if target.exists():
                    raise SystemExit(f"error: {target} already exists")
                shutil.copytree(stage / "single" / new_tier / key, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def verify(before: dict, after: dict, assignments: dict) -> list:
    errors = []
    if len(before) != len(after):
        errors.append(f"case count {len(before)} -> {len(after)}")
    for (old_tier, key), new_tier in assignments.items():
        b, a = before[(old_tier, key)], after.get((new_tier, key))
        if a is None:
            errors.append(f"{key}: missing from {new_tier}")
        elif b != a:
            field = next(f for f in b if b[f] != a[f])
            errors.append(f"{old_tier}/{key} -> {new_tier}: {field} changed")
    expected = {(new_tier, key) for (_, key), new_tier in assignments.items()}
    errors += [f"unexpected case {k}" for k in sorted(set(after) - expected)]
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--assignments", type=Path, help="assignment file or directory")
    ap.add_argument("--write-identity", type=Path, metavar="FILE")
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args()

    root = args.bundle_dir.resolve()
    if args.write_identity:
        write_identity(root, args.write_identity)
        return 0
    if not args.assignments:
        ap.error("--assignments is required")

    assignments = load_assignments(args.assignments)
    before = snapshot(root)
    sweeps, out, single_moves = build_plan(root, assignments)
    moved = sum(1 for (t, _), n in assignments.items() if t != n)
    rewritten = sum(1 for o in out.values() if not o["unchanged"])
    print(
        f"{len(before)} cases, {moved} change tier; "
        f"{rewritten} sweep dirs rewritten, "
        f"{sum(1 for t, n, _, _ in single_moves if t != n)} single bundles moved"
    )
    if not args.apply:
        print("dry run: nothing written (use --apply)")
        return 0

    apply_plan(root, sweeps, out, single_moves)
    errors = verify(before, snapshot(root), assignments)
    if errors:
        print(f"VERIFY FAILED: {len(errors)} errors", file=sys.stderr)
        for e in errors[:30]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("verify: every case is identical before and after, in its assigned tier")
    return 0


if __name__ == "__main__":
    sys.exit(main())
