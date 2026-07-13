#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Import a single graph JSON into the bundle tree with duplicate detection.

The incremental, idempotent counterpart to the batch ``place_bundles.py``:
safe to run repeatedly, never creates duplicate cases.

Duplicate detection (two levels):
  * Exact dup — expand each existing case; if any expands byte-identical
    to the input graph AND same seed/inputs -> DUPLICATE (skip by default).
  * Structural dup, new knobs — same skeleton hash, different dims/strides/
    dtype/attrs -> legitimate new case: append to existing sweep.json.
  * No structural match — create a new single-case template+sweep.

Dup policy: default is skip-and-report (idempotent, safe to re-run).
  --force   appends even if an exact dup exists.
  --strict  turns an exact dup into a non-zero exit (for CI gates).

Usage::

    import_graph.py --graph case.json --bundle-dir integration_test_bundles/ \\
        [--tier quick] [--meta ported_from="..."] [--dry-run] [--strict] [--force]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

from bundle_utils import (
    TENSOR_ALWAYS,
    TENSOR_IF_VARIES,
    TOP_LEVEL_IF_VARIES,
    assign_case_ids,
    canon,
    canonical_uid_map,
    derive_operation,
    expand,
    infer_layout,
    remap_graph,
    sanitize,
    skeleton_hash,
    tensors_by_uid,
)


# --------------------------------------------------------------------------
# Scan existing bundles
# --------------------------------------------------------------------------


def _index_existing(bundle_dir: Path) -> dict:
    """Build {skeleton_hash: [(template, sweep, sweep_path)]} from the tree."""
    index = {}
    if not bundle_dir.is_dir():
        return index
    for sweep_path in sorted(bundle_dir.rglob("sweep.json")):
        template_path = sweep_path.parent / "graph.template.json"
        if not template_path.exists():
            continue
        try:
            with open(template_path) as f:
                template = json.load(f)
            with open(sweep_path) as f:
                sweep = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        expanded_rep = expand(template, sweep["cases"][0].get("values", {}))
        h = skeleton_hash(expanded_rep)
        index.setdefault(h, []).append((template, sweep, sweep_path))
    return index


# --------------------------------------------------------------------------
# Build single-case values from a graph (inverse of template expansion)
# --------------------------------------------------------------------------


def _extract_values(graph: dict, template: dict) -> dict:
    """Extract per-case values from a concrete graph given its template."""
    values = {}
    for fld in TOP_LEVEL_IF_VARIES:
        if isinstance(template.get(fld), str) and template[fld].startswith("${case."):
            values[fld] = graph.get(fld)

    tgraph = tensors_by_uid(graph)
    tv = []
    for tt in template.get("tensors", []):
        uid = tt.get("uid")
        src = tgraph.get(uid, {})
        entry = {"uid": uid}
        for fld in TENSOR_ALWAYS + TENSOR_IF_VARIES:
            if isinstance(tt.get(fld), str) and tt[fld].startswith("${case."):
                if fld in src:
                    entry[fld] = src[fld]
        tv.append(entry)
    values["tensors"] = tv
    return values


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--graph", type=Path, required=True, help="path to the graph JSON to import"
    )
    ap.add_argument(
        "--bundle-dir", type=Path, required=True, help="root of the bundle tree"
    )
    ap.add_argument("--tier", default="quick", help="tier folder (default: quick)")
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="key=value metadata pairs (repeatable)",
    )
    ap.add_argument("--seed", type=int, default=None, help="global seed for metadata")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument(
        "--force", action="store_true", help="append even if exact dup exists"
    )
    ap.add_argument(
        "--strict", action="store_true", help="exit non-zero on exact dup (CI mode)"
    )
    args = ap.parse_args()

    try:
        with open(args.graph) as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"import_graph: cannot read {args.graph}: {e}", file=sys.stderr)
        return 1

    graph_canon = remap_graph(graph, canonical_uid_map(graph))
    h = skeleton_hash(graph_canon)
    op = derive_operation(graph_canon)
    graph_json = canon(graph_canon)

    meta = {"format_version": 1}
    for kv in args.meta:
        if "=" in kv:
            k, v = kv.split("=", 1)
            meta[k] = v
    if args.seed is not None:
        meta["seed"] = args.seed

    print(f"  skeleton hash: {h}", file=sys.stderr)
    print(f"  operation:     {op}", file=sys.stderr)

    index = _index_existing(args.bundle_dir)
    matches = index.get(h, [])

    # --- Check for exact duplicates ---
    for template, sweep, sweep_path in matches:
        for case in sweep.get("cases", []):
            expanded = expand(template, case.get("values", {}))
            expanded_canon = remap_graph(expanded, canonical_uid_map(expanded))
            if canon(expanded_canon) == graph_json:
                existing_meta = case.get("metadata", {})
                seed_match = meta.get("seed") is None or existing_meta.get(
                    "seed"
                ) == meta.get("seed")
                inputs_match = canon(meta.get("inputs", {})) == canon(
                    existing_meta.get("inputs", {})
                )
                if seed_match and inputs_match:
                    print(f"  DUPLICATE of {sweep_path}:{case['id']}", file=sys.stderr)
                    if args.strict:
                        return 1
                    if not args.force:
                        print("  skipped (use --force to append)", file=sys.stderr)
                        return 0

    # --- Structural match: append to existing sweep ---
    if matches:
        template, sweep, sweep_path = matches[0]
        values = _extract_values(graph_canon, template)
        new_case = {"id": None, "values": values, "metadata": meta}
        all_cases = sweep.get("cases", []) + [new_case]
        assign_case_ids(all_cases)

        expanded = expand(template, values)
        if canon(expanded) != canon(graph_canon):
            print("  ERROR: round-trip verify failed after extraction", file=sys.stderr)
            return 1

        if not args.dry_run:
            sweep["cases"] = [
                {"id": c["id"], "values": c["values"], "metadata": c["metadata"]}
                for c in all_cases
            ]
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)
                f.write("\n")
        print(f"  appended case '{new_case['id']}' to {sweep_path}", file=sys.stderr)
        return 0

    # --- No structural match: create new template+sweep ---
    template = copy.deepcopy(graph_canon)
    for fld in TOP_LEVEL_IF_VARIES:
        if fld in template:
            template[fld] = f"${{case.{fld}}}"
    for t in template.get("tensors", []):
        for fld in TENSOR_ALWAYS:
            if fld in t:
                t[fld] = f"${{case.{fld}}}"

    values = _extract_values(graph_canon, template)

    expanded = expand(template, values)
    if canon(expanded) != canon(graph_canon):
        print("  ERROR: round-trip verify failed for new template", file=sys.stderr)
        return 1

    tmap = tensors_by_uid(graph_canon)
    first = tmap[min(tmap)] if tmap else {}
    layout = infer_layout(first.get("dims"), first.get("strides")) or "Default"
    topo_name = sanitize(layout).replace(" ", "_").title() or "Default"

    out_dir = args.bundle_dir / args.tier / op / topo_name
    case_entry = {"id": "case", "values": values, "metadata": meta}
    sweep_out = {"version": 1, "cases": [case_entry]}

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "graph.template.json", "w") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        with open(out_dir / "sweep.json", "w") as f:
            json.dump(sweep_out, f, indent=2)
            f.write("\n")
    print(f"  created new bundle: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
