#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Hop C — verify that the C++ graph -> bundle migration lost nothing.

Three-way reconciliation:
  1. census count  == captured count           (nothing lost in Hop A)
  2. captured count == placed bundle-case count (nothing lost in Hop B)
  3. for every placed case: expand(template, values) == Hop-A original graph
     AND case.metadata.seed/inputs == Hop-A .meta.json   (byte-exact)

Inputs:
  --census     census.json from census.py --graph-only
  --capture-dir  Hop A output (standalone per-case JSONs)
  --bundle-dir   Hop B output (template+sweep tree)

Exit 0 on full reconciliation, 1 on any discrepancy.

Usage::

    verify_migration.py --census census.json \\
        --capture-dir /tmp/captured --bundle-dir integration-test-bundles
"""

import argparse
import json
import sys
from pathlib import Path

from bundle_utils import (
    canon,
    canonical_uid_map,
    expand,
    inputs_by_name,
    remap_graph,
)


# --------------------------------------------------------------------------
# Load captured cases (Hop A output)
# --------------------------------------------------------------------------


def _load_captured(capture_dir: Path) -> dict:
    """Return {reference_source_key: (graph, meta)} for every captured case."""
    captured = {}
    if not capture_dir.is_dir():
        return captured
    for graph_path in sorted(capture_dir.rglob("*.json")):
        if graph_path.name.endswith(".meta.json"):
            continue
        case_dir = graph_path.parent
        case_name = case_dir.name
        if graph_path.stem != case_name:
            continue
        suite_rel = case_dir.parent.relative_to(capture_dir)
        suite_name = str(suite_rel)
        key = f"c++ integration suite: {suite_name}.{case_name}"
        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta_path = case_dir / f"{case_name}.meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        captured[key] = (graph, meta)
    return captured


# --------------------------------------------------------------------------
# Load placed bundles (Hop B output)
# --------------------------------------------------------------------------


def _load_placed(bundle_dir: Path) -> dict:
    """Return {reference_source_key: (expanded_graph, metadata)} for every placed case."""
    placed = {}
    if not bundle_dir.is_dir():
        return placed

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
        for case in sweep.get("cases", []):
            meta = case.get("metadata", {})
            key = meta.get("reference_source")
            if not key:
                continue
            expanded = expand(template, case.get("values", {}))
            placed[key] = (expanded, meta)

    for json_path in sorted(bundle_dir.rglob("*.json")):
        if json_path.name in ("sweep.json", "graph.template.json"):
            continue
        if json_path.name.endswith(".meta.json"):
            meta_path = json_path
            graph_path = json_path.parent / json_path.name.replace(
                ".meta.json", ".json"
            )
            if not graph_path.exists():
                continue
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            key = meta.get("reference_source")
            if key and key not in placed:
                placed[key] = (graph, meta)
    return placed


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------


def _compare_meta(
    captured_meta: dict,
    placed_meta: dict,
    captured_graph: dict,
    placed_graph: dict,
) -> list:
    """Return list of mismatch descriptions between captured and placed metadata.

    ``inputs`` is keyed by tensor UID, but place_bundles canonicalizes UIDs by
    tensor name, so the captured metadata (original UIDs) and placed metadata
    (canonical UIDs) will disagree on the raw keys even when they describe the
    same tensors. Re-key both by tensor name (inputs_by_name) so the fill specs
    are compared per logical tensor, independent of UID assignment.
    """
    mismatches = []
    cap_seed = captured_meta.get("seed")
    plc_seed = placed_meta.get("seed")
    if cap_seed is not None and plc_seed is not None and cap_seed != plc_seed:
        mismatches.append(f"seed: captured={cap_seed} placed={plc_seed}")

    cap_inputs = inputs_by_name(captured_meta, captured_graph)
    plc_inputs = inputs_by_name(placed_meta, placed_graph)
    if cap_inputs and canon(cap_inputs) != canon(plc_inputs):
        mismatches.append("inputs differ")
    return mismatches


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--census", type=Path, help="census.json from census.py --graph-only"
    )
    ap.add_argument(
        "--capture-dir", type=Path, required=True, help="Hop A output directory"
    )
    ap.add_argument(
        "--bundle-dir", type=Path, required=True, help="Hop B output directory"
    )
    args = ap.parse_args()

    errors = []

    # --- Step 1: load census denominator (optional) ---
    census_keys = None
    if args.census and args.census.exists():
        with open(args.census) as f:
            census = json.load(f)
        census_keys = set()
        for c in census.get("cases", []):
            census_keys.add(c["full"])
        print(f"  census denominator: {len(census_keys)}", file=sys.stderr)

    # --- Step 2: load captured cases ---
    captured = _load_captured(args.capture_dir)
    print(f"  captured cases:     {len(captured)}", file=sys.stderr)

    if census_keys is not None and len(captured) != len(census_keys):
        errors.append(
            f"census={len(census_keys)} vs captured={len(captured)} — "
            f"Hop A missed {len(census_keys) - len(captured)} cases"
        )

    # --- Step 3: load placed bundles ---
    placed = _load_placed(args.bundle_dir)
    # Count only placed cases that came from the C++ migration; pre-existing golden
    # bundles (e.g. PyTorch-sourced) live in the same tree but are not captured here.
    placed_migrated = {k for k in placed if k.startswith("c++ integration suite:")}
    print(
        f"  placed cases:       {len(placed)} "
        f"({len(placed_migrated)} from c++ migration)",
        file=sys.stderr,
    )

    if len(placed_migrated) != len(captured):
        errors.append(
            f"captured={len(captured)} vs placed(migrated)={len(placed_migrated)} — "
            f"Hop B lost {len(captured) - len(placed_migrated)} cases"
        )

    # --- Step 4: per-case graph + metadata verification ---
    graph_match = 0
    graph_mismatch = []
    meta_mismatch = []
    missing_in_placed = []

    for key, (cap_graph, cap_meta) in sorted(captured.items()):
        if key not in placed:
            missing_in_placed.append(key)
            continue
        plc_graph, plc_meta = placed[key]
        cap_uid_map = canonical_uid_map(cap_graph)
        cap_canon = remap_graph(cap_graph, cap_uid_map)
        plc_canon = remap_graph(plc_graph, canonical_uid_map(plc_graph))
        if canon(cap_canon) != canon(plc_canon):
            graph_mismatch.append(key)
        else:
            graph_match += 1
        meta_issues = _compare_meta(cap_meta, plc_meta, cap_graph, plc_graph)
        if meta_issues:
            meta_mismatch.append((key, meta_issues))

    # --- Report ---
    print(f"\n== verify_migration ==", file=sys.stderr)
    print(f"  graph match:     {graph_match}", file=sys.stderr)
    print(f"  graph mismatch:  {len(graph_mismatch)}", file=sys.stderr)
    print(f"  meta mismatch:   {len(meta_mismatch)}", file=sys.stderr)
    print(f"  missing placed:  {len(missing_in_placed)}", file=sys.stderr)

    if missing_in_placed:
        errors.append(f"{len(missing_in_placed)} cases missing from placed bundles")
        for k in missing_in_placed[:20]:
            print(f"    MISSING: {k}", file=sys.stderr)

    if graph_mismatch:
        errors.append(f"{len(graph_mismatch)} cases have graph mismatches")
        for k in graph_mismatch[:20]:
            print(f"    GRAPH MISMATCH: {k}", file=sys.stderr)

    if meta_mismatch:
        errors.append(f"{len(meta_mismatch)} cases have metadata mismatches")
        for k, issues in meta_mismatch[:20]:
            print(f"    META MISMATCH: {k}: {'; '.join(issues)}", file=sys.stderr)

    if errors:
        print(f"\n  FAIL: {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"    - {e}", file=sys.stderr)
        return 1

    total = graph_match
    print(f"\n  OK D={total}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
