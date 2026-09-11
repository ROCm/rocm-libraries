#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Place captured graph bundles into the template+sweep format (ALMIOPEN-2221, AC #8).

Hop B of the migration pipeline:
    C++ graph test --(Hop A: --capture-bundles)--> standalone bundle
                   --(Hop B: this script)--------> template+sweep

Reads the flat per-case output of ``--capture-bundles`` and groups graphs by
STRUCTURE (topology), collapsing cases that share the same graph skeleton and
differ only in per-case data (dims/strides/dtype/inline values/node attributes)
into one ``graph.template.json`` + ``sweep.json``.

Conforms to the Compressed Template Sweeps spec:
    integration-test-bundles/{Tier}/{Operation}/{TopologyName}/
        graph.template.json
        sweep.json

Design principle -- derive, don't classify. "Structural" is defined narrowly and
mechanically (node types + wiring + tensor set); everything else is a per-case
knob. Round-trip verification is the correctness proof: any bad merge fails
loudly and falls back to a standalone single-graph bundle. Nothing is dropped.

Usage::

    place_bundles.py --capture-dir <path> --output-dir <path> [--dry-run] [--no-verify]
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bundle_utils import (
    DTYPE_TOKEN,
    NODE_STRUCTURAL,
    TENSOR_ALWAYS,
    TENSOR_IF_VARIES,
    TIER_MAP,
    TOP_LEVEL_IF_VARIES,
    ambiguous_attr_keys,
    assign_case_ids,
    canon,
    canonical_uid_map_by_name,
    derive_operation,
    expand,
    infer_layout,
    node_attr_items,
    raw_node_attrs,
    remap_graph,
    remap_meta_inputs,
    sanitize,
    skeleton_hash,
    tensors_by_uid,
)


@dataclass
class CapturedCase:
    suite: str
    case_name: str
    graph_path: Path
    meta_path: Path
    graph: dict
    meta: dict
    tier: str = "quick"
    original_graph: dict = None


@dataclass
class Bucket:
    """A set of captured cases sharing one graph skeleton."""

    skeleton_hash: str
    tier: str
    operation: str
    cases: list = field(default_factory=list)
    topology_name: str = ""


@dataclass
class Stats:
    cases_found: int = 0
    buckets: int = 0
    sweeps_written: int = 0
    sweep_cases: int = 0
    standalone_written: int = 0
    verify_pass: int = 0
    verify_fail: int = 0
    errors: list = field(default_factory=list)


# --------------------------------------------------------------------------
# 1. Discovery
# --------------------------------------------------------------------------


def discover_captures(capture_dir: Path) -> list[CapturedCase]:
    """Walk capture directory and load all captured cases.

    C++ capture writes <capture-dir>/<suiteName>/<safeCaseName>/<safeCaseName>.json
    where suiteName may contain '/' (e.g. 'Smoke/IntegrationGpuConvFp32'). We find
    cases by locating .json files (excluding .meta.json) whose stem matches the
    parent directory name.
    """
    cases = []
    if not capture_dir.is_dir():
        print(
            f"place_bundles: capture dir does not exist: {capture_dir}", file=sys.stderr
        )
        return cases

    for graph_path in sorted(capture_dir.rglob("*.json")):
        if graph_path.name.endswith(".meta.json"):
            continue

        case_dir = graph_path.parent
        case_name = case_dir.name
        if graph_path.stem != case_name:
            continue

        suite_rel = case_dir.parent.relative_to(capture_dir)
        suite_name = str(suite_rel)

        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: bad graph {graph_path}: {e}", file=sys.stderr)
            continue

        meta_path = case_dir / f"{case_name}.meta.json"
        meta = {"format_version": 1}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARN: bad meta {meta_path}: {e}", file=sys.stderr)

        tier = TIER_MAP.get(suite_rel.parts[0], "quick") if suite_rel.parts else "quick"

        cases.append(
            CapturedCase(
                suite=suite_name,
                case_name=case_name,
                graph_path=graph_path,
                meta_path=meta_path,
                graph=graph,
                meta=meta,
                tier=tier,
            )
        )
    return cases


# --------------------------------------------------------------------------
# 2. Knob detection + template / sweep construction
# --------------------------------------------------------------------------


def detect_and_build(bucket: Bucket):
    """Analyze a bucket, returning (template, sweep_cases, error)."""
    cases = bucket.cases
    rep = cases[0].graph
    ambiguous = ambiguous_attr_keys(rep)

    top_varies = set()
    for fld in TOP_LEVEL_IF_VARIES:
        vals = {json.dumps(c.graph.get(fld)) for c in cases}
        if len(vals) > 1:
            top_varies.add(fld)

    rep_tensors = tensors_by_uid(rep)
    tensor_value_varies = {}
    for uid in rep_tensors:
        varies = set()
        for fld in TENSOR_IF_VARIES:
            vals = {
                json.dumps(tensors_by_uid(c.graph).get(uid, {}).get(fld)) for c in cases
            }
            if len(vals) > 1:
                varies.add(fld)
        tensor_value_varies[uid] = varies

    attr_varies = set()
    for ni in range(len(rep.get("nodes", []))):
        for attr_key, _ in node_attr_items(rep["nodes"][ni], ni, ambiguous):
            vals = set()
            for c in cases:
                nodes = c.graph.get("nodes", [])
                if ni < len(nodes):
                    d = dict(node_attr_items(nodes[ni], ni, ambiguous))
                    vals.add(json.dumps(d.get(attr_key)))
            if len(vals) > 1:
                attr_varies.add(attr_key)

    template = copy.deepcopy(rep)
    for fld in top_varies:
        template[fld] = f"${{case.{fld}}}"
    for t in template.get("tensors", []):
        uid = t.get("uid")
        for fld in TENSOR_ALWAYS:
            if fld in t:
                t[fld] = f"${{case.{fld}}}"
        for fld in tensor_value_varies.get(uid, set()):
            if fld in t:
                t[fld] = f"${{case.{fld}}}"
    for ni, node in enumerate(template.get("nodes", [])):
        _apply_attr_placeholders(node, ni, ambiguous, attr_varies)

    sweep_cases = []
    for c in cases:
        values = {}
        for fld in top_varies:
            values[fld] = c.graph.get(fld)
        tv = []
        ctensors = tensors_by_uid(c.graph)
        for uid in sorted(rep_tensors):
            entry = {"uid": uid}
            src = ctensors.get(uid, {})
            for fld in TENSOR_ALWAYS:
                if fld in src:
                    entry[fld] = src[fld]
            for fld in tensor_value_varies.get(uid, set()):
                if fld in src:
                    entry[fld] = src[fld]
            tv.append(entry)
        values["tensors"] = tv
        attrs = {}
        for ni, node in enumerate(c.graph.get("nodes", [])):
            for attr_key, v in node_attr_items(node, ni, ambiguous):
                if attr_key in attr_varies:
                    attrs[attr_key] = v
        if attrs:
            values["attributes"] = attrs

        meta = dict(c.meta)
        meta.setdefault("format_version", 1)
        meta["reference_source"] = f"c++ integration suite: {c.suite}.{c.case_name}"

        sweep_cases.append(
            {"id": None, "values": values, "metadata": meta, "_origin": c}
        )
    return template, sweep_cases, None


def _apply_attr_placeholders(
    node: dict, node_index: int, ambiguous: set, attr_varies: set
):
    def nskey(base):
        return f"n{node_index}__{base}" if base in ambiguous else base

    for k in list(node.keys()):
        if k in NODE_STRUCTURAL or k in ("inputs", "outputs", "parameters"):
            continue
        if nskey(k) in attr_varies:
            node[k] = f"${{case.attributes.{nskey(k)}}}"
    params = node.get("parameters")
    if isinstance(params, dict):
        for k in list(params.keys()):
            if nskey(f"parameters__{k}") in attr_varies:
                params[k] = f"${{case.attributes.{nskey(f'parameters__{k}')}}}"
    for section in ("inputs", "outputs"):
        sec = node.get(section, {})
        if isinstance(sec, dict):
            for k in list(sec.keys()):
                if not k.endswith("_tensor_uid") and nskey(k) in attr_varies:
                    sec[k] = f"${{case.attributes.{nskey(k)}}}"


# --------------------------------------------------------------------------
# 3. Verify (round-trip: expand template, compare to original)
# --------------------------------------------------------------------------


def verify_case(template: dict, entry: dict) -> bool:
    expanded = expand(template, entry["values"])
    original = entry["_origin"].graph
    return canon(expanded) == canon(original)


# --------------------------------------------------------------------------
# 4. Writers
# --------------------------------------------------------------------------


def write_sweep(
    target: Path, bucket: Bucket, template: dict, sweep_cases: list, dry_run: bool
):
    out_dir = target / bucket.tier / bucket.operation / bucket.topology_name
    cases_out = []
    for e in sweep_cases:
        cases_out.append(
            {"id": e["id"], "values": e["values"], "metadata": e["metadata"]}
        )
    sweep = {"version": 1, "cases": cases_out}
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "graph.template.json", "w") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        with open(out_dir / "sweep.json", "w") as f:
            json.dump(sweep, f, indent=2)
            f.write("\n")
    return out_dir


def write_standalone(target: Path, case: CapturedCase, reason: str, dry_run: bool):
    graph = case.original_graph if case.original_graph is not None else case.graph
    op = derive_operation(graph)
    tensors = tensors_by_uid(graph)
    first = tensors[min(tensors)] if tensors else {}
    dt = graph.get("io_data_type") or first.get("data_type") or "unknown"
    dtok = DTYPE_TOKEN.get(str(dt).lower(), str(dt).lower())
    layout = infer_layout(first.get("dims"), first.get("strides")) or "any"
    bundle_name = sanitize(case.case_name)
    out_dir = target / case.tier / op / layout / dtok / bundle_name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{bundle_name}.json", "w") as f:
            json.dump(graph, f, indent=2)
            f.write("\n")
        meta = dict(case.meta)
        meta.setdefault("format_version", 1)
        meta["reference_source"] = (
            f"c++ integration suite: {case.suite}.{case.case_name}"
        )
        meta["standalone_reason"] = reason
        with open(out_dir / f"{bundle_name}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
    return out_dir


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------


def assign_topology_names(buckets: list):
    by_op = defaultdict(list)
    for b in buckets:
        by_op[(b.tier, b.operation)].append(b)
    for _, group in by_op.items():
        group.sort(key=lambda b: b.skeleton_hash)
        for i, b in enumerate(group):
            b.topology_name = "Default" if i == 0 else f"Variant{i + 1}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="root of --capture-bundles output",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="root of output tree (e.g. integration-test-bundles/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip round-trip verification (NOT recommended)",
    )
    args = ap.parse_args()

    stats = Stats()

    cases = discover_captures(args.capture_dir)
    stats.cases_found = len(cases)
    if not cases:
        print("place_bundles: no captured cases found", file=sys.stderr)
        return 1

    # Canonicalize UIDs by tensor name so that the same logical tensor carries
    # the same UID in every case of a topology. The C++ builder auto-assigns
    # UIDs non-deterministically (GraphTensorIds.hpp iterates an unordered_set),
    # so without this, sibling cases disagree on which UID is Y / MEAN / momentum
    # and can never share a template. Graph and metadata inputs are remapped in
    # lockstep so each tensor keeps its fill spec (canonical_uid_map_by_name).
    for c in cases:
        uid_map = canonical_uid_map_by_name(c.graph)
        c.graph = remap_graph(c.graph, uid_map)
        c.meta = remap_meta_inputs(c.meta, uid_map)

    grouped: dict = defaultdict(list)
    for c in cases:
        h = skeleton_hash(c.graph)
        c.original_graph = c.graph
        grouped[(c.tier, h)].append(c)

    buckets = []
    for (tier, h), group in grouped.items():
        buckets.append(
            Bucket(
                skeleton_hash=h,
                tier=tier,
                operation=derive_operation(group[0].graph),
                cases=group,
            )
        )
    stats.buckets = len(buckets)
    assign_topology_names(buckets)

    topology_map = []

    for bucket in sorted(buckets, key=lambda b: (b.tier, b.operation, b.topology_name)):
        if len(bucket.cases) == 1:
            reason = "single-case topology (no sweep benefit)"
            out = write_standalone(
                args.output_dir, bucket.cases[0], reason, args.dry_run
            )
            stats.standalone_written += 1
            print(f"  standalone: {out}  ({reason})", file=sys.stderr)
            continue

        template, sweep_cases, err = detect_and_build(bucket)
        if err is not None:
            for c in bucket.cases:
                write_standalone(args.output_dir, c, err, args.dry_run)
                stats.standalone_written += 1
            print(
                f"  SKIP->standalone {bucket.operation}/{bucket.topology_name} "
                f"({len(bucket.cases)} cases): {err}",
                file=sys.stderr,
            )
            continue

        assign_case_ids(sweep_cases)

        if not args.no_verify:
            failed = [e for e in sweep_cases if not verify_case(template, e)]
            if failed:
                for c in bucket.cases:
                    write_standalone(
                        args.output_dir, c, "round-trip verify failed", args.dry_run
                    )
                    stats.standalone_written += 1
                stats.verify_fail += len(failed)
                print(
                    f"  VERIFY FAIL {bucket.operation}/{bucket.topology_name}: "
                    f"{len(failed)}/{len(sweep_cases)} cases -> standalone",
                    file=sys.stderr,
                )
                continue
            stats.verify_pass += len(sweep_cases)

        out = write_sweep(args.output_dir, bucket, template, sweep_cases, args.dry_run)
        stats.sweeps_written += 1
        stats.sweep_cases += len(sweep_cases)
        topology_map.append(
            {
                "skeleton_hash": bucket.skeleton_hash,
                "tier": bucket.tier,
                "operation": bucket.operation,
                "topology_name": bucket.topology_name,
                "path": str(out),
                "case_count": len(sweep_cases),
            }
        )
        print(
            f"  sweep: {bucket.tier}/{bucket.operation}/{bucket.topology_name}  "
            f"({len(sweep_cases)} graphs -> 1 sweep)",
            file=sys.stderr,
        )

    if not args.dry_run and topology_map:
        report_dir = args.output_dir.parent / ".migration_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        map_path = report_dir / "topology_map.json"
        with open(map_path, "w") as f:
            json.dump({"version": 1, "topologies": topology_map}, f, indent=2)
            f.write("\n")
        print(f"  topology map:      {map_path}", file=sys.stderr)

    print("== place_bundles ==", file=sys.stderr)
    print(f"  capture dir:       {args.capture_dir}", file=sys.stderr)
    print(f"  cases found:       {stats.cases_found}", file=sys.stderr)
    print(f"  skeletons:         {stats.buckets}", file=sys.stderr)
    print(
        f"  sweeps written:    {stats.sweeps_written} ({stats.sweep_cases} cases)",
        file=sys.stderr,
    )
    print(f"  standalone:        {stats.standalone_written}", file=sys.stderr)
    if not args.no_verify:
        print(f"  verify pass:       {stats.verify_pass}", file=sys.stderr)
        print(f"  verify fail:       {stats.verify_fail}", file=sys.stderr)
    total_out = stats.sweep_cases + stats.standalone_written
    print(f"  accounted graphs:  {total_out} / {stats.cases_found}", file=sys.stderr)

    if total_out != stats.cases_found:
        print(
            f"  ERROR: {stats.cases_found - total_out} graphs unaccounted for!",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
