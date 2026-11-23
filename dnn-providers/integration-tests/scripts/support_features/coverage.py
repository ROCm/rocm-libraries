#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature coverage per op, from the bundle tree and its support claims.

For every catalogued op this reports two shares of the op's API features:

- **tested** -- features at least one bundle graph exercises, claimed or not;
- **claimed on <arch>** -- features at least one graph exercises that some
  engine claims on that arch (any engine, any platform).

A feature counts once however many graphs reach it, and features count singly,
not in combination: bf16 and causal each being claimed says nothing about bf16
*with* causal. The per-feature table below the summary carries the graph
counts.

A graph holding several nodes of one op counts once for it, with the union of
their features. A fused graph counts toward each catalogued op it contains.

Nothing is written or committed: like the support matrix, this is a view of
the committed graphs and sidecars, regenerated on demand.

To check a number, list the graphs behind it and open them: ``--graphs``
names every graph exercising one feature, and ``--engine``, ``--arch`` and
``--platform`` narrow it to the graphs claimed there, the evidence for one
cell of the support matrix by API option.

Usage:

    python3 dnn-providers/integration-tests/scripts/support_features/coverage.py
    python3 dnn-providers/integration-tests/scripts/support_features/coverage.py \\
        --graphs Sdpa dtype:bf16 --arch gfx942 [--engine E] [--platform linux]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from bundle_discovery import DEFAULT_BUNDLES_DIR  # noqa: E402
from support_features import FeatureError  # noqa: E402
from support_features import registry  # noqa: E402
from support_matrix_units import (  # noqa: E402
    ClaimUnit,
    collect_units,
    node_type_label,
    report_error,
    write_stdout,
)


@dataclass
class OpCoverage:
    """Feature counts for one catalogued op."""

    node_type: str
    features: tuple[str, ...]
    graphs: int = 0
    # arch -> graphs claimed on it
    claimed_graphs: Counter = field(default_factory=Counter)
    # feature -> graphs exercising it
    tested: Counter = field(default_factory=Counter)
    # arch -> feature -> claimed graphs exercising it
    claimed: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def tested_count(self) -> int:
        return sum(1 for f in self.features if self.tested[f])

    def claimed_count(self, arch: str) -> int:
        per_feature = self.claimed.get(arch, Counter())
        return sum(1 for f in self.features if per_feature[f])


@dataclass
class Coverage:
    ops: dict[str, OpCoverage]
    # Every arch any unit claims, so an op with no claims on one shows 0%.
    archs: list[str]


def unit_location(unit: ClaimUnit) -> str:
    return unit.bundle + (f" case '{unit.case_id}'" if unit.case_id else "")


def tensors_by_uid(graph: dict) -> dict:
    return {
        t["uid"]: t
        for t in graph.get("tensors") or []
        if isinstance(t, dict) and "uid" in t
    }


def unit_features(unit: ClaimUnit, errors: list[str] | None) -> dict[str, set[str]]:
    """Node type -> features the unit's graph exercises, per op.

    A node the catalog cannot place is reported, and its op is left out for
    this unit.
    """
    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for node in unit.graph.get("nodes") or []:
        if isinstance(node, dict):
            nodes_by_type[node.get("type") or ""].append(node)

    tensors = tensors_by_uid(unit.graph)
    found: dict[str, set[str]] = {}
    for node_type, nodes in nodes_by_type.items():
        try:
            module = registry.op_module(node_type)
            features: set[str] = set()
            for node in nodes:
                features |= module.extract(node, tensors)
        except (FeatureError, KeyError, TypeError, IndexError) as exc:
            report_error(errors, f"{unit_location(unit)}: {node_type}: {exc}")
            continue
        unknown = features - {f.id for f in module.FEATURES}
        if unknown:
            report_error(
                errors,
                f"{unit_location(unit)}: {node_type}: extract() returned "
                f"features its catalog does not list: {sorted(unknown)}",
            )
            continue
        found[node_type] = features
    return found


def compute_coverage(
    units: list[ClaimUnit], errors: list[str] | None = None
) -> Coverage:
    ops = {
        node_type: OpCoverage(node_type, tuple(f.id for f in module.FEATURES))
        for node_type, module in registry.OPS.items()
    }
    archs: set[str] = set()

    for unit in units:
        unit_archs = {arch for pairs in unit.claims.values() for arch, _ in pairs}
        archs |= unit_archs
        for node_type, features in unit_features(unit, errors).items():
            # unit_features returns only registered types, all keys of ops.
            op = ops[node_type]
            op.graphs += 1
            op.tested.update(features)
            for arch in unit_archs:
                op.claimed_graphs[arch] += 1
                op.claimed[arch].update(features)

    return Coverage(ops=ops, archs=sorted(archs))


# --------------------------------------------------------------------------
# Drill-down: the graphs behind one feature
# --------------------------------------------------------------------------


def resolve_node_type(op: str) -> str:
    """``Sdpa`` or ``SdpaAttributes`` -> ``SdpaAttributes``, or raise."""
    for node_type in registry.OPS:
        if op in (node_type, node_type_label(node_type)):
            return node_type
    raise ValueError(f"unknown op {op!r}; catalogued: {sorted(registry.OPS)}")


def matching_claims(
    unit: ClaimUnit,
    engine: str | None,
    arch: str | None,
    platform: str | None,
) -> list[tuple[str, str, str]]:
    """The unit's (engine, arch, platform) claims that pass every filter."""
    return sorted(
        (e, a, p)
        for e, pairs in unit.claims.items()
        for a, p in pairs
        if engine in (None, e) and arch in (None, a) and platform in (None, p)
    )


def graphs_with(
    units: list[ClaimUnit],
    node_type: str,
    feature: str,
    engine: str | None = None,
    arch: str | None = None,
    platform: str | None = None,
    errors: list[str] | None = None,
) -> list[tuple[ClaimUnit, list[tuple[str, str, str]]]]:
    """Graphs exercising ``feature`` of ``node_type``, with their claims.

    With no filter, every graph that exercises the feature, claimed or not:
    the *tested* count. With any of ``engine``, ``arch``, ``platform``, only
    graphs claimed where all the given filters hold: the graphs behind one
    cell of the support matrix by API option.
    """
    filtered = (engine, arch, platform) != (None, None, None)
    found = []
    for unit in units:
        if feature not in unit_features(unit, errors).get(node_type, ()):
            continue
        claims = matching_claims(unit, engine, arch, platform)
        if filtered and not claims:
            continue
        found.append((unit, claims))
    return sorted(found, key=lambda item: unit_location(item[0]))


def render_graph_list(
    found: list[tuple[ClaimUnit, list[tuple[str, str, str]]]],
) -> str:
    lines = []
    for unit, claims in found:
        tiers = ", ".join(sorted(unit.tiers))
        claimed = ", ".join(f"{e} {a}/{p}" for e, a, p in claims) or "not claimed"
        lines.append(f"{unit_location(unit)} [{tiers}]: {claimed}")
    lines.append(f"{len(found)} graph(s)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def share(count: int, total: int) -> str:
    if not total:
        return "-"
    return f"{count}/{total} ({100 * count / total:.0f}%)"


def render_markdown(coverage: Coverage) -> str:
    archs = coverage.archs
    lines = [
        "# API feature coverage per op",
        "",
        "Share of each op's API features (catalogued from the flatbuffers "
        "schema) that bundle graphs exercise (*tested*), and that claimed "
        "graphs exercise on each arch. Features count singly, not in "
        "combination.",
        "",
        "| Op | Graphs | Tested | " + " | ".join(archs) + " |",
        "|---|---:|---:|" + "---:|" * len(archs),
    ]
    for node_type in sorted(coverage.ops):
        op = coverage.ops[node_type]
        total = len(op.features)
        cells = [share(op.claimed_count(arch), total) for arch in archs]
        lines.append(
            f"| {node_type} | {op.graphs} | {share(op.tested_count(), total)} | "
            + " | ".join(cells)
            + " |"
        )

    for node_type in sorted(coverage.ops):
        op = coverage.ops[node_type]
        lines += [
            "",
            f"## {node_type}",
            "",
            "Graphs exercising each feature; per arch, only claimed graphs.",
            "",
            "| Feature | Graphs | " + " | ".join(archs) + " |",
            "|---|---:|" + "---:|" * len(archs),
        ]
        lines.append(
            f"| *(all graphs)* | {op.graphs} | "
            + " | ".join(str(op.claimed_graphs[arch]) for arch in archs)
            + " |"
        )
        for feature in op.features:
            per_arch = [str(op.claimed.get(arch, Counter())[feature]) for arch in archs]
            lines.append(
                f"| {feature} | {op.tested[feature]} | " + " | ".join(per_arch) + " |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report API feature coverage per op from the bundle tree "
        "and its support claims."
    )
    parser.add_argument(
        "--bundles-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLES_DIR,
        help=f"Bundle tree to scan (default: {DEFAULT_BUNDLES_DIR}).",
    )
    parser.add_argument(
        "--graphs",
        nargs=2,
        metavar=("OP", "FEATURE"),
        help="List the graphs exercising one feature instead of the report, "
        "e.g. --graphs Sdpa dtype:bf16. With --engine, --arch or --platform, "
        "only graphs claimed there: the evidence behind one cell.",
    )
    parser.add_argument("--engine", help="With --graphs: claimed by this engine.")
    parser.add_argument("--arch", help="With --graphs: claimed on this arch.")
    parser.add_argument(
        "--platform", choices=("linux", "windows"), help="With --graphs: on this OS."
    )
    args = parser.parse_args(argv)

    if not args.bundles_dir.is_dir():
        print(f"error: {args.bundles_dir} is not a directory", file=sys.stderr)
        return 2
    if not args.graphs and (args.engine or args.arch or args.platform):
        print("error: --engine, --arch and --platform need --graphs", file=sys.stderr)
        return 2
    if args.graphs:
        op, feature = args.graphs
        try:
            node_type = resolve_node_type(op)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        catalogued = [f.id for f in registry.op_module(node_type).FEATURES]
        if feature not in catalogued:
            print(
                f"error: {node_type} has no feature {feature!r}; "
                f"catalogued: {catalogued}",
                file=sys.stderr,
            )
            return 2

    errors: list[str] = []
    units = collect_units(args.bundles_dir, errors)
    if args.graphs:
        found = graphs_with(
            units, node_type, feature, args.engine, args.arch, args.platform, errors
        )
    else:
        coverage = compute_coverage(units, errors)
    if errors:
        # A graph left out of a share makes the share wrong, so none is shown.
        print(
            f"error: {len(errors)} defect(s); no coverage reported",
            file=sys.stderr,
        )
        return 1
    if args.graphs:
        write_stdout(render_graph_list(found))
    else:
        write_stdout(render_markdown(coverage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
