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

Usage:

    python3 dnn-providers/integration-tests/scripts/support_features/coverage.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from support_features import FeatureError  # noqa: E402
from support_features import registry  # noqa: E402
from support_matrix_units import ClaimUnit, collect_units, report_error  # noqa: E402

DEFAULT_BUNDLES_DIR = SCRIPTS_DIR.parent / "integration-test-bundles"


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
    args = parser.parse_args(argv)

    if not args.bundles_dir.is_dir():
        print(f"error: {args.bundles_dir} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []
    units = collect_units(args.bundles_dir, errors)
    coverage = compute_coverage(units, errors)
    if errors:
        # A graph left out of a share makes the share wrong, so none is shown.
        print(
            f"error: {len(errors)} defect(s); no coverage reported",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(render_markdown(coverage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
