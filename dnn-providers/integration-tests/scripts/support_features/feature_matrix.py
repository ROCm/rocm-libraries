#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The support matrix by API feature: what share of each op's features is claimed.

The graph-based matrix answers "how many of the op's graphs does an engine
claim". This one answers "how much of the op's API does an engine claim": a
cell is the share of the op's catalogued features (``support_features/<op>.py``)
that at least one claimed graph exercises on that GPU and OS. The rendered
document calls features "options", the plainer word for readers.

Same layout as the graph-based percentage view: a dashboard of op groups by
GPU and OS target, one table per architecture family; then one table per
engine; then the op groups and GPU targets the tables link to. Rows are API
node types rather than bundle directories, because features belong to node
types: a fused ConvolutionFwdPointwise graph counts toward both
ConvolutionFwd and Pointwise.

Features count singly, not in combination: bf16 and causal each being claimed
says nothing about bf16 *with* causal. ``coverage.py`` prints the per-feature
graph counts behind every cell.

Usage:

    python3 dnn-providers/integration-tests/scripts/support_features/feature_matrix.py \\
        [--output PATH]

Without ``--output`` the document goes to stdout.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from support_features import registry  # noqa: E402
from support_features.coverage import unit_features  # noqa: E402
from support_matrix_markdown import FULL, PARTIAL  # noqa: E402
from support_matrix_units import ClaimUnit, collect_units  # noqa: E402

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_BUNDLES_DIR = SCRIPTS_DIR.parent / "integration-test-bundles"
REGEN_COMMAND = f"python3 {SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()}"

NOT_CLAIMED = "❌"
PLATFORMS = ("linux", "windows")

Target = tuple[str, str]  # (arch, platform)

# --------------------------------------------------------------------------
# Op groups and GPU families
# --------------------------------------------------------------------------

# First match wins, over the display name (node type without "Attributes").
GROUP_RULES = (
    ("Attention", lambda op: op.startswith("Sdpa")),
    ("RMSNorm", lambda op: op.startswith("RMSNorm")),
    ("Batchnorm", lambda op: op.startswith("Batchnorm")),
    ("Layernorm", lambda op: op.startswith("Layernorm")),
    ("Convolution", lambda op: op.startswith("Convolution")),
    ("Matmul", lambda op: "Matmul" in op),
    ("Resample", lambda op: op.startswith("Resample")),
    ("Reduction", lambda op: op.startswith("Reduction")),
    ("Pointwise", lambda op: op.startswith("Pointwise")),
    ("Block scale", lambda op: op.startswith("BlockScale")),
)

# arch -> (architecture generation, example products)
GPU_INFO = {
    "gfx908": ("CDNA", "Instinct MI100"),
    "gfx90a": ("CDNA 2", "Instinct MI200 series"),
    "gfx942": ("CDNA 3", "Instinct MI300 series"),
    "gfx950": ("CDNA 4", "Instinct MI350 series"),
    "gfx1030": ("RDNA 2", "Radeon RX 6800 / 6900, Radeon PRO W6800"),
    "gfx1100": ("RDNA 3", "Radeon RX 7900, Radeon PRO W7900"),
    "gfx1101": ("RDNA 3", "Radeon RX 7800 / 7700"),
    "gfx1102": ("RDNA 3", "Radeon RX 7600"),
    "gfx1150": ("RDNA 3.5", "Ryzen AI 300 (Strix Point)"),
    "gfx1151": ("RDNA 3.5", "Ryzen AI Max (Strix Halo)"),
    "gfx1200": ("RDNA 4", "Radeon RX 9060"),
    "gfx1201": ("RDNA 4", "Radeon RX 9070, Radeon AI PRO R9700"),
}

# gfx1xxx is RDNA; gfx9xx is CDNA.
FAMILIES = (
    ("RDNA (consumer)", lambda arch: arch.startswith("gfx1")),
    ("CDNA (data center)", lambda arch: arch.startswith("gfx9")),
)


def display_name(node_type: str) -> str:
    """``SdpaAttributes`` -> ``Sdpa``; the suffix is on every node type."""
    return node_type.replace("Attributes", "", 1)


def group_of(node_type: str) -> str:
    name = display_name(node_type)
    for group, matches in GROUP_RULES:
        if matches(name):
            return group
    return name


def arch_key(arch: str) -> tuple[int, str]:
    return (len(arch), arch)


# --------------------------------------------------------------------------
# Names and anchors
# --------------------------------------------------------------------------


def target_name(target: Target) -> str:
    return f"{target[0]}-{target[1]}"


def target_anchor(target: Target) -> str:
    return f"gpu-{target_name(target)}"


def group_anchor(group: str) -> str:
    return "group-" + group.lower().replace(" ", "-")


def op_anchor(node_type: str) -> str:
    return "op-" + display_name(node_type).lower()


def heading_anchor(heading: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces to '-'."""
    kept = "".join(c for c in heading.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def link(text: str, anchor: str) -> str:
    return f"[{text}](#{anchor})"


def short_target_link(target: Target) -> str:
    arch, platform = target
    return f"{link(arch, target_anchor(target))}<br>{platform[0].upper()}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out + [""]


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------


@dataclass
class FeatureClaims:
    """Which of each op's features are exercised, and where they are claimed.

    Every map is keyed by node type first, so feature ids never mix across
    ops; a group's share sums its ops' counts.
    """

    # node type -> catalogued features, in catalog order
    features: dict[str, tuple[str, ...]]
    # group -> graphs holding any of its ops; a graph with two of them
    # counts once
    group_graphs: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # node type -> features any bundle graph exercises, claimed or not
    tested: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # (node type, target) -> features a graph claimed there by any engine
    # exercises
    any_engine: dict[tuple[str, Target], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # (node type, engine, target) -> features a graph that engine claims
    # there exercises
    by_engine: dict[tuple[str, str, Target], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    engines: list[str] = field(default_factory=list)
    # GPU order (gfx90a, gfx942, ..., gfx1201), Linux before Windows.
    targets: list[Target] = field(default_factory=list)
    unit_count: int = 0

    @property
    def node_types(self) -> list[str]:
        return sorted(self.features, key=lambda t: display_name(t).lower())

    @property
    def groups(self) -> list[str]:
        return sorted({group_of(t) for t in self.features}, key=str.lower)

    def group_members(self, group: str) -> list[str]:
        return [t for t in self.node_types if group_of(t) == group]

    def total(self, node_types: list[str]) -> int:
        return sum(len(self.features[t]) for t in node_types)

    def on_target(self, node_types: list[str], target: Target) -> int:
        return sum(len(self.any_engine.get((t, target), ())) for t in node_types)

    def engine_on_target(self, node_type: str, engine: str, target: Target) -> int:
        return len(self.by_engine.get((node_type, engine, target), ()))


def compute_feature_claims(
    units: list[ClaimUnit], errors: list[str] | None = None
) -> FeatureClaims:
    claims = FeatureClaims(
        features={
            node_type: tuple(f.id for f in module.FEATURES)
            for node_type, module in registry.OPS.items()
        }
    )
    claims.unit_count = len(units)
    engines: set[str] = set()
    found: set[Target] = set()
    for unit in units:
        engines |= set(unit.claims)
        for pairs in unit.claims.values():
            found |= pairs
        found_features = unit_features(unit, errors)
        for group in {group_of(t) for t in found_features}:
            claims.group_graphs[group] += 1
        for node_type, features in found_features.items():
            claims.tested[node_type] |= features
            for engine, pairs in unit.claims.items():
                for target in pairs:
                    claims.any_engine[(node_type, target)] |= features
                    claims.by_engine[(node_type, engine, target)] |= features

    claims.engines = sorted(engines)
    archs = sorted({a for a, _ in found}, key=arch_key)
    claims.targets = [(a, p) for a in archs for p in PLATFORMS if (a, p) in found]
    return claims


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


def percent_cell(count: int, total: int) -> str:
    if count == 0:
        return NOT_CLAIMED
    if count == total:
        return FULL
    # Floor, but never down to 0% for a cell that holds a feature.
    return f"{PARTIAL} {max(1, 100 * count // total)}%"


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


def render_dashboard(claims: FeatureClaims) -> list[str]:
    every_op = claims.node_types
    options_link = link("Options per op", heading_anchor("Options per op"))
    out = [
        "## Summary by architecture",
        "",
        "- **Op group (options)**: the number is how many options the "
        "group's ops offer in the API (data types, mask modes, layouts, "
        "optional inputs, ...), catalogued from the flatbuffers schema. Each "
        f"option is counted once and listed in {options_link}, split into "
        "those a test graph uses and those none does.",
        "- **GPU columns**: share of those options used by at least one graph "
        f"that an engine claims on that GPU and OS. `{FULL}` all · "
        f"`{PARTIAL} 45%` 45% · `{NOT_CLAIMED}` none. GPUs with the most "
        "options claimed come first. Combinations aren't measured: a GPU can "
        "cover fp8 and causal masking in separate graphs without any graph "
        "testing them together.",
        "",
    ]
    for family, in_family in FAMILIES:
        targets = [t for t in claims.targets if in_family(t[0])]
        out += [f"### {family}", ""]
        if not targets:
            out += ["_No support claimed yet._", ""]
            continue
        targets.sort(
            key=lambda t: (-claims.on_target(every_op, t), claims.targets.index(t))
        )
        rows = []
        for group in claims.groups:
            members = claims.group_members(group)
            total = claims.total(members)
            rows.append(
                [
                    f"**{link(group, group_anchor(group))}** ({total})",
                    *(
                        percent_cell(claims.on_target(members, t), total)
                        for t in targets
                    ),
                ]
            )
        out += table(
            [
                "Op group (options)",
                *(link(target_name(t), target_anchor(t)) for t in targets),
            ],
            rows,
        )
    return out


def render_by_engine(claims: FeatureClaims) -> list[str]:
    engines = " · ".join(link(e, heading_anchor(e)) for e in claims.engines)
    out = [
        "## By engine",
        "",
        f"**Engines:** {engines}",
        "",
        "- **Op (options)**: the number is how many options the op offers in "
        "the API, listed in "
        f"{link('Options per op', heading_anchor('Options per op'))}.",
        "- **GPU columns**: share of the op's options that a graph this "
        f"engine claims on that GPU and OS uses. `{FULL}` all · "
        f"`{PARTIAL} 72%` 72% · `{NOT_CLAIMED}` none. L = Linux, "
        "W = Windows.",
        "",
        "An op appears under an engine that claims at least one of its graphs "
        "somewhere.",
        "",
    ]
    for engine in claims.engines:
        rows = []
        for node_type in claims.node_types:
            counts = [
                claims.engine_on_target(node_type, engine, t) for t in claims.targets
            ]
            if not any(counts):
                continue
            total = len(claims.features[node_type])
            rows.append(
                [
                    f"**{link(display_name(node_type), op_anchor(node_type))}** "
                    f"({total})",
                    *(percent_cell(c, total) for c in counts),
                ]
            )
        out += [f"### {engine}", ""]
        out += table(["Op (options)", *map(short_target_link, claims.targets)], rows)
    return out


def render_op_groups(claims: FeatureClaims) -> list[str]:
    rows = []
    for group in claims.groups:
        members = claims.group_members(group)
        listed = ", ".join(
            f"{link(display_name(t), op_anchor(t))} ({len(claims.features[t])})"
            for t in members
        )
        rows.append(
            [
                f'<a id="{group_anchor(group)}"></a>**{group}**',
                str(claims.group_graphs.get(group, 0)),
                str(claims.total(members)),
                listed,
            ]
        )
    return [
        "## Op groups",
        "",
        "The API node types (`<Op>Attributes` in the flatbuffers schema) that "
        "make up each group, with their option counts. *Graphs* counts "
        "the test graphs holding an op of the group, claimed or not.",
        "",
        *table(["Op group", "Graphs", "Options", "Ops (options)"], rows),
    ]


def render_features_per_op(claims: FeatureClaims) -> list[str]:
    rows = []
    for node_type in claims.node_types:
        features = claims.features[node_type]
        used = claims.tested.get(node_type, set())
        exercised = [f for f in features if f in used]
        missing = [f for f in features if f not in used]
        rows.append(
            [
                f'<a id="{op_anchor(node_type)}"></a>**{display_name(node_type)}** '
                f"({len(features)})",
                ", ".join(f"`{f}`" for f in exercised) or "—",
                ", ".join(f"`{f}`" for f in missing) or "—",
            ]
        )
    return [
        "## Options per op",
        "",
        "Every API option the percentages count, catalogued per op in "
        "`dnn-providers/integration-tests/scripts/support_features/<op>.py`. "
        "An option is one choice a graph can make for an op: a data type "
        "(`dtype:bf16`), a mode (`mask:causal_top_left`), a tensor-shape case "
        "(`heads:gqa`), or an optional input or output (`out:stats`). *Test "
        "graph* means a graph in `integration-test-bundles/`, any tier, "
        "claimed or not. An option no test graph uses cannot be claimed on "
        "any GPU until one does.",
        "",
        *table(["Op (options)", "Used by a test graph", "Used by no test graph"], rows),
    ]


def render_gpu_targets(claims: FeatureClaims) -> list[str]:
    rows = []
    for target in claims.targets:
        generation, products = GPU_INFO.get(target[0], ("", ""))
        rows.append(
            [
                f'<a id="{target_anchor(target)}"></a>**{target_name(target)}**',
                generation,
                products,
            ]
        )
    return [
        "## GPU targets",
        "",
        "The GPU and OS targets the tables' columns link to.",
        "",
        *table(["Target", "Architecture", "Products"], rows),
    ]


TITLE = "Support matrix by API option"


def render(claims: FeatureClaims) -> str:
    sections = [
        *(name for name, _ in FAMILIES),
        "By engine",
        "Op groups",
        "Options per op",
        "GPU targets",
    ]
    jump = " · ".join(link(s, heading_anchor(s)) for s in sections)
    lines = [
        f"# {TITLE}",
        "",
        f"<!-- Generated by {REGEN_COMMAND}; edit the script, not this file. -->",
        "",
        f"Generated from the committed support claims ({claims.unit_count} "
        f"graphs, {claims.total(claims.node_types)} catalogued options). "
        "Percentages are shares of each op's API options, not of its graphs. "
        "Options count singly, not in combination.",
        "",
        f"**Jump to:** {jump}",
        "",
        *render_dashboard(claims),
        *render_by_engine(claims),
        *render_op_groups(claims),
        *render_features_per_op(claims),
        *render_gpu_targets(claims),
    ]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundles-dir",
        type=pathlib.Path,
        default=DEFAULT_BUNDLES_DIR,
        help=f"Bundle tree to scan (default: {DEFAULT_BUNDLES_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Write the document here instead of stdout.",
    )
    args = parser.parse_args(argv)

    if not args.bundles_dir.is_dir():
        print(f"error: {args.bundles_dir} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []
    claims = compute_feature_claims(collect_units(args.bundles_dir, errors), errors)
    if errors:
        # A graph left out of a share makes the share wrong, so none is shown.
        print(f"error: {len(errors)} defect(s); nothing rendered", file=sys.stderr)
        return 1
    document = render(claims)
    if args.output is None:
        sys.stdout.write(document)
    else:
        args.output.write_text(document, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
