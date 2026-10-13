#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Two candidate support matrix layouts, for the team to compare and vote on.

Reads the committed support sidecars through ``support_matrix_units.py`` and
writes two markdown files to ``different_renderer/`` at the repository root:

    python3 dnn-providers/integration-tests/scripts/support_matrix_layout_vote.py

Both files open with the same dashboard: op groups down the side, GPU and OS
targets across, one table per architecture family (RDNA, CDNA). Below it comes
a per-engine table of ops by target. The two files differ only in how that
per-engine table fills a partially claimed cell: a percentage of the op's
graphs, or the count of claimed graphs.

Scratch for the layout discussion; nothing in CI reads the output.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from support_matrix_markdown import FULL, PARTIAL  # noqa: E402
from support_matrix_units import ClaimUnit, collect_units  # noqa: E402

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_BUNDLES_DIR = SCRIPT_PATH.parent.parent / "integration-test-bundles"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "different_renderer"
REGEN_COMMAND = f"python3 {SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()}"

NOT_CLAIMED = "❌"
PLATFORMS = ("linux", "windows")

# --------------------------------------------------------------------------
# Op groups and GPU families
# --------------------------------------------------------------------------

# First match wins, so RMSNorm sits ahead of Pointwise
# (PointwiseRMSNormBackward is an RMSNorm graph).
GROUP_RULES = (
    ("Attention", lambda op: op.startswith("Sdpa")),
    ("RMSNorm", lambda op: "RMSNorm" in op),
    ("Batchnorm", lambda op: op.startswith("Batchnorm")),
    ("Layernorm", lambda op: op.startswith("Layernorm")),
    ("Convolution", lambda op: op.startswith("Convolution")),
    ("Matmul", lambda op: "Matmul" in op),
    ("Resample", lambda op: op.startswith("Resample")),
    ("Reduction", lambda op: op.startswith("Reduction")),
    ("Pointwise", lambda op: op.startswith("Pointwise")),
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


def group_of(op: str) -> str:
    for group, matches in GROUP_RULES:
        if matches(op):
            return group
    return op


def arch_key(arch: str) -> tuple[int, str]:
    return (len(arch), arch)


# --------------------------------------------------------------------------
# Names and anchors
# --------------------------------------------------------------------------


def target_name(target: tuple[str, str]) -> str:
    return f"{target[0]}-{target[1]}"


def target_anchor(target: tuple[str, str]) -> str:
    return f"gpu-{target_name(target)}"


def group_anchor(group: str) -> str:
    return f"group-{group.lower()}"


def heading_anchor(heading: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces to '-'."""
    kept = "".join(c for c in heading.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")


def link(text: str, anchor: str) -> str:
    return f"[{text}](#{anchor})"


def short_target_link(target: tuple[str, str]) -> str:
    arch, platform = target
    return f"{link(arch, target_anchor(target))}<br>{platform[0].upper()}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out + [""]


# --------------------------------------------------------------------------
# Claim counting
# --------------------------------------------------------------------------


class Claims:
    def __init__(self, units: list[ClaimUnit]):
        self.units = units
        self.by_op: dict[str, list[ClaimUnit]] = defaultdict(list)
        for unit in units:
            self.by_op[unit.op].append(unit)
        self.ops = sorted(self.by_op)
        self.by_group: dict[str, list[ClaimUnit]] = defaultdict(list)
        for unit in units:
            self.by_group[group_of(unit.op)].append(unit)
        self.groups = sorted(self.by_group, key=str.lower)
        self.engines = sorted({e for u in units for e in u.claims})
        found = {t for u in units for pairs in u.claims.values() for t in pairs}
        archs = sorted({a for a, _ in found}, key=arch_key)
        # GPU order (gfx90a, gfx942, ..., gfx1201), Linux before Windows.
        self.targets = [(a, p) for a in archs for p in PLATFORMS if (a, p) in found]

    @staticmethod
    def any_engine(units: list[ClaimUnit], target: tuple[str, str]) -> int:
        return sum(1 for u in units if any(target in s for s in u.claims.values()))

    def on_target(self, target: tuple[str, str]) -> int:
        """Graphs claimed by any engine on ``target``."""
        return self.any_engine(self.units, target)

    def group_on_target(self, group: str, target: tuple[str, str]) -> int:
        return self.any_engine(self.by_group[group], target)

    def op_by_engine(self, op: str, engine: str, target: tuple[str, str]) -> int:
        return sum(1 for u in self.by_op[op] if u.supported_by(engine, *target))


def count_cell(claimed: int, total: int) -> str:
    if claimed == 0:
        return NOT_CLAIMED
    return f"{FULL if claimed == total else PARTIAL} {claimed}"


def percent_cell(claimed: int, total: int) -> str:
    if claimed == 0:
        return NOT_CLAIMED
    if claimed == total:
        return FULL
    # Floor, but never down to 0% for a cell that holds a claim.
    return f"{PARTIAL} {max(1, 100 * claimed // total)}%"


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


def render_dashboard(claims: Claims) -> list[str]:
    out = [
        "## Summary by architecture",
        "",
        f"`{FULL} 15` all 15 of the group's graphs claimed · "
        f"`{PARTIAL} 112` 112 graphs claimed, not all · "
        f"`{NOT_CLAIMED}` none claimed. "
        "A graph counts when any engine claims it on that GPU and OS. "
        "Within each table, the GPUs with the most claimed graphs come first.",
        "",
    ]
    for family, in_family in FAMILIES:
        targets = [t for t in claims.targets if in_family(t[0])]
        out += [f"### {family}", ""]
        if not targets:
            out += ["_No support claimed yet._", ""]
            continue
        targets.sort(key=lambda t: (-claims.on_target(t), claims.targets.index(t)))
        header = [
            "Op group",
            "Graphs",
            *(link(target_name(t), target_anchor(t)) for t in targets),
        ]
        rows = []
        for group in claims.groups:
            total = len(claims.by_group[group])
            rows.append(
                [
                    f"**{link(group, group_anchor(group))}**",
                    str(total),
                    *(
                        count_cell(claims.group_on_target(group, t), total)
                        for t in targets
                    ),
                ]
            )
        out += table(header, rows)
    return out


def render_by_engine(claims: Claims, percent: bool) -> list[str]:
    cell = percent_cell if percent else count_cell
    example = (
        f"`{PARTIAL} 72%` 72% of the op's graphs claimed"
        if percent
        else (f"`{PARTIAL} 112` 112 of the op's graphs claimed")
    )
    full = f"`{FULL}`" if percent else f"`{FULL} 15`"
    out = [
        "## By engine",
        "",
        f"{full} all of the op's graphs claimed · {example} · "
        f"`{NOT_CLAIMED}` none claimed. L = Linux, W = Windows.",
        "",
    ]
    for engine in claims.engines:
        claimed_ops = [
            op
            for op in claims.ops
            if any(claims.op_by_engine(op, engine, t) for t in claims.targets)
        ]
        out += [f"### {engine}", ""]
        rows = []
        for op in claimed_ops:
            total = len(claims.by_op[op])
            rows.append(
                [
                    f"**{link(op, group_anchor(group_of(op)))}**",
                    str(total),
                    *(
                        cell(claims.op_by_engine(op, engine, t), total)
                        for t in claims.targets
                    ),
                ]
            )
        out += table(["Op", "Graphs", *map(short_target_link, claims.targets)], rows)
        unclaimed = [op for op in claims.ops if op not in claimed_ops]
        if unclaimed:
            out += [f"_No claims on any GPU:_ {', '.join(unclaimed)}", ""]
    return out


def render_op_groups(claims: Claims) -> list[str]:
    rows = []
    for group in claims.groups:
        ops = sorted({u.op for u in claims.by_group[group]})
        members = ", ".join(f"{op} ({len(claims.by_op[op])})" for op in ops)
        rows.append(
            [
                f'<a id="{group_anchor(group)}"></a>**{group}**',
                str(len(claims.by_group[group])),
                members,
            ]
        )
    return [
        "## Op groups",
        "",
        "The op directories under `integration-test-bundles/<tier>/` that "
        "make up each group, with their graph counts.",
        "",
        *table(["Op group", "Graphs", "Ops (graphs)"], rows),
    ]


def render_gpu_targets(claims: Claims) -> list[str]:
    rows = []
    for target in claims.targets:
        generation, products = GPU_INFO.get(target[0], ("", ""))
        rows.append(
            [
                f'<a id="{target_anchor(target)}"></a>**{target_name(target)}**',
                generation,
                products,
                str(claims.on_target(target)),
            ]
        )
    return [
        "## GPU targets",
        "",
        f"Graphs claimed: by any engine, out of {len(claims.units)}.",
        "",
        *table(["Target", "Architecture", "Products", "Graphs claimed"], rows),
    ]


def render(claims: Claims, title: str, percent: bool) -> str:
    sections = [
        *(name for name, _ in FAMILIES),
        "By engine",
        "Op groups",
        "GPU targets",
    ]
    jump = " · ".join(link(s, heading_anchor(s)) for s in sections)
    engines = " · ".join(link(e, heading_anchor(e)) for e in claims.engines)
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by " f"{REGEN_COMMAND}; edit the script, not this file. -->",
        "",
        f"Generated from the committed support claims: {len(claims.units)} "
        "graphs. One of two layouts for the team to compare; they differ "
        "only in the cells of the **By engine** tables.",
        "",
        f"**Jump to:** {jump}",
        "",
        f"**Engines:** {engines}",
        "",
        *render_dashboard(claims),
        *render_by_engine(claims, percent),
        *render_op_groups(claims),
        *render_gpu_targets(claims),
    ]
    return "\n".join(lines).rstrip() + "\n"


LAYOUTS = (
    (
        "option_1_engine_percentages.md",
        "Support matrix, option 1: engine tables show percentages",
        True,
    ),
    (
        "option_2_engine_counts.md",
        "Support matrix, option 2: engine tables show graph counts",
        False,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles-dir", type=pathlib.Path, default=DEFAULT_BUNDLES_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    errors: list[str] = []
    units = collect_units(args.bundles_dir, errors)
    if errors:
        # Each was printed as found.
        print(
            f"error: {len(errors)} bundle defect(s); nothing rendered", file=sys.stderr
        )
        return 1
    claims = Claims(units)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, title, percent in LAYOUTS:
        path = args.output_dir / filename
        path.write_text(render(claims, title, percent), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
