#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Markdown pass of the support matrix renderer: ``ClaimUnit`` -> document.

``render_support_matrix.py`` is the entry point; ``support_matrix_units.py``
collects the units this module renders. Nothing here reads the bundle tree, so
another output format is another module over the same units.

The document has four zoom levels: per-target overview, per-op variant table,
per-(variant, dtype) detail, and HTML traceability comments naming the bundle
each row came from. ``overview_only`` keeps the first level alone.
"""

from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from support_matrix_units import NO_LAYOUT, ClaimUnit  # noqa: E402

# --------------------------------------------------------------------------
# Regeneration commands quoted back to the reader
# --------------------------------------------------------------------------

_CLI_PATH = pathlib.Path(__file__).resolve().with_name("render_support_matrix.py")
_REPO_ROOT = _CLI_PATH.parents[3]

# No --output: the CLI defaults to the full matrix path, so the shortest command
# that reproduces the document is the one worth quoting back to the reader.
REGEN_COMMAND = f"python3 {_CLI_PATH.relative_to(_REPO_ROOT).as_posix()}"
REGEN_OVERVIEW_COMMAND = f"{REGEN_COMMAND} --overview-only"

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

FULL = "✅"
PARTIAL = "🟡"
NONE = "—"

# Short key at the top — just enough to read the tables. The full reference
# (what a claim means, how variants are tagged, how tiers merge) goes at the
# end so the reader hits the data first.
LEGEND_DETAIL = (
    "Expand a row for variant and per-dtype detail. "
    "Full legend at the [end of the document](#reading-guide)."
)

_LEGEND_CELLS = (
    f"`{FULL} 108/108` all graphs claimed · "
    f"`{PARTIAL} 72/108` 72 of 108 claimed · "
    f"`{NONE}` none claimed (not the same as *known unsupported*)"
)

LEGEND_OVERVIEW = (
    f"{_LEGEND_CELLS}\n"
    "\n"
    "A box is one op (row) × engine (column) on this target: claimed graphs "
    "out of all the op's graphs, with each template-sweep case counted as one "
    "graph. A claim is a CI pass recorded in a committed sidecar, so new "
    "passes show up only after a harvest."
)

_LEGEND_PREFIX = f"{_LEGEND_CELLS}. "


def _legend(overview_only: bool) -> list[str]:
    if overview_only:
        return [LEGEND_OVERVIEW, ""]
    return [_LEGEND_PREFIX + LEGEND_DETAIL, ""]


# The full reference, rendered after the last target section.
READING_GUIDE = [
    "## Reading guide",
    "",
    "_Each overview row is one op directory of the bundle tree. Open a "
    "disclosure triangle to expand it into per-variant rows, and the one "
    "inside that for full per-(variant, dtype) rows. Every level counts the "
    "same way, so an op's "
    "variant counts sum to its overview count and a variant's dtype counts sum "
    "to the variant._",
    "",
    "### Cell vocabulary",
    "",
    "| Cell | Meaning |",
    "|------|---------|",
    f"| `{FULL} 108/108` | every graph in the row is claimed by that engine "
    "on that target |",
    f"| `{PARTIAL} 72/108` | 72 of the row's 108 graphs are claimed |",
    f"| `{PARTIAL} 8/25 NCHW, NHWC` | the same count, plus the layouts the "
    "claimed graphs cover |",
    f"| `{NONE}` | no claim recorded |",
    f"| `{NO_LAYOUT}` | the graph's rank carries no NCHW-family layout "
    "(a rank-2 matmul operand, say) |",
    "",
    "**Denominator.** Every graph the bundle tree holds for that row: a "
    "single-graph bundle counts once, a template sweep counts once per case. "
    "It does not depend on the target, so the same row shows the same "
    "denominator in every `arch / platform` section — only the numerator "
    "moves.",
    "",
    "**A claim is an assertion, not a measurement taken here.** Each one is an "
    "entry in a `.support.json` sidecar (RFC 0015 §4) saying the engine "
    "supports that exact graph, and the integration tests enforce it. So "
    f"`{NONE}` — and the unclaimed share of a `{PARTIAL}` — means *unclaimed*, "
    "which is not the same as *known unsupported*: the engine may genuinely "
    "reject those graphs, or no one has claimed them yet.",
    "",
    "**Targets.** Sidecars key claims by gfx target, so that is what the "
    "section headings use. One gfx target covers a whole series, so a claim on "
    "`gfx90a` is a claim for every MI200 part.",
    "",
    "**Variants** read `[feature tags] + fusion chain`: whatever is fused onto "
    "the graph's first node. Where that node is not the op the row is filed "
    "under (`ConvolutionFwdPointwise` graphs start with `ConvolutionFwd`), the "
    "variant names it first. The tags each mark a "
    "departure from the simplest case, so an *untagged* row means batch 1, "
    "unit stride, unit dilation, no padding and no groups:",
    "",
    "| Tag | Read off |",
    "|-----|----------|",
    "| `multi_batch` | leading dimension above 1 (rank 3 and up — a rank-2 "
    "operand has no batch axis) |",
    "| `grouped` | filter spanning fewer channels than the activation "
    "carries (`x`/`w` forward, `dx`/`w` data-grad, `x`/`dw` weight-grad) |",
    "| `stride` | any stride other than 1 |",
    "| `dilation` | any dilation other than 1 |",
    "| `padding` | any non-zero pre- or post-padding |",
    "",
    "They exist because the fusion chain alone cannot separate a strided "
    "grouped convolution from a plain one, so an engine that takes the second "
    "and refuses the first would have nowhere to say so.",
    "",
    "**Tiers are merged.** A row can draw on `quick`, `standard`, "
    "`comprehensive`, and `full` bundles at once. A graph is identified by "
    "its path below the tier directory (plus its case id, for a sweep), so "
    "the same graph committed to two tiers counts once, with both sidecars' "
    "claims merged. The bundle paths and tiers behind each row are in HTML "
    "comments in the appendix below, "
    "recorded once because provenance does not vary by target.",
    "",
]

# --------------------------------------------------------------------------
# Cell rendering
# --------------------------------------------------------------------------


def count_cell(supported: int, total: int) -> str:
    if total == 0 or supported == 0:
        return NONE
    if supported == total:
        return f"{FULL} {supported}/{total}"
    return f"{PARTIAL} {supported}/{total}"


def variant_sort_key(variant: str) -> tuple[int, str]:
    """Alphabetical, with the unfused graph pulled to the front.

    Fused variants all begin with ``" + "``, so a plain sort buries ``(bare)``
    -- the one row a reader scanning a family wants first -- under them.
    """
    return (1, variant) if variant != "(bare)" else (0, variant)


def layout_sort_key(layout: str) -> tuple[int, str]:
    """Alphabetical, with the "no layout" label sorted last."""
    return (1, layout) if layout == NO_LAYOUT else (0, layout)


def layout_cell(units: list[ClaimUnit], engine: str, arch: str, platform: str) -> str:
    """Supported layouts for a row, behind the same count the overview uses.

    The layouts cannot carry the row on their own. A family with one layout
    throughout -- Sdpa is BHSD everywhere -- renders every row identically, so
    a variant claimed 4 of 5 and one claimed 8 of 15 both read ``PARTIAL BHSD``
    and the reader cannot tell a nearly-complete row from a mostly-empty one.

    Leading with the count also makes the zoom levels checkable against each
    other: a family's variant counts must sum to its overview count, and a
    variant's dtype counts must sum to the variant. A cell that disagrees with
    its own children is then visible rather than merely wrong.
    """
    supported = [u for u in units if u.supported_by(engine, arch, platform)]
    if not supported:
        return NONE
    layouts = sorted({u.layout for u in supported}, key=layout_sort_key)
    return f"{count_cell(len(supported), len(units))} {', '.join(layouts)}"


def summary_cell(units: list[ClaimUnit], engine: str, arch: str, platform: str) -> str:
    supported = sum(1 for u in units if u.supported_by(engine, arch, platform))
    return count_cell(supported, len(units))


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def traceability_comment(
    row: str, units: list[ClaimUnit], max_case_ids: int
) -> list[str]:
    """One HTML comment per row, listing every bundle that fed it.

    Invisible when rendered, but it is what makes every aggregated row lead
    back to the exact bundle directory that produced it -- and from there to
    the case ids, via that bundle's ``sweep.json``. The ids are deliberately
    *not* inlined by default: enumerating all of them costs several times the
    size of the visible document and pushes it past the point where GitHub
    renders markdown at all. Pass ``--max-case-ids`` to inline them anyway.

    One comment per row rather than per (row, bundle): a row draws on 2.7
    bundles on average, and repeating its label -- now long enough to carry a
    tag list and a clipped pointwise chain -- once per bundle spent most of the
    comment bytes saying nothing new, on a document with a hard size ceiling.

    Emitted as a block *after* the table it annotates: a comment line between
    table rows would terminate the table in GitHub-flavored markdown.
    """
    # Grouped by bundle alone, with the union of its tiers: a sweep whose cases
    # are split across tiers is still one entry, so retiering a case changes
    # the tier list and nothing else.
    by_bundle: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_bundle[unit.bundle].append(unit)

    entries = []
    for bundle in sorted(by_bundle):
        members = by_bundle[bundle]
        tiers = sorted({tier for u in members for tier in u.tiers})
        entry = f"{bundle} ({'+'.join(tiers)}, {len(members)} case(s))"
        if max_case_ids != 0:
            labels = sorted(u.label for u in members)
            if 0 < max_case_ids < len(labels):
                shown = ", ".join(labels[:max_case_ids])
                entry += f": {shown}, … +{len(labels) - max_case_ids} more"
            else:
                entry += ": " + ", ".join(labels)
        entries.append(entry)

    if not entries:
        return []
    return [f"<!-- row: {row} | bundles: " + "; ".join(entries) + " -->"]


def detail_row_label(primary: str, variant: str, dtypes: str) -> str:
    """``Batchnorm[multi_batch] + Pointwise:RELU_FWD [io=fp16, ...]``.

    Shared with the traceability appendix so a row and its provenance entry are
    keyed by the same string.
    """
    suffix = "" if variant == "(bare)" else variant
    return f"{primary}{suffix} {dtypes}"


def group_detail_rows(
    units: list[ClaimUnit],
) -> list[tuple[str, list[ClaimUnit]]]:
    """(row label, units) per (variant, dtype) row, in display order."""
    grouped: dict[tuple[str, str], list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        grouped[(unit.variant_label, unit.dtypes)].append(unit)
    rows = []
    for key in sorted(grouped, key=lambda key: (variant_sort_key(key[0]), key[1])):
        members = grouped[key]
        # Every member shares the variant label, and with it the head node
        # and fusion chain the row label spells out.
        label = detail_row_label(members[0].primary, members[0].variant, key[1])
        rows.append((label, members))
    return rows


def render_detail(
    lines: list[str],
    units: list[ClaimUnit],
    engines: list[str],
    arch: str,
    platform: str,
) -> None:
    """The innermost drill-down: one row per (variant, dtype)."""
    lines.append("<details>")
    lines.append("<summary>🔎 per-(variant, dtype) detail</summary>")
    lines.append("")
    lines.append("| Operations | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))

    for label, rows in group_detail_rows(units):
        cells = [layout_cell(rows, e, arch, platform) for e in engines]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("</details>")


def render_family(
    lines: list[str],
    op: str,
    units: list[ClaimUnit],
    engines: list[str],
    arch: str,
    platform: str,
) -> None:
    lines.append("<details>")
    lines.append(f"<summary>📂 <b>{op}</b></summary>")
    lines.append("")
    lines.append("| Variant | " + " | ".join(engines) + " |")
    lines.append("|" + "---|" * (len(engines) + 1))

    by_variant: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_variant[unit.variant_label].append(unit)
    for variant in sorted(by_variant, key=variant_sort_key):
        rows = by_variant[variant]
        cells = [layout_cell(rows, e, arch, platform) for e in engines]
        lines.append(f"| `{variant}` | " + " | ".join(cells) + " |")

    lines.append("")
    dtypes = sorted({u.dtypes for u in units})
    lines.append(f"_Dtypes observed: {', '.join(dtypes)}_")
    lines.append("")
    render_detail(lines, units, engines, arch, platform)
    lines.append("</details>")
    lines.append("")


def render_traceability(
    lines: list[str], units: list[ClaimUnit], max_case_ids: int
) -> None:
    """The bundle provenance index, emitted once for the whole document.

    Which bundles feed a row does not depend on the target, so emitting this
    inside each ``arch / platform`` section duplicated all of it verbatim --
    growing linearly with every target added,
    on a document that stops rendering on GitHub past a few hundred KB.
    """
    by_op: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_op[unit.op].append(unit)

    lines.append("<!-- Row provenance. Target-independent, so recorded once. -->")
    for op in sorted(by_op):
        for label, rows in group_detail_rows(by_op[op]):
            lines.extend(traceability_comment(f"{op}: {label}", rows, max_case_ids))


def render_markdown(
    units: list[ClaimUnit],
    max_case_ids: int,
    *,
    overview_only: bool = False,
) -> str:
    engines = sorted({e for u in units for e in u.claims})
    targets = sorted({pair for u in units for s in u.claims.values() for pair in s})

    lines = [
        "# Combined Engine Support Matrix",
        "",
        "Generated by `render_support_matrix.py` from committed `.support.json` "
        "sidecars.",
        "Do not hand-edit — regenerate with: "
        f"`{REGEN_OVERVIEW_COMMAND if overview_only else REGEN_COMMAND}`",
        *(
            [
                "",
                "For the full per-variant and per-dtype detail matrix, run: "
                f"`{REGEN_COMMAND}`",
            ]
            if overview_only
            else []
        ),
        "",
        *_legend(overview_only),
    ]

    if not units:
        lines.append("_No claim-bearing bundles found._")
        return "\n".join(lines) + "\n"

    if not targets:
        lines.append(
            f"_{len(units)} bundle(s) found, none carrying any support claim._"
        )
        return "\n".join(lines) + "\n"

    by_op: dict[str, list[ClaimUnit]] = defaultdict(list)
    for unit in units:
        by_op[unit.op].append(unit)
    ops = sorted(by_op)

    for arch, platform in targets:
        if overview_only:
            lines.append(f"## {arch} / {platform}")
        else:
            lines.append("<details>")
            lines.append(f"<summary><big><b>{arch} / {platform}</b></big></summary>")
        lines.append("")
        if not overview_only:
            lines.append("### Overview")
            lines.append("")
        lines.append("| Op | " + " | ".join(engines) + " |")
        lines.append("|----|" + "|".join(["----------"] * len(engines)) + "|")
        for op in ops:
            cells = [summary_cell(by_op[op], e, arch, platform) for e in engines]
            lines.append(f"| **{op}** | " + " | ".join(cells) + " |")
        lines.append("")

        if not overview_only:
            for op in ops:
                render_family(lines, op, by_op[op], engines, arch, platform)

            if lines and lines[-1] == "":
                lines.pop()
            lines.append("</details>")
            lines.append("<br>")

    if not overview_only:
        lines.append("<details>")
        lines.append("<summary><big><b>Reading guide</b></big></summary>")
        lines.append('<a id="reading-guide"></a>')
        lines.append("")
        lines.extend(READING_GUIDE[2:])
        render_traceability(lines, units, max_case_ids)
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines).rstrip("\n") + "\n"
