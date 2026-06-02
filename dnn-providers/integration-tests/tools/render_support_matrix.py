# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Render a combined engine support matrix in markdown from one or more
`*.supported.toml` sidecars.

The schema is documented in
`dnn-providers/integration-tests/docs/support-claims-schema.md`. Each
sidecar's `[meta].engine` field becomes a column in the rendered matrix.

Two layout styles are available via `--style`:

* `zoom` (default) — Google-Maps-style. An overview table at the top
  shows each op family's coverage fraction per engine (e.g. `✅ 30/30`,
  `🟡 27/30`, `—`). Each op family expands into a per-variant layout
  table via a `<details>` disclosure widget, and that table further
  expands into the full per-(variant, dtype) detail. Requires a viewer
  that renders `<details>` (GitHub, GitLab, VS Code preview, etc.).
* `classic` — one row per (op_chain, dtype_combo). Verbose but lookup-
  friendly; useful for forensic "did the engine return support for this
  exact tuple?" questions.

Usage:
    python render_support_matrix.py \\
        MIOPEN_ENGINE.supported.toml \\
        HIPBLASLT_ENGINE.supported.toml \\
        HIP_KERNEL_ENGINE.supported.toml \\
        -o SupportMatrix.md
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

CHECK = "✅"


# ---------------------------------------------------------------------------
# Loading / common shaping
# ---------------------------------------------------------------------------


def describe_graph(op_chain: str, combo: dict) -> str:
    io_dt = combo["io"]
    out_dt = combo.get("output")
    compute = combo["compute"]
    intermediate = combo.get("intermediate")
    if out_dt is None or out_dt == io_dt:
        head = f"[io={io_dt}"
    else:
        head = f"[in={io_dt}, out={out_dt}"
    head += f", compute={compute}"
    if intermediate:
        head += f", intermediate={intermediate}"
    head += "]"
    return f"{op_chain} {head}"


def section_key(op_chain: str) -> str:
    """Section heading: bare name of the first op (strip [flags] and :MODE)."""
    first = op_chain.split(" ", 1)[0]
    first = first.split("[", 1)[0]
    first = first.split(":", 1)[0]
    return first


def variant_of(op_chain: str) -> str:
    """For an op_chain, return its variant suffix (e.g. '[1x1]' or
    ' + Pointwise:RELU_FWD[lower_clip]'), or '(bare)' if none."""
    base = section_key(op_chain)
    suffix = op_chain[len(base) :]
    return suffix if suffix else "(bare)"


def combo_key(combo: dict) -> tuple:
    return (
        combo.get("io"),
        combo.get("output"),
        combo.get("compute"),
        combo.get("intermediate"),
    )


def combo_to_str(combo: dict) -> str:
    io_dt = combo["io"]
    out_dt = combo.get("output")
    compute = combo["compute"]
    intermediate = combo.get("intermediate")
    if out_dt is None or out_dt == io_dt:
        parts = [f"io={io_dt}"]
    else:
        parts = [f"in={io_dt}", f"out={out_dt}"]
    parts.append(f"compute={compute}")
    if intermediate:
        parts.append(f"intermediate={intermediate}")
    return "[" + ", ".join(parts) + "]"


def load_sidecar(path: Path) -> tuple[str, int, list[dict]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    engine = meta.get("engine")
    if not engine:
        raise SystemExit(f"{path}: [meta].engine is required")
    return engine, meta.get("version"), data.get("supported", [])


def collect_engine_rows(blocks: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict[tuple[str, tuple], tuple]] = defaultdict(dict)
    for block in blocks:
        arch = block["arch"]
        platform = block.get("platform", "any")
        key = (arch, platform)
        for matcher in block.get("matchers", []):
            op_chains = matcher["op_chains"]
            dtype_combos = matcher["dtype_combos"]
            layouts = tuple(sorted(set(matcher["layouts"])))
            for op_chain in op_chains:
                for combo in dtype_combos:
                    ck = combo_key(combo)
                    existing = out[key].get((op_chain, ck))
                    if existing is None:
                        out[key][(op_chain, ck)] = (dict(combo), layouts)
                    else:
                        merged = tuple(sorted(set(existing[1]) | set(layouts)))
                        out[key][(op_chain, ck)] = (existing[0], merged)
    return out


def assemble_master(parsed):
    """Returns (engine_cols, per_block_rows, per_block_support).

    per_block_rows: (arch, platform) -> {(op_chain, combo_key): combo_dict}
    per_block_support: (arch, platform) -> {(op_chain, combo_key): {engine: layouts_tuple}}
    """
    engine_cols = [engine for _, engine, _, _ in parsed]
    if len(set(engine_cols)) != len(engine_cols):
        raise SystemExit(f"Duplicate engine name in sidecar set: {engine_cols}.")

    per_block_support: dict[
        tuple[str, str], dict[tuple[str, tuple], dict[str, tuple]]
    ] = defaultdict(lambda: defaultdict(dict))
    per_block_rows: dict[tuple[str, str], dict[tuple[str, tuple], dict]] = defaultdict(
        dict
    )
    empty_engines: list[str] = []

    for _path, engine, _version, blocks in parsed:
        rows_by_ap = collect_engine_rows(blocks)
        if not rows_by_ap:
            empty_engines.append(engine)
            continue
        for ap_key, rows in rows_by_ap.items():
            for row_key, (combo_dict, layouts) in rows.items():
                per_block_support[ap_key][row_key][engine] = layouts
                per_block_rows[ap_key].setdefault(row_key, combo_dict)

    # Backfill empty engines so columns appear with all `-`.
    for ap_key in per_block_support:
        for engine in empty_engines:
            for row_key in per_block_support[ap_key]:
                per_block_support[ap_key][row_key].setdefault(engine, ())

    return engine_cols, per_block_rows, per_block_support


def support_cell(layouts: tuple[str, ...]) -> str:
    return "—" if not layouts else f"{CHECK} {', '.join(layouts)}"


def write_header(engine_cols: list[str], sidecars: list[Path], style: str) -> list[str]:
    parts: list[str] = []
    title = (
        f"# {engine_cols[0]} Engine Support Matrix"
        if len(engine_cols) == 1
        else "# Combined Engine Support Matrix"
    )
    parts.append(title + "\n")
    source_list = ", ".join(f"`{p.name}`" for p in sidecars)
    suffix = "" if style == "zoom" else f" --style {style}"
    parts.append(
        f"Generated by `render_support_matrix.py{suffix}` from {source_list}. "
        "Do not hand-edit — regenerate from the sidecars when they change.\n"
    )
    return parts


# ---------------------------------------------------------------------------
# Coverage stats shared by both renderers
# ---------------------------------------------------------------------------


def family_stats(rows, support_for_row, engine_cols):
    """Op-family -> engine -> (supported_count, total_observed)."""
    total = len(rows)
    return {
        e: (sum(1 for rk in rows if support_for_row[rk].get(e)), total)
        for e in engine_cols
    }


def family_indicator(n: int, total: int) -> str:
    if total == 0 or n == 0:
        return "—"
    if n == total:
        return f"✅ {n}/{total}"
    return f"🟡 {n}/{total}"


def variant_layouts(op_chain, rows_in_family, support_for_row, engine_cols):
    """For one op_chain, return engine -> (layouts_tuple, all_dtypes_agree)."""
    matching = [rk for rk in rows_in_family if rk[0] == op_chain]
    result = {}
    for e in engine_cols:
        layout_sets = [tuple(support_for_row[rk].get(e, ())) for rk in matching]
        non_empty = [ls for ls in layout_sets if ls]
        agree = len({tuple(ls) for ls in layout_sets}) <= 1
        if not non_empty:
            result[e] = ((), agree)
        else:
            union = tuple(sorted({l for ls in non_empty for l in ls}))
            result[e] = (union, agree)
    return result


def family_dtype_summary(rows_in_family, combos_for_row) -> str:
    seen = set()
    combos = []
    for rk in rows_in_family:
        if rk[1] not in seen:
            seen.add(rk[1])
            combos.append(combos_for_row[rk])
    return ", ".join(combo_to_str(c) for c in combos)


# ---------------------------------------------------------------------------
# Style: classic — one row per (op_chain, dtype_combo)
# ---------------------------------------------------------------------------


def render_classic(engine_cols, per_block_rows, per_block_support, sidecars):
    parts = write_header(engine_cols, sidecars, "classic")
    if not per_block_rows:
        parts.append("_No `[[supported]]` blocks across the provided sidecars._\n")
        return "\n".join(parts)

    for arch, platform in sorted(per_block_rows):
        parts.append(f"## {arch} / {platform}\n")
        sections: dict[str, list] = {}
        for rk in sorted(per_block_rows[(arch, platform)]):
            sections.setdefault(section_key(rk[0]), []).append(rk)

        for sec_name, rows in sections.items():
            parts.append(f"### {sec_name}\n")
            parts.append("| Operations |" + "".join(f" {e} |" for e in engine_cols))
            parts.append(
                "|------------|"
                + "".join("-" * (len(e) + 2) + "|" for e in engine_cols)
            )
            for rk in rows:
                combo = per_block_rows[(arch, platform)][rk]
                desc = describe_graph(rk[0], combo)
                cells = " | ".join(
                    support_cell(per_block_support[(arch, platform)][rk].get(e, ()))
                    for e in engine_cols
                )
                parts.append(f"| {desc} | {cells} |")
            parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Style: zoom — Google-Maps-style overview + nested <details>
# ---------------------------------------------------------------------------


def render_zoom(engine_cols, per_block_rows, per_block_support, sidecars):
    parts = write_header(engine_cols, sidecars, "zoom")
    parts.append(
        "_Zoom out: each row in the overview is one op family. Click a "
        "disclosure triangle to expand into per-variant layout detail, and "
        "again for full per-(variant, dtype) rows._\n"
    )
    if not per_block_rows:
        parts.append("_No `[[supported]]` blocks across the provided sidecars._\n")
        return "\n".join(parts)

    for arch, platform in sorted(per_block_rows):
        parts.append(f"## {arch} / {platform}\n")
        combos_for_row = per_block_rows[(arch, platform)]
        support_for_row = per_block_support[(arch, platform)]

        sections: dict[str, list] = {}
        for rk in combos_for_row:
            sections.setdefault(section_key(rk[0]), []).append(rk)

        # ---- Level 1: overview table ----
        parts.append("### Overview\n")
        parts.append("| Op family |" + "".join(f" {e} |" for e in engine_cols))
        parts.append(
            "|-----------|" + "".join("-" * (len(e) + 2) + "|" for e in engine_cols)
        )
        for sec_name in sorted(sections):
            stats = family_stats(sections[sec_name], support_for_row, engine_cols)
            cells = " | ".join(family_indicator(*stats[e]) for e in engine_cols)
            parts.append(f"| **{sec_name}** | {cells} |")
        parts.append("")

        # ---- Level 2/3: per-family <details> ----
        for sec_name in sorted(sections):
            rows = sections[sec_name]
            stats = family_stats(rows, support_for_row, engine_cols)
            summary_bits = " · ".join(
                f"{e} {family_indicator(*stats[e])}" for e in engine_cols
            )
            parts.append(
                f"<details>\n<summary>📂 <b>{sec_name}</b> — {summary_bits}</summary>\n"
            )

            # Per-variant layout table.
            unique_ops = sorted({rk[0] for rk in rows})
            parts.append("| Variant |" + "".join(f" {e} |" for e in engine_cols))
            parts.append(
                "|---------|" + "".join("-" * (len(e) + 2) + "|" for e in engine_cols)
            )
            footnote = False
            for op in unique_ops:
                v = variant_of(op)
                layouts_by_engine = variant_layouts(
                    op, rows, support_for_row, engine_cols
                )
                cells = []
                for e in engine_cols:
                    layouts, agree = layouts_by_engine[e]
                    if not layouts:
                        cells.append("—")
                    else:
                        marker = "" if agree else " ⚠"
                        cells.append(f"✅ {', '.join(layouts)}{marker}")
                        if not agree:
                            footnote = True
                parts.append(f"| `{v}` | {' | '.join(cells)} |")
            parts.append("")
            parts.append(
                f"_Dtypes observed in this family: "
                f"{family_dtype_summary(rows, combos_for_row)}._"
            )
            if footnote:
                parts.append(
                    "_⚠ marks a (variant, engine) pair whose layout coverage "
                    "differs by dtype — expand the per-row detail below to see "
                    "which dtypes._"
                )
            parts.append("")

            # Per-(variant, dtype) detail nested below.
            parts.append(
                "<details>\n<summary>🔎 per-(variant, dtype) detail</summary>\n"
            )
            parts.append("| Operations |" + "".join(f" {e} |" for e in engine_cols))
            parts.append(
                "|------------|"
                + "".join("-" * (len(e) + 2) + "|" for e in engine_cols)
            )
            for rk in sorted(rows):
                combo = combos_for_row[rk]
                desc = describe_graph(rk[0], combo)
                cells = " | ".join(
                    support_cell(support_for_row[rk].get(e, ())) for e in engine_cols
                )
                parts.append(f"| {desc} | {cells} |")
            parts.append("")
            parts.append("</details>")
            parts.append("")
            parts.append("</details>")
            parts.append("")
    return "\n".join(parts)


STYLE_RENDERERS = {
    "zoom": render_zoom,
    "classic": render_classic,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sidecars", type=Path, nargs="+", help="One or more *.supported.toml sidecars"
    )
    parser.add_argument(
        "--style",
        choices=list(STYLE_RENDERERS),
        default="zoom",
        help="Output style (default: zoom)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output markdown path"
    )
    args = parser.parse_args()

    parsed = [(p, *load_sidecar(p)) for p in args.sidecars]
    engine_cols, per_block_rows, per_block_support = assemble_master(parsed)

    md = STYLE_RENDERERS[args.style](
        engine_cols, per_block_rows, per_block_support, args.sidecars
    )
    if args.output is None:
        sys.stdout.write(md)
    else:
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
