#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
T2.6 — Comparison report: Dispatcher vs Tile Engine sweep results.

Takes one or two Parquet files (one from sweep_runner.py for the dispatcher,
one optional from the TE benchmark runner) and produces a human-readable
Markdown (or HTML) report showing:

  • Per-shape rows: TE time, dispatcher time, % delta, validation verdict.
  • Roll-up tables by dtype, pipeline, tile size.
  • A one-line "is the port working?" summary at the top.

Usage:
    # Dispatcher-only report (no TE baseline):
    python compare_report.py results/dispatcher.parquet

    # Full comparison against TE baseline:
    python compare_report.py results/dispatcher.parquet --te results/te.parquet

    # Write HTML instead of Markdown:
    python compare_report.py results/dispatcher.parquet --format html -o report.html

    # Filter to a specific dtype:
    python compare_report.py results/dispatcher.parquet --dtype fp16

Output columns (Markdown table):
    Identifier | M×N×K | Verdict | Disp TFLOP/s | TE TFLOP/s | Delta % | Notes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"identifier", "M", "N", "K", "verdict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def _merge(disp: pd.DataFrame, te: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Join dispatcher and TE results on (identifier, M, N, K)."""
    key = ["identifier", "M", "N", "K"]
    disp = disp.copy()
    disp = disp.rename(columns={"verdict": "disp_verdict", "tflops": "disp_tflops",
                                 "error_msg": "disp_error"})
    if te is None:
        disp["te_tflops"] = None
        disp["te_verdict"] = None
        disp["delta_pct"] = None
        return disp

    te = te.copy()
    te = te.rename(columns={"verdict": "te_verdict", "tflops": "te_tflops",
                             "error_msg": "te_error"})
    merged = disp.merge(te[key + ["te_verdict", "te_tflops"]], on=key, how="left")

    def _delta(row):
        d, t = row["disp_tflops"], row["te_tflops"]
        if pd.notna(d) and pd.notna(t) and t != 0:
            return (d - t) / t * 100
        return None

    merged["delta_pct"] = merged.apply(_delta, axis=1)
    return merged


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

_PASS_EMOJI  = "✅"
_FAIL_EMOJI  = "❌"
_SKIP_EMOJI  = "⏭"
_WARN_EMOJI  = "⚠️"


def _verdict_icon(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    v = str(v)
    if v == "PASSED":
        return _PASS_EMOJI
    if v in ("FAILED", "ERROR"):
        return _FAIL_EMOJI
    if v in ("SKIPPED", "DRYRUN"):
        return _SKIP_EMOJI
    return v


def _fmt_tflops(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    return f"{float(v):.2f}"


def _fmt_delta(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    v = float(v)
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _rollup_table(df: pd.DataFrame, group_col: str, title: str) -> str:
    lines = [f"### {title}\n",
             f"| {group_col} | Total | Passed | Failed | Error | Skipped | Pass% |",
             f"|---|---|---|---|---|---|---|"]
    for val, grp in df.groupby(group_col):
        total   = len(grp)
        passed  = (grp["disp_verdict"] == "PASSED").sum()
        failed  = (grp["disp_verdict"] == "FAILED").sum()
        errored = (grp["disp_verdict"] == "ERROR").sum()
        skipped = grp["disp_verdict"].isin(["SKIPPED", "DRYRUN"]).sum()
        pct     = f"{passed/total*100:.1f}%" if total else "—"
        lines.append(f"| {val} | {total} | {passed} | {failed} | {errored} | {skipped} | {pct} |")
    return "\n".join(lines) + "\n"


def _build_markdown(df: pd.DataFrame, disp_path: Path,
                    te_path: Optional[Path], dtype_filter: Optional[str]) -> str:
    total   = len(df)
    passed  = (df["disp_verdict"] == "PASSED").sum()
    failed  = (df["disp_verdict"] == "FAILED").sum()
    errored = (df["disp_verdict"] == "ERROR").sum()
    skipped = df["disp_verdict"].isin(["SKIPPED", "DRYRUN"]).sum()
    pct     = f"{passed/total*100:.1f}%" if total else "—"

    overall_ok = failed == 0 and errored == 0
    status_line = (
        f"{_PASS_EMOJI} **Port verified** — {passed}/{total} combinations pass, "
        f"0 failures." if overall_ok else
        f"{_FAIL_EMOJI} **Failures present** — {passed}/{total} pass, "
        f"{failed+errored} fail/error."
    )

    lines = [
        "# Dispatcher ↔ Tile Engine Parity Report",
        "",
        f"**Dispatcher results:** `{disp_path}`",
    ]
    if te_path:
        lines.append(f"**TE baseline:** `{te_path}`")
    if dtype_filter:
        lines.append(f"**Filter:** dtype = `{dtype_filter}`")
    lines += [
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total (kernel × size) | {total} |",
        f"| Passed | {passed} |",
        f"| Failed | {failed} |",
        f"| Error | {errored} |",
        f"| Skipped | {skipped} |",
        f"| Pass rate | {pct} |",
        "",
        status_line,
        "",
    ]

    # Roll-up tables (spec T2.6: "by dtype and by layout"; also pipeline and tile)
    if "datatype" in df.columns:
        lines.append(_rollup_table(df, "datatype", "By dtype"))
        lines.append("")
    if "layout" in df.columns:
        lines.append(_rollup_table(df, "layout", "By layout"))
        lines.append("")
    if "pipeline" in df.columns:
        lines.append(_rollup_table(df, "pipeline", "By pipeline"))
        lines.append("")
    if {"tile_m", "tile_n", "tile_k"}.issubset(df.columns):
        df_tile = df.copy()
        df_tile["tile"] = (
            df_tile["tile_m"].astype(str) + "×"
            + df_tile["tile_n"].astype(str) + "×"
            + df_tile["tile_k"].astype(str)
        )
        lines.append(_rollup_table(df_tile, "tile", "By tile shape"))
        lines.append("")

    # Per-shape detail table (failures first, then skipped, then passing)
    lines += [
        "## Per-shape detail",
        "",
        "| Identifier | M×N×K | Disp | Disp TFLOP/s | TE TFLOP/s | Δ% | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    def sort_key(row):
        order = {"ERROR": 0, "FAILED": 1, "SKIPPED": 2, "DRYRUN": 3, "PASSED": 4}
        return order.get(row["disp_verdict"], 5)

    df_sorted = df.iloc[sorted(range(len(df)), key=lambda i: sort_key(df.iloc[i]))]

    for _, row in df_sorted.iterrows():
        ident   = row["identifier"]
        mnk     = f"{int(row['M'])}×{int(row['N'])}×{int(row['K'])}"
        icon    = _verdict_icon(row["disp_verdict"])
        d_tfl   = _fmt_tflops(row.get("disp_tflops"))
        te_tfl  = _fmt_tflops(row.get("te_tflops"))
        delta   = _fmt_delta(row.get("delta_pct"))
        note    = str(row.get("disp_error", "") or "")[:80]
        lines.append(f"| `{ident}` | {mnk} | {icon} | {d_tfl} | {te_tfl} | {delta} | {note} |")

    lines += ["", "---", "",
              "_Generated by `compare_report.py`._"]
    return "\n".join(lines) + "\n"


def _md_to_html(md: str) -> str:
    try:
        import markdown
        return f"<!DOCTYPE html><html><body>\n{markdown.markdown(md, extensions=['tables'])}\n</body></html>\n"
    except ImportError:
        # Fallback: wrap in pre
        return f"<!DOCTYPE html><html><body><pre>{md}</pre></body></html>\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dispatcher", type=Path, help="Dispatcher sweep Parquet (from sweep_runner.py)")
    ap.add_argument("--te", type=Path, default=None, metavar="TE_PARQUET",
                    help="Tile Engine sweep Parquet for comparison (optional)")
    ap.add_argument("--dtype", default=None, help="Filter to one datatype (e.g., fp16)")
    ap.add_argument("--format", choices=["markdown", "html"], default="markdown",
                    help="Output format (default: markdown)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Write report to file (default: print to stdout)")
    args = ap.parse_args()

    if not args.dispatcher.exists():
        print(f"error: {args.dispatcher} not found", file=sys.stderr)
        return 1

    try:
        disp = _load(args.dispatcher)
    except (ValueError, Exception) as e:
        print(f"error loading dispatcher Parquet: {e}", file=sys.stderr)
        return 1

    te = None
    if args.te:
        if not args.te.exists():
            print(f"error: {args.te} not found", file=sys.stderr)
            return 1
        try:
            te = _load(args.te)
        except (ValueError, Exception) as e:
            print(f"error loading TE Parquet: {e}", file=sys.stderr)
            return 1

    if args.dtype:
        if "datatype" in disp.columns:
            disp = disp[disp["datatype"] == args.dtype]
        if te is not None and "datatype" in te.columns:
            te = te[te["datatype"] == args.dtype]

    merged = _merge(disp, te)
    md = _build_markdown(merged, args.dispatcher, args.te, args.dtype)
    report = _md_to_html(md) if args.format == "html" else md

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Non-zero exit if failures present.
    failed = (merged["disp_verdict"].isin(["FAILED", "ERROR"])).sum()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
