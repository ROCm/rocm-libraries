#!/usr/bin/env python3
"""V1/V2/V3/V4 4-way decision-bench analyzer.

Reads CSVs from decision_bench_v4.sh and emits a concise markdown report:
  - Equivalence: per workload, all 4 versions' exit codes match.
  - Compile time: median ms per (workload, variant, version) + V4-vs-V3 delta.
  - Runtime: median ms per (workload, variant, version) + V4-vs-V3 delta.
  - Codegen: ASM lines / VGPR / SGPR / scratch per version + V4-vs-V3 delta.
  - Pairwise Wilcoxon: V4 vs V3 (V3 was the winner of the V1/V2/V3 sweep).

Usage: decision_analyze_v4.py <data_dir>
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

WORKLOADS = ["simple", "medium", "complex"]
VARIANTS  = ["literal", "placeholder"]
VERSIONS  = ["v1", "v2", "v3", "v4"]
P_THRESHOLD = 0.02


def load_paired(path, value_col):
    out = defaultdict(list)
    if not path.exists():
        return {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["version"]].append(float(row[value_col]))
    return {v: np.array(out[v]) for v in VERSIONS if v in out}


def load_singlerow(path, fields):
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["version"]] = {k: row.get(k, "?") for k in fields}
    return out


def median_str(arr):
    if len(arr) == 0:
        return "n/a"
    return f"{np.median(arr):.2f}"


def pct_delta(new, ref):
    if ref == 0:
        return "n/a"
    return f"{100 * (new - ref) / ref:+.1f}%"


def wilcoxon_paired(new, ref):
    """Returns (p, median_pct_delta) on round-paired arrays."""
    n = min(len(new), len(ref))
    if n < 2:
        return (None, None)
    a = np.asarray(new[:n])
    b = np.asarray(ref[:n])
    diff = a - b
    if np.all(diff == 0):
        return (1.0, 0.0)
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return (None, None)
    median_delta = float(np.median(diff))
    return (float(p), median_delta)


def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    args = ap.parse_args()

    out = []
    out.append("# V1 / V2 / V3 / V4 Decision Benchmark Report")
    out.append("")
    out.append(f"Data directory: `{args.data_dir}`")
    out.append("")
    out.append(f"Statistical methodology: paired Wilcoxon signed-rank, "
               f"two-tailed, p < {P_THRESHOLD} threshold. Comparisons are "
               f"V4-vs-V3 (V3 was the V1/V2/V3 sweep winner).")
    out.append("")

    # --------------------------------------------------------------- verify
    out.append("## Equivalence")
    out.append("")
    out.append("| Workload / Variant | V1 | V2 | V3 | V4 | Equiv |")
    out.append("|---|---|---|---|---|---|")
    all_equiv = True
    for wl in WORKLOADS:
        for var in VARIANTS:
            path = args.data_dir / f"verify_{wl}_{var}.csv"
            d = load_singlerow(path, ["exit_code"])
            rcs = {v: d.get(v, {}).get("exit_code", "?") for v in VERSIONS}
            unique = set(rcs.values())
            equiv = "PASS" if len(unique) == 1 and "?" not in unique else "**FAIL**"
            if equiv != "PASS":
                all_equiv = False
            out.append(f"| {wl} / {var} | {rcs['v1']} | {rcs['v2']} | {rcs['v3']} | {rcs['v4']} | {equiv} |")
    out.append("")
    out.append(f"**Overall equivalence: {'ALL PASS' if all_equiv else 'FAIL'}**")
    out.append("")

    # --------------------------------------------------------------- codegen
    out.append("## Codegen (gfx90a)")
    out.append("")
    fields = ["asm_lines", "vgpr", "sgpr", "scratch"]
    for wl in WORKLOADS:
        for var in VARIANTS:
            out.append(f"### {wl} / {var}")
            out.append("")
            out.append("| Version | ASM lines | VGPR | SGPR | Scratch |")
            out.append("|---|---|---|---|---|")
            d = load_singlerow(args.data_dir / f"codegen_{wl}_{var}.csv", fields)
            v3_row = d.get("v3", {})
            for v in VERSIONS:
                row = d.get(v, {})
                if not row:
                    out.append(f"| {v} | ? | ? | ? | ? |")
                    continue
                cells = [row.get(f, "?") for f in fields]
                if v == "v4" and v3_row:
                    deltas = []
                    for f, c in zip(fields, cells):
                        try:
                            new = int(c); ref = int(v3_row.get(f, 0))
                            d_pct = pct_delta(new, ref)
                            deltas.append(f"{c} ({d_pct})")
                        except (ValueError, TypeError):
                            deltas.append(c)
                    out.append(f"| **v4** (vs v3) | {deltas[0]} | {deltas[1]} | {deltas[2]} | {deltas[3]} |")
                else:
                    out.append(f"| {v} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
            out.append("")

    # --------------------------------------------------------------- compile
    out.append("## Compile time (median ms over 10 paired rebuilds)")
    out.append("")
    out.append("| Workload / Variant | V1 | V2 | V3 | V4 | V4 vs V3 | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for var in VARIANTS:
            d = load_paired(args.data_dir / f"compile_{wl}_{var}.csv", "compile_ms")
            row = [median_str(d.get(v, np.array([]))) for v in VERSIONS]
            v3_arr = d.get("v3", np.array([]))
            v4_arr = d.get("v4", np.array([]))
            if len(v3_arr) and len(v4_arr):
                p, md = wilcoxon_paired(v4_arr, v3_arr)
                d_pct = pct_delta(np.median(v4_arr), np.median(v3_arr))
                p_str = fmt_p(p)
                if p is not None and p < P_THRESHOLD:
                    d_pct = f"**{d_pct}**"
                    p_str = f"**{p_str}**"
            else:
                d_pct = "n/a"
                p_str = "n/a"
            out.append(f"| {wl} / {var} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {d_pct} | {p_str} |")
    out.append("")

    # --------------------------------------------------------------- runtime
    out.append("## Runtime (median ms over 20 paired rounds, 10 000 mapCoord/thread)")
    out.append("")
    out.append("| Workload / Variant | V1 | V2 | V3 | V4 | V4 vs V3 | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|")
    for wl in WORKLOADS:
        for var in VARIANTS:
            d = load_paired(args.data_dir / f"runtime_{wl}_{var}.csv", "runtime_ms")
            row = [median_str(d.get(v, np.array([]))) for v in VERSIONS]
            v3_arr = d.get("v3", np.array([]))
            v4_arr = d.get("v4", np.array([]))
            if len(v3_arr) and len(v4_arr):
                p, md = wilcoxon_paired(v4_arr, v3_arr)
                d_pct = pct_delta(np.median(v4_arr), np.median(v3_arr))
                p_str = fmt_p(p)
                if p is not None and p < P_THRESHOLD:
                    d_pct = f"**{d_pct}**"
                    p_str = f"**{p_str}**"
            else:
                d_pct = "n/a"
                p_str = "n/a"
            out.append(f"| {wl} / {var} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {d_pct} | {p_str} |")
    out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
