#!/usr/bin/env python3
"""Collect Tensile benchmark CSVs into a flat summary.

Tensile writes under <out>/1_BenchmarkProblems/**/*.csv (and sometimes
2_BenchmarkData). We keep winner/time/gflops-ish columns when present and
tag each row with the param group + mode.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def find_csvs(out_dir: Path):
    patterns = [
        "1_BenchmarkProblems/**/*.csv",
        "2_BenchmarkData/**/*.csv",
        "**/*Benchmark*.csv",
        "**/*.csv",
    ]
    seen = set()
    for pat in patterns:
        for p in out_dir.glob(pat):
            if p.name.endswith("_Granularity.csv"):
                continue
            if "tensile_run" in p.name:
                continue
            if p in seen:
                continue
            # skip our own summaries
            if p.name.startswith("summary"):
                continue
            seen.add(p)
            yield p


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        # Tensile CSVs are occasionally irregular; use excel dialect
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        return reader.fieldnames or [], rows


def collect_one(out_dir: Path, group: str, mode: str) -> Path:
    out_dir = Path(out_dir)
    rows_out = []
    fields_union = ["param_group", "mode", "source_csv"]
    for csv_path in find_csvs(out_dir):
        try:
            fields, rows = load_csv(csv_path)
        except Exception as e:
            print(f"skip {csv_path}: {e}", file=sys.stderr)
            continue
        for f in fields:
            if f not in fields_union:
                fields_union.append(f)
        rel = str(csv_path.relative_to(out_dir))
        for r in rows:
            row = {"param_group": group, "mode": mode, "source_csv": rel}
            row.update({k: (r.get(k) or "") for k in fields})
            rows_out.append(row)

    summary = out_dir / f"summary_{mode}.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_union, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    n_csv = len({r["source_csv"] for r in rows_out})
    print(f"{group}: wrote {summary} ({len(rows_out)} rows from {n_csv} csvs)")
    return summary


def merge(root: Path, mode: str):
    root = Path(root)
    summaries = sorted(root.glob(f"*/summary_{mode}.csv"))
    if not summaries:
        # also accept results*/summary
        summaries = sorted(root.glob(f"*/results*/summary_{mode}.csv"))
    all_fields = []
    all_rows = []
    for s in summaries:
        with s.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for field in r.fieldnames or []:
                if field not in all_fields:
                    all_fields.append(field)
            all_rows.extend(list(r))
    out = root / f"summary_{mode}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"merged {len(summaries)} -> {out} ({len(all_rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", help="Tensile output directory for one group")
    ap.add_argument("--group", default="")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--merge", default="", help="Merge all group summaries under this root")
    args = ap.parse_args()
    if args.merge:
        merge(Path(args.merge), args.mode)
        return
    if not args.out_dir:
        ap.error("out_dir required unless --merge")
    collect_one(Path(args.out_dir), args.group or Path(args.out_dir).parent.name, args.mode)


if __name__ == "__main__":
    main()
