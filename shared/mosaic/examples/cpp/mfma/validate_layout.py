#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import difflib
import sys
from pathlib import Path


def read_csv(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        print(f"Error: '{path}' not found.", file=sys.stderr)
        sys.exit(1)


def normalize_layout(text: str) -> str:
    lines = text.splitlines()
    normalized = []
    header_normalized = False

    for line in lines:
        if not header_normalized:
            columns = line.split(",")
            if columns and columns[0].strip().lower() == "lane":
                normalized_columns = ["lane"]
                normalized_columns.extend(
                    f"column_{idx}" for idx in range(1, len(columns))
                )
                normalized.append(",".join(normalized_columns))
                header_normalized = True
                continue
        normalized.append(line)

    return "\n".join(normalized)


def compare_csvs(a_path: Path, b_path: Path) -> int:
    a_text = read_csv(a_path)
    b_text = read_csv(b_path)

    normalized_a = normalize_layout(a_text)
    normalized_b = normalize_layout(b_text)

    if normalized_a == normalized_b:
        if a_text == b_text:
            print("Layouts are identical.")
        else:
            print("Layouts match when ignoring column names.")
        return 0

    print("Layouts differ!")
    a_lines = normalized_a.splitlines()
    b_lines = normalized_b.splitlines()
    diff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile=str(a_path),
        tofile=str(b_path),
        lineterm="",
    )
    for line in diff:
        print(line)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that two MFMA layout CSV files are identical."
    )
    parser.add_argument("csv_a", type=Path, help="Path to first layout CSV")
    parser.add_argument("csv_b", type=Path, help="Path to second layout CSV")

    args = parser.parse_args()
    return compare_csvs(args.csv_a, args.csv_b)


if __name__ == "__main__":
    sys.exit(main())
