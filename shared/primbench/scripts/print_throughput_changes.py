import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from pandas import DataFrame, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "old_csv_dir",
        type=Path,
        help="directory containing old primbench CSV files",
    )
    parser.add_argument(
        "new_csv_dir",
        type=Path,
        help="directory containing new primbench CSV files",
    )
    parser.add_argument(
        "--improvements",
        action="store_true",
        help="print improvements instead of regressions",
    )
    args = parser.parse_args()

    if not args.old_csv_dir.is_dir():
        raise ValueError(f"Not a directory: {args.old_csv_dir}")
    if not args.new_csv_dir.is_dir():
        raise ValueError(f"Not a directory: {args.new_csv_dir}")

    old_csvs = {p.name: p for p in args.old_csv_dir.glob("*.csv")}
    new_csvs = {p.name: p for p in args.new_csv_dir.glob("*.csv")}

    common_files = sorted(old_csvs.keys() & new_csvs.keys())

    if not common_files:
        raise SystemExit("ERROR: no matching CSV filenames found between directories")

    results: List[Dict[str, Any]] = []

    for filename in common_files:
        old_path = old_csvs[filename]
        new_path = new_csvs[filename]

        dfs = [
            (str(old_path), parse_csv(str(old_path))),
            (str(new_path), parse_csv(str(new_path))),
        ]

        df = merge_dataframes(dfs)

        old_column, new_column = df.columns[0], df.columns[-1]

        percent_change = ((df[new_column] / df[old_column]) - 1) * 100

        min_idx = percent_change.idxmin()
        max_idx = percent_change.idxmax()

        min_change = percent_change.loc[min_idx]
        max_change = percent_change.loc[max_idx]

        algo = filename.removeprefix("benchmark_").removesuffix(".csv")
        algo = re.sub(r"_gfx\d+$", "", algo)

        results.append(
            {
                "algo": algo,
                "min": min_change,
                "max": max_change,
                "min_spec": min_idx,
                "max_spec": max_idx,
            }
        )

    if args.improvements:
        results.sort(key=lambda x: x["max"], reverse=True)
    else:
        results.sort(key=lambda x: x["min"])

    for result in results:
        percent = result["max"] if args.improvements else result["min"]
        print(f"{percent:+.1f}%: {result['algo']}")

        specialization = result["max_spec"] if args.improvements else result["min_spec"]
        print(f"    {specialization}")


def parse_csv(csv_path: str) -> DataFrame:
    """Parse primbench CSV file into a DataFrame."""
    df = read_csv(csv_path)
    df["file"] = csv_path
    return cast(DataFrame, df[["name", "bytes_per_second", "file"]])


def merge_dataframes(dfs: List[Tuple[str, DataFrame]]) -> DataFrame:
    """Merge multiple DataFrames from different files."""
    merged: Optional[DataFrame] = None

    for file_path, df in dfs:
        df = df.copy()
        df["index_label"] = df["name"]
        df = df.set_index("index_label")
        df.index.name = None

        subset = cast(DataFrame, df[["bytes_per_second"]])
        df = subset.rename(columns={"bytes_per_second": file_path})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, how="outer", left_index=True, right_index=True)

            nan_rows = merged[merged.isna().any(axis=1)]
            if not nan_rows.empty:
                key = nan_rows.index[0]
                row = merged.loc[[key]]

                missing_cols = row.columns[row.isna().any(axis=0)].tolist()
                present_cols = row.columns[row.notna().all(axis=0)].tolist()

                raise SystemExit(
                    f"ERROR: key mismatch while merging files:\n"
                    f"  Key: {key}\n"
                    f"  Present in: {present_cols[0] if present_cols else 'NONE'}\n"
                    f"  Missing from: {missing_cols[0] if missing_cols else 'NONE'}"
                )

    assert merged is not None
    return merged.sort_index()


if __name__ == "__main__":
    main()