"""
Grapher for primbench results: takes JSON and CSV files containing benchmark specialization
data and generates a comparison graph showing performance metrics across all
specializations. By default, shows relative comparison (percentage change). The graph displays
specializations on the Y-axis with their throughput changes, and bytes/second (or percentage)
on the X-axis. The legend lists all input files used.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pygal  # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame, read_csv
from pygal.style import DefaultStyle  # pyright: ignore[reportMissingTypeStubs]
from rich_argparse import ArgumentDefaultsRichHelpFormatter
from scipy import stats


def print_performance_stats(df: DataFrame, algo: str):
    """Calculates and prints the 'Good vs Bad' metrics with a 3% noise threshold."""
    old_column, new_column = df.columns[0], df.columns[-1]
    threshold = 3.0  # 3% noise threshold

    new = df[new_column]
    old = df[old_column]
    changes = ((new / old) - 1) * 100

    # Basic stats
    mean_change = np.mean(changes)
    median_change = np.median(changes)

    # Win rate (percentage of cases that improved)
    wins = np.sum(changes > 0)
    win_rate = (wins / len(changes)) * 100

    # 95% Confidence Interval
    conf_int = stats.t.interval(
        0.95, len(changes) - 1, loc=mean_change, scale=stats.sem(changes)
    )

    print("=" * 40)
    print(f"Algorithm:           {algo}")
    print(f"Mean Change:         {mean_change:+.2f}%")
    print(f"Median Change:       {median_change:+.2f}%")
    print(f"95% Conf. Int:       [{conf_int[0]:+.2f}%, {conf_int[1]:+.2f}%]")
    print(
        f"Improvement Rate:    {win_rate:.1f}% ({wins}/{len(changes)} specializations)"
    )

    # Verdict Logic with 3% threshold
    if conf_int[0] > threshold:
        print(f"Verdict:             SIGNIFICANT IMPROVEMENT (>{threshold}%)")
    elif conf_int[1] < -threshold:
        print(f"Verdict:             SIGNIFICANT REGRESSION (< -{threshold}%)")
    else:
        # If the confidence interval overlaps with the threshold, or the mean is inside it
        direction = "POSITIVE" if mean_change > 0 else "NEGATIVE"
        print(f"Verdict:             INSIGNIFICANT CHANGE (Trending {direction})")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsRichHelpFormatter)
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="paths to input .json or .csv primbench files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path to output .svg graph",
    )
    parser.add_argument(
        "--algo",
        type=str,
        help="algorithm name for the chart title (required if only CSV files are passed)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help="GPU arch name for the chart title (required if only CSV files are passed)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="regex pattern of specializations to include",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="perform absolute instead of relative comparison",
    )
    args = parser.parse_args()

    # Validate input files
    has_json = False
    for input_file in args.input_files:
        ext = Path(input_file).suffix.lower()
        if ext not in [".json", ".csv"]:
            raise ValueError(f"Input file must be .json or .csv, got: {input_file}")
        if ext == ".json":
            has_json = True

    # Validate output file
    output_ext = Path(args.output).suffix.lower()
    if output_ext != ".svg":
        raise ValueError(f"Output file must be a .svg, got: {args.output}")

    # Determine algorithm and arch for title
    algo: Optional[str] = None
    arch: Optional[str] = None
    if has_json:
        # Extract from the first JSON file's context
        for input_file in args.input_files:
            if Path(input_file).suffix.lower() == ".json":
                with open(input_file) as f:
                    data = json.load(f)
                algo = data["context"]["general"]["algorithm"]
                arch = data["context"]["general"]["gpu"]["arch"]
                break
    else:
        # Only CSV files, so algo and arch must be provided
        if not args.algo:
            raise ValueError(
                "--algo must be passed on the command line when only CSV files are passed"
            )
        if not args.arch:
            raise ValueError(
                "--arch must be passed on the command line when only CSV files are passed"
            )
        algo = args.algo
        arch = args.arch
    assert algo
    assert arch

    # Parse all input files
    dfs: List[Tuple[str, DataFrame]] = []
    for input_file in args.input_files:
        df = parse_primbench_file(input_file)
        check_duplicates(df, input_file)
        dfs.append((input_file, df))

    # Merge all dataframes
    merged_df = merge_dataframes(dfs)

    # Apply filter if provided
    if args.filter:
        merged_df = apply_filter(merged_df, args.filter)

    # Print stats before generating chart
    print_performance_stats(merged_df, algo)

    # Generate chart
    chart = get_chart(merged_df, algo, arch, args.absolute)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    chart.render_to_file(args.output)

    print(f"Graph saved to {args.output}")


def parse_primbench_file(file_path: str):
    """Parse a primbench JSON or CSV file and return a DataFrame."""
    ext = Path(file_path).suffix.lower()

    if ext == ".json":
        return parse_json(file_path)

    return parse_csv(file_path)


def parse_json(json_path: str):
    """Parse primbench JSON file into a DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    specializations = data["specializations"]

    rows: List[Dict[str, Union[str, float]]] = []
    for spec in specializations:
        rows.append(
            {
                "name": spec["name"],
                "bytes_per_second": spec["bytes_per_second"],
                "file": json_path,
            }
        )

    return DataFrame(rows)


def parse_csv(csv_path: str):
    """Parse primbench CSV file into a DataFrame."""
    df = read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["name", "bytes_per_second"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df["file"] = csv_path
    return df.loc[:, ["name", "bytes_per_second", "file"]]


def check_duplicates(df: DataFrame, file_path: str):
    """Check for duplicate 'name' entries in a DataFrame and exit if any found."""
    duplicated_mask = df.duplicated(subset=["name"], keep=False)
    if duplicated_mask.any():
        duplicate_names = df.loc[duplicated_mask, "name"]
        first_duplicate = duplicate_names.iloc[0]
        exit(
            f"ERROR: duplicate parameter name found in {file_path}:\n"
            f"  {first_duplicate}"
        )


def merge_dataframes(dfs: List[Tuple[str, DataFrame]]):
    """Merge multiple DataFrames from different files."""
    merged: Optional[DataFrame] = None

    for file_path, df in dfs:
        current_df = df.copy()
        current_df["index_label"] = current_df["name"]
        current_df = current_df.set_index("index_label")
        current_df.index.name = None

        # Use filename as column name for bytes_per_second values
        renamed_df = current_df[["bytes_per_second"]]
        renamed_df.columns = [file_path]
        current_df = cast(DataFrame, renamed_df)

        if merged is None:
            merged = current_df
        else:
            merged = merged.merge(
                current_df, how="outer", left_index=True, right_index=True
            )

            nan_rows = merged[merged.isna().any(axis=1)]
            if not nan_rows.empty:
                key = nan_rows.index[0]

                row = merged.loc[[key]]

                missing_cols = row.columns[row.isna().any(axis=0)].tolist()
                present_cols = row.columns[row.notna().all(axis=0)].tolist()

                exit(
                    f"ERROR: key mismatch while merging files:\n"
                    f"  Key: {key}\n"
                    f"  Present in: {present_cols[0] if present_cols else 'NONE'}\n"
                    f"  Missing from: {missing_cols[0] if missing_cols else 'NONE'}"
                )

    assert merged is not None
    return merged.sort_index()


def apply_filter(df: DataFrame, pattern: str):
    """Filter DataFrame rows based on regex pattern matching specialization names."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    # Keep rows where the index (specialization name) matches the pattern
    mask = df.index.str.contains(regex)
    filtered_df = cast(DataFrame, df[mask])

    if filtered_df.empty:
        raise ValueError(f"No specializations matched the filter pattern: {pattern}")

    return filtered_df


def get_chart(df: DataFrame, algo: str, arch: str, absolute: bool):
    """Generate a pygal HorizontalBar chart from the merged DataFrame."""

    old_column, new_column = df.columns[0], df.columns[-1]
    old_series = df[old_column]
    new_series = df[new_column]

    percent_change = ((new_series / old_series) - 1) * 100

    # Sort DataFrame and percent_change by percent_change
    sort_order = percent_change.sort_values(ascending=False).index
    df = df.loc[sort_order]
    percent_change = percent_change.loc[sort_order]

    style = DefaultStyle(label_font_size=11)

    if absolute:
        # Absolute mode: show actual bytes/sec values
        data_df = df
        chart = pygal.HorizontalBar(
            style=style,
            x_title="Throughput (bytes/sec)",
            value_formatter=lambda x: f"{x:.2e}",
        )
        bar_count = len(df.columns)

        # Add all series normally
        for column in data_df.columns:
            chart.add(column, data_df[column].tolist())
    else:
        # Relative mode: show percentage change relative to baseline
        baseline = df.iloc[:, 0]

        relative_series = (new_series / baseline - 1) * 100

        data_df = relative_series.to_frame(name=f"{old_column} vs {new_column}")

        chart = pygal.HorizontalBar(
            # The "black" here sets the legend dot's color
            style=DefaultStyle(label_font_size=11, colors=["black"]),
            x_title="Throughput change (%)",
            value_formatter=lambda x: f"{x:+.1f}%" if x != 0 else "0%",
        )
        bar_count = 1

        # Retrieve Pygal's default colors from the style
        negative_color = style.colors[0]
        positive_color = style.colors[1]

        # Add bars with tooltips showing all bytes/sec
        bars_with_tooltip: List[Dict[str, Union[str, float]]] = []
        for _, row in df.iterrows():
            val = float(((row[new_column] / row[old_column]) - 1) * 100)

            color = positive_color if val >= 0 else negative_color

            tooltip_text = ", ".join([f"{col}: {row[col]:.2e}" for col in df.columns])

            bars_with_tooltip.append(
                {"value": val, "color": color, "label": tooltip_text}
            )

        chart.add(data_df.columns[0], bars_with_tooltip)

    # Let x_labels include percent change
    chart.x_labels = [
        f"{label}, {change:+.1f}%" for label, change in zip(df.index, percent_change)
    ]

    # Title with worst/best percent change
    chart.title = f"{algo} {arch} (worst {percent_change.min():+.1f}%, best {percent_change.max():+.1f}%)"

    # Common styling
    longest_label = max(df.index, key=len)
    chart.width = 15 * len(longest_label) + 500
    chart.height = 15 * len(df.index) * bar_count + 200
    chart.legend_at_bottom = True
    chart.truncate_legend = False
    chart.legend_at_bottom_columns = 1

    return chart


if __name__ == "__main__":
    main()