# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Utility functions for benchmark output parsing and processing.

Provides functions to parse hipBLASLt benchmark output files into structured
DataFrames, handling both individual files and batch processing of directories.

Functions:
    parse_benchmark_output: Parse individual benchmark output file.
    parse_benchmark_output_dir: Parse all benchmark files in a directory.
    as_dashboard_format: Convert DataFrame to Executive Dashboard format.
"""

import pandas as pd
import io
import re

from pathlib import Path
from geko.constants import GEMM_FIELDS, PERF_FIELDS
import logging
logger = logging.getLogger("GEKO")

# Quote parenthesized complex values like (1,0) so embedded commas do not
# split CSV fields.
_COMPLEX_CSV_RE = re.compile(r'\(([^()]*,[^()]*)\)')

import yaml
try:
    SafeLoader = yaml.CSafeLoader
except (ModuleNotFoundError, AttributeError):
    SafeLoader = yaml.SafeLoader

def parse_benchmark_output(file: str | Path) -> pd.DataFrame:
    """
    Parse hipBLASLt benchmark output file into a pandas DataFrame.

    Args:
        file (str | Path): Path to the hipBLASLt benchmark output file.

    Returns:
        pandas.DataFrame: DataFrame containing parsed benchmark results.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If the input file content is not valid.
    """

    with open(file) as f:
        blocks = f.read().split("]:")[1:]

    if len(blocks) == 0:
        raise ValueError("The benchmark output file does not have the correct format")

    try:
        header = blocks[0].split("\n")[0]
        data = [header] + [b.split("\n")[1].strip() for b in blocks]
        data = [_COMPLEX_CSV_RE.sub(r'"(\1)"', line) for line in data]
        df = pd.read_csv(io.StringIO("\n".join(data)))
        
        kernel_col = []
        solution_col = []
        solution_idx_col = []
        for b in blocks:
            for line in b.split("\n"):
                if "--kernel name" in line:
                    kernel_col.append(line.split("--kernel name:")[1].strip())
                if "--Solution name" in line:
                    solution_col.append(line.split("--Solution name:")[1].strip())
                if "--Solution index" in line:
                    solution_idx_col.append(line.split("--Solution index:")[1].strip())
        if kernel_col:
            df["kernel"] = kernel_col
        if solution_col:
            df["solution"] = solution_col
        if solution_idx_col:
            df["solutionIdx"] = solution_idx_col

        required_cols = list(GEMM_FIELDS) + list(PERF_FIELDS)
        if not all(c in df.columns for c in required_cols) or df[required_cols].isnull().values.any():
            raise ValueError(f"The benchmark output is not complete")

    except (pd.errors.EmptyDataError, IndexError) as e:
        raise ValueError("The benchmark output file may be corrupted")

    return df


def parse_benchmark_output_dir(benchmark_dir: str | Path, suffix: str = "") -> pd.DataFrame:
    """Parse all benchmark output files in a directory into a single DataFrame.

    Args:
        benchmark_dir (str | Path): Directory containing benchmark output files.
        suffix (str, optional): Filename pattern to filter files.
            Defaults to "".

    Returns:
        pd.DataFrame: Combined DataFrame with all benchmark results.

    Raises:
        ValueError: If no valid latency log files found in directory.
    """
    benchmark_dir = Path(benchmark_dir)
    results = []
    for f in benchmark_dir.glob(f"*{suffix}"):
        if f.suffix == ".yaml":
            continue
        results.append(parse_benchmark_output(f))

    if len(results) == 0:
        raise ValueError(f"No latency logs found in '{benchmark_dir}'")

    return pd.concat(results, ignore_index=True).reset_index(drop=True)


def as_dashboard_format(df: pd.DataFrame) -> pd.DataFrame:
    """Converts input DataFrame to a format that can be loaded by the
       Executive Dashboard.

    Args:
        df (DataFrame): Input DataFrame to convert.

    Returns:
        pd.DataFrame: Converted DataFrame.

    """
    df = df[[c for c in df.columns if "reference" not in c]].copy()
    df.drop(["lib", "ratio", "error_tuned"], axis=1, errors="ignore", inplace=True)
    df.rename(
        columns={k: k.removesuffix("_tuned") for k in df.columns if k.endswith("_tuned")},
        inplace=True,
    )
    df.rename(columns={"kernel": "solution_name", "solutionIdx": "solution_index"}, inplace=True)
    return df

def update_lib_source(df: pd.DataFrame, match_table_path: str | Path) -> pd.DataFrame:
    """Inserting a lib_source column to DataFrame showing the source of the libraries
    
    Args: 
        df (DataFrame): Input DataFrame to insert new column.
        match_table_path (str | Path): Path to MatchTable.yaml to extract the solutions.
    
    Returns:
        pd.DataFrame: Updated DataFrame

    Raises:
        FileNotFoundError: If MatchTable.yaml files not found in the path.
    """
    try:
        with open(match_table_path) as f:
            match_table = yaml.load(f, Loader=SafeLoader)
            df["lib_source"] = [Path(match_table[int(idx)][0]).parts[-2] for idx in df['solutionIdx']]
    except FileNotFoundError:
            logger.warning("MatchTable.yaml file not found.")
            df["lib_source"] = "NotFound"

    return df
