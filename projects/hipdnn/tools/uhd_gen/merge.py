# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Join sweeps from several machines of one architecture into one corpus.

A UHD is keyed by arch (RFC 0019 §3.1). `device` is the device's kind, its DeviceKey hash
(RFC 0019.13 §4.1), so two machines with the same card measure the same problems: a
configuration measured on both is one measurement, repeated, and the merge keeps the fastest
usable one (`training_table.best_per_shape`'s rule). Different kinds of card stay apart.
Refuses mismatched column sets (a missing column would be NaN exactly on one machine's rows).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .corpus_io import read_corpus_frame
from .dataset.training_table import (
    BEST_STATISTIC,
    measurement_error,
    measurement_key,
)

__all__ = [
    "MergeError",
    "merge_corpora",
    "add_merge_arguments",
    "run_merge",
]

logger = logging.getLogger(__name__)

#: The DeviceKey hash: one value for every GPU of one kind, distinct between kinds.
DEVICE_COLUMN = "device"

BENCHMARK_COLUMN = "benchmark"


class MergeError(Exception):
    """A refusal raised while merging; nothing is written."""


def _load(path: Path) -> pd.DataFrame:
    # Same reader as `train`, so any input it accepts merges too.
    try:
        frame = read_corpus_frame(path)
    except (OSError, ValueError, ImportError) as error:
        raise MergeError(f"cannot read corpus {path}: {error}") from error
    if frame.empty:
        raise MergeError(f"{path} has no rows; an empty corpus contributes nothing")
    return frame


def merge_corpora(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Concatenate corpora with identical columns; return the frame and a per-file report."""
    if len(paths) < 2:
        raise MergeError("merging needs at least two corpora")

    frames: list[pd.DataFrame] = []
    per_file: list[dict] = []
    reference: set[str] | None = None
    reference_path: Path | None = None

    for path in paths:
        frame = _load(path)
        columns = set(frame.columns)
        if reference is None:
            reference, reference_path = columns, path
        elif columns != reference:
            missing = sorted(reference - columns)
            extra = sorted(columns - reference)
            raise MergeError(
                f"{path} does not describe the same feature space as {reference_path}: "
                + (f"missing {missing}; " if missing else "")
                + (f"unexpected {extra}; " if extra else "")
                + "merging them would leave a column NaN exactly where one machine's "
                "rows are, which trains as a fact about that machine. Re-collect the "
                "odd one out with the same build."
            )

        if DEVICE_COLUMN not in frame.columns:
            raise MergeError(
                f"{path} has no {DEVICE_COLUMN!r} column, so its rows cannot be told "
                "apart from another machine's; a problem is (benchmark, device) and "
                "without the second half both boards collapse into one oracle"
            )

        devices = sorted(map(str, frame[DEVICE_COLUMN].dropna().unique()))
        problems = (
            frame.groupby([BENCHMARK_COLUMN, DEVICE_COLUMN], dropna=False).ngroups
            if BENCHMARK_COLUMN in frame.columns
            else 0
        )
        per_file.append(
            {
                "path": str(path),
                "rows": int(len(frame)),
                "devices": devices,
                "problems": problems,
            }
        )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)

    # Machines with the same kind of card repeat each other's measurements; one row each.
    seen: dict[str, list[str]] = {}
    for entry in per_file:
        for device in entry["devices"]:
            seen.setdefault(device, []).append(entry["path"])
    repeated = {device: files for device, files in seen.items() if len(files) > 1}
    merged, superseded = _keep_fastest_repeat(merged)
    if superseded:
        logger.info(
            "%d repeated measurement(s) of one configuration on one kind of device were "
            "set aside for the fastest",
            superseded,
        )

    report = {
        "corpora": per_file,
        "rows": int(len(merged)),
        "devices": sorted(map(str, merged[DEVICE_COLUMN].dropna().unique())),
        "problems": (
            merged.groupby([BENCHMARK_COLUMN, DEVICE_COLUMN], dropna=False).ngroups
            if BENCHMARK_COLUMN in merged.columns
            else 0
        ),
        "repeated_devices": {
            device: files for device, files in sorted(repeated.items())
        },
        "superseded_rows": superseded,
    }
    return merged, report


def _keep_fastest_repeat(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """`frame` with each repeated usable measurement reduced to its fastest row, and how
    many rows were set aside. Rows that cannot be a label are kept as they are: whatever
    reads the corpus reports them."""
    if not {BENCHMARK_COLUMN, "arch"} <= set(frame.columns):
        return frame, 0
    best: dict = {}
    dropped: set = set()
    for index, row in zip(frame.index, frame.to_dict(orient="records")):
        if measurement_error(row) is not None:
            continue
        key = measurement_key(row)
        time = float(row[BEST_STATISTIC])
        held = best.get(key)
        if held is None:
            best[key] = (time, index)
        elif time < held[0]:
            dropped.add(held[1])
            best[key] = (time, index)
        else:
            dropped.add(index)
    return frame.drop(index=sorted(dropped)).reset_index(drop=True), len(dropped)


def add_merge_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the `merge` flags."""
    parser.add_argument(
        "inputs",
        nargs="+",
        help="benchmark corpora to join, one per machine",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="where to write the merged corpus",
    )


def run_merge(args: argparse.Namespace) -> int:
    paths = [Path(value) for value in args.inputs]
    try:
        merged, report = merge_corpora(paths)
    except MergeError as error:
        logger.error("%s", error)
        return 1

    # Format follows the output suffix; CSV would drop the dataset's column types.
    if Path(args.output).suffix == ".parquet":
        merged.to_parquet(args.output, index=False)
    else:
        merged.to_csv(args.output, index=False, lineterminator="\n")

    print(f"\nMerged {len(report['corpora'])} corpora -> {args.output}")
    print(f"  {'rows':>10}  {'problems':>9}  {'devices':>7}  corpus")
    for entry in report["corpora"]:
        print(
            f"  {entry['rows']:>10,}  {entry['problems']:>9,}  "
            f"{len(entry['devices']):>7}  {entry['path']}"
        )
    print(
        f"  {report['rows']:>10,}  {report['problems']:>9,}  "
        f"{len(report['devices']):>7}  TOTAL"
    )
    print(f"\n  device identities: {', '.join(report['devices'])}")
    if len(report["devices"]) < 2:
        print("  !! every row carries one device identity -- this is one board's data,")
        print("  !! and a model fitted on it knows nothing about the others")
    return 0
