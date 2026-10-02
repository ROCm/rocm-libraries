# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Join sweeps from several boards of one architecture into one corpus.

A UHD is keyed by arch, not board (RFC 0019 §3.1), and a problem is (benchmark, device),
so concatenation keeps boards apart. Refuses mismatched column sets (a missing column
would be NaN exactly on one board's rows) and warns when one device appears twice.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .corpus_io import read_corpus_frame

__all__ = [
    "MergeError",
    "merge_corpora",
    "add_merge_arguments",
    "run_merge",
]

logger = logging.getLogger(__name__)

#: Hex DeviceKey hash: stable for one board, distinct between two.
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

    # One device in two files doubles that board's weight; allowed, but always warned.
    seen: dict[str, list[str]] = {}
    for entry in per_file:
        for device in entry["devices"]:
            seen.setdefault(device, []).append(entry["path"])
    repeated = {device: files for device, files in seen.items() if len(files) > 1}
    for device, files in sorted(repeated.items()):
        logger.warning(
            "device %s appears in %d corpora (%s). Those rows share problem identity, so "
            "that board carries proportionally more weight than the others. Intended when "
            "resampling one machine; a mistake if you meant to merge two.",
            device,
            len(files),
            ", ".join(files),
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
    }
    return merged, report


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
        merged.to_csv(args.output, index=False)

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
