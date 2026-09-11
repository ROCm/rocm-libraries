# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Join sweeps from several machines of one architecture into one corpus.

A UHD is keyed by arch, not by board ([RFC 0019 §3.1]), so one gfx942 model serves
MI300X, MI325X, MI308X and MI300A. Training it on a corpus from a single board teaches
it that board; the others get a model fitted to hardware they are not.

The runtime already keeps the boards apart. A problem is `(benchmark, device)`, and
`device` is the hex DeviceKey hash -- a fold over arch, warp size, compute units and the
memory facts -- so two boards land under two identities and two identical boards land
under one, which is correct in both directions. Concatenating the CSVs is therefore
almost the whole job.

Almost, because two things go wrong silently and this refuses both:

* **Schema drift.** A corpus collected before a feature column existed merges into a
  frame where that column is NaN for those rows. Training then fits a column that is
  absent exactly where one board's data is, which looks like a signal about that board.
* **The same board twice.** Two runs of one machine share a device identity, so their
  rows collapse into the same problems and that board silently carries double weight in
  every regret figure. Sometimes intended -- more samples of a noisy card -- so this
  warns rather than refuses, and says which identity it saw twice.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

__all__ = [
    "MergeError",
    "merge_corpora",
    "add_merge_arguments",
    "run_merge",
]

logger = logging.getLogger(__name__)

#: The column carrying device identity. Written by the runtime as the hex DeviceKey
#: hash, so it is stable for one board and distinct between two.
DEVICE_COLUMN = "device"

#: The column carrying problem identity within a device.
BENCHMARK_COLUMN = "benchmark"


class MergeError(Exception):
    """A refusal raised while merging; nothing is written."""


def _load(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as error:
        raise MergeError(f"cannot read corpus {path}: {error}") from error
    if frame.empty:
        raise MergeError(f"{path} has no rows; an empty corpus contributes nothing")
    return frame


def merge_corpora(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Concatenate corpora that describe the same feature space.

    Returns the merged frame and a report naming what came from where, so a training
    run can record which machines its model was fitted on.
    """
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
            {"path": str(path), "rows": int(len(frame)), "devices": devices, "problems": problems}
        )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)

    # One identity in two files is one board swept twice. Legitimate -- a noisy card is
    # worth resampling -- but it doubles that board's weight in every figure derived
    # from the corpus, so it is never allowed to pass unsaid.
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
        "repeated_devices": {device: files for device, files in sorted(repeated.items())},
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

    merged.to_csv(args.output, index=False)

    print(f"\nMerged {len(report['corpora'])} corpora -> {args.output}")
    print(f"  {'rows':>10}  {'problems':>9}  {'devices':>7}  corpus")
    for entry in report["corpora"]:
        print(
            f"  {entry['rows']:>10,}  {entry['problems']:>9,}  "
            f"{len(entry['devices']):>7}  {entry['path']}"
        )
    print(f"  {report['rows']:>10,}  {report['problems']:>9,}  "
          f"{len(report['devices']):>7}  TOTAL")
    # The identities themselves, because a merged corpus that turned out to hold one
    # board is the failure this command exists to make visible.
    print(f"\n  device identities: {', '.join(report['devices'])}")
    if len(report["devices"]) < 2:
        print("  !! every row carries one device identity -- this is one board's data,")
        print("  !! and a model fitted on it knows nothing about the others")
    return 0
