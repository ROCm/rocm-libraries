# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turning collected benchmark CSVs into the Parquet dataset training consumes.

RFC 0019.13 §8.3 collects as CSV and publishes as Parquet, because the two ends want opposite
things: a run lasting days must be resumable from a partial file and its shards must merge by
appending, neither of which Parquet does, while training wants a typed columnar dataset.

This is also the only place §8.3's checks are enforced. While training read the collected CSV
directly there was nothing between producer and consumer to apply them, so rules describing a
merge spanning inconsistent candidate sets described a check nothing performed.

Takes results from any producer, not only ours. The minimum a foreign CSV must carry is `q.*`,
`kernel.*`, `device.*` and a measurement; everything else has a default, and the metrics are
derived rather than demanded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import pandas as pd

from results_import.derive import derive_metrics

__all__ = ["ValidationError", "load_csvs", "build_dataset", "write_parquet"]

#: Collection bookkeeping, meaningless once the shards are merged (§8.3).
COLLECTION_ONLY = ["shard_id"]

#: What a producer may omit. A provider publishing tuning results is publishing a sweep, not
#: hand-tuned partials, so completeness defaults true rather than forcing every external corpus
#: to disclaim a caveat that does not apply to it.
DEFAULTS = {"problem_complete": True, "error": ""}

TIMING_COLUMNS = ["minTimeMs", "avgTimeMs", "stddevMs", "iters"]


class ValidationError(Exception):
    """A collected corpus that §8.3 rejects.

    Raised rather than warned. Every condition checked here makes the dataset silently wrong
    downstream -- a merge of two candidate sets trains a model on a catalog that never existed,
    and a row claiming both a measurement and an error is a producer bug whose rows cannot be
    trusted either way.
    """


def load_csvs(paths: Iterable[pathlib.Path]) -> pd.DataFrame:
    """Reads and concatenates collected CSVs.

    Appending is the whole reason collection stays CSV, so a merge is a concatenation here and
    nothing more. Empty fields arrive as NaN, which is how §8.3 spells "no measurement".
    """
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValidationError("no input CSVs")
    return pd.concat(frames, ignore_index=True)


def _apply_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    for column, default in DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)
    return frame


def _query_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("q.")]


def _kernel_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("kernel.")]


def _validate(frame: pd.DataFrame) -> None:
    """§8.3's checks, applied where they can finally be applied."""
    for group in ("q.", "kernel.", "device."):
        if not any(c.startswith(group) for c in frame.columns):
            raise ValidationError(f"no {group}* columns; a corpus must identify its {group[:-1]}")

    measured = frame["minTimeMs"].notna()
    has_error = frame["error"].astype(str).str.len() > 0

    # A row is a measurement or a failure, never both and never neither. The error message is
    # the whole flag -- there is no validity column that could disagree with it.
    both = measured & has_error
    if both.any():
        raise ValidationError(f"{int(both.sum())} rows carry both a measurement and an error")
    neither = ~measured & ~has_error
    if neither.any():
        raise ValidationError(
            f"{int(neither.sum())} rows carry neither a measurement nor an error; a row that "
            "was never attempted does not belong in the results"
        )

    if (frame.loc[measured, "minTimeMs"] > frame.loc[measured, "avgTimeMs"]).any():
        raise ValidationError("minTimeMs exceeds avgTimeMs on a measured row")
    if "stddevMs" in frame.columns and (frame.loc[measured, "stddevMs"] < 0).any():
        raise ValidationError("negative stddevMs")

    query = _query_columns(frame)
    kernels = _kernel_columns(frame)
    for _, rows in frame.groupby(query, dropna=False):
        if rows["problem_complete"].nunique() > 1:
            raise ValidationError("problem_complete disagrees across rows of one problem")

        # A problem whose candidate space was fully measured must present the same candidates
        # wherever it came from. Two collections taken against different kernel sets merge into
        # a catalog that never existed, and argmax seeks precisely the configurations that were
        # added after training.
        if kernels and bool(rows["problem_complete"].iloc[0]):
            tuples = rows[kernels].apply(tuple, axis=1)
            if tuples.duplicated().any():
                raise ValidationError(
                    "a complete problem carries the same kernel configuration twice, so it spans "
                    "two collections with different candidate sets"
                )


def _mark_incomplete_where_errored(frame: pd.DataFrame) -> pd.DataFrame:
    """A candidate that could not be measured means the space was not fully measured.

    Downgrades that problem rather than failing the run: a hardware fault on one pair should not
    discard a multi-hour sweep, but the problem must not present as exact either, or regret over
    it silently becomes a lower bound.
    """
    query = _query_columns(frame)
    errored = frame["error"].astype(str).str.len() > 0
    if not errored.any() or not query:
        return frame
    bad = frame.loc[errored, query].apply(tuple, axis=1)
    keys = frame[query].apply(tuple, axis=1)
    frame.loc[keys.isin(set(bad)), "problem_complete"] = False
    return frame


def build_dataset(frame: pd.DataFrame, opmeta: dict) -> pd.DataFrame:
    """Validates, derives the metrics, and drops what was only ever collection bookkeeping."""
    frame = _apply_defaults(frame.copy())
    _validate(frame)
    frame = _mark_incomplete_where_errored(frame)

    query = _query_columns(frame)
    metrics = [
        derive_metrics(
            {c[2:]: row[c] for c in query},
            None if pd.isna(row["minTimeMs"]) else float(row["minTimeMs"]),
            opmeta,
        )
        for _, row in frame.iterrows()
    ]
    frame["tflops"] = [m["tflops"] for m in metrics]
    frame["gbs"] = [m["gbs"] for m in metrics]

    return frame.drop(columns=[c for c in COLLECTION_ONLY if c in frame.columns])


def write_parquet(frame: pd.DataFrame, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True, type=pathlib.Path)
    parser.add_argument("--opmeta", required=True, type=pathlib.Path,
                        help="the operation's .opmeta.json, whose flops/elements are evaluated")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    with args.opmeta.open() as handle:
        opmeta = json.load(handle)

    try:
        dataset = build_dataset(load_csvs(args.csv), opmeta)
    except ValidationError as error:
        print(f"results_import: {error}", file=sys.stderr)
        return 1

    write_parquet(dataset, args.out)
    print(f"results_import: wrote {len(dataset)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
