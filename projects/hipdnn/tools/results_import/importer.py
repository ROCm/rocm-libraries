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
from results_import.descriptor import ABSENT, expand, slots_used_by

__all__ = [
    "ValidationError",
    "load_csvs",
    "build_dataset",
    "write_parquet",
    "expand_descriptors",
    "resolve_duplicates",
]

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


def _paired_identities(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Columns carrying two spellings of one identity: `X` and `X_id`.

    A convention rather than a fixed list, so this reads any producer's corpus. Where only one
    of a pair is present there is nothing to disagree, and the check does not apply.
    """
    return [(column, f"{column}_id") for column in frame.columns
            if f"{column}_id" in frame.columns]


def _validate_identity_is_unambiguous(frame: pd.DataFrame) -> None:
    """Two spellings of one identity must agree one-for-one.

    A name bound to two ids -- or an id to two names -- means the corpus merges collections
    taken against different versions of the engine, where a candidate was renamed or an id
    reused. Nothing else in the file records which version a row came from.

    Caught rather than tolerated because of what it does downstream silently: the candidate
    identity spans every `kernel.*` column, so one candidate wearing two names becomes two.
    Every problem's candidate count inflates and regret is computed over a catalog that
    existed on no single machine.
    """
    for name, identifier in _paired_identities(frame):
        _validate_pair_agrees(frame, name, identifier)


def _validate_pair_agrees(frame: pd.DataFrame, name: str, identifier: str) -> None:
    pairs = frame[[name, identifier]].dropna().drop_duplicates()
    for left, right in ((name, identifier), (identifier, name)):
        bindings = pairs.groupby(left)[right].nunique()
        ambiguous = bindings[bindings > 1]
        if not ambiguous.empty:
            first = ambiguous.index[0]
            bound = sorted(pairs.loc[pairs[left] == first, right].tolist())
            raise ValidationError(
                f"{len(ambiguous)} {left} value(s) are ambiguous: {first!r} is bound to "
                f"{len(bound)} different {right} values {bound}. The corpus spans engine "
                "versions that disagree on this candidate."
            )


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

    _validate_identity_is_unambiguous(frame)

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


def expand_descriptors(
    frame: pd.DataFrame,
    columns: Iterable[str],
    scope_by: str | None = None,
) -> pd.DataFrame:
    """Replace opaque configuration strings with features a grouped model can select on.

    The source column is kept: it is the human-readable identity of a configuration, and every
    report that names a winner wants it.

    `<column>.variant` is emitted as the descriptor's *word shape*, a string. RFC 0019 §6.5
    gives numbering to the training tool, which observes the values, ships the map in the UHD's
    `categorical_encoding`, and has it covered by `features_hash` -- so a code cannot change
    underneath a trained model without the contract check seeing it.

    With `scope_by`, each group gets its own columns (`<column>.s<group>_f<n>`) instead of one
    shared set of positions. A configuration schema that varies with the kernel makes a shared
    position meaningless -- slot 3 a tile width for one group and a stage count for another --
    and it is the *first* layer of a grouped model that pays, because it is the one that sees
    every row. Which positions a group uses is observed from the corpus; nothing here consults
    the library that produced the descriptors.
    """
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            raise ValidationError(
                f"--expand-descriptor names {column!r}, which this corpus does not carry "
                f"(it has {', '.join(_kernel_columns(frame)) or 'no kernel.* columns'})"
            )
        rows, shapes, slots = expand(frame[column].tolist())

        if scope_by is None:
            for index in range(slots):
                frame[f"{column}.cfg{index}"] = [row[index] for row in rows]
        else:
            if scope_by not in frame.columns:
                raise ValidationError(
                    f"--scope-by names {scope_by!r}, which this corpus does not carry"
                )
            groups = frame[scope_by].tolist()
            for group, positions in sorted(slots_used_by(rows, groups).items(), key=str):
                for index in positions:
                    # A row outside this group takes the absent value, which is what a kernel
                    # with no such field means -- the same state an unfilled slot already has.
                    frame[f"{column}.s{group}_f{index}"] = [
                        row[index] if member == group else ABSENT
                        for row, member in zip(rows, groups)
                    ]
        frame[f"{column}.variant"] = shapes
    return frame


def resolve_duplicates(frame: pd.DataFrame, latest_column: str, best_column: str) -> pd.DataFrame:
    """Keep, per problem, only the most recent occasion it was measured.

    `_validate` rejects a complete problem carrying one configuration twice, because that
    usually means two collections with different candidate sets were merged. A problem that was
    simply re-measured trips the same check, and the rule for it is: latest deduplicates,
    fastest breaks ties within one occasion.

    Resolved per problem rather than globally. Taking the newest occasion in the *file* would
    delete every problem that occasion did not cover -- typically the ones an older, broader
    sweep measured -- which is a silent loss of exactly the problems with the widest candidate
    coverage. Per problem, one occasion also means a problem's candidates were all measured
    against each other, which is what makes their times comparable at all.
    """
    query = _query_columns(frame)
    if not query:
        return frame

    frame = frame.copy()
    problem = frame[query].astype(str).agg("|".join, axis=1)
    occasion = frame[latest_column]

    # The most recent occasion each problem was measured on, then only that occasion's rows.
    newest = occasion.groupby(problem).transform("max")
    frame = frame[occasion == newest]

    # Within it, a repeated candidate is a repeated measurement: repeats differ by contention
    # and clocks, not by anything about the kernel, so the best stands for the candidate.
    kernels = _kernel_columns(frame)
    if kernels:
        keep = frame[query + kernels].astype(str).agg("|".join, axis=1)
        frame = (frame.sort_values(best_column, kind="mergesort")
                      .loc[lambda f: ~keep.loc[f.index].duplicated()])
    return frame.sort_index()


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
    parser.add_argument(
        "--expand-descriptor", action="append", default=[], dest="expand_descriptor",
        metavar="COLUMN",
        help="expand a configuration string column (e.g. kernel.descriptor) into "
             "COLUMN.cfg0..N (numbers) and COLUMN.variant (the word shape, as a string -- "
             "RFC 0019 §6.5 has the training tool number it and ship the map). Repeatable.",
    )
    parser.add_argument(
        "--scope-by", default=None, dest="scope_by", metavar="COLUMN",
        help="give each value of COLUMN (e.g. kernel.solver_id) its own expanded columns, for "
             "an engine whose configuration schema varies by kernel. Without it one set of "
             "positions is shared, and a position then means different things in different "
             "groups.",
    )
    parser.add_argument(
        "--resolve-duplicates", default=None, dest="resolve_duplicates", metavar="COLUMN",
        help="keep, per problem, only the most recent occasion it was measured, ordered by "
             "COLUMN (e.g. date_run), breaking ties within it by --best-column. Without this "
             "a re-measured problem is rejected as two merged collections.",
    )
    parser.add_argument(
        "--best-column", default="minTimeMs", dest="best_column",
        help="the column a repeat is resolved by, smallest kept (default: minTimeMs)",
    )
    args = parser.parse_args(argv)

    with args.opmeta.open() as handle:
        opmeta = json.load(handle)

    # The order is the point. Resolution happens first because it settles the very duplicates
    # validation would reject; expansion happens last so those checks see the producer's own
    # columns rather than this tool's derived ones.
    try:
        frame = load_csvs(args.csv)
        if args.resolve_duplicates is not None:
            if args.resolve_duplicates not in frame.columns:
                raise ValidationError(
                    f"--resolve-duplicates needs {args.resolve_duplicates!r} to order "
                    "occasions by, and this corpus does not carry it"
                )
            frame = resolve_duplicates(frame, args.resolve_duplicates, args.best_column)
        dataset = build_dataset(frame, opmeta)
        if args.expand_descriptor:
            dataset = expand_descriptors(dataset, args.expand_descriptor, args.scope_by)
    except ValidationError as error:
        print(f"results_import: {error}", file=sys.stderr)
        return 1

    write_parquet(dataset, args.out)
    print(f"results_import: wrote {len(dataset)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
