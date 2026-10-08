# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turning collected benchmark CSVs into the Parquet dataset training consumes.

RFC 0019.13 §8.3 collects as CSV (resumable, shards merge by appending) and publishes as
Parquet; this is the only place §8.3's checks are enforced. Accepts any producer's corpus:
`kernel.*` and `device.*` are reserved roots, and any other namespaced column is a problem
column, whatever op token it is bound under.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import pandas as pd

from ..correctness import (
    DERIVED_LABELS,
    REASON,
    SUPPRESSED_TIMINGS,
    VERDICT,
    known_wrong,
)
from .metrics import derive_metrics
from .config_features import ABSENT, expand, slots_used_by

__all__ = [
    "ValidationError",
    "load_csvs",
    "build_dataset",
    "write_parquet",
    "expand_descriptors",
    "resolve_duplicates",
    "publish_frame",
]

#: Collection bookkeeping, dropped once shards are merged (§8.3). `is_valid`/`skip_reason`
#: are first translated into `error`, so a failure is published once, in one column.
COLLECTION_ONLY = ["shard_id", "is_valid", "skip_reason"]

#: What a producer may omit. A published sweep is complete unless it says otherwise.
DEFAULTS = {"problem_complete": True, "error": ""}

TIMING_COLUMNS = ["minTimeMs", "avgTimeMs", "stddevMs", "iters"]

#: Identity columns, pinned to text at the read so an all-digit device id or benchmark name
#: does not make the published dtype depend on which board was swept.
IDENTITY_DTYPES = {"benchmark": str, "device": str}


class ValidationError(Exception):
    """A collected corpus that §8.3 rejects; raised, never warned."""


def load_csvs(paths: Iterable[pathlib.Path]) -> pd.DataFrame:
    """Reads and concatenates collected CSVs; empty fields arrive as NaN ("no measurement")."""
    frames = [pd.read_csv(path, dtype=IDENTITY_DTYPES) for path in paths]
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


#: Roots with a defined meaning; any other namespaced column describes the problem.
_RESERVED_ROOTS = ("kernel.", "device.")


def _query_columns(frame: pd.DataFrame) -> list[str]:
    """The columns describing the problem, whatever root the producer bound them under.

    Identified by complement: the root is the op's own token. Dotless columns are envelope.
    """
    return [c for c in frame.columns if "." in c and not c.startswith(_RESERVED_ROOTS)]


def _short_name(column: str) -> str:
    """A problem column without its namespace: `attention_dense.seqlen_kv` -> `seqlen_kv`.

    Splits on the first dot only, so a nested token (`attention_dense.q.uid`) keeps its shape.
    """
    return column.split(".", 1)[1]


def _kernel_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("kernel.")]


def _problem_key_columns(frame: pd.DataFrame) -> list[str]:
    """The columns that identify a problem: the shape AND the machine that measured it.

    The same shape on two GPUs is two problems; keyed on shape alone, a multi-board corpus
    fails validation and has healthy problems downgraded. Device is `device`/`device_id`, else
    the `device.*` properties; `benchmark` joins the key wherever present.
    """
    identity = [c for c in ("benchmark", "device", "device_id") if c in frame.columns]
    if "device" not in identity and "device_id" not in identity:
        identity += sorted(c for c in frame.columns if c.startswith("device."))
    return _query_columns(frame) + identity


def _paired_identities(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Columns carrying two spellings of one identity: `X` and `X_id`."""
    return [
        (column, f"{column}_id")
        for column in frame.columns
        if f"{column}_id" in frame.columns
    ]


def _validate_identity_is_unambiguous(frame: pd.DataFrame) -> None:
    """Two spellings of one identity must agree one-for-one.

    Otherwise the corpus mixes engine versions, and one candidate wearing two names becomes
    two candidates, inflating every problem's catalog.
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


def _translate_collector_failure(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrites the collector's `is_valid=False` + `skip_reason` as §8.3's `error`.

    Done here, not in the collector, which must stay an appendable log (§8.8). Only an
    explicit false translates; a blank flag is not a claim of failure.
    """
    if "is_valid" not in frame.columns:
        return frame
    failed = frame["is_valid"].astype(str).str.strip().str.lower().isin({"false", "0"})
    # Never overwrite a recorded error; with no reason, the flag itself is the reason.
    reason = (
        frame["skip_reason"].fillna("").astype(str).str.strip()
        if "skip_reason" in frame.columns
        else pd.Series("", index=frame.index)
    )
    reason = reason.where(reason.str.len() > 0, "is_valid=False")
    blank = frame["error"].astype(str).str.strip().str.len() == 0
    frame.loc[failed & blank, "error"] = reason[failed & blank]
    return frame


def _translate_numerical_failure(frame: pd.DataFrame) -> pd.DataFrame:
    """A candidate checked wrong is published as the failure it is, verdict and reason kept.

    RFC 0019 §13.2: timings go null (so derived rates do too) and `error` gets the reason; the
    verdict columns stay. Only an explicit False translates; null is unknown.
    """
    wrong = known_wrong(frame)
    if not wrong.any():
        return frame
    for column in (*SUPPRESSED_TIMINGS, *DERIVED_LABELS):
        if column in frame.columns:
            frame[column] = frame[column].astype("float64").where(~wrong)
    # Never overwrite a recorded error; with no reason, the verdict itself is the reason.
    reason = (
        frame[REASON].fillna("").astype(str).str.strip()
        if REASON in frame.columns
        else pd.Series("", index=frame.index)
    )
    reason = reason.where(reason.str.len() > 0, f"{VERDICT}=False")
    blank = frame["error"].astype(str).str.strip().str.len() == 0
    frame.loc[wrong & blank, "error"] = reason[wrong & blank]
    return frame


def _validate(frame: pd.DataFrame) -> None:
    """§8.3's checks."""
    if not _query_columns(frame):
        raise ValidationError(
            "no problem columns; a corpus must identify its problem. Every value describing "
            "one is published under the token that bound it (`attention_dense.seqlen_kv`), so "
            "what is missing here is any namespaced column outside `kernel.*` and `device.*`"
        )
    for group in _RESERVED_ROOTS:
        if not any(c.startswith(group) for c in frame.columns):
            raise ValidationError(
                f"no {group}* columns; a corpus must identify its {group[:-1]}"
            )

    measured = frame["minTimeMs"].notna()
    has_error = frame["error"].astype(str).str.len() > 0

    # A row is a measurement or a failure, never both and never neither.
    both = measured & has_error
    if both.any():
        raise ValidationError(
            f"{int(both.sum())} rows carry both a measurement and an error"
        )
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

    key = _problem_key_columns(frame)
    kernels = _kernel_columns(frame)
    for _, rows in frame.groupby(key, dropna=False):
        if rows["problem_complete"].nunique() > 1:
            raise ValidationError(
                "problem_complete disagrees across rows of one problem"
            )

        # A complete problem must present each candidate once; a repeat means two
        # collections with different kernel sets were merged.
        if kernels and bool(rows["problem_complete"].iloc[0]):
            tuples = rows[kernels].apply(tuple, axis=1)
            if tuples.duplicated().any():
                raise ValidationError(
                    "a complete problem carries the same kernel configuration twice, so it spans "
                    "two collections with different candidate sets"
                )


def _mark_incomplete_where_errored(frame: pd.DataFrame) -> pd.DataFrame:
    """A candidate that could not be measured means the space was not fully measured.

    Downgrades the problem rather than failing the run, so its regret is not reported as exact.
    """
    key = _problem_key_columns(frame)
    errored = frame["error"].astype(str).str.len() > 0
    if not errored.any() or not key:
        return frame
    bad = frame.loc[errored, key].apply(tuple, axis=1)
    keys = frame[key].apply(tuple, axis=1)
    frame.loc[keys.isin(set(bad)), "problem_complete"] = False
    return frame


def expand_descriptors(
    frame: pd.DataFrame,
    columns: Iterable[str],
    scope_by: str | None = None,
) -> pd.DataFrame:
    """Replace opaque configuration strings with features a grouped model can select on.

    The source column is kept. `<column>.variant` stays a string: the training tool numbers it
    (RFC 0019 §6.5). With `scope_by`, each group gets its own `<column>.s<group>_f<n>` columns,
    since slot meaning differs per kernel.
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
            for group, positions in sorted(
                slots_used_by(rows, groups).items(), key=str
            ):
                for index in positions:
                    # Rows outside this group take ABSENT, same as an unfilled slot.
                    frame[f"{column}.s{group}_f{index}"] = [
                        row[index] if member == group else ABSENT
                        for row, member in zip(rows, groups)
                    ]
        frame[f"{column}.variant"] = shapes
    return frame


def resolve_duplicates(
    frame: pd.DataFrame, latest_column: str, best_column: str
) -> pd.DataFrame:
    """Keep, per problem, only the most recent occasion it was measured.

    Latest deduplicates; fastest breaks ties within one occasion. Per problem, not per file, so
    problems the newest occasion did not cover are kept.
    """
    # Full problem identity (shape, graph, device): one shape on two boards is two problems.
    query = _problem_key_columns(frame)
    if not query:
        return frame

    frame = frame.copy()
    problem = frame[query].astype(str).agg("|".join, axis=1)
    occasion = frame[latest_column]

    # The most recent occasion each problem was measured on, then only that occasion's rows.
    newest = occasion.groupby(problem).transform("max")
    frame = frame[occasion == newest]

    # A repeated candidate within one occasion is a repeated measurement; keep the best.
    kernels = _kernel_columns(frame)
    if kernels:
        keep = frame[query + kernels].astype(str).agg("|".join, axis=1)
        frame = frame.sort_values(best_column, kind="mergesort").loc[
            lambda f: ~keep.loc[f.index].duplicated()
        ]
    return frame.sort_index()


def build_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validates, derives the metrics, and drops what was only ever collection bookkeeping."""
    frame = _apply_defaults(frame.copy())
    frame = _translate_collector_failure(frame)
    frame = _translate_numerical_failure(frame)
    _validate(frame)
    frame = _mark_incomplete_where_errored(frame)

    # Derived from `avgTimeMs`, the statistic calibrated labels are defined on (RFC 0019 §13.4).
    query = _query_columns(frame)
    metrics = [
        derive_metrics(
            {_short_name(c): row[c] for c in query},
            None if pd.isna(row["avgTimeMs"]) else float(row["avgTimeMs"]),
        )
        for _, row in frame.iterrows()
    ]
    frame["tflops"] = [m["tflops"] for m in metrics]
    frame["gbs"] = [m["gbs"] for m in metrics]

    return frame.drop(columns=[c for c in COLLECTION_ONLY if c in frame.columns])


def publish_frame(
    csvs: Iterable[pathlib.Path],
    *,
    expand_descriptor: Iterable[str] = (),
    scope_by: str | None = None,
    resolve_duplicates_by: str | None = None,
    best_column: str = "avgTimeMs",
) -> pd.DataFrame:
    """Collected CSVs -> the validated dataset, as `python -m uhd_gen.dataset` publishes it.

    Resolve first: it settles duplicates validation would reject. Expand last, so the checks
    see the producer's columns rather than derived ones.
    """
    frame = load_csvs(csvs)
    if resolve_duplicates_by is not None:
        if resolve_duplicates_by not in frame.columns:
            raise ValidationError(
                f"--resolve-duplicates needs {resolve_duplicates_by!r} to order "
                "occasions by, and this corpus does not carry it"
            )
        frame = resolve_duplicates(frame, resolve_duplicates_by, best_column)
    dataset = build_dataset(frame)
    expand_descriptor = list(expand_descriptor)
    if expand_descriptor:
        dataset = expand_descriptors(dataset, expand_descriptor, scope_by)
    return dataset


def write_parquet(frame: pd.DataFrame, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def main(argv: list[str] | None = None) -> int:
    # Otherwise argparse prints `usage: __main__.py`.
    parser = argparse.ArgumentParser(
        prog="python -m uhd_gen.dataset", description=__doc__
    )
    parser.add_argument("--csv", nargs="+", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument(
        "--expand-descriptor",
        action="append",
        default=[],
        dest="expand_descriptor",
        metavar="COLUMN",
        help="expand a configuration string column (e.g. kernel.descriptor) into "
        "COLUMN.cfg0..N (numbers) and COLUMN.variant (the word shape, as a string -- "
        "RFC 0019 §6.5 has the training tool number it and ship the map). Repeatable.",
    )
    parser.add_argument(
        "--scope-by",
        default=None,
        dest="scope_by",
        metavar="COLUMN",
        help="give each value of COLUMN (e.g. kernel.solver_id) its own expanded columns, for "
        "an engine whose configuration schema varies by kernel. Without it one set of "
        "positions is shared, and a position then means different things in different "
        "groups.",
    )
    parser.add_argument(
        "--resolve-duplicates",
        default=None,
        dest="resolve_duplicates",
        metavar="COLUMN",
        help="keep, per problem, only the most recent occasion it was measured, ordered by "
        "COLUMN (e.g. date_run), breaking ties within it by --best-column. Without this "
        "a re-measured problem is rejected as two merged collections.",
    )
    parser.add_argument(
        "--best-column",
        default="avgTimeMs",
        dest="best_column",
        help="the column a repeat is resolved by, smallest kept (default: avgTimeMs, the "
        "statistic the published rates are derived from)",
    )
    args = parser.parse_args(argv)

    try:
        dataset = publish_frame(
            args.csv,
            expand_descriptor=args.expand_descriptor,
            scope_by=args.scope_by,
            resolve_duplicates_by=args.resolve_duplicates,
            best_column=args.best_column,
        )
    except ValidationError as error:
        print(f"uhd_gen.dataset: {error}", file=sys.stderr)
        return 1

    write_parquet(dataset, args.out)
    print(f"uhd_gen.dataset: wrote {len(dataset)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
