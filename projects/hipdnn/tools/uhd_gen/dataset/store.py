# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The stored form of training data: one Parquet dataset per engine, arch, revision and role.

RFC 0019.13 collects as CSV and trains from Parquet. Collection keeps its raw record -- a run
lasting hours must be resumable from a partial file, and its shards merge by appending, which
Parquet does not do -- and conversion turns that record, or any other source's, into the one
form training reads. This module is the conversion and the store; it decides nothing about
training.

A dataset is a directory:

* `dataset.parquet` -- every row ever converted in, each tagged with the contribution it came
  from (`contribution_id`), when it was measured (`collected_at`), the metric source it answers
  (`source`), what it is for (`purpose`: training, holdout or eval) and the corpus it measured.
* `dataset_manifest.json` -- what the rows cannot say: one entry per contribution with
  everything a collection's manifest recorded (devices, graphs, commands, knob addressing,
  published features, failed graphs), and the binding every contribution must share.

Rows are only ever appended. Which measurement of a shape is the label is decided when a run
reads the dataset (`uhd_gen.generate.load_dataset`), as it was when collections were read
directly: the file keeps every measurement, as the collections did.

Needs pandas and a Parquet engine, nothing else -- the same budget as `publish`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Iterable, Iterator

import pandas as pd

from .. import addressing
from .publish import ValidationError, publish_frame

DATASET_SCHEMA = "uhd_gen.dataset/1"
DATA_FILE = "dataset.parquet"
MANIFEST_FILE = "dataset_manifest.json"
COLLECTION_SCHEMA = "uhd_gen.collection/1"
COLLECTION_MANIFEST = "collection_manifest.json"

#: Columns this store adds to every row. A converted row carrying one of them already is refused.
BOOKKEEPING = ("contribution_id", "collected_at", "source", "purpose", "corpus_id")

#: What every contribution of one dataset must share: a model is trained against one binding.
SHARED = (
    "role",
    "engine_name",
    "engine_id",
    "selector_revision",
    "trained_against",
    "collection_knobs",
)

PURPOSES = ("training", "holdout", "eval")

#: The role an engine-immediate (L1) collection is taken for.
IMMEDIATE_ROLE = "predict_engine"


class DatasetError(ValueError):
    """A contribution that cannot join this dataset, or a dataset that cannot be read."""


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


def read_manifest(directory: pathlib.Path) -> dict[str, Any] | None:
    path = pathlib.Path(directory) / MANIFEST_FILE
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise DatasetError(f"{directory} is not a {DATASET_SCHEMA} dataset")
    return manifest


def _native(value: Any) -> Any:
    """A Parquet value as the JSON record it was converted from: a missing value is None."""
    if value is None or isinstance(value, (str, bytes, bool, list, dict)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):  # an array-like cell: a value, not a missing one
        return value


def read_rows(directory: pathlib.Path) -> list[dict[str, Any]]:
    """Every row of the dataset as records, missing values None -- the records a collection's
    JSON corpus held, so a reader cannot tell which store they came from."""
    path = pathlib.Path(directory) / DATA_FILE
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    return [
        {key: _native(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


# ---------------------------------------------------------------------------------------------
# Writing: one append path, under a lock
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _locked(directory: pathlib.Path, timeout_s: float = 600.0) -> Iterator[None]:
    """`mkdir` is atomic on NFS where `flock` is not; the directory is the lock."""
    lock = directory / ".lock"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise DatasetError(
                    f"{lock} held for over {timeout_s:.0f} s; another conversion is writing "
                    "this dataset, or one died holding the lock (remove it if so)"
                ) from None
            time.sleep(1.0)
    try:
        yield
    finally:
        lock.rmdir()


def _write_atomically(path: pathlib.Path, write) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(temporary)
    os.replace(temporary, path)


def _agree(manifest: dict[str, Any], entry: dict[str, Any]) -> None:
    for key in SHARED:
        if manifest.get(key) != entry.get(key):
            raise DatasetError(
                f"contribution {entry['contribution_id']} disagrees with this dataset on "
                f"{key}: {manifest.get(key)!r} vs {entry.get(key)!r}; a dataset holds one "
                "binding, as a model is trained against one"
            )


def _tag(
    entry: dict[str, Any], rows_by_source: dict[Any, list[dict[str, Any]]]
) -> tuple[dict, list]:
    """The contribution's rows with the store's bookkeeping, and its entry with its columns."""
    identity = entry["contribution_id"]
    if entry.get("purpose", "training") not in PURPOSES:
        raise DatasetError(f"purpose {entry.get('purpose')!r} is not one of {PURPOSES}")
    tagged = []
    for source, rows in rows_by_source.items():
        for row in rows:
            clash = [key for key in BOOKKEEPING if key in row]
            if clash:
                raise DatasetError(f"a converted row already carries {clash}")
            tagged.append(
                {
                    **row,
                    "contribution_id": identity,
                    "collected_at": entry["collected_at"],
                    "source": None if source is None else str(source),
                    "purpose": entry.get("purpose", "training"),
                    "corpus_id": entry.get("corpus_id"),
                }
            )
    if not tagged:
        raise DatasetError(f"contribution {identity} holds no rows")
    # The keys this contribution's rows carried, per source. The table holds every
    # contribution's columns, so without this a row read back would gain keys its collection
    # never wrote -- and a reader that iterates a record's keys would see them.
    entry = {
        **entry,
        "columns": {
            str(source): sorted({key for row in rows for key in row})
            for source, rows in rows_by_source.items()
        },
    }
    return entry, tagged


def add_many(
    directory: pathlib.Path,
    contributions: list[tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]],
) -> list[dict[str, Any]]:
    """Append contributions in one write: their rows, tagged, and their manifest entries.

    Returns one {"added": bool, "rows": n, "contribution_id": id} per contribution. One already
    in the dataset is not added again -- converting the same collection twice is a no-op, not a
    duplicate measurement. One whose binding differs from the dataset's, or whose knob
    addressing contradicts it, is refused, and then nothing of this call is written.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = [_tag(entry, rows) for entry, rows in contributions]
    results = []
    with _locked(directory):
        manifest = read_manifest(directory)
        if manifest is None:
            first = prepared[0][0]
            manifest = {
                "schema": DATASET_SCHEMA,
                **{key: first.get(key) for key in SHARED},
                "contributions": [],
            }
        present = {c["contribution_id"] for c in manifest["contributions"]}
        fresh, frames = [], []
        for entry, tagged in prepared:
            identity = entry["contribution_id"]
            if identity in present:
                results.append({"added": False, "rows": 0, "contribution_id": identity})
                continue
            _agree(manifest, entry)
            present.add(identity)
            fresh.append(entry)
            frames.append(pd.DataFrame(tagged))
            results.append(
                {"added": True, "rows": len(tagged), "contribution_id": identity}
            )
        if not fresh:
            return results
        try:
            addressing.merge_manifests(
                [c.get("knob_encodings") for c in manifest["contributions"] + fresh]
            )
        except ValueError as error:
            raise DatasetError(
                f"knob addressing contradicts the dataset: {error}"
            ) from None
        data = directory / DATA_FILE
        if data.is_file():
            frames.insert(0, pd.read_parquet(data))
        frame = pd.concat(frames, ignore_index=True)
        try:
            _write_atomically(data, lambda path: frame.to_parquet(path, index=False))
        except (
            Exception
        ) as error:  # pyarrow refuses a column whose types the rows disagree on
            raise DatasetError(
                f"rows cannot be stored as one Parquet table: {error}"
            ) from None
        manifest["contributions"].extend(fresh)
        _write_atomically(
            directory / MANIFEST_FILE,
            lambda path: path.write_text(
                json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
            ),
        )
    return results


def add(
    directory: pathlib.Path,
    entry: dict[str, Any],
    rows_by_source: dict[Any, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Append one contribution (`add_many` of one)."""
    return add_many(directory, [(entry, rows_by_source)])[0]


# ---------------------------------------------------------------------------------------------
# Converters: each source's record -> (manifest entry, rows by source). Nothing else.
# ---------------------------------------------------------------------------------------------


def _check_immediate(rows_by_source: dict[Any, list[dict[str, Any]]]) -> None:
    """An L1 contribution's rows pass the checks training applies to them (§13.2, the binding)."""
    from ..immediate import normalize_corpus

    for rows in rows_by_source.values():
        normalize_corpus(pd.DataFrame(rows))


def from_collection(
    collection: pathlib.Path,
    *,
    purpose: str = "training",
    corpus_id: str | None = None,
    contribution_id: str | None = None,
) -> tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """A raw collection (`generate --collect-only`) as a contribution."""
    collection = pathlib.Path(collection)
    manifest = json.loads(
        (collection / COLLECTION_MANIFEST).read_text(encoding="utf-8")
    )
    if manifest.get("schema") != COLLECTION_SCHEMA:
        raise DatasetError(f"{collection} is not a {COLLECTION_SCHEMA} collection")
    sources = manifest["sources"]
    rows_by_source = {}
    for source in sources:
        name = "corpus" if len(sources) == 1 else f"corpus_{source}"
        rows_by_source[source] = json.loads(
            (collection / f"{name}.json").read_text(encoding="utf-8")
        )
    if manifest["role"] == IMMEDIATE_ROLE:
        _check_immediate(rows_by_source)
    entry = {
        **{key: manifest.get(key) for key in manifest if key != "schema"},
        "contribution_id": contribution_id or collection.name,
        "origin": {"kind": "collection", "path": str(collection.resolve())},
        "purpose": purpose,
        "corpus_id": corpus_id,
    }
    return entry, rows_by_source


def from_generation(
    generation: pathlib.Path,
    *,
    purpose: str = "training",
    corpus_id: str | None = None,
    contribution_id: str | None = None,
) -> tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """An old one-shot `generate` run's measurements, which lived only in its model's folder.

    Every manifest field is taken from what that run recorded: its rows (engine, arch, devices,
    binding), its generation manifest (engine id, provenance, graphs, commands), and the time of
    its first bench call, the closest record of when it measured.
    """
    from datetime import datetime, timezone

    generation = pathlib.Path(generation)
    recorded = json.loads(
        (generation / "generation_manifest.json").read_text(encoding="utf-8")
    )
    if recorded.get("promotion_role") != IMMEDIATE_ROLE:
        raise DatasetError(
            f"{generation} generated {recorded.get('promotion_role')!r}; only engine-immediate "
            f"({IMMEDIATE_ROLE}) runs are converted"
        )
    models = recorded.get("models") or []
    metrics = [m.get("requested_metric") or m.get("metric") for m in models] or [
        "tflops"
    ]
    rows_by_source = {}
    for metric in metrics:
        name = "corpus.json" if len(metrics) == 1 else f"corpus_{metric}.json"
        rows_by_source[metric] = json.loads(
            (generation / name).read_text(encoding="utf-8")
        )
    every = [row for rows in rows_by_source.values() for row in rows]
    revisions = {json.loads(row["binding"])["selector_revision"] for row in every}
    if len(revisions) != 1:
        raise DatasetError(f"{generation} measured under revisions {sorted(revisions)}")
    _check_immediate(rows_by_source)
    published: set[str] = set()
    for row in every:
        published.update(json.loads(row["features"]))
    logs = sorted((generation / "commands").glob("*.stdout.json"))
    first = min(
        (p.stat().st_mtime for p in logs),
        default=(generation / "corpus.json").stat().st_mtime,
    )
    entry = {
        "collected_at": datetime.fromtimestamp(first, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "role": IMMEDIATE_ROLE,
        "engine": every[0].get("engine_name"),
        "engine_id": recorded.get("engine_id"),
        "engine_name": every[0].get("engine_name"),
        "sources": metrics,
        "trained_against": recorded.get("trained_against"),
        "selector_revision": revisions.pop(),
        "arches": sorted({row["arch"] for row in every}),
        "devices": sorted({str(row["device"]) for row in every}),
        "graph_count": len({row["benchmark"] for row in every}),
        "row_counts": {str(m): len(rows_by_source[m]) for m in metrics},
        "published": sorted(published),
        "kernel_fields": [],
        "knob_encodings": recorded.get("knob_encodings", {}),
        "shipping_knobs": recorded.get("shipping_knobs", []),
        "collection_knobs": recorded.get("collection_knobs", []),
        "graphs": recorded.get("graphs", []),
        "commands": recorded.get("commands", []),
        "failed_graphs": recorded.get("failed_graphs", []),
        "contribution_id": contribution_id or f"generation-{generation.parent.name}",
        "origin": {"kind": "generation", "path": str(generation.resolve())},
        "purpose": purpose,
        "corpus_id": corpus_id,
    }
    return entry, rows_by_source


def from_csv(
    csvs: Iterable[pathlib.Path],
    *,
    engine: str,
    engine_id: int,
    role: str,
    arch: str,
    selector_revision: str,
    collected_at: str,
    contribution_id: str,
    purpose: str = "training",
    corpus_id: str | None = None,
    expand_descriptor: Iterable[str] = (),
    scope_by: str | None = None,
    resolve_duplicates_by: str | None = None,
    best_column: str = "avgTimeMs",
) -> tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """§8.3 CSVs from any producer (a sweep run elsewhere, the MIOpen perf DB's translation),
    validated and expanded exactly as `publish` does, as one contribution.

    Such rows carry no engine binding, so the binding is what the caller states: the engine,
    and the selector revision the measurements were taken under.
    """
    frame = publish_frame(
        [pathlib.Path(p) for p in csvs],
        expand_descriptor=expand_descriptor,
        scope_by=scope_by,
        resolve_duplicates_by=resolve_duplicates_by,
        best_column=best_column,
    )
    if "arch" not in frame.columns:
        frame["arch"] = arch
    rows = [
        {key: _native(value) for key, value in r.items()}
        for r in frame.to_dict(orient="records")
    ]
    kernel_fields = sorted(
        c[len("kernel.") :] for c in frame.columns if c.startswith("kernel.")
    )
    published = sorted(
        c for c in frame.columns if "." in c and not c.startswith("kernel.")
    )
    trained_against = {"selector_revision": selector_revision}
    entry = {
        "collected_at": collected_at,
        "role": role,
        "engine": engine,
        "engine_id": engine_id,
        "engine_name": engine,
        "sources": [None],
        "trained_against": trained_against,
        "selector_revision": selector_revision,
        "arches": sorted({str(r.get("arch")) for r in rows}),
        "devices": sorted({str(r.get("device")) for r in rows}),
        "graph_count": len({r.get("benchmark") for r in rows}),
        "row_counts": {"None": len(rows)},
        "published": published,
        "kernel_fields": kernel_fields,
        "knob_encodings": {},
        "shipping_knobs": [],
        "collection_knobs": [],
        "graphs": [],
        "commands": [],
        "failed_graphs": [],
        # How the rows were made trainable: generate reads these to train as `train` would.
        "training_options": {
            "expand_descriptor": list(expand_descriptor),
            "scope_by": scope_by,
        },
        "contribution_id": contribution_id,
        "origin": {
            "kind": "csv",
            "paths": [str(pathlib.Path(p).resolve()) for p in csvs],
        },
        "purpose": purpose,
        "corpus_id": corpus_id,
    }
    return entry, {None: rows}


def export_rows(
    directory: pathlib.Path,
    *,
    contributions: Iterable[str] | None = None,
    purposes: Iterable[str] | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Rows of the named contributions (or of every contribution with one of `purposes`), each
    exactly as its source wrote it: no bookkeeping, only the keys it carried.

    `source` keeps one metric's rows where a contribution measured several; a contribution that
    measured one answers in it whatever it is called, as its collection's lone `corpus.json` did.

    For a stage that reads rows rather than a dataset -- `evaluate` scoring a model on an
    evaluation corpus's measurements.
    """
    manifest = read_manifest(directory)
    if manifest is None:
        raise DatasetError(f"{directory} holds no dataset")
    wanted = set(contributions or [])
    chosen = {
        c["contribution_id"]: c
        for c in manifest["contributions"]
        if (
            c["contribution_id"] in wanted
            if wanted
            else c.get("purpose", "training") in set(purposes or PURPOSES)
        )
    }
    missing = sorted(wanted - set(chosen))
    if missing:
        raise DatasetError(f"{directory} holds no contribution {missing}")
    rows = []
    for row in read_rows(directory):
        entry = chosen.get(row["contribution_id"])
        if entry is None:
            continue
        if (
            source is not None
            and len(entry.get("sources") or [None]) > 1
            and row["source"] != source
        ):
            continue
        kept = (entry.get("columns") or {}).get(str(row["source"]))
        rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in BOOKKEEPING
                and (key in kept if kept else value is not None)
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# `python -m uhd_gen.dataset add`
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m uhd_gen.dataset add",
        description="Convert one source's measurements into a stored dataset (append).",
    )
    parser.add_argument(
        "--into", required=True, type=pathlib.Path, help="Dataset directory"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--collection",
        nargs="+",
        type=pathlib.Path,
        help="Raw collection directories (appended in one write)",
    )
    source.add_argument(
        "--generation", type=pathlib.Path, help="An old generate run's folder"
    )
    source.add_argument("--csv", nargs="+", type=pathlib.Path, help="§8.3 CSV files")
    parser.add_argument("--purpose", default="training", choices=PURPOSES)
    parser.add_argument("--corpus-id")
    parser.add_argument("--contribution-id")
    # --csv only: the binding the rows do not carry, and publish's options.
    parser.add_argument("--engine")
    parser.add_argument("--engine-id", type=int)
    parser.add_argument("--role", default="sort_kernel_catalog")
    parser.add_argument("--arch")
    parser.add_argument("--selector-revision")
    parser.add_argument("--collected-at")
    parser.add_argument("--expand-descriptor", action="append", default=[])
    parser.add_argument("--scope-by")
    parser.add_argument("--resolve-duplicates")
    parser.add_argument("--best-column", default="avgTimeMs")
    args = parser.parse_args(argv)

    common = {
        "purpose": args.purpose,
        "corpus_id": args.corpus_id,
        "contribution_id": args.contribution_id,
    }
    try:
        if args.collection:
            if args.contribution_id and len(args.collection) > 1:
                parser.error(
                    "--contribution-id names one contribution; give one --collection"
                )
            results = add_many(
                args.into, [from_collection(c, **common) for c in args.collection]
            )
        elif args.generation:
            results = [add(args.into, *from_generation(args.generation, **common))]
        else:
            missing = [
                flag
                for flag, value in (
                    ("--engine", args.engine),
                    ("--engine-id", args.engine_id),
                    ("--arch", args.arch),
                    ("--selector-revision", args.selector_revision),
                    ("--collected-at", args.collected_at),
                    ("--contribution-id", args.contribution_id),
                )
                if value is None
            ]
            if missing:
                parser.error(
                    f"--csv needs {', '.join(missing)}: the rows carry no binding"
                )
            results = [
                add(
                    args.into,
                    *from_csv(
                        args.csv,
                        engine=args.engine,
                        engine_id=args.engine_id,
                        role=args.role,
                        arch=args.arch,
                        selector_revision=args.selector_revision,
                        collected_at=args.collected_at,
                        expand_descriptor=args.expand_descriptor,
                        scope_by=args.scope_by,
                        resolve_duplicates_by=args.resolve_duplicates,
                        best_column=args.best_column,
                        **common,
                    ),
                )
            ]
    except (DatasetError, ValidationError, ValueError, OSError) as error:
        print(f"uhd_gen.dataset add: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"dataset": str(args.into), "contributions": results}))
    return 0


def export_main(argv: list[str] | None = None) -> int:
    """`python -m uhd_gen.dataset export --from DIR --out FILE.json (--contribution ID... |
    --purpose P...) [--metric M]`."""
    parser = argparse.ArgumentParser(
        prog="python -m uhd_gen.dataset export",
        description="Write contributions' rows as a JSON corpus, as their sources wrote them.",
    )
    parser.add_argument("--from", dest="source", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    chosen = parser.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--contribution", nargs="+")
    chosen.add_argument("--purpose", nargs="+", choices=PURPOSES)
    parser.add_argument(
        "--metric", help="One metric's rows, where a contribution measured several"
    )
    args = parser.parse_args(argv)
    try:
        rows = export_rows(
            args.source,
            contributions=args.contribution,
            purposes=args.purpose,
            source=args.metric,
        )
    except DatasetError as error:
        print(f"uhd_gen.dataset export: {error}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}))
    return 0
