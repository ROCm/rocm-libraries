# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Training data stored as one Parquet dataset per engine, arch, revision and role.

Collection writes a resumable raw record; this module converts it, or any other source, into
the form training reads. A dataset is a directory:

* `dataset.parquet` -- every converted row, tagged with its `contribution_id`, `collected_at`,
  metric `source`, `purpose` (training, holdout or eval) and `corpus_id`.
* `training.parquet` -- what training reads, rebuilt after every change: the training
  contributions' usable rows, one per configuration per shape, the best (training_table.py).
* `dataset_manifest.json` -- one entry per contribution with what its collection manifest
  recorded, the binding every contribution must share, and a `history` of every contribution
  removed, replaced or retagged.

A contribution is stored under an id derived from its binding and its rows, so the same
measurements added twice are added once and different measurements never share an id; its
`label` is the readable name it was added under. `add` only appends. Removing, replacing and
retagging are their own operations (`remove`, `add --replace`, `retag`).
`uhd_gen.generate.load_dataset` picks each shape's label when it reads.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

import pandas as pd

from .. import addressing
from .publish import ValidationError, publish_frame

DATASET_SCHEMA = "uhd_gen.dataset/1"
DATA_FILE = "dataset.parquet"
TRAINING_FILE = "training.parquet"
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

#: Hex digits of the digest a contribution id keeps.
ID_LENGTH = 16

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
    """Every row of the dataset as records, missing values None, as a collection's JSON
    corpus held them."""
    return _read(pathlib.Path(directory) / DATA_FILE)


def read_training_rows(directory: pathlib.Path) -> list[dict[str, Any]]:
    """`training.parquet`'s rows, as `read_rows` returns the dataset's."""
    return _read(pathlib.Path(directory) / TRAINING_FILE)


def _read(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    return [
        {key: _native(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def training_contributions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The contributions training reads by default: purpose `training`, oldest first."""
    return sorted(
        (
            c
            for c in manifest["contributions"]
            if c.get("purpose", "training") == "training"
        ),
        key=lambda c: (c["collected_at"], c["contribution_id"]),
    )


def measurements(
    rows: Iterable[dict[str, Any]], chosen: list[dict[str, Any]], sources: Iterable[Any]
) -> dict[Any, list[tuple[tuple, dict[str, Any], dict[str, Any]]]]:
    """source -> [((contribution order, device), record, stored row)] for the `chosen`
    contributions' rows, in the order measured. A record is the row exactly as its source
    wrote it: no bookkeeping, only the keys it carried."""
    order = {c["contribution_id"]: index for index, c in enumerate(chosen)}
    columns = {c["contribution_id"]: c.get("columns") for c in chosen}
    buckets: dict = {}
    for row in rows:
        identity = row["contribution_id"]
        if identity not in order:
            continue
        kept = columns[identity] and columns[identity].get(str(row["source"]))
        record = {
            key: value
            for key, value in row.items()
            if key not in BOOKKEEPING and (key in kept if kept else value is not None)
        }
        buckets.setdefault((row["source"], order[identity]), []).append((record, row))
    measured = {}
    for source in sources:
        key = None if source is None else str(source)
        measured[source] = [
            ((index, str(record["device"])), record, row)
            for index in range(len(chosen))
            for record, row in buckets.get((key, index), [])
        ]
    return measured


def _training_view(
    manifest: dict[str, Any], frame: pd.DataFrame | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`training.parquet`'s rows and the manifest's summary of them, from the dataset's rows."""
    from .training_table import best_per_shape

    chosen = training_contributions(manifest)
    sources = sorted(
        {source for c in chosen for source in c.get("sources") or [None]},
        key=lambda source: "" if source is None else str(source),
    )
    rows = [] if frame is None else frame.to_dict(orient="records")
    rows = [{key: _native(value) for key, value in row.items()} for row in rows]
    kept, summary = [], {}
    try:
        for source, measured in measurements(rows, chosen, sources).items():
            indices, superseded, excluded = best_per_shape(
                [(session, record) for session, record, _ in measured]
            )
            kept.extend(measured[index][2] for index in indices)
            summary[str(source)] = {
                "rows": len(indices),
                "superseded": superseded,
                "excluded": excluded,
            }
    except ValueError as error:
        raise DatasetError(str(error)) from None
    return pd.DataFrame(kept), {
        "file": TRAINING_FILE,
        "contributions": [c["contribution_id"] for c in chosen],
        "sources": summary,
    }


def _file_digest(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_training_view(
    directory: pathlib.Path, manifest: dict[str, Any], view: pd.DataFrame, summary
) -> None:
    path = directory / TRAINING_FILE
    _write_data(path, view)
    # The manifest is written after the table, so a write interrupted between the two leaves
    # a table the manifest does not describe; the digest is how a reader tells.
    manifest["training_view"] = {**summary, "sha256": _file_digest(path)}


def training_view_matches(directory: pathlib.Path, manifest: dict[str, Any]) -> bool:
    """Whether `training.parquet` is the table the manifest's `training_view` describes."""
    view = manifest.get("training_view") or {}
    return "sha256" in view and view["sha256"] == _file_digest(
        pathlib.Path(directory) / TRAINING_FILE
    )


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


def contribution_digest(
    entry: dict[str, Any], rows_by_source: dict[Any, list[dict[str, Any]]]
) -> str:
    """The id a contribution is stored under: a digest of its binding and its rows.

    The same measurements get the same id whatever they are called, and different measurements
    never share one. What can change without the measurements changing (`label`, `purpose`,
    `corpus_id`, `origin`) is not part of it. Values are compared as Parquet returns them --
    an integral float as the integer, NaN as missing -- and a missing value is the same fact
    as an absent key (an exported row carries every column of its contribution), so rows
    exported and added again keep their id.
    """

    def canonical(value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value):
                return None
            return int(value) if value.is_integer() else value
        if isinstance(value, dict):
            items = ((str(key), canonical(item)) for key, item in value.items())
            return {key: item for key, item in items if item is not None}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value

    payload = canonical(
        {
            "binding": {key: entry.get(key) for key in SHARED},
            "rows": [
                [str(source), rows_by_source[source]]
                for source in sorted(rows_by_source, key=str)
            ],
        }
    )
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ID_LENGTH]


def select(
    contributions: Iterable[dict[str, Any]], names: Iterable[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """The contributions `names` designate, each name an id or a label, in their stored order,
    and the names that designate none."""
    names = set(names)
    chosen = [
        c
        for c in contributions
        if c["contribution_id"] in names or c.get("label") in names
    ]
    matched = {c["contribution_id"] for c in chosen} | {c.get("label") for c in chosen}
    return chosen, sorted(names - matched)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_data(path: pathlib.Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        path.unlink(missing_ok=True)
        return
    try:
        _write_atomically(path, lambda target: frame.to_parquet(target, index=False))
    except (
        Exception
    ) as error:  # pyarrow refuses a column whose types the rows disagree on
        raise DatasetError(
            f"rows cannot be stored as one Parquet table: {error}"
        ) from None


def _write_manifest(directory: pathlib.Path, manifest: dict[str, Any]) -> None:
    _write_atomically(
        directory / MANIFEST_FILE,
        lambda path: path.write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
        ),
    )


def _tag(
    entry: dict[str, Any], rows_by_source: dict[Any, list[dict[str, Any]]]
) -> tuple[dict, list]:
    """The contribution's rows with the store's bookkeeping, and its entry with its id and
    columns."""
    identity = contribution_digest(entry, rows_by_source)
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
        raise DatasetError(
            f"contribution {entry.get('label') or identity} holds no rows"
        )
    # The keys each source's rows carried, so a row reads back without other contributions'
    # columns.
    entry = {
        **entry,
        "contribution_id": identity,
        "columns": {
            str(source): sorted({key for row in rows for key in row})
            for source, rows in rows_by_source.items()
        },
    }
    return entry, tagged


def add_many(
    directory: pathlib.Path,
    contributions: list[tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]],
    *,
    replace: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Append contributions in one write: their rows, tagged, and their manifest entries.

    Returns one {"added", "rows", "contribution_id", "label"} per contribution. Measurements
    already stored (the same id) are skipped. One whose rows are stored but whose manifest
    entry is not (a write interrupted between the two files) gets its entry, never its rows a
    second time. A differing binding or contradicting knob addressing is refused, and then
    nothing of this call is written.

    `replace` names contributions (ids or labels) removed in the same write, and recorded in
    the manifest's history as replaced by these.
    """
    return _write(directory, contributions, list(replace))[0]


def _write(
    directory: pathlib.Path,
    contributions: list[tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]],
    replace: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`add_many`'s write; also returns the entries `replace` removed."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = [_tag(entry, rows) for entry, rows in contributions]
    results = []
    with _locked(directory):
        manifest = read_manifest(directory)
        if manifest is None:
            if replace or not prepared:
                raise DatasetError(f"{directory} holds no dataset")
            first = prepared[0][0]
            manifest = {
                "schema": DATASET_SCHEMA,
                **{key: first.get(key) for key in SHARED},
                "contributions": [],
            }
        removed, unknown = select(manifest["contributions"], replace)
        if unknown:
            raise DatasetError(f"{directory} holds no contribution {unknown}")
        removed_ids = {c["contribution_id"] for c in removed}
        manifest["contributions"] = [
            c
            for c in manifest["contributions"]
            if c["contribution_id"] not in removed_ids
        ]
        present = {c["contribution_id"] for c in manifest["contributions"]}
        data = directory / DATA_FILE
        existing = pd.read_parquet(data) if data.is_file() else None
        if existing is not None and removed_ids:
            existing = existing[~existing["contribution_id"].isin(removed_ids)]
        stored = (
            existing["contribution_id"].value_counts().to_dict()
            if existing is not None
            else {}
        )
        fresh, frames = [], []
        for entry, tagged in prepared:
            identity = entry["contribution_id"]
            result = {"contribution_id": identity, "label": entry.get("label")}
            if identity in present:
                results.append({**result, "added": False, "rows": 0})
                continue
            _agree(manifest, entry)
            present.add(identity)
            fresh.append(entry)
            if identity in stored:
                if stored[identity] != len(tagged):
                    raise DatasetError(
                        f"{data} holds {stored[identity]} row(s) of contribution {identity}, "
                        f"which its manifest does not record, and the contribution has "
                        f"{len(tagged)}; the dataset was written by something else"
                    )
            else:
                frames.append(pd.DataFrame(tagged))
            results.append({**result, "added": True, "rows": len(tagged)})
        if not fresh and not removed:
            return results, removed
        try:
            addressing.merge_manifests(
                [c.get("knob_encodings") for c in manifest["contributions"] + fresh]
            )
        except ValueError as error:
            raise DatasetError(
                f"knob addressing contradicts the dataset: {error}"
            ) from None
        parts = ([] if existing is None else [existing]) + frames
        full = pd.concat(parts, ignore_index=True) if parts else None
        manifest["contributions"].extend(fresh)
        view, summary = _training_view(manifest, full)
        if frames or removed:
            _write_data(data, pd.DataFrame() if full is None else full)
        _write_training_view(directory, manifest, view, summary)
        for entry in removed:
            manifest.setdefault("history", []).append(
                {
                    "action": "replace" if prepared else "remove",
                    "contribution_id": entry["contribution_id"],
                    "label": entry.get("label"),
                    "at": _now(),
                    "replaced_by": [c["contribution_id"] for c in fresh],
                    "entry": entry,
                }
            )
        _write_manifest(directory, manifest)
    return results, removed


def remove(directory: pathlib.Path, names: Iterable[str]) -> list[dict[str, Any]]:
    """Remove the named contributions (ids or labels), rows and entries, in one write, and
    return their ids and labels; the manifest's history keeps each removed entry."""
    names = list(names)
    if not names:
        raise DatasetError("name at least one contribution to remove")
    return [
        {"contribution_id": c["contribution_id"], "label": c.get("label")}
        for c in _write(directory, [], names)[1]
    ]


def retag(
    directory: pathlib.Path,
    names: Iterable[str],
    *,
    purpose: str | None = None,
    corpus_id: str | None = None,
) -> list[str]:
    """Change the purpose or corpus of the named contributions (ids or labels), on their
    entries and their rows; returns their ids. The measurements, and so the ids, are kept.
    """
    if purpose is None and corpus_id is None:
        raise DatasetError("retag needs --purpose or --corpus-id")
    if purpose is not None and purpose not in PURPOSES:
        raise DatasetError(f"purpose {purpose!r} is not one of {PURPOSES}")
    directory = pathlib.Path(directory)
    with _locked(directory):
        manifest = read_manifest(directory)
        if manifest is None:
            raise DatasetError(f"{directory} holds no dataset")
        chosen, unknown = select(manifest["contributions"], names)
        if unknown or not chosen:
            raise DatasetError(f"{directory} holds no contribution {unknown}")
        changes = {
            key: value
            for key, value in (("purpose", purpose), ("corpus_id", corpus_id))
            if value is not None
        }
        ids = [c["contribution_id"] for c in chosen]
        data = directory / DATA_FILE
        frame = pd.read_parquet(data) if data.is_file() else None
        if frame is not None:
            mine = frame["contribution_id"].isin(ids)
            for key, value in changes.items():
                frame.loc[mine, key] = value
        for entry in chosen:
            manifest.setdefault("history", []).append(
                {
                    "action": "retag",
                    "contribution_id": entry["contribution_id"],
                    "label": entry.get("label"),
                    "at": _now(),
                    "from": {key: entry.get(key) for key in changes},
                    "to": changes,
                }
            )
            entry.update(changes)
        view, summary = _training_view(manifest, frame)
        if frame is not None:
            _write_data(data, frame)
        _write_training_view(directory, manifest, view, summary)
        _write_manifest(directory, manifest)
    return ids


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
    """An L1 contribution's rows pass the checks training applies to them (§13.2, the binding).
    Its repeats (one graph measured on several GPUs of one kind) are stored as measured;
    training reduces them."""
    from ..immediate import normalize_corpus

    for rows in rows_by_source.values():
        normalize_corpus(pd.DataFrame(rows), repeats=True)


def from_collection(
    collection: pathlib.Path,
    *,
    purpose: str = "training",
    corpus_id: str | None = None,
    label: str | None = None,
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
        "label": label or collection.name,
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
    label: str | None = None,
) -> tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """A one-shot `generate` run's measurements, read from its model folder, as a contribution.

    Fields come from its rows (engine, arch, devices, binding), its generation manifest (engine
    id, provenance, graphs, commands), and its first bench call's time as `collected_at`.
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
    read = []
    for metric in metrics:
        # Everything the run measured; a run that predates measurements.json kept it in
        # its corpus files.
        stem = "" if len(metrics) == 1 else f"_{metric}"
        name = next(
            (
                f"{kind}{stem}.json"
                for kind in ("measurements", "corpus")
                if (generation / f"{kind}{stem}.json").is_file()
            ),
            f"corpus{stem}.json",
        )
        read.append(generation / name)
        rows_by_source[metric] = json.loads(read[-1].read_text(encoding="utf-8"))
    every = [row for rows in rows_by_source.values() for row in rows]
    revisions = {json.loads(row["binding"])["selector_revision"] for row in every}
    if len(revisions) != 1:
        raise DatasetError(f"{generation} measured under revisions {sorted(revisions)}")
    _check_immediate(rows_by_source)
    published: set[str] = set()
    for row in every:
        published.update(json.loads(row["features"]))
    # When the first bench call ran, else when the rows it read were written.
    logs = sorted((generation / "commands").glob("*.stdout.json"))
    first = min(p.stat().st_mtime for p in (logs or read))
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
        "label": label or f"generation-{generation.parent.name}",
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
    label: str | None = None,
    purpose: str = "training",
    corpus_id: str | None = None,
    expand_descriptor: Iterable[str] = (),
    scope_by: str | None = None,
    resolve_duplicates_by: str | None = None,
    best_column: str = "avgTimeMs",
) -> tuple[dict[str, Any], dict[Any, list[dict[str, Any]]]]:
    """§8.3 CSVs from any producer, validated and expanded as `publish` does, as one
    contribution. The rows carry no binding, so the caller states the engine and selector
    revision.
    """
    csvs = [pathlib.Path(p) for p in csvs]
    frame = publish_frame(
        csvs,
        expand_descriptor=expand_descriptor,
        scope_by=scope_by,
        resolve_duplicates_by=resolve_duplicates_by,
        best_column=best_column,
    )
    if "arch" not in frame.columns:
        frame["arch"] = arch
    # A stored measurement is keyed by its graph and its device kind (training_table.py);
    # `device.*` features describe the device but do not name it.
    missing = [c for c in ("benchmark", "device") if c not in frame.columns]
    if missing:
        raise DatasetError(
            f"{', '.join(str(p) for p in csvs)}: no {' or '.join(map(repr, missing))} "
            "column; a stored measurement is keyed by its graph's content key and its "
            "device kind (hipdnn_bench writes both)"
        )
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
        "label": label or csvs[0].stem,
        "origin": {
            "kind": "csv",
            "paths": [str(p.resolve()) for p in csvs],
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
    """Rows of the named contributions (ids or labels; or of every contribution with one of
    `purposes`), each as its source wrote it: no bookkeeping, only the keys it carried.

    `source` keeps one metric's rows where a contribution measured several; a single-metric
    contribution answers whatever `source` is.
    """
    manifest = read_manifest(directory)
    if manifest is None:
        raise DatasetError(f"{directory} holds no dataset")
    if contributions:
        selected, missing = select(manifest["contributions"], contributions)
        if missing:
            raise DatasetError(f"{directory} holds no contribution {missing}")
    else:
        selected = [
            c
            for c in manifest["contributions"]
            if c.get("purpose", "training") in set(purposes or PURPOSES)
        ]
    chosen = {c["contribution_id"]: c for c in selected}
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
        "--generation", type=pathlib.Path, help="A one-shot generate run's folder"
    )
    source.add_argument("--csv", nargs="+", type=pathlib.Path, help="§8.3 CSV files")
    parser.add_argument("--purpose", default="training", choices=PURPOSES)
    parser.add_argument("--corpus-id")
    parser.add_argument(
        "--label",
        help="A readable name for the contribution (default: the collection's directory, the "
        "generation's folder, or the first CSV's name); selectable wherever an id is",
    )
    parser.add_argument(
        "--replace",
        nargs="+",
        default=[],
        metavar="CONTRIBUTION",
        help="Contributions (ids or labels) this one replaces, removed in the same write",
    )
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
        "label": args.label,
    }
    try:
        if args.collection:
            if args.label and len(args.collection) > 1:
                parser.error("--label names one contribution; give one --collection")
            results = add_many(
                args.into,
                [from_collection(c, **common) for c in args.collection],
                replace=args.replace,
            )
        elif args.generation:
            results = add_many(
                args.into,
                [from_generation(args.generation, **common)],
                replace=args.replace,
            )
        else:
            missing = [
                flag
                for flag, value in (
                    ("--engine", args.engine),
                    ("--engine-id", args.engine_id),
                    ("--arch", args.arch),
                    ("--selector-revision", args.selector_revision),
                    ("--collected-at", args.collected_at),
                )
                if value is None
            ]
            if missing:
                parser.error(
                    f"--csv needs {', '.join(missing)}: the rows carry no binding"
                )
            results = add_many(
                args.into,
                [
                    from_csv(
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
                    )
                ],
                replace=args.replace,
            )
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
    args.out.write_text(json.dumps(rows) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}))
    return 0


def remove_main(argv: list[str] | None = None) -> int:
    """`python -m uhd_gen.dataset remove --from DIR --contribution NAME...`."""
    parser = argparse.ArgumentParser(
        prog="python -m uhd_gen.dataset remove",
        description="Remove contributions' rows and entries; the manifest's history keeps "
        "each removed entry.",
    )
    parser.add_argument("--from", dest="source", required=True, type=pathlib.Path)
    parser.add_argument(
        "--contribution", nargs="+", required=True, help="Contribution ids or labels"
    )
    args = parser.parse_args(argv)
    try:
        removed = remove(args.source, args.contribution)
    except DatasetError as error:
        print(f"uhd_gen.dataset remove: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"dataset": str(args.source), "removed": removed}))
    return 0


def retag_main(argv: list[str] | None = None) -> int:
    """`python -m uhd_gen.dataset retag --in DIR --contribution NAME... [--purpose P]
    [--corpus-id C]`."""
    parser = argparse.ArgumentParser(
        prog="python -m uhd_gen.dataset retag",
        description="Change contributions' purpose or corpus; their measurements are kept.",
    )
    parser.add_argument("--in", dest="target", required=True, type=pathlib.Path)
    parser.add_argument(
        "--contribution", nargs="+", required=True, help="Contribution ids or labels"
    )
    parser.add_argument("--purpose", choices=PURPOSES)
    parser.add_argument("--corpus-id")
    args = parser.parse_args(argv)
    try:
        ids = retag(
            args.target,
            args.contribution,
            purpose=args.purpose,
            corpus_id=args.corpus_id,
        )
    except DatasetError as error:
        print(f"uhd_gen.dataset retag: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"dataset": str(args.target), "retagged": ids}))
    return 0
