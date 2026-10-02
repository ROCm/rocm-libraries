# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collect engine-immediate or enrolled-candidate measurements and generate UHDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import addressing
from .catalog import DeterministicCatalogError, candidate_density
from .correctness import (
    REASON,
    VERDICT,
    known_wrong,
    numerical_verdict,
    suppress_timings,
)
from .coverage import device_field_coverage, enforce_device_coverage, propose_features
from .evaluate import BENCHMARK_COLUMN, problem_keys, resolve_grouping, split_problems
from .features import (
    build_features_signature,
    require_admissible_kernel_axes,
    signature_references,
)
from .knobs import graph_bound_twins
from .provenance import ROLES, descriptor_id, snapshot_provenance
from .immediate import (
    LABEL_STATISTIC,
    ROLE,
    normalize_row,
    normalize_corpus,
    training_binding,
    validate_signature,
)
from .ranking_metrics import DEFAULT_RANKING_METRIC, RANKING_METRICS, ranking_metric

logger = logging.getLogger(__name__)

#: The graph list `hipdnn_corpus_gen` writes at a corpus root (CorpusManifest.hpp).
MANIFEST = "manifest.json"


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--graphs",
        nargs="+",
        help="Graph files -- JSON, or the binary FlatBuffers hipdnn_corpus_gen writes "
        "as graphs/*.fb -- or directories. A hipdnn_corpus_gen root (or its "
        "manifest.json) is read through the manifest's graph list; any other "
        "directory is searched recursively",
    )
    inputs.add_argument(
        "--collection",
        nargs="+",
        metavar="DIR",
        help="Train from recorded collections (written by --collect-only) instead of "
        "measuring; several are merged into one measurement of each configuration "
        "on each shape, the newest winning",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Measure --graphs, write the collection to --output-dir, and stop",
    )
    parser.add_argument(
        "--shard",
        metavar="K/N",
        help="With --collect-only: measure the K-th of N slices of --graphs "
        "(0-based, every N-th graph in path order), so N GPUs can collect "
        "one corpus in parallel as N collections",
    )
    parser.add_argument(
        "--descriptor-tree",
        required=True,
        help="Shipping descriptor tree; authored knobs are preserved",
    )
    parser.add_argument(
        "--engine", help="UED name/UUID or canonical immediate engine name"
    )
    parser.add_argument(
        "--engine-id",
        type=int,
        help="Public hipDNN engine ID used by hipdnn_bench (required to measure)",
    )
    parser.add_argument(
        "--bench", default="hipdnn_bench", help="Public hipdnn_bench executable"
    )
    parser.add_argument("--plugin-dir")
    parser.add_argument(
        "--device",
        action="append",
        help="HIP_VISIBLE_DEVICES selection; repeat to collect multiple devices",
    )
    parser.add_argument(
        "--knob",
        action="append",
        default=[],
        help="Explicit NAME=INTEGER collection pin (repeatable)",
    )
    parser.add_argument(
        "--workspace-limit", type=int, help="L1 immediate workspace constraint in bytes"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory for reproducible collection and model artifacts",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--features", nargs="+", help="Explicit full published raw feature names"
    )
    source.add_argument(
        "--feature-signature", help="Explicit canonical inline JSON feature array"
    )
    parser.add_argument(
        "--dim-tile",
        action="append",
        default=[],
        metavar="DIMENSION=KERNEL_FIELD",
        help="Author-declared dimension-to-tile pair used to propose ceil_div/remainder features",
    )
    parser.add_argument(
        "--feature-evaluator", help="Shared hipdnn_uhd_features executable"
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument(
        "--recall",
        action="store_true",
        help="The engine's shape space is closed -- it serves only the shapes it was "
        "compiled for (a pack-bound engine) -- so no unseen shape can reach the "
        "model. Train on every shape and report accuracy over all of them as "
        "recall, instead of holding --eval-fraction of them out of the model",
    )
    parser.add_argument(
        "--eval-benchmarks",
        metavar="FILE",
        help="JSON list of graph ids (benchmark) to hold out, on every device, instead of "
        "drawing --eval-fraction of the problems. Fixed when a corpus is generated, so the "
        "same graphs score every model trained from it however its collections grow",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument("--name", default="Generated UHD")
    parser.add_argument(
        "--arch", help="Promotion arch; otherwise infer one observed architecture"
    )
    parser.add_argument(
        "--uhd-id",
        action="append",
        default=[],
        dest="uhd_ids",
        metavar="METRIC=UUID",
        help="UHD id for one metric's model (repeatable). A bare UUID names the "
        "model of a single-metric run. An engine with no UED reads only the "
        "ids its provider declares per metric: those default to the id its "
        "description reports, and are required when it reports none",
    )
    parser.add_argument(
        "--max-graph-failures",
        type=float,
        default=0.05,
        metavar="FRACTION",
        help="Fraction of graphs whose collection may fail and be skipped (each is "
        "recorded in generation_manifest.json); more fails the run with the "
        "list (default: 0.05)",
    )
    parser.add_argument("--role", default="sort_kernel_catalog", choices=ROLES)
    parser.add_argument(
        "--metric",
        nargs="+",
        action="extend",
        choices=tuple(RANKING_METRICS),
        help="Ranking metric(s) to train, one UHD per metric from the same "
        f"collection (repeatable; default: {DEFAULT_RANKING_METRIC})",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Validate installation but leave shipping descriptors untouched",
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _absent_as_null(value):
    """Raw collected rows, with every non-finite float replaced by null.

    `allow_nan=False` is the right guard for a model, a provenance snapshot or a manifest:
    NaN is not JSON, and a number that cannot be written is a number that should not have
    been computed. It is the wrong guard for the raw measurement log, where a missing
    optional field -- `stddevMs` on a single-iteration run, `iters` on a run that reported
    none -- arrives as NaN through pandas and means "absent", which JSON spells null.

    Without this, an engine that measured its whole corpus successfully loses the lot at
    the final write: run 67929365 collected 600 gfx950 graphs and died on
    "Out of range float values are not JSON compliant: nan" after the measurement was done.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _absent_as_null(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_absent_as_null(item) for item in value]
    return value


#: The recorded form of one measuring run, read back by `--collection`.
COLLECTION_SCHEMA = "uhd_gen.collection/1"
COLLECTION_MANIFEST = "collection_manifest.json"


def _corpus_name(sources: list, source) -> str:
    """One corpus per metric source, named plainly when there is only one."""
    return "corpus" if len(sources) == 1 else f"corpus_{source}"


def _eval_benchmarks(path: str) -> set[str]:
    """The graph ids `--eval-benchmarks` names: a JSON list, or an object whose `benchmarks`
    is one (the evaluation slice a corpus records)."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    listed = document.get("benchmarks") if isinstance(document, dict) else document
    if not isinstance(listed, list) or not all(
        isinstance(b, str) and b for b in listed
    ):
        raise ValueError(f"{path}: --eval-benchmarks needs a list of graph ids")
    if not listed:
        raise ValueError(f"{path}: --eval-benchmarks names no graph")
    return set(listed)


def write_collection(
    stage: Path,
    *,
    collected_at: str,
    role: str,
    engine: str | None,
    engine_id: int,
    sources: list,
    rows: dict,
    published: set,
    commands: list,
    graph_inputs: list,
    provenance,
    knob_encodings: dict,
    shipping_knobs: list,
    collection_knobs: list,
    kernel_fields: set,
    shard: str | None = None,
    failed_graphs: list | None = None,
) -> dict:
    """Write what a measuring run produced so a later `--collection` can train from it.

    Everything training reads from a collection is here, and nothing training decides is:
    no split, no feature recipe, no model. For an engine-immediate collection the binding
    is checked now, so a collection that cannot train (mixed selector revisions within one
    run) is refused while it is still one run's problem rather than a merge's.
    """
    immediate = role == ROLE
    selector_revision = None
    for source in sources:
        frame = pd.DataFrame(rows[source])
        if immediate:
            _, binding = training_binding(normalize_corpus(frame))
            if provenance is not None and binding["trained_against"] != provenance:
                raise ValueError(
                    "the engine's descriptor provenance changed between metric collections"
                )
            provenance = binding["trained_against"]
            selector_revision = binding["selector_revision"]
        name = _corpus_name(sources, source)
        frame.to_csv(stage / f"{name}.csv", index=False)
        _write_json(stage / f"{name}.json", _absent_as_null(rows[source]))
    every = [row for source in sources for row in rows[source]]
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "collected_at": collected_at,
        "role": role,
        "engine": engine,
        "engine_id": engine_id,
        "engine_name": next(
            (row.get("engine_name") for row in every if row.get("engine_name")), None
        ),
        "sources": sources,
        "trained_against": provenance,
        "selector_revision": selector_revision,
        "arches": sorted({row["arch"] for row in every}),
        "devices": sorted({str(row["device"]) for row in every}),
        "graph_count": len({row["benchmark"] for row in every}),
        "row_counts": {str(source): len(rows[source]) for source in sources},
        "published": sorted(published),
        "kernel_fields": sorted(kernel_fields),
        "knob_encodings": knob_encodings,
        "shipping_knobs": shipping_knobs,
        "collection_knobs": collection_knobs,
        "graphs": graph_inputs,
        "commands": commands,
        # Graphs skipped within the --max-graph-failures budget, each with its error: what
        # this collection does not cover, recorded where training will read it.
        "failed_graphs": list(failed_graphs or []),
    }
    if shard:
        # Which slice of its corpus this is: N shards of one corpus are N collections, and
        # together -- not alone -- they are the corpus's measurement.
        manifest["shard"] = shard
    _write_json(stage / COLLECTION_MANIFEST, manifest)
    return manifest


def parse_shard(text: str) -> tuple[int, int]:
    """`K/N` -> (K, N), with 0 <= K < N."""
    try:
        index, count = (int(part) for part in str(text).split("/"))
    except ValueError:
        raise ValueError(f"--shard takes K/N, not {text!r}") from None
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard {text}: need 0 <= K < N")
    return index, count


def _measurement_key(row: dict) -> tuple:
    """What a measurement is of: a configuration on a shape, on an architecture.

    A shape is a graph -- the graph id is content-derived, so the same geometry has the
    same id wherever it was generated -- and the same shape on another arch is another
    problem. A catalog sweep times many kernel configurations on one shape, and each is a
    row of its own; an L1 row carries no configuration (the engine chose it), so its key
    is the shape alone.
    """
    return (
        str(row["benchmark"]),
        str(row["arch"]),
        str(row.get("kernel")),
        str(row.get("knob_settings")),
    )


def one_measurement_per_shape(measured: list) -> tuple[list, int]:
    """Keep one measurement of each configuration on each shape: the newest session's.

    `measured` is `(session, row)` in the order measured, a session being one run on one
    device. Returns (rows, dropped). A configuration measured twice on a shape -- on two
    GPUs of the arch, or in two collections -- is one training problem, not two: kept
    twice it is weighted twice, and split across the held-out boundary it scores the
    model on a problem it was trained on (the split keys on (graph, device), and another
    GPU of the same arch is another device). Distinct configurations on the same shape
    are distinct rows and are all kept. Older measurements stay in their collections;
    this only picks the label.

    A repeat taken under another binding (selector revision, descriptor set) is refused,
    not superseded: dropping it would hide the mix the corpus check exists to refuse.
    """
    newest: dict = {}
    binding: dict = {}
    for session, row in measured:
        key = _measurement_key(row)
        if binding.setdefault(key, row.get("binding")) != row.get("binding"):
            raise ValueError(
                f"{key[0]} on {key[1]} was measured under two engine bindings; "
                "the selector or descriptor provenance changed between measurements"
            )
        newest[key] = session
    kept = [
        row for session, row in measured if newest[_measurement_key(row)] == session
    ]
    return kept, len(measured) - len(kept)


def load_collections(paths: list, *, role: str, sources: list) -> dict:
    """Merge recorded collections into one measurement per shape, the newest winning.

    Collections are taken oldest to newest by `collected_at` (not the order named), and
    `one_measurement_per_shape` keeps the newest measurement of each configuration on
    each shape. The count of rows set aside is returned, so the model can say how much of
    the store it did not use.

    Refused rather than merged: different roles or engines, different trained_against
    (a model is bound to one selector revision or descriptor set), different knob
    addressing, and a requested metric a collection did not measure.
    """
    loaded = []
    for supplied in paths:
        directory = Path(supplied).resolve()
        manifest = json.loads(
            (directory / COLLECTION_MANIFEST).read_text(encoding="utf-8")
        )
        if manifest.get("schema") != COLLECTION_SCHEMA:
            raise ValueError(f"{directory} is not a {COLLECTION_SCHEMA} collection")
        loaded.append((manifest["collected_at"], str(directory), manifest))
    if not loaded:
        raise ValueError("--collection names no collection")
    loaded.sort(key=lambda item: (item[0], item[1]))
    first = loaded[0][2]
    for _, directory, manifest in loaded:
        if manifest["role"] != role:
            raise ValueError(
                f"{directory} was collected for {manifest['role']}, not {role}"
            )
        for key, what in (
            ("engine_name", "engine"),
            ("engine_id", "engine id"),
            ("selector_revision", "selector revision"),
            ("trained_against", "trained_against provenance"),
            ("collection_knobs", "collection knobs"),
        ):
            if manifest.get(key) != first.get(key):
                raise ValueError(
                    f"collections disagree on {what}: {first.get(key)!r} vs "
                    f"{manifest.get(key)!r} ({directory}); a model is trained against one"
                )
        missing = [source for source in sources if source not in manifest["sources"]]
        if missing:
            raise ValueError(
                f"{directory} did not measure {missing}; it holds {manifest['sources']}"
            )
    # Each collection records the addressing its own candidates showed: a subset of the
    # engine's numbering, which shards of one corpus split between them. Merged, refusing a
    # pin two of them read differently.
    try:
        knob_encodings = addressing.merge_manifests(
            m.get("knob_encodings") for _, _, m in loaded
        )
    except ValueError as error:
        raise ValueError(f"collections disagree on knob addressing: {error}") from None
    rows, superseded, published = {source: [] for source in sources}, 0, set()
    for source in sources:
        measured = []
        for order, (_, directory, manifest) in enumerate(loaded):
            corpus = (
                Path(directory) / f"{_corpus_name(manifest['sources'], source)}.json"
            )
            measured.extend(
                ((order, str(row["device"])), row)
                for row in json.loads(corpus.read_text(encoding="utf-8"))
            )
        rows[source], dropped = one_measurement_per_shape(measured)
        superseded += dropped
    for _, _, manifest in loaded:
        published.update(manifest["published"])
    return {
        "rows": rows,
        "published": published,
        "superseded_rows": superseded,
        "provenance": first["trained_against"],
        "knob_encodings": knob_encodings,
        "shipping_knobs": first["shipping_knobs"],
        "collection_knobs": first["collection_knobs"],
        "kernel_fields": set(first["kernel_fields"]),
        "engine_id": first["engine_id"],
        "graphs": [
            dict(graph, collection=directory)
            for _, directory, manifest in loaded
            for graph in manifest["graphs"]
        ],
        "commands": [
            dict(command, collection=directory)
            for _, directory, manifest in loaded
            for command in manifest["commands"]
        ],
        "failed_graphs": [
            dict(failure, collection=directory)
            for _, directory, manifest in loaded
            for failure in manifest.get("failed_graphs") or []
        ],
        "collections": [
            {
                "path": directory,
                "collected_at": collected_at,
                "devices": manifest["devices"],
                "rows": manifest["row_counts"],
            }
            for collected_at, directory, manifest in loaded
        ],
    }


def _descriptor(tree: Path, suffix: str, identity: str) -> tuple[Path, dict]:
    matches = []
    for path in sorted(tree.rglob("*" + suffix)):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("id") == identity:
            matches.append((path, document))
    if not matches:
        raise ValueError(f"descriptor {identity} ({suffix}) does not resolve in {tree}")
    if any(document != matches[0][1] for _, document in matches[1:]):
        raise ValueError(f"descriptor {identity} has conflicting definitions")
    return matches[0]


def _run_json(
    command: list[str], environment: dict, log_dir: Path, ordinal: int, commands: list
) -> dict:
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    entry = {
        "argv": command,
        "returncode": result.returncode,
        "environment": {
            key: environment.get(key)
            for key in (
                "HIPDNN_DESCRIPTOR_PATH",
                "HIPDNN_DESCRIPTOR_DIR",
                "HIPDNN_DESCRIPTOR_RUNTIME_DIR",
                "HIP_VISIBLE_DEVICES",
            )
        },
    }
    commands.append(entry)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{ordinal:06d}.stdout.json").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{ordinal:06d}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(
            f"hipdnn_bench failed ({result.returncode}): {result.stderr.strip()}"
        )
    try:
        response = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError(
            "hipdnn_bench did not emit one JSON response; see captured command output"
        ) from error
    if not isinstance(response, dict):
        raise ValueError("hipdnn_bench response must be an object")
    return response


def _identity(response: dict) -> tuple:
    identity = tuple(
        response.get(key)
        for key in (
            "engine_id",
            "graph_id",
            "device_id",
            "device_arch",
            "engine_descriptor_id",
            "engine_name",
        )
    )
    if any(value is None or value == "" for value in identity):
        raise ValueError("benchmark response lacks engine/graph/device identity")
    return identity


def _feature_map(response: dict, key: str) -> dict:
    mapping = response.get(key)
    if not isinstance(mapping, dict) or any(
        not isinstance(name, str) or not name or name.startswith("$")
        for name in mapping
    ):
        raise ValueError(f"{key} must contain canonical published names without '$'")
    return mapping


def _knob_tuple(candidate: dict) -> tuple:
    knobs = candidate.get("knob_settings")
    if not isinstance(knobs, dict) or any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int)
        for name, value in knobs.items()
    ):
        raise ValueError("candidate lacks an integer-valued enrolled knob tuple")
    return tuple(sorted(knobs.items()))


def _finite_positive(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def collect_graph(
    command: list[str],
    environment: dict,
    log_dir: Path,
    commands: list,
    *,
    engine_descriptor_id: str,
    addressing_table: dict | None = None,
) -> tuple[list[dict], set[str]]:
    """One bench invocation per graph: the sweep enumerates and times in one process.

    RFC 0019 §13.2: "Sweeping inside one process amortises" the plugin load, the graph
    build and the kernel compilation that a process per row pays once each. The enumerate
    call used to be a second process per graph that built the same catalog and threw the
    timings away; `--sweep --json` now returns the catalog it timed, so one startup covers
    both. The checks below are unchanged -- they simply read one response instead of two.
    """
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    seen_tuples: set[tuple] = set()
    measured = _run_json(
        [*command, "--sweep", "--json"], environment, log_dir, len(commands), commands
    )
    first = measured
    identity = _identity(measured)
    if str(identity[0]) != command[command.index("--engine-id") + 1]:
        raise ValueError("the sweep returned another engine's catalog")
    if identity[4] != engine_descriptor_id:
        raise ValueError("swept engine does not own the recorded UED provenance")
    total = measured.get("total_count")
    if not isinstance(total, int) or total < 0:
        raise ValueError("the sweep lacks a bounded total_count")
    batch = measured.get("candidates")
    if not isinstance(batch, list):
        raise ValueError(
            "the sweep lacks a candidates array; it must report what it timed"
        )
    # A sweep holds the whole catalog in one process, so there is no continuation to
    # follow -- but the count is still checked, because a page limit silently truncating
    # the catalog would train a model on a subset and call it complete.
    if measured.get("next_offset") is not None or len(batch) != total:
        raise ValueError(
            "the sweep did not time the whole catalog; refusing silent truncation"
        )
    for candidate in batch:
        candidate_id = candidate.get("id")
        knobs = _knob_tuple(candidate)
        if not candidate_id or candidate_id in seen_ids or knobs in seen_tuples:
            raise ValueError(
                "candidate identities and complete enrolled knob tuples must be unique"
            )
        seen_ids.add(candidate_id)
        seen_tuples.add(knobs)
        _feature_map(candidate, "kernel_features")
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"no matched candidates for graph {measured.get('graph_id')}")
    # What each pinned integer addressed, learned from the engine's own answer. Accumulated
    # across graphs because one graph's catalog shows only the values ITS candidates carry.
    if addressing_table is not None:
        addressing.observe(candidates, addressing_table)
    rows = []
    published = set(_feature_map(first, "problem_features")) | set(
        _feature_map(first, "device_features")
    )
    # The trade one process per graph accepts: a kernel that CRASHES takes the whole
    # graph's rows with it rather than its own row. A kernel that merely fails to build or
    # run does not -- autotune reports it as an unsucceeded result
    # (makeCompileFailedResult and its siblings), so it still reaches the corpus as the
    # failure it is.
    results = measured.get("results")
    if not isinstance(results, list):
        raise ValueError("sweep response lacks a results array")
    by_id = {}
    for result in results:
        identity = result.get("candidate_id")
        if identity in by_id:
            raise ValueError("one enrolled tuple must time exactly one candidate")
        by_id[identity] = result
    # A bijection, checked both ways. Enumeration decided the candidate set, so a sweep that
    # times a subset has silently dropped rows the corpus would then be missing without
    # saying so, and one that times something else was not the catalog that was enumerated.
    if set(by_id) != {candidate["id"] for candidate in candidates}:
        raise ValueError("the sweep did not time exactly the catalog it reported")
    for candidate in candidates:
        result = by_id[candidate["id"]]
        if _knob_tuple(result) != _knob_tuple(candidate):
            raise ValueError("timed knobs did not resolve to the enrolled candidate")
        if _feature_map(result, "kernel_features") != _feature_map(
            candidate, "kernel_features"
        ):
            raise ValueError("timing kernel metadata differs from enrolled candidate")
        if not isinstance(result.get("is_valid"), bool):
            raise ValueError(
                "timing response must preserve the benchmark's is_valid verdict"
            )
        # RFC 0019 §13.2: "A timing is only a training label once the candidate is known
        # correct ... and records the verdict on the row." Required, not defaulted: a bench
        # that emits no verdict has performed no check, and reading that as valid is exactly
        # the inverted oracle the section exists to prevent. `null` is the honest verdict
        # when the cross-check could decide nothing, and it is spelled differently from
        # `true` precisely so it cannot be mistaken for one.
        if VERDICT not in result or not isinstance(result.get(REASON), str):
            raise ValueError(
                "timing response must carry a numerical-validation verdict "
                "(RFC 0019 §13.2); this benchmark performed no correctness check"
            )
        verdict = numerical_verdict(result[VERDICT])
        elapsed = result.get("robust_time_ms")
        if result.get("succeeded") and (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError(
                "successful timing requires a positive finite robust_time_ms"
            )
        row = {
            "benchmark": first["graph_id"],
            "device": first["device_id"],
            "arch": first["device_arch"].split(":", 1)[0],
            "device_arch": first["device_arch"],
            "engine": first["engine_id"],
            "kernel": candidate["id"],
            "is_valid": result["is_valid"],
            # A second column beside `is_valid`, never folded into it. `is_valid` means
            # "a measurement was obtained" and §8.1 plus `evaluate`'s exclusion counters
            # both read it that way; a row that ran and computed the wrong answer is a
            # different fact from a row that never ran, and §13.2 keeps both.
            VERDICT: verdict,
            REASON: result[REASON],
            "succeeded": result.get("succeeded"),
            "skip_reason": result.get("skip_reason"),
            "robustMeanMs": elapsed,
            "minTimeMs": result.get("min_time_ms"),
            "avgTimeMs": result.get("avg_time_ms"),
            # RFC 0019.13 §8.3 makes `stddevMs` and `iters` columns of the result
            # envelope and §8.5 records the spread "so it can be used, not merely
            # stored": `evaluate`'s tie band keys on exactly these two names and is
            # inert on a corpus that drops them.
            "stddevMs": result.get("stddev_ms"),
            "iters": result.get("iterations"),
            "knob_settings": json.dumps(candidate["knob_settings"], sort_keys=True),
        }
        if verdict is False:
            # §13.2: the row "is written with its measurement suppressed and an explicit
            # invalid marker, so the model learns the failure surface instead of inferring
            # one from absence". Suppressed here, at the one place the corpus row is built,
            # so every consumer of it sees the same thing: `corpus.json`, `corpus.csv`, the
            # `evaluate` regret pass that reads the corpus back, and any later retrain. A
            # wrong-but-fast kernel holds the best time in its group, so leaving the number
            # in place and relying on each consumer to filter is how it becomes the label.
            suppress_timings(row)
        for mapping in (
            first["problem_features"],
            first["device_features"],
            candidate["kernel_features"],
        ):
            collision = set(row) & set(mapping)
            if collision:
                raise ValueError(
                    f"published feature names collide with envelope fields: {sorted(collision)}"
                )
            row.update(mapping)
            published.update(mapping)
        # RFC 0019.13 §8.3 (:1501-1506) derives throughput as `flops / time`, and §8.4
        # names the same quantity as the target. Derived here, where the engine's own
        # published `graph.flops` has just been merged in, so the corpus carries the
        # calibrated column rather than leaving the caller to reconstruct it. The mean,
        # not the robust mean: §11.2 (:2003) pins a calibrated score to `avgTimeMs`.
        work, average = row.get("graph.flops"), row["avgTimeMs"]
        if _finite_positive(work) and _finite_positive(average):
            row["tflops"] = work / (average * 1e9)
        rows.append(row)
    return rows, published


def collect_immediate_graph(
    command: list[str], environment: dict, log_dir: Path, commands: list, *, metric: str
) -> tuple[list[dict], set[str]]:
    """Measure one engine's ordinary no-search selection without inspecting its catalog.

    The engine chooses its kernel at plan build with its ranker for the request's metric
    (RFC 0019 §11.4), so an L1 label for `metric` is a measurement of the selection a
    request for `metric` actually gets -- which is why the metric is part of the request.
    """
    if "--knob" in command or "enumerate" in command:
        raise ValueError("L1 collection cannot pin knobs or enumerate candidates")
    response = _run_json(
        [*command, "--collect-immediate", "--ranking-metric", metric, "--json"],
        environment,
        log_dir,
        len(commands),
        commands,
    )
    if response.get("metric") != metric:
        raise ValueError(
            f"immediate measurement was taken for metric {response.get('metric')!r}, "
            f"not the requested {metric!r}"
        )
    row = normalize_row(response)
    if str(row["engine"]) != command[command.index("--engine-id") + 1]:
        raise ValueError("immediate measurement returned another engine")
    return [row], set(json.loads(row["features"]))


def _catalog_label(
    metric: str, usable: pd.DataFrame, defaulted: bool, role: str
) -> tuple:
    """(declared metric, target, calibrated, timing statistic) for a catalog ranker.

    RFC 0019 §13.4: each metric fixes its label, so the only choice left is whether the
    corpus can supply it.
    """
    label = ranking_metric(metric).label
    if label in usable.columns and bool(
        pd.to_numeric(usable[label], errors="coerce").gt(0).all()
    ):
        # RFC 0019 §11.1 (:1501-1506) gives `sort_kernel_catalog` a cross-engine
        # role, and §11.2's `B only` ranking row exists only for a score that is a
        # comparable absolute quantity. `avgTimeMs` rather than the robust mean -- as the
        # label itself, or as the time `tflops` is derived from -- because §11.2 (:2003)
        # pins a calibrated score to the mean.
        return metric, label, True, LABEL_STATISTIC
    if not defaulted:
        raise ValueError(
            f"--metric {metric} needs a positive {label!r} on every measured candidate, and "
            "this corpus does not carry one (tflops needs the engine to publish graph.flops)"
        )
    # A millisecond score ranks this engine's own catalog just as well and forfeits the
    # cross-engine role -- legal under RFC 0019.13 §2.5 (:122-123) and §15.1 (:2387-2390),
    # but a smaller model than the role is. It declares no metric: it estimates none, and
    # is the engine's metric-less default ranker (RFC 0019 §3.1).
    logger.warning(
        "Not every measured candidate carries a positive graph.flops and avgTimeMs, so "
        "%s is trained on robustMeanMs/min with no score.metric and score.calibrated=false. "
        "That ranks this engine's catalog correctly (RFC 0019.13 §2.5, §15.1) but forfeits "
        "the cross-engine role RFC 0019 §11.1 gives it: the score is not comparable with "
        "another engine's, so §11.2's `B only` ranking row does not apply and the thorough "
        "policy falls back to this engine's L1 prediction instead of its configuration score.",
        role,
    )
    return None, "robustMeanMs", False, "robustMeanMs"


def read_regime_manifest(manifest: Path) -> dict:
    """benchmark -> its regime columns, from one `hipdnn_corpus_gen` manifest.csv.

    The manifest carries `regime` and then one column per facet the operation declares, up to
    `source`. They go onto rows as envelope columns (`regime`, `regime.<facet>`,
    `regime.operation`), never features.
    """
    labels: dict = {}
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if "benchmark" not in columns or "regime" not in columns:
            return labels
        end = (
            columns.index("source")
            if "source" in columns
            else columns.index("regime") + 1
        )
        facets = columns[columns.index("regime") + 1 : end]
        for record in reader:
            labels[record["benchmark"]] = {
                "regime": record["regime"],
                **{f"regime.{facet}": record[facet] for facet in facets},
                # Which declaration the label is under: a quota asks corpus_gen for a regime
                # of one operation, and two operations may share a label.
                **({"regime.operation": record["op"]} if record.get("op") else {}),
            }
    return labels


def corpus_regimes(graph_paths) -> dict:
    """Each graph's regime, as the corpus that generated it labelled it: benchmark -> columns.

    `hipdnn_corpus_gen` writes `manifest.csv` beside `graphs/`. Its labels go onto every
    measured row so `evaluate`'s per-regime table (RFC 0019.13 §11.2) exists for every model
    trained from a collection, and a later sizing run can say which populations a model is
    weak on without finding the corpus again.
    """
    labels: dict = {}
    for directory in sorted(
        {
            parent
            for path in graph_paths
            for parent in (Path(path).parent, Path(path).parent.parent)
        }
    ):
        if (directory / "manifest.csv").is_file():
            labels.update(read_regime_manifest(directory / "manifest.csv"))
    return labels


def _measure(
    args: argparse.Namespace, tree: Path, stage: Path, sources: list, immediate: bool
) -> dict:
    """Run the benchmark over `--graphs` into `stage`: the measuring half of generate.

    Returns exactly what training reads, so `--collect-only` can write it down and
    `--collection` can hand the same values back without measuring.
    """
    if args.engine_id is None:
        raise ValueError("--engine-id is required to measure")
    bench = shutil.which(args.bench)
    if bench is None:
        raise ValueError(f"hipdnn_bench executable {args.bench!r} was not found")
    graphs = discover_graphs(args.graphs)
    if args.shard:
        index, count = parse_shard(args.shard)
        # Every N-th in path order, not a contiguous block: a corpus is written regime by
        # regime, so a block would hand one GPU all the decode shapes.
        total = len(graphs)
        graphs = graphs[index::count]
        if not graphs:
            raise ValueError(f"shard {args.shard} of {total} graph(s) is empty")
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    provenance, ued, exposed, kernel_fields, ordinals = None, {}, {}, set(), {}
    if not immediate:
        provenance = snapshot_provenance(tree, args.engine, args.arch)
        ued_path, ued = _descriptor(tree, ".ued.json", provenance["ued"]["id"])
        _, kmd = _descriptor(tree, ".kmd.json", provenance["kmd"]["id"])
        kernel_fields = {"kernel." + field["name"] for field in kmd["fields"]}
    # The tree this run measures is the ONE root the bench loads from: DIR replaces the
    # provider's installed tree, while RUNTIME_DIR and PATH only add roots beside it, and
    # the loader keeps the FIRST definition of an id. Inherited additive roots are
    # cleared for both roles, and the measured tree goes in as the replacement, so an
    # earlier root's selector can never stand in for the one being generated against --
    # an L1 label is a measurement of the selection this tree makes, nothing else's.
    environment = dict(os.environ)
    environment.pop("HIPDNN_DESCRIPTOR_PATH", None)
    environment.pop("HIPDNN_DESCRIPTOR_RUNTIME_DIR", None)
    if immediate:
        environment["HIPDNN_DESCRIPTOR_DIR"] = str(tree)
    else:
        collection_tree = stage / "collection_descriptors"
        shutil.copytree(tree, collection_tree)
        exposed = dict(ued)
        # RFC 0019 13.2: the collection UED exposes EVERY KMD field, so the knob tuple
        # equals the metadata tuple and every catalog entry is individually reachable.
        # Exposing only `int` made two kernels differing in e.g. `dtype` share a tuple
        # and abort the run on the collision at _knob_tuple.
        #
        # Which of those fields ends up addressed BY AN ORDINAL is the engine's
        # decision, not this tool's: a knob value is an int64 end to end, so the
        # ingestor numbers each non-integer field over its own value set and reports
        # the pin it chose on every enumerated candidate. The mapping is read back off
        # the collection below (addressing.observe) rather than re-derived here, so
        # there is no second numbering to disagree with the engine's.
        exposed["knobs"] = [field["name"] for field in kmd["fields"]]
        _write_json(collection_tree / ued_path.relative_to(tree), exposed)
        _write_json(stage / "shipping_ued.json", ued)
        environment["HIPDNN_DESCRIPTOR_DIR"] = str(collection_tree)
    rows = {source: [] for source in sources}
    commands, graph_inputs, failed_graphs = [], [], []
    published = set()
    for graph_index, graph in enumerate(graphs):
        graph_rows = {source: [] for source in sources}
        graph_names = set()
        payload = graph.read_bytes()
        # `hipdnn_bench` tells the two serialized forms apart by content rather than
        # by extension, so a renamed file still loads; the staged copy follows the
        # same rule and keeps whichever form the source was in.
        binary = not payload.lstrip().startswith(b"{")
        saved_graph = (
            stage / "graphs" / f"{graph_index:06d}{'.fb' if binary else '.json'}"
        )
        saved_graph.parent.mkdir(exist_ok=True)
        graph_input = {
            "source": str(graph),
            "copy": str(saved_graph.relative_to(stage)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        graph_inputs.append(graph_input)
        # One graph's failure costs that graph, not the run: a bench that crashes or a
        # response that fails validation on one problem says nothing about the others.
        # All of the graph's rows go together, for every device and metric, so the
        # metrics' corpora stay over one problem set. The budget below still ends a run
        # whose failures are systematic rather than incidental.
        try:
            if immediate and not binary:
                graph_document = json.loads(payload.decode("utf-8"))
                if not isinstance(graph_document, dict):
                    raise ValueError("graph input must be a JSON object")
                if not graph_document.get("id"):
                    canonical = json.dumps(
                        graph_document,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    graph_document["id"] = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, "hipdnn:graph:" + canonical)
                    )
                _write_json(saved_graph, graph_document)
            else:
                # A serialized graph already carries its own id, and the bench preserves
                # it across the deserialize/serialize round trip it does for L1, so there
                # is nothing to inject: the identity the corpus records is the one the
                # benchmark reports back as `graph_id`, keyed to this copy's sha256.
                saved_graph.write_bytes(payload)
            command = [
                bench,
                "--graph",
                str(saved_graph),
                "--engine-id",
                str(args.engine_id),
            ]
            if args.plugin_dir:
                command.extend(["--plugin-dir", str(Path(args.plugin_dir).resolve())])
            if args.workspace_limit is not None:
                command.extend(["--workspace-limit", str(args.workspace_limit)])
            for knob in args.knob:
                command.extend(["--knob", knob])
            for device in args.device or [environment.get("HIP_VISIBLE_DEVICES")]:
                run_env = dict(environment)
                if device is not None:
                    run_env["HIP_VISIBLE_DEVICES"] = device
                for source in sources:
                    if immediate:
                        collected, names = collect_immediate_graph(
                            command,
                            run_env,
                            stage / "commands",
                            commands,
                            metric=source,
                        )
                    else:
                        collected, names = collect_graph(
                            command,
                            run_env,
                            stage / "commands",
                            commands,
                            addressing_table=ordinals,
                            engine_descriptor_id=ued["id"],
                        )
                    graph_rows[source].extend(collected)
                    graph_names.update(names)
        except ValueError as error:
            failed_graphs.append({**graph_input, "error": str(error)})
            logger.warning("graph %s skipped: %s", graph, error)
            continue
        if immediate and not args.collect_only and not any(rows.values()):
            # Settled on the first measured graph, not after the corpus: the id an opaque
            # engine reads is its declaration and does not vary by graph, so a run that could
            # only train an unread model stops here. A collect-only run trains nothing.
            requested = _requested_uhd_ids(args.uhd_ids, list(sources))
            for source in sources:
                if graph_rows[source]:
                    _declared_uhd_id(
                        json.loads(graph_rows[source][0]["binding"]),
                        source,
                        requested.get(source),
                    )
        for source in sources:
            rows[source].extend(graph_rows[source])
        published.update(graph_names)
    if len(failed_graphs) > args.max_graph_failures * len(graphs):
        _write_json(stage / "failed_graphs.json", failed_graphs)
        listed = "\n".join(
            f"  {failure['source']}: {failure['error']}"
            for failure in failed_graphs[:10]
        )
        more = (
            f"\n  ... and {len(failed_graphs) - 10} more"
            if len(failed_graphs) > 10
            else ""
        )
        raise ValueError(
            f"{len(failed_graphs)} of {len(graphs)} graph(s) failed collection, over "
            f"the --max-graph-failures budget of {args.max_graph_failures:g}:\n"
            f"{listed}{more}"
        )
    regimes = corpus_regimes(graphs)
    for source in sources:
        for row in rows[source]:
            row.update(regimes.get(str(row["benchmark"]), {}))
    return {
        "collected_at": collected_at,
        "rows": rows,
        "published": published,
        "commands": commands,
        "graph_inputs": graph_inputs,
        "provenance": provenance,
        "kernel_fields": kernel_fields,
        "knob_encodings": addressing.as_manifest(ordinals),
        "shipping_knobs": ued.get("knobs", []),
        "collection_knobs": exposed.get("knobs", []),
        "failed_graphs": failed_graphs,
    }


def discover_graphs(supplied: list[str]) -> list[Path]:
    """The graph files `--graphs` names, sorted and each once.

    A `hipdnn_corpus_gen` root is read through its `manifest.json` graph list rather than
    searched: the manifest sits beside the graphs, is JSON, and is not one of them (T6 --
    collected as a graph, it aborted the run). A listed graph that is absent is an error,
    because a silently shorter corpus is not the one the manifest describes.
    """
    graphs = set()
    for text in supplied:
        path = Path(text).resolve()
        manifest = (
            path / MANIFEST
            if path.is_dir()
            else path if path.name == MANIFEST else None
        )
        if manifest is not None and manifest.is_file():
            graphs.update(_manifest_graphs(manifest))
        elif path.is_dir():
            # `hipdnn_corpus_gen` writes its problems as binary FlatBuffers under
            # `graphs/<operation>_<n>.fb`, so a generated corpus composes with `generate`
            # only if that form is collected alongside hand-written JSON. A nested corpus
            # root's manifest is not a graph either.
            graphs.update(
                found
                for found in [*path.rglob("*.json"), *path.rglob("*.fb")]
                if found.name != MANIFEST
            )
        else:
            graphs.add(path)
    if not graphs or any(not path.is_file() for path in graphs):
        raise ValueError("--graphs must identify existing graph .json or .fb files")
    return sorted(graphs)


def _manifest_graphs(manifest: Path) -> list[Path]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    rows = document.get("graphs") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest}: expected a nonempty graphs list")
    graphs = []
    for row in rows:
        named = row.get("file") if isinstance(row, dict) else None
        if not isinstance(named, str) or not named:
            raise ValueError(f"{manifest}: every graphs entry must name its file")
        graph = (manifest.parent / named).resolve()
        if not graph.is_file():
            raise ValueError(f"{manifest} lists {named}, which does not exist")
        graphs.append(graph)
    return graphs


def _requested_uhd_ids(values: list[str], metrics: list[str]) -> dict[str, str]:
    """`--uhd-id` as requested metric -> UUID for this run's metrics."""
    from .promote import parse_uhd_ids

    ids = parse_uhd_ids(values)
    if None in ids:
        if len(metrics) != 1:
            raise ValueError(
                "a bare --uhd-id names one UHD, and this run emits one per metric; "
                "name each as METRIC=UUID"
            )
        return {metrics[0]: ids[None]}
    unrequested = sorted(set(ids) - set(metrics))
    if unrequested:
        raise ValueError(
            f"--uhd-id names {', '.join(unrequested)}, which --metric does not request"
        )
    return ids


def _declared_uhd_id(binding: dict, metric: str, requested: str | None) -> str | None:
    """The id an L1 model for `metric` must carry, or None to mint one.

    An engine with no UED binds its models by UUIDs its provider declares per metric, and
    its description reports the one for the requested metric as `binding.uhd_id`. Training
    under any other id ships a model the engine never reads, so the declaration is the
    default, a contradicting --uhd-id is refused, and with neither the run stops here --
    after one graph, not after the whole corpus -- rather than minting an unread id.
    """
    declared = binding.get("uhd_id")
    if "ued" in binding["trained_against"]:
        # A UED role map binds whatever id promotion writes into it.
        return requested
    if declared is not None:
        declared = descriptor_id(declared, "binding.uhd_id")
        if requested is not None and requested != declared:
            raise ValueError(
                f"--uhd-id {metric}={requested} contradicts the id "
                f"{binding['engine']} declares for {metric} ({declared})"
            )
        return declared
    if requested is None:
        raise ValueError(
            f"{binding['engine']} owns no UED, so it reads only the UHD ids its provider "
            f"declares per metric, and its description reports none for {metric}. Pass "
            f"--uhd-id {metric}=<uuid> naming the id the provider declares; a minted id would "
            "install a model the engine never reads"
        )
    return requested


def withheld_kernel_fields(
    kmd_fields: list[str], knobs: list[str], published
) -> list[dict]:
    """The KMD fields generation never offers as `$kernel.*` features, each with why.

    The collection UED exposes every KMD field (RFC 0019 §13.2) so every catalog entry is
    reachable while timing; the model, though, ships under the AUTHORED UED, and the runtime
    admits it only if each `$kernel.*` axis is one of that UED's knobs. A field the matcher
    binds from the graph is no loss: its value is on the problem side under the twin column
    named here, which is where the model reads it.
    """
    twins = graph_bound_twins(published)
    withheld = []
    for field in kmd_fields:
        if field in knobs:
            continue
        if field in twins:
            withheld.append(
                {
                    "field": field,
                    "reason": "graph_bound",
                    "read_instead": twins[field],
                    "detail": "the matcher binds it from the graph, so the problem column "
                    "carries the same value",
                }
            )
        else:
            withheld.append(
                {
                    "field": field,
                    "reason": "not_a_shipping_knob",
                    "detail": "the shipping UED does not expose it, so the runtime would "
                    "refuse a model ranking on it",
                }
            )
    return withheld


def feature_recipe(
    train_frame: pd.DataFrame,
    published: set[str],
    authored: list | None,
    kmd_fields: list[str] | None,
    knobs: list[str],
    pairs: list[tuple[str, str]],
) -> tuple[list, list]:
    """(signature, omitted proposals): the authored recipe, or one proposed from the corpus.

    `kmd_fields` is None for an engine-level run, which reads no kernel fields at all. For a
    catalog run the proposal offers `$kernel.*` only for the shipping UED's knobs, and an
    authored recipe reading any other kernel field is refused here, before training spends
    hours on a model the runtime would not use.
    """
    omitted = []
    if authored is not None:
        signature = authored
    else:
        offered = {"kernel." + field for field in kmd_fields or () if field in knobs}
        scalar_columns = [
            name
            for name in sorted(published)
            if train_frame[name].notna().all()
            and train_frame[name]
            .map(lambda value: isinstance(value, (str, int, float, bool)))
            .all()
        ]
        legal_kernel_fields = {
            name for name in scalar_columns if name.split("[", 1)[0] in offered
        }
        signature, omitted = propose_features(
            train_frame[scalar_columns], legal_kernel_fields, pairs
        )
    if not isinstance(signature, list) or not signature:
        raise ValueError("the feature recipe must be a nonempty canonical array")
    if kmd_fields is not None:
        require_admissible_kernel_axes(
            signature, knobs, kmd_fields, "the feature recipe"
        )
    return signature, omitted


def run_generate(args: argparse.Namespace) -> int:
    from .__main__ import main
    from .promote import PromoteError, build_plan, run_promote, add_promote_arguments

    stage = None
    immediate = args.role == ROLE
    if args.workspace_limit is not None and (not immediate or args.workspace_limit < 0):
        logger.error("--workspace-limit requires %s and a nonnegative byte count", ROLE)
        return 1
    # One timing run, one UHD per metric (RFC 0019 §13.4). Order is kept so the first
    # metric named is the one whose collection proposes the shared feature recipe.
    metrics = list(dict.fromkeys(args.metric or [DEFAULT_RANKING_METRIC]))
    single = len(metrics) == 1
    try:
        uhd_ids = _requested_uhd_ids(args.uhd_ids, metrics)
        if not 0 <= args.max_graph_failures < 1:
            raise ValueError("--max-graph-failures must be a fraction in [0, 1)")
        output = Path(args.output_dir).resolve()
        tree = Path(args.descriptor_tree).resolve()
        if output.exists():
            raise ValueError(
                "--output-dir must not exist; generation never overwrites a previous run"
            )
        if tree == output or tree in output.parents:
            raise ValueError(
                "--output-dir must be outside the shipping descriptor tree"
            )
        if args.eval_benchmarks and args.recall:
            raise ValueError(
                "--eval-benchmarks holds graphs out; --recall trains on every one of them"
            )
        if (
            not args.recall
            and not args.eval_benchmarks
            and not 0 < args.eval_fraction < 1
        ):
            raise ValueError(
                "generate requires a true problem holdout: 0 < --eval-fraction < 1 "
                "(or --recall for an engine whose shape space is closed)"
            )
        if args.shard and not args.collect_only:
            raise ValueError("--shard slices a collection; it needs --collect-only")
        if args.collect_only and args.collection:
            raise ValueError(
                "--collect-only measures --graphs; it cannot also read --collection"
            )
        if args.collection and (
            args.knob or args.device or args.workspace_limit is not None
        ):
            raise ValueError(
                "--knob, --device and --workspace-limit shape a measurement, and a "
                "--collection run measures nothing"
            )
        if immediate:
            if args.knob or args.dim_tile:
                raise ValueError(
                    "L1 generation cannot use kernel knobs or dimension/tile candidate features"
                )
            if not tree.is_dir():
                raise ValueError(
                    "--descriptor-tree must be an existing descriptor root (it may be empty)"
                )
        # The catalog sweep times every candidate once, whatever the metric: one timing
        # run feeds every metric's label (RFC 0019 §13.4). An immediate run measures the
        # engine's own kernel choice, which follows the requested metric, so each metric
        # is its own measurement.
        sources = metrics if immediate else [None]
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".uhd-generate-", dir=output.parent))
        superseded, collections = 0, None
        if args.graphs:
            measured = _measure(args, tree, stage, sources, immediate)
            engine_id = args.engine_id
            if args.collect_only:
                manifest = write_collection(
                    stage,
                    collected_at=measured["collected_at"],
                    role=args.role,
                    engine=args.engine,
                    engine_id=engine_id,
                    sources=sources,
                    rows=measured["rows"],
                    published=measured["published"],
                    commands=measured["commands"],
                    graph_inputs=measured["graph_inputs"],
                    provenance=measured["provenance"],
                    knob_encodings=measured["knob_encodings"],
                    shipping_knobs=measured["shipping_knobs"],
                    collection_knobs=measured["collection_knobs"],
                    kernel_fields=measured["kernel_fields"],
                    shard=args.shard,
                    failed_graphs=measured["failed_graphs"],
                )
                stage.rename(output)
                stage = None
                print(
                    f"Collected {sum(manifest['row_counts'].values())} measurement(s) of "
                    f"{manifest['graph_count']} graph(s): {output}"
                )
                return 0
            graph_inputs = measured["graph_inputs"]
            for source in sources:
                measured["rows"][source], dropped = one_measurement_per_shape(
                    [(str(row["device"]), row) for row in measured["rows"][source]]
                )
                superseded += dropped
        else:
            # Measuring nothing: the rows, and everything training reads beside them, come
            # back from the collections exactly as a measuring run would have produced them.
            measured = load_collections(
                args.collection, role=args.role, sources=sources
            )
            engine_id = (
                args.engine_id if args.engine_id is not None else measured["engine_id"]
            )
            graph_inputs = measured["graphs"]
            superseded, collections = (
                measured["superseded_rows"],
                measured["collections"],
            )
            logger.info(
                "training from %d collection(s); %d row(s) superseded by a newer "
                "measurement of the same configuration on the same shape",
                len(collections),
                superseded,
            )
        rows, published, commands = (
            measured["rows"],
            measured["published"],
            measured["commands"],
        )
        provenance, kernel_fields = measured["provenance"], measured["kernel_fields"]
        knob_encodings = measured["knob_encodings"]
        shipping_knobs, collection_knobs = (
            measured["shipping_knobs"],
            measured["collection_knobs"],
        )
        # What the shipping UED exposes and the KMD declares, as the measurement recorded them:
        # the catalog admission rule (feature_recipe) checks the recipe against both.
        knobs = list(shipping_knobs)
        kmd_fields = (
            None
            if immediate
            else sorted(f.removeprefix("kernel.") for f in kernel_fields)
        )
        failed_graphs = measured.get("failed_graphs", [])
        if immediate:
            for source in sources:
                if rows[source]:
                    # The engine's declaration, read off its own response: it does not vary by
                    # graph, so the first row of each metric settles it, measured or collected.
                    uhd_ids[source] = _declared_uhd_id(
                        json.loads(rows[source][0]["binding"]),
                        source,
                        uhd_ids.get(source),
                    )

        # One corpus per source, named plainly when there is only one.
        def staged(stem: str, source) -> str:
            return stem if len(sources) == 1 else f"{stem}_{source}"

        frames, corpora = {}, {}
        for source in sources:
            frame = pd.DataFrame(rows[source])
            if immediate:
                frame, binding = training_binding(normalize_corpus(frame))
                if provenance is not None and binding["trained_against"] != provenance:
                    raise ValueError(
                        "the engine's descriptor provenance changed between metric collections"
                    )
                provenance = binding["trained_against"]
                if args.engine and args.engine not in (
                    binding["engine"],
                    provenance.get("ued", {}).get("id"),
                ):
                    raise ValueError(
                        "--engine does not match the collected engine binding"
                    )
            elif frame.duplicated(["benchmark", "device", "kernel"]).any():
                raise ValueError(
                    "the graph/device corpus contains duplicate candidate measurements"
                )
            frame.to_csv(stage / f"{staged('corpus', source)}.csv", index=False)
            corpora[source] = stage / f"{staged('corpus', source)}.json"
            _write_json(corpora[source], _absent_as_null(rows[source]))
            frames[source] = frame
        _write_json(stage / "provenance.json", provenance)
        # Three conditions, because they are three different facts about a candidate and
        # §13.2 keeps them apart: `succeeded` says the engine ran it, `is_valid` says a
        # measurement came back, and `numerically_valid is not False` says nothing showed
        # the result to be wrong. The last one is the label gate -- a wrong-but-fast kernel
        # holds the best time in its group, so admitting it trains the ranker to prefer it.
        # `ne(False)` rather than `eq(True)`: an undecidable verdict is null, and null is
        # the pre-existing state of every corpus collected before there was a reference to
        # check against (Open Question 19). Gating on it would train on nothing at all.
        # The row itself is not dropped -- it is already in `corpus.json`/`corpus.csv` above,
        # with its measurement suppressed and its marker, which is what §13.2 asks for.
        # The verdict gates both roles: an engine whose immediate pick computed the wrong
        # answer must not teach the estimator its time either. An immediate row is always
        # `is_valid` (`normalize_row`), so only the verdict gates it.
        usable = {}
        for source, frame in frames.items():
            keep = ~known_wrong(frame)
            if not immediate:
                keep &= frame["is_valid"] & frame["succeeded"].eq(True)
            usable[source] = frame[keep].copy()
        if any(candidates.empty for candidates in usable.values()):
            raise ValueError("the benchmark produced no successful valid timings")
        # Checked here, the first moment it is knowable, rather than at the
        # `problems_scored` gate below. Without this the run trains a model and evaluates
        # it before dying on "held-out corpus has no evaluable candidate ranking" -- a true
        # statement that names the symptom and not the cause, and whose documented remedy
        # ("give the catalog a knob its matcher does not pin") is right for an engine whose
        # variants were collapsed by an over-broad match and wrong for one whose kernel
        # choice is a total function of the problem by design. Those two want opposite
        # things, and only the second wants L1. The collected measurements survive into the
        # preserved stage either way, which is the point: they are exactly the labels the
        # L1 run needs, so the sweep is not wasted, only the role was.
        density = None
        if not immediate:
            density = candidate_density(usable[None])
            if density.deterministic:
                raise DeterministicCatalogError(density.diagnosis(args.engine))
            thin = density.near_deterministic_warning()
            if thin:
                logger.warning("%s", thin)
        # (declared metric, target, calibrated, timing statistic, source) per UHD emitted.
        # L1 is always calibrated, and every calibrated label comes from `avgTimeMs`.
        labels = []
        for metric in metrics:
            if immediate:
                labels.append(
                    (
                        metric,
                        ranking_metric(metric).label,
                        True,
                        LABEL_STATISTIC,
                        metric,
                    )
                )
            else:
                labels.append(
                    (
                        *_catalog_label(
                            metric, usable[None], args.metric is None, args.role
                        ),
                        None,
                    )
                )
        grouping = resolve_grouping(frames[sources[0]])
        held = _eval_benchmarks(args.eval_benchmarks) if args.eval_benchmarks else None
        eval_corpora = {}
        if held is not None:
            measured = set(frames[sources[0]][BENCHMARK_COLUMN].astype(str)) & held
            if not measured:
                raise ValueError(
                    f"none of the {len(held)} --eval-benchmarks graphs was measured in this "
                    "corpus; there is nothing to evaluate on"
                )
            # The graphs scored are exactly the ones named, every row of them on every device:
            # evaluate scores the whole of this file rather than drawing its own split.
            for source in sources:
                eval_corpora[source] = stage / f"{staged('eval_corpus', source)}.json"
                _write_json(
                    eval_corpora[source],
                    _absent_as_null(
                        [
                            row
                            for row in rows[source]
                            if str(row.get(BENCHMARK_COLUMN)) in held
                        ]
                    ),
                )
        train_frames = {}
        for source in sources:
            candidates = usable[source]
            if args.recall:
                # Every shape the engine can ever be asked about is in the corpus: holding some
                # out would ship a model that has not seen shapes it will certainly meet.
                train_frames[source] = candidates
            elif held is not None:
                train_frames[source] = candidates[
                    ~candidates[BENCHMARK_COLUMN].astype(str).isin(held)
                ]
            else:
                split = split_problems(
                    problem_keys(frames[source], grouping),
                    args.eval_fraction,
                    args.seed,
                )
                train_frames[source] = candidates[
                    ~problem_keys(candidates, grouping).isin(split.eval_problems)
                ]
            if len(set(problem_keys(train_frames[source], grouping))) < 5:
                raise ValueError(
                    "generation needs at least five training graph/device groups plus held-out problems"
                )
        # The first source proposes and checks the one feature recipe every metric shares.
        train_frame = train_frames[sources[0]]
        pairs = []
        for pair in args.dim_tile:
            parts = pair.split("=", 1)
            if len(parts) != 2:
                raise ValueError("--dim-tile requires DIMENSION=KERNEL_FIELD")
            pairs.append(tuple(part.removeprefix("$") for part in parts))
        if args.feature_signature:
            authored = json.loads(
                Path(args.feature_signature).read_text(encoding="utf-8")
            )
        elif args.features:
            authored = build_features_signature(args.features)
        else:
            authored = None
        signature, omitted = feature_recipe(
            train_frame, published, authored, kmd_fields, knobs, pairs
        )
        withheld = withheld_kernel_fields(kmd_fields or [], knobs, published)
        if immediate:
            # Leakage only; whether every entry evaluates on each graph's published features
            # is the shared evaluator's answer, which training asks once the encoding exists.
            validate_signature(signature)
        unknown = {ref[1:] for ref in signature_references(signature)} - published
        if unknown:
            raise ValueError(
                f"features are not published by this engine: {sorted(unknown)}"
            )
        coverage = device_field_coverage(train_frame)
        enforce_device_coverage(signature, coverage)
        arches = sorted(usable[sources[0]]["arch"].unique())
        if args.arch not in (None, "default") and args.arch not in arches:
            raise ValueError(
                "promotion arch is absent from the observed device architectures"
            )
        if args.arch is None and len(arches) != 1:
            raise ValueError(
                "multiple observed architectures require an explicit --arch promotion target"
            )
        _write_json(stage / "features.json", signature)
        models = []
        # Each directory is named for the metric requested, which a metric-less fallback
        # still answers to.
        for requested, (declared, target, calibrated, statistic, source) in zip(
            metrics, labels
        ):
            model_dir = stage / ("model" if single else f"model_{requested}")
            source_train = train_frames[source]
            train_path = stage / f"{staged('train', source)}.json"
            source_train.to_csv(stage / f"{staged('train', source)}.csv", index=False)
            _write_json(
                train_path, _absent_as_null(source_train.to_dict(orient="records"))
            )
            train_args = [
                "train",
                "--input",
                str(train_path),
                "--feature-signature",
                str(stage / "features.json"),
                "--provenance",
                str(stage / "provenance.json"),
                *(
                    ["--metric", declared]
                    if declared
                    else ["--target", target, "--objective", "min"]
                ),
                "--timing-statistic",
                statistic,
                "--role",
                args.role,
                "--group-by",
                *grouping.columns,
                "--output-dir",
                str(model_dir),
                "--name",
                args.name if single else f"{args.name} ({requested})",
                "--num-boost-round",
                str(args.num_boost_round),
                "--early-stopping",
                str(args.early_stopping),
                "--training-arches",
                *arches,
            ]
            if calibrated:
                train_args.append("--calibrated")
            if immediate:
                train_args.extend(["--arch", args.arch or arches[0]])
            if uhd_ids.get(requested):
                train_args.extend(["--uhd-id", uhd_ids[requested]])
            if args.feature_evaluator:
                train_args.extend(["--feature-evaluator", args.feature_evaluator])
            if main(train_args):
                raise ValueError(
                    f"training {requested} failed; no generated model was published"
                )
            # Recall scores every shape, all of them trained on: the question is how well the
            # model reproduces the space it will serve, not how it extrapolates.
            eval_args = [
                "evaluate",
                "--input",
                str(eval_corpora.get(source, corpora[source])),
                "--model-dir",
                str(model_dir),
                "--eval-fraction",
                "1.0" if args.recall or held is not None else str(args.eval_fraction),
                "--seed",
                str(args.seed),
                "--include-per-problem",
            ]
            if args.feature_evaluator:
                eval_args.extend(["--feature-evaluator", args.feature_evaluator])
            if main(eval_args):
                raise ValueError(
                    f"{requested} artifact evaluation failed; no generated model was published"
                )
            report_path = model_dir / "eval_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not report["metrics"]["problems_scored"]:
                # The density check above already refused a wholly deterministic catalog, so
                # reaching here that way means the holdout alone came out single-candidate --
                # a thin corpus rather than an inert engine. Say which, because the remedies
                # differ and this message has historically been read as the other one.
                if report["metrics"].get("deterministic_catalog"):
                    raise DeterministicCatalogError(
                        "every held-out problem has a single candidate, though the corpus as a "
                        "whole does not: the evaluation slice landed entirely on problems with "
                        "nothing to rank. Collect more contested problems, or -- if this engine "
                        f"pins its kernel choice by design -- train --role {ROLE}."
                    )
                raise ValueError(
                    "held-out corpus has no evaluable immediate predictions"
                    if immediate
                    else "held-out corpus has no evaluable candidate ranking"
                )
            evaluated_keys = {
                tuple(key) for key in report["split"]["eval_problem_keys"]
            }
            training_keys = set(problem_keys(source_train, grouping))
            # A held-out score must be held out; a recall score is by definition over trained shapes.
            if not args.recall and training_keys & evaluated_keys:
                raise ValueError("evaluation includes a problem seen during training")
            report["holdout_integrity"] = {
                "status": "held_out",
                "detail": "Verified disjoint graph/device identities in recorded training and evaluation slices",
            }
            _write_json(report_path, report)
            models.append(
                {
                    "metric": declared,
                    "requested_metric": requested,
                    "uhd_id": uhd_ids.get(requested),
                    "model_dir": str(model_dir),
                    "corpus": str(corpora[source]),
                    "training_arguments": train_args,
                    "evaluation_arguments": eval_args,
                    "training_problem_keys": sorted(training_keys),
                    "eval_problem_keys": sorted(evaluated_keys),
                }
            )
        _write_json(
            stage / "generation_manifest.json",
            {
                "schema": "uhd_gen.generation/2",
                "trained_against": provenance,
                "graphs": graph_inputs,
                # Where the measurements came from when this run measured nothing, and how many
                # rows were set aside so each configuration on a shape is trained on once.
                "collections": collections,
                "superseded_rows": superseded,
                # Graphs whose collection failed and were skipped, each with its error, within
                # the --max-graph-failures budget; their staged copies stay under graphs/.
                "failed_graphs": failed_graphs,
                "max_graph_failures": args.max_graph_failures,
                "commands": commands,
                "features_signature": signature,
                "omitted_proposals": omitted,
                # KMD fields never offered as `$kernel.*` features, and why: the runtime admits only
                # the shipping UED's knobs, and a graph-bound field is read from its problem twin.
                "withheld_kernel_fields": withheld,
                # One entry per UHD emitted: its metric (null for a metric-less ranker), where it
                # was trained, and the exact commands that trained and evaluated it.
                "models": models,
                "device_coverage": coverage,
                # How much of this corpus the ranker could actually learn from. A run that
                # reaches here had *some* contested problems, but "some" spans a model fitted
                # on every problem and one fitted on four of them, and the metrics beside it
                # report only the second without saying so.
                "catalog_density": density.as_dict() if density else None,
                "seed": args.seed,
                "eval_fraction": (
                    None if args.recall or held is not None else args.eval_fraction
                ),
                # "recall": trained on every shape and scored on all of them (a closed shape
                # space); "holdout": scored on a drawn fraction of shapes the model never saw;
                # "eval_set": scored on the named graphs, fixed when their corpus was made.
                "evaluation": (
                    "recall"
                    if args.recall
                    else ("eval_set" if held is not None else "holdout")
                ),
                "eval_benchmarks": (
                    None
                    if held is None
                    else {
                        "path": str(Path(args.eval_benchmarks).resolve()),
                        "listed": len(held),
                        "measured": len(
                            set(frames[sources[0]][BENCHMARK_COLUMN].astype(str)) & held
                        ),
                    }
                ),
                "shipping_knobs": shipping_knobs,
                "collection_knobs": collection_knobs,
                # What each knob's pinned integer addressed, as the engine reported it on the
                # candidates this corpus enumerated. Recorded for reading, not for use: the
                # runtime derives its own numbering, and an ordinal in a stored row is
                # unreadable without knowing which value it named.
                "knob_encodings": knob_encodings,
                "engine_id": engine_id,
                "training_arches": arches,
                "promotion_role": args.role,
                "promotion_arch": args.arch or arches[0],
            },
        )
        # Validate installation against the original tree before publishing any artifacts.
        for model in models:
            build_plan(
                Path(model["model_dir"]),
                tree,
                args.engine,
                role=args.role,
                arch=args.arch or arches[0],
                corpus=Path(model["corpus"]),
                uhd_ids={None: model["uhd_id"]} if model["uhd_id"] else None,
                feature_evaluator=args.feature_evaluator,
            )
        # Where each model and its corpus land once the stage is renamed into place.
        published_models = [
            (
                output / Path(model["model_dir"]).relative_to(stage),
                output / Path(model["corpus"]).relative_to(stage),
                model["uhd_id"],
            )
            for model in models
        ]
        # Recorded paths must refer to the final output rather than the staging directory.
        old_root = str(stage)
        for path in stage.rglob("*.json"):
            if (
                "collection_descriptors" in path.relative_to(stage).parts
                or "graphs" in path.relative_to(stage).parts
            ):
                continue
            document = json.loads(path.read_text(encoding="utf-8"))

            def relocate(value):
                if isinstance(value, str):
                    return (
                        str(output) + value[len(old_root) :]
                        if value.startswith(old_root)
                        else value
                    )
                if isinstance(value, list):
                    return [relocate(item) for item in value]
                if isinstance(value, dict):
                    return {key: relocate(item) for key, item in value.items()}
                return value

            _write_json(path, relocate(document))
        stage.rename(output)
        stage = None
        if not args.no_promote:
            parser = argparse.ArgumentParser()
            add_promote_arguments(parser)
            # In sequence, each against the role map the previous one wrote: promotion adds a
            # UHD beside the other metrics' and replaces only its own metric's.
            for model_dir, corpus, identity in published_models:
                promote_args = [
                    "--model-dir",
                    str(model_dir),
                    "--descriptor-tree",
                    str(tree),
                    "--role",
                    args.role,
                    "--arch",
                    args.arch or arches[0],
                    "--corpus",
                    str(corpus),
                ]
                if args.engine:
                    promote_args.extend(["--engine", args.engine])
                if identity:
                    promote_args.extend(["--uhd-id", identity])
                if args.feature_evaluator:
                    promote_args.extend(["--feature-evaluator", args.feature_evaluator])
                if run_promote(parser.parse_args(promote_args)):
                    raise ValueError(
                        f"promotion failed; validated model and reproducible collection remain at {output}"
                    )
        for model_dir, _, _ in published_models:
            print(
                f"Generated {'installable' if args.no_promote else 'installed'} UHD: {model_dir}"
            )
        return 0
    except (OSError, TypeError, ValueError, KeyError, PromoteError) as error:
        # The stage holds hours of benchmarking -- the collected corpus, the captured
        # command output, and by this point often a trained model too. RFC 0019.13 §8.7:
        # "measurements outlive the strategy that requested them", so a failure anywhere
        # after collection reports where they are instead of deleting them. Every path
        # that raises after `collect_graph` reaches here, including the empty-holdout
        # check, which fires before the stage is renamed into place. It is removed only
        # by the successful rename below, so nothing is left behind by a run that worked.
        if stage is not None and stage.exists():
            logger.error(
                "%s; the collected corpus and any trained model are preserved at %s "
                "(delete it once you no longer need the measurements)",
                error,
                stage,
            )
        else:
            logger.error("%s", error)
        return 1
