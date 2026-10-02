# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Canonical inline UHD features and the shared runtime evaluator protocol."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAX_SAFE_NUMERIC_LITERAL = 1e15


def feature_reference(column: str) -> str:
    """Use published names verbatim, adding only the reference marker."""
    if not isinstance(column, str) or not column or column.startswith("$"):
        raise ValueError(
            f"feature column must be a full published name without '$': {column!r}"
        )
    if any(char.isspace() for char in column):
        raise ValueError(f"feature column contains whitespace: {column!r}")
    return "$" + column


def build_features_signature(feature_cols: list[str]) -> list[str]:
    return [feature_reference(column) for column in feature_cols]


def parse_signature_entry(entry):
    """Only canonical references and inline AST objects are descriptor entries."""
    if isinstance(entry, str) and entry.startswith("$") and len(entry) > 1:
        feature_reference(entry[1:])
        return entry
    if isinstance(entry, dict) and len(entry) == 1:
        _validate_numeric_literals(entry)
        return entry
    raise ValueError(
        "features_signature entries must be bare $references or inline expression objects"
    )


def signature_references(signature: list) -> list[str]:
    """Collect reference leaves without implementing any expression semantics."""
    references = []
    seen = set()

    def visit(node):
        if isinstance(node, str) and node.startswith("$"):
            feature_reference(node[1:])
            if node not in seen:
                seen.add(node)
                references.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for entry in signature:
        visit(parse_signature_entry(entry))
    return references


_KERNEL_REFERENCE = "$kernel."
# The kernel's declared priority, bound for every candidate: not a KMD field or knob
# (RFC 0019 §6.1).
_KERNEL_PRIORITY = "$kernel.priority"


def kernel_axes(signature: list) -> set[str]:
    """The KMD fields a signature reads through `$kernel.*`, by base name.

    `$kernel.tile[0]` reads `tile`; nested references count; `$kernel.priority` reads no
    KMD field. The single definition of the axes RFC 0019 §6.3 check 2 admits a model on,
    mirroring the runtime's `FeatureExtractor::kernelFieldOf`.
    """
    return {
        reference[len(_KERNEL_REFERENCE) :].split("[", 1)[0]
        for reference in signature_references(signature)
        if reference.startswith(_KERNEL_REFERENCE) and reference != _KERNEL_PRIORITY
    }


def require_admissible_kernel_axes(
    signature: list, knobs, kmd_fields, where: str
) -> None:
    """Refuse a signature the runtime would refuse to rank with (RFC 0019 §6.3 check 2).

    Every `$kernel.*` axis other than `$kernel.priority` must be a KMD field AND a knob of
    the shipping UED (not the collection UED, which exposes every field). Graph-bound
    fields belong in the graph's own column.
    """
    axes = kernel_axes(signature)
    undeclared = sorted(axes - set(kmd_fields))
    unexposed = sorted(axes - set(knobs))
    problems = []
    if undeclared:
        problems.append(
            f"reads [{', '.join(undeclared)}], which the KMD does not declare as fields"
        )
    if unexposed:
        problems.append(
            f"ranks on [{', '.join(unexposed)}], which the UED does not expose as knobs "
            f"[{', '.join(sorted(knobs)) or '<none>'}]"
        )
    if problems:
        raise ValueError(
            f"{where} {'; and '.join(problems)}. The runtime refuses such a model and ranks by "
            "priority, then id (RFC 0019 §6.3 check 2). Read problem-side facts from the graph's "
            "own column, not $kernel.*"
        )


def _validate_numeric_literals(node) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        if abs(node) >= MAX_SAFE_NUMERIC_LITERAL or not math.isfinite(node):
            raise ValueError(
                f"features_signature numeric literal {node!r} must be finite with magnitude below 1e15"
            )
    elif isinstance(node, list):
        for value in node:
            _validate_numeric_literals(value)
    elif isinstance(node, dict):
        for value in node.values():
            _validate_numeric_literals(value)


def compute_features_hash(
    signature: list,
    categorical_encoding: dict | None = None,
    executable: str | Path | None = None,
) -> str:
    """The descriptor's `features_hash`, asked of the evaluator the loader verifies with.

    RFC 0019 §6.3 allows one definition (FeatureExtractor::computeHash), so it is never
    recomputed in Python. No rows are needed: §6.5 hashes only signature and codes.
    """
    digest, _, _ = _run_feature_evaluator(
        signature, categorical_encoding, [], executable
    )
    return digest


def encode_feature_value(reference: str, value) -> float:
    if isinstance(value, (bool, int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"{reference}: feature value must be finite, got {value!r}"
            )
        return numeric
    if isinstance(value, str):
        raise ValueError(
            f"{reference}: {value!r} is a string and no categorical_encoding declares this reference"
        )
    raise TypeError(
        f"{reference}: cannot use {type(value).__name__} as a feature value"
    )


def derive_categorical_encoding(
    df, feature_cols: list[str]
) -> dict[str, dict[str, int]]:
    """Stable per-reference codes, preserving the exact published string values.

    Absent bindings get no code; they reach the evaluator as JSON null.
    """
    encoding = {}
    for column in feature_cols:
        if column not in df.columns:
            continue
        series = df[column]
        if getattr(series.dtype, "kind", "O") in "biuf":
            continue
        values = set()
        other_types = set()
        for value in series:
            if isinstance(value, str):
                values.add(value)
            elif not _is_absent(value):
                other_types.add(type(value).__name__)
        if not values:
            continue
        if other_types:
            raise ValueError(
                f"feature column {column!r} mixes strings with {', '.join(sorted(other_types))}"
            )
        encoding[feature_reference(column)] = {
            value: code for code, value in enumerate(sorted(values))
        }
    return encoding


def _is_absent(value) -> bool:
    """A binding the row does not publish: None, or pandas' NaN/NA filling for a missing key.

    Published values are finite (§8.3), so NaN means absent. Arrays are never absent.
    """
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict)) or getattr(value, "ndim", 0):
        return False
    import pandas as pd

    return bool(pd.isna(value))


EVALUATOR_NAME = "hipdnn_uhd_features"
EVALUATOR_ENV_VAR = "HIPDNN_UHD_FEATURE_EVALUATOR"

#: Evaluator locations under a search root: install/venv `bin`, or in-tree `build/bin`.
_EVALUATOR_RELATIVE_DIRS = ("bin", "build/bin")


def _evaluator_search_roots() -> list[Path]:
    """Roots relative to this package and sys.prefix, so the search survives remounts.

    Four parents reach the checkout root without wandering above it.
    """
    package = Path(__file__).resolve().parent
    return [Path(sys.prefix), *package.parents[:4]]


def resolve_feature_evaluator(executable: str | Path | None = None) -> str:
    """Locate the shared evaluator: explicit path, then environment, then build, then PATH.

    An explicitly requested executable that cannot run is an error, not a fall-through.
    """
    for source, requested in (
        (" (--feature-evaluator)", str(executable) if executable else None),
        (f" ({EVALUATOR_ENV_VAR})", os.environ.get(EVALUATOR_ENV_VAR)),
    ):
        if requested:
            resolved = shutil.which(requested)
            if resolved is None:
                raise ValueError(
                    f"feature evaluator {requested!r}{source} is not a runnable executable"
                )
            return resolved
    searched = []
    for root in _evaluator_search_roots():
        for relative in _EVALUATOR_RELATIVE_DIRS:
            candidate = root / relative
            searched.append(str(candidate))
            resolved = shutil.which(str(candidate / EVALUATOR_NAME))
            if resolved is not None:
                return resolved
    resolved = shutil.which(EVALUATOR_NAME)
    if resolved is None:
        raise ValueError(
            f"{EVALUATOR_NAME} was not found and features_hash has no Python implementation "
            f"to fall back on; set {EVALUATOR_ENV_VAR} to the built executable, pass "
            f"--feature-evaluator, or put it on PATH. Searched: {', '.join(searched)}"
        )
    return resolved


def evaluator_feature_semantics_revision(executable: str | Path | None = None) -> int:
    """The evaluator's feature semantics revision (`FeatureSemantics.hpp`).

    Asked of the binary rather than mirrored in Python, so it cannot drift from C++.
    """
    return _run_feature_evaluator([], None, [], executable)[2]


def _run_feature_evaluator(
    signature: list,
    categorical_encoding: dict | None,
    rows: list,
    executable: str | Path | None,
) -> tuple[str, list[list[float]], int]:
    """The only crossing into FeatureExtractor, which owns both the digest and the values.

    Entries are parsed first only for a readable error; the evaluator does the hashing.
    """
    parsed = [parse_signature_entry(entry) for entry in signature]
    request = {
        "signature": parsed,
        "categorical_encoding": categorical_encoding or {},
        "rows": rows,
    }
    result = subprocess.run(
        [resolve_feature_evaluator(executable)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"{EVALUATOR_NAME} failed ({result.returncode}): {result.stderr.strip()}"
        )
    try:
        response = json.loads(result.stdout)
        digest, values = response["features_hash"], response["values"]
        # An evaluator that omits it predates the field: revision 1 by definition.
        revision = response.get("feature_semantics_revision", 1)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError(
                f"feature_semantics_revision must be an integer >= 1, got {revision!r}"
            )
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("missing features_hash")
        if len(values) != len(rows) or any(len(row) != len(parsed) for row in values):
            raise ValueError("feature matrix shape does not match the request")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for row in values
            for value in row
        ):
            raise ValueError("non-finite or non-numeric feature result")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {EVALUATOR_NAME} response: {error}") from error
    return digest, values, revision


def evaluate_feature_maps(
    rows: list[dict],
    signature: list,
    categorical_encoding: dict | None = None,
    executable: str | Path | None = None,
) -> tuple[str, list[list[float]]]:
    """Feature maps (name -> value, None for absent) through the runtime's FeatureExtractor.

    An unbound bare reference fails the whole batch, as the engine refuses that row.
    """
    digest, values, _ = _run_feature_evaluator(
        signature, categorical_encoding, rows, executable
    )
    return digest, values


def evaluate_feature_rows(
    df,
    signature: list,
    categorical_encoding: dict | None = None,
    executable: str | Path | None = None,
) -> tuple[str, list[list[float]]]:
    """One batch through the exact C++ expression implementation used at runtime.

    Unpublished names go over as JSON null; absence is the evaluator's to interpret.
    """
    columns = [reference[1:] for reference in signature_references(signature)]
    present = [column for column in columns if column in df.columns]
    # to_dict keeps full precision and lists, but yields no records for zero columns.
    records = (
        df[present].to_dict(orient="records")
        if present
        else [{} for _ in range(len(df))]
    )
    rows = [
        {
            column: None if _is_absent(record.get(column)) else record[column]
            for column in columns
        }
        for record in records
    ]
    return evaluate_feature_maps(rows, signature, categorical_encoding, executable)
