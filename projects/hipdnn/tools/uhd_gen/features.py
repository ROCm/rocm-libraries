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
        raise ValueError(f"feature column must be a full published name without '$': {column!r}")
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
    raise ValueError("features_signature entries must be bare $references or inline expression objects")


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


def _validate_numeric_literals(node) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        if abs(node) >= MAX_SAFE_NUMERIC_LITERAL or not math.isfinite(node):
            raise ValueError(f"features_signature numeric literal {node!r} must be finite with magnitude below 1e15")
    elif isinstance(node, list):
        for value in node:
            _validate_numeric_literals(value)
    elif isinstance(node, dict):
        for value in node.values():
            _validate_numeric_literals(value)


def compute_features_hash(signature: list, categorical_encoding: dict | None = None,
                          executable: str | Path | None = None) -> str:
    """The descriptor's `features_hash`, from the routine the loader verifies it with.

    RFC 0019 §6.3 requires generation and verification to share ONE definition of this
    digest, and that definition is FeatureExtractor::computeHash. So the digest is asked
    for rather than recomputed here: the evaluator canonicalises the AST and the
    categorical vocabulary and hashes them itself. Zero rows, because §6.5 folds only the
    signature and the codes into the digest -- no corpus is needed to ask for it.

    This module used to carry a second, pure-Python implementation. It agreed with C++
    only because a test pinned the same literal digest on both sides; a change to either
    canonicalisation would have shipped a descriptor the runtime refuses to load rather
    than failing a test here.
    """
    digest, _ = _run_feature_evaluator(signature, categorical_encoding, [], executable)
    return digest


def encode_feature_value(reference: str, value) -> float:
    if isinstance(value, (bool, int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{reference}: feature value must be finite, got {value!r}")
        return numeric
    if isinstance(value, str):
        raise ValueError(f"{reference}: {value!r} is a string and no categorical_encoding declares this reference")
    raise TypeError(f"{reference}: cannot use {type(value).__name__} as a feature value")


def derive_categorical_encoding(df, feature_cols: list[str]) -> dict[str, dict[str, int]]:
    """Stable per-reference codes, preserving the exact published string values.

    An absent binding (a column no row publishes, or a row that does not publish it) has no
    code: it reaches the evaluator as JSON null, never as a vocabulary entry.
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
            raise ValueError(f"feature column {column!r} mixes strings with {', '.join(sorted(other_types))}")
        encoding[feature_reference(column)] = {
            value: code for code, value in enumerate(sorted(values))
        }
    return encoding


def _is_absent(value) -> bool:
    """A binding the row does not publish: None, or pandas' NaN/NA filling for a missing key.

    Published values are finite (the §8.3 envelope rejects anything else), so a NaN in a
    corpus frame is the hole pandas leaves when rows of different operations publish
    different names -- a conv-fwd row has no `dy`. Arrays are values, never absent.
    """
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict)) or getattr(value, "ndim", 0):
        return False
    import pandas as pd

    return bool(pd.isna(value))


EVALUATOR_NAME = "hipdnn_uhd_features"
EVALUATOR_ENV_VAR = "HIPDNN_UHD_FEATURE_EVALUATOR"

#: Where a build or an install puts the evaluator, relative to a search root: `bin` is
#: CMAKE_INSTALL_BINDIR under an install prefix (and a virtualenv), `build/bin` is an
#: in-tree build beside the sources.
_EVALUATOR_RELATIVE_DIRS = ("bin", "build/bin")


def _evaluator_search_roots() -> list[Path]:
    """Roots derived from where this package sits, never from an absolute path.

    A batch script that named the executable absolutely worked on the login node and
    broke inside the container, where the same tree is mounted at a different root. Every
    root here is relative to this file (or to the interpreter's own prefix), so the search
    moves with the checkout. Four parents reaches `tools/`, `projects/hipdnn/`,
    `projects/` and the checkout root -- far enough for a `build/` beside the sources,
    short enough not to wander into whatever shared directory holds the checkout.
    """
    package = Path(__file__).resolve().parent
    return [Path(sys.prefix), *package.parents[:4]]


def resolve_feature_evaluator(executable: str | Path | None = None) -> str:
    """Locate the shared evaluator: explicit path, then environment, then build, then PATH.

    Every signature needs it now, raw references included: RFC 0019 §6.3 leaves the digest
    with one definition, and that definition is in the binary. Guessing one in Python would
    put a plausible but unverified hash in a shipped descriptor, which fails much later and
    much further away, at load time on a user's machine.

    A name that was supplied and cannot be run is an error rather than a reason to keep
    looking: silently falling through to a different binary than the one asked for is how
    a typo becomes a model stamped by something nobody chose.
    """
    for source, requested in ((" (--feature-evaluator)", str(executable) if executable else None),
                              (f" ({EVALUATOR_ENV_VAR})", os.environ.get(EVALUATOR_ENV_VAR))):
        if requested:
            resolved = shutil.which(requested)
            if resolved is None:
                raise ValueError(f"feature evaluator {requested!r}{source} is not a runnable executable")
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


def _run_feature_evaluator(signature: list, categorical_encoding: dict | None, rows: list,
                           executable: str | Path | None) -> tuple[str, list[list[float]]]:
    """The only crossing into FeatureExtractor, which owns both the digest and the values.

    Entries are parsed before the request is built so an unauthorable signature fails with
    this module's message (which names the offending entry) instead of a subprocess exit
    code. That parse is authoring validation, not a second canonicalisation: the bytes that
    get hashed are the ones the evaluator dumps, never the ones serialised here.
    """
    parsed = [parse_signature_entry(entry) for entry in signature]
    request = {"signature": parsed, "categorical_encoding": categorical_encoding or {}, "rows": rows}
    result = subprocess.run(
        [resolve_feature_evaluator(executable)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode:
        raise ValueError(f"{EVALUATOR_NAME} failed ({result.returncode}): {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
        digest, values = response["features_hash"], response["values"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("missing features_hash")
        if len(values) != len(rows) or any(len(row) != len(parsed) for row in values):
            raise ValueError("feature matrix shape does not match the request")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for row in values for value in row):
            raise ValueError("non-finite or non-numeric feature result")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {EVALUATOR_NAME} response: {error}") from error
    return digest, values


def evaluate_feature_maps(rows: list[dict], signature: list, categorical_encoding: dict | None = None,
                          executable: str | Path | None = None) -> tuple[str, list[list[float]]]:
    """Feature maps (name -> value, None for absent) through the runtime's FeatureExtractor.

    A map that leaves a bare reference unbound fails the whole batch, exactly as the engine
    refuses that row; `value_or_default`/`present` over an absent name evaluate as they do
    at runtime. That is the only definition of "this signature evaluates" there is.
    """
    return _run_feature_evaluator(signature, categorical_encoding, rows, executable)


def evaluate_feature_rows(df, signature: list, categorical_encoding: dict | None = None,
                          executable: str | Path | None = None) -> tuple[str, list[list[float]]]:
    """One batch through the exact C++ expression implementation used at runtime.

    Every referenced name goes over as JSON null where the row does not publish it,
    including a column no row has: absence is the evaluator's to interpret, and NaN is not
    JSON at all.
    """
    columns = [reference[1:] for reference in signature_references(signature)]
    present = [column for column in columns if column in df.columns]
    # to_dict preserves full floating-point precision and native list-valued bindings. With
    # no referenced column present it returns no records at all, not one empty map per row.
    records = df[present].to_dict(orient="records") if present else [{} for _ in range(len(df))]
    rows = [{column: None if _is_absent(record.get(column)) else record[column] for column in columns}
            for record in records]
    return evaluate_feature_maps(rows, signature, categorical_encoding, executable)
