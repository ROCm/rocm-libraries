# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Canonical inline UHD features and the shared runtime evaluator protocol."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
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


def canonicalize_signature(signature: list) -> str:
    parsed = [parse_signature_entry(entry) for entry in signature]
    return json.dumps(parsed, separators=(",", ":"), sort_keys=True, ensure_ascii=False, allow_nan=False)


def compute_features_hash(signature: list, categorical_encoding: dict | None = None) -> str:
    """Preserve raw-reference hashes; computed training uses the helper's hash."""
    serialized = canonicalize_signature(signature)
    if categorical_encoding:
        serialized += "|" + json.dumps(
            categorical_encoding, separators=(",", ":"), sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


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
    """Stable per-reference codes, preserving the exact published string values."""
    encoding = {}
    for column in feature_cols:
        series = df[column]
        if getattr(series.dtype, "kind", "O") in "biuf":
            continue
        values = set()
        other_types = set()
        for value in series:
            if isinstance(value, str):
                values.add(value)
            else:
                other_types.add(type(value).__name__)
        if not values:
            continue
        if other_types:
            raise ValueError(f"feature column {column!r} mixes strings with {', '.join(sorted(other_types))}")
        encoding[feature_reference(column)] = {
            value: code for code, value in enumerate(sorted(values))
        }
    return encoding


def resolve_feature_evaluator(executable: str | Path | None = None) -> str:
    requested = str(executable) if executable else os.environ.get("HIPDNN_UHD_FEATURE_EVALUATOR", "hipdnn_uhd_features")
    resolved = shutil.which(requested)
    if resolved is None:
        raise ValueError(
            f"shared feature evaluator {requested!r} is unavailable; pass --feature-evaluator, "
            "set HIPDNN_UHD_FEATURE_EVALUATOR, or install hipdnn_uhd_features on PATH"
        )
    return resolved


def evaluate_feature_rows(df, signature: list, categorical_encoding: dict | None = None,
                          executable: str | Path | None = None) -> tuple[str, list[list[float]]]:
    """One batch through the exact C++ expression implementation used at runtime."""
    references = signature_references(signature)
    columns = [reference[1:] for reference in references]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    # to_dict preserves full floating-point precision and native list-valued bindings.
    rows = df[columns].to_dict(orient="records")
    request = {"signature": signature, "categorical_encoding": categorical_encoding or {}, "rows": rows}
    result = subprocess.run(
        [resolve_feature_evaluator(executable)],
        input=json.dumps(request, ensure_ascii=False, allow_nan=False),
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode:
        raise ValueError(f"hipdnn_uhd_features failed ({result.returncode}): {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
        digest, values = response["features_hash"], response["values"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("missing features_hash")
        if len(values) != len(df) or any(len(row) != len(signature) for row in values):
            raise ValueError("feature matrix shape does not match the request")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for row in values for value in row):
            raise ValueError("non-finite or non-numeric feature result")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid hipdnn_uhd_features response: {error}") from error
    return digest, values
