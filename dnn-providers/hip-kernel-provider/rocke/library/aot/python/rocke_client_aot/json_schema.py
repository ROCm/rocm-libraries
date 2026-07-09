# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""JSON Schema helpers for rocKE client AOT schema tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


class SchemaValidationError(ValueError):
    """Raised when a JSON value does not satisfy a checked-in schema."""


def validate_json_schema(
    value: Any,
    schema: bool | Mapping[str, Any],
    *,
    schema_path: str | Path | None = None,
) -> None:
    """Validate ``value`` with jsonschema's Draft 2020-12 validator."""

    resolved_schema = _inline_file_refs(
        schema,
        Path(schema_path).resolve() if schema_path is not None else None,
    )
    try:
        Draft202012Validator.check_schema(resolved_schema)
        Draft202012Validator(resolved_schema).validate(value)
    except (SchemaError, ValidationError) as exc:
        raise SchemaValidationError(exc.message) from exc


def load_json_schema(path: str | Path) -> dict[str, Any]:
    """Load a JSON Schema file as an object."""

    schema_path = Path(path)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"schema {schema_path} must be a JSON object")
    return schema


def _inline_file_refs(
    schema: Any, schema_path: Path | None
) -> bool | dict[str, Any] | list[Any] | Any:
    """Return a copy of a schema with relative file references expanded."""

    if isinstance(schema, bool):
        return schema
    if isinstance(schema, list):
        return [_inline_file_refs(item, schema_path) for item in schema]
    if not isinstance(schema, Mapping):
        return schema

    ref = schema.get("$ref")
    if isinstance(ref, str) and _is_file_ref(ref):
        resolved = _resolve_file_ref(ref, schema_path)
        siblings = {key: value for key, value in schema.items() if key != "$ref"}
        if not siblings:
            return _inline_file_refs(resolved, _ref_path(ref, schema_path))
        return {
            "allOf": [
                _inline_file_refs(resolved, _ref_path(ref, schema_path)),
                _inline_file_refs(siblings, schema_path),
            ]
        }

    return {
        key: _inline_file_refs(value, schema_path)
        for key, value in copy.deepcopy(dict(schema)).items()
    }


def _is_file_ref(ref: str) -> bool:
    """Return whether a reference points to another local schema file."""

    path, _, _ = ref.partition("#")
    return bool(path) and "://" not in path


def _resolve_file_ref(ref: str, schema_path: Path | None) -> bool | dict[str, Any]:
    """Resolve a relative file reference to the referenced schema node."""

    if schema_path is None:
        raise SchemaValidationError(
            f"cannot resolve relative ref {ref!r} without schema_path"
        )

    path, _, fragment = ref.partition("#")
    target_path = (schema_path.parent / path).resolve()
    node: Any = load_json_schema(target_path)
    if fragment:
        for raw_part in fragment.removeprefix("/").split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, Mapping) or part not in node:
                raise SchemaValidationError(f"unresolvable schema ref {ref!r}")
            node = node[part]
    if not isinstance(node, Mapping) and not isinstance(node, bool):
        raise SchemaValidationError(f"schema ref {ref!r} does not resolve to a schema")
    return node


def _ref_path(ref: str, schema_path: Path | None) -> Path | None:
    """Return the filesystem path that should anchor a resolved reference."""

    path, _, _ = ref.partition("#")
    if not path or schema_path is None:
        return schema_path
    return (schema_path.parent / path).resolve()


__all__ = ["SchemaValidationError", "load_json_schema", "validate_json_schema"]
