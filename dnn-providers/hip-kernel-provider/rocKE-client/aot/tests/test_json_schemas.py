# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocke_client_aot.json_schema import (
    SchemaValidationError,
    load_json_schema,
    validate_json_schema,
)

AOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = AOT_DIR / "schemas"


def _load_schema(name: str) -> tuple[dict, Path]:
    schema_path = SCHEMA_DIR / name
    return load_json_schema(schema_path), schema_path


def test_common_instance_schema_accepts_generic_envelope():
    schema, schema_path = _load_schema("instance.schema.json")
    instance = {
        "schema": "rocke.aot.instance/v1",
        "name": "example",
        "op": "test_op",
        "family": "test_family",
        "arch": "gfx000",
        "compile_spec": {"operation_specific": True},
        "selection": {
            "attribute_constraints": {
                "mode": {"one_of": ["a", "b"]},
            },
        },
        "test_profiles": [{"batch": 1}],
    }

    validate_json_schema(instance, schema, schema_path=schema_path)


def test_common_instance_schema_rejects_top_level_drift():
    schema, schema_path = _load_schema("instance.schema.json")
    instance = {
        "schema": "rocke.aot.instance/v1",
        "name": "example",
        "op": "test_op",
        "family": "test_family",
        "arch": "gfx000",
        "compile_spec": {},
        "selection": {},
        "test_profiles": [],
        "extra": True,
    }

    with pytest.raises(SchemaValidationError, match="Additional properties"):
        validate_json_schema(instance, schema, schema_path=schema_path)


def test_common_sidecar_schema_accepts_generic_envelope():
    schema, schema_path = _load_schema("sidecar.schema.json")
    sidecar = {
        "schema": "rocke.aot.sidecar/v1",
        "cache_key": "example-cache-key",
        "artifact": {"operation_specific": True},
        "selection": {"operation_specific": True},
        "launch": {"operation_specific": True},
        "args_signature": [{"operation_specific": True}],
    }

    validate_json_schema(sidecar, schema, schema_path=schema_path)


def test_common_sidecar_schema_rejects_missing_cache_key():
    schema, schema_path = _load_schema("sidecar.schema.json")
    sidecar = {
        "schema": "rocke.aot.sidecar/v1",
        "artifact": {},
        "selection": {},
        "launch": {},
        "args_signature": [],
    }

    with pytest.raises(SchemaValidationError, match="cache_key"):
        validate_json_schema(sidecar, schema, schema_path=schema_path)


def test_schema_files_are_valid_json():
    for schema_path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        assert json.loads(schema_path.read_text(encoding="utf-8"))["$schema"]
