# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import rocke_client_aot.json_schema as json_schema
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


def test_load_json_schema_requires_top_level_object(tmp_path):
    schema_path = tmp_path / "array.schema.json"
    schema_path.write_text("[]", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="must be a JSON object"):
        load_json_schema(schema_path)


def test_relative_file_ref_without_siblings_inlines_target_schema(tmp_path):
    child_path = tmp_path / "child.schema.json"
    child_path.write_text(
        json.dumps({"type": "object", "required": ["ok"]}), encoding="utf-8"
    )
    schema_path = tmp_path / "root.schema.json"
    schema_path.write_text(json.dumps({"$ref": "child.schema.json"}), encoding="utf-8")
    schema = load_json_schema(schema_path)

    validate_json_schema({"ok": True}, schema, schema_path=schema_path)
    with pytest.raises(SchemaValidationError, match="'ok' is a required property"):
        validate_json_schema({}, schema, schema_path=schema_path)


def test_relative_file_ref_with_siblings_combines_constraints(tmp_path):
    child_path = tmp_path / "child.schema.json"
    child_path.write_text(
        json.dumps({"type": "object", "required": ["from_child"]}),
        encoding="utf-8",
    )
    schema_path = tmp_path / "root.schema.json"
    schema_path.write_text(
        json.dumps({"$ref": "child.schema.json", "required": ["from_root"]}),
        encoding="utf-8",
    )
    schema = load_json_schema(schema_path)

    validate_json_schema(
        {"from_child": True, "from_root": True}, schema, schema_path=schema_path
    )
    with pytest.raises(
        SchemaValidationError, match="'from_root' is a required property"
    ):
        validate_json_schema({"from_child": True}, schema, schema_path=schema_path)


def test_local_ref_path_keeps_current_schema_path(tmp_path):
    schema_path = tmp_path / "root.schema.json"

    assert json_schema._ref_path("#/$defs/item", schema_path) == schema_path


def test_relative_file_ref_requires_schema_path():
    with pytest.raises(SchemaValidationError, match="without schema_path"):
        validate_json_schema({}, {"$ref": "child.schema.json"})


def test_relative_file_ref_fragment_errors_are_reported(tmp_path):
    defs_path = tmp_path / "defs.schema.json"
    defs_path.write_text(
        json.dumps(
            {
                "$defs": {
                    "object": {"type": "object"},
                    "scalar": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    schema_path = tmp_path / "root.schema.json"

    schema_path.write_text(
        json.dumps({"$ref": "defs.schema.json#/$defs/missing"}), encoding="utf-8"
    )
    with pytest.raises(SchemaValidationError, match="unresolvable schema ref"):
        validate_json_schema({}, load_json_schema(schema_path), schema_path=schema_path)

    schema_path.write_text(
        json.dumps({"$ref": "defs.schema.json#/$defs/scalar"}), encoding="utf-8"
    )
    with pytest.raises(SchemaValidationError, match="does not resolve to a schema"):
        validate_json_schema({}, load_json_schema(schema_path), schema_path=schema_path)
