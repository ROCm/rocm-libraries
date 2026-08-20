# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from rocke.analysis.lds.model import (
    AccessClassification,
    AccessResult,
    ConflictSummary,
    LdsConflictResult,
    ModelValidationError,
    NormalizedRequest,
    ProfileIdentity,
)
from rocke.analysis.lds.serialization import dumps, loads


def _result() -> LdsConflictResult:
    accesses = (
        AccessResult(
            access_id=0,
            lane=0,
            lds_byte_address=16,
            access_width_bytes=4,
            coordinate=None,
            classification=AccessClassification.NORMAL,
        ),
    )
    groups = ()
    return LdsConflictResult(
        profile=ProfileIdentity("gfx90a", 1),
        request=NormalizedRequest("ds_read_b32", "read", 4, 64, (0,)),
        coordinate_axes=(),
        accesses=accesses,
        conflict_groups=groups,
        summary=ConflictSummary.from_results(accesses, groups),
    )


def test_canonical_json_round_trip_is_stable():
    result = _result()

    document = dumps(result)

    assert document == dumps(loads(document))
    assert document.startswith('{"accesses":')
    assert " " not in document
    assert loads(document) == result


def test_loads_rejects_unknown_schema_version():
    document = _result().as_dict()
    document["schema_version"] = 2

    with pytest.raises(ModelValidationError, match="unsupported schema_version: 2"):
        loads(json.dumps(document))


def test_loads_rejects_duplicate_keys_and_invalid_json():
    with pytest.raises(ModelValidationError, match="duplicate JSON object key"):
        loads('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ModelValidationError, match="invalid JSON document"):
        loads("{")


def test_serialization_accepts_utf8_bytes_and_rejects_other_types():
    result = _result()

    assert loads(dumps(result).encode("utf-8")) == result
    with pytest.raises(TypeError):
        dumps(result.as_dict())
    with pytest.raises(TypeError):
        loads(1)


def _load_schema() -> dict[str, object]:
    schema_path = (
        Path(__file__).parents[2]
        / "python"
        / "rocke"
        / "analysis"
        / "lds"
        / "schema"
        / "lds-conflict-result-v1.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_checked_in_schema_is_valid_json_for_schema_version_one():
    schema = _load_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"] == {"const": 1}


def test_checked_in_schema_accepts_canonical_document_when_jsonschema_is_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=json.loads(dumps(_result())), schema=schema)
