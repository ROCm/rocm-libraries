# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import rocke_client_aot.instance_schema as instance_schema

import pytest

from rocke_client_aot.instance_schema import (
    InstanceError,
    attributes_match_constraints,
    normalize_attribute_constraints,
    parse_instance_list,
    require_int,
    require_mapping,
    require_string,
)


def test_attribute_constraints_match_all_core_operators():
    constraints = normalize_attribute_constraints(
        {
            "mask_mode": {"equals": "none"},
            "dropout_probability": {"not_equals": 0.5},
            "scale_policy": {"one_of": ["default_1_over_sqrt_d", "explicit"]},
        }
    )

    assert attributes_match_constraints(
        {
            "mask_mode": "none",
            "dropout_probability": 0.0,
            "scale_policy": "explicit",
        },
        constraints,
    )
    assert not attributes_match_constraints(
        {
            "mask_mode": "none",
            "dropout_probability": 0.5,
            "scale_policy": "explicit",
        },
        constraints,
    )
    assert not attributes_match_constraints(
        {
            "mask_mode": "causal",
            "dropout_probability": 0.0,
            "scale_policy": "explicit",
        },
        constraints,
    )
    assert not attributes_match_constraints(
        {
            "mask_mode": "none",
            "dropout_probability": 0.0,
            "scale_policy": "unsupported",
        },
        constraints,
    )


def test_attribute_constraints_reject_unknown_operator():
    with pytest.raises(InstanceError, match="unsupported operators"):
        normalize_attribute_constraints({"mask_mode": {"contains": "none"}})


def test_attribute_constraints_require_non_empty_one_of():
    with pytest.raises(InstanceError, match="one_of"):
        normalize_attribute_constraints({"scale_policy": {"one_of": []}})


def test_attribute_constraints_reject_empty_rule():
    with pytest.raises(InstanceError, match="must not be empty"):
        normalize_attribute_constraints({"mask_mode": {}})


def test_attribute_constraints_require_object_and_string_keys():
    with pytest.raises(
        InstanceError, match="selection.attribute_constraints must be an object"
    ):
        normalize_attribute_constraints([])

    with pytest.raises(InstanceError, match="key must be a non-empty string"):
        normalize_attribute_constraints({1: {"equals": "none"}})


def test_scalar_require_helpers_reject_wrong_types():
    with pytest.raises(InstanceError, match="must be an object"):
        require_mapping([], "value")

    with pytest.raises(InstanceError, match="must be a non-empty string"):
        require_string("", "value")

    with pytest.raises(InstanceError, match="must be an integer"):
        require_int(True, "value")


def _instance_data(name="valid", *, op="test_op", family="test_family", arch="gfx000"):
    return {
        "schema": "rocke.aot.instance/v1",
        "name": name,
        "op": op,
        "family": family,
        "arch": arch,
        "compile_spec": {"dtype": "fp16"},
        "selection": {"attribute_constraints": {"mask_mode": {"equals": "none"}}},
        "test_profiles": [{"batch": 1}],
    }


def _write_list(path, instances):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instances), encoding="utf-8")
    return instances


def _write_handler(
    kernel_dir,
    *,
    op="test_op",
    family="test_family",
    include_parse=True,
    include_build=True,
    include_sidecar=True,
):
    lines = [f"OP = {op!r}", f"FAMILY = {family!r}"]
    if include_parse:
        lines.extend(
            [
                "def parse_instance_fields(instance, source):",
                "    return (",
                "        {",
                "            'compile_spec': {'dtype': instance['compile_spec']['dtype'], 'normalized': True},",
                "            'selection': instance['selection'],",
                "            'test_profiles': instance['test_profiles'],",
                "        },",
                "        {'spec': str(source)},",
                "        'handler accepted instance',",
                "    )",
            ]
        )
    if include_build:
        lines.extend(
            [
                "def build_kernel(spec, *, arch):",
                "    return {'spec': spec, 'arch': arch}",
            ]
        )
    if include_sidecar:
        lines.extend(
            [
                "def emit_sidecar(instance, spec, artifact, hsaco_filename):",
                "    return {'hsaco_filename': hsaco_filename}",
            ]
        )
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "aot_instance.py").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_parse_instance_list_loads_handler_from_arch_tree(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = kernel_dir / "gfx000" / "aot_list.json"
    _write_list(list_path, [_instance_data()])
    _write_handler(kernel_dir)

    parsed = parse_instance_list(list_path)

    assert len(parsed) == 1
    one = parsed[0]
    assert one.path == list_path
    assert one.data["compile_spec"] == {"dtype": "fp16", "normalized": True}
    assert one.compile_spec is one.data["compile_spec"]
    assert one.selection is one.data["selection"]
    assert one.test_profiles == [{"batch": 1}]
    assert one.validation_reason == "handler accepted instance"
    assert one.actions.build_kernel(one.spec, arch="gfx000") == {
        "spec": one.spec,
        "arch": "gfx000",
    }
    assert one.actions.emit_sidecar(one, one.spec, object(), "kernel.hsaco") == {
        "hsaco_filename": "kernel.hsaco"
    }


def test_parse_instance_list_parses_every_element(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = kernel_dir / "gfx000" / "aot_list.json"
    _write_list(
        list_path,
        [_instance_data(name="first"), _instance_data(name="second")],
    )
    _write_handler(kernel_dir)

    parsed = parse_instance_list(list_path)

    assert [one.data["name"] for one in parsed] == ["first", "second"]


def test_parse_instance_list_rejects_duplicate_names(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = kernel_dir / "gfx000" / "aot_list.json"
    _write_list(list_path, [_instance_data(), _instance_data()])
    _write_handler(kernel_dir)

    with pytest.raises(InstanceError, match="duplicate instance name"):
        parse_instance_list(list_path)


def test_parse_instance_list_enforces_expected_arch(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = kernel_dir / "gfx000" / "aot_list.json"
    _write_list(list_path, [_instance_data(arch="gfx000")])
    _write_handler(kernel_dir)

    with pytest.raises(InstanceError, match="does not match expected arch"):
        parse_instance_list(list_path, expected_arch="gfx999")


def test_parse_instance_list_requires_kernel_dir_outside_arch_tree(tmp_path):
    list_path = tmp_path / "loose.json"
    _write_list(list_path, [_instance_data()])

    with pytest.raises(InstanceError, match="kernel_dir is required"):
        parse_instance_list(list_path)


def test_parse_instance_list_rejects_unreadable_malformed_or_non_array(tmp_path):
    missing_path = tmp_path / "missing" / "aot_list.json"
    with pytest.raises(InstanceError, match="failed to read instance list"):
        parse_instance_list(missing_path, kernel_dir=tmp_path)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("[", encoding="utf-8")
    with pytest.raises(InstanceError, match="failed to parse instance list"):
        parse_instance_list(malformed_path, kernel_dir=tmp_path)

    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps(_instance_data()), encoding="utf-8")
    with pytest.raises(InstanceError, match="must be a JSON array"):
        parse_instance_list(object_path, kernel_dir=tmp_path)

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    with pytest.raises(InstanceError, match="must not be empty"):
        parse_instance_list(empty_path, kernel_dir=tmp_path)


def test_parse_instance_list_rejects_invalid_envelope_before_handler(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_handler(kernel_dir)

    data = _instance_data()
    data["extra"] = True
    _write_list(list_path, [data])
    with pytest.raises(InstanceError, match="unsupported top-level fields"):
        parse_instance_list(list_path, kernel_dir=kernel_dir)

    data.pop("extra")
    data["test_profiles"] = {}
    _write_list(list_path, [data])
    with pytest.raises(InstanceError, match="test_profiles must be an array"):
        parse_instance_list(list_path, kernel_dir=kernel_dir)


def test_parse_instance_list_requires_selection_attribute_constraints(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_handler(kernel_dir)

    data = _instance_data()
    data.pop("selection")
    _write_list(list_path, [data])
    with pytest.raises(InstanceError, match="selection must be an object"):
        parse_instance_list(list_path, kernel_dir=kernel_dir)

    data["selection"] = {}
    _write_list(list_path, [data])
    with pytest.raises(
        InstanceError, match="selection.attribute_constraints must be an object"
    ):
        parse_instance_list(list_path, kernel_dir=kernel_dir)


@pytest.mark.parametrize(
    ("normalized_fields", "message"),
    [
        (
            "{'compile_spec': {}, 'selection': {}, 'extra': True}",
            "normalized fields contain unsupported entries",
        ),
        (
            "{'compile_spec': {}, 'selection': {}, 'test_profiles': {}}",
            "normalized test_profiles must be an array",
        ),
        (
            "{'compile_spec': {}, 'selection': {}, 'test_profiles': []}",
            "selection.attribute_constraints must be an object",
        ),
    ],
)
def test_parse_instance_list_rejects_invalid_normalized_fields(
    tmp_path, normalized_fields, message
):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_list(list_path, [_instance_data()])
    _write_handler(kernel_dir)
    handler_path = kernel_dir / "aot_instance.py"

    handler_path.write_text(
        "\n".join(
            [
                "OP = 'test_op'",
                "FAMILY = 'test_family'",
                "def parse_instance_fields(instance, source):",
                f"    return ({normalized_fields}, object(), 'bad')",
                "def build_kernel(spec, *, arch):",
                "    return spec",
                "def emit_sidecar(instance, spec, artifact, hsaco_filename):",
                "    return {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InstanceError, match=message):
        parse_instance_list(list_path, kernel_dir=kernel_dir)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("parse", "parse_instance_fields"),
        ("build", "build_kernel"),
        ("sidecar", "emit_sidecar"),
    ],
)
def test_parse_instance_list_rejects_incomplete_handler(tmp_path, missing, message):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_list(list_path, [_instance_data()])
    _write_handler(
        kernel_dir,
        include_parse=missing != "parse",
        include_build=missing != "build",
        include_sidecar=missing != "sidecar",
    )

    with pytest.raises(InstanceError, match=message):
        parse_instance_list(list_path, kernel_dir=kernel_dir)


def test_parse_instance_list_rejects_unloadable_handler_spec(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_list(list_path, [_instance_data()])
    _write_handler(kernel_dir)
    monkeypatch.setattr(
        instance_schema.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InstanceError, match="failed to load kernel handler"):
        parse_instance_list(list_path, kernel_dir=kernel_dir)


def test_parse_instance_list_rejects_missing_handler_file(tmp_path):
    kernel_dir = tmp_path / "kernel"
    list_path = tmp_path / "aot_list.json"
    _write_list(list_path, [_instance_data()])
    kernel_dir.mkdir()

    with pytest.raises(InstanceError, match="missing aot_instance.py"):
        parse_instance_list(list_path, kernel_dir=kernel_dir)
