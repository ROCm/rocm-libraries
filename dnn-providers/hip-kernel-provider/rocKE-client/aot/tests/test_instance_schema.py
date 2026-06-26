# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from rocke_client_aot.instance_schema import (
    InstanceError,
    attributes_match_constraints,
    normalize_attribute_constraints,
    parse_instance,
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


def _write_instance(path, *, op="test_op", family="test_family"):
    data = {
        "schema": "rocke.aot.instance/v1",
        "name": path.name.removesuffix(".instance.json"),
        "op": op,
        "family": family,
        "arch": "gfx000",
        "compile_spec": {"dtype": "fp16"},
        "selection": {"attribute_constraints": {"mask_mode": {"equals": "none"}}},
        "test_profiles": [{"batch": 1}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


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
                "def parse_instance_fields(instance, path):",
                "    return (",
                "        {",
                "            'compile_spec': {'dtype': instance['compile_spec']['dtype'], 'normalized': True},",
                "            'selection': instance['selection'],",
                "            'test_profiles': instance['test_profiles'],",
                "        },",
                "        {'spec': str(path)},",
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


def test_parse_instance_loads_handler_from_instances_tree(tmp_path):
    kernel_dir = tmp_path / "kernel"
    instance_path = kernel_dir / "instances" / "gfx000" / "valid.instance.json"
    _write_instance(instance_path)
    _write_handler(kernel_dir)

    parsed = parse_instance(instance_path)

    assert parsed.path == instance_path
    assert parsed.data["compile_spec"] == {"dtype": "fp16", "normalized": True}
    assert parsed.compile_spec is parsed.data["compile_spec"]
    assert parsed.selection is parsed.data["selection"]
    assert parsed.test_profiles == [{"batch": 1}]
    assert parsed.validation_reason == "handler accepted instance"
    assert parsed.actions.build_kernel(parsed.spec, arch="gfx000") == {
        "spec": parsed.spec,
        "arch": "gfx000",
    }
    assert parsed.actions.emit_sidecar(
        parsed, parsed.spec, object(), "kernel.hsaco"
    ) == {"hsaco_filename": "kernel.hsaco"}


def test_parse_instance_requires_kernel_dir_for_copied_instance(tmp_path):
    instance_path = tmp_path / "valid.instance.json"
    _write_instance(instance_path)

    with pytest.raises(InstanceError, match="kernel_dir is required"):
        parse_instance(instance_path)


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("parse", "parse_instance_fields"),
        ("build", "build_kernel"),
        ("sidecar", "emit_sidecar"),
    ],
)
def test_parse_instance_rejects_incomplete_handler(tmp_path, missing, message):
    kernel_dir = tmp_path / "kernel"
    instance_path = tmp_path / "valid.instance.json"
    _write_instance(instance_path)
    _write_handler(
        kernel_dir,
        include_parse=missing != "parse",
        include_build=missing != "build",
        include_sidecar=missing != "sidecar",
    )

    with pytest.raises(InstanceError, match=message):
        parse_instance(instance_path, kernel_dir=kernel_dir)
