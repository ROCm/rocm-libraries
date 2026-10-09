################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.CustomYamlLoader``: the event-based YAML
parser (`parse_general`/`parse_sequence`/`parse_mapping`/`parse_scalar`) and the
stream/sequence-item/dict-item/logic-gfx-arch loaders, driven over tmp yaml."""

import pytest

from Tensile.CustomYamlLoader import (
    DEFAULT_YAML_LOADER,
    is_float,
    load_yaml_stream,
    load_yaml_sequence_item,
    load_yaml_dict_item,
    load_logic_gfx_arch,
)

pytestmark = pytest.mark.unit


def _w(tmp_path, text, name="d.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_is_float():
    assert is_float("1.5") and is_float("3") and not is_float("abc")


def test_load_yaml_stream_scalar_types(tmp_path, snapshot):
    # Map of every parse_scalar branch: true/false/null/~/empty/int/float/str +
    # quoted-null (style set -> stays string), nested seq + map.
    doc = (
        "t: true\n"
        "f: False\n"
        "n: null\n"
        "tilde: ~\n"
        "empty:\n"
        "i: -42\n"
        "fl: 3.14\n"
        "s: hello\n"
        "qnull: 'null'\n"
        "seq: [1, 2, three]\n"
        "nested: {a: 1, b: two}\n"
    )
    assert load_yaml_stream(_w(tmp_path, doc), DEFAULT_YAML_LOADER) == snapshot


def test_load_yaml_sequence_items(tmp_path, snapshot):
    p = _w(tmp_path, "- alpha\n- beta\n- gamma\n")
    assert {
        "idx0": load_yaml_sequence_item(p, DEFAULT_YAML_LOADER, 0),
        "idx2": load_yaml_sequence_item(p, DEFAULT_YAML_LOADER, 2),
        "idx_oob": load_yaml_sequence_item(p, DEFAULT_YAML_LOADER, 9),
    } == snapshot


def test_load_yaml_sequence_item_root_not_sequence_raises(tmp_path):
    p = _w(tmp_path, "a: 1\nb: 2\n")
    with pytest.raises(RuntimeError):
        load_yaml_sequence_item(p, DEFAULT_YAML_LOADER, 0)


def test_load_yaml_dict_items(tmp_path, snapshot):
    p = _w(tmp_path, "a: 1\nb: two\nc: [10, 20]\n")
    assert {
        "a": load_yaml_dict_item(p, DEFAULT_YAML_LOADER, "a"),
        "c": load_yaml_dict_item(p, DEFAULT_YAML_LOADER, "c"),
        "missing": load_yaml_dict_item(p, DEFAULT_YAML_LOADER, "zzz"),
    } == snapshot


def test_load_yaml_dict_item_root_not_map_raises(tmp_path):
    p = _w(tmp_path, "- 1\n- 2\n")
    with pytest.raises(RuntimeError):
        load_yaml_dict_item(p, DEFAULT_YAML_LOADER, "a")


def test_load_logic_gfx_arch_sequence_string(tmp_path, snapshot):
    # Root sequence, item index 2 is a plain arch string.
    p = _w(tmp_path, "- {}\n- 1\n- gfx942\n- more\n")
    assert load_logic_gfx_arch(p, DEFAULT_YAML_LOADER) == snapshot


def test_load_logic_gfx_arch_sequence_dict(tmp_path, snapshot):
    # Root sequence, item index 2 is a dict carrying Architecture.
    p = _w(tmp_path, "- {}\n- 1\n- {Architecture: gfx90a}\n")
    assert load_logic_gfx_arch(p, DEFAULT_YAML_LOADER) == snapshot


def test_load_logic_gfx_arch_map_fallback(tmp_path, snapshot):
    # Root is a map -> sequence read raises -> fallback to ArchitectureName key.
    p = _w(tmp_path, "ArchitectureName: gfx1100\nOther: x\n")
    assert load_logic_gfx_arch(p, DEFAULT_YAML_LOADER) == snapshot
