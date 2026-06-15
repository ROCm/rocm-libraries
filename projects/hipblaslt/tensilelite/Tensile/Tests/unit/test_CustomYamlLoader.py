################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.CustomYamlLoader.

CustomYamlLoader is a thin, dependency-light wrapper around PyYAML's event API
that performs strict scalar typing and partial reads of Tensile logic/config
YAML. These tests exercise the real functions against real YAML inputs (no
mocking) and assert on concrete parsed values.
"""

import pytest

from Tensile.CustomYamlLoader import (
    DEFAULT_YAML_LOADER,
    parse_scalar,
    load_yaml_stream,
    load_yaml_sequence_item,
    load_yaml_dict_item,
    load_logic_gfx_arch,
)

pytestmark = pytest.mark.unit


def _parse_scalar(text):
    """Drive a fresh YAML loader to the document's first scalar and parse it.

    The single-scalar event stream is always StreamStart, DocumentStart,
    Scalar, ... so two get_event() calls leave the loader positioned on the
    scalar that parse_scalar consumes.
    """
    loader = DEFAULT_YAML_LOADER(text)
    loader.get_event()  # StreamStartEvent
    loader.get_event()  # DocumentStartEvent
    return parse_scalar(loader)


def _write_yaml(tmp_path, text):
    path = tmp_path / "logic.yaml"
    path.write_text(text)
    return path


class TestParseScalarTyping:
    """parse_scalar applies strict, case-insensitive typing to plain scalars."""

    @pytest.mark.parametrize("text,expected", [
        ("true", True),
        ("True", True),
        ("false", False),
        ("False", False),
    ])
    def test_boolean_keywords_case_insensitive(self, text, expected):
        assert _parse_scalar(text) is expected

    @pytest.mark.parametrize("text", ["null", "Null", "~"])
    def test_unquoted_null_is_none(self, text):
        assert _parse_scalar(text) is None

    @pytest.mark.parametrize("text,expected", [
        ("5", 5),
        ("+5", 5),
        ("-3", -3),
        ("007", 7),
    ])
    def test_integers(self, text, expected):
        result = _parse_scalar(text)
        assert result == expected
        assert type(result) is int  # not bool (a subclass of int)

    @pytest.mark.parametrize("text,expected", [
        ("1.5", 1.5),
        ("-2.5", -2.5),
        ("1e3", 1000.0),
    ])
    def test_floats(self, text, expected):
        result = _parse_scalar(text)
        assert result == expected
        assert type(result) is float

    @pytest.mark.parametrize("text", ["yes", "no", "gfx942", "Yes"])
    def test_non_bool_words_stay_strings(self, text):
        # yes/no are deliberately NOT treated as booleans, and the original
        # (un-lowercased) text is returned.
        result = _parse_scalar(text)
        assert result == text
        assert type(result) is str


class TestParseScalarQuoting:
    """Quoting only affects the null/empty branch -- the only branch that
    inspects evt.style. Quoted booleans and numbers are still coerced, a
    likely-unintended asymmetry that these tests pin down.
    """

    def test_quoted_null_stays_string(self):
        assert _parse_scalar('"null"') == "null"
        assert _parse_scalar("'null'") == "null"

    def test_quoted_empty_stays_string(self):
        assert _parse_scalar('""') == ""

    def test_quoted_boolean_is_still_coerced(self):
        assert _parse_scalar('"true"') is True

    def test_quoted_number_is_still_coerced(self):
        result = _parse_scalar('"5"')
        assert result == 5
        assert type(result) is int


class TestLoadYamlStream:
    def test_nested_mapping_and_sequence(self, tmp_path):
        text = (
            "Name: gfx942\n"
            "Count: 8\n"
            "Ratio: 1.5\n"
            "Enabled: true\n"
            "Disabled: false\n"
            "Empty:\n"
            "Quoted: \"null\"\n"
            "Sub:\n"
            "  Inner: 1\n"
            "  Flag: false\n"
            "Nested:\n"
            "  - 1\n"
            "  - foo\n"
            "  - true\n"
        )
        result = load_yaml_stream(_write_yaml(tmp_path, text), DEFAULT_YAML_LOADER)
        assert result == {
            "Name": "gfx942",
            "Count": 8,
            "Ratio": 1.5,
            "Enabled": True,
            "Disabled": False,
            "Empty": None,
            "Quoted": "null",
            "Sub": {"Inner": 1, "Flag": False},
            "Nested": [1, "foo", True],
        }

    def test_sequence_root(self, tmp_path):
        result = load_yaml_stream(_write_yaml(tmp_path, "- a\n- 2\n- gfx90a\n"), DEFAULT_YAML_LOADER)
        assert result == ["a", 2, "gfx90a"]

    def test_scalar_root_is_typed(self, tmp_path):
        # A bare-scalar document is typed just like any nested scalar.
        result = load_yaml_stream(_write_yaml(tmp_path, "42\n"), DEFAULT_YAML_LOADER)
        assert result == 42
        assert type(result) is int


class TestLoadYamlSequenceItem:
    def test_returns_item_at_index(self, tmp_path):
        path = _write_yaml(tmp_path, "- a\n- 2\n- gfx90a\n")
        assert load_yaml_sequence_item(path, DEFAULT_YAML_LOADER, 0) == "a"
        assert load_yaml_sequence_item(path, DEFAULT_YAML_LOADER, 1) == 2
        assert load_yaml_sequence_item(path, DEFAULT_YAML_LOADER, 2) == "gfx90a"

    def test_out_of_range_returns_none(self, tmp_path):
        path = _write_yaml(tmp_path, "- a\n- b\n")
        assert load_yaml_sequence_item(path, DEFAULT_YAML_LOADER, 5) is None

    def test_map_root_raises_runtime_error(self, tmp_path):
        path = _write_yaml(tmp_path, "key: value\n")
        with pytest.raises(RuntimeError):
            load_yaml_sequence_item(path, DEFAULT_YAML_LOADER, 0)


class TestLoadYamlDictItem:
    def test_returns_value_for_key(self, tmp_path):
        path = _write_yaml(tmp_path, "ArchitectureName: gfx1100\nCount: 8\n")
        assert load_yaml_dict_item(path, DEFAULT_YAML_LOADER, "ArchitectureName") == "gfx1100"
        assert load_yaml_dict_item(path, DEFAULT_YAML_LOADER, "Count") == 8

    def test_missing_key_returns_none(self, tmp_path):
        path = _write_yaml(tmp_path, "ArchitectureName: gfx1100\n")
        assert load_yaml_dict_item(path, DEFAULT_YAML_LOADER, "Nope") is None

    def test_sequence_root_raises_runtime_error(self, tmp_path):
        path = _write_yaml(tmp_path, "- a\n- b\n")
        with pytest.raises(RuntimeError):
            load_yaml_dict_item(path, DEFAULT_YAML_LOADER, "key")


class TestLoadLogicGfxArch:
    # A logic file is a sequence whose third element (index 2) carries the
    # architecture, either as a bare string or a dict with an "Architecture"
    # key. A map-rooted file instead exposes it under "ArchitectureName".
    def test_sequence_root_string_arch(self, tmp_path):
        text = "- {MinimumRequiredVersion: 4.33.0}\n- Aldebaran\n- gfx90a\n"
        assert load_logic_gfx_arch(_write_yaml(tmp_path, text)) == "gfx90a"

    def test_sequence_root_dict_arch(self, tmp_path):
        text = "- {MinimumRequiredVersion: 4.33.0}\n- Aldebaran\n- {Architecture: gfx942, CUCount: 304}\n"
        assert load_logic_gfx_arch(_write_yaml(tmp_path, text)) == "gfx942"

    def test_sequence_too_short_returns_none(self, tmp_path):
        # Fewer than three elements: index 2 is absent, so arch resolves to None.
        text = "- {MinimumRequiredVersion: 4.33.0}\n- Aldebaran\n"
        assert load_logic_gfx_arch(_write_yaml(tmp_path, text)) is None

    def test_map_root_falls_back_to_architecture_name(self, tmp_path):
        text = "ArchitectureName: gfx1100\nOther: 1\n"
        assert load_logic_gfx_arch(_write_yaml(tmp_path, text)) == "gfx1100"
