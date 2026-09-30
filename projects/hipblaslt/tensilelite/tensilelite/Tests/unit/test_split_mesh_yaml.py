# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.Utilities.split_mesh_yaml."""

import csv
import gzip

import pytest

from Tensile.Utilities.split_mesh_yaml import (
    _clear_exact_logic,
    _get_exact_logic,
    _get_library_type,
    _write_csv_gz,
)

pytestmark = pytest.mark.unit


class TestGetLibraryType:
    def test_dict_format(self):
        data = {"LibraryType": "MeshBased"}
        assert _get_library_type(data) == ("MeshBased", "dict")

    def test_list_format(self):
        data = [None] * 12
        data[11] = "MeshBased"
        assert _get_library_type(data) == ("MeshBased", "list")

    def test_list_too_short(self):
        assert _get_library_type([1, 2, 3]) == (None, None)

    def test_dict_missing_key(self):
        assert _get_library_type({}) == (None, "dict")

    def test_non_dict_non_list(self):
        assert _get_library_type("string") == (None, None)


class TestGetExactLogic:
    def test_dict_format(self):
        table = [[[128, 128, 1, 64], [0, 0.0]]]
        data = {"ExactLogic": table}
        assert _get_exact_logic(data, "dict") == table

    def test_list_format(self):
        table = [[[128, 128, 1, 64], [0, 0.0]]]
        data = [None] * 12
        data[7] = table
        assert _get_exact_logic(data, "list") == table

    def test_unknown_format(self):
        assert _get_exact_logic({}, "unknown") is None


class TestClearExactLogic:
    def test_dict_format(self):
        data = {"ExactLogic": [[[128, 128, 1, 64], [0, 0.0]]]}
        _clear_exact_logic(data, "dict")
        assert data["ExactLogic"] is None

    def test_list_format(self):
        data = [None] * 12
        data[7] = [[[128, 128, 1, 64], [0, 0.0]]]
        _clear_exact_logic(data, "list")
        assert data[7] is None


class TestWriteCsvGz:
    def test_round_trip(self, tmp_path):
        exact_logic = [
            [[128, 256, 1, 512], [7, 0.0]],
            [[64, 64, 2, 1024], [3, 0.0]],
        ]
        out_path = str(tmp_path / "test.csv.gz")
        _write_csv_gz(exact_logic, out_path)

        with gzip.open(out_path, "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["M", "N", "batch", "K", "solutionIdx"]
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0] == ["128", "256", "1", "512", "7"]
        assert rows[1] == ["64", "64", "2", "1024", "3"]
