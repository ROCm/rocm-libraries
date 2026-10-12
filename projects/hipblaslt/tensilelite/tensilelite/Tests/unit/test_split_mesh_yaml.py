# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.Utilities.split_mesh_yaml."""

import csv
import lzma

import pytest

from Tensile.Utilities.split_mesh_yaml import (
    _clear_exact_logic,
    _get_exact_logic,
    _get_library_type,
    _get_solution_indices,
    _write_csv_xz,
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


class TestWriteCsvXz:
    def test_round_trip_with_speed(self, tmp_path):
        exact_logic = [
            [[128, 256, 1, 512], [7, 273.64]],
            [[64, 64, 2, 1024], [3, 0.0]],
            [[32, 32, 1, 256], [5, 6.57484e-05]],
        ]
        out_path = str(tmp_path / "test.csv.xz")
        _write_csv_xz(exact_logic, out_path)

        with lzma.open(out_path, "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["M", "N", "batch", "K", "solutionIdx", "speed"]
            rows = list(reader)

        assert len(rows) == 3
        # sorted by M, N, K
        assert rows[0] == ["32", "32", "1", "256", "5", "6.57484e-05"]
        assert rows[1] == ["64", "64", "2", "1024", "3", "0.0"]
        assert rows[2] == ["128", "256", "1", "512", "7", "273.64"]

    def test_round_trip_all_zero_speed(self, tmp_path):
        exact_logic = [
            [[128, 256, 1, 512], [7, 0.0]],
            [[64, 64, 2, 1024], [3, 0.0]],
        ]
        out_path = str(tmp_path / "test.csv.xz")
        _write_csv_xz(exact_logic, out_path)

        with lzma.open(out_path, "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["M", "N", "batch", "K", "solutionIdx"]
            rows = list(reader)

        assert len(rows) == 2
        # sorted by M, N, K
        assert rows[0] == ["64", "64", "2", "1024", "3"]
        assert rows[1] == ["128", "256", "1", "512", "7"]

    def test_rows_sorted_by_m_n_k(self, tmp_path):
        exact_logic = [
            [[256, 128, 1, 64], [0, 0.0]],
            [[128, 256, 1, 128], [1, 0.0]],
            [[128, 128, 1, 256], [2, 0.0]],
            [[128, 128, 1, 64], [3, 0.0]],
        ]
        out_path = str(tmp_path / "test.csv.xz")
        _write_csv_xz(exact_logic, out_path)

        with lzma.open(out_path, "rt") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)

        assert rows[0] == ["128", "128", "1", "64", "3"]
        assert rows[1] == ["128", "128", "1", "256", "2"]
        assert rows[2] == ["128", "256", "1", "128", "1"]
        assert rows[3] == ["256", "128", "1", "64", "0"]

    def test_bad_key_length(self, tmp_path):
        exact_logic = [
            [[128, 256, 1, 512], [7, 0.0]],
            [[128, 256, 1], [3, 0.0]],
        ]
        out_path = str(tmp_path / "test.csv.xz")
        with pytest.raises(ValueError, match="entry 1: expected 4-element key"):
            _write_csv_xz(exact_logic, out_path)

    def test_bad_value_length(self, tmp_path):
        exact_logic = [
            [[128, 256, 1, 512], [7]],
        ]
        out_path = str(tmp_path / "test.csv.xz")
        with pytest.raises(ValueError, match="entry 0: expected 2-element value"):
            _write_csv_xz(exact_logic, out_path)


class TestGetSolutionIndices:
    def test_dict_format(self):
        data = {
            "Solutions": [
                {"SolutionIndex": 0},
                {"SolutionIndex": 3},
                {"SolutionIndex": 7},
            ]
        }
        assert _get_solution_indices(data, "dict") == {0, 3, 7}

    def test_list_format(self):
        data = [None] * 12
        data[5] = [
            {"SolutionIndex": 0},
            {"SolutionIndex": 5},
        ]
        assert _get_solution_indices(data, "list") == {0, 5}

    def test_dict_no_solutions(self):
        assert _get_solution_indices({}, "dict") == set()

    def test_unknown_format(self):
        assert _get_solution_indices({}, "unknown") == set()
