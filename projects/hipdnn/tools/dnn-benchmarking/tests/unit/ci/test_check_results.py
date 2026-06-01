# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the CI benchmark gate checker (``ci/check_results.py``).

The gate lives outside the ``dnn_benchmarking`` package (it is a CI helper),
so it is loaded by file path rather than imported as a module.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parents[3] / "ci"
_CHECK_RESULTS = _CI_DIR / "check_results.py"

_spec = importlib.util.spec_from_file_location("ci_check_results", _CHECK_RESULTS)
check_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_results)


def _write_results(tmp_path: Path, *, passed: int, errored: int) -> Path:
    """Write a minimal results.json with the metadata counts the gate reads."""
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "pass_combinations": passed,
                    "error_combinations": errored,
                },
                "graphs": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestCheckResultsGate:
    def test_pass_when_passes_and_no_errors(self, tmp_path):
        results = _write_results(tmp_path, passed=3, errored=0)
        assert check_results.main([str(results)]) == 0

    def test_fail_when_errors_present(self, tmp_path):
        results = _write_results(tmp_path, passed=2, errored=1)
        assert check_results.main([str(results)]) == 1

    def test_fail_when_all_skipped(self, tmp_path):
        # pass==0, error==0 -> nothing ran successfully (silent-skip hole).
        results = _write_results(tmp_path, passed=0, errored=0)
        assert check_results.main([str(results)]) == 1

    def test_error_when_metadata_missing(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"graphs": []}), encoding="utf-8")
        assert check_results.main([str(results)]) == 1

    def test_error_when_field_missing(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps({"metadata": {"pass_combinations": 1}}), encoding="utf-8"
        )
        assert check_results.main([str(results)]) == 1

    def test_error_when_file_missing(self, tmp_path):
        assert check_results.main([str(tmp_path / "nope.json")]) == 1

    def test_error_when_invalid_json(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("{not json", encoding="utf-8")
        assert check_results.main([str(results)]) == 1


class TestEvaluate:
    def test_evaluate_returns_message(self, tmp_path):
        results = _write_results(tmp_path, passed=1, errored=0)
        ok, message = check_results.evaluate(results)
        assert ok is True
        assert "PASSED" in message

    def test_evaluate_raises_on_schema_drift(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
        with pytest.raises(check_results.GateError):
            check_results.evaluate(results)
