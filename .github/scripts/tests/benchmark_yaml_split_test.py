# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The benchmark must not call skipped or incomplete runs successful."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_yaml_split as benchmark


class CompletenessTest(unittest.TestCase):
    def test_collection_requires_exact_count(self):
        with patch.dict(os.environ, SPLIT_BENCH_EXPECTED="2"):
            benchmark.pytest_collection_modifyitems(None, None, [object(), object()])
            with self.assertRaises(pytest.UsageError):
                benchmark.pytest_collection_modifyitems(None, None, [object()])

    def test_only_all_passed_is_complete(self):
        for outcomes, complete in [
            (["passed", "passed"], True),
            (["passed"], False),
            (["passed", "skipped"], False),
            (["passed", "failed"], False),
            (["passed", "passed", "passed"], False),
        ]:
            with self.subTest(
                outcomes=outcomes
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "verdicts.json"
                session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
                reports = [
                    {
                        "nodeid": str(i),
                        "phase": "call",
                        "outcome": outcome,
                        "numerical": {"outcomes": {"PASSED": 1}},
                    }
                    for i, outcome in enumerate(outcomes)
                ]
                with patch.dict(
                    os.environ, SPLIT_BENCH_EXPECTED="2", SPLIT_BENCH_VERDICTS=str(path)
                ):
                    with patch.object(benchmark, "_reports", reports):
                        benchmark.pytest_sessionfinish(session, 0)
                self.assertEqual(json.loads(path.read_text())["complete"], complete)
                self.assertEqual(session.exitstatus, 0 if complete else 1)

    def test_failed_setup_is_recorded(self):
        with patch.object(benchmark, "_reports", []):
            benchmark.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid="case",
                    when="setup",
                    failed=True,
                    skipped=False,
                    outcome="failed",
                )
            )
            self.assertEqual(benchmark._reports[0]["phase"], "setup")

    def test_numerical_identity_ignores_performance_fields(self):
        header = "run,problem-progress,operation,problem-sizes,bias-type,factor-dim,activation-type,solution,validation,time-us,gflops\n"
        row = '0,1/1,GEMM,"(1,2,3,4)",None,0,None,kernel,PASSED,123,456\n'
        first = benchmark.numerical_verdicts(header + row)
        second = benchmark.numerical_verdicts(
            header + row.replace("123,456", "999,888")
        )
        self.assertEqual(first, second)
        self.assertEqual(first["outcomes"], {"PASSED": 1})
        self.assertEqual(sum(first["cases"].values()), 1)
        self.assertNotEqual(
            first, benchmark.numerical_verdicts(header + row.replace("kernel", "other"))
        )

    def test_pass_without_numerical_execution_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verdicts.json"
            session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
            with patch.dict(
                os.environ, SPLIT_BENCH_EXPECTED="1", SPLIT_BENCH_VERDICTS=str(path)
            ):
                report = {"nodeid": "case", "phase": "call", "outcome": "passed"}
                with patch.object(benchmark, "_reports", [report]):
                    benchmark.pytest_sessionfinish(session, 0)
            self.assertFalse(json.loads(path.read_text())["complete"])

    def test_duplicate_pass_does_not_replace_missing_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verdicts.json"
            session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
            with patch.dict(
                os.environ, SPLIT_BENCH_EXPECTED="2", SPLIT_BENCH_VERDICTS=str(path)
            ):
                report = {
                    "nodeid": "same-case",
                    "phase": "call",
                    "outcome": "passed",
                    "numerical": {"outcomes": {"PASSED": 1}},
                }
                with patch.object(benchmark, "_reports", [report, report]):
                    benchmark.pytest_sessionfinish(session, 0)
            self.assertFalse(json.loads(path.read_text())["complete"])
            self.assertEqual(session.exitstatus, 1)

    def test_lock_scope_preserves_hardware_serialization(self):
        spec = importlib.util.spec_from_file_location(
            "benchmark_conftest", benchmark.PACKAGE / "tensilelite/Tests/conftest.py"
        )
        conftest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conftest)
        factory = SimpleNamespace(getbasetemp=lambda: Path("/tmp/pytest/worker"))
        for scope, expected in (("gpu", "gpu0"), ("worker", "gw2")):
            with self.subTest(scope=scope), patch.dict(
                os.environ, HSA_MODEL_MEMFILE=""
            ):
                config = SimpleNamespace(getoption=lambda key: scope)
                lock = conftest.worker_lock_path.__wrapped__(
                    factory, "gw2", "0", config
                )
                self.assertEqual(lock.name, f"client_execution_{expected}.lock")


if __name__ == "__main__":
    unittest.main()
