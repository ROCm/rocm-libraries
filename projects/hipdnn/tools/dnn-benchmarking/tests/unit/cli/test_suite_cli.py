# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite CLI argument parsing and run_suite() workflow."""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    SuiteMetadata,
    SuiteResult,
    TimingStats,
)


def _mock_hipdnn():
    """Create a mock hipdnn_frontend module with a Handle class."""
    mock_module = ModuleType("hipdnn_frontend")
    mock_module.Handle = MagicMock  # type: ignore[attr-defined]
    return mock_module


class TestParserGlobAndFilters:
    """Tests for --graph glob pattern and --engine filter flags."""

    def test_graph_accepts_glob_pattern_string(self) -> None:
        """Test 1: --graph accepts a glob pattern string and stores as-is."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "graphs/*.json"])
        assert isinstance(args.graph, str)
        assert args.graph == "graphs/*.json"

    def test_engine_flag_stores_int(self) -> None:
        """Test 2: --engine flag stores an int engine ID (default None)."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "3"])
        assert args.engine == 3
        assert isinstance(args.engine, int)

    def test_engine_flag_default_none(self) -> None:
        """Test 2b: --engine defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json"])
        assert args.engine is None


class TestMainRouting:
    """Tests for main() routing logic (glob resolution -> run_suite vs run_benchmark)."""

    def _create_graph_files(self, tmpdir: Path, count: int) -> list:
        """Create temporary graph JSON files."""
        paths = []
        for i in range(count):
            p = tmpdir / f"graph_{i}.json"
            p.write_text(json.dumps({"name": f"graph_{i}", "nodes": [], "tensors": []}))
            paths.append(str(p))
        return paths

    @patch("dnn_benchmarking.cli.main.run_suite")
    @patch("dnn_benchmarking.cli.main.run_benchmark")
    def test_multi_file_glob_routes_to_run_suite(
        self, mock_run_benchmark: MagicMock, mock_run_suite: MagicMock
    ) -> None:
        """Test 4: When --graph resolves to multiple files, main() routes to run_suite()."""
        mock_run_suite.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_graph_files(Path(tmpdir), 3)
            glob_pattern = os.path.join(tmpdir, "*.json")

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", glob_pattern]):
                result = main()

            mock_run_suite.assert_called_once()
            mock_run_benchmark.assert_not_called()
            assert result == 0

    @patch("dnn_benchmarking.cli.main.run_suite")
    @patch("dnn_benchmarking.cli.main.run_benchmark")
    def test_single_file_routes_to_run_benchmark(
        self, mock_run_benchmark: MagicMock, mock_run_suite: MagicMock
    ) -> None:
        """Test 5: Single file routes to existing run_benchmark() (backward compatible)."""
        mock_run_benchmark.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", paths[0]]):
                result = main()

            mock_run_benchmark.assert_called_once()
            mock_run_suite.assert_not_called()
            assert result == 0

    def test_zero_files_glob_returns_error(self) -> None:
        """Test 6: When glob resolves to zero files, main() returns 1."""
        from dnn_benchmarking.cli.main import main

        with patch(
            "sys.argv",
            ["dnn-benchmark", "--graph", "/nonexistent/path/*.json"],
        ):
            result = main()

        assert result == 1


class TestRunSuiteWorkflow:
    """Tests for run_suite() function behavior."""

    def _make_graph_result(self, name: str, status: str = "success") -> GraphResult:
        """Helper to create a GraphResult with one ProviderEngineResult."""
        correctness = CorrectnessResult(
            execution_success=status == "success",
            tolerance_match=True if status == "success" else None,
            rtol=1e-5,
            atol=1e-8,
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=0,
            status=status,
            correctness=correctness,
            error_message="some error" if status == "error" else None,
        )
        return GraphResult(
            graph_name=name,
            graph_path=f"/path/{name}.json",
            results=[pe],
        )

    def _make_graph_files(self, tmpdir: Path, count: int) -> list:
        """Create graph files and return paths."""
        paths = []
        for i in range(count):
            p = tmpdir / f"graph_{i}.json"
            p.write_text(
                json.dumps(
                    {
                        "name": f"graph_{i}",
                        "nodes": [
                            {
                                "op_type": "ConvolutionForward",
                                "inputs": {},
                                "outputs": {"y": 100 + i},
                            }
                        ],
                        "tensors": [
                            {
                                "uid": 1 + i * 10,
                                "dims": [1, 3, 4, 4],
                                "strides": [48, 16, 4, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                            {
                                "uid": 100 + i,
                                "dims": [1, 3, 4, 4],
                                "strides": [48, 16, 4, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                        ],
                    }
                )
            )
            paths.append(p)
        return paths

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_all_pass_returns_zero(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 7: run_suite() with all passing graphs returns exit code 0."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.side_effect = [
            self._make_graph_result("g0"),
            self._make_graph_result("g1"),
        ]

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            result = run_suite(paths, config)

        assert result == 0

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_one_failure_still_processes_second(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 8: One failing graph still processes the second (per D-08), returns 1."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.side_effect = [
            self._make_graph_result("g0", status="error"),
            self._make_graph_result("g1"),
        ]

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            result = run_suite(paths, config)

        # Both graphs were processed
        assert mock_run.call_count == 2
        # Returns 1 because one had errors
        assert result == 1

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_correctness_failure_returns_two(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 9: Correctness failure returns exit code 2 (per D-09)."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        # Create a result with correctness failure
        correctness_fail = CorrectnessResult(
            execution_success=True,
            tolerance_match=False,
            rtol=1e-5,
            atol=1e-8,
        )
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=0,
            status="success",
            correctness=correctness_fail,
        )
        fail_result = GraphResult(
            graph_name="g0",
            graph_path="/path/g0.json",
            results=[pe],
        )
        mock_run.return_value = fail_result

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            config = SuiteConfig()
            result = run_suite(paths, config)

        assert result == 2

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_json_output_written_when_output_specified(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 10: run_suite() writes JSON to --output path when specified (per D-16)."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            output_file = Path(tmpdir) / "results.json"
            config = SuiteConfig()
            run_suite(paths, config, output_path=output_file)

            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "metadata" in data
            assert "graphs" in data

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_no_json_output_when_output_not_specified(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 11: run_suite() does not write JSON when --output not specified (per D-16)."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            config = SuiteConfig()
            # No output_path
            run_suite(paths, config)

        # No crash, no file written -- just a smoke test that it completes

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_warmup_iters_passed_per_graph(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Test 12: Warmup and benchmark iterations apply per graph independently (per D-06)."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig(warmup_iters=20, benchmark_iters=200)
            run_suite(paths, config)

        # Each graph gets the same config (warmup/iters applied per graph)
        assert mock_run.call_count == 2
        for call_args in mock_run.call_args_list:
            passed_config = call_args[0][3]  # 4th positional arg is config
            assert passed_config.warmup_iters == 20
            assert passed_config.benchmark_iters == 200

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_graph_load_error_continues_to_next(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite() catches GraphLoadError per graph and continues (per D-08)."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g1")

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            # First file has invalid JSON
            bad_file = Path(tmpdir) / "bad_graph.json"
            bad_file.write_text("{invalid json")

            good_paths = self._make_graph_files(Path(tmpdir), 1)
            paths = [bad_file] + good_paths
            config = SuiteConfig()
            result = run_suite(paths, config)

        # Should still process the second graph
        assert mock_run.call_count == 1
        # Returns 1 because of the error on first graph
        assert result == 1
