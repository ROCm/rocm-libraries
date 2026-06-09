# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for suite CLI argument parsing and run_suite() workflow."""

import json
import importlib
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.cli.suite_runner_cli import run_suite_benchmark
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.reporter import Reporter
from dnn_benchmarking.reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    SuiteMetadata,
    SuiteResult,
)

MAIN_MODULE = importlib.import_module("dnn_benchmarking.cli.main")


def _mock_hipdnn():
    """Create a mock hipdnn_frontend module with a Handle class."""
    mock_module = ModuleType("hipdnn_frontend")
    mock_module.Handle = MagicMock  # type: ignore[attr-defined]
    return mock_module


class TestParserGlobAndFilters:
    """Tests for --graph glob pattern and --engine filter flags."""

    def test_graph_accepts_glob_pattern_string(self) -> None:
        """--graph accepts a glob pattern string and stores as a list."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "graphs/*.json"])
        assert isinstance(args.graph, list)
        assert args.graph == ["graphs/*.json"]

    def test_engine_flag_stores_single_id_as_list(self) -> None:
        """--engine ID stores a one-element list."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "3"])
        assert args.engine == [3]

    def test_engine_flag_accepts_comma_separated_list(self) -> None:
        """--engine 1,2,3 stores [1, 2, 3]."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "1,2,3"])
        assert args.engine == [1, 2, 3]

    def test_engine_flag_strips_whitespace(self) -> None:
        """--engine '1, 2' tolerates spaces around commas."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "1, 2"])
        assert args.engine == [1, 2]

    def test_engine_flag_rejects_non_integer(self) -> None:
        """--engine with a non-integer item raises SystemExit (argparse error)."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--graph", "g.json", "--engine", "1,abc"])

    def test_engine_flag_accepts_negative_id(self) -> None:
        """--engine accepts negative IDs (FNV-1a engine hashes can have the
        high bit set when interpreted as signed int64)."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "-1234567890"])
        assert args.engine == [-1234567890]

    def test_engine_flag_default_none(self) -> None:
        """--engine defaults to None (= run all discovered engines)."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json"])
        assert args.engine is None

    def test_engine_flag_preserves_duplicates(self) -> None:
        """--engine entries are ordered execution selections, not a set."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "1,1,1"])
        assert args.engine == [1, 1, 1]

        args = parser.parse_args(["--graph", "g.json", "--engine", "3,1,3,2"])
        assert args.engine == [3, 1, 3, 2]

    def test_plugin_path_accepts_comma_separated_list(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            ["--graph", "g.json", "--plugin-path", "/plugins/a,/plugins/b"]
        )
        assert args.plugin_path == [Path("/plugins/a"), Path("/plugins/b")]

    def test_verbose_flag_default_false(self) -> None:
        """No -v / --verbose => args.verbose is False."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json"])
        assert args.verbose is False

    def test_verbose_flag_short_form(self) -> None:
        """-v sets args.verbose to True."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "-v"])
        assert args.verbose is True

    def test_verbose_flag_long_form(self) -> None:
        """--verbose sets args.verbose to True."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--verbose"])
        assert args.verbose is True


class TestMainRouting:
    """Tests for main() routing — single and multi files both go through the orchestrator."""

    def _create_graph_files(self, tmpdir: Path, count: int) -> list:
        """Create temporary graph JSON files."""
        paths = []
        for i in range(count):
            p = tmpdir / f"graph_{i}.json"
            p.write_text(json.dumps({"name": f"graph_{i}", "nodes": [], "tensors": []}))
            paths.append(str(p))
        return paths

    @patch("dnn_benchmarking.cli.main.run_suite_cli")
    def test_multi_file_glob_routes_to_orchestrator(
        self, mock_orchestrate: MagicMock
    ) -> None:
        """Multi-file glob routes to the unified orchestrator."""
        mock_orchestrate.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_graph_files(Path(tmpdir), 3)
            glob_pattern = os.path.join(tmpdir, "*.json")

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", glob_pattern]):
                result = main()

            mock_orchestrate.assert_called_once()
            call_args = mock_orchestrate.call_args
            graph_paths = (
                call_args.args[1]
                if len(call_args.args) > 1
                else call_args.kwargs["graph_paths"]
            )
            assert len(graph_paths) == 3
            assert result == 0

    @patch("dnn_benchmarking.cli.main.run_suite_cli")
    def test_single_file_also_routes_to_orchestrator(
        self, mock_orchestrate: MagicMock
    ) -> None:
        """Single file routes through the unified orchestrator (no separate run_benchmark)."""
        mock_orchestrate.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", paths[0]]):
                result = main()

            mock_orchestrate.assert_called_once()
            call_args = mock_orchestrate.call_args
            graph_paths = (
                call_args.args[1]
                if len(call_args.args) > 1
                else call_args.kwargs["graph_paths"]
            )
            assert len(graph_paths) == 1
            assert result == 0

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark")
    def test_verbose_flag_propagates_to_suite_config(
        self, mock_benchmark: MagicMock
    ) -> None:
        """-v sets SuiteConfig.verbose=True when routing through the orchestrator."""
        mock_benchmark.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", paths[0], "-v"]):
                main()

        suite_config = mock_benchmark.call_args.kwargs["config"]
        assert suite_config.verbose is True

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark")
    def test_engine_list_propagates_to_suite_config(
        self, mock_benchmark: MagicMock
    ) -> None:
        """--engine 1,2 lands in SuiteConfig.engine_filter as [1, 2]."""
        mock_benchmark.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                ["dnn-benchmark", "--graph", paths[0], "--engine", "1,2"],
            ):
                main()

        suite_config = mock_benchmark.call_args.kwargs["config"]
        assert suite_config.engine_filter == [1, 2]

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark")
    def test_plugin_paths_propagate_to_suite_config(
        self, mock_benchmark: MagicMock
    ) -> None:
        mock_benchmark.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                [
                    "dnn-benchmark",
                    "--graph",
                    paths[0],
                    "--engine",
                    "2,1",
                    "--plugin-path",
                    "/plugins/b,/plugins/a",
                ],
            ):
                main()

        suite_config = mock_benchmark.call_args.kwargs["config"]
        assert suite_config.engine_filter == [2, 1]
        assert suite_config.plugin_paths == [Path("/plugins/b"), Path("/plugins/a")]

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark")
    def test_same_engine_plugin_paths_propagate_as_ordered_selections(
        self, mock_benchmark: MagicMock
    ) -> None:
        mock_benchmark.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                [
                    "dnn-benchmark",
                    "--graph",
                    paths[0],
                    "--engine",
                    "1,1",
                    "--plugin-path",
                    "/plugins/a,/plugins/b",
                ],
            ):
                main()

        suite_config = mock_benchmark.call_args.kwargs["config"]
        selections = suite_config.engine_selections_for(suite_config.engine_filter)
        assert suite_config.engine_filter == [1, 1]
        assert [s.plugin_path for s in selections] == [
            Path("/plugins/a"),
            Path("/plugins/b"),
        ]

    def test_plugin_path_count_mismatch_rejected_at_cli_layer(self) -> None:
        from dnn_benchmarking.cli.suite_runner_cli import run_suite_cli

        parser = create_parser()
        args = parser.parse_args(
            [
                "--graph",
                "g.json",
                "--engine",
                "1,2,3",
                "--plugin-path",
                "/plugins/a,/plugins/b",
            ]
        )
        reporter = MagicMock(spec=Reporter)

        rc = run_suite_cli(args, graph_paths=[Path("g.json")], reporter=reporter)

        assert rc == 1
        reporter.print_error.assert_called_once()
        assert "entry count" in reporter.print_error.call_args[0][0]

    @patch("dnn_benchmarking.cli.main.run_pytorch_cli")
    @patch("dnn_benchmarking.cli.main.run_suite_cli")
    def test_pytorch_backend_single_file_uses_pytorch_path(
        self,
        mock_orchestrate: MagicMock,
        mock_run_pytorch: MagicMock,
    ) -> None:
        """--backend pytorch on single file goes to run_pytorch_cli, not unified."""
        mock_run_pytorch.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                ["dnn-benchmark", "--graph", paths[0], "--backend", "pytorch"],
            ):
                result = main()

            mock_run_pytorch.assert_called_once()
            mock_orchestrate.assert_not_called()
            assert result == 0

    def test_pytorch_backend_multi_file_rejected(self) -> None:
        """--backend pytorch with a glob exits 1 (suite not supported)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_graph_files(Path(tmpdir), 3)
            glob_pattern = os.path.join(tmpdir, "*.json")

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                ["dnn-benchmark", "--graph", glob_pattern, "--backend", "pytorch"],
            ):
                result = main()

            assert result == 1

    @patch("dnn_benchmarking.cli.main.run_suite_cli")
    def test_recursive_glob_matches_nested_directories(
        self, mock_orchestrate: MagicMock
    ) -> None:
        """`**` glob with recursive=True matches graphs in nested directories."""
        mock_orchestrate.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create nested structure: tmpdir/a/g0.json, tmpdir/a/b/g1.json
            (root / "a" / "b").mkdir(parents=True)
            (root / "a" / "g0.json").write_text(
                json.dumps({"name": "g0", "nodes": [], "tensors": []})
            )
            (root / "a" / "b" / "g1.json").write_text(
                json.dumps({"name": "g1", "nodes": [], "tensors": []})
            )

            glob_pattern = os.path.join(tmpdir, "**", "*.json")

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", glob_pattern]):
                main()

            mock_orchestrate.assert_called_once()
            call_args = mock_orchestrate.call_args
            graph_paths = (
                call_args.args[1]
                if len(call_args.args) > 1
                else call_args.kwargs["graph_paths"]
            )
            assert len(graph_paths) == 2

    def test_zero_files_glob_returns_error(self) -> None:
        """When glob resolves to zero files, main() returns 1."""
        from dnn_benchmarking.cli.main import main

        with patch(
            "sys.argv",
            ["dnn-benchmark", "--graph", "/nonexistent/path/*.json"],
        ):
            result = main()

        assert result == 1


class TestRunSuiteWorkflow:
    """Tests for run_suite_benchmark() workflow — graph iteration, exit codes, JSON output."""

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
            x_uid = 1 + i * 10
            w_uid = 2 + i * 10
            y_uid = 100 + i
            p.write_text(
                json.dumps(
                    {
                        "name": f"graph_{i}",
                        "nodes": [
                            {
                                "type": "ConvolutionFwdAttributes",
                                "name": "conv",
                                "inputs": {
                                    "x_tensor_uid": x_uid,
                                    "w_tensor_uid": w_uid,
                                },
                                "outputs": {"y_tensor_uid": y_uid},
                                "parameters": {
                                    "conv_mode": "CROSS_CORRELATION",
                                    "pre_padding": [0, 0],
                                    "post_padding": [0, 0],
                                    "stride": [1, 1],
                                    "dilation": [1, 1],
                                },
                            }
                        ],
                        "tensors": [
                            {
                                "uid": x_uid,
                                "dims": [1, 3, 4, 4],
                                "strides": [48, 16, 4, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                            {
                                "uid": w_uid,
                                "dims": [3, 3, 1, 1],
                                "strides": [3, 1, 1, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                            {
                                "uid": y_uid,
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

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_all_pass_returns_zero_exit_code(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """All-passing graphs => exit 0."""
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

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            result = run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert result == 0

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_one_failure_still_processes_second(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """One failing graph still processes the second; exit code is 1."""
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

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            result = run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert mock_run.call_count == 2
        assert result == 1

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_correctness_failure_returns_two(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Correctness failure => exit code 2."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
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

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            config = SuiteConfig()
            result = run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert result == 2

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_json_output_written_when_output_specified(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite_benchmark writes JSON to --output path when specified."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            output_file = Path(tmpdir) / "results.json"
            config = SuiteConfig()
            run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=output_file,
                reporter=Reporter(),
            )

            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "metadata" in data
            assert "graphs" in data

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_no_json_output_when_output_not_specified(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite_benchmark does not write JSON when --output is not specified."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = self._make_graph_files(tmp_path, 1)
            inputs_before = {p.resolve() for p in tmp_path.rglob("*.json")}

            config = SuiteConfig()
            run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

            inputs_after = {p.resolve() for p in tmp_path.rglob("*.json")}
            new_files = inputs_after - inputs_before
            assert new_files == set(), f"Unexpected JSON files written: {new_files}"

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_warmup_iters_passed_per_graph(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Warmup and benchmark iterations apply per graph independently."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig(warmup_iters=20, benchmark_iters=200)
            run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert mock_run.call_count == 2
        for call_args in mock_run.call_args_list:
            passed_config = call_args[0][3]
            assert passed_config.warmup_iters == 20
            assert passed_config.benchmark_iters == 200

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_empty_nodes_graph_records_error_and_continues(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """Graph with no operation nodes fails loader.validate(), which produces
        an error entry and the suite continues to the next graph."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g_good")

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty_graph.json"
            empty_file.write_text(
                json.dumps({"name": "empty_graph", "nodes": [], "tensors": []})
            )

            good_paths = self._make_graph_files(Path(tmpdir), 1)
            paths = [empty_file] + good_paths
            config = SuiteConfig()
            result = run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert mock_run.call_count == 1
        assert result == 1

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_graph_load_error_continues_to_next(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite_benchmark catches GraphLoadError per graph and continues."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g1")

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "bad_graph.json"
            bad_file.write_text("{invalid json")

            good_paths = self._make_graph_files(Path(tmpdir), 1)
            paths = [bad_file] + good_paths
            config = SuiteConfig()
            result = run_suite_benchmark(
                graph_paths=paths,
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert mock_run.call_count == 1
        assert result == 1


class TestBackendEngineRouting:
    """Tests for engine selection rules across execution backends."""

    def _create_graph(self, tmpdir: Path) -> Path:
        p = tmpdir / "g.json"
        p.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
        return p

    def test_engine_list_with_pytorch_backend_rejected(self) -> None:
        from dnn_benchmarking.cli.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = self._create_graph(Path(tmpdir))
            with patch(
                "sys.argv",
                [
                    "dnn-benchmark",
                    "--graph",
                    str(graph),
                    "--engine",
                    "1,2",
                    "--backend",
                    "pytorch",
                ],
            ):
                result = main()
        assert result == 1

    @patch("dnn_benchmarking.cli.pytorch_runner_cli.run_pytorch_benchmark")
    def test_single_engine_with_pytorch_backend_accepted(
        self, mock_run_pytorch: MagicMock
    ) -> None:
        """A single --engine ID is fine with --backend pytorch."""
        mock_run_pytorch.return_value = 0
        from dnn_benchmarking.cli.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = self._create_graph(Path(tmpdir))
            with patch(
                "sys.argv",
                [
                    "dnn-benchmark",
                    "--graph",
                    str(graph),
                    "--engine",
                    "3",
                    "--backend",
                    "pytorch",
                ],
            ):
                result = main()
        assert result == 0
        mock_run_pytorch.assert_called_once()
        cfg = mock_run_pytorch.call_args.args[0]
        assert cfg.engine_id == 3


class TestValidationStartupGate:
    """--validate is a hard gate. If the reference provider isn't registered
    or available, the orchestrator returns 1 before iterating any graph."""

    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    def test_unregistered_reference_provider_fails_at_startup(
        self, mock_run: MagicMock
    ) -> None:

        config = SuiteConfig.__new__(SuiteConfig)
        # Bypass the SuiteConfig validator (which restricts reference_provider
        # to the known set) so we can simulate an unregistered name.
        config.warmup_iters = 1
        config.benchmark_iters = 1
        config.seed = None
        config.engine_filter = None
        config.rtol = 1e-5
        config.atol = 1e-8
        config.gpu_backend = "none"
        config.reference_provider = "definitely_not_registered"
        config.verbose = False

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "g.json"
            graph.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
            result = run_suite_benchmark(
                graph_paths=[graph],
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert result == 1
        # Startup gate fires before any graph runs.
        mock_run.assert_not_called()

    @patch("dnn_benchmarking.cli.suite_runner_cli.ReferenceProviderRegistry")
    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    def test_unavailable_reference_provider_fails_at_startup(
        self, mock_run: MagicMock, mock_registry: MagicMock
    ) -> None:

        provider_mock = MagicMock()
        provider_mock.is_available.return_value = False
        mock_registry.get_provider.return_value = provider_mock

        config = SuiteConfig(reference_provider="pytorch")

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "g.json"
            graph.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
            result = run_suite_benchmark(
                graph_paths=[graph],
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert result == 1
        mock_run.assert_not_called()

    @patch("dnn_benchmarking.cli.suite_runner_cli.ReferenceProviderRegistry")
    @patch("dnn_benchmarking.cli.suite_runner_cli.run_graph_all_providers")
    @patch("dnn_benchmarking.reporting.suite_results.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_available_reference_provider_proceeds_to_graph_iteration(
        self,
        mock_env: MagicMock,
        mock_run: MagicMock,
        mock_registry: MagicMock,
    ) -> None:

        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        provider_mock = MagicMock()
        provider_mock.is_available.return_value = True
        mock_registry.get_provider.return_value = provider_mock

        # Successful benchmark with passing correctness
        mock_run.return_value = GraphResult(
            graph_name="g",
            graph_path="/tmp/g.json",
            results=[
                ProviderEngineResult(
                    provider="MIOPEN_ENGINE",
                    engine_id=0,
                    status="success",
                    cpu_build_time_ms=1.0,
                    correctness=CorrectnessResult(
                        execution_success=True,
                        tolerance_match=True,
                        rtol=1e-5,
                        atol=1e-8,
                    ),
                )
            ],
        )

        config = SuiteConfig(reference_provider="pytorch")

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "g.json"
            graph.write_text(
                json.dumps(
                    {
                        "name": "g",
                        "nodes": [
                            {
                                "type": "ConvolutionFwdAttributes",
                                "name": "conv",
                                "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                                "outputs": {"y_tensor_uid": 100},
                                "parameters": {
                                    "conv_mode": "CROSS_CORRELATION",
                                    "pre_padding": [0, 0],
                                    "post_padding": [0, 0],
                                    "stride": [1, 1],
                                    "dilation": [1, 1],
                                },
                            }
                        ],
                        "tensors": [
                            {
                                "uid": 1,
                                "dims": [1, 3, 4, 4],
                                "strides": [48, 16, 4, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                            {
                                "uid": 2,
                                "dims": [3, 3, 1, 1],
                                "strides": [3, 1, 1, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                            {
                                "uid": 100,
                                "dims": [1, 3, 4, 4],
                                "strides": [48, 16, 4, 1],
                                "data_type": "FLOAT",
                                "is_virtual": False,
                            },
                        ],
                    }
                )
            )
            result = run_suite_benchmark(
                graph_paths=[graph],
                config=config,
                output_path=None,
                reporter=Reporter(),
            )

        assert result == 0
        mock_run.assert_called_once()


class TestProfilingFlagParsing:
    """Argparse-layer coverage for the seven profiling flags. The
    behavioural rejection of `--pmc all` without `--pmc-allow-multipass`
    lives at the MetricsConfig layer (tests in test_metrics_config.py)
    and is re-asserted here at the CLI boundary so a regression in the
    propagation between argparse → MetricsConfig is caught."""

    def test_pmc_flag_accepts_named_sets(self):
        parser = create_parser()
        for set_name in ("basic", "memory", "flops", "all"):
            args = parser.parse_args(["--graph", "g.json", "--pmc", set_name])
            assert args.pmc == set_name

    def test_pmc_flag_rejects_unknown_set(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--graph", "g.json", "--pmc", "bogus"])

    def test_pmc_allow_multipass_default_false(self):
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json"])
        assert args.pmc_allow_multipass is False

    def test_pmc_allow_multipass_sets_true(self):
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--pmc-allow-multipass"])
        assert args.pmc_allow_multipass is True

    def test_emit_trace_accepts_pftrace_and_kineto(self):
        parser = create_parser()
        for fmt in ("pftrace", "kineto"):
            args = parser.parse_args(["--graph", "g.json", "--emit-trace", fmt])
            assert args.emit_trace == fmt

    def test_emit_trace_rejects_unknown_format(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--graph", "g.json", "--emit-trace", "bogus"])

    def test_perf_flag_default_false_store_true_when_set(self):
        parser = create_parser()
        assert parser.parse_args(["--graph", "g.json"]).perf is False
        assert parser.parse_args(["--graph", "g.json", "--perf"]).perf is True

    def test_roofline_flag_default_false_store_true_when_set(self):
        parser = create_parser()
        assert parser.parse_args(["--graph", "g.json"]).roofline is False
        assert parser.parse_args(["--graph", "g.json", "--roofline"]).roofline is True

    def test_roofline_data_type_flag_does_not_exist(self):
        """``--roofline-data-type`` was removed because it's only valid on
        ``rocprof-compute analyze``, not ``profile``. Anyone trying to
        pass it should get an argparse error, not a silent acceptance."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--graph", "g.json", "--roofline-data-type", "FP16"])

    def test_profiling_output_dir_default_none_accepts_path(self, tmp_path):
        parser = create_parser()
        assert parser.parse_args(["--graph", "g.json"]).profiling_output_dir is None
        args = parser.parse_args(
            ["--graph", "g.json", "--profiling-output-dir", str(tmp_path / "out")]
        )
        assert args.profiling_output_dir == Path(str(tmp_path / "out"))

    def test_profiling_timeout_default_and_override(self):
        """600s default matches MetricsConfig.profiling_timeout_s default
        — if these drift, a user-passed flag could silently revert to a
        different per-source value."""
        parser = create_parser()
        assert parser.parse_args(["--graph", "g.json"]).profiling_timeout == 600
        args = parser.parse_args(["--graph", "g.json", "--profiling-timeout", "1800"])
        assert args.profiling_timeout == 1800
        # 0 is the documented "disable timeout" sentinel and must parse.
        args = parser.parse_args(["--graph", "g.json", "--profiling-timeout", "0"])
        assert args.profiling_timeout == 0


class TestProfilingFlagPropagation:
    """All seven profiling flags must round-trip from argparse Namespace
    into MetricsConfig the way `run_suite_cli` constructs it. Anything
    that drops or renames a flag here would silently disable a CLI
    option without test failure."""

    def _make_args(self, **overrides):
        parser = create_parser()
        cli = ["--graph", "g.json"]
        for k, v in overrides.items():
            if isinstance(v, bool):
                if v:
                    cli.append(k)
            else:
                cli.extend([k, str(v)])
        return parser.parse_args(cli)

    def test_all_flags_set_lands_in_metrics_config(self, tmp_path, monkeypatch):
        """Drive the full propagation chain: argparse Namespace ->
        run_suite_cli -> MetricsConfig fields. We stub
        run_suite_benchmark so this stays a pure unit test (no graph
        files, no rocprofv3 binary)."""
        from dnn_benchmarking.cli.suite_runner_cli import run_suite_cli
        from dnn_benchmarking.reporting.reporter import Reporter

        args = self._make_args(
            **{
                "--warmup": 1,
                "--iters": 1,
                "--pmc": "memory",
                "--pmc-allow-multipass": True,
                "--emit-trace": "pftrace",
                "--perf": True,
                "--roofline": True,
                "--profiling-output-dir": str(tmp_path / "out"),
                "--profiling-timeout": 1234,
            }
        )

        captured = {}

        def fake_run(config, **kwargs):
            captured["metrics"] = config.metrics
            return 0

        with patch(
            "dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark",
            side_effect=lambda graph_paths, config, **kw: fake_run(
                config, graph_paths=graph_paths, **kw
            ),
        ):
            rc = run_suite_cli(
                args,
                graph_paths=[Path("g.json")],
                reporter=Reporter(),
            )
        assert rc == 0
        m = captured["metrics"]
        assert m.pmc_set == "memory"
        assert m.pmc_allow_multipass is True
        assert m.emit_trace == "pftrace"
        assert m.perf is True
        assert m.roofline is True
        assert m.profiling_output_dir == Path(str(tmp_path / "out"))
        assert m.profiling_timeout_s == 1234

    def test_pmc_all_without_multipass_rejected_at_cli_layer(self):
        """The dataclass-level check in MetricsConfig.__post_init__ must
        bubble through run_suite_cli's ValueError handler as exit-code 1
        plus a reporter error. Catches regressions where the propagation
        site swallows the dataclass error or routes around the gate."""
        from dnn_benchmarking.cli.suite_runner_cli import run_suite_cli
        from dnn_benchmarking.reporting.reporter import Reporter

        args = self._make_args(
            **{
                "--warmup": 1,
                "--iters": 1,
                "--pmc": "all",
                # No --pmc-allow-multipass: MetricsConfig.__post_init__ raises.
            }
        )

        reporter = MagicMock(spec=Reporter)
        rc = run_suite_cli(args, graph_paths=[Path("g.json")], reporter=reporter)
        assert rc == 1
        reporter.print_error.assert_called_once()
        err_text = reporter.print_error.call_args[0][0]
        assert "Suite configuration error" in err_text
        assert "multipass" in err_text.lower() or "pmc" in err_text.lower()

    def test_pmc_all_with_multipass_accepted_at_cli_layer(self):
        """Positive control for the previous test."""
        from dnn_benchmarking.cli.suite_runner_cli import run_suite_cli
        from dnn_benchmarking.reporting.reporter import Reporter

        args = self._make_args(
            **{
                "--warmup": 1,
                "--iters": 1,
                "--pmc": "all",
                "--pmc-allow-multipass": True,
            }
        )

        with patch(
            "dnn_benchmarking.cli.suite_runner_cli.run_suite_benchmark",
            return_value=0,
        ):
            rc = run_suite_cli(args, graph_paths=[Path("g.json")], reporter=Reporter())
        assert rc == 0
