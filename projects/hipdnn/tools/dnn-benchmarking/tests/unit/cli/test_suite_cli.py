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
)


def _mock_hipdnn():
    """Create a mock hipdnn_frontend module with a Handle class."""
    mock_module = ModuleType("hipdnn_frontend")
    mock_module.Handle = MagicMock  # type: ignore[attr-defined]
    return mock_module


class TestParserGlobAndFilters:
    """Tests for --graph glob pattern and --engine filter flags."""

    def test_graph_accepts_glob_pattern_string(self) -> None:
        """--graph accepts a glob pattern string and stores as-is."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "graphs/*.json"])
        assert isinstance(args.graph, str)
        assert args.graph == "graphs/*.json"

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

    def test_engine_flag_deduplicates_preserving_order(self) -> None:
        """--engine 1,1,1 -> [1]; '3,1,3,2' -> [3, 1, 2] (first-seen order)."""
        parser = create_parser()
        args = parser.parse_args(["--graph", "g.json", "--engine", "1,1,1"])
        assert args.engine == [1]

        args = parser.parse_args(["--graph", "g.json", "--engine", "3,1,3,2"])
        assert args.engine == [3, 1, 2]

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

    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
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
            kwargs = mock_orchestrate.call_args.kwargs
            assert len(kwargs["graph_paths"]) == 3
            assert result == 0

    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
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
            kwargs = mock_orchestrate.call_args.kwargs
            assert len(kwargs["graph_paths"]) == 1
            assert result == 0

    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
    def test_verbose_flag_propagates_to_suite_config(
        self, mock_orchestrate: MagicMock
    ) -> None:
        """-v sets SuiteConfig.verbose=True when routing through the orchestrator."""
        mock_orchestrate.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch("sys.argv", ["dnn-benchmark", "--graph", paths[0], "-v"]):
                main()

        suite_config = mock_orchestrate.call_args.kwargs["config"]
        assert suite_config.verbose is True

    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
    def test_engine_list_propagates_to_suite_config(
        self, mock_orchestrate: MagicMock
    ) -> None:
        """--engine 1,2 lands in SuiteConfig.engine_filter as [1, 2]."""
        mock_orchestrate.return_value = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_graph_files(Path(tmpdir), 1)

            from dnn_benchmarking.cli.main import main

            with patch(
                "sys.argv",
                ["dnn-benchmark", "--graph", paths[0], "--engine", "1,2"],
            ):
                main()

        suite_config = mock_orchestrate.call_args.kwargs["config"]
        assert suite_config.engine_filter == [1, 2]

    @patch("dnn_benchmarking.cli.main.run_pytorch_benchmark")
    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
    def test_pytorch_backend_single_file_uses_pytorch_path(
        self, mock_orchestrate: MagicMock, mock_run_pytorch: MagicMock
    ) -> None:
        """--backend pytorch on single file goes to run_pytorch_benchmark, not unified."""
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

    @patch("dnn_benchmarking.cli.main._orchestrate_suite_cli")
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
            kwargs = mock_orchestrate.call_args.kwargs
            # Both nested files should have been resolved
            assert len(kwargs["graph_paths"]) == 2

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
    """Tests for run_suite() pure-runner behavior and the CLI orchestrator."""

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
    def test_all_pass_returns_zero_exit_code(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite() + _suite_exit_code: all-passing graphs => exit 0."""
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

        from dnn_benchmarking.cli.main import run_suite, _suite_exit_code

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            suite_result = run_suite(paths, config, handle=MagicMock())

        assert _suite_exit_code(suite_result) == 0

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
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

        from dnn_benchmarking.cli.main import run_suite, _suite_exit_code

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig()
            suite_result = run_suite(paths, config, handle=MagicMock())

        # Both graphs were processed
        assert mock_run.call_count == 2
        # Exit code is 1 because one had errors
        assert _suite_exit_code(suite_result) == 1

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
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

        from dnn_benchmarking.cli.main import run_suite, _suite_exit_code

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            config = SuiteConfig()
            suite_result = run_suite(paths, config, handle=MagicMock())

        assert _suite_exit_code(suite_result) == 2

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_json_output_written_when_output_specified(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """The orchestrator writes JSON to --output path when specified."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        from dnn_benchmarking.cli.main import _orchestrate_suite_cli

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 1)
            output_file = Path(tmpdir) / "results.json"
            config = SuiteConfig()
            _orchestrate_suite_cli(
                graph_paths=paths,
                config=config,
                output_path=output_file,
                plugin_path=None,
            )

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
        """The orchestrator does not write JSON when --output is not specified."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g0")

        from dnn_benchmarking.cli.main import _orchestrate_suite_cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            paths = self._make_graph_files(tmp_path, 1)
            # Snapshot the JSON files that exist before the run (input graphs)
            inputs_before = {p.resolve() for p in tmp_path.rglob("*.json")}

            config = SuiteConfig()
            _orchestrate_suite_cli(
                graph_paths=paths,
                config=config,
                output_path=None,
                plugin_path=None,
            )

            # Assert no NEW *.json files were created in tmpdir
            inputs_after = {p.resolve() for p in tmp_path.rglob("*.json")}
            new_files = inputs_after - inputs_before
            assert new_files == set(), f"Unexpected JSON files written: {new_files}"

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
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

        from dnn_benchmarking.cli.main import run_suite

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._make_graph_files(Path(tmpdir), 2)
            config = SuiteConfig(warmup_iters=20, benchmark_iters=200)
            run_suite(paths, config, handle=MagicMock())

        # Each graph gets the same config (warmup/iters applied per graph)
        assert mock_run.call_count == 2
        for call_args in mock_run.call_args_list:
            passed_config = call_args[0][3]  # 4th positional arg is config
            assert passed_config.warmup_iters == 20
            assert passed_config.benchmark_iters == 200

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
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

        from dnn_benchmarking.cli.main import run_suite, _suite_exit_code

        with tempfile.TemporaryDirectory() as tmpdir:
            # First graph has empty nodes -> validator should reject it
            empty_file = Path(tmpdir) / "empty_graph.json"
            empty_file.write_text(
                json.dumps({"name": "empty_graph", "nodes": [], "tensors": []})
            )

            good_paths = self._make_graph_files(Path(tmpdir), 1)
            paths = [empty_file] + good_paths
            config = SuiteConfig()
            suite_result = run_suite(paths, config, handle=MagicMock())

        # Good graph still got processed (run_graph_all_providers called once).
        assert mock_run.call_count == 1
        # Empty graph triggered an error path -> exit code 1
        assert _suite_exit_code(suite_result) == 1

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    def test_graph_load_error_continues_to_next(
        self, mock_env: MagicMock, mock_run: MagicMock
    ) -> None:
        """run_suite() catches GraphLoadError per graph and continues."""
        mock_env.return_value = {
            "rocm_version": None,
            "gpu_model": None,
            "python_version": "3.10.0",
            "hipdnn_version": None,
        }
        mock_run.return_value = self._make_graph_result("g1")

        from dnn_benchmarking.cli.main import run_suite, _suite_exit_code

        with tempfile.TemporaryDirectory() as tmpdir:
            # First file has invalid JSON
            bad_file = Path(tmpdir) / "bad_graph.json"
            bad_file.write_text("{invalid json")

            good_paths = self._make_graph_files(Path(tmpdir), 1)
            paths = [bad_file] + good_paths
            config = SuiteConfig()
            suite_result = run_suite(paths, config, handle=MagicMock())

        # Should still process the second graph
        assert mock_run.call_count == 1
        # Exit code is 1 because of the error on first graph
        assert _suite_exit_code(suite_result) == 1


class TestEngineFlagModeRejection:
    """--engine list is incompatible with A/B and PyTorch single-engine modes."""

    def _create_graph(self, tmpdir: Path) -> Path:
        p = tmpdir / "g.json"
        p.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
        return p

    def test_engine_list_with_ab_mode_rejected(self) -> None:
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
                    "--AId",
                    "1",
                    "--BId",
                    "2",
                ],
            ):
                result = main()
        assert result == 1

    def test_single_engine_with_ab_mode_also_rejected(self) -> None:
        """Even a single-element --engine list is rejected in A/B (it has --AId/--BId)."""
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
                    "5",
                    "--AId",
                    "1",
                    "--BId",
                    "2",
                ],
            ):
                result = main()
        assert result == 1

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

    @patch("dnn_benchmarking.cli.main.run_pytorch_benchmark")
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

    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    def test_unregistered_reference_provider_fails_at_startup(
        self, mock_run: MagicMock
    ) -> None:
        from dnn_benchmarking.cli.main import _orchestrate_suite_cli

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
            result = _orchestrate_suite_cli(
                graph_paths=[graph],
                config=config,
                output_path=None,
                plugin_path=None,
            )

        assert result == 1
        # Startup gate fires before any graph runs.
        mock_run.assert_not_called()

    @patch("dnn_benchmarking.cli.main.ReferenceProviderRegistry")
    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    def test_unavailable_reference_provider_fails_at_startup(
        self, mock_run: MagicMock, mock_registry: MagicMock
    ) -> None:
        from dnn_benchmarking.cli.main import _orchestrate_suite_cli

        provider_mock = MagicMock()
        provider_mock.is_available.return_value = False
        mock_registry.get_provider.return_value = provider_mock

        config = SuiteConfig(reference_provider="pytorch")

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = Path(tmpdir) / "g.json"
            graph.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))
            result = _orchestrate_suite_cli(
                graph_paths=[graph],
                config=config,
                output_path=None,
                plugin_path=None,
            )

        assert result == 1
        mock_run.assert_not_called()

    @patch("dnn_benchmarking.cli.main.ReferenceProviderRegistry")
    @patch("dnn_benchmarking.cli.main.run_graph_all_providers")
    @patch("dnn_benchmarking.cli.main.collect_environment_info")
    @patch.dict(sys.modules, {"hipdnn_frontend": _mock_hipdnn()})
    def test_available_reference_provider_proceeds_to_graph_iteration(
        self,
        mock_env: MagicMock,
        mock_run: MagicMock,
        mock_registry: MagicMock,
    ) -> None:
        from dnn_benchmarking.cli.main import _orchestrate_suite_cli

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
                                "op_type": "ConvolutionForward",
                                "inputs": {},
                                "outputs": {"y": 100},
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
            result = _orchestrate_suite_cli(
                graph_paths=[graph],
                config=config,
                output_path=None,
                plugin_path=None,
            )

        assert result == 0
        mock_run.assert_called_once()
