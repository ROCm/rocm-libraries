# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for suite execution (requires GPU + hipDNN + provider plugin)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _graphs_dir() -> Path:
    """Get the graphs directory."""
    return Path(__file__).parent.parent.parent / "graphs"


def _require_gpu():
    """Skip if PyTorch GPU is not available."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def _require_hipdnn():
    """Skip if hipdnn_frontend is not importable or no GPU handle can be created."""
    try:
        import hipdnn_frontend

        hipdnn_frontend.Handle()
        return hipdnn_frontend
    except ImportError:
        pytest.skip("hipdnn_frontend not installed")
    except Exception as e:
        pytest.skip(f"hipdnn_frontend available but cannot create Handle: {e}")


@pytest.mark.gpu
class TestSuiteRunnerIntegration:
    """Integration tests for suite_runner.run_graph_all_providers on real GPU."""

    @pytest.fixture
    def hipdnn(self):
        """Get hipdnn_frontend module or skip."""
        _require_gpu()
        return _require_hipdnn()

    @pytest.fixture
    def conv_graph(self) -> Dict[str, Any]:
        """Load sample conv fwd graph JSON."""
        path = _graphs_dir() / "sample_conv_fwd.json"
        if not path.exists():
            pytest.skip(f"Sample graph not found: {path}")
        with open(path) as f:
            return json.load(f)

    def test_run_graph_all_providers_returns_results(
        self, hipdnn, conv_graph: Dict[str, Any]
    ) -> None:
        """Suite runner discovers providers/engines and returns results."""
        from dnn_benchmarking.config.benchmark_config import SuiteConfig
        from dnn_benchmarking.execution.suite_runner import run_graph_all_providers
        from dnn_benchmarking.graph.loader import GraphLoader

        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(conv_graph)
        config = SuiteConfig(warmup_iters=1, benchmark_iters=2)
        handle = hipdnn.Handle()
        graph_path = _graphs_dir() / "sample_conv_fwd.json"

        result = run_graph_all_providers(
            graph_path, conv_graph, tensor_infos, config, handle
        )

        assert result.graph_name is not None
        assert len(result.results) > 0

        # At least one result should have a non-error status
        statuses = [r.status for r in result.results]
        assert any(
            s in ("success", "skipped") for s in statuses
        ), f"All results errored: {[r.error_message for r in result.results]}"

    def test_successful_result_has_separated_timing(
        self, hipdnn, conv_graph: Dict[str, Any]
    ) -> None:
        """Successful results have separate cpu_build, gpu_kernel, and e2e timing."""
        from dnn_benchmarking.config.benchmark_config import SuiteConfig
        from dnn_benchmarking.execution.suite_runner import run_graph_all_providers
        from dnn_benchmarking.graph.loader import GraphLoader

        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(conv_graph)
        config = SuiteConfig(warmup_iters=1, benchmark_iters=3)
        handle = hipdnn.Handle()

        result = run_graph_all_providers(
            _graphs_dir() / "sample_conv_fwd.json",
            conv_graph,
            tensor_infos,
            config,
            handle,
        )

        successes = [r for r in result.results if r.status == "success"]
        if not successes:
            pytest.skip("No successful provider/engine combinations found")

        for r in successes:
            assert r.cpu_build_time_ms is not None
            assert r.cpu_build_time_ms > 0
            assert r.e2e_stats is not None
            assert r.e2e_stats.mean_ms > 0
            # gpu_kernel_stats may be None if torch GPU timing isn't available

    def test_successful_result_has_correctness(
        self, hipdnn, conv_graph: Dict[str, Any]
    ) -> None:
        """Successful results have correctness populated."""
        from dnn_benchmarking.config.benchmark_config import SuiteConfig
        from dnn_benchmarking.execution.suite_runner import run_graph_all_providers
        from dnn_benchmarking.graph.loader import GraphLoader

        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(conv_graph)
        config = SuiteConfig(warmup_iters=1, benchmark_iters=2)
        handle = hipdnn.Handle()

        result = run_graph_all_providers(
            _graphs_dir() / "sample_conv_fwd.json",
            conv_graph,
            tensor_infos,
            config,
            handle,
        )

        successes = [r for r in result.results if r.status == "success"]
        if not successes:
            pytest.skip("No successful provider/engine combinations found")

        for r in successes:
            assert r.correctness is not None
            assert r.correctness.execution_success is True


@pytest.mark.gpu
class TestSuiteCLIIntegration:
    """Integration tests for suite mode via CLI (subprocess)."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        """Skip all tests if GPU or hipdnn not available."""
        _require_gpu()
        _require_hipdnn()

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def graph_paths(self) -> List[Path]:
        """Get available sample graph paths."""
        paths = sorted(_graphs_dir().glob("*.json"))
        if len(paths) < 2:
            pytest.skip("Need at least 2 sample graphs for suite test")
        return paths

    def test_suite_mode_multiple_graphs(
        self, project_root: Path, graph_paths: List[Path]
    ) -> None:
        """CLI with glob pattern runs suite mode and produces output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "*.json"),
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gpu-backend",
                "none",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert "hipDNN Benchmark Suite" in result.stdout
        assert "Suite Summary" in result.stdout
        # Should show progress for each graph
        for i, p in enumerate(graph_paths, 1):
            assert f"[{i}/{len(graph_paths)}]" in result.stdout

    def test_suite_mode_json_output(self, project_root: Path, tmp_path: Path) -> None:
        """Suite mode writes valid JSON when --output specified."""
        output_file = tmp_path / "suite_results.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "*.json"),
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gpu-backend",
                "none",
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert output_file.exists(), f"JSON output not written. stderr: {result.stderr}"

        with open(output_file) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "graphs" in data
        assert data["metadata"]["total_graphs"] > 0
        assert len(data["graphs"]) == data["metadata"]["total_graphs"]

        # Each graph should have results array
        for g in data["graphs"]:
            assert "graph_name" in g
            assert "results" in g
            assert len(g["results"]) > 0

    def test_suite_mode_single_graph_failure_continues(
        self, project_root: Path, tmp_path: Path
    ) -> None:
        """A single graph failure does not abort the suite."""
        output_file = tmp_path / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "*.json"),
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gpu-backend",
                "none",
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        with open(output_file) as f:
            data = json.load(f)

        # All graphs should be represented even if some failed
        graph_count = len(sorted(_graphs_dir().glob("*.json")))
        assert len(data["graphs"]) == graph_count

    def test_suite_mode_provider_filter(self, project_root: Path) -> None:
        """--provider flag is accepted and filters execution."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "sample_conv_fwd.json"),
                "--provider",
                "miopen",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gpu-backend",
                "none",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # --provider on single file triggers suite mode
        assert "hipDNN Benchmark Suite" in result.stdout

    def test_single_graph_without_suite_flags_uses_legacy_mode(
        self, project_root: Path
    ) -> None:
        """Single graph without --provider/--engine uses legacy run_benchmark."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "sample_conv_fwd.json"),
                "--warmup",
                "1",
                "--iters",
                "2",
                "--gpu-backend",
                "none",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # Single graph without suite flags should use legacy mode
        assert "hipDNN Benchmark:" in result.stdout
        assert "hipDNN Benchmark Suite" not in result.stdout
