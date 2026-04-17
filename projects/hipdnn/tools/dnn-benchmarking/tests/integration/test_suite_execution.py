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


def _find_plugin_path():
    """Find the hipDNN engine plugin directory."""
    project_root = Path(__file__).parent.parent.parent
    candidates = [
        project_root.parent.parent.parent.parent
        / "dnn-providers"
        / "miopen-provider"
        / "build"
        / "lib"
        / "hipdnn_plugins"
        / "engines",
        Path("/opt/rocm/lib/hipdnn_plugins/engines"),
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("*.so")):
            return str(p)
    return None


def _require_hipdnn():
    """Skip if hipdnn_frontend is not importable or no GPU handle can be created."""
    try:
        import hipdnn_frontend

        plugin_path = _find_plugin_path()
        if plugin_path is not None:
            hipdnn_frontend.set_engine_plugin_paths([plugin_path])

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
    def cli_plugin_args(self) -> List[str]:
        """Return --plugin-path args for CLI, or empty list."""
        path = _find_plugin_path()
        if path is None:
            return []
        return ["--plugin-path", path]

    @pytest.fixture
    def graph_paths(self) -> List[Path]:
        """Get available sample graph paths."""
        paths = sorted(_graphs_dir().glob("*.json"))
        if len(paths) < 2:
            pytest.skip("Need at least 2 sample graphs for suite test")
        return paths

    def test_suite_mode_multiple_graphs(
        self, project_root: Path, graph_paths: List[Path], cli_plugin_args: List[str]
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
                "--no-kernel-timing",
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
        assert "hipDNN Benchmark Suite" in result.stdout
        assert "Suite Summary" in result.stdout
        # Should show progress for each graph
        for i, p in enumerate(graph_paths, 1):
            assert f"[{i}/{len(graph_paths)}]" in result.stdout

    def test_suite_mode_json_output(
        self, project_root: Path, tmp_path: Path, cli_plugin_args: List[str]
    ) -> None:
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
                "--no-kernel-timing",
                "--output",
                str(output_file),
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
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
        self, project_root: Path, tmp_path: Path, cli_plugin_args: List[str]
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
                "--no-kernel-timing",
                "--output",
                str(output_file),
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"

        with open(output_file) as f:
            data = json.load(f)

        # All graphs should be represented even if some failed
        graph_count = len(sorted(_graphs_dir().glob("*.json")))
        assert len(data["graphs"]) == graph_count

    def test_single_graph_uses_unified_suite_path(
        self, project_root: Path, cli_plugin_args: List[str]
    ) -> None:
        """Single graph now flows through the unified suite path (default summary)."""
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
                "--no-kernel-timing",
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
        # Suite header is now used for single-graph too
        assert "hipDNN Benchmark Suite" in result.stdout
        assert "Suite Summary" in result.stdout

    def test_single_graph_verbose_renders_rich_block(
        self, project_root: Path, cli_plugin_args: List[str]
    ) -> None:
        """Single graph + -v renders the legacy single-graph rich format inside the suite wrapper."""
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
                "--no-kernel-timing",
                "-v",
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
        # Suite wrapper still present
        assert "hipDNN Benchmark Suite" in result.stdout
        # Rich per-engine block also present
        assert "hipDNN Benchmark:" in result.stdout
        assert "Execution Statistics" in result.stdout

    def test_engine_comma_list_runs_multiple_engines(
        self, project_root: Path, tmp_path: Path, cli_plugin_args: List[str]
    ) -> None:
        """--engine 0,1 produces results for both engines (when both are discovered)."""
        output_file = tmp_path / "multi_engine.json"
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
                "--no-kernel-timing",
                "--engine",
                "0,1",
                "--output",
                str(output_file),
            ]
            + cli_plugin_args,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
            2,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
        if not output_file.exists():
            pytest.skip(
                "No JSON output written; provider/engine discovery may be empty"
            )

        data = json.loads(output_file.read_text())
        assert len(data["graphs"]) == 1
        engine_ids = sorted(r["engine_id"] for r in data["graphs"][0]["results"])
        # The discovery on this system may produce just [1] or [0,1] depending on plugin.
        # We require at least one of the requested IDs landed.
        assert any(
            eid in (0, 1) for eid in engine_ids
        ), f"Expected engines 0 or 1, got {engine_ids}"

    def test_pytorch_backend_single_graph_still_works(self, project_root: Path) -> None:
        """--backend pytorch on a single graph remains on the dedicated PyTorch path."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("PyTorch GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "sample_conv_fwd.json"),
                "--backend",
                "pytorch",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--no-kernel-timing",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert result.returncode in (
            0,
            1,
        ), f"Unexpected exit code {result.returncode}. stderr: {result.stderr}"
        # PyTorch path uses its own header
        assert "PyTorch CUDA Benchmark" in result.stdout
        # And specifically does NOT use the suite wrapper
        assert "hipDNN Benchmark Suite" not in result.stdout

    def test_pytorch_backend_multi_graph_rejected(self, project_root: Path) -> None:
        """--backend pytorch with a glob exits 1 with the not-supported message."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(_graphs_dir() / "*.json"),
                "--backend",
                "pytorch",
                "--warmup",
                "1",
                "--iters",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 1
        assert "not supported with --backend pytorch" in result.stderr
