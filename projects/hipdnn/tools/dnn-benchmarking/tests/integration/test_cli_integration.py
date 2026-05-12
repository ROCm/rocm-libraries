# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIIntegration:
    """Integration tests for CLI invocation."""

    @pytest.fixture
    def sample_graph_path(self) -> Path:
        """Get path to sample graph."""
        return Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"

    def test_cli_help(self) -> None:
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "dnn_benchmarking", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "dnn-benchmark" in result.stdout or "dnn_benchmarking" in result.stdout
        assert "--graph" in result.stdout
        assert "--warmup" in result.stdout
        assert "--iters" in result.stdout

    def test_cli_missing_required_arg(self) -> None:
        """Test that missing --graph arg gives error."""
        result = subprocess.run(
            [sys.executable, "-m", "dnn_benchmarking"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "graph" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_cli_nonexistent_graph(self) -> None:
        """Test error handling for nonexistent graph file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                "/nonexistent/path/graph.json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "no graph files found" in combined or "error" in combined

    @pytest.mark.gpu
    def test_cli_full_run(self, sample_graph_path: Path, plugin_path_cli_args) -> None:
        """Test full CLI run with sample graph (requires GPU)."""
        if not sample_graph_path.exists():
            pytest.skip(f"Sample graph not found: {sample_graph_path}")

        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("PyTorch GPU not available")
        except ImportError as e:
            pytest.skip(f"PyTorch not available: {e}")

        # Check if hipdnn is available
        try:
            import hipdnn_frontend

            hipdnn_frontend.Handle()
        except Exception as e:
            pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(sample_graph_path),
                "--warmup",
                "1",
                "--iters",
                "2",
                "-v",
            ]
            + plugin_path_cli_args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Check output contains expected sections
        assert "hipDNN Benchmark" in result.stdout
        assert "Execution Statistics" in result.stdout
        assert "Mean" in result.stdout

        # Should succeed
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "graph_name,expected_name",
        [
            ("sample_conv_fwd.json", "sample_conv_fwd_16x16x16x16_k16_3x3"),
            pytest.param(
                "sample_matmul.json",
                "sample_matmul_256x512x1024",
                marks=pytest.mark.xfail(
                    reason="MIOpen plugin doesn't support matmul operations yet"
                ),
            ),
            pytest.param(
                "sample_relu.json",
                "sample_relu_activation_64x128x56x56",
                marks=pytest.mark.xfail(
                    reason="MIOpen plugin doesn't support pointwise operations yet"
                ),
            ),
            pytest.param(
                "sample_add.json",
                "sample_pointwise_add_128x256x14x14",
                marks=pytest.mark.xfail(
                    reason="MIOpen plugin doesn't support pointwise operations yet"
                ),
            ),
            ("sample_batchnorm.json", "sample_batchnorm_inference_32x64x28x28"),
        ],
    )
    def test_cli_all_sample_graphs(
        self, graph_name: str, expected_name: str, plugin_path_cli_args
    ) -> None:
        """Test CLI execution with all sample graph types."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / graph_name

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("PyTorch GPU not available")
        except ImportError as e:
            pytest.skip(f"PyTorch not available: {e}")

        # Check if hipdnn is available
        try:
            import hipdnn_frontend

            hipdnn_frontend.Handle()
        except Exception as e:
            pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(sample_path),
                "--warmup",
                "1",
                "--iters",
                "2",
                "-v",
            ]
            + plugin_path_cli_args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Check output contains expected sections
        assert "hipDNN Benchmark" in result.stdout
        assert expected_name in result.stdout
        assert "Execution Statistics" in result.stdout
        assert "Mean" in result.stdout

        # Should succeed
        assert result.returncode == 0, f"CLI failed for {graph_name}: {result.stderr}"


class TestCLIParser:
    """Unit tests for CLI parser."""

    def test_parse_default_values(self) -> None:
        """Test parsing with default values."""
        from dnn_benchmarking.cli.parser import create_parser

        args = create_parser().parse_args(["--graph", "/test/graph.json"])

        assert args.graph == ["/test/graph.json"]
        assert args.warmup == 10
        assert args.iters == 100
        # --engine defaults to None (= run all discovered engines)
        assert args.engine is None

    def test_parse_custom_values(self) -> None:
        """Test parsing with custom values."""
        from dnn_benchmarking.cli.parser import create_parser

        args = create_parser().parse_args(
            [
                "--graph",
                "/test/graph.json",
                "--warmup",
                "20",
                "--iters",
                "200",
                "--engine",
                "2",
            ]
        )

        assert args.graph == ["/test/graph.json"]
        assert args.warmup == 20
        assert args.iters == 200
        assert args.engine == [2]

    def test_parse_short_options(self) -> None:
        """Test parsing with short option names."""
        from dnn_benchmarking.cli.parser import create_parser

        args = create_parser().parse_args(
            ["-g", "/test/graph.json", "-w", "5", "-i", "50", "-e", "3"]
        )

        assert args.graph == ["/test/graph.json"]
        assert args.warmup == 5
        assert args.iters == 50
        assert args.engine == [3]

    def test_engine_comma_list_parses_at_subprocess_level(self) -> None:
        """A comma-separated --engine value is accepted by the CLI without error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                "/nonexistent/path/graph.json",
                "--engine",
                "1,2,3",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        # Should fail because graph doesn't exist (rc=1), not because of arg parse (rc=2)
        assert result.returncode == 1, (
            f"Expected rc=1 (graph not found), got {result.returncode}. "
            f"stderr: {result.stderr}"
        )

    def test_engine_invalid_value_rejected(self) -> None:
        """A non-integer --engine value is rejected by the parser (rc=2)."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                "/x.json",
                "--engine",
                "abc",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 2  # argparse error

    def test_verbose_flag_accepted(self) -> None:
        """The -v / --verbose flag is accepted by the parser."""
        for flag in ("-v", "--verbose"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dnn_benchmarking",
                    "--graph",
                    "/nonexistent/path/graph.json",
                    flag,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            # Same as above: rc=1 means parser accepted the flag and we got past arg parse
            assert result.returncode == 1, (
                f"{flag}: expected rc=1 (graph not found), got {result.returncode}. "
                f"stderr: {result.stderr}"
            )
