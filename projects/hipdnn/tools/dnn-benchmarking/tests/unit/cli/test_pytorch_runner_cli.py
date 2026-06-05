# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch backend CLI runner."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[3] / "src"


def test_pytorch_backend_reports_missing_torch_without_uncaught_import(
    tmp_path: Path,
) -> None:
    graph = tmp_path / "graph.json"
    graph.write_text('{"name": "g", "nodes": [], "tensors": []}')

    script = textwrap.dedent(
        """
        import builtins
        import io
        import sys
        from pathlib import Path

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("blocked torch")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocking_import

        from dnn_benchmarking.cli.pytorch_runner_cli import run_pytorch_benchmark
        from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
        from dnn_benchmarking.reporting.reporter import Reporter

        output = io.StringIO()
        result = run_pytorch_benchmark(
            BenchmarkConfig(
                graph_path=Path(sys.argv[1]),
                warmup_iters=0,
                benchmark_iters=1,
            ),
            Reporter(output=output),
        )
        print(f"RESULT={result}")
        print(output.getvalue())
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [str(SRC_DIR), env.get("PYTHONPATH", "")] if path
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(graph)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "RESULT=1" in result.stdout
    assert (
        "ERROR: PyTorch not available. Install with: pip install torch" in result.stdout
    )
    assert "Unexpected error" not in result.stdout
