# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch backend CLI runner."""

import builtins
import io
from pathlib import Path

from dnn_benchmarking.cli.pytorch_runner_cli import run_pytorch_benchmark
from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.reporting.reporter import Reporter


def test_pytorch_backend_reports_missing_torch_without_uncaught_import(
    tmp_path: Path, monkeypatch
) -> None:
    graph = tmp_path / "graph.json"
    graph.write_text('{"name": "g", "nodes": [], "tensors": []}')

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    output = io.StringIO()
    result = run_pytorch_benchmark(
        BenchmarkConfig(
            graph_path=graph,
            warmup_iters=0,
            benchmark_iters=1,
        ),
        Reporter(output=output),
    )

    text = output.getvalue()

    assert result == 1
    assert "ERROR: PyTorch not available. Install with: pip install torch" in text
    assert "Unexpected error" not in text
