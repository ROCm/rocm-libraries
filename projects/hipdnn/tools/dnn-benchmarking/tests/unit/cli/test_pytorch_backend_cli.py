# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the PyTorch execution backend gate in the suite runner CLI."""

import builtins
import io
import sys
import types
from pathlib import Path

from dnn_benchmarking.cli.suite_runner_cli import run_suite_benchmark
from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.reporting.reporter import Reporter


def _write_graph(tmp_path: Path) -> Path:
    graph = tmp_path / "graph.json"
    graph.write_text('{"name": "g", "nodes": [], "tensors": []}')
    return graph


def test_pytorch_backend_reports_missing_torch_without_uncaught_import(
    tmp_path: Path, monkeypatch
) -> None:
    graph = _write_graph(tmp_path)

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    output = io.StringIO()
    result = run_suite_benchmark(
        graph_paths=[graph],
        config=SuiteConfig(backend="pytorch", warmup_iters=0, benchmark_iters=1),
        output_path=None,
        reporter=Reporter(output=output),
    )

    text = output.getvalue()

    assert result == 1
    assert "ERROR: PyTorch not available. Install with: pip install torch" in text
    assert "Unexpected error" not in text


def test_pytorch_backend_reports_missing_gpu(tmp_path: Path, monkeypatch) -> None:
    graph = _write_graph(tmp_path)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    output = io.StringIO()
    result = run_suite_benchmark(
        graph_paths=[graph],
        config=SuiteConfig(backend="pytorch", warmup_iters=0, benchmark_iters=1),
        output_path=None,
        reporter=Reporter(output=output),
    )

    text = output.getvalue()

    assert result == 1
    assert "ERROR: PyTorch GPU not available." in text
    # The hipDNN handle must never be constructed for the PyTorch backend.
    assert "Initializing hipDNN" not in text
