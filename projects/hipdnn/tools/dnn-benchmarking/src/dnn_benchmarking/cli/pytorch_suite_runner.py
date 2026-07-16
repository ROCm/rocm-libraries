# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch suite benchmark startup and graph dispatch."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import torch_support
from ..config.benchmark_config import SuiteConfig
from ..execution.suite_runner import run_graph_pytorch_backend
from ..reporting.reporter import Reporter
from ..reporting.suite_results import GraphResult
from .suite_runner_cli import _run_suite_graphs_after_startup


def _run_loaded_pytorch_graph(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    reporter: Reporter,
) -> GraphResult:
    return run_graph_pytorch_backend(
        graph_path, graph_json, tensor_infos, config, reporter=reporter
    )


def run_pytorch_suite_benchmark(
    graph_paths: List[Path],
    config: SuiteConfig,
    output_path: Optional[Path],
    reporter: Reporter,
    tarball_source: Optional[str] = None,
) -> int:
    """Run suite graphs through the PyTorch backend."""
    reporter.print_suite_header(
        len(graph_paths),
        tarball_source=tarball_source,
        extra_profiling_runs=config.metrics.extra_runs_per_engine,
    )

    # PyTorch availability is the authoritative GPU/runtime check for
    # this backend: CPU-only torch cannot execute these GPU benchmarks
    # even if ROCm management tools can see a device.
    if not torch_support.module_available():
        reporter.print_error("PyTorch not available. Install with: pip install torch")
        return 1
    if not torch_support.gpu_available():
        reporter.print_error(
            "PyTorch GPU not available. Install PyTorch with CUDA or ROCm support."
        )
        return 1

    return _run_suite_graphs_after_startup(
        graph_paths=graph_paths,
        config=config,
        output_path=output_path,
        reporter=reporter,
        run_loaded_graph=_run_loaded_pytorch_graph,
    )
