# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, List, Optional

import torch

from ..config.benchmark_config import BenchmarkConfig
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from . import pytorch_ops
from .timing import Timer, TorchGpuTimer, _is_torch_available


class PyTorchExecutionError(Exception):
    """Error during PyTorch graph execution."""

    pass


class PyTorchCudaExecutor:
    """Executes hipDNN-format graphs using PyTorch on GPU.

    This class handles:
    - Validating graph operations are supported
    - Running warmup iterations
    - Running timed benchmark iterations with TorchGpuTimer
    - Returning BenchmarkResult with E2E and kernel timings
    """

    def __init__(
        self,
        graph_json: Dict[str, Any],
        config: BenchmarkConfig,
        device: str = "cuda:0",
    ) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            config: Benchmark configuration.
            device: CUDA/ROCm device to use (e.g., "cuda:0").

        Raises:
            PyTorchExecutionError: If PyTorch GPU is not available.
        """
        if not _is_torch_available():
            raise PyTorchExecutionError(
                "PyTorch GPU not available. Install PyTorch with CUDA or ROCm support."
            )

        self._graph_json = graph_json
        self._config = config
        self._device = torch.device(device)
        self._init_time_ms: float = 0.0
        self._prepared = False

    def prepare(self) -> None:
        """Validate graph and prepare for execution.

        Raises:
            PyTorchExecutionError: If graph contains unsupported operations.
        """
        with Timer() as t:
            # Check all operations are supported
            unsupported = pytorch_ops.get_unsupported_operations(self._graph_json)
            if unsupported:
                raise PyTorchExecutionError(
                    f"Graph contains unsupported operations: {unsupported}. "
                    f"Supported: {list(pytorch_ops.get_supported_operations())}"
                )

            # Warm up CUDA/ROCm context if needed
            torch.cuda.init()

            self._prepared = True

        self._init_time_ms = t.elapsed_ms

    def warmup(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Run warmup iterations (timing discarded).

        Args:
            tensors: Mapping of tensor UIDs to CUDA tensors.

        Raises:
            PyTorchExecutionError: If executor not prepared.
        """
        if not self._prepared:
            raise PyTorchExecutionError("Executor not prepared. Call prepare() first.")

        for _ in range(self._config.warmup_iters):
            self._execute_graph(tensors)
            torch.cuda.synchronize()

    def benchmark(
        self,
        tensors: Dict[int, torch.Tensor],
        graph_name: str = "",
    ) -> BenchmarkResult:
        """Run benchmark iterations and collect timing.

        Collects both E2E (wall-clock) timing and GPU kernel timing.

        Args:
            tensors: Mapping of tensor UIDs to CUDA tensors.
            graph_name: Optional name/identifier for the graph being benchmarked.

        Returns:
            BenchmarkResult with E2E and kernel timings, plus metadata.

        Raises:
            PyTorchExecutionError: If executor not prepared or execution fails.
        """
        if not self._prepared:
            raise PyTorchExecutionError("Executor not prepared. Call prepare() first.")

        e2e_timings: List[float] = []
        kernel_timings: List[float] = []
        gpu_timer = TorchGpuTimer()

        for _ in range(self._config.benchmark_iters):
            with Timer() as t:
                gpu_timer.start()
                self._execute_graph(tensors)
                gpu_timer.stop()
                torch.cuda.synchronize()

            kernel_timings.append(gpu_timer.elapsed_ms())
            e2e_timings.append(t.elapsed_ms)

        # Build metadata
        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            gpu_backend="torch",
            execution_backend="pytorch",
        )

        return BenchmarkResult(
            e2e_timings=e2e_timings,
            kernel_timings=kernel_timings,
            metadata=metadata,
        )

    def _execute_graph(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Execute all graph operations in order.

        Args:
            tensors: Mapping of tensor UIDs to CUDA tensors.

        Raises:
            PyTorchExecutionError: If execution fails.
        """
        try:
            pytorch_ops.execute_graph(self._graph_json, tensors)
        except Exception as e:
            raise PyTorchExecutionError(f"Graph execution failed: {e}") from e

    @property
    def init_time_ms(self) -> float:
        """Get graph initialization time in milliseconds."""
        return self._init_time_ms

    @property
    def device(self) -> torch.device:
        """Get the CUDA/ROCm device being used."""
        return self._device
