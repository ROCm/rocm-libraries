# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, List, Optional

import torch

from ..common import torch_support

from ..config.benchmark_config import BenchmarkConfig
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from . import pytorch_ops
from .timing import HipGpuTimer, Timer, _is_torch_available, create_stream_synchronizer


class PyTorchExecutionError(Exception):
    """Error during PyTorch graph execution."""

    pass


class PyTorchCudaExecutor:
    """Executes hipDNN-format graphs using PyTorch on GPU.

    This class handles:
    - Validating graph operations are supported
    - Running warmup iterations
    - Running timed benchmark iterations with direct HIP event timing
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
        if not torch_support.gpu_available():
            raise PyTorchExecutionError(
                "PyTorch GPU not available. Install PyTorch with CUDA or ROCm support."
            )

        self._graph_json = graph_json
        self._config = config
        self._device = torch.device(device)
        self._init_time_ms: float = 0.0
        self._prepared = False
        self._stream: Optional[Any] = None
        self._stream_synchronizer: Optional[Any] = None

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

            # Pin all PyTorch graph execution to one stream, then synchronize
            # that stream through HIP events instead of using torch.cuda APIs.
            torch.cuda.init()
            self._stream = torch.cuda.default_stream(self._device)
            try:
                self._stream_synchronizer = create_stream_synchronizer(
                    self._timing_stream()
                )
            except RuntimeError as e:
                raise PyTorchExecutionError(str(e)) from e
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

        with torch.cuda.stream(self._get_stream()):
            for _ in range(self._config.warmup_iters):
                self._execute_graph(tensors)
        if self._config.warmup_iters > 0:
            self._synchronize_stream()

    def execute_once(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Execute the graph once and synchronize.

        Used after timed loops to collect clean reference outputs without
        including output zeroing or extraction in benchmark timings.
        """
        if not self._prepared:
            raise PyTorchExecutionError("Executor not prepared. Call prepare() first.")

        with torch.cuda.stream(self._get_stream()):
            self._execute_graph(tensors)
        self._synchronize_stream()

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
        try:
            gpu_timer = HipGpuTimer(stream=self._timing_stream())
        except RuntimeError as e:
            raise PyTorchExecutionError(str(e)) from e

        for _ in range(self._config.benchmark_iters):
            with Timer() as t:
                with torch.cuda.stream(self._get_stream()):
                    gpu_timer.start()
                    self._execute_graph(tensors)
                    gpu_timer.stop()
                    kernel_ms = gpu_timer.elapsed_ms()

            kernel_timings.append(kernel_ms)
            e2e_timings.append(t.elapsed_ms)

        # Build metadata
        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            timing_backend="hip",
            execution_backend="pytorch",
        )

        return BenchmarkResult(
            e2e_timings=e2e_timings,
            kernel_timings=kernel_timings,
            metadata=metadata,
        )

    def _get_stream(self) -> Any:
        """Return the PyTorch stream used by all graph execution."""
        if self._stream is None:
            raise PyTorchExecutionError("Executor not prepared. Call prepare() first.")
        return self._stream

    def _synchronize_stream(self) -> None:
        """Synchronize the PyTorch graph stream through hipdnn_frontend bindings."""
        try:
            if self._stream_synchronizer is None:
                self._stream_synchronizer = create_stream_synchronizer(
                    self._timing_stream()
                )
            self._stream_synchronizer.synchronize()
        except RuntimeError as e:
            raise PyTorchExecutionError(str(e)) from e

    def _timing_stream(self) -> int:
        """Return the PyTorch graph stream pointer for HIP events."""
        return int(self._get_stream().cuda_stream)

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
