# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, List, Literal, Optional

import torch

from ..common import torch_support

from ..config.benchmark_config import BenchmarkConfig
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from . import pytorch_ops
from .timing import HipGpuTimer, Timer, _is_torch_available, create_gpu_timer


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
        timing_backend: Optional[Literal["hip", "auto", "none"]] = "auto",
    ) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            config: Benchmark configuration.
            device: CUDA/ROCm device to use (e.g., "cuda:0").
            timing_backend: GPU timer backend to use:
                - "hip": Force direct HIP event timing
                - "auto": Auto-detect direct HIP timing
                - "none": Disable GPU kernel timing, use synchronized E2E timing

        Raises:
            PyTorchExecutionError: If PyTorch GPU is not available.
        """
        if not torch_support.gpu_available():
            raise PyTorchExecutionError(
                "PyTorch GPU not available. Install PyTorch with CUDA or ROCm support."
            )

        if timing_backend is None:
            timing_backend = "auto"
        valid_timing_backends = {"hip", "auto", "none"}
        if timing_backend not in valid_timing_backends:
            raise ValueError(
                f"Invalid timing_backend: '{timing_backend}'. "
                f"Valid options: {valid_timing_backends}"
            )

        self._graph_json = graph_json
        self._config = config
        self._device = torch.device(device)
        self._timing_backend = timing_backend
        self._init_time_ms: float = 0.0
        self._prepared = False
        self._stream: Optional[Any] = None
        self._stream_sync_timer: Optional[HipGpuTimer] = None

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
            with torch.cuda.device(self._device):
                torch.cuda.init()
                self._stream = torch.cuda.default_stream(self._device)
                try:
                    self._stream_sync_timer = HipGpuTimer(stream=self._timing_stream())
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

        with torch.cuda.device(self._device):
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

        with torch.cuda.device(self._device):
            with torch.cuda.stream(self._get_stream()):
                self._execute_graph(tensors)
            self._synchronize_stream()

    def benchmark(
        self,
        tensors: Dict[int, torch.Tensor],
        graph_name: str = "",
    ) -> BenchmarkResult:
        """Run benchmark iterations and collect timing.

        Collects E2E timing and GPU kernel timing when enabled.

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
        kernel_timings: Optional[List[float]] = None
        gpu_timer: Optional[HipGpuTimer] = None
        timing_backend_name = ""
        with torch.cuda.device(self._device):
            if self._timing_backend != "none":
                try:
                    requested_backend = (
                        "hip" if self._timing_backend == "hip" else "auto"
                    )
                    gpu_timer = create_gpu_timer(
                        requested_backend, stream=self._timing_stream()
                    )
                except RuntimeError as e:
                    raise PyTorchExecutionError(str(e)) from e
                if gpu_timer is not None:
                    kernel_timings = []
                    timing_backend_name = gpu_timer.backend_name

        for _ in range(self._config.benchmark_iters):
            kernel_ms: Optional[float] = None
            with torch.cuda.device(self._device):
                with Timer() as t:
                    with torch.cuda.stream(self._get_stream()):
                        if gpu_timer is not None:
                            gpu_timer.start()
                        self._execute_graph(tensors)
                        if gpu_timer is not None:
                            gpu_timer.stop()
                            kernel_ms = gpu_timer.elapsed_ms()
                        else:
                            self._synchronize_stream()

            if kernel_ms is not None:
                assert kernel_timings is not None
                kernel_timings.append(kernel_ms)
            e2e_timings.append(t.elapsed_ms)

        # Build metadata
        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            timing_backend=timing_backend_name,
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
        """Synchronize the PyTorch graph stream through a HIP event."""
        with torch.cuda.device(self._device):
            try:
                if self._stream_sync_timer is None:
                    self._stream_sync_timer = HipGpuTimer(stream=self._timing_stream())
                self._stream_sync_timer.synchronize_stream()
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
