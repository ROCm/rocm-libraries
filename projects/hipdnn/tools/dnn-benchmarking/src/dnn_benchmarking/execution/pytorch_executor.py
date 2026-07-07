# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch GPU executor for graph benchmarking."""

from typing import Any, Dict, List, Optional

import torch

from ..common import torch_support
from ..common.exceptions import UnsupportedGraphError

from ..config.benchmark_config import (
    BenchmarkConfig,
    ExecutionBackendName,
    TimingBackendName,
)
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from . import pytorch_ops
from .timing import (
    GpuTimerInterface,
    HipGpuTimer,
    StalledRegionTimer,
    Timer,
    _is_staged_hip_available,
    create_gpu_timer,
    run_staged_iterations,
)


class PyTorchExecutionError(Exception):
    """Error during PyTorch graph execution."""

    pass


class PyTorchCudaExecutor:
    """Executes hipDNN-format graphs using PyTorch on GPU.

    This class handles:
    - Validating graph operations are supported
    - Running warmup iterations
    - Running timed benchmark iterations with GPU event timing
      (HIP events on ROCm, torch.cuda events on CUDA)
    - Returning BenchmarkResult with E2E and kernel timings
    """

    def __init__(
        self,
        graph_json: Dict[str, Any],
        config: BenchmarkConfig,
        device: str = "cuda:0",
        collect_kernel_timing: bool = True,
    ) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            config: Benchmark configuration.
            device: CUDA/ROCm device to use (e.g., "cuda:0").
            collect_kernel_timing: When True, record per-iteration GPU kernel
                timings (HIP events on ROCm torch, torch.cuda events on CUDA
                torch); otherwise collect only E2E timing with stream sync.

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
        self._collect_kernel_timing = collect_kernel_timing
        self._init_time_ms: float = 0.0
        self._prepared = False
        self._stream: Optional[Any] = None
        self._stream_sync_timer: Optional[HipGpuTimer] = None
        self._compiled: Optional[pytorch_ops.CompiledGraph] = None

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

            self._compiled = pytorch_ops.compile_graph(self._graph_json)

            # Pin all PyTorch graph execution to one stream. On ROCm the
            # stream is synchronized through HIP events instead of torch.cuda
            # APIs; on CUDA (or with explicit torch timing) synchronization
            # goes through the torch stream itself.
            with torch.cuda.device(self._device):
                torch.cuda.init()
                self._stream = torch.cuda.default_stream(self._device)
                if self._hip_sync_selected():
                    try:
                        self._stream_sync_timer = HipGpuTimer(
                            stream=self._timing_stream()
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

        # Prefer stalled-queue staging (the same path the hipDNN executor uses)
        # so the GPU event span is pure device time and the A/B comparison is
        # symmetric. PyTorch reference ops that read host scalars (epsilon/scale
        # via .item()) are made async-safe by resolving those scalars once,
        # before the gated loop, via ReplayTensors. Staging is HIP-only; CUDA
        # torch and capability mismatches fall back to direct GPU-event timing.
        staged_timer: Optional[StalledRegionTimer] = None
        if (
            self._collect_kernel_timing
            and self._resolve_timing_backend() is TimingBackendName.HIP
            and _is_staged_hip_available()
        ):
            with torch.cuda.device(self._device):
                try:
                    staged_timer = StalledRegionTimer(self._timing_stream())
                except RuntimeError:
                    staged_timer = None

        if staged_timer is not None:
            return self._benchmark_staged(staged_timer, tensors, graph_name)

        host_timings: List[float] = []
        kernel_timings: Optional[List[float]] = None
        gpu_timer: Optional[GpuTimerInterface] = None
        timing_backend_name = ""

        if self._collect_kernel_timing:
            with torch.cuda.device(self._device):
                try:
                    gpu_timer = create_gpu_timer(
                        self._resolve_timing_backend(),
                        stream=self._timing_stream(),
                        torch_stream=self._get_stream(),
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
            host_timings.append(t.elapsed_ms)

        # Build metadata
        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            timing_backend=timing_backend_name,
            execution_backend=ExecutionBackendName.PYTORCH.value,
        )

        return BenchmarkResult(
            host_timings=host_timings,
            kernel_timings=kernel_timings,
            metadata=metadata,
        )

    def _benchmark_staged(
        self,
        timer: StalledRegionTimer,
        tensors: Dict[int, torch.Tensor],
        graph_name: str,
    ) -> BenchmarkResult:
        """Benchmark via the stalled-queue staged timer (pure GPU spans).

        Wraps the inputs in a ``ReplayTensors`` so host scalar reads
        (epsilon/scale ``.item()``) are resolved once, before the gated loop;
        each timed iteration is then a pure asynchronous enqueue that cannot
        block the stalled stream.
        """
        replay = pytorch_ops.ReplayTensors(tensors)

        with torch.cuda.device(self._device):
            # Untimed priming pass: populate the scalar cache (and the caching
            # allocator) so the first gated iteration submits no device->host
            # sync. run_staged_iterations then drains the device (barrier)
            # before measurement.
            with torch.cuda.stream(self._get_stream()):
                self._execute_graph(replay)

            def enqueue() -> None:
                with torch.cuda.stream(self._get_stream()):
                    self._execute_graph(replay)

            host_timings, kernel_timings = run_staged_iterations(
                timer, self._config.benchmark_iters, enqueue
            )

        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            timing_backend=TimingBackendName.HIP.value,
            execution_backend=ExecutionBackendName.PYTORCH.value,
        )

        return BenchmarkResult(
            host_timings=host_timings,
            kernel_timings=kernel_timings,
            metadata=metadata,
        )

    def _resolve_timing_backend(self) -> TimingBackendName:
        """Resolve the GPU timer backend for this run.

        Resolve from the torch runtime driving the stream, not from raw HIP
        device visibility: ROCm torch uses HIP events, CUDA torch must use
        torch.cuda events. Resolving from torch_support avoids recording HIP
        events on a CUDA stream pointer on a mixed host (CUDA torch with
        visible ROCm/hipDNN).
        """
        return (
            TimingBackendName.HIP
            if torch_support.is_rocm_build()
            else TimingBackendName.TORCH
        )

    def _hip_sync_selected(self) -> bool:
        """Return True when stream synchronization should use HIP events.

        Mirrors :meth:`_resolve_timing_backend`: HIP sync is used only when
        the resolved backend is "hip" (explicit, or auto/none on ROCm torch).
        """
        return self._resolve_timing_backend() is TimingBackendName.HIP

    def _get_stream(self) -> Any:
        """Return the PyTorch stream used by all graph execution."""
        if self._stream is None:
            raise PyTorchExecutionError("Executor not prepared. Call prepare() first.")
        return self._stream

    def _synchronize_stream(self) -> None:
        """Synchronize the PyTorch graph stream.

        Uses a HIP event when HIP sync was selected at prepare() time,
        otherwise synchronizes the torch stream directly.
        """
        with torch.cuda.device(self._device):
            if self._stream_sync_timer is not None:
                try:
                    self._stream_sync_timer.synchronize_stream()
                except RuntimeError as e:
                    raise PyTorchExecutionError(str(e)) from e
            else:
                self._get_stream().synchronize()

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
            assert self._compiled is not None
            self._compiled.execute(tensors)
        except UnsupportedGraphError:
            raise
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
