# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph execution with timing for benchmarks."""

import json
from typing import Any, Dict, List, Literal, Optional

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import BenchmarkConfig
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from .timing import GpuTimerInterface, Timer, create_gpu_timer

# Map graph JSON data type strings to hipdnn DataType enum names.
# The hipdnn.DataType enum is only available when hipdnn_frontend is imported,
# so we map to attribute name strings and resolve at runtime.
_DATA_TYPE_STR_MAP = {
    "FLOAT": "FLOAT",
    "DOUBLE": "DOUBLE",
    "HALF": "HALF",
    "BFLOAT16": "BFLOAT16",
    "INT8": "INT8",
    "INT32": "INT32",
    "UINT8": "UINT8",
    "INT64": "INT64",
    "BOOLEAN": "BOOLEAN",
}


def _resolve_data_type(hipdnn: Any, type_str: str, fallback: str = "FLOAT") -> Any:
    """Resolve a data type string to a hipdnn.DataType enum value.

    Args:
        hipdnn: The hipdnn_frontend module.
        type_str: Data type string from graph JSON (e.g. "FLOAT", "HALF").
        fallback: Fallback enum name if type_str is not recognized.

    Returns:
        hipdnn.DataType enum value.
    """
    enum_name = _DATA_TYPE_STR_MAP.get(type_str.upper(), fallback)
    return getattr(hipdnn.DataType, enum_name, hipdnn.DataType.FLOAT)


class Executor:
    """Executes hipDNN graphs with warmup and timed benchmark loops.

    This class handles:
    - Loading graph from JSON into hipdnn
    - Setting engine preferences
    - Building the operation graph
    - Running warmup iterations
    - Running timed benchmark iterations
    """

    def __init__(
        self,
        graph_json_str: str,
        config: BenchmarkConfig,
        gpu_backend: Optional[Literal["torch", "auto", "none"]] = "auto",
    ) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json_str: The graph as a JSON string.
            config: Benchmark configuration.
            gpu_backend: GPU timer backend to use:
                - "torch": Force PyTorch backend (CUDA or ROCm)
                - "auto": Auto-detect (uses PyTorch if available)
                - "none": Disable GPU timing, use only E2E timing
        """
        self._graph_json_str = graph_json_str
        self._config = config
        self._gpu_backend = gpu_backend
        self._graph: Any = None
        self._workspace: Any = None
        self._workspace_ptr: int = 0
        self._init_time_ms: float = 0.0

    def prepare(self, handle: Any, engine_id: Optional[int] = None) -> None:
        """Build the operation graph and prepare for execution.

        Args:
            handle: hipdnn.Handle instance.
            engine_id: Optional engine ID to use. If specified, overrides
                       any engine ID in the graph JSON.

        Raises:
            ExecutionError: If graph building fails.
        """
        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        with Timer() as t:
            # Create and configure graph
            self._graph = hipdnn.Graph()

            # Parse data types from the graph JSON; fall back to FLOAT
            try:
                graph_dict = json.loads(self._graph_json_str)
            except (json.JSONDecodeError, TypeError):
                graph_dict = {}

            io_dt = _resolve_data_type(hipdnn, graph_dict.get("io_data_type", "FLOAT"))
            intermediate_dt = _resolve_data_type(
                hipdnn, graph_dict.get("intermediate_data_type", "FLOAT")
            )
            compute_dt = _resolve_data_type(
                hipdnn, graph_dict.get("compute_data_type", "FLOAT")
            )

            self._graph.set_io_data_type(io_dt)
            self._graph.set_intermediate_data_type(intermediate_dt)
            self._graph.set_compute_data_type(compute_dt)

            # Deserialize from JSON
            result = self._graph.from_json(self._graph_json_str)
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to deserialize graph: {result.get_message()}"
                )

            # Set engine preference if specified
            if engine_id is not None:
                self._graph.set_preferred_engine_id_ext(engine_id)

            # Validate
            result = self._graph.validate()
            if result.is_bad():
                raise ExecutionError(f"Graph validation failed: {result.get_message()}")

            # Build operation graph
            result = self._graph.build_operation_graph(handle)
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to build operation graph: {result.get_message()}"
                )

            # Create execution plans
            result = self._graph.create_execution_plans()
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to create execution plans: {result.get_message()}"
                )

            # Check support
            result = self._graph.check_support()
            if result.is_bad():
                raise ExecutionError(
                    f"Backend support check failed: {result.get_message()}"
                )

            # Build plans
            result = self._graph.build_plans()
            if result.is_bad():
                raise ExecutionError(f"Failed to build plans: {result.get_message()}")

            # Allocate workspace
            workspace_size = self._graph.get_workspace_size()
            if workspace_size > 0:
                self._workspace = hipdnn.DeviceBuffer(workspace_size)
                self._workspace_ptr = self._workspace.ptr()

        self._init_time_ms = t.elapsed_ms

    def warmup(self, handle: Any, variant_pack: Dict[int, int]) -> None:
        """Run warmup iterations (timing discarded).

        Args:
            handle: hipdnn.Handle instance.
            variant_pack: Mapping of tensor UIDs to device pointers.

        Raises:
            ExecutionError: If graph not prepared or execution fails.
        """
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")

        for _ in range(self._config.warmup_iters):
            result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
            if result.is_bad():
                raise ExecutionError(f"Warmup execution failed: {result.get_message()}")

    def benchmark(
        self,
        handle: Any,
        variant_pack: Dict[int, int],
        graph_name: str = "",
    ) -> BenchmarkResult:
        """Run benchmark iterations and collect timing.

        Collects both E2E (wall-clock) timing and GPU kernel timing when available.

        Args:
            handle: hipdnn.Handle instance.
            variant_pack: Mapping of tensor UIDs to device pointers.
            graph_name: Optional name/identifier for the graph being benchmarked.

        Returns:
            BenchmarkResult with E2E and optional kernel timings, plus metadata.

        Raises:
            ExecutionError: If graph not prepared or execution fails.
        """
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")

        e2e_timings: List[float] = []
        kernel_timings: Optional[List[float]] = None
        gpu_timer: Optional[GpuTimerInterface] = None
        backend_name: str = ""
        torch_sync = None

        # Set up torch sync for accurate E2E timing (needed regardless of GPU timer)
        try:
            import torch

            if torch.cuda.is_available():
                torch_sync = torch.cuda.synchronize
        except ImportError:
            torch_sync = None

        # Create GPU timer if requested and available
        if self._gpu_backend != "none":
            try:
                gpu_timer = create_gpu_timer(
                    "torch" if self._gpu_backend == "torch" else "auto"
                )
            except RuntimeError as e:
                raise ExecutionError(str(e)) from e
            kernel_timings = []
            backend_name = gpu_timer.backend_name

        for _ in range(self._config.benchmark_iters):
            with Timer() as t:
                if gpu_timer:
                    gpu_timer.start()
                result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
                if result.is_bad():
                    raise ExecutionError(
                        f"Benchmark execution failed: {result.get_message()}"
                    )
                if gpu_timer:
                    gpu_timer.stop()
                if torch_sync:
                    torch_sync()

            if gpu_timer:
                kernel_timings.append(gpu_timer.elapsed_ms())
            e2e_timings.append(t.elapsed_ms)

        # Build metadata
        metadata = BenchmarkMetadata(
            graph_name=graph_name,
            graph_path=str(self._config.graph_path),
            warmup_iters=self._config.warmup_iters,
            benchmark_iters=self._config.benchmark_iters,
            engine_id=self._config.engine_id,
            gpu_backend=backend_name,
            execution_backend="hipdnn",
        )

        return BenchmarkResult(
            e2e_timings=e2e_timings,
            kernel_timings=kernel_timings,
            metadata=metadata,
        )

    @property
    def init_time_ms(self) -> float:
        """Get graph initialization time in milliseconds."""
        return self._init_time_ms

    @property
    def graph(self) -> Any:
        """Get the underlying hipdnn graph object."""
        return self._graph
