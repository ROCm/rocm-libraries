# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph execution with timing for benchmarks."""

import json
from typing import Any, Dict, List, Optional

from ..common import torch_support
from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import (
    BenchmarkConfig,
    ExecutionBackendName,
)
from ..reporting.statistics import BenchmarkMetadata, BenchmarkResult
from .timing import GpuTimerInterface, HipGpuTimer, Timer, create_gpu_timer

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


def _resolve_data_type(hipdnn: Any, type_str: str) -> Optional[Any]:
    """Resolve a data type string to a hipdnn.DataType enum value.

    Args:
        hipdnn: The hipdnn_frontend module.
        type_str: Data type string from graph JSON (e.g. "FLOAT", "HALF").

    Returns:
        Matching hipdnn.DataType enum value, or None if the string is not
        a recognised data type. Callers should skip configuring the graph
        attribute when None is returned and let hipDNN inference resolve it.
    """
    enum_name = _DATA_TYPE_STR_MAP.get(type_str.upper())
    if enum_name is None:
        return None
    return getattr(hipdnn.DataType, enum_name, None)


def _get_handle_stream(handle: Any) -> int:
    """Return a hipDNN handle's HIP stream pointer encoded as an integer."""
    get_stream = getattr(handle, "get_stream", None)
    if callable(get_stream):
        return int(get_stream())
    return 0


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
        collect_kernel_timing: bool = True,
    ) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json_str: The graph as a JSON string.
            config: Benchmark configuration.
            collect_kernel_timing: When True, record per-iteration GPU kernel
                timings via HIP events when available; otherwise collect only
                E2E timing with stream synchronization.
        """
        self._graph_json_str = graph_json_str
        self._config = config
        self._collect_kernel_timing = collect_kernel_timing
        self._execution_stream: Optional[int] = None
        self._graph: Any = None
        self._workspace: Any = None
        self._workspace_ptr: int = 0
        self._workspace_size: int = 0
        self._init_time_ms: float = 0.0
        self._stream_sync_timer: Optional[HipGpuTimer] = None
        self._selected_engine_id: Optional[int] = None

    def _build_through_operation_graph(self, handle: Any) -> Any:
        """Create the hipdnn graph and run it up to ``build_operation_graph``.

        Shared by ``discover_engines`` and ``prepare``. Configures only the
        data-type attributes that are explicitly present in the graph JSON
        (other types are left for hipDNN inference). Engine selection happens
        afterwards: ``prepare`` hard-selects a forced engine once the operation
        graph is built, and ``discover_engines`` queries the ranked list.

        Args:
            handle: hipdnn.Handle instance.

        Returns:
            The hipdnn module (so callers can keep using its enums/types
            without re-importing).

        Raises:
            ExecutionError: If hipdnn_frontend is unavailable or any of the
                graph-build steps fail.
        """
        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        self._graph = hipdnn.Graph()

        try:
            graph_dict = json.loads(self._graph_json_str)
        except (json.JSONDecodeError, TypeError):
            graph_dict = {}

        if "io_data_type" in graph_dict:
            io_dt = _resolve_data_type(hipdnn, graph_dict["io_data_type"])
            if io_dt is not None:
                self._graph.set_io_data_type(io_dt)
        if "intermediate_data_type" in graph_dict:
            intermediate_dt = _resolve_data_type(
                hipdnn, graph_dict["intermediate_data_type"]
            )
            if intermediate_dt is not None:
                self._graph.set_intermediate_data_type(intermediate_dt)
        if "compute_data_type" in graph_dict:
            compute_dt = _resolve_data_type(hipdnn, graph_dict["compute_data_type"])
            if compute_dt is not None:
                self._graph.set_compute_data_type(compute_dt)

        # Normalise node compute_data_type: from_json rejects "unset", which
        # hipDNN emits when the caller leaves the field unset. Promote to the
        # graph-level compute type so the serialised form round-trips cleanly.
        if graph_dict.get("nodes"):
            graph_cdt = graph_dict.get("compute_data_type", "float")
            changed = False
            for node in graph_dict["nodes"]:
                if node.get("compute_data_type", "").lower() == "unset":
                    node["compute_data_type"] = graph_cdt
                    changed = True
            if changed:
                self._graph_json_str = json.dumps(graph_dict)

        result = self._graph.from_json(self._graph_json_str)
        if result.is_bad():
            raise ExecutionError(f"Failed to deserialize graph: {result.get_message()}")

        result = self._graph.validate()
        if result.is_bad():
            raise ExecutionError(f"Graph validation failed: {result.get_message()}")

        result = self._graph.build_operation_graph(handle)
        if result.is_bad():
            raise ExecutionError(
                f"Failed to build operation graph: {result.get_message()}"
            )

        return hipdnn

    def discover_engines(self, handle: Any) -> List[int]:
        """Build the operation graph and return ranked engine IDs.

        Runs the same setup as ``prepare`` up to ``build_operation_graph``,
        then queries ``get_ranked_engine_ids``. Does not allocate a workspace
        or set a preferred engine; callers iterate the returned IDs and create
        a fresh Executor per engine for execution.

        Args:
            handle: hipdnn.Handle instance.

        Returns:
            List of int engine IDs ranked by the backend's heuristics.

        Raises:
            ExecutionError: If any graph-build step fails.
        """
        self._build_through_operation_graph(handle)
        try:
            return [int(eid) for eid in self._graph.get_ranked_engine_ids()]
        except RuntimeError as e:
            raise UnsupportedGraphError(str(e)) from e

    def prepare(self, handle: Any, engine_id: Optional[int] = None) -> None:
        """Build the operation graph and prepare for execution.

        Args:
            handle: hipdnn.Handle instance.
            engine_id: Optional engine ID to use. If specified, overrides
                       any engine ID in the graph JSON.

        Raises:
            ExecutionError: If graph building fails.
        """
        with Timer() as t:
            self._execution_stream = _get_handle_stream(handle)
            hipdnn = self._build_through_operation_graph(handle)

            if engine_id is not None:
                # Hard engine selection: build the plan for exactly this engine.
                # create_execution_plan_ext reports a bad result if the engine is
                # not valid/applicable, so it can never silently fall back to a
                # different engine the way the soft preferred-engine path could.
                result = self._graph.create_execution_plan_ext(engine_id)
                if result.is_bad():
                    raise UnsupportedGraphError(
                        f"Forced engine {engine_id} not applicable to this graph: "
                        f"{result.get_message()}"
                    )
            else:
                result = self._graph.create_execution_plans()
                if result.is_bad():
                    raise ExecutionError(
                        f"Failed to create execution plans: {result.get_message()}"
                    )

            result = self._graph.check_support()
            if result.is_bad():
                raise UnsupportedGraphError(
                    f"Backend support check failed: {result.get_message()}"
                )

            result = self._graph.build_plans()
            if result.is_bad():
                raise ExecutionError(f"Failed to build plans: {result.get_message()}")

            self._record_selected_engine(engine_id)

            workspace_size = self._graph.get_workspace_size()
            self._workspace_size = int(workspace_size)
            if workspace_size > 0:
                self._workspace = hipdnn.DeviceBuffer(workspace_size)
                self._workspace_ptr = self._workspace.ptr()

        self._init_time_ms = t.elapsed_ms

    @property
    def selected_engine_id(self) -> Optional[int]:
        """Engine ID that actually backed the built plan, or None if unknown."""
        return self._selected_engine_id

    def _record_selected_engine(self, requested_engine_id: Optional[int]) -> None:
        """Read back and record the engine that actually backs the built plan.

        ``get_execution_plan_engine_id`` is the authoritative source for the
        engine that will run, regardless of how it was chosen. A forced engine
        that differs from the engine actually selected should be impossible on
        the hard-select path, so any mismatch is treated as an unsupported-graph
        skip rather than mislabeled timings.
        """
        # prepare() has already created the execution plan, so the frontend has
        # cached the engine actually selected for it; the getter returns that
        # cached id.
        actual = int(self._graph.get_execution_plan_engine_id())
        self._selected_engine_id = actual
        if requested_engine_id is not None and actual != requested_engine_id:
            raise UnsupportedGraphError(
                f"Forced engine {requested_engine_id} was not selected; the "
                f"backend ran engine {actual} (silent fallback). Skipping to "
                f"avoid mislabeled results."
            )

    def _get_execution_stream(self, handle: Any) -> int:
        """Return the prepared hipDNN handle stream and reject stream drift."""
        stream = _get_handle_stream(handle)
        if self._execution_stream is None:
            self._execution_stream = stream
        elif stream != self._execution_stream:
            raise ExecutionError(
                "hipDNN handle stream changed after prepare: "
                f"prepared stream {self._execution_stream}, current stream {stream}"
            )
        return stream

    def _get_stream_sync_timer(self, stream: int) -> HipGpuTimer:
        """Return the reusable event-backed synchronizer for the execution stream."""
        if self._stream_sync_timer is None:
            try:
                self._stream_sync_timer = HipGpuTimer(stream)
            except RuntimeError as e:
                raise ExecutionError(str(e)) from e
        return self._stream_sync_timer

    def execute_once(self, handle: Any, variant_pack: Dict[int, int]) -> None:
        """Execute the prepared graph once without collecting timings."""
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")
        if self._workspace is not None:
            self._workspace.zeros()

        stream = self._get_execution_stream(handle)
        result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
        if result.is_bad():
            raise ExecutionError(f"Graph execution failed: {result.get_message()}")
        self._get_stream_sync_timer(stream).synchronize_stream()

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

        stream = self._get_execution_stream(handle)
        for _ in range(self._config.warmup_iters):
            result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
            if result.is_bad():
                raise ExecutionError(f"Warmup execution failed: {result.get_message()}")

        # hipDNN graph execution is asynchronous. Drain untimed warmup work before
        # benchmark() starts the measured loop, otherwise the first timed E2E
        # iteration can include queued warmup kernels.
        if self._config.warmup_iters > 0:
            self._get_stream_sync_timer(stream).synchronize_stream()

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
        timing_backend_name: str = ""
        stream_sync_timer = None
        stream = self._get_execution_stream(handle)

        # Create GPU timer when kernel timing is requested and available.
        if self._collect_kernel_timing:
            try:
                gpu_timer = create_gpu_timer(stream=stream)
            except RuntimeError as e:
                raise ExecutionError(str(e)) from e
            if gpu_timer is not None:
                kernel_timings = []
                timing_backend_name = gpu_timer.backend_name

        for _ in range(self._config.benchmark_iters):
            kernel_ms: Optional[float] = None
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
                    kernel_ms = gpu_timer.elapsed_ms()
                else:
                    if stream_sync_timer is None:
                        stream_sync_timer = self._get_stream_sync_timer(stream)
                    stream_sync_timer.synchronize_stream()

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
            execution_backend=ExecutionBackendName.HIPDNN.value,
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
    def workspace_size(self) -> int:
        """Bytes hipDNN reserved for the operation graph workspace.

        Zero before :meth:`prepare` runs. Surfaced so the suite runner
        can record it as an always-on metric without re-querying the
        graph object.
        """
        return self._workspace_size

    @property
    def graph(self) -> Any:
        """Get the underlying hipdnn graph object."""
        return self._graph
