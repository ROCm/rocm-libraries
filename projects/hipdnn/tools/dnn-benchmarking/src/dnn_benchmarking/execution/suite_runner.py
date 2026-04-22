# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner for per-graph engine iteration with granular timing.

Iterates the engine IDs discovered for a graph via
``Graph.get_ranked_engine_ids`` (per D-01: real runtime discovery, no
hardcoded engine lists). For each engine, captures separated CPU build time
(TIME-01), GPU kernel time (TIME-02), and E2E wall-clock time (TIME-03).
Performs correctness validation by comparing GPU output against a reference
provider via ArrayComparator (CORR-02).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import BenchmarkConfig, SuiteConfig
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor, _resolve_data_type
from ..reporting.statistics import BenchmarkStats
from ..reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
)
from ..validation.comparison import ArrayComparator
from ..validation.reference_provider import (
    ReferenceProvider,
    ReferenceProviderRegistry,
)

logger = logging.getLogger(__name__)

# Keywords in error messages that indicate an unsupported combination
# rather than a hard error.
_SUPPORT_CHECK_KEYWORDS = (
    "support check failed",
    "not supported",
    "unsupported",
    "no engine",
)


def _build_discovery_graph(
    handle: Any, graph_json_str: str, graph_json: Dict[str, Any]
) -> Any:
    """Build a hipdnn Graph far enough to call ``get_ranked_engine_ids``.

    Mirrors the data-type setup + ``from_json`` + ``validate`` +
    ``build_operation_graph`` flow used by ``Executor.prepare``. We don't
    reuse ``Executor`` here because discovery should not allocate a
    workspace or set a preferred engine; it's just used to enumerate the
    engine IDs the backend reports for this graph.

    Args:
        handle: hipdnn.Handle instance.
        graph_json_str: Graph as JSON string.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        A hipdnn.Graph with the operation graph built (ready for
        ``get_ranked_engine_ids``).

    Raises:
        ExecutionError: If any hipdnn step fails.
    """
    import hipdnn_frontend as hipdnn

    graph = hipdnn.Graph()

    io_dt = _resolve_data_type(hipdnn, graph_json.get("io_data_type", "FLOAT"))
    intermediate_dt = _resolve_data_type(
        hipdnn, graph_json.get("intermediate_data_type", "FLOAT")
    )
    compute_dt = _resolve_data_type(
        hipdnn, graph_json.get("compute_data_type", "FLOAT")
    )
    graph.set_io_data_type(io_dt)
    graph.set_intermediate_data_type(intermediate_dt)
    graph.set_compute_data_type(compute_dt)

    result = graph.from_json(graph_json_str)
    if result.is_bad():
        raise ExecutionError(
            f"Discovery: failed to deserialize graph: {result.get_message()}"
        )

    result = graph.validate()
    if result.is_bad():
        raise ExecutionError(
            f"Discovery: graph validation failed: {result.get_message()}"
        )

    result = graph.build_operation_graph(handle)
    if result.is_bad():
        raise ExecutionError(
            f"Discovery: failed to build operation graph: {result.get_message()}"
        )

    return graph


def discover_engine_ids(
    handle: Any, graph_json_str: str, graph_json: Dict[str, Any]
) -> List[int]:
    """Discover engine IDs ranked by hipDNN heuristics for this graph.

    Per D-01: runtime discovery via ``Graph.get_ranked_engine_ids`` --
    no hardcoded engine lists. Builds a throwaway operation graph just
    far enough to query the backend for ranked engines.

    Args:
        handle: hipdnn.Handle instance.
        graph_json_str: Graph as JSON string.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        List of int engine IDs ranked by the backend's heuristics.

    Raises:
        ExecutionError: If discovery graph construction fails or the
            backend's discovery API errors.
    """
    discovery_graph = _build_discovery_graph(handle, graph_json_str, graph_json)
    return [int(eid) for eid in discovery_graph.get_ranked_engine_ids()]


def _resolve_engine_name(engine_id: int) -> str:
    """Resolve an engine ID to its registered name.

    Looks up the name via ``hipdnn_frontend.engine_id_to_name``. If the ID
    isn't registered (returns empty string), falls back to a hex display
    string so callers always have something printable.

    Args:
        engine_id: int engine ID.

    Returns:
        Registered engine name or ``f"engine_0x..."`` fallback.
    """
    try:
        import hipdnn_frontend as hipdnn

        name = hipdnn.engine_id_to_name(engine_id)
        if name:
            return name
    except Exception:
        pass
    return f"engine_{engine_id:#x}"


def _is_support_error(error_msg: str) -> bool:
    """Check if an error message indicates a support/compatibility issue.

    Args:
        error_msg: The error message string.

    Returns:
        True if the error indicates an unsupported combination.
    """
    lower = error_msg.lower()
    return any(kw in lower for kw in _SUPPORT_CHECK_KEYWORDS)


def _get_reference_provider(
    config: SuiteConfig, graph_json: Dict[str, Any]
) -> Optional[ReferenceProvider]:
    """Attempt to get and validate a reference provider for this graph.

    Args:
        config: Suite configuration with reference_provider name.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        ReferenceProvider instance if available and supports the graph,
        None otherwise. Caller distinguishes "not requested" from
        "requested but unsupported" via ``config.reference_provider``.
    """
    try:
        provider = ReferenceProviderRegistry.get_provider(config.reference_provider)
    except ValueError:
        logger.info("Reference provider '%s' not registered", config.reference_provider)
        return None

    if not provider.is_available():
        logger.info("Reference provider '%s' not available", config.reference_provider)
        return None

    if not provider.supports_graph(graph_json):
        logger.info(
            "Reference provider '%s' does not support this graph",
            config.reference_provider,
        )
        return None

    return provider


def _check_correctness(
    buffer_manager: BufferManager,
    tensor_infos: list,
    graph_json: Dict[str, Any],
    ref_provider: ReferenceProvider,
    config: SuiteConfig,
) -> CorrectnessResult:
    """Perform correctness comparison between GPU output and reference.

    Per CORR-02: compares GPU output against reference provider output
    using ArrayComparator. Validation was requested by caller (we are
    inside the ``ref_provider is not None`` branch), so when no outputs
    are comparable we report ``tolerance_match=False`` rather than
    silently passing.

    Args:
        buffer_manager: Buffer manager with output data from GPU execution.
        tensor_infos: List of TensorInfo objects for the graph.
        graph_json: Parsed graph JSON dictionary.
        ref_provider: Reference provider for computing expected output.
        config: Suite configuration with tolerance settings.

    Returns:
        CorrectnessResult with tolerance_match populated from comparison.
    """
    try:
        # Collect input data
        input_data: Dict[int, Any] = {}
        for ti in tensor_infos:
            if not ti.is_virtual and not ti.is_output:
                data = buffer_manager.get_input_data(ti.uid)
                if data is not None:
                    input_data[ti.uid] = data

        # Compute reference output
        ref_outputs = ref_provider.compute_reference(graph_json, input_data)

        # Compare each output tensor
        comparator = ArrayComparator(rtol=config.rtol, atol=config.atol)
        all_passed = True
        worst_abs_diff = 0.0
        worst_rel_diff = 0.0

        output_count = 0
        for ti in tensor_infos:
            if not ti.is_output:
                continue

            actual = buffer_manager.get_output_data(ti.uid)
            if actual is None:
                continue

            if ti.uid not in ref_outputs:
                continue

            expected = ref_outputs[ti.uid].data
            result = comparator.compare(actual, expected, "hipDNN", ref_provider.name)
            output_count += 1

            if not result.passed:
                all_passed = False

            if result.max_abs_diff > worst_abs_diff:
                worst_abs_diff = result.max_abs_diff
            if result.max_rel_diff > worst_rel_diff:
                worst_rel_diff = result.max_rel_diff

        if output_count == 0:
            # Validation was requested but nothing comparable surfaced
            # (e.g. reference omitted every output). Treat as failure
            # so --validate stays a hard gate (per W3).
            return CorrectnessResult(
                execution_success=True,
                tolerance_match=False,
                rtol=config.rtol,
                atol=config.atol,
                error_message="No output tensors to compare",
            )

        return CorrectnessResult(
            execution_success=True,
            tolerance_match=all_passed,
            rtol=config.rtol,
            atol=config.atol,
            max_abs_diff=worst_abs_diff,
            max_rel_diff=worst_rel_diff,
        )

    except Exception as e:
        return CorrectnessResult(
            execution_success=True,
            tolerance_match=False,
            rtol=config.rtol,
            atol=config.atol,
            error_message=str(e),
        )


def run_graph_all_providers(
    graph_path: Path,
    graph_json: Dict[str, Any],
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
) -> GraphResult:
    """Run a single graph against every engine the backend ranks for it.

    Discovers engine IDs via ``Graph.get_ranked_engine_ids`` (per D-01),
    applies any user engine filter (D-03), and runs the benchmark for
    each remaining ID. Captures separated CPU build time, GPU kernel
    time, and E2E wall-clock time per engine. Performs correctness
    checking against a reference provider when ``--validate`` was
    requested.

    Args:
        graph_path: Path to the graph JSON file.
        graph_json: Parsed graph JSON dictionary.
        tensor_infos: List of TensorInfo objects for the graph.
        config: Suite configuration.
        handle: hipdnn.Handle instance.

    Returns:
        GraphResult with one ProviderEngineResult per engine.
    """
    graph_name = graph_json.get("name", graph_path.stem)
    graph_json_str = json.dumps(graph_json)

    validation_requested = config.reference_provider != "none"

    # Discover engines via real backend heuristics. A discovery failure
    # is a graph-level error (record it and stop iterating engines), but
    # "no engine configurations available" / "not supported" messages are
    # really an unsupported-graph signal -- record as skipped so the
    # suite exit code stays 0 when nothing is wrong, just nothing to run.
    try:
        engine_ids = discover_engine_ids(handle, graph_json_str, graph_json)
    except (ExecutionError, RuntimeError) as e:
        msg = str(e)
        status = "skipped" if _is_support_error(msg) else "error"
        result_kwargs: Dict[str, Any] = {
            "provider": "unknown",
            "engine_id": 0,
            "status": status,
            "correctness": CorrectnessResult.failed(
                rtol=config.rtol, atol=config.atol, error_message=msg
            ),
        }
        if status == "error":
            result_kwargs["error_message"] = f"Engine discovery failed: {msg}"
        else:
            result_kwargs["skip_reason"] = msg
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[ProviderEngineResult(**result_kwargs)],
        )

    if config.engine_filter is not None:
        engine_ids = [e for e in engine_ids if e in config.engine_filter]

    if not engine_ids:
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider="unknown",
                    engine_id=0,
                    status="error",
                    error_message=(
                        "No engines discovered for graph"
                        if config.engine_filter is None
                        else "No discovered engines matched --engine filter"
                    ),
                )
            ],
        )

    ref_provider = _get_reference_provider(config, graph_json)

    pe_results: List[ProviderEngineResult] = []
    for engine_id in engine_ids:
        engine_name = _resolve_engine_name(engine_id)
        pe_result = _run_single_provider_engine(
            graph_path=graph_path,
            graph_json_str=graph_json_str,
            graph_name=graph_name,
            tensor_infos=tensor_infos,
            config=config,
            handle=handle,
            provider=engine_name,
            engine_id=engine_id,
            ref_provider=ref_provider,
            validation_requested=validation_requested,
            graph_json=graph_json,
        )
        pe_results.append(pe_result)

    return GraphResult(
        graph_name=graph_name,
        graph_path=str(graph_path),
        results=pe_results,
    )


def _run_single_provider_engine(
    graph_path: Path,
    graph_json_str: str,
    graph_name: str,
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    provider: str,
    engine_id: int,
    ref_provider: Optional[ReferenceProvider],
    validation_requested: bool,
    graph_json: Dict[str, Any],
) -> ProviderEngineResult:
    """Execute a single engine for a graph (single attempt, per D-10)."""
    try:
        bench_config = BenchmarkConfig(
            graph_path=graph_path,
            warmup_iters=config.warmup_iters,
            benchmark_iters=config.benchmark_iters,
            engine_id=engine_id,
        )

        executor = Executor(
            graph_json_str=graph_json_str,
            config=bench_config,
            gpu_backend=config.gpu_backend,
        )
        executor.prepare(handle, engine_id=engine_id)
        cpu_build_time_ms = executor.init_time_ms

        with BufferManager(tensor_infos) as bm:
            bm.allocate_all()
            bm.fill_inputs_random(seed=config.seed)
            bm.zero_outputs()

            variant_pack = bm.create_variant_pack()
            executor.warmup(handle, variant_pack)

            bench_result = executor.benchmark(
                handle, variant_pack, graph_name=graph_name
            )

            e2e_stats = BenchmarkStats.from_timings(bench_result.e2e_timings)
            gpu_kernel_stats = None
            if bench_result.has_kernel_timings:
                gpu_kernel_stats = BenchmarkStats.from_timings(
                    bench_result.kernel_timings
                )

            if ref_provider is not None:
                correctness = _check_correctness(
                    bm, tensor_infos, graph_json, ref_provider, config
                )
            elif validation_requested:
                # User asked for validation but the reference provider
                # didn't support this graph. Treat as a correctness
                # failure (per W3) so --validate stays a hard gate.
                correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=False,
                    rtol=config.rtol,
                    atol=config.atol,
                    error_message=(
                        f"Reference provider '{config.reference_provider}' "
                        f"does not support this graph"
                    ),
                )
            else:
                correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=None,
                    rtol=config.rtol,
                    atol=config.atol,
                    error_message="No reference provider requested",
                )

        return ProviderEngineResult(
            provider=provider,
            engine_id=engine_id,
            status="success",
            cpu_build_time_ms=cpu_build_time_ms,
            gpu_kernel_stats=gpu_kernel_stats,
            e2e_stats=e2e_stats,
            correctness=correctness,
        )

    except ExecutionError as e:
        error_msg = str(e)
        if _is_support_error(error_msg):
            return ProviderEngineResult(
                provider=provider,
                engine_id=engine_id,
                status="skipped",
                skip_reason=error_msg,
                correctness=CorrectnessResult.failed(
                    rtol=config.rtol, atol=config.atol, error_message=error_msg
                ),
            )
        return ProviderEngineResult(
            provider=provider,
            engine_id=engine_id,
            status="error",
            error_message=error_msg,
            correctness=CorrectnessResult.failed(
                rtol=config.rtol, atol=config.atol, error_message=error_msg
            ),
        )

    except Exception as e:
        error_msg = str(e)
        return ProviderEngineResult(
            provider=provider,
            engine_id=engine_id,
            status="error",
            error_message=error_msg,
            correctness=CorrectnessResult.failed(
                rtol=config.rtol, atol=config.atol, error_message=error_msg
            ),
        )
