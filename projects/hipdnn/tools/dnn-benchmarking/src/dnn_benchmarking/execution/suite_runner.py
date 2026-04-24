# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner for per-graph engine iteration with granular timing.

Iterates the engine IDs discovered for a graph via
``Graph.get_ranked_engine_ids`` (real runtime discovery, no hardcoded engine
lists). For each engine, captures separated CPU build time, GPU kernel time,
and E2E wall-clock time. Performs correctness validation by comparing GPU
output against a reference provider via ArrayComparator.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import BenchmarkConfig, SuiteConfig
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor
from ..reporting.statistics import BenchmarkStats
from ..reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
)
from ..validation.reference_provider import (
    ReferenceProvider,
    ReferenceProviderRegistry,
)
from ..validation.validator import Validator

# Keywords in error messages that indicate an unsupported combination
# rather than a hard error.
_SUPPORT_CHECK_KEYWORDS = (
    "support check failed",
    "not supported",
    "unsupported",
    "no engine",
)


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
        print(
            f"Reference provider '{config.reference_provider}' not registered",
            file=sys.stderr,
        )
        return None

    if not provider.is_available():
        print(
            f"Reference provider '{config.reference_provider}' not available",
            file=sys.stderr,
        )
        return None

    if not provider.supports_graph(graph_json):
        print(
            f"Reference provider '{config.reference_provider}' "
            "does not support this graph",
            file=sys.stderr,
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

    Compares GPU output against reference provider output using
    ArrayComparator. Validation was requested by caller (we are inside
    the ``ref_provider is not None`` branch), so when no outputs are
    comparable we report ``tolerance_match=False`` rather than silently
    passing.

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

        # Delegate per-output comparison to Validator and aggregate results.
        validator = Validator(rtol=config.rtol, atol=config.atol)
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
            result = validator.validate(actual, ti, reference_data=expected)
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
            # so --validate stays a hard gate.
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

    Discovers engine IDs via ``Graph.get_ranked_engine_ids``, applies any
    user engine filter, and runs the benchmark for each remaining ID.
    Captures separated CPU build time, GPU kernel time, and E2E wall-clock
    time per engine. Performs correctness checking against a reference
    provider when ``--validate`` was requested.

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
    discovery_config = BenchmarkConfig(
        graph_path=graph_path,
        warmup_iters=config.warmup_iters,
        benchmark_iters=config.benchmark_iters,
    )
    try:
        discovery_executor = Executor(
            graph_json_str=graph_json_str,
            config=discovery_config,
            gpu_backend=config.gpu_backend,
        )
        engine_ids = discovery_executor.discover_engines(handle)
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
    """Execute a single engine for a graph (single attempt)."""
    # Initialise the result conservatively as an error and mutate fields as
    # the run progresses; on success, status flips to "success" at the end.
    result = ProviderEngineResult(
        provider=provider,
        engine_id=engine_id,
        status="error",
    )

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
        result.cpu_build_time_ms = executor.init_time_ms

        with BufferManager(tensor_infos) as bm:
            bm.allocate_all()
            bm.fill_inputs_random(seed=config.seed)
            bm.zero_outputs()

            variant_pack = bm.create_variant_pack()
            executor.warmup(handle, variant_pack)

            bench_result = executor.benchmark(
                handle, variant_pack, graph_name=graph_name
            )

            result.e2e_stats = BenchmarkStats.from_timings(bench_result.e2e_timings)
            if bench_result.has_kernel_timings:
                result.gpu_kernel_stats = BenchmarkStats.from_timings(
                    bench_result.kernel_timings
                )

            if ref_provider is not None:
                result.correctness = _check_correctness(
                    bm, tensor_infos, graph_json, ref_provider, config
                )
            elif validation_requested:
                # User asked for validation but the reference provider
                # didn't support this graph. Treat as a correctness
                # failure so --validate stays a hard gate.
                result.correctness = CorrectnessResult(
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
                result.correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=None,
                    rtol=config.rtol,
                    atol=config.atol,
                    error_message="No reference provider requested",
                )

        result.status = "success"
        return result

    except ExecutionError as e:
        error_msg = str(e)
        # Drop any partial timing collected before the failure so the JSON
        # output never carries half-populated stats on an error/skip entry.
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.e2e_stats = None
        result.correctness = CorrectnessResult.failed(
            rtol=config.rtol, atol=config.atol, error_message=error_msg
        )
        if _is_support_error(error_msg):
            result.status = "skipped"
            result.skip_reason = error_msg
        else:
            result.status = "error"
            result.error_message = error_msg
        return result

    except Exception as e:
        error_msg = str(e)
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.e2e_stats = None
        result.status = "error"
        result.error_message = error_msg
        result.correctness = CorrectnessResult.failed(
            rtol=config.rtol, atol=config.atol, error_message=error_msg
        )
        return result
