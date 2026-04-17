# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite runner for per-graph provider/engine iteration with granular timing.

Iterates all available hipDNN providers and engines for a single graph,
capturing separated CPU build time (TIME-01), GPU kernel time (TIME-02),
and E2E wall-clock time (TIME-03) per combination. Performs correctness
validation by comparing GPU output against a reference provider via
ArrayComparator (CORR-02).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import BenchmarkConfig, SuiteConfig
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor
from ..reporting.suite_results import (
    CorrectnessResult,
    GraphResult,
    ProviderEngineResult,
    TimingStats,
)
from ..validation.comparison import ArrayComparator
from ..validation.reference_provider import (
    ReferenceProvider,
    ReferenceProviderRegistry,
)

logger = logging.getLogger(__name__)

# Known providers to attempt when dynamic discovery is unavailable.
# The runner records skipped/error for any that fail at runtime.
_DEFAULT_PROVIDER_NAMES = ["miopen"]

# Keywords in error messages that indicate an unsupported combination
# rather than a hard error.
_SUPPORT_CHECK_KEYWORDS = (
    "support check failed",
    "not supported",
    "unsupported",
    "no engine",
)


def discover_providers(handle: Any) -> List[str]:
    """Discover available hipDNN providers at runtime.

    Attempts to use the hipDNN API to enumerate provider names. Falls back
    to a minimal default list if the API does not expose enumeration.

    Per D-01: no hardcoded provider list -- use dynamic discovery.

    Args:
        handle: hipdnn.Handle instance.

    Returns:
        List of provider name strings.
    """
    # Try hipDNN API for provider enumeration
    try:
        if hasattr(handle, "get_provider_names"):
            return list(handle.get_provider_names())
    except Exception:
        pass

    try:
        import hipdnn_frontend as hipdnn

        if hasattr(hipdnn, "get_available_providers"):
            return list(hipdnn.get_available_providers(handle))
    except (ImportError, Exception):
        pass

    # Fallback: attempt to get the provider list from the handle
    try:
        if hasattr(handle, "get_providers"):
            return list(handle.get_providers())
    except Exception:
        pass

    # Last resort: return a list of commonly known providers to attempt.
    # The runner will record skipped/error for any that don't work.
    logger.warning(
        "Could not enumerate providers via hipDNN API; "
        "falling back to default provider list"
    )
    return list(_DEFAULT_PROVIDER_NAMES)


def discover_engines(handle: Any, provider: str) -> List[int]:
    """Discover available engine IDs for a given provider.

    Per D-01: dynamic discovery of engine IDs.

    Args:
        handle: hipdnn.Handle instance.
        provider: Provider name string.

    Returns:
        List of engine ID integers.
    """
    try:
        if hasattr(handle, "get_engine_ids"):
            return list(handle.get_engine_ids(provider))
    except Exception:
        pass

    try:
        import hipdnn_frontend as hipdnn

        if hasattr(hipdnn, "get_available_engines"):
            return list(hipdnn.get_available_engines(handle, provider))
    except (ImportError, Exception):
        pass

    # Fallback: return a reasonable default set of engine IDs to try.
    # The runner records skipped/error for unsupported ones.
    return [0, 1]


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
    """Attempt to get and validate a reference provider.

    Args:
        config: Suite configuration with reference_provider name.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        ReferenceProvider instance if available and supports the graph,
        None otherwise.
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
    using ArrayComparator.

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

        tolerance_match = all_passed if output_count > 0 else None
        error_message = None if output_count > 0 else "No output tensors to compare"

        return CorrectnessResult(
            execution_success=True,
            tolerance_match=tolerance_match,
            rtol=config.rtol,
            atol=config.atol,
            max_abs_diff=worst_abs_diff if output_count > 0 else None,
            max_rel_diff=worst_rel_diff if output_count > 0 else None,
            error_message=error_message,
        )

    except Exception as e:
        return CorrectnessResult(
            execution_success=True,
            tolerance_match=None,
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
    """Run a single graph against all available providers and engines.

    For each (provider, engine) combination, captures separated CPU build
    time, GPU kernel time, and E2E wall-clock time. Performs correctness
    checking against a reference provider when available.

    Args:
        graph_path: Path to the graph JSON file.
        graph_json: Parsed graph JSON dictionary.
        tensor_infos: List of TensorInfo objects for the graph.
        config: Suite configuration.
        handle: hipdnn.Handle instance.

    Returns:
        GraphResult with one ProviderEngineResult per provider/engine combo.
    """
    graph_name = graph_json.get("name", graph_path.stem)
    graph_json_str = json.dumps(graph_json)

    # Discover available providers
    providers = discover_providers(handle)

    # Get reference provider once (outside the loop)
    ref_provider = _get_reference_provider(config, graph_json)

    pe_results: List[ProviderEngineResult] = []

    for provider in providers:
        # Discover engines for this provider
        engines = discover_engines(handle, provider)

        # Apply engine filter (D-03)
        if config.engine_filter is not None:
            engines = [e for e in engines if e == config.engine_filter]

        for engine_id in engines:
            pe_result = _run_single_provider_engine(
                graph_json_str=graph_json_str,
                graph_name=graph_name,
                tensor_infos=tensor_infos,
                config=config,
                handle=handle,
                provider=provider,
                engine_id=engine_id,
                ref_provider=ref_provider,
                graph_json=graph_json,
            )
            pe_results.append(pe_result)

    return GraphResult(
        graph_name=graph_name,
        graph_path=str(graph_path),
        results=pe_results,
    )


def _run_single_provider_engine(
    graph_json_str: str,
    graph_name: str,
    tensor_infos: list,
    config: SuiteConfig,
    handle: Any,
    provider: str,
    engine_id: int,
    ref_provider: Optional[ReferenceProvider],
    graph_json: Dict[str, Any],
) -> ProviderEngineResult:
    """Execute a single provider/engine combination.

    Single attempt, no retry (per D-10).

    Args:
        graph_json_str: Graph as JSON string.
        graph_name: Human-readable graph name.
        tensor_infos: List of TensorInfo objects.
        config: Suite configuration.
        handle: hipdnn.Handle instance.
        provider: Provider name.
        engine_id: Engine ID to use.
        ref_provider: Optional reference provider for correctness checking.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        ProviderEngineResult for this combination.
    """
    try:
        # Create a fresh BenchmarkConfig for this combination
        bench_config = BenchmarkConfig(
            graph_path=Path("unused"),  # graph_json_str used directly
            warmup_iters=config.warmup_iters,
            benchmark_iters=config.benchmark_iters,
            engine_id=engine_id,
        )

        # Create executor and prepare (CPU build step -- TIME-01)
        executor = Executor(
            graph_json_str=graph_json_str,
            config=bench_config,
            gpu_backend=config.gpu_backend,
        )
        executor.prepare(handle, engine_id=engine_id)
        cpu_build_time_ms = executor.init_time_ms

        # Allocate buffers and run benchmark
        with BufferManager(tensor_infos) as bm:
            bm.allocate_all()
            bm.fill_inputs_random(seed=config.seed)
            bm.zero_outputs()

            variant_pack = bm.create_variant_pack()

            # Warmup
            executor.warmup(handle, variant_pack)

            # Benchmark -- captures e2e and kernel timings (TIME-02/TIME-03)
            bench_result = executor.benchmark(
                handle, variant_pack, graph_name=graph_name
            )

            # Build timing stats
            e2e_stats = TimingStats.from_timings(bench_result.e2e_timings)
            gpu_kernel_stats = None
            if bench_result.has_kernel_timings:
                gpu_kernel_stats = TimingStats.from_timings(bench_result.kernel_timings)

            # Correctness check (CORR-02)
            if ref_provider is not None:
                correctness = _check_correctness(
                    bm, tensor_infos, graph_json, ref_provider, config
                )
            else:
                correctness = CorrectnessResult(
                    execution_success=True,
                    tolerance_match=None,
                    rtol=config.rtol,
                    atol=config.atol,
                    error_message="No reference provider available",
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
        # Determine if this is a support issue or hard error
        if _is_support_error(error_msg):
            return ProviderEngineResult(
                provider=provider,
                engine_id=engine_id,
                status="skipped",
                skip_reason=error_msg,
                correctness=CorrectnessResult(
                    execution_success=False,
                    tolerance_match=None,
                    rtol=config.rtol,
                    atol=config.atol,
                    error_message=error_msg,
                ),
            )
        return ProviderEngineResult(
            provider=provider,
            engine_id=engine_id,
            status="error",
            error_message=error_msg,
            correctness=CorrectnessResult(
                execution_success=False,
                tolerance_match=None,
                rtol=config.rtol,
                atol=config.atol,
                error_message=error_msg,
            ),
        )

    except Exception as e:
        error_msg = str(e)
        return ProviderEngineResult(
            provider=provider,
            engine_id=engine_id,
            status="error",
            error_message=error_msg,
            correctness=CorrectnessResult(
                execution_success=False,
                tolerance_match=None,
                rtol=config.rtol,
                atol=config.atol,
                error_message=error_msg,
            ),
        )
