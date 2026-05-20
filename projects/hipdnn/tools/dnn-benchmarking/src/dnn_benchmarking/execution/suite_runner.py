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

from ..common.exceptions import ExecutionError, UnsupportedGraphError
from ..config.benchmark_config import BenchmarkConfig, SuiteConfig
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor
from ..execution.timing import Timer
from ..metrics import (
    CpuTimeProbe,
    GpuSmiProbe,
    compute_flops,
    compute_io_bytes,
    derive_throughputs,
)
from ..metrics._diagnostic import warn_once
from ..reporting.reporter import Reporter
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


def _get_reference_provider(
    config: SuiteConfig, graph_json: Dict[str, Any]
) -> Optional[ReferenceProvider]:
    """Attempt to get and validate a reference provider for this graph.

    Args:
        config: Suite configuration with reference_provider name.
        graph_json: Parsed graph JSON dictionary.

    Returns:
        ReferenceProvider instance if available and supports the graph,
        None if validation was not requested (``config.reference_provider``
        is ``"none"``) or if the provider is unavailable/unsupported.
    """
    if config.reference_provider == "none":
        return None

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

    except (ValueError, RuntimeError, ExecutionError) as e:
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
    reporter: Optional[Reporter] = None,
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
    except UnsupportedGraphError as e:
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider="unknown",
                    engine_id=0,
                    status="skipped",
                    skip_reason=str(e),
                    correctness=CorrectnessResult.failed(
                        rtol=config.rtol, atol=config.atol, error_message=str(e)
                    ),
                )
            ],
        )
    except (ExecutionError, RuntimeError) as e:
        msg = str(e)
        return GraphResult(
            graph_name=graph_name,
            graph_path=str(graph_path),
            results=[
                ProviderEngineResult(
                    provider="unknown",
                    engine_id=0,
                    status="error",
                    error_message=f"Engine discovery failed: {msg}",
                    correctness=CorrectnessResult.failed(
                        rtol=config.rtol, atol=config.atol, error_message=msg
                    ),
                )
            ],
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

    # Compute analytical metrics once per graph — they're a function of
    # the graph shape only and don't change across engines. Both calls
    # are pure-Python and microsecond-cheap, but no point repeating
    # them per engine. Failures route through warn_once and yield None.
    analytical_flops: Optional[int] = None
    analytical_flops_partial = False
    analytical_io_bytes: Optional[int] = None
    if config.metrics.basic_enabled:
        try:
            analytical_flops, analytical_flops_partial = compute_flops(graph_json)
        except Exception as e:
            warn_once("analytical", f"compute_flops failed for {graph_name}: {e}")
        try:
            analytical_io_bytes = compute_io_bytes(tensor_infos)
        except Exception as e:
            warn_once("analytical", f"compute_io_bytes failed for {graph_name}: {e}")

    pe_results: List[ProviderEngineResult] = []
    for engine_id in engine_ids:
        engine_name = _resolve_engine_name(engine_id)
        if reporter is not None:
            reporter.print_engine_start(engine_name)
        with Timer() as t:
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
                analytical_flops=analytical_flops,
                analytical_flops_partial=analytical_flops_partial,
                analytical_io_bytes=analytical_io_bytes,
            )
        pe_result.elapsed_time_ms = t.elapsed_ms
        if reporter is not None:
            reporter.print_engine_result(pe_result)
        pe_results.append(pe_result)

    return GraphResult(
        graph_name=graph_name,
        graph_path=str(graph_path),
        results=pe_results,
        engine_ids=engine_ids,
    )


def _collect_basic_metrics_post_loop(
    result: ProviderEngineResult,
    cpu_time_probe: Optional[CpuTimeProbe],
    benchmark_iters: int,
    analytical_flops: Optional[int],
    analytical_flops_partial: bool,
    analytical_io_bytes: Optional[int],
) -> None:
    """Populate the basic always-on metric fields on ``result``.

    Called once after the timed loop when ``metrics.tier == "basic"``.
    Pulled out of :func:`_run_single_provider_engine` to keep that
    function focused on the timed loop itself; the basic-tier book
    keeping is otherwise just a long sequence of conditionals on
    intermediate results.
    """
    if cpu_time_probe is not None and cpu_time_probe.delta is not None:
        # Per-iter microseconds is the interpretable unit: the loop
        # total is dominated by Python dispatch cost, and per-iter
        # lets users compare directly against the kernel mean (also
        # reported per-iter).
        iters = max(benchmark_iters, 1)
        result.cpu_user_time_per_iter_us = (
            cpu_time_probe.delta.user_time_ms * 1000.0 / iters
        )
        result.cpu_kernel_time_per_iter_us = (
            cpu_time_probe.delta.kernel_time_ms * 1000.0 / iters
        )

    # Analytical totals were computed once at the graph level; propagate
    # them onto every engine's result so JSON consumers don't have to
    # look up across structures.
    result.analytical_flops = analytical_flops
    result.analytical_flops_partial = analytical_flops_partial
    result.analytical_io_bytes = analytical_io_bytes

    # Derived throughputs use the *arithmetic mean* of post-warmup kernel
    # timings — no trimming, no outlier rejection. A single noisy iter
    # (context switch, thermal throttle) skews the headline number; for
    # tighter signal use gpu_kernel_stats.min_ms or p95_ms.
    kernel_mean = (
        result.gpu_kernel_stats.mean_ms if result.gpu_kernel_stats is not None else None
    )
    tflops, gbytes = derive_throughputs(
        analytical_flops, analytical_io_bytes, kernel_mean
    )
    result.derived_tflops_per_s = tflops
    result.derived_gbytes_per_s = gbytes

    # VRAM is sampled here (still inside the BufferManager context, so
    # workspace + I/O buffers are still allocated). This is the only
    # amdsmi field we expose per-engine — see ProviderEngineResult
    # docstring.
    try:
        snap = GpuSmiProbe().snapshot()
        result.vram_used_mb = snap.get("vram_used_mb")
    except Exception as e:
        warn_once("gpu_smi", f"vram snapshot failed: {e}")


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
    analytical_flops: Optional[int] = None,
    analytical_flops_partial: bool = False,
    analytical_io_bytes: Optional[int] = None,
) -> ProviderEngineResult:
    """Execute a single engine for a graph (single attempt)."""
    # Initialise the result conservatively as an error and mutate fields as
    # the run progresses; on success, status flips to "success" at the end.
    result = ProviderEngineResult(
        provider=provider,
        engine_id=engine_id,
        status="error",
    )

    metrics_basic = config.metrics.basic_enabled

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
        if metrics_basic:
            result.workspace_bytes = executor.workspace_size

        with BufferManager(tensor_infos) as bm:
            bm.allocate_all()
            bm.fill_inputs_random(seed=config.seed)
            bm.zero_outputs()

            variant_pack = bm.create_variant_pack()
            executor.warmup(handle, variant_pack)

            # Wrap the timed loop with always-on host probes when basic
            # tier is enabled. The probes are designed to be no-cost on
            # the GPU side: CPU-time sampling is two read-only kernel
            # calls before/after the loop, and the amdsmi snapshot fires
            # once after the benchmark returns. Failures inside probes
            # are swallowed and surface as None fields on the result.
            cpu_time_probe = CpuTimeProbe() if metrics_basic else None
            if cpu_time_probe is not None:
                cpu_time_probe.__enter__()
            try:
                bench_result = executor.benchmark(
                    handle, variant_pack, graph_name=graph_name
                )
            finally:
                if cpu_time_probe is not None:
                    cpu_time_probe.__exit__(None, None, None)

            result.e2e_stats = BenchmarkStats.from_timings(bench_result.e2e_timings)
            if bench_result.has_kernel_timings:
                result.gpu_kernel_stats = BenchmarkStats.from_timings(
                    bench_result.kernel_timings
                )

            if metrics_basic:
                _collect_basic_metrics_post_loop(
                    result=result,
                    cpu_time_probe=cpu_time_probe,
                    benchmark_iters=config.benchmark_iters,
                    analytical_flops=analytical_flops,
                    analytical_flops_partial=analytical_flops_partial,
                    analytical_io_bytes=analytical_io_bytes,
                )

            # Opt-in profiling pass — runs *after* the timed pass and
            # always-on probes, so the profiler's overhead can't
            # pollute the headline numbers. The orchestrator handles
            # tool-missing / paranoid / parse failures internally and
            # never raises; we still wrap defensively because anything
            # bubbling out would otherwise mark a successful timed run
            # as failed.
            if config.metrics.opt_in_pass_requested:
                try:
                    from ..metrics.profiling_orchestrator import (
                        run_profiling_passes,
                    )

                    extra = run_profiling_passes(
                        graph_path=graph_path,
                        engine_id=engine_id,
                        seed=config.seed,
                        warmup_iters=config.warmup_iters,
                        benchmark_iters=config.benchmark_iters,
                        metrics_config=config.metrics,
                        plugin_path=config.plugin_path,
                    )
                    if extra:
                        result.extra_metrics = extra
                except Exception as e:
                    warn_once(
                        "profiling_orchestrator",
                        f"profiling pass failed: {e}",
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

    except UnsupportedGraphError as e:
        result.cpu_build_time_ms = None
        result.gpu_kernel_stats = None
        result.e2e_stats = None
        result.status = "skipped"
        result.skip_reason = str(e)
        result.correctness = CorrectnessResult.failed(
            rtol=config.rtol, atol=config.atol, error_message=str(e)
        )
        return result

    except ExecutionError as e:
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

    except (ValueError, RuntimeError, OSError) as e:
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
