# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Hidden ``--internal-profiling-run`` sub-mode.

This is the workload that the profiling orchestrator wraps under
rocprofv3 / perf / rocprof-compute. We deliberately re-exec the whole
process (rather than running another loop in-place) because the outer
profiler expects a fresh process tree and a clean address space — that
is how kernel-trace / PMC counters scope what they record.

Short-circuits relative to the full CLI:
  * No backend startup checks — the parent already constructed the
    hipDNN handle and passed the exact engine ID.
  * No Reporter console output — the profiler scrapes stderr and writes
    its own files; chatty stdout only confuses log capture.
  * No always-on metrics, no JSON output.
"""

import argparse
import json
import sys
from pathlib import Path

from ..common.exceptions import GraphLoadError
from ..config.benchmark_config import MetricsConfig, SuiteConfig
from ..execution.buffer_manager import generate_input_data
from ..execution.suite_runner import run_single_provider_engine, set_plugin_path
from ..graph.loader import GraphLoader


def run_internal_profiling(args: argparse.Namespace) -> int:
    """Run a single (graph, engine) workload for the orchestrator. Returns exit code."""
    graph_path: Path = args.internal_profiling_graph
    engine_id: int = args.internal_profiling_engine
    if graph_path is None or engine_id is None:
        print(
            "internal-profiling-run requires --internal-profiling-graph and "
            "--internal-profiling-engine",
            file=sys.stderr,
        )
        return 1

    try:
        import hipdnn_frontend as hipdnn
    except ImportError:
        print(
            "internal-profiling-run: hipdnn_frontend not importable",
            file=sys.stderr,
        )
        return 1

    plugin_path = None
    if args.plugin_path:
        if len(args.plugin_path) != 1:
            print(
                "internal-profiling-run: expected exactly one --plugin-path",
                file=sys.stderr,
            )
            return 1
        plugin_path = args.plugin_path[0]

    try:
        set_plugin_path(hipdnn, plugin_path)
        handle = hipdnn.Handle()
    except RuntimeError as e:
        print(
            f"internal-profiling-run: failed to create hipDNN handle: {e}",
            file=sys.stderr,
        )
        return 1

    try:
        loader = GraphLoader()
        graph_json = loader.load_json(graph_path)
        loader.validate(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)
    except GraphLoadError as e:
        print(f"internal-profiling-run: graph load failed: {e}", file=sys.stderr)
        return 1

    # Always-on probes are pure CPU work and would be measured by perf
    # if we left them on, polluting the inner run. Force tier=off for
    # the inner pass; the parent already collected basic metrics on the
    # timed pass.
    #
    # `plugin_path` is forwarded so the child SuiteConfig matches the
    # parent's selected engine/plugin row. The outer suite runner passes
    # exactly one plugin path for this single-engine subprocess.
    suite_config = SuiteConfig(
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        seed=args.seed,
        engine_filter=[engine_id],
        gpu_backend="auto",
        reference_provider="none",
        verbose=False,
        metrics=MetricsConfig(tier="off"),
        plugin_paths=[plugin_path] if plugin_path is not None else None,
    )

    try:
        result = run_single_provider_engine(
            graph_path=graph_path,
            graph_json_str=json.dumps(graph_json),
            graph_name=graph_json.get("name", graph_path.stem),
            tensor_infos=tensor_infos,
            config=suite_config,
            handle=handle,
            provider="profiling-inner",
            engine_id=engine_id,
            plugin_path=plugin_path,
            reference_outputs=None,
            reference_error=None,
            input_data=generate_input_data(tensor_infos, args.seed),
            validation_requested=False,
            graph_json=graph_json,
        )
    except Exception as e:
        print(f"internal-profiling-run: execution failed: {e}", file=sys.stderr)
        return 1

    if result.status != "success":
        msg = result.error_message or result.skip_reason or "unknown"
        print(
            f"internal-profiling-run: engine {engine_id} status={result.status}: {msg}",
            file=sys.stderr,
        )
        return 1
    # Quiet success — the profiler captures everything it needs from
    # the run itself. Avoid emitting anything on stdout so trace-format
    # consumers don't accidentally treat our text as part of their data.
    return 0
