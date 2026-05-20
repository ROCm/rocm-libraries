# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Re-exec orchestrator for opt-in profiling sources.

When the user passes ``--emit-trace`` or ``--perf``, the timed pass
runs first to keep its numbers clean. After it succeeds, this
orchestrator runs the workload again — once per requested source —
under the corresponding external profiler (rocprofv3 trace, perf).
The results are merged into a single dict that populates
``ProviderEngineResult.extra_metrics``.

Architecture: the orchestrator builds a hidden re-exec argv that
re-invokes ``python -m dnn_benchmarking`` with the
``--internal-profiling-run`` sub-mode and a single
(graph, engine) pair. All profiling flags are stripped from the inner
argv to prevent infinite recursion. The sub-mode short-circuits engine
discovery, Reporter output, and gpu_check; it just runs warmup +
benchmark for the given engine and exits.

Failures of individual sources never raise — each module returns a dict
slice with a ``skipped`` / ``error_tail`` / ``warnings`` key, and a
single ``warn_once`` line goes to stderr.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import MetricsConfig
from ._diagnostic import warn_once
from . import perf as _perf_mod
from . import rocprof_trace as _trace_mod


def resolve_output_dir(metrics_config: MetricsConfig) -> Path:
    """Pick the root profiling-output directory, defaulting to a UTC stamp.

    Mutates ``metrics_config.profiling_output_dir`` so the same path is
    reused across (graph, engine) pairs in one suite.
    """
    if metrics_config.profiling_output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        metrics_config.profiling_output_dir = Path("profiling-output") / stamp
    metrics_config.profiling_output_dir.mkdir(parents=True, exist_ok=True)
    return metrics_config.profiling_output_dir


def build_inner_argv(
    graph_path: Path,
    engine_id: int,
    seed: Optional[int],
    warmup_iters: int,
    benchmark_iters: int,
    plugin_path: Optional[Path],
) -> List[str]:
    """Construct the argv for the ``--internal-profiling-run`` sub-mode.

    Always omits any opt-in profiling flag so the child process can't
    recurse back into the orchestrator.
    """
    argv = [
        sys.executable,
        "-m",
        "dnn_benchmarking",
        "--internal-profiling-run",
        "--internal-profiling-graph",
        str(graph_path),
        "--internal-profiling-engine",
        str(engine_id),
        "--graph",
        str(graph_path),
        "--warmup",
        str(warmup_iters),
        "--iters",
        str(benchmark_iters),
        "--engine",
        str(engine_id),
        "--metrics-tier",
        "off",
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if plugin_path is not None:
        argv += ["--plugin-path", str(plugin_path)]
    return argv


def _subdir(out_dir: Path, graph_path: Path, engine_id: int, source: str) -> Path:
    name = f"{graph_path.stem}_{engine_id}_{source}"
    sub = out_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def run_profiling_passes(
    graph_path: Path,
    engine_id: int,
    seed: Optional[int],
    warmup_iters: int,
    benchmark_iters: int,
    metrics_config: MetricsConfig,
    plugin_path: Optional[Path],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every requested profiling source. Returns a merged dict.

    Source slices are merged at the top level so consumers can address
    them via ``extra_metrics["trace"]``, ``extra_metrics["perf"]`` etc.

    Args:
        graph_path: Graph file passed to the inner process.
        engine_id: Single engine ID for the inner process.
        seed: Reproducibility seed for fill_inputs_random; passed
            through to the inner process so the profiler observes the
            same input distribution as the timed pass.
        warmup_iters: Inner warmup iteration count.
        benchmark_iters: Inner benchmark iteration count.
        metrics_config: Decides which sources fire.
        plugin_path: Optional plugin path forwarded to the inner CLI.
        out_dir: Override the resolved profiling-output root (test hook).

    Never raises. Source-specific failures end up in their slice's
    ``skipped`` / ``error_tail`` / ``warnings`` keys.
    """
    if not metrics_config.opt_in_pass_requested:
        return {}

    if out_dir is None:
        out_dir = resolve_output_dir(metrics_config)
    inner_argv = build_inner_argv(
        graph_path=graph_path,
        engine_id=engine_id,
        seed=seed,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        plugin_path=plugin_path,
    )

    aggregated: Dict[str, Any] = {}

    if metrics_config.emit_trace is not None:
        try:
            aggregated.update(
                _trace_mod.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(
                        out_dir,
                        graph_path,
                        engine_id,
                        f"trace_{metrics_config.emit_trace}",
                    ),
                    fmt=metrics_config.emit_trace,
                )
            )
        except Exception as e:
            warn_once("rocprof_trace", f"unexpected error in trace pass: {e}")
            aggregated.setdefault("trace", {})["unexpected_error"] = str(e)

    if metrics_config.perf:
        try:
            aggregated.update(
                _perf_mod.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(out_dir, graph_path, engine_id, "perf"),
                )
            )
        except Exception as e:
            warn_once("perf", f"unexpected error in perf pass: {e}")
            aggregated.setdefault("perf", {})["unexpected_error"] = str(e)

    return aggregated
