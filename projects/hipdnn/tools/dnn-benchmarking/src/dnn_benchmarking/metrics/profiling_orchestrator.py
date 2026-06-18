# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Re-exec orchestrator for opt-in profiling sources.

When the user passes ``--pmc``, ``--emit-trace``, ``--perf``, or
``--roofline``, the timed pass runs first to keep its numbers clean.
After it succeeds, this orchestrator runs the workload again — once per
requested source — under the corresponding external profiler (rocprofv3,
perf, rocprof-compute). The results are merged into a single dict that
populates ``ProviderEngineResult.extra_metrics``.

Architecture: the orchestrator builds a hidden re-exec argv that
re-invokes ``python -m dnn_benchmarking`` with the
``--internal-profiling-run`` sub-mode and a single
(graph, engine) pair. All profiling flags are stripped from the inner
argv to prevent infinite recursion. The sub-mode short-circuits engine
discovery and Reporter output; it just runs warmup + benchmark for the
given engine and exits.

Failures of individual sources never raise — each module returns a dict
slice with a ``skipped`` / ``error_tail`` / ``warnings`` key, and a
single ``warn_once`` line goes to stderr.
"""

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.benchmark_config import MetricsConfig
from ._diagnostic import warn_once
from . import perf as _perf_mod
from . import rocprof_pmc as _pmc_mod
from . import rocprof_trace as _trace_mod
from . import roofline as _roofline_mod


def resolve_output_dir(metrics_config: MetricsConfig) -> Path:
    """Pick the root profiling-output directory, defaulting to a UTC stamp.

    **Mutates the shared MetricsConfig instance**: on first call with
    ``profiling_output_dir is None``, this writes the resolved path
    back into the config so every subsequent (graph, engine) pair in
    the same suite lands under the same root. Without this, each engine
    would generate its own timestamped directory and per-suite output
    would stop being a single browsable tree.

    Callers that need a fresh resolution per call (e.g. parallel suites
    sharing a config) must clone the MetricsConfig first — today's
    sequential suite runner is the only caller and shares one config
    intentionally.
    """
    if metrics_config.profiling_output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        metrics_config.profiling_output_dir = Path("profiling-output") / stamp
    metrics_config.profiling_output_dir.mkdir(parents=True, exist_ok=True)
    return metrics_config.profiling_output_dir


# Characters that are safe in path segments across Linux filesystems
# without quoting at the shell. Anything else (slash, space, colon,
# brackets, $, &, ...) gets collapsed to '_' so a future plugin that
# returns an awkward engine name can't break the artifact tree or
# require the user to shell-quote artifact paths.
_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_segment(name: str) -> str:
    return _PATH_SAFE_RE.sub("_", name) or "unnamed"


def _graph_anchor(graph_path: Path) -> str:
    """The canonical string used both for the hash and the ``.source`` file.

    Resolves to an absolute path so it's stable across runs (and useful
    on its own when read back from ``.source``). Falls back to the
    string form when ``resolve()`` errors — typically a test path that
    doesn't exist on disk.
    """
    try:
        return str(graph_path.resolve())
    except OSError:
        return str(graph_path)


def _graph_segment(graph_path: Path) -> str:
    """``<stem>-<6 hex>`` disambiguator for the per-graph subdir.

    Suite mode accepts directories and globs, so two graphs with the same
    file stem (``a/conv.json`` and ``b/conv.json``) would otherwise
    collide on ``<root>/conv/<engine>/<source>/results.db`` and silently
    overwrite each other. The hash is derived from the resolved absolute
    path so it's stable across runs and distinguishes two graphs that
    share a stem.

    Hash length is 6 hex chars (24 bits) — collision probability across a
    realistic suite (<10⁴ graphs) is negligible, and 6 chars keeps the
    segment short enough to type when chasing an artifact path. The
    hash is opaque on its own; ``_subdir`` writes a ``.source`` file in
    the graph dir so users can map it back without re-hashing.
    """
    digest = hashlib.sha1(_graph_anchor(graph_path).encode("utf-8")).hexdigest()[:6]
    return f"{_safe_segment(graph_path.stem)}-{digest}"


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


def _subdir(out_dir: Path, graph_path: Path, engine_name: str, source: str) -> Path:
    """Per-source output directory:
    ``<out_dir>/<graph_stem>-<hash6>/<engine_name>/<source>/``.

    Three semantic levels under the user-controlled root: graph, then
    engine, then profiling source. Replaces the legacy flat scheme
    ``<graph>_<engine_id>_<source>/`` whose engine_id segment was a
    19-digit hash that no one could read or type.

    The graph segment carries a 6-hex disambiguator (see
    ``_graph_segment``) so same-stem graphs from different directories
    don't collide; engine name is sanitised because a future plugin
    could return slashes/spaces.
    """
    graph_dir = out_dir / _graph_segment(graph_path)
    sub = graph_dir / _safe_segment(engine_name) / _safe_segment(source)
    sub.mkdir(parents=True, exist_ok=True)
    # Drop a `.source` file at the graph-segment level so
    # `cat conv-7a3f1c/.source` answers "which graph is this?" without
    # the user having to recompute the hash. Idempotent — multiple
    # (engine, source) calls for the same graph overwrite with
    # identical content.
    (graph_dir / ".source").write_text(_graph_anchor(graph_path) + "\n")
    return sub


def run_profiling_passes(
    graph_path: Path,
    engine_id: int,
    engine_name: str,
    seed: Optional[int],
    warmup_iters: int,
    benchmark_iters: int,
    metrics_config: MetricsConfig,
    plugin_path: Optional[Path],
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every requested profiling source. Returns a merged dict.

    Source slices are merged at the top level so consumers can address
    them via ``extra_metrics["pmc"]``, ``extra_metrics["trace"]`` etc.

    Args:
        graph_path: Graph file passed to the inner process.
        engine_id: Single engine ID for the inner process.
        engine_name: Human-readable engine name (e.g.
            ``"MIOPEN_ENGINE"``) used as the per-engine output
            subdirectory; resolved by the caller via
            ``suite_runner._resolve_engine_name``.
        seed: Reproducibility seed for fill_inputs_random; passed
            through to the inner process so PMC counts are over the
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
        # Single iter — warmups should be enough to stabilise; re-evaluate
        # if any source shows noisy counters in practice.
        benchmark_iters=1,
        plugin_path=plugin_path,
    )

    aggregated: Dict[str, Any] = {}

    timeout_s = metrics_config.profiling_timeout_s

    if metrics_config.pmc_set is not None:
        try:
            aggregated.update(
                _pmc_mod.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(
                        out_dir,
                        graph_path,
                        engine_name,
                        f"pmc_{metrics_config.pmc_set}",
                    ),
                    pmc_set=metrics_config.pmc_set,
                    timeout_s=timeout_s,
                )
            )
        except Exception as e:
            warn_once("rocprof_pmc", f"unexpected error in PMC pass: {e}")
            aggregated.setdefault("pmc", {})["unexpected_error"] = str(e)

    if metrics_config.emit_trace is not None:
        try:
            aggregated.update(
                _trace_mod.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(
                        out_dir,
                        graph_path,
                        engine_name,
                        f"trace_{metrics_config.emit_trace}",
                    ),
                    fmt=metrics_config.emit_trace,
                    timeout_s=timeout_s,
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
                    out_dir=_subdir(out_dir, graph_path, engine_name, "perf"),
                    timeout_s=timeout_s,
                )
            )
        except Exception as e:
            warn_once("perf", f"unexpected error in perf pass: {e}")
            aggregated.setdefault("perf", {})["unexpected_error"] = str(e)

    if metrics_config.roofline:
        try:
            aggregated.update(
                _roofline_mod.run(
                    inner_argv=inner_argv,
                    out_dir=_subdir(out_dir, graph_path, engine_name, "roofline"),
                    timeout_s=timeout_s,
                )
            )
        except Exception as e:
            warn_once("roofline", f"unexpected error in roofline pass: {e}")
            aggregated.setdefault("roofline", {})["unexpected_error"] = str(e)

    return aggregated
