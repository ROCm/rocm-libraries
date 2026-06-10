# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        graph_path: Path to the JSON-serialized hipDNN graph file.
        warmup_iters: Number of warmup iterations before benchmarking.
        benchmark_iters: Number of benchmark iterations for timing.
        engine_id: Engine ID to use (1 = MIOpen).
    """

    graph_path: Path
    warmup_iters: int = 10
    benchmark_iters: int = 100
    engine_id: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values.

        Note: engine_id is a 64-bit identifier (FNV-1a hash of the engine
        name) and may be negative when interpreted as signed int64, so we
        do not bound-check it.
        """
        if isinstance(self.graph_path, str):
            self.graph_path = Path(self.graph_path)

        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")

        if self.benchmark_iters <= 0:
            raise ValueError("benchmark_iters must be positive")


@dataclass
class ValidationConfig:
    """Configuration for reference validation.

    Attributes:
        provider: Reference provider name ("pytorch", "cpu_plugin", or "none").
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
    """

    provider: str = "none"
    rtol: float = 1e-5
    atol: float = 1e-8

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_providers = {"none", "pytorch", "cpu_plugin"}
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{self.provider}'. "
                f"Valid options: {valid_providers}"
            )
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")

    @property
    def enabled(self) -> bool:
        """Check if validation is enabled."""
        return self.provider != "none"


@dataclass
class MetricsConfig:
    """Controls which metric sources are collected during benchmarking.

    Two collection modes:

    * Always-on (``tier``) — zero-overhead probes wrapped around the
      timed loop: analytical FLOPs/IO, workspace, host rusage + RAM,
      amdsmi GPU snapshot, machine metadata.
    * Opt-in profiling pass (``pmc_set``, ``emit_trace``, ``perf``,
      ``roofline``) — each runs the workload again under an external
      profiling tool. Kept separate from the timed pass so PMC sampling
      and roofline replay don't pollute the headline timing.

    Attributes:
        tier: ``basic`` enables always-on probes. ``off`` disables all
            metric collection — useful for clean engine-comparison timing.
        emit_trace: ``pftrace`` or ``kineto`` — re-run benchmark under
            ``rocprofv3 --kernel-trace --memory-copy-trace`` and write a
            trace file. ``kineto`` falls back to pftrace if the rocpd
            Python module is not importable.
        pmc_set: ``basic`` | ``memory`` | ``flops`` | ``all`` — re-run
            under ``rocprofv3 --pmc <set>`` and fold per-kernel counter
            aggregates into ``extra_metrics["pmc"]``. ``all`` requires
            ``pmc_allow_multipass`` because the union of sets crosses the
            single-pass replay budget on most arches.
        perf: Re-run wrapped in ``perf stat -x,`` to collect CPU cycles
            and instructions. Kernel-space events are dropped silently
            when ``/proc/sys/kernel/perf_event_paranoid > 1``.
        roofline: Re-run under ``rocprof-compute profile --roof-only``
            to capture empirical HBM/compute ceilings at rocprof-
            compute's default datatype (FP32). The CSV outputs
            (``roofline.csv``, ``sysinfo.csv``) and the workload
            directory land in ``extra_metrics["roofline"]`` as
            ``roofline_csv`` / ``sysinfo_csv`` / ``workload_path``. The
            PDF/HTML plot is rendered post-hoc by the user via
            ``rocprof-compute analyze --path <workload_path>
            [--roofline-data-type FP16]``; we don't expose a profile-
            time datatype knob because rocprof-compute's ``profile``
            mode doesn't accept one.
        pmc_allow_multipass: Required for ``--pmc all``. Without it,
            ``all`` is rejected at config-build time because the rocprofv3
            multi-pass replay budget is easy to exceed and the resulting
            multi-pass replay has been observed to hang for minutes on
            sub-second workloads.
        profiling_output_dir: Root directory for profiling artefacts.
            ``None`` until the orchestrator resolves a default
            (``./profiling-output/<utc-timestamp>/``) at suite start.
        profiling_timeout_s: Wall-clock budget (seconds) for each external
            profiler subprocess. Default 600 s; ``0`` disables. Sized to
            absorb a heavy graph under multi-pass PMC replay on a slow
            host while still bounding a wedged child so the suite doesn't
            block indefinitely.
    """

    tier: Literal["basic", "off"] = "basic"
    emit_trace: Optional[Literal["pftrace", "kineto"]] = None
    pmc_set: Optional[Literal["basic", "memory", "flops", "all"]] = None
    perf: bool = False
    roofline: bool = False
    pmc_allow_multipass: bool = False
    profiling_output_dir: Optional[Path] = None
    profiling_timeout_s: int = 600

    def __post_init__(self) -> None:
        # Normalise empty / whitespace-only strings to None. The CLI
        # uses argparse `choices=` so this only matters for programmatic
        # / TOML callers, but the `Optional[Literal[...]]` annotation
        # isn't enforced at runtime: without this, ``pmc_set=""`` slips
        # past every downstream check (the `is not None` guards see a
        # truthy-empty string, `== "all"` is false, and the empty value
        # rides into the orchestrator as ``--pmc ""``).
        if isinstance(self.emit_trace, str) and not self.emit_trace.strip():
            self.emit_trace = None
        if isinstance(self.pmc_set, str) and not self.pmc_set.strip():
            self.pmc_set = None

        valid_tiers = {"basic", "off"}
        if self.tier not in valid_tiers:
            raise ValueError(
                f"Invalid metrics tier: '{self.tier}'. " f"Valid options: {valid_tiers}"
            )
        if self.emit_trace is not None and self.emit_trace not in {
            "pftrace",
            "kineto",
        }:
            raise ValueError(
                f"Invalid emit_trace: '{self.emit_trace}'. "
                "Valid options: pftrace, kineto"
            )
        if self.pmc_set is not None and self.pmc_set not in {
            "basic",
            "memory",
            "flops",
            "all",
        }:
            raise ValueError(
                f"Invalid pmc_set: '{self.pmc_set}'. "
                "Valid options: basic, memory, flops, all"
            )
        # The 'all' PMC set unions every counter group; rocprofv3 falls
        # back to multi-pass replay, which has been observed to hang for
        # minutes on what should be a sub-second run. Require the explicit
        # opt-in so users discover the cost.
        if self.pmc_set == "all" and not self.pmc_allow_multipass:
            raise ValueError(
                "--pmc all requires --pmc-allow-multipass: rocprofv3 "
                "falls back to multi-pass replay for the unioned counter "
                "set, which can hang on small workloads. Pick "
                "--pmc basic|memory|flops for a single-pass run."
            )
        if isinstance(self.profiling_output_dir, str):
            self.profiling_output_dir = Path(self.profiling_output_dir)
        if self.profiling_timeout_s < 0:
            raise ValueError(
                f"profiling_timeout_s must be >= 0 (0 disables); "
                f"got {self.profiling_timeout_s}"
            )

    @property
    def basic_enabled(self) -> bool:
        """True when always-on probes should run."""
        return self.tier == "basic"

    @property
    def opt_in_pass_requested(self) -> bool:
        """True when any opt-in profiling source was requested."""
        return bool(self.emit_trace or self.pmc_set or self.perf or self.roofline)

    @property
    def extra_runs_per_engine(self) -> int:
        """How many additional workload runs each opt-in source contributes.

        Each opt-in profiling source re-runs the workload once under its
        external tool. The basic always-on tier wraps the timed pass and
        does not add a run. Used by the reporter to give the user an
        upfront cost estimate.
        """
        return (
            int(self.pmc_set is not None)
            + int(self.emit_trace is not None)
            + int(self.perf)
            + int(self.roofline)
        )


@dataclass(frozen=True)
class EngineSelection:
    """One ordered engine execution selection.

    The plugin path is attached to the selection row rather than looked up by
    engine ID so repeated engine IDs can be benchmarked against different
    plugin builds.
    """

    engine_id: int
    plugin_path: Optional[Path] = None


@dataclass
class SuiteConfig:
    """Configuration for suite execution mode.

    Controls how the suite runner iterates providers/engines and validates
    correctness for each graph.

    Attributes:
        warmup_iters: Number of warmup iterations per provider/engine.
        benchmark_iters: Number of benchmark iterations for timing.
        seed: Optional random seed for reproducible inputs.
        engine_filter: If set, ordered engine selections to run.
        rtol: Optional relative tolerance override for correctness comparison.
            If only one of ``rtol`` or ``atol`` is set, the provided tolerance is
            used for both. If neither is set, validation uses dtype-aware
            defaults.
        atol: Optional absolute tolerance override for correctness comparison.
        gpu_backend: GPU timer backend to use.
        reference_provider: Reference provider name for correctness checking.
        verbose: If True, print rich per-engine block per graph instead of summary.
        metrics: Metric collection configuration. Defaults to ``basic`` tier
            (always-on probes, no extra runs).
    """

    warmup_iters: int = 10
    benchmark_iters: int = 100
    seed: Optional[int] = None
    engine_filter: Optional[List[int]] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    gpu_backend: str = "auto"
    reference_provider: str = "none"
    verbose: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    plugin_paths: Optional[List[Path]] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.benchmark_iters <= 0:
            raise ValueError("benchmark_iters must be positive")
        if self.rtol is not None and self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol is not None and self.atol < 0:
            raise ValueError("atol must be non-negative")
        if self.engine_filter is not None:
            if len(self.engine_filter) == 0:
                raise ValueError("engine_filter must be non-empty when set")
            # engine IDs are FNV-1a hashes -- may be negative as signed int64.
        if self.plugin_paths is not None:
            if len(self.plugin_paths) == 0:
                raise ValueError("plugin_paths must be non-empty when set")
            self.plugin_paths = [Path(p) for p in self.plugin_paths]

            if len(self.plugin_paths) > 1:
                if self.engine_filter is None:
                    raise ValueError(
                        "--plugin-path with multiple entries requires --engine"
                    )
                if len(self.plugin_paths) != len(self.engine_filter):
                    raise ValueError(
                        "--plugin-path entry count must be 1 or match --engine count"
                    )
        valid_gpu_backends = {"torch", "auto", "none"}
        if self.gpu_backend not in valid_gpu_backends:
            raise ValueError(
                f"Invalid gpu_backend: '{self.gpu_backend}'. "
                f"Valid options: {valid_gpu_backends}"
            )
        valid_reference_providers = {"none", "pytorch", "cpu_plugin"}
        if self.reference_provider not in valid_reference_providers:
            raise ValueError(
                f"Invalid reference_provider: '{self.reference_provider}'. "
                f"Valid options: {valid_reference_providers}"
            )

    @property
    def tolerance_override(self) -> Optional[tuple[float, float]]:
        """Return explicit validation tolerances, or None for dtype-aware defaults."""
        if self.rtol is None and self.atol is None:
            return None

        value = self.rtol if self.rtol is not None else self.atol
        return (
            self.rtol if self.rtol is not None else value,
            self.atol if self.atol is not None else value,
        )

    @property
    def plugin_path(self) -> Optional[Path]:
        """Return the shared plugin path when exactly one path is configured."""
        if self.plugin_paths is None or len(self.plugin_paths) != 1:
            return None
        return self.plugin_paths[0]

    def engine_selections_for(self, engine_ids: List[int]) -> List[EngineSelection]:
        """Return ordered engine selections for the provided engine IDs.

        ``engine_ids`` is either the explicit ``--engine`` list, where duplicate
        IDs are meaningful selections, or the backend-discovered engine list.
        Multiple plugin paths are only valid with an explicit engine list and
        are associated positionally with that list.
        """
        if self.plugin_paths is None:
            return [EngineSelection(engine_id) for engine_id in engine_ids]

        if len(self.plugin_paths) == 1:
            plugin_path = self.plugin_paths[0]
            return [
                EngineSelection(engine_id, plugin_path=plugin_path)
                for engine_id in engine_ids
            ]

        if self.engine_filter is None or len(engine_ids) != len(self.plugin_paths):
            raise ValueError(
                "--plugin-path entry count must be 1 or match --engine count"
            )

        return [
            EngineSelection(engine_id, plugin_path=plugin_path)
            for engine_id, plugin_path in zip(engine_ids, self.plugin_paths)
        ]
