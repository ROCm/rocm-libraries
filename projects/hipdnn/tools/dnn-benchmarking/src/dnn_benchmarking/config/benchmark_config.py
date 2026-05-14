# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


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
class ABTestConfig:
    """Configuration for A/B testing mode.

    Attributes:
        a_path: Plugin path for configuration A (None = default).
        a_id: Engine ID for configuration A.
        b_path: Plugin path for configuration B (None = default).
        b_id: Engine ID for configuration B.
        rtol: Relative tolerance for np.allclose comparison.
        atol: Absolute tolerance for np.allclose comparison.
    """

    a_path: Optional[Path] = None
    a_id: int = 1
    b_path: Optional[Path] = None
    b_id: int = 1
    rtol: float = 1e-5
    atol: float = 1e-8

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if isinstance(self.a_path, str):
            self.a_path = Path(self.a_path)
        if isinstance(self.b_path, str):
            self.b_path = Path(self.b_path)

        # a_id / b_id are FNV-1a engine ID hashes that may be negative when
        # interpreted as signed int64; do not bound-check them.
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")

    def validate_paths(self) -> None:
        """Validate that plugin paths exist if specified.

        Raises:
            ValueError: If a specified path does not exist.
        """
        if self.a_path is not None and not self.a_path.exists():
            raise ValueError(f"Plugin path A does not exist: {self.a_path}")
        if self.b_path is not None and not self.b_path.exists():
            raise ValueError(f"Plugin path B does not exist: {self.b_path}")


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
    * Opt-in profiling pass (``emit_trace``, ``perf``) — each runs the
      workload again under an external profiling tool. Kept separate
      from the timed pass so the profiler's overhead doesn't pollute
      the headline timing.

    Attributes:
        tier: ``basic`` enables always-on probes. ``off`` disables all
            metric collection — useful for clean A/B timing comparisons.
        emit_trace: ``pftrace`` or ``kineto`` — re-run benchmark under
            ``rocprofv3 --kernel-trace --memory-copy-trace`` and write a
            trace file. ``kineto`` falls back to pftrace if the rocpd
            Python module is not importable.
        perf: Re-run wrapped in ``perf stat -x,`` to collect CPU cycles
            and instructions. Kernel-space events are dropped silently
            when ``/proc/sys/kernel/perf_event_paranoid > 1``.
        profiling_output_dir: Root directory for profiling artefacts.
            ``None`` until the orchestrator resolves a default
            (``./profiling-output/<utc-timestamp>/``) at suite start.
    """

    tier: Literal["basic", "off"] = "basic"
    emit_trace: Optional[Literal["pftrace", "kineto"]] = None
    perf: bool = False
    profiling_output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
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
        if isinstance(self.profiling_output_dir, str):
            self.profiling_output_dir = Path(self.profiling_output_dir)

    @property
    def basic_enabled(self) -> bool:
        """True when always-on probes should run."""
        return self.tier == "basic"

    @property
    def opt_in_pass_requested(self) -> bool:
        """True when any opt-in profiling source was requested."""
        return bool(self.emit_trace or self.perf)

    @property
    def extra_runs_per_engine(self) -> int:
        """How many additional workload runs each opt-in source contributes.

        Each opt-in profiling source re-runs the workload once under its
        external tool. The basic always-on tier wraps the timed pass and
        does not add a run. Used by the reporter to give the user an
        upfront cost estimate.
        """
        return int(self.emit_trace is not None) + int(self.perf)


@dataclass
class SuiteConfig:
    """Configuration for suite execution mode.

    Controls how the suite runner iterates providers/engines and validates
    correctness for each graph.

    Attributes:
        warmup_iters: Number of warmup iterations per provider/engine.
        benchmark_iters: Number of benchmark iterations for timing.
        seed: Optional random seed for reproducible inputs.
        engine_filter: If set, only iterate engine IDs in this list.
        rtol: Relative tolerance for correctness comparison.
        atol: Absolute tolerance for correctness comparison.
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
    rtol: float = 1e-5
    atol: float = 1e-8
    gpu_backend: str = "auto"
    reference_provider: str = "none"
    verbose: bool = False
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # Forwarded to the orchestrator's inner subprocess so the child
    # picks up the same plugin .so directory the parent loaded. Not used
    # outside of the opt-in profiling path.
    plugin_path: Optional[Path] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.benchmark_iters <= 0:
            raise ValueError("benchmark_iters must be positive")
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")
        if self.engine_filter is not None:
            if len(self.engine_filter) == 0:
                raise ValueError("engine_filter must be non-empty when set")
            # engine IDs are FNV-1a hashes -- may be negative as signed int64.
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
