# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark configuration dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
