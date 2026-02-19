# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
        """Validate configuration values."""
        if isinstance(self.graph_path, str):
            self.graph_path = Path(self.graph_path)

        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")

        if self.benchmark_iters <= 0:
            raise ValueError("benchmark_iters must be positive")

        if self.engine_id < 0:
            raise ValueError("engine_id must be non-negative")


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

        if self.a_id < 0:
            raise ValueError("a_id must be non-negative")
        if self.b_id < 0:
            raise ValueError("b_id must be non-negative")
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
