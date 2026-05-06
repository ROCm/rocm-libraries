# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Benchmark statistics calculation."""

import json
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _get_hostname() -> str:
    """Get machine hostname for result identification."""
    return socket.gethostname()


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkStats:
    """Statistics from benchmark execution.

    Attributes:
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation of execution time in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        p95_ms: 95th percentile execution time in milliseconds.
        p99_ms: 99th percentile execution time in milliseconds.
    """

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_timings(cls, timings: List[float]) -> "BenchmarkStats":
        """Calculate statistics from a list of timing values.

        Args:
            timings: List of execution times in milliseconds.

        Returns:
            BenchmarkStats with calculated statistics.

        Raises:
            ValueError: If timings list is empty.
        """
        if not timings:
            raise ValueError("timings list cannot be empty")

        arr = np.array(timings)

        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }


@dataclass
class BenchmarkMetadata:
    """Metadata for benchmark results export.

    Attributes:
        timestamp: UTC timestamp when benchmark was run.
        graph_name: Name/identifier of the graph being benchmarked.
        graph_path: Path to the graph JSON file.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
        engine_id: Engine ID used for execution.
        gpu_backend: GPU timer backend used ("torch" or "").
        execution_backend: Execution backend used ("hipdnn", "pytorch", or "").
        hostname: Machine hostname where benchmark was run.
    """

    timestamp: str = field(default_factory=_get_timestamp)
    graph_name: str = ""
    graph_path: str = ""
    warmup_iters: int = 0
    benchmark_iters: int = 0
    engine_id: int = 0
    gpu_backend: str = ""
    execution_backend: str = ""
    hostname: str = field(default_factory=_get_hostname)


@dataclass
class BenchmarkResult:
    """Raw benchmark timing results.

    Holds both E2E (wall-clock) and optional kernel (GPU event) timings,
    with metadata for cross-device comparison.

    Attributes:
        e2e_timings: List of end-to-end execution times in milliseconds.
        kernel_timings: Optional list of GPU kernel times in milliseconds.
        metadata: Optional metadata for result identification and comparison.
    """

    e2e_timings: List[float]
    kernel_timings: Optional[List[float]] = None
    metadata: Optional[BenchmarkMetadata] = None

    @property
    def has_kernel_timings(self) -> bool:
        """Check if kernel timings are available."""
        return self.kernel_timings is not None and len(self.kernel_timings) > 0

    @property
    def gpu_backend(self) -> str:
        """Return GPU backend used for kernel timing."""
        if self.metadata:
            return self.metadata.gpu_backend
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the result.
        """
        result: Dict[str, Any] = {
            "e2e_timings": self.e2e_timings,
            "kernel_timings": self.kernel_timings,
        }
        if self.metadata:
            result["metadata"] = asdict(self.metadata)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Save results to JSON file.

        Args:
            path: Path to the output JSON file.
        """
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary.

        Args:
            data: Dictionary with result data.

        Returns:
            BenchmarkResult instance.
        """
        metadata = None
        if "metadata" in data and data["metadata"]:
            metadata = BenchmarkMetadata(**data["metadata"])
        return cls(
            e2e_timings=data["e2e_timings"],
            kernel_timings=data.get("kernel_timings"),
            metadata=metadata,
        )

    @classmethod
    def load_json(cls, path: str) -> "BenchmarkResult":
        """Load results from JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            BenchmarkResult loaded from file.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class CombinedBenchmarkStats:
    """Combined statistics for E2E and kernel timing.

    Attributes:
        e2e_stats: Statistics from wall-clock timing.
        kernel_stats: Optional statistics from GPU kernel timing.
    """

    e2e_stats: BenchmarkStats
    kernel_stats: Optional[BenchmarkStats] = None

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> "CombinedBenchmarkStats":
        """Create combined stats from a BenchmarkResult.

        Args:
            result: BenchmarkResult with E2E and optional kernel timings.

        Returns:
            CombinedBenchmarkStats with calculated statistics.
        """
        e2e = BenchmarkStats.from_timings(result.e2e_timings)
        kernel = (
            BenchmarkStats.from_timings(result.kernel_timings)
            if result.has_kernel_timings
            else None
        )
        return cls(e2e_stats=e2e, kernel_stats=kernel)
