# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite result data model with JSON serialization.

Implements the D-11 graph-first nesting structure, D-12 environment metadata,
D-13 full timing statistics, D-15 correctness with tolerances, and D-07 error
entries with message only.
"""

import json
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TimingStats:
    """Statistical summary of timing measurements.

    Per D-13: mean, std, min, max, p95, p99.

    Attributes:
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        p95_ms: 95th percentile in milliseconds.
        p99_ms: 99th percentile in milliseconds.
    """

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_timings(cls, timings: List[float]) -> "TimingStats":
        """Calculate statistics from raw timing list.

        Uses same numpy logic as BenchmarkStats.

        Args:
            timings: List of execution times in milliseconds.

        Returns:
            TimingStats with calculated statistics.

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
class CorrectnessResult:
    """Per D-15 and CORR-01/02/03: correctness tracking.

    Attributes:
        execution_success: CORR-01: did it execute without error?
        tolerance_match: CORR-02: within rtol/atol? None if execution failed
            or reference provider unavailable.
        rtol: D-15: relative tolerance used.
        atol: D-15: absolute tolerance used.
        max_abs_diff: Maximum absolute difference (if comparison was performed).
        max_rel_diff: Maximum relative difference (if comparison was performed).
        error_message: Explanation when tolerance_match is None.
    """

    execution_success: bool
    tolerance_match: Optional[bool]
    rtol: float
    atol: float
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def passed(self) -> bool:
        """CORR-03: overall pass = executed successfully AND tolerance matched."""
        return self.execution_success and (self.tolerance_match is True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: Dict[str, Any] = {
            "passed": self.passed,
            "execution_success": self.execution_success,
            "tolerance_match": self.tolerance_match,
            "rtol": self.rtol,
            "atol": self.atol,
        }
        if self.max_abs_diff is not None:
            d["max_abs_diff"] = self.max_abs_diff
        if self.max_rel_diff is not None:
            d["max_rel_diff"] = self.max_rel_diff
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d


@dataclass
class ProviderEngineResult:
    """Result for one provider/engine combination on one graph.

    Attributes:
        provider: Provider name.
        engine_id: Engine ID used.
        status: One of 'success', 'error', 'skipped'.
        cpu_build_time_ms: TIME-01: CPU graph-build time.
        gpu_kernel_stats: TIME-02: GPU kernel timing statistics.
        e2e_stats: TIME-03: End-to-end wall-clock timing statistics.
        correctness: CORR-01/02/03: correctness comparison result.
        error_message: D-07: error message only (no partial timing).
        skip_reason: D-02: reason for skip.
    """

    provider: str
    engine_id: int
    status: str
    cpu_build_time_ms: Optional[float] = None
    gpu_kernel_stats: Optional[TimingStats] = None
    e2e_stats: Optional[TimingStats] = None
    correctness: Optional[CorrectnessResult] = None
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Per D-07: error entries have status + error_message only, no timing.
        """
        d: Dict[str, Any] = {
            "provider": self.provider,
            "engine_id": self.engine_id,
            "status": self.status,
        }
        if self.status == "success":
            d["cpu_build_time_ms"] = self.cpu_build_time_ms
            d["gpu_kernel_stats"] = (
                self.gpu_kernel_stats.to_dict() if self.gpu_kernel_stats else None
            )
            d["e2e_stats"] = self.e2e_stats.to_dict() if self.e2e_stats else None
            d["correctness"] = (
                self.correctness.to_dict() if self.correctness else None
            )
        elif self.status == "error":
            d["error_message"] = self.error_message
        elif self.status == "skipped":
            d["skip_reason"] = self.skip_reason
        return d


@dataclass
class GraphResult:
    """Result for one graph across all provider/engine combinations.

    Attributes:
        graph_name: Name of the graph.
        graph_path: File path to the graph JSON.
        results: List of ProviderEngineResult for each combination.
    """

    graph_name: str
    graph_path: str
    results: List[ProviderEngineResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Per D-11: graph entry with 'results' array of provider/engine entries.
        """
        return {
            "graph_name": self.graph_name,
            "graph_path": self.graph_path,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class SuiteMetadata:
    """Per D-12: essential + environment info.

    Attributes:
        timestamp: UTC timestamp when suite was run.
        hostname: Machine hostname.
        total_graphs: Total number of graphs in suite.
        pass_count: Number of passed graphs.
        fail_count: Number of failed graphs.
        skip_count: Number of skipped graphs.
        rocm_version: ROCm version string.
        gpu_model: GPU model name.
        python_version: Python version string.
        hipdnn_version: hipDNN version string.
    """

    timestamp: str
    hostname: str
    total_graphs: int
    pass_count: int
    fail_count: int
    skip_count: int
    rocm_version: Optional[str] = None
    gpu_model: Optional[str] = None
    python_version: Optional[str] = None
    hipdnn_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "total_graphs": self.total_graphs,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "rocm_version": self.rocm_version,
            "gpu_model": self.gpu_model,
            "python_version": self.python_version,
            "hipdnn_version": self.hipdnn_version,
        }


@dataclass
class SuiteResult:
    """Per D-11: top-level suite result with graph-first nesting.

    Attributes:
        metadata: Suite-level metadata.
        graphs: List of per-graph results.
    """

    metadata: SuiteMetadata
    graphs: List[GraphResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns dict with "metadata" and "graphs" keys (per D-11).
        """
        return {
            "metadata": self.metadata.to_dict(),
            "graphs": [g.to_dict() for g in self.graphs],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        """Write suite results to JSON file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_json())


def collect_environment_info() -> Dict[str, Optional[str]]:
    """Collect ROCm version, GPU model, Python version, hipDNN version.

    Per D-12 metadata requirements.

    Returns:
        Dictionary with environment info fields.
    """
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )
    rocm_version: Optional[str] = None
    gpu_model: Optional[str] = None
    hipdnn_version: Optional[str] = None

    try:
        import torch

        if hasattr(torch.version, "hip"):
            rocm_version = torch.version.hip
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import hipdnn_frontend

        hipdnn_version = getattr(hipdnn_frontend, "__version__", None)
    except ImportError:
        pass

    return {
        "rocm_version": rocm_version,
        "gpu_model": gpu_model,
        "python_version": python_version,
        "hipdnn_version": hipdnn_version,
    }
