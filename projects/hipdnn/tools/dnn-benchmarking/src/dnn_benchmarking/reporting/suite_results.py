# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Suite result data model with JSON serialization.

Top-level structure is graph-first: SuiteResult contains metadata plus a
list of GraphResult, each holding ProviderEngineResult entries with timing
statistics and correctness data. Error entries carry status + message only.
"""

import json
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional

from .statistics import BenchmarkStats


@dataclass
class CorrectnessResult:
    """Correctness tracking for a single provider/engine run.

    Attributes:
        execution_success: Did the run complete without error?
        tolerance_match: Within rtol/atol? None if execution failed or
            reference provider unavailable.
        rtol: Relative tolerance used.
        atol: Absolute tolerance used.
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
        """Overall pass = executed successfully AND tolerance matched."""
        return self.execution_success and (self.tolerance_match is True)

    @classmethod
    def failed(
        cls, rtol: float, atol: float, error_message: str
    ) -> "CorrectnessResult":
        """Build a CorrectnessResult representing an execution failure.

        Used at error/skip sites where the GPU run did not complete and no
        comparison was performed.

        Args:
            rtol: Relative tolerance configured for comparison.
            atol: Absolute tolerance configured for comparison.
            error_message: Explanation of the failure.

        Returns:
            CorrectnessResult with execution_success=False and
            tolerance_match=None.
        """
        return cls(
            execution_success=False,
            tolerance_match=None,
            rtol=rtol,
            atol=atol,
            error_message=error_message,
        )

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
        cpu_build_time_ms: CPU graph-build time.
        gpu_kernel_stats: GPU kernel timing statistics.
        e2e_stats: End-to-end wall-clock timing statistics.
        correctness: Correctness comparison result.
        error_message: Error message only (no partial timing on error).
        skip_reason: Reason this combination was skipped.
    """

    _VALID_STATUSES = {"success", "error", "skipped"}

    provider: str
    engine_id: int
    status: Literal["success", "error", "skipped"]
    cpu_build_time_ms: Optional[float] = None
    gpu_kernel_stats: Optional[BenchmarkStats] = None
    e2e_stats: Optional[BenchmarkStats] = None
    elapsed_time_ms: float = 0.0
    correctness: Optional[CorrectnessResult] = None
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate status field."""
        if self.status not in self._VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. "
                f"Must be one of: {self._VALID_STATUSES}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Error entries serialize status + error_message only, no timing.
        Correctness, when present, is always serialized regardless of status
        so that error/skip entries can carry their failure context.
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
            d["elapsed_time_ms"] = self.elapsed_time_ms
        elif self.status == "error":
            d["error_message"] = self.error_message
        elif self.status == "skipped":
            d["skip_reason"] = self.skip_reason

        if self.correctness is not None:
            d["correctness"] = self.correctness.to_dict()
        return d


class StatusCounts(NamedTuple):
    """Counts of provider/engine results bucketed by outcome.

    Attributes:
        passed: Successful runs whose correctness either matched or was not
            checked (tolerance_match is True or None).
        failed: Successful runs whose correctness comparison failed
            (tolerance_match is False).
        skipped: Runs marked as 'skipped' (unsupported combinations).
        errored: Runs marked as 'error' (hard failure).
    """

    passed: int
    failed: int
    skipped: int
    errored: int


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
    engine_ids: List[int] = field(default_factory=list)

    def is_no_engine_graph(self) -> bool:
        """True when this graph result represents a no-engine outcome."""
        return len(self.engine_ids) == 0

    def count_by_status(self) -> StatusCounts:
        """Bucket results into pass/fail/skip/error counts.

        A 'pass' is a successful run whose correctness check either passed or
        was not performed (tolerance_match is True or None). A 'fail' is a
        successful run whose correctness check explicitly failed
        (tolerance_match is False).

        Returns:
            StatusCounts with the four bucket counts.
        """
        passed = sum(
            1
            for r in self.results
            if r.status == "success"
            and (r.correctness is None or r.correctness.tolerance_match is not False)
        )
        failed = sum(
            1
            for r in self.results
            if r.status == "success"
            and r.correctness is not None
            and r.correctness.tolerance_match is False
        )
        skipped = sum(1 for r in self.results if r.status == "skipped")
        errored = sum(1 for r in self.results if r.status == "error")
        return StatusCounts(
            passed=passed, failed=failed, skipped=skipped, errored=errored
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Graph entry with 'results' array of provider/engine entries.
        """
        return {
            "graph_name": self.graph_name,
            "graph_path": self.graph_path,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class SuiteMetadata:
    """Suite-level summary plus environment info.

    Attributes:
        timestamp: UTC timestamp when suite was run.
        hostname: Machine hostname.
        total_graphs: Total number of graphs in suite.
        total_combinations: Total provider/engine combinations across all graphs.
        pass_combinations: Combinations that passed correctness.
        fail_combinations: Combinations that failed correctness.
        skip_combinations: Combinations skipped (unsupported).
        error_combinations: Combinations that errored during execution.
        rocm_version: ROCm version string.
        gpu_model: GPU model name.
        python_version: Python version string.
        hipdnn_version: hipDNN version string.
    """

    timestamp: str
    hostname: str
    total_graphs: int
    total_combinations: int
    pass_combinations: int
    fail_combinations: int
    skip_combinations: int
    error_combinations: int
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
            "total_combinations": self.total_combinations,
            "pass_combinations": self.pass_combinations,
            "fail_combinations": self.fail_combinations,
            "skip_combinations": self.skip_combinations,
            "error_combinations": self.error_combinations,
            "rocm_version": self.rocm_version,
            "gpu_model": self.gpu_model,
            "python_version": self.python_version,
            "hipdnn_version": self.hipdnn_version,
        }


@dataclass
class SuiteResult:
    """Top-level suite result with graph-first nesting.

    Attributes:
        metadata: Suite-level metadata.
        graphs: List of per-graph results.
    """

    metadata: SuiteMetadata
    graphs: List[GraphResult]

    @classmethod
    def from_graph_results(
        cls, graph_results: List[GraphResult], total_graphs: int
    ) -> "SuiteResult":
        """Build a SuiteResult from per-graph results with auto-computed metadata."""
        env_info = collect_environment_info()
        total_pass = total_fail = total_skip = total_error = 0
        for gr in graph_results:
            c = gr.count_by_status()
            total_pass += c.passed
            total_fail += c.failed
            total_skip += c.skipped
            total_error += c.errored
        metadata = SuiteMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            total_graphs=total_graphs,
            total_combinations=total_pass + total_fail + total_skip + total_error,
            pass_combinations=total_pass,
            fail_combinations=total_fail,
            skip_combinations=total_skip,
            error_combinations=total_error,
            rocm_version=env_info.get("rocm_version"),
            gpu_model=env_info.get("gpu_model"),
            python_version=env_info.get("python_version"),
            hipdnn_version=env_info.get("hipdnn_version"),
        )
        return cls(metadata=metadata, graphs=graph_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns dict with "metadata" and "graphs" keys.
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())


def collect_environment_info() -> Dict[str, Optional[str]]:
    """Collect ROCm version, GPU model, Python version, hipDNN version.

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
