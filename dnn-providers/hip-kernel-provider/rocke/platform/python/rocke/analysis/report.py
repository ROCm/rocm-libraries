# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Variant reports combining compile, static analysis, and benchmark data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..benchmark import BenchmarkSummary
from ..helpers import KernelArtifact
from .ir import LlvmIrStats, analyze_llvm_ir
from .isa import IsaStats, ResourceInfo, analyze_hsaco


@dataclass(frozen=True)
class VariantReport:
    """One fully measured kernel variant."""

    name: str
    spec: Mapping[str, Any]
    artifact: KernelArtifact
    benchmark: Optional[BenchmarkSummary] = None
    llvm: LlvmIrStats = field(default_factory=LlvmIrStats)
    isa: Optional[IsaStats] = None
    resources: Optional[ResourceInfo] = None

    @classmethod
    def from_artifact(
        cls,
        *,
        name: str,
        spec: Mapping[str, Any],
        artifact: KernelArtifact,
        benchmark: Optional[BenchmarkSummary] = None,
        hsaco_path: Optional[Path] = None,
        objdump: str = "llvm-objdump",
        readelf: str = "llvm-readelf",
    ) -> "VariantReport":
        hsaco = None
        if hsaco_path is not None:
            hsaco = analyze_hsaco(hsaco_path, objdump=objdump, readelf=readelf)
        return cls(
            name=name,
            spec=dict(spec),
            artifact=artifact,
            benchmark=benchmark,
            llvm=analyze_llvm_ir(artifact.llvm_text),
            isa=hsaco.isa if hsaco is not None else None,
            resources=hsaco.resources if hsaco is not None else None,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spec": dict(self.spec),
            "kernel_name": self.artifact.kernel_name,
            "hsaco_bytes": self.artifact.hsaco_bytes,
            "timings": dict(self.artifact.timings),
            "pass_stats": self.artifact.pass_stats.__dict__,
            "benchmark": self.benchmark.as_dict() if self.benchmark else None,
            "llvm": self.llvm.as_dict(),
            "isa": self.isa.as_dict() if self.isa else None,
            "resources": self.resources.as_dict() if self.resources else None,
        }


def compare_variant_reports(reports: Sequence[VariantReport]) -> List[Dict[str, Any]]:
    """Return compact rows sorted by median TFLOPS if benchmarked."""

    rows: List[Dict[str, Any]] = []
    for r in reports:
        bench = r.benchmark
        rows.append(
            {
                "name": r.name,
                "median_tflops": bench.median_tflops if bench else None,
                "spread_pct": bench.spread_pct if bench else None,
                "bad_count": bench.bad_count if bench else None,
                "llvm_mfma": r.llvm.mfma_calls,
                "llvm_async": r.llvm.async_buffer_load_lds_calls,
                "llvm_raw_load": r.llvm.raw_buffer_load_calls,
                "isa_mfma": r.isa.mfma if r.isa else None,
                "isa_buffer_load": r.isa.buffer_load if r.isa else None,
                "isa_buffer_load_lds": r.isa.buffer_load_lds if r.isa else None,
                "vgpr": r.resources.vgpr_count if r.resources else None,
                "sgpr": r.resources.sgpr_count if r.resources else None,
                "lds_bytes": r.resources.lds_bytes if r.resources else None,
            }
        )
    return sorted(
        rows,
        key=lambda x: (x["median_tflops"] is not None, x["median_tflops"] or 0.0),
        reverse=True,
    )
