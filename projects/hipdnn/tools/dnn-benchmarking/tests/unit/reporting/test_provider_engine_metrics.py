# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the always-on metric fields on ProviderEngineResult.to_dict.

The legacy success-path JSON shape must stay backward-compatible: new
metric fields appear only when populated, never as null sentinels.
"""

from dnn_benchmarking.reporting.statistics import BenchmarkStats
from dnn_benchmarking.reporting.suite_results import (
    ProviderEngineResult,
    SuiteMetadata,
)


def _bench_stats(mean: float = 1.0) -> BenchmarkStats:
    return BenchmarkStats(
        mean_ms=mean,
        std_ms=0.1,
        min_ms=mean - 0.1,
        max_ms=mean + 0.1,
        p95_ms=mean + 0.05,
        p99_ms=mean + 0.09,
    )


class TestProviderEngineResultLegacyShape:
    def test_no_new_fields_when_metrics_unset(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            cpu_build_time_ms=12.3,
            gpu_kernel_stats=_bench_stats(),
            e2e_stats=_bench_stats(),
            elapsed_time_ms=200.0,
        )
        d = pe.to_dict()
        # Legacy keys still present
        assert d["status"] == "success"
        assert d["cpu_build_time_ms"] == 12.3
        # No metric fields leaked into JSON when None
        for key in (
            "workspace_bytes",
            "analytical_flops",
            "analytical_io_bytes",
            "derived_tflops_per_s",
            "derived_gbytes_per_s",
            "cpu_user_time_per_iter_us",
            "cpu_kernel_time_per_iter_us",
            "vram_used_mb",
            "extra_metrics",
        ):
            assert key not in d
        # Fields removed from per-engine in the suite-scope cleanup must
        # never appear, even by accident (no setattr leak).
        for removed in (
            "host_rss_mb",
            "host_ram_available_mb",
            "gpu_smi_snapshot",
            "cpu_user_time_ms",
            "cpu_kernel_time_ms",
        ):
            assert removed not in d

    def test_partial_flag_emitted_only_when_true(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            analytical_flops=42,
            analytical_flops_partial=False,
        )
        d = pe.to_dict()
        assert d["analytical_flops"] == 42
        assert "analytical_flops_partial" not in d

    def test_partial_flag_serialised_when_true(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            analytical_flops=42,
            analytical_flops_partial=True,
        )
        d = pe.to_dict()
        assert d["analytical_flops_partial"] is True


class TestProviderEngineResultFullShape:
    def test_all_metric_fields_serialise(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            cpu_build_time_ms=10.0,
            gpu_kernel_stats=_bench_stats(0.5),
            e2e_stats=_bench_stats(1.0),
            elapsed_time_ms=200.0,
            workspace_bytes=4096,
            analytical_flops=10**9,
            analytical_io_bytes=10**6,
            derived_tflops_per_s=2.0,
            derived_gbytes_per_s=2.0,
            cpu_user_time_per_iter_us=40.0,
            cpu_kernel_time_per_iter_us=2.5,
            vram_used_mb=4096.0,
        )
        d = pe.to_dict()
        assert d["workspace_bytes"] == 4096
        assert d["analytical_flops"] == 10**9
        assert d["analytical_io_bytes"] == 10**6
        assert d["derived_tflops_per_s"] == 2.0
        assert d["derived_gbytes_per_s"] == 2.0
        assert d["cpu_user_time_per_iter_us"] == 40.0
        assert d["cpu_kernel_time_per_iter_us"] == 2.5
        assert d["vram_used_mb"] == 4096.0

    def test_extra_metrics_passthrough(self):
        # Always-on collection never populates this; the schema must
        # still round-trip an arbitrary dict so opt-in profiling
        # payloads land cleanly.
        payload = {"trace": {"format": "pftrace", "path": "/tmp/x.pftrace"}}
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="success",
            extra_metrics=payload,
        )
        assert pe.to_dict()["extra_metrics"] == payload


class TestErrorAndSkipPathsUnaffected:
    def test_error_status_emits_only_error_message(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="error",
            error_message="boom",
            # Setting metric fields on an error path must NOT leak to JSON.
            workspace_bytes=999,
            analytical_flops=999,
        )
        d = pe.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] == "boom"
        assert "workspace_bytes" not in d
        assert "analytical_flops" not in d

    def test_skipped_status_emits_only_skip_reason(self):
        pe = ProviderEngineResult(
            provider="miopen",
            engine_id=1,
            status="skipped",
            skip_reason="unsupported",
            workspace_bytes=999,
        )
        d = pe.to_dict()
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "unsupported"
        assert "workspace_bytes" not in d


class TestSuiteMetadataMachineFields:
    def test_machine_fields_in_to_dict(self):
        meta = SuiteMetadata(
            timestamp="2026-05-11T00:00:00Z",
            hostname="test-host",
            total_graphs=1,
            total_combinations=1,
            pass_combinations=1,
            fail_combinations=0,
            skip_combinations=0,
            error_combinations=0,
            cpu_model="EPYC 9654",
            cpu_count=192,
            numa_nodes=2,
            total_ram_gb=1536.0,
            kernel_version="6.8.0-31-generic",
            gpu_compute_units=304,
            gpu_hbm_gb=192.0,
            gpu_pcie_link="gen4 x16",
            amdgpu_driver_version="6.14.5",
        )
        d = meta.to_dict()
        assert d["cpu_model"] == "EPYC 9654"
        assert d["cpu_count"] == 192
        assert d["numa_nodes"] == 2
        assert d["total_ram_gb"] == 1536.0
        assert d["kernel_version"] == "6.8.0-31-generic"
        assert d["gpu_compute_units"] == 304
        assert d["gpu_hbm_gb"] == 192.0
        assert d["gpu_pcie_link"] == "gen4 x16"
        assert d["amdgpu_driver_version"] == "6.14.5"

    def test_footprint_fields_in_to_dict(self):
        # Process RSS / VRAM live on SuiteMetadata (not per-engine)
        # because they're flat across the suite.
        meta = SuiteMetadata(
            timestamp="2026-05-11T00:00:00Z",
            hostname="test-host",
            total_graphs=1,
            total_combinations=1,
            pass_combinations=1,
            fail_combinations=0,
            skip_combinations=0,
            error_combinations=0,
            host_rss_mb=843.3,
            host_ram_available_mb=2177515.0,
            vram_used_mb=4096.0,
            vram_total_mb=196608.0,
        )
        d = meta.to_dict()
        assert d["host_rss_mb"] == 843.3
        assert d["host_ram_available_mb"] == 2177515.0
        assert d["vram_used_mb"] == 4096.0
        assert d["vram_total_mb"] == 196608.0
