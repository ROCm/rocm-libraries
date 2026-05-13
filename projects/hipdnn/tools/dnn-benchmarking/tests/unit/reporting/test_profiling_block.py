# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the Reporter Profiling: block rendering.

Each opt-in profiling source contributes one optional line; the block
itself is suppressed when extra_metrics is empty or only carries
non-source keys.
"""

import io

from dnn_benchmarking.reporting import Reporter
from dnn_benchmarking.reporting.suite_results import ProviderEngineResult


def _make_pe(extra_metrics):
    return ProviderEngineResult(
        provider="miopen",
        engine_id=1,
        status="success",
        extra_metrics=extra_metrics,
    )


class TestNoProfilingDataSuppressesBlock:
    def test_none_extra_metrics_emits_nothing(self):
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(None))
        assert out.getvalue() == ""

    def test_empty_dict_emits_nothing(self):
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe({}))
        assert out.getvalue() == ""


class TestTraceRendering:
    def test_pftrace_path_renders(self):
        extra = {"trace": {"format": "pftrace", "path": "/tmp/out/results.pftrace"}}
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        assert "Trace (pftrace)" in out.getvalue()
        assert "/tmp/out/results.pftrace" in out.getvalue()


class TestPerfRendering:
    def test_ipc_and_cycles_render(self):
        extra = {
            "perf": {
                "ipc_user": 0.795,
                "cycles_user": 1234567890,
                "instructions_user": 987654321,
                "task_clock_ms": 123.4,
            }
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        text = out.getvalue()
        assert "CPU (perf)" in text
        assert "IPC=0.80" in text
        assert "task_clock=123.4ms" in text

    def test_perf_skipped_renders_reason(self):
        extra = {"perf": {"skipped": "perf binary not found on PATH"}}
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        assert "skipped — perf binary not found on PATH" in out.getvalue()
