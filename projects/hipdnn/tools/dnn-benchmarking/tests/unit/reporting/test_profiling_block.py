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


class TestPmcRendering:
    def test_counters_present_renders_first_three(self):
        extra = {
            "pmc": {
                "set": "basic",
                "arch": "gfx942",
                "counters": {
                    "GRBM_GUI_ACTIVE": {"sum": 12345678, "mean_per_kernel": 1.0},
                    "SQ_WAVES": {"sum": 987654, "mean_per_kernel": 2.0},
                    "SQ_INSTS_VALU": {"sum": 4321, "mean_per_kernel": 3.0},
                    "SQ_BUSY_CYCLES": {"sum": 11, "mean_per_kernel": 4.0},
                },
            }
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        text = out.getvalue()
        assert "Profiling:" in text
        assert "PMC (basic, gfx942)" in text
        assert "GRBM_GUI_ACTIVE=12,345,678" in text
        # Fourth counter is collapsed into a "more" suffix.
        assert "SQ_BUSY_CYCLES" not in text
        assert "[1 more, see JSON]" in text

    def test_pmc_skipped_renders_reason(self):
        extra = {
            "pmc": {"set": "basic", "arch": "gfx942", "skipped": "no counters defined"}
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        assert "PMC (basic, gfx942):  skipped — no counters defined" in out.getvalue()


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


class TestRooflineRendering:
    def test_pdf_path_renders(self):
        extra = {"roofline": {"data_type": "FP16", "pdf_path": "/tmp/roofline.pdf"}}
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        assert "Roofline (FP16)" in out.getvalue()
        assert "/tmp/roofline.pdf" in out.getvalue()
