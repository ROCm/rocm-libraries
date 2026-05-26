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

    def test_db_path_renders_with_analyze_hint(self):
        """The rocpd db is the source of truth — both the aggregates we
        derived AND the additional speed-of-light blocks rocprof-compute
        can render. Surface the db path + the analyze command so the
        user has a copy-paste path into the full dashboard."""
        extra = {
            "pmc": {
                "set": "basic",
                "arch": "gfx942",
                "counters": {"GRBM_GUI_ACTIVE": {"sum": 1, "mean_per_kernel": 1.0}},
                "db_path": "/tmp/prof/sample/MIOPEN_ENGINE/pmc_basic/results.db",
            }
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        rendered = out.getvalue()
        assert "PMC db:" in rendered
        assert "/tmp/prof/sample/MIOPEN_ENGINE/pmc_basic/results.db" in rendered
        # analyze takes the db's parent dir, not the db itself.
        assert (
            "rocprof-compute analyze --path " "/tmp/prof/sample/MIOPEN_ENGINE/pmc_basic"
        ) in rendered


class TestTraceRendering:
    def test_pftrace_path_renders_with_perfetto_hint(self):
        extra = {"trace": {"format": "pftrace", "path": "/tmp/out/results.pftrace"}}
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        text = out.getvalue()
        assert "Trace (pftrace)" in text
        assert "/tmp/out/results.pftrace" in text
        assert "ui.perfetto.dev" in text

    def test_kineto_db_only_path_renders_convert_hint(self):
        """When rocpd convert failed (no `path`, only `db_path`), the
        user has the source db but no viewable trace. Surface the
        manual convert command so they don't have to dig it out of
        the source."""
        extra = {
            "trace": {
                "format": "kineto",
                "db_path": "/tmp/out/results.db",
                "kineto_unavailable": "rocpd convert failed",
            }
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        text = out.getvalue()
        assert "Trace DB:" in text
        assert "/tmp/out/results.db" in text
        assert "python -m rocpd convert" in text
        # The Perfetto hint must NOT fire here — the db is not directly
        # openable in Perfetto, and surfacing the pftrace hint next to
        # a sqlite path would mislead.
        assert "ui.perfetto.dev" not in text


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
    def test_csv_and_analyze_hint_render(self):
        """``profile --roof-only`` produces CSVs; we surface roofline.csv
        and analyze-command hints (ASCII + GUI) the user can copy-paste
        to render the roofline in any datatype."""
        extra = {
            "roofline": {
                "roofline_csv": "/tmp/r/workload/gfx90a/roofline.csv",
                "sysinfo_csv": "/tmp/r/workload/gfx90a/sysinfo.csv",
                "workload_path": "/tmp/r/workload/gfx90a",
            }
        }
        out = io.StringIO()
        Reporter(output=out)._print_profiling_block(_make_pe(extra))
        rendered = out.getvalue()
        assert "Roofline CSV:" in rendered
        assert "/tmp/r/workload/gfx90a/roofline.csv" in rendered
        # ASCII hint includes --block 4 (avoids the warning-flooded
        # full speed-of-light output) and --gui hint includes the doc
        # pointer for the analyze venv setup gotcha.
        assert "rocprof-compute analyze --path /tmp/r/workload/gfx90a" in rendered
        assert "--block 4" in rendered
        assert "--gui" in rendered
