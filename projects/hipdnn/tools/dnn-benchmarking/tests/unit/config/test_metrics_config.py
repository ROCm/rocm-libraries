# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for MetricsConfig dataclass and its embedding in SuiteConfig."""

from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import MetricsConfig, SuiteConfig


class TestMetricsConfigDefaults:
    def test_default_tier_is_basic(self):
        cfg = MetricsConfig()
        assert cfg.tier == "basic"
        assert cfg.basic_enabled is True
        assert cfg.opt_in_pass_requested is False

    def test_off_disables_basic(self):
        cfg = MetricsConfig(tier="off")
        assert cfg.basic_enabled is False

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Invalid metrics tier"):
            MetricsConfig(tier="ultra")  # type: ignore[arg-type]

    def test_invalid_emit_trace_raises(self):
        with pytest.raises(ValueError, match="Invalid emit_trace"):
            MetricsConfig(emit_trace="csv")  # type: ignore[arg-type]


class TestOptInDetection:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"emit_trace": "pftrace"},
            {"perf": True},
        ],
    )
    def test_any_opt_in_flag_marks_opt_in(self, kwargs):
        cfg = MetricsConfig(**kwargs)
        assert cfg.opt_in_pass_requested is True


class TestSuiteConfigEmbedding:
    def test_metrics_default_factory(self):
        sc = SuiteConfig()
        # Each SuiteConfig must own its own MetricsConfig (not a shared
        # mutable default).
        sc2 = SuiteConfig()
        assert sc.metrics is not sc2.metrics
        assert sc.metrics.tier == "basic"

    def test_metrics_can_be_overridden(self):
        sc = SuiteConfig(metrics=MetricsConfig(tier="off"))
        assert sc.metrics.basic_enabled is False


class TestProfilingFields:
    def test_profiling_output_dir_str_coerced_to_path(self):
        cfg = MetricsConfig(profiling_output_dir="/tmp/out")
        assert isinstance(cfg.profiling_output_dir, Path)
        assert cfg.profiling_output_dir == Path("/tmp/out")
