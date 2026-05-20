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

    def test_pmc_all_without_multipass_flag_raises(self):
        with pytest.raises(
            ValueError, match="--pmc all requires --pmc-allow-multipass"
        ):
            MetricsConfig(pmc_set="all")

    def test_pmc_all_with_multipass_flag_accepted(self):
        cfg = MetricsConfig(pmc_set="all", pmc_allow_multipass=True)
        assert cfg.pmc_set == "all"
        assert cfg.opt_in_pass_requested is True

    def test_pmc_basic_does_not_require_multipass(self):
        cfg = MetricsConfig(pmc_set="basic")
        assert cfg.pmc_set == "basic"


class TestEmptyStringNormalization:
    """The CLI uses argparse `choices=` so falsy strings never reach
    these fields from the command line. Programmatic and TOML callers
    don't have that gate — ``pmc_set=""`` or ``emit_trace=""`` would
    pass the `is not None` checks downstream, ride into the
    orchestrator as ``--pmc ""``, and crash rocprofv3 with a confusing
    error. Normalise to None at construction time so falsy strings
    are equivalent to "field unset"."""

    def test_empty_pmc_set_normalised_to_none(self):
        cfg = MetricsConfig(pmc_set="")
        assert cfg.pmc_set is None
        assert cfg.opt_in_pass_requested is False

    def test_whitespace_only_pmc_set_normalised_to_none(self):
        cfg = MetricsConfig(pmc_set="   ")
        assert cfg.pmc_set is None

    def test_empty_emit_trace_normalised_to_none(self):
        cfg = MetricsConfig(emit_trace="")
        assert cfg.emit_trace is None

    def test_whitespace_only_emit_trace_normalised_to_none(self):
        cfg = MetricsConfig(emit_trace="\t\n ")
        assert cfg.emit_trace is None
