# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for BenchmarkConfig."""

from pathlib import Path

import pytest

from dnn_benchmarking.config import BenchmarkConfig, SuiteConfig, ValidationConfig


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are applied correctly."""
        config = BenchmarkConfig(graph_path=Path("/test/graph.json"))

        assert config.warmup_iters == 10
        assert config.benchmark_iters == 100
        assert config.engine_id == 1

    def test_custom_values(self) -> None:
        """Test that custom values are stored correctly."""
        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=20,
            benchmark_iters=200,
            engine_id=2,
        )

        assert config.graph_path == Path("/test/graph.json")
        assert config.warmup_iters == 20
        assert config.benchmark_iters == 200
        assert config.engine_id == 2

    def test_string_path_converted_to_path(self) -> None:
        """Test that string path is converted to Path object."""
        config = BenchmarkConfig(graph_path="/test/graph.json")  # type: ignore

        assert isinstance(config.graph_path, Path)
        assert config.graph_path == Path("/test/graph.json")

    def test_negative_warmup_raises(self) -> None:
        """Test that negative warmup_iters raises ValueError."""
        with pytest.raises(ValueError, match="warmup_iters must be non-negative"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), warmup_iters=-1)

    def test_zero_warmup_allowed(self) -> None:
        """Test that zero warmup_iters is allowed."""
        config = BenchmarkConfig(graph_path=Path("/test/graph.json"), warmup_iters=0)
        assert config.warmup_iters == 0

    def test_zero_benchmark_iters_raises(self) -> None:
        """Test that zero benchmark_iters raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_iters must be positive"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), benchmark_iters=0)

    def test_negative_benchmark_iters_raises(self) -> None:
        """Test that negative benchmark_iters raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_iters must be positive"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), benchmark_iters=-1)

    def test_negative_engine_id_accepted(self) -> None:
        """Engine IDs are FNV-1a hashes; negative values (high bit set in
        signed int64) must be accepted."""
        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            engine_id=-4567890123456789012,
        )
        assert config.engine_id == -4567890123456789012


class TestSuiteConfigPluginPaths:
    """Tests for SuiteConfig engine/plugin path selection."""

    def test_single_plugin_path_applies_to_all_engines(self) -> None:
        config = SuiteConfig(
            engine_filter=[1, 2],
            plugin_paths=[Path("/plugins/a")],
        )
        selections = config.engine_selections_for([1, 2])

        assert [s.engine_id for s in selections] == [1, 2]
        assert [s.plugin_path for s in selections] == [
            Path("/plugins/a"),
            Path("/plugins/a"),
        ]
        assert config.plugin_path == Path("/plugins/a")

    def test_multiple_plugin_paths_follow_engine_order(self) -> None:
        config = SuiteConfig(
            engine_filter=[2, 1],
            plugin_paths=[Path("/plugins/b"), Path("/plugins/a")],
        )
        selections = config.engine_selections_for([2, 1])

        assert [s.engine_id for s in selections] == [2, 1]
        assert [s.plugin_path for s in selections] == [
            Path("/plugins/b"),
            Path("/plugins/a"),
        ]
        assert config.plugin_path is None

    def test_repeated_engine_ids_keep_distinct_plugin_paths(self) -> None:
        config = SuiteConfig(
            engine_filter=[1, 1],
            plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
        )

        selections = config.engine_selections_for([1, 1])

        assert [s.engine_id for s in selections] == [1, 1]
        assert [s.plugin_path for s in selections] == [
            Path("/plugins/a"),
            Path("/plugins/b"),
        ]

    def test_multiple_plugin_paths_require_engine_filter(self) -> None:
        with pytest.raises(ValueError, match="requires --engine"):
            SuiteConfig(plugin_paths=[Path("/plugins/a"), Path("/plugins/b")])

    def test_plugin_path_count_must_match_engine_count(self) -> None:
        with pytest.raises(ValueError, match="entry count"):
            SuiteConfig(
                engine_filter=[1, 2, 3],
                plugin_paths=[Path("/plugins/a"), Path("/plugins/b")],
            )


class TestSuiteConfigTolerances:
    """Tests for SuiteConfig validation tolerance overrides."""

    def test_defaults_use_dtype_aware_tolerances(self) -> None:
        config = SuiteConfig()
        assert config.rtol is None
        assert config.atol is None
        assert config.tolerance_override is None

    def test_both_tolerances_override_dtype_defaults(self) -> None:
        config = SuiteConfig(rtol=1e-3, atol=1e-4)
        assert config.tolerance_override == (1e-3, 1e-4)

    def test_single_tolerance_applies_to_both_values(self) -> None:
        assert SuiteConfig(rtol=1e-3).tolerance_override == (1e-3, 1e-3)
        assert SuiteConfig(atol=1e-4).tolerance_override == (1e-4, 1e-4)


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are applied correctly."""
        config = ValidationConfig()

        assert config.provider == "none"
        assert config.rtol == 1e-5
        assert config.atol == 1e-8
        assert config.enabled is False

    def test_custom_values(self) -> None:
        """Test that custom values are stored correctly."""
        config = ValidationConfig(
            provider="pytorch",
            rtol=1e-3,
            atol=1e-6,
        )

        assert config.provider == "pytorch"
        assert config.rtol == 1e-3
        assert config.atol == 1e-6

    def test_enabled_property_none(self) -> None:
        """Test enabled property with provider='none'."""
        config = ValidationConfig(provider="none")

        assert config.enabled is False

    def test_enabled_property_pytorch(self) -> None:
        """Test enabled property with provider='pytorch'."""
        config = ValidationConfig(provider="pytorch")

        assert config.enabled is True

    def test_enabled_property_cpu_plugin(self) -> None:
        """Test enabled property with provider='cpu_plugin'."""
        config = ValidationConfig(provider="cpu_plugin")

        assert config.enabled is True

    def test_invalid_provider_raises(self) -> None:
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Invalid provider"):
            ValidationConfig(provider="invalid_provider")

    def test_negative_rtol_raises(self) -> None:
        """Test that negative rtol raises ValueError."""
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            ValidationConfig(rtol=-1e-5)

    def test_negative_atol_raises(self) -> None:
        """Test that negative atol raises ValueError."""
        with pytest.raises(ValueError, match="atol must be non-negative"):
            ValidationConfig(atol=-1e-8)
