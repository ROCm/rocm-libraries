# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for BenchmarkConfig."""

from pathlib import Path

import pytest

from dnn_benchmarking.config import ABTestConfig, BenchmarkConfig, ValidationConfig


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


class TestABTestConfig:
    """Tests for ABTestConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are applied correctly."""
        config = ABTestConfig()

        assert config.a_path is None
        assert config.a_id == 1
        assert config.b_path is None
        assert config.b_id == 1
        assert config.rtol == 1e-5
        assert config.atol == 1e-8

    def test_custom_values(self) -> None:
        """Test that custom values are stored correctly."""
        config = ABTestConfig(
            a_path=Path("/path/to/pluginA"),
            a_id=1,
            b_path=Path("/path/to/pluginB"),
            b_id=2,
            rtol=1e-3,
            atol=1e-6,
        )

        assert config.a_path == Path("/path/to/pluginA")
        assert config.a_id == 1
        assert config.b_path == Path("/path/to/pluginB")
        assert config.b_id == 2
        assert config.rtol == 1e-3
        assert config.atol == 1e-6

    def test_string_path_converted_to_path(self) -> None:
        """Test that string paths are converted to Path objects."""
        config = ABTestConfig(
            a_path="/path/to/pluginA",  # type: ignore
            b_path="/path/to/pluginB",  # type: ignore
        )

        assert isinstance(config.a_path, Path)
        assert isinstance(config.b_path, Path)
        assert config.a_path == Path("/path/to/pluginA")
        assert config.b_path == Path("/path/to/pluginB")

    def test_negative_ids_accepted(self) -> None:
        """a_id / b_id may be negative (FNV-1a engine ID hashes)."""
        config = ABTestConfig(a_id=-1, b_id=-2)
        assert config.a_id == -1
        assert config.b_id == -2

    def test_negative_rtol_raises(self) -> None:
        """Test that negative rtol raises ValueError."""
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            ABTestConfig(rtol=-1e-5)

    def test_negative_atol_raises(self) -> None:
        """Test that negative atol raises ValueError."""
        with pytest.raises(ValueError, match="atol must be non-negative"):
            ABTestConfig(atol=-1e-8)

    def test_validate_paths_with_existing_paths(self, tmp_path: Path) -> None:
        """Test validate_paths succeeds with existing paths."""
        plugin_a = tmp_path / "pluginA"
        plugin_b = tmp_path / "pluginB"
        plugin_a.mkdir()
        plugin_b.mkdir()

        config = ABTestConfig(a_path=plugin_a, b_path=plugin_b)
        # Should not raise
        config.validate_paths()

    def test_validate_paths_with_none_paths(self) -> None:
        """Test validate_paths succeeds with None paths."""
        config = ABTestConfig()
        # Should not raise
        config.validate_paths()

    def test_validate_paths_nonexistent_a_path(self, tmp_path: Path) -> None:
        """Test validate_paths raises for nonexistent a_path."""
        config = ABTestConfig(a_path=tmp_path / "nonexistent")

        with pytest.raises(ValueError, match="Plugin path A does not exist"):
            config.validate_paths()

    def test_validate_paths_nonexistent_b_path(self, tmp_path: Path) -> None:
        """Test validate_paths raises for nonexistent b_path."""
        config = ABTestConfig(b_path=tmp_path / "nonexistent")

        with pytest.raises(ValueError, match="Plugin path B does not exist"):
            config.validate_paths()


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
