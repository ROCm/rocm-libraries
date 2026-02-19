# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ReferenceProvider and ReferenceProviderRegistry."""

import numpy as np
import pytest

from dnn_benchmarking.validation import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)


class TestReferenceOutput:
    """Tests for ReferenceOutput dataclass."""

    def test_create_reference_output(self) -> None:
        """Test creating a ReferenceOutput."""
        data = np.array([1.0, 2.0, 3.0])
        output = ReferenceOutput(data=data, tensor_uid=42)

        assert np.array_equal(output.data, data)
        assert output.tensor_uid == 42
        assert output.metadata is None

    def test_reference_output_with_metadata(self) -> None:
        """Test ReferenceOutput with metadata."""
        data = np.array([1.0, 2.0])
        metadata = {"operation": "relu", "dtype": "float32"}
        output = ReferenceOutput(data=data, tensor_uid=1, metadata=metadata)

        assert output.metadata == metadata
        assert output.metadata["operation"] == "relu"


class TestReferenceProviderRegistry:
    """Tests for ReferenceProviderRegistry."""

    def test_list_registered_providers(self) -> None:
        """Test listing registered providers."""
        registered = ReferenceProviderRegistry.list_registered()

        # Should have at least pytorch and cpu_plugin registered
        assert "pytorch" in registered
        assert "cpu_plugin" in registered

    def test_get_pytorch_provider(self) -> None:
        """Test getting pytorch provider from registry."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.name == "pytorch"

    def test_get_cpu_plugin_provider(self) -> None:
        """Test getting cpu_plugin provider from registry."""
        provider = ReferenceProviderRegistry.get_provider("cpu_plugin")

        assert provider.name == "cpu_plugin"

    def test_get_unknown_provider_raises(self) -> None:
        """Test that getting unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ReferenceProviderRegistry.get_provider("unknown_provider")

        assert "Unknown reference provider" in str(exc_info.value)

    def test_list_available_providers(self) -> None:
        """Test listing available providers."""
        available = ReferenceProviderRegistry.list_available()

        # pytorch should be available if torch is installed
        # cpu_plugin should not be available (stubbed)
        assert "cpu_plugin" not in available


class TestCPUPluginProvider:
    """Tests for CPUPluginReferenceProvider (stubbed)."""

    def test_cpu_plugin_not_available(self) -> None:
        """Test that CPU plugin reports not available."""
        provider = ReferenceProviderRegistry.get_provider("cpu_plugin")

        assert provider.is_available() is False

    def test_cpu_plugin_compute_raises(self) -> None:
        """Test that CPU plugin compute raises NotImplementedError."""
        provider = ReferenceProviderRegistry.get_provider("cpu_plugin")
        graph_json = {"nodes": [], "tensors": []}
        input_data = {}

        with pytest.raises(NotImplementedError) as exc_info:
            provider.compute_reference(graph_json, input_data)

        assert "CPU reference plugin not available" in str(exc_info.value)

    def test_cpu_plugin_name(self) -> None:
        """Test CPU plugin provider name."""
        provider = ReferenceProviderRegistry.get_provider("cpu_plugin")

        assert provider.name == "cpu_plugin"
