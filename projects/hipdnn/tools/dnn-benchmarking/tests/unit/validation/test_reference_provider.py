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

        assert registered == ["pytorch"]

    def test_get_pytorch_provider(self) -> None:
        """Test getting pytorch provider from registry."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.name == "pytorch"

    def test_get_unknown_provider_raises(self) -> None:
        """Test that getting unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ReferenceProviderRegistry.get_provider("unknown_provider")

        assert "Unknown reference provider" in str(exc_info.value)

    def test_list_available_providers(self) -> None:
        """Test listing available providers."""
        available = ReferenceProviderRegistry.list_available()

        assert set(available).issubset({"pytorch"})
