# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for GraphValidator."""

from typing import Any, Dict

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import GraphValidator


class TestGraphValidator:
    """Tests for GraphValidator."""

    def test_validates_conv_fwd(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test that Conv Fwd graph validates successfully."""
        validator = GraphValidator()

        # Should not raise
        validator.validate_conv_fwd_only(sample_conv_fwd_json)

    def test_accepts_matmul(self, sample_matmul_json: Dict[str, Any]) -> None:
        """Test that Matmul graph is accepted."""
        validator = GraphValidator()

        # Should not raise - we accept all operations now
        validator.validate_conv_fwd_only(sample_matmul_json)

    def test_rejects_empty_nodes(self) -> None:
        """Test that graph with no nodes is rejected."""
        validator = GraphValidator()
        graph_json = {"nodes": []}

        with pytest.raises(GraphLoadError, match="no operation nodes"):
            validator.validate_conv_fwd_only(graph_json)

    def test_rejects_missing_nodes(self) -> None:
        """Test that graph with missing nodes key is rejected."""
        validator = GraphValidator()
        graph_json = {}

        with pytest.raises(GraphLoadError, match="no operation nodes"):
            validator.validate_conv_fwd_only(graph_json)

    def test_accepts_mixed_operations(self) -> None:
        """Test that graph with mixed operations is accepted."""
        validator = GraphValidator()
        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes", "name": "conv"},
                {"type": "PointwiseAttributes", "name": "relu"},
            ]
        }

        # Should not raise - we accept all operations now
        validator.validate_conv_fwd_only(graph_json)

    def test_get_supported_types(self) -> None:
        """Test get_supported_types returns copy of supported types."""
        validator = GraphValidator()
        types = validator.get_supported_types()

        # Check all supported operation types
        assert "ConvolutionFwdAttributes" in types
        assert "MatmulAttributes" in types
        assert "PointwiseAttributes" in types
        assert "BatchnormInferenceAttributes" in types

        # Modifying returned set should not affect validator
        types.add("NewType")
        assert "NewType" not in validator.get_supported_types()

    def test_custom_supported_types(self) -> None:
        """Test validator with custom supported types."""
        custom_types = {"MatmulAttributes", "ConvolutionFwdAttributes"}
        validator = GraphValidator(supported_types=custom_types)

        # Verify the custom types are stored
        types = validator.get_supported_types()
        assert types == custom_types

        # Both operations should be valid with validate_conv_fwd_only
        # (since it accepts any operation now)
        matmul_json = {"nodes": [{"type": "MatmulAttributes", "name": "matmul"}]}
        validator.validate_conv_fwd_only(matmul_json)  # Should not raise

        conv_json = {"nodes": [{"type": "ConvolutionFwdAttributes", "name": "conv"}]}
        validator.validate_conv_fwd_only(conv_json)  # Should not raise
