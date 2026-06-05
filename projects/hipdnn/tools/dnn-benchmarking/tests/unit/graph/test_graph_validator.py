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
        validator.validate(sample_conv_fwd_json)

    def test_accepts_matmul(self, sample_matmul_json: Dict[str, Any]) -> None:
        """Test that Matmul graph is accepted."""
        validator = GraphValidator()

        # Should not raise - validation is deferred to hipDNN
        validator.validate(sample_matmul_json)

    def test_rejects_empty_nodes(self) -> None:
        """Test that graph with no nodes is rejected."""
        validator = GraphValidator()
        graph_json = {"nodes": []}

        with pytest.raises(GraphLoadError, match="no operation nodes"):
            validator.validate(graph_json)

    def test_rejects_missing_nodes(self) -> None:
        """Test that graph with missing nodes key is rejected."""
        validator = GraphValidator()
        graph_json = {}

        with pytest.raises(GraphLoadError, match="no operation nodes"):
            validator.validate(graph_json)

    def test_accepts_mixed_operations(self) -> None:
        """Test that graph with mixed operations is accepted."""
        validator = GraphValidator()
        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes", "name": "conv"},
                {"type": "PointwiseAttributes", "name": "relu"},
            ]
        }

        # Should not raise - validation is deferred to hipDNN
        validator.validate(graph_json)
