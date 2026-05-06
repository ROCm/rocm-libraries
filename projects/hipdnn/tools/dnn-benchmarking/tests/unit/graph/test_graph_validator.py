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

        # Should not raise
        validator.validate(sample_matmul_json)

    def test_rejects_empty_nodes(self) -> None:
        """Test that graph with no nodes is rejected."""
        validator = GraphValidator()
        graph_json = {"tensors": [], "nodes": []}

        with pytest.raises(GraphLoadError, match="no operation nodes"):
            validator.validate(graph_json)

    def test_rejects_missing_nodes_key(self) -> None:
        """Test that graph missing the nodes key is rejected."""
        validator = GraphValidator()
        graph_json = {"tensors": []}

        with pytest.raises(GraphLoadError, match="required top-level keys"):
            validator.validate(graph_json)

    def test_rejects_missing_tensors_key(self) -> None:
        """Test that graph missing the tensors key is rejected."""
        validator = GraphValidator()
        graph_json = {"nodes": [{"type": "ConvolutionFwdAttributes"}]}

        with pytest.raises(GraphLoadError, match="required top-level keys"):
            validator.validate(graph_json)

    def test_rejects_missing_both_top_level_keys(self) -> None:
        """Test that graph missing both tensors and nodes is rejected."""
        validator = GraphValidator()
        graph_json = {}

        with pytest.raises(GraphLoadError, match="required top-level keys"):
            validator.validate(graph_json)

    def test_accepts_mixed_operations(
        self, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test that graph with mixed operations is accepted."""
        validator = GraphValidator()
        # PointwiseAttributes has no entry in _REQUIRED_NODE_FIELDS, so field
        # validation is skipped for it and deferred to the hipDNN backend.
        graph_json = {
            "tensors": sample_conv_fwd_json["tensors"],
            "nodes": sample_conv_fwd_json["nodes"]
            + [{"type": "PointwiseAttributes", "name": "relu"}],
        }

        # Should not raise
        validator.validate(graph_json)

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

    def test_custom_supported_types(
        self, sample_conv_fwd_json: Dict[str, Any], sample_matmul_json: Dict[str, Any]
    ) -> None:
        """Test validator with custom supported types."""
        custom_types = {"MatmulAttributes", "ConvolutionFwdAttributes"}
        validator = GraphValidator(supported_types=custom_types)

        # Verify the custom types are stored
        types = validator.get_supported_types()
        assert types == custom_types

        validator.validate(sample_matmul_json)  # Should not raise
        validator.validate(sample_conv_fwd_json)  # Should not raise

    def test_rejects_missing_required_fields(self) -> None:
        """Test that nodes missing required fields are rejected."""
        validator = GraphValidator()

        # BatchnormBackwardAttributes without dx_tensor_uid output
        graph_json = {
            "tensors": [],
            "nodes": [
                {
                    "type": "BatchnormBackwardAttributes",
                    "name": "bnorm_bwd",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "dy_tensor_uid": 2,
                        "mean_tensor_uid": 3,
                        "inv_variance_tensor_uid": 4,
                        "scale_tensor_uid": 5,
                    },
                    "outputs": {
                        # dx_tensor_uid intentionally omitted
                        "dscale_tensor_uid": 7,
                        "dbias_tensor_uid": 8,
                    },
                }
            ],
        }

        with pytest.raises(GraphLoadError, match="dx_tensor_uid"):
            validator.validate(graph_json)

    def test_rejects_missing_field_conv_fwd(self) -> None:
        """Test that ConvolutionFwdAttributes missing w_tensor_uid is rejected."""
        validator = GraphValidator()
        graph_json = {
            "tensors": [],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "name": "conv",
                    "inputs": {
                        "x_tensor_uid": 1,
                        # w_tensor_uid intentionally omitted
                    },
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "conv_mode": "CROSS_CORRELATION",
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        with pytest.raises(GraphLoadError, match="w_tensor_uid"):
            validator.validate(graph_json)

    def test_rejects_missing_field_matmul(self) -> None:
        """Test that MatmulAttributes missing b_tensor_uid is rejected."""
        validator = GraphValidator()
        graph_json = {
            "tensors": [],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "name": "matmul",
                    "inputs": {
                        "a_tensor_uid": 1,
                        # b_tensor_uid intentionally omitted
                    },
                    "outputs": {"c_tensor_uid": 2},
                }
            ],
        }

        with pytest.raises(GraphLoadError, match="b_tensor_uid"):
            validator.validate(graph_json)

    def test_rejects_node_missing_entire_section(self) -> None:
        """Test that a node with no 'inputs' section is rejected for types with required inputs."""
        validator = GraphValidator()
        graph_json = {
            "tensors": [],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "name": "conv",
                    # no "inputs" key at all
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "conv_mode": "CROSS_CORRELATION",
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        with pytest.raises(GraphLoadError, match="x_tensor_uid"):
            validator.validate(graph_json)
