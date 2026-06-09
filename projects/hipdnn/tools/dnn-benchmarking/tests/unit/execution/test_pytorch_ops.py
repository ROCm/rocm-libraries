# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for PyTorch operation handlers error paths."""

import pytest

torch = pytest.importorskip("torch")

from dnn_benchmarking.execution import pytorch_ops


class TestPyTorchOpsErrorPaths:
    """Tests for error handling in pytorch_ops."""

    def test_execute_graph_unsupported_operation_raises(self) -> None:
        """Test that unsupported operation raises ValueError."""
        graph_json = {"nodes": [{"type": "UnknownOperation"}]}

        with pytest.raises(ValueError, match="Unsupported operation type"):
            pytorch_ops.execute_graph(graph_json, {})

    def test_conv_missing_x_tensor_uid_raises(self) -> None:
        """Test that conv with missing x tensor UID raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"w_tensor_uid": 2},  # missing x_tensor_uid
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {},
                }
            ]
        }

        with pytest.raises(ValueError, match="missing required tensor UIDs"):
            pytorch_ops.execute_graph(graph_json, {2: torch.zeros(1)})

    def test_conv_missing_w_tensor_uid_raises(self) -> None:
        """Test that conv with missing w tensor UID raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1},  # missing w_tensor_uid
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {},
                }
            ]
        }

        with pytest.raises(ValueError, match="missing required tensor UIDs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(1)})

    def test_conv_missing_y_tensor_uid_raises(self) -> None:
        """Test that conv with missing y tensor UID raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {},  # missing y_tensor_uid
                    "parameters": {},
                }
            ]
        }

        with pytest.raises(ValueError, match="missing required tensor UIDs"):
            pytorch_ops.execute_graph(
                graph_json, {1: torch.zeros(1), 2: torch.zeros(1)}
            )

    def test_matmul_missing_tensor_uids_raises(self) -> None:
        """Test that matmul with missing tensor UIDs raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1},  # missing b_tensor_uid
                    "outputs": {"c_tensor_uid": 3},
                }
            ]
        }

        with pytest.raises(ValueError, match="missing required tensor UIDs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(2, 2)})

    def test_pointwise_missing_input_raises(self) -> None:
        """Test that pointwise with missing input tensor raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {"operation": "relu_fwd"},  # missing in_0_tensor_uid
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="missing required tensor UIDs"):
            pytorch_ops.execute_graph(graph_json, {})

    def test_pointwise_add_missing_second_input_raises(self) -> None:
        """Test that add operation without second input raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "add",
                        "in_0_tensor_uid": 1,
                        # missing in_1_tensor_uid
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="Add operation requires two inputs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(3)})

    def test_pointwise_mul_missing_second_input_raises(self) -> None:
        """Test that mul operation without second input raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "mul",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="Mul operation requires two inputs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(3)})

    def test_pointwise_sub_missing_second_input_raises(self) -> None:
        """Test that sub operation without second input raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "sub",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="Sub operation requires two inputs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(3)})

    def test_pointwise_div_missing_second_input_raises(self) -> None:
        """Test that div operation without second input raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "div",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="Div operation requires two inputs"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(3)})

    def test_pointwise_unsupported_operation_raises(self) -> None:
        """Test that unsupported pointwise operation raises ValueError."""
        graph_json = {
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "unknown_op",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ]
        }

        with pytest.raises(ValueError, match="Unsupported pointwise operation"):
            pytorch_ops.execute_graph(graph_json, {1: torch.zeros(3)})


class TestPyTorchOpsGetHandler:
    """Tests for get_handler function."""

    def test_get_handler_returns_handler_for_supported(self) -> None:
        """Test that get_handler returns handler for supported operations."""
        handler = pytorch_ops.get_handler("ConvolutionFwdAttributes")
        assert handler is not None
        assert callable(handler)

    def test_get_handler_returns_none_for_unsupported(self) -> None:
        """Test that get_handler returns None for unsupported operations."""
        handler = pytorch_ops.get_handler("UnknownOp")
        assert handler is None


class TestPyTorchOpsSupportsGraph:
    """Tests for graph support checking."""

    def test_supports_graph_empty_nodes(self) -> None:
        """Test that empty graph is supported."""
        graph_json = {"nodes": []}
        assert pytorch_ops.supports_graph(graph_json) is True

    def test_supports_graph_mixed_supported_unsupported(self) -> None:
        """Test that graph with unsupported ops returns False."""
        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "UnknownOp"},
            ]
        }
        assert pytorch_ops.supports_graph(graph_json) is False

    def test_get_unsupported_operations_returns_all_unsupported(self) -> None:
        """Test that all unsupported ops are returned."""
        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "Unknown1"},
                {"type": "Unknown2"},
            ]
        }
        unsupported = pytorch_ops.get_unsupported_operations(graph_json)
        assert "Unknown1" in unsupported
        assert "Unknown2" in unsupported
        assert "ConvolutionFwdAttributes" not in unsupported


class TestPyTorchOpsNewHandlers:
    """Registration and focused correctness checks for hipDNN op references."""

    @pytest.mark.parametrize(
        "op_type",
        [
            "SdpaAttributes",
        ],
    )
    def test_get_handler_returns_handler_for_new_ops(self, op_type: str) -> None:
        assert callable(pytorch_ops.get_handler(op_type))

    def test_grouped_conv_fwd_matches_torch(self) -> None:
        graph_json = {
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 3},
                    "parameters": {
                        "conv_mode": "CROSS_CORRELATION",
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ]
        }
        x = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4) / 10
        w = torch.arange(108, dtype=torch.float32).reshape(6, 2, 3, 3) / 20
        tensors = {1: x, 2: w}
        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(
            tensors[3], torch.nn.functional.conv2d(x, w, groups=2)
        )

    def test_sdpa_nonzero_dropout_raises(self) -> None:
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.5},
                }
            ]
        }
        q = torch.randn(1, 1, 2, 4)
        with pytest.raises(ValueError, match="Nonzero SDPA dropout"):
            pytorch_ops.execute_graph(graph_json, {1: q, 2: q, 3: q})
