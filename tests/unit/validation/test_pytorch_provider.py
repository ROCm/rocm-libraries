# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for PyTorchReferenceProvider.

These tests verify the PyTorch reference provider works correctly
without requiring GPU/hipDNN - they use stubbed input data and
verify the reference computations are correct.
"""

import numpy as np
import pytest

from dnn_benchmarking.validation import ReferenceProviderRegistry

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")


class TestPyTorchProviderAvailability:
    """Tests for PyTorchReferenceProvider availability."""

    def test_pytorch_provider_is_available(self) -> None:
        """Test that PyTorch provider is available when torch is installed."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.is_available() is True

    def test_pytorch_provider_name(self) -> None:
        """Test PyTorch provider name."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        assert provider.name == "pytorch"


class TestPyTorchProviderConv:
    """Tests for convolution operations."""

    def test_conv_fwd_basic(self) -> None:
        """Test basic 2D convolution forward pass."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        # Create simple graph JSON for conv
        graph_json = {
            "tensors": [
                {"uid": 0, "name": "output", "dims": [1, 1, 3, 3]},
                {"uid": 1, "name": "input", "dims": [1, 1, 5, 5]},
                {"uid": 2, "name": "weight", "dims": [1, 1, 3, 3]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "pre_padding": [0, 0],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        # Create input data
        input_x = np.ones((1, 1, 5, 5), dtype=np.float32)
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        input_data = {1: input_x, 2: weight}

        # Compute reference
        outputs = provider.compute_reference(graph_json, input_data)

        # Check output
        assert 0 in outputs
        output = outputs[0].data

        # 3x3 conv with all-ones kernel on all-ones input gives 9.0 at each position
        assert output.shape == (1, 1, 3, 3)
        assert np.allclose(output, 9.0)

    def test_conv_fwd_with_padding(self) -> None:
        """Test 2D convolution with padding."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 0, "name": "output", "dims": [1, 1, 5, 5]},
                {"uid": 1, "name": "input", "dims": [1, 1, 5, 5]},
                {"uid": 2, "name": "weight", "dims": [1, 1, 3, 3]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {
                        "pre_padding": [1, 1],
                        "stride": [1, 1],
                        "dilation": [1, 1],
                    },
                }
            ],
        }

        input_x = np.ones((1, 1, 5, 5), dtype=np.float32)
        weight = np.ones((1, 1, 3, 3), dtype=np.float32)
        input_data = {1: input_x, 2: weight}

        outputs = provider.compute_reference(graph_json, input_data)

        # With same padding, output shape matches input shape
        assert outputs[0].data.shape == (1, 1, 5, 5)


class TestPyTorchProviderMatmul:
    """Tests for matrix multiplication operations."""

    def test_matmul_basic(self) -> None:
        """Test basic matrix multiplication."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [2, 3]},
                {"uid": 2, "name": "b", "dims": [3, 4]},
                {"uid": 3, "name": "c", "dims": [2, 4]},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }

        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        assert 3 in outputs
        output = outputs[3].data

        # (2x3) @ (3x4) with all ones = 3.0 at each position
        assert output.shape == (2, 4)
        assert np.allclose(output, 3.0)

    def test_matmul_identity(self) -> None:
        """Test matmul with identity matrix."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3, 3]},
                {"uid": 2, "name": "b", "dims": [3, 3]},
                {"uid": 3, "name": "c", "dims": [3, 3]},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }

        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        b = np.eye(3, dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        # A @ I = A
        assert np.allclose(outputs[3].data, a)


class TestPyTorchProviderPointwise:
    """Tests for pointwise operations."""

    def test_relu(self) -> None:
        """Test ReLU activation."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [4]},
                {"uid": 2, "name": "output", "dims": [4]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "relu_fwd",
                        "in_0_tensor_uid": 1,
                        "relu_lower_clip": 0.0,
                        "relu_upper_clip": float("inf"),
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        assert np.allclose(outputs[2].data, expected)

    def test_relu6(self) -> None:
        """Test ReLU6 (clipped ReLU)."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [6]},
                {"uid": 2, "name": "output", "dims": [6]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "relu_fwd",
                        "in_0_tensor_uid": 1,
                        "relu_lower_clip": 0.0,
                        "relu_upper_clip": 6.0,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([-1.0, 0.0, 3.0, 6.0, 7.0, 10.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([0.0, 0.0, 3.0, 6.0, 6.0, 6.0], dtype=np.float32)
        assert np.allclose(outputs[2].data, expected)

    def test_add(self) -> None:
        """Test element-wise addition."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3]},
                {"uid": 2, "name": "b", "dims": [3]},
                {"uid": 3, "name": "c", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "add",
                        "in_0_tensor_uid": 1,
                        "in_1_tensor_uid": 2,
                    },
                    "outputs": {"out_0_tensor_uid": 3},
                }
            ],
        }

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([5.0, 7.0, 9.0], dtype=np.float32)
        assert np.allclose(outputs[3].data, expected)

    def test_mul(self) -> None:
        """Test element-wise multiplication."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [3]},
                {"uid": 2, "name": "b", "dims": [3]},
                {"uid": 3, "name": "c", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "mul",
                        "in_0_tensor_uid": 1,
                        "in_1_tensor_uid": 2,
                    },
                    "outputs": {"out_0_tensor_uid": 3},
                }
            ],
        }

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        input_data = {1: a, 2: b}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.array([4.0, 10.0, 18.0], dtype=np.float32)
        assert np.allclose(outputs[3].data, expected)

    def test_tanh(self) -> None:
        """Test tanh activation."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [3]},
                {"uid": 2, "name": "output", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "tanh_fwd",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        input_data = {1: input_x}

        outputs = provider.compute_reference(graph_json, input_data)

        expected = np.tanh(input_x)
        assert np.allclose(outputs[2].data, expected)


class TestPyTorchProviderGraphSupport:
    """Tests for graph support checking."""

    def test_supports_graph_with_known_ops(self) -> None:
        """Test supports_graph returns True for supported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "MatmulAttributes"},
                {"type": "PointwiseAttributes"},
            ]
        }

        assert provider.supports_graph(graph_json) is True

    def test_supports_graph_with_unknown_ops(self) -> None:
        """Test supports_graph returns False for unsupported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "UnknownOperation"},
            ]
        }

        assert provider.supports_graph(graph_json) is False

    def test_get_unsupported_operations(self) -> None:
        """Test getting list of unsupported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "nodes": [
                {"type": "ConvolutionFwdAttributes"},
                {"type": "UnknownOp1"},
                {"type": "UnknownOp2"},
            ]
        }

        unsupported = provider.get_unsupported_operations(graph_json)

        assert "UnknownOp1" in unsupported
        assert "UnknownOp2" in unsupported
        assert "ConvolutionFwdAttributes" not in unsupported

    def test_supported_operations(self) -> None:
        """Test getting set of supported operations."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        supported = provider.supported_operations()

        assert "ConvolutionFwdAttributes" in supported
        assert "MatmulAttributes" in supported
        assert "PointwiseAttributes" in supported


class TestPyTorchProviderErrors:
    """Tests for error handling."""

    def test_unsupported_operation_raises(self) -> None:
        """Test that unsupported operation raises ValueError."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {"nodes": [{"type": "UnsupportedOperation"}]}

        with pytest.raises(ValueError) as exc_info:
            provider.compute_reference(graph_json, {})

        assert "unsupported" in str(exc_info.value).lower()

    def test_unsupported_pointwise_operation_raises(self) -> None:
        """Test that unsupported pointwise operation raises ValueError."""
        provider = ReferenceProviderRegistry.get_provider("pytorch")

        graph_json = {
            "tensors": [
                {"uid": 1, "name": "input", "dims": [3]},
                {"uid": 2, "name": "output", "dims": [3]},
            ],
            "nodes": [
                {
                    "type": "PointwiseAttributes",
                    "inputs": {
                        "operation": "unknown_pointwise_op",
                        "in_0_tensor_uid": 1,
                    },
                    "outputs": {"out_0_tensor_uid": 2},
                }
            ],
        }

        input_data = {1: np.array([1.0, 2.0, 3.0])}

        with pytest.raises(ValueError) as exc_info:
            provider.compute_reference(graph_json, input_data)

        assert "unsupported" in str(exc_info.value).lower()
