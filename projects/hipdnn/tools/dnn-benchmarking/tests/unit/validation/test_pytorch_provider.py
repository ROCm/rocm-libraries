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

    def test_matmul_bfloat16_uses_graph_dtype_and_returns_float32(self) -> None:
        torch = pytest.importorskip("torch")
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "tensors": [
                {"uid": 1, "name": "a", "dims": [2, 3], "data_type": "bfloat16"},
                {"uid": 2, "name": "b", "dims": [3, 2], "data_type": "bfloat16"},
                {"uid": 3, "name": "c", "dims": [2, 2], "data_type": "bfloat16"},
            ],
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ],
        }
        a = np.array(
            [[1.00390625, -2.25, 0.33398438], [4.5, 1.0078125, -0.5]],
            dtype=np.float32,
        )
        b = np.array(
            [[0.25, -1.5], [2.0, 0.75], [-3.0, 1.25]],
            dtype=np.float32,
        )

        outputs = provider.compute_reference(graph_json, {1: a, 2: b})
        expected = (
            torch.matmul(
                torch.from_numpy(a).to(dtype=torch.bfloat16),
                torch.from_numpy(b).to(dtype=torch.bfloat16),
            )
            .float()
            .numpy()
        )

        assert outputs[3].data.dtype == np.float32
        np.testing.assert_array_equal(outputs[3].data, expected)


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
        assert "SdpaAttributes" in supported


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


class TestPyTorchProviderNewOps:
    """Reference correctness for newly supported hipDNN operation types."""

    def test_matmul_batched_broadcast(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        graph_json = {
            "nodes": [
                {
                    "type": "MatmulAttributes",
                    "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                    "outputs": {"c_tensor_uid": 3},
                }
            ]
        }
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b = np.arange(20, dtype=np.float32).reshape(1, 4, 5)
        outputs = provider.compute_reference(graph_json, {1: a, 2: b})

        np.testing.assert_allclose(
            outputs[3].data,
            torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
        )

    def test_sdpa_forward_matches_torch_and_returns_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 6, "dims": [1, 1, 2, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0
        )
        expected_stats = torch.logsumexp(
            torch.matmul(q_t, k_t.transpose(-2, -1)) / torch.sqrt(torch.tensor(2.0)),
            dim=-1,
            keepdim=True,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, expected_stats.numpy(), rtol=1e-6)
        assert outputs[6].data.shape == (1, 1, 2, 1)

    def test_sdpa_causal_forward_matches_torch_and_returns_masked_stats(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 3, 2], "data_type": "float"},
                {"uid": 6, "dims": [1, 1, 3, 1], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, "stats_tensor_uid": 6},
                    "attributes": {"causal_mask": True, "dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        q_t = torch.from_numpy(q)
        k_t = torch.from_numpy(k)
        v_t = torch.from_numpy(v)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, is_causal=True
        )
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(2.0)
        )
        causal = torch.ones(3, 3, dtype=torch.bool).tril()
        expected_stats = torch.logsumexp(
            scores.masked_fill(~causal, float("-inf")), dim=-1, keepdim=True
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(outputs[6].data, expected_stats.numpy(), rtol=1e-6)

    def test_sdpa_additive_attention_mask_matches_torch(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        mask = np.array([[[[0.0, -100.0], [0.0, 0.0]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "float"},
                {"uid": 5, "dims": [1, 1, 2, 2], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "attn_mask_tensor_uid": 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v, 5: mask})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            attn_mask=torch.from_numpy(mask),
            dropout_p=0.0,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_gqa_matches_repeated_key_value_heads(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 10.0
        k = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4) / 11.0
        v = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4) / 13.0
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 2, 2, 4], "data_type": "float"},
                {"uid": 2, "dims": [1, 1, 2, 4], "data_type": "float"},
                {"uid": 3, "dims": [1, 1, 2, 4], "data_type": "float"},
                {"uid": 4, "dims": [1, 2, 2, 4], "data_type": "float"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k).repeat_interleave(2, dim=-3),
            torch.from_numpy(v).repeat_interleave(2, dim=-3),
            dropout_p=0.0,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_attention_scale_value_matches_torch(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {
                        "attn_scale_value": 0.25,
                        "dropout_probability": 0.0,
                    },
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q),
            torch.from_numpy(k),
            torch.from_numpy(v),
            dropout_p=0.0,
            scale=0.25,
        )

        np.testing.assert_allclose(outputs[4].data, expected.numpy(), rtol=1e-6)

    def test_sdpa_bfloat16_uses_graph_dtype(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[0.10, 0.20], [0.30, 0.40]]]], dtype=np.float32)
        k = np.array([[[[0.50, 0.60], [0.70, 0.80]]]], dtype=np.float32)
        v = np.array([[[[0.90, 0.25], [0.125, 0.75]]]], dtype=np.float32)
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 2, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 3, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
                {"uid": 4, "dims": [1, 1, 2, 2], "data_type": "bfloat16"},
            ],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        outputs = provider.compute_reference(graph_json, {1: q, 2: k, 3: v})
        expected = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q).to(torch.bfloat16),
            torch.from_numpy(k).to(torch.bfloat16),
            torch.from_numpy(v).to(torch.bfloat16),
            dropout_p=0.0,
        ).to(torch.float32)

        assert outputs[4].data.dtype == np.float32
        np.testing.assert_array_equal(outputs[4].data, expected.numpy())

    @pytest.mark.parametrize(
        "optional_output",
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "rng_dump_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    def test_sdpa_forward_rejects_unsupported_optional_outputs(
        self, optional_output: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        k = q.copy()
        v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4, optional_output: 5},
                    "attributes": {"dropout_probability": 0.0},
                }
            ]
        }

        with pytest.raises(ValueError, match=optional_output):
            provider.compute_reference(graph_json, {1: q, 2: k, 3: v})

    @pytest.mark.parametrize(
        "optional_input",
        [
            "seq_len_q_tensor_uid",
            "seq_len_kv_tensor_uid",
            "seed_tensor_uid",
            "offset_tensor_uid",
            "dropout_mask_tensor_uid",
            "dropout_scale_tensor_uid",
            "page_table_k_tensor_uid",
            "page_table_v_tensor_uid",
            "block_mask_tensor_uid",
            "sink_token_tensor_uid",
            "descale_q_tensor_uid",
            "descale_k_tensor_uid",
            "descale_v_tensor_uid",
            "descale_s_tensor_uid",
            "scale_s_tensor_uid",
            "scale_o_tensor_uid",
        ],
    )
    def test_sdpa_forward_rejects_unsupported_optional_inputs(
        self, optional_input: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        optional_input: 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0},
                }
            ],
        }

        with pytest.raises(ValueError, match=optional_input):
            provider.compute_reference(
                graph_json,
                {1: q, 2: q, 3: q, 5: np.array([1], dtype=np.int32)},
            )

    @pytest.mark.parametrize(
        "attributes,match",
        [
            ({"alibi_mask": True}, "alibi/padding"),
            ({"padding_mask": True}, "alibi/padding"),
            ({"causal_mask_bottom_right": True}, "bottom-right causal"),
            ({"diagonal_alignment": "BOTTOM_RIGHT"}, "TOP_LEFT"),
            ({"left_bound": 1}, "sliding-window"),
            ({"right_bound": 1}, "sliding-window"),
        ],
    )
    def test_sdpa_forward_rejects_unsupported_attributes(
        self, attributes: dict, match: str
    ) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {"q_tensor_uid": 1, "k_tensor_uid": 2, "v_tensor_uid": 3},
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"dropout_probability": 0.0, **attributes},
                }
            ],
        }

        with pytest.raises(ValueError, match=match):
            provider.compute_reference(graph_json, {1: q, 2: q, 3: q})

    def test_sdpa_forward_rejects_attn_mask_with_causal_mask(self) -> None:
        provider = ReferenceProviderRegistry.get_provider("pytorch")
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        graph_json = {
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "attn_mask_tensor_uid": 5,
                    },
                    "outputs": {"o_tensor_uid": 4},
                    "attributes": {"causal_mask": True, "dropout_probability": 0.0},
                }
            ],
        }

        with pytest.raises(ValueError, match="both attn_mask and causal_mask"):
            provider.compute_reference(
                graph_json,
                {1: q, 2: q, 3: q, 5: np.zeros((1, 1, 2, 2), dtype=np.float32)},
            )
