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

    def test_supported_operations(self) -> None:
        """Test that expected operations are registered as supported."""
        supported = pytorch_ops.get_supported_operations()
        assert "ConvolutionFwdAttributes" in supported
        assert "MatmulAttributes" in supported
        assert "PointwiseAttributes" in supported
        assert "SdpaAttributes" in supported

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

    @pytest.mark.parametrize(
        "op_type",
        [
            "LayernormBackwardAttributes",
            "CustomOpAttributes",
            "BlockScaleDequantizeAttributes",
            "BlockScaleQuantizeAttributes",
        ],
    )
    def test_intentionally_unsupported_ops_remain_unsupported(
        self, op_type: str
    ) -> None:
        graph_json = {"nodes": [{"type": op_type}]}

        assert pytorch_ops.supports_graph(graph_json) is False
        assert pytorch_ops.get_handler(op_type) is None


class TestPyTorchOpsNewHandlers:
    """Registration and focused correctness checks for hipDNN op references."""

    @pytest.mark.parametrize(
        "op_type",
        [
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
            "BatchnormAttributes",
            "BatchnormInferenceAttributes",
            "BatchnormInferenceAttributesVarianceExt",
            "BatchnormBackwardAttributes",
            "SdpaAttributes",
            "SdpaBackwardAttributes",
            "LayernormAttributes",
            "RMSNormAttributes",
            "RMSNormBackwardAttributes",
            "ReductionAttributes",
            "ResampleFwdAttributes",
        ],
    )
    def test_get_handler_returns_handler_for_new_ops(self, op_type: str) -> None:
        assert callable(pytorch_ops.get_handler(op_type))

    def test_conv_dgrad_matches_torch_grad(self) -> None:
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 3, 3]},
                {"uid": 2, "dims": [1, 1, 2, 2]},
                {"uid": 3, "dims": [1, 1, 4, 4]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionBwdAttributes",
                    "inputs": {"dy_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"dx_tensor_uid": 3},
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
        dy = torch.ones(1, 1, 3, 3)
        w = torch.ones(1, 1, 2, 2)
        tensors = {1: dy, 2: w}
        pytorch_ops.execute_graph(graph_json, tensors)

        expected = torch.nn.grad.conv2d_input((1, 1, 4, 4), w, dy)
        torch.testing.assert_close(tensors[3], expected)

    def test_conv_wgrad_matches_torch_grad(self) -> None:
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [1, 1, 4, 4]},
                {"uid": 2, "dims": [1, 1, 3, 3]},
                {"uid": 3, "dims": [1, 1, 2, 2]},
            ],
            "nodes": [
                {
                    "type": "ConvolutionWrwAttributes",
                    "inputs": {"x_tensor_uid": 1, "dy_tensor_uid": 2},
                    "outputs": {"dw_tensor_uid": 3},
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
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        dy = torch.ones(1, 1, 3, 3)
        tensors = {1: x, 2: dy}
        pytorch_ops.execute_graph(graph_json, tensors)

        expected = torch.nn.grad.conv2d_weight(x, (1, 1, 2, 2), dy)
        torch.testing.assert_close(tensors[3], expected)

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

    def test_grouped_conv_dgrad_and_wgrad_match_torch(self) -> None:
        params = {
            "conv_mode": "CROSS_CORRELATION",
            "pre_padding": [0, 0],
            "post_padding": [0, 0],
            "stride": [1, 1],
            "dilation": [1, 1],
        }
        dgrad_graph = {
            "tensors": [{"uid": 3, "dims": [1, 4, 4, 4]}],
            "nodes": [
                {
                    "type": "ConvolutionBwdAttributes",
                    "inputs": {"dy_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"dx_tensor_uid": 3},
                    "parameters": params,
                }
            ],
        }
        wrw_graph = {
            "tensors": [{"uid": 4, "dims": [6, 2, 3, 3]}],
            "nodes": [
                {
                    "type": "ConvolutionWrwAttributes",
                    "inputs": {"x_tensor_uid": 3, "dy_tensor_uid": 1},
                    "outputs": {"dw_tensor_uid": 4},
                    "parameters": params,
                }
            ],
        }
        dy = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2) / 10
        w = torch.arange(108, dtype=torch.float32).reshape(6, 2, 3, 3) / 20
        x = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4) / 30

        dgrad_tensors = {1: dy, 2: w}
        pytorch_ops.execute_graph(dgrad_graph, dgrad_tensors)
        torch.testing.assert_close(
            dgrad_tensors[3],
            torch.nn.grad.conv2d_input((1, 4, 4, 4), w, dy, groups=2),
        )

        wrw_tensors = {1: dy, 3: x}
        pytorch_ops.execute_graph(wrw_graph, wrw_tensors)
        torch.testing.assert_close(
            wrw_tensors[4],
            torch.nn.grad.conv2d_weight(x, (6, 2, 3, 3), dy, groups=2),
        )

    def test_batchnorm_backward_outputs_expected_reductions(self) -> None:
        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 2,
                        "mean_tensor_uid": 3,
                        "inv_variance_tensor_uid": 4,
                        "scale_tensor_uid": 5,
                    },
                    "outputs": {
                        "dx_tensor_uid": 6,
                        "dscale_tensor_uid": 7,
                        "dbias_tensor_uid": 8,
                    },
                }
            ]
        }
        x = torch.tensor([[[[1.0, 3.0]]]])
        dy = torch.tensor([[[[2.0, 4.0]]]])
        tensors = {
            1: dy,
            2: x,
            3: torch.tensor([[[[2.0]]]]),
            4: torch.tensor([[[[1.0]]]]),
            5: torch.tensor([[[[1.0]]]]),
        }

        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(tensors[7], torch.tensor([[[[2.0]]]]))
        torch.testing.assert_close(tensors[8], torch.tensor([[[[6.0]]]]))
        torch.testing.assert_close(tensors[6], torch.zeros(1, 1, 1, 2))

    def test_batchnorm_backward_matches_autograd(self) -> None:
        torch.manual_seed(0)
        x_t = torch.randn(4, 3, 2, 2, requires_grad=True)
        gamma = torch.randn(3, requires_grad=True)
        beta = torch.randn(3, requires_grad=True)
        dy = torch.randn(4, 3, 2, 2)

        # Independent oracle: differentiate the batchnorm-training forward
        # expression with autograd. The handler uses a hand-derived closed-form
        # backward, so autograd is a genuinely independent check (recompute path:
        # no saved mean/inv-variance, epsilon 1e-5, biased variance).
        reduce = (0, 2, 3)
        mean = x_t.mean(dim=reduce, keepdim=True)
        var = x_t.var(dim=reduce, unbiased=False, keepdim=True)
        x_hat = (x_t - mean) * torch.rsqrt(var + 1e-5)
        y = gamma.view(1, 3, 1, 1) * x_hat + beta.view(1, 3, 1, 1)
        y.backward(dy)

        graph_json = {
            "nodes": [
                {
                    "type": "BatchnormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 2,
                        "scale_tensor_uid": 3,
                    },
                    "outputs": {
                        "dx_tensor_uid": 4,
                        "dscale_tensor_uid": 5,
                        "dbias_tensor_uid": 6,
                    },
                }
            ]
        }
        tensors = {1: dy, 2: x_t.detach(), 3: gamma.detach()}

        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(tensors[4], x_t.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(
            tensors[5].reshape(-1), gamma.grad, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            tensors[6].reshape(-1), beta.grad, rtol=1e-4, atol=1e-5
        )

    def test_layernorm_matches_torch_and_aux_outputs(self) -> None:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [2, 3, 4]},
                {"uid": 6, "dims": [2, 3, 1]},
                {"uid": 7, "dims": [2, 3, 1]},
            ],
            "nodes": [
                {
                    "type": "LayernormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "bias_tensor_uid": 3,
                        "epsilon_tensor_uid": 4,
                    },
                    "outputs": {
                        "y_tensor_uid": 5,
                        "mean_tensor_uid": 6,
                        "inv_variance_tensor_uid": 7,
                    },
                    "attributes": {"normalized_dim_count": 1},
                }
            ],
        }
        tensors = {
            1: x,
            2: torch.tensor([1.0, 0.5, 2.0, -1.0]),
            3: torch.tensor([0.0, 1.0, -0.5, 0.25]),
            4: torch.tensor([1e-5]),
        }

        pytorch_ops.execute_graph(graph_json, tensors)

        expected = torch.nn.functional.layer_norm(
            x, (4,), weight=tensors[2], bias=tensors[3], eps=1e-5
        )
        torch.testing.assert_close(tensors[5], expected)
        torch.testing.assert_close(tensors[6], x.mean(dim=2, keepdim=True))
        torch.testing.assert_close(
            tensors[7], torch.rsqrt(x.var(dim=2, unbiased=False, keepdim=True) + 1e-5)
        )

    def test_rmsnorm_trailing_matches_manual_formula(self) -> None:
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
        scale = torch.tensor([1.0, 0.5, 2.0, -1.0])
        graph_json = {
            "tensors": [{"uid": 4, "dims": [2, 3, 4]}],
            "nodes": [
                {
                    "type": "RMSNormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "epsilon_tensor_uid": 3,
                    },
                    "outputs": {"y_tensor_uid": 4},
                }
            ],
        }
        tensors = {1: x, 2: scale, 3: torch.tensor([1e-5])}

        pytorch_ops.execute_graph(graph_json, tensors)

        inv = torch.rsqrt(x.square().mean(dim=2, keepdim=True) + 1e-5)
        torch.testing.assert_close(tensors[4], x * inv * scale.reshape(1, 1, 4))

    def test_rmsnorm_channel_bias_and_inv_outputs(self) -> None:
        x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 2, 2, 4)
        scale = torch.tensor([[[[2.0]], [[0.5]]]])
        bias = torch.tensor([[[[0.25]], [[-1.0]]]])
        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [1, 2, 2, 4]},
                {"uid": 6, "dims": [1, 2, 1, 1]},
            ],
            "nodes": [
                {
                    "type": "RMSNormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "epsilon_tensor_uid": 3,
                        "bias_tensor_uid": 4,
                    },
                    "outputs": {"y_tensor_uid": 5, "inv_rms_tensor_uid": 6},
                }
            ],
        }
        tensors = {1: x, 2: scale, 3: torch.tensor([1e-5]), 4: bias}

        pytorch_ops.execute_graph(graph_json, tensors)

        inv = torch.rsqrt(x.square().mean(dim=(2, 3), keepdim=True) + 1e-5)
        torch.testing.assert_close(tensors[6], inv)
        torch.testing.assert_close(tensors[5], x * inv * scale + bias)

    def test_rmsnorm_backward_matches_autograd(self) -> None:
        x = (
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        ).requires_grad_()
        scale = torch.tensor([1.0, 0.5, 2.0, -1.0], requires_grad=True)
        dy = torch.linspace(-0.2, 0.3, steps=24).reshape(2, 3, 4)
        y = x * torch.rsqrt(x.square().mean(dim=2, keepdim=True) + 1e-5) * scale
        y.backward(dy)

        inv = torch.rsqrt(x.detach().square().mean(dim=2, keepdim=True) + 1e-5)
        graph_json = {
            "tensors": [
                {"uid": 5, "dims": [2, 3, 4]},
                {"uid": 6, "dims": [4]},
                {"uid": 7, "dims": [4]},
            ],
            "nodes": [
                {
                    "type": "RMSNormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 2,
                        "scale_tensor_uid": 3,
                        "inv_rms_tensor_uid": 4,
                    },
                    "outputs": {
                        "dx_tensor_uid": 5,
                        "dscale_tensor_uid": 6,
                        "dbias_tensor_uid": 7,
                    },
                }
            ],
        }
        tensors = {1: dy, 2: x.detach(), 3: scale.detach(), 4: inv}

        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(tensors[5], x.grad)
        torch.testing.assert_close(tensors[6], scale.grad)
        torch.testing.assert_close(tensors[7], dy.sum(dim=(0, 1)))

    def test_batchnorm_variance_ext_uses_builtin_batch_norm(self) -> None:
        x = torch.randn(2, 3, 4, 4)
        mean = torch.randn(3)
        variance = torch.rand(3) + 0.5
        scale = torch.randn(3)
        bias = torch.randn(3)
        graph_json = {
            "tensors": [{"uid": 7, "dims": [2, 3, 4, 4]}],
            "nodes": [
                {
                    "type": "BatchnormInferenceAttributesVarianceExt",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "mean_tensor_uid": 2,
                        "variance_tensor_uid": 3,
                        "scale_tensor_uid": 4,
                        "bias_tensor_uid": 5,
                        "epsilon_tensor_uid": 6,
                    },
                    "outputs": {"y_tensor_uid": 7},
                }
            ],
        }
        tensors = {
            1: x,
            2: mean.reshape(1, 3, 1, 1),
            3: variance.reshape(1, 3, 1, 1),
            4: scale.reshape(1, 3, 1, 1),
            5: bias.reshape(1, 3, 1, 1),
            6: torch.tensor([1e-4]),
        }

        pytorch_ops.execute_graph(graph_json, tensors)

        expected = torch.nn.functional.batch_norm(
            x, mean, variance, weight=scale, bias=bias, training=False, eps=1e-4
        )
        torch.testing.assert_close(tensors[7], expected)
        # Uses the PyTorch batchnorm primitive, so it is not flagged as manual.
        assert pytorch_ops.get_reference_warnings(graph_json) == []

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("ADD", torch.tensor([1.0])),
            ("MUL", torch.tensor([0.0])),
            ("MIN", torch.tensor([-5.0])),
            ("MAX", torch.tensor([4.0])),
            ("AMAX", torch.tensor([5.0])),
            ("AVG", torch.tensor([1.0 / 6.0])),
            ("NORM1", torch.tensor([15.0])),
            ("NORM2", torch.tensor([(55.0) ** 0.5])),
            ("MUL_NO_ZEROS", torch.tensor([120.0])),
        ],
    )
    def test_reduction_modes_match_torch(
        self, mode: str, expected: torch.Tensor
    ) -> None:
        graph_json = {
            "tensors": [{"uid": 2, "dims": [1]}],
            "nodes": [
                {
                    "type": "ReductionAttributes",
                    "inputs": {"in_tensor_uid": 1},
                    "outputs": {"out_tensor_uid": 2},
                    "attributes": {"mode": mode},
                }
            ],
        }
        tensors = {1: torch.tensor([[-2.0, 0.0, 3.0], [4.0, -5.0, 1.0]])}

        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(tensors[2], expected)

    def test_resample_maxpool_matches_torch_and_indices(self) -> None:
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        graph_json = {
            "tensors": [
                {"uid": 2, "dims": [1, 1, 2, 2]},
                {"uid": 3, "dims": [1, 1, 2, 2]},
            ],
            "nodes": [
                {
                    "type": "ResampleFwdAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2, "index_tensor_uid": 3},
                    "attributes": {
                        "window": [2, 2],
                        "stride": [2, 2],
                        "pre_padding": [0, 0],
                        "post_padding": [0, 0],
                        "resample_mode": "MAXPOOL",
                        "padding_mode": "NEG_INF_PAD",
                    },
                }
            ],
        }
        tensors = {1: x}

        pytorch_ops.execute_graph(graph_json, tensors)

        expected, indices = torch.nn.functional.max_pool2d(
            x, kernel_size=(2, 2), stride=(2, 2), return_indices=True
        )
        torch.testing.assert_close(tensors[2], expected)
        torch.testing.assert_close(tensors[3], indices)

    def test_resample_avgpool_asymmetric_exclude_matches_valid_count(self) -> None:
        x = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, 5)
        graph_json = {
            "tensors": [{"uid": 2, "dims": [1, 1, 2]}],
            "nodes": [
                {
                    "type": "ResampleFwdAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2},
                    "attributes": {
                        "pre_padding": [1],
                        "post_padding": [0],
                        "window": [3],
                        "stride": [2],
                        "resample_mode": "AVGPOOL_EXCLUDE_PADDING",
                        "padding_mode": "ZERO_PAD",
                    },
                }
            ],
        }
        tensors = {1: x}

        pytorch_ops.execute_graph(graph_json, tensors)

        torch.testing.assert_close(tensors[2], torch.tensor([[[1.5, 3.0]]]))

    def test_reference_warnings_describe_manual_paths(self) -> None:
        graph_json = {
            "tensors": [
                {"uid": 1, "dims": [2, 3, 4]},
                {"uid": 2, "dims": [4]},
                {"uid": 8, "dims": [1, 1, 5]},
            ],
            "nodes": [
                {
                    "name": "ln",
                    "type": "LayernormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "bias_tensor_uid": 2,
                        "epsilon_tensor_uid": 3,
                    },
                    "outputs": {"y_tensor_uid": 4, "mean_tensor_uid": 5},
                },
                {
                    "name": "rms_bwd",
                    "type": "RMSNormBackwardAttributes",
                    "inputs": {
                        "dy_tensor_uid": 1,
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "inv_rms_tensor_uid": 6,
                    },
                    "outputs": {"dx_tensor_uid": 4, "dscale_tensor_uid": 2},
                },
                {
                    "name": "mul_no_zeros",
                    "type": "ReductionAttributes",
                    "inputs": {"in_tensor_uid": 1},
                    "outputs": {"out_tensor_uid": 7},
                    "attributes": {"mode": "MUL_NO_ZEROS"},
                },
                {
                    "name": "avgpool",
                    "type": "ResampleFwdAttributes",
                    "inputs": {"x_tensor_uid": 8},
                    "outputs": {"y_tensor_uid": 9},
                    "attributes": {
                        "pre_padding": [1],
                        "post_padding": [0],
                        "window": [3],
                        "stride": [2],
                        "resample_mode": "AVGPOOL_EXCLUDE_PADDING",
                        "padding_mode": "ZERO_PAD",
                    },
                },
                {
                    "name": "sdpa_bwd",
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 1,
                        "v_tensor_uid": 1,
                        "o_tensor_uid": 1,
                        "do_tensor_uid": 1,
                        "stats_tensor_uid": 2,
                    },
                    "outputs": {
                        "dq_tensor_uid": 1,
                        "dk_tensor_uid": 1,
                        "dv_tensor_uid": 1,
                    },
                },
            ],
        }

        warnings = pytorch_ops.get_reference_warnings(graph_json)

        assert len(warnings) == 5
        assert all("not solely built-in PyTorch operator time" in w for w in warnings)
        assert any("LayernormAttributes" in w for w in warnings)
        assert any("RMSNormBackwardAttributes" in w for w in warnings)
        assert any("MUL_NO_ZEROS" in w for w in warnings)
        assert any("AVGPOOL_EXCLUDE_PADDING" in w for w in warnings)
        assert any("SdpaBackwardAttributes" in w for w in warnings)

    def test_builtin_rmsnorm_reference_has_no_warning_when_available(self) -> None:
        graph_json = {
            "tensors": [{"uid": 1, "dims": [2, 3, 4]}, {"uid": 2, "dims": [4]}],
            "nodes": [
                {
                    "type": "RMSNormAttributes",
                    "inputs": {
                        "x_tensor_uid": 1,
                        "scale_tensor_uid": 2,
                        "epsilon_tensor_uid": 3,
                    },
                    "outputs": {"y_tensor_uid": 4},
                }
            ],
        }

        warnings = pytorch_ops.get_reference_warnings(graph_json)

        if hasattr(torch.nn.functional, "rms_norm"):
            assert warnings == []
        else:
            assert warnings

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

    @staticmethod
    def _sdpa_autograd_reference(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        scale: float,
        causal: bool,
        repeat_k: int,
        repeat_v: int,
    ):
        q_g = q.clone().requires_grad_()
        k_g = k.clone().requires_grad_()
        v_g = v.clone().requires_grad_()
        k_e = k_g.repeat_interleave(repeat_k, dim=1) if repeat_k > 1 else k_g
        v_e = v_g.repeat_interleave(repeat_v, dim=1) if repeat_v > 1 else v_g
        scores = (q_g @ k_e.transpose(-2, -1)) * scale
        if causal:
            mask = torch.ones(
                scores.shape[-2], scores.shape[-1], dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = probs @ v_e
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        out.backward(do)
        return out.detach(), lse.detach(), q_g.grad, k_g.grad, v_g.grad

    @staticmethod
    def _run_sdpa_backward(q, k, v, o, do, stats, attrs):
        graph_json = {
            "tensors": [
                {"uid": 10, "dims": list(q.shape)},
                {"uid": 11, "dims": list(k.shape)},
                {"uid": 12, "dims": list(v.shape)},
            ],
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "o_tensor_uid": 4,
                        "do_tensor_uid": 5,
                        "stats_tensor_uid": 6,
                    },
                    "outputs": {
                        "dq_tensor_uid": 10,
                        "dk_tensor_uid": 11,
                        "dv_tensor_uid": 12,
                    },
                    "attributes": attrs,
                }
            ],
        }
        tensors = {1: q, 2: k, 3: v, 4: o, 5: do, 6: stats}
        pytorch_ops.execute_graph(graph_json, tensors)
        return tensors[10], tensors[11], tensors[12]

    @pytest.mark.parametrize("causal", [False, True])
    def test_sdpa_backward_matches_autograd_with_consistent_stats(
        self, causal: bool
    ) -> None:
        torch.manual_seed(0)
        scale = 1.0 / (8**0.5)
        q = torch.randn(2, 3, 5, 8)
        k = torch.randn(2, 3, 5, 8)
        v = torch.randn(2, 3, 5, 8)
        do = torch.randn(2, 3, 5, 8)
        o, lse, dq_ref, dk_ref, dv_ref = self._sdpa_autograd_reference(
            q, k, v, do, scale, causal, 1, 1
        )

        attrs = {"attn_scale_value": scale}
        if causal:
            attrs["causal_mask"] = True
        dq, dk, dv = self._run_sdpa_backward(q, k, v, o, do, lse, attrs)

        torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)

    def test_sdpa_backward_gqa_matches_autograd(self) -> None:
        torch.manual_seed(1)
        scale = 1.0 / (8**0.5)
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 8)
        do = torch.randn(2, 4, 5, 8)
        o, lse, dq_ref, dk_ref, dv_ref = self._sdpa_autograd_reference(
            q, k, v, do, scale, False, 2, 2
        )

        dq, dk, dv = self._run_sdpa_backward(
            q, k, v, o, do, lse, {"attn_scale_value": scale}
        )

        torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)

    def test_sdpa_backward_consumes_stats_input(self) -> None:
        torch.manual_seed(2)
        scale = 1.0 / (8**0.5)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        do = torch.randn(1, 2, 4, 8)
        o, lse, _, _, _ = self._sdpa_autograd_reference(q, k, v, do, scale, False, 1, 1)

        dq_true, _, _ = self._run_sdpa_backward(
            q, k, v, o, do, lse, {"attn_scale_value": scale}
        )
        dq_shift, _, _ = self._run_sdpa_backward(
            q, k, v, o, do, lse + 0.5, {"attn_scale_value": scale}
        )

        # hipDNN consumes stats verbatim (no renormalization); perturbing it must
        # change the gradient, which is exactly why autograd cannot be used.
        assert (dq_true - dq_shift).abs().max().item() > 1e-3

    def test_sdpa_backward_rejects_dbias(self) -> None:
        q = torch.randn(1, 1, 2, 4)
        graph_json = {
            "tensors": [{"uid": 10, "dims": [1, 1, 2, 4]}],
            "nodes": [
                {
                    "type": "SdpaBackwardAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                        "o_tensor_uid": 4,
                        "do_tensor_uid": 5,
                        "stats_tensor_uid": 6,
                    },
                    "outputs": {
                        "dq_tensor_uid": 10,
                        "dk_tensor_uid": 11,
                        "dv_tensor_uid": 12,
                        "dbias_tensor_uid": 13,
                    },
                }
            ],
        }
        stats = torch.zeros(1, 1, 2, 1)
        with pytest.raises(ValueError, match="dBias"):
            pytorch_ops.execute_graph(
                graph_json, {1: q, 2: q, 3: q, 4: q, 5: q, 6: stats}
            )

    def test_sdpa_backward_independent_kv_heads(self) -> None:
        # Hq == Hk but Hv < Hq: regression for the single-flag GQA bug that
        # skipped V expansion/reduction whenever Hq == Hk.
        torch.manual_seed(3)
        scale = 1.0 / (8**0.5)
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 1, 5, 8)
        do = torch.randn(2, 4, 5, 8)
        o, lse, dq_ref, dk_ref, dv_ref = self._sdpa_autograd_reference(
            q, k, v, do, scale, False, 1, 4
        )

        dq, dk, dv = self._run_sdpa_backward(
            q, k, v, o, do, lse, {"attn_scale_value": scale}
        )

        assert dk.shape == k.shape
        assert dv.shape == v.shape
        torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)

    def test_sdpa_backward_distinct_kv_head_counts(self) -> None:
        # Hk != Hv, both < Hq.
        torch.manual_seed(4)
        scale = 1.0 / (8**0.5)
        q = torch.randn(2, 8, 5, 8)
        k = torch.randn(2, 4, 5, 8)
        v = torch.randn(2, 2, 5, 8)
        do = torch.randn(2, 8, 5, 8)
        o, lse, dq_ref, dk_ref, dv_ref = self._sdpa_autograd_reference(
            q, k, v, do, scale, False, 2, 4
        )

        dq, dk, dv = self._run_sdpa_backward(
            q, k, v, o, do, lse, {"attn_scale_value": scale}
        )

        assert dk.shape == k.shape
        assert dv.shape == v.shape
        torch.testing.assert_close(dq, dq_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dk, dk_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dv, dv_ref, rtol=1e-4, atol=1e-4)

    def test_sdpa_backward_rejects_indivisible_v_heads(self) -> None:
        scale = 1.0 / (8**0.5)
        q = torch.randn(1, 4, 4, 8)
        v = torch.randn(1, 3, 4, 8)  # 3 does not divide Hq=4
        stats = torch.zeros(1, 4, 4, 1)
        with pytest.raises(ValueError, match="V head count"):
            self._run_sdpa_backward(q, q, v, q, q, stats, {"attn_scale_value": scale})

    def test_sdpa_forward_independent_kv_heads(self) -> None:
        # Hk != Hv (2 and 1, both divide Hq=4): explicit independent repeat.
        torch.manual_seed(5)
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 1, 3, 8)
        graph_json = {
            "tensors": [{"uid": 4, "dims": [1, 4, 3, 8]}],
            "nodes": [
                {
                    "type": "SdpaAttributes",
                    "inputs": {
                        "q_tensor_uid": 1,
                        "k_tensor_uid": 2,
                        "v_tensor_uid": 3,
                    },
                    "outputs": {"o_tensor_uid": 4},
                }
            ],
        }
        tensors = {1: q, 2: k, 3: v}

        pytorch_ops.execute_graph(graph_json, tensors)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, dim=-3),
            v.repeat_interleave(4, dim=-3),
        )
        torch.testing.assert_close(tensors[4], expected, rtol=1e-5, atol=1e-5)
