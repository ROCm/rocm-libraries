# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for metrics.analytical (FLOPs / IO bytes derivation).

Expected values are hand-computed using the MIOpen driver convention
(``conv_driver.hpp:1706-1710`` for conv, ``2*M*N*K`` for GEMM, FMA
counted as 2 FLOPs).
"""

from typing import Any, Dict

import pytest

from dnn_benchmarking.graph.tensor_info import TensorInfo
from dnn_benchmarking.metrics.analytical import (
    compute_flops,
    compute_io_bytes,
    derive_throughputs,
    list_unsupported_node_types,
)


def _conv_graph(
    n: int = 16,
    c: int = 16,
    h: int = 16,
    w: int = 16,
    k: int = 16,
    r: int = 3,
    s: int = 3,
    h_out: int = 16,
    w_out: int = 16,
    group_count: int = 1,
) -> Dict[str, Any]:
    """Build a minimal conv-fwd graph dict with explicit dims for test math."""
    return {
        "tensors": [
            {"uid": 1, "dims": [n, c, h, w], "data_type": "float", "virtual": False},
            {"uid": 2, "dims": [k, c, r, s], "data_type": "float", "virtual": False},
            {
                "uid": 3,
                "dims": [n, k, h_out, w_out],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "conv",
                "type": "ConvolutionFwdAttributes",
                "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                "outputs": {"y_tensor_uid": 3},
                "parameters": {"group_count": group_count},
            }
        ],
    }


def _matmul_graph(
    m: int = 256, n: int = 1024, k: int = 512, batch_dims=None
) -> Dict[str, Any]:
    if batch_dims:
        a_dims = list(batch_dims) + [m, k]
        b_dims = list(batch_dims) + [k, n]
        c_dims = list(batch_dims) + [m, n]
    else:
        a_dims, b_dims, c_dims = [m, k], [k, n], [m, n]
    return {
        "tensors": [
            {"uid": 1, "dims": a_dims, "data_type": "float", "virtual": False},
            {"uid": 2, "dims": b_dims, "data_type": "float", "virtual": False},
            {"uid": 3, "dims": c_dims, "data_type": "float", "virtual": False},
        ],
        "nodes": [
            {
                "name": "mm",
                "type": "MatmulAttributes",
                "inputs": {"a_tensor_uid": 1, "b_tensor_uid": 2},
                "outputs": {"c_tensor_uid": 3},
            }
        ],
    }


def _bnorm_graph() -> Dict[str, Any]:
    return {
        "tensors": [
            {
                "uid": 1,
                "dims": [32, 64, 28, 28],
                "data_type": "float",
                "virtual": False,
            },
            {
                "uid": 2,
                "dims": [32, 64, 28, 28],
                "data_type": "float",
                "virtual": False,
            },
        ],
        "nodes": [
            {
                "name": "bn",
                "type": "BatchnormInferenceAttributes",
                "inputs": {"x_tensor_uid": 1},
                "outputs": {"y_tensor_uid": 2},
            }
        ],
    }


def _relu_graph() -> Dict[str, Any]:
    return {
        "tensors": [
            {"uid": 1, "dims": [4, 8, 16, 16], "data_type": "float", "virtual": False},
            {"uid": 2, "dims": [4, 8, 16, 16], "data_type": "float", "virtual": False},
        ],
        "nodes": [
            {
                "name": "relu",
                "type": "PointwiseAttributes",
                "inputs": {"operation": "relu_fwd", "in_0_tensor_uid": 1},
                "outputs": {"out_0_tensor_uid": 2},
            }
        ],
    }


class TestComputeFlops:
    def test_conv_fwd_matches_miopen_formula(self):
        # 2 * N * C * R * S * K * H_out * W_out / group
        # = 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16 / 1
        graph = _conv_graph()
        flops, partial = compute_flops(graph)
        assert flops == 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16
        assert partial is False

    def test_conv_fwd_with_groups(self):
        graph = _conv_graph(group_count=4)
        flops, partial = compute_flops(graph)
        assert flops == (2 * 16 * 16 * 3 * 3 * 16 * 16 * 16) // 4
        assert partial is False

    def test_conv_fwd_with_zero_output_uid(self):
        # Regression: sample_conv_fwd.json uses uid=0 for the output
        # tensor. An ``or`` chain on get() would treat 0 as falsy and
        # mask the real UID, returning None for FLOPs.
        graph = {
            "tensors": [
                {
                    "uid": 0,
                    "dims": [16, 16, 16, 16],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 1,
                    "dims": [16, 16, 16, 16],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "dims": [16, 16, 3, 3],
                    "data_type": "float",
                    "virtual": False,
                },
            ],
            "nodes": [
                {
                    "name": "conv",
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {"group_count": 1},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops == 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16
        assert partial is False

    def test_matmul_2d(self):
        flops, partial = compute_flops(_matmul_graph(m=256, n=1024, k=512))
        assert flops == 2 * 256 * 1024 * 512
        assert partial is False

    def test_matmul_batched(self):
        flops, partial = compute_flops(
            _matmul_graph(m=128, n=64, k=32, batch_dims=[8, 4])
        )
        assert flops == 2 * 8 * 4 * 128 * 64 * 32
        assert partial is False

    def test_pointwise_one_flop_per_element(self):
        flops, partial = compute_flops(_relu_graph())
        assert flops == 4 * 8 * 16 * 16
        assert partial is False

    def test_batchnorm_inference_4_flops_per_element(self):
        # 32 * 64 * 28 * 28 elements × 4 ops/elem.
        flops, partial = compute_flops(_bnorm_graph())
        assert flops == 4 * 32 * 64 * 28 * 28
        assert partial is False

    def test_batchnorm_attributes_treated_as_fwd_training(self):
        # Real MIOpen workloads use BatchnormAttributes (without
        # Fwd/Bwd suffix) for forward training. 8 ops/elem on the
        # y_tensor output.
        graph = _bnorm_graph()
        graph["nodes"][0]["type"] = "BatchnormAttributes"
        flops, partial = compute_flops(graph)
        assert flops == 8 * 32 * 64 * 28 * 28
        assert partial is False

    def test_batchnorm_backward_uses_dx_for_element_count(self):
        # Real MIOpen wgrad emits BatchnormBackwardAttributes whose
        # output is dx_tensor_uid (no y_tensor_uid). 8 ops/elem on dx.
        graph = {
            "tensors": [
                {"uid": 1, "dims": [16, 64, 28, 28], "data_type": "float"},
                {"uid": 2, "dims": [16, 64, 28, 28], "data_type": "float"},
            ],
            "nodes": [
                {
                    "name": "bn_bwd",
                    "type": "BatchnormBackwardAttributes",
                    "inputs": {"dy_tensor_uid": 1, "x_tensor_uid": 1},
                    "outputs": {
                        "dx_tensor_uid": 2,
                        "dscale_tensor_uid": 1,
                        "dbias_tensor_uid": 1,
                    },
                }
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops == 8 * 16 * 64 * 28 * 28
        assert partial is False

    def test_conv_bwd_attributes_uses_dx_for_input_shape_dy_for_output_spatial(self):
        # ConvolutionBwdAttributes (dgrad): dx = conv_transposed(dy, w).
        # Input gradient dx has shape [N, C_in, H_in, W_in], dy has
        # [N, K, H_out, W_out]. FLOPs equal forward conv.
        n, c, h_in, w_in = 8, 16, 32, 32
        k, h_out, w_out = 32, 30, 30
        r, s = 3, 3
        graph = {
            "tensors": [
                {"uid": 1, "dims": [n, k, h_out, w_out], "data_type": "float"},  # dy
                {"uid": 2, "dims": [k, c, r, s], "data_type": "float"},  # w
                {"uid": 3, "dims": [n, c, h_in, w_in], "data_type": "float"},  # dx
            ],
            "nodes": [
                {
                    "name": "dgrad",
                    "type": "ConvolutionBwdAttributes",
                    "inputs": {"dy_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"dx_tensor_uid": 3},
                    "parameters": {"group_count": 1},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        # 2 * N * C_in * R * S * K * H_out * W_out
        assert flops == 2 * n * c * r * s * k * h_out * w_out
        assert partial is False

    def test_conv_wrw_attributes_uses_x_dy_for_shape(self):
        # ConvolutionWrwAttributes (wgrad): dw = conv(x, dy). Output
        # dw has the weight shape; FLOPs equal forward conv.
        n, c, h_in, w_in = 8, 16, 32, 32
        k, h_out, w_out = 32, 30, 30
        r, s = 3, 3
        graph = {
            "tensors": [
                {"uid": 1, "dims": [n, c, h_in, w_in], "data_type": "float"},  # x
                {"uid": 2, "dims": [n, k, h_out, w_out], "data_type": "float"},  # dy
                {"uid": 3, "dims": [k, c, r, s], "data_type": "float"},  # dw
            ],
            "nodes": [
                {
                    "name": "wgrad",
                    "type": "ConvolutionWrwAttributes",
                    "inputs": {"x_tensor_uid": 1, "dy_tensor_uid": 2},
                    "outputs": {"dw_tensor_uid": 3},
                    "parameters": {"group_count": 1},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops == 2 * n * c * r * s * k * h_out * w_out
        assert partial is False

    def test_layernorm_8_flops_per_element(self):
        graph = {
            "tensors": [
                {
                    "uid": 1,
                    "dims": [8, 256, 1024],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "dims": [8, 256, 1024],
                    "data_type": "float",
                    "virtual": False,
                },
            ],
            "nodes": [
                {
                    "name": "ln",
                    "type": "LayernormAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops == 8 * 8 * 256 * 1024
        assert partial is False

    def test_softmax_4_flops_per_element(self):
        graph = {
            "tensors": [
                {"uid": 1, "dims": [4, 1024], "data_type": "float", "virtual": False},
                {"uid": 2, "dims": [4, 1024], "data_type": "float", "virtual": False},
            ],
            "nodes": [
                {
                    "name": "sm",
                    "type": "SoftmaxAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops == 4 * 4 * 1024
        assert partial is False

    def test_reduction_one_flop_per_input_element(self):
        graph = {
            "tensors": [
                {"uid": 1, "dims": [8, 1024], "data_type": "float", "virtual": False},
                {"uid": 2, "dims": [8, 1], "data_type": "float", "virtual": False},
            ],
            "nodes": [
                {
                    "name": "red",
                    "type": "ReductionAttributes",
                    "inputs": {"x_tensor_uid": 1},
                    "outputs": {"y_tensor_uid": 2},
                }
            ],
        }
        flops, partial = compute_flops(graph)
        # Reduction work scales with the input size, not the (smaller) output.
        assert flops == 8 * 1024
        assert partial is False

    def test_unknown_node_type_marks_partial(self):
        graph = _conv_graph()
        graph["nodes"].append(
            {
                "name": "unknown",
                "type": "MysteryAttributes",
                "inputs": {},
                "outputs": {},
            }
        )
        flops, partial = compute_flops(graph)
        # Conv is still counted; the unknown node only flips partial.
        assert flops == 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16
        assert partial is True

    def test_all_unknown_nodes_returns_none_not_zero(self):
        # A graph whose nodes are all unmodellable must report unknown
        # (None), not a misleading 0 FLOPs that reads as a real count.
        graph = {
            "tensors": [{"uid": 1, "dims": [2, 2]}],
            "nodes": [
                {"name": "m", "type": "MysteryAttributes", "inputs": {}, "outputs": {}}
            ],
        }
        flops, partial = compute_flops(graph)
        assert flops is None
        assert partial is True

    def test_empty_graph(self):
        flops, partial = compute_flops({"nodes": []})
        assert flops is None
        assert partial is False

    def test_mixed_graph_sums_all_recognised_flops(self):
        # Build a combined graph manually so conv and BN don't share UIDs
        # — _conv_graph and _bnorm_graph each start from uid=1, so simply
        # concatenating their tensors would let BN tensors overwrite
        # conv tensors and yield bogus conv shapes.
        graph = {
            "tensors": [
                # Conv tensors (uid 0..2)
                {
                    "uid": 0,
                    "dims": [16, 16, 16, 16],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 1,
                    "dims": [16, 16, 16, 16],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 2,
                    "dims": [16, 16, 3, 3],
                    "data_type": "float",
                    "virtual": False,
                },
                # BN tensors (uid 10..11)
                {
                    "uid": 10,
                    "dims": [32, 64, 28, 28],
                    "data_type": "float",
                    "virtual": False,
                },
                {
                    "uid": 11,
                    "dims": [32, 64, 28, 28],
                    "data_type": "float",
                    "virtual": False,
                },
            ],
            "nodes": [
                {
                    "name": "conv",
                    "type": "ConvolutionFwdAttributes",
                    "inputs": {"x_tensor_uid": 1, "w_tensor_uid": 2},
                    "outputs": {"y_tensor_uid": 0},
                    "parameters": {"group_count": 1},
                },
                {
                    "name": "bn",
                    "type": "BatchnormInferenceAttributes",
                    "inputs": {"x_tensor_uid": 10},
                    "outputs": {"y_tensor_uid": 11},
                },
            ],
        }
        flops, partial = compute_flops(graph)
        expected_conv = 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16
        expected_bn = 4 * 32 * 64 * 28 * 28
        assert flops == expected_conv + expected_bn
        assert partial is False


class TestComputeIoBytes:
    def test_skips_virtual_tensors(self):
        infos = [
            TensorInfo(
                uid=1,
                name="x",
                dims=[16, 16],
                strides=[16, 1],
                data_type="float",
                is_virtual=False,
            ),
            TensorInfo(
                uid=2,
                name="v",
                dims=[16, 16],
                strides=[16, 1],
                data_type="float",
                is_virtual=True,
            ),
            TensorInfo(
                uid=3,
                name="y",
                dims=[16, 16],
                strides=[16, 1],
                data_type="float",
                is_virtual=False,
                is_output=True,
            ),
        ]
        # Only the two non-virtual tensors count: 16*16*4 each.
        assert compute_io_bytes(infos) == 2 * 16 * 16 * 4

    def test_dtype_size_respected(self):
        infos = [
            TensorInfo(
                uid=1,
                name="x",
                dims=[8, 8],
                strides=[8, 1],
                data_type="half",
                is_virtual=False,
            ),
        ]
        assert compute_io_bytes(infos) == 8 * 8 * 2

    def test_empty(self):
        assert compute_io_bytes([]) == 0


class TestDeriveThroughputs:
    def test_both_when_inputs_present(self):
        # 1e9 FLOPs in 1 ms = 1 TFLOPs/s; 1e6 bytes in 1 ms = 1 GB/s.
        tflops, gbytes = derive_throughputs(
            flops=10**9, io_bytes=10**6, kernel_mean_ms=1.0
        )
        assert tflops == pytest.approx(1.0)
        assert gbytes == pytest.approx(1.0)

    def test_none_kernel_time_returns_pair_of_none(self):
        assert derive_throughputs(10**9, 10**6, None) == (None, None)

    def test_zero_kernel_time_returns_pair_of_none(self):
        assert derive_throughputs(10**9, 10**6, 0.0) == (None, None)

    def test_missing_flops_returns_none_for_tflops_only(self):
        tflops, gbytes = derive_throughputs(None, 10**6, 1.0)
        assert tflops is None
        assert gbytes == pytest.approx(1.0)


class TestListUnsupportedNodeTypes:
    def test_lists_unique_unknowns_only(self):
        graph = _conv_graph()
        graph["nodes"].append(
            {"name": "x1", "type": "MysteryAttributes", "inputs": {}, "outputs": {}}
        )
        graph["nodes"].append(
            {"name": "x2", "type": "MysteryAttributes", "inputs": {}, "outputs": {}}
        )
        graph["nodes"].append(
            {"name": "y", "type": "OtherUnknownAttributes", "inputs": {}, "outputs": {}}
        )
        # Conv is supported; the two unknown types each appear once and
        # in first-seen order.
        assert list_unsupported_node_types(graph) == [
            "MysteryAttributes",
            "OtherUnknownAttributes",
        ]
