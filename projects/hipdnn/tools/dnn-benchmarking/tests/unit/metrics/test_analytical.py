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

    def test_bandwidth_bound_returns_none(self):
        flops, partial = compute_flops(_bnorm_graph())
        # BN-only graph has no compute nodes counted; FLOPs is None and
        # partial stays False because no unrecognised types appeared.
        assert flops is None
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

    def test_empty_graph(self):
        flops, partial = compute_flops({"nodes": []})
        assert flops is None
        assert partial is False

    def test_mixed_compute_and_bandwidth_returns_compute_only(self):
        graph = _conv_graph()
        graph["nodes"].append(_bnorm_graph()["nodes"][0])
        flops, partial = compute_flops(graph)
        # BN is silently skipped (bandwidth-bound), conv is counted.
        assert flops == 2 * 16 * 16 * 3 * 3 * 16 * 16 * 16
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
            {
                "name": "bn",
                "type": "BatchnormInferenceAttributes",
                "inputs": {},
                "outputs": {},
            }
        )
        # ConvolutionFwdAttributes is supported, BN is bandwidth-bound
        # (intentionally skipped from the list), Mystery* appears once.
        assert list_unsupported_node_types(graph) == ["MysteryAttributes"]
