# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived convolution features read a 2-D and a 3-D graph alike, through the runtime."""
import pytest

pd = pytest.importorskip("pandas")

from uhd_gen.features import evaluate_feature_rows, signature_references
from uhd_gen.physics import derived_features

NODE = "graph.nodes[0]"


def _tensor(role, dims):
    return {f"{NODE}.{role}.dims[{i}]": value for i, value in enumerate(dims)}


def _params(**arrays):
    return {
        f"{NODE}.{name}[{i}]": value
        for name, values in arrays.items()
        for i, value in enumerate(values)
    }


def _forward(spatial_in, spatial_out, filt, stride, n=2, c=8, k=16):
    rank = len(spatial_in)
    return {
        **_tensor("x", [n, c, *spatial_in]),
        **_tensor("w", [k, c, *filt]),
        **_tensor("y", [n, k, *spatial_out]),
        **_params(
            stride=stride,
            dilation=[1] * rank,
            pre_padding=[0] * rank,
            post_padding=[0] * rank,
        ),
    }


def _evaluate(frame, evaluator):
    features = derived_features(frame)
    _, rows = evaluate_feature_rows(
        frame, list(features.values()), executable=evaluator
    )
    return [dict(zip(features, row)) for row in rows]


def test_a_2d_and_a_3d_convolution_read_their_own_axes(evaluator):
    """dims[3] is W on a 2-D graph and H on a 3-D one; the derived axes are not."""
    frame = pd.DataFrame(
        [
            _forward([10, 12], [5, 6], [3, 3], [2, 2]),
            _forward([4, 10, 12], [2, 5, 6], [3, 3, 3], [2, 2, 2]),
        ]
    )
    flat, deep = _evaluate(frame, evaluator)

    assert (flat["conv.in_d"], flat["conv.in_h"], flat["conv.in_w"]) == (1, 10, 12)
    assert (deep["conv.in_d"], deep["conv.in_h"], deep["conv.in_w"]) == (4, 10, 12)
    assert (flat["conv.stride_d"], flat["conv.stride_h"], flat["conv.stride_w"]) == (
        1,
        2,
        2,
    )
    assert (deep["conv.stride_d"], deep["conv.stride_h"], deep["conv.stride_w"]) == (
        2,
        2,
        2,
    )
    assert (flat["conv.spatial_rank"], deep["conv.spatial_rank"]) == (2, 3)
    # Forward implicit GEMM: M = N * output spatial, N = K / groups, K = C/groups * filter.
    assert (flat["conv.gemm_m"], flat["conv.gemm_n"], flat["conv.gemm_k"]) == (
        60,
        16,
        72,
    )
    assert (deep["conv.gemm_m"], deep["conv.gemm_n"], deep["conv.gemm_k"]) == (
        120,
        16,
        216,
    )
    assert flat["conv.3x3_unit_stride"] == deep["conv.3x3_unit_stride"] == 0


def test_backward_weights_reads_only_names_its_graphs_publish(evaluator):
    """An engine serving one direction never publishes another direction's tensors, so a
    reference to one would make the engine's recipe unpublishable."""
    row = {
        **_tensor("x", [2, 8, 10, 12]),
        **_tensor("dw", [16, 8, 3, 3]),
        **_tensor("dy", [2, 16, 10, 12]),
        **_params(
            stride=[1, 1], dilation=[1, 1], pre_padding=[1, 1], post_padding=[1, 1]
        ),
    }
    frame = pd.DataFrame([row])
    referenced = {
        ref[1:] for ref in signature_references(list(derived_features(frame).values()))
    }
    assert referenced <= set(frame.columns)

    (values,) = _evaluate(frame, evaluator)
    # Backward weights: M = K / groups, N = C/groups * filter, K = N * output spatial.
    assert (values["conv.gemm_m"], values["conv.gemm_n"], values["conv.gemm_k"]) == (
        16,
        72,
        240,
    )
    assert values["conv.3x3_unit_stride"] == 1
    # A 2-D-only corpus has no depth to learn from.
    assert "conv.in_d" not in values


def test_a_graph_of_another_operation_reads_absent(evaluator):
    conv = _forward([10, 12], [10, 12], [1, 1], [1, 1])
    other = {"graph.flops": 1e9, f"{NODE}.x.dims[0]": 4}
    rows = _evaluate(pd.DataFrame([conv, other]), evaluator)
    assert rows[0]["conv.pointwise"] == 1
    assert set(rows[1].values()) == {-1}


def test_no_convolution_no_derived_features():
    frame = pd.DataFrame([{f"{NODE}.q.dims[0]": 1, "graph.flops": 1e9}])
    assert derived_features(frame) == {}
