# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived features, as the runtime evaluates them, on hand-worked graphs."""
import math

import pandas as pd
import pytest

from uhd_gen.features import evaluate_feature_rows
from uhd_gen.generate import feature_recipe
from uhd_gen.physics import ABSENT, derived_features

_BN_INFERENCE_FLOPS = "graph.flops_by_type.BatchnormInferenceAttributes"


def _tensor(row: dict, role: str, dims: list[int]) -> None:
    for index, dim in enumerate(dims):
        row[f"graph.nodes[0].{role}.dims[{index}]"] = dim


def _conv(x_role, w_role, y_role, x, w, y, stride=1) -> dict:
    row = {
        "graph.nodes[0].stride[0]": stride,
        "graph.nodes[0].dilation[0]": 1,
        _BN_INFERENCE_FLOPS: 0,
    }
    for role, dims in ((x_role, x), (w_role, w), (y_role, y)):
        _tensor(row, role, dims)
    return row


def _frame() -> pd.DataFrame:
    batchnorm = {_BN_INFERENCE_FLOPS: 1e6}
    _tensor(batchnorm, "x", [4, 64, 28, 28])
    _tensor(batchnorm, "y", [4, 64, 28, 28])
    return pd.DataFrame(
        [
            # Forward, two groups of 32 input channels, 3x3 same.
            _conv("x", "w", "y", [2, 64, 56, 56], [128, 32, 3, 3], [2, 128, 56, 56]),
            # Backward-weights, depthwise 5x5 stride 2.
            _conv("x", "dw", "dy", [8, 32, 28, 28], [32, 1, 5, 5], [8, 32, 14, 14], 2),
            # Backward-data, 3-D 3x3x3.
            _conv(
                "dx",
                "w",
                "dy",
                [1, 16, 8, 16, 16],
                [32, 16, 3, 3, 3],
                [1, 32, 8, 16, 16],
            ),
            batchnorm,
        ]
    )


def test_derived_features_are_the_hand_worked_values(evaluator):
    """The implicit-GEMM view differs by direction: forward is (N*out, K/g, C/g*filter),
    backward-data (N*in, C/g, K/g*filter), backward-weights (K/g, C/g*filter, N*out). A
    feature of an operation the graph is not reads ABSENT rather than failing."""
    frame = _frame()
    features = derived_features(frame)
    _, rows = evaluate_feature_rows(frame, list(features.values()), None, evaluator)
    got = [dict(zip(features, row)) for row in rows]

    forward, weights, data, batchnorm = got
    assert {
        k: forward[k]
        for k in (
            "conv.gemm_m",
            "conv.gemm_n",
            "conv.gemm_k",
            "conv.groups",
            "conv.tiles_128",
        )
    } == {
        "conv.gemm_m": 2 * 56 * 56,
        "conv.gemm_n": 64,
        "conv.gemm_k": 32 * 9,
        "conv.groups": 2,
        "conv.tiles_128": 2 * 49 * 1,
    }
    assert forward["conv.cin_alignment"] == 5 and forward["conv.cout_alignment"] == 6
    assert forward["conv.3x3_unit_stride"] == 1 and forward["conv.depthwise"] == 0
    assert math.isclose(forward["conv.gemm_mn_over_k"], 6272 * 64 / 288)

    assert (weights["conv.gemm_m"], weights["conv.gemm_n"], weights["conv.gemm_k"]) == (
        1,
        25,
        8 * 14 * 14,
    )
    assert weights["conv.depthwise"] == 1 and weights["conv.groups"] == 32
    assert weights["conv.3x3_unit_stride"] == 0

    assert (data["conv.gemm_m"], data["conv.gemm_n"], data["conv.gemm_k"]) == (
        8 * 16 * 16,
        16,
        32 * 27,
    )
    assert data["conv.filter_elems"] == 27 and data["conv.in_spatial"] == 2048

    assert batchnorm["batchnorm.channels"] == 64
    assert batchnorm["batchnorm.reduction_length"] == 4 * 28 * 28
    assert all(batchnorm[k] == ABSENT for k in features if k.startswith("conv."))
    assert all(
        row[k] == ABSENT
        for row in (forward, weights, data)
        for k in features
        if k.startswith("batchnorm.")
    )


@pytest.mark.parametrize(
    "rows, families",
    [
        (slice(0, 3), {"conv"}),
        (slice(3, 4), {"batchnorm"}),
        (slice(0, 4), {"conv", "batchnorm"}),
    ],
)
def test_a_proposed_recipe_adds_the_families_its_corpus_holds(rows, families):
    frame = _frame().iloc[rows].reset_index(drop=True)
    frame["graph.flops"] = 1e9
    signature, _ = feature_recipe(frame, set(frame.columns), None, None, [], [])
    added = {
        name.split(".")[0]
        for name, expression in derived_features(_frame()).items()
        if expression in signature
    }
    assert added == families


def test_an_authored_recipe_gets_no_derived_features():
    frame = _frame()
    authored = ["$graph.nodes[0].x.dims[0]"]
    signature, _ = feature_recipe(frame, set(frame.columns), authored, None, [], [])
    assert signature == authored
