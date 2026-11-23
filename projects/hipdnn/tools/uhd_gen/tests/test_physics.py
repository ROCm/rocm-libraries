# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derived features, as the runtime evaluates them, on hand-worked graphs."""
import argparse
import math

import pandas as pd
import pytest

from uhd_gen import physics
from uhd_gen.features import evaluate_feature_rows, signature_references
from uhd_gen.generate import _feature_signature, feature_recipe
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


def _attention() -> dict:
    """Grouped-query: 8 query heads over 2 KV heads, Sq 1000 against Skv 4096, d 128. The
    FLOPs are half the unmasked count, as a causal-style mask would leave."""
    row = {_BN_INFERENCE_FLOPS: 0}
    _tensor(row, "q", [2, 8, 1000, 128])
    _tensor(row, "k", [2, 2, 4096, 128])
    _tensor(row, "v", [2, 2, 4096, 128])
    _tensor(row, "o", [2, 8, 1000, 128])
    row["graph.nodes[0].flops"] = 2 * 16 * 1000 * 4096 * 256 / 2
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
            _attention(),
        ]
    )


def test_derived_features_are_the_hand_worked_values(evaluator):
    """A convolution's implicit-GEMM view differs by direction: forward is (N*out, K/g, C/g*filter),
    backward-data (N*in, C/g, K/g*filter), backward-weights (K/g, C/g*filter, N*out). A
    feature of an operation the graph is not reads ABSENT rather than failing."""
    frame = _frame()
    features = derived_features(frame)
    _, rows = evaluate_feature_rows(frame, list(features.values()), None, evaluator)
    got = [dict(zip(features, row)) for row in rows]

    forward, weights, data, batchnorm, attention = got
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
    assert {k: v for k, v in attention.items() if k.startswith("sdpa.")} == {
        "sdpa.batch_heads": 16,
        "sdpa.kv_group": 4,
        "sdpa.score_pairs": 16 * 1000 * 4096,
        "sdpa.kv_over_q": 4.096,
        "sdpa.q_tiles_64": 16 * 16,
        "sdpa.q_tiles_128": 16 * 8,
        "sdpa.kv_steps_128": 32,
        "sdpa.q_fill_128": 1000 / 1024,
        "sdpa.kv_fill_128": 1,
        "sdpa.work_kept": 0.5,
    }
    families = {
        "conv": (forward, weights, data),
        "batchnorm": (batchnorm,),
        "sdpa": (attention,),
    }
    for family, rows in families.items():
        others = [
            row for name, group in families.items() if name != family for row in group
        ]
        assert all(
            row[k] == ABSENT
            for row in others
            for k in features
            if k.startswith(family + ".")
        ), family


@pytest.mark.parametrize(
    "rows, families",
    [
        (slice(0, 3), {"conv"}),
        (slice(3, 4), {"batchnorm"}),
        (slice(4, 5), {"sdpa"}),
        (slice(0, 5), {"conv", "batchnorm", "sdpa"}),
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


def _batchnorm_row(x: list[int]) -> dict:
    row = {_BN_INFERENCE_FLOPS: 1e6}
    _tensor(row, "x", x)
    _tensor(row, "y", x)
    return row


#: Corpora as one engine collects them: every graph of the operations and ranks it serves.
_CORPORA = {
    "conv_fwd_2d": [
        _conv("x", "w", "y", [2, 64, 56, 56], [128, 32, 3, 3], [2, 128, 56, 56]),
        _conv("x", "w", "y", [4, 16, 14, 14], [16, 16, 1, 1], [4, 16, 14, 14]),
    ],
    "conv_fwd_2d_and_3d": [
        _conv("x", "w", "y", [2, 64, 56, 56], [128, 32, 3, 3], [2, 128, 56, 56]),
        _conv("x", "w", "y", [1, 16, 8, 16, 16], [32, 16, 3, 3, 3], [1, 32, 8, 16, 16]),
    ],
    "conv_bwd_data_2d": [
        _conv("dx", "w", "dy", [2, 64, 28, 28], [128, 64, 3, 3], [2, 128, 28, 28]),
        _conv("dx", "w", "dy", [8, 32, 14, 14], [32, 1, 5, 5], [8, 32, 7, 7], 2),
    ],
    "batchnorm_2d": [_batchnorm_row([4, 64, 28, 28]), _batchnorm_row([8, 32, 14, 14])],
}


def _written(frame: pd.DataFrame) -> dict:
    """The families the frame holds, as written: before specialization."""
    return {
        name: expression
        for holds, family in physics._FAMILIES
        if holds(frame)
        for name, expression in family().items()
    }


@pytest.mark.parametrize("corpus", sorted(_CORPORA))
def test_a_derived_feature_reads_only_what_the_corpus_publishes_and_keeps_its_value(
    evaluator, corpus
):
    """A 2-D forward engine publishes no `dx` and no `dims[4]`; the families still serve
    it, read only through its own fields, with the values the written expressions give.
    """
    frame = pd.DataFrame(_CORPORA[corpus])
    written = _written(frame)
    specialized = derived_features(frame)
    assert set(specialized) == set(written)
    assert {ref[1:] for ref in signature_references(list(specialized.values()))} <= set(
        frame.columns
    )
    _, before = evaluate_feature_rows(frame, list(written.values()), None, evaluator)
    _, after = evaluate_feature_rows(
        frame, [specialized[name] for name in written], None, evaluator
    )
    assert after == before


def test_a_corpus_with_three_d_graphs_keeps_the_depth_and_a_two_d_one_reads_none(
    evaluator,
):
    def references(corpus):
        frame = pd.DataFrame(_CORPORA[corpus])
        features = derived_features(frame)
        return frame, features, signature_references(list(features.values()))

    _, _, flat = references("conv_fwd_2d")
    backward = (".dx.", ".dw.", ".dy.")
    assert [
        ref for ref in flat if "dims[4]" in ref or any(role in ref for role in backward)
    ] == []
    frame, features, mixed = references("conv_fwd_2d_and_3d")
    assert "$graph.nodes[0].x.dims[4]" in mixed
    _, rows = evaluate_feature_rows(
        frame, [features["conv.in_spatial"]], None, evaluator
    )
    assert [row[0] for row in rows] == [56 * 56, 8 * 16 * 16]


def test_generate_accepts_the_proposed_recipe_of_a_two_d_forward_convolution(evaluator):
    """generate refuses a recipe reading a field the engine does not publish; the derived
    convolution features of a 2-D forward corpus no longer read one."""
    frame = pd.DataFrame(_CORPORA["conv_fwd_2d"])
    args = argparse.Namespace(
        dim_tile=[], feature_signature=None, features=None, feature_evaluator=evaluator
    )
    signature, _, _ = _feature_signature(
        args, frame, set(frame.columns), None, [], False
    )
    for expression in derived_features(frame).values():
        assert expression in signature


_X = "$graph.nodes[0].x.dims[0]"
_GONE = "$graph.nodes[0].dx.dims[0]"


@pytest.mark.parametrize(
    "written, specialized",
    [
        ({"value_or_default": [_GONE, 7]}, 7),
        (
            {"value_or_default": [_X, {"value_or_default": [_GONE, 1]}]},
            {"value_or_default": [_X, 1]},
        ),
        ({"present": [_X, _GONE]}, False),
        ({"not_present": [_X, _GONE]}, {"not_present": [_X]}),
        ({"not_present": [_GONE]}, True),
        (
            {"if": [{"present": [_GONE]}, 1, {"present": [_X]}, _X, 0]},
            {"if": [{"present": [_X]}, _X, 0]},
        ),
        ({"if": [{"not_present": [_GONE]}, _X, 0]}, _X),
        ({"if": [{"present": [_X]}, _X]}, {"if": [{"present": [_X]}, _X]}),
        ({"if": [{"present": [_GONE]}, _X]}, physics._MISSING),
        (
            {"if": [{"present": [_X]}, {"value_or_default": [_GONE, ABSENT]}, ABSENT]},
            ABSENT,
        ),
        (
            {"if": [{">": [_X, 1]}, ABSENT, ABSENT]},
            {"if": [{">": [_X, 1]}, ABSENT, ABSENT]},
        ),
        ({"or": [{"present": [_X]}, {"present": [_GONE]}]}, {"present": [_X]}),
        ({"and": [{"present": [_X]}, {"present": [_GONE]}]}, False),
        ({"*": [_X, _GONE]}, physics._MISSING),
        ({"value_or_default": [_X, _GONE]}, physics._MISSING),
    ],
)
def test_specialization_drops_only_what_no_published_graph_reaches(
    written, specialized
):
    published = {_X[1:]}
    assert physics._specialize(written, published) == specialized


def test_a_feature_that_specializes_to_a_constant_or_an_unstatable_value_is_left_out():
    """Attention's share of unmasked work reads the node's declared FLOPs; without them it is
    the constant ABSENT on every row, which no model can split on."""
    frame = pd.DataFrame([_attention()]).drop(columns=["graph.nodes[0].flops"])
    assert "sdpa.work_kept" not in derived_features(frame)
    assert "sdpa.work_kept" in derived_features(pd.DataFrame([_attention()]))
