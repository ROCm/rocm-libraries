# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Models read every feature a valid graph may leave unbound through value_or_default."""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import pytest

pytest.importorskip("flatbuffers")
pd = pytest.importorskip("pandas")

from uhd_gen.features import compute_features_hash, evaluate_feature_maps
from uhd_gen.generate import feature_recipe
from uhd_gen.optional import (
    ATTRIBUTE_DEFAULTS,
    SENTINEL,
    lint,
    missing_defaults,
    optional_features,
    safe_signature,
)

SCALE = "$graph.nodes[0].attn_scale_value"
Q_DIM = "$graph.nodes[0].q.dims[3]"
FLOPS = "$graph.flops"

pytestmark = pytest.mark.usefixtures("evaluator")


def test_every_optional_attribute_hipdnn_lists_has_a_default(evaluator):
    """A field a schema makes optional fails here until someone states what a model reads
    when a graph leaves it unset."""
    features = optional_features(evaluator)
    assert "graph.nodes[*].attn_scale_value" in {
        feature["pattern"] for feature in features
    }
    assert missing_defaults(features) == []


def test_a_safe_signature_changes_nothing_bound_and_reads_defaults_for_the_absent(
    evaluator,
):
    signature, wrapped = safe_signature([SCALE, Q_DIM, FLOPS], "time", evaluator)
    assert [item["reference"] for item in wrapped] == [FLOPS, SCALE]
    assert signature[1] == Q_DIM
    assert lint(signature, "time", evaluator) == []

    bound = {SCALE[1:]: 0.125, Q_DIM[1:]: 64, FLOPS[1:]: 3.0e9}
    _, bare = evaluate_feature_maps([bound], [SCALE, Q_DIM, FLOPS], None, evaluator)
    _, safe = evaluate_feature_maps([bound], signature, None, evaluator)
    assert safe == bare
    _, absent = evaluate_feature_maps(
        [{Q_DIM[1:]: 64, SCALE[1:]: None, FLOPS[1:]: None}], signature, None, evaluator
    )
    assert absent == [[1.0, 64, SENTINEL]]
    with pytest.raises(ValueError, match="Undefined variable"):
        evaluate_feature_maps([{Q_DIM[1:]: 64}], [SCALE, Q_DIM], None, evaluator)


def test_unknown_work_stays_bare_in_a_model_of_a_metric_derived_from_it(evaluator):
    """Where the work is unknown so is throughput: a tflops model rightly reports nothing."""
    signature, wrapped = safe_signature([FLOPS, SCALE], "tflops", evaluator)
    assert signature[0] == FLOPS
    assert [item["reference"] for item in wrapped] == [SCALE]


def test_a_gdn_scale_defaults_to_rsqrt_of_its_own_nodes_key_width(evaluator):
    scale = "$graph.nodes[1].scale"
    signature, _ = safe_signature([scale], "time", evaluator)
    _, values = evaluate_feature_maps(
        [{"graph.nodes[1].q.dims[2]": 64, "graph.nodes[0].q.dims[2]": 16}],
        signature,
        None,
        evaluator,
    )
    assert values == [[pytest.approx(1 / math.sqrt(64))]]


# What the API means by an unset optional attribute (or a vector element past the end),
# written from the schemas and the frontend rather than read back from ATTRIBUTE_DEFAULTS.
# A boolean reads as 0.0 (false) or 1.0 (true); -1 is the sentinel where unset has no number.
UNSET_MEANINGS = {
    "graph.nodes[*].attn_scale_value": 1.0,  # sdpa_attributes.fbs: a unit scale
    "graph.nodes[*].dropout_probability": 0.0,  # no dropout
    "graph.nodes[*].generate_stats": 0.0,  # false
    "graph.nodes[*].left_bound": -1.0,  # unbounded
    "graph.nodes[*].right_bound": -1.0,  # unbounded
    "graph.nodes[*].max_seq_len_kv": -1.0,  # not declared
    "graph.nodes[*].relu_lower_clip": 0.0,  # cuDNN pointwise defaults
    "graph.nodes[*].relu_upper_clip": -1.0,  # no upper bound
    "graph.nodes[*].relu_lower_clip_slope": 0.0,
    "graph.nodes[*].swish_beta": 1.0,
    "graph.nodes[*].elu_alpha": 1.0,
    "graph.nodes[*].softplus_beta": 1.0,
    "graph.nodes[*].axis_tensor_uid": -1.0,  # no axis
    "graph.nodes[*].axis": -1.0,  # no axis
    "graph.nodes[*].generate_index": 0.0,  # false
    "graph.nodes[*].scale": 1 / math.sqrt(64),  # 1/sqrt(K) with K the node's q.dims[2]
    "graph.nodes[*].gate_lower_bound": -5.0,  # kda_attributes.fbs: "absent: -5.0"
    "graph.nodes[*].pre_padding[*]": 0.0,  # a dimension the node does not have:
    "graph.nodes[*].post_padding[*]": 0.0,  # neither pads,
    "graph.nodes[*].stride[*]": 1.0,  # strides,
    "graph.nodes[*].dilation[*]": 1.0,  # dilates,
    "graph.nodes[*].window[*]": 1.0,  # windows
    "graph.nodes[*].block_size[*]": 1.0,  # nor blocks
}


@pytest.mark.parametrize("pattern", sorted(ATTRIBUTE_DEFAULTS))
def test_an_absent_optional_attribute_reads_what_the_api_means_by_unset(
    evaluator, pattern
):
    reference = "$" + pattern.replace("[*]", "[3]", 1).replace("[*]", "[2]")
    signature, wrapped = safe_signature([reference], "time", evaluator)
    assert [item["reference"] for item in wrapped] == [reference]
    # The GDN/KDA default reads the node's own key width, not another node's.
    absent = {
        reference[1:]: None,
        "graph.nodes[3].q.dims[2]": 64,
        "graph.nodes[0].q.dims[2]": 16,
    }
    _, values = evaluate_feature_maps([absent], signature, None, evaluator)
    assert values == [[pytest.approx(UNSET_MEANINGS[pattern])]]


_DEFAULTED_SCALE = {"value_or_default": [SCALE, 1.0]}
_GUARDED_SCALE = {"value_or_default": [SCALE, 0.5]}


@pytest.mark.parametrize(
    "signature,expected,wrapped_references,absent,bound",
    [
        pytest.param(
            [{"*": [SCALE, 2]}],
            [{"*": [_DEFAULTED_SCALE, 2]}],
            [SCALE],
            [2.0],
            [0.25],
            id="nested_operator",
        ),
        pytest.param(
            [_GUARDED_SCALE],
            [_GUARDED_SCALE],
            [],
            [0.5],
            [0.125],
            id="pre_guarded_read_is_left_alone",
        ),
        pytest.param(
            [_GUARDED_SCALE, SCALE],
            [_GUARDED_SCALE, _DEFAULTED_SCALE],
            [SCALE],
            [0.5, 1.0],
            [0.125, 0.125],
            id="pre_guarded_read_keeps_its_own_default",
        ),
        pytest.param(
            [{"if": [{"present": [SCALE]}, SCALE, -1]}, SCALE],
            [{"if": [{"present": [SCALE]}, _DEFAULTED_SCALE, -1]}, _DEFAULTED_SCALE],
            [SCALE],
            [-1.0, 1.0],
            [0.125, 0.125],
            id="present_condition_still_sees_the_absence",
        ),
        pytest.param(
            [{"if": [{"not_present": [SCALE]}, -1, SCALE]}, SCALE],
            [
                {"if": [{"not_present": [SCALE]}, -1, _DEFAULTED_SCALE]},
                _DEFAULTED_SCALE,
            ],
            [SCALE],
            [-1.0, 1.0],
            [0.125, 0.125],
            id="not_present_condition_still_sees_the_absence",
        ),
        pytest.param(
            [{"if": [{">": [SCALE, 0]}, 1, 2]}],
            [{"if": [{">": [_DEFAULTED_SCALE, 0]}, 1, 2]}],
            [SCALE],
            [1.0],
            [1.0],
            id="comparison_condition_reads_the_default",
        ),
    ],
)
def test_a_safe_signature_wraps_reads_inside_expressions_and_keeps_presence_conditions(
    evaluator, signature, expected, wrapped_references, absent, bound
):
    rewritten, wrapped = safe_signature(signature, "time", evaluator)
    assert rewritten == expected
    assert [item["reference"] for item in wrapped] == wrapped_references
    _, absent_values = evaluate_feature_maps(
        [{SCALE[1:]: None}], rewritten, None, evaluator
    )
    assert absent_values == [absent]
    _, bound_values = evaluate_feature_maps(
        [{SCALE[1:]: 0.125}], rewritten, None, evaluator
    )
    assert bound_values == [bound]


@pytest.mark.parametrize(
    "column,values,kept",
    [
        pytest.param(
            SCALE[1:], [0.125, None, 0.0625, None, 0.125, 0.25], True, id="float"
        ),
        pytest.param(
            "graph.nodes[0].generate_stats",
            [True, None, False, True, None, False],
            True,
            id="boolean_object_column",
        ),
        pytest.param(
            "graph.nodes[0].dropout_probability", [None] * 6, False, id="all_absent"
        ),
        pytest.param(
            Q_DIM[1:], [64, None, 128, 64, None, 32], False, id="not_optional"
        ),
    ],
)
def test_automatic_selection_keeps_an_optional_column_and_drops_any_other_gap(
    evaluator, column, values, kept
):
    frame = pd.DataFrame(
        {column: values, "graph.nodes[0].q.dims[0]": [1, 2, 4, 8, 16, 32]}
    )
    signature, _ = feature_recipe(
        frame, set(frame.columns), None, None, [], [], optional_features(evaluator)
    )
    assert ("$" + column in signature) is kept
    assert "$graph.nodes[0].q.dims[0]" in signature


def test_generate_proposes_an_optional_column_some_rows_leave_unbound(evaluator):
    from uhd_gen.generate import _feature_signature

    frame = pd.DataFrame(
        {
            SCALE[1:]: [0.125, None, 0.0625, None, 0.125, 0.25],
            Q_DIM[1:]: [64, None, 128, 64, None, 32],
            "graph.nodes[0].q.dims[0]": [1, 2, 4, 8, 16, 32],
        }
    )
    args = argparse.Namespace(
        dim_tile=[], feature_signature=None, features=None, feature_evaluator=evaluator
    )
    signature, _, _ = _feature_signature(
        args, frame, set(frame.columns), None, [], False
    )
    assert signature == [SCALE, "$graph.nodes[0].q.dims[0]"]


def _installed_model(tmp_path, signature, metric="tflops"):
    from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model

    directory = tmp_path / "sort_kernel_catalog" / "gfx942" / metric
    directory.mkdir(parents=True)
    features_hash = compute_features_hash(signature)
    split = {
        "split_feature": 0,
        "threshold": 0.1,
        "decision_type": "<=",
        "default_left": True,
        "missing_type": "None",
        "left_child": {"leaf_value": 1.0},
        "right_child": {"leaf_value": 2.0},
    }
    data = build_gbdt_model(
        {
            "tree_info": [{"tree_structure": split}],
            "max_feature_idx": len(signature) - 1,
            "objective": "regression",
        },
        features_hash,
    )
    (directory / "model.bin").write_bytes(data)
    descriptor = {
        "version": "1.0",
        "id": "0b8f5f3e-8f0c-4d6c-9d2e-3a3c1b0f6a11",
        "name": "test model",
        "adapter": "tree_data",
        "features_signature": signature,
        "features_hash": features_hash,
        "objective": "max",
        "score": {"metric": metric, "calibrated": True, "transform": "log"},
        "tree_data": {
            "artifact": "model.bin",
            "hash": hashlib.sha256(data).hexdigest(),
        },
    }
    path = directory / "heuristic.uhd.json"
    path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    return path


def _rewrite_args(paths, **overrides):
    from uhd_gen.rewrite import add_rewrite_arguments

    parser = argparse.ArgumentParser()
    add_rewrite_arguments(parser)
    args = parser.parse_args([str(path) for path in paths])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_rewrite_keeps_the_trees_and_restates_every_hash(tmp_path, evaluator):
    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    from uhd_gen.promote import _verify_artifact
    from uhd_gen.rewrite import _same, rewrite_descriptor

    path = _installed_model(tmp_path, [SCALE, Q_DIM, FLOPS])
    artifact = path.parent / "model.bin"
    before = GbdtModelT.InitFromPackedBuf(bytearray(artifact.read_bytes()), 0)

    report = rewrite_descriptor(path, _rewrite_args([path]))

    assert report["status"] == "rewritten"
    assert report["proof"]["features_identical"]
    assert report["proof"]["absent_now_evaluable"] == [SCALE]
    # A bare scale fails every row without it; the rewrite scores them all.
    assert report["proof"]["old_refused_rows_when_absent"] == {
        SCALE: report["proof"]["rows_compared"]
    }
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    assert descriptor["features_signature"] == [
        {"value_or_default": [SCALE, 1.0]},
        Q_DIM,
        FLOPS,
    ]
    assert descriptor["features_hash"] == compute_features_hash(
        descriptor["features_signature"]
    )
    # The loader's checks: the embedded hash and digest agree with the descriptor.
    assert _verify_artifact(descriptor, artifact, "sort_kernel_catalog") == (
        descriptor["tree_data"]["hash"]
    )
    after = GbdtModelT.InitFromPackedBuf(bytearray(artifact.read_bytes()), 0)
    assert after.featuresHash.decode() == descriptor["features_hash"]
    before.featuresHash = after.featuresHash = None
    assert _same(before, after)

    files = (path.read_bytes(), artifact.read_bytes())
    assert rewrite_descriptor(path, _rewrite_args([path]))["status"] == "already_safe"
    assert (path.read_bytes(), artifact.read_bytes()) == files


def test_a_dry_run_proves_without_writing(tmp_path, evaluator):
    from uhd_gen.rewrite import rewrite_descriptor

    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    files = (path.read_bytes(), (path.parent / "model.bin").read_bytes())
    report = rewrite_descriptor(path, _rewrite_args([path], dry_run=True))
    assert report["status"] == "rewritten"
    assert (path.read_bytes(), (path.parent / "model.bin").read_bytes()) == files


def test_a_bare_read_that_short_circuits_is_rewritten_and_reported_as_tolerated(
    tmp_path, evaluator
):
    """The scale sits behind a conjunct that is false on every positive dimension, so the old
    signature scores rows without it: the rewrite proves each one scores, and reports that
    the old signature refused none, rather than refusing the model."""
    from uhd_gen.rewrite import rewrite_descriptor

    guarded = {"and": [{"<": [Q_DIM, 0]}, {">": [SCALE, 0]}]}
    path = _installed_model(tmp_path, [guarded, Q_DIM])
    report = rewrite_descriptor(path, _rewrite_args([path]))

    assert report["status"] == "rewritten"
    assert report["proof"]["rows_compared"] > 0
    assert report["proof"]["old_refused_rows_when_absent"] == {SCALE: 0}
    rewritten = json.loads(path.read_text(encoding="utf-8"))["features_signature"]
    assert rewritten == [
        {"and": [{"<": [Q_DIM, 0]}, {">": [{"value_or_default": [SCALE, 1.0]}, 0]}]},
        Q_DIM,
    ]
    for row in ({Q_DIM[1:]: 64, SCALE[1:]: 0.5}, {Q_DIM[1:]: -3, SCALE[1:]: 0.5}):
        _, old = evaluate_feature_maps([row], [guarded, Q_DIM], None, evaluator)
        _, new = evaluate_feature_maps([row], rewritten, None, evaluator)
        assert new == old
    # Every row without the scale scores, including one where the old signature reads it.
    for dimension, expected in ((64, [0.0, 64.0]), (-3, [1.0, -3.0])):
        _, values = evaluate_feature_maps(
            [{Q_DIM[1:]: dimension, SCALE[1:]: None}], rewritten, None, evaluator
        )
        assert values == [expected]
