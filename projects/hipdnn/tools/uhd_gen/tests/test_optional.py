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


def test_automatic_selection_keeps_an_optional_column_and_drops_any_other_gap(
    evaluator,
):
    frame = pd.DataFrame(
        {
            SCALE[1:]: [0.125, None, 0.0625, None, 0.125, 0.25],
            Q_DIM[1:]: [64, None, 128, 64, None, 32],
            "graph.nodes[0].q.dims[0]": [1, 2, 4, 8, 16, 32],
        }
    )
    signature, _ = feature_recipe(
        frame, set(frame.columns), None, None, [], [], optional_features(evaluator)
    )
    assert SCALE in signature
    assert Q_DIM not in signature
    assert "$graph.nodes[0].q.dims[0]" in signature


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
