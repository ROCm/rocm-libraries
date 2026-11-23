# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Models read every feature a valid graph may leave unbound through value_or_default."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("flatbuffers")
pd = pytest.importorskip("pandas")

from uhd_gen.features import compute_features_hash, evaluate_feature_maps, run_evaluator
from uhd_gen.generate import feature_recipe
from uhd_gen.optional import (
    ATTRIBUTE_DEFAULTS,
    SENTINEL,
    lint,
    missing_defaults,
    optional_features,
    safe_signature,
)
from uhd_gen.tests.model_bin import leaf_model

SCALE = "$graph.nodes[0].attn_scale_value"
Q_DIM = "$graph.nodes[0].q.dims[3]"
K_DIM = "$graph.nodes[0].k.dims[3]"
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


# The rank the absent-attribute rows bind for their node's input, from which an unset
# BlockScaleQuantize axis (the last one) is read.
X_RANK = 4

# What the API means by an unset optional attribute (or a vector element past the end),
# written from the schemas and the frontend rather than read back from ATTRIBUTE_DEFAULTS.
# A boolean reads as 0.0 (false) or 1.0 (true). Where unset has no number, a marker stands in:
# -1 for an unset uid or sequence length.
UNSET_MEANINGS = {
    "graph.nodes[*].attn_scale_value": 1.0,  # SdpaAttributes.hpp: 1.0 with no value or tensor
    "graph.nodes[*].dropout_probability": 0.0,  # no dropout
    "graph.nodes[*].generate_stats": 0.0,  # false
    "graph.nodes[*].left_bound": -1.0,  # unbounded
    "graph.nodes[*].right_bound": -1.0,  # unbounded
    "graph.nodes[*].max_seq_len_kv": -1.0,  # not declared
    "graph.nodes[*].relu_lower_clip": 0.0,  # cuDNN pointwise defaults
    # cuDNN's unset upper clip is the float maximum (numpy.finfo(numpy.float32).max)
    "graph.nodes[*].relu_upper_clip": 3.4028234663852886e38,
    "graph.nodes[*].relu_lower_clip_slope": 0.0,
    "graph.nodes[*].swish_beta": 1.0,
    "graph.nodes[*].elu_alpha": 1.0,
    "graph.nodes[*].softplus_beta": 1.0,
    "graph.nodes[*].axis_tensor_uid": -1.0,  # no axis: a marker
    "graph.nodes[*].axis": X_RANK - 1,  # BlockScaleQuantizeNode.hpp: the last axis of x
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
    # A GDN/KDA scale and a BlockScaleQuantize axis read the node's own operands, not
    # another node's.
    absent = {
        reference[1:]: None,
        "graph.nodes[3].q.dims[2]": 64,
        "graph.nodes[0].q.dims[2]": 16,
        "graph.nodes[3].x.rank": X_RANK,
        "graph.nodes[0].x.rank": X_RANK + 1,
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


#: The artifact root class of each adapter whose artifact embeds a features_hash.
_ROOTS = {
    "tree_data": "GbdtModel",
    "linear_data": "LinearModel",
    "mlp_data": "MlpModel",
}
#: The linear and MLP fixtures of their own test modules read four features.
_FOUR = [SCALE, Q_DIM, FLOPS, K_DIM]


def _artifact(adapter, features_hash, signature, metric) -> bytes:
    if adapter == "tree_data":
        return leaf_model(features_hash, len(signature))
    assert len(signature) == 4, "the linear and MLP fixtures read four features"
    if adapter == "linear_data":
        from uhd_gen.linear_to_flatbuffer import build_linear_model
        from uhd_gen.tests.test_linear_model import fixture_ranker

        return build_linear_model(fixture_ranker(), features_hash)
    from uhd_gen.mlp_to_flatbuffer import build_mlp_model
    from uhd_gen.tests.test_mlp_model import fixture_ensemble

    return build_mlp_model(fixture_ensemble(), metric, features_hash)


def _installed_model(
    tmp_path,
    signature,
    metric="tflops",
    *,
    role="sort_kernel_catalog",
    adapter="tree_data",
    encoding=None,
):
    directory = tmp_path / role / "gfx942" / metric
    directory.mkdir(parents=True)
    features_hash = compute_features_hash(signature, encoding)
    data = _artifact(adapter, features_hash, signature, metric)
    (directory / "model.bin").write_bytes(data)
    descriptor = {
        "version": "1.0",
        "id": "0b8f5f3e-8f0c-4d6c-9d2e-3a3c1b0f6a11",
        "name": "test model",
        "adapter": adapter,
        "features_signature": signature,
        "features_hash": features_hash,
        "objective": "max",
        "score": {"metric": metric, "calibrated": True, "transform": "log"},
        adapter: {
            "artifact": "model.bin",
            "hash": hashlib.sha256(data).hexdigest(),
        },
    }
    if encoding:
        descriptor["categorical_encoding"] = encoding
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


def _files(path):
    """Every file beside a descriptor, with its bytes: a rewrite leaves nothing else there."""
    return {file.name: file.read_bytes() for file in path.parent.iterdir()}


@pytest.mark.parametrize("adapter", sorted(_ROOTS))
def test_rewrite_keeps_the_model_and_restates_every_hash(tmp_path, evaluator, adapter):
    from uhd_gen.promote import _verify_artifact
    from uhd_gen.rewrite import _same, rewrite_descriptor

    root = _ROOTS[adapter]
    module = importlib.import_module(f"hipdnn_flatbuffers_sdk.data_objects.{root}")
    path = _installed_model(tmp_path, _FOUR, adapter=adapter)
    artifact = path.parent / "model.bin"
    before = getattr(module, root + "T").InitFromPackedBuf(
        bytearray(artifact.read_bytes()), 0
    )

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
        K_DIM,
    ]
    assert descriptor["features_hash"] == compute_features_hash(
        descriptor["features_signature"]
    )
    # The loader's checks: the embedded hash and digest agree with the descriptor.
    assert _verify_artifact(descriptor, artifact, "sort_kernel_catalog") == (
        descriptor[adapter]["hash"]
    )
    after = getattr(module, root + "T").InitFromPackedBuf(
        bytearray(artifact.read_bytes()), 0
    )
    assert after.featuresHash.decode() == descriptor["features_hash"]
    before.featuresHash = after.featuresHash = None
    assert _same(before, after)

    files = _files(path)
    assert rewrite_descriptor(path, _rewrite_args([path]))["status"] == "already_safe"
    assert _files(path) == files


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


def test_a_gdn_scale_is_rewritten_though_the_signature_reads_no_q_dimension(
    tmp_path, evaluator
):
    """The default reads q.dims[2], which the proof rows must bind though the old signature
    never named it."""
    from uhd_gen.rewrite import rewrite_descriptor

    scale = "$graph.nodes[0].scale"
    path = _installed_model(tmp_path, [scale, "$graph.nodes[0].k.dims[2]"])
    report = rewrite_descriptor(path, _rewrite_args([path]))

    assert report["status"] == "rewritten"
    assert [item["default"] for item in report["wrapped"]] == [
        {"rsqrt": ["$graph.nodes[0].q.dims[2]"]}
    ]
    assert report["proof"]["old_refused_rows_when_absent"] == {
        scale: report["proof"]["rows_compared"]
    }


def test_an_adapter_whose_artifact_embeds_the_hash_without_a_rule_is_refused(
    tmp_path, evaluator
):
    """A table model records features_hash too, and the loader rejects one that differs from
    the descriptor's: rewriting only the descriptor would break the model."""
    from uhd_gen.rewrite import rewrite_descriptor

    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["adapter"] = "table"
    descriptor["table"] = descriptor.pop("tree_data")
    path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    files = _files(path)

    with pytest.raises(ValueError, match="no rewrite rule"):
        rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


def test_an_encoded_categorical_cannot_take_a_default(tmp_path, evaluator):
    from uhd_gen.rewrite import rewrite_descriptor

    path = _installed_model(
        tmp_path, [SCALE, Q_DIM], encoding={SCALE: {"a": 1, "b": 2}}
    )
    files = _files(path)
    with pytest.raises(ValueError, match="encoded categorical"):
        rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


def test_a_rewrite_that_would_change_a_bound_feature_is_refused(
    tmp_path, evaluator, monkeypatch
):
    from uhd_gen import rewrite

    altered = [{"value_or_default": [SCALE, 1.0]}, {"*": [Q_DIM, 2]}]
    monkeypatch.setattr(
        rewrite,
        "safe_signature",
        lambda *_: (altered, [{"reference": SCALE, "default": 1.0}]),
    )
    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    files = _files(path)

    with pytest.raises(ValueError, match="changes features"):
        rewrite.rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


def test_a_rewrite_that_leaves_a_graph_without_the_feature_unscorable_is_refused(
    tmp_path, evaluator, monkeypatch
):
    """The wrap claims the scale but leaves it bare: bound rows agree, and every row without
    the scale is still unscorable."""
    from uhd_gen import rewrite

    monkeypatch.setattr(
        rewrite,
        "safe_signature",
        lambda *_: ([SCALE, Q_DIM], [{"reference": SCALE, "default": 1.0}]),
    )
    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    files = _files(path)

    with pytest.raises(ValueError, match="attn_scale_value"):
        rewrite.rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


#: The features_hash the repack tests restamp an artifact with.
NEW_HASH = "sha256:0123456789abcdef"


def _gbdt_pack_that(monkeypatch, change):
    """Patches the tree artifact's packer to apply `change` to a model before packing it
    under the new features_hash, as a repacking bug would."""
    from hipdnn_flatbuffers_sdk.data_objects import GbdtModel

    pack = GbdtModel.GbdtModelT.Pack

    def changed(self, builder):
        if self.featuresHash == NEW_HASH:
            saved = dict(vars(self))
            change(self)
            offset = pack(self, builder)
            vars(self).update(saved)
            return offset
        return pack(self, builder)

    monkeypatch.setattr(GbdtModel.GbdtModelT, "Pack", changed)


def test_a_repack_that_loses_the_new_hash_is_refused(monkeypatch):
    from uhd_gen.rewrite import restamp_artifact

    data = leaf_model("sha256:fedcba9876543210", 2)
    _gbdt_pack_that(monkeypatch, lambda model: setattr(model, "featuresHash", None))
    with pytest.raises(ValueError, match="does not record the new features_hash"):
        restamp_artifact(data, "tree_data", NEW_HASH)


def test_a_repack_that_changes_another_value_is_refused(monkeypatch):
    from uhd_gen.rewrite import restamp_artifact

    data = leaf_model("sha256:fedcba9876543210", 2)
    _gbdt_pack_that(monkeypatch, lambda model: setattr(model, "learningRate", 0.5))
    with pytest.raises(ValueError, match="other than features_hash"):
        restamp_artifact(data, "tree_data", NEW_HASH)

    # Unpatched, the same artifact restamps.
    monkeypatch.undo()
    restamped, _ = restamp_artifact(data, "tree_data", NEW_HASH)
    assert restamped != data


def test_an_artifact_the_loader_would_refuse_leaves_both_files_untouched(
    tmp_path, evaluator, monkeypatch
):
    from uhd_gen import rewrite
    from uhd_gen.promote import PromoteError

    def refuse(descriptor, artifact, role):
        # What the staged artifact holds is already the restamped one.
        assert artifact.read_bytes() != (path.parent / "model.bin").read_bytes()
        raise PromoteError("the engine would refuse this artifact")

    monkeypatch.setattr(rewrite, "_verify_artifact", refuse)
    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    files = _files(path)

    with pytest.raises(PromoteError, match="would refuse"):
        rewrite.rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


def test_a_descriptor_that_cannot_be_replaced_puts_the_artifact_back(
    tmp_path, evaluator, monkeypatch
):
    """The artifact is swapped in first and the descriptor, which binds it, last: when that
    last replace fails the artifact must not stay under a descriptor that disowns it."""
    from uhd_gen import rewrite

    real = os.replace

    def replace(source, destination):
        if Path(destination).name == "heuristic.uhd.json":
            raise OSError("disk full")
        real(source, destination)

    monkeypatch.setattr(rewrite.os, "replace", replace)
    path = _installed_model(tmp_path, [SCALE, Q_DIM])
    files = _files(path)

    with pytest.raises(OSError, match="disk full"):
        rewrite.rewrite_descriptor(path, _rewrite_args([path]))
    assert _files(path) == files


def test_rewriting_a_ranker_reports_the_l1_models_whose_selector_revision_it_changes(
    tmp_path, evaluator
):
    """A ranker's model and features hashes are part of its engine's selector revision, so
    the L1 models trained against the old one are unavailable until restamped."""
    from uhd_gen.rewrite import rewrite_descriptor

    l1 = _installed_model(tmp_path / "engine", [Q_DIM], role="predict_engine")
    ranker = _installed_model(tmp_path / "engine", [SCALE, Q_DIM])
    report = rewrite_descriptor(ranker, _rewrite_args([ranker]))
    assert report["selector_revision"] == {
        "changes": True,
        "l1_models_to_restamp_or_retrain": [str(l1)],
    }

    # An L1 model is not part of its own engine's selector.
    scorer = _installed_model(tmp_path / "other", [SCALE, Q_DIM], role="predict_engine")
    report = rewrite_descriptor(scorer, _rewrite_args([scorer]))
    assert report["selector_revision"] == {"changes": False}


def test_the_command_rewrites_what_it_can_reports_what_it_cannot_and_exits_1(
    tmp_path, evaluator, capsys
):
    from uhd_gen.__main__ import main

    good = _installed_model(tmp_path, [SCALE, Q_DIM])
    # No role directory in its path and no --role: unresolvable.
    bad = tmp_path / "elsewhere" / "heuristic.uhd.json"
    bad.parent.mkdir()
    bad.write_bytes(good.read_bytes())
    report_file = tmp_path / "report.json"

    code = main(
        [
            "rewrite-signature",
            str(good),
            str(bad),
            "--feature-evaluator",
            evaluator,
            "--report",
            str(report_file),
        ]
    )

    out = capsys.readouterr().out
    assert code == 1
    reports = json.loads(out)
    assert [report["status"] for report in reports] == ["rewritten", "failed"]
    assert "name its role" in reports[1]["error"]
    assert json.loads(good.read_text(encoding="utf-8"))["features_signature"][0] == (
        {"value_or_default": [SCALE, 1.0]}
    )
    assert report_file.read_text(encoding="utf-8") == out


def test_the_evaluator_refuses_what_it_cannot_run_and_says_why(evaluator):
    with pytest.raises(ValueError, match="Usage: hipdnn_uhd_features"):
        run_evaluator(None, evaluator, "--bogus")
    with pytest.raises(ValueError, match="requires a signature array"):
        run_evaluator({}, evaluator, "--lint")
    # An operator nobody defines is refused by the compile that precedes the lint, and the
    # error says to rebuild an evaluator that predates the optional-feature checks.
    with pytest.raises(ValueError, match="nosuchoperator.*rebuild it"):
        lint([{"nosuchoperator": [1]}], "time", evaluator)


def test_a_model_without_a_metric_lints_the_work_features_as_unguarded(evaluator):
    """Only a tflops model may read the work bare; a metric-less one may not."""
    assert [found["reference"] for found in lint([FLOPS], None, evaluator)] == [FLOPS]
    assert lint([FLOPS], "tflops", evaluator) == []
