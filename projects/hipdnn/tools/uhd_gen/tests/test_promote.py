# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Promotion preserves every untargeted model and refuses unsafe writes."""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import uuid
from pathlib import Path

import pytest

# Every model here is a real artifact built by the shipping converter, which needs both.
pytest.importorskip("flatbuffers")
pytest.importorskip("lightgbm")
pytest.importorskip("pandas")

from uhd_gen.features import (
    compute_features_hash,
    evaluator_feature_semantics_revision,
    resolve_feature_evaluator,
)
from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model
from uhd_gen.promote import (
    PromoteError,
    _apply,
    add_promote_arguments,
    build_plan,
    parse_uhd_ids,
    run_promote,
)
from uhd_gen.provenance import snapshot_provenance

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
OLD = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
NEW = "cf37fa30-32dc-4a21-a008-68ef5e0d30a6"
OTHER = "edc1d5b4-6f12-4a40-a749-403966474bc9"
KERNEL_SIGNATURE = ("$kernel.block_size",)
OPAQUE_SIGNATURE = ("$graph.work",)

# Promoting a feature-reading model asks the shared evaluator for the build's feature
# semantics revision (FeatureSemantics.hpp), exactly as training and evaluation already
# ask it for the digest: without a built evaluator there is nothing to promote against.
pytestmark = pytest.mark.usefixtures("evaluator")


@functools.cache
def _hash(signature=KERNEL_SIGNATURE, encoding=None):
    """The digest the loader recomputes, which is the only one promote now accepts."""
    return compute_features_hash(
        list(signature),
        json.loads(encoding) if encoding else None,
        resolve_feature_evaluator(),
    )


def _weights(
    leaf=1.0,
    *,
    grouped=False,
    signature=KERNEL_SIGNATURE,
    features_hash=None,
    training_arches=None,
    num_features=None,
):
    """A `tree_data` artifact the runtime would load: promote verifies what it installs."""
    ensemble = {
        "tree_info": [{"tree_structure": {"leaf_value": leaf}}],
        "max_feature_idx": (num_features or len(signature)) - 1,
        "objective": "regression",
    }
    return build_gbdt_model(
        ensemble,
        features_hash or _hash(tuple(signature)),
        training_arches=training_arches,
        group_by_feature_index=0 if grouped else -1,
        groups=[(0.0, ensemble)] if grouped else None,
    )


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree(root):
    _write(
        root / "engine.ued.json",
        {
            "version": "1.0",
            "id": UED,
            "name": "test:engine",
            "metadata": KMD,
            "knobs": ["block_size", "tile_m"],
            "sort_kernel_catalog": {"gfx942": OLD, "gfx950": OTHER, "default": OTHER},
            "predict_engine": {"gfx942": OTHER},
        },
    )
    _write(
        root / "metadata.kmd.json",
        {
            "version": "1.0",
            "id": KMD,
            "fields": [
                {"name": "block_size", "type": "int"},
                {"name": "tile_m", "type": "int"},
            ],
        },
    )
    # The incumbent gfx942 ranker is installed: promotion reads a bound UHD's metric to
    # decide whether the incoming one replaces it.
    _installed(root / "old.uhd.json", OLD)
    return root


def _installed(path, identity, score=None):
    """An installed static_order UHD, metric-less unless `score` names one."""
    doc = {
        "version": "1.0",
        "id": identity,
        "name": "installed",
        "adapter": "static_order",
        "static_order": {},
    }
    if score is not None:
        doc.update(
            objective={"tflops": "max", "time": "min"}[score["metric"]], score=score
        )
    return _write(path, doc)


def _model(
    root,
    tree,
    *,
    identity=NEW,
    artifact="model.bin",
    provenance=None,
    metric=None,
    signature=KERNEL_SIGNATURE,
):
    provenance = (
        provenance
        if provenance is not None
        else snapshot_provenance(tree, arch="gfx942")
    )
    doc = {
        "version": "1.0",
        "id": identity,
        "name": "model",
        "adapter": "tree_data",
        "objective": "max",
        "features_signature": list(signature),
        "features_hash": _hash(tuple(signature)),
        "trained_against": provenance,
        "tree_data": {"artifact": artifact},
    }
    if metric is not None:
        doc.update(
            objective={"tflops": "max", "time": "min"}[metric],
            score={"metric": metric, "calibrated": True, "transform": "log1p"},
        )
    _write(root / "heuristic.uhd.json", doc)
    _write(
        root / "train_manifest.json",
        {"training_arches": ["gfx942"], "trained_against": provenance},
    )
    payload = root / artifact
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(_weights(signature=signature))
    return root


def _files(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_promotion_updates_only_the_requested_arch_and_role(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    model = _model(tmp_path / "model", tree)
    _apply(build_plan(model, tree))
    expected = copy.deepcopy(original)
    expected["sort_kernel_catalog"]["gfx942"] = [NEW]
    assert _read(tree / "engine.ued.json") == expected


def test_role_and_explicit_default_target_preserve_other_maps(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    model = _model(tmp_path / "model", tree)
    _apply(build_plan(model, tree, role="predict_applicable_kernels", arch="default"))
    expected = copy.deepcopy(original)
    expected["predict_applicable_kernels"] = {"default": NEW}
    assert _read(tree / "engine.ued.json") == expected


def test_install_rewrites_nested_artifact_path_portably(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree, artifact="nested/model.bin")
    plan = build_plan(model, tree)
    _apply(plan)
    installed = _read(plan.destination_descriptor)
    artifact = plan.destination_descriptor.parent / installed["tree_data"]["artifact"]
    assert artifact.read_bytes() == _weights()


@pytest.mark.parametrize("binding", ["role", "architecture", "engine"])
def test_default_filenames_preserve_other_model_bindings(tmp_path, binding):
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued["sort_kernel_catalog"] = {"gfx942": OLD}
    ued.pop("predict_engine")
    _write(tree / "engine.ued.json", ued)
    first = _model(tmp_path / "first", tree)
    (first / "model.bin").write_bytes(_weights(1.0))
    first_plan = build_plan(first, tree)
    _apply(first_plan)
    first_descriptor = first_plan.destination_descriptor.read_bytes()
    first_document = _read(first_plan.destination_descriptor)
    first_artifact = (
        first_plan.destination_descriptor.parent
        / first_document["tree_data"]["artifact"]
    )

    engine, role, arch = "test:engine", "sort_kernel_catalog", "gfx942"
    if binding == "engine":
        engine = "test:second"
        second_ued = copy.deepcopy(ued)
        second_ued.update(id=UED[:-1] + "3", name=engine)
        _write(tree / "second.ued.json", second_ued)
    elif binding == "architecture":
        arch = "gfx950"
    else:
        role = "predict_engine"
    provenance = snapshot_provenance(tree, engine=engine, arch=arch)
    signature = KERNEL_SIGNATURE
    if binding == "role":
        # An engine-level model records the selector revision it was measured under, as
        # every bench binding does, and reads only the graph.
        provenance["selector_revision"] = "provider-1"
        signature = ("$graph.flops",)
    second = _model(
        tmp_path / "second",
        tree,
        identity=OTHER,
        provenance=provenance,
        signature=signature,
    )
    document = _read(second / "heuristic.uhd.json")
    if binding == "role":
        document.update(
            score={"metric": "tflops", "calibrated": True, "transform": "identity"}
        )
    _write(second / "heuristic.uhd.json", document)
    _write(
        second / "train_manifest.json",
        {"training_arches": [arch], "trained_against": provenance},
    )
    (second / "model.bin").write_bytes(_weights(2.0, signature=signature))
    second_plan = build_plan(second, tree, engine=engine, role=role, arch=arch)
    _apply(second_plan)

    assert first_plan.destination_descriptor.read_bytes() == first_descriptor
    assert first_artifact.read_bytes() == _weights(1.0)
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [NEW]
    installed = {
        _read(path)["id"]: (path, _read(path)) for path in tree.rglob("*.uhd.json")
    }
    [selected] = _read(second_plan.ued_path)[role][arch]
    path, document = installed[selected]
    assert (path.parent / document["tree_data"]["artifact"]).read_bytes() == _weights(
        2.0, signature=signature
    )

    (second / "model.bin").write_bytes(_weights(3.0, signature=signature))
    _apply(build_plan(second, tree, engine=engine, role=role, arch=arch))
    assert first_artifact.read_bytes() == _weights(1.0)
    assert (path.parent / document["tree_data"]["artifact"]).read_bytes() == _weights(
        3.0, signature=signature
    )


def test_dry_run_makes_no_writes(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    before = _files(tmp_path)
    parser = argparse.ArgumentParser()
    add_promote_arguments(parser)
    args = parser.parse_args(
        ["--model-dir", str(model), "--descriptor-tree", str(tree), "--dry-run"]
    )
    assert run_promote(args) == 0
    assert _files(tmp_path) == before


@pytest.mark.parametrize(
    "role,arch", [("sort_kernel_catalog", "gfx950"), ("predict_engine", "gfx942")]
)
def test_shared_identity_cannot_be_rewritten_from_another_binding(tmp_path, role, arch):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree, identity=OTHER)
    incumbent = _read(model / "heuristic.uhd.json")
    incumbent["id"] = OTHER
    _write(tree / "heuristic.uhd.json", incumbent)
    (tree / "model.bin").write_bytes(b"untargeted model")
    ued = _read(tree / "engine.ued.json")
    ued["sort_kernel_catalog"] = {"gfx942": OLD}
    ued.pop("predict_engine")
    ued.setdefault(role, {})[arch] = OTHER
    _write(tree / "engine.ued.json", ued)
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def test_same_id_in_another_directory_is_not_installed_twice(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _write(tree / "sub" / "different.uhd.json", _read(model / "heuristic.uhd.json"))
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def test_shared_artifact_collision_is_found_outside_destination_directory(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    destination = build_plan(model, tree).destination_descriptor.parent
    incumbent = _read(model / "heuristic.uhd.json")
    incumbent["id"] = OTHER
    incumbent["tree_data"]["artifact"] = "../model.bin"
    _write(destination / "other" / "other.uhd.json", incumbent)
    (destination / "model.bin").write_bytes(b"untargeted model")
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    assert _files(tmp_path) == before


@pytest.mark.parametrize(
    "missing",
    [
        "trained_against",
        "features_hash",
        "features_signature",
        "objective",
        "tree_data",
        "name",
        "version",
    ],
)
def test_absent_required_headers_fail_before_writes(tmp_path, missing):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    doc = _read(model / "heuristic.uhd.json")
    del doc[missing]
    _write(model / "heuristic.uhd.json", doc)
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("mutation", ["identity", "major", "minor", "manifest"])
def test_incompatible_training_provenance_is_never_restamped(tmp_path, mutation):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    doc = _read(model / "heuristic.uhd.json")
    if mutation == "identity":
        doc["trained_against"]["ued"]["id"] = OTHER
    elif mutation == "major":
        doc["trained_against"]["kmd"]["revision"] = "2.0"
    elif mutation == "minor":
        doc["trained_against"]["ued"]["revision"] = "1.1"
    else:
        manifest = _read(model / "train_manifest.json")
        manifest["trained_against"]["ued"]["revision"] = "1.1"
        _write(model / "train_manifest.json", manifest)
    _write(model / "heuristic.uhd.json", doc)
    if mutation != "manifest":
        (model / "train_manifest.json").unlink()
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree, arch="gfx942")
    assert _files(tmp_path) == before


def test_training_feature_pruning_never_removes_authored_knobs(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    manifest = _read(model / "train_manifest.json")
    manifest["dropped_constant_features"] = [{"column": "kernel.tile_m", "value": 128}]
    _write(model / "train_manifest.json", manifest)
    _apply(build_plan(model, tree))
    ued = _read(tree / "engine.ued.json")
    assert ued["knobs"] == ["block_size", "tile_m"]
    assert "revision" not in ued


def test_explicit_knob_removal_requires_retraining_for_prospective_revision(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree, remove_knobs=["tile_m"])
    assert _files(tmp_path) == before


def test_explicit_knob_removal_bumps_semantic_not_file_revision(tmp_path):
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued.pop("predict_engine")
    ued["sort_kernel_catalog"] = {"gfx942": OLD}
    _write(tree / "engine.ued.json", ued)
    provenance = snapshot_provenance(tree)
    provenance["ued"]["revision"] = "2.0"
    model = _model(tmp_path / "model", tree, provenance=provenance)
    plan = build_plan(model, tree, remove_knobs=["tile_m"])
    _apply(plan)
    revised = _read(tree / "engine.ued.json")
    assert revised["knobs"] == ["block_size"]
    assert revised["version"] == "1.0"
    assert revised["revision"] == "2.0"
    assert _read(plan.destination_descriptor)["trained_against"] == provenance


def test_explicit_knob_removal_cannot_invalidate_other_role_models(tmp_path):
    tree = _tree(tmp_path / "tree")
    incumbent_provenance = snapshot_provenance(tree)
    provenance = copy.deepcopy(incumbent_provenance)
    provenance["ued"]["revision"] = "2.0"
    model = _model(tmp_path / "model", tree, provenance=provenance)
    other = _read(model / "heuristic.uhd.json")
    other["id"] = OTHER
    other["tree_data"]["artifact"] = "other.bin"
    other["trained_against"] = incumbent_provenance
    _write(tree / "other.uhd.json", other)
    (tree / "other.bin").write_bytes(b"other")
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree, remove_knobs=["tile_m"])
    assert _files(tmp_path) == before


@pytest.mark.parametrize("arches", [[], ["gfx942", "gfx950"]])
def test_ambiguous_architecture_requires_an_explicit_target(tmp_path, arches):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _write(model / "train_manifest.json", {"training_arches": arches})
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    _installed(tree / "other.uhd.json", OTHER)
    _apply(build_plan(model, tree, arch="default"))
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["default"] == [NEW]


def test_training_architecture_cannot_be_silently_retargeted(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    with pytest.raises(PromoteError):
        build_plan(model, tree, arch="gfx950")


def test_additive_semantic_revision_accepts_without_rewriting_training_snapshot(
    tmp_path,
):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    trained = _read(model / "heuristic.uhd.json")["trained_against"]
    ued = _read(tree / "engine.ued.json")
    ued["revision"] = "1.12"
    _write(tree / "engine.ued.json", ued)
    plan = build_plan(model, tree)
    _apply(plan)
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [NEW]
    assert _read(plan.destination_descriptor)["trained_against"] == trained


def _bind(tree, value):
    ued = _read(tree / "engine.ued.json")
    ued["sort_kernel_catalog"]["gfx942"] = value
    _write(tree / "engine.ued.json", ued)


def test_a_new_metric_joins_the_arch_list_beside_the_incumbent(tmp_path):
    """RFC 0019 §3.1: one UHD per metric under an arch key. A single id is a one-element
    list, so binding a second metric turns it into two, and neither file is disturbed.
    """
    tree = _tree(tmp_path / "tree")
    _installed(tree / "old.uhd.json", OLD, {"metric": "tflops", "calibrated": False})
    first = _files(tree)
    plan = build_plan(_model(tmp_path / "model", tree, metric="time"), tree)
    _apply(plan)
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [
        OLD,
        NEW,
    ]
    assert plan.old_heuristic is None
    assert plan.destination_descriptor.parent.name == "time"
    assert all(
        _files(tree)[path] == content
        for path, content in first.items()
        if path.name != "engine.ued.json"
    )


def test_a_retrained_metric_replaces_only_its_own_entry_in_place(tmp_path):
    tree = _tree(tmp_path / "tree")
    _installed(tree / "old.uhd.json", OLD, {"metric": "time", "calibrated": False})
    _installed(
        tree / "other.uhd.json", OTHER, {"metric": "tflops", "calibrated": False}
    )
    _bind(tree, [OLD, OTHER])
    plan = build_plan(_model(tmp_path / "model", tree, metric="time"), tree)
    _apply(plan)
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [
        NEW,
        OTHER,
    ]
    assert plan.old_heuristic == OLD


def test_a_metric_less_ranker_replaces_only_the_metric_less_entry(tmp_path):
    tree = _tree(tmp_path / "tree")
    _installed(
        tree / "other.uhd.json", OTHER, {"metric": "tflops", "calibrated": False}
    )
    _bind(tree, [OTHER, OLD])
    _apply(build_plan(_model(tmp_path / "model", tree), tree))
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [
        OTHER,
        NEW,
    ]


def test_a_bound_uhd_the_tree_does_not_contain_cannot_be_classified(tmp_path):
    """The metric lives in the UHD, not the UED, so an unreadable binding could be the
    incoming metric's incumbent or another metric's model; either guess corrupts the map.
    """
    tree = _tree(tmp_path / "tree")
    (tree / "old.uhd.json").unlink()
    model = _model(tmp_path / "model", tree, metric="time")
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match="not installed"):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def test_an_arch_already_binding_two_uhds_of_one_metric_is_refused(tmp_path):
    tree = _tree(tmp_path / "tree")
    _installed(tree / "old.uhd.json", OLD, {"metric": "time", "calibrated": False})
    _installed(tree / "other.uhd.json", OTHER, {"metric": "time", "calibrated": False})
    _bind(tree, [OLD, OTHER])
    with pytest.raises(PromoteError, match="at most one"):
        build_plan(_model(tmp_path / "model", tree, metric="time"), tree)


@pytest.mark.parametrize(
    "score,objective",
    [
        ({"metric": "latency", "transform": "log1p"}, "min"),
        ({"metric": "time", "transform": "log1p"}, "max"),
        ({"metric": "tflops", "transform": "log1p"}, "min"),
        ({"calibrated": True, "transform": "log1p"}, "max"),
        ({"units": "tflops", "transform": "log1p"}, "max"),
    ],
)
def test_a_score_must_name_a_registered_metric_in_its_own_direction(
    tmp_path, score, objective
):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    doc = _read(model / "heuristic.uhd.json")
    doc.update(score=score, objective=objective)
    _write(model / "heuristic.uhd.json", doc)
    with pytest.raises(PromoteError):
        build_plan(model, tree)


def test_a_static_order_declaring_criteria_is_refused_before_any_write(tmp_path):
    """static_order ranks by UKD priority, then descriptor id, and takes no parameters; the
    runtime refuses a body declaring `order`, so promote must not install one."""
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _write(
        model / "heuristic.uhd.json",
        {
            "version": "1.0",
            "id": NEW,
            "name": "model",
            "adapter": "static_order",
            "static_order": {"order": ["priority", "id"]},
        },
    )
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match="static_order.order is not supported"):
        build_plan(model, tree)
    assert _files(tmp_path) == before


@pytest.mark.parametrize(
    "payload", ["missing.bin", "../outside.bin", "metadata.kmd.json"]
)
def test_invalid_artifact_destination_or_source_never_writes(tmp_path, payload):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    doc = _read(model / "heuristic.uhd.json")
    doc["tree_data"]["artifact"] = payload
    _write(model / "heuristic.uhd.json", doc)
    if payload != "missing.bin":
        (model / payload).write_bytes(b"unsafe payload")
    before = _files(tmp_path)
    with pytest.raises(PromoteError):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def _train(tmp_path, *extra):
    for dependency in ("lightgbm", "pandas", "flatbuffers"):
        pytest.importorskip(dependency)
    from uhd_gen.__main__ import main

    tree = _tree(tmp_path / "tree")
    provenance = _write(tmp_path / "provenance.json", snapshot_provenance(tree))
    csv = tmp_path / "bench.csv"
    rows = ["kernel.block_size,tflops"]
    for index in range(40):
        rows.extend((f"64,{90 + index * .01}", f"256,{50 + index * .01}"))
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return main(
        [
            "train",
            "--input",
            str(csv),
            "--features",
            "kernel.block_size",
            "--target",
            "tflops",
            "--output-dir",
            str(tmp_path / "model"),
            "--provenance",
            str(provenance),
            "--num-boost-round",
            "10",
            "--early-stopping",
            "5",
            *extra,
        ]
    )


def test_train_retains_explicit_model_identity(tmp_path, evaluator):
    assert _train(tmp_path, "--uhd-id", NEW) == 0
    assert _read(tmp_path / "model" / "heuristic.uhd.json")["id"] == NEW
    assert _read(tmp_path / "model" / "train_manifest.json")["uhd_id"] == NEW


def test_train_mints_identity_when_omitted(tmp_path, evaluator):
    assert _train(tmp_path) == 0
    identity = _read(tmp_path / "model" / "heuristic.uhd.json")["id"]
    assert str(uuid.UUID(identity)) == identity


def _record_semantics(model, revision):
    """Rewrite what `model` recorded, in the UHD and its training manifest alike."""
    for name in ("heuristic.uhd.json", "train_manifest.json"):
        document = _read(model / name)
        document["trained_against"]["feature_semantics_revision"] = revision
        _write(model / name, document)
    return model


def test_train_records_the_evaluators_revision_and_promote_refuses_it_elsewhere(
    tmp_path, evaluator, evaluator_reporting
):
    """FeatureSemantics.hpp end to end: a model trained by a build computing revision N
    records N, installs for that build, and is refused -- naming both revisions -- for a
    build computing anything else, as the loader would refuse it."""
    current = evaluator_feature_semantics_revision(evaluator)
    bumped = evaluator_reporting(current + 1)
    assert _train(tmp_path, "--feature-evaluator", bumped) == 0
    model = tmp_path / "model"
    assert (
        _read(model / "heuristic.uhd.json")["trained_against"][
            "feature_semantics_revision"
        ]
        == current + 1
    )
    assert (
        _read(model / "train_manifest.json")["trained_against"][
            "feature_semantics_revision"
        ]
        == current + 1
    )
    tree = tmp_path / "tree"
    build_plan(model, tree, arch="gfx942", feature_evaluator=bumped)
    before = _files(tmp_path)
    with pytest.raises(
        PromoteError, match=rf"revision {current + 1}\b.*revision {current}\b"
    ):
        build_plan(model, tree, arch="gfx942", feature_evaluator=evaluator)
    assert _files(tmp_path) == before


@pytest.mark.parametrize(
    "opaque", [False, True], ids=["catalog_ranker", "engine_prediction"]
)
def test_a_model_recording_no_feature_semantics_is_revision_1(
    tmp_path, evaluator_reporting, opaque
):
    """Every model shipped before the revision existed records none, so absent means 1: it
    promotes against a build computing 1 and is refused, naming both, by one computing 2.
    """
    tree = _tree(tmp_path / "tree")
    if opaque:
        model = _opaque_model(tmp_path / "model")
        target = {
            "engine": "ASM_SDPA_ENGINE",
            "role": "predict_engine",
            "arch": "gfx950",
        }
    else:
        model = _model(tmp_path / "model", tree)
        target = {}
    assert (
        "feature_semantics_revision"
        not in _read(model / "heuristic.uhd.json")["trained_against"]
    )
    build_plan(model, tree, feature_evaluator=evaluator_reporting(1), **target)
    with pytest.raises(PromoteError, match=r"revision 1\b.*revision 2\b"):
        build_plan(model, tree, feature_evaluator=evaluator_reporting(2), **target)
    _record_semantics(model, 2)
    build_plan(model, tree, feature_evaluator=evaluator_reporting(2), **target)


@pytest.mark.parametrize(
    "malformed", ["not-a-uuid", "", "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d6"]
)
def test_train_rejects_malformed_identity_before_outputs(tmp_path, malformed):
    assert _train(tmp_path, "--uhd-id", malformed) == 1
    assert not (tmp_path / "model" / "heuristic.uhd.json").exists()


def _opaque_model(root, *, identity=NEW, revision="aiter-fwd-1", declared=NEW):
    """A model for an engine with no UED: trained against a selector revision only.

    `declared` is the id the engine's description carried for the metric while the corpus
    was collected (binding.uhd_id), i.e. the provider's declaration; None when it had none.
    """
    provenance = {"selector_revision": revision}
    doc = {
        "version": "1.0",
        "id": identity,
        "name": "model",
        "adapter": "tree_data",
        "objective": "max",
        "features_signature": list(OPAQUE_SIGNATURE),
        "features_hash": _hash(OPAQUE_SIGNATURE),
        "trained_against": provenance,
        "score": {"metric": "tflops", "calibrated": True, "transform": "identity"},
        "tree_data": {"artifact": "model.bin"},
    }
    _write(root / "heuristic.uhd.json", doc)
    binding = {"engine": "ASM_SDPA_ENGINE", "metric": "tflops"}
    if declared is not None:
        binding["uhd_id"] = declared
    _write(
        root / "train_manifest.json",
        {
            "training_arches": ["gfx950"],
            "trained_against": provenance,
            "role": "predict_engine",
            "binding": binding,
        },
    )
    (root / "model.bin").write_bytes(_weights(signature=OPAQUE_SIGNATURE))
    return root


def test_a_model_for_an_engine_with_no_ued_installs_without_touching_a_role_map(
    tmp_path,
):
    """AITER and MIOpen own no descriptor set: RFC 0019 4.1 / Open Question 7 binds their
    model by a UUID declared in provider code, so promotion writes the document where the
    loader scans and edits nothing. Before this, `select_engine` ended the run with
    "expected one UED for --engine 'ASM_SDPA_ENGINE', found 0" -- after the model had
    already trained (run 67929588)."""
    tree = _tree(tmp_path / "tree")
    before = _read(tree / "engine.ued.json")
    model = _opaque_model(tmp_path / "model")

    plan = build_plan(
        model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950"
    )
    _apply(plan)

    assert plan.ued_path is None
    assert _read(tree / "engine.ued.json") == before, "no role map may change"
    installed = (
        tree / "heuristics" / "ASM_SDPA_ENGINE" / "predict_engine" / "gfx950" / "tflops"
    )
    assert _read(installed / "heuristic.uhd.json")["id"] == NEW
    assert (installed / "model.bin").read_bytes() == _weights(
        signature=OPAQUE_SIGNATURE
    )


def test_an_opaque_model_cannot_take_an_identity_already_installed(tmp_path):
    """The UUID is the whole binding: the provider looks for exactly one, so a second
    document carrying it makes the engine's model ambiguous rather than replaced."""
    tree = _tree(tmp_path / "tree")
    _write(
        tree / "elsewhere" / "other.uhd.json",
        {
            "version": "1.0",
            "id": NEW,
            "name": "other",
            "adapter": "tree_data",
            "objective": "max",
            "features_signature": ["$graph.work"],
            "features_hash": "sha256:" + "0" * 16,
            "trained_against": {"selector_revision": "r"},
            "tree_data": {"artifact": "other.bin"},
        },
    )
    model = _opaque_model(tmp_path / "model")
    with pytest.raises(PromoteError, match="already installed"):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")


def test_an_opaque_model_with_no_declared_id_is_refused_with_guidance(tmp_path):
    """The engine reads only the ids its provider declares, so an id nobody declared would
    install a model that is never read. Without a recorded declaration, --uhd-id is required.
    """
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model", declared=None)
    with pytest.raises(PromoteError, match="--uhd-id tflops="):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")
    plan = build_plan(
        model,
        tree,
        "ASM_SDPA_ENGINE",
        role="predict_engine",
        arch="gfx950",
        uhd_ids={"tflops": NEW},
    )
    assert plan.descriptor_id == NEW


@pytest.mark.parametrize(
    "declared,requested",
    [(OTHER, None), (None, {"tflops": OTHER}), (NEW, {"tflops": OTHER})],
)
def test_an_opaque_model_under_an_undeclared_id_is_refused(
    tmp_path, declared, requested
):
    """Promote never renames a model: one trained under another id is refused, and so is a
    --uhd-id that contradicts the id the engine itself declared at collection."""
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model", declared=declared)
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match=OTHER):
        build_plan(
            model,
            tree,
            "ASM_SDPA_ENGINE",
            role="predict_engine",
            arch="gfx950",
            uhd_ids=requested,
        )
    assert _files(tmp_path) == before


def test_repromoting_an_identical_opaque_model_is_idempotent(tmp_path):
    """D2: the declared UUID is one model. The same content promoted again -- for the same
    or another arch directory -- binds what is installed and writes nothing new."""
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model")
    _apply(
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")
    )
    installed = _files(tree)
    plan = build_plan(
        model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="default"
    )
    _apply(plan)
    assert (plan.copies, plan.write_descriptor) == ([], False)
    assert _files(tree) == installed


def _shipped_asm_model(tree, arches, metric="tflops"):
    """A fixed-id model where the build ships it: beside the provider, not where promote
    would put it, under the provider's own file names."""
    directory = tree / "asm_sdpa_engine" / "descriptors" / "predict_engine" / "gfx950"
    artifact = _weights(signature=OPAQUE_SIGNATURE, training_arches=arches)
    directory.mkdir(parents=True)
    (directory / "asm_sdpa_engine_tflops.bin").write_bytes(artifact)
    _write(
        directory / "asm_sdpa_engine_tflops.uhd.json",
        {
            "version": "1.0",
            "id": NEW,
            "name": "shipped",
            "adapter": "tree_data",
            "objective": {"tflops": "max", "time": "min"}[metric],
            "features_signature": list(OPAQUE_SIGNATURE),
            "features_hash": _hash(OPAQUE_SIGNATURE),
            "trained_against": {"selector_revision": "aiter-fwd-1"},
            "score": {"metric": metric, "calibrated": True, "transform": "identity"},
            "tree_data": {
                "artifact": "asm_sdpa_engine_tflops.bin",
                "hash": hashlib.sha256(artifact).hexdigest(),
            },
        },
    )
    return directory


def test_a_retrained_fixed_id_model_replaces_the_shipped_one_where_it_is_installed(
    tmp_path,
):
    """The provider declares the UUID per arch and metric, so the model installed under it
    -- for this metric, trained for this arch -- is this slot's model wherever it lies.
    Promote replaces it in place rather than refusing it for not being at the path promote
    would have chosen, which left a hand-copy as the only way to ship an ASM retrain."""
    tree = _tree(tmp_path / "tree")
    shipped = _shipped_asm_model(tree, ["gfx950"])
    model = _opaque_model(tmp_path / "model")
    retrained = _weights(2.0, signature=OPAQUE_SIGNATURE, training_arches=["gfx950"])
    (model / "model.bin").write_bytes(retrained)

    _apply(
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")
    )

    assert (shipped / "asm_sdpa_engine_tflops.bin").read_bytes() == retrained
    installed = _read(shipped / "asm_sdpa_engine_tflops.uhd.json")
    assert installed["tree_data"] == {
        "artifact": "asm_sdpa_engine_tflops.bin",
        "hash": hashlib.sha256(retrained).hexdigest(),
    }
    assert [path for path in tree.rglob("*.uhd.json") if _read(path)["id"] == NEW] == [
        shipped / "asm_sdpa_engine_tflops.uhd.json"
    ], "one document per declared id"


@pytest.mark.parametrize(
    "arches,metric",
    [(["gfx942"], "tflops"), (["gfx942", "gfx950"], "tflops"), (["gfx950"], "time")],
)
def test_a_fixed_id_model_serving_another_slot_is_not_replaced(
    tmp_path, arches, metric
):
    """In-place replacement is for this slot's model only: an installed model under the id
    that another arch still scores with, or that estimates another metric, is refused.
    """
    tree = _tree(tmp_path / "tree")
    _shipped_asm_model(tree, arches, metric)
    model = _opaque_model(tmp_path / "model")
    (model / "model.bin").write_bytes(
        _weights(2.0, signature=OPAQUE_SIGNATURE, training_arches=["gfx950"])
    )
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match="already installed"):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")
    assert _files(tmp_path) == before


@pytest.mark.parametrize("config", [{}, {"x": 1}])
def test_a_custom_library_configuration_is_promoted_only_when_empty(tmp_path, config):
    """UhdParser admits an omitted or empty `custom_library.config` and refuses any other,
    before the library is loaded; promote must not install what the engine refuses."""
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model")
    doc = _read(model / "heuristic.uhd.json")
    doc["adapter"] = "custom_library"
    doc.pop("tree_data")
    doc["custom_library"] = {
        "library": "scorer.dll",
        "symbol": "score",
        "config": config,
    }
    _write(model / "heuristic.uhd.json", doc)
    (model / "scorer.dll").write_bytes(b"MZ")
    if config:
        with pytest.raises(PromoteError, match="configuration is not supported"):
            build_plan(
                model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950"
            )
    else:
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")


def test_an_engine_with_no_catalog_cannot_be_given_a_catalog_ranker(tmp_path):
    """sort_kernel_catalog ranks an engine's own enumerated configurations. An engine that
    publishes none has nothing for that model to order, so the role is refused here rather
    than installed and silently never consulted."""
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model")
    (model / "train_manifest.json").unlink()
    with pytest.raises(PromoteError, match="catalog it does not own"):
        build_plan(
            model, tree, "ASM_SDPA_ENGINE", role="sort_kernel_catalog", arch="gfx950"
        )


def _corpus(root, rows):
    """The corpus `generate` stages beside the model directory it trains into."""
    return _write(root / "corpus.json", rows)


def _row(kernel, **overrides):
    return {
        "benchmark": "graph-7",
        "device": "board",
        "kernel": kernel,
        "engine": 7,
        "succeeded": True,
        "is_valid": True,
        "numerically_valid": True,
        "validation": "agrees_with_catalog: 3 of 3 cross-checked candidates produced this output",
        "robustMeanMs": 2.5,
        "avgTimeMs": 2.6,
    } | overrides


def test_emission_is_refused_while_a_candidate_carries_an_invalid_marker(tmp_path):
    """RFC 0019 §13.4: package emission fails while any invalid marker is unresolved.

    The model cannot stand in for this. §13.2: the scorer cannot exclude a candidate, so a
    kernel that is applicable and incorrect stays selectable through every path that does
    not consult the model -- a knob pin, a winner that fails to build, any `static_order`
    fallback. A learned demotion is a preference, and this is the last stage able to refuse.
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _corpus(
        tmp_path,
        [
            _row("kernel-a"),
            _row(
                "kernel-b",
                numerically_valid=False,
                validation="output_mismatch: tensor 'Y' element 2 is 9.9e+01",
                robustMeanMs=None,
                avgTimeMs=None,
            ),
        ],
    )

    with pytest.raises(PromoteError) as refusal:
        build_plan(model, tree)

    # Named, because §13.2's remedy is a pack edit the author has to make by hand and
    # "the corpus contains an invalid candidate" does not say which kernel or which problem.
    assert "kernel-b" in str(refusal.value) and "graph-7" in str(refusal.value)
    assert "output_mismatch" in str(refusal.value)
    # The remedy the RFC gives, so the message is actionable on its own.
    assert "UMD" in str(refusal.value) and "UKD" in str(refusal.value)


def test_an_invalid_candidate_blocks_promotion_without_being_erased(tmp_path):
    """§13.2 keeps the invalid rows "in the dataset for diagnostics either way".

    Promotion refusing and the corpus retaining the row are complementary: the refusal
    stops the pack shipping, and the row is how the author finds out what to change. A
    gate that pruned the corpus to clear itself would destroy that evidence.
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    corpus = _corpus(
        tmp_path,
        [
            _row(
                "kernel-b",
                numerically_valid=False,
                validation="output_mismatch: tensor 'Y'",
            )
        ],
    )
    before = _read(corpus)

    with pytest.raises(PromoteError):
        build_plan(model, tree)

    assert _read(corpus) == before


def test_a_clean_corpus_promotes_and_an_undecided_one_promotes_with_a_warning(tmp_path):
    """Only a decided failure blocks; an undecided verdict is reported, never silent.

    Open Question 19(a) leaves the per-op reference open, so a corpus of nulls is today's
    expected state and refusing it would block every promotion. It still has to be visible:
    "we could not check" must not reach an author looking the same as "we checked".
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _corpus(tmp_path, [_row("kernel-a"), _row("kernel-b")])
    assert build_plan(model, tree).warnings == []

    _corpus(
        tmp_path,
        [
            _row("kernel-a"),
            _row(
                "kernel-b",
                numerically_valid=None,
                validation="no_reference: one candidate ran",
            ),
        ],
    )
    warnings = build_plan(model, tree).warnings
    assert len(warnings) == 1 and "1 of 2" in warnings[0]


def test_a_model_with_no_visible_corpus_reports_that_it_was_not_gated(tmp_path):
    """A promotion that could not read the markers says so rather than passing quietly.

    Refusing outright would make an archived model unpromotable, and staying silent would
    make an ungated promotion indistinguishable from a gated one -- which is the failure
    §13.2 names: a missing check must never read as a passing check.
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)

    warnings = build_plan(model, tree).warnings

    assert len(warnings) == 1 and "--corpus" in warnings[0]


def test_an_explicit_corpus_outranks_the_sibling_default(tmp_path):
    """`--corpus` names the corpus an archived model was trained from.

    Without precedence a stale `corpus.json` left beside the model directory would decide
    the gate, and a clean leftover file would wave through the run that found the defect.
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _corpus(tmp_path, [_row("kernel-a")])
    archived = _write(
        tmp_path / "archive" / "corpus.json",
        [
            _row(
                "kernel-b",
                numerically_valid=False,
                validation="output_mismatch: tensor 'Y'",
            )
        ],
    )

    with pytest.raises(PromoteError, match="kernel-b"):
        build_plan(model, tree, corpus=archived)


def test_a_failed_verdict_spelled_as_text_still_refuses_emission(tmp_path):
    """Promotion reads the verdict with the reader every other entrance uses, so a corpus
    that carries `numerically_valid` as CSV text is gated like one carrying a boolean,
    rather than a known-wrong `"False"` passing as neither failed nor unchecked."""
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    _corpus(
        tmp_path,
        [_row("kernel-b", numerically_valid="False", validation="output_mismatch: Y")],
    )
    with pytest.raises(PromoteError, match="kernel-b"):
        build_plan(model, tree)


MATCH_A = "11be5fe7-02a7-4ec2-b79c-e849951f8c24"
MATCH_B = "22be5fe7-02a7-4ec2-b79c-e849951f8c24"


def _packs(tree):
    """One pack per arch, each with its own matcher, as a multi-arch engine ships them."""
    _write(tree / "a.umd.json", {"version": "1.0", "id": MATCH_A})
    _write(tree / "b.umd.json", {"version": "1.0", "id": MATCH_B})
    for arch, matcher, pack in (
        ("gfx942", MATCH_A, "33be5fe7-02a7-4ec2-b79c-e849951f8c24"),
        ("gfx950", MATCH_B, "44be5fe7-02a7-4ec2-b79c-e849951f8c24"),
    ):
        _write(
            tree / f"{arch}.kdp.json",
            {
                "version": "1.0",
                "id": pack,
                "engine": UED,
                "arch": [arch],
                "matchers": [matcher],
            },
        )
    # The gfx950 key binds a metric-less incumbent this promotion replaces; it must be
    # installed for its metric to be read.
    _installed(tree / "other.uhd.json", OTHER)
    return tree


def _multi_arch_model(root, tree, leaf=1.0):
    """A model collected over both arches' packs: its provenance records both matchers."""
    provenance = snapshot_provenance(tree, arch="default")
    assert [item["id"] for item in provenance["umd"]] == [MATCH_A, MATCH_B]
    model = _model(root, tree, provenance=provenance)
    _write(
        root / "train_manifest.json",
        {"training_arches": ["gfx942", "gfx950"], "trained_against": provenance},
    )
    (root / "model.bin").write_bytes(_weights(leaf))
    return model


def test_one_model_over_two_arches_packs_binds_both_arch_keys_as_one_model(tmp_path):
    """R5 + D2: each arch ignores the matchers of the other arch's pack, so the model is
    valid on both; the second promotion binds the installed copy rather than a second one,
    and promoting it again changes nothing."""
    tree = _packs(_tree(tmp_path / "tree"))
    model = _multi_arch_model(tmp_path / "model", tree)
    _apply(build_plan(model, tree, arch="gfx942"))
    first = _files(tree)
    _apply(build_plan(model, tree, arch="gfx950"))

    ued = _read(tree / "engine.ued.json")
    assert (
        ued["sort_kernel_catalog"]["gfx942"],
        ued["sort_kernel_catalog"]["gfx950"],
    ) == ([NEW], [NEW])
    assert (
        len([path for path in tree.rglob("*.uhd.json") if _read(path)["id"] == NEW])
        == 1
    )
    second = _files(tree)
    assert {path for path in second if second[path] != first.get(path)} == {
        Path("engine.ued.json")
    }
    _apply(build_plan(model, tree, arch="gfx950"))
    assert _files(tree) == second


@pytest.mark.parametrize("change", ["incompatible", "unowned"])
def test_an_arch_relevant_or_unowned_recorded_matcher_is_still_refused(
    tmp_path, change
):
    """Only other arches' matchers are ignored. A recorded matcher this arch's pack uses
    must still be compatible, and one no pack of the engine owns is a removed dependency.
    """
    tree = _packs(_tree(tmp_path / "tree"))
    model = _multi_arch_model(tmp_path / "model", tree)
    if change == "incompatible":
        _write(
            tree / "b.umd.json", {"version": "1.0", "id": MATCH_B, "revision": "2.0"}
        )
        build_plan(model, tree, arch="gfx942")
        with pytest.raises(PromoteError, match=MATCH_B):
            build_plan(model, tree, arch="gfx950")
    else:
        pack = _read(tree / "gfx950.kdp.json")
        pack["matchers"] = []
        _write(tree / "gfx950.kdp.json", pack)
        for arch in ("gfx942", "gfx950"):
            with pytest.raises(PromoteError, match=MATCH_B):
                build_plan(model, tree, arch=arch)


def test_one_id_with_different_content_is_refused_under_another_arch(tmp_path):
    """D2: a UUID bound under several arch keys is one model, so a different model cannot
    take the id for one arch while another arch keeps scoring with the installed one."""
    tree = _packs(_tree(tmp_path / "tree"))
    model = _multi_arch_model(tmp_path / "model", tree)
    _apply(build_plan(model, tree, arch="gfx942"))
    (model / "model.bin").write_bytes(_weights(2.0))
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match="different content"):
        build_plan(model, tree, arch="gfx950")
    assert _files(tmp_path) == before


def test_promote_writes_the_artifact_digest_the_trainer_left_out(tmp_path):
    """R6: without a declared hash the runtime's model identity ignores the artifact bytes."""
    tree = _tree(tmp_path / "tree")
    plan = build_plan(_model(tmp_path / "model", tree), tree)
    _apply(plan)
    assert (
        _read(plan.destination_descriptor)["tree_data"]["hash"]
        == hashlib.sha256(_weights()).hexdigest()
    )


@pytest.mark.parametrize(
    "defect",
    [
        "identifier",
        "declared_hash",
        "features_hash",
        "truncated",
        "unterminated_string",
        "arity",
    ],
)
def test_an_artifact_the_runtime_would_refuse_is_never_installed(tmp_path, defect):
    """R8: promote applies TreeDataAdapter's load checks to the bytes it installs: the
    FlatBuffers verifier's structure (here, a string with no NUL terminator, which every
    accessor still decodes), and the model's arity against its signature, which
    EnginePredictor and UhdKernelHeuristic compare whatever the features_hash says."""
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    data = _weights()
    if defect == "identifier":
        data = data[:4] + b"NOPE" + data[8:]
    elif defect == "features_hash":
        data = _weights(features_hash="sha256:" + "1" * 16)
    elif defect == "truncated":
        data = data[:12]
    elif defect == "unterminated_string":
        end = data.index(_hash().encode()) + len(_hash())
        data = data[:end] + b"!" + data[end + 1 :]
    elif defect == "arity":
        data = _weights(num_features=len(KERNEL_SIGNATURE) + 1)
    else:
        doc = _read(model / "heuristic.uhd.json")
        doc["tree_data"]["hash"] = hashlib.sha256(b"other bytes").hexdigest()
        _write(model / "heuristic.uhd.json", doc)
    (model / "model.bin").write_bytes(data)
    before = _files(tmp_path)
    with pytest.raises(PromoteError, match="refuse"):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def test_a_grouped_artifact_is_refused_for_engine_prediction_only(tmp_path):
    """R9: the runtime scores an L1 model's root ensemble alone, so a grouped artifact has no
    engine-level meaning; as a catalog ranker it is the two-layer model it was built as.
    """
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "opaque")
    (model / "model.bin").write_bytes(
        _weights(grouped=True, signature=OPAQUE_SIGNATURE)
    )
    with pytest.raises(PromoteError, match="grouped"):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx950")
    catalog = _model(tmp_path / "catalog", tree)
    (catalog / "model.bin").write_bytes(_weights(grouped=True))
    build_plan(catalog, tree)


#: The gfx950 attention-dense engine as shipped when its rankers were found unloadable: a
#: UED exposing two knobs over a KMD whose other fields its matcher binds from the graph.
ATTENTION_UED = "4b3a0123-578f-4e9c-a965-b010a18ff107"
ATTENTION_KMD = "589bc6c6-d94e-4400-95d7-3e517f9b6b67"
ATTENTION_FIELDS = (
    "dtype",
    "head_size",
    "num_query_heads",
    "num_kv_heads",
    "causal",
    "ragged",
    "sliding_window",
    "batch",
    "seqlen_q",
    "seqlen_kv",
    "block_m",
    "block_n",
)
#: Every `$kernel.*` axis the shipped gfx950 rankers (tflops and time alike) read, plus two of
#: their graph columns -- one of them the problem-side twin of `$kernel.causal`.
SHIPPED_RANKER_SIGNATURE = (
    "$kernel.block_m",
    "$kernel.block_n",
    "$kernel.causal",
    "$kernel.dtype",
    "$kernel.head_size",
    "$kernel.num_kv_heads",
    "$kernel.num_query_heads",
    "$gfx950_attention_dense.causal",
    "$graph.flops",
)
SHIPPED_ENCODING = json.dumps({"$kernel.dtype": {"BF16": 0, "FP16": 1}})


def _attention_tree(root):
    _write(
        root / "gfx950_attention_dense.ued.json",
        {
            "version": "1.0",
            "id": ATTENTION_UED,
            "name": "hipkernel:Gfx950AttentionDense",
            "metadata": ATTENTION_KMD,
            "knobs": ["block_m", "block_n"],
        },
    )
    _write(
        root / "gfx950_attention_dense.kmd.json",
        {
            "version": "1.0",
            "id": ATTENTION_KMD,
            "name": "Gfx950AttentionDense variant fields",
            "fields": [
                {"name": name, "type": "string" if name == "dtype" else "int"}
                for name in ATTENTION_FIELDS
            ],
        },
    )
    return root


def _attention_ranker(root, tree, metric, signature, encoding=None):
    provenance = snapshot_provenance(tree, arch="gfx950")
    doc = {
        "version": "1.0",
        "id": NEW,
        "name": "ranker",
        "adapter": "tree_data",
        "objective": {"tflops": "max", "time": "min"}[metric],
        "score": {"metric": metric, "calibrated": True, "transform": "log1p"},
        "features_signature": list(signature),
        "features_hash": _hash(signature, encoding),
        "trained_against": provenance,
        "tree_data": {"artifact": "model.bin"},
    }
    if encoding:
        doc["categorical_encoding"] = json.loads(encoding)
    _write(root / "heuristic.uhd.json", doc)
    _write(
        root / "train_manifest.json",
        {"training_arches": ["gfx950"], "trained_against": provenance},
    )
    (root / "model.bin").write_bytes(
        _weights(signature=signature, features_hash=_hash(signature, encoding))
    )
    return root


@pytest.mark.parametrize("metric", ["tflops", "time"])
def test_the_shipped_gfx950_rankers_signature_is_refused_for_reading_unexposed_kernel_fields(
    tmp_path, metric
):
    """RFC 0019 §6.3 check 2: the runtime drops a model whose `$kernel.*` axes are not knobs
    of its UED, and ranks by declared order. Promote accepted exactly that model -- generation
    fitted it on the collection UED, which exposes every KMD field -- so the shipped rankers
    installed cleanly and were never used. The same model reading the graph's twin columns
    and only the shipped knobs installs."""
    tree = _attention_tree(tmp_path / "tree")
    shipped = _attention_ranker(
        tmp_path / "shipped", tree, metric, SHIPPED_RANKER_SIGNATURE, SHIPPED_ENCODING
    )
    before = _files(tmp_path)
    with pytest.raises(PromoteError) as refusal:
        build_plan(shipped, tree, arch="gfx950")
    assert "[causal, dtype, head_size, num_kv_heads, num_query_heads]" in str(
        refusal.value
    )
    assert "[block_m, block_n]" in str(refusal.value)
    assert _files(tmp_path) == before

    admissible = tuple(
        entry
        for entry in SHIPPED_RANKER_SIGNATURE
        if not entry.startswith("$kernel.")
        or entry in ("$kernel.block_m", "$kernel.block_n")
    )
    build_plan(
        _attention_ranker(tmp_path / "admissible", tree, metric, admissible),
        tree,
        arch="gfx950",
    )


def test_a_kernel_axis_the_kmd_does_not_declare_is_refused_even_when_it_is_a_knob(
    tmp_path,
):
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued["knobs"].append("waves")
    _write(tree / "engine.ued.json", ued)
    with pytest.raises(
        PromoteError, match=r"\[waves\], which the KMD does not declare"
    ):
        build_plan(_model(tmp_path / "model", tree, signature=("$kernel.waves",)), tree)


def test_a_features_hash_that_is_not_its_signatures_digest_is_refused(tmp_path):
    """The loader recomputes the digest from the signature and drops the model on a mismatch.
    Comparing the artifact's stored copy with the descriptor's proves only that one run
    stamped both: a signature edited after training kept both stale copies in agreement.
    """
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    stale = _read(model / "heuristic.uhd.json")
    stale["features_signature"] = ["$kernel.tile_m"]
    _write(model / "heuristic.uhd.json", stale)
    before = _files(tmp_path)
    with pytest.raises(
        PromoteError,
        match=rf"{_hash()} is not the digest .*{_hash(('$kernel.tile_m',))}",
    ):
        build_plan(model, tree)
    assert _files(tmp_path) == before


def test_an_engine_level_model_for_a_descriptor_engine_must_record_its_selector_revision(
    tmp_path,
):
    """The runtime holds a descriptor-backed engine's L1 model to the selector revision it
    was measured under and reports UNAVAILABLE without one, so installing it ships a model
    the engine never scores."""
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued.pop("predict_engine")
    _write(tree / "engine.ued.json", ued)

    def l1(root, provenance):
        model = _model(root, tree, provenance=provenance, signature=("$graph.flops",))
        document = _read(model / "heuristic.uhd.json")
        document["score"] = {
            "metric": "tflops",
            "calibrated": True,
            "transform": "identity",
        }
        _write(model / "heuristic.uhd.json", document)
        return model

    provenance = snapshot_provenance(tree, arch="gfx942")
    with pytest.raises(PromoteError, match="selector_revision"):
        build_plan(l1(tmp_path / "bare", provenance), tree, role="predict_engine")
    build_plan(
        l1(tmp_path / "recorded", {**provenance, "selector_revision": "provider-1"}),
        tree,
        role="predict_engine",
    )


def test_uhd_id_names_the_model_being_promoted_and_never_renames_it(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree, metric="time")
    _installed(tree / "old.uhd.json", OLD, {"metric": "tflops", "calibrated": False})
    assert (
        build_plan(
            model, tree, uhd_ids=parse_uhd_ids([f"time={NEW}", f"tflops={OTHER}"])
        ).descriptor_id
        == NEW
    )
    with pytest.raises(PromoteError, match="never renames"):
        build_plan(model, tree, uhd_ids=parse_uhd_ids([f"time={OTHER}"]))
    with pytest.raises(PromoteError, match="estimates time"):
        build_plan(model, tree, uhd_ids=parse_uhd_ids([f"tflops={NEW}"]))


@pytest.mark.parametrize(
    "values",
    [
        [NEW, f"time={OTHER}"],
        [f"latency={NEW}"],
        [f"time={NEW}", f"time={OTHER}"],
        ["time=not-a-uuid"],
    ],
)
def test_malformed_uhd_id_lists_are_refused(values):
    with pytest.raises(ValueError):
        parse_uhd_ids(values)
