# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Promotion preserves every untargeted model and refuses unsafe writes."""
from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path

import pytest

from uhd_gen.promote import PromoteError, _apply, add_promote_arguments, build_plan, run_promote
from uhd_gen.provenance import snapshot_provenance

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
OLD = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
NEW = "cf37fa30-32dc-4a21-a008-68ef5e0d30a6"
OTHER = "edc1d5b4-6f12-4a40-a749-403966474bc9"


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree(root):
    _write(root / "engine.ued.json", {"version": "1.0", "id": UED, "name": "test:engine",
           "metadata": KMD, "knobs": ["block_size", "tile_m"],
           "sort_kernel_catalog": {"gfx942": OLD, "gfx950": OTHER, "default": OTHER},
           "predict_engine_tflops": {"gfx942": OTHER}})
    _write(root / "metadata.kmd.json", {"version": "1.0", "id": KMD, "fields": []})
    return root


def _model(root, tree, *, identity=NEW, artifact="model.bin", provenance=None):
    provenance = provenance if provenance is not None else snapshot_provenance(tree, arch="gfx942")
    doc = {"version": "1.0", "id": identity, "name": "model", "adapter": "tree_data",
           "objective": "max", "features_signature": ["$kernel.block_size"],
           "features_hash": "sha256:" + "0" * 16, "trained_against": provenance,
           "tree_data": {"artifact": artifact}}
    _write(root / "heuristic.uhd.json", doc)
    _write(root / "train_manifest.json", {"training_arches": ["gfx942"], "trained_against": provenance})
    payload = root / artifact
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"incoming model")
    return root


def _files(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_promotion_updates_only_the_requested_arch_and_role(tmp_path):
    tree = _tree(tmp_path / "tree")
    original = _read(tree / "engine.ued.json")
    model = _model(tmp_path / "model", tree)
    _apply(build_plan(model, tree))
    expected = copy.deepcopy(original)
    expected["sort_kernel_catalog"]["gfx942"] = NEW
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
    assert artifact.read_bytes() == b"incoming model"


@pytest.mark.parametrize("binding", ["role", "architecture", "engine"])
def test_default_filenames_preserve_other_model_bindings(tmp_path, binding):
    tree = _tree(tmp_path / "tree")
    ued = _read(tree / "engine.ued.json")
    ued["sort_kernel_catalog"] = {"gfx942": OLD}
    ued.pop("predict_engine_tflops")
    _write(tree / "engine.ued.json", ued)
    first = _model(tmp_path / "first", tree)
    (first / "model.bin").write_bytes(b"first model weights")
    first_plan = build_plan(first, tree)
    _apply(first_plan)
    first_descriptor = first_plan.destination_descriptor.read_bytes()
    first_document = _read(first_plan.destination_descriptor)
    first_artifact = first_plan.destination_descriptor.parent / first_document["tree_data"]["artifact"]

    engine, role, arch = "test:engine", "sort_kernel_catalog", "gfx942"
    if binding == "engine":
        engine = "test:second"
        second_ued = copy.deepcopy(ued)
        second_ued.update(id=UED[:-1] + "3", name=engine)
        _write(tree / "second.ued.json", second_ued)
    elif binding == "architecture":
        arch = "gfx950"
    else:
        role = "predict_engine_tflops"
    provenance = snapshot_provenance(tree, engine=engine, arch=arch)
    second = _model(tmp_path / "second", tree, identity=OTHER, provenance=provenance)
    document = _read(second / "heuristic.uhd.json")
    if binding == "role":
        document.update(features_signature=["$graph.flops"],
                        score={"units": "tflops", "calibrated": True, "transform": "identity"})
    _write(second / "heuristic.uhd.json", document)
    _write(second / "train_manifest.json", {"training_arches": [arch], "trained_against": provenance})
    (second / "model.bin").write_bytes(b"second model weights")
    second_plan = build_plan(second, tree, engine=engine, role=role, arch=arch)
    _apply(second_plan)

    assert first_plan.destination_descriptor.read_bytes() == first_descriptor
    assert first_artifact.read_bytes() == b"first model weights"
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == NEW
    installed = {
        _read(path)["id"]: (path, _read(path))
        for path in tree.rglob("*.uhd.json")
    }
    selected = _read(second_plan.ued_path)[role][arch]
    path, document = installed[selected]
    assert (path.parent / document["tree_data"]["artifact"]).read_bytes() == b"second model weights"

    (second / "model.bin").write_bytes(b"replacement weights")
    _apply(build_plan(second, tree, engine=engine, role=role, arch=arch))
    assert first_artifact.read_bytes() == b"first model weights"
    assert (path.parent / document["tree_data"]["artifact"]).read_bytes() == b"replacement weights"

def test_dry_run_makes_no_writes(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    before = _files(tmp_path)
    parser = argparse.ArgumentParser()
    add_promote_arguments(parser)
    args = parser.parse_args(["--model-dir", str(model), "--descriptor-tree", str(tree), "--dry-run"])
    assert run_promote(args) == 0
    assert _files(tmp_path) == before


@pytest.mark.parametrize("role,arch", [("sort_kernel_catalog", "gfx950"), ("predict_engine_tflops", "gfx942")])
def test_shared_identity_cannot_be_rewritten_from_another_binding(tmp_path, role, arch):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree, identity=OTHER)
    incumbent = _read(model / "heuristic.uhd.json")
    incumbent["id"] = OTHER
    _write(tree / "heuristic.uhd.json", incumbent)
    (tree / "model.bin").write_bytes(b"untargeted model")
    ued = _read(tree / "engine.ued.json")
    ued["sort_kernel_catalog"] = {"gfx942": OLD}
    ued.pop("predict_engine_tflops")
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


@pytest.mark.parametrize("missing", ["trained_against", "features_hash", "features_signature", "objective", "tree_data", "name", "version"])
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
    manifest["dropped_constant_features"] = ["kernel.tile_m"]
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
    ued.pop("predict_engine_tflops")
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
    _apply(build_plan(model, tree, arch="default"))
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["default"] == NEW


def test_training_architecture_cannot_be_silently_retargeted(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    with pytest.raises(PromoteError):
        build_plan(model, tree, arch="gfx950")


def test_additive_semantic_revision_accepts_without_rewriting_training_snapshot(tmp_path):
    tree = _tree(tmp_path / "tree")
    model = _model(tmp_path / "model", tree)
    trained = _read(model / "heuristic.uhd.json")["trained_against"]
    ued = _read(tree / "engine.ued.json")
    ued["revision"] = "1.12"
    _write(tree / "engine.ued.json", ued)
    plan = build_plan(model, tree)
    _apply(plan)
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == NEW
    assert _read(plan.destination_descriptor)["trained_against"] == trained


@pytest.mark.parametrize("payload", ["missing.bin", "../outside.bin", "metadata.kmd.json"])
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
    return main(["train", "--input", str(csv), "--features", "kernel.block_size", "--target", "tflops",
                 "--output-dir", str(tmp_path / "model"), "--provenance", str(provenance),
                 "--num-boost-round", "10", "--early-stopping", "5", *extra])


def test_train_retains_explicit_model_identity(tmp_path):
    assert _train(tmp_path, "--uhd-id", NEW) == 0
    assert _read(tmp_path / "model" / "heuristic.uhd.json")["id"] == NEW
    assert _read(tmp_path / "model" / "train_manifest.json")["uhd_id"] == NEW


def test_train_mints_identity_when_omitted(tmp_path):
    assert _train(tmp_path) == 0
    identity = _read(tmp_path / "model" / "heuristic.uhd.json")["id"]
    assert str(uuid.UUID(identity)) == identity


@pytest.mark.parametrize("malformed", ["not-a-uuid", "", "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d6"])
def test_train_rejects_malformed_identity_before_outputs(tmp_path, malformed):
    assert _train(tmp_path, "--uhd-id", malformed) == 1
    assert not (tmp_path / "model" / "heuristic.uhd.json").exists()


# ------------------------------------------------------- engines that own no descriptors


def _opaque_model(root, *, identity=NEW, revision="aiter-fwd-1"):
    """A model for an engine with no UED: trained against a selector revision only."""
    provenance = {"selector_revision": revision}
    doc = {"version": "1.0", "id": identity, "name": "model", "adapter": "tree_data",
           "objective": "max", "features_signature": ["$graph.work"],
           "features_hash": "sha256:" + "0" * 16, "trained_against": provenance,
           "score": {"units": "tflops", "calibrated": True, "transform": "identity"},
           "tree_data": {"artifact": "model.bin"}}
    _write(root / "heuristic.uhd.json", doc)
    _write(root / "train_manifest.json",
           {"training_arches": ["gfx950"], "trained_against": provenance,
            "role": "predict_engine_tflops"})
    (root / "model.bin").write_bytes(b"incoming model")
    return root


def test_a_model_for_an_engine_with_no_ued_installs_without_touching_a_role_map(tmp_path):
    """AITER and MIOpen own no descriptor set: RFC 0019 4.1 / Open Question 7 binds their
    model by a UUID declared in provider code, so promotion writes the document where the
    loader scans and edits nothing. Before this, `select_engine` ended the run with
    "expected one UED for --engine 'ASM_SDPA_ENGINE', found 0" -- after the model had
    already trained (run 67929588)."""
    tree = _tree(tmp_path / "tree")
    before = _read(tree / "engine.ued.json")
    model = _opaque_model(tmp_path / "model")

    plan = build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine_tflops", arch="gfx950")
    _apply(plan)

    assert plan.ued_path is None
    assert _read(tree / "engine.ued.json") == before, "no role map may change"
    installed = tree / "heuristics" / "ASM_SDPA_ENGINE" / "predict_engine_tflops" / "gfx950"
    assert _read(installed / "heuristic.uhd.json")["id"] == NEW
    assert (installed / "model.bin").read_bytes() == b"incoming model"


def test_an_opaque_model_cannot_take_an_identity_already_installed(tmp_path):
    """The UUID is the whole binding: the provider looks for exactly one, so a second
    document carrying it makes the engine's model ambiguous rather than replaced."""
    tree = _tree(tmp_path / "tree")
    _write(tree / "elsewhere" / "other.uhd.json", {"version": "1.0", "id": NEW, "name": "other",
           "adapter": "tree_data", "objective": "max", "features_signature": ["$graph.work"],
           "features_hash": "sha256:" + "0" * 16, "trained_against": {"selector_revision": "r"},
           "tree_data": {"artifact": "other.bin"}})
    model = _opaque_model(tmp_path / "model")
    with pytest.raises(PromoteError, match="already installed"):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="predict_engine_tflops", arch="gfx950")


def test_an_engine_with_no_catalog_cannot_be_given_a_catalog_ranker(tmp_path):
    """sort_kernel_catalog ranks an engine's own enumerated configurations. An engine that
    publishes none has nothing for that model to order, so the role is refused here rather
    than installed and silently never consulted."""
    tree = _tree(tmp_path / "tree")
    model = _opaque_model(tmp_path / "model")
    (model / "train_manifest.json").unlink()
    with pytest.raises(PromoteError, match="catalog it does not own"):
        build_plan(model, tree, "ASM_SDPA_ENGINE", role="sort_kernel_catalog", arch="gfx950")
