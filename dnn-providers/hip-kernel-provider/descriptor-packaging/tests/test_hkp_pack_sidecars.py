# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A trained UHD names a model file, and the packer has to carry it.

Discovery globs `*.json`, so every other descriptor the packer handles is fully
described by a file it already found. A `tree_data` UHD is the exception: its
`tree_data.artifact` is a path to a model the runtime reads on every candidate
score, and it has to reach the packed tree or the runtime finds the artifact
missing and drops the engine.

These cases pin the three things that can go wrong: the file is not resolved, it
is resolved from the wrong place, or it is resolved and then not staged.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hkp_pack.descriptors import HkpPackError, load_flat_input
from hkp_pack.pipeline import compile_intermediate

pytestmark = pytest.mark.quick


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _model_uhd(artifact: str) -> dict:
    return {
        "version": "1.0",
        "id": "133a8b19-8e34-4f74-86d6-b6495a6483f3",
        "name": "Trained heuristic",
        "adapter": "tree_data",
        "features_signature": ["$kernel.tile_m"],
        "features_hash": "sha256:" + "0" * 16,
        "trained_against": {
            "ued": {"id": "699a8b19-8e34-4f74-86d6-b6495a6483f3", "revision": "1.0"},
            "kmd": {"id": "799a8b19-8e34-4f74-86d6-b6495a6483f3", "revision": "1.0"},
            "umd": [],
        },
        "objective": "max",
        "tree_data": {"artifact": artifact},
    }


def _native_uhd() -> dict:
    return {
        "version": "1.0",
        "id": "233a8b19-8e34-4f74-86d6-b6495a6483f3",
        "name": "Native heuristic",
        "adapter": "native",
        "objective": "max",
        "native": {"symbol": "hipkernel.pointwise.score"},
    }


def _root_with_model_uhd(tmp_path: Path, artifact: str = "model.bin") -> Path:
    """A minimal root: one model UHD and the artifact it names."""
    root = tmp_path / "src"
    _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd(artifact))
    path = root / "pack" / "model.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"HGBM-stub")
    return root


class TestResolution:
    def test_the_named_artifact_becomes_a_sidecar(self, tmp_path: Path):
        flat = load_flat_input(_root_with_model_uhd(tmp_path), log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.name == "model.bin"
        assert sidecar.rel_dir == Path("pack")
        assert sidecar.source.read_bytes() == b"HGBM-stub"

    def test_a_native_uhd_names_a_symbol_not_a_file(self, tmp_path: Path):
        """A native UHD's body names an in-process symbol.

        Resolving it as a path would reject every UHD shipping today, all of
        which are native.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "native.uhd.json", _native_uhd())

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars == []

    def test_a_native_uhd_carries_nothing_from_its_folder(self, tmp_path: Path):
        """Only an adapter that reads a model names a file. A native UHD shares a
        folder with whatever else the pack author put there and must not adopt it.

        This is the case the old convention could not express: carriage was "every
        non-descriptor file beside the UHD", so a stray file shipped to a customer.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "native.uhd.json", _native_uhd())
        (root / "pack" / "unrelated.bin").write_bytes(b"not mine")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars == []

    def test_an_unnamed_file_beside_a_model_uhd_is_not_carried(self, tmp_path: Path):
        """The descriptor names exactly one file, so exactly one travels.

        Training inputs, notes and stale artifacts sit beside a heuristic in
        every tree that exists; none of them belong in a shipped pack.
        """
        root = _root_with_model_uhd(tmp_path)
        (root / "pack" / "training_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert [s.name for s in uhd.sidecars] == ["model.bin"]

    def test_the_artifact_resolves_relative_to_its_own_descriptor(self, tmp_path: Path):
        """Not root-relative, matching how a hip UKD's `source` resolves.

        A root-relative fallback would fire exactly when the descriptor-local
        file is missing, turning a typo into a silent bind to a same-named file
        elsewhere in the tree.
        """
        root = _root_with_model_uhd(tmp_path)
        # A decoy at the root with the same name: resolution must not reach it.
        (root / "model.bin").write_bytes(b"decoy")
        (root / "pack" / "model.bin").write_bytes(b"correct")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars[0].source.read_bytes() == b"correct"

    def test_a_shared_artifact_keeps_its_authored_position(self, tmp_path: Path):
        """`../shared/x` is how one artifact is shared between sibling packs.

        The staged copy has to keep the same relative position, or the authored
        path stops resolving in the packed tree even though it resolved in the
        source.
        """
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.bin"),
        )
        shared = root / "rocKE" / "shared" / "model.bin"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.rel_dir == Path("rocKE/shared")
        assert sidecar.name == "model.bin"


class TestRejection:
    def test_missing_artifact_is_an_error_not_a_warning(self, tmp_path: Path):
        """A UHD packed without its artifact costs the whole engine at runtime.

        DescriptorLoader drops an engine whose model artifact is absent. Catching
        it at pack time reports it against the source tree, where the missing file
        can still be added, rather than on a customer's machine as an engine that
        is simply not there.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("absent.bin"))

        with pytest.raises(HkpPackError, match="payload source not found"):
            load_flat_input(root, log=lambda *_: None)

    def test_an_adapter_that_reads_a_model_must_name_one(self, tmp_path: Path):
        """A `tree_data` UHD with no body is a descriptor that can never score.

        The runtime would take it as absent and drop the engine; the packer sees
        the whole document and can say so against the source tree.
        """
        root = tmp_path / "src"
        doc = _model_uhd("model.bin")
        del doc["tree_data"]
        _write_json(root / "pack" / "heuristic.uhd.json", doc)

        with pytest.raises(HkpPackError):
            load_flat_input(root, log=lambda *_: None)

    def test_artifact_outside_the_root_is_rejected(self, tmp_path: Path):
        """Containment, mirroring the hip source check and the runtime's treeRoot bound.

        An artifact the packer resolved outside the root could not be staged
        anywhere the runtime would find it, and a path that walks out of the
        tree is the descriptor-side half of a path-traversal.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("../../outside.bin"))
        (tmp_path.parent / "outside.bin").write_bytes(b"outside")

        with pytest.raises(HkpPackError, match="payload escapes the source root"):
            load_flat_input(root, log=lambda *_: None)

    def test_escape_is_checked_before_existence(self, tmp_path: Path):
        """An escaping path that also does not exist reports the escape.

        Reporting 'not found' would send an author looking for a missing file
        when the real fault is where they pointed.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("../../nowhere.bin"))

        with pytest.raises(HkpPackError, match="payload escapes the source root"):
            load_flat_input(root, log=lambda *_: None)


class TestIntermediateStaging:
    """compile_intermediate mirrors the authored tree before anything is pruned."""

    def test_sidecar_is_mirrored_into_the_intermediate_tree(self, tmp_path: Path):
        root = _root_with_model_uhd(tmp_path)
        flat = load_flat_input(root, log=lambda *_: None)
        inter_dir = tmp_path / "inter" / "gfx942"

        compile_intermediate(
            flat, root, "gfx942", hipcc=None, inter_arch_dir=inter_dir, log=lambda *_: None
        )

        assert (inter_dir / "pack" / "heuristic.uhd.json").is_file()
        staged = inter_dir / "pack" / "model.bin"
        assert staged.is_file(), "the artifact the UHD names must ride with it"
        assert staged.read_bytes() == b"HGBM-stub"

    def test_sidecar_keeps_its_authored_subpath(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.bin"),
        )
        shared = root / "rocKE" / "shared" / "model.bin"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")
        flat = load_flat_input(root, log=lambda *_: None)
        inter_dir = tmp_path / "inter" / "gfx942"

        compile_intermediate(
            flat, root, "gfx942", hipcc=None, inter_arch_dir=inter_dir, log=lambda *_: None
        )

        # Staged where `../shared/model.bin` still reaches it from the UHD.
        assert (inter_dir / "rocKE" / "shared" / "model.bin").read_bytes() == b"shared"
        assert not (inter_dir / "rocKE" / "attn" / "model.bin").exists()


@pytest.mark.parametrize("missing", ["objective", "features_signature", "features_hash", "trained_against", "tree_data"])
def test_feature_models_require_complete_headers_before_packaging(tmp_path, missing):
    root = _root_with_model_uhd(tmp_path)
    doc = _model_uhd("model.bin")
    del doc[missing]
    _write_json(root / "pack" / "heuristic.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


def test_custom_library_uses_library_and_carries_the_shared_object(tmp_path):
    root = tmp_path / "src"
    doc = _native_uhd()
    del doc["native"]
    doc["adapter"] = "custom_library"
    doc["custom_library"] = {"library": "lib/model.so", "symbol": "score"}
    _write_json(root / "custom.uhd.json", doc)
    (root / "lib").mkdir()
    (root / "lib" / "model.so").write_bytes(b"shared object")
    flat = load_flat_input(root, log=lambda *_: None)
    assert flat.descriptors[0].sidecars[0].source.read_bytes() == b"shared object"
    doc["custom_library"]["artifact"] = doc["custom_library"].pop("library")
    _write_json(root / "custom.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


def test_explicit_semantic_revision_does_not_change_format_admission(tmp_path):
    root = tmp_path / "src"
    _write_json(root / "metadata.kmd.json", {
        "version": "1.0", "revision": "12.34",
        "id": "799a8b19-8e34-4f74-86d6-b6495a6483f3", "name": "metadata",
        "fields": [{"name": "block_size", "type": "int"}],
    })
    flat = load_flat_input(root, log=lambda *_: None)
    assert flat.descriptors[0].doc["revision"] == "12.34"
    doc = _native_uhd()
    doc["version"] = "2.0"
    _write_json(root / "native.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


@pytest.mark.parametrize("mutation", [
    "valid", "missing_objective", "missing_provenance", "legacy_provenance",
    "wrong_body", "two_bodies", "legacy_derived", "bare_feature", "empty_feature",
    "invalid_hash", "unknown_header", "unsupported_format",
])
def test_packaging_and_canonical_schema_agree_on_uhd_headers(tmp_path, mutation):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = next(parent / "projects/hipdnn/plugin_sdk/schemas/uhd.schema.json"
                       for parent in Path(__file__).resolve().parents
                       if (parent / "projects/hipdnn/plugin_sdk/schemas/uhd.schema.json").is_file())
    validator = jsonschema.Draft7Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    doc = _model_uhd("model.bin")
    if mutation == "missing_objective":
        del doc["objective"]
    elif mutation == "missing_provenance":
        del doc["trained_against"]
    elif mutation == "legacy_provenance":
        doc["trained_against"] = {"ued": "1.0", "kmd": "1.0", "umd": "1.0"}
    elif mutation == "wrong_body":
        doc["table"] = doc.pop("tree_data")
    elif mutation == "two_bodies":
        doc["native"] = {"symbol": "score"}
    elif mutation == "legacy_derived":
        doc["derived"] = {"tiles": {"ceil_div": [100, "$kernel.tile_m"]}}
    elif mutation == "bare_feature":
        doc["features_signature"] = ["kernel.tile_m"]
    elif mutation == "empty_feature":
        doc["features_signature"] = []
    elif mutation == "invalid_hash":
        doc["features_hash"] = "sha256:not-hex"
    elif mutation == "unknown_header":
        doc["unknown"] = True
    elif mutation == "unsupported_format":
        doc["version"] = "1.1"
    root = tmp_path / "src"
    _write_json(root / "heuristic.uhd.json", doc)
    (root / "model.bin").write_bytes(b"artifact")
    valid = mutation == "valid"
    assert validator.is_valid(doc) == valid
    if valid:
        flat = load_flat_input(root, log=lambda *_: None)
        assert flat.descriptors[0].sidecars[0].source.read_bytes() == b"artifact"
    else:
        with pytest.raises(HkpPackError):
            load_flat_input(root, log=lambda *_: None)


def test_all_role_and_arch_models_remain_reachable_when_packaging(tmp_path, main_fixture):
    from hkp_pack.descriptors import reachable_generic_ids
    root = tmp_path / "src"
    shutil.copytree(main_fixture, root)
    ued_path = root / "pointwise.ued.json"
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    original = next(iter(ued["sort_kernel_catalog"].values()))
    second = _native_uhd()
    third = _native_uhd()
    third["id"] = "333a8b19-8e34-4f74-86d6-b6495a6483f3"
    _write_json(root / "estimate.uhd.json", second)
    _write_json(root / "generator.uhd.json", third)
    ued["sort_kernel_catalog"]["gfx950"] = second["id"]
    ued["predict_engine_tflops"] = {"gfx942": second["id"]}
    ued["predict_applicable_kernels"] = {"default": third["id"]}
    _write_json(ued_path, ued)
    flat = load_flat_input(root, log=lambda *_: None)
    retained = reachable_generic_ids(flat, flat.kdps())
    assert {original, second["id"], third["id"]} <= retained


@pytest.mark.parametrize("legacy", [
    {"heuristic": "233a8b19-8e34-4f74-86d6-b6495a6483f3"},
    {"sort_kernel_catalog": "233a8b19-8e34-4f74-86d6-b6495a6483f3"},
])
def test_packaging_rejects_legacy_heuristic_spellings(tmp_path, legacy):
    root = tmp_path / "src"
    _write_json(root / "engine.ued.json", {
        "version": "1.0", "id": "699a8b19-8e34-4f74-86d6-b6495a6483f3",
        "name": "test:engine", **legacy,
    })
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)
