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
        "id": "uhd-model",
        "name": "Trained heuristic",
        "adapter": "tree_data",
        "features_signature": ["$kernel.tile_m"],
        "features_hash": "sha256:0000000000000000",
        "objective": "max",
        "tree_data": {"artifact": artifact},
    }


def _native_uhd() -> dict:
    return {
        "version": "1.0",
        "id": "uhd-native",
        "name": "Native heuristic",
        "adapter": "native",
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

        with pytest.raises(HkpPackError, match="requires 'tree_data.artifact'"):
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
