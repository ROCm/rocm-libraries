# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A `kind: "model"` UHD names a file, and the packer has to carry it.

Discovery globs `*.json`, so every other descriptor the packer handles is fully
described by a file it already found. A trained UHD is the exception: its
`payload` is a `.uhd.fb` path, and that file plus the model artifact it in turn
names have to reach the packed tree or the runtime loads the descriptor, fails to
find the artifact, and silently degrades to declared order.

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


def _model_uhd(payload: str) -> dict:
    return {
        "version": "1.0",
        "id": "uhd-model",
        "name": "Trained heuristic",
        "kind": "model",
        "payload": payload,
    }


def _native_uhd() -> dict:
    return {
        "version": "1.0",
        "id": "uhd-native",
        "name": "Native heuristic",
        "kind": "native",
        "payload": "hipkernel.pointwise.score",
    }


def _root_with_model_uhd(tmp_path: Path, payload: str = "heuristic.uhd.fb") -> Path:
    """A minimal root: one model UHD and the artifact it names."""
    root = tmp_path / "src"
    _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd(payload))
    artifact = root / "pack" / "heuristic.uhd.fb"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"\x00\x00\x00\x00HUHDstub")
    return root


class TestResolution:
    def test_model_payload_becomes_a_sidecar(self, tmp_path: Path):
        flat = load_flat_input(_root_with_model_uhd(tmp_path), log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.name == "heuristic.uhd.fb"
        assert sidecar.rel_dir == Path("pack")
        assert sidecar.source.read_bytes() == b"\x00\x00\x00\x00HUHDstub"

    def test_native_payload_is_a_symbol_not_a_file(self, tmp_path: Path):
        """A native UHD's payload names an in-process symbol.

        Resolving it as a path would reject every UHD shipping today, all of
        which are native.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "native.uhd.json", _native_uhd())

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars == []

    def test_payload_resolves_relative_to_its_own_descriptor(self, tmp_path: Path):
        """Not root-relative, matching how a hip UKD's `source` resolves.

        A root-relative fallback would fire exactly when the descriptor-local
        file is missing, turning a typo into a silent bind to a same-named file
        elsewhere in the tree.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("heuristic.uhd.fb"))
        # A decoy at the root with the same name: resolution must not reach it.
        (root / "heuristic.uhd.fb").write_bytes(b"decoy")
        (root / "pack" / "heuristic.uhd.fb").write_bytes(b"correct")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars[0].source.read_bytes() == b"correct"

    def test_a_shared_artifact_keeps_its_authored_position(self, tmp_path: Path):
        """`../shared/x` is how one artifact is shared between sibling packs.

        The staged copy has to keep the same relative position, or `payload`
        stops resolving in the packed tree even though it resolved in the source.
        """
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.uhd.fb"),
        )
        shared = root / "rocKE" / "shared" / "model.uhd.fb"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.rel_dir == Path("rocKE/shared")
        assert sidecar.name == "model.uhd.fb"


class TestRejection:
    def test_missing_artifact_is_an_error_not_a_warning(self, tmp_path: Path):
        """The runtime degrades silently on a missing artifact; the packer must not.

        DescriptorLoader logs a warning and ranks by priority when the artifact
        is absent, so a UHD that shipped without one produces an engine that
        quietly stops using its model. Catching it at pack time is the only place
        it is loud.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("absent.uhd.fb"))

        with pytest.raises(HkpPackError, match="payload source not found"):
            load_flat_input(root, log=lambda *_: None)

    def test_artifact_outside_the_root_is_rejected(self, tmp_path: Path):
        """Containment, mirroring the hip source check and the runtime's treeRoot bound.

        An artifact the packer resolved outside the root could not be staged
        anywhere the runtime would find it, and a `payload` that walks out of the
        tree is the descriptor-side half of a path-traversal.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("../../outside.uhd.fb"))
        (tmp_path.parent / "outside.uhd.fb").write_bytes(b"outside")

        with pytest.raises(HkpPackError, match="payload escapes the source root"):
            load_flat_input(root, log=lambda *_: None)

    def test_escape_is_checked_before_existence(self, tmp_path: Path):
        """An escaping path that also does not exist reports the escape.

        Reporting 'not found' would send an author looking for a missing file
        when the real fault is where they pointed.
        """
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("../../nowhere.uhd.fb"))

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
        staged = inter_dir / "pack" / "heuristic.uhd.fb"
        assert staged.is_file(), "the artifact the UHD names must ride with it"
        assert staged.read_bytes() == b"\x00\x00\x00\x00HUHDstub"

    def test_sidecar_keeps_its_authored_subpath(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.uhd.fb"),
        )
        shared = root / "rocKE" / "shared" / "model.uhd.fb"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")
        flat = load_flat_input(root, log=lambda *_: None)
        inter_dir = tmp_path / "inter" / "gfx942"

        compile_intermediate(
            flat, root, "gfx942", hipcc=None, inter_arch_dir=inter_dir, log=lambda *_: None
        )

        # Staged where `../shared/model.uhd.fb` still reaches it from the UHD.
        assert (inter_dir / "rocKE" / "shared" / "model.uhd.fb").read_bytes() == b"shared"
        assert not (inter_dir / "rocKE" / "attn" / "model.uhd.fb").exists()
