#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for bundle_discovery.py.

Two jobs. ``TestHeaderParity`` reads ``BundleDiscovery.hpp`` and fails if the
Python constants no longer match the C++ ones -- that is the whole reason the
mirror is allowed to exist. The rest pin the discovery behaviour itself, using
the same fixtures a bundle author would hit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

import pytest

from bundle_discovery import (
    BUNDLES_DIR_NAME,
    COMPANION_KINDS,
    DEFAULT_BUNDLES_DIR,
    SWEEP_MANIFEST_NAME,
    SWEEP_SUPPORT_NAME,
    SWEEP_TEMPLATE_NAME,
    find_graph_files,
    find_sweep_roots,
    graph_for_sidecar,
    is_descendant_of,
    is_graph_file,
    is_single_graph_sidecar,
    is_sweep_root,
    load_json,
    meta_for_graph,
    read_sweep_cases,
    sidecar_for_graph,
)

HEADER = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "harness"
    / "bundle"
    / "BundleDiscovery.hpp"
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _make_sweep(directory: Path) -> Path:
    _touch(directory / SWEEP_TEMPLATE_NAME)
    _touch(directory / SWEEP_MANIFEST_NAME)
    return directory


class TestHeaderParity:
    """The C++ header is the source of truth; these fail on drift."""

    def test_header_exists(self) -> None:
        assert HEADER.is_file(), f"{HEADER} moved; update this test's path"

    def test_companion_kinds_match(self) -> None:
        text = HEADER.read_text(encoding="utf-8")
        match = re.search(
            r"s_kinds\s*=\s*\{([^}]*)\}",
            text,
        )
        assert match, "could not find companionKinds() initializer in the header"
        kinds = set(re.findall(r'"([^"]+)"', match.group(1)))
        assert kinds == set(COMPANION_KINDS), (
            f"COMPANION_KINDS drifted: header has {sorted(kinds)}, "
            f"Python has {sorted(COMPANION_KINDS)}"
        )

    def test_sweep_template_name_matches(self) -> None:
        text = HEADER.read_text(encoding="utf-8")
        match = re.search(
            r"isSweepTemplateFile[^{]*\{[^}]*?filename\(\)\s*==\s*\"([^\"]+)\"",
            text,
            re.DOTALL,
        )
        assert match, "could not find isSweepTemplateFile() literal in the header"
        assert match.group(1) == SWEEP_TEMPLATE_NAME

    def test_sweep_manifest_name_matches(self) -> None:
        text = HEADER.read_text(encoding="utf-8")
        match = re.search(
            r"isSweepManifestFile[^{]*\{[^}]*?filename\(\)\s*==\s*\"([^\"]+)\"",
            text,
            re.DOTALL,
        )
        assert match, "could not find isSweepManifestFile() literal in the header"
        assert match.group(1) == SWEEP_MANIFEST_NAME

    def test_sweep_root_rule_matches(self) -> None:
        """isSweepBundleRoot() must test for exactly the two control files."""
        text = HEADER.read_text(encoding="utf-8")
        match = re.search(
            r"isSweepBundleRoot[^{]*\{(.*?)\n\}",
            text,
            re.DOTALL,
        )
        assert match, "could not find isSweepBundleRoot() body in the header"
        literals = set(re.findall(r'"([^"]+)"', match.group(1)))
        assert literals == {SWEEP_TEMPLATE_NAME, SWEEP_MANIFEST_NAME}


class TestIsGraphFile:
    def test_plain_graph_accepted(self, tmp_path: Path) -> None:
        assert is_graph_file(tmp_path / "ConvFwd.json")

    def test_non_json_rejected(self, tmp_path: Path) -> None:
        assert not is_graph_file(tmp_path / "ConvFwd.yaml")
        assert not is_graph_file(tmp_path / "README.md")

    def test_sweep_control_files_rejected(self, tmp_path: Path) -> None:
        assert not is_graph_file(tmp_path / SWEEP_TEMPLATE_NAME)
        assert not is_graph_file(tmp_path / SWEEP_MANIFEST_NAME)

    def test_bare_companion_rejected(self, tmp_path: Path) -> None:
        assert not is_graph_file(tmp_path / "meta.json")
        assert not is_graph_file(tmp_path / "support.json")

    def test_suffixed_companion_rejected(self, tmp_path: Path) -> None:
        assert not is_graph_file(tmp_path / "Small.meta.json")
        assert not is_graph_file(tmp_path / "Small.support.json")

    def test_unrelated_dotted_name_accepted(self, tmp_path: Path) -> None:
        """Only the *final* dotted segment is a companion marker."""
        assert is_graph_file(tmp_path / "model.fp16.json")
        assert is_graph_file(tmp_path / "resnet50.v2.json")
        assert is_graph_file(tmp_path / "meta.conv.json")


class TestIsSweepRoot:
    def test_both_control_files_present(self, tmp_path: Path) -> None:
        assert is_sweep_root(_make_sweep(tmp_path))

    def test_template_alone_is_not_a_root(self, tmp_path: Path) -> None:
        _touch(tmp_path / SWEEP_TEMPLATE_NAME)
        assert not is_sweep_root(tmp_path)

    def test_manifest_alone_is_not_a_root(self, tmp_path: Path) -> None:
        _touch(tmp_path / SWEEP_MANIFEST_NAME)
        assert not is_sweep_root(tmp_path)

    def test_empty_directory_is_not_a_root(self, tmp_path: Path) -> None:
        assert not is_sweep_root(tmp_path)


class TestFindSweepRoots:
    def test_missing_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert find_sweep_roots(tmp_path / "absent") == []

    def test_root_itself_counts(self, tmp_path: Path) -> None:
        _make_sweep(tmp_path)
        assert find_sweep_roots(tmp_path) == [tmp_path]

    def test_nested_roots_found_and_sorted(self, tmp_path: Path) -> None:
        first = _make_sweep(tmp_path / "quick" / "Batchnorm" / "Default")
        second = _make_sweep(tmp_path / "quick" / "Conv" / "Default")
        assert find_sweep_roots(tmp_path) == sorted([first, second])

    def test_deduplicates(self, tmp_path: Path) -> None:
        """A root that is also the scan root appears exactly once."""
        _make_sweep(tmp_path)
        _make_sweep(tmp_path / "inner")
        assert len(find_sweep_roots(tmp_path)) == 2


class TestFindGraphFiles:
    def test_missing_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert find_graph_files(tmp_path / "absent") == []

    def test_finds_graphs_and_skips_companions(self, tmp_path: Path) -> None:
        graph = _touch(tmp_path / "quick" / "Conv" / "ConvFwd.json")
        _touch(tmp_path / "quick" / "Conv" / "ConvFwd.support.json")
        _touch(tmp_path / "quick" / "Conv" / "ConvFwd.meta.json")
        assert find_graph_files(tmp_path) == [graph]

    def test_skips_everything_under_a_sweep_root(self, tmp_path: Path) -> None:
        """Sweep graphs are reached through the sweep, not as direct tests."""
        sweep = _make_sweep(tmp_path / "quick" / "Batchnorm" / "Default")
        _touch(sweep / "golden" / "case_0" / "tensors.json")
        direct = _touch(tmp_path / "quick" / "Conv" / "ConvFwd.json")
        assert find_graph_files(tmp_path) == [direct]

    def test_accepts_precomputed_sweep_roots(self, tmp_path: Path) -> None:
        """Callers that already scanned should not pay for a second walk."""
        sweep = _make_sweep(tmp_path / "sweepy")
        _touch(sweep / "Extra.json")
        direct = _touch(tmp_path / "Direct.json")
        roots = find_sweep_roots(tmp_path)
        assert find_graph_files(tmp_path, roots) == [direct]

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        for name in ("c.json", "a.json", "b.json"):
            _touch(tmp_path / name)
        found = find_graph_files(tmp_path)
        assert found == sorted(found)


class TestIsDescendantOf:
    def test_direct_child(self, tmp_path: Path) -> None:
        assert is_descendant_of(tmp_path / "a" / "b.json", tmp_path)

    def test_self_counts(self, tmp_path: Path) -> None:
        assert is_descendant_of(tmp_path, tmp_path)

    def test_sibling_does_not(self, tmp_path: Path) -> None:
        assert not is_descendant_of(tmp_path / "a", tmp_path / "b")


class TestDefaultBundlesDir:
    def test_is_the_checkouts_bundle_tree(self) -> None:
        assert DEFAULT_BUNDLES_DIR.name == BUNDLES_DIR_NAME
        assert DEFAULT_BUNDLES_DIR.is_dir()
        assert DEFAULT_BUNDLES_DIR.parent == HEADER.parents[3]


class TestSingleGraphCompanionNames:
    def test_sidecar_and_meta_sit_beside_the_graph(self) -> None:
        graph = Path("a") / "b" / "Small.json"
        assert sidecar_for_graph(graph) == Path("a") / "b" / "Small.support.json"
        assert meta_for_graph(graph) == Path("a") / "b" / "Small.meta.json"

    def test_keeps_the_path_flavour(self) -> None:
        sidecar = sidecar_for_graph(PurePosixPath("a/Small.json"))
        assert type(sidecar) is PurePosixPath

    def test_dotted_graph_name_keeps_its_dots(self) -> None:
        graph = Path("model.fp16.json")
        assert sidecar_for_graph(graph) == Path("model.fp16.support.json")
        assert graph_for_sidecar(sidecar_for_graph(graph)) == graph

    def test_round_trip(self) -> None:
        sidecar = Path("a") / "Small.support.json"
        assert sidecar_for_graph(graph_for_sidecar(sidecar)) == sidecar

    def test_companions_are_not_graphs(self, tmp_path: Path) -> None:
        graph = tmp_path / "Small.json"
        assert is_graph_file(graph)
        assert not is_graph_file(sidecar_for_graph(graph))
        assert not is_graph_file(meta_for_graph(graph))

    def test_bare_sweep_sidecar_is_not_a_single_graph_sidecar(self) -> None:
        bare = Path("sweep") / SWEEP_SUPPORT_NAME
        assert is_single_graph_sidecar(Path("Small.support.json"))
        assert not is_single_graph_sidecar(bare)
        with pytest.raises(ValueError):
            graph_for_sidecar(bare)

    def test_other_names_are_not_sidecars(self) -> None:
        assert not is_single_graph_sidecar(Path("Small.json"))
        with pytest.raises(ValueError):
            graph_for_sidecar(Path("Small.meta.json"))


class TestLoadJson:
    def test_parses_a_document(self, tmp_path: Path) -> None:
        path = tmp_path / "a.json"
        path.write_text('{"cases": [1, 2]}', encoding="utf-8")
        assert load_json(path) == ({"cases": [1, 2]}, None)

    def test_null_document_is_not_an_error(self, tmp_path: Path) -> None:
        """Callers must be able to tell `null` from a file that failed."""
        path = tmp_path / "a.json"
        path.write_text("null", encoding="utf-8")
        assert load_json(path) == (None, None)

    @pytest.mark.parametrize(
        "content, reason",
        [
            (b"{", "invalid JSON"),
            (b'{"a": "\xff"}', "not valid UTF-8"),
            (b"[" * 100000, "nested too deeply"),
        ],
        ids=["truncated", "non-utf8", "deeply-nested"],
    )
    def test_unparseable_file_gives_a_reason(
        self, tmp_path: Path, content: bytes, reason: str
    ) -> None:
        path = tmp_path / "a.json"
        path.write_bytes(content)
        data, error = load_json(path)
        assert data is None
        assert reason in error

    def test_missing_file_gives_a_reason(self, tmp_path: Path) -> None:
        data, error = load_json(tmp_path / "absent.json")
        assert data is None
        assert "cannot read" in error


class TestReadSweepCases:
    """Each rejection mirrors a throw in readSweepCaseIds()."""

    def _write(self, tmp_path: Path, manifest: object) -> Path:
        path = tmp_path / SWEEP_MANIFEST_NAME
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_returns_the_cases(self, tmp_path: Path) -> None:
        cases = [{"id": "a", "values": {}}, {"id": "b"}]
        assert read_sweep_cases(self._write(tmp_path, {"cases": cases})) == (
            cases,
            None,
        )

    @pytest.mark.parametrize(
        "manifest, reason",
        [
            ([], "missing cases[] array"),
            ({}, "missing cases[] array"),
            ({"cases": {}}, "missing cases[] array"),
            ({"cases": ["a"]}, "case 0 has no string id"),
            ({"cases": [{"id": 3}]}, "case 0 has no string id"),
            ({"cases": [{"id": "a"}, {"id": ""}]}, "case 1 has an empty id"),
            ({"cases": [{"id": "a"}, {"id": "a"}]}, "duplicate case id 'a'"),
        ],
    )
    def test_rejects_what_the_harness_rejects(
        self, tmp_path: Path, manifest: object, reason: str
    ) -> None:
        assert read_sweep_cases(self._write(tmp_path, manifest)) == (None, reason)

    def test_unparseable_manifest_passes_the_load_error_through(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / SWEEP_MANIFEST_NAME
        path.write_text("{", encoding="utf-8")
        cases, error = read_sweep_cases(path)
        assert cases is None
        assert "invalid JSON" in error
