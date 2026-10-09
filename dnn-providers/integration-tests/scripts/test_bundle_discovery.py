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

import re
from pathlib import Path

from bundle_discovery import (
    COMPANION_KINDS,
    SWEEP_MANIFEST_NAME,
    SWEEP_TEMPLATE_NAME,
    find_graph_files,
    find_sweep_roots,
    is_descendant_of,
    is_graph_file,
    is_sweep_root,
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
    path.write_text("{}")
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
