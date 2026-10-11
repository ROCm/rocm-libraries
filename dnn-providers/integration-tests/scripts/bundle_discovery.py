#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python mirror of ``src/harness/bundle/BundleDiscovery.hpp``.

The C++ harness decides what counts as a bundle at test-registration time; the
Python claim tools have to reach the same verdict from a plain checkout. Two
independent transcriptions of that rule would drift the first time a companion
suffix is added, so both tools import this one.

``BundleDiscovery.hpp`` is the source of truth. ``test_bundle_discovery.py``
parses the header and fails if the constants below no longer match it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Mirrors isSweepTemplateFile() / isSweepManifestFile().
SWEEP_TEMPLATE_NAME = "graph.template.json"
SWEEP_MANIFEST_NAME = "sweep.json"

# The bare sidecar of a template-sweep bundle. Not part of the C++ discovery
# rule -- the harness reaches it via the sweep root -- but it belongs with the
# other on-disk bundle filenames.
SWEEP_SUPPORT_NAME = "support.json"

# Mirrors companionKinds(): suffixes that mark a .json as metadata *for* a
# graph rather than a graph test in its own right.
COMPANION_KINDS = frozenset({"meta", "support"})


def is_graph_file(json_path: Path) -> bool:
    """Mirror isGraphFile(): true only for a direct-bundle graph ``.json``.

    A companion is excluded when its whole stem is a companion kind
    (``meta.json``) or its final dotted segment is one (``Small.meta.json``).
    Other dotted names stay valid graphs, e.g. ``model.fp16.json``.
    """
    if json_path.suffix != ".json":
        return False
    if json_path.name in (SWEEP_TEMPLATE_NAME, SWEEP_MANIFEST_NAME):
        return False

    stem = json_path.stem
    if stem in COMPANION_KINDS:
        return False

    dot = stem.rfind(".")
    return dot == -1 or stem[dot + 1 :] not in COMPANION_KINDS


def is_sweep_root(directory: Path) -> bool:
    """Mirror isSweepBundleRoot(): holds both sweep control files."""
    return (directory / SWEEP_TEMPLATE_NAME).is_file() and (
        directory / SWEEP_MANIFEST_NAME
    ).is_file()


def is_descendant_of(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def walk_bundle_tree(root: Path) -> tuple[list[Path], list[Path]]:
    """Mirror forEachBundleTreeEntry(): every (directory, file) under ``root``.

    Directory symlinks are followed at any depth and entries keep their path
    through the link. A linked directory that resolves to (or above) a level
    the walk is already inside is a cycle and is skipped with a warning. A
    directory that cannot be listed is skipped with a warning, so one
    unreadable folder does not end discovery. Both lists are sorted.
    """
    directories: list[Path] = []
    files: list[Path] = []
    if not root.is_dir():
        return directories, files

    # Each stack item carries the canonical directory of every level the walk
    # is inside at that point, starting with the root.
    stack = [(root, (Path(os.path.realpath(root)),))]
    while stack:
        directory, ancestry = stack.pop()
        try:
            with os.scandir(directory) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            _warn(
                f"skipping bundle directory that cannot be listed: {directory} ({exc})"
            )
            continue
        for entry in entries:
            path = directory / entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=True)
            except OSError:
                is_dir = False
            if not is_dir:
                files.append(path)
                continue
            if entry.is_symlink():
                canonical = Path(os.path.realpath(path))
                if any(is_descendant_of(level, canonical) for level in ancestry):
                    _warn(
                        "not following bundle directory that contains one already "
                        f"being walked (cycle): {path} -> {canonical}"
                    )
                    continue
            else:
                canonical = ancestry[-1] / entry.name
            directories.append(path)
            stack.append((path, ancestry + (canonical,)))
    return sorted(directories), sorted(files)


def _sweep_roots_in(root: Path, directories: list[Path]) -> list[Path]:
    return sorted(d for d in [root, *directories] if is_sweep_root(d))


def _graph_files_in(files: list[Path], sweep_roots: list[Path]) -> list[Path]:
    sweep_set = set(sweep_roots)
    graphs = []
    for json_path in files:
        # Graphs inside a sweep root are reached through their sweep.
        if any(parent in sweep_set for parent in json_path.parents):
            continue
        if is_graph_file(json_path) and json_path.is_file():
            graphs.append(json_path)
    return graphs


def discover_bundle_tree(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """(sweep roots, direct-bundle graph files, every file) from one walk.

    The full file list is for callers that must account for files the bundle
    rule does not pick up -- a sidecar with no graph beside it, say.
    """
    if not root.is_dir():
        return [], [], []
    directories, files = walk_bundle_tree(root)
    sweep_roots = _sweep_roots_in(root, directories)
    return sweep_roots, _graph_files_in(files, sweep_roots), files


def discover_bundles(root: Path) -> tuple[list[Path], list[Path]]:
    """(sweep roots, direct-bundle graph files) under ``root``, from one walk."""
    sweep_roots, graph_files, _ = discover_bundle_tree(root)
    return sweep_roots, graph_files


def find_sweep_roots(root: Path) -> list[Path]:
    """Every sweep root under (or equal to) ``root``, sorted."""
    return discover_bundles(root)[0]


def find_graph_files(root: Path, sweep_roots: list[Path] | None = None) -> list[Path]:
    """Every direct-bundle graph under ``root``, sorted.

    Excludes companion sidecars, sweep control files, and anything living
    inside a sweep root -- those graphs are reached through their sweep.
    """
    if sweep_roots is None:
        return discover_bundles(root)[1]
    if not root.is_dir():
        return []
    _, files = walk_bundle_tree(root)
    return _graph_files_in(files, sweep_roots)
