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


def find_sweep_roots(root: Path) -> list[Path]:
    """Every sweep root under (or equal to) ``root``, sorted."""
    if not root.is_dir():
        return []

    roots = set()
    if is_sweep_root(root):
        roots.add(root)
    for entry in root.rglob("*"):
        if entry.is_dir() and is_sweep_root(entry):
            roots.add(entry)
    return sorted(roots)


def is_descendant_of(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def find_graph_files(root: Path, sweep_roots: list[Path] | None = None) -> list[Path]:
    """Every direct-bundle graph under ``root``, sorted.

    Excludes companion sidecars, sweep control files, and anything living
    inside a sweep root -- those graphs are reached through their sweep.
    """
    if not root.is_dir():
        return []

    if sweep_roots is None:
        sweep_roots = find_sweep_roots(root)

    graphs = []
    for json_path in sorted(root.rglob("*.json")):
        if not json_path.is_file():
            continue
        if any(is_descendant_of(json_path, sweep) for sweep in sweep_roots):
            continue
        if is_graph_file(json_path):
            graphs.append(json_path)
    return graphs
