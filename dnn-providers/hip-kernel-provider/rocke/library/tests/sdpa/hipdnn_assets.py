# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate the hipDNN sources the SDPA contract tests cross-check against.

The SDPA graph template and the golden bundles live in the hipDNN part of the
monorepo, outside rocke. An installed test tree carries a copy under ``hipdnn/``
beside this module (``platform/CMakeLists.txt`` installs it when it configures
from a monorepo checkout); a source checkout reads the monorepo. The tests that
read them skip when neither exists, unless ``--rocke-require-hipdnn-assets`` is
given (the installed CTest entry passes it whenever it installed the copy).
"""

from __future__ import annotations

from pathlib import Path

from rocke.assets import library_root

_SDPA_GRAPH_TEMPLATE = (
    "projects/hipdnn/tools/IngestorGenerator/configs/templates/"
    "attention_sdpa_fwd.graph.json"
)
_GOLDEN_BUNDLES = "dnn-providers/integration-tests/integration-test-bundles"
# The installed copy keeps each file's name under these two roots.
_INSTALLED = Path(__file__).resolve().parent / "hipdnn"
_INSTALLED_TEMPLATE = _INSTALLED / Path(_SDPA_GRAPH_TEMPLATE).name
_INSTALLED_BUNDLES = _INSTALLED / Path(_GOLDEN_BUNDLES).name
SKIP_REASON = (
    "hipDNN sources are not next to this rocke tree and no installed copy exists "
    "(not a monorepo checkout)"
)


def repo_root() -> Path | None:
    """The monorepo root holding ``projects/hipdnn``, or ``None`` outside a checkout."""
    for parent in library_root().resolve().parents:
        if (parent / "projects" / "hipdnn").is_dir():
            return parent
    return None


def sdpa_graph_template() -> Path | None:
    if _INSTALLED_TEMPLATE.is_file():
        return _INSTALLED_TEMPLATE
    root = repo_root()
    path = None if root is None else root / _SDPA_GRAPH_TEMPLATE
    return path if path is not None and path.is_file() else None


def golden_bundle_root() -> Path | None:
    if _INSTALLED_BUNDLES.is_dir():
        return _INSTALLED_BUNDLES
    root = repo_root()
    path = None if root is None else root / _GOLDEN_BUNDLES
    return path if path is not None and path.is_dir() else None


def sdpa_fwd_golden_graphs() -> list[Path]:
    """The checked-in SdpaFwd graphs.

    Excludes meta, support, and sweep files, and sweep-bundle templates
    (``graph.template.json``). A template still carries ``${case.x}``
    placeholders filled per case from the bundle's ``sweep.json``, so it is
    not a complete graph.
    """
    bundles = golden_bundle_root()
    if bundles is None:
        return []
    return sorted(
        p
        for p in bundles.glob("*/SdpaFwd*/**/*.json")
        if not p.name.endswith(
            ("meta.json", "support.json", "sweep.json", "graph.template.json")
        )
        and "Default" not in p.parts
    )
