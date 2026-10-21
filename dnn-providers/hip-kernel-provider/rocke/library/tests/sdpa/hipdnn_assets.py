# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate the hipDNN sources the SDPA contract tests cross-check against.

The SDPA graph template and the golden bundles live in the hipDNN part of the
monorepo, outside rocke. The tests that read them skip when it is absent, which
is the normal case for an installed rocke tree. Set ``ROCKE_REPO_ROOT`` to a
monorepo checkout to run them from an installed tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from rocke.assets import library_root

_SDPA_GRAPH_TEMPLATE = (
    "projects/hipdnn/tools/IngestorGenerator/configs/templates/"
    "attention_sdpa_fwd.graph.json"
)
_GOLDEN_BUNDLES = "dnn-providers/integration-tests/integration-test-bundles"
SKIP_REASON = (
    "hipDNN sources are not next to this rocke tree (set ROCKE_REPO_ROOT to a "
    "monorepo checkout)"
)


def repo_root() -> Path | None:
    """The monorepo root holding ``projects/hipdnn``, or ``None`` outside a checkout."""
    override = os.environ.get("ROCKE_REPO_ROOT")
    if override:
        return Path(override)
    for parent in library_root().resolve().parents:
        if (parent / "projects" / "hipdnn").is_dir():
            return parent
    return None


def sdpa_graph_template() -> Path | None:
    root = repo_root()
    path = None if root is None else root / _SDPA_GRAPH_TEMPLATE
    return path if path is not None and path.is_file() else None


def golden_bundle_root() -> Path | None:
    root = repo_root()
    path = None if root is None else root / _GOLDEN_BUNDLES
    return path if path is not None and path.is_dir() else None


def sdpa_fwd_golden_graphs() -> list[Path]:
    """The checked-in SdpaFwd graphs (not their meta, support or sweep files)."""
    bundles = golden_bundle_root()
    if bundles is None:
        return []
    return sorted(
        p
        for p in bundles.glob("*/SdpaFwd*/**/*.json")
        if not p.name.endswith(("meta.json", "support.json", "sweep.json"))
        and "Default" not in p.parts
    )
