# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Structural guard for the intra-library layering rule (see rocke/AGENTS.md
# "Layout"). `library` is layered lowest -> highest:
#
#     kernels -> dispatch -> builders -> benchmarks
#
# A package may import anything BELOW it and nothing at or above it. Skipping a
# layer downward is allowed (builders imports kernels directly); importing
# upward is not, because it creates a cycle.
#
# `dispatch` sits BELOW `builders`, which is the opposite of the order this file
# first declared. `dispatch` is pure selection policy over `kernels` -- given a
# request it picks a candidate and constructs the spec that ships -- and it
# imports nothing but `kernels`. A builder is a host-side harness, and a harness
# that measures anything other than the shipped spec is a decoration, not a gate;
# so consuming `dispatch.<arch>.*_spec_for_request` is a permanent, one-way need
# of the harness layer, not an accident of one file. Declaring builders below
# dispatch made that legitimate edge a "cycle" while leaving the direction that
# would actually be a cycle (dispatch -> builders, which no file does and none
# should) unremarked.
#
# This is an AST check, not an import check, on purpose: the cheapest way to
# "fix" a cycle is to bury the offending import inside a function body, which a
# runtime import graph would not see. We walk every Import/ImportFrom node at
# any nesting depth so a deferred import is caught exactly like a top-level one.
#
# `tests/` is deliberately NOT a layer -- it sits above everything and may
# import any package.
#
# KNOWN_VIOLATIONS below records the one pre-existing back-edge so this guard
# passes on current code while still failing on any NEW one. It is an allowlist
# to burn down, not a place to add to.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library

# Lowest layer first. Index in this tuple IS the layer rank.
LAYERS = ("kernels", "dispatch", "builders", "benchmarks")
_RANK = {name: i for i, name in enumerate(LAYERS)}

# (file relative to library/, imported layer) pairs that predate this guard.
# `attention_unified` keeps arch-specific spec construction out of arch-neutral
# `kernels/common/` by deferring it into the dispatch functions; the deferred
# target currently lives in `builders`. Moving it down into `kernels` is an
# attention refactor, tracked separately. Do not extend this list.
KNOWN_VIOLATIONS = {
    ("kernels/common/attention_unified.py", "builders"),
}


def _imported_layers(path: Path):
    """Yield (layer, lineno) for every intra-library import in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _RANK:
                    yield root, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import: intra-package, never cross-layer.
            if node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root in _RANK:
                    yield root, node.lineno


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_does_not_import_upward(layer: str) -> None:
    """No package imports a package at or above its own layer."""
    pkg = _LIBROOT / layer
    if not pkg.is_dir():
        pytest.skip(f"{layer}/ not present")

    violations = []
    seen_known = set()
    for py in sorted(pkg.rglob("*.py")):
        rel = py.relative_to(_LIBROOT).as_posix()
        for imported, lineno in _imported_layers(py):
            if _RANK[imported] >= _RANK[layer] and imported != layer:
                if (rel, imported) in KNOWN_VIOLATIONS:
                    seen_known.add((rel, imported))
                    continue
                violations.append(f"  {rel}:{lineno}: {layer} -> {imported}")

    assert not violations, (
        f"'{layer}' imports at/above its own layer, creating a dependency cycle.\n"
        f"Allowed direction is {' -> '.join(LAYERS)} (import downward only).\n"
        + "\n".join(violations)
    )

    # Keep the allowlist honest: once a back-edge is removed, its entry must go
    # too, or it silently licenses a future re-introduction.
    stale = {v for v in KNOWN_VIOLATIONS if v[0].startswith(f"{layer}/")} - seen_known
    assert (
        not stale
    ), f"KNOWN_VIOLATIONS entries no longer apply -- remove them: {sorted(stale)}"
