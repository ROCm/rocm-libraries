# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The source-tree helpers return their tree or skip without it.

A helper's skip hides every check built on it, so a helper that skipped a present
tree would turn those checks off silently. These tests point each helper at a tree
through its documented override (``ROCKE_DSL_DOCS``, ``ROCKE_LIBRARY_ROOT``): a
present tree must come back (a skip there fails the test), and an absent tree must
skip, naming the tree and the path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from rocke.tests.source_tree import require_dsl_docs, require_library_tree

# (helper, the environment variable that relocates its tree, the tree's name)
_HELPERS = [
    pytest.param(require_dsl_docs, "ROCKE_DSL_DOCS", "dsl_docs", id="dsl_docs"),
    pytest.param(require_library_tree, "ROCKE_LIBRARY_ROOT", "library", id="library"),
]


def _require_present(helper: Callable[[], Path]) -> Path:
    """Call ``helper`` where its tree exists, turning its skip into a failure."""
    try:
        return helper()
    except pytest.skip.Exception as skip:
        pytest.fail(f"{helper.__name__}() skipped a present tree: {skip}")


@pytest.mark.parametrize(("helper", "variable", "name"), _HELPERS)
def test_returns_a_present_tree(monkeypatch, tmp_path, helper, variable, name):
    tree = tmp_path / name
    tree.mkdir()
    monkeypatch.setenv(variable, str(tree))
    assert _require_present(helper) == tree


@pytest.mark.parametrize(("helper", "variable", "name"), _HELPERS)
def test_skips_an_absent_tree_and_names_it(
    monkeypatch, tmp_path, helper, variable, name
):
    tree = tmp_path / name
    monkeypatch.setenv(variable, str(tree))
    with pytest.raises(pytest.skip.Exception) as skip:
        helper()
    assert str(skip.value) == (
        f"source-tree check: no {name} tree at {tree}; the installed test "
        "payload does not ship it"
    )
