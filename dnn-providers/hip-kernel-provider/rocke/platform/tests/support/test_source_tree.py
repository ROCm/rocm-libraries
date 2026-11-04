# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The source-tree helper returns the ``dsl_docs`` tree or skips without it.

The helper's skip hides every check built on it, so a helper that skipped a present
tree would turn those checks off silently. These tests point the helper at a tree
through its documented ``ROCKE_DSL_DOCS`` override: a present tree must come back
(a skip there fails the test), and an absent tree must skip, naming the path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rocke.tests.source_tree import require_dsl_docs


def _require_present_dsl_docs() -> Path:
    """Call the helper where the tree exists, turning its skip into a failure."""
    try:
        return require_dsl_docs()
    except pytest.skip.Exception as skip:
        pytest.fail(f"require_dsl_docs() skipped a present tree: {skip}")


def test_returns_a_present_tree(monkeypatch, tmp_path):
    docs = tmp_path / "dsl_docs"
    docs.mkdir()
    monkeypatch.setenv("ROCKE_DSL_DOCS", str(docs))
    assert _require_present_dsl_docs() == docs


def test_skips_an_absent_tree_and_names_it(monkeypatch, tmp_path):
    docs = tmp_path / "dsl_docs"
    monkeypatch.setenv("ROCKE_DSL_DOCS", str(docs))
    with pytest.raises(pytest.skip.Exception) as skip:
        require_dsl_docs()
    assert str(skip.value) == (
        f"source-tree check: no dsl_docs tree at {docs}; the installed test "
        "payload does not ship it"
    )
