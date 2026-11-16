# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helpers: reach the ``dsl_docs`` and ``library`` source trees, which exist
only in a source checkout.

The installed test payload carries the ``rocke`` package and most of the test
tree, but not ``dsl_docs/`` (documentation and authoring tools) and not the
``library/`` source root beside ``platform/`` (the installed payload stages a
copy of its packages for the library tests, which ``rocke.assets.library_root``
does not point at). A test that reads either tree is a source-tree check. It
runs where the tree exists and skips, with the reason, in the installed layout.

    def test_doc_names_every_target():
        doc = require_dsl_docs() / "reference" / "dtype_support.md"
        ...

The skip is keyed on the tree, not on the file the test reads. Where the tree
exists, a file missing from it (a renamed or deleted doc) fails the test.

Use this for a source-tree check inside a test file that is otherwise installed.
A test file that is source-tree tooling throughout is instead left out of the
install (the exclusion list in ``platform/CMakeLists.txt``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rocke.assets import dsl_docs_dir, library_root

__all__ = ["require_dsl_docs", "require_library_tree"]


def _require_source_tree(tree: Path, name: str) -> Path:
    """Return ``tree``, or skip the test, naming ``name`` and the path, when it
    is not a directory in this layout."""
    if not tree.is_dir():
        pytest.skip(
            f"source-tree check: no {name} tree at {tree}; the installed test "
            "payload does not ship it"
        )
    return tree


def require_dsl_docs() -> Path:
    """Return the ``dsl_docs`` tree, or skip the test when this layout lacks it."""
    return _require_source_tree(dsl_docs_dir(), "dsl_docs")


def require_library_tree() -> Path:
    """Return the ``library`` source root, or skip the test when this layout lacks it."""
    return _require_source_tree(library_root(), "library")
