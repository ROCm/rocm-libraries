# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: reach the ``dsl_docs`` tree, which exists only in a source checkout.

The installed test payload carries the ``rocke`` package and most of the test
tree, but not ``dsl_docs/``: that tree is documentation and authoring tools, and
it stays in the source checkout. A test that reads it is a source-tree check. It
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

from rocke.assets import dsl_docs_dir

__all__ = ["require_dsl_docs"]


def require_dsl_docs() -> Path:
    """Return the ``dsl_docs`` tree, or skip the test when this layout lacks it."""
    docs = dsl_docs_dir()
    if not docs.is_dir():
        pytest.skip(
            f"source-tree check: no dsl_docs tree at {docs}; the installed test "
            "payload does not ship it"
        )
    return docs
