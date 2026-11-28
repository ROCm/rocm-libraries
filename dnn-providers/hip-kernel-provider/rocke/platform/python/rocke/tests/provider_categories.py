# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: the CTest labels that the provider build gives a list of test names.

The hip-kernel-provider build labels rocKE's installed CTest entries from the
provider's ``ROCKE_ENGINE_test_categories_external.yaml`` with the monorepo's
shared category parser (``shared/ctest/parse_ctest_categories.py``). TheRock CI
selects tests by those labels, so an entry that no pattern matches never runs
there. Ask the parser itself rather than matching the patterns in the test, so
the check applies the parser's own matching rules.

    labels = provider_category_labels(["rocke_python_import_smoke", name], tmp_path)
    assert labels["rocke_python_import_smoke"] <= labels.get(name, set())

Both files exist only in a monorepo checkout; in a standalone rocKE tree or the
installed test layout the helper skips the test. A sparse checkout that leaves
``shared/ctest`` out (CI's pre-commit job checks out only the directories a
change touches) still has the parser in its git index, so the helper runs the
indexed copy rather than skipping.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from ..assets import platform_root

__all__ = [
    "provider_categories_file",
    "provider_category_labels",
    "shared_category_parser",
]

_LABELS_LINE = re.compile(
    r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)'
)
# The shared category parser, relative to the monorepo root.
_PARSER = "shared/ctest/parse_ctest_categories.py"


def shared_category_parser(monorepo: Path, scratch: Path) -> Path | None:
    """The monorepo's shared category parser: the checked-out file, else the git
    index's copy written into ``scratch`` (a sparse checkout without
    ``shared/ctest``), else None (``monorepo`` is no git checkout whose index
    holds the parser). Raises RuntimeError when the index holds the parser but
    git cannot read it (a blobless clone whose on-demand fetch failed), so the
    caller fails rather than skipping."""
    parser = monorepo / _PARSER
    if parser.is_file():
        return parser
    git = shutil.which("git")
    if git is None or not monorepo.is_dir():
        return None
    indexed_path = f":./{_PARSER}"
    in_index = subprocess.run(
        [git, "-C", str(monorepo), "rev-parse", "--verify", "--quiet", indexed_path],
        capture_output=True,
        check=False,
    )
    if in_index.returncode != 0:
        return None
    indexed = subprocess.run(
        [git, "-C", str(monorepo), "show", indexed_path],
        capture_output=True,
        check=False,
    )
    if indexed.returncode != 0:
        stderr = indexed.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"reading the shared category parser from the git index failed - "
            f"path={_PARSER!r}, git said {stderr!r}, expected git to read the "
            f"indexed blob (fetch it, or check out shared/ctest)"
        )
    copy = scratch / parser.name
    copy.write_bytes(indexed.stdout)
    return copy


def provider_categories_file() -> Path:
    """The provider's ``ROCKE_ENGINE_test_categories_external.yaml``, beside
    this rocke tree in a monorepo checkout (absent elsewhere)."""
    return platform_root().parent.parent / "ROCKE_ENGINE_test_categories_external.yaml"


def provider_category_labels(
    names: Iterable[str], scratch: Path
) -> dict[str, set[str]]:
    """Return the labels the provider build gives each of ``names``; a name that
    gets no label is absent. ``scratch`` is a directory for the stand-in CTest
    file the parser reads. Skips the test when this tree has no provider
    categories or no shared parser, or when PyYAML is missing."""
    categories = provider_categories_file()
    provider = categories.parent
    if not categories.is_file():
        pytest.skip(
            f"no provider CTest categories to check: {categories} does not exist"
        )
    monorepo = provider.parent.parent
    parser = shared_category_parser(monorepo, scratch)
    if parser is None:
        pytest.skip(
            f"no provider CTest categories to check: {monorepo / _PARSER} is in "
            "neither the checkout nor its git index"
        )
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")
    ctests = scratch / "CTestTestfile.cmake"
    ctests.write_text("".join(f"add_test({name} python)\n" for name in names))
    result = subprocess.run(
        [sys.executable, str(parser), str(categories), str(ctests)],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        name: set(value.split(";"))
        for name, value in _LABELS_LINE.findall(result.stdout)
    }
