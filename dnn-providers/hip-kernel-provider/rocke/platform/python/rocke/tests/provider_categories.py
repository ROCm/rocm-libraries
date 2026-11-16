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
installed test layout the helper skips the test.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from ..assets import platform_root

__all__ = ["provider_category_labels"]

_LABELS_LINE = re.compile(
    r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)'
)


def provider_category_labels(
    names: Iterable[str], scratch: Path
) -> dict[str, set[str]]:
    """Return the labels the provider build gives each of ``names``; a name that
    gets no label is absent. ``scratch`` is a directory for the stand-in CTest
    file the parser reads. Skips the test when this tree has no provider
    categories or no shared parser, or when PyYAML is missing."""
    provider = platform_root().parent.parent
    categories = provider / "ROCKE_ENGINE_test_categories_external.yaml"
    parser = provider.parent.parent / "shared/ctest/parse_ctest_categories.py"
    for path in (categories, parser):
        if not path.is_file():
            pytest.skip(f"no provider CTest categories to check: {path} does not exist")
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
