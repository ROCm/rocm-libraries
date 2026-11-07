# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate references in source and relocatable installed test trees."""

from __future__ import annotations

from pathlib import Path


def installed_test_root(tests_dir: Path) -> Path | None:
    """Return an installed library test tree's test root, or None in a source checkout.

    Both provider and standalone installs put library tests at
    <test-root>/tests/library/tests. Source tests live at <rocke>/library/tests.
    """
    tests_dir = tests_dir.resolve()
    if tests_dir.parts[-3:] == ("tests", "library", "tests"):
        return tests_dir.parent.parent.parent
    return None


def default_bundle_path(
    tests_dir: Path, architecture: str, *, operation: str, variant: str
) -> Path:
    """Keep installed references in the test-only architecture packaging domain.

    Each variant's bundle sits below its architecture directory, which stays
    the first processor-named directory that TheRock splits test payloads by.
    Select by layout, not bundle existence, so missing installed data cannot
    silently fall back to a stale bundle in the generic test tree.
    """
    test_root = installed_test_root(tests_dir)
    if test_root is not None:
        domain = test_root / "engines/test_arch_content/rocke"
    else:
        domain = tests_dir.resolve() / "reference_bundles"
    return domain / operation / architecture / variant
