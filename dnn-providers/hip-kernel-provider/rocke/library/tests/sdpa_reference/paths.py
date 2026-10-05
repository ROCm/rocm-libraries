# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate references in source and relocatable installed test trees."""

from __future__ import annotations

from pathlib import Path


def default_bundle_path(tests_dir: Path, architecture: str) -> Path:
    """Keep installed references in the test-only architecture packaging domain.

    Both provider and standalone installs put library tests at
    <test-root>/tests/library/tests. Source tests live at <rocke>/library/tests.
    Select by layout, not bundle existence, so missing installed data cannot
    silently fall back to a stale bundle in the generic test tree.
    """
    tests_dir = tests_dir.resolve()
    if tests_dir.parts[-3:] == ("tests", "library", "tests"):
        test_root = tests_dir.parent.parent.parent
        return test_root / "engines/test_arch_content/rocke/sdpa" / architecture
    return tests_dir / "reference_bundles" / "sdpa" / architecture
