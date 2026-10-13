# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``dispatch.attention`` must resolve to the library package, not to these tests.

This directory is itself named ``dispatch/attention``. Run with ``tests/`` first on
``sys.path`` (pytest's rootdir insertion, or a CTest working directory of ``tests/``),
it would shadow the real package and every test here would import the wrong module.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import dispatch.attention


class DispatchAttentionResolution(unittest.TestCase):
    def test_the_package_is_the_librarys_not_the_test_directorys(self):
        package_dir = Path(dispatch.attention.__file__).resolve().parent
        library_root = package_dir.parents[1]
        self.assertEqual(package_dir.parent.name, "dispatch")
        self.assertTrue(
            (library_root / "kernels").is_dir(),
            f"{package_dir} is not under a library root (no sibling kernels/ package)",
        )


if __name__ == "__main__":
    unittest.main()
