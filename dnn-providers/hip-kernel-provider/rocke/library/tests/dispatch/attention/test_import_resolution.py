# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``dispatch.attention`` resolves to the library package, not this same-named test
directory (which would shadow it with ``tests/`` first on ``sys.path``)."""

from __future__ import annotations

import unittest
from pathlib import Path

import dispatch.attention


class DispatchAttentionResolution(unittest.TestCase):
    def test_the_package_is_the_librarys_not_the_test_directorys(self):
        package_dir = Path(dispatch.attention.__file__).resolve().parent
        self.assertTrue((package_dir.parents[1] / "kernels").is_dir(), package_dir)


if __name__ == "__main__":
    unittest.main()
