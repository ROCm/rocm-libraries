#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for miopen_wrapper_libs.py.

Both run_forwarding_parity.py and check_wrapper_abi.py resolve their library pair
through it, so the rules for which files count are tested here once. The callers'
own tests only check that a problem found here fails their run.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_miopen_wrapper_libs
"""

import os
import tempfile
import unittest
from pathlib import Path

from miopen_wrapper_libs import find_library, resolve_pair


class WrapperLibsTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.lib = self.tree / "lib"
        self.lib.mkdir()

    def test_the_one_versioned_file_is_found(self):
        (self.lib / "libMIOpen.so.1.0").touch()
        self.assertEqual(
            find_library([self.lib], "libMIOpen"), (self.lib / "libMIOpen.so.1.0", None)
        )

    def test_another_stem_sharing_the_prefix_is_not_a_match(self):
        """libMIOpen_private.so.* must not count as a second libMIOpen.so.*."""
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen_private.so.1.0").touch()
        path, problem = find_library([self.lib], "libMIOpen")
        self.assertEqual((path, problem), (self.lib / "libMIOpen.so.1.0", None))

    @unittest.skipUnless(os.name == "posix", "needs symlinks")
    def test_symlinks_are_skipped(self):
        """The usual soname links must not look like a second build."""
        real = self.lib / "libMIOpen.so.1.0"
        real.touch()
        (self.lib / "libMIOpen.so.1").symlink_to(real.name)
        self.assertEqual(find_library([self.lib], "libMIOpen"), (real, None))

    def test_two_versioned_files_are_refused_rather_than_picked_between(self):
        """One of them is an earlier build, and filename order is not version order."""
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen.so.10.0").touch()
        path, problem = find_library([self.lib], "libMIOpen")
        self.assertIsNone(path)
        self.assertIn("more than one libMIOpen.so.*", problem)

    def test_a_missing_library_names_where_it_looked(self):
        other = self.tree / "lib64"
        other.mkdir()
        path, problem = find_library([self.lib, other], "libMIOpen")
        self.assertIsNone(path)
        self.assertIn("no libMIOpen.so.* found", problem)
        self.assertIn(str(self.lib), problem)
        self.assertIn(str(other), problem)

    def test_the_first_directory_with_a_match_wins(self):
        lib64 = self.tree / "lib64"
        lib64.mkdir()
        (lib64 / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen.so.1.0").touch()
        self.assertEqual(
            find_library([self.lib, lib64], "libMIOpen"),
            (self.lib / "libMIOpen.so.1.0", None),
        )

    def test_a_pair_in_one_directory_resolves(self):
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen_private.so.1.0").touch()
        self.assertEqual(
            resolve_pair([self.lib]),
            (self.lib / "libMIOpen.so.1.0", self.lib / "libMIOpen_private.so.1.0", []),
        )

    def test_both_missing_libraries_are_reported(self):
        wrapper, private, problems = resolve_pair([self.lib])
        self.assertEqual((wrapper, private), (None, None))
        self.assertEqual(len(problems), 2, problems)

    def test_a_pair_split_across_two_directories_is_refused(self):
        """lib and lib64 both present, with one half of the pair in each.

        Only one directory can go first on LD_LIBRARY_PATH, so a replay would load
        mismatched halves -- what the co-versioning check exists to catch.
        """
        lib64 = self.tree / "lib64"
        lib64.mkdir()
        (self.lib / "libMIOpen.so.1.0").touch()
        (lib64 / "libMIOpen_private.so.1.0").touch()
        wrapper, private, problems = resolve_pair([self.lib, lib64])
        self.assertEqual((wrapper, private), (None, None))
        [problem] = problems
        self.assertIn("different directories", problem)


if __name__ == "__main__":
    unittest.main()
