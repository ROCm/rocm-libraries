#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for miopen_wrapper_libs.py.

Both run_forwarding_parity.py and check_wrapper_abi.py resolve their library pair
through it, so the rules for which files count are tested here once. The callers'
own tests only check that a problem found here fails their run.

Each test names the platform layout it builds, so both layouts are tested on
either host.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_miopen_wrapper_libs
"""

import os
import tempfile
import unittest
from pathlib import Path

from miopen_wrapper_libs import (
    PRIVATE,
    WRAPPER,
    default_lib_dirs,
    find_library,
    loader_path_variable,
    resolve_pair,
)


class LinuxLayoutTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.lib = self.tree / "lib"
        self.lib.mkdir()

    def test_default_lib_dirs_come_from_the_install_prefix(self):
        """The scripts install to <prefix>/bin/miopen_wrapper_tests/."""
        (self.tree / "lib64").mkdir()
        (self.tree / "bin" / "lib_decoy").mkdir(parents=True)
        script_dir = self.tree / "bin" / "miopen_wrapper_tests"
        script_dir.mkdir()
        self.assertEqual(
            default_lib_dirs(script_dir, windows=False), [self.lib, self.tree / "lib64"]
        )

    def test_the_one_versioned_file_is_found(self):
        (self.lib / "libMIOpen.so.1.0").touch()
        self.assertEqual(
            find_library([self.lib], WRAPPER, windows=False),
            (self.lib / "libMIOpen.so.1.0", None),
        )

    def test_another_stem_sharing_the_prefix_is_not_a_match(self):
        """libMIOpen_private.so.* must not count as a second libMIOpen.so.*."""
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen_private.so.1.0").touch()
        path, problem = find_library([self.lib], WRAPPER, windows=False)
        self.assertEqual((path, problem), (self.lib / "libMIOpen.so.1.0", None))

    @unittest.skipUnless(os.name == "posix", "needs symlinks")
    def test_symlinks_are_skipped(self):
        """The usual soname links must not look like a second build."""
        real = self.lib / "libMIOpen.so.1.0"
        real.touch()
        (self.lib / "libMIOpen.so.1").symlink_to(real.name)
        self.assertEqual(find_library([self.lib], WRAPPER, windows=False), (real, None))

    def test_two_versioned_files_are_refused_rather_than_picked_between(self):
        """One of them is an earlier build, and filename order is not version order."""
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen.so.10.0").touch()
        path, problem = find_library([self.lib], WRAPPER, windows=False)
        self.assertIsNone(path)
        self.assertIn("more than one libMIOpen.so.*", problem)

    def test_a_missing_library_names_where_it_looked(self):
        other = self.tree / "lib64"
        other.mkdir()
        path, problem = find_library([self.lib, other], WRAPPER, windows=False)
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
            find_library([self.lib, lib64], WRAPPER, windows=False),
            (self.lib / "libMIOpen.so.1.0", None),
        )

    def test_a_pair_in_one_directory_resolves(self):
        (self.lib / "libMIOpen.so.1.0").touch()
        (self.lib / "libMIOpen_private.so.1.0").touch()
        self.assertEqual(
            resolve_pair([self.lib], windows=False),
            (self.lib / "libMIOpen.so.1.0", self.lib / "libMIOpen_private.so.1.0", []),
        )

    def test_both_missing_libraries_are_reported(self):
        wrapper, private, problems = resolve_pair([self.lib], windows=False)
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
        wrapper, private, problems = resolve_pair([self.lib, lib64], windows=False)
        self.assertEqual((wrapper, private), (None, None))
        [problem] = problems
        self.assertIn("different directories", problem)

    def test_the_loader_searches_ld_library_path(self):
        self.assertEqual(loader_path_variable(windows=False), "LD_LIBRARY_PATH")


class WindowsLayoutTest(unittest.TestCase):
    """DLLs carry no lib prefix and no version, and install to bin/."""

    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.bin = self.tree / "bin"
        self.script_dir = self.bin / "miopen_wrapper_tests"
        self.script_dir.mkdir(parents=True)

    def test_default_lib_dirs_is_the_bin_directory_above_the_scripts(self):
        """Windows finds a program's DLLs beside it, and the programs are in bin/."""
        (self.tree / "lib").mkdir()
        self.assertEqual(default_lib_dirs(self.script_dir, windows=True), [self.bin])

    def test_the_dll_is_found(self):
        (self.bin / "MIOpen.dll").touch()
        self.assertEqual(
            find_library([self.bin], WRAPPER, windows=True),
            (self.bin / "MIOpen.dll", None),
        )

    def test_an_import_library_or_linux_library_is_not_a_match(self):
        """MIOpen.lib is the import library, which cannot be loaded."""
        (self.bin / "MIOpen.lib").touch()
        (self.bin / "libMIOpen.so.1.0").touch()
        path, problem = find_library([self.bin], WRAPPER, windows=True)
        self.assertIsNone(path)
        self.assertIn(f"no MIOpen.dll found under {self.bin}", problem)

    def test_the_private_dll_is_not_taken_for_the_wrapper(self):
        (self.bin / "MIOpen_private.dll").touch()
        path, problem = find_library([self.bin], WRAPPER, windows=True)
        self.assertIsNone(path)
        self.assertIn("no MIOpen.dll found", problem)

    def test_a_pair_in_bin_resolves(self):
        (self.bin / "MIOpen.dll").touch()
        (self.bin / "MIOpen_private.dll").touch()
        self.assertEqual(
            resolve_pair(default_lib_dirs(self.script_dir, windows=True), windows=True),
            (self.bin / "MIOpen.dll", self.bin / "MIOpen_private.dll", []),
        )

    def test_a_missing_private_dll_is_reported(self):
        (self.bin / "MIOpen.dll").touch()
        wrapper, private, problems = resolve_pair([self.bin], windows=True)
        self.assertEqual((wrapper, private), (None, None))
        self.assertEqual(problems, [f"no {PRIVATE}.dll found under {self.bin}"])

    def test_the_loader_searches_path(self):
        """Windows ignores LD_LIBRARY_PATH."""
        self.assertEqual(loader_path_variable(windows=True), "PATH")


if __name__ == "__main__":
    unittest.main()
