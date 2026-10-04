#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for rock_dev_bootstrap.gpu_short, which names the default build dir.

A concrete target keeps its whole name so ``gfx1250`` and ``gfx1250-strict`` get distinct
build dirs; a TheRock family or generic name is cut at its first hyphen so existing
default build dirs keep their names.
"""

import unittest

from rock_dev_bootstrap import gpu_short


class TestGpuShort(unittest.TestCase):
    def test_concrete_targets_are_kept_whole(self):
        for target in (
            "gfx900",
            "gfx90a",
            "gfx942",
            "gfx950",
            "gfx1100",
            "gfx1201",
            "gfx1250",
            "gfx1250-strict",
        ):
            with self.subTest(target=target):
                self.assertEqual(gpu_short(target), target)

    def test_family_names_are_cut_at_first_hyphen(self):
        for name, short in (
            ("gfx94X-dcgpu", "gfx94X"),
            ("gfx125X-dcgpu", "gfx125X"),
            ("gfx950-dcgpu", "gfx950"),
            ("gfx90a-dcgpu", "gfx90a"),
            ("gfx906-dgpu", "gfx906"),
            ("gfx90c-igpu", "gfx90c"),
            ("gfx950-all", "gfx950"),
        ):
            with self.subTest(name=name):
                self.assertEqual(gpu_short(name), short)

    def test_family_variants_are_cut_with_their_family(self):
        # The family word is not the last segment.
        for name, short in (
            ("gfx950-dcgpu-asan", "gfx950"),
            ("gfx950-dcgpu-tests", "gfx950"),
            ("gfx90a-dcgpu-asan", "gfx90a"),
        ):
            with self.subTest(name=name):
                self.assertEqual(gpu_short(name), short)

    def test_generic_targets_are_cut_at_first_hyphen(self):
        for name, short in (
            ("gfx11-generic", "gfx11"),
            ("gfx12-5-generic", "gfx12"),
            ("gfx9-4-generic", "gfx9"),
        ):
            with self.subTest(name=name):
                self.assertEqual(gpu_short(name), short)

    def test_a_word_inside_a_longer_segment_does_not_make_a_family(self):
        # "strictall" is not the family word "all"; the suffix is kept as part of the target.
        self.assertEqual(gpu_short("gfx1250-strictall"), "gfx1250-strictall")


if __name__ == "__main__":
    unittest.main()
