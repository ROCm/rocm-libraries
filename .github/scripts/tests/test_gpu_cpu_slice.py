# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for gpu_cpu_slice.py: slicing, the GPU/GPU-count defaults, and the exec path."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gpu_cpu_slice as S  # noqa: E402


class CpuSliceTest(unittest.TestCase):
    def test_four_gpus_get_equal_non_overlapping_quarters(self):
        cpus = list(range(256))
        slices = [S.cpu_slice(cpus, g, 4) for g in range(4)]
        self.assertEqual([len(s) for s in slices], [64] * 4)
        self.assertEqual(sorted(sum(slices, [])), cpus)
        self.assertEqual(slices[1][:2], [64, 65])

    def test_uses_the_mask_not_cpu_numbers(self):
        # A mask that already excludes CPUs (e.g. a cpuset) is sliced by position.
        self.assertEqual(S.cpu_slice([2, 3, 10, 11], 1, 2), [10, 11])

    def test_gpu_index_wraps_and_tiny_masks_keep_one_cpu(self):
        self.assertEqual(S.cpu_slice(range(8), 5, 4), [2, 3])
        self.assertEqual(S.cpu_slice([7], 3, 4), [7])
        self.assertEqual(S.cpu_slice(range(8), 0, 0), list(range(8)))

    def test_cpu_ranges(self):
        self.assertEqual(S.cpu_ranges([0, 1, 2, 5, 7, 8]), "0-2,5,7-8")
        self.assertEqual(S.cpu_ranges([]), "")


class ParseArgsTest(unittest.TestCase):
    def test_defaults_from_environment(self):
        args = S.parse_args(
            ["--", "true"], {"ROCR_VISIBLE_DEVICES": "2", "GPUS_PER_HOST": "4"}
        )
        self.assertEqual((args.gpu, args.gpus_per_host, args.command), (2, 4, ["true"]))

    def test_flags_override_environment(self):
        args = S.parse_args(
            ["--gpu", "1", "--gpus-per-host", "2", "--", "tox", "-e", "py3"],
            {"ROCR_VISIBLE_DEVICES": "3", "GPUS_PER_HOST": "4"},
        )
        self.assertEqual(
            (args.gpu, args.gpus_per_host, args.command), (1, 2, ["tox", "-e", "py3"])
        )

    def test_no_environment_means_the_whole_mask(self):
        args = S.parse_args(["--", "true"], {"ROCR_VISIBLE_DEVICES": "GPU-uuid"})
        self.assertEqual((args.gpu, args.gpus_per_host), (0, 1))

    def test_first_of_several_visible_devices(self):
        self.assertEqual(
            S.parse_args(["--", "true"], {"ROCR_VISIBLE_DEVICES": "3,1"}).gpu, 3
        )

    def test_missing_command_is_an_error(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            S.parse_args(["--"], {})


class MainTest(unittest.TestCase):
    @mock.patch("builtins.print")
    @mock.patch.dict(S.os.environ, {}, clear=False)
    @mock.patch.object(S.os, "execvp")
    @mock.patch.object(S.os, "sched_setaffinity")
    @mock.patch.object(S.os, "sched_getaffinity", return_value=set(range(16)))
    def test_pins_exports_and_execs(self, _getaff, setaff, execvp, printed):
        S.main(["--gpu", "3", "--gpus-per-host", "4", "--", "tox", "-e", "py3"])
        setaff.assert_called_once_with(0, [12, 13, 14, 15])
        self.assertEqual(S.os.environ["LOKY_MAX_CPU_COUNT"], "4")
        execvp.assert_called_once_with("tox", ["tox", "-e", "py3"])
        self.assertIn("4 of 16 CPUs: 12-15", printed.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
