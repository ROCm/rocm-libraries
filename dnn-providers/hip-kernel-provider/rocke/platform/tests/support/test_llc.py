# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for the llc test helpers: the compiler-backed checks skip when
llc cannot target the GPU, and fail on any other compiler error."""

from __future__ import annotations

import subprocess
import unittest

from rocke.tests.llc import checked_llc_output, lists_amdgpu_backend

_LLC = ["llc", "-mtriple=amdgcn-amd-amdhsa"]


class TestLlcHelpers(unittest.TestCase):
    def test_amdgpu_backend_is_read_from_the_version_text(self):
        with_amdgpu = (
            "LLVM version 20.0.0git\n  Registered Targets:\n"
            "    amdgcn - AMD GCN GPUs\n    x86-64 - 64-bit X86: EM64T and AMD64\n"
        )
        without_amdgpu = (
            "LLVM version 20.0.0git\n  Registered Targets:\n"
            "    x86-64 - 64-bit X86: EM64T and AMD64\n"
        )
        self.assertTrue(lists_amdgpu_backend(with_amdgpu))
        self.assertFalse(lists_amdgpu_backend(without_amdgpu))

    def test_unknown_processor_is_not_an_error_even_on_a_failed_run(self):
        result = subprocess.CompletedProcess(
            _LLC,
            returncode=1,
            stdout=b"",
            stderr=b"'gfx1154' is not a recognized processor for this target "
            b"(ignoring processor)\n",
        )
        self.assertIsNone(checked_llc_output(result))

    def test_other_compiler_errors_fail(self):
        result = subprocess.CompletedProcess(
            _LLC, returncode=1, stdout=b"", stderr=b"error: invalid register name\n"
        )
        with self.assertRaisesRegex(
            AssertionError,
            "llc failed with exit status 1: error: invalid register name",
        ):
            checked_llc_output(result)

    def test_successful_run_returns_the_output(self):
        result = subprocess.CompletedProcess(
            _LLC, returncode=0, stdout=b"asm", stderr=b""
        )
        self.assertEqual(checked_llc_output(result), b"asm")
