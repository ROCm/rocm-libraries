#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The .so cache key must track the ctypes export set by itself.

setup_multiple_{rowcolquant,tensorquant}_dispatchers() reuse a cached .so when
the filename matches. The filename therefore has to encode the exported C ABI:
without it, an artifact built before dispatcher_get_tile_n()/dispatcher_get_pad_n()
existed is a name match, gets loaded, and fails at attribute lookup with a bare
"undefined symbol" -- on someone else's machine, in a cache directory, long after
the change that caused it.

A hand-bumped integer only works while everyone remembers to bump it, and
nothing can tell that they did not. The token is derived from the export
signatures in the ctypes source instead. These tests hold that derivation to its
promise: it must move when the ABI moves, stay put when it does not, and be the
same rule in both modules.

Run: python3 -m pytest tests/test_so_abi_token.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import grouped_gemm_rowcolquant_utils as rc  # noqa: E402
import grouped_gemm_tensorquant_utils as tq  # noqa: E402

MODULES = {"rowcolquant": rc, "tensorquant": tq}


class TestSoAbiToken(unittest.TestCase):
    def test_token_is_derived_not_hardcoded(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self.assertEqual(mod._SO_ABI, mod._ctypes_abi_token())
                self.assertNotEqual(mod._SO_ABI, mod._SO_ABI_FALLBACK)

    def test_token_is_stable_across_calls(self):
        # It ends up in a filename, so an unstable token would defeat the cache
        # entirely rather than fail loudly.
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self.assertEqual(mod._ctypes_abi_token(), mod._ctypes_abi_token())

    def test_the_current_export_set_is_captured(self):
        # If the regex silently stops matching, the token freezes and the whole
        # mechanism goes quiet. Pin the exports the Python side binds to.
        expected = {
            "dispatcher_initialize",
            "dispatcher_init",
            "dispatcher_run_gemm",
            "dispatcher_get_kernel_name",
            "dispatcher_get_kernel_count",
            "dispatcher_get_tile_n",
            "dispatcher_get_pad_n",
            "dispatcher_cleanup",
        }
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                found = {
                    m[1]
                    for m in mod._EXPORT_SIGNATURE_RE.findall(
                        mod._CTYPES_LIB_SRC.read_text()
                    )
                }
                self.assertEqual(found, expected)

    def test_token_changes_when_an_export_is_added(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self._assert_token_moves(
                    mod,
                    lambda s: s.replace(
                        "int dispatcher_get_kernel_count()",
                        "int dispatcher_probe_export(int x);\nint dispatcher_get_kernel_count()",
                        1,
                    ),
                )

    def test_token_changes_when_a_return_type_changes(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self._assert_token_moves(
                    mod,
                    lambda s: s.replace(
                        "int dispatcher_get_tile_n()", "long dispatcher_get_tile_n()", 1
                    ),
                )

    def test_token_changes_when_a_parameter_type_changes(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self._assert_token_moves(
                    mod, lambda s: s.replace("int k_batch,", "long k_batch,", 1)
                )

    def test_token_does_not_change_for_a_comment_edit(self):
        # Deliberate: keying on the whole file would invalidate every cached
        # artifact for a typo fix.
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                before = mod._ctypes_abi_token()
                self.assertEqual(
                    self._token_of(mod, lambda s: s + "\n// a trailing comment\n"),
                    before,
                )

    def test_the_two_modules_use_the_same_rule(self):
        # The helper is duplicated because there are two modules and two sources.
        # Pin the rule so the copies cannot diverge into two different cache-key
        # schemes.
        self.assertEqual(rc._EXPORT_SIGNATURE_RE.pattern, tq._EXPORT_SIGNATURE_RE.pattern)
        self.assertEqual(rc._SO_ABI_FALLBACK, tq._SO_ABI_FALLBACK)

    def test_fallback_is_used_when_the_source_is_unreadable(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                original = mod._CTYPES_LIB_SRC
                try:
                    mod._CTYPES_LIB_SRC = Path("/nonexistent/ctypes_lib.cpp")
                    self.assertEqual(mod._ctypes_abi_token(), mod._SO_ABI_FALLBACK)
                finally:
                    mod._CTYPES_LIB_SRC = original

    def _token_of(self, mod, mutate):
        """Token for a mutated COPY of the ctypes source.

        The checked-in file is never written to: a test that edits the working
        tree corrupts it if it is interrupted, and these run under CTest where
        that is a real possibility.
        """
        original = mod._CTYPES_LIB_SRC.read_text()
        mutated = mutate(original)
        self.assertNotEqual(mutated, original, "mutation did not apply")
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / mod._CTYPES_LIB_SRC.name
            probe.write_text(mutated)
            real = mod._CTYPES_LIB_SRC
            try:
                mod._CTYPES_LIB_SRC = probe
                return mod._ctypes_abi_token()
            finally:
                mod._CTYPES_LIB_SRC = real

    def _assert_token_moves(self, mod, mutate):
        self.assertNotEqual(
            self._token_of(mod, mutate),
            mod._ctypes_abi_token(),
            "the ABI token did not move for a real ABI change, so a stale "
            "cached .so would still be reused",
        )


if __name__ == "__main__":
    unittest.main()
