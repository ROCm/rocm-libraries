#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""Unit tests for sync_group_docs.py (run: python3 -m unittest test_sync_group_docs)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sync_group_docs  # noqa: E402

DOC = "/*! \\brief Scale.\n*  @param[in] x vector.\n*/"


def declaration(name, prefix="HIPSPARSE_EXPORT"):
    return f"{prefix}\nhipsparseStatus_t {name}(int x);\n"


def group(first_doc, body):
    return f"{first_doc}\n/**@{{*/\n{body}/**@}}*/\n"


SYNCED = group(
    DOC,
    declaration("hipsparseSaxpyi")
    + "\n"
    + DOC
    + "\n"
    + declaration("hipsparseDaxpyi"),
)


class SynchronizeFileTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.header = Path(directory.name) / "hipsparse_test.h"

    def write(self, text):
        self.header.write_bytes(text.encode("utf-8"))

    def check(self):
        return sync_group_docs.synchronize_file(self.header, write=False)

    def fix(self):
        return sync_group_docs.synchronize_file(self.header, write=True)

    def text(self):
        return self.header.read_bytes().decode("utf-8")

    def test_synchronized_group_passes(self):
        self.write(SYNCED)
        self.assertEqual(self.check(), [])
        self.assertEqual(self.fix(), [])
        self.assertEqual(self.text(), SYNCED)

    def test_missing_copy_is_inserted(self):
        self.write(
            group(
                DOC,
                declaration("hipsparseSaxpyi")
                + "\n"
                + declaration("hipsparseDaxpyi"),
            )
        )
        self.assertEqual(self.check(), ["hipsparseDaxpyi"])
        self.fix()
        self.assertEqual(self.text(), SYNCED)

    def test_copy_separated_by_blank_line_is_replaced_in_place(self):
        self.write(
            SYNCED.replace(
                DOC + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                DOC + "\n\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
            )
        )
        self.assertEqual(self.check(), ["hipsparseDaxpyi"])
        self.fix()
        self.assertEqual(self.text(), SYNCED)
        self.assertEqual(self.text().count("\\brief Scale."), 2)

    def test_deprecated_prefix_is_kept(self):
        prefix = 'DEPRECATED_CUDA_12000("old")\nHIPSPARSE_EXPORT'
        expected = group(
            DOC,
            declaration("hipsparseSaxpyi", prefix)
            + "\n"
            + DOC
            + "\n"
            + declaration("hipsparseDaxpyi", prefix),
        )
        self.write(
            group(
                DOC,
                declaration("hipsparseSaxpyi", prefix)
                + "\n"
                + declaration("hipsparseDaxpyi", prefix),
            )
        )
        self.fix()
        self.assertEqual(self.text(), expected)

    def test_trailing_whitespace_in_first_block_is_reported(self):
        self.write(SYNCED.replace("\\brief Scale.\n", "\\brief Scale. \n", 1))
        self.assertEqual(self.check(), ["hipsparseSaxpyi"])
        self.fix()
        self.assertEqual(self.text(), SYNCED)
        self.assertEqual(self.check(), [])

    def test_blank_lines_before_group_marker_are_reported(self):
        self.write(SYNCED.replace("*/\n/**@{*/", "*/\n\n/**@{*/", 1))
        self.assertEqual(self.check(), ["hipsparseSaxpyi"])
        self.fix()
        self.assertEqual(self.text(), SYNCED)

    def test_unmatched_group_marker_fails_with_file_name(self):
        self.write(
            SYNCED.replace(
                "*/\n/**@{*/", "*/\n// clang-format off\n/**@{*/", 1
            )
        )
        with self.assertRaisesRegex(
            sync_group_docs.SyncError, "hipsparse_test.h: 1 of 1"
        ):
            self.check()

    def test_unparsable_declaration_fails_with_file_name(self):
        self.write(
            SYNCED.replace(
                "HIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                "hipsparseStatus_t hipsparseD",
            )
        )
        with self.assertRaisesRegex(
            sync_group_docs.SyncError, "hipsparse_test.h: .*hipsparseDaxpyi"
        ):
            self.check()

    def assert_refused(self, text, message):
        self.write(text)
        for write in (False, True):
            with self.assertRaisesRegex(sync_group_docs.SyncError, message):
                sync_group_docs.synchronize_file(self.header, write=write)
        self.assertEqual(self.text(), text)

    def test_doc_on_first_declaration_is_refused(self):
        self.assert_refused(
            SYNCED.replace("/**@{*/\n", "/**@{*/\n/*! \\brief Old. */\n", 1),
            "hipsparseSaxpyi has its own documentation block",
        )

    def test_doxygen_style_copy_is_refused(self):
        self.assert_refused(
            SYNCED.replace(
                DOC + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                "/** \\brief Old. */\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
            ),
            r"unexpected '/\*\* \\brief Old. \*/' before hipsparseDaxpyi",
        )

    def test_copy_above_preprocessor_line_is_refused(self):
        self.assert_refused(
            SYNCED.replace(
                DOC + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                DOC
                + "\n#if(!defined(CUDART_VERSION))\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
            ).replace("int x);\n/**@}*/", "int x);\n#endif\n/**@}*/"),
            "separated from its declaration by '#if",
        )

    def test_stacked_copies_are_refused(self):
        self.assert_refused(
            SYNCED.replace(
                DOC + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                DOC
                + "\n"
                + DOC
                + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
            ),
            r"unexpected '/\*! \\brief Scale.' before hipsparseDaxpyi",
        )

    def test_line_comment_between_copy_and_declaration_is_refused(self):
        self.assert_refused(
            SYNCED.replace(
                DOC + "\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
                DOC
                + "\n// note\nHIPSPARSE_EXPORT\nhipsparseStatus_t hipsparseD",
            ),
            "before hipsparseDaxpyi",
        )

    def test_member_with_other_return_type_is_refused(self):
        self.assert_refused(
            SYNCED.replace(
                "/**@}*/",
                "HIPSPARSE_EXPORT\nconst char* hipsparseGetName(int x);\n/**@}*/",
            ),
            "unexpected 'HIPSPARSE_EXPORT' after hipsparseDaxpyi",
        )

    def test_doc_above_conditional_outside_group_is_refused(self):
        self.assert_refused(
            DOC
            + "\n// clang-format off\n#if(!defined(CUDART_VERSION))\n"
            + declaration("hipsparseAxpby")
            + "#endif\n",
            "line 1: documentation block is separated from its declaration by "
            r"'#if\(!defined\(CUDART_VERSION\)\)'",
        )

    def test_doc_below_conditional_passes(self):
        text = (
            "/*! \\file */\n#ifndef HIPSPARSE_TEST_H\n"
            "#if(!defined(CUDART_VERSION))\n"
            + DOC
            + "\n"
            + declaration("hipsparseAxpby")
            + "#endif\n"
            "/*! \\brief A macro. */\n#define HIPSPARSE_TEST 1\n"
            "typedef enum {\n    A = 0, /**< first */\n#if 1\n    B = 1\n#endif\n} e;\n"
            "#endif\n"
        )
        self.write(text)
        self.assertEqual(self.check(), [])
        self.assertEqual(self.fix(), [])
        self.assertEqual(self.text(), text)

    def test_write_uses_lf_line_endings(self):
        self.write(
            group(
                DOC,
                declaration("hipsparseSaxpyi")
                + "\n"
                + declaration("hipsparseDaxpyi"),
            ).replace("\n", "\r\n")
        )
        self.fix()
        self.assertNotIn(b"\r", self.header.read_bytes())
        self.assertEqual(self.text(), SYNCED)


if __name__ == "__main__":
    unittest.main()
