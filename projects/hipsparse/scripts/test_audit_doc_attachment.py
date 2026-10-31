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

"""Unit tests for the AST parsing in audit_doc_attachment.py (no clang needed)."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_doc_attachment  # noqa: E402

HEADER = "/include/internal/level1/hipsparse_axpyi.h"


def function(name, loc, documented):
    node = {"kind": "FunctionDecl", "name": name, "loc": loc, "inner": []}
    if documented:
        node["inner"].append({"kind": "FullComment", "inner": []})
    node["inner"].append({"kind": "ParmVarDecl", "name": "handle"})
    return json.dumps(node, indent=2)


class ParseAstTest(unittest.TestCase):
    def test_documented_and_undocumented_functions(self):
        dump = "\n".join(
            [
                function(
                    "hipsparseSaxpyi", {"file": HEADER, "line": 40}, True
                ),
                function(
                    "hipsparseDaxpyi", {"file": HEADER, "line": 52}, False
                ),
            ]
        )
        self.assertEqual(
            audit_doc_attachment.parse_ast(dump),
            [
                (Path(HEADER).resolve(), 40, "hipsparseSaxpyi", True),
                (Path(HEADER).resolve(), 52, "hipsparseDaxpyi", False),
            ],
        )

    def test_macro_location_uses_expansion(self):
        loc = {
            "spellingLoc": {"file": "<scratch space>", "line": 3},
            "expansionLoc": {"file": HEADER, "line": 61},
        }
        [(path, line, _, _)] = audit_doc_attachment.parse_ast(
            function("hipsparseCaxpyi", loc, True)
        )
        self.assertEqual((path, line), (Path(HEADER).resolve(), 61))

    def test_other_declarations_are_ignored(self):
        dump = json.dumps(
            {"kind": "TypedefDecl", "name": "hipsparseHandle_t", "loc": {}}
        ) + function("rocsparse_axpyi", {"file": HEADER, "line": 1}, False)
        self.assertEqual(audit_doc_attachment.parse_ast(dump), [])

    def test_missing_location_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "hipsparseZaxpyi"):
            audit_doc_attachment.parse_ast(
                function("hipsparseZaxpyi", {"line": 70}, True)
            )


if __name__ == "__main__":
    unittest.main()
