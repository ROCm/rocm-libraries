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

"""Unit tests for audit_doc_attachment.py (no clang needed)."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_doc_attachment  # noqa: E402

HEADER = "/include/internal/level1/hipsparse_axpyi.h"


def text(value):
    return {"kind": "TextComment", "text": value}


def paragraph(*texts):
    return {"kind": "ParagraphComment", "inner": [text(t) for t in texts]}


BRIEF = {
    "kind": "BlockCommandComment",
    "name": "brief",
    "inner": [paragraph(" Scale. ")],
}
# The FullComment Clang attaches for the comments named in each key.
COMMENTS = {
    "doc": [paragraph(" "), BRIEF],
    "doc + /**@{*/": [paragraph(" "), BRIEF, paragraph("@", "{")],
    "/**@}*/": [paragraph("@", "}")],
    "/**@{*/": [paragraph("@", "{")],
}


def function(name, loc, documented, comment=None):
    node = {"kind": "FunctionDecl", "name": name, "loc": loc, "inner": []}
    node["inner"].append({"kind": "ParmVarDecl", "name": "handle"})
    if comment is None and documented:
        comment = "doc"
    if comment is not None:
        node["inner"].append(
            {"kind": "FullComment", "inner": COMMENTS[comment]}
        )
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

    def test_group_marker_alone_is_not_documentation(self):
        loc = {"file": HEADER, "line": 80}
        for comment, documented in [
            ("/**@}*/", False),
            ("/**@{*/", False),
            ("doc + /**@{*/", True),
        ]:
            with self.subTest(comment=comment):
                [(_, _, _, has_docs)] = audit_doc_attachment.parse_ast(
                    function("hipsparseXaxpyi", loc, None, comment)
                )
                self.assertEqual(has_docs, documented)


class ClangBinaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rocm = self.root / "rocm"
        self.path_dir = self.root / "bin"
        self.path_dir.mkdir()

    def executable(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def clang_binary(self, hipsparse_clang=None):
        environment = {"PATH": str(self.path_dir)}
        if hipsparse_clang is not None:
            environment["HIPSPARSE_CLANG"] = hipsparse_clang
        with mock.patch.dict(os.environ, environment, clear=True):
            return audit_doc_attachment.clang_binary(self.rocm)

    def test_missing_hipsparse_clang_is_an_error(self):
        self.executable(self.rocm / "llvm" / "bin" / "clang")
        for requested in (str(self.root / "no-clang"), "clang-404"):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(FileNotFoundError, requested):
                    self.clang_binary(requested)

    def test_hipsparse_clang_name_is_found_on_path(self):
        expected = self.executable(self.path_dir / "clang-19")
        self.executable(self.rocm / "llvm" / "bin" / "clang")
        self.assertEqual(self.clang_binary("clang-19"), expected)

    def test_rocm_clang_is_preferred_over_path(self):
        expected = self.executable(self.rocm / "llvm" / "bin" / "clang")
        self.executable(self.path_dir / "clang")
        self.assertEqual(self.clang_binary(), expected)

    def test_path_clang_is_the_fallback(self):
        expected = self.executable(self.path_dir / "clang")
        self.assertEqual(self.clang_binary(), expected)

    def test_no_clang_is_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "HIPSPARSE_CLANG"):
            self.clang_binary()


class MainTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.include = Path(directory.name).resolve()
        self.umbrella = self.include / "hipsparse.h"
        self.umbrella.write_text(
            '#include "hipsparse_axpyi.h"\n', encoding="utf-8"
        )
        self.header = self.include / "hipsparse_axpyi.h"
        self.header.write_text(
            "hipsparseStatus_t hipsparseSaxpyi(int x);\n", encoding="utf-8"
        )

    def main(self, stdout="", returncode=0, stderr=""):
        completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
        argv = ["audit_doc_attachment.py", str(self.include)]
        argv += ["--umbrella", str(self.umbrella)]
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(
                mock.patch.object(
                    audit_doc_attachment, "clang_binary", return_value="clang"
                )
            )
            run = stack.enter_context(
                mock.patch.object(
                    audit_doc_attachment.subprocess,
                    "run",
                    return_value=completed,
                )
            )
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(errors))
            status = audit_doc_attachment.main()
        run.assert_called_once()
        return status, output.getvalue(), errors.getvalue()

    def dump(self, documented):
        loc = {"file": str(self.header), "line": 1}
        return function("hipsparseSaxpyi", loc, documented)

    def test_documented_header_passes(self):
        status, output, _ = self.main(self.dump(True))
        self.assertEqual(status, 0)
        self.assertIn("Functions documented:   1", output)

    def test_undocumented_function_fails(self):
        status, output, _ = self.main(self.dump(False))
        self.assertEqual(status, 1)
        self.assertIn("hipsparseSaxpyi", output)
        self.assertIn("UNDOCUMENTED", output)

    def test_clang_failure_is_an_error(self):
        status, _, errors = self.main(
            self.dump(True), returncode=1, stderr="fatal error: hip.h\n"
        )
        self.assertEqual(status, 2)
        self.assertIn("fatal error: hip.h", errors)
        self.assertIn("clang exited with status 1", errors)

    def test_header_missing_from_ast_is_an_error(self):
        other = self.include / "hipsparse_other.h"
        other.write_text(
            "hipsparseStatus_t hipsparseOther(int x);\n", encoding="utf-8"
        )
        status, _, errors = self.main(self.dump(True))
        self.assertEqual(status, 2)
        self.assertIn(f"{other} declares hipsparse functions", errors)

    def test_no_functions_is_an_error(self):
        self.header.write_text("typedef int hipsparseInt;\n", encoding="utf-8")
        status, _, errors = self.main("")
        self.assertEqual(status, 2)
        self.assertIn("no hipsparse functions found", errors)


if __name__ == "__main__":
    unittest.main()
