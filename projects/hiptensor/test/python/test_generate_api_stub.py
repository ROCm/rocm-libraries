#!/usr/bin/env python3
###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

"""Unit tests for generate_api_stub.py.

Exercise the header parsing and both render modes (coverage / stub) with
synthetic headers, plus a check against the real public header. A regression in
the parser would silently weaken the stub or its completeness check, so it is
worth catching directly here.

Run:
    cd test/04_stub && python3 -m unittest test_generate_api_stub
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
# .../test/04_stub -> project root
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STUB_DIR = PROJECT_ROOT / "library" / "stub"
sys.path.insert(0, str(STUB_DIR))

import generate_api_stub as gen  # noqa: E402

PUBLIC_HEADER = PROJECT_ROOT / "library" / "include" / "hiptensor" / "hiptensor.h"


class _HeaderTmpTestCase(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write(self, text: str) -> Path:
        path = Path(self.tmpdir.name) / "hdr.h"
        path.write_text(text)
        return path


class ParseExportedDeclarationsTest(_HeaderTmpTestCase):
    def test_captures_return_type_name_and_params(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* h);\n"
            "HIPTENSOR_EXPORT int hiptensorGetHiprtVersion();\n"
        )
        fns = gen.parse_exported_declarations(hdr)
        self.assertEqual([f.name for f in fns], ["hiptensorCreate", "hiptensorGetHiprtVersion"])
        create = fns[0]
        self.assertEqual(create.return_type, "hiptensorStatus_t")
        self.assertEqual(create.params, "hiptensorHandle_t* h")
        self.assertEqual(fns[1].return_type, "int")
        self.assertEqual(fns[1].params, "")

    def test_names_only_helper(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorZeta(int);\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorAlpha(int);\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorAlpha(int);\n"  # dup
        )
        self.assertEqual(gen.parse_exported_functions(hdr), ["hiptensorAlpha", "hiptensorZeta"])

    def test_handles_multiline_declarations(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t\n"
            "    hiptensorCreateContraction(const hiptensorHandle_t handle,\n"
            "                               hiptensorOperationDescriptor_t* desc);\n"
        )
        fns = gen.parse_exported_declarations(hdr)
        self.assertEqual(len(fns), 1)
        self.assertEqual(fns[0].name, "hiptensorCreateContraction")
        self.assertEqual(fns[0].return_type, "hiptensorStatus_t")
        # Whitespace (incl. newlines) normalized to single spaces.
        self.assertIn("const hiptensorHandle_t handle", fns[0].params)
        self.assertIn("hiptensorOperationDescriptor_t* desc", fns[0].params)

    def test_ignores_commented_declarations(self):
        hdr = self._write(
            "// HIPTENSOR_EXPORT hiptensorStatus_t hiptensorGhostLine(int);\n"
            "/* HIPTENSOR_EXPORT hiptensorStatus_t hiptensorGhostBlock(int); */\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorReal(int);\n"
        )
        self.assertEqual(gen.parse_exported_functions(hdr), ["hiptensorReal"])

    def test_ignores_export_macro_guard(self):
        # The header guards the macro with #if !defined(HIPTENSOR_EXPORT) etc.
        # Those preprocessor lines must not leak into the first decl's return type.
        hdr = self._write(
            "#if !defined(HIPTENSOR_EXPORT)\n"
            "#define HIPTENSOR_EXPORT\n"
            "#endif\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
        )
        fns = gen.parse_exported_declarations(hdr)
        self.assertEqual(len(fns), 1)
        self.assertEqual(fns[0].name, "hiptensorCreate")
        self.assertEqual(fns[0].return_type, "hiptensorStatus_t")

    def test_empty_header_returns_empty(self):
        hdr = self._write("// nothing exported here\n")
        self.assertEqual(gen.parse_exported_declarations(hdr), [])


class RenderCoverageTest(unittest.TestCase):
    def _fns(self, *specs):
        return [gen.ExportedFunction(rt, n, p) for rt, n, p in specs]

    def test_one_address_row_per_function(self):
        out = gen.render_coverage(
            self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int"),
                      ("int", "hiptensorBeta", ""))
        )
        self.assertIn("reinterpret_cast<void*>(&hiptensorAlpha)", out)
        self.assertIn("reinterpret_cast<void*>(&hiptensorBeta)", out)
        self.assertEqual(out.count("reinterpret_cast<void*>"), 2)

    def test_includes_public_header_and_main(self):
        out = gen.render_coverage(self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int")))
        self.assertIn("#include <hiptensor/hiptensor.h>", out)
        self.assertIn("int main()", out)

    def test_no_trailing_comma_in_initializer(self):
        out = gen.render_coverage(
            self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int"),
                      ("hiptensorStatus_t", "hiptensorBeta", "int"))
        )
        body = out.split("kPublicApiSymbols[] = {", 1)[1].split("}", 1)[0]
        self.assertTrue(body.strip().endswith(")"))


class RenderStubTest(unittest.TestCase):
    def _fns(self, *specs):
        return [gen.ExportedFunction(rt, n, p) for rt, n, p in specs]

    def test_emits_definition_per_status_function(self):
        out = gen.render_stub(
            self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int x"),
                      ("hiptensorStatus_t", "hiptensorBeta", "hiptensorHandle_t"))
        )
        self.assertIn("hiptensorStatus_t hiptensorAlpha(int x)", out)
        self.assertIn("hiptensorStatus_t hiptensorBeta(hiptensorHandle_t)", out)
        self.assertEqual(out.count("return HIPTENSOR_STATUS_NOT_SUPPORTED;"), 2)

    def test_skips_non_status_functions(self):
        # Informational helpers (non-status return types) are hand-written, not
        # generated -- they must have no DEFINITION here so there is no duplicate
        # symbol. (Their names may still appear in the file's header comment.)
        out = gen.render_stub(
            self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int"),
                      ("const char*", "hiptensorGetErrorString", "hiptensorStatus_t"),
                      ("int", "hiptensorGetHiprtVersion", ""),
                      ("size_t", "hiptensorGetVersion", ""))
        )
        self.assertIn("hiptensorStatus_t hiptensorAlpha(int)", out)
        self.assertNotIn("hiptensorGetErrorString(", out)
        self.assertNotIn("hiptensorGetHiprtVersion(", out)
        self.assertNotIn("hiptensorGetVersion(", out)
        self.assertEqual(out.count("return HIPTENSOR_STATUS_NOT_SUPPORTED;"), 1)

    def test_includes_public_header(self):
        out = gen.render_stub(self._fns(("hiptensorStatus_t", "hiptensorAlpha", "int")))
        self.assertIn("#include <hiptensor/hiptensor.h>", out)


class MainTest(_HeaderTmpTestCase):
    def _run(self, header_text: str, mode: str = "coverage", out_name: str = "out.cpp"):
        hdr = self._write(header_text)
        out = Path(self.tmpdir.name) / out_name
        argv = ["prog", "--mode", mode, "--header", str(hdr), "--output", str(out)]
        old = sys.argv
        sys.argv = argv
        try:
            rc = gen.main()
        finally:
            sys.argv = old
        return rc, out

    def test_coverage_mode_writes_output(self):
        rc, out = self._run(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
        )
        self.assertEqual(rc, 0)
        self.assertIn("&hiptensorCreate", out.read_text())

    def test_stub_mode_writes_output(self):
        rc, out = self._run(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
            "HIPTENSOR_EXPORT int hiptensorGetHiprtVersion();\n",
            mode="stub",
        )
        self.assertEqual(rc, 0)
        text = out.read_text()
        self.assertIn("hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*)", text)
        self.assertIn("return HIPTENSOR_STATUS_NOT_SUPPORTED;", text)
        # The non-status helper is not generated (name may appear in a comment,
        # but there must be no definition/call form).
        self.assertNotIn("hiptensorGetHiprtVersion(", text)

    def test_default_mode_is_coverage(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
        )
        out = Path(self.tmpdir.name) / "out.cpp"
        old = sys.argv
        sys.argv = ["prog", "--header", str(hdr), "--output", str(out)]
        try:
            rc = gen.main()
        finally:
            sys.argv = old
        self.assertEqual(rc, 0)
        self.assertIn("reinterpret_cast<void*>", out.read_text())

    def test_creates_missing_output_directory(self):
        rc, out = self._run(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n",
            out_name="nested/dir/out.cpp",
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())

    def test_empty_header_is_an_error(self):
        rc, out = self._run("// nothing\n")
        self.assertEqual(rc, 1)
        self.assertFalse(out.exists())


@unittest.skipUnless(PUBLIC_HEADER.exists(), f"public header not found: {PUBLIC_HEADER}")
class RealHeaderTest(unittest.TestCase):
    def test_public_header_has_exports(self):
        names = gen.parse_exported_functions(PUBLIC_HEADER)
        self.assertGreater(len(names), 0)
        self.assertIn("hiptensorCreate", names)
        self.assertIn("hiptensorDestroy", names)

    def test_special_helpers_are_not_generated(self):
        # These carry real behavior and are hand-written in
        # hiptensor_stub_special.cpp; the generator must not DEFINE them (their
        # names may still appear in the generated file's header comment).
        out = gen.render_stub(gen.parse_exported_declarations(PUBLIC_HEADER))
        for helper in (
            "hiptensorGetErrorString",
            "hiptensorGetHiprtVersion",
            "hiptensorGetVersion",
        ):
            self.assertNotIn(f"{helper}(", out)

    def test_all_status_functions_are_generated(self):
        fns = gen.parse_exported_declarations(PUBLIC_HEADER)
        status_fns = [f for f in fns if f.return_type == "hiptensorStatus_t"]
        out = gen.render_stub(fns)
        self.assertGreater(len(status_fns), 0)
        for f in status_fns:
            self.assertIn(f"{f.name}(", out)
        self.assertEqual(
            out.count("return HIPTENSOR_STATUS_NOT_SUPPORTED;"), len(status_fns)
        )


if __name__ == "__main__":
    unittest.main()
