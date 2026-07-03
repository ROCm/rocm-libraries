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

"""Unit tests for generate_api_coverage.py.

These exercise the header parsing and source rendering in isolation (with
synthetic headers) and once against the real public header, so a regression in
the parser -- which would silently weaken the stub completeness check -- is
caught directly.

Run:
    python3 -m pytest library/stub/test_generate_api_coverage.py -v
or:
    python3 -m unittest library.stub.test_generate_api_coverage
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
# .../test/04_stub -> project root
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STUB_DIR = PROJECT_ROOT / "library" / "stub"
sys.path.insert(0, str(STUB_DIR))

import generate_api_coverage as gen  # noqa: E402

PUBLIC_HEADER = PROJECT_ROOT / "library" / "include" / "hiptensor" / "hiptensor.h"
STUB_SOURCE = STUB_DIR / "hiptensor_stub.cpp"


class ParseExportedFunctionsTest(unittest.TestCase):
    def _write(self, text: str) -> Path:
        path = Path(self.tmpdir.name) / "hdr.h"
        path.write_text(text)
        return path

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_finds_basic_declarations(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* h);\n"
            "HIPTENSOR_EXPORT int hiptensorGetHiprtVersion();\n"
        )
        self.assertEqual(
            gen.parse_exported_functions(hdr),
            ["hiptensorCreate", "hiptensorGetHiprtVersion"],
        )

    def test_result_is_sorted_and_deduplicated(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorZeta(int);\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorAlpha(int);\n"
            # A re-declaration must not produce a duplicate.
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorAlpha(int);\n"
        )
        self.assertEqual(
            gen.parse_exported_functions(hdr), ["hiptensorAlpha", "hiptensorZeta"]
        )

    def test_handles_multiline_declarations(self):
        hdr = self._write(
            "HIPTENSOR_EXPORT hiptensorStatus_t\n"
            "    hiptensorCreateContraction(const hiptensorHandle_t handle,\n"
            "                               hiptensorOperationDescriptor_t* desc);\n"
        )
        self.assertEqual(
            gen.parse_exported_functions(hdr), ["hiptensorCreateContraction"]
        )

    def test_ignores_commented_declarations(self):
        # Names that appear only inside comments must not be collected, so doc
        # prose mentioning a removed/future API cannot create a phantom symbol.
        hdr = self._write(
            "// HIPTENSOR_EXPORT hiptensorStatus_t hiptensorGhostLine(int);\n"
            "/* HIPTENSOR_EXPORT hiptensorStatus_t hiptensorGhostBlock(int); */\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorReal(int);\n"
        )
        self.assertEqual(gen.parse_exported_functions(hdr), ["hiptensorReal"])

    def test_ignores_export_macro_guard(self):
        # The header guards the macro itself; that line is not a declaration.
        hdr = self._write(
            "#if !defined(HIPTENSOR_EXPORT)\n"
            "#define HIPTENSOR_EXPORT\n"
            "#endif\n"
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
        )
        self.assertEqual(gen.parse_exported_functions(hdr), ["hiptensorCreate"])

    def test_empty_header_returns_empty(self):
        hdr = self._write("// nothing exported here\n")
        self.assertEqual(gen.parse_exported_functions(hdr), [])


class RenderTest(unittest.TestCase):
    def test_one_address_row_per_name(self):
        out = gen.render(["hiptensorAlpha", "hiptensorBeta"])
        self.assertIn("reinterpret_cast<void*>(&hiptensorAlpha)", out)
        self.assertIn("reinterpret_cast<void*>(&hiptensorBeta)", out)
        self.assertEqual(out.count("reinterpret_cast<void*>"), 2)

    def test_includes_public_header_and_main(self):
        out = gen.render(["hiptensorAlpha"])
        self.assertIn("#include <hiptensor/hiptensor.h>", out)
        self.assertIn("int main()", out)

    def test_rows_comma_separated_without_trailing_comma(self):
        out = gen.render(["hiptensorAlpha", "hiptensorBeta"])
        # The brace-enclosed initializer must not end with a dangling comma.
        body = out.split("kPublicApiSymbols[] = {", 1)[1].split("}", 1)[0]
        self.assertNotIn(",\n    }", body + "}")
        self.assertEqual(body.strip().endswith(")"), True)


class MainTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)

    def _run(self, header_text: str, out_name: str = "out.cpp"):
        hdr = self.root / "hdr.h"
        hdr.write_text(header_text)
        out = self.root / out_name
        argv = ["prog", "--header", str(hdr), "--output", str(out)]
        old = sys.argv
        sys.argv = argv
        try:
            rc = gen.main()
        finally:
            sys.argv = old
        return rc, out

    def test_writes_output_and_returns_zero(self):
        rc, out = self._run(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n"
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())
        self.assertIn("&hiptensorCreate", out.read_text())

    def test_creates_missing_output_directory(self):
        rc, out = self._run(
            "HIPTENSOR_EXPORT hiptensorStatus_t hiptensorCreate(hiptensorHandle_t*);\n",
            out_name="nested/dir/out.cpp",
        )
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())

    def test_empty_header_is_an_error(self):
        # No exported functions almost certainly means the parser broke; failing
        # here prevents silently generating an empty (always-passing) check.
        rc, out = self._run("// nothing\n")
        self.assertEqual(rc, 1)
        self.assertFalse(out.exists())


@unittest.skipUnless(PUBLIC_HEADER.exists(), f"public header not found: {PUBLIC_HEADER}")
class RealHeaderTest(unittest.TestCase):
    def test_public_header_has_exports(self):
        names = gen.parse_exported_functions(PUBLIC_HEADER)
        self.assertGreater(len(names), 0)
        # Sanity: a couple of stable, well-known entry points are present.
        self.assertIn("hiptensorCreate", names)
        self.assertIn("hiptensorDestroy", names)

    @unittest.skipUnless(STUB_SOURCE.exists(), f"stub source not found: {STUB_SOURCE}")
    def test_every_public_function_is_defined_in_stub(self):
        # Mirrors the build-time link check at the source level: every public API
        # function must have a definition in the stub.
        names = gen.parse_exported_functions(PUBLIC_HEADER)
        stub_text = STUB_SOURCE.read_text()
        missing = [
            name
            for name in names
            # A definition is "<name>( ... ) {"; a bare declaration would not match.
            if not _has_definition(stub_text, name)
        ]
        self.assertEqual(missing, [], f"stub is missing definitions for: {missing}")


def _has_definition(source: str, name: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", source) is not None


if __name__ == "__main__":
    unittest.main()
