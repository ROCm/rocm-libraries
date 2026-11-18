#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_bare_link.py.

They build two small shared libraries shaped like a wrapper build: a private library
that defines a symbol, and a wrapper with a DT_NEEDED entry on it. A symbol only the
private library defines must fail a bare -lMIOpen link, and one the wrapper defines
must pass. That is the property the real check relies on, so a linker that stops
behaving this way, or a check that stops asking, fails here.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_check_bare_link
"""

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import check_bare_link as bare_link

COMPILER = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
C_COMPILER = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

PUBLIC = "miopenPublicEntry"
HIDDEN = "miopenHiddenEntry"


def build(directory, wrapper_defines):
    """Write libMIOpen_private.so.1 and libMIOpen.so.1 and their unversioned links.

    The private library defines the public entry's _impl form and the hidden entry.
    The wrapper defines `wrapper_defines` and calls into the private library.
    """
    directory = Path(directory)
    (directory / "private.c").write_text(
        f"int {PUBLIC}_impl(void) {{ return 1; }}\nint {HIDDEN}(void) {{ return 2; }}\n"
    )
    (directory / "wrapper.c").write_text(
        f"int {PUBLIC}_impl(void);\n"
        + "".join(
            f"int {name}(void) {{ return {PUBLIC}_impl(); }}\n"
            for name in wrapper_defines
        )
    )
    for command in (
        [C_COMPILER, "-shared", "-fPIC", "private.c", "-o", "libMIOpen_private.so.1"]
        + ["-Wl,-soname,libMIOpen_private.so.1"],
        [C_COMPILER, "-shared", "-fPIC", "wrapper.c", "-o", "libMIOpen.so.1"]
        + ["-Wl,-soname,libMIOpen.so.1", "-L.", "-l:libMIOpen_private.so.1"],
    ):
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
    for name in ("libMIOpen_private", "libMIOpen"):
        (directory / f"{name}.so").symlink_to(f"{name}.so.1")


@unittest.skipUnless(
    sys.platform.startswith("linux") and COMPILER and C_COMPILER,
    "needs Linux and a C and C++ compiler",
)
class LinkTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_symbols_the_wrapper_defines_resolve(self):
        build(self.dir.name, wrapper_defines=[PUBLIC, HIDDEN])
        ok, unresolved, _ = bare_link.check_bare_link(
            COMPILER, self.dir.name, {PUBLIC, HIDDEN}
        )
        self.assertTrue(ok)
        self.assertEqual(unresolved, [])

    def test_a_symbol_only_the_private_library_defines_is_unresolved(self):
        # The case the check exists for: the symbol is exported, but from the
        # indirect library, so only a link that names it finds it.
        build(self.dir.name, wrapper_defines=[PUBLIC])
        ok, unresolved, _ = bare_link.check_bare_link(
            COMPILER, self.dir.name, {PUBLIC, HIDDEN}
        )
        self.assertFalse(ok)
        self.assertEqual(unresolved, [HIDDEN])

    def test_naming_the_private_library_would_have_hidden_the_problem(self):
        build(self.dir.name, wrapper_defines=[PUBLIC])
        source = Path(self.dir.name) / "probe.cpp"
        source.write_text(bare_link.render_source({HIDDEN}))
        result = subprocess.run(
            [COMPILER, str(source), "-o", str(Path(self.dir.name) / "probe")]
            + [f"-L{self.dir.name}", "-lMIOpen", "-lMIOpen_private"]
            + [f"-Wl,-rpath-link,{self.dir.name}"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_missing_unversioned_link_is_an_error_not_a_pass(self):
        build(self.dir.name, wrapper_defines=[PUBLIC])
        (Path(self.dir.name) / "libMIOpen.so").unlink()
        with self.assertRaises(bare_link.AbiError):
            bare_link.check_bare_link(COMPILER, self.dir.name, {PUBLIC})


class SourceTest(unittest.TestCase):
    def test_every_symbol_is_declared_and_referenced_once(self):
        source = bare_link.render_source({"miopenB", "miopenA"})
        self.assertEqual(source.count("void miopenA(void);"), 1)
        self.assertEqual(source.count("void miopenB(void);"), 1)
        self.assertIn("    miopenA,\n    miopenB,\n", source)


class LinkerMessageTest(unittest.TestCase):
    """Linkers word their errors differently; the check reads both common forms."""

    def test_gnu_ld_and_lld_undefined_symbol_messages_are_read(self):
        gnu = "/usr/bin/ld: bare.o: undefined reference to `miopenFoo'"
        lld = "ld.lld: error: undefined symbol: miopenFoo\n>>> referenced by bare.o"
        self.assertEqual(bare_link.UNDEFINED_RE.findall(gnu), ["miopenFoo"])
        self.assertEqual(bare_link.UNDEFINED_RE.findall(lld), ["miopenFoo"])

    def test_a_linker_that_rejects_the_dt_needed_option_gets_a_retry(self):
        calls = []
        rejection = "ld.lld: error: unknown argument '--no-copy-dt-needed-entries'"

        def fake_link(compiler, source, output, lib_dir, copy_dt_needed_option):
            calls.append(copy_dt_needed_option)
            if copy_dt_needed_option:
                return subprocess.CompletedProcess([], 1, "", rejection)
            return subprocess.CompletedProcess([], 0, "", "")

        original = bare_link.link
        bare_link.link = fake_link
        try:
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "libMIOpen.so").touch()
                ok, unresolved, _ = bare_link.check_bare_link(
                    "c++", directory, {PUBLIC}
                )
        finally:
            bare_link.link = original
        self.assertTrue(ok)
        self.assertEqual(unresolved, [])
        self.assertEqual(calls, [True, False])


class MainTest(unittest.TestCase):
    def run_main(self, argv, windows=False):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = bare_link.main(argv, windows=windows)
        return code, out.getvalue()

    def test_windows_is_skipped(self):
        code, output = self.run_main([], windows=True)
        # 77 is the SKIP_RETURN_CODE that CMake gives the ctest entry.
        self.assertEqual(code, 77)
        self.assertIn("SKIP", output)

    def test_no_compiler_is_a_skip_unless_one_was_named(self):
        original = bare_link.shutil.which
        bare_link.shutil.which = lambda name: None
        try:
            code, output = self.run_main([])
            self.assertEqual(code, 77)
            self.assertIn("no C++ compiler", output)
            code, output = self.run_main(["--compiler", "no-such-compiler"])
        finally:
            bare_link.shutil.which = original
        self.assertEqual(code, 2)
        self.assertIn("compiler not found", output)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and COMPILER and C_COMPILER,
        "needs Linux and a C and C++ compiler",
    )
    def test_a_failure_names_the_symbol_and_the_remedy(self):
        with tempfile.TemporaryDirectory() as directory:
            build(directory, wrapper_defines=[PUBLIC])
            baseline = Path(directory) / "baseline.txt"
            baseline.write_text(f"# comment\n{PUBLIC}\n{HIDDEN}\n")
            code, output = self.run_main(
                ["--lib-dir", directory, "--baseline", str(baseline)]
            )
        self.assertEqual(code, 1)
        self.assertIn(f"  {HIDDEN}\n", output)
        self.assertNotIn(f"  {PUBLIC}\n", output)
        self.assertIn("generate_wrapper_sources.py", output)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and COMPILER and C_COMPILER,
        "needs Linux and a C and C++ compiler",
    )
    def test_success_reports_the_count(self):
        with tempfile.TemporaryDirectory() as directory:
            build(directory, wrapper_defines=[PUBLIC, HIDDEN])
            baseline = Path(directory) / "baseline.txt"
            baseline.write_text(f"{PUBLIC}\n{HIDDEN}\n")
            code, output = self.run_main(
                ["--lib-dir", directory, "--baseline", str(baseline)]
            )
        self.assertEqual(code, 0, output)
        self.assertIn("all 2 baseline symbols", output)


if __name__ == "__main__":
    unittest.main()
