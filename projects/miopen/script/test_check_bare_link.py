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
from unittest import mock

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


def run_check_with_linker(results, symbols):
    """Run check_bare_link with the linker replaced by canned results.

    `results` maps copy_dt_needed_option to (returncode, stderr). Returns the check's
    result and the list of copy_dt_needed_option values the linker was called with.
    """
    calls = []

    def fake_link(compiler, source, output, lib_dir, copy_dt_needed_option):
        calls.append(copy_dt_needed_option)
        returncode, stderr = results[copy_dt_needed_option]
        return subprocess.CompletedProcess([], returncode, "", stderr)

    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "libMIOpen.so").touch()
        with mock.patch.object(bare_link, "link", fake_link):
            result = bare_link.check_bare_link("c++", directory, symbols)
    return result, calls


class LinkerMessageTest(unittest.TestCase):
    """Linkers word their errors differently; the check reads each form."""

    def test_each_linkers_undefined_symbol_message_is_read(self):
        for linker, message in (
            ("GNU ld", "/usr/bin/ld: bare.o: undefined reference to `miopenFoo'"),
            (
                "gold",
                "bare.o:bare.cpp:function main: error: "
                "undefined reference to 'miopenFoo'",
            ),
            ("curly quotes", "undefined reference to \u2018miopenFoo\u2019"),
            (
                "lld",
                "ld.lld: error: undefined symbol: miopenFoo\n>>> referenced by bare.o",
            ),
        ):
            with self.subTest(linker=linker):
                (ok, unresolved, _), calls = run_check_with_linker(
                    {True: (1, message)}, {"miopenFoo"}
                )
                self.assertFalse(ok)
                self.assertEqual(unresolved, ["miopenFoo"])
                self.assertEqual(calls, [True])

    def test_every_unresolved_baseline_symbol_is_reported(self):
        message = "\n".join(
            [
                "ld.lld: error: undefined symbol: miopenB",
                "ld.lld: error: undefined symbol: miopenA",
                "ld.lld: error: undefined symbol: miopenB",
            ]
        )
        (ok, unresolved, _), _ = run_check_with_linker(
            {True: (1, message)}, {"miopenA", "miopenB", "miopenC"}
        )
        self.assertFalse(ok)
        self.assertEqual(unresolved, ["miopenA", "miopenB"])

    def test_an_undefined_symbol_outside_the_baseline_is_not_reported(self):
        message = (
            "ld.lld: error: undefined symbol: __gxx_personality_v0\n"
            "ld.lld: error: undefined symbol: miopenA"
        )
        (ok, unresolved, _), _ = run_check_with_linker(
            {True: (1, message)}, {"miopenA"}
        )
        self.assertFalse(ok)
        self.assertEqual(unresolved, ["miopenA"])

    def test_a_link_that_fails_for_another_reason_is_not_a_pass(self):
        message = "/usr/bin/ld: cannot find -lMIOpen: No such file or directory"
        (ok, unresolved, text), calls = run_check_with_linker(
            {True: (1, message)}, {PUBLIC}
        )
        self.assertFalse(ok)
        self.assertEqual(unresolved, [])
        self.assertIn(message, text)
        self.assertEqual(calls, [True])

    def test_a_linker_that_rejects_the_dt_needed_option_gets_a_retry(self):
        for linker, rejection in (
            ("lld", "ld.lld: error: unknown argument '--no-copy-dt-needed-entries'"),
            (
                "GNU ld",
                "/usr/bin/ld: unrecognized option '--no-copy-dt-needed-entries'",
            ),
            ("unknown option", "ld: unknown option: --no-copy-dt-needed-entries"),
            ("capitalised", "ld: Unknown Option --no-copy-dt-needed-entries"),
        ):
            with self.subTest(linker=linker):
                (ok, unresolved, _), calls = run_check_with_linker(
                    {True: (1, rejection), False: (0, "")}, {PUBLIC}
                )
                self.assertTrue(ok)
                self.assertEqual(unresolved, [])
                self.assertEqual(calls, [True, False])

    def test_a_failure_unrelated_to_the_dt_needed_option_is_not_retried(self):
        for message in (
            "ld.lld: error: unknown argument '--some-other-option'",
            "ld.lld: error: undefined symbol: miopenA",
            "unrecognized option\nnote: --no-copy-dt-needed-entries is not the cause",
        ):
            with self.subTest(message=message):
                (ok, _, _), calls = run_check_with_linker(
                    {True: (1, message)}, {"miopenA"}
                )
                self.assertFalse(ok)
                self.assertEqual(calls, [True])

    def test_a_rejection_after_which_the_retry_fails_is_still_reported(self):
        rejection = "ld.lld: error: unknown argument '--no-copy-dt-needed-entries'"
        undefined = "ld.lld: error: undefined symbol: miopenA"
        (ok, unresolved, _), calls = run_check_with_linker(
            {True: (1, rejection), False: (1, undefined)}, {"miopenA"}
        )
        self.assertFalse(ok)
        self.assertEqual(unresolved, ["miopenA"])
        self.assertEqual(calls, [True, False])

    def test_a_successful_link_is_not_retried_for_a_warning(self):
        warning = "ld: warning: unknown option '--no-copy-dt-needed-entries' ignored"
        (ok, unresolved, _), calls = run_check_with_linker(
            {True: (0, warning)}, {PUBLIC}
        )
        self.assertTrue(ok)
        self.assertEqual(unresolved, [])
        self.assertEqual(calls, [True])


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

    def test_a_link_failing_for_another_reason_prints_the_linker_output(self):
        message = "/usr/bin/ld: cannot find -lMIOpen: No such file or directory"

        def fake_link(compiler, source, output, lib_dir, copy_dt_needed_option):
            return subprocess.CompletedProcess([], 1, "", message)

        with tempfile.TemporaryDirectory() as directory:
            for name in ("libMIOpen.so", "libMIOpen.so.1", "libMIOpen_private.so.1"):
                (Path(directory) / name).touch()
            baseline = Path(directory) / "baseline.txt"
            baseline.write_text(f"{PUBLIC}\n")
            with mock.patch.object(bare_link, "link", fake_link):
                code, output = self.run_main(
                    ["--compiler", sys.executable, "--lib-dir", directory]
                    + ["--baseline", str(baseline)]
                )
        self.assertEqual(code, 1)
        self.assertNotIn("PASS", output)
        self.assertIn("could not resolve 0 symbols", output)
        self.assertIn("The link failed for another reason:", output)
        self.assertIn(message, output)

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
