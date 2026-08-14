# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for opt-in source locations and DWARF line-table emission.

Covers:
  * default off: no ``Op.loc``, no ``debug_info`` attr, and emitted LLVM IR
    byte-identical to a build with the feature absent (the byte-identity gate
    and the IR goldens depend on this);
  * capture on: every op gets a location, including the control-flow ops that
    build their ``Op`` directly instead of going through ``IRBuilder._op``;
  * emitted metadata: the ``Debug Info Version`` module flag (without which
    LLVM silently drops every ``!dbg``), a ``DISubprogram`` on the kernel, one
    ``!dbg`` per instruction and never two on one instruction;
  * multi-file kernels are attributed per file via ``DILexicalBlockFile``;
  * locations survive the ``ck.dsl.ir/v1`` round-trip, so the C++ engine sees
    them too, and the C++ engine emits the same bytes from them -- including
    for a native Windows path, which the two ``os.path`` modules split
    differently;
  * an installed rocke captures the same locations as a checkout;
  * the metadata survives assembly and codegen into a real AMDGPU object, which
    is where every downstream consumer actually reads it from.
"""

from __future__ import annotations

import itertools
import ntpath
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from rocke.core import ir as ir_mod
from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm

THIS_FILE = os.path.abspath(__file__)


def build_flat(capture_loc=None):
    """A straight-line kernel: load, add, store."""
    b = IRBuilder("dbg_flat", capture_loc=capture_loc)
    x = b.param("X", PtrType(F32, "global"), noalias=True, align=16)
    i = b.const_i32(0)
    v = b.global_load_f32(x, i)
    s = b.fadd(v, b.const_f32(1.0))
    b.global_store(x, i, s)
    b.ret()
    return b.kernel


def build_loop(capture_loc=None):
    """A kernel whose scf.for lowers to a header/body/latch/exit diamond."""
    b = IRBuilder("dbg_loop", capture_loc=capture_loc)
    x = b.param("X", PtrType(F32, "global"), noalias=True, align=16)
    c0 = b.const_i32(0)
    loop = b.scf_for_iter(
        c0, b.const_i32(8), b.const_i32(1), [("acc", b.const_f32(0.0))], iv_name="k"
    )
    with loop as (k, (acc,)):
        b.scf_yield(b.fadd(acc, b.global_load_f32(x, k)))
    b.global_store(x, c0, loop.results[0])
    b.ret()
    return b.kernel


def body_instructions(ll):
    """The instruction lines inside the kernel definition, minus block labels."""
    body = ll.split("define ", 1)[1].split("\n}", 1)[0].split("\n")[1:]
    return [
        line
        for line in body
        if line.strip() and not re.match(r"^[\w.]+:\s*$", line.strip())
    ]


class TestDefaultOff(unittest.TestCase):
    """The feature must be invisible unless asked for."""

    def test_no_locations_and_no_attr(self):
        kernel = build_flat()
        self.assertNotIn("debug_info", kernel.attrs)
        self.assertEqual(
            [op.loc for op in kernel.body.ops], [None] * len(kernel.body.ops)
        )

    def test_emitted_ir_carries_no_debug_metadata(self):
        for build in (build_flat, build_loop):
            ll = lower_kernel_to_llvm(build(), arch="gfx950")
            for token in (
                "!dbg",
                "DICompileUnit",
                "DISubprogram",
                "DILocation",
                "llvm.dbg.cu",
                "Debug Info Version",
            ):
                self.assertNotIn(token, ll, f"{build.__name__} emitted {token}")

    def test_explicit_false_matches_default(self):
        self.assertEqual(
            lower_kernel_to_llvm(build_flat(capture_loc=False), arch="gfx950"),
            lower_kernel_to_llvm(build_flat(capture_loc=None), arch="gfx950"),
        )

    def test_env_var_drives_the_default(self):
        prev = os.environ.get(ir_mod.LOC_CAPTURE_ENV)
        try:
            os.environ[ir_mod.LOC_CAPTURE_ENV] = "1"
            self.assertTrue(ir_mod.loc_capture_default())
            self.assertIn("debug_info", build_flat().attrs)
            os.environ[ir_mod.LOC_CAPTURE_ENV] = "0"
            self.assertFalse(ir_mod.loc_capture_default())
            self.assertNotIn("debug_info", build_flat().attrs)
        finally:
            if prev is None:
                os.environ.pop(ir_mod.LOC_CAPTURE_ENV, None)
            else:
                os.environ[ir_mod.LOC_CAPTURE_ENV] = prev

    def test_explicit_flag_overrides_the_env_var(self):
        prev = os.environ.get(ir_mod.LOC_CAPTURE_ENV)
        try:
            os.environ[ir_mod.LOC_CAPTURE_ENV] = "1"
            self.assertNotIn("debug_info", build_flat(capture_loc=False).attrs)
        finally:
            if prev is None:
                os.environ.pop(ir_mod.LOC_CAPTURE_ENV, None)
            else:
                os.environ[ir_mod.LOC_CAPTURE_ENV] = prev


def frames(loc):
    """Split a captured loc into (path, line, col, func) tuples, innermost first."""
    out = []
    for part in loc.split(ir_mod.LOC_FRAME_SEP):
        path, line, col, func = part.rsplit(":", 3)
        out.append((path, int(line), int(col), func))
    return out


class TestLocationCapture(unittest.TestCase):
    def test_every_op_gets_a_location_in_this_file(self):
        kernel = build_flat(capture_loc=True)
        self.assertTrue(kernel.attrs["debug_info"])
        for op in kernel.body.ops:
            self.assertIsNotNone(op.loc, f"{op.name} has no location")
            path, line, _, func = frames(op.loc)[0]
            # The innermost frame outside core/ is this test file, not ir.py.
            self.assertEqual(os.path.abspath(path), THIS_FILE)
            self.assertGreater(line, 0)
            self.assertEqual(func, "build_flat")

    def test_control_flow_ops_are_covered(self):
        """scf.for builds its Op directly, so _op alone would have missed it."""
        kernel = build_loop(capture_loc=True)
        for_ops = [op for op in kernel.body.ops if op.name == "scf.for"]
        self.assertEqual(len(for_ops), 1)
        self.assertIsNotNone(for_ops[0].loc)
        nested = for_ops[0].regions[0].ops
        self.assertTrue(nested)
        for op in nested:
            self.assertIsNotNone(op.loc, f"nested {op.name} has no location")

    def test_paths_are_absolute(self):
        for op in build_flat(capture_loc=True).body.ops:
            for path, _, _, _ in frames(op.loc):
                self.assertTrue(os.path.isabs(path))

    def test_the_whole_call_stack_is_captured(self):
        """The chain is what makes a one-line helper interpretable."""
        kernel = build_flat(capture_loc=True)
        chain = frames(kernel.body.ops[0].loc)
        self.assertGreater(len(chain), 1, "expected the caller above build_flat")
        funcs = [f for _, _, _, f in chain]
        self.assertEqual(funcs[0], "build_flat")
        self.assertIn("test_the_whole_call_stack_is_captured", funcs)

    def test_test_runner_frames_are_excluded(self):
        """unittest and site-packages frames emitted no instruction."""
        for path, _, _, _ in frames(build_flat(capture_loc=True).body.ops[0].loc):
            self.assertNotIn("site-packages", path)
            self.assertFalse(path.startswith("<"))
            self.assertNotEqual(os.path.basename(path), "case.py")  # unittest

    def test_columns_separate_ops_that_share_a_line(self):
        b = IRBuilder("dbg_cols", capture_loc=True)
        x = b.param("X", PtrType(F32, "global"), noalias=True, align=16)
        # Three ops, one line: without columns they would be indistinguishable.
        b.global_store(x, b.const_i32(0), b.fadd(b.const_f32(1.0), b.const_f32(2.0)))
        b.ret()
        positions = set()
        for op in b.kernel.body.ops:
            _, line, col, _ = frames(op.loc)[0]
            positions.add((line, col))
        lines = {line for line, _ in positions}
        self.assertLess(len(lines), len(positions), "columns did not disambiguate")

    def test_frame_chain_is_depth_capped(self):
        def recurse(n, builder):
            if n:
                return recurse(n - 1, builder)
            return builder.const_i32(0)

        b = IRBuilder("dbg_deep", capture_loc=True)
        recurse(40, b)
        self.assertLessEqual(len(frames(b.kernel.body.ops[0].loc)), 16)


class TestEmittedDebugMetadata(unittest.TestCase):
    def setUp(self):
        self.ll = lower_kernel_to_llvm(build_loop(capture_loc=True), arch="gfx950")

    def test_module_flag_is_present(self):
        # LLVM discards every !dbg attachment without this, and says nothing.
        self.assertIn('!"Debug Info Version", i32 3', self.ll)
        self.assertRegex(self.ll, r"!llvm\.module\.flags = !\{!\d+\}")
        self.assertRegex(self.ll, r"!llvm\.dbg\.cu = !\{!\d+\}")

    def test_compile_unit_is_line_tables_only(self):
        self.assertIn("emissionKind: LineTablesOnly", self.ll)
        self.assertRegex(self.ll, r"!\d+ = distinct !DICompileUnit\(")

    def test_kernel_definition_carries_a_subprogram(self):
        define = next(
            line for line in self.ll.split("\n") if line.startswith("define ")
        )
        match = re.search(r"#0 !dbg !(\d+) \{$", define)
        self.assertIsNotNone(match, define)
        self.assertIn(
            f'!{match.group(1)} = distinct !DISubprogram(name: "dbg_loop"', self.ll
        )

    def test_every_instruction_is_labelled_exactly_once(self):
        for line in body_instructions(self.ll):
            self.assertIn(", !dbg !", line, f"unlabelled: {line.strip()}")
            self.assertEqual(line.count("!dbg"), 1, f"duplicate !dbg: {line.strip()}")

    def test_locations_point_at_this_test_file(self):
        self.assertIn(f'!DIFile(filename: "{os.path.basename(THIS_FILE)}"', self.ll)
        self.assertIn(f'directory: "{os.path.dirname(THIS_FILE)}"', self.ll)
        for line in re.findall(r"!DILocation\(line: (\d+)", self.ll):
            self.assertGreater(int(line), 0)

    def test_referenced_metadata_ids_are_all_defined(self):
        defined = {int(m) for m in re.findall(r"^!(\d+) = ", self.ll, re.M)}
        for ref in re.findall(r"!(\d+)", self.ll):
            self.assertIn(int(ref), defined, f"dangling metadata reference !{ref}")

    def test_debug_metadata_does_not_collide_with_amdgpu_markers(self):
        """The lowerer hardcodes !1/!2/!3; debug nodes must not reuse them."""
        ids = {int(m) for m in re.findall(r"^!(\d+) = ", self.ll, re.M)}
        self.assertFalse(ids & {1, 2, 3})


class TestInliningChains(unittest.TestCase):
    """A captured stack becomes DILocations linked by inlinedAt."""

    def setUp(self):
        self.ll = lower_kernel_to_llvm(build_flat(capture_loc=True), arch="gfx950")

    def test_inlined_at_chain_is_emitted(self):
        self.assertRegex(self.ll, r"!DILocation\([^)]*inlinedAt: !\d+\)")

    def test_each_python_function_gets_its_own_subprogram(self):
        names = set(re.findall(r'!DISubprogram\(name: "([^"]+)"', self.ll))
        # The kernel's own subprogram plus the builder function it was inlined from.
        self.assertIn("dbg_flat", names)
        self.assertIn("build_flat", names)

    def test_chain_ends_at_the_kernel_subprogram(self):
        """LLVM requires the outermost inlinedAt to be scoped to the function."""
        define = next(
            line for line in self.ll.split("\n") if line.startswith("define ")
        )
        kernel_sp = re.search(r"#0 !dbg !(\d+) \{$", define).group(1)
        locs = dict(re.findall(r"!(\d+) = !DILocation\((.*)\)$", self.ll, re.M))
        chained = [mid for mid, body in locs.items() if "inlinedAt" in body]
        self.assertTrue(chained)
        for mid in chained:
            # Walk out to the end of the chain.
            seen = set()
            while "inlinedAt" in locs[mid]:
                self.assertNotIn(mid, seen, "inlinedAt chain loops")
                seen.add(mid)
                mid = re.search(r"inlinedAt: !(\d+)", locs[mid]).group(1)
            self.assertRegex(locs[mid], rf"scope: !{kernel_sp}\b")

    def test_no_dangling_or_self_referential_metadata(self):
        defined = {int(m) for m in re.findall(r"^!(\d+) = ", self.ll, re.M)}
        for ref in re.findall(r"!(\d+)", self.ll):
            self.assertIn(int(ref), defined)


class TestSingleFrameFallback(unittest.TestCase):
    """Locations that arrive without a call stack still work."""

    def test_lone_frame_in_another_file_uses_a_lexical_block_file(self):
        kernel = build_flat(capture_loc=True)
        other = os.path.join(os.path.dirname(THIS_FILE), "helper_emitter.py")
        kernel.body.ops[-1].loc = f"{other}:42"
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn('!DIFile(filename: "helper_emitter.py"', ll)
        scope = re.search(r"!(\d+) = !DILexicalBlockFile", ll).group(1)
        self.assertRegex(ll, rf"!DILocation\(line: 42, column: 0, scope: !{scope}\)")

    def test_unparseable_location_is_skipped_not_fatal(self):
        kernel = build_flat(capture_loc=True)
        kernel.body.ops[0].loc = "no-line-number-here"
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("DICompileUnit", ll)

    def test_a_path_containing_a_colon_still_parses(self):
        kernel = build_flat(capture_loc=True)
        kernel.body.ops[-1].loc = "/tmp/od:d/weird.py:7"
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn('filename: "weird.py"', ll)
        self.assertIn('directory: "/tmp/od:d"', ll)


class TestInstalledLayout(unittest.TestCase):
    """An installed rocke authors kernels just like a checkout does.

    ``helpers/`` and ``instances/`` are where a shipped kernel is written, so
    those frames are the ones worth showing. Treating them as the harness that
    launched the build -- which is what classifying all of site-packages as a
    runner did -- ends the stack walk before it captures anything, and a
    pip-installed rocke silently produced no locations at all while a checkout
    produced a full chain.
    """

    CHECKOUT = os.path.join("/src", "rocke", "platform", "python", "rocke")
    INSTALLED = os.path.join("/venv", "lib", "python3.12", "site-packages", "rocke")

    def roles_under(self, root):
        """Roles of the same modules with the package rooted at ``root``."""
        saved = (ir_mod._CORE_PREFIX, ir_mod._ROCKE_PREFIX, ir_mod._FRAME_ROLE)
        ir_mod._CORE_PREFIX = os.path.join(root, "core") + os.sep
        ir_mod._ROCKE_PREFIX = root + os.sep
        ir_mod._FRAME_ROLE = {}
        try:
            return {
                name: ir_mod._frame_role(os.path.join(root, *name.split("/")))
                for name in ("core/ir.py", "helpers/loads.py", "instances/common/x.py")
            }
        finally:
            (ir_mod._CORE_PREFIX, ir_mod._ROCKE_PREFIX, ir_mod._FRAME_ROLE) = saved

    def test_installed_and_checkout_agree(self):
        self.assertEqual(
            self.roles_under(self.INSTALLED), self.roles_under(self.CHECKOUT)
        )

    def test_authoring_modules_are_user_code_when_installed(self):
        roles = self.roles_under(self.INSTALLED)
        self.assertEqual(roles["helpers/loads.py"], "user")
        self.assertEqual(roles["instances/common/x.py"], "user")
        self.assertEqual(roles["core/ir.py"], "core")

    def test_other_site_packages_are_still_runners(self):
        """Only rocke is exempt; a test runner installed beside it still stops
        the walk, or every kernel would carry pytest's frames."""
        saved = (ir_mod._ROCKE_PREFIX, ir_mod._FRAME_ROLE)
        ir_mod._ROCKE_PREFIX = self.INSTALLED + os.sep
        ir_mod._FRAME_ROLE = {}
        try:
            beside = os.path.dirname(self.INSTALLED)
            self.assertEqual(
                ir_mod._frame_role(os.path.join(beside, "_pytest", "python.py")),
                "runner",
            )
            self.assertEqual(
                ir_mod._frame_role(
                    os.path.join(ir_mod._STDLIB_DIR, "unittest", "case.py")
                ),
                "runner",
            )
        finally:
            (ir_mod._ROCKE_PREFIX, ir_mod._FRAME_ROLE) = saved

    def test_capture_reaches_the_caller_through_a_package_helper(self):
        """The end-to-end shape of the bug: an op emitted by a rocke helper.

        With the helper misread as a runner the walk stopped at it and the op
        came back with no location at all, so this asserts the chain reaches
        back out to the test.
        """
        loc = build_flat(capture_loc=True).body.ops[0].loc
        self.assertIsNotNone(loc)
        funcs = [f for _, _, _, f in frames(loc)]
        self.assertIn("test_capture_reaches_the_caller_through_a_package_helper", funcs)


def engine_or_skip():
    try:
        import rocke_engine  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"rocke_engine not importable: {exc}")


class TestBothEnginesEmitTheSameDebugInfo(unittest.TestCase):
    """The C++ engine is the default; it must emit the same metadata.

    ``debug_info`` and every ``@loc`` ride the ck.dsl.ir/v1 serialization, so
    the engine has what the Python lowerer has. When it ignored them, a normal
    installation ran the capture and produced an object with no DWARF in it --
    and ``ROCKE_BACKEND=both`` failed outright, because the two engines really
    were emitting different bytes.
    """

    def lower_with(self, backend, kernel):
        prev = os.environ.get("ROCKE_BACKEND")
        os.environ["ROCKE_BACKEND"] = backend
        try:
            return lower_kernel_to_llvm(kernel, arch="gfx950")
        finally:
            if prev is None:
                os.environ.pop("ROCKE_BACKEND", None)
            else:
                os.environ["ROCKE_BACKEND"] = prev

    def test_engines_agree_byte_for_byte(self):
        engine_or_skip()
        for build in (build_flat, build_loop):
            kernel = build(capture_loc=True)
            cpp = self.lower_with("cpp", kernel)
            python = self.lower_with("python", kernel)
            self.assertIn("!dbg", cpp, f"{build.__name__}: cpp emitted no debug info")
            self.assertEqual(cpp, python, build.__name__)

    def test_both_mode_accepts_a_debug_build(self):
        """`both` is the gate that would catch any drift, so run it directly."""
        engine_or_skip()
        ll = self.lower_with("both", build_loop(capture_loc=True))
        self.assertIn("DICompileUnit", ll)

    def test_engines_agree_with_capture_off(self):
        engine_or_skip()
        kernel = build_loop(capture_loc=False)
        self.assertEqual(
            self.lower_with("cpp", kernel), self.lower_with("python", kernel)
        )

    def test_engines_agree_on_a_windows_style_location(self):
        """A drive-letter path has to split the same way in both engines.

        The Python side names files with ``os.path.split``, which is ntpath on
        Windows and posixpath everywhere else, so on this host the backslashes
        are ordinary filename bytes and the whole path is the basename. A C++
        side that split on backslash regardless of platform would disagree here,
        and one that never split on it would disagree on Windows -- where it did,
        emitting the entire path as the filename with no directory at all.
        """
        engine_or_skip()
        kernel = build_flat(capture_loc=True)
        kernel.body.ops[-1].loc = r"C:\proj\rocke\emit.py:42"
        cpp = self.lower_with("cpp", kernel)
        self.assertEqual(cpp, self.lower_with("python", kernel))
        # Backslashes survive as escaped pairs in the metadata string.
        directory, filename = os.path.split(r"C:\proj\rocke\emit.py")
        self.assertIn(
            'filename: "{}", directory: "{}"'.format(
                filename.replace("\\", "\\\\"), directory.replace("\\", "\\\\")
            ),
            cpp,
        )


# Enough to pin the separator handling: absolute and relative paths on both
# platforms, the roots, UNC and verbatim prefixes, drive-relative paths, and the
# doubled separators where off-by-one splits hide.
PATH_SPLIT_CASES = (
    r"C:\Users\me\proj\file.py",
    r"C:\file.py",
    r"C:file.py",
    "C:\\",
    "C:",
    "C",
    "file.py",
    r"\file.py",
    r"a\b\c.py",
    "",
    r"\\server\share\file.py",
    r"\\server\share",
    "\\\\server\\share\\\\",
    r"\\server",
    "\\\\",
    r"\\?\C:\long\path\file.py",
    r"\\?\UNC\srv\shr\f.py",
    r"\\?\unc\srv\shr\f.py",
    r"\\.\device\f.py",
    "//server/share/file.py",
    "/usr/lib/x.py",
    "/x.py",
    "//x.py",
    "///x.py",
    "/a//x.py",
    "/",
    "//",
    "dir/",
    "dir//",
    "dir\\",
    "C:/Users/me/file.py",
    "C:\\Users\\me/file.py",
    "/tmp/od:d/weird.py",
    "/:",
    "/:/",
    ".",
    "..",
)

# Reads paths on stdin and reports where the mirror cuts each one, under both
# platforms' rules, as byte offsets the caller resolves against its own copy.
_SPLIT_DRIVER = """
#include <stdio.h>
#include <string.h>
#include "rocke/py_path_split.h"

int main(void)
{
    char line[8192];
    while(fgets(line, sizeof(line), stdin))
    {
        size_t n = strlen(line);
        while(n && (line[n - 1] == '\\n' || line[n - 1] == '\\r'))
            n--;
        rocke_py_path_split_t nt = rocke_py_path_split(line, n, true);
        rocke_py_path_split_t px = rocke_py_path_split(line, n, false);
        printf("%zu %zu %zu %zu\\n", nt.head_len, nt.tail_off, px.head_len, px.tail_off);
    }
    return 0;
}
"""


class TestPythonPathSplitMirror(unittest.TestCase):
    """The C++ DIFile split must agree with ``os.path.split`` on both platforms.

    Byte-identity between the engines is a per-host property: the Python side
    calls ``os.path.split``, which is a different module on Windows than it is
    here. Only the POSIX half of that is reachable through the lowerers on this
    host, so this compiles the split by itself and checks each half against the
    module it mirrors -- ntpath included, since a Windows host is exactly where
    the divergence this covers was invisible.
    """

    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++") or shutil.which("g++")
        if cls.compiler is None:
            raise unittest.SkipTest("no host C++ compiler")
        cls.include = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE))),
            "cpp",
            "include",
        )
        header = os.path.join(cls.include, "rocke", "py_path_split.h")
        if not os.path.isfile(header):
            raise unittest.SkipTest(f"missing {header}")

    def splits(self, paths):
        """``(ntpath, posixpath)`` splits of each path, as the C++ mirror cuts them."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "driver.cpp")
            exe = os.path.join(tmp, "driver")
            with open(src, "w") as fh:
                fh.write(_SPLIT_DRIVER)
            subprocess.run(
                [self.compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror"]
                + ["-I", self.include, "-o", exe, src],
                check=True,
                capture_output=True,
                text=True,
            )
            out = subprocess.run(
                [exe],
                input="\n".join(paths) + "\n",
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        self.assertEqual(len(out), len(paths))
        cuts = []
        for path, line in zip(paths, out):
            nt_head, nt_tail, px_head, px_tail = (int(v) for v in line.split())
            cuts.append(
                (
                    (path[:nt_head], path[nt_tail:]),
                    (path[:px_head], path[px_tail:]),
                )
            )
        return cuts

    def test_matches_ntpath_and_posixpath(self):
        for path, got in zip(PATH_SPLIT_CASES, self.splits(PATH_SPLIT_CASES)):
            self.assertEqual(
                got, (ntpath.split(path), posixpath.split(path)), f"splitting {path!r}"
            )

    def test_matches_over_every_short_separator_string(self):
        """Exhaustive over the alphabet that decides a split, to length four.

        The fixed cases above are the paths a Python frame really produces; this
        is the part that catches an off-by-one in a root or prefix nobody thought
        to write down.
        """
        alphabet = "/\\:aCUN?"
        paths = [
            "".join(combo)
            for n in range(1, 5)
            for combo in itertools.product(alphabet, repeat=n)
        ]
        want = [(ntpath.split(p), posixpath.split(p)) for p in paths]
        self.assertEqual(self.splits(paths), want)


LLVM_BIN_ENV = "ROCKE_TEST_LLVM_BIN"


def llvm_tool(name):
    """An LLVM tool from the configured toolchain, PATH, or the ROCm default.

    An explicitly set ``ROCM_PATH`` (or ``ROCKE_TEST_LLVM_BIN``) is preferred
    over PATH, because a system LLVM of a different major version shadowing the
    AMD one is a normal state for a host to be in, and the kernel is going to be
    built by the AMD one. The system tool assembles the IR happily and then
    reports none of the inlining, which reads as this test failing rather than
    as the wrong ``llc`` having answered. PATH stays the route when nothing is
    configured, so a hand-arranged toolchain still wins.
    """
    configured = []
    if os.environ.get(LLVM_BIN_ENV):
        configured.append(os.environ[LLVM_BIN_ENV])
    if os.environ.get("ROCM_PATH"):
        configured.append(os.path.join(os.environ["ROCM_PATH"], "llvm", "bin"))
    for directory in configured:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    candidate = os.path.join("/opt/rocm", "llvm", "bin", name)
    return candidate if os.path.isfile(candidate) else None


class TestLlvmToolDiscovery(unittest.TestCase):
    """Which ``llc`` answers decides whether the round trip below means anything.

    A host with a distro LLVM ahead of the AMD one on PATH is ordinary, and the
    older tool assembles the IR without complaint while reporting none of the
    inlining -- so the round trip failed on a host that had a perfectly good
    toolchain installed, and the failure pointed at the metadata instead of at
    the toolchain.
    """

    def fake_bin(self, tmp, name):
        directory = os.path.join(tmp, "llvm", "bin")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w"):
            pass
        return path

    def test_configured_rocm_wins_over_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = self.fake_bin(tmp, "llc")
            with mock.patch.dict(os.environ, {"ROCM_PATH": tmp}) as env:
                # An inherited override outranks ROCM_PATH, so a host that has
                # one set -- which is what the variable is for -- would answer
                # from a route above the one under test.
                env.pop(LLVM_BIN_ENV, None)
                self.assertEqual(llvm_tool("llc"), expected)

    def test_explicit_llvm_bin_wins_over_rocm_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "override")
            os.makedirs(override)
            expected = os.path.join(override, "llc")
            with open(expected, "w"):
                pass
            self.fake_bin(tmp, "llc")
            with mock.patch.dict(
                os.environ, {"ROCM_PATH": tmp, LLVM_BIN_ENV: override}
            ):
                self.assertEqual(llvm_tool("llc"), expected)

    def test_path_answers_when_nothing_is_configured(self):
        """PATH stays the normal route for a deliberately arranged toolchain."""
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("ROCM_PATH", None)
            env.pop(LLVM_BIN_ENV, None)
            on_path = shutil.which("llc")
            if on_path is None:
                self.skipTest("no llc on PATH")
            self.assertEqual(llvm_tool("llc"), on_path)

    def test_a_configured_root_without_the_tool_falls_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"ROCM_PATH": tmp}) as env:
                env.pop(LLVM_BIN_ENV, None)
                self.assertEqual(llvm_tool("llc"), shutil.which("llc"))


class TestObjectRoundTrip(unittest.TestCase):
    """The metadata has to survive into a real code object, not just the IR.

    Everything downstream reads DWARF out of the object rocprofv3 dumps, so
    asserting on the ``.ll`` alone would not notice the emitted metadata being
    dropped by the assembler or the AMDGPU backend. This runs the same path the
    capture does -- assemble, codegen to an object, dump the DWARF -- and checks
    the inlining survives it.
    """

    @classmethod
    def setUpClass(cls):
        cls.tools = {n: llvm_tool(n) for n in ("llvm-as", "llc", "llvm-dwarfdump")}
        missing = [n for n, p in cls.tools.items() if p is None]
        if missing:
            raise unittest.SkipTest(f"missing LLVM tools: {', '.join(missing)}")

    def dwarf_for(self, kernel):
        ll = lower_kernel_to_llvm(kernel, arch="gfx950")
        with tempfile.TemporaryDirectory() as tmp:
            paths = {ext: os.path.join(tmp, f"k.{ext}") for ext in ("ll", "bc", "o")}
            with open(paths["ll"], "w") as fh:
                fh.write(ll)
            subprocess.run(
                [self.tools["llvm-as"], paths["ll"], "-o", paths["bc"]], check=True
            )
            subprocess.run(
                [
                    self.tools["llc"],
                    "-mtriple=amdgcn-amd-amdhsa",
                    "-mcpu=gfx950",
                    "-filetype=obj",
                    paths["bc"],
                    "-o",
                    paths["o"],
                ],
                check=True,
            )
            return subprocess.run(
                [self.tools["llvm-dwarfdump"], "--debug-info", paths["o"]],
                check=True,
                capture_output=True,
                text=True,
            ).stdout

    def test_inlined_subroutines_reach_the_object(self):
        dwarf = self.dwarf_for(build_flat(capture_loc=True))
        self.assertIn("DW_TAG_compile_unit", dwarf)
        self.assertIn("DW_TAG_subprogram", dwarf)
        # The inline tree is the whole point: without it the sidecar has only
        # the leaf line and cannot say which phase of the kernel asked for it.
        self.assertIn("DW_TAG_inlined_subroutine", dwarf)

    def test_call_sites_survive_with_file_and_line(self):
        dwarf = self.dwarf_for(build_flat(capture_loc=True))
        for attr in ("DW_AT_call_file", "DW_AT_call_line", "DW_AT_abstract_origin"):
            self.assertIn(attr, dwarf, f"{attr} did not survive to the object")
        self.assertIn(os.path.basename(THIS_FILE), dwarf)

    def test_the_builder_function_is_named(self):
        """The sidecar labels a frame with this name, so it has to be there."""
        dwarf = self.dwarf_for(build_flat(capture_loc=True))
        self.assertRegex(dwarf, r'DW_AT_name\s+\("build_flat"\)')

    def test_capture_off_emits_no_debug_info_section(self):
        self.assertNotIn("DW_TAG_subprogram", self.dwarf_for(build_flat()))


class TestSerializationRoundTrip(unittest.TestCase):
    def test_locations_survive_and_lower_identically(self):
        kernel = build_loop(capture_loc=True)
        text = serialize(kernel)
        self.assertIn("@loc ", text)
        reparsed = parse(text)
        self.assertEqual(serialize(reparsed), text)
        self.assertEqual(
            lower_kernel_to_llvm(reparsed, arch="gfx950"),
            lower_kernel_to_llvm(kernel, arch="gfx950"),
        )


if __name__ == "__main__":
    unittest.main()
