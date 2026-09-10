#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The .so cache key must track the ctypes export set, and loudly.

setup_multiple_{rowcolquant,tensorquant}_dispatchers() reuse a cached .so when
the filename matches. The filename therefore has to encode the exported C ABI:
without it, an artifact built before dispatcher_get_tile_n()/dispatcher_get_pad_n()
existed is a name match, gets loaded, and fails at attribute lookup with a bare
"undefined symbol" -- on someone else's machine, in a cache directory, long after
the change that caused it.

The token used to be derived at import time by regex-scanning the ctypes source.
That is withdrawn (see the _SO_ABI comment in either utils module): a regex that
cannot read "unsigned long long dispatcher_x()" returns nothing for it, the export
is dropped, the token does not move, and the stale .so is reused anyway. A silent
parser on the cache-key path reproduces the very bug it was added to prevent.

The parse still happens -- here, where it is allowed to fail loudly. This file:

  1. extracts the export signatures with a parser that assumes nothing about the
     shape of the return type,
  2. checks that no dispatcher_* name in the source went unparsed, so a parser
     weakness is a red test and not a quiet omission,
  3. checks the source still matches a frozen snapshot of the signatures, and
  4. derives the expected token from that frozen snapshot, so updating the
     snapshot without updating the literal in the utils module fails too.

Changing the ctypes ABI therefore costs two red tests and two mechanical edits,
and cannot cost a silently frozen cache key.

Run: python3 -m pytest tests/test_so_abi_token.py -v
"""

import hashlib
import re
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import grouped_gemm_rowcolquant_utils as rc  # noqa: E402
import grouped_gemm_tensorquant_utils as tq  # noqa: E402

MODULES = {"rowcolquant": rc, "tensorquant": tq}

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")

# The prefix capture is deliberately shape-agnostic: everything on the line before
# the name, as long as it starts with an identifier character and contains none of
# ; { } ( ). That reads "int", "const char*", "unsigned long long", "std::size_t"
# and "const unsigned char*" alike, which the withdrawn runtime regex did not.
# Statements that merely call an export are excluded by _CONTROL_KEYWORDS below --
# "return dispatcher_initialize();" has a prefix, it is just not a return type --
# and a bare call has no prefix at all, so it never matches.
_DECLARATION = re.compile(
    r"^[ \t]*([A-Za-z_][^;{}()\n]*?)\s*\b(dispatcher_[A-Za-z0-9_]+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_CONTROL_KEYWORDS = frozenset(
    {"return", "if", "while", "for", "switch", "else", "do", "sizeof", "case"}
)

# Frozen snapshot of each operator's exported C ABI.
#
# If a test below reports that this no longer matches the ctypes source: update the
# entry to the printed value, then update _SO_ABI in the corresponding utils module
# to the token this file will then demand. Both edits are required; each is checked.
_RUN_GEMM = (
    "int dispatcher_run_gemm(const void* A, const void* B, const void* AQ, "
    "const void* BQ, void* C, int64_t M, int64_t N, int64_t K, "
    "int64_t stride_A, int64_t stride_B, int64_t stride_AQ, "
    "int64_t stride_BQ, int64_t stride_C, int64_t QK_A, int64_t QK_B, "
    "int k_batch, float* time_ms)"
)
_COMMON_EXPORTS = (
    "const char* dispatcher_get_kernel_name()",
    "int dispatcher_get_kernel_count()",
    "int dispatcher_get_pad_n()",
    "int dispatcher_get_tile_n()",
    "int dispatcher_init()",
    "int dispatcher_initialize()",
    _RUN_GEMM,
    "void dispatcher_cleanup()",
)
FROZEN_EXPORTS = {
    "rowcolquant": _COMMON_EXPORTS,
    "tensorquant": _COMMON_EXPORTS,
}


def strip_comments(text: str) -> str:
    """Remove C and C++ comments.

    Not cosmetic: the ctypes sources document their exports in prose, and
    " * Short-name alias for dispatcher_initialize()" is declaration-shaped to any
    line-anchored regex. A comment must not be able to move a cache key.
    """
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(" ", text))


def extern_c_block(text: str) -> str:
    """The `extern "C" { ... }` region, which is where the exports live."""
    start = text.index('extern "C"')
    return text[start : text.rindex("}")]


def parse_exports(text: str):
    """Return (signatures, unparsed_names) for one ctypes source.

    ``unparsed_names`` is every dispatcher_* identifier present in the export block
    that no signature accounted for. It is the point of this function: a parser that
    returns fewer exports than the file contains is exactly the failure that made the
    runtime derivation unsafe, so it is reported rather than absorbed.
    """
    block = extern_c_block(strip_comments(text))
    signatures = set()
    for ret, name, args in _DECLARATION.findall(block):
        tokens = ret.split()
        if not tokens or tokens[-1] in _CONTROL_KEYWORDS:
            continue
        params = (
            ", ".join(" ".join(a.split()) for a in args.split(",")) if args.strip() else ""
        )
        signatures.add("{} {}({})".format(" ".join(tokens), name, params))

    parsed_names = {s.split("(", 1)[0].rsplit(None, 1)[-1] for s in signatures}
    present_names = set(re.findall(r"\bdispatcher_[A-Za-z0-9_]+", block))
    return sorted(signatures), sorted(present_names - parsed_names)


def token_for(signatures) -> str:
    """The _SO_ABI literal a given export set requires."""
    return "r" + hashlib.sha256(";".join(sorted(signatures)).encode()).hexdigest()[:8]


class TestExportParser(unittest.TestCase):
    """The parser itself, on the shapes that broke its predecessor."""

    def _sigs(self, body):
        return parse_exports('extern "C" {\n' + body + "\n}\n")

    def test_reads_return_types_the_old_regex_dropped(self):
        for decl, expected in [
            ("int dispatcher_a()", "int dispatcher_a()"),
            ("const char* dispatcher_b()", "const char* dispatcher_b()"),
            ("unsigned long long dispatcher_c()", "unsigned long long dispatcher_c()"),
            ("std::size_t dispatcher_d()", "std::size_t dispatcher_d()"),
            (
                "const unsigned char* dispatcher_e()",
                "const unsigned char* dispatcher_e()",
            ),
            ("const char *dispatcher_f()", "const char * dispatcher_f()"),
            ("void dispatcher_g(int x)", "void dispatcher_g(int x)"),
        ]:
            with self.subTest(decl=decl):
                sigs, unparsed = self._sigs(decl + " { }")
                self.assertEqual(sigs, [expected])
                self.assertEqual(unparsed, [])

    def test_reports_a_name_it_could_not_parse(self):
        # Guard on the guard: if a future edit narrows the prefix pattern, this is
        # what turns the omission into a failure instead of a quiet drop.
        sigs, unparsed = self._sigs("int dispatcher_a()\n{\n    ;\n}\n(dispatcher_zz)()")
        self.assertEqual(sigs, ["int dispatcher_a()"])
        self.assertIn("dispatcher_zz", unparsed)

    def test_a_call_is_not_an_export(self):
        sigs, unparsed = self._sigs(
            "int dispatcher_a() { return 0; }\n"
            "int dispatcher_b()\n{\n    return dispatcher_a();\n}"
        )
        self.assertEqual(sigs, ["int dispatcher_a()", "int dispatcher_b()"])
        self.assertEqual(unparsed, [])

    def test_a_comment_is_not_an_export(self):
        sigs, unparsed = self._sigs(
            "// unsigned long long dispatcher_ghost()\n"
            "/* int dispatcher_phantom(); */\n"
            "int dispatcher_a() { return 0; }"
        )
        self.assertEqual(sigs, ["int dispatcher_a()"])
        self.assertEqual(unparsed, [])

    def test_the_token_moves_for_every_kind_of_abi_change(self):
        base = ("int dispatcher_a()", "void dispatcher_b(int x)")
        for label, mutated in [
            ("export added", base + ("std::size_t dispatcher_c()",)),
            ("export removed", base[:1]),
            ("return type changed", ("long dispatcher_a()", base[1])),
            ("parameter type changed", (base[0], "void dispatcher_b(long x)")),
        ]:
            with self.subTest(change=label):
                self.assertNotEqual(token_for(base), token_for(mutated))

    def test_the_token_is_order_independent(self):
        # It lands in a filename; declaration order must not invalidate a cache.
        base = ("int dispatcher_a()", "void dispatcher_b(int x)")
        self.assertEqual(token_for(base), token_for(tuple(reversed(base))))


class TestSoAbiToken(unittest.TestCase):
    def test_no_export_goes_unparsed(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                _, unparsed = parse_exports(mod._CTYPES_LIB_SRC.read_text())
                self.assertEqual(
                    unparsed,
                    [],
                    f"{unparsed} appear in {mod._CTYPES_LIB_SRC.name} but no signature "
                    "accounted for them; the parser in this file must be widened "
                    "before the ABI snapshot below can be trusted",
                )

    def test_source_matches_the_frozen_snapshot(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                sigs, _ = parse_exports(mod._CTYPES_LIB_SRC.read_text())
                self.assertEqual(
                    sigs,
                    sorted(FROZEN_EXPORTS[name]),
                    f"the exported C ABI of {mod._CTYPES_LIB_SRC.name} changed; "
                    f"update FROZEN_EXPORTS[{name!r}] in this file, then update "
                    "_SO_ABI in the matching utils module to the value the next "
                    "test demands",
                )

    def test_module_token_matches_the_snapshot(self):
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                expected = token_for(FROZEN_EXPORTS[name])
                self.assertEqual(
                    mod._SO_ABI,
                    expected,
                    f"set _SO_ABI = {expected!r} in {mod.__name__}; the cached .so "
                    "filename must move when the exported C ABI moves, or a stale "
                    "artifact is reused and fails later at attribute lookup",
                )

    def test_token_is_a_plain_literal(self):
        # No import-time file read, no fallback branch: an install that ships the
        # Python without the C++ sources must not key every build the same.
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                self.assertIsInstance(mod._SO_ABI, str)
                self.assertTrue(mod._SO_ABI.isalnum())
                for withdrawn in ("_ctypes_abi_token", "_SO_ABI_FALLBACK",
                                  "_EXPORT_SIGNATURE_RE"):
                    self.assertFalse(
                        hasattr(mod, withdrawn),
                        f"{withdrawn} is back on the runtime path in {mod.__name__}",
                    )

    def test_the_python_side_binds_only_exports_that_exist(self):
        # The failure the token exists to prevent is an attribute lookup on a symbol
        # the .so does not have. Check the binding sites against the parsed set.
        for name, mod in MODULES.items():
            with self.subTest(op=name):
                sigs, _ = parse_exports(mod._CTYPES_LIB_SRC.read_text())
                exported = {s.split("(", 1)[0].rsplit(None, 1)[-1] for s in sigs}
                bound = set(
                    re.findall(
                        r"\.(dispatcher_[A-Za-z0-9_]+)",
                        Path(mod.__file__).read_text(),
                    )
                )
                self.assertTrue(bound, "found no ctypes binding sites to check")
                self.assertEqual(bound - exported, set())


if __name__ == "__main__":
    unittest.main()
