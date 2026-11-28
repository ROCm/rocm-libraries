# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Overlapping work across open PRs: the extractors, body normalization and similarity,
the group rules (same new thing, same logic, copies of a develop pattern) with the coincidence filter, the land order,
the cache keyed by head, recorded decisions, and the checks wiring. Temp git repos, no network.
"""

import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import overlaps as ov  # noqa: E402
from rkmon_lib import review  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

K = "rk"


def g(repo, *a):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def names(defs):
    return [(d["kind"], d["name"]) for d in defs]


def by_name(defs, name):
    return next(d for d in defs if d["name"] == name)


STREQ = (
    "static bool rocke_ew_streq(const char* a, const char* b)\n{\n"
    "    return a != NULL && b != NULL && strcmp(a, b) == 0;\n}\n"
)


class TestPython(unittest.TestCase):
    SRC = '''"""Module doc."""
import os

LIMIT_BYTES = 4096
TABLE = {
    "a": 1,
    "b": 2,
}
lower = 3


def helper(x, y=2):
    """Doc that says what it does."""
    # a comment
    return x + y


async def fetch(url):
    return url


def _private(
        first,
        second,
):
    def nested():
        return first
    return nested() + second


class Spec:
    """A spec."""

    def block_size(self) -> int:
        return 256

    class Inner:
        def go(self):
            return 1
'''

    def test_definitions_kinds_names_and_lines(self):
        defs = ov.python_defs(self.SRC)
        self.assertEqual(
            names(defs),
            [
                ("const", "LIMIT_BYTES"),
                ("const", "TABLE"),
                ("def", "helper"),
                ("def", "fetch"),
                ("def", "_private"),
                ("class", "Spec"),
                ("method", "Spec.block_size"),
                ("class", "Spec.Inner"),
                ("method", "Spec.Inner.go"),
            ],
        )
        self.assertEqual(by_name(defs, "helper")["line"], 12)
        self.assertEqual(
            by_name(defs, "TABLE")["lines"], 4
        )  # a constant's value spans its brackets
        self.assertEqual(
            by_name(defs, "_private")["lines"], 7
        )  # a multi-line signature, the nested def inside
        self.assertTrue(by_name(defs, "_private")["private"])
        self.assertFalse(by_name(defs, "helper")["private"])
        self.assertNotIn("nested", [d["name"] for d in defs])

    def test_docstrings_comments_and_parameter_names_do_not_count(self):
        a = ov.python_defs(
            'def f(x, y):\n    """Doc."""\n    # note\n    return x + y\n'
        )[0]
        b = ov.python_defs("def g(p, q):\n    return p + q  # the same\n")[0]
        c = ov.python_defs("def h(p, q):\n    return p - q\n")[0]
        self.assertEqual(a["h"], b["h"])
        self.assertEqual(ov.similarity(a["sk"], b["sk"]), 1.0)
        self.assertNotEqual(a["h"], c["h"])  # identifiers and operators are kept


class TestCpp(unittest.TestCase):
    SRC = """#ifndef ROCKE_FOO_H
#define ROCKE_FOO_H
#include <string.h>
#define ROCKE_LIMIT 64
#define ROCKE_MAX(a, b) \\
    ((a) > (b) ? (a) : (b))

typedef struct
{
    int x;
    int y;
} rocke_point_t;

enum class Kind : int { A, B };

static constexpr int kTile = 16;
static const char* const GREETING = "hi";

bool streq(const char* a, const char* b);

static inline int
rocke_twice(int v)
{
    return 2 * v;
}

namespace
{
int hidden(int v) { return v; }
}  // namespace

extern "C" {
int exported(int v) { return v + 1; }
}

class Builder
{
  public:
    explicit Builder(const char* name) : n(name) {}
    ~Builder() { release(); }
    int size() const { return 3; }

  private:
    const char* n;
};

int Builder_total = compute_total(1);
Foo global_foo(1, 2);
TEST(Suite, Name) { EXPECT_TRUE(true); }
#endif
"""

    def test_definitions(self):
        defs = ov.cpp_defs(self.SRC)
        got = names(defs)
        for want in [
            ("macro", "ROCKE_LIMIT"),
            ("fmacro", "ROCKE_MAX"),
            ("type", "rocke_point_t"),
            ("enum", "Kind"),
            ("cconst", "kTile"),
            ("cconst", "GREETING"),
            ("cfunc", "streq"),
            ("cfunc", "rocke_twice"),
            ("cfunc", "hidden"),
            ("cfunc", "exported"),
            ("type", "Builder"),
            ("cfunc", "Builder::size"),
            ("cfunc", "Builder::Builder"),
            ("cfunc", "Builder::~Builder"),
        ]:
            self.assertIn(want, got)
        for not_a_def in (
            "ROCKE_FOO_H",
            "compute_total",
            "global_foo",
            "TEST",
            "Builder_total",
        ):
            self.assertNotIn(
                not_a_def, [n for _, n in got]
            )  # include guards, calls, objects, macro calls
        self.assertTrue(by_name(defs, "streq")["decl"])
        self.assertEqual(
            by_name(defs, "rocke_twice")["line"], 22
        )  # the name's line: the return type is above it
        self.assertTrue(by_name(defs, "rocke_twice")["private"])  # static
        self.assertTrue(by_name(defs, "hidden")["private"])  # anonymous namespace
        self.assertFalse(by_name(defs, "exported")["private"])
        self.assertEqual(by_name(defs, "ROCKE_MAX")["lines"], 2)  # a continued macro
        self.assertEqual(by_name(defs, "rocke_point_t")["line"], 8)

    def test_visibility_follows_the_file(self):
        hdr = ov.file_defs("rk/platform/cpp/include/rocke/foo.h", self.SRC)
        src = ov.file_defs("rk/platform/cpp/instances/common/foo.cpp", self.SRC)
        self.assertFalse(by_name(hdr, "ROCKE_LIMIT")["private"])
        self.assertTrue(
            by_name(src, "ROCKE_LIMIT")["private"]
        )  # a source file's macro: no other file sees it
        self.assertTrue(
            by_name(src, "Builder::size")["private"]
        )  # a member of a source file's class
        self.assertFalse(by_name(src, "streq")["private"])
        test = ov.file_defs("rk/platform/tests/core/test_foo.cpp", self.SRC)
        self.assertTrue(all(d["test"] for d in test))

    def test_renamed_copies_compare_equal(self):
        a = ov.cpp_defs(STREQ)[0]
        b = ov.cpp_defs(
            "static bool streq(const char* x, const char* y)\n{\n    // same\n    return x != NULL && "
            "y != NULL && strcmp(x, y) == 0;\n}\n"
        )[0]
        c = ov.cpp_defs(
            "static bool streq(const char* a, const char* b)\n{\n    return a != nullptr && b != nullptr "
            "&& strcmp(a, b) == 0;\n}\n"
        )[0]
        self.assertEqual(
            a["h"], b["h"]
        )  # the name, the parameter names and comments do not count
        self.assertGreater(ov.similarity(a["sk"], c["sk"]), ov.SIM_SAME)
        self.assertLess(ov.similarity(a["sk"], c["sk"]), 1.0)

    def test_odd_files_never_stop_the_scan(self):
        for text in (
            "{{{{ int f(",
            "}}}} ;;; )))",
            "class X : public\n",
            "#define\n",
            "template <typename T\nstruct",
            "namespace a { namespace b {\nint f() { return 1; }\n",
            '\x00\x01\xff garbage "unterminated',
        ):
            self.assertIsInstance(ov.file_defs("rk/x.cpp", text), list)
            self.assertIsInstance(ov.file_defs("rk/x.py", text.replace("{", "(")), list)
        self.assertEqual(
            names(
                ov.file_defs("rk/x.cpp", "namespace a {\nint f(int v) { return v; }\n")
            ),
            [("cfunc", "f")],
        )  # an unclosed namespace still yields what it holds
        with mock.patch.object(ov, "cpp_defs", side_effect=RecursionError("deep")):
            self.assertEqual(ov.file_defs("rk/x.cpp", "int f();\n"), [])

    def test_a_long_name_before_an_unreadable_tail_is_quick(self):
        # Every split of a long UPPER name, of `&& &&`, `[[a]] [[b]]` or `-> AB -> AB` was tried before the regex gave
        # up, and a tick hung on headers such as develop's rapidjson ones; each case is quick.
        tails = [
            " " + "CK_TILE_HOST_DEVICE_" * 3 + " + 1",
            " " + "&& " * 40 + "x",
            " " + "[[nodiscard]] " * 40 + "x",
            " -> AB" * 40 + " (x",
            " -> A B C" * 40 + " (x",
        ]
        start = time.monotonic()
        for tail in tails:
            self.assertIsNone(ov.C_AFTER_DECL.fullmatch(tail))
            self.assertEqual(ov.cpp_defs(f"int f(int v){tail};\n"), [])
        rapidjson = (
            "\nRAPIDJSON_DIAG_PUSH\nRAPIDJSON_DIAG_OFF(4512)\n" * 6 + "\nnamespace x"
        )
        self.assertIsNone(ov.C_AFTER_DECL.fullmatch(rapidjson))
        self.assertLess(time.monotonic() - start, 1.0)
        for good in (
            " const",
            " const noexcept override",
            " && = delete",
            " = 0",
            " [[nodiscard]] ROCKE_API",
            " -> std::pair<int, T*>",
            " -> int CK_TILE_DEPRECATED(x)",
            " ROCKE_EXPORT(1) final",
            " ->MACRO(x)",
            " -> T const override",
        ):
            self.assertIsNotNone(ov.C_AFTER_DECL.fullmatch(good), good)
        self.assertEqual(
            names(ov.cpp_defs("auto f(int v) -> int CK_TILE_HOST_DEVICE;\n")),
            [("cfunc", "f")],
        )

    def test_cmake_tests(self):
        defs = ov.cmake_defs(
            "add_test(NAME rocke_golden COMMAND x)\n# add_test(NAME old COMMAND y)\n"
            "add_test(rocke_old_style cmd)\nadd_test(NAME ${name} COMMAND z)\n"
        )
        self.assertEqual(
            names(defs), [("ctest", "rocke_golden"), ("ctest", "rocke_old_style")]
        )


class TestDiffRanges(unittest.TestCase):
    def test_added_ranges(self):
        diff = (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -3,0 +4,2 @@\n+def f():\n++++ odd content\n"
            "@@ -9 +11 @@\n-old\n\\ No newline at end of file\n+new\n"
            "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n"
            "diff --git a/n.cpp b/n.cpp\n--- /dev/null\n+++ b/n.cpp\n@@ -0,0 +1,3 @@\n+a\n+b\n+c\n"
        )
        self.assertEqual(
            ov.added_ranges(diff), {"x.py": [(4, 5), (11, 11)], "n.cpp": [(1, 3)]}
        )


def _defs(path, src, n_lines=None):
    out = ov.file_defs(path, src)
    for d in out:
        d["path"] = path
    return out


def _view(files: dict) -> dict:
    """develop_view over files {path: text}, as develop_update would store them (without git)."""
    cache = {"develop": {"sha": "d" * 40, "files": {}}, "blobs": {}, "patterns": {}}
    counts, sk = {}, {}
    for i, (path, text) in enumerate(sorted(files.items())):
        blob = f"{i:040x}"
        cache["develop"]["files"][path] = blob
        defs = ov.file_defs(path, text)
        cache["blobs"][blob] = [ov._compact(d) for d in defs]
        for d in defs:
            h = ov._compact(d)[3]
            if h:
                counts[h] = counts.get(h, 0) + 1
                sk[h] = d["sk"]
    cache["patterns"] = {
        h: sk[h] for h, c in counts.items() if c >= ov.MIN_DEVELOP_COPIES
    }
    view = ov.develop_view(cache)
    bodies = {}
    for path, text in files.items():
        for d in ov.file_defs(path, text):
            bodies[(path, d["line"])] = d["sk"]
    return view, (
        lambda members: {
            (p, ln): bodies[(p, ln)] for p, ln, _ in members if (p, ln) in bodies
        }
    )


SHA1 = (
    "std::string sha1_hex(const std::string& msg)\n{\n    unsigned char d[20];\n"
    "    sha1(msg.data(), msg.size(), d);\n"
    "    return to_hex(d, 20);\n}\n"
)
SHA1_OTHER = (
    "std::string sha1_hex(const std::string& s)\n{\n    Sha1 h;\n    h.update(s);\n"
    "    std::ostringstream o;\n"
    "    for (auto b : h.digest()) o << std::hex << int(b);\n    return o.str();\n}\n"
)
LOGIC = "\n".join(
    [
        "static int NAME(int a, int b)",
        "{",
        "    int t = a * 3;",
        "    t += b << 2;",
        "    if (t > 100)",
        "        t -= 7;",
        "    t ^= a + b;",
        "    return t % 97;",
        "}",
        "",
    ]
)


class TestGroups(unittest.TestCase):
    P = "rk/platform/cpp/instances/common/"

    def compute(self, prs, develop=None, related=None):
        view, sketches = _view(develop or {})
        return ov.compute(prs, view, related or set(), sketches)

    def test_the_same_new_macro_in_two_prs_is_a_group(self):
        """The same new macro and value in two PRs' headers (ROCKE_COMGR_FRESH_PROCESS_ATTR) is a group."""
        m = '#define ROCKE_COMGR_FRESH_PROCESS_ATTR "comgr_fresh_process"\n'
        res = self.compute(
            {
                1: _defs("rk/platform/cpp/include/rocke/helper_x.h", m),
                2: _defs("rk/platform/cpp/include/rocke/compile_options.h", m),
            }
        )
        (grp,) = res["groups"]
        self.assertEqual(
            (grp["key"], grp["rule"], [x["n"] for x in grp["members"]]),
            ("c.const/ROCKE_COMGR_FRESH_PROCESS_ATTR", "same", [1, 2]),
        )
        self.assertEqual(
            ov.layer_tag("rk/platform/cpp/include/rocke/helper_x.h"),
            " [generated header]",
        )
        self.assertEqual(
            ov.layer_tag("rk/platform/cpp/include/rocke/compile_options.h"),
            " [shared layer]",
        )
        # the same macro in two source files: each is its own file's, a coincidence of names
        res = self.compute(
            {1: _defs(self.P + "a.cpp", m), 2: _defs(self.P + "b.cpp", m)}
        )
        self.assertEqual(res["groups"], [])

    def test_a_name_develop_defines_is_not_new_work(self):
        f = "int rocke_count(int v)\n{\n    return v + 1;\n}\n"
        prs = {1: _defs(self.P + "a.cpp", f), 2: _defs(self.P + "b.cpp", f)}
        self.assertEqual(len(self.compute(prs)["groups"]), 1)
        self.assertEqual(self.compute(prs, {self.P + "old.cpp": f})["groups"], [])
        # a definition develop has in the same file is an edit, not an addition
        res = self.compute(
            {1: _defs(self.P + "a.cpp", f), 2: _defs(self.P + "a.cpp", f)},
            {self.P + "a.cpp": f},
        )
        self.assertEqual(res["groups"], [])

    def test_coincidences_are_dropped(self):
        """Coincidences, not overlaps: private test helpers with unrelated bodies (`_cfg`), and public constants,
        types and one-word functions with unrelated bodies in different modules (`ARMS`, `Plan`, `family`).
        """
        cfg_a = "def _cfg(**kw):\n    return dict(BASE, **kw)\n"
        cfg_b = "def _cfg():\n    s = Spec(n=4, m=8, k=16, tile=(64, 64))\n    s.check()\n    return s\n"
        res = self.compute(
            {
                1: _defs("rk/library/tests/test_a.py", cfg_a),
                2: _defs("rk/library/tests/test_b.py", cfg_b),
            }
        )
        self.assertEqual(res["groups"], [])
        arms_a = 'ARMS = {"fast": 1, "slow": 2}\n'
        arms_b = "ARMS = [Arm(i, scale=2 ** i) for i in range(8)]\n"
        fam_a = "def family(m, fam, x, preds):\n    return preds[fam](m, x) if fam in preds else None\n"
        fam_b = "def family(identity):\n    return identity['op'] + '_' + identity['dtype']\n"
        res = self.compute(
            {
                1: _defs("rk/library/x.py", arms_a + fam_a),
                2: _defs("rk/platform/python/rocke/y.py", arms_b + fam_b),
            }
        )
        self.assertEqual(res["groups"], [])

    def test_what_is_kept_without_similar_bodies(self):
        # two visible functions whose names say what they do (sha1_hex, 0.46 similar)
        res = self.compute(
            {1: _defs(self.P + "a.cpp", SHA1), 2: _defs(self.P + "b.cpp", SHA1_OTHER)}
        )
        self.assertEqual([x["key"] for x in res["groups"]], ["c.func/sha1_hex"])
        # both add it to the same file (h_asm_operands in lower_hip_mma.cpp)
        a = "static int pick(int v)\n{\n    return v * 2;\n}\n"
        b = "static int pick(int v)\n{\n    if (v < 0)\n        return 0;\n    return table[v];\n}\n"
        res = self.compute(
            {1: _defs(self.P + "a.cpp", a), 2: _defs(self.P + "a.cpp", b)}
        )
        self.assertEqual([x["key"] for x in res["groups"]], ["c.func/pick"])
        # one CTest name space
        res = self.compute(
            {
                1: _defs(
                    "rk/platform/CMakeLists.txt", "add_test(NAME rocke_x COMMAND a)\n"
                ),
                2: _defs(
                    "rk/library/CMakeLists.txt",
                    "add_test(NAME rocke_x COMMAND b c d)\n",
                ),
            }
        )
        self.assertEqual([x["key"] for x in res["groups"]], ["cmake.test/rocke_x"])

    def test_similar_private_helpers_are_kept(self):
        """The same i32 constant helper (c32) in one PR's parity emitter and another's test is kept."""
        a = (
            "static rocke_value_t* c32(rocke_ir_builder_t* b, int64_t v)\n{\n"
            "    return rocke_const_i32(b, (int32_t)v);\n}\n"
        )
        b = (
            "rocke_value_t* c32(rocke_ir_builder_t* bld, int64_t x)\n{\n"
            "    return rocke_const_i32(bld, (int32_t)x);\n}\n"
        )
        res = self.compute(
            {
                1: _defs("rk/library/tests/parity/a_emit.c", a),
                2: _defs("rk/platform/tests/t.cpp", b),
            }
        )
        (grp,) = res["groups"]
        self.assertGreaterEqual(grp["sims"]["1-2"], ov.SIM_SAME)

    def test_same_logic_under_other_names(self):
        a, b = LOGIC.replace("NAME", "mix_a"), LOGIC.replace("NAME", "blend_b")
        res = self.compute(
            {1: _defs(self.P + "a.cpp", a), 2: _defs(self.P + "b.cpp", b)}
        )
        (grp,) = res["groups"]
        self.assertEqual(
            (grp["rule"], grp["key"], sorted(x["n"] for x in grp["members"])),
            ("logic", "like/c.func/blend_b+mix_a", [1, 2]),
        )
        # copies inside one PR are the code quality pass's; test helpers alone are not shared work
        self.assertEqual(
            self.compute({1: _defs(self.P + "a.cpp", a) + _defs(self.P + "b.cpp", b)})[
                "groups"
            ],
            [],
        )
        res = self.compute(
            {
                1: _defs("rk/platform/tests/a_test.cpp", a),
                2: _defs("rk/platform/tests/b_test.cpp", b),
            }
        )
        self.assertEqual(res["groups"], [])
        # a short body is not enough to call it the same logic
        short = "static int {n}(int a)\n{{\n    return a * 3 + 1;\n}}\n"
        res = self.compute(
            {
                1: _defs(self.P + "a.cpp", short.format(n="one")),
                2: _defs(self.P + "b.cpp", short.format(n="two")),
            }
        )
        self.assertEqual(res["groups"], [])

    def test_a_copy_of_a_develop_pattern_is_a_lead_not_a_group(self):
        """With develop holding several `*_streq` (three variants) and `findTensor` copies, PRs that each add `streq`
        add one more copy of develop's pattern, not new cross-PR work."""
        dev = {
            self.P + "elementwise.cpp": STREQ,
            self.P
            + "tiled.cpp": STREQ.replace("rocke_ew_streq", "rocke_t2d_streq").replace(
                "return a != NULL && b != NULL && strcmp(a, b) == 0;",
                "if (a == NULL || b == NULL)\n        return a == b;\n    return strcmp(a, b) == 0;",
            ),
            self.P
            + "glue.cpp": STREQ.replace("rocke_ew_streq", "rocke_glue_streq").replace(
                "return a != NULL && b != NULL && strcmp(a, b) == 0;",
                "if (a == b)\n        return true;\n    return a && b && !strcmp(a, b);",
            ),
        }
        mine = STREQ.replace("static bool rocke_ew_streq", "bool streq")
        nullptr = mine.replace("NULL", "nullptr")
        res = self.compute(
            {
                1: _defs(self.P + "dense.cpp", mine),
                2: _defs(self.P + "bwd.cpp", mine),
                3: _defs(self.P + "gemm.cpp", nullptr),
            },
            dev,
        )
        self.assertEqual(res["groups"], [])
        self.assertEqual(
            [(x["n"], x["name"], x["by"], x["count"]) for x in res["leads"]],
            [(1, "streq", "name", 3), (2, "streq", "name", 3), (3, "streq", "name", 3)],
        )
        self.assertEqual(
            res["leads"][0]["copies"][0][2], "rocke_ew_streq"
        )  # the most similar first
        self.assertEqual(res["leads"][0]["also"], [2, 3])
        # a body develop has two exact copies of, under any name
        find = (
            "const Tensor* findTensor(const Ctx& context, int64_t uid)\n{\n"
            "    for (const auto* t : context.tensors)"
            "\n    {\n        if (t->uid() == uid)\n            return t;\n    }\n    return nullptr;\n}\n"
        )
        dev = {
            "src/packs/A.cpp": find,
            "src/packs/B.cpp": find.replace("findTensor", "lookupTensor"),
        }
        res = self.compute(
            {1: _defs("src/packs/C.cpp", find), 2: _defs("src/packs/D.cpp", find)}, dev
        )
        self.assertEqual(
            [(x["n"], x["by"], x["count"]) for x in res["leads"]],
            [(1, "body", 2), (2, "body", 2)],
        )
        self.assertEqual(res["groups"], [])

    def test_leads_other_prs_also_copy_come_first(self):
        """A lead other open PRs also copy (`findTensor`, in three PRs) comes before bigger single-PR leads
        (`*KpackModuleCache`, 13 copies) and is never cut: it needs coordinating first.
        """
        big, shared = LOGIC.replace("NAME", "big_one"), SHA1.replace(
            "sha1_hex", "digest_of"
        )
        dev = {
            f"{self.P}d{i}.cpp": big.replace("big_one", f"big_{i}") for i in range(4)
        }
        dev.update(
            {
                f"{self.P}s{i}.cpp": shared.replace("digest_of", f"digest_{i}")
                for i in range(2)
            }
        )
        res = self.compute(
            {
                1: _defs(self.P + "p1.cpp", big + shared),
                2: _defs(self.P + "p2.cpp", shared),
            },
            dev,
        )
        self.assertEqual(
            [(x["n"], x["name"], x["count"], x["also"]) for x in res["leads"]],
            [(1, "digest_of", 2, [2]), (1, "big_one", 4, []), (2, "digest_of", 2, [1])],
        )

    def test_a_copy_in_the_prs_own_file_is_not_a_pattern(self):
        """A rename in place: develop's copies in the PR's own file are its old self."""
        f = LOGIC.replace("NAME", "old_name")
        dev = {self.P + "a.cpp": f + f.replace("old_name", "old_twin")}
        res = self.compute(
            {1: _defs(self.P + "a.cpp", f.replace("old_name", "new_name"))}, dev
        )
        self.assertEqual(res["leads"], [])

    def test_a_stack_shares_its_base(self):
        f = SHA1
        prs = {1: _defs(self.P + "a.cpp", f), 2: _defs(self.P + "a.cpp", f)}
        self.assertEqual(len(self.compute(prs)["groups"]), 1)
        self.assertEqual(self.compute(prs, related={frozenset((1, 2))})["groups"], [])
        st = empty_state()
        st["prs"] = {
            "1": {
                "n": 1,
                "status": "open",
                "draft": False,
                "head": "a" * 40,
                "base": "develop",
                "head_ref": "u/one",
            },
            "2": {
                "n": 2,
                "status": "open",
                "draft": False,
                "head": "b" * 40,
                "base": "u/one",
                "head_ref": "u/two",
            },
            "3": {
                "n": 3,
                "status": "open",
                "draft": False,
                "head": "c" * 40,
                "base": "u/two",
                "head_ref": "u/three",
            },
        }
        self.assertEqual(
            ov.related_pairs(st),
            {frozenset((1, 2)), frozenset((2, 3)), frozenset((1, 3))},
        )
        self.assertEqual(
            ov.head_keys(st)["2"], ("b" * 12 + "^" + "a" * 12, "a" * 40)
        )  # read against its base PR
        self.assertEqual(ov.head_keys(st)["1"], ("a" * 12, ov.DEVELOP))


class TestLandOrder(unittest.TestCase):
    def st(self):
        st = empty_state()
        st["prs"] = {str(n): {"n": n, "status": "open"} for n in (10, 20, 30, 40)}
        st["queue"] = [
            {"n": 10, "stage": None, "blockers": ["G1: x", "G3: y"]},
            {"n": 20, "stage": "candidate", "blockers": []},
            {"n": 30, "stage": None, "blockers": ["G1: x"]},
            {"n": 40, "stage": "ready", "blockers": []},
        ]
        return st

    def test_without_a_plan(self):
        st = self.st()
        self.assertEqual(
            [n for n, _ in ov.land_order(st, [10, 20, 30, 40])], [40, 20, 30, 10]
        )
        st["develop_fixes"] = {"10": ["your priority: urgent"]}
        order = ov.land_order(st, [10, 20, 30, 40])
        self.assertEqual(order[0], (10, "your priority; not ready: 2 failing gates"))
        self.assertEqual(order[1], (40, "ready"))

    def test_the_merge_plan_comes_first(self):
        st = self.st()
        st["merge_plan"] = {
            "nodes": {"30": {"rank": 0, "wave": 1}, "20": {"rank": 1, "wave": 2}},
            "why": {},
            "dev_conflicts": {},
            "waves": [[30], [20]],
        }
        order = ov.land_order(st, [10, 20, 30, 40])
        self.assertEqual([n for n, _ in order], [30, 20, 40, 10])
        self.assertEqual(order[0], (30, "merge plan: wave 1"))
        st["merge_plan"] = {"unavailable": "no near-merge PR"}
        self.assertEqual(
            [n for n, _ in ov.land_order(st, [10, 20, 30, 40])], [40, 20, 30, 10]
        )


class Repo:
    """A tiny repository: develop at refs/remotes/origin/develop, PR heads as commits."""

    def __init__(self, d):
        self.path = pathlib.Path(d) / "repo"
        self.path.mkdir()
        g(self.path, "init", "-q", "-b", "develop")
        self.commit(
            {"rk/platform/cpp/instances/common/base.cpp": STREQ, "rk/README.md": "x\n"},
            "base",
        )
        g(self.path, "update-ref", "refs/remotes/origin/develop", "HEAD")

    def commit(self, files, msg):
        for f, text in files.items():
            p = self.path / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        g(self.path, "add", "-A")
        g(self.path, "commit", "-qm", msg)
        return g(self.path, "rev-parse", "HEAD")

    def branch(self, name, start="refs/remotes/origin/develop"):
        g(self.path, "checkout", "-q", "-B", name, start)


class TestRefresh(unittest.TestCase):
    P = "rk/platform/cpp/include/rocke/"

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.repo = Repo(self.d.name)
        m = '#define ROCKE_SHARED_ATTR "shared"\n'
        self.repo.branch("p1")
        self.h1 = self.repo.commit(
            {self.P + "one.h": m + "int one_only(int v);\n"}, "pr 1"
        )
        self.repo.branch("p2")
        self.h2 = self.repo.commit(
            {
                self.P + "two.h": m,
                "rk/platform/cpp/instances/common/base.cpp": STREQ + "\n",
            },
            "pr 2",
        )
        self.repo.branch("develop", "refs/remotes/origin/develop")
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.cfg = {
            "rocke_path": K,
            "paths": {"repo": str(self.repo.path)},
            "operator": {"github_login": "me"},
        }
        self.st = empty_state()
        self.st["develop"]["sha"] = g(
            self.repo.path, "rev-parse", "refs/remotes/origin/develop"
        )
        rec = {"status": "open", "draft": False, "base": "develop", "author": "a"}
        self.st["prs"] = {
            "1": dict(rec, n=1, head=self.h1, head_ref="p1"),
            "2": dict(rec, n=2, head=self.h2, head_ref="p2"),
        }

    def test_heads_are_read_once_and_the_groups_stored(self):
        out = ov.refresh(self.cfg, self.st, self.store)
        self.assertEqual((out["heads"], out["errors"]), (2, []))
        self.assertEqual(
            [x["key"] for x in self.st["overlaps"]["groups"]],
            ["c.const/ROCKE_SHARED_ATTR"],
        )
        self.assertTrue((self.store.dir / ov.CACHE_FILE).exists())
        with mock.patch.object(
            ov, "head_defs", side_effect=AssertionError("read again")
        ):
            self.assertEqual(
                ov.refresh(self.cfg, self.st, self.store)["heads"], 0
            )  # nothing moved: nothing read
        # a push: only the new head is read; the old one leaves the cache
        self.repo.branch("p1", self.h1)
        h1b = self.repo.commit(
            {self.P + "one.h": '#define ROCKE_SHARED_ATTR "shared"\n'}, "pr 1 again"
        )
        self.st["prs"]["1"]["head"] = h1b
        real = ov.head_defs
        with mock.patch.object(ov, "head_defs", side_effect=real) as spy:
            out = ov.refresh(self.cfg, self.st, self.store)
        self.assertEqual(
            (out["heads"], [c.args[1] for c in spy.call_args_list]), (1, [h1b])
        )
        cache = ov.load_cache(self.store)
        self.assertEqual(
            sorted(v["head"] for v in cache["heads"].values()), sorted([h1b, self.h2])
        )
        self.assertEqual(
            self.st["overlaps"]["scanned"], {"1": h1b[:12], "2": self.h2[:12]}
        )

    def test_an_edit_of_a_develop_definition_is_not_added_work(self):
        ov.refresh(self.cfg, self.st, self.store)
        cache = ov.load_cache(self.store)
        two = next(v for v in cache["heads"].values() if v["n"] == 2)
        self.assertIn("rk/platform/cpp/instances/common/base.cpp", two["paths"])
        self.assertNotIn(
            "rocke_ew_streq", [d["name"] for d in two["defs"]]
        )  # the diff added only a blank line

    def test_develop_moves_rereads_only_its_changed_files(self):
        ov.refresh(self.cfg, self.st, self.store)
        before = ov.load_cache(self.store)["develop"]["files"]
        self.repo.commit(
            {self.P + "dev.h": '#define ROCKE_SHARED_ATTR "shared"\n'},
            "develop adds it",
        )
        dev = g(self.repo.path, "rev-parse", "HEAD")
        g(self.repo.path, "update-ref", "refs/remotes/origin/develop", dev)
        self.st["develop"]["sha"] = dev
        out = ov.refresh(self.cfg, self.st, self.store)
        self.assertEqual(out["develop_read"], 1)
        self.assertEqual(
            len(ov.load_cache(self.store)["develop"]["files"]), len(before) + 1
        )
        self.assertEqual(
            self.st["overlaps"]["groups"], []
        )  # develop has the name now: not new work

    def test_an_unreadable_head_waits_and_is_retried_later(self):
        self.st["prs"]["3"] = {
            "n": 3,
            "status": "open",
            "draft": False,
            "base": "develop",
            "head": "f" * 40,
            "head_ref": "p3",
        }
        out = ov.refresh(self.cfg, self.st, self.store)
        self.assertEqual(
            self.st["overlaps"]["pending"], {"3": "its head is not fetched yet"}
        )
        self.assertEqual(out["heads"], 2)
        with mock.patch.object(
            ov.gitops, "has_commit", side_effect=AssertionError("tried again")
        ):
            ov.refresh(
                self.cfg, self.st, self.store
            )  # within RETRY_MIN: not tried again
        self.st["overlaps"]["waiting"]["3"]["at"] = "2000-01-01T00:00:00Z"
        with mock.patch.object(ov.gitops, "has_commit", return_value=False) as spy:
            ov.refresh(self.cfg, self.st, self.store)
        self.assertTrue(spy.called)

    def test_blobs_that_are_not_local_wait_for_the_fetch_budget(self):
        calls = []

        def fake(repo, head, base, fetch=False):
            calls.append(fetch)
            if not fetch:
                raise ov.NeedsBlobs(head)
            return {"mb": "x", "paths": [], "defs": []}

        with mock.patch.object(ov, "head_defs", side_effect=fake):
            out = ov.refresh(
                dict(self.cfg, overlaps={"max_fetches": 1}), self.st, self.store
            )
        self.assertEqual(
            (out["fetched"], out["deferred"], calls), (1, 1, [False, True, False])
        )
        self.assertEqual(
            list(self.st["overlaps"]["pending"].values()),
            ["waits for the next tick (blob fetch budget spent)"],
        )
        calls.clear()
        st = empty_state()
        st.update(develop=self.st["develop"], prs=self.st["prs"])
        with mock.patch.object(ov, "head_defs", side_effect=fake):
            ov.refresh(
                dict(self.cfg, overlaps={"max_fetches": 0}),
                st,
                Store(pathlib.Path(self.d.name) / "s2"),
            )
        self.assertEqual(
            set(st["overlaps"]["pending"].values()),
            {"its changed files are not local yet"},
        )


class TestDecisionsAndSection(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.store = Store(pathlib.Path(self.d.name))
        self.cfg = {
            "rocke_path": K,
            "paths": {"repo": self.d.name},
            "operator": {"github_login": "me"},
            "features": {"overlap-management": True},
        }
        self.st = empty_state()
        rec = {"status": "open", "draft": False, "base": "develop", "head_ref": "x"}
        self.st["prs"] = {
            "1": dict(rec, n=1, head="1" * 40, author="ann"),
            "2": dict(rec, n=2, head="2" * 40, author="bob"),
        }
        self.st["queue"] = [
            {"n": 1, "stage": "ready", "blockers": []},
            {"n": 2, "stage": None, "blockers": ["G1"]},
        ]
        m = '#define ROCKE_SHARED_ATTR "shared"\n'
        res = ov.compute(
            {
                1: _defs("rk/platform/cpp/include/rocke/a.h", m),
                2: _defs("rk/platform/cpp/include/rocke/b.h", m),
            },
            _view({})[0],
        )
        self.st["overlaps"] = {
            "at": "2026-11-03T20:00:00Z",
            "develop": "d" * 12,
            "indexed": "d" * 12,
            "keys": {"1": "1" * 12, "2": "2" * 12},
            "scanned": {"1": "1" * 12, "2": "2" * 12},
            "pending": {},
            "groups": res["groups"],
            "leads": [],
            "lead_total": 0,
        }

    def test_the_section_names_the_other_pr_and_the_proposed_owner(self):
        text = "\n".join(ov.section(self.cfg, self.st, self.store, 2))
        self.assertIn(
            "1. macro `ROCKE_SHARED_ATTR`: the same new name in 2 open PRs [key c.const/ROCKE_SHARED_ATTR]",
            text,
        )
        self.assertIn(
            "   here: rk/platform/cpp/include/rocke/b.h:1 [shared layer]  #define ROCKE_SHARED_ATTR",
            text,
        )
        self.assertIn(
            "   #1 (ann): rk/platform/cpp/include/rocke/a.h:1 [shared layer], similarity 1.00",
            text,
        )
        self.assertIn(
            "land order: #1 (ready), #2 (not ready: 1 failing gate) → proposed owner #1",
            text,
        )
        self.assertNotIn("decided:", text)
        self.assertEqual(ov.section(self.cfg, self.st, self.store, 3)[0][:3], "#3 ")

    def test_a_recorded_decision_shows_in_every_member_and_after_the_owner_lands(self):
        with self.assertRaises(ValueError):
            ov.decide(
                self.st, self.store, "c.const/ROCKE_SHARED_ATTR", 7, None, "", "me"
            )  # not a member
        with self.assertRaises(ValueError):
            ov.decide(
                self.st, self.store, "no/such", 1, None, "", "me"
            )  # no group and no members given
        self.assertFalse((self.store.dir / ov.DECIDED_FILE).exists())
        rec = ov.decide(
            self.st,
            self.store,
            "c.const/ROCKE_SHARED_ATTR",
            2,
            None,
            "b.h is the public home",
            "me",
        )
        self.assertEqual((rec["owner"], rec["members"]), (2, [1, 2]))
        self.assertIn("overlap", self.store.journal_tail(1)[0])
        for n in (1, 2):
            text = "\n".join(ov.section(self.cfg, self.st, self.store, n))
            self.assertIn("   decided: #2 owns it (", text)
            self.assertIn(
                "land order: #1 (ready), #2 (not ready: 1 failing gate) → decided: #2 owns it (below)",
                text,
            )
            self.assertNotIn(
                "proposed owner", text
            )  # the decision, not the land order, names the owner
        # the owner lands: develop now defines the name and the group is gone; #1 still adds its copy
        self.st["prs"]["2"]["status"] = "merged"
        self.st["overlaps"]["groups"] = []
        mine = _defs(
            "rk/platform/cpp/include/rocke/a.h", '#define ROCKE_SHARED_ATTR "shared"\n'
        )
        cache = {
            "v": ov.CACHE_VERSION,
            "heads": {"1" * 12: {"n": 1, "head": "1" * 40, "paths": [], "defs": mine}},
            "develop": {},
            "blobs": {},
            "patterns": {},
            "named": {},
        }
        ov.save_cache(self.store, cache)
        text = "\n".join(ov.section(self.cfg, self.st, self.store, 1))
        self.assertIn(
            "Decisions recorded earlier for shared code this PR also adds:", text
        )
        self.assertIn("[key c.const/ROCKE_SHARED_ATTR] decided: #2 owns it", text)
        self.assertIn("it has landed: reuse develop's copy", text)
        # once #1 drops its copy, nothing is left to remind it of
        cache["heads"]["1" * 12]["defs"] = []
        ov.save_cache(self.store, cache)
        self.assertEqual(ov.section(self.cfg, self.st, self.store, 1)[-1], "none found")

    def test_none_found_and_not_computed(self):
        self.st["overlaps"]["groups"] = []
        self.assertEqual(ov.section(self.cfg, self.st, self.store, 1)[-1], "none found")
        self.st.pop("overlaps")
        self.assertIn(
            "not computed yet", ov.section(self.cfg, self.st, self.store, 1)[0]
        )

    def test_the_cli(self):
        import rkmon

        out = []
        self.store.save(self.st)
        with mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["overlaps", "--section"]), 4)
            self.assertEqual(rkmon.main(["overlaps", "2", "--section"]), 0)
            self.assertEqual(rkmon.main(["overlaps"]), 0)
            self.assertEqual(
                rkmon.main(["overlap-decide", "c.const/ROCKE_SHARED_ATTR", "9"]), 4
            )
            self.assertEqual(
                rkmon.main(
                    [
                        "overlap-decide",
                        "c.const/ROCKE_SHARED_ATTR",
                        "1",
                        "--members",
                        "1,#2",
                        "--note",
                        "first to land",
                    ]
                ),
                0,
            )
        text = "\n".join(out)
        self.assertIn("--section needs a PR number", text)
        self.assertIn(
            "REFUSED: #9 is not a member of c.const/ROCKE_SHARED_ATTR (#1, #2)", text
        )
        self.assertIn(
            "decided: c.const/ROCKE_SHARED_ATTR: #1 owns it (members #1, #2)", text
        )
        self.assertIn("Overlapping work across the open PRs", text)
        self.assertEqual(
            ov.decisions(self.store)["c.const/ROCKE_SHARED_ATTR"]["by"], "me"
        )


class TestChecksWiring(unittest.TestCase):
    PLAN = {
        "families": [],
        "tests": [],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_every_review_shows_the_section_and_a_gate_does_not(self):
        cfg = {
            "rocke_path": K,
            "repo": "ROCm/r",
            "paths": {"working_dir": "/nonexistent"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f"sec '{ov.SECTION}'", 1)[1].split("sec '", 1)[0]
        self.assertIn("$M/rk overlaps $N --section", sec)
        self.assertLess(
            script.index("sec 'Threads and reviews since our last review'"),
            script.index(ov.SECTION),
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        self.assertNotIn(
            ov.SECTION,
            review.checks_script(
                cfg, 7, "a" * 40, "first-round", self.PLAN, None, mode="gate"
            ),
        )


class TestNoticesDue(unittest.TestCase):
    def test_each_open_non_owner_member_needs_one_overlap_notice(self):
        """The PRs that drop their copy get an overlap notice naming the owner, the order and the rebase."""
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {str(n): {"n": n, "status": "open"} for n in (1, 2, 3, 4)}
            st["prs"]["4"]["status"] = "merged"
            ov.decide(st, store, "findTensor", 1, [1, 2, 3, 4], "rule 1", "me")
            ov.decide(st, store, "sdpa-helpers", 1, [1, 2], "rule 1", "me")
            self.assertEqual(
                ov.notices_due(st, store),
                [
                    "OVERLAP_NOTICE_DUE #2: findTensor (owner #1); sdpa-helpers (owner #1): post the notice "
                    "(`rk notice 2 FILE --kind overlap --key K[,K]`)",
                    "OVERLAP_NOTICE_DUE #3: findTensor (owner #1): post the notice (`rk notice 3 FILE --kind overlap --key "
                    "K[,K]`)",
                ],
            )  # the owner and the merged member need none
            store.ledger_add(
                "notice-overlap", 2, "findTensor+sdpa-helpers", "u"
            )  # "+" joins both groups
            store.ledger_add("notice-overlap", 3, "findTensor", "u")
            self.assertEqual(ov.notices_due(st, store), [])

    def test_a_group_key_with_plus_counts_whole_and_commas_join_groups(self):
        """A "like/" group key names its members' types with "+", so notices join groups with ","."""
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {str(n): {"n": n, "status": "open"} for n in (1, 2, 3)}
            like = "like/c.type/AProblem+BProblem"
            ov.decide(st, store, like, 1, [1, 2, 3], "one home", "me")
            ov.decide(st, store, "reduce-module", 1, [1, 3], "one home", "me")
            store.ledger_add("notice-overlap", 2, like, "u")  # one group, its key whole
            store.ledger_add(
                "notice-overlap", 3, f"{like},reduce-module", "u"
            )  # two groups, joined by ","
            self.assertEqual(ov.notices_due(st, store), [])

    def test_a_decision_filed_again_under_another_key_counts_its_notices(self):
        """A decision that names a constant too (require_tools_or_skip) becomes a group of its own once another PR
        adopts it. Filed again with --same-as, the earlier notice covers it: no second notice.
        """
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {str(n): {"n": n, "status": "open"} for n in (1, 2, 3)}
            ov.decide(st, store, "py.func/helper", 1, [1, 2, 3], "first", "me")
            store.ledger_add("notice-overlap", 2, "py.func/helper", "u")
            with self.assertRaisesRegex(
                ValueError, "no decision recorded under that key"
            ):
                ov.decide(
                    st,
                    store,
                    "py.const/HELPER_ENV",
                    1,
                    [1, 2],
                    "",
                    "me",
                    same_as="py.func/other",
                )
            with self.assertRaisesRegex(
                ValueError, "that decision's owner is #1, not #2"
            ):
                ov.decide(
                    st,
                    store,
                    "py.const/HELPER_ENV",
                    2,
                    [1, 2],
                    "",
                    "me",
                    same_as="py.func/helper",
                )
            ov.decide(
                st,
                store,
                "py.const/HELPER_ENV",
                1,
                [1, 2, 3],
                "",
                "me",
                same_as="py.func/helper",
            )
            self.assertEqual(
                ov.decisions(store)["py.const/HELPER_ENV"]["same_as"], "py.func/helper"
            )
            # #2 got the notice for the repeated decision; #3 never got one, for either key
            self.assertEqual(
                ov.notices_due(st, store),
                [
                    "OVERLAP_NOTICE_DUE #3: py.const/HELPER_ENV (owner #1); py.func/helper (owner #1): post the notice "
                    "(`rk notice 3 FILE --kind overlap --key K[,K]`)"
                ],
            )


if __name__ == "__main__":
    unittest.main()
