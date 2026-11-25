# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Clashes with other open PRs: the definitions, uses and references each PR's
lines carry, the report, the checks wiring, and both land orders on a small git repo where two branches clash silently
(an enum value renamed, a function removed, a header moved, a selector changed under the other's new test); the pair
builds' flags, read as data from the shared build while its slot is kept for it. Temp git repos, no network.
"""

import inspect
import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import clashes, registry, review, slots  # noqa: E402
from rkmon_lib import container_layout as L  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write(repo: pathlib.Path, files: dict) -> None:
    for path, text in files.items():
        p = repo / path
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))


def names(defs) -> set:
    return {(kind, name) for kind, name, _, _ in defs}


class TestLimits(unittest.TestCase):
    def test_the_limits_are_the_package_s_entry_by_entry(self):
        """The clash scan's limits are config.defaults.json's (config clashes): an entry a configuration gives
        replaces that entry alone, and a scan given none takes the package's."""
        package = registry.default("clashes")
        self.assertEqual(sorted(package), ["budget_s", "max_fetch", "merge_timeout_s"])
        self.assertEqual(clashes.settings({}), package)
        self.assertEqual(
            clashes.settings({"clashes": {"max_fetch": 9}}), dict(package, max_fetch=9)
        )
        given = inspect.signature(clashes.scan).parameters
        for name in package:
            self.assertIsNone(given[name].default, name)


class TestDefinitions(unittest.TestCase):
    def test_python_defs_fields_and_globals_but_not_nested_functions(self):
        src = """\
            X = 1
            a, b = 2, 3
            def f():
                def inner():
                    pass
            class C:
                field: int = 0
                other = 1
                def m(self):
                    pass
            if True:
                def g():
                    pass
            try:
                import foo
            except ImportError:
                def h():
                    pass
            """
        got = names(clashes.defs("x.py", textwrap.dedent(src)))
        self.assertEqual(
            got,
            {
                ("global", "X"),
                ("global", "a"),
                ("global", "b"),
                ("def", "f"),
                ("class", "C"),
                ("field", "field"),
                ("field", "other"),
                ("method", "m"),
                ("def", "g"),
                ("def", "h"),
            },
        )
        self.assertEqual(
            clashes.defs("x.py", "def broken(:\n")[:0], []
        )  # unparsable: the regex fallback, no raise

    def test_c_enum_values_fields_and_overlaps_kinds(self):
        src = """\
            #pragma once
            #define ROCKE_LIMIT 4
            typedef enum {
                ROCKE_A = 0, /* a */
                ROCKE_B,
                ROCKE_C [[deprecated]] = 3
            } r_t;
            enum class Mode : int { Fast, Slow };
            struct S {
                int x, *y, z[4];
                unsigned flag : 1;
                int (*fn)(int);
                std::map<int, int> table;
                void method() { int local = 0; }
                static_assert(sizeof(int) == 4, "");
            };
            int keep_fn(int a);
            """
        got = names(clashes.defs("rk/inc/api.h", textwrap.dedent(src)))
        for want in (
            [
                ("enumerator", n)
                for n in ("ROCKE_A", "ROCKE_B", "ROCKE_C", "Fast", "Slow")
            ]
            + [("field", n) for n in ("x", "y", "z", "flag", "fn", "table")]
            + [("macro", "ROCKE_LIMIT"), ("cfunc", "keep_fn"), ("type", "S")]
        ):
            self.assertIn(want, got)
        self.assertFalse(
            {n for _, n in got} & {"local", "int", "sizeof", "method_local"}
        )
        enum_lines = {
            n: line
            for _, n, line, _ in clashes.defs("rk/inc/api.h", textwrap.dedent(src))
        }
        self.assertEqual((enum_lines["ROCKE_A"], enum_lines["ROCKE_C"]), (4, 6))
        # A source file's enum values and fields are file-local: only its own uses count.
        self.assertTrue(
            all(
                local
                for k, _, _, local in clashes.defs("rk/x.cpp", "enum { V1, V2 };\n")
                if k == "enumerator"
            )
        )

    def test_py_logic_ignores_docstrings_and_comments(self):
        a = 'def sel(p):\n    """Doc."""\n    return [x for x in p if \'D\' not in x]\n'
        b = 'def sel(p):\n    """Other doc."""\n    # a comment\n    return [x for x in p if \'D\' not in x]\n'
        c = 'def sel(p):\n    """Doc."""\n    return [x for x in p if x != \'T\']\n'
        self.assertEqual(clashes.py_logic(a)["sel"][2], clashes.py_logic(b)["sel"][2])
        self.assertNotEqual(
            clashes.py_logic(a)["sel"][2], clashes.py_logic(c)["sel"][2]
        )

    def test_bindings_count_as_names_but_a_plain_from_import_does_not(self):
        src = (
            "from m import a\nfrom m import b as bb\nimport q as qq\ndef f(p):\n    loc = 1\n    for i in p:\n"
            "        pass\n    try:\n        pass\n    except E as err:\n        pass\n"
        )
        self.assertEqual(
            clashes.py_bindings(src), {"bb", "qq", "f", "p", "loc", "i", "err"}
        )
        self.assertEqual(
            clashes.module_words(
                "import a.b.c\nfrom d.e import f\nP = 'x/hipdnn_assets.py'\n"
            ),
            ["a", "b", "c", "d", "e", "f", "hipdnn_assets"],
        )
        self.assertEqual(
            (
                clashes.module_stem("x/y/sel.py"),
                clashes.module_stem("x/pkg/__init__.py"),
            ),
            ("sel", "pkg"),
        )


class TestReferences(unittest.TestCase):
    def test_includes_imports_and_path_strings_on_added_lines(self):
        c = '#include "a/b.h"\n#include <c.h>\n// see old/thing.h\nconst char* p = "data/x.json";\n'
        got = {
            (r["kind"], r["ref"]): r for r in clashes.refs("rk/src/u.cpp", c, [(1, 4)])
        }
        self.assertEqual(got[("include", "a/b.h")]["tails"], ["a/b.h"])
        self.assertEqual(got[("include", "a/b.h")]["exact"], ["rk/src/a/b.h"])
        self.assertEqual(
            got[("include", "c.h")]["tails"], []
        )  # one part: only next to the file
        self.assertIn(("path", "data/x.json"), got)
        self.assertNotIn(("path", "old/thing.h"), got)  # a comment names no file
        py = "import x.y.z\nfrom . import mod\nfrom ..pkg.sub import name\nP = 'cfg/a.yaml'  # or b/c.yaml\n"
        got = {
            (r["kind"], r["ref"]): r for r in clashes.refs("rk/p/q/f.py", py, [(1, 4)])
        }
        self.assertEqual(
            got[("import", "x.y.z")]["tails"], ["x/y/z.py", "x/y/z/__init__.py"]
        )
        self.assertEqual(
            got[("import", ".mod")]["exact"],
            ["rk/p/q/mod.py", "rk/p/q/mod/__init__.py"],
        )
        self.assertIn(
            "rk/p/pkg/sub/name.py", got[("import", "..pkg.sub.name")]["exact"]
        )
        self.assertIn(("path", "cfg/a.yaml"), got)
        self.assertNotIn(("path", "b/c.yaml"), got)
        cm = "add_library(x src/a.cpp)  # old/b.cpp\n"
        self.assertEqual(
            [r["ref"] for r in clashes.refs("rk/CMakeLists.txt", cm, [(1, 1)])],
            ["src/a.cpp"],
        )
        self.assertEqual(
            clashes.refs("rk/src/u.cpp", c, [(3, 3)]), []
        )  # only added lines

    def test_hunks_split_removed_and_added_lines(self):
        diff = textwrap.dedent(
            """\
            diff --git a/f.py b/f.py
            --- a/f.py
            +++ b/f.py
            @@ -3,0 +4,2 @@ def x():
            +new1
            +new2
            @@ -10,2 +12 @@
            -old1
            -old2
            +changed
            \\ No newline at end of file
            diff --git a/gone.py b/gone.py
            deleted file mode 100644
            --- a/gone.py
            +++ /dev/null
            @@ -1,2 +0,0 @@
            -a
            -b
            """
        )
        h = clashes.hunks(diff)
        self.assertEqual(
            (h["f.py"]["plus"], h["f.py"]["minus"]), ([(4, 5), (12, 12)], [(10, 11)])
        )
        self.assertEqual((h["gone.py"]["new"], h["gone.py"]["minus"]), (None, [(1, 2)]))


def pair_result(**kw) -> dict:
    empty = {"names": [], "paths": [], "leads": []}
    p = {
        "n": 22,
        "head": "b" * 40,
        "title": "the other",
        "author": "bob",
        "draft": False,
        "merged": True,
        "fwd": dict(empty),
        "rev": dict(empty),
        "orders": {},
    }
    p.update(kw)
    return p


class TestReport(unittest.TestCase):
    def test_render_puts_silent_clashes_first_and_names_the_land_order(self):
        silent = pair_result(
            fwd={
                "names": [
                    {
                        "name": "ROLE_A",
                        "kind": "enumerator",
                        "path": "rk/inc/api.h",
                        "line": 3,
                        "uses": [["rk/src/use.cpp", 7, False], ["rk/x.cpp", 9, True]],
                    }
                ],
                "paths": [],
                "leads": [],
            },
            orders={
                "nm": {
                    "conflicts": ["rk/shared.txt"],
                    "moddel": [],
                    "special": [],
                    "first_stands_in": False,
                    "dev_conflicts": [],
                },
                "mn": {
                    "conflicts": ["rk/shared.txt"],
                    "moddel": [
                        {
                            "path": "rk/gone.py",
                            "deleted_by": 22,
                            "type": "CONFLICT (modify/delete)",
                        }
                    ],
                    "special": [],
                    "first_stands_in": True,
                    "dev_conflicts": ["rk/d.txt"],
                },
            },
        )
        lead = pair_result(
            n=33,
            merged=False,
            rev={
                "names": [],
                "paths": [],
                "leads": [
                    {
                        "name": "select",
                        "path": "rk/sel.py",
                        "line": 1,
                        "uses": [["rk/tests/test_use.py", 5]],
                    }
                ],
            },
        )
        res = {
            "n": 11,
            "head": "a" * 40,
            "develop": "d" * 40,
            "mb": "e" * 40,
            "secs": 3.0,
            "fetched": 0,
            "pairs": [lead, silent, pair_result(n=44, merged=False)],
            "not_run": [(55, "its head is not in the " "mirror")],
            "skipped_stack": [66],
            "dev_conflicts": 0,
        }
        lines = clashes.render(
            res,
            "rk",
            {22: "expected: this PR first (merge plan: wave 1), so #22 does it"},
        )
        text = "\n".join(lines)
        self.assertTrue(
            lines[0].startswith(
                "#11 at aaaaaaaaaa against 3 open PRs on develop dddddddddd"
            )
        )
        self.assertLess(text.index("#22 "), text.index("#33 "))
        self.assertIn(
            "silent: `ROLE_A` (enumerator): this PR removes it (rk/inc/api.h:3); #22 uses it at "
            "rk/src/use.cpp:7 (merges clean); also in conflicting files: rk/x.cpp:9",
            text,
        )
        self.assertIn(
            "modify/delete: rk/gone.py: #22 deletes or moves it, this PR changes it",
            text,
        )
        self.assertIn("conflicts, either order (1): rk/shared.txt", text)
        self.assertIn(
            "(#22 does not merge with develop: its head stands in for its landing)",
            text,
        )
        self.assertIn(
            "lead: `select` (rk/sel.py:1): #33 changes its logic; this PR's new test lines call it at "
            "rk/tests/test_use.py:5 (merges clean)",
            text,
        )
        self.assertIn(
            "land order: whichever lands second ports the uses of `ROLE_A`, carries 1 modify/delete file",
            text,
        )
        self.assertIn(
            "; expected: this PR first (merge plan: wave 1), so #22 does it", text
        )
        self.assertIn("NOT RUN (1): #55 (its head is not in the mirror)", text)
        self.assertIn("stacked with this PR (not compared): #66", text)
        self.assertIn(
            "clean (1: no shared file, or both orders merge clean; no silent clash): #44",
            text,
        )
        self.assertEqual(
            lines[-1],
            "CLASH fail 1 PR clash: #22 (1 silent, 1 modify/delete, 1 conflicts); 1 NOT RUN",
        )

    def test_a_scan_that_ran_nothing_never_reads_clean(self):
        self.assertEqual(
            clashes.verdict({"n": 1, "error": "its head is not in the mirror"})[0],
            "skip",
        )
        self.assertEqual(
            clashes.verdict({"n": 1, "pairs": [], "not_run": [(2, "x")]}),
            ("skip", "no other open PR could be compared (1 NOT RUN)"),
        )
        self.assertEqual(
            clashes.verdict(
                {"n": 1, "pairs": [pair_result(merged=False)], "not_run": []}
            ),
            ("pass", "no clash with 1 open PRs"),
        )
        unmerged = pair_result(
            orders={"nm": None, "mn": None},
            why_unmerged=["#22: its blobs are not local"],
        )
        text = "\n".join(
            clashes.render(
                {
                    "n": 1,
                    "head": "a" * 40,
                    "develop": "d" * 40,
                    "secs": 1,
                    "pairs": [unmerged],
                    "not_run": [],
                    "skipped_stack": [],
                },
                "rk",
            )
        )
        self.assertIn(
            "NOT RUN: both land orders could not be merged (#22: its blobs are not local)",
            text,
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

    def test_every_review_runs_the_scan_and_a_gate_does_not(self):
        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f"sec '{clashes.SECTION}'", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            f"nice -n 10 timeout {review.CLASH_TIMEOUT_S} python -I -B $M/tools/clash_scan.py $N --head "
            f"{'a' * 40} --cache $CA/clash --json \"$OUT/clash.json\"",
            sec,
        )
        self.assertIn("st clash ", sec)
        self.assertIn(
            "CLASH skip no result", sec
        )  # a scan that wrote nothing never reads clean
        self.assertLess(
            script.index("$M/rk overlaps $N --section"), script.index(clashes.SECTION)
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            cfg, 7, "a" * 40, "first-round", self.PLAN, None, mode="gate"
        )
        self.assertNotIn(clashes.SECTION, gate)


class TestTwoBranches(unittest.TestCase):
    """develop, then N and M from it: N renames an enum value, removes a function, moves a header and changes a
    selector; M's new lines use the old enum value, call the removed function, include the moved header and test the
    selector. Both also edit one file (a conflict), and M deletes a file N changes (modify/delete). Decoys: a name M
    binds itself (an `as` import), and one N keeps."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        r.mkdir()
        g(r, "init", "-q", "-b", "develop")
        write(
            r,
            {
                "rk/lib/api.h": """\
                      #pragma once
                      typedef enum {
                          ROLE_A = 0,
                          ROLE_B
                      } role_t;
                      typedef struct { int a_len; int b_len; } op_t;
                      """,
                "rk/lib/util.py": "def make_widget(x):\n    return x + 1\n\n\ndef other(x):\n    return x\n",
                "rk/inc/thing.h": "#pragma once\nint thing(void);\n",
                "rk/sel.py": "def select(paths):\n    return [p for p in paths if 'Default' not in p]\n",
                "rk/shared.txt": "one\ntwo\nthree\n",
                "rk/gone.py": "def gone():\n    return 1\n",
            },
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        base = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-qb", "n")
        write(
            r,
            {
                "rk/lib/api.h": """\
                      #pragma once
                      typedef enum {
                          ROLE_SRC0 = 0,
                          ROLE_B
                      } role_t;
                      typedef struct { int src0_len; int b_len; } op_t;
                      """,
                "rk/lib/util.py": "def other(x):\n    return x\n",
                "rk/sel.py": "def select(paths):\n    return [p for p in paths if p != 'template']\n",
                "rk/shared.txt": "one\nN two\nthree\n",
                "rk/gone.py": "def gone():\n    return 2\n",
            },
        )
        (r / "rk/lib2/inc").mkdir(parents=True)
        g(r, "mv", "rk/inc/thing.h", "rk/lib2/inc/thing.h")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "n")
        self.n_head = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", base)
        g(r, "checkout", "-qb", "m")
        write(
            r,
            {
                "rk/src/use.cpp": """\
                      #include "inc/thing.h"
                      #include "lib/api.h"
                      int f(op_t* o) { return ROLE_A + ROLE_B + o->a_len + thing(); }
                      """,
                "rk/tests/test_use.py": """\
                      from lib.util import make_widget
                      import sel


                      def test_use():
                          assert make_widget(1) == 2
                          assert sel.select(["a"]) == ["a"]
                      """,
                "rk/tests/test_alias.py": """\
                      from lib.util import other as make_widget


                      def test_alias():
                          assert make_widget(3) == 3
                      """,
                "rk/shared.txt": "one\nM two\nthree\n",
                "rk/gone.py": None,
            },
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "m")
        self.m_head = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", "develop")
        write(r, {"rk/dev.txt": "develop moved on\n"})
        g(r, "add", "-A")
        g(r, "commit", "-qm", "develop moves")
        self.develop = g(r, "rev-parse", "HEAD")
        self.prs = {
            1: {
                "n": 1,
                "head": self.n_head,
                "base": None,
                "draft": False,
                "title": "N",
                "author": "a",
                "stack": set(),
            },
            2: {
                "n": 2,
                "head": self.m_head,
                "base": None,
                "draft": False,
                "title": "M",
                "author": "b",
                "stack": set(),
            },
        }

    def scan(self, n: int, cache=None) -> dict:
        res = clashes.scan(str(self.repo), n, self.prs, self.develop, cache=cache)
        self.assertNotIn("error", res)
        self.assertEqual(res["not_run"], [])
        return res

    def test_silent_clashes_both_ways_and_both_orders(self):
        with tempfile.TemporaryDirectory() as cache:
            res = self.scan(1, cache)
            p = res["pairs"][0]
            got = {x["name"]: x for x in p["fwd"]["names"]}
            self.assertEqual(set(got), {"ROLE_A", "make_widget", "a_len"})
            self.assertEqual(got["ROLE_A"]["kind"], "enumerator")
            self.assertEqual(
                (got["ROLE_A"]["path"], got["ROLE_A"]["line"]), ("rk/lib/api.h", 3)
            )
            self.assertEqual(
                [u[:2] for u in got["ROLE_A"]["uses"]], [["rk/src/use.cpp", 3]]
            )
            # the `as make_widget` binding in test_alias.py is its own: only test_use.py's import counts
            self.assertEqual(
                [u[:2] for u in got["make_widget"]["uses"]],
                [["rk/tests/test_use.py", 1], ["rk/tests/test_use.py", 6]],
            )
            moved = p["fwd"]["paths"][0]
            self.assertEqual(
                (moved["old"], moved["new"], moved["moved"]),
                ("rk/inc/thing.h", "rk/lib2/inc/thing.h", True),
            )
            self.assertEqual([x["name"] for x in p["fwd"]["leads"]], ["select"])
            for key in ("nm", "mn"):
                self.assertEqual(
                    p["orders"][key]["conflicts"], ["rk/gone.py", "rk/shared.txt"]
                )
                self.assertEqual(
                    [(x["path"], x["deleted_by"]) for x in p["orders"][key]["moddel"]],
                    [("rk/gone.py", 2)],
                )
            self.assertEqual(clashes.verdict(res)[0], "fail")
            # The same scan from M's side: the clashes run the other way.
            back = self.scan(2, cache)["pairs"][0]
            self.assertEqual(
                {x["name"] for x in back["rev"]["names"]},
                {"ROLE_A", "make_widget", "a_len"},
            )
            self.assertEqual(back["fwd"]["names"], [])
            # Cached facts (per head and merge base) and landings (per develop) give the same answer.
            self.assertTrue(list(pathlib.Path(cache).glob("develop-*.json")))
            again = self.scan(1, cache)["pairs"][0]
            self.assertEqual(
                json.dumps(again, sort_keys=True, default=sorted),
                json.dumps(p, sort_keys=True, default=sorted),
            )
        # No temporary landing is left as a ref.
        self.assertEqual(
            g(self.repo, "for-each-ref", "--format=%(refname)").split(),
            ["refs/heads/develop", "refs/heads/m", "refs/heads/n"],
        )

    def test_a_stacked_pr_skips_its_stack_and_a_kept_name_is_no_clash(self):
        r = self.repo
        g(r, "checkout", "-qb", "k", self.m_head)
        write(r, {"rk/src/k.cpp": "int k() { return ROLE_B; }\n"})
        g(r, "add", "-A")
        g(r, "commit", "-qm", "k")
        self.prs[3] = {
            "n": 3,
            "head": g(r, "rev-parse", "HEAD"),
            "base": None,
            "draft": False,
            "title": "K",
            "author": "c",
            "stack": {2},
        }
        self.prs[2]["stack"] = {3}
        res = self.scan(3)
        self.assertEqual(
            res["skipped_stack"], [2]
        )  # stacked: compared with nothing of its own stack
        self.assertNotIn("ROLE_B", {x["name"] for x in res["pairs"][0]["rev"]["names"]})

    def test_a_pr_on_a_feature_branch_is_read_against_it_when_the_mirror_has_it(self):
        """PRs that target feature branches are read against their branch, so their facts never hold the branch's
        own work."""
        st = {
            "prs": {
                "1": {"status": "open", "head": self.n_head, "base": "users/user/feat"},
                "2": {"status": "open", "head": self.m_head, "base": "develop"},
            }
        }
        prs = clashes.open_prs(st)
        self.assertEqual(
            (prs[1]["base"], prs[1]["base_branch"], prs[2]["base_branch"]),
            (None, "users/user/feat", None),
        )
        self.assertEqual(
            clashes.own_base(str(self.repo), prs[1], self.develop), self.develop
        )  # not in the mirror
        g(self.repo, "update-ref", "refs/remotes/origin/users/user/feat", self.n_head)
        self.assertEqual(
            clashes.own_base(str(self.repo), prs[1], self.develop),
            "refs/remotes/origin/users/user/feat",
        )
        self.assertEqual(
            clashes.own_base(str(self.repo), prs[2], self.develop), self.develop
        )

    def test_a_head_that_is_not_local_is_not_run(self):
        self.prs[2]["head"] = "f" * 40
        res = clashes.scan(str(self.repo), 1, self.prs, self.develop)
        self.assertEqual(res["not_run"], [(2, "its head is not in the mirror")])
        self.assertEqual(clashes.verdict(res)[0], "skip")
        self.prs[1]["head"] = "f" * 40
        self.assertEqual(
            clashes.verdict(clashes.scan(str(self.repo), 1, self.prs, self.develop))[0],
            "skip",
        )


class TestArity(unittest.TestCase):
    """#33333 (a draft) adds a parameter to rocke_gemm_emit_load_phase_ex (header and source); #44444's new
    gemm_build_kloop_tdm.cpp:281 calls it with the old seven arguments. Both merge clean; together they do not
    compile ("too few arguments"), and the clash scan says so."""

    def test_declarations_definitions_and_calls_with_their_counts(self):
        src = textwrap.dedent(
            """\
            #define EMIT(x) emit_one(x)
            rocke_value_t* emit_frag(ctx_t* ctx,
                                     int frag_len,
                                     rocke_value_t* swz_row);
            static int pick(int a, int b = 2, ...) { return a; }
            void none(void);
            int f(ctx_t* ctx)
            {
                rocke_value_t* v = emit_frag(ctx, 8, make(ctx, g(1, 2)));
                if(pick(1))
                    ROCKE_FAIL(reason, "a, b");  /* emit_frag(x) in a comment */
                return none() + h(arr[1, 2], {3, 4});
            }
            """
        )
        sigs, calls = clashes.c_signatures(src)
        self.assertEqual(sigs["emit_frag"], [[2, 4, 3, 3]])
        self.assertEqual(sigs["pick"], [[5, 5, 1, clashes.VARIADIC]])
        self.assertEqual(sigs["none"], [[6, 6, 0, 0]])
        self.assertEqual(sigs["f"], [[7, 7, 1, 1]])
        self.assertEqual(
            sorted(
                c
                for c in calls
                if c[0]
                in (
                    "emit_frag",
                    "make",
                    "g",
                    "pick",
                    "ROCKE_FAIL",
                    "none",
                    "h",
                    "emit_one",
                )
            ),
            [
                ["ROCKE_FAIL", 11, 2],
                ["emit_frag", 9, 3],
                ["emit_one", 1, 1],
                ["g", 9, 2],
                ["h", 12, 2],
                ["make", 9, 2],
                ["none", 12, 0],
                ["pick", 10, 1],
            ],
        )

    def test_a_changed_parameter_list_and_an_overload_that_keeps_the_old_one(self):
        old = "void load_phase(ctx_t* c,\n          int k,\n          hook_t h);\nvoid keep_both(int a);\n"
        new = (
            "void load_phase(ctx_t* c,\n          int k,\n          hook_t h,\n          queue_t* q);\n"
            "void keep_both(int a);\nvoid keep_both(int a, int b);\n"
        )
        got = clashes.changed_signatures("x.h", old, new, [(3, 3)], [(3, 4), (6, 6)])
        self.assertEqual(
            [(x["name"], x["line"], x["old"], x["new"]) for x in got],
            [("load_phase", 1, [[3, 3]], [[4, 4]])],
        )

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        r.mkdir()
        g(r, "init", "-q", "-b", "develop")
        write(
            r,
            {
                "rk/include/emit.h": "#pragma once\nvoid emit_load(ctx_t* ctx,\n               int k,\n"
                "               hook_t before);\n",
                "rk/src/emit.c": '#include "emit.h"\nvoid emit_load(ctx_t* ctx, int k, hook_t before)\n{\n}\n',
                "rk/src/user.c": "void use(ctx_t* ctx) { emit_load(ctx, 1, NULL); }\n",
            },
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        base = g(r, "rev-parse", "HEAD")
        g(
            r, "checkout", "-qb", "n"
        )  # #33333's side: one more parameter, its own caller updated
        write(
            r,
            {
                "rk/include/emit.h": "#pragma once\nvoid emit_load(ctx_t* ctx,\n               int k,\n"
                "               hook_t before,\n               queue_t* defer);\n",
                "rk/src/emit.c": '#include "emit.h"\nvoid emit_load(ctx_t* ctx, int k, hook_t before, '
                "queue_t* defer)\n{\n}\n",
                "rk/src/user.c": "void use(ctx_t* ctx) { emit_load(ctx, 1, NULL, NULL); }\n",
            },
        )
        g(r, "commit", "-qam", "n")
        n_head = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-q", base)
        g(r, "checkout", "-qb", "m")  # #44444's side: a new file calling it the old way
        write(
            r,
            {
                "rk/src/kloop.c": '#include "emit.h"\nstatic void refill(ctx_t* ctx)\n{\n'
                "    emit_load(ctx, 2,\n              NULL);\n}\n"
            },
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "m")
        m_head = g(r, "rev-parse", "HEAD")
        self.develop = base
        self.prs = {
            1: {
                "n": 1,
                "head": n_head,
                "base": None,
                "draft": True,
                "title": "N",
                "author": "a",
                "stack": set(),
            },
            2: {
                "n": 2,
                "head": m_head,
                "base": None,
                "draft": False,
                "title": "M",
                "author": "b",
                "stack": set(),
            },
        }

    def test_a_call_with_the_old_arity_is_a_silent_clash_both_ways(self):
        res = clashes.scan(str(self.repo), 2, self.prs, self.develop)
        p = res["pairs"][0]
        self.assertEqual(p["orders"]["nm"]["conflicts"], [])  # they merge clean
        x = p["rev"]["arity"]
        self.assertEqual(
            [
                (
                    a["name"],
                    a["path"],
                    a["old"],
                    a["new"],
                    a["argc"],
                    [u[:2] for u in a["uses"]],
                )
                for a in x
            ],
            [
                (
                    "emit_load",
                    "rk/include/emit.h",
                    [[3, 3]],
                    [[4, 4]],
                    [3],
                    [["rk/src/kloop.c", 4]],
                )
            ],
        )
        self.assertEqual(clashes.verdict(res)[0], "fail")
        text = "\n".join(clashes.render(res, "rk"))
        self.assertIn(
            "silent (arity, a lead from a parse of the signatures: build the merge): `emit_load` "
            "(rk/include/emit.h:2): #1 changes its parameters (3 -> 4); this PR calls it with 3 "
            "arguments at rk/src/kloop.c:4 (merges clean)",
            text,
        )
        self.assertIn("ports the calls of `emit_load` to the new parameters", text)
        # From the other side: the same clash, the other way.
        back = clashes.scan(str(self.repo), 1, self.prs, self.develop)["pairs"][0]
        self.assertEqual([a["name"] for a in back["fwd"]["arity"]], ["emit_load"])
        self.assertEqual(back["rev"]["arity"], [])


class TestCompileDb(unittest.TestCase):
    """The pair builds' flags: the shared build's core `compile_commands.json`, which PR code's build wrote, read by the
    result reader's rules and parsed as data, and only while the build's slot is kept for it.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.tree = self.d / "wt7m"
        self.workers = self.d / "workers"
        (self.workers / "pr7" / "checks").mkdir(parents=True)
        self.cfg = {"paths": {"working_dir": str(self.d), "workers": str(self.workers)}}
        t = self.tree
        self.entries = [
            {
                "directory": "/b",
                "file": f"{t}/rk/src/a.cpp",
                "arguments": ["c++", f"-I{t}/rk/include", "-c", f"{t}/rk/src/a.cpp"],
            },
            {
                "directory": "/b",
                "file": f"{t}/rk/src/b.cpp",
                "command": "c++ -DX -c b.cpp",
            },
            "not an entry",
            {"directory": "/b", "file": f"{t}/rk/src/c.cpp", "arguments": {"c++": 1}},
            {"directory": "/b", "file": "/elsewhere/d.cpp", "arguments": ["c++"]},
        ]

    def build(self, build: pathlib.Path, text: str | None = None) -> pathlib.Path:
        """`build` named as the review's shared build, its core build's commands `text` (else the entries)."""
        (build / "core").mkdir(parents=True)
        db = build / "core" / "compile_commands.json"
        db.write_text(json.dumps(self.entries) if text is None else text)
        rec = self.workers / "pr7" / "checks" / review.SHARED_BUILD
        rec.write_text(f"{build}\n{self.tree}\n{'a' * 40}\n")
        return db

    def test_the_core_builds_commands_read_as_data(self):
        self.build(self.d / "build" / "wt7m")
        db, tree = clashes.compile_db(self.cfg, 7)
        self.assertEqual(tree, str(self.tree))
        self.assertEqual(
            db,
            {
                "rk/src/a.cpp": [
                    "c++",
                    f"-I{self.tree}/rk/include",
                    "-c",
                    f"{self.tree}/rk/src/a.cpp",
                ],
                "rk/src/b.cpp": ["c++", "-DX", "-c", "b.cpp"],
            },
        )

    def test_a_link_too_large_or_too_deep_file_is_not_read(self):
        db = self.build(self.d / "build" / "wt7m")
        planted = self.d / "elsewhere.json"
        planted.write_text(json.dumps(self.entries))
        db.unlink()
        db.symlink_to(planted)
        self.assertEqual(clashes.compile_db(self.cfg, 7), ({}, ""))
        db.unlink()
        db.write_text("[" * 100000 + "]" * 100000)
        self.assertEqual(clashes.compile_db(self.cfg, 7), ({}, ""))
        db.write_text(json.dumps(self.entries))
        sys.path.insert(0, str(ROOT / "worker"))
        import pr_results

        with mock.patch.object(pr_results, "FILE_MAX", 64):
            self.assertEqual(clashes.compile_db(self.cfg, 7), ({}, ""))

    def test_a_build_in_a_slot_only_while_the_slot_is_kept_for_it(self):
        top, locks = self.d / "slots", self.d / "slots" / "locks"
        locks.mkdir(parents=True)
        build = top / "rkpr3" / "build" / "wt7m"
        self.build(build)
        with mock.patch.multiple(L, SLOTS_DIR=str(top), SLOT_LOCKS=str(locks)):
            self.assertEqual(clashes.compile_db(self.cfg, 7), ({}, ""))
            slots.Records().write("rkpr3", {"state": "kept", "build": str(build)})
            db, tree = clashes.compile_db(self.cfg, 7)
            self.assertEqual(sorted(db), ["rk/src/a.cpp", "rk/src/b.cpp"])
            self.assertEqual(tree, str(self.tree))


if __name__ == "__main__":
    unittest.main()
