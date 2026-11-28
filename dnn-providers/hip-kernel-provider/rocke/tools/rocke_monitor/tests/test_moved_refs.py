# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/moved_refs.py: the links and paths a doc names, a CMake variable before
a path, the report, the checks wiring, and the check end to end on a small repo whose PR moves a doc, a module and a
header and deletes a script, leaving a backticked path and a link in a README (as #11111's first round left
README_conv_direct_grouped.md:273 and :300), an import, a mock.patch target, a CMake install path and a workflow step
behind; an include that still resolves, a changelog and a backup file are not reported, and a doc the PR adds has a
broken link. Temp git repos, no network."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "moved_refs.py"
sys.path.insert(0, str(TOOL.parent))
import moved_refs as mr  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestDocs(unittest.TestCase):
    def test_links_outside_fenced_code_and_what_they_name(self):
        text = (
            'See [a](../x/a.md "title") and <a href="b.md">b</a>.\n'
            "```\n[not a link](c.md)\n```\n"
            "[ref]: ./d/e.md\n"
            "An RST `name <f.rst>`_ link.\n"
        )
        self.assertEqual(
            mr.link_targets(text),
            [(1, "../x/a.md"), (1, "b.md"), (5, "./d/e.md"), (6, "f.rst")],
        )
        doc = "rk/docs/guide.md"
        self.assertEqual(mr.link_path(doc, "../x/a.md#part"), "rk/x/a.md")
        self.assertEqual(mr.link_path(doc, "/rk/top.md"), "rk/top.md")
        self.assertEqual(mr.link_path(doc, "with%20space.md"), "rk/docs/with space.md")
        for skipped in (
            "https://example.com/a.md",
            "mailto:a@b",
            "#anchor",
            "{{ page }}.md",
            "../../../../out.md",
            "<x>",
        ):
            self.assertIsNone(mr.link_path(doc, skipped), skipped)

    def test_a_docs_references_are_its_links_and_its_paths(self):
        text = "x\nsee `platform/docs/guide.md` and [g](../g.md)\n"
        refs = mr.doc_refs("rk/lib/README.md", text, {2})
        self.assertEqual(
            [(r["kind"], r["ref"], r["exact"], r["tails"]) for r in refs],
            [
                ("link", "../g.md", ["rk/g.md"], []),
                (
                    "path",
                    "platform/docs/guide.md",
                    ["rk/lib/platform/docs/guide.md"],
                    ["platform/docs/guide.md"],
                ),
            ],
        )
        mods = {"rocke.old_mod": {}}
        got = mr.module_refs(
            "rk/a.md",
            "run `python -m rocke.old_mod.cli` or rocke.old_modx\n",
            {1},
            mods,
        )
        self.assertEqual(
            [(r["ref"], r["module"]) for r in got],
            [("rocke.old_mod.cli", "rocke.old_mod")],
        )

    def test_cmake_variables_and_the_files_own_directory(self):
        text = (
            'install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/g.md" "${LIB}/t/x.py")\n'
        )
        got = mr.unvar("rk/CMakeLists.txt", text)
        self.assertEqual(len(got), len(text))
        self.assertIn('"' + " " * 26 + './docs/g.md"', got)
        self.assertIn('"' + " " * 7 + 't/x.py"', got)
        # In an included .cmake file CMAKE_CURRENT_SOURCE_DIR is the includer's: no `./` there.
        self.assertNotIn("./docs", mr.unvar("rk/cmake/T.cmake", text))
        self.assertTrue(
            mr.HISTORY.search("rk/CHANGELOG.md")
            and mr.HISTORY.search("docs/release-notes.rst")
        )
        self.assertFalse(mr.HISTORY.search("rk/history_buffer.py"))
        self.assertTrue(mr.BACKUP.search("rk/a.py.orig"))


class TestReport(unittest.TestCase):
    def test_stale_references_and_broken_links_fail_and_nothing_to_check_skips(self):
        res = {
            "head": "h" * 40,
            "mb": "m" * 40,
            "error": None,
            "secs": 1.0,
            "candidates": 3,
            "docs": 1,
            "links": 2,
            "moved": [["rk/a/g.md", "rk/b/g.md"], ["rk/t.py", None]],
            "stale": [
                {
                    "path": "rk/lib/R.md",
                    "line": 3,
                    "kind": "path",
                    "ref": "a/g.md",
                    "old": "rk/a/g.md",
                    "new": "rk/b/g.md",
                }
            ],
            "broken": [
                {"path": "rk/N.md", "line": 1, "ref": "x.md", "target": "rk/x.md"}
            ],
        }
        lines = mr.render(res, "rk")
        self.assertIn("  lib/R.md:3 `a/g.md` (path): moved to b/g.md", lines)
        self.assertIn("  N.md:1 `x.md` -> x.md (missing)", lines)
        self.assertEqual(
            lines[-1],
            "MOVEDREFS fail 1 reference name moved or deleted files (R.md:3); 1 broken link in "
            "changed docs (N.md:1)",
        )
        self.assertEqual(
            mr.verdict(dict(res, stale=[], broken=[]), "rk"),
            (
                "pass",
                "no reference names a moved or deleted file (1 moved, 1 deleted); 2 links in 1 "
                "changed docs resolve",
            ),
        )
        self.assertEqual(
            mr.verdict(dict(res, moved=[], docs=0), "rk")[1],
            "nothing to check: the PR moves or deletes no file and changes no doc",
        )
        # Files the cap left unread: NOT RUN, never clean.
        self.assertEqual(
            mr.verdict(
                dict(res, stale=[], broken=[], candidates=mr.CANDIDATES + 5), "rk"
            ),
            ("skip", "NOT RUN in part: 5 of the files that name one were not read"),
        )
        self.assertEqual(
            mr.render({"error": "x"}, "rk"), ["NOT RUN: x", "MOVEDREFS skip NOT RUN: x"]
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

    def test_every_review_runs_it_at_the_head_and_a_gate_does_not(self):
        from rkmon_lib import review

        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "first-round", self.PLAN, None)
        sec = script.split(f"sec '{review.MOVED_REFS_SECTION}'", 1)[1].split(
            "\nsec ", 1
        )[0]
        self.assertIn(
            f"nice -n 10 timeout {review.MOVED_REFS_BUDGET_S + 60} python -I -B $M/tools/moved_refs.py --repo "
            f"$R0 --head {'a' * 40} --rocke-path $K --base $BREF --json \"$OUT/moved_refs.json\" --budget "
            f"{review.MOVED_REFS_BUDGET_S}",
            sec,
        )
        self.assertIn("st movedrefs ", sec)
        self.assertIn("MOVEDREFS skip no result", sec)
        self.assertIn("movedrefs", review.LEAD_SECTIONS)
        self.assertGreater(
            script.index(review.MOVED_REFS_SECTION),
            script.rindex('\nif [ "$MERGED" = 1 ]; then\n'),
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
        self.assertNotIn("moved_refs.py", gate)


class TestCheck(unittest.TestCase):
    """The provider `prov`, rocKE at prov/rk."""

    BASE = {
        "prov/rk/platform/docs/guide.md": "# Guide\n",
        "prov/rk/library/keep.md": "keep\n",
        "prov/rk/library/kernels/README.md": "# Kernels\nSee `platform/docs/guide.md` for the guide.\n"
        "Also [the guide](../../platform/docs/guide.md).\n"
        "And [kept](../keep.md) and [web](https://example.com/platform/docs/guide.md).\n",
        "prov/rk/platform/python/rocke/__init__.py": "",
        "prov/rk/platform/python/rocke/old_mod.py": "def f():\n    return 1\n",
        "prov/rk/library/kernels/use.py": "from rocke import old_mod\nfrom unittest import mock\n"
        'PATCH = "rocke.old_mod.f"\n',
        "prov/rk/platform/cpp/include/rocke/hdr.h": "#pragma once\n",
        "prov/rk/platform/cpp/src/x.cpp": '#include "rocke/hdr.h"\nint x;\n',
        "prov/rk/platform/CMakeLists.txt": 'install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/guide.md" '
        "DESTINATION docs)\n",
        "prov/rk/platform/tools/gone.py": "print('bye')\n",
        "prov/rk/CHANGELOG.md": "- moved platform/docs/guide.md\n",
        "prov/rk/library/kernels/README.md.orig": "See `platform/docs/guide.md`.\n",
        ".github/workflows/ci.yml": "steps:\n  - run: python prov/rk/platform/tools/gone.py\n",
    }

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        for path, text in self.BASE.items():
            (r / path).parent.mkdir(parents=True, exist_ok=True)
            (r / path).write_text(text)
        g(r, "init", "-q", "-b", "develop")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        g(r, "checkout", "-qb", "pr")
        for old, new in (
            ("prov/rk/platform/docs/guide.md", "prov/rk/library/docs/guide.md"),
            (
                "prov/rk/platform/python/rocke/old_mod.py",
                "prov/rk/library/kernels/new_mod.py",
            ),
            (
                "prov/rk/platform/cpp/include/rocke/hdr.h",
                "prov/rk/library/cpp/include/rocke/hdr.h",
            ),
        ):
            (r / new).parent.mkdir(parents=True, exist_ok=True)
            g(r, "mv", old, new)
        g(r, "rm", "-q", "prov/rk/platform/tools/gone.py")
        (r / "prov/rk/library/NEW.md").write_text(
            "[missing](missing/file.md), [dir](kernels/), [top](#top), "
            "[t]({{ x }})\n"
        )
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")

    def test_the_references_left_behind_and_a_broken_link(self):
        res = mr.check(str(self.repo), self.head, "prov/rk", "develop")
        self.assertIsNone(res["error"], res)
        self.assertEqual(
            sorted(res["moved"]),
            [
                [
                    "prov/rk/platform/cpp/include/rocke/hdr.h",
                    "prov/rk/library/cpp/include/rocke/hdr.h",
                ],
                ["prov/rk/platform/docs/guide.md", "prov/rk/library/docs/guide.md"],
                [
                    "prov/rk/platform/python/rocke/old_mod.py",
                    "prov/rk/library/kernels/new_mod.py",
                ],
                ["prov/rk/platform/tools/gone.py", None],
            ],
        )
        got = [(s["path"], s["line"], s["kind"], s["ref"]) for s in res["stale"]]
        self.assertEqual(
            got,
            [
                ("prov/rk/library/kernels/use.py", 1, "import", "rocke.old_mod"),
                ("prov/rk/library/kernels/use.py", 3, "module", "rocke.old_mod.f"),
                (
                    ".github/workflows/ci.yml",
                    2,
                    "path",
                    "prov/rk/platform/tools/gone.py",
                ),
                ("prov/rk/platform/CMakeLists.txt", 1, "path", "./docs/guide.md"),
                (
                    "prov/rk/library/kernels/README.md",
                    2,
                    "path",
                    "platform/docs/guide.md",
                ),
                (
                    "prov/rk/library/kernels/README.md",
                    3,
                    "link",
                    "../../platform/docs/guide.md",
                ),
            ],
        )
        # The include still resolves (rocke/hdr.h under the library's include root); the changelog and the backup
        # file are not read.
        self.assertFalse(
            any(
                s["path"].endswith(("x.cpp", "CHANGELOG.md", ".orig"))
                for s in res["stale"]
            )
        )
        self.assertEqual(
            [(b["path"], b["ref"]) for b in res["broken"]],
            [("prov/rk/library/NEW.md", "missing/file.md")],
        )
        self.assertEqual(mr.verdict(res, "prov/rk")[0], "fail")

    def test_the_cli_prints_the_section_and_writes_the_json(self):
        out = pathlib.Path(self.d.name) / "moved.json"
        p = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                "--repo",
                str(self.repo),
                "--head",
                self.head,
                "--rocke-path",
                "prov/rk",
                "--base",
                "develop",
                "--json",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(
            p.stdout.splitlines()[-1].startswith(
                "MOVEDREFS fail 6 references name moved or deleted " "files"
            )
        )
        self.assertIn(
            "  library/kernels/README.md:2 `platform/docs/guide.md` (path): moved to library/docs/guide.md",
            p.stdout,
        )
        self.assertEqual(len(json.loads(out.read_text())["stale"]), 6)


if __name__ == "__main__":
    unittest.main()
