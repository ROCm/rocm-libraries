# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The checked-in tree carries what a build needs: every generated part (tools/generated.py: the docs' generated
sections and the generated files) is committed and in step with its source, made over the tree or over the files the
image takes from it, the documents are the package's own, and the build's own checks (the refresh, run alone; a build
stopped by a part out of step; a release's VERSION, written into the tree once its checks pass; the tests' PATH and
bytecode folder) work.
What the package holds is one list (package_files.tracked), which the build, the lints and the tests' copy of
the package take: never a file git ignores, in a git work tree, out of one, or in a folder git ignores whole.

The generator (tools/generated.py): the docs index and each doc's kind; the markdown it writes a cell, a
code span, a table and a fenced block with; the markers, and what is wrong with them; each section from its source,
through the functions the commands print with, naming rkmon whatever runs it; the internal settings doc, made whole
from the registry; and drift and the refresh over a tree:
a section or a file out of step with its source fails, naming it and the refresh, the refresh mends it, writing only
what differs, and a second refresh changes nothing; a section that a doc of its place lacks is named with the markers
to add, and the refresh never adds one; a marker problem, a section out of its place, or a text its source cannot
make, stops the refresh before it writes; VERSION's form is checked, and the refresh never writes it.
"""

import ast
import contextlib
import datetime as dt
import io
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT), str(ROOT / "tests")]
sys.dont_write_bytecode = True

import build_context  # noqa: E402
import generated  # noqa: E402
import make_package  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    agentic,
    commands,
    container_layout,
    deploy_plan,
    deployment,
    features,
    file_types,
    gh,
    image,
    outcome,
    outward,
    package_files,
    paths,
    post,
    program,
    registry,
    secrets_registry,
    settings_file,
    states,
    steps,
)
from rkmon_lib.checks import Command  # noqa: E402

G = generated
DEPLOY, BUILD = file_types.DEPLOY, file_types.BUILD


def body(name: str, root=ROOT, place: str = "") -> str:
    """A section's text between its markers, as section_text writes it: its BEGIN's three lines and a blank line
    before it, a blank line and its END after it."""
    s = G.SECTIONS[name]
    lines = G.section_text(s, pathlib.Path(root), place or s.places[0]).split("\n")
    return "\n".join(lines[4:-2])


# The build settings reference's header: each key, what the build is given from it, and its description
BUILD_HEADER = ("Key", "Given to the build as", "What it is")


def table_rows(text: str, *header) -> list:
    """The rows of the table in TEXT under HEADER, each a list of its cells as written."""
    lines = text.split("\n")
    at = lines.index("| " + " | ".join(header) + " |")
    out = []
    for line in lines[at + 2 :]:
        if not line.startswith("|"):
            break
        out.append([c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]])
    return out


class TestCommitted(unittest.TestCase):
    def test_every_generated_part_is_in_step_with_its_source(self):
        """Over the package's own tree: each section between markers in its docs is the generator's
        text, each generated file is there and its source's text, and VERSION is in its form; a marker the generator
        does not know, or a BEGIN with no END, is a problem too. The failure names each, with the refresh.
        """
        problems = G.drift(ROOT)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_each_generated_part_made_over_the_files_the_image_takes_is_the_committed_one(
        self,
    ):
        """The image's suite runs the test above over the image's own files, which leave out what .dockerignore does
        (image.PACKAGE_ONLY among it): so each section of the package's docs, and each generated file, made over a copy
        of the package's files without those, is the committed one."""
        patterns = build_context.patterns(ROOT)
        with tempfile.TemporaryDirectory() as d:
            copy = pathlib.Path(d)
            for f in package_files.tracked(ROOT):
                rel = f.relative_to(ROOT).as_posix()
                if not build_context.ignored(patterns, rel):
                    (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, copy / rel)
            self.assertEqual([p for p in image.PACKAGE_ONLY if (copy / p).exists()], [])
            unlike = []
            for doc in package_files.tracked(ROOT):
                if doc.suffix != ".md":
                    continue
                rel, text = doc.relative_to(ROOT).as_posix(), doc.read_text("utf-8")
                lines = text.split("\n")
                for b in G.blocks(text)[0]:
                    made = G.section_text(G.SECTIONS[b.name], copy, rel)
                    if made != "\n".join(lines[b.first : b.last + 1]):
                        unlike.append(f"{rel}: the section {b.name}")
            for f in G.FILES:
                if f.render and f.render(copy) != (ROOT / f.path).read_text("utf-8"):
                    unlike.append(f.path)
        self.assertEqual(unlike, [])

    def test_the_documents_and_the_agentic_files_are_the_package_s_own(self):
        """The design, the operations guide and the agentic files are the package's own files, as the restart runbook
        and the review pipeline guide are: in the tree, and not among the files a build makes.
        """
        for own, in_tree in (
            ("docs/design.md", "docs/design.md"),
            ("docs/operations_guide.md", "docs/operations_guide.md"),
            ("agentic", "agentic/skills/rocke-monitor/SKILL.md"),
        ):
            self.assertTrue((ROOT / in_tree).is_file(), in_tree)
            self.assertNotIn(own, {f.path for f in G.FILES})


class TestWhatACopyLeaves(unittest.TestCase):
    """What a developer's copy leaves wherever it runs has one home, paths.COPY_OWN_LEFTOVERS: git leaves each out
    wherever it is (.gitignore), as the image's build does (.dockerignore), and the scan for stray private files never
    looks into the folders among them (deployment.STRAY_SKIP)."""

    def places(self) -> list:
        """Each leftover at the top and deep in the package's own entries, a folder's with a file in it."""
        out = []
        for p in paths.COPY_OWN_LEFTOVERS:
            name = p.rstrip("/").replace("*", "x")
            for where in (name, f"tests/{name}", f"rkmon_lib/deep/{name}"):
                out.append(f"{where}/v/x" if p.endswith("/") else where)
        return out

    def test_git_leaves_each_out_wherever_it_is(self):
        """As git reads the package's .gitignore, in a repository of its own."""
        places = self.places()
        with tempfile.TemporaryDirectory() as d:
            try:
                subprocess.run(["git", "init", "-q", d], check=True)
            except (OSError, subprocess.CalledProcessError):
                self.skipTest("no git")
            (pathlib.Path(d) / ".gitignore").write_text(
                (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
            )
            p = subprocess.run(
                ["git", "-C", d, "check-ignore", "--no-index", *places],
                capture_output=True,
                text=True,
            )
        self.assertEqual(p.stdout.split(), places)

    def test_the_image_s_build_leaves_each_out_wherever_it_is(self):
        pats = build_context.patterns(ROOT)
        for where in self.places():
            with self.subTest(path=where):
                self.assertTrue(build_context.ignored(pats, where))

    def test_the_scan_never_looks_into_the_folders_among_them(self):
        folders = [p[:-1] for p in paths.COPY_OWN_LEFTOVERS if p.endswith("/")]
        self.assertTrue(folders)
        self.assertEqual(deployment.STRAY_SKIP, (".git", *folders))


class TestBuildChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)

    def build(self, *argv):
        """make_package's main with ARGV: (its exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = make_package.main(list(argv))
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue()

    def test_the_refresh_runs_alone_and_writes_through_the_generator(self):
        with mock.patch.object(
            G, "refresh", return_value=["README.md", "docs/internal_settings.md"]
        ) as r:
            self.assertEqual(
                self.build("--refresh-generated"),
                (
                    0,
                    "wrote README.md, docs/internal_settings.md: review the change and commit it\n",
                    "",
                ),
            )
        r.assert_called_once_with(make_package.ROOT)
        with mock.patch.object(G, "refresh", return_value=[]):
            rc, out, _ = self.build("--refresh-generated")
        self.assertEqual(
            (rc, out),
            (0, "nothing to write: every generated part is in step with its source\n"),
        )
        with mock.patch.object(
            G, "refresh", side_effect=ValueError("nothing written: x")
        ):
            self.assertEqual(
                self.build("--refresh-generated"), ("nothing written: x", "", "")
            )
        with mock.patch.object(G, "refresh") as r:
            rc, _, err = self.build(
                "--refresh-generated", "--out", str(self.d), "--no-tests"
            )
        self.assertEqual(rc, 2)
        self.assertIn(
            "--refresh-generated runs alone, with no other option: --out, --no-tests",
            err,
        )
        r.assert_not_called()

    def test_a_build_needs_its_folder_and_deny_list(self):
        rc, _, err = self.build()
        self.assertEqual(rc, 2)
        self.assertIn("the following arguments are required: --out, --deny", err)

    def test_a_build_stops_when_a_generated_part_is_out_of_step(self):
        """Step 3 compares the generated parts with their sources (generated.drift) and stops, naming each; no
        privacy check and no tarball follow."""
        deny = self.d / "deny.txt"
        deny.write_text("invented" + "-deny-value-q\n", encoding="utf-8")
        out = self.d / "dist"
        found = [f"README.md:3: the section states is out of step: run {G.REFRESH}"]
        with mock.patch.object(
            make_package, "tracked", return_value=[make_package.ROOT / "VERSION"]
        ), mock.patch.object(G, "drift", return_value=found) as d:
            rc, _, _ = self.build("--out", str(out), "--deny", str(deny), "--no-tests")
        d.assert_called_once_with(make_package.ROOT)
        self.assertEqual(
            rc,
            "the generated parts are out of step with their sources:\n  " + found[0],
        )
        self.assertFalse((out / "privacy_report.txt").exists())
        self.assertEqual(list(out.glob("*.tar.gz")), [])

    def test_a_build_refuses_a_folder_where_git_tracks_no_file(self):
        """The package is the files git lists in a clone of its repository: in a folder where git tracks none, a copy
        of every file would package whatever was left there, a secret file among them, so the build refuses there
        before it writes anything. The readers of such a folder (a built one: the lints, the drift check) take every
        file but a Python environment's.
        """
        tree, out, deny = self.d / "tree", self.d / "dist", self.d / "deny.txt"
        (tree / ".venv").mkdir(parents=True)
        for rel in ("VERSION", "config.json", "token.txt", ".venv/pyvenv.cfg"):
            (tree / rel).write_text("invented\n", encoding="utf-8")
        deny.write_text("invented" + "-deny-value-q\n", encoding="utf-8")
        # git looks for a repository no higher than the test's own folder
        with mock.patch.dict(
            os.environ, {"GIT_CEILING_DIRECTORIES": str(self.d)}
        ), mock.patch.object(make_package, "ROOT", tree), mock.patch.object(
            G, "drift"
        ) as d:
            rc, stdout, _ = self.build(
                "--out", str(out), "--deny", str(deny), "--no-tests"
            )
            readers = package_files.tracked(tree)
        self.assertEqual(
            rc,
            f"git tracks no file in {tree}: the package is built from a clone of its repository",
        )
        self.assertFalse(out.exists())
        d.assert_not_called()
        self.assertEqual(
            sorted(f.relative_to(tree).as_posix() for f in readers),
            ["VERSION", "config.json", "token.txt"],
        )

    EARLIER = make_package.version_text(dt.date(2025, 1, 2))

    def build_tree(self, *flags, check="pass"):
        """A build, with FLAGS, of a tree of its own whose one file is VERSION, an earlier day's; the drift check finds
        nothing, and the privacy check runs the Python CHECK in the built folder: (the build's exit code, its stdout,
        the drift check's mock, the tree's VERSION after it, the built folder's)."""
        tree, out, deny = self.d / "tree", self.d / "dist", self.d / "deny.txt"
        tree.mkdir()
        (tree / "VERSION").write_text(self.EARLIER, encoding="utf-8")
        deny.write_text("invented" + "-deny-value-q\n", encoding="utf-8")
        argv = ("--out", str(out), "--deny", str(deny), "--no-tests", *flags)
        with mock.patch.object(make_package, "ROOT", tree), mock.patch.object(
            make_package, "tracked", return_value=[tree / "VERSION"]
        ), mock.patch.object(G, "drift", return_value=[]) as d, mock.patch.object(
            make_package.privacy,
            "command",
            return_value=[sys.executable, "-B", "-c", check],
        ):
            rc, stdout, _ = self.build(*argv)
        return (
            rc,
            stdout,
            d,
            (tree / "VERSION").read_text(encoding="utf-8"),
            (out / make_package.NAME / "VERSION").read_text(encoding="utf-8"),
        )

    def tarballs(self) -> list:
        """The package files build_tree's build wrote, by name."""
        return sorted(p.name for p in (self.d / "dist").glob("*.tar.gz"))

    def test_a_release_writes_today_s_version_into_the_folder_then_the_tree(self):
        rc, stdout, d, tree, built = self.build_tree("--new-version")
        today = make_package.version_text()
        self.assertEqual((rc, tree, built), (0, today, today))
        self.assertIn(f"wrote VERSION, {dt.date.today():%Y.%m.%d}, into ", stdout)
        # the package file is named by the release's version, as the image built from it is tagged
        self.assertEqual(
            self.tarballs(), [f"rocke_monitor-{dt.date.today():%Y.%m.%d}.tar.gz"]
        )
        # the tree's VERSION, whatever its form, is the release's to replace: the drift check reads every other part
        d.assert_called_once()
        self.assertEqual(
            [f.path for f in d.call_args.kwargs["files"]],
            [f.path for f in G.FILES if f.path != "VERSION"],
        )
        self.assertTrue(G.RELEASE.endswith(" --new-version"))

    def test_a_build_keeps_the_tree_s_version(self):
        rc, stdout, d, tree, built = self.build_tree()
        self.assertEqual((rc, tree, built), (0, self.EARLIER, self.EARLIER))
        self.assertNotIn("VERSION", stdout)
        d.assert_called_once_with(self.d / "tree")
        # named by the version the package holds, as the image's tag is, whatever the day it was built
        self.assertEqual(self.tarballs(), ["rocke_monitor-2025.01.02.tar.gz"])

    def test_a_release_whose_checks_fail_leaves_the_tree_s_version(self):
        rc, _, _, tree, built = self.build_tree(
            "--new-version", check="raise SystemExit(3)"
        )
        self.assertEqual(
            rc,
            "the privacy check could not run in the built folder: see its error above",
        )
        self.assertEqual((tree, built), (self.EARLIER, make_package.version_text()))
        self.assertEqual(self.tarballs(), [])

    def test_the_tests_run_on_a_path_of_the_build_s_own(self):
        venv = self.d / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
        env = make_package.test_env(
            {
                "PATH": f"{venv / 'bin'}:/opt/tools/bin:/usr/bin",
                "VIRTUAL_ENV": str(venv),
            },
            self.d / "shim",
            self.d / "pycache",
        )
        parts = env["PATH"].split(os.pathsep)
        self.assertEqual(parts[0], str(self.d / "shim"))
        self.assertNotIn(str(venv / "bin"), parts)
        self.assertIn("/opt/tools/bin", parts)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertEqual(env["PYTHONPYCACHEPREFIX"], str(self.d / "pycache"))
        make_package.python_shim(self.d / "shim")
        shim = (self.d / "shim" / "python").read_text()
        self.assertIn(sys.executable, shim)
        self.assertTrue(os.access(self.d / "shim" / "python", os.X_OK))

    def test_version_text(self):
        self.assertTrue(
            make_package.version_text(dt.date(2026, 1, 2)).endswith("\n2026.01.02\n")
        )
        self.assertTrue(re.search(r"Copyright", make_package.version_text()))


class TestWhatThePackageHolds(unittest.TestCase):
    """package_files.tracked, the one list of the package's files: what git tracks or would add, never what it ignores
    (a root certificate a builder left in the folder: .gitignore), a Python environment's files, state, caches or
    bytecode; the tests' copy of the package (tests/sandbox.py) holds that list and the blank config.json.
    """

    HELD = [".gitignore", "docker/README.md", "rk.py"]

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.base = pathlib.Path(t.name)
        self.pkg = self.base / "pkg"
        for name, text in (
            (".gitignore", "*.crt\n"),
            ("rk.py", "print()\n"),
            ("docker/README.md", "the recipe\n"),
            ("docker/root.crt", "a machine's root\n"),
            ("state/state.json", "{}\n"),
            ("rkmon_lib/__pycache__/x.cpython.pyc", ""),
        ):
            (self.pkg / name).parent.mkdir(parents=True, exist_ok=True)
            (self.pkg / name).write_text(text, encoding="utf-8")
        # git looks for no repository above the test's folder
        p = mock.patch.dict(os.environ, GIT_CEILING_DIRECTORIES=str(self.base))
        p.start()
        self.addCleanup(p.stop)

    def names(self, root: pathlib.Path, files=None) -> list[str]:
        files = package_files.tracked(root) if files is None else files
        return sorted(str(f.relative_to(root)) for f in files)

    def venv(self, folder: pathlib.Path) -> None:
        """A Python environment at FOLDER as a Python before 3.13 makes one: no .gitignore of its own."""
        for rel in ("pyvenv.cfg", "bin/python", "lib/python3.9/site-packages/x.py"):
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text("invented\n", encoding="utf-8")

    def test_a_folder_git_does_not_track(self):
        """The image's copy of the package, an unpacked tarball: the folder's own ignore rules still hold."""
        self.assertEqual(self.names(self.pkg), self.HELD)

    def test_a_git_work_tree(self):
        sandbox.git(self.pkg, "init", "-q")
        sandbox.git(self.pkg, "add", ".gitignore", "rk.py")
        (self.pkg / "new.py").write_text("print()\n", encoding="utf-8")
        self.assertEqual(self.names(self.pkg), sorted([*self.HELD, "new.py"]))

    def test_a_python_environment_in_a_clone(self):
        """A folder that holds pyvenv.cfg is a Python environment, whatever its name and wherever it lies: none of its
        files is the package's, though git lists them."""
        self.venv(self.pkg / "env")
        self.venv(self.pkg / "docker" / "env")
        sandbox.git(self.pkg, "init", "-q")
        sandbox.git(self.pkg, "add", ".gitignore", "rk.py")
        self.assertEqual(self.names(self.pkg), self.HELD)

    def test_a_python_environment_outside_a_repository(self):
        self.venv(self.pkg / "env")
        self.assertEqual(self.names(self.pkg), self.HELD)

    def test_a_folder_git_ignores_whole(self):
        """A build's folder in a repository's ignored dist/: git lists none of it, and its own ignore rules hold."""
        repo = self.base / "repo"
        build = repo / "dist" / "pkg"
        build.parent.mkdir(parents=True)
        (repo / ".gitignore").write_text("/dist/\n", encoding="utf-8")
        sandbox.git(repo, "init", "-q")
        self.pkg.rename(build)
        self.assertEqual(self.names(build), self.HELD)

    def test_a_folder_in_a_repository_that_tracks_none_of_it(self):
        """An unpacked tarball, or a package folder, in another repository's work tree: that repository's ignore
        rules do not hold, the folder's own do."""
        repo = self.base / "repo"
        repo.mkdir()
        (repo / ".gitignore").write_text("*.md\n", encoding="utf-8")
        sandbox.git(repo, "init", "-q")
        sandbox.git(repo, "add", ".gitignore")
        self.pkg.rename(repo / "pkg")
        self.assertEqual(self.names(repo / "pkg"), self.HELD)

    def test_a_global_ignore_file(self):
        """Where git tracks none of the folder, the ignore file of git's global settings (core.excludesFile) leaves
        out none of it either."""
        ignore, settings = self.base / "ignore", self.base / "gitconfig"
        ignore.write_text("*.md\n", encoding="utf-8")
        settings.write_text(f"[core]\n\texcludesFile = {ignore}\n", encoding="utf-8")
        with mock.patch.dict(os.environ, GIT_CONFIG_GLOBAL=str(settings)):
            self.assertEqual(self.names(self.pkg), self.HELD)

    def test_the_tests_copy_of_the_package(self):
        copy = sandbox.copy_package(self.base / "copy", self.pkg)
        files = [f for f in copy.rglob("*") if f.is_file()]
        self.assertEqual(self.names(copy, files), sorted([*self.HELD, "config.json"]))
        self.assertEqual(
            (copy / "config.json").read_text(encoding="utf-8"),
            package_files.BLANK_CONFIG,
        )


class TestDocsIndex(unittest.TestCase):
    def test_the_index_lists_every_doc_in_its_order(self):
        self.assertEqual(
            [d.path for d in G.DOCS],
            [
                "README.md",
                "QUICK_BUILD_GUIDE.md",
                "QUICK_DEPLOY_GUIDE.md",
                "QUICK_USER_GUIDE.md",
                "docs/FAQ.md",
                "docs/restart.md",
                "docs/design.md",
                "docs/operations_guide.md",
                "docs/review_pipeline.md",
                "docs/maintaining.md",
                "docs/internal_settings.md",
                "agentic/README.md",
                ".claude/skills/rkmon-setup/SKILL.md",
            ],
        )
        self.assertEqual(G.kind_of("docs/internal_settings.md"), G.MAINTAINER)

    def test_each_doc_has_one_row_and_a_kind(self):
        found = [d.path for d in G.DOCS]
        self.assertEqual(len(found), len(set(found)))
        self.assertEqual(found[0], "README.md")
        self.assertEqual({d.kind for d in G.DOCS} - set(G.KINDS), set())
        for d in G.DOCS:
            with self.subTest(doc=d.path):
                self.assertTrue(d.reader and d.job)

    def test_a_doc_s_kind_is_the_index_s_else_by_its_folder(self):
        for path, kind in (
            ("README.md", G.USER),
            ("docs/restart.md", G.USER),
            ("docs/design.md", G.MAINTAINER),
            ("docs/operations_guide.md", G.MAINTAINER),
            ("agentic/README.md", G.SKILL),
            ("agentic/skills/rocke-monitor/SKILL.md", G.SKILL),
            (".claude/skills/rkmon-setup/SKILL.md", G.SKILL),
            ("docs/some_notes.md", G.MAINTAINER),
            ("NOTES.md", G.MAINTAINER),
        ):
            with self.subTest(path=path):
                self.assertEqual(G.kind_of(path), kind)

    def test_the_doc_map_lists_every_doc_of_the_index_each_linked_from_its_place(self):
        """The map is the index's, whatever files the tree it is made in holds: here a tree with none of the docs, as
        the image's tree holds no .claude/ (image.PACKAGE_ONLY)."""
        header = ("Doc", "Reader", "Its one job")
        with tempfile.TemporaryDirectory() as d:
            top = table_rows(body("doc-map", d, "README.md"), *header)
            below = table_rows(body("doc-map", d, "docs/FAQ.md"), *header)
        self.assertEqual(
            top, [[f"[{x.path}]({x.path})", x.reader, x.job] for x in G.DOCS]
        )
        self.assertEqual(
            [r[0] for r in below],
            [
                (
                    f"[{x.path}]({x.path[len('docs/'):]})"
                    if x.path.startswith("docs/")
                    else f"[{x.path}](../{x.path})"
                )
                for x in G.DOCS
            ],
        )


class TestMarkdown(unittest.TestCase):
    def test_a_cell_writes_plain_words_with_markup_escaped(self):
        for text, want in (
            ("a | b", "a \\| b"),
            ("<org>/<repo>", "\\<org>/\\<repo>"),
            ("`x` *y* ~z~ [a](b) $1", "\\`x\\` \\*y\\* \\~z\\~ \\[a](b) \\$1"),
            ("a\\b", "a\\\\b"),
            ("snake_case_word", "snake_case_word"),
            ("_lead and trail_ a_ _b", "\\_lead and trail\\_ a\\_ \\_b"),
            ("R&D &amp; &#169; &#xA9; & x", "R&D &amp;amp; &amp;#169; &amp;#xA9; & x"),
            ("one\n  two", "one<br>  two"),
            ("  padded  ", "padded"),
        ):
            with self.subTest(text=text):
                self.assertEqual(G.cell(text), want)

    def test_a_markdown_cell_escapes_only_its_bars(self):
        self.assertEqual(
            G.cell(" a `x|y` _b_\nc ", markdown=True), "a `x\\|y` _b_<br>c"
        )

    def test_a_code_span_holds_its_backticks_and_in_a_table_its_bars(self):
        for args, want in (
            (("a|b",), "`a\\|b`"),
            (("a|b", False), "`a|b`"),
            (("x `y` z",), "``x `y` z``"),
            (("`x`",), "`` `x` ``"),
            (("a ``b`` c",), "```a ``b`` c```"),
            (("a\nb",), "`a b`"),
        ):
            with self.subTest(args=args):
                self.assertEqual(G.code(*args), want)

    def test_a_table_has_its_header_its_rule_and_a_line_for_each_row(self):
        self.assertEqual(
            G.table(("A", "B"), [("1", "2"), ("3", "4")]),
            "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
        )
        with self.assertRaisesRegex(ValueError, "a row of 1 cells under 2 columns"):
            G.table(("A", "B"), [("1",)])

    def test_a_fenced_block_s_fence_is_longer_than_any_it_holds(self):
        self.assertEqual(G.fenced("\na  \n```\nb\n"), "````text\na\n```\nb\n````")
        self.assertEqual(G.fenced("a", "yaml"), "```yaml\na\n```")


class TestMarkers(unittest.TestCase):
    def test_a_section_has_its_three_line_begin_and_its_end(self):
        text = (
            "intro\n<!-- BEGIN GENERATED: exit-codes\n     source: x\n     refresh: y -->\n\nbody\n\n"
            "<!-- END GENERATED: exit-codes -->\nafter\n"
        )
        self.assertEqual(G.blocks(text), ([G.Block("exit-codes", 1, 7)], []))

    def test_a_begin_on_one_line_is_a_marker_too(self):
        text = "<!-- BEGIN GENERATED: exit-codes -->\nbody\n<!-- END GENERATED: exit-codes -->"
        self.assertEqual(G.blocks(text), ([G.Block("exit-codes", 0, 2)], []))

    def test_what_is_wrong_with_the_markers_is_named_by_its_line(self):
        begin, end = "<!-- BEGIN GENERATED:", "<!-- END GENERATED:"
        for text, found, problems in (
            (
                f"{begin} no-such -->\nx\n{end} no-such -->",
                [],
                [f"1: no section no-such in tools/generated.py: its sections are "],
            ),
            (
                f"a\n{begin} exit-codes -->\nx\n",
                [],
                ["2: the section exit-codes has no END"],
            ),
            (
                f"x\n{end} exit-codes -->",
                [],
                [f"2: an END with no BEGIN: {end} exit-codes -->"],
            ),
            (
                f"{begin} exit-codes -->\n{begin} states -->\nx\n{end} states -->",
                [G.Block("states", 1, 3)],
                [
                    "2: a BEGIN inside the section exit-codes, which has no END before it"
                ],
            ),
            (
                f"{begin} exit-codes -->\nx\n{end} states -->",
                [],
                ["3: the END of states inside the section exit-codes"],
            ),
            (
                f"{begin} Exit_Codes -->",
                [],
                [f"1: not a marker: a BEGIN is {begin} NAME, NAME lowercase words "],
            ),
            (
                f"{begin} exit-codes -->\nx\n{end} exit-codes",
                [],
                [f"3: not a marker: an END is {end} NAME -->"],
            ),
            (
                f"{begin} exit-codes\n     source: x\n",
                [],
                ["1: the BEGIN of exit-codes has no --> to end its comment"],
            ),
        ):
            with self.subTest(text=text):
                got, wrong = G.blocks(text)
                self.assertEqual(got, found)
                self.assertEqual(len(wrong), len(problems), wrong)
                for w, p in zip(wrong, problems):
                    self.assertTrue(w.startswith(p), (w, p))

    def test_a_marker_in_fenced_code_is_text(self):
        for text in (
            "```\n<!-- BEGIN GENERATED: no-such -->\n```\n",
            "~~~~\n<!-- END GENERATED: exit-codes -->\n~~~~\n",
            "````\n```\n<!-- BEGIN GENERATED: no-such -->\n````\n",
            "    <!-- BEGIN GENERATED: no-such -->\n",
            "text `<!-- BEGIN GENERATED: no-such -->` in a line\n",
        ):
            with self.subTest(text=text):
                self.assertEqual(G.blocks(text), ([], []))

    def test_a_section_names_its_source_and_the_refresh(self):
        s = G.SECTIONS["exit-codes"]
        lines = G.section_text(s, ROOT, "README.md").split("\n")
        self.assertEqual(
            lines[:4],
            [
                "<!-- BEGIN GENERATED: exit-codes",
                "     source: rkmon_lib/outcome.py, EXIT_CODES",
                f"     refresh: {G.REFRESH} -->",
                "",
            ],
        )
        self.assertEqual(lines[-2:], ["", "<!-- END GENERATED: exit-codes -->"])
        self.assertEqual(
            G.blocks("\n".join(lines)), ([G.Block("exit-codes", 0, len(lines) - 1)], [])
        )
        self.assertEqual(
            G.REFRESH, "python3 -B tools/make_package.py --refresh-generated"
        )

    def test_each_section_s_source_is_in_the_package_and_its_places_are_docs(self):
        """A section's source starts with a file of the package, or a folder of it (named with its trailing /); its
        places are docs of the index, or files of the agentic folders (kind_of: a skill's), such as the dispatch
        table's rows."""
        docs = {d.path for d in G.DOCS}
        for s in G.SECTIONS.values():
            with self.subTest(section=s.name):
                self.assertRegex(s.name, rf"^{G.NAME}$")
                first = s.source.split(",")[0]
                self.assertTrue(
                    (
                        (ROOT / first).is_dir()
                        if first.endswith("/")
                        else (ROOT / first).is_file()
                    ),
                    s.source,
                )
                self.assertTrue(s.places)
                self.assertEqual(
                    {
                        p
                        for p in s.places
                        if p not in docs
                        and not (p.startswith(G.SKILL_FOLDERS) and (ROOT / p).is_file())
                    },
                    set(),
                )


class TestRenderers(unittest.TestCase):
    def test_each_text_names_rkmon_whatever_runs_the_generator(self):
        with mock.patch.object(sys, "argv", ["/opt/elsewhere/rk", "selftest"]):
            self.assertNotEqual(program.name(), "rkmon")
            text = body("build-steps")
            self.assertEqual(sys.argv, ["/opt/elsewhere/rk", "selftest"])
        first = steps.BUILD_STEPS[0]
        self.assertIn(f"`rkmon {first.command}`", text)
        self.assertNotIn("elsewhere", text)

    def test_capabilities_a_row_each_from_the_registry(self):
        invented = types.SimpleNamespace(
            title="A | B", when="while x_", what="does `y`", needs="<n> & m"
        )
        with mock.patch.dict(features.CAPABILITIES, {"a": invented}, clear=True):
            rows = table_rows(
                body("capabilities"),
                "Capability",
                "Runs while",
                "What it does",
                "What it needs",
            )
        self.assertEqual(rows, [["A \\| B", "while x\\_", "does \\`y\\`", "\\<n> & m"]])
        rows = table_rows(
            body("capabilities"),
            "Capability",
            "Runs while",
            "What it does",
            "What it needs",
        )
        self.assertEqual(
            [r[0] for r in rows],
            [G.cell(c.title) for c in features.CAPABILITIES.values()],
        )

    def test_profiles_their_values_and_the_command_that_writes_each(self):
        rows = table_rows(
            body("profiles"), "Profile", "What it runs", "Its values", "Write it with"
        )
        found = settings_file.profiles(DEPLOY)
        self.assertEqual([r[0] for r in rows], [f"`{p.name}`" for p in found])
        for p, row in zip(found, rows):
            with self.subTest(profile=p.name):
                self.assertEqual(row[1], G.cell(p.what))
                self.assertEqual(
                    row[3],
                    f"`rkmon template deploy --profile {p.name} FILE`",
                )
                if p.values:
                    self.assertEqual(
                        row[2].split("<br>"),
                        [f"`{k}: {json.dumps(v)}`" for k, v in p.values.items()],
                    )
                else:
                    self.assertEqual(row[2], "every key's default")

    def test_each_path_a_row_for_each_step_with_its_other_commands(self):
        for name, path in (
            ("build-steps", steps.BUILD_STEPS),
            ("deploy-steps", steps.DEPLOY_STEPS),
        ):
            with self.subTest(section=name):
                rows = table_rows(body(name), "#", "Step", "Command", "What it does")
                self.assertEqual([r[0] for r in rows], [str(s.n) for s in path])
                for s, row in zip(path, rows):
                    self.assertEqual(row[1], G.cell(s.name))
                    self.assertEqual(row[3], G.cell(s.what))
                    # the command, then the lines `rkmon help` puts after it (steps.aside), each command in code
                    shown = lambda c: G.code(f"rkmon {c}")  # noqa: E731
                    self.assertEqual(
                        row[2],
                        "<br>".join(
                            [shown(s.command), *(x for x in steps.aside(s, shown) if x)]
                        ),
                    )
        # Deploy step 4 in full: its other command, when it needs none, and the command that checks its work.
        rows = table_rows(body("deploy-steps"), "#", "Step", "Command", "What it does")
        self.assertEqual(
            rows[3][2],
            "`rkmon image load IMAGE_FILE`<br>or: `rkmon image pull`; nothing when the image is here"
            "<br>(`rkmon image check` checks it)",
        )

    def test_each_state_has_examples_each_of_them_that_state(self):
        examples = G._state_examples()
        self.assertEqual(sorted(examples), [s.n for s in states.TABLE])
        for n, changes in examples.items():
            for c in changes:
                with self.subTest(state=n, changes=c):
                    self.assertEqual(states.current(G._facts(**c)).n, n)
        known = {e.name for e in secrets_registry.entries({})}
        self.assertEqual(set(examples[7][0]["logins_missing"]) - known, set())

    def test_states_a_row_for_each_example_then_standby_and_a_refusal(self):
        text = body("states")
        rows = table_rows(text, "#", "State", "What it says", "Next line")
        examples = G._state_examples()
        want = []
        # The table's lines are past deploy step 8; the paragraph after it shows a step's words before it.
        ran = states.SinceStart(True, "2026-01-02 03:04 UTC")
        with G._as_rkmon():
            for s in states.TABLE:
                for i, c in enumerate(examples[s.n]):
                    f = G._facts(**c)
                    want.append(
                        [
                            "" if i else str(s.n),
                            "" if i else G.cell(s.name),
                            G.cell(states.sentence(s, f)),
                            G.code(states.next_line(s, f, ran)),
                        ]
                    )
            self.assertEqual(rows, want)
            out = io.StringIO()
            first = states.TABLE[0]
            # This folder's container cannot be asked while its name is another deployment's.
            states.refuse(
                first,
                G._facts(**examples[first.n][0]),
                out=out,
                since=states.NOT_STARTED,
            )
        self.assertIn(G.fenced(out.getvalue()), text)
        self.assertIn("next: step 3, confirm: rkmon confirm --settings", out.getvalue())
        self.assertIn(
            "Until deploy step 8 has passed after a start (a tick has run since a start: the first command that sees "
            "one keeps that in this folder's records, until `rkmon remove --delete-data` removes the deployment's "
            "volumes), "
            "a next command that is a deploy step's carries that step's number and name: in state 7, "
            "`next: step 5, login: rkmon login`. While the container runs, the line of states 14 and 16 is "
            "`next: step 7, doctor: rkmon doctor` until the doctor has passed since the start, then "
            "`next: step 8, first tick: rkmon first-tick` until a tick has run.",
            " ".join(text.split()),
        )
        self.assertIn("every other's ends \"; in standby: ", text)
        self.assertIn(
            f"A refusal exits {outcome.EARLIER_STEP} in states 2, 4, 5, 6, 7 and 8, where an earlier step is not "
            f"done, and {outcome.code('REFUSED')} in any other.",
            text,
        )

    def test_a_state_with_no_example_or_another_state_s_is_an_error(self):
        examples = G._state_examples()
        missing = {n: c for n, c in examples.items() if n != 16}
        with mock.patch.object(G, "_state_examples", lambda: missing):
            with self.assertRaisesRegex(ValueError, "state 16, .*, has no example"):
                body("states")
        wrong = {**examples, 12: [{}]}
        with mock.patch.object(G, "_state_examples", lambda: wrong):
            with self.assertRaisesRegex(
                ValueError, "an example of state 12 is state 16"
            ):
                body("states")

    def test_exit_codes_each_with_its_words(self):
        rows = table_rows(body("exit-codes"), "Exit code", "What it means")
        self.assertEqual(
            rows, [[str(c), G.cell(long)] for c, _short, long in outcome.EXIT_CODES]
        )

    def test_the_settings_references_print_each_description_as_it_is(self):
        """Each comment of the every-key text, its closing sentence included, in a row of its key: the deploy keys
        outside model_bindings, then model_bindings and each of its three forms, and the build keys, each with what
        the build is given from it; a section's row in bold."""
        deploy = settings_file.comments(DEPLOY)
        self.assertEqual(
            table_rows(body("deploy-settings"), "Key", "What it is"),
            [
                [f"**`{c.key}`**" if c.groups else f"`{c.key}`", G.cell(c.text)]
                for c in deploy
                if not G._under(c.key, "model_bindings")
            ],
        )
        given = G.given_to_the_build()
        self.assertEqual(
            table_rows(body("build-settings"), *BUILD_HEADER),
            [
                (
                    [f"**`{c.key}`**", "", G.cell(c.text)]
                    if c.groups
                    else [f"`{c.key}`", given.get(c.key, "—"), G.cell(c.text)]
                )
                for c in settings_file.comments(BUILD)
            ],
        )
        self.assertIn('Default: "rocke-monitor".', body("deploy-settings"))
        text = body("model-bindings")
        bindings = [c for c in deploy if G._under(c.key, "model_bindings")]
        self.assertTrue(text.startswith(G.cell(bindings[0].text) + "\n\n"))
        forms = [c for c in bindings[1:] if c.groups and c.key.count(".") == 1]
        self.assertEqual(len(forms), 3)
        self.assertEqual(
            text.count("| Key | What it is |"), len(forms), "a table for each form"
        )
        for c in bindings:
            with self.subTest(key=c.key):
                self.assertIn(G.cell(c.text), text)
        with mock.patch.object(settings_file, "comments", lambda ft: []):
            with self.assertRaisesRegex(ValueError, "has no model_bindings"):
                body("model-bindings")

    def test_quick_build_s_row_4_reads_the_build_settings_reference(self):
        """QUICK_BUILD_GUIDE.md's "Before you start" row 4 sends you to README.md, "Build settings", for which
        atlassian tool, where its binary comes from, and how to get its SHA-256: the build schema's
        atlassian_cli section says them, and the build-settings section, which README.md carries, prints them.
        """
        self.assertIn("README.md", G.SECTIONS["build-settings"].places)
        rows = {r[0]: r[-1] for r in table_rows(body("build-settings"), *BUILD_HEADER)}
        self.assertIn(
            "the atlassian tool, whose commands are", rows["**`atlassian_cli`**"]
        )
        self.assertIn("binary", rows["`atlassian_cli.package_url`"])
        self.assertIn("sha256sum FILE", rows["`atlassian_cli.package_sha256`"])

    def test_required_keys_the_placeholders_then_what_a_condition_needs(self):
        text = body("required-keys")
        wanted = [
            G._placeholder_keys(settings_file.template(DEPLOY, p.name))
            for p in settings_file.profiles(DEPLOY)
        ]
        self.assertEqual(
            len({tuple(w) for w in wanted}), 1, "one list for every profile"
        )
        self.assertTrue(
            text.startswith(
                "In each profile's template, these keys have a `<...>` value for you to fill in: "
                + ", ".join(f"`{k}`" for k in wanted[0][:-1])
                + f" and `{wanted[0][-1]}`."
            ),
            text,
        )
        rows = dict(table_rows(text, "Required when", "Keys"))
        self.assertEqual(
            rows["the develop guard is on and the merge queue is live"],
            "in `features.merge_queue.develop_guard`: `ci_red_label`, `stability_label`",
        )
        self.assertIn("`model_bindings.<name>.api_key_header`", text)
        # a key the templates give a value is no key to fill in
        self.assertNotIn("`features.review_automation.mode`", text)
        self.assertNotIn("`mode`", text)

    def test_a_template_s_placeholder_keys_in_the_file_s_order(self):
        text = 'a:\n  b: "<x>"\n  c: 1\nl: ["<y>"]\n"<name>":\n  d: "<z>"\n'
        self.assertEqual(G._placeholder_keys(text), ["a.b", "l"])

    def test_keys_in_a_cell_under_the_section_they_share(self):
        for keys, want in (
            (["a.b"], "`a.b`"),
            (["a.b.c", "a.b.d"], "in `a.b`: `c`, `d`"),
            (["a.b.c", "a.d"], "in `a`: `b.c`, `d`"),
            (["a.b", "c.d"], "`a.b`, `c.d`"),
        ):
            with self.subTest(keys=keys):
                self.assertEqual(G._keys_cell(keys), want)

    def test_logins_a_row_each_from_the_registry_then_the_permissions(self):
        sr = secrets_registry
        text = body("logins")
        rows = table_rows(text, "Type", "Name", "Kept for", "Installed by", "Needed by")
        found = sr.entries({})
        self.assertEqual([r[1] for r in rows], [f"`{e.name}`" for e in found])
        for e, row in zip(found, rows):
            with self.subTest(entry=e.name):
                self.assertEqual(row[0], "login" if e.shape == sr.LOGIN else "secret")
                self.assertEqual(row[2], f"the {e.keeper}")
                self.assertEqual(
                    row[3], G.code(str(Command(("rkmon", *sr.set_command(e)))))
                )
                needs = [sr.condition_words(c) for c in e.needed_when]
                self.assertEqual(
                    row[4],
                    G.cell("; ".join(needs) or "nothing: it is used when it is set"),
                )
        made = table_rows(
            text,
            "Login",
            "What to make",
            "Repository permissions",
            "Organization permissions",
        )
        with_permissions = [e for e in found if e.permissions]
        self.assertEqual(
            [r[0] for r in made], [f"`{e.name}`" for e in with_permissions]
        )
        for e, row in zip(with_permissions, made):
            with self.subTest(permissions=e.name):
                for permission, how in e.permissions:
                    # the repository's permissions, then the organization's, each in its own column
                    on, off = (
                        (3, 2) if permission in gh.ORGANIZATION_PERMISSIONS else (2, 3)
                    )
                    self.assertRegex(
                        row[on],
                        rf"(^|; )[^;]*\b{re.escape(permission)}\b[^;]*: {how}(;|$)",
                    )
                    self.assertNotRegex(row[off], rf"\b{re.escape(permission)}\b")
        self.assertIn(G.PUBLIC_ONLY, text)

    def test_env_and_ports_a_row_for_each_variable_of_the_plan_s_map(self):
        """docker/.env's variables in deploy_plan.ENV_KEYS' order, each with the keys it is made from: a key's default
        with what it means, the team dashboard's switch keys as the condition it is there under, and for a variable
        made from none, the value the plan writes. Another map makes other rows."""
        rows = table_rows(body("env-and-ports"), "Variable", "Made from")
        self.assertEqual([r[0] for r in rows], [f"`{n}`" for n in deploy_plan.ENV_KEYS])
        made = {r[0].strip("`"): r[1] for r in rows}
        self.assertEqual(made["COMPOSE_PROJECT_NAME"], "`container.name`")
        self.assertEqual(
            made["RKMON_ADMIN_HOST"],
            f"no setting: always `{container_layout.LOOPBACK}`",
        )
        self.assertEqual(
            made["RKMON_TEAM_PORT"],
            "`features.dashboards.team.serve.port`; only while the container serves the team dashboard (in "
            "`features.dashboards.team`: `enabled`, `mode`, `delivery`)",
        )
        self.assertEqual(
            made["RKMON_UID"],
            "`host.users.monitor_uid` (at its default: your own user id on this host)",
        )
        invented = {
            "RKMON_PORT": ("features.dashboards.admin.port",),
            "RKMON_CPUS": ("host.resources.cpus", "features.dashboards.team.delivery"),
        }
        with mock.patch.dict(deploy_plan.ENV_KEYS, invented, clear=True):
            rows = table_rows(body("env-and-ports"), "Variable", "Made from")
        self.assertEqual(
            rows,
            [
                ["`RKMON_PORT`", "`features.dashboards.admin.port`"],
                [
                    "`RKMON_CPUS`",
                    "`host.resources.cpus`; only while the container serves the team dashboard "
                    "(`features.dashboards.team.delivery`)",
                ],
            ],
        )
        unwritten = {"RKMON_PORT": invented["RKMON_PORT"], "RKMON_NEW": ()}
        with mock.patch.dict(deploy_plan.ENV_KEYS, unwritten, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "RKMON_NEW: made from no deploy setting, and the plan writes no value of its own",
            ):
                body("env-and-ports")

    def test_env_and_ports_the_override_the_plan_makes_while_the_team_dashboard_is_served(
        self,
    ):
        text = body("env-and-ports")
        example = G._served_example()
        self.assertTrue(deploy_plan.served(example.layer["settings"]))
        shown = [x for x in example.override.split("\n") if x and not x.startswith("#")]
        self.assertIn(G.fenced("\n".join(shown), "yaml"), text)
        for line in deploy_plan.OVERRIDE_HEADER:
            self.assertNotIn(line, text)
        self.assertIn(f"- target: {container_layout.TEAM_PORT}\n", text)
        # the example's RAM disk has a part of a GiB, which the plan rounds up
        self.assertNotEqual(G.EXAMPLE_RAMDISK_GIB % 1, 0)
        size = math.ceil(G.EXAMPLE_RAMDISK_GIB)
        self.assertIn(f"- {container_layout.RAMDISK}:size={size}g,", text)
        self.assertIn(
            "docker/compose.override.yaml is made from `host.resources.ramdisk_gib`, "
            "`features.dashboards.team.enabled`, `features.dashboards.team.mode` and "
            "`features.dashboards.team.delivery`. While the container serves the team dashboard, with "
            f"`host.resources.ramdisk_gib` at {G.EXAMPLE_RAMDISK_GIB:g}, it adds this to docker/compose.yaml:",
            text,
        )

    def test_env_and_ports_refuses_a_team_dashboard_it_cannot_serve(self):
        with mock.patch.object(deploy_plan, "served", lambda values: False):
            with self.assertRaisesRegex(
                ValueError, "no values of the team dashboard's switch keys serve it"
            ):
                body("env-and-ports")
        port = "features.dashboards.team.serve.port"
        with mock.patch.object(deploy_plan, "TEAM_SWITCH", (port,)):
            with self.assertRaisesRegex(
                ValueError, f"{re.escape(port)}, .* is neither a boolean nor a choice"
            ):
                body("env-and-ports")

    def test_takes_effect_a_row_for_each_class_and_the_keys_with_none(self):
        def words(text):
            return G.cell(text.replace("{program}", "rkmon"))

        # the keys read at their next use, in the schema's order: a section whose keys all are, by its name
        no_restart = [
            "monitor.full_tick_max_interval_minutes",
            "features.review_automation.schedule",
            "features.review_automation.parallel_reviews",
            "features.review_automation.large_prs",
            "features.review_automation.spend.enabled",
            "features.merge_queue.develop_guard.issue_repository",
            "features.merge_queue.develop_guard.ci_red_label",
            "features.merge_queue.develop_guard.ci_red_extra_labels",
            "features.merge_queue.develop_guard.stability_label",
            "features.weekly_report",
            "integrations.github_cli.max_requests_per_hour",
        ]
        rows = table_rows(body("takes-effect"), "You change", "It applies")
        self.assertEqual(
            rows,
            [
                [
                    ", ".join(f"`{k}`" for k in no_restart),
                    words(G.CLASS_WORDS["no_restart"]),
                ],
                ["anything else", words(G.CLASS_WORDS["new_container"])],
                ["`container.name`", words(G.NO_CLASS["container.name"])],
                ["`updates`", words(G.NO_CLASS["updates"])],
            ],
        )

    def test_takes_effect_rows_name_each_section_whose_keys_share_a_class(self):
        classes = {
            "a.x": "no_restart",
            "a.y": "no_restart",
            "b.x": "no_restart",
            "b.y": "new_container",
            "container.name": None,
            "updates.repository": None,
            "updates.branch": None,
        }
        with G._as_rkmon():
            rows = G.takes_effect_rows(classes, {"b.x": "b.<name>"})
        self.assertEqual(
            [r[0] for r in rows],
            ["`a`, `b.<name>`", "anything else", "`container.name`", "`updates`"],
        )
        self.assertEqual(
            rows[0][1],
            G.cell(G.CLASS_WORDS["no_restart"].replace("{program}", "rkmon")),
        )
        with G._as_rkmon():
            rows = G.takes_effect_rows({"a.x": "new_container"})
        self.assertEqual([r[0] for r in rows], ["anything else"])

    def test_takes_effect_refuses_a_class_or_a_key_it_cannot_place(self):
        for classes, words in (
            ({"a.x": "at_dawn"}, "x-applies at_dawn: no such class"),
            ({"a.x": None}, "no x-applies for a.x: give each its class"),
            ({"container.name": "no_restart"}, "container.name has a class"),
        ):
            with self.subTest(classes=classes):
                with self.assertRaisesRegex(ValueError, re.escape(words)):
                    G.takes_effect_rows(classes)

    def test_rkmon_commands_a_row_each_from_the_command_table(self):
        """Each row of the table, in its order: its form, the row's own or its name, required options and arguments,
        as the overview lists it; and its summary."""
        invented = (
            commands.Row(
                "pick",
                "pick {program}'s | best",
                args=(commands.Arg("NAME"),),
                options=(
                    commands.Option(("--from",), "where", metavar="DIR", required=True),
                    commands.Option(("--all",), "every one"),
                ),
            ),
            commands.Row("image ls", "list `images`", form="image ls [--all]"),
        )
        with mock.patch.object(commands, "TABLE", invented):
            text = body("rkmon-commands")
        self.assertEqual(
            text.split("\n")[0],
            "Each command's help, with its options and examples: `rkmon help COMMAND`.",
        )
        self.assertEqual(
            table_rows(text, "Command", "What it does"),
            [
                ["`rkmon pick --from DIR NAME`", "pick rkmon's \\| best"],
                ["`rkmon image ls [--all]`", "list \\`images\\`"],
            ],
        )
        # The package's own table: each command in commands.form's form, the one `rkmon help` lists.
        rows = table_rows(body("rkmon-commands"), "Command", "What it does")
        self.assertEqual(
            rows,
            [
                [G.code(f"rkmon {commands.form(r)}"), G.cell(program.named(r.summary))]
                for r in commands.TABLE
            ],
        )
        migrate = commands.find(("migrate",), commands.TABLE)
        self.assertIn(
            ["`rkmon migrate --settings FILE`", G.cell(program.named(migrate.summary))],
            rows,
        )

    def test_rk_commands_what_rk_help_prints(self):
        text = body("rk-commands")
        self.assertTrue(text.startswith("```text\n") and text.endswith("\n```"))
        inner = text[len("```text\n") : -len("\n```")]
        doc = ast.get_docstring(
            ast.parse((ROOT / "rkmon.py").read_text(encoding="utf-8")), clean=False
        )
        self.assertEqual(
            inner, "\n".join(line.rstrip() for line in doc.strip("\n").split("\n"))
        )
        p = subprocess.run(
            [sys.executable, "-B", str(ROOT / "rkmon.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(inner, p.stdout)

    OUTWARD = ("Command", "Sends to", "Gate", "Feature", "A stop refuses it")

    @staticmethod
    def title(*names, joiner="or"):
        return f" {joiner} ".join(features.CAPABILITIES[n].title for n in names)

    @staticmethod
    def choices(cmd: str, option: str) -> list:
        """The choices rk CMD's OPTION takes, as its help prints them."""
        import rkmon

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            with contextlib.suppress(SystemExit):
                rkmon.main([cmd, "--help"])
        return (
            re.search(rf"{option} \{{([^}}]*)\}}", out.getvalue()).group(1).split(",")
        )

    def test_outward_commands_a_row_each_that_sends_from_the_table(self):
        """Each row of rkmon_lib/outward.py's table that sends, in its order, and no other: where it sends, its gate
        (one rk gates at the send naming what goes out), its features (two, either one, an issue command's arguments
        picking one; none, each choice of the option that picks them), and whether a stop refuses it.
        """
        C = outward.Command
        invented = (
            C("pick", outward.GITHUB, outward.OWN, ("review-posting",)),
            C("probe", outward.NONE, outward.NONE, writer=outward.SUDO),
        )
        with mock.patch.object(outward, "TABLE", invented):
            rows = table_rows(body("outward-commands"), *self.OUTWARD)
        self.assertEqual(
            rows,
            [
                [
                    "`rk pick`",
                    "GitHub",
                    "the command's own, before each send",
                    self.title("review-posting"),
                    "yes",
                ]
            ],
        )
        rows = table_rows(body("outward-commands"), *self.OUTWARD)
        sends = [r for r in outward.TABLE if r.dest != outward.NONE]
        self.assertEqual([row[0] for row in rows], [f"`rk {r.name}`" for r in sends])
        found = {r.name: row[1:] for r, row in zip(sends, rows)}
        for r in sends:
            with self.subTest(cmd=r.name):
                self.assertEqual(found[r.name][0], G.cell(G.SENDS_TO[r.dest]))
                self.assertEqual(
                    found[r.name][1], G.cell(G.GATES[r.gate].format(what=r.what))
                )
                self.assertEqual(found[r.name][3].startswith("yes"), r.stops)
        self.assertEqual(found["disarm"][3], "no: it only takes auto-merge off")
        self.assertEqual(
            found["needs-you-teams"][3],
            "yes, but not with `--dry-run`, which sends nothing",
        )
        self.assertEqual(
            found["atlassian"][1:],
            [
                "at the send, before a JIRA write goes out, whoever asked for it",
                self.title("follow-up-tickets"),
                "yes, at its gate",
            ],
        )
        self.assertEqual(
            found["issue-file"][2],
            self.title(*outward.ISSUES) + " (its arguments pick one)",
        )
        self.assertEqual(
            found[outward.TEAMS_TEST][2], self.title("chat-alerts", "needs-you-alerts")
        )
        # rk notice's features by its --kind, each of the choices its parser takes
        kinds = self.choices("notice", "--kind")
        self.assertNotIn("fix", kinds)
        self.assertEqual(
            found["notice"][2],
            "<br>".join(
                ["by its `--kind`:"]
                + [
                    f"`{k}`: {self.title(*post.NOTICE_FEATURE[k], joiner='and')}"
                    for k in kinds
                ]
            ),
        )
        # rk teams-send's by its --type, the table's message column
        messages = {r.message: r.features for r in outward.TABLE if r.message}
        self.assertEqual(self.choices("teams-send", "--type"), list(messages))
        self.assertEqual(
            found["teams-send"][2],
            "<br>".join(
                ["by its `--type`:"]
                + [
                    f"`{m}`: {self.title(*f, joiner='and')}"
                    for m, f in messages.items()
                ]
            ),
        )

    def test_outward_commands_refuse_a_row_they_cannot_say(self):
        C = outward.Command
        for rows, words in (
            (
                (C("a", "pigeon", outward.OWN, ("review-posting",)),),
                "rk a: no words for its destination pigeon",
            ),
            (
                (C("a", outward.GITHUB, "later", ("review-posting",)),),
                "or its gate later",
            ),
            (
                (C("a", outward.GITHUB, outward.OWN),),
                "rk a sends, and its row has no feature",
            ),
            (
                (C("a", outward.GITHUB, outward.OWN, ("no-such",)),),
                "no feature no-such in rkmon_lib/features.py",
            ),
        ):
            with self.subTest(words=words), mock.patch.object(outward, "TABLE", rows):
                with self.assertRaisesRegex(ValueError, re.escape(words)):
                    body("outward-commands")
        with mock.patch.object(post, "NOTICE_CHOICES", ("incident", "nope")):
            with self.assertRaisesRegex(
                ValueError, re.escape("rk notice --kind nope sends for no feature")
            ):
                body("outward-commands")

    def test_agentic_placeholders_a_row_each_its_words_as_written(self):
        rows = table_rows(body("agentic-placeholders"), "Placeholder", "What it is")
        self.assertEqual(
            rows,
            [
                [f"`{name}`", what.replace("|", "\\|")]
                for name, (what, _get) in agentic.PLACEHOLDERS.items()
            ],
        )

    def member(self, root: pathlib.Path, file: str, **front) -> pathlib.Path:
        """A team member file in ROOT's team members folder, with FRONT as its front matter, then a body."""
        path = root / G.TEAM_MEMBERS / file
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(f"{k}: {v}\n" for k, v in front.items())
        path.write_text(
            f"---\n{lines}---\n\n## Team Member: not read\n", encoding="utf-8"
        )
        return path

    def test_dispatch_rows_a_row_each_from_its_team_member_s_front_matter(self):
        """The workspace dispatch table's rows: one for each team member file but the rows' own, with its role, name,
        file, context and domain as its front matter has them (a | escaped), sorted by role. A file whose front matter
        lacks one of them cannot make the rows."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.member(
                root, "rocke_b.md", role="Zeta", name="B", context="Any", domain="b, c"
            )
            self.member(
                root,
                "rocke_a.md",
                role="Alpha",
                name="A",
                context="Any",
                domain="a | *a*",
            )
            (root / G.DISPATCH_ROWS).write_text("rows\n", encoding="utf-8")
            header = ("Generic Role", "Team Member", "File", "Context", "Domain")
            self.assertEqual(
                table_rows(body("dispatch-rows", root), *header),
                [
                    ["Alpha", "A", "`rocke_a.md`", "Any", "a \\| *a*"],
                    ["Zeta", "B", "`rocke_b.md`", "Any", "b, c"],
                ],
            )
            self.member(root, "rocke_c.md", role="R", name="C", context="Any")
            with self.assertRaisesRegex(
                ValueError, "rocke_c.md: its front matter has no domain"
            ):
                body("dispatch-rows", root)

    def test_the_dispatch_rows_follow_a_changed_front_matter_after_the_refresh(self):
        """A team member's changed front matter puts the dispatch table's rows out of step, named with the refresh,
        which writes them again."""
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            member = self.member(
                root, "rocke_a.md", role="R", name="N", context="Any", domain="one"
            )
            rows = root / G.DISPATCH_ROWS
            rows.write_text(
                f"head\n\n{G.BEGIN} dispatch-rows -->\n{G.END} dispatch-rows -->\n",
                encoding="utf-8",
            )
            docs = (G.DISPATCH_ROWS,)
            self.assertEqual(G.refresh(root, files=(), docs=docs), [G.DISPATCH_ROWS])
            self.assertEqual(G.drift(root, files=(), docs=docs), [])
            text = member.read_text(encoding="utf-8")
            member.write_text(
                text.replace("domain: one", "domain: two"), encoding="utf-8"
            )
            source = G.SECTIONS["dispatch-rows"].source
            self.assertEqual(
                G.drift(root, files=(), docs=docs),
                [
                    f"{G.DISPATCH_ROWS}:3: the section dispatch-rows is out of step with its source ({source}): run "
                    f"{G.REFRESH}"
                ],
            )
            self.assertEqual(G.refresh(root, files=(), docs=docs), [G.DISPATCH_ROWS])
            self.assertIn(
                "\n| R | N | `rocke_a.md` | Any | two |\n",
                rows.read_text(encoding="utf-8"),
            )


class TestFiles(unittest.TestCase):
    def test_the_golden_texts_keep_their_version(self):
        """The refresh writes tests/settings_template_texts.py whole from the writer (tools/template_texts.py): over a
        copy with another VERSION, it keeps that VERSION, the deploy texts' image.tag; over a file with no VERSION, it
        writes nothing."""
        golden = next(f for f in G.FILES if f.path == G.GOLDEN)
        committed = (ROOT / G.GOLDEN).read_text(encoding="utf-8")
        version = re.search(r'^VERSION = "([^"]*)"$', committed, re.M).group(1)
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "tests").mkdir()
            path = root / G.GOLDEN
            path.write_text(
                committed.replace(
                    f'\nVERSION = "{version}"\n', '\nVERSION = "2030.01.02"\n'
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                G.drift(root, files=(golden,), docs=()),
                [
                    f"{G.GOLDEN}: out of step with its source ({golden.source}): run {G.REFRESH}"
                ],
            )
            self.assertEqual(G.refresh(root, files=(golden,), docs=()), [G.GOLDEN])
            written = path.read_text(encoding="utf-8")
            self.assertEqual(written.count('\n  tag: "2030.01.02"\n'), 4)
            self.assertEqual(G.drift(root, files=(golden,), docs=()), [])
            path.write_text("TEXTS = {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no VERSION line to keep"):
                G.refresh(root, files=(golden,), docs=())
            self.assertEqual(path.read_text(encoding="utf-8"), "TEXTS = {}\n")


def part(text: str, heading: str) -> str:
    """The part of a doc's TEXT from its `## HEADING` to the next heading of that level."""
    lines = text.split("\n")
    at = lines.index(f"## {heading}")
    end = next(
        (i for i in range(at + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    return "\n".join(lines[at:end])


class TestInternalSettings(unittest.TestCase):
    """docs/internal_settings.md, generated whole from the registry (rkmon_lib/registry.py) with the package's header:
    the package, state and derived keys, each owner's in a table of its own in the registry's order, each key with its
    description as the registry has them (a state key with its key in the state file, a derived key with its default:
    a package or state key's description closes with its own; an entry of a key's value the registry describes right
    after the key), then each section with one of them in it, by name.
    """

    OWNERS = (
        (registry.PACKAGE, "Package defaults", ("Key", "What it is")),
        (registry.STATE, "Internal state", ("Key", "In the state file", "What it is")),
        (registry.DERIVED, "Derived", ("Key", "Default", "What it is")),
    )

    def setUp(self):
        self.file = next(f for f in G.FILES if f.path == G.INTERNAL_SETTINGS)

    def text(self) -> str:
        return self.file.render(ROOT)

    def test_the_package_s_header_and_the_registry_as_its_source(self):
        text = self.text()
        self.assertEqual(self.file.source, "rkmon_lib/registry.py")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "internal_settings.md"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(make_package.header_problem(path), "")
        self.assertIn(G.REFRESH, text)
        self.assertEqual(G.kind_of(G.INTERNAL_SETTINGS), G.MAINTAINER)

    def test_each_internal_key_once_in_its_owner_s_table(self):
        text = self.text()
        for owner, heading, header in self.OWNERS:
            with self.subTest(owner=owner):
                # the key rows: a described entry of a key's value follows its key (the entries' own test)
                rows = [
                    r
                    for r in table_rows(part(text, heading), *header)
                    if registry.kind(r[0].strip("`")) != registry.ENTRY
                ]
                mine = [k for k in registry.keys() if k.owner == owner]
                self.assertTrue(mine)
                self.assertEqual([r[0] for r in rows], [G.code(k.name) for k in mine])
                for k, row in zip(mine, rows):
                    self.assertEqual(row[-1], G.cell(k.description), k.name)
                    if owner == registry.STATE:
                        self.assertEqual(
                            row[1], "<br>".join(G.code(f) for f in k.file_keys)
                        )
        named = set(re.findall(r"^\| `([^`]+)` \|", text, re.M))
        deploy = {k.name for k in registry.keys() if k.owner == registry.DEPLOY}
        self.assertEqual(named & deploy, set(), "a deploy key is in a settings file")

    def test_a_default_as_json_cut_when_long_and_none_for_a_key_without_one(self):
        """A derived key's default, beside its description, which closes with its rule."""
        _owner, heading, header = self.OWNERS[-1]
        rows = {r[0]: r[-2] for r in table_rows(part(self.text(), heading), *header)}
        for k in registry.keys():
            if k.owner != registry.DERIVED:
                continue
            with self.subTest(key=k.name):
                shown = rows[G.code(k.name)]
                if k.default is registry.MISSING:
                    self.assertEqual(shown, "")
                    continue
                whole = json.dumps(k.default, ensure_ascii=False)
                if len(whole) <= G.SHOWN_MAX:
                    self.assertEqual(shown, G.code(whole))
                else:
                    self.assertEqual(shown, G.code(whole[: G.SHOWN_MAX - 3] + "..."))
        self.assertTrue(
            any(
                registry.key(n).default is registry.MISSING
                for n in registry.DERIVED_NOT_IN_FILES
            )
        )

    def test_a_described_entry_follows_its_key(self):
        described = dict(
            registry.descriptions(), **{"jira.priority_map.P1": "An invented entry."}
        )
        with mock.patch.object(registry, "descriptions", return_value=described):
            rows = table_rows(
                part(self.text(), "Package defaults"), "Key", "What it is"
            )
        names = [r[0] for r in rows]
        at = names.index(G.code("jira.priority_map"))
        self.assertEqual(
            rows[at + 1], [G.code("jira.priority_map.P1"), "An invented entry."]
        )

    def test_each_section_with_an_internal_key_in_it_once_by_name(self):
        rows = table_rows(part(self.text(), "Sections"), "Section", "What it is")
        names = [r[0].strip("`") for r in rows]
        self.assertEqual(names, sorted(set(names)))
        descriptions = registry.descriptions()
        internal = [k.name for k in registry.keys() if k.owner != registry.DEPLOY]
        for name, row in zip(names, rows):
            self.assertEqual(registry.kind(name), registry.SECTION, name)
            self.assertTrue(any(n.startswith(name + ".") for n in internal), name)
            self.assertEqual(row[1], G.cell(descriptions[name]))
        for n in internal:
            parts = n.split(".")
            for above in (".".join(parts[:i]) for i in range(1, len(parts))):
                if above in descriptions:
                    self.assertIn(above, names, n)

    def test_a_changed_registry_is_out_of_step_until_the_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            files = (self.file,)
            self.assertEqual(
                G.refresh(root, files=files, docs=()), [G.INTERNAL_SETTINGS]
            )
            self.assertEqual(G.drift(root, files=files, docs=()), [])
            rows = tuple(
                (
                    k._replace(description="An invented description.")
                    if k.name == "draft_parallel_max"
                    else k
                )
                for k in registry.keys()
            )
            with mock.patch.object(registry, "keys", return_value=rows):
                self.assertEqual(
                    G.drift(root, files=files, docs=()),
                    [
                        f"{G.INTERNAL_SETTINGS}: out of step with its source (rkmon_lib/registry.py): run "
                        f"{G.REFRESH}"
                    ],
                )
                self.assertEqual(
                    G.refresh(root, files=files, docs=()), [G.INTERNAL_SETTINGS]
                )
                self.assertIn(
                    "| `draft_parallel_max` | ",
                    (root / G.INTERNAL_SETTINGS).read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "| An invented description. |",
                    (root / G.INTERNAL_SETTINGS).read_text(encoding="utf-8"),
                )
                self.assertEqual(G.refresh(root, files=files, docs=()), [])


class TestDriftAndRefresh(unittest.TestCase):
    """drift and refresh over a tree of their own: a doc with a section, a generated file and VERSION. The generator's
    sections are exit-codes alone here, so the doc lacks none of its place's."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        only = mock.patch.dict(
            G.SECTIONS, {"exit-codes": G.SECTIONS["exit-codes"]}, clear=True
        )
        only.start()
        self.addCleanup(only.stop)
        self.readme = self.root / "README.md"
        self.old = "intro\n\n<!-- BEGIN GENERATED: exit-codes -->\nold\n<!-- END GENERATED: exit-codes -->\n\nafter\n"
        self.readme.write_text(self.old, encoding="utf-8")
        self.made = G.File("made.txt", "an invented source", lambda root: "made\n")
        self.version = next(f for f in G.FILES if f.path == "VERSION")
        self.files = (self.made, self.version)
        self.release = make_package.version_text(dt.date(2030, 1, 2))
        (self.root / "VERSION").write_text(self.release, encoding="utf-8")

    def drift(self, docs=("README.md",)):
        return G.drift(self.root, files=self.files, docs=docs)

    def refresh(self, docs=("README.md",)):
        return G.refresh(self.root, files=self.files, docs=docs)

    def test_out_of_step_parts_are_named_with_the_refresh_which_mends_them_once(self):
        source = G.SECTIONS["exit-codes"].source
        self.assertEqual(
            self.drift(),
            [
                f"README.md:3: the section exit-codes is out of step with its source ({source}): run {G.REFRESH}",
                f"made.txt: missing: run {G.REFRESH}",
            ],
        )
        self.assertEqual(self.refresh(), ["README.md", "made.txt"])
        text = self.readme.read_text(encoding="utf-8")
        self.assertEqual(
            text,
            "intro\n\n"
            + G.section_text(G.SECTIONS["exit-codes"], self.root, "README.md")
            + "\n\nafter\n",
        )
        self.assertEqual((self.root / "made.txt").read_text(encoding="utf-8"), "made\n")
        self.assertEqual(self.drift(), [])
        # a second refresh changes nothing
        self.assertEqual(self.refresh(), [])
        self.assertEqual(self.readme.read_text(encoding="utf-8"), text)
        self.assertEqual(
            (self.root / "VERSION").read_text(encoding="utf-8"), self.release
        )

    def test_the_refresh_writes_and_names_only_the_paths_whose_text_differs(self):
        (self.root / "notes.md").write_text("no section here\n", encoding="utf-8")
        docs = ("README.md", "notes.md")
        self.refresh(docs=docs)
        self.readme.write_text(self.old, encoding="utf-8")
        then = 1_000_000_000 * 10**9  # in 2001: no write of this run has that time
        for p in self.root.iterdir():
            os.utime(p, ns=(then, then))
        self.assertEqual(self.refresh(docs=docs), ["README.md"])
        self.assertEqual(
            {p.name: p.stat().st_mtime_ns == then for p in self.root.iterdir()},
            {"README.md": False, "made.txt": True, "notes.md": True, "VERSION": True},
        )

    def test_a_changed_source_is_out_of_step_until_the_refresh(self):
        self.refresh()
        invented = (*outcome.EXIT_CODES, (99, "invented", "an invented code"))
        with mock.patch.object(outcome, "EXIT_CODES", invented):
            problems = self.drift()
            self.assertEqual(len(problems), 1)
            self.assertIn(
                "the section exit-codes is out of step with its source", problems[0]
            )
            self.assertIn(G.REFRESH, problems[0])
            self.assertEqual(self.refresh(), ["README.md"])
            self.assertIn(
                "| 99 | an invented code |", self.readme.read_text(encoding="utf-8")
            )
            self.assertEqual(self.drift(), [])
        self.assertEqual(len(self.drift()), 1)
        self.assertEqual(
            (self.root / "VERSION").read_text(encoding="utf-8"), self.release
        )

    def test_a_section_out_of_its_place_is_named_and_stops_the_refresh(self):
        self.refresh()
        guide = self.root / "QUICK_USER_GUIDE.md"
        guide.write_text(self.readme.read_text(encoding="utf-8"), encoding="utf-8")
        misplaced = (
            "QUICK_USER_GUIDE.md:3: the section exit-codes goes in README.md, not here"
        )
        docs = ("README.md", "QUICK_USER_GUIDE.md")
        self.assertEqual(self.drift(docs=docs), [misplaced])
        # both docs out of step: the refresh fills in neither, nor anything else
        self.readme.write_text(self.old, encoding="utf-8")
        guide.write_text(self.old, encoding="utf-8")
        (self.root / "made.txt").unlink()
        with self.assertRaises(ValueError) as caught:
            self.refresh(docs=docs)
        self.assertEqual(
            str(caught.exception),
            f"nothing written: the refresh stops on these\n  {misplaced}",
        )
        self.assertEqual(self.readme.read_text(encoding="utf-8"), self.old)
        self.assertEqual(guide.read_text(encoding="utf-8"), self.old)
        self.assertFalse((self.root / "made.txt").exists())

    def test_a_section_its_place_lacks_is_named_with_its_markers_and_never_added(self):
        """A doc that is a section's place and holds no block of it is named, with the markers to add, once its
        markers are right; the refresh writes the rest and adds no section, and once the markers are in, it fills the
        section in."""
        invented = G.Section(
            "invented", "an invented source", ("README.md",), lambda root, place: "x"
        )
        missing = (
            "README.md: the section invented is missing: add <!-- BEGIN GENERATED: invented --> and "
            f"<!-- END GENERATED: invented --> where it goes, then run {G.REFRESH}"
        )
        with mock.patch.dict(G.SECTIONS, {"invented": invented}):
            self.assertEqual(
                self.drift()[1:], [missing, f"made.txt: missing: run {G.REFRESH}"]
            )
            self.assertEqual(self.refresh(), ["README.md", "made.txt"])
            self.assertEqual(self.drift(), [missing])
            # a doc that is not among those read, or whose markers are wrong, is not named for it
            self.assertEqual(self.drift(docs=()), [])
            text = self.readme.read_text(encoding="utf-8")
            end = "<!-- END GENERATED: exit-codes -->"
            self.readme.write_text(text + end + "\n", encoding="utf-8")
            self.assertEqual(
                self.drift(),
                [f"README.md:{text.count(chr(10)) + 1}: an END with no BEGIN: {end}"],
            )
            self.readme.write_text(
                text
                + "\n<!-- BEGIN GENERATED: invented -->\n<!-- END GENERATED: invented -->\n",
                encoding="utf-8",
            )
            self.assertEqual(len(self.drift()), 1)
            self.assertIn("the section invented is out of step", self.drift()[0])
            self.assertEqual(self.refresh(), ["README.md"])
            self.assertEqual(self.drift(), [])
            self.assertTrue(
                self.readme.read_text(encoding="utf-8").endswith(
                    G.section_text(invented, self.root, "README.md") + "\n"
                )
            )

    def test_a_marker_problem_or_an_unmade_text_stops_the_refresh_before_it_writes(
        self,
    ):
        (self.root / "a.md").write_text(
            "<!-- BEGIN GENERATED: no-such -->\n<!-- END GENERATED: no-such -->\n",
            encoding="utf-8",
        )
        (self.root / "b.md").write_text(
            "x\n<!-- BEGIN GENERATED: exit-codes -->\n", encoding="utf-8"
        )
        docs = ("README.md", "a.md", "b.md")
        problems = self.drift(docs)
        self.assertTrue(
            problems[0].startswith("README.md:3: the section exit-codes is out of step")
        )
        self.assertTrue(
            problems[1].startswith("a.md:1: no section no-such in tools/generated.py")
        )
        self.assertEqual(problems[2], "b.md:2: the section exit-codes has no END")
        before = {p: p.read_text(encoding="utf-8") for p in self.root.iterdir()}
        with self.assertRaises(ValueError) as caught:
            self.refresh(docs=docs)
        self.assertIn("nothing written", str(caught.exception))
        self.assertIn("a.md:1: no section no-such", str(caught.exception))
        self.assertIn(
            "b.md:2: the section exit-codes has no END", str(caught.exception)
        )
        self.assertEqual(
            {p: p.read_text(encoding="utf-8") for p in self.root.iterdir()}, before
        )

        def cannot(root):
            raise ValueError("no way")

        self.files = (G.File("made.txt", "an invented source", cannot), self.version)
        (self.root / "made.txt").write_text("made\n", encoding="utf-8")
        self.assertIn(
            "made.txt: cannot be made from its source (an invented source): no way",
            self.drift(),
        )
        with self.assertRaisesRegex(ValueError, "made.txt: cannot be made"):
            self.refresh()
        self.assertEqual(self.readme.read_text(encoding="utf-8"), before[self.readme])

    def test_version_s_form_is_checked_and_the_refresh_never_writes_it(self):
        self.refresh()
        self.assertEqual(self.drift(), [])
        wrong = [
            f"VERSION: not the header and a YYYY.MM.DD line: a release writes it: {G.RELEASE}"
        ]
        path = self.root / "VERSION"
        path.write_text("junk\n", encoding="utf-8")
        self.assertEqual(self.drift(), wrong)
        self.assertEqual(self.refresh(), [])
        self.assertEqual(path.read_text(encoding="utf-8"), "junk\n")
        path.unlink()
        self.assertEqual(self.drift(), wrong)
        self.assertEqual(self.refresh(), [])
        self.assertFalse(path.exists())
        for text in (
            self.release.replace("2030.01.02", "2030.02.30"),
            self.release.replace("# SPDX", "# spdx"),
            self.release + "\n",
        ):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(self.drift(), wrong)


if __name__ == "__main__":
    unittest.main()
