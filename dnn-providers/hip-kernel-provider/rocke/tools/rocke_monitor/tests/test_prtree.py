# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon_lib/prtree.py: a PR's changes with git's rename detection, the paths a commit has, a commit's files written to
a scratch directory, and rocKE's module names; the folder a check looks at. A temp git repo, no network.
"""

import argparse
import contextlib
import importlib.util
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import config, paths, prtree  # noqa: E402


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestFolder(unittest.TestCase):
    # The checks whose blank --rocke-path means a folder of none, each with what it means.
    NO_FOLDER = {
        "tidy_check.py": "every file the tree's HEAD adds or changes, from the repository's root",
        "since_review.py": "the files shown from the repository's root",
    }

    def test_a_check_s_folder_is_the_one_it_is_given_else_the_configuration_s(self):
        """--rocke-path's type: the folder given, with no slash at either end; without one, the folder of the
        running configuration (paths.repository_folder), and a refusal that says why when there is none.
        """
        self.assertEqual(paths.repository_folder({"rocke_path": "/lib/k/"}), "lib/k")
        self.assertEqual(paths.repository_folder({}), "")
        self.assertEqual(prtree.folder_argument("/lib/k/"), "lib/k")
        with mock.patch(
            "rkmon_lib.store.load_config", return_value={"rocke_path": "/src/proj/"}
        ):
            self.assertEqual(prtree.folder_argument(""), "src/proj")
        with mock.patch("rkmon_lib.store.load_config", return_value={"rocke_path": ""}):
            with self.assertRaisesRegex(
                argparse.ArgumentTypeError,
                r"^no folder given, and the configuration names none \(repository\.folder\)$",
            ):
                prtree.folder_argument("")
        with mock.patch(
            "rkmon_lib.store.load_config",
            side_effect=config.NoConfiguration("no configuration at /x/config.json"),
        ):
            with self.assertRaisesRegex(
                argparse.ArgumentTypeError,
                r"^no folder given, and the configuration cannot be read: no configuration at /x/config\.json$",
            ):
                prtree.folder_argument("")

    def test_a_check_without_a_folder_takes_the_configuration_s(self):
        """A check run without --rocke-path looks at the configuration's folder: no check keeps one of its own."""
        spec = importlib.util.spec_from_file_location(
            "doc_refs_tool", ROOT / "tools" / "doc_refs.py"
        )
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        seen = []

        def check(repo, head, K, base, budget):
            seen.append(K)
            return {}

        with mock.patch(
            "rkmon_lib.store.load_config", return_value={"rocke_path": "src/proj/"}
        ), mock.patch.object(tool, "check", check), mock.patch.object(
            tool, "render", lambda res: []
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            tool.main(["--repo", "r", "--head", "h"])
            tool.main(["--repo", "r", "--head", "h", "--rocke-path", "lib/k"])
        self.assertEqual(seen, ["src/proj", "lib/k"])

    def test_every_check_asks_the_configuration_for_its_folder(self):
        """Every check that takes --rocke-path takes its folder through prtree.folder_argument, but those whose blank
        folder means one of none (NO_FOLDER)."""
        takers = sorted(
            f.name
            for f in (ROOT / "tools").glob("*.py")
            if '"--rocke-path"' in f.read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(takers), 9, takers)
        asks = [
            name
            for name in takers
            if "prtree.folder_argument"
            in (ROOT / "tools" / name).read_text(encoding="utf-8")
        ]
        self.assertEqual(sorted(set(takers) - set(asks)), sorted(self.NO_FOLDER))


class TestModuleName(unittest.TestCase):
    def test_the_import_roots_and_packages(self):
        self.assertEqual(
            prtree.module_name("rk/platform/python/rocke/core/backend.py", "rk"),
            "rocke.core.backend",
        )
        self.assertEqual(
            prtree.module_name("rk/library/kernels/conv/__init__.py", "rk"),
            "kernels.conv",
        )
        self.assertEqual(
            prtree.module_name("rk/library/kernels/conv/engine_bridge.py", "rk"),
            "kernels.conv.engine_bridge",
        )
        self.assertIsNone(
            prtree.module_name("rk/platform/tools/gen.py", "rk")
        )  # not under an import root
        self.assertIsNone(
            prtree.module_name("rk/library/gfx950-tools/run.py", "rk")
        )  # no identifier
        self.assertIsNone(prtree.module_name("rk/library/kernels/README.md", "rk"))


class TestTrees(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        r = self.repo = pathlib.Path(self.d.name) / "repo"
        (r / "rk/a").mkdir(parents=True)
        g(r, "init", "-q", "-b", "develop")
        for name, text in {
            "rk/a/keep.py": "x = 1\n",
            "rk/a/gone.py": "y = 2\n",
            "rk/a/moved.md": "".join(f"line {i}\n" for i in range(40)),
        }.items():
            (r / name).write_text(text)
        g(r, "add", "-A")
        g(r, "commit", "-qm", "base")
        self.base = g(r, "rev-parse", "HEAD")
        g(r, "checkout", "-qb", "pr")
        (r / "rk/a/keep.py").write_text("x = 3\n")
        (r / "rk/a/gone.py").unlink()
        (r / "rk/b").mkdir()
        (r / "rk/a/moved.md").rename(r / "rk/b/moved.md")
        (r / "rk/b/new.py").write_text("z = 1\n")
        g(r, "add", "-A")
        g(r, "commit", "-qm", "pr")
        self.head = g(r, "rev-parse", "HEAD")

    def test_changes_read_adds_changes_deletes_and_moves(self):
        got = sorted(
            (c["status"], c["old"], c["new"])
            for c in prtree.changes(str(self.repo), self.base, self.head)
        )
        self.assertEqual(
            got,
            [
                ("A", None, "rk/b/new.py"),
                ("D", "rk/a/gone.py", None),
                ("M", "rk/a/keep.py", "rk/a/keep.py"),
                ("R", "rk/a/moved.md", "rk/b/moved.md"),
            ],
        )
        self.assertEqual(
            prtree.merge_base(str(self.repo), "develop", self.head), self.base
        )
        with self.assertRaises(prtree.NotRun):
            prtree.merge_base(str(self.repo), "develop", "f" * 40)

    def test_present_and_extract(self):
        repo = str(self.repo)
        self.assertEqual(
            prtree.present(repo, self.head, ["rk/b", "rk/a/gone.py", "rk/a/keep.py"]),
            ["rk/b", "rk/a/keep.py"],
        )
        into = pathlib.Path(self.d.name) / "x"
        prtree.extract(repo, self.base, ["rk/a"], into)
        self.assertEqual(
            sorted(p.name for p in (into / "rk/a").iterdir()),
            ["gone.py", "keep.py", "moved.md"],
        )
        self.assertEqual((into / "rk/a/keep.py").read_text(), "x = 1\n")
        self.assertFalse((into / ".rkmon_archive.tar").exists())
        with self.assertRaises(RuntimeError):
            prtree.extract(repo, self.base, ["rk/nothing"], into)


if __name__ == "__main__":
    unittest.main()
