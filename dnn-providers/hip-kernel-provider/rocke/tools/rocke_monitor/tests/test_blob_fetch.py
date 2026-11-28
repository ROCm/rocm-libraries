# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The blobs a check reads of the partial clone, fetched before the check runs in a PR runner's slot, where git fetches
nothing: rkmon_lib/prtree.py's reads and fetch, and each check's `--fetch` (tools/ctest_labels.py,
tools/import_probe.py, tools/mutation_sweep.py, tools/lane_of.py), after which what the check reads there is in the
clone. A blobless clone of a small local repository, no network."""

import contextlib
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import ctest_labels as cl  # noqa: E402
import import_probe as ip  # noqa: E402
import lane_of as lo  # noqa: E402
import mutation_sweep as ms  # noqa: E402
from rkmon_lib import gitops, prtree  # noqa: E402

K = "a/b/rocke"
CORE = f"{K}/platform/python/rocke/core.py"
TEST = f"{K}/platform/tests/test_core.py"
OLD, NEW = f"{K}/library/kernels/old.py", f"{K}/library/kernels/new.py"
X, Y = "tools/x.py", "tools/y.py"  # a move outside the rocKE tree
ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, **ENV),
    ).stdout.strip()


def write(root: pathlib.Path, files: dict) -> None:
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


def blobless(base: pathlib.Path) -> dict:
    """An origin with rocKE at K, under its provider's tree, and a PR branch of two commits (`since`, then `head`) on
    develop's first commit (`mb`), develop a commit further on; and a blobless clone of it, which holds no blob. The
    head moves a file in the rocKE tree (OLD to NEW) and one outside it (X to Y), each with a line changed.
    """
    origin = base / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "develop")
    lines = "".join(f"V{i} = {i}\n" for i in range(30))
    write(
        origin,
        {
            "a/b/CMakeLists.txt": "project(p)\n",
            "a/b/src/CMakeLists.txt": "# the provider's install\n",
            "a/b/tiers.yaml": "quick: []\n",
            "shared/ctest/TestCategories.cmake": "# the categories\n",
            "other/z.txt": "z\n",
            CORE: "def f(x):\n    return x\n",
            OLD: lines,
            TEST: "def test_f():\n    pass\n",
            X: lines.replace("V", "W"),
        },
    )
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "develop: first")
    mb = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-qb", "pr7")
    guard = "    if x is None:\n        raise ValueError('no x')\n"
    write(origin, {CORE: f"def f(x):\n{guard}    return x\n"})
    git(origin, "commit", "-qam", "pr: a check")
    since = git(origin, "rev-parse", "HEAD")
    write(
        origin,
        {
            CORE: f"def f(x):\n{guard}    if x < 0:\n        raise ValueError('negative')\n    return x\n",
            TEST: "from rocke.core import f\n\n\ndef test_f():\n    assert f(1) == 1\n",
            NEW: lines.replace("V29 = 29", "V29 = 30"),
            Y: lines.replace("V", "W").replace("W0 = 0", "W0 = 1"),
        },
    )
    (origin / OLD).unlink()
    (origin / X).unlink()
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "pr: another check, and a move")
    head = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-q", "develop")
    write(origin, {"other/z.txt": "z2\n"})
    git(origin, "commit", "-qam", "develop: second")
    git(origin, "config", "uploadpack.allowFilter", "true")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    clone = base / "clone"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            origin.as_uri(),
            str(clone),
        ],
        check=True,
        capture_output=True,
    )
    return {
        "origin": origin,
        "clone": str(clone),
        "mb": mb,
        "since": since,
        "head": head,
    }


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)
        self.r = blobless(self.d)
        self.repo = self.r["clone"]

    def off(self):
        """Lazy fetches off, as the launcher sets them in a PR runner's slot."""
        return mock.patch.dict(os.environ, {"GIT_NO_LAZY_FETCH": "1"})

    def id_at(self, rev: str, path: str) -> str:
        return git(self.r["origin"], "rev-parse", f"{rev}:{path}")

    def main(self, tool, *argv) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tool.main(list(argv))
        return code, out.getvalue()

    def fetch(self, tool, *argv) -> tuple[int, str]:
        """The tool's `--fetch` on the clone, as the monitor runs it: its exit code and what it printed."""
        return self.main(tool, "--repo", self.repo, "--rocke-path", K, *argv, "--fetch")


class TestReadsAndFetch(Base):
    def test_the_blobs_of_a_tree_and_of_a_diff_read_from_the_trees_alone(self):
        r = self.r
        self.assertEqual(
            sorted(prtree.blobs(self.repo, r["head"], [f"{K}/platform"])),
            sorted([self.id_at(r["head"], CORE), self.id_at(r["head"], TEST)]),
        )
        self.assertEqual(prtree.blobs(self.repo, r["head"], []), [])
        both = {
            self.id_at(rev, p) for rev in (r["mb"], r["head"]) for p in (CORE, TEST)
        }
        both |= {self.id_at(r["mb"], OLD), self.id_at(r["head"], NEW)}
        self.assertEqual(
            set(prtree.diff_blobs(self.repo, r["mb"], r["head"], [K])), both
        )
        # the whole tree: the move outside the rocKE tree too; develop's later change is no change of the PR's
        self.assertEqual(
            set(prtree.diff_blobs(self.repo, r["mb"], r["head"])),
            both | {self.id_at(r["mb"], X), self.id_at(r["head"], Y)},
        )
        self.assertEqual(
            set(prtree.diff_blobs(self.repo, r["mb"], r["head"], [f"{K}/library"])),
            {self.id_at(r["mb"], OLD), self.id_at(r["head"], NEW)},
        )
        # none of them was fetched to tell
        self.assertEqual(set(prtree.missing(self.repo, both)), both)

    def test_an_archive_reads_the_blobs_of_its_paths_alone(self):
        r = self.r
        prtree.extract(self.repo, r["head"], [K], self.d / "x")  # lazy fetches on
        self.assertEqual(prtree.missing_blobs(self.repo, r["head"], [K]), [])
        outside = prtree.blobs(
            self.repo, r["head"], ["a/b/CMakeLists.txt", "other", "shared"]
        )
        self.assertEqual(sorted(prtree.missing(self.repo, outside)), sorted(outside))

    def test_fetch_takes_the_missing_blobs_none_where_lazy_fetches_are_off(self):
        r = self.r
        ids = prtree.blobs(self.repo, r["head"], [K])
        with self.off():
            self.assertEqual(sorted(prtree.fetch(self.repo, ids)), sorted(set(ids)))
            with self.assertRaises(RuntimeError):  # the archive fetches nothing either
                prtree.extract(self.repo, r["head"], [K], self.d / "x")
        self.assertEqual(sorted(prtree.missing(self.repo, ids)), sorted(set(ids)))
        self.assertEqual(prtree.fetch(self.repo, ids), [])
        self.assertEqual(prtree.missing(self.repo, ids), [])
        with self.off():  # what is in the clone, it reads
            prtree.extract(self.repo, r["head"], [K], self.d / "y")
        self.assertTrue((self.d / "y" / NEW).is_file())

    def test_a_fetch_git_cannot_make_is_an_error(self):
        ids = prtree.blobs(self.repo, self.r["head"], [K])
        git(self.repo, "remote", "set-url", "origin", (self.d / "gone").as_uri())
        with self.assertRaises(gitops.GitError):
            prtree.fetch(self.repo, ids)
        code, said = self.main_fetch(lambda: ids)
        self.assertEqual(code, 1)
        self.assertTrue(
            said.startswith(
                "fetch: the blobs this check reads were not fetched: "
                f"git fetch of {len(set(ids))} blob(s): "
            ),
            said,
        )
        # a clone git cannot read: an error, never "nothing missing"
        with self.assertRaises(gitops.GitError):
            prtree.missing(str(self.d / "none"), ids)

    def test_lazy_fetches_are_off_as_git_reads_its_switch(self):
        for value, off in (
            ("", False),
            ("0", False),
            ("false", False),
            ("No", False),
            ("off", False),
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            ("2", True),
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    gitops.lazy_fetch_off({"GIT_NO_LAZY_FETCH": value}), off
                )
        self.assertFalse(gitops.lazy_fetch_off({}))

    def test_a_checks_fetch_says_what_the_clone_lacks(self):
        ids = prtree.blobs(self.repo, self.r["head"], [K])

        def gone():
            raise prtree.NotRun("the head is not in the mirror")

        with self.off():
            got = self.main_fetch(lambda: ids)
        self.assertEqual(
            got,
            (
                1,
                f"fetch: {len(set(ids))} blob(s) this check reads are not in the clone "
                "(lazy fetches are off here)\n",
            ),
        )
        self.assertEqual(self.main_fetch(lambda: ids), (0, ""))
        self.assertEqual(
            self.main_fetch(gone),
            (
                1,
                "fetch: the blobs this check reads were not fetched: the head is not in the mirror\n",
            ),
        )

    def main_fetch(self, need) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = prtree.fetch_main(self.repo, need)
        return code, out.getvalue()


class TestEachCheck(Base):
    """Each check's `--fetch`, as the monitor runs it, then the check's reads with lazy fetches off, as in its slot."""

    def test_the_labels_read_both_trees(self):
        r = self.r
        trees = [
            (rev, cl.tree_paths(self.repo, rev, K)) for rev in (r["mb"], r["head"])
        ]
        with self.off(), self.assertRaises(RuntimeError):
            prtree.extract(self.repo, r["head"], trees[1][1], self.d / "before")
        self.assertEqual(self.fetch(cl, "--head", r["head"]), (0, ""))
        with self.off():
            for i, (rev, paths) in enumerate(trees):
                prtree.extract(self.repo, rev, paths, self.d / f"t{i}")
        self.assertTrue((self.d / "t1" / "shared/ctest/TestCategories.cmake").is_file())
        self.assertTrue((self.d / "t0" / "a/b/tiers.yaml").is_file())

    def test_the_import_probe_reads_the_changes_and_both_trees(self):
        r = self.r
        with self.off(), self.assertRaises(gitops.GitError):
            prtree.changes(self.repo, r["mb"], r["head"])
        self.assertEqual(self.fetch(ip, "--head", r["head"]), (0, ""))
        with self.off():
            got = {
                (c["status"], c["old"], c["new"])
                for c in prtree.changes(self.repo, r["mb"], r["head"])
            }
            for rev in (r["head"], r["mb"]):
                prtree.extract(self.repo, rev, [K], self.d / rev[:7])
        self.assertLessEqual({("R", OLD, NEW), ("R", X, Y)}, got)

    def test_the_sweep_reads_the_checks_the_push_and_the_head(self):
        r = self.r
        with self.off(), self.assertRaises(RuntimeError):
            ms.find_checks(self.repo, r["mb"], r["head"], K)
        self.assertEqual(
            self.fetch(ms, "--head", r["head"], "--since", r["since"]), (0, "")
        )
        with self.off():
            checks, tests, _cpp = ms.find_checks(self.repo, r["mb"], r["head"], K)
            touched = ms.interdiff(self.repo, r["since"], r["head"], K)
            tree, how = ms.scratch_copy(self.repo, r["head"], K, None, self.d / "copy")
            # an earlier round's check, read at the head
            carried = ms.git(self.repo, "show", f"{r['head']}:{CORE}")
        self.assertEqual(len(checks), 2)
        self.assertEqual(tests, ["platform/tests/test_core.py"])
        self.assertIn("platform/python/rocke/core.py", touched)
        self.assertEqual(
            (how, (tree / "library/kernels/new.py").is_file()),
            ("git archive of the head", True),
        )
        self.assertIn("negative", carried)

    def test_lane_of_reads_its_tree_and_the_targets(self):
        r = self.r
        with self.off(), self.assertRaises(gitops.GitError):
            gitops.git(self.repo, "show", f"{r['head']}:{TEST}")
        self.assertEqual(self.fetch(lo, "--rev", r["head"], "test_core.py"), (0, ""))
        with self.off():
            prtree.extract(
                self.repo,
                r["head"],
                cl.tree_paths(self.repo, r["head"], K),
                self.d / "t",
            )
            self.assertIn(
                "def test_f", gitops.git(self.repo, "show", f"{r['head']}:{TEST}")
            )

    def test_a_run_needs_its_out_and_a_fetch_none(self):
        argv = ["--repo", self.repo, "--head", self.r["head"], "--rocke-path", K]
        for tool in (cl, ip, ms):
            with self.subTest(tool=tool.__name__):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit) as e:
                        tool.main(argv)
                self.assertEqual(e.exception.code, 2)
                self.assertIn("--out", err.getvalue())
                self.assertEqual(self.fetch(tool, "--head", self.r["head"]), (0, ""))


if __name__ == "__main__":
    unittest.main()
