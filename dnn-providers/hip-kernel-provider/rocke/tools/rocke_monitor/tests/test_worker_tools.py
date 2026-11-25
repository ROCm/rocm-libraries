# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The worker tools (worker/): their settings come from the job's RKMON_* environment or `rk env`, never from a fixed
home folder, login or repository; mkwt.py makes the worktrees from a clone made as rk provision makes it; rule_scan
reads the configured tracker keys; fetch_pr.py leaves out the operator's own posts (config `me`).
"""

import json
import os
import re
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fakerepo  # noqa: E402
import home_repository  # noqa: E402
from rkmon_lib import config, container_layout, deploy  # noqa: E402


def cfg_for(d: str, url_repo: str = "acme/widgets") -> dict:
    c = config.load_defaults()
    c.update(
        repo=url_repo,
        rocke_path=fakerepo.ROCKE_PATH,
        operator={"github_login": "op-login"},
        scope={"labels": ["x"], "path_labels": []},
        worker={"model": "m:1", "url": "http://127.0.0.1:9"},
        clone_since=fakerepo.SINCE,
    )
    c["paths"] = dict(c["paths"], working_dir=f"{d}/work")
    return config.normalize(c)


class TestRuleScan(unittest.TestCase):
    TK, OTHER = (
        "TK" + "EY",
        "OTH" + "ER",
    )  # fake ticket keys, built at run time (no source line looks like one)
    DIFF = textwrap.dedent(
        f"""\
        +++ b/a/b/rocke/platform/k.py
        +# see {TK}-1470 and {OTHER}-12; SHA-256 and UTF-8 are not tickets
        +x = 1
        """
    )

    def scan(self, **env):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "diff.patch"
            p.write_text(self.DIFF, encoding="utf-8")
            e = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
            return subprocess.run(
                [sys.executable, "-I", str(WORKER / "rule_scan.py"), str(p)],
                capture_output=True,
                text=True,
                env=dict(e, **env),
                check=True,
            ).stdout

    def test_any_jira_style_id_without_configured_keys(self):
        out = self.scan()
        self.assertIn("== tracker: 2", out)
        self.assertIn(f"a/b/rocke/platform/k.py: {self.TK}-1470", out)
        self.assertNotIn("SHA-256", out)

    def test_only_the_configured_keys_and_paths_relative_to_rocke(self):
        out = self.scan(RKMON_TRACKER_KEYS=self.TK, RKMON_ROCKE_PATH="a/b/rocke")
        self.assertIn("== tracker: 1", out)
        self.assertIn(f"  platform/k.py: {self.TK}-1470", out)
        self.assertNotIn(f"{self.OTHER}-12", out)


class TestRuleScanCommittedFiles(unittest.TestCase):
    """The data files' figures, PR references and dated narratives in any committed file."""

    MD = (
        "11" + "-05"
    )  # a month and day, built at run time (no source line reads as a dated narrative)
    DIFF = textwrap.dedent(
        f"""\
        +++ b/experiments/run.json
        +{{"tflops": 512.3, "tok_per_s": 1800, "shape": [1, 2]}}
        +{{"latency_ms": 0.21}}
        +++ b/experiments/notes.md
        +| kernel | TFLOP/s |
        +decode reached 1800 tok/s on one GPU
        +See #1234 and PR 1234 and https://github.com/ROCm/x/pull/1234 for the change.
        +{MD}: reran the sweep
        +Measured on 2026-11-05 with the new pin.
        +++ b/a/k.py
        +# 2026-{MD} 14:00Z: switched the default
        +x = "version 11-05"  # not a narrative
        +y = 2  # see PR 1234
        +++ b/PR_BODY.md
        +Follows PR 1234; {MD}: measured
        """
    )

    def test_the_new_sections(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "diff.patch"
            p.write_text(self.DIFF, encoding="utf-8")
            out = subprocess.run(
                [sys.executable, "-I", str(WORKER / "rule_scan.py"), str(p)],
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if not k.startswith("RKMON_")},
                check=True,
            ).stdout
        sections = dict(re.findall(r"^== ([\w ()/-]+?): (\d+)", out, re.M))
        self.assertEqual(sections["perf (data files)"], "2")
        self.assertIn('experiments/run.json: {"tflops": 512.3', out)
        self.assertIn(
            "experiments/notes.md: decode reached 1800 tok/s", out
        )  # the perf section's unit
        self.assertEqual(sections["PR refs (committed files)"], "2")
        self.assertIn("experiments/notes.md: See #1234", out)
        self.assertIn("a/k.py: y = 2  # see PR 1234", out)
        self.assertEqual(sections["dated narrative"], "3")
        self.assertIn(f"experiments/notes.md: {self.MD}: reran the sweep", out)
        self.assertIn(f"a/k.py: # 2026-{self.MD} 14:00Z: switched the default", out)
        self.assertNotIn('version 11-05"', out.split("== dated narrative", 1)[1])
        self.assertNotIn("PR_BODY.md", out.split("== perf (data files)", 1)[1])
        # the drafter reads the new sections' counts with its own pattern
        read = dict(re.findall(r"^== ([\w ()-]+?): (\d+)", out, re.M))
        self.assertEqual(
            [
                read.get(k)
                for k in (
                    "perf (data files)",
                    "PR refs (committed files)",
                    "dated narrative",
                )
            ],
            ["2", "2", "3"],
        )


class TestEnvironment(unittest.TestCase):
    def test_rkenv_reads_rk_env_when_the_job_gave_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            tk = pathlib.Path(d) / "toolkit"
            tk.mkdir()
            (tk / "rk").write_text(
                "#!/bin/sh\necho 'export RKMON_WORKING_DIR=/w RKMON_REPO_DIR=/w/repo RKMON_REPO=a/b "
                "RKMON_ROCKE_PATH=x/rocke'\n"
            )
            (tk / "rk").chmod(0o755)
            e = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
            # A worker script's start (rkenv.load): the values reach this process and every tool it runs.
            load = (
                f"import os, subprocess, sys; sys.path.insert(0, {str(WORKER)!r}); import rkenv; rkenv.load(); "
                "subprocess.run(['sh', '-c', 'echo $RKMON_REPO $RKMON_WORKING_DIR'])"
            )
            r = subprocess.run(
                [sys.executable, "-B", "-c", load],
                capture_output=True,
                text=True,
                env=dict(e, RKMON_TOOLKIT=str(tk)),
            )
            self.assertEqual(r.stdout.strip(), "a/b /w", r.stderr)
            r = subprocess.run(
                [sys.executable, "-B", "-c", load + "; print('reached')"],
                capture_output=True,
                text=True,
                env=dict(e, RKMON_TOOLKIT=f"{d}/none"),
            )
            self.assertEqual(r.returncode, 2)
            self.assertEqual(
                r.stderr,
                f"rkenv: no RKMON_* settings and no toolkit at {d}/none/rk: run this from a "
                "monitor job, or set RKMON_TOOLKIT\n",
            )
            self.assertNotIn("reached", r.stdout)
            # rk env without the clone's folder: the first missing setting, and its fix.
            (tk / "rk").write_text(
                "#!/bin/sh\necho 'export RKMON_WORKING_DIR=/w RKMON_REPO=a/b'\n"
            )
            r = subprocess.run(
                [sys.executable, "-B", "-c", load],
                capture_output=True,
                text=True,
                env=dict(e, RKMON_TOOLKIT=str(tk)),
            )
            self.assertEqual(
                (r.returncode, r.stderr),
                (
                    2,
                    "rkenv: RKMON_REPO_DIR is not set (check config.json with `rk doctor`)\n",
                ),
            )

    def test_a_worker_tool_installed_apart_reads_the_package_s_fixed_values(self):
        """A copy of worker/ installed apart from the package (rk worker-tools --install: no rkmon_lib beside it)
        reads the package's fixed values through rkenv, from the toolkit RKMON_TOOLKIT names: the reviewer model's
        client tells this machine's address from another's by the one list of its names (container_layout.LOCAL_HOSTS).
        """
        with tempfile.TemporaryDirectory() as d:
            tools = pathlib.Path(d) / "tools"
            shutil.copytree(WORKER, tools, ignore=shutil.ignore_patterns("__pycache__"))
            probe = (
                f"import sys; sys.path.insert(0, {str(tools)!r}); import reviewer_client as c; "
                f"print(c.is_local('http://{container_layout.LOOPBACK}:9/v1'), "
                "c.is_local('http://localhost:9'), c.is_local('https://models.example.com/v1'))"
            )
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            r = subprocess.run(
                [sys.executable, "-B", "-c", probe],
                capture_output=True,
                text=True,
                cwd=d,
                env=dict(env, RKMON_TOOLKIT=str(ROOT)),
            )
            self.assertEqual(r.stdout.split(), ["True", "True", "False"], r.stderr)

    def test_rkenv_clone_since(self):
        with tempfile.TemporaryDirectory() as d:
            sys.path.insert(0, str(WORKER))
            try:
                import rkenv
            finally:
                sys.path.remove(str(WORKER))
            with unittest.mock.patch.dict(
                os.environ, {"RKMON_CLONE_SINCE": "2001-01-01"}
            ):
                self.assertEqual(rkenv.clone_since(d), "2001-01-01")
            with unittest.mock.patch.dict(os.environ, {"RKMON_CLONE_SINCE": ""}):
                self.assertEqual(rkenv.clone_since(d), "")  # a full clone: nothing
                (pathlib.Path(d) / ".git").mkdir()
                (pathlib.Path(d) / ".git" / "shallow").write_text("")
                self.assertEqual(rkenv.clone_since(d), "")

    def test_no_tool_names_a_home_folder_login_or_repository(self):
        # the home repository, its name, and the top of its folder, as the package's repository defaults name them
        repository, top = (
            home_repository.REPOSITORY,
            home_repository.FOLDER.split("/")[0],
        )
        for p in sorted(x for x in WORKER.iterdir() if x.is_file()):
            text = p.read_text(encoding="utf-8")
            for bad in (
                "$HOME/",
                "Path.home()",
                "rocke_review",
                repository,
                repository.split("/")[1],
                f"{top}/",
            ):
                self.assertNotIn(bad, text, f"{p.name} names {bad}")

    def test_failure_ids_are_compared_relative_to_the_configured_folder_s_parent(self):
        """faildiff2.py compares two runs' failure IDs relative to the parent of the folder the job's RKMON_ROCKE_PATH
        names (from the folder on, for a folder at the repository's root): the same failure in two trees, or as a path
        from the repository's root, is one; a path that does not pass through the parent is kept whole.
        """
        for folder, a, b, whole in (
            (
                "a/src/kern",
                "/w/wt1/a/src/kern/tests/test_a.py::test_x",
                "a/src/kern/tests/test_a.py::test_x",
                "/w/wt1/x/mysrc/t.py::z",
            ),
            (
                "a/src/kern",
                "/w/wt1/a/src/lib/t.py::test_y",
                "/w/wt2/a/src/lib/t.py::test_y",
                "/w/wt1/a/srcx/t.py::z",
            ),
            (
                "kern",
                "/w/wt1/x/kern/tests/test_b.py::test_y",
                "kern/tests/test_b.py::test_y",
                "/w/wt1/x/kern2/t.py::z",
            ),
        ):
            with self.subTest(folder=folder, a=a), tempfile.TemporaryDirectory() as d:
                one, two = pathlib.Path(d) / "a.txt", pathlib.Path(d) / "b.txt"
                one.write_text(f"FAILED {a} - boom\nFAILED {whole}\n")
                two.write_text(f"FAILED {b} - boom\n")
                out = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        str(WORKER / "faildiff2.py"),
                        str(one),
                        str(two),
                    ],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, RKMON_ROCKE_PATH=f"/{folder}/"),
                ).stdout
                self.assertEqual(
                    out,
                    f"A-only: ['FAILED {whole}']\nB-only: []\ncommon 1\n",
                )

    def test_the_launcher_refuses_without_a_review_book(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
        r = subprocess.run(
            [sys.executable, "-B", str(WORKER / "review_agent_run.py"), "7"],
            capture_output=True,
            text=True,
            env=dict(
                e,
                RKMON_WORKING_DIR="/nonexistent",
                RKMON_REPO_DIR="/nonexistent",
                RKMON_REPO="a/b",
                RKMON_ROCKE_PATH="r",
                RKMON_WORKERS="/nonexistent",
                RKMON_REVIEW_BOOK="",
            ),
        )
        self.assertEqual(r.returncode, 2)
        self.assertEqual(
            r.stdout,
            "no review book at '' (paths.review_book): the container's first start puts it there (rkmon start)\n",
        )


class TestWorktrees(unittest.TestCase):
    def test_mkwt_on_a_clone_made_as_rk_provision_makes_it(self):
        with tempfile.TemporaryDirectory() as d:
            url = fakerepo.make(d)
            cfg = cfg_for(d)
            deploy.layout(cfg)
            deploy.run(deploy.clone_commands(cfg, url))
            self.assertIsNone(deploy.clone_check(cfg))
            # wtdev's folder is made first, as a tree PR code reads (worker/perms.py tree), then git fills it.
            first = deploy.wtdev_commands(cfg)[0]
            self.assertEqual(
                first[2:],
                [str(WORKER / "perms.py"), "tree", "--group", container_layout.PR_GROUP]
                + [str(pathlib.Path(cfg["paths"]["working_dir"]) / "wtdev")],
            )
            deploy.run(deploy.wtdev_commands(cfg))
            env = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
            env.update(config.env(cfg))
            r = subprocess.run(
                [sys.executable, "-B", str(WORKER / "mkwt.py"), "7"],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("behind/ahead: 1\t1", r.stdout)
            self.assertIn("conflicting files vs develop:", r.stdout)
            w = pathlib.Path(cfg["paths"]["working_dir"])
            self.assertEqual(
                (w / "wt7" / fakerepo.ROCKE_PATH / "platform" / "x.txt").read_text(),
                "x2\n",
            )
            self.assertEqual(
                (w / "wt7m" / "other").exists(), False
            )  # sparse: rocKE's provider and .github only
            self.assertEqual(
                (w / "wt7m" / fakerepo.ROCKE_PATH / "platform" / "x.txt").read_text(),
                "x2\n",
            )
            self.assertTrue((w / "wtdev" / ".github" / "CODEOWNERS").exists())

    def test_mkwt_puts_the_pr_trees_on_the_ramdisk_behind_links(self):
        """With the RAM disk on, wt<N> and wt<N>m live there and the working folder links to each; wtdev stays on the
        disk; a link whose tree is gone (the RAM disk emptied by a restart) is made again.
        """
        with tempfile.TemporaryDirectory() as d:
            url = fakerepo.make(d)
            rd = pathlib.Path(d) / "rd"
            cfg = cfg_for(d)
            cfg = config.normalize(
                dict(cfg, ramdisk={"enabled": True, "path": str(rd), "size_gb": 4})
            )
            deploy.layout(cfg)
            deploy.run(deploy.clone_commands(cfg, url))
            deploy.run(deploy.wtdev_commands(cfg))
            env = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
            env.update(config.env(cfg))
            w = pathlib.Path(cfg["paths"]["working_dir"])
            for _round in range(2):
                r = subprocess.run(
                    [sys.executable, "-B", str(WORKER / "mkwt.py"), "7"],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                for name in ("wt7", "wt7m"):
                    self.assertTrue((w / name).is_symlink())
                    self.assertEqual((w / name).resolve(), (rd / name).resolve())
                self.assertEqual(
                    (
                        w / "wt7m" / fakerepo.ROCKE_PATH / "platform" / "x.txt"
                    ).read_text(),
                    "x2\n",
                )
                self.assertFalse((w / "wtdev").is_symlink())
                for name in ("wt7", "wt7m"):  # a restart empties the RAM disk
                    shutil.rmtree(rd / name)


class TestFetchPr(unittest.TestCase):
    def test_the_operators_own_posts_are_left_out(self):
        pr = {
            "number": 7,
            "title": "t",
            "body": "b",
            "author": {"login": "dev-a"},
            "labels": [],
            "state": "OPEN",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "headRefName": "h",
            "headRefOid": "a" * 40,
            "baseRefName": "develop",
            "createdAt": "c",
            "updatedAt": "u",
            "additions": 1,
            "deletions": 0,
            "commits": [],
            "files": [],
            "reviews": [
                {
                    "author": {"login": "op-login"},
                    "state": "COMMENTED",
                    "body": "ours",
                    "submittedAt": "s",
                },
                {
                    "author": {"login": "dev-b"},
                    "state": "APPROVED",
                    "body": "theirs",
                    "submittedAt": "s",
                },
            ],
            "comments": [
                {"author": {"login": "op-login"}, "body": "ours", "createdAt": "c"}
            ],
        }
        threads = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [],
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as d:
            bin_ = pathlib.Path(d) / "bin"
            bin_.mkdir()
            (bin_ / "gh").write_text(
                textwrap.dedent(
                    f"""\
                #!{sys.executable}
                import json, sys
                a = sys.argv[1:]
                open({str(pathlib.Path(d) / 'calls')!r}, "a").write(" ".join(a) + "\\n")
                if a[:2] == ["pr", "view"]:
                    print(json.dumps({json.dumps(pr)!r} and json.loads({json.dumps(pr)!r})))
                elif a[:2] == ["api", "graphql"]:
                    print({json.dumps(threads)!r})
                elif a[:2] == ["pr", "diff"]:
                    print("diff")
                elif a[0] == "api" and a[-1] == ".base.sha":
                    print("b" * 40)
                """
                )
            )
            (bin_ / "gh").chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("RKMON_")}
            env.update(
                PATH=f"{bin_}:{env['PATH']}",
                RKMON_REPO="acme/widgets",
                RKMON_OPERATOR_LOGIN="op-login",
                RKMON_REPO_DIR="/nonexistent",
            )
            r = subprocess.run(
                [sys.executable, "-B", str(WORKER / "fetch_pr.py"), "7", f"{d}/pr"],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("(omitted 2 of our own items)", r.stdout)
            self.assertNotIn(
                "ours", (pathlib.Path(d) / "pr" / "reviews.md").read_text()
            )
            self.assertIn("theirs", (pathlib.Path(d) / "pr" / "reviews.md").read_text())
            calls = (pathlib.Path(d) / "calls").read_text()
            self.assertIn("--repo acme/widgets", calls)
            self.assertIn("owner=acme", calls)


if __name__ == "__main__":
    unittest.main()
