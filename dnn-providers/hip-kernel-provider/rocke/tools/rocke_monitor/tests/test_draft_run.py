# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/draft_run.py's staging, with stand-ins for fetch_pr.py and mkwt.py under a temporary review root: the PR's
data and trees, the `staged` marker the checks wait for, and a failed staging's exit; two drafts that stage at once
take turns (the staging lock); and the shadow mode's own folder, staged on the live review's trees once they hold its
head. The draft itself is tests/test_drafter.py's."""

import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "draft_run.py"
sys.path.insert(0, str(ROOT))

from rkmon_lib import config  # noqa: E402

FETCH = (
    "import os, pathlib, sys, time\nd = pathlib.Path(sys.argv[2]); d.mkdir(parents=True, exist_ok=True)\n"
    "log = os.environ.get('FETCH_LOG')\nif log: open(log, 'a').write('start\\n')\n"
    "time.sleep(float(os.environ.get('FETCH_SLEEP', '0')))\nif log: open(log, 'a').write('end\\n')\n"
    "(d / 'meta.md').write_text('# PR #' + sys.argv[1] + ': t\\n')\n"
)
MKWT = (
    "import os, sys\nD = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))\n"
    "os.makedirs(f'{D}/wt{sys.argv[1]}', exist_ok=True); os.makedirs(f'{D}/wt{sys.argv[1]}m', exist_ok=True)\n"
    "print('head=aaaaaaaaaa develop=bbbbbbbbbb merge-base=cccccccccc')\nprint('behind/ahead: 0\\t1')\n"
)


def git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class TestDraftRun(unittest.TestCase):
    def env(self, root: pathlib.Path, **env) -> dict:
        """The script's environment under `root`; its `rk stage-ours` reads the test's own configuration
        (RKMON_HOME)."""
        home = root / "home"
        home.mkdir(exist_ok=True)
        cfg = dict(
            config.load_defaults(),
            repo="ROCm/r",
            rocke_path="proj",
            operator={"github_login": "dev-op"},
            merge_authority="none",
            scope={"labels": ["x"], "path_labels": []},
        )
        cfg["paths"] = dict(cfg.get("paths") or {}, working_dir=str(root))
        (home / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return {
            "PATH": "/usr/bin:/bin",
            "RKMON_REVIEW_ROOT": str(root),
            "RKMON_HOME": str(home),
            **env,
        }

    def run_script(
        self, root: pathlib.Path, *args, **env
    ) -> subprocess.CompletedProcess:
        """draft_run.py N ARGS under `root`."""
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "7", *args],
            capture_output=True,
            text=True,
            timeout=60,
            env=self.env(root, **env),
        )

    def tools(self, root: pathlib.Path, mkwt: str = MKWT) -> None:
        (root / "tools").mkdir()
        (root / "tools" / "fetch_pr.py").write_text(FETCH)
        (root / "tools" / "mkwt.py").write_text(mkwt)

    def test_staging_writes_the_data_the_trees_and_the_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.tools(root)
            p = self.run_script(root, "--stage-only")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            ws = root / "workers" / "pr7"
            self.assertTrue((ws / "staged").is_file())
            self.assertRegex(
                (ws / "staged").read_text(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n$"
            )
            self.assertEqual((ws / "pr" / "meta.md").read_text(), "# PR #7: t\n")
            self.assertIn("behind/ahead: 0\t1", (ws / "mkwt.log").read_text())
            self.assertTrue((root / "wt7m").is_dir())
            self.assertIn(
                "#7", p.stdout
            )  # rk stage-ours says what it staged (here: no review of ours)
            self.assertTrue(
                p.stdout.endswith(
                    "head=aaaaaaaaaa develop=bbbbbbbbbb merge-base=cccccccccc\n"
                    "behind/ahead: 0\t1\n"
                ),
                p.stdout,
            )

    def test_a_failed_staging_ends_the_job_without_the_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.tools(root, "import sys\nprint('fatal: no such PR')\nsys.exit(1)\n")
            p = self.run_script(root, "--stage-only")
            self.assertEqual(p.returncode, 1)
            self.assertTrue(
                p.stdout.endswith("mkwt failed:\nfatal: no such PR\n"), p.stdout
            )
            self.assertFalse((root / "workers" / "pr7" / "staged").exists())

    def test_two_drafts_stage_one_at_a_time(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.tools(root)
            log = root / "fetch.log"
            env = self.env(root, FETCH_LOG=str(log), FETCH_SLEEP="1.5")
            first = subprocess.Popen(
                [sys.executable, "-B", str(SCRIPT), "7", "--stage-only"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            second = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), "8", "--stage-only"],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.wait(60), 0)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(
                log.read_text().split(), ["start", "end", "start", "end"]
            )  # never overlapping

    def test_a_shadow_draft_stages_its_own_folder_on_the_live_trees(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            self.tools(
                root, "import sys\nprint('the shadow never makes trees')\nsys.exit(1)\n"
            )
            wt = root / "wt7"
            wt.mkdir()
            git(wt, "init", "-q")
            git(wt, "commit", "-q", "--allow-empty", "-m", "head")
            head = git(wt, "rev-parse", "HEAD")
            subprocess.run(
                ["git", "clone", "-q", str(wt), str(root / "wt7m")], check=True
            )
            git(
                root / "wt7m",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "the merge with develop",
            )
            live = root / "workers" / "pr7"
            live.mkdir(parents=True)
            (live / "review_agent.md").write_text("the live draft\n")
            # Not yet: the live job's mkwt.log names another head.
            (live / "mkwt.log").write_text(
                "head=bbbbbbbbbb develop=c merge-base=c\nconflicting files vs develop:\n"
            )
            p = self.run_script(
                root,
                "--shadow",
                "cand",
                head,
                "--stage-only",
                RKMON_SHADOW_WAIT_MIN="0",
            )
            self.assertEqual(p.returncode, 1)
            self.assertIn(
                f"the live trees never held {head[:10]}: no shadow draft", p.stdout
            )
            ws = root / "workers" / "shadow_cand" / "pr7"
            self.assertFalse((ws / "staged").exists())
            (live / "mkwt.log").write_text(
                f"head={head[:10]} develop=c merge-base=c\nbehind/ahead: 0\t1\n"
                "conflicting files vs develop:\n\n"
            )
            p = self.run_script(
                root,
                "--shadow",
                "cand",
                head,
                "--stage-only",
                RKMON_SHADOW_WAIT_MIN="0",
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertTrue((ws / "staged").is_file())
            self.assertEqual((ws / "pr" / "meta.md").read_text(), "# PR #7: t\n")
            self.assertEqual(
                (ws / "mkwt.log").read_text(), (live / "mkwt.log").read_text()
            )
            self.assertFalse((ws / "review_agent.md").exists())  # never the live draft
            self.assertEqual((live / "review_agent.md").read_text(), "the live draft\n")

    def test_a_shadow_draft_needs_its_binding_and_head(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_script(pathlib.Path(d), "--shadow", "cand")
            self.assertEqual(p.returncode, 1)
            self.assertIn("usage: draft_run.py N --shadow BINDING HEAD", p.stderr)


if __name__ == "__main__":
    unittest.main()
