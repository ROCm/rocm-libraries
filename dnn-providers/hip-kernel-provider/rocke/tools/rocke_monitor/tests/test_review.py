# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for Phase 1: post guards, family detection, the generated check script, and the job runner."""

import json
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

TK = (
    "TK" + "EY"
)  # a fake ticket key, built at run time so no source line looks like a ticket

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rkmon_lib import config, jobs, paths, post, prcode, review  # noqa: E402
from rkmon_lib.store import Store  # noqa: E402
import home_repository  # noqa: E402

CFG = {
    "review_first_line": f"Automated review by rocke-monitor: Model - qwen, effort: {config.MODEL_DEFAULT}",
    "worker": {
        "model": "qwen"
    },  # the reviewer model line 1 gives (bindings.first_line)
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "hot_spots": {"RG": "rk/g.json", "DG": "rk/d.json"},
    "paths": {"working_dir": "/nonexistent"},
}
REVIEWING = dict(
    CFG, features={"review_automation": {"mode": "live"}}
)  # a deployment that posts reviews


class TestConflictList(unittest.TestCase):
    """The checks list every conflicting file and count them (all of them, never a part)."""

    def run_merge_lines(self, log_text):
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        # The line that lists them (the NC line before it only counts them, for the merge verdict).
        line = next(
            ln for ln in script.splitlines() if "mkwt.log" in ln and "f[++n]" in ln
        )
        with tempfile.TemporaryDirectory() as d:
            log = pathlib.Path(d) / "workers/pr1/mkwt.log"
            log.parent.mkdir(parents=True)
            log.write_text(log_text)
            r = subprocess.run(
                ["bash", "-c", f"D={d}; WK={d}/workers; N=1\n" + line],
                capture_output=True,
                text=True,
            )
        return r.stdout

    def test_every_conflict_is_listed_with_the_count(self):
        files = [f"dir/f{i}.cpp" for i in range(11)]
        log = (
            "Auto-merging dir/f0.cpp\n"
            + "".join(f"CONFLICT (content): Merge conflict in {f}\n" for f in files)
            + "!! merge with develop has conflicts in wt1m\nhead=a develop=b merge-base=c\nbehind/ahead: 33\t30\n"
            + "conflicting files vs develop:\n"
            + "".join(f + "\n" for f in files)
            + "\n"
        )
        out = self.run_merge_lines(log)
        self.assertIn("!! merge with develop has conflicts in wt1m", out)
        self.assertIn("behind/ahead: 33\t30", out)
        self.assertIn("conflicting files vs develop: 11", out)
        for f in files:
            self.assertIn("  " + f, out)
        self.assertNotIn("CONFLICT (content)", out)

    def test_patch_fingerprint_ignores_how_git_aligns_repeated_lines(self):
        # An unchanged patch whose repeated "# ----" line git places elsewhere after develop touched the file is not
        # substantive to the follow-up classifier
        a = "@@ -1 +1,3 @@\n+# ----\n+x = 1\n+# ----\n@@ -9,0 +12 @@\n+y = 2\n"
        b = "@@ -1 +1,2 @@\n+# ----\n+x = 1\n@@ -9,0 +11,2 @@\n+y = 2\n+# ----\n"
        self.assertEqual(review._patch_fingerprint(a), review._patch_fingerprint(b))
        self.assertNotEqual(
            review._patch_fingerprint(a),
            review._patch_fingerprint(a.replace("y = 2", "y = 3")),
        )
        self.assertNotEqual(
            review._patch_fingerprint(a),
            review._patch_fingerprint(a.replace("+x = 1", "-x = 1")),
        )

    def test_a_clean_merge_says_zero(self):
        out = self.run_merge_lines(
            "head=a develop=b merge-base=c\nbehind/ahead: 8\t15\nconflicting files vs develop:\n"
        )
        self.assertEqual(
            out.splitlines(), ["behind/ahead: 8\t15", "conflicting files vs develop: 0"]
        )


class TestBaseBranch(unittest.TestCase):
    """A PR into a feature branch is checked against that branch, never develop: the merge, the tests' merge base
    and every scripted check, so none reads the feature branch's own work as the PR's.
    """

    PLAN = {
        "families": [],
        "tests": ["platform/tests/test_x.py"],
        "build": False,
        "golden": False,
        "engine_wide": False,
        "outside_rocke": [],
        "libparity": False,
        "_dirs": {},
    }

    def test_the_checks_merge_and_compare_with_the_base_branch(self):
        base = "users/dev-x/feature-a"
        script = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", self.PLAN, None, reviewed="b" * 40, base=base
        )
        self.assertIn(f"BREF=refs/remotes/origin/{base}\n", script)
        self.assertIn(
            f"echo 'base branch: {base} (not develop): the merge, the merge base and the comparisons below "
            "are with it'",
            script,
        )
        self.assertIn(f"grep -qF ' base={base}' $WK/pr$N/mkwt.log", script)
        self.assertIn("MBS=$(git -C $R0 merge-base $BREF ", script)
        self.assertIn(
            "$(git -C $W rev-parse $BREF 2>/dev/null)", script
        )  # the base without a merged tree
        for tool in (
            "mutation_sweep.py",
            "ctest_labels.py",
            "moved_refs.py",
            "doc_refs.py",
            "import_probe.py",
            "since_review.py",
        ):
            line = next(ln for ln in script.splitlines() if f"$M/tools/{tool}" in ln)
            self.assertIn("--base $BREF", line, tool)
        self.assertIn(f"$M/tools/abi_scan.py $R0 $K {'a' * 40} $BREF", script)
        self.assertNotIn("refs/remotes/origin/develop", script)
        # CI at head compares with the base branch's runs, and checks.md's header names the base and its commit
        self.assertIn("DVS=$(git -C $R0 rev-list -n 6 $BREF ", script)
        self.assertIn(
            f'echo \'base: {base} @\' "$(cut -c1-10 "$OUT/develop_sha" 2>/dev/null)" >> "$R"',
            script,
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        # develop: the same ref the checks always used, and no base line
        dev = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", self.PLAN, None, reviewed="b" * 40
        )
        self.assertIn("BREF=refs/remotes/origin/develop\n", dev)
        self.assertNotIn("base branch:", dev)
        self.assertIn("echo 'base: develop @' ", dev)

    def dg_run(self, helps: str) -> tuple[str, str]:
        """The differential golden section of a feature-branch PR's checks, run on a tree whose run_diff.py --help
        prints `helps` (and passes both modes): its output and status lines."""
        script = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", self.PLAN, None, base="users/x/y"
        ).splitlines()
        start = next(i for i, ln in enumerate(script) if ln.startswith("DGH="))
        end = next(i for i in range(start, len(script)) if script[i] == "fi")
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            diff = root / "P" / "tests" / "instances" / "differential"
            diff.mkdir(parents=True)
            (diff / "run_diff.py").write_text(
                f"import sys\nif '--help' in sys.argv:\n    print({helps!r})\nsys.exit(0)\n"
            )
            shim = root / "bin"
            shim.mkdir()
            (shim / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
            (shim / "python").chmod(0o755)
            (root / "out").mkdir()
            head = (
                f"{prcode.SHELL}\nP={root}/P; OUT={root}/out; M={ROOT}; PATH={shim}:$PATH; TMPDIR={d}\n"
                'st() { echo "$1 $2 $3" >> $OUT/status; }\ncutc() { cat; }\nprslot || exit 9\n'
            )
            p = subprocess.run(
                ["bash", "-c", head + "\n".join(script[start : end + 1])],
                capture_output=True,
                text=True,
                timeout=60,
            )
            status = (root / "out" / "status").read_text()
        return p.stdout + p.stderr, status

    def test_the_differential_golden_probes_the_trees_run_diff_flags_first(self):
        """A feature base's run_diff.py without a flag the CI commands pass reads NOT RUN, and G5 stays pending."""
        out, status = self.dg_run("usage: run_diff.py --mode {ll,ir} --flavor F")
        self.assertIn(
            "NOT RUN: this tree's run_diff.py --help names no --python-only --check-golden --canonical "
            "(base users/x/y)",
            out,
        )
        self.assertEqual(
            status,
            "dg skip run_diff.py has no --python-only --check-golden --canonical\n",
        )
        self.assertEqual(
            review.gates_from_status(
                {"golden": ("pass", "ok"), "dg": ("skip", "run_diff.py has no --x")}
            )["G5"][0],
            "pending",
        )
        out, status = self.dg_run(
            "--mode --python-only --check-golden --flavor --canonical"
        )
        self.assertNotIn("NOT RUN", out)
        self.assertEqual(status, "dg pass GOLDEN OK\n")

    def test_mkwt_merges_the_base_branch(self):
        self.assertEqual(
            review.remerge_script(CFG, 7, None),
            review.remerge_script(CFG, 7, "develop"),
        )
        self.assertIn(
            f"\nRKMON_BASE=users/x/y python3 -B {paths.worker_tool('mkwt.py')} $N ",
            review.remerge_script(CFG, 7, "users/x/y"),
        )
        self.assertIn(
            " RKMON_BASE=users/x/y SPARSE_EXTRA=",
            review.worker_script(CFG, 7, [], "follow-up", base="users/x/y"),
        )
        self.assertNotIn(
            "RKMON_BASE", review.worker_script(CFG, 7, [], "follow-up", base="develop")
        )

    def test_a_follow_up_is_classified_against_the_base_branch(self):
        """The feature branch's commits brought in by a merge are not the PR's own: a merge-only push."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            origin, mirror = d / "origin", d / "repo"

            def g(repo, *a):
                return subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "-c",
                        "user.name=t",
                        "-c",
                        "user.email=t@t",
                        *a,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            def commit(files, msg):
                for f, text in files.items():
                    (origin / f).parent.mkdir(parents=True, exist_ok=True)
                    (origin / f).write_text(text)
                g(origin, "add", "-A")
                g(origin, "commit", "-qm", msg)
                return g(origin, "rev-parse", "HEAD")

            origin.mkdir()
            g(origin, "init", "-q", "-b", "develop")
            commit({"rk/a.py": "1\n", "rk/b.py": "1\n"}, "base")
            g(origin, "checkout", "-qb", "feat")
            commit({"rk/f.py": "1\n"}, "feature work")
            g(origin, "checkout", "-qb", "pr")
            old = commit({"rk/a.py": "2\n"}, "the PR")
            g(origin, "checkout", "-q", "feat")
            commit({"rk/b.py": "2\n", "rk/f.py": "2\n"}, "the feature branch moves on")
            g(origin, "checkout", "-q", "pr")
            g(origin, "merge", "-q", "--no-edit", "feat")
            new = g(origin, "rev-parse", "HEAD")
            g(origin, "update-ref", "refs/pull/5/head", new)
            g(origin, "checkout", "-q", "develop")
            g(d, "clone", "-q", "--bare", str(origin), str(mirror))
            g(mirror, "remote", "set-url", "origin", str(origin))
            g(mirror, "fetch", "-q", "origin", "+develop:refs/remotes/origin/develop")
            cfg = dict(CFG, paths={"repo": str(mirror)})
            rec = {"n": 5, "head": new, "base": "feat"}
            self.assertEqual(review.classify(cfg, rec, old)["kind"], "merge-only")
            self.assertEqual(review._changed_files(cfg, rec), ["rk/a.py"])
            self.assertEqual(
                review.classify(cfg, dict(rec, base="develop"), old)["kind"],
                "substantive",
            )


class TestThreadsDigest(unittest.TestCase):
    """Reviewers' bodies are quoted in the checks (a reviewer's "## 🔴 Blockers: head does not build" never reads as a
    section of the checks, nor as a build blocker the checks found)."""

    def test_a_reviewer_heading_is_quoted_and_the_entry_lines_are_not(self):
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", plan, "2026-11-01T00:00:00Z"
        )
        line = next(ln for ln in script.splitlines() if "prthreads.py" in ln)
        digest = (
            "== reviews with bodies\n2026-11-04T22:54 dev-r COMMENTED :: Review at head `04e1654`.\n\n"
            "## 🔴 Blockers: head does not build\n- `KernelDefinition.hpp:72`\n"
        )
        cmd = 'printf "%s" "$DIGEST"' + line.split("2>/dev/null", 1)[1]
        out = subprocess.run(
            ["bash", "-c", cmd],
            env={"DIGEST": digest, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        self.assertEqual(
            out,
            [
                "== reviews with bodies",
                "2026-11-04T22:54 dev-r COMMENTED :: Review at head `04e1654`.",
                "    > ",
                "    > ## 🔴 Blockers: head does not build",
                "    > - `KernelDefinition.hpp:72`",
            ],
        )


class TestPipelining(unittest.TestCase):
    """A finished draft waits for verification while the next review starts; a moved head drops the stale draft."""

    def setUp(self):
        from unittest import mock
        from rkmon_lib.store import empty_state

        self.mock = mock
        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name))
        self.st = empty_state()
        self.st["prs"] = {
            str(n): {
                "n": n,
                "head": str(n) * 40,
                "status": "open",
                "draft": False,
                "size": 10,
            }
            for n in (1, 2)
        }
        self.st["review_pipeline"] = {
            "pr": 1,
            "kind": "follow-up",
            "stage": "worker",
            "head": "1" * 40,
            "worker": {"id": "w", "started": "2000-01-01T00:00:00Z"},
            "checks": {"id": "c", "started": "2000-01-01T00:00:00Z"},
        }
        self.plan = [
            {"kind": "FOLLOW_UP", "pr": 1, "text": "x"},
            {"kind": "FIRST_ROUND", "pr": 2, "text": "y"},
        ]

    def tearDown(self):
        self.d.cleanup()

    def test_ready_moves_aside_and_next_starts(self):
        m = self.mock
        with m.patch.object(
            review.jobs, "status", lambda store, job: "done" if job else "none"
        ), m.patch.object(
            review, "read_checks_status", lambda path: ({}, None, None)
        ), m.patch.object(
            review, "has_draft", lambda cfg, n: True
        ), m.patch.object(
            review,
            "start_review",
            lambda cfg, st, store, n, kind, again=False: [
                f"REVIEW_START #{n} ({kind})"
            ],
        ):
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertTrue(lines[0].startswith("REVIEW_READY #1"), lines)
        self.assertEqual(
            lines[1:], ["REVIEW_START #2 (first-round)"]
        )  # #1 is skipped: it awaits verification
        self.assertEqual([r["pr"] for r in self.st["review_ready"]], [1])
        self.assertEqual(review.pipes(self.st), [])  # (the stand-in start adds no pipe)
        self.assertNotIn(
            "review_pipeline", self.st
        )  # the state before the pipes became the first pipe

    def test_failed_worker_without_draft_retries_once_then_skips(self):
        m = self.mock
        with m.patch.object(
            review.jobs,
            "status",
            lambda store, job: "failed(rc=1)" if job and job["id"] == "w" else "done",
        ), m.patch.object(review, "has_draft", lambda cfg, n: False), m.patch.object(
            review, "pick_next", lambda cfg, st, store, plan, **kw: None
        ):
            lines = review.advance(CFG, self.st, self.store, self.plan)
            self.assertTrue(
                lines[0].startswith("REVIEW_FAIL #1") and "will retry" in lines[0],
                lines,
            )
            self.assertNotIn(1, [r["pr"] for r in self.st.get("review_ready", [])])
            self.assertEqual(review.pipes(self.st), [])
            self.st["review_pipes"] = [
                {
                    "pr": 1,
                    "kind": "follow-up",
                    "stage": "worker",
                    "head": "1" * 40,
                    "worker": {"id": "w", "started": "2000-01-01T00:00:00Z"},
                    "checks": {"id": "c", "started": "2000-01-01T00:00:00Z"},
                }
            ]
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertIn("skipped until", lines[0])
        self.assertIn(1, self.st["review_skip"])

    def test_moved_head_stops_the_running_worker(self):
        """A head that moves mid-draft stops its running worker, so no two qwen workers run at once."""
        killed = []
        self.st["review_pipeline"] = {
            "pr": 1,
            "kind": "follow-up",
            "stage": "worker",
            "head": "1" * 40,
            "worker": {"id": "w", "started": "2000-01-01T00:00:00Z"},
            "checks": {"id": "c", "started": "2000-01-01T00:00:00Z"},
        }
        self.st.setdefault("jobs", {})["heavy"] = {"id": "c"}
        self.st["prs"]["1"]["head"] = "f" * 40
        m = self.mock
        with m.patch.object(
            review.jobs, "status", lambda store, job: "running"
        ), m.patch.object(
            review.jobs, "kill", lambda store, job: killed.append(job["id"])
        ), m.patch.object(
            review, "pick_next", lambda cfg, st, store, plan, **kw: None
        ):
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertTrue(lines[0].startswith("REVIEW_DROP #1"), lines)
        self.assertEqual(killed, ["w", "c"])
        self.assertIsNone(self.st["jobs"]["heavy"])
        self.assertEqual(review.pipes(self.st), [])

    def test_checks_wait_for_a_pending_develop_sanity(self):
        """The review step does not re-take the heavy slot every time a check finishes while a develop sanity waits."""
        self.st["review_pipeline"]["checks"] = None
        self.st["develop"]["pending_sanity"] = ["a" * 40]
        started = []
        m = self.mock
        with m.patch.object(
            review.jobs, "status", lambda store, job: "done" if job else "none"
        ), m.patch.object(
            review, "pick_next", lambda cfg, st, store, plan, **kw: None
        ), m.patch.object(
            review.jobs, "start", lambda *a, **k: started.append(a) or {"id": "x"}
        ):
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertEqual(
            lines,
            ["REVIEW #1: worker done; checks wait for develop sanity on aaaaaaaaaa"],
        )
        self.assertEqual(started, [])
        self.assertIsNone(self.st["jobs"].get("heavy"))
        self.st["guard"] = {
            "last_sanity_green": "a" * 40
        }  # already green there: nothing to wait for
        with m.patch.object(
            review.jobs, "status", lambda store, job: "done" if job else "none"
        ), m.patch.object(review, "pick_next", lambda cfg, st, store, plan, **kw: None):
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertNotIn("develop sanity", lines[0])

    def test_moved_head_drops_the_waiting_draft(self):
        self.st["review_pipeline"] = None
        self.st["review_ready"] = [
            {
                "pr": 1,
                "kind": "follow-up",
                "head": "1" * 40,
                "at": "2000-01-01T00:00:00Z",
            }
        ]
        self.st["prs"]["1"]["head"] = "f" * 40
        with self.mock.patch.object(
            review, "pick_next", lambda cfg, st, store, plan, **kw: None
        ):
            lines = review.advance(CFG, self.st, self.store, self.plan)
        self.assertTrue(lines[0].startswith("REVIEW_DROP #1"), lines)
        self.assertEqual(self.st["review_ready"], [])


class TestPostGuards(unittest.TestCase):
    def test_clean_post(self):
        body = (
            CFG["review_first_line"]
            + "\n\n## PR #1 — t — 🟢 ACCEPT: ok\nsee https://github.com/ROCm/x/pull/1, 1.3x faster"
        )
        self.assertEqual(post.check(CFG, body), [])

    def test_refusals(self):
        body = (
            "Automated review: Model - Claude, effort: max\n"
            "https://wiki.example.net/x 3.2 "
            + "ms ["
            + TK
            + "-12](https://tracker.example.net/browse/"
            + TK
            + "-12)"
        )
        problems = " | ".join(post.check(CFG, body))
        self.assertIn("line 1 must be exactly", problems)
        self.assertIn("non-public host", problems)
        self.assertIn("absolute performance number", problems)
        self.assertIn("ticket tags must be bare", problems)

    def test_round_header(self):
        """A stale first-round draft never goes out as a follow-up: the header's round must match."""

        def body(n, rnd):
            return f"{CFG['review_first_line']}\n\n## PR #{n} — t (a, {TK}-1) — 🟠 {rnd}: x — y\n\n| a | b |\n"

        self.assertEqual(
            post.round_check(body(7, "First-round"), 7, reviewed=False), []
        )
        self.assertEqual(post.round_check(body(7, "Follow-up"), 7, reviewed=True), [])
        self.assertEqual(post.round_check(body(7, "Follow-up 2"), 7, reviewed=True), [])
        self.assertEqual(
            post.round_check(body(7, "First-round re-review"), 7, reviewed=True), []
        )
        self.assertEqual(
            post.round_check(
                body(7, "First-round (posted after the merge)"), 7, reviewed=False
            ),
            [],
        )
        self.assertEqual(
            post.round_check(
                body(7, "Follow-up").replace("🟠", "🟢🟡"), 7, reviewed=True
            ),
            [],
        )
        self.assertIn(
            "already has our review",
            " ".join(post.round_check(body(7, "First-round"), 7, reviewed=True)),
        )
        self.assertIn(
            "not reviewed #7 yet",
            " ".join(post.round_check(body(7, "Follow-up"), 7, reviewed=False)),
        )
        self.assertIn(
            "names #8, not #7",
            " ".join(post.round_check(body(8, "Follow-up"), 7, reviewed=True)),
        )
        self.assertIn(
            "no review header",
            " ".join(post.round_check(CFG["review_first_line"] + "\nhi", 7, True)),
        )

    def test_header_tag_must_be_the_prs_own(self):
        """A draft's ticket tag must be the PR's own, never invented or copied from the review book's example."""
        hdr = f"## PR #7 — t (a, {TK}-2884) — 🟡 Follow-up: x"
        self.assertEqual(post.tag_check(hdr, f"feat: t\n\nISSUE ID : {TK}-2884\n"), [])
        self.assertEqual(post.tag_check("## PR #7 — t (a) — 🟡 Follow-up: x", ""), [])
        self.assertIn(
            f"{TK}-2884 is not in the PR's",
            " ".join(post.tag_check(hdr, f"fix ({TK}-28840) and {TK}-288")),
        )

    def test_post_refuses_an_invented_tag_before_any_write(self):
        from unittest import mock

        st = {"prs": {"7": {"head": "a" * 40, "ours": None}}}

        class Store:
            def ledger_has(self, *a):
                return False

        def rest(path, jq=None, paginate=False, timeout=120):
            if path.endswith("/commits?per_page=100"):
                return ["feat: t", "fix: review comments"]
            return (
                {"title": "feat: t", "body": "no tag here"}
                if jq == "{title, body}"
                else "a" * 40
            )

        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "p.md"
            f.write_text(
                f"{CFG['review_first_line']}\n\n## PR #7 — t (a, {TK}-2299) — 🟡 First-round: x — z\n",
                encoding="utf-8",
            )
            with mock.patch("rkmon_lib.gh.rest", rest), self.assertRaises(
                post.PostRefused
            ) as cm:
                post.post(REVIEWING, st, Store(), 7, str(f))
        self.assertIn(f"{TK}-2299 is not in the PR's", str(cm.exception))

    def test_post_refuses_a_wrong_round_before_any_write(self):
        st = {
            "prs": {
                "7": {"head": "a" * 40, "ours": {"verdict": "🟡", "head": "b" * 40}}
            }
        }
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "p.md"
            f.write_text(
                f"{CFG['review_first_line']}\n\n## PR #7 — t — 🟠 First-round: stale — z\n",
                encoding="utf-8",
            )
            with self.assertRaises(post.PostRefused) as cm:
                post.post(
                    REVIEWING, st, None, 7, str(f)
                )  # store None: refused before the ledger or GitHub is touched
        self.assertIn("already has our review", str(cm.exception))


class TestRerecordAfterFindings(unittest.TestCase):
    """A re-bless after our 🟠 is the fix, so it is reviewed, never skipped as rebase+re-record."""

    def run_start(self, verdict, kind):
        from unittest import mock

        rec = {
            "n": 9,
            "head": "b" * 40,
            "ours": {
                "head": "a" * 40,
                "at": "2026-10-29T00:00:00Z",
                "verdict": verdict,
            },
        }
        st = {"prs": {"9": rec}}
        c = {
            "kind": kind,
            "old": "a" * 40,
            "new": "b" * 40,
            "changed_files": ["g/x.json"],
            "n_changed": 1,
        }
        store = mock.Mock()
        with mock.patch.object(review, "classify", return_value=c), mock.patch.object(
            review, "_changed_files", return_value=[]
        ), mock.patch.object(
            review, "worker_script", return_value="true"
        ), mock.patch.object(
            review.gh, "rest", return_value={"state": "open", "head": "b" * 40}
        ), mock.patch.object(
            review.jobs, "start", return_value={"id": "w"}
        ):
            out = review.start_review(CFG, st, store, 9, "follow-up")
        return out, rec

    def test_a_rerecord_after_an_orange_review_is_reviewed(self):
        out, rec = self.run_start("\U0001f7e0", "rebase+re-record")
        self.assertTrue(out[-1].startswith("REVIEW_START #9"), out)
        self.assertNotIn("equivalent_heads", rec["ours"])

    def test_a_rerecord_after_a_clean_review_is_skipped(self):
        out, rec = self.run_start("\U0001f7e1", "rebase+re-record")
        self.assertTrue(out[0].startswith("REVIEW_SKIP #9"), out)
        self.assertEqual(rec["ours"]["equivalent_heads"], ["b" * 40])

    def test_a_pure_rebase_after_an_orange_review_is_still_skipped(self):
        out, rec = self.run_start("\U0001f7e0", "rebase-only")
        self.assertTrue(out[0].startswith("REVIEW_SKIP #9"), out)


class TestHotSpotDirectories(unittest.TestCase):
    """A PR that turns the representative golden into per-family shards: RG lists the file and the directory."""

    def test_a_directory_hot_spot_matches_its_shards(self):
        from rkmon_lib import queue

        cfg = dict(CFG, hot_spots={"RG": ["rk/g.json", "rk/g/"], "DG": "rk/d.json"})
        self.assertEqual(queue.hot_paths(cfg, "RG"), ["rk/g.json", "rk/g/"])
        self.assertEqual(queue.hot_paths(cfg, "DG"), ["rk/d.json"])
        self.assertTrue(queue.path_hits(queue.hot_paths(cfg, "RG"), "rk/g/gemm.json"))
        self.assertTrue(queue.path_hits(queue.hot_paths(cfg, "RG"), "rk/g.json"))
        self.assertFalse(queue.path_hits(queue.hot_paths(cfg, "RG"), "rk/g.jsonx"))
        self.assertFalse(queue.path_hits(queue.hot_paths(cfg, "DG"), "rk/d.json/x"))


# The checks' folders: the monitor's out, the working folder, develop's commit, the tmp folder, and the report.
FOLDER_VAR = re.compile(r"\$(OUT|D|TD|TP|R)\b|\$\{(OUT|D|TD|TP|R)[}:]")


def unquoted_uses(line: str) -> list[str]:
    """The uses of a FOLDER_VAR variable in a shell line outside double quotes, as bash reads it: single quotes are
    literal, and a $( ) inside double quotes starts a context of its own."""
    found, stack, i = [], ["top"], 0
    while i < len(line):
        c, ctx = line[i], stack[-1]
        if ctx == "single":
            if c == "'":
                stack.pop()
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == "'" and ctx != "double":
            stack.append("single")
        elif c == '"':
            if ctx == "double":
                stack.pop()
            else:
                stack.append("double")
        elif line.startswith("$(", i):
            stack.append("sub")
            i += 2
            continue
        elif c == ")" and ctx == "sub":
            stack.pop()
        elif c == "$" and ctx != "double":
            m = FOLDER_VAR.match(line, i)
            if m:
                found.append(m.group(0))
        i += 1
    return found


class TestChecksPlan(unittest.TestCase):
    FAMS = {
        "gfx942_attention_tiled_2d": "library/tests/parity",
        "attention_bwd_lds": "platform/tests/instances/parity",
        "conv_implicit_gemm": "platform/tests/instances/parity",
    }

    def test_families(self):
        files = [
            "library/kernels/gfx942/attention_tiled_2d.py",
            "platform/cpp/instances/attention_bwd/lds.cpp",
        ]
        self.assertEqual(
            review.touched_families(files, self.FAMS),
            ["attention_bwd_lds", "gfx942_attention_tiled_2d"],
        )
        self.assertEqual(
            review.touched_families(
                ["library/kernels/gfx950/attention_tiled_2d.py"], self.FAMS
            ),
            [],
        )

    def test_a_capped_test_list_says_so(self):
        """A capped test list says so: the checks name the changed test files they did not run."""
        files = [f"rk/library/tests/test_t{i:02d}.py" for i in range(53)]
        plan = review.plan_checks(CFG, {}, files, self.FAMS)
        self.assertEqual((len(plan["tests"]), plan["tests_total"]), (40, 53))
        plan["_dirs"] = {}
        script = review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None)
        self.assertEqual(
            script.count("ran 40 of 53 changed test files (cap 40)"), 2
        )  # merged and head-only sections
        r = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        small = review.plan_checks(CFG, {}, files[:3], self.FAMS)
        small["_dirs"] = {}
        self.assertNotIn(
            "changed test files (cap",
            review.checks_script(CFG, 1, "a" * 40, "follow-up", small, None),
        )

    def test_the_prs_changed_tests_run_in_the_installed_tree(self):
        """L-526 (L-142): a test that reads provider pack headers through Path(__file__).parents[3] fails CI's
        installed rocke_library_attention_cpu_pytest while it passes in the source layout. The PR's own test
        files (uncapped; companions are not the PR's) go to tools/installed_tests.py, after this run's native build,
        in the merged-tree part, in review and gate scripts alike. The ids develop's cache lacks run in a develop job
        of their own, in a slot of its own, and the monitor, never PR code, adds that job's outcomes to the cache.
        """
        files = [
            "rk/library/tests/test_new.py",
            "rk/library/tests/test_gone.py",
            "rk/platform/CMakeLists.txt",
        ]
        plan = review.plan_checks(CFG, {}, files, self.FAMS)
        self.assertEqual(
            plan["changed_tests"],
            ["library/tests/test_gone.py", "library/tests/test_new.py"],
        )
        self.assertIn(
            "platform/tests/test_engine_linkage.py", plan["tests"]
        )  # a companion: the tests section runs it
        many = review.plan_checks(
            CFG,
            {},
            [f"rk/library/tests/test_t{i:02d}.py" for i in range(53)],
            self.FAMS,
        )
        self.assertEqual(
            len(many["changed_tests"]), 53
        )  # uncapped: over 40, the tool runs the entries whole
        plan["_dirs"] = {}
        for mode in ("review", "gate"):
            script = review.checks_script(
                CFG, 1, "a" * 40, "follow-up", plan, None, mode=mode
            )
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n"], input=script, capture_output=True, text=True
                ).returncode,
                0,
            )
            lines = script.splitlines()
            sec = lines.index(
                'sec "Changed tests in the installed tree (as CI runs them)"'
            )
            self.assertGreater(
                sec, lines.index('if [ "$MERGED" = 1 ]; then')
            )  # no merged tree, no install
            self.assertGreater(
                sec,
                next(i for i, ln in enumerate(lines) if "build_native.py ${WTN}" in ln),
            )
            body = "\n".join(lines[sec : sec + 7])
            tool = (
                "prc nice -n 10 timeout 2400 python3 -I -B $M/tools/installed_tests.py"
            )
            self.assertIn(
                f'inst() {{ o=$1; shift; {prcode.RESULTS_VAR}="$OUT/installed" {tool} --out $SO/$o --tree $W/$K '
                "--build $SB/${WTN}/core --prefix "
                "$SB/${WTN}/inst --build-ok $SI/native_ok --grace $M/tools/worker_grace.py --dev-tree $IT "
                '--cache "$IC" "$@" --budget 1200 library/tests/test_gone.py library/tests/test_new.py > '
                '"$OUT/installed_report.txt" 2>&1; }',
                body,
            )
            self.assertIn(
                '(prslot || exit 3; prin "$OUT/installed_todo/develop_todo.tsv"; '
                f"{tool} develop --out $SO/develop --todo $SI/develop_todo.tsv --dev-tree $IT "
                f'--dev-sha "$DS" --dev-build "python3 -B {paths.worker_tool("build_native.py")} wtbase" '
                "--dev-core $SB/wtbase/core --dev-prefix $SB/wtbase/inst --grace $M/tools/worker_grace.py "
                '--min-free-gb 16 --budget 1200 > "$OUT/installed_develop.txt" 2>&1; '
                'prr develop "$OUT/installed_develop"; prfree)',
                body,
            )
            # develop's outcomes at the same develop SHA: PR code reads the cache; the monitor alone writes it.
            self.assertIn(
                '[ -n "$DS" ] && mkdir -p $CP && IC=$CP/installed_develop_${DS:0:12}.tsv',
                body,
            )
            self.assertIn(
                'DO="$OUT/installed_develop/develop_outcomes.tsv"; python3 -I -B $M/tools/installed_tests.py '
                'cache-add $DO "$IC"',
                body,
            )
            self.assertEqual(body.count("installed_tests.py cache-add"), 1)
            self.assertNotIn(
                "prc python3 -I -B $M/tools/installed_tests.py cache-add", body
            )
        # native_ok marks this run's own build: an earlier build is never installed.
        self.assertIn("then echo 'native build OK'; : > \"$OUT/native_ok\";", script)
        plan = review.plan_checks(
            CFG, {}, ["rk/library/kernels/gfx942/attention_tiled_2d.py"], self.FAMS
        )
        plan["_dirs"] = {f: self.FAMS[f] for f in plan["families"]}
        self.assertNotIn(
            "installed_tests.py",
            review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None),
        )

    def installed_section(self, tool_body: str) -> tuple[list[str], pathlib.Path]:
        """The checks' installed section run on its own, in host slots (prcode.SHELL with no launcher), with
        `tool_body` as tools/installed_tests.py: its output and the folder it ran in."""
        plan = review.plan_checks(CFG, {}, ["rk/library/tests/test_new.py"], self.FAMS)
        plan["_dirs"] = {}
        lines = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", plan, None
        ).splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith("base_tree; IT=-;"))
        j = next(
            k for k, ln in enumerate(lines) if ln.startswith("i=$(grep '^INSTALLED '")
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d, root = tmp.name, pathlib.Path(tmp.name)
        (root / "tools").mkdir()
        (root / "tools" / "installed_tests.py").write_text(tool_body)
        (root / "bin").mkdir()
        (root / "bin" / "python").write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'
        )
        (root / "bin" / "python").chmod(0o755)
        r = subprocess.run(
            [
                "bash",
                "-c",
                f"{prcode.SHELL}\n{review.RESULT_FN}\n{review.U8_FN}\nPATH={root}/bin:$PATH; "
                f'st() {{ echo "ST $*"; }}; TMPDIR={d}; K=rk; W={d}/wt; BW={d}/wtbase; '
                f"base_tree() {{ BR=x; }}; BR=''; D={d}; M={d}; T={d}/tools; CA={d}/cache; "
                f"CP={d}/cache/develop; OUT={d}/out; mkdir $OUT; echo {'d' * 40} > $OUT/develop_sha; "
                "prslot || exit 9\n" + "\n".join(lines[i : j + 1]),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return r.stdout.splitlines(), root

    def test_the_installed_section_reports_the_tools_last_line_and_never_a_silent_pass(
        self,
    ):
        last = (
            "INSTALLED fail 1 run of 1 changed test files, 3 s: 1 failing (on the merge only): "
            "library/tests/test_new.py::test_x - FileNotFoundError: [Errno 2] No such file or directory"
        )
        row = "rocke_library_attention_cpu_pytest (1 files, 2 s, rc=1): 1 failed"
        out, root = self.installed_section(f"print({row!r})\nprint({last!r})\n")
        self.assertEqual(
            out,
            [
                row,
                "result: FAIL " + last.split(" ", 2)[2],
                "ST installed fail " + last.split(" ", 2)[2][:200],
            ],
        )
        # The tool reads develop's cached outcomes; with nothing for a develop job, nothing is added to the cache.
        self.assertEqual(list((root / "cache" / "develop").iterdir()), [])
        out, _ = self.installed_section(
            "raise SystemExit('Traceback: the tool broke')\n"
        )  # no last line: a failure, never a pass
        self.assertEqual(out[-1], "ST installed fail no result")

    def test_the_ids_the_cache_lacks_run_in_a_develop_job_and_the_monitor_adds_its_outcomes(
        self,
    ):
        """No PR job builds develop to fill a cache. The merge's run lists the ids develop's cache
        lacks; a develop job runs them in a slot of its own; the monitor adds that job's outcomes to the cache; and
        the run is made again with them, its results the ones kept."""
        fake = textwrap.dedent(
            """\
            import json, os, sys
            a = sys.argv[1:]
            def opt(name):
                return a[a.index(name) + 1] if name in a else ""
            if a[0] == "cache-add":
                with open(a[2], "a") as fh:
                    fh.write(open(a[1]).read())
                sys.exit(0)
            out = opt("--out")
            os.makedirs(out)
            if a[0] == "develop":
                with open(os.path.join(out, "develop_outcomes.tsv"), "w") as fh:
                    fh.write(open(opt("--todo")).read().replace("\\n", "\\tfailed\\n"))
                with open(os.path.join(out, "where.txt"), "w") as fh:
                    fh.write(out)
                print("INSTALLED_DEVELOP 1 of 1 outcomes at develop dddddddddd, 1 s")
                sys.exit(0)
            with open(os.path.join(out, "results.json"), "w") as fh:
                json.dump({"out": out, "outcomes": bool(opt("--outcomes"))}, fh)
            if opt("--outcomes"):
                print("INSTALLED fail with develop's: " + open(opt("--outcomes")).read().strip())
            else:
                with open(os.path.join(out, "develop_todo.tsv"), "w") as fh:
                    fh.write("e\\tt.py::x\\n")
                print("INSTALLED fail not compared yet")
            """
        )
        out, root = self.installed_section(fake)
        self.assertIn(
            "develop job: INSTALLED_DEVELOP 1 of 1 outcomes at develop dddddddddd, 1 s",
            out,
        )
        self.assertEqual(
            out[-1], "ST installed fail with develop's: e\tt.py::x\tfailed"
        )
        self.assertEqual(
            (root / "cache/develop/installed_develop_dddddddddddd.tsv").read_text(),
            "e\tt.py::x\tfailed\n",
        )
        kept = json.loads((root / "out/installed/results.json").read_text())
        self.assertTrue(kept["outcomes"])  # the second run's results
        # The develop job ran in a slot of its own, never the merge's.
        job = (root / "out/installed_develop/where.txt").read_text()
        self.assertNotEqual(
            pathlib.Path(job).parents[1], pathlib.Path(kept["out"]).parents[1]
        )

    def test_an_installed_tree_failure_fails_g6(self):
        ok = {
            "merge": ("pass", "clean"),
            "build": ("pass", "100%"),
            "tests": ("pass", "9 passed"),
        }
        g = review.gates_from_status(
            dict(
                ok, installed=("fail", "1 run of 1 changed test files, 3 s: 1 failing")
            )
        )
        self.assertEqual(
            g["G6"],
            ["fail", "installed: 1 run of 1 changed test files, 3 s: 1 failing"],
        )
        g = review.gates_from_status(
            dict(ok, installed=("skip", "NOT RUN, 0 s: no native build of this merge"))
        )
        self.assertEqual(
            g["G6"], ["pass", "merge pass, build pass, tests pass, installed skip"]
        )

    def test_a_capped_compile_family_list_says_so_in_the_report_and_the_status(self):
        """A capped compile family list says so in the report and the status: no family goes uncompiled unsaid."""
        plan = review.plan_checks(
            CFG, {}, ["rk/library/kernels/gfx942/attention_tiled_2d.py"], self.FAMS
        )
        plan["families"] = [f"fam{i}" for i in range(7)]
        plan["_dirs"] = {f: "rk/parity" for f in plan["families"]}
        script = review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None)
        self.assertIn(
            "CAPNOTE='compiled 6 of 7 touched families (cap 6); not compiled here: fam6'",
            script,
        )
        self.assertIn("rk/parity:fam5 ", script)
        self.assertNotIn("rk/parity:fam6", script)
        self.assertEqual(
            script.count("${CAPNOTE:+; $CAPNOTE}"), 3
        )  # every compile status carries it
        r = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        plan["families"] = plan["families"][:2]
        self.assertIn(
            "CAPNOTE=''",
            review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None),
        )

    def test_without_the_llvm_tools_the_compile_lanes_skip_and_g7_stays_pending(self):
        """A blank paths.rocm_llvm_bin does not refuse a review: the monitor runs, the compile section
        says why nothing compiled, and G7 is pending, never passed for a PR nothing compiled for.
        """
        plan = review.plan_checks(
            CFG, {}, ["rk/library/kernels/gfx942/attention_tiled_2d.py"], self.FAMS
        )
        plan["families"], plan["_dirs"] = ["fam0"], {"fam0": "rk/parity"}
        script = review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None)
        self.assertIn(
            f"if [ -z \"$LB\" ]; then echo '   {review.COMPILE_NO_LLVM}'; st compile skip",
            script,
        )
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        lines = script.splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith('if [ -z "$LB" ]'))
        with tempfile.TemporaryDirectory() as d:  # the section itself, without the tools: it skips, and says so
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"OUT={d}; LB=''; st() {{ echo \"$1 $2 $3\" >> $OUT/status; }}\n"
                    + lines[i]
                    + " :; fi",
                ],
                capture_output=True,
                text=True,
            )
            status = (pathlib.Path(d) / "status").read_text()
        self.assertIn(review.COMPILE_NO_LLVM, r.stdout)
        self.assertEqual(status, f"compile skip {review.COMPILE_NO_LLVM}\n")
        g = review.gates_from_status(
            {"merge": ("pass", "clean"), "compile": ("skip", review.COMPILE_NO_LLVM)}
        )
        self.assertEqual(g["G7"][0], "pending")
        self.assertIn("no ROCm LLVM tools", g["G7"][1])
        self.assertEqual(
            review.gates_from_status({"merge": ("pass", "clean")})["G7"][0], "pass"
        )  # nothing to compile

    def test_flavor_loops_follow_the_trees_emission_columns_and_the_golden_runs_once(
        self,
    ):
        """The flavor loops follow the tree's emission columns (llvm24 among them), never a fixed llvm20/22/23 list,
        and the golden gate runs once (L-325, L-523)."""
        from rkmon_lib import guard

        files = [
            "rk/library/kernels/gfx942/attention_tiled_2d.py",
            "rk/platform/cpp/core/ir.c",
        ]
        plan = review.plan_checks(CFG, {}, files, self.FAMS)
        plan["_dirs"] = {f: self.FAMS[f] for f in plan["families"]}
        sanity = guard.sanity_script(CFG, "b" * 40, "/w/wt", "/w/out", ["gemm"], False)
        for script in (
            review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None),
            sanity,
        ):
            self.assertNotIn("for FL in llvm20 llvm22 llvm23", script)
            self.assertIn(review.FLAVORS_FN, script)
            self.assertIn("FLS=$(flavors); for FL in $FLS; do", script)
            self.assertNotIn("rocke_installed_golden_test.py", script)
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n"], input=script, capture_output=True, text=True
                ).returncode,
                0,
            )
        # The report reads the gate's results copied back into a folder of their own.
        self.assertIn(
            'python3 -I -B $M/tools/golden_vs_base.py report "$OUT/golden" $P/tests $GB "$GC" $GDO',
            review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None),
        )
        self.assertIn(
            "python3 -I -B $M/tools/golden_vs_base.py report $OUT/golden $P/tests self",
            sanity,
        )
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "bin").mkdir()
            (root / "bin" / "python").symlink_to(sys.executable)
            core = root / "new" / "python" / "rocke" / "core"
            core.mkdir(parents=True)
            for p in (core.parent, core):
                (p / "__init__.py").write_text("")
            (core / "llvm_profile.py").write_text(
                'LLVM_EMISSION_FLAVORS = ("llvm20", "llvm22", "llvm23", "llvm24")\n'
            )
            (root / "old" / "python").mkdir(parents=True)

            def flavors(tree):
                env = {"PATH": f"{root / 'bin'}:/usr/bin:/bin", "TMPDIR": d}
                script = f"{prcode.SHELL}\nP={root / tree}\n{review.FLAVORS_FN}\nprslot || exit 9; flavors"
                return subprocess.run(
                    ["bash", "-c", script],
                    capture_output=True,
                    text=True,
                    env=env,
                ).stdout.strip()

            self.assertEqual(flavors("new"), "llvm20 llvm22 llvm23 llvm24")
            self.assertEqual(
                flavors("old"), "llvm20 llvm22 llvm23"
            )  # no profile module: the three lanes

    def test_a_cmake_registration_change_runs_the_registration_contract_test(self):
        """A new include in platform/tests/CMakeLists.txt runs the registration contract test."""
        reg = "platform/tests/tools/test_cmake_test_registration.py"
        for f in (
            "rk/platform/tests/CMakeLists.txt",
            "rk/platform/tests/cmake/NativeGoldenTests.cmake",
        ):
            self.assertIn(reg, review.plan_checks(CFG, {}, [f], self.FAMS)["tests"], f)
        self.assertNotIn(
            reg,
            review.plan_checks(CFG, {}, ["rk/platform/CMakeLists.txt"], self.FAMS)[
                "tests"
            ],
        )
        both = review.plan_checks(
            CFG, {}, ["rk/platform/tests/CMakeLists.txt", "rk/" + reg], self.FAMS
        )["tests"]
        self.assertEqual(both.count(reg), 1)

    def test_a_platform_cmake_change_runs_the_installed_ctest_contract_test(self):
        """A change to platform/CMakeLists.txt, which writes the installed CTest entries and their ENVIRONMENT, runs
        the installed CTest contract test."""
        link = "platform/tests/test_engine_linkage.py"
        self.assertIn(
            link,
            review.plan_checks(CFG, {}, ["rk/platform/CMakeLists.txt"], self.FAMS)[
                "tests"
            ],
        )
        for f in (
            "rk/platform/tests/CMakeLists.txt",
            "rk/library/CMakeLists.txt",
            "rk/platform/cmake/Foo.cmake",
        ):
            self.assertNotIn(
                link, review.plan_checks(CFG, {}, [f], self.FAMS)["tests"], f
            )

    def test_the_test_cap_never_cuts_a_companion_contract_test(self):
        """The test cap keeps every companion contract test, not only the first files by name."""
        many = [
            f"rk/library/tests/test_a{i:02d}.py" for i in range(review.TESTS_CAP + 5)
        ]
        plan = review.plan_checks(
            CFG, {}, many + ["rk/platform/CMakeLists.txt"], self.FAMS
        )
        self.assertEqual(len(plan["tests"]), review.TESTS_CAP)
        self.assertEqual(plan["tests_total"], review.TESTS_CAP + 5 + 3)
        for t in (
            "platform/tests/test_engine_linkage.py",
            "platform/tests/tools/test_arch_domain_artifact.py",
            "platform/tests/tools/test_run_checks.py",
        ):
            self.assertIn(t, plan["tests"])

    def test_a_platform_cmake_change_runs_the_tests_that_read_its_text(self):
        """A change to platform/CMakeLists.txt (a cmake-lint restyle) runs the develop tests that read its text,
        which no CI lane runs."""
        tests = review.plan_checks(CFG, {}, ["rk/platform/CMakeLists.txt"], self.FAMS)[
            "tests"
        ]
        for t in (
            "platform/tests/tools/test_arch_domain_artifact.py",
            "platform/tests/tools/test_run_checks.py",
        ):
            self.assertIn(t, tests)
        self.assertNotIn(
            "platform/tests/tools/test_run_checks.py",
            review.plan_checks(
                CFG, {}, ["rk/platform/tests/CMakeLists.txt"], self.FAMS
            )["tests"],
        )

    def test_a_gfx1250_lowering_config_change_runs_the_gate_verdict_census(self):
        """New gfx1250_lowering parity configs run the gate-verdict census, which no CI lane collects
        (tests/instances/differential/)."""
        gate = "platform/tests/instances/differential/test_gate_verdict.py"
        plan = lambda f: review.plan_checks(CFG, {}, ["rk/" + f], self.FAMS)["tests"]
        for f in (
            "platform/tests/instances/parity/gfx1250_lowering_emit.py",
            "platform/tests/instances/parity/gfx1250_lowering_emit.c",
            "platform/tests/instances/differential/run_diff.py",
            "platform/tests/instances/differential/ir_artifact_diff.py",
        ):
            self.assertIn(gate, plan(f), f)
        for f in (
            "platform/tests/instances/parity/gemm_emit.py",
            "platform/tests/instances/differential/numeric.py",
        ):
            self.assertNotIn(gate, plan(f), f)

    def test_an_arch_data_or_decl_table_change_runs_the_arch_domain_tests(self):
        """A change to an arch data or decl table (a dense WMMA decl key) runs the arch-domain tests."""
        dom, ident = (
            "platform/tests/tools/test_arch_domain_artifact.py",
            "platform/tests/core/test_arch_target_identity.py",
        )
        plan = lambda f: review.plan_checks(CFG, {}, [f], self.FAMS)["tests"]
        specs = plan("rk/platform/python/rocke/core/arch/data/arch_specs.json")
        self.assertIn(dom, specs)
        self.assertIn(ident, specs)
        for f in (
            "rk/platform/python/rocke/core/arch/data/intrinsic_arch_domain.llvm24.json",
            "rk/platform/python/rocke/core/lower_llvm.py",
            "rk/platform/cpp/core/lower_llvm/data.cpp",
        ):
            self.assertIn(dom, plan(f), f)
            self.assertNotIn(ident, plan(f), f)
        self.assertNotIn(
            dom, plan("rk/platform/python/rocke/helpers/tiling/fragments.py")
        )

    def test_companion_tests_for_dispatch_and_gdn_decode_changes(self):
        """L-538: #56789 changed the conv dispatch policy and its instance lists, and the checks never ran the reference
        contract test CI runs; #34567 fu5's GDN case-set change needed the C++/Python parity test, in no CI lane, by
        hand."""
        conv, gdn = (
            "library/tests/test_conv_reference_contract.py",
            "library/tests/test_gdn_decode_ir_cpp_parity.py",
        )
        plan = lambda f: review.plan_checks(CFG, {}, ["rk/" + f], self.FAMS)["tests"]
        for f in (
            "library/dispatch/grouped_convolution.py",
            "library/dispatch/conv_winograd_plan.py",
            "library/dispatch/grouped_convolution_instances/gfx950_dgrad.json",
        ):
            self.assertIn(conv, plan(f), f)
            self.assertNotIn(gdn, plan(f), f)
        for f in (
            "library/dispatch/gdn/rules.py",
            "library/dispatch/gdn/__init__.py",
            "library/kernels/gfx950/gdn_decode.py",
            "library/tests/test_gdn_decode_golden.py",
            "library/tests/golden/gdn_decode_gfx950_ir_sha256.json",
        ):
            self.assertIn(gdn, plan(f), f)
            self.assertNotIn(conv, plan(f), f)
        for f in (
            "library/dispatch/attention/gfx950_dense.py",
            "library/dispatch/kda/gfx950.py",
            "library/kernels/gfx950/gdn_decode_extra.py",
            "library/tests/test_gdn_decode_fused.py",
            "library/dispatch/grouped_convolution.md",
            "platform/python/rocke/core/lower_hip.py",
        ):
            self.assertFalse({conv, gdn} & set(plan(f)), f)
        both = review.plan_checks(
            CFG, {}, ["rk/library/dispatch/gdn/rules.py", "rk/" + gdn], self.FAMS
        )["tests"]
        self.assertEqual(
            both.count(gdn), 1
        )  # a PR that changes the companion too runs it once

    def test_script_is_valid_bash(self):
        files = [
            "rk/library/kernels/gfx942/attention_tiled_2d.py",
            "rk/library/tests/test_attention_x.py",
            "rk/platform/cpp/core/ir.c",
            "projects/hipdnn/x.cpp",
        ]
        plan = review.plan_checks(CFG, {}, files, self.FAMS)
        self.assertTrue(plan["build"] and plan["engine_wide"] and plan["golden"])
        self.assertEqual(plan["outside_rocke"], ["projects/hipdnn"])
        plan["_dirs"] = {f: self.FAMS[f] for f in plan["families"]}
        script = review.checks_script(
            CFG, 10303, "a" * 40, "follow-up", plan, "2026-10-27T00:00:00Z"
        )
        r = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--only gfx942_attention_tiled_2d", script)
        self.assertIn("check_byte_identity.py --build-root", script)  # engine-wide run
        # The PR's tests all run, and their failures are compared with develop (no pytest -x).
        self.assertIn("python3 -I -B $M/tools/pytest_vs_base.py", script)
        self.assertNotRegex(script, r"pytest [^\n]* -x\b")
        # Only test files that exist run: the PR may delete one.
        self.assertIn("not run (the PR deletes it)", script)
        self.assertIn("pytest_vs_base.py", script)
        self.assertRegex(script, r"library \$TS \$CT >")
        # The PR body is scanned for perf numbers: a squash merge commits it.
        self.assertIn(
            "sec 'PR body (a squash merge commits it to develop): perf numbers'", script
        )
        # An armed auto-merge's stored title and message are compared with the live ones (L-136).
        self.assertIn("tools/automerge_msg.py " + CFG["repo"] + " $N", script)
        # With no merged tree, the PR's tests still run at head against the merge base, and feed no gate.
        self.assertIn("sec 'Tests the PR adds or changes, at head", script)
        self.assertIn(
            'WH="$D/wt$N"; MBS=$(git -C $R0 merge-base $BREF ' + "a" * 40, script
        )
        self.assertIn("st head_tests", script)
        # The ABI section compares struct layouts (a mid-struct field is an ABI change).
        self.assertIn("tools/abi_scan.py $R0 $K " + "a" * 40, script)
        # Our open findings' saved probes re-run on the merge, or the head without one (L-525), unless a
        # draft ran them all on that very tree.
        self.assertIn(
            'PT=$W; [ "$MERGED" = 1 ] || PT="$D/wt$N"; if [ "$MERGED" = 1 ]; then echo "on the merge ',
            script,
        )
        self.assertIn('PH="# probes of pr$N at', script)
        self.assertIn(
            "else PROBE_DETAIL=1 prc python3 -I -B $M/tools/run_probes.py $PB/pr$N $PT; fi; ",
            script,
        )
        self.assertIn(
            "PROBE_DETAIL=1 prc python3 -I -B $M/tools/run_probes.py $PB/pr$N \"$D/wt$N\" | sed 's/^/   /'",
            script,
        )
        self.assertIn("sed -n '/^## Description/,$p' $WK/pr$N/pr/meta.md", script)
        gate = review.checks_script(
            CFG, 10303, "a" * 40, "follow-up", plan, None, mode="gate"
        )
        self.assertNotIn("PR body", gate)
        self.assertNotIn(
            "Tests the PR adds or changes, at head", gate
        )  # a merge gate needs the merge
        self.assertNotIn("run_probes.py", gate)
        # Each merge section reads its own mkwt log (never the review's stale conflict list);
        # a batch step merges in its own script and reads none.
        self.assertIn("$WK/pr$N/mkwt.log 2>/dev/null", script)
        self.assertIn("$WK/pr$N/gate_mkwt.log 2>/dev/null", gate)
        self.assertNotIn("/mkwt.log", gate)
        step = review.checks_script(
            CFG,
            10303,
            "a" * 40,
            "batch step 1",
            plan,
            None,
            mode="gate",
            wt="wtbatch",
            out="/w/x",
            base_sha="b" * 40,
        )
        self.assertNotIn("mkwt.log", step)
        for s in (gate, step):
            r = subprocess.run(["bash", "-n"], input=s, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_shared_helper_selects_the_families_that_use_it(self):
        """A change to a shared helper (conv_fwd_wmma_gate) selects the families that call it for byte identity."""
        fams = {
            f: "library/tests/parity"
            for f in (
                "conv_direct_fwd_wmma",
                "conv_fwd_small_wmma",
                "conv_direct_grouped",
                "conv_direct_dgrad_wmma",
            )
        }
        changed = [
            "rk/library/kernels/common/conv_fwd_wmma_gate.py",
            "rk/platform/cpp/include/rocke/instance_conv_fwd_wmma_gate_internal.h",
        ]
        with tempfile.TemporaryDirectory() as d:
            k = pathlib.Path(d) / "rk"
            for f, text in {
                "library/kernels/common/conv_direct_fwd_wmma.py": "from .conv_fwd_wmma_gate import wmma_fwd_gate\n",
                "library/kernels/common/conv_fwd_small_wmma.py": "from . import conv_fwd_wmma_gate\n",
                "library/kernels/common/conv_direct_grouped.py": "def one_padding_reason(p):\n    pass\n",
                "library/kernels/common/conv_direct_dgrad_wmma.py": "x = 1\n",
                "platform/cpp/instances/common/conv_direct_dgrad_wmma.cpp": '#include "rocke/instance_conv_fwd_wmma_gate_internal.h"\n',
                "library/kernels/common/conv_fwd_wmma_gate.py": "def wmma_fwd_gate(p):\n    pass\n",
            }.items():
                (k / f).parent.mkdir(parents=True, exist_ok=True)
                (k / f).write_text(text)
            plan = review.plan_checks(CFG, {}, changed, fams, source=d)
            names_only = review.plan_checks(CFG, {}, changed, fams)
            # The same from git at a commit, as gate jobs and develop sanity plan (no worktree yet)
            subprocess.run(["git", "init", "-q", d], check=True)
            subprocess.run(["git", "-C", d, "add", "rk"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    d,
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "commit",
                    "-qm",
                    "c",
                ],
                check=True,
            )
            sha = subprocess.run(
                ["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            from_git = review.plan_checks(
                CFG, {}, changed, fams, source=("git", d, sha)
            )
        self.assertEqual(
            (from_git["families"], from_git["helpers"]),
            (plan["families"], plan["helpers"]),
        )
        self.assertEqual(
            plan["families"],
            ["conv_direct_dgrad_wmma", "conv_direct_fwd_wmma", "conv_fwd_small_wmma"],
        )
        self.assertEqual(
            plan["helpers"],
            {
                "conv_fwd_wmma_gate": ["conv_direct_fwd_wmma", "conv_fwd_small_wmma"],
                "instance_conv_fwd_wmma_gate_internal": ["conv_direct_dgrad_wmma"],
            },
        )
        self.assertFalse(plan["engine_wide"])
        self.assertEqual(
            names_only["families"], []
        )  # without the tree: names only, as before
        plan["_dirs"] = {f: fams[f] for f in plan["families"]}
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        self.assertIn(
            "helpers=conv_fwd_wmma_gate->conv_direct_fwd_wmma+conv_fwd_small_wmma;",
            script,
        )

    def test_a_family_owns_files_its_name_starts(self):
        match = review._family_match("gemm", anchored=True)
        self.assertFalse(
            match("library/kernels/common/conv_implicit_gemm.py")
        )  # conv_implicit_gemm's, not gemm's
        self.assertTrue(
            match("library/kernels/common/gemm.py")
            and match("library/tests/parity/gemm_emit.c")
        )
        self.assertTrue(
            review._family_match("attention_bwd_lds", anchored=True)(
                "platform/cpp/instances/attention_bwd/lds.cpp"
            )
        )

    def test_a_helper_many_families_share_checks_every_family(self):
        fams = {f"fam{i}_kernel": "library/tests/parity" for i in range(10)}
        with tempfile.TemporaryDirectory() as d:
            k = pathlib.Path(d) / "rk/library/kernels/common"
            k.mkdir(parents=True)
            for i in range(10):
                (k / f"fam{i}_kernel.py").write_text(
                    "from .shared_epilogue import store\n"
                )
            plan = review.plan_checks(
                CFG,
                {},
                ["rk/library/kernels/common/shared_epilogue.py"],
                fams,
                source=d,
            )
        self.assertTrue(plan["engine_wide"] and plan["build"] and plan["golden"])
        self.assertEqual(plan["families"], [])

    def test_a_conftest_change_runs_the_tests_it_governs(self):
        """A change to library/tests/conftest.py alone runs the tests it governs."""
        plan = review.plan_checks(CFG, {}, ["rk/library/tests/conftest.py"], self.FAMS)
        self.assertEqual(
            (plan["tests"], plan["conftests"]), ([], ["library/tests/conftest.py"])
        )
        plan["_dirs"] = {}
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        line = next(ln for ln in script.splitlines() if ln.startswith("CT=''"))
        with tempfile.TemporaryDirectory() as d:
            t = pathlib.Path(d) / "rk/library/tests"
            t.mkdir(parents=True)
            (t / "conftest.py").write_text(
                'BWD = ("test_attention_bwd", "test_attention_dense_bwd")\n'
            )
            for name in (
                "test_attention_bwd_a.py",
                "test_attention_bwd_b.py",
                "test_attention_dense_bwd_c.py",
                "test_other.py",
            ):
                (t / name).write_text("")
            r = subprocess.run(
                ["bash", "-c", f'W={d}; K=rk\n{line}\necho "$CT"'],
                capture_output=True,
                text=True,
            )
        self.assertEqual(
            r.stdout.split(),
            [
                "library/tests/test_attention_bwd_a.py",
                "library/tests/test_attention_bwd_b.py",
                "library/tests/test_attention_dense_bwd_c.py",
            ],
            r.stderr,
        )
        # No conftest changed: the selection is empty, not one space.
        plan = review.plan_checks(CFG, {}, ["rk/library/tests/test_x.py"], self.FAMS)
        plan["_dirs"] = {}
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        line = next(ln for ln in script.splitlines() if ln.startswith("CT=''"))
        r = subprocess.run(
            ["bash", "-c", f"W=/nonexistent; K=rk\n{line}\nprintf '[%s]' \"$CT\""],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.stdout, "[]", r.stderr)

    def test_an_engine_wide_change_checks_every_family_at_every_flavor(self):
        """The every-family byte identity runs at every flavor it names, not the host's flavor only."""
        plan = review.plan_checks(
            CFG, {}, ["rk/platform/cpp/core/ir/ir_flow.cpp"], self.FAMS
        )
        self.assertTrue(plan["engine_wide"])
        plan["_dirs"] = {}
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        line = next(ln for ln in script.splitlines() if "bi_all_$FL.log" in ln)
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "tools").mkdir()
            # A stub that is GREEN everywhere but llvm23.
            (pathlib.Path(d) / "tools/check_byte_identity.py").write_text(
                "import os\nprint('RESULT RED' if os.environ['ROCKE_LLVM_FLAVOR'] == 'llvm23' else 'RESULT GREEN')\n"
            )
            (pathlib.Path(d) / "bin").mkdir()
            (pathlib.Path(d) / "bin/python").symlink_to(
                sys.executable
            )  # the checks run in a venv that has `python`
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"{prcode.SHELL}\nPATH={d}/bin:$PATH; flavors() {{ echo llvm20 llvm22 llvm23 llvm24; }}; "
                    f'{review.U8_FN}; st() {{ echo "ST $*"; }}; P={d}; OUT={d}; D={d}; BD={d}/build; WTN=wt1; '
                    f"TMPDIR={d}; prslot || exit 9\n{line}",
                ],
                capture_output=True,
                text=True,
            )
        self.assertIn("llvm24, every family: RESULT GREEN", r.stdout, r.stderr)
        self.assertIn("ST byteid_all fail not GREEN at llvm23", r.stdout, r.stderr)

    def run_tests_status(self, changed, ts):
        """The tests section's status line, with a stub pytest_vs_base.py that reports 693 passes."""
        plan = review.plan_checks(CFG, {}, changed, self.FAMS)
        plan["_dirs"] = {}
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        line = next(
            ln
            for ln in script.splitlines()
            if ln.startswith('if [ -z "$(echo $TS $CT)" ]')
        )
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "tools").mkdir()
            (pathlib.Path(d) / "tools/pytest_vs_base.py").write_text(
                "print('TESTS pass 693 passed, 203 skipped')\n"
            )
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"{prcode.SHELL}\n{review.U8_FN}; st() {{ echo \"ST $*\"; }}; M={d}; OUT={d}; TS='{ts}'; "
                    f"CT='library/tests/test_a.py'; TMPDIR={d}; prslot || exit 9\n{line}",
                ],
                capture_output=True,
                text=True,
            )
        return next(
            (ln for ln in r.stdout.splitlines() if ln.startswith("ST ")), r.stderr
        )

    def test_a_conftest_sample_alone_says_its_passes_may_prove_nothing(self):
        """A conftest sample alone says its passes may prove nothing (memo-free files its conftest never acts on)."""
        self.assertEqual(
            self.run_tests_status(["rk/library/tests/conftest.py"], ""),
            "ST tests pass conftest sample by name, may not exercise the change: 693 passed, 203 skipped",
        )
        # The PR's own test files ran too: their passes count, so the status is the plain one.
        self.assertEqual(
            self.run_tests_status(
                ["rk/library/tests/conftest.py", "rk/library/tests/test_x.py"],
                "library/tests/test_x.py",
            ),
            "ST tests pass 693 passed, 203 skipped",
        )

    def test_recheck_reruns_checks_for_a_waiting_draft_and_records_its_gates(self):
        """A recheck re-runs the checks of a waiting draft at the same head, after a fix to the checks."""
        from unittest import mock
        from rkmon_lib.store import empty_state

        st = empty_state()
        st["prs"] = {
            "7": {
                "n": 7,
                "head": "7" * 40,
                "size": 10,
                "ours": {"at": "2026-10-29T06:00:00Z"},
            }
        }
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            self.assertIn(
                "refused", review.recheck(CFG, st, store, 7)[0]
            )  # nothing awaiting verification
            st["review_ready"] = [
                {"pr": 7, "kind": "follow-up", "head": "7" * 40, "at": "x"}
            ]
            with mock.patch.object(review.jobs, "status", return_value="running"):
                st["jobs"] = {"heavy": {"id": "busy"}}
                self.assertIn(
                    "all parallel builds are busy", review.recheck(CFG, st, store, 7)[0]
                )
            st["jobs"] = {}
            with mock.patch.object(
                review, "emitter_families", return_value={}
            ), mock.patch.object(
                review, "trees_gone", return_value=[]
            ), mock.patch.object(
                review, "_changed_files", return_value=["rk/library/tests/test_x.py"]
            ), mock.patch.object(
                review.jobs, "start", return_value={"id": "c1"}
            ) as start:
                self.assertIn("checks started", review.recheck(CFG, st, store, 7)[0])
            self.assertEqual(start.call_args[0][1:3], ("checks", "pr7"))
            self.assertEqual(st["rechecks"], {"7": {"id": "c1"}})
            with mock.patch.object(
                review.jobs, "status", return_value="done"
            ), mock.patch.object(review, "_record_checks_gates") as rec_gates:
                lines = review.finish_rechecks(CFG, st, store)
            self.assertTrue(lines[0].startswith("RECHECK_DONE #7"), lines)
            rec_gates.assert_called_once()
            self.assertEqual(st["rechecks"], {})

    def test_a_merged_tree_gone_with_the_ramdisk_is_made_again_before_use(self):
        """A restart empties the RAM disk: a draft awaiting verification has lost wt<N> and wt<N>m. rk review-recheck,
        rk dispatch-pack and the code quality pass's start make them again with mkwt.py before anything reads them.
        """
        from unittest import mock
        from rkmon_lib import dispatch
        from rkmon_lib.store import empty_state

        with tempfile.TemporaryDirectory() as d:
            work = pathlib.Path(d) / "work"
            (work / "workers").mkdir(parents=True)
            cfg = dict(CFG, paths={"working_dir": str(work)})
            calls = []

            def mkwt(cmd, **kw):  # a fake mkwt.py run: it makes the two trees
                calls.append(cmd[-1])
                for w in ("wt7", "wt7m"):
                    (work / w).mkdir(exist_ok=True)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            self.assertEqual(review.trees_gone(cfg, 7), ["wt7", "wt7m"])
            lines = review.ensure_merge_tree(cfg, {"base": "develop"}, 7, run=mkwt)
            self.assertTrue(
                lines[0].startswith("WORKTREES #7: wt7, wt7m made again with mkwt.py"),
                lines,
            )
            self.assertIn(f"python3 -B {paths.worker_tool('mkwt.py')} $N", calls[0])
            self.assertEqual(
                review.ensure_merge_tree(cfg, {}, 7, run=mkwt), []
            )  # there: nothing runs
            self.assertEqual(len(calls), 1)
            # mkwt.py that fails: said, and rk review-recheck refuses rather than plan from no tree.
            for w in ("wt7", "wt7m"):
                (work / w).rmdir()
            failed = lambda cmd, **kw: subprocess.CompletedProcess(
                cmd, 1, "", "fetch failed"
            )  # noqa: E731
            self.assertIn(
                "did not make wt7, wt7m again (fetch failed)",
                review.ensure_merge_tree(cfg, {}, 7, run=failed)[0],
            )
            st = empty_state()
            st["prs"] = {"7": {"n": 7, "head": "7" * 40, "size": 10, "status": "open"}}
            st["review_ready"] = [
                {"pr": 7, "kind": "first-round", "head": "7" * 40, "at": "x"}
            ]
            store = Store(pathlib.Path(d) / "state")
            with mock.patch.object(
                review, "ensure_merge_tree", return_value=["WORKTREES #7: x"]
            ), mock.patch.object(review.scaler, "room", return_value=(True, "")):
                self.assertEqual(
                    review.recheck(cfg, st, store, 7)[-1],
                    "RECHECK #7: refused: its merged tree is not there",
                )
            # The code quality pass's start: the worker is done and the trees are gone, so its job makes them first.
            (work / "workers" / "pr7" / "pr").mkdir(parents=True)
            (work / "workers" / "pr7" / "pr" / "meta.md").write_text("x")
            pipe = {"pr": 7, "head": "7" * 40, "worker": {"id": "w1"}}
            with mock.patch.object(
                review, "quality_binding", return_value={"binding": "b"}
            ), mock.patch.object(
                review.jobs, "status", return_value="done"
            ), mock.patch.object(
                review.bindings,
                "drafter_settings",
                return_value={"quality_wall_min": 30},
            ), mock.patch.object(
                review.jobs, "start", return_value={"id": "q1"}
            ) as start:
                self.assertIn(
                    "code quality pass started",
                    review.quality_start(cfg, st, store, pipe)[0],
                )
            self.assertIn("mkwt.py", start.call_args[0][3])
            pipe = {"pr": 7, "head": "7" * 40, "worker": {"id": "w1"}}
            with mock.patch.object(
                review, "quality_binding", return_value={"binding": "b"}
            ), mock.patch.object(
                review.jobs, "status", return_value="running"
            ):  # still staging: it waits
                self.assertEqual(review.quality_start(cfg, st, store, pipe), [])
            # rk dispatch-pack makes them before the briefs read them.
            with mock.patch.object(
                dispatch.prep, "prepare", return_value=(["review dir x"], False)
            ), mock.patch.object(
                dispatch.review,
                "ensure_merge_tree",
                return_value=["WORKTREES #7: made"],
            ) as ens, mock.patch.object(
                dispatch, "ci_lines", return_value=[]
            ), mock.patch.object(
                dispatch, "checks_lines", return_value=[]
            ):
                lines, _ok = dispatch.pack(
                    dict(cfg, operator={"github_login": "op"}),
                    st,
                    store,
                    7,
                    "fr",
                    pathlib.Path(d) / "v7fr",
                )
            ens.assert_called_once()
            self.assertIn("WORKTREES #7: made", lines)

    def test_a_settling_head_in_the_last_plan_is_picked_once_it_has_settled(self):
        """A head settling in the last tick's plan is picked once it has settled, before lower-priority first rounds,
        without waiting a tick."""
        from datetime import datetime, timedelta, timezone
        from rkmon_lib.store import empty_state

        def ago(minutes):
            return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        st = empty_state()
        st["prs"] = {
            "1": {"n": 1, "head": "1" * 40, "size": 10, "head_seen_at": ago(40)},
            "2": {"n": 2, "head": "2" * 40, "size": 10, "head_seen_at": ago(500)},
        }
        plan = [
            {
                "kind": "FIRST_ROUND",
                "pr": 1,
                "text": "first-round review of #1 (wait: head is 25 min old)",
            },
            {"kind": "FIRST_ROUND", "pr": 2, "text": "first-round review of #2"},
        ]
        cfg = dict(CFG, settle_min=30)
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            self.assertEqual(
                review.pick_next(cfg, st, store, plan)["pr"], 1
            )  # settled since the plan was made
            st["prs"]["1"]["head_seen_at"] = ago(5)  # still settling
            self.assertEqual(review.pick_next(cfg, st, store, plan)["pr"], 2)

    def test_the_settle_counts_from_the_head_commit_not_from_a_late_tick(self):
        """The settle counts from the head commit, not from a late tick that first sees it (lessons L-518)."""
        from datetime import datetime, timedelta, timezone
        from rkmon_lib.store import head_age_min

        def ago(minutes):
            return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        cfg = dict(CFG, settle_min=30)
        rec = {"n": 1, "head_seen_at": ago(2), "head_committed_at": ago(180)}
        self.assertTrue(review.settled(cfg, {}, rec))
        self.assertGreater(head_age_min(rec), 170)
        rec["head_committed_at"] = ago(5)  # a fresh push still waits out the settle
        self.assertFalse(review.settled(cfg, {}, rec))
        rec["head_committed_at"] = (
            "2099-01-01T00:00:00Z"  # a committer clock ahead counts from the sighting
        )
        self.assertLess(head_age_min(rec), 3)
        rec["head_committed_at"] = None
        self.assertLess(head_age_min(rec), 3)

    def test_a_review_too_large_for_the_autopilot_is_named_not_hidden(self):
        """A review over autopilot_max_lines is never hidden behind "nothing to start now". Once nothing else can
        start, the advance names the first one for the orchestrator to start by hand."""
        from datetime import datetime, timedelta, timezone
        from rkmon_lib.store import empty_state

        seen = (datetime.now(timezone.utc) - timedelta(hours=9)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        st = empty_state()
        st["prs"] = {
            str(n): {
                "n": n,
                "head": str(n) * 40,
                "size": size,
                "head_seen_at": seen,
                "status": "open",
            }
            for n, size in ((1, 38039), (2, 10), (3, 64296))
        }
        st["prs"]["1"]["ours"] = {"head": "0" * 40}  # our review of an earlier head
        st[review.APPROVED] = [review.approval(3)]  # #3's in-depth review, approved
        plan = [
            {"kind": "FOLLOW_UP", "pr": 1, "text": "follow-up on #1"},
            {"kind": "FIRST_ROUND", "pr": 2, "text": "first-round review of #2"},
            {"kind": "FIRST_ROUND", "pr": 3, "text": "first-round review of #3"},
        ]
        cfg = dict(CFG, settle_min=30, autopilot_max_lines=30000)
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            self.assertEqual(
                review.pick_next(cfg, st, store, plan)["pr"], 2
            )  # the normal-size one still goes first
            st["review_ready"] = [{"pr": 2, "head": "2" * 40}]
            self.assertIsNone(review.pick_next(cfg, st, store, plan))
            self.assertEqual(
                review.advance(cfg, st, store, plan),
                [
                    "REVIEW_MANUAL #1 (follow-up, 38039 lines > 30000): too large for the autopilot; start it by hand now "
                    "with `rk review-start 1`, and at its REVIEW_READY brief one light verifier on our open findings and "
                    "the delta (no split, no code quality pass) "
                    "(then #3)"
                ],
            )
            self.assertEqual([b["pr"] for b in review.oversized(cfg, st, plan)], [1, 3])
            st["review_pipeline"] = {
                "pr": 1
            }  # the state before the pipes: it counts as the first pipe
            self.assertEqual([b["pr"] for b in review.oversized(cfg, st, plan)], [3])
            st["prs"]["3"]["ours"] = {
                "head": "3" * 40
            }  # posted since the plan was made
            self.assertEqual(review.oversized(cfg, st, plan, store), [])
            # A head that has not settled is not named yet (a minute after its push).
            st["prs"]["3"].update(
                ours={},
                head_seen_at=(
                    datetime.now(timezone.utc) - timedelta(minutes=1)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            self.assertEqual(review.oversized(cfg, st, plan, store), [])
            st["prs"]["3"]["head_seen_at"] = seen
            self.assertEqual(
                [b["pr"] for b in review.oversized(cfg, st, plan, store)], [3]
            )

    def test_a_manual_review_holds_back_lower_priority_ones(self):
        """A manual review (P1, too large for the autopilot) holds back lower-priority ones."""
        from datetime import datetime, timedelta, timezone
        from rkmon_lib.store import empty_state

        seen = (datetime.now(timezone.utc) - timedelta(hours=9)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        st = empty_state()
        st["prs"] = {
            str(n): {
                "n": n,
                "head": str(n) * 40,
                "size": size,
                "head_seen_at": seen,
                "status": "open",
            }
            for n, size in ((1, 91000), (2, 10), (4, 20))
        }
        st["prs"]["1"]["ours"] = {"head": "0" * 40}  # our review of an earlier head
        plan = [
            {"prio": 1, "kind": "FOLLOW_UP", "pr": 1, "text": "follow-up on #1"},
            {"prio": 1, "kind": "FOLLOW_UP", "pr": 4, "text": "follow-up on #4"},
            {"prio": 4, "kind": "FOLLOW_UP", "pr": 2, "text": "follow-up on #2"},
        ]
        cfg = dict(CFG, settle_min=30, autopilot_max_lines=30000)
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            self.assertEqual(
                review.pick_next(cfg, st, store, plan)["pr"], 4
            )  # the same priority still goes
            st["review_ready"] = [{"pr": 4, "head": "4" * 40}]
            self.assertIsNone(
                review.pick_next(cfg, st, store, plan)
            )  # the P4 waits behind the P1 by hand
            self.assertEqual(
                review.advance(cfg, st, store, plan),
                [
                    "REVIEW_MANUAL #1 (follow-up, 91000 lines > 30000): too large for the autopilot; start it by hand now "
                    "with `rk review-start 1`, and at its REVIEW_READY brief one light verifier on our open findings and "
                    "the delta (no split, no code quality pass)"
                ],
            )
            # Not yet startable (a fresh push): it holds nothing back.
            st["prs"]["1"]["head_seen_at"] = (
                datetime.now(timezone.utc) - timedelta(minutes=1)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.assertEqual(review.pick_next(cfg, st, store, plan)["pr"], 2)
            st["prs"]["1"]["head_seen_at"] = seen
            st["review_skip"] = [1]  # started or skipped by hand: the rest go on
            self.assertEqual(review.pick_next(cfg, st, store, plan)["pr"], 2)

    def test_a_changed_cap_decides_the_next_advance_on_the_same_state(self):
        """autopilot_max_lines takes effect with no restart: on one state, each call judges a 30-line review by the cap
        it is given, and the REVIEW_MANUAL line names the cap that decided."""
        from datetime import datetime, timedelta, timezone
        from rkmon_lib.store import empty_state

        seen = (datetime.now(timezone.utc) - timedelta(hours=9)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        st = empty_state()
        st["prs"] = {
            "1": {
                "n": 1,
                "head": "1" * 40,
                "size": 30,
                "head_seen_at": seen,
                "status": "open",
            }
        }
        st[review.APPROVED] = [review.approval(1)]  # its in-depth review, approved
        plan = [{"kind": "FIRST_ROUND", "pr": 1, "text": "first-round review of #1"}]
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            for cap, big in ((20, True), (40, False), (25, True)):
                cfg = dict(CFG, settle_min=30, autopilot_max_lines=cap)
                with self.subTest(cap=cap):
                    self.assertEqual(
                        review.oversized(cfg, st, plan, store),
                        (
                            [{"pr": 1, "kind": "first-round", "size": 30, "cap": cap}]
                            if big
                            else []
                        ),
                    )
                    self.assertEqual(
                        review.pick_next(cfg, st, store, plan),
                        (
                            None
                            if big
                            else {"pr": 1, "kind": "first-round", "again": False}
                        ),
                    )
                    self.assertEqual(
                        [
                            line.partition(": ")[0]
                            for line in review.manual_lines(cfg, st, store, plan)
                        ],
                        (
                            [f"REVIEW_MANUAL #1 (first-round, 30 lines > {cap})"]
                            if big
                            else []
                        ),
                    )

    def test_ci_at_head_reports_ctest_timeouts(self):
        """CI at head reports CTest timeouts (a test past its limit on every shard), not only ***Failed lines."""
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(CFG, 1, "a" * 40, "first-round", plan, None)
        line = next(ln for ln in script.splitlines() if "ci_sha.py" in ln)
        pattern = line.split("ci_sha.py ", 1)[1].split("'")[1]
        # tools/ci_sha.py reads the pattern as a Python regular expression: the same lines as grep -E.
        log = (
            "25/45 Test #52: rocke_differential_golden_ll ....***Timeout 300.10 sec\n"
            "26/45 Test #53: rocke_differential_golden_canonical ....***Failed   31.38 sec\n"
            "27/45 Test #54: rocke_x ....   Passed    1.00 sec\n"
        )
        r = subprocess.run(
            ["grep", "-E", pattern], input=log, capture_output=True, text=True
        )
        self.assertEqual(len(r.stdout.splitlines()), 2, r.stdout)
        self.assertIn("***Timeout", r.stdout)
        self.assertEqual(
            [ln for ln in log.splitlines() if re.search(pattern, ln)],
            r.stdout.splitlines(),
        )

    def test_ci_at_head_reads_earlier_attempts(self):
        """CI at head reads earlier attempts: failures in attempt 1 behind a hand re-run keep the head red, never
        infra (lessons L-108)."""
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        script = review.checks_script(CFG, 1, "a" * 40, "follow-up", plan, None)
        sec = script.split("sec 'CI at head'", 1)[1].split("sec '", 1)[0]
        self.assertIn(f"tools/ci_attempts.py {'a' * 40} {CFG['repo']}", sec)
        sha_tool = (
            pathlib.Path(review.__file__).resolve().parents[1] / "tools" / "ci_sha.py"
        ).read_text(encoding="utf-8")
        self.assertIn("jobs?filter=all&", sha_tool)

    def test_merge_detection_on_real_merges(self):
        """Only our merge counts, with the PR head as first parent (single PR) or second (batch step). A head that is
        itself the author's merge of develop is never taken for "the merge with develop".
        """
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }

        def git(repo, *a):
            return subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    *a,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        def detect(w, head, base_sha=None, mkwt_develop=None):
            script = review.checks_script(
                CFG, 1, head, "follow-up", plan, None, base_sha=base_sha
            )
            lines = script.splitlines()
            i = next(k for k, ln in enumerate(lines) if ln.startswith("H1="))
            j = next(
                k
                for k in range(i, len(lines))
                if lines[k].endswith('> "$OUT/develop_sha"; fi')
            )
            out = pathlib.Path(w) / "out"
            out.mkdir(exist_ok=True)
            dd = pathlib.Path(w).parent / "D"
            (dd / "workers/pr1").mkdir(parents=True, exist_ok=True)
            log = dd / "workers/pr1/mkwt.log"
            log.write_text(
                f"head=x develop={mkwt_develop} merge-base=y\n" if mkwt_develop else ""
            )
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"W={w}; OUT={out}; D={dd}; WK={dd}/workers; N=1\n"
                    + "\n".join(lines[i : j + 1])
                    + "\necho $MERGED",
                ],
                capture_output=True,
                text=True,
            )
            return (
                r.stdout.strip().splitlines()[-1],
                (out / "head_sha").read_text().strip(),
                (out / "develop_sha").read_text().strip(),
            )

        with tempfile.TemporaryDirectory() as d:
            w = pathlib.Path(d) / "w"
            w.mkdir()
            git(w, "init", "-q", "-b", "main")
            (w / "f").write_text("0\n")
            git(w, "add", "f")
            git(w, "commit", "-qm", "base")
            base = git(w, "rev-parse", "HEAD")
            git(w, "checkout", "-qb", "dev")
            (w / "d").write_text("dev\n")
            git(w, "add", "d")
            git(w, "commit", "-qm", "develop")
            dev = git(w, "rev-parse", "HEAD")
            git(w, "checkout", "-qb", "pr", base)
            (w / "p").write_text("pr\n")
            git(w, "add", "p")
            git(w, "commit", "-qm", "pr")
            git(
                w, "merge", "-q", "--no-ff", "--no-edit", dev
            )  # the author merges develop into the PR
            head = git(w, "rev-parse", "HEAD")
            self.assertEqual(
                detect(w, head)[0], "0"
            )  # our merge failed or never ran: the head is not "the merge"
            git(w, "checkout", "-qb", "dev2", dev)  # develop moves on
            (w / "d2").write_text("dev2\n")
            git(w, "add", "d2")
            git(w, "commit", "-qm", "develop 2")
            dev2 = git(w, "rev-parse", "HEAD")
            git(
                w, "checkout", "-q", "--detach", head
            )  # our merge: develop into the PR head (mkwt, as r@x)
            git(
                w,
                "-c",
                f"user.email={review.MKWT_EMAIL}",
                "merge",
                "-q",
                "--no-ff",
                "--no-edit",
                dev2,
            )
            self.assertEqual(detect(w, head), ("1", head, dev2))
            git(
                w, "checkout", "-q", "--detach", dev2
            )  # a batch step: the PR merged into develop (as rkmon)
            git(
                w,
                "-c",
                f"user.email={review.BATCH_EMAIL}",
                "merge",
                "-q",
                "--no-ff",
                "--no-edit",
                head,
            )
            self.assertEqual(detect(w, head, base_sha=dev2), ("1", head, dev2))
            # The author merged today's develop (0 behind): mkwt's merge is a no-op and the head is the merge (never
            # "no clean merge"). The develop is the one mkwt merged (its log).
            git(w, "checkout", "-q", "--detach", head)
            git(w, "merge", "-q", "--no-ff", "--no-edit", dev2)
            head3 = git(w, "rev-parse", "HEAD")
            self.assertEqual(
                detect(w, head3, mkwt_develop=dev2[:10]), ("1", head3, dev2)
            )
            self.assertEqual(
                detect(w, head3)[0], "0"
            )  # no mkwt log: nothing says which develop was merged
            git(
                w, "checkout", "-qb", "dev3", dev2
            )  # mkwt merged a develop the head does not contain
            (w / "d3").write_text("dev3\n")
            git(w, "add", "d3")
            git(w, "commit", "-qm", "develop 3")
            dev3 = git(w, "rev-parse", "HEAD")
            git(w, "checkout", "-q", "--detach", head3)
            self.assertEqual(detect(w, head3, mkwt_develop=dev3[:10])[0], "0")

    def test_a_head_mismatch_is_said_plainly_never_as_no_clean_merge(self):
        """Checks generated for an old head while mkwt merged the live head say the mismatch plainly: never "no
        clean merge" naming our own merge commit beside 0 conflicting files, never that commit as head_sha, and no
        merged-tree check skipped as if the PR conflicted."""
        plan = {
            "families": [],
            "tests": ["t/test_x.py"],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }

        def git(repo, *a):
            return subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    *a,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        def commit(w, name):
            (w / name).write_text(name + "\n")
            git(w, "add", name)
            git(w, "commit", "-qm", name)
            return git(w, "rev-parse", "HEAD")

        def run(w, d, head, conflicts=()):
            """The merge detection, the Merge with develop verdict, the skipped sections' note and the probes line."""
            script = review.checks_script(CFG, 1, head, "follow-up", plan, None)
            self.assertEqual(
                subprocess.run(
                    ["bash", "-n"], input=script, capture_output=True, text=True
                ).returncode,
                0,
            )
            lines = script.splitlines()
            i = next(k for k, ln in enumerate(lines) if ln.startswith("H1="))
            j = next(
                k
                for k in range(i, len(lines))
                if lines[k].endswith('> "$OUT/develop_sha"; fi')
            )
            pick = [
                ln for ln in lines if ln.startswith(("sec() ", "u8() ", "st() "))
            ] + lines[i : j + 1]
            pick += [
                ln
                for ln in lines
                if ln.startswith(("NC=", 'if [ $MERGED = 1 ]; then echo "merge'))
            ]
            pick += [
                ln for ln in lines if "SKIPPED: $MM" in ln or "run_probes.py" in ln
            ]
            out = pathlib.Path(d) / "out"
            out.mkdir(exist_ok=True)
            (out / "status").write_text("")
            log = pathlib.Path(d) / "workers/pr1/mkwt.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                "head=x develop=y merge-base=z\nbehind/ahead: 3\t2\nconflicting files vs develop:\n"
                + "".join(f + "\n" for f in conflicts)
                + "\n"
            )
            r = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"W={w}; OUT={out}; D={d}; WK={d}/workers; N=1; WTN=wt1m; M=/nonexistent\n"
                    + "\n".join(pick),
                ],
                capture_output=True,
                text=True,
            )
            return (
                r.stdout,
                (out / "status").read_text(),
                (out / "head_sha").read_text().strip(),
            )

        with tempfile.TemporaryDirectory() as d:
            w = pathlib.Path(d) / "w"
            w.mkdir()
            git(w, "init", "-q", "-b", "main")
            base = commit(w, "base")
            git(w, "checkout", "-qb", "pr")
            old = commit(w, "pr1")  # the head the state still had
            new = commit(w, "pr2")  # the live head the worker and mkwt fetched
            git(w, "checkout", "-qb", "dev", base)
            dev = commit(w, "dev")
            git(
                w, "checkout", "-q", "--detach", new
            )  # mkwt: develop merged into the live head, as r@x
            git(
                w,
                "-c",
                f"user.email={review.MKWT_EMAIL}",
                "merge",
                "-q",
                "--no-ff",
                "--no-edit",
                dev,
            )
            merge = git(w, "rev-parse", "HEAD")
            text, status, head_sha = run(w, d, old)
            mm = f"head mismatch: worktree has {new[:10]}, checks expected {old[:10]}"
            self.assertIn(f"{mm}: the worktree is not the head under review", text)
            self.assertIn(f"SKIPPED: {mm}.", text)
            self.assertIn(
                f"not run: {mm}", text
            )  # the probes: wt1 is at the other head too
            self.assertNotIn("no clean merge", text)
            self.assertNotIn("does not merge cleanly", text)
            self.assertEqual(status, f"merge fail {mm}\n")
            self.assertEqual(
                head_sha, old
            )  # the head the checks were for, never our merge commit
            self.assertNotEqual(head_sha, merge)
            # Generated for the live head, the same tree is the merge.
            text, status, head_sha = run(w, d, new)
            self.assertIn(f"merge {merge[:7]}", text)
            self.assertIn(f"= head {new[:10]} + develop {dev[:10]}", text)
            self.assertEqual((status, head_sha), ("merge pass clean\n", new))
            # No merge in the tree (mkwt's merge stopped): "no clean merge" only with a conflicting file to show for it.
            git(w, "checkout", "-q", "--detach", new)
            text, status, head_sha = run(w, d, new, conflicts=["a.py", "b.py"])
            self.assertIn(
                f"no clean merge: head {new[:10]} does not merge into develop", text
            )
            self.assertIn("SKIPPED: the PR does not merge cleanly with develop", text)
            self.assertEqual(
                (status, head_sha), ("merge fail no clean merge with develop\n", new)
            )
            text, status, head_sha = run(w, d, new)
            self.assertIn(
                f"no merged tree: head {new[:10]} has no merge with develop", text
            )
            self.assertNotIn("no clean merge", text)
            self.assertNotIn("does not merge cleanly", text)
            self.assertEqual(
                status, "merge fail no merged tree, and no conflicting file listed\n"
            )

    def test_the_probes_run_once_a_review(self):
        """The re-probed section gives a draft's probe lines when the draft ran every saved probe on this very tree
        (its probes.txt names the tree's commit and the probes, the live folder's or the shadow drafter's), else it
        runs them."""
        plan = {
            "families": [],
            "tests": [],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        cfg = dict(
            CFG,
            roles={"shadow_drafter": "cand"},
            model_bindings={
                "cand": {
                    "provider": "openai",
                    "url": "http://127.0.0.1:9",
                    "model": "Cand-1",
                }
            },
        )
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            cfg["paths"] = dict(
                CFG.get("paths") or {}, root=str(d), workers=str(d / "workers")
            )
            line = next(
                ln
                for ln in review.checks_script(
                    cfg, 1, "a" * 40, "follow-up", plan, None
                ).splitlines()
                if ln.startswith(
                    'if [ -n "$MM" ]; then echo "not run: $MM"; else PT=$W;'
                )
            )
            w = d / "wt1m"
            w.mkdir()
            subprocess.run(["git", "-C", str(w), "init", "-q"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(w),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "commit",
                    "-q",
                    "--allow-empty",
                    "-m",
                    "x",
                ],
                check=True,
            )
            sha = subprocess.run(
                ["git", "-C", str(w), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            (d / "probes" / "pr1").mkdir(parents=True)
            (d / "probes" / "pr1" / "N1_x.sh").write_text(
                'echo "still there"\nexit 1\n'
            )
            subprocess.run(
                ["git", "clone", "-q", str(w), str(d / "wt1")], check=True
            )  # the head alone (wt<N>)
            out_dir = d / "out"
            out_dir.mkdir()
            (out_dir / "head_sha").write_text(sha + "\n")
            (out_dir / "develop_sha").write_text("d" * 40 + "\n")

            def section():
                return subprocess.run(
                    [
                        "bash",
                        "-c",
                        f"{prcode.SHELL}\nMM=''; MERGED=1; W={w}; D={d}; N=1; M={ROOT}; OUT={out_dir}; "
                        f"PB={d}/probes; WK={d}/workers; TMPDIR={d}; prslot || exit 9\n{line}",
                    ],
                    capture_output=True,
                    text=True,
                ).stdout

            # No draft's run: they run, on the merge and on the head alone, each line with its exit code, tree and
            # time; only the merge's lines are unindented (the ones merge_only_flags compares).
            out = section()
            self.assertRegex(
                out,
                rf"^on the merge {sha[:10]} = head {sha[:10]} \+ develop dddddddddd:\n"
                rf"N1_x: REPRODUCES rc=1 \(tree {sha[:10]}, \d+ s\): still there\n"
                rf"on the head alone {sha[:10]}:\n"
                rf"   N1_x: REPRODUCES rc=1 \(tree {sha[:10]}, \d+ s\): still there\n$",
            )
            self.assertEqual(
                review.probe_results("## Our open findings, re-probed\n\n" + out),
                {"N1_x": "REPRODUCES"},
            )
            (out_dir / "head_sha").write_text("e" * 40 + "\n")
            self.assertIn(
                f"on the head alone: not run (wt1 is at {sha[:10]})", section()
            )
            (out_dir / "head_sha").write_text(sha + "\n")
            shadow = d / "workers" / "shadow_cand" / "pr1"
            shadow.mkdir(parents=True)
            (shadow / "probes.txt").write_text(
                f"# probes of pr1 at {sha}: N1_x\nN1_x: FIXED: the draft's run\n"
            )
            out = section()
            self.assertIn("N1_x: FIXED: the draft's run", out)
            self.assertIn(
                f"(a draft ran these probes on this tree: {shadow}/probes.txt)", out
            )
            self.assertNotIn(
                "\nN1_x: REPRODUCES", out
            )  # the merge's lines are the draft's; the head alone still runs
            self.assertEqual(
                review.probe_results("## Our open findings, re-probed\n\n" + out),
                {"N1_x": "FIXED"},
            )
            (d / "probes" / "pr1" / "N2_y.sh").write_text(
                "exit 0\n"
            )  # another probe since: run them all again
            out = section()
            self.assertIn(f"\nN1_x: REPRODUCES rc=1 (tree {sha[:10]}, ", out)
            self.assertIn("N2_y: FIXED", out)
            (shadow / "probes.txt").write_text(
                f"# probes of pr1 at {'b' * 40}: N1_x,N2_y\nN1_x: FIXED: old tree\n"
            )
            self.assertNotIn("old tree", section())  # another tree: run them

    def test_tests_at_head_are_not_run_on_a_head_mismatch(self):
        """Without a merged tree the PR's tests run at head in wt<N>, which holds the worktree's head: not on a
        mismatch, which would run them against the old head's merge base."""
        plan = {
            "families": [],
            "tests": ["t/test_x.py"],
            "build": False,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        lines = review.checks_script(
            CFG, 1, "a" * 40, "follow-up", plan, None
        ).splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.startswith('WH="$D/wt$N"'))
        self.assertEqual(
            lines[i - 1], 'if [ -n "$MM" ]; then echo "not run: $MM"; else'
        )

    def test_the_checks_quote_their_folders_wherever_they_name_them(self):
        """A working folder may hold a space: every kind of checks run names the monitor's out ($OUT), the working
        folder ($D), develop's commit ($TD), the tmp folder ($TP) and the report ($R) in double quotes, and its opening
        lines write the head, develop's commit, the report's header and the status into that out, and nowhere else.
        """
        plan = {
            "families": ["fam_a"],
            "helpers": {"h": ["fam_a"]},
            "tests": ["platform/tests/test_x.py"],
            "tests_total": 3,
            "changed_tests": ["platform/tests/test_x.py"],
            "conftests": ["platform/tests/conftest.py"],
            "modules": ["platform/python/rocke/x.py"],
            "build": True,
            "golden": True,
            "engine_wide": True,
            "outside_rocke": ["other/x.txt"],
            "libparity": True,
            "_dirs": {"fam_a": "rk/library/fam_a"},
        }
        head, dev, reviewed = "a" * 40, "d" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(CFG, paths=dict(CFG["paths"], working_dir=f"{d}/a b"))
            scripts = {
                "first round": review.checks_script(
                    cfg, 7, head, "first-round", plan, None
                ),
                "follow-up": review.checks_script(
                    cfg,
                    7,
                    head,
                    "follow-up",
                    plan,
                    None,
                    reviewed=reviewed,
                    base="users/x/y",
                ),
                "merge only": review.checks_script(
                    cfg,
                    7,
                    head,
                    "follow-up",
                    plan,
                    None,
                    reviewed=reviewed,
                    merge_only=True,
                ),
                "gate": review.checks_script(
                    cfg, 7, head, "first-round", plan, None, mode="gate"
                ),
                "batch step": review.checks_script(
                    cfg,
                    7,
                    head,
                    "first-round",
                    plan,
                    None,
                    mode="gate",
                    wt="wtb",
                    out=f"{d}/a b/workers/batch/step1",
                    base_sha=dev,
                ),
            }
            for name, script in scripts.items():
                with self.subTest(run=name):
                    self.assertEqual(
                        [ln for ln in script.splitlines() if unquoted_uses(ln)], []
                    )
            lines = scripts["first round"].splitlines()
            opening = next(k for k, ln in enumerate(lines) if ln.startswith("OUT="))

            def line(start):
                return next(ln for ln in lines if ln.startswith(start))

            # The opening lines as they run, with the merge's head and develop's commit as git would give them.
            run = subprocess.run(
                [
                    "bash",
                    "-c",
                    "\n".join(
                        lines[: opening + 1]
                        + [
                            line("u8() "),
                            line("st() "),
                            f"TH={head}; TD={dev}",
                            line("MM=''"),
                        ]
                        + [
                            line('echo "# Checks for'),
                            line("echo 'base: "),
                            "st merge pass clean",
                        ]
                    ),
                ],
                cwd=d,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            out = pathlib.Path(d) / "a b/workers/pr7/checks"
            self.assertEqual((out / "status").read_text(), "merge pass clean\n")
            self.assertEqual((out / "head_sha").read_text(), head + "\n")
            self.assertEqual((out / "develop_sha").read_text(), dev + "\n")
            report = (out / "checks.md").read_text()
            self.assertTrue(
                report.startswith(
                    "# Checks for #7 at aaaaaaaaaa (first-round, review)"
                ),
                report,
            )
            self.assertIn(f"\nbase: develop @ {dev[:10]}\n", report)
            self.assertEqual([p.name for p in pathlib.Path(d).iterdir()], ["a b"])


class TestJobs(unittest.TestCase):
    def test_detached_job_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            job = jobs.start(store, "test", "x", "echo hello; sleep 1", timeout_min=5)
            self.assertEqual(jobs.status(store, job), "running")
            for _ in range(50):
                if jobs.status(store, job) != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(jobs.status(store, job), "done")
            self.assertIn("hello", jobs.tail(store, job))
            bad = jobs.start(store, "test", "y", "exit 3", timeout_min=5)
            for _ in range(50):
                if jobs.status(store, bad) != "running":
                    break
                time.sleep(0.1)
            self.assertEqual(jobs.status(store, bad), "failed(rc=3)")


def _ago(minutes):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


class TestSettleByHandAndBursts(unittest.TestCase):
    """Lessons L-534: hand starts wait out the settle, and a burst of pushes settles 45 min after the last push."""

    def test_the_burst_s_settle_and_a_develop_fix_s_most_are_the_configuration_s(self):
        """settle_burst_min sets how long after its last push a burst settles, and settle_develop_fix_min the most a
        develop fix waits; a configuration without them has the registry's defaults (config.defaults.json).
        """
        from rkmon_lib import registry
        from rkmon_lib.store import settle_left

        burst = {
            "n": 1,
            "head_seen_at": _ago(20),
            "head_committed_at": _ago(120),
            "push_times": [_ago(50), _ago(20)],
        }
        fresh = {"n": 1, "head_seen_at": _ago(5), "head_committed_at": _ago(5)}
        fix = {"develop_fixes": {"1": ["red-develop #33333"]}}
        bare = {"settle_min": 30}
        for cfg, after, most in (
            (dict(bare, settle_burst_min=60, settle_develop_fix_min=8), 60, 8),
            (
                bare,
                registry.default("settle_burst_min"),
                registry.default("settle_develop_fix_min"),
            ),
        ):
            with self.subTest(cfg=cfg):
                left, why = settle_left(cfg, {}, burst)
                self.assertAlmostEqual(left, after - 20, delta=1)
                self.assertIn(f"it settles {after} min after the last, 20 min ago", why)
                self.assertAlmostEqual(
                    settle_left(cfg, fix, fresh)[0], most - 5, delta=1
                )

    def test_a_changed_settle_counts_at_the_next_call_on_the_same_state(self):
        """settle_min takes effect with no restart: on one state, each call counts the settle of the configuration it
        is given."""
        from rkmon_lib.store import settle_left

        st = {}
        rec = {"n": 1, "head_seen_at": _ago(20), "head_committed_at": _ago(20)}
        for settle in (600, 500, 30):
            with self.subTest(settle=settle):
                left, why = settle_left({"settle_min": settle}, st, rec)
                self.assertAlmostEqual(left, settle - 20, delta=1)
                self.assertEqual(why, "head is 20 min old")

    def test_burst_pushes_extend_settle(self):
        """Burst pushes extend the settle, so no follow-up starts only for the next push to kill it. Push times
        are the ticks', never commit dates."""
        from rkmon_lib import events, plan
        from rkmon_lib.store import empty_state, settle_left

        cfg = dict(CFG, settle_min=30)
        rec = {
            "n": 1,
            "head_seen_at": _ago(20),
            "head_committed_at": _ago(120),
            "push_times": [_ago(50), _ago(20)],
        }
        left, why = settle_left(cfg, {}, rec)
        self.assertAlmostEqual(
            left, 25, delta=1
        )  # 45 min after the last push, though the commit is 120 min old
        self.assertIn(
            "2 pushes in 90 min; it settles 45 min after the last, 20 min ago", why
        )
        self.assertFalse(review.settled(cfg, {}, rec))
        rec["push_times"] = [
            _ago(100),
            _ago(20),
        ]  # one push in the window: the plain rule, from the commit date
        self.assertTrue(review.settled(cfg, {}, rec))
        rec["push_times"] = [
            _ago(80),
            _ago(46),
        ]  # a burst whose last push is 46 min old
        self.assertTrue(review.settled(cfg, {}, rec))
        # Develop fixes and the user's priorities keep their 10 min, bursts included.
        fresh = {
            "n": 1,
            "head_seen_at": _ago(12),
            "head_committed_at": _ago(12),
            "push_times": [_ago(50), _ago(12)],
        }
        self.assertFalse(review.settled(cfg, {}, fresh))
        for why in (["red-develop #33333"], ["your priority: golden shards"]):
            self.assertTrue(
                review.settled(cfg, {"develop_fixes": {"1": why}}, fresh), why
            )
        fresh.update(head_seen_at=_ago(5), head_committed_at=_ago(5))
        left, why = settle_left(
            cfg, {"develop_fixes": {"1": ["red-develop #33333"]}}, fresh
        )
        self.assertAlmostEqual(left, 5, delta=1)
        self.assertEqual(why, "head is 5 min old")
        # The plan says why it waits, and the driver keeps waiting.
        st = empty_state()
        st["prs"] = {
            "1": {
                "n": 1,
                "head": "1" * 40,
                "status": "open",
                "draft": False,
                "size": 10,
                "author": "a",
                "ours": None,
                "head_seen_at": _ago(20),
                "head_committed_at": _ago(120),
                "push_times": [_ago(50), _ago(20)],
            }
        }
        item = next(i for i in plan.make_plan(cfg, st, [], []) if i["pr"] == 1)
        self.assertIn("(wait: 2 pushes in 90 min;", item["text"])
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(review.pick_next(cfg, st, Store(pathlib.Path(d)), [item]))
        # Each push a tick sees is kept (the last 8); a PR's first sighting is no push.
        st = empty_state()

        def snap(head):
            return {
                "prs": [
                    {
                        "n": 1,
                        "draft": False,
                        "author": "a",
                        "head": head,
                        "ci_rollup": "SUCCESS",
                        "review_decision": None,
                        "reviews": [],
                        "mergeable": "MERGEABLE",
                    }
                ]
            }

        events.diff_and_apply(
            st, snap("a" * 40), lambda n: {"state": "open"}, ts=_ago(200)
        )
        self.assertIsNone(st["prs"]["1"].get("push_times"))
        seen = [_ago(100 - 10 * i) for i in range(10)]
        for h, ts in zip("bcdefghijk", seen):
            events.diff_and_apply(st, snap(h * 40), lambda n: {"state": "open"}, ts=ts)
        self.assertEqual(st["prs"]["1"]["push_times"], seen[2:])
        self.assertEqual(
            st["prs"]["1"]["push_times"][-1], st["prs"]["1"]["head_seen_at"]
        )

    def test_review_start_refuses_unsettled_head(self):
        """rk review-start refuses a head that has not settled (lessons L-223)."""
        import rkmon
        from unittest import mock
        from rkmon_lib.store import empty_state

        cfg = dict(
            CFG,
            settle_min=30,
            features={"review_automation": {"mode": "live"}},
            merge_authority="propose",
        )
        out, started, forced = [], [], []

        def start(cfg, st, store, n, kind, force=False, again=False):
            started.append((n, kind))
            forced.append(
                force
            )  # --now reaches start_review: a newer live head starts unsettled too
            return [f"REVIEW_START #{n} ({kind}): qwen worker started"]

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "7": {
                    "n": 7,
                    "head": "7" * 40,
                    "status": "open",
                    "draft": False,
                    "size": 10,
                    "ours": None,
                    "head_seen_at": _ago(3),
                    "head_committed_at": _ago(3),
                    "push_times": [_ago(3)],
                }
            }
            store.save(st)
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(rkmon, "Store", lambda: store), mock.patch.object(
                review, "start_review", start
            ), mock.patch(
                "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
            ):
                self.assertEqual(rkmon.main(["review-start", "7"]), 4)
                self.assertEqual(started, [])
                self.assertIn(
                    "REFUSED: #7's head 7777777777 has not settled (head is 3 min old); it settles in",
                    out[-1],
                )
                self.assertIn("now anyway with --now", out[-1])
                self.assertEqual(rkmon.main(["review-start", "7", "--now"]), 0)
                self.assertEqual(started, [(7, "first-round")])
                # A burst: the commit is an hour old, but the second push in 90 min was seen 10 min ago.
                st = store.load()
                st["prs"]["7"].update(
                    head_committed_at=_ago(60),
                    head_seen_at=_ago(10),
                    push_times=[_ago(40), _ago(10)],
                )
                store.save(st)
                self.assertEqual(rkmon.main(["review-start", "7"]), 4)
                self.assertIn(
                    "(2 pushes in 90 min; it settles 45 min after the last, 10 min ago)",
                    out[-1],
                )
                st["prs"]["7"]["push_times"] = [
                    _ago(100),
                    _ago(60),
                ]  # settled: it starts
                store.save(st)
                self.assertEqual(rkmon.main(["review-start", "7"]), 0)
        self.assertEqual(started, [(7, "first-round"), (7, "first-round")])
        self.assertEqual(forced, [True, False])


class TestWorkerRoundAndLine1(unittest.TestCase):
    def test_worker_script_passes_round_and_line1(self):
        """Lessons L-513: the worker knows which round it writes and the exact line 1 (L-001), so a draft's footer
        names the right round. The launcher hands both on to fetch_pr.py, which stages them in meta.md.
        """
        from rkmon_lib import bindings

        # Line 1 gives the reviewer model as it is named: quotes and a $ reach the worker as they are.
        cfg = dict(CFG, worker={"model": "qwen3.6:35b-a3b ('quoted' $HOME)"})
        line1 = bindings.first_line(cfg)
        self.assertEqual(
            line1,
            "Automated review by rocke-monitor: Model - qwen3.6:35b-a3b ('quoted' $HOME), "
            f"effort: {config.MODEL_DEFAULT}",
        )
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "tools").mkdir()
            # A stand-in launcher whose child (as fetch_pr.py is) records what reached it.
            (pathlib.Path(d) / "tools" / "review_agent_run.py").write_text(
                'import os, sys; open(os.environ["GOT"], "w").write("|".join([sys.argv[1]] + '
                '[os.environ.get(k, "-") for k in ("REVIEW_ROUND", "REVIEW_LINE1", "SPARSE_EXTRA")]))\n'
            )
            for kind, word in (
                ("first-round", "First-round"),
                ("follow-up", "Follow-up"),
            ):
                script = review.worker_script(cfg, 7, ["projects/hipdnn"], kind)
                self.assertEqual(
                    subprocess.run(
                        ["bash", "-n"], input=script, capture_output=True, text=True
                    ).returncode,
                    0,
                )
                # Only the launch line runs here: the lines before it clear workers/pr7 and stop the ollama models.
                launch = script.splitlines()[-1]
                self.assertTrue(
                    launch.endswith(
                        f"python3 -B {paths.worker_tool('review_agent_run.py')} $N"
                    ),
                    launch,
                )
                r = subprocess.run(
                    ["bash", "-c", f"T={d}/tools; N=7; export GOT={d}/got\n{launch}"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(
                    (pathlib.Path(d) / "got").read_text(),
                    f"7|{word}|{line1}|projects/hipdnn",
                )
        with self.assertRaises(
            KeyError
        ):  # only the two rounds (`rk review-start --kind` offers no other)
            review.worker_script(cfg, 7, [], "second-round")


class TestStagedLineOne(unittest.TestCase):
    """rk post takes the line 1 its review was staged with: the review folder's pr/meta.md `Line 1:` (fetch_pr.py's,
    from the job's REVIEW_LINE1) when its `Head:` is the review's head. A review in flight keeps the line of the model
    that drafted it after a change of the passes, and any other line is refused, the configuration's own
    among them. A staging of another head is not the review's, nor is no staging: the configuration's line then.
    """

    HEAD = "c" * 40

    class Posted:
        """A ledger with a post at every head: the guard after line 1 and the round refuses there."""

        def ledger_has(self, *a):
            return True

    def setUp(self):
        from rkmon_lib import bindings

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)
        self.cfg = dict(REVIEWING, paths={"working_dir": str(self.d)})
        self.now = bindings.first_line(self.cfg)
        # the line the passes gave before their change: another model drafted the review
        self.staged = bindings.first_line(dict(self.cfg, worker={"model": "llama"}))
        self.assertNotEqual(self.staged, self.now)
        self.st = {"prs": {"7": {"head": self.HEAD, "ours": None}}}

    def stage(self, head: str) -> None:
        """PR 7's review staged at HEAD with the earlier line, in fetch_pr.py's meta.md."""
        meta = self.d / "workers" / "pr7" / "pr" / "meta.md"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            f"# PR #7: t\n- Author: @dev\n- Head: b7 @ {head[:10]}; base: develop @ {'0' * 10}\n"
            f"- Round: First-round\n- Line 1: {self.staged}\n\n## Description\nx\n",
            encoding="utf-8",
        )

    def refusal(self, line1: str) -> str:
        """rk post's refusal of a review of PR 7 whose line 1 is LINE1."""
        f = self.d / "p.md"
        f.write_text(
            f"{line1}\n\n## PR #7 — t — 🟡 First-round: x — z\n", encoding="utf-8"
        )
        with self.assertRaises(post.PostRefused) as cm:
            post.preflight(self.cfg, self.st, self.Posted(), 7, str(f))
        return str(cm.exception)

    def test_a_review_posts_with_the_line_1_it_was_staged_with(self):
        self.stage(self.HEAD)
        self.assertEqual(post.review_line(self.cfg, 7, self.HEAD), self.staged)
        said = self.refusal(self.staged)
        self.assertTrue(said.startswith("already posted on #7"), said)
        self.assertEqual(
            self.refusal(self.now),
            f"line 1 must be exactly {self.staged!r} (got {self.now!r})",
        )

    def test_without_a_staging_of_the_review_s_head_the_configured_line(self):
        for staged in (None, "d" * 40):
            with self.subTest(staged=staged):
                if staged:
                    self.stage(staged)
                self.assertEqual(post.review_line(self.cfg, 7, self.HEAD), self.now)
                said = self.refusal(self.now)
                self.assertTrue(said.startswith("already posted on #7"), said)
                self.assertEqual(
                    self.refusal(self.staged),
                    f"line 1 must be exactly {self.now!r} (got {self.staged!r})",
                )

    def test_a_line_1_in_the_pr_s_description_is_never_the_staged_one(self):
        """A staging of the review's head with no line 1 of its own (fetch_pr.py run by hand stages none), whose
        description, the PR author's text, holds a line of the staged line's form: the configured line is the one
        rk post takes, and the description's is refused."""
        theirs = "Automated review by another-bot: no findings"
        meta = self.d / "workers" / "pr7" / "pr" / "meta.md"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(
            f"# PR #7: t\n- Author: @dev\n- Head: b7 @ {self.HEAD[:10]}; base: develop @ {'0' * 10}\n\n"
            f"## Description\n- Line 1: {theirs}\n",
            encoding="utf-8",
        )
        self.assertEqual(post.review_line(self.cfg, 7, self.HEAD), self.now)
        self.assertEqual(
            self.refusal(theirs),
            f"line 1 must be exactly {self.now!r} (got {theirs!r})",
        )
        said = self.refusal(self.now)
        self.assertTrue(said.startswith("already posted on #7"), said)


class TestTestTools(unittest.TestCase):
    """The checks' test runs get the tools CI's lanes have (cmake, PyYAML) from config `paths.*`, and say what they
    skipped (agent_script_overlap.md finding 4)."""

    def test_the_configured_cmake_and_pyyaml_reach_the_test_runs(self):
        self.assertEqual(
            review.tool_env_lines(CFG, "$OUT/pyshim"), []
        )  # blank: PATH and the venv decide
        with tempfile.TemporaryDirectory() as d:
            site = pathlib.Path(d) / "site-packages"
            (site / "yaml").mkdir(parents=True)
            (site / "yaml" / "__init__.py").write_text("")
            (
                site / "numpy"
            ).mkdir()  # the rest of that site-packages stays out of the tests' PYTHONPATH
            cmake_dir = pathlib.Path(d) / "cmake bin"
            for key, value in (("pyyaml", str(site)), ("pyyaml", str(site / "yaml"))):
                cfg = dict(CFG, paths={"cmake": str(cmake_dir), key: value})
                # The link's folder has a space: the lines quote it.
                lines = review.tool_env_lines(cfg, f"{d}/out dir/pyshim")
                self.assertEqual(lines[0], f"export PATH='{cmake_dir}':$PATH")
                r = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "\n".join(
                            lines
                            + [
                                'echo "$PATH"; echo "$RKMON_PYEXTRA"; '
                                'ls "$RKMON_PYEXTRA"; readlink "$RKMON_PYEXTRA/yaml"'
                            ]
                        ),
                    ],
                    cwd=d,  # a path split at its space would leave its folders here
                    capture_output=True,
                    text=True,
                    check=True,
                )
                path, extra, listed, target = r.stdout.splitlines()
                self.assertTrue(path.startswith(f"{cmake_dir}:"), path)
                self.assertEqual(
                    (extra, listed, target),
                    (f"{d}/out dir/pyshim", "yaml", str(site / "yaml")),
                )
                self.assertEqual(
                    sorted(p.name for p in pathlib.Path(d).iterdir()),
                    ["out dir", "site-packages"],
                )

    def test_the_checks_and_drift_scripts_export_them_before_any_test_runs(self):
        cfg = dict(
            CFG,
            paths={
                "cmake": "/opt/cmake/bin",
                "pyyaml": "/opt/py/site-packages",
                "repo": "/r",
                "working_dir": "/w",
            },
        )
        plan = review.plan_checks(cfg, {}, ["rk/library/tests/test_new.py"], {})
        plan["_dirs"] = {}
        script = review.checks_script(cfg, 1, "a" * 40, "follow-up", plan, None)
        env = script.index("export PATH=/opt/cmake/bin:$PATH")
        shim = script.index('"$SI/pyshim/yaml" && export RKMON_PYEXTRA="$SI/pyshim"')
        self.assertLess(script.index("prslot $N ||"), shim)
        self.assertLess(shim, script.index("sec '"))
        self.assertLess(env, script.index("pytest_vs_base.py"))
        self.assertIn('res "$t"', script)  # each test section ends on its result line
        self.assertIn('res "$i"', script)
        from rkmon_lib import drift

        d = drift.script(
            cfg,
            "/w/drift/d1",
            "d" * 40,
            [{"n": 7, "head": "b" * 40, "tests": ["library/tests/t.py"]}],
        )
        shim = d.index('export RKMON_PYEXTRA="$SI/pyshim"')
        self.assertLess(d.index("(prslot 7 || exit 3; "), shim)
        self.assertLess(shim, d.index("pytest_vs_base.py"))

    def test_the_result_line_never_reads_a_skip_as_a_pass(self):
        def res(line):
            return subprocess.run(
                ["bash", "-c", review.RESULT_FN + '\nres "$1"', "-", line],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

        self.assertEqual(
            res("TESTS pass (2 skipped: 2 needs a GPU) 9 passed, 2 skipped"),
            "result: PASS (2 skipped: 2 needs a GPU) 9 passed, 2 skipped",
        )
        self.assertEqual(
            res("TESTS skip NOT RUN: cmake missing (1 test); 9 passed"),
            "result: NOT RUN: cmake missing (1 test); 9 passed",
        )
        self.assertEqual(
            res("INSTALLED fail 1 run of 1: x"), "result: FAIL 1 run of 1: x"
        )
        self.assertEqual(
            res("TESTS skip no test to run"), "result: SKIP no test to run"
        )

    def test_the_worktrees_check_out_the_shared_ctest_parser(self):
        # The home repository's test_sparse names the shared CTest parser; none configured: only the PR's own paths.
        bare = dict(CFG, rocke_path="team/x/rocke")
        cfg = home_repository.config(bare)
        self.assertEqual(review.sparse_extra(cfg, ["team/x/a.py"]), ["shared/ctest"])
        self.assertEqual(review.sparse_extra(bare, ["team/x/a.py"]), [])
        self.assertEqual(review.sparse_extra(bare, ["other/lib/a.cpp"]), ["other/lib"])
        self.assertEqual(
            review.sparse_extra(
                cfg, ["other/lib/a.cpp", "shared/ctest/p.py", "team/x"]
            ),
            ["shared/ctest", "other/lib"],
        )
        self.assertEqual(
            review.sparse_extra(dict(cfg, test_sparse=["a/b"]), ["team/x/a.py"]),
            ["a/b"],
        )


class TestPrefill(unittest.TestCase):
    """No heavy job in a worker's first minutes, and no worker in a heavy job's first minutes (a build alongside a
    worker's first prompt can wedge the runner)."""

    def test_both_ways(self):
        from unittest import mock
        from rkmon_lib.store import empty_state, now

        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["prs"] = {
                "2": {
                    "n": 2,
                    "head": "2" * 40,
                    "status": "open",
                    "draft": False,
                    "size": 10,
                }
            }
            plan = [{"kind": "FIRST_ROUND", "pr": 2, "text": "y"}]
            local = dict(CFG, worker={"url": "http://127.0.0.1:11434", "model": "m:1"})
            st["review_pipeline"] = {
                "pr": 2,
                "worker": {"id": "w", "started": now()},
            }  # the state before the pipes
            self.assertTrue(review.worker_prefill(local, st))
            self.assertFalse(
                review.worker_prefill(
                    dict(
                        CFG,
                        worker={"url": "https://models.example.com", "model": "m:1"},
                    ),
                    st,
                )
            )  # a remote reviewer
            st["review_pipes"][0]["worker"]["started"] = "2000-01-01T00:00:00Z"
            self.assertFalse(review.worker_prefill(local, st))
            st["review_pipes"] = []
            st["jobs"] = {"heavy": {"id": "gate-pr1-x", "started": now()}}
            with mock.patch.object(
                review.jobs, "status", lambda store, job: "running"
            ), mock.patch.object(
                review, "start_review", side_effect=AssertionError("started a worker")
            ):
                out = review._advance(local, st, store, plan)
            self.assertEqual(len(out), 1)
            self.assertTrue(
                out[0].startswith(
                    "REVIEW_WAIT: the next worker starts once gate-pr1-x"
                ),
                out,
            )
            st["jobs"]["heavy"][
                "started"
            ] = "2000-01-01T00:00:00Z"  # past its build: the worker starts
            with mock.patch.object(
                review.jobs, "status", lambda store, job: "running"
            ), mock.patch.object(
                review, "start_review", return_value=["REVIEW_START #2 (first-round)"]
            ) as sr:
                self.assertEqual(
                    review._advance(local, st, store, plan),
                    ["REVIEW_START #2 (first-round)"],
                )
            sr.assert_called_once()


class TestLiveHead(unittest.TestCase):
    """One live head per review: a review started from a stale state uses the live head throughout (the worker,
    mkwt and the checks alike), so its post needs no --again."""

    OLD, NEW, EIGHT = "a" * 40, "b" * 40, "8" * 40

    def setUp(self):
        from rkmon_lib.store import empty_state

        self.d = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.d.name) / "state")
        self.cfg = dict(
            CFG,
            settle_min=30,
            features={"review_automation": {"mode": "live"}},
            merge_authority="propose",
            review={"pipes": 1},
            paths={"working_dir": self.d.name},
        )  # one pipe, as before the pipes
        self.st = empty_state()
        self.st["last_tick"] = _ago(5)
        common = {
            "status": "open",
            "draft": False,
            "size": 10,
            "ours": None,
            "ci_rollup": "SUCCESS",
            "review_decision": None,
            "reviews": [],
            "mergeable": "MERGEABLE",
            "updated_at": None,
        }
        self.st["prs"] = {
            "7": dict(
                common,
                n=7,
                head=self.OLD,
                head_seen_at=_ago(300),
                head_committed_at=_ago(300),
                push_times=[_ago(300)],
                gates={"head": self.OLD},
            ),
            "8": dict(
                common,
                n=8,
                head=self.EIGHT,
                head_seen_at=_ago(300),
                head_committed_at=_ago(300),
            ),
        }
        self.plan = [
            {
                "prio": 3,
                "kind": "FIRST_ROUND",
                "pr": 7,
                "text": "first-round review of #7",
            },
            {
                "prio": 3,
                "kind": "FIRST_ROUND",
                "pr": 8,
                "text": "first-round review of #8",
            },
        ]
        self.jobs = []
        self.gets = []

    def tearDown(self):
        self.d.cleanup()

    def rest(self, state="open", committed=None, fail=False):
        """A fake gh.rest: #7 is live at NEW (committed `committed`), #8 at its state head."""

        def rest(path, jq=None, paginate=False, timeout=120):
            self.gets.append(path)
            if fail:
                raise review.gh.GhError("gh api repos/ROCm/r/pulls/7: HTTP 502")
            if path == "repos/ROCm/r/pulls/7":
                return {"state": state, "head": self.NEW}
            if path == "repos/ROCm/r/pulls/8":
                return {"state": "open", "head": self.EIGHT}
            if path == f"repos/ROCm/r/commits/{self.NEW}":
                return committed or _ago(60)
            raise AssertionError(f"unexpected GET {path}")

        return rest

    def patched(self, rest, worker="running"):
        from contextlib import ExitStack
        from unittest import mock

        def start(store, kind, name, script, timeout_min, env=None):
            self.jobs.append((kind, name, script))
            return {"id": f"{kind}-{name}", "started": "2000-01-01T00:00:00Z", "pid": 0}

        stack = ExitStack()
        stack.enter_context(mock.patch.object(review.gh, "rest", rest))
        stack.enter_context(
            mock.patch.object(review, "_changed_files", return_value=[])
        )
        stack.enter_context(mock.patch.object(review.jobs, "start", start))
        stack.enter_context(
            mock.patch.object(
                review.jobs, "status", lambda store, job: worker if job else "none"
            )
        )
        return stack

    def test_a_stale_state_head_is_replaced_by_the_live_head(self):
        from rkmon_lib import events

        with self.patched(self.rest()):
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        note = f"#7: state head {self.OLD[:10]} was out of date; using live head {self.NEW[:10]}"
        self.assertEqual(
            lines,
            [
                f"review {note}",
                f"REVIEW_START #7 (first-round): reviewer model worker started "
                f"at {self.NEW[:10]} (parallel review 1 of 1)",
            ],
        )
        rec = self.st["prs"]["7"]
        self.assertEqual(
            (rec["head"], rec["gates"]), (self.NEW, None)
        )  # the gates were the old head's
        self.assertEqual(rec["push_times"][-1], rec["head_seen_at"])
        self.assertLess(review.minutes_since(rec["head_seen_at"]), 2)
        self.assertGreater(review.minutes_since(rec["head_committed_at"]), 59)
        self.assertEqual(review.pipe_of(self.st, 7)["head"], self.NEW)
        self.assertIn(f"review   {note}", self.store.journal_path.read_text())
        # The next tick reports the push it would have seen, once, so the queue still acts on it (an armed head).
        push = {"kind": "PUSH", "pr": 7, "old": self.OLD[:10], "new": self.NEW[:10]}
        self.assertEqual(self.st["pending_events"], [push])
        self.assertEqual(events.take_pending(self.st), [push])
        self.assertNotIn("pending_events", self.st)
        snap = {
            "prs": [
                {k: v for k, v in r.items() if k in events.PR_FIELDS or k == "n"}
                for r in self.st["prs"].values()
            ]
        }
        self.assertEqual(
            events.diff_and_apply(self.st, snap, lambda n: {"state": "open"}), []
        )
        # The checks, generated once the worker is past its first minutes, expect that same head.
        (pathlib.Path(self.d.name) / "wt7m").mkdir()
        with self.patched(self.rest()):
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertTrue(
            lines[0].startswith("REVIEW #7: worker running; checks started"), lines
        )
        kind, name, script = self.jobs[-1]
        self.assertEqual((kind, name), ("checks", "pr7"))
        self.assertIn(f'if [ "$TH" != "{self.NEW}" ]', script)
        self.assertIn(f"commits/{self.NEW}/check-runs", script)
        self.assertNotIn(self.OLD, script)

    def test_the_tick_reports_a_push_a_review_start_found(self):
        import rkmon
        from unittest import mock

        self.st["pending_events"] = [
            {"kind": "PUSH", "pr": 7, "old": self.OLD[:10], "new": self.NEW[:10]}
        ]
        self.st["prs"]["7"]["head"] = self.NEW
        snap = [
            {k: v for k, v in r.items() if k != "status"}
            for r in self.st["prs"].values()
        ]
        with mock.patch.object(
            rkmon.snapshot, "list_prs", return_value=snap
        ), mock.patch.object(
            rkmon.snapshot, "scope_filter", lambda cfg, prs, cache: prs
        ), mock.patch.object(
            rkmon.snapshot, "correct_rollups", lambda cfg, prs, cache: prs
        ), mock.patch.object(
            rkmon.snapshot, "develop_snapshot", return_value=None
        ), mock.patch.object(
            rkmon, "refresh_ours", return_value=0
        ):
            evs = rkmon.do_poll(self.cfg, self.store, self.st)
        self.assertEqual([(e["kind"], e["pr"]) for e in evs], [("PUSH", 7)])
        self.assertEqual(
            [e["kind"] for e in self.store.read_jsonl("events.jsonl")], ["PUSH"]
        )
        self.assertNotIn("pending_events", self.st)

    def test_an_unsettled_live_head_waits_and_the_next_pr_starts(self):
        with self.patched(self.rest(committed=_ago(2))):
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines[1],
            f"REVIEW_SETTLING #7: its live head {self.NEW[:10]} has not settled (head is 2 min "
            "old); it settles in 28 min",
        )
        self.assertEqual(
            lines[2],
            f"REVIEW_START #8 (first-round): reviewer model worker started at "
            f"{self.EIGHT[:10]} (parallel review 1 of 1)",
        )
        self.assertEqual(
            self.st["prs"]["7"]["head"], self.NEW
        )  # recorded, though not started
        self.assertEqual([p["pr"] for p in review.pipes(self.st)], [8])
        # The plan never saw that head, so the next advances leave #7 until it settles, without asking GitHub again.
        self.st["review_pipes"] = []
        self.assertIsNone(
            review.pick_next(self.cfg, self.st, self.store, self.plan[:1])
        )

    def test_a_pr_closed_on_github_starts_nothing_and_the_next_does(self):
        with self.patched(self.rest(state="closed")):
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines[0],
            "REVIEW_CLOSED #7: it is closed on GitHub; nothing started (the next tick catches up)",
        )
        self.assertTrue(lines[1].startswith("REVIEW_START #8"), lines)
        self.assertEqual(
            self.st["prs"]["7"]["head"], self.OLD
        )  # the tick records the closure
        self.assertNotIn("pending_events", self.st)

    def test_an_unread_live_head_starts_nothing(self):
        with self.patched(self.rest(fail=True)):
            lines = review.advance(self.cfg, self.st, self.store, self.plan)
        self.assertEqual(
            lines,
            [
                "REVIEW_WAIT #7: its live head was not read (gh api repos/ROCm/r/pulls/7: HTTP 502); "
                "nothing started"
            ],
        )
        self.assertEqual(
            self.gets, ["repos/ROCm/r/pulls/7"]
        )  # it stops: #8 is not tried
        self.assertEqual(review.pipes(self.st), [])
        self.assertEqual(self.st["prs"]["7"]["head"], self.OLD)

    def test_by_hand_a_newer_unsettled_live_head_is_refused_unless_now(self):
        import rkmon
        from unittest import mock

        out = []
        self.store.save(self.st)
        with self.patched(self.rest(committed=_ago(2))), mock.patch.object(
            rkmon, "load_config", lambda: self.cfg
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            self.assertEqual(rkmon.main(["review-start", "7"]), 4)
            self.assertTrue(out[-1].startswith("REVIEW_SETTLING #7"), out)
            st = self.store.load()  # saved, not started
            self.assertEqual((st["prs"]["7"]["head"], review.pipes(st)), (self.NEW, []))
            self.assertEqual(len(st["pending_events"]), 1)
            self.assertEqual(rkmon.main(["review-start", "7", "--now"]), 0)
        self.assertEqual(
            out[-1],
            f"REVIEW_START #7 (first-round): reviewer model worker started at "
            f"{self.NEW[:10]} (parallel review 1 of 1)",
        )
        self.assertEqual(review.pipe_of(self.store.load(), 7)["head"], self.NEW)


class TestTickStale(unittest.TestCase):
    """With the last tick over 25 minutes old (the tick loop may be down), the review driver starts no review from
    the stale state."""

    def test_it_fires_after_25_minutes_and_not_before(self):
        self.assertIsNone(review.tick_stale({"last_tick": _ago(24)}))
        last = _ago(26)
        self.assertEqual(
            review.tick_stale({"last_tick": last}),
            f"TICK_STALE: the last tick was at {last} (26 min "
            "ago): the tick loop may be down; run rk tick",
        )
        self.assertEqual(
            review.tick_stale({"last_tick": None}),
            "TICK_STALE: no tick on record: the tick loop may be down; run rk tick",
        )

    def test_review_advance_prints_it_and_starts_nothing(self):
        import rkmon
        from unittest import mock
        from rkmon_lib.store import empty_state

        cfg = dict(
            CFG,
            settle_min=30,
            features={"review_automation": {"mode": "live"}},
            merge_authority="propose",
        )
        out = []
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["last_tick"] = _ago(26)
            st["last_plan"] = [
                {"kind": "FIRST_ROUND", "pr": 7, "text": "first-round review of #7"}
            ]
            store.save(st)
            before = store.state_path.read_bytes()
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(rkmon, "Store", lambda: store), mock.patch(
                "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
            ):
                with mock.patch.object(
                    review,
                    "advance",
                    side_effect=AssertionError("advanced from a stale state"),
                ):
                    self.assertEqual(rkmon.main(["review-advance"]), 0)
                self.assertEqual(out, [review.tick_stale(st)])
                self.assertTrue(
                    out[0].startswith(
                        f"TICK_STALE: the last tick was at {st['last_tick']} (26 min ago)"
                    )
                )
                self.assertEqual(
                    store.state_path.read_bytes(), before
                )  # nothing changed
                st["last_tick"] = _ago(24)  # a tick clears it
                store.save(st)
                with mock.patch.object(
                    review, "advance", return_value=["REVIEW #7: worker running"]
                ) as adv:
                    self.assertEqual(rkmon.main(["review-advance"]), 0)
                adv.assert_called_once()
        self.assertEqual(out[1:], ["REVIEW #7: worker running"])

    def test_the_driver_exits_with_the_line(self):
        """The review driver (rk review-driver, which tools/review_driver.py runs) prints the line and exits, having
        started nothing."""
        import rkmon
        from unittest import mock
        from rkmon_lib.store import empty_state

        cfg = dict(
            CFG,
            settle_min=30,
            features={"review_automation": {"mode": "live"}},
            merge_authority="propose",
        )
        out = []
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["last_tick"] = _ago(198)
            st["last_plan"] = [
                {"kind": "FIRST_ROUND", "pr": 7, "text": "first-round review of #7"}
            ]
            store.save(st)
            with mock.patch.object(
                rkmon, "load_config", lambda: cfg
            ), mock.patch.object(
                review,
                "advance",
                side_effect=AssertionError("advanced from a stale state"),
            ):
                self.assertEqual(
                    rkmon.cmd_review_driver(
                        cfg, store, sleep=lambda s: None, out=out.append
                    ),
                    0,
                )
        self.assertEqual(out[0], review.tick_stale(st))
        self.assertTrue(out[0].startswith("TICK_STALE: the last tick was at"))
        # Then the lines to act on, as `rk tick-filter --driver` prints them.
        self.assertEqual(
            out[1:], ["---\n" + "\n".join(rkmon.tickfilter.driver_lines(out[0]))]
        )


class TestStatusLinesAreWholeCharacters(unittest.TestCase):
    """GNU `cut -c` counts bytes: a detail cut inside "…" or "—" wrote a status line that was not UTF-8 and stopped the
    tick that read it. The checks' and the sanity run's st write whole characters only, and cutc never cuts one. The
    byte cuts below are `cut -b`, which counts bytes with GNU's cut and uutils' alike (uutils' `cut -c`, Ubuntu
    26.04's, counts characters)."""

    def run_st(self, fn_lines, body):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(
                ["bash", "-c", "\n".join([f"OUT={d}", *fn_lines, body])],
                check=True,
                timeout=30,
                capture_output=True,
            )
            raw = (pathlib.Path(d) / "status").read_bytes()
        return raw.decode("utf-8")  # strict: a cut character fails here

    def test_the_checks_status(self):
        script = review.checks_script(
            CFG,
            1,
            "a" * 40,
            "first-round",
            {
                "families": [],
                "tests": [],
                "build": False,
                "golden": False,
                "engine_wide": False,
                "outside_rocke": [],
                "libparity": False,
                "_dirs": {},
            },
            None,
        )
        self.assertIn(review.U8_FN, script)
        self.assertIn(review.ST_FN, script)
        self.assertNotRegex(script, r"\| cut -[bc]1-[0-9]+")  # every piped cut is cutc
        text = "abc — d…e"  # "—" and "…" take three bytes each
        out = self.run_st(
            [review.U8_FN, review.ST_FN],
            f"for n in 4 5 6 7 11 12; do st k$n fail \"$(printf '%s' '{text}' | cut -b1-$n)\"; done; "
            f"st c pass \"$(printf '%s' '{text}' | cutc 6)\"",
        )
        self.assertEqual(
            out.splitlines(),
            [
                "k4 fail abc ",
                "k5 fail abc ",
                "k6 fail abc ",
                "k7 fail abc —",
                "k11 fail abc — d",
                "k12 fail abc — d…",
                "c pass abc ",
            ],
        )

    def test_the_sanity_status(self):
        from rkmon_lib import guard

        script = guard.sanity_script(
            CFG, "a" * 40, "/x/wt", "/x/out", [], False, sections=[]
        )
        st = next(x for x in script.splitlines() if x.startswith("st() {"))
        self.assertIn(review.U8_FN, script)
        out = self.run_st(
            [review.U8_FN, st],
            "st golden fail \"$(printf '%s' 'x…y' | cut -b1-2)\" > /dev/null",
        )
        self.assertEqual(out, "golden fail x\n")


if __name__ == "__main__":
    unittest.main()
