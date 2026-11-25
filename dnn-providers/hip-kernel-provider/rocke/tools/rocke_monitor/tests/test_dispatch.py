# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One call per orchestrator step (rkmon_lib/dispatch.py): `rk dispatch-pack N` (the briefs' inputs, compactly, and the
lane rule), `rk spec-post N` (the spec, then the post, CHECK lines and --devbase-recheck), `rk wait-quality DIR`, the
review driver's exit lines (REVIEW_READY last), the tick loop's wake lines, and the checks' shared build (one in a slot's
folder only while the slot is kept for it).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rkmon  # noqa: E402
from rkmon_lib import container_layout as L  # noqa: E402
from rkmon_lib import (
    dispatch,
    facts,
    gitops,
    overlaps,
    review,
    slots,
    tickfilter,
)  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def g(repo, *a) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestLaneRule(unittest.TestCase):
    def c(self, kind, files, n=None):
        return {
            "kind": kind,
            "old": "o" * 40,
            "new": "n" * 40,
            "changed_files": files,
            "n_changed": len(files) if n is None else n,
        }

    def test_the_rule(self):
        cases = [
            ({"kind": "never reviewed"}, "lane: needed (a first round)"),
            (
                self.c("merge-only", ["platform/x.py"]),
                "lane: exempt (only develop merges since our review)",
            ),
            (self.c("rebase-only", []), "lane: exempt (a rebase only)"),
            (
                self.c("substantive", ["docs/a.md", "platform/README.rst"]),
                "lane: exempt (only docs changed)",
            ),
            (
                self.c(
                    "substantive",
                    [
                        "library/tests/test_a.py",
                        "platform/tests/conftest.py",
                        "docs/a.md",
                    ],
                ),
                "lane: exempt (only test files and docs changed: if they answer our own open findings; else run the "
                "lane)",
            ),
            # A CI list or CMake file under tests/ decides what runs: the lane checks it.
            (
                self.c(
                    "substantive",
                    ["library/tests/test_a.py", "platform/tests/CMakeLists.txt"],
                ),
                "lane: needed (1 non-test file, e.g. platform/tests/CMakeLists.txt)",
            ),
            (
                self.c(
                    "substantive",
                    ["library/tests/test_a.py", "ROCKE_ENGINE_test_categories.yaml"],
                ),
                "lane: needed (1 non-test file, e.g. ROCKE_ENGINE_test_categories.yaml)",
            ),
            (
                self.c(
                    "substantive",
                    [
                        "platform/cpp/include/rocke/abi.h",
                        "platform/tests/core/test_x.py",
                    ],
                ),
                "lane: needed (1 non-test file, e.g. platform/cpp/include/rocke/abi.h)",
            ),
            (
                self.c("substantive", ["library/tests/test_a.py"], n=41),
                "lane: needed (41 files changed, more than " "classify lists)",
            ),
            (self.c("same-head", []), "lane: needed (same-head)"),
            (None, "lane: needed (unknown)"),
        ]
        for c, want in cases:
            with self.subTest(want=want):
                self.assertEqual(dispatch.lane_rule(c), want)
        self.assertEqual(
            dispatch.lane_rule(self.c("merge-only", []), ["thread x"]),
            "lane: exempt (a micro round: the verifier alone, on its new items)",
        )


FACTS = """# Facts for #7, read 2026-11-10T19:29:28Z
== head and state
{head} open draft=false mergeable=true blocked by me into develop updated 2026-11-10T17:14:29Z | feat(rk): x
== force-pushes (every head the PR had, newest last)
none
== our last review: header, flagged rows, findings (copy them into the brief; never recall them)
from PR_FOLLOWUP_7.md
## PR #7 — feat(rk): x (dev) — 🟡 Follow-up: y
N5 🟡 a finding
== rk abi
binary: develop 15
== unresolved threads
total 3, unresolved 1
* rk/a.py:3 yr: please explain
== op-login's hand-written comments, in full (our automated posts share the account)
--- op-login 2026-11-10T12:00:00Z
merge after #9
== other people's comments since our last review, in full (bots left out)
--- dev a long reply
"""


class TestPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.repo = self.d / "repo"
        self.repo.mkdir()
        g(self.repo, "init", "-q", "-b", "develop")
        (self.repo / "a.txt").write_text("a\n")
        g(self.repo, "add", "-A")
        g(self.repo, "commit", "-qm", "base")
        self.old = g(self.repo, "rev-parse", "HEAD")
        g(self.repo, "checkout", "-q", "-b", "dev2")
        (self.repo / "b.txt").write_text("b\n")
        g(self.repo, "add", "-A")
        g(self.repo, "commit", "-qm", "develop change")
        g(self.repo, "checkout", "-q", "-b", "pr", self.old)
        (self.repo / "c.txt").write_text("c\n")
        g(self.repo, "add", "-A")
        g(self.repo, "commit", "-qm", "own change")
        g(self.repo, "merge", "-q", "--no-edit", "dev2")
        self.new = g(self.repo, "rev-parse", "HEAD")
        self.cfg = {
            "operator": {"github_login": "op-login"},
            "repo": "o/r",
            "rocke_path": "rk",
            "paths": {
                "repo": str(self.repo),
                "workers": str(self.d / "workers"),
                "root": str(self.d),
            },
        }
        self.chk = self.d / "workers" / "pr7" / "checks"
        self.chk.mkdir(parents=True)
        self.store = Store(self.d / "state")

    def test_fact_sections(self):
        lines = dispatch.fact_sections(FACTS.format(head="h" * 40), "op-login")
        self.assertEqual(lines[0], "== head and state")
        self.assertIn("== our last review: header, flagged rows, findings", lines)
        self.assertIn("N5 🟡 a finding", lines)
        self.assertIn("* rk/a.py:3 yr: please explain", lines)
        self.assertIn("merge after #9", lines)
        self.assertNotIn("binary: develop 15", lines)  # not a section the pack prints
        self.assertNotIn("--- dev a long reply", lines)

    def test_the_sections_it_prints_are_titles_of_the_facts(self):
        """Each facts section the pack prints starts a title rk facts prints (facts.SECTIONS)."""
        titles = [title.format(me="op-login") for title, _ in facts.SECTIONS]
        for start, _ in dispatch.FACT_SECTIONS:
            start = start.format(me="op-login")
            self.assertTrue(any(t.startswith(start) for t in titles), start)

    def test_delta_shared_build_checks_decisions(self):
        c = {
            "kind": "substantive",
            "old": self.old,
            "new": self.new,
            "changed_files": ["c.txt"],
            "n_changed": 1,
            "new_commits": [f"{self.new[:7]} own change"],
        }
        lines = dispatch.delta_lines(self.cfg, {}, c)
        self.assertEqual(
            lines[0],
            f"delta: substantive {self.old[:10]} -> {self.new[:10]}: 1 of the PR's own files "
            "changed",
        )
        self.assertIn(
            "own commits since our review (git log --no-merges, develop's left out): 1; merges: 1",
            lines,
        )
        self.assertEqual(
            dispatch.delta_lines(self.cfg, {}, {"kind": "never reviewed"}),
            ["delta: never reviewed"],
        )
        # The shared build: named by the checks, and still the tree's commit.
        self.assertTrue(
            dispatch.shared_build(self.cfg, 7).startswith("shared build: none")
        )
        build = (
            self.d / "build dir" / "wt7m"
        )  # a space: the record keeps the path whole
        build.mkdir(parents=True)
        (self.chk / review.SHARED_BUILD).write_text(
            f"{build}\n{self.repo}\n{self.new}\n"
        )
        self.assertEqual(
            dispatch.shared_build(self.cfg, 7),
            f"shared build: {build} (core, pybind) of {self.repo} at {self.new[:10]}: both agents reuse "
            "it read-only and build only a mutant or another tree",
        )
        g(self.repo, "checkout", "-q", self.old)  # a gate re-made the tree
        self.assertTrue(
            dispatch.shared_build(self.cfg, 7).startswith(
                f"shared build: gone ({self.repo} is at " f"{self.old[:10]}"
            )
        )
        (self.chk / "status").write_text(
            "merge pass clean\nthreads fail 2 live threads\nclash fail 1 PR clash\n"
            "tests skip no test to run\nbuild skip low memory\n"
        )
        (self.chk / "head_sha").write_text(self.new + "\n")
        self.assertTrue(
            dispatch.checks_lines(self.cfg, 7)[0].startswith(
                f"checks at {self.new[:10]}: flags: threads; not run: build; leads: clash ("
            )
        )
        decided = {
            "k1": {"owner": 7, "members": [7, 9]},
            "k2": {"owner": 9, "members": [7, 9, 11]},
            "k3": {"owner": 11, "members": [9, 11]},
        }
        (self.store.dir / overlaps.DECIDED_FILE).write_text(json.dumps(decided))
        self.assertEqual(
            dispatch.decision_lines(self.store, 7),
            [
                "overlap decisions naming #7: 2",
                "   k1: #7 owner (members #7, #9)",
                "   k2: #7 drops its copy for #9 (members #7, #9, #11)",
            ],
        )

    def test_a_build_in_a_slot_is_shared_only_while_the_slot_is_kept_for_it(self):
        top, locks = self.d / "slots", self.d / "slots" / "locks"
        locks.mkdir(parents=True)
        build = top / "rkpr3" / "build" / "wt7m"
        build.mkdir(parents=True)
        (self.chk / review.SHARED_BUILD).write_text(
            f"{build}\n{self.repo}\n{self.new}\n"
        )
        gone = f"shared build: gone (the slot that held {build} was freed): each agent builds its own"
        with mock.patch.multiple(L, SLOTS_DIR=str(top), SLOT_LOCKS=str(locks)):
            self.assertEqual(dispatch.shared_build(self.cfg, 7), gone)
            kept = {"state": "kept", "pr": "7", "head": self.new, "build": str(build)}
            slots.Records().write("rkpr3", kept)
            self.assertTrue(
                dispatch.shared_build(self.cfg, 7).startswith(
                    f"shared build: {build} (core, pybind) of {self.repo}"
                )
            )
            # taken back for another job
            slots.Records().write("rkpr3", {"state": "taken", "pid": 41, "pr": "9"})
            self.assertEqual(dispatch.shared_build(self.cfg, 7), gone)

    def test_the_checks_record_keeps_a_path_with_a_space_whole(self):
        """The checks write the shared build record one path to a line (review.SHARED_BUILD_LINES), so a build folder,
        tree or checks folder with a space reads back whole, as `rk dispatch-pack` names it.
        """
        cfg = dict(
            self.cfg, paths=dict(self.cfg["paths"], workers=str(self.d / "work dir"))
        )
        out = self.d / "work dir" / "pr7" / "checks"
        out.mkdir(parents=True)
        tree = self.d / "tree dir"
        shutil.copytree(self.repo, tree)
        sb = self.d / "slot dir" / "build"
        (sb / "wt7m").mkdir(parents=True)
        subprocess.run(
            ["bash", "-c", review.SHARED_BUILD_LINES],
            env=dict(os.environ, SB=str(sb), WTN="wt7m", W=str(tree), OUT=str(out)),
            cwd=self.d,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            review.shared_build_record(cfg, 7), [f"{sb}/wt7m", str(tree), self.new]
        )
        self.assertEqual(
            dispatch.shared_build(cfg, 7),
            f"shared build: {sb}/wt7m (core, pybind) of {tree} at {self.new[:10]}: both agents reuse it "
            "read-only and build only a mutant or another tree",
        )

    def test_the_pack(self):
        st = empty_state()
        st["prs"] = {
            "7": {
                "n": 7,
                "head": self.new,
                "status": "open",
                "ours": {"head": self.old},
            }
        }
        st["review_ready"] = [
            {
                "pr": 7,
                "kind": "follow-up",
                "head": self.new,
                "micro": ["thread yr at rk/a.py:3 u"],
            }
        ]
        out = self.d / "tmp" / "v7fu2"

        def prepare(cfg, st_, store, n, folder):
            folder.mkdir(parents=True)
            (folder / "facts.txt").write_text(FACTS.format(head=self.new))
            (folder / "classify.txt").write_text(
                json.dumps(
                    {
                        "kind": "merge-only",
                        "old": self.old,
                        "new": self.new,
                        "changed_files": [],
                        "n_changed": 0,
                        "new_commits": [],
                        "merges": 1,
                    }
                )
            )
            return [
                f"review dir {folder}",
                "facts exit 0",
                "classify exit 0",
                "x" * 300,
                f"head {self.new[:10]} mb {self.old[:10]} commits 3",
                "2",
            ], True

        fails = (
            [
                "head " + self.new,
                "RUN\tLinux shard 1\tin_progress\thttps://u/1",
                "FAIL\tWindows shard 2\tfailure\thttps://u/2",
                "runs 40 names 30",
            ],
            True,
        )
        with mock.patch.object(dispatch.prep, "prepare", prepare), mock.patch.object(
            dispatch.ci_runs, "fails", return_value=fails
        ):
            lines, ok = dispatch.pack(self.cfg, st, self.store, 7, "fu2", out)
        self.assertTrue(ok)
        self.assertEqual(
            lines[0],
            f"== #7 fu2: review dir {out}; facts exit 0; classify exit 0; head {self.new[:10]} mb "
            f"{self.old[:10]} commits 3; 2 files",
        )
        for want in (
            "== our last review: header, flagged rows, findings",
            "N5 🟡 a finding",
            "merge after #9",
            "== since our last review",
            "CI (rk ci-fails): 1 failing, 1 running",
            "FAIL\tWindows shard 2\tfailure\thttps://u/2",
            "overlap decisions naming #7: none",
            "== micro round: brief the verifier on these items alone",
            "   thread yr at rk/a.py:3 u",
            "checks: no status in " + str(self.chk),
        ):
            self.assertIn(want, lines)
        self.assertEqual(
            lines[-1],
            "lane: exempt (a micro round: the verifier alone, on its new items)",
        )
        self.assertNotIn(
            "x" * 300, "\n".join(lines)
        )  # the prep summary itself is not repeated
        self.assertLess(len(lines), 60)
        # Each section once: the delta (its line, files and commits) under its one title.
        titles = [
            x for x in lines if x.startswith("== ") and not x.startswith("== #7 ")
        ]
        self.assertEqual(len(titles), len(set(titles)), titles)
        self.assertEqual(sum(x.startswith("delta: ") for x in lines), 1)
        self.assertEqual(
            sum(x.startswith("own commits since our review") for x in lines), 1
        )


class TestSpecPost(unittest.TestCase):
    CFG = {
        "paths": {"root": "/r", "repo": "/r/repo"},
        "operator": {"github_login": "op"},
    }

    def run_it(
        self, spec_said, spec_code, check=None, posted=(["posted: u"], 0), devbase=False
    ):
        calls = []
        with mock.patch.object(
            dispatch.spec_mod, "write", return_value=(spec_said, spec_code)
        ), mock.patch.object(
            dispatch.premises,
            "load",
            return_value={"SRC": "/r/tmp/v7fu2/pr7_fu2.md", "DEVBASE": "a" * 10},
        ), mock.patch.object(
            dispatch.premises,
            "post_checked",
            lambda *a, **k: calls.append(("post", a, k)) or posted,
        ), mock.patch.object(
            dispatch.premises, "check", side_effect=check or []
        ), mock.patch.object(
            dispatch.gitops, "git", return_value=" rk/a.py | 2 +-\n 1 file changed\n"
        ):
            lines, code = dispatch.spec_post(
                self.CFG,
                None,
                7,
                "fu2",
                "/r/tmp/v7fu2",
                None,
                None,
                None,
                None,
                False,
                False,
                devbase,
                news=None,
            )
        return lines, code, calls

    def test_check_lines_stop_it(self):
        lines, code, calls = self.run_it(
            [
                "CHECK: k fails now and the notes do not account for it: re-verify",
                "wrote /r/tmp/spec7fu2.json: rk post ...",
            ],
            3,
        )
        self.assertEqual((code, calls), (3, []))
        self.assertEqual(
            lines[0],
            "CHECK: k fails now and the notes do not account for it: re-verify",
        )
        self.assertTrue(
            lines[-1].startswith(
                "spec-post: the spec has 1 CHECK line (above): nothing posted"
            )
        )
        lines, code, calls = self.run_it(["REFUSED: no notes.md"], 4)
        self.assertEqual((lines, code, calls), (["REFUSED: no notes.md"], 4, []))

    def test_a_clean_spec_posts(self):
        lines, code, calls = self.run_it(
            ["FAILOK x", "wrote /r/tmp/spec7fu2.json: rk post ..."], 0
        )
        self.assertEqual(
            (lines, code), (["wrote /r/tmp/spec7fu2.json: rk post ...", "posted: u"], 0)
        )
        self.assertEqual(
            calls[0][1][3:5], ("/r/tmp/v7fu2/pr7_fu2.md", "/r/tmp/spec7fu2.json")
        )

    def test_devbase_recheck(self):
        stop = [
            "live: x",
            "develop: bbbbbbbbbb",
            "PR files: 3",
            "rk/a.py",
            "premise: develop changed 1 of this PR's own files since DEVBASE aaaaaaaaaa (above): re-check them at "
            "bbbbbbbbbb (review_procedure step 4), then set DEVBASE: stop",
        ]
        held = (["live: x", "develop: bbbbbbbbbb", "conflicts: [] "], {"copy": "c"})
        lines, code, calls = self.run_it(
            ["wrote s"], 0, check=[(stop, None), held], devbase=True
        )
        self.assertEqual((code, calls), (4, []))  # nothing posted
        self.assertIn(
            "devbase-recheck: develop changed 1 of the PR's files from aaaaaaaaaa to bbbbbbbbbb:",
            lines,
        )
        self.assertIn("    rk/a.py | 2 +-", lines)
        self.assertTrue(
            lines[-1].startswith(
                "devbase-recheck: every other premise is true at bbbbbbbbbb (conflicts: [] "
                "). Run the saved probes"
            ),
            lines[-1],
        )
        self.assertIn(
            "unchanged: set `develop bbbbbbbbbb` in notes.md's premises: line and run spec-post again",
            lines[-1],
        )
        self.assertIn("`rk post 7 FILE --premises SPEC`", lines[-1])
        self.assertNotIn("and post;", lines[-1])
        other = (
            [
                "live: x",
                "premise: threads changed: now 6 5, the spec has 5 5 (total, resolved): stop",
            ],
            None,
        )
        lines, code, _ = self.run_it(
            ["wrote s"], 0, check=[(stop, None), other], devbase=True
        )
        self.assertEqual(
            lines[-1],
            "devbase-recheck: at bbbbbbbbbb another premise stops it too: premise: threads "
            "changed: now 6 5, the spec has 5 5 (total, resolved): stop",
        )
        lines, code, _ = self.run_it(
            ["wrote s"], 0, check=[(other[0], None)], devbase=True
        )
        self.assertEqual(
            lines[-1],
            "devbase-recheck: the stop is not develop's change to the PR's files: nothing to "
            "re-check",
        )


class TestWaitQuality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        (self.d / "facts.txt").write_text("facts\n")
        os.utime(self.d / "facts.txt", (1000, 1000))
        self.t = 10_000.0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.on_sleep:
            self.on_sleep(self.t)

    def wait(self, on_sleep=None, **kw):
        self.on_sleep = on_sleep
        return dispatch.wait_quality(self.d, clock=self.clock, sleep=self.sleep, **kw)

    def quality(self, row):
        (self.d / "quality.md").write_text(f"| Code quality | {row} |\n")
        os.utime(self.d / "quality.md", (2000, 2000))

    def test_ready_waiting_timeout(self):
        self.quality("🟢 Clean<br>• reuses helpers (read)")
        self.assertEqual(self.wait(), (f"ready: {self.d / 'quality.md'} is final", 0))
        self.quality(
            "pending<br>• in progress at abc"
        )  # pending: wait; the lane finishes 3 minutes in
        line, code = self.wait(
            on_sleep=lambda t: t >= 10_180 and self.quality("🟡 Minor<br>• one name")
        )
        self.assertEqual((line[:6], code), ("ready:", 0))
        self.assertFalse((self.d / dispatch.MARK).exists())
        self.quality("pending")
        line, code = (
            self.wait()
        )  # each call returns at its slice, under the prompt cache's 5 minutes
        self.assertEqual(
            (line, code),
            ("waiting: quality.md not final yet (4.5 of 20 min): call it again", 0),
        )
        self.assertLessEqual(self.t - 10_180, dispatch.SLICE_MIN * 60 + 11)
        for _ in range(3):
            self.assertEqual(self.wait()[0][:8], "waiting:")
        line, code = (
            self.wait()
        )  # the fifth call reaches the cap, 20 minutes after the first
        self.assertEqual(code, 1)
        self.assertTrue(line.startswith("timeout: quality.md not final after 20 min"))
        (
            self.d / "quality.md"
        ).unlink()  # missing, or older than facts.txt (a stopped round's): not final
        self.assertFalse(dispatch.quality_final(self.d))
        self.quality("🟢 Clean")
        os.utime(self.d / "quality.md", (500, 500))
        self.assertFalse(dispatch.quality_final(self.d))


class TestDriverAndLoopLines(unittest.TestCase):
    READY = (
        "REVIEW_READY #11111 (follow-up): no drafter: verify without a draft, checks done. Inputs: x. Verify "
        "alone, then `rk post 11111 FILE --premises SPEC`"
    )
    ADVANCE = "\n".join(
        [
            "REVIEW_DROP #9: closed or head moved before verification; will re-pick",
            READY,
            "#22222 changed substantively: 2 files, e.g. a.py",
            "REVIEW_START #22222 (follow-up): staging " "started at e44a46f2a0",
            "SHADOW_READY #14: x",
            "",
        ]
    )

    def test_driver_lines(self):
        lines = tickfilter.driver_lines(self.ADVANCE)
        self.assertEqual(
            lines[0],
            "DRIVER_EXIT: 3 lines for the orchestrator: SHADOW_READY #14, REVIEW_DROP #9, "
            "REVIEW_READY #11111",
        )
        self.assertEqual(lines[-1], self.READY)
        self.assertEqual(
            set(lines[1:-1]),
            {
                "SHADOW_READY #14: x",
                "REVIEW_DROP #9: closed or head moved before verification; will re-pick",
            },
        )
        self.assertEqual(tickfilter.driver_lines("REVIEW #7: worker running\n"), [])

    @unittest.skipIf(sys.platform != "linux", "the driver runs on Linux (flock)")
    def test_the_driver_ends_on_the_ready_line(self):
        """rk review-driver (tools/review_driver.py runs it) prints its last advance whole, then the lines to act on."""
        from rkmon_lib.store import Store, empty_state, now

        printed = []
        with tempfile.TemporaryDirectory() as d:
            store = Store(pathlib.Path(d))
            st = empty_state()
            st["last_tick"] = now()
            store.save(st)
            with mock.patch.object(rkmon, "load_config", lambda: {}), mock.patch.object(
                rkmon.review,
                "advance",
                lambda cfg, st, store, plan: self.ADVANCE.rstrip("\n").split("\n"),
            ), mock.patch.object(rkmon.daylog, "from_lines", lambda *a: None):
                rc = rkmon.cmd_review_driver(
                    {}, store, sleep=lambda s: None, out=printed.append
                )
        text = "\n".join(printed)
        out = text.splitlines()
        self.assertEqual(rc, 0)
        self.assertTrue(
            text.startswith(self.ADVANCE.rstrip("\n")), text
        )  # the advance, whole
        self.assertEqual(out[-1], self.READY)
        self.assertEqual(
            out[-5:-3],
            [
                "---",
                "DRIVER_EXIT: 3 lines for the orchestrator: SHADOW_READY #14, REVIEW_DROP #9, "
                "REVIEW_READY #11111",
            ],
        )

    def test_the_checks_name_the_shared_build_in_a_review_only(self):
        plan = {
            "families": [],
            "tests": ["library/tests/test_a.py"],
            "build": True,
            "golden": False,
            "engine_wide": False,
            "outside_rocke": [],
            "libparity": False,
            "_dirs": {},
        }
        cfg = {
            "rocke_path": "rk",
            "repo": "o/r",
            "paths": {"working_dir": "/w"},
        }
        script = review.checks_script(cfg, 7, "a" * 40, "follow-up", plan, None)
        sec = script.split("sec 'Native build + ctest'", 1)[1].split("\nsec ", 1)[0]
        self.assertIn(
            f'> "$OUT/{review.SHARED_BUILD}"; echo "shared build: $SB/${{WTN}} (core: librocke_core and '
            "the native tests; pybind: rocke_engine) of the tree $W at",
            sec,
        )
        self.assertLess(
            sec.index("native_ok"), sec.index(review.SHARED_BUILD)
        )  # only once the build succeeded
        self.assertEqual(
            subprocess.run(
                ["bash", "-n"], input=script, capture_output=True, text=True
            ).returncode,
            0,
        )
        gate = review.checks_script(
            cfg, 7, "a" * 40, "first-round", plan, None, mode="gate"
        )
        self.assertNotIn(review.SHARED_BUILD, gate)


class TestCli(unittest.TestCase):
    def setUp(self):
        """A configuration and a state of the test's own: the package's config.json is blank, and a developer's copy
        holds a machine's settings, which no test reads."""
        from rkmon_lib import config

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = pathlib.Path(tmp.name)
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
        )
        c["paths"] = dict(c["paths"], working_dir=str(base / "work"))
        cfg = config.normalize(c)
        store = Store(base / "state")
        store.save(empty_state())
        for patch in (
            mock.patch.object(rkmon, "load_config", lambda: cfg),
            mock.patch.object(rkmon, "Store", lambda: store),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_wait_quality_cli(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "quality.md").write_text("| Code quality | 🟢 Clean |\n")
            with mock.patch("builtins.print") as p:
                self.assertEqual(rkmon.main(["wait-quality", d]), 0)
            p.assert_called_once_with(
                f"ready: {pathlib.Path(d) / 'quality.md'} is final"
            )

    def test_a_bad_round_is_refused(self):
        with mock.patch("builtins.print") as p:
            self.assertEqual(rkmon.main(["dispatch-pack", "7", "--round", "x9"]), 4)
        self.assertIn("REFUSED: --round x9", p.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
