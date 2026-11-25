# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for Phase 2: sanity and bisect scripts, the ABI history invariant, and incident handling."""

import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import guard, jobs  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

CFG = {
    "rocke_path": "rk",
    "repo": "ROCm/r",
    "paths": {"working_dir": "/nonexistent", "repo": "/nonexistent"},
    "revert_offer_after_min": 120,
}
TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "abi_check.py"


def bash_ok(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)


class TestScripts(unittest.TestCase):
    def test_sanity_and_bisect_scripts_parse(self):
        s = guard.sanity_script(CFG, "a" * 40, "/x/wtsanity", "/x/out", ["gemm"], True)
        self.assertEqual(bash_ok(s).returncode, 0, bash_ok(s).stderr)
        for sec in (
            "st build",
            "st golden",
            "st dg",
            "st byteid",
            "st byteid_all",
            "st abi",
        ):
            self.assertIn(sec, s)
        only = guard.sanity_script(CFG, "a" * 40, "/x/wtb", "/x/out", [], False, ["dg"])
        self.assertIn("st dg", only)
        self.assertNotIn("st build", only)
        b = guard.bisect_script(
            CFG, "INC-2026-01-01-1", [("b" * 40, 12), ("c" * 40, None)], ["dg"], []
        )
        self.assertEqual(bash_ok(b).returncode, 0, bash_ok(b).stderr)
        self.assertIn("culprit " + "b" * 40 + " 12", b)


class TestAbiCheck(unittest.TestCase):
    def run_check(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as f:
            f.write(text)
        self.addCleanup(pathlib.Path(f.name).unlink)
        return subprocess.run(
            [sys.executable, "-I", str(TOOL), f.name], capture_output=True, text=True
        ).stdout.strip()

    def test_invariant(self):
        ok = "/* x\n * 2: a\n * 3: b */\n#define ROCKE_ABI_VERSION 3\n/* w\n * 2: c */\n#define ROCKE_RECIPE_ABI 2\n"
        self.assertTrue(self.run_check(ok).startswith("OK"))
        dup = ok.replace(" * 3: b", " * 3: b\n * 3: again")
        self.assertIn("appear more than once", self.run_check(dup))
        behind = ok.replace("ROCKE_ABI_VERSION 3", "ROCKE_ABI_VERSION 4")
        self.assertIn("has no history line", self.run_check(behind))


class TestIncidents(unittest.TestCase):
    def test_open_freeze_close(self):
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            inc = guard.open_incident(
                store, st, "a" * 40, ["dg: mismatch"], "b" * 40, [("c" * 12, 7)]
            )
            self.assertEqual(st["frozen"]["incident"], inc["id"])
            again = guard.open_incident(
                store, st, "d" * 40, ["build: failed"], "b" * 40, []
            )
            self.assertEqual(
                again["id"], inc["id"]
            )  # one open incident at a time; signals merged
            self.assertEqual(len(guard.incidents(store)), 1)
            closed = guard.close_incident(store, st, "e" * 40, fixed_by=9)
            self.assertEqual((closed["status"], closed["fixed_by"]), ("closed", 9))
            self.assertIsNone(st["frozen"])

    def test_an_attributable_break_during_a_ci_only_incident_gets_its_own(self):
        """An attributable golden break during a CI-only incident (no candidates) gets an incident of its own, and
        its bisect has candidates to run."""
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            ci = guard.open_incident(
                store,
                st,
                "a" * 40,
                ["develop CI: shard 2/2"],
                "b" * 40,
                [],
                ci_jobs=["shard 2/2"],
            )
            gold = guard.open_incident(
                store,
                st,
                "c" * 40,
                ["golden: not PASS"],
                "a" * 40,
                [("d" * 12, 10808), ("c" * 12, 2222)],
            )
            self.assertNotEqual(gold["id"], ci["id"])
            self.assertEqual([n for _c, n in gold["candidates"]], [10808, 2222])
            self.assertEqual(st["frozen"]["incident"], gold["id"])
            # A CI red with no candidates folds into the newest open incident; a later sanity red into the attributable
            again = guard.open_incident(
                store, st, "e" * 40, ["develop CI: shard 1/2"], "a" * 40, []
            )
            self.assertEqual(again["id"], gold["id"])
            more = guard.open_incident(
                store, st, "f" * 40, ["dg: mismatch"], "c" * 40, [("f" * 12, 10909)]
            )
            self.assertEqual(more["id"], gold["id"])
            self.assertEqual([n for _c, n in more["candidates"]], [10808, 2222, 10909])
            # The bisect takes the attributable one
            with mock.patch.object(
                guard.gitops, "git", lambda repo, *a: "0" * 40
            ), mock.patch.object(
                guard.jobs,
                "start",
                lambda store, kind, key, script, timeout_min: "job-1",
            ):
                self.assertTrue(
                    guard.start_bisect(CFG, st, store)[0].startswith(
                        f"BISECT_START {gold['id']}: 3"
                    )
                )
            # A sanity green records on the incidents with sanity signals only; the freeze moves, then lifts
            self.assertIsNone(
                guard.close_incident(store, st, "9" * 40, kind="sanity")
            )  # gold still has CI red
            closed = guard.close_incident(
                store, st, "9" * 40, kind="ci", inc_id=gold["id"]
            )
            self.assertEqual(closed["id"], gold["id"])
            self.assertEqual(
                st["frozen"]["incident"], ci["id"]
            )  # the 2345 one is still open
            self.assertEqual(
                guard.close_incident(store, st, "9" * 40, kind="ci", inc_id=ci["id"])[
                    "id"
                ],
                ci["id"],
            )
            self.assertIsNone(st["frozen"])

    def test_each_kind_of_signal_clears_on_its_own_green(self):
        """A green sanity run does not close a Windows CI-test incident while develop CI still runs, nor does a later
        commit whose CI skipped the rocKE jobs; only the failed job passing again counts.
        """
        win = "Windows::release / Test gfx110X-all / Test hipkernelprovider (shard 1/2)"
        red, other, later = "a" * 40, "c" * 40, "9" * 40

        def ci(verdict, passed=()):
            run = {
                "id": 1,
                "status": "completed",
                "conclusion": verdict,
                "rocke_passed": list(passed),
            }
            if verdict == "failure":
                run["rocke_failed"] = [win]
            return {"complete": True, "runs": {"TheRock Multi-Arch CI": run}}

        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            inc = guard.open_incident(
                store, st, red, [f"develop CI: {win}"], "b" * 40, [], ci_jobs=[win]
            )
            self.assertIsNone(guard.close_incident(store, st, other, kind="sanity"))
            self.assertTrue(st["frozen"])
            st["develop"]["ci"] = {
                other: ci("success"),
                red: ci("failure"),
            }  # newest first; rocKE jobs skipped
            self.assertEqual(guard.advance(CFG, st, store, []), [])
            self.assertTrue(st["frozen"])
            st["develop"]["ci"] = {
                later: ci("failure"),
                other: ci("success"),
                red: ci("success", [win]),
            }
            self.assertEqual(
                guard.advance(CFG, st, store, []), []
            )  # a newer red: no evidence past it
            st["develop"]["ci"] = {
                other: ci("success"),
                red: ci("success", [win]),
            }  # the red SHA's re-run passed
            lines = guard.advance(CFG, st, store, [])
            self.assertIn(
                f"INCIDENT_CLOSED {inc['id']}: develop CI green again at aaaaaaaaaa (the failed rocKE jobs "
                "passed)",
                lines,
            )
            self.assertIsNone(st["frozen"])
            st["develop"]["ci"] = {}
            guard.open_incident(store, st, "d" * 40, ["dg: mismatch"], "c" * 40, [])
            guard.open_incident(
                store, st, "e" * 40, ["develop CI: build"], "c" * 40, []
            )
            self.assertIsNone(
                guard.close_incident(store, st, "f" * 40, kind="ci")
            )  # sanity still red
            guard.open_incident(
                store, st, "f" * 40, ["develop CI: tests"], "c" * 40, []
            )  # CI red again
            self.assertIsNone(
                guard.close_incident(store, st, "0" * 40, kind="sanity")
            )  # that voided the CI green
            self.assertEqual(
                guard.close_incident(store, st, "1" * 40, kind="ci")["status"], "closed"
            )

    def test_a_short_red_sha_still_stops_the_walk(self):
        """The DEVELOP_RED event's 10-char SHA matches the 40-char CI key, so an incident does not close at the older
        green SHA while the red SHA's re-run is still running."""
        job = "Linux::release / Test gfx94X-dcgpu / Test hipkernelprovider (shard 1/2)"
        red, older = "7cd54ff9cd" + "7" * 30, "c9833f2d81" + "0" * 30

        def ci(verdict, passed=()):
            return {
                "complete": verdict != "pending",
                "runs": {
                    "TheRock Multi-Arch CI": {
                        "id": 1,
                        "status": (
                            "in_progress" if verdict == "pending" else "completed"
                        ),
                        "conclusion": None if verdict == "pending" else verdict,
                        "rocke_passed": list(passed),
                    }
                },
            }

        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            inc = guard.open_incident(
                store, st, red[:10], [f"develop CI: {job}"], older, [], ci_jobs=[job]
            )
            st["develop"]["ci"] = {
                red: ci("pending"),
                older: ci("success", [job]),
            }  # the re-run is running
            self.assertEqual(guard.advance(CFG, st, store, []), [])
            self.assertTrue(st["frozen"])
            st["develop"]["ci"] = {
                red: ci("success", [job]),
                older: ci("success", [job]),
            }  # the re-run passed
            self.assertIn(
                f"INCIDENT_CLOSED {inc['id']}: develop CI green again at 7cd54ff9cd (the failed rocKE jobs "
                "passed)",
                guard.advance(CFG, st, store, []),
            )

    def test_an_aged_incident_without_a_culprit_is_told_once_not_offered_a_revert(self):
        """An incident with no culprit (a test past its timeout) never asks "revert #?" every tick."""
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            inc = guard.open_incident(
                store, st, "a" * 40, ["develop CI: shard"], "b" * 40, []
            )
            incs = guard.incidents(store)
            incs[0].update(start="2026-10-31T00:00:00Z", issue=2345)
            guard.save_incidents(store, incs)
            first = [ln for ln in guard.advance(CFG, st, store, []) if "INC" in ln]
            self.assertEqual(len(first), 1, first)
            self.assertTrue(
                first[0].startswith(f"INCIDENT_AGED {inc['id']}: open "), first
            )
            self.assertIn("the fix for #2345", first[0])
            self.assertEqual(
                [ln for ln in guard.advance(CFG, st, store, []) if "INC" in ln], []
            )  # told once
            incs = guard.incidents(store)
            incs[0]["culprit"] = 1234
            guard.save_incidents(store, incs)
            again = [ln for ln in guard.advance(CFG, st, store, []) if "INC" in ln]
            self.assertTrue(
                again and again[0].startswith(f"REVERT_OFFER {inc['id']}:"), again
            )
            self.assertIn("revert of #1234", again[0])

    def test_no_revert_offer_once_the_fix_merged(self):
        """Once #3456 fixes #3333's break, the tick no longer offers to revert #1111."""
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            inc = guard.open_incident(
                store, st, "a" * 40, ["develop CI: shard"], "b" * 40, []
            )
            incs = guard.incidents(store)
            incs[0].update(start="2026-11-04T00:00:00Z", culprit=1111, fixed_by=3456)
            guard.save_incidents(store, incs)
            st["prs"] = {"3456": {"n": 3456, "status": "merged", "merge_sha": "f" * 40}}
            lines = [ln for ln in guard.advance(CFG, st, store, []) if "INC" in ln]
            self.assertEqual(len(lines), 1, lines)
            self.assertTrue(
                lines[0].startswith(
                    f"INCIDENT_FIX_MERGED {inc['id']}: #3456 merged as ffffffffff"
                ),
                lines,
            )
            self.assertEqual(
                [ln for ln in guard.advance(CFG, st, store, []) if "INC" in ln], []
            )  # told once

    def test_a_red_before_a_merged_fix_is_old_news(self):
        """Develop runs on commits before #3456's fix stop feeding the incident once the fix lands."""
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            job = "Linux::release / Test gfx94X-dcgpu / Test hipkernelprovider (shard 1/2)"
            guard.open_incident(
                store, st, "a" * 40, ["develop CI: x"], "b" * 40, [], ci_jobs=[job]
            )
            incs = guard.incidents(store)
            incs[0].update(fixed_by=3456, status="closed")
            guard.save_incidents(store, incs)
            st["frozen"] = None
            st["prs"] = {"3456": {"n": 3456, "status": "merged", "merge_sha": "f" * 40}}
            ev = {"kind": "DEVELOP_RED", "sha": "c" * 10, "rocke_jobs": [job]}
            with mock.patch.object(guard.gitops, "is_ancestor", return_value=True):
                lines = guard.advance(CFG, st, store, [ev])
            self.assertTrue(
                any(
                    ln.startswith(
                        "DEVELOP_RED_STALE cccccccccc: its commit predates #3456"
                    )
                    for ln in lines
                ),
                lines,
            )
            self.assertFalse(st.get("frozen"))
            self.assertEqual([i["status"] for i in guard.incidents(store)], ["closed"])
            # after the fix, or a job that incident never saw fail: a new incident
            for anc, jobs_red in (
                (False, [job]),
                (True, ["Windows::release / Test gfx110X-all / Test other"]),
            ):
                with mock.patch.object(guard.gitops, "is_ancestor", return_value=anc):
                    lines = guard.advance(
                        CFG, st, store, [dict(ev, rocke_jobs=jobs_red)]
                    )
                self.assertTrue(
                    any(ln.startswith("INCIDENT_OPEN") for ln in lines), lines
                )
                incs = guard.incidents(store)
                incs[-1]["status"] = "closed"
                guard.save_incidents(store, incs)
                st["frozen"] = None

    def test_red_sanity_opens_incident_with_single_culprit(self):
        with tempfile.TemporaryDirectory() as d:
            store, st = Store(pathlib.Path(d)), empty_state()
            out = pathlib.Path(d) / "out"
            out.mkdir()
            (out / "status").write_text(
                "build pass ok\ndg fail mismatch at target_intrinsics\n"
            )
            job = jobs.start(store, "sanity", "x", "true", timeout_min=5)
            for _ in range(50):
                if jobs.status(store, job) != "running":
                    break
                time.sleep(0.1)
            st["guard"] = {
                "running": {
                    "sha": "a" * 40,
                    "job": job,
                    "families": [],
                    "full": False,
                    "landed": [("c" * 12, 10606)],
                    "base": "b" * 40,
                }
            }
            st["jobs"]["heavy"] = job
            with mock.patch.object(guard, "_out", lambda cfg, sha: str(out)):
                lines = guard.advance(CFG, st, store, [])
            self.assertTrue(
                any(line.startswith("INCIDENT_OPEN") for line in lines), lines
            )
            self.assertIn("CULPRIT", " ".join(lines))
            self.assertEqual(guard.incidents(store)[0]["culprit"], 10606)
            self.assertTrue(st["frozen"])


if __name__ == "__main__":
    unittest.main()
