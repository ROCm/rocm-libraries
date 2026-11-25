# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""snapshot.corrected_rollup / correct_rollups: a cancelled duplicate whose check also passed is not a CI failure."""
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rkmon_lib import gh, snapshot  # noqa: E402


def run(name, status="completed", conclusion="success"):
    return {"name": name, "status": status, "conclusion": conclusion}


class TestCorrectedRollup(unittest.TestCase):
    def test_a_cancelled_twin_of_a_passing_check_is_ignored(self):
        runs = [
            run("therock-pr-bot", conclusion="cancelled"),
            run("therock-pr-bot"),
            run("pre-commit"),
        ]
        self.assertEqual(
            snapshot.corrected_rollup(
                runs, [{"context": "base-freshness", "state": "success"}]
            ),
            "SUCCESS",
        )

    def test_still_running_is_pending(self):
        runs = [
            run("therock-pr-bot", conclusion="cancelled"),
            run("therock-pr-bot"),
            run("Test shard 1/2", status="in_progress", conclusion=None),
        ]
        self.assertEqual(snapshot.corrected_rollup(runs, []), "PENDING")

    def test_a_cancelled_run_with_no_passing_twin_and_a_failure_stay_failures(self):
        self.assertEqual(
            snapshot.corrected_rollup([run("build", conclusion="cancelled")], []),
            "FAILURE",
        )
        self.assertEqual(
            snapshot.corrected_rollup(
                [run("build", conclusion="failure"), run("build")], []
            ),
            "FAILURE",
        )
        self.assertEqual(
            snapshot.corrected_rollup(
                [run("build")], [{"context": "x", "state": "error"}]
            ),
            "FAILURE",
        )


class TestCorrectRollups(unittest.TestCase):
    def test_only_failure_rollups_are_read_and_final_answers_are_cached(self):
        prs = [
            {"n": 1, "head": "a" * 40, "ci_rollup": "FAILURE"},
            {"n": 2, "head": "b" * 40, "ci_rollup": "SUCCESS"},
        ]
        replies = [[run("bot", conclusion="cancelled"), run("bot")], []]
        cache = {"9": {"head": "c" * 40, "state": "SUCCESS"}}
        with mock.patch.object(gh, "rest", side_effect=replies) as rest:
            out = snapshot.correct_rollups({"repo": "ROCm/r"}, prs, cache)
        self.assertEqual(rest.call_count, 2)
        self.assertEqual(
            (out[0]["ci_rollup"], out[0]["ci_rollup_raw"]), ("SUCCESS", "FAILURE")
        )
        self.assertEqual(out[1]["ci_rollup"], "SUCCESS")
        self.assertNotIn("9", cache)
        again = [{"n": 1, "head": "a" * 40, "ci_rollup": "FAILURE"}]
        with mock.patch.object(gh, "rest") as rest:
            snapshot.correct_rollups({"repo": "ROCm/r"}, again, cache)
        rest.assert_not_called()
        self.assertEqual(again[0]["ci_rollup"], "SUCCESS")

    def test_a_read_error_keeps_githubs_answer(self):
        prs = [{"n": 1, "head": "a" * 40, "ci_rollup": "FAILURE"}]
        with mock.patch.object(gh, "rest", side_effect=gh.GhError("HTTP 502")):
            snapshot.correct_rollups({"repo": "ROCm/r"}, prs, {})
        self.assertEqual(prs[0]["ci_rollup"], "FAILURE")
        self.assertNotIn("ci_rollup_raw", prs[0])


if __name__ == "__main__":
    unittest.main()
