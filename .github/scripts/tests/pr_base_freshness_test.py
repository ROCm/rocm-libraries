# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import pr_base_freshness as pbf
from pr_base_freshness import PullRequest, Verdict


NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
CUTOFF = "c0ff33"
PR = PullRequest(number=42, head_sha="deadbeef", draft=False)


def iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeGitHub:
    """In-memory stand-in for the REST client, keyed by request path prefix."""

    def __init__(self, responses: dict, failures: tuple = (), budget: int = 10_000):
        self._responses = responses
        self._failures = failures
        self._budget = budget
        self.paths: list[str] = []

    def has_budget(self, cost: int = 1) -> bool:
        return self._budget >= cost

    def get(self, path: str, **params):
        self.paths.append(path)
        if path.startswith(self._failures):
            raise pbf.GitHubError(f"boom: {path}")
        for prefix, response in self._responses.items():
            if path.startswith(prefix):
                return response
        raise AssertionError(f"unexpected path {path}")


def comparison(status: str, behind_by: int = 0, merge_base: str = "base1"):
    return {
        "status": status,
        "behind_by": behind_by,
        "merge_base_commit": {"sha": merge_base},
    }


class EvaluateTest(unittest.TestCase):
    def test_head_containing_cutoff_is_fresh(self):
        for status in ("ahead", "identical"):
            gh = FakeGitHub({f"compare/{CUTOFF}": comparison(status)})
            verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
            self.assertTrue(verdict.fresh)
            self.assertEqual(verdict.missing_commits, 0)
            # A fresh PR must cost exactly one request.
            self.assertEqual(len(gh.paths), 1)

    def test_diverged_head_is_stale_and_dated(self):
        gh = FakeGitHub(
            {
                f"compare/{CUTOFF}": comparison("diverged", behind_by=7),
                "compare/base1": {
                    "commits": [{"commit": {"committer": {"date": iso(5.5)}}}]
                },
            }
        )
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
        self.assertFalse(verdict.fresh)
        self.assertEqual(verdict.missing_commits, 7)
        self.assertAlmostEqual(verdict.stale_days, 5.5, places=2)

    def test_stale_verdict_survives_a_failed_dating_call(self):
        gh = FakeGitHub(
            {f"compare/{CUTOFF}": comparison("behind", behind_by=3)},
            failures=("compare/base1",),
        )
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
        self.assertFalse(verdict.fresh)
        self.assertIsNone(verdict.stale_days)

    def test_api_failure_fails_open(self):
        gh = FakeGitHub({}, failures=("compare/",))
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
        self.assertTrue(verdict.fresh)
        self.assertIsNotNone(verdict.error)


class BudgetTest(unittest.TestCase):
    def test_pr_is_not_started_without_budget_to_publish_it(self):
        gh = FakeGitHub({f"compare/{CUTOFF}": comparison("ahead")}, budget=1)
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
        self.assertTrue(verdict.skipped)
        self.assertEqual(gh.paths, [])

    def test_cosmetic_dating_call_is_dropped_first(self):
        # Enough budget to answer the question, not enough for the day count.
        gh = FakeGitHub(
            {f"compare/{CUTOFF}": comparison("diverged", behind_by=7)},
            budget=pbf.REQUESTS_PER_PR,
        )
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF, NOW)
        self.assertFalse(verdict.skipped)
        self.assertFalse(verdict.fresh)
        self.assertEqual(verdict.missing_commits, 7)
        self.assertIsNone(verdict.stale_days)
        self.assertEqual(len(gh.paths), 1)

    def test_reserve_governs_has_budget(self):
        gh = pbf.GitHub("o/r", "t", reserve=200)
        # Nothing observed yet, so the first request is always allowed.
        self.assertTrue(gh.has_budget(50))
        gh.rate_limit_remaining = 260
        self.assertTrue(gh.has_budget(50))
        self.assertFalse(gh.has_budget(61))

    def test_request_exceptions_become_github_errors(self):
        gh = pbf.GitHub("o/r", "t")

        class ExplodingSession:
            def request(self, *args, **kwargs):
                raise requests.ConnectTimeout("connection timed out")

        gh.session = ExplodingSession()
        # A bare RequestException here would escape evaluate() and abort the
        # whole run, so it must arrive as GitHubError with the cause kept.
        with self.assertRaises(pbf.GitHubError) as caught:
            gh.get("compare/a...b")
        self.assertIsInstance(caught.exception.__cause__, requests.RequestException)
        self.assertEqual(gh.request_count, 1)


class ReportingTest(unittest.TestCase):
    def test_report_mode_never_fails(self):
        stale = Verdict(fresh=False, missing_commits=4, stale_days=6.0)
        self.assertEqual(pbf.status_state(stale, enforcing=False), "success")
        self.assertEqual(pbf.status_state(stale, enforcing=True), "failure")

    def test_unevaluated_prs_never_fail(self):
        unknown = Verdict(fresh=True, error="boom")
        self.assertEqual(pbf.status_state(unknown, enforcing=True), "success")

    def test_report_mode_description_is_marked(self):
        stale = Verdict(fresh=False, missing_commits=4, stale_days=6.0)
        self.assertTrue(
            pbf.describe(stale, "develop", 3, enforcing=False).startswith("WOULD FAIL:")
        )
        self.assertFalse(
            pbf.describe(stale, "develop", 3, enforcing=True).startswith("WOULD FAIL:")
        )

    def test_description_fits_githubs_limit(self):
        stale = Verdict(fresh=False, missing_commits=99999, stale_days=123.456)
        text = pbf.describe(
            stale, "a-very-long-base-branch-name" * 6, 3, enforcing=False
        )
        self.assertLessEqual(len(text), pbf.MAX_DESCRIPTION_LENGTH)


if __name__ == "__main__":
    unittest.main()
