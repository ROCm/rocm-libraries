# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import os
import sys
import threading
import time
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

    def claim_budget(self, cost: int) -> bool:
        if self._budget < cost:
            return False
        self._budget -= cost
        return True

    def release_budget(self, cost: int) -> None:
        self._budget += cost

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
            verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
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
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
        self.assertFalse(verdict.fresh)
        self.assertEqual(verdict.missing_commits, 7)
        self.assertAlmostEqual(pbf.stale_days(verdict, NOW), 5.5, places=2)

    def test_stale_verdict_survives_a_failed_dating_call(self):
        gh = FakeGitHub(
            {f"compare/{CUTOFF}": comparison("behind", behind_by=3)},
            failures=("compare/base1",),
        )
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
        self.assertFalse(verdict.fresh)
        self.assertIsNone(verdict.oldest_missing)

    def test_api_failure_fails_open(self):
        gh = FakeGitHub({}, failures=("compare/",))
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
        self.assertTrue(verdict.fresh)
        self.assertIsNotNone(verdict.error)


class BudgetTest(unittest.TestCase):
    def test_pr_is_not_started_without_budget_to_publish_it(self):
        gh = FakeGitHub({f"compare/{CUTOFF}": comparison("ahead")}, budget=1)
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
        self.assertTrue(verdict.skipped)
        self.assertEqual(gh.paths, [])

    def test_cosmetic_dating_call_is_dropped_first(self):
        # Enough budget to answer the question, not enough for the day count.
        gh = FakeGitHub(
            {f"compare/{CUTOFF}": comparison("diverged", behind_by=7)},
            budget=pbf.REQUESTS_PER_PR,
        )
        verdict = pbf.evaluate(gh, PR, "develop", CUTOFF)
        self.assertFalse(verdict.skipped)
        self.assertFalse(verdict.fresh)
        self.assertEqual(verdict.missing_commits, 7)
        self.assertIsNone(verdict.oldest_missing)
        self.assertEqual(len(gh.paths), 1)

    def test_reserve_governs_has_budget(self):
        gh = pbf.GitHub("o/r", "t", reserve=200)
        # Nothing observed yet, so the first request is always allowed.
        self.assertTrue(gh.has_budget(50))
        gh.rate_limit_remaining = 260
        self.assertTrue(gh.has_budget(50))
        self.assertFalse(gh.has_budget(61))

    def test_a_claim_is_held_against_later_callers(self):
        gh = pbf.GitHub("o/r", "t", reserve=100)
        gh.rate_limit_remaining = 106
        # Nothing has been spent and `remaining` has not moved, so a plain
        # check would let all three of these through.
        self.assertTrue(gh.claim_budget(3))
        self.assertTrue(gh.claim_budget(3))
        self.assertFalse(gh.claim_budget(3))
        gh.release_budget(3)
        self.assertTrue(gh.claim_budget(3))

    def test_concurrent_workers_cannot_all_clear_the_same_floor(self):
        gh = pbf.GitHub("o/r", "t", reserve=1000)
        gh.rate_limit_remaining = 1030
        # Headroom for exactly ten claims of three, contended by more workers
        # than that, all starting from the same observed `remaining`.
        workers = pbf.MAX_WORKERS
        start = threading.Barrier(workers)
        admitted = []

        def claim():
            start.wait()
            admitted.append(gh.claim_budget(pbf.REQUESTS_PER_PR))

        threads = [threading.Thread(target=claim) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(admitted), 10)

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


class PublishedStatusTest(unittest.TestCase):
    def test_reads_back_our_own_context(self):
        gh = FakeGitHub(
            {
                "commits/": {
                    "statuses": [
                        {"context": "other", "state": "failure", "description": "no"},
                        {
                            "context": "base-freshness",
                            "state": "success",
                            "description": "Base is within 3 days of develop",
                        },
                    ]
                }
            }
        )
        self.assertEqual(
            pbf.current_status(gh, "deadbeef", "base-freshness"),
            ("success", "Base is within 3 days of develop"),
        )

    def test_absent_context_reads_as_nothing_published(self):
        gh = FakeGitHub({"commits/": {"statuses": []}})
        self.assertIsNone(pbf.current_status(gh, "deadbeef", "base-freshness"))

    def test_unreadable_status_does_not_suppress_the_post(self):
        # Returning None makes the caller post rather than assume it matches.
        gh = FakeGitHub({}, failures=("commits/",))
        self.assertIsNone(pbf.current_status(gh, "deadbeef", "base-freshness"))

    def test_skipped_pr_is_not_read_back(self):
        gh = FakeGitHub({}, budget=0)
        assessment = pbf.assess(gh, PR, "develop", CUTOFF, "base-freshness")
        self.assertTrue(assessment.verdict.skipped)
        self.assertIsNone(assessment.published)
        self.assertEqual(gh.paths, [])


class PacerTest(unittest.TestCase):
    def test_first_write_is_not_delayed(self):
        pacer = pbf.PostPacer(per_minute=60)
        start = time.monotonic()
        pacer.wait()
        self.assertLess(time.monotonic() - start, 0.1)

    def test_subsequent_writes_are_spaced(self):
        pacer = pbf.PostPacer(per_minute=600)  # 100ms apart
        pacer.wait()
        start = time.monotonic()
        pacer.wait()
        self.assertGreaterEqual(time.monotonic() - start, 0.05)

    def test_pacing_can_be_disabled(self):
        pacer = pbf.PostPacer(per_minute=0)
        pacer.wait()
        start = time.monotonic()
        pacer.wait()
        self.assertLess(time.monotonic() - start, 0.1)


class RetryableErrorTest(unittest.TestCase):
    def test_transport_failures_are_retryable(self):
        # No response at all, so the next attempt is as good as this one.
        self.assertTrue(pbf.GitHubError("timed out").retryable)

    def test_rate_limits_and_server_errors_are_retryable(self):
        for code in (403, 429, 500, 502, 503):
            self.assertTrue(pbf.GitHubError("x", status_code=code).retryable)

    def test_client_errors_are_not_retryable(self):
        for code in (404, 422):
            self.assertFalse(pbf.GitHubError("x", status_code=code).retryable)


class RetryBudgetTest(unittest.TestCase):
    def test_spending_is_capped_at_what_is_left(self):
        budget = pbf.RetryBudget(10.0)
        self.assertEqual(budget.spend(4.0), 4.0)
        self.assertEqual(budget.spend(30.0), 6.0)
        self.assertEqual(budget.spend(1.0), 0.0)


class FakePoster:
    """Records status posts, raising each queued error before succeeding."""

    def __init__(self, errors: list):
        self._errors = list(errors)
        self.attempts = 0

    def post(self, path: str, payload: dict):
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return {}


class PublishStatusTest(unittest.TestCase):
    """A status that never lands is what an enforcing ruleset cannot survive."""

    def publish(self, errors, budget=pbf.POST_RETRY_BUDGET_SECONDS):
        gh = FakePoster(errors)
        slept: list[float] = []
        error = pbf.publish_status(
            gh,
            pbf.PostPacer(per_minute=0),
            pbf.RetryBudget(budget),
            "deadbeef",
            {"state": "success"},
            sleep=slept.append,
        )
        return gh, error, slept

    def test_first_attempt_succeeds_without_sleeping(self):
        gh, error, slept = self.publish([])
        self.assertIsNone(error)
        self.assertEqual(gh.attempts, 1)
        self.assertEqual(slept, [])

    def test_secondary_rate_limit_is_retried_when_told_to_wait(self):
        # 403 with Retry-After is how the content-generation limit reports
        # itself, which is the write failure a sweep is most likely to hit.
        gh, error, slept = self.publish(
            [pbf.GitHubError("limited", status_code=403, retry_after=7)]
        )
        self.assertIsNone(error)
        self.assertEqual(gh.attempts, 2)
        self.assertEqual(slept, [7])

    def test_client_errors_are_not_retried(self):
        gh, error, slept = self.publish(
            [pbf.GitHubError("gone", status_code=404)] * pbf.POST_ATTEMPTS
        )
        self.assertIn("gone", error)
        self.assertEqual(gh.attempts, 1)
        self.assertEqual(slept, [])

    def test_persistent_failure_is_reported_not_swallowed(self):
        gh, error, slept = self.publish(
            [pbf.GitHubError("down", status_code=503)] * pbf.POST_ATTEMPTS
        )
        self.assertIn("down", error)
        self.assertEqual(gh.attempts, pbf.POST_ATTEMPTS)

    def test_exhausted_retry_budget_stops_the_waiting(self):
        # Several hundred PRs against a hard-down API must not sleep the job
        # into its timeout, which would lose the summary as well.
        gh, error, slept = self.publish(
            [pbf.GitHubError("down", status_code=503)] * pbf.POST_ATTEMPTS,
            budget=0.0,
        )
        self.assertIsNotNone(error)
        self.assertEqual(gh.attempts, 1)
        self.assertEqual(slept, [])


class DayCountTest(unittest.TestCase):
    """``--max-age-days`` is reachable from a manual dispatch."""

    def test_accepts_a_positive_day_count(self):
        self.assertEqual(pbf.day_count("3"), 3)

    def test_rejects_non_numeric_input(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            pbf.day_count("three")

    def test_rejects_values_that_would_stale_the_whole_queue(self):
        # Zero or negative puts the cutoff at or after now, making develop's
        # own tip the cutoff, so every PR not exactly on it reads as stale.
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                pbf.day_count(value)


def stale_verdict(missing_commits: int = 4, days_ago: float = 6.0) -> Verdict:
    return Verdict(
        fresh=False,
        missing_commits=missing_commits,
        oldest_missing=NOW - timedelta(days=days_ago),
    )


class ReportingTest(unittest.TestCase):
    def test_report_mode_never_fails(self):
        stale = stale_verdict()
        self.assertEqual(pbf.status_state(stale, enforcing=False), "success")
        self.assertEqual(pbf.status_state(stale, enforcing=True), "failure")

    def test_unevaluated_prs_never_fail(self):
        unknown = Verdict(fresh=True, error="boom")
        self.assertEqual(pbf.status_state(unknown, enforcing=True), "success")

    def test_report_mode_description_is_marked(self):
        stale = stale_verdict()
        self.assertTrue(
            pbf.describe(stale, "develop", 3, enforcing=False).startswith("WOULD FAIL:")
        )
        self.assertFalse(
            pbf.describe(stale, "develop", 3, enforcing=True).startswith("WOULD FAIL:")
        )

    def test_description_fits_githubs_limit(self):
        stale = stale_verdict(missing_commits=99999, days_ago=123.456)
        text = pbf.describe(
            stale, "a-very-long-base-branch-name" * 6, 3, enforcing=False
        )
        self.assertLessEqual(len(text), pbf.MAX_DESCRIPTION_LENGTH)


class StableDescriptionTest(unittest.TestCase):
    """The description must not move unless the PR does; see the write dedupe."""

    def test_stale_description_is_a_date_not_an_age(self):
        text = pbf.describe(stale_verdict(days_ago=6.0), "develop", 3, enforcing=True)
        self.assertIn("2026-07-14", text)
        self.assertNotIn("6.0", text)

    def test_drifting_commit_count_stays_out_of_the_description(self):
        # behind_by is counted from a cutoff that advances every run, so it
        # moves even when the PR does not.
        few = pbf.describe(stale_verdict(missing_commits=82), "develop", 3, True)
        many = pbf.describe(stale_verdict(missing_commits=97), "develop", 3, True)
        self.assertEqual(few, many)

    def test_undated_fallback_is_also_stable(self):
        undated = Verdict(fresh=False, missing_commits=5)
        self.assertEqual(
            pbf.describe(undated, "develop", 3, enforcing=True),
            "Base is more than 3 days behind develop. Merge or rebase develop",
        )

    def test_fresh_description_carries_no_varying_value(self):
        fresh = Verdict(fresh=True, missing_commits=0)
        self.assertEqual(
            pbf.describe(fresh, "develop", 3, enforcing=True),
            "Base is within 3 days of develop",
        )


if __name__ == "__main__":
    unittest.main()
