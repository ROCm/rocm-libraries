# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import email
import io
import os
import re
import sys
import threading
import time
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import pr_ci_freshness as pcf
from pr_ci_freshness import CIRun, Cutoff, PullRequest, Verdict


NOW = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
WORKFLOW = pcf.DEFAULT_CI_WORKFLOW
RUNS = f"actions/workflows/{WORKFLOW}/runs"
CONTEXT = "ci-freshness"
# The develop tip three and a half days ago, with a three day limit.
CUTOFF = Cutoff(sha="c0ff33", landed_at=NOW - timedelta(days=3.5))
PR = PullRequest(number=42, head_sha="deadbeef", draft=False)


def iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def runs(
    created_days_ago: float,
    attempt: int = 1,
    run_id: int = 7,
    bases: tuple[str, ...] = ("develop",),
) -> dict:
    """A workflow-runs response holding one run, newest first as GitHub sends.

    ``bases`` are the base branches of the PRs the run lists; empty is how a
    fork PR's run looks.
    """
    return {
        "workflow_runs": [
            {
                "id": run_id,
                "html_url": f"https://github.com/o/r/actions/runs/{run_id}",
                "run_attempt": attempt,
                "created_at": iso(created_days_ago),
                # A re-run moves this and not created_at, and it is the
                # wrong one to read: the merge was fixed at creation.
                "run_started_at": iso(0),
                "pull_requests": [
                    {"number": 42, "base": {"ref": base}} for base in bases
                ],
            }
        ]
    }


NO_RUNS = {"workflow_runs": []}


class FakeGitHub:
    """In-memory stand-in for the REST client, keyed by request path prefix."""

    def __init__(self, responses: dict, failures: tuple = (), budget: int = 10_000):
        self._responses = responses
        self._failures = failures
        self._budget = budget
        self.paths: list[str] = []
        self.params: list[dict] = []
        # Claimed but not yet released, so a test can assert who is holding
        # what after a call returns.
        self.outstanding = 0

    def has_budget(self, cost: int = 1) -> bool:
        return self._budget >= cost

    def claim_budget(self, cost: int) -> bool:
        if self._budget < cost:
            return False
        self._budget -= cost
        self.outstanding += cost
        return True

    def release_budget(self, cost: int) -> None:
        self._budget += cost
        self.outstanding -= cost

    def get(self, path: str, **params):
        self.paths.append(path)
        self.params.append(params)
        if path.startswith(self._failures):
            raise pcf.GitHubError(f"boom: {path}")
        for prefix, response in self._responses.items():
            if path.startswith(prefix):
                return response
        raise AssertionError(f"unexpected path {path}")


class FindCutoffTest(unittest.TestCase):
    def commits(self, body):
        return FakeGitHub({"commits": body})

    def test_cutoff_carries_when_it_landed(self):
        gh = self.commits(
            [{"sha": "c0ff33", "commit": {"committer": {"date": iso(3.5)}}}]
        )
        cutoff = pcf.find_cutoff_commit(gh, "develop", 3, NOW)
        self.assertEqual(cutoff, CUTOFF)
        self.assertEqual(gh.params[0]["until"], iso(3))

    def test_no_commit_old_enough_is_an_error(self):
        with self.assertRaises(pcf.GitHubError):
            pcf.find_cutoff_commit(self.commits([]), "develop", 3, NOW)

    def test_an_undatable_cutoff_is_an_error_not_a_crash(self):
        # main() fails the whole run open on GitHubError; anything else would
        # escape it and post nothing at all.
        for body in ([{"sha": "c0ff33"}], [{"sha": "c0ff33", "commit": None}]):
            with self.assertRaises(pcf.GitHubError):
                pcf.find_cutoff_commit(self.commits(body), "develop", 3, NOW)


class EvaluateTest(unittest.TestCase):
    def evaluate(self, body, cutoff=CUTOFF, pr=PR):
        gh = FakeGitHub({RUNS: body})
        return gh, pcf.evaluate(gh, pr, WORKFLOW, "develop", cutoff)

    def test_a_run_from_before_a_retarget_is_stale_however_recent(self):
        # A stacked PR retargeted onto develop when its parent merged: the
        # run is a day old, but it merged the parent branch, not develop.
        retargeted = PullRequest(
            42, "deadbeef", False, retargeted_at=NOW - timedelta(hours=6)
        )
        _, verdict = self.evaluate(runs(created_days_ago=1), pr=retargeted)
        self.assertFalse(verdict.fresh)
        self.assertTrue(verdict.retargeted)

    def test_a_run_made_after_a_retarget_is_judged_on_age_alone(self):
        retargeted = PullRequest(
            42, "deadbeef", False, retargeted_at=NOW - timedelta(days=2)
        )
        _, verdict = self.evaluate(runs(created_days_ago=1), pr=retargeted)
        self.assertTrue(verdict.fresh)
        self.assertFalse(verdict.retargeted)

    def test_head_only_mode_ignores_ci_for_prs_into_other_branches(self):
        # The CI-run trigger sees stacked PRs too; posting on them would
        # judge a feature branch's PR by develop's cutoff.
        head_only = PullRequest(None, "deadbeef", False)
        _, verdict = self.evaluate(
            runs(created_days_ago=5, bases=("users/x/parent",)), pr=head_only
        )
        self.assertIn("users/x/parent", verdict.ignored)

    def test_head_only_mode_still_judges_fork_runs(self):
        # A fork PR's run lists no PR, so there is no base to rule it out.
        head_only = PullRequest(None, "deadbeef", False)
        _, verdict = self.evaluate(runs(created_days_ago=5, bases=()), pr=head_only)
        self.assertIsNone(verdict.ignored)
        self.assertFalse(verdict.fresh)

    def test_head_only_mode_leaves_re_runs_to_the_sweep(self):
        # Nothing new was merged, and without the PR's retarget history this
        # mode could only overwrite a better verdict with a worse one.
        head_only = PullRequest(None, "deadbeef", False)
        _, verdict = self.evaluate(runs(created_days_ago=1, attempt=2), pr=head_only)
        self.assertIsNotNone(verdict.ignored)

    def test_a_run_into_develop_is_judged_in_head_only_mode(self):
        head_only = PullRequest(None, "deadbeef", False)
        _, verdict = self.evaluate(
            runs(created_days_ago=5, bases=("users/x/parent", "develop")), pr=head_only
        )
        self.assertIsNone(verdict.ignored)
        self.assertFalse(verdict.fresh)

    def test_run_created_after_the_cutoff_landed_is_fresh(self):
        gh, verdict = self.evaluate(runs(created_days_ago=1))
        self.assertTrue(verdict.fresh)
        self.assertEqual(verdict.ci_run.created_at, NOW - timedelta(days=1))
        # A PR must cost exactly one request to evaluate.
        self.assertEqual(len(gh.paths), 1)

    def test_run_created_before_the_cutoff_landed_is_stale(self):
        _, verdict = self.evaluate(runs(created_days_ago=5))
        self.assertFalse(verdict.fresh)
        self.assertIsNone(verdict.error)
        self.assertAlmostEqual(pcf.ci_age_days(verdict, NOW), 5.0, places=2)

    def test_a_rerun_of_an_old_run_stays_stale(self):
        # The whole point: a re-run reuses the original merge, so a fresh
        # run_started_at must not read as fresh CI.
        _, verdict = self.evaluate(runs(created_days_ago=5, attempt=3))
        self.assertFalse(verdict.fresh)
        self.assertEqual(verdict.ci_run.attempt, 3)

    def test_a_quiet_develop_never_makes_ci_stale(self):
        # Nothing has landed for ten days, so a four-day-old run merged the
        # same develop a run made today would, and is missing nothing.
        quiet = Cutoff(sha="c0ff33", landed_at=NOW - timedelta(days=10))
        _, verdict = self.evaluate(runs(created_days_ago=4), cutoff=quiet)
        self.assertTrue(verdict.fresh)

    def test_a_head_with_no_ci_run_is_fresh(self):
        # Whatever run it gets will merge the current develop.
        _, verdict = self.evaluate(NO_RUNS)
        self.assertTrue(verdict.fresh)
        self.assertIsNone(verdict.ci_run)
        self.assertIsNone(verdict.error)

    def test_lookup_asks_for_the_newest_pull_request_run_of_this_head(self):
        gh, _ = self.evaluate(runs(created_days_ago=1))
        self.assertEqual(gh.paths, [RUNS])
        self.assertEqual(
            gh.params[0],
            {"head_sha": PR.head_sha, "event": "pull_request", "per_page": 1},
        )

    def test_a_missing_attempt_defaults_to_the_first(self):
        body = runs(created_days_ago=1)
        del body["workflow_runs"][0]["run_attempt"]
        _, verdict = self.evaluate(body)
        self.assertEqual(verdict.ci_run.attempt, 1)

    def test_api_failure_fails_open(self):
        gh = FakeGitHub({}, failures=(RUNS,))
        verdict = pcf.evaluate(gh, PR, WORKFLOW, "develop", CUTOFF)
        self.assertTrue(verdict.fresh)
        self.assertIsNotNone(verdict.error)

    def test_a_malformed_body_fails_open_rather_than_raising(self):
        # An exception here would leave pool.map with nothing to return and
        # end the run before a single status was posted.
        bad_date = runs(created_days_ago=1)
        bad_date["workflow_runs"][0]["created_at"] = "yesterday-ish"
        for body in (
            {"unexpected": "shape"},
            None,
            [],
            {"workflow_runs": None},
            {"workflow_runs": [{"id": 1}]},
            bad_date,
        ):
            _, verdict = self.evaluate(body)
            self.assertTrue(verdict.fresh, body)
            self.assertIsNotNone(verdict.error, body)


class BudgetTest(unittest.TestCase):
    def test_pr_is_not_started_without_budget_to_publish_it(self):
        gh = FakeGitHub({RUNS: runs(created_days_ago=1)}, budget=1)
        assessment = pcf.assess(gh, PR, WORKFLOW, "develop", CUTOFF, CONTEXT)
        self.assertTrue(assessment.verdict.skipped)
        self.assertFalse(assessment.holds_write_claim)
        self.assertEqual(gh.paths, [])

    def test_a_stale_pr_costs_no_more_than_a_fresh_one(self):
        # Exactly enough to answer the question and publish it.
        gh = FakeGitHub(
            {RUNS: runs(created_days_ago=5), "commits/": {"statuses": []}},
            budget=pcf.REQUESTS_PER_PR,
        )
        assessment = pcf.assess(gh, PR, WORKFLOW, "develop", CUTOFF, CONTEXT)
        self.assertFalse(assessment.verdict.skipped)
        self.assertFalse(assessment.verdict.fresh)
        self.assertEqual(len(gh.paths), pcf.REQUESTS_PER_PR - pcf.REQUESTS_PER_WRITE)

    def test_the_write_claim_outlives_the_assessment(self):
        # The write happens in the serial phase, long after this returns, so
        # releasing its share here would leave every post in the sweep
        # outside the reserve the reads were careful to respect.
        gh = FakeGitHub(
            {RUNS: runs(created_days_ago=1), "commits/": {"statuses": []}},
            budget=100,
        )
        assessment = pcf.assess(gh, PR, WORKFLOW, "develop", CUTOFF, CONTEXT)
        self.assertTrue(assessment.holds_write_claim)
        self.assertEqual(gh.outstanding, pcf.REQUESTS_PER_WRITE)
        gh.release_budget(pcf.REQUESTS_PER_WRITE)
        self.assertEqual(gh.outstanding, 0)

    def test_the_read_back_is_inside_the_claim(self):
        # Budget for the run lookup alone must not admit the PR, since the
        # read-back would then be spent outside the reserve.
        gh = FakeGitHub(
            {RUNS: runs(created_days_ago=1)},
            budget=pcf.REQUESTS_PER_PR - 1,
        )
        assessment = pcf.assess(gh, PR, WORKFLOW, "develop", CUTOFF, CONTEXT)
        self.assertTrue(assessment.verdict.skipped)
        self.assertEqual(gh.paths, [])

    def test_reserve_governs_has_budget(self):
        gh = pcf.GitHub("o/r", "t", reserve=200)
        # Nothing observed yet, so the first request is always allowed.
        self.assertTrue(gh.has_budget(50))
        gh.rate_limit_remaining = 260
        self.assertTrue(gh.has_budget(50))
        self.assertFalse(gh.has_budget(61))

    def test_a_claim_is_held_against_later_callers(self):
        gh = pcf.GitHub("o/r", "t", reserve=100)
        gh.rate_limit_remaining = 106
        # Nothing has been spent and `remaining` has not moved, so a plain
        # check would let all three of these through.
        self.assertTrue(gh.claim_budget(3))
        self.assertTrue(gh.claim_budget(3))
        self.assertFalse(gh.claim_budget(3))
        gh.release_budget(3)
        self.assertTrue(gh.claim_budget(3))

    def test_concurrent_workers_cannot_all_clear_the_same_floor(self):
        gh = pcf.GitHub("o/r", "t", reserve=1000)
        gh.rate_limit_remaining = 1030
        # Headroom for exactly ten claims of three, contended by more workers
        # than that, all starting from the same observed `remaining`.
        workers = pcf.MAX_WORKERS
        start = threading.Barrier(workers)
        admitted = []

        def claim():
            start.wait()
            admitted.append(gh.claim_budget(pcf.REQUESTS_PER_PR))

        threads = [threading.Thread(target=claim) for _ in range(workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(admitted), 10)

    def test_transport_failures_become_github_errors(self):
        gh = pcf.GitHub("o/r", "t")

        def explode(request, timeout=None):
            raise TimeoutError("connection timed out")

        # A bare transport exception here would escape evaluate() and abort
        # the whole run, so it must arrive as GitHubError with the cause kept.
        with mock.patch.object(pcf.urllib.request, "urlopen", explode):
            with self.assertRaises(pcf.GitHubError) as caught:
                gh.get(RUNS)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(gh.request_count, 1)

    def test_http_errors_keep_the_status_code_and_retry_after(self):
        # publish_status decides whether to retry from these two, so losing
        # them in translation would turn a retryable 403 into a hard failure.
        gh = pcf.GitHub("o/r", "t")
        headers = email.message_from_string(
            "Retry-After: 30\nx-ratelimit-limit: 5000\nx-ratelimit-remaining: 12"
        )

        def explode(request, timeout=None):
            raise urllib.error.HTTPError(
                "https://example/x", 403, "Forbidden", headers, io.BytesIO(b"slow down")
            )

        with mock.patch.object(pcf.urllib.request, "urlopen", explode):
            with self.assertRaises(pcf.GitHubError) as caught:
                gh.post("statuses/abc", {})
        self.assertEqual(caught.exception.status_code, 403)
        self.assertEqual(caught.exception.retry_after, 30)
        self.assertTrue(caught.exception.retryable)
        self.assertIn("slow down", str(caught.exception))
        # An error response still reports the allowance, so it must be read.
        self.assertEqual(gh.rate_limit_remaining, 12)

    def test_graphql_answers_do_not_move_the_core_reserve(self):
        # GraphQL is a separate pool reported in the same headers. Reading it
        # as the core allowance would aim the reserve at the wrong budget.
        gh = pcf.GitHub("o/r", "t")
        gh.rate_limit_remaining = 3000
        gh._note_rate_limit(
            {
                "x-ratelimit-resource": "graphql",
                "x-ratelimit-limit": "5000",
                "x-ratelimit-remaining": "4999",
            }
        )
        self.assertEqual(gh.rate_limit_remaining, 3000)
        gh._note_rate_limit(
            {"x-ratelimit-resource": "core", "x-ratelimit-remaining": "2999"}
        )
        self.assertEqual(gh.rate_limit_remaining, 2999)

    def test_graphql_without_data_is_an_error(self):
        gh = pcf.GitHub("o/r", "t")
        for body in ({"errors": [{"message": "bad"}]}, {"data": None}, None):
            with mock.patch.object(gh, "_request", return_value=body):
                with self.assertRaises(pcf.GitHubError):
                    gh.graphql("query { x }", {})


class FakeGraphQL:
    """Answers retarget queries from ``createdAt`` values keyed by PR number."""

    def __init__(self, retargets: dict, missing: tuple = ()):
        self.repo = "o/r"
        self._retargets = retargets
        self._missing = missing
        self.queries: list[str] = []

    def graphql(self, query: str, variables: dict):
        self.queries.append(query)
        numbers = [int(n) for n in re.findall(r"pullRequest\(number: (\d+)\)", query)]
        repository = {}
        for n in numbers:
            if n in self._missing:
                repository[f"pr{n}"] = None
                continue
            nodes = (
                [{"createdAt": iso(self._retargets[n])}] if n in self._retargets else []
            )
            repository[f"pr{n}"] = {"timelineItems": {"nodes": nodes}}
        return {"repository": repository}


class RetargetTest(unittest.TestCase):
    def test_reports_only_prs_that_changed_base(self):
        gh = FakeGraphQL({7: 0.25})
        self.assertEqual(
            pcf.find_retargets(gh, [7, 8]), {7: NOW - timedelta(days=0.25)}
        )

    def test_a_pr_graphql_cannot_resolve_is_left_out_not_fatal(self):
        gh = FakeGraphQL({7: 1.0}, missing=(8,))
        self.assertEqual(list(pcf.find_retargets(gh, [7, 8])), [7])

    def test_a_sweep_is_batched_rather_than_one_query_per_pr(self):
        gh = FakeGraphQL({})
        pcf.find_retargets(gh, list(range(1, 2 * pcf.RETARGET_BATCH + 2)))
        self.assertEqual(len(gh.queries), 3)

    def test_a_malformed_answer_is_an_error(self):
        gh = FakeGraphQL({})
        gh.graphql = lambda query, variables: {"repository": "nope"}
        with self.assertRaises(pcf.GitHubError):
            pcf.find_retargets(gh, [7])


class PublishedStatusTest(unittest.TestCase):
    def test_reads_back_our_own_context(self):
        gh = FakeGitHub(
            {
                "commits/": {
                    "statuses": [
                        {"context": "other", "state": "failure", "description": "no"},
                        {
                            "context": CONTEXT,
                            "state": "success",
                            "description": "CI merged develop within the last 3 days",
                            "target_url": None,
                        },
                    ]
                }
            }
        )
        self.assertEqual(
            pcf.current_status(gh, "deadbeef", CONTEXT),
            ("success", "CI merged develop within the last 3 days", None),
        )

    def test_a_status_posted_from_a_publication_reads_back_as_unchanged(self):
        # The dedupe compares the two shapes directly, so they must agree.
        verdict = Verdict(fresh=True, ci_run=ci_run(days_ago=1))
        state, description, url = pcf.publication(verdict, "develop", 3, False, None)
        gh = FakeGitHub(
            {
                "commits/": {
                    "statuses": [
                        {
                            "context": CONTEXT,
                            "state": state,
                            "description": description,
                            "target_url": url,
                        }
                    ]
                }
            }
        )
        self.assertEqual(
            pcf.current_status(gh, "deadbeef", CONTEXT),
            pcf.publication(verdict, "develop", 3, False, None),
        )

    def test_a_new_run_on_the_same_head_is_republished(self):
        # A fresh PR's description never changes, so without the link in the
        # comparison its status would keep pointing at the run it replaced.
        old = Verdict(fresh=True, ci_run=ci_run(days_ago=2, run_id=7))
        new = Verdict(fresh=True, ci_run=ci_run(days_ago=0.1, run_id=8))
        self.assertNotEqual(
            pcf.publication(old, "develop", 3, False, None),
            pcf.publication(new, "develop", 3, False, None),
        )

    def test_a_head_without_a_run_is_stable_from_sweep_to_sweep(self):
        # Linking this job would change the status every sweep.
        no_run = Verdict(fresh=True)
        self.assertEqual(
            pcf.publication(no_run, "develop", 3, False, "https://x/runs/1"),
            pcf.publication(no_run, "develop", 3, False, "https://x/runs/2"),
        )

    def test_absent_context_reads_as_nothing_published(self):
        gh = FakeGitHub({"commits/": {"statuses": []}})
        self.assertIsNone(pcf.current_status(gh, "deadbeef", CONTEXT))

    def test_unreadable_status_does_not_suppress_the_post(self):
        # Returning None makes the caller post rather than assume it matches.
        gh = FakeGitHub({}, failures=("commits/",))
        self.assertIsNone(pcf.current_status(gh, "deadbeef", CONTEXT))

    def test_a_malformed_body_reads_as_nothing_published(self):
        for body in ({"unexpected": "shape"}, None, []):
            gh = FakeGitHub({"commits/": body})
            self.assertIsNone(pcf.current_status(gh, "deadbeef", CONTEXT))

    def test_skipped_pr_is_not_read_back(self):
        gh = FakeGitHub({}, budget=0)
        assessment = pcf.assess(gh, PR, WORKFLOW, "develop", CUTOFF, CONTEXT)
        self.assertTrue(assessment.verdict.skipped)
        self.assertIsNone(assessment.published)
        self.assertEqual(gh.paths, [])


class PacerTest(unittest.TestCase):
    def test_first_write_is_not_delayed(self):
        pacer = pcf.PostPacer(per_minute=60)
        start = time.monotonic()
        pacer.wait()
        self.assertLess(time.monotonic() - start, 0.1)

    def test_subsequent_writes_are_spaced(self):
        pacer = pcf.PostPacer(per_minute=600)  # 100ms apart
        pacer.wait()
        start = time.monotonic()
        pacer.wait()
        self.assertGreaterEqual(time.monotonic() - start, 0.05)

    def test_pacing_can_be_disabled(self):
        pacer = pcf.PostPacer(per_minute=0)
        pacer.wait()
        start = time.monotonic()
        pacer.wait()
        self.assertLess(time.monotonic() - start, 0.1)


class PagingFakeGitHub:
    """Serves ``pulls`` pages from a list, counting requests spent."""

    def __init__(self, numbers: list[int], budget: int = 10_000):
        self._numbers = numbers
        self._budget = budget
        self.pages_fetched = 0

    def has_budget(self, cost: int = 1) -> bool:
        return self._budget >= cost

    def get(self, path: str, **params):
        assert path == "pulls", path
        self.pages_fetched += 1
        self._budget -= 1
        per_page, page = params["per_page"], params["page"]
        window = self._numbers[(page - 1) * per_page : page * per_page]
        return [
            {"number": n, "head": {"sha": f"sha{n}"}, "draft": False} for n in window
        ]


class ListOpenPRsTest(unittest.TestCase):
    def test_a_short_queue_is_not_reported_as_truncated(self):
        gh = PagingFakeGitHub(list(range(150)))
        listing = pcf.list_open_prs(gh, "develop", include_drafts=False, limit=1000)
        self.assertEqual(len(listing.prs), 150)
        self.assertFalse(listing.truncated_at_limit)
        self.assertFalse(listing.truncated_by_reserve)

    def test_a_queue_longer_than_the_cap_is_reported(self):
        # The PRs past the cap get no status at all, so a sweep that stops on
        # a round number has to say whether it was the cap or the queue.
        gh = PagingFakeGitHub(list(range(250)))
        listing = pcf.list_open_prs(gh, "develop", include_drafts=False, limit=120)
        self.assertEqual(len(listing.prs), 120)
        self.assertTrue(listing.truncated_at_limit)

    def test_a_queue_exactly_at_the_cap_is_not_truncated(self):
        # The boundary the old loop could not tell apart: stopping because
        # the queue ran out looks the same as stopping because of the cap.
        gh = PagingFakeGitHub(list(range(120)))
        listing = pcf.list_open_prs(gh, "develop", include_drafts=False, limit=120)
        self.assertEqual(len(listing.prs), 120)
        self.assertFalse(listing.truncated_at_limit)

    def test_listing_stops_at_the_reserve(self):
        # Discovery is only a few requests, but the reserve is a promise
        # about the whole allowance, not just the per-PR part of it.
        gh = PagingFakeGitHub(list(range(500)), budget=2)
        listing = pcf.list_open_prs(gh, "develop", include_drafts=False, limit=1000)
        self.assertTrue(listing.truncated_by_reserve)
        self.assertEqual(gh.pages_fetched, 2)
        self.assertEqual(len(listing.prs), 200)

    def test_drafts_are_filtered_without_being_counted_against_the_cap(self):
        gh = PagingFakeGitHub(list(range(10)))
        gh.get = lambda path, **p: [
            {"number": n, "head": {"sha": f"sha{n}"}, "draft": n % 2 == 0}
            for n in ([] if p["page"] > 1 else range(10))
        ]
        listing = pcf.list_open_prs(gh, "develop", include_drafts=False, limit=1000)
        self.assertEqual([pr.number for pr in listing.prs], [1, 3, 5, 7, 9])


class RetryableErrorTest(unittest.TestCase):
    def test_transport_failures_are_retryable(self):
        # No response at all, so the next attempt is as good as this one.
        self.assertTrue(pcf.GitHubError("timed out").retryable)

    def test_rate_limits_and_server_errors_are_retryable(self):
        for code in (403, 429, 500, 502, 503):
            self.assertTrue(pcf.GitHubError("x", status_code=code).retryable)

    def test_client_errors_are_not_retryable(self):
        for code in (404, 422):
            self.assertFalse(pcf.GitHubError("x", status_code=code).retryable)


class RetryBudgetTest(unittest.TestCase):
    def test_spending_is_capped_at_what_is_left(self):
        budget = pcf.RetryBudget(10.0)
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

    def publish(self, errors, budget=pcf.POST_RETRY_BUDGET_SECONDS):
        gh = FakePoster(errors)
        slept: list[float] = []
        error = pcf.publish_status(
            gh,
            pcf.PostPacer(per_minute=0),
            pcf.RetryBudget(budget),
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
            [pcf.GitHubError("limited", status_code=403, retry_after=7)]
        )
        self.assertIsNone(error)
        self.assertEqual(gh.attempts, 2)
        self.assertEqual(slept, [7])

    def test_client_errors_are_not_retried(self):
        gh, error, slept = self.publish(
            [pcf.GitHubError("gone", status_code=404)] * pcf.POST_ATTEMPTS
        )
        self.assertIn("gone", error)
        self.assertEqual(gh.attempts, 1)
        self.assertEqual(slept, [])

    def test_persistent_failure_is_reported_not_swallowed(self):
        gh, error, slept = self.publish(
            [pcf.GitHubError("down", status_code=503)] * pcf.POST_ATTEMPTS
        )
        self.assertIn("down", error)
        self.assertEqual(gh.attempts, pcf.POST_ATTEMPTS)

    def test_exhausted_retry_budget_stops_the_waiting(self):
        # Several hundred PRs against a hard-down API must not sleep the job
        # into its timeout, which would lose the summary as well.
        gh, error, slept = self.publish(
            [pcf.GitHubError("down", status_code=503)] * pcf.POST_ATTEMPTS,
            budget=0.0,
        )
        self.assertIsNotNone(error)
        self.assertEqual(gh.attempts, 1)
        self.assertEqual(slept, [])


class DayCountTest(unittest.TestCase):
    """``--max-age-days`` is reachable from a manual dispatch."""

    def test_accepts_a_positive_day_count(self):
        self.assertEqual(pcf.day_count("3"), 3)

    def test_rejects_non_numeric_input(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            pcf.day_count("three")

    def test_rejects_values_that_would_stale_the_whole_queue(self):
        # Zero or negative puts the cutoff at or after now, making develop's
        # own tip the cutoff, so every PR not exactly on it reads as stale.
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                pcf.day_count(value)


def ci_run(days_ago: float = 6.0, attempt: int = 1, run_id: int = 7) -> CIRun:
    return CIRun(
        id=run_id,
        url=f"https://github.com/o/r/actions/runs/{run_id}",
        attempt=attempt,
        created_at=NOW - timedelta(days=days_ago),
    )


def stale_verdict(
    days_ago: float = 6.0, attempt: int = 1, retargeted: bool = False
) -> Verdict:
    return Verdict(fresh=False, ci_run=ci_run(days_ago, attempt), retargeted=retargeted)


class FetchPRTest(unittest.TestCase):
    def pull(self, base: str):
        return FakeGitHub(
            {
                "pulls/": {
                    "number": 9,
                    "head": {"sha": "abc"},
                    "draft": False,
                    "base": {"ref": base},
                }
            }
        )

    def test_a_pr_into_develop_is_fetched(self):
        pr = pcf.fetch_pr(self.pull("develop"), 9, "develop")
        self.assertEqual((pr.number, pr.head_sha), (9, "abc"))

    def test_a_pr_into_another_branch_is_not_ours(self):
        self.assertIsNone(pcf.fetch_pr(self.pull("users/x/parent"), 9, "develop"))


class ReportingTest(unittest.TestCase):
    def test_report_mode_never_fails(self):
        stale = stale_verdict()
        self.assertEqual(pcf.status_state(stale, enforcing=False), "success")
        self.assertEqual(pcf.status_state(stale, enforcing=True), "failure")

    def test_unevaluated_prs_never_fail(self):
        unknown = Verdict(fresh=True, error="boom")
        self.assertEqual(pcf.status_state(unknown, enforcing=True), "success")

    def test_report_mode_description_is_marked(self):
        stale = stale_verdict()
        self.assertTrue(
            pcf.describe(stale, "develop", 3, enforcing=False).startswith("WOULD FAIL:")
        )
        self.assertFalse(
            pcf.describe(stale, "develop", 3, enforcing=True).startswith("WOULD FAIL:")
        )

    def test_stale_description_says_a_rerun_will_not_help(self):
        text = pcf.describe(stale_verdict(), "develop", 3, enforcing=True)
        self.assertIn("re-running jobs reuses that base", text)

    def test_description_fits_githubs_limit(self):
        for retargeted in (False, True):
            stale = stale_verdict(retargeted=retargeted)
            for branch in ("develop", "a-very-long-base-branch-name" * 6):
                text = pcf.describe(stale, branch, 3, enforcing=False)
                self.assertLessEqual(len(text), pcf.MAX_DESCRIPTION_LENGTH)
            # The real branch must not need truncating, or the advice gets cut.
            self.assertTrue(
                pcf.describe(stale, "develop", 3, enforcing=False).endswith(
                    "reuses that base"
                )
            )

    def test_a_retargeted_pr_is_told_why_rather_than_given_a_date(self):
        text = pcf.describe(stale_verdict(retargeted=True), "develop", 3, True)
        self.assertIn("retargeted to develop", text)
        self.assertNotIn("2026-", text)

    def test_status_links_to_the_ci_run_it_judged(self):
        this_run = "https://github.com/o/r/actions/runs/99"
        self.assertEqual(pcf.status_target_url(stale_verdict(), this_run), ci_run().url)
        # An error is worth a link to this job's log; "no run yet" is not.
        self.assertEqual(
            pcf.status_target_url(Verdict(fresh=True, error="boom"), this_run), this_run
        )
        self.assertIsNone(pcf.status_target_url(Verdict(fresh=True), this_run))

    def test_a_head_only_pr_is_labelled_by_its_sha(self):
        self.assertEqual(PR.label, "#42")
        self.assertEqual(
            PullRequest(None, "0123456789abcdef", False).label, "0123456789ab"
        )


class StableDescriptionTest(unittest.TestCase):
    """The description must not move unless the CI run does; see the write dedupe."""

    def test_stale_description_is_a_date_not_an_age(self):
        text = pcf.describe(stale_verdict(days_ago=6.0), "develop", 3, enforcing=True)
        self.assertIn("2026-07-14", text)
        self.assertNotIn("6.0", text)

    def test_a_rerun_does_not_change_the_description(self):
        # Same run, same merge, so the same words; anything else would be a
        # write that tells the PR nothing.
        first = pcf.describe(stale_verdict(attempt=1), "develop", 3, True)
        again = pcf.describe(stale_verdict(attempt=4), "develop", 3, True)
        self.assertEqual(first, again)

    def test_fresh_description_carries_no_varying_value(self):
        for days_ago in (0.1, 2.9):
            fresh = Verdict(fresh=True, ci_run=ci_run(days_ago))
            self.assertEqual(
                pcf.describe(fresh, "develop", 3, enforcing=True),
                "CI merged develop within the last 3 days",
            )

    def test_no_run_description_is_constant(self):
        self.assertEqual(
            pcf.describe(Verdict(fresh=True), "develop", 3, enforcing=True),
            "No CI run on this head yet; the next one merges current develop",
        )


class WorkflowTest(unittest.TestCase):
    """The workflow must run this script from the default branch tip."""

    WORKFLOWS = Path(__file__).parents[2] / "workflows"
    WORKFLOW = WORKFLOWS / "pr-ci-freshness.yml"

    def setting(self, text: str, key: str) -> str:
        """The value of the one ``key:`` line in ``text``."""
        values = [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.strip().startswith(f"{key}:")
        ]
        self.assertEqual(len(values), 1, f"expected one {key}: line, got {values}")
        return values[0]

    def test_checkout_ref_is_the_default_branch(self):
        # A PR's base SHA can predate this script, and the PR head is
        # untrusted code in a job holding `statuses: write`.
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(
            self.setting(text, "ref"), "${{ github.event.repository.default_branch }}"
        )

    def test_trigger_and_lookup_name_the_same_ci_workflow(self):
        # The trigger takes the workflow's `name:` and the run lookup its file
        # name. If they drift apart the status describes a workflow that is
        # not the one that fired it, or never fires at all.
        text = self.WORKFLOW.read_text(encoding="utf-8")
        ci_file = self.setting(text, "CI_WORKFLOW_FILE")
        self.assertEqual(ci_file, pcf.DEFAULT_CI_WORKFLOW)
        ci_text = (self.WORKFLOWS / ci_file).read_text(encoding="utf-8")
        ci_name = next(
            line.split(":", 1)[1].strip()
            for line in ci_text.splitlines()
            if line.startswith("name:")
        )
        self.assertEqual(self.setting(text, "workflows"), f"['{ci_name}']")


if __name__ == "__main__":
    unittest.main()
