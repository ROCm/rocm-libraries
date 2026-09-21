# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Report how far behind ``develop`` an open pull request's base is.

A PR that was branched a week ago is tested against a week-old ``develop``, so a
green CI run says very little about whether the merge result actually works.
GitHub's only built-in control here is "require branches to be up to date",
which demands zero commits behind and would put every PR in this repo on an
update-rerun-CI treadmill. This script implements the softer rule we actually
want: the base may drift, but not for more than a few days.

Staleness definition
--------------------
Let ``cutoff`` be the commit that was ``develop``'s tip ``--max-age-days`` ago.
A PR is *stale* when its head does not contain ``cutoff``, i.e. when it is
missing at least one commit that has been on ``develop`` for that long. Phrasing
it against a time-based cutoff rather than "age of the merge base" means a quiet
``develop`` never makes anyone stale: if nothing landed, there is nothing to
miss.

Reporting
---------
The verdict is published as a commit status (not a check run) on the PR head,
because a scheduled run has no PR event to attach a check run to and staleness
is a function of wall-clock time: a PR that passes today must be able to turn
red on its own three days from now with no push in between. A fixed context name
lets the same status be required by a ruleset later.

In ``--mode report`` every status is posted as ``success`` with the verdict in
the description, so the rule can be observed on live traffic before it gates
anything. ``--mode enforce`` posts ``failure`` for stale PRs.

Evaluation is fail-open: if the API cannot be reached or a PR cannot be
evaluated, we post ``success`` and say so. A required status that never reports
blocks its PR forever, so a broken freshness job must never wedge the repo.

The one case that is not fail-open is the rate-limit reserve. The token's
hourly allowance is shared with every other workflow in the repository, so a
sweep stops rather than exhaust it, and the PRs it did not reach get no status
from that run. Their PR-event status (if any) stands. Before enforcement is
enabled, confirm from the job summary that a full sweep finishes inside the
allowance; a sweep that routinely hits the reserve would leave PRs with no
status at all, which a required check would turn into a permanent block.

Usage:
  python pr_base_freshness.py --pr 1234
  python pr_base_freshness.py --all-open
"""

import argparse
import concurrent.futures
import os
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import requests

API_ROOT = "https://api.github.com"

# GitHub truncates commit status descriptions past this length.
MAX_DESCRIPTION_LENGTH = 140

# Concurrency for the per-PR ancestry checks. Kept modest on purpose: the
# GITHUB_TOKEN rate limit is shared across everything running in this repo.
MAX_WORKERS = 8

# Requests a PR costs before the optional dating call: one ancestry compare
# plus the status post. A PR is not started unless both can still be paid for,
# so we never evaluate a PR and then fail to publish the answer.
REQUESTS_PER_PR = 2

# The dating call that turns "over the limit" into "6.0 days" is cosmetic, so
# it is only spent when there is this much slack left. Under pressure the
# human-readable day count is the first thing dropped, never the verdict.
COSMETIC_SLACK = 50


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an unusable response."""


class GitHub:
    """Minimal REST client scoped to one repository, with a rate-limit guard.

    The token's hourly allowance is shared with every other workflow running in
    this repository, so the client refuses to spend past ``reserve`` remaining.
    The ceiling is read from the live ``x-ratelimit-remaining`` header rather
    than assumed, because the allowance differs between a plain ``GITHUB_TOKEN``
    and an Enterprise Cloud one.
    """

    def __init__(self, repo: str, token: str, reserve: int = 0) -> None:
        self.repo = repo
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.reserve = reserve
        self.request_count = 0
        self.rate_limit_limit: Optional[int] = None
        self.rate_limit_remaining: Optional[int] = None
        # The per-PR work runs on a thread pool, so the counters it reads to
        # make budget decisions need to be consistent.
        self._lock = threading.Lock()

    def has_budget(self, cost: int = 1) -> bool:
        """Whether ``cost`` more requests can be spent without hitting reserve."""
        with self._lock:
            if self.rate_limit_remaining is None:
                # Nothing observed yet; the first request establishes the floor.
                return True
            return self.rate_limit_remaining - cost >= self.reserve

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with self._lock:
            self.request_count += 1
        try:
            response = self.session.request(
                method, f"{API_ROOT}/repos/{self.repo}/{path}", timeout=60, **kwargs
            )
        except requests.RequestException as e:
            # A timeout or connection reset has to look like every other API
            # failure. Callers only handle GitHubError, and an exception that
            # escapes them aborts the whole run before any status is posted,
            # which is exactly what the fail-open contract forbids.
            raise GitHubError(f"{method} {path} -> {e}") from e
        with self._lock:
            self.rate_limit_limit = _header_int(response, "x-ratelimit-limit")
            self.rate_limit_remaining = _header_int(response, "x-ratelimit-remaining")
        if not response.ok:
            raise GitHubError(
                f"{method} {path} -> {response.status_code} {response.text[:200]}"
            )
        try:
            return response.json() if response.content else None
        except ValueError as e:
            raise GitHubError(f"{method} {path} -> malformed response body") from e

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, json=payload)


def _header_int(response: "requests.Response", name: str) -> Optional[int]:
    try:
        return int(response.headers[name])
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class PullRequest:
    number: int
    head_sha: str
    draft: bool


@dataclass
class Verdict:
    """The freshness result for one PR, before it is translated to a status."""

    fresh: bool
    # Commits on develop older than the cutoff that this PR is missing. None
    # when the PR could not be evaluated.
    missing_commits: Optional[int] = None
    # Age in days of the oldest develop commit this PR is missing. Only
    # computed for stale PRs, where it is the number a human wants to see.
    stale_days: Optional[float] = None
    error: Optional[str] = None
    # Set when the rate-limit reserve stopped us before this PR was looked at.
    # A skipped PR gets no status at all, since posting one would itself cost
    # the request we just declined to spend.
    skipped: bool = False


def parse_timestamp(value: str) -> datetime:
    """Parses a GitHub ISO-8601 timestamp into an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def find_cutoff_commit(
    gh: GitHub, base_branch: str, max_age_days: int, now: datetime
) -> str:
    """Returns the SHA that was ``base_branch``'s tip ``max_age_days`` ago."""
    until = (now - timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = gh.get("commits", sha=base_branch, until=until, per_page=1)
    if not commits:
        raise GitHubError(f"no commit on {base_branch} older than {until}")
    return commits[0]["sha"]


def evaluate(
    gh: GitHub, pr: PullRequest, base_branch: str, cutoff_sha: str, now: datetime
) -> Verdict:
    """Decides whether ``pr`` contains the cutoff commit."""
    # Only start a PR we can also publish the answer for.
    if not gh.has_budget(REQUESTS_PER_PR):
        return Verdict(fresh=True, skipped=True)

    try:
        comparison = gh.get(f"compare/{cutoff_sha}...{pr.head_sha}", per_page=1)
    except GitHubError as e:
        return Verdict(fresh=True, error=str(e))

    # "identical" or "ahead" means the cutoff is an ancestor of the PR head.
    if comparison["status"] in ("identical", "ahead"):
        return Verdict(fresh=True, missing_commits=0)

    missing = comparison["behind_by"]
    stale_days = None
    if gh.has_budget(COSMETIC_SLACK):
        try:
            # Only stale PRs pay for this second call. The merge base of the
            # cutoff and the head is also the merge base of develop and the
            # head, since a stale PR by definition diverged before the cutoff.
            merge_base = comparison["merge_base_commit"]["sha"]
            since_merge_base = gh.get(
                f"compare/{merge_base}...{base_branch}", per_page=1
            )
            oldest = since_merge_base["commits"][0]["commit"]["committer"]["date"]
            stale_days = (now - parse_timestamp(oldest)).total_seconds() / 86400
        except (GitHubError, KeyError, IndexError):
            # The headline verdict stands; we just cannot put a number on it.
            pass

    return Verdict(fresh=False, missing_commits=missing, stale_days=stale_days)


def describe(
    verdict: Verdict, base_branch: str, max_age_days: int, enforcing: bool
) -> str:
    """Builds the one-line commit status description."""
    if verdict.error:
        text = f"Could not evaluate base freshness ({verdict.error})"
    elif verdict.fresh:
        text = f"Base is within {max_age_days} days of {base_branch}"
    else:
        age = (
            f"{verdict.stale_days:.1f} days" if verdict.stale_days else "over the limit"
        )
        text = (
            f"Base is {age} behind {base_branch} "
            f"({verdict.missing_commits} commits older than {max_age_days} days). "
            f"Merge or rebase {base_branch}"
        )
        if not enforcing:
            text = f"WOULD FAIL: {text}"
    return text[:MAX_DESCRIPTION_LENGTH]


def status_state(verdict: Verdict, enforcing: bool) -> str:
    return (
        "failure"
        if enforcing and not verdict.fresh and not verdict.error
        else "success"
    )


def list_open_prs(
    gh: GitHub, base_branch: str, include_drafts: bool, limit: int
) -> list[PullRequest]:
    """Lists open PRs targeting ``base_branch``, least recent activity first.

    Ordering matters when the reserve cuts a sweep short. The PR-event trigger
    already posts a status on every open and push, so a PR touched minutes ago
    has an accurate one. The PRs whose statuses are most likely to be wrong are
    the ones nothing has touched in days, so they go first and the freshly
    evaluated ones are what gets dropped.
    """
    prs: list[PullRequest] = []
    page = 1
    while len(prs) < limit:
        batch = gh.get(
            "pulls",
            state="open",
            base=base_branch,
            sort="updated",
            direction="asc",
            per_page=100,
            page=page,
        )
        if not batch:
            break
        for item in batch:
            if item["draft"] and not include_drafts:
                continue
            prs.append(PullRequest(item["number"], item["head"]["sha"], item["draft"]))
        page += 1
    return prs[:limit]


def fetch_pr(gh: GitHub, number: int) -> PullRequest:
    item = gh.get(f"pulls/{number}")
    return PullRequest(item["number"], item["head"]["sha"], item["draft"])


def write_summary(lines: Iterable[str]) -> None:
    """Appends markdown to the GitHub Actions job summary, if there is one."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--base-branch", default="develop")
    parser.add_argument("--max-age-days", type=int, default=3)
    parser.add_argument(
        "--context", default="base-freshness", help="Commit status context name"
    )
    parser.add_argument(
        "--mode",
        choices=("report", "enforce"),
        default="report",
        help="report: always post success, with the verdict in the description",
    )
    parser.add_argument("--pr", type=int, help="Evaluate a single PR")
    parser.add_argument(
        "--all-open", action="store_true", help="Evaluate every open PR"
    )
    parser.add_argument(
        "--max-prs", type=int, default=400, help="Safety cap on PRs evaluated per run"
    )
    parser.add_argument(
        "--rate-limit-reserve",
        type=int,
        default=200,
        help="Stop the sweep rather than drive the shared hourly limit below this",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print verdicts, post nothing"
    )
    args = parser.parse_args()

    if bool(args.pr) == bool(args.all_open):
        parser.error("pass exactly one of --pr or --all-open")
    token = os.environ.get("GH_TOKEN")
    if not token:
        parser.error("GH_TOKEN is not set")

    enforcing = args.mode == "enforce"
    now = datetime.now(timezone.utc)
    # A single PR triggered by its own event is worth spending reserve on; only
    # the sweep can plausibly exhaust the shared allowance.
    reserve = 0 if args.pr else args.rate_limit_reserve
    gh = GitHub(args.repo, token, reserve=reserve)

    try:
        cutoff_sha = find_cutoff_commit(gh, args.base_branch, args.max_age_days, now)
    except GitHubError as e:
        # Fail open, loudly. Posting nothing is worse than posting success:
        # an unreported required status blocks its PR with no way out.
        print(f"::warning::could not determine freshness cutoff, skipping run: {e}")
        return 0

    try:
        if args.pr:
            # Single-PR mode evaluates drafts too, so the status is already
            # present if the PR is marked ready and merged before the next
            # scheduled run.
            prs = [fetch_pr(gh, args.pr)]
        else:
            prs = list_open_prs(
                gh, args.base_branch, include_drafts=False, limit=args.max_prs
            )
    except GitHubError as e:
        print(f"::warning::could not list pull requests, skipping run: {e}")
        return 0

    target_url = None
    server, run_id = os.environ.get("GITHUB_SERVER_URL"), os.environ.get(
        "GITHUB_RUN_ID"
    )
    if server and run_id:
        target_url = f"{server}/{args.repo}/actions/runs/{run_id}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        verdicts = list(
            pool.map(
                lambda pr: evaluate(gh, pr, args.base_branch, cutoff_sha, now), prs
            )
        )

    stale = []
    skipped = []
    for pr, verdict in zip(prs, verdicts):
        if verdict.skipped:
            skipped.append(pr)
            continue
        state = status_state(verdict, enforcing)
        description = describe(verdict, args.base_branch, args.max_age_days, enforcing)
        print(f"PR #{pr.number}: {state} - {description}")
        if not verdict.fresh and not verdict.error:
            stale.append((pr, verdict))
        if args.dry_run:
            continue
        payload = {"state": state, "context": args.context, "description": description}
        if target_url:
            payload["target_url"] = target_url
        try:
            gh.post(f"statuses/{pr.head_sha}", payload)
        except GitHubError as e:
            print(f"::warning::could not post status for #{pr.number}: {e}")

    if skipped:
        # Loud on purpose. Before enforcement is switched on, a sweep that
        # cannot finish within the shared allowance is a blocker, since a PR
        # with no status at all would be stuck behind a required check.
        print(
            f"::warning::rate-limit reserve reached; {len(skipped)} of {len(prs)} "
            f"PRs were not evaluated and have no status from this run"
        )

    evaluated = len(prs) - len(skipped)
    summary = [
        f"## Base freshness ({args.mode} mode)",
        "",
        f"Cutoff commit (`{args.base_branch}` tip {args.max_age_days} days ago): "
        f"`{cutoff_sha[:12]}`",
        "",
        f"{len(stale)} of {evaluated} evaluated PRs are stale."
        + (f" {len(skipped)} skipped for rate-limit reserve." if skipped else ""),
        "",
    ]
    if stale:
        summary += ["| PR | Days behind | Missing commits |", "| --- | --- | --- |"]
        for pr, verdict in sorted(stale, key=lambda s: -(s[1].stale_days or 0)):
            days = f"{verdict.stale_days:.1f}" if verdict.stale_days else "?"
            summary.append(f"| #{pr.number} | {days} | {verdict.missing_commits} |")
        summary.append("")
    summary.append(
        f"_{gh.request_count} API requests; rate limit "
        f"{gh.rate_limit_remaining}/{gh.rate_limit_limit} remaining "
        f"(reserve {gh.reserve})_"
    )
    write_summary(summary)

    # The job itself always succeeds. The verdict travels as a commit status,
    # which is what a ruleset can require; a red job here would only be noise.
    return 0


if __name__ == "__main__":
    sys.exit(main())
