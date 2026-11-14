# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Report how old the ``develop`` behind each open pull request's CI result is.

PR CI does not build the branch as pushed. On a ``pull_request`` event GitHub
hands the workflow a test merge commit, the PR head merged into ``develop`` as
it stood at that moment, and that merge is what TheRock Multi-Arch CI builds
and tests. How long ago the branch itself last merged ``develop`` says nothing
about what CI saw; how long ago CI made its merge says everything.

That merge is fixed when the run is created. Re-running failed jobs, or the
whole run, reuses it, so a re-run a week later still tests the week-old
``develop``. Only a new run (a push, a reopen, a label change) merges the
current tip. GitHub's built-in "require branches to be up to date" rule would
force a new run after every commit to ``develop`` and put every PR in this repo
on an update-rerun-CI treadmill. This script implements the softer rule we
actually want: the ``develop`` that CI tested against may lag, but not by more
than a few days.

Staleness definition
--------------------
Let ``cutoff`` be the commit that was ``develop``'s tip ``--max-age-days`` ago.
A PR's CI is *stale* when the newest CI run for its head merged a ``develop``
that does not contain ``cutoff``, i.e. when CI missed at least one commit that
has been on ``develop`` for that long. Phrasing it against a time-based cutoff
rather than "age of the CI run" means a quiet ``develop`` never makes anyone
stale: if nothing landed, there is nothing to miss.

Which ``develop`` a run merged is inferred rather than read back, because
GitHub keeps no record of it: the run carries the PR head, not the merge
commit, and its ``pull_requests[].base`` is the PR's base as of the query, not
as of the run. What does survive is the run's ``created_at``, which, unlike
``run_started_at``, does not move on a re-run. The run merged ``develop`` as it
stood then, so it contains ``cutoff`` exactly when it was created after
``cutoff`` landed. That is good to within the few minutes GitHub can lag in
refreshing a PR's merge commit, which is immaterial against a threshold
measured in days.

A run is also stale when the PR was retargeted to ``develop`` after the run
was created. A retarget starts no CI run, so that run merged some other
branch; this is the normal path for a stacked PR whose parent has merged.
The run cannot reveal it (its ``pull_requests[].base`` already reads
``develop``), so the PR timeline is asked instead.

A head with no CI run yet is fresh, since whatever run it gets will merge the
current tip.

The CI-run trigger fires for stacked PRs into feature branches as well. Those
get no status: they are judged against a different branch, if at all, and the
sweep never revisits them.

Only TheRock Multi-Arch CI is judged. Component CI and the Jenkins Math CI
build a merge too, each with its own base, and are not covered here.

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

Writes, not reads, are the scarce resource. The hourly request allowance is
generous, but commit statuses count against GitHub's much tighter
content-generation limit, so a sweep skips any post that would republish an
identical status and paces the rest. In steady state most of the queue is
unchanged and costs no write at all.

Fail-open covers a bad answer, not a missing one. Two things can leave a PR
with no status at all from a run, and under enforcement a required context
that is simply absent is a permanent block, so both are reported loudly and
neither is allowed to pass unnoticed:

* The rate-limit reserve. The allowance is shared with every other workflow in
  the repository, so a sweep stops rather than exhaust it, and the PRs it did
  not reach are counted in a warning.
* A status post that keeps failing. Writes are retried within a per-run time
  allowance, and any that still do not land are named in the job summary and
  raised as an annotation.

Before enforcement is enabled, confirm from the job summary that a full sweep
finishes inside the allowance and publishes every status.

Usage:
  python pr_ci_freshness.py --pr 1234
  python pr_ci_freshness.py --head-sha 0123abcd...
  python pr_ci_freshness.py --all-open
"""

import argparse
import concurrent.futures
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

API_ROOT = "https://api.github.com"

# Deliberately no third-party HTTP client. This runs in a base-repo
# `workflow_run` job holding `statuses: write`, so anything it installs
# is code running at that privilege, and an unpinned install is a standing
# invitation. The workload is a handful of JSON GETs and POSTs, which the
# standard library does perfectly well. The only cost is the lost connection
# pooling, measured at about 1.3x per request against the real API, which over
# a full sweep is seconds against a 20 minute timeout.
TIMEOUT_SECONDS = 60
USER_AGENT = "rocm-libraries-pr-ci-freshness"

# The workflow whose runs are judged, by file name as the Actions API takes it.
DEFAULT_CI_WORKFLOW = "therock-multi-arch-ci.yml"

# GitHub truncates commit status descriptions past this length.
MAX_DESCRIPTION_LENGTH = 140

# Concurrency for the per-PR reads. The hourly allowance is generous enough
# that reads are not the constraint; this is sized to keep a full sweep to a
# couple of minutes without leaning on the secondary per-minute limits.
MAX_WORKERS = 16

# Writes are the scarce resource, not reads. Commit statuses count against
# GitHub's content-generation secondary limit (roughly 80 per minute, 500 per
# hour), which is far tighter than the hourly request allowance, and it is
# shared with the statuses the CI-run trigger is posting at the same time.
# Hence both the pacing below and skipping writes that would change nothing.
POSTS_PER_MINUTE = 70

# Requests a PR costs: the lookup of its newest CI run, the read of its
# current status, and the status post. A PR is not started unless all three
# can be paid for, so we never evaluate a PR and then fail to publish the
# answer.
REQUESTS_PER_PR = 3

# Of those, the one spent in the serial write phase, long after the reads are
# done. Its share of the claim has to outlive the assessment that took it;
# releasing the whole claim when the reads finish would leave every write in
# the sweep outside the reserve those reads were careful to respect.
REQUESTS_PER_WRITE = 1

# A write that fails is the one outcome this job cannot shrug off: in enforce
# mode a required context that is simply absent blocks its PR with no way out,
# and the job would be green while it happened. Status posts are therefore
# retried, which matters most for the 403 GitHub returns when the
# content-generation limit is hit, since that is the limit the pacer aims at.
POST_ATTEMPTS = 3
POST_RETRY_BACKOFF_SECONDS = 5.0

# Retries are for a blip, not an outage. Several hundred PRs against an API
# that is hard down would otherwise leave the job asleep for longer than its
# timeout, losing the run summary too, so the waiting is capped for the run as
# a whole rather than per post.
POST_RETRY_BUDGET_SECONDS = 120.0


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an unusable response."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """Whether repeating the request could plausibly succeed."""
        if self.status_code is None:
            # A transport failure or an unparseable body; no verdict from the
            # server at all, so the next attempt is as good as this one.
            return True
        # 403 is how the secondary content-generation limit reports itself,
        # and 429 is the primary one. Both clear on their own.
        return self.status_code in (403, 429) or self.status_code >= 500


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
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": USER_AGENT,
        }
        self.reserve = reserve
        self.request_count = 0
        self.rate_limit_limit: Optional[int] = None
        self.rate_limit_remaining: Optional[int] = None
        # Requests that have been admitted but not yet sent. Without this a
        # check-then-spend is not enough: every worker in the pool can read
        # the same ``remaining`` and then all spend against it, so the floor
        # they were each respecting gets crossed by the group.
        self._in_flight = 0
        # The per-PR work runs on a thread pool, so the counters it reads to
        # make budget decisions need to be consistent.
        self._lock = threading.Lock()

    def _fits(self, cost: int) -> bool:
        """Whether ``cost`` fits above reserve. Caller must hold ``_lock``."""
        if self.rate_limit_remaining is None:
            # Nothing observed yet; the first request establishes the floor.
            return True
        return self.rate_limit_remaining - self._in_flight - cost >= self.reserve

    def has_budget(self, cost: int = 1) -> bool:
        """Whether ``cost`` more requests fit, without claiming them.

        For work outside the per-PR claims, such as listing pages, where
        losing a race costs nothing worse than a shorter list. Anything that
        must not be started unless it can also be finished goes through
        ``claim_budget``.
        """
        with self._lock:
            return self._fits(cost)

    def claim_budget(self, cost: int) -> bool:
        """Reserves ``cost`` requests, so concurrent callers queue up.

        The claim is released once the work it covers is done, by which time
        the requests it paid for have refreshed ``rate_limit_remaining``
        themselves. Counting both for that window overstates the spend, which
        is the safe direction to be wrong in.
        """
        with self._lock:
            if not self._fits(cost):
                return False
            self._in_flight += cost
            return True

    def release_budget(self, cost: int) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - cost)

    def _note_rate_limit(self, headers: Any) -> None:
        """Records the allowance the server just reported, error or not.

        Only the core REST allowance is tracked. GraphQL is metered from a
        separate pool and reports it in the same headers, so letting it in
        would point the reserve at the wrong budget.
        """
        resource = headers.get("x-ratelimit-resource") if headers else None
        if resource not in (None, "core"):
            return
        with self._lock:
            self.rate_limit_limit = _header_int(headers, "x-ratelimit-limit")
            self.rate_limit_remaining = _header_int(headers, "x-ratelimit-remaining")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        payload: Optional[dict] = None,
        url: Optional[str] = None,
    ) -> Any:
        with self._lock:
            self.request_count += 1
        url = url or f"{API_ROOT}/repos/{self.repo}/{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=body, method=method, headers=dict(self._headers)
        )
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                self._note_rate_limit(response.headers)
                raw = response.read()
        except urllib.error.HTTPError as e:
            # An error response still carries the rate-limit headers, and for
            # a secondary limit it is the only place Retry-After shows up.
            self._note_rate_limit(e.headers)
            detail = e.read()[:200].decode("utf-8", "replace")
            raise GitHubError(
                f"{method} {path} -> {e.code} {detail}",
                status_code=e.code,
                retry_after=_header_int(e.headers, "retry-after"),
            ) from e
        except (OSError, http.client.HTTPException) as e:
            # A timeout or connection reset has to look like every other API
            # failure. Callers only handle GitHubError, and an exception that
            # escapes them aborts the whole run before any status is posted,
            # which is exactly what the fail-open contract forbids.
            raise GitHubError(f"{method} {path} -> {e}") from e
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError as e:
            raise GitHubError(f"{method} {path} -> malformed response body") from e

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, payload=payload)

    def graphql(self, query: str, variables: dict) -> Any:
        """Runs a GraphQL query, raising if it produced no data at all.

        A partial answer (some fields resolved, some errored) is returned as
        is; the caller decides what a missing field means.
        """
        body = self._request(
            "POST",
            "graphql",
            payload={"query": query, "variables": variables},
            url=f"{API_ROOT}/graphql",
        )
        if not isinstance(body, dict) or not body.get("data"):
            errors = body.get("errors") if isinstance(body, dict) else body
            raise GitHubError(f"POST graphql -> no data: {str(errors)[:200]}")
        return body["data"]


def _header_int(headers: Any, name: str) -> Optional[int]:
    """Reads an integer header, tolerating absent, empty and malformed ones."""
    try:
        return int(headers[name])
    except (KeyError, TypeError, ValueError):
        return None


class PostPacer:
    """Spaces out writes so a sweep stays under the content-generation limit."""

    def __init__(self, per_minute: int) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._last: Optional[float] = None

    def wait(self) -> None:
        if not self._interval:
            return
        if self._last is not None:
            pause = self._interval - (time.monotonic() - self._last)
            if pause > 0:
                time.sleep(pause)
        self._last = time.monotonic()


class RetryBudget:
    """A whole-run allowance for sleeping between status-post attempts."""

    def __init__(self, seconds: float) -> None:
        self.remaining = seconds

    def spend(self, seconds: float) -> float:
        """Returns how much of ``seconds`` is affordable, and charges for it."""
        allowed = max(0.0, min(seconds, self.remaining))
        self.remaining -= allowed
        return allowed


@dataclass
class PullRequest:
    # None when the run was scoped by head SHA alone, as the CI-run trigger
    # does: a fork PR's run does not say which PR it belongs to, and the
    # status lands on the head either way.
    number: Optional[int]
    head_sha: str
    draft: bool
    # When the PR was last pointed at a different base branch. A retarget
    # starts no CI run, so a run from before it merged some other branch.
    retargeted_at: Optional[datetime] = None

    @property
    def label(self) -> str:
        return f"#{self.number}" if self.number else self.head_sha[:12]


@dataclass
class Cutoff:
    """The commit that was the base branch tip ``--max-age-days`` ago."""

    sha: str
    landed_at: datetime


@dataclass
class CIRun:
    """The newest CI run for a PR head, which is the result the PR shows."""

    id: int
    url: str
    attempt: int
    # When the run, and with it the merge of develop it tests, was created.
    # Fixed across re-runs, unlike run_started_at, which is exactly why a
    # re-run cannot make stale CI fresh. Also what lets the status
    # description stay byte-identical between sweeps and dedupe cleanly.
    created_at: datetime
    # Base branches of the PRs the run lists, as they stand now rather than
    # when the run was made. Empty for a fork PR, whose runs list no PR.
    base_refs: tuple[str, ...] = ()


@dataclass
class Verdict:
    """The freshness result for one PR, before it is translated to a status."""

    fresh: bool
    # None when the head has no CI run yet, or could not be evaluated.
    ci_run: Optional[CIRun] = None
    # Stale because the PR was retargeted after its CI run merged a base.
    retargeted: bool = False
    error: Optional[str] = None
    # Why this head gets no status at all, when it is not ours to report on.
    ignored: Optional[str] = None
    # Set when the rate-limit reserve stopped us before this PR was looked at.
    # A skipped PR gets no status at all, since posting one would itself cost
    # the request we just declined to spend.
    skipped: bool = False


@dataclass
class Assessment:
    """A verdict plus whatever status the PR already carries for our context."""

    verdict: Verdict
    # (state, description, target_url), the same shape ``publication`` builds.
    published: Optional[tuple[str, str, Optional[str]]] = None
    # Whether this assessment is still holding the request it claimed for its
    # status write. That write happens in the serial phase after the whole
    # pool has finished, so the claim outlives the assessment and the writer
    # hands it back, whichever way the PR ends up going.
    holds_write_claim: bool = False


def parse_timestamp(value: str) -> datetime:
    """Parses a GitHub ISO-8601 timestamp into an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def find_cutoff_commit(
    gh: GitHub, base_branch: str, max_age_days: int, now: datetime
) -> Cutoff:
    """Returns the commit that was ``base_branch``'s tip ``max_age_days`` ago."""
    until = (now - timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = gh.get("commits", sha=base_branch, until=until, per_page=1)
    if not commits:
        raise GitHubError(f"no commit on {base_branch} older than {until}")
    try:
        # The committer date of a squash or merge-queue merge is when it
        # landed on the base branch, which is the moment that matters here.
        return Cutoff(
            sha=commits[0]["sha"],
            landed_at=parse_timestamp(commits[0]["commit"]["committer"]["date"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise GitHubError(f"cannot date the cutoff commit: {e}") from e


def find_ci_run(gh: GitHub, head_sha: str, workflow: str) -> Optional[CIRun]:
    """Returns the newest ``pull_request`` run of ``workflow`` for ``head_sha``.

    The newest run is the one whose result the PR shows. A label change or a
    reopen starts a new run for the same head with a fresh merge, so it is
    the newest run that matters, not the first.
    """
    response = gh.get(
        f"actions/workflows/{workflow}/runs",
        head_sha=head_sha,
        event="pull_request",
        per_page=1,
    )
    runs = response["workflow_runs"]
    # len() rather than truthiness, so a null list is an error, not "no run".
    if len(runs) == 0:
        return None
    run = runs[0]
    return CIRun(
        id=run["id"],
        url=run["html_url"],
        attempt=run.get("run_attempt") or 1,
        created_at=parse_timestamp(run["created_at"]),
        base_refs=tuple(p["base"]["ref"] for p in run.get("pull_requests") or ()),
    )


def evaluate(
    gh: GitHub, pr: PullRequest, workflow: str, base_branch: str, cutoff: Cutoff
) -> Verdict:
    """Decides whether the CI run ``pr`` shows merged the cutoff commit.

    Assumes the caller has already claimed the budget for this PR; see
    ``assess``, which is the only thing that should be calling it.

    Never raises. A verdict this cannot reach comes back as an error verdict,
    because an exception escaping here travels out through ``pool.map`` and
    ends the run before anything at all is published.
    """
    try:
        ci_run = find_ci_run(gh, pr.head_sha, workflow)
    except (GitHubError, KeyError, TypeError, ValueError, AttributeError) as e:
        return Verdict(fresh=True, error=str(e))
    if ci_run is None:
        return Verdict(fresh=True)
    if pr.number is None:
        # Scoped by head alone, which is the CI-run trigger. That fires for
        # every PR the CI workflow runs on, stacked PRs into feature branches
        # included, and those are judged against some other branch's cutoff
        # if at all. A re-run merges nothing new and its head carries no
        # retarget history here, so the sweep, which has both, owns it.
        if ci_run.base_refs and base_branch not in ci_run.base_refs:
            bases = ", ".join(ci_run.base_refs)
            return Verdict(
                fresh=True, ci_run=ci_run, ignored=f"CI for a PR into {bases}"
            )
        if ci_run.attempt > 1:
            return Verdict(fresh=True, ci_run=ci_run, ignored="a re-run of earlier CI")
    # The run merged the base it was given as it stood when it was created,
    # so it contains the cutoff exactly when the cutoff landed first, and it
    # merged this base at all only if the PR already targeted it.
    retargeted = pr.retargeted_at is not None and ci_run.created_at < pr.retargeted_at
    return Verdict(
        fresh=ci_run.created_at >= cutoff.landed_at and not retargeted,
        ci_run=ci_run,
        retargeted=retargeted,
    )


RETARGET_BATCH = 50


def find_retargets(gh: GitHub, numbers: list[int]) -> dict[int, datetime]:
    """Returns when each PR in ``numbers`` last changed its base branch.

    PRs that never did are absent. This is the one thing the run records
    cannot say: a run's ``pull_requests[].base`` is the PR's base now, so a
    stacked PR retargeted onto develop when its parent merged looks, from its
    run, as if CI had merged develop all along.

    GraphQL, because REST has no way to ask for one event type, and batched,
    because a timeline lookup per PR would double the cost of a sweep. A
    batch of this size costs one point from the separate GraphQL allowance.
    Raises ``GitHubError`` when a batch returns nothing usable; a partial
    answer is kept.
    """
    owner, name = gh.repo.split("/", 1)
    found: dict[int, datetime] = {}
    for start in range(0, len(numbers), RETARGET_BATCH):
        batch = numbers[start : start + RETARGET_BATCH]
        fields = " ".join(
            f"pr{n}: pullRequest(number: {int(n)}) {{ timelineItems("
            f"itemTypes: [BASE_REF_CHANGED_EVENT], last: 1) "
            f"{{ nodes {{ ... on BaseRefChangedEvent {{ createdAt }} }} }} }}"
            for n in batch
        )
        query = (
            "query($owner: String!, $name: String!) "
            f"{{ repository(owner: $owner, name: $name) {{ {fields} }} }}"
        )
        data = gh.graphql(query, {"owner": owner, "name": name})
        try:
            repository = data["repository"]
            for n in batch:
                nodes = (
                    (repository.get(f"pr{n}") or {}).get("timelineItems") or {}
                ).get("nodes") or []
                if nodes and nodes[-1].get("createdAt"):
                    found[n] = parse_timestamp(nodes[-1]["createdAt"])
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise GitHubError(f"malformed retarget answer: {e}") from e
    return found


def current_status(
    gh: GitHub, head_sha: str, context: str
) -> Optional[tuple[str, str, Optional[str]]]:
    """Returns the ``(state, description, target_url)`` published for ``context``.

    Never raises, for the same reason ``evaluate`` does not. Anything it
    cannot read is reported as nothing published, which costs a redundant
    write rather than a missing status.
    """
    try:
        combined = gh.get(f"commits/{head_sha}/status", per_page=100)
        for status in combined["statuses"]:
            if status.get("context") == context:
                return (
                    status.get("state", ""),
                    status.get("description") or "",
                    status.get("target_url") or None,
                )
    except (GitHubError, KeyError, TypeError, AttributeError):
        return None
    return None


def assess(
    gh: GitHub,
    pr: PullRequest,
    workflow: str,
    base_branch: str,
    cutoff: Cutoff,
    context: str,
) -> Assessment:
    """Evaluates ``pr`` and reads back whatever status it already carries.

    Admission happens here, for the whole PR at once: the CI run lookup, the
    read-back, and the write that comes later. Claiming all three up front is
    what makes "never evaluate a PR we cannot publish for" true, and holding
    the claim rather than merely checking it is what stops the other workers
    admitting themselves against headroom this one already spoke for.

    The read share is given back as it is spent. The write share is not, since
    the write has not happened yet; the caller releases that one.
    """
    if not gh.claim_budget(REQUESTS_PER_PR):
        return Assessment(Verdict(fresh=True, skipped=True))
    verdict = evaluate(gh, pr, workflow, base_branch, cutoff)
    published = current_status(gh, pr.head_sha, context)
    gh.release_budget(REQUESTS_PER_PR - REQUESTS_PER_WRITE)
    return Assessment(verdict, published, holds_write_claim=True)


def describe(
    verdict: Verdict, base_branch: str, max_age_days: int, enforcing: bool
) -> str:
    """Builds the one-line commit status description.

    Every branch here is worded to be stable for an unchanged PR. A status
    that says "6.0 days old" would have to be rewritten on every sweep as
    that number drifts, which for a queue this size is a lot of writes to say
    nothing new, so the stale case states the date of the CI run instead,
    which only a new run can change.

    The stale case also says what does not help. Re-running is the obvious
    reaction to a CI result that looks old, and it is the one that reuses
    the old merge.
    """
    if verdict.error:
        text = f"Could not evaluate CI freshness ({verdict.error})"
    elif verdict.fresh and verdict.ci_run is None:
        text = f"No CI run on this head yet; the next one merges current {base_branch}"
    elif verdict.fresh:
        text = f"CI merged {base_branch} within the last {max_age_days} days"
    else:
        advice = (
            f"Push or merge {base_branch} for a new run; re-running jobs "
            f"reuses that base"
        )
        if verdict.retargeted:
            text = f"CI ran before this PR was retargeted to {base_branch}. {advice}"
        else:
            since = verdict.ci_run.created_at.date().isoformat()
            text = f"CI tested against {base_branch} from {since}. {advice}"
        if not enforcing:
            text = f"WOULD FAIL: {text}"
    return text[:MAX_DESCRIPTION_LENGTH]


def ci_age_days(verdict: Verdict, now: datetime) -> float:
    """How old the merge ``verdict``'s CI run tested is, for the run summary."""
    if not verdict.ci_run:
        return 0.0
    return (now - verdict.ci_run.created_at).total_seconds() / 86400


def status_state(verdict: Verdict, enforcing: bool) -> str:
    return (
        "failure"
        if enforcing and not verdict.fresh and not verdict.error
        else "success"
    )


def status_target_url(verdict: Verdict, this_run_url: Optional[str]) -> Optional[str]:
    """Where the status links: the CI run being judged, when there is one.

    That run's date is what the verdict rests on. An error links to this job,
    whose log says what went wrong. A head with no run yet links nowhere:
    this job's URL changes every sweep, and a link that changes is a write.
    """
    if verdict.ci_run:
        return verdict.ci_run.url
    return this_run_url if verdict.error else None


def publication(
    verdict: Verdict,
    base_branch: str,
    max_age_days: int,
    enforcing: bool,
    this_run_url: Optional[str],
) -> tuple[str, str, Optional[str]]:
    """The ``(state, description, target_url)`` a verdict should be shown as.

    The same shape ``current_status`` reads back, so an unchanged status can
    be skipped by comparing the two. The link is part of it: a new CI run on
    the same head leaves a fresh PR's description untouched, and comparing
    the text alone would keep linking the run it replaced.
    """
    return (
        status_state(verdict, enforcing),
        describe(verdict, base_branch, max_age_days, enforcing),
        status_target_url(verdict, this_run_url),
    )


def publish_status(
    gh: GitHub,
    pacer: PostPacer,
    retries: RetryBudget,
    head_sha: str,
    payload: dict,
    sleep=time.sleep,
) -> Optional[str]:
    """Posts a commit status, retrying failures that can clear on their own.

    Returns ``None`` once the status is up, or the last error if it never got
    there. A caller that drops the error on the floor would leave an enforcing
    ruleset waiting on a context that is simply absent, so it is reported
    rather than merely warned about.
    """
    last = ""
    for attempt in range(1, POST_ATTEMPTS + 1):
        pacer.wait()
        try:
            gh.post(f"statuses/{head_sha}", payload)
            return None
        except GitHubError as e:
            last = str(e)
            if not e.retryable or attempt == POST_ATTEMPTS:
                break
            # GitHub says when to come back for a rate limit; otherwise back
            # off linearly. Either way it comes out of the run's allowance.
            delay = retries.spend(e.retry_after or POST_RETRY_BACKOFF_SECONDS * attempt)
            if not delay:
                break
            print(f"::warning::status post failed ({e}); retrying in {delay:.0f}s")
            sleep(delay)
    return last


@dataclass
class Listing:
    """The PRs a sweep will look at, and why the list stops where it does.

    Both truncations leave real PRs with no status at all from the run, which
    under enforcement is a block, so neither is allowed to be inferred from a
    suspiciously round count.
    """

    prs: list[PullRequest] = field(default_factory=list)
    truncated_at_limit: bool = False
    truncated_by_reserve: bool = False


def list_open_prs(
    gh: GitHub, base_branch: str, include_drafts: bool, limit: int
) -> Listing:
    """Lists open PRs targeting ``base_branch``, least recent activity first.

    Ordering matters when the reserve cuts a sweep short. The CI-run trigger
    already posts a status whenever a PR gets a new CI run, so a PR touched
    minutes ago has an accurate one. The PRs whose statuses are most likely
    to be wrong are the ones nothing has touched in days, so they go first
    and the freshly evaluated ones are what gets dropped.

    Paging deliberately runs one PR past ``limit`` so that hitting the cap can
    be reported rather than guessed at: a list that stops exactly on the cap
    looks identical whether or not there was more behind it.

    Listing is only a handful of requests, but the reserve is a promise about
    the shared allowance as a whole, so the pages are checked against it too
    rather than treated as overhead that does not count.
    """
    prs: list[PullRequest] = []
    page = 1
    while len(prs) <= limit:
        if not gh.has_budget(1):
            return Listing(prs[:limit], truncated_by_reserve=True)
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
    return Listing(prs[:limit], truncated_at_limit=len(prs) > limit)


def fetch_pr(gh: GitHub, number: int, base_branch: str) -> Optional[PullRequest]:
    """Fetches one PR, or None if it targets some other base branch."""
    item = gh.get(f"pulls/{number}")
    if item["base"]["ref"] != base_branch:
        return None
    return PullRequest(item["number"], item["head"]["sha"], item["draft"])


def day_count(value: str) -> int:
    """Argparse type for ``--max-age-days``, which a manual dispatch can set.

    Believing a bad value here is worse than refusing it. Zero or negative
    puts the cutoff at or after now, so ``develop``'s own tip becomes the
    cutoff and every PR that is not exactly on it reads as stale, which under
    enforcement is the whole queue blocked at once.
    """
    try:
        days = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number of days: {value!r}")
    if days < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 day, got {days}")
    return days


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
    parser.add_argument("--max-age-days", type=day_count, default=3)
    parser.add_argument(
        "--context", default="ci-freshness", help="Commit status context name"
    )
    parser.add_argument(
        "--ci-workflow",
        default=DEFAULT_CI_WORKFLOW,
        help="Workflow file whose newest PR run for a head is judged",
    )
    parser.add_argument(
        "--mode",
        choices=("report", "enforce"),
        default="report",
        help="report: always post success, with the verdict in the description",
    )
    parser.add_argument("--pr", type=int, help="Evaluate a single PR")
    parser.add_argument(
        "--head-sha",
        help="Evaluate a single PR head, as the CI-run trigger does; fork PR "
        "runs do not say which PR they belong to",
    )
    parser.add_argument(
        "--all-open", action="store_true", help="Evaluate every open PR"
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        # Sized to cover the whole queue with room to grow rather than to
        # ration requests: at roughly three per PR even a thousand of them is
        # a fifth of the hourly allowance.
        default=1000,
        help="Safety cap on PRs evaluated per run",
    )
    parser.add_argument(
        "--rate-limit-reserve",
        type=int,
        default=2000,
        help="Stop the sweep rather than drive the shared hourly limit below this",
    )
    parser.add_argument(
        "--posts-per-minute",
        type=int,
        default=POSTS_PER_MINUTE,
        help="Throttle status writes to stay under the content-generation limit",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print verdicts, post nothing"
    )
    args = parser.parse_args()

    if [bool(args.pr), bool(args.head_sha), args.all_open].count(True) != 1:
        parser.error("pass exactly one of --pr, --head-sha or --all-open")
    token = os.environ.get("GH_TOKEN")
    if not token:
        parser.error("GH_TOKEN is not set")

    enforcing = args.mode == "enforce"
    now = datetime.now(timezone.utc)
    single = bool(args.pr or args.head_sha)
    # A single PR triggered by its own event is worth spending reserve on; only
    # the sweep can plausibly exhaust the shared allowance.
    reserve = 0 if single else args.rate_limit_reserve
    gh = GitHub(args.repo, token, reserve=reserve)

    try:
        cutoff = find_cutoff_commit(gh, args.base_branch, args.max_age_days, now)
    except GitHubError as e:
        # Fail open, loudly. Posting nothing is worse than posting success:
        # an unreported required status blocks its PR with no way out.
        print(f"::warning::could not determine freshness cutoff, skipping run: {e}")
        return 0

    try:
        # Single-PR modes evaluate drafts too, so the status is already
        # present if the PR is marked ready and merged before the next
        # scheduled run.
        if args.pr:
            pr = fetch_pr(gh, args.pr, args.base_branch)
            if pr is None:
                print(
                    f"PR #{args.pr} does not target {args.base_branch}; nothing to do"
                )
                return 0
            listing = Listing([pr])
        elif args.head_sha:
            listing = Listing([PullRequest(None, args.head_sha, draft=False)])
        else:
            listing = list_open_prs(
                gh, args.base_branch, include_drafts=False, limit=args.max_prs
            )
    except GitHubError as e:
        print(f"::warning::could not list pull requests, skipping run: {e}")
        return 0
    prs = listing.prs

    if listing.truncated_at_limit:
        # Same class of problem as the reserve below: PRs that exist, target
        # develop, and get nothing from this run. Raising --max-prs is the
        # fix, and it has to happen before enforcement rather than after.
        print(
            f"::warning::more open PRs target {args.base_branch} than the "
            f"--max-prs cap of {args.max_prs}; the remainder were not "
            f"evaluated and have no status from this run"
        )
    if listing.truncated_by_reserve:
        print(
            "::warning::rate-limit reserve reached while listing pull "
            "requests; this sweep is working from a partial list"
        )

    numbered = [pr.number for pr in prs if pr.number]
    if numbered:
        try:
            retargets = find_retargets(gh, numbered)
        except GitHubError as e:
            # Fail open on this signal alone: the age check still stands, it
            # just cannot see a retarget this run.
            print(f"::warning::could not read PR retargets; judging on age only: {e}")
            retargets = {}
        for pr in prs:
            pr.retargeted_at = retargets.get(pr.number)

    this_run_url = None
    server, run_id = os.environ.get("GITHUB_SERVER_URL"), os.environ.get(
        "GITHUB_RUN_ID"
    )
    if server and run_id:
        this_run_url = f"{server}/{args.repo}/actions/runs/{run_id}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        assessments = list(
            pool.map(
                lambda pr: assess(
                    gh, pr, args.ci_workflow, args.base_branch, cutoff, args.context
                ),
                prs,
            )
        )

    stale = []
    skipped = []
    unpublished = []
    posted = 0
    unchanged = 0
    pacer = PostPacer(args.posts_per_minute)
    retries = RetryBudget(POST_RETRY_BUDGET_SECONDS)
    for pr, assessment in zip(prs, assessments):
        verdict = assessment.verdict
        if verdict.skipped:
            skipped.append(pr)
            continue
        try:
            if verdict.ignored:
                print(f"PR {pr.label}: no status - {verdict.ignored}")
                continue
            published = publication(
                verdict, args.base_branch, args.max_age_days, enforcing, this_run_url
            )
            state, description, target_url = published
            print(f"PR {pr.label}: {state} - {description}")
            if not verdict.fresh and not verdict.error:
                stale.append((pr, verdict))
            if args.dry_run:
                continue
            if assessment.published == published:
                # Re-posting an identical status would change nothing on the
                # PR and spend from the much tighter write budget. In steady
                # state this is most of the queue, since a fresh PR's
                # description is a constant string.
                unchanged += 1
                continue
            payload = {
                "state": state,
                "context": args.context,
                "description": description,
            }
            if target_url:
                payload["target_url"] = target_url
            error = publish_status(gh, pacer, retries, pr.head_sha, payload)
            if error:
                unpublished.append((pr, error))
            else:
                posted += 1
        finally:
            # The claim this PR has been holding since it was admitted. It is
            # given back here whether the write happened, was unnecessary, or
            # failed, so the headroom does not leak away over a long sweep.
            if assessment.holds_write_claim:
                gh.release_budget(REQUESTS_PER_WRITE)

    if unpublished:
        # An annotation rather than a non-zero exit, because the run did its
        # job for everything else and a red job here would be read as a
        # verdict. But this is the one outcome an enforcing ruleset turns into
        # a permanent block, so it has to be impossible to miss. The next
        # sweep retries these; it is a standing failure that needs the alarm.
        print(
            f"::error::{len(unpublished)} of {len(prs)} PRs have no "
            f"{args.context} status from this run: "
            + ", ".join(pr.label for pr, _ in unpublished[:20])
        )

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
        f"## CI freshness ({args.mode} mode)",
        "",
        f"Cutoff commit (`{args.base_branch}` tip {args.max_age_days} days ago): "
        f"`{cutoff.sha[:12]}`, landed {cutoff.landed_at.isoformat()}. A PR's "
        f"CI is stale when its newest `{args.ci_workflow}` run was created "
        f"before that, so it merged a `{args.base_branch}` without it, or "
        f"before the PR was retargeted to `{args.base_branch}`, so it merged "
        f"some other branch.",
        "",
        f"{len(stale)} of {evaluated} evaluated PRs have stale CI."
        + (f" {len(skipped)} skipped for rate-limit reserve." if skipped else ""),
        "",
    ]
    if listing.truncated_at_limit:
        summary += [
            f"**The queue is longer than the `--max-prs` cap of "
            f"{args.max_prs}.** The PRs past the cap were not evaluated and "
            f"have no status from this run.",
            "",
        ]
    if listing.truncated_by_reserve:
        summary += [
            "**Listing stopped at the rate-limit reserve.** This sweep ran "
            "against a partial list of open PRs.",
            "",
        ]
    if stale:
        # The age belongs here rather than in the status description: this
        # table is rebuilt every run anyway, so a drifting number costs
        # nothing.
        summary += [
            "| PR | CI run | Attempt | CI age (days) | Why |",
            "| --- | --- | --- | --- | --- |",
        ]
        for pr, verdict in sorted(stale, key=lambda s: -ci_age_days(s[1], now)):
            run = verdict.ci_run
            why = "retargeted after the run" if verdict.retargeted else "age"
            summary.append(
                f"| {pr.label} | [{run.id}]({run.url}) | {run.attempt} | "
                f"{ci_age_days(verdict, now):.1f} | {why} |"
            )
        summary.append("")
    if unpublished:
        # Named individually, because clearing this is per-PR work: these are
        # the ones an enforcing ruleset would have nothing to check against.
        summary += [
            f"**{len(unpublished)} statuses could not be published** after "
            f"{POST_ATTEMPTS} attempts:",
            "",
        ]
        summary += [f"- {pr.label}: {error}" for pr, error in unpublished]
        summary.append("")
    summary.append(
        f"_{gh.request_count} API requests ({posted} statuses written, "
        f"{unchanged} already correct"
        + (f", {len(unpublished)} failed" if unpublished else "")
        + f"); rate limit {gh.rate_limit_remaining}/{gh.rate_limit_limit} "
        f"remaining (reserve {gh.reserve})_"
    )
    write_summary(summary)

    # The job itself always succeeds. The verdict travels as a commit status,
    # which is what a ruleset can require; a red job here would only be noise.
    return 0


if __name__ == "__main__":
    sys.exit(main())
