#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Collect and publish a concise pull-request CI overview.

GitHub's pull-request check rollup contains one entry for every Actions job,
including every matrix shard. This collector deliberately uses two coarser
inputs instead:

* the checks GitHub identifies as required for the pull request; and
* the latest top-level Actions run for each non-gating workflow.

The normalized result is rendered by :mod:`pr_ci_overview`. By default the
Markdown is printed. ``--update-comment`` creates or updates one marker-tagged
pull-request comment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from pr_ci_overview import InputError, parse_overview, render_overview


COMMENT_MARKER = "<!-- rocm-libraries-ci-overview -->"
COMMENT_AUTHOR = "github-actions[bot]"
_EPOCH = "1970-01-01T00:00:00Z"

# These workflows manage the pull request rather than test its contents.
_IGNORED_WORKFLOWS = {
    "Auto Label PR",
    "Docs Preview Cleanup",
    "Libraries PR Bot",
    "PR CI Overview",
}

_STATE_MAP = {
    "ACTION_REQUIRED": "failure",
    "CANCELLED": "cancelled",
    "ERROR": "failure",
    "EXPECTED": "pending",
    "FAILURE": "failure",
    "IN_PROGRESS": "pending",
    "NEUTRAL": "neutral",
    "PENDING": "pending",
    "QUEUED": "pending",
    "SKIPPED": "skipped",
    "STALE": "failure",
    "STARTUP_FAILURE": "failure",
    "SUCCESS": "success",
    "TIMED_OUT": "failure",
    "WAITING": "pending",
}


class GitHubError(RuntimeError):
    """Raised when GitHub data cannot be collected or updated."""


def gh_json(
    *args: str,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> Any:
    """Run ``gh`` and decode its JSON output."""
    try:
        completed = subprocess.run(
            ["gh", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GitHubError(f"could not run gh: {exc}") from exc
    if completed.returncode not in allowed_returncodes:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitHubError(
            f"gh {' '.join(args[:2])} failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    if not completed.stdout.strip():
        return []
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GitHubError("gh returned invalid JSON") from exc


def _pr_metadata(repo: str, pr_number: int) -> dict[str, Any]:
    return _object(
        gh_json(
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo,
            "--json",
            "number,headRefOid,baseRefName,state",
        ),
        "pull request",
    )


def collect_overview(repo: str, pr_number: int) -> dict[str, list[dict[str, Any]]]:
    """Collect normalized required checks and top-level workflow runs."""
    pr = _pr_metadata(repo, pr_number)
    head_sha = _string(pr.get("headRefOid"), "pull request headRefOid")
    base_ref = _string(pr.get("baseRefName"), "pull request baseRefName")

    expected_contexts = _required_contexts(repo, base_ref)
    reported_required = _reported_required_checks(repo, pr_number)
    required_checks = _normalize_required(expected_contexts, reported_required)

    gating_run_ids = {
        run_id
        for item in reported_required
        if (run_id := _run_id_from_link(item.get("link"))) is not None
    }
    runs = _workflow_runs(repo, head_sha, pr_number)
    gating_workflow_ids = {
        workflow_id
        for run in runs
        if run.get("id") in gating_run_ids
        and isinstance((workflow_id := run.get("workflow_id")), int)
        and not isinstance(workflow_id, bool)
    }
    workflow_runs = _normalize_workflow_runs(runs, gating_run_ids, gating_workflow_ids)
    return {
        "required_checks": required_checks,
        "workflow_runs": workflow_runs,
    }


def resolve_prs(repo: str, sha: str) -> list[int]:
    """Return every open pull request associated with ``sha``."""
    pulls = gh_json("api", f"repos/{repo}/commits/{sha}/pulls")
    if not isinstance(pulls, list):
        raise GitHubError("commit pulls response must be an array")
    candidates = [item for item in pulls if item.get("state") == "open"]
    numbers: list[int] = []
    for candidate in candidates:
        number = candidate.get("number")
        if isinstance(number, bool) or not isinstance(number, int):
            raise GitHubError("associated pull request has no numeric number")
        if number not in numbers:
            numbers.append(number)
    return numbers


def resolve_pr(repo: str, sha: str) -> int | None:
    """Resolve one open pull request or require an explicit number."""
    numbers = resolve_prs(repo, sha)
    if len(numbers) > 1:
        joined = ", ".join(str(number) for number in numbers)
        raise GitHubError(
            f"commit {sha} belongs to multiple open pull requests ({joined}); use --pr"
        )
    return numbers[0] if numbers else None


def resolve_targets(
    repo: str, *, pr_number: int | None = None, sha: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Return a GitHub Actions matrix of validated open pull requests."""
    numbers = [pr_number] if pr_number is not None else resolve_prs(repo, str(sha))
    include: list[dict[str, Any]] = []
    for number in numbers:
        pr = _pr_metadata(repo, number)
        if str(pr.get("state") or "").upper() != "OPEN":
            continue
        include.append(
            {
                "pr": number,
                "head_sha": _string(pr.get("headRefOid"), "pull request headRefOid"),
            }
        )
    return {"include": include}


def upsert_comment(repo: str, pr_number: int, body: str) -> None:
    """Create or update this reporter's marker-tagged pull-request comment."""
    pages = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/issues/{pr_number}/comments?per_page=100",
    )
    comments = _flatten_pages(pages, dict)
    existing = next(
        (
            comment
            for comment in comments
            if COMMENT_MARKER in str(comment.get("body", ""))
            and _comment_author(comment) == COMMENT_AUTHOR
        ),
        None,
    )
    if existing:
        if existing.get("body") == body:
            return
        comment_id = existing.get("id")
        if isinstance(comment_id, bool) or not isinstance(comment_id, int):
            raise GitHubError("existing overview comment has no numeric id")
        endpoint = f"repos/{repo}/issues/comments/{comment_id}"
        method = "PATCH"
    else:
        endpoint = f"repos/{repo}/issues/{pr_number}/comments"
        method = "POST"
    gh_json("api", "--method", method, endpoint, "-f", f"body={body}")


def _required_contexts(repo: str, base_ref: str) -> list[str]:
    encoded_ref = quote(base_ref, safe="")
    pages = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/rules/branches/{encoded_ref}?per_page=100",
    )
    rules = _flatten_pages(pages, dict)

    contexts: list[str] = []
    integrations: dict[str, object] = {}
    for rule in rules:
        rule_type = rule.get("type")
        if rule_type in {"workflows", "required_workflows"}:
            raise GitHubError(
                "required-workflow rules are not yet supported by the CI overview"
            )
        if rule_type != "required_status_checks":
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            continue
        required = parameters.get("required_status_checks", [])
        if not isinstance(required, list):
            continue
        for item in required:
            if not isinstance(item, dict):
                continue
            context = item.get("context")
            if not isinstance(context, str) or not context:
                continue
            integration = item.get("integration_id")
            if context in integrations and integrations[context] != integration:
                raise GitHubError(
                    f"required context {context!r} is configured for multiple apps"
                )
            integrations[context] = integration
            if context not in contexts:
                contexts.append(context)
    return contexts


def _reported_required_checks(repo: str, pr_number: int) -> list[dict[str, Any]]:
    result = gh_json(
        "pr",
        "checks",
        str(pr_number),
        "--repo",
        repo,
        "--required",
        "--json",
        "bucket,completedAt,link,name,startedAt,state,workflow",
        # `gh pr checks` uses 1 for failures and 8 while checks are pending.
        allowed_returncodes=frozenset({0, 1, 8}),
    )
    if not isinstance(result, list):
        raise GitHubError("required checks response must be an array")
    return [item for item in result if isinstance(item, dict)]


def _workflow_runs(repo: str, head_sha: str, pr_number: int) -> list[dict[str, Any]]:
    pages = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/actions/runs?head_sha={head_sha}&event=pull_request&per_page=100",
    )
    objects = _flatten_pages(pages, dict)
    runs: list[dict[str, Any]] = []
    for page in objects:
        page_runs = page.get("workflow_runs", [])
        if isinstance(page_runs, list):
            for item in page_runs:
                if not isinstance(item, dict):
                    continue
                pull_requests = item.get("pull_requests")
                if not isinstance(pull_requests, list):
                    continue
                if any(
                    isinstance(pr, dict) and pr.get("number") == pr_number
                    for pr in pull_requests
                ):
                    runs.append(item)
    return runs


def _normalize_required(
    expected_contexts: Sequence[str], reported: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    reported_names: set[str] = set()
    name_counts = Counter(
        str(item.get("name") or "").strip()
        for item in reported
        if str(item.get("name") or "").strip()
    )
    for index, item in enumerate(reported, start=1):
        name = _string(item.get("name"), "required check name")
        reported_names.add(name)
        workflow = str(item.get("workflow") or "").strip()
        identity = workflow or "external-status"
        display_name = name
        if name_counts[name] > 1:
            display_name = f"{name} — {workflow or 'external status'}"
        normalized.append(
            {
                "key": f"required:{identity}:{name}",
                "name": display_name,
                "state": _normalize_state(item.get("state")),
                "updated_at": _check_timestamp(item),
                "id": _id_from_link(item.get("link"), index),
                "url": item.get("link") or None,
            }
        )

    for name in expected_contexts:
        if name not in reported_names:
            normalized.append(
                {
                    "key": f"required:missing:{name}",
                    "name": name,
                    "state": "pending",
                    "updated_at": _EPOCH,
                    "id": 0,
                }
            )
    return normalized


def _normalize_workflow_runs(
    runs: Iterable[Mapping[str, Any]],
    gating_run_ids: set[int],
    gating_workflow_ids: set[int],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for run in runs:
        name = str(run.get("name") or "").strip()
        if not name or name in _IGNORED_WORKFLOWS:
            continue
        workflow_id = run.get("workflow_id")
        run_id = run.get("id")
        if isinstance(workflow_id, bool) or not isinstance(workflow_id, int):
            continue
        if isinstance(run_id, bool) or not isinstance(run_id, int):
            continue
        if run_id in gating_run_ids or workflow_id in gating_workflow_ids:
            continue
        normalized.append(
            {
                "key": f"workflow:{workflow_id}",
                "name": name,
                "state": _workflow_state(run),
                "updated_at": _run_timestamp(run),
                "id": run_id,
                "url": run.get("html_url") or None,
            }
        )
    return normalized


def _workflow_state(run: Mapping[str, Any]) -> str:
    status = str(run.get("status") or "").upper()
    if status != "COMPLETED":
        return "pending"
    return _normalize_state(run.get("conclusion"))


def _normalize_state(value: object) -> str:
    return _STATE_MAP.get(str(value or "").upper(), "pending")


def _check_timestamp(item: Mapping[str, Any]) -> str:
    # The start time identifies the newest attempt. An older check can finish
    # cancelling after its replacement starts, so completedAt is not a safe
    # rerun ordering key.
    return _first_timestamp(item.get("startedAt"), item.get("completedAt"))


def _run_timestamp(item: Mapping[str, Any]) -> str:
    return _first_timestamp(
        item.get("run_started_at"), item.get("created_at"), item.get("updated_at")
    )


def _first_timestamp(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            return value
    return _EPOCH


def _id_from_link(value: object, fallback: int) -> int:
    if isinstance(value, str):
        for pattern in (r"/job/(\d+)(?:[/?#]|$)", r"/runs?/(\d+)(?:[/?#]|$)"):
            match = re.search(pattern, value)
            if match:
                return int(match.group(1))
    return fallback


def _run_id_from_link(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"/actions/runs/(\d+)(?:[/?#]|$)", value)
    return int(match.group(1)) if match else None


def _flatten_pages(value: object, item_type: type) -> list[Any]:
    if not isinstance(value, list):
        raise GitHubError("paginated GitHub response must be an array")
    if not value:
        return []
    if all(isinstance(item, item_type) for item in value):
        return list(value)
    flattened: list[Any] = []
    for page in value:
        if not isinstance(page, list):
            raise GitHubError("paginated GitHub response has an unexpected shape")
        flattened.extend(item for item in page if isinstance(item, item_type))
    return flattened


def _object(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GitHubError(f"{description} response must be an object")
    return value


def _comment_author(comment: Mapping[str, Any]) -> str:
    user = comment.get("user")
    if not isinstance(user, dict):
        return ""
    login = user.get("login")
    return login if isinstance(login, str) else ""


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GitHubError(f"{description} must be a non-empty string")
    return value.strip()


def _comment_body(repo: str, pr_number: int, markdown: str) -> str:
    checks_url = f"https://github.com/{repo}/pull/{pr_number}/checks"
    return (
        f"{COMMENT_MARKER}\n{markdown}\n"
        "<sub>Required checks come from the pull request's effective GitHub "
        "rules. Informational counts use top-level workflow conclusions; "
        "individual jobs and matrix shards are omitted. "
        f"[View all checks]({checks_url}).</sub>"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repository")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pr", type=int, help="pull-request number")
    target.add_argument("--sha", help="commit SHA used to resolve an open PR")
    parser.add_argument(
        "--update-comment",
        action="store_true",
        help="create or update the overview comment instead of printing Markdown",
    )
    parser.add_argument(
        "--resolve-targets",
        action="store_true",
        help="print an Actions matrix of open PRs instead of collecting results",
    )
    parser.add_argument(
        "--expected-head-sha",
        help="skip publication if the pull-request head changed after resolution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.resolve_targets:
            targets = resolve_targets(
                args.repo,
                pr_number=args.pr,
                sha=args.sha,
            )
            print(json.dumps(targets, separators=(",", ":")))
            return 0
        pr_number = args.pr if args.pr is not None else resolve_pr(args.repo, args.sha)
        if pr_number is None:
            print(f"No open pull request found for {args.sha}; nothing to report.")
            return 0
        if args.expected_head_sha and not _head_matches(
            args.repo, pr_number, args.expected_head_sha
        ):
            print(
                f"Pull request #{pr_number} moved to a new head; skipping stale update."
            )
            return 0
        normalized = collect_overview(args.repo, pr_number)
        markdown = render_overview(parse_overview(normalized))
        if args.update_comment:
            if args.expected_head_sha and not _head_matches(
                args.repo, pr_number, args.expected_head_sha
            ):
                print(
                    f"Pull request #{pr_number} moved while results were collected; "
                    "skipping stale update."
                )
                return 0
            upsert_comment(
                args.repo, pr_number, _comment_body(args.repo, pr_number, markdown)
            )
            print(f"Updated CI overview for {args.repo}#{pr_number}.")
        else:
            sys.stdout.write(markdown)
    except (GitHubError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _head_matches(repo: str, pr_number: int, expected: str) -> bool:
    current = _string(
        _pr_metadata(repo, pr_number).get("headRefOid"),
        "pull request headRefOid",
    )
    return current == expected


if __name__ == "__main__":
    raise SystemExit(main())
