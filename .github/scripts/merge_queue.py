#!/usr/bin/env python3

"""
Merge Queue Logic
-----------------
Shared functions for the hipDNN/provider merge queue system.

Provides queue detection, FIFO ordering, multi-queue coordination,
CI status checking, and PR merge operations.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from github_cli_client import GitHubCLIClient
from merge_queue_config import (
    ALL_QUEUES,
    LABEL_ACTIVE,
    LABEL_PREFIX,
    LABEL_QUEUED,
    MERGE_METHOD,
    METADATA_COMMENT_MARKER,
    PATH_TO_QUEUES,
    STATUS_COMMENT_MARKER,
)

logger = logging.getLogger(__name__)


def detect_queues(changed_files: List[str]) -> List[str]:
    """Map changed file paths to the set of merge queues the PR should enter.

    A file matching ``projects/hipdnn/`` enters all queues (core can break
    providers).  Provider files enter only their own queue.  The union across
    all changed files is returned, sorted and deduplicated.
    """
    queues: set[str] = set()
    for filepath in changed_files:
        for prefix, queue_list in PATH_TO_QUEUES.items():
            if filepath.startswith(prefix):
                queues.update(queue_list)
                break  # first prefix match per file is enough
    return sorted(queues)


# ── Metadata comment helpers ─────────────────────────────────────────


def _parse_metadata_comment(body: str) -> Optional[dict]:
    """Extract JSON from a merge-queue metadata HTML comment."""
    pattern = re.compile(
        rf"{re.escape(METADATA_COMMENT_MARKER)}\s*(\{{.*?\}})\s*-->",
        re.DOTALL,
    )
    match = pattern.search(body)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _build_metadata_comment(metadata: dict) -> str:
    """Build the hidden HTML comment that stores queue metadata on a PR."""
    return f"{METADATA_COMMENT_MARKER} {json.dumps(metadata)} -->"


def get_enqueue_metadata(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> Optional[dict]:
    """Read merge-queue metadata from PR comments.  Returns None if not queued."""
    comments = client.get_comments(repo, pr_number)
    for comment in comments:
        meta = _parse_metadata_comment(comment.get("body", ""))
        if meta is not None:
            meta["_comment_id"] = comment["id"]
            return meta
    return None


def _delete_metadata_comment(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> None:
    """Remove the metadata comment from a PR (cleanup on dequeue)."""
    comments = client.get_comments(repo, pr_number)
    for comment in comments:
        if METADATA_COMMENT_MARKER in comment.get("body", ""):
            url = f"{client.api_url}/repos/{repo}/issues/comments/{comment['id']}"
            client._request_json("DELETE", url, None, "Failed to delete metadata comment")
            break


# ── Enqueue / Dequeue ────────────────────────────────────────────────


def enqueue_pr(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    queues: List[str],
    user: str,
) -> None:
    """Add a PR to the merge queue.

    Adds the ``mq:queued`` label plus one ``mq:<queue>`` label per queue,
    and posts a hidden metadata comment for FIFO ordering.
    """
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "enqueued_at": now,
        "queues": queues,
        "enqueued_by": user,
    }

    labels = [LABEL_QUEUED] + [f"{LABEL_PREFIX}{q}" for q in queues]
    client.add_labels(repo, pr_number, labels)
    client.add_comment(repo, pr_number, _build_metadata_comment(metadata))

    logger.info(f"PR #{pr_number} enqueued in {queues} by @{user}")


def dequeue_pr(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    reason: str,
) -> None:
    """Remove a PR from all merge queues.

    Strips every ``mq:*`` label and posts a comment explaining why.
    """
    existing = client.get_existing_labels_on_pr(repo, pr_number)
    mq_labels = [l for l in existing if l.startswith(LABEL_PREFIX)]
    for label in mq_labels:
        client.remove_label(repo, pr_number, label)

    _delete_metadata_comment(client, repo, pr_number)
    client.add_comment(repo, pr_number, f"**Merge Queue:** {reason}")
    logger.info(f"PR #{pr_number} dequeued: {reason}")


# ── Queue membership queries ────────────────────────────────────────


def get_queue_members(
    client: GitHubCLIClient, repo: str, queue: str
) -> List[dict]:
    """Return all PRs in a given queue, sorted oldest-first (FIFO).

    Each entry: ``{"pr_number": int, "enqueued_at": str, "queues": [str]}``.
    """
    label = f"{LABEL_PREFIX}{queue}"
    # Search for open PRs with the queue label.  Any PR carrying this label
    # is a queue member regardless of whether it is mq:queued or mq:active.
    query = f"repo:{repo} is:pr is:open label:\"{label}\""
    items = client.search_issues(query, sort="created", order="asc")

    members: list[dict] = []
    for item in items:
        pr_num = item["number"]
        meta = get_enqueue_metadata(client, repo, pr_num)
        if meta:
            members.append(
                {
                    "pr_number": pr_num,
                    "enqueued_at": meta["enqueued_at"],
                    "queues": meta["queues"],
                }
            )
        else:
            # Fallback: use the issue created_at if metadata is missing
            members.append(
                {
                    "pr_number": pr_num,
                    "enqueued_at": item.get("created_at", ""),
                    "queues": [],
                }
            )

    # Sort by enqueue timestamp (FIFO)
    members.sort(key=lambda m: m["enqueued_at"])
    return members


def get_queue_head(
    client: GitHubCLIClient, repo: str, queue: str
) -> Optional[int]:
    """Return the PR number at the front of a queue, or None if empty."""
    members = get_queue_members(client, repo, queue)
    return members[0]["pr_number"] if members else None


def is_at_front_of_all_queues(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    queues: List[str],
) -> Tuple[bool, List[str]]:
    """Check whether a PR is at the head of every queue it belongs to.

    Returns ``(is_ready, blocking_queues)`` where *blocking_queues* lists
    the queues where another PR is ahead.
    """
    blocking: list[str] = []
    for queue in queues:
        head = get_queue_head(client, repo, queue)
        if head != pr_number:
            blocking.append(queue)
    return (len(blocking) == 0, blocking)


# ── CI status ────────────────────────────────────────────────────────


def check_ci_status(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> str:
    """Check CI status for the PR's current HEAD commit.

    Returns ``"success"``, ``"pending"``, or ``"failure"``.
    """
    pr_data = client.get_pr_by_number(repo, pr_number)
    if not pr_data:
        return "failure"

    sha = pr_data.get("head", {}).get("sha", "")
    if not sha:
        return "failure"

    # Check both check-runs and commit statuses
    check_runs = client.get_check_runs(repo, sha)
    combined = client.get_combined_status(repo, sha)

    # If there are no checks at all, treat as pending (CI hasn't started)
    statuses = combined.get("statuses", [])
    if not check_runs and not statuses:
        return "pending"

    # Check runs
    for run in check_runs:
        # Skip the merge queue's own status checks
        if run.get("name", "").startswith("Merge Queue"):
            continue
        status = run.get("status", "")
        conclusion = run.get("conclusion", "")
        if status != "completed":
            return "pending"
        if conclusion not in ("success", "skipped", "neutral"):
            return "failure"

    # Commit statuses (from status API)
    combined_state = combined.get("state", "pending")
    if combined_state == "failure" or combined_state == "error":
        return "failure"
    if combined_state == "pending" and statuses:
        return "pending"

    return "success"


# ── Branch update and merge ──────────────────────────────────────────


def update_pr_branch(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> bool:
    """Merge the base branch into the PR branch.

    Returns True on success, False on merge conflict.
    """
    pr_data = client.get_pr_by_number(repo, pr_number)
    if not pr_data:
        return False

    # Check if already up to date
    mergeable_state = pr_data.get("mergeable_state", "")
    if mergeable_state == "clean":
        logger.info(f"PR #{pr_number} is already up to date")
        return True

    return client.update_pr_branch(repo, pr_number)


def merge_pr(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> bool:
    """Squash-merge a PR via the GitHub API."""
    return client.merge_pr(repo, pr_number, method=MERGE_METHOD)


# ── Status comment ───────────────────────────────────────────────────


def _find_status_comment(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> Optional[int]:
    """Find the ID of the existing status comment on a PR, if any."""
    comments = client.get_comments(repo, pr_number)
    for comment in comments:
        if STATUS_COMMENT_MARKER in comment.get("body", ""):
            return comment["id"]
    return None


def update_status_comment(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    queues: List[str],
    blocking: List[str],
) -> None:
    """Create or update the queue status table comment on a PR."""
    rows: list[str] = []
    for queue in queues:
        members = get_queue_members(client, repo, queue)
        total = len(members)
        position = next(
            (i + 1 for i, m in enumerate(members) if m["pr_number"] == pr_number),
            total,
        )
        if queue in blocking:
            ahead_pr = members[0]["pr_number"] if members else "?"
            status = f"Waiting (PR #{ahead_pr} ahead)"
        elif position == 1:
            status = "At front"
        else:
            status = f"Position {position}"
        rows.append(f"| `{queue}` | {position}/{total} | {status} |")

    if blocking:
        overall = f"Waiting for: {', '.join(f'`{q}`' for q in blocking)}"
    else:
        overall = "At front of all queues — processing"

    table = "\n".join(rows)
    body = (
        f"{STATUS_COMMENT_MARKER} -->\n"
        f"## Merge Queue Status\n\n"
        f"| Queue | Position | Status |\n"
        f"|-------|----------|--------|\n"
        f"{table}\n\n"
        f"**Overall:** {overall}"
    )

    comment_id = _find_status_comment(client, repo, pr_number)
    if comment_id:
        client.update_comment(repo, comment_id, body)
    else:
        client.add_comment(repo, pr_number, body)
