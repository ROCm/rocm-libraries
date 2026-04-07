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
)

# All merge-queue state for a PR lives in a single comment.  The hidden
# metadata marker stores the JSON payload; the visible portion shows the
# queue status table.  Both ``enqueue_pr`` and ``update_status_comment``
# write to this same comment, keeping PR timelines clean.

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


def _build_status_body(
    metadata: dict,
    queues: List[str],
    queue_positions: Optional[List[dict]] = None,
    blocking: Optional[List[str]] = None,
    extra_msg: Optional[str] = None,
) -> str:
    """Build the single comment body that holds metadata + status table."""
    hidden = f"{METADATA_COMMENT_MARKER} {json.dumps(metadata)} -->"

    rows: list[str] = []
    if queue_positions:
        for qp in queue_positions:
            rows.append(
                f"| `{qp['queue']}` | {qp['position']}/{qp['total']} | {qp['status']} |"
            )
    else:
        # Initial enqueue — no position data yet
        for q in queues:
            rows.append(f"| `{q}` | — | Queued |")

    blocking = blocking or []
    if blocking:
        overall = f"Waiting for: {', '.join(f'`{q}`' for q in blocking)}"
    elif queue_positions:
        overall = "At front of all queues — processing"
    else:
        overall = "Queued"

    table = "\n".join(rows)
    user = metadata.get("enqueued_by", "")
    header = f"**Merge Queue** — enqueued by @{user}" if user else "**Merge Queue**"

    body = (
        f"{hidden}\n"
        f"## {header}\n\n"
        f"| Queue | Position | Status |\n"
        f"|-------|----------|--------|\n"
        f"{table}\n\n"
        f"**Overall:** {overall}"
    )
    if extra_msg:
        body += f"\n\n{extra_msg}"
    return body


def _find_mq_comment(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> Optional[dict]:
    """Find the merge-queue comment on a PR.  Returns {id, body} or None."""
    comments = client.get_comments(repo, pr_number)
    for comment in comments:
        if METADATA_COMMENT_MARKER in comment.get("body", ""):
            return {"id": comment["id"], "body": comment["body"]}
    return None


def get_enqueue_metadata(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> Optional[dict]:
    """Read merge-queue metadata from the PR's queue comment."""
    comment = _find_mq_comment(client, repo, pr_number)
    if comment:
        meta = _parse_metadata_comment(comment["body"])
        if meta is not None:
            meta["_comment_id"] = comment["id"]
            return meta
    return None


# ── Enqueue / Dequeue ────────────────────────────────────────────────


def enqueue_pr(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    queues: List[str],
    user: str,
) -> None:
    """Add a PR to the merge queue.

    Adds labels and posts a single comment containing both the hidden
    metadata and the initial status table.
    """
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "enqueued_at": now,
        "queues": queues,
        "enqueued_by": user,
    }

    labels = [LABEL_QUEUED] + [f"{LABEL_PREFIX}{q}" for q in queues]
    client.add_labels(repo, pr_number, labels)

    body = _build_status_body(metadata, queues)
    client.add_comment(repo, pr_number, body)

    logger.info(f"PR #{pr_number} enqueued in {queues} by @{user}")


def dequeue_pr(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    reason: str,
) -> None:
    """Remove a PR from all merge queues.

    Strips every ``mq:*`` label and updates the queue comment with the reason.
    """
    existing = client.get_existing_labels_on_pr(repo, pr_number)
    mq_labels = [l for l in existing if l.startswith(LABEL_PREFIX)]
    for label in mq_labels:
        client.remove_label(repo, pr_number, label)

    # Update the existing queue comment rather than posting a new one
    comment = _find_mq_comment(client, repo, pr_number)
    if comment:
        client.update_comment(
            repo, comment["id"], f"**Merge Queue:** {reason}"
        )
    else:
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

    # Check if actually behind using the compare API.
    # mergeable_state can be "clean" even when behind (it only indicates
    # conflict status, not whether the branch is up-to-date).
    head_branch = pr_data.get("head", {}).get("ref", "")
    base_branch = pr_data.get("base", {}).get("ref", "")
    if head_branch and base_branch:
        compare = client._get_json(
            f"https://api.github.com/repos/{repo}/compare/{base_branch}...{head_branch}",
            f"compare {base_branch}...{head_branch}",
        )
        behind_by = compare.get("behind_by", 0) if compare else 0
    else:
        behind_by = 0

    if behind_by == 0:
        logger.info(
            f"PR #{pr_number} branch is up to date with {base_branch}"
        )
        return True

    logger.info(
        f"PR #{pr_number} is {behind_by} commit(s) behind {base_branch}, "
        f"updating branch"
    )
    return client.update_pr_branch(repo, pr_number)


def merge_pr(
    client: GitHubCLIClient, repo: str, pr_number: int
) -> bool:
    """Squash-merge a PR via the GitHub API."""
    return client.merge_pr(repo, pr_number, method=MERGE_METHOD)


# ── Status comment ───────────────────────────────────────────────────


def update_status_comment(
    client: GitHubCLIClient,
    repo: str,
    pr_number: int,
    queues: List[str],
    blocking: List[str],
) -> None:
    """Update the queue comment in place with current position data."""
    comment = _find_mq_comment(client, repo, pr_number)
    if not comment:
        return  # no queue comment to update

    metadata = _parse_metadata_comment(comment["body"])
    if not metadata:
        return

    queue_positions: list[dict] = []
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
        queue_positions.append(
            {"queue": queue, "position": position, "total": total, "status": status}
        )

    body = _build_status_body(metadata, queues, queue_positions, blocking)
    client.update_comment(repo, comment["id"], body)
