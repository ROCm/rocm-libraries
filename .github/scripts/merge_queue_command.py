#!/usr/bin/env python3

"""
Merge Queue Command Handler
----------------------------
Handles ``/merge`` and ``/dequeue`` PR comment commands.

Called by the ``merge-queue-command.yml`` workflow when a comment is
created on a pull request.

Environment variables (set by the workflow):
    GH_TOKEN      – GitHub API token
    REPO          – owner/repo (e.g. ``ROCm/rocm-libraries``)
    PR_NUMBER     – pull request number
    COMMENT_BODY  – the full comment text
    COMMENT_AUTHOR – GitHub login of the commenter
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from github_cli_client import GitHubCLIClient
from merge_queue import (
    detect_queues,
    dequeue_pr,
    enqueue_pr,
    get_enqueue_metadata,
    get_queue_members,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    client = GitHubCLIClient()
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])
    comment_body = os.environ["COMMENT_BODY"].strip()
    comment_author = os.environ["COMMENT_AUTHOR"]

    if comment_body.startswith("/merge"):
        handle_merge(client, repo, pr_number, comment_author)
    elif comment_body.startswith("/dequeue"):
        handle_dequeue(client, repo, pr_number, comment_author)
    else:
        logger.info("Comment does not contain a merge queue command")


def _has_write_access(client: GitHubCLIClient, repo: str, username: str) -> bool:
    perm = client.get_collaborator_permission(repo, username)
    return perm in ("admin", "maintain", "write")


def handle_merge(
    client: GitHubCLIClient, repo: str, pr_number: int, user: str
) -> None:
    # Permission check
    if not _has_write_access(client, repo, user):
        client.add_comment(
            repo,
            pr_number,
            f"@{user} You need write permission to use `/merge`.",
        )
        return

    # Already queued?
    existing = get_enqueue_metadata(client, repo, pr_number)
    if existing:
        client.add_comment(
            repo,
            pr_number,
            "This PR is already in the merge queue. "
            "Use `/dequeue` to remove it first.",
        )
        return

    # Must have at least one approving review
    # TODO: Re-enable once deployed to the real repo
    # reviews = client.get_pr_reviews(repo, pr_number)
    # approved = any(r.get("state") == "APPROVED" for r in reviews)
    # if not approved:
    #     client.add_comment(
    #         repo,
    #         pr_number,
    #         "This PR needs at least one approving review before "
    #         "it can enter the merge queue.",
    #     )
    #     return

    # Detect queues from changed files
    changed_files = client.get_changed_files(repo, pr_number)
    queues = detect_queues(changed_files)
    if not queues:
        client.add_comment(
            repo,
            pr_number,
            "This PR does not touch any merge-queue-managed paths "
            "(`projects/hipdnn/` or `dnn-providers/*/`). "
            "It can be merged normally.",
        )
        return

    # Enqueue
    enqueue_pr(client, repo, pr_number, queues, user)

    # Build confirmation with queue positions
    lines = [f"**Merge Queue:** PR #{pr_number} has been enqueued by @{user}.\n"]
    lines.append("| Queue | Position |")
    lines.append("|-------|----------|")
    for q in queues:
        members = get_queue_members(client, repo, q)
        position = next(
            (i + 1 for i, m in enumerate(members) if m["pr_number"] == pr_number),
            len(members),
        )
        lines.append(f"| `{q}` | {position}/{len(members)} |")

    client.add_comment(repo, pr_number, "\n".join(lines))
    logger.info(f"PR #{pr_number} enqueued in {queues}")


def handle_dequeue(
    client: GitHubCLIClient, repo: str, pr_number: int, user: str
) -> None:
    # PR author or write access can dequeue
    pr_data = client.get_pr_by_number(repo, pr_number)
    is_author = pr_data and pr_data.get("user", {}).get("login") == user
    if not is_author and not _has_write_access(client, repo, user):
        client.add_comment(
            repo,
            pr_number,
            f"@{user} Only the PR author or a collaborator with write "
            "permission can use `/dequeue`.",
        )
        return

    dequeue_pr(client, repo, pr_number, f"Removed from queue by @{user} via `/dequeue`.")
    logger.info(f"PR #{pr_number} dequeued by @{user}")


if __name__ == "__main__":
    main()
