#!/usr/bin/env python3

"""
Merge Queue Processor
---------------------
Scheduled processor that advances the merge queue each cycle.

For each queue, finds the head PR (oldest ``enqueued_at``).  A PR is
only processed when it is at the front of **all** queues it belongs to.

Processing steps for an eligible PR:
1. Transition from ``mq:queued`` → ``mq:active``
2. Update PR branch (merge develop into it)
3. Wait for CI on the next cycle
4. On CI success → squash-merge
5. On CI failure → eject from all queues

Environment variables (set by the workflow):
    GH_TOKEN – GitHub API token
    REPO     – owner/repo
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from github_cli_client import GitHubCLIClient
from merge_queue import (
    check_ci_status,
    dequeue_pr,
    get_queue_members,
    is_at_front_of_all_queues,
    merge_pr,
    update_pr_branch,
    update_status_comment,
)
from merge_queue_config import ALL_QUEUES, LABEL_ACTIVE, LABEL_QUEUED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    client = GitHubCLIClient()
    repo = os.environ["REPO"]

    # Collect every unique queued PR across all queues so we can update
    # status comments for *all* of them, not just the heads.
    all_queued: dict[int, dict] = {}  # pr_number -> member dict
    for queue in ALL_QUEUES:
        for member in get_queue_members(client, repo, queue):
            if member["pr_number"] not in all_queued and member["queues"]:
                all_queued[member["pr_number"]] = member

    # Update status comments for every queued PR
    for pr_number, member in all_queued.items():
        pr_queues = member["queues"]
        _, blocking = is_at_front_of_all_queues(
            client, repo, pr_number, pr_queues
        )
        update_status_comment(client, repo, pr_number, pr_queues, blocking)

    # Now process heads: only the PR at the front of ALL its queues advances.
    processed: set[int] = set()

    for queue in ALL_QUEUES:
        members = get_queue_members(client, repo, queue)
        if not members:
            continue

        head = members[0]
        pr_number = head["pr_number"]

        if pr_number in processed:
            continue
        processed.add(pr_number)

        pr_queues = head["queues"]
        if not pr_queues:
            logger.warning(f"PR #{pr_number} has no queue metadata, skipping")
            continue

        is_ready, blocking = is_at_front_of_all_queues(
            client, repo, pr_number, pr_queues
        )

        if not is_ready:
            logger.info(
                f"PR #{pr_number} blocked by queues: {blocking}"
            )
            continue

        # PR is at the front of all its queues — process it
        labels = client.get_existing_labels_on_pr(repo, pr_number)
        is_active = LABEL_ACTIVE in labels

        if not is_active:
            # First time at front: activate and update branch
            logger.info(f"Activating PR #{pr_number}: updating branch")
            client.add_labels(repo, pr_number, [LABEL_ACTIVE])
            if LABEL_QUEUED in labels:
                client.remove_label(repo, pr_number, LABEL_QUEUED)

            success = update_pr_branch(client, repo, pr_number)
            if not success:
                dequeue_pr(
                    client,
                    repo,
                    pr_number,
                    "Merge conflict with the base branch. "
                    "Please rebase and re-enqueue with `/merge`.",
                )
            # Either way, wait for next cycle (CI needs to run on the
            # updated branch, or the PR was ejected).
            continue

        # Already active — check CI
        ci = check_ci_status(client, repo, pr_number)
        logger.info(f"PR #{pr_number} CI status: {ci}")

        if ci == "pending":
            continue

        if ci == "failure":
            dequeue_pr(
                client,
                repo,
                pr_number,
                "CI checks failed while at the front of the merge queue. "
                "Please fix the failures and re-enqueue with `/merge`.",
            )
            continue

        if ci == "success":
            logger.info(f"Merging PR #{pr_number}")
            success = merge_pr(client, repo, pr_number)
            if success:
                client.add_comment(
                    repo,
                    pr_number,
                    "**Merge Queue:** PR merged successfully.",
                )
            else:
                dequeue_pr(
                    client,
                    repo,
                    pr_number,
                    "Merge failed unexpectedly. "
                    "Please check for conflicts and re-enqueue with `/merge`.",
                )


if __name__ == "__main__":
    main()
