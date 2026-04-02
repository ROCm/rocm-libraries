#!/usr/bin/env python3

"""
Merge Queue Configuration
-------------------------
Path-to-queue mappings and constants for the hipDNN/provider merge queue system.

Queue model:
- Each provider has its own independent FIFO queue
- hipDNN core changes enter ALL queues (core can break providers)
- Provider changes only enter their own queue
- A PR merges only when at the front of every queue it belongs to
"""

# Path prefix -> list of queues the PR enters.
# Order matters for matching: first prefix match wins per file.
PATH_TO_QUEUES = {
    "projects/hipdnn/": [
        "hipdnn",
        "miopen-provider",
        "hipblaslt-provider",
        "hip-kernel-provider",
        "fusilli-provider",
        "integration-tests",
    ],
    "dnn-providers/miopen-provider/": ["miopen-provider"],
    "dnn-providers/hipblaslt-provider/": ["hipblaslt-provider"],
    "dnn-providers/hip-kernel-provider/": ["hip-kernel-provider"],
    "dnn-providers/fusilli-provider/": ["fusilli-provider"],
    "dnn-providers/integration-tests/": ["integration-tests"],
}

ALL_QUEUES = sorted(
    set(queue for queues in PATH_TO_QUEUES.values() for queue in queues)
)

LABEL_PREFIX = "mq:"
LABEL_QUEUED = f"{LABEL_PREFIX}queued"
LABEL_ACTIVE = f"{LABEL_PREFIX}active"

METADATA_COMMENT_MARKER = "<!-- merge-queue-metadata"

# The base branch that queued PRs target.
TARGET_BRANCH = "develop"

# Merge method used when the queue merges a PR.
MERGE_METHOD = "squash"
