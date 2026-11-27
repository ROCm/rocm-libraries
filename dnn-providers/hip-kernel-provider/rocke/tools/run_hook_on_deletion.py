#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run a pre-commit hook for a change that deletes files it takes and edits none.

pre-commit gives a hook only the changed files that still exist, so a change
that deletes files a hook takes, and adds or edits none, skips that hook. A
check that reads the whole tree can still fail on such a change: deleting a
test file listed in the rocKE test-file registry leaves its record stale. Run
from an always_run hook, this script runs the named hook's command when the
change deletes a path the hook takes and adds or edits none; when it adds or
edits one, pre-commit has run the hook already.

The change is the one pre-commit checks: with PRE_COMMIT_FROM_REF and
PRE_COMMIT_TO_REF set (``pre-commit run --from-ref/--to-ref``, as CI runs it,
and the pre-push stage), the diff between them, computed as pre-commit computes
it; otherwise the staged changes. A change that deletes nothing costs one
``git diff``. The named hook must be a ``repo: local`` hook that takes no file
names (``pass_filenames: false``) and runs in this script's environment. Which
paths a hook takes follows ``precommit_hooks``: by ``files`` and ``exclude``
only, not ``types``.

Usage, from the repository root (where pre-commit runs hooks)::

    python run_hook_on_deletion.py <hook id> [--config .pre-commit-config.yaml]

Exit status: the hook's, or 0 when it need not run; 2 when the hook cannot be
run this way.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from precommit_hooks import load_local_hook

EXIT_CANNOT_RUN = 2

# pre-commit's own file-list command (pre_commit.git), so this script and
# pre-commit agree on which files a change touches.
_DIFF = ("git", "diff", "--name-only", "--no-ext-diff", "-z")


def _git_paths(*arguments: str) -> list[str]:
    listed = subprocess.run(
        [*_DIFF, *arguments], check=True, capture_output=True, text=True
    )
    return [path for path in listed.stdout.split("\0") if path]


def _range_paths(from_ref: str, to_ref: str, *arguments: str) -> list[str]:
    """Paths the diff ``from_ref...to_ref`` lists; like pre-commit, a full
    ``from_ref..to_ref`` diff when the two refs have no merge base."""
    try:
        return _git_paths(*arguments, f"{from_ref}...{to_ref}")
    except subprocess.CalledProcessError:
        return _git_paths(*arguments, f"{from_ref}..{to_ref}")


def deleted_paths() -> list[str]:
    """The paths the change deletes. A rename counts as deleting its old path."""
    deleted = ("--no-renames", "--diff-filter=D")
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref and to_ref:
        return _range_paths(from_ref, to_ref, *deleted)
    return _git_paths("--staged", *deleted)


def kept_paths() -> list[str]:
    """The paths pre-commit gives hooks for the change: the changed paths that
    still exist."""
    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if from_ref and to_ref:
        changed = _range_paths(from_ref, to_ref)
    else:
        changed = _git_paths("--staged", "--diff-filter=ACMRTUXB")
    return [path for path in changed if os.path.lexists(path)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hook_id", help="the id of the hook to run")
    parser.add_argument("--config", type=Path, default=Path(".pre-commit-config.yaml"))
    args = parser.parse_args(argv)

    deleted = deleted_paths()
    if not deleted:
        return 0
    try:
        hook = load_local_hook(args.config, args.hook_id)
    except OSError as error:
        print(
            f"running a hook for deleted files failed - config={args.config} "
            f"cannot be read ({error.strerror}), expected the pre-commit config "
            "(run from the repository root)",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    except (KeyError, ValueError) as error:
        # A KeyError's str() is its message quoted.
        print(error.args[0], file=sys.stderr)
        return EXIT_CANNOT_RUN
    if hook.pass_filenames:
        print(
            f"running a hook for deleted files failed - hook={hook.hook_id!r} "
            "takes file names, expected pass_filenames: false (a change that "
            "only deletes files has none to give it)",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN
    taken = [path for path in deleted if hook.takes(path)]
    if not taken or any(hook.takes(path) for path in kept_paths()):
        return 0
    print(
        f"{hook.hook_id}: the change deletes {len(taken)} file(s) it takes and "
        f"edits none, so it runs here; first: {taken[0]}"
    )
    return subprocess.run(
        [*shlex.split(hook.entry), *hook.args], check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
