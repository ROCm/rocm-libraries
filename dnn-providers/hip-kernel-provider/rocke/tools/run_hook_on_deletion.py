#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run pre-commit hooks for a change that deletes files they take and edits none.

pre-commit gives a hook only the changed files that still exist, so a change
that deletes files a hook takes, and adds or edits none, skips that hook. A
check that reads the whole tree can still fail on such a change: deleting a
test file listed in the rocKE test-file registry leaves its record stale, and
deleting a file listed on an allowlist leaves its line stale. Run from an
always_run hook, this script runs each named hook's command, in the order
named, when the change deletes a path that hook takes and adds or edits none;
when it adds or edits one, pre-commit has run that hook already.

The change is the one pre-commit checks: with PRE_COMMIT_FROM_REF and
PRE_COMMIT_TO_REF set (``pre-commit run --from-ref/--to-ref``, as CI runs it,
and the pre-push stage), the diff between them, computed as pre-commit computes
it; otherwise the staged changes. A change that deletes nothing costs one
``git diff``. Each named hook must be a ``repo: local`` hook that takes no file
names (``pass_filenames: false``) and runs in this script's environment; every
one is checked before any runs. Which paths a hook takes follows
``precommit_hooks``: by ``files`` and ``exclude`` only, not ``types``.

Usage, from the repository root (where pre-commit runs hooks)::

    python run_hook_on_deletion.py <hook id> [<hook id> ...] \
        [--config .pre-commit-config.yaml]

Exit status: that of the first hook that ran and failed, or 0 when every hook
that ran passed or none needed to run; 2 when a named hook cannot be run this
way.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from precommit_hooks import PrecommitHook, load_local_hook

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


def _load_runnable_hook(config: Path, hook_id: str) -> PrecommitHook:
    """The hook ``hook_id`` of ``config``, if this script can run it. Raises
    ``ValueError`` with the message to print when it cannot."""
    try:
        hook = load_local_hook(config, hook_id)
    except OSError as error:
        raise ValueError(
            f"running a hook for deleted files failed - config={config} "
            f"cannot be read ({error.strerror}), expected the pre-commit config "
            "(run from the repository root)"
        ) from error
    except (KeyError, ValueError) as error:
        # A KeyError's str() is its message quoted.
        raise ValueError(error.args[0]) from error
    if hook.pass_filenames:
        raise ValueError(
            f"running a hook for deleted files failed - hook={hook.hook_id!r} "
            "takes file names, expected pass_filenames: false (a change that "
            "only deletes files has none to give it)"
        )
    return hook


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hook_ids", nargs="+", help="the ids of the hooks to run")
    parser.add_argument("--config", type=Path, default=Path(".pre-commit-config.yaml"))
    args = parser.parse_args(argv)

    deleted = deleted_paths()
    if not deleted:
        return 0
    try:
        hooks = [_load_runnable_hook(args.config, hook_id) for hook_id in args.hook_ids]
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_CANNOT_RUN
    kept = kept_paths()
    first_failure = 0
    for hook in hooks:
        taken = [path for path in deleted if hook.takes(path)]
        if not taken or any(hook.takes(path) for path in kept):
            continue
        print(
            f"{hook.hook_id}: the change deletes {len(taken)} file(s) it takes and "
            f"edits none, so it runs here; first: {taken[0]}",
            flush=True,
        )
        status = subprocess.run(
            [*shlex.split(hook.entry), *hook.args], check=False
        ).returncode
        first_failure = first_failure or status
    return first_failure


if __name__ == "__main__":
    sys.exit(main())
