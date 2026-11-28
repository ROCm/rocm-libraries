# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keep a git hook's repository variables away from the git commands tests run.

Git exports ``GIT_DIR``, ``GIT_INDEX_FILE`` and its other repository-local
variables to every hook it runs. A test session started from a hook (a
pre-commit hook that ``git commit`` runs) inherits them, and each git command a
test runs then acts on the repository being committed to, whatever its working
directory: ``git init`` in a scratch directory re-initialises that repository
as bare. The platform tests' conftest drops the variables when the session
starts, so each git command finds its repository from its own working
directory, as it does when pytest is run from a shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess

__all__ = ["drop_inherited_git_repository_variables"]


def drop_inherited_git_repository_variables() -> list[str]:
    """Remove from ``os.environ`` each variable git names as local to a
    repository (``git rev-parse --local-env-vars``), and return the names that
    were set. Does nothing without git."""
    git = shutil.which("git")
    if git is None:
        return []
    listed = subprocess.run(
        [git, "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    )
    dropped = [name for name in listed.stdout.split() if name in os.environ]
    for name in dropped:
        del os.environ[name]
    return dropped
