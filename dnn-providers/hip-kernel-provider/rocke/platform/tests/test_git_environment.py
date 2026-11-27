# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A session that inherits a git hook's repository variables drops them, so a
``git init`` in a scratch directory leaves the hook's repository as it was."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rocke.assets import platform_root

GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="needs git on PATH")

# Runs in a child process given a hook's environment: drops the variables,
# initialises a scratch repository, and reports what it dropped and what is left.
CHILD = """
import json, os, subprocess, sys
from rocke.tests.git_environment import drop_inherited_git_repository_variables
dropped = drop_inherited_git_repository_variables()
subprocess.run(["git", "init", "-q", sys.argv[1]], check=True)
print(json.dumps({"dropped": sorted(dropped),
                  "author": os.environ.get("GIT_AUTHOR_NAME")}))
"""


def _isolated_environment() -> dict[str, str]:
    """This process's environment without a developer's git configuration."""
    return {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def test_a_hook_s_repository_variables_are_dropped_and_its_repository_kept(
    tmp_path: Path,
):
    committed_to = tmp_path / "committed_to"
    environment = _isolated_environment()
    subprocess.run([GIT, "init", "-q", str(committed_to)], check=True, env=environment)
    git_dir = committed_to / ".git"
    # What git exports to a hook that `git commit` runs in committed_to.
    environment.update(
        {
            "GIT_DIR": str(git_dir),
            "GIT_INDEX_FILE": str(git_dir / "index"),
            "GIT_AUTHOR_NAME": "rocke",
            "PYTHONPATH": str(platform_root() / "python"),
        }
    )
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path / "scratch")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(child.stdout)
    assert report == {"dropped": ["GIT_DIR", "GIT_INDEX_FILE"], "author": "rocke"}
    assert (tmp_path / "scratch" / ".git").is_dir()
    bare = subprocess.run(
        [GIT, "-C", str(committed_to), "config", "--get", "core.bare"],
        check=True,
        capture_output=True,
        text=True,
        env=_isolated_environment(),
    )
    assert bare.stdout.strip() == "false"
