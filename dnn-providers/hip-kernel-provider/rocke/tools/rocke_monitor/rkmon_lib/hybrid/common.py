# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Small helpers the hybrid stages share: read-only git on the mirror and the test-path rule."""

import os
import subprocess


def git_env() -> dict:
    """The caller's environment without GIT_* location variables, lazy fetches and prompts off."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git(repo: str, *args, timeout: int = 120) -> tuple[int, str]:
    """(returncode, stdout) of a read-only git command on `repo`; never fetches."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            env=git_env(),
            timeout=timeout,
            errors="replace",
        )
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def is_test_path(p: str) -> bool:
    b = p.rsplit("/", 1)[-1]
    return (
        "/tests/" in p
        or b.startswith("test_")
        or b.endswith("_test.py")
        or "/test/" in p
    )
