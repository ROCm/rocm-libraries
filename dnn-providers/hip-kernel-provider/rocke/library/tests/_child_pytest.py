# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Give a child pytest, started on a throwaway suite, a rootdir of its own."""

from __future__ import annotations

from pathlib import Path

__all__ = ["pin_child_rootdir"]


def pin_child_rootdir(suite: Path) -> Path:
    """Make ``suite`` the rootdir of a pytest started on a file inside it.

    pytest looks for a config file from the suite upward and takes the first
    one's folder as the rootdir. Under ``/tmp/pytest-of-<user>/...`` that search
    can reach another user's ``/tmp/pyproject.toml`` (or ``pytest.ini``,
    ``tox.ini``, ``setup.cfg``), so the child runs with ``/tmp`` as its rootdir
    and collecting there raises ``PermissionError`` on entries it cannot read.
    A one-line ``pytest.ini`` in ``suite`` stops the search there.
    """
    (suite / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return suite
