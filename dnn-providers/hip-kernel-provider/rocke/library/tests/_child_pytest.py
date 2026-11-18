# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Give a child pytest, started on a throwaway suite, a rootdir of its own."""

from __future__ import annotations

from pathlib import Path

__all__ = ["pin_child_rootdir"]


def pin_child_rootdir(suite: Path) -> Path:
    """Make ``suite`` the rootdir of a pytest started on a file inside it.

    Without a config file in or above the suite, pytest takes the common
    ancestor of its arguments as the rootdir and then looks upward for one:
    a suite under ``/tmp/pytest-of-<user>/...`` finds none, and the child
    collects ``.`` from a directory that holds other users' files. Any entry it
    cannot read (``/tmp/latest``, an editor's auth socket) raises
    ``PermissionError``. An empty ``pytest.ini`` stops the search at ``suite``.
    """
    (suite / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return suite
