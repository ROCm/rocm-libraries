# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What every rocKE pytest root installs in its test session.

A pytest root is a test tree whose ``conftest.py`` a CI entry loads as the root
of its run: the trees of ``pending_lanes.QUARANTINE_TREES`` (``platform/tests``,
``library/tests`` and the tiling tests inside the ``rocke`` package). pytest
loads only the conftest files on a collected path's own ancestor chain, so a
tree run on its own (the tiling entries collect ``rocke/tests/helpers/tiling``)
gets none of another root's hooks. Each root therefore calls
:func:`configure_pytest_root` from its ``pytest_configure``, with its own
directory and that directory's path relative to the rocke directory::

    def pytest_configure(config):
        configure_pytest_root(config, Path(__file__).resolve().parent, "library/tests")
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .ctest_shard import register_shard_cost_marker
from .ctest_workers import register_run_alone_marker
from .quarantine import register_quarantine
from .session_gc import use_test_session_gc
from .sweep_guard import register_sweep_guard

__all__ = ["configure_pytest_root"]


def configure_pytest_root(
    config: pytest.Config, tree_root: Path, tree_path: str
) -> None:
    """Apply the test session's garbage collector setting
    (``rocke.tests.session_gc``), declare the shard-cost and run-alone marks for
    runs that do not load the ``rocke.tests.ctest_shard`` and
    ``rocke.tests.ctest_workers`` plugins, fail a sweep whose subtests all
    skipped (``rocke.tests.sweep_guard``) and skip the quarantined tests under
    ``tree_root`` (``rocke.tests.quarantine``); ``tree_path`` is ``tree_root``
    relative to the rocke directory, one of ``pending_lanes.QUARANTINE_TREES``."""
    use_test_session_gc()
    register_shard_cost_marker(config)
    register_run_alone_marker(config)
    register_sweep_guard(config)
    register_quarantine(config, tree_root, tree_path)
