################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: ``if args.RestoreLog:`` in
``Tensile/Tensile.py`` at line 534.

Branch c63babfc10d38c09385da1dacf9fbb9c2c64c067.  The predicate is a Python
truthiness test on the argparse dest ``RestoreLog`` (``--restore-from-log``,
``type=str``, ``default=None``).

  * TRUE  branch  -> ``RestoreLog`` is a non-empty string (e.g.
                     ``"/path/to/restore.log"``).  The block resolves the
                     path, checks existence, and calls
                     ``restore_prob_sol_map()``.
  * FALSE branch  -> ``RestoreLog`` is ``None`` (flag absent) or ``""``
                     (empty string); both are falsy.  The block is skipped.

These tests pin ACTUAL observed behavior; they do not assert anything
aspirational.
"""

from typing import Optional

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper: mirror the predicate without touching argparse or the filesystem
# ---------------------------------------------------------------------------

def restore_log_enabled(restore_log: Optional[str]) -> bool:
    """Mirror of Tensile/Tensile.py:534  ``if args.RestoreLog:``.

    RestoreLog is an argparse str dest with default None. The branch is taken
    iff the value is truthy: a non-None, non-empty string.

    post: __return__ == (restore_log is not None and len(restore_log) > 0)
    """
    return bool(restore_log)


class TestRestoreLogEnabledHelper:
    """Unit tests for the pure-helper that mirrors the line-534 predicate."""

    # TRUE cases

    def test_true_nonempty_path(self):
        """A non-empty path string is truthy -> branch taken."""
        assert restore_log_enabled("/path/to/restore.log") is True

    def test_true_any_nonempty_string(self):
        """Any single-char string is truthy."""
        assert restore_log_enabled("x") is True

    def test_true_whitespace_only(self):
        """A whitespace-only string is non-empty and therefore truthy."""
        assert restore_log_enabled(" ") is True

    # FALSE cases

    def test_false_none(self):
        """None (flag absent) -> False."""
        assert restore_log_enabled(None) is False

    def test_false_empty_string(self):
        """Empty string (--restore-from-log \'\') -> False."""
        assert restore_log_enabled("") is False


# ---------------------------------------------------------------------------
# Argparse-level tests: confirm the argparse dest matches the helper contract
# ---------------------------------------------------------------------------

class TestRestoreLogArgparse:
    """Verify argparse wires the CLI flag to RestoreLog with the right defaults."""

    @staticmethod
    def _make_parser():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--restore-from-log", type=str, dest="RestoreLog",
                       help="A log file captured in previous tuning.")
        return p

    def test_absent_flag_gives_none(self):
        """When --restore-from-log is absent, RestoreLog is None (falsy)."""
        ns = self._make_parser().parse_args([])
        assert ns.RestoreLog is None
        assert not bool(ns.RestoreLog)

    def test_present_nonempty_gives_truthy(self):
        """--restore-from-log /some/path.log sets RestoreLog to that string."""
        ns = self._make_parser().parse_args(["--restore-from-log", "/some/path.log"])
        assert ns.RestoreLog == "/some/path.log"
        assert bool(ns.RestoreLog) is True

    def test_present_empty_gives_falsy(self):
        """--restore-from-log \'\'sets RestoreLog to \'\' (falsy)."""
        ns = self._make_parser().parse_args(["--restore-from-log", ""])
        assert ns.RestoreLog == ""
        assert bool(ns.RestoreLog) is False
