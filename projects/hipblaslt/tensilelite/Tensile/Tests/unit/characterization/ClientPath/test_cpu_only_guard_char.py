################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Characterization tests for the --cpu-only / synthetic-CSV safety guard
(``Tensile.Tensile.warnIfCpuOnlyWithLibraryLogic``).

Under ``--cpu-only`` the benchmark step writes a synthetic results CSV, so any
LibraryLogic generated from it tunes on fake perf. The guard must make that
provenance unmistakable. These tests pin its behavior:

* fires (loud warning) only when CpuOnly is set AND a LibraryLogic step is
  present (including the ``LibraryLogic: None`` generation form);
* stays silent when CpuOnly is off, so the default path is unchanged;
* stays silent when there is no LibraryLogic step.

Assertions are on observable side-effects (return value + captured warning
text); no golden snapshots are needed. CpuOnly is set via ``monkeypatch.setitem``
so it auto-reverts and never leaks into other tests.
"""

import pytest

import Tensile.Tensile as T
from Tensile.Common.GlobalParameters import globalParameters

pytestmark = pytest.mark.unit


def _set_cpu_only(monkeypatch, value):
    monkeypatch.setitem(globalParameters, "CpuOnly", value)


class TestCpuOnlyLibraryLogicGuard:
    """warnIfCpuOnlyWithLibraryLogic — the loud synthetic-perf guard."""

    def test_fires_when_cpu_only_and_librarylogic(self, monkeypatch, capsys):
        """CpuOnly + a LibraryLogic step → guard fires with a loud warning."""
        _set_cpu_only(monkeypatch, True)
        fired = T.warnIfCpuOnlyWithLibraryLogic({"LibraryLogic": {"some": "cfg"}})
        out = capsys.readouterr().out
        assert fired is True
        assert "Tensile::WARNING" in out
        assert "SYNTHETIC" in out
        assert "--cpu-only" in out

    def test_fires_when_librarylogic_is_none(self, monkeypatch, capsys):
        """``LibraryLogic: None`` is still a generation step (key present)."""
        _set_cpu_only(monkeypatch, True)
        fired = T.warnIfCpuOnlyWithLibraryLogic({"LibraryLogic": None})
        out = capsys.readouterr().out
        assert fired is True
        assert "SYNTHETIC" in out

    def test_silent_when_cpu_only_without_librarylogic(self, monkeypatch, capsys):
        """CpuOnly but no LibraryLogic step → no warning, no fire."""
        _set_cpu_only(monkeypatch, True)
        fired = T.warnIfCpuOnlyWithLibraryLogic({"BenchmarkProblems": []})
        out = capsys.readouterr().out
        assert fired is False
        assert "Tensile::WARNING" not in out

    def test_silent_when_not_cpu_only(self, monkeypatch, capsys):
        """Default (non-cpu-only) path is unchanged: guard never fires."""
        _set_cpu_only(monkeypatch, False)
        fired = T.warnIfCpuOnlyWithLibraryLogic({"LibraryLogic": {"some": "cfg"}})
        out = capsys.readouterr().out
        assert fired is False
        assert "Tensile::WARNING" not in out

    def test_silent_when_cpu_only_key_absent(self, monkeypatch, capsys):
        """A missing CpuOnly key defaults to off (no fire), like a real run."""
        monkeypatch.delitem(globalParameters, "CpuOnly", raising=False)
        fired = T.warnIfCpuOnlyWithLibraryLogic({"LibraryLogic": {"some": "cfg"}})
        out = capsys.readouterr().out
        assert fired is False
        assert "Tensile::WARNING" not in out
