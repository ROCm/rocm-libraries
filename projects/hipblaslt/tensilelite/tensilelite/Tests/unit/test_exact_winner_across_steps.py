# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An exact size benchmarked in several steps keeps its fastest kernel.

Each step's CSV is read in turn. The stored winner score is the performance
metric (GFlops per MHz under UseEffLike), so a later step must beat it in that
unit, not by comparing its raw GFlops against it.
"""

import pytest

from tensilelite import LibraryLogic
from tensilelite.Common.GlobalParameters import globalParameters
from tensilelite.LibraryLogic import LogicAnalyzer

pytestmark = pytest.mark.unit

_SIZE = (32, 64, 1, 128, 32, 32, 32, 128)
_HEADER = "GFlops, SizeI, SizeJ, SizeK, SizeL, LDD, LDC, LDA, LDB, TotalFlops, GbpsBW, Kernel"


def _writeStep(path, gflops):
    row = ", ".join(str(v) for v in (gflops,) + _SIZE + (1, 1, gflops))
    path.write_text(_HEADER + "\n" + row + "\n")
    return str(path)


def _analyzer():
    analyzer = LogicAnalyzer.__new__(LogicAnalyzer)
    analyzer.numIndices = len(_SIZE)
    analyzer.exactProblemSizes = {_SIZE}
    analyzer.rangeProblemSizes = set()
    analyzer.exactWinners = {}
    return analyzer


@pytest.mark.parametrize("useEffLike", [True, False])
def test_faster_earlier_step_keeps_the_size(tmp_path, monkeypatch, useEffLike):
    monkeypatch.setitem(globalParameters, "UseEffLike", useEffLike)
    monkeypatch.setattr(LibraryLogic, "read_max_freq", lambda: 2000.0)
    analyzer = _analyzer()

    analyzer.addFromCSV(_writeStep(tmp_path / "step0.csv", 3000.0), 1, {0: 0})
    analyzer.addFromCSV(_writeStep(tmp_path / "step1.csv", 1000.0), 1, {0: 1})

    assert analyzer.exactWinners[_SIZE][0] == 0


def test_faster_later_step_takes_the_size(tmp_path, monkeypatch):
    monkeypatch.setitem(globalParameters, "UseEffLike", True)
    monkeypatch.setattr(LibraryLogic, "read_max_freq", lambda: 2000.0)
    analyzer = _analyzer()

    analyzer.addFromCSV(_writeStep(tmp_path / "step0.csv", 1000.0), 1, {0: 0})
    analyzer.addFromCSV(_writeStep(tmp_path / "step1.csv", 3000.0), 1, {0: 1})

    assert analyzer.exactWinners[_SIZE][0] == 1
    assert analyzer.exactWinners[_SIZE][1] == pytest.approx(1.5)
