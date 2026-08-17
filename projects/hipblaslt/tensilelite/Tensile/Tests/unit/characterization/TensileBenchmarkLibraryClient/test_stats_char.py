################################################################################
# Characterization tests for Tensile.TensileBenchmarkLibraryClient — stats.
#
# ADD-ONLY. The client driver (BenchmarkProblemSize subprocess, PrintStats,
# TensileBenchmarkLibraryClient) is resistance; this pins the pure stats helpers
# and the median() py3 bug.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileBenchmarkLibraryClient")


def test_mean():
    assert M.mean([2, 4, 6]) == 4


def test_stddev():
    # sample stddev of [2,4,6]: mean 4, sumsq 8, /(n-1)=4 -> 2.0
    assert M.stddev([2, 4, 6]) == 2.0


def test_median_is_broken_in_py3():
    # LATENT BUG (pinned): median uses sortedList[len(sortedList)/2]; in py3 `/`
    # yields a float, and a float is not a valid list index -> TypeError.
    with pytest.raises(TypeError):
        M.median([3, 1, 2])
