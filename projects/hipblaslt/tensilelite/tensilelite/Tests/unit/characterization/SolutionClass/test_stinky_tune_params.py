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
# SPDX-License-Identifier: MIT

"""Tests for the tuning-only ``StinkyTofuParameters`` solution key: it is moved
to the internal ``_StinkyTofuParameters``, tags the kernel name only when set,
and is dropped before the logic yaml is written."""

import sys

import pytest

from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

KEY = "_StinkyTofuParameters"


@pytest.fixture
def tuned(solution):
    """Yield the shared solution with the internal key controlled by the test."""
    solution.dropStinkyTofuParameters()
    yield solution
    solution.dropStinkyTofuParameters()


def _names(sol):
    return getSolutionNameMin(sol, False), getKernelNameMin(sol, False)


def test_unset_leaves_name_unchanged(tuned):
    assert KEY not in tuned
    assert "_STP" not in _names(tuned)[0]
    assert "_STP" not in _names(tuned)[1]


def test_set_tags_name_and_drop_restores_it(tuned):
    base = _names(tuned)
    tuned[KEY] = {"DsReadPerCap": 4}
    tagged = _names(tuned)
    assert all("_STP_DsReadPerCap4" in n for n in tagged)
    assert tagged != base
    tuned.dropStinkyTofuParameters()
    assert KEY not in tuned
    assert _names(tuned) == base


def test_tag_is_human_readable_and_filename_safe(tuned):
    tuned[KEY] = {"WmmaBatchSize": 2, "DsReadPerCap": -1, "Scale": 0.5}
    for n in _names(tuned):
        assert "_STP_DsReadPerCapn1_Scale0p5_WmmaBatchSize2" in n
        assert "/" not in n and " " not in n


def test_tag_depends_on_values_not_order(tuned):
    tuned[KEY] = {"DsReadPerCap": 4, "WmmaBatchSize": 2}
    a = _names(tuned)
    tuned[KEY] = {"WmmaBatchSize": 2, "DsReadPerCap": 4}
    assert _names(tuned) == a
    tuned[KEY] = {"DsReadPerCap": 3, "WmmaBatchSize": 2}
    assert _names(tuned) != a


def test_constructor_moves_public_key_to_internal(solution, assembler, isa_info_map):
    config = dict(solution)
    config["StinkyTofuParameters"] = {"DsReadPerCap": 4}
    s = Solution(config, False, False, False, assembler, isa_info_map)
    assert "StinkyTofuParameters" not in s
    assert s[KEY] == {"DsReadPerCap": 4}


def test_constructor_unset_adds_no_key(solution, assembler, isa_info_map):
    config = dict(solution)
    config.pop(KEY, None)
    config.pop("StinkyTofuParameters", None)
    s = Solution(config, False, False, False, assembler, isa_info_map)
    assert KEY not in s
    assert "StinkyTofuParameters" not in s


def test_info_notice_announced_once_per_distinct_value(solution, assembler, isa_info_map, capsys):
    S = sys.modules["Tensile.SolutionStructs.Solution"]  # the package re-exports the class
    from Tensile.Common import setVerbosity

    setVerbosity(1)
    config = dict(solution)
    config["StinkyTofuParameters"] = {"WmmaBatchSize": 7}
    for _ in range(2):
        Solution(dict(config), False, False, False, assembler, isa_info_map)
    assert capsys.readouterr().out.count("StinkyTofuParameters {'WmmaBatchSize': 7}") == 1
    assert (("WmmaBatchSize", 7),) in S._stinkyTuneAnnounced
