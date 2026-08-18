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
################################################################################

"""Shared fixture for the Naming characterization suite: a factory that builds a
solution-state dict complete enough to drive ``_getName`` /
``getKeyNoInternalArgs`` (a real ``ProblemType`` plus the ``GlobalSplitU`` /
internal-args / tile keys those functions read), overridable per test.
"""

import pytest

from Tensile.SolutionStructs.Problem import ProblemType


def _base_state():
    pt = ProblemType({"DataType": 0}, False)
    return {
        "ProblemType": pt,
        "GlobalSplitU": 1,
        "UseCustomMainLoopSchedule": False,
        # Internal args — required to be present for getKeyNoInternalArgs (it
        # backs them up and masks them).
        "WorkGroupMapping": 1,
        "WorkGroupMappingXCC": 1,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": 0,
        "StaggerUStride": 0,
        "StaggerUMapping": 0,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "SFCWGM": [],
        # Read by _getName under the Full required-parameter set (L194).
        "SpaceFillingAlgo": [],
        # Common naming params.
        "MacroTile0": 128,
        "MacroTile1": 128,
        "DepthU": 16,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MIWaveTile": [2, 2],
    }


@pytest.fixture
def make_state():
    def _make(**overrides):
        s = _base_state()
        s.update(overrides)
        return s

    return _make
