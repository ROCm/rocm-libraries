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

"""Mutation-killing characterization test for ``_chipIdDirFromPath``.

Targets survivor ``x__chipIdDirFromPath__mutmut_12`` which changes the loop
slice from ``filepath.parts[:-1]`` (drop only the filename, walk every ancestor
directory) to ``filepath.parts[:-2]`` (additionally drop the file's immediate
parent directory). The immediate-parent directory is therefore never inspected
under the mutant.

The distinguishing input is a path whose *only* chip-ID directory is the file's
immediate parent: ``gfx950_id75a3/logic.yaml``. The original returns that
directory as a valid chip-ID dir; the mutant skips it and falls through to the
no-arch-directory result.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidChipId import _chipIdDirFromPath

pytestmark = pytest.mark.unit


def test_chip_id_dir_immediate_parent_is_inspected():
    """The chip-ID directory that is the file's immediate parent must be matched.

    Kills mutant 12 (parts[:-1] -> parts[:-2]): under the mutant the immediate
    parent ``gfx950_id75a3`` is dropped from the walk, so the function would
    fall through to ``hasChipIdDir=False``.
    """
    result = _chipIdDirFromPath("gfx950", Path("gfx950_id75a3/logic.yaml"))
    assert result.hasChipIdDir is True
    assert result.isValidFormat is True
    assert result.chipId == "id=75a3"
    assert result.dirName == "gfx950_id75a3"
