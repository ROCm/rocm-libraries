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

"""Mutation-targeted characterization tests for
``TensileLogic.ValidWorkGroup._validateWorkGroup``.

Targets survivor ``x__validateWorkGroup__mutmut_6``, which replaces the
diagnostic ``print(f"Error: Validation failed: ...")`` argument in the
``except AssertionError`` branch with ``print(None)``. The return value
(``False``) is identical between original and mutant, so only the printed
text distinguishes them. This test pins the ACTUAL printed message on the
rejection path, so it passes on clean source and fails once the mutant
prints ``None`` instead of the formatted diagnostic.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidWorkGroup import _validateWorkGroup

pytestmark = pytest.mark.unit

_FILE = Path("logic/asm/aquavanjaram/wg.yaml")


def test_rejection_prints_formatted_diagnostic(capsys):
    # [7, 7, 7] is out-of-table -> validateWorkGroup asserts -> except branch.
    sol = {"WorkGroup": [7, 7, 7], "Valid": True, "SolutionIndex": 1}

    ret = _validateWorkGroup(sol, _FILE)

    # Return value contract (unchanged by the mutant, pinned for context).
    assert ret is False

    out = capsys.readouterr().out
    # Original prints a formatted diagnostic naming the file and index; the
    # mutant prints the literal "None". Pin the real message content.
    assert out.strip() != "None"
    assert "Error: Validation failed:" in out
    assert str(_FILE) in out
    assert "index: 1" in out
