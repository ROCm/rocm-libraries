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
``TensileLogic.ValidMatrixInstruction._validateMatrixInstruction``.

Companion to ``test_validmatrixinstruction_char.py``. The base suite pins the
*return value* of the accept/reject paths but does not pin the diagnostic that
the reject (``except AssertionError``) branch prints. Mutant ``__mutmut_9``
replaces the formatted error message with the literal ``None``:

    print(f"Error: Validation failed: {e} ...")  ->  print(None)

The return value is ``False`` either way, so only stdout distinguishes the
original from the mutant. The test below drives the same reject path used by
the base suite and asserts the actual diagnostic text, killing the mutant.
"""

import pytest

from Tensile.TensileLogic.ValidMatrixInstruction import _validateMatrixInstruction

pytestmark = pytest.mark.unit

_FILE = "logic/asm/aquavanjaram/mi.yaml"

# Invalid MFMA 4-item MI with SolutionIndex=-1 so the validator sets
# Valid=False (instead of raising), driving the wrapper's
# ``assert solution["Valid"]`` into the caught AssertionError -> except branch.
_REJECT_SOLUTION = {
    "MatrixInstruction": [16, 16, 4, 1], "EnableMatrixInstruction": True,
    "ISA": (9, 0, 10), "ProblemType": {"DataType": "h"},
    "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": 4, "MatrixInstB": 1,
    "MatrixInstBM": 1, "MIWaveTile": [1, 1], "MIWaveGroup": [1, 1],
    "MIBlock": [16, 16, 4, 1, 1, 1], "WavefrontSize": 64,
    "SolutionIndex": -1,
}


def test_reject_branch_prints_error_diagnostic(capsys, isa_info_map):
    """Kills __mutmut_9: the except branch prints the formatted error message,
    not ``None``. On clean source stdout contains the tagged diagnostic
    (``"Error: Validation failed:"`` plus the filepath and SolutionIndex);
    the mutant prints only ``None``."""
    sol = dict(_REJECT_SOLUTION)
    ret = _validateMatrixInstruction(sol, isa_info_map, _FILE)
    out = capsys.readouterr().out

    # Return value is False on both original and mutant; pin it for context.
    assert ret is False
    # Distinguishing assertions: the original emits the formatted diagnostic.
    assert "Error: Validation failed:" in out
    assert _FILE in out
    assert "index: -1" in out
