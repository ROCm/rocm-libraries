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

"""Characterization tests for ``TensileLogic.ValidWorkGroup._validateWorkGroup``.

``_validateWorkGroup`` is a thin CLI wrapper around
``Validators.WorkGroup.validateWorkGroup``: it runs the validator, then
asserts ``solution["Valid"]`` and returns ``True``; on any ``AssertionError``
it prints a ``filepath``/``SolutionIndex``-tagged message and returns ``False``.

Three behaviours are pinned (snapshot ``{returned, valid}``):

* **accept** — valid WorkGroup with ``Valid=True`` -> ``True``.
* **reject via validator** — an out-of-table WorkGroup makes
  ``validateWorkGroup`` assert -> caught -> ``False``.
* **reject via Valid flag** — a table-valid WorkGroup but ``Valid=False``
  trips the wrapper's own ``assert solution["Valid"]`` -> ``False``.

``Valid`` must be pre-set: ``validateWorkGroup`` never writes it, so a
solution lacking the key would ``KeyError`` (that is the contract — logic-file
solutions always carry ``Valid``). A synthetic, stable ``filepath`` keeps the
(incidental) printed message deterministic; only ``{returned, valid}`` is
snapshotted.
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidWorkGroup import _validateWorkGroup

pytestmark = pytest.mark.unit

_FILE = Path("logic/asm/aquavanjaram/wg.yaml")


def _result(solution):
    ret = _validateWorkGroup(solution, _FILE)
    return {"returned": ret, "valid": solution.get("Valid", "unset")}


def test_accept_valid_workgroup(snapshot):
    sol = {"WorkGroup": [16, 16, 1], "Valid": True, "SolutionIndex": 0}
    assert _result(sol) == snapshot


def test_reject_invalid_workgroup_value(snapshot):
    # [7, 7, 7] is not in makeValidWorkGroups() -> validateWorkGroup asserts.
    sol = {"WorkGroup": [7, 7, 7], "Valid": True, "SolutionIndex": 1}
    assert _result(sol) == snapshot


def test_reject_when_valid_flag_false(snapshot):
    # Table-valid WorkGroup, but the solution is already marked invalid ->
    # the wrapper's own `assert solution["Valid"]` fires.
    sol = {"WorkGroup": [16, 16, 1], "Valid": False, "SolutionIndex": 2}
    assert _result(sol) == snapshot
