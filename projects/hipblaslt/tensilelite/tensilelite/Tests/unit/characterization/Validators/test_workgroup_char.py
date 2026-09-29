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

"""Characterization tests for ``Validators.WorkGroup.validateWorkGroup``.

The function is assertion-guarded: it asserts the solution has a "WorkGroup"
key and that the value is in ``makeValidWorkGroups()``, returning True on
success and raising ``AssertionError`` otherwise. We pin both behaviours:
the accept path (snapshot the True return) and each reject path
(``pytest.raises(AssertionError)``).
"""

import pytest

from Tensile.SolutionStructs.Validators.WorkGroup import validateWorkGroup

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name,wg", [
    ("wg_16_16_1", [16, 16, 1]),
    ("wg_16_16_2", [16, 16, 2]),
    ("wg_1_32_1",  [1, 32, 1]),
])
def test_valid_workgroup_accepted(name, wg, snapshot):
    assert validateWorkGroup({"WorkGroup": wg}) == snapshot(name=name)


def test_invalid_workgroup_value_rejected():
    # Value not in makeValidWorkGroups() -> AssertionError.
    with pytest.raises(AssertionError):
        validateWorkGroup({"WorkGroup": [7, 7, 7]})


def test_missing_workgroup_key_rejected():
    # No "WorkGroup" key -> the first assertion fails.
    with pytest.raises(AssertionError):
        validateWorkGroup({})
