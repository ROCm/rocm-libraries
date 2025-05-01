################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest

import Tensile.Component  as Component
import Tensile.Components as Components
from .test_Component import MockWriter

@pytest.fixture
def aggressive():
    return {"kernel": {'AggressivePerfMode': True}}

@pytest.fixture
def non_aggressive():
    return {'kernel': {'AggressivePerfMode': False}}

def test_aggressive(aggressive):
    writer = MockWriter(**aggressive)

    found = Component.Component.Priority.find(writer)
    assert isinstance(found, Components.Priority.AggressivePriority)

    firstRaise  = found(writer, 1, "comment")
    secondRaise = found(writer, 1, "comment")
    lower = found(writer, 0)

    assert "s_setprio 1" in firstRaise
    assert "comment" in firstRaise
    assert secondRaise == ""
    assert "s_setprio 0" in lower

def test_non_aggressive(non_aggressive):
    writer = MockWriter(**non_aggressive)

    found = Component.Component.Priority.find(writer)
    assert isinstance(found, Components.Priority.ConstantPriority)

    firstRaise  = found(writer, 1, "comment")
    secondRaise = found(writer, 1, "comment")
    lower = found(writer, 0)

    assert firstRaise == ""
    assert secondRaise == ""
    assert lower == ""
