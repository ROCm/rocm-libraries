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

"""Characterization tests for the ``is*`` predicate surface of
``Tensile.Common.DataType.DataType`` (and the module-level ``_is8bitFloat``
helper it delegates to).

The predicate set is discovered by introspection so the matrix tracks the
class automatically; every no-argument ``is*`` method is evaluated against
*every* dtype in the table. One snapshot per predicate, as a
``{dtype_name: bool}`` mapping, keeps the ``.ambr`` reviewable while pinning
the full truth table.
"""

import inspect

import pytest

from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


_ENUMS = [p["enum"] for p in DataType.properties]

# Every zero-arg `is*` method on the class (all current predicates take only
# self). Sorted for deterministic parametrization / snapshot ids.
_PREDICATES = sorted(
    name
    for name, fn in inspect.getmembers(DataType, predicate=inspect.isfunction)
    if name.startswith("is")
)


def test_predicate_set_is_stable(snapshot):
    # Pin the discovered predicate roster itself, so adding/removing a method
    # surfaces as a snapshot diff rather than silently changing coverage.
    assert _PREDICATES == snapshot


@pytest.mark.parametrize("method", _PREDICATES, ids=_PREDICATES)
def test_predicate_matrix(method, snapshot):
    result = {e.name: getattr(DataType(e), method)() for e in _ENUMS}
    assert result == snapshot
