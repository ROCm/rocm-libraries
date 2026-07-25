################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

"""Characterization tests pinning CURRENT behavior of Configuration.py source findings.

These tests deliberately assert the present (buggy or dead-code) behavior of
Tensile.Configuration so that any source fix or cleanup for the slice-4 findings
is FORCED to update them. Each test names the finding it guards and the change
that will break it. The tests must stay green against the current source; when a
fix lands they should fail and be rewritten to assert the corrected behavior.

Findings (see slice-4 SOURCE-FINDINGS.md / GitHub issues):
  - Finding A/B: createBinaryOp/createUnaryOp bool() coercion is dead (opKey by
    lambda identity + guard-list typos). And/Or return raw operands, not bools.
  - Finding C: ReadWriteTransformDict.__deepcopy__ drops memo -> cyclic recursion.
  - Dead code: checkConstraints dead store + UnboundLocalError on no constraints;
    set{Read,Write}Transform redundant pop default.
"""

import copy
import sys

import pytest

from Tensile.Configuration import (
    CallableParameter,
    ProjectConfig,
    ReadWriteTransformDict,
)


class TestBinaryOpBoolCoercionIsDead:
    """Finding A/B: And/Or currently return the raw operand, not a coerced bool.

    A fix that resolves opKey by name (and corrects the Le/Ge -> LtE/GtE guard
    typos) will make these return real booleans, breaking every assert here.
    """

    def test_and_returns_operand_not_bool(self):
        binOp = CallableParameter.createBinaryOp(2, 3, "And")
        result = binOp()
        assert result == 3
        assert not isinstance(result, bool)

    def test_and_returns_falsy_operand(self):
        binOp = CallableParameter.createBinaryOp(0, 3, "And")
        result = binOp()
        assert result == 0
        assert not isinstance(result, bool)

    def test_or_returns_operand_not_bool(self):
        binOp = CallableParameter.createBinaryOp(0, 5, "Or")
        result = binOp()
        assert result == 5
        assert not isinstance(result, bool)


class TestDeepCopyDropsMemo:
    """Finding C: cyclic deep-copy recurses infinitely (RecursionError today).

    Threading memo into the recursive deepcopy will make this copy succeed and
    break the pinned RecursionError expectation.
    """

    def test_cyclic_deepcopy_raises_recursionerror(self):
        d = ReadWriteTransformDict()
        d.writeNoTransform("cycle", d)

        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            with pytest.raises(RecursionError):
                copy.deepcopy(d)
        finally:
            sys.setrecursionlimit(original_limit)


class TestCheckConstraintsNoConstraints:
    """Dead code / latent edge: checkConstraints raises when there are no constraints.

    Deleting the dead `result = True` store and returning True explicitly in the
    no-constraint case will replace this UnboundLocalError with a True return.
    """

    def test_no_constraints_raises_unboundlocalerror(self):
        cfg = ProjectConfig()
        with pytest.raises(UnboundLocalError):
            cfg.checkConstraints()


class TestSetTransformNoneRemovesKey:
    """Dead store: the redundant pop default in set{Read,Write}Transform.

    Pins the observable contract the cleanup must preserve: setting a transform
    to None removes it.
    """

    def _identity_read(self, obj, key):
        return obj.readNoTransform(key)

    def _identity_write(self, obj, key, value):
        obj.writeNoTransform(key, value)

    def test_set_read_transform_none_removes(self):
        d = ReadWriteTransformDict()
        d.setReadTransform(self._identity_read)
        assert d.hasReadTransform()
        d.setReadTransform(None)
        assert not d.hasReadTransform()

    def test_set_write_transform_none_removes(self):
        d = ReadWriteTransformDict()
        d.setWriteTransform(self._identity_write)
        assert d.hasWriteTransform()
        d.setWriteTransform(None)
        assert not d.hasWriteTransform()
