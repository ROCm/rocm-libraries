################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurface characterization: the ``isinstance(rhs, Parameter)``
guard in ``Parameter.__lt__`` (Tensile/Configuration.py:218).

Branch 766aca336236aae2b14b573c47e639e5adb3307d. The predicate is a runtime
type dispatch on the caller-supplied right-hand operand of ``<``:

  * TRUE branch  -> rhs is a Parameter (or subclass, e.g. CallableParameter);
                    comparison delegates to ``self.value < rhs.value``.
  * FALSE branch -> rhs is a plain value (int/str/...); comparison delegates to
                    ``self.value < rhs``.

These tests pin ACTUAL observed behavior; they do not assert anything aspirational.
"""

import pytest

from Tensile.Configuration import Parameter, CallableParameter

pytestmark = pytest.mark.unit


def _p(name, val):
    return Parameter(name, val)


# --- TRUE branch: rhs IS a Parameter -> compares .value vs .value ------------

def test_lt_guard_true_parameter_rhs():
    a, b = _p("a", 1), _p("b", 2)
    assert (a < b) is True
    assert (b < a) is False


def test_lt_guard_true_callableparameter_rhs_subclass():
    # CallableParameter inherits from Parameter, so isinstance(rhs, Parameter)
    # is True and the value path is taken (subclass closure).
    assert issubclass(CallableParameter, Parameter)


# --- FALSE branch: rhs is a plain scalar -> compares .value vs rhs -----------

def test_lt_guard_false_int_rhs():
    a = _p("a", 1)
    assert (a < 5) is True
    assert (a < 0) is False


def test_lt_guard_false_str_rhs():
    s = _p("s", "abc")
    assert (s < "abd") is True
    assert (s < "abb") is False
