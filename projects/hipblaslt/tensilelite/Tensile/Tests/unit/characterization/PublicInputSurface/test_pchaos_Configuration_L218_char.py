################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

import pytest
from Tensile.Configuration import CallableParameter, Parameter

pytestmark = pytest.mark.unit


def _p(name, val):
    return Parameter(name, val)


def test_lt_true_parameter_rhs_lt():
    a, b = _p("a", 1), _p("b", 2)
    assert (a < b) is True


def test_lt_true_parameter_rhs_not_lt():
    a, b = _p("a", 1), _p("b", 2)
    assert (b < a) is False


def test_lt_true_callableparameter_is_subclass():
    assert issubclass(CallableParameter, Parameter)


def test_lt_true_callableparameter_rhs_lt():
    a = _p("a", 1)
    cp = CallableParameter("cp", lambda self: 5)
    assert isinstance(cp, Parameter)
    assert (a < cp) is True


def test_lt_false_int_rhs_lt():
    a = _p("a", 1)
    assert isinstance(5, Parameter) is False
    assert (a < 5) is True


def test_lt_false_int_rhs_not_lt():
    a = _p("a", 1)
    assert (a < 0) is False


def test_lt_false_str_rhs_lt():
    s = _p("s", "abc")
    assert isinstance("abd", Parameter) is False
    assert (s < "abd") is True


def test_lt_false_str_rhs_not_lt():
    s = _p("s", "abc")
    assert (s < "abb") is False
