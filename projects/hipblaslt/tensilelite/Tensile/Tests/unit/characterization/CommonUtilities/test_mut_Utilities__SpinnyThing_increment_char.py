################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.SpinnyThing.increment``.

Pins the exact observable behavior of ``increment``: on each call it writes
``"\\b" + self.chars[self.index]`` to stdout (the char at the pre-advance
index), flushes, and advances ``self.index`` by ``value`` modulo
``len(self.chars)`` (4). The ``value`` parameter defaults to 1, so a bare
``increment()`` steps forward by one; a non-default ``value`` steps by that
amount and is genuinely used by the index update.

Required kills:
* ``mutmut_1`` (``value=1`` -> ``value=2``): the default advance changes from one
  step to two, observable both through the signature default and through the
  index after a bare ``increment()``. Pinned by ``inspect.signature`` below and
  by the per-call index assertions.
* ``mutmut_4`` (``"\\b"`` -> ``"XX\\bXX"``): changes the written erase prefix.
  Pinned by exact ``capsys`` payload assertions.
"""

import importlib
import inspect

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def test_increment_writes_exact_backspace_prefix(capsys):
    # Fresh instance: index 0 -> chars[0] == "|". The written payload is
    # exactly "\b|": a single backspace followed by the current spinner char.
    # Kills mutmut_4 ("\b" -> "XX\bXX", which would produce "XX\bXX|").
    spinner = U.SpinnyThing()
    spinner.increment()
    out = capsys.readouterr().out
    assert out == "\b|"
    assert "X" not in out


def test_increment_writes_current_char_each_call(capsys):
    # Over a full cycle the written prefixes are the chars at the *pre-advance*
    # index: "\b|", "\b/", "\b-", "\b\\". Pins both the literal "\b" prefix and
    # the ordering of chars, reinforcing the mutmut_4 kill.
    spinner = U.SpinnyThing()
    for expected in ["\b|", "\b/", "\b-", "\b\\"]:
        spinner.increment()
        assert capsys.readouterr().out == expected


def test_increment_advances_index_forward_by_one():
    # index starts at 0, chars has length 4.
    spinner = U.SpinnyThing()
    assert spinner.index == 0
    assert len(spinner.chars) == 4

    # (0 + 1) % 4 == 1; (1 + 1) % 4 == 2.
    spinner.increment()
    assert spinner.index == 1
    spinner.increment()
    assert spinner.index == 2


def test_increment_wraps_modulo_chars_length():
    # Full forward cycle: 1, 2, 3, 0. A backward step or step-by-two would
    # produce a different sequence, and true division (%->/) would leave index
    # as a float rather than an int.
    spinner = U.SpinnyThing()
    observed = []
    for _ in range(len(spinner.chars)):
        spinner.increment()
        observed.append(spinner.index)
    assert observed == [1, 2, 3, 0]
    assert all(isinstance(i, int) for i in observed)


def test_increment_value_parameter_default_is_one():
    # The `value` parameter defaults to 1. Pinned via signature introspection.
    # Kills mutmut_1 (value=1 -> value=2).
    sig = inspect.signature(U.SpinnyThing.increment)
    assert sig.parameters["value"].default == 1


def test_increment_uses_passed_value_argument(capsys):
    # `value` is added to the index (mod len(chars)), so it genuinely drives the
    # advance: increment(99) from index 0 lands at (0 + 99) % 4 == 3, and a
    # subsequent value=0 leaves the index unchanged. The written payload is
    # always the char at the *pre-advance* index. Kills mutants that drop the
    # `+ value` term (which would advance by one regardless of the argument).
    spinner = U.SpinnyThing()
    spinner.increment(99)
    assert spinner.index == 3
    assert capsys.readouterr().out == "\b|"
    spinner.increment(value=0)
    assert spinner.index == 3
    assert capsys.readouterr().out == "\b\\"
