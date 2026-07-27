# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-focused assertions for ``SolutionStructs.Utilities.reject``."""

import pytest

from Tensile.SolutionStructs.Utilities import reject

pytestmark = pytest.mark.unit


def test_reject_default_prints_exact_reason_and_mutates_state(capsys):
    state = {"Valid": True}

    assert reject(state, True, "bad tile", 7) is True
    assert state["Valid"] is False
    assert capsys.readouterr().out == "\nreject: bad tile\n7\n"


def test_reject_default_print_flag_is_enabled(capsys):
    state = {"Valid": True}

    assert reject(state) is True
    assert capsys.readouterr().out == "\nreject: "


def test_reject_none_state_with_default_print_is_safe(capsys):
    assert reject(None) is None
    assert capsys.readouterr().out == "\nreject: "


def test_reject_print_with_missing_solution_index_uses_sentinel(capsys):
    state = {"Valid": True}

    assert reject(state, True) is True
    assert state["Valid"] is False
    assert capsys.readouterr().out == "\nreject: "
