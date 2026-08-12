# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-focused assertions for ``SolutionStructs.Utilities`` helpers."""

import pytest

from Tensile.SolutionStructs.Utilities import isSubtileIterateMode, reject

pytestmark = pytest.mark.unit


class _ElementType:
    def __init__(self, num_bytes):
        self._num_bytes = num_bytes

    def numBytes(self):
        return self._num_bytes


def test_subtile_iterate_mode_requires_subtile_and_tdm_gates():
    assert isSubtileIterateMode({}, "A") is False
    assert isSubtileIterateMode({"UseSubtileImpl": True}, "A") is False
    assert isSubtileIterateMode(
        {
            "enableTDMA": True,
            "DepthU": 129,
            "ProblemType": {"DataTypeA": _ElementType(8)},
        },
        "A",
    ) is False


@pytest.mark.parametrize("tc", ["A", "B"])
def test_subtile_iterate_mode_uses_strict_pad_interval_boundary(tc):
    state = {
        "UseSubtileImpl": True,
        f"enableTDM{tc}": True,
        "DepthU": 128,
        "ProblemType": {f"DataType{tc}": _ElementType(8)},
    }

    assert isSubtileIterateMode(state, tc) is False

    state["DepthU"] = 129
    assert isSubtileIterateMode(state, tc) is True


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


def test_reject_missing_solution_index_does_not_raise(capsys):
    state = {"Valid": True}

    assert reject(state, True) is True
    assert state["Valid"] is False
    assert capsys.readouterr().out == "\nreject: "
