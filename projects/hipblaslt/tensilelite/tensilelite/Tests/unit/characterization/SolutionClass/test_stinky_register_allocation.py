# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``StinkyTofuRegisterAllocation`` takes a mode or a StinkyTofu config string.
StinkyTofu parses it, a solution keeps its shortest spelling, and the kernel
name records only the mode, off the default."""

import pytest

from Tensile.Common.ValidParameters import checkParametersAreValid, validParameters
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

KEY = "StinkyTofuRegisterAllocation"


@pytest.fixture
def make(solution, assembler, isa_info_map):
    def build(value):
        config = dict(solution)
        config[KEY] = value
        return Solution(config, False, False, False, assembler, isa_info_map)
    return build


@pytest.mark.parametrize("value,canonical", [
    (0, 0),
    (2, 2),
    ("apply", 2),
    ("2,classes=s", "2,classes=s"),
    ("mode=apply,allocator=greedy-compact,unbankable=auto", "2,allocator=greedy-compact"),
    ("off,classes=vs", 0),
    ("apply,classes=vs", 2),
    ("classes=vs,2", 2),
    ("force,unbankable=hold,classes=vs", "3,unbankable=hold"),
])
def test_a_solution_keeps_the_shortest_spelling(make, value, canonical):
    assert make(value)[KEY] == canonical


def test_the_default_is_left_out_of_the_name(make):
    name = getKernelNameMin(make(0), False)
    assert "STRA" not in name
    assert getKernelNameMin(make("off,classes=vs"), False) == name


def test_any_config_is_named_by_its_mode_alone(make):
    names = {getKernelNameMin(make(value), False)
             for value in (2, "apply", "2,classes=s", "apply,classes=vs,unbankable=hold")}
    assert len(names) == 1
    (name,) = names
    assert "_STRA2_" in name or name.endswith("_STRA2")


@pytest.mark.parametrize("value", [
    0, 3, "shadow", "apply,classes=vs,allocator=greedy-compact-program+greedy-compact,quadrant=prefer",
])
def test_validation_accepts_modes_and_config_strings(value):
    checkParametersAreValid((KEY, [value]), validParameters)


@pytest.mark.parametrize("value,reason", [
    (True, "mode 0..3 or a string"),
    (7, "not one of 0..3"),
    (2.0, "mode 0..3 or a string"),
    ("bogus", "unknown register allocation setting 'bogus'"),
    ("apply,report", "stinkytofu-opt setting"),
    ("apply,allocator=nope", "allocator 'nope' is not registered"),
    ("unbankable=freeze", "hold, allocate or auto"),
])
def test_validation_refuses_what_stinkytofu_cannot_run(value, reason):
    with pytest.raises(Exception, match=KEY) as raised:
        checkParametersAreValid((KEY, [value]), validParameters)
    assert reason in str(raised.value)
