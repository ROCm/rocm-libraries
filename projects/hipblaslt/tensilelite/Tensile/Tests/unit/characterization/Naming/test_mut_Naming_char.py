# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-focused assertions for the canonical solution naming helpers."""

from copy import deepcopy

import pytest

from Tensile.Common.Constants import MAX_FILENAME_LENGTH
from Tensile.SolutionStructs.Problem import ProblemType
import Tensile.SolutionStructs.Naming as N

pytestmark = pytest.mark.unit


class _StateWrapper:
    def __init__(self, state):
        self._state = state


def test_key_no_internal_args_uses_wrapped_state_and_restores_every_field(make_state):
    state = make_state(GlobalSplitU=2, WorkGroupMappingXCC=8, WorkGroupMapping=7)
    before = deepcopy(state)
    raw_state = deepcopy(state)

    wrapped_key = N.getKeyNoInternalArgs(_StateWrapper(state), splitGSU=True)
    raw_key = N.getKeyNoInternalArgs(raw_state, splitGSU=True)

    assert wrapped_key == raw_key
    assert state == before


def test_key_no_internal_args_masks_raw_grouped_gemm_config(make_state):
    grouped = make_state(ProblemType={"DataType": 0, "GroupedGemm": True})
    plain = make_state(ProblemType={"DataType": 0, "GroupedGemm": False})

    assert N.getKeyNoInternalArgs(grouped, splitGSU=False) == N.getKeyNoInternalArgs(
        plain, splitGSU=False
    )
    assert grouped["ProblemType"]["GroupedGemm"] is True


@pytest.mark.parametrize(("support_user_args", "same_key"), [(False, True), (True, False)])
def test_key_no_internal_args_masks_grouped_gemm_only_without_user_args(
    make_state, support_user_args, same_key
):
    grouped = make_state(
        ProblemType=ProblemType(
            {"DataType": 0, "GroupedGemm": True, "SupportUserArgs": support_user_args}, False
        )
    )
    plain = make_state(
        ProblemType=ProblemType(
            {"DataType": 0, "GroupedGemm": False, "SupportUserArgs": support_user_args}, False
        )
    )

    grouped_key = N.getKeyNoInternalArgs(grouped, splitGSU=False)
    plain_key = N.getKeyNoInternalArgs(plain, splitGSU=False)

    assert (grouped_key == plain_key) is same_key
    assert ("_GG" in grouped_key) is support_user_args
    assert "_GG" not in plain_key
    assert grouped["ProblemType"]["GroupedGemm"] is True


def test_key_no_internal_args_normalizes_only_fixed_wgmxcc(make_state):
    auto = N.getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=-1), splitGSU=False)
    fixed_four = N.getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=4), splitGSU=False)
    fixed_eight = N.getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=8), splitGSU=False)

    assert fixed_four == fixed_eight
    assert auto != fixed_four


def test_key_no_internal_args_split_gsu_boundaries(make_state):
    keys = {
        gsu: N.getKeyNoInternalArgs(make_state(GlobalSplitU=gsu), splitGSU=True)
        for gsu in (-2, -1, 1, 2, 3)
    }

    assert keys[2] == keys[3] == keys[-1]
    assert keys[1] != keys[2]
    assert keys[-2] != keys[-1]


def test_kernel_name_split_gsu_boundary_two_preserves_pinned_typeerror(make_state):
    # GSU=4 alone cannot distinguish `> 1` from a `> 2` boundary mutant.
    with pytest.raises(TypeError):
        N.getKernelNameMin(make_state(GlobalSplitU=2), splitGSU=True)


def test_kernel_name_split_gsu_auto_preserves_pinned_typeerror(make_state):
    # Keep the automatic sentinel in its own node so mutmut selects this path
    # independently from the parametrized characterization test.
    with pytest.raises(TypeError):
        N.getKernelNameMin(make_state(GlobalSplitU=-1), splitGSU=True)


def test_kernel_name_unsplit_gsu_auto_is_masked(make_state):
    auto = N.getKernelNameMin(make_state(GlobalSplitU=-1), splitGSU=False)
    active = N.getKernelNameMin(make_state(GlobalSplitU=1), splitGSU=False)

    assert auto == active


def test_key_no_internal_args_forwards_naming_flags(make_state, monkeypatch):
    calls = []

    def fake_get_name(state, required, split_gsu, ignore_internal):
        calls.append((split_gsu, ignore_internal))
        return "key"

    monkeypatch.setattr(N, "_getName", fake_get_name)

    assert N.getKeyNoInternalArgs(make_state(), splitGSU=True) == "key"
    assert calls == [(True, False)]


def test_parameter_value_abbreviation_forwards_key_recursively(monkeypatch):
    calls = []

    def fake_primitive(key, value):
        calls.append((key, value))
        return str(value)

    monkeypatch.setattr(N, "getPrimitiveParameterValueAbbreviation", fake_primitive)

    assert N.getParameterValueAbbreviation("Token", [1, 2]) == "1_2"
    assert calls == [("Token", 1), ("Token", 2)]


def test_names_distinguish_auto_and_fixed_wgmxcc_and_restore_state(make_state):
    auto_state = make_state(WorkGroupMappingXCC=-1)
    fixed_state = make_state(WorkGroupMappingXCC=8)
    another_fixed_state = make_state(WorkGroupMappingXCC=4)

    auto = N.getSolutionNameFull(auto_state, splitGSU=False)
    fixed = N.getSolutionNameFull(fixed_state, splitGSU=False)
    another_fixed = N.getSolutionNameFull(another_fixed_state, splitGSU=False)

    assert auto != fixed
    assert fixed == another_fixed
    assert auto_state["WorkGroupMappingXCC"] == -1
    assert fixed_state["WorkGroupMappingXCC"] == 8


@pytest.mark.parametrize(("support_user_args", "same_name"), [(False, True), (True, False)])
def test_kernel_name_masks_grouped_gemm_only_without_user_args(
    make_state, support_user_args, same_name
):
    grouped = make_state(
        ProblemType=ProblemType(
            {"DataType": 0, "GroupedGemm": True, "SupportUserArgs": support_user_args}, False
        )
    )
    plain = make_state(
        ProblemType=ProblemType(
            {"DataType": 0, "GroupedGemm": False, "SupportUserArgs": support_user_args}, False
        )
    )

    grouped_name = N.getKernelNameMin(grouped, splitGSU=False)
    plain_name = N.getKernelNameMin(plain, splitGSU=False)

    assert (grouped_name == plain_name) is same_name
    assert ("_GG" in grouped_name) is support_user_args
    assert "_GG" not in plain_name
    assert grouped["ProblemType"]["GroupedGemm"] is True


def test_kernel_name_masks_raw_grouped_gemm_config(make_state):
    grouped = make_state(ProblemType={"DataType": 0, "GroupedGemm": True})
    plain = make_state(ProblemType={"DataType": 0, "GroupedGemm": False})

    assert N.getKernelNameMin(grouped, splitGSU=False) == N.getKernelNameMin(
        plain, splitGSU=False
    )
    assert grouped["ProblemType"]["GroupedGemm"] is True


@pytest.mark.parametrize("missing", ["MacroTile0", "MacroTile1", "DepthU"])
def test_name_skips_incomplete_macro_tile_triplet(make_state, missing):
    state = make_state()
    state.pop(missing)

    assert "_MT" not in N.getSolutionNameFull(state, splitGSU=False)


def test_raw_problem_type_is_constructed_without_diagnostic_output(make_state, monkeypatch):
    calls = []

    class FakeProblemType:
        def __init__(self, config, printIndexAssignmentInfo):
            calls.append((config, printIndexAssignmentInfo))

        def __str__(self):
            return "PT"

    monkeypatch.setattr(N, "ProblemType", FakeProblemType)
    state = make_state(ProblemType={"DataType": 0, "GroupedGemm": False})

    assert N.getSolutionNameFull(state, splitGSU=False).startswith("PT_")
    assert calls == [({"DataType": 0, "GroupedGemm": False}, False)]


def test_name_uses_physical_non_square_mi_block(make_state):
    state = make_state(MatrixInstM=8, MatrixInstN=32, MIBlock=[32, 8])

    name = N.getSolutionNameFull(state, splitGSU=False)

    assert "_MI32x8x1_" in name
    assert "_MI8x32x1_" not in name


def test_kernel_name_adds_thread_tile_for_non_mi_kernel(make_state):
    state = make_state(ThreadTile=[4, 8])
    for key in ("MatrixInstM", "MatrixInstN", "MatrixInstB", "MIWaveTile"):
        state.pop(key)

    assert "_TT4_8" in N.getKernelNameMin(state, splitGSU=False)


def test_lds_segment_interleave_is_named_only_when_applied(make_state):
    names = {
        value: N.getKernelNameMin(make_state(LDSSegmentInterleave=value), splitGSU=False)
        for value in (0, 1, 2)
    }

    assert "_LDSSI1" in names[1]
    assert "_LDSSI" not in names[0]
    assert "_LDSSI" not in names[2]


def test_empty_custom_kernel_name_is_not_emitted_as_parameter(make_state):
    name = N.getSolutionNameFull(make_state(CustomKernelName=""), splitGSU=False)

    assert "_CKN" not in name


def test_isa_parameter_uses_architecture_encoding(make_state):
    name = N.getSolutionNameFull(make_state(ISA=(9, 4, 10)), splitGSU=False)

    assert "_ISA94a" in name


@pytest.mark.parametrize("wrapper", [N.getSolutionNameMin, N.getSolutionNameFull])
def test_solution_name_wrappers_forward_split_and_internal_flags(
    make_state, monkeypatch, wrapper
):
    calls = []

    def fake_get_name(state, required, split_gsu, ignore_internal):
        calls.append((split_gsu, ignore_internal))
        return "name"

    monkeypatch.setattr(N, "_getName", fake_get_name)

    assert wrapper(make_state(), splitGSU=True) == "name"
    assert calls == [(True, False)]


def test_kernel_name_wrapper_forwards_split_and_internal_flags(make_state, monkeypatch):
    calls = []

    def fake_get_name(state, required, split_gsu, ignore_internal):
        calls.append((split_gsu, ignore_internal))
        return "name"

    monkeypatch.setattr(N, "_getName", fake_get_name)

    assert N.getKernelNameMin(make_state(), splitGSU=True) == "name"
    assert calls == [(True, True)]


def test_shorten_file_base_forwards_split_gsu(monkeypatch):
    calls = []

    def fake_kernel_name(kernel, split_gsu):
        calls.append((kernel, split_gsu))
        return "short"

    kernel = {"sentinel": True}
    monkeypatch.setattr(N, "getKernelNameMin", fake_kernel_name)

    assert N.shortenFileBase(True, kernel) == "short"
    assert calls == [(kernel, True)]


def test_shorten_file_base_preserves_exact_length_boundary(monkeypatch):
    base = "x" * MAX_FILENAME_LENGTH
    monkeypatch.setattr(N, "getKernelNameMin", lambda kernel, split_gsu: base)

    assert N.shortenFileBase(False, {}) == base


def test_kernel_file_base_custom_name_bypasses_shortening(monkeypatch):
    def unexpected(*args):
        raise AssertionError("custom names must bypass generated-name shortening")

    monkeypatch.setattr(N, "shortenFileBase", unexpected)

    assert N.getKernelFileBase(True, {"CustomKernelName": "custom"}) == "custom"


def test_kernel_file_base_forwards_split_gsu_to_shortener(monkeypatch):
    calls = []

    def fake_shorten(split_gsu, kernel):
        calls.append((split_gsu, kernel))
        return "generated"

    kernel = {"CustomKernelName": ""}
    monkeypatch.setattr(N, "shortenFileBase", fake_shorten)

    assert N.getKernelFileBase(True, kernel) == "generated"
    assert calls == [(True, kernel)]
