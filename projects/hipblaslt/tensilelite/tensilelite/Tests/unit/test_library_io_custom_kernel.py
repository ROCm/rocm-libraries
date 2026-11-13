# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct-lane coverage for the custom-kernel branch of
``LibraryIO.parseLibraryLogicData``.

The heavy parse path (live ``Solution`` construction) is driven against the
vendored production logic file used by the LibraryIO characterization suite,
with a real ``amdclang++`` assembler + per-ISA capability map. Unlike the
characterization copy, these live in a direct ``Tests/unit`` module so the
coverage lane credits the changed lines, and they exercise the *new* branches:
the ``CustomKernel`` mapping-name detection and the skip-on-missing-config path
(which returns ``None`` and drops the solution).
"""

import copy
from pathlib import Path
from typing import List

import pytest

import tensilelite.LibraryIO as L
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit


_FIXTURE = (Path(__file__).parent / "characterization" / "LibraryIO"
            / "data" / "logic_gfx942_HSS_BH.yaml")


@pytest.fixture(autouse=True)
def no_features(monkeypatch):
    # The tests below use synthetic kernel names with no file to read, so by
    # default the config is empty and declares no Features.
    monkeypatch.setattr(L, "readCustomKernelConfig", lambda name: {})


def _raw_dict():
    data = L.read(str(_FIXTURE), True)
    assert isinstance(data, List)
    return L.parseLibraryLogicList(copy.deepcopy(data), str(_FIXTURE))


def test_custom_kernel_legacy_name_is_resolved(assembler, isa_info_map, monkeypatch):
    # Legacy flat CustomKernelName + an (empty) config -> the merge branch runs
    # and the solution constructs normally (no handwritten-kernel path).
    seen = []
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp, kernelConfig=None: seen.append(name) or {})
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "synthetic_legacy_kernel"
    # Present InternalSupportParams -> the isp-extraction branch also runs.
    data["Solutions"][0]["InternalSupportParams"] = {"KernelLanguage": "Assembly"}

    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    assert "synthetic_legacy_kernel" in seen
    assert len(logic.solutions) >= 1


def test_custom_kernel_mapping_name_skipped_on_missing_config(
        assembler, isa_info_map, monkeypatch, capsys):
    # A solution using the new CustomKernel: { name: ... } mapping form whose
    # config can't be read is dropped (returns None), emitting a per-kernel and
    # an aggregate "Skipped N" warning. Appending (rather than replacing) keeps
    # the vendored fixture's solution-index table intact so only the new,
    # dropped kernel is exercised.
    def _raise(name, isp, kernelConfig=None):
        raise RuntimeError("no custom.config")

    monkeypatch.setattr(L, "getCustomKernelConfig", _raise)
    data = _raw_dict()
    total = len(data["Solutions"])
    data["Solutions"].append(
        {"KernelLanguage": "Assembly", "CustomKernel": {"name": "map_broken_kernel"}}
    )

    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    out = capsys.readouterr().out
    assert "map_broken_kernel" in out              # mapping-name resolved
    assert "Skipped 1 solution(s)" in out          # aggregate skip warning
    assert len(logic.solutions) == total           # only the appended kernel dropped


def test_custom_kernel_bad_matrix_instruction_raises(assembler, isa_info_map, monkeypatch):
    monkeypatch.setattr(
        L, "getCustomKernelConfig", lambda name, isp, kernelConfig=None: {"MatrixInstruction": [1, 2, 3]}
    )
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "bad_mi_kernel"

    with pytest.raises(ValueError):
        L.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        )


def test_custom_config_bias_and_activation_stay_off_the_shared_problem_type(
        assembler, isa_info_map, monkeypatch):
    # The library ProblemType enables bias and hipblaslt_all. A kernel whose
    # custom.config turns both off must not inherit that, and must not change
    # the other solutions that still share the library problem type.
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp, kernelConfig=None: {})
    monkeypatch.setattr(
        L,
        "readCustomKernelConfig",
        lambda name: {"Features": {"SupportsBias": False, "SupportsActivation": False}},
    )
    data = _raw_dict()
    # Keep an unmodified copy of the same solution so the shared library
    # problem type is still observable after the custom kernel is restricted.
    data["Solutions"].append(copy.deepcopy(data["Solutions"][0]))
    data["Solutions"][0]["CustomKernelName"] = "skinny_no_bias"

    logic = L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    restricted = logic.solutions[0]["ProblemType"]
    untouched = logic.solutions[-1]["ProblemType"]
    assert restricted["UseBias"] == 0
    assert restricted["ActivationType"] == "none"
    assert untouched["UseBias"] == 1
    assert untouched["ActivationType"] == "hipblaslt_all"


def _shared_problem_type():
    state = _raw_dict()["ProblemType"]
    L.normalizeLogicProblemType(state)
    state.update(UseBias=1, UseScaleAlphaVec=1)
    return ProblemType(state, False, raiseOnTypeMismatch=False)


@pytest.mark.parametrize(
    "restriction, expected",
    [
        ({"SupportsBias": False}, {"UseBias": 0, "BetaOnlyUseBias": False}),
        ({"SupportsScaleAlpha": False}, {"UseScaleAlphaVec": 0}),
        ({"SupportsActivation": False}, {"Activation": False, "ActivationType": "none"}),
    ],
)
def test_problem_type_for_custom_kernel_restricts_a_copy(restriction, expected):
    shared = _shared_problem_type()
    stateBefore = dict(shared.state)
    features = dict(restriction)

    restricted = L._problemTypeForCustomKernel(shared, features, False, "test.yaml")

    assert isinstance(restricted, ProblemType)
    assert restricted is not shared
    changed = {k: restricted[k] for k in shared.state if restricted[k] != shared[k]}
    assert changed == expected
    assert shared.state == stateBefore
    assert features == restriction


@pytest.mark.parametrize(
    "features",
    [None, {"SupportsUserArgs": False, "SupportsGSU": False}],
)
def test_problem_type_for_custom_kernel_without_restriction_returns_the_shared_object(features):
    shared = _shared_problem_type()

    assert L._problemTypeForCustomKernel(shared, features, False, "test.yaml") is shared


def test_problem_type_for_custom_kernel_returns_the_shared_object_when_nothing_changes():
    # The logic file already has bias, scale-alpha-vector and activation off, so the
    # restriction is a no-op and the solutions keep sharing one ProblemType.
    state = _raw_dict()["ProblemType"]
    L.normalizeLogicProblemType(state)
    state.pop("Activation", None)
    state.update(UseBias=0, UseScaleAlphaVec=0, ActivationType="none")
    shared = ProblemType(state, False, raiseOnTypeMismatch=False)
    features = {"SupportsBias": False, "SupportsActivation": False, "SupportsScaleAlpha": False}

    assert L._problemTypeForCustomKernel(shared, features, False, "test.yaml") is shared


def test_custom_config_is_read_once_and_shared(assembler, isa_info_map, monkeypatch):
    # LibraryIO takes Features from the same raw config it hands to
    # getCustomKernelConfig, instead of reading the file a second time.
    raw = {"Features": {"SupportsBias": False}}
    reads = []
    received = []

    def _read(name):
        reads.append(name)
        return raw

    def _get(name, isp, kernelConfig=None):
        received.append(kernelConfig)
        return {}

    monkeypatch.setattr(L, "readCustomKernelConfig", _read)
    monkeypatch.setattr(L, "getCustomKernelConfig", _get)
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "read_once_kernel"

    L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )

    assert reads == ["read_once_kernel"]
    assert len(received) == 1
    assert received[0] is raw


def _parse_with_config(assembler, isa_info_map, monkeypatch, declared, configured, file_default=None):
    """Parse one custom-kernel solution whose logic file declares `declared` (None for no key).

    `file_default` is an AssertSizeLessThan placed in the file's DefaultSolution.
    """
    monkeypatch.setattr(
        L, "getCustomKernelConfig", lambda name, isp, kernelConfig=None: {"AssertSizeLessThan": configured}
    )
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "replaced_predicate_kernel"
    data["Solutions"][0].pop("AssertSizeLessThan", None)
    if declared is not None:
        data["Solutions"][0]["AssertSizeLessThan"] = declared
    if file_default is not None:
        data["DefaultSolution"] = {"AssertSizeLessThan": file_default}
    return L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )


def test_custom_config_warns_when_it_replaces_a_declared_predicate(
        assembler, isa_info_map, monkeypatch, capsys):
    logic = _parse_with_config(assembler, isa_info_map, monkeypatch, {0: 8}, {0: 5})

    out = capsys.readouterr().out
    assert "Tensile::WARNING" in out
    assert "AssertSizeLessThan" in out
    assert "{0: 5}" in out and "{0: 8}" in out
    assert logic.solutions[0]["AssertSizeLessThan"] == {0: 5}


def test_custom_config_does_not_warn_when_it_agrees_with_the_logic_file(
        assembler, isa_info_map, monkeypatch, capsys):
    _parse_with_config(assembler, isa_info_map, monkeypatch, {0: 5}, {0: 5})

    assert "replaces" not in capsys.readouterr().out


def test_custom_config_does_not_warn_for_a_predicate_the_logic_file_leaves_to_defaults(
        assembler, isa_info_map, monkeypatch, capsys):
    logic = _parse_with_config(assembler, isa_info_map, monkeypatch, None, {0: 5})

    assert "replaces" not in capsys.readouterr().out
    assert logic.solutions[0]["AssertSizeLessThan"] == {0: 5}


def test_custom_config_does_not_warn_for_a_predicate_only_the_file_default_declares(
        assembler, isa_info_map, monkeypatch, capsys):
    _parse_with_config(assembler, isa_info_map, monkeypatch, None, {0: 5}, file_default={0: 8})

    assert "replaces" not in capsys.readouterr().out
