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
from tensilelite.Common.Architectures import SUPPORTED_ISA
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.SolutionStructs.Problem import ProblemType
from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

pytestmark = pytest.mark.unit


_FIXTURE = (Path(__file__).parent / "characterization" / "LibraryIO"
            / "data" / "logic_gfx942_HSS_BH.yaml")


@pytest.fixture(autouse=True)
def no_features(monkeypatch):
    # The tests below fake getCustomKernelConfig, so the kernel has no file to read
    # Features from; by default it declares none.
    monkeypatch.setattr(L, "readCustomKernelConfig", lambda name: {})


@pytest.fixture(scope="module")
def cxx_compiler():
    return validateToolchain("amdclang++")


@pytest.fixture(scope="module")
def isa_info_map(cxx_compiler):
    return makeIsaInfoMap(SUPPORTED_ISA, cxx_compiler)


@pytest.fixture(scope="module")
def assembler(cxx_compiler):
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx_compiler, bundler, "default").assembler


def _raw_dict():
    data = L.read(str(_FIXTURE), True)
    assert isinstance(data, List)
    return L.parseLibraryLogicList(copy.deepcopy(data), str(_FIXTURE))


def test_custom_kernel_legacy_name_is_resolved(assembler, isa_info_map, monkeypatch):
    # Legacy flat CustomKernelName + an (empty) config -> the merge branch runs
    # and the solution constructs normally (no handwritten-kernel path).
    seen = []
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: seen.append(name) or {})
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
    def _raise(name, isp):
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
        L, "getCustomKernelConfig", lambda name, isp: {"MatrixInstruction": [1, 2, 3]}
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
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: {})
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
    [
        None,
        {},
        {"SupportsBias": True, "SupportsActivation": True, "SupportsScaleAlpha": True},
        {"SupportsUserArgs": False, "SupportsGSU": False},
    ],
)
def test_problem_type_for_custom_kernel_without_restriction_returns_the_shared_object(features):
    shared = _shared_problem_type()

    assert L._problemTypeForCustomKernel(shared, features, False, "test.yaml") is shared


def test_config_problem_type_flags_do_not_restrict_without_features(
        assembler, isa_info_map, monkeypatch):
    # A config that sets Activation False in its ProblemType but has no Features block
    # (the shipped Custom_Cijk kernels) keeps the logic file's problem type.
    monkeypatch.setattr(
        L,
        "getCustomKernelConfig",
        lambda name, isp: {"ProblemType": {"UseBias": 0, "Activation": False}},
    )
    monkeypatch.setattr(L, "readCustomKernelConfig", lambda name: {})
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "no_features_kernel"

    solution = _parse(data, assembler, isa_info_map).solutions[0]

    assert solution["ProblemType"]["UseBias"] == 1
    assert solution["ProblemType"]["ActivationType"] == "hipblaslt_all"


def _parse(data, assembler, isa_info_map):
    return L.parseLibraryLogicData(
        data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )


def _custom_solution_data(config, monkeypatch, **logicKeys):
    monkeypatch.setattr(L, "getCustomKernelConfig", lambda name, isp: dict(config))
    data = _raw_dict()
    data["Solutions"][0]["CustomKernelName"] = "overlay_kernel"
    data["Solutions"][0].update(logicKeys)
    return data


def test_logic_file_cannot_loosen_what_custom_config_declares(assembler, isa_info_map, monkeypatch):
    config = {
        "AssertSizeLessThan": {0: 1048577, 1: 1048577},
        "AssertTensorBytesLessThan": 2**32,
    }
    data = _custom_solution_data(
        config,
        monkeypatch,
        AssertSizeLessThan={0: 10**9},
        AssertTensorBytesLessThan=2**40,
        AssertSizeGreaterThan={0: 5},
    )

    solution = _parse(data, assembler, isa_info_map).solutions[0]

    assert solution["AssertSizeLessThan"] == config["AssertSizeLessThan"]
    assert solution["AssertTensorBytesLessThan"] == 2**32
    # A key the config does not declare can only add a predicate.
    assert solution["AssertSizeGreaterThan"] == {0: 5}


@pytest.mark.parametrize(
    "key, value, error, message",
    [
        ("AssertSizeLessThan", 5, RuntimeError, "must be a dict"),
        ("AssertStrideAEqual", [1], RuntimeError, "must be a dict"),
        ("AssertSizeMultiple", {3: 0}, Exception, "Multiple must be > 0"),
        ("AssertStrideAMultiple", {1: 0}, Exception, "Multiple must be > 0"),
        ("AssertTensorBytesLessThan", 0, RuntimeError, "positive int or -1"),
    ],
)
def test_invalid_logic_file_assert_values_raise_while_parsing(
        assembler, isa_info_map, monkeypatch, key, value, error, message):
    data = _custom_solution_data({}, monkeypatch, **{key: value})

    with pytest.raises(error, match=message):
        _parse(data, assembler, isa_info_map)
