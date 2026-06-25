# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the pure logic of Tensile.ExperimentalLibrary.

Lives under ``Tests/extras`` (not ``Tests/unit``) on purpose: the ``unit``
conftest imports ``streamk5_test_helpers`` -> ``rocisa.code``, so collecting any
test there requires a built rocisa. The pure-logic tests below
(``coerce_value``, ``parse_set_arg``, ``augment_config`` round-trip) need no
toolchain because ``Tensile.ExperimentalLibrary`` keeps its rocisa-dependent
imports lazy. The ``validate_sets`` tests genuinely need ``validParameters``
(which pulls in rocisa) and are guarded so they skip gracefully without a build.
"""

import pytest

from Tensile.ExperimentalLibrary import (
    ExperimentalLibraryError,
    augment_config,
    coerce_value,
    parse_set_arg,
    validate_sets,
)


def _require_validparameters():
    """Skip cleanly when the rocisa-backed validParameters registry is absent."""
    pytest.importorskip("rocisa")
    try:
        from Tensile.Common.ValidParameters import validParameters  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Tensile.Common.ValidParameters unavailable: {e}")


def test_coerce_value():
    assert coerce_value("1") == 1 and isinstance(coerce_value("1"), int)
    assert coerce_value("0") == 0
    assert coerce_value("True") is True
    assert coerce_value("false") is False
    assert coerce_value("1.5") == 1.5
    assert coerce_value("MultipleBuffer") == "MultipleBuffer"
    assert coerce_value("[16, 16]") == [16, 16]


def test_parse_set_arg_good():
    assert parse_set_arg("StreamKFixupTreeReduction=1") == (
        "StreamKFixupTreeReduction",
        [1],
    )
    assert parse_set_arg("StreamK=0,1") == ("StreamK", [0, 1])


def test_parse_set_arg_bracketed_list_is_one_value():
    # A bracketed list value must stay a single token despite its commas.
    assert parse_set_arg("MatrixInstruction=[16,16,16,1]") == (
        "MatrixInstruction",
        [[16, 16, 16, 1]],
    )


@pytest.mark.parametrize("bad", ["NoEquals", "=5", "Name="])
def test_parse_set_arg_bad(bad):
    with pytest.raises(ExperimentalLibraryError):
        parse_set_arg(bad)


def test_validate_sets_good():
    _require_validparameters()
    # Both a [0,1] feature param and a multi-value enum param.
    validate_sets([("StreamKFixupTreeReduction", [0, 1])])
    validate_sets([("GlobalSplitUAlgorithm", ["MultipleBuffer"])])


def test_validate_sets_unknown_name_suggests():
    _require_validparameters()
    with pytest.raises(ExperimentalLibraryError) as ei:
        validate_sets([("StreamKFixupTreeReductn", [1])])
    msg = str(ei.value)
    assert "Unknown solution parameter" in msg
    # Close-match suggestion should surface the real name.
    assert "StreamKFixupTreeReduction" in msg


def test_validate_sets_bad_value_lists_allowed():
    _require_validparameters()
    with pytest.raises(ExperimentalLibraryError) as ei:
        validate_sets([("StreamKFixupTreeReduction", [5])])
    msg = str(ei.value)
    assert "Invalid value" in msg
    assert "Allowed values" in msg


def _base_config():
    return {
        "GlobalParameters": {"NumElementsToValidate": 0},
        "BenchmarkProblems": [
            [
                {"OperationType": "GEMM", "DataType": "s"},
                {
                    "InitialSolutionParameters": None,
                    "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                    "ForkParameters": [
                        {"PrefetchGlobalRead": [2]},
                        {"StreamK": [1]},
                        {"Groups": [[{"MatrixInstruction": [16, 16, 16, 1]}]]},
                    ],
                    "BenchmarkFinalParameters": [
                        {"ProblemSizes": [{"Exact": [256, 256, 1, 256]}]}
                    ],
                },
            ]
        ],
        "LibraryLogic": {"ArchitectureName": "gfx950"},
    }


def _fork(config):
    return config["BenchmarkProblems"][0][1]["ForkParameters"]


def test_augment_injects_new_param_before_groups():
    config = _base_config()
    augment_config(config, [("StreamKFixupTreeReduction", [1])])
    fork = _fork(config)
    # New entry present.
    entries = {k: v for d in fork for k, v in d.items()}
    assert entries["StreamKFixupTreeReduction"] == [1]
    # Groups stays the last entry.
    assert "Groups" in fork[-1]


def test_augment_overrides_existing_param_in_place():
    config = _base_config()
    augment_config(config, [("StreamK", [3])])
    fork = _fork(config)
    streamk_entries = [d for d in fork if "StreamK" in d]
    # Overridden in place, not duplicated.
    assert len(streamk_entries) == 1
    assert streamk_entries[0]["StreamK"] == [3]


def test_augment_preserves_structure_and_other_keys():
    config = _base_config()
    before_size_group_keys = set(config["BenchmarkProblems"][0][1].keys())
    augment_config(config, [("StreamKFixupTreeReduction", [1])])
    after = config["BenchmarkProblems"][0][1]
    assert set(after.keys()) == before_size_group_keys
    assert after["BenchmarkCommonParameters"] == [{"KernelLanguage": ["Assembly"]}]
    assert after["BenchmarkFinalParameters"][0]["ProblemSizes"] == [
        {"Exact": [256, 256, 1, 256]}
    ]
    # Untouched top-level sections survive.
    assert config["LibraryLogic"]["ArchitectureName"] == "gfx950"


def test_augment_missing_benchmark_problems_raises():
    with pytest.raises(ExperimentalLibraryError):
        augment_config({"GlobalParameters": {}}, [("StreamK", [1])])
