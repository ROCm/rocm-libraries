# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from copy import deepcopy

import pytest

from Tensile.CustomKernelCompatibility import (
    compareCustomKernelProblemTypes,
    formatCustomKernelProblemTypeMismatches,
)


pytestmark = pytest.mark.unit


def _problem_type(**overrides):
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "S",
        "DestDataType": "S",
        "ComputeDataType": "S",
        "TransposeA": False,
        "TransposeB": True,
        "Batched": True,
        "UseBeta": True,
        "HighPrecisionAccumulate": False,
    }
    problem_type.update(overrides)
    return problem_type


def _fields(mismatches):
    return {mismatch.field for mismatch in mismatches}


def test_exact_problem_types_are_compatible():
    problem_type = _problem_type(
        UseBias=1,
        BiasDataTypeList=["S"],
        Activation=True,
        ActivationType="hipblaslt_all",
        UseScaleAlphaVec=1,
    )

    assert compareCustomKernelProblemTypes(problem_type, deepcopy(problem_type)) == []


def test_omitted_capabilities_normalize_to_disabled():
    library = _problem_type(
        UseBias=0,
        UseScaleAlphaVec=0,
        Activation=False,
        UseE=False,
        Gradient=False,
        UseScaleAB="",
        UseScaleCD=False,
        UseGateResidual=False,
    )
    kernel = _problem_type()

    assert compareCustomKernelProblemTypes(library, kernel) == []


def test_kernel_capability_superset_is_compatible():
    library = _problem_type()
    kernel = _problem_type(
        UseBias=3,
        BiasDataTypeList=["S"],
        UseScaleAlphaVec=3,
        Activation=True,
        ActivationType="all",
        UseScaleAB="Vector",
        UseScaleCD=True,
    )

    assert compareCustomKernelProblemTypes(library, kernel) == []


@pytest.mark.parametrize(
    "field,shared_overrides",
    [
        (
            "UseE",
            {"Activation": True, "ActivationType": "all"},
        ),
        ("Gradient", {"UseBias": 1}),
        ("UseGateResidual", {}),
        ("OutputAmaxD", {}),
        (
            "ActivationNoGuard",
            {
                "Activation": True,
                "ActivationType": "all",
                "Gradient": True,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "library_value,kernel_value",
    [(False, True), (True, False)],
)
def test_exact_boolean_fields_require_equality(
    field, shared_overrides, library_value, kernel_value
):
    library_overrides = {**shared_overrides, field: library_value}
    kernel_overrides = {**shared_overrides, field: kernel_value}
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(**library_overrides),
        _problem_type(**kernel_overrides),
    )

    assert field in _fields(mismatches)


def test_scale_cd_kernel_superset_is_compatible():
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(UseScaleCD=False),
            _problem_type(UseScaleCD=True),
        )
        == []
    )


def test_scale_cd_insufficient_kernel_is_rejected():
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(UseScaleCD=True),
        _problem_type(UseScaleCD=False),
    )

    assert _fields(mismatches) == {"UseScaleCD"}


@pytest.mark.parametrize("field", ["UseBias", "UseScaleAlphaVec"])
def test_direction_bitmask_superset_is_compatible(field):
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(**{field: 1}),
            _problem_type(**{field: 3}),
        )
        == []
    )


@pytest.mark.parametrize("field", ["UseBias", "UseScaleAlphaVec"])
@pytest.mark.parametrize(
    "library_value,kernel_value",
    [(2, 1), (3, 1), (3, 2)],
)
def test_insufficient_direction_bitmask_is_rejected(
    field, library_value, kernel_value
):
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(**{field: library_value}),
        _problem_type(**{field: kernel_value}),
    )

    assert field in _fields(mismatches)


@pytest.mark.parametrize("kernel_mode", ["Scalar", "Vector"])
def test_disabled_scale_ab_accepts_enabled_kernel_mode(kernel_mode):
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(UseScaleAB=""),
            _problem_type(UseScaleAB=kernel_mode),
        )
        == []
    )


@pytest.mark.parametrize(
    "library_mode,kernel_mode",
    [("Scalar", "Vector"), ("Vector", "Scalar"), ("Vector", "")],
)
def test_scale_ab_requires_the_requested_mode(library_mode, kernel_mode):
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(UseScaleAB=library_mode),
        _problem_type(UseScaleAB=kernel_mode),
    )

    assert _fields(mismatches) == {"UseScaleAB"}


@pytest.mark.parametrize(
    "library_type,kernel_type",
    [
        ("hipblaslt_all", "all"),
        ("all", "hipblaslt_all"),
    ],
)
def test_custom_activation_type_is_non_authoritative(
    library_type, kernel_type
):
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(Activation=True, ActivationType=library_type),
            _problem_type(Activation=True, ActivationType=kernel_type),
        )
        == []
    )


def test_raw_activation_false_is_rejected():
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(Activation=True, ActivationType="hipblaslt_all"),
        _problem_type(Activation=False),
    )

    assert _fields(mismatches) == {"Activation"}


def test_activation_true_with_omitted_type_covers_library_activation():
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(Activation=True, ActivationType="hipblaslt_all"),
            _problem_type(Activation=True),
        )
        == []
    )


def test_structural_mismatch_is_rejected():
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(TransposeA=False),
        _problem_type(TransposeA=True),
    )

    assert "TransposeA" in _fields(mismatches)


@pytest.mark.parametrize(
    "field,contradictory_value",
    [
        ("IndexAssignmentsA", [3, 0, 2]),
        ("IndexAssignmentsB", [3, 1, 2]),
        ("IndexAssignmentsMetadata", [0, 3, 2]),
        ("NumIndicesC", 2),
        ("NumIndicesLD", 5),
        ("IndexAssignmentsLD", [4, 5, 6, 8]),
    ],
)
def test_explicit_contradictory_structural_declaration_is_rejected(
    field, contradictory_value
):
    kernel = _problem_type(**{field: contradictory_value})
    original_kernel = deepcopy(kernel)

    mismatches = compareCustomKernelProblemTypes(_problem_type(), kernel)

    assert field in _fields(mismatches)
    assert kernel == original_kernel


def test_omitted_structural_declarations_are_derived_and_compatible():
    assert compareCustomKernelProblemTypes(_problem_type(), _problem_type()) == []


def test_missing_embedded_problem_type_is_rejected():
    mismatches = compareCustomKernelProblemTypes(_problem_type(), None)

    assert _fields(mismatches) == {"ProblemType"}


def test_bias_data_types_use_directional_coverage():
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(UseBias=1, BiasDataTypeList=["S"]),
            _problem_type(UseBias=1, BiasDataTypeList=["H", "S"]),
        )
        == []
    )

    mismatches = compareCustomKernelProblemTypes(
        _problem_type(UseBias=1, BiasDataTypeList=["H", "S"]),
        _problem_type(UseBias=1, BiasDataTypeList=["S"]),
    )
    assert _fields(mismatches) == {"BiasDataTypeList"}


def test_inactive_dependent_metadata_is_ignored():
    assert (
        compareCustomKernelProblemTypes(
            _problem_type(DataTypeE="S", ActivationComputeDataType="S"),
            _problem_type(DataTypeE="H", ActivationComputeDataType="H"),
        )
        == []
    )


def test_active_activation_compute_type_must_match():
    library = _problem_type(
        DataType="H",
        DestDataType="H",
        ComputeDataType="S",
        HighPrecisionAccumulate=True,
        Activation=True,
        ActivationType="all",
        ActivationComputeDataType="S",
    )
    kernel = deepcopy(library)
    kernel["ActivationComputeDataType"] = "H"

    mismatches = compareCustomKernelProblemTypes(library, kernel)

    assert _fields(mismatches) == {"ActivationComputeDataType"}


def test_formatter_includes_actionable_context_and_values():
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(UseBias=1),
        _problem_type(),
    )

    message = formatCustomKernelProblemTypeMismatches(
        mismatches, "logic/path.yaml", 7, "CustomKernel"
    )

    assert "logic path=logic/path.yaml" in message
    assert "solution index=7" in message
    assert "custom kernel=CustomKernel" in message
    assert "field=UseBias library=1 kernel=0" in message
