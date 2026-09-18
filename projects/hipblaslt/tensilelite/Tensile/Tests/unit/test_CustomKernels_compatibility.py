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


@pytest.mark.parametrize(
    "field,library_value,kernel_value",
    [
        ("UseBias", 0, 1),
        ("UseBias", 1, 3),
        ("UseScaleAlphaVec", 0, 1),
        ("UseScaleAlphaVec", 1, 3),
        ("UseScaleAB", "", "Vector"),
        ("UseScaleAB", "", "Scalar"),
        ("UseScaleCD", False, True),
        ("Activation", False, True),
    ],
)
def test_kernel_capability_superset_is_rejected(field, library_value, kernel_value):
    # The host appends epilogue kernargs conditionally on the logic's
    # ProblemType, and a custom kernel's .s has fixed kernarg offsets, so a
    # kernel expecting more reads a slot the host never wrote.
    library = _problem_type(**{field: library_value})
    kernel = _problem_type(
        **{field: kernel_value},
        **({"BiasDataTypeList": ["S"]} if field == "UseBias" else {}),
        **({"ActivationType": "all"} if field == "Activation" else {}),
    )

    assert field in _fields(compareCustomKernelProblemTypes(library, kernel))


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


def test_scale_cd_insufficient_kernel_is_rejected():
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(UseScaleCD=True),
        _problem_type(UseScaleCD=False),
    )

    assert _fields(mismatches) == {"UseScaleCD"}


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


def test_missing_embedded_problem_type_is_accepted():
    # A custom.config may hold only InternalSupportParams. Declaring no
    # ProblemType declares no constraints, so the pairing stands.
    library = _problem_type(
        UseBias=1,
        BiasDataTypeList=["S"],
        Activation=True,
        UseScaleAlphaVec=1,
    )

    assert compareCustomKernelProblemTypes(library, None) == []


@pytest.mark.parametrize("kernel_problem_type", ["ProblemType", ["S"], 7])
def test_malformed_embedded_problem_type_is_rejected(kernel_problem_type):
    # Absent is fine; present-but-not-a-mapping is a broken custom.config.
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(), kernel_problem_type
    )

    assert _fields(mismatches) == {"ProblemType"}


def test_declared_problem_type_omitting_bias_is_rejected():
    # A custom.config that declares a ProblemType but leaves out UseBias and
    # UseScaleAlphaVec. Unlike the no-ProblemType case above, a declared block is
    # authoritative, so the omission means "unsupported".
    library = _problem_type(UseBias=1, BiasDataTypeList=["S"], UseScaleAlphaVec=1)
    kernel = _problem_type()

    mismatches = compareCustomKernelProblemTypes(library, kernel)

    # BetaOnlyUseBias is derived from UseBias and rides along.
    assert {"UseBias", "UseScaleAlphaVec"}.issubset(_fields(mismatches))


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


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"NumIndicesC": 3},
        {"IndexAssignmentsLD": [4, 5, 6, 7]},
        {"UseBias": 1, "BiasDataTypeList": ["S"]},
        {"Activation": True, "ActivationType": "hipblaslt_all"},
        {"Activation": True},
        {"UseE": True, "Activation": True, "DataTypeE": "S"},
        {"UseScaleAB": "Vector"},
        {"UseScaleCD": True},
        {"Batched": False},
        {"TransposeA": True, "TransposeB": False},
        {"UseGateResidual": True},
        {"OutputAmaxD": True, "DataTypeAmaxD": "S"},
        {"Gradient": True, "UseBias": 1, "BiasDataTypeList": ["S"], "BiasSrc": "A"},
    ],
)
def test_comparison_is_reflexive(overrides):
    # Equal inputs must never mismatch, including when one field is stated
    # outright and would otherwise be derived on only one side.
    problem_type = _problem_type(**overrides)

    assert (
        compareCustomKernelProblemTypes(
            deepcopy(problem_type), deepcopy(problem_type)
        )
        == []
    )


def test_self_contradictory_structural_declaration_is_rejected():
    # NumIndicesC=2 on a batched GEMM contradicts what the same config derives
    # (3). Checked against its own derivation, so reflexivity holds.
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(), _problem_type(NumIndicesC=2)
    )

    assert _fields(mismatches) == {"NumIndicesC"}


def test_activation_type_omission_does_not_disable_use_e():
    # ProblemType reads a missing ActivationType as 'none' and uses that to
    # force UseE and ActivationNoGuard off, which must not happen when
    # ActivationType is treated as non-authoritative.
    library = _problem_type(
        Activation=True, ActivationType="hipblaslt_all", UseE=True, DataTypeE="S"
    )
    kernel = _problem_type(Activation=True, UseE=True, DataTypeE="S")

    assert compareCustomKernelProblemTypes(library, kernel) == []


@pytest.mark.parametrize(
    "library_problem_type,kernel_problem_type",
    [
        (None, {"OperationType": "GEMM", "DataType": "S"}),
        ({}, {"OperationType": "GEMM"}),
        (_problem_type(), {"OperationType": "GEMM", "DataType": "S",
                           "DestDataType": "D", "ComputeDataType": "D"}),
        (_problem_type(), {"UseScaleAB": 0}),
    ],
)
def test_unconstructible_problem_type_is_reported_not_raised(
    library_problem_type, kernel_problem_type
):
    # _runChecks fans out over ParallelMap2, where an escaping exception takes
    # down a whole batch instead of counting one rejected solution.
    mismatches = compareCustomKernelProblemTypes(
        library_problem_type, kernel_problem_type
    )

    assert mismatches
    assert "ProblemType" in _fields(mismatches)


# ActivationComputeDataType is only independently settable on a mixed-precision
# problem; on an all-f32 one ProblemType derives it and both sides collapse.
_MIXED = {
    "DataType": "H",
    "DestDataType": "H",
    "ComputeDataType": "S",
    "HighPrecisionAccumulate": True,
}
_BIAS_ON = {"UseBias": 1, "BiasDataTypeList": ["S"]}
_ACTIVATION_ON = {"Activation": True, "ActivationType": "all"}

# Each entry is (id, library overrides, kernel overrides, field). An unenforced
# dependent field lets a mismatched bias type, stride or aux data type through.
_DEPENDENT_ENFORCED = [
    ("UseBias-SetConstStrideBias",
     {**_BIAS_ON, "SetConstStrideBias": [[0, 1]]}, _BIAS_ON, "SetConstStrideBias"),
    ("UseBias-BiasDataTypeList",
     {"UseBias": 1, "BiasDataTypeList": ["S", "H"]}, _BIAS_ON, "BiasDataTypeList"),
    ("Activation-ActivationComputeDataType",
     {**_MIXED, **_ACTIVATION_ON, "ActivationComputeDataType": "S"},
     {**_MIXED, **_ACTIVATION_ON, "ActivationComputeDataType": "H"},
     "ActivationComputeDataType"),
    ("UseE-DataTypeE",
     {**_MIXED, **_ACTIVATION_ON, "UseE": True, "DataTypeE": "S"},
     {**_MIXED, **_ACTIVATION_ON, "UseE": True, "DataTypeE": "H"}, "DataTypeE"),
    ("OutputAmaxD-DataTypeAmaxD",
     {"OutputAmaxD": True, "DataTypeAmaxD": "S"},
     {"OutputAmaxD": True, "DataTypeAmaxD": "H"}, "DataTypeAmaxD"),
    ("UseGateResidual-GateResidualDataTypeList",
     {"UseGateResidual": True, "GateResidualDataTypeList": ["S", "H"]},
     {"UseGateResidual": True, "GateResidualDataTypeList": ["S"]},
     "GateResidualDataTypeList"),
    ("UseGateResidual-SetConstStrideGate",
     {"UseGateResidual": True, "SetConstStrideGate": [[0, 1]]},
     {"UseGateResidual": True}, "SetConstStrideGate"),
    ("Gradient-BiasSrc",
     {**_BIAS_ON, "Gradient": True, "BiasSrc": "A"},
     {**_BIAS_ON, "Gradient": True, "BiasSrc": "B"}, "BiasSrc"),
]


@pytest.mark.parametrize(
    "library_overrides,kernel_overrides,field",
    [case[1:] for case in _DEPENDENT_ENFORCED],
    ids=[case[0] for case in _DEPENDENT_ENFORCED],
)
def test_dependent_field_enforced_when_capability_enabled(
    library_overrides, kernel_overrides, field
):
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(**library_overrides), _problem_type(**kernel_overrides)
    )

    assert field in _fields(mismatches)


@pytest.mark.parametrize(
    "field,library_value,kernel_value",
    [
        ("SetConstStrideBias", [[0, 1]], []),
        ("BiasDataTypeList", ["S", "H"], ["S"]),
        ("DataTypeE", "S", "H"),
        ("DataTypeAmaxD", "S", "H"),
        ("GateResidualDataTypeList", ["S", "H"], ["S"]),
        ("SetConstStrideGate", [[0, 1]], []),
        ("BiasSrc", "A", "B"),
    ],
)
def test_dependent_field_ignored_when_capability_disabled(
    field, library_value, kernel_value
):
    # With the capability off the field is dead weight; rejecting a kernel over
    # a leftover default it never reads would be a false positive.
    mismatches = compareCustomKernelProblemTypes(
        _problem_type(**{field: library_value}),
        _problem_type(**{field: kernel_value}),
    )

    assert field not in _fields(mismatches)


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
