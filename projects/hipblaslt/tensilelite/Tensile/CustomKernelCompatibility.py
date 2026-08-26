# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, NamedTuple


class CustomKernelProblemTypeMismatch(NamedTuple):
    field: str
    library_value: Any
    kernel_value: Any


_BITMASK_CAPABILITIES = ("UseBias", "UseScaleAlphaVec")
_EXACT_BOOLEAN_FIELDS = (
    "UseE",
    "Gradient",
    "UseGateResidual",
    "OutputAmaxD",
    "ActivationNoGuard",
)
_DIRECTIONAL_BOOLEAN_CAPABILITIES = ("UseScaleCD",)
_EXPLICIT_STRUCTURAL_FIELDS = (
    "IndexAssignmentsA",
    "IndexAssignmentsB",
    "IndexAssignmentsMetadata",
    "NumIndicesC",
    "NumIndicesLD",
    "IndexAssignmentsLD",
)
_CAPABILITY_FIELDS = {
    "Activation",
    "ActivationType",
    "UseScaleAB",
    *_BITMASK_CAPABILITIES,
    *_EXACT_BOOLEAN_FIELDS,
    *_DIRECTIONAL_BOOLEAN_CAPABILITIES,
}
_DEPENDENT_FIELDS = {
    "ActivationComputeDataType",
    "BetaOnlyUseBias",
    "BiasDataTypeList",
    "BiasSrc",
    "DataTypeAmaxD",
    "DataTypeE",
    "GateResidualDataTypeList",
    "SetConstStrideBias",
    "SetConstStrideGate",
}


def _canonicalProblemTypeValue(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalProblemTypeValue(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalProblemTypeValue(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalProblemTypeValue(item) for item in value)
    if hasattr(value, "value"):
        return _canonicalProblemTypeValue(value.value)
    return value


def _normalizeProblemType(
    problemType: Mapping, preserveExplicitStructuralFields: bool = False
) -> dict:
    # Keep this import local. Solution imports CustomKernels, so importing
    # ProblemType while parallel TensileLogic workers initialize modules creates
    # a CustomKernels -> ProblemType -> Solution -> CustomKernels cycle.
    from Tensile.SolutionStructs.Problem import ProblemType, _defaultProblemType

    if isinstance(problemType, ProblemType):
        state = problemType.state
        explicitStructuralValues = {}
    else:
        config = deepcopy(dict(problemType))
        explicitStructuralValues = {
            field: deepcopy(config[field])
            for field in _EXPLICIT_STRUCTURAL_FIELDS
            if preserveExplicitStructuralFields and field in config
        }
        # Embedded custom configs historically bypassed ProblemType's type
        # validator, and some shipped kernels use 0/1 for boolean fields. Keep
        # their established meaning while normalizing before comparison.
        for field, defaultValue in _defaultProblemType.items():
            value = config.get(field)
            if type(defaultValue) is bool and type(value) is int and value in (0, 1):
                config[field] = bool(value)
            elif type(defaultValue) is int and type(value) is bool:
                config[field] = int(value)
        state = ProblemType(config, False).state
    normalized = {
        field: _canonicalProblemTypeValue(value)
        for field, value in state.items()
    }
    normalized.update(
        {
            field: _canonicalProblemTypeValue(value)
            for field, value in explicitStructuralValues.items()
        }
    )
    return normalized


def _appendMismatch(
    mismatches: list[CustomKernelProblemTypeMismatch],
    field: str,
    library: dict,
    kernel: dict,
) -> None:
    mismatches.append(
        CustomKernelProblemTypeMismatch(field, library[field], kernel[field])
    )


def compareCustomKernelProblemTypes(
    libraryProblemType: Mapping, kernelProblemType: Any
) -> list[CustomKernelProblemTypeMismatch]:
    """Compare a logic ProblemType with a custom kernel's embedded ProblemType.

    Structural fields and exact runtime predicates must be identical.
    Directional capabilities may be a custom-kernel superset of what the logic
    advertises; Bias and ScaleAlphaVec require full bitmask coverage. A custom
    ActivationType is non-authoritative and is ignored.
    """
    if not isinstance(kernelProblemType, Mapping):
        return [
            CustomKernelProblemTypeMismatch(
                "ProblemType", "compatible mapping", kernelProblemType
            )
        ]

    library = _normalizeProblemType(libraryProblemType)
    kernel = _normalizeProblemType(
        kernelProblemType, preserveExplicitStructuralFields=True
    )
    mismatches: list[CustomKernelProblemTypeMismatch] = []

    ignoredFields = _CAPABILITY_FIELDS | _DEPENDENT_FIELDS
    for field in sorted((library.keys() | kernel.keys()) - ignoredFields):
        if library.get(field) != kernel.get(field):
            mismatches.append(
                CustomKernelProblemTypeMismatch(
                    field, library.get(field), kernel.get(field)
                )
            )

    for field in _BITMASK_CAPABILITIES:
        if library[field] & kernel[field] != library[field]:
            _appendMismatch(mismatches, field, library, kernel)

    libraryActivation = library["Activation"]
    kernelActivation = kernel["Activation"]
    activationCovered = not libraryActivation or kernelActivation
    if not activationCovered:
        _appendMismatch(mismatches, "Activation", library, kernel)

    if library["UseScaleAB"] and library["UseScaleAB"] != kernel["UseScaleAB"]:
        _appendMismatch(mismatches, "UseScaleAB", library, kernel)

    exactBooleanMatches = {}
    for field in _EXACT_BOOLEAN_FIELDS:
        exactBooleanMatches[field] = library[field] == kernel[field]
        if not exactBooleanMatches[field]:
            _appendMismatch(mismatches, field, library, kernel)

    for field in _DIRECTIONAL_BOOLEAN_CAPABILITIES:
        if library[field] and not kernel[field]:
            _appendMismatch(mismatches, field, library, kernel)

    biasCovered = library["UseBias"] & kernel["UseBias"] == library["UseBias"]
    if library["UseBias"] and biasCovered:
        if not set(library["BiasDataTypeList"]).issubset(
            kernel["BiasDataTypeList"]
        ):
            _appendMismatch(mismatches, "BiasDataTypeList", library, kernel)
        if library["SetConstStrideBias"] != kernel["SetConstStrideBias"]:
            _appendMismatch(mismatches, "SetConstStrideBias", library, kernel)

    if (
        libraryActivation
        and activationCovered
        and library["ActivationComputeDataType"]
        != kernel["ActivationComputeDataType"]
    ):
        _appendMismatch(mismatches, "ActivationComputeDataType", library, kernel)

    if (
        library["UseE"]
        and exactBooleanMatches["UseE"]
        and library["DataTypeE"] != kernel["DataTypeE"]
    ):
        _appendMismatch(mismatches, "DataTypeE", library, kernel)

    if (
        library["Gradient"]
        and exactBooleanMatches["Gradient"]
        and library["UseBias"]
        and biasCovered
        and library["BiasSrc"] != kernel["BiasSrc"]
    ):
        _appendMismatch(mismatches, "BiasSrc", library, kernel)

    if library["UseGateResidual"] and exactBooleanMatches["UseGateResidual"]:
        if not set(library["GateResidualDataTypeList"]).issubset(
            kernel["GateResidualDataTypeList"]
        ):
            _appendMismatch(
                mismatches, "GateResidualDataTypeList", library, kernel
            )
        if library["SetConstStrideGate"] != kernel["SetConstStrideGate"]:
            _appendMismatch(mismatches, "SetConstStrideGate", library, kernel)

    if (
        library["OutputAmaxD"]
        and exactBooleanMatches["OutputAmaxD"]
        and library["DataTypeAmaxD"] != kernel["DataTypeAmaxD"]
    ):
        _appendMismatch(mismatches, "DataTypeAmaxD", library, kernel)

    return mismatches


def formatCustomKernelProblemTypeMismatches(
    mismatches: list[CustomKernelProblemTypeMismatch],
    logicPath: Any,
    solutionIndex: Any,
    customKernelName: str,
) -> str:
    """Format custom-kernel incompatibilities consistently for all callers."""
    lines = [
        "Custom kernel ProblemType is incompatible with logic: "
        f"logic path={logicPath}, solution index={solutionIndex}, "
        f"custom kernel={customKernelName}"
    ]
    lines.extend(
        f"  field={mismatch.field} library={mismatch.library_value!r} "
        f"kernel={mismatch.kernel_value!r}"
        for mismatch in mismatches
    )
    lines.append(
        "Use a custom kernel whose embedded custom.config.ProblemType covers "
        "the logic ProblemType, or remove the solution from this logic file."
    )
    return "\n".join(lines)
