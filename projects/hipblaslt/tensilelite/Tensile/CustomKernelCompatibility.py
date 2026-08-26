# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from typing import Any, NamedTuple


class CustomKernelProblemTypeMismatch(NamedTuple):
    field: str
    library_value: Any
    kernel_value: Any


# Exact equality, in both directions. The host appends epilogue kernargs
# conditionally on the logic's ProblemType, and a custom kernel's .s has fixed
# kernarg offsets, so a kernel expecting a capability the logic does not
# advertise reads a slot the host never wrote.
_EXACT_CAPABILITY_FIELDS = (
    "Activation",
    "UseBias",
    "UseScaleAlphaVec",
    "UseScaleAB",
    "UseScaleCD",
    "UseE",
    "Gradient",
    "UseGateResidual",
    "OutputAmaxD",
    "ActivationNoGuard",
)
# A kernel may declare Activation without a type and let the logic pick.
_NON_AUTHORITATIVE_FIELDS = {"ActivationType"}
_EXPLICIT_STRUCTURAL_FIELDS = (
    "IndexAssignmentsA",
    "IndexAssignmentsB",
    "IndexAssignmentsMetadata",
    "NumIndicesC",
    "NumIndicesLD",
    "IndexAssignmentsLD",
)
# Only meaningful when the capability gating them is enabled; comparing them
# unconditionally would flag leftover defaults on kernels that never use them.
_DEPENDENT_FIELDS = {
    "ActivationComputeDataType",
    "BiasDataTypeList",
    "BiasSrc",
    "DataTypeAmaxD",
    "DataTypeE",
    "GateResidualDataTypeList",
    "SetConstStrideBias",
    "SetConstStrideGate",
}


def _canonicalProblemTypeValue(value: Any) -> Any:
    # Keep these imports local, for the cycle described in _normalizeProblemType.
    from Tensile.Activation import ActivationType
    from Tensile.Common.DataType import DataType

    if isinstance(value, Mapping):
        return {key: _canonicalProblemTypeValue(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalProblemTypeValue(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalProblemTypeValue(item) for item in value)
    # Match by type rather than duck-typing on `.value`, so an unrelated value
    # that happens to expose `.value` is not unwrapped.
    if isinstance(value, (DataType, ActivationType, Enum)):
        return _canonicalProblemTypeValue(value.value)
    return value


def _explicitStructuralDeclarations(problemType: Any) -> dict:
    """Structural fields a raw custom.config states outright, if any.

    These are normally derived from TransposeA/B and Batched. A config that
    states one anyway is validated against its own derivation, never against the
    other side of the comparison, which would break reflexivity.
    """
    if not isinstance(problemType, Mapping):
        return {}
    return {
        field: deepcopy(problemType[field])
        for field in _EXPLICIT_STRUCTURAL_FIELDS
        if field in problemType
    }


def _normalizeProblemType(problemType: Mapping) -> dict:
    """Reduce a ProblemType, or a raw config for one, to comparable values.

    Runs the input through ProblemType to materialize defaults and settle
    derived fields, then unwraps DataType/ActivationType into plain values. Both
    sides go through this unchanged, which is what makes comparison reflexive.

    Raises whatever ProblemType raises; the caller reports that as a mismatch.
    """
    # Keep this import local. Solution imports CustomKernels, so importing
    # ProblemType while parallel TensileLogic workers initialize modules creates
    # a CustomKernels -> ProblemType -> Solution -> CustomKernels cycle.
    from Tensile.SolutionStructs.Problem import ProblemType, _defaultProblemType

    if isinstance(problemType, ProblemType):
        state = problemType.state
    else:
        config = deepcopy(dict(problemType))
        # Some custom configs use 0/1 for boolean fields. Preserve that meaning.
        for field, defaultValue in _defaultProblemType.items():
            value = config.get(field)
            if type(defaultValue) is bool and type(value) is int and value in (0, 1):
                config[field] = bool(value)
            elif type(defaultValue) is int and type(value) is bool:
                config[field] = int(value)
        # ProblemType reads a missing ActivationType as 'none' and uses that to
        # force UseE and ActivationNoGuard off, which would defeat treating
        # ActivationType as non-authoritative.
        if config.get("Activation") and not config.get("ActivationType"):
            config["ActivationType"] = "hipblaslt_all"
        state = ProblemType(config, False).state
    return {f: _canonicalProblemTypeValue(v) for f, v in state.items()}


def _appendMismatch(
    mismatches: list[CustomKernelProblemTypeMismatch],
    field: str,
    library: dict,
    kernel: dict,
) -> None:
    """Record a mismatch for a field ProblemType always populates.

    Indexes directly, unlike the generic loop, which walks the union of both key
    sets and must tolerate a field present on only one side.
    """
    mismatches.append(
        CustomKernelProblemTypeMismatch(field, library[field], kernel[field])
    )


def compareCustomKernelProblemTypes(
    libraryProblemType: Mapping, kernelProblemType: Any
) -> list[CustomKernelProblemTypeMismatch]:
    """Compare a logic ProblemType with a custom kernel's embedded ProblemType.

    Structural fields and capabilities must match exactly, in both directions
    (see _EXACT_CAPABILITY_FIELDS). Fields that only select among runtime values
    without moving a kernarg offset -- BiasDataTypeList,
    GateResidualDataTypeList -- are checked for coverage instead. A custom
    ActivationType is non-authoritative and is ignored.

    A custom.config that omits ProblemType entirely declares no constraints and
    is accepted. Omitting a single field from a declared ProblemType is
    different: the block is authoritative, so a capability missing from it
    normalizes to disabled and is enforced.

    Reflexive: equal inputs yield no mismatches. A config that contradicts
    itself is still caught, separately.

    Never raises. A ProblemType that cannot be constructed is reported as a
    mismatch so it is counted and attributed.
    """
    if kernelProblemType is None:
        return []
    if not isinstance(kernelProblemType, Mapping):
        return [
            CustomKernelProblemTypeMismatch(
                "ProblemType", "compatible mapping", kernelProblemType
            )
        ]

    try:
        library = _normalizeProblemType(libraryProblemType)
    except Exception as err:
        return [
            CustomKernelProblemTypeMismatch(
                "ProblemType", f"<unusable logic ProblemType: {err}>", "n/a"
            )
        ]
    try:
        kernel = _normalizeProblemType(kernelProblemType)
    except Exception as err:
        return [
            CustomKernelProblemTypeMismatch(
                "ProblemType", "constructible", f"<unusable custom.config: {err}>"
            )
        ]
    mismatches: list[CustomKernelProblemTypeMismatch] = []

    # Validate a stated structural field against the same config's derivation;
    # comparing it to the library side would break reflexivity.
    for field, declared in _explicitStructuralDeclarations(kernelProblemType).items():
        derived = kernel.get(field)
        if _canonicalProblemTypeValue(declared) != derived:
            mismatches.append(
                CustomKernelProblemTypeMismatch(field, derived, declared)
            )

    ignoredFields = _NON_AUTHORITATIVE_FIELDS | _DEPENDENT_FIELDS
    for field in sorted((library.keys() | kernel.keys()) - ignoredFields):
        if library.get(field) != kernel.get(field):
            mismatches.append(
                CustomKernelProblemTypeMismatch(
                    field, library.get(field), kernel.get(field)
                )
            )

    # A dependent field matters only when its capability is on and agreed.
    def enabledAndAgreed(field: str) -> bool:
        return bool(library[field]) and library[field] == kernel[field]

    if enabledAndAgreed("UseBias"):
        if not set(library["BiasDataTypeList"]).issubset(kernel["BiasDataTypeList"]):
            _appendMismatch(mismatches, "BiasDataTypeList", library, kernel)
        if library["SetConstStrideBias"] != kernel["SetConstStrideBias"]:
            _appendMismatch(mismatches, "SetConstStrideBias", library, kernel)

    if (
        enabledAndAgreed("Activation")
        and library["ActivationComputeDataType"]
        != kernel["ActivationComputeDataType"]
    ):
        _appendMismatch(mismatches, "ActivationComputeDataType", library, kernel)

    if enabledAndAgreed("UseE") and library["DataTypeE"] != kernel["DataTypeE"]:
        _appendMismatch(mismatches, "DataTypeE", library, kernel)

    if (
        enabledAndAgreed("Gradient")
        and enabledAndAgreed("UseBias")
        and library["BiasSrc"] != kernel["BiasSrc"]
    ):
        _appendMismatch(mismatches, "BiasSrc", library, kernel)

    if enabledAndAgreed("UseGateResidual"):
        if not set(library["GateResidualDataTypeList"]).issubset(
            kernel["GateResidualDataTypeList"]
        ):
            _appendMismatch(mismatches, "GateResidualDataTypeList", library, kernel)
        if library["SetConstStrideGate"] != kernel["SetConstStrideGate"]:
            _appendMismatch(mismatches, "SetConstStrideGate", library, kernel)

    if (
        enabledAndAgreed("OutputAmaxD")
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
