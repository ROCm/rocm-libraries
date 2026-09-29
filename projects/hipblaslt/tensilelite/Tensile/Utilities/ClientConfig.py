# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensile client options shared by generation and packaged-library testing."""

from collections.abc import Mapping


def problemTypeOptions(problem):
    """Accept a Contractions.ProblemType or its serialized runtime state.

    Keep generator classes out of this module so installed client tools need
    neither rocisa nor the rest of the generation pipeline to read metadata.
    Runtime metadata omits scalar types, which default to computeType.
    """
    get = problem.get if isinstance(problem, Mapping) else lambda k, d=None: getattr(problem, k, d)
    options = {}

    def add(key, field, default=None):
        value = get(field, default)
        if value is not None:
            if hasattr(value, "toEnum"):
                value = value.toEnum()
            elif hasattr(value, "toName"):
                value = value.toName()
            options[key] = value

    add("problem-identifier", "operationIdentifier")
    for tensor in "AB":
        add(
            f"compute-input-type-{tensor}",
            f"computeInputType{tensor}",
            get("computeInputType", get(f"{tensor.lower()}Type")),
        )
    for tensor in "abcd":
        add(f"{tensor}-type", f"{tensor}Type")
    if get("useE"):
        add("e-type", "eType")
    if get("outputAmaxD"):
        add("amaxD-type", "amaxDType", get("computeType"))
    add("alpha-type", "alphaType", get("computeType"))
    add("beta-type", "betaType", get("computeType"))
    add("f32-xdl-math-op", "f32XdlMathOp")
    add("activation-compute-type", "activationComputeDataType")
    add("use-gradient", "useGradient")
    add("use-bias", "useBias")
    options["bias-source"] = get("biasSrcWhiteList")[0]
    for key, field in (
        ("use-e", "useE"),
        ("use-gate-residual", "useGateResidual"),
        ("output-amaxD", "outputAmaxD"),
        ("use-scaleAB", "useScaleAB"),
        ("use-scaleCD", "useScaleCD"),
        ("use-scaleAlphaVec", "useScaleAlphaVec"),
        ("swizzle-tensor-a", "swizzleTensorA"),
        ("swizzle-tensor-b", "swizzleTensorB"),
        ("fused-gemm-a2a", "fusedGemmA2A"),
    ):
        add(key, field)
    for tensor in "AB":
        if get(f"mxBlock{tensor}"):
            add(f"mx-{tensor.lower()}-block", f"mxBlock{tensor}")
            add(f"mx-{tensor.lower()}-type", f"mxType{tensor}")
    for key, field in (
        ("sparse", "sparse"),
        ("metadata-layout", "metadataLayout"),
        ("high-precision-accumulate", "highPrecisionAccumulate"),
        ("strided-batched", "stridedBatched"),
        ("grouped-gemm", "groupedGemm"),
        ("activation-type", "activationType"),
        ("activation-no-guard", "activationNoGuard"),
    ):
        add(key, field)
    return options
