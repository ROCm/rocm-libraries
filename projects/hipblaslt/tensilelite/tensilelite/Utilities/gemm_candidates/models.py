# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Immutable GEMM cases, profile normalization, and packed matrix layouts."""

from dataclasses import dataclass
from pathlib import Path

# Supported candidate types: (storage bytes, hipblaslt-bench spelling).
# Generator DataType supports a wider domain and depends on rocisa. Keep this
# small client-facing subset together, checked against that registry in tests.
_TYPE_INFO = {
    "Float": (4, "f32_r"),
    "Half": (2, "f16_r"),
    "BFloat16": (2, "bf16_r"),
    "Float8": (1, "f8_r"),
    "BFloat8": (1, "bf8_r"),
    "Float8_fnuz": (1, "f8_fnuz_r"),
    "BFloat8_fnuz": (1, "bf8_fnuz_r"),
    "Int8": (1, "i8_r"),
    "Int32": (4, "i32_r"),
}


def bench_type(name: str) -> str:
    """Convert a supported Tensile type name to a bench input type."""
    return _TYPE_INFO[name][1]


@dataclass(frozen=True, slots=True)
class GemmShape:
    """Positive GEMM dimensions, with batch count separate from matrix axes."""

    m: int
    n: int
    k: int
    batch_count: int = 1

    @property
    def tensile_sizes(self) -> tuple[int, int, int, int]:
        """The client's index order is M, N, batch, K."""
        return self.m, self.n, self.batch_count, self.k


@dataclass(frozen=True, slots=True)
class GemmProfile:
    """Types, transposes and optional features taken from one solution.

    Both adapters use alpha=1, beta=0, no activation and packed column-major
    matrices. Bias/scales follow the metadata; this is not an epilogue sweep.
    Type names are Tensile names, such as ``Half`` and ``Float``.
    """

    a_type: str
    b_type: str
    c_type: str
    d_type: str
    compute_type: str
    trans_a: bool
    trans_b: bool
    compute_input_a: str
    compute_input_b: str
    high_precision_accumulate: bool
    use_bias: bool
    bias_type: str
    bias_source: int
    scale_ab: str
    scale_cd: bool
    scale_alpha_vector: bool
    math_op: str
    activation_type: str
    activation_compute_type: str
    activation_no_guard: bool

    @classmethod
    def from_problem_type(cls, p: dict) -> "GemmProfile":
        """Normalize a runtime problem type to this tool's supported GEMM profile."""
        unsupported = (
            "groupedGemm",
            "fusedGemmA2A",
            "sparse",
            "useGradient",
            "useE",
            "useGateResidual",
            "outputAmaxD",
            "mxBlockA",
            "mxBlockB",
            "useInitialStridesAB",
            "useInitialStridesCD",
            "swizzleTensorA",
            "swizzleTensorB",
            "mirrorDimsA",
            "mirrorDimsB",
        )
        features = [key for key in unsupported if p.get(key)]
        if features:
            raise ValueError(f"unsupported features: {', '.join(features)}")
        if (
            any(p[f"{t}Type"] not in _TYPE_INFO for t in "abcd")
            or p["cType"] != p["dType"]
            or p["computeType"] not in {"Float", "Int32"}
            or not p.get("stridedBatched", True)
            or p.get("activationType", "None") not in {"None", "All", "Hipblaslt_all"}
            or p.get("useScaleAB", "") not in {"", "Scalar", "Vector"}
            or p.get("useBias", 0) not in {0, 1}
            or p.get("useScaleAlphaVec", 0) not in {0, 1}
            or p.get("f32XdlMathOp", "Float") not in {"Float", "XFloat32"}
        ):
            raise ValueError("unsupported GEMM profile")
        profile = cls(
            a_type=p["aType"],
            b_type=p["bType"],
            c_type=p["cType"],
            d_type=p["dType"],
            compute_type=p["computeType"],
            trans_a=bool(p["transA"]),
            trans_b=bool(p["transB"]),
            compute_input_a=p.get("computeInputTypeA", p.get("computeInputType", p["aType"])),
            compute_input_b=p.get("computeInputTypeB", p.get("computeInputType", p["bType"])),
            high_precision_accumulate=bool(p.get("highPrecisionAccumulate")),
            use_bias=bool(p.get("useBias")),
            bias_type=(p.get("biasDataTypeWhiteList") or [p["computeType"]])[0],
            bias_source=(p.get("biasSrcWhiteList") or [3])[0],
            scale_ab=p.get("useScaleAB", ""),
            scale_cd=bool(p.get("useScaleCD")),
            scale_alpha_vector=bool(p.get("useScaleAlphaVec")),
            math_op=p.get("f32XdlMathOp", "Float"),
            activation_type=p.get("activationType", "None"),
            activation_compute_type=p.get("activationComputeDataType", p["computeType"]),
            activation_no_guard=bool(p.get("activationNoGuard")),
        )
        if p["operationIdentifier"] != profile.operation_identifier:
            raise ValueError("not an ordinary Cijk GEMM")
        if profile.bias_source not in {0, 1, 3} or profile.bias_type not in _TYPE_INFO:
            raise ValueError("unsupported bias profile")
        if any(t not in _TYPE_INFO for t in (profile.compute_input_a, profile.compute_input_b)):
            raise ValueError("unsupported compute input types")
        return profile

    def as_problem_type(self) -> dict:
        """Return normalized runtime fields for the shared client-option mapper."""
        return {
            "operationIdentifier": self.operation_identifier,
            "transA": self.trans_a,
            "transB": self.trans_b,
            "aType": self.a_type,
            "bType": self.b_type,
            "cType": self.c_type,
            "dType": self.d_type,
            "computeType": self.compute_type,
            "computeInputTypeA": self.compute_input_a,
            "computeInputTypeB": self.compute_input_b,
            "highPrecisionAccumulate": self.high_precision_accumulate,
            "stridedBatched": True,
            "useBias": int(self.use_bias),
            "biasDataTypeWhiteList": [self.bias_type],
            "biasSrcWhiteList": [self.bias_source],
            "useScaleAB": self.scale_ab,
            "useScaleCD": self.scale_cd,
            "useScaleAlphaVec": int(self.scale_alpha_vector),
            "f32XdlMathOp": self.math_op,
            "activationType": self.activation_type,
            "activationComputeDataType": self.activation_compute_type,
            "activationNoGuard": self.activation_no_guard,
        }

    def data_bytes(self, shape: GemmShape) -> int:
        """Combined A/B/C/D storage, excluding workspace and optional vectors."""
        m, n, k = shape.m, shape.n, shape.k
        elements = (m * k, n * k, m * n, m * n)
        types = (self.a_type, self.b_type, self.c_type, self.d_type)
        return shape.batch_count * sum(
            count * _TYPE_INFO[t][0] for count, t in zip(elements, types)
        )

    @property
    def operation_identifier(self) -> str:
        a = "lik" if self.trans_a else "ilk"
        b = "jlk" if self.trans_b else "ljk"
        return f"Contraction_l_A{a}_B{b}_Cijk_Dijk"


@dataclass(frozen=True, slots=True)
class GemmCase:
    """One candidate, its intended purpose, and enough data for either client.

    The case has passed the generator's necessary dimensional constraints.
    It has not been checked against the native library or a GPU. ``reason``
    describes selection intent, not measured control-flow coverage.
    """

    solution_index: int
    solution_name: str
    kernel_name: str
    metadata_file: Path
    code_objects: tuple[Path, ...]
    shape: GemmShape
    profile: GemmProfile
    id: str
    reason: str

    @property
    def leading_dimensions(self) -> tuple[int, int, int, int]:
        """Column strides of packed A/B/C/D matrices, in elements."""
        s, p = self.shape, self.profile
        return s.k if p.trans_a else s.m, s.n if p.trans_b else s.k, s.m, s.m

    @property
    def batch_strides(self) -> tuple[int, int, int, int]:
        """Batch strides of packed A/B/C/D matrices, in elements."""
        s = self.shape
        return s.m * s.k, s.n * s.k, s.m * s.n, s.m * s.n
