# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Immutable GEMM cases, profile normalization, and packed matrix layouts."""

from dataclasses import dataclass
from pathlib import Path

# Supported candidate types: (container bytes, elements per container, bench spelling).
# Generator DataType supports a wider domain and depends on rocisa. Keep this
# small client-facing subset together, checked against that registry in tests.
_TYPE_INFO = {
    "Float": (4, 1, "f32_r"),
    "Half": (2, 1, "f16_r"),
    "BFloat16": (2, 1, "bf16_r"),
    "Float8": (1, 1, "f8_r"),
    "BFloat8": (1, 1, "bf8_r"),
    "Float8_fnuz": (1, 1, "f8_fnuz_r"),
    "BFloat8_fnuz": (1, 1, "bf8_fnuz_r"),
    "Float4": (1, 2, "f4_r"),
    "Float6": (24, 32, "f6_r"),
    "BFloat6": (24, 32, "bf6_r"),
    "Int8": (1, 1, "i8_r"),
    "Int32": (4, 1, "i32_r"),
}
# hipblaslt_scaling_format values. The bench YAML schema only gives symbolic
# names to none/Scalar/Vector; block modes must use the numeric enum values.
_MX_SCALE_TYPES = {"E8": {32: 3, 16: 4}, "Float8": {32: 5, 16: 6}, "E5M3": {32: 7, 16: 8}}


def bench_type(name: str) -> str:
    """Convert a supported Tensile type name to a bench input type."""
    return _TYPE_INFO[name][2]


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

    Bias/scales follow the metadata; this is not an epilogue sweep.
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
    mx_block_a: int = 0
    mx_block_b: int = 0
    mx_type_a: str = "E8"
    mx_type_b: str = "E8"
    mx_scale_format: int = 0

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
            or _TYPE_INFO[p["dType"]][1] != 1
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
            # Bench has one scale mode per operand. MX block scales take its
            # place; optional scalar/vector A/B scaling can be disabled by the
            # runtime UseScaleAB predicate. Both clients use that same choice.
            scale_ab="" if p.get("mxBlockA") or p.get("mxBlockB") else p.get("useScaleAB", ""),
            scale_cd=bool(p.get("useScaleCD")),
            scale_alpha_vector=bool(p.get("useScaleAlphaVec")),
            math_op=p.get("f32XdlMathOp", "Float"),
            activation_type=p.get("activationType", "None"),
            activation_compute_type=p.get("activationComputeDataType", p["computeType"]),
            activation_no_guard=bool(p.get("activationNoGuard")),
            mx_block_a=p.get("mxBlockA", 0),
            mx_block_b=p.get("mxBlockB", 0),
            mx_type_a=p.get("mxTypeA", "E8"),
            mx_type_b=p.get("mxTypeB", "E8"),
            mx_scale_format=p.get("mxScaleFormat", 0),
        )
        if p["operationIdentifier"] != profile.operation_identifier:
            raise ValueError("not an ordinary Cijk GEMM")
        if profile.bias_source not in {0, 1, 3} or profile.bias_type not in _TYPE_INFO:
            raise ValueError("unsupported bias profile")
        if any(t not in _TYPE_INFO for t in (profile.compute_input_a, profile.compute_input_b)):
            raise ValueError("unsupported compute input types")
        for block, scale_type in (
            (profile.mx_block_a, profile.mx_type_a),
            (profile.mx_block_b, profile.mx_type_b),
        ):
            if type(block) is not int or block not in {0, 16, 32}:
                raise ValueError(f"unsupported MX block size: {block}")
            if block and scale_type not in _MX_SCALE_TYPES:
                raise ValueError(f"unsupported MX scale type: {scale_type}")
        if profile.has_mx and profile.mx_scale_format not in {0, 1, 2}:
            raise ValueError(f"unsupported MX scale layout: {profile.mx_scale_format}")
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
            "mxBlockA": self.mx_block_a,
            "mxBlockB": self.mx_block_b,
            "mxTypeA": self.mx_type_a,
            "mxTypeB": self.mx_type_b,
            "mxScaleFormat": self.mx_scale_format,
        }

    @property
    def has_mx(self) -> bool:
        return bool(self.mx_block_a or self.mx_block_b)

    @property
    def packing_multiples(self) -> tuple[tuple[str, int], ...]:
        """Keep each stored column and batch aligned to whole packed containers."""
        return (
            ("k" if self.trans_a else "m", _TYPE_INFO[self.a_type][1]),
            ("n" if self.trans_b else "k", _TYPE_INFO[self.b_type][1]),
        )

    def bench_scale(self, tensor: str, target: str) -> str | int:
        """Translate the operand's block/type/layout to a native bench scale mode."""
        block, scale_type = (
            (self.mx_block_a, self.mx_type_a)
            if tensor == "a"
            else (self.mx_block_b, self.mx_type_b)
        )
        if not block:
            return self.scale_ab or "none"
        if self.mx_scale_format == 1:
            if target != "gfx950" or (block, scale_type) != (32, "E8"):
                raise ValueError("Bench host-preswizzled MX scales require gfx950, block 32, E8")
            return 1001  # Block_32_UE8M0_32_8_EXT
        if self.mx_scale_format == 0 and target.startswith("gfx1250"):
            raise ValueError(
                "Bench always swizzles gfx1250 MX scales; use tensile_ini for NoSwizzle"
            )
        return _MX_SCALE_TYPES[scale_type][block]

    def data_bytes(self, shape: GemmShape) -> int:
        """Combined A/B/C/D storage, excluding workspace, bias and scale buffers."""
        m, n, k = shape.m, shape.n, shape.k
        elements = (m * k, n * k, m * n, m * n)
        types = (self.a_type, self.b_type, self.c_type, self.d_type)
        return shape.batch_count * sum(
            ((count + packing - 1) // packing) * size
            for count, (size, packing, _) in zip(elements, (_TYPE_INFO[t] for t in types))
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
    target: str
    shape: GemmShape
    profile: GemmProfile
    id: str
    reason: str
    # Selected runtime values, separate from the solution's allowed profile.
    # The initial policy chooses these defaults; formatters consume the values.
    alpha: float = 1
    beta: float = 0
    activation: str = "None"

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
