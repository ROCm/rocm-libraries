# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A bounded, deterministic set of GEMMs around a solution's size boundaries."""

from dataclasses import dataclass
from math import lcm
from pathlib import Path

from .metadata import LibraryMetadata, _Solution

_TYPE_BYTES = {
    "Float": 4,
    "Half": 2,
    "BFloat16": 2,
    "Float8": 1,
    "BFloat8": 1,
    "Float8_fnuz": 1,
    "BFloat8_fnuz": 1,
    "Int8": 1,
    "Int32": 4,
}


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


def _profile(solution: _Solution) -> GemmProfile:
    p = solution.problem
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
        raise ValueError(f"Solution {solution.index}: unsupported features: {', '.join(features)}")
    if (
        any(p[f"{t}Type"] not in _TYPE_BYTES for t in "abcd")
        or p["cType"] != p["dType"]
        or p["computeType"] not in {"Float", "Int32"}
        or not p.get("stridedBatched", True)
        or p.get("activationType", "None") not in {"None", "All", "Hipblaslt_all"}
        or p.get("useScaleAB", "") not in {"", "Scalar", "Vector"}
        or p.get("useBias", 0) not in {0, 1}
        or p.get("useScaleAlphaVec", 0) not in {0, 1}
        or p.get("f32XdlMathOp", "Float") not in {"Float", "XFloat32"}
    ):
        raise ValueError(f"Solution {solution.index}: unsupported GEMM profile")
    profile = GemmProfile(
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
        raise ValueError(f"Solution {solution.index}: not an ordinary Cijk GEMM")
    if profile.bias_source not in {0, 1, 3} or profile.bias_type not in _TYPE_BYTES:
        raise ValueError(f"Solution {solution.index}: unsupported bias profile")
    if any(t not in _TYPE_BYTES for t in (profile.compute_input_a, profile.compute_input_b)):
        raise ValueError(f"Solution {solution.index}: unsupported compute input types")
    return profile


@dataclass
class _Dimension:
    minimum: int = 1
    maximum: int = 8192
    multiple: int = 1

    def at_least(self, value: int, *, multiple: int = 1) -> int | None:
        step = lcm(self.multiple, multiple)
        value = max(value, self.minimum)
        value = (value + step - 1) // step * step
        return value if value <= self.maximum else None

    def remainder_at_least(self, value: int, *, divisor: int, multiple: int = 1) -> int | None:
        """First legal value at or above ``value`` not divisible by ``divisor``."""
        step = lcm(self.multiple, multiple)
        if step % divisor == 0:
            return None
        result = self.at_least(value, multiple=multiple)
        if result is not None and result % divisor == 0:
            result = self.at_least(result + step, multiple=multiple)
        return result


def _conjuncts(predicate):
    if predicate["type"] == "And":
        for child in predicate["value"]:
            yield from _conjuncts(child)
    else:
        # In particular, the children of Or/Not are not independent bounds.
        yield predicate


def _bounds(solution: _Solution, max_dimension: int) -> dict[str, _Dimension]:
    dimensions = {
        name: _Dimension(maximum=max_dimension) for name in ("m", "n", "batch_count", "k")
    }
    named = {
        "Free0SizeMultiple": ("m", "multiple"),
        "Free1SizeMultiple": ("n", "multiple"),
        "BatchSizeMultiple": ("batch_count", "multiple"),
        "BoundSizeMultiple": ("k", "multiple"),
        "BatchSizeEqual": ("batch_count", "equal"),
        "LeadingFree0SizesGreaterOrEqual": ("m", "minimum"),
        "LeadingFree1SizesGreaterOrEqual": ("n", "minimum"),
    }
    indexed = {
        "SizeMultiple": "multiple",
        "SizeEqual": "equal",
        "ProblemSizeEqual": "equal",
        "SizeGreaterThan": "greater",
        "SizeLessThan": "less",
        "SizeInRange": "range",
    }
    for predicate in _conjuncts(solution.predicate):
        kind, value, index = predicate["type"], predicate.get("value"), predicate.get("index", 0)
        if kind in named:
            name, operation = named[kind]
            if index not in ((0, -1) if kind == "BoundSizeMultiple" else (0,)):
                raise ValueError(f"Invalid ordinary GEMM dimension: {predicate}")
        elif kind in indexed:
            if index not in range(4):
                raise ValueError(f"Invalid ordinary GEMM dimension: {predicate}")
            name, operation = tuple(dimensions)[index], indexed[kind]
        elif kind == "GlobalSplitUCheckMinK":
            name, operation = "k", "minimum"
            value = value[0] * value[1] if value[1] > 1 else 1
        else:
            continue  # Coupled, disjunctive and hardware predicates remain native checks.
        bound = dimensions[name]
        if operation == "multiple":
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid size multiple: {predicate}")
            bound.multiple = lcm(bound.multiple, value)
        if operation in {"minimum", "greater", "equal", "range"}:
            lower = value["min"] if operation == "range" else value
            bound.minimum = max(bound.minimum, lower + (operation == "greater"))
        if operation in {"less", "equal", "range"}:
            upper = value["max"] if operation == "range" else value
            bound.maximum = min(bound.maximum, upper - (operation in {"less", "range"}))
    return dimensions


def candidate_gemms(
    library: LibraryMetadata,
    solution_index: int,
    *,
    max_cases: int = 10,
    max_dimension: int = 8192,
    max_data_bytes: int = 128 * 1024 * 1024,
) -> list[GemmCase]:
    """Return distinct candidate problems for one explicit solution index.

    The fixed priority is: minimum, short/long K reductions, K tail and split
    remainder, full/partial/multiple output tiles, then multiple batches.
    ``max_cases`` selects a prefix; it is a cap, not a promise to fill a quota.
    Constraints can make proposals coincide or exceed the dimension/data caps.

    Fixed split-K uses ``DepthU * GlobalSplitU`` as its reduction scale. For
    automatic split-K and Stream-K, the scale is DepthU alone: their actual
    per-workgroup work depends on the shape and hardware. No loop coverage or
    native support is inferred from a candidate's label.

    Matrices are packed and strided-batched. ``max_data_bytes`` bounds their
    combined A/B/C/D storage; workspace and optional bias/scale buffers are
    separate. Unsupported profiles and an empty feasible candidate set raise
    ``ValueError``. An index not loaded in ``library`` raises ``KeyError``.
    This function performs no file I/O, GPU calls or random sampling.
    """
    if type(solution_index) is not int or solution_index < 0:
        raise ValueError("solution_index must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 for n in (max_cases, max_dimension, max_data_bytes)):
        raise ValueError("Candidate limits must be positive integers")
    solution = library._solutions[solution_index]
    profile = _profile(solution)
    bounds = _bounds(solution, max_dimension)
    first = {name: bound.at_least(1) for name, bound in bounds.items()}
    if any(value is None for value in first.values()):
        raise ValueError(f"Solution {solution_index}: no positive shape within the dimension limit")

    tile_m, tile_n = solution.mapping["macroTile"][:2]
    depth, split = solution.mapping["depthU"], solution.mapping["globalSplitU"]
    if min(tile_m, tile_n, depth) <= 0:
        raise ValueError("Macro tile and DepthU must be positive")
    stream_k = solution.mapping.get("tileProcessingStrategy") == "StreamK"
    fixed_split = split > 1 and not stream_k
    scale = depth * split if fixed_split else depth
    scale_name = "DepthU × configured GSU" if fixed_split else "DepthU"
    cases, seen = [], set()

    def add(case_id, reason, **axes):
        if len(cases) == max_cases or any(value is None for value in axes.values()):
            return False
        fitted = {
            name: bound.at_least(axes.get(name, first[name])) for name, bound in bounds.items()
        }
        if any(value is None for value in fitted.values()):
            return False
        shape = GemmShape(**fitted)
        sizes = (shape.m * shape.k, shape.n * shape.k, shape.m * shape.n, shape.m * shape.n)
        types = (profile.a_type, profile.b_type, profile.c_type, profile.d_type)
        data_bytes = shape.batch_count * sum(n * _TYPE_BYTES[t] for n, t in zip(sizes, types))
        if shape in seen or data_bytes > max_data_bytes:
            return False
        seen.add(shape)
        cases.append(
            GemmCase(
                solution_index=solution.index,
                solution_name=solution.name,
                kernel_name=solution.kernel_name,
                metadata_file=solution.metadata_file,
                code_objects=solution.code_objects,
                shape=shape,
                profile=profile,
                id=case_id,
                reason=reason,
            )
        )
        return True

    add("minimum", "Smallest shape satisfying the extracted necessary dimensional bounds")
    if not cases:
        raise ValueError(f"Solution {solution_index}: the minimum candidate exceeds the data limit")

    k_bound = bounds["k"]
    # Start at the first legal full reduction block. A minimum above DepthU
    # must not collapse every short-K proposal onto the same shape.
    full_k = k_bound.at_least(scale, multiple=scale)
    if full_k is not None:
        step = lcm(scale, k_bound.multiple)
        for offset in range(3):
            k = full_k + offset * step
            add(f"k_{offset + 1}", f"K = {k // scale} × {scale_name}", k=k)
    for blocks in (8, 4):
        if add(
            "k_long",
            f"Longer reduction: {blocks} × the first full K block",
            k=blocks * full_k if full_k is not None else None,
        ):
            break

    tail_k = k_bound.remainder_at_least(2 * scale + 1, divisor=depth)
    add("k_tail", "K has a nonzero remainder modulo DepthU", k=tail_k)
    if fixed_split:
        uneven_k = k_bound.remainder_at_least(2 * scale + depth, divisor=scale, multiple=depth)
        add(
            "split_remainder",
            "Full DepthU blocks divide unevenly between split groups",
            k=uneven_k,
        )

    reference_k = k_bound.at_least(2 * scale) or first["k"]
    full_m = bounds["m"].at_least(tile_m, multiple=tile_m)
    full_n = bounds["n"].at_least(tile_n, multiple=tile_n)
    add("tile", "M and N contain only full macro tiles", m=full_m, n=full_n, k=reference_k)

    edge_m = bounds["m"].remainder_at_least(tile_m + 1, divisor=tile_m)
    edge_n = bounds["n"].remainder_at_least(tile_n + 1, divisor=tile_n)
    if edge_m is not None or edge_n is not None:
        add(
            "mn_edge",
            "A partial output tile, combined with a K tail when permitted",
            m=edge_m if edge_m is not None else (full_m or first["m"]),
            n=edge_n if edge_n is not None else (full_n or first["n"]),
            k=tail_k or reference_k,
        )
    add(
        "multi_tile",
        "Multiple output tiles in M and N",
        m=bounds["m"].at_least(2 * tile_m, multiple=tile_m),
        n=bounds["n"].at_least(3 * tile_n, multiple=tile_n),
        k=reference_k,
    )
    add(
        "batch",
        "More than one strided batch",
        batch_count=max(2, first["batch_count"]),
        k=reference_k,
    )
    return cases
