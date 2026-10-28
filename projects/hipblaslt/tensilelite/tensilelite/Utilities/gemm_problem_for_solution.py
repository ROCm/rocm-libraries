# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Construct a small GEMM workload for a chosen library solution.

A solution describes a GPU kernel and the matrix layouts it expects. This module
uses that description to choose a small problem within its size constraints,
starting with one output tile and a specified number of reduction blocks, without
importing the kernel generator or using
an installed GPU. Execution and native solution support checks happen later.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import lcm
from pathlib import Path
from types import MappingProxyType

from ..ExecutionPolicy import normalize_execution_policy
from .BinaryLibraryIO import iterSolutionRecords
from .DataTypes import TYPE_PROPERTIES

# Older metadata can name both FP8 operands with a single ordered pair.
_INPUT_TYPE_PAIRS = {
    "Float8BFloat8": ("Float8", "BFloat8"),
    "BFloat8Float8": ("BFloat8", "Float8"),
    "Float8BFloat8_fnuz": ("Float8_fnuz", "BFloat8_fnuz"),
    "BFloat8Float8_fnuz": ("BFloat8_fnuz", "Float8_fnuz"),
}


class UnsupportedSolution(Exception):
    """A known feature for which this workload generator has no implementation.

    A sweep may catch this exception, record the index and reasons as a coverage
    gap, and continue. Invalid arguments or metadata are separate errors and must
    not be treated as unsupported solutions.
    """

    def __init__(self, solution_index: int, reasons: Iterable[str]):
        self.solution_index = solution_index
        self.reasons = tuple(reasons)
        super().__init__(
            f"Solution {solution_index}: unsupported features: {', '.join(self.reasons)}"
        )


def _size_predicates(predicate):
    """Retain dimension constraints without keeping unrelated native predicates."""
    if not isinstance(predicate, dict) or not isinstance(predicate.get("type"), str):
        raise ValueError("Invalid problem predicate")
    kind = predicate["type"]
    if kind in {"And", "Or", "Not"}:
        children = predicate["value"]
        if kind == "Not":
            children = [children]
        if not isinstance(children, list):
            raise ValueError(f"Invalid {kind} problem predicate")
        sizes = tuple(p for child in children for p in _size_predicates(child))
        return sizes if kind == "And" or not sizes else (predicate,)
    return (predicate,) if "Size" in kind or kind == "GlobalSplitUCheckMinK" else ()


def _choose_sizes(index, preferred, reduction_multiple, predicates):
    # Native GEMM problem sizes are ordered M, N, batch, K. Free and bound
    # predicate indices address their own lists, each with one entry here.
    lower, upper, multiples = [1] * 4, [None] * 4, [1] * 4
    dimension_types = {
        "Free0SizeMultiple": (0, {0}),
        "Free1SizeMultiple": (1, {0}),
        "BatchSizeMultiple": (2, {0}),
        "BatchSizeEqual": (2, {0}),
        "BoundSizeMultiple": (3, {0, -1}),
        "LeadingFree0SizesGreaterOrEqual": (0, None),
        "LeadingFree1SizesGreaterOrEqual": (1, None),
    }
    for predicate in predicates:
        kind, value = predicate["type"], predicate["value"]
        if kind == "GlobalSplitUCheckMinK":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(type(v) is not int for v in value)
                or value[0] < 0
                or value[1] < -1
            ):
                raise ValueError(f"Solution {index}: invalid {kind}")
            if value[1] > 1:
                lower[3] = max(lower[3], value[0] * value[1])
            continue
        if kind in dimension_types:
            dim, indices = dimension_types[kind]
            if indices is not None and (
                type(predicate.get("index")) is not int or predicate["index"] not in indices
            ):
                raise ValueError(f"Solution {index}: invalid {kind} index")
        elif kind in {
            "SizeEqual",
            "ProblemSizeEqual",
            "SizeGreaterThan",
            "SizeLessThan",
            "SizeMultiple",
            "SizeInRange",
        }:
            dim = predicate.get("index")
            if type(dim) is not int or not 0 <= dim < 4:
                raise ValueError(f"Solution {index}: invalid {kind} index")
        else:
            raise UnsupportedSolution(index, [f"problem size predicate {kind}"])
        if kind == "SizeInRange":
            if not isinstance(value, dict):
                raise ValueError(f"Solution {index}: invalid {kind}")
            lo, hi = value.get("min", 0), value.get("max", (1 << 64) - 1)
            if any(type(v) is not int or v < 0 for v in (lo, hi)):
                raise ValueError(f"Solution {index}: invalid {kind}")
            lower[dim] = max(lower[dim], lo)
            upper[dim] = hi - 1 if upper[dim] is None else min(upper[dim], hi - 1)
            continue
        if type(value) is not int or value < 0 or (kind.endswith("Multiple") and value == 0):
            raise ValueError(f"Solution {index}: invalid {kind} value")
        if kind.endswith("Multiple"):
            multiples[dim] = lcm(multiples[dim], value)
        elif kind.endswith("Equal") and not kind.endswith("GreaterOrEqual"):
            lower[dim] = max(lower[dim], value)
            upper[dim] = value if upper[dim] is None else min(upper[dim], value)
        elif kind == "SizeLessThan":
            upper[dim] = value - 1 if upper[dim] is None else min(upper[dim], value - 1)
        else:
            lower[dim] = max(lower[dim], value + (kind == "SizeGreaterThan"))
    result = []
    for dim, target in enumerate(preferred):
        lo, hi, multiple = lower[dim], upper[dim], multiples[dim]
        first = ((lo + multiple - 1) // multiple) * multiple
        if hi is not None and first > hi:
            raise ValueError(f"Solution {index}: contradictory size predicates for dimension {dim}")
        if dim == 2 and (lo > 1 or (hi is not None and hi < 1) or multiple != 1):
            raise UnsupportedSolution(index, ["batch count other than one"])
        if dim == 3:
            multiple = lcm(multiple, reduction_multiple)
        chosen = ((max(lo, target) + multiple - 1) // multiple) * multiple
        if hi is not None and chosen > hi:
            chosen = (hi // multiple) * multiple
        if chosen < lo:
            raise UnsupportedSolution(index, ["size bounds exclude complete reduction blocks"])
        result.append(chosen)
    return result[0], result[1], result[3]


@dataclass(frozen=True, slots=True)
class GemmProblem:
    """A packed matrix multiplication and the information needed to select its kernel.

    The index, solution/kernel names, library directory and GPU target identify
    the chosen solution within one library build.

    Scalars are alpha=1 and beta=0, batch count is one, and optional epilogue
    features are disabled. Native execution still checks solution support.
    TODO(newling): Cover other scalar values, batches and optional output
    operations for broader race detection (AIHPBLAS-3605).
    MX block sizes of zero mean unscaled operands. MX scale format is 0 for
    ordinary scales, 1 for host-preswizzled gfx950 scales, or 2 for gfx1250 TDM.
    """

    solution_index: int
    solution_name: str
    kernel_name: str
    library_directory: Path
    target: str
    m: int
    n: int
    k: int
    a_type: str
    b_type: str
    c_type: str
    d_type: str
    compute_type: str
    compute_input_a: str
    compute_input_b: str
    trans_a: bool
    trans_b: bool
    math_op: str
    conjugate_a: bool = False
    conjugate_b: bool = False
    mx_block_a: int = 0
    mx_block_b: int = 0
    mx_type_a: str = "E8"
    mx_type_b: str = "E8"
    mx_scale_format: int = 0

    @property
    def leading_dimensions(self) -> tuple[int, int, int, int]:
        """Column strides, in elements, for the stored A/B/C/D matrices.

        Column-major address(row, col) = base + row + col * leading_dimension.
        Packed storage has no padding, so each stride equals the stored row
        count. Transposing A/B changes that row count; C/D are both M-by-N and
        have no transpose operation in the hipBLASLt matmul descriptor.
        """
        return (
            self.k if self.trans_a else self.m,
            self.n if self.trans_b else self.k,
            self.m,
            self.m,
        )

    @property
    def batch_strides(self) -> tuple[int, int, int, int]:
        return self.m * self.k, self.n * self.k, self.m * self.n, self.m * self.n


@dataclass(frozen=True, slots=True)
class LibraryMetadata:
    """Descriptions of the kernels in one built library.

    ``indices`` lists the retained solutions. ``choose_single_problem`` reuses
    their loaded records to construct a workload, without reading more files or
    using a GPU.
    Records retain construction metadata; native predicates are checked by the
    runtime against the matching device library.
    """

    directory: Path
    target: str
    indices: tuple[int, ...]
    _records: Mapping[int, dict] = field(repr=False)

    def choose_single_problem(
        self, solution_index: int, *, reduction_blocks: int = 8
    ) -> GemmProblem:
        """Choose one of the matrix multiplications this solution may support.

        This choice starts with one full output tile and honors serialized size
        bounds and divisibility constraints; a solution can accept other sizes.
        M and N are the dimensions of the output tile computed by one workgroup.
        K is ``reduction_blocks * DepthU * fixed_split``: each split workgroup
        processes ``reduction_blocks`` full blocks of DepthU reduction elements.
        Persistent scheduling (Stream-K or DataParallel) uses no fixed split
        multiplier; the runtime distributes those blocks among workgroups.
        Bounds can change M/N or the number of complete reduction blocks. Blocks
        include prefetch and final stages; loop backedges can differ.

        Raise UnsupportedSolution for explicitly unimplemented features. Invalid
        indices, missing fields and malformed metadata remain ordinary errors.
        """
        if type(solution_index) is not int or solution_index < 0:
            raise ValueError("solution_index must be a nonnegative integer")
        if type(reduction_blocks) is not int or reduction_blocks <= 0:
            raise ValueError("reduction_blocks must be a positive integer")
        record = self._records[solution_index]
        p, mapping = record["problemType"], record["sizeMapping"]
        m, n = mapping["macroTile"][:2]
        depth, split = mapping["depthU"], mapping["globalSplitU"]
        if any(type(value) is not int or value <= 0 for value in (m, n, depth)):
            raise ValueError(
                f"Solution {solution_index}: tile and DepthU must be positive integers"
            )
        policy_fields = (
            ("tileProcessingStrategy", "TileProcessingStrategy"),
            ("workAssignment", "WorkAssignment"),
            ("streamK", "StreamK"),
            ("streamKForceDPOnly", "StreamKForceDPOnly"),
            ("streamKAtomic", "StreamKAtomic"),
        )
        policy = {name: mapping[key] for key, name in policy_fields if key in mapping}
        try:
            strategy = normalize_execution_policy(policy, regenerate=False)[
                "TileProcessingStrategy"
            ]
        except ValueError as error:
            raise ValueError(f"Solution {solution_index}: {error}") from error
        # Persistent scheduling disables fixed split-K with zero, which remains
        # invalid for an ordinary solution.
        if type(split) is not int or split < -1 or (split == 0 and strategy == "None"):
            raise ValueError(f"Solution {solution_index}: invalid globalSplitU {split!r}")
        reasons = []
        if split == -1 or mapping.get("adaptiveGemmGSUA", 0):
            reasons.append("automatic split-K")
        # TODO(newling): Add grouped and sparse GEMM, required output operations
        # and the excluded matrix layouts (AIHPBLAS-3605).
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
        reasons.extend(key for key in unsupported if p.get(key))
        if not p.get("stridedBatched", True):
            reasons.append("non-strided batching")
        activation = p.get("activationType", "None")
        if activation not in {"None", "All", "Hipblaslt_all"}:
            reasons.append(f"fixed activation {activation}")
        if reasons:
            raise UnsupportedSolution(solution_index, reasons)
        mx_blocks = tuple(p.get(f"mxBlock{operand}", 0) for operand in "AB")
        if any(type(block) is not int or block < 0 for block in mx_blocks):
            raise ValueError(
                f"Solution {solution_index}: MX block sizes must be nonnegative integers"
            )
        mx_types = tuple(p.get(f"mxType{operand}", "E8") for operand in "AB")
        mx_format = p.get("mxScaleFormat", 0)
        if type(mx_format) is not int or mx_format not in (0, 1, 2):
            raise ValueError(f"Solution {solution_index}: invalid mxScaleFormat {mx_format!r}")
        types = {key: p[f"{key}Type"] for key in "abcd"}
        compute_a = p.get("computeInputTypeA", p.get("computeInputType", types["a"]))
        compute_b = p.get("computeInputTypeB", p.get("computeInputType", types["b"]))
        scaled_types = [dtype for block, dtype in zip(mx_blocks, mx_types) if block]
        for dtype in (*types.values(), compute_a, compute_b, p["computeType"], *scaled_types):
            if dtype not in TYPE_PROPERTIES:
                raise ValueError(f"Solution {solution_index}: unknown metadata datatype {dtype!r}")
        compute_a = _INPUT_TYPE_PAIRS.get(compute_a, (compute_a, compute_a))[0]
        compute_b = _INPUT_TYPE_PAIRS.get(compute_b, (compute_b, compute_b))[1]
        if types["c"] != types["d"]:
            raise UnsupportedSolution(solution_index, ["different C/D datatypes"])
        a_indices = "lik" if p["transA"] else "ilk"
        b_indices = "jlk" if p["transB"] else "ljk"
        operation = p["operationIdentifier"]
        # Complex conjugation is encoded by a C suffix on the A/B index names.
        conjugate_a = f"_A{a_indices}C_" in operation
        conjugate_b = f"_B{b_indices}C_" in operation
        a_op, b_op = "C" if conjugate_a else "", "C" if conjugate_b else ""
        if operation != f"Contraction_l_A{a_indices}{a_op}_B{b_indices}{b_op}_Cijk_Dijk":
            raise UnsupportedSolution(solution_index, [f"contraction {operation}"])
        reduction_multiple = lcm(
            depth * (split if strategy == "None" else 1), *(b for b in mx_blocks if b)
        )
        m, n, k = _choose_sizes(
            solution_index,
            (m, n, 1, reduction_blocks * depth * (split if strategy == "None" else 1)),
            reduction_multiple,
            record.get("sizePredicates", ()),
        )
        return GemmProblem(
            solution_index,
            record["name"],
            record["kernelName"],
            self.directory,
            self.target,
            m,
            n,
            k,
            types["a"],
            types["b"],
            types["c"],
            types["d"],
            p["computeType"],
            compute_a,
            compute_b,
            bool(p["transA"]),
            bool(p["transB"]),
            p.get("f32XdlMathOp", "Float"),
            conjugate_a,
            conjugate_b,
            mx_block_a=mx_blocks[0],
            mx_block_b=mx_blocks[1],
            mx_type_a=mx_types[0],
            mx_type_b=mx_types[1],
            mx_scale_format=mx_format,
        )


def load_library_metadata(
    library_dir: str | Path, *, target: str, solution_indices: Iterable[int] | None = None
) -> LibraryMetadata:
    """Read the solution descriptions for one GPU target from a built library.

    The build writes binary .dat files containing kernel descriptions and
    selection tables; .dat.zlib is the same data compressed with zlib. The older
    (legacy) format stores descriptions in an array; the indexed format adds an
    index-to-record lookup table. The shared reader handles both formats.

    Read each matching file once and retain all solutions, or just the requested
    indices. Compressed copies take precedence over uncompressed copies. Indices
    belong to this particular build; duplicates and missing requested indices are
    errors. Selecting one index still reads/decompresses every matching file.
    """
    directory = Path(library_dir).resolve()
    if (directory / target).is_dir():
        directory /= target
    wanted = None
    if solution_indices is not None:
        wanted = set()
        for index in solution_indices:
            if type(index) is not int or index < 0:
                raise ValueError("Solution indices must be nonnegative integers")
            wanted.add(index)
    shards = {
        path.name.removesuffix(".zlib"): path
        for path in sorted(directory.glob(f"*_{target}.dat*"))
        if path.name.endswith((".dat", ".dat.zlib"))
    }
    if not shards:
        raise ValueError(f"No {target} solution metadata in {directory}")
    records, seen = {}, set()
    for path in shards.values():
        for index, record in iterSolutionRecords(path, wanted):
            if index in seen:
                raise ValueError(f"{path}: duplicate solution index {index}")
            seen.add(index)
            if record is not None:
                # Native predicates and custom-kernel data can dominate catalog
                # memory. Retain construction fields and dimension constraints.
                records[index] = {
                    key: record[key]
                    for key in ("name", "kernelName", "problemType", "sizeMapping")
                    if key in record
                }
                records[index]["sizePredicates"] = _size_predicates(
                    record.get("problemPredicate", {"type": "TruePred"})
                )
    if wanted is not None and wanted - seen:
        raise KeyError(f"Solution indices not found: {sorted(wanted - seen)}")
    return LibraryMetadata(directory, target, tuple(sorted(records)), MappingProxyType(records))
