# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Construct a small GEMM workload for a chosen library solution.

A solution describes a GPU kernel and the matrix layouts it expects. This module
uses that description to choose one full output tile and a specified number of
reduction blocks per workgroup, without importing the kernel generator or using
an installed GPU. Execution and native solution support checks happen later.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

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


@dataclass(frozen=True, slots=True)
class GemmProblem:
    """A packed matrix multiplication and the information needed to select its kernel.

    The index, solution/kernel names, library directory and GPU target identify
    the chosen solution within one library build.

    Scalars are alpha=1 and beta=0, batch count is one, and optional epilogue
    features are disabled. Native execution still checks solution support.
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

    ``indices`` lists the retained solutions. ``get_problem`` reuses their loaded
    records to construct a workload, without reading more files or using a GPU.
    """

    directory: Path
    target: str
    indices: tuple[int, ...]
    _records: Mapping[int, dict] = field(repr=False)

    def get_problem(self, solution_index: int, *, reduction_blocks: int = 8) -> GemmProblem:
        """Choose one full output tile for this solution's matrix multiplication.

        M and N are the dimensions of the output tile computed by one workgroup.
        K is ``reduction_blocks * DepthU * fixed_split``: each split workgroup
        processes ``reduction_blocks`` full blocks of DepthU reduction elements.
        Blocks include prefetch and final stages; loop backedges can differ.

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
        strategy = mapping.get("tileProcessingStrategy", "None")
        # Persistent scheduling disables fixed split-K with zero. Its workload
        # is deferred, while zero remains invalid for an ordinary solution.
        if type(split) is not int or split < -1 or (split == 0 and strategy == "None"):
            raise ValueError(f"Solution {solution_index}: invalid globalSplitU {split!r}")
        reasons = []
        if strategy != "None":
            reasons.append(f"tileProcessingStrategy={strategy}")
        if split == -1 or mapping.get("adaptiveGemmGSUA", 0):
            reasons.append("automatic split-K")
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
            "mxBlockA",
            "mxBlockB",
        )
        reasons.extend(key for key in unsupported if p.get(key))
        if not p.get("stridedBatched", True):
            reasons.append("non-strided batching")
        activation = p.get("activationType", "None")
        if activation not in {"None", "All", "Hipblaslt_all"}:
            reasons.append(f"fixed activation {activation}")
        if reasons:
            raise UnsupportedSolution(solution_index, reasons)
        types = {key: p[f"{key}Type"] for key in "abcd"}
        compute_a = p.get("computeInputTypeA", p.get("computeInputType", types["a"]))
        compute_b = p.get("computeInputTypeB", p.get("computeInputType", types["b"]))
        for dtype in (*types.values(), compute_a, compute_b, p["computeType"]):
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
        return GemmProblem(
            solution_index,
            record["name"],
            record["kernelName"],
            self.directory,
            self.target,
            m,
            n,
            reduction_blocks * depth * split,
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
                records[index] = record
    if wanted is not None and wanted - seen:
        raise KeyError(f"Solution indices not found: {sorted(wanted - seen)}")
    return LibraryMetadata(directory, target, tuple(sorted(records)), MappingProxyType(records))
