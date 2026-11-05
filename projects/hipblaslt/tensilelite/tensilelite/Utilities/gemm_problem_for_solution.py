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
from math import isfinite, lcm
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

# Mirror ContractionSolution.hpp: the split-K synchronizer buffer holds
# GsuSynchronizerElements entries in each of SynchronizerGroupedSlots slots.
_SYNCHRONIZER_ELEMENTS = 409600 * 16

# An automatic split-K kernel (globalSplitU -1) lets the runtime choose the
# split as floor(K / DepthU / 3), bounded by the CU count. The chosen K spans
# this many splits of the requested reduction blocks.
AUTOMATIC_SPLIT_GROUPS = 2

# Serialized bias-source codes: the bias reduction of a gradient epilogue reads
# A, B or D. Code 2 (C) does not occur in shipped solutions.
_BIAS_SOURCES = {0: "A", 1: "B", 3: "D"}


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


def _activation_whitelist(predicate):
    """Activations a solution's predicate allows, or None when it does not restrict them."""
    if not isinstance(predicate, dict):
        return None
    if predicate.get("type") == "ActivationEnumWhiteList":
        return tuple(predicate["value"])
    if predicate.get("type") == "And":
        for child in predicate["value"]:
            allowed = _activation_whitelist(child)
            if allowed is not None:
                return allowed
    return None


def _choose_sizes(index, preferred, reduction_multiple, predicates):
    # Native GEMM problem sizes are ordered M, N, batch, K. Free and bound
    # predicate indices address their own lists, each with one entry here.
    lower, upper, multiples = [1] * 4, [None] * 4, [1] * 4
    synchronizer_checks = []
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
        if kind == "SynchronizerSizeCheck":
            if (
                not isinstance(value, list)
                or len(value) != 6
                or any(type(v) is not int for v in value)
                or min(value[:2]) <= 0
                or min(value[2:5]) < 0
            ):
                raise ValueError(f"Solution {index}: invalid {kind}")
            synchronizer_checks.append(value)
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
    for tile0, tile1, *per_tile, split in synchronizer_checks:
        # The runtime skips this check for unsplit or automatically split
        # launches; otherwise each output tile reserves synchronizer entries.
        tiles = -(-result[0] // tile0) * -(-result[1] // tile1)
        if split not in (-1, 1) and tiles * per_tile[0] * per_tile[1] * per_tile[2] > (
            _SYNCHRONIZER_ELEMENTS
        ):
            raise UnsupportedSolution(index, ["split-K synchronizer buffer too small"])
    return result[0], result[1], result[3]


def _required_epilogue(index, problem_type, allowed_activations):
    """GemmProblem epilogue fields for a solution's gradient, E and amaxD outputs.

    A gradient solution produces the bias gradient, so it needs a bias. Solutions
    with E are also given their bias, so that the same epilogue is exercised as
    for gradient solutions. E is used with ReLU: forward solutions write E, and
    gradient solutions read it and apply the ReLU derivative.
    """
    p = problem_type
    gradient, aux = bool(p.get("useGradient")), bool(p.get("useE"))
    epilogue = {"gradient": gradient, "amax_d": bool(p.get("outputAmaxD"))}
    if not (gradient or aux):
        return epilogue
    if not p.get("useBias"):
        raise UnsupportedSolution(index, ["gradient or E epilogue without bias support"])
    sources = p.get("biasSrcWhiteList", ())
    dtypes = p.get("biasDataTypeWhiteList", ())
    source = next((_BIAS_SOURCES[s] for s in sources if s in _BIAS_SOURCES), None)
    if source is None or not dtypes:
        raise UnsupportedSolution(index, [f"bias sources {list(sources)}, types {list(dtypes)}"])
    bias_type = "Float" if "Float" in dtypes else dtypes[0]
    if bias_type not in TYPE_PROPERTIES:
        raise ValueError(f"Solution {index}: unknown bias datatype {bias_type!r}")
    epilogue["bias"] = GemmBias(bias_type, source)
    if aux:
        aux_type = p.get("eType", p["dType"])
        if aux_type not in TYPE_PROPERTIES:
            raise ValueError(f"Solution {index}: unknown E datatype {aux_type!r}")
        required = "Drelu" if gradient else "Relu"
        if allowed_activations is not None and required not in allowed_activations:
            raise UnsupportedSolution(index, [f"activation {required} not allowed"])
        epilogue["aux_type"] = aux_type
        epilogue["activation"] = "Relu"
    return epilogue


@dataclass(frozen=True, slots=True)
class GemmBias:
    """A bias vector: its storage type and the matrix it belongs to (A, B or D).

    Without a gradient the bias is added to each output row. With a gradient it
    is an output: the bias gradient, reduced from the source matrix.
    """

    dtype: str
    source: str


@dataclass(frozen=True, slots=True)
class GemmProblem:
    """Packed matrix multiplications and the information needed to select their kernel.

    The index, solution/kernel names, library directory and GPU target identify
    the chosen solution within one library build.

    Scalars default to alpha=1 and beta=0, and batch count is one. The epilogue
    follows the solution's outputs: ``gradient`` computes a bias gradient;
    ``aux_type`` is the E tensor's type when the solution has E (written
    forward, read with a gradient), otherwise None; ``bias`` accompanies
    gradient and E, and grouped solutions that use one; ``activation`` is "Relu"
    with E, meaning its derivative when ``gradient`` is set; ``amax_d`` also
    returns the absolute maximum of D.
    Native execution still checks solution support.
    MX block sizes of zero mean unscaled operands. MX scale format is 0 for
    ordinary scales, 1 for host-preswizzled gfx950 scales, or 2 for gfx1250 TDM.
    ``grouped_members`` is None for a single GEMM, or the ordered (M, N, K)
    shapes in a grouped dispatch. The scalar dimensions describe its first member.
    Swizzle flags request packed device layouts; adapters rearrange the logical
    column-major inputs into those layouts before execution.
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
    alpha: int | float = 1
    beta: int | float = 0
    gradient: bool = False
    aux_type: str | None = None
    bias: GemmBias | None = None
    activation: str = "None"
    amax_d: bool = False
    grouped_members: tuple[tuple[int, int, int], ...] | None = None
    scale_ab: str | None = None
    swizzle_a: bool = False
    swizzle_b: bool = False

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
        self,
        solution_index: int,
        *,
        reduction_blocks: int = 8,
        alpha: int | float = 1,
        beta: int | float = 0,
    ) -> GemmProblem:
        """Choose one of the matrix multiplications this solution may support.

        This choice starts with one full output tile and honors serialized size
        bounds and divisibility constraints; a solution can accept other sizes.
        M and N are the dimensions of the output tile computed by one workgroup.
        K is ``reduction_blocks * DepthU * fixed_split``: each split workgroup
        processes ``reduction_blocks`` full blocks of DepthU reduction elements.
        Automatic split-K uses ``AUTOMATIC_SPLIT_GROUPS`` in place of the fixed
        split. The runtime then chooses floor(2 * reduction_blocks / 3) splits,
        at least one; three blocks give two splits of three blocks each.
        Persistent scheduling (Stream-K or DataParallel) uses no fixed split
        multiplier; the runtime distributes those blocks among workgroups.
        Bounds can change M/N or the number of complete reduction blocks. Blocks
        include prefetch and final stages; loop backedges can differ.

        Raise UnsupportedSolution for explicitly unimplemented features. Invalid
        indices, missing fields and malformed metadata remain ordinary errors.
        A grouped solution gets two members of the chosen shape with independent
        buffers, using its supported bias datatype and scalar input scaling.
        """
        if type(solution_index) is not int or solution_index < 0:
            raise ValueError("solution_index must be a nonnegative integer")
        if type(reduction_blocks) is not int or reduction_blocks <= 0:
            raise ValueError("reduction_blocks must be a positive integer")
        try:
            finite_scalars = all(
                type(value) in (int, float) and isfinite(value) for value in (alpha, beta)
            )
        except OverflowError:
            finite_scalars = False
        if not finite_scalars:
            raise ValueError("alpha and beta must be finite real numbers")
        record = self._records[solution_index]
        p, mapping = record["problemType"], record["sizeMapping"]
        if beta != 0 and not p.get("useBeta", True):
            raise UnsupportedSolution(solution_index, ["nonzero beta without beta support"])
        if p["computeType"] == "Int32" and any(
            int(value) != value or not -(1 << 31) <= value < (1 << 31) for value in (alpha, beta)
        ):
            raise UnsupportedSolution(solution_index, ["integer-compute scalar must fit Int32"])
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
        if mapping.get("adaptiveGemmGSUA", 0):
            reasons.append("adaptive split-K")
        # TODO(newling): Add sparse GEMM, gate residuals and the excluded matrix
        # layouts (AIHPBLAS-3605).
        unsupported = (
            "fusedGemmA2A",
            "sparse",
            "useGateResidual",
            "useInitialStridesAB",
            "useInitialStridesCD",
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
        epilogue = _required_epilogue(solution_index, p, record.get("activationWhiteList"))
        if strategy != "None":
            k_splits, fixed_split = 1, 1
        elif split == -1:
            k_splits, fixed_split = AUTOMATIC_SPLIT_GROUPS, 1
        else:
            k_splits, fixed_split = split, split
        reduction_multiple = lcm(depth * fixed_split, *(b for b in mx_blocks if b))
        m, n, k = _choose_sizes(
            solution_index,
            (m, n, 1, reduction_blocks * depth * k_splits),
            reduction_multiple,
            record.get("sizePredicates", ()),
        )
        grouped_members, scale_ab = None, None
        if p.get("groupedGemm"):
            # Two independent buffers exercise the grouped dispatch; each member
            # satisfies the same serialized size predicates as the first.
            grouped_members = ((m, n, k), (m, n, k))
            if p.get("useBias") and "bias" not in epilogue:
                if p["useBias"] != 1 or 3 not in p["biasSrcWhiteList"]:
                    raise UnsupportedSolution(solution_index, ["grouped bias outside D rows"])
                choices = p["biasDataTypeWhiteList"] or [p["computeType"]]
                bias_type = p["computeType"] if p["computeType"] in choices else choices[0]
                if bias_type not in TYPE_PROPERTIES:
                    raise ValueError(
                        f"Solution {solution_index}: unknown bias datatype {bias_type!r}"
                    )
                epilogue["bias"] = GemmBias(bias_type, "D")
            scale_ab = p.get("useScaleAB") or None
            if scale_ab not in {None, "Scalar"}:
                raise UnsupportedSolution(solution_index, [f"grouped input scaling {scale_ab}"])
        return GemmProblem(
            solution_index=solution_index,
            solution_name=record["name"],
            kernel_name=record["kernelName"],
            library_directory=self.directory,
            target=self.target,
            m=m,
            n=n,
            k=k,
            a_type=types["a"],
            b_type=types["b"],
            c_type=types["c"],
            d_type=types["d"],
            compute_type=p["computeType"],
            compute_input_a=compute_a,
            compute_input_b=compute_b,
            trans_a=bool(p["transA"]),
            trans_b=bool(p["transB"]),
            math_op=p.get("f32XdlMathOp", "Float"),
            conjugate_a=conjugate_a,
            conjugate_b=conjugate_b,
            mx_block_a=mx_blocks[0],
            mx_block_b=mx_blocks[1],
            mx_type_a=mx_types[0],
            mx_type_b=mx_types[1],
            mx_scale_format=mx_format,
            alpha=alpha,
            beta=beta,
            **epilogue,
            grouped_members=grouped_members,
            scale_ab=scale_ab,
            swizzle_a=bool(p.get("swizzleTensorA", False)),
            swizzle_b=bool(p.get("swizzleTensorB", False)),
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
                predicate = record.get("problemPredicate", {"type": "TruePred"})
                records[index]["sizePredicates"] = _size_predicates(predicate)
                records[index]["activationWhiteList"] = _activation_whitelist(predicate)
    if wanted is not None and wanted - seen:
        raise KeyError(f"Solution indices not found: {sorted(wanted - seen)}")
    return LibraryMetadata(directory, target, tuple(sorted(records)), MappingProxyType(records))
