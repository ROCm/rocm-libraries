# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One full output tile and eight reduction blocks per fixed split workgroup."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from .LibraryIO import iterSolutionRecords

_BENCH_TYPES = {
    "Float": "f32_r",
    "Half": "f16_r",
    "BFloat16": "bf16_r",
    "Float8": "f8_r",
    "BFloat8": "bf8_r",
    "Float8_fnuz": "f8_fnuz_r",
    "BFloat8_fnuz": "bf8_fnuz_r",
    "Int8": "i8_r",
    "Int32": "i32_r",
}


@dataclass(frozen=True, slots=True)
class GemmProblem:
    """A regular, packed GEMM and its identity in one library build.

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

    @property
    def leading_dimensions(self) -> tuple[int, int, int, int]:
        """Packed column-major leading dimensions in A/B/C/D order."""
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
    """Reusable solution records; ``get_problem`` performs no I/O or native calls."""

    directory: Path
    target: str
    indices: tuple[int, ...]
    _records: Mapping[int, dict] = field(repr=False)

    def get_problem(self, solution_index: int) -> GemmProblem:
        """Return M/N=macro tile, K=8*DepthU*fixed split, with no size search.

        Eight refers to full reduction blocks assigned to each workgroup,
        including blocks processed by the prefetch and final-loop stages.
        Automatic split-K and persistent/Stream-K scheduling are unsupported.
        """
        if type(solution_index) is not int or solution_index < 0:
            raise ValueError("solution_index must be a nonnegative integer")
        record = self._records[solution_index]
        p, mapping = record["problemType"], record["sizeMapping"]
        split = mapping["globalSplitU"]
        if (
            mapping.get("tileProcessingStrategy", "None") != "None"
            or split == -1
            or mapping.get("adaptiveGemmGSUA", 0)
        ):
            raise ValueError(f"Solution {solution_index}: requires ordinary fixed split-K")
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
        if any(p.get(key) for key in unsupported) or not p.get("stridedBatched", True):
            raise ValueError(
                f"Solution {solution_index}: requires ordinary strided GEMM without MX"
            )
        types = {key: p[f"{key}Type"] for key in "abcd"}
        compute_a = p.get("computeInputTypeA", p.get("computeInputType", types["a"]))
        compute_b = p.get("computeInputTypeB", p.get("computeInputType", types["b"]))
        if (
            any(t not in _BENCH_TYPES for t in (*types.values(), compute_a, compute_b))
            or types["c"] != types["d"]
            or p["computeType"] not in {"Float", "Int32"}
            or p.get("activationType", "None") not in {"None", "All", "Hipblaslt_all"}
            or p.get("f32XdlMathOp", "Float") not in {"Float", "XFloat32"}
        ):
            raise ValueError(f"Solution {solution_index}: unsupported GEMM types or activation")
        a_indices, b_indices = ("lik" if p["transA"] else "ilk"), ("jlk" if p["transB"] else "ljk")
        if p["operationIdentifier"] != f"Contraction_l_A{a_indices}_B{b_indices}_Cijk_Dijk":
            raise ValueError(f"Solution {solution_index}: not an ordinary Cijk GEMM")
        m, n = mapping["macroTile"][:2]
        depth = mapping["depthU"]
        if any(type(value) is not int or value <= 0 for value in (m, n, depth, split)):
            raise ValueError(
                f"Solution {solution_index}: tile, DepthU and split must be positive integers"
            )
        return GemmProblem(
            solution_index,
            record["name"],
            record["kernelName"],
            self.directory,
            self.target,
            m,
            n,
            8 * depth * split,
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
        )


def load_library_metadata(
    library_dir: str | Path, *, target: str, solution_indices: Iterable[int] | None = None
) -> LibraryMetadata:
    """Scan legacy/indexed .dat[.zlib] shards once and retain requested records.

    Indices belong to this particular build. Compressed copies take precedence;
    duplicate indices across shards and missing requested indices are errors.
    Even selective loading reads/decompresses every matching shard.
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


def bench_yaml(problems: Iterable[GemmProblem]) -> str:
    """Format one validation-only bench execution per problem, selecting its index.

    Run with HIPBLASLT_TENSILE_LIBPATH set to the problems' library directory.
    Split-K solutions may additionally launch reduction/helper kernels.
    """
    rows, source = [], None
    for problem in problems:
        identity = (problem.library_directory, problem.target)
        if source is not None and source != identity:
            raise ValueError("Problems must come from the same library directory and target")
        source = identity
        row = {
            "function": "matmul",
            "algo_method": 2,
            "solution_index": problem.solution_index,
            "requested_solution_num": 1,
            "M": problem.m,
            "N": problem.n,
            "K": problem.k,
            "batch_count": 1,
            "transA": "T" if problem.trans_a else "N",
            "transB": "T" if problem.trans_b else "N",
            "a_type": _BENCH_TYPES[problem.a_type],
            "b_type": _BENCH_TYPES[problem.b_type],
            "c_type": _BENCH_TYPES[problem.c_type],
            "d_type": _BENCH_TYPES[problem.d_type],
            "compute_type": "c_i32_r" if problem.compute_type == "Int32" else "c_f32_r",
            "scale_type": _BENCH_TYPES[problem.compute_type],
            "alpha": 1,
            "beta": 0,
            "initialization": "rand_int",
            "bias_vector": False,
            "activation_type": "none",
            "scaleA": "none",
            "scaleB": "none",
            "scaleC": False,
            "scaleD": False,
            "scaleAlpha_vector": False,
            "norm_check": 1,
            "norm_check_assert": True,
            "allclose_check": 1,
            "iters": 0,
            "cold_iters": 0,
            "adaptive": False,
            "use_gpu_timer": False,
            "print_kernel_info": True,
            "skip_slow_solution_ratio": 0.0,
            "user_allocated_workspace": 128 * 1024 * 1024,
        }
        if problem.compute_type != "Int32":
            row["compute_input_typeA"] = _BENCH_TYPES[problem.compute_input_a]
            row["compute_input_typeB"] = _BENCH_TYPES[problem.compute_input_b]
        if problem.math_op == "XFloat32":
            row["compute_type"] = "c_xf32_r"
        row.update(zip(("lda", "ldb", "ldc", "ldd"), problem.leading_dimensions))
        row.update(zip(("stride_a", "stride_b", "stride_c", "stride_d"), problem.batch_strides))
        rows.append(row)
    return yaml.dump(rows, Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper), sort_keys=False)
