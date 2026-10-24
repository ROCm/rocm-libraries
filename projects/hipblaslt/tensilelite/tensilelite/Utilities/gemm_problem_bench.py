# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Translate generated GEMM descriptors into hipblaslt-bench YAML input."""

from collections.abc import Iterable

import yaml

from .gemm_problem_for_solution import GemmProblem

# Metadata names -> hipblaslt-bench storage-type spellings. This translation
# belongs to the client adapter; it does not restrict problem generation.
_BENCH_DATATYPE_NAMES = {
    "Float": "f32_r",
    "Double": "f64_r",
    "ComplexFloat": "f32_c",
    "ComplexDouble": "f64_c",
    "Half": "f16_r",
    "BFloat16": "bf16_r",
    "Float8": "f8_r",
    "BFloat8": "bf8_r",
    "Float8_fnuz": "f8_fnuz_r",
    "BFloat8_fnuz": "bf8_fnuz_r",
    "Int8": "i8_r",
    "Int32": "i32_r",
    "Float4": "f4_r",
    "Float6": "f6_r",
    "BFloat6": "bf6_r",
    "E8": "e8_r",
    "E5M3": "e5m3_r",
}
_COMPUTE_NAMES = {
    "Float": "c_f32_r",
    "Double": "c_f64_r",
    "Int32": "c_i32_r",
    "ComplexFloat": "c_f32_r",
    "ComplexDouble": "c_f64_r",
}
# Values belong to the client's hipblaslt_scaling_format, not the public API enum.
_MX_SCALE_MODES = {
    (32, "E8"): 3,
    (16, "E8"): 4,
    (32, "Float8"): 5,
    (16, "Float8"): 6,
    (32, "E5M3"): 7,
    (16, "E5M3"): 8,
}


class UnsupportedBenchProblem(Exception):
    """The descriptor is valid but this bench adapter cannot express it."""

    def __init__(self, solution_index: int, reason: str):
        self.solution_index = solution_index
        self.reason = reason
        super().__init__(f"Solution {solution_index}: bench adapter: {reason}")


def _datatype_name(problem: GemmProblem, dtype: str) -> str:
    try:
        return _BENCH_DATATYPE_NAMES[dtype]
    except KeyError:
        raise UnsupportedBenchProblem(problem.solution_index, f"datatype {dtype}") from None


def _mx_scale_mode(problem: GemmProblem, operand: str) -> int | str:
    block = getattr(problem, f"mx_block_{operand}")
    if not block:
        return "none"
    dtype = getattr(problem, f"mx_type_{operand}")
    layout = problem.mx_scale_format
    if layout == 1:
        # The gfx950 preswizzled client layout requires A transposed and B normal.
        if (block, dtype) == (32, "E8") and getattr(problem, f"trans_{operand}") == (
            operand == "a"
        ):
            return 1001
    elif layout in (0, 2) and (block, dtype) in _MX_SCALE_MODES:
        # The client prepares the TDM scale layout for gfx1250 automatically.
        if (layout == 2) == problem.target.startswith("gfx1250"):
            return _MX_SCALE_MODES[block, dtype]
    raise UnsupportedBenchProblem(
        problem.solution_index,
        f"MX {operand.upper()} block={block}, type={dtype}, layout={layout}, target={problem.target}",
    )


def bench_yaml(problems: Iterable[GemmProblem]) -> str:
    """Format one validation-only bench execution per problem, selecting its index.

    The one execution comes from numerical validation. ``iters=0`` disables
    timed launches and ``cold_iters=0`` leaves just the validation launch.
    Run with HIPBLASLT_TENSILE_LIBPATH set to the problems' library directory.
    Split-K solutions may additionally launch reduction/helper kernels.
    """
    rows, source = [], None
    for problem in problems:
        if problem.compute_type not in _COMPUTE_NAMES:
            raise UnsupportedBenchProblem(
                problem.solution_index, f"compute type {problem.compute_type}"
            )
        if problem.math_op not in {"Float", "XFloat32"}:
            raise UnsupportedBenchProblem(
                problem.solution_index, f"math operation {problem.math_op}"
            )
        if (problem.conjugate_a and not problem.trans_a) or (
            problem.conjugate_b and not problem.trans_b
        ):
            raise UnsupportedBenchProblem(problem.solution_index, "conjugation without transpose")
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
            "transA": "C" if problem.conjugate_a else "T" if problem.trans_a else "N",
            "transB": "C" if problem.conjugate_b else "T" if problem.trans_b else "N",
            "a_type": _datatype_name(problem, problem.a_type),
            "b_type": _datatype_name(problem, problem.b_type),
            "c_type": _datatype_name(problem, problem.c_type),
            "d_type": _datatype_name(problem, problem.d_type),
            "compute_type": _COMPUTE_NAMES[problem.compute_type],
            "scale_type": _datatype_name(problem, problem.compute_type),
            "alpha": 1,
            "beta": 0,
            # Bounded MX inputs avoid overflowing FP16 outputs at large DepthU.
            "initialization": "hpl" if problem.mx_block_a or problem.mx_block_b else "rand_int",
            "bias_vector": False,
            "activation_type": "none",
            "scaleA": _mx_scale_mode(problem, "a"),
            "scaleB": _mx_scale_mode(problem, "b"),
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
        # Bench's explicit compute-input fields accept narrow/FP32 inputs only.
        # Its compute enum supplies the real component type for double, complex
        # and integer computation when these fields are omitted.
        if problem.compute_type == "Float":
            row["compute_input_typeA"] = _datatype_name(problem, problem.compute_input_a)
            row["compute_input_typeB"] = _datatype_name(problem, problem.compute_input_b)
        if problem.math_op == "XFloat32":
            row["compute_type"] = "c_xf32_r"
        row.update(zip(("lda", "ldb", "ldc", "ldd"), problem.leading_dimensions))
        row.update(zip(("stride_a", "stride_b", "stride_c", "stride_d"), problem.batch_strides))
        rows.append(row)
    return yaml.dump(rows, Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper), sort_keys=False)
