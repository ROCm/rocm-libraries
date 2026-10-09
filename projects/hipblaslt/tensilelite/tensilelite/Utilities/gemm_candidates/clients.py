# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Small, pure format adapters for native numerical validation."""

from collections.abc import Iterable
from pathlib import Path

import yaml

from ..ClientConfig import problemTypeOptions
from .models import GemmCase, bench_type


def bench_yaml(cases: Iterable[GemmCase], *, workspace_bytes: int = 128 * 1024 * 1024) -> str:
    """Format cases for ``hipblaslt-bench --yaml cases.yaml``.

    Each row explicitly selects its solution and enables numerical validation.
    Rows may mix solutions and profiles for one target in a library installation.
    Set ``HIPBLASLT_TENSILE_LIBPATH`` when running bench against that library.
    Neither the library nor the code objects are opened by this formatter.
    """
    rows = []
    library = None
    for case in cases:
        source = (case.metadata_file.parent, case.target)
        if library is not None and library != source:
            raise ValueError("Bench cases must come from the same library directory and target")
        library = source
        p, s = case.profile, case.shape
        row = {
            "function": "matmul",
            "algo_method": 2,
            "solution_index": case.solution_index,
            "requested_solution_num": 1,
            "M": s.m,
            "N": s.n,
            "K": s.k,
            "batch_count": s.batch_count,
            "transA": "T" if p.trans_a else "N",
            "transB": "T" if p.trans_b else "N",
            "a_type": bench_type(p.a_type),
            "b_type": bench_type(p.b_type),
            "c_type": bench_type(p.c_type),
            "d_type": bench_type(p.d_type),
            "compute_type": "c_i32_r" if p.compute_type == "Int32" else "c_f32_r",
            "scale_type": bench_type(p.compute_type),
            "alpha": case.alpha,
            "beta": case.beta,
            "initialization": "rand_int",
            "bias_vector": p.use_bias,
            "bias_type": bench_type(p.bias_type),
            "bias_source": {0: "a", 1: "b", 3: "d"}[p.bias_source],
            "scaleA": p.bench_scale("a", case.target),
            "scaleB": p.bench_scale("b", case.target),
            "scaleC": p.scale_cd,
            "scaleD": p.scale_cd,
            "scaleAlpha_vector": p.scale_alpha_vector,
            "activation_type": case.activation.lower(),
            "norm_check": 1,
            "norm_check_assert": True,
            "allclose_check": 1,
            "iters": 1,
            "cold_iters": 0,
            "use_gpu_timer": False,
            "print_kernel_info": True,
            "user_allocated_workspace": workspace_bytes,
            "skip_slow_solution_ratio": 0.0,
            "adaptive": False,
        }
        # Bench accepts only floating-point compute-input overrides. Integer
        # computation is selected through compute_type instead.
        if p.compute_type != "Int32":
            row["compute_input_typeA"] = bench_type(p.compute_input_a)
            row["compute_input_typeB"] = bench_type(p.compute_input_b)
        if p.math_op == "XFloat32":
            row["compute_type"] = "c_xf32_r"
        row.update(zip(("lda", "ldb", "ldc", "ldd"), case.leading_dimensions))
        row.update(zip(("stride_a", "stride_b", "stride_c", "stride_d"), case.batch_strides))
        rows.append(row)
    return yaml.dump(rows, Dumper=getattr(yaml, "CSafeDumper", yaml.SafeDumper), sort_keys=False)


def tensile_ini(
    cases: Iterable[GemmCase],
    *,
    results_file: str | Path = "results.csv",
    workspace_bytes: int = 128 * 1024 * 1024,
) -> str:
    """Format one solution's cases for ``tensilelite-client --config-file``.

    All cases must have the same solution, profile and library files. Mixing
    them raises ``ValueError``: the client applies common options to every
    problem, so combining unrelated solutions would create a cross product.
    This enables full numerical validation with one enqueue per case and no
    timing sweep. Code objects are needed at execution time, not formatting.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("A Tensile configuration needs at least one case")
    first = cases[0]

    def identity(case):
        return (
            case.solution_index,
            case.solution_name,
            case.kernel_name,
            case.profile,
            case.target,
            case.metadata_file,
            case.code_objects,
            case.alpha,
            case.beta,
            case.activation,
        )

    if any(identity(case) != identity(first) for case in cases[1:]):
        raise ValueError(
            "A Tensile configuration must contain only one solution and runtime profile"
        )
    p = first.profile
    # The client exposes scalar initialization modes rather than arbitrary
    # scalar literals. Keep this translation out of the selection policy.
    scalar_modes = {0: "Zero", 1: "One", 2: "Two"}
    if first.alpha not in scalar_modes or first.beta not in scalar_modes:
        raise ValueError("Tensile scalar initialization currently supports only 0, 1 and 2")
    options = {
        "library-file": first.metadata_file,
        "results-file": results_file,
        "solution-start-idx": first.solution_index,
        "num-solutions": 1,
        **problemTypeOptions(p.as_problem_type()),
        "bias-type-args": p.bias_type,
        "activation-enum-args": first.activation,
        "activation-additional-args": "2.0,2.0",
        "init-seed": 1,  # Zero asks the client to seed from the clock.
        "init-a": "Random",
        "init-b": "Random",
        "init-c": "Zero",
        "init-d": "Zero",
        "init-alpha": scalar_modes[first.alpha],
        "init-beta": scalar_modes[first.beta],
        "init-bias": "Random",
        "init-scaleA": "Two",
        "init-scaleB": "Two",
        "init-scaleC": "Two",
        "init-scaleD": "Two",
        "init-scaleAlphaVec": "One",
        "c-equal-d": False,
        "num-elements-to-validate": -1,
        "num-benchmarks": 1,
        "num-warmups": 0,
        "num-enqueues-per-sync": 1,
        "max-enqueues-per-sync": 1,
        "num-syncs-per-benchmark": 0,
        "use-gpu-timer": False,
        "hardware-monitor": False,
        "sleep-percent": 0,
        "max-workspace-size": workspace_bytes,
        "granularity-threshold": 0.0,
        "print-max": 4,
        "log-level": "Debug",
        "use-user-args": False,
    }
    if p.has_mx:
        # Both clients use their existing MX generator to initialize data and
        # scales together. The client flag enables its architecture-specific
        # scale transform; the solution's mxScaleFormat selects the layout.
        options.update(
            {
                "init-mx-a": "Random",
                "init-mx-b": "Random",
                "mx-scale-format": int(p.mx_scale_format != 0),
            }
        )
    lines = [f"{key}={value}" for key, value in options.items()]
    lines += [f"code-object={path}" for path in first.code_objects]
    for case in cases:
        lines.append("problem-size=" + ",".join(map(str, case.shape.tensile_sizes)))
        for tensor, ld, stride in zip("abcd", case.leading_dimensions, case.batch_strides):
            lines.append(f"{tensor}-strides=1,{ld},{stride}")
    return "\n".join(lines) + "\n"
