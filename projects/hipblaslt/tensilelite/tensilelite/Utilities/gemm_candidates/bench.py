# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Format candidate cases for hipblaslt-bench numerical validation."""

from collections.abc import Iterable

import yaml

from .models import _BENCH_ACTIVATIONS, GemmCase, bench_type


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
        if case.activation not in _BENCH_ACTIVATIONS:
            raise ValueError(f"Bench cannot represent activation {case.activation!r}")
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
            "activation_type": _BENCH_ACTIVATIONS[case.activation],
            "activation_arg1": case.activation_args[0],
            "activation_arg2": case.activation_args[1],
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
