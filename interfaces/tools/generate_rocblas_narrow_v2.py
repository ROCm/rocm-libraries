#!/usr/bin/env python3
"""Generate the complete rocBLAS public facade over the ten-call BLAS v2 protocol.

The generator deliberately contains operation-aware mappings. A callable that is
not in the checked categorization ledger, or a new primitive, is a hard error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CUSTOM = {
    "rocblas_create_handle",
    "rocblas_destroy_handle",
    "rocblas_get_pointer_mode",
    "rocblas_get_stream",
    "rocblas_set_pointer_mode",
    "rocblas_set_stream",
}
REQUEST = {
    "vector_transform": "rocm_blas_v2_vector_transform_request",
    "vector_reduce": "rocm_blas_v2_vector_reduce_request",
    "vector_index_reduce": "rocm_blas_v2_vector_reduce_request",
    "vector_rotate": "rocm_blas_v2_rotation_request",
    "rotation_parameters": "rocm_blas_v2_rotation_request",
    "vector_rotate_modified": "rocm_blas_v2_rotation_request",
    "modified_rotation_parameters": "rocm_blas_v2_rotation_request",
    "matrix_vector": "rocm_blas_v2_matrix_vector_request",
    "structured_matrix_vector": "rocm_blas_v2_matrix_vector_request",
    "triangular_vector": "rocm_blas_v2_matrix_vector_request",
    "rank_update": "rocm_blas_v2_rank_update_request",
    "structured_rank_update": "rocm_blas_v2_rank_update_request",
    "matmul": "rocm_blas_v2_matmul_request",
    "structured_matmul": "rocm_blas_v2_structured_matrix_request",
    "structured_rank_k": "rocm_blas_v2_structured_matrix_request",
    "triangular_matrix": "rocm_blas_v2_triangular_matrix_request",
    "triangular_inverse": "rocm_blas_v2_triangular_matrix_request",
    "matrix_transform": "rocm_blas_v2_matrix_transform_request",
    "diagonal_matrix_multiply": "rocm_blas_v2_matrix_transform_request",
}


def sig(f: dict) -> str:
    ps = [f"{p['type'].replace('_Bool', 'bool')} {p['name']}" for p in f["parameters"]]
    if f["variadic"]:
        ps.append("...")
    return ", ".join(ps) or "void"


def param_map(f: dict) -> dict[str, str]:
    return {p["name"].lower(): p["name"] for p in f["parameters"]}


def pick(m: dict[str, str], *names: str, default: str = "0") -> str:
    return next((m[n.lower()] for n in names if n.lower() in m), default)


def dtype(m: dict[str, str], pointer: str) -> str:
    explicit = pick(m, f"{pointer}_type", default="")
    return (
        explicit
        or f"rocm::interfaces::narrow_v2_pointer_type({pick(m, pointer, default='(float*)nullptr')})"
    )


def stride(m: dict[str, str], name: str) -> str:
    return pick(m, f"stride_{name}", f"stride{name}")


def batch(row: dict, m: dict[str, str]) -> tuple[str, str]:
    kind = {
        "single": "ROCM_BLAS_V2_BATCH_SINGLE",
        "pointer_array": "ROCM_BLAS_V2_BATCH_POINTER_ARRAY",
        "strided": "ROCM_BLAS_V2_BATCH_STRIDED",
    }.get(row["batch_kind"])
    if not kind:
        raise ValueError(f"unsupported batch kind {row['batch_kind']} in {row['name']}")
    return kind, pick(m, "batch_count", default="1")


def scalar(lines: list[str], field: str, m: dict[str, str], name: str) -> None:
    p = pick(m, name, default="nullptr")
    lines.append(
        f"  request.{field} = rocm::interfaces::narrow_v2_scalar(handle, {p}, {dtype(m, name)});"
    )


def vector(
    lines: list[str], field: str, m: dict[str, str], name: str, length: str
) -> None:
    p = pick(m, name, default="nullptr")
    inc = pick(m, f"inc{name}", default="1")
    lines.append(
        f"  request.{field} = rocm::interfaces::narrow_v2_vector({p}, {dtype(m, name)}, {length}, {inc}, {stride(m, name)});"
    )


def matrix(
    lines: list[str], field: str, m: dict[str, str], name: str, rows: str, cols: str
) -> None:
    p = pick(m, name, default="nullptr")
    ld = pick(m, f"ld{name}", f"ldinv{name}", default="0")
    lines.append(
        f"  request.{field} = rocm::interfaces::narrow_v2_matrix({p}, {dtype(m, name)}, {rows}, {cols}, {ld}, {stride(m, name)});"
    )


def populate(f: dict, row: dict) -> list[str]:
    m = param_map(f)
    prim = row["narrow_primitive"]
    op = row["operation"]
    typ = REQUEST[prim]
    width = (
        "ROCM_BLAS_V2_INDEX_64" if row["index_width"] == 64 else "ROCM_BLAS_V2_INDEX_32"
    )
    bk, bc = batch(row, m)
    n = pick(m, "n")
    mm = pick(m, "m", default=n)
    k = pick(m, "k")
    lines = [
        f"  {typ} request{{}};",
        "  request.header = rocm::interfaces::narrow_v2_header(sizeof(request));",
        f"  request.execution = rocm::interfaces::narrow_v2_execution(handle, {width}, {bk}, {bc});",
    ]
    if prim == "vector_transform":
        enum = {"scal": "SCALE", "copy": "COPY", "swap": "SWAP", "axpy": "AXPY"}[op]
        lines.append(f"  request.operation = ROCM_BLAS_V2_VECTOR_{enum};")
        lines.append(
            f"  request.compute_type = {pick(m,'execution_type',default='rocblas_datatype_invalid')};"
        )
        scalar(lines, "alpha", m, "alpha")
        vector(lines, "x", m, "x", n)
        vector(lines, "y", m, "y", n)
    elif prim in ("vector_reduce", "vector_index_reduce"):
        enum = {
            "dot": "DOT",
            "dotu": "DOT",
            "dotc": "DOT_CONJUGATE_X",
            "nrm2": "NORM_2",
            "asum": "ABSOLUTE_SUM",
            "iamax": "ABSOLUTE_MAX_INDEX",
            "iamin": "ABSOLUTE_MIN_INDEX",
        }[op]
        lines.append(f"  request.operation = ROCM_BLAS_V2_REDUCE_{enum};")
        lines.append(
            f"  request.compute_type = {pick(m,'execution_type',default='rocblas_datatype_invalid')};"
        )
        vector(lines, "x", m, "x", n)
        vector(lines, "y", m, "y", n)
        result = pick(m, "result", "results")
        lines += [
            f"  request.result = {result};",
            f"  request.result_type = {dtype(m, 'result')};",
            "  request.result_location = rocm::interfaces::narrow_v2_pointer_mode(handle);",
        ]
    elif prim in (
        "vector_rotate",
        "rotation_parameters",
        "vector_rotate_modified",
        "modified_rotation_parameters",
    ):
        enum = {
            "vector_rotate": "ROTATE",
            "rotation_parameters": "ROTATION_PARAMETERS",
            "vector_rotate_modified": "ROTATE_MODIFIED",
            "modified_rotation_parameters": "MODIFIED_ROTATION_PARAMETERS",
        }[prim]
        lines.append(f"  request.operation = ROCM_BLAS_V2_{enum};")
        lines.append(
            f"  request.compute_type = {pick(m,'execution_type',default='rocblas_datatype_invalid')};"
        )
        vector(lines, "x", m, "x", n)
        vector(lines, "y", m, "y", n)
        parameter_names = (
            ("d1", "d2", "x1", "y1")
            if prim == "modified_rotation_parameters"
            else (("c", "s") if prim == "vector_rotate" else ("a", "b", "c", "s"))
        )
        for i, name in enumerate(parameter_names):
            scalar(lines, f"parameters[{i}]", m, name)
        pp = pick(m, "param", default="nullptr")
        lines += [
            f"  request.parameter_block = rocm::interfaces::narrow_v2_memory({pp}, {stride(m,'param')});",
            f"  request.parameter_type = {dtype(m,'param')};",
        ]
    elif prim in ("matrix_vector", "structured_matrix_vector", "triangular_vector"):
        enum = "MATRIX_VECTOR_MULTIPLY"
        if prim == "triangular_vector":
            enum = (
                "TRIANGULAR_VECTOR_SOLVE"
                if op.endswith("sv")
                else "TRIANGULAR_VECTOR_MULTIPLY"
            )
        lines.append(f"  request.operation = ROCM_BLAS_V2_{enum};")
        lines.append(
            f"  request.transpose = {pick(m,'trans','transa',default='rocblas_operation_none')};"
        )
        scalar(lines, "alpha", m, "alpha")
        scalar(lines, "beta", m, "beta")
        matrix(lines, "matrix", m, "a", mm, n)
        vector(lines, "x", m, "x", n)
        vector(lines, "y", m, "y", mm)
        if prim == "structured_matrix_vector":
            kind = "HERMITIAN" if op.startswith("h") else "SYMMETRIC"
            lines.append(f"  request.matrix.kind = ROCM_BLAS_V2_MATRIX_{kind};")
        if op[1:2] == "p" or op.startswith(("sp", "hp", "tp")):
            lines.append("  request.matrix.storage = ROCM_BLAS_V2_STORAGE_PACKED;")
        if "bmv" in op or "bsv" in op:
            lines += [
                "  request.matrix.storage = ROCM_BLAS_V2_STORAGE_BANDED;",
                f"  request.matrix.upper_bandwidth = {k};",
            ]
        lines += [
            f"  request.matrix.fill = {pick(m,'uplo',default='rocblas_fill_full')};",
            f"  request.matrix.diagonal = {pick(m,'diag',default='rocblas_diagonal_non_unit')};",
        ]
    elif prim in ("rank_update", "structured_rank_update"):
        enum = (
            "RANK_TWO"
            if op.endswith("2")
            else ("RANK_ONE_CONJUGATE_Y" if op == "gerc" else "RANK_ONE")
        )
        lines.append(f"  request.operation = ROCM_BLAS_V2_{enum};")
        scalar(lines, "alpha", m, "alpha")
        vector(lines, "x", m, "x", n)
        vector(lines, "y", m, "y", n)
        matrix(lines, "matrix", m, "a", mm, n)
        if pick(m, "ap", default=""):
            matrix(lines, "matrix", m, "ap", n, n)
            lines.append("  request.matrix.storage = ROCM_BLAS_V2_STORAGE_PACKED;")
        if prim == "structured_rank_update":
            lines += [
                f"  request.matrix.fill = {pick(m,'uplo')};",
                f"  request.matrix.kind = ROCM_BLAS_V2_MATRIX_{'HERMITIAN' if op.startswith('h') else 'SYMMETRIC'};",
            ]
    elif prim == "matmul":
        lines += [
            f"  request.operation_a = {pick(m,'transa')};",
            f"  request.operation_b = {pick(m,'transb')};",
            f"  request.output_fill = {pick(m,'uplo',default='rocblas_fill_full')};",
            f"  request.compute_type = {pick(m,'compute_type',default=dtype(m,'c'))};",
        ]
        scalar(lines, "alpha", m, "alpha")
        scalar(lines, "beta", m, "beta")
        matrix(lines, "a", m, "a", mm, k)
        matrix(lines, "b", m, "b", k, n)
        matrix(lines, "c", m, "c", mm, n)
        matrix(lines, "d", m, "d", mm, n)
        lines += [
            f"  request.public_algorithm = {pick(m,'algo',default='rocblas_gemm_algo_standard')};",
            f"  request.public_solution_index = {pick(m,'solution_index')};",
            f"  request.public_flags = {pick(m,'flags')};",
        ]
    elif prim in ("structured_matmul", "structured_rank_k"):
        enum = (
            "STRUCTURED_MATMUL"
            if prim == "structured_matmul"
            else ("STRUCTURED_RANK_2K" if "2k" in op else "STRUCTURED_RANK_K")
        )
        if op.endswith("x"):
            enum = "STRUCTURED_RANK_K_EXTENDED"
        lines += [
            f"  request.operation = ROCM_BLAS_V2_{enum};",
            f"  request.side = {pick(m,'side',default='rocblas_side_left')};",
            f"  request.compute_type = {pick(m,'execution_type',default='rocblas_datatype_invalid')};",
            f"  request.operation_a = {pick(m,'trans','transa',default='rocblas_operation_none')};",
            f"  request.operation_b = {pick(m,'transb',default='rocblas_operation_none')};",
        ]
        scalar(lines, "alpha", m, "alpha")
        scalar(lines, "beta", m, "beta")
        matrix(lines, "a", m, "a", n, k)
        matrix(lines, "b", m, "b", k, n)
        matrix(lines, "c", m, "c", n, n)
        kind = "HERMITIAN" if op.startswith(("he", "her")) else "SYMMETRIC"
        lines += [
            f"  request.c.kind = ROCM_BLAS_V2_MATRIX_{kind};",
            f"  request.c.fill = {pick(m,'uplo')};",
        ]
    elif prim in ("triangular_matrix", "triangular_inverse"):
        enum = (
            "TRIANGULAR_INVERSE"
            if prim == "triangular_inverse"
            else ("TRIANGULAR_SOLVE" if op == "trsm" else "TRIANGULAR_MATMUL")
        )
        lines += [
            f"  request.operation = ROCM_BLAS_V2_{enum};",
            f"  request.side = {pick(m,'side',default='rocblas_side_left')};",
            f"  request.compute_type = {pick(m,'compute_type',default='rocblas_datatype_invalid')};",
            f"  request.transpose = {pick(m,'transa',default='rocblas_operation_none')};",
        ]
        scalar(lines, "alpha", m, "alpha")
        matrix(lines, "a", m, "a", n, n)
        matrix(lines, "b", m, "b", mm, n)
        matrix(lines, "d", m, "c", mm, n)
        if prim == "triangular_inverse":
            matrix(lines, "d", m, "inva", n, n)
        lines += [
            f"  request.a.fill = {pick(m,'uplo')};",
            f"  request.a.diagonal = {pick(m,'diag')};",
            f"  request.inverse_a = rocm::interfaces::narrow_v2_memory({pick(m,'inva',default='nullptr')}, {stride(m,'inva')});",
            f"  request.inverse_a_size = {pick(m,'inva_size')};",
        ]
    else:
        enum = (
            "DIAGONAL_MATRIX_MULTIPLY"
            if prim == "diagonal_matrix_multiply"
            else "MATRIX_ADD"
        )
        lines += [
            f"  request.operation = ROCM_BLAS_V2_{enum};",
            f"  request.side = {pick(m,'side',default='rocblas_side_left')};",
            f"  request.compute_type = {pick(m,'compute_type',default='rocblas_datatype_invalid')};",
            f"  request.has_public_extended_operation = {1 if 'geam_ex_op' in m else 0};",
            f"  request.public_extended_operation = {pick(m,'geam_ex_op',default='rocblas_geam_ex_operation_min_plus')};",
            f"  request.auxiliary_dimension = {k};",
            f"  request.operation_a = {pick(m,'transa',default='rocblas_operation_none')};",
            f"  request.operation_b = {pick(m,'transb',default='rocblas_operation_none')};",
        ]
        scalar(lines, "alpha", m, "alpha")
        scalar(lines, "beta", m, "beta")
        matrix(lines, "a", m, "a", mm, n)
        matrix(lines, "b", m, "b", mm, n)
        matrix(lines, "c", m, "c", mm, n)
        matrix(lines, "d", m, "d", mm, n)
        vector(lines, "diagonal", m, "x", n)
    lines.append("  return rocm::interfaces::narrow_v2_dispatch(handle, &request);")
    return lines


def edge_body(f: dict) -> list[str]:
    m = param_map(f)
    n = f["name"]
    if f["return_type"] == "rocblas_status":
        if pick(m, "handle", default=""):
            return [
                "  if (!handle) return rocblas_status_invalid_handle;",
                "  return rocblas_status_not_implemented;",
            ]
        return ["  return rocblas_status_not_implemented;"]
    if f["return_type"].replace("_Bool", "bool") == "bool":
        return ["  return false;"]
    if f["return_type"] == "void":
        return ["  std::abort();" if n == "rocblas_abort" else "  return;"]
    if f["return_type"] == "const char *":
        return ['  return "rocBLAS narrow v2 spike";']
    if f["return_type"] == "rocblas_pointer_mode":
        return ["  return rocblas_pointer_mode_host;"]
    raise ValueError(f"edge return unsupported: {n} {f['return_type']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--categorization", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    funcs = {
        d["name"]: d
        for d in json.loads(a.snapshot.read_text())["declarations"]
        if d["kind"] == "function"
    }
    rows = {r["name"]: r for r in json.loads(a.categorization.read_text())["callables"]}
    if funcs.keys() != rows.keys():
        raise ValueError("snapshot and categorization callable sets differ")
    out = [
        "// Generated by generate_rocblas_narrow_v2.py. Do not edit.",
        '#include "rocblas_narrow_v2_runtime.h"',
        "#include <cstdlib>",
        "",
        'extern "C" {',
    ]
    translated = 0
    for name in sorted(funcs):
        if name in CUSTOM:
            continue
        f = funcs[name]
        row = rows[name]
        out += ["", f"{f['return_type'].replace('_Bool','bool')} {name}({sig(f)}) {{"]
        out += [f"  (void){p['name']};" for p in f["parameters"]]
        if row["narrow_primitive"] in REQUEST:
            if not f["parameters"] or f["parameters"][0]["type"] != "rocblas_handle":
                raise ValueError(f"compute call lacks handle: {name}")
            out += ["  if (!handle) return rocblas_status_invalid_handle;"] + populate(
                f, row
            )
            translated += 1
        elif row["narrow_primitive"] in ("none", "compatibility_bridge"):
            out += edge_body(f)
        else:
            raise ValueError(f"unmapped primitive {row['narrow_primitive']} in {name}")
        out.append("}")
    if translated != 1156:
        raise ValueError(f"expected 1156 semantic adapters, got {translated}")
    out += ["", '}  // extern "C"', ""]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(out))


if __name__ == "__main__":
    main()
