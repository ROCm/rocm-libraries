#!/usr/bin/env python3
"""Generate the exhaustive rocBLAS public-callable narrowing ledger."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


OPERATION_CLUSTERS = {
    "scal": ("vector.transform", "vector_transform"),
    "copy": ("vector.transform", "vector_transform"),
    "swap": ("vector.transform", "vector_transform"),
    "axpy": ("vector.transform", "vector_transform"),
    "rot": ("vector.rotation", "vector_rotate"),
    "rotg": ("vector.rotation", "rotation_parameters"),
    "rotm": ("vector.rotation", "vector_rotate_modified"),
    "rotmg": ("vector.rotation", "modified_rotation_parameters"),
    "dot": ("vector.reduction", "vector_reduce"),
    "dotc": ("vector.reduction", "vector_reduce"),
    "dotu": ("vector.reduction", "vector_reduce"),
    "nrm2": ("vector.reduction", "vector_reduce"),
    "asum": ("vector.reduction", "vector_reduce"),
    "iamax": ("vector.reduction", "vector_index_reduce"),
    "iamin": ("vector.reduction", "vector_index_reduce"),
    "gemv": ("matrix.vector", "matrix_vector"),
    "gbmv": ("matrix.vector", "matrix_vector"),
    "symv": ("matrix.vector", "structured_matrix_vector"),
    "hemv": ("matrix.vector", "structured_matrix_vector"),
    "sbmv": ("matrix.vector", "structured_matrix_vector"),
    "hbmv": ("matrix.vector", "structured_matrix_vector"),
    "spmv": ("matrix.vector", "structured_matrix_vector"),
    "hpmv": ("matrix.vector", "structured_matrix_vector"),
    "trmv": ("matrix.triangular_vector", "triangular_vector"),
    "tbmv": ("matrix.triangular_vector", "triangular_vector"),
    "tpmv": ("matrix.triangular_vector", "triangular_vector"),
    "trsv": ("matrix.triangular_vector", "triangular_vector"),
    "tbsv": ("matrix.triangular_vector", "triangular_vector"),
    "tpsv": ("matrix.triangular_vector", "triangular_vector"),
    "ger": ("matrix.rank_update", "rank_update"),
    "geru": ("matrix.rank_update", "rank_update"),
    "gerc": ("matrix.rank_update", "rank_update"),
    "syr": ("matrix.rank_update", "structured_rank_update"),
    "her": ("matrix.rank_update", "structured_rank_update"),
    "spr": ("matrix.rank_update", "structured_rank_update"),
    "hpr": ("matrix.rank_update", "structured_rank_update"),
    "syr2": ("matrix.rank_update", "structured_rank_update"),
    "her2": ("matrix.rank_update", "structured_rank_update"),
    "spr2": ("matrix.rank_update", "structured_rank_update"),
    "hpr2": ("matrix.rank_update", "structured_rank_update"),
    "gemm": ("matrix.matmul", "matmul"),
    "gemmt": ("matrix.matmul", "matmul"),
    "symm": ("matrix.structured", "structured_matmul"),
    "hemm": ("matrix.structured", "structured_matmul"),
    "syrk": ("matrix.structured", "structured_rank_k"),
    "herk": ("matrix.structured", "structured_rank_k"),
    "syr2k": ("matrix.structured", "structured_rank_k"),
    "her2k": ("matrix.structured", "structured_rank_k"),
    "syrkx": ("matrix.structured", "structured_rank_k"),
    "herkx": ("matrix.structured", "structured_rank_k"),
    "trmm": ("matrix.triangular", "triangular_matrix"),
    "trsm": ("matrix.triangular", "triangular_matrix"),
    "trtri": ("matrix.triangular", "triangular_inverse"),
    "geam": ("matrix.transform", "matrix_transform"),
    "dgmm": ("matrix.transform", "diagonal_matrix_multiply"),
}

EDGE_RULES = {
    "rocblas_abort": ("edge.diagnostic", "edge_local"),
    "rocblas_create_handle": ("edge.lifecycle", "edge_local"),
    "rocblas_destroy_handle": ("edge.lifecycle", "edge_local"),
    "rocblas_initialize": ("edge.lifecycle", "edge_local"),
    "rocblas_status_to_string": ("edge.version", "edge_local"),
    "rocblas_pointer_to_mode": ("edge.policy", "edge_local"),
}

EDGE_PREFIXES = (
    ("rocblas_get_version_", "edge.version", "edge_local"),
    ("rocblas_get_commit_hash_", "edge.version", "edge_local"),
    ("rocblas_get_matrix", "edge.transfer", "edge_local"),
    ("rocblas_set_matrix", "edge.transfer", "edge_local"),
    ("rocblas_get_vector", "edge.transfer", "edge_local"),
    ("rocblas_set_vector", "edge.transfer", "edge_local"),
    ("rocblas_device_malloc_", "edge.memory", "bridge_only"),
    ("rocblas_get_device_memory_", "edge.memory", "edge_local"),
    ("rocblas_set_device_memory_", "edge.memory", "edge_local"),
    ("rocblas_start_device_memory_", "edge.memory", "edge_local"),
    ("rocblas_stop_device_memory_", "edge.memory", "edge_local"),
    ("rocblas_is_device_memory_", "edge.memory", "edge_local"),
    ("rocblas_is_managing_device_", "edge.memory", "edge_local"),
    ("rocblas_is_user_managing_", "edge.memory", "edge_local"),
    ("rocblas_set_optimal_device_", "edge.memory", "bridge_only"),
    ("rocblas_get_", "edge.policy", "edge_local"),
    ("rocblas_set_", "edge.policy", "edge_local"),
)

# The v2 matmul request describes one homogeneous problem shape. Grouped GEMM
# carries an array of shapes, operations, leading dimensions, and scalars, so it
# cannot be represented by setting the ordinary pointer-array batch flag. Keep
# these callables in the typed compatibility bridge until the narrow protocol
# has an audited per-group descriptor. The explicit set makes a future grouped
# spelling fail classification instead of inheriting this decision silently.
GROUPED_BATCHED_BRIDGE = {
    "rocblas_dgemm_grouped_batched",
    "rocblas_dgemm_grouped_batched_64",
    "rocblas_gemm_grouped_batched_ex",
    "rocblas_gemm_grouped_batched_ex_64",
    "rocblas_sgemm_grouped_batched",
    "rocblas_sgemm_grouped_batched_64",
}


@dataclass(frozen=True)
class Row:
    name: str
    cluster: str
    disposition: str
    narrow_primitive: str
    operation: str | None
    index_width: int | str
    batch_kind: str
    explicit_datatypes: bool
    source_file: str
    source_line: int


def operation_for(name: str) -> str | None:
    spelling = name.removeprefix("rocblas_")
    if re.match(r"^i[sdcz]amax(?:_|$)", spelling):
        return "iamax"
    if re.match(r"^i[sdcz]amin(?:_|$)", spelling):
        return "iamin"
    for operation in sorted(OPERATION_CLUSTERS, key=len, reverse=True):
        # Precision prefixes include ordinary s/d/c/z/h/bf, mixed two/three-letter
        # forms, and the leading i on index reductions.
        pattern = rf"^(?:[a-z]{{0,4}}){re.escape(operation)}(?:_|$)"
        if re.match(pattern, spelling):
            return operation
    return None


def categorize(declaration: dict[str, object]) -> Row:
    name = declaration.get("name")
    source_file = declaration.get("file")
    source_line = declaration.get("line")
    parameters = declaration.get("parameters")
    if (
        not isinstance(name, str)
        or not isinstance(source_file, str)
        or not isinstance(source_line, int)
        or not isinstance(parameters, list)
    ):
        raise ValueError("malformed rocBLAS declaration")
    operation = operation_for(name)
    if name in GROUPED_BATCHED_BRIDGE:
        cluster = "matrix.matmul"
        disposition = "bridge_only"
        primitive = "compatibility_bridge"
    elif "grouped_batched" in name:
        raise ValueError(f"unclassified grouped-batched rocBLAS callable: {name}")
    elif operation is not None:
        cluster, primitive = OPERATION_CLUSTERS[operation]
        disposition = "normalized_provider"
    elif name in EDGE_RULES:
        cluster, disposition = EDGE_RULES[name]
        primitive = "none"
    else:
        match = next((rule for rule in EDGE_PREFIXES if name.startswith(rule[0])), None)
        if match is None:
            raise ValueError(f"unclassified rocBLAS callable: {name}")
        _, cluster, disposition = match
        primitive = "compatibility_bridge" if disposition == "bridge_only" else "none"
    parameter_types = [
        parameter.get("type") for parameter in parameters if isinstance(parameter, dict)
    ]
    if any(parameter_type == "int64_t" for parameter_type in parameter_types):
        index_width: int | str = 64
    elif any(parameter_type == "rocblas_int" for parameter_type in parameter_types):
        index_width = 32
    else:
        index_width = "not_applicable"
    if name in GROUPED_BATCHED_BRIDGE:
        batch_kind = "grouped"
    elif "strided_batched" in name:
        batch_kind = "strided"
    elif "batched" in name:
        batch_kind = "pointer_array"
    else:
        batch_kind = "single"
    explicit_datatypes = "_ex" in name or any(
        parameter_type == "rocblas_datatype" for parameter_type in parameter_types
    )
    return Row(
        name,
        cluster,
        disposition,
        primitive,
        operation,
        index_width,
        batch_kind,
        explicit_datatypes,
        source_file,
        source_line,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.snapshot.open(encoding="utf-8") as stream:
        document = json.load(stream)
    declarations = document.get("declarations")
    if not isinstance(declarations, list):
        raise ValueError("snapshot declarations must be an array")
    rows = [
        categorize(declaration)
        for declaration in declarations
        if isinstance(declaration, dict) and declaration.get("kind") == "function"
    ]
    rows.sort(key=lambda row: row.name)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.cluster] = counts.get(row.cluster, 0) + 1
    output = {
        "schema_version": 1,
        "source_snapshot": "api/snapshots/rocblas.json",
        "callable_count": len(rows),
        "cluster_counts": dict(sorted(counts.items())),
        "callables": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
