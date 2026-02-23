# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt-bench format benchmarking utilities for rocRoller performance testing."""

from itertools import chain
from pathlib import Path

import pandas as pd
import yaml

TYPE_SIZE_BYTES = {
    "fp4": 0.5,
    "fp6": 0.75,
    "fp8": 1.0,
    "half": 2.0,
    "float": 4.0,
    "bf6": 0.75,
    "bf8": 1.0,
}

# Column name mappings from rocRoller to hipblaslt-bench format
RENAMES = {
    "type_A": "a_type",
    "type_B": "b_type",
    "type_C": "c_type",
    "type_D": "d_type",
    "trans_A": "transA",
    "trans_B": "transB",
    "M": "m",
    "N": "n",
    "K": "k",
    "scale_A": "scaleA",
    "scale_B": "scaleB",
}

# Headers to include in the hipblaslt-bench format CSV output
INCLUDED_HEADERS = [
    "transA",
    "transB",
    "grouped_gemm",
    "batch_count",
    "m",
    "n",
    "k",
    "alpha",
    "lda",
    "stride_a",
    "beta",
    "ldb",
    "stride_b",
    "ldc",
    "stride_c",
    "ldd",
    "stride_d",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
    "scaleC",
    "scaleD",
    "amaxD",
    "activation_type",
    "bias_vector",
    "bias_type",
    "rotating_buffer",
    "rocroller-Gflops",
    "gbs",
    "us",
    "macro_tile",
    "grid_size",
    "extra_fields",
]


def compute_gflops(m, n, k, runtime_ns):
    """Calculate GFLOPS (giga floating-point operations per second)."""
    flo = 2 * m * n * k
    gflops = flo / runtime_ns
    return gflops


def compute_gbs(m, n, k, runtime_ns, element_size):
    """Calculate GB/s (gigabytes per second) bandwidth."""
    element_size = float(TYPE_SIZE_BYTES.get(element_size.lower(), 4.0))
    data_movement_bytes = (m * k + k * n + m * n) * element_size
    gbs = data_movement_bytes / runtime_ns
    return gbs


def merge_types(data):
    """Merge type information and process kernel execution data."""
    rec = dict(data)
    if any(key in rec for key in ("problem", "solution", "benchmark")):
        flattened = {
            key: value
            for key, value in rec.items()
            if key not in ("problem", "solution", "benchmark")
        }
        for key in ("problem", "solution", "benchmark"):
            section = rec.get(key)
            if isinstance(section, dict):
                flattened.update(section)
        rec = flattened

    if "types" in rec and isinstance(rec["types"], dict):
        rec.update(rec["types"])
        del rec["types"]

    kernel_exec = rec.get("kernelExecute")
    if isinstance(kernel_exec, list) and kernel_exec:
        rec["us"] = (sum(kernel_exec) / len(kernel_exec)) / 1e3

    mac_m = rec.get("mac_m")
    mac_n = rec.get("mac_n")
    mac_k = rec.get("mac_k")
    if mac_m is not None and mac_n is not None and mac_k is not None:
        rec["macro_tile"] = f"{mac_m}x{mac_n}x{mac_k}"
    return rec


def dump_csv(suite: str, rundir: Path, outdir: Path = None):
    """
    Consolidate performance data and dump a hipblaslt-bench compatible CSV.
    Only the specified columns (INCLUDED_HEADERS) are included,
    with empty fields for missing data.
    """
    if outdir is None:
        outdir = rundir
    results = []
    for jpath in chain(rundir.glob("gemm-*.yaml"), rundir.glob("codegen-*.yaml")):
        yamldata = yaml.safe_load(jpath.read_text())
        if isinstance(yamldata, dict):
            results.append(merge_types(yamldata))
        elif isinstance(yamldata, list):
            for entry in yamldata:
                results.append(merge_types(entry))

    for rec in results:
        rec["batch_count"] = rec.get("batch_count", 1)
        rec["compute_type"] = rec.get("compute_type", "f32")
        rec["rocroller-Gflops"] = round(
            compute_gflops(
                rec.get("M", 0), rec.get("N", 0), rec.get("K", 0), rec.get("us", 0)
            ),
            4,
        )
        rec["gbs"] = round(
            compute_gbs(
                rec.get("M", 0),
                rec.get("N", 0),
                rec.get("K", 0),
                rec.get("us", 0),
                rec.get("type_A", 4.0),
            ),
            4,
        )

    df = pd.DataFrame(results)
    df = df.rename(columns=RENAMES)
    for col in INCLUDED_HEADERS:
        if col not in df.columns:
            df[col] = ""
    df = df[INCLUDED_HEADERS]
    outpath = outdir / f"{suite}.csv"
    print(f"Outpath: {outpath}")
    df.to_csv(outpath, index=False)
    print(f"Wrote CSV: {outpath.resolve()}")
