#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Pointwise Runtime Scalar Golden Reference Bundle Generator

Generates pre-computed reference data for pointwise operations where one
operand is a runtime pass-by-value scalar tensor (dims=[1], strides=[1]),
rather than a compile-time baked scalar.

Supported operations:
  add  — out = in0 + scalar
  mul  — out = in0 * scalar

Output: {base_filename}.json + {base_filename}.tensor{uid}.bin + {base_filename}.meta.json

Usage:
    python generate_pointwise_runtime_scalar_golden.py \\
        --base-filename bundles/PointwiseFwdRuntimeScalar/Add/Small/Small \\
        --operation add --dims 2 4 6 8 --scalar-value 100.0 --dtype float --seed 42

    python generate_pointwise_runtime_scalar_golden.py \\
        --base-filename bundles/PointwiseFwdRuntimeScalar/Mul/Small/Small \\
        --operation mul --dims 2 4 6 8 --scalar-value 10.0 --dtype float --seed 42
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import torch

# Bump when generator logic changes in a way that affects output data.
# 1.0.0 — Initial: pointwise ADD and MUL with runtime scalar operand (in_1)
GENERATOR_VERSION = "1.0.0"

DTYPE_MAP = {
    "float": {"torch": torch.float32, "json": "float", "bytes": 4},
    "half": {"torch": torch.float16, "json": "half", "bytes": 2},
    "bfloat16": {"torch": torch.bfloat16, "json": "bfloat16", "bytes": 2},
}

OPERATION_MAP = {
    "add": "add",
    "mul": "mul",
}


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def execute_pointwise(in0, scalar_val, operation, compute_dtype):
    in0_f = in0.to(compute_dtype)
    scalar = torch.full([1], scalar_val, dtype=compute_dtype)
    if operation == "add":
        out = in0_f + scalar
    elif operation == "mul":
        out = in0_f * scalar
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    return out.to(in0.dtype), scalar


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        raw = t.view(torch.uint8).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def build_graph_json(dims, strides, scalar_val, operation, dtype_str):
    # UIDs:
    #   0 = in0  (main input tensor)
    #   1 = in1  (runtime scalar, dims=[1], strides=[1])
    #   2 = out0 (output tensor)
    tensors = [
        {
            "uid": 0,
            "name": "in0",
            "dims": dims,
            "strides": strides,
            "data_type": dtype_str,
            "virtual": False,
        },
        {
            "uid": 1,
            "name": "scalar",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            # No "value_type" / "value" fields — this is the runtime form
        },
        {
            "uid": 2,
            "name": "out0",
            "dims": dims,
            "strides": strides,
            "data_type": dtype_str,
            "virtual": False,
        },
    ]

    graph = {
        "nodes": [
            {
                "type": "PointwiseAttributes",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "operation": OPERATION_MAP[operation],
                    "in_0_tensor_uid": 0,
                    "in_1_tensor_uid": 1,
                    "in_2_tensor_uid": None,
                    "axis_tensor_uid": None,
                    "relu_lower_clip": None,
                    "relu_upper_clip": None,
                    "relu_lower_clip_slope": None,
                    "swish_beta": None,
                    "elu_alpha": None,
                    "softplus_beta": None,
                },
                "outputs": {
                    "out_0_tensor_uid": 2,
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": dtype_str,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }
    return graph


def _get_generator_sha256():
    script_path = os.path.abspath(__file__)
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_meta_json(config, pytorch_version):
    rocm_ver = ""
    if "+rocm" in pytorch_version:
        rocm_ver = pytorch_version.split("+rocm")[1]

    return {
        "format_version": 1,
        "operation": f"PointwiseFwdRuntimeScalar/{config['operation'].upper()}",
        "generator": "generate_pointwise_runtime_scalar_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_strategy": "pytorch_elementwise",
        "rocm_version": rocm_ver,
        "seed": config["seed"],
        "notes": (
            f"Runtime scalar bundle: in_1 (uid=1) is a runtime pass-by-value tensor "
            f"(no value_type/value fields in JSON). Detection value={config['scalar_value']} is "
            f"intentionally far from any compile-time default to expose providers that "
            f"ignore the runtime scalar."
        ),
        "config": {
            "operation": config["operation"],
            "dims": config["dims"],
            "dtype": config["dtype"],
            "scalar_uid": 1,
            "scalar_value": config["scalar_value"],
        },
    }


def generate_bundle(
    base_filename,
    operation,
    dims,
    dtype="float",
    scalar_value=100.0,
    seed=42,
    min_val=0.1,
    max_val=1.0,
):
    if dtype not in DTYPE_MAP:
        print(
            f"ERROR: --dtype must be one of {list(DTYPE_MAP.keys())} (got '{dtype}')",
            file=sys.stderr,
        )
        sys.exit(1)
    if operation not in OPERATION_MAP:
        print(
            f"ERROR: --operation must be one of {list(OPERATION_MAP.keys())} (got '{operation}')",
            file=sys.stderr,
        )
        sys.exit(1)

    dtype_info = DTYPE_MAP[dtype]
    torch_dtype = dtype_info["torch"]
    strides = compute_contiguous_strides(dims)

    os.makedirs(os.path.dirname(os.path.abspath(base_filename)), exist_ok=True)

    print(f"Generating pointwise runtime scalar bundle: {base_filename}")
    print(f"  operation: {operation}")
    print(f"  dims: {dims}, strides: {strides}, dtype: {dtype}")
    print(f"  scalar uid=1, value={scalar_value} (runtime — no value field in JSON)")
    print(f"  seed: {seed}, input range: [{min_val}, {max_val}]")

    rng = torch.Generator().manual_seed(seed)
    in0 = torch.empty(dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)

    out0, scalar_tensor = execute_pointwise(in0, scalar_value, operation, torch.float32)

    assert not torch.isnan(out0).any(), "NaN in output"
    assert not torch.isinf(out0).any(), "Inf in output"

    # Write tensor .bin files
    for name, tensor, uid in [
        ("in0", in0, 0),
        ("scalar", scalar_tensor, 1),
        ("out0", out0, 2),
    ]:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_bytes = os.path.getsize(bin_path)
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_bytes} bytes"
        )

    # Write graph JSON
    graph_json = build_graph_json(
        dims, strides, scalar_value, operation, dtype_info["json"]
    )
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    # Write meta JSON
    config = {
        "operation": operation,
        "dims": dims,
        "dtype": dtype,
        "scalar_value": scalar_value,
        "seed": seed,
    }
    meta_json = build_meta_json(config, torch.__version__)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pointwise runtime scalar golden reference bundles"
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Path prefix for output files (no extension)",
    )
    parser.add_argument(
        "--operation",
        required=True,
        choices=list(OPERATION_MAP.keys()),
        help="Pointwise operation: add or mul",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        required=True,
        help="Dimensions of in0 tensor (e.g. --dims 2 4 6 8)",
    )
    parser.add_argument(
        "--scalar-value",
        type=float,
        required=True,
        help=(
            "Value for the runtime scalar tensor (uid=1). "
            "Use a value far from any compile-time default to enable detection "
            "(e.g. 100.0 for ADD, 10.0 for MUL)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float",
        choices=list(DTYPE_MAP.keys()),
        help="Data type for in0 and out0 tensors (default: float)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=0.1, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    args = parser.parse_args()

    generate_bundle(
        base_filename=args.base_filename,
        operation=args.operation,
        dims=args.dims,
        dtype=args.dtype,
        scalar_value=args.scalar_value,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
