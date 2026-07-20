#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Batchnorm Runtime Scalar Golden Reference Bundle Generator

Generates pre-computed reference data for batchnorm operations where a
scalar (epsilon or momentum) is provided as a runtime pass-by-value tensor
(dims=[1], strides=[1]) rather than a compile-time baked value.

Supported scalars:
  epsilon  — BN inference with variance_ext; frontend REJECTS runtime epsilon
             (validateScalarParameter enforces pass_by_value==true for epsilon).
             This generates a NEGATIVE test bundle: the graph JSON is valid to
             parse, but validate() will return INVALID_VALUE. No golden tensor
             data is written for this case.
  momentum — BN forward training; frontend accepts a runtime momentum tensor.
             A detection value far from the compile-time default (0.5f) is used.

Output for momentum: {base_filename}.json + .tensor{uid}.bin + .meta.json
Output for epsilon:  {base_filename}.json + .meta.json (no .bin — negative test)

Usage:
    # Momentum (positive test — generates golden output)
    python generate_batchnorm_runtime_scalar_golden.py \\
        --base-filename bundles/BatchnormFwdTrainingRuntimeScalar/Momentum/Small/Small \\
        --scalar momentum --scalar-value 0.9 --size 2 3 4 5 --seed 42

    # Epsilon (negative test — generates graph JSON only, no golden output)
    python generate_batchnorm_runtime_scalar_golden.py \\
        --base-filename bundles/BatchnormFwdInferenceRuntimeScalar/Epsilon/Small/Small \\
        --scalar epsilon --scalar-value 1.0 --size 2 3 4 5 --seed 42
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

# Bump when generator logic changes in a way that affects output data.
# 1.0.0 — Initial: BN momentum (positive) and epsilon (negative) runtime scalar bundles
# 1.1.0 — Epsilon bundle now writes golden .bin files; mean_out/inv_var_out derived
#          from execute_batchnorm_fwd_training (not recomputed independently)
GENERATOR_VERSION = "1.1.0"


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu().float()
    raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def _get_generator_sha256():
    script_path = os.path.abspath(__file__)
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Epsilon bundle (negative test — no golden output)
# ---------------------------------------------------------------------------


def build_epsilon_graph_json(x_dims):
    """BN inference (variance_ext style) with epsilon as a runtime tensor.

    The frontend's validateScalarParameter will reject this graph at validate()
    time with INVALID_VALUE because epsilon must be pass_by_value==true.
    No golden tensor output is generated.

    Tensor UIDs:
      0 = x            (input)
      1 = mean          (input)
      2 = inv_variance  (input)
      3 = scale         (input)
      4 = bias          (input)
      5 = epsilon       (runtime scalar — no value_type/value fields)
      6 = y             (output)
    """
    channel_dims = [1, x_dims[1]] + [1] * (len(x_dims) - 2)
    channel_strides = compute_contiguous_strides(channel_dims)
    x_strides = compute_contiguous_strides(x_dims)

    tensors = [
        {
            "uid": 0,
            "name": "x",
            "dims": x_dims,
            "strides": x_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 1,
            "name": "mean",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 2,
            "name": "variance",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 3,
            "name": "scale",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 4,
            "name": "bias",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 5,
            "name": "epsilon",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            # Intentionally no "value_type"/"value" — runtime form.
            # The frontend will reject this via validateScalarParameter.
        },
        {
            "uid": 6,
            "name": "y",
            "dims": x_dims,
            "strides": x_strides,
            "data_type": "float",
            "virtual": False,
        },
    ]

    return {
        "nodes": [
            {
                "type": "BatchnormInferenceAttributesVarianceExt",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "x_tensor_uid": 0,
                    "mean_tensor_uid": 1,
                    "variance_tensor_uid": 2,
                    "scale_tensor_uid": 3,
                    "bias_tensor_uid": 4,
                    "epsilon_tensor_uid": 5,
                },
                "outputs": {
                    "y_tensor_uid": 6,
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": "float",
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }


def execute_batchnorm_inference_variance_ext(x, mean, variance, scale, bias, epsilon):
    """BN inference (variance_ext form) reference computation.

    Computes y = scale * (x - mean) / sqrt(variance + epsilon) + bias.
    This matches what the provider kernel does when it receives variance + epsilon
    and computes inv_variance internally.
    """
    x_f = x.float()
    inv_var = 1.0 / torch.sqrt(variance.float() + float(epsilon))
    y = scale.float() * (x_f - mean.float()) * inv_var + bias.float()
    return y.to(x.dtype)


def generate_epsilon_bundle(base_filename, x_dims, scalar_value, seed):
    """Generate the epsilon negative-test bundle.

    Writes graph JSON + golden .bin files + meta. The graph uses a runtime
    epsilon (no value_type/value in JSON), so validate() returns INVALID_VALUE
    and a correct provider never executes. Golden outputs are computed with
    the correct epsilon (1e-5), so if a broken provider bypasses validation
    and executes with epsilon=scalar_value instead, its output diverges.
    """
    os.makedirs(os.path.dirname(os.path.abspath(base_filename)), exist_ok=True)

    print(
        f"Generating BN epsilon runtime scalar bundle (NEGATIVE TEST): {base_filename}"
    )
    print(f"  x_dims: {x_dims}, epsilon uid=5, detection value={scalar_value}")
    print(
        f"  Golden computed with correct epsilon=1e-5; broken provider using "
        f"epsilon={scalar_value} produces detectably wrong output."
    )

    C = x_dims[1]
    channel_dims = [1, C] + [1] * (len(x_dims) - 2)

    rng = torch.Generator().manual_seed(seed)
    x = torch.empty(x_dims, dtype=torch.float32).uniform_(-1.0, 1.0, generator=rng)
    mean = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        -0.5, 0.5, generator=rng
    )
    # variance must be non-negative
    variance = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        0.1, 2.0, generator=rng
    )
    scale = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        0.5, 2.0, generator=rng
    )
    bias = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        -0.5, 0.5, generator=rng
    )
    # epsilon runtime tensor: detection value, but golden computed with correct 1e-5
    epsilon_tensor = torch.full([1], scalar_value, dtype=torch.float32)
    correct_epsilon = 1e-5

    y = execute_batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, correct_epsilon
    )

    graph_json = build_epsilon_graph_json(x_dims)
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    # Write .bin files: uid order matches tensor UIDs in graph JSON
    # uid 0=x, 1=mean, 2=variance, 3=scale, 4=bias, 5=epsilon, 6=y
    tensor_data = [
        ("x", x, 0),
        ("mean", mean, 1),
        ("variance", variance, 2),
        ("scale", scale, 3),
        ("bias", bias, 4),
        ("epsilon", epsilon_tensor, 5),
        ("y", y, 6),
    ]
    for name, tensor, uid in tensor_data:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        print(f"  tensor{uid} ({name}): {bin_path} ({os.path.getsize(bin_path)} bytes)")

    meta = {
        "format_version": 1,
        "operation": "BatchnormFwdInferenceRuntimeScalar/Epsilon",
        "generator": "generate_batchnorm_runtime_scalar_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {torch.__version__}",
        "reference_strategy": "pytorch_bn_inference_variance_ext",
        "rocm_version": "",
        "seed": seed,
        "notes": (
            "NEGATIVE TEST: epsilon (uid=5) is a runtime tensor (no value_type/value fields). "
            "hipDNN's validateScalarParameter requires epsilon to be pass_by_value==true, so "
            f"validate() must return INVALID_VALUE. Golden output is computed with correct "
            f"epsilon=1e-5. Detection value={scalar_value} is intentionally far from 1e-5: "
            "a broken provider that bypasses validation and executes with the wrong epsilon "
            "will produce output that diverges from the golden data."
        ),
        "config": {
            "scalar": "epsilon",
            "x_dims": x_dims,
            "scalar_uid": 5,
            "scalar_value": scalar_value,
            "correct_epsilon": 1e-5,
            "expected_failure": "validate",
        },
    }
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")


# ---------------------------------------------------------------------------
# Momentum bundle (positive test — golden output generated)
# ---------------------------------------------------------------------------


def execute_batchnorm_fwd_training(
    x, scale, bias, prev_running_mean, prev_running_var, momentum
):
    """BN forward training reference via PyTorch batch_norm (training=True).

    Returns (y, mean_out, inv_var_out, next_running_mean, next_running_var).
    mean_out and inv_var_out are derived from the same population statistics
    that PyTorch's batch_norm kernel uses internally, ensuring they match
    provider output.
    """
    C = x.shape[1]
    reduce_dims = [d for d in range(x.ndim) if d != 1]
    shape = prev_running_mean.shape

    # Compute batch statistics (population, matching PyTorch's internal kernel)
    x_f = x.float()
    mean_out = x_f.mean(dim=reduce_dims, keepdim=True)
    var_batch = x_f.var(dim=reduce_dims, unbiased=False, keepdim=True)
    inv_var_out = 1.0 / torch.sqrt(var_batch + 1e-5)

    # PyTorch batch_norm expects 1-D running stats
    rm = prev_running_mean.view(C).clone().float()
    rv = prev_running_var.view(C).clone().float()
    s = scale.view(C).float()
    b = bias.view(C).float()

    y = F.batch_norm(
        x_f,
        rm,
        rv,
        weight=s,
        bias=b,
        training=True,
        momentum=float(momentum),
        eps=1e-5,
    )

    # After batch_norm(training=True), rm and rv are updated in-place by PyTorch
    next_running_mean = rm.view(shape)
    next_running_var = rv.view(shape)

    return (
        y.to(x.dtype),
        mean_out.view(shape),
        inv_var_out.view(shape),
        next_running_mean,
        next_running_var,
    )


def build_momentum_graph_json(x_dims, momentum_value):
    """BN forward training with momentum as a runtime tensor.

    Tensor UIDs:
      0  = x                    (input)
      1  = scale                (input)
      2  = bias                 (input)
      3  = epsilon              (compile-time scalar, value=1e-5)
      4  = prev_running_mean    (input)
      5  = prev_running_var     (input)
      6  = momentum             (runtime scalar — no value_type/value fields)
      7  = y                    (output)
      8  = mean_out             (output)
      9  = inv_var_out          (output)
      10 = next_running_mean    (output)
      11 = next_running_var     (output)
    """
    x_strides = compute_contiguous_strides(x_dims)
    channel_dims = [1, x_dims[1]] + [1] * (len(x_dims) - 2)
    channel_strides = compute_contiguous_strides(channel_dims)

    tensors = [
        {
            "uid": 0,
            "name": "x",
            "dims": x_dims,
            "strides": x_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 1,
            "name": "scale",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 2,
            "name": "bias",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        # epsilon stays compile-time — only momentum is runtime in this bundle
        {
            "uid": 3,
            "name": "epsilon",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            "value_type": "Float32Value",
            "value": 1e-5,
        },
        {
            "uid": 4,
            "name": "prev_running_mean",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 5,
            "name": "prev_running_var",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 6,
            "name": "momentum",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            # Intentionally no "value_type"/"value" — runtime form.
        },
        {
            "uid": 7,
            "name": "y",
            "dims": x_dims,
            "strides": x_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 8,
            "name": "mean_out",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 9,
            "name": "inv_var_out",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 10,
            "name": "next_running_mean",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
        {
            "uid": 11,
            "name": "next_running_var",
            "dims": channel_dims,
            "strides": channel_strides,
            "data_type": "float",
            "virtual": False,
        },
    ]

    return {
        "nodes": [
            {
                "type": "BatchnormAttributes",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "x_tensor_uid": 0,
                    "scale_tensor_uid": 1,
                    "bias_tensor_uid": 2,
                    "epsilon_tensor_uid": 3,
                    "peer_stats_tensor_uid": [],
                    "prev_running_mean_tensor_uid": 4,
                    "prev_running_variance_tensor_uid": 5,
                    "momentum_tensor_uid": 6,
                },
                "outputs": {
                    "y_tensor_uid": 7,
                    "mean_tensor_uid": 8,
                    "inv_variance_tensor_uid": 9,
                    "next_running_mean_tensor_uid": 10,
                    "next_running_variance_tensor_uid": 11,
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": "float",
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }


def generate_momentum_bundle(
    base_filename, x_dims, scalar_value, seed, min_val, max_val
):
    """Generate the momentum positive-test bundle (graph JSON + .bin + meta)."""
    os.makedirs(os.path.dirname(os.path.abspath(base_filename)), exist_ok=True)

    print(f"Generating BN momentum runtime scalar bundle: {base_filename}")
    print(f"  x_dims: {x_dims}, momentum uid=6, detection value={scalar_value}")
    print(f"  seed: {seed}, input range: [{min_val}, {max_val}]")

    rng = torch.Generator().manual_seed(seed)
    C = x_dims[1]
    channel_dims = [1, C] + [1] * (len(x_dims) - 2)

    x = torch.empty(x_dims, dtype=torch.float32).uniform_(
        min_val, max_val, generator=rng
    )
    scale = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        0.5, 2.0, generator=rng
    )
    bias = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        -0.5, 0.5, generator=rng
    )
    prev_running_mean = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=rng
    )
    prev_running_var = torch.empty(channel_dims, dtype=torch.float32).uniform_(
        0.1, 2.0, generator=rng
    )
    momentum_tensor = torch.full([1], scalar_value, dtype=torch.float32)

    y, mean_out, inv_var_out, next_running_mean, next_running_var = (
        execute_batchnorm_fwd_training(
            x, scale, bias, prev_running_mean, prev_running_var, scalar_value
        )
    )

    assert not torch.isnan(y).any(), "NaN in y"
    assert not torch.isinf(y).any(), "Inf in y"

    # Write tensor .bin files
    tensor_data = [
        ("x", x, 0),
        ("scale", scale, 1),
        ("bias", bias, 2),
        # uid=3 is epsilon (compile-time, baked in JSON — still write bin for input)
        ("epsilon", torch.tensor([1e-5]), 3),
        ("prev_running_mean", prev_running_mean, 4),
        ("prev_running_var", prev_running_var, 5),
        ("momentum", momentum_tensor, 6),
        ("y", y, 7),
        ("mean_out", mean_out, 8),
        ("inv_var_out", inv_var_out, 9),
        ("next_running_mean", next_running_mean, 10),
        ("next_running_var", next_running_var, 11),
    ]
    for name, tensor, uid in tensor_data:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_bytes = os.path.getsize(bin_path)
        print(f"  {name} (uid={uid}): {list(tensor.shape)} -> {size_bytes} bytes")

    graph_json = build_momentum_graph_json(x_dims, scalar_value)
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    rocm_ver = ""
    if "+rocm" in torch.__version__:
        rocm_ver = torch.__version__.split("+rocm")[1]

    meta = {
        "format_version": 1,
        "operation": "BatchnormFwdTrainingRuntimeScalar/Momentum",
        "generator": "generate_batchnorm_runtime_scalar_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {torch.__version__}",
        "reference_strategy": "pytorch_batch_norm_training",
        "rocm_version": rocm_ver,
        "seed": seed,
        "notes": (
            "Runtime scalar bundle: momentum (uid=6) is a runtime tensor (no value_type/value "
            f"field in JSON). Detection value={scalar_value} is intentionally far from the "
            "compile-time default (0.5) to expose providers that ignore the runtime momentum."
        ),
        "config": {
            "scalar": "momentum",
            "x_dims": x_dims,
            "scalar_uid": 6,
            "scalar_value": scalar_value,
        },
    }
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate batchnorm runtime scalar golden reference bundles. "
            "For epsilon: generates a negative-test bundle (graph JSON only, no golden). "
            "For momentum: generates a positive-test bundle with full golden output."
        )
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Path prefix for output files (no extension)",
    )
    parser.add_argument(
        "--scalar",
        required=True,
        choices=["epsilon", "momentum"],
        help="Which scalar to make runtime: 'epsilon' (negative test) or 'momentum' (positive test)",
    )
    parser.add_argument(
        "--scalar-value",
        type=float,
        required=True,
        help=(
            "Detection value for the runtime scalar tensor. "
            "Use a value far from the compile-time default: "
            "epsilon default=1e-5 (use ~1.0), momentum default=0.5 (use ~0.9)."
        ),
    )
    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        required=True,
        help="Dimensions of the x tensor (e.g. --size 2 3 4 5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=0.1, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    args = parser.parse_args()

    if args.scalar == "epsilon":
        generate_epsilon_bundle(
            base_filename=args.base_filename,
            x_dims=args.size,
            scalar_value=args.scalar_value,
            seed=args.seed,
        )
    else:
        generate_momentum_bundle(
            base_filename=args.base_filename,
            x_dims=args.size,
            scalar_value=args.scalar_value,
            seed=args.seed,
            min_val=args.min_val,
            max_val=args.max_val,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
