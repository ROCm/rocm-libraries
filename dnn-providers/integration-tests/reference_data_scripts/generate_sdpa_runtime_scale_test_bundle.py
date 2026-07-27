#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Forward Runtime Scale Golden Reference Bundle Generator

Generates pre-computed reference data for SDPA forward where the attention
scale is provided as a runtime pass-by-value tensor (dims=[1], strides=[1],
uid=4) rather than a compile-time baked scalar (attn_scale_value attribute).

In existing SDPA bundles the scale is baked into the node's `attn_scale_value`
attribute. This script instead wires it as a runtime tensor via `scale_tensor_uid`,
leaving `attn_scale_value` as null. A provider that ignores the runtime scale
and falls back to a compile-time default will produce detectably wrong output.

Uses PyTorch's SDPBackend.MATH as the golden reference source.
Supports BF16 and FP16 data types.

Output: {base_filename}.json + {base_filename}.tensor{uid}.bin + {base_filename}.meta.json

Usage:
    python generate_sdpa_runtime_scale_golden.py \\
        --base-filename bundles/SdpaFwdRuntimeScale/bhsd/bf16/Small/Small \\
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 \\
        --scale-value 1.0 --dtype bf16 --seed 42
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Bump when generator logic changes in a way that affects output data.
# 1.0.0 — Initial: SDPA forward with runtime scale tensor (uid=4)
GENERATOR_VERSION = "1.0.0"

DTYPE_MAP = {
    "bf16": {"torch": torch.bfloat16, "json": "bfloat16", "bytes": 2},
    "fp16": {"torch": torch.float16, "json": "half", "bytes": 2},
}

# UID assignments
UID_Q = 0
UID_K = 1
UID_V = 2
UID_O = 3
UID_SCALE = 4  # runtime scalar tensor — dims=[1], strides=[1]


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def compute_forward(Q, K, V, scale, H_q, H_kv):
    """SDPA forward via PyTorch Math backend."""
    with sdpa_kernel(SDPBackend.MATH):
        O = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            scale=scale,
            enable_gqa=(H_q != H_kv),
        )
    return O


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu()
    if t.dtype in (torch.bfloat16, torch.float16):
        raw = t.view(torch.uint8).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def build_graph_json(q_dims, k_dims, v_dims, o_dims, dtype_str):
    """Build the SDPA graph JSON with scale wired as a runtime tensor.

    Key difference from the compile-time variant (generate_sdpa_fwd_golden.py):
      - scale_tensor_uid = UID_SCALE (pure runtime PBV tensor, no baked fallback)
      - attn_scale_value = null (forces CPU reference to read from variantPack)
    A provider that falls back to a compile-time default (e.g. 1/sqrt(headDim))
    instead of reading the runtime tensor produces output that diverges from the
    CPU reference.
    """
    tensors = [
        {
            "uid": UID_Q,
            "name": "Q",
            "dims": q_dims,
            "strides": compute_contiguous_strides(q_dims),
            "data_type": dtype_str,
            "virtual": False,
        },
        {
            "uid": UID_K,
            "name": "K",
            "dims": k_dims,
            "strides": compute_contiguous_strides(k_dims),
            "data_type": dtype_str,
            "virtual": False,
        },
        {
            "uid": UID_V,
            "name": "V",
            "dims": v_dims,
            "strides": compute_contiguous_strides(v_dims),
            "data_type": dtype_str,
            "virtual": False,
        },
        {
            "uid": UID_O,
            "name": "O",
            "dims": o_dims,
            "strides": compute_contiguous_strides(o_dims),
            "data_type": dtype_str,
            "virtual": False,
        },
        {
            "uid": UID_SCALE,
            "name": "scale",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            "is_runtime_pass_by_value": True,
        },
    ]

    graph = {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "q_tensor_uid": UID_Q,
                    "k_tensor_uid": UID_K,
                    "v_tensor_uid": UID_V,
                    "attn_mask_tensor_uid": None,
                    "scale_tensor_uid": UID_SCALE,  # runtime tensor
                    "seq_len_q_tensor_uid": None,
                    "seq_len_kv_tensor_uid": None,
                    "seed_tensor_uid": None,
                    "offset_tensor_uid": None,
                    "dropout_mask_tensor_uid": None,
                    "dropout_scale_tensor_uid": None,
                    "page_table_k_tensor_uid": None,
                    "page_table_v_tensor_uid": None,
                    "block_mask_tensor_uid": None,
                    "sink_token_tensor_uid": None,
                    "descale_q_tensor_uid": None,
                    "descale_k_tensor_uid": None,
                    "descale_v_tensor_uid": None,
                    "descale_s_tensor_uid": None,
                    "scale_s_tensor_uid": None,
                    "scale_o_tensor_uid": None,
                },
                "outputs": {
                    "o_tensor_uid": UID_O,
                    "stats_tensor_uid": None,
                    "max_tensor_uid": None,
                    "sum_exp_tensor_uid": None,
                    "rng_dump_tensor_uid": None,
                    "amax_s_tensor_uid": None,
                    "amax_o_tensor_uid": None,
                },
                "attributes": {
                    "generate_stats": None,
                    "alibi_mask": False,
                    "padding_mask": False,
                    "causal_mask": False,
                    "causal_mask_bottom_right": False,
                    "dropout_probability": None,
                    "attn_scale_value": None,  # null — scale is runtime, not baked
                    "left_bound": None,
                    "right_bound": None,
                    "max_seq_len_kv": None,
                    "diagonal_alignment": "TOP_LEFT",
                    "mma_core_mode": "float",
                    "implementation": "AUTO",
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
        "operation": "SdpaFwdRuntimeScale",
        "generator": "generate_sdpa_runtime_scale_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_backend": "pytorch_math_backend",
        "rocm_version": rocm_ver,
        "seed": config["seed"],
        "notes": (
            f"Runtime scale bundle: scale (uid={UID_SCALE}) is a runtime tensor "
            "(no value_type/value field in JSON; attn_scale_value attribute is null). "
            f"Detection value={config['scale_value']} is intentionally far from the "
            "compile-time default (1/sqrt(D_qk)) to expose providers that use a "
            "baked-in compile-time scale instead of the runtime tensor."
        ),
        "config": {
            "q_dims": config["q_dims"],
            "v_dims": config["v_dims"],
            "dtype": config["dtype"],
            "scale_uid": UID_SCALE,
            "scale_value": config["scale_value"],
            "gqa_ratio": config["q_dims"][1] // config["v_dims"][1],
        },
    }


def generate_bundle(
    base_filename,
    q_dims,
    v_dims,
    dtype="bf16",
    scale_value=1.0,
    seed=42,
    min_val=-1.0,
    max_val=1.0,
):
    if dtype not in DTYPE_MAP:
        print(
            f"ERROR: --dtype must be one of {list(DTYPE_MAP.keys())} (got '{dtype}')",
            file=sys.stderr,
        )
        sys.exit(1)

    B, H_q, S_q, D_qk = q_dims
    B_v, H_kv, S_kv, D_v = v_dims
    k_dims = [B, H_kv, S_kv, D_qk]
    o_dims = [B, H_q, S_q, D_v]

    errors = []
    if B_v != B:
        errors.append(f"Batch mismatch: Q batch={B}, V batch={B_v}")
    if H_q % H_kv != 0:
        errors.append(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
    if min_val >= max_val:
        errors.append(f"--min ({min_val}) must be less than --max ({max_val})")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    dtype_info = DTYPE_MAP[dtype]
    torch_dtype = dtype_info["torch"]

    os.makedirs(os.path.dirname(os.path.abspath(base_filename)), exist_ok=True)

    compile_time_default = 1.0 / math.sqrt(D_qk)
    print(f"Generating SDPA runtime scale bundle: {base_filename}")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O: {o_dims}")
    print(f"  dtype: {dtype}, seed: {seed}")
    print(
        f"  scale uid={UID_SCALE}, runtime value={scale_value} "
        f"(compile-time default would be {compile_time_default:.6f})"
    )

    rng = torch.Generator().manual_seed(seed)
    Q = torch.empty(q_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    K = torch.empty(k_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    V = torch.empty(v_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    scale_tensor = torch.full([1], scale_value, dtype=torch.float32)

    try:
        O = compute_forward(Q, K, V, scale_value, H_q, H_kv)
    except RuntimeError as e:
        print(f"ERROR: PyTorch SDPA failed: {e}", file=sys.stderr)
        sys.exit(1)

    for name, t in [("Q", Q), ("K", K), ("V", V), ("O", O)]:
        assert not torch.isnan(t).any(), f"NaN in {name}"
        assert not torch.isinf(t).any(), f"Inf in {name}"

    # Write tensor .bin files
    tensor_list = [
        ("Q", Q, UID_Q),
        ("K", K, UID_K),
        ("V", V, UID_V),
        ("O", O, UID_O),
        ("scale", scale_tensor, UID_SCALE),
    ]
    for name, tensor, uid in tensor_list:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_bytes = os.path.getsize(bin_path)
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_bytes} bytes"
        )

    # Write graph JSON
    graph_json = build_graph_json(q_dims, k_dims, v_dims, o_dims, dtype_info["json"])
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    # Write meta JSON
    config = {
        "q_dims": q_dims,
        "v_dims": v_dims,
        "dtype": dtype,
        "scale_value": scale_value,
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
        description="Generate SDPA forward runtime scale golden reference bundles"
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Path prefix for output files (no extension)",
    )
    parser.add_argument(
        "--q-dims",
        nargs=4,
        type=int,
        required=True,
        metavar=("B", "H_Q", "S_Q", "D_QK"),
        help="Query tensor dims: batch, heads_q, seq_q, head_dim_qk",
    )
    parser.add_argument(
        "--v-dims",
        nargs=4,
        type=int,
        required=True,
        metavar=("B", "H_KV", "S_KV", "D_V"),
        help="Value tensor dims: batch, heads_kv, seq_kv, head_dim_v",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Input/output tensor dtype (default: bf16)",
    )
    parser.add_argument(
        "--scale-value",
        type=float,
        default=1.0,
        help=(
            "Detection value for the runtime scale tensor (uid=4). "
            "Default=1.0 (no scaling) is far from the typical compile-time "
            "default of 1/sqrt(D_qk) to enable detection of incorrect handling."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    args = parser.parse_args()

    generate_bundle(
        base_filename=args.base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        dtype=args.dtype,
        scale_value=args.scale_value,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
