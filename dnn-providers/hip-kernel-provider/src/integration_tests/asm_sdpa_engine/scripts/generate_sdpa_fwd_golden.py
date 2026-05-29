#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Forward Golden Reference Bundle Generator

Generates pre-computed reference data for SDPA forward kernel validation.
Uses PyTorch's math backend with like-for-like precision (BF16 inputs,
FP32 intermediates) to match the AITER ASM kernel's rounding trajectory.

Output: {base_filename}.json + {base_filename}.tensor{uid}.bin + {base_filename}.meta.json

Usage:
    python generate_sdpa_fwd_golden.py \
        --base-filename golden_data/SdpaFwd/bf16/Small \
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def compute_forward(Q_bf16, K_bf16, V_bf16, scale, H_q, H_kv):
    """SDPA forward: BF16 inputs, FP32 intermediates, BF16 output."""
    Q_f = Q_bf16.float()
    K_f = K_bf16.float()
    V_f = V_bf16.float()

    if H_q != H_kv:
        K_f = K_f.repeat_interleave(H_q // H_kv, dim=1)
        V_f = V_f.repeat_interleave(H_q // H_kv, dim=1)

    scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    O = torch.matmul(P, V_f).to(torch.bfloat16)
    return O


def save_tensor_bin(tensor, path):
    t = tensor.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.uint8).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw)


def build_graph_json(q_dims, k_dims, v_dims, o_dims, scale, dtype_str="bfloat16"):
    tensors = []
    for uid, name, dims in [
        (0, "Q", q_dims),
        (1, "K", k_dims),
        (2, "V", v_dims),
        (3, "O", o_dims),
    ]:
        tensors.append(
            {
                "uid": uid,
                "name": name,
                "dims": dims,
                "strides": compute_contiguous_strides(dims),
                "data_type": dtype_str,
                "virtual": False,
            }
        )

    graph = {
        "nodes": [
            {
                "type": "SdpaAttributes",
                "compute_data_type": "float",
                "name": "",
                "inputs": {
                    "q_tensor_uid": 0,
                    "k_tensor_uid": 1,
                    "v_tensor_uid": 2,
                    "attn_mask_tensor_uid": None,
                    "scale_tensor_uid": None,
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
                    "o_tensor_uid": 3,
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
                    "attn_scale_value": scale,
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


def build_meta_json(config, pytorch_version):
    return {
        "generator": "generate_sdpa_fwd_golden.py",
        "generator_version": "1.0.0",
        "reference_source": "pytorch_math_backend",
        "pytorch_version": pytorch_version,
        "generation_precision": "like-for-like: BF16 inputs, FP32 intermediates",
        "direction": "forward",
        "seed": config["seed"],
        "input_range": [config["min_val"], config["max_val"]],
        "deterministic": True,
        "config": {
            "batch": config["q_dims"][0],
            "num_heads_q": config["q_dims"][1],
            "num_heads_kv": config["v_dims"][1],
            "seq_q": config["q_dims"][2],
            "seq_kv": config["v_dims"][2],
            "head_dim_qk": config["q_dims"][3],
            "head_dim_v": config["v_dims"][3],
            "dtype": "bf16",
            "causal": False,
            "scale": config["scale"],
            "gqa_ratio": config["q_dims"][1] // config["v_dims"][1],
        },
    }


def generate_forward_bundle(
    base_filename, q_dims, v_dims, seed=42, min_val=-1.0, max_val=1.0, attn_scale=None
):
    B, H_q, S_q, D_qk = q_dims
    _, H_kv, S_kv, D_v = v_dims
    k_dims = [B, H_kv, S_kv, D_qk]
    o_dims = [B, H_q, S_q, D_v]

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(D_qk)

    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"

    os.makedirs(os.path.dirname(base_filename) or ".", exist_ok=True)

    print(f"Generating forward bundle: {base_filename}")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O: {o_dims}")
    print(f"  H_q={H_q}, H_kv={H_kv}, GQA ratio={H_q // H_kv}")
    print(f"  Scale: {attn_scale:.10f}, Seed: {seed}")

    rng = torch.Generator().manual_seed(seed)

    Q = (
        torch.empty(q_dims, dtype=torch.float32)
        .uniform_(min_val, max_val, generator=rng)
        .to(torch.bfloat16)
    )
    K = (
        torch.empty(k_dims, dtype=torch.float32)
        .uniform_(min_val, max_val, generator=rng)
        .to(torch.bfloat16)
    )
    V = (
        torch.empty(v_dims, dtype=torch.float32)
        .uniform_(min_val, max_val, generator=rng)
        .to(torch.bfloat16)
    )

    O = compute_forward(Q, K, V, attn_scale, H_q, H_kv)

    for name, tensor in [("Q", Q), ("K", K), ("V", V), ("O", O)]:
        assert not torch.isnan(tensor).any(), f"NaN in {name}"
        assert not torch.isinf(tensor).any(), f"Inf in {name}"

    tensor_list = [("Q", Q, 0), ("K", K, 1), ("V", V, 2), ("O", O, 3)]
    for name, tensor, uid in tensor_list:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_kb:.1f} KB"
        )

    graph_json = build_graph_json(q_dims, k_dims, v_dims, o_dims, attn_scale)
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
    print(f"  Graph JSON: {json_path}")

    config = {
        "q_dims": q_dims,
        "v_dims": v_dims,
        "seed": seed,
        "min_val": min_val,
        "max_val": max_val,
        "scale": attn_scale,
    }
    meta_json = build_meta_json(config, torch.__version__)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=4)
    print(f"  Meta JSON: {meta_path}")

    total_bytes = sum(
        os.path.getsize(f"{base_filename}.tensor{uid}.bin") for _, _, uid in tensor_list
    )
    print(f"  Total tensor data: {total_bytes / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDPA forward golden reference bundles"
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    parser.add_argument(
        "--attn-scale",
        type=float,
        default=None,
        help="Attention scale (default: 1/sqrt(D_qk))",
    )
    args = parser.parse_args()

    generate_forward_bundle(
        base_filename=args.base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        attn_scale=args.attn_scale,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
