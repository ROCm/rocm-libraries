#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Forward Golden Reference Bundle Generator

Generates pre-computed reference data for SDPA forward kernel validation.
Uses PyTorch's SDPBackend.MATH as the golden reference source.
Supports BF16 and FP16 data types.

Output: {base_filename}.json + {base_filename}.tensor{uid}.bin + {base_filename}.meta.json

Usage:
    python generate_sdpa_fwd_golden.py \
        --base-filename golden_data/SdpaFwd/bf16_hd128_nomask_batch/Small \
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 --seed 42

    python generate_sdpa_fwd_golden.py \
        --base-filename golden_data/SdpaFwd/fp16_hd128_nomask_batch/Small \
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 --dtype fp16 --seed 42
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# Bump when generator logic changes in a way that affects output data.
# (e.g., different reference backend, precision handling, tensor layout)
GENERATOR_VERSION = "1.0.0"

DTYPE_MAP = {
    "bf16": {"torch": torch.bfloat16, "json": "bfloat16", "bytes": 2},
    "fp16": {"torch": torch.float16, "json": "float16", "bytes": 2},
}


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def compute_forward(Q, K, V, scale, H_q, H_kv):
    """SDPA forward via PyTorch Math backend (half->FP32->half)."""
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


def _get_generator_sha256():
    """SHA-256 of this script's contents — git-independent version marker."""
    script_path = os.path.abspath(__file__)
    with open(script_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_meta_json(config, pytorch_version):
    rocm_ver = ""
    if "+rocm" in pytorch_version:
        rocm_ver = pytorch_version.split("+rocm")[1]

    return {
        "generator": "generate_sdpa_fwd_golden.py",
        "generator_sha256": _get_generator_sha256(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_backend": "pytorch_math_backend",
        "rocm_version": rocm_ver,
        "generator_version": GENERATOR_VERSION,
        "generation_precision": f"like-for-like: {config['dtype'].upper()} inputs, FP32 intermediates",
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
            "dtype": config["dtype"],
            "causal": False,
            "scale": config["scale"],
            "gqa_ratio": config["q_dims"][1] // config["v_dims"][1],
        },
    }


AITER_VERSION = "0.1.13"
AITER_WHEEL_BASE = f"https://github.com/ROCm/aiter/releases/download/v{AITER_VERSION}"


def _get_rocm_version():
    """Extract ROCm version (e.g. '7.2') from the installed PyTorch build."""
    version = torch.__version__
    if "+rocm" in version:
        return version.split("+rocm")[1]
    return ""


def _get_python_tag():
    """Return cpython tag like 'cp310' or 'cp312'."""
    v = sys.version_info
    return f"cp{v.major}{v.minor:02d}"


def install_aiter():
    """Install the aiter wheel matching the system's ROCm version and Python."""
    rocm_ver = _get_rocm_version()
    if not rocm_ver:
        print(
            "ERROR: Cannot detect ROCm version from PyTorch. "
            "Install a ROCm-enabled PyTorch first.",
            file=sys.stderr,
        )
        sys.exit(1)

    py_tag = _get_python_tag()
    wheel_name = (
        f"amd_aiter-{AITER_VERSION}+rocm{rocm_ver}.manylinux.2.28"
        f"-{py_tag}-{py_tag}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    wheel_url = f"{AITER_WHEEL_BASE}/{wheel_name}"

    print(f"Installing aiter v{AITER_VERSION} for ROCm {rocm_ver} (Python {py_tag})")
    print(f"  URL: {wheel_url}")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_url],
        capture_output=False,
    )
    if result.returncode != 0:
        print(
            f"ERROR: pip install failed (exit code {result.returncode})",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  aiter v{AITER_VERSION} installed successfully.")


def validate_against_aiter(
    Q_bhsd,
    K_bhsd,
    V_bhsd,
    golden_O,
    scale,
    causal=False,
    window_left=-1,
    window_right=-1,
    stats=False,
):
    """Cross-check golden output against AITER fmha_v3_fwd GPU kernel.

    Layout:
      - Golden data uses BHSD (batch, heads, seq, dim) — PyTorch SDPA convention
      - AITER fmha_v3_fwd uses BSHD (batch, seq, heads, dim) — transpose dims 1,2

    Why transpose instead of stride manipulation:
      AITER's Python fmha_v3_fwd API requires contiguous BSHD tensors — it does
      not accept arbitrary strides. The hip-kernel-provider's C++ path, by contrast,
      passes raw per-axis strides (s_Seqs, s_Hs, s_Bs) to the kernel and handles
      any layout without transposing. These are two different interfaces to the same
      ASM kernel: Python API enforces BSHD shape, C++ path is stride-agnostic.
    """

    try:
        import aiter
        from aiter import fmha_v3_fwd

        aiter_ver = getattr(aiter, "__version__", "unknown")
        print(f"  AITER version: {aiter_ver}")
        print(
            f"  Args: is_causal={causal}, window=({window_left},{window_right}), "
            f"return_softmax_lse={stats}"
        )

        Q_bshd = Q_bhsd.cuda().transpose(1, 2).contiguous()
        K_bshd = K_bhsd.cuda().transpose(1, 2).contiguous()
        V_bshd = V_bhsd.cuda().transpose(1, 2).contiguous()

        O_aiter_bshd, _, _, _ = fmha_v3_fwd(
            Q_bshd,
            K_bshd,
            V_bshd,
            dropout_p=0.0,
            softmax_scale=scale,
            is_causal=causal,
            window_size_left=window_left,
            window_size_right=window_right,
            return_softmax_lse=stats,
            return_dropout_randval=False,
            how_v3_bf16_cvt=0,  # RTNE — matches hip-kernel-provider
        )
        O_aiter_bhsd = O_aiter_bshd.transpose(1, 2).contiguous().cpu()

        diff_aiter = (golden_O.float() - O_aiter_bhsd.float()).abs()
        max_aiter = diff_aiter.max().item()
        print(
            f"  AITER v3 fwd:  max_abs={max_aiter:.6f}  "
            f"p99={diff_aiter.quantile(0.99).item():.6f}  "
            f"mean={diff_aiter.mean().item():.6f}"
        )
        if max_aiter > 0.01:
            print(f"  WARNING: Golden vs AITER exceeds 0.01 (got {max_aiter:.6f})")
            return False
        return True
    except ImportError:
        print("  AITER:         SKIPPED (aiter not installed)")
        return True


def generate_forward_bundle(
    base_filename,
    q_dims,
    v_dims,
    dtype="bf16",
    causal=False,
    window_left=-1,
    window_right=-1,
    stats=False,
    seed=42,
    min_val=-1.0,
    max_val=1.0,
    attn_scale=None,
    validate=False,
):
    B, H_q, S_q, D_qk = q_dims
    B_v, H_kv, S_kv, D_v = v_dims
    k_dims = [B, H_kv, S_kv, D_qk]
    o_dims = [B, H_q, S_q, D_v]

    if dtype not in DTYPE_MAP:
        print(
            f"ERROR: --dtype must be one of {list(DTYPE_MAP.keys())} (got '{dtype}')",
            file=sys.stderr,
        )
        sys.exit(1)
    dtype_info = DTYPE_MAP[dtype]
    torch_dtype = dtype_info["torch"]

    errors = []
    for name, dims in [("q", q_dims), ("v", v_dims)]:
        for i, d in enumerate(dims):
            if d <= 0:
                errors.append(f"--{name}-dims[{i}] must be positive (got {d})")
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

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(D_qk)

    os.makedirs(os.path.dirname(base_filename) or ".", exist_ok=True)

    print(f"Generating forward bundle: {base_filename}")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O: {o_dims}")
    print(f"  dtype: {dtype} ({torch_dtype})")
    print(f"  H_q={H_q}, H_kv={H_kv}, GQA ratio={H_q // H_kv}")
    print(f"  Scale: {attn_scale:.10f}, Seed: {seed}")

    rng = torch.Generator().manual_seed(seed)

    Q = torch.empty(q_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    K = torch.empty(k_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)
    V = torch.empty(v_dims, dtype=torch_dtype).uniform_(min_val, max_val, generator=rng)

    try:
        O = compute_forward(Q, K, V, attn_scale, H_q, H_kv)
    except RuntimeError as e:
        print(f"ERROR: PyTorch SDPA failed: {e}", file=sys.stderr)
        sys.exit(1)

    for name, tensor in [("Q", Q), ("K", K), ("V", V), ("O", O)]:
        assert not torch.isnan(tensor).any(), f"NaN in {name}"
        assert not torch.isinf(tensor).any(), f"Inf in {name}"

    # Write raw tensor data as .bin files (one per tensor UID)
    tensor_list = [("Q", Q, 0), ("K", K, 1), ("V", V, 2), ("O", O, 3)]
    for name, tensor, uid in tensor_list:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_kb:.1f} KB"
        )

    # Write graph JSON (operation definition: node type, tensor metadata, attributes)
    graph_json = build_graph_json(
        q_dims, k_dims, v_dims, o_dims, attn_scale, dtype_str=dtype_info["json"]
    )
    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    # Write meta JSON (provenance: generator version, reference source, config)
    config = {
        "q_dims": q_dims,
        "v_dims": v_dims,
        "dtype": dtype,
        "seed": seed,
        "min_val": min_val,
        "max_val": max_val,
        "scale": attn_scale,
    }
    meta_json = build_meta_json(config, torch.__version__)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")

    # Optional: cross-check golden output against AITER GPU kernel
    if validate:
        ok = validate_against_aiter(
            Q,
            K,
            V,
            O,
            attn_scale,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
            stats=stats,
        )
        if ok:
            print("  Validation: PASSED")
        else:
            print("  Validation: FAILED — check warnings above")


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
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=list(DTYPE_MAP.keys()),
        help="Input/output tensor dtype (default: bf16)",
    )
    parser.add_argument(
        "--causal",
        default="none",
        choices=[
            "none"
        ],  # TODO: add "top_left", "bottom_right" when causal kernels are imported
        help="Causal mask type (default: none)",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="Sliding window left bound, -1 = unbounded (default: -1)",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=-1,
        help="Sliding window right bound, -1 = unbounded (default: -1)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Enable LSE output tensor (uid=4, shape [B, H_q, S_q], dtype FP32)",
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-check golden output against AITER fmha_v3_fwd (requires GPU)",
    )
    parser.add_argument(
        "--install-aiter",
        action="store_true",
        help=f"Install aiter v{AITER_VERSION} wheel for the detected ROCm version",
    )
    args = parser.parse_args()

    if args.install_aiter:
        install_aiter()

    generate_forward_bundle(
        base_filename=args.base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        dtype=args.dtype,
        causal=(args.causal != "none"),
        window_left=args.window_left,
        window_right=args.window_right,
        stats=args.stats,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        attn_scale=args.attn_scale,
        validate=args.validate,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
