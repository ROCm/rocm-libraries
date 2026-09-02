#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SDPA Forward Runtime Scale Test Bundle Generator

Generates graph-only reference bundles for SDPA forward where the attention
scale is provided as a runtime pass-by-value tensor (dims=[1], strides=[1])
rather than a compile-time baked scalar (attn_scale_value attribute).

In existing SDPA bundles the scale is baked into the node's `attn_scale_value`
attribute. This script instead wires it as a runtime tensor via `scale_tensor_uid`,
leaving `attn_scale_value` as null. A provider that ignores the runtime scale
and falls back to a compile-time default will produce detectably wrong output.

These bundles carry no tensor blobs: the harness synthesizes Q/K/V/etc. itself
and verifies provider output against its own CPU reference executor. The only
thing this generator needs to communicate beyond the graph shape is which
runtime tensor should be pinned to a fixed detection value — done via the
`inputs` field in the companion `.meta.json`. There is no golden data to
compute, so this script has no numeric/tensor-framework dependency.

Tensor UID slots (fixed convention across this bundle family):
  0=Q  1=K  2=V  3=O  4=stats(LSE, if --stats)  5=SeqLenQ  6=SeqLenKv (if
  --variable-seq-lens)  7/8/9=DescaleQ/K/V (if --dtype fp8)  10=scale (always
  — the scale tensor's UID is never referenced positionally, only via the
  node's explicit scale_tensor_uid field, so a single fixed slot above the
  highest optional-tensor slot avoids any collision regardless of flags).

Output: {base_filename}.json + {base_filename}.meta.json

Usage:
    python generate_sdpa_runtime_scale_test_bundle.py \\
        --base-filename bundles/SdpaFwdRuntimeScale/bhsd/bf16/hd128_nomask_batch/Small \\
        --q-dims 2 4 256 128 --v-dims 2 4 256 128 \\
        --name SdpaFwdRuntimeScale_bhsd_bf16_hd128_nomask_batch_Small
"""

import argparse
import datetime
import json
import os
import sys

# Bump when generator logic changes in a way that affects output data.
# 1.0.0 — Initial: SDPA forward with runtime scale tensor (dynamic UID: 4
#         normally, displaced to 10 when stats or fp8 also claim slot 4/7-9),
#         with --causal (bottom_right), --variable-seq-lens (group/ragged
#         batch), --stats (LSE), and --dtype fp8 (+ descale) support. Carries
#         no tensor blobs and no PyTorch dependency: this bundle family's
#         inputs are synthesized and outputs are reference-verified by the
#         harness at test time, so there is no golden data for this script to
#         compute.
GENERATOR_VERSION = "1.0.0"

DTYPE_JSON = {
    "bf16": "bfloat16",
    "fp16": "half",
    "fp8": "fp8_e4m3",
}

# Tensor UID assignment (stable contract with the JSON consumer / loader).
UID_Q = 0
UID_K = 1
UID_V = 2
UID_O = 3
UID_STATS = 4
UID_SEQ_LEN_Q = 5
UID_SEQ_LEN_KV = 6
UID_DESCALE_Q = 7
UID_DESCALE_K = 8
UID_DESCALE_V = 9
UID_SCALE = 10


def compute_contiguous_strides(dims):
    strides = []
    stride = 1
    for d in reversed(dims):
        strides.append(stride)
        stride *= d
    strides.reverse()
    return strides


def build_graph_json(q_dims, k_dims, v_dims, o_dims, dtype, causal, group, stats, name):
    """Build the SDPA graph JSON with scale wired as a runtime tensor.

    Key difference from the compile-time variant (generate_sdpa_fwd_golden.py):
      - scale_tensor_uid points at the runtime scale tensor (no baked fallback)
      - attn_scale_value = null (forces the CPU reference to read from variantPack)
    A provider that falls back to a compile-time default (e.g. 1/sqrt(headDim))
    instead of reading the runtime tensor produces output that diverges from the
    CPU reference.
    """
    B = q_dims[0]
    is_fp8 = dtype == "fp8"
    qkv_json_dtype = DTYPE_JSON[dtype]
    # fp8 attention output is bf16; io_data_type mirrors that (matches the
    # committed fp8 bundles, even though Q/K/V are individually fp8_e4m3).
    o_json_dtype = "bfloat16" if is_fp8 else qkv_json_dtype
    io_json_dtype = o_json_dtype

    scale_uid = UID_SCALE

    tensors = [
        {
            "uid": UID_Q,
            "name": "Q",
            "dims": q_dims,
            "strides": compute_contiguous_strides(q_dims),
            "data_type": qkv_json_dtype,
            "virtual": False,
        },
        {
            "uid": UID_K,
            "name": "K",
            "dims": k_dims,
            "strides": compute_contiguous_strides(k_dims),
            "data_type": qkv_json_dtype,
            "virtual": False,
        },
        {
            "uid": UID_V,
            "name": "V",
            "dims": v_dims,
            "strides": compute_contiguous_strides(v_dims),
            "data_type": qkv_json_dtype,
            "virtual": False,
        },
        {
            "uid": UID_O,
            "name": "O",
            "dims": o_dims,
            "strides": compute_contiguous_strides(o_dims),
            "data_type": o_json_dtype,
            "virtual": False,
        },
    ]

    stats_tensor_uid = None
    if stats:
        lse_dims = [B, q_dims[1], q_dims[2], 1]
        tensors.append(
            {
                "uid": UID_STATS,
                "name": "LSE",
                "dims": lse_dims,
                "strides": compute_contiguous_strides(lse_dims),
                "data_type": "float",
                "virtual": False,
            }
        )
        stats_tensor_uid = UID_STATS

    seq_len_q_tensor_uid = None
    seq_len_kv_tensor_uid = None
    if group:
        for uid, tname in [(UID_SEQ_LEN_Q, "SeqLenQ"), (UID_SEQ_LEN_KV, "SeqLenKv")]:
            tensors.append(
                {
                    "uid": uid,
                    "name": tname,
                    "dims": [B + 1],
                    "strides": [1],
                    "data_type": "int32",
                    "virtual": False,
                }
            )
        seq_len_q_tensor_uid = UID_SEQ_LEN_Q
        seq_len_kv_tensor_uid = UID_SEQ_LEN_KV

    descale_q_tensor_uid = descale_k_tensor_uid = descale_v_tensor_uid = None
    if is_fp8:
        for uid, tname in [
            (UID_DESCALE_Q, "DescaleQ"),
            (UID_DESCALE_K, "DescaleK"),
            (UID_DESCALE_V, "DescaleV"),
        ]:
            tensors.append(
                {
                    "uid": uid,
                    "name": tname,
                    "dims": [1],
                    "strides": [1],
                    "data_type": "float",
                    "virtual": False,
                }
            )
        descale_q_tensor_uid = UID_DESCALE_Q
        descale_k_tensor_uid = UID_DESCALE_K
        descale_v_tensor_uid = UID_DESCALE_V

    tensors.append(
        {
            "uid": scale_uid,
            "name": "scale",
            "dims": [1],
            "strides": [1],
            "data_type": "float",
            "virtual": False,
            "is_runtime_pass_by_value": True,
        }
    )

    causal_mask_bottom_right = causal == "bottom_right"
    diagonal_alignment = "BOTTOM_RIGHT" if causal_mask_bottom_right else "TOP_LEFT"
    left_bound = -1 if causal_mask_bottom_right else None
    right_bound = 0 if causal_mask_bottom_right else None

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
                    "scale_tensor_uid": scale_uid,  # runtime tensor
                    "seq_len_q_tensor_uid": seq_len_q_tensor_uid,
                    "seq_len_kv_tensor_uid": seq_len_kv_tensor_uid,
                    "seed_tensor_uid": None,
                    "offset_tensor_uid": None,
                    "dropout_mask_tensor_uid": None,
                    "dropout_scale_tensor_uid": None,
                    "page_table_k_tensor_uid": None,
                    "page_table_v_tensor_uid": None,
                    "block_mask_tensor_uid": None,
                    "sink_token_tensor_uid": None,
                    "descale_q_tensor_uid": descale_q_tensor_uid,
                    "descale_k_tensor_uid": descale_k_tensor_uid,
                    "descale_v_tensor_uid": descale_v_tensor_uid,
                    "descale_s_tensor_uid": None,
                    "scale_s_tensor_uid": None,
                    "scale_o_tensor_uid": None,
                },
                "outputs": {
                    "o_tensor_uid": UID_O,
                    "stats_tensor_uid": stats_tensor_uid,
                    "max_tensor_uid": None,
                    "sum_exp_tensor_uid": None,
                    "rng_dump_tensor_uid": None,
                    "amax_s_tensor_uid": None,
                    "amax_o_tensor_uid": None,
                },
                "attributes": {
                    "generate_stats": True if stats else None,
                    "alibi_mask": False,
                    "padding_mask": False,
                    "causal_mask": False,
                    "causal_mask_bottom_right": causal_mask_bottom_right,
                    "dropout_probability": None,
                    "attn_scale_value": None,  # null — scale is runtime, not baked
                    "left_bound": left_bound,
                    "right_bound": right_bound,
                    "max_seq_len_kv": None,
                    "diagonal_alignment": diagonal_alignment,
                    "mma_core_mode": "float",
                    "implementation": "AUTO",
                },
            }
        ],
        "tensors": tensors,
        "io_data_type": io_json_dtype,
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": name,
    }
    return graph, scale_uid


def build_meta_json(scale_uid, scale_value):
    return {
        "format_version": 1,
        "operation": "SdpaFwdRuntimeScale",
        "generator": "generate_sdpa_runtime_scale_test_bundle.py",
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": "synthesis+cpu_reference",
        "notes": (
            f"Pure-runtime scale bundle: scale (uid={scale_uid}) has "
            "is_runtime_pass_by_value=true with no baked value. The harness "
            f"supplies the detection value ({scale_value}) via the inputs "
            "field; both the CPU reference and the provider read it from the "
            "variantPack. A provider that ignores the runtime tensor and "
            "falls back to a hardcoded scale produces diverging output."
        ),
        "inputs": {str(scale_uid): {"kind": "fixed", "value": scale_value}},
    }


def generate_bundle(
    base_filename,
    q_dims,
    v_dims,
    dtype="bf16",
    causal="none",
    group=False,
    stats=False,
    scale_value=2.0,
    name="",
):
    if dtype not in DTYPE_JSON:
        print(
            f"ERROR: --dtype must be one of {list(DTYPE_JSON.keys())} (got '{dtype}')",
            file=sys.stderr,
        )
        sys.exit(1)
    if causal not in ("none", "bottom_right"):
        print(
            f"ERROR: --causal must be one of ('none', 'bottom_right') (got '{causal}')",
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
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(base_filename)), exist_ok=True)

    print(f"Generating SDPA runtime scale bundle: {base_filename}")
    print(f"  Q: {q_dims}, K: {k_dims}, V: {v_dims}, O: {o_dims}")
    print(f"  dtype: {dtype}, causal: {causal}, group: {group}, stats: {stats}")

    graph_json, scale_uid = build_graph_json(
        q_dims, k_dims, v_dims, o_dims, dtype, causal, group, stats, name
    )
    print(f"  scale uid={scale_uid}, detection value={scale_value}")

    json_path = f"{base_filename}.json"
    with open(json_path, "w") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    meta_json = build_meta_json(scale_uid, scale_value)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SDPA forward runtime scale test bundles (graph + meta only)"
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
        choices=list(DTYPE_JSON.keys()),
        help="Q/K/V tensor dtype (default: bf16). 'fp8' emits fp8_e4m3 "
        "Q/K/V with a bf16 output and per-tensor descale_q/k/v runtime tensors.",
    )
    parser.add_argument(
        "--causal",
        default="none",
        choices=["none", "bottom_right"],
        help="Causal masking mode (default: none).",
    )
    parser.add_argument(
        "--variable-seq-lens",
        action="store_true",
        help="Declare runtime seq_len_q/seq_len_kv tensors (uid 5/6) for a "
        "ragged/padded batch, in addition to Q/K/V/O.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Declare a runtime stats (LSE) output tensor (uid=4).",
    )
    parser.add_argument(
        "--scale-value",
        type=float,
        default=2.0,
        help=(
            "Detection value for the runtime scale tensor, recorded in "
            "meta.json's 'inputs' field for the harness to inject. Default=2.0 "
            "is far from the typical compile-time default of 1/sqrt(D_qk) to "
            "enable detection of incorrect handling."
        ),
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Graph-level 'name' field, e.g. "
        "SdpaFwdRuntimeScale_bhsd_bf16_hd128_causal_batch_Small",
    )
    args = parser.parse_args()

    generate_bundle(
        base_filename=args.base_filename,
        q_dims=args.q_dims,
        v_dims=args.v_dims,
        dtype=args.dtype,
        causal=args.causal,
        group=args.variable_seq_lens,
        stats=args.stats,
        scale_value=args.scale_value,
        name=args.name,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
