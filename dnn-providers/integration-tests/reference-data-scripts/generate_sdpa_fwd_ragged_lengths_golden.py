#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate ragged (RFC 0014 packed BSHD) SDPA forward golden bundles with explicit
per-batch Q and K/V lengths.

generate_sdpa_fwd_golden.py --ragged-offsets derives the per-batch lengths from
(B, S_q, S_kv) and always gives batch 0 the full S_max. This script takes the lengths
verbatim so a bundle can pin the shapes that stress how an engine sizes and addresses a
packed batch from the graph's S_max:

  * the longest sequence in any batch position (not only batch 0);
  * declared dims S_q / S_kv larger than every per-batch length (the data SDK only
    requires S_max >= each ragged_offset block, RaggedTensor.hpp validateRaggedOffsets);
  * zero-length batches and Q-only / KV-only empty batches;
  * per-batch Q and K/V lengths that differ, so bottom-right causal alignment must use
    each batch's own lengths (fully masked rows output 0, as CpuFpReferenceSdpaRagged).

The packed Q/K/V/O buffers hold exactly sum(lens) tokens and the ragged offsets step in
token units with ragged_offset_multiplier = the sequence stride (H * D), the layout
generate_sdpa_fwd_golden.py emits and ASM_SDPA_ENGINE accepts.

Example:

    python generate_sdpa_fwd_ragged_lengths_golden.py \\
        --base-filename .../hd128_nomask_ragged_long_last/hd128_nomask_ragged_long_last \\
        --heads 8 8 --head-dims 128 128 --seq-dims 1000 1100 \\
        --lens-q 1 2 ... 1000 --lens-kv 3 1 ... 1100
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, SCRIPT_DIR)
sys.path.insert(1, os.path.join(SCRIPT_DIR, "utilities"))
import torch  # noqa: E402
from generate_sdpa_fwd_golden import (  # noqa: E402
    UID_K,
    UID_O,
    UID_Q,
    UID_RAGGED_OFFSET_KV,
    UID_RAGGED_OFFSET_QO,
    UID_V,
    build_graph_json,
    ragged_offsets,
)
from sdpa_common import (  # noqa: E402
    build_mask,
    compute_forward,
    save_tensor_bin,
    script_sha256,
)

# Bump when the emitted tensors or the .meta.json schema change.
# 1.0.0 — Initial explicit-length ragged forward generator.
GENERATOR_VERSION = "1.0.0"


def packed_forward(Q, K, V, lens_q, lens_kv, scale, H_q, H_kv, causal):
    """Per-batch SDPA over packed [T, H, D] tensors; returns packed O [T_q, H_q, D_v].

    Rows with no attendable key (an empty K/V batch, or bottom-right causal rows above
    the diagonal when the batch's Q is longer than its K/V) output 0, the convention of
    CpuFpReferenceSdpaRagged.
    """
    D_v = V.shape[-1]
    O = torch.zeros((Q.shape[0], H_q, D_v), dtype=Q.dtype)
    q0 = k0 = 0
    for lq, lk in zip(lens_q, lens_kv):
        if lq > 0 and lk > 0:
            q_b = Q[q0 : q0 + lq].transpose(0, 1).unsqueeze(0)  # [1, H, lq, D]
            k_b = K[k0 : k0 + lk].transpose(0, 1).unsqueeze(0)
            v_b = V[k0 : k0 + lk].transpose(0, 1).unsqueeze(0)
            keep = build_mask(lq, lk, causal, -1, -1)
            o_b = compute_forward(q_b, k_b, v_b, scale, H_q, H_kv, keep)[0]
            if keep is not None:
                o_b[:, ~keep.any(dim=-1), :] = 0
            O[q0 : q0 + lq] = o_b.transpose(0, 1).to(Q.dtype)
        q0 += lq
        k0 += lk
    return O


def write_dvc_pointer(base_filename, uids):
    """Write {base}.tensors.dvc listing every .bin with its md5 and size."""
    lines = ["outs:"]
    for uid in sorted(uids, key=str):
        path = f"{base_filename}.tensor{uid}.bin"
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        lines += [
            f"- path: {os.path.basename(path)}",
            "  remote: golden-data",
            f"  md5: {md5}",
            f"  size: {os.path.getsize(path)}",
            "  hash: md5",
        ]
    with open(f"{base_filename}.tensors.dvc", "w") as f:
        f.write("\n".join(lines) + "\n")


def generate(args):
    lens_q, lens_kv = args.lens_q, args.lens_kv
    H_q, H_kv = args.heads
    D_qk, D_v = args.head_dims
    S_q, S_kv = args.seq_dims
    B = len(lens_q)

    errors = []
    if len(lens_kv) != B:
        errors.append(f"--lens-q has {B} entries but --lens-kv has {len(lens_kv)}")
    if H_q % H_kv != 0:
        errors.append(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
    if any(l < 0 for l in lens_q + lens_kv):
        errors.append("per-batch lengths must be >= 0")
    # RaggedTensor.hpp validateRaggedOffsets: every block fits in the declared S_max.
    if max(lens_q) > S_q or max(lens_kv) > S_kv:
        errors.append(
            f"declared S_q/S_kv ({S_q}/{S_kv}) must be >= every per-batch length "
            f"(max {max(lens_q)}/{max(lens_kv)})"
        )
    if sum(lens_q) == 0 or sum(lens_kv) == 0:
        errors.append("at least one Q and one K/V token are needed")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    scale = args.attn_scale if args.attn_scale is not None else 1.0
    rng = torch.Generator().manual_seed(args.seed)
    T_q, T_kv = sum(lens_q), sum(lens_kv)
    Q = torch.empty((T_q, H_q, D_qk), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=rng
    )
    K = torch.empty((T_kv, H_kv, D_qk), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=rng
    )
    V = torch.empty((T_kv, H_kv, D_v), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=rng
    )
    O = packed_forward(Q, K, V, lens_q, lens_kv, scale, H_q, H_kv, args.causal)
    assert torch.isfinite(O.float()).all(), "non-finite reference output"

    base = args.base_filename
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    tensors = [
        (UID_Q, Q),
        (UID_K, K),
        (UID_V, V),
        (UID_O, O),
        (UID_RAGGED_OFFSET_QO, ragged_offsets(lens_q)),
        (UID_RAGGED_OFFSET_KV, ragged_offsets(lens_kv)),
    ]
    for uid, t in tensors:
        save_tensor_bin(t, f"{base}.tensor{uid}.bin")

    graph = build_graph_json(
        [B, H_q, S_q, D_qk],
        [B, H_kv, S_kv, D_qk],
        [B, H_kv, S_kv, D_v],
        [B, H_q, S_q, D_v],
        scale,
        dtype_str="bfloat16",
        causal=args.causal,
        layout="bshd",
        ragged=True,
    )
    with open(f"{base}.json", "w") as f:
        json.dump(graph, f, indent=4)
        f.write("\n")

    version = torch.__version__
    meta = {
        "format_version": 1,
        "generator": os.path.basename(__file__),
        "generator_sha256": script_sha256(os.path.abspath(__file__)),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {version}",
        "reference_backend": "pytorch_math_backend",
        "rocm_version": version.split("+rocm")[1] if "+rocm" in version else "",
        "generator_version": GENERATOR_VERSION,
        "generation_precision": "like-for-like: BF16 inputs, FP32 intermediates",
        "direction": "forward",
        "seed": args.seed,
        "input_range": [-1.0, 1.0],
        "deterministic": True,
        "description": args.description,
        "config": {
            "batch": B,
            "num_heads_q": H_q,
            "num_heads_kv": H_kv,
            "seq_q": S_q,
            "seq_kv": S_kv,
            "head_dim_qk": D_qk,
            "head_dim_v": D_v,
            "dtype": "bf16",
            "causal": args.causal,
            "stats": False,
            "scale": scale,
            "gqa_ratio": H_q // H_kv,
            "layout": "bshd",
            "ragged": True,
            "ragged_physical_lens": {"q": lens_q, "kv": lens_kv},
            "total_tokens": {"q": T_q, "kv": T_kv},
        },
    }
    with open(f"{base}.meta.json", "w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")

    support = f"{base}.support.json"
    if not os.path.exists(support):
        with open(support, "w") as f:
            json.dump({"claims": {}, "version": 1}, f, indent=2)
            f.write("\n")

    write_dvc_pointer(base, [uid for uid, _ in tensors])
    print(
        f"{base}: B={B} H={H_q}/{H_kv} D={D_qk}/{D_v} S_dims={S_q}/{S_kv} "
        f"max={max(lens_q)}/{max(lens_kv)} total={T_q}/{T_kv} causal={args.causal}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-filename", required=True)
    parser.add_argument(
        "--heads", nargs=2, type=int, required=True, metavar=("H_Q", "H_KV")
    )
    parser.add_argument(
        "--head-dims", nargs=2, type=int, required=True, metavar=("D_QK", "D_V")
    )
    parser.add_argument(
        "--seq-dims",
        nargs=2,
        type=int,
        required=True,
        metavar=("S_Q", "S_KV"),
        help="Declared S_max of Q/O and K/V (the graph's dims[2]); >= every length",
    )
    parser.add_argument("--lens-q", nargs="+", type=int, required=True)
    parser.add_argument("--lens-kv", nargs="+", type=int, required=True)
    parser.add_argument("--causal", default="none", choices=["none", "bottom_right"])
    parser.add_argument("--attn-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--description",
        default="",
        help="What the bundle is meant to catch (meta only)",
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
