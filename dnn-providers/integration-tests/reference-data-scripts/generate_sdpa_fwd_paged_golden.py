#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Paged-KV SDPA Forward Golden Reference Bundle Generator

Writes a golden bundle for a paged K/V cache SDPA forward graph (vLLM paged
attention, cuDNN paged semantics), with dense or ragged (packed RFC 0014) Q.

Graph (logical dims are always [B, H, S, D], cuDNN convention):
  - Q [B, Hq, Sq, D] / O [B, Hq, Sq, Dv]: BSHD strides. Dense, or ragged with
    ragged_offset_tensor_uid 10 (token offsets, multiplier = token stride) packed
    with no gaps between batches; ragged Q needs --seq-lens-q (Sq = max of them).
  - K / V block containers [num_blocks, Hkv, block_size, D(v)]. The physical
    layout is vLLM's [num_blocks, block_size, Hkv, D]. --interleaved-kv doubles the
    pitch of every D row to 2*D, so each container keeps its values in [.., 0:D]
    and a finite filler in the other half (arbitrary-stride coverage).
  - page_table_k (uid 12) / page_table_v (uid 13): INT32 [B, 1, M, 1], strides
    [M, M, 1, 1] (vLLM block_table zero copy), M = ceil(max_seq_len_kv /
    block_size). --shared-page-table binds uid 12 to both.
  - seq_len_kv (uid 6) always; seq_len_q (uid 5) with ragged Q.
  - max_seq_len_kv attribute; diagonal band mask (left/right bound + alignment).
  - Optional LSE (uid 4) [B, Hq, Sq, 1] float: dense for dense Q, packed with
    the Q offsets (multiplier Hq) for ragged Q.
  - Optional attention sink (uid 14, --sinks) [1, Hq, 1, 1], strides [Hq, 1, 1, 1]
    (vLLM layer.sinks [Hq] viewed zero copy): one raw logit per query head, drawn
    N(0, 1), in the io dtype (--sink-dtype io, what the engines take) or FLOAT.

Data: Q/K/V uniform(min, max) from a seeded torch.Generator. Physical blocks are
a random permutation of distinct block ids; unused page-table entries are -1.
The containers hold a few spare blocks, and the spare blocks plus the unused
tail of each partial last block hold SPARE_FILL, so a wrong read shows up.

Golden (like-for-like): per batch, gather the pages of the low-precision
containers into dense K/V, then run PyTorch math-backend SDPA in FP32 with the
bottom-right/top-left band mask and GQA expansion, and round O to the io dtype.
A live row that is fully masked (including seq_len_kv == 0) is zero.
With --sinks the golden appends each head's sink as one extra logit column (not
scaled), takes the softmax and the log-sum-exp over all columns, and drops the
sink column from P before P @ V; a live row with no visible key then gives O = 0
and LSE = sink (the hipDNN sink_token contract).

Output: {base}.json, {base}.tensor{uid}.bin, {base}.meta.json,
{base}.tensors.dvc (md5/size filled in) and {base}.support.json (empty claims,
written only when absent so recorded claims survive regeneration).

Usage (decode, dense Q):
    python generate_sdpa_fwd_paged_golden.py \\
        --base-filename quick/SdpaFwd/bshd/bf16/hd128_causal_paged/Decode/Decode \\
        --num-heads-q 8 --num-heads-kv 2 --head-dim 128 --seq-q 1 \\
        --block-size 16 --max-seq-len-kv 128 --seq-lens-kv 37 0 128 16 \\
        --causal bottom_right --seed 42
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import torch

from generate_sdpa_fwd_golden import (
    DTYPE_MAP,
    SDPA_SEQ_AXIS,
    UID_K,
    UID_LSE,
    UID_O,
    UID_Q,
    UID_RAGGED_OFFSET_QO,
    UID_SEQ_LEN_KV,
    UID_SEQ_LEN_Q,
    UID_V,
    build_graph_json,
    build_mask,
    compute_forward,
    compute_lse,
    compute_strides,
    pack_ragged,
    ragged_offsets,
    save_tensor_bin,
    to_physical_layout,
)

# Bump when generator logic changes in a way that affects emitted output.
# 1.0.0 - Initial paged-KV generator (dense or ragged Q, gather-then-dense golden)
# 1.1.0 - Added --sinks / --sink-dtype (attention sink_token, uid 14); bundles
#         without sinks are unchanged
GENERATOR_VERSION = "1.1.0"

UID_PAGE_TABLE_K = 12
UID_PAGE_TABLE_V = 13
UID_SINK = 14

# Sink dtypes: the io dtype (vLLM keeps layer.sinks in the model dtype; what the
# rocKE-backed engines accept) or FLOAT (cuDNN's sample dtype).
SINK_DTYPES = ("io", "float")

# Fill for every container slot no live token maps to (spare blocks and the tail
# of a partial last block). Large but finite: a kernel that reads one of them
# produces a visibly wrong O instead of a NaN that a comparator might skip.
SPARE_FILL = 100.0
# Fill for the unused half of each D row with --interleaved-kv.
INTERLEAVE_FILL = 0.0

DVC_REMOTE = "golden-data"
SUPPORT_SIDECAR = {"claims": {}, "version": 1}


def element_space(dims, strides):
    """Elements a strided tensor spans: 1 + sum((dim - 1) * stride)."""
    return 1 + sum((d - 1) * s for d, s in zip(dims, strides))


def kv_container_strides(dims, row_pitch):
    """Strides of a logical [num_blocks, Hkv, block_size, D] container stored as
    vLLM's [num_blocks, block_size, Hkv, D], with each D row row_pitch apart."""
    _, h_kv, block_size, _ = dims
    return [block_size * h_kv * row_pitch, row_pitch, h_kv * row_pitch, 1]


def materialize_strided(logical, strides, filler):
    """Flat buffer of element_space(dims, strides) elements, filled with filler,
    holding logical at the given strides (the bytes a strided .bin stores)."""
    dims = list(logical.shape)
    buf = torch.full((element_space(dims, strides),), filler, dtype=logical.dtype)
    buf.as_strided(dims, strides).copy_(logical)
    return buf


def blocks_needed(seq_len_kv, block_size):
    return (seq_len_kv + block_size - 1) // block_size


def build_page_table(seq_lens_kv, block_size, max_blocks, num_blocks, generator):
    """[B, 1, M, 1] int32 page table over a shuffled set of distinct block ids.

    Used entries take the first sum(blocks_needed) ids of a random permutation of
    range(num_blocks), batch-major; unused entries are -1 (never read).
    """
    perm = torch.randperm(num_blocks, generator=generator).tolist()
    table = torch.full((len(seq_lens_kv), 1, max_blocks, 1), -1, dtype=torch.int32)
    next_id = 0
    for b, skv in enumerate(seq_lens_kv):
        for j in range(blocks_needed(skv, block_size)):
            table[b, 0, j, 0] = perm[next_id]
            next_id += 1
    return table


def scatter_pages(container, source, page_table, seq_lens_kv):
    """Write source[b, :, t, :] (t < seq_lens_kv[b]) to its paged container slot."""
    block_size = container.shape[2]
    for b, skv in enumerate(seq_lens_kv):
        for j in range(blocks_needed(skv, block_size)):
            block = int(page_table[b, 0, j, 0])
            t0 = j * block_size
            t1 = min(skv, t0 + block_size)
            container[block, :, : t1 - t0, :] = source[b, :, t0:t1, :]


def gather_pages(container, page_table_row, seq_len_kv):
    """Dense [H, seq_len_kv, D] K or V of one batch, read through its page table."""
    num_blocks, h_kv, block_size, dim = container.shape
    pages = []
    for j in range(blocks_needed(seq_len_kv, block_size)):
        block = int(page_table_row[j])
        if block < 0 or block >= num_blocks:
            raise ValueError(
                f"page-table entry {j} = {block} outside [0, {num_blocks})"
            )
        pages.append(container[block])
    if not pages:
        return container.new_zeros((h_kv, 0, dim))
    return torch.cat(pages, dim=1)[:, :seq_len_kv, :]


def compute_paged_forward(
    Q,
    k_container,
    v_container,
    page_table_k,
    page_table_v,
    seq_lens_q,
    seq_lens_kv,
    scale,
    causal,
    window_left,
    window_right,
    o_torch_dtype,
    want_lse,
    sinks=None,
):
    """Gather-then-dense golden. Q is logical [B, Hq, Sq, D] in the io dtype.

    Returns O [B, Hq, Sq, Dv] (io dtype; zero rows where no key is visible or
    the row is past seq_lens_q) and, if requested, LSE [B, Hq, Sq, 1] FP32.
    sinks is an optional FP32 [Hq] of per-head sink logits (see compute_sink_forward).
    """
    B, H_q, S_q, _ = Q.shape
    H_kv = k_container.shape[1]
    D_v = v_container.shape[3]
    O = torch.zeros((B, H_q, S_q, D_v), dtype=o_torch_dtype)
    lse = torch.zeros((B, H_q, S_q, 1), dtype=torch.float32) if want_lse else None
    for b in range(B):
        sq = seq_lens_q[b]
        skv = seq_lens_kv[b]
        if sq == 0:
            continue
        if skv == 0:
            # No key: O stays zero; with a sink the LSE is the sink itself.
            if want_lse and sinks is not None:
                lse[b, :, :sq, 0] = sinks.view(H_q, 1)
            continue
        k_b = gather_pages(k_container, page_table_k[b, 0, :, 0], skv)
        v_b = gather_pages(v_container, page_table_v[b, 0, :, 0], skv)
        # Like-for-like: low-precision inputs, FP32 math.
        q_f = Q[b : b + 1, :, :sq, :].float()
        k_f = k_b.unsqueeze(0).float()
        v_f = v_b.unsqueeze(0).float()
        keep = build_mask(sq, skv, causal, window_left, window_right)
        if sinks is not None:
            o_b, lse_b = compute_sink_forward(q_f, k_f, v_f, scale, keep, sinks)
            O[b : b + 1, :, :sq, :] = o_b.to(o_torch_dtype)
            if want_lse:
                lse[b : b + 1, :, :sq, :] = lse_b
            continue
        o_b = compute_forward(q_f, k_f, v_f, scale, H_q, H_kv, keep)
        if keep is not None:
            # Fully masked live rows are zero (contract), whatever softmax made of them.
            o_b[:, :, ~keep.any(dim=1), :] = 0.0
        O[b : b + 1, :, :sq, :] = o_b.to(o_torch_dtype)
        if want_lse:
            lse[b : b + 1, :, :sq, :] = compute_lse(q_f, k_f, scale, H_q, H_kv, keep)
    return O, lse


def compute_sink_forward(q_f, k_f, v_f, scale, keep_mask, sinks):
    """FP32 attention with one sink logit per query head (gpt-oss / cuDNN sink_token).

    q_f [1, Hq, Sq, D], k_f/v_f [1, Hkv, Skv, D(v)], sinks [Hq] (raw logits, not
    scaled). The sink is appended as an extra logit column; softmax and logsumexp
    run over all columns, and the sink column is dropped from P before P @ V.
    Returns (O [1, Hq, Sq, Dv], LSE [1, Hq, Sq, 1]).
    """
    H_q = q_f.shape[1]
    H_kv = k_f.shape[1]
    if H_q != H_kv:
        k_f = k_f.repeat_interleave(H_q // H_kv, dim=1)
        v_f = v_f.repeat_interleave(H_q // H_kv, dim=1)
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if keep_mask is not None:
        scores = scores.masked_fill(~keep_mask, float("-inf"))
    sink_col = sinks.view(1, H_q, 1, 1).expand(1, H_q, scores.shape[2], 1)
    logits = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(logits, dim=-1)[..., :-1]
    return torch.matmul(probs, v_f), torch.logsumexp(logits, dim=-1, keepdim=True)


def fully_masked_live_rows(seq_lens_q, seq_lens_kv, causal, window_left, window_right):
    """(batch, row) pairs of live query rows that see no key."""
    rows = []
    for b, (sq, skv) in enumerate(zip(seq_lens_q, seq_lens_kv)):
        if sq == 0:
            continue
        if skv == 0:
            rows += [(b, r) for r in range(sq)]
            continue
        keep = build_mask(sq, skv, causal, window_left, window_right)
        if keep is not None:
            rows += [(b, int(r)) for r in torch.nonzero(~keep.any(dim=1)).flatten()]
    return rows


def build_paged_graph_json(
    q_dims,
    k_dims,
    v_dims,
    o_dims,
    k_strides,
    v_strides,
    scale,
    dtype_str,
    stats,
    causal,
    window_left,
    window_right,
    ragged_q,
    max_seq_len_kv,
    max_blocks,
    shared_page_table,
    sink_dtype_str=None,
):
    """Paged SDPA graph in the build_graph_json schema. sink_dtype_str (a json
    dtype name) adds the sink_token tensor (uid 14)."""
    graph = build_graph_json(
        q_dims,
        k_dims,
        v_dims,
        o_dims,
        scale,
        dtype_str=dtype_str,
        stats=stats,
        causal=causal,
        window_left=window_left,
        window_right=window_right,
        layout="bshd",
    )
    B, H_q, S_q, _ = q_dims
    by_uid = {t["uid"]: t for t in graph["tensors"]}
    by_uid[UID_K]["strides"] = k_strides
    by_uid[UID_V]["strides"] = v_strides
    if ragged_q:
        for uid in (UID_Q, UID_O):
            by_uid[uid]["ragged_offset_tensor_uid"] = UID_RAGGED_OFFSET_QO
            by_uid[uid]["ragged_offset_multiplier"] = by_uid[uid]["strides"][
                SDPA_SEQ_AXIS
            ]
        if stats:
            lse = by_uid[UID_LSE]
            lse["strides"] = compute_strides(lse["dims"], "bshd")
            lse["ragged_offset_tensor_uid"] = UID_RAGGED_OFFSET_QO
            lse["ragged_offset_multiplier"] = lse["strides"][SDPA_SEQ_AXIS]

    def aux(uid, name, dims, strides):
        return {
            "uid": uid,
            "name": name,
            "dims": dims,
            "strides": strides,
            "data_type": "int32",
            "virtual": False,
        }

    if ragged_q:
        graph["tensors"].append(
            aux(UID_SEQ_LEN_Q, "SeqLenQ", [B, 1, 1, 1], [1, 1, 1, 1])
        )
    graph["tensors"].append(aux(UID_SEQ_LEN_KV, "SeqLenKv", [B, 1, 1, 1], [1, 1, 1, 1]))
    if ragged_q:
        graph["tensors"].append(
            aux(UID_RAGGED_OFFSET_QO, "RaggedOffsetQ", [B + 1, 1, 1, 1], [1, 1, 1, 1])
        )
    table_dims = [B, 1, max_blocks, 1]
    table_strides = [max_blocks, max_blocks, 1, 1]
    graph["tensors"].append(
        aux(
            UID_PAGE_TABLE_K,
            "PageTable" if shared_page_table else "PageTableK",
            table_dims,
            table_strides,
        )
    )
    if not shared_page_table:
        graph["tensors"].append(
            aux(UID_PAGE_TABLE_V, "PageTableV", table_dims, table_strides)
        )
    if sink_dtype_str is not None:
        graph["tensors"].append(
            {
                "uid": UID_SINK,
                "name": "SinkToken",
                "dims": [1, H_q, 1, 1],
                "strides": [H_q, 1, 1, 1],
                "data_type": sink_dtype_str,
                "virtual": False,
            }
        )

    node = graph["nodes"][0]
    inputs = node["inputs"]
    inputs["seq_len_q_tensor_uid"] = UID_SEQ_LEN_Q if ragged_q else None
    inputs["seq_len_kv_tensor_uid"] = UID_SEQ_LEN_KV
    inputs["page_table_k_tensor_uid"] = UID_PAGE_TABLE_K
    inputs["page_table_v_tensor_uid"] = (
        UID_PAGE_TABLE_K if shared_page_table else UID_PAGE_TABLE_V
    )
    inputs["sink_token_tensor_uid"] = UID_SINK if sink_dtype_str is not None else None
    attributes = node["attributes"]
    attributes["padding_mask"] = False
    attributes["max_seq_len_kv"] = max_seq_len_kv
    return graph


def _sha256_of(path):
    """SHA-256 of a file; for this script it names the exact generator bytes, so
    any edit here makes already checked-in .meta.json values stale (intended)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_paged_meta_json(config, pytorch_version):
    rocm_ver = ""
    if "+rocm" in pytorch_version:
        rocm_ver = pytorch_version.split("+rocm")[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        "format_version": 1,
        "generator": "generate_sdpa_fwd_paged_golden.py",
        "generator_sha256": _sha256_of(os.path.abspath(__file__)),
        "helpers_sha256": _sha256_of(
            os.path.join(script_dir, "generate_sdpa_fwd_golden.py")
        ),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "reference_source": f"PyTorch {pytorch_version}",
        "reference_backend": "pytorch_math_backend (gather-then-dense)",
        "rocm_version": rocm_ver,
        "generator_version": GENERATOR_VERSION,
        "generation_precision": (
            f"like-for-like: {config['dtype'].upper()} inputs, FP32 intermediates, "
            f"{config['dtype'].upper()} output"
        ),
        "direction": "forward",
        "seed": config["seed"],
        "input_range": [config["min_val"], config["max_val"]],
        "deterministic": True,
        "config": config["meta_config"],
    }


def write_tensors_dvc(base_filename, bin_paths):
    """{base}.tensors.dvc listing every .bin with md5/size filled in (the format
    `dvc commit` writes), routed to the golden-data remote."""
    lines = ["outs:"]
    for path in sorted(bin_paths, key=os.path.basename):
        with open(path, "rb") as f:
            data = f.read()
        lines += [
            f"- path: {os.path.basename(path)}",
            f"  remote: {DVC_REMOTE}",
            f"  md5: {hashlib.md5(data).hexdigest()}",
            f"  size: {len(data)}",
            "  hash: md5",
        ]
    dvc_path = f"{base_filename}.tensors.dvc"
    with open(dvc_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return dvc_path


def write_support_sidecar(base_filename):
    """Canonical empty support sidecar; an existing one (with recorded claims)
    is kept."""
    path = f"{base_filename}.support.json"
    if os.path.exists(path):
        return path, False
    with open(path, "w", newline="\n") as f:
        f.write(json.dumps(SUPPORT_SIDECAR, indent=2, sort_keys=True) + "\n")
    return path, True


def _validate_args(
    dtype,
    H_q,
    H_kv,
    D_qk,
    D_v,
    S_q,
    block_size,
    max_seq_len_kv,
    seq_lens_kv,
    ragged_q,
    seq_lens_q,
    spare_blocks,
    min_val,
    max_val,
):
    errors = []
    if dtype not in DTYPE_MAP:
        errors.append(f"--dtype must be one of {list(DTYPE_MAP)} (got '{dtype}')")
    for name, value in [
        ("--num-heads-q", H_q),
        ("--num-heads-kv", H_kv),
        ("--head-dim", D_qk),
        ("--head-dim-v", D_v),
        ("--block-size", block_size),
        ("--max-seq-len-kv", max_seq_len_kv),
    ]:
        if value <= 0:
            errors.append(f"{name} must be positive (got {value})")
    if H_kv > 0 and H_q % H_kv != 0:
        errors.append(
            f"--num-heads-q ({H_q}) must be divisible by --num-heads-kv ({H_kv})"
        )
    if spare_blocks < 0:
        errors.append(f"--spare-blocks must be >= 0 (got {spare_blocks})")
    if min_val >= max_val:
        errors.append(f"--min ({min_val}) must be less than --max ({max_val})")
    if not seq_lens_kv:
        errors.append("--seq-lens-kv needs one entry per batch")
    for i, s in enumerate(seq_lens_kv or []):
        if s < 0 or s > max_seq_len_kv:
            errors.append(
                f"--seq-lens-kv[{i}]={s} must be in [0, max_seq_len_kv={max_seq_len_kv}]"
            )
    if ragged_q:
        if S_q is not None:
            errors.append(
                "--seq-q is for dense Q; ragged Q takes Sq = max(--seq-lens-q)"
            )
        if seq_lens_q is None:
            errors.append("--ragged-q requires --seq-lens-q")
        else:
            if len(seq_lens_q) != len(seq_lens_kv or []):
                errors.append(
                    f"--seq-lens-q has {len(seq_lens_q)} entries; --seq-lens-kv has "
                    f"{len(seq_lens_kv or [])} (one per batch each)"
                )
            if any(s < 0 for s in seq_lens_q):
                errors.append("--seq-lens-q entries must be >= 0")
            if sum(seq_lens_q) == 0:
                errors.append("--seq-lens-q must hold at least one token in total")
    else:
        if seq_lens_q is not None:
            errors.append("--seq-lens-q requires --ragged-q (dense Q uses --seq-q)")
        if S_q is None or S_q <= 0:
            errors.append("dense Q needs a positive --seq-q")
    return errors


def generate_paged_bundle(
    base_filename,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_size,
    max_seq_len_kv,
    seq_lens_kv,
    head_dim_v=None,
    seq_q=None,
    ragged_q=False,
    seq_lens_q=None,
    dtype="bf16",
    causal="bottom_right",
    window_left=-1,
    window_right=-1,
    stats=False,
    shared_page_table=False,
    interleaved_kv=False,
    spare_blocks=2,
    seed=42,
    min_val=-1.0,
    max_val=1.0,
    attn_scale=None,
    sinks=False,
    sink_dtype="io",
):
    H_q, H_kv, D_qk = num_heads_q, num_heads_kv, head_dim
    D_v = head_dim if head_dim_v is None else head_dim_v
    errors = _validate_args(
        dtype,
        H_q,
        H_kv,
        D_qk,
        D_v,
        seq_q,
        block_size,
        max_seq_len_kv,
        seq_lens_kv,
        ragged_q,
        seq_lens_q,
        spare_blocks,
        min_val,
        max_val,
    )
    B = len(seq_lens_kv or [])
    if not errors:
        S_q = max(max(seq_lens_q), 1) if ragged_q else seq_q
        live_q = list(seq_lens_q) if ragged_q else [S_q] * B
        if sink_dtype not in SINK_DTYPES:
            errors.append(f"--sink-dtype must be one of {list(SINK_DTYPES)}")
        # With sinks a keyless row has LSE = sink (finite), so only sink-free bundles
        # must avoid fully masked live rows.
        if stats and not sinks:
            masked = fully_masked_live_rows(
                live_q, seq_lens_kv, causal, window_left, window_right
            )
            if masked:
                errors.append(
                    f"--stats with fully masked live rows {masked[:4]}"
                    f"{' ...' if len(masked) > 4 else ''}: their LSE would be -inf, "
                    "which golden bundles may not hold"
                )
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if attn_scale is None:
        # An unset attn_scale_value means no scaling in hipDNN, as in cuDNN.
        attn_scale = 1.0

    info = DTYPE_MAP[dtype]
    io_torch = info["torch"]
    io_json = info["json"]

    max_blocks = blocks_needed(max_seq_len_kv, block_size)
    used_blocks = sum(blocks_needed(s, block_size) for s in seq_lens_kv)
    num_blocks = used_blocks + spare_blocks
    if num_blocks == 0:
        # Every seq_len_kv is 0: keep one (spare) block so the containers exist.
        num_blocks = 1

    q_dims = [B, H_q, S_q, D_qk]
    o_dims = [B, H_q, S_q, D_v]
    k_dims = [num_blocks, H_kv, block_size, D_qk]
    v_dims = [num_blocks, H_kv, block_size, D_v]
    k_strides = kv_container_strides(k_dims, 2 * D_qk if interleaved_kv else D_qk)
    v_strides = kv_container_strides(v_dims, 2 * D_v if interleaved_kv else D_v)

    os.makedirs(os.path.dirname(base_filename) or ".", exist_ok=True)

    print(f"Generating paged forward bundle: {base_filename}")
    print(f"  Q: {q_dims} ({'ragged' if ragged_q else 'dense'}), O: {o_dims}")
    print(f"  K: {k_dims} strides {k_strides}, V: {v_dims} strides {v_strides}")
    print(
        f"  block_size={block_size}, max_seq_len_kv={max_seq_len_kv}, "
        f"max_blocks_per_seq={max_blocks}, num_blocks={num_blocks} "
        f"({used_blocks} used + {num_blocks - used_blocks} spare)"
    )
    print(f"  seq_lens_q={live_q}, seq_lens_kv={list(seq_lens_kv)}")
    print(
        f"  dtype={dtype}, causal={causal}, window=({window_left},{window_right}), "
        f"stats={stats}, shared_page_table={shared_page_table}, "
        f"interleaved_kv={interleaved_kv}"
    )
    print(f"  Scale: {attn_scale:.10f}, Seed: {seed}")
    if sinks:
        print(f"  sinks: N(0, 1) per query head, dtype {sink_dtype}")

    rng = torch.Generator().manual_seed(seed)
    Q = torch.empty(q_dims, dtype=io_torch).uniform_(min_val, max_val, generator=rng)
    k_source = torch.empty([B, H_kv, max_seq_len_kv, D_qk], dtype=io_torch).uniform_(
        min_val, max_val, generator=rng
    )
    v_source = torch.empty([B, H_kv, max_seq_len_kv, D_v], dtype=io_torch).uniform_(
        min_val, max_val, generator=rng
    )
    page_table_k = build_page_table(
        seq_lens_kv, block_size, max_blocks, num_blocks, rng
    )
    page_table_v = (
        page_table_k
        if shared_page_table
        else build_page_table(seq_lens_kv, block_size, max_blocks, num_blocks, rng)
    )
    # Drawn last so bundles without sinks keep their exact data.
    sink_store = None
    sink_values = None
    sink_json = None
    if sinks:
        sink_torch = io_torch if sink_dtype == "io" else torch.float32
        sink_json = io_json if sink_dtype == "io" else "float"
        sink_store = (
            torch.randn([H_q], dtype=torch.float32, generator=rng)
            .to(sink_torch)
            .reshape(1, H_q, 1, 1)
        )
        # The golden uses exactly the stored (possibly rounded) sink.
        sink_values = sink_store.float().reshape(H_q)

    k_logical = torch.full(k_dims, SPARE_FILL, dtype=io_torch)
    v_logical = torch.full(v_dims, SPARE_FILL, dtype=io_torch)
    scatter_pages(k_logical, k_source, page_table_k, seq_lens_kv)
    scatter_pages(v_logical, v_source, page_table_v, seq_lens_kv)
    k_buffer = materialize_strided(k_logical, k_strides, INTERLEAVE_FILL)
    v_buffer = materialize_strided(v_logical, v_strides, INTERLEAVE_FILL)

    # Golden reads K/V back through the stored strides, as a kernel would.
    O, lse = compute_paged_forward(
        Q,
        k_buffer.as_strided(k_dims, k_strides),
        v_buffer.as_strided(v_dims, v_strides),
        page_table_k,
        page_table_v,
        live_q,
        seq_lens_kv,
        attn_scale,
        causal,
        window_left,
        window_right,
        io_torch,
        want_lse=stats,
        sinks=sink_values,
    )

    assert torch.isfinite(O.float()).all(), "non-finite value in O"
    if lse is not None:
        assert torch.isfinite(lse).all(), "non-finite value in LSE"
    for name, buf in (("K", k_buffer), ("V", v_buffer)):
        assert torch.isfinite(buf.float()).all(), f"non-finite value in {name}"

    if ragged_q:
        q_store = pack_ragged(Q, live_q)
        o_store = pack_ragged(O, live_q)
        lse_store = pack_ragged(lse, live_q) if lse is not None else None
    else:
        q_store = to_physical_layout(Q, "bshd")
        o_store = to_physical_layout(O, "bshd")
        lse_store = lse

    def int32_bshape(values):
        return torch.tensor(values, dtype=torch.int32).reshape(len(values), 1, 1, 1)

    tensor_list = [
        ("Q", q_store, UID_Q),
        ("K", k_buffer, UID_K),
        ("V", v_buffer, UID_V),
        ("O", o_store, UID_O),
    ]
    if lse_store is not None:
        tensor_list.append(("LSE", lse_store, UID_LSE))
    if ragged_q:
        tensor_list.append(("SeqLenQ", int32_bshape(live_q), UID_SEQ_LEN_Q))
    tensor_list.append(("SeqLenKv", int32_bshape(list(seq_lens_kv)), UID_SEQ_LEN_KV))
    if ragged_q:
        tensor_list.append(
            ("RaggedOffsetQ", ragged_offsets(live_q), UID_RAGGED_OFFSET_QO)
        )
    tensor_list.append(
        (
            "PageTable" if shared_page_table else "PageTableK",
            page_table_k,
            UID_PAGE_TABLE_K,
        )
    )
    if not shared_page_table:
        tensor_list.append(("PageTableV", page_table_v, UID_PAGE_TABLE_V))
    if sink_store is not None:
        tensor_list.append(("SinkToken", sink_store, UID_SINK))

    # Drop .bin files of an earlier run whose tensor set differed (e.g. a former
    # separate page_table_v), so the bundle and its .dvc list only current tensors.
    current = {f"{base_filename}.tensor{uid}.bin" for _, _, uid in tensor_list}
    for stale in glob.glob(f"{glob.escape(base_filename)}.tensor*.bin"):
        if os.path.normpath(stale) not in {os.path.normpath(p) for p in current}:
            os.remove(stale)

    bin_paths = []
    for name, tensor, uid in tensor_list:
        bin_path = f"{base_filename}.tensor{uid}.bin"
        save_tensor_bin(tensor, bin_path)
        bin_paths.append(bin_path)
        size_kb = os.path.getsize(bin_path) / 1024
        print(
            f"  {name} (uid={uid}): {list(tensor.shape)} {tensor.dtype} -> {size_kb:.1f} KB"
        )

    graph_json = build_paged_graph_json(
        q_dims,
        k_dims,
        v_dims,
        o_dims,
        k_strides,
        v_strides,
        attn_scale,
        io_json,
        stats,
        causal,
        window_left,
        window_right,
        ragged_q,
        max_seq_len_kv,
        max_blocks,
        shared_page_table,
        sink_dtype_str=sink_json,
    )
    json_path = f"{base_filename}.json"
    with open(json_path, "w", newline="\n") as f:
        json.dump(graph_json, f, indent=4)
        f.write("\n")
    print(f"  Graph JSON: {json_path}")

    def table_rows(table):
        return [table[b, 0, :, 0].tolist() for b in range(B)]

    meta_config = {
        "batch": B,
        "num_heads_q": H_q,
        "num_heads_kv": H_kv,
        "seq_q": S_q,
        "head_dim_qk": D_qk,
        "head_dim_v": D_v,
        "dtype": dtype,
        "causal": causal,
        "window_left": window_left,
        "window_right": window_right,
        "stats": stats,
        "scale": attn_scale,
        "gqa_ratio": H_q // H_kv,
        "layout": "bshd",
        "ragged_q": ragged_q,
        "seq_lens_q": list(seq_lens_q) if ragged_q else None,
        "seq_lens_kv": list(seq_lens_kv),
        "max_seq_len_kv": max_seq_len_kv,
        "block_size": block_size,
        "max_blocks_per_seq": max_blocks,
        "num_blocks": num_blocks,
        "used_blocks": used_blocks,
        "spare_blocks": num_blocks - used_blocks,
        "kv_container_layout": "[num_blocks, block_size, H_kv, D] (vLLM)",
        "interleaved_kv": interleaved_kv,
        "k_strides": k_strides,
        "v_strides": v_strides,
        "spare_fill": SPARE_FILL,
        "interleave_fill": INTERLEAVE_FILL if interleaved_kv else None,
        "shared_page_table": shared_page_table,
        "page_table_k": table_rows(page_table_k),
        "page_table_v": table_rows(page_table_v),
        "sinks": sink_values.tolist() if sink_values is not None else None,
        "sink_dtype": sink_dtype if sinks else None,
    }
    config = {
        "dtype": dtype,
        "seed": seed,
        "min_val": min_val,
        "max_val": max_val,
        "meta_config": meta_config,
    }
    meta_json = build_paged_meta_json(config, torch.__version__)
    meta_path = f"{base_filename}.meta.json"
    with open(meta_path, "w", newline="\n") as f:
        json.dump(meta_json, f, indent=4)
        f.write("\n")
    print(f"  Meta JSON: {meta_path}")

    dvc_path = write_tensors_dvc(base_filename, bin_paths)
    print(f"  DVC pointer: {dvc_path}")
    support_path, created = write_support_sidecar(base_filename)
    print(f"  Support sidecar: {support_path} ({'created' if created else 'kept'})")

    bundle_dir = os.path.dirname(base_filename) or "."
    total = sum(
        os.path.getsize(os.path.join(bundle_dir, f))
        for f in os.listdir(bundle_dir)
        if os.path.isfile(os.path.join(bundle_dir, f))
    )
    print(f"  Bundle size: {total / 1024:.1f} KB")
    if total > 1024 * 1024:
        print(
            f"WARNING: bundle is {total} bytes, above the 1 MiB verify_golden_bundles.py "
            "warning threshold",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate paged-KV SDPA forward golden reference bundles"
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Path prefix for output files (no extension)",
    )
    parser.add_argument("--num-heads-q", type=int, required=True)
    parser.add_argument("--num-heads-kv", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True, help="D_qk")
    parser.add_argument(
        "--head-dim-v", type=int, default=None, help="D_v (default: --head-dim)"
    )
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--max-seq-len-kv", type=int, required=True)
    parser.add_argument(
        "--seq-lens-kv",
        nargs="+",
        type=int,
        required=True,
        help="Per-batch KV lengths (sets B); each in [0, max_seq_len_kv]",
    )
    parser.add_argument(
        "--seq-q", type=int, default=None, help="Dense Q sequence length"
    )
    parser.add_argument(
        "--ragged-q",
        action="store_true",
        help="Packed RFC-0014 Q/O (no gaps) with ragged offset uid 10 and seq_len_q uid 5",
    )
    parser.add_argument(
        "--seq-lens-q",
        nargs="+",
        type=int,
        default=None,
        help="Per-batch Q lengths for --ragged-q (0 allowed); Sq = max of them",
    )
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument(
        "--causal",
        default="bottom_right",
        choices=["none", "top_left", "bottom_right"],
        help="Diagonal alignment of the causal band (default: bottom_right)",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="Sliding window left bound, -1 = open",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=-1,
        help="Sliding window right bound, -1 = open",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Emit LSE (uid 4), float [B, Hq, Sq, 1]"
    )
    parser.add_argument(
        "--shared-page-table",
        action="store_true",
        help="Bind one page table (uid 12) to both page_table_k and page_table_v",
    )
    parser.add_argument(
        "--interleaved-kv",
        action="store_true",
        help="Store K/V with a 2*D row pitch (values in [0:D], filler in [D:2D])",
    )
    parser.add_argument(
        "--spare-blocks",
        type=int,
        default=2,
        help="Container blocks no page table references, filled with SPARE_FILL (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min", type=float, default=-1.0, dest="min_val")
    parser.add_argument("--max", type=float, default=1.0, dest="max_val")
    parser.add_argument(
        "--attn-scale",
        type=float,
        default=None,
        help="Attention scale (default: 1.0, no scaling, as for an unset attn_scale_value)",
    )
    parser.add_argument(
        "--sinks",
        action="store_true",
        help="Add an attention sink_token (uid 14) [1, Hq, 1, 1], N(0, 1) per head",
    )
    parser.add_argument(
        "--sink-dtype",
        default="io",
        choices=list(SINK_DTYPES),
        help="Sink data type: the io dtype (default) or float",
    )
    args = parser.parse_args()

    generate_paged_bundle(
        base_filename=args.base_filename,
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        head_dim=args.head_dim,
        head_dim_v=args.head_dim_v,
        block_size=args.block_size,
        max_seq_len_kv=args.max_seq_len_kv,
        seq_lens_kv=args.seq_lens_kv,
        seq_q=args.seq_q,
        ragged_q=args.ragged_q,
        seq_lens_q=args.seq_lens_q,
        dtype=args.dtype,
        causal=args.causal,
        window_left=args.window_left,
        window_right=args.window_right,
        stats=args.stats,
        shared_page_table=args.shared_page_table,
        interleaved_kv=args.interleaved_kv,
        spare_blocks=args.spare_blocks,
        seed=args.seed,
        min_val=args.min_val,
        max_val=args.max_val,
        attn_scale=args.attn_scale,
        sinks=args.sinks,
        sink_dtype=args.sink_dtype,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
