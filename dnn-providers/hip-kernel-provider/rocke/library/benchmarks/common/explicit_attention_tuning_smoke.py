#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness/timing smoke for dispatcher-owned explicit attention specs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace


def _request(arch: str, path: str):
    from dispatch.attention import AttentionRequest

    if path == "2d":
        sq, sk = 512, 512
    else:
        sq, sk = 1, 1024
    return AttentionRequest(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=sq,
        seqlen_k=sk,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        mask_type=1,
        kv_block_size=16,
    )


def _prefix(arch: str, path: str) -> str:
    if arch == "gfx950" and path == "2d":
        return "attention_gfx950_u2d_transposed32_nw2_mw32_t4xb_llvm"
    if arch == "gfx942" and path == "2d":
        return "attention_gfx942_u2d_transposed_x8_nw2_mw32_t4xb_llvm"
    return f"attention_{arch}_u3d_splitkv_seg64_t1xb"


def _pack_paged(x, page: int):
    import torch

    batch, seqlen, heads, dim = x.shape
    pages = (seqlen + page - 1) // page
    padded = torch.zeros(
        batch, pages * page, heads, dim, dtype=x.dtype, device=x.device
    )
    padded[:, :seqlen] = x
    cache = padded.reshape(batch * pages, page, heads, dim).contiguous()
    table = torch.arange(batch * pages, dtype=torch.int32, device=x.device).reshape(
        batch, pages
    )
    return cache, table


def _reference(q, k, v):
    import torch

    batch, sq, hq, dim = q.shape
    sk = k.shape[1]
    groups = hq // k.shape[2]
    kr = k.repeat_interleave(groups, dim=2)
    vr = v.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), kr.float())
    scores *= 1.0 / math.sqrt(dim)
    qi = torch.arange(sq, device=q.device)[:, None]
    ki = torch.arange(sk, device=q.device)[None, :]
    allowed = ki <= qi + (sk - sq)
    scores.masked_fill_(~allowed[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, vr.float()).to(q.dtype)


def run_one(arch: str, path: str, *, warmup: int, iters: int) -> dict:
    import torch

    from dispatch.attention import attention_candidates
    from dispatch.attention.common import _problem
    from kernels import run_unified_attention_torch
    from rocke.runtime import synchronize_and_release, time_launches

    req = _request(arch, path)
    candidate = next(
        c for c in attention_candidates() if c.name.startswith(_prefix(arch, path))
    )
    tuned_req = replace(req, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
    spec = candidate.select_spec(tuned_req)
    problem = _problem(tuned_req)

    torch.manual_seed(7)
    dtype = torch.bfloat16
    q = torch.randn(
        req.batch,
        req.seqlen_q,
        req.nhead_q,
        req.hdim_q,
        dtype=dtype,
        device="cuda",
    )
    k = (
        torch.randn(
            req.batch,
            req.seqlen_k,
            req.nhead_k,
            req.hdim_q,
            dtype=dtype,
            device="cuda",
        )
        * 0.2
    )
    v = torch.randn_like(k) * 0.2
    k_cache, table = _pack_paged(k, req.kv_block_size)
    v_cache, _ = _pack_paged(v, req.kv_block_size)
    q_flat = q.reshape(-1, req.nhead_q, req.hdim_q).contiguous()
    out = torch.zeros_like(q_flat)
    cu = torch.arange(
        0,
        (req.batch + 1) * req.seqlen_q,
        req.seqlen_q,
        dtype=torch.int32,
        device="cuda",
    )
    used = torch.full((req.batch,), req.seqlen_k, dtype=torch.int32, device="cuda")
    stream = int(torch.cuda.current_stream().cuda_stream)

    def call():
        run_unified_attention_torch(
            problem=problem,
            q=q_flat,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            softmax_scale=1.0 / math.sqrt(req.hdim_q),
            block_table=table,
            softcap=0.0,
            backend="tiled" if path == "2d" else "3d",
            stream=stream,
            tuning_spec=spec,
        )

    call()
    torch.cuda.synchronize()
    ref = _reference(q, k, v)
    max_abs = float((out.reshape_as(ref).float() - ref.float()).abs().max().item())
    ms = time_launches(call, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    if max_abs > 0.03:
        raise AssertionError(f"{arch} {path} max_abs={max_abs} exceeds 0.03")
    return {
        "arch": arch,
        "path": path,
        "candidate": candidate.name,
        "tuning_id": spec.tuning_id,
        "kernel": spec.kernel_name(),
        "grid": candidate.grid(spec, tuned_req),
        "block": candidate.block(spec),
        "max_abs": max_abs,
        "us": ms * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("gfx942", "gfx950"), required=True)
    parser.add_argument("--path", choices=("2d", "3d", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    paths = ("2d", "3d") if args.path == "all" else (args.path,)
    for path in paths:
        print(
            json.dumps(run_one(args.arch, path, warmup=args.warmup, iters=args.iters))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
