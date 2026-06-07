# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Definitive correctness + perf check over EVERY shape in shapes.json.

Runs the production dispatcher (run_unified_attention_torch backend="auto") for
each canonical shape, checks correctness vs the fp32 paged reference, and (for
perf/decode shapes) times CK vs a CORRECT Torch SDPA baseline:
  - prefill (square): is_causal=True
  - decode  (q_len=1): is_causal=False  (the lone query attends to all keys;
    is_causal=True would use upper-left alignment -> attends only key[0]).
The Torch baseline is itself verified against the reference.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ck_dsl.instances import UnifiedAttentionProblem
from ck_dsl.instances.common import attention_unified as au
from ck_dsl.runtime import synchronize_and_release, time_launches
from parity_unified_attention import (
    attention_tflops, compare, load_shapes, make_inputs, ref_paged_attn,
)


def torch_baseline(s, data, is_causal):
    nrep = s.heads // s.kv_heads
    klen = data["kv_lens_list"][0]
    ks, vs = [], []
    for bi in range(s.batch):
        bt = data["block_tables"][bi]
        ks.append(data["key_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen])
        vs.append(data["value_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen])
    kh = torch.stack(ks, 0).transpose(1, 2).repeat_interleave(nrep, 1).contiguous()
    vh = torch.stack(vs, 0).transpose(1, 2).repeat_interleave(nrep, 1).contiguous()
    qh = data["query"].view(s.batch, s.seqlen_q, s.heads, s.head_size).transpose(1, 2).contiguous()
    sink = torch.zeros(1, device="cuda")

    def once():
        oo = F.scaled_dot_product_attention(qh, kh, vh, is_causal=is_causal, scale=data["scale"])
        sink.copy_(oo.sum())
    st = int(torch.cuda.current_stream().cuda_stream)
    once(); torch.cuda.synchronize()
    return time_launches(once, warmup=10, iters=40, stream=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["correctness", "perf", "decode"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    print(f"device,{torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
    shapes = [s for s in load_shapes() if s.group in args.groups]
    stream = int(torch.cuda.current_stream().cuda_stream)
    n_pass = n_fail = n_win = 0
    print(f"{'shape':32s} {'dt':4s} {'CK us':>9} {'torch us':>9} {'CK/torch':>9} {'maxabs':>9}  res")
    for s in shapes:
        data = make_inputs(s)
        ref = ref_paged_attn(data["query"], data["key_cache"], data["value_cache"],
                             data["query_lens"], data["kv_lens_list"],
                             data["block_tables"], data["scale"]).float()
        decode = s.seqlen_q == 1
        problem = UnifiedAttentionProblem(
            total_q=data["query"].shape[0], num_seqs=s.batch, num_query_heads=s.heads,
            num_kv_heads=s.kv_heads, head_size=s.head_size, block_size=64,
            max_seqlen_q=(1 if decode else data["max_query_len"]),
            max_seqlen_k=data["max_kv_len"], dtype=s.dtype, num_sms=120)
        out = torch.empty_like(data["query"])

        def call():
            au.run_unified_attention_torch(
                problem=problem, q=data["query"], k=data["key_cache"], v=data["value_cache"],
                out=out, cu_seqlens_q=data["cu_q"], seqused_k=data["kv_lens"],
                softmax_scale=data["scale"], block_table=data["block_tables"], softcap=0.0,
                backend="auto", stream=stream)
        call(); torch.cuda.synchronize()
        mx = compare(ref, out)["max_abs"]
        tol = 2e-2 if s.dtype == "fp16" else 4e-2
        ok = mx < tol
        n_pass += ok; n_fail += (not ok)
        ck_ms = time_launches(call, warmup=args.warmup, iters=args.iters, stream=stream)
        synchronize_and_release(stream)
        tms = torch_baseline(s, data, is_causal=(not decode))
        ratio = ck_ms / tms if tms > 0 else 0.0
        if ratio < 1.0:
            n_win += 1
        res = "PASS" if ok else "FAIL"
        win = " WIN" if ratio < 1.0 else ""
        print(f"{s.name:32s} {s.dtype:4s} {ck_ms*1000:9.1f} {tms*1000:9.1f} {ratio:9.3f} "
              f"{mx:9.1e}  {res}{win}", flush=True)
        del data, ref
        torch.cuda.empty_cache()
    print(f"\ncorrectness: {n_pass} PASS / {n_fail} FAIL ; CK beats torch on {n_win}/{len(shapes)} shapes")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
