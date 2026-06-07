# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exhaustive gfx942 3D split-KV DECODE micro-lever sweep vs Torch SDPA.

The 3D decode path is a segment + reduce two-kernel pipeline with a workspace
and (in production) CUDA-graph replay. Rather than re-implement that
orchestration, this sweep drives the real production entrypoint
``run_unified_attention_torch(backend="3d")`` and forces each lever combo by
monkeypatching the four 3D selectors -- all of which feed ``_tiled_3d_cache_key``
so forcing them is cache-safe (distinct binaries per combo). Timing is through
the production graph-replay path (the real decode deployment).

Levers: num_segments (the split-KV parallelism knob), tile_size_override,
waves_per_eu, use_invariant_hoist.

The 3D caches are cleared between configs so per-config workspaces/graphs don't
accumulate GPU memory across the sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path


def _force(au, segs, tile, wpe, hoist):
    au._num_segments = lambda p, _s=segs: int(_s)
    au._gfx942_3d_tile_size_override = lambda p, _t=tile: (None if not _t else int(_t))
    au._select_3d_waves_per_eu = lambda p, _w=wpe: _w
    au._enable_gfx942_3d_invariant_hoist = lambda p, _h=hoist: bool(_h)


def _clear_3d(au, torch):
    for cache in (au._3D_GRAPHS, au._3D_GRAPH_REFS, au._3D_PIPELINES,
                  au._3D_BOUND_VALUES, au._ATTN_3D_TILED_CACHE):
        cache.clear()
    torch.cuda.empty_cache()


def _grid(args):
    keys = ["num_segments", "tile_override", "waves_per_eu", "hoist"]
    vals = [args.segments, args.tiles, args.wpe, args.hoist]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-sizes", type=int, nargs="+", default=[128, 64])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 128])
    p.add_argument("--kv", type=int, nargs="+", default=[4096])
    p.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--tiles", type=int, nargs="+", default=[0, 32, 64])  # 0 == None
    p.add_argument("--wpe", type=int, nargs="+", default=[0, 2])  # 0 == None
    p.add_argument("--hoist", type=int, nargs="+", default=[0, 1])
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--tol", type=float, default=5e-3)
    p.add_argument("--out", type=str, default="~/sweeps/decode_3d.jsonl")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from ck_dsl.instances import UnifiedAttentionProblem
    from ck_dsl.instances.common import attention_unified as au
    from ck_dsl.runtime import synchronize_and_release, time_launches
    from parity_unified_attention import (
        Shape, attention_tflops, compare, make_inputs, ref_paged_attn,
    )

    # normalize 0 -> None sentinels
    args.tiles = [None if t == 0 else t for t in args.tiles]
    args.wpe = [None if w == 0 else w for w in args.wpe]

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "result":
                done.add((r["shape"], r["config"]))
    fout = out_path.open("a")

    def emit(rec):
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    print(f"device,{torch.cuda.get_device_name(0)}", flush=True)

    def torch_decode(s, data):
        # Fair baseline: each sequence reads its OWN distinct paged KV (like the
        # reference + CK), NOT a single shared block_tables[0] expanded across the
        # batch (which would be L2-cached and unfairly fast for decode).
        nrep = s.heads // s.kv_heads
        klen = data["kv_lens_list"][0]
        bts = data["block_tables"]
        ks, vs = [], []
        for bi in range(s.batch):
            bt = bts[bi]
            kc = data["key_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen]
            vc = data["value_cache"][bt].reshape(-1, s.kv_heads, s.head_size)[:klen]
            ks.append(kc)
            vs.append(vc)
        kh = torch.stack(ks, 0).transpose(1, 2).repeat_interleave(nrep, 1)
        vh = torch.stack(vs, 0).transpose(1, 2).repeat_interleave(nrep, 1)
        qh = data["query"].view(s.batch, 1, s.heads, s.head_size).transpose(1, 2)

        def once():
            # decode q_len=1 attends to ALL keys -> is_causal=False (is_causal=True
            # attends only key[0] via upper-left alignment: wrong + fake-fast).
            F.scaled_dot_product_attention(qh, kh, vh, is_causal=False, scale=data["scale"])

        st = int(torch.cuda.current_stream().cuda_stream)
        once(); torch.cuda.synchronize()
        return time_launches(once, warmup=10, iters=40, stream=st)

    for hd in args.head_sizes:
        for kv in args.kv:
            for b in args.batches:
                shname = f"D{hd}_b{b}_kv{kv}"
                s = Shape(shname, "fp16", 1, kv, hd, 32, 8, b, True, "decode")
                data = make_inputs(s)
                ref = ref_paged_attn(data["query"], data["key_cache"], data["value_cache"],
                                     data["query_lens"], data["kv_lens_list"],
                                     data["block_tables"], data["scale"]).float()
                problem = UnifiedAttentionProblem(
                    total_q=data["query"].shape[0], num_seqs=b, num_query_heads=32,
                    num_kv_heads=8, head_size=hd, block_size=64, max_seqlen_q=1,
                    max_seqlen_k=data["max_kv_len"], dtype="fp16", num_sms=120)
                stream = int(torch.cuda.current_stream().cuda_stream)
                tms = torch_decode(s, data)
                emit({"kind": "torch", "shape": shname, "us": tms * 1000,
                      "tflops": attention_tflops(s, tms)})
                print(f"\n== {shname} ==  torch={tms*1000:.1f}us", flush=True)
                out = torch.empty_like(data["query"])
                best = None
                combos = list(_grid(args))
                if args.limit > 0:
                    combos = combos[: args.limit]
                for ci, lv in enumerate(combos):
                    cfg_id = f"seg{lv['num_segments']}_t{lv['tile_override']}_w{lv['waves_per_eu']}_h{lv['hoist']}"
                    if (shname, cfg_id) in done:
                        continue
                    _clear_3d(au, torch)
                    _force(au, lv["num_segments"], lv["tile_override"], lv["waves_per_eu"], lv["hoist"])
                    t0 = time.time()
                    try:
                        def call():
                            au.run_unified_attention_torch(
                                problem=problem, q=data["query"], k=data["key_cache"],
                                v=data["value_cache"], out=out, cu_seqlens_q=data["cu_q"],
                                seqused_k=data["kv_lens"], softmax_scale=data["scale"],
                                block_table=data["block_tables"], softcap=0.0,
                                backend="3d", stream=stream)
                        call(); torch.cuda.synchronize()
                        mx = compare(ref, out)["max_abs"]
                        correct = mx < args.tol
                        if correct:
                            ms = time_launches(call, warmup=args.warmup, iters=args.iters, stream=stream)
                            synchronize_and_release(stream)
                        else:
                            ms = float("nan")
                        rec = {"kind": "result", "shape": shname, "config": cfg_id,
                               "head_size": hd, "batch": b, "kv": kv,
                               "us": (ms * 1000 if correct else None),
                               "tflops": (attention_tflops(s, ms) if correct else None),
                               "vs_torch": (ms / tms if correct and tms > 0 else None),
                               "max_abs": mx, "correct": bool(correct),
                               "levers": lv, "bench_s": round(time.time() - t0, 2)}
                        emit(rec)
                        if correct and (best is None or ms < best[0]):
                            best = (ms, cfg_id)
                    except Exception as exc:  # noqa: BLE001
                        emit({"kind": "result", "shape": shname, "config": cfg_id,
                              "head_size": hd, "batch": b, "kv": kv, "correct": False,
                              "error": f"{type(exc).__name__}:{str(exc)[:160]}", "levers": lv,
                              "bench_s": round(time.time() - t0, 2)})
                    if (ci + 1) % 20 == 0:
                        bstr = f"{best[0]*1000:.1f}us {best[1]}" if best else "n/a"
                        print(f"  [{shname}] {ci+1}/{len(combos)} best={bstr}", flush=True)
                _clear_3d(au, torch)
                if best:
                    print(f"  [{shname}] BEST {best[0]*1000:.1f}us ({best[1]})  "
                          f"vs torch {tms*1000:.1f}us = {best[0]/tms:.3f}x", flush=True)
                del data, ref
                torch.cuda.empty_cache()
    fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
