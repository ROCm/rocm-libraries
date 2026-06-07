# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""High-precision validation of sweep-winning lever configs across seqlens.

Builds an explicit short-list of candidate configs (sweep winners + current
production defaults), then times each at warmup=20/iters=100 across multiple
sequence lengths against Torch SDPA, with a correctness gate. This confirms the
broad-pass winners generalize before any production re-wiring.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from ck_dsl import compile_kernel
from ck_dsl.instances import UnifiedAttentionProblem
from ck_dsl.instances.common.attention_unified import _attn_signature, _attn_values
from ck_dsl.instances.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from ck_dsl.runtime import KernelLauncher, LaunchConfig, synchronize_and_release, time_launches
from parity_unified_attention import Shape, attention_tflops, compare, make_inputs, ref_paged_attn

from exhaustive_sweep import _make_spec


def _lv(tile_size, num_warps, waves_per_eu, kstaging, cfvst, masklimit, q_direct,
        kv_cache_policy="all"):
    return dict(tile_size=tile_size, num_warps=num_warps, waves_per_eu=waves_per_eu,
                kstaging=kstaging, cfvst=cfvst, masklimit=masklimit, q_direct=q_direct,
                kv_cache_policy=kv_cache_policy, global_load_lds_k=False,
                q_major_grid=False, early_v_schedule=False, iglp_opt=False,
                cfv_ck_vlds=True, cfv_store_split=True)


CANDIDATES = {
    128: [
        ("best_t64ring_ml_qd", _lv(64, 4, 2, "ring", True, True, True)),
        ("t64ring_ml_noqd", _lv(64, 4, 2, "ring", True, True, False)),
        ("t64ringldsseq_noml", _lv(64, 4, 2, "ring_ldsseq", True, False, False)),
        ("shipped_t128ring_wpe2", _lv(128, 4, 2, "ring", True, False, False)),
        ("prev_t64cfvst_double", _lv(64, 4, 2, "double", True, False, False)),
    ],
    64: [
        ("best_nw2_single_ml_qd", _lv(64, 2, 2, "single", False, True, True)),
        ("nw2_single_ml_noqd", _lv(64, 2, 2, "single", False, True, False)),
        ("shipped_d64_nw4_double_cfvst", _lv(64, 4, 2, "double", True, True, True)),
    ],
}


def torch_baseline(shape, data, is_causal=True):
    nrep = shape.heads // shape.kv_heads
    bt = data["block_tables"][0]
    klen = data["kv_lens_list"][0]
    kc = data["key_cache"][bt].reshape(-1, shape.kv_heads, shape.head_size)[:klen]
    vc = data["value_cache"][bt].reshape(-1, shape.kv_heads, shape.head_size)[:klen]
    qh = data["query"].view(1, shape.seqlen_q, shape.heads, shape.head_size).transpose(1, 2)
    kh = kc.view(1, klen, shape.kv_heads, shape.head_size).transpose(1, 2).repeat_interleave(nrep, 1)
    vh = vc.view(1, klen, shape.kv_heads, shape.head_size).transpose(1, 2).repeat_interleave(nrep, 1)

    def once():
        F.scaled_dot_product_attention(qh, kh, vh, is_causal=is_causal, scale=data["scale"])

    st = int(torch.cuda.current_stream().cuda_stream)
    once(); torch.cuda.synchronize()
    return time_launches(once, warmup=20, iters=100, stream=st)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-sizes", type=int, nargs="+", default=[128, 64])
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()
    print(f"device,{torch.cuda.get_device_name(0)}", flush=True)
    sig = _attn_signature("fp16", include_bt_stride=True)
    rows = []
    for hd in args.head_sizes:
        for sl in args.seqlens:
            s = Shape(f"D{hd}_S{sl}", "fp16", sl, sl, hd, 32, 8, 1, True, "perf")
            data = make_inputs(s)
            ref = ref_paged_attn(data["query"], data["key_cache"], data["value_cache"],
                                 data["query_lens"], data["kv_lens_list"],
                                 data["block_tables"], data["scale"]).float()
            problem = UnifiedAttentionProblem(
                total_q=data["query"].shape[0], num_seqs=1, num_query_heads=32,
                num_kv_heads=8, head_size=hd, block_size=64,
                max_seqlen_q=data["max_query_len"], max_seqlen_k=data["max_kv_len"],
                dtype="fp16", num_sms=120)
            stream = int(torch.cuda.current_stream().cuda_stream)
            tms = torch_baseline(s, data)
            out = torch.empty_like(data["query"])
            print(f"\n== D{hd} S{sl} ==  torch={tms*1000:.1f}us {attention_tflops(s, tms):.1f}TF", flush=True)
            for nm, lv in CANDIDATES[hd]:
                try:
                    spec = _make_spec(UnifiedAttention2DTiledSpec, hd, 32, 8, lv)
                    art = compile_kernel(build_unified_attention_2d_tiled(spec, arch="gfx942"),
                                         arch="gfx942", capture_ir_text=False)
                    launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
                    vals = _attn_values(problem=problem, q=data["query"], k=data["key_cache"],
                                        v=data["value_cache"], out=out, cu_seqlens_q=data["cu_q"],
                                        seqused_k=data["kv_lens"], softmax_scale=data["scale"],
                                        block_table=data["block_tables"], softcap=0.0, sinks=None,
                                        bt_stride=int(data["block_tables"].stride(0)), include_bt_stride=True)
                    nqb = data["query"].shape[0] // spec.block_q + 1
                    cfg = LaunchConfig(grid=(8, int(nqb), 1), block=(64 * spec.num_warps, 1, 1), stream=stream)

                    def call():
                        launcher(vals, config=cfg)

                    call(); torch.cuda.synchronize()
                    mx = compare(ref, out)["max_abs"]
                    ms = time_launches(call, warmup=args.warmup, iters=args.iters, stream=stream)
                    synchronize_and_release(stream)
                    ratio = ms / tms
                    rows.append((f"D{hd}_S{sl}", nm, ms * 1000, ratio, mx))
                    flag = "  <-- BEATS TORCH" if ratio < 1.0 else ""
                    print(f"  {nm:34s} {ms*1000:8.1f}us {attention_tflops(s, ms):6.1f}TF "
                          f"{ratio:6.3f}x max={mx:.1e}{flag}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {nm:34s} ERROR {type(exc).__name__}:{str(exc)[:80]}", flush=True)
    print("\nJSON " + json.dumps(rows))


if __name__ == "__main__":
    main()
