#!/usr/bin/env python3
"""Dump decode dispatch + 3D builder spec, then probe CUDA-graph capture."""

from __future__ import annotations

import os
import traceback

# gfx950 3D graph is default-on; leave the env unset so the probe
# exercises the production gate. HIPDNN_GFX950_3D_GRAPH=0 still disables.

from dispatch.attention import AttentionRequest, registered_attention_combos
from dispatch.attention.common import _problem
from kernels.common.attention_unified import (
    _enable_3d_graph_replay,
    _num_segments,
    supports_native_unified_attention_3d_tiled,
    supports_native_unified_attention_tiled,
)
from rocke.helpers.attention import use_2d_kernel


def _req(**kw):
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
        kv_block_size=16,
        num_cus=120,
    )
    base.update(kw)
    return AttentionRequest(**base)


def dump_shape(label, req):
    print(f"\n=== {label} ===")
    combos = registered_attention_combos(req)
    print("registry combos:")
    for cand, spec in combos:
        path = getattr(spec, "path", type(spec).__name__)
        print(f"  {cand.name:40} alg={cand.algorithm:16} path={path}")
        print(
            f"    spec.kernel_name={spec.kernel_name() if hasattr(spec, 'kernel_name') else spec}"
        )
    p = _problem(req)
    n2d = p.total_num_q_blocks_upper_bound * p.num_kv_heads
    want_2d = use_2d_kernel(
        head_size=p.head_size,
        sliding_window=p.sliding_window,
        all_decode=p.all_decode,
        max_seqlen_q=p.max_seqlen_q,
        max_seqlen_k=p.max_seqlen_k,
        target_num_prgms=p._effective_target_ctas,
        num_2d_prgms=n2d,
    )
    print(
        f"problem: path={p.select_path()} all_decode={p.all_decode} "
        f"num_cus={p.num_cus} target_ctas={p._effective_target_ctas} "
        f"num_2d_prgms={n2d} use_2d_kernel={want_2d}"
    )
    print(f"num_segments={_num_segments(p)} graph_replay={_enable_3d_graph_replay(p)}")
    ok3, why3 = supports_native_unified_attention_3d_tiled(p)
    ok2, why2 = supports_native_unified_attention_tiled(p)
    print(f"supports 3d_tiled={ok3} ({why3})")
    print(f"supports 2d_tiled={ok2} ({why2})")
    from builders.common.attention_spec_builder import _tiled_3d_spec_from_problem

    spec3 = _tiled_3d_spec_from_problem(p)
    print(
        f"3d builder kernel={spec3.kernel_name()} tile={spec3.tile_size} "
        f"seg={spec3.num_segments} wpe={spec3.waves_per_eu} "
        f"wide_kv={spec3.use_wide_kv_load} hoist={spec3.use_invariant_hoist}"
    )


def probe_graph():
    import torch
    from kernels import run_unified_attention_torch
    from rocke.runtime import no_fence, time_launches

    req = _req()
    p = _problem(req)
    dt = torch.bfloat16
    B, Hq, Hkv, D, Sk, page = 1, 32, 8, 128, 1024, 16
    q = torch.randn(B, Hq, D, dtype=dt, device="cuda")
    n_pages = (Sk + page - 1) // page
    k = torch.randn(B * n_pages, page, Hkv, D, dtype=dt, device="cuda") * 0.2
    v = torch.randn_like(k)
    out = torch.zeros_like(q)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
    seqused = torch.full((B,), Sk, dtype=torch.int32, device="cuda")
    table = torch.arange(B * n_pages, device="cuda", dtype=torch.int32).view(B, n_pages)
    stream = int(torch.cuda.current_stream().cuda_stream)

    def call():
        run_unified_attention_torch(
            problem=p,
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=seqused,
            softmax_scale=D**-0.5,
            block_table=table,
            softcap=0.0,
            backend="3d",
            stream=stream,
        )

    print("\n=== CUDA graph probes (Llama-3-8B Sk=1024) ===")
    for _ in range(3):
        call()
    torch.cuda.synchronize()

    print("\n=== timing (event, 50 iters) ===")
    internal_ms = time_launches(call, warmup=10, iters=50, stream=stream)
    print(f"  internal/eager call: {internal_ms * 1e3:.2f} us")
    g_outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_outer):
        call()
    torch.cuda.synchronize()
    outer_ms = time_launches(g_outer.replay, warmup=10, iters=50, stream=stream)
    print(f"  outer graph.replay:  {outer_ms * 1e3:.2f} us")

    def try_capture(title, wrap):
        print(f"\n-- {title}")
        try:
            g = torch.cuda.CUDAGraph()
            wrap(g, call)
            torch.cuda.synchronize()
            print("  capture=OK")
            g.replay()
            torch.cuda.synchronize()
            print("  replay=OK")
            return g
        except Exception as exc:
            print(f"  capture/replay FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            try:
                torch.cuda.synchronize()
            except Exception:
                print("  stream still poisoned after failure")
            return None

    def bare(g, fn):
        with torch.cuda.graph(g):
            fn()

    def with_fence_off(g, fn):
        with torch.cuda.graph(g):
            with no_fence():
                fn()

    try_capture("torch.cuda.graph around run_unified (no no_fence)", bare)
    try_capture("torch.cuda.graph + no_fence around run_unified", with_fence_off)


def main():
    print(f"HIPDNN_GFX950_3D_GRAPH={os.environ.get('HIPDNN_GFX950_3D_GRAPH')}")
    dump_shape("Llama-3-8B Sk=1024 cus=120 block=16", _req())
    dump_shape("Llama-3-8B Sk=1024 cus=120 block=64", _req(kv_block_size=64))
    dump_shape("Llama-3.1-405B Sk=4096 cus=120", _req(nhead_q=128, seqlen_k=4096))
    dump_shape(
        "Qwen3-30B-A3B Sk=4096 cus=120", _req(nhead_q=32, nhead_k=4, seqlen_k=4096)
    )
    probe_graph()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
