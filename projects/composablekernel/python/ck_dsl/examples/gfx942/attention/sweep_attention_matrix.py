# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repeatable gfx942 attention matrix sweep against Torch SDPA.

This runner is intentionally production-dispatch focused: variants are expressed
as environment overlays around ``run_unified_attention_torch`` so the benchmark
exercises the same cache keys, launchers, and selectors used outside the
standalone spec harness.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from ck_dsl.instances import UnifiedAttentionProblem, run_unified_attention_torch
from ck_dsl.runtime import no_fence, synchronize_and_release, time_launches

from parity_unified_attention import (
    Shape,
    attention_tflops,
    compare,
    load_shapes,
    make_inputs,
    ref_paged_attn,
    select_shapes,
)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    backend: str
    env: Dict[str, str]
    shapes: Optional[set[str]] = None

    def applies(self, shape: Shape) -> bool:
        return self.shapes is None or shape.name in self.shapes


ENV_KEYS = (
    "HIPDNN_GFX942_FLASH_WIDE",
    "HIPDNN_GFX942_FLASH_MLIM",
    "HIPDNN_GFX942_CFV",
    "HIPDNN_GFX942_CFV_STORE",
    "HIPDNN_GFX942_CFV_STORE_SPLIT",
    "HIPDNN_GFX942_CFV_CK_VLDS",
    "HIPDNN_GFX942_K_SLICED_RING",
    "HIPDNN_GFX942_K_LDSSEQ",
    "HIPDNN_GFX942_3D_HOIST",
    "HIPDNN_GFX942_KV_CACHE_POLICY",
    "HIPDNN_GFX942_Q_DIRECT",
    "HIPDNN_GFX942_GLOBAL_LOAD_LDS_K",
    "HIPDNN_GFX942_WAVES_PER_EU",
    "HIPDNN_GFX942_NUM_WARPS",
    "HIPDNN_GFX942_Q_MAJOR_GRID",
    "HIPDNN_GFX942_CFV_SCALAR_READ",
    "HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD",
    "HIPDNN_GFX942_CFV_STORE_SCATTER",
    "HIPDNN_GFX942_CFV_STORE_PREZERO",
    "HIPDNN_GFX942_CFV_STORE_SEPOFF",
    "HIPDNN_GFX942_SWIZZLE_VLDS",
)


@contextlib.contextmanager
def env_overlay(env: Dict[str, str]):
    old = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in env.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dense_qkv(shape: Shape, data):
    q = (
        data["query"]
        .view(shape.batch, shape.seqlen_q, shape.heads, shape.head_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    max_blocks = (shape.seqlen_k + 63) // 64
    block_tables = data["block_tables"].detach().cpu().tolist()
    ks, vs = [], []
    for bidx in range(shape.batch):
        idx = torch.tensor(
            block_tables[bidx][:max_blocks],
            device=data["key_cache"].device,
            dtype=torch.long,
        )
        k = data["key_cache"][idx].reshape(-1, shape.kv_heads, shape.head_size)
        v = data["value_cache"][idx].reshape(-1, shape.kv_heads, shape.head_size)
        ks.append(k[: shape.seqlen_k])
        vs.append(v[: shape.seqlen_k])
    k_dense = torch.stack(ks, dim=0).permute(0, 2, 1, 3).contiguous()
    v_dense = torch.stack(vs, dim=0).permute(0, 2, 1, 3).contiguous()
    return q, k_dense, v_dense


def _torch_sdpa(q, k, v, scale: float, *, is_causal: bool):
    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=(q.shape[1] != k.shape[1]),
        )
    except TypeError:
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale
        )


def _problem(shape: Shape, data) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=data["query"].shape[0],
        num_seqs=shape.batch,
        num_query_heads=shape.heads,
        num_kv_heads=shape.kv_heads,
        head_size=shape.head_size,
        block_size=64,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=shape.dtype,
        sliding_window=0,
        softcap=0.0,
        use_sinks=False,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        num_sms=120,
    )


def _run_ck(
    shape: Shape,
    data,
    variant: SweepVariant,
    *,
    warmup: int,
    attempts: int,
    graph: bool = False,
):
    out = torch.empty_like(data["query"])
    problem = _problem(shape, data)
    stream = int(torch.cuda.current_stream().cuda_stream)

    def call_once():
        run_unified_attention_torch(
            problem=problem,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=0.0,
            backend=variant.backend,
            stream=stream,
        )

    if graph:
        # Build all launchers/workspace first outside capture.
        with no_fence():
            call_once()
        synchronize_and_release(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with no_fence():
                call_once()
        synchronize_and_release(stream)
        ms = time_launches(g.replay, warmup=warmup, iters=attempts, stream=stream)
    else:
        ms = time_launches(call_once, warmup=warmup, iters=attempts, stream=stream)
    synchronize_and_release(stream)
    return out, ms


def variants_for_phase(phase: str) -> List[SweepVariant]:
    d128 = {"Fp16_Prefill_GQA_S2048_D128"}
    d64 = {"Fp16_Prefill_GQA_S2048_D64"}
    decode = {"Fp16_Decode_GQA_S2048_D128", "Fp16_Decode_GQA_S2048_D64"}

    common = [
        SweepVariant("auto", "auto", {}),
        SweepVariant("2d", "tiled", {}, d128 | d64),
        SweepVariant("3d", "3d", {}, decode),
    ]
    prefill = [
        SweepVariant("d128_mlim", "auto", {"HIPDNN_GFX942_FLASH_MLIM": "d128"}, d128),
        SweepVariant("mlim_all", "auto", {"HIPDNN_GFX942_FLASH_MLIM": "all"}, d128 | d64),
        SweepVariant("mlim_off", "auto", {"HIPDNN_GFX942_FLASH_MLIM": "0"}, d128 | d64),
        SweepVariant("ksring", "auto", {"HIPDNN_GFX942_K_SLICED_RING": "1"}, d128),
        SweepVariant(
            "ksldsseq",
            "auto",
            {"HIPDNN_GFX942_K_SLICED_RING": "1", "HIPDNN_GFX942_K_LDSSEQ": "1"},
            d128,
        ),
        SweepVariant("wide2", "auto", {"HIPDNN_GFX942_FLASH_WIDE": "2"}, d128 | d64),
        SweepVariant("l4", "auto", {"HIPDNN_GFX942_FLASH_WIDE": "0"}, d128 | d64),
    ]
    decode_variants = [
        SweepVariant("3d_hoist", "3d", {"HIPDNN_GFX942_3D_HOIST": "1"}, decode),
    ]
    if phase == "smoke":
        return common
    if phase == "prefill":
        return common + prefill
    if phase == "decode":
        return common + decode_variants
    if phase == "all":
        return common + prefill + decode_variants
    raise ValueError(f"unknown phase {phase!r}")


def run(args) -> int:
    shapes = select_shapes(load_shapes(), args.scenario)
    variants = variants_for_phase(args.phase)
    print(f"torch,{torch.__version__}")
    print(f"device,{torch.cuda.get_device_name(0)}")
    print(
        "shape,variant,backend,ck_us,ck_tflops,torch_us,torch_tflops,"
        "latency_pct_torch,max_abs,mean_abs"
    )
    with torch.inference_mode():
        for shape in shapes:
            data = make_inputs(shape)
            ref = ref_paged_attn(
                data["query"],
                data["key_cache"],
                data["value_cache"],
                data["query_lens"],
                data["kv_lens_list"],
                data["block_tables"],
                data["scale"],
            ).float()
            q, k, v = _dense_qkv(shape, data)
            is_causal = shape.group != "decode"
            _torch_sdpa(q, k, v, data["scale"], is_causal=is_causal)
            torch.cuda.synchronize()
            stream = int(torch.cuda.current_stream().cuda_stream)

            def torch_once():
                _torch_sdpa(q, k, v, data["scale"], is_causal=is_causal)

            torch_ms = time_launches(
                torch_once, warmup=args.torch_warmup, iters=args.torch_attempts, stream=stream
            )
            synchronize_and_release(stream)
            torch_tf = attention_tflops(shape, torch_ms)
            for variant in variants:
                if not variant.applies(shape):
                    continue
                try:
                    with env_overlay(variant.env):
                        out, ck_ms = _run_ck(
                            shape,
                            data,
                            variant,
                            warmup=args.warmup,
                            attempts=args.attempts,
                            graph=args.ck_graph,
                        )
                    diff = compare(ref, out)
                    ck_tf = attention_tflops(shape, ck_ms)
                    pct = 100.0 * ck_ms / torch_ms if torch_ms > 0 else 0.0
                    print(
                        f"{shape.name},{variant.name},{variant.backend},"
                        f"{ck_ms * 1000:.2f},{ck_tf:.3f},"
                        f"{torch_ms * 1000:.2f},{torch_tf:.3f},"
                        f"{pct:.1f},{diff['max_abs']:.3e},{diff['mean_abs']:.3e}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"{shape.name},{variant.name},{variant.backend},ERROR,0,"
                        f"{torch_ms * 1000:.2f},{torch_tf:.3f},0,{type(exc).__name__}:{exc},0",
                        flush=True,
                    )
            del data, ref, q, k, v
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--phase", choices=("smoke", "prefill", "decode", "all"), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--torch-warmup", type=int, default=10)
    parser.add_argument("--torch-attempts", type=int, default=30)
    parser.add_argument("--ck-graph", action="store_true", help="time CK path through CUDA/HIP graph replay")
    args = parser.parse_args()
    if args.scenario is None:
        args.scenario = ["perf", "decode"]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
