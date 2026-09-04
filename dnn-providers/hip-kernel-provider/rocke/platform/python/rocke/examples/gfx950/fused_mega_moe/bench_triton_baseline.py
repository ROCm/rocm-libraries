"""vLLM Triton fused-MoE baseline on the same shapes as the rocKE harness.

The point of this file is that a comparison is only meaningful when both sides
run on the same device, in the same session, over the same routing. It therefore
takes ``--routing-from`` (the numpy harness's cache directory) so both kernels
activate exactly the same experts, and it reports its own measured HBM read
bandwidth so the result can be read as a fraction of what the box was doing at
that moment rather than as an absolute.

One ``fused_experts`` call performs both expert GEMMs (gate/up then down), which
is the same work as one rocKE mega-kernel launch.

Deliberately kept in its own process and its own file: it imports torch, which
must never share a process with a rocKE Comgr compile (the harness's own
docstring explains why). The ``rocke not in sys.modules`` assert below enforces
that from this side.

    python3 -u rocke/examples/gfx950/fused_mega_moe/bench_triton_baseline.py \\
        --shape qwen3 --routing-from $ROCKE_MOE_BENCH_CACHE/qwen3_e128_seed11939
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from functools import partial

import torch

GROUP = 128
FP8_MAX = 448.0


@dataclass(frozen=True)
class Shape:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int


SHAPES = {
    "qwen3": Shape("qwen3", 32, 128, 8, 2048, 768),
    "canonical": Shape("canonical", 8, 8, 2, 4096, 7168),
    "tiny": Shape("tiny", 8, 8, 2, 1024, 512),
}


def log(m: str) -> None:
    print(m, flush=True)


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def measure_bandwidth(nbytes: int = 1 << 30) -> "tuple[float, float]":
    """Achievable HBM bandwidth, as (copy, read-only) GB/s.

    MoE weight streaming is almost pure read, and a read-only stream sustains
    noticeably more than a copy (which pays for writes too), so the read figure
    is the right ceiling to judge these kernels against.
    """
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms_copy = time_ms(partial(dst.copy_, src), warmup=5, iters=20)
    copy_gbs = 2 * nbytes / (ms_copy * 1e-3) / 1e9

    view = src.view(torch.float32)
    ms_read = time_ms(partial(torch.sum, view), warmup=5, iters=20)
    read_gbs = nbytes / (ms_read * 1e-3) / 1e9
    del src, dst, view
    torch.cuda.empty_cache()
    return copy_gbs, read_gbs


def build_inputs(s: Shape, fp8_dtype, routing_dir=None):
    """Random fp8 block-scaled expert weights + routing for the shape.

    When ``routing_dir`` is given the routing is loaded from the numpy
    harness's cache so both kernels activate exactly the same experts -- the
    active-expert count drives total weight traffic, so leaving it to two
    independent RNGs would confound the comparison.
    """
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11939)
    E, I, H, T = s.experts, s.intermediate, s.hidden, s.tokens

    x = (torch.randn(T, H, generator=g, device=dev, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )

    def rand_fp8(*shape):
        r = torch.randn(*shape, generator=g, device=dev, dtype=torch.float32)
        return (r.clamp(-FP8_MAX, FP8_MAX) * 0.05).to(fp8_dtype)

    # w1 holds gate and up stacked along the output axis; w2 is the down proj.
    w1 = rand_fp8(E, 2 * I, H)
    w2 = rand_fp8(E, H, I)
    w1_scale = torch.full(
        (E, (2 * I) // GROUP, H // GROUP), 0.01, device=dev, dtype=torch.float32
    )
    w2_scale = torch.full(
        (E, H // GROUP, I // GROUP), 0.01, device=dev, dtype=torch.float32
    )

    if routing_dir is not None:
        import numpy as np

        ids = np.load(f"{routing_dir}/topk_ids.npy")
        wts = np.load(f"{routing_dir}/topk_weights.npy")
        topk_ids = torch.from_numpy(ids).to(dev).to(torch.int32)
        topk_weights = torch.from_numpy(wts).to(dev).to(torch.float32)
        log(f"routing: loaded from {routing_dir}")
    else:
        logits = torch.randn(T, E, generator=g, device=dev, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(logits, s.topk, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
    return x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="qwen3", choices=sorted(SHAPES))
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--routing-from",
        default="",
        help="numpy-harness cache dir to share routing with (e.g. "
        ".cache/qwen3_e128_seed11939)",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        default="",
        help="write the measurement here as JSON (used by rocke-serve)",
    )
    args = ap.parse_args()

    assert "rocke" not in sys.modules, "baseline must not share a process with rocKE"

    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.platforms import current_platform

    s = SHAPES[args.shape]
    fp8_dtype = current_platform.fp8_dtype()
    props = torch.cuda.get_device_properties(0)
    log(
        f"device: {props.name}  CUs={props.multi_processor_count}  "
        f"vram={props.total_memory / 1e9:.1f} GB  fp8={fp8_dtype}"
    )
    log(
        f"shape={s.name} T={s.tokens} E={s.experts} K={s.topk} "
        f"H={s.hidden} I={s.intermediate}"
    )

    t = time.time()
    copy_gbs, bw = measure_bandwidth()
    log(
        f"achievable HBM bandwidth: read={bw:.0f} GB/s  copy={copy_gbs:.0f} GB/s "
        f"({time.time() - t:.1f}s)"
    )

    x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids = build_inputs(
        s, fp8_dtype, args.routing_from or None
    )
    quant = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale, w2_scale=w2_scale, block_shape=[GROUP, GROUP]
    )

    def call():
        fused_experts(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            global_num_experts=s.experts,
            quant_config=quant,
        )

    try:
        call()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        log(f"fused_experts FAILED: {type(exc).__name__}: {exc}")
        return 1

    active = int(torch.unique(topk_ids).numel())
    weight_bytes = active * s.intermediate * s.hidden * 3  # gate + up + down, 1B

    ms = time_ms(call, args.warmup, args.iters)
    us = ms * 1000.0
    log(
        f"\nvLLM Triton fused_experts: {us:.1f} us/call\n"
        f"  active experts: {active}/{s.experts}\n"
        f"  expert weights streamed: {weight_bytes / 1e6:.0f} MB\n"
        f"  achieved: {weight_bytes / (ms * 1e-3) / 1e9:.0f} GB/s "
        f"({weight_bytes / (ms * 1e-3) / 1e9 / bw * 100:.0f}% of measured peak)"
    )
    if args.json_out:
        import json
        from pathlib import Path

        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "framework": "vllm_triton_fused_experts",
                    "shape": s.name,
                    "latency_us": us,
                    "latency_ms": ms,
                    "active_experts": active,
                    "weight_bytes": weight_bytes,
                    "achieved_gbs": weight_bytes / (ms * 1e-3) / 1e9,
                    "read_peak_gbs": bw,
                    "copy_peak_gbs": copy_gbs,
                    "iters": args.iters,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
