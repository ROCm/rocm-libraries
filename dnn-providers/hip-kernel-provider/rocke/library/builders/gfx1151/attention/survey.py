# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""25-shape survey of the best single-wave WMMA FMHA kernels on gfx1151.

For each shape it runs BOTH best single-wave kernels -- ``fmha_pipelined`` (software
pipelining, the D64 record) and ``fmha_singlewave`` with ``fuse_k`` (the D128 record)
-- keeps the faster, and also times PyTorch SDPA (the only available reference on
this APU is the MATH fallback; flash/efficient are runtime-disabled on gfx1151).
Reports TFLOP/s and % of the ~59 TF f16 WMMA peak for each.
"""

from __future__ import annotations


from .bench_v_staging import _find_objdump
from .fmha_singlewave import SingleWaveCfg
from .fmha_pipelined import PipelinedCfg
from .tune import Shape
from . import tune as opt_tune
from . import sp_tune

PEAK_TF = 59.0  # Radeon 8060S f16 WMMA peak: 40 CU * 512 FLOP/clk * 2.9 GHz


# 25 shapes: vary head_size, seqlen, batch, heads, GQA, causal.
SHAPES = [
    # D64 sweep (seqlen / batch / causal)
    Shape(batch=4, heads=8, seqlen_q=512, seqlen_k=512, head_size=64),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=64),
    Shape(batch=8, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64),
    Shape(batch=2, heads=16, seqlen_q=1024, seqlen_k=1024, head_size=64),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64, causal=True),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=64, causal=True),
    Shape(batch=2, heads=8, kv_heads=2, seqlen_q=1024, seqlen_k=1024, head_size=64),
    # D128 sweep
    Shape(batch=4, heads=8, seqlen_q=512, seqlen_k=512, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128),
    Shape(batch=8, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=2, heads=16, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=1, heads=32, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=128, causal=True),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128, causal=True),
    Shape(batch=2, heads=8, kv_heads=2, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=2, heads=32, kv_heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128),
    # larger sequence / occupancy
    Shape(batch=1, heads=16, seqlen_q=4096, seqlen_k=4096, head_size=64),
    Shape(batch=1, heads=16, seqlen_q=4096, seqlen_k=4096, head_size=128),
    Shape(batch=1, heads=16, seqlen_q=4096, seqlen_k=4096, head_size=128, causal=True),
    Shape(batch=8, heads=16, seqlen_q=512, seqlen_k=512, head_size=64),
    Shape(batch=8, heads=16, seqlen_q=512, seqlen_k=512, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=4096, seqlen_k=4096, head_size=128),
    Shape(batch=2, heads=16, seqlen_q=2048, seqlen_k=2048, head_size=64, causal=True),
]


def _sdpa_tf(shape: Shape):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import time

    dev = "cuda"
    B, H, Hk = shape.batch, shape.heads, shape.kvh
    Sq, Sk, D = shape.seqlen_q, shape.seqlen_k, shape.head_size
    q = torch.randn(B, H, Sq, D, device=dev, dtype=torch.float16) * 0.3
    k = torch.randn(B, Hk, Sk, D, device=dev, dtype=torch.float16) * 0.3
    v = torch.randn(B, Hk, Sk, D, device=dev, dtype=torch.float16) * 0.3
    if Hk != H:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)

    def f():
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=shape.causal
            )

    for _ in range(5):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    N = 30
    for _ in range(N):
        f()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / N * 1e3
    flops = 4.0 * B * H * Sq * Sk * D * (0.5 if shape.causal else 1.0)
    return flops / (ms * 1e-3) / 1e12


def _best_mine(shape: Shape, objdump):
    """Run fmha_pipelined and fmha_singlewave(fuse_k auto); return the faster verified one."""
    results = []
    mask = "causal" if shape.causal else "none"
    sp = PipelinedCfg(
        head_size=shape.head_size,
        num_query_heads=shape.heads,
        num_kv_heads=shape.kv_heads,
        mask_mode=mask,
    )
    opt = SingleWaveCfg(
        head_size=shape.head_size,
        num_query_heads=shape.heads,
        num_kv_heads=shape.kv_heads,
        mask_mode=mask,
        fuse_k=None,
    )
    try:
        r = sp_tune.verify_and_time(sp, shape, objdump=objdump)
        if r["ok"]:
            results.append(("sp", r))
    except Exception as e:  # noqa: BLE001
        print(f"    sp FAIL: {str(e)[:60]}")
    try:
        r = opt_tune.verify_and_time(opt, shape, objdump=objdump)
        if r["ok"]:
            results.append(("opt", r))
    except Exception as e:  # noqa: BLE001
        print(f"    opt FAIL: {str(e)[:60]}")
    if not results:
        return None
    return max(results, key=lambda kr: kr[1]["tflops"])


def main():
    objdump = _find_objdump()
    print(f"{'shape':46s} {'mine':>16s} {'%pk':>5s}  {'sdpa':>9s}  {'speedup':>7s}")
    rows = []
    for sh in SHAPES:
        tag = (
            f"B{sh.batch} H{sh.heads}"
            + (f"/{sh.kvh}" if sh.kvh != sh.heads else "")
            + f" S{sh.seqlen_q} D{sh.head_size}"
            + ("c" if sh.causal else "")
        )
        best = _best_mine(sh, objdump)
        try:
            sdpa = _sdpa_tf(sh)
        except Exception as e:  # noqa: BLE001
            sdpa = float("nan")
            print(f"    sdpa FAIL: {str(e)[:60]}")
        if best is None:
            print(f"{tag:46s} {'BUILD/VERIFY FAIL':>16s}")
            continue
        kind, r = best
        tf = r["tflops"]
        pk = tf / PEAK_TF * 100.0
        spd = tf / sdpa if sdpa == sdpa and sdpa > 0 else float("nan")
        rows.append((tag, kind, tf, pk, sdpa, spd))
        print(f"{tag:46s} {tf:7.2f}TF({kind:3s}) {pk:4.1f}  {sdpa:7.2f}TF  {spd:6.1f}x")

    if rows:
        best = max(rows, key=lambda x: x[2])
        avg_pk = sum(x[3] for x in rows) / len(rows)
        avg_spd = sum(x[5] for x in rows if x[5] == x[5]) / len(rows)
        print(
            f"\nbest: {best[0]} -> {best[2]:.2f} TF ({best[3]:.1f}% of {PEAK_TF} peak)"
        )
        print(
            f"avg: {avg_pk:.1f}% of peak, {avg_spd:.1f}x vs SDPA over {len(rows)} shapes"
        )
        n25 = sum(1 for x in rows if x[3] >= 25.0)
        print(f"shapes >=25% peak: {n25}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
