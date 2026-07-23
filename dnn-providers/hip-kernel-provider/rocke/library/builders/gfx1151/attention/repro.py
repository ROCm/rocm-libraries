# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reproducible, hardware-validated FMHA configs + peak-performance record (gfx1151).

All numbers measured on a gfx1151 board (stx-halo-mini, Radeon 8060S / Strix Halo,
40 CU, wave32, 32 MB MALL), dense (non-causal), heads=24, batch=1, fp16. Small-L
configs are numpy-verified bit-close before timing; large-L (>=8K) is timing-only
(a full SxS numpy reference is infeasible -- kernel logic is seqlen-agnostic and is
covered by the small-L verifies; the large-L kernels re-verify correct at L<=2048,
max_abs ~1e-4 for the f16-O-carry paths).

=============================================================================
PEAK PERFORMANCE PER SHAPE (the deliverable record)
=============================================================================
D128 (production head size) -- best kernel per sequence length:

  L        TFLOPS  kernel / config                        regime
  512      ~24     swapqk d128_prod (MQ1)                  compute-bound
  1024     25.45   swapqk d128_mq2_of16 (MQ2 + f16 O)      compute-bound (PEAK)
  2048     24.33   swapqk d128_mq2_of16                    compute-bound
  4096     24.07   swapqk d128_mq2_of16                    knee (MQ2 reuse)
  8192     15.83   pmq  (persistent+MQ2+of16, pers144)     MALL-resident
  16384    12.65   pmq  pers144                            MALL-resident
  32768    11.34   pmq  pers144                            MALL-resident
  # MQ2 register reuse holds ~24-25 TF out to L4K (2x the MQ1 knee); past L4K the
  # persistent qb_major work-queue keeps per-head KV hot in the 32 MB MALL and
  # sustains 11-16 TF (vs the non-persistent cliff: 10.6/4.7/2.6 @ 8K/16K/32K).

D64 (reference) -- register query-blocking dominates the whole range:
  L        TFLOPS  kernel / config
  512-4096 ~29     swapqk d64_mq2 (MQ2, f32 O, bn64)
  8192     25.3    swapqk d64_mq2

=============================================================================
HOW TO REPRODUCE (all via the tracked benchmark.py driver)
=============================================================================
Named swapqk configs also run directly here:
    python -m builders.gfx1151.attention.repro <name> [--verify] [--L 2048 4096 ...]

D128 compute-bound (L<=4096), swapqk MQ2 + f16 O-carry:
    python -m builders.gfx1151.attention.benchmark --kernel swapqk \\
        --seqlen-q 4096 --seqlen-k 4096 --head-size 128 --heads 24 --batch 1 \\
        --set n_waves=2 --set q_block=2 --set o_f16=1 --set block_n=32 \\
        --set qk_ilp=2 --set buffer_gather=1 --set dual_gather=1 --set fast_exp2=1 \\
        --no-verify

D128 large-Sq (L>=8192), PMQ persistent + MQ2 + f16 O-carry (pers144 sweet spot):
    python -m builders.gfx1151.attention.benchmark --kernel pmq \\
        --seqlen-q 16384 --seqlen-k 16384 --head-size 128 --heads 24 --batch 1 \\
        --set n_waves=2 --set q_block=2 --set o_f16=1 --set block_n=32 \\
        --set qk_ilp=2 --set num_persistent=144 --set buffer_gather=1 \\
        --set dual_gather=1 --set fast_exp2=1 --no-verify

Compile-here / run-on-board (host needs no GPU to emit): add ``--emit /tmp/a``,
rsync /tmp/a to the board, then rerun with ``--prebuilt /tmp/a``.

The knob values in CONFIGS below ARE the reproduction record; keep them in sync
with any config change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Validated configs. Each maps a name -> (SwapQKCfg field overrides, sweep Ls,
# and the hardware numbers observed). head_size/heads/batch are set per-run.
# ---------------------------------------------------------------------------
CONFIGS = {
    # Production baseline: transposed-QK D128 (the campaign winner).
    "d128_prod": dict(
        head_size=128,
        n_waves=2,
        block_n=32,
        qk_ilp=2,
        q_block=1,
        o_f16=False,
        buffer_gather=True,
        dual_gather=True,
        lazy_rescale=True,
        fast_exp2=True,
        sched_mode="pingpong",
        _Ls=(512, 1024, 2048, 4096),
        # measured (isolated, cooldown): ~22-24 TF, peak ~24.7 @ L1024;
        # DRAM-bound past ~L4096 (~17-18 TF).
        _expect="~23-24 TF @ L<=2048 (peak 24.7 @ L1024); vgpr=197 spill=0",
    ),
    # Large-Sq query-blocked (MQ=2) at D64 -- the barrier-free reuse win that
    # breaks the DRAM roofline. THIS is the "big win to ~L8K (~26 TF)" config.
    "d64_mq2": dict(
        head_size=64,
        n_waves=2,
        block_n=64,
        qk_ilp=2,
        q_block=2,  # MQ=2 query-blocking (reuse each K/V fragment across 2 groups)
        o_f16=False,
        buffer_gather=True,
        dual_gather=True,
        lazy_rescale=True,
        fast_exp2=True,
        sched_mode="pingpong",
        _Ls=(512, 1024, 2048, 4096, 8192),
        # measured (isolated, cooldown): max_abs=3.05e-5; vgpr=246 spill=0
        #   L512 29.0  L1024 29.2  L2048 29.5  L4096 28.6  L8192 25.3  TF
        _expect="~29 TF through L4096, 25 TF @ L8192; vgpr=246 spill=0 (max_abs 3.05e-5)",
    ),
    # D128 query-blocked (MQ=2) with f16 accumulator carry -- the ONLY way MQ2
    # fits D128's register budget (8 f32 C-frags x 2 groups spills hard; f16
    # C-frags halve accumulator VGPRs). Spills 38 at vgpr=256 but the 2x KV
    # reuse dominates: pushes the DRAM-roofline knee from ~L2K (MQ1) out to ~L4K.
    "d128_mq2_of16": dict(
        head_size=128,
        n_waves=2,
        block_n=32,
        qk_ilp=2,
        q_block=2,  # MQ=2 register KV reuse across 2 query groups
        o_f16=True,  # f16 accumulator carry -- required for MQ2 to fit at D128
        buffer_gather=True,
        dual_gather=True,
        lazy_rescale=True,
        fast_exp2=True,
        sched_mode="pingpong",
        _Ls=(1024, 2048, 4096, 8192, 16384),
        # measured (isolated, cooldown): vgpr=256 spill=38
        #   L1024 25.45  L2048 24.33  L4096 24.07  L8192 10.63  L16384 4.72  TF
        _expect="~24-25 TF through L4096 (peak 25.45 @ L1024), collapses past L8K "
        "(10.6 @ L8192): register reuse extends the knee to ~L4K; vgpr=256 spill=38",
    ),
    # Baseline D64 (MQ=1) for the A/B -- collapses at large L (DRAM roofline).
    "d64_base": dict(
        head_size=64,
        n_waves=2,
        block_n=64,
        qk_ilp=2,
        q_block=1,
        o_f16=False,
        buffer_gather=True,
        dual_gather=True,
        lazy_rescale=True,
        fast_exp2=True,
        sched_mode="pingpong",
        _Ls=(2048, 4096, 8192),
        _expect="~18-24 TF @ L2048 then collapses (13 @ L4096, ~9 @ L8192): DRAM-bound",
    ),
}


def make_cfg(name: str, *, heads: int = 24, kv_heads: int = 0, causal: bool = False):
    """Build the SwapQKCfg for a validated config name."""
    from kernels.gfx1151.wmma_fmha_swapqk import SwapQKCfg

    spec = {k: v for k, v in CONFIGS[name].items() if not k.startswith("_")}
    return SwapQKCfg(
        num_query_heads=heads,
        num_kv_heads=kv_heads,
        mask_mode="causal" if causal else "none",
        **spec,
    )


def main():
    import argparse

    from .benchmark import Shape, verify_and_time_swapqk
    from .bench_v_staging import _find_objdump

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=sorted(CONFIGS))
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--L", type=int, nargs="+", default=None, help="override seqlens")
    ap.add_argument("--verify", action="store_true", help="numpy-check (small L only)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    entry = CONFIGS[args.name]
    Ls = args.L if args.L is not None else list(entry["_Ls"])
    obj = _find_objdump()
    print(f"# {args.name}: expect {entry['_expect']}")
    for L in Ls:
        cfg = make_cfg(args.name, heads=args.heads)
        shape = Shape(
            batch=args.batch,
            heads=args.heads,
            kv_heads=0,
            seqlen_q=L,
            seqlen_k=L,
            head_size=cfg.head_size,
            causal=False,
        )
        r = verify_and_time_swapqk(
            cfg,
            shape,
            objdump=obj,
            verify=args.verify,
            warmup=args.warmup,
            iters=args.iters,
        )
        tag = f"max_abs={r['max_abs']:.2e} ok={r['ok']} " if args.verify else ""
        print(
            f"  L={L:>6}: {tag}{r['us']:9.1f}us {r['tflops']:6.2f} TF "
            f"vgpr={r.get('vgpr')} spill={r.get('vspill')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
