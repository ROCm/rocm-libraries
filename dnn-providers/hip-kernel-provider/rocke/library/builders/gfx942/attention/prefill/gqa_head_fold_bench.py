# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 GQA head-fold A/B: the fold vs the SAME tiled kernel with the fold off.

Reproduces the measurement behind ``gqa_head_fold_case_study.md``. Both arms run
``build_gfx942_4warp_gqa``; the only difference is the value of
``gfx942_gqa_fold_eligible``, the single predicate that drives BOTH the builder
and the launch grid. The fold-off arm is therefore the exact pre-fold kernel, not
a different code path or a different builder.

Two things this script has to get right, and both are easy to get wrong:

  * **Caches.** ``_tiled_cache_key`` is derived from problem attributes only.
    That is correct in production -- the fold is a pure function of head_size,
    num_queries_per_kv, sliding_window, dtype and block_size, all of which are
    already key components -- but it means the key does NOT move when the
    predicate is patched, so a stale launcher would be reused and the "A/B" would
    time the same kernel twice. All three caches are cleared between arms.
  * **Correctness per arm.** A speedup on a wrong kernel is not a speedup, so
    every point is checked. Up to ``_ORACLE_MAX_SQ`` the check is the fp32
    windowed paged-attention oracle; above it the oracle would materialise a
    [heads, sq, sq] score tensor (16 GiB at sq=16384), so the fold is checked
    against the unfolded kernel's own output instead -- they must agree exactly,
    since the fold only repacks which M-tile row holds which (token, head) pair.

The sliding-window cohort cannot be expressed in this directory's own
``Shape``/``parity_unified_attention.py`` (its torch reference is causal-only, no
``sliding_window`` field), so the gfx950 prefill harness is loaded for its
windowed ``ref_paged_attn`` + randn paged ``make_inputs``. That reference is pure
torch and device-agnostic; only the rocke kernel it launches is arch-specific.

Run on a gfx942 device:
    python rocke/library/builders/gfx942/attention/prefill/gqa_head_fold_bench.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# prefill -> attention -> gfx942 -> builders -> library
_LIBROOT = HERE.parents[3]
_PYROOT = _LIBROOT.parent / "platform" / "python"
for _p in (str(_PYROOT), str(_LIBROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

_HARNESS = _LIBROOT / "builders/gfx950/attention/prefill/parity_unified_attention.py"
_spec = importlib.util.spec_from_file_location(
    "parity_unified_attention_fold_ab", str(_HARNESS)
)
H = importlib.util.module_from_spec(_spec)
sys.modules["parity_unified_attention_fold_ab"] = H
_spec.loader.exec_module(H)

from kernels.common import attention_unified as au  # noqa: E402

_REAL_PREDICATE = au.gfx942_gqa_fold_eligible

# Above this the fp32 oracle OOMs; the fold is checked against the unfolded
# kernel instead (they must agree exactly).
_ORACLE_MAX_SQ = 8192

SEQLENS = (512, 1024, 2048, 4096, 8192, 16384)
BLOCK_SIZES = (16, 32)


def _reset_caches() -> None:
    au._2D_LAUNCHERS.clear()
    au._2D_LAUNCH_META.clear()
    au._ATTN_TILED_CACHE.clear()


def _scenario(sq: int, bs: int, window: int):
    return H.Scenario(
        name=f"d128swa_fold_ab_S{sq}_bs{bs}_w{window}",
        seq_lens=[(sq, sq)],
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=bs,
        dtype=torch.bfloat16,
        sliding_window=window,
    )


def _arm(s, data, *, fold: bool, warmup: int, attempts: int):
    au.gfx942_gqa_fold_eligible = _REAL_PREDICATE if fold else (lambda *a, **k: False)
    _reset_caches()
    try:
        return H.run_unified("rocke", s, data, warmup=warmup, attempts=attempts)
    finally:
        au.gfx942_gqa_fold_eligible = _REAL_PREDICATE
        _reset_caches()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=20)
    ap.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=list(SEQLENS),
        help="query/kv seqlens to sweep",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device unavailable; exiting", file=sys.stderr)
        return 1
    dev = torch.cuda.get_device_name(0)
    print(f"device: {dev}")
    print(f"cohort: bf16 D128 GQA 32/8 (4:1), sliding_window={args.window}, paged")
    print()
    print(
        f"{'sq':>7} {'bs':>4} {'nofold ms':>11} {'fold ms':>10} "
        f"{'speedup':>8} {'delta%':>8} {'max_abs':>9}  checked vs"
    )

    rows = []
    for bs in BLOCK_SIZES:
        for sq in args.seqlens:
            s = _scenario(sq, bs, args.window)
            data = H.make_inputs(s, seed=0)
            out_n, ms_n = _arm(
                s, data, fold=False, warmup=args.warmup, attempts=args.attempts
            )
            out_f, ms_f = _arm(
                s, data, fold=True, warmup=args.warmup, attempts=args.attempts
            )
            if sq <= _ORACLE_MAX_SQ:
                ref = H.run_reference(s, data)
                max_abs = float(H.compare(ref, out_f)["max_abs"])
                against = "fp32 ref"
                del ref
            else:
                max_abs = float(H.compare(out_n.float(), out_f)["max_abs"])
                against = "nofold"
            speedup = ms_n / ms_f
            rows.append((sq, bs, ms_n, ms_f, speedup, max_abs))
            print(
                f"{sq:>7} {bs:>4} {ms_n:>11.4f} {ms_f:>10.4f} "
                f"{speedup:>7.3f}x {100 * (speedup - 1):>7.1f}% "
                f"{max_abs:>9.5f}  {against}"
            )
            del data, out_n, out_f
            torch.cuda.empty_cache()

    print()
    best = max(rows, key=lambda r: r[4])
    worst = min(rows, key=lambda r: r[4])
    print(f"best  {best[4]:.3f}x at sq={best[0]} bs={best[1]}")
    print(f"worst {worst[4]:.3f}x at sq={worst[0]} bs={worst[1]}")
    regressions = [(r[0], r[1], round(r[4], 3)) for r in rows if r[4] < 1.0]
    print(f"regressions (<1.00x): {regressions}")
    # The fold must never lose: it strictly removes redundant KV traffic.
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
