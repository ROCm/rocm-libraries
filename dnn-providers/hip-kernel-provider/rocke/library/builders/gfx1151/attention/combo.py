# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Combine every GPU-verifiable single-wave lever in every way, per shape.

The campaign measured each lever in isolation (``tune``/``sp_tune``/``bn_tune``).
This driver runs the FULL cartesian product of every compatible lever across all
three single-wave kernels -- ``fmha_singlewave`` (``bm_tiles`` x ``p_mode`` x
``v_mode`` x ``q_preload`` x ``fuse_k``), ``fmha_pipelined`` (software pipelining x
``sched`` x ``p_xpose`` x ``fuse_k``), and ``fmha_blockn`` (``bn_tiles`` x
``fuse_k``) -- GPU-verifies each against the numpy reference (tol 2e-2), times it,
and reports the single best combination per shape and globally.

It exists to answer "did any *combination* of levers beat what each scores
alone?" The standout positive interaction it surfaced: ``q_preload + fuse_k``
together on D128 (each is neutral / a win *alone*; together they are the best
single-wave config and clear 20% of the 59 TF peak on the compute-heavy shapes).

Run: ``python -m rocke.examples.gfx1151.attention.combo`` (sweeps a default set
of representative shapes); ``--quick`` for a 2-shape smoke.
"""

from __future__ import annotations

import argparse
import itertools

from .bench_v_staging import _find_objdump
from .fmha_blockn import BlockNCfg
from .fmha_singlewave import SingleWaveCfg
from .fmha_pipelined import PipelinedCfg
from .tune import Shape
from . import tune as opt_tune
from . import sp_tune
from . import bn_tune

PEAK_TF = 59.0


# Representative shapes spanning D64/D128, seqlen, GQA, causal. Chosen to include
# the survey's top performers (compute-heavy D128) where 20% is in reach.
DEFAULT_SHAPES = [
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=64),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128),
    Shape(batch=2, heads=8, kv_heads=2, seqlen_q=1024, seqlen_k=1024, head_size=128),
    Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64, causal=True),
    Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128, causal=True),
]


def _opt_configs(shape: Shape):
    mask = "causal" if shape.causal else "none"
    # bm_tiles=2 only where seqlen allows the larger BLOCK_M; 1 always.
    bms = [1]
    if shape.seqlen_q % 32 == 0:
        bms.append(2)
    grid = itertools.product(
        bms,
        ["lds"],
        ["gather", "lds_t"],
        [0, 1],
        [0, 1],  # bm, p, v, qpreload, fusek
    )
    for bm, pm, vm, qp, fk in grid:
        yield SingleWaveCfg(
            head_size=shape.head_size,
            num_query_heads=shape.heads,
            num_kv_heads=shape.kv_heads,
            mask_mode=mask,
            bm_tiles=bm,
            p_mode=pm,
            v_mode=vm,
            q_preload=bool(qp),
            fuse_k=bool(fk),
        )


def _sp_configs(shape: Shape):
    mask = "causal" if shape.causal else "none"
    for sched, xp, fk in itertools.product([0, 1], ["lds", "shuffle"], [0, 1]):
        yield PipelinedCfg(
            head_size=shape.head_size,
            num_query_heads=shape.heads,
            num_kv_heads=shape.kv_heads,
            mask_mode=mask,
            sched=bool(sched),
            p_xpose=xp,
            fuse_k=bool(fk),
        )


def _bn_configs(shape: Shape):
    mask = "causal" if shape.causal else "none"
    for bn, fk in itertools.product([2, 4], [0, 1]):
        yield BlockNCfg(
            head_size=shape.head_size,
            num_query_heads=shape.heads,
            num_kv_heads=shape.kv_heads,
            mask_mode=mask,
            bn_tiles=bn,
            fuse_k=bool(fk),
        )


def _tag(shape: Shape):
    return (
        f"B{shape.batch} H{shape.heads}"
        + (f"/{shape.kvh}" if shape.kvh != shape.heads else "")
        + f" S{shape.seqlen_q} D{shape.head_size}"
        + ("c" if shape.causal else "")
    )


def _label(kind, cfg):
    if kind == "opt":
        return (
            f"opt bm{cfg.bm_tiles} p{cfg.p_mode} v{cfg.v_mode} "
            f"qp{int(cfg.q_preload)} fk{cfg.fuse_k}"
        )
    if kind == "sp":
        return f"sp sched{int(cfg.sched)} xp{cfg.p_xpose} fk{cfg.fuse_k}"
    return f"bn bn{cfg.bn_tiles} fk{cfg.fuse_k}"


def sweep_shape(shape: Shape, objdump, *, verbose=True):
    runners = [
        ("opt", _opt_configs(shape), opt_tune.verify_and_time),
        ("sp", _sp_configs(shape), sp_tune.verify_and_time),
        ("bn", _bn_configs(shape), bn_tune.verify_and_time),
    ]
    best = None
    for kind, cfgs, vt in runners:
        for cfg in cfgs:
            try:
                r = vt(cfg, shape, objdump=objdump)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    {_label(kind, cfg):46s} BUILD/RUN FAIL: {str(e)[:40]}")
                continue
            ok = r.get("ok")
            tf = r.get("tflops", 0.0)
            if verbose:
                flag = "Y" if ok else "N"
                print(
                    f"    {_label(kind, cfg):46s} {flag} {tf:7.2f} TF "
                    f"spill={r.get('vspill', '-')}"
                )
            if ok and (best is None or tf > best[2]):
                best = (kind, cfg, tf, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true", help="only the two compute-heavy D128 shapes"
    )
    ap.add_argument("--verbose", action="store_true", help="print every config")
    args = ap.parse_args()

    shapes = DEFAULT_SHAPES
    if args.quick:
        shapes = [
            Shape(batch=4, heads=8, seqlen_q=2048, seqlen_k=2048, head_size=128),
            Shape(batch=4, heads=8, seqlen_q=1024, seqlen_k=1024, head_size=64),
        ]

    objdump = _find_objdump()
    rows = []
    for sh in shapes:
        print(f"\n### {_tag(sh)}")
        best = sweep_shape(sh, objdump, verbose=args.verbose)
        if best is None:
            print(f"  {_tag(sh):30s} NO VERIFIED CONFIG")
            continue
        kind, cfg, tf, r = best
        pk = tf / PEAK_TF * 100.0
        rows.append((_tag(sh), kind, cfg, tf, pk))
        print(
            f"  BEST {_tag(sh):30s} -> {tf:7.2f} TF ({pk:4.1f}%) "
            f"[{_label(kind, cfg)}]  max_abs={r['max_abs']:.1e}"
        )

    if rows:
        print("\n==== best combination per shape ====")
        for tag, kind, cfg, tf, pk in rows:
            print(f"{tag:30s} {tf:7.2f} TF {pk:5.1f}%  {_label(kind, cfg)}")
        gbest = max(rows, key=lambda x: x[3])
        avg = sum(x[4] for x in rows) / len(rows)
        n20 = sum(1 for x in rows if x[4] >= 20.0)
        print(
            f"\nglobal best: {gbest[0]} -> {gbest[3]:.2f} TF ({gbest[4]:.1f}%) "
            f"[{_label(gbest[1], gbest[2])}]"
        )
        print(f"avg of per-shape best: {avg:.1f}% of {PEAK_TF} peak")
        print(f"shapes >=20% peak: {n20}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
