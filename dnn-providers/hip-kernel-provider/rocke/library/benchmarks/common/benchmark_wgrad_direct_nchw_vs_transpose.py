# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Research spike: direct-style native-NCHW wgrad vs. every other NCHW option.

Sibling of :mod:`benchmarks.common.benchmark_wgrad_nchw_proto_vs_transpose`
(whose shape grid, transposes, repack, NHWC sweep and tuning helpers it
reuses). Arms, all re-measured in the same process:

* ``dnat`` -- :mod:`kernels.common.conv_direct_nchw_wgrad` (the
  direct-style NCHW family: one X window per spatial tile in LDS, shared by
  every tap), reading NCHW ``X`` / NKHW ``dY`` and writing KCYX ``dW``;
* ``inat`` -- :mod:`kernels.common.conv_wgrad_nchw_proto` (the
  implicit-GEMM-style NCHW prototype);
* ``igemm`` pipeline -- transpose ``X`` and ``dY`` to NHWC, the production
  NHWC implicit-GEMM wgrad (single-stage and two-stage legs), repack ``dW``
  KYXC -> KCYX;
* ``direct`` pipeline -- the same transposes, rocke's NHWC direct wgrad
  :func:`kernels.common.conv_direct_wgrad_tiled.build_direct_wgrad_tiled`
  (its output is an fp32 KYXC ``dW``, zeroed before each launch), the
  workspace cast ``conv_wgrad_workspace_reduce`` (``ws_replicas = 1``: fp32
  -> fp16) and the repack. ``conv_direct_wgrad_tiled`` is a stride-1
  algorithm, and ``conv_direct_wgrad_gather`` only serves ``KW * C <= 96``
  (stem layers), so the direct pipeline exists for the stride-1 shapes only.

Fair tuning: each arm's knob space is swept (the NHWC igemm and the igemm NCHW
prototype exactly as in the sibling script; the direct prototype over its
spec grid, thinned by ``--dnat-sample``; the tiled kernel over the AOT
cache's knob rows plus a wider wave / pipeline / segment / grid grid), the
fastest ``--finalists`` of each arm are verified on device against an fp32
CPU ``torch.nn.grad.conv2d_weight`` reference and re-timed, the best verified
one is kept. Each pipeline is verified end to end too. Stage timing: the
minimum over ``--repeats`` of a ``--iters``-sample staged median; ``e2e`` runs
all pipeline launches as one sample.

    python -m benchmarks.common.benchmark_wgrad_direct_nchw_vs_transpose --phase build
    flock <lock> python -m benchmarks.common.benchmark_wgrad_direct_nchw_vs_transpose --phase run
"""

from __future__ import annotations

import argparse
import functools
import itertools
import os
import pickle
import random
import tempfile
from dataclasses import dataclass, replace
from typing import List, Optional

from benchmarks.common import benchmark_wgrad_nchw_proto_vs_transpose as base
from benchmarks.common.benchmark_wgrad_nchw_proto_vs_transpose import (
    Cand,
    Shape,
    _Art,
    _fenced,
    _launcher,
    _rel_err,
    _repack,
    _time_us,
    _transpose,
    _tune,
    _u8,
    full_grid,
)

_TOL = base._TOL
_SPLITS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_TILED = "direct_wgrad_tiled"


# ---------------------------------------------------------------------------
# Knob spaces
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _auto_pads_key(problem, tile_c, tile_h, tile_w, stage_group, y_groups):
    from kernels.common.conv_direct_nchw_wgrad import (
        DirectNchwWgradSpec,
        conflict_free_pads,
    )

    spec = DirectNchwWgradSpec(
        problem=problem,
        tile_m=16,
        tile_c=tile_c,
        tile_h=tile_h,
        tile_w=tile_w,
        waves_m=1,
        waves_c=tile_c // 16,
        stage_group=stage_group,
        y_groups=y_groups,
        kmap="2row",
        epilogue="direct",
    )
    _ways, pad, wpad = conflict_free_pads(spec)
    return pad, wpad


def _auto_pads(spec):
    """Conflict-free ``(lds_pad, lds_wpad)`` for a "2row" spec (model search;
    the X-read pattern depends only on the window geometry and tile_c)."""
    return _auto_pads_key(
        spec.problem,
        spec.tile_c,
        spec.tile_h,
        spec.tile_w,
        spec.stage_group,
        spec.y_groups,
    )


def _with_layout(spec, layout):
    """``spec`` with its X-window layout: ("row", pad) or ("2row", None = model pads)."""
    kmap, pad = layout
    if kmap == "row":
        return replace(spec, kmap="row", lds_pad=pad, lds_wpad=0)
    s2 = replace(
        spec, kmap="2row", lds_pad=0, lds_wpad=0, waves_per_eu=0, prefetch=True
    )
    try:
        s2.validate()
    except ValueError:
        return None
    pad, wpad = _auto_pads(s2)
    return replace(spec, kmap="2row", lds_pad=pad, lds_wpad=wpad)


_LAYOUTS = (("row", 0), ("row", 8), ("2row", None))


def dnat_specs(shape: Shape, arch: str, sample: float, seed: int) -> list:
    from kernels.common.conv_direct_nchw_wgrad import (
        DirectNchwWgradSpec,
        is_valid_direct_nchw_wgrad_spec,
    )

    p = shape.nchw_problem()

    def make(layout, **kw):
        kw.setdefault("epilogue", "lds" if kw.get("y_groups", 1) == 1 else "direct")
        spec = DirectNchwWgradSpec(problem=p, **kw)
        if spec.acc_vgprs > 192 or spec.taps_per_wave > spec.wg_taps:
            return None
        try:
            spec.validate()
        except ValueError:
            return None
        spec = _with_layout(spec, layout)
        if spec is None or not is_valid_direct_nchw_wgrad_spec(spec, arch)[0]:
            return None
        return spec

    grid = []
    for tm, tc, (th, tw), (wm, wc, wt), yg, g, layout, (pf, wpe) in itertools.product(
        (32, 64, 128),
        (32, 64, 128),
        (
            (1, 32),
            (2, 32),
            (4, 32),
            (8, 32),
            (1, 64),
            (2, 64),
            (4, 64),
            (2, 16),
            (4, 16),
            (8, 16),
            (4, 8),
            (8, 8),
        ),
        (
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (4, 1, 1),
            (1, 4, 1),
            (1, 1, 3),
            (2, 1, 3),
            (1, 2, 3),
            (2, 2, 3),
            (1, 1, 9),
        ),
        (1, 3),
        (4, 8),
        _LAYOUTS,
        ((True, 0), (False, 0), (True, 2), (False, 2)),
    ):
        if wpe and tm * tc // (wm * wc * 256) * (9 // yg // wt) * 4 < 64:
            continue  # nothing for the occupancy cap to buy
        spec = make(
            layout,
            tile_m=tm,
            tile_c=tc,
            tile_h=th,
            tile_w=tw,
            waves_m=wm,
            waves_c=wc,
            waves_t=wt,
            y_groups=yg,
            stage_group=g,
            waves_per_eu=wpe,
            prefetch=pf,
        )
        if spec is not None:
            grid.append(spec)
    # A small curated set, always kept.
    curated = []
    for tm, tc, (th, tw), (wm, wc, wt), yg, layout, (pf, wpe) in itertools.product(
        (32, 64),
        (32, 64),
        ((2, 32), (4, 32), (8, 32), (4, 16)),
        ((2, 2, 1), (2, 1, 1), (1, 1, 3), (2, 1, 3)),
        (1, 3),
        _LAYOUTS,
        ((True, 0), (False, 2)),
    ):
        spec = make(
            layout,
            tile_m=tm,
            tile_c=tc,
            tile_h=th,
            tile_w=tw,
            waves_m=wm,
            waves_c=wc,
            waves_t=wt,
            y_groups=yg,
            prefetch=pf,
            waves_per_eu=wpe,
        )
        if spec is not None:
            curated.append(spec)
    # Plain-store epilogue (split 1, no dW memset): curated points only -- at
    # split 1 a workgroup walks every spatial tile serially, which is slower
    # than split-K + memset on most shapes.
    curated += [replace(s, atomic=False) for s in curated if s.epilogue == "lds"]
    rng = random.Random(seed)
    picked = [s for s in grid if rng.random() < sample] if sample < 1 else grid
    return list(dict.fromkeys(picked + curated))


def dnat_splits(spec) -> list:
    """Split degrees worth timing: 128..2048 workgroups, at most one tile each
    (split 1 only for the plain-store epilogue)."""
    if not spec.atomic:
        return [1]
    gx, gy, _ = spec.grid(1)
    base_wg = gx * gy
    out = [d for d in _SPLITS if d <= spec.n_tiles and 128 <= base_wg * d <= 2048]
    if not out:
        cands = [d for d in _SPLITS if d <= spec.n_tiles]
        out = [max(cands)] if base_wg * max(cands) < 128 else [min(cands)]
    return out


def tiled_knobs() -> List[dict]:
    """The AOT cache's tiled-wgrad rows plus a wider grid around them."""
    from kernels.conv import direct_aot_jobs as aj

    out = []
    waves = set(aj.WGRAD_TILED_WAVES) | {
        (2, 2, 2, 4),
        (1, 1, 2, 2),
        (1, 2, 2, 2),
        (2, 1, 4, 2),
        (2, 2, 1, 2),
        (2, 2, 2, 1),
        (1, 1, 4, 4),
        (2, 1, 2, 4),
        (4, 1, 2, 2),
        (1, 4, 2, 2),
    }
    pipes = set(aj.WGRAD_TILED_PIPE) | {(2, 4), (3, 0), (3, 4), (4, 0)}
    for (wk, wc, nk, nc), (pf, ru), rps, tb in itertools.product(
        sorted(waves),
        sorted(pipes),
        (0,) + tuple(aj.WGRAD_TILED_RPS) + (60,),
        tuple(aj.WGRAD_TILED_BLOCKS) + (1024,),
    ):
        out.append(
            dict(
                wave_k16=wk,
                wave_c16=wc,
                waves_k=nk,
                waves_c=nc,
                prefetch=pf,
                row_unroll=ru,
                rows_per_seg=rps,
                target_blocks=tb,
            )
        )
    w, pp, rpss, tbs, uns = aj.WGRAD_TILED_PINGPONG
    for rps, tb, un in itertools.product(rpss, tbs, uns):
        out.append(
            dict(
                wave_k16=w[0],
                wave_c16=w[1],
                waves_k=w[2],
                waves_c=w[3],
                prefetch=pp[0],
                row_unroll=pp[1],
                rows_per_seg=rps,
                target_blocks=tb,
                unfold_n=un,
                q_groups=2,
                pingpong=True,
            )
        )
    for w, pp, rps, un in aj.WGRAD_TILED_PP_HALVES:
        out.append(
            dict(
                wave_k16=w[0],
                wave_c16=w[1],
                waves_k=w[2],
                waves_c=w[3],
                prefetch=pp[0],
                row_unroll=pp[1],
                rows_per_seg=rps,
                target_blocks=aj.WGRAD_TILED_PP_HALVES_BLOCKS,
                unfold_n=un,
                pingpong=True,
            )
        )
    for w, pp, un in aj.WGRAD_TILED_AUTO:
        out.append(
            dict(
                wave_k16=w[0],
                wave_c16=w[1],
                waves_k=w[2],
                waves_c=w[3],
                prefetch=pp[0],
                row_unroll=pp[1],
                rows_per_seg=0,
                target_blocks=aj.WGRAD_TILED_AUTO_BLOCKS,
                unfold_n=un,
            )
        )
    uniq = []
    for k in out:
        if k not in uniq:
            uniq.append(k)
    return uniq


# ---------------------------------------------------------------------------
# Build (no GPU)
# ---------------------------------------------------------------------------


def _build_dnat_one(args_tuple):
    spec, arch = args_tuple
    from rocke import compile_kernel
    from kernels.common.conv_direct_nchw_wgrad import build_direct_conv_nchw_wgrad

    try:
        art = compile_kernel(build_direct_conv_nchw_wgrad(spec, arch=arch), arch=arch)
    except Exception:
        return None
    return spec, _Art(art.kernel_name, art.hsaco)


def _build_tiled_one(args_tuple):
    knobs, arch = args_tuple
    from rocke import compile_kernel
    from kernels.conv.direct_aot_jobs import DirectCaps, probe_problem
    from kernels.conv.direct_variants import (
        make_direct_spec,
        select_direct_kernel_builder,
        validate_direct_spec,
    )

    caps = DirectCaps(KH=3, PAD=1, stride=1, cpg=0, kpg=0)
    try:
        probe = probe_problem(caps, "fp16", _TILED, knobs)
        spec = make_direct_spec(_TILED, probe, knobs)
        if not validate_direct_spec(_TILED, spec, arch)[0]:
            return None
        art = compile_kernel(
            select_direct_kernel_builder(_TILED, spec, arch), arch=arch
        )
    except Exception:
        return None
    return knobs, _Art(art.kernel_name, art.hsaco)


def _dump_atomic(obj, path: str) -> None:
    """Pickle to ``path`` via a rename, so a reader never sees a partial file."""
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)


def _tiled_cache(args) -> str:
    return os.path.join(args.cache_dir, "wgrad_direct_bench_tiled.pkl")


def _shape_cache(args, shape: Shape) -> str:
    return os.path.join(args.cache_dir, f"wgrad_direct_bench_{shape.label()}.pkl")


def build_tiled(arch: str, args) -> list:
    from benchmarks.common.benchmark_implicit_gemm_conv import _build_ir_parallel
    from benchmarks.common import benchmark_wgrad_direct_nchw_vs_transpose as me

    knobs = tiled_knobs()
    print(f"[build] tiled NHWC direct wgrad: {len(knobs)} knob sets", flush=True)
    built = _build_ir_parallel(
        [(k, arch) for k in knobs], me._build_tiled_one, args.jobs
    )
    print(f"  built {len(built)}", flush=True)
    _dump_atomic(built, _tiled_cache(args))
    return built


def build_phase(shape: Shape, arch: str, args) -> None:
    import torch
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _build_and_compile_wgrad_one,
        _build_and_compile_wgrad_two_stage_one,
        _build_ir_parallel,
    )
    from benchmarks.common import benchmark_wgrad_direct_nchw_vs_transpose as me

    print(f"[build] {shape.label()}", flush=True)
    dn = dnat_specs(shape, arch, args.dnat_sample, args.seed)
    print(f"  dnat: {len(dn)} specs", flush=True)
    dn_built = _build_ir_parallel(
        [(s, arch) for s in dn], me._build_dnat_one, args.jobs
    )
    nat = base.native_specs(shape, arch)
    print(f"  inat: {len(nat)} specs", flush=True)
    nat_built = _build_ir_parallel(
        [(s, arch) for s in nat], base._build_native_one, args.jobs
    )
    p = shape.nhwc_problem()
    single, two = base.nhwc_combos(shape, arch, args)
    print(
        f"  igemm: {len(single)} single-stage + {len(two)} two-stage combos", flush=True
    )
    s_built = _build_ir_parallel(
        [(c, p, "fp16", arch) for c in single], _build_and_compile_wgrad_one, args.jobs
    )
    t_built = _build_ir_parallel(
        [(c, p, "fp16", arch) for c in two],
        _build_and_compile_wgrad_two_stage_one,
        args.jobs,
    )
    torch.manual_seed(0)
    X = torch.empty(shape.N, shape.C, shape.H, shape.W).uniform_(-1.0, 1.0).half()
    dY = torch.empty(shape.N, shape.K, shape.Ho, shape.Wo).uniform_(-1.0, 1.0).half()
    torch.set_num_threads(os.cpu_count() or 1)
    ref = torch.nn.grad.conv2d_weight(
        X.float(),
        (shape.K, shape.C, shape.KS, shape.KS),
        dY.float(),
        stride=shape.stride,
        padding=shape.PAD,
    )
    data = {
        "dnat": dn_built,
        "native": [(s, _Art(a.kernel_name, a.hsaco)) for s, a in nat_built],
        "single": [(c, s, k, _Art(a.kernel_name, a.hsaco)) for c, s, k, a in s_built],
        "two": [
            (c, s, k, _Art(a1.kernel_name, a1.hsaco), _Art(a2.kernel_name, a2.hsaco))
            for c, s, k, a1, a2 in t_built
        ],
        "X": X,
        "dY": dY,
        "ref": ref,
    }
    _dump_atomic(data, _shape_cache(args, shape))
    print(
        f"  built dnat={len(dn_built)} inat={len(nat_built)} igemm single={len(s_built)} "
        f"two={len(t_built)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Run (GPU)
# ---------------------------------------------------------------------------


def _tune_f32(
    cands: List[Cand], rt, shape, ref_kcyx, args, label: str
) -> Optional[Cand]:
    """:func:`_tune` for an fp32 KYXC output (the tiled direct kernel)."""
    import torch

    from rocke.runtime import synchronize_and_release

    ref = ref_kcyx.permute(0, 2, 3, 1).contiguous()
    nbytes = shape.K * shape.C * shape.KS * shape.KS * 4
    for c in cands:
        try:
            c.quick_us = _time_us(c.fn, args.quick_warmup, args.quick_iters)
        except Exception as e:
            print(f"  [{label}] {c.desc}: launch failed ({e})", flush=True)
            c.quick_us = float("nan")
    synchronize_and_release(0)
    ranked = sorted(
        (c for c in cands if c.quick_us == c.quick_us), key=lambda c: c.quick_us
    )
    best, n_fail = None, 0
    for c in ranked[: args.finalists]:
        c.fn()  # the candidate zeroes dW itself; poison is pointless for atomics
        synchronize_and_release(0)
        out = torch.empty(shape.K, shape.KS, shape.KS, shape.C, dtype=torch.float32)
        rt.memcpy_d2h(_u8(out), c.out_dev, nbytes)
        c.rel = _rel_err(out, ref)
        if c.rel >= _TOL:
            n_fail += 1
            print(f"  [{label}] FAIL verify {c.desc}: rel_err={c.rel:.3e}", flush=True)
            continue
        c.us = _time_us(c.fn, args.warmup, args.iters)
        if best is None or c.us < best.us:
            best = c
    print(
        f"  [{label}] {len(cands)} candidates timed, {min(len(ranked), args.finalists)} "
        f"finalists verified ({n_fail} failed); best: "
        + (f"{best.desc} rel_err={best.rel:.2e}" if best else "none"),
        flush=True,
    )
    return best


@dataclass
class Row:
    label: str
    t: dict
    desc: dict
    rel: dict

    def to_json(self) -> str:
        import json

        return json.dumps(
            {"label": self.label, "t": self.t, "desc": self.desc, "rel": self.rel}
        )

    @staticmethod
    def from_json(line: str) -> "Row":
        import json

        d = json.loads(line)
        return Row(d["label"], d["t"], d["desc"], d["rel"])


def run_phase(shape: Shape, data: dict, tiled: list, arch: str, args) -> Optional[Row]:
    import torch

    from rocke import compile_kernel
    from rocke.runtime import resolve_stream, synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import LaunchConfig
    from benchmarks.common.benchmark_implicit_gemm_conv import _stage2_launch
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nchw_wgrad import (
        direct_nchw_wgrad_launch_values,
        direct_nchw_wgrad_signature,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_signature,
    )
    from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

    print(f"[run] {shape.label()}", flush=True)
    X, dY, ref = data["X"], data["dY"], data["ref"]
    taps = shape.KS * shape.KS
    dw_nb = shape.K * shape.C * taps * 2
    stream = resolve_stream(0)
    rt = Runtime()
    names = ("X", "dY", "Xt", "dYt", "dWd", "dWn", "dWk", "dWr", "dW32")
    sizes = (
        X.nbytes,
        dY.nbytes,
        X.nbytes,
        dY.nbytes,
        dw_nb,
        dw_nb,
        dw_nb,
        dw_nb,
        2 * dw_nb,
    )
    dev = {n: rt.alloc(sz) for n, sz in zip(names, sizes)}
    bufs = list(dev.values())
    try:
        rt.memcpy_h2d(dev["X"], _u8(X), X.nbytes)
        rt.memcpy_h2d(dev["dY"], _u8(dY), dY.nbytes)
        HW, HoWo = shape.H * shape.W, shape.Ho * shape.Wo
        xT = _transpose(rt, arch, shape.C, HW, shape.N, dev["X"], dev["Xt"])
        dyT = _transpose(rt, arch, shape.K, HoWo, shape.N, dev["dY"], dev["dYt"])
        rp = _repack(arch, shape.K, taps, shape.C, dev["dWk"], dev["dWr"])
        for la, vals, cfg in (xT, dyT):
            la(vals, config=_fenced(cfg))
        Xt = torch.empty(shape.N, shape.H, shape.W, shape.C, dtype=torch.float16)
        dYt = torch.empty(shape.N, shape.Ho, shape.Wo, shape.K, dtype=torch.float16)
        rt.memcpy_d2h(_u8(Xt), dev["Xt"], Xt.nbytes)
        rt.memcpy_d2h(_u8(dYt), dev["dYt"], dYt.nbytes)
        ok_t = torch.equal(Xt, X.permute(0, 2, 3, 1)) and torch.equal(
            dYt, dY.permute(0, 2, 3, 1)
        )
        if not ok_t:
            print("  [FAIL] transposes", flush=True)
            return None

        # ---- arm: direct NCHW prototype --------------------------------
        dn: List[Cand] = []
        for spec, art in data["dnat"]:
            la = base._try_launcher(art, direct_nchw_wgrad_signature(spec))
            if la is None:
                continue
            for sk in dnat_splits(spec):
                vals = direct_nchw_wgrad_launch_values(
                    dev["dY"], dev["X"], dev["dWd"], dY.nbytes, X.nbytes, sk
                )
                run = la.bind(
                    vals,
                    config=LaunchConfig(
                        grid=spec.launch_grid(sk), block=(spec.threads, 1, 1)
                    ),
                )

                if spec.atomic:

                    def fn(_run=run):
                        rt.memset_async(dev["dWd"], 0, dw_nb, stream)
                        _run()

                else:
                    fn = run  # every dW element has one plain-store writer
                dn.append(
                    Cand(
                        f"{art.kernel_name} split{sk}",
                        fn,
                        dev["dWd"],
                        "kcyx",
                        hsaco=art.hsaco,
                    )
                )
        best_d = _tune(dn, rt, shape, ref, args, "dnat")

        # ---- arm: implicit-GEMM NCHW prototype -------------------------
        nat = base.native_cands(
            data["native"],
            rt,
            stream,
            dev["dY"],
            dev["X"],
            dev["dWn"],
            dY.nbytes,
            X.nbytes,
            dw_nb,
        )
        best_n = _tune(nat, rt, shape, ref, args, "inat")

        # ---- arm: NHWC igemm (on transposed data) ----------------------
        nh, ws_dev = base.nhwc_cands(
            data["single"],
            data["two"],
            rt,
            stream,
            shape,
            arch,
            dev["dYt"],
            dev["Xt"],
            dev["dWk"],
            dY.nbytes,
            X.nbytes,
            dw_nb,
        )
        if ws_dev is not None:
            bufs.append(ws_dev)
        best_h = _tune(nh, rt, shape, ref, args, "igemm")

        # ---- arm: NHWC direct (tiled, stride 1 only) --------------------
        best_t = None
        cast = None
        if shape.stride == 1 and tiled:
            p_dir = shape.nchw_problem()
            sig = conv_direct_args_signature("fp16", direction="wgrad")
            vals = ConvArgs.from_problem(p_dir, direction="wgrad").to_launch_values(
                int(dev["dYt"]),
                int(dev["Xt"]),
                int(dev["dW32"]),
                dY.nbytes,
                X.nbytes,
                2 * dw_nb,
            )
            tc: List[Cand] = []
            for knobs, art in tiled:
                spec = make_direct_spec(_TILED, p_dir, knobs)
                if not validate_direct_spec(_TILED, spec, arch)[0]:
                    continue
                grid, block = spec.launch_geometry()
                la = base._try_launcher(art, sig)
                if la is None:
                    continue
                run = la.bind(vals, config=LaunchConfig(grid=grid, block=block))

                def fn(_run=run):
                    rt.memset_async(dev["dW32"], 0, 2 * dw_nb, stream)
                    _run()

                tc.append(
                    Cand(
                        f"{art.kernel_name}", fn, dev["dW32"], "kyxc32", hsaco=art.hsaco
                    )
                )
            best_t = _tune_f32(tc, rt, shape, ref, args, "tiled")
            cs = WgradReduceSpec(
                problem=shape.nhwc_problem(), dtype_d="fp16", groups=1, ws_replicas=1
            )
            cart = compile_kernel(
                build_conv_wgrad_workspace_reduce(cs, arch=arch), arch=arch
            )
            cv, cc = _stage2_launch(
                cs,
                ws_ptr=dev["dW32"],
                ws_bytes=2 * dw_nb,
                dw_ptr=dev["dWk"],
                dw_bytes=dw_nb,
            )
            cast = _launcher(cart, wgrad_reduce_signature(cs)).bind(cv, config=cc)
        if best_d is None or best_n is None or best_h is None:
            return None

        # ---- measured run ------------------------------------------------
        run_x = xT[0].bind(xT[1], config=xT[2])
        run_dy = dyT[0].bind(dyT[1], config=dyT[2])
        run_rp = rp[0].bind(rp[1], config=rp[2])

        def pipe_i():
            run_x()
            run_dy()
            best_h.fn()
            run_rp()

        def pipe_d():
            run_x()
            run_dy()
            best_t.fn()
            cast()
            run_rp()

        out = torch.empty(shape.K, shape.C, shape.KS, shape.KS, dtype=torch.float16)
        rel = {}
        pipes = [("pipe_i", pipe_i)] + (
            [("pipe_d", pipe_d)] if best_t is not None else []
        )
        for name, pf in pipes:
            for n in ("Xt", "dYt"):
                rt.memset(dev[n], 0xFF, X.nbytes if n == "Xt" else dY.nbytes)
            rt.memset(dev["dWk"], 0xFF, dw_nb)
            rt.memset(dev["dWr"], 0xFF, dw_nb)
            pf()
            synchronize_and_release(0)
            rt.memcpy_d2h(_u8(out), dev["dWr"], dw_nb)
            rel[name] = _rel_err(out, ref)
        for name, c in (("dnat", best_d), ("inat", best_n)):
            rt.memset(c.out_dev, 0xFF, dw_nb)
            c.fn()
            synchronize_and_release(0)
            rt.memcpy_d2h(_u8(out), c.out_dev, dw_nb)
            rel[name] = _rel_err(out, ref)
        if best_t is not None:
            rel["tiled"] = best_t.rel
        if max(rel.values()) >= _TOL:
            print(f"  [FAIL] measured run: {rel}", flush=True)
            return None
        stages = {
            "dnat": best_d.fn,
            "inat": best_n.fn,
            "xT": run_x,
            "dyT": run_dy,
            "igemm": best_h.fn,
            "repack": run_rp,
            "e2e_i": pipe_i,
        }
        if best_t is not None:
            stages.update({"tiled": best_t.fn, "cast": cast, "e2e_d": pipe_d})
        reps = {k: [] for k in stages}
        for _ in range(args.repeats):
            for k, f in stages.items():
                reps[k].append(_time_us(f, args.warmup, args.iters))
        print(
            "  repeats (us): "
            + "  ".join(
                f"{k}=" + "/".join(f"{v:.1f}" for v in vs) for k, vs in reps.items()
            ),
            flush=True,
        )
        t = {k: min(vs) for k, vs in reps.items()}
        desc = {"dnat": best_d.desc, "inat": best_n.desc, "igemm": best_h.desc}
        if best_t is not None:
            desc["tiled"] = best_t.desc
        synchronize_and_release(0)
        if args.isa_dir:
            os.makedirs(args.isa_dir, exist_ok=True)
            for tag, c in (
                ("dnat", best_d),
                ("inat", best_n),
                ("igemm", best_h),
                ("tiled", best_t),
            ):
                if c is not None:
                    with open(
                        os.path.join(args.isa_dir, f"{shape.label()}_{tag}.hsaco"), "wb"
                    ) as f:
                        f.write(c.hsaco)
        return Row(shape.label(), t, desc, rel)
    finally:
        for buf in bufs:
            rt.free(buf)


def print_rows(rows: List[Row]) -> None:
    print("\n" + "=" * 150)
    print(
        "wgrad, NCHW caller (us). pipe_i = xT + dyT + NHWC igemm wgrad + repack; "
        "pipe_d = xT + dyT + NHWC direct (tiled) wgrad + fp32->fp16 cast + repack. "
        "e2e = all pipeline launches as one sample; sum = stages timed one by one. "
        "Every wgrad stage includes its memset."
    )
    hdr = (
        f"{'shape':<26}{'dnat':>8}{'inat':>8}{'xT':>7}{'dyT':>7}{'igemm':>8}{'tiled':>8}"
        f"{'cast':>7}{'repack':>7}{'e2e_i':>8}{'e2e_d':>8}"
        f"{'i/d e2e':>9}{'d/dn e2e':>9}{'inat/dn':>9}{'dn/igemm':>9}{'dn/tiled':>9}"
    )
    print(hdr)
    for r in rows:
        t = r.t
        g = lambda k: t.get(k, float("nan"))  # noqa: E731
        print(
            f"{r.label:<26}{g('dnat'):>8.1f}{g('inat'):>8.1f}{g('xT'):>7.1f}{g('dyT'):>7.1f}"
            f"{g('igemm'):>8.1f}{g('tiled'):>8.1f}{g('cast'):>7.1f}{g('repack'):>7.1f}"
            f"{g('e2e_i'):>8.1f}{g('e2e_d'):>8.1f}"
            f"{g('e2e_i') / g('dnat'):>8.2f}x{g('e2e_d') / g('dnat'):>8.2f}x"
            f"{g('inat') / g('dnat'):>8.2f}x{g('dnat') / g('igemm'):>8.2f}x{g('dnat') / g('tiled'):>8.2f}x"
        )
    print("-" * 150)
    for r in rows:
        print(r.label)
        for k, v in r.desc.items():
            print(f"    {k:<6} {v}")
        print("    rel " + "  ".join(f"{k}={v:.1e}" for k, v in r.rel.items()))


# ---------------------------------------------------------------------------
# Shared-GPU guard
# ---------------------------------------------------------------------------


class ForeignGuard:
    """Samples ``rocm-smi --showpids`` in a thread; records any GPU process
    other than ``gpuagent`` and this one, so a shape whose run overlapped
    foreign work can be discarded and re-run."""

    def __init__(self, period: float = 3.0):
        import shutil
        import threading

        self.smi = shutil.which("rocm-smi")
        self.period = period
        self.seen: List[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._loop, daemon=True)
        if self.smi:
            self._t.start()

    def foreign_now(self) -> List[str]:
        import subprocess

        if not self.smi:
            return []
        try:
            out = subprocess.run(
                [self.smi, "--showpids"], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception:
            return []
        rows = []
        for line in out.splitlines():
            f = line.split()
            if (
                len(f) >= 2
                and f[0].isdigit()
                and f[1] != "gpuagent"
                and int(f[0]) != os.getpid()
            ):
                rows.append(f"pid={f[0]} name={f[1]} cu={f[-1]}")
        return rows

    def _loop(self):
        import time

        while not self._stop.is_set():
            rows = self.foreign_now()
            if rows:
                stamp = time.strftime("%H:%M:%S")
                with self._lock:
                    self.seen.extend(f"{stamp} {r}" for r in rows)
            self._stop.wait(self.period)

    def mark(self) -> int:
        with self._lock:
            return len(self.seen)

    def since(self, mark: int) -> List[str]:
        with self._lock:
            return self.seen[mark:]

    def wait_idle(self, poll: float = 30.0) -> None:
        import time

        while True:
            rows = self.foreign_now()
            if not rows:
                return
            print(f"  [guard] foreign GPU work, waiting: {rows}", flush=True)
            time.sleep(poll)

    def stop(self):
        self._stop.set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--phase", choices=("build", "run", "all"), default="all")
    parser.add_argument(
        "--shapes", default="", help="comma-separated grid indices (default: all)"
    )
    parser.add_argument("--cache-dir", default=tempfile.gettempdir())
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument(
        "--sample", type=float, default=0.03, help="NHWC igemm grid fraction"
    )
    parser.add_argument(
        "--dnat-sample", type=float, default=0.25, help="direct proto grid fraction"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-tiles", type=int, default=200)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--quick-warmup", type=int, default=1)
    parser.add_argument("--quick-iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--isa-dir", default="", help="write the best hsaco per arm here"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="re-runs of a shape hit by foreign GPU work",
    )
    parser.add_argument(
        "--rows-json", default="", help="append each measured row here (JSON lines)"
    )
    parser.add_argument(
        "--report", default="", help="print the table of a --rows-json file and exit"
    )
    parser.add_argument(
        "--fork",
        action="store_true",
        help="run phase: one child process per shape (the HIP code objects of every "
        "candidate stay loaded for a process' lifetime)",
    )
    parser.add_argument(
        "--no-tiled", action="store_true", help="skip the NHWC direct arm"
    )
    args = parser.parse_args()

    if args.report:
        with open(args.report) as f:
            rows = [Row.from_json(line) for line in f if line.strip()]
        order = {s.label(): i for i, s in enumerate(full_grid())}
        print_rows(sorted(rows, key=lambda r: order.get(r.label, 99)))
        return 0
    shapes = full_grid()
    idx = [int(s) for s in args.shapes.split(",") if s] or list(range(len(shapes)))
    if args.phase == "run" and args.fork and len(idx) > 1:
        import subprocess
        import sys

        rc = 0
        for i in idx:
            argv = [a for a in sys.argv[1:] if a != "--fork"]
            j = argv.index("--shapes") if "--shapes" in argv else -1
            if j >= 0:
                del argv[j : j + 2]
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.common.benchmark_wgrad_direct_nchw_vs_transpose",
            ]
            rc |= subprocess.call(cmd + argv + ["--shapes", str(i)])
        if args.rows_json:
            with open(args.rows_json) as f:
                rows = [Row.from_json(line) for line in f if line.strip()]
            print_rows(rows)
        return rc
    for i in idx:
        print(f"#{i}: {shapes[i].label()}")
    want_tiled = not args.no_tiled and any(shapes[i].stride == 1 for i in idx)
    if args.phase in ("build", "all"):
        if want_tiled:
            build_tiled(args.arch, args)
        for i in idx:
            build_phase(shapes[i], args.arch, args)
    if args.phase in ("run", "all"):
        tiled = []
        if want_tiled:
            with open(_tiled_cache(args), "rb") as f:
                tiled = pickle.load(f)
        rows: List[Row] = []
        guard = ForeignGuard()
        for i in idx:
            with open(_shape_cache(args, shapes[i]), "rb") as f:
                data = pickle.load(f)
            row = None
            for attempt in range(args.retries + 1):
                guard.wait_idle()
                m = guard.mark()
                row = run_phase(shapes[i], data, tiled, args.arch, args)
                hits = guard.since(m)
                if not hits:
                    break
                print(
                    f"  [guard] run overlapped foreign GPU work, discarding: {hits[:5]}",
                    flush=True,
                )
                row = None
            if row is not None:
                rows.append(row)
                print_rows([row])
                if args.rows_json:
                    with open(args.rows_json, "a") as f:
                        f.write(row.to_json() + "\n")
            if not args.keep_cache:
                os.remove(_shape_cache(args, shapes[i]))
        guard.stop()
        if guard.seen:
            print(
                f"[guard] foreign GPU sightings: {len(guard.seen)}; first/last: "
                f"{guard.seen[0]} / {guard.seen[-1]}",
                flush=True,
            )
        if rows:
            print_rows(rows)
    return 0


if __name__ == "__main__":
    random.seed(0)
    raise SystemExit(main())
