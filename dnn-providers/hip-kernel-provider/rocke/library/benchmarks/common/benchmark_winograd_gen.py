# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the generalized unfused Winograd pipeline (any filter extent).

K1 / K2 / K4 come from ``kernels/common/conv_winograd_gen.py`` (``F(my x mx,
ry x rx)``, selectable points, segmented long filters along W); K3 is the
TDM + WMMA batched GEMM (``conv_winograd_gemm_wmma``) with ``taps = S``.

Usage::

    python benchmark_winograd_gen.py --miopen-cmd "convfp16 ... -y 1 -x 11 ..." \\
        --variant 4s3 --verify

``--variant``: comma list of ``<mx>[s<S>]`` (W axis; ``S`` segments of
``F(mx, mx)``) optionally ``<my>x<mx>`` for 2-D filters; e.g. ``4s3``,
``6s2``, ``4``, ``3x3``. Each variant prints one ``WINOG_RESULT`` line per
repeat.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

from benchmarks.common.benchmark_winograd_conv import (  # noqa: E402
    WinogradPipeline,
    make_inputs,
    reference,
    verify_error,
)


def parse_variant(problem, text: str):
    """``(ax_y, ax_x)`` for a variant string (see the module doc)."""
    from kernels.common.conv_winograd_gen import WinoAxis

    my = 0
    if "x" in text:
        a, text = text.split("x", 1)
        my = int(a)
    seg = 1
    if "s" in text:
        text, s = text.split("s", 1)
        seg = int(s)
    mx = int(text)
    ay = WinoAxis(1, 1) if problem.Y == 1 else WinoAxis(my or mx, problem.Y)
    if problem.X == 1:
        ax = WinoAxis(1, 1)
        if problem.Y > 1 and not my:
            ay = WinoAxis(mx, problem.Y)
    else:
        ax = WinoAxis(mx, problem.X, seg)
    return ay, ax


def small_channel_configs(cg: int, kg: int) -> list:
    """K3 tiles for Cg / Kg < 128: tile_k matched to Cg (a wider chunk only
    multiplies zero-filled channels), N tile matched to Kg, more tile rows."""
    if cg >= 128 and kg >= 128:
        return []
    tk = 32 if cg <= 32 else 64
    out = []
    for wm, wn in ((4, 1), (8, 1), (2, 2), (4, 2)):
        fn = max(1, min(4, kg // (16 * wn)))
        for fm in (4, 8):
            for st in (2, 3):
                out.append(
                    dict(
                        waves_m=wm,
                        waves_n=wn,
                        frags_m=fm,
                        frags_n=fn,
                        tile_k=tk,
                        stages=st,
                    )
                )
    return out


class GenPipeline(WinogradPipeline):
    """``WinogradPipeline`` with the generalized transforms."""

    def compile_transforms(self):
        from rocke import compile_kernel
        from kernels.common import conv_winograd_gen as WG

        out = {}
        for st in WG.transform_stages(self.spec):
            art = compile_kernel(st.build(self.spec, arch=self.arch), arch=self.arch)
            out[st.name] = (art, st.grid, st.signature, st.block)
        return out


def k3_sweep(pipe, prob, cands, args, ref, y_t):
    from rocke.runtime import time_launches
    from benchmarks.common import winograd_k3_wmma as K3W

    B = pipe.bufs
    plans = K3W.candidate_plans(prob, cands, K3W.device_cus(), tails=True)

    def verify(run):
        pipe.set_k3_run(run)
        pipe.poison()
        pipe.run_once()
        err, _ = verify_error(pipe.read_y(y_t), ref)
        return err

    def time_fn(run):
        return time_launches(run, warmup=args.warmup, iters=args.iters) * 1e3

    best = K3W.sweep(
        prob,
        plans,
        lambda p: p.bind(prob, B["V"], B["U"], B["M"]),
        verify=verify if args.verify else None,
        time_fn=time_fn,
        log=lambda m: print(m, flush=True),
    )
    if best is None:
        return None
    plan, us, _ = best
    return us, plan.label(), plan.bind(prob, B["V"], B["U"], B["M"])


def run_case(problem, io_dtype, variant, args, arch, ref_cache):
    from rocke.runtime import time_launches
    from benchmarks.common import winograd_k3_wmma as K3W
    from kernels.common.conv_winograd_gen import WinogradGenSpec

    ay, ax = parse_variant(problem, variant)
    spec = WinogradGenSpec(
        problem=problem,
        ax_y=ay,
        ax_x=ax,
        io_dtype=io_dtype,
        vec=args.vec,
        block_size=args.block,
    )
    prob = spec.gemm_problem()
    tag = f"{ay.label()} {ax.label()} bins={spec.bins} S={spec.S} T={spec.num_tiles}"
    print(f"\n== {problem.short()} {tag}", flush=True)
    sz = spec.buffer_bytes()
    print(
        "  MiB: " + " ".join(f"{k}={v / 2**20:.1f}" for k, v in sz.items()), flush=True
    )
    # Host-side bounds of every K3 TDM access: V rows m0 + tap .. < T + S - 1,
    # U slabs (g * S + tap) < G * S, M rows < T (the descriptors clip to these).
    need = {
        "V": (prob.T + spec.S - 1) * prob.G * prob.Cg * 2,
        "U": prob.G * spec.S * prob.Kg * prob.Cg * 2,
        "M": prob.T * prob.G * prob.Kg * 2,
    }
    for k, nb in need.items():
        if sz[k] < nb:
            raise ValueError(f"{k} buffer {sz[k]} B < {nb} B needed by K3")
    pipe = GenPipeline(spec=spec, arch=arch)
    transforms = pipe.compile_transforms()
    configs = {"short": K3W.K3W_CONFIGS, "tiny": K3W.K3W_CONFIGS_TINY}[args.k3_sweep]
    configs = list(configs)
    if args.k3_sweep == "short":
        configs += small_channel_configs(prob.Cg, prob.Kg)
    configs = [dict(c, taps=spec.S) for c in configs]
    cands = K3W.build_all(configs, "fp16", arch, jobs=args.jobs)
    print(f"  K3 candidates built: {len(cands)}/{len(configs)}", flush=True)
    if not cands:
        return None
    x_t, w_t, y_t = make_inputs(problem, io_dtype, seed=args.seed)
    ref = None
    if args.verify:
        key = (problem, io_dtype)
        if key not in ref_cache:
            ref_cache[key] = reference(x_t, w_t, problem, io_dtype)
        ref = ref_cache[key]
    pipe.alloc()
    try:
        pipe.upload(x_t, w_t)
        pipe.make_launchers(transforms)
        best = k3_sweep(pipe, prob, cands, args, ref, y_t)
        if best is None:
            print("  no K3 plan passed", file=sys.stderr)
            return None
        k3_us, k3_label, run = best
        pipe.set_k3_run(run)
        err = l2 = None
        if args.verify:
            pipe.poison()
            pipe.run_once()
            err, l2 = verify_error(pipe.read_y(y_t), ref)
        out = []
        for rep in range(args.repeats):
            # Rule 9: cool down before each repeat, total first, short loops.
            if args.cooldown > 0:
                import time as _time

                _time.sleep(args.cooldown)
            total = (
                time_launches(pipe.bound_total(), warmup=args.warmup, iters=args.iters)
                * 1e3
            )
            times = {
                k: time_launches(pipe.bound(k), warmup=args.warmup, iters=args.iters)
                * 1e3
                for k in ("K1", "K2", "K3", "K4")
            }
            status = "NA" if err is None else ("PASS" if err < 5e-2 else "FAIL")
            gflops = spec.gemm_flops()
            line = "\t".join(
                [
                    "WINOG_RESULT",
                    f"shape={problem.short()}",
                    f"variant={variant}",
                    f"ax={ay.label()}/{ax.label()}",
                    f"rep={rep}",
                    f"k1_us={times['K1']:.2f}",
                    f"k2_us={times['K2']:.2f}",
                    f"k3_us={times['K3']:.2f}",
                    f"k4_us={times['K4']:.2f}",
                    f"total_us={total:.2f}",
                    f"k3_tf={gflops / (times['K3'] * 1e-6) * 1e-12:.0f}",
                    f"eq_tf={problem.flops / (total * 1e-6) * 1e-12:.0f}",
                    f"k3={k3_label}",
                    f"verify={status}",
                    f"err={err if err is not None else float('nan'):.3e}",
                    f"relL2={l2 if l2 is not None else float('nan'):.3e}",
                ]
            )
            print(line, flush=True)
            out.append(line)
        return out
    finally:
        pipe.free()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miopen-cmd")
    ap.add_argument("--miopen-file")
    ap.add_argument("--variant", default="4s3")
    ap.add_argument("--vec", type=int, default=4)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--k3-sweep", default="short", choices=("short", "tiny"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cooldown", type=float, default=15.0, help="s before each repeat")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args(argv)

    import torch  # noqa: F401 -- claim the HIP context first

    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd
    from rocke.runtime.hip_module import get_device_arch

    arch = args.arch or get_device_arch(0)
    cmds: List[str] = []
    if args.miopen_cmd:
        cmds.append(args.miopen_cmd)
    if args.miopen_file:
        with open(args.miopen_file) as f:
            cmds += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not cmds:
        ap.error("give --miopen-cmd or --miopen-file")
    rc = 0
    ref_cache: dict = {}
    for cmd in cmds:
        problem, dtype, forw = parse_miopen_cmd(cmd)
        if forw != 1 or dtype not in ("fp16", "bf16"):
            print(f"[skip] {cmd}", flush=True)
            continue
        for variant in args.variant.split(","):
            try:
                res = run_case(problem, dtype, variant, args, arch, ref_cache)
            except ValueError as e:
                # A requested variant that cannot run is a failure under
                # --verify (no numeric check ran), a skip otherwise.
                tag = "fail" if args.verify else "skip"
                print(f"[{tag}] {problem.short()} {variant}: {e}", flush=True)
                if args.verify:
                    rc = 1
                continue
            if not res or any("verify=FAIL" in r for r in res):
                rc = 1
        ref_cache.clear()
    return rc


if __name__ == "__main__":
    sys.exit(main())
