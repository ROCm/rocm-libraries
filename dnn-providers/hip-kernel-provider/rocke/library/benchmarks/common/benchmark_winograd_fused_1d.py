# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the fused 1-D Winograd conv (``conv_winograd_fused_1d_wmma``).

Two launches: K1 (generalized filter transform, ``conv_winograd_gen``) writes
U; the fused kernel goes from x to y. ``total`` counts both; ``fused`` is the
kernel alone (the cached-weights number).

Usage::

    python benchmark_winograd_fused_1d.py --miopen-cmd "convfp16 ... -y 1 -x 11" \\
        --variant 4s3 --cfgs "[dict(q=4, fk=2)]" --verify

Timing follows the measurement rules: ``--warmup 3 --iters 10``, the total
first, ``--cooldown`` seconds before each repeat. Prints ``WINOF_RESULT``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

from benchmarks.common.benchmark_winograd_conv import (  # noqa: E402
    _u8,
    make_inputs,
    reference,
    verify_error,
)
from benchmarks.common.benchmark_winograd_gen import parse_variant  # noqa: E402


def build_cfg(item):
    """Picklable worker: ``(cfg, spec, art)`` or None."""
    problem, ax, cfg, arch = item
    from rocke import compile_kernel
    from kernels.common.conv_winograd_fused_1d_wmma import (
        FusedWino1DSpec,
        build_fused_wino_1d,
    )

    try:
        try:
            spec = FusedWino1DSpec(problem=problem, ax=ax, **cfg)
        except ValueError as e:
            if "not exact in fp16" not in str(e):
                raise
            cfg = dict(cfg, xf="valu")  # B^T not fp16-exact: VALU transform
            spec = FusedWino1DSpec(problem=problem, ax=ax, **cfg)
        art = compile_kernel(build_fused_wino_1d(spec, arch=arch), arch=arch)
    except ValueError as e:
        print(f"  [skip cfg] {cfg}: {e}", flush=True)
        return None
    return cfg, spec, art


def run_case(problem, variant, cfgs, args, arch, ref_cache):
    import torch  # noqa: F401
    from rocke import compile_kernel
    from rocke.runtime import time_launches
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from kernels.common import conv_winograd_gen as WG
    from kernels.common.conv_winograd_fused_1d_wmma import fused_signature

    ay, ax = parse_variant(problem, variant)
    gspec = WG.WinogradGenSpec(problem=problem, ax_y=ay, ax_x=ax)
    st_k1 = WG.transform_stages(gspec)[0]
    k1_art = compile_kernel(st_k1.build(gspec, arch=arch), arch=arch)
    work = [(problem, ax, c, arch) for c in cfgs]
    if args.jobs > 1 and len(work) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from benchmarks.common import benchmark_winograd_fused_1d as _self

        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            built = [r for r in pool.map(_self.build_cfg, work) if r]
    else:
        built = [r for r in map(build_cfg, work) if r]
    print(
        f"\n== {problem.short()} {ax.label()} cfgs built {len(built)}/{len(cfgs)}",
        flush=True,
    )
    if not built:
        return None

    x_t, w_t, y_t = make_inputs(problem, "fp16", seed=args.seed)
    ref = None
    if args.verify:
        if problem not in ref_cache:
            ref_cache[problem] = reference(x_t, w_t, problem, "fp16")
        ref = ref_cache[problem]
    rt = Runtime()
    sizes = dict(built[0][1].buffer_bytes())
    sizes["w"] = gspec.buffer_bytes()["w"]
    assert sizes["u"] == gspec.buffer_bytes()["U"], "U layout mismatch"
    bufs = {k: int(rt.alloc(max(v, 4))) for k, v in sizes.items()}
    out = []
    try:
        rt.sync()
        for key, t in (("x", x_t), ("w", w_t)):
            h = t.contiguous().pin_memory()
            rt.memcpy_h2d(bufs[key], _u8(h), h.nbytes)
            rt.sync()
        k1 = KernelLauncher(
            hsaco=k1_art.hsaco,
            kernel_name=k1_art.kernel_name,
            signature=st_k1.signature,
        )
        k1_run = k1.bind(
            dict(w=bufs["w"], u=bufs["u"], w_bytes=sizes["w"], u_bytes=sizes["u"]),
            config=LaunchConfig(grid=st_k1.grid, block=(st_k1.block, 1, 1)),
        )
        k1_run()
        rt.sync()

        def read_y():
            import torch

            rt.sync()
            o = torch.empty_like(y_t, device="cpu").pin_memory()
            rt.memcpy_d2h(_u8(o), bufs["y"], o.nbytes)
            rt.sync()
            return o

        for cfg, spec, art in built:
            lau = KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=fused_signature(spec),
            )
            grid = (spec.grid(), 1, 1)
            if min(grid) <= 0:
                raise ValueError("empty grid")
            vals = dict(
                x=bufs["x"],
                u=bufs["u"],
                y=bufs["y"],
                x_bytes=sizes["x"],
                u_bytes=sizes["u"],
                y_bytes=sizes["y"],
            )
            fused = lau.bind(
                vals, config=LaunchConfig(grid=grid, block=(spec.waves * 32, 1, 1))
            )
            err = l2 = None
            if args.verify:
                rt.sync()
                rt.memset(bufs["y"], 0x7F, sizes["y"])
                rt.sync()
                fused()
                err, l2 = verify_error(read_y(), ref)
                if not err < 5e-2:
                    print(f"  {spec.short()} FAIL err={err:.3e}", flush=True)
                    out.append(None)
                    continue

            def total(_a=k1_run, _b=fused):
                _a()
                _b()

            for rep in range(args.repeats):
                if args.cooldown > 0:
                    time.sleep(args.cooldown)
                tot = time_launches(total, warmup=args.warmup, iters=args.iters) * 1e3
                fus = time_launches(fused, warmup=args.warmup, iters=args.iters) * 1e3
                k1us = time_launches(k1_run, warmup=args.warmup, iters=args.iters) * 1e3
                line = "\t".join(
                    [
                        "WINOF_RESULT",
                        f"shape={problem.short()}",
                        f"variant={variant}",
                        f"cfg={spec.short()}",
                        f"rep={rep}",
                        f"total_us={tot:.2f}",
                        f"fused_us={fus:.2f}",
                        f"k1_us={k1us:.2f}",
                        f"gemm_tf={spec.gemm_flops() / (fus * 1e-6) * 1e-12:.0f}",
                        f"eq_tf={problem.flops / (tot * 1e-6) * 1e-12:.0f}",
                        f"verify={'NA' if err is None else 'PASS'}",
                        f"err={err if err is not None else float('nan'):.3e}",
                        f"relL2={l2 if l2 is not None else float('nan'):.3e}",
                    ]
                )
                print(line, flush=True)
                out.append(line)
    finally:
        from rocke.runtime import synchronize_and_release

        synchronize_and_release(0)
        for ptr in bufs.values():
            rt.free(ptr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miopen-cmd")
    ap.add_argument("--miopen-file")
    ap.add_argument("--variant", default="4s3")
    ap.add_argument("--cfgs", default="[dict(q=4, fk=2)]")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args(argv)

    import torch  # noqa: F401

    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd
    from rocke.runtime.hip_module import get_device_arch

    arch = args.arch or get_device_arch(0)
    cfgs = eval(args.cfgs)  # noqa: S307 -- local benchmark CLI
    cmds: List[str] = []
    if args.miopen_cmd:
        cmds.append(args.miopen_cmd)
    if args.miopen_file:
        with open(args.miopen_file) as f:
            cmds += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    rc = 0
    ref_cache: dict = {}
    for cmd in cmds:
        problem, dtype, forw = parse_miopen_cmd(cmd)
        if forw != 1 or dtype != "fp16":
            print(f"[skip] {cmd}", flush=True)
            continue
        for variant in args.variant.split(","):
            try:
                res = run_case(problem, variant, cfgs, args, arch, ref_cache)
            except ValueError as e:
                # A requested variant that cannot run is a failure under
                # --verify (no numeric check ran), a skip otherwise.
                tag = "fail" if args.verify else "skip"
                print(f"[{tag}] {problem.short()} {variant}: {e}", flush=True)
                if args.verify:
                    rc = 1
                continue
            if not res or any(r is None for r in res):
                rc = 1
        ref_cache.clear()
    return rc


if __name__ == "__main__":
    sys.exit(main())
