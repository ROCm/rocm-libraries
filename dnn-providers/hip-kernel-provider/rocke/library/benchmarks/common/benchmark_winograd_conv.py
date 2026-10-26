# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark the unfused Winograd ``F(m x m, 3 x 3)`` forward conv pipeline.

The pipeline (see ``kernels/common/conv_winograd_fwd.py``)::

    K1 filter transform  w -> U
    K2 input transform   x -> V (+ e)
    K3 batched GEMM over the n2 * g (bin, group) pairs  V, U -> M
    K4 output transform  M (x 2**e) -> y

K1, K2 and K4 are JIT-built for the problem. K3 (``--k3``) is either

* ``wmma`` (default): the TDM + WMMA batched GEMM
  ``kernels/common/conv_winograd_gemm_wmma.py``. ``winograd_k3_wmma`` builds a
  short list of tile configs and times every launch plan (one config, or a
  main config for the whole rounds of blocks plus a second one for the rest);
  the fastest plan that verifies is kept;
* ``implicit``: the existing implicit-GEMM conv as a grouped 1x1 conv, built
  against a non-pointwise probe problem (as the AOT cache does: the JIT
  pointwise fast path ignores the group base of A and D) and launched on the
  real grouped 1x1 problem through ``ConvArgs``; a short list of tile configs
  is swept;
* ``both``: both sweeps, the faster K3 is kept.

Timing uses pre-bound launchers (packing outside the timed loop) and
``time_launches``; every stage is timed alone and the full K1..K4 chain is
timed as one closure. TFLOPS are those of the equivalent direct conv.

Usage::

    python benchmark_winograd_conv.py --miopen-cmd "convfp16 -n 8 -c 640 ..." \\
        --m 4 --verify
    python benchmark_winograd_conv.py --miopen-file shapes.txt --m both

Each result prints one tab-separated ``WINO_RESULT`` line.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

# (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_mn, pipeline, epilogue).
# Taken from the gfx1250 AOT grid (kernels.conv.aot_jobs CACHE_* axes); every
# entry is a point of that grid. gfx1250 only has 16x16 WMMA atoms, so the
# warp tile is 16.
K3_CONFIGS_SHORT: Tuple[tuple, ...] = (
    (256, 256, 64, 2, 4, 16, "mem", "cshuffle"),
    (256, 256, 64, 2, 4, 16, "wavelet", "cshuffle"),
    (256, 256, 64, 4, 4, 16, "mem", "cshuffle"),
    (256, 256, 64, 4, 4, 16, "wavelet", "cshuffle"),
    (256, 128, 64, 4, 2, 16, "mem", "cshuffle"),
    (256, 128, 64, 4, 2, 16, "wavelet", "cshuffle"),
    (128, 256, 64, 2, 4, 16, "mem", "cshuffle"),
    (128, 256, 64, 2, 4, 16, "wavelet", "cshuffle"),
    (128, 128, 64, 2, 2, 16, "mem", "cshuffle"),
    (128, 128, 64, 2, 2, 16, "wavelet", "cshuffle"),
    (256, 256, 32, 4, 4, 16, "mem", "cshuffle"),
    (256, 256, 32, 4, 4, 16, "wavelet", "cshuffle"),
    (128, 128, 32, 2, 2, 16, "mem", "cshuffle"),
    (64, 128, 64, 2, 2, 16, "mem", "cshuffle"),
    (256, 256, 64, 2, 4, 16, "mem", "default"),
    (128, 128, 64, 2, 2, 16, "mem", "default"),
)
# Small configs for tiny problems (tests): always valid, quick to compile.
K3_CONFIGS_TINY: Tuple[tuple, ...] = (
    (32, 32, 32, 1, 1, 16, "mem", "cshuffle"),
    (64, 64, 64, 2, 2, 16, "mem", "default"),
)


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _pow2_vec(n: int, cap: int = 8) -> int:
    v = cap
    while v > 1 and n % v:
        v //= 2
    return v


def resolve_prescale(mode: str, m: int, io_dtype: str) -> bool:
    """``auto``: on for bf16 I/O with F(4,3), off otherwise."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    return io_dtype == "bf16" and m == 4


def resolve_operand(mode: str, m: int, io_dtype: str) -> str:
    """``auto``: fp16 for F(4,3) and fp16 I/O; bf16 for F(2,3) with bf16 I/O."""
    if mode != "auto":
        return mode
    return "bf16" if (m == 2 and io_dtype == "bf16") else "fp16"


# ---------------------------------------------------------------------------
# K3 build (picklable worker)
# ---------------------------------------------------------------------------


def k3_spec(cfg: tuple, gemm_problem, dtype: str, arch: str):
    """``ImplicitGemmConvSpec`` for one K3 config, built on the probe problem.

    Vector widths come from the *real* grouped 1x1 problem (per-group channel
    counts), since the probe's are not the ones the binary will read.
    """
    from rocke.core.arch import ArchTarget
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec
    from kernels.conv.aot_jobs import _probe_problem

    tm, tn, tk, wm, wn, wt, pipe, epi = cfg
    target = ArchTarget.from_gfx(arch)
    family = "wmma" if target.wave_size == 32 else "mma"
    atom = target.mma.select_largest_k(
        family=family,
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=wt,
        n=wt,
        k_max=tk,
    )
    if atom is None:
        return None
    vec_ab = _pow2_vec(gemm_problem.cpg)
    # The direct-store epilogue only supports scalar stores.
    vec_c = _pow2_vec(gemm_problem.kpg) if epi != "default" else 1
    probe = _probe_problem("fwd", {"grouped": True})
    return ImplicitGemmConvSpec(
        problem=probe,
        name="rocke_wino_k3",
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_tile_m=wt,
        warp_tile_n=wt,
        warp_tile_k=atom.k,
        wave_size=target.wave_size,
        pipeline=pipe,
        epilogue=epi,
        vector_size_a=vec_ab,
        vector_size_b=vec_ab,
        vector_size_c=vec_c,
        groups=probe.groups,
    )


def k3_build_one(item):
    """``(cfg, spec, artifact)`` or None if the config is invalid."""
    cfg, gemm_problem, dtype, arch = item
    from rocke import compile_kernel
    from kernels.common.conv_implicit_gemm import (
        build_implicit_gemm_conv,
        is_valid_spec,
    )

    try:
        spec = k3_spec(cfg, gemm_problem, dtype, arch)
    except ValueError:
        return None
    if spec is None:
        return None
    ok, _ = is_valid_spec(spec, arch)
    if not ok:
        return None
    if -(-gemm_problem.M // spec.tile_m) > 65535:
        return None
    try:
        kernel = build_implicit_gemm_conv(spec, arch=arch)
        art = compile_kernel(kernel, arch=arch)
    except Exception:  # noqa: BLE001 -- an invalid config is skipped
        return None
    return cfg, spec, art


def build_k3_candidates(configs, gemm_problem, dtype: str, arch: str, jobs: int):
    work = [(cfg, gemm_problem, dtype, arch) for cfg in configs]
    if jobs == 1 or len(work) == 1:
        res = [k3_build_one(w) for w in work]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        try:
            ctx = multiprocessing.get_context("forkserver")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")
        # By package name, so the worker pickles when this file runs as __main__.
        from benchmarks.common import benchmark_winograd_conv as _self

        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            res = list(pool.map(_self.k3_build_one, work))
    order = {cfg: i for i, cfg in enumerate(configs)}
    out = [r for r in res if r is not None]
    out.sort(key=lambda r: order[r[0]])
    return out


def k3_label(cfg: tuple) -> str:
    tm, tn, tk, wm, wn, wt, pipe, epi = cfg
    return f"{tm}x{tn}x{tk}_w{wm}x{wn}_a{wt}_{pipe}_{epi}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class WinogradPipeline:
    """Device buffers + launchers of one Winograd pipeline instance."""

    spec: object  # WinogradFwdSpec
    arch: str = "gfx1250"
    rt: object = None
    bufs: Dict[str, int] = field(default_factory=dict)
    sizes: Dict[str, int] = field(default_factory=dict)
    stages: Dict[str, Tuple[object, dict, tuple, tuple]] = field(default_factory=dict)
    k3: Optional[Tuple[object, dict, tuple, tuple]] = None
    k3_cfg: Optional[tuple] = None
    # Pre-bound K3 of the WMMA batched GEMM (set_k3_run); replaces ``k3``.
    k3_run: Optional[object] = None

    @property
    def gemm_problem(self):
        from kernels.common.conv_winograd_fwd import winograd_gemm_problem

        return winograd_gemm_problem(self.spec.problem, self.spec.m)

    # -- setup --

    def compile_transforms(self):
        from rocke import compile_kernel
        from kernels.common import conv_winograd_fwd as W

        s = self.spec
        out = {}
        # K1, (K2a,) K2, K4 -- see conv_winograd_fwd.transform_stages.
        for st in W.transform_stages(s):
            art = compile_kernel(st.build(s, arch=self.arch), arch=self.arch)
            out[st.name] = (art, st.grid, st.signature, st.block)
        return out

    def alloc(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = self.rt or Runtime()
        sz = self.spec.buffer_bytes()
        if not self.spec.prescale:
            sz["e"] = 4  # never read; keep a valid pointer
        for k, nb in sz.items():
            self.sizes[k] = nb
            self.bufs[k] = int(self.rt.alloc(max(nb, 4)))

    def free(self):
        from rocke.runtime import synchronize_and_release

        synchronize_and_release(0)
        for k, ptr in list(self.bufs.items()):
            self.rt.free(ptr)
        self.bufs.clear()

    def upload(self, x_t, w_t):
        # Pinned host staging, device drained around each copy: a pageable copy
        # registers host memory through KFD's SVM path, and that page-table
        # update while one of our kernels is in flight hangs gfx1250.
        self.rt.sync()
        for key, t in (("x", x_t), ("w", w_t)):
            h = t.contiguous().pin_memory()
            self.rt.memcpy_h2d(self.bufs[key], _u8(h), h.nbytes)
            self.rt.sync()

    def poison(self, byte: int = 0x7F):
        for k in ("U", "V", "M", "e", "y"):
            self.rt.memset(self.bufs[k], byte, self.sizes[k])

    def make_launchers(self, transforms):
        from rocke.runtime.launcher import KernelLauncher

        B, S = self.bufs, self.sizes
        # Transform-kernel argument -> buffer; "<arg>_bytes" is its size.
        buf_of = {"w": "w", "u": "U", "x": "x", "v": "V", "e": "e", "mm": "M"}
        buf_of.update(m="M", y="y")
        for name, (art, grid, sig, block) in transforms.items():
            if min(grid) <= 0:
                raise ValueError(f"{name}: empty grid {grid}")
            vals = {}
            for a in sig:
                nm = a["name"]
                if nm.endswith("_bytes"):
                    vals[nm] = S[buf_of[nm[: -len("_bytes")]]]
                else:
                    vals[nm] = B[buf_of[nm]]
            lau = KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
            )
            self.stages[name] = (lau, vals, grid, (block, 1, 1))

    def set_k3(self, cfg, spec, art):
        from kernels.common.conv_abi import conv_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke.runtime.launcher import KernelLauncher

        gp = self.gemm_problem
        B, S = self.bufs, self.sizes
        lau = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_args_signature(self.spec.operand_dtype, is_3d=False),
        )
        values = ConvArgs.from_problem(
            gp, tile_m=spec.tile_m, tile_n=spec.tile_n
        ).to_launch_values(B["V"], B["U"], B["M"], S["V"], S["U"], S["M"])
        grid = (
            -(-gp.N_gemm // spec.tile_n),
            -(-gp.M // spec.tile_m),
            gp.groups,
        )
        if min(grid) <= 0 or grid[1] > 65535 or grid[2] > 65535:
            raise ValueError(f"K3 grid {grid} not launchable")
        self.k3 = (lau, values, grid, (spec.launch_block_size, 1, 1))
        self.k3_cfg = cfg
        self.k3_run = None

    def set_k3_run(self, run):
        """Use the pre-bound launch(es) ``run`` (a WMMA K3 plan) as K3."""
        self.k3_run = run

    def _pre_k3(self):
        """Transform stages before K3: K1, the split-prescale K2a, K2."""
        return [k for k in ("K1", "K2a", "K2") if k in self.stages]

    def _all(self):
        pre = [self.stages[k] for k in self._pre_k3()]
        return pre + [self.k3, self.stages["K4"]]

    # -- run --

    def run_once(self):
        from rocke.runtime.launcher import LaunchConfig

        if self.k3_run is not None:
            for k in self._pre_k3():
                lau, vals, grid, block = self.stages[k]
                lau(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
            self.k3_run()
            self.rt.sync()
            lau, vals, grid, block = self.stages["K4"]
            lau(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
            return
        for lau, vals, grid, block in self._all():
            lau(vals, config=LaunchConfig(grid=grid, block=block, fence=True))

    def bound(self, which: str):
        from rocke.runtime.launcher import LaunchConfig

        if which == "K3" and self.k3_run is not None:
            return self.k3_run
        if which == "K2" and "K2a" in self.stages:
            # The split prescale's max pass belongs to K2's time.
            fa, fb = self.bound("K2a"), self.bound("K2b")

            def k2(_a=fa, _b=fb):
                _a()
                _b()

            return k2
        if which == "K2b":
            which = "K2"
        st = self.k3 if which == "K3" else self.stages[which]
        lau, vals, grid, block = st
        return lau.bind(vals, config=LaunchConfig(grid=grid, block=block, stream=0))

    def bound_total(self):
        fns = [self.bound(k) for k in ("K1", "K2", "K3", "K4")]

        def run(_f=fns):
            for f in _f:
                f()

        return run

    def read_y(self, like):
        import torch

        self.rt.sync()
        out = torch.empty_like(like, device="cpu").pin_memory()
        self.rt.memcpy_d2h(_u8(out), self.bufs["y"], out.nbytes)
        self.rt.sync()
        return out


def verify_error(out, ref) -> Tuple[float, float]:
    """``(rocke metric, relL2)``: max|out-ref| / max(max|ref|, 1) as the other
    conv benchmarks compute it, plus the relative L2 error for context."""
    o = out.float().cpu()
    r = ref.float().cpu()
    d = (o - r).abs()
    err = float(d.max() / r.abs().max().clamp(min=1.0))
    if err != err:  # NaN
        err = float("inf")
    l2 = float((o - r).norm() / r.norm().clamp(min=1e-30))
    return err, l2


def make_inputs(problem, io_dtype: str, seed: int = 42):
    import torch

    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[io_dtype]
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(problem.N, problem.Hi, problem.Wi, problem.C).uniform_(
        -1.0, 1.0, generator=g
    )
    w = torch.empty(problem.K, problem.Y, problem.X, problem.cpg).uniform_(
        -1.0, 1.0, generator=g
    )
    y = torch.empty(problem.N, problem.Ho, problem.Wo, problem.K, dtype=tdt)
    return x.to(tdt), w.to(tdt), y


def reference(x, w, problem, io_dtype: str):
    import torch
    from builders.common import conv_reference as _cr

    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[io_dtype]
    return _cr.conv_reference(x, w, problem, out_dtype=tdt).cpu()


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def _k3_verify(pipe, ref, y_t) -> float:
    pipe.poison()
    pipe.run_once()
    err, _ = verify_error(pipe.read_y(y_t), ref)
    return err


def _k3_sweep_implicit(pipe, cands, args, ref, y_t):
    """The implicit-GEMM K3: time each candidate alone, verify the pipeline
    with it; ``(us, label, install)`` of the fastest passing one."""
    from rocke.runtime import time_launches

    best = None
    for cfg, kspec, art in cands:
        try:
            pipe.set_k3(cfg, kspec, art)
        except ValueError as e:
            print(f"  [skip] {k3_label(cfg)}: {e}", flush=True)
            continue
        err = None
        if args.verify:
            err = _k3_verify(pipe, ref, y_t)
            if not err < 5e-2:
                print(f"  K3 {k3_label(cfg):40s} FAIL err={err:.2e}", flush=True)
                continue
        us = time_launches(pipe.bound("K3"), warmup=args.warmup, iters=args.iters)
        us *= 1e3
        print(
            f"  K3 {k3_label(cfg):40s} {us:9.2f} us"
            + (f"  err={err:.2e}" if err is not None else ""),
            flush=True,
        )
        if best is None or us < best[0]:
            best = (us, cfg, kspec, art)
    if best is None:
        return None
    us, cfg, kspec, art = best
    return us, "implicit:" + k3_label(cfg), lambda: pipe.set_k3(cfg, kspec, art)


def _k3_sweep_wmma(pipe, cands, args, ref, y_t):
    """The WMMA batched-GEMM K3: time every plan (single configs and
    main + tail batch splits), verify the finalists through the pipeline;
    ``(us, label, install)`` of the fastest passing plan."""
    from rocke.runtime import time_launches
    from benchmarks.common import winograd_k3_wmma as K3W
    from kernels.common.conv_winograd_gemm_wmma import WinogradGemmProblem

    prob = WinogradGemmProblem.from_conv(pipe.spec.problem, pipe.spec.m)
    B = pipe.bufs
    plans = K3W.candidate_plans(
        prob, cands, K3W.device_cus(), tails=args.k3_tails == "on"
    )

    def verify(run):
        pipe.set_k3_run(run)
        return _k3_verify(pipe, ref, y_t)

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
    run = plan.bind(prob, B["V"], B["U"], B["M"])
    return us, "wmma:" + plan.label(), lambda: pipe.set_k3_run(run)


def run_case(problem, io_dtype: str, m: int, args, arch: str, ref_cache: dict):
    from rocke.runtime import time_launches
    from kernels.common.conv_winograd_fwd import (
        WinogradFwdSpec,
        tuned_transform_options,
    )

    prescale = resolve_prescale(args.prescale, m, io_dtype)
    operand = resolve_operand(args.operand, m, io_dtype)
    xf = tuned_transform_options(m, prescale)
    if args.vec:
        xf["vec"] = args.vec
    if args.k2_lds:
        bh, bw = (int(v) for v in args.k2_lds.lower().split("x"))
        xf["input_block_tiles"] = (bh, bw)
    if args.k2_tiles:
        xf["tiles_per_thread"] = args.k2_tiles
    if args.k2_lanes:
        xf["lds_lanes"] = args.k2_lanes
    vec = xf["vec"]
    spec = WinogradFwdSpec(
        problem=problem,
        m=m,
        io_dtype=io_dtype,
        operand_dtype=operand,
        block_size=args.block,
        prescale=prescale,
        prescale_mode=args.prescale_mode,
        nt_stores=args.nt_stores == "on",
        **xf,
    )
    tag = (
        f"F({m},3) io={io_dtype} op={operand} prescale={'on' if prescale else 'off'} "
        f"vec={vec} T={spec.num_tiles}"
    )
    print(f"\n== {problem.short()} {tag}", flush=True)
    ws = spec.workspace_bytes()
    print(
        "  workspace MiB: " + " ".join(f"{k}={v / 2**20:.1f}" for k, v in ws.items()),
        flush=True,
    )

    pipe = WinogradPipeline(spec=spec, arch=arch)
    transforms = pipe.compile_transforms()
    impls = ("wmma", "implicit") if args.k3 == "both" else (args.k3,)
    built = {}
    if "implicit" in impls:
        configs = {
            "short": K3_CONFIGS_SHORT,
            "tiny": K3_CONFIGS_TINY,
            "first": K3_CONFIGS_SHORT[:1],
        }[args.k3_sweep]
        cands = build_k3_candidates(
            configs, pipe.gemm_problem, operand, arch, jobs=args.jobs
        )
        print(f"  K3 implicit candidates built: {len(cands)}/{len(configs)}")
        built["implicit"] = cands
    if "wmma" in impls:
        from benchmarks.common import winograd_k3_wmma as K3W

        configs = {
            "short": K3W.K3W_CONFIGS,
            "tiny": K3W.K3W_CONFIGS_TINY,
            "first": K3W.K3W_CONFIGS[:1],
        }[args.k3_sweep]
        cands = K3W.build_all(configs, operand, arch, jobs=args.jobs)
        print(f"  K3 wmma candidates built: {len(cands)}/{len(configs)}", flush=True)
        built["wmma"] = cands
    if not any(built.values()):
        print("  no K3 candidate built", file=sys.stderr)
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
        # K3 sweep per implementation: (us, label, install) of the fastest
        # candidate that verifies through the whole pipeline.
        choices = []
        if built.get("implicit"):
            c = _k3_sweep_implicit(pipe, built["implicit"], args, ref, y_t)
            if c:
                choices.append(c)
        if built.get("wmma"):
            c = _k3_sweep_wmma(pipe, built["wmma"], args, ref, y_t)
            if c:
                choices.append(c)
        if not choices:
            print("  no K3 candidate passed", file=sys.stderr)
            return None
        for us, label, _ in choices:
            print(f"  K3 best {label}: {us:.2f} us", flush=True)
        _, k3_name, install = min(choices, key=lambda c: c[0])
        install()

        err = l2 = None
        if args.verify:
            pipe.poison()
            pipe.run_once()
            err, l2 = verify_error(pipe.read_y(y_t), ref)

        results = []
        for rep in range(args.repeats):
            times = {
                k: time_launches(pipe.bound(k), warmup=args.warmup, iters=args.iters)
                * 1e3
                for k in ("K1", "K2", "K3", "K4")
            }
            total = (
                time_launches(pipe.bound_total(), warmup=args.warmup, iters=args.iters)
                * 1e3
            )
            sz = pipe.sizes
            k2_bytes = sz["x"] + sz["V"] + (sz["e"] if prescale else 0)
            k4_bytes = sz["M"] + sz["y"] + (sz["e"] if prescale else 0)
            tflops = problem.flops / (total * 1e-6) * 1e-12
            status = "NA" if err is None else ("PASS" if err < 5e-2 else "FAIL")
            line = "\t".join(
                [
                    "WINO_RESULT",
                    f"shape={problem.short()}",
                    f"m={m}",
                    f"io={io_dtype}",
                    f"op={operand}",
                    f"prescale={int(prescale)}",
                    f"vec={vec}",
                    f"rep={rep}",
                    f"k1_us={times['K1']:.2f}",
                    f"k2_us={times['K2']:.2f}",
                    f"k3_us={times['K3']:.2f}",
                    f"k4_us={times['K4']:.2f}",
                    f"total_us={total:.2f}",
                    f"tflops={tflops:.1f}",
                    f"k2_gbps={k2_bytes / (times['K2'] * 1e-6) * 1e-9:.0f}",
                    f"k4_gbps={k4_bytes / (times['K4'] * 1e-6) * 1e-9:.0f}",
                    f"k3={k3_name}",
                    f"verify={status}",
                    f"err={err if err is not None else float('nan'):.3e}",
                    f"relL2={l2 if l2 is not None else float('nan'):.3e}",
                ]
            )
            print(line, flush=True)
            results.append(line)
        return results
    finally:
        pipe.free()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miopen-cmd", help="one MIOpenDriver command line")
    ap.add_argument("--miopen-file", help="file with one MIOpenDriver command per line")
    ap.add_argument("--m", default="4", choices=("2", "4", "both"))
    ap.add_argument("--prescale", default="auto", choices=("on", "off", "auto"))
    ap.add_argument(
        "--operand",
        default="auto",
        choices=("auto", "fp16", "bf16"),
        help="U/V/M dtype; auto = bf16 for F(2,3) with bf16 I/O, else fp16",
    )
    ap.add_argument("--vec", type=int, default=0, help="0 = tuned (8 for F2, 4 for F4)")
    ap.add_argument(
        "--k2-lds",
        default="",
        help="K2 LDS region in tiles, e.g. 4x8; 0x0 = register path; '' = tuned",
    )
    ap.add_argument("--k2-tiles", type=int, default=0, help="K2 tiles per thread")
    ap.add_argument("--k2-lanes", type=int, default=0, help="K2 LDS lanes per tile")
    ap.add_argument("--prescale-mode", default="fused", choices=("fused", "split"))
    ap.add_argument(
        "--nt-stores",
        default="on",
        choices=("on", "off"),
        help="streaming (TH_STORE_NT) V / y stores",
    )
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument(
        "--k3",
        default="wmma",
        choices=("wmma", "implicit", "both"),
        help="K3 kernel: the WMMA batched GEMM, the implicit-GEMM conv, or the "
        "faster of both",
    )
    ap.add_argument("--k3-sweep", default="short", choices=("short", "tiny", "first"))
    ap.add_argument(
        "--k3-tails",
        default="on",
        choices=("on", "off"),
        help="also try WMMA plans that run the last partial round of blocks "
        "with a second tile config",
    )
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
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

    ms = (2, 4) if args.m == "both" else (int(args.m),)
    rc = 0
    ref_cache: dict = {}
    for cmd in cmds:
        problem, dtype, forw = parse_miopen_cmd(cmd)
        if forw != 1:
            print(f"[skip] not a forward command: {cmd}", flush=True)
            continue
        if dtype not in ("fp16", "bf16"):
            print(f"[skip] dtype {dtype} unsupported: {cmd}", flush=True)
            continue
        for m in ms:
            try:
                res = run_case(problem, dtype, m, args, arch, ref_cache)
            except ValueError as e:
                print(f"[skip] {problem.short()} F({m},3): {e}", flush=True)
                continue
            if not res or any("verify=FAIL" in r for r in res):
                rc = 1
        ref_cache.clear()
    return rc


if __name__ == "__main__":
    sys.exit(main())
