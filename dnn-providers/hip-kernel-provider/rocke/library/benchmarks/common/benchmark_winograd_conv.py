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
import time
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


def _m_pair(m) -> Tuple[int, int]:
    """``4`` / ``"4"`` / ``"6x4"`` / ``(6, 4)`` -> ``(mh, mw)``."""
    if isinstance(m, (tuple, list)):
        return int(m[0]), int(m[1])
    t = str(m).lower().split("x")
    return (int(t[0]), int(t[-1]))


def m_label(m) -> str:
    mh, mw = _m_pair(m)
    return str(mh) if mh == mw else f"{mh}x{mw}"


def resolve_prescale(mode: str, m: int, io_dtype: str) -> bool:
    """``auto``: on for bf16 I/O with F(4,3), off otherwise."""
    if mode == "on":
        return True
    if mode == "off":
        return False
    return io_dtype == "bf16" and max(_m_pair(m)) >= 4


def resolve_operand(mode: str, m: int, io_dtype: str) -> str:
    """``auto``: fp16 for F(4,3) and fp16 I/O; bf16 for F(2,3) with bf16 I/O."""
    if mode != "auto":
        return mode
    return "bf16" if (_m_pair(m) == (2, 2) and io_dtype == "bf16") else "fp16"


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
    """Device buffers + launchers of one Winograd pipeline instance.

    ``spec.problem`` is one *batch chunk*: the conv is run as ``chunks``
    pipelines over consecutive N-ranges of x / y, sharing U (K1 runs once)
    and the V / M / e workspaces (the chunks run in stream order). That keeps
    every buffer below the 2 GiB the 32-bit buffer offsets address (batch
    split, ``--batch-split``). x and y are one buffer per chunk.

    ``merge`` names transform stages launched as one kernel
    (:func:`conv_winograd_fwd.build_winograd_merged`), e.g. ``("K2", "K1")``:
    the first chunk's input transform and the filter transform share a grid.
    """

    spec: object  # WinogradFwdSpec (of one chunk)
    arch: str = "gfx1250"
    rt: object = None
    chunks: int = 1
    merge: Tuple[str, ...] = ()
    bufs: Dict[str, int] = field(default_factory=dict)
    sizes: Dict[str, int] = field(default_factory=dict)
    xs: List[int] = field(default_factory=list)
    ys: List[int] = field(default_factory=list)
    # name -> (launcher, values of chunk 0, grid, block)
    stages: Dict[str, Tuple[object, dict, tuple, tuple]] = field(default_factory=dict)
    # name -> per-chunk values (stages that read x or write y)
    chunk_vals: Dict[str, List[dict]] = field(default_factory=dict)
    k3: Optional[Tuple[object, dict, tuple, tuple]] = None
    k3_cfg: Optional[tuple] = None
    # Pre-bound K3 of the WMMA batched GEMM (set_k3_run); replaces ``k3``.
    k3_run: Optional[object] = None
    # (plan, problem) of the WMMA K3, to re-bind it on another stream.
    k3_plan: Optional[tuple] = None

    @property
    def gemm_problem(self):
        from kernels.common.conv_winograd_fwd import winograd_gemm_problem

        return winograd_gemm_problem(self.spec.problem, self.spec.tile_mn)

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
        if self.merge:
            lay = W.merged_stage(s, self.merge)
            k = W.build_winograd_merged(
                W.WinogradMergedSpec(s, tuple(self.merge)), arch=self.arch
            )
            art = compile_kernel(k, arch=self.arch)
            out["MERGED"] = (art, lay.grid, lay.signature, lay.block)
        return out

    def alloc(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = self.rt or Runtime()
        sz = self.spec.buffer_bytes()
        if not self.spec.prescale:
            sz["e"] = 4  # never read; keep a valid pointer
        for k, nb in sz.items():
            self.sizes[k] = nb
            if k in ("x", "y"):
                lst = [int(self.rt.alloc(max(nb, 4))) for _ in range(self.chunks)]
                (self.xs if k == "x" else self.ys)[:] = lst
                self.bufs[k] = lst[0]
            else:
                self.bufs[k] = int(self.rt.alloc(max(nb, 4)))

    def free(self):
        from rocke.runtime import synchronize_and_release

        synchronize_and_release(0)
        for k, ptr in list(self.bufs.items()):
            if k not in ("x", "y"):
                self.rt.free(ptr)
        for ptr in self.xs[1:] + self.ys[1:]:
            self.rt.free(ptr)
        for ptr in self.xs[:1] + self.ys[:1]:
            self.rt.free(ptr)
        self.bufs.clear()
        self.xs.clear()
        self.ys.clear()

    def _nc(self) -> int:
        return self.spec.problem.N

    def upload(self, x_t, w_t):
        # Pinned host staging, device drained around each copy: a pageable copy
        # registers host memory through KFD's SVM path, and that page-table
        # update while one of our kernels is in flight hangs gfx1250.
        self.rt.sync()
        nc = self._nc()
        items = [(self.xs[i], x_t[i * nc : (i + 1) * nc]) for i in range(self.chunks)]
        items.append((self.bufs["w"], w_t))
        for ptr, t in items:
            h = t.contiguous().pin_memory()
            self.rt.memcpy_h2d(ptr, _u8(h), h.nbytes)
            self.rt.sync()
            del h

    def poison(self, byte: int = 0x7F):
        for k in ("U", "V", "M", "e"):
            self.rt.memset(self.bufs[k], byte, self.sizes[k])
        for ptr in self.ys:
            self.rt.memset(ptr, byte, self.sizes["y"])

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
            per_chunk = False
            for a in sig:
                nm = a["name"]
                if nm.endswith("_bytes"):
                    vals[nm] = S[buf_of[nm[: -len("_bytes")]]]
                else:
                    vals[nm] = B[buf_of[nm]]
                    per_chunk |= buf_of[nm] in ("x", "y")
            lau = KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
            )
            self.stages[name] = (lau, vals, grid, (block, 1, 1))
            if per_chunk:
                cv = []
                for i in range(self.chunks):
                    d = dict(vals)
                    if "x" in d:
                        d["x"] = self.xs[i]
                    if "y" in d:
                        d["y"] = self.ys[i]
                    cv.append(d)
                self.chunk_vals[name] = cv

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
        self.k3_plan = None

    def set_k3_run(self, run, plan=None):
        """Use the pre-bound launch(es) ``run`` (a WMMA K3 plan) as K3;
        ``plan = (K3Plan, WinogradGemmProblem)`` allows re-binding it."""
        self.k3_run = run
        self.k3_plan = plan

    # -- launch sequences --

    def _pre_k3(self):
        """Transform stages before K3: K1, the split-prescale K2a, K2."""
        return [k for k in ("K1", "K2a", "K2") if k in self.stages]

    def sequence(self, *, infer: bool = False, merged: bool = True) -> list:
        """The launch order as ``(stage name, chunk)`` items; ``"K3"`` is the
        GEMM. ``infer``: U is pre-transformed (no K1). ``merged``: use the
        merged kernel (if built) for the stages in ``self.merge`` of the
        first chunk."""
        pre = [k for k in self._pre_k3() if k != "K1"]
        use_merge = merged and not infer and "MERGED" in self.stages
        seq = []
        if not infer and not use_merge:
            seq.append(("K1", 0))
        for i in range(self.chunks):
            for k in pre:
                if use_merge and i == 0 and k in self.merge:
                    if ("MERGED", 0) not in seq:
                        seq.append(("MERGED", 0))
                    continue
                seq.append((k, i))
            seq += [("K3", i), ("K4", i)]
        return seq

    def _stage_vals(self, name: str, chunk: int):
        lau, vals, grid, block = self.stages[name]
        if name in self.chunk_vals:
            vals = self.chunk_vals[name][chunk]
        return lau, vals, grid, block

    def run_once(self):
        from rocke.runtime.launcher import LaunchConfig

        for name, i in self.sequence():
            if name == "K3":
                if self.k3_run is not None:
                    self.k3_run()
                    self.rt.sync()
                else:
                    lau, vals, grid, block = self.k3
                    lau(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
                continue
            lau, vals, grid, block = self._stage_vals(name, i)
            lau(vals, config=LaunchConfig(grid=grid, block=block, fence=True))

    def bound(self, which: str, chunk: int = 0, stream: int = 0):
        from rocke.runtime.launcher import LaunchConfig

        if which == "K3" and self.k3_run is not None:
            if stream == 0:
                return self.k3_run
            plan, prob = self.k3_plan
            B = self.bufs
            return plan.bind(prob, B["V"], B["U"], B["M"], stream=stream)
        if which == "K2" and "K2a" in self.stages:
            # The split prescale's max pass belongs to K2's time.
            fa = self.bound("K2a", chunk, stream)
            fb = self.bound("K2b", chunk, stream)

            def k2(_a=fa, _b=fb):
                _a()
                _b()

            return k2
        if which == "K2b":
            which = "K2"
        if which == "K3":
            lau, vals, grid, block = self.k3
        else:
            lau, vals, grid, block = self._stage_vals(which, chunk)
        return lau.bind(
            vals, config=LaunchConfig(grid=grid, block=block, stream=stream)
        )

    def bound_seq(self, seq, stream: int = 0):
        fns = [self.bound(k, i, stream) for k, i in seq]

        def run(_f=tuple(fns)):
            for f in _f:
                f()

        return run

    def bound_total(self, stream: int = 0, merged: bool = True):
        return self.bound_seq(self.sequence(merged=merged), stream)

    def bound_infer(self, stream: int = 0):
        """K2..K4 only: the weights-pretransformed (inference) mode."""
        return self.bound_seq(self.sequence(infer=True), stream)

    def bound_two_streams(self, stream: int = 0):
        """K1 on a second stream, overlapping K2 of the first chunk; K3 waits
        for it. Each call starts K1 only after the previous call's work on
        the main stream (a fork / join per call), so calls do not overlap.
        ``stream``: main stream handle (0: torch's current stream); usable
        inside a HIP graph capture on that stream."""
        import torch

        if stream:
            s1 = torch.cuda.ExternalStream(stream)
        else:
            s1 = torch.cuda.current_stream()
        s2 = torch.cuda.Stream()
        k1 = self.bound("K1", 0, s2.cuda_stream)
        rest = []
        for name, i in self.sequence(infer=True):
            rest.append((name, self.bound(name, i, stream)))
        ev_fork = torch.cuda.Event()
        ev_u = torch.cuda.Event()

        def run():
            ev_fork.record(s1)
            s2.wait_event(ev_fork)
            k1()
            ev_u.record(s2)
            joined = False
            for name, f in rest:
                if name == "K3" and not joined:
                    s1.wait_event(ev_u)
                    joined = True
                f()

        return run

    def read_y(self, like):
        import torch

        self.rt.sync()
        nc = self._nc()
        out = torch.empty_like(like, device="cpu")
        for i in range(self.chunks):
            h = torch.empty_like(like[i * nc : (i + 1) * nc], device="cpu")
            h = h.pin_memory()
            self.rt.memcpy_d2h(_u8(h), self.ys[i], h.nbytes)
            self.rt.sync()
            out[i * nc : (i + 1) * nc] = h
        return out


def time_graph(fn, *, warmup: int, iters: int) -> float:
    """Microseconds per call of ``fn(stream)`` captured into one HIP graph
    (torch.cuda.CUDAGraph) and replayed: the GPU-side time without the host
    launch cost. ``fn`` builds the launches bound to ``stream``."""
    import torch

    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    run = fn(s.cuda_stream)
    with torch.cuda.stream(s):
        run()  # warm (module loads) outside the capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        run()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        for _ in range(warmup):
            g.replay()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(s)
        for _ in range(iters):
            g.replay()
        e1.record(s)
    e1.synchronize()
    us = e0.elapsed_time(e1) * 1e3 / iters
    torch.cuda.synchronize()
    del g
    return us


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

    prob = WinogradGemmProblem.from_conv(pipe.spec.problem, pipe.spec.tile_mn)
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
    return us, "wmma:" + plan.label(), lambda: pipe.set_k3_run(run, (plan, prob))


def batch_chunks(problem, m: int, mode: str, **spec_kw) -> int:
    """Number of N-chunks of the batch split: ``mode`` is ``auto`` (the
    fewest chunks, a divisor of N, whose chunk pipeline keeps every buffer
    and GEMM operand within the 32-bit offsets) or an explicit count."""
    import dataclasses

    from kernels.common.conv_winograd_fwd import WinogradFwdSpec
    from kernels.common.conv_winograd_gemm_wmma import (
        WinogradGemmProblem,
        WinogradGemmWmmaSpec,
        winograd_gemm_problem_reason,
    )

    N = problem.N
    if mode != "auto":
        c = int(mode)
        if c < 1 or N % c:
            raise ValueError(f"--batch-split {c} must divide N={N}")
        return c
    for c in range(1, N + 1):
        if N % c:
            continue
        sub = dataclasses.replace(problem, N=N // c)
        try:
            WinogradFwdSpec(problem=sub, m=m[0], m_w=m[1], **spec_kw)
        except ValueError:
            continue
        prob = WinogradGemmProblem.from_conv(sub, m)
        if winograd_gemm_problem_reason(WinogradGemmWmmaSpec(), prob) is None:
            return c
    raise ValueError(f"no batch split of N={N} fits the 32-bit offsets")


def run_case(problem, io_dtype: str, m, args, arch: str, ref_cache: dict):
    """One pipeline; ``m`` is the tile ``(mh, mw)`` (or an int, square)."""
    import dataclasses

    from rocke.runtime import time_launches
    from kernels.common.conv_winograd_fwd import (
        WinogradFwdSpec,
        tuned_transform_options,
    )

    m = _m_pair(m)
    prescale = resolve_prescale(args.prescale, m, io_dtype)
    operand = resolve_operand(args.operand, m, io_dtype)
    xf = tuned_transform_options(m, prescale, problem)
    if args.vec:
        xf["vec"] = args.vec
    if args.k2_lds:
        bh, bw = (int(v) for v in args.k2_lds.lower().split("x"))
        xf["input_block_tiles"] = (bh, bw)
    if args.k2_tiles:
        xf["tiles_per_thread"] = args.k2_tiles
    if args.k2_lanes:
        xf["lds_lanes"] = args.k2_lanes
    if args.k1_vec:
        xf["filter_vec"] = args.k1_vec
    if args.k1_rows:
        xf["filter_rows"] = (m[0] + 2) if args.k1_rows == "all" else int(args.k1_rows)
    vec = xf["vec"]
    spec_kw = dict(
        io_dtype=io_dtype,
        operand_dtype=operand,
        block_size=args.block,
        prescale=prescale,
        prescale_mode=args.prescale_mode,
        nt_stores=args.nt_stores == "on",
        **xf,
    )
    chunks = batch_chunks(problem, m, args.batch_split, **spec_kw)
    sub = dataclasses.replace(problem, N=problem.N // chunks)
    spec = WinogradFwdSpec(problem=sub, m=m[0], m_w=m[1], **spec_kw)
    tag = (
        f"F({m_label(m)},3) io={io_dtype} op={operand} prescale={'on' if prescale else 'off'} "
        f"vec={vec} T={spec.num_tiles * chunks}"
        + (f" chunks={chunks}x(N={sub.N}, T={spec.num_tiles})" if chunks > 1 else "")
    )
    print(f"\n== {problem.short()} {tag}", flush=True)
    ws = spec.workspace_bytes()
    print(
        "  workspace MiB: " + " ".join(f"{k}={v / 2**20:.1f}" for k, v in ws.items()),
        flush=True,
    )

    merge: Tuple[str, ...] = ()
    merge_arg = args.merge
    if merge_arg == "auto":
        from kernels.common.conv_winograd_fwd import merge_worthwhile

        merge_arg = "K2,K1" if merge_worthwhile(spec) else "off"
    if merge_arg != "off":
        names = [s.strip() for s in merge_arg.split(",")]
        first = "K2a" if (prescale and spec.prescale_split) else "K2"
        merge = tuple(first if s == "K2" else s for s in names)
    pipe = WinogradPipeline(spec=spec, arch=arch, chunks=chunks, merge=merge)
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

        def timed(fn):
            return time_launches(fn, warmup=args.warmup, iters=args.iters) * 1e3

        results = []
        for rep in range(args.repeats):
            if args.cooldown > 0:
                # Long timing loops throttle the GPU (power / sclk); let it
                # cool before each repeat.
                time.sleep(args.cooldown)
            # The total first (it is the number that counts), then the
            # variants, then the per-stage times (one chunk's launch, times
            # the chunk count).
            total = timed(pipe.bound_total())
            extra = {}
            if merge:
                extra["total_unmerged_us"] = timed(pipe.bound_total(merged=False))
                extra["merged_us"] = timed(pipe.bound("MERGED"))
            extra["infer_us"] = timed(pipe.bound_infer())
            if args.two_streams == "on":
                extra["total_2s_us"] = timed(pipe.bound_two_streams())
            if args.graph == "on":
                gkw = dict(warmup=args.warmup, iters=args.iters)
                extra["total_graph_us"] = time_graph(
                    lambda s: pipe.bound_total(stream=s), **gkw
                )
                extra["infer_graph_us"] = time_graph(
                    lambda s: pipe.bound_infer(stream=s), **gkw
                )
                if args.two_streams == "on":
                    extra["total_2s_graph_us"] = time_graph(
                        lambda s: pipe.bound_two_streams(stream=s), **gkw
                    )
            times = {
                k: timed(pipe.bound(k)) * (1 if k == "K1" else chunks)
                for k in ("K1", "K2", "K3", "K4")
            }
            sz = pipe.sizes
            k2_bytes = (sz["x"] + sz["V"] + (sz["e"] if prescale else 0)) * chunks
            k4_bytes = (sz["M"] + sz["y"] + (sz["e"] if prescale else 0)) * chunks
            tflops = problem.flops / (total * 1e-6) * 1e-12
            status = "NA" if err is None else ("PASS" if err < 5e-2 else "FAIL")
            line = "\t".join(
                [
                    "WINO_RESULT",
                    f"shape={problem.short()}",
                    f"m={m_label(m)}",
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
                    f"chunks={chunks}",
                    f"merge={'+'.join(merge) or 'off'}",
                ]
                + [f"{k}={v:.2f}" for k, v in extra.items()]
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
    ap.add_argument(
        "--m",
        default="4",
        help="tile sizes, comma-separated: 2, 4, 6 or mixed HxW (e.g. 6x4, 4x6); "
        "'both' = 2,4; 'auto' / 'autoN' = the N (default 2) tiles the cost "
        "model ranks best (each is measured, the fastest verified one counts)",
    )
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
    ap.add_argument("--k1-vec", type=int, default=0, help="K1 vector width (0 = vec)")
    ap.add_argument(
        "--k1-rows",
        default="",
        help="K1 rows of G g per thread: 1, a divisor of n, or 'all' ('' = tuned)",
    )
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
    ap.add_argument(
        "--batch-split",
        default="auto",
        help="N-chunks run as separate pipelines (U shared): 'auto' = the "
        "fewest that keep every buffer < 2 GiB, or a divisor of N",
    )
    ap.add_argument(
        "--merge",
        default="off",
        help="transform stages launched as one kernel, in block order, e.g. "
        "'K2,K1' (K2 = the first stage of the input transform); 'off'; 'auto' "
        "= K2,K1 unless K2's LDS would starve K1's blocks of occupancy",
    )
    ap.add_argument(
        "--graph",
        default="off",
        choices=("on", "off"),
        help="also time the chain (and the inference chain) replayed as a HIP graph",
    )
    ap.add_argument(
        "--two-streams",
        default="off",
        choices=("on", "off"),
        help="also time K1 on a second stream, overlapping K2",
    )
    ap.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to sleep before each timing repeat (GPU throttling)",
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

    ms_arg = "2,4" if args.m == "both" else args.m
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
        ms = []
        for tok in (t.strip() for t in ms_arg.split(",")):
            if not tok:
                continue
            if tok.startswith("auto"):
                from benchmarks.common.winograd_k3_wmma import device_cus
                from kernels.common.conv_winograd_fwd import auto_tiles

                got = auto_tiles(
                    problem, dtype, count=int(tok[4:] or 2), num_cus=device_cus()
                )
                print(f"auto tiles: {[m_label(m) for m in got]}", flush=True)
            else:
                got = [_m_pair(tok)]
            ms += [m for m in got if m not in ms]
        for m in ms:
            try:
                res = run_case(problem, dtype, m, args, arch, ref_cache)
            except ValueError as e:
                print(f"[skip] {problem.short()} F({m_label(m)},3): {e}", flush=True)
                continue
            if not res or any("verify=FAIL" in r for r in res):
                rc = 1
        ref_cache.clear()
    return rc


if __name__ == "__main__":
    sys.exit(main())
