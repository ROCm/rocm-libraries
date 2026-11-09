# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""AOT Winograd ``F(m x m, 3 x 3)`` forward: compile once, run every shape from the cache.

``benchmark_winograd_conv.py`` builds the transform kernels for each problem
(the extents are baked in). This driver uses the shape-generic transforms of
:mod:`kernels.common.conv_winograd_fwd_rt` and the K3 GEMMs (the WMMA batched
GEMM and the implicit-GEMM conv on the probe problem, both shape-generic
already), so the whole pipeline is a set of cached binaries:

* ``--compile-all --cache-dir DIR``: build every binary of the grid below
  into ``DIR/<arch>/wino_rt`` (CPU only, ``--jobs`` workers);
* ``--run-from-cache DIR``: for each 3x3 stride-1 forward case (from
  ``--miopen-file``, ``--miopen-cmd`` or the shape flags; the other cases are
  skipped with the reason) and each tile (all of
  :data:`conv_winograd_fwd_rt.AOT_TILES`), pick each stage's fastest fitting
  binary by timing it alone (K3: every plan, as ``benchmark_winograd_conv.py``
  sweeps them), verify the assembled pipeline against the reference conv, and
  time the whole chain as one call (``--timing``, best of ``--repeats``).

The cache run takes the inputs and writes the rows of the other AOT conv
sweeps: ``--results`` writes one :mod:`benchmarks.common.case_results` row per
case (engine ``winograd``, the fastest tile's pipeline, the next tiles as
runners-up), ``--skip-measured`` resumes, and the ``Best:`` line is the one
``benchmark_conv_compare.py`` reads -- the comparison runs this sweep beside
the direct-conv and implicit-GEMM ones, and ``run_conv_shards.py --script
winograd`` shards it.

The prescale and operand dtype follow ``benchmark_winograd_conv.py``'s
``auto`` rules (bf16 operands only for F(2,3) with bf16 I/O, also tried with
fp16 operands; prescale for bf16 I/O with a tile side >= 4).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("ROCKE_CPP_QUIET_FALLBACK", "1")

VECS = (8, 4, 2, 1)
IOS = ("fp16", "bf16")


def _tiles():
    from kernels.common.conv_winograd_fwd_rt import AOT_TILES

    return tuple(AOT_TILES)


def _operands(tile, io: str) -> Tuple[str, ...]:
    """Operand dtypes timed per tile: the ``auto`` one, and fp16 operands as
    well for bf16 F(2,3)."""
    from kernels.common.conv_winograd_fwd_rt import pipeline_dtypes

    op, _ = pipeline_dtypes(tile, io)
    return (op, "fp16") if op == "bf16" else (op,)


def _prescale(tile, io: str) -> bool:
    from kernels.common.conv_winograd_fwd_rt import pipeline_dtypes

    return pipeline_dtypes(tile, io)[1]


# ---------------------------------------------------------------------------
# The binary grid
# ---------------------------------------------------------------------------


def transform_specs() -> list:
    """The transform binaries of the grid (``conv_winograd_fwd_rt.aot_stage_specs``)."""
    from kernels.common.conv_winograd_fwd_rt import aot_stage_specs

    return aot_stage_specs()


def _k3i_name(cfg, dtype: str, vab: int, vc: int) -> str:
    from benchmarks.common.benchmark_winograd_conv import k3_label

    return f"wino_rt_k3i_{dtype}_{k3_label(cfg)}_va{vab}_vc{vc}"


def _k3i_spec(cfg, dtype: str, vab: int, vc: int, arch: str):
    """The implicit-GEMM K3 spec with the given vector widths (probe problem)."""
    from benchmarks.common.benchmark_winograd_conv import k3_spec

    fake = SimpleNamespace(cpg=vab, kpg=vc)
    return k3_spec(cfg, fake, dtype, arch)


def _job_list(arch: str) -> list:
    from benchmarks.common import winograd_k3_wmma as K3W
    from benchmarks.common.benchmark_winograd_conv import K3_CONFIGS_SHORT

    jobs = [("T", s) for s in transform_specs()]
    for dt in IOS:
        for knobs in K3W.K3W_CONFIGS:
            jobs.append(("W", (dict(knobs), dt)))
        for cfg in K3_CONFIGS_SHORT:
            for vab in VECS:
                for vc in VECS:
                    if cfg[7] == "default" and vc != 1:
                        continue
                    jobs.append(("I", (cfg, dt, vab, vc)))
    return jobs


def _compile_one(item):
    """``(name, meta, hsaco, symbol)`` or ``(name, None, error, None)``."""
    kind, payload, arch = item
    from rocke import compile_kernel

    try:
        if kind == "T":
            from kernels.common import conv_winograd_fwd_rt as W

            spec = payload
            name = spec.kernel_name()
            art = compile_kernel(W.build_stage(spec, arch=arch), arch=arch)
            meta = dict(kind="T", stage=spec.stage, cls=type(spec).__name__,
                        knobs=dataclasses.asdict(spec))  # fmt: skip
        elif kind == "W":
            from benchmarks.common import winograd_k3_wmma as K3W
            from kernels.common.conv_winograd_gemm_wmma import (
                build_winograd_gemm_wmma,
                winograd_gemm_reason,
            )

            knobs, dt = payload
            spec = K3W.spec_of(knobs, dt)
            name = "wino_rt_k3w_" + spec.kernel_name()
            if winograd_gemm_reason(spec, arch) is not None:
                return name, None, "invalid", None
            art = compile_kernel(build_winograd_gemm_wmma(spec, arch=arch), arch=arch)
            meta = dict(kind="W", stage="K3", knobs=knobs, dtype=dt)
        else:
            from kernels.common.conv_implicit_gemm import (
                build_implicit_gemm_conv,
                is_valid_spec,
            )

            cfg, dt, vab, vc = payload
            name = _k3i_name(cfg, dt, vab, vc)
            spec = _k3i_spec(cfg, dt, vab, vc, arch)
            if spec is None or not is_valid_spec(spec, arch)[0]:
                return name, None, "invalid", None
            art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
            meta = dict(kind="I", stage="K3", cfg=list(cfg), dtype=dt, vab=vab, vc=vc)
    except Exception as e:  # noqa: BLE001 -- reported, the grid goes on
        return f"{kind}:{payload!r}", None, f"{type(e).__name__}: {e}", None
    return name, meta, bytes(art.hsaco), art.kernel_name


def cache_index(cache_dir: str, arch: str) -> Path:
    """The index ``--compile-all`` writes last: a cache run needs it."""
    return Path(cache_dir) / arch / "wino_rt" / "index.json"


def compile_all(cache_dir: str, arch: str, jobs: int) -> int:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    out = cache_index(cache_dir, arch).parent
    out.mkdir(parents=True, exist_ok=True)
    work = [(k, p, arch) for k, p in _job_list(arch)]
    print(f"compile-all: {len(work)} binaries -> {out}", flush=True)
    t0 = time.time()
    index: Dict[str, dict] = {}
    n_bad = 0
    try:
        ctx = multiprocessing.get_context("forkserver")
    except ValueError:
        ctx = multiprocessing.get_context("spawn")
    from benchmarks.common import benchmark_winograd_aot as _self

    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=ctx) as pool:
        for i, (name, meta, data, sym) in enumerate(
            pool.map(_self._compile_one, work, chunksize=4), 1
        ):
            if meta is None:
                if data != "invalid":
                    n_bad += 1
                    print(f"  [fail] {name}: {data}", flush=True)
                continue
            (out / f"{name}.hsaco").write_bytes(data)
            meta["symbol"] = sym
            index[name] = meta
            if i % 200 == 0:
                print(f"  {i}/{len(work)} ({time.time() - t0:.0f}s)", flush=True)
    cache_index(cache_dir, arch).write_text(json.dumps(index, indent=1, sort_keys=True))
    print(
        f"compile-all done: {len(index)} binaries, {n_bad} failed "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 1 if n_bad else 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Binary:
    name: str
    meta: dict
    hsaco: bytes
    kernel_name: str
    spec: object = None

    @property
    def art(self):
        """The artifact ``K3Plan.bind`` launches (``.hsaco``, ``.kernel_name``)."""
        return self


def load_cache(cache_dir: str, arch: str) -> List[Binary]:
    from kernels.common import conv_winograd_fwd_rt as W

    path = cache_index(cache_dir, arch)
    d = path.parent
    index = json.loads(path.read_text())
    out = []
    for name, meta in sorted(index.items()):
        spec = None
        if meta["kind"] == "T":
            knobs = dict(meta["knobs"])
            if "lds" in knobs:
                knobs["lds"] = tuple(knobs["lds"])
            spec = getattr(W, meta["cls"])(**knobs)
        out.append(
            Binary(name, meta, (d / f"{name}.hsaco").read_bytes(), meta["symbol"], spec)
        )
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def _chunks(problem, tile, prescale: bool) -> Optional[int]:
    """Fewest N-chunks (a divisor of N) whose buffers and K3 operands fit the
    32-bit byte offsets. (The WMMA K3's own constraints are checked per plan:
    a problem it cannot run may still run on the implicit-GEMM K3.)"""
    from kernels.common import conv_winograd_fwd_rt as W

    for c in range(1, problem.N + 1):
        if problem.N % c:
            continue
        sub = dataclasses.replace(problem, N=problem.N // c)
        geo = W.RtGeometry(sub, *tile)
        if geo.check(prescale):
            continue
        g = _gemm_prob(geo)
        elems = (g.T * g.G * g.Cg, g.G * g.Kg * g.Cg, g.T * g.G * g.Kg)
        if 2 * max(elems) < 2**31:
            return c
    return None


def _gemm_prob(geo):
    from kernels.common.conv_winograd_gemm_wmma import WinogradGemmProblem

    p = geo.problem
    return WinogradGemmProblem(
        T=geo.num_tiles, G=geo.n2 * p.groups, Cg=p.cpg, Kg=p.K // p.groups
    )


def _gemm_conv_problem(geo):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    p = geo.problem
    return ConvProblem(
        N=geo.num_tiles,
        Hi=1,
        Wi=1,
        C=geo.n2 * p.C,
        K=geo.n2 * p.K,
        Y=1,
        X=1,
        groups=geo.n2 * p.groups,
    )


class Case:
    """Device buffers of one case: x / y (all chunks) and w are shared by the tiles."""

    def __init__(self, rt, problem, io: str, seed: int, verify: bool):
        from benchmarks.common.benchmark_winograd_conv import make_inputs, reference

        self.rt = rt
        self.problem = problem
        self.io = io
        self.x_t, self.w_t, self.y_t = make_inputs(problem, io, seed=seed)
        self.ref = reference(self.x_t, self.w_t, problem, io) if verify else None
        self.xb = self.x_t.numel() * 2
        self.wb = self.w_t.numel() * 2
        self.yb = self.y_t.numel() * 2
        self.x = int(rt.alloc(max(self.xb, 4)))
        self.w = int(rt.alloc(max(self.wb, 4)))
        self.y = int(rt.alloc(max(self.yb, 4)))
        from benchmarks.common.benchmark_winograd_conv import _u8

        rt.sync()
        for ptr, t in ((self.x, self.x_t), (self.w, self.w_t)):
            h = t.contiguous().pin_memory()
            rt.memcpy_h2d(ptr, _u8(h), h.nbytes)
            rt.sync()
            del h

    def read_y(self):
        import torch

        from benchmarks.common.benchmark_winograd_conv import _u8

        self.rt.sync()
        h = torch.empty_like(self.y_t, device="cpu").pin_memory()
        self.rt.memcpy_d2h(_u8(h), self.y, h.nbytes)
        self.rt.sync()
        return h.clone()

    def free(self):
        from rocke.runtime import synchronize_and_release

        synchronize_and_release(0)
        for p in (self.x, self.w, self.y):
            self.rt.free(p)


class TilePipe:
    """Workspaces and bound launches of one tile's pipeline on one case."""

    def __init__(self, case: Case, tile, op: str, prescale: bool, chunks: int):
        from kernels.common import conv_winograd_fwd_rt as W

        self.case = case
        self.tile = tuple(tile)
        self.op = op
        self.prescale = prescale
        self.chunks = chunks
        self.sub = dataclasses.replace(case.problem, N=case.problem.N // chunks)
        self.geo = W.RtGeometry(self.sub, *tile)
        sz = self.geo.buffer_bytes(prescale)
        self.sizes = sz
        rt = case.rt
        self.ws = {k: int(rt.alloc(max(sz[k], 4))) for k in ("U", "V", "M", "e")}
        self.xc = sz["x"]
        self.yc = sz["y"]

    def free(self):
        from rocke.runtime import synchronize_and_release

        synchronize_and_release(0)
        for p in self.ws.values():
            self.case.rt.free(p)

    def poison(self):
        rt = self.case.rt
        for k, p in self.ws.items():
            rt.memset(p, 0x7F, self.sizes[k])
        rt.memset(self.case.y, 0x7F, self.case.yb)

    # -- launches --

    def bind_stage(self, binary: Binary, chunk: int = 0):
        from kernels.common import conv_winograd_fwd_rt as W
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        spec = binary.spec
        grid, block, vals = W.stage_launch(spec, self.geo)
        x = self.case.x + chunk * self.xc
        y = self.case.y + chunk * self.yc
        bufs = dict(
            w=(self.case.w, self.sizes["w"]),
            u=(self.ws["U"], self.sizes["U"]),
            x=(x, self.xc),
            v=(self.ws["V"], self.sizes["V"]),
            e=(self.ws["e"], self.sizes["e"]),
            mm=(self.ws["M"], self.sizes["M"]),
            m=(self.ws["M"], self.sizes["M"]),
            y=(y, self.yc),
        )
        sig = W.stage_signature(spec)
        args = dict(vals)
        for a in sig:
            nm = a["name"]
            if nm in args:
                continue
            if nm.endswith("_bytes"):
                args[nm] = bufs[nm[: -len("_bytes")]][1]
            else:
                args[nm] = bufs[nm][0]
        lau = KernelLauncher(
            hsaco=binary.hsaco, kernel_name=binary.kernel_name, signature=sig
        )
        return lau.bind(args, config=LaunchConfig(grid=grid, block=(block, 1, 1)))

    def bind_k3(self, plan_or_impl):
        """A WMMA ``K3Plan`` or an implicit-GEMM ``(binary, spec)``."""
        from benchmarks.common.winograd_k3_wmma import K3Plan

        if isinstance(plan_or_impl, K3Plan):
            return plan_or_impl.bind(
                _gemm_prob(self.geo), self.ws["V"], self.ws["U"], self.ws["M"]
            )
        from kernels.common.conv_abi import conv_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        binary, spec = plan_or_impl
        gp = _gemm_conv_problem(self.geo)
        lau = KernelLauncher(
            hsaco=binary.hsaco,
            kernel_name=binary.kernel_name,
            signature=conv_args_signature(self.op, is_3d=False),
        )
        S = self.sizes
        vals = ConvArgs.from_problem(
            gp, tile_m=spec.tile_m, tile_n=spec.tile_n
        ).to_launch_values(
            self.ws["V"], self.ws["U"], self.ws["M"], S["V"], S["U"], S["M"]
        )
        grid = (-(-gp.N_gemm // spec.tile_n), -(-gp.M // spec.tile_m), gp.groups)
        if min(grid) <= 0 or grid[1] > 65535 or grid[2] > 65535:
            raise ValueError(f"K3 grid {grid} not launchable")
        return lau.bind(
            vals, config=LaunchConfig(grid=grid, block=(spec.launch_block_size, 1, 1))
        )

    def chain(self, pick: dict, k3):
        """The whole forward: K1 once, then per chunk K2a, K2, K3, K4."""
        fns = [self.bind_stage(pick["K1"])]
        k3f = self.bind_k3(k3)
        for i in range(self.chunks):
            if "K2a" in pick:
                fns.append(self.bind_stage(pick["K2a"], i))
            fns.append(self.bind_stage(pick["K2"], i))
            fns.append(k3f)
            fns.append(self.bind_stage(pick["K4"], i))

        def run(_f=tuple(fns)):
            for f in _f:
                f()

        return run


def _fits(binary: Binary, tp: TilePipe) -> bool:
    s = binary.spec
    m = binary.meta
    if m["kind"] != "T":
        return False
    k = m["knobs"]
    if k.get("io_dtype") != tp.case.io:
        return False
    if m["stage"] != "K2a":
        if (k["mh"], k["mw"]) != tp.tile or k["operand_dtype"] != tp.op:
            return False
        if m["stage"] in ("K2", "K4") and k["prescale"] != tp.prescale:
            return False
    return s.reason(tp.geo) is None


def _time_us(run, args) -> float:
    """One call of ``run`` (a whole pipeline) in microseconds, measured the
    ``--timing`` way: ``staged`` is the median of single stalled calls, as
    hipDNN times a convolution; ``block`` the average of back-to-back calls."""
    if args.timing == "staged":
        from statistics import median

        from rocke.runtime.stall_gate import time_launches_staged

        samples = time_launches_staged(run, warmup=args.warmup, iters=args.iters)
        return median(samples) * 1e3
    from rocke.runtime import time_launches

    return time_launches(run, warmup=args.warmup, iters=args.iters) * 1e3


def run_case(
    case: Case, tile, op: str, cache: List[Binary], args, log
) -> Optional[dict]:
    """The best verified pipeline of one tile: ``{label, ms, binaries, err}``;
    ``{"failed": True}`` when no K3 finalist verifies, ``None`` when no
    pipeline fits."""
    from benchmarks.common import winograd_k3_wmma as K3W
    from benchmarks.common.benchmark_winograd_conv import verify_error
    from rocke.runtime import time_launches

    prescale = _prescale(tile, case.io)
    chunks = _chunks(case.problem, tile, prescale)
    if chunks is None:
        log(f"  [skip] F{tile}: no batch split fits")
        return None
    tp = TilePipe(case, tile, op, prescale, chunks)
    try:

        # Ranking a stage's binaries times each alone, back to back; only the
        # assembled pipeline is timed the --timing way.
        def us(fn):
            return time_launches(fn, warmup=args.warmup, iters=args.iters) * 1e3

        ranked: Dict[str, List[Tuple[float, Binary]]] = {}
        for st in ("K1", "K2a", "K2", "K4") if prescale else ("K1", "K2", "K4"):
            cands = [b for b in cache if b.meta.get("stage") == st and _fits(b, tp)]
            timed = []
            for b in cands:
                try:
                    timed.append((us(tp.bind_stage(b)), b))
                except ValueError as e:
                    log(f"  [skip] {b.name}: {e}")
            timed.sort(key=lambda r: r[0])
            ranked[st] = timed
        # The prescale's input side: a fused K2, or K2a + a split K2.
        if prescale:
            fused = [r for r in ranked["K2"] if r[1].spec.fused]
            split = [r for r in ranked["K2"] if not r[1].spec.fused]
            opts = []
            if fused:
                opts.append((fused[0][0], {"K2": fused[0][1]}))
            if split and ranked["K2a"]:
                k2a = ranked["K2a"][0]
                opts.append((split[0][0] + k2a[0], {"K2a": k2a[1], "K2": split[0][1]}))
            if not opts:
                log(f"  [skip] F{tile}: no K2 binary fits")
                return None
            k2t, k2pick = min(opts, key=lambda o: o[0])
            ranked["K2"] = [(k2t, k2pick["K2"])]
            if "K2a" not in k2pick:
                del ranked["K2a"]
        for st in ("K1", "K2", "K4"):
            if not ranked[st]:
                log(f"  [skip] F{tile}: no {st} binary fits")
                return None
        stages = tuple(st for st in ("K1", "K2a", "K2", "K4") if st in ranked)
        pick = {st: ranked[st][0][1] for st in stages}
        if "K2a" in stages:
            pick["K2a"] = k2pick["K2a"]

        # K3: every WMMA plan (single + main/tail splits) and every implicit config.
        prob = _gemm_prob(tp.geo)
        gp = _gemm_conv_problem(tp.geo)
        built = [
            (b.meta["knobs"], K3W.spec_of(b.meta["knobs"], tp.op), b.art)
            for b in cache
            if b.meta["kind"] == "W" and b.meta["dtype"] == tp.op
        ]
        plans = K3W.candidate_plans(prob, built, args.cus, tails=True)
        k3c = []
        for pl in plans:
            if pl.reason(prob):
                continue
            try:
                k3c.append((us(tp.bind_k3(pl)), "wmma:" + pl.label(), pl,
                            [s.art.name for s in pl.steps]))  # fmt: skip
            except ValueError:
                continue
        vab = _pow2(gp.cpg)
        vc_max = _pow2(gp.kpg)
        for b in cache:
            m = b.meta
            if m["kind"] != "I" or m["dtype"] != tp.op or m["vab"] != vab:
                continue
            cfg = tuple(m["cfg"])
            want_vc = 1 if cfg[7] == "default" else vc_max
            if m["vc"] != want_vc:
                continue
            spec = _k3i_spec(cfg, tp.op, m["vab"], m["vc"], args.arch)
            if spec is None or -(-gp.M // spec.tile_m) > 65535:
                continue
            try:
                k3c.append(
                    (
                        us(tp.bind_k3((b, spec))),
                        "implicit:" + b.name,
                        (b, spec),
                        [b.name],
                    )
                )
            except ValueError:
                continue
        if not k3c:
            log(f"  [skip] F{tile}: no K3 candidate")
            return None
        k3c.sort(key=lambda r: r[0])

        # Assemble: the fastest K3 that verifies through the whole pipeline.
        best = None
        for k3us, k3label, k3, k3bins in k3c[: args.k3_finalists]:
            run = tp.chain(pick, k3)
            err = None
            if case.ref is not None:
                tp.poison()
                run()
                err, _ = verify_error(case.read_y(), case.ref)
                if not err < 5e-2:
                    log(f"  F{tile} op={op} K3 {k3label[:70]} FAIL err={err:.2e}")
                    continue
            best = (run, k3label, k3bins, err)
            break
        if best is None:
            return dict(failed=True)
        run, k3label, k3bins, err = best
        total = None
        for _ in range(args.repeats):
            if args.cooldown > 0:
                time.sleep(args.cooldown)
            t = _time_us(run, args)
            total = t if total is None else min(total, t)
        bins = [pick[st].name for st in stages] + list(k3bins)
        k2a = f"-k2a={pick['K2a'].name}" if "K2a" in pick else ""
        label = (
            f"winograd-{case.io}-F{tile[0]}x{tile[1]}-op={op}-ps={int(prescale)}"
            f"-k1={pick['K1'].name}{k2a}-k2={pick['K2'].name}-k3={k3label}"
            f"-k4={pick['K4'].name}"
        )
        log(
            f"  F{tile[0]}x{tile[1]} op={op} ps={int(prescale)} chunks={chunks}: "
            f"{total:9.2f} us  err={err if err is not None else float('nan'):.2e}  "
            + " ".join(f"{st}={ranked[st][0][0]:.1f}" for st in stages)
            + f" K3={k3c[0][0]:.1f} [{k3label[:60]}]"
        )
        return dict(label=label, ms=total / 1e3, binaries=bins, err=err)
    finally:
        tp.free()


def _pow2(n: int, cap: int = 8) -> int:
    v = cap
    while v > 1 and n % v:
        v //= 2
    return v


def winograd_case_reason(problem, dtype: str, direction: str) -> Optional[str]:
    """Why the AOT pipeline does not run a case (``None``: it may).

    The ``winograd_aot`` route's rule (``winograd_ineligible_reason``),
    narrowed to the 3x3 filters the shape-generic transforms build."""
    from kernels.common.conv_winograd_select import winograd_ineligible_reason

    if direction != "fwd":
        return "Winograd runs the forward direction only"
    why = winograd_ineligible_reason(problem, dtype)
    if why is None and (problem.Y, problem.X) != (3, 3):
        why = f"{problem.Y}x{problem.X} filter (the AOT transforms are 3x3)"
    return why


def cases_of(args) -> list:
    """``(problem, dtype, direction)`` of every case the run was given.

    As the other conv sweeps read them: an MIOpenDriver line runs the
    directions of its ``-F`` and its own dtype, ``--direction`` / ``--dtype``
    override both, and the shape flags are one case (fp16 fwd by default)."""
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _FORW_TO_DIRECTIONS,
        parse_miopen_cmd,
    )

    if args.miopen_file or args.miopen_cmd:
        if args.miopen_cmd:
            lines = [args.miopen_cmd]
        else:
            lines = Path(args.miopen_file).read_text().splitlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            problem, dtype, forw = parse_miopen_cmd(line)
            directions = (
                [args.direction]
                if args.direction
                else _FORW_TO_DIRECTIONS.get(forw, ["fwd"])
            )
            out += [(problem, args.dtype or dtype, d) for d in directions]
        return out
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    problem = ConvProblem(
        N=args.N,
        Hi=args.Hi,
        Wi=args.Wi,
        C=args.C,
        K=args.K,
        Y=args.Y,
        X=args.X,
        sH=args.sH,
        sW=args.sW,
        pH=args.pH,
        pW=args.pW,
        groups=args.groups,
    )
    return [(problem, args.dtype or "fp16", args.direction or "fwd")]


def run_from_cache(args) -> int:
    from benchmarks.common.case_results import (
        RUNNERS_UP,
        ResultsWriter,
        measured_keys,
        same_results_file,
    )
    from benchmarks.common.early_stop import case_key

    # Filtered before anything is loaded: a comparison hands this sweep every
    # case, and the ones it cannot run should cost a line, not a cache load.
    cases = []
    for problem, dtype, direction in cases_of(args):
        why = winograd_case_reason(problem, dtype, direction)
        if why is not None:
            print(
                f"[skip] {problem.short()} {dtype} {direction}: "
                f"not a Winograd case: {why}",
                flush=True,
            )
            continue
        cases.append((problem, dtype))
    if not cases:
        return 0
    # Read before the writer opens --results: the two may name one file.
    done = measured_keys(args.skip_measured, "winograd", args.timing)

    import torch  # noqa: F401 -- claim the HIP context first

    from rocke.runtime.hip_module import Runtime

    if not args.cus:
        from benchmarks.common.winograd_k3_wmma import device_cus

        args.cus = device_cus()
    if args.timing == "staged":
        # Checked once, as batched_timing.time_batch_staged does: a device
        # without the stall gate would otherwise fail every pipeline alike.
        from rocke.runtime.hip_module import HipError
        from rocke.runtime.stall_gate import default_gate

        try:
            default_gate()
        except HipError as e:
            raise HipError(
                f"{e}; this device cannot be timed with --timing staged, rerun "
                f"with --timing block"
            ) from e

    cache = load_cache(args.run_from_cache, args.arch)
    print(f"cache: {len(cache)} binaries", flush=True)
    results_out = ResultsWriter(
        args.results,
        "winograd",
        args.timing,
        append=same_results_file(args.results, args.skip_measured),
    )
    rt = Runtime()
    rc = 0
    n_skipped = 0
    for ci, (problem, dtype) in enumerate(cases, 1):
        key = case_key(problem, dtype, "fwd")
        if key in done:
            n_skipped += 1
            continue
        print(f"\n== [{ci}/{len(cases)}] {problem.short()} {dtype}", flush=True)
        case = Case(rt, problem, dtype, args.seed, args.verify)
        res = []
        failed = False
        try:
            for tile in _tiles():
                for op in _operands(tile, dtype):
                    try:
                        r = run_case(
                            case, tile, op, cache, args, lambda m: print(m, flush=True)
                        )
                    except (ValueError, RuntimeError) as e:
                        print(f"  [skip] F{tile} op={op}: {e}", flush=True)
                        r = None
                    if r is None:
                        continue
                    if r.get("failed"):
                        failed = True
                        continue
                    res.append(r)
        finally:
            case.free()
        if not res:
            if failed:
                # A pipeline that fits and computes the wrong answer is a bug;
                # one that does not fit is coverage.
                print("  no Winograd pipeline verified", flush=True)
                rc = 1
            else:
                print("  no Winograd pipeline fits this case", flush=True)
            continue
        res.sort(key=lambda r: r["ms"])
        best = res[0]
        sec = best["ms"] * 1e-3
        tflops = problem.flops / sec * 1e-12
        gbps = (case.xb + case.wb + case.yb) / sec * 1e-9
        # The line benchmark_conv_compare.py reads, as the other sweeps print.
        print(
            f"\nBest: {tflops:.1f} TFLOPS — {best['label']}  {best['ms']:.3f} ms",
            flush=True,
        )
        results_out.row(
            key=key,
            case=ci,
            shape=problem.short(),
            dtype=dtype,
            direction="fwd",
            tflops=tflops,
            ms=best["ms"],
            gbps=gbps,
            kernel=best["label"],
            candidates=len(res),
            # Every pipeline reported was verified, when --verify asked.
            passed=True if args.verify else None,
            runners_up=[r["label"] for r in res[1 : 1 + RUNNERS_UP]],
        )
    results_out.close()
    if n_skipped:
        print(
            f"\n{n_skipped} case(s) skipped: already in {args.skip_measured}",
            flush=True,
        )
    return rc


def main(argv=None) -> int:
    from benchmarks.common import batched_timing, case_results

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="gfx1250")
    ap.add_argument("--compile-all", action="store_true")
    ap.add_argument("--cache-dir", default="./kernel_cache")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--run-from-cache", default=None, metavar="DIR")
    ap.add_argument(
        "--miopen-file", default=None, help="MIOpenDriver commands, one per line"
    )
    ap.add_argument("--miopen-cmd", default=None, help="one MIOpenDriver command")
    ap.add_argument(
        "--dtype",
        default=None,
        choices=["fp16", "bf16"],
        help="I/O dtype (default: fp16, or each MIOpenDriver command's)",
    )
    ap.add_argument(
        "--direction",
        default=None,
        choices=["fwd", "dgrad", "wgrad"],
        help="direction (default: fwd, or each command's -F); Winograd runs fwd only",
    )
    case_results.add_results_arg(ap, merged=False)
    case_results.add_skip_measured_arg(ap)
    batched_timing.add_timing_arg(ap)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cooldown", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k3-finalists", type=int, default=4)
    ap.add_argument("--cus", type=int, default=0, help="0: query the device")
    shape = ap.add_argument_group("Shape", "one case, without an MIOpenDriver input")
    for name, default in (
        ("N", 8), ("Hi", 56), ("Wi", 56), ("C", 64), ("K", 64), ("Y", 3), ("X", 3),
        ("sH", 1), ("sW", 1), ("pH", 1), ("pW", 1), ("groups", 1),
    ):  # fmt: skip
        shape.add_argument(f"--{name}", type=int, default=default)
    args = ap.parse_args(argv)
    if args.compile_all:
        return compile_all(args.cache_dir, args.arch, args.jobs)
    if not args.run_from_cache:
        ap.error("give --compile-all or --run-from-cache")
    return run_from_cache(args)


if __name__ == "__main__":
    sys.exit(main())
