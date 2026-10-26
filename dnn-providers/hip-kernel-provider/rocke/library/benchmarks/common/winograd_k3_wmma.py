# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host side of the Winograd K3 batched GEMM (``conv_winograd_gemm_wmma``).

Builds a short list of tile configs, turns them into launch *plans* and picks
the fastest plan that verifies. A plan is one or two launches over disjoint
batch ranges: the main config takes as many batches as fill whole rounds of
blocks (one block per CU at the large tiles), a second config takes the rest,
so the last, partial round is not paid at full tile cost.

Used by ``benchmark_winograd_conv.py`` (``--k3 wmma``) and by standalone
sweeps; every function here is host-only except :meth:`K3Plan.bind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

# Knob dicts for WinogradGemmWmmaSpec (dtype is added per problem). The
# 256x256 tile (8 waves, 8x4 fragments) is the throughput config; the
# smaller ones serve small T / small batch counts and the tail of a plan.
K3W_CONFIGS: Tuple[dict, ...] = (
    dict(waves_m=2, waves_n=4, frags_m=8, frags_n=4, tile_k=128, stages=2),
    dict(waves_m=2, waves_n=4, frags_m=8, frags_n=4, tile_k=64, stages=3),
    dict(waves_m=2, waves_n=4, frags_m=4, frags_n=4, tile_k=128, stages=2),
    dict(waves_m=2, waves_n=2, frags_m=4, frags_n=8, tile_k=128, stages=2),
    dict(waves_m=2, waves_n=2, frags_m=4, frags_n=4, tile_k=128, stages=2),
    dict(waves_m=1, waves_n=4, frags_m=4, frags_n=2, tile_k=128, stages=2),
    dict(waves_m=1, waves_n=4, frags_m=4, frags_n=4, tile_k=128, stages=3),
)
# Always valid and quick to compile (tests, tiny problems).
K3W_CONFIGS_TINY: Tuple[dict, ...] = (
    dict(waves_m=1, waves_n=2, frags_m=2, frags_n=2, tile_k=64, stages=2),
    dict(waves_m=2, waves_n=2, frags_m=2, frags_n=2, tile_k=32, stages=3),
)
DEFAULT_CUS = 256


def device_cus(default: int = DEFAULT_CUS) -> int:
    """Compute units of device 0 (``default`` if the query fails)."""
    try:
        from rocke.runtime.device_info import get_device_info

        n = int(get_device_info(0)._properties.multiProcessorCount)
        return n if n > 0 else default
    except Exception:  # noqa: BLE001 -- the count only tunes the plan
        return default


def spec_of(knobs: dict, dtype: str):
    from kernels.common.conv_winograd_gemm_wmma import WinogradGemmWmmaSpec

    return WinogradGemmWmmaSpec(dtype=dtype, **knobs)


def build_one(item):
    """``(knobs, spec, artifact)``, or None for an invalid config (picklable
    worker)."""
    knobs, dtype, arch = item
    from rocke import compile_kernel
    from kernels.common.conv_winograd_gemm_wmma import (
        build_winograd_gemm_wmma,
        winograd_gemm_reason,
    )

    spec = spec_of(knobs, dtype)
    if winograd_gemm_reason(spec, arch) is not None:
        return None
    try:
        art = compile_kernel(build_winograd_gemm_wmma(spec, arch=arch), arch=arch)
    except Exception:  # noqa: BLE001 -- an invalid config is skipped
        return None
    return knobs, spec, art


def build_all(configs: Sequence[dict], dtype: str, arch: str, jobs: int = 8) -> list:
    work = [(k, dtype, arch) for k in configs]
    if jobs <= 1 or len(work) <= 1:
        res = [build_one(w) for w in work]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        try:
            ctx = multiprocessing.get_context("forkserver")
        except ValueError:
            ctx = multiprocessing.get_context("spawn")
        from benchmarks.common import winograd_k3_wmma as _self

        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            res = list(pool.map(_self.build_one, work))
    return [r for r in res if r is not None]


def build_reduce(dtype: str, arch: str):
    from rocke import compile_kernel
    from kernels.common.conv_winograd_gemm_wmma import (
        WinogradGemmReduceSpec,
        build_winograd_gemm_reduce,
    )

    return compile_kernel(
        build_winograd_gemm_reduce(WinogradGemmReduceSpec(dtype), arch=arch),
        arch=arch,
    )


@dataclass
class K3Step:
    spec: object
    art: object
    batches: Tuple[int, int]
    splits: int = 1


@dataclass
class K3Plan:
    """One or more K3 launches covering all G batches."""

    steps: List[K3Step] = field(default_factory=list)

    def label(self) -> str:
        parts = []
        for s in self.steps:
            lab = s.spec.label()
            if s.spec.split_k:
                lab += f"/s{s.splits}"
            parts.append(f"{lab}@g{s.batches[0]}+{s.batches[1]}")
        return " | ".join(parts)

    def workspace_bytes(self, prob) -> int:
        return max(
            (s.spec.workspace_bytes(prob, s.splits) for s in self.steps), default=0
        )

    def reason(self, prob) -> Optional[str]:
        from kernels.common.conv_winograd_gemm_wmma import (
            winograd_gemm_problem_reason,
        )

        covered = 0
        for s in self.steps:
            why = winograd_gemm_problem_reason(s.spec, prob, s.splits, s.batches)
            if why:
                return why
            if s.batches[0] != covered:
                return "batch ranges must be contiguous"
            covered += s.batches[1]
        if covered != prob.G:
            return f"plan covers {covered} of {prob.G} batches"
        return None

    def bind(
        self,
        prob,
        v: int,
        u: int,
        m: int,
        ws: int = 0,
        reduce_art=None,
    ) -> Callable[[], None]:
        """Pre-bound launches of the whole plan (grid math checked)."""
        from kernels.common.conv_winograd_gemm_wmma import (
            winograd_gemm_reduce_grid,
            winograd_gemm_reduce_signature,
            winograd_gemm_reduce_values,
            winograd_gemm_signature,
        )
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        why = self.reason(prob)
        if why:
            raise ValueError(f"K3 plan {self.label()}: {why}")
        fns = []
        for s in self.steps:
            grid = s.spec.grid(prob, s.splits, s.batches)
            if min(grid) <= 0:
                raise ValueError(f"K3 grid {grid} not launchable")
            lau = KernelLauncher(
                hsaco=s.art.hsaco,
                kernel_name=s.art.kernel_name,
                signature=winograd_gemm_signature(s.spec),
            )
            d = ws if s.spec.split_k else m
            vals = s.spec.launch_values(prob, v, u, d, s.splits, s.batches)
            cfg = LaunchConfig(grid=grid, block=(s.spec.threads_per_block, 1, 1))
            fns.append(lau.bind(vals, config=cfg))
            if s.spec.split_k:
                if reduce_art is None or not ws:
                    raise ValueError("split-K needs the reduce kernel and a workspace")
                if s.batches != (0, prob.G):
                    raise ValueError("split-K steps must cover every batch")
                rgrid = winograd_gemm_reduce_grid(prob)
                red = KernelLauncher(
                    hsaco=reduce_art.hsaco,
                    kernel_name=reduce_art.kernel_name,
                    signature=winograd_gemm_reduce_signature(s.spec.dtype),
                )
                rv = winograd_gemm_reduce_values(prob, ws, m, vals["splits"])
                fns.append(
                    red.bind(rv, config=LaunchConfig(grid=rgrid, block=(256, 1, 1)))
                )

        def run(_f=tuple(fns)):
            for f in _f:
                f()

        return run


def main_batches(spec, prob, cus: int) -> int:
    """Batches the main config of a two-step plan takes: as many as fill
    whole rounds of ``cus`` blocks (0 if not even one round)."""
    gx, gy, _ = spec.tiles(prob)
    per_batch = gx * gy
    rounds = (prob.G * per_batch) // cus
    return min(prob.G, (rounds * cus) // per_batch)


def candidate_plans(
    prob, built: Sequence[tuple], cus: int, *, tails: bool = True
) -> List[K3Plan]:
    """Every single-config plan, plus (main, tail) batch splits for each pair
    of configs where the main config leaves a partial round."""
    plans = [K3Plan([K3Step(spec, art, (0, prob.G))]) for _, spec, art in built]
    if not tails:
        return plans
    for _, spec, art in built:
        g1 = main_batches(spec, prob, cus)
        if not 0 < g1 < prob.G:
            continue
        for _, tspec, tart in built:
            plans.append(
                K3Plan(
                    [
                        K3Step(spec, art, (0, g1)),
                        K3Step(tspec, tart, (g1, prob.G - g1)),
                    ]
                )
            )
    return plans


def sweep(
    prob,
    plans: Sequence[K3Plan],
    bind: Callable[[K3Plan], Callable[[], None]],
    *,
    verify: Optional[Callable[[Callable[[], None]], float]],
    time_fn: Callable[[Callable[[], None]], float],
    tol: float = 5e-2,
    log: Callable[[str], None] = print,
    finalists: int = 4,
) -> Optional[Tuple[K3Plan, float, Optional[float]]]:
    """Time every plan once, re-time the best ``finalists`` (slower of two
    timings counts) and return ``(plan, us, err)`` of the fastest one that
    verifies. ``verify(run) -> err``; ``time_fn(run) -> us``."""
    first: List[Tuple[float, K3Plan, Callable[[], None]]] = []
    for plan in plans:
        if plan.reason(prob):
            continue
        try:
            run = bind(plan)
        except ValueError as e:
            log(f"  [skip] {plan.label()}: {e}")
            continue
        us = time_fn(run)
        first.append((us, plan, run))
    first.sort(key=lambda r: r[0])
    best = None
    for us0, plan, run in first[:finalists]:
        err = None
        if verify is not None:
            err = verify(run)
            if not err < tol:
                log(f"  K3 {plan.label():60s} FAIL err={err:.2e}")
                continue
        us = max(us0, time_fn(run))
        log(
            f"  K3 {plan.label():60s} {us:9.2f} us"
            + (f"  err={err:.2e}" if err is not None else "")
        )
        if best is None or us < best[1]:
            best = (plan, us, err)
    return best
