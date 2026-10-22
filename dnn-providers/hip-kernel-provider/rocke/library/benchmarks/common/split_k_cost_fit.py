# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure split-K degree curves and fit the split-K cost model to them.

The constants of :class:`rocke.helpers.split_k_planner.SplitKCostModel` are
fitted, not derived. This tool is how they are produced, so they can be
refreshed after a kernel change and fitted for a new arch:

``sweep``
    Runs :mod:`benchmark_implicit_gemm_conv` ``--run-from-cache`` once per
    wgrad shape, timing every cached split-K binary at every power-of-two
    degree up to ``--max-degree`` (the degree is a kernarg, so one binary
    covers them all) next to the unsplit kernels, with ``--verify`` on. One
    log per shape, resumable::

        python -m benchmarks.common.split_k_cost_fit sweep \\
            --shapes wgrad_shapes.txt --cache kernel_cache --out sweep_logs

    ``--shapes`` holds one MIOpenDriver command per line.

``fit``
    Fits the model to those logs in two stages::

        python -m benchmarks.common.split_k_cost_fit fit --logs sweep_logs --arch gfx950

    1. **Degree.** One curve per (shape, split-K binary) over the degree. The
       K-loop, round and atomic-traffic constants are fitted so the model's
       argmin lands on each curve's measured best.
    2. **Reduction.** A reduction's setup (zeroing the output, or clearing a
       workspace and folding it) costs the same at every degree, so it cannot
       move a curve's argmin; it decides between binaries instead. For each
       shape and main loop (same tile, warps, atom, pipeline and A width), the
       unsplit, packed-atomic and workspace kernels at every degree are one
       decision, and ``fill_bytes_per_latency`` is fitted to it.

    Only kernels within ``--competitive`` of their shape's best time count --
    the ones dispatch or a benchmark could pick. Validation holds shapes out:
    stage 2 is cross-validated over ``--holdout`` folds (every shape held out
    once), stage 1 too with ``--cv`` (else one fold). The report gives the
    pooled held-out regret (time of the choice over the best measured) for
    the model and for the previous CK-formula rule; the constants printed as a
    ``SplitKCostModel(...)`` literal are then fitted on every shape.

``margin``
    Measures :attr:`~rocke.helpers.split_k_planner.SplitKCostModel.ladder_margin`
    on a sweep that also timed degrees off the power-of-two ladder
    (``sweep --degrees 2-160,...``)::

        python -m benchmarks.common.split_k_cost_fit margin --logs offladder_logs

    Each margin is scored against always choosing the ladder, per binary and per
    shape; the recommendation is the best shape geomean whose worst shape loses
    at most ``--max-worse``.

Only relative costs matter, so the model is invariant to its time unit; the
unit is the load latency of one K iteration, and every fitted constant is a
ratio to it (no absolute time or bandwidth is produced). Stage 1 evaluates a
vectorised copy of :func:`predict_split_k_cost`; the tool checks at the end
that the scalar model picks the same degree on every curve.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from rocke.helpers.split_k_planner import (
    MAX_GRID_DIM_Z,
    SplitKKernel,
    SplitKProblem,
    SplitKReduction,
)

_TIMED = re.compile(
    r"\]\s+[\d.]+ TFLOPS\s+([\d.]+) ms\s+(wgrad-\S+)(?: @split_k=(\d+))?\s*$",
    re.MULTILINE,
)
_DONE = ("\nBest:", "No successful launches", "no compatible kernel")
_DEGREE_PARAMS = (
    "flops_per_latency",
    "load_bytes_per_latency",
    "cta_cost",
    "atomic:bf16",
    "atomic:fp16",
    "atomic:fp32",
)


def _degrees(max_degree: int) -> list[int]:
    return [1 << i for i in range(1, max_degree.bit_length()) if 1 << i <= max_degree]


def _parse_degrees(spec: str) -> list[int]:
    """``"2-160,192,256"`` -> the listed degrees and inclusive ranges."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    if not out or min(out) < 2:
        raise argparse.ArgumentTypeError(f"degrees must be >= 2: {spec!r}")
    return sorted(out)


# --------------------------------------------------------------------- sweep


def cmd_sweep(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = [
        s.strip() for s in Path(args.shapes).read_text().splitlines() if s.strip()
    ]
    degrees = ",".join(map(str, args.degrees or _degrees(args.max_degree)))
    lib = Path(__file__).resolve().parents[2]
    for idx, shape in enumerate(shapes, 1):
        log = out / f"shape_{idx:03d}.log"
        if log.exists() and any(m in log.read_text(errors="replace") for m in _DONE):
            continue
        print(f"[{idx}/{len(shapes)}] {shape}", flush=True)
        with log.open("w") as f:
            f.write(f"# CMD {shape}\n")
            f.flush()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.common.benchmark_implicit_gemm_conv",
                    "--run-from-cache",
                    args.cache,
                    "--cache-dir",
                    args.cache,
                    "--arch",
                    args.arch,
                    "--direction",
                    "wgrad",
                    "--split-k-degrees",
                    degrees,
                    # Both reductions of each main loop: stage 2 of the fit
                    # needs them side by side.
                    "--two-stage",
                    "always",
                    "--verify",
                    "--split-k-prune",
                    str(args.prune),
                    "--miopen-cmd",
                    shape,
                    *args.bench_args,
                ],
                cwd=lib,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    return 0


# ---------------------------------------------------------------- measurements


@dataclass(frozen=True)
class _Kernel:
    """One cached binary on one shape, with the planner's view of it."""

    shape: str  # the log it was timed in
    cmd: str  # the MIOpenDriver command: the shape's identity across logs
    label: str
    problem: SplitKProblem
    kernel: SplitKKernel
    reduction: SplitKReduction | None  # None for the unsplit kernel
    main_loop: tuple[str | int | bool, ...]  # binaries sharing this run one K loop
    times: dict[int, float]  # degree -> ms


def _load(log_dirs: Sequence[str], fill_dirs: Sequence[str] = ()) -> list[_Kernel]:
    """Timed curves from ``log_dirs``; ``fill_dirs`` (logs re-timing points a
    sweep skipped) only fill degrees a matching (shape, kernel) curve lacks."""

    from kernels.common.conv_implicit_gemm_wgrad import (
        _DEFAULT_WS_REPLICAS,
        _wg_K,
        _wg_M,
        _wg_N,
    )
    from kernels.common.conv_split_k import ATOMIC_16BIT_MAX_SPLIT_K

    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd
    from benchmarks.common.kernel_cache import KernelIdentity

    out: dict[tuple, _Kernel] = {}
    for d in log_dirs:
        for log in sorted(Path(d).glob("*.log")):
            text = log.read_text(errors="replace")
            if not text.startswith("# CMD ") or not any(m in text for m in _DONE):
                continue
            cmd = text.splitlines()[0][6:]
            problem, dtype, _forw = parse_miopen_cmd(cmd)
            nbytes = 4 if dtype == "fp32" else 2
            for m in _TIMED.finditer(text):
                ms, label = float(m.group(1)), m.group(2)
                degree = int(m.group(3) or 1)
                key = (str(log), label)
                if key not in out:
                    i = KernelIdentity.from_label(label)
                    if i.streamk != "off":
                        continue
                    gm = max(1, i.group_merge)
                    red = None
                    if i.split_k > 1 and i.two_stage:
                        copies = i.ws_replicas or _DEFAULT_WS_REPLICAS
                        red = SplitKReduction(
                            "workspace", "fp32", workspace=True, workspace_copies=copies
                        )
                    elif i.split_k > 1 and i.dtype_d in ("bf16", "fp16"):
                        red = SplitKReduction(
                            "atomic", i.dtype_d, max_split_k=ATOMIC_16BIT_MAX_SPLIT_K
                        )
                    elif i.split_k > 1:
                        red = SplitKReduction("atomic", "fp32")
                    out[key] = _Kernel(
                        shape=str(log),
                        cmd=cmd,
                        label=label,
                        problem=SplitKProblem(
                            m=_wg_M(problem) * gm,
                            n=_wg_N(problem) * gm,
                            k=_wg_K(problem),
                            batch=max(1, problem.groups // gm),
                            operand_bytes=nbytes,
                            output_bytes=nbytes,
                        ),
                        kernel=SplitKKernel(
                            i.tile_m, i.tile_n, i.tile_k, i.warp_m * i.warp_n
                        ),
                        reduction=red,
                        main_loop=(
                            str(log),
                            i.tile_m,
                            i.tile_n,
                            i.tile_k,
                            i.warp_m,
                            i.warp_n,
                            i.warp_tile_m,
                            i.warp_tile_k,
                            i.pipeline,
                            i.vector_size_a,
                            i.lds_k_outer,
                            i.prefetch_addr,
                            gm,
                        ),
                        times={},
                    )
                times = out[key].times
                times[degree] = min(ms, times.get(degree, math.inf))
    kernels = list(out.values())
    by_cmd_label: dict[tuple, list] = {}
    for k in kernels:
        by_cmd_label.setdefault((k.cmd, k.label), []).append(k)
    for d in fill_dirs:
        for log in sorted(Path(d).glob("*.log")):
            text = log.read_text(errors="replace")
            if not text.startswith("# CMD "):
                continue
            cmd = text.splitlines()[0][6:]
            for m in _TIMED.finditer(text):
                for k in by_cmd_label.get((cmd, m.group(2)), []):
                    k.times.setdefault(int(m.group(3) or 1), float(m.group(1)))
    return kernels


def _competitive(kernels: list[_Kernel], factor: float) -> np.ndarray:
    """Kernels within ``factor`` of the best time measured on their shape."""
    best = np.array([min(k.times.values()) for k in kernels])
    cmds = np.array([k.cmd for k in kernels])
    shape_best = {c: best[cmds == c].min() for c in set(cmds)}
    return best <= factor * np.array([shape_best[c] for c in cmds])


# ---------------------------------------------------------- stage 1: degree


@dataclass
class _Curves:
    """Split-K binaries as arrays: one row per binary, one column per degree."""

    rows: list
    m: np.ndarray
    n: np.ndarray
    k: np.ndarray
    batch: np.ndarray
    tm: np.ndarray
    tn: np.ndarray
    tk: np.ndarray
    waves: np.ndarray
    operand_bytes: np.ndarray
    accum: np.ndarray
    t: np.ndarray  # ns, nan = not timed
    degrees: np.ndarray
    feasible: np.ndarray  # degrees the planner may choose (production rule)
    filled: np.ndarray  # t, with untimed feasible degrees at the nearest timed one


def _curves(kernels: list[_Kernel], max_degree: int) -> _Curves:
    degrees = np.array(_degrees(max_degree))
    rows = [
        k
        for k in kernels
        if k.reduction is not None and sum(d in k.times for d in degrees) >= 2
    ]
    t = np.full((len(rows), len(degrees)), np.nan)
    for r, k in enumerate(rows):
        for j, d in enumerate(degrees):
            t[r, j] = k.times.get(int(d), np.nan) * 1e6
    # What production may choose: every degree within the K iterations, the
    # gridDim.z budget and the reduction's cap. An untimed one (the sweep
    # early-stopped or pruned it) takes its nearest timed neighbour's time: a
    # lower bound, since the sweep skips a degree only after its neighbours
    # showed it to be slower.
    k_iters = np.array([math.ceil(k.problem.k / k.kernel.tile_k) for k in rows])
    batch = np.array([k.problem.batch for k in rows])
    cap = np.array([k.reduction.max_split_k or degrees[-1] for k in rows])
    feasible = (
        (degrees[None, :] <= k_iters[:, None])
        & (batch[:, None] * degrees[None, :] <= MAX_GRID_DIM_Z)
        & (degrees[None, :] <= cap[:, None])
    )
    filled = t.copy()
    for r in range(len(rows)):
        timed = np.flatnonzero(~np.isnan(t[r]))
        for j in np.flatnonzero(np.isnan(t[r]) & feasible[r]):
            filled[r, j] = t[r, timed[np.argmin(np.abs(timed - j))]]

    def col(f):
        return np.array([f(k) for k in rows], float)

    return _Curves(
        rows=rows,
        m=col(lambda k: k.problem.m),
        n=col(lambda k: k.problem.n),
        k=col(lambda k: k.problem.k),
        batch=col(lambda k: k.problem.batch),
        tm=col(lambda k: k.kernel.tile_m),
        tn=col(lambda k: k.kernel.tile_n),
        tk=col(lambda k: k.kernel.tile_k),
        waves=col(lambda k: k.kernel.waves_per_block),
        operand_bytes=col(lambda k: k.problem.operand_bytes),
        accum=np.array([k.reduction.accum_dtype for k in rows]),
        t=t,
        degrees=degrees,
        feasible=feasible,
        filled=filled,
    )


def _predict_curves(c: _Curves, x: dict) -> np.ndarray:
    """Vectorised :func:`predict_split_k_cost` without the per-launch setup term
    (a constant along each curve, so it cannot move the argmin)."""
    s = c.degrees[None, :].astype(float)
    tiles = np.ceil(c.m / c.tm) * np.ceil(c.n / c.tn) * c.batch
    ctas = tiles[:, None] * s
    iters = np.ceil(np.ceil(c.k / c.tk)[:, None] / s)
    per_cu = np.maximum(1.0, np.floor(x["max_waves_per_cu"] / c.waves))[:, None]
    slots = x["num_cus"] * per_cu
    flop = (2.0 * c.tm * c.tn * c.tk)[:, None]
    m_eff, n_eff = np.minimum(c.tm, c.m), np.minimum(c.tn, c.n)
    byte = ((m_eff + n_eff) * c.tk * c.operand_bytes)[:, None]

    def tau(a):
        return 1.0 + np.maximum(
            a * flop / x["flops_per_latency"], a * byte / x["load_bytes_per_latency"]
        )

    full = np.floor(ctas / slots)
    rest = ctas - full * slots
    t = full * iters * tau(slots / x["num_cus"])
    t += (rest > 0) * iters * tau(np.ceil(rest / x["num_cus"]))
    t += ctas * x["cta_cost"]
    bw = np.array([x[f"atomic:{a}"] for a in c.accum])[:, None]
    return t + ctas * (m_eff * n_eff)[:, None] * 4.0 / bw


def _picks(c: _Curves, pred: np.ndarray) -> np.ndarray:
    """The degree index production would choose: argmin over feasible degrees."""
    return np.argmin(np.where(c.feasible, pred, np.inf), axis=1)


def _curve_regret(c: _Curves, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    j = _picks(c, pred)
    return (c.filled[np.arange(len(c.t)), j] / np.nanmin(c.t, axis=1))[mask]


def _untimed_picks(c: _Curves, pred: np.ndarray, mask: np.ndarray) -> int:
    j = _picks(c, pred)
    return int(np.isnan(c.t[np.arange(len(c.t)), j])[mask].sum())


def _search(obj, x0: dict, names: Sequence[str], iters: int, seed: int) -> dict:
    """Multiplicative random local search minimising ``obj(x)``."""
    rng = np.random.default_rng(seed)
    best, cur, scale = dict(x0), obj(x0), 0.5
    for it in range(iters):
        x = dict(best)
        for name in rng.choice(names, size=rng.integers(1, 3), replace=False):
            x[name] = x[name] * math.exp(rng.normal(0.0, scale))
        o = obj(x)
        if o < cur:
            best, cur = x, o
        if it % 500 == 499:
            scale *= 0.6
    return best


# -------------------------------------------------------- stage 2: reduction


@dataclass(frozen=True)
class _Decision:
    """The binaries sharing one main loop on one shape, as one choice.

    One entry per measured option (binary and degree) in ``base`` (model time
    without the reduction's setup), ``setup_bytes`` and ``measured`` (ms).
    """

    shape: str
    base: np.ndarray
    setup_bytes: np.ndarray
    measured: np.ndarray
    members: list


def _feasible_times(m: _Kernel, max_degree: int) -> dict:
    """Degree -> ms for every degree production may choose for ``m``: 1 for
    the unsplit kernel; powers of two within the K iterations, the gridDim.z
    budget and the reduction's cap for a split-K one. An untimed degree takes
    its nearest timed neighbour's time (a lower bound)."""
    if m.reduction is None:
        return {1: m.times[1]} if 1 in m.times else {}
    p = m.problem
    cap = min(
        max_degree,
        math.ceil(p.k / m.kernel.tile_k),
        MAX_GRID_DIM_Z // p.batch,
        m.reduction.max_split_k or max_degree,
    )
    timed = sorted(d for d in m.times if d > 1)
    if not timed:
        return {}
    out = {}
    for d in _degrees(cap):
        near = min(timed, key=lambda t: abs(math.log2(t) - math.log2(d)))
        out[d] = m.times.get(d, m.times[near])
    return out


def _decisions(kernels: list[_Kernel], cost, max_degree: int = 1024) -> list[_Decision]:
    """One :class:`_Decision` per main loop with more than one reduction."""
    from rocke.helpers.split_k_planner import predict_split_k_cost

    groups: dict[tuple, list] = {}
    for k in kernels:
        groups.setdefault(k.main_loop, []).append(k)
    out = []
    for members in groups.values():
        kinds = {
            None if m.reduction is None else m.reduction.workspace for m in members
        }
        if len(kinds) < 2:
            continue  # nothing to decide between
        # One option per (reduction kind, degree, workspace copies): binaries
        # of one option differ only in vector width / epilogue, which the model
        # does not see, so each option is timed at its best binary.
        opts: dict[tuple, list] = {}
        for m in members:
            p, red = m.problem, m.reduction
            kind = "none" if red is None else ("ws" if red.workspace else "atomic")
            out_bytes = p.batch * p.m * p.n * p.output_bytes
            for degree, ms in _feasible_times(m, max_degree).items():
                if red is None:
                    b = predict_split_k_cost(p, m.kernel, 1, None, cost)
                    nb = 0.0
                else:
                    no_setup = replace(cost, fill_bytes_per_latency=math.inf)
                    b = predict_split_k_cost(p, m.kernel, degree, red, no_setup)
                    if red.workspace:
                        ws = p.batch * p.m * p.n * 4.0 * red.workspace_copies
                        nb = 2 * ws + out_bytes
                    else:
                        nb = float(out_bytes)
                # Workspace binaries differ in setup traffic by replica count,
                # so each count is its own option: a time stays paired with
                # the setup bytes of the binary that produced it.
                key = (kind, degree, red.workspace_copies if kind == "ws" else 0)
                prev = opts.get(key)
                opts[key] = [b, nb, min(ms, prev[2]) if prev else ms]
        base = [v[0] for v in opts.values()]
        nbytes = [v[1] for v in opts.values()]
        measured = [v[2] for v in opts.values()]
        out.append(
            _Decision(
                shape=members[0].cmd,
                base=np.array(base),
                setup_bytes=np.array(nbytes),
                measured=np.array(measured),
                members=members,
            )
        )
    return out


def _decision_regret(decisions: list[_Decision], fill: float) -> np.ndarray:
    r = []
    for d in decisions:
        j = np.argmin(d.base + d.setup_bytes / fill)
        r.append(d.measured[j] / d.measured.min())
    return np.array(r)


def _legacy_decision_regret(decisions_kernels, max_degree: int = 1024) -> np.ndarray:
    """The previous rule: the packed atomic whenever built, else the workspace,
    at the CK degree -- timed at the best binary of that kind. A CK degree of 1
    is the unsplit kernel; a decision without a timed unsplit one is skipped."""
    from rocke.helpers.split_k import select_split_k_wgrad

    r = []
    for members in decisions_kernels:
        best = min(min(m.times.values()) for m in members)
        atomic = [m for m in members if m.reduction and not m.reduction.workspace]
        ws = [m for m in members if m.reduction and m.reduction.workspace]
        chosen = atomic or ws
        if not chosen:
            continue
        p, q = chosen[0].problem, chosen[0].kernel
        ck = select_split_k_wgrad(
            wg_M=p.m,
            wg_N=p.n,
            wg_K=p.k,
            tile_m=q.tile_m,
            tile_n=q.tile_n,
            tile_k=q.tile_k,
            groups=p.batch,
            block_size=q.waves_per_block * 64,
        ).split_k
        if ck <= 1:
            unsplit = [
                m.times[1] for m in members if m.reduction is None and 1 in m.times
            ]
            if unsplit:
                r.append(min(unsplit) / best)
            continue
        times = []
        for m in chosen:
            opts = _feasible_times(m, max_degree)
            if opts:
                d = min(opts, key=lambda t: abs(math.log2(t) - math.log2(ck)))
                times.append(opts[d])
        if times:
            r.append(min(times) / best)
    return np.array(r)


# ------------------------------------------------------------------------ fit


def _summary(r: np.ndarray) -> str:
    if not len(r):
        return "(no cases)"
    return (
        f"geomean {np.exp(np.log(r).mean()):.3f}  p90 {np.quantile(r, 0.9):.3f}  "
        f"max {r.max():.2f}  within 3% {np.mean(r <= 1.03):.0%}  (n={len(r)})"
    )


def _x_of(cost) -> dict:
    x = {
        "num_cus": cost.num_cus,
        "max_waves_per_cu": cost.max_waves_per_cu,
        "flops_per_latency": cost.flops_per_latency,
        "load_bytes_per_latency": cost.load_bytes_per_latency,
        "cta_cost": max(cost.cta_cost, 1e-4),
        "fill_bytes_per_latency": cost.fill_bytes_per_latency,
        # Carried, not searched: hardware fact and the separately fitted margin.
        "num_xcds": cost.num_xcds,
        "ladder_margin": cost.ladder_margin,
    }
    for k, v in cost.atomic_bytes_per_latency.items():
        x[f"atomic:{k}"] = v
    return x


def _sig(v: float) -> float:
    return float(f"{v:.6g}")


def _model_of(x: dict, max_degree: int = 1024):
    from rocke.helpers.split_k_planner import SplitKCostModel

    return SplitKCostModel(
        num_cus=int(x["num_cus"]),
        max_waves_per_cu=int(x["max_waves_per_cu"]),
        flops_per_latency=_sig(x["flops_per_latency"]),
        load_bytes_per_latency=_sig(x["load_bytes_per_latency"]),
        cta_cost=_sig(x["cta_cost"]),
        atomic_bytes_per_latency={
            k.split(":")[1]: _sig(v) for k, v in x.items() if k.startswith("atomic:")
        },
        fill_bytes_per_latency=_sig(x["fill_bytes_per_latency"]),
        max_split_k=max_degree,
        ladder_margin=x["ladder_margin"],
        num_xcds=int(x["num_xcds"]),
    )


def cmd_fit(args: argparse.Namespace) -> int:
    from rocke.helpers.split_k import _ARCH_NUM_CUS, select_split_k_wgrad
    from rocke.helpers.split_k_planner import predict_split_k_cost, split_k_cost_model

    kernels = _load(args.logs, args.fill_logs)
    if not kernels:
        print("no complete sweep logs found", file=sys.stderr)
        return 1
    comp_all = _competitive(kernels, args.competitive)
    shapes = sorted({k.cmd for k in kernels})
    # An unfitted arch starts from gfx950's fitted constants as a search seed,
    # but its hardware facts are its own.
    start = split_k_cost_model(args.arch)
    if start is None:
        num_cus = args.num_cus or _ARCH_NUM_CUS.get(args.arch.split(":", 1)[0])
        if num_cus is None:
            print(
                f"unknown CU count for {args.arch!r}: pass --num-cus", file=sys.stderr
            )
            return 1
        start = replace(
            split_k_cost_model("gfx950"),
            num_cus=num_cus,
            max_waves_per_cu=args.max_waves_per_cu,
        )
    elif args.num_cus:
        start = replace(start, num_cus=args.num_cus)
    x0 = _x_of(start)

    # Stage 1: degree. Held out by shape: every --holdout-th shape (fold 0),
    # or with --cv every shape once across --holdout folds.
    comp_of = {id(k): bool(c) for k, c in zip(kernels, comp_all)}
    c = _curves(kernels, args.max_degree)
    comp = np.array([comp_of[id(k)] for k in c.rows])
    row_shape = np.array([k.cmd for k in c.rows])
    print(
        f"stage 1: {len(c.rows)} split-K curves over {len(shapes)} shapes, "
        f"{comp.sum()} competitive"
    )

    def fit_degree(train):
        return _search(
            lambda x: float(
                np.log(_curve_regret(c, _predict_curves(c, x), train)).mean()
            ),
            x0,
            _DEGREE_PARAMS,
            args.iters,
            args.seed,
        )

    ck = np.array(
        [
            select_split_k_wgrad(
                wg_M=k.problem.m,
                wg_N=k.problem.n,
                wg_K=k.problem.k,
                tile_m=k.kernel.tile_m,
                tile_n=k.kernel.tile_n,
                tile_k=k.kernel.tile_k,
                arch=args.arch,
                groups=k.problem.batch,
                block_size=k.kernel.waves_per_block * 64,
            ).split_k
            for k in c.rows
        ]
    )
    ck_pred = np.abs(np.log2(c.degrees[None, :]) - np.log2(np.maximum(ck, 1))[:, None])
    ck_r, model_r, ck_u, model_u = [], [], 0, 0
    fold_x: dict[int, dict] = {}
    for fold in range(args.holdout if args.cv else 1):
        test = np.isin(row_shape, shapes[fold :: args.holdout])
        xf = fold_x[fold] = fit_degree(~test & comp)
        pred = _predict_curves(c, xf)
        ck_r.append(_curve_regret(c, ck_pred, test & comp))
        model_r.append(_curve_regret(c, pred, test & comp))
        ck_u += _untimed_picks(c, ck_pred, test & comp)
        model_u += _untimed_picks(c, pred, test & comp)
    print("  held-out degree regret, competitive kernels (untimed picks at the")
    print("  nearest timed degree, a lower bound):")
    print(f"    CK formula  {_summary(np.concatenate(ck_r))}  untimed picks {ck_u}")
    print(
        f"    cost model  {_summary(np.concatenate(model_r))}  untimed picks {model_u}"
    )
    # The constants shipped are fitted on every shape.
    x = fit_degree(comp)

    # Stage 2: reduction (and split vs unsplit). Each decision is small, so
    # one held-out split is too noisy to judge on: every shape is held out
    # once across ``--holdout`` folds and the held-out regrets are pooled.
    comp_kernels = [k for k in kernels if comp_of[id(k)]]
    decisions = _decisions(comp_kernels, _model_of(x, args.max_degree), args.max_degree)
    print(
        f"stage 2: {len(decisions)} main loops with more than one reduction, "
        f"{args.holdout}-fold cross-validated by shape"
    )
    if decisions:
        grid = list(np.geomspace(1e3, 1e8, 101))

        def fit_setup(ds):
            return min(grid, key=lambda fb: np.log(_decision_regret(ds, fb)).mean())

        model_r, legacy_r = [], []
        for fold in range(args.holdout):
            out = set(shapes[fold :: args.holdout])
            # The degree constants of a fold are fitted without its shapes too,
            # or the held-out decisions would see their own timings.
            if fold not in fold_x:
                fold_x[fold] = fit_degree(~np.isin(row_shape, list(out)) & comp)
            fold_d = _decisions(
                comp_kernels, _model_of(fold_x[fold], args.max_degree), args.max_degree
            )
            train_d = [d for d in fold_d if d.shape not in out]
            test_d = [d for d in fold_d if d.shape in out]
            if not train_d or not test_d:
                continue
            model_r.append(_decision_regret(test_d, fit_setup(train_d)))
            legacy_r.append(
                _legacy_decision_regret([d.members for d in test_d], args.max_degree)
            )
        if model_r:
            print("  held-out decision regret, competitive main loops:")
            print(f"    previous rule {_summary(np.concatenate(legacy_r))}")
            print(f"    cost model    {_summary(np.concatenate(model_r))}")
        else:
            print("  no fold had decisions on both sides: nothing held out to score")
        x["fill_bytes_per_latency"] = float(fit_setup(decisions))
    else:
        # The setup cost only shows between reductions of one main loop, so
        # without such pairs it cannot be fitted: say so rather than print a
        # literal that silently carries the seed value.
        print(
            "WARNING: no main loop has both a packed-atomic and a workspace binary "
            "timed (sweep with --two-stage always), so fill_bytes_per_latency is "
            f"NOT fitted and keeps its seed value {x['fill_bytes_per_latency']:.6g}",
            file=sys.stderr,
        )

    fitted = _model_of(x, args.max_degree)
    # The vectorised stage-1 copy must choose what the production model chooses.
    vec = _picks(c, _predict_curves(c, _x_of(fitted)))
    for r, k in enumerate(c.rows):
        feasible = [int(d) for d, ok in zip(c.degrees, c.feasible[r]) if ok]
        scalar = min(
            feasible,
            key=lambda d: (
                predict_split_k_cost(k.problem, k.kernel, d, k.reduction, fitted),
                d,
            ),
        )
        if scalar != c.degrees[vec[r]]:
            print(
                f"model mismatch on {k.label}: scalar {scalar} vs vectorised "
                f"{c.degrees[vec[r]]}",
                file=sys.stderr,
            )
            return 1
    print(f"\n{args.arch!r}: {fitted!r}")
    return 0


# ------------------------------------------------------------ ladder margin


def cmd_margin(args: argparse.Namespace) -> int:
    """Measured effect of :attr:`SplitKCostModel.ladder_margin`.

    For every split-K binary timed at both its best ladder degree (1 or a power
    of two) and the model's unrestricted pick, each margin chooses the pick only
    when the model finds it cheaper than the ladder by more than the margin.
    Reported against always choosing the ladder: per binary, and per shape (the
    best binary at its chosen degree). The recommendation is the margin with
    the best shape geomean whose worst shape loses at most ``--max-worse``.
    """
    from rocke.helpers.split_k_planner import select_split_k_plan, split_k_cost_model

    cost = split_k_cost_model(args.arch)
    if cost is None:
        print(f"no cost model for {args.arch!r}", file=sys.stderr)
        return 1
    exact = replace(cost, ladder_margin=0.0)
    rows, untimed = [], 0
    for k in _load(args.logs):
        if k.reduction is None:
            continue
        try:
            plan = select_split_k_plan(
                k.problem, k.kernel, [k.reduction], exact, allow_unsplit=False
            )
        except ValueError:
            continue
        pred = {d: t for d, _, t in plan.candidates}
        ladder = min((t, d) for d, t in pred.items() if d & (d - 1) == 0)[1]
        pick = plan.split_k
        if ladder not in k.times or pick not in k.times:
            untimed += 1
            continue
        rows.append((k, ladder, pick, pred[ladder] / pred[pick]))
    if not rows:
        print("no binary timed at both its ladder degree and its pick", file=sys.stderr)
        return 1
    print(
        f"{len(rows)} split-K binaries over {len({k.cmd for k, *_ in rows})} shapes "
        f"({untimed} skipped: a degree untimed); {sum(p != l for _, l, p, _ in rows)} "
        "pick off the ladder"
    )

    def geo(v):
        return math.exp(float(np.mean(np.log(v))))

    best = None
    for m in args.margins:
        per_bin, shapes = [], {}
        for k, ladder, pick, gain in rows:
            d = pick if gain > 1.0 + m else ladder
            per_bin.append(k.times[d] / k.times[ladder])
            lad, got = shapes.get(k.cmd, (math.inf, math.inf))
            shapes[k.cmd] = (min(lad, k.times[ladder]), min(got, k.times[d]))
        per_shape = [g / l for l, g in shapes.values()]
        worst = max(per_shape)
        print(
            f"  margin {m:5.3f}: binaries geo {geo(per_bin):.4f} "
            f"(>1% slower {np.mean(np.array(per_bin) > 1.01):4.0%}) | shapes geo "
            f"{geo(per_shape):.4f} best {min(per_shape):.3f} worst {worst:.3f}"
        )
        if worst <= 1.0 + args.max_worse and (
            best is None or geo(per_shape) < best[1] - 1e-9
        ):
            best = (m, geo(per_shape))
    if best is None:
        print(f"no margin keeps every shape within {args.max_worse:.1%}")
        return 1
    print(f"\nrecommended ladder_margin={best[0]} (shapes geo {best[1]:.4f})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep", help="time every split-K degree per shape")
    sw.add_argument("--shapes", required=True, help="file of MIOpenDriver commands")
    sw.add_argument("--cache", required=True, help="AOT kernel cache directory")
    sw.add_argument("--out", required=True, help="directory for the per-shape logs")
    sw.add_argument("--arch", default="gfx950")
    sw.add_argument("--max-degree", type=int, default=1024)
    sw.add_argument(
        "--degrees",
        type=_parse_degrees,
        default=None,
        help="degrees to time instead of the powers of two up to --max-degree, "
        "e.g. 2-160,192,256 (the off-ladder sweep `margin` reads)",
    )
    sw.add_argument(
        "--prune",
        type=float,
        default=20.0,
        help="--split-k-prune for the benchmark (percent)",
    )
    sw.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="extra benchmark arguments after '--'",
    )
    ft = sub.add_parser("fit", help="fit the cost model to sweep logs")
    ft.add_argument("--logs", nargs="+", required=True)
    ft.add_argument(
        "--fill-logs",
        nargs="*",
        default=[],
        help="logs re-timing degrees the sweep skipped; they only fill holes",
    )
    ft.add_argument("--arch", default="gfx950")
    ft.add_argument("--max-degree", type=int, default=1024)
    ft.add_argument(
        "--num-cus",
        type=int,
        default=None,
        help="CU count of --arch (default: the static table in rocke.helpers.split_k)",
    )
    ft.add_argument(
        "--max-waves-per-cu",
        type=int,
        default=32,
        help="hardware wave slots per CU, for an arch without a fitted model",
    )
    ft.add_argument("--competitive", type=float, default=1.25)
    ft.add_argument("--holdout", type=int, default=4)
    ft.add_argument(
        "--cv",
        action="store_true",
        help="cross-validate stage 1 over all --holdout folds (one search per fold)",
    )
    ft.add_argument("--iters", type=int, default=3000)
    ft.add_argument("--seed", type=int, default=0)
    mg = sub.add_parser(
        "margin", help="measure the ladder margin on off-ladder sweep logs"
    )
    mg.add_argument("--logs", nargs="+", required=True)
    mg.add_argument("--arch", default="gfx950")
    mg.add_argument(
        "--margins",
        type=lambda v: [float(x) for x in v.split(",")],
        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    )
    mg.add_argument(
        "--max-worse",
        type=float,
        default=0.01,
        help="largest per-shape slowdown against the ladder the margin may allow",
    )
    args = ap.parse_args(argv)
    if args.cmd == "margin":
        return cmd_margin(args)
    if args.cmd == "sweep":
        args.bench_args = [a for a in args.bench_args if a != "--"]
        return cmd_sweep(args)
    return cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
