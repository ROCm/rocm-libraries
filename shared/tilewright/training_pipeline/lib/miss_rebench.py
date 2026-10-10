# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Measure the misses of a deployed evaluation instead of guessing them.

A miss (lib/evaluate) is a pick the runtime runs without a timed row of its
own: a skip-screened kernel, or one off the GEMM's planned list. The stages
that decide (stage04b's cell decisions, stage06's deploy gate) and stage08's
report charge it by their configured rule. With `MissBench` they evaluate,
collect the misses, re-bench each missed GEMM (lib/rebench.py
`rebench_picks`: the pick, the GEMM's timed winner and up to two more timed
kernels in one bench line, in the config's kernel order rotated over the
runs, on the run's devices and bench, as few stage03 blocks as the lines
per GPU need) and evaluate again with
`pick_measurement="rebench"`, so a miss costs its measured ratio to the
winner (`measured_evaluation`).

An off-list pick has no row at all; its re-bench line lists it explicitly.
When the bench cannot run it (its predicates fail) or never times it with
the winner, the GEMM gets no ratio and stays a miss under the fallback of
the "rebench" rule; such GEMMs are counted (`n_rebench_failed`).

A GEMM of another execution context (cu_budget > 0 or another schedule) is
re-benched in that context: through the wrapper of lib/contexts.py (the
bench with `--sm_count_target`, the hybrid assignment policy, and
`hipblaslt-cotenant` holding the other CUs for a cotenant context), the
context taken from the config's `contexts` entries with the GEMM's budget
and schedule. A context GEMM whose context the config does not list, or
whose cotenant cannot run, is not re-benched and is counted
(`n_context_skipped`).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import contexts as ctxlib
from . import evaluate as ev
from . import gpu_lease
from . import kernel_identity as ki
from . import rebench as rb
from .bench_yaml import (
    kernel_order_seed,
    read_bench_lines,
    render_bench_line,
    shape_of_line,
)
from .shapes import Shape

CONTEXT_BENCH = Path(__file__).resolve().parent / "context_bench.py"
RECORD_FILE = "rebench_misses.json"


@dataclass(frozen=True)
class Miss:
    """A miss of one evaluation: the GEMM (its index in the evaluated list),
    the pick and why it has no timing of its own."""

    index: int
    pick: int
    reason: str


def find_misses(
    evaluation: ev.Evaluation, gemms: Sequence[Mapping[str, Any]]
) -> List[Miss]:
    """The model-served picks of `evaluation` without a timed row of their
    own; "screened" when the GEMM has a skip row of the pick, else
    "not_benched"."""
    out: List[Miss] = []
    for r in evaluation.results:
        if r.served_by != "model" or r.pick_sol_idx is None:
            continue
        timed, screened = ev.measurements(gemms[r.index])
        sid = int(r.pick_sol_idx)
        if sid in timed:
            continue
        out.append(Miss(r.index, sid, "screened" if sid in screened else "not_benched"))
    return out


def read_lines(paths: Sequence[Path]) -> Dict[rb.Shape, str]:
    """{shape: bench line} of bench yamls, the first line of a shape
    winning; missing files are skipped."""
    out: Dict[rb.Shape, str] = {}
    for p in paths:
        if not Path(p).is_file():
            continue
        for line in read_bench_lines(Path(p)):
            shape = shape_of_line(line)
            if shape is not None:
                out.setdefault(shape, line)
    return out


@dataclass
class MissBench:
    """How a stage re-benches its misses: the bench and devices, the stage03
    options of the run (config, watchdog, pool-only library), the bench
    lines of the benched shapes (else a line is rendered from the config),
    what a context bench needs (device CUs, cotenant, PATH additions) and
    the seed of the lines' kernel order (`rebench.run_order`; None: as
    listed). `measure` benches under the lease of the devices' GPUs
    (lib/gpu_lease.py; LeaseError when another process tree holds one)."""

    bench_binary: Path
    work_dir: Path
    cfg: Mapping[str, Any]
    devices: str = "0"
    runs: int = rb.DEFAULT_RUNS
    stage03_args: Sequence[str] = ()
    lines: Mapping[rb.Shape, str] = field(default_factory=dict)
    n_cu: Optional[int] = None
    cotenant: Optional[Path] = None
    path_append: Sequence[str] = ()
    quiet: bool = True
    order_seed: Optional[int] = None

    def line_for(self, g: Mapping[str, Any]) -> Optional[str]:
        shape = _shape(g)
        line = self.lines.get(shape)
        if line is not None:
            return line
        if not isinstance(self.cfg.get("hipblaslt"), Mapping):
            return None
        return render_bench_line(Shape(*shape), dict(self.cfg))

    def _context(self, g: Mapping[str, Any]) -> Tuple[Optional[Any], str]:
        """(the BenchContext of a non-exclusive GEMM, or None, and why
        not)."""
        c = ev.context_of(g)
        for ctx in ctxlib.contexts_from_config(self.cfg, self.n_cu):
            if (ctx.cu_budget, ctx.schedule) != (c.cu_budget, c.schedule):
                continue
            if ctx.cotenant and (self.cotenant is None or self.n_cu is None):
                return None, "cotenant_unavailable"
            return ctx, ""
        return None, "context_not_configured"

    def _bench_binary(self, ctx: Any, where: Path) -> Path:
        if ctx is None:
            return Path(self.bench_binary)
        spec = ctxlib.bench_spec(
            ctx,
            int(self.n_cu or 0),
            str(self.bench_binary),
            str(self.cotenant) if self.cotenant else None,
            str(where / "cotenant_logs"),
            path_append=self.path_append,
        )
        return ctxlib.write_wrapper(where, spec, sys.executable, CONTEXT_BENCH)

    def measure(
        self, gemms: Sequence[Dict[str, Any]], misses: Sequence[Miss], tag: str
    ) -> Dict[str, Any]:
        """Re-bench `misses` of `gemms` and attach the ratios
        (`rebench_picks`, `attach_rebench`); a miss whose GEMM already
        carries a ratio of the pick is reused. Returns the record of what
        was measured."""
        t0 = time.time()
        rows = [_miss_row(gemms[m.index], m) for m in misses]
        groups: Dict[Tuple[int, str], List[int]] = {}
        contexts: Dict[Tuple[int, str], Any] = {}
        for pos, m in enumerate(misses):
            g, row = gemms[m.index], rows[pos]
            if m.pick in (g.get("rebench") or {}):
                row.update(status="reused", ratio=float(g["rebench"][m.pick]))
                continue
            c = ev.context_of(g)
            key = (c.cu_budget, c.schedule)
            if not c.is_exclusive():
                ctx, why = self._context(g)
                if ctx is None:
                    row["status"] = why
                    continue
                contexts[key] = ctx
            if self.line_for(g) is None:
                row["status"] = "no_bench_line"
                continue
            groups.setdefault(key, []).append(pos)
        rcs: List[int] = []
        devices = [int(d) for d in str(self.devices).split(",") if d.strip()]
        with gpu_lease.hold(
            devices if groups else [], f"miss re-bench {tag}", self.work_dir
        ):
            for key, members in groups.items():
                ctx = contexts.get(key)
                label = ctx.label if ctx else "exclusive"
                where = Path(self.work_dir) / tag / label
                run = rb.stage03_runner(
                    self._bench_binary(ctx, where),
                    where,
                    devices=self.devices,
                    extra_args=self.stage03_args,
                    returncodes=rcs,
                )
                group = [gemms[misses[pos].index] for pos in members]
                picks = {_shape(g): misses[pos].pick for pos, g in zip(members, group)}
                lines = {_shape(g): str(self.line_for(g)) for g in group}
                t = time.time()
                ratios, _ = rb.rebench_picks(
                    group,
                    picks,
                    lines,
                    run,
                    runs=self.runs,
                    order_seed=self.order_seed,
                )
                rb.attach_rebench(group, ratios)
                if not self.quiet:
                    print(
                        f"[rebench] {tag} {label}: {len(ratios)} of {len(members)} "
                        f"misses measured in {time.time() - t:.1f}s",
                        flush=True,
                    )
                for pos, g in zip(members, group):
                    pr = ratios.get(_shape(g))
                    if pr is None:
                        rows[pos]["status"] = "failed"
                        continue
                    rows[pos].update(
                        status="measured",
                        ratio=pr.ratio,
                        pick_us=pr.pick_us,
                        winner_us=pr.winner_us,
                        winner_sol=pr.winner_sol,
                        n_runs=pr.n_runs,
                    )
        wall = time.time() - t0
        n_benched = sum(len(v) for v in groups.values())
        status = [r["status"] for r in rows]
        return {
            "runs": int(self.runs),
            "n_misses": len(rows),
            "n_misses_by_reason": {
                why: sum(1 for m in misses if m.reason == why)
                for why in ev.MISS_REASONS
            },
            "n_rebenched": status.count("measured"),
            "n_rebench_reused": status.count("reused"),
            "n_rebench_failed": status.count("failed") + status.count("no_bench_line"),
            "n_context_skipped": status.count("context_not_configured")
            + status.count("cotenant_unavailable"),
            "n_benched_gemms": n_benched,
            "bench_rc": sorted(set(rcs)),
            "wall_s": wall,
            "s_per_benched_gemm": wall / n_benched if n_benched else None,
            "gemms": rows,
        }


def _shape(g: Mapping[str, Any]) -> rb.Shape:
    return (int(g["m"]), int(g["n"]), int(g["k"]), int(g.get("batch_count", 1)))


def _miss_row(g: Mapping[str, Any], m: Miss) -> Dict[str, Any]:
    c = ev.context_of(g)
    m_, n, k, b = _shape(g)
    return {
        "m": m_,
        "n": n,
        "k": k,
        "batch": b,
        "cu_budget": c.cu_budget,
        "schedule": c.schedule,
        "pick": m.pick,
        "reason": m.reason,
        "status": None,
        "ratio": None,
    }


# ── evaluation with measured misses ─────────────────────────────────────────


@dataclass
class Measured:
    """An evaluation whose misses were re-benched (`evaluation`, under
    "rebench"), the one under the configured rule (`pre`), and the
    re-bench record; without a MissBench `evaluation` is `pre` and `record`
    is None."""

    evaluation: ev.Evaluation
    pre: ev.Evaluation
    record: Optional[Dict[str, Any]] = None


def measured_evaluation(
    evaluate: Callable[[str], ev.Evaluation],
    gemms: Sequence[Dict[str, Any]],
    bench: Optional[MissBench],
    pick_measurement: str,
    tag: str,
) -> Measured:
    """`evaluate(pick_measurement)`, then with `bench` its misses re-benched
    and `evaluate("rebench")`. Under "any" an off-list kernel is passed
    over, so the misses come from an evaluation under "rebench" without
    ratios (the picks of "timed" and "warmup")."""
    pre = evaluate(pick_measurement)
    if bench is None:
        return Measured(pre, pre)
    probe = pre if pick_measurement != "any" else evaluate(ev.REBENCH_PICK_MEASUREMENT)
    misses = find_misses(probe, gemms)
    record = bench.measure(gemms, misses, tag)
    post = evaluate(ev.REBENCH_PICK_MEASUREMENT) if misses else _relabel(pre)
    return Measured(post, pre, record)


def _relabel(e: ev.Evaluation) -> ev.Evaluation:
    return ev.Evaluation(
        results=e.results,
        weight_dtype=e.weight_dtype,
        min_scored=e.min_scored,
        n_pool_configs=e.n_pool_configs,
        n_routing_mismatch=e.n_routing_mismatch,
        pick_measurement=ev.REBENCH_PICK_MEASUREMENT,
    )


def pre_rebench(summary: Mapping[str, Any], rule: str) -> Dict[str, Any]:
    """The configured rule's headline numbers next to the re-benched ones."""
    se = summary.get("sel_eff")
    return {
        "pick_measurement": rule,
        "sel_eff": se if isinstance(se, float) and math.isfinite(se) else None,
        "n_miss": summary.get("n_miss"),
        "n_miss_by_reason": summary.get("n_miss_by_reason"),
    }


def record_counts(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The counts of a re-bench record, without its per-GEMM rows."""
    return {k: v for k, v in (record or {}).items() if k != "gemms"}


def named_record(
    record: Optional[Mapping[str, Any]], identity: Optional[ki.LibraryIdentity]
) -> Dict[str, Any]:
    """A re-bench record (`MissBench.measure`) as the stages keep it: the
    digest of the library its solution indices name and, next to each
    GEMM's pick and re-benched winner, their signature and kernel id
    (`lib.kernel_identity.naming`)."""
    out = dict(record or {}, library_digest=identity.digest if identity else None)
    if "gemms" not in out:
        return out
    rows = []
    for row in out["gemms"]:
        named: Dict[str, Any] = {}
        for key, value in row.items():
            named[key] = value
            if key == "pick":
                named.update(ki.naming(identity, value, "pick_"))
            elif key == "winner_sol":
                named.update(ki.naming(identity, value, "winner_"))
        rows.append(named)
    out["gemms"] = rows
    return out


OFF_LIST_RECORD = "held_out"


def off_list_charge(
    records: Optional[Mapping[str, Any]],
    *,
    min_misses: int,
    fallback: Optional[float] = None,
) -> Tuple[ev.OffListCharge, Dict[str, Any]]:
    """What stage05 charges an off-list pick, from stage04b's `RECORD_FILE`
    (its `held_out` record: the previous model's misses on the round's new
    GEMMs, re-benched): the geomean of min(1, ratio), as the rebench rule
    charges a pick, over the measured off-list misses, and per (cu_budget,
    schedule) for each context with at least `min_misses` of them when they
    come from more than one. With fewer than `min_misses` overall (or no
    record) the charge is `fallback`. Returns (charge, report)."""
    rows = ((records or {}).get(OFF_LIST_RECORD) or {}).get("gemms") or ()
    by: Dict[Tuple[int, str], List[float]] = {}
    for g in rows:
        ratio = g.get("ratio")
        if g.get("reason") != "not_benched" or g.get("status") != "measured":
            continue
        if not ratio or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            continue
        c = ev.context_of(g)
        by.setdefault((c.cu_budget, c.schedule), []).append(min(1.0, float(ratio)))
    every = [x for xs in by.values() for x in xs]
    label = lambda key: f"{key[0]}/{key[1]}"  # noqa: E731
    report: Dict[str, Any] = {
        "n": len(every),
        "n_by_context": {label(k): len(v) for k, v in sorted(by.items())},
        "min_misses": int(min_misses),
        "fallback": fallback,
    }
    if len(every) < int(min_misses):
        report.update(source="fallback", sel_eff=fallback, by_context={})
        return ev.OffListCharge(fallback), report
    per = (
        {k: ev.geomean(v) for k, v in by.items() if len(v) >= int(min_misses)}
        if len(by) > 1
        else {}
    )
    charge = ev.OffListCharge(ev.geomean(every), per)
    report.update(
        source="rebench",
        sel_eff=charge.sel_eff,
        by_context={label(k): v for k, v in sorted(per.items())},
    )
    return charge, report


# ── stage options ────────────────────────────────────────────────────────────


def add_arguments(ap: argparse.ArgumentParser, *, bench: bool = True) -> None:
    """The stage options of a miss re-bench; `bench` adds --bench-binary and
    --devices (for stages that have none)."""
    g = ap.add_argument_group("miss re-bench (validate.rebench_misses)")
    g.add_argument(
        "--rebench-misses",
        action="store_true",
        help="re-bench every miss (pick, winner and up to two timed kernels) "
        "and charge it by the measured ratio (lib/miss_rebench.py)",
    )
    g.add_argument("--rebench-runs", type=int, default=rb.DEFAULT_RUNS)
    g.add_argument(
        "--rebench-lines",
        type=Path,
        action="append",
        default=[],
        help="bench yaml of the benched shapes (repeatable); a shape without "
        "a line gets one rendered from --config-yaml",
    )
    if bench:
        g.add_argument("--bench-binary", type=Path, default=None)
        g.add_argument("--devices", default="0")
    g.add_argument("--cotenant-binary", type=Path, default=None)
    g.add_argument("--path-append", action="append", default=[])
    g.add_argument("--stall-timeout-s", type=float, default=None)
    g.add_argument("--startup-grace-s", type=float, default=None)
    g.add_argument(
        "--pool-only-library",
        action="append",
        default=[],
        metavar="STEM",
        help="bench against a pool-only library directory of these stems "
        "(with --library-dir), as the run's stage03",
    )


def from_args(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    work_dir: Path,
    n_cu: Optional[int],
) -> Optional[MissBench]:
    """The MissBench of the stage options, None without --rebench-misses;
    ValueError when the options cannot bench."""
    if not getattr(args, "rebench_misses", False):
        return None
    if args.bench_binary is None or not Path(args.bench_binary).exists():
        raise ValueError("--rebench-misses needs an existing --bench-binary")
    if int(args.rebench_runs) < 1:
        raise ValueError("--rebench-runs must be at least 1")
    extra: List[str] = []
    if getattr(args, "config_yaml", None) is not None:
        extra += ["--config-yaml", str(args.config_yaml)]
    for flag, value in (
        ("--stall-timeout-s", args.stall_timeout_s),
        ("--startup-grace-s", args.startup_grace_s),
    ):
        if value is not None:
            extra += [flag, str(value)]
    if args.pool_only_library:
        if getattr(args, "library_dir", None) is None:
            raise ValueError("--pool-only-library needs --library-dir")
        extra += ["--library-dir", str(args.library_dir)]
        for stem in args.pool_only_library:
            extra += ["--pool-only-library", stem]
    cotenant = args.cotenant_binary or Path(args.bench_binary).parent / (
        "hipblaslt-cotenant"
    )
    return MissBench(
        bench_binary=Path(args.bench_binary),
        work_dir=Path(work_dir),
        cfg=cfg,
        devices=str(args.devices),
        runs=int(args.rebench_runs),
        stage03_args=extra,
        lines=read_lines(args.rebench_lines),
        n_cu=n_cu,
        cotenant=cotenant if cotenant.exists() else None,
        path_append=list(args.path_append),
        quiet=bool(getattr(args, "quiet", False)),
        order_seed=kernel_order_seed(cfg),
    )
