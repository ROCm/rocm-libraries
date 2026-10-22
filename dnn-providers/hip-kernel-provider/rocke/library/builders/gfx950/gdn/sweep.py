#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tiles x knobs sweep of the gfx950 GDN/KDA decode kernel (runbook Step 0).

``tune.py`` times each registered tile at its base configuration only. This
tool walks the whole tuning space dispatch registers for a request -- every
legal tile x knob set x ``waves_per_eu`` value (``dispatch_gdn_decode_all`` at
the chosen sweep level) -- so a case's ``auto`` row in
``dispatch.gdn.axes.AUTO_DEFAULTS`` can be regenerated from measurements
(ALGORITHM.md §8, follow-up 10).

Three subcommands, so the space can be split across devices and jobs:

``run``
    Compile every configuration of one shard (``--shard I/N``: configurations
    ``I, I+N, ...`` in registry order) in a process pool (``--jobs``), then on
    the device: gate each one against the fp32 reference (output and written
    state within tolerance, every other pool page bit-unchanged) and time it
    at each batch. One JSON line per (configuration, batch) goes to
    ``--jsonl``.
``merge``
    Read the shards' JSON lines and rank the configurations per cell, and
    across the cell's batches by geometric mean of the ratio to each batch's
    fastest configuration. Prints ratios only.
``retime``
    Re-time the top configurations of a merged ranking (plus ``auto``)
    interleaved over several rounds, so the final pick is not a single-sample
    winner.

Cold memory (``--cold C``): every timed launch reads and writes its own
state slots. ``C`` copies of the batch's read and write slots are rotated over
the captured launches, so consecutive launches touch disjoint state and the
rotation covers ``2 * C * batch`` slots; size ``C`` so that footprint is well
above the last-level cache. ``--cold 0`` re-reads one slot set (warm).

Absolute times are written only to the ``--jsonl`` file the caller names;
``merge`` and ``retime`` print ratios.

Example (one shard of three, f32 state, Hk 16 / Hv 48)::

    PYTHONPATH=<rocke>/library:<rocke>/platform/python python3 -m \\
        builders.gfx950.gdn.sweep run --geometries 16/48 --state-dtype f32 \\
        --batches 1,2,3,4 --cold 64 --level full --shard 0/3 --jobs 12 \\
        --jsonl sweep0.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

from dispatch.gdn import (
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
)
from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode
from rocke.dispatch.tuning import sweep_level

ARCH = "gfx950"
# Relative bound on the written f32 state against the fp32 reference: the
# kernel and the reference differ only in summation order, so anything above
# this is a defect, not rounding. 16-bit states keep the builder's TOL.
F32_STATE_RTOL = 1e-4


# --------------------------------------------------------------------- space
def parse_shard(text: str) -> Tuple[int, int]:
    """``"I/N"`` -> ``(I, N)`` with ``0 <= I < N``."""
    i, n = (int(x) for x in text.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"bad shard {text!r}: need I/N with 0 <= I < N")
    return i, n


def parse_geometries(text: str) -> List[Tuple[int, int]]:
    return [tuple(int(v) for v in item.split("/")) for item in text.split(",")]


def request_for(args, num_k_heads: int, num_v_heads: int, batch: int):
    return GdnDecodeRequest(
        batch=batch,
        arch=ARCH,
        gate_kind=args.gate_kind,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=args.head_k_dim,
        head_v_dim=args.head_v_dim,
        state_dtype=args.state_dtype,
        fuse_conv=args.fuse_conv,
        fuse_out_norm=args.fuse_out_norm,
        num_tokens=getattr(args, "num_tokens", 1),
        norm_gate_act=getattr(args, "norm_gate_act", "auto"),
    )


def space(req: GdnDecodeRequest, level: str):
    """Every registered configuration for ``req`` at ``level``, registry order.

    The kernel set does not depend on the batch (it only sizes the grid), so
    one listing serves every batch of a cell.
    """
    with sweep_level(level):
        return dispatch_gdn_decode_all(req)


def shard_of(items: Sequence, shard: Tuple[int, int]) -> List:
    i, n = shard
    return [item for k, item in enumerate(items) if k % n == i]


def record_key(result) -> Dict[str, object]:
    """What names one configuration in a JSON line (no measurement)."""
    spec = result.spec
    k = spec.kernel_spec
    return {
        "spec_id": result.candidate.spec_id,
        "tuning_id": spec.tuning_id,
        "knobs": {name: value for name, value in spec.knobs},
        "tile": [k.num_warps, k.warp_threads_k, k.blocks_per_v_dim],
        "waves_per_eu": k.waves_per_eu,
        "kernel_name": k.kernel_name(),
    }


# ------------------------------------------------------------------- compile
def _compile_one(spec: GdnDecodeSpec):
    """Worker: lower and compile one spec; ``(name, hsaco or None, error)``."""
    from rocke.helpers.compile import compile_kernel

    try:
        art = compile_kernel(build_gdn_decode(spec, arch=ARCH), arch=ARCH)
        return spec.kernel_name(), art.hsaco, art.kernel_name, ""
    except Exception as exc:  # reported per configuration, never fatal
        return spec.kernel_name(), None, None, f"{type(exc).__name__}: {exc}"


def compile_all(specs: Iterable[GdnDecodeSpec], jobs: int):
    """``kernel_name -> (hsaco, symbol, error)`` for every distinct spec."""
    unique = {}
    for spec in specs:
        unique.setdefault(spec.kernel_name(), spec)
    out = {}
    if jobs <= 1:
        for spec in unique.values():
            name, hsaco, sym, err = _compile_one(spec)
            out[name] = (hsaco, sym, err)
        return out
    # spawn, not fork: the parent may already hold an initialized HIP runtime.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        for name, hsaco, sym, err in pool.map(
            _compile_one, list(unique.values()), chunksize=4
        ):
            out[name] = (hsaco, sym, err)
    return out


# -------------------------------------------------------------------- device
def _device():
    """The ROCm-only measurement backend, imported on first use."""
    import torch

    from builders.gfx950.gdn import gdn_decode as host

    return torch, host


class Cell:
    """Inputs, reference and cold-rotation slot sets of one (geometry, batch)."""

    def __init__(self, base: GdnDecodeSpec, batch: int, copies: int, seed: int = 0):
        torch, host = _device()
        self.batch = batch
        self.copies = max(1, copies)
        T = base.num_tokens
        per = batch * (1 + T)  # one read and T snapshot slots per sequence
        depth = per * self.copies + 1
        self.inp = host.make_inputs(base, batch, seed=seed, pool_depth=depth)
        self.ref_out, self.ref_state = host.ref_fp32(base, self.inp)
        self.before = self.inp["state"].clone()
        self.written = self.inp["write_indices"].long()
        self.untouched = torch.ones(depth, dtype=torch.bool, device=self.written.device)
        self.untouched[self.written] = False
        dev = self.written.device
        ar = torch.arange(batch, dtype=torch.int32, device=dev)
        wr = self.inp["write_indices"]
        # Copy c reads slots [per c, per c + b) and writes the next b * T;
        # copy 0 is make_inputs' own layout.
        self.slots = [(ar + per * c, wr + per * c) for c in range(self.copies)]


def gate(spec: GdnDecodeSpec, cell: Cell, launcher) -> Tuple[float, float, float]:
    """``(out_err, state_err, out_rel)`` of one launch against the fp32
    reference: ``out_err`` the builder's ``out_error`` (bound ``TOL``),
    ``out_rel`` its per-row relative error (``out_rel_error``, bound
    ``OUT_RTOL``). An untouched pool page that changed by one bit makes
    ``state_err`` infinite."""
    torch, host = _device()
    values, cfg = host.prepare(spec, cell.inp, cell.batch)
    host.launch(launcher, values, cfg)
    host.drain()
    out_err = host.out_error(spec, values["out"], cell.ref_out)
    out_rel = host.out_rel_error(values["out"], cell.ref_out)
    got = values["state"].float()[cell.written]
    state_err = (got - cell.ref_state).abs().max().item()
    if spec.state_dtype == "f32":
        state_err /= max(cell.ref_state.abs().max().item(), 1e-3)
    if cell.untouched.any() and bool(
        (values["state"][cell.untouched] != cell.before[cell.untouched]).any()
    ):
        state_err = math.inf
    return out_err, state_err, out_rel


def passes(
    spec: GdnDecodeSpec, out_err: float, state_err: float, out_rel: float
) -> bool:
    _, host = _device()
    state_tol = F32_STATE_RTOL if spec.state_dtype == "f32" else host.TOL
    return out_err <= host.TOL and out_rel <= host.OUT_RTOL and state_err <= state_tol


def time_launches(spec, cell: Cell, launcher, *, reps: int = 64, samples: int = 20):
    """Per-launch device time (min over ``samples`` graph replays) with the
    cold rotation, or ``None`` if graph capture fails."""
    torch, host = _device()
    values, cfg = host.prepare(spec, cell.inp, cell.batch)
    rot = []
    for rd, wr in cell.slots:
        v = dict(values)
        v["read_indices"], v["write_indices"] = rd, wr
        rot.append(v)
    for k in range(2 * len(rot)):
        host.launch(launcher, rot[k % len(rot)], cfg)
    torch.cuda.synchronize()
    n = max(reps, len(rot))
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for k in range(n):
                host.launch(launcher, rot[k % len(rot)], cfg)
    except Exception:
        host.drain()
        return None
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    best = math.inf
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / n)
    del graph
    host.drain()
    return best


# ---------------------------------------------------------------------- run
def cmd_run(args) -> int:
    torch, host = _device()
    if not torch.cuda.is_available():
        print("no HIP device visible", file=sys.stderr)
        return 2
    from rocke.runtime.launcher import KernelLauncher
    from kernels.gfx950.gdn_decode import gdn_decode_signature

    shard = parse_shard(args.shard)
    batches = [int(b) for b in args.batches.split(",")]
    out = open(args.jsonl, "a", encoding="utf-8")
    failed = 0
    for hk, hv in parse_geometries(args.geometries):
        results = space(request_for(args, hk, hv, batches[0]), args.level)
        mine = shard_of(results, shard)
        if args.limit:
            mine = mine[: args.limit]
        auto_name = dispatch_gdn_decode(
            request_for(args, hk, hv, batches[0])
        ).spec.kernel_spec.kernel_name()
        print(
            f"Hk{hk}/Hv{hv} {args.state_dtype}: {len(results)} configurations, "
            f"shard {args.shard} has {len(mine)}; compiling with {args.jobs} jobs",
            flush=True,
        )
        arts = compile_all((r.spec.kernel_spec for r in mine), args.jobs)
        cells = (
            {
                b: Cell(mine[0].spec.kernel_spec if mine else None, b, args.cold)
                for b in batches
            }
            if mine
            else {}
        )
        for k, result in enumerate(mine):
            spec = result.spec.kernel_spec
            key = record_key(result)
            hsaco, sym, err = arts[spec.kernel_name()]
            common = {
                "geometry": [hk, hv],
                "head_dims": [args.head_k_dim, args.head_v_dim],
                "gate_kind": args.gate_kind,
                "state_dtype": args.state_dtype,
                "fuse_conv": args.fuse_conv,
                "fuse_out_norm": args.fuse_out_norm,
                "num_tokens": args.num_tokens,
                "norm_gate_act": args.norm_gate_act,
                "cold": args.cold,
                "level": args.level,
                "auto": key["kernel_name"] == auto_name,
                **key,
            }
            if hsaco is None:
                failed += 1
                out.write(
                    json.dumps({**common, "status": "compile", "error": err}) + "\n"
                )
                continue
            launcher = KernelLauncher(
                hsaco=hsaco, kernel_name=sym, signature=gdn_decode_signature(spec)
            )
            for b in batches:
                cell = cells[b]
                oe, se, orel = gate(spec, cell, launcher)
                rec = {
                    **common,
                    "batch": b,
                    "out_err": oe,
                    "out_rel": orel,
                    "state_err": se,
                }
                if not passes(spec, oe, se, orel):
                    failed += 1
                    rec["status"] = "incorrect"
                else:
                    us = time_launches(spec, cell, launcher, reps=args.reps)
                    rec["status"] = "ok" if us is not None else "nocapture"
                    rec["us"] = us
                out.write(json.dumps(rec) + "\n")
            out.flush()
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(mine)} done", flush=True)
    out.close()
    print(f"run done; {failed} configuration-cells failed to compile or gate")
    return 0


# -------------------------------------------------------------------- merge
def load(paths: Sequence[str]) -> List[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def cell_of(rec: dict) -> Tuple:
    return (
        tuple(rec["geometry"]),
        tuple(rec["head_dims"]),
        rec["gate_kind"],
        rec["state_dtype"],
        rec["fuse_conv"],
        rec["fuse_out_norm"],
        rec["cold"],
        rec.get("num_tokens", 1),
        rec.get("norm_gate_act", "auto"),
    )


def rank(rows: List[dict]):
    """Per case (cell without batch): configurations ranked by the geometric
    mean over the case's batches of ``us / fastest us`` at that batch; a
    configuration missing a batch is dropped. Returns
    ``{case: [(geomean, worst, kernel_name, rec_by_batch), ...]}``."""
    by_case = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("status") == "ok" and r.get("us"):
            by_case[cell_of(r)][r["kernel_name"]][r["batch"]] = r
    ranked = {}
    for case, confs in by_case.items():
        batches = sorted({b for per in confs.values() for b in per})
        best = {
            b: min(per[b]["us"] for per in confs.values() if b in per) for b in batches
        }
        entries = []
        for name, per in confs.items():
            if any(b not in per for b in batches):
                continue
            ratios = [per[b]["us"] / best[b] for b in batches]
            gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            entries.append((gm, max(ratios), name, per))
        entries.sort(key=lambda e: (e[0], e[1]))
        ranked[case] = (batches, entries)
    return ranked


def cmd_merge(args) -> int:
    rows = load(args.inputs)
    bad = defaultdict(int)
    for r in rows:
        if r.get("status") != "ok":
            bad[r.get("status")] += 1
    if bad:
        print(
            "non-ok records: " + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
        )
    for case, (batches, entries) in rank(rows).items():
        print(f"\n=== {case}: {len(entries)} configurations x batches {batches}")
        auto = [e for e in entries if next(iter(e[3].values())).get("auto")]
        for pos, (gm, worst, name, per) in enumerate(entries[: args.top], start=1):
            rec = next(iter(per.values()))
            mark = " <- auto" if rec.get("auto") else ""
            print(
                f"  {pos:4d} geomean x{gm:.3f} worst x{worst:.3f} "
                f"{rec['spec_id']} wpe{rec['waves_per_eu']} {rec['knobs']}{mark}"
            )
        for e in auto:
            print(
                f"  auto rank {entries.index(e) + 1}/{len(entries)} geomean x{e[0]:.3f}"
            )
    return 0


# ------------------------------------------------------------------- retime
def cmd_retime(args) -> int:
    torch, host = _device()
    from rocke.runtime.launcher import KernelLauncher
    from kernels.gfx950.gdn_decode import gdn_decode_signature

    rows = load(args.inputs)
    out = open(args.jsonl, "a", encoding="utf-8")
    for case, (batches, entries) in rank(rows).items():
        picks = entries[: args.top]
        picks += [
            e for e in entries[args.top :] if next(iter(e[3].values())).get("auto")
        ]
        (hk, hv), (dk, dv), gate_kind, state_dtype, fc, fn, cold, ntok, nga = case
        ns = argparse.Namespace(
            num_tokens=ntok,
            norm_gate_act=nga,
            gate_kind=gate_kind,
            head_k_dim=dk,
            head_v_dim=dv,
            state_dtype=state_dtype,
            fuse_conv=fc,
            fuse_out_norm=fn,
        )
        names = {e[2] for e in picks}
        results = [
            r
            for r in space(request_for(ns, hk, hv, batches[0]), "full")
            if r.spec.kernel_spec.kernel_name() in names
        ]
        arts = compile_all((r.spec.kernel_spec for r in results), args.jobs)
        launchers = {}
        for r in results:
            spec = r.spec.kernel_spec
            hsaco, sym, _ = arts[spec.kernel_name()]
            launchers[spec.kernel_name()] = (
                r,
                KernelLauncher(
                    hsaco=hsaco, kernel_name=sym, signature=gdn_decode_signature(spec)
                ),
            )
        for b in batches:
            cell = Cell(results[0].spec.kernel_spec, b, cold)
            times = defaultdict(list)
            for _ in range(args.rounds):
                for name, (r, launcher) in launchers.items():
                    spec = r.spec.kernel_spec
                    if not passes(spec, *gate(spec, cell, launcher)):
                        times[name].append(math.inf)
                        continue
                    us = time_launches(spec, cell, launcher, reps=args.reps)
                    times[name].append(us if us is not None else math.inf)
            med = {n: sorted(t)[len(t) // 2] for n, t in times.items()}
            fastest = min(med.values())
            print(f"\n=== {case} batch {b}: retime x{args.rounds} rounds (median)")
            for name in sorted(med, key=med.get):
                r = launchers[name][0]
                key = record_key(r)
                mark = (
                    " <- auto"
                    if next((e[3][b].get("auto") for e in picks if e[2] == name), False)
                    else ""
                )
                print(
                    f"  x{med[name] / fastest:.3f} {key['spec_id']} "
                    f"wpe{key['waves_per_eu']} {key['knobs']}{mark}"
                )
                out.write(
                    json.dumps(
                        {
                            "case": list(map(str, case)),
                            "batch": b,
                            "median_us": med[name],
                            "rounds": times[name],
                            **key,
                        }
                    )
                    + "\n"
                )
    out.close()
    return 0


# --------------------------------------------------------------------- main
def _add_case_args(p):
    p.add_argument("--gate-kind", default="gdn", choices=("gdn", "kda"))
    p.add_argument("--geometries", default="16/32", help="Hk/Hv pairs, comma-separated")
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--state-dtype", default="bf16", choices=("bf16", "f16", "f32"))
    p.add_argument("--fuse-conv", action="store_true")
    p.add_argument("--fuse-out-norm", action="store_true")
    p.add_argument("--num-tokens", type=int, default=1, help="tokens per sequence")
    p.add_argument(
        "--norm-gate-act",
        default="auto",
        choices=("auto", "silu", "sigmoid"),
        help="fused output norm gate activation (with --fuse-out-norm)",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="compile, gate and time one shard")
    _add_case_args(run)
    run.add_argument("--batches", default="1,2,3,4")
    run.add_argument("--level", default="full", choices=("production", "full"))
    run.add_argument("--shard", default="0/1")
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    run.add_argument("--cold", type=int, default=64, help="rotated slot copies; 0 warm")
    run.add_argument("--reps", type=int, default=64, help="launches per captured graph")
    run.add_argument("--limit", type=int, default=0, help="debug: first N of the shard")
    run.add_argument("--jsonl", required=True)
    mrg = sub.add_parser("merge", help="rank the configurations of JSON-line shards")
    mrg.add_argument("inputs", nargs="+")
    mrg.add_argument("--top", type=int, default=20)
    rt = sub.add_parser("retime", help="interleaved re-time of the merged leaders")
    rt.add_argument("inputs", nargs="+")
    rt.add_argument("--top", type=int, default=12)
    rt.add_argument("--rounds", type=int, default=5)
    rt.add_argument("--reps", type=int, default=64)
    rt.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    rt.add_argument("--jsonl", required=True)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "merge": cmd_merge, "retime": cmd_retime}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
