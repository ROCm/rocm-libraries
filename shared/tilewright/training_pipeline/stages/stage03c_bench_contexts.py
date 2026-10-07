#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03c_bench_contexts -- bench the round's shapes under the configured
non-exclusive execution contexts (lib/contexts.py).

stage02 / stage03 bench the exclusive context. For every entry of the
config's `contexts.list` and `contexts.holdout` this stage

  1. picks the shapes the context benches: every shape, or
     ceil(fraction * n) per cell drawn with `contexts.seed`;
  2. writes their bench lines (the round's own lines: the same kernel list
     per shape as the exclusive bench, or the context's re-planned lines from
     --context-shapes-dir) to `<label>/shapes.yaml`;
  3. writes `<label>/hipblaslt-bench`, a wrapper that runs the bench with
     `--sm_count_target <cu_budget>` (and `--hybrid_assignment_policy` for
     the dynamic / auto schedules), inside `hipblaslt-cotenant --cus <device
     CUs - cu_budget>` for a cotenant context; the device CU count comes from
     the KFD topology (or the config's `hardware.n_cu`);
  4. runs stage02 (unless timing is adaptive) and stage03 on them with that
     wrapper as the bench binary (argv templates from --commands, `{ctx_dir}`
     standing for `<out-dir>/<label>`), so every context is calibrated and
     benched under itself.

The held-out contexts' logs are then enriched by the stage04 template of
--commands into `<out-dir>/holdout/stage04` (one --context-log-dir per
context): stage04b evaluates the model on them, stage05 never reads them. The
trained contexts' logs (`<label>/stage03/logs`) go to the round's stage04.

Outputs in --out-dir:
  <label>/                 shapes.yaml, context.json, hipblaslt-bench,
                           cotenant_logs/, stage02/, stage03/, *.log
  holdout/stage04/         enriched rows of the held-out contexts
  libdir_pool/             with --pool-only-library (lib/pool_libdir.py)
  contexts.json            per context: shapes, wrapper, sub-stage rc / time
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import hardware as hwlib  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import read_bench_lines, shape_of_line  # noqa: E402
from lib.contexts import (  # noqa: E402
    BenchContext,
    bench_spec,
    context_errors,
    contexts_from_config,
    select_shapes,
    write_wrapper,
)
from lib.fs import atomic_write_text  # noqa: E402

CONTEXT_BENCH = PIPELINE_DIR / "lib" / "context_bench.py"
TOKEN = "{ctx_dir}"
SUMMARY = "contexts.json"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="stage03c -- bench the round's shapes under each "
        "non-exclusive execution context."
    )
    ap.add_argument(
        "--shapes-yaml",
        type=Path,
        action="append",
        required=True,
        help="bench yaml of the round's shapes (repeatable; a shape's first "
        "line wins)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config-yaml", type=Path, required=True)
    ap.add_argument("--bench-binary", type=Path, required=True)
    ap.add_argument(
        "--cotenant-binary",
        type=Path,
        default=None,
        help="hipblaslt-cotenant (default: next to --bench-binary)",
    )
    ap.add_argument(
        "--hardware-device",
        type=int,
        default=0,
        help="device whose KFD topology gives the CU count",
    )
    ap.add_argument(
        "--commands",
        required=True,
        help="JSON object: stage02 (or null), stage03, stage04 -> argv "
        "(script first) with {ctx_dir} for a context's directory; stage04 "
        "(held-out contexts) gets one --context-log-dir per context appended",
    )
    ap.add_argument(
        "--context-shapes-dir",
        type=Path,
        default=None,
        help="re-planned lines per context: <dir>/<label>/shapes.yaml and "
        "plan.json, used for the contexts that have them",
    )
    ap.add_argument(
        "--pool-only-library",
        action="append",
        default=[],
        metavar="STEM",
        help="wrap the benches with a pool-only library directory built from "
        "--library-dir keeping these stems (lib/pool_libdir.py)",
    )
    ap.add_argument("--library-dir", type=Path, default=None)
    ap.add_argument(
        "--path-append",
        action="append",
        default=[],
        help="directory appended to PATH of the wrapped benches (rocminfo for "
        "hipblaslt-cotenant)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.pool_only_library and args.library_dir is None:
        ap.error("--pool-only-library needs --library-dir")
    return args


def _substitute(cmd: Sequence[str], ctx_dir: Path) -> List[str]:
    return [str(c).replace(TOKEN, str(ctx_dir)) for c in cmd]


def _replace_value(cmd: List[str], flag: str, value: str) -> List[str]:
    out = list(cmd)
    if flag in out:
        out[out.index(flag) + 1] = value
    return out


def run_sub_stage(cmd: Sequence[str], log: Path) -> int:
    with log.open("w") as f:
        return subprocess.run(
            [sys.executable, "-u", *[str(c) for c in cmd]],
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode


def round_lines(paths: Sequence[Path]) -> Dict[tuple, str]:
    """{shape: bench line} over `paths`, first line of a shape wins."""
    out: Dict[tuple, str] = {}
    for p in paths:
        for line in read_bench_lines(p):
            shape = shape_of_line(line)
            if shape is not None:
                out.setdefault(shape, line)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()
    try:
        cfg = yaml.safe_load(args.config_yaml.read_text()) or {}
        cmds = json.loads(args.commands)
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
    except (OSError, ValueError, RuntimeError) as e:
        ui.err("cfg", str(e))
        return 1
    n_cu = int(hardware.n_cu)
    try:
        kfd_cu = int(hwlib.kfd_device_hardware(args.hardware_device).n_cu)
    except (OSError, ValueError, RuntimeError):
        kfd_cu = n_cu
    if kfd_cu != n_cu:
        ui.warn(
            "cfg",
            f"hardware.n_cu {n_cu} differs from the device's {kfd_cu} CUs; "
            f"cotenants size for {n_cu}",
        )
    errors = context_errors(cfg, n_cu)
    missing = [k for k in ("stage03",) if not cmds.get(k)]
    contexts = contexts_from_config(cfg)
    if any(c.holdout for c in contexts) and not cmds.get("stage04"):
        missing.append("stage04")
    if missing:
        errors.append(f"--commands lacks {missing}")
    if errors:
        for e in errors:
            ui.err("cfg", e)
        return 1
    seed = int((cfg.get("contexts") or {}).get("seed", 42))
    cotenant = args.cotenant_binary or args.bench_binary.parent / "hipblaslt-cotenant"
    if any(c.cotenant for c in contexts) and not cotenant.exists():
        ui.err("cfg", f"hipblaslt-cotenant not found: {cotenant}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    library_path = None
    if args.pool_only_library:
        from lib.pool_libdir import build_pool_libdir

        dst = args.out_dir / "libdir_pool"
        try:
            build_pool_libdir(args.library_dir, args.pool_only_library, dst)
        except (OSError, ValueError) as e:
            ui.err("libdir", str(e))
            return 1
        library_path = str(dst)

    lines = round_lines(args.shapes_yaml)
    shapes = list(lines)
    if not args.quiet:
        ui.banner("stage03c bench execution contexts", ui.C.HEAD)
        ui.info("cfg", f"shapes       : {len(shapes):,}")
        ui.info("cfg", f"device CUs   : {n_cu}")
        for c in contexts:
            ui.info(
                "cfg",
                f"context      : {c.label}  cu_budget={c.cu_budget} "
                f"schedule={c.schedule} cotenant_cus={c.cotenant_cus(n_cu)} "
                f"fraction={c.fraction:g}{'  (held out)' if c.holdout else ''}",
            )

    records: List[Dict[str, Any]] = []
    rc_all = 0
    for ctx in contexts:
        record = _bench_context(
            ctx, args, cmds, lines, shapes, seed, n_cu, cotenant, library_path
        )
        records.append(record)
        if record["rc"] != 0:
            rc_all = record["rc"]
            break

    holdout = [r for r in records if r["holdout"] and r["n_shapes"]]
    holdout_rc = None
    if rc_all == 0 and holdout:
        hdir = args.out_dir / "holdout"
        hdir.mkdir(parents=True, exist_ok=True)
        cmd = _substitute(cmds["stage04"], hdir)
        for r in holdout:
            cmd += [
                "--context-log-dir",
                str(r["cu_budget"]),
                r["schedule"],
                r["log_dir"],
            ]
            if r.get("plan"):
                cmd += ["--context-plan", str(r["cu_budget"]), r["schedule"], r["plan"]]
        holdout_rc = run_sub_stage(cmd, hdir / "stage04.log")
        if holdout_rc != 0:
            ui.err(
                "holdout", f"stage04 of the held-out contexts failed (rc={holdout_rc})"
            )
            rc_all = holdout_rc

    summary = {
        "n_cu": n_cu,
        "seed": seed,
        "n_shapes": len(shapes),
        "library_path": library_path,
        "contexts": records,
        "holdout_stage04_rc": holdout_rc,
        "elapsed_s": time.time() - t0,
    }
    atomic_write_text(args.out_dir / SUMMARY, json.dumps(summary, indent=2) + "\n")
    if not args.quiet:
        say = ui.ok if rc_all == 0 else ui.err
        say("done", f"{len(records)} context(s) in {ui.fmt_dur(time.time() - t0)}")
    return rc_all


def _bench_context(
    ctx: BenchContext,
    args: argparse.Namespace,
    cmds: Dict[str, Any],
    lines: Dict[tuple, str],
    shapes: List[tuple],
    seed: int,
    n_cu: int,
    cotenant: Path,
    library_path: Optional[str],
) -> Dict[str, Any]:
    ctx_dir = args.out_dir / ctx.label
    ctx_dir.mkdir(parents=True, exist_ok=True)
    logs = ctx_dir / "stage03" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    own_lines, plan = lines, None
    if args.context_shapes_dir is not None:
        replanned = args.context_shapes_dir / ctx.label / "shapes.yaml"
        if replanned.is_file():
            own_lines = {**lines, **round_lines([replanned])}
            p = replanned.with_name("plan.json")
            plan = str(p) if p.is_file() else None
    chosen = select_shapes(shapes, ctx, seed)
    atomic_write_text(ctx_dir / "shapes.yaml", "".join(own_lines[s] for s in chosen))
    record: Dict[str, Any] = {
        **ctx.to_dict(),
        "cotenant_cus": ctx.cotenant_cus(n_cu),
        "n_shapes": len(chosen),
        "dir": str(ctx_dir),
        "log_dir": str(logs),
        "plan": plan,
        "rc": 0,
        "elapsed_s": {},
    }
    if not chosen:
        return record
    spec = bench_spec(
        ctx,
        n_cu,
        str(args.bench_binary),
        str(cotenant),
        str(ctx_dir / "cotenant_logs"),
        library_path=library_path,
        path_append=args.path_append,
    )
    wrapper = write_wrapper(ctx_dir, spec, sys.executable, CONTEXT_BENCH)
    record["wrapper"] = str(wrapper)
    for key in ("stage02", "stage03"):
        if not cmds.get(key):
            continue
        cmd = _replace_value(
            _substitute(cmds[key], ctx_dir), "--bench-binary", str(wrapper)
        )
        if plan is not None:
            cmd = _replace_value(cmd, "--plan", plan)
        if not args.quiet:
            ui.info(ctx.label, f"{key}: {len(chosen):,} shapes -> {ctx_dir / key}")
        t = time.time()
        rc = run_sub_stage(cmd, ctx_dir / f"{key}.log")
        record["elapsed_s"][key] = round(time.time() - t, 1)
        if rc != 0:
            ui.err(ctx.label, f"{key} failed (rc={rc}); see {ctx_dir / (key + '.log')}")
            record["rc"] = rc
            return record
        if not args.quiet:
            ui.ok(ctx.label, f"{key}: done in {ui.fmt_dur(time.time() - t)}")
    return record


if __name__ == "__main__":
    sys.exit(main())
