#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage03c_bench_contexts -- bench the round's shapes under the configured
non-exclusive execution contexts (lib/contexts.py).

stage03 benches the exclusive context. For every entry of the config's
`contexts.list` and `contexts.holdout` this stage

  1. picks the shapes the context benches: every shape, or
     ceil(fraction * n) per cell drawn with `contexts.seed`, the shapes
     stage01 drew near the round's tail GEMMs of that context
     (--targeted-shapes, `lib.contexts.targeted_for`) among them;
  2. writes their bench lines (the round's own lines: the same kernel list
     per shape as the exclusive bench, or the context's re-planned lines from
     --context-shapes-dir), with the context's summation order and workspace
     under launch rule version 3 (`lib.bench_yaml.context_line`), to
     `<label>/shapes.yaml` (a co-tenant context's without the clustered
     kernels, `lib.contexts.cotenant_line`, and without a line left with
     none); a launch-keyed workspace context keeps only the
     lines on which some kernel of the library --launch-library-dir /
     --library-stem launches otherwise than in the full workspace
     (`lib.contexts.launch_changing_shapes`), and with `contexts.canaries`
     checks the results of such a context with its canaries in
     `<label>/canary/` (a sample of (line, kernel) pairs the bench can
     check on integer-exact operands, `--post_bench_verify_iters` on a few
     lines, all against the bench's host reference; a failed check, or a
     canary set that performed none, stops the stage);
  3. writes `<label>/hipblaslt-bench`, a wrapper that runs the bench with
     `--sm_count_target <cu_budget>` (and `--hybrid_assignment_policy` for
     the dynamic / auto schedules), inside `hipblaslt-cotenant --cus <device
     CUs - cu_budget>` for a cotenant context; the device CU count comes from
     the KFD topology (or the config's `hardware.n_cu`);
  4. runs stage03 on them with that wrapper as the bench binary (the argv
     template from --commands, `{ctx_dir}` standing for `<out-dir>/<label>`),
     so every context is benched under itself.

The held-out contexts' logs are then enriched by the stage04 template of
--commands into `<out-dir>/holdout/stage04` (one --context-log-dir per
context): stage04b evaluates the model on them, stage05 never reads them. The
trained contexts' logs (`<label>/stage03/logs`) go to the round's stage04.

Outputs in --out-dir:
  <label>/                 shapes.yaml, context.json, hipblaslt-bench,
                           cotenant_logs/, stage03/, stage03.log
  holdout/stage04/         enriched rows of the held-out contexts
  libdir_pool/             with --pool-only-library (lib/pool_libdir.py)
  contexts.json            per context: shapes, wrapper, sub-stage rc / time

The stage exits 2 when a context's stage03 exited 2 or a sub-stage was
killed by SIGHUP, SIGINT, SIGKILL or SIGTERM (another attempt may cure it),
3 when a context's stage03 stopped for a wedged GPU or host (lib/procs.py
WEDGED_EXIT; no later context is benched), and 1 on any other failure.
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

from lib import dat  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import hipblaslt_launch  # noqa: E402
from lib import procs  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    context_line,
    line_solution_list,
    read_bench_lines,
    shape_of_line,
)
from lib.contexts import (  # noqa: E402
    CANARY_LOG_GLOB,
    BenchContext,
    bench_spec,
    canary_lines,
    canary_refusal,
    canary_settings,
    canary_verdict,
    checkable_pairs,
    context_errors,
    contexts_from_config,
    cotenant_kernels,
    cotenant_line,
    launch_changing_shapes,
    select_shapes,
    targeted_for,
    write_wrapper,
)
from lib.fs import atomic_write_text  # noqa: E402

CONTEXT_BENCH = PIPELINE_DIR / "lib" / "context_bench.py"
TOKEN = "{ctx_dir}"
SUMMARY = "contexts.json"
SUB_STAGES = ("stage03", "stage04")


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
        help="JSON object: stage03, stage04 -> argv (script first) with "
        "{ctx_dir} for a context's directory; stage04 (held-out contexts) "
        "gets one --context-log-dir per context appended",
    )
    ap.add_argument(
        "--targeted-shapes",
        type=Path,
        default=None,
        help="stage01's targeted_shapes.json: the shapes drawn near a tail "
        "GEMM of a context are benched in that context",
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
        "--launch-library-dir",
        type=Path,
        default=None,
        help="library directory whose --library-stem kernels a launch-keyed "
        "context plans (needed when the config has one)",
    )
    ap.add_argument(
        "--library-stem",
        default=None,
        help="library stem of --launch-library-dir (default: the config's "
        "hipblaslt.library_stem)",
    )
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
    """Run a sub-stage, appending its output to `log`: its stage03 keeps a
    context an earlier invocation benched over the same lines and continues
    one an interruption cut off (lib/bench_resume.py)."""
    with log.open("a") as f:
        return subprocess.run(
            [sys.executable, "-u", *[str(c) for c in cmd]],
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=procs.die_with_parent(),
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
    contexts = [] if errors else contexts_from_config(cfg, n_cu)
    if any(c.holdout for c in contexts) and not cmds.get("stage04"):
        missing.append("stage04")
    if missing:
        errors.append(f"--commands lacks {missing}")
    unknown = sorted(set(cmds) - set(SUB_STAGES))
    if unknown:
        errors.append(
            f"--commands: unknown sub-stages {unknown} (stage03c runs "
            f"{list(SUB_STAGES)})"
        )
    if errors:
        for e in errors:
            ui.err("cfg", e)
        return 1
    seed = int((cfg.get("contexts") or {}).get("seed", 42))
    cotenant = args.cotenant_binary or args.bench_binary.parent / "hipblaslt-cotenant"
    if any(c.cotenant for c in contexts) and not cotenant.exists():
        ui.err("cfg", f"hipblaslt-cotenant not found: {cotenant}")
        return 1
    launches = None
    if any(c.launch_keyed for c in contexts):
        try:
            launches = launch_pass(args, cfg, hardware)
        except (OSError, ValueError) as e:
            ui.err("cfg", f"launch-keyed contexts: {e}")
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
                f"fraction={c.fraction:g} "
                f"uso={'on' if c.uniform_summation_order else 'off'} "
                f"workspace={c.workspace_bytes}"
                f"{'  (launch-keyed)' if c.launch_keyed else ''}"
                f"{'  (held out)' if c.holdout else ''}",
            )

    records: List[Dict[str, Any]] = []
    rc_all = 0
    for ctx in contexts:
        record = _bench_context(
            ctx,
            args,
            cfg,
            cmds,
            lines,
            shapes,
            seed,
            n_cu,
            cotenant,
            library_path,
            launches,
        )
        records.append(record)
        if record["rc"] != 0:
            rc_all = procs.retry_status(record["rc"], bench=True)
            break
        if ctx.launch_keyed and record["n_shapes"] and canary_settings(cfg)["enabled"]:
            canaries = _canary_context(
                ctx, args, cfg, cmds, record, n_cu, cotenant, library_path, launches
            )
            record["canaries"] = canaries
            refusal = canary_refusal(canaries)
            if refusal is not None:
                ui.err(ctx.label, f"{refusal}; see {canaries['dir']}")
                rc_all = 1
                break
            if not args.quiet:
                ui.ok(
                    ctx.label,
                    f"canaries: {canaries['checks']} check(s), "
                    f"{canaries['exact_checks']} on integer-exact operands",
                )

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
            rc_all = procs.retry_status(holdout_rc)

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


def launch_pass(
    args: argparse.Namespace, cfg: Dict[str, Any], hardware: Any
) -> Dict[str, Any]:
    """What a launch-keyed context plans the launches of its lines with: the
    kernels of the library stem (--library-stem, else the config's
    `hipblaslt.library_stem`) in --launch-library-dir, their launch on the
    bench device and the problems' matrix-instruction type."""
    hl = cfg.get("hipblaslt") or {}
    stem = args.library_stem or hl.get("library_stem")
    if args.launch_library_dir is None or not stem:
        raise ValueError(
            "a launch-keyed context needs --launch-library-dir and a library stem"
        )
    try:
        mi_dtype = fs.dtype_from_hipblaslt(hl.get("a_type"))
    except ValueError:
        mi_dtype = ""
    return {
        "kernels": dat.load_library_kernels(args.launch_library_dir, str(stem)),
        "launch": hipblaslt_launch.HipblasltLaunch.for_library(
            str(stem), hardware.num_xcd
        ),
        "mi_dtype": mi_dtype,
    }


def _canary_context(
    ctx: BenchContext,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    cmds: Dict[str, Any],
    record: Dict[str, Any],
    n_cu: int,
    cotenant: Path,
    library_path: Optional[str],
    launches: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check the results of `ctx`, which benched the lines of
    `<dir>/shapes.yaml` (`record`), with its canaries
    (lib/contexts.py `canary_lines`, `contexts.canaries`): stage03 of the
    template benches them in `<dir>/canary/` under the context, through a
    wrapper that adds `--post_bench_verify_iters` and `--verify_strict`;
    returns what their logs say (`canary_verdict`, the checks counted in
    the block logs), their counts, the directory and the sub-stage's rc
    (None when no line was drawn: nothing ran, and nothing is verified)."""
    settings = canary_settings(cfg)
    seed = int((cfg.get("contexts") or {}).get("seed", 42))
    kernels = [
        int(k["sol_idx_global"])
        for k in (launches or {}).get("kernels") or []
        if k.get("sol_idx_global") is not None
    ]
    benched = read_bench_lines(Path(record["dir"]) / "shapes.yaml")
    arch = str(cfg.get("arch") or "")
    exact, verify = canary_lines(benched, kernels, settings, seed, ctx.label, arch)
    cdir = Path(record["dir"]) / "canary"
    counts = {
        "dir": str(cdir),
        "exact_lines": len(exact),
        "checkable_pairs": len(checkable_pairs(benched, kernels, arch)),
        "verify_lines": len(verify),
        "verify_iters": int(settings["verify_iters"]),
    }
    if not exact and not verify:
        return {**counts, "rc": None, **canary_verdict([])}
    (cdir / "stage03" / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write_text(cdir / "shapes.yaml", "".join(exact + verify))
    spec = bench_spec(
        ctx,
        n_cu,
        str(args.bench_binary),
        str(cotenant),
        str(cdir / "cotenant_logs"),
        library_path=library_path,
        path_append=args.path_append,
    )
    spec["bench_args"] = [
        "--post_bench_verify_iters",
        str(int(settings["verify_iters"])),
        "--verify_strict",
    ]
    wrapper = write_wrapper(cdir, spec, sys.executable, CONTEXT_BENCH)
    cmd = _replace_value(
        _substitute(cmds["stage03"], cdir), "--bench-binary", str(wrapper)
    )
    rc = run_sub_stage(cmd, cdir / "stage03.log")
    logs = sorted((cdir / "stage03" / "logs").glob("*.log"))
    texts = {p: p.read_text(errors="replace") for p in logs}
    counted = [t for p, t in texts.items() if p.match(CANARY_LOG_GLOB)]
    return {
        **counts,
        "rc": rc,
        **canary_verdict(texts.values(), exact, verify, counted),
    }


def _bench_context(
    ctx: BenchContext,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    cmds: Dict[str, Any],
    lines: Dict[tuple, str],
    shapes: List[tuple],
    seed: int,
    n_cu: int,
    cotenant: Path,
    library_path: Optional[str],
    launches: Optional[Dict[str, Any]] = None,
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
    chosen = select_shapes(shapes, ctx, seed, targeted_for(args.targeted_shapes, ctx))
    drawn = len(chosen)
    if ctx.launch_keyed and launches is not None:
        chosen = launch_changing_shapes(
            chosen,
            ctx,
            launches["kernels"],
            launches["launch"],
            n_cu,
            launches["mi_dtype"],
        )
    texts = [context_line(own_lines[s], cfg, ctx) for s in chosen]
    if ctx.cotenant:
        clustered, plain = cotenant_kernels(cfg)
        kept = [(s, cotenant_line(t, clustered, plain)) for s, t in zip(chosen, texts)]
        kept = [(s, t) for s, t in kept if line_solution_list(t)]
        chosen, texts = [s for s, _ in kept], [t for _, t in kept]
    atomic_write_text(ctx_dir / "shapes.yaml", "".join(texts))
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
    if ctx.launch_keyed:
        record["n_shapes_unchanged"] = drawn - len(chosen)
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
    cmd = _replace_value(
        _substitute(cmds["stage03"], ctx_dir), "--bench-binary", str(wrapper)
    )
    if not args.quiet:
        ui.info(ctx.label, f"stage03: {len(chosen):,} shapes -> {ctx_dir / 'stage03'}")
    t = time.time()
    rc = run_sub_stage(cmd, ctx_dir / "stage03.log")
    record["elapsed_s"]["stage03"] = round(time.time() - t, 1)
    if rc != 0:
        ui.err(ctx.label, f"stage03 failed (rc={rc}); see {ctx_dir / 'stage03.log'}")
        record["rc"] = rc
        return record
    if not args.quiet:
        ui.ok(ctx.label, f"stage03: done in {ui.fmt_dur(time.time() - t)}")
    return record


if __name__ == "__main__":
    sys.exit(main())
