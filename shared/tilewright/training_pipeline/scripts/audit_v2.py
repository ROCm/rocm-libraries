#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set's protocol v2 measurement (lib/audit_v2.py), in
`<audit>/v2/`; the audit's own files are only read.

  audit_v2.py plan  --audit-dir A --config C [--round-dir R ...]
                    [--model NAME=FILE ...] [settings of a first plan]
  audit_v2.py bench --audit-dir A [--device D]
  audit_v2.py run   --audit-dir A --config C (--gpus 1,2 | --devices 0,1)
                    [models as for plan]
  audit_v2.py status --audit-dir A

plan registers the models (a round's candidate as audit_eval.py picks it,
named `<run>.<round>`; a model file by the name given; with --name-with-sha
each name ends in the first 12 hex digits of its file's sha256, so that a
retrained model never clashes with the one registered before it) with
their top-1 picks on every audit GEMM, and plans the generation that
benches the GEMMs whose sets lack a pick: all of them the first time, with
the candidates of the exhaustive pass. Settings (--repetitions,
--canary-attempts, ...) apply to a first plan only.

bench works through the planned parts no other worker holds on one GPU,
device D of its environment (scripts/gpu_run.py gives it its GPU alone as
device 0), and returns when none is left. It refuses (exit 2) to bench a
generation whose solution indices name other kernels in the library the
plan recorded (for a plan that recorded none, the library its config
gives) than they did when it was planned: the hipBLASLt build changed
(lib/kernel_identity.py; for a generation planned before kernel identities
were recorded, by the audit set's kernel digests as of its plan), and to
bench without a library to check against. plan refuses to plan on top of
such a generation or model.

run is plan, then one bench worker per --gpus GPU, each under that GPU's
lease (scripts/gpu_run.py), waiting for all of them; worker logs go to
`v2/worker_gpu<N>.log`. --devices names the GPUs by device index in this
process's environment instead, as runtime.devices does: run leases their
GPUs itself (a lease an ancestor holds, such as the orchestrator's, is
used as it is) and gives each worker its device's GPU alone. Exit status:
0; 1 when a worker failed; 2 for bad arguments, a refused lease or a
refused plan.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import audit as auditlib  # noqa: E402
from lib import audit_v2 as v2  # noqa: E402
from lib import run_layout  # noqa: E402
from lib import ui  # noqa: E402

GPU_RUN = PIPELINE_DIR / "scripts" / "gpu_run.py"
SETTING_FLAGS = {
    "repetitions": int,
    "set_tolerance": float,
    "set_max": int,
    "spread_limit": float,
    "min_readings": int,
    "canary_tolerance": float,
    "canary_attempts": int,
    "canary_timings": int,
    "canary_reference_readings": int,
    "canary_backoff_s": float,
    "seed": int,
}


def add_model_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=Path, required=True, help="the run's config")
    ap.add_argument("--round-dir", type=Path, action="append", default=[])
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="a models.pt or MLREC file registered as NAME",
    )
    ap.add_argument("--weight-dtype", default=None)
    ap.add_argument("--library-dir", type=Path, default=None)
    ap.add_argument("--hardware-device", type=int, default=None)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--audit-dir", type=Path, required=True)
        add_model_arguments(p)
        for key, typ in SETTING_FLAGS.items():
            p.add_argument("--" + key.replace("_", "-"), type=typ, default=None)
        p.add_argument(
            "--name-with-sha",
            action="store_true",
            help="end every model's name in the first 12 hex digits of its sha256",
        )
        if name == "run":
            where = p.add_mutually_exclusive_group(required=True)
            where.add_argument("--gpus", help="physical GPUs, e.g. 1,2")
            where.add_argument(
                "--devices", help="device indices of this environment, e.g. 0,1"
            )
            p.add_argument(
                "--wait", type=float, default=0.0, help="seconds to wait for a lease"
            )
    b = sub.add_parser("bench")
    b.add_argument("--audit-dir", type=Path, required=True)
    b.add_argument("--device", default="0")
    s = sub.add_parser("status")
    s.add_argument("--audit-dir", type=Path, required=True)
    return ap.parse_args(argv)


def model_jobs(args: argparse.Namespace) -> List[Tuple[str, Path, str]]:
    """(name, file, source) of the --round-dir and --model models."""
    import audit_eval

    out = []
    for rd in args.round_dir:
        path, source = audit_eval.round_model(rd)
        out.append((f"{Path(rd).resolve().parent.name}.{Path(rd).name}", path, source))
    for spec in args.model:
        name, sep, file = str(spec).partition("=")
        if not sep or not name or not file:
            raise SystemExit(f"--model needs NAME=FILE, got {spec!r}")
        out.append((name, Path(file), "model file"))
    return out


def _context(args: argparse.Namespace):
    """(cfg, config path, spec, hardware, weight dtype, paths)."""
    import run_orchestrator as ro
    from lib import hardware as hwlib

    cfg, _ = ro.load_config(args.config)
    paths = ro.derive_paths(cfg)
    library_dir = args.library_dir or paths.library_dir
    device = (
        args.hardware_device
        if args.hardware_device is not None
        else int(cfg["runtime"]["devices"][0])
    )
    spec = auditlib.Spec(
        audit_dir=args.audit_dir,
        library_dir=library_dir,
        library_stem=str(cfg["hipblaslt"]["library_stem"]),
        hardware_device=device,
        pick_measurement=str(ro.cfg_get(cfg, "validate.pick_measurement")),
        protocol="v1",
    )
    hardware = hwlib.device_hardware(cfg, device)
    cfg, _ = ro.resolve_contexts(cfg, device)
    dtype = args.weight_dtype or str(ro.cfg_get(cfg, "deploy.weight_dtype"))
    return cfg, spec, hardware, dtype, paths, device


def _wrappers(
    d: Path, cfg: Mapping[str, Any], paths: Any, units: Mapping[str, Any], device: int
) -> Dict[str, str]:
    """The stage03c co-tenant wrapper of every context group, in
    `<v2>/bench/<label>/`."""
    import run_orchestrator as ro
    from lib import contexts as ctxlib
    from lib import hardware as hwlib

    out = {}
    n_cu = int(hwlib.device_hardware(cfg, device).n_cu)
    for label, unit in units.items():
        if unit.get("context") is None:
            continue
        ctx = ctxlib.BenchContext.from_mapping(unit["context"])
        cdir = d / "bench" / label
        spec = ctxlib.bench_spec(
            ctx,
            n_cu,
            str(paths.bench_binary),
            str(ro.context_cotenant(cfg, paths)),
            str(cdir / "cotenant_logs"),
            path_append=ro.cfg_get(cfg, "contexts.path_append") or [ro.rocm_bin()],
        )
        script = PIPELINE_DIR / "lib" / "context_bench.py"
        out[label] = str(ctxlib.write_wrapper(cdir, spec, sys.executable, script))
    return out


def plan(args: argparse.Namespace) -> Optional[int]:
    """Register the models and plan their generation (see the module
    docstring); returns its index, or None when nothing needs a bench."""
    import run_orchestrator as ro
    from lib.bench_yaml import read_bench_lines, shape_of_line

    audit = Path(args.audit_dir)
    cfg, spec, hardware, dtype, paths, device = _context(args)
    st, why = auditlib.status(audit)
    if st != "ok":
        raise v2.V2Error(f"the audit set is not complete: {why}")
    units = auditlib.units(audit) or {}
    gemms = auditlib.evaluable_gemms(spec, cfg)
    from lib import evaluate as ev

    from lib.bank_pool import config_pool

    identity = ev.library_identity(
        spec.library_dir, spec.library_stem, cfg, spec.hardware_device
    )
    pool = config_pool(cfg, spec.library_dir, spec.library_stem, spec.hardware_device)
    kernels = v2.AuditKernels.of(
        auditlib.manifest(audit) or {}, auditlib.pool_digests(pool.kernels)
    )
    lines: Dict[v2.Shape, str] = {}
    try:
        planned = run_layout.plan_dir(audit) / "shapes.yaml"
    except ValueError as e:
        raise v2.V2Error(str(e)) from None
    for ln in read_bench_lines(planned):
        s = shape_of_line(ln)
        if s is not None:
            lines.setdefault(tuple(s), ln)
    parts = {
        label: [[tuple(s) for s in p] for p in unit["parts"]]
        for label, unit in units.items()
    }
    new_models = []
    for name, path, source in model_jobs(args):
        if not path.is_file():
            raise v2.V2Error(f"{path} does not exist")
        data = auditlib.model_bytes(path, str(cfg["arch"]), dtype)
        sha = v2.sha256_of(data)
        if getattr(args, "name_with_sha", False):
            name = f"{name}.{sha[:12]}"
        picks = auditlib.picks(data, spec, cfg, hardware=hardware)
        new_models.append(v2.Model(name, sha, f"{source}: {path}", dict(picks)))
        ui.info("model", f"{name}: {source} {path} ({len(picks)} picks)")
    d = v2.v2_dir(audit)
    d.mkdir(parents=True, exist_ok=True)
    bench = {
        "binary": str(paths.bench_binary),
        "config_yaml": str(Path(args.config).resolve()),
        "wrappers": _wrappers(d, cfg, paths, units, device),
        "stall_timeout_s": float(ro.cfg_get(cfg, "bench.stall_timeout_s")),
        "startup_grace_s": float(ro.cfg_get(cfg, "bench.startup_grace_s")),
        "library": {"dir": str(spec.library_dir), "stems": list(identity.stems)},
    }
    settings = {
        k: getattr(args, k) for k in SETTING_FLAGS if getattr(args, k) is not None
    }
    gen = v2.plan(
        d,
        gemms=gemms,
        lines=lines,
        parts=parts,
        new_models=new_models,
        settings=settings,
        bench=bench,
        part_shapes=int(auditlib.settings(cfg)["part_shapes"]),
        identity=identity,
        kernels=kernels,
    )
    v2.write_progress(d)
    if gen is None:
        ui.ok("plan", "every set holds every registered pick; nothing to bench")
    else:
        g = (v2.manifest(d) or {})["generations"][gen]
        ui.ok("plan", f"generation {gen}: {g['n_gemms']} GEMMs in {g['n_parts']} parts")
    return gen


def canary_key(device: str) -> str:
    """The GPU a worker's canary reference belongs to: the physical GPU of
    its device, else (every GPU hidden) the device index."""
    from lib import kfd

    try:
        gpu = kfd.physical_gpu(int(device))
    except (kfd.GpuMapError, ValueError):
        gpu = None
    return str(gpu) if gpu is not None else f"device{device}"


def config_library(m: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The bench library of a v2 manifest that records none
    (`bench.library`): the one its config (`bench.config_yaml`) gives, as
    plan takes it; None when the manifest records one or the config cannot
    tell."""
    import run_orchestrator as ro
    from lib import evaluate as ev

    b = m.get("bench") or {}
    if b.get("library") or not b.get("config_yaml"):
        return None
    try:
        cfg, _ = ro.load_config(Path(b["config_yaml"]))
        library_dir = ro.derive_paths(cfg).library_dir
        identity = ev.library_identity(
            library_dir,
            str(cfg["hipblaslt"]["library_stem"]),
            cfg,
            int(cfg["runtime"]["devices"][0]),
        )
    except (
        ro.ConfigError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        IndexError,
        TypeError,
    ) as e:
        ui.warn("bench", f"cannot read the library of {b['config_yaml']}: {e}")
        return None
    return {"dir": str(library_dir), "stems": list(identity.stems)}


def bench(args: argparse.Namespace) -> int:
    from lib import procs

    d = v2.v2_dir(args.audit_dir)
    m = v2.manifest(d)
    if m is None:
        ui.err("bench", f"no v2 plan in {d}; run plan first")
        return 2
    try:
        v2.check_bench_library(
            d, m, auditlib.manifest(args.audit_dir) or {}, config_library(m)
        )
    except v2.V2Error as e:
        ui.err("bench", str(e))
        return 2
    b = m.get("bench") or {}
    gpu = canary_key(args.device)
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):
        runner = v2.stage03_runner(
            Path(b["config_yaml"]) if b.get("config_yaml") else None,
            device=str(args.device),
            stall_timeout_s=float(b.get("stall_timeout_s", 300.0)),
            startup_grace_s=float(b.get("startup_grace_s", 900.0)),
            children=children,
        )
        worker = v2.Bench(d, m, runner, gpu, log=lambda s: ui.info("v2", s))
        n = v2.work(worker)
    ui.ok("bench", f"GPU {gpu}: benched {n} part(s); {v2.status_line(d)}")
    return 0


def run_on_devices(args: argparse.Namespace, d: Path) -> int:
    """The bench workers of `run --devices`: one per device, under the
    lease of its GPU, each seeing that GPU alone."""
    from lib import gpu_lease, kfd

    try:
        devices = [int(x) for x in str(args.devices).split(",") if x.strip()]
    except ValueError:
        ui.err("run", f"--devices {args.devices!r}: give device indices")
        return 2
    if not devices:
        ui.err("run", "--devices names no device")
        return 2
    t0 = time.time()
    try:
        with gpu_lease.hold(
            devices,
            f"audit v2 bench of {args.audit_dir}",
            args.audit_dir,
            wait_s=args.wait,
        ):
            workers = []
            for dev in devices:
                gpu = kfd.physical_gpu(dev)
                device, env = kfd.one_gpu_launch(dev, os.environ)
                cmd = [sys.executable, str(Path(__file__).resolve()), "bench"]
                cmd += ["--audit-dir", str(args.audit_dir), "--device", device]
                name = f"gpu{gpu}" if gpu is not None else f"device{dev}"
                log = (d / f"worker_{name}.log").open("a")
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env
                )
                workers.append((dev, proc, log))
                ui.info("run", f"device {dev}: worker pid {proc.pid}, log {log.name}")
            bad = 0
            for dev, proc, log in workers:
                rc = proc.wait()
                log.close()
                (ui.ok if rc == 0 else ui.err)(
                    "run", f"device {dev}: worker exited {rc}"
                )
                bad += rc != 0
    except gpu_lease.LeaseError as e:
        ui.err("run", str(e))
        return 2
    p = v2.write_progress(d)
    ui.ok(
        "run",
        f"{v2.status_line(d)}; {p['gemms_done']} of {p['gemms_total']} done in "
        f"{ui.fmt_dur(time.time() - t0)}",
    )
    return 1 if bad else 0


def run(args: argparse.Namespace) -> int:
    plan(args)
    d = v2.v2_dir(args.audit_dir)
    if args.devices is not None:
        return run_on_devices(args, d)
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip()]
    procs_ = []
    t0 = time.time()
    for g in gpus:
        cmd = [sys.executable, str(GPU_RUN), "--gpu", str(g), "--wait", str(args.wait)]
        cmd += ["--purpose", f"audit v2 bench of {args.audit_dir}", "--"]
        cmd += [sys.executable, str(Path(__file__).resolve()), "bench"]
        cmd += ["--audit-dir", str(args.audit_dir), "--device", "0"]
        log = (d / f"worker_gpu{g}.log").open("a")
        procs_.append(
            (g, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        )
        ui.info("run", f"GPU {g}: worker pid {procs_[-1][1].pid}, log {log.name}")
    bad = 0
    for g, p, log in procs_:
        rc = p.wait()
        log.close()
        (ui.ok if rc == 0 else ui.err)("run", f"GPU {g}: worker exited {rc}")
        bad += rc != 0
    p = v2.write_progress(d)
    ui.ok(
        "run",
        f"{v2.status_line(d)}; {p['gemms_done']} of {p['gemms_total']} done in "
        f"{ui.fmt_dur(time.time() - t0)}",
    )
    return 1 if bad else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "status":
        print(v2.status_line(v2.v2_dir(args.audit_dir)))
        return 0
    if args.cmd == "bench":
        return bench(args)
    try:
        if args.cmd == "plan":
            plan(args)
            return 0
        return run(args)
    except v2.V2Error as e:
        ui.err("v2", str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
