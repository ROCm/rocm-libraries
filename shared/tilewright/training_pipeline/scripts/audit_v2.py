#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set's protocol v2 measurement (lib/audit_v2.py), in
`<audit>/v2/`; the audit's own files are only read.

  audit_v2.py plan  --audit-dir A --config C [--round-dir R ...]
                    [--model NAME=FILE ...] [settings of a first plan]
  audit_v2.py bench --audit-dir A [--device D] [--sustained]
  audit_v2.py run   --audit-dir A --config C (--gpus 1,2 | --devices 0,1)
                    [models as for plan] [--no-sustained]
  audit_v2.py status --audit-dir A

plan registers the models (a round's candidate as audit_eval.py picks it,
named `<run>.<round>`; a model file by the name given; with --name-with-sha
each name ends in the first 12 hex digits of its file's sha256, so that a
retrained model never clashes with the one registered before it) with
their top-1 picks on every audit GEMM (with --without-alignment-veto the
picks of each model without the alignment veto, as a run scores its rounds'
bundles: audit_eval.py), and plans the generation that
benches the GEMMs whose sets lack a pick: all of them the first time, with
the candidates of the exhaustive pass. Settings (--repetitions,
--canary-attempts, --kernel-order, ...) apply to a first plan only; the
width of the sets (--set-tolerance, --set-tolerance-us, --set-max) is the
config's (`audit.v2_set_tolerance`, `audit.v2_set_tolerance_us`,
`audit.v2_set_max`) unless a flag gives it. A setting asked for that differs
from the one a measurement was planned with is not applied, with a warning.

The sustained composite's measurement (lib/audit_sustained.py, in
`v2/sustained/`) is planned from the v2 measurement once it is benched:
`run` benches v2, then plans the sustained generation the v2 measurement
asks for (its power-capped exclusive GEMMs not planned yet, and those whose
sets lack a registered pick) and benches it, unless --no-sustained; `plan`
plans it too once the v2 measurement is benched. Its settings
(--sustained-repetitions, --sustained-min-us, ...) apply to its first plan
only; the compute-bound rate is read from the config's first device then,
unless --sustained-compute-bound-tflops gives it, and the absolute floor of
its sets is the config's `audit.v2_set_tolerance_us` unless
--sustained-set-tolerance-us gives it (its relative widths stay its own).
As for v2, a setting asked for that differs from the one the measurement
was planned with is not applied, with a warning. `bench --sustained`
benches the sustained parts.

No set holds, and no bench times, a kernel the run's blocklist (the
config's, lib/correctness.py) or a blocklist the audit set was built with
leaves out (lib/audit.py `audit_blocklist`): plan writes their union into
`v2/blocklists/` (`v2/sustained/blocklists/` for the sustained
measurement), names it by its sha256 in the manifest and the generation it
plans, and refuses a measurement whose canary it blocks; bench leaves out of
a part's lines whatever the manifest's latest blocklist blocks, and refuses
a blocklist file that is not the one the manifest names by its sha256.

bench works through the planned parts no other worker holds on one GPU,
device D of its environment (scripts/gpu_run.py gives it its GPU alone as
device 0), and returns when none is left. A line that hangs in a part is
benched once more, then quarantined in the run under the measurement's own
protocol and its GEMM dropped from the part (lib/audit_v2.py, Hung lines),
and a line whose bench leaves its GPU wedged is quarantined so at once;
the canary's lines are never quarantined. While the run guard the
environment names holds its GPU (lib/gpu_health.py, with the config's
bench.health on), it takes no part, and returns once no part is left or
after bench.health.max_wait_s, or at once while the run holds its GPU or
the host as wedged (only a reboot ends that); a part whose stage03 runs
bench nothing is left unwritten for another worker (lib/audit_v2.py,
Unmeasured parts), and the worker then waits so, or exits 1 when the run
does not hold its GPU. It refuses (exit 2) to bench a
generation whose solution indices name other kernels in the library the
plan recorded (for a plan that recorded none, the library its config
gives) than they did when it was planned: the hipBLASLt build changed
(lib/kernel_identity.py; for a generation planned before kernel identities
were recorded, by the audit set's kernel digests as of its plan), and to
bench without a library to check against. plan refuses to plan on top of
such a generation or model.

run is plan, then one bench worker per --gpus GPU, each under that GPU's
lease (scripts/gpu_run.py), waiting for all of them; worker logs go to
`v2/worker_gpu<N>.log`; then the same for the sustained measurement
(`v2/sustained/worker_gpu<N>.log`). --devices names the GPUs by device
index in this process's environment instead, as runtime.devices does: run
leases their GPUs itself (a lease an ancestor holds, such as the
orchestrator's, is used as it is) for both phases and gives each worker its
device's GPU alone. The workers of both phases share one run guard
(lib/gpu_health.py: the GPUs' baseline and hold, the health canary, the
quarantined lines) and count each other's benches as their own load
(lib/host_load.py): the orchestrator's when it runs them, else `v2/guard/`
with one session per invocation of run. Exit status: 0; 1 when a worker
failed or GEMM benches are left once the workers end (theirs stopped, as
bench says); 2 for bad arguments, a refused lease or a refused plan.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib import audit as auditlib  # noqa: E402
from lib import audit_sustained as asus  # noqa: E402
from lib import audit_v2 as v2  # noqa: E402
from lib import run_layout  # noqa: E402
from lib import ui  # noqa: E402

GPU_RUN = PIPELINE_DIR / "scripts" / "gpu_run.py"


def _kernel_order(text: str) -> str:
    if text not in v2.KERNEL_ORDERS:
        raise argparse.ArgumentTypeError(f"one of {', '.join(v2.KERNEL_ORDERS)}")
    return text


def _flag_bool(text: str) -> bool:
    low = str(text).lower()
    if low not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
        raise argparse.ArgumentTypeError("true or false")
    return low in ("true", "1", "yes", "on")


SETTING_FLAGS = {
    "repetitions": int,
    "set_tolerance": float,
    "set_tolerance_us": float,
    "set_max": int,
    "spread_limit": float,
    "min_readings": int,
    "canary_tolerance": float,
    "canary_tolerance_us": float,
    "canary_attempts": int,
    "canary_timings": int,
    "canary_reference_readings": int,
    "canary_backoff_s": float,
    "seed": int,
    "kernel_order": _kernel_order,
    "sacrificial_kernel": _flag_bool,
    "warmup_line": _flag_bool,
}
# The settings of a first v2 plan the config gives (its keys) when no flag
# does.
CONFIG_SETTINGS = {
    "set_tolerance": "audit.v2_set_tolerance",
    "set_tolerance_us": "audit.v2_set_tolerance_us",
    "set_max": "audit.v2_set_max",
}
# The settings of a first sustained plan the config gives when no
# --sustained flag does: the absolute floor of the sets, an error of single
# readings on the device; the relative widths are the sustained
# measurement's own.
SUSTAINED_CONFIG_SETTINGS = {"set_tolerance_us": "audit.v2_set_tolerance_us"}
# The sustained measurement's settings, as --sustained-<key>.
SUSTAINED_FLAGS = {
    **SETTING_FLAGS,
    "compute_bound_set_tolerance": float,
    "compute_bound_set_max": int,
    "min_us": float,
    "very_long_us": float,
    "compute_bound_min_us": float,
    "compute_bound_fraction": float,
    "compute_bound_tflops": float,
    "warmup_time": float,
    "measure_time": float,
    "max_measure_time": float,
    "part_gemms": int,
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
    ap.add_argument(
        "--without-alignment-veto",
        action="store_true",
        help="register each model's picks without the alignment veto "
        "(deploy.alignment_veto), as the run scores its rounds' bundles",
    )


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
        for key, typ in SUSTAINED_FLAGS.items():
            p.add_argument(
                "--sustained-" + key.replace("_", "-"),
                dest=f"sustained_{key}",
                type=typ,
                default=None,
            )
        p.add_argument(
            "--name-with-sha",
            action="store_true",
            help="end every model's name in the first 12 hex digits of its sha256",
        )
        p.add_argument(
            "--no-sustained",
            dest="sustained",
            action="store_false",
            help="leave the sustained composite's measurement as it is",
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
    b.add_argument(
        "--sustained",
        action="store_true",
        help="bench the sustained composite's measurement (v2/sustained/)",
    )
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


def _config_value(cfg: Mapping[str, Any], dotted: str) -> Tuple[bool, Any]:
    """(whether `cfg` sets the key `dotted`, its value)."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def first_plan_settings(
    args: argparse.Namespace, cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """The settings asked for a first v2 plan: the flags given, and the keys
    of CONFIG_SETTINGS the config sets that no flag gives; the others are
    lib/audit_v2.py's DEFAULTS, which the config's defaults equal."""
    out = {}
    for k, key in CONFIG_SETTINGS.items():
        given, value = _config_value(cfg, key)
        if given:
            out[k] = value
    out.update(
        {k: getattr(args, k) for k in SETTING_FLAGS if getattr(args, k) is not None}
    )
    return out


def sustained_first_plan_settings(
    args: argparse.Namespace, cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """The settings asked for a first sustained plan: the --sustained-*
    flags given, and the keys of SUSTAINED_CONFIG_SETTINGS the config sets
    that no flag gives; the others are lib/audit_sustained.py's DEFAULTS."""
    out = {}
    for k, key in SUSTAINED_CONFIG_SETTINGS.items():
        given, value = _config_value(cfg, key)
        if given:
            out[k] = value
    out.update(
        {
            k: getattr(args, f"sustained_{k}")
            for k in SUSTAINED_FLAGS
            if getattr(args, f"sustained_{k}", None) is not None
        }
    )
    return out


def unapplied_settings(
    m: Optional[Mapping[str, Any]], settings: Mapping[str, Any]
) -> List[str]:
    """One line per setting of `settings` that the measurement of manifest
    `m`, planned already, keeps at another value; none without one."""
    if m is None:
        return []
    have = v2.settings_of(m)
    return [
        f"{k} {settings[k]!r} (the measurement keeps {have.get(k)!r})"
        for k in settings
        if have.get(k) != settings[k]
    ]


def audit_kernels(audit: Path, spec: Any, cfg: Mapping[str, Any]) -> Any:
    """The audit set's kernels as the solution indices of the pool name them
    now (`lib.audit_v2.AuditKernels`), by which a v2 record without kernel
    identities is checked and read: every kernel of the config's pool, those
    a blocklist leaves out too. A blocklist leaves an index naming the same
    kernel, and a record planned before a blocklist named one of its kernels
    names it still: a view without that kernel would take it for a kernel of
    another build."""
    from lib.bank_pool import config_pool

    pool = config_pool(
        cfg,
        spec.library_dir,
        spec.library_stem,
        spec.hardware_device,
        apply_blocklist=False,
    )
    return v2.AuditKernels.of(
        auditlib.manifest(audit) or {}, auditlib.pool_digests(pool.kernels)
    )


def plan_blocklist(
    d: Path, cfg: Mapping[str, Any], audit: Path, identity: Any
) -> Tuple[Set[int], Dict[str, Any]]:
    """(the solutions of `identity`'s library that the run's blocklist or one
    the audit set was built with leaves out, the record of their union,
    written into `<d>/blocklists/`). V2Error when one cannot be read."""
    both, sources = _plan_blocklists(cfg, audit)
    rec = auditlib.write_blocklist_file(d, "plan", both, sources)
    return both.blocked_indices(identity.ids, identity.kernels), rec


def _plan_blocklists(cfg: Mapping[str, Any], audit: Path) -> Tuple[Any, List[Any]]:
    """(the union of the run's blocklist and the audit set's, the sources'
    records). V2Error when one cannot be read."""
    from lib import correctness as cx

    try:
        own = cx.config_blocklist(cfg) or cx.Blocklist()
        built, recs = auditlib.audit_blocklist(audit)
    except ValueError as e:
        raise v2.V2Error(str(e)) from None
    both = own.merged(built) if built else own
    sources = [{"from": "the run's config", **cx.applied_record(own)}]
    sources += [{"from": "the audit set", **r} for r in recs]
    return both, sources


def plan_guards(cfg: Mapping[str, Any], audit: Path, identity: Any) -> Dict[int, Any]:
    """{solution: guards} of `identity`'s library that the blocklists of
    `plan_blocklist` guard: no set holds one on a GEMM its guards deny."""
    both, _ = _plan_blocklists(cfg, audit)
    return both.guarded_indices(identity.ids, identity.kernels)


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

    identity = ev.library_identity(
        spec.library_dir, spec.library_stem, cfg, spec.hardware_device
    )
    kernels = audit_kernels(audit, spec, cfg)
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
        data = auditlib.model_bytes(
            path,
            str(cfg["arch"]),
            dtype,
            cfg,
            alignment_veto=not getattr(args, "without_alignment_veto", False),
        )
        sha = v2.sha256_of(data)
        if getattr(args, "name_with_sha", False):
            name = f"{name}.{sha[:12]}"
        picks = auditlib.picks(data, spec, cfg, hardware=hardware)
        new_models.append(v2.Model(name, sha, f"{source}: {path}", dict(picks)))
        ui.info("model", f"{name}: {source} {path} ({len(picks)} picks)")
    d = v2.v2_dir(audit)
    d.mkdir(parents=True, exist_ok=True)
    blocked, blocklist = plan_blocklist(d, cfg, audit, identity)
    bench = {
        "binary": str(paths.bench_binary),
        "config_yaml": str(Path(args.config).resolve()),
        "wrappers": _wrappers(d, cfg, paths, units, device),
        "stall_timeout_s": float(ro.cfg_get(cfg, "bench.stall_timeout_s")),
        "startup_grace_s": float(ro.cfg_get(cfg, "bench.startup_grace_s")),
        "library": {"dir": str(spec.library_dir), "stems": list(identity.stems)},
    }
    settings = first_plan_settings(args, cfg)
    unapplied = unapplied_settings(v2.manifest(d), settings)
    if unapplied:
        ui.warn(
            "plan",
            f"the v2 measurement in {d} was planned with other settings, which "
            f"its generations keep: {', '.join(unapplied)} not applied; move "
            f"{d} aside to measure with them",
        )
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
        blocked=blocked,
        blocklist=blocklist,
        guarded=plan_guards(cfg, audit, identity),
    )
    v2.write_progress(d)
    if gen is None:
        ui.ok("plan", "every set holds every registered pick; nothing to bench")
    else:
        g = (v2.manifest(d) or {})["generations"][gen]
        ui.ok("plan", f"generation {gen}: {g['n_gemms']} GEMMs in {g['n_parts']} parts")
    return gen


def plan_sustained(args: argparse.Namespace) -> Optional[int]:
    """Plan the sustained measurement's generation that the v2 measurement
    asks for now (lib/audit_sustained.py); returns its index, or None when
    nothing needs a bench."""
    from lib import evaluate as ev
    from lib import hardware as hwlib

    audit = Path(args.audit_dir)
    cfg, spec, _, _, _, device = _context(args)
    st, why = auditlib.status(audit)
    if st != "ok":
        raise v2.V2Error(f"the audit set is not complete: {why}")
    gemms = auditlib.evaluable_gemms(spec, cfg)
    identity = ev.library_identity(
        spec.library_dir, spec.library_stem, cfg, spec.hardware_device
    )
    try:
        pool, _ = auditlib.evaluation_pool(spec, cfg)
    except ValueError as e:
        raise v2.V2Error(str(e)) from None
    kernels = audit_kernels(audit, spec, cfg)
    in_pool = {int(k["sol_idx_global"]) for k in pool.kernels}
    d = v2.v2_dir(audit)
    measured = {
        s: auditlib.pool_medians(mm, in_pool)
        for (label, s), mm in v2.measurements(d, identity, kernels).items()
        if label == auditlib.EXCLUSIVE
    }
    settings = sustained_first_plan_settings(args, cfg)
    sd = asus.sustained_dir(audit)
    sd.mkdir(parents=True, exist_ok=True)
    blocked, blocklist = plan_blocklist(sd, cfg, audit, identity)
    unapplied = unapplied_settings(v2.manifest(sd), settings)
    if unapplied:
        ui.warn(
            "sustained",
            f"the sustained measurement in {sd} was planned with other settings, "
            f"which its generations keep: {', '.join(unapplied)} not applied; "
            f"move {sd} aside to measure with them",
        )
    peak, note = None, ""
    if v2.manifest(sd) is None and settings.get("compute_bound_tflops") is None:
        rates = hwlib.roofline_rates(cfg, device)
        peak, note = rates.peak_tflops, f"bench device {device}: {rates.peak_note}"
    gen = asus.plan(
        sd,
        v2d=d,
        gemms={s: g for (label, s), g in gemms.items() if label == auditlib.EXCLUSIVE},
        measured=measured,
        config=cfg,
        settings=settings,
        peak_tflops=peak,
        peak_note=note,
        identity=identity,
        blocked=blocked,
        blocklist=blocklist,
    )
    v2.write_progress(sd)
    m = v2.manifest(sd) or {}
    rate = (m.get("settings") or {}).get("compute_bound_tflops")
    ui.info(
        "sustained",
        "compute-bound "
        + (f"from {rate:g} TFLOP/s" if rate is not None else "never")
        + f": {m.get('compute_bound')}",
    )
    if gen is None:
        ui.ok(
            "sustained",
            "every power-capped exclusive GEMM's set holds every registered pick; "
            "nothing to bench",
        )
    else:
        g = m["generations"][gen]
        strata = ", ".join(f"{k} {n}" for k, n in sorted(g["strata"].items()))
        ui.ok(
            "sustained",
            f"generation {gen}: {g['n_gemms']} power-capped GEMMs ({strata}) in "
            f"{g['n_parts']} parts",
        )
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


def _worker_hold(
    device: str,
    config_yaml: Optional[Path],
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Any, Optional[str], Any]:
    """(the run guard the environment names, the key of `device`'s GPU in
    it, the config's bench.health settings); no guard with the health
    checks off, without a named guard or for a GPU the topology cannot
    name."""
    import yaml

    from lib import gpu_health, kfd

    env = os.environ if environ is None else environ
    cfg: Any = {}
    if config_yaml is not None:
        try:
            with Path(config_yaml).open() as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, ValueError, yaml.YAMLError):
            cfg = {}
    settings = gpu_health.Settings.of(cfg if isinstance(cfg, dict) else {})
    where = env.get(gpu_health.GUARD_DIR_VAR)
    if not settings.enabled or not where:
        return None, None, settings
    try:
        phys = kfd.physical_gpu(int(device), env)
    except (kfd.GpuMapError, ValueError):
        phys = None
    gpu = next((g for g in kfd.gpus() if g.index == phys), None)
    if phys is None or gpu is None:
        return None, None, settings
    guard = gpu_health.Guard(Path(where), "audit_v2", by=f"audit worker {os.getpid()}")
    return guard, gpu_health.gpu_key(gpu), settings


def held_check(
    device: str,
    config_yaml: Optional[Path],
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Callable[[], Optional[str]], Any]:
    """(what says why the run holds the GPU of `device`, None while it does
    not; the config's bench.health settings), from the run guard the
    environment names (lib/gpu_health.py), as the worker's stage03 runs
    read it: held while the hold holds that GPU (alone or with every GPU),
    never with the health checks off, without a named guard or for a GPU
    the topology cannot name."""
    from lib import gpu_health

    guard, key, settings = _worker_hold(device, config_yaml, environ)

    def held() -> Optional[str]:
        if guard is None:
            return None
        hold = guard.hold()
        if not gpu_health.holds(hold, key):
            return None
        assert hold is not None
        entry = gpu_health.hold_entries(hold).get(str(key)) or hold
        return "; ".join(str(p) for p in entry.get("problems") or []) or "held"

    return held, settings


def wedge_check(
    device: str,
    config_yaml: Optional[Path],
    environ: Optional[Mapping[str, str]] = None,
) -> Callable[[], Optional[str]]:
    """What says why the run holds the GPU of `device`, or the host, as
    wedged (lib/gpu_health.py `wedge_holds`), which no wait ends; None while
    it does not (as `held_check` reads the run guard)."""
    from lib import gpu_health

    guard, key, _ = _worker_hold(device, config_yaml, environ)

    def wedged() -> Optional[str]:
        if guard is None:
            return None
        hold = guard.hold()
        if not gpu_health.wedge_holds(hold, [str(key)]):
            return None
        entries = gpu_health.hold_entries(hold)
        entry = entries.get(str(key))
        if not gpu_health.is_wedge(entry):
            entry = entries.get(gpu_health.HOST) or {}
        return "; ".join(str(p) for p in entry.get("problems") or []) or "wedged"

    return wedged


def bench(args: argparse.Namespace) -> int:
    from lib import gpu_health, procs

    regime = "sustained" if args.sustained else "v2"
    d = (
        asus.sustained_dir(args.audit_dir)
        if args.sustained
        else v2.v2_dir(args.audit_dir)
    )
    m = v2.manifest(d)
    if m is None:
        ui.err("bench", f"no {regime} plan in {d}; run plan first")
        return 2
    library = config_library(m)
    try:
        v2.check_bench_library(d, m, auditlib.manifest(args.audit_dir) or {}, library)
        blocked = bench_blocked(d, m, library)
        v2.blocked_canary(m, blocked)
    except v2.V2Error as e:
        ui.err("bench", str(e))
        return 2
    b = m.get("bench") or {}
    gpu = canary_key(args.device)
    children = procs.ChildGroups()
    with procs.stop_children_on_exit(children):

        def runner(scope: str) -> v2.Runner:
            return v2.stage03_runner(
                Path(b["config_yaml"]) if b.get("config_yaml") else None,
                device=str(args.device),
                stall_timeout_s=float(b.get("stall_timeout_s", 300.0)),
                startup_grace_s=float(b.get("startup_grace_s", 900.0)),
                children=children,
                scope=scope,
                retry_hung=scope != gpu_health.NO_SCOPE,
                warmup_line=bool(v2.settings_of(m)["warmup_line"]),
                library=b.get("library") or library,
            )

        config_yaml = Path(b["config_yaml"]) if b.get("config_yaml") else None
        held, health = held_check(args.device, config_yaml)
        worker = v2.Bench(
            d,
            m,
            runner(v2.hung_scope(regime)),
            gpu,
            log=lambda s: ui.info(regime, s),
            canary_runner=runner(gpu_health.NO_SCOPE),
            held=held,
            poll_s=max(1.0, health.poll_s),
            max_wait_s=health.max_wait_s,
            wedged=wedge_check(args.device, config_yaml),
            blocked=frozenset(blocked),
        )
        try:
            n = v2.work(worker)
        except v2.Unmeasured as e:
            ui.err("bench", f"GPU {gpu}: {e}; the worker stops")
            ui.info("bench", v2.status_line(d, regime))
            return 1
    ui.ok("bench", f"GPU {gpu}: benched {n} part(s); {v2.status_line(d, regime)}")
    return 0


def bench_blocked(
    d: Path, m: Mapping[str, Any], library: Optional[Mapping[str, Any]] = None
) -> Set[int]:
    """The solutions the latest blocklist of the measurement in `d` (its
    manifest `m`'s `blocklist`, written by plan) leaves out, in the bench's
    library (`bench.library`, else `library`); none for a measurement
    planned before plans recorded one, or without a library, which benches
    nothing (`lib.audit_v2.check_bench_library`). V2Error when it cannot be
    read, or its file no longer holds what the sha256 the manifest records
    says (as `lib.audit.audit_blocklist` checks the audit set's)."""
    import hashlib

    from lib import correctness as cx
    from lib import kernel_identity as ki

    rec = m.get("blocklist")
    lib = (m.get("bench") or {}).get("library") or library
    if not rec or not lib:
        return set()
    try:
        path = Path(d) / str(rec["file"])
        data = path.read_bytes()
        same = hashlib.sha256(data).hexdigest() == rec.get("sha256")
        blocklist = cx.read_list(path) if same else None
        identity = ki.library_identity(lib["dir"], lib["stems"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise v2.V2Error(f"cannot read the blocklist of {d / v2.MANIFEST}: {e}")
    if blocklist is None:
        raise v2.V2Error(
            f"the blocklist {path} is not the one {d / v2.MANIFEST} records (its "
            f"sha256 differs): no bench goes on with another blocklist than its "
            f"plan's; plan again to record the blocklist that applies now"
        )
    return blocklist.blocked_indices(identity.ids, identity.kernels)


def worker_env(d: Path, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment of run's bench workers: `environ` (default this
    process's) with the run guard, session and process tree an orchestrator
    names (lib/gpu_health.py, lib/host_load.py), else a guard in the v2
    directory `d`, a session of this invocation and this process's tree."""
    from lib import gpu_health, host_load

    out = dict(os.environ if environ is None else environ)
    if not out.get(gpu_health.GUARD_DIR_VAR):
        out[gpu_health.GUARD_DIR_VAR] = str(Path(d) / gpu_health.GUARD_DIR)
    if not out.get(gpu_health.SESSION_VAR):
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out[gpu_health.SESSION_VAR] = f"audit_v2_{stamp}_{os.getpid()}"
    if host_load.run_root(environ=out) is None:
        out[host_load.RUN_ROOT_VAR] = str(os.getpid())
    return out


# A phase of `run`: the measurement's directory, the bench worker's extra
# arguments and what plans it before its workers start (None: planned).
Phase = Tuple[Path, List[str], Optional[Callable[[], Any]]]


def _phases(args: argparse.Namespace) -> List[Phase]:
    out: List[Phase] = [(v2.v2_dir(args.audit_dir), [], None)]
    if args.sustained:
        out.append(
            (
                asus.sustained_dir(args.audit_dir),
                ["--sustained"],
                lambda: plan_sustained(args),
            )
        )
    return out


def _summary(d: Path, regime: str, t0: float) -> int:
    """Report a phase whose workers ended; 1 when GEMM benches are left
    (their workers stopped: the run held their GPUs, or they could not
    bench), else 0."""
    p = v2.write_progress(d)
    ui.ok(
        "run",
        f"{v2.status_line(d, regime)}; {p['gemms_done']} of {p['gemms_total']} "
        f"done in {ui.fmt_dur(time.time() - t0)}",
    )
    left = v2.pending(d)
    if left:
        ui.err(
            "run",
            f"{len(left)} GEMM bench(es) of the {regime} measurement are left: no "
            f"worker could bench them (see the worker logs in {d})",
        )
    return 1 if left else 0


def run_on_devices(
    args: argparse.Namespace,
    phases: Sequence[Phase],
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """The bench workers of `run --devices`, phase by phase: one per device
    whose GPU no other process tree leases, under the lease of its GPU, held
    over every phase, each seeing that GPU alone, in `env` (`worker_env`)."""
    from lib import gpu_lease, kfd

    try:
        devices = [int(x) for x in str(args.devices).split(",") if x.strip()]
    except ValueError:
        ui.err("run", f"--devices {args.devices!r}: give device indices")
        return 2
    if not devices:
        ui.err("run", "--devices names no device")
        return 2
    env = env if env is not None else worker_env(v2.v2_dir(args.audit_dir))
    bad = 0
    try:
        with gpu_lease.hold_usable(
            devices,
            f"audit v2 bench of {args.audit_dir}",
            args.audit_dir,
            wait_s=args.wait,
            report=lambda m: ui.warn("lease", m),
        ) as leases:
            for d, extra, prepare in phases:
                regime = "sustained" if extra else "v2"
                if prepare is not None:
                    try:
                        prepare()
                    except v2.V2Error as e:
                        ui.err(regime, str(e))
                        return 2
                    if not v2.pending(d):
                        continue
                t0 = time.time()
                workers = []
                for dev in leases.devices:
                    gpu = kfd.physical_gpu(dev)
                    device, one = kfd.one_gpu_launch(dev, env)
                    cmd = [sys.executable, str(Path(__file__).resolve()), "bench"]
                    cmd += ["--audit-dir", str(args.audit_dir), "--device", device]
                    name = f"gpu{gpu}" if gpu is not None else f"device{dev}"
                    log = (d / f"worker_{name}.log").open("a")
                    proc = subprocess.Popen(
                        cmd + extra, stdout=log, stderr=subprocess.STDOUT, env=one
                    )
                    workers.append((dev, proc, log))
                    ui.info(
                        "run",
                        f"device {dev}: {regime} worker pid {proc.pid}, log {log.name}",
                    )
                for dev, proc, log in workers:
                    rc = proc.wait()
                    log.close()
                    (ui.ok if rc == 0 else ui.err)(
                        "run", f"device {dev}: {regime} worker exited {rc}"
                    )
                    bad += rc != 0
                bad += _summary(d, regime, t0)
    except gpu_lease.LeaseError as e:
        ui.err("run", str(e))
        return 2
    return 1 if bad else 0


def run_on_gpus(
    args: argparse.Namespace,
    phases: Sequence[Phase],
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """The bench workers of `run --gpus`, phase by phase: one per GPU, each
    under that GPU's lease (scripts/gpu_run.py), in `env` (`worker_env`)."""
    gpus = [int(g) for g in str(args.gpus).split(",") if g.strip()]
    env = env if env is not None else worker_env(v2.v2_dir(args.audit_dir))
    bad = 0
    for d, extra, prepare in phases:
        regime = "sustained" if extra else "v2"
        if prepare is not None:
            try:
                prepare()
            except v2.V2Error as e:
                ui.err(regime, str(e))
                return 2
            if not v2.pending(d):
                continue
        t0 = time.time()
        procs_ = []
        for g in gpus:
            cmd = [sys.executable, str(GPU_RUN), "--gpu", str(g)]
            cmd += ["--wait", str(args.wait)]
            cmd += ["--purpose", f"audit {regime} bench of {args.audit_dir}", "--"]
            cmd += [sys.executable, str(Path(__file__).resolve()), "bench"]
            cmd += ["--audit-dir", str(args.audit_dir), "--device", "0", *extra]
            log = (d / f"worker_gpu{g}.log").open("a")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            procs_.append((g, proc, log))
            ui.info("run", f"GPU {g}: {regime} worker pid {proc.pid}, log {log.name}")
        for g, p, log in procs_:
            rc = p.wait()
            log.close()
            (ui.ok if rc == 0 else ui.err)(
                "run", f"GPU {g}: {regime} worker exited {rc}"
            )
            bad += rc != 0
        bad += _summary(d, regime, t0)
    return 1 if bad else 0


def run(args: argparse.Namespace) -> int:
    plan(args)
    phases = _phases(args)
    env = worker_env(v2.v2_dir(args.audit_dir))
    if args.devices is not None:
        return run_on_devices(args, phases, env)
    return run_on_gpus(args, phases, env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.cmd == "status":
        print(v2.status_line(v2.v2_dir(args.audit_dir)))
        print(v2.status_line(asus.sustained_dir(args.audit_dir), "sustained"))
        return 0
    if args.cmd == "bench":
        return bench(args)
    try:
        if args.cmd == "plan":
            plan(args)
            if args.sustained and v2.manifest(v2.v2_dir(args.audit_dir)):
                done = v2.progress(v2.v2_dir(args.audit_dir))
                if done["gemms_done"] == done["gemms_total"]:
                    plan_sustained(args)
            return 0
        return run(args)
    except v2.V2Error as e:
        ui.err("v2", str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
