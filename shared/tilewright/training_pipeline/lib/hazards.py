# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernels of the hazard classes, the canary that runs them alone, and the
quarantine of those that hang or fault.

A kernel can stop its device in a way no signal to the process ends: the
device takes no more work until it is reset, and the other GPUs of the host
can stop with it. A bench process holds many kernels, so one such kernel
would take a whole block, and the host, with it. A kernel of the pool whose
parameters put it in a class of `HAZARD_CLASSES` (parameter values checked
as a precaution) therefore runs alone first
(stages/stage00a_hazard_canary.py), before any other stage of an invocation
launches it: one hipblaslt-bench process per kernel on the shapes of
`canary_shapes`, one such process at a time on each GPU, each GPU leased
(lib/gpu_lease.py), under the stall watchdog of the benches (lib/procs.py).
The class comes from the kernel's logic parameters where the library records
them (`clusterDim`) and from its solution name otherwise (`name_values`:
TensileLite names a parameter by the capitals of its name, ClusterDim `CD`,
PrefetchGL2 `PGL`, TDMIterateMode `TDMIM`).

A kernel is quarantined when its process prints nothing for the canary's
timeout (`hang`), ends by a crash, a fatal GPU error or a non-zero exit
(`fault`), or runs none of the canary's problems (`untested`: it never ran
alone, so it does not run among others either). A quarantined kernel leaves
every plan: the orchestrator adds the quarantine to the invocation's
blocklist (lib/correctness.py), and the correctness gate and the audit set
leave it out too. A kernel whose run was cut short from outside (its bench
killed by a signal the stage did not send) has no verdict: it stays out of
the invocation all the same (`kept_out`), and the next one runs it alone
again.

Records go to `canary.jsonl` in the canary's cache directory
(`CanaryStore`), shared by every run that names the directory: a `start`
record before a kernel's process starts and an `end` record with its
verdict after, each naming the process that wrote it (`process_owner`). A
quarantine holds for the kernel's name on the run's `arch` whatever build,
device or protocol later runs see: a kernel that stopped a device once may
do so again. A run that never ended (its process, the stage or the host went
down while it ran) quarantines its kernel as well; one whose process still
runs, another invocation's, does not, and its kernel stays out of this
invocation unchecked unless that run passes meanwhile. A pass holds only for
the kernel id, device, code digest and protocol it was made with. Removing a
kernel's lines from the file checks it again.
"""
from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import kernel_identity as ki
from . import kfd, procs
from .bench_log import LogProbe, parse_log_file
from .bench_yaml import canary_line, name_entry, render_bench_line
from .dat import mx_block_size_for_scale_mode
from .shapes import Shape

HAZARDS_DIR = "hazards"
CACHE_FILE = "canary.jsonl"
QUARANTINE_FILE = "quarantine.json"
VERDICTS_FILE = "verdicts.json"
SUMMARY_FILE = "canary_summary.json"
DEFAULTS: Dict[str, Any] = {"enabled": True, "timeout_s": 300.0, "runs": 1}
# The `source` of the canary's blocklist entries (lib/correctness.py).
SOURCE = "hazard canary"
PROTOCOL_VERSION = 1
# Verdicts that quarantine a kernel; `interrupted` and `error` close a run
# without blaming it.
QUARANTINE_VERDICTS = ("hang", "fault", "untested")
# Seconds a killed bench process gets to be reaped, beyond the wait of the
# kill itself (lib/procs.py), before its GPU counts as unusable.
REAP_S = 60.0
# A process of the GPU driver's process list counts as stuck when every one
# of STUCK_SAMPLES readings, STUCK_INTERVAL_S apart, finds it in
# uninterruptible sleep.
STUCK_SAMPLES = 4
STUCK_INTERVAL_S = 0.5


# ── classes ──────────────────────────────────────────────────────────────────


def name_values(name: str, abbrev: str) -> Optional[Tuple[int, ...]]:
    """The values of the parameter TensileLite abbreviates `abbrev` in the
    solution name `name` (`..._CD2_4_...` gives (2, 4) for `CD`; `n1` is
    -1); None when the name does not carry it."""
    m = re.search(rf"(?:^|_){re.escape(abbrev)}(n?\d+(?:_n?\d+)*)(?=_|$)", str(name))
    if m is None:
        return None
    return tuple(
        -int(v[1:]) if v.startswith("n") else int(v) for v in m.group(1).split("_")
    )


def _name(kernel: Mapping[str, Any]) -> str:
    return str(kernel.get("kernel_name") or "")


def _second_cluster_dims(kernel: Mapping[str, Any]) -> Set[int]:
    """The second dimension of the kernel's ClusterDim as its logic file's
    `clusterDim` and its name give it."""
    out: Set[int] = set()
    attrs = kernel.get("attributes") or {}
    if "cluster_dim_y" in attrs:
        out.add(int(attrs["cluster_dim_y"]))
    named = name_values(_name(kernel), "CD")
    if named is not None and len(named) >= 2:
        out.add(named[1])
    return out


def _named_value(kernel: Mapping[str, Any], abbrev: str) -> Optional[int]:
    named = name_values(_name(kernel), abbrev)
    return named[0] if named else None


@dataclass(frozen=True)
class HazardClass:
    name: str
    text: str
    test: Callable[[Mapping[str, Any]], bool]


HAZARD_CLASSES: Tuple[HazardClass, ...] = (
    HazardClass(
        "cluster_dim_y4",
        "ClusterDim with a second dimension of 4",
        lambda k: 4 in _second_cluster_dims(k),
    ),
    HazardClass(
        "prefetch_gl2_2", "PrefetchGL2 2", lambda k: _named_value(k, "PGL") == 2
    ),
    HazardClass(
        "tdm_iterate_mode_3",
        "TDMIterateMode 3",
        lambda k: _named_value(k, "TDMIM") == 3,
    ),
)


def hazard_classes(kernel: Mapping[str, Any]) -> Tuple[str, ...]:
    """The names of the classes of HAZARD_CLASSES a kernel
    (`lib.dat.kernel_dat_info`) belongs to."""
    return tuple(c.name for c in HAZARD_CLASSES if c.test(kernel))


def class_text(names: Iterable[str]) -> str:
    texts = {c.name: c.text for c in HAZARD_CLASSES}
    return ", ".join(texts.get(n, n) for n in names)


# ── settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """The config's `hazards` section over DEFAULTS."""

    enabled: bool = DEFAULTS["enabled"]
    timeout_s: float = DEFAULTS["timeout_s"]
    runs: int = DEFAULTS["runs"]
    devices: Optional[Tuple[int, ...]] = None
    cache_dir: Optional[str] = None

    @classmethod
    def of(cls, cfg: Optional[Mapping[str, Any]]) -> "Settings":
        sec = dict((cfg or {}).get("hazards") or {})
        devices = sec.get("devices")
        return cls(
            enabled=bool(sec.get("enabled", DEFAULTS["enabled"])),
            timeout_s=float(sec.get("timeout_s") or DEFAULTS["timeout_s"]),
            runs=int(sec.get("runs") or DEFAULTS["runs"]),
            devices=tuple(int(d) for d in devices) if devices else None,
            cache_dir=str(sec["cache_dir"]) if sec.get("cache_dir") else None,
        )


def config_arch(cfg: Mapping[str, Any]) -> str:
    """The library subtree a quarantine holds for: the config's `arch`
    without target features."""
    return str(cfg.get("arch") or "").split(":", 1)[0]


# ── the store ────────────────────────────────────────────────────────────────


def _boot_id() -> Optional[str]:
    try:
        text = (kfd.PROC / "sys" / "kernel" / "random" / "boot_id").read_text()
    except OSError:
        return None
    return text.strip() or None


def _start_ticks(pid: int) -> Optional[int]:
    """When process `pid` started, in clock ticks after boot."""
    try:
        text = (kfd.PROC / str(int(pid)) / "stat").read_text()
        return int(text[text.rindex(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def process_owner(pid: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """What names process `pid` (default this one) on this host, also once
    its pid is reused: the boot it runs in, its pid and its start time; None
    when the host does not tell."""
    pid = os.getpid() if pid is None else int(pid)
    boot, start = _boot_id(), _start_ticks(pid)
    if boot is None or start is None:
        return None
    return {"boot": boot, "pid": pid, "start": start}


def owner_alive(owner: Any) -> bool:
    """Whether the process a record's `owner` (`process_owner`) names still
    runs: false for one of another host or boot, and for a record without
    one."""
    if not isinstance(owner, Mapping) or owner.get("boot") != _boot_id():
        return False
    try:
        return _start_ticks(int(owner["pid"])) == owner.get("start")
    except (KeyError, TypeError, ValueError):
        return False


class CanaryStore:
    """Append-only canary records in `<dir>/canary.jsonl`, one JSON object
    per line, written under an exclusive lock so that runs sharing the
    directory can add to it at once."""

    def __init__(self, directory: Any) -> None:
        self.dir = Path(directory)
        self.path = self.dir / CACHE_FILE
        self._records: Optional[List[Dict[str, Any]]] = None

    def records(self) -> List[Dict[str, Any]]:
        if self._records is None:
            out = []
            if self.path.is_file():
                for line in self.path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and rec.get("token") and rec.get("name"):
                        out.append(rec)
            self._records = out
        return self._records

    def refresh(self) -> None:
        """Read the file again: other invocations may have added to it."""
        self._records = None

    def add(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in records)
        with open(self.path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._records = None

    def _open_starts(self, arch: str) -> List[Dict[str, Any]]:
        """The start records on `arch` of runs without an end."""
        ended = {str(r["token"]) for r in self.records() if r.get("event") == "end"}
        return [
            r
            for r in self.records()
            if r.get("arch") == arch
            and r.get("event") == "start"
            and str(r["token"]) not in ended
        ]

    def quarantined(
        self, arch: str, running: Iterable[str] = ()
    ) -> Dict[str, Dict[str, Any]]:
        """{kernel name: the record that quarantines it} on `arch`: the end
        of a run with a verdict of QUARANTINE_VERDICTS, or the start of a run
        that has no end and whose process no longer runs (`owner_alive`),
        other than the runs of `running` (tokens of runs still going)."""
        out: Dict[str, Dict[str, Any]] = {}
        for rec in self.records():
            if (
                rec.get("arch") == arch
                and rec.get("event") == "end"
                and rec.get("verdict") in QUARANTINE_VERDICTS
            ):
                out.setdefault(str(rec["name"]), rec)
        live = set(running)
        for rec in self._open_starts(arch):
            if str(rec["token"]) in live or owner_alive(rec.get("owner")):
                continue
            out.setdefault(
                str(rec["name"]),
                dict(
                    rec,
                    verdict="unfinished",
                    reason="its canary run never ended: the bench, the stage or "
                    "the host went down while it ran",
                ),
            )
        return out

    def in_flight(self, arch: str) -> Set[str]:
        """The names of the kernels on `arch` whose canary run another
        process that still runs has started and not ended."""
        return {
            str(r["name"])
            for r in self._open_starts(arch)
            if owner_alive(r.get("owner"))
        }

    def passed(
        self, kid: str, device: str, code: str, proto: str
    ) -> Optional[Dict[str, Any]]:
        """A pass of the kernel `kid` on `device` with code digest `code`
        under protocol `proto`."""
        for rec in self.records():
            if (
                rec.get("event") == "end"
                and rec.get("verdict") == "pass"
                and (rec.get("kernel_id"), rec.get("device")) == (kid, device)
                and (rec.get("code"), rec.get("protocol")) == (code, proto)
            ):
                return rec
        return None


def quarantined_names(cfg: Mapping[str, Any]) -> Set[str]:
    """The kernel names the canary keeps out of the config's invocation:
    those its store (`hazards.cache_dir`, which the orchestrator resolves)
    quarantines on the config's `arch`, and the canary's entries of the
    invocation's blocklist (`correctness.blocklist`, once written), which
    also hold the kernels the invocation keeps out unchecked; none with the
    canary off."""
    from . import correctness as cx

    st = Settings.of(cfg)
    if not st.enabled:
        return set()
    out: Set[str] = set()
    if st.cache_dir:
        out |= set(CanaryStore(st.cache_dir).quarantined(config_arch(cfg)))
    sec = cfg.get("correctness")
    path = sec.get("blocklist") if isinstance(sec, Mapping) else None
    if path and Path(str(path)).is_file():
        try:
            listed = cx.read_list(path)
        except ValueError:
            return out
        out |= {e.name for e in listed.entries if e.source == SOURCE}
    return out


def without_quarantined(
    cfg: Mapping[str, Any], kernels: Iterable[Mapping[str, Any]]
) -> List[Mapping[str, Any]]:
    held = quarantined_names(cfg)
    return [k for k in kernels if _name(k) not in held]


# ── protocol and shapes ──────────────────────────────────────────────────────


def protocol(runs: int, host: str) -> str:
    """Digest of what a pass means: the canary's shape rule, its runs per
    kernel and the hipBLASLt host libraries that launch the kernels
    (`lib.correctness_gate.host_digest`)."""
    body = {"version": PROTOCOL_VERSION, "runs": int(runs), "host": host}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def canary_shapes(kernel: Mapping[str, Any], mx_block: int = 0) -> List[Shape]:
    """The problems a kernel runs alone: two by two of its macro tiles per
    cluster, K four DepthU iterations (a multiple of the MX block), then
    partial tiles in M and N with a batch of 2."""
    attrs = kernel.get("attributes") or {}
    mt_m = max(1, int(kernel.get("mt_m") or 1))
    mt_n = max(1, int(kernel.get("mt_n") or 1))
    depth_u = max(1, int(kernel.get("mt_k") or 1))
    step = depth_u * max(1, int(mx_block)) // math.gcd(depth_u, max(1, int(mx_block)))
    cx = max(1, int(attrs.get("cluster_dim_x") or 1))
    cy = max(1, int(attrs.get("cluster_dim_y") or 1))
    named = name_values(_name(kernel), "CD")
    if named is not None and len(named) >= 2:
        cx, cy = max(cx, named[0], 1), max(cy, named[1], 1)
    m, n, k = 2 * mt_m * cx, 2 * mt_n * cy, 4 * step
    return [
        Shape(m, n, k, 1),
        Shape(m + mt_m // 2 + 1, n + mt_n // 2 + 1, k, 2),
    ]


def canary_yaml(
    kernel: Mapping[str, Any], cfg: Mapping[str, Any], run: int
) -> List[str]:
    """The bench lines of one canary run of `kernel` (solution index
    `sol_idx_global`): its `canary_shapes` in the config's problem type,
    each timing the kernel alone without a screen."""
    hl = cfg["hipblaslt"]
    block = max(
        mx_block_size_for_scale_mode(hl.get("scaleA")),
        mx_block_size_for_scale_mode(hl.get("scaleB")),
    )
    sid = int(kernel["sol_idx_global"])
    out = []
    for i, shape in enumerate(canary_shapes(kernel, block)):
        line = canary_line(render_bench_line(shape, dict(cfg)), kernel=sid)
        assert line is not None
        out.append(name_entry(line, f"canary_{sid}_r{run}_{i}"))
    return out


# ── stuck processes ──────────────────────────────────────────────────────────


def _state(pid: int) -> Optional[str]:
    try:
        text = (kfd.PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    try:
        return text[text.rindex(")") + 2]
    except (ValueError, IndexError):
        return None


def _comm(pid: int) -> str:
    try:
        return (kfd.PROC / str(pid) / "comm").read_text().strip()
    except OSError:
        return "?"


def stuck_processes(
    samples: int = STUCK_SAMPLES, interval_s: float = STUCK_INTERVAL_S
) -> List[Tuple[int, str]]:
    """(pid, command) of the processes the GPU driver lists (KFD's process
    directory) that stay in uninterruptible sleep over `samples` readings
    `interval_s` apart: a kernel launched now could be blamed for what
    stopped them."""
    proc_dir = kfd.kfd_root() / "proc"
    try:
        pids = sorted(int(p.name) for p in proc_dir.iterdir() if p.name.isdigit())
    except OSError:
        return []
    stuck = {p for p in pids if _state(p) == "D"}
    for _ in range(max(0, samples - 1)):
        if not stuck:
            break
        time.sleep(interval_s)
        stuck = {p for p in stuck if _state(p) == "D"}
    return [(p, _comm(p)) for p in sorted(stuck)]


# ── one canary run ───────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """What one bench process of the canary did: `verdict` pass, hang,
    fault, untested or interrupted; `reaped` false for a process that did
    not end after it was killed."""

    verdict: str
    reason: str
    rc: Optional[int]
    problems: int = 0
    unsupported: int = 0
    wall_s: float = 0.0
    reaped: bool = True
    log: str = ""
    names: List[str] = field(default_factory=list)
    foreign_wait_s: float = 0.0

    def evidence(self) -> Dict[str, Any]:
        return {
            "rc": self.rc,
            "problems": self.problems,
            "unsupported": self.unsupported,
            "wall_s": round(self.wall_s, 3),
            "foreign_wait_s": round(self.foreign_wait_s, 1),
            "log": self.log,
        }


def judge(
    sid: int,
    log: Path,
    rc: int,
    reason: Optional[str],
    timeout_s: float,
    stopping: bool,
) -> RunResult:
    """The verdict of a canary run of solution `sid` from the watchdog's
    `reason` (lib/procs.py), the exit status `rc` and the problems of its
    log; `stopping` when the stage stopped the run."""
    problems = parse_log_file(log) if log.is_file() else []
    ran = [p for p in problems if any(t.sol_index == sid for t in p.tested)]
    names = sorted(
        {t.sol_name for p in problems for t in p.tested if t.sol_index == sid}
    )
    out = RunResult(
        verdict="pass",
        reason="",
        rc=rc,
        problems=len(ran),
        unsupported=len(problems) - len(ran),
        log=log.name,
        names=[n for n in names if n],
    )
    if stopping:
        out.verdict, out.reason = "interrupted", "the stage stopped the run"
    elif reason == "stall":
        out.verdict = "hang"
        out.reason = f"its bench printed nothing for {timeout_s:g} s"
    elif reason == "fatal":
        out.verdict, out.reason = "fault", "its bench reported a fatal GPU error"
    elif rc != 0:
        sig = procs.killed_by(rc)
        if sig is not None:
            out.verdict = "interrupted"
            out.reason = f"its bench was killed by {sig.name} from outside"
        else:
            out.verdict, out.reason = "fault", f"its bench ended with rc={rc}"
    elif not ran:
        out.verdict = "untested"
        out.reason = "its bench ran none of the canary's problems with it"
    return out


def run_kernel(
    lines: Sequence[str],
    sid: int,
    device: int,
    bench_binary: Path,
    env: Mapping[str, str],
    log: Path,
    timeout_s: float,
    startup_grace_s: float,
    children: procs.ChildGroups,
    stopping: Callable[[], bool] = lambda: False,
) -> RunResult:
    """Bench `lines` (one kernel's) in one hipblaslt-bench process on
    device `device` of `env`, which sees that GPU alone, its output in
    `log`, under the stall watchdog: killed once it prints nothing for
    `timeout_s` (before it has loaded its library, not before
    `startup_grace_s`: a kernel that hangs does so after the load) or prints
    a fatal GPU error."""
    log.parent.mkdir(parents=True, exist_ok=True)
    yaml_path = log.with_suffix(".yaml")
    yaml_path.write_text(
        "".join(ln if ln.endswith("\n") else ln + "\n" for ln in lines)
    )
    bench_device, launch_env = kfd.one_gpu_launch(device, env)
    cmd = [str(bench_binary), "--device", bench_device, "--yaml", str(yaml_path)]
    probe = LogProbe(log, 0)
    t0 = time.time()
    with log.open("w", buffering=1) as out:
        proc = children.popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(launch_env),
        )
        rc, reason = procs.wait_with_stall_watchdog(
            proc,
            lambda: procs.file_size(log),
            timeout_s,
            poll_s=procs.poll_interval(timeout_s),
            probe_fatal=probe.fatal_seen,
            startup_grace_s=startup_grace_s,
            probe_progress=probe.loaded,
        )
    reaped = proc.poll() is not None
    if not reaped:
        try:
            proc.wait(timeout=REAP_S)
            reaped = True
        except subprocess.TimeoutExpired:
            reaped = False
    result = judge(sid, log, rc, reason, timeout_s, stopping())
    result.wall_s = time.time() - t0
    result.reaped = reaped
    if not reaped:
        result.reason += "; the process did not end after SIGKILL"
    return result


# ── records ──────────────────────────────────────────────────────────────────


def kernel_record(
    kernel: Mapping[str, Any], classes: Sequence[str], **extra: Any
) -> Dict[str, Any]:
    return {
        "kernel_id": ki.kernel_id(kernel),
        "name": _name(kernel),
        "signature": ki.signature_label(kernel),
        "classes": list(classes),
        **extra,
    }


def quarantine_reason(rec: Mapping[str, Any]) -> str:
    """What a quarantining record says of its kernel, for the blocklist."""
    verdict = rec.get("verdict") or "unfinished"
    why = rec.get("reason") or ""
    return f"hazard canary: {verdict} ({class_text(rec.get('classes') or ())})" + (
        f": {why}" if why else ""
    )


# ── one canary pass over a pool ──────────────────────────────────────────────


class CanaryError(RuntimeError):
    """The canary cannot run: no kernel is blamed."""


@dataclass
class Spec:
    """Where one canary pass reads the pool and writes its results."""

    cfg: Mapping[str, Any]
    dat_dir: Path
    library_stem: str
    out_dir: Path
    cache_dir: Path
    bench_binary: Path
    build_dir: Optional[Path] = None
    hardware_device: int = 0
    devices: Sequence[int] = (0,)
    lease_wait_s: float = 0.0
    startup_grace_s: float = 900.0
    dry_run: bool = False


def _write(path: Path, doc: Any) -> None:
    from .fs import atomic_write_text

    atomic_write_text(path, json.dumps(doc, indent=1) + "\n")


def run_canary(
    spec: Spec,
    say: Callable[[str], None] = lambda _m: None,
    warn: Callable[[str], None] = lambda _m: None,
    children: Optional[procs.ChildGroups] = None,
) -> Dict[str, Any]:
    """Run every kernel of the pool's hazard classes that has no verdict
    alone (see the module docstring), record the verdicts in the store and
    write `quarantine.json`, `verdicts.json` and `canary_summary.json` into
    `spec.out_dir`; returns the summary, whose `rc` is the stage's exit
    status: 0, or 3 when a GPU became unusable (a killed bench that did not
    end, a control kernel that failed before the first kernel, or after a
    hang or fault one of: a control kernel that failed or ran no canary
    problem, a process stuck in the GPU driver, a pool without a kernel
    outside the hazard classes to check the device with) or a process stuck
    in the GPU driver kept the canary from starting. Besides the
    quarantine, `quarantine.json` holds the hazard-class kernels left
    without a verdict, outside a dry run: their runs were cut short, or
    another invocation runs them, so they stay out of the invocation.
    CanaryError when the canary cannot run at all, and when no control
    kernel runs a canary problem before the first kernel: the canary's
    bench lines do not fit the library."""
    from . import bench_env as benv
    from . import correctness as cx
    from . import correctness_gate as cg
    from . import gpu_lease
    from .bank_pool import bank_device, config_identity, config_pool

    t0 = time.time()
    cfg = spec.cfg
    st = Settings.of(cfg)
    arch = config_arch(cfg)
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    children = children if children is not None else procs.ChildGroups()
    try:
        pool = config_pool(
            cfg,
            spec.dat_dir,
            spec.library_stem,
            spec.hardware_device,
            apply_blocklist=False,
        )
        listed = cx.lists_blocklist(cfg)
    except (OSError, ValueError, RuntimeError) as e:
        raise CanaryError(f"cannot read the kernel pool: {e}") from None
    kernels = {int(k["sol_idx_global"]): k for k in pool.kernels}
    classes = {sid: hazard_classes(k) for sid, k in kernels.items()}
    hazard = sorted(
        s for s, c in classes.items() if c and not listed.blocks(kernels[s])
    )
    store = CanaryStore(spec.cache_dir)
    held = store.quarantined(arch)
    busy = store.in_flight(arch)
    summary: Dict[str, Any] = {
        "library_stem": spec.library_stem,
        "arch": arch,
        "classes": {c.name: c.text for c in HAZARD_CLASSES},
        "pool": len(kernels),
        "hazard": len(hazard),
        "by_class": dict(
            collections.Counter(c for s in hazard for c in classes[s]).most_common()
        ),
        "listed": sum(1 for s in classes if classes[s] and listed.blocks(kernels[s])),
        "quarantined_before": sum(1 for s in hazard if _name(kernels[s]) in held),
        "passed_before": 0,
        "running_elsewhere": 0,
        "checked": 0,
        "devices": [],
        "device": None,
        "protocol": None,
        "settings": {"timeout_s": st.timeout_s, "runs": st.runs},
        "stopped": None,
        "dry_run": bool(spec.dry_run),
        "rc": 0,
    }
    found: Dict[int, Dict[str, Any]] = {}
    pending: List[int] = []
    elsewhere: List[int] = []
    device = proto = None
    code: Dict[int, str] = {}
    if hazard:
        try:
            identity = config_identity(
                cfg, spec.dat_dir, spec.library_stem, spec.hardware_device
            )
            owner: Dict[int, str] = {}
            for stem in identity.stems:
                for sid in identity.file_ids.get(stem, {}):
                    owner.setdefault(int(sid), stem)
            stems = sorted({owner[s] for s in hazard})
            digests = cg.code_digests(spec.dat_dir, stems)
            device = cg.device_key(bank_device(cfg, spec.hardware_device))
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            raise CanaryError(f"cannot identify the pool's kernels: {e}") from None
        code = {s: digests[owner[s]] for s in hazard}
        proto = protocol(st.runs, cg.host_digest(spec.build_dir))
        summary.update(device=device, protocol=proto)
        for sid in hazard:
            k = kernels[sid]
            if _name(k) in held:
                continue
            rec = store.passed(ki.kernel_id(k), device, code[sid], proto)
            if rec is not None:
                found[sid] = rec
            elif _name(k) in busy:
                elsewhere.append(sid)
            else:
                pending.append(sid)
        summary["passed_before"] = len(found)
        summary["running_elsewhere"] = len(elsewhere)
    say(
        f"pool {len(kernels):,} kernels: {len(hazard):,} of the hazard classes "
        f"({summary['quarantined_before']:,} quarantined, {len(found):,} passed "
        f"before, {len(pending):,} to run alone"
        + (
            f", {len(elsewhere):,} running alone in another invocation"
            if elsewhere
            else ""
        )
        + ")"
    )
    stop = threading.Event()
    lock = threading.Lock()
    ran: Dict[int, RunResult] = {}
    problems: List[str] = []
    errors: List[BaseException] = []
    if pending and spec.dry_run:
        warn(f"dry run: {len(pending):,} hazard-class kernels left without a verdict")
    elif pending:
        stuck = stuck_processes()
        if stuck:
            what = ", ".join(f"{p} ({c})" for p, c in stuck[:4])
            summary["stopped"] = (
                f"processes the GPU driver lists stay in uninterruptible sleep "
                f"({what}): a kernel run now could be blamed for what stopped them; "
                f"no kernel ran"
            )
            summary["rc"] = 3
            pending = []
    if pending and not spec.dry_run:
        gpus = kfd.bench_gpus(spec.devices)
        if not gpus:
            raise CanaryError(f"devices {list(spec.devices)} reach no GPU here")
        summary["devices"] = list(spec.devices)
        env = benv.bench_env_with(benv.runtime_env(cfg))
        controls = [
            s
            for s in sorted(kernels)
            if not classes[s] and not listed.blocks(kernels[s])
        ][:3]
        queue: "collections.deque[int]" = collections.deque(pending)
        logs = out_dir / "logs"
        owner = process_owner()

        def record(sid: int, token: str, run: int, gpu: int, **end: Any) -> None:
            k = kernels[sid]
            base = kernel_record(
                k,
                classes[sid],
                event="end" if end else "start",
                token=token,
                arch=arch,
                device=device,
                gpu=gpu,
                code=code[sid],
                protocol=proto,
                run=run,
                time=kfd.utc_iso(),
                owner=owner,
            )
            store.add([dict(base, **end)])

        def control(d: int, tag: str) -> Optional[bool]:
            """Whether device `d` runs a kernel of no hazard class: the first
            of `controls` that runs a canary problem passes; None when none
            of them runs one."""
            for sid in controls:
                r = run_kernel(
                    canary_yaml(kernels[sid], cfg, 0),
                    sid,
                    d,
                    spec.bench_binary,
                    env,
                    logs / f"control_{tag}_dev{d}.log",
                    st.timeout_s,
                    spec.startup_grace_s,
                    children,
                    stop.is_set,
                )
                if r.verdict == "untested":
                    continue
                if r.verdict not in ("pass", "interrupted"):
                    with lock:
                        problems.append(
                            f"device {d}: the control kernel "
                            f"{ki.label_of_name(_name(kernels[sid]))} did not pass "
                            f"({r.verdict}: {r.reason})"
                        )
                return r.verdict == "pass" or stop.is_set()
            return None

        def after(d: int, sid: int, verdict: str) -> Optional[str]:
            """Why device `d` takes no more kernels after `sid`'s hang or
            fault: a process stuck in the GPU driver, no kernel outside the
            hazard classes to check it with, or no control kernel that
            passes; None when one passes."""
            stuck = stuck_processes()
            if stuck:
                what = ", ".join(f"{p} ({c})" for p, c in stuck[:4])
                return (
                    f"processes the GPU driver lists stay in uninterruptible "
                    f"sleep after a {verdict} ({what})"
                )
            if not controls:
                return (
                    f"the pool has no kernel outside the hazard classes to check "
                    f"it with after a {verdict}"
                )
            ok = control(d, f"after_{sid}")
            if ok is None:
                return f"no control kernel ran a canary problem after a {verdict}"
            return None if ok else f"no control kernel passed after a {verdict}"

        def wait_foreign(d: int) -> float:
            """Seconds waited for other processes to leave device `d`'s GPU
            before a kernel runs on it (the correctness gate's wait)."""
            gpu = kfd.physical_gpu(d)
            return 0.0 if gpu is None else cg._wait_foreign(gpu, stop)

        def unusable(d: int, why: str) -> None:
            with lock:
                if summary["stopped"] is None:
                    summary["stopped"] = f"device {d}: {why}"
                summary["rc"] = 3
            stop.set()

        def worker(d: int) -> None:
            try:
                ok = control(d, "before") if controls else True
                if ok is None and not stop.is_set():
                    raise CanaryError(
                        f"no control kernel ran a canary problem on device {d} "
                        f"(their logs are in {logs}): the canary's bench lines "
                        f"do not fit the library, so no kernel is blamed"
                    )
                if not ok:
                    if not stop.is_set():
                        unusable(d, "no control kernel passed before the first run")
                    return
                while not stop.is_set():
                    with lock:
                        if not queue:
                            return
                        sid = queue.popleft()
                    k = kernels[sid]
                    result: Optional[RunResult] = None
                    for run in range(1, st.runs + 1):
                        token = f"{os.getpid()}-{time.time_ns()}-{sid}-{run}"
                        waited = wait_foreign(d)
                        record(sid, token, run, d)
                        try:
                            result = run_kernel(
                                canary_yaml(k, cfg, run),
                                sid,
                                d,
                                spec.bench_binary,
                                env,
                                logs / f"{sid}_r{run}.log",
                                st.timeout_s,
                                spec.startup_grace_s,
                                children,
                                stop.is_set,
                            )
                        except BaseException:
                            record(
                                sid,
                                token,
                                run,
                                d,
                                verdict="interrupted",
                                reason="the stage stopped before the run ended",
                            )
                            raise
                        result.foreign_wait_s = waited
                        wrong = [n for n in result.names if n != _name(k)]
                        if wrong:
                            record(
                                sid,
                                token,
                                run,
                                d,
                                verdict="error",
                                reason="the bench ran another kernel",
                            )
                            raise CanaryError(
                                f"solution {sid}: the bench ran "
                                f"{ki.label_of_name(wrong[0])}, the library holds "
                                f"{ki.kernel_id(k)} there: the bench does not load "
                                f"the library the pipeline reads"
                            )
                        extra = {}
                        if result.verdict == "hang":
                            extra["timeout_s"] = st.timeout_s
                        record(
                            sid,
                            token,
                            run,
                            d,
                            verdict=result.verdict,
                            reason=result.reason,
                            evidence=result.evidence(),
                            **extra,
                        )
                        if result.verdict != "pass":
                            break
                    assert result is not None
                    with lock:
                        ran[sid] = result
                    if result.verdict in QUARANTINE_VERDICTS:
                        warn(
                            f"device {d}: {ki.label_of_name(_name(k))} "
                            f"({class_text(classes[sid])}) is quarantined: "
                            f"{result.verdict}, {result.reason}"
                        )
                    if not result.reaped:
                        unusable(d, f"a killed bench did not end ({result.reason})")
                        return
                    why = (
                        after(d, sid, result.verdict)
                        if result.verdict in ("hang", "fault")
                        else None
                    )
                    if why is not None:
                        if not stop.is_set():
                            unusable(d, why)
                        return
                    with lock:
                        n = len(ran)
                    if n % 10 == 0 or n == len(pending):
                        say(f"{n}/{len(pending)} hazard-class kernels ran alone")
            except BaseException as e:  # noqa: BLE001
                with lock:
                    errors.append(e)
                stop.set()

        threads = [
            threading.Thread(target=worker, args=(d,), daemon=True)
            for d in spec.devices
        ]
        with gpu_lease.hold(
            spec.devices,
            "hazard canary",
            out_dir,
            wait_s=spec.lease_wait_s,
            report=warn,
        ):
            for t in threads:
                t.start()
            try:
                for t in threads:
                    t.join()
            except BaseException:
                stop.set()
                children.stop_all()
                for t in threads:
                    t.join(timeout=REAP_S)
                raise
        summary["checked"] = len(ran)
        if errors:
            first = errors[0]
            if isinstance(first, CanaryError):
                raise first
            raise CanaryError(f"the canary could not run: {first!r}")
    for p in problems:
        warn(p)
    store.refresh()
    held = store.quarantined(arch)
    for sid in elsewhere:
        rec = store.passed(ki.kernel_id(kernels[sid]), device, code[sid], proto)
        if rec is not None:
            found[sid] = rec
    rows = []
    blocked = []
    kept_out = []
    shown = sorted(set(hazard) | {s for s, k in kernels.items() if _name(k) in held})
    for sid in shown:
        k = kernels[sid]
        base = kernel_record(k, classes[sid], solution_index=sid)
        q = held.get(_name(k))
        if q is not None:
            source = "canary" if sid in ran else "cache"
            row = dict(
                base,
                verdict="quarantined",
                source=source,
                reason=quarantine_reason(q),
                evidence=q.get("evidence"),
            )
            blocked.append(
                cx.Entry(
                    name=_name(k),
                    reason=quarantine_reason(q),
                    kernel_id=ki.kernel_id(k),
                    signature=ki.signature_label(k),
                    source=SOURCE,
                )
            )
        elif sid in ran and ran[sid].verdict == "pass":
            row = dict(base, verdict="pass", source="canary", reason="")
        elif sid in found:
            row = dict(base, verdict="pass", source="cache", reason="")
        else:
            row = dict(base, verdict="unchecked", source=None, reason="")
            if not spec.dry_run:
                # A kernel that has not run alone does not run among others:
                # it stays out of this invocation, and the next runs it.
                if sid in ran:
                    why = ran[sid].reason
                elif sid in elsewhere:
                    why = "another invocation was running it alone"
                else:
                    why = "it did not run"
                row["reason"] = (
                    f"hazard canary: unchecked ({class_text(classes[sid])}): {why}; "
                    f"kept out until a run alone passes"
                )
                kept_out.append(
                    cx.Entry(
                        name=_name(k),
                        reason=row["reason"],
                        kernel_id=ki.kernel_id(k),
                        signature=ki.signature_label(k),
                        source=SOURCE,
                    )
                )
        rows.append(row)
    counts = collections.Counter(r["verdict"] for r in rows)
    summary["verdicts"] = dict(sorted(counts.items()))
    summary["quarantined"] = len(blocked)
    summary["kept_out"] = len(kept_out)
    summary["seconds"] = round(time.time() - t0, 1)
    head = {"library_stem": spec.library_stem, "arch": arch, "device": device}
    _write(out_dir / VERDICTS_FILE, dict(head, kernels=rows))
    cx.write_blocklist(
        out_dir / QUARANTINE_FILE, cx.Blocklist(blocked + kept_out), **head
    )
    _write(out_dir / SUMMARY_FILE, summary)
    return summary
