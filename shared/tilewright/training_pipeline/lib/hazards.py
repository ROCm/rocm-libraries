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

The hang screen (`hazards.screen`, off unless enabled) looks for kernels of
no hazard class that hang the device at some problems and not at others,
after the canary: on each problem of `screen_shapes`, one bench process
lists every kernel of the pool outside the hazard classes that has not
cleared the problem, each timed once (`screen_line`), so that a row prints
as each one ends. A process that prints nothing for `stall_s`, reports a
fatal GPU error or exits otherwise than 0 stopped at the first kernel after
its last row that the bench runs for the problem: once the device has passed
the control kernel, the kernels after that row run alone on the problem, in
list order, until the bench runs one of them. It is quarantined when it
hangs or faults alone too, as a canary run that does so quarantines its
kernel, and the problem's process resumes after it; one that runs alone
clears the problem, since the screen judges a kernel by what it does alone.
A run alone gets the canary's `timeout_s` when that is longer than
`stall_s`, and blames its kernel only on a problem that a kernel of the
screen has run to its row on (`blame`): before then a kernel that stops
alone is held back until one does, and a second one leaves the problem with
neither blamed, since the bench may run no kernel on it; a held-back run
that never ends quarantines nothing. A kernel of a hazard class never joins
the screen's processes: one that stopped its device for good there could
not be run alone to name it.

The store records each kernel that clears a problem as it does (it ran on
the problem, or the bench does not run it there; a SCREENED record under
the problem's own protocol) and each kernel a process stopped at (STOPPED),
so that a later screen of the problem on the device lists only the kernels
that have not cleared it, and runs a kernel an earlier process stopped at
alone after the others: one that stops its device for good there then does
so in a run that names it. A kernel that has cleared every problem passes
the screen, which holds for its kernel id, device, code digest and the
screen's protocol (`screen_protocol`).
"""
from __future__ import annotations

import collections
import fcntl
import hashlib
import json
import math
import os
import random
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
from .bench_yaml import (
    ADAPTIVE_KEYS,
    canary_line,
    drop_line_knob,
    format_solution_list,
    name_entry,
    render_bench_line,
    set_line_knob,
)
from .dat import mx_block_size_for_scale_mode
from .shapes import Shape

HAZARDS_DIR = "hazards"
CACHE_FILE = "canary.jsonl"
QUARANTINE_FILE = "quarantine.json"
VERDICTS_FILE = "verdicts.json"
SUMMARY_FILE = "canary_summary.json"
DEFAULTS: Dict[str, Any] = {"enabled": True, "timeout_s": 300.0, "runs": 1}
SCREEN_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "shapes": [],
    "per_cell": 1,
    "seed": 0,
    "stall_s": 120.0,
}
# The bench settings of a line of the hang screen (`screen_line`): one timed
# call per kernel, no flush and no rotating buffers.
SCREEN_KNOBS: Dict[str, Any] = {
    "iters": 1,
    "cold_iters": 1,
    "flush": "false",
    "rotating": 0,
}
# The events of the hang screen's records besides a run's start and end:
# kernels that cleared a problem, and the kernel a process stopped at.
SCREENED = "screened"
STOPPED = "stopped"
# The `source` of the canary's blocklist entries (lib/correctness.py).
SOURCE = "hazard canary"
# How the reason of an entry starts that keeps a kernel out unchecked
# (`unchecked_reason`, `kept_out_unchecked`).
UNCHECKED = f"{SOURCE}: unchecked"
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


def unchecked_reason(classes: Iterable[str], why: str) -> str:
    """The reason of the blocklist entry that keeps a kernel of the hazard
    classes `classes` out of an invocation unchecked: its run alone was cut
    short or did not happen (`why`)."""
    return (
        f"{UNCHECKED} ({class_text(classes)}): {why}; "
        f"kept out until a run alone passes"
    )


def kept_out_unchecked(entry: Any) -> bool:
    """Whether a blocklist entry (lib/correctness.py `Entry`) is the
    canary's keeping a kernel out of one invocation unchecked
    (`unchecked_reason`) rather than its quarantine: nothing was found
    against the kernel, and the next invocation runs it alone again."""
    return entry.source == SOURCE and entry.reason.startswith(f"{UNCHECKED} (")


# ── settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenSettings:
    """The config's `hazards.screen` section over SCREEN_DEFAULTS; `shapes`
    as (M, N, K, batch), batch 1 where an entry gives three values."""

    enabled: bool = SCREEN_DEFAULTS["enabled"]
    shapes: Tuple[Tuple[int, int, int, int], ...] = ()
    per_cell: int = SCREEN_DEFAULTS["per_cell"]
    seed: int = SCREEN_DEFAULTS["seed"]
    stall_s: float = SCREEN_DEFAULTS["stall_s"]

    @classmethod
    def of(cls, sec: Optional[Mapping[str, Any]]) -> "ScreenSettings":
        sec = dict(sec or {})
        given = {k: sec[k] for k in ("per_cell", "seed") if sec.get(k) is not None}
        return cls(
            enabled=bool(sec.get("enabled", SCREEN_DEFAULTS["enabled"])),
            shapes=tuple(
                (int(s[0]), int(s[1]), int(s[2]), int(s[3]) if len(s) > 3 else 1)
                for s in sec.get("shapes") or ()
            ),
            per_cell=int(given.get("per_cell", SCREEN_DEFAULTS["per_cell"])),
            seed=int(given.get("seed", SCREEN_DEFAULTS["seed"])),
            stall_s=float(sec.get("stall_s") or SCREEN_DEFAULTS["stall_s"]),
        )


@dataclass(frozen=True)
class Settings:
    """The config's `hazards` section over DEFAULTS."""

    enabled: bool = DEFAULTS["enabled"]
    timeout_s: float = DEFAULTS["timeout_s"]
    runs: int = DEFAULTS["runs"]
    devices: Optional[Tuple[int, ...]] = None
    cache_dir: Optional[str] = None
    screen: ScreenSettings = ScreenSettings()

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
            screen=ScreenSettings.of(sec.get("screen")),
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


@dataclass
class ScreenProgress:
    """What the store holds of the hang screen's problems on one device
    (`CanaryStore.screen_progress`), by problem: the kernels (kernel id, code
    digest) that cleared it, whether a kernel ran to its row on it, and the
    kernels a process stopped at there."""

    cleared: Dict[str, Set[Tuple[str, str]]] = field(
        default_factory=lambda: collections.defaultdict(set)
    )
    ran: Set[str] = field(default_factory=set)
    stopped: Dict[str, Set[Tuple[str, str]]] = field(
        default_factory=lambda: collections.defaultdict(set)
    )


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
                    if not isinstance(rec, dict) or not rec.get("token"):
                        continue
                    # A SCREENED record names many kernels, every other one
                    # its kernel.
                    if rec.get("name") or rec.get("event") == SCREENED:
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
        other than the runs of `running` (tokens of runs still going) and the
        runs that may not blame their kernel (`blame` false: a run alone of
        the hang screen held back)."""
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
            if (
                str(rec["token"]) in live
                or owner_alive(rec.get("owner"))
                or rec.get("blame") is False
            ):
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

    def passes(self) -> Set[Tuple[Any, Any, Any, Any]]:
        """(kernel id, device, code digest, protocol) of every pass, for
        `passed` over many kernels at once."""
        return {
            (r.get("kernel_id"), r.get("device"), r.get("code"), r.get("protocol"))
            for r in self.records()
            if r.get("event") == "end" and r.get("verdict") == "pass"
        }

    def screen_progress(
        self, device: Optional[str], protocols: Mapping[str, str]
    ) -> ScreenProgress:
        """The hang screen's SCREENED and STOPPED records of the problems of
        `protocols` ({problem: its protocol, `screen_protocol` of it alone})
        on `device`."""
        out = ScreenProgress()
        for rec in self.records():
            at = rec.get("screen")
            key = at.get("shape") if isinstance(at, Mapping) else None
            if (
                key not in protocols
                or rec.get("protocol") != protocols[key]
                or rec.get("device") != device
            ):
                continue
            if rec.get("event") == SCREENED:
                kernels = rec.get("kernels")
                for c, kids in kernels.items() if isinstance(kernels, Mapping) else ():
                    out.cleared[key].update((str(kid), str(c)) for kid in kids)
                if at.get("ran"):
                    out.ran.add(key)
            elif rec.get("event") == STOPPED:
                out.stopped[key].add((rec.get("kernel_id"), rec.get("code")))
        return out


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


def shape_key(shape: Shape) -> str:
    return "x".join(str(v) for v in shape.as_tuple())


def _cell_seed(seed: int, cell: str) -> int:
    """The seed of the screen's draw in `cell`, the same in every process
    and Python version."""
    digest = hashlib.sha256(f"hang screen\x1f{int(seed)}\x1f{cell}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def screen_shapes(cfg: Mapping[str, Any], st: ScreenSettings) -> List[Shape]:
    """The problems of the hang screen, each once: `st.shapes` in their
    order, then `st.per_cell` shapes of every cell of the grid
    (lib/grid.py), drawn from `st.seed` as stage01 draws a cell's
    (lib/shapes.py `synthesize_cell_shapes` under the config's
    `ShapeFilter`). ValueError for a config whose device memory is not a
    size."""
    from .grid import all_cell_labels
    from .shapes import ShapeFilter, device_memory_budget, synthesize_cell_shapes

    out = [Shape(*s) for s in st.shapes]
    if st.per_cell > 0:
        filt = ShapeFilter.from_config(cfg, budget=device_memory_budget(cfg))
        for cell in all_cell_labels():
            rng = random.Random(_cell_seed(st.seed, cell))
            drawn = synthesize_cell_shapes(cell, st.per_cell, rng, filt)
            out += [Shape(*s) for s in drawn]
    return list(dict.fromkeys(out))


def screen_protocol(host: str, shapes: Iterable[Shape]) -> str:
    """Digest of what a pass of the hang screen means: its problems, the
    bench settings of its lines (SCREEN_KNOBS) and the hipBLASLt host
    libraries. Not `stall_s`: a kernel that ran on every problem passed
    whatever the patience it ran under."""
    body = {
        "version": PROTOCOL_VERSION,
        "screen": sorted(list(s.as_tuple()) for s in shapes),
        "knobs": SCREEN_KNOBS,
        "host": host,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def screen_line(
    shape: Shape, cfg: Mapping[str, Any], sids: Sequence[int], name: str
) -> str:
    """The bench line of a hang screen process on `shape`, named `name`:
    the solution indices `sids` in list order, which the bench runs in that
    order, each timed once (SCREEN_KNOBS) with no skip ratio."""
    knobs = {k: SCREEN_KNOBS[k] for k in ("flush", "rotating")}
    line = render_bench_line(shape, dict(cfg), knobs)
    for key in ("adaptive", *ADAPTIVE_KEYS):
        line = drop_line_knob(line, key)
    line = set_line_knob(line, "skip_slow_solution_ratio", 0)
    for key in ("iters", "cold_iters"):
        line = set_line_knob(line, key, SCREEN_KNOBS[key])
    line = set_line_knob(line, "solution_index_list", format_solution_list(sids))
    return name_entry(line, name)


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


@dataclass
class BenchEnd:
    """How one bench process ended: its exit status `rc` and the stall
    watchdog's `reason` (lib/procs.py); `reaped` false for a process that
    did not end after it was killed, `stuck` for one left running stuck in
    the GPU driver."""

    rc: int
    reason: Optional[str]
    reaped: bool
    stuck: bool
    wall_s: float


def bench_process(
    lines: Sequence[str],
    device: int,
    bench_binary: Path,
    env: Mapping[str, str],
    log: Path,
    timeout_s: float,
    startup_grace_s: float,
    children: procs.ChildGroups,
) -> BenchEnd:
    """Bench `lines` in one hipblaslt-bench process on device `device` of
    `env`, which sees that GPU alone, its output in `log`, under the stall
    watchdog: killed once it prints nothing for `timeout_s` (before it has
    loaded its library, not before `startup_grace_s`: a kernel that hangs
    does so after the load) or prints a fatal GPU error. A process that
    failed after printing a fatal GPU error ends as one killed for it, even
    when it exited before the watchdog read the error."""
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
    if reason is None and rc != 0 and probe.fatal_seen():
        reason = "fatal"
    stuck = rc == procs.STUCK_RC
    reaped = proc.poll() is not None
    if not reaped and not stuck:
        try:
            proc.wait(timeout=REAP_S)
            reaped = True
        except subprocess.TimeoutExpired:
            reaped = False
    return BenchEnd(rc, reason, reaped, stuck, time.time() - t0)


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
    """Bench `lines` (one kernel's) in one process (`bench_process`) and
    judge the run of solution `sid`."""
    end = bench_process(
        lines, device, bench_binary, env, log, timeout_s, startup_grace_s, children
    )
    result = judge(sid, log, end.rc, end.reason, timeout_s, stopping())
    result.wall_s = end.wall_s
    result.reaped = end.reaped
    if end.stuck:
        result.reason += (
            "; the process stayed stuck in the GPU driver and is left running, "
            "not killed"
        )
    elif not end.reaped:
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
    screen = rec.get("screen")
    if isinstance(screen, Mapping) and screen.get("shape"):
        what = f"hang screen at {screen['shape']}"
    else:
        what = class_text(rec.get("classes") or ())
    return f"hazard canary: {verdict} ({what})" + (f": {why}" if why else "")


# ── one canary pass over a pool ──────────────────────────────────────────────


class CanaryError(RuntimeError):
    """The canary cannot run: no kernel is blamed."""


def other_kernel(sid: int, kernel: Mapping[str, Any], ran: str) -> CanaryError:
    """The error of a bench that ran the kernel named `ran` as solution
    `sid`, where the pool holds `kernel`."""
    return CanaryError(
        f"solution {sid}: the bench ran {ki.label_of_name(ran)}, the library "
        f"holds {ki.kernel_id(kernel)} there: the bench does not load the "
        f"library the pipeline reads"
    )


def _ended(end: BenchEnd) -> str:
    """How a bench process that did not end cleanly ended, for a message."""
    if end.reason == "stall":
        return "hung"
    if end.reason == "fatal":
        return "reported a fatal GPU error"
    sig = procs.killed_by(end.rc)
    if sig is not None:
        return f"was killed by {sig.name}"
    return f"ended with rc={end.rc}"


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
    bench lines do not fit the library.

    With `hazards.screen.enabled` the hang screen (see the module
    docstring) runs after the canary has run on every device, on the same
    devices under the same leases, and stops the same way: a killed bench
    that did not end, a process stuck in the GPU driver, or a control kernel
    that does not pass before its first process or after one stopped. Its
    quarantine joins the canary's, and a kernel whose run alone was cut
    short from outside stays out of the invocation. The kernels that have
    cleared every problem, in this or earlier invocations, pass, also when
    the screen stops. CanaryError also when a process of the screen ends
    before it loaded the library, or runs a kernel its line does not list."""
    from . import bench_env as benv
    from . import correctness as cx
    from . import correctness_gate as cg
    from . import gpu_lease
    from .bank_pool import bank_device, config_identity, config_pool
    from .planner import MAX_BENCH_LIST

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
    try:
        held = store.quarantined(arch)
        busy = store.in_flight(arch)
    except (OSError, ValueError) as e:
        raise CanaryError(
            f"cannot read the canary's store {spec.cache_dir}: {e}"
        ) from None
    screen = st.screen
    shapes: List[Shape] = []
    candidates: List[int] = []
    if screen.enabled:
        try:
            shapes = screen_shapes(cfg, screen)
        except ValueError as e:
            raise CanaryError(f"cannot draw the hang screen's problems: {e}") from None
        if not shapes:
            warn("the hang screen has no problem to run its kernels on")
        candidates = [
            s
            for s in sorted(kernels)
            if shapes
            and not classes[s]
            and not listed.blocks(kernels[s])
            and _name(kernels[s]) not in held
        ]
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
        "screen": None,
        "stopped": None,
        "dry_run": bool(spec.dry_run),
        "rc": 0,
    }
    found: Dict[int, Dict[str, Any]] = {}
    pending: List[int] = []
    elsewhere: List[int] = []
    screen_todo: List[int] = []
    device = proto = screen_proto = None
    code: Dict[int, str] = {}
    problem_proto: Dict[str, str] = {}
    progress = ScreenProgress()
    cleared_on: Dict[int, Set[str]] = collections.defaultdict(set)
    if hazard or candidates:
        try:
            identity = config_identity(
                cfg, spec.dat_dir, spec.library_stem, spec.hardware_device
            )
            owner: Dict[int, str] = {}
            for stem in identity.stems:
                for sid in identity.file_ids.get(stem, {}):
                    owner.setdefault(int(sid), stem)
            stems = sorted({owner[s] for s in hazard + candidates})
            digests = cg.code_digests(spec.dat_dir, stems)
            device = cg.device_key(bank_device(cfg, spec.hardware_device))
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            raise CanaryError(f"cannot identify the pool's kernels: {e}") from None
        code = {s: digests[owner[s]] for s in hazard + candidates}
        host = cg.host_digest(spec.build_dir)
        proto = protocol(st.runs, host)
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
        if candidates:
            screen_proto = screen_protocol(host, shapes)
            passes = store.passes()
            screen_todo = [
                s
                for s in candidates
                if (ki.kernel_id(kernels[s]), device, code[s], screen_proto)
                not in passes
            ]
            problem_proto = {shape_key(s): screen_protocol(host, [s]) for s in shapes}
            progress = store.screen_progress(device, problem_proto)
            for s in screen_todo:
                ident = (ki.kernel_id(kernels[s]), code[s])
                cleared_on[s] = {
                    p for p, done in progress.cleared.items() if ident in done
                }
    screened_before = len(candidates) - len(screen_todo)
    unscreened = [s for s in screen_todo if not cleared_on[s] >= set(problem_proto)]
    screen_problems = [
        (si, shape)
        for si, shape in enumerate(shapes)
        if any(shape_key(shape) not in cleared_on[s] for s in unscreened)
    ]
    swept_before = len(shapes) - len(screen_problems)
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
    if screen.enabled:
        say(
            f"hang screen: {len(shapes):,} problems, {len(candidates):,} kernels "
            f"outside the hazard classes ({len(candidates) - len(screen_todo):,} "
            f"passed before, {len(unscreened):,} to screen on "
            f"{len(screen_problems):,} problems)"
        )
    stop = threading.Event()
    lock = threading.Lock()
    ran: Dict[int, RunResult] = {}
    problems: List[str] = []
    errors: List[BaseException] = []
    screen_hung: Set[int] = set()
    screen_cut: Dict[int, str] = {}
    stalls: List[Dict[str, Any]] = []
    suspected: List[Dict[str, Any]] = []
    swept: List[int] = []
    left: List[str] = []
    screen_passed: List[int] = []
    screened = 0
    owner = process_owner()
    # A run alone gets the canary's patience when it is longer, so that a
    # kernel only slower than `stall_s` is not blamed for a hang.
    patience = max(screen.stall_s, st.timeout_s)
    if (pending or screen_problems) and spec.dry_run:
        if pending:
            warn(
                f"dry run: {len(pending):,} hazard-class kernels left without a "
                f"verdict"
            )
        if screen_problems:
            warn(f"dry run: {len(unscreened):,} kernels left unscreened")
    elif pending or screen_problems:
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
            screen_problems = []
    if (pending or screen_problems) and not spec.dry_run:
        gpus = kfd.bench_gpus(spec.devices)
        if not gpus:
            raise CanaryError(f"devices {list(spec.devices)} reach no GPU here")
        summary["devices"] = list(spec.devices)
        env = benv.bench_env_with(benv.runtime_env(cfg))
        controls = [
            s
            for s in sorted(kernels)
            if not classes[s]
            and not listed.blocks(kernels[s])
            and _name(kernels[s]) not in held
        ][:3]
        queue: "collections.deque[int]" = collections.deque(pending)
        logs = out_dir / "logs"

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

        def after(d: int, sid: Any, verdict: str) -> Optional[str]:
            """Why device `d` takes no more kernels after the hang or fault
            of `sid`'s run (a kernel's, or a screen process's tag): a
            process stuck in the GPU driver, no kernel outside the hazard
            classes to check it with, or no control kernel that passes; None
            when one passes."""
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

        problem_queue: "collections.deque[Tuple[int, Shape]]" = collections.deque(
            screen_problems
        )

        def screen_record(
            sid: int, event: str, token: str, gpu: int, key: str, **more: Any
        ) -> None:
            base = kernel_record(
                kernels[sid],
                classes[sid],
                event=event,
                token=token,
                arch=arch,
                device=device,
                gpu=gpu,
                code=code[sid],
                protocol=problem_proto[key],
                screen={"shape": key},
                time=kfd.utc_iso(),
                owner=owner,
            )
            store.add([dict(base, **more)])

        def clear(key: str, sids: Sequence[int], ran: bool) -> None:
            """Record that the kernels `sids` cleared the problem `key`: they
            ran on it, or the bench does not run them for it; `ran` when one
            of them ran to its row."""
            if not sids:
                return
            by_code: Dict[str, List[str]] = {}
            for s in sids:
                by_code.setdefault(code[s], []).append(ki.kernel_id(kernels[s]))
            store.add(
                [
                    {
                        "event": SCREENED,
                        "token": f"{os.getpid()}-{time.time_ns()}-{key}",
                        "arch": arch,
                        "device": device,
                        "protocol": problem_proto[key],
                        "screen": {"shape": key, "ran": bool(ran)},
                        "kernels": by_code,
                        "time": kfd.utc_iso(),
                        "owner": owner,
                    }
                ]
            )
            with lock:
                for s in sids:
                    cleared_on[s].add(key)

        def named(sid: int) -> Dict[str, Any]:
            return {
                "solution": sid,
                "kernel": ki.kernel_id(kernels[sid]),
                "name": _name(kernels[sid]),
            }

        def alone(d: int, sid: int, si: int, shape: Shape, blame: bool) -> RunResult:
            """Run kernel `sid` alone on `shape` on device `d`, recorded as a
            canary run is: its start before, its end after, each with
            `blame`. A kernel that runs alone without hanging or faulting, or
            that the bench does not run for the problem, is `cleared`; one
            that hangs or faults is quarantined with `blame`, else
            `unconfirmed`."""
            k = kernels[sid]
            key = shape_key(shape)
            name = f"screen{si}_{sid}"
            token = f"{os.getpid()}-{time.time_ns()}-{sid}-{name}"
            waited = wait_foreign(d)
            screen_record(sid, "start", token, d, key, blame=blame)
            try:
                result = run_kernel(
                    [screen_line(shape, cfg, [sid], name)],
                    sid,
                    d,
                    spec.bench_binary,
                    env,
                    logs / f"{name}.log",
                    patience,
                    spec.startup_grace_s,
                    children,
                    stop.is_set,
                )
            except BaseException:
                screen_record(
                    sid,
                    "end",
                    token,
                    d,
                    key,
                    blame=blame,
                    verdict="interrupted",
                    reason="the stage stopped before the run ended",
                )
                raise
            result.foreign_wait_s = waited
            wrong = [n for n in result.names if n != _name(k)]
            if wrong:
                screen_record(
                    sid,
                    "end",
                    token,
                    d,
                    key,
                    blame=blame,
                    verdict="error",
                    reason="the bench ran another kernel",
                )
                raise other_kernel(sid, k, wrong[0])
            verdict, reason = result.verdict, result.reason
            if verdict == "pass":
                verdict, reason = "cleared", "it ran alone"
            elif verdict == "untested":
                verdict, reason = "cleared", "the bench does not run it for the problem"
            elif verdict in ("hang", "fault") and not blame:
                verdict, reason = "unconfirmed", (
                    f"{verdict} before a kernel of the screen ran to its row on "
                    f"the problem: {reason}"
                )
            extra = {"timeout_s": patience} if result.verdict == "hang" else {}
            screen_record(
                sid,
                "end",
                token,
                d,
                key,
                blame=blame,
                verdict=verdict,
                reason=reason,
                evidence=result.evidence(),
                **extra,
            )
            return result

        def condemn(d: int, sid: int, key: str, r: RunResult) -> None:
            with lock:
                screen_hung.add(sid)
            warn(
                f"device {d}: {ki.label_of_name(_name(kernels[sid]))} is "
                f"quarantined: {r.verdict} alone on {key}, {r.reason}"
            )

        def sweep(d: int, si: int, shape: Shape) -> Optional[str]:
            """Run the hang screen's kernels that have not cleared `shape` on
            it on device `d`, many to a process, then alone those an earlier
            process stopped at there (see the module docstring); why `d` takes
            no more work, None when it does."""
            key = shape_key(shape)
            with lock:
                todo = [
                    s
                    for s in screen_todo
                    if s not in screen_hung and key not in cleared_on[s]
                ]
            stopped_at = progress.stopped.get(key, ())
            later = [
                s for s in todo if (ki.kernel_id(kernels[s]), code[s]) in stopped_at
            ]
            todo = [s for s in todo if s not in later]
            part = 0
            # Until a kernel has run to its row on the problem, one that
            # stops alone there cannot be told from a problem the bench runs
            # no kernel on: it is held back, and blamed once one has.
            runs = key in progress.ran
            held: Optional[Tuple[int, RunResult]] = None

            def settle(
                sid: int, out: List[Dict[str, Any]]
            ) -> Tuple[str, Optional[str]]:
                """Run `sid` alone on the problem and act on its verdict,
                which joins `out`: ("next", None) when the bench does not run
                it, ("done", None) once it is settled, ("end", why) when the
                sweep ends there (why `d` takes no more work, None when it
                does)."""
                nonlocal runs, held
                r = alone(d, sid, si, shape, runs)
                out.append({"kernel": ki.kernel_id(kernels[sid]), "verdict": r.verdict})
                if r.verdict == "untested":
                    clear(key, [sid], False)
                    return "next", None
                if r.verdict == "pass":
                    clear(key, [sid], True)
                    runs = True
                elif r.verdict in ("hang", "fault"):
                    if runs:
                        condemn(d, sid, key, r)
                    if not r.reaped:
                        return "end", f"a killed bench did not end ({r.reason})"
                    why = after(d, sid, r.verdict)
                    if why is not None:
                        return "end", why
                    if not runs and held is not None:
                        warn(
                            f"device {d}: "
                            f"{ki.label_of_name(_name(kernels[held[0]]))} and "
                            f"{ki.label_of_name(_name(kernels[sid]))} stopped "
                            f"alone on {key}, which no kernel of the hang screen "
                            f"has run to its row on: neither is blamed, and the "
                            f"screen leaves the problem"
                        )
                        with lock:
                            left.append(key)
                        return "end", None
                    if not runs:
                        held = (sid, r)
                elif stop.is_set():
                    return "end", None
                else:
                    with lock:
                        screen_cut[sid] = (
                            f"{UNCHECKED} (hang screen at {key}): its run alone "
                            f"was cut short ({r.reason}); kept out until the "
                            f"screen runs it again"
                        )
                return "done", None

            while todo and not stop.is_set():
                batch = todo[:MAX_BENCH_LIST]
                tag = f"screen{si}_p{part}"
                part += 1
                log = logs / f"{tag}.log"
                wait_foreign(d)
                end = bench_process(
                    [screen_line(shape, cfg, batch, tag)],
                    d,
                    spec.bench_binary,
                    env,
                    log,
                    screen.stall_s,
                    spec.startup_grace_s,
                    children,
                )
                if stop.is_set():
                    return None
                parsed = parse_log_file(log) if log.is_file() else []
                place = {s: i for i, s in enumerate(batch)}
                rows = [t for p in parsed for t in p.tested if t.sol_index is not None]
                for t in rows:
                    if t.sol_index not in place:
                        raise CanaryError(
                            f"the hang screen's bench on {key} ran solution "
                            f"{t.sol_index}, which its line does not list (its log "
                            f"is {log})"
                        )
                    if t.sol_name and t.sol_name != _name(kernels[t.sol_index]):
                        raise other_kernel(
                            t.sol_index, kernels[t.sol_index], t.sol_name
                        )
                runs = runs or bool(rows)
                if end.rc == 0 and end.reason is None:
                    clear(key, batch, bool(rows))
                    todo = todo[len(batch) :]
                    continue
                how = _ended(end)
                loaded = any(p.n_supported is not None or p.has_body for p in parsed)
                last = max((place[t.sol_index] for t in rows), default=-1)
                # A row for every kernel the bench runs for the problem, or no
                # kernel to run there (`NO solution found`, printed instead of
                # the `Is supported` count): what stopped it came after them,
                # and it runs none of the kernels listed after the last row.
                complete = any(
                    p.winner_seen
                    or p.no_solution
                    or (p.n_supported is not None and len(p.tested) >= p.n_supported)
                    for p in parsed
                )
                suspect = (
                    batch[last + 1]
                    if loaded and not complete and last + 1 < len(batch)
                    else None
                )
                stopped: Dict[str, Any] = {
                    "shape": key,
                    "gpu": d,
                    "listed": len(batch),
                    "rows": len(rows),
                    "ended": how,
                    "suspect": None if suspect is None else named(suspect),
                    "log": str(log),
                    "alone": [],
                }
                with lock:
                    stalls.append(stopped)
                clear(key, batch[: last + 1], True)
                at = ""
                if suspect is not None:
                    screen_record(
                        suspect,
                        STOPPED,
                        f"{os.getpid()}-{time.time_ns()}-{suspect}-{tag}",
                        d,
                        key,
                        screen={"shape": key, "rows": len(rows), "log": str(log)},
                    )
                    at = (
                        f"; the first kernel its line lists after its last row, "
                        f"solution {suspect} ({ki.kernel_id(kernels[suspect])}), "
                        f"runs alone after the problem's other kernels next time "
                        f"(its log is {log})"
                    )
                if not end.reaped:
                    stuck = (
                        "; it stayed stuck in the GPU driver and is left running, "
                        "not killed"
                    )
                    return (
                        f"a killed bench of the hang screen on {key} did not end"
                        + (stuck if end.stuck else "")
                        + at
                    )
                if not loaded:
                    raise CanaryError(
                        f"the hang screen's bench on {key} {how} before it had the "
                        f"problem's kernels (its log is {log}): no kernel is blamed"
                    )
                why = after(d, tag, f"screen bench that {how} on {key}")
                if why is not None:
                    return why + at
                if complete:
                    clear(key, batch[last + 1 :], False)
                    todo = todo[len(batch) :]
                    continue
                resume = len(batch)
                for i in range(last + 1, len(batch)):
                    step, why = settle(batch[i], stopped["alone"])
                    if step == "end":
                        return why
                    if step == "done":
                        resume = i + 1
                        break
                todo = todo[resume:]
            # A kernel an earlier process stopped at on the problem runs alone,
            # and after the others: one that stops its device for good there
            # then does so in a run that names it, on a problem other kernels
            # have run to their rows on.
            for sid in later:
                if stop.is_set():
                    return None
                entry: Dict[str, Any] = dict(shape=key, gpu=d, **named(sid), alone=[])
                with lock:
                    suspected.append(entry)
                step, why = settle(sid, entry["alone"])
                if step == "end":
                    return why
            if held is not None and not stop.is_set():
                sid, r = held
                if runs:
                    extra = {"timeout_s": patience} if r.verdict == "hang" else {}
                    screen_record(
                        sid,
                        "end",
                        f"{os.getpid()}-{time.time_ns()}-{sid}-screen{si}",
                        d,
                        key,
                        verdict=r.verdict,
                        reason=f"{r.reason}; other kernels run on the problem",
                        evidence=r.evidence(),
                        **extra,
                    )
                    condemn(d, sid, key, r)
                else:
                    warn(
                        f"device {d}: {ki.label_of_name(_name(kernels[sid]))} "
                        f"stopped alone on {key}, which no kernel of the hang screen "
                        f"ran to its row on: it is not blamed"
                    )
                    with lock:
                        left.append(key)
            return None

        def screen_worker(d: int) -> None:
            try:
                ok = control(d, "before_screen") if controls else True
                if ok is None and not stop.is_set():
                    raise CanaryError(
                        f"no control kernel ran a canary problem on device {d} "
                        f"(their logs are in {logs}): the canary's bench lines "
                        f"do not fit the library, so no kernel is blamed"
                    )
                if not ok:
                    if not stop.is_set():
                        unusable(d, "no control kernel passed before the hang screen")
                    return
                while not stop.is_set():
                    with lock:
                        if not problem_queue:
                            return
                        si, shape = problem_queue.popleft()
                    why = sweep(d, si, shape)
                    if why is not None:
                        if not stop.is_set():
                            unusable(d, why)
                        return
                    with lock:
                        swept.append(si)
                        n = len(swept)
                    if n % 10 == 0 or n == len(screen_problems):
                        say(f"hang screen: {n}/{len(screen_problems)} problems")
            except BaseException as e:  # noqa: BLE001
                with lock:
                    errors.append(e)
                stop.set()

        def on_devices(target: Callable[[int], None]) -> None:
            threads = [
                threading.Thread(target=target, args=(d,), daemon=True)
                for d in spec.devices
            ]
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

        with gpu_lease.hold(
            spec.devices,
            "hazard canary",
            out_dir,
            wait_s=spec.lease_wait_s,
            report=warn,
        ):
            if pending:
                on_devices(worker)
            # A kernel of a hazard class running alone on one GPU can stop
            # the others with it, which would be blamed on a kernel of the
            # screen: the screen starts once the canary has ended everywhere.
            if screen_problems and not stop.is_set():
                screened = len(unscreened)
                on_devices(screen_worker)
        summary["checked"] = len(ran)
        if errors:
            first = errors[0]
            if isinstance(first, CanaryError):
                raise first
            raise CanaryError(f"the canary could not run: {first!r}")
    if screen_todo and not spec.dry_run:
        kept = screen_hung | set(screen_cut)
        screen_passed = [
            s
            for s in screen_todo
            if s not in kept and cleared_on[s] >= set(problem_proto)
        ]
        stamp = kfd.utc_iso()
        store.add(
            [
                kernel_record(
                    kernels[s],
                    classes[s],
                    event="end",
                    token=f"{os.getpid()}-{time.time_ns()}-{s}-screen",
                    arch=arch,
                    device=device,
                    code=code[s],
                    protocol=screen_proto,
                    screen={"problems": len(shapes)},
                    verdict="pass",
                    reason="",
                    time=stamp,
                    owner=owner,
                )
                for s in screen_passed
            ]
        )
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
    shown = sorted(
        set(hazard)
        | {s for s, k in kernels.items() if _name(k) in held}
        | set(screen_cut)
    )
    for sid in shown:
        k = kernels[sid]
        base = kernel_record(k, classes[sid], solution_index=sid)
        q = held.get(_name(k))
        if q is not None:
            source = (
                "canary" if sid in ran else "screen" if sid in screen_hung else "cache"
            )
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
        elif sid in screen_cut:
            row = dict(
                base, verdict="unchecked", source="screen", reason=screen_cut[sid]
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
                row["reason"] = unchecked_reason(classes[sid], why)
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
    if screen.enabled:
        summary["screen"] = {
            "problems": [shape_key(s) for s in shapes],
            "kernels": len(candidates),
            "passed_before": screened_before,
            "swept_before": swept_before,
            "screened": screened,
            "passed": len(screen_passed),
            "quarantined": len(screen_hung),
            "kept_out": len(screen_cut),
            "stops": stalls,
            "suspects": suspected,
            "left": left,
            "protocol": screen_proto,
            "settings": {
                "shapes": len(screen.shapes),
                "per_cell": screen.per_cell,
                "seed": screen.seed,
                "stall_s": screen.stall_s,
            },
        }
    summary["seconds"] = round(time.time() - t0, 1)
    head = {"library_stem": spec.library_stem, "arch": arch, "device": device}
    _write(out_dir / VERDICTS_FILE, dict(head, kernels=rows))
    cx.write_blocklist(
        out_dir / QUARANTINE_FILE, cx.Blocklist(blocked + kept_out), **head
    )
    _write(out_dir / SUMMARY_FILE, summary)
    return summary
