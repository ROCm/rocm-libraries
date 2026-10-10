# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Whether the GPUs a run benches on are still as they were when the run
started, and the run-wide record (`Guard`) of what its stages decided
about them.

A GPU that went through a reset can come back with its power management
off, another performance level or power cap, kernel workers of the ttm
memory manager stuck in uninterruptible sleep, or uncorrectable ECC errors;
its benches then still run and print latencies, which would become labels.
So every stage that benches checks, before and while it benches, per GPU
(read from the sysfs of its PCI device, through the DRM render node its
KFD node names, without root and without amd-smi, which can hang while a
reset is in flight):

  power management   `pp_features` lists an enabled feature (the test
                     amd-smi's POWER_MANAGEMENT makes)
  performance level  `power_dpm_force_performance_level`
  power cap          hwmon `power1_cap`, in W
  uncorrectable ECC  the sum of the `ue:` counts of the files under `ras/`

against the values the run recorded for that GPU (by PCI address) the first
time it checked it (`baseline`), and, unless `Settings.ttm` is off, per
host that no kernel worker of the ttm workqueue is in D state in every one
of `TTM_SAMPLES` readings `TTM_GAP_S` apart (one reading catches workers
that clear within a second while benches run). A value the GPU did not give
at the baseline is not checked; one it gave then and does not give now, or
cannot be read now, fails. Power management off fails whatever the baseline
holds: no GPU benches without it. A failed check is read again after
`CONFIRM_S` and counts only when it fails again.

A failed check raises the run's hold (`Guard.settle`), which keeps its
problems per GPU (`entries`, by PCI address; the host's under HOST). The
departures of single GPUs hold those GPUs alone (`holds`): no stage of the
run benches on them (stage03 aborts their running blocks, which go back on
its work queue, and keeps them out until the run stops holding them),
while the run's other GPUs go on. The host's problems (ttm workers), a
hold recorded without entries, and departures a check finds on every GPU
it reads, two or more (a host-wide condition, such as a reset of every
GPU), hold every GPU of the run (`run_wide`); every stage then stops
benching and waits, saying why. A check settles the entries of the GPUs it
read and the host's, never another GPU's, and reads every GPU the hold
names besides its own (`check`), so that whichever stage of the run checks
next can lift it. The hold lifts when no entry is left; a GPU reads its
canary again before it benches once the run stops holding it
(`canary_due`). A GPU the run leaves out (runtime.exclude_gpus) has its
entry dropped (`Guard.release`).

The canary of a GPU: reference lines (the run's first bench stage chooses
them among its lines, `choose_canary_lines`, and every later canary of the
run uses the same ones) benched in one process of their own, each restamped
to time one kernel alone, a warm-up copy of each first and then `timings`
copies of every line in turn; a line's reading is the median of its copies.
The first canary of a GPU in the run is its reference. Every later one is
judged against it (`judge`): it passes when the median over the lines of
the log of reading / reference is within log(1 + tolerance) of 0, the
tolerance and rule the run recorded with its canary setup before any
canary was judged. A GPU reads its canary before its first bench of every
invocation of the run (`session`) and after every hold of it; one that
fails takes no part until a later canary passes (stage03 leaves it out of
its blocks after bench.health.max_wait_s). A canary that cannot be read
(its bench fails, its output does not line up with its entries, or it
reads fewer than half its lines) is not judged, the first or a later
one: a first one becomes no reference and a later one gives no verdict.
Nothing then judges the GPU, which benches, and its next canary tries
again, so that lines a kernel cannot time never hold a run.

The setup records the kernel each line names by solution index, with its
kernel id and signature when the stage knows the library
(lib/kernel_identity.py); a later stage that knows the library restamps a
line to the index its kernel has now and leaves out a line whose kernel it
no longer holds (`canary_lines_now`), and a setup that can time fewer than
half its lines is chosen anew, every reference dropped. Every canary
records the kernel each line timed (the name the bench printed); a line
that timed another kernel than the GPU's reference is not judged, and a
canary that has fewer than half its lines to judge for that reason becomes
the GPU's reference anew, saying why.

`Guard` keeps all of this in `<guard dir>/guard.json`, read and written
under a file lock by every process of the run, together with the bench
lines the run quarantined because they hung (`quarantine_lines`), each
under the protocol it hung in (`scope`: BENCH_SCOPE for the run's own
benches, another name for a protocol such as the audit's repeated
measurement), so that a line that hung leaves the later stages of that
protocol in the run and no other, and the lines let back in
(`release_lines`, under RELEASED_LINES: `released` holds when the run
stopped holding each GPU). The orchestrator gives its stages the
run's guard directory and the invocation's session in GUARD_DIR_VAR and
SESSION_VAR; a stage run on its own uses one in its output directory and
a session of its own.

`TILEWRIGHT_SYSFS_ROOT` replaces `/sys` (lib/kfd.py) and
`TILEWRIGHT_PROC_ROOT` replaces `/proc` for the ttm workers
(lib/host_load.py).
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import math
import os
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from . import kfd
from .fs import write_json
from .host_load import proc_root

GUARD_DIR_VAR = "TILEWRIGHT_GUARD_DIR"
SESSION_VAR = "TILEWRIGHT_GUARD_SESSION"
GUARD_DIR = "guard"
GUARD_FILE = "guard.json"
LOCK_FILE = ".guard.lock"
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "poll_s": 30.0,
    "max_wait_s": 0.0,
    "ttm": True,
}
CANARY_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "lines": 4,
    "timings": 5,
    "tolerance": 0.05,
    "retry_s": 900.0,
}
CANARY_RULE = (
    "a GPU's canary passes when the median over its lines of ln(reading / "
    "reference) is within ln(1 + tolerance) of 0; a line's reading is the "
    "median of its timings, the reference the GPU's first canary of the run"
)
CONFIRM_S = 2.0
TTM_SAMPLES = 10
TTM_GAP_S = 0.5
REPORT_S = 600.0
KEEP_RECORDS = 200
# The protocol a hung line is quarantined under (`Guard.quarantine_lines`):
# the run's own benches, unless a stage names another (an audit protocol);
# NO_SCOPE quarantines nothing.
BENCH_SCOPE = "bench"
NO_SCOPE = "none"
# The hold entry of the host's problems (the ttm workers), and the one a
# hold recorded without entries reads as: any check may settle either.
HOST = "host"
UNKEYED = "run"
# The record's quarantined lines let back in; `released` maps a GPU (or
# EVERY_GPU) to when the run last stopped holding it.
RELEASED_LINES = "released_lines"
_UE_RE = re.compile(r"^\s*ue\s*:\s*(\d+)\s*$", re.MULTILINE)

Reading = Union[None, bool, int, float, str]


@dataclass(frozen=True)
class CanarySettings:
    enabled: bool = CANARY_DEFAULTS["enabled"]
    lines: int = CANARY_DEFAULTS["lines"]
    timings: int = CANARY_DEFAULTS["timings"]
    tolerance: float = CANARY_DEFAULTS["tolerance"]
    retry_s: float = CANARY_DEFAULTS["retry_s"]


@dataclass(frozen=True)
class Settings:
    """`bench.health` of a run config, over DEFAULTS and CANARY_DEFAULTS.
    The canary is read only while the health checks are on
    (`canary_on`)."""

    enabled: bool = DEFAULTS["enabled"]
    poll_s: float = DEFAULTS["poll_s"]
    max_wait_s: float = DEFAULTS["max_wait_s"]
    ttm: bool = DEFAULTS["ttm"]
    canary: CanarySettings = field(default_factory=CanarySettings)

    @classmethod
    def of(cls, cfg: Mapping[str, Any]) -> "Settings":
        blk = (cfg.get("bench") or {}).get("health") if cfg else None
        blk = blk if isinstance(blk, Mapping) else {}
        can = blk.get("canary")
        can = can if isinstance(can, Mapping) else {}
        return cls(
            enabled=bool(blk.get("enabled", DEFAULTS["enabled"])),
            poll_s=float(blk.get("poll_s", DEFAULTS["poll_s"])),
            max_wait_s=float(blk.get("max_wait_s", DEFAULTS["max_wait_s"])),
            ttm=bool(blk.get("ttm", DEFAULTS["ttm"])),
            canary=CanarySettings(
                enabled=bool(can.get("enabled", CANARY_DEFAULTS["enabled"])),
                lines=int(can.get("lines", CANARY_DEFAULTS["lines"])),
                timings=int(can.get("timings", CANARY_DEFAULTS["timings"])),
                tolerance=float(can.get("tolerance", CANARY_DEFAULTS["tolerance"])),
                retry_s=float(can.get("retry_s", CANARY_DEFAULTS["retry_s"])),
            ),
        )

    @property
    def canary_on(self) -> bool:
        return self.enabled and self.canary.enabled

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def utc() -> str:
    return kfd.utc_iso()


def record_id(rec: Mapping[str, Any]) -> str:
    """What tells a canary setup or reference apart from another one taken
    in the same second (`replaces` names it)."""
    when = rec.get("created") or rec.get("taken")
    return str(rec.get("id") or f"{when} {rec.get('by')}")


# ── readings ─────────────────────────────────────────────────────────────────


def _value(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """(text, None) of a sysfs file; (None, None) when it does not exist;
    (None, why) when it exists and cannot be read."""
    try:
        return path.read_text().strip(), None
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError) as e:
        code = getattr(e, "errno", None)
        name = errno.errorcode.get(code, str(code)) if code else type(e).__name__
        return None, f"unreadable ({name})"


def _device_dir(gpu: kfd.Gpu) -> Optional[Path]:
    if gpu.render_minor is None:
        return None
    path = kfd.sysfs_root() / "class" / "drm" / f"renderD{gpu.render_minor}"
    path = path / "device"
    return path if path.is_dir() else None


def _power_management(device: Path) -> Reading:
    text, why = _value(device / "pp_features")
    if why is not None or text is None:
        return why
    return any(re.search(r"\senabled$", ln.rstrip()) for ln in text.splitlines())


def _power_cap_w(device: Path) -> Reading:
    for h in sorted(device.glob("hwmon/hwmon*")):
        text, why = _value(h / "power1_cap")
        if why is not None:
            return why
        if text is None:
            continue
        try:
            return int(text) / 1e6
        except ValueError:
            return f"unreadable ({text!r})"
    return None


def _ecc_uncorrectable(device: Path) -> Reading:
    ras = device / "ras"
    try:
        files = sorted(p for p in ras.iterdir() if p.is_file())
    except FileNotFoundError:
        return None
    except OSError as e:
        return f"unreadable ({errno.errorcode.get(e.errno, e.errno)})"
    total: Optional[int] = None
    for p in files:
        text, why = _value(p)
        if why is not None:
            return why
        for m in _UE_RE.finditer(text or ""):
            total = (total or 0) + int(m.group(1))
    return total


CHECKS = ("power_management", "perf_level", "power_cap_w", "ecc_uncorrectable")


def read_gpu(gpu: kfd.Gpu) -> Dict[str, Reading]:
    """The CHECKS values of `gpu` now (see the module docstring); None for
    one its sysfs does not give."""
    d = _device_dir(gpu)
    if d is None:
        return {k: None for k in CHECKS}
    level, why = _value(d / "power_dpm_force_performance_level")
    return {
        "power_management": _power_management(d),
        "perf_level": why if why is not None else level,
        "power_cap_w": _power_cap_w(d),
        "ecc_uncorrectable": _ecc_uncorrectable(d),
    }


def ttm_workers_in_d_state(root: Optional[Path] = None) -> List[str]:
    """The names of the kernel workers of the ttm workqueue in D state now."""
    base = proc_root() if root is None else Path(root)
    out: List[str] = []
    try:
        entries = [p for p in base.iterdir() if p.name.isdigit()]
    except OSError:
        return out
    for p in entries:
        try:
            text = (p / "stat").read_text()
        except (OSError, UnicodeDecodeError):
            continue
        i, j = text.find("("), text.rfind(")")
        if i < 0 or j < 0:
            continue
        comm, state = text[i + 1 : j], text[j + 2 : j + 3]
        if state == "D" and comm.startswith("kworker") and "ttm" in comm:
            out.append(comm)
    return sorted(out)


def stuck_ttm_workers(
    samples: int = TTM_SAMPLES,
    gap_s: float = TTM_GAP_S,
    sleep: Callable[[float], None] = time.sleep,
) -> List[str]:
    """The ttm workers in D state in the last of `samples` readings `gap_s`
    apart when every reading finds some; [] as soon as one finds none."""
    last: List[str] = []
    for i in range(int(samples)):
        last = ttm_workers_in_d_state()
        if not last:
            return []
        if i + 1 < samples:
            sleep(gap_s)
    return last


def _is_number(v: Reading) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def changes(
    base: Mapping[str, Reading], now: Mapping[str, Reading]
) -> List[Tuple[str, str]]:
    """How the readings `now` of one GPU depart from its baseline `base`:
    (check, what) pairs, `check` one of CHECKS. Power management off
    departs whatever the baseline holds."""
    out: List[Tuple[str, str]] = []
    pm, pm0 = now.get("power_management"), base.get("power_management")
    if pm is False and pm0 is not True:
        out.append(
            (
                "power_management",
                "power management is off (off too when the run first checked the "
                "GPU): no GPU benches without it",
            )
        )
    elif pm0 is True and pm is not True:
        state = "off" if pm is False else pm
        out.append(
            (
                "power_management",
                f"power management is {state} (on when the run started)",
            )
        )
    level, level0 = now.get("perf_level"), base.get("perf_level")
    if isinstance(level0, str) and not level0.startswith("unreadable"):
        if level != level0:
            out.append(
                (
                    "perf_level",
                    f"performance level {level!r} ({level0!r} when the run started)",
                )
            )
    cap, cap0 = now.get("power_cap_w"), base.get("power_cap_w")
    if _is_number(cap0) and not (_is_number(cap) and abs(cap - cap0) < 0.5):
        shown = f"{cap:g} W" if _is_number(cap) else str(cap)
        out.append(
            ("power_cap_w", f"power cap {shown} ({cap0:g} W when the run started)")
        )
    ue, ue0 = now.get("ecc_uncorrectable"), base.get("ecc_uncorrectable")
    if _is_number(ue0) and ue != ue0:
        out.append(
            (
                "ecc_uncorrectable",
                f"uncorrectable ECC errors {ue} ({ue0} when the run started): the "
                f"GPU may have returned wrong results",
            )
        )
    return out


def gpu_label(gpu: kfd.Gpu) -> str:
    return f"GPU {gpu.index} ({gpu.bdf or 'no PCI address'})"


# ── the run's record ─────────────────────────────────────────────────────────


def guard_dir(fallback: Path) -> Path:
    """The run's guard directory: GUARD_DIR_VAR, else `fallback`."""
    named = os.environ.get(GUARD_DIR_VAR)
    return Path(named) if named else Path(fallback)


def session() -> str:
    """The invocation of the run this process belongs to: SESSION_VAR, else
    one of this process's own."""
    return os.environ.get(SESSION_VAR) or f"pid{os.getpid()}@{utc()}"


def gpu_key(gpu: kfd.Gpu) -> str:
    """The key of a GPU in the run's record: its PCI address, which survives
    a renumbering of the GPUs; its index when the topology gives none."""
    return str(gpu.bdf) if gpu.bdf else f"gpu{gpu.index}"


def keys_of(indices: Iterable[int]) -> Set[str]:
    """The keys of the GPUs of the KFD topology with these physical
    indices."""
    want = {int(i) for i in indices}
    return {gpu_key(g) for g in kfd.gpus() if g.index in want}


def hold_entries(hold: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{key: entry} of a hold, a copy; a hold recorded without entries is
    one entry under UNKEYED."""
    if not isinstance(hold, Mapping):
        return {}
    entries = hold.get("entries")
    if isinstance(entries, Mapping):
        return {str(k): dict(v) for k, v in entries.items() if isinstance(v, Mapping)}
    return {
        UNKEYED: {
            "problems": list(hold.get("problems") or []),
            "checks": [],
            "raised": hold.get("raised"),
            "seen": hold.get("seen"),
            "by": hold.get("by"),
        }
    }


def _problems_of(entries: Mapping[str, Mapping[str, Any]]) -> List[str]:
    order = sorted(k for k in entries if k not in (HOST, UNKEYED))
    order += [k for k in (HOST, UNKEYED) if k in entries]
    return [str(p) for k in order for p in entries[k].get("problems") or []]


def run_wide(hold: Optional[Mapping[str, Any]]) -> bool:
    """Whether `hold` holds every GPU of the run: the host's problems (ttm
    workers), a hold recorded without entries, or departures its last check
    found on every GPU it read, two or more (`host_wide`). A hold of single
    GPUs' departures holds those GPUs alone."""
    if not isinstance(hold, Mapping):
        return False
    entries = hold_entries(hold)
    return HOST in entries or UNKEYED in entries or bool(hold.get("host_wide"))


def holds(hold: Optional[Mapping[str, Any]], key: Optional[str]) -> bool:
    """Whether `hold` holds the GPU of `key` (`gpu_key`): a hold of every
    GPU (`run_wide`) or one with an entry of that GPU. Every hold holds a
    GPU a stage cannot name (`key` None)."""
    if not isinstance(hold, Mapping):
        return False
    return key is None or run_wide(hold) or key in hold_entries(hold)


def held_keys(hold: Optional[Mapping[str, Any]]) -> List[str]:
    """The GPUs a hold of single GPUs holds, by key."""
    return sorted(k for k in hold_entries(hold) if k not in (HOST, UNKEYED))


# The `released` key under which the end of a hold of every GPU is noted.
EVERY_GPU = "*"


def _release_times(doc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """`released` of a record, {GPU key or EVERY_GPU: when}; None when the
    record holds none as a mapping."""
    released = doc.get("released")
    return dict(released) if isinstance(released, Mapping) else None


def released_lines(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The quarantined lines a record let back in (RELEASED_LINES), with
    those a record keeps under `released` as a list."""
    old = doc.get("released")
    lines = list(old) if isinstance(old, list) else []
    lines += list(doc.get(RELEASED_LINES) or [])
    return [dict(e) for e in lines if isinstance(e, Mapping)]


def _upgrade(doc: Dict[str, Any]) -> None:
    """Keep a record's released lines under RELEASED_LINES alone, so that
    `released` holds nothing but the times the run stopped holding GPUs
    (`_note_released`): a `released` that is not a mapping is read as
    released lines (`released_lines`) and moved there."""
    old = doc.get("released")
    if old is None or isinstance(old, Mapping):
        return
    doc[RELEASED_LINES] = released_lines(doc)
    del doc["released"]


def note_accepted(doc: Dict[str, Any], keys: Iterable[str]) -> None:
    """Note in the record `doc` that the run took the state of the GPUs of
    `keys` as its baseline now (scripts/run_guard.py accept), as when it
    stops holding them (`released`): each reads its canary again, and a
    stage compares its settings with its other GPUs' before it benches."""
    now = utc()
    times = _release_times(doc) or {}
    times.update({str(k): now for k in keys})
    if times:
        doc["released"] = times


def _note_released(
    doc: Dict[str, Any],
    hold: Optional[Mapping[str, Any]],
    entries: Mapping[str, Any],
    host_wide: bool,
    now: str,
) -> None:
    """Note in `doc["released"]` when the run stops holding GPUs that the
    hold as it was (`hold`) held: a GPU whose entry is gone, and every GPU
    when a hold of every GPU ends (each then reads its canary again)."""
    if hold is None:
        return
    before = hold_entries(hold)
    released = _release_times(doc) or {}
    if run_wide(hold) and not (HOST in entries or UNKEYED in entries or host_wide):
        released[EVERY_GPU] = now
    for key in before:
        if key not in entries and key not in (HOST, UNKEYED):
            released[key] = now
    if released:
        doc["released"] = released


@dataclass
class Check:
    """What one check read (`keys`: the GPUs' keys and HOST) and found per
    key: (check, text) pairs, `check` one of CHECKS or "ttm"."""

    keys: List[str]
    found: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    @property
    def problems(self) -> List[str]:
        return [t for k in self.keys for _, t in self.found.get(k, [])]


class Guard:
    """The run's record in `<directory>/guard.json` (see the module
    docstring), for the process of `session`, named `by` in what it
    writes."""

    def __init__(self, directory: Path, session_id: str, by: str = "") -> None:
        self.dir = Path(directory)
        self.session = str(session_id)
        self.by = by or f"pid {os.getpid()}"
        self.path = self.dir / GUARD_FILE

    def read(self) -> Dict[str, Any]:
        try:
            doc = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    @contextlib.contextmanager
    def update(self) -> Iterator[Dict[str, Any]]:
        """The record, locked against the run's other processes (its released
        lines under RELEASED_LINES alone, `_upgrade`); written back when the
        block leaves without an exception."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            doc = self.read()
            _upgrade(doc)
            yield doc
            write_json(self.path, doc)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # ── baseline ──

    def baseline(
        self, gpus: Sequence[kfd.Gpu], read: Callable[[kfd.Gpu], Dict] = read_gpu
    ) -> Dict[str, Dict[str, Reading]]:
        """{key: readings} the run recorded for `gpus`; a GPU it has none for
        gets the readings it gives now."""
        doc = self.read()
        known = (doc.get("baseline") or {}).get("gpus") or {}
        if all(gpu_key(g) in known for g in gpus):
            return {gpu_key(g): known[gpu_key(g)]["readings"] for g in gpus}
        with self.update() as doc:
            base = doc.setdefault("baseline", {"recorded": utc(), "gpus": {}})
            for g in gpus:
                if gpu_key(g) not in base["gpus"]:
                    base["gpus"][gpu_key(g)] = {
                        "gpu": g.index,
                        "recorded": utc(),
                        "by": self.by,
                        "readings": read(g),
                    }
            known = dict(base["gpus"])
        return {gpu_key(g): known[gpu_key(g)]["readings"] for g in gpus}

    # ── hold ──

    def hold(self) -> Optional[Dict[str, Any]]:
        h = self.read().get("hold")
        return h if isinstance(h, dict) else None

    def settle(self, result: Check) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Take `result` into the run's hold: the entries of the keys it read
        become what it found there (an entry keeps when it was raised), the
        other GPUs' entries stay, and an unkeyed one goes when it found
        nothing. The hold is host-wide (`run_wide`) while `result` found
        departures on every GPU it read, two or more. (what happened, the
        hold): "raised" (it was not held), "added" (held, and a key of
        `result` has a new entry), "held", "cleared" (no entry is left: the
        hold as it was, `cleared` set) or "" (not held, nothing found)."""
        now = utc()
        read = [k for k in result.keys if k not in (HOST, UNKEYED)]
        everywhere = len(read) >= 2 and all(result.found.get(k) for k in read)
        with self.update() as doc:
            hold = doc.get("hold") if isinstance(doc.get("hold"), dict) else None
            entries = hold_entries(hold)
            added = False
            for key in result.keys:
                got = result.found.get(key)
                if not got:
                    entries.pop(key, None)
                    continue
                entry = entries.get(key) or {"raised": now, "by": self.by}
                added = added or key not in entries
                entry.update(
                    problems=[t for _, t in got], checks=[c for c, _ in got], seen=now
                )
                entries[key] = entry
            if not result.problems:
                entries.pop(UNKEYED, None)
            return self._write_hold(
                doc,
                hold,
                entries,
                now,
                added=added,
                seen=bool(result.problems),
                host_wide=everywhere,
            )

    def release(
        self, keys: Optional[Iterable[str]], by: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Drop the hold's entries of `keys` (every entry when None), as `by`
        decided, which leaves the GPUs still held judged one by one (not
        host-wide) until the next check; (what happened, the hold) as
        `settle` says it."""
        now = utc()
        with self.update() as doc:
            hold = doc.get("hold") if isinstance(doc.get("hold"), dict) else None
            entries = hold_entries(hold)
            for key in list(entries) if keys is None else list(keys):
                entries.pop(str(key), None)
            return self._write_hold(doc, hold, entries, now, by=by, host_wide=False)

    def _write_hold(
        self,
        doc: Dict[str, Any],
        hold: Optional[Dict[str, Any]],
        entries: Dict[str, Dict[str, Any]],
        now: str,
        *,
        added: bool = False,
        seen: bool = False,
        by: Optional[str] = None,
        host_wide: bool = False,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        _note_released(doc, hold, entries, host_wide, now)
        if entries:
            problems = _problems_of(entries)
            if hold is None:
                doc["hold"] = {
                    "raised": now,
                    "seen": now,
                    "by": self.by,
                    "session": self.session,
                    "problems": problems,
                    "entries": entries,
                    "host_wide": bool(host_wide),
                }
                return "raised", dict(doc["hold"])
            if seen:
                hold["seen"] = now
            hold.update(problems=problems, entries=entries, host_wide=bool(host_wide))
            return ("added" if added else "held"), dict(hold)
        if hold is None:
            return "", None
        done = dict(hold, cleared=now, cleared_by=by or self.by)
        doc["hold"] = None
        doc["holds"] = (list(doc.get("holds") or []) + [done])[-KEEP_RECORDS:]
        doc["last_cleared"] = done["cleared"]
        return "cleared", done

    # ── canary ──

    def canary_setup(self) -> Optional[Dict[str, Any]]:
        s = (self.read().get("canary") or {}).get("setup")
        return s if isinstance(s, dict) else None

    def set_canary_setup(
        self,
        lines: Sequence[str],
        settings: CanarySettings,
        source: str,
        kernels: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
        replaces: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record the run's canary lines, the kernel each names
        (`canary_kernels`), timings, tolerance and rule unless another
        process did first; the setup that holds. With `replaces` (the
        `record_id` of a setup the library can no longer time), that setup
        is kept in `setups` and replaced, and every reference dropped, unless
        another process replaced it first."""
        with self.update() as doc:
            can = doc.setdefault("canary", {})
            old = can.get("setup") if isinstance(can.get("setup"), dict) else None
            stale = old is not None and replaces and record_id(old) == replaces
            if old is None or stale:
                if stale:
                    can["setups"] = (list(can.get("setups") or []) + [old])[
                        -KEEP_RECORDS:
                    ]
                    can["references"] = {}
                can["setup"] = {
                    "id": uuid.uuid4().hex[:16],
                    "lines": [ln.rstrip("\n") for ln in lines],
                    "kernels": [
                        dict(k) if k else None for k in (kernels or [None] * len(lines))
                    ],
                    "timings": int(settings.timings),
                    "tolerance": float(settings.tolerance),
                    "rule": CANARY_RULE,
                    "created": utc(),
                    "by": self.by,
                    "source": source,
                }
                if stale:
                    can["setup"]["replaces"] = replaces
            return dict(can["setup"])

    def reference(self, key: str) -> Optional[Dict[str, Any]]:
        r = ((self.read().get("canary") or {}).get("references") or {}).get(key)
        return r if isinstance(r, dict) else None

    def released_at(self, key: str, doc: Optional[Mapping[str, Any]] = None) -> str:
        """When the run last stopped holding the GPU of `key` (of the record
        `doc`, default the one on disk): its own part of the hold ended, a
        hold of every GPU did, or the run took the GPU's state as its own
        (`released`, `note_accepted`; in a record without those times, when
        the run's last hold was cleared); "" when it never did."""
        doc = self.read() if doc is None else doc
        times = _release_times(doc)
        if times is None:
            got = [doc.get("last_cleared")]
        else:
            got = [times.get(EVERY_GPU), times.get(key)]
        return max((str(t) for t in got if t), default="")

    def canary_due(self, key: str) -> bool:
        """True when the GPU of `key` has not passed a canary in this session
        since the run last stopped holding it (`released_at`)."""
        doc = self.read()
        passed = ((doc.get("canary") or {}).get("passed") or {}).get(key)
        if not isinstance(passed, dict) or passed.get("session") != self.session:
            return True
        latest = self.released_at(key, doc)
        return bool(latest) and str(passed.get("at", "")) < latest

    def record_canary(self, key: str, rec: Mapping[str, Any]) -> Dict[str, Any]:
        """Record one canary of the GPU of `key`: its reference when it has
        none (another process's, taken meanwhile, holds) or when
        `rec["replaces"]` is the `record_id` of the reference it has, a pass
        for this session when `rec["passed"]`; returns `rec` with
        `reference` set when it became one."""
        out = {"by": self.by, **rec}
        with self.update() as doc:
            can = doc.setdefault("canary", {})
            refs = can.setdefault("references", {})
            old = refs.get(key)
            replaces = out.get("replaces")
            if out.get("reference") and (
                old is None or (replaces and record_id(old) == replaces)
            ):
                refs[key] = {
                    k: out[k]
                    for k in ("readings_us", "kernels", "gpu", "taken", "session", "by")
                    if k in out
                }
            elif out.get("reference"):
                out["reference"] = False
            if out.get("passed"):
                can.setdefault("passed", {})[key] = {
                    "session": self.session,
                    "at": utc(),
                    "shift": out.get("shift"),
                }
            checks = list(can.get("checks") or [])
            checks.append({k: v for k, v in out.items() if k != "readings_us"})
            can["checks"] = checks[-KEEP_RECORDS:]
        return out

    # ── quarantined lines ──

    def quarantined_shapes(
        self, scope: str = BENCH_SCOPE
    ) -> Set[Tuple[int, int, int, int]]:
        """The shapes of the lines the run quarantined under the protocol
        `scope` (an entry recorded without one is BENCH_SCOPE's)."""
        out: Set[Tuple[int, int, int, int]] = set()
        for e in self.read().get("quarantined_lines") or []:
            if str(e.get("scope") or BENCH_SCOPE) != scope:
                continue
            try:
                out.add(tuple(int(v) for v in e["shape"]))  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def quarantine_lines(self, entries: Iterable[Mapping[str, Any]]) -> None:
        """Record lines that hung, each under the protocol its `scope`
        names (BENCH_SCOPE when it names none)."""
        new = [
            dict(e, scope=str(e.get("scope") or BENCH_SCOPE), at=utc(), by=self.by)
            for e in entries
        ]
        if not new:
            return
        with self.update() as doc:
            doc["quarantined_lines"] = list(doc.get("quarantined_lines") or []) + new

    def release_lines(
        self, shapes: Optional[Iterable[Sequence[int]]]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Let the quarantined lines of `shapes` (M, N, K, batch; every line
        when None) be benched again in the run, in every protocol they were
        quarantined for: they move to RELEASED_LINES, each with when. (The
        lines let back in, how many stay quarantined.)"""
        want = None if shapes is None else {tuple(int(v) for v in s) for s in shapes}

        def chosen(e: Any) -> bool:
            if want is None:
                return True
            try:
                return tuple(int(v) for v in e.get("shape") or ()) in want
            except (AttributeError, TypeError, ValueError):
                return False

        with self.update() as doc:
            lines = list(doc.get("quarantined_lines") or [])
            gone = [e for e in lines if chosen(e)]
            keep = [e for e in lines if not chosen(e)]
            now = utc()
            doc["quarantined_lines"] = keep
            doc[RELEASED_LINES] = released_lines(doc) + [
                dict(e, released=now) for e in gone if isinstance(e, Mapping)
            ]
        return gone, len(keep)


def release_excluded(guard: Guard, refs: Sequence[Any]) -> List[str]:
    """Drop the run's hold of the GPUs `refs` names (runtime.exclude_gpus,
    `kfd.gpu_refs`; a PCI address also when the topology lists it no more),
    which no stage checks any more; the keys dropped."""
    if not refs:
        return []
    keys = keys_of(kfd.gpu_refs(refs)[0])
    for ref in refs:
        text = str(ref).strip().lower()
        if isinstance(ref, str) and text.count(":") in (1, 2):
            keys.add(text if text.count(":") == 2 else f"0000:{text}")
    gone = sorted(k for k in hold_entries(guard.hold()) if k.lower() in keys)
    if gone:
        guard.release(gone, by="runtime.exclude_gpus")
    return gone


def check(
    guard: Guard,
    gpus: Sequence[kfd.Gpu],
    *,
    host: bool = True,
    read: Callable[[kfd.Gpu], Dict] = read_gpu,
    ttm: Callable[[], List[str]] = stuck_ttm_workers,
    sleep: Callable[[float], None] = time.sleep,
) -> Check:
    """What departs from the run's baseline on `gpus`, on every other GPU of
    the topology the run's hold names, and (`host`) on the host now, each
    GPU's change confirmed by a second reading CONFIRM_S later. The host
    counts as read without `host`, so that turning the ttm check off
    settles its entry."""
    every = list(gpus)
    named = set(hold_entries(guard.hold())) - {gpu_key(g) for g in every}
    if named:
        every += [g for g in kfd.gpus() if gpu_key(g) in named]
    base = guard.baseline(every, read)

    def now() -> Dict[str, List[Tuple[str, str]]]:
        out = {}
        for g in every:
            got = changes(base[gpu_key(g)], read(g))
            if got:
                out[gpu_key(g)] = got
        return out

    first = now()
    found: Dict[str, List[Tuple[str, str]]] = {}
    if first:
        sleep(CONFIRM_S)
        again = now()
        labels = {gpu_key(g): gpu_label(g) for g in every}
        for key, got in first.items():
            later = dict(again.get(key, []))
            kept = [(c, f"{labels[key]}: {later[c]}") for c, _ in got if c in later]
            if kept:
                found[key] = kept
    stuck = ttm() if host else []
    if stuck:
        more = f" and {len(stuck) - 3} more" if len(stuck) > 3 else ""
        shown = ", ".join(stuck[:3]) + more
        found[HOST] = [
            (
                "ttm",
                f"{len(stuck)} kernel worker(s) of the ttm memory manager stuck in "
                f"D state ({shown}), as after a GPU reset",
            )
        ]
    return Check([gpu_key(g) for g in every] + [HOST], found)


def hold_text(hold: Mapping[str, Any], where: Path) -> str:
    """What holds the run's GPU work and how it ends, in one line."""
    entries = hold_entries(hold)
    checks = {c for e in entries.values() for c in e.get("checks") or []}
    remedies = [
        "restore the setting, or leave the GPU out with runtime.exclude_gpus, or "
        "take the GPUs' current state as the run's with scripts/run_guard.py accept"
    ]
    if "power_management" in checks:
        remedies.append(
            "power management off ends only when it is on again or the GPU is "
            "left out"
        )
    if "ttm" in checks:
        remedies.append(
            "the ttm workers are the host's: that problem ends only when they "
            "do, or with bench.health.ttm: false"
        )
    if run_wide(hold):
        scope = (
            ". This holds every GPU of the run (the host's problem, or one on "
            "every GPU checked): no GPU work of the run starts until they are"
        )
    else:
        keys = held_keys(hold)
        them = "them until they are" if len(keys) > 1 else "it until it is"
        scope = (
            f". This holds {', '.join(keys)} alone: no GPU work of the run "
            f"starts on {them}, and the run's other GPUs go on"
        )
    return (
        "the run's GPUs are not as they were when it started: "
        + "; ".join(_problems_of(entries))
        + f"{scope} (record: {where}); "
        + "; ".join(remedies)
    )


def wait_healthy(
    guard: Guard,
    gpus: Sequence[kfd.Gpu],
    settings: Settings,
    *,
    report: Callable[[str], None],
    stop: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    checker: Optional[Callable[[], Check]] = None,
    blocks: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> bool:
    """Block while the run's hold holds the work waiting (`blocks(hold)`;
    default: any hold does), checking `gpus` (`check`) and settling what it
    finds into the hold; True once it does not, False when `stop()` turns
    true or the wait passes `settings.max_wait_s` (0: no limit). A hold
    that the work goes on beside is reported once."""
    run = (
        checker
        if checker is not None
        else (lambda: check(guard, gpus, host=settings.ttm))
    )
    t0 = clock()
    noted = None
    while True:
        what, hold = guard.settle(run())
        if what == "cleared":
            assert hold is not None
            report(
                f"the run's GPUs pass their checks again after a hold since "
                f"{hold['raised']}; benching resumes once each GPU reads its "
                f"canary"
            )
        if what in ("cleared", ""):
            return True
        assert hold is not None
        new = what in ("raised", "added")
        if blocks is not None and not blocks(hold):
            if new or noted is None:
                report(hold_text(hold, guard.path))
            return True
        if new or noted is None or clock() - noted >= REPORT_S:
            report(hold_text(hold, guard.path))
            noted = clock()
        limit = settings.max_wait_s
        if stop() or (limit > 0 and clock() - t0 >= limit):
            return False
        sleep(max(1.0, settings.poll_s))


# ── canary ───────────────────────────────────────────────────────────────────


def choose_canary_lines(
    lines: Sequence[str],
    n: int,
    restamp: Callable[[str], Optional[str]],
    shape_of: Callable[[str], Optional[Tuple[int, int, int, int]]],
) -> List[str]:
    """Up to `n` canary lines from `lines`: the distinct shapes ordered by
    work (M * N * K * batch), taken evenly from the least to the median
    (one line: the shape halfway between them), each restamped by `restamp`
    (None: that line cannot be a canary)."""
    by_shape: Dict[Tuple[int, int, int, int], str] = {}
    for ln in lines:
        s = shape_of(ln)
        if s is not None and s not in by_shape:
            by_shape[s] = ln
    order = sorted(by_shape, key=lambda s: (s[0] * s[1] * s[2] * s[3], s))
    usable = [(s, restamp(by_shape[s])) for s in order]
    usable = [(s, ln) for s, ln in usable if ln is not None]
    if not usable or n < 1:
        return []
    half = usable[: max(1, (len(usable) + 1) // 2)]
    if len(half) <= n:
        picks = list(range(len(half)))
    elif n == 1:
        picks = [(len(half) - 1) // 2]
    else:
        picks = sorted({round(i * (len(half) - 1) / (n - 1)) for i in range(n)})
    return [str(half[i][1]).rstrip("\n") for i in picks]


def canary_kernels(
    lines: Sequence[str],
    listed: Callable[[str], Optional[List[int]]],
    identity: Any = None,
) -> List[Optional[Dict[str, Any]]]:
    """Per canary line the kernel it names by solution index (`listed`
    gives a line's list; a canary line names one): its index and, with
    `identity` (lib/kernel_identity.LibraryIdentity), the library file that
    holds it, its kernel id and signature; None for a line that names
    none."""
    from . import kernel_identity as ki

    out: List[Optional[Dict[str, Any]]] = []
    for ln in lines:
        ids = listed(ln) or []
        if len(ids) != 1:
            out.append(None)
            continue
        stems = [] if identity is None else list(identity.stems)
        stem = next((t for t in stems if int(ids[0]) in identity.file_ids[t]), None)
        out.append({"index": int(ids[0]), "stem": stem, **ki.naming(identity, ids[0])})
    return out


def canary_lines_now(
    setup: Mapping[str, Any],
    identity: Any,
    restamp: Callable[[str, int], str],
) -> Tuple[List[Optional[str]], List[str]]:
    """The setup's canary lines as this stage benches them: with `identity`
    holding the library file a line's kernel was recorded in, a line whose
    recorded index names another kernel there now is restamped to the index
    its recorded kernel has now (`restamp(line, index)`), or left out (None)
    when the library holds that kernel under no index or several. (lines,
    what changed)."""
    lines = [str(ln) for ln in setup.get("lines") or []]
    kernels = list(setup.get("kernels") or [])
    kernels += [None] * (len(lines) - len(kernels))
    out: List[Optional[str]] = []
    notes: List[str] = []
    for ln, k in zip(lines, kernels):
        k = k if isinstance(k, Mapping) else {}
        kid, index, stem = k.get("kernel_id"), k.get("index"), k.get("stem")
        if identity is None or not kid or index is None or stem not in identity.stems:
            out.append(ln)
            continue
        if identity.file_ids[stem].get(int(index)) == kid:
            out.append(ln)
            continue
        now = identity.index_of(kid)
        name = k.get("signature") or kid
        if now is None:
            out.append(None)
            notes.append(f"{name}: no single solution index of the library now")
        else:
            out.append(restamp(ln, int(now)))
            notes.append(f"{name}: solution index {index} is {now} now")
    return out, notes


def canary_entries(
    lines: Sequence[str], timings: int, name: Callable[[str, str], str]
) -> List[str]:
    """The yaml entries of a canary process: a warm-up copy of every line,
    then `timings` rounds of every line in turn, each named apart by
    `name(line, entry name)`."""
    out = [name(ln, f"hcanary{i}_warm") for i, ln in enumerate(lines)]
    for t in range(int(timings)):
        out += [name(ln, f"hcanary{i}_t{t}") for i, ln in enumerate(lines)]
    return out


def canary_readings(
    latencies: Sequence[Optional[float]], n_lines: int, timings: int
) -> List[Optional[float]]:
    """Per line, the median of its timed copies' latencies (`latencies`, one
    per entry of `canary_entries` in order, None for an entry without one);
    None for a line with no timing."""
    timed = list(latencies[n_lines:])
    out: List[Optional[float]] = []
    for i in range(n_lines):
        got = [
            timed[t * n_lines + i]
            for t in range(int(timings))
            if t * n_lines + i < len(timed) and timed[t * n_lines + i] is not None
        ]
        out.append(statistics.median(got) if got else None)
    return out


def timed_kernels(
    names: Sequence[Optional[str]], n_lines: int, timings: int
) -> List[Optional[str]]:
    """Per line, the kernel its timed copies ran (`names`, one per entry of
    `canary_entries` in order, None for an entry the bench named none for);
    None when they name none or disagree."""
    timed = list(names[n_lines:])
    out: List[Optional[str]] = []
    for i in range(n_lines):
        got = {
            timed[t * n_lines + i]
            for t in range(int(timings))
            if t * n_lines + i < len(timed) and timed[t * n_lines + i]
        }
        out.append(got.pop() if len(got) == 1 else None)
    return out


def judge(
    reference: Sequence[Optional[float]],
    readings: Sequence[Optional[float]],
    tolerance: float,
    reference_kernels: Optional[Sequence[Optional[str]]] = None,
    kernels: Optional[Sequence[Optional[str]]] = None,
) -> Dict[str, Any]:
    """The verdict of a canary with `readings` against `reference` (one per
    line): `shift`, the median over lines read in both of ln(reading /
    reference), and `passed` when |shift| <= ln(1 + tolerance) with at
    least half the lines read. A line whose timed kernel (`kernels`) is not
    the one at the reference (`reference_kernels`), both known, is not
    read; `other_kernels` counts those lines."""
    ref_k = list(reference_kernels or [])
    now_k = list(kernels or [])
    logs: List[float] = []
    other = 0
    for i, (ref, r) in enumerate(zip(reference, readings)):
        if not (ref and r and ref > 0 and r > 0):
            continue
        a = ref_k[i] if i < len(ref_k) else None
        b = now_k[i] if i < len(now_k) else None
        if a and b and a != b:
            other += 1
            continue
        logs.append(math.log(r / ref))
    need = max(1, math.ceil(len(reference) / 2))
    if len(logs) < need:
        return {
            "shift": None,
            "lines_read": len(logs),
            "other_kernels": other,
            "passed": False,
            "why": f"{len(logs)} of {len(reference)} canary lines read"
            + (f", {other} timing another kernel than the reference" if other else ""),
        }
    shift = statistics.median(logs)
    ok = abs(shift) <= math.log1p(float(tolerance))
    return {
        "shift": shift,
        "ratio": math.exp(shift),
        "lines_read": len(logs),
        "other_kernels": other,
        "passed": ok,
        "why": (
            ""
            if ok
            else f"the canary reads {math.exp(shift):.3f}x its reference, beyond "
            f"the tolerance {float(tolerance):g}"
        ),
    }
