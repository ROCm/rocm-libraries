# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Load on the CPUs a stage benches from, read from /proc without root.

A bench's latencies include its host thread's work, so another process
running on the cores the stage is pinned to slows the benches, a
launch-bound GEMM the most. `Monitor` samples those cores (the stage's CPU
affinity, which its benches inherit) in windows of `every_s`: per core the
share of the window it was not idle (`/proc/stat`; iowait counts as idle),
and over the set the share the run's bench stages used (`bench_trees`:
user and system time of every live process of their trees, plus what their
processes reaped from children that exited, from `/proc/<pid>/stat`),
which is subtracted: the rest is `foreign` load. The bench stages are the
stage's own process tree and, when RUN_ROOT_VAR names an ancestor of the
stage (the orchestrator, or `scripts/audit_v2.py run`, names itself), the
tree of every process under it whose command line names the stage's script,
as the audit's per-part stages are: concurrent stages of one run then do
not count each other's benches as other load, while the run's other work
(training beside a parallel audit build) still counts. Own CPU time is
spread over the set, so it is subtracted from the set's mean only, not per
core.

A window is quiet when its foreign load is at most `quiet_busy` and, when
nothing of the stage ran in it, no core was busier than `quiet_core_busy`
(with the stage's benches running, their own cores are busy by design; the
default 1.0 leaves the rule out, for a stage pinned to many cores, where
one busy core does not reach a bench). A window is hot when its foreign
load is above `hot_busy`. The stage waits
before a bench part until the latest window is quiet (`wait_quiet`, at most
`max_wait_s`) and aborts a part that sees `abort_samples` hot windows
(`hot_since`).

The first window is `FIRST_WINDOW_S` long, so that a stage does not wait a
full window before its first part. When `/proc/stat` gives none of the
cores, the monitor cannot measure: it says so once (`report`) and every
window counts as quiet.

`TILEWRIGHT_PROC_ROOT` replaces `/proc` for these reads (tests use a fake
tree); process identity elsewhere in the pipeline keeps the real `/proc`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

PROC_ROOT_VAR = "TILEWRIGHT_PROC_ROOT"
RUN_ROOT_VAR = "TILEWRIGHT_GUARD_ROOT"
DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "every_s": 10.0,
    "quiet_busy": 0.15,
    "quiet_core_busy": 1.0,
    "hot_busy": 0.25,
    "abort_samples": 2,
    "max_wait_s": 28800.0,
}
FIRST_WINDOW_S = 2.0
# Below this share of the set's CPU time the stage's own tree counts as idle
# in a window, and the per-core rule applies.
OWN_IDLE_SHARE = 0.005
REPORT_S = 600.0
SERIES_FILE = "host_load.jsonl"


def proc_root() -> Path:
    return Path(os.environ.get(PROC_ROOT_VAR) or "/proc")


@dataclass(frozen=True)
class Settings:
    """`bench.quiet_host` of a run config, over DEFAULTS."""

    enabled: bool = DEFAULTS["enabled"]
    every_s: float = DEFAULTS["every_s"]
    quiet_busy: float = DEFAULTS["quiet_busy"]
    quiet_core_busy: float = DEFAULTS["quiet_core_busy"]
    hot_busy: float = DEFAULTS["hot_busy"]
    abort_samples: int = DEFAULTS["abort_samples"]
    max_wait_s: float = DEFAULTS["max_wait_s"]

    @classmethod
    def of(cls, cfg: Mapping[str, Any]) -> "Settings":
        blk = (cfg.get("bench") or {}).get("quiet_host") if cfg else None
        blk = blk if isinstance(blk, Mapping) else {}
        return cls(
            enabled=bool(blk.get("enabled", DEFAULTS["enabled"])),
            every_s=float(blk.get("every_s", DEFAULTS["every_s"])),
            quiet_busy=float(blk.get("quiet_busy", DEFAULTS["quiet_busy"])),
            quiet_core_busy=float(
                blk.get("quiet_core_busy", DEFAULTS["quiet_core_busy"])
            ),
            hot_busy=float(blk.get("hot_busy", DEFAULTS["hot_busy"])),
            abort_samples=int(blk.get("abort_samples", DEFAULTS["abort_samples"])),
            max_wait_s=float(blk.get("max_wait_s", DEFAULTS["max_wait_s"])),
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        wait = f"at most {self.max_wait_s:g} s" if self.max_wait_s > 0 else "no limit"
        core = (
            f", every core <= {self.quiet_core_busy:g} while nothing of the stage "
            f"runs"
            if self.quiet_core_busy < 1.0
            else ""
        )
        return (
            f"a part starts when the latest {self.every_s:g} s window of its cores "
            f"is quiet (other load <= {self.quiet_busy:g}{core}; waiting {wait}) "
            f"and is aborted and benched again after {self.abort_samples} "
            f"window(s) with other load > {self.hot_busy:g}"
        )


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def cpu_times(
    cores: Iterable[int], root: Optional[Path] = None
) -> Dict[int, Tuple[int, int]]:
    """{core: (busy, total)} jiffies of `cores` from `<proc>/stat`; a core
    the file does not list is left out."""
    want = {f"cpu{int(c)}" for c in cores}
    text = _read((proc_root() if root is None else Path(root)) / "stat")
    out: Dict[int, Tuple[int, int]] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts or parts[0] not in want:
            continue
        try:
            values = [int(v) for v in parts[1:9]]
        except ValueError:
            continue
        if len(values) < 5:
            continue
        total = sum(values)
        out[int(parts[0][3:])] = (total - values[3] - values[4], total)
    return out


def _stat_fields(text: str) -> List[str]:
    """The fields of a `/proc/<pid>/stat` after the command name."""
    return text[text.rfind(")") + 2 :].split()


def _table(base: Path) -> Tuple[Dict[int, List[int]], Dict[int, List[str]]]:
    """({pid: child pids}, {pid: stat fields}) of every process now."""
    kids: Dict[int, List[int]] = {}
    stats: Dict[int, List[str]] = {}
    try:
        entries = [p for p in base.iterdir() if p.name.isdigit()]
    except OSError:
        return kids, stats
    for p in entries:
        text = _read(p / "stat")
        if text is None:
            continue
        fields = _stat_fields(text)
        try:
            kids.setdefault(int(fields[1]), []).append(int(p.name))
        except (IndexError, ValueError):
            continue
        stats[int(p.name)] = fields
    return kids, stats


def _subtree(pid: int, kids: Mapping[int, List[int]]) -> List[int]:
    out: List[int] = []
    todo, seen = [int(pid)], set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
        todo.extend(kids.get(cur, ()))
    return out


def _jiffies(fields: List[str]) -> int:
    try:
        return sum(int(v) for v in fields[11:15])
    except (IndexError, ValueError):
        return 0


def tree_cpu(pid: int, root: Optional[Path] = None) -> int:
    """User and system jiffies of process `pid` and its live descendants,
    plus what they reaped from children that exited: every process's time
    is counted once, by itself while it lives and by its reaper after."""
    kids, stats = _table(proc_root() if root is None else Path(root))
    return sum(_jiffies(stats[p]) for p in _subtree(pid, kids) if p in stats)


def _cmdline(path: Path) -> str:
    try:
        return (
            (path / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except OSError:
        return ""


def bench_trees(
    own: int,
    run: Optional[int] = None,
    marker: Optional[str] = None,
    root: Optional[Path] = None,
) -> Dict[Tuple[int, str], int]:
    """{(pid, start time): jiffies of its tree (`tree_cpu`)} of the run's
    bench stages: the process `own` and, within the tree of `run`, every
    process whose command line contains `marker`; a process inside another
    one's tree is counted with that one."""
    base = proc_root() if root is None else Path(root)
    kids, stats = _table(base)
    tops = [int(own)]
    if run is not None and marker:
        tops += [
            p
            for p in _subtree(run, kids)
            if p != int(own) and marker in _cmdline(base / str(p))
        ]
    trees = {t: set(_subtree(t, kids)) for t in dict.fromkeys(tops)}
    out: Dict[Tuple[int, str], int] = {}
    for t, members in trees.items():
        if t not in stats or any(t in m for u, m in trees.items() if u != t):
            continue
        start = stats[t][19] if len(stats[t]) > 19 else ""
        out[(t, start)] = sum(_jiffies(stats[p]) for p in members if p in stats)
    return out


def own_jiffies(
    before: Mapping[Tuple[int, str], int], after: Mapping[Tuple[int, str], int]
) -> int:
    """The jiffies the bench stages (`bench_trees`) used between two
    readings: each tree's growth, all of a tree that started in between;
    what a tree that ended in between used before it ended is lost, not
    taken back."""
    return sum(j - before[k] if k in before else j for k, j in after.items())


def run_root(
    pid: Optional[int] = None, environ: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """The process RUN_ROOT_VAR of `environ` (default: this process's)
    names when it is process `pid` (default: this one) or one of its
    ancestors, else None."""
    from .kfd import ancestors

    env = os.environ if environ is None else environ
    try:
        named = int(env.get(RUN_ROOT_VAR) or 0)
    except ValueError:
        return None
    if named <= 0:
        return None
    return named if named in ancestors(os.getpid() if pid is None else pid) else None


def loadavg(root: Optional[Path] = None) -> Optional[List[float]]:
    text = _read((proc_root() if root is None else Path(root)) / "loadavg")
    try:
        return [float(x) for x in (text or "").split()[:3]] or None
    except ValueError:
        return None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Sample:
    """One window of the monitored cores: its end (UTC) and length, the
    mean and the largest per-core busy share, the share the stage's own
    tree used, the rest (`foreign`) and the load average at its end."""

    t: str
    seconds: float
    mean_busy: float
    max_core_busy: float
    own_share: float
    foreign: float
    loadavg: Optional[List[float]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "s": round(self.seconds, 2),
            "mean": round(self.mean_busy, 4),
            "max_core": round(self.max_core_busy, 4),
            "own": round(self.own_share, 4),
            "foreign": round(self.foreign, 4),
            "loadavg": self.loadavg,
        }


def window_sample(
    before: Mapping[int, Tuple[int, int]],
    after: Mapping[int, Tuple[int, int]],
    own_jiffies: int,
    seconds: float,
) -> Optional[Sample]:
    """The Sample of one window from the cores' (busy, total) jiffies at its
    start and end and the jiffies the stage's tree used in it; None when no
    core is in both readings."""
    cores = [c for c in after if c in before]
    if not cores:
        return None
    per: List[float] = []
    busy = total = 0
    for c in cores:
        db = after[c][0] - before[c][0]
        dt = after[c][1] - before[c][1]
        per.append(db / dt if dt > 0 else 0.0)
        busy += max(0, db)
        total += max(0, dt)
    mean = busy / total if total else 0.0
    own = min(1.0, max(0, own_jiffies) / total) if total else 0.0
    return Sample(
        t=_utc(),
        seconds=float(seconds),
        mean_busy=mean,
        max_core_busy=max(per),
        own_share=own,
        foreign=max(0.0, mean - own),
    )


class Monitor:
    """Samples `cores` in windows of `settings.every_s` on a thread of its
    own (`start`, `stop`), the first window FIRST_WINDOW_S long; the run's
    bench stages are the process `own_root` and, under `run_root`, the
    processes whose command line contains `marker` (`bench_trees`). Every
    window is appended to `series_path` when given. Workers wait for a
    quiet window (`wait_quiet`, `quiet_now`) and count hot ones (`mark`,
    `hot_since`); `summary` is what the stage records."""

    def __init__(
        self,
        cores: Iterable[int],
        own_root: int,
        settings: Settings,
        *,
        run_root: Optional[int] = None,
        marker: Optional[str] = None,
        series_path: Optional[Path] = None,
        report: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        first_window_s: float = FIRST_WINDOW_S,
    ) -> None:
        self.cores = sorted({int(c) for c in cores})
        self.own_root = int(own_root)
        self.run_root = None if run_root is None else int(run_root)
        self.marker = marker
        self.settings = settings
        self.series_path = series_path
        self.report = report
        self.clock = clock
        self.first_window_s = float(first_window_s)
        self.samples: List[Sample] = []
        self.unmeasurable: Optional[str] = None
        self.waits: List[Dict[str, Any]] = []
        self.gave_up = False
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="host-load", daemon=True)
        self._noted = -REPORT_S

    # ── sampling ──

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=self.settings.every_s + 5.0)

    def _own(self) -> Dict[Tuple[int, str], int]:
        return bench_trees(self.own_root, self.run_root, self.marker)

    def _measure(self, seconds: float) -> Optional[Sample]:
        before = cpu_times(self.cores)
        own0 = self._own()
        t0 = self.clock()
        if self._stop.wait(seconds):
            return None
        after = cpu_times(self.cores)
        own1 = self._own()
        if not after:
            return None
        s = window_sample(before, after, own_jiffies(own0, own1), self.clock() - t0)
        if s is not None:
            s.loadavg = loadavg()
        return s

    def _run(self) -> None:
        if not cpu_times(self.cores):
            why = (
                f"{proc_root() / 'stat'} gives none of the cores "
                f"{_cpulist(self.cores)}"
            )
            with self._cond:
                self.unmeasurable = why
                self._cond.notify_all()
            if self.report is not None:
                self.report(
                    f"{why}: the load on them is not measured; every part starts"
                )
            return
        window = self.first_window_s
        while not self._stop.is_set():
            s = self._measure(window)
            window = self.settings.every_s
            if s is None:
                continue
            with self._cond:
                self.samples.append(s)
                self._cond.notify_all()
            if self.series_path is not None:
                try:
                    with open(self.series_path, "a") as f:
                        f.write(json.dumps(s.to_json()) + "\n")
                except OSError:
                    pass

    # ── verdicts ──

    def quiet(self, s: Sample) -> bool:
        st = self.settings
        own_idle = s.own_share < OWN_IDLE_SHARE
        return s.foreign <= st.quiet_busy and (
            not own_idle or s.max_core_busy <= st.quiet_core_busy
        )

    def hot(self, s: Sample) -> bool:
        return s.foreign > self.settings.hot_busy

    def latest(self) -> Optional[Sample]:
        with self._cond:
            return self.samples[-1] if self.samples else None

    def quiet_now(self) -> bool:
        """Whether the latest window is quiet (`wait_quiet` would not wait);
        True for a monitor that cannot measure."""
        with self._cond:
            s = self.samples[-1] if self.samples else None
            unmeasurable = self.unmeasurable
        return unmeasurable is not None or (s is not None and self.quiet(s))

    def mark(self) -> int:
        """The number of windows so far, to count hot ones from."""
        with self._cond:
            return len(self.samples)

    def hot_since(self, mark: int) -> int:
        with self._cond:
            return sum(1 for s in self.samples[mark:] if self.hot(s))

    def _note(self, msg: str) -> None:
        now = self.clock()
        with self._cond:
            due = now - self._noted >= REPORT_S
            if due:
                self._noted = now
        if due and self.report is not None:
            self.report(msg)

    def wait_quiet(self, stop: Callable[[], bool] = lambda: False) -> bool:
        """Block until the latest window is quiet (True), `stop()` turns true
        or the wait passes `max_wait_s` (False; `gave_up` then stays set). A
        monitor that cannot measure never waits."""
        t0 = self.clock()
        waited = False
        while True:
            with self._cond:
                s = self.samples[-1] if self.samples else None
                unmeasurable = self.unmeasurable
            if unmeasurable is not None or (s is not None and self.quiet(s)):
                if waited:
                    self._record_wait(t0, "quiet")
                return True
            if stop() or self.gave_up:
                if waited:
                    self._record_wait(t0, "stopped")
                return False
            limit = self.settings.max_wait_s
            if limit > 0 and self.clock() - t0 >= limit:
                with self._cond:
                    self.gave_up = True
                self._record_wait(t0, "gave up")
                return False
            if s is not None:
                waited = True
                self._note(
                    f"cores {_cpulist(self.cores)} are not quiet: other load "
                    f"{s.foreign:.2f} (quiet <= {self.settings.quiet_busy:g}), "
                    f"busiest core {s.max_core_busy:.2f}, load average "
                    f"{(s.loadavg or ['?'])[0]}; no bench part starts until "
                    f"they are"
                )
            with self._cond:
                self._cond.wait(timeout=1.0)

    def _record_wait(self, t0: float, how: str) -> None:
        with self._cond:
            self.waits.append({"s": round(self.clock() - t0, 1), "ended": how})

    # ── records ──

    def window(self, start: int, end: Optional[int] = None) -> Dict[str, Any]:
        """What the windows from `start` (a `mark`) to `end` saw."""
        with self._cond:
            got = self.samples[start:end]
        if not got:
            return {"samples": 0}
        f = [s.foreign for s in got]
        return {
            "samples": len(got),
            "foreign_mean": round(sum(f) / len(f), 4),
            "foreign_max": round(max(f), 4),
            "hot": sum(1 for s in got if self.hot(s)),
        }

    def summary(self) -> Dict[str, Any]:
        with self._cond:
            got = list(self.samples)
            waits = list(self.waits)
        out: Dict[str, Any] = {
            "cores": _cpulist(self.cores),
            "run_root": self.run_root,
            "settings": self.settings.to_json(),
            "first_window_s": self.first_window_s,
            "unmeasurable": self.unmeasurable,
            "samples": len(got),
            "waits": len(waits),
            "waited_s": round(sum(w["s"] for w in waits), 1),
            "gave_up": self.gave_up,
        }
        if got:
            f = [s.foreign for s in got]
            out.update(
                foreign_mean=round(sum(f) / len(f), 4),
                foreign_max=round(max(f), 4),
                hot=sum(1 for s in got if self.hot(s)),
                not_quiet=sum(1 for s in got if not self.quiet(s)),
                loadavg_max=max((s.loadavg[0] for s in got if s.loadavg), default=None),
            )
        if self.series_path is not None:
            out["series"] = self.series_path.name
        return out


def _cpulist(cpus: Iterable[int]) -> str:
    from .fingerprint import format_cpulist

    return format_cpulist(cpus)


def stage_cores() -> List[int]:
    """The CPUs this process may run on, which its benches inherit."""
    return sorted(os.sched_getaffinity(0))
