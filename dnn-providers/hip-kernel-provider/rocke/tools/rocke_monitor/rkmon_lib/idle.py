# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The idle short-circuit of `rk tick`: a tick whose inputs have not changed since the last full tick, and whose timers
have not run out, does none of the heavy work.

Every tick first reads a change fingerprint: the parts registered here (`part`). A part is one input of the tick, read
cheaply: the open PRs' heads and update times, the base branch's head and CI, the state file, the jobs the state refers
to, and so on. A full tick records the fingerprint it started from, with the deadlines of every registered timer
(`timer`): the moments a rule of the tick changes its answer with time alone (a head settles, a gap ends, a report falls
due). The next tick is idle when:
- a full tick is on record, less than `idle_full_tick_min` minutes old (0: every tick is full);
- every part reads the same as at that full tick;
- no recorded deadline has passed;
- and the tick was not run with `--full`.

An idle tick refreshes only the dashboards' "checked" stamps (the registered `stamp`s), records itself in the state and
prints one line. Any doubt makes the tick full: a part that cannot be read, a timer that cannot be computed, a change to
the code or the config.

Each stage registers its own parts, timers and stamps, in its own module, next to the rule each one stands for. The
registry is all this engine knows of the stages: a new stage, job kind or dashboard adds its entries and changes nothing
here.

Part values are plain JSON. A part is one of three kinds:
- remote (`remote=True`): read from GitHub or another host, in parallel with the other remote parts;
- local, read at the start of the tick (the default);
- local, re-read when the full tick ends (`end=True`): inputs the tick itself writes, such as the state file. `merge`
  combines the value read at the start with the one read at the end, when the tick's own reading of the input must win
  (the jobs: a job that ended during the tick, after its step looked at it, is news for the next tick).

Timers return (label, UTC time) pairs. A timer is one-shot by default: a deadline already past when the full tick ends
has had that tick. `again=True` keeps past deadlines, for work a tick leaves for the next one (a retry, a pass cut short
by its budget): the next tick is then full.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import shlex
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import deploy_tree, registry
from .store import minutes_since, now, parse_ts

KEY = "idle"  # st[KEY]: the last full tick's fingerprint, deadlines, time and reasons
STATS = "tick_stats"  # st[STATS]: every tick of the last day, [time, "idle" | "full", seconds]
STAMP_KEYS = (
    "last_tick",
    KEY,
    STATS,
)  # what an idle tick writes; the state part leaves them out
ROOT = pathlib.Path(__file__).resolve().parent.parent
# The tick and loop settings (config keys), each with its default in the registry (rkmon_lib/registry.py)
SETTINGS = (
    "idle_full_tick_min",  # a full tick at least this often (0: every tick is full)
    "tick_interval_min",  # tools/tick_loop.py: minutes from one tick's start to the next one's
    "loop_heartbeat_min",  # tools/tick_loop.py: it exits with LOOP_HEARTBEAT this long after it started
    "tick_timeout_min",  # tools/tick_loop.py: a tick still running this long is stopped
    "loop_log_max_mb",  # tools/tick_loop.py: the loop log moves to <log>.1 past this size
)
# tools/tick_loop.py's files, in state/: its lock, its log, its pid, when its last tick started (epoch seconds), and the
# standing lines it has reported (`rk tick-filter --seen`)
LOOP_FILES = {
    "LOOP_LOCK": "tick_loop.lock",
    "LOOP_LOG": "tick_loop.log",
    "LOOP_PID": "tick_loop.pid",
    "LOOP_LAST": "tick_loop.last",
    "LOOP_SEEN": "tick_loop.seen",
}
STATS_WINDOW_MIN = 24 * 60


@dataclass(frozen=True)
class Part:
    name: str
    read: Callable[[dict, dict, Any], Any]  # (cfg, st, store) -> a JSON value
    remote: bool = False
    end: bool = False
    merge: Callable[[Any, Any], Any] | None = (
        None  # (start value, end value) -> the value recorded
    )


@dataclass(frozen=True)
class Timer:
    name: str
    due: Callable[
        [dict, dict, Any], Any
    ]  # (cfg, st, store) -> iterable of (label, UTC time)
    again: bool = False


PARTS: dict[str, Part] = {}
TIMERS: dict[str, Timer] = {}
STAMPS: dict[str, Callable[[dict, Any, str], list[str]]] = {}


def part(name: str, remote: bool = False, end: bool = False, merge=None):
    """Register a fingerprint part: decorate its read(cfg, st, store)."""

    def wrap(fn):
        PARTS[name] = Part(name, fn, remote, end, merge)
        return fn

    return wrap


def timer(name: str, again: bool = False):
    """Register a timer: decorate its due(cfg, st, store), which yields (label, UTC time) pairs."""

    def wrap(fn):
        TIMERS[name] = Timer(name, fn, again)
        return fn

    return wrap


def stamp(name: str):
    """Register a "checked" stamp: decorate its refresh(cfg, store, ts), which returns advisory lines (a failure)."""

    def wrap(fn):
        STAMPS[name] = fn
        return fn

    return wrap


def file_part(name: str, path_of: Callable[[Any], pathlib.Path]):
    """Register a file the full tick reads and other commands write: its size and modification time, re-read when the
    full tick ends (`path_of(store)` names it)."""

    def read(cfg, st, store):
        p = pathlib.Path(path_of(store))
        try:
            s = p.stat()
        except FileNotFoundError:
            return None
        return [s.st_size, s.st_mtime_ns]

    PARTS[name] = Part(name, read, end=True)


# --- small helpers for the registrations ------------------------------------------------------------------------------


def digest(value) -> str:
    return hashlib.sha1(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def plus(ts: str, minutes: float) -> str:
    """UTC time `ts` plus `minutes`, in the state's format."""
    return (parse_ts(ts) + dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def settings(cfg: dict) -> dict:
    """The tick and loop settings (SETTINGS), from config with the registry's defaults; a value that is not a number
    >= 0 is refused."""
    out = {}
    for k in SETTINGS:
        v = cfg.get(k, registry.default(k))
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"config {k} must be a number of minutes >= 0, not {v!r}")
        out[k] = v
    return out


# --- the engine's own parts -------------------------------------------------------------------------------------------


@part("state", end=True)
def _state(cfg, st, store):
    """The state file, but for what an idle tick writes: any change by another command (a hold, a clear, a review
    driver step, a post) makes the next tick full."""
    return digest({k: v for k, v in st.items() if k not in STAMP_KEYS})


@part("config")
def _config(cfg, st, store):
    return digest(cfg)


@part("code")
def _code(cfg, st, store):
    """The toolkit's Python files: a tick right after a code change runs in full."""
    files = sorted(list(ROOT.glob("*.py")) + list(ROOT.glob("rkmon_lib/*.py")))
    return digest(
        [
            [str(f.relative_to(ROOT)), f.stat().st_size, f.stat().st_mtime_ns]
            for f in files
        ]
    )


# --- reading the fingerprint and the deadlines ------------------------------------------------------------------------


def probe(
    cfg: dict, st: dict, store, parts: dict | None = None
) -> tuple[dict, dict, list[str]]:
    """Read every part: (values, seconds per part, errors). Remote parts run in parallel."""
    parts = PARTS if parts is None else parts
    values, secs, errors = {}, {}, []

    def one(p: Part):
        t0 = time.monotonic()
        try:
            return p.name, p.read(cfg, st, store), None, time.monotonic() - t0
        except (
            Exception
        ) as exc:  # noqa: BLE001 - any failure makes the tick full (fail closed)
            return (
                p.name,
                None,
                f"{p.name}: {' '.join(str(exc).split())[:160]}",
                time.monotonic() - t0,
            )

    remote = [p for p in parts.values() if p.remote]
    results = []
    if remote:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(remote)) as pool:
            futures = [pool.submit(one, p) for p in remote]
            results += [one(p) for p in parts.values() if not p.remote]
            results += [f.result() for f in futures]
    else:
        results += [one(p) for p in parts.values()]
    for name, value, err, s in results:
        secs[name] = round(s, 2)
        if err:
            errors.append(err)
        else:
            values[name] = value
    return values, secs, errors


def deadlines(
    cfg: dict, st: dict, store, timers: dict | None = None, at: str | None = None
) -> dict[str, str]:
    """Every timer's deadlines, label -> UTC time: one-shot ones only when still ahead of `at` (default: now). A timer
    that cannot be computed is due at once."""
    t = at or now()
    out: dict[str, str] = {}
    for tm in (TIMERS if timers is None else timers).values():
        try:
            pairs = list(tm.due(cfg, st, store) or ())
        except (
            Exception
        ) as exc:  # noqa: BLE001 - fail closed: the next tick runs in full
            out[f"{tm.name} (not computed: {' '.join(str(exc).split())[:80]})"] = t
            continue
        for label, ts in pairs:
            if ts and (tm.again or ts > t):
                out[label] = min(out.get(label, ts), ts)
    return out


@dataclass
class Decision:
    idle: bool
    reasons: list[str]
    fp: dict | None  # None when a part could not be read
    secs: dict
    at: str


def decide(
    cfg: dict, st: dict, store, full: bool = False, parts: dict | None = None
) -> Decision:
    """Idle or full, and why it is full. Reads every part (the full tick records them); writes nothing."""
    t = now()
    values, secs, errors = probe(cfg, st, store, parts)
    rec = st.get(KEY) or {}
    reasons = ["--full"] if full else []
    if errors:
        reasons.append("fingerprint not read (" + "; ".join(errors)[:300] + ")")
    every = settings(cfg)["idle_full_tick_min"]
    old, last = rec.get("fp"), rec.get("full_at")
    if not old or not last:
        reasons.append("no full tick on record")
    else:
        if minutes_since(last, t) >= every:
            reasons.append(
                f"the last full tick is {every:g} min old"
                if every
                else f"{deploy_tree.name('idle_full_tick_min')} is 0"
            )
        if not errors:
            changed = sorted(
                n for n in set(values) | set(old) if values.get(n) != old.get(n)
            )
            if changed:
                reasons.append("changed: " + ", ".join(changed))
        due = sorted(
            (ts, label) for label, ts in (rec.get("due") or {}).items() if ts <= t
        )
        if due:
            reasons.append(
                "due: "
                + ", ".join(label for _, label in due[:6])
                + (f" and {len(due) - 6} more" if len(due) > 6 else "")
            )
    return Decision(not reasons, reasons, None if errors else values, secs, t)


# --- after a tick -----------------------------------------------------------------------------------------------------


def _stat(st: dict, kind: str, secs: float, at: str) -> None:
    s = st.setdefault(STATS, {})
    s["recent"] = [
        r for r in s.get("recent") or [] if minutes_since(r[0], at) < STATS_WINDOW_MIN
    ]
    s["recent"].append([at, kind, round(secs, 1)])


def record_full(
    cfg: dict,
    store,
    d: Decision,
    secs: float,
    parts: dict | None = None,
    timers: dict | None = None,
) -> dict:
    """After a full tick, under its lock: record the fingerprint it started from (its end parts re-read now), the
    deadlines its state leaves, and its time. Returns the record."""
    parts = PARTS if parts is None else parts
    st = store.load()
    fp = None
    if d.fp is not None:
        fp = dict(d.fp)
        for p in parts.values():
            if not p.end:
                continue
            try:
                v = p.read(cfg, st, store)
            except (
                Exception
            ):  # noqa: BLE001 - not on record: the next tick runs in full
                fp = None
                break
            fp[p.name] = p.merge(d.fp.get(p.name), v) if p.merge else v
    t = now()
    st[KEY] = {
        "fp": fp,
        "due": deadlines(cfg, st, store, timers, t),
        "full_at": d.at,
        "why": d.reasons[:8],
    }
    _stat(st, "full", secs, t)
    store.save(st)
    return st[KEY]


def idle_tick(cfg: dict, st: dict, store, secs_so_far: float) -> list[str]:
    """An idle tick, under the lock: the "checked" stamps, the tick's record, and its line (and a stamp's failure)."""
    t0 = time.monotonic()
    t = now()
    notes = []
    for name, fn in STAMPS.items():
        try:
            notes += fn(cfg, store, t) or []
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a stamp is advisory: it never fails the tick
            notes.append(
                f"({name} stamp not refreshed: {' '.join(str(exc).split())[:160]})"
            )
    st["last_tick"] = t
    _stat(st, "idle", secs_so_far + time.monotonic() - t0, t)
    store.save(st)
    return [idle_line(st)] + notes


def idle_line(st: dict) -> str:
    rec = st.get(KEY) or {}
    n = sum(1 for r in (st.get("prs") or {}).values() if r.get("status") == "open")
    since = (rec.get("full_at") or "?" * 16)[11:16]
    sha = ((st.get("develop") or {}).get("sha") or "?")[:10]
    return f"TICK idle: no change since {since}Z ({n} PRs, develop {sha})"


# --- reports ----------------------------------------------------------------------------------------------------------


def counts(st: dict, at: str | None = None) -> dict:
    """{kind: (ticks, average seconds)} over the last day."""
    t = at or now()
    rows = [
        r
        for r in (st.get(STATS) or {}).get("recent") or []
        if minutes_since(r[0], t) < STATS_WINDOW_MIN
    ]
    out = {}
    for kind in ("idle", "full"):
        s = [r[2] for r in rows if r[1] == kind]
        out[kind] = (len(s), sum(s) / len(s) if s else 0.0)
    return out


def stats_text(st: dict) -> str:
    """The board's Monitor row: "ticks in 24 h: 120 idle (avg 6 s), 24 full (avg 131 s)"; "" before the first."""
    c = counts(st)
    if not c["idle"][0] and not c["full"][0]:
        return ""
    return "ticks in 24 h: " + ", ".join(
        f"{n} {k} (avg {a:.0f} s)" for k, (n, a) in c.items()
    )


def status_lines(cfg: dict, st: dict) -> list[str]:
    """The last full tick and why it ran, what makes the next one full, and the day's tick counts."""
    rec = st.get(KEY) or {}
    due = sorted((ts, label) for label, ts in (rec.get("due") or {}).items())
    every = settings(cfg)["idle_full_tick_min"]
    nxt = [f"{label} at {ts[11:16]}Z" for ts, label in due[:5]] + (
        [f"and {len(due) - 5} more"] if len(due) > 5 else []
    )
    if rec.get("full_at") and every:
        nxt.append(
            f"the {every:g}-min full tick at {plus(rec['full_at'], every)[11:16]}Z"
        )
    when = lambda ts: f"{ts[:16]}Z" if ts else "-"  # noqa: E731
    return [
        f"last tick {when(st.get('last_tick'))}; last full tick {when(rec.get('full_at'))}"
        + (f" ({'; '.join(rec.get('why') or [])})" if rec.get("why") else ""),
        "the next full tick: any change, or "
        + (", ".join(nxt) or "-")
        + (
            ""
            if rec.get("fp")
            else " (no fingerprint on record: the next tick is full)"
        ),
        stats_text(st) or "no tick recorded in the last 24 h",
    ]


# --- tools/tick_loop.py -----------------------------------------------------------------------------------------------


def loop_config(cfg: dict, store) -> list[str]:
    """`rk loop-config`: the loop's settings in seconds and bytes, and its files, as shell assignments."""
    s = settings(cfg)
    vals = {
        "TICK_INTERVAL_S": round(s["tick_interval_min"] * 60),
        "LOOP_HEARTBEAT_S": round(s["loop_heartbeat_min"] * 60),
        "TICK_TIMEOUT_S": round(s["tick_timeout_min"] * 60),
        "LOOP_LOG_MAX_BYTES": round(s["loop_log_max_mb"] * 2**20),
    }
    vals.update({k: str(store.dir / name) for k, name in LOOP_FILES.items()})
    return [f"{k}={shlex.quote(str(v))}" for k, v in vals.items()]


def loop_pid(store) -> int | None:
    """The running tick loop's pid (0 when its pid file is unreadable), or None when no loop holds its lock."""
    lock = store.dir / LOOP_FILES["LOOP_LOCK"]
    if not lock.exists():
        return None
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            return int((store.dir / LOOP_FILES["LOOP_PID"]).read_text().strip())
        except (OSError, ValueError):
            return 0
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)
