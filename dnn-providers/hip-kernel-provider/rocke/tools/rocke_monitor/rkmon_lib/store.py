# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Configuration, on-disk state, journal and the outward-action ledger.

State is one JSON document written whole (state_store.write_file) under an exclusive lock, so an interrupted
tick never leaves a half-written state, and two ticks never run at once.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import pathlib

from . import config, registry, state_store

ROOT = pathlib.Path(__file__).resolve().parent.parent  # the toolkit's code (read-only)
SCHEMA = 1


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def minutes_since(s: str, ref: str | None = None) -> float:
    end = parse_ts(ref) if ref else dt.datetime.now(dt.timezone.utc)
    return (end - parse_ts(s)).total_seconds() / 60.0


def head_age_min(rec: dict) -> float:
    """Minutes since PR `rec`'s head was pushed: since its commit's committer date, or since a tick first saw the head
    when that is earlier. A late tick never restarts the settle (after a gap in the ticks every head would look fresh;
    lessons L-518)."""
    ages = [minutes_since(rec.get("head_seen_at") or now())]
    committed = rec.get("head_committed_at")
    if committed:
        try:
            at = dt.datetime.fromisoformat(committed.replace("Z", "+00:00"))
            ages.append((dt.datetime.now(dt.timezone.utc) - at).total_seconds() / 60.0)
        except ValueError:
            pass
    return max(ages)


# A PR its author keeps pushing: BURST_PUSHES or more pushes in BURST_WINDOW_MIN minutes settle until config
# `settle_burst_min` minutes after the last one, counted from the ticks that saw the pushes, never from commit dates
# (lessons L-534: a review started between pushes dies with the next push, and its worker and draft with it).
BURST_PUSHES, BURST_WINDOW_MIN = 2, 90


def recent_pushes(rec: dict) -> list[float]:
    """Minutes since each push of PR `rec` that a tick saw in the last BURST_WINDOW_MIN minutes, newest first
    (rec["push_times"], kept by events.diff_and_apply)."""
    ages = []
    for ts in rec.get("push_times") or []:
        try:
            age = minutes_since(ts)
        except ValueError:
            continue
        if age <= BURST_WINDOW_MIN:
            ages.append(age)
    return sorted(ages)


def settle_left(cfg: dict, st: dict, rec: dict) -> tuple[float, str]:
    """Minutes until PR `rec`'s head has settled (0 once it has), and what it waits for. One rule for the plan, the
    review driver and `rk review-start`: a head is reviewed once it is `settle_min` old (head_age_min); a develop fix
    or a PR the user floated to the top after `settle_develop_fix_min` minutes at most, bursts included; a burst of
    pushes `settle_burst_min` minutes after its last push. A key the configuration lacks has the registry's default.
    """
    age = head_age_min(rec)
    why = f"head is {age:.0f} min old"
    if (st.get("develop_fixes") or {}).get(str(rec["n"])):
        most = cfg.get(
            "settle_develop_fix_min", registry.default("settle_develop_fix_min")
        )
        return max(0.0, min(cfg["settle_min"], most) - age), why
    left = cfg["settle_min"] - age
    pushes = recent_pushes(rec)
    burst = cfg.get("settle_burst_min", registry.default("settle_burst_min"))
    if len(pushes) >= BURST_PUSHES and burst - pushes[0] > left:
        left = burst - pushes[0]
        why = (
            f"{len(pushes)} pushes in {BURST_WINDOW_MIN} min; it settles {burst} min after the last, "
            f"{pushes[0]:.0f} min ago"
        )
    return max(0.0, left), why


BOOT_ID = pathlib.Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> str:
    try:
        return BOOT_ID.read_text().strip()
    except OSError:
        return ""


def stopped(st: dict) -> dict | None:
    """The stop in force (`rk stop` or `rk kill`), or None. A stop ends at a reboot, as the monitor starts again then
    (the container's start runs `rk start`); `rk pause` keeps it from acting.
    """
    s = st.get("stopped")
    if not s:
        return None
    if s.get("boot") and boot_id() and s["boot"] != boot_id():
        return None
    return s


def proc_start(pid: int) -> int | None:
    """Process `pid`'s start time (clock ticks since boot, /proc/<pid>/stat field 22), None when it is gone or a
    zombie: a pid and its start time name one process, never a later one that reuses the pid.
    """
    try:
        raw = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    try:
        return None if not fields or fields[0] == "Z" else int(fields[19])
    except (IndexError, ValueError):
        return None


def draining(st: dict) -> dict | None:
    """A drain in progress (`rk drain`, `rk stop --drain`): nothing new starts while the work in flight finishes."""
    return st.get("draining") or None


# A drain past its `until_min` (the wait's limit) by this much is stale: what follows the wait (an apply's code install,
# restart, load and doctor) fits well within it; a drain whose process is gone is stale at once.
DRAIN_MARGIN_MIN = 60


def drain_stale(d: dict) -> str:
    """Why a recorded drain no longer holds ("" while it does): the process that set it is gone (an `rk drain` or
    `rk stop --drain` killed or interrupted before it could clear it), or it is past its `until_min` and
    DRAIN_MARGIN_MIN."""
    pid = d.get("pid")
    if pid and proc_start(int(pid)) != d.get("pstart"):
        return f"the process that set it ({pid}) is gone"
    until, since = d.get("until_min"), d.get("since")
    if until is not None and since:
        try:
            if minutes_since(since) > float(until) + DRAIN_MARGIN_MIN:
                return f"it is past its {float(until):g} min"
        except ValueError:
            pass
    return ""


def drain_record(why: str, until_min: float) -> dict:
    """The `draining` record a drain sets: its reason, its time limit, and this process (pid and start time), so a
    drain whose process died holds nothing (drain_stale)."""
    return {
        "since": now(),
        "why": why,
        "until_min": until_min,
        "pid": os.getpid(),
        "pstart": proc_start(os.getpid()),
    }


def halted(st: dict) -> str:
    """Why nothing new may start: "" when it may; else a few words naming the stop or the drain. A stale drain
    (drain_stale) halts nothing: the tick clears it."""
    s = stopped(st)
    if s:
        return f"the monitor is stopped (rk {s.get('how') or 'stop'}, {s.get('at')}); rk start resumes it"
    d = draining(st)
    if d and not drain_stale(d):
        return (
            f"the monitor is draining ({d.get('why') or 'rk stop --drain'}, since {d.get('since')}): nothing new "
            "starts"
        )
    return ""


_WARNED: set[str] = set()


def _warn_once(text: str) -> None:
    """A warning on stderr, once per process (a reader that runs many times in a tick says it once)."""
    import sys

    if text not in _WARNED:
        _WARNED.add(text)
        print(text, file=sys.stderr)


def load_config(path: pathlib.Path | None = None) -> dict:
    """config.json, normalized and checked (config.load): a blank required key or a bad value is refused by name."""
    return config.load(path)


def configured_paths(cfg: dict | None = None) -> dict:
    """Every path by name (paths.resolve): root (the working folder), tmp, scratch, the outputs folder and the files in
    it (daily_dir, html, insights_dir, posted_dir, spend_dir), posts and the others, from `cfg`, or from the monitor
    home's config.json when `cfg` is None (read without checks, so importing a module never fails on it; "" for a path
    nothing configures)."""
    from . import paths

    if cfg is None:
        try:
            cfg = config.load(check=False)
        except (config.ConfigError, OSError, ValueError):
            cfg = {}
    return paths.resolve(cfg)


def flock_within(fd: int, timeout: float, what: str) -> None:
    """An exclusive flock on fd within `timeout` seconds, else TimeoutError naming `what`."""
    import time

    end = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= end:
                raise TimeoutError(
                    f"{what} is still held after {timeout:.0f} s"
                ) from None
            time.sleep(0.1)


def empty_state() -> dict:
    return {
        "schema": SCHEMA,
        "enabled": False,
        "frozen": None,
        "cleared": [],
        "created": now(),
        "last_tick": None,
        "develop": {
            "sha": None,
            "seen_at": None,
            "last_green": None,
            "ci": {},
            "pending_sanity": [],
        },
        "prs": {},
        "jobs": {"heavy": None, "worker": None, "history": []},
        "incidents": {},
        "queue": [],
        "counters": {"posts": []},
    }


class Store:
    """Owns state/ (state.json, journal.md, actions.jsonl, lock): in the monitor's home by default (config.state_dir)."""

    def __init__(self, state_dir: pathlib.Path | None = None):
        self.dir = state_dir or config.state_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        for sub in ("jobs", "gates", "incidents"):
            (self.dir / sub).mkdir(exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.journal_path = self.dir / "journal.md"
        self.ledger_path = self.dir / "actions.jsonl"

    @contextlib.contextmanager
    def locked(self, wait: bool = False, timeout: float | None = None):
        """Exclusive lock for a whole tick; raises BlockingIOError if another tick holds it and wait is False. With
        `timeout`, waits at most that many seconds, then raises TimeoutError."""
        fd = os.open(self.dir / "lock", os.O_CREAT | os.O_RDWR, 0o664)
        try:
            if timeout is None:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            else:
                flock_within(fd, timeout, "the state lock")
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def load(self) -> dict:
        if not self.state_path.exists():
            return empty_state()
        st = json.loads(self.state_path.read_text())
        if st.get("schema") != SCHEMA:
            raise ValueError(
                f"state schema {st.get('schema')} != {SCHEMA}; migrate before running"
            )
        return st

    def save(self, st: dict) -> None:
        """The state ST as state.json, written whole and durable (state_store.write_file: at the mode of the files the
        monitor and the writer share), so a crash or a power loss leaves the old state or the new one, never an empty
        file."""
        state_store.write_file(self.state_path, st)

    def journal(self, kind: str, text: str) -> None:
        with self.journal_path.open("a") as f:
            f.write(f"{now()} {kind:<8} {text}\n")

    def journal_tail(self, n: int = 15) -> list[str]:
        """The journal's last `n` lines, read from its end (the journal grows to retention.log_max_mb); just after a
        rotation the rotated journal's last lines come first."""
        from .retention import with_older

        out: list[str] = []
        for path in reversed(with_older(self.journal_path)):
            with path.open("rb") as f:
                size = f.seek(0, os.SEEK_END)
                block, data = 8192, b""
                while size > 0 and data.count(b"\n") <= n:
                    step = min(block, size)
                    size -= step
                    f.seek(size)
                    data = f.read(step) + data
                    block *= 2
            out = data.decode("utf-8", errors="replace").splitlines() + out
            if len(out) >= n:
                break
        return out[-n:] if n else []

    def journal_lines(self) -> list[str]:
        """Every journal line still kept: the rotated journal (retention.py) first, then the current one."""
        from .retention import with_older

        return [
            ln
            for p in with_older(self.journal_path)
            for ln in p.read_text(errors="replace").splitlines()
        ]

    # The ledger records every outward action. Wrappers check it first, so nothing is done twice.
    def ledger_add(
        self, kind: str, pr: int, head: str, ref: str, note: str = ""
    ) -> dict:
        """Append one record. A torn last line (a write cut short: a full disk, a kill) is moved to actions.torn first,
        so the new record starts a line of its own and the ledger stays readable."""
        rec = {
            "ts": now(),
            "kind": kind,
            "pr": pr,
            "head": head,
            "ref": ref,
            "note": note,
        }
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.ledger_path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o664)
        try:
            # the writers' own lock: a repair never cuts another writer's line
            fcntl.flock(fd, fcntl.LOCK_EX)
            size = os.lseek(fd, 0, os.SEEK_END)
            if size and os.pread(fd, 1, size - 1) != b"\n":
                self._cut_torn(fd, size)
            os.write(fd, line)
        finally:
            os.close(fd)
        return rec

    def _cut_torn(self, fd: int, size: int) -> None:
        start = max(0, size - 65536)
        tail = os.pread(fd, size - start, start)
        cut = start + tail.rfind(b"\n") + 1 if b"\n" in tail else start
        with (self.dir / "actions.torn").open("ab") as f:
            f.write(tail[cut - start :] + b"\n")
        os.ftruncate(fd, cut)
        self.journal(
            "ledger",
            f"a torn last line of {self.ledger_path.name} ({size - cut} bytes) was moved to actions.torn",
        )

    def ledger(self) -> list[dict]:
        """Every record. A last line that does not parse is a write cut short (or one still being written): it is
        skipped, with a warning; a bad line anywhere else still raises."""
        if not self.ledger_path.exists():
            return []
        lines = [ln for ln in self.ledger_path.read_text().splitlines() if ln.strip()]
        out = [json.loads(ln) for ln in lines[:-1]]
        if lines:
            try:
                out.append(json.loads(lines[-1]))
            except ValueError:
                _warn_once(
                    f"warning: the last line of {self.ledger_path} is torn ({len(lines[-1])} characters); "
                    "it is skipped, and the next record moves it to actions.torn"
                )
        return out

    # Structured history for the weekly statistics: every event, and one metrics snapshot per tick.
    def append_jsonl(self, name: str, records: list[dict]) -> None:
        if not records:
            return
        with (self.dir / name).open("a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def read_jsonl(self, name: str, lenient: bool = False) -> list[dict]:
        """`lenient`: skip a line that does not parse (a write cut short), so an advisory history cannot take the
        dashboard down. A rotated log's older part (`<name>.1`, retention.py) is read first.
        """
        from .retention import with_older

        out = []
        for line in [
            ln for p in with_older(self.dir / name) for ln in p.read_text().splitlines()
        ]:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                if not lenient:
                    raise
        return out

    def ledger_has(self, kind: str, pr: int, head: str) -> bool:
        return any(
            r["kind"] == kind and r["pr"] == pr and r["head"] == head
            for r in self.ledger()
        )
