# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GitHub request cap: the deployment's container makes at most
integrations.github_cli.max_requests_per_hour requests to GitHub in any hour (the configuration's
gh.max_requests_per_hour; 0 is no cap), the monitor's, its Claude session's and its writer's alike.

The counter, docker/rk_gh.py, is the first `gh` on every PATH in the container (container_layout.GH_COUNTER_DIR): it
counts each gh call that asks GitHub (`asks_github`) in RECORD, then runs the real gh. A call over the cap waits until
the last hour has room, then counts. A PR runner has no GitHub login and no monitor home (RKMON_HOME), and is not
counted; nor is a call while there is no cap, or one whose record cannot be read or written (an OSError): it runs as it
would without the counter, and the counter says so (`unusable`). A call that pages through results (gh api --paginate, a
list whose --limit takes more than one page) counts once, though each page is a request to GitHub: the counter sees one
gh call (the cap key's description asks the operator to leave headroom for it). A wait is open in the record while it
lasts, so that a call stopped while it waited (a tick stopped at tick_timeout_min, an interrupted command) is told too:
its open wait turns into a wait of a stopped call once its call has not looked at the record for STOPPED_SECONDS (a
waiting call looks at least every LOOK_SECONDS).

- RECORD, in the monitor's state folder (`state_folder`: $RKMON_HOME/<paths.STATE>), which the monitor and the writer
  share (worker/perms.py SHARED_FILE_MODE), changed under LOCK beside it (perms.shared_lock):
  {"cap_per_hour": the cap the configuration last gave it, "requests_by_minute": {minute: requests} for the minutes
  of the last hour (a minute is the seconds since the epoch over 60), "waits": [{"at": when a call that waited went
  on, "waited_seconds": how long it waited}, or for a call stopped while it waited {"at": when the record found it
  stopped, "waited_seconds": up to its last look, "stopped": true}] for the last day, "open_waits": {a key of each
  call's own: {"since": when the call began to wait, "looked": when it last looked at the record}} for the calls that
  wait, "reported_until": the time up to which the day log has the waits, once a full tick told it some}, times in
  seconds since the epoch. A file that is no such record reads as a new one, with no cap. Each write of the record
  turns the open waits of stopped calls into their waits.
- `asks_github(args) -> bool`: whether the gh call ARGS asks GitHub: every call but gh's own help, version,
  completion, configuration and aliases, and the auth commands that print or install the login gh holds (LOCAL,
  LOCAL_AUTH, HELP).
- `state_folder(environ=None) -> folder | None`: the state folder of the monitor's home, RKMON_HOME; None without one.
- `count(args, environ=None, clock=time.time, sleep=time.sleep, say=on_stderr)`: the counter's step: ARGS counted,
  after waiting while the last hour has no room; SAY, when given, is told once, as the wait begins, and when the
  record cannot be read or written (`unusable`).
- `wait_for_room(args, environ=None, clock=time.time, sleep=time.sleep, say=on_stderr) -> seconds`: gh.py's step
  before it starts gh: the same wait, said the same way, counting nothing (the counter counts), so that gh.py's time
  limits and retries run on gh's own time. The seconds it waited.
- `on_stderr(text)`: how the counter and gh.py say a wait: a line of gh's on standard error ("gh: waiting until
  HH:MM:SSZ for the GitHub request cap (SETTING)"), which the tick filter knows.
- `set_cap(state, cap, clock=time.time) -> [line]`: CAP as the record's cap in the state folder STATE: what the
  counter reads at each request. Each tick gives it the configuration's CONFIG_KEY (rkmon.py cmd_tick), and it writes
  the record each time while there is a cap (or the record holds one), so that a record that cannot be written shows
  at each tick: then a GH_CAP_OFF line (`unusable`), which the tick filter knows.
- `unusable(state, exc) -> text`: a record that cannot be read or written (EXC): no call is counted; the record, the
  reason and the fix.
- `waits(state, since, clock=time.time) -> [wait]`: the record's waits at SINCE or later, and the open waits of the
  calls that still wait and looked at the record since then ({"at": their last look, "waited_seconds": so far,
  "waiting": true}): what a tick's GH_CAP_WAIT line tells (rkmon.py cmd_tick); none when the record cannot be read.
- `report(state, until) -> [wait]`: the record's waits after its reported_until, up to UNTIL, which becomes its
  reported_until, the waits of the calls found stopped included: what a full tick tells the day log (rkmon.py
  gh_cap_log), whose entries the admin dashboard shows. An open wait is told once it ends.
- `waiting_text(state, now) -> text`: the calls that wait now, as the clause the tick loop's stop line ends with
  (tools/tick_loop.py); "" when no call waits, or the record cannot be read.
- `wait_text(waits, cap) -> text`: how many gh calls waited, how long in all, how many still wait or were stopped
  while they waited, and for which cap.

Standard library only: the counter loads it by its path from the image's code under python3 -I (docker/rk_gh.py),
and it loads the package's files it uses by their paths from its own code (CODE): rkmon_lib/paths.py and
rkmon_lib/whole_file.py, and worker/perms.py through paths.worker_module.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
import runpy
import sys
import time
import types

CODE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # this file's code
paths = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "paths.py"))
)
whole_file = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "whole_file.py"))
)

RECORD, LOCK = "gh_requests.json", "gh_requests.lock"
MINUTE_SECONDS, HOUR_MINUTES, DAY_SECONDS = 60, 60, 86400
# The longest sleep between two looks at the record while a call waits: another process may count, or the cap change,
# in the meantime.
LOOK_SECONDS = 60
# How long an open wait's call may go without looking at the record before the record takes it as stopped.
STOPPED_SECONDS = 3 * LOOK_SECONDS
# gh's commands that ask GitHub nothing: its help and version, its shell completion, its own configuration and
# aliases; the auth commands that print or install the login gh holds (git's credential helper runs one per fetch);
# and any command asked for its help.
LOCAL = ("--version", "version", "help", "completion", "config", "alias")
LOCAL_AUTH = ("token", "setup-git", "git-credential")
HELP = ("--help", "-h")
SETTING = "integrations.github_cli.max_requests_per_hour"
CONFIG_KEY = "gh.max_requests_per_hour"  # SETTING's configuration key


def _perms():
    return paths.worker_module("perms")


def asks_github(args: list[str]) -> bool:
    """Whether the gh call ARGS asks GitHub: see the module's docstring."""
    if not args or args[0] in LOCAL or any(a in HELP for a in args):
        return False
    return not (args[0] == "auth" and args[1:2] and args[1] in LOCAL_AUTH)


def on_stderr(text: str) -> None:
    """How the counter and gh.py say a wait: see the module's docstring."""
    print(f"gh: {text}", file=sys.stderr, flush=True)


def state_folder(environ=None) -> pathlib.Path | None:
    """The state folder of the monitor's home, RKMON_HOME; None without one."""
    home = (os.environ if environ is None else environ).get("RKMON_HOME", "").strip()
    return pathlib.Path(os.path.expanduser(home)) / paths.STATE if home else None


# --- the record -------------------------------------------------------------------------------------------------------


def _new() -> dict:
    return {"cap_per_hour": 0, "requests_by_minute": {}, "waits": [], "open_waits": {}}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _is_record(data) -> bool:
    try:
        return (
            isinstance(data["cap_per_hour"], int)
            and _number(data["cap_per_hour"])
            and all(
                k.isdigit() and isinstance(n, int) and _number(n)
                for k, n in data["requests_by_minute"].items()
            )
            and all(
                _number(w["at"]) and _number(w["waited_seconds"]) for w in data["waits"]
            )
            and all(
                _number(w["since"]) and _number(w["looked"])
                for w in data["open_waits"].values()
            )
            and _number(data.get("reported_until", 0))
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _read(state: pathlib.Path) -> dict:
    """The record in the state folder STATE; a new one when there is none, or none that reads as a record. An OSError
    when it cannot be read."""
    try:
        data = json.loads((state / RECORD).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _new()
    except ValueError:  # no JSON, or no UTF-8
        return _new()
    return data if _is_record(data) else _new()


def _stopped(record: dict, now: float) -> list[str]:
    """The keys of RECORD's open waits whose calls were stopped: not looked at for STOPPED_SECONDS by NOW."""
    return [
        key
        for key, w in record["open_waits"].items()
        if w["looked"] < now - STOPPED_SECONDS
    ]


def _close(record: dict, key: str, now: float, stopped: bool) -> None:
    """The open wait KEY of RECORD, as a wait that ended at NOW: its call goes on, or was STOPPED (it waited up to its
    last look)."""
    w = record["open_waits"].pop(key)
    wait = {"at": now, "waited_seconds": round(now - w["since"], 1)}
    if stopped:
        wait.update(waited_seconds=round(w["looked"] - w["since"], 1), stopped=True)
    record["waits"].append(wait)


def _write(state: pathlib.Path, record: dict, now: float) -> None:
    """RECORD as the record in STATE, without the minutes before the last hour and the waits before the last day, the
    open waits of stopped calls as their waits."""
    for key in _stopped(record, now):
        _close(record, key, now, stopped=True)
    minute = int(now // MINUTE_SECONDS)
    record["requests_by_minute"] = {
        k: n
        for k, n in record["requests_by_minute"].items()
        if int(k) > minute - HOUR_MINUTES
    }
    record["waits"] = [w for w in record["waits"] if w["at"] > now - DAY_SECONDS]
    text = json.dumps(record, sort_keys=True) + "\n"
    whole_file.write(state / RECORD, text, _perms().SHARED_FILE_MODE)


def _room_at(record: dict, now: float) -> float:
    """When the last hour has room for one more request under RECORD's cap: NOW when it has room, or there is no cap;
    else when enough of its oldest minutes have left it."""
    cap = record["cap_per_hour"]
    minute = int(now // MINUTE_SECONDS)
    live = sorted(
        (int(k), n)
        for k, n in record["requests_by_minute"].items()
        if minute - HOUR_MINUTES < int(k) <= minute
    )
    # The requests that must leave the last hour before one more fits.
    over = sum(n for _, n in live) - cap + 1
    if not cap or over <= 0:
        return now
    for m, n in live:
        over -= n
        if over <= 0:
            return float((m + HOUR_MINUTES) * MINUTE_SECONDS)
    return now


def _hold(state: pathlib.Path, counts: bool, clock, sleep, say) -> float:
    """Wait while the last hour has no room under the record's cap, the wait open in the record, then count one request
    when COUNTS: see the module's docstring. The seconds waited, which the record keeps.
    """
    if not _read(state)["cap_per_hour"]:
        return 0.0  # no cap: nothing to count, and no lock to take
    started, key = clock(), ""
    try:
        while True:
            with _perms().shared_lock(state / LOCK):
                record, now = _read(state), clock()
                room = _room_at(record, now)
                if room <= now:
                    counted = counts and record["cap_per_hour"] > 0
                    if counted:
                        minute = str(int(now // MINUTE_SECONDS))
                        by_minute = record["requests_by_minute"]
                        by_minute[minute] = by_minute.get(minute, 0) + 1
                    if key in record["open_waits"]:
                        _close(record, key, now, stopped=False)
                    if counted or key:
                        _write(state, record, now)
                    return now - started if key else 0.0
                first, key = not key, key or os.urandom(6).hex()
                record["open_waits"][key] = {"since": started, "looked": now}
                _write(state, record, now)
            if first and say:
                when = time.strftime("%H:%M:%SZ", time.gmtime(room))
                say(f"waiting until {when} for the GitHub request cap ({SETTING})")
            sleep(min(room - now, LOOK_SECONDS))
    except BaseException:
        if key:  # stopped while it waited (an interrupt, an exit): its wait ends here
            _stop(state, key, clock())
        raise


def _stop(state: pathlib.Path, key: str, now: float) -> None:
    """The open wait KEY, as the wait of a call stopped at NOW, as far as the record can be written."""
    try:
        with _perms().shared_lock(state / LOCK):
            record = _read(state)
            if key in record["open_waits"]:
                record["open_waits"][key]["looked"] = now
                _close(record, key, now, stopped=True)
                _write(state, record, now)
    except OSError:
        pass


def unusable(state: pathlib.Path, exc: OSError) -> str:
    """A record that cannot be read or written: see the module's docstring."""
    reason = exc.strerror or errno.errorcode.get(exc.errno or 0) or str(exc)
    return (
        f"the GitHub request cap ({SETTING}) counts no gh call: its record {state / RECORD} cannot be read or "
        f"written ({reason}). A restart of the container gives the monitor's state folder its permissions again"
    )


def count(
    args: list[str], environ=None, clock=time.time, sleep=time.sleep, say=on_stderr
) -> None:
    """The counter's step: see the module's docstring."""
    state = state_folder(environ)
    if state is None or not asks_github(args):
        return
    try:
        _hold(state, True, clock, sleep, say)
    except OSError as exc:  # the call runs uncounted, and says so
        if say:
            say(unusable(state, exc))


def wait_for_room(
    args: list[str], environ=None, clock=time.time, sleep=time.sleep, say=on_stderr
) -> float:
    """gh.py's step before it starts gh: see the module's docstring."""
    state = state_folder(environ)
    if state is None or not asks_github(args):
        return 0.0
    try:
        return _hold(state, False, clock, sleep, say)
    except OSError:
        return 0.0


def set_cap(state: pathlib.Path, cap: int, clock=time.time) -> list[str]:
    """CAP as the record's cap, written while there is one: see the module's docstring."""
    try:
        if cap == _read(state)["cap_per_hour"] == 0:
            return []  # no cap, and none in the record: nothing to keep
        with _perms().shared_lock(state / LOCK):
            record = _read(state)
            record["cap_per_hour"] = cap
            _write(state, record, clock())
    except OSError as exc:
        return [f"GH_CAP_OFF: {unusable(state, exc)}"]
    return []


def waits(state: pathlib.Path, since: float, clock=time.time) -> list[dict]:
    """The record's waits at SINCE or later, and the open waits looked at since then: see the module's docstring."""
    try:
        record, now = _read(state), clock()
    except OSError:
        return []
    stopped = _stopped(record, now)
    return [w for w in record["waits"] if w["at"] >= since] + [
        {
            "at": w["looked"],
            "waited_seconds": round(now - w["since"], 1),
            "waiting": True,
        }
        for key, w in record["open_waits"].items()
        if w["looked"] >= since and key not in stopped
    ]


def report(state: pathlib.Path, until: float) -> list[dict]:
    """The record's waits the day log does not have, up to UNTIL, marked as had: see the module's docstring."""

    def untold(record: dict) -> list[dict]:
        after = record.get("reported_until", 0)
        return [w for w in record["waits"] if after < w["at"] <= until]

    try:
        record = _read(state)
        if not untold(record) and not _stopped(record, until):
            return []  # nothing to tell: no lock, and no file written
        with _perms().shared_lock(state / LOCK):
            record = _read(state)
            for key in _stopped(record, until):
                _close(record, key, until, stopped=True)
            told = untold(record)
            record["reported_until"] = until
            _write(state, record, until)
    except OSError:
        return []
    return told


def waiting_text(state: pathlib.Path, now: float) -> str:
    """The calls that wait now, as the tick loop's clause: see the module's docstring."""
    try:
        record = _read(state)
    except OSError:
        return ""
    stopped = _stopped(record, now)
    since = [w["since"] for k, w in record["open_waits"].items() if k not in stopped]
    if not since:
        return ""
    calls = f"{len(since)} gh call{'' if len(since) == 1 else 's'}"
    first = time.strftime("%H:%MZ", time.gmtime(min(since)))
    return f"while {calls} waited for the GitHub request cap ({SETTING}), the first since {first}"


def wait_text(waits: list[dict], cap: int) -> str:
    """How many gh calls waited, how long in all, how many still wait or were stopped, and for which cap: see the
    module's docstring."""
    seconds = sum(w["waited_seconds"] for w in waits)
    took = (
        f"{seconds / MINUTE_SECONDS:.0f} minutes"
        if seconds >= 2 * MINUTE_SECONDS
        else f"{seconds:.0f} seconds"
    )
    calls = f"{len(waits)} gh call{'' if len(waits) == 1 else 's'}"
    # A lone call's state, or how many of the calls are in it.
    notes = [
        what if len(waits) == 1 else f"{n} of them {what}"
        for n, what in (
            (sum(1 for w in waits if w.get("waiting")), "still waiting"),
            (sum(1 for w in waits if w.get("stopped")), "stopped while waiting"),
        )
        if n
    ]
    states = f", {' and '.join(notes)}," if notes else ""
    return (
        f"{calls} waited {took} in all{states} for the cap of {cap} GitHub requests an "
        f"hour ({SETTING})"
    )
