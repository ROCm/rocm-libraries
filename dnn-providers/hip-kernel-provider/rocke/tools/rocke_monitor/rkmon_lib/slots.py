# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PR runners' slots: which job holds which slot, and the one way to take, keep and
free one. rkmon_lib/prcode.py runs PR code in them; docker/rk_prcode.py (the launcher) works in them as the slot's user.

The container has one PR runner per slot (container_layout.PR_RUNNERS, SLOTS of them). A slot's folder,
SLOTS_DIR/<runner>, is the runner's, group rkmon, mode 2751; the container's start makes it (docker/rk_layout.py). When a
job takes the slot, the launcher stops every process of the slot's user, empties the folder and makes the four of
SLOT_FOLDERS in it, fresh: `out` (the job's results), `build`, `tmp` and `home`. SLOT_INPUTS/<runner> holds what the
monitor hands the job to read.

The records: one file per slot that is not free, `<runner>.json` in SLOT_LOCKS (the monitor's folder: no runner reads
it). Every change is made under one lock, SLOT_LOCKS/take.lock (`Records.lock`).
- taken by a process (a job script, one command): {"state": "taken", "pid", "start", "boot", "job", "pr", "since"}.
  The slot is free again once that process is gone: its pid names no process, or another one (its start time, or the
  machine's boot, differs). What the process left is the next job's start's to empty.
- taken by named work (`rk pr-run --slot NAME`, NAME pr<N>-<word>: NAME_RE): {"state": "taken", "name", "pr",
  "head", "owner"?, "since"}, until the work ends (`rk pr-run --slot NAME --end`), its review ends, or its owner is
  gone: `head` is the head of PR N's review in flight when the work took the slot ("" when none was), `owner` the
  process that started the work (`Records.owner_of`: the session or the agent whose shell ran rk; none when only
  shells lead up to the container's first process).
- kept (the checks job's slot, kept for its review: its build is the review's shared build): {"state": "kept", "pr",
  "head", "build", "record", "since"}. No job runs in it; the other runners read its `build` folder and nothing else of
  it (the launcher's `keep`). `record` is the file that names the shared build (review.SHARED_BUILD): its first line
  is `build` (shared_build_lines).
- freeing: {"state": "freeing", "pid", "start", "boot", "since"}: the slot is being emptied, by that process.
- broken: {"state": "broken", "init", "boot", "why", "since"}: PR code changed the slot's folder (its owner, group or
  mode), which the launcher then refuses; or a signal ended the launcher's begin or end while processes of the runner
  still ran (a process of the runner can end every process of the runner's, the launcher's calls included, so only a
  restart of the container stops them: `why` says so). No job takes it until the container's next start makes the
  folder again (a new start of PID 1: `init` is its start time). `rk pr-slot status` and `rk doctor` show it.

- `take`: the first free slot; when none is, the kept slot kept longest (its record changed in place, the shared build
  record removed); when none is kept either, it waits, WAIT_S between tries, until one is free (none, when every slot
  is broken). Named work already holding a slot gets that slot back, as it is.
- `take_begun`: `take`, then the launcher's `begin` on a slot taken fresh (every process of its user stopped, its
  folder emptied, the four made): a slot it refuses is marked broken, its why the launcher's own last line; one
  whose `begin` a signal ended is marked broken while processes of its user still run, else let go, as one whose
  processes outlive SIGKILL is; and the next is taken.
- `free`: the record made `freeing` (a kept slot's shared build record removed first), the launcher's `end` (every
  process of the slot's user stopped, its folder emptied), then the record removed, or marked broken when a signal
  ended that `end` while processes of the slot's user still run. A record another process wrote meanwhile stays.
- `keep`: the record checked to be the taking of the process that keeps the slot, the launcher's `keep` (every
  process stopped, the slot's other folders made fresh, the build folder readable by the other runners), the record
  made kept; the other slots kept for the same PR are freed.
- `free_ended`: the kept slots and the named work whose review is no longer in flight, and the named work whose
  owner is gone, freed (each advance of the reviews: rkmon_lib/review.py end_slots).
- `begin`, `end`, `keep_call`: the launcher's begin, end and keep as the slot's user; with each begin and end the
  slot's inputs folder (SLOT_INPUTS/<runner>, the monitor's) is emptied too (`clear_inputs`): what the monitor hands
  one job is never the next one's. `lines_of`: what a call printed, then `failed`'s line when it failed.
- `slot_of(path)`: the runner whose slot's folder holds a path. `split(path)`: a path in a slot's folder as that folder
  and the path under it, the way the result reader (worker/pr_results.py) reads it.
- `kept_build(path)`: whether the build at `path` may still be reused as its review's shared build: one in no slot's
  folder (a host's build folder) always, one in a slot's folder while the slot is kept for it. A slot freed, or taken
  back for another job, holds that job's files: `rk dispatch-pack` (dispatch.shared_build), the session's guard
  (workguard.shared_paths) and the clash builds (clashes.compile_db) ask this before they name or read a shared build.
- `shared_build_lines(path)`: a shared build record's lines (review.SHARED_BUILD): the build folder, its tree and the
  tree's commit, one to a line, so a path with a space stays whole.
- `by_pr()`: each PR's slots now (taken by its job or its named work, or kept for its review): what the RAM disk's
  footprints count as the PR's (rkmon_lib/ramdisk.py slot_folders).
The launcher's calls (`call`) are the caller's: `free`, `keep` and `free_ended` take them as functions, so the
records' rules are tested without a launcher.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import time

from . import container_layout as L, paths, whole_file

# A record's states (its "state"; a slot with no record is free), as the module's docstring describes each.
TAKEN, KEPT, FREEING, BROKEN = "taken", "kept", "freeing", "broken"
# The fix of a slot whose runner's processes outlive every launcher call: the container's next start stops them.
RESTART = "restart the container"
RUNNER_VAR = "RKMON_PR_RUNNER"  # the image's launcher (docker/Dockerfile sets it)
LOCK = "take.lock"
WAIT_S = 5
LAUNCH_TIMEOUT_S = 1800  # the launcher's begin, end and keep: emptying a slot of a native build takes a while
NAME_RE = re.compile(r"^pr([1-9][0-9]*)-[a-z0-9][a-z0-9._-]{0,39}$")
# The commands that only lead to another (a shell, a wrapper): named work's owner is the first ancestor of rk that is
# none of these.
PASS_THROUGH = frozenset(
    {
        "sh",
        "bash",
        "dash",
        "zsh",
        "ksh",
        "env",
        "timeout",
        "nice",
        "ionice",
        "nohup",
        "setsid",
        "stdbuf",
        "setpriv",
        "sudo",
        "xargs",
        "flock",
        "rk",
    }
)


def launcher(environ=None) -> str:
    """The launcher's path when PR code runs in slots here (the image), else ""."""
    e = os.environ if environ is None else environ
    return e.get(RUNNER_VAR, "")


def folder(runner: str, name: str = "") -> str:
    """A slot's folder, or one of its SLOT_FOLDERS."""
    return f"{L.SLOTS_DIR}/{runner}" + (f"/{name}" if name else "")


def inputs(runner: str) -> str:
    """The folder of what the monitor hands the slot's job to read."""
    return f"{L.SLOT_INPUTS}/{runner}"


def call(
    runner: str,
    args: list,
    launcher_path: str,
    data: bytes | None = None,
    run=subprocess.run,
    timeout: float | None = LAUNCH_TIMEOUT_S,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """The launcher as the slot's user (`sudo -n -u RUNNER LAUNCHER ARGS...`), `data` on its stdin (else nothing), in
    the folder / (the monitor starts nothing in a folder a runner writes). Output captured as bytes
    unless `capture` is false; a timeout or a launch that fails reads as exit 124 or 127.
    """
    argv = ["sudo", "-n", "-u", runner, launcher_path, *[str(a) for a in args]]
    kw = {"cwd": "/", "timeout": timeout}
    if capture:
        kw["capture_output"] = True
    if data is None:
        kw["stdin"] = subprocess.DEVNULL
    else:
        kw["input"] = data
    try:
        return run(argv, **kw)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv, 124, b"", f"timed out after {exc.timeout} s".encode()
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, b"", str(exc).encode())


def failed(runner: str, what: str, code: int) -> str:
    """The line that says the launcher's `what` failed with exit `code`."""
    return f"PR_SLOT {runner}: {what} failed (exit {code})"


def lines_of(p: subprocess.CompletedProcess, runner: str, what: str) -> list[str]:
    """What a launcher call printed, and `failed`'s line last when it failed."""
    out = [
        ln
        for ln in ((p.stdout or b"") + (p.stderr or b""))
        .decode("utf-8", "replace")
        .splitlines()
        if ln.strip()
    ]
    if p.returncode:
        out.append(failed(runner, what, p.returncode))
    return out


def own_last(lines: list[str], runner: str, what: str, code: int) -> str:
    """The launcher's own last line in the `lines` of its failed `what` (lines_of's): the last one but `failed`'s,
    which stands in when the launcher printed none."""
    mark = failed(runner, what, code)
    return next((ln for ln in reversed(lines) if ln != mark), mark)


def stuck(runner: str, what: str, code: int, left: int) -> str:
    """Why a slot is broken when a signal (exit `code`) ended the launcher's `what` while `left` processes of its
    runner still ran."""
    return (
        f"a signal ({-code}) ended the launcher's {what} while {left} process(es) of {runner} still ran, which "
        f"can end every launcher call: {RESTART} to stop them"
    )


def clear_inputs(runner: str, top: str | None = None) -> list[str]:
    """Empty the slot's inputs folder (SLOT_INPUTS/<runner>: the monitor's own, which no runner writes): the lines
    when something stays."""
    d = os.path.join(top or L.SLOT_INPUTS, runner)
    left = []
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [f"PR_SLOT {runner}: its inputs folder was not read ({exc.strerror})"]
    for name in names:
        p = os.path.join(d, name)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.unlink(p)
        except OSError as exc:
            left.append(
                f"PR_SLOT {runner}: {name} stays in its inputs ({exc.strerror})"
            )
    return left


def begin(
    runner: str, launcher_path: str, run=subprocess.run, inputs: str | None = None
) -> tuple[int, list[str]]:
    """The launcher's begin as `runner` (every process of it stopped, its slot emptied, the four folders made fresh),
    then its inputs folder emptied: (the exit code, the lines); inputs that stay read as exit 1.
    """
    p = call(runner, ["begin"], launcher_path, run=run)
    lines = lines_of(p, runner, "begin")
    if p.returncode:
        return p.returncode, lines
    left = clear_inputs(runner, inputs)
    return (1 if left else 0), lines + left


def end(
    runner: str, launcher_path: str, run=subprocess.run, inputs: str | None = None
) -> tuple[int, list[str]]:
    """The launcher's end as `runner` (every process of it stopped, its slot emptied), then its inputs folder
    emptied: (the launcher's exit code, the lines)."""
    p = call(runner, ["end"], launcher_path, run=run)
    return p.returncode, lines_of(p, runner, "end") + clear_inputs(runner, inputs)


def keep_call(
    runner: str, launcher_path: str, run=subprocess.run
) -> tuple[bool, list[str]]:
    """The launcher's keep as `runner` (every process stopped, the slot's other folders made fresh, the build readable
    by the other runners): (kept, the lines)."""
    p = call(runner, ["keep"], launcher_path, run=run)
    return p.returncode == 0, lines_of(p, runner, "keep")


def _start(pid, proc: str) -> str:
    """The process's start time (/proc/PID/stat field 22), "" when there is none."""
    try:
        with open(f"{proc}/{int(pid)}/stat", "rb") as f:
            data = f.read()
    except (OSError, ValueError, TypeError):
        return ""
    fields = data.rsplit(b")", 1)[-1].split()
    return fields[19].decode() if len(fields) > 19 else ""


def _parent(pid, proc: str) -> tuple[int, str, int] | None:
    """(pid, its command's name, its parent's pid) from /proc/PID/stat; None when there is no such process."""
    try:
        with open(f"{proc}/{int(pid)}/stat", "rb") as f:
            data = f.read()
    except (OSError, ValueError, TypeError):
        return None
    head, _sep, rest = data.rpartition(b")")
    fields = rest.split()
    try:
        return (
            int(pid),
            head.partition(b"(")[2].decode("utf-8", "replace"),
            int(fields[1]),
        )
    except (IndexError, ValueError):
        return None


def _boot(proc: str) -> str:
    try:
        with open(f"{proc}/sys/kernel/random/boot_id", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return ""


class Records:
    """The slots' records in `folder` (default SLOT_LOCKS), for `runners` (the PR runners' names, in the order slots
    are taken)."""

    def __init__(
        self,
        folder: str | None = None,
        runners=None,
        proc: str = "/proc",
        clock=time.time,
    ):
        self.folder, self.proc, self.clock = folder or L.SLOT_LOCKS, proc, clock
        self.runners = list(runners or (u.name for u in L.PR_RUNNERS))

    @contextlib.contextmanager
    def lock(self):
        # The lock file is one the monitor shares (worker/perms.py, read when first asked: docker/rkmon's deploy path
        # imports this module).
        fd = os.open(
            os.path.join(self.folder, LOCK),
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _path(self, runner: str) -> str:
        return os.path.join(self.folder, f"{runner}.json")

    def read(self, runner: str) -> dict | None:
        """The slot's record; None when it is free (no record, or one that cannot be read)."""
        try:
            with open(self._path(runner), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) else None

    def write(self, runner: str, rec: dict) -> None:
        """The slot's record, written whole (whole_file), at the mode of the files the monitor shares, as the lock's."""
        whole_file.write(
            self._path(runner),
            json.dumps(rec, sort_keys=True),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )

    def remove(self, runner: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._path(runner))

    def all(self) -> dict[str, dict]:
        """Every slot's record that is not free, by runner."""
        return {r: rec for r in self.runners if (rec := self.read(r)) is not None}

    def holder(self, pid: int) -> dict:
        """What names process `pid` across pid reuse and reboots."""
        return {
            "pid": int(pid),
            "start": _start(pid, self.proc),
            "boot": _boot(self.proc),
        }

    def held(self, rec: dict) -> bool:
        """Whether `rec` still holds its slot: kept, named work, a process that still runs, or broken since this start
        of the container."""
        state = rec.get("state")
        if state == KEPT:
            return True
        if state == BROKEN:
            return (
                bool(rec.get("init"))
                and _start(1, self.proc) == rec["init"]
                and _boot(self.proc) == rec.get("boot")
            )
        if state not in (TAKEN, FREEING):
            return False
        if "pid" not in rec:
            owner = rec.get("owner")
            return bool(rec.get("name")) and (
                not isinstance(owner, dict) or self.runs(owner)
            )
        return self.runs(rec)

    def left(self, runner: str) -> list[int]:
        """The processes of `runner`, a PR runner, that still run (worker/pr_results.py running, on this /proc); none
        for a name that is no PR runner's."""
        uid = next((u.uid for u in L.PR_RUNNERS if u.name == runner), None)
        if uid is None:
            return []
        return paths.worker_module("pr_results").running(uid, self.proc)

    def runs(self, who: dict) -> bool:
        """Whether the process `who` names ({"pid", "start", "boot"}: `holder`) still runs."""
        return (
            bool(who.get("start"))
            and _start(who.get("pid"), self.proc) == who["start"]
            and _boot(self.proc) == who.get("boot")
        )

    def owner_of(self, pid: int) -> dict | None:
        """What names the process that started the work of process `pid` (`holder`): the first ancestor of `pid` that
        is none of PASS_THROUGH (a shell, or a command that only starts another); None when only those lead up to the
        container's first process, or to a process this one cannot read."""
        cur = _parent(pid, self.proc)
        for _ in range(64):
            if cur is None:
                return None
            cur = _parent(cur[2], self.proc)
            if cur is None or cur[0] <= 1:
                return None
            if cur[1] not in PASS_THROUGH:
                return self.holder(cur[0])
        return None


def shared_build_lines(path) -> list[str]:
    """The lines of a shared build record (review.SHARED_BUILD, which the checks write: review.SHARED_BUILD_LINES): the
    build folder, its tree and the tree's commit, one to a line, so a path with a space stays whole; [] when it cannot
    be read."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    except (OSError, ValueError):
        return []


def drop_record(rec: dict) -> bool:
    """Remove a kept slot's shared build record while it still names the slot's build: True when it did."""
    path, build = rec.get("record"), rec.get("build")
    if not path or not build:
        return False
    if shared_build_lines(path)[:1] != [build]:
        return False
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    return True


def take(
    records: Records,
    *,
    job: str = "",
    pr="",
    pid: int | None = None,
    name: str = "",
    head: str = "",
    owner: dict | None = None,
    wait_s: float = WAIT_S,
    sleep=time.sleep,
    say=None,
    skip=(),
) -> tuple[str, bool]:
    """A slot for a process (`pid`) or for named work (`name`, for the review of `head`, started by `owner`, a
    `holder`): (its runner, whether its folders must be made fresh). Named work that still holds a slot gets it back.
    Else the first free slot; when none is, the kept slot kept longest; when none is kept either, it waits for one
    (`say`, once, says so). The runners in `skip` are not taken. ("", False) when every slot it may take is broken:
    none will be free before the container's next start."""
    if not name and pid is None:
        raise ValueError("take: a pid or a name")
    waited = False
    while True:
        with records.lock():
            recs = records.all()
            if name:
                for r, rec in recs.items():
                    if taken_by(rec, name=name) and records.held(rec):
                        return r, False
            mine = [r for r in records.runners if r not in skip]
            pick = next(
                (r for r in mine if r not in recs or not records.held(recs[r])),
                None,
            )
            if pick is None:
                kept = sorted(
                    (recs[r].get("since") or 0, r)
                    for r in mine
                    if recs[r].get("state") == KEPT
                )
                if kept:
                    pick = kept[0][1]
                    drop_record(recs[pick])
                elif all(recs[r].get("state") == BROKEN for r in mine):
                    return "", False
            if pick is not None:
                rec = {
                    "state": TAKEN,
                    "job": str(job or ""),
                    "pr": str(pr or ""),
                    "since": records.clock(),
                }
                rec.update(
                    {"name": name, "head": str(head or "")}
                    if name
                    else records.holder(pid)
                )
                if name and owner:
                    rec["owner"] = dict(owner)
                records.write(pick, rec)
                return pick, True
        if say and not waited:
            say("PR_SLOT every slot is taken or kept: waiting for one")
            waited = True
        sleep(wait_s)


def taken_by(rec: dict, pid=None, name: str = "") -> bool:
    """Whether `rec` is the taking of process `pid` or of named work `name`."""
    if rec.get("state") != TAKEN:
        return False
    return rec.get("name") == name if name else rec.get("pid") == pid


def release(records: Records, runner: str, *, pid=None, name: str = "") -> None:
    """Remove the record of `runner`'s slot while it is still the taking of process `pid` (or named work `name`),
    without the launcher: the slot is free, and the next job to take it begins it again.
    """
    with records.lock():
        rec = records.read(runner)
        if rec is not None and taken_by(rec, pid, name):
            records.remove(runner)


def broken(records: Records, why: str) -> dict:
    """A broken slot's record, until the container's next start, saying `why`."""
    return {
        "state": BROKEN,
        "init": _start(1, records.proc),
        "boot": _boot(records.proc),
        "why": why,
        "since": records.clock(),
    }


def mark_broken(records: Records, runner: str, why: str) -> None:
    """Mark `runner`'s slot broken until the container's next start, saying `why`."""
    with records.lock():
        records.write(runner, broken(records, why))


REFUSED = 3  # the launcher's exit when the slot's folder is not as the start made it (docker/rk_prcode.py)
LEFT = 125  # its exit when a process of the slot's user outlived SIGKILL


def take_begun(records: Records, begin, **take_kw) -> tuple[str, list[str]]:
    """`take`, then `begin(runner)` (the launcher's begin: (exit, lines)) on a slot taken fresh. A slot the launcher
    refuses (REFUSED) is marked broken, why the launcher's own last line. One whose begin a signal ended (a process of
    the slot's user can end the launcher: they are one user) is marked broken while processes of its user still run
    (`stuck`), else let go, as one whose processes outlive SIGKILL (LEFT) is; either way the next slot is taken.
    (runner, the lines); runner is "" when no slot could be begun (the lines say why).
    """
    lines, tried = [], set(take_kw.pop("skip", ()))
    pid, name = take_kw.get("pid"), take_kw.get("name", "")
    while True:
        runner, fresh = take(records, skip=tried, **take_kw)
        if not runner:
            return "", lines + [
                "PR_SLOT no slot can be taken: each is broken or was let go (the container's next start makes "
                "them again)"
            ]
        if not fresh:
            return runner, lines
        code, out = begin(runner)
        lines += out
        if code == 0:
            return runner, lines
        if code == REFUSED:
            mark_broken(records, runner, own_last(out, runner, "begin", code)[:300])
        elif code < 0 and (left := records.left(runner)):
            why = stuck(runner, "begin", code, len(left))
            mark_broken(records, runner, why)
            lines.append(f"PR_SLOT {runner}: broken: {why}")
        else:
            release(records, runner, pid=pid, name=name)
            if code != LEFT and code >= 0:
                return "", lines
        tried.add(runner)


def free(records: Records, runner: str, end, holds=None) -> list[str]:
    """Free `runner`'s slot when its record is one `holds` accepts (default: any record): the record made `freeing` (a
    kept slot's shared build record removed first), `end(runner)` (the launcher's: (exit, lines)), the record removed,
    or marked broken when a signal ended that `end` while processes of the slot's user still run (`stuck`); a record
    another process wrote meanwhile stays. The lines; [] when the slot is free already or another holds it.
    """
    me = os.getpid()
    with records.lock():
        rec = records.read(runner)
        if rec is None or (holds is not None and not holds(rec)):
            return []
        if rec.get("state") == KEPT:
            drop_record(rec)
        records.write(
            runner,
            {"state": FREEING, **records.holder(me), "since": records.clock()},
        )
    code, lines = end(runner)
    with records.lock():
        cur = records.read(runner)
        if cur and cur.get("state") == FREEING and cur.get("pid") == me:
            left = records.left(runner) if code < 0 else []
            if left:
                why = stuck(runner, "end", code, len(left))
                records.write(runner, broken(records, why))
                lines = lines + [f"PR_SLOT {runner}: broken: {why}"]
            else:
                records.remove(runner)
    return lines


def keep(
    records: Records,
    runner: str,
    keep_call,
    end,
    *,
    pid: int,
    pr,
    head: str,
    build: str,
    record: str,
) -> tuple[bool, list[str]]:
    """Keep the slot process `pid` took, for PR `pr`'s review of `head`: once its record is checked to be that
    process's taking, `keep_call(runner)` (the launcher's: (ok, lines)), then the record, checked again, kept, naming
    `build` and its shared build `record`; the other slots kept for that PR are freed. (kept, the lines).
    """
    not_kept = f"PR_SLOT {runner}: not taken by process {pid}: not kept"
    with records.lock():
        if not taken_by(records.read(runner) or {}, pid=int(pid)):
            return False, [not_kept]
    ok, lines = keep_call(runner)
    if not ok:
        return False, lines
    with records.lock():
        if not taken_by(records.read(runner) or {}, pid=int(pid)):
            return False, lines + [not_kept]
        records.write(
            runner,
            {
                "state": KEPT,
                "pr": str(pr),
                "head": head,
                "build": build,
                "record": record,
                "since": records.clock(),
            },
        )
        others = {
            r: x
            for r, x in records.all().items()
            if r != runner and x.get("state") == KEPT and x.get("pr") == str(pr)
        }
    for r, x in others.items():
        lines += free(records, r, end, holds=lambda cur, x=x: cur == x)
    return True, lines


def free_ended(records: Records, live, end) -> list[str]:
    """Free the kept slots and the named work whose review is no longer in flight (`live(pr, head)`), and the named
    work whose owner is gone (`Records.held`): the lines. Named work taken with no review in flight stays until it
    ends or its owner is gone."""
    out = []
    for r, rec in records.all().items():
        pr, head = rec.get("pr") or "", rec.get("head") or ""
        if rec.get("state") == KEPT:
            ended = not live(pr, head)
        elif rec.get("state") == TAKEN and rec.get("name"):
            ended = (bool(head) and not live(pr, head)) or not records.held(rec)
        else:
            continue
        if ended:
            out += free(records, r, end, holds=lambda cur, rec=rec: cur == rec)
    return out


def by_pr(records: Records | None = None) -> dict[int, list[str]]:
    """PR number -> the runners whose slots hold its work now: taken by its job or by its named work, or kept for its
    review (the records that name a PR); a slot whose holder is gone is no PR's."""
    records = records or Records()
    out: dict[int, list[str]] = {}
    for r, rec in records.all().items():
        pr = str(rec.get("pr") or "")
        if pr.isdigit() and records.held(rec):
            out.setdefault(int(pr), []).append(r)
    return out


def slot_of(path: str) -> str:
    """The runner whose slot's folder is `path` or holds it; "" when none does."""
    top = L.SLOTS_DIR.rstrip("/") + "/"
    p = os.path.normpath(str(path))
    runner = p[len(top) :].partition("/")[0] if p.startswith(top) else ""
    return runner if runner in {u.name for u in L.PR_RUNNERS} else ""


def split(path: str) -> tuple[str, str]:
    """`path` in a slot's folder as (that folder, the path under it); any other path as (itself, "")."""
    p = os.path.normpath(str(path))
    runner = slot_of(p)
    if not runner:
        return p, ""
    top = folder(runner)
    return top, p[len(top) + 1 :]


def kept_build(path: str, records: Records | None = None) -> bool:
    """Whether the build at `path` may still be reused as its review's shared build: True for a path in no slot's
    folder (no slot decides it), else only while that slot's record is kept and names `path` as its build.
    """
    runner = slot_of(path)
    if not runner:
        return True
    rec = (records or Records()).read(runner)
    return (
        rec is not None
        and rec.get("state") == KEPT
        and os.path.normpath(str(rec.get("build") or "")) == os.path.normpath(str(path))
    )
