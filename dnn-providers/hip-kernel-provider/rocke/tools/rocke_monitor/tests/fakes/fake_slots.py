# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PR runners' slots for a test, outside a container (rkmon_lib/slots.py, rkmon_lib/prcode.py).

`Slots(folder)` moves the layout's slot folders (SLOTS_DIR, SLOT_LOCKS, SLOT_INPUTS, with each runner's inputs folder,
as the container's start makes them) into a folder of the test's and, while `here()` holds, names a launcher in the
environment and stands in for it behind sudo: every `sudo -n -u RUNNER LAUNCHER ...` that subprocess.run is asked for
is answered here, and every other command runs as it is.
- begin: the slot's four folders made fresh; end: the slot emptied; keep: nothing; kill: one line, as the launcher's
  (`stopped`: how many processes it says it stopped, by runner).
- run [-C DIR] -- CMD: nothing runs. `ran` gets the runner, CMD, the folder it would run in and the environment it was
  handed on stdin; `on_run(slot, cmd)`, when set, may write the job's results into the slot; it prints `printed` (on
  the caller's stdout when the output is not captured) and exits `rc`.
`calls` lists each call as (runner, verb), `asked` as (runner, the launcher's arguments).
"""

import contextlib
import os
import pathlib
import shutil
import subprocess
import sys
from unittest import mock

from rkmon_lib import container_layout as L
from rkmon_lib import slots

sys.path.insert(0, str(pathlib.Path(slots.__file__).resolve().parent.parent / "worker"))
import pr_env  # noqa: E402

LAUNCHER = "/stand-in/rk_prcode.py"


class Slots:
    def __init__(self, folder, printed: bytes = b"AS-PR-RUNNER\n"):
        self.dir = pathlib.Path(folder) / "slots"
        self.locks = self.dir / "locks"
        self.inputs = self.dir / "inputs"
        self.locks.mkdir(parents=True, exist_ok=True)
        for u in L.PR_RUNNERS:
            (self.inputs / u.name).mkdir(parents=True, exist_ok=True)
        self.printed, self.rc, self.on_run = printed, 0, None
        self.stopped: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []
        self.asked: list[tuple[str, list[str]]] = []
        self.ran: list[dict] = []
        self._real = subprocess.run

    @contextlib.contextmanager
    def here(self):
        with mock.patch.multiple(
            L,
            SLOTS_DIR=str(self.dir),
            SLOT_LOCKS=str(self.locks),
            SLOT_INPUTS=str(self.inputs),
        ), mock.patch.dict(os.environ, {slots.RUNNER_VAR: LAUNCHER}), mock.patch.object(
            subprocess, "run", self.run
        ):
            yield self

    def records(self) -> dict:
        return slots.Records().all()

    def slot(self, runner: str) -> pathlib.Path:
        return self.dir / runner

    def _done(self, argv, kw, out: bytes, rc: int = 0):
        if kw.get("capture_output"):
            return subprocess.CompletedProcess(argv, rc, out, b"")
        sys.stdout.write(out.decode())
        sys.stdout.flush()
        return subprocess.CompletedProcess(argv, rc)

    def run(self, argv, **kw):
        a = [str(x) for x in argv]
        if a[:3] != ["sudo", "-n", "-u"] or len(a) < 6 or a[4] != LAUNCHER:
            return self._real(argv, **kw)
        runner, verb, args = a[3], a[5], a[6:]
        self.calls.append((runner, verb))
        self.asked.append((runner, a[5:]))
        slot = self.slot(runner)
        if verb == "begin":
            shutil.rmtree(slot, ignore_errors=True)
            for name in L.SLOT_FOLDERS:
                (slot / name).mkdir(parents=True)
            return self._done(argv, kw, b"")
        if verb == "end":
            shutil.rmtree(slot, ignore_errors=True)
            slot.mkdir(parents=True)
            return self._done(argv, kw, b"")
        if verb == "keep":
            return self._done(argv, kw, f"PR_SLOT {runner}: kept\n".encode())
        if verb == "kill":
            n = self.stopped.get(runner, 0)
            return self._done(
                argv, kw, f"PR_SLOT {runner}: stopped {n} process(es)\n".encode()
            )
        assert verb == "run", a
        where = ""
        if args[:1] == ["-C"]:
            where, args = args[1], args[2:]
        assert args[:1] == ["--"], a
        cmd = args[1:]
        self.ran.append(
            {
                "runner": runner,
                "cmd": cmd,
                "cwd": os.path.join(slot, where) if where else str(slot / L.SLOT_HOME),
                "env": pr_env.parse(kw.get("input") or b""),
            }
        )
        if self.on_run:
            self.on_run(slot, cmd)
        return self._done(argv, kw, self.printed, self.rc)
