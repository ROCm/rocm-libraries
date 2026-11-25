# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The folders a deployment's commands read, for the image test's several-deployments stage
(tests/container/stage_several_deployments.py). The stage puts this folder first on PYTHONPATH for each rkmon command
of a deployment, so that each Python process of the command that reads PYTHONPATH imports this module as it starts,
and it logs each file that process opens and each folder it lists in the folders of the other deployments on the host.
It changes nothing else, and never stops the process.

    FOLDER_READS_WATCH  the folders to watch, joined by os.pathsep
    FOLDER_READS_LOG    the file it logs to, a JSON line each: one as each process starts ({"t", "pid", "started":
                        the process's own arguments}), so that the stage knows it ran; then one for each file opened or
                        folder listed in a watched folder ({"t", "pid", "event", "path"})

With either variable empty it does nothing.
"""

import json
import os
import sys
import time

WATCH = tuple(
    os.path.join(os.path.realpath(p), "")
    for p in os.environ.get("FOLDER_READS_WATCH", "").split(os.pathsep)
    if p
)
LOG = os.environ.get("FOLDER_READS_LOG", "")
# The audit events of a file opened and a folder listed: their first argument is its path, or a file descriptor
EVENTS = frozenset(["open", "os.listdir", "os.scandir"])
_busy = []  # set while the hook writes, so that the log's own open is not looked at


def _log(line: dict) -> None:
    _busy.append(True)
    try:
        fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(line) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass
    finally:
        _busy.pop()


def _hook(event: str, args: tuple) -> None:
    if event not in EVENTS or _busy or not args:
        return
    path = args[0]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    try:
        where = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError, OSError):
        return
    if os.path.join(where, "").startswith(WATCH):
        _log(
            {
                "t": round(time.time(), 3),
                "pid": os.getpid(),
                "event": event,
                "path": where,
            }
        )


if WATCH and LOG:
    _log(
        {
            "t": round(time.time(), 3),
            "pid": os.getpid(),
            "started": list(getattr(sys, "orig_argv", [])),
        }
    )
    sys.addaudithook(_hook)
