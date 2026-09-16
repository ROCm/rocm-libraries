# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""The child that actually runs the engine.

`Engine.run()` is synchronous, blocking, multi-hour and has no cancellation
API, so it runs here -- in a process the supervisor can tree-kill -- rather than
in a thread it could only ask nicely to stop.

The launch spec arrives as one JSON document on stdin. Three private frames go
back out on stdout, newline-delimited:

    {"t":"log","text":"..."}                      the engine's own prose
    {"t":"pid","step":"<id>","pid":<n>}           a step's child, while alive
    {"t":"done","status":"ok|failed","error":...} the engine returned

That stream is private between this process and its supervisor; it is not the
MCP stdio stream, which belongs to the server's own parent. Diagnostics that are
not frames go to stderr, which the supervisor attaches to a failure the manifest
never got to record.

Exit code is 0 only when the run succeeded. A `ConfigError` out of `run()`'s
preflight or run-directory creation happens before the first checkpoint, so it
leaves no manifest at all -- the supervisor reconciles that as a failed run from
the exit code and this process's stderr.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from runner.engine import Engine
from runner.errors import OrchestratorError
from runner.flow import Flow, Step
from runner.toolreg import ToolRegistry


class Channel:
    """The private frame stream. Line-delimited, flushed, never interleaved."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def _emit(self, frame: dict[str, Any]) -> None:
        line = json.dumps(frame, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def log(self, text: str) -> None:
        self._emit({"t": "log", "text": str(text)})

    def pid(self, step: Step, process: subprocess.Popen) -> None:
        """Relay a step's child while it is still alive.

        This is the only moment the pid exists and is killable: `launch()`
        returns after the child has exited, so a pid observed from its return
        value is already dead and a cancellation built on one is a no-op. The
        engine's iteration is not part of the callback's contract and is not
        needed here -- the supervisor uses these pids to kill, not to report.
        """
        self._emit({"t": "pid", "step": step.id, "pid": process.pid})

    def done(self, status: str, error: str | None) -> None:
        self._emit({"t": "done", "status": status, "error": error})


def main() -> int:
    channel = Channel(sys.stdout)
    spec = json.loads(sys.stdin.read() or "{}")
    try:
        flow = Flow.load(spec["flow"])
        registry = ToolRegistry.load(spec["toolsPath"], spec.get("profile"))
        engine = Engine(
            flow,
            registry,
            dict(spec.get("inputs") or {}),
            run_dir=Path(spec["runDir"]),
            run_id=spec["runId"],
            max_iterations=spec.get("maxIterations"),
            profile=spec.get("profile"),
            log=channel.log,
            on_step_process=channel.pid,
        )
        report = engine.run()
    except OrchestratorError as failure:
        print(str(failure), file=sys.stderr, flush=True)
        channel.done("failed", str(failure))
        return 1
    channel.done(report.status, report.error)
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
