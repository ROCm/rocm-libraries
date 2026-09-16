# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Process launching: streamed output, hard timeouts, and process-tree kills.

Agent steps run for hours and emit a lot. Output therefore goes straight to files
rather than through a pipe into memory, and a timeout kills the whole tree -- an agent
CLI that spawns compilers leaves orphans behind if only the parent is signalled. The
timeout is a deadline on the whole interaction, not just on waiting: a prompt bigger
than the pipe buffer is delivered on a thread so a child that never reads stdin cannot
block the caller outside its own timeout.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class ProcResult:
    exit_code: int
    timed_out: bool
    duration_s: float


def build_env(
    tool_env: Mapping[str, str] = {},
    step_env: Mapping[str, str] = {},
    path_prepend: Sequence[str] = (),
) -> dict[str, str]:
    """Process env: inherited, then tool wiring, then step overrides (step wins)."""
    env = dict(os.environ)
    env.update(tool_env)
    env.update(step_env)
    if path_prepend:
        existing = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*[str(item) for item in path_prepend], existing])
    return env


def launch(
    argv: Sequence[str],
    *,
    cwd: str | Path | None,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: str | None = None,
    timeout: float | None = None,
    tee: bool = False,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> ProcResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with open(stdout_path, "wb") as out_file, open(stderr_path, "wb") as err_file:
        if tee:
            stdout_target: object = subprocess.PIPE
            stderr_target: object = subprocess.PIPE
        else:
            stdout_target, stderr_target = out_file, err_file

        process = subprocess.Popen(
            [str(item) for item in argv],
            cwd=str(cwd) if cwd else None,
            env=dict(env),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            **_new_group(),
        )
        # Handed over while the child is still alive. `launch` returns only after
        # `process.wait()`, so a caller that needs a killable pid -- a supervisor
        # cancelling a run -- can only get one here.
        if on_start is not None:
            on_start(process)

        pumps: list[threading.Thread] = []
        if tee:
            pumps = [
                _pump(process.stdout, out_file, sys.stdout),
                _pump(process.stderr, err_file, sys.stderr),
            ]

        # One deadline covers the whole interaction, delivery included.
        writer = (
            _write_stdin(process, stdin_text.encode("utf-8"))
            if stdin_text is not None and process.stdin is not None
            else None
        )

        timed_out = False
        try:
            remaining = (
                None
                if timeout is None
                else max(0.0, started + timeout - time.monotonic())
            )
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_tree(process)
            process.wait()
        if writer is not None:
            writer.join(timeout=5)
        for pump in pumps:
            pump.join(timeout=5)

    return ProcResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        timed_out=timed_out,
        duration_s=round(time.monotonic() - started, 3),
    )


def kill_tree(process: subprocess.Popen) -> None:
    """Kill the process and everything it spawned. Best effort, never raises."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
        else:
            group = os.getpgid(process.pid)
            os.killpg(group, signal.SIGTERM)
            for _ in range(20):
                if process.poll() is not None:
                    return
                time.sleep(0.1)
            os.killpg(group, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _write_stdin(process: subprocess.Popen, payload: bytes) -> threading.Thread:
    """Deliver the prompt without holding the deadline hostage.

    A prompt is routinely larger than the pipe buffer, so a synchronous write blocks
    until the child drains it. A child that never reads stdin -- hung, or busy with
    something of its own -- then blocks the orchestrator *outside* the timeout, which is
    precisely the case the timeout exists for. Writing on a thread puts delivery inside
    the same deadline as execution.
    """

    def run() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(payload)
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _new_group() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _pump(source, *sinks) -> threading.Thread:
    def run() -> None:
        if source is None:
            return
        for chunk in iter(lambda: source.read(4096), b""):
            for sink in sinks:
                buffer = getattr(sink, "buffer", sink)
                try:
                    buffer.write(chunk)
                    sink.flush()
                except (ValueError, OSError):
                    pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
