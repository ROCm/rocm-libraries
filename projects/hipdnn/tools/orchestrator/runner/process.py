# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Process launching: streamed output, hard timeouts, and process-tree kills.

Agent steps run for hours and emit a lot. Output therefore goes straight to files
rather than through a pipe into memory, and a timeout kills the whole tree -- an agent
CLI that spawns compilers leaves orphans behind if only the parent is signalled.
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
from typing import Mapping, Sequence


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

        pumps: list[threading.Thread] = []
        if tee:
            pumps = [
                _pump(process.stdout, out_file, sys.stdout),
                _pump(process.stderr, err_file, sys.stderr),
            ]

        if stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin_text.encode("utf-8"))
            except (BrokenPipeError, OSError):
                pass
            finally:
                process.stdin.close()

        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_tree(process)
            process.wait()
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
