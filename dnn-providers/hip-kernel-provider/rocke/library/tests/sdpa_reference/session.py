# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reuse isolated interpreters while executing the selected, unchanged worker."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

# Import through the child's selected PYTHONPATH, including the frozen baseline
# runner. Only transport changes; validation and kernel execution stay in run().
_SERVER = """
import json
import sys
from pathlib import Path
from sdpa_reference.worker import run

for line in sys.stdin:
    request_path = Path(json.loads(line))
    work = request_path.parent
    run(json.loads(request_path.read_text()), work)
    (work / 'complete').touch()
"""
_ACTIVE = ContextVar("sdpa_worker_session", default=None)


class _Worker:
    def __init__(self, env: dict[str, str]) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="rocke-sdpa-worker-")
        self.log_path = Path(self.directory.name) / "worker.log"
        self.log = self.log_path.open("w")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-s", "-u", "-c", _SERVER],
                cwd=self.directory.name,
                env=env,
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=self.log,
                text=True,
            )
        except BaseException:
            self.log.close()
            self.directory.cleanup()
            raise

    def execute(self, request: Path, *, timeout: float) -> None:
        try:
            self.process.stdin.write(json.dumps(str(request.resolve())) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(self.error()) from error
        deadline = time.monotonic() + timeout
        while not request.with_name("complete").exists():
            if self.process.poll() is not None:
                raise RuntimeError(self.error())
            if time.monotonic() >= deadline:
                self.process.kill()
                self.process.wait()
                raise TimeoutError("SDPA worker exceeded its request timeout")
            time.sleep(0.01)

    def error(self) -> str:
        return (
            "SDPA worker failed:\n" + self.log_path.read_text(errors="replace")[-16000:]
        )

    def close(self) -> None:
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.log.close()
            self.directory.cleanup()


class WorkerSession:
    """Keep baseline and candidate processes separate and close them on exit."""

    def __init__(self, *, timeout: float = 300) -> None:
        self.timeout = timeout
        self.workers: dict[tuple[str, str], _Worker] = {}

    def execute(self, mode: str, request: Path, env: dict[str, str]) -> None:
        """Reuse only workers with the same role and selected import roots."""
        key = (mode, env["PYTHONPATH"])
        if key not in self.workers:
            self.workers[key] = _Worker(env)
        try:
            self.workers[key].execute(request, timeout=self.timeout)
        except BaseException:
            self.workers.pop(key).close()
            raise

    def close(self) -> None:
        """Reap every child before removing its temporary working directory."""
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()


@contextmanager
def reuse_workers() -> Iterator[WorkerSession]:
    """Reuse workers only within an explicit verification scope."""
    session = WorkerSession()
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)
        session.close()
