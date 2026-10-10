# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-GPU leases: one process tree benches on a GPU at a time.

A lease is an exclusive flock on `<lease dir>/gpu<N>.lock`, N the physical
GPU index (the index ROCR_VISIBLE_DEVICES takes, `lib/kfd.py`). The lock
file holds a JSON record of the holder (pid, user, purpose, run directory,
start time), so that a refusal names it. The kernel drops the lock when the
last descriptor of the holder's lock closes: a lease ends with its holder, or
with the children that inherited the descriptor (`scripts/gpu_run.py`).

A process that holds a GPU's lease, or whose ancestor holds it, uses the GPU
without taking the lease again: the orchestrator leases a stage's GPUs, and
the stage, its sub-stages and their benches work under that lease.

The lease directory is `TILEWRIGHT_GPU_LEASE_DIR`, else
`tilewright-gpu-leases` in the system temp directory. Processes see each
other's leases only when they use the same directory.

`hold(devices)` leases the GPUs of the stage's `--device` indices
(`kfd.bench_gpus`; each bench sees its GPU alone). An environment that
hides every GPU (`-1`) leaves nothing to lease.

A caller that waits for a lease (`wait_s`) can pass `report`, which is told
that it waits, naming the holder, then every `WAIT_REPORT_S` that it still
waits, and when it got the lease.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pwd
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from . import kfd
from .ui import fmt_dur

LEASE_DIR_VAR = "TILEWRIGHT_GPU_LEASE_DIR"
LEASE_DIR_NAME = "tilewright-gpu-leases"
RULE = "one process tree per GPU"
WAIT_REPORT_S = 300.0


class LeaseError(RuntimeError):
    """A GPU cannot be leased."""


@dataclass(frozen=True)
class Holder:
    gpu: int
    pid: int
    user: Optional[str]
    purpose: str
    run_dir: Optional[str]
    started: str

    @classmethod
    def parse(cls, text: str) -> Optional["Holder"]:
        try:
            d = json.loads(text)
            return cls(
                gpu=int(d["gpu"]),
                pid=int(d["pid"]),
                user=d.get("user"),
                purpose=str(d.get("purpose") or ""),
                run_dir=d.get("run_dir"),
                started=str(d.get("started") or ""),
            )
        except (ValueError, TypeError, KeyError):
            return None

    def describe(self) -> str:
        out = f"pid {self.pid}"
        if self.user:
            out += f" (user {self.user})"
        out += f" for {self.purpose!r}"
        if self.run_dir:
            out += f" in {self.run_dir}"
        out += f" since {self.started}"
        if kfd.proc_info(self.pid).gone:
            out += "; that pid has exited and a process it started holds the lock"
        return out


def _who(holder: Optional[Holder]) -> str:
    if holder is None:
        return "a process that has not recorded itself"
    return holder.describe()


class LeaseBusy(LeaseError):
    def __init__(self, gpu: int, holder: Optional[Holder], path: Path) -> None:
        super().__init__(
            f"refused ({RULE}): GPU {gpu} is leased by {_who(holder)} ({path})"
        )
        self.gpu = gpu
        self.holder = holder
        self.path = path


def lease_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if environ is None else environ
    explicit = env.get(LEASE_DIR_VAR)
    if explicit:
        return Path(explicit)
    # Not XDG_RUNTIME_DIR: logind removes it when the user's last session
    # ends, while runs in tmux go on locking fresh files there.
    return Path(tempfile.gettempdir()) / LEASE_DIR_NAME


@dataclass
class Leases:
    """The GPUs a `hold` block may use: taken here (`acquired`, whose lock
    descriptors are `fds`) or held by this process or an ancestor."""

    gpus: List[int] = field(default_factory=list)
    acquired: List[int] = field(default_factory=list)
    inherited: List[int] = field(default_factory=list)
    fds: List[int] = field(default_factory=list)


def lock_path(gpu: int, environ: Optional[Mapping[str, str]] = None) -> Path:
    return lease_dir(environ) / f"gpu{int(gpu)}.lock"


def read_holder(path: Path, attempts: int = 5) -> Optional[Holder]:
    """The record in a lock file; None when it is empty or unreadable (a
    new holder writes it right after locking, so a few reads are tried)."""
    holder: Optional[Holder] = None
    for i in range(attempts):
        if i:
            time.sleep(0.02)
        try:
            holder = Holder.parse(Path(path).read_text())
        except OSError:
            holder = None
        if holder is not None:
            break
    return holder


def _user() -> Optional[str]:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return None


def _release(fd: int) -> None:
    try:
        os.ftruncate(fd, 0)
    except OSError:
        pass
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _acquire(
    path: Path,
    gpu: int,
    purpose: str,
    run_dir: Optional[str],
    wait_s: float,
    held: Leases,
    report: Optional[Callable[[str], None]] = None,
) -> None:
    start = time.monotonic()
    deadline = start + max(0.0, float(wait_s))
    next_report = start
    waited = False
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            holder = read_holder(path)
            if holder is not None and holder.pid in kfd.ancestors(os.getpid()):
                held.inherited.append(gpu)
                return
            now = time.monotonic()
            left = deadline - now
            if left <= 0:
                raise LeaseBusy(gpu, holder, path) from None
            if report is not None and now >= next_report:
                report(
                    f"still waiting for GPU {gpu} ({fmt_dur(now - start)} of "
                    f"{fmt_dur(wait_s)}), leased by {_who(holder)}"
                    if waited
                    else f"GPU {gpu} is leased by {_who(holder)} ({path}); "
                    f"waiting up to {fmt_dur(wait_s)} for it"
                )
                next_report = now + WAIT_REPORT_S
            waited = True
            time.sleep(min(1.0, left))
            continue
        if report is not None and waited:
            report(f"GPU {gpu} leased after {fmt_dur(time.monotonic() - start)}")
        rec = Holder(gpu, os.getpid(), _user(), purpose, run_dir, kfd.utc_iso())
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, (json.dumps(asdict(rec)) + "\n").encode(), 0)
        except OSError:
            _release(fd)
            raise
        held.fds.append(fd)
        held.acquired.append(gpu)
        return


@contextlib.contextmanager
def hold_gpus(
    gpus: Iterable[int],
    purpose: str,
    run_dir: Optional[os.PathLike] = None,
    *,
    wait_s: float = 0.0,
    environ: Optional[Mapping[str, str]] = None,
    report: Optional[Callable[[str], None]] = None,
) -> Iterator[Leases]:
    """Lease the physical `gpus` for the block, in ascending order; a GPU
    another process tree holds raises LeaseBusy after `wait_s` seconds (0:
    at once), releasing what was taken; `report` hears about the wait. A
    lock descriptor the block made inheritable (shared with a command) is
    only closed on exit: its lease ends with the last process that holds
    it."""
    held = Leases(gpus=sorted({int(g) for g in gpus}))
    directory = lease_dir(environ)
    try:
        try:
            if held.gpus:
                directory.mkdir(parents=True, exist_ok=True)
            for gpu in held.gpus:
                _acquire(
                    directory / f"gpu{gpu}.lock",
                    gpu,
                    purpose,
                    str(run_dir) if run_dir is not None else None,
                    wait_s,
                    held,
                    report,
                )
        except OSError as e:
            raise LeaseError(f"cannot take a lease in {directory}: {e}") from None
        yield held
    finally:
        for fd in held.fds:
            if os.get_inheritable(fd):
                os.close(fd)
            else:
                _release(fd)
        held.fds.clear()


def hold(
    devices: Iterable[int],
    purpose: str,
    run_dir: Optional[os.PathLike] = None,
    *,
    wait_s: float = 0.0,
    environ: Optional[Mapping[str, str]] = None,
    report: Optional[Callable[[str], None]] = None,
) -> contextlib.AbstractContextManager:
    """`hold_gpus` of the GPUs benches on `devices` use
    (`kfd.bench_gpus`); LeaseError also when the device lists do not name
    them."""
    devices = list(devices)
    try:
        gpus = kfd.bench_gpus(devices, environ)
    except kfd.GpuMapError as e:
        raise LeaseError(f"cannot lease the GPUs of devices {devices}: {e}") from None
    return hold_gpus(
        gpus, purpose, run_dir, wait_s=wait_s, environ=environ, report=report
    )


def _locked_inodes() -> Optional[Set[Tuple[int, int, int]]]:
    """(device major, minor, inode) of every file with a granted flock."""
    try:
        lines = (kfd.PROC / "locks").read_text().splitlines()
    except OSError:
        return None
    out: Set[Tuple[int, int, int]] = set()
    for line in lines:
        parts = line.split()
        if len(parts) < 6 or parts[1] != "FLOCK":
            continue
        try:
            major, minor, inode = parts[5].split(":")
            out.add((int(major, 16), int(minor, 16), int(inode)))
        except ValueError:
            continue
    return out


def status(environ: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    """Every lease file: its GPU, whether it is held (None when /proc/locks
    cannot be read) and the holder's record. Reads only, never locks."""
    locked = _locked_inodes()
    out: List[Dict[str, Any]] = []
    directory = lease_dir(environ)
    paths = sorted(
        (p for p in directory.glob("gpu*.lock") if p.stem[3:].isdigit()),
        key=lambda p: int(p.stem[3:]),
    )
    for path in paths:
        try:
            st = path.stat()
        except OSError:
            continue
        key = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
        held = None if locked is None else key in locked
        holder = read_holder(path, attempts=1) if held is not False else None
        out.append(
            {
                "gpu": int(path.stem[3:]),
                "held": held,
                "holder": asdict(holder) if holder is not None else None,
                "path": str(path),
            }
        )
    return out
