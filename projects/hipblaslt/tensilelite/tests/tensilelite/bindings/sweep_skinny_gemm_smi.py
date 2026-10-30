# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocm-smi helpers of the skinny GEMM sweep: GPU selection, clock sampling and
the clock lock around the timed stage.

Nothing here touches HIP, so the main process of the sweep can use it.
"""

import atexit
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field

SMI_TIMEOUT_S = 30
# rocm-smi does not always fail its exit code when a request is refused.
REFUSAL = re.compile(r"sudo|not supported|unable|error|fail", re.IGNORECASE)
CARD = re.compile(r"card(\d+)")
MHZ = re.compile(r"(\d+)\s*mhz", re.IGNORECASE)


class SweepAbort(Exception):
    """A condition the user must fix; reported without a traceback."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class SmiResult:
    ok: bool
    rc: int
    message: str
    stdout: str


def run_smi(args: list[str]) -> SmiResult:
    """Runs rocm-smi with the plain user's rights; never raises. The message is
    the output on one line without the banner, for logging."""
    try:
        proc = subprocess.run(
            ["rocm-smi", *args],
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return SmiResult(False, -1, str(err), "")
    lines = (proc.stderr + proc.stdout).splitlines()
    message = " ".join(
        line.strip() for line in lines if line.strip() and not line.startswith("=")
    )
    return SmiResult(proc.returncode == 0, proc.returncode, message, proc.stdout)


def smi_json(args: list[str]) -> dict[int, dict]:
    """The per-GPU mappings of a --json query, keyed by GPU id; empty when
    rocm-smi fails."""
    result = run_smi([*args, "--json"])
    if not result.ok:
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    cards = {}
    for key, value in data.items():
        match = CARD.fullmatch(key)
        if match and isinstance(value, dict):
            cards[int(match.group(1))] = value
    return cards


def read_use() -> dict[int, float]:
    """The GPU use in percent of every GPU."""
    use = {}
    for gpu, values in smi_json(["--showuse"]).items():
        try:
            use[gpu] = float(values["GPU use (%)"])
        except (KeyError, ValueError):
            continue
    return use


def read_perf_level(gpu: int) -> str | None:
    values = smi_json(["-d", str(gpu), "--showperflevel"]).get(gpu, {})
    return values.get("Performance Level")


def read_max_sclk(gpu: int) -> int | None:
    values = smi_json(["-d", str(gpu), "--showsclkrange"]).get(gpu, {})
    clocks = [int(value) for value in MHZ.findall(str(values.get("Valid sclk range")))]
    return max(clocks, default=None)


def sample_clocks(gpu: int) -> dict:
    """The current sclk and mclk in MHz, best effort: a clock that cannot be
    read is left out."""
    values = smi_json(["-d", str(gpu), "--showclocks"]).get(gpu, {})
    sample = {}
    keys = (("sclk clock speed:", "sclkMHz"), ("mclk clock speed:", "mclkMHz"))
    for key, name in keys:
        match = MHZ.search(str(values.get(key)))
        if match:
            sample[name] = int(match.group(1))
    return sample


def parse_gpu_list(spec: str) -> list[int]:
    try:
        gpus = [int(token) for token in spec.split(",")]
    except ValueError:
        raise SweepAbort(f"--gpus must be auto or a comma-separated id list: {spec!r}")
    return list(dict.fromkeys(gpus))


def select_gpus(spec: str, max_busy: float, allow_busy: bool) -> list[int]:
    """The GPUs to time on: with auto the ones below max_busy percent use, else
    the listed ones, which must be idle unless allow_busy."""
    use = read_use()
    if not use:
        raise SweepAbort("cannot read the GPU use from rocm-smi")
    if spec == "auto":
        gpus = [gpu for gpu, percent in sorted(use.items()) if percent < max_busy]
        if not gpus:
            raise SweepAbort(f"no GPU is below {max_busy}% use: {use}")
        return gpus
    gpus = parse_gpu_list(spec)
    unknown = [gpu for gpu in gpus if gpu not in use]
    if unknown:
        raise SweepAbort(f"GPU(s) {unknown} are not reported by rocm-smi")
    busy = [gpu for gpu in gpus if use[gpu] >= max_busy]
    if busy and not allow_busy:
        raise SweepAbort(
            f"GPU(s) {busy} are at least {max_busy}% busy (use {use}); "
            "pass --allow-busy to time on them anyway"
        )
    return gpus


@dataclass
class GpuLock:
    """The clock lock of one GPU: how it was locked, if it was, and why not."""

    gpu: int
    prior_level: str | None
    mode: str | None = None
    failures: list[str] = field(default_factory=list)
    restored: bool = False

    @property
    def locked(self) -> bool:
        return self.mode is not None

    def record(self) -> dict:
        return {
            "status": "locked" if self.locked else "unavailable",
            "mode": self.mode,
            "priorPerfLevel": self.prior_level,
            "reason": "; ".join(self.failures) or None,
        }


def raise_exit(signum, frame) -> None:
    """Turns SIGTERM into a normal exit so that the finally clauses run."""
    sys.exit(128 + signum)


class GpuStabilizer:
    """Locks the clocks of the GPUs for the timed stage and restores them on
    exit, on SIGINT and SIGTERM and from an atexit handler, only for the GPUs
    where a lock attempt succeeded (a failed lock never restores anything).

    Locking is opt-in through require_stable (--require-stable-clocks); when it
    is not requested nothing is attempted, rocm-smi is never invoked and no
    elevated rights are ever needed.

    The lock tries the performance determinism mode with the highest sclk,
    then the performance level high, which holds sclk and mclk at their
    maximum; both need rights that an ordinary user may lack, in which case
    the sweep aborts.
    """

    def __init__(self, gpus: list[int], require_stable: bool):
        self.gpus = gpus
        self.require_stable = require_stable
        self.locks: dict[int, GpuLock] = {}
        self.previous_sigterm = None

    def __enter__(self):
        if not self.gpus:
            return self
        atexit.register(self.restore)
        self.previous_sigterm = signal.signal(signal.SIGTERM, raise_exit)
        try:
            self.lock_all()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def close(self) -> None:
        self.restore()
        atexit.unregister(self.restore)
        if self.previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self.previous_sigterm)
            self.previous_sigterm = None

    def lock_all(self) -> None:
        if not self.require_stable:
            # Not even a read: rocm-smi must never be invoked unless requested.
            for gpu in self.gpus:
                self.locks[gpu] = GpuLock(gpu, prior_level=None)
            return
        for gpu in self.gpus:
            self.locks[gpu] = self.lock(gpu)
        unavailable = [gpu for gpu, lock in self.locks.items() if not lock.locked]
        if unavailable:
            raise SweepAbort(f"cannot lock the clocks of GPU(s) {unavailable}")

    def lock(self, gpu: int) -> GpuLock:
        lock = GpuLock(gpu, read_perf_level(gpu))
        sclk = read_max_sclk(gpu)
        attempts = []
        if sclk is not None:
            attempts.append(("perfdeterminism", ["--setperfdeterminism", str(sclk)]))
        attempts.append(("perflevel-high", ["--setperflevel", "high"]))
        for mode, args in attempts:
            result = run_smi(["-d", str(gpu), *args])
            refused = not result.ok or REFUSAL.search(result.message)
            if not refused:
                lock.mode = mode
                log(f"GPU {gpu}: clocks locked with {mode}")
                return lock
            lock.failures.append(f"{mode}: rc={result.rc} {result.message[:200]}")
        log(
            f"warning: GPU {gpu}: clock lock unavailable ({'; '.join(lock.failures)}); "
            "continuing with warmup, LLC flush, min-of-N and interleaved refinement"
        )
        return lock

    def restore(self) -> None:
        """Undoes the successful locks; safe to call more than once."""
        for gpu, lock in reversed(list(self.locks.items())):
            if not lock.locked or lock.restored:
                continue
            lock.restored = True
            if lock.mode == "perfdeterminism":
                args = ["--resetperfdeterminism"]
            elif lock.prior_level is None:
                # Without a known prior level there is nothing safe to restore
                # to; the user must reset the level by hand.
                log(
                    f"warning: GPU {gpu}: the prior performance level is unknown, "
                    f"not restoring; run rocm-smi -d {gpu} --setperflevel auto"
                )
                continue
            else:
                args = ["--setperflevel", lock.prior_level]
            result = run_smi(["-d", str(gpu), *args])
            if result.ok:
                log(f"GPU {gpu}: clocks restored ({' '.join(args)})")
                continue
            log(
                f"warning: GPU {gpu}: restoring the clocks failed (rc={result.rc} "
                f"{result.message[:200]}); run rocm-smi -d {gpu} {' '.join(args)}"
            )

    def summary(self) -> dict:
        records = {gpu: lock.record() for gpu, lock in self.locks.items()}
        statuses = {record["status"] for record in records.values()}
        if not statuses:
            return {"status": "not applicable"}
        status = statuses.pop() if len(statuses) == 1 else "partial"
        reasons = sorted({r["reason"] for r in records.values() if r["reason"]})
        return {
            "status": status,
            "reason": "; ".join(reasons) or None,
            "perGpu": records,
        }


def git_head(directory: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))
