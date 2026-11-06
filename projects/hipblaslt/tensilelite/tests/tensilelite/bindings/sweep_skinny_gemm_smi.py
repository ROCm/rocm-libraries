# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""amd-smi helpers of the skinny GEMM sweep: GPU selection, clock sampling and
the clock lock around the timed stage.

Nothing here touches HIP, so the main process of the sweep can use it. The GPU
ids are the indices of the amd-smi processor handles.
"""

import atexit
import importlib
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field

SMI_TIMEOUT_S = 30
PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"
# Where the ROCm installs ship the amdsmi python package when it is not on the
# path of the interpreter.
AMDSMI_SEARCH_DIRS = ("share/amd_smi",)


class SweepAbort(Exception):
    """A condition the user must fix; reported without a traceback."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class SmiResult:
    ok: bool
    message: str


class AmdSmi:
    """The amdsmi python bindings, initialized once per process."""

    def __init__(self):
        self.module = load_amdsmi_module()
        self.module.amdsmi_init()
        self.pid = os.getpid()
        self.handles = self.module.amdsmi_get_processor_handles()
        atexit.register(self.shut_down)

    def shut_down(self) -> None:
        # A forked child must not shut down the library of its parent.
        if os.getpid() != self.pid:
            return
        try:
            self.module.amdsmi_shut_down()
        except Exception:
            pass

    def handle(self, gpu: int):
        if not 0 <= gpu < len(self.handles):
            raise SweepAbort(f"GPU {gpu} is not reported by amd-smi")
        return self.handles[gpu]


def load_amdsmi_module():
    try:
        return importlib.import_module("amdsmi")
    except ImportError:
        pass
    roots = [os.environ.get("ROCM_PATH"), "/opt/rocm"]
    for root in filter(None, roots):
        for directory in AMDSMI_SEARCH_DIRS:
            path = os.path.join(root, directory)
            if os.path.isdir(os.path.join(path, "amdsmi")):
                sys.path.append(path)
                return importlib.import_module("amdsmi")
    raise SweepAbort(
        "cannot import the amdsmi python package; add <rocm>/share/amd_smi "
        "to PYTHONPATH"
    )


_smi: AmdSmi | None = None


def get_smi() -> AmdSmi:
    """The amd-smi bindings of this process, initialized on first use."""
    global _smi
    if _smi is None or _smi.pid != os.getpid():
        try:
            _smi = AmdSmi()
        except SweepAbort:
            raise
        except Exception as err:
            raise SweepAbort(f"cannot initialize amd-smi: {err}")
    return _smi


def run_smi(action, *args) -> SmiResult:
    """Calls an amdsmi function; never raises. The message is the failure."""
    try:
        action(*args)
    except Exception as err:
        return SmiResult(False, f"{type(err).__name__}: {err}")
    return SmiResult(True, "")


def read_use() -> dict[int, float]:
    """The GPU use in percent of every GPU."""
    smi = get_smi()
    use = {}
    for gpu, handle in enumerate(smi.handles):
        try:
            activity = smi.module.amdsmi_get_gpu_activity(handle)
            use[gpu] = float(activity["gfx_activity"])
        except Exception:
            continue
    return use


def read_perf_level(gpu: int) -> str | None:
    smi = get_smi()
    try:
        level = smi.module.amdsmi_get_gpu_perf_level(smi.handle(gpu))
    except Exception:
        return None
    return str(level).removeprefix(PERF_LEVEL_PREFIX)


def read_clock(gpu: int, clock_type: str) -> dict | None:
    smi = get_smi()
    try:
        kind = getattr(smi.module.AmdSmiClkType, clock_type)
        return smi.module.amdsmi_get_clock_info(smi.handle(gpu), kind)
    except Exception:
        return None


def read_max_sclk(gpu: int) -> int | None:
    info = read_clock(gpu, "GFX")
    if not info or not isinstance(info.get("max_clk"), int):
        return None
    return info["max_clk"]


def sample_clocks(gpu: int) -> dict:
    """The current sclk and mclk in MHz, best effort: a clock that cannot be
    read is left out."""
    sample = {}
    for clock_type, name in (("GFX", "sclkMHz"), ("MEM", "mclkMHz")):
        info = read_clock(gpu, clock_type)
        if info and isinstance(info.get("clk"), int):
            sample[name] = info["clk"]
    return sample


def set_perf_level(gpu: int, level: str) -> SmiResult:
    smi = get_smi()
    try:
        value = smi.module.AmdSmiDevPerfLevel[level.upper()]
    except KeyError:
        return SmiResult(False, f"unknown performance level {level!r}")
    return run_smi(smi.module.amdsmi_set_gpu_perf_level, smi.handle(gpu), value)


def set_perf_determinism(gpu: int, sclk_mhz: int) -> SmiResult:
    smi = get_smi()
    return run_smi(
        smi.module.amdsmi_set_gpu_perf_determinism_mode, smi.handle(gpu), sclk_mhz
    )


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
        raise SweepAbort("cannot read the GPU use from amd-smi")
    if spec == "auto":
        gpus = [gpu for gpu, percent in sorted(use.items()) if percent < max_busy]
        if not gpus:
            raise SweepAbort(f"no GPU is below {max_busy}% use: {use}")
        return gpus
    gpus = parse_gpu_list(spec)
    unknown = [gpu for gpu in gpus if gpu not in use]
    if unknown:
        raise SweepAbort(f"GPU(s) {unknown} are not reported by amd-smi")
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
    is not requested nothing is attempted, amd-smi is never invoked and no
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
            # Not even a read: amd-smi must never be invoked unless requested.
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
            attempts.append(("perfdeterminism", lambda: set_perf_determinism(gpu, sclk)))
        attempts.append(("perflevel-high", lambda: set_perf_level(gpu, "high")))
        for mode, attempt in attempts:
            result = attempt()
            if result.ok:
                lock.mode = mode
                log(f"GPU {gpu}: clocks locked with {mode}")
                return lock
            lock.failures.append(f"{mode}: {result.message[:200]}")
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
                level = "auto"
            elif lock.prior_level is None:
                # Without a known prior level there is nothing safe to restore
                # to; the user must reset the level by hand.
                log(
                    f"warning: GPU {gpu}: the prior performance level is unknown, "
                    f"not restoring; run amd-smi reset -g {gpu} --clocks"
                )
                continue
            else:
                level = lock.prior_level
            result = set_perf_level(gpu, level)
            if result.ok:
                log(f"GPU {gpu}: clocks restored (performance level {level})")
                continue
            log(
                f"warning: GPU {gpu}: restoring the clocks failed "
                f"({result.message[:200]}); run amd-smi reset -g {gpu} --clocks"
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
