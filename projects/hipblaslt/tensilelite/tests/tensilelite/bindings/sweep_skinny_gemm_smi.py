# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""amd-smi helpers of the skinny GEMM sweep: GPU selection, clock and throttle
sampling and the optional clock lock held for the whole run.

Nothing here touches HIP, so the main process of the sweep can use it. The GPU
ids are HIP ordinals, the numbering of HIP_VISIBLE_DEVICES that the workers
use, and every amd-smi call maps the ordinal to its amd-smi handle by PCI
address (map_hip_ordinals); amd-smi's own device indices can differ from the HIP
ordinals, as they do on some machines. Without a mapping the ids are the
indices of the amd-smi processor handles. Reading (use, process list, clocks,
metrics, PCI address) needs no elevated rights; only setting the performance
level or determinism mode, which happens only with --require-stable-clocks,
may.
"""

import atexit
import importlib
import multiprocessing
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

SMI_TIMEOUT_S = 30
# The longest the probe of the HIP devices may take.
PROBE_TIMEOUT_S = 120
PERF_LEVEL_PREFIX = "AMDSMI_DEV_PERF_LEVEL_"
# The use of a GPU is the mean of BUSY_SAMPLES readings BUSY_INTERVAL_S apart.
BUSY_SAMPLES = 5
BUSY_INTERVAL_S = 0.2
# Period of the clock and throttle sampling while a case is timed.
SAMPLE_INTERVAL_S = 0.25
DETERMINISM_LEVEL = "DETERMINISM"
HIGH_LEVEL = "HIGH"
AUTO_LEVEL = "AUTO"
UNKNOWN_LEVEL = "UNKNOWN"
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
        self.by_address = self.handles_by_address()
        atexit.register(self.shut_down)

    def handles_by_address(self) -> dict:
        """The handles keyed by (domain, bus, device); a handle whose address
        cannot be read is left out."""
        handles = {}
        for handle in self.handles:
            try:
                key = parse_pci_address(self.module.amdsmi_get_gpu_device_bdf(handle))
            except Exception:
                continue
            if key is not None:
                handles[key] = handle
        return handles

    def shut_down(self) -> None:
        # A forked child must not shut down the library of its parent.
        if os.getpid() != self.pid:
            return
        try:
            self.module.amdsmi_shut_down()
        except Exception:
            pass

    def handle(self, gpu: int):
        """The handle of the HIP ordinal gpu, or of the amd-smi index gpu when
        no mapping is set."""
        if _hip_ordinals is None:
            if not 0 <= gpu < len(self.handles):
                raise SweepAbort(f"GPU {gpu} is not reported by amd-smi")
            return self.handles[gpu]
        handle = self.by_address.get(parse_pci_address(_hip_ordinals.get(gpu)))
        if handle is None:
            raise SweepAbort(f"{describe(gpu)} is not reported by amd-smi")
        return handle


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
# HIP ordinal -> PCI address; None means the ids are amd-smi indices.
_hip_ordinals: dict[int, str] | None = None
# The sampler thread of a worker may race the first use of the main thread.
_smi_lock = threading.Lock()


def get_smi() -> AmdSmi:
    """The amd-smi bindings of this process, initialized on first use."""
    global _smi
    with _smi_lock:
        if _smi is not None and _smi.pid == os.getpid():
            return _smi
        try:
            _smi = AmdSmi()
        except SweepAbort:
            raise
        except Exception as err:
            raise SweepAbort(f"cannot initialize amd-smi: {err}")
        return _smi


def use_hip_ordinals(mapping: dict[int, str] | None) -> None:
    """Makes the GPU ids of this process HIP ordinals with the given PCI
    addresses; None goes back to the amd-smi indices."""
    global _hip_ordinals
    _hip_ordinals = None if mapping is None else dict(mapping)


def describe(gpu: int) -> str:
    if _hip_ordinals is None:
        return f"GPU {gpu}"
    return f"HIP device {gpu} ({_hip_ordinals.get(gpu, 'unknown address')})"


def probe_target(conn, function=None) -> None:
    """The body of the probe child: sends ("ok", result) or ("error", text)."""
    try:
        if function is None:
            from sweep_skinny_gemm_worker import hip_pci_addresses as function
        conn.send(("ok", function()))
    except BaseException as err:
        conn.send(("error", f"{type(err).__name__}: {err}"))


def stop_process(process, grace_s: float) -> None:
    process.join(grace_s)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def probe_hip_ordinals(timeout: float = PROBE_TIMEOUT_S, function=None) -> dict[int, str]:
    """The PCI address of every HIP ordinal, probed in a spawned child so that
    this process never initializes HIP. The timeout bounds the whole probe: a
    hung child is terminated, and killed if it ignores that."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=probe_target, args=(child, function), daemon=True)
    process.start()
    child.close()
    answered = False
    try:
        if not parent.poll(timeout):
            raise TimeoutError(f"no answer within {timeout} s")
        try:
            status, value = parent.recv()
        except EOFError:
            raise RuntimeError("the probe process died")
        answered = True
    finally:
        stop_process(process, 5 if answered else 0)
        parent.close()
    if status != "ok":
        raise RuntimeError(value)
    return value


def map_hip_ordinals() -> None:
    """Maps the HIP ordinals to the amd-smi devices by PCI address; if the
    probe fails, warns and keeps the amd-smi indices, which the check of the
    ready message still verifies."""
    try:
        mapping = probe_hip_ordinals()
    except Exception as err:
        log(
            f"warning: cannot probe the HIP devices ({type(err).__name__}: "
            f"{err}); using the amd-smi numbering for the GPU ids"
        )
        return
    use_hip_ordinals(mapping)


def run_smi(action, *args) -> SmiResult:
    """Calls an amdsmi function; never raises. The message is the failure."""
    try:
        action(*args)
    except Exception as err:
        return SmiResult(False, f"{type(err).__name__}: {err}")
    return SmiResult(True, "")


def read_activity(smi: AmdSmi, handle) -> float | None:
    """One gfx activity reading in percent; None if it is not a number or the
    GPU has no handle."""
    if handle is None:
        return None
    try:
        value = smi.module.amdsmi_get_gpu_activity(handle)["gfx_activity"]
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def gpu_handles(smi: AmdSmi) -> dict:
    """The handle of every GPU id: the HIP ordinals of the mapping, else the
    amd-smi indices. An ordinal that amd-smi does not report has no handle."""
    if _hip_ordinals is None:
        return dict(enumerate(smi.handles))
    handles = {}
    for gpu in sorted(_hip_ordinals):
        try:
            handles[gpu] = smi.handle(gpu)
        except SweepAbort:
            handles[gpu] = None
    return handles


def read_use(sleep: Callable[[float], None] = time.sleep) -> dict[int, float | None]:
    """The mean GPU use in percent of every GPU id over about a second; None for
    a GPU without a single numeric reading (amd-smi reports N/A)."""
    smi = get_smi()
    handles = gpu_handles(smi)
    readings: dict[int, list[float]] = {gpu: [] for gpu in handles}
    for sample in range(BUSY_SAMPLES):
        if sample:
            sleep(BUSY_INTERVAL_S)
        for gpu, handle in handles.items():
            value = read_activity(smi, handle)
            if value is not None:
                readings[gpu].append(value)
    return {
        gpu: statistics.fmean(values) if values else None
        for gpu, values in readings.items()
    }


def process_pid(process) -> int | None:
    pid = process.get("pid") if isinstance(process, dict) else None
    return pid if isinstance(pid, int) else None


def read_process_counts() -> dict[int, int]:
    """The number of processes other than this one using every GPU; a GPU whose
    list cannot be read counts as 0, its use is judged by the activity."""
    smi = get_smi()
    counts = {}
    for gpu, handle in gpu_handles(smi).items():
        try:
            processes = smi.module.amdsmi_get_gpu_process_list(handle)
        except Exception:
            processes = []
        # An entry without a pid (older amdsmi) counts as another process on
        # purpose: it cannot be shown to be this one.
        counts[gpu] = sum(1 for p in processes if process_pid(p) != os.getpid())
    return counts


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


def read_throttle_status(gpu: int) -> int | None:
    smi = get_smi()
    try:
        metrics = smi.module.amdsmi_get_gpu_metrics_info(smi.handle(gpu))
    except Exception:
        return None
    for key in ("throttle_status", "indep_throttle_status"):
        value = metrics.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def sample_metrics(gpu: int) -> dict:
    """The current sclk and mclk in MHz and the throttle status, best effort and
    never raising: a value that cannot be read is left out. The throttle status
    is a bit mask whose bits differ between ASICs; a nonzero value only tells
    that some throttler was active."""
    sample = {}
    try:
        for clock_type, name in (("GFX", "sclkMHz"), ("MEM", "mclkMHz")):
            info = read_clock(gpu, clock_type)
            if info and isinstance(info.get("clk"), int):
                sample[name] = info["clk"]
        status = read_throttle_status(gpu)
    except Exception:
        return sample
    if status is not None:
        sample["throttleStatus"] = status
    return sample


def summarize_samples(samples: list[dict]) -> dict:
    """The clock statistics of the samples and how many saw a throttler."""
    sclk = [s["sclkMHz"] for s in samples if "sclkMHz" in s]
    summary = {"samples": len(samples)}
    if sclk:
        summary["sclkMinMHz"] = min(sclk)
        summary["sclkMeanMHz"] = round(statistics.fmean(sclk), 1)
        summary["sclkMaxMHz"] = max(sclk)
    summary["throttledSamples"] = sum(1 for s in samples if s.get("throttleStatus"))
    return summary


class ClockSampler:
    """Samples the clocks and the throttle status of a GPU from a daemon thread
    while the body of the with statement runs."""

    def __init__(
        self,
        gpu: int,
        sampler: Callable[[int], dict] = sample_metrics,
        interval: float = SAMPLE_INTERVAL_S,
    ):
        self.gpu = gpu
        self.sampler = sampler
        self.interval = interval
        self.samples: list[dict] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        # The first sample is taken at once, so that a short body is sampled.
        while True:
            self.samples.append(self.sampler(self.gpu))
            if self.stopped.wait(self.interval):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exception) -> None:
        self.stopped.set()
        # A hung amd-smi call must not block the case; the thread is a daemon.
        self.thread.join(SMI_TIMEOUT_S)

    def summary(self) -> dict:
        return summarize_samples(self.samples)


def device_bdf(gpu: int) -> str | None:
    """The PCI address of the amd-smi device, None if it cannot be read."""
    smi = get_smi()
    try:
        bdf = smi.module.amdsmi_get_gpu_device_bdf(smi.handle(gpu))
    except Exception:
        return None
    return bdf if isinstance(bdf, str) else None


def format_pci_address(domain: int, bus: int, device: int) -> str:
    """The address in the format of amd-smi; the function is always 0."""
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def parse_pci_address(address: str | None) -> tuple[int, int, int] | None:
    """The (domain, bus, device) of dddd:bb:dd.f; the function is ignored."""
    pattern = r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.[0-9]+"
    match = re.fullmatch(pattern, address or "")
    if match is None:
        return None
    domain, bus, device = (int(group, 16) for group in match.groups())
    return domain, bus, device


def check_same_device(gpu: int, hip_address: str | None) -> None:
    """Aborts if the amd-smi device gpu is not the HIP device gpu; warns and
    goes on if either PCI address is unavailable."""
    smi_address = device_bdf(gpu)
    smi_id = parse_pci_address(smi_address)
    hip_id = parse_pci_address(hip_address)
    if smi_id is None or hip_id is None:
        log(
            f"warning: GPU {gpu}: cannot compare the amd-smi PCI address "
            f"({smi_address}) with the HIP one ({hip_address}); assuming the "
            "same numbering"
        )
        return
    if smi_id != hip_id:
        raise SweepAbort(
            f"GPU {gpu}: amd-smi device {smi_address} is not HIP device {gpu} "
            f"({hip_address}); the ids are not the same numbering, pick ids "
            "that name the same device in both or make the enumerations agree"
        )


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


def busy_reasons(use: dict[int, float | None], max_busy: float) -> dict[int, str]:
    """Why each busy GPU is busy; a GPU of unknown use counts as busy."""
    processes = read_process_counts()
    reasons = {}
    for gpu, percent in sorted(use.items()):
        count = processes.get(gpu, 0)
        if percent is None:
            reasons[gpu] = f"{describe(gpu)} has an unknown use (amd-smi reports N/A)"
        elif percent >= max_busy:
            reasons[gpu] = f"{describe(gpu)} is {percent:.1f}% busy"
        elif count:
            reasons[gpu] = f"{describe(gpu)} has {count} processes"
    return reasons


def select_gpus(spec: str, max_busy: float, allow_busy: bool) -> list[int]:
    """The GPUs to time on: with auto the idle ones, else the listed ones, which
    must be idle unless allow_busy. A GPU is busy when its mean use is at least
    max_busy percent, when a process uses it, or when its use is unknown."""
    use = read_use()
    if not use:
        raise SweepAbort("cannot read the GPU use from amd-smi")
    reasons = busy_reasons(use, max_busy)
    if spec == "auto":
        gpus = [gpu for gpu in sorted(use) if gpu not in reasons]
        if not gpus:
            raise SweepAbort(f"no GPU is idle: {'; '.join(reasons.values())}")
        return gpus
    gpus = parse_gpu_list(spec)
    unknown = [gpu for gpu in gpus if gpu not in use]
    if unknown:
        raise SweepAbort(f"GPU(s) {unknown} are not reported (by HIP or amd-smi)")
    busy = [reasons[gpu] for gpu in gpus if gpu in reasons]
    if busy and not allow_busy:
        raise SweepAbort(
            f"{'; '.join(busy)}; pass --allow-busy to time on them anyway"
        )
    for reason in busy:
        log(f"warning: {reason}; timing on it anyway")
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


def restore_level(lock: GpuLock) -> bool:
    """Sets the prior performance level of the lock and verifies it by read-back;
    on failure falls back to a reset to auto. True only if the prior level is
    back."""
    gpu, prior = lock.gpu, lock.prior_level
    result = set_perf_level(gpu, prior)
    reason = result.message[:200]
    if result.ok:
        actual = read_perf_level(gpu)
        if actual == prior:
            log(f"GPU {gpu}: clocks restored (performance level {prior})")
            return True
        reason = f"read back {actual}"
    if prior == AUTO_LEVEL:
        log(
            f"warning: GPU {gpu}: restoring {prior} failed ({reason}); run "
            f"amd-smi reset -g {gpu} --clocks"
        )
        return False
    log(f"warning: GPU {gpu}: restoring {prior} failed ({reason}); reset to auto")
    if not set_perf_level(gpu, AUTO_LEVEL).ok:
        log(
            f"warning: GPU {gpu}: the reset failed; run "
            f"amd-smi reset -g {gpu} --clocks"
        )
    return False


class GpuStabilizer:
    """Locks the clocks of the GPUs for the whole run, builds included, and
    restores them on exit, on SIGINT and SIGTERM and from an atexit handler,
    only for the GPUs where a lock attempt succeeded (a failed lock never
    restores anything). A forked child never restores.

    Locking is opt-in through require_stable (--require-stable-clocks). Without
    it nothing is set and no elevated rights are needed; amd-smi is still read
    for the GPU use and for the clock samples of the timed cases.

    The lock refuses a GPU whose prior performance level cannot be read, then
    tries the performance determinism mode with the highest sclk and the
    performance level high, which holds sclk and mclk at their maximum; each
    set is verified by reading the level back. Both need rights that an
    ordinary user may lack, in which case the sweep aborts. The restore sets
    the prior level back, also verified, and falls back to auto.
    """

    def __init__(self, gpus: list[int], require_stable: bool):
        self.gpus = gpus
        self.require_stable = require_stable
        self.locks: dict[int, GpuLock] = {}
        self.previous_sigterm = None
        self.pid: int | None = None

    def __enter__(self):
        if not self.gpus:
            return self
        self.pid = os.getpid()
        if self.require_stable:
            # Initialized first: atexit runs the handlers last in first out, so
            # the restore must come after the registration of the shut down.
            get_smi()
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
        # A failed restore stays registered, so that exit retries it once more.
        if all(lock.restored or not lock.locked for lock in self.locks.values()):
            atexit.unregister(self.restore)
        if self.previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self.previous_sigterm)
            self.previous_sigterm = None

    def lock_all(self) -> None:
        if not self.require_stable:
            # Nothing is set, so there is nothing to read either.
            for gpu in self.gpus:
                self.locks[gpu] = GpuLock(gpu, prior_level=None)
            return
        for gpu in self.gpus:
            self.lock(gpu)
        unavailable = [gpu for gpu, lock in self.locks.items() if not lock.locked]
        if unavailable:
            raise SweepAbort(f"cannot lock the clocks of GPU(s) {unavailable}")

    def lock(self, gpu: int) -> GpuLock:
        prior = read_perf_level(gpu)
        lock = GpuLock(gpu, prior)
        # Registered before the first set, so that an interrupt in the middle of
        # the lock still leaves something for restore to undo.
        self.locks[gpu] = lock
        if prior is None or prior == UNKNOWN_LEVEL:
            lock.failures.append("cannot read the prior performance level")
            self.log_unavailable(lock)
            return lock
        sclk = read_max_sclk(gpu)
        attempts = []
        if sclk is not None:
            determinism = partial(set_perf_determinism, gpu, sclk)
            attempts.append(("perfdeterminism", DETERMINISM_LEVEL, determinism))
        high = partial(set_perf_level, gpu, HIGH_LEVEL)
        attempts.append(("perflevel-high", HIGH_LEVEL, high))
        for mode, expected, attempt in attempts:
            if self.try_mode(lock, mode, expected, attempt):
                log(f"GPU {gpu}: clocks locked with {mode}")
                return lock
        self.log_unavailable(lock)
        return lock

    def try_mode(self, lock: GpuLock, mode: str, expected: str, attempt) -> bool:
        """One lock attempt, verified by reading the level back; a set that did
        not take effect is undone at once. The mode is recorded before the set,
        so that an interrupt during or after it is undone by restore."""
        lock.mode = mode
        result = attempt()
        if not result.ok:
            lock.mode = None
            lock.failures.append(f"{mode}: {result.message[:200]}")
            return False
        actual = read_perf_level(lock.gpu)
        if actual == expected:
            return True
        lock.failures.append(f"{mode}: the level reads back {actual}, not {expected}")
        restore_level(lock)
        lock.mode = None
        return False

    def log_unavailable(self, lock: GpuLock) -> None:
        log(
            f"warning: GPU {lock.gpu}: clock lock unavailable "
            f"({'; '.join(lock.failures)}); continuing with warmup, LLC flush, "
            "the mean of N launches and interleaved refinement"
        )

    def restore(self) -> None:
        """Undoes the successful locks; safe to call more than once. A GPU whose
        restore failed is retried by the next call."""
        if os.getpid() != self.pid:
            return
        for lock in reversed(list(self.locks.values())):
            if not lock.locked or lock.restored:
                continue
            lock.restored = restore_level(lock)

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
