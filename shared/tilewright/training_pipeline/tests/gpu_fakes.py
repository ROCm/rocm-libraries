# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A fake sysfs for the GPU lease, watch and platform tests (KFD topology
and per-process memory, DRM devices, CPUs), a fake `/proc/stat` whose CPUs
are as busy as a test says (`FakeStat`), and a process outside the test's
process tree. Every value is made up."""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FakeGpu:
    node: int
    gpu_id: int
    location_id: int
    unique_id: int
    render_minor: int

    @property
    def bdf(self) -> str:
        loc = self.location_id
        return f"0000:{loc >> 8:02x}:{(loc >> 3) & 0x1F:02x}.{loc & 7}"


# Node order differs from bus order, as on real hosts.
GPUS = (
    FakeGpu(1, 1001, 0x0B00, 0x1111, 128),
    FakeGpu(2, 1002, 0x0A00, 0x2222, 136),
    FakeGpu(3, 1003, 0x0C00, 0x3333, 144),
)


IDLE_CARD_BYTES = 1 << 20


def kfd_root(sys_root: Path) -> Path:
    return Path(sys_root) / "class" / "kfd" / "kfd"


def set_card_used(sys_root: Path, gpu: FakeGpu, used: int) -> None:
    """The VRAM in use on `gpu` as its PCI device reports it."""
    d = Path(sys_root) / "bus" / "pci" / "devices" / gpu.bdf
    d.mkdir(parents=True, exist_ok=True)
    (d / "mem_info_vram_used").write_text(f"{int(used)}\n")


def isa_properties(target: str, revision: int) -> str:
    """The KFD node properties naming compiler target `target` (gfx950) and
    ASIC revision `revision`, among other capability bits."""
    digits = target[len("gfx") :]
    version = int(digits[:-2]) * 10000 + int(digits[-2], 16) * 100
    version += int(digits[-1], 16)
    return f"gfx_target_version {version}\ncapability {revision << 22 | 0x4002280}\n"


def write_sysfs(
    sys_root: Path,
    gpus: Sequence[FakeGpu] = GPUS,
    *,
    perf_level: Optional[Sequence[Optional[str]]] = None,
    power_cap_w: Optional[Sequence[Optional[float]]] = None,
    isa: Optional[Sequence[Optional[Tuple[str, int]]]] = None,
    xcc: Optional[Sequence[Optional[Tuple[int, Optional[str]]]]] = None,
    cus: Optional[Sequence[Optional[int]]] = None,
) -> Path:
    """A CPU node 0, then one KFD node per GPU with its DRM device (the
    performance level and power cap of each GPU, "auto" and 100 W unless
    given; None leaves the file out) and its PCI device, `IDLE_CARD_BYTES`
    in use. `isa` gives each GPU's (compiler target, ASIC revision); a GPU
    without one has neither in its node. `xcc` gives each GPU's (XCCs of
    its node, compute partition mode of its PCI device, None: no such
    file); a GPU without one has neither. `cus` gives each GPU's CU count;
    a GPU without one has SIMDs but no `simd_per_cu`."""
    nodes = kfd_root(sys_root) / "topology" / "nodes"
    cpu = nodes / "0"
    cpu.mkdir(parents=True)
    (cpu / "gpu_id").write_text("0\n")
    (cpu / "properties").write_text("cpu_cores_count 4\nsimd_count 0\n")
    (kfd_root(sys_root) / "proc").mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(gpus):
        d = nodes / str(g.node)
        d.mkdir()
        (d / "gpu_id").write_text(f"{g.gpu_id}\n")
        simds = (
            f"simd_count {4 * cus[i]}\nsimd_per_cu 4\n"
            if cus is not None and cus[i]
            else "simd_count 64\n"
        )
        (d / "properties").write_text(
            f"cpu_cores_count 0\n{simds}location_id {g.location_id}\n"
            f"domain 0\ndrm_render_minor {g.render_minor}\n"
            f"unique_id {g.unique_id}\n"
            + (isa_properties(*isa[i]) if isa is not None and isa[i] else "")
            + (f"num_xcc {xcc[i][0]}\n" if xcc is not None and xcc[i] else "")
        )
        dev = Path(sys_root) / "class" / "drm" / f"renderD{g.render_minor}" / "device"
        hwmon = dev / "hwmon" / f"hwmon{i}"
        hwmon.mkdir(parents=True)
        level = perf_level[i] if perf_level is not None else "auto"
        if level is not None:
            (dev / "power_dpm_force_performance_level").write_text(level + "\n")
        cap = power_cap_w[i] if power_cap_w is not None else 100.0
        if cap is not None:
            (hwmon / "power1_cap").write_text(f"{int(cap * 1e6)}\n")
            (hwmon / "power1_cap_default").write_text(f"{int(cap * 1e6)}\n")
        (dev / "vbios_version").write_text("fake-vbios\n")
        set_card_used(sys_root, g, IDLE_CARD_BYTES)
        if xcc is not None and xcc[i] and xcc[i][1] is not None:
            pci = Path(sys_root) / "bus" / "pci" / "devices" / g.bdf
            (pci / "current_compute_partition").write_text(f"{xcc[i][1]}\n")
    return Path(sys_root)


def set_memory(sys_root: Path, pid: int, used: Mapping[int, int]) -> None:
    """KFD's per-process files of `pid`: one per GPU of the topology, with
    `used` bytes on the gpu_ids it names and 0 elsewhere."""
    d = kfd_root(sys_root) / "proc" / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    for g in GPUS:
        (d / f"vram_{g.gpu_id}").write_text(f"{int(used.get(g.gpu_id, 0))}\n")


def write_cpus(sys_root: Path, n: int = 4, governor: str = "performance") -> None:
    """`n` online CPUs, CPU i and i + n/2 SMT siblings, boost on."""
    cpu = Path(sys_root) / "devices" / "system" / "cpu"
    (cpu / "cpufreq").mkdir(parents=True)
    (cpu / "smt").mkdir()
    (cpu / "online").write_text(f"0-{n - 1}\n")
    (cpu / "cpufreq" / "boost").write_text("1\n")
    (cpu / "smt" / "active").write_text("1\n")
    half = n // 2
    for c in range(n):
        d = cpu / f"cpu{c}"
        (d / "cpufreq").mkdir(parents=True)
        (d / "topology").mkdir()
        (d / "cpufreq" / "scaling_governor").write_text(governor + "\n")
        a, b = sorted((c, (c + half) % n))
        (d / "topology" / "thread_siblings_list").write_text(f"{a},{b}\n")


@contextlib.contextmanager
def outside_process(seconds: int = 300) -> Iterator[int]:
    """The pid of a sleeping process that is neither a descendant of this
    one nor in a session it started: its parent exits at once."""
    quiet = (
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
    )
    launcher = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess\n"
            f"p = subprocess.Popen(['sleep', '{int(seconds)}'], "
            f"start_new_session=True, {quiet})\n"
            "print(p.pid)\n",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    pid = int(launcher.stdout.strip())
    try:
        yield pid
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def wait_for(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class FakeStat:
    """A fake `/proc` whose `stat` counts `rate[c]` busy jiffies out of
    every 100 per CPU every few milliseconds, as other load would."""

    def __init__(self, root, cores, rate=0.0):
        self.root = root
        self.cores = list(cores)
        self.rate = {c: rate for c in self.cores}
        self.busy = {c: 0 for c in self.cores}
        self.total = {c: 0 for c in self.cores}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self.write()

    def set(self, rate, core=None):
        for c in self.cores if core is None else [core]:
            self.rate[c] = rate

    def write(self):
        rows = ["cpu  0 0 0 0 0 0 0 0 0 0"]
        for c in self.cores:
            idle = self.total[c] - self.busy[c]
            rows.append(f"cpu{c} {self.busy[c]} 0 0 {idle} 0 0 0 0 0 0")
        tmp = self.root / "stat.tmp"
        tmp.write_text("\n".join(rows) + "\nintr 0\n")
        tmp.replace(self.root / "stat")
        (self.root / "loadavg").write_text("1.50 1.00 0.50 1/100 42\n")

    def _run(self):
        while not self._stop.wait(0.005):
            for c in self.cores:
                self.total[c] += 100
                self.busy[c] += int(round(100 * self.rate[c]))
            self.write()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
