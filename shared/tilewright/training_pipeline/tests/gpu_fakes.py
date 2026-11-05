# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A fake sysfs for the GPU lease, watch and platform tests (KFD topology
and per-process memory, DRM devices, the compute partitions of a card,
CPUs), a fake `/proc/stat` whose CPUs are as busy as a test says
(`FakeStat`), a process outside the test's process tree and one in it
that names a bench stage's script. Every value is made up."""
from __future__ import annotations

import contextlib
import os
import shutil
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

# The two partitions of a card in DPX mode (`partition_card`): KFD gives
# partition 1 the card's address with 1 as the function.
DPX = (
    FakeGpu(2, 2001, 0x0D00, 0x5555, 160),
    FakeGpu(3, 2002, 0x0D01, 0x6666, 161),
)
DPX_HOST = (GPUS[0], *DPX)


IDLE_CARD_BYTES = 1 << 20


def kfd_root(sys_root: Path) -> Path:
    return Path(sys_root) / "class" / "kfd" / "kfd"


def pci_dir(sys_root: Path, gpu: FakeGpu) -> Path:
    """The PCI device of `gpu`'s address."""
    return Path(sys_root) / "bus" / "pci" / "devices" / gpu.bdf


def set_card_used(sys_root: Path, gpu: FakeGpu, used: int) -> None:
    """The VRAM in use on `gpu` as its PCI device reports it."""
    d = pci_dir(sys_root, gpu)
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


def write_strict_sysfs(sys_root: Path, gpus: Sequence[FakeGpu] = GPUS) -> Path:
    """`write_sysfs` of `gpus`, each a gfx1250 of ASIC revision 0, which
    loads the library subtree gfx1250-strict with HSA_DISABLE_GFX12_STRICT=0
    in a run's environment (lib/arch.py): the host of a run of that arch,
    whose preflight reads from each device's KFD node which subtree it
    loads."""
    return write_sysfs(sys_root, gpus, isa=[("gfx1250", 0)] * len(gpus))


def drive_with_amdgpu(sys_root: Path, gpu: FakeGpu) -> None:
    """Bind the PCI device of `gpu`'s address to the amdgpu driver."""
    driver = Path(sys_root) / "bus" / "pci" / "drivers" / "amdgpu"
    driver.mkdir(parents=True, exist_ok=True)
    (pci_dir(sys_root, gpu) / "driver").symlink_to(driver)


def partition_card(
    sys_root: Path, partitions: Sequence[FakeGpu], *, other_function: bool = True
) -> None:
    """Lay out the GPUs `partitions` that `write_sysfs` wrote, on one PCI
    slot with the first one's address the card's, as amdgpu lays out the
    compute partitions of a card: the card's PCI device, which amdgpu
    drives, holds what `write_sysfs` gave the first GPU's DRM device, and
    the first render node leads to it; every other partition's render node
    leads to a platform device of its own without those files, and the
    address KFD gives that partition names a function of the slot that is
    a device of another kind (`other_function`) or none."""
    root = Path(sys_root)
    card = pci_dir(root, partitions[0])
    drive_with_amdgpu(root, partitions[0])
    first = root / "class" / "drm" / f"renderD{partitions[0].render_minor}" / "device"
    for entry in list(first.iterdir()):
        entry.rename(card / entry.name)
    first.rmdir()
    first.symlink_to(card)
    for i, g in enumerate(partitions[1:]):
        render = root / "class" / "drm" / f"renderD{g.render_minor}" / "device"
        shutil.rmtree(render)
        platform = root / "devices" / "platform" / f"amdgpu_xcp_{i}"
        platform.mkdir(parents=True)
        render.symlink_to(platform)
        shutil.rmtree(pci_dir(root, g))
        if other_function:
            pci_dir(root, g).mkdir()
            (pci_dir(root, g) / "class").write_text("0x020000\n")


def write_dpx_sysfs(sys_root: Path, *, other_function: bool = True, **kw) -> Path:
    """`write_sysfs` of DPX_HOST (`kw` as it takes them): GPU 0, which is no
    partition and whose PCI device amdgpu drives, and GPUs 1 and 2, the
    partitions of a card in DPX mode (`partition_card`), each node with 3
    XCCs unless `xcc` says otherwise."""
    kw.setdefault("xcc", [(3, "SPX"), (3, "DPX"), (3, "DPX")])
    root = write_sysfs(sys_root, DPX_HOST, **kw)
    drive_with_amdgpu(root, GPUS[0])
    partition_card(root, DPX, other_function=other_function)
    return root


def set_memory(sys_root: Path, pid: int, used: Mapping[int, int]) -> None:
    """KFD's per-process files of `pid`: one per GPU of `GPUS` and per
    gpu_id `used` names, with `used` bytes on the gpu_ids it names and 0
    elsewhere."""
    d = kfd_root(sys_root) / "proc" / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    for gpu_id in sorted({g.gpu_id for g in GPUS} | set(used)):
        (d / f"vram_{gpu_id}").write_text(f"{int(used.get(gpu_id, 0))}\n")


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


@contextlib.contextmanager
def bench_stage(script: str, seconds: int = 300) -> Iterator[Tuple[int, int]]:
    """(pid, its child's pid) of a sleeping child of this process whose
    command line names `script`, as a bench stage's does, and of its own
    sleeping child, as a bench of that stage."""
    code = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen(['sleep', '{int(seconds)}'])\n"
        "print(p.pid, flush=True)\n"
        "p.wait()\n"
    )
    stage = subprocess.Popen(
        [sys.executable, "-c", code, script],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        yield stage.pid, int(stage.stdout.readline())
    finally:
        try:
            os.killpg(stage.pid, 9)
        except ProcessLookupError:
            pass
        stage.wait()
        stage.stdout.close()


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
