# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The platform settings a measurement depends on, recorded with it.

Per GPU, read-only from the sysfs of its PCI device (found through the DRM
render node its KFD node names): the performance level
(`power_dpm_force_performance_level`), the power cap and its default (hwmon
`power1_cap`, `power1_cap_default`, in W), the VBIOS version, the CUs
and XCCs of its KFD node and the compute partition mode of the PCI device
(`kfd.gpu_cus`, `kfd.gpu_xcc`), and the GPU's
identity (`lib/kfd.py`: physical index, KFD node, gpu_id, PCI address,
unique id, and amd-smi's index of it) and, when the session starts, its VRAM
in use and how much of it no process KFD lists holds (`vram_use`). Per host:
the amdgpu driver version, the kernel release and the ROCm version. For
CPU-timed measurements (`cpu_settings`): the CPU affinity and whether it
pins the process to fewer than the online CPUs, the cpufreq governors of
those CPUs, boost, SMT, the SMT siblings of the pinned CPUs outside the
affinity, and the load average; `CpuWindow` measures how busy those
siblings were over a stretch of time. Per session, optionally, the variables
of the benches' environment that select the library subtree and mode they
run with (`lib.bench_env.recorded_env`) and how the benches launch the
kernels they time (`lib.launch_settings.bench_record`).

A value that cannot be read is None (null in the JSON) and never fails a
check. `start_session` writes `platform.json` and raises PlatformError when
the session's GPUs disagree on a setting of `MUST_MATCH`: the performance
level, the power cap, and the CU count, partition mode and XCCs (a run
reads its device values from one of its devices), or when amd-smi and the
KFD topology disagree on the GPUs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import kfd
from .fs import write_json

PLATFORM_FILE = "platform.json"
MUST_MATCH = (
    ("perf_level", "performance level"),
    ("power_cap_w", "power cap"),
    ("n_cu", "CU count"),
    ("compute_partition", "compute partition mode"),
    ("num_xcc", "XCC count"),
)


class PlatformError(RuntimeError):
    """The session's GPUs do not share the settings a measurement needs."""


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _watts(text: Optional[str]) -> Optional[float]:
    try:
        return int(text) / 1e6 if text is not None else None
    except ValueError:
        return None


def parse_cpulist(text: Optional[str]) -> Optional[List[int]]:
    """CPUs of a sysfs list such as `0-3,8`; None when it does not parse."""
    if not text:
        return None
    out: List[int] = []
    try:
        for part in text.split(","):
            lo, _, hi = part.strip().partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        return None
    return sorted(set(out))


def format_cpulist(cpus: Iterable[int]) -> str:
    runs: List[Tuple[int, int]] = []
    for c in sorted(set(cpus)):
        if runs and c == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], c)
        else:
            runs.append((c, c))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


# ── GPU ──────────────────────────────────────────────────────────────────────


def _device_dir(gpu: Optional[kfd.Gpu]) -> Optional[Path]:
    if gpu is None or gpu.render_minor is None:
        return None
    path = kfd.sysfs_root() / "class" / "drm" / f"renderD{gpu.render_minor}" / "device"
    return path if path.is_dir() else None


def _hwmon(device_dir: Path, name: str) -> Optional[float]:
    for h in sorted(device_dir.glob("hwmon/hwmon*")):
        value = _watts(_read(h / name))
        if value is not None:
            return value
    return None


def gpu_settings(
    device: Optional[int], physical: Optional[int], gpu: Optional[kfd.Gpu]
) -> Dict[str, Any]:
    """The record of one GPU of a session: the stage's device index, the
    physical GPU, its identity and settings."""
    out: Dict[str, Any] = {"device": device, "gpu": physical}
    for key in ("kfd_node", "gpu_id", "bdf", "unique_id"):
        out[key] = None
    if gpu is not None:
        out.update(gpu.to_json())
    out["amd_smi_gpu"] = None
    d = _device_dir(gpu)
    out["perf_level"] = _read(d / "power_dpm_force_performance_level") if d else None
    out["power_cap_w"] = _hwmon(d, "power1_cap") if d else None
    out["power_cap_default_w"] = _hwmon(d, "power1_cap_default") if d else None
    out["vbios"] = _read(d / "vbios_version") if d else None
    num_xcc, partition = kfd.gpu_xcc(gpu) if gpu is not None else (None, None)
    out["n_cu"] = kfd.gpu_cus(gpu) if gpu is not None else None
    out["compute_partition"] = partition
    out["num_xcc"] = num_xcc
    return out


def host_settings() -> Dict[str, Any]:
    rocm = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    return {
        "amdgpu_driver": _read(kfd.sysfs_root() / "module" / "amdgpu" / "version"),
        "kernel": os.uname().release,
        "rocm": _read(rocm / ".info" / "version"),
    }


def mismatches(
    gpus: Sequence[Mapping[str, Any]], who: str = "the session's GPUs"
) -> List[str]:
    """One message per setting of MUST_MATCH the GPUs (`gpu_settings`
    records, `who` in the message) with a known value disagree on."""
    out: List[str] = []
    for key, label in MUST_MATCH:
        known = [g for g in gpus if g.get(key) is not None]
        if len({g[key] for g in known}) > 1:
            cards = ", ".join(
                f"GPU {g['gpu']} ({g.get('bdf') or 'no PCI address'}) {g[key]}"
                for g in known
            )
            out.append(f"{who} disagree on the {label}: {cards}")
    return out


def vram_use(gpu: Optional[kfd.Gpu]) -> Dict[str, Any]:
    """The card's VRAM in use, the part the processes KFD lists hold, and
    `unattributed`: the bytes beyond those and the allowance (0 within
    it; None when the card's use is unknown)."""
    if gpu is None:
        card, attributed = None, None
    else:
        _, card, attributed = kfd.vram_balance(gpu)
    excess = None if card is None or attributed is None else card - attributed
    return {
        "vram_used_bytes": card,
        "vram_attributed_bytes": attributed if card is not None else None,
        "unattributed": (
            None
            if excess is None
            else (excess if excess > kfd.UNATTRIBUTED_ALLOWANCE_BYTES else 0)
        ),
    }


# ── CPU ──────────────────────────────────────────────────────────────────────


def _cpu_dir() -> Path:
    return kfd.sysfs_root() / "devices" / "system" / "cpu"


def _flag(text: Optional[str]) -> Optional[bool]:
    return {"1": True, "0": False}.get(text or "")


def loadavg() -> Optional[List[float]]:
    try:
        return [float(x) for x in (kfd.PROC / "loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        return None


def cpu_settings() -> Dict[str, Any]:
    cpu_dir = _cpu_dir()
    affinity = sorted(os.sched_getaffinity(0))
    online = parse_cpulist(_read(cpu_dir / "online"))
    pinned = None if online is None else set(affinity) != set(online)
    governors: Dict[str, List[int]] = {}
    for c in affinity:
        g = _read(cpu_dir / f"cpu{c}" / "cpufreq" / "scaling_governor")
        governors.setdefault(g or "", []).append(c)
    siblings: Optional[List[int]] = None
    if pinned:
        found = set()
        for c in affinity:
            listed = _read(cpu_dir / f"cpu{c}" / "topology" / "thread_siblings_list")
            found.update(parse_cpulist(listed) or ())
        siblings = sorted(found - set(affinity))
    return {
        "affinity": format_cpulist(affinity),
        "pinned": pinned,
        "governors": [
            {"governor": g or None, "cpus": format_cpulist(cs)}
            for g, cs in sorted(governors.items())
        ],
        "boost": _flag(_read(cpu_dir / "cpufreq" / "boost")),
        "smt_active": _flag(_read(cpu_dir / "smt" / "active")),
        "smt_siblings_outside_affinity": (
            None if siblings is None else format_cpulist(siblings)
        ),
        "loadavg": loadavg(),
    }


def _cpu_times() -> Dict[int, Tuple[int, int]]:
    """{cpu: (total, idle + iowait)} jiffies from /proc/stat."""
    out: Dict[int, Tuple[int, int]] = {}
    try:
        lines = (kfd.PROC / "stat").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].startswith("cpu") or parts[0] == "cpu":
            continue
        try:
            values = [int(v) for v in parts[1:]]
            out[int(parts[0][3:])] = (sum(values[:8]), values[3] + values[4])
        except (ValueError, IndexError):
            continue
    return out


class CpuWindow:
    """How busy `cpus` were from construction to `stop()`, from /proc/stat:
    per CPU the share of the time not idle."""

    def __init__(self, cpus: Optional[Iterable[int]]) -> None:
        self.cpus = None if cpus is None else sorted(set(cpus))
        self.start = _cpu_times()
        self.load_start = loadavg()

    def stop(self) -> Dict[str, Any]:
        end = _cpu_times()
        busy: List[float] = []
        for c in self.cpus or ():
            a, b = self.start.get(c), end.get(c)
            if a is None or b is None or b[0] <= a[0]:
                continue
            busy.append(1.0 - (b[1] - a[1]) / (b[0] - a[0]))
        return {
            "cpus": None if self.cpus is None else format_cpulist(self.cpus),
            "busy_max": max(busy) if busy else None,
            "busy_mean": sum(busy) / len(busy) if busy else None,
            "loadavg_start": self.load_start,
            "loadavg_end": loadavg(),
        }


# ── session ──────────────────────────────────────────────────────────────────


def start_session(
    devices: Iterable[int],
    out_dir: Path,
    *,
    cpu: bool = False,
    environ: Optional[Mapping[str, str]] = None,
    environment: Optional[Mapping[str, Optional[str]]] = None,
    launch: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record the settings of the GPUs a stage benches on `devices` (and,
    with `cpu`, of its CPUs, with `environment`, the variables of the
    benches' environment that select what they run: lib/bench_env.py
    `recorded_env`, and with `launch`, how the benches launch the kernels:
    lib/launch_settings.py `bench_record`) in `<out_dir>/platform.json`;
    PlatformError names the GPUs and values when a check fails."""
    topology = kfd.gpus()
    by_index = {g.index: g for g in topology}
    problems: List[str] = []
    smi_index: Dict[int, Optional[int]] = {}
    smi_check: Optional[str] = None
    smi = kfd.amd_smi_list() if topology else None
    if smi is not None:
        try:
            smi_index = kfd.verify_with_amd_smi(topology, smi)
            smi_check = "agrees"
        except kfd.GpuMapError as e:
            smi_check = "disagrees"
            problems.append(f"GPU numbering: {e}")
    rows = []
    for d in devices:
        phys = kfd.physical_gpu(d, environ)
        gpu = by_index.get(phys) if phys is not None else None
        row = gpu_settings(d, phys, gpu)
        row["amd_smi_gpu"] = smi_index.get(phys) if phys is not None else None
        row.update(vram_use(gpu))
        rows.append(row)
    problems += mismatches(rows)
    doc: Dict[str, Any] = {
        "recorded": kfd.utc_iso(),
        "host": host_settings(),
        "amd_smi_check": smi_check,
        "unattributed_allowance_bytes": kfd.UNATTRIBUTED_ALLOWANCE_BYTES,
        "gpus": rows,
    }
    if cpu:
        doc["cpu"] = cpu_settings()
    if environment is not None:
        doc["environment"] = dict(environment)
    if launch is not None:
        doc["launch"] = dict(launch)
    doc["problems"] = problems
    write_json(Path(out_dir) / PLATFORM_FILE, doc)
    if problems:
        raise PlatformError("; ".join(problems))
    return doc


def block_fields(
    device: int, physical: Optional[int], gpu: Optional[kfd.Gpu]
) -> Dict[str, Any]:
    """The settings recorded with one bench block, read when it starts."""
    row = gpu_settings(device, physical, gpu)
    return {k: row[k] for k in ("device", "gpu", "gpu_id", "perf_level", "power_cap_w")}
