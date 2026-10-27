# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The host's GPUs as the amdgpu KFD driver lists them, and the processes
that hold memory on them, read from sysfs and /proc without opening a device.

GPU index: ROCr numbers the GPUs (the index ROCR_VISIBLE_DEVICES takes, also
called the physical index here) in the order of their KFD topology nodes,
which need not be PCI bus order; `amd-smi list` numbers them by bus address.
`gpus()` reads the nodes in order; `verify_with_amd_smi` checks each node's
gpu_id and PCI address against amd-smi's, which also names the node of every
GPU it lists. A stage's device D is an index into the GPUs its environment
makes visible (ROCR_VISIBLE_DEVICES, then HIP_VISIBLE_DEVICES or
CUDA_VISIBLE_DEVICES); `physical_gpu` resolves it. A bench for device D sees
that GPU alone (`one_gpu_launch`: ROCR_VISIBLE_DEVICES=<its GPU>,
`--device 0`), since hipblaslt-bench also allocates memory on device 0 of
the GPUs it sees. `gpu_isa` reads a GPU's compiler target and ASIC revision
from its node.

Process use: `<kfd>/proc/<pid>/vram_<gpu_id>` is above 0 for the GPUs a
process holds memory on; every KFD process has the file for every GPU.
`GpuWatch` samples them while the benches of a stage run on one GPU and
names the processes on it that do not belong to the stage. KFD names a
process by its pid in the initial pid namespace, so the check needs the
stage to run there (`pids_comparable`). The card's VRAM in use
(`mem_info_vram_used` of its PCI device) beyond what the processes KFD
lists hold and `UNATTRIBUTED_ALLOWANCE_BYTES` is unattributed use (a
process KFD does not show us, a graphics client) and counts as another
process once it outlasts its settle window (`settle_window_s`): memory a
process frees, or holds when it exits, stays in the card's use for a while
after the process's entry stops counting it, the longer the more memory.
`StageGpus` holds a stage's GPUs, their watches and the GPUs it stopped
using; a GPU retired for other use returns once it has been clean for
`READMIT_CLEAN_S`.

`TILEWRIGHT_SYSFS_ROOT` replaces `/sys` (tests use fake trees);
`TILEWRIGHT_AMD_SMI` names the amd-smi executable, empty to skip it.
"""
from __future__ import annotations

import datetime
import json
import os
import pwd
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .fs import write_json

SYSFS_ROOT_VAR = "TILEWRIGHT_SYSFS_ROOT"
AMD_SMI_VAR = "TILEWRIGHT_AMD_SMI"
PROC = Path("/proc")
INIT_PID_NAMESPACE = "pid:[4026531836]"
AMD_SMI_TIMEOUT_S = 60.0
ROCR_VAR = "ROCR_VISIBLE_DEVICES"
HIP_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
# Poll period of the stall watchdog of a bench whose GPU is watched.
WATCH_POLL_S = 5.0
QUARANTINE_DIR = "quarantine"
QUARANTINE_RECORD = "quarantine.json"
# The driver's own use of a card no process holds memory on stays well below
# this and does not grow with the processes' use; anything beyond it that no
# listed process holds is someone else's.
UNATTRIBUTED_ALLOWANCE_BYTES = 512 << 20
# The card's use and the processes' entries are not read at one instant, and
# memory leaves the card's use some time after a process's entry stops
# counting it: unattributed use counts once every reading, UNATTRIBUTED_STEP_S
# apart, shows it over its settle window, UNATTRIBUTED_CONFIRM_S plus
# UNATTRIBUTED_SETTLE_S_PER_GIB per GiB of the memory that may be leaving, at
# most UNATTRIBUTED_SETTLE_MAX_S.
UNATTRIBUTED_CONFIRM_S = 2.0
UNATTRIBUTED_STEP_S = 0.25
UNATTRIBUTED_SETTLE_S_PER_GIB = 1.0
UNATTRIBUTED_SETTLE_MAX_S = 30.0
# A GPU retired for another process's use is checked every RECHECK_S and
# returns after READMIT_CLEAN_S without any; when every GPU of a stage is
# retired, the stage waits up to ALL_RETIRED_WAIT_S for one to return.
RECHECK_S = 10.0
READMIT_CLEAN_S = 180.0
ALL_RETIRED_WAIT_S = 1800.0


class GpuMapError(RuntimeError):
    """The GPU numbering cannot be trusted, or a device list does not name
    a GPU."""


def sysfs_root() -> Path:
    return Path(os.environ.get(SYSFS_ROOT_VAR) or "/sys")


def kfd_root() -> Path:
    return sysfs_root() / "class" / "kfd" / "kfd"


def utc_iso(ts: Optional[float] = None) -> str:
    now = datetime.datetime.fromtimestamp(
        time.time() if ts is None else ts, datetime.timezone.utc
    )
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _properties(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in (_read(path) or "").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return out


def _numbered(path: Path) -> List[Path]:
    try:
        entries = [p for p in path.iterdir() if p.name.isdigit()]
    except OSError:
        return []
    return sorted(entries, key=lambda p: int(p.name))


# ── topology ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Gpu:
    """One GPU: its physical index (ROCr's), KFD node and gpu_id, PCI
    address, unique id and DRM render minor (None when the topology does
    not give one)."""

    index: int
    node: int
    gpu_id: int
    bdf: Optional[str] = None
    unique_id: Optional[int] = None
    render_minor: Optional[int] = None

    @property
    def uuid(self) -> Optional[str]:
        """The `GPU-<hex>` form ROCR_VISIBLE_DEVICES accepts."""
        return None if self.unique_id is None else f"GPU-{self.unique_id:016x}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "gpu": self.index,
            "kfd_node": self.node,
            "gpu_id": self.gpu_id,
            "bdf": self.bdf,
            "unique_id": self.uuid,
        }


def _bdf(props: Mapping[str, int]) -> Optional[str]:
    loc = props.get("location_id")
    if loc is None:
        return None
    domain = props.get("domain", 0)
    return f"{domain:04x}:{(loc >> 8) & 0xFF:02x}:{(loc >> 3) & 0x1F:02x}.{loc & 0x7}"


def gpus(root: Optional[Path] = None) -> List[Gpu]:
    """The GPU nodes of the KFD topology (gpu_id not 0) in node order, which
    is ROCr's GPU order; empty without a topology."""
    root = kfd_root() if root is None else Path(root)
    out: List[Gpu] = []
    for node in _numbered(root / "topology" / "nodes"):
        try:
            gpu_id = int(_read(node / "gpu_id") or "0")
        except ValueError:
            continue
        if gpu_id == 0:
            continue
        props = _properties(node / "properties")
        out.append(
            Gpu(
                index=len(out),
                node=int(node.name),
                gpu_id=gpu_id,
                bdf=_bdf(props),
                unique_id=props.get("unique_id") or None,
                render_minor=props.get("drm_render_minor"),
            )
        )
    return out


def isa_name(version: int) -> Optional[str]:
    """The compiler target a KFD `gfx_target_version` names (major * 10000 +
    minor * 100 + stepping, the last two written as hex digits: 90010 is
    gfx90a), or None for 0."""
    if int(version) <= 0:
        return None
    major, minor, stepping = version // 10000, version // 100 % 100, version % 100
    return f"gfx{major}{minor:x}{stepping:x}"


# KFD's HSA_CAP_ASIC_REVISION: bits 25:22 of a node's `capability`.
_ASIC_REVISION_SHIFT = 22
_ASIC_REVISION_MASK = 0xF


def gpu_isa(
    gpu: Gpu, root: Optional[Path] = None
) -> Tuple[Optional[str], Optional[int]]:
    """(compiler target, ASIC revision) of `gpu`, read from its KFD node
    alone; None for a value the node does not give."""
    root = kfd_root() if root is None else Path(root)
    props = _properties(root / "topology" / "nodes" / str(gpu.node) / "properties")
    version, capability = props.get("gfx_target_version"), props.get("capability")
    revision = None
    if capability is not None:
        revision = (capability >> _ASIC_REVISION_SHIFT) & _ASIC_REVISION_MASK
    return (isa_name(version) if version else None), revision


# The compute partition mode of a GPU that runs as one device.
FULL_DEVICE_PARTITION = "SPX"


def compute_partition(
    props: Mapping[str, int], sys_root: Optional[Path] = None
) -> Optional[str]:
    """The compute partition mode, such as SPX or CPX, that the PCI device
    of a KFD node with properties `props` reports in the sysfs tree at
    `sys_root` (default `sysfs_root()`); None when it reports none."""
    bdf = _bdf(props)
    if bdf is None:
        return None
    root = sysfs_root() if sys_root is None else Path(sys_root)
    pci = root / "bus" / "pci" / "devices" / bdf
    return _read(pci / "current_compute_partition") or None


def gpu_cus(gpu: Gpu, root: Optional[Path] = None) -> Optional[int]:
    """The CUs of `gpu`'s KFD node (`simd_count / simd_per_cu`); None when
    the node does not give them."""
    root = kfd_root() if root is None else Path(root)
    props = _properties(root / "topology" / "nodes" / str(gpu.node) / "properties")
    simds, per_cu = props.get("simd_count"), props.get("simd_per_cu")
    return simds // per_cu if simds and per_cu else None


def gpu_xcc(
    gpu: Gpu, root: Optional[Path] = None
) -> Tuple[Optional[int], Optional[str]]:
    """(the XCCs of `gpu`'s KFD node, the compute partition mode of its PCI
    device, such as SPX or CPX); None for a value sysfs does not give. In a
    partition mode other than FULL_DEVICE_PARTITION the node is one
    partition, and its XCCs are not the device's."""
    root = kfd_root() if root is None else Path(root)
    props = _properties(root / "topology" / "nodes" / str(gpu.node) / "properties")
    return (props.get("num_xcc") or None), compute_partition(props)


def _entries(environ: Mapping[str, str], var: str) -> Optional[List[str]]:
    """The entries of a device list up to its first negative one; None when
    the variable is unset or empty."""
    raw = str(environ.get(var, "")).strip()
    if not raw:
        return None
    out: List[str] = []
    for token in raw.split(","):
        token = token.strip()
        if token.lstrip("-").isdigit() and int(token) < 0:
            break
        out.append(token)
    return out


def physical_gpu(
    device: int, environ: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """The physical GPU a process with `environ` reaches as HIP device
    `device`, or None when the environment hides every GPU. GpuMapError when
    the device lists do not name it."""
    env = os.environ if environ is None else environ
    ordinal = int(device)
    if ordinal < 0:
        raise GpuMapError(f"device {device} is not a device index")
    for var in HIP_VARS:
        hip = _entries(env, var)
        if hip is None:
            continue
        if not hip:
            return None
        if not 0 <= ordinal < len(hip) or not hip[ordinal].isdigit():
            raise GpuMapError(
                f"{var}={env[var]} does not name a device {device} (of the "
                f"GPUs {ROCR_VAR} exposes)"
            )
        ordinal = int(hip[ordinal])
        break
    rocr = _entries(env, ROCR_VAR)
    if rocr is None:
        return ordinal
    if not rocr:
        return None
    if not 0 <= ordinal < len(rocr):
        raise GpuMapError(
            f"device {ordinal} is not one of the {len(rocr)} GPU(s) "
            f"{ROCR_VAR}={env[ROCR_VAR]} exposes"
        )
    token = rocr[ordinal]
    if token.isdigit():
        return int(token)
    for g in gpus():
        if g.uuid is not None and g.uuid.lower() == token.lower():
            return g.index
    raise GpuMapError(f"{ROCR_VAR}={env[ROCR_VAR]}: no GPU has the id {token!r}")


def bench_gpus(
    devices: Iterable[int], environ: Optional[Mapping[str, str]] = None
) -> List[int]:
    """The physical GPUs benches on `devices` use (`one_gpu_launch`)."""
    out: Set[int] = set()
    for d in devices:
        g = physical_gpu(d, environ)
        if g is not None:
            out.add(g)
    return sorted(out)


def gpu_refs(
    refs: Iterable[Union[int, str]], topology: Optional[Sequence[Gpu]] = None
) -> Tuple[Set[int], List[str]]:
    """(the physical GPUs `refs` name, the refs that name no GPU of the KFD
    topology). An integer is a physical index (ROCR's), and so is a string
    of decimal digits (a quoted index such as "3"); any other string a PCI
    address (`0000:4c:00.0`, or `4c:00.0` in domain 0) or a unique id
    (`GPU-<hex>`). An index the topology does not list still counts."""
    topology = list(gpus() if topology is None else topology)
    out: Set[int] = set()
    unknown: List[str] = []
    for ref in refs:
        if isinstance(ref, str) and ref.strip().isascii() and ref.strip().isdigit():
            ref = int(ref.strip())
        if isinstance(ref, int) and not isinstance(ref, bool):
            out.add(int(ref))
            if topology and all(g.index != ref for g in topology):
                unknown.append(str(ref))
            continue
        text = str(ref).strip().lower()
        full = text if text.count(":") == 2 else f"0000:{text}"
        hit = [
            g
            for g in topology
            if (g.bdf or "").lower() == full
            or (g.uuid is not None and g.uuid.lower() == text)
        ]
        if hit:
            out.update(g.index for g in hit)
        else:
            unknown.append(str(ref))
    return out, unknown


def without_gpus(
    devices: Iterable[int],
    excluded: Iterable[int],
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[List[int], List[int]]:
    """(`devices` whose physical GPU is not `excluded`, those whose is); a
    device the environment hides or does not name stays."""
    banned = set(excluded)
    kept: List[int] = []
    dropped: List[int] = []
    for d in devices:
        try:
            g = physical_gpu(int(d), environ)
        except GpuMapError:
            g = None
        (dropped if g is not None and g in banned else kept).append(int(d))
    return kept, dropped


def one_gpu_launch(device: int, env: Mapping[str, str]) -> Tuple[str, Dict[str, str]]:
    """(`--device` value, environment) of a process meant for `device` of
    `env` that sees that GPU alone: ROCR_VISIBLE_DEVICES names its physical
    GPU, the HIP and CUDA lists are dropped and the device is 0. `env` and
    `device` as they are when `env` hides every GPU."""
    physical = physical_gpu(device, env)
    if physical is None:
        return str(device), dict(env)
    out = {k: v for k, v in env.items() if k not in HIP_VARS}
    out[ROCR_VAR] = str(physical)
    return "0", out


def amd_smi_list(
    timeout_s: float = AMD_SMI_TIMEOUT_S,
) -> Optional[List[Dict[str, Any]]]:
    """The entries of `amd-smi list --json` (a read-only query), or None
    when amd-smi is unavailable or its output is not a list of objects."""
    exe = os.environ.get(AMD_SMI_VAR, "amd-smi")
    path = shutil.which(exe) if exe else None
    if path is None:
        return None
    try:
        out = subprocess.run(
            [path, "list", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        return None
    return data


def verify_with_amd_smi(
    topology: Iterable[Gpu], smi: Optional[Iterable[Mapping[str, Any]]]
) -> Dict[int, Optional[int]]:
    """{physical index: amd-smi's GPU index} after checking every amd-smi
    entry against the topology: the node it names must have its gpu_id
    (`kfd_id`) and PCI address. Entries without those fields are not
    compared; GpuMapError names a disagreement."""
    by_id = {g.gpu_id: g for g in topology}
    out: Dict[int, Optional[int]] = {g.index: None for g in by_id.values()}
    for e in smi or ():
        kfd_id, node = e.get("kfd_id"), e.get("node_id")
        if not isinstance(kfd_id, int) or not isinstance(node, int):
            continue
        g = by_id.get(kfd_id)
        bdf = str(e.get("bdf") or "").lower() or None
        if g is None or g.node != node or (bdf and g.bdf and bdf != g.bdf):
            ours = f"node {g.node}, {g.bdf}" if g else "no KFD node"
            raise GpuMapError(
                f"amd-smi lists GPU {e.get('gpu')} as gpu_id {kfd_id} on KFD node "
                f"{node} ({bdf}); the KFD topology has {ours} for that gpu_id"
            )
        if isinstance(e.get("gpu"), int):
            out[g.index] = int(e["gpu"])
    return out


# ── processes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    gone: bool = False
    comm: Optional[str] = None
    uid: Optional[int] = None
    user: Optional[str] = None
    ppid: Optional[int] = None
    session: Optional[int] = None
    start_s: Optional[float] = None
    state: Optional[str] = None

    @property
    def exited(self) -> bool:
        """No entry, or a zombie: the process runs no more."""
        return self.gone or self.state in ("Z", "X")


def boot_clock() -> float:
    """Seconds since boot, the clock of a process start time."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def proc_info(pid: int) -> ProcInfo:
    """What /proc tells about `pid`: `gone` when it has no entry; fields of
    an entry this user may not read are None."""
    base = PROC / str(int(pid))
    try:
        stat = (base / "stat").read_text()
        st_uid = base.stat().st_uid
    except (FileNotFoundError, ProcessLookupError):
        return ProcInfo(int(pid), gone=True)
    except OSError:
        return ProcInfo(int(pid))
    head, _, tail = stat.rpartition(")")
    fields = tail.split()
    try:
        user: Optional[str] = pwd.getpwuid(st_uid).pw_name
    except KeyError:
        user = None
    try:
        start_s = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        ppid, session = int(fields[1]), int(fields[3])
    except (IndexError, ValueError):
        start_s, ppid, session = None, None, None
    return ProcInfo(
        int(pid),
        comm=head.partition("(")[2] or None,
        uid=st_uid,
        user=user,
        ppid=ppid,
        session=session,
        start_s=start_s,
        state=fields[0] if fields else None,
    )


def ancestors(pid: int) -> List[int]:
    """`pid` and its ancestors, nearest first, as far as /proc can tell."""
    out: List[int] = []
    cur: Optional[int] = int(pid)
    while cur and cur not in out and len(out) < 4096:
        out.append(cur)
        cur = proc_info(cur).ppid
    return out


def pids_comparable() -> bool:
    """True in the initial pid namespace, where KFD's pids are this
    process's pids."""
    try:
        return os.readlink(PROC / "self" / "ns" / "pid") == INIT_PID_NAMESPACE
    except OSError:
        return False


def memory_users(root: Optional[Path] = None) -> Dict[int, Dict[int, int]]:
    """{pid: {gpu_id: bytes}} of the KFD processes with memory on a GPU."""
    root = kfd_root() if root is None else Path(root)
    out: Dict[int, Dict[int, int]] = {}
    for proc in _numbered(root / "proc"):
        try:
            files = list(proc.glob("vram_*"))
        except OSError:
            continue
        for f in files:
            try:
                gpu_id, used = int(f.name[5:]), int(_read(f) or "0")
            except ValueError:
                continue
            if used > 0:
                out.setdefault(int(proc.name), {})[gpu_id] = used
    return out


def card_vram_used(gpu: Gpu) -> Optional[int]:
    """Bytes of `gpu`'s VRAM in use by anyone, from its PCI device; None when
    sysfs does not give it."""
    if gpu.bdf is None:
        return None
    text = _read(
        sysfs_root() / "bus" / "pci" / "devices" / gpu.bdf / "mem_info_vram_used"
    )
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def vram_balance(
    gpu: Gpu, root: Optional[Path] = None
) -> Tuple[Dict[int, Dict[int, int]], Optional[int], int]:
    """(`memory_users`, the card's use, what the listed processes hold on
    it). The card is read before and after the processes and the lower
    reading kept, so a process allocating or freeing meanwhile does not
    show as unattributed."""
    before = card_vram_used(gpu)
    users = memory_users(root)
    after = card_vram_used(gpu)
    used = None if before is None or after is None else min(before, after)
    return users, used, sum(u.get(gpu.gpu_id, 0) for u in users.values())


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"


def settle_window_s(nbytes: int) -> float:
    """Seconds unattributed use must last to count while `nbytes` of memory
    may still be leaving the card."""
    gib = max(0, int(nbytes)) / float(1 << 30)
    return min(
        UNATTRIBUTED_SETTLE_MAX_S,
        UNATTRIBUTED_CONFIRM_S + gib * UNATTRIBUTED_SETTLE_S_PER_GIB,
    )


# ── contamination watch ──────────────────────────────────────────────────────


@dataclass
class Foreign:
    """A process outside the stage with memory on the watched GPU."""

    pid: int
    comm: Optional[str]
    user: Optional[str]
    uid: Optional[int]
    vram_bytes: int
    first_seen: str
    last_seen: str
    started: Optional[str] = None

    def describe(self) -> str:
        who = ", ".join(
            x
            for x in (
                self.comm and f"{self.comm!r}",
                self.user and f"user {self.user}",
                self.started and f"started {self.started}",
            )
            if x
        )
        return f"pid {self.pid}" + (f" ({who})" if who else "")

    def to_json(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "comm": self.comm,
            "user": self.user,
            "uid": self.uid,
            "vram_bytes": self.vram_bytes,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "started": self.started,
        }


@dataclass
class Unattributed:
    """VRAM of the watched GPU in use beyond what the processes KFD lists
    hold and `UNATTRIBUTED_ALLOWANCE_BYTES`: the largest such reading, the
    card's use and the listed processes' at that reading, and the settle
    window it outlasted."""

    vram_bytes: int
    card_used_bytes: int
    attributed_bytes: int
    first_seen: str
    last_seen: str
    settle_s: float = 0.0

    def describe(self) -> str:
        return (
            f"{_gib(self.vram_bytes)} of VRAM no listed process holds (card "
            f"{_gib(self.card_used_bytes)}, processes "
            f"{_gib(self.attributed_bytes)}, allowance "
            f"{_gib(UNATTRIBUTED_ALLOWANCE_BYTES)}, for over "
            f"{self.settle_s:.1f} s)"
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "vram_bytes": self.vram_bytes,
            "card_used_bytes": self.card_used_bytes,
            "attributed_bytes": self.attributed_bytes,
            "allowance_bytes": UNATTRIBUTED_ALLOWANCE_BYTES,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "settle_s": round(self.settle_s, 3),
        }


OtherUse = Union[Foreign, Unattributed]


@dataclass
class GpuWatch:
    """Samples the processes with memory on `gpu` while benches run on it.

    The stage's own processes are its descendants (`own_root`, default this
    process) and every process in the session of a bench passed to
    `add_bench`; any other process with memory on the GPU is foreign, named
    by the scan that sees it. A process without a /proc entry has exited
    and is not named. VRAM of the card no listed process holds counts as
    `unattributed` when every reading over its settle window shows it, the
    window of the larger of that memory and `bench_peak_bytes` (the most
    the benches held on the GPU at one scan: memory a bench frees, or holds
    when it exits, lingers on the card); a watch that named a foreign
    process does not wait for it. The benches' own memory may be on `gpu`
    only; memory elsewhere means the GPU numbering is wrong
    (`mapping_error`). `clock` and `sleep` pace the settle window."""

    gpu: Gpu
    own_root: int = field(default_factory=os.getpid)
    root: Optional[Path] = None
    min_interval_s: float = 1.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    benches: Set[int] = field(default_factory=set)
    own: Set[int] = field(default_factory=set)
    foreign: Dict[int, Foreign] = field(default_factory=dict)
    unattributed: Optional[Unattributed] = None
    bench_memory: Dict[int, Set[int]] = field(default_factory=dict)
    bench_peak_bytes: int = 0
    _last: Optional[float] = None

    def add_bench(self, pid: int) -> None:
        self.benches.add(int(pid))

    def _is_own(self, info: ProcInfo) -> bool:
        if info.pid in self.own or self.own_root in ancestors(info.pid):
            self.own.add(info.pid)
            return True
        return False

    def _scan(self, started_before: Optional[float] = None) -> None:
        now = utc_iso()
        users, card, attributed = vram_balance(self.gpu, self.root)
        benches_here = 0
        for pid, used in users.items():
            info = proc_info(pid)
            if info.gone:
                continue
            if pid in self.benches or info.session in self.benches:
                self.bench_memory.setdefault(pid, set()).update(used)
                benches_here += used.get(self.gpu.gpu_id, 0)
                continue
            here = used.get(self.gpu.gpu_id, 0)
            if not here or self._is_own(info):
                continue
            seen = self.foreign.get(pid)
            if seen is not None:
                seen.last_seen = now
                seen.vram_bytes = max(seen.vram_bytes, here)
                continue
            if (
                started_before is not None
                and info.start_s is not None
                and info.start_s > started_before
            ):
                continue
            started = None
            if info.start_s is not None:
                started = utc_iso(time.time() - (boot_clock() - info.start_s))
            self.foreign[pid] = Foreign(
                pid, info.comm, info.user, info.uid, here, now, now, started
            )
        self.bench_peak_bytes = max(self.bench_peak_bytes, benches_here)
        if self.unattributed is None and not self.foreign:
            self._confirm_unattributed(card, attributed, now)

    def _confirm_unattributed(
        self, card: Optional[int], attributed: int, now: str
    ) -> None:
        if card is None or card - attributed <= UNATTRIBUTED_ALLOWANCE_BYTES:
            return
        seen = Unattributed(card - attributed, card, attributed, now, now)
        start = self.clock()
        while True:
            seen.settle_s = settle_window_s(max(seen.vram_bytes, self.bench_peak_bytes))
            if self.clock() - start >= seen.settle_s:
                break
            self.sleep(UNATTRIBUTED_STEP_S)
            _, card, attributed = vram_balance(self.gpu, self.root)
            if card is None or card - attributed <= UNATTRIBUTED_ALLOWANCE_BYTES:
                return
            seen.last_seen = utc_iso()
            if card - attributed > seen.vram_bytes:
                seen.vram_bytes = card - attributed
                seen.card_used_bytes, seen.attributed_bytes = card, attributed
        self.unattributed = seen

    def sample(self) -> bool:
        """Scan (at most every `min_interval_s`); True once another process's
        use or a mapping error has been seen."""
        t = self.clock()
        if self._last is None or t - self._last >= self.min_interval_s:
            self._last = t
            self._scan()
        return self.contaminated

    def check_end(self, ended: Optional[float] = None) -> bool:
        """The scan after a bench exits at `ended` (`boot_clock()`, default
        now): processes started after it are not counted. True when the
        watch is contaminated."""
        self._scan(started_before=boot_clock() if ended is None else ended)
        return self.contaminated

    def mapping_error(self) -> Optional[str]:
        for pid, ids in sorted(self.bench_memory.items()):
            extra = sorted(ids - {self.gpu.gpu_id})
            if extra:
                return (
                    f"bench pid {pid} holds memory on gpu_id(s) {extra}, but it "
                    f"benches on GPU {self.gpu.index} (gpu_id {self.gpu.gpu_id}, "
                    f"{self.gpu.bdf}) alone: the GPU numbering is wrong"
                )
        return None

    def others(self) -> List[OtherUse]:
        out: List[OtherUse] = list(self.foreign.values())
        return out + ([self.unattributed] if self.unattributed else [])

    @property
    def contaminated(self) -> bool:
        return bool(self.others()) or self.mapping_error() is not None

    def rule(self) -> Optional[str]:
        if self.mapping_error() is not None:
            return "gpu-numbering"
        if self.foreign:
            return "foreign-process"
        return "unattributed-memory" if self.unattributed else None

    def reason(self) -> str:
        err = self.mapping_error()
        if err is not None:
            return err
        who = "; ".join(f.describe() for f in self.others())
        return (
            f"{who} on GPU {self.gpu.index} (gpu_id "
            f"{self.gpu.gpu_id}, {self.gpu.bdf}) while the bench ran"
        )

    def record(self) -> Dict[str, Any]:
        return {
            "rule": self.rule(),
            "reason": self.reason() if self.contaminated else None,
            "gpu": self.gpu.to_json(),
            "foreign": [f.to_json() for f in self.foreign.values()],
            "unattributed": self.unattributed.to_json() if self.unattributed else None,
            "bench_pids": sorted(self.benches),
        }


def foreign_now(gpu: Gpu, own_root: Optional[int] = None) -> List[OtherUse]:
    """Other use of `gpu` now (`GpuWatch` without benches): the foreign
    processes, else unattributed use that outlasts its settle window."""
    w = GpuWatch(gpu, own_root=os.getpid() if own_root is None else own_root)
    w.check_end()
    return w.others()


def quarantine(
    dest: Path, files: Iterable[Optional[Path]], record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Move the existing `files` into `dest` and write `record`, with the
    moved names and the time, as `dest/quarantine.json`; returns it."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in files:
        if p is not None and Path(p).exists():
            shutil.move(str(p), str(dest / Path(p).name))
            moved.append(Path(p).name)
    rec = {**record, "quarantined": utc_iso(), "files": moved}
    write_json(dest / QUARANTINE_RECORD, rec)
    return rec


class StageGpus:
    """The GPUs a stage benches on: each device's physical GPU and its KFD
    identity (None when unknown), watches for its benches, and the devices
    the stage stopped using (`retire`). A device retired for another
    process's use (`temporary`) returns through `recheck` once it has been
    clean for `READMIT_CLEAN_S`; `returned` lists those spells."""

    def __init__(
        self,
        devices: Sequence[int],
        environ: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        topology = {g.index: g for g in gpus()}
        self.physical = {d: physical_gpu(d, environ) for d in devices}
        self.gpu = {
            d: topology.get(p) if p is not None else None
            for d, p in self.physical.items()
        }
        self.watching = pids_comparable()
        self.retired: Dict[int, str] = {}
        self.returned: List[Dict[str, Any]] = []
        self.clock = clock
        self._temporary: Dict[int, str] = {}
        self._clean_since: Dict[int, float] = {}
        self._all_retired_since: Optional[float] = None
        self._lock = threading.Lock()

    def unwatched(self) -> List[int]:
        if not self.watching:
            return sorted(self.gpu)
        return sorted(d for d, g in self.gpu.items() if g is None)

    def unwatched_note(self) -> Optional[str]:
        devices = self.unwatched()
        if not devices:
            return None
        if not self.watching:
            why = "KFD names processes by pids of another pid namespace"
        elif all(self.physical.get(d) is None for d in devices):
            why = "the environment hides every GPU"
        else:
            why = "the KFD topology does not list them"
        return f"other processes on device(s) {devices} are not checked: {why}"

    def watch(self, device: int) -> Optional[GpuWatch]:
        g = self.gpu.get(device)
        if g is None or not self.watching:
            return None
        return GpuWatch(g)

    def busy(self, device: int) -> List[OtherUse]:
        g = self.gpu.get(device)
        if g is None or not self.watching:
            return []
        return foreign_now(g)

    def retire(self, device: int, why: str, temporary: bool = False) -> bool:
        """Stop using `device`, until `recheck` finds it clean when
        `temporary`; False when it was retired before."""
        with self._lock:
            if device in self.retired:
                return False
            self.retired[device] = why
            if temporary:
                self._temporary[device] = utc_iso()
            if self._all_retired_since is None and set(self.gpu) <= set(self.retired):
                self._all_retired_since = self.clock()
            return True

    def active(self, device: int) -> bool:
        with self._lock:
            return device not in self.retired

    def first_active(self, devices: Sequence[int]) -> Optional[int]:
        return next((d for d in devices if self.active(d)), None)

    def may_return(self, device: int) -> bool:
        with self._lock:
            return device not in self.retired or device in self._temporary

    def recheck(self, device: int) -> bool:
        """Check a temporarily retired `device`; True when it is in service
        (again): no other use of it for `READMIT_CLEAN_S`."""
        if not self.may_return(device):
            return False
        if self.active(device):
            return True
        busy = self.busy(device)
        t = self.clock()
        with self._lock:
            if busy:
                self._clean_since.pop(device, None)
                return False
            since = self._clean_since.setdefault(device, t)
            if t - since < READMIT_CLEAN_S:
                return False
            self.returned.append(
                {
                    "device": device,
                    "why": self.retired.pop(device),
                    "retired": self._temporary.pop(device),
                    "returned": utc_iso(),
                }
            )
            del self._clean_since[device]
            self._all_retired_since = None
            return True

    def all_retired_for(self) -> Optional[float]:
        """Seconds every device of the stage has been retired; None while
        one is in service."""
        with self._lock:
            since = self._all_retired_since
        return None if since is None else self.clock() - since
