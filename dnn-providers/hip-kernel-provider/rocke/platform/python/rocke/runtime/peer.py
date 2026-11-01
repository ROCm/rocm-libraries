# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host side of a multi-GPU rocKE group: symmetric peer arenas over HIP IPC.

Host only (ctypes over the same ``libamdhip64`` handle :mod:`.hip_module`
resolves, so torch and rocKE share one HIP runtime). It emits no IR and no
other runtime module imports it, so every existing build is unaffected.

One :class:`PeerGroup` per rank (one process per GPU, ``mode="ipc"``) owns one
device arena (``mem="uncached"`` by default, the receive mailboxes of the
in-kernel collectives) and maps every other rank's arena through HIP IPC. A
kernel gets the arenas as 64-bit global addresses (:meth:`PeerGroup.kernargs`:
``peer0 .. peer{N-1}``, ``ctrl``, ``arena`` (own), ``rank``), never as buffer descriptors.

``mode="p2p"``: one process drives several devices (microbenchmarks and
topology probes); the arenas are plain allocations made peer-accessible with
``hipDeviceEnablePeerAccess``. :class:`P2PGroup` builds one group per device.

Rendezvous (``mode="ipc"``):

* ``"file:<dir>"``: torch-free. Rank ``r`` writes ``<dir>/rank<r>.<gen>.json``
  (write + rename), then waits for the others; a barrier is one such
  generation. ``<dir>`` must be fresh per group (the caller's job dir).
* ``"gloo"`` / ``"torch"``: an initialised ``torch.distributed`` group
  (``group=`` or the default one): ``all_gather_object`` + ``barrier``.

Arena layout of the control words (first 256 bytes of every arena, owned by
the kernels; :mod:`kernels.gfx950.qwen38_mono.ccl.layout` repeats the offsets):
``epoch`` u32 at 0 (tag base of the next launch, 1 after :meth:`reset`),
``abort`` u32 at 4 (set by any peer that aborts), ``exit`` u32 at 8 (CTA exit
counter), ``claim`` u32 at 12 (first-error claim), ``launches`` u32 at 16,
``slotbase`` u32 at 20 (slot of the next launch's first use).

The error record (``ctrl``) is host memory mapped into the device
(``hipHostMalloc`` mapped + coherent): 16 u32 words written by the first CTA
that fails, read by :meth:`PeerGroup.check` after a launch or by a watchdog
while a launch runs. Codes: :data:`ERROR_NAMES`.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .hip_module import (  # same bound functions (one argtypes per HIP symbol)
    HipError,
    _b,
    _check,
    _ensure_hip_init,
    _hipDeviceSynchronize,
    _hipFree,
    _hipGetDevice,
    _hipGetDeviceCount,
    _hipMalloc,
    _hipMemset,
    _hipSetDevice,
)
from .hip_module import _hipMemcpy as _hipMemcpyHtoD

__all__ = [
    "ARENA_ALIGN",
    "CTRL_WORDS",
    "ERROR_NAMES",
    "MEM_KINDS",
    "OFF_ABORT",
    "OFF_CLAIM",
    "OFF_EPOCH",
    "OFF_EXIT",
    "OFF_LAUNCHES",
    "OFF_GSYNC",
    "OFF_SLOTBASE",
    "CONTROL_BYTES",
    "P2PGroup",
    "PeerError",
    "PeerGroup",
    "ErrorRecord",
    "FileRendezvous",
    "TorchRendezvous",
    "tag_add",
    "topology",
]

# ---------------------------------------------------------------- constants

HANDLE_BYTES = 64
ARENA_ALIGN = 2 << 20  # arenas are whole 2 MiB pages (the NUMA homing model is per page)
MEM_KINDS = ("uncached", "fine", "coarse")
_EXT_MALLOC_FLAGS = {"fine": 0x1, "uncached": 0x3}  # hipDeviceMallocFinegrained / Uncached
_IPC_LAZY_ENABLE_PEER_ACCESS = 0x1
_HOST_MALLOC_MAPPED = 0x2
_HOST_MALLOC_COHERENT = 0x40000000

OFF_EPOCH, OFF_ABORT, OFF_EXIT, OFF_CLAIM, OFF_LAUNCHES, OFF_SLOTBASE, OFF_GSYNC = 0, 4, 8, 12, 16, 20, 24
CONTROL_BYTES = 256

CTRL_WORDS = 16  # error record: u32 words
# record word indices
REC_STATUS, REC_CODE, REC_STAGE, REC_USE, REC_PEER = 0, 1, 2, 3, 4
REC_TAG_EXPECTED, REC_TAG_SEEN, REC_CTA, REC_RT_LO, REC_RT_HI, REC_RANK = 5, 6, 7, 8, 9, 10

ERROR_NAMES = {
    0: "OK",
    -1: "RESIDENCY_TIMEOUT",
    -2: "GRID_SYNC_TIMEOUT",
    -3: "PEER_TIMEOUT",
    -4: "QSA_TIMEOUT",
    -5: "CCL_MISMATCH",
    -6: "PEER_ABORT",
    -7: "TAG_AHEAD",
}

TAG_MOD = (1 << 32) - 1  # tags live in [1, 2**32 - 1]; 0 never matches zeroed memory


def tag_add(epoch: int, use: int) -> int:
    """Host twin of the device tag arithmetic: the tag ``use`` steps after
    ``epoch`` in the sequence 1, 2, ..., 2**32 - 1, 1, ... (0 skipped).
    Device form: ``s_add_u32`` + ``s_addc_u32 0`` (the carry adds the skip)."""
    if not 1 <= epoch <= TAG_MOD:
        raise ValueError(f"epoch {epoch} outside [1, 2**32 - 1]")
    if not 0 <= use < TAG_MOD:
        raise ValueError(f"use {use} outside [0, 2**32 - 1)")
    return (epoch - 1 + use) % TAG_MOD + 1


# ----------------------------------------------------------------- HIP calls


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * HANDLE_BYTES)]


_vp = ctypes.c_void_p
_hipExtMallocWithFlags = _b("hipExtMallocWithFlags", ctypes.POINTER(_vp), ctypes.c_size_t, ctypes.c_uint)
_hipIpcGetMemHandle = _b("hipIpcGetMemHandle", ctypes.POINTER(_IpcHandle), _vp)
_hipIpcOpenMemHandle = _b("hipIpcOpenMemHandle", ctypes.POINTER(_vp), _IpcHandle, ctypes.c_uint)
_hipIpcCloseMemHandle = _b("hipIpcCloseMemHandle", _vp)
_hipDeviceCanAccessPeer = _b("hipDeviceCanAccessPeer", ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int)
_hipDeviceEnablePeerAccess = _b("hipDeviceEnablePeerAccess", ctypes.c_int, ctypes.c_uint)
_hipHostMalloc = _b("hipHostMalloc", ctypes.POINTER(_vp), ctypes.c_size_t, ctypes.c_uint)
_hipHostFree = _b("hipHostFree", _vp)
_hipHostGetDevicePointer = _b("hipHostGetDevicePointer", ctypes.POINTER(_vp), _vp, ctypes.c_uint)
_hipDeviceGetPCIBusId = _b("hipDeviceGetPCIBusId", ctypes.c_char_p, ctypes.c_int, ctypes.c_int)

_HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704


def _set_device(device: int) -> None:
    _check(_hipSetDevice(int(device)), f"hipSetDevice({device})")


def _get_device() -> int:
    d = ctypes.c_int(-1)
    _check(_hipGetDevice(ctypes.byref(d)), "hipGetDevice")
    return int(d.value)


def pci_bus_id(device: int) -> str:
    buf = ctypes.create_string_buffer(64)
    _check(_hipDeviceGetPCIBusId(buf, 64, int(device)), "hipDeviceGetPCIBusId")
    return buf.value.decode()


def _round_up(n: int, a: int) -> int:
    return -(-int(n) // a) * a


def alloc_arena(nbytes: int, mem: str) -> int:
    """Device allocation of ``nbytes`` (whole 2 MiB pages) of kind ``mem``
    on the current device, zeroed."""
    if mem not in MEM_KINDS:
        raise ValueError(f"mem must be one of {MEM_KINDS}, got {mem!r}")
    p = _vp(0)
    if mem == "coarse":
        _check(_hipMalloc(ctypes.byref(p), nbytes), f"hipMalloc({nbytes})")
    else:
        _check(
            _hipExtMallocWithFlags(ctypes.byref(p), nbytes, _EXT_MALLOC_FLAGS[mem]),
            f"hipExtMallocWithFlags({nbytes}, {mem})",
        )
    _check(_hipMemset(p, 0, nbytes), "hipMemset")
    return int(p.value)


# -------------------------------------------------------------- error record


@dataclass(frozen=True)
class ErrorRecord:
    status: int
    code: int
    stage: int
    use: int
    peer: int
    tag_expected: int
    tag_seen: int
    cta: int
    realtime: int
    rank: int

    @property
    def name(self) -> str:
        return ERROR_NAMES.get(self.code, f"UNKNOWN({self.code})")

    @classmethod
    def from_words(cls, w: Sequence[int]) -> "ErrorRecord":
        s32 = lambda x: x - (1 << 32) if x >= 1 << 31 else x  # noqa: E731
        return cls(
            status=int(w[REC_STATUS]),
            code=s32(int(w[REC_CODE])),
            stage=int(w[REC_STAGE]),
            use=int(w[REC_USE]),
            peer=int(w[REC_PEER]),
            tag_expected=int(w[REC_TAG_EXPECTED]),
            tag_seen=int(w[REC_TAG_SEEN]),
            cta=int(w[REC_CTA]),
            realtime=int(w[REC_RT_LO]) | (int(w[REC_RT_HI]) << 32),
            rank=int(w[REC_RANK]),
        )


class PeerError(RuntimeError):
    def __init__(self, rank: int, record: ErrorRecord):
        self.rank = rank
        self.record = record
        super().__init__(
            f"rank {rank}: {record.name} (code {record.code}) stage {record.stage} use {record.use} "
            f"peer {record.peer} tag expected {record.tag_expected} seen {record.tag_seen} cta {record.cta}"
        )


class _HostRecord:
    """``CTRL_WORDS`` u32 of mapped, coherent host memory."""

    def __init__(self) -> None:
        nbytes = 4 * CTRL_WORDS
        self.host = _vp(0)
        _check(
            _hipHostMalloc(ctypes.byref(self.host), nbytes, _HOST_MALLOC_MAPPED | _HOST_MALLOC_COHERENT),
            "hipHostMalloc(ctrl)",
        )
        self.words = (ctypes.c_uint32 * CTRL_WORDS).from_address(int(self.host.value))
        dev = _vp(0)
        _check(_hipHostGetDevicePointer(ctypes.byref(dev), self.host, 0), "hipHostGetDevicePointer")
        self.device = int(dev.value)
        self.clear()

    def clear(self) -> None:
        for i in range(CTRL_WORDS):
            self.words[i] = 0

    def read(self) -> ErrorRecord:
        return ErrorRecord.from_words([int(self.words[i]) for i in range(CTRL_WORDS)])

    def free(self) -> None:
        if self.host:
            _hipHostFree(self.host)
            self.host = _vp(0)


# ---------------------------------------------------------------- rendezvous


class FileRendezvous:
    """Torch-free rendezvous over a directory every rank can see."""

    def __init__(self, path: str, rank: int, ranks: int, timeout_s: float = 120.0):
        self.path, self.rank, self.ranks, self.timeout_s = path, rank, ranks, timeout_s
        self.gen = 0
        os.makedirs(path, exist_ok=True)

    def _file(self, r: int, gen: int) -> str:
        return os.path.join(self.path, f"rank{r}.{gen}.json")

    def all_gather(self, obj) -> list:
        gen = self.gen
        self.gen += 1
        tmp = self._file(self.rank, gen) + ".tmp"
        with open(tmp, "w", newline="\n") as fh:
            json.dump(obj, fh)
        for _ in range(500):  # Windows: replace fails while a reader holds the target open
            try:
                os.replace(tmp, self._file(self.rank, gen))
                break
            except PermissionError:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"rendezvous {self.path}: cannot publish {tmp}")
        out: List = [None] * self.ranks
        t_end = time.monotonic() + self.timeout_s
        pending = set(range(self.ranks))
        while pending:
            for r in sorted(pending):
                f = self._file(r, gen)
                if os.path.exists(f):
                    try:
                        with open(f) as fh:
                            out[r] = json.load(fh)
                    except (OSError, ValueError):
                        continue  # being published right now: read it next round
                    pending.discard(r)
            if pending:
                if time.monotonic() > t_end:
                    raise TimeoutError(f"rendezvous {self.path} gen {gen}: ranks {sorted(pending)} absent")
                time.sleep(0.005)
        return out

    def barrier(self) -> None:
        self.all_gather(0)


class TorchRendezvous:
    """Over an initialised ``torch.distributed`` group (gloo for CPU objects)."""

    def __init__(self, group=None):
        import torch.distributed as dist

        self.dist, self.group = dist, group

    def all_gather(self, obj) -> list:
        out = [None] * self.dist.get_world_size(self.group)
        self.dist.all_gather_object(out, obj, group=self.group)
        return out

    def barrier(self) -> None:
        self.dist.barrier(group=self.group)


def _make_rendezvous(rendezvous, rank: int, ranks: int, group, timeout_s: float):
    if not isinstance(rendezvous, str):
        return rendezvous  # an object with all_gather / barrier
    if rendezvous.startswith("file:"):
        return FileRendezvous(rendezvous[5:], rank, ranks, timeout_s=max(timeout_s, 60.0))
    if rendezvous in ("gloo", "torch"):
        return TorchRendezvous(group)
    raise ValueError(f"unknown rendezvous {rendezvous!r} (file:<dir> | gloo)")


# --------------------------------------------------------------- peer group


class PeerGroup:
    """One rank's arena, its mappings of the peers' arenas and its error record.

    Every rank constructs one with the same ``nbytes`` and ``ranks`` before any
    rank launches. :meth:`reset` and :meth:`close` are collective."""

    def __init__(
        self,
        rank: int,
        ranks: int,
        nbytes: int = 64 << 20,
        *,
        mem: str = "uncached",
        rendezvous="file:",
        group=None,
        device: Optional[int] = None,
        timeout_s: float = 10.0,
        alloc: Optional[Callable[[int, str], int]] = None,
        ipc: Optional[dict] = None,
    ):
        if ranks < 1 or not 0 <= rank < ranks:
            raise ValueError(f"rank {rank} outside a group of {ranks}")
        if ranks > 8:
            raise ValueError("at most 8 ranks (one node)")
        if nbytes <= CONTROL_BYTES:
            raise ValueError("arena smaller than its control block")
        if mem not in MEM_KINDS:
            raise ValueError(f"mem must be one of {MEM_KINDS}, got {mem!r}")
        self.rank, self.ranks, self.mem, self.timeout_s = rank, ranks, mem, timeout_s
        self.nbytes = _round_up(nbytes, ARENA_ALIGN)
        self._fake = alloc is not None
        if not self._fake:
            _ensure_hip_init()
            if device is not None:
                _set_device(device)
            self.device = _get_device()
        else:
            self.device = -1 if device is None else device
        self._rv = _make_rendezvous(rendezvous, rank, ranks, group, timeout_s) if ranks > 1 else None
        self._ipc = ipc or {}
        self.base = (alloc or alloc_arena)(self.nbytes, mem)
        self._mapped: Dict[int, int] = {}
        bases = [0] * ranks
        bases[rank] = self.base
        if ranks > 1:
            meta = {"rank": rank, "handle": self._export(), "nbytes": self.nbytes, "mem": mem}
            if not self._fake:
                meta["bus"] = pci_bus_id(self.device)
            allm = self._rv.all_gather(meta)
            for m in allm:
                if m["nbytes"] != self.nbytes or m["mem"] != mem:
                    raise ValueError(f"rank {m['rank']} arena {m['nbytes']} {m['mem']} != {self.nbytes} {mem}")
            self.buses = [m.get("bus") for m in allm]
            for peer, m in enumerate(allm):
                if peer != rank:
                    bases[peer] = self._mapped[peer] = self._open(m["handle"])
        else:
            self.buses = [None if self._fake else pci_bus_id(self.device)]
        self.bases = tuple(bases)
        self._rec = None if self._fake else _HostRecord()
        self.ctrl = 0 if self._rec is None else self._rec.device
        self.reset()

    # -- IPC
    def _export(self) -> str:
        if self._fake:
            return self._ipc.get("export", lambda base: f"{base:x}")(self.base)
        h = _IpcHandle()
        _check(_hipIpcGetMemHandle(ctypes.byref(h), _vp(self.base)), "hipIpcGetMemHandle")
        return bytes(ctypes.string_at(ctypes.byref(h), HANDLE_BYTES)).hex()

    def _open(self, handle_hex: str) -> int:
        if self._fake:
            return self._ipc.get("open", lambda h: int(h, 16))(handle_hex)
        raw = bytes.fromhex(handle_hex)
        if len(raw) != HANDLE_BYTES:
            raise ValueError("bad IPC handle size")
        h = _IpcHandle()
        ctypes.memmove(ctypes.byref(h), raw, HANDLE_BYTES)
        p = _vp(0)
        _check(_hipIpcOpenMemHandle(ctypes.byref(p), h, _IPC_LAZY_ENABLE_PEER_ACCESS), "hipIpcOpenMemHandle")
        return int(p.value)

    # -- protocol
    def kernargs(self) -> dict:
        d = {f"peer{r}": b for r, b in enumerate(self.bases)}
        d["ctrl"] = self.ctrl
        d["arena"] = self.base
        d["rank"] = self.rank
        return d

    def barrier(self) -> None:
        if self._rv is not None:
            self._rv.barrier()

    def _sync(self) -> None:
        if not self._fake:
            _set_device(self.device)
            _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")

    def write_word(self, offset: int, value: int) -> None:
        """Host write of one u32 of this rank's arena (synchronous)."""
        if self._fake:
            return
        v = ctypes.c_uint32(value & 0xFFFFFFFF)
        _check(_hipMemcpyHtoD(_vp(self.base + offset), _vp(ctypes.addressof(v)), 4, 1), "hipMemcpyHtoD")

    def reset(self, epoch: int = 1) -> None:
        """Collective: synchronize, barrier, zero the arena, ``epoch`` word,
        synchronize, barrier. Group creation and error recovery only (tags
        continue across launches from the device-resident epoch)."""
        if not 1 <= epoch <= TAG_MOD:
            raise ValueError("epoch outside [1, 2**32 - 1]")
        self._sync()
        self.barrier()  # a peer that timed out may still push into this arena until it exits
        if not self._fake:
            _check(_hipMemset(_vp(self.base), 0, self.nbytes), "hipMemset")
            self.write_word(OFF_EPOCH, epoch)
        if self._rec is not None:
            self._rec.clear()
        self._sync()
        self.barrier()

    def record(self) -> Optional[ErrorRecord]:
        return None if self._rec is None else self._rec.read()

    def check(self) -> None:
        """Raise :class:`PeerError` when a kernel recorded an error (call after
        the launch completed, or from a watchdog while it runs)."""
        rec = self.record()
        if rec is not None and rec.status != 0:
            raise PeerError(self.rank, rec)

    def close(self) -> None:
        """Collective: synchronize, barrier, unmap peers, barrier, free."""
        if self.base == 0:
            return
        self._sync()
        self.barrier()
        if not self._fake:
            for p in self._mapped.values():
                _check(_hipIpcCloseMemHandle(_vp(p)), "hipIpcCloseMemHandle")
        self._mapped.clear()
        self.barrier()
        if not self._fake:
            _check(_hipFree(_vp(self.base)), "hipFree")
        if self._rec is not None:
            self._rec.free()
        self.base = 0

    def __enter__(self) -> "PeerGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()


class P2PGroup:
    """Single process, several devices (``mode="p2p"``): one arena per device,
    every ordered pair peer-enabled. ``member(i)`` is the per-device view with
    the same API subset as :class:`PeerGroup` (``bases``, ``kernargs``,
    ``ctrl``, ``record`` / ``check``); launches go to device ``devices[i]``
    after ``set_device``."""

    def __init__(self, devices: Sequence[int], nbytes: int = 64 << 20, *, mem: str = "uncached"):
        _ensure_hip_init()
        self.devices = [int(d) for d in devices]
        n = len(self.devices)
        if not 1 <= n <= 8:
            raise ValueError("1..8 devices")
        for a in self.devices:
            for b in self.devices:
                if a != b:
                    ok = ctypes.c_int(0)
                    _check(_hipDeviceCanAccessPeer(ctypes.byref(ok), a, b), "hipDeviceCanAccessPeer")
                    if not ok.value:
                        raise HipError(f"device {a} cannot access device {b}")
        self.nbytes = _round_up(nbytes, ARENA_ALIGN)
        self.mem = mem
        self.bases: List[int] = []
        self._recs: List[_HostRecord] = []
        for a in self.devices:
            _set_device(a)
            for b in self.devices:
                if a != b:
                    rc = _hipDeviceEnablePeerAccess(b, 0)
                    if rc not in (0, _HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED):
                        _check(rc, f"hipDeviceEnablePeerAccess({a}->{b})")
            self.bases.append(alloc_arena(self.nbytes, mem))
            self._recs.append(_HostRecord())
        self.reset()

    @property
    def ranks(self) -> int:
        return len(self.devices)

    def set_device(self, i: int) -> None:
        _set_device(self.devices[i])

    def kernargs(self, i: int) -> dict:
        d = {f"peer{r}": b for r, b in enumerate(self.bases)}
        d["ctrl"] = self._recs[i].device
        d["arena"] = self.bases[i]
        d["rank"] = i
        return d

    def reset(self, epoch: int = 1) -> None:
        for i, d in enumerate(self.devices):
            _set_device(d)
            _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")
        for i, d in enumerate(self.devices):
            _set_device(d)
            _check(_hipMemset(_vp(self.bases[i]), 0, self.nbytes), "hipMemset")
            v = ctypes.c_uint32(epoch)
            _check(_hipMemcpyHtoD(_vp(self.bases[i] + OFF_EPOCH), _vp(ctypes.addressof(v)), 4, 1), "hipMemcpyHtoD")
            self._recs[i].clear()
            _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")

    def record(self, i: int) -> ErrorRecord:
        return self._recs[i].read()

    def check(self) -> None:
        for i in range(self.ranks):
            rec = self.record(i)
            if rec.status != 0:
                raise PeerError(i, rec)

    def close(self) -> None:
        for i, d in enumerate(self.devices):
            _set_device(d)
            _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")
            _check(_hipFree(_vp(self.bases[i])), "hipFree")
            self._recs[i].free()
        self.bases = []


def topology(devices: Optional[Sequence[int]] = None) -> dict:
    """``hipDeviceCanAccessPeer`` matrix and PCI bus ids of ``devices``
    (default: all visible)."""
    _ensure_hip_init()
    if devices is None:
        n = ctypes.c_int(0)
        _check(_hipGetDeviceCount(ctypes.byref(n)), "hipGetDeviceCount")
        devices = list(range(n.value))
    m = []
    for a in devices:
        row = []
        for b in devices:
            if a == b:
                row.append(1)
                continue
            ok = ctypes.c_int(0)
            _check(_hipDeviceCanAccessPeer(ctypes.byref(ok), a, b), "hipDeviceCanAccessPeer")
            row.append(int(ok.value))
        m.append(row)
    return {"devices": list(devices), "can_access_peer": m, "bus": [pci_bus_id(d) for d in devices]}
