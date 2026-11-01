# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-rank driver of the in-kernel CCL kernels (one process per GPU).

A :class:`Rank` owns the torch device, the ``rocke.runtime.peer.PeerGroup``
(IPC arenas, gloo rendezvous), a compile cache and the buffers of each
collective spec. It launches on torch's current stream so torch events time
rocKE launches and RCCL calls alike, and checks every result bit for bit
against the CPU references of ``kernels.gfx950.qwen38_mono.ccl.ref`` (every
rank can compute every rank's input: inputs come from a seed).

Needs the rocKE ``library/`` root on ``sys.path``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

RTC_HZ = 100_000_000


def _lib():
    try:
        from kernels.gfx950.qwen38_mono.ccl import collectives as C
        from kernels.gfx950.qwen38_mono.ccl import layout as L
        from kernels.gfx950.qwen38_mono.ccl import lint as LI
        from kernels.gfx950.qwen38_mono.ccl import ref as RF
    except ImportError as e:  # pragma: no cover
        raise ImportError("the CCL example needs the rocKE library/ root on sys.path") from e
    return C, L, LI, RF


C, L, LI, RF = _lib()


def code_digest() -> str:
    """Short hash of the CCL sources this process runs (device code, host
    runtime, this harness): provenance of every suite / benchmark row."""
    import kernels.gfx950.qwen38_mono.ccl as pkg
    from rocke.runtime import peer as P

    h = hashlib.sha256()
    files = sorted(
        [os.path.join(os.path.dirname(pkg.__file__), f) for f in os.listdir(os.path.dirname(pkg.__file__)) if f.endswith(".py")]
        + [P.__file__, __file__, os.path.join(os.path.dirname(__file__), "bench_ccl.py")]
        + [f for f in [os.path.join(os.path.dirname(__file__), "bench_ccl_lm.py")] if os.path.exists(f)]
    )
    for f in files:
        with open(f, "rb") as fh:
            h.update(os.path.basename(f).encode() + b"\0" + fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def src_wire(spec, rank: int, seed: int) -> np.ndarray:
    """Rank ``rank``'s input of the first collective (wire dtype; bf16 bits as
    uint16), from ``seed`` (every rank can rebuild every rank's input)."""
    h = int.from_bytes(hashlib.sha256(f"{seed}/{rank}/{spec.rows}/{spec.cols}/{spec.dtype}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(h)
    x = rng.standard_normal(spec.rows * spec.cols).astype(np.float32) * np.float32(1 + rank)
    return RF.f32_to_wire(x, spec.dtype)


@dataclass
class Built:
    spec: object
    kernel: object
    fn: object
    nctas: int
    threads: int
    lay: object


class Rank:
    def __init__(self, *, device: Optional[int] = None, mem: str = "uncached", arena_bytes: int = 256 << 20, group=None):
        import torch
        import torch.distributed as dist

        self.torch, self.dist = torch, dist
        self.pg = group  # CPU (gloo) group for objects and barriers; None = the default group
        self.rank = dist.get_rank() if group is None else dist.get_rank(group)
        self.ranks = dist.get_world_size(group)
        local = int(os.environ.get("LOCAL_RANK", self.rank))
        self.device = local if device is None else device
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize()
        from rocke.runtime import peer as P
        from rocke.runtime.hip_module import Runtime

        self.P = P
        self.rt = Runtime()
        self.group = P.PeerGroup(self.rank, self.ranks, arena_bytes, mem=mem, rendezvous="gloo", group=group, device=self.device)
        self.mem = mem
        self._cache: Dict[str, Built] = {}
        self._active: Optional[str] = None  # the layout that last ran on the arena
        self._mods = []
        self.bufs: Dict[str, object] = {}
        self.utabs: Dict[tuple, object] = {}
        # all-reduce chain growth (ref.chain_scales): 1.0 = W3's contracting chain; CCL-LM gates set ref.LM_GROWTH
        self.chain_growth = 1.0

    # ------------------------------------------------------------ plumbing
    @property
    def stream(self) -> int:
        return int(self.torch.cuda.current_stream().cuda_stream)

    def barrier(self) -> None:
        self.group.barrier()

    def sync(self) -> None:
        self.torch.cuda.synchronize()

    def all_gather(self, obj) -> list:
        out = [None] * self.ranks
        self.dist.all_gather_object(out, obj, group=self.pg)
        return out

    def build(self, spec, builder, rename: str = "") -> Built:
        """Compile (cached by name). ``rename``: a suffix of the kernel symbol
        (the same IR compiled again under another name: the recompiled
        control arm of a benchmark)."""
        key = spec.name() + rename
        b = self._cache.get(key)
        if b is not None:
            return b
        from rocke.helpers.compile import compile_kernel

        lay = L.make_layout([spec])
        if lay.nbytes > self.group.nbytes:
            raise ValueError(f"{key}: layout needs {lay.nbytes} B, arena has {self.group.nbytes}")
        k = builder(spec, lay)
        if rename:
            k.name = k.name + rename
        # peer-address + receive-arena lint on every build; only the declared poll_cpol lever
        # builds (sc1, none) get the explicit waiver for local poll scope
        LI.check(k, allow_local_scope=spec.knobs.poll_cpol != "sc0sc1")
        art = compile_kernel(k, arch="gfx950", backend=C.backend_for(k))
        mod = self.rt.load_module(art.hsaco)
        self._mods.append(mod)
        fn = mod.get_function(art.kernel_name)
        nctas = 1 if isinstance(spec, L.ArgmaxSpec) else spec.grid()
        b = self._cache[key] = Built(spec, k, fn, nctas, spec.threads, lay)
        return b

    def with_ctas(self, b: Built, nctas: int) -> Built:
        """The same build launched on ``nctas`` CTAs (``xbuf``-carried
        builds only: every wave loops over its units)."""
        if getattr(b.spec, "reg_carry", False):
            raise ValueError("a register-carried chain needs nctas = spec.grid()")
        return replace(b, nctas=nctas)

    def quiesce_for(self, b: Built) -> None:
        """Before a collective whose arena layout differs from the one that
        ran last: synchronize and barrier on every rank. Two layouts overlap
        in the arena, so a rank already running the new one could otherwise
        overwrite slots a slower rank still polls under the old one. A
        lamport build (empty = zero) also needs its payload regions zeroed:
        other layouts leave tags and data there (``lamport_clear``)."""
        if self._active != b.spec.name():
            self.sync()
            self.barrier()
            if getattr(b.spec.knobs, "form", "") == "lamport":
                self.lamport_clear(b)
            self._active = b.spec.name()

    def lamport_clear(self, b: Built) -> None:
        """Zero this rank's payload regions of a lamport build's exchanges,
        then synchronize and barrier (every rank, before any rank pushes).
        A successful lamport launch leaves them zero again (every pushed line
        is consumed and reset in the same launch)."""
        for ex in b.lay.exchanges:
            self.rt.memset(self.group.base + ex.payload, 0, ex.payload_bytes)
        self.sync()
        self.barrier()

    def reset(self, epoch: int = 1) -> None:
        """Collective group reset (``PeerGroup.reset``) plus the striped
        grid-sync words (``gsb``, device memory)."""
        self.group.reset(epoch=epoch)
        if "gsb" in self.bufs:
            self.bufs["gsb"].zero_()
            self.sync()
        self._active = None

    def calibrate_flips(self, pages) -> int:
        """IOD-home calibration of arena pages (``homing="dual"``): bit p of
        the result names the IOD-local flag copy of page p (the same-XCD
        ping-pong of ``persistent_mailbox.harness.calibrate_iod_home`` on this
        rank's arena). It overwrites arena words, so it ends with the
        collective ``reset`` (call on every rank)."""
        from rocke.examples.gfx950.persistent_mailbox.harness import Dev, calibrate_iod_home

        dev = Dev()
        flips = 0
        for p in sorted(set(pages)):
            f = calibrate_iod_home(dev, "gfx950", self.group.base + p * (2 << 20), 2 << 20)
            flips |= (int(f[0]) & 1) << p
        dev.close()
        self.sync()
        self.reset()
        return flips

    def tensor(self, name: str, nbytes: int):
        t = self.bufs.get(name)
        if t is None or t.numel() < nbytes:
            t = self.bufs[name] = self.torch.zeros(max(nbytes, 64), dtype=self.torch.uint8, device=f"cuda:{self.device}")
        return t

    def put(self, name: str, arr: np.ndarray):
        t = self.tensor(name, arr.nbytes)
        t[: arr.nbytes].copy_(self.torch.from_numpy(np.frombuffer(arr.tobytes(), np.uint8).copy()))
        return t

    def get(self, name: str, nbytes: int, dtype) -> np.ndarray:
        return self.bufs[name][:nbytes].cpu().numpy().view(dtype).copy()

    def read_arena(self, off: int, nbytes: int, rank: Optional[int] = None) -> np.ndarray:
        base = self.group.bases[self.rank if rank is None else rank]
        buf = (ctypes.c_uint8 * nbytes)()
        self.rt.memcpy_d2h(buf, base + off, nbytes)
        return np.frombuffer(bytes(buf), np.uint8).copy()

    def write_arena(self, off: int, data: np.ndarray) -> None:
        raw = data.tobytes()
        host = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        self.rt.memcpy_h2d(self.group.base + off, host, len(raw))

    def control(self) -> Dict[str, int]:
        w = self.read_arena(0, 32).view(np.uint32)
        return {k: int(w[v // 4]) for k, v in L.OFF.items()}

    # ------------------------------------------------------------- launches
    def args(self, b: Built, iters: int, extra: Dict[str, object], *, dl_first_s: float = 10.0, dl_s: float = 1.0, flips: int = 0) -> bytes:
        v = dict(self.group.kernargs())
        v.update(nctas=b.nctas, iters=iters, flips=flips, dl_first=int(dl_first_s * RTC_HZ), dl=int(dl_s * RTC_HZ))
        v["stamps"] = self.tensor("stamps", b.nctas * L.stamp_cols(b.spec.knobs) * 8).data_ptr()
        if b.spec.knobs.hooks:
            v["htab"] = self.tensor("htab", 64).data_ptr()
        if getattr(b.spec.knobs, "gsync", "word") == "stripe":
            v["gsb"] = self.tensor("gsb", L.GSB_WORDS * 4).data_ptr()  # zero at creation; each launch's last CTA re-zeroes it
        if getattr(b.spec.knobs, "place", "rr") == "table":
            key = (b.spec.name(), b.nctas)
            if key not in self.utabs:
                self.set_utab(b)
            v["utab"] = self.utabs[key].data_ptr()
        v.update(extra)
        return C.pack_kernargs(b.kernel, v)

    def set_utab(self, b: Built, shares=None, table=None) -> list:
        """``place="table"`` builds: the unit table of ``b`` at its CTA count
        (``layout.xcd_table`` from per-XCD ``shares``, or an explicit
        ``table``); kept per CTA count until replaced."""
        tab = table if table is not None else L.xcd_table(b.spec.units, b.nctas, b.spec.knobs.waves, shares)
        t = self.torch.tensor(tab, dtype=self.torch.int32, device=f"cuda:{self.device}")
        self.utabs[(b.spec.name(), b.nctas)] = t
        return tab

    def launch(self, b: Built, args: bytes) -> None:
        self.rt.launch(b.fn, (b.nctas, 1, 1), (b.threads, 1, 1), args, stream=self.stream)

    def set_hooks(self, drop_use: int = -1, abort_use: int = -1, salt: int = 0, smax: int = 0, late_cta: int = -1, late_units: int = 0) -> None:
        self.put("htab", np.array([drop_use, abort_use, salt, smax, late_cta, late_units, 0, 0], np.int32))

    # ---------------------------------------------------------- all-reduce
    def ar_prepare(self, spec, seed: int):
        b = self.build(spec, C.allreduce_kernel_in)
        self.quiesce_for(b)
        x = src_wire(spec, self.rank, seed)
        self.put("src", x)
        n = spec.rows * spec.cols
        self.tensor("dst", n * 4)
        self.tensor("xbuf", getattr(spec, "in_dwords", spec.payload_dwords) * 4)
        return b

    def ar_args(self, b: Built, iters: int, **kw) -> bytes:
        sc = RF.chain_scales(self.ranks if b.spec.ranks == self.ranks else b.spec.ranks, self.chain_growth)[self.rank if b.spec.ranks > 1 else 0]
        extra = {"src": self.bufs["src"].data_ptr(), "dst": self.bufs["dst"].data_ptr(), "xbuf": self.bufs["xbuf"].data_ptr(), "scale": sc}
        return self.args(b, iters, extra, **kw)

    def ar_ref(self, spec, seed: int, iters: int, xs=None) -> np.ndarray:
        xs = xs if xs is not None else [src_wire(spec, r, seed) for r in range(spec.ranks)]
        out, _ = RF.allreduce_chain(xs, spec.dtype, iters, RF.chain_scales(spec.ranks, self.chain_growth), twoshot=spec.knobs.ar == "twoshot",
                                    wire=getattr(spec, "wire_dtype", spec.dtype), canon=spec.knobs.form == "lamport")
        return out

    def ar_result(self, spec) -> np.ndarray:
        return self.get("dst", spec.rows * spec.cols * 4, np.float32)

    # ---------------------------------------------------------- all-gather
    def ag_prepare(self, spec, seed: int):
        b = self.build(spec, C.allgather_kernel_in)
        self.quiesce_for(b)
        self.put("src", src_wire(spec, self.rank, seed))
        self.tensor("dst", spec.payload_dwords * 4 * spec.ranks)
        self.tensor("xbuf", spec.payload_dwords * 4)
        return b

    def ag_args(self, b: Built, iters: int, **kw) -> bytes:
        extra = {"src": self.bufs["src"].data_ptr(), "dst": self.bufs["dst"].data_ptr(), "xbuf": self.bufs["xbuf"].data_ptr()}
        return self.args(b, iters, extra, **kw)

    def ag_ref(self, spec, seed: int, iters: int) -> np.ndarray:
        xs = [src_wire(spec, r, seed).reshape(spec.rows, spec.cols) for r in range(spec.ranks)]
        out, _ = RF.allgather_chain(xs, iters)
        return np.ascontiguousarray(out)

    def ag_result(self, spec) -> np.ndarray:
        dt = np.uint16 if spec.dtype == "bf16" else np.float32
        return self.get("dst", spec.payload_dwords * 4 * spec.ranks, dt).reshape(spec.rows, spec.ranks * spec.cols)

    # -------------------------------------------------------------- argmax
    def amax_inputs(self, spec, seed: int, iters: int, rank: int):
        """Per-rank (max, global id) per row with planted cross-rank ties."""
        rng = np.random.default_rng(seed * 131 + 7)
        base = rng.integers(0, 6, size=(iters, spec.rows)).astype(np.float32)  # small range: many ties
        rr = np.random.default_rng(seed * 131 + 11 + rank)
        vals = (base + rr.integers(0, 2, size=base.shape)).astype(np.float32)
        vocab_shard = 1 << 16
        ids = (rank * vocab_shard + rr.integers(0, vocab_shard, size=base.shape)).astype(np.int32)
        return vals, ids

    def amax_launch(self, b: Built, seed: int, iters: int, **kw):
        vals, ids = self.amax_inputs(b.spec, seed, iters, self.rank)
        self.put("vals", vals)
        self.put("ids", ids)
        self.tensor("ovals", vals.nbytes)
        self.tensor("oids", ids.nbytes)
        extra = {k: self.bufs[k].data_ptr() for k in ("vals", "ids", "ovals", "oids")}
        self.launch(b, self.args(b, iters, extra, **kw))
        return vals, ids

    def amax_run(self, spec, seed: int, iters: int, **kw):
        b = self.build(spec, C.argmax_kernel_in)
        self.quiesce_for(b)
        vals, ids = self.amax_launch(b, seed, iters, **kw)
        self.sync()
        ov = self.get("ovals", vals.nbytes, np.float32).reshape(iters, spec.rows)
        oi = self.get("oids", ids.nbytes, np.int32).reshape(iters, spec.rows)
        allv = [self.amax_inputs(spec, seed, iters, r) for r in range(spec.ranks)]
        wv, wi = np.empty_like(ov), np.empty_like(oi)
        for it in range(iters):
            wv[it], wi[it] = RF.argmax_merge(np.stack([a[0][it] for a in allv]), np.stack([a[1][it] for a in allv]))
        return ov, oi, wv, wi

    # --------------------------------------------------------------- stamps
    def stamps(self, b: Built) -> np.ndarray:
        sc = L.stamp_cols(b.spec.knobs)
        return self.get("stamps", b.nctas * sc * 8, np.int64).reshape(b.nctas, sc)

    def close(self) -> None:
        self.sync()
        for m in self._mods:
            m.unload()
        self._mods.clear()
        self.group.close()


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def gpu_inventory(device: int) -> dict:
    """Bus id and the processes amd-smi sees on our GPUs (idle check)."""
    from rocke.runtime import peer as P

    info = {"device": device, "bus": P.pci_bus_id(device)}
    return info


def wait_launch_done(rank: Rank, timeout_s: float) -> bool:
    """Poll a stream event up to ``timeout_s``; False on a hang."""
    ev = rank.torch.cuda.Event()
    ev.record()
    t_end = time.monotonic() + timeout_s
    while not ev.query():
        if time.monotonic() > t_end:
            return False
        time.sleep(0.001)
    return True


def first_mismatch(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    av, bv = np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)
    if av.shape != bv.shape:
        return -1
    d = np.nonzero(av != bv)[0]
    return int(d[0]) if d.size else None
