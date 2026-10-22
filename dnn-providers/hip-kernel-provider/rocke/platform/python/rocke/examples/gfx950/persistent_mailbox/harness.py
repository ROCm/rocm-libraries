# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch-free device harness for the mailbox microkernels.

Every measurement is a device ``s_memrealtime`` tick difference (100 MHz);
results are returned in ticks plus derived nanoseconds, and every run checks
its numerics (payloads, checksums, stamp order) before reporting a time.
"""

from __future__ import annotations

import ctypes
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rocke.core.ir import KernelDef, uses_python_only_ops
from rocke.helpers import mailbox as M
from rocke.helpers.compile import compile_kernel

from . import kernels as K

RTC_HZ = 100_000_000
NS_PER_TICK = 1e9 / RTC_HZ


def compile_for(kernel: KernelDef, arch: str = "gfx950", backend: str | None = None):
    """Compile with ``backend`` (default: ``python`` when the kernel uses a
    Python-only opcode, else ``cpp`` when the binding exists, else
    ``python``). Returns (artifact, backend actually used)."""
    if backend is None:
        if uses_python_only_ops(kernel):
            backend = "python"
        else:
            import importlib.util

            backend = "cpp" if importlib.util.find_spec("rocke_engine") else "python"
    return compile_kernel(kernel, arch=arch, backend=backend), backend


def env_problem() -> str | None:
    """Why this process cannot run the GPU microkernels, or ``None``.

    Two known misconfigurations (both seen in the q38 venv job mode): the
    HIP runtime ``rocke.runtime.hip_module`` loads is not the one whose COMGR
    ``ROCKE_COMGR_LIB`` pins (two different ``libamd_comgr`` files end up
    mapped, and code objects or allocations fail), or a first
    ``hipMalloc`` + ``hipMemset`` fails. Fix: point ``ROCKE_HIP_LIB`` at the
    ``libamdhip64.so`` that ships with that COMGR (torch's ``lib/``)."""
    hint = (
        "set ROCKE_HIP_LIB to the libamdhip64.so matching ROCKE_COMGR_LIB "
        "(e.g. torch's lib/libamdhip64.so)"
    )
    try:
        from rocke.runtime import comgr
        from rocke.runtime.hip_module import Runtime

        rt = Runtime()
        p = rt.alloc(64)
        rt.memset(p, 0, 64)
        rt.sync()
        rt.free(p)
        comgr._resolve_lib()
    except Exception as e:  # noqa: BLE001
        return f"HIP/COMGR setup failed ({type(e).__name__}: {str(e)[:160]}); {hint}"
    import os

    maps = "/proc/self/maps"
    if os.path.exists(maps):
        libs = set()
        with open(maps) as fh:
            for ln in fh:
                path = ln.split()[-1] if ln.strip() else ""
                if "libamd_comgr" in path:
                    libs.add(os.path.realpath(path))
        if len(libs) > 1:
            return f"two COMGR libraries are loaded ({sorted(libs)}); {hint}"
    return None


class DeviceHang(RuntimeError):
    """A launch did not complete within the watchdog."""


class Dev:
    """Allocations + launches over ``rocke.runtime.hip_module.Runtime``."""

    def __init__(self, watchdog_s: float | None = None):
        self.watchdog_s = watchdog_s
        from rocke.runtime.hip_module import Runtime

        self.rt = Runtime()
        self._bufs: list[int] = []
        self._mods = []
        self._fns: dict = {}

    def alloc(self, nbytes: int, zero: bool = True) -> int:
        p = self.rt.alloc(nbytes)
        self._bufs.append(p)
        if zero:
            self.rt.memset(p, 0, nbytes)
        return p

    def put(self, arr: np.ndarray) -> int:
        p = self.alloc(arr.nbytes, zero=False)
        host = (ctypes.c_uint8 * arr.nbytes).from_buffer_copy(arr.tobytes())
        self.rt.memcpy_h2d(p, host, arr.nbytes)
        return p

    def write(self, p: int, arr: np.ndarray) -> None:
        host = (ctypes.c_uint8 * arr.nbytes).from_buffer_copy(arr.tobytes())
        self.rt.memcpy_h2d(p, host, arr.nbytes)

    def get(self, p: int, nbytes: int, dtype) -> np.ndarray:
        out = (ctypes.c_uint8 * nbytes)()
        self.rt.memcpy_d2h(out, p, nbytes)
        return np.frombuffer(bytes(out), dtype=dtype).copy()

    def function(self, art):
        key = (art.kernel_name, art.hsaco)
        fn = self._fns.get(key)
        if fn is None:
            mod = self.rt.load_module(art.hsaco)
            self._mods.append(mod)
            fn = self._fns[key] = mod.get_function(art.kernel_name)
        return fn

    def launch(self, fn, grid: int, block: int, args: bytes, stream: int = 0):
        self.rt.launch(fn, (grid, 1, 1), (block, 1, 1), args, stream=stream)

    def sync(self) -> None:
        """Wait for the device. With ``watchdog_s`` set, poll a completion
        event and raise :class:`DeviceHang` after that many seconds instead
        of blocking forever on a kernel that never finishes (the process must
        then exit: the hung kernel keeps its queue)."""
        if self.watchdog_s is None:
            self.rt.sync()
            return
        import time

        ev = self.rt.event()
        ev.record(0)
        t_end = time.monotonic() + self.watchdog_s
        while not ev.query():
            if time.monotonic() > t_end:
                raise DeviceHang(f"device work not done after {self.watchdog_s} s")
            time.sleep(0.001)
        ev.destroy()
        self.rt.sync()

    def close(self) -> None:
        try:
            self.sync()
        except DeviceHang:
            return  # a kernel still spins on our buffers: leak them, never free
        for m in self._mods:
            m.unload()
        for p in self._bufs:
            self.rt.free(p)
        self._mods.clear()
        self._fns.clear()
        self._bufs.clear()


# ------------------------------------------------------------------ gate


def run_gate(dev: Dev, form: str, arch: str, backend: str | None = None) -> dict:
    art, used = compile_for(K.build_gate_kernel(form), arch, backend)
    src = (np.arange(64, dtype=np.int32) * 5 - 17).astype(np.int32)
    ps = dev.put(src)
    pd = dev.alloc(256)
    dev.launch(dev.function(art), 1, 64, struct.pack("<QQi", ps, pd, 256))
    dev.sync()
    got = dev.get(pd, 256, np.int32)
    ok = bool(np.array_equal(got, src * 2 + 1))
    return {"form": form, "backend": used, "ok": ok}


# ------------------------------------------------------------- ping-pong


@dataclass
class PingPongResult:
    name: str
    backend: str
    a: int
    bb: int
    xcc_a: int
    xcc_b: int
    iters: int
    rtt_ticks: float
    ok: bool
    monotone: bool

    @property
    def rtt_ns(self) -> float:
        return self.rtt_ticks * NS_PER_TICK


def run_pingpong(
    dev: Dev,
    spec: K.PingPongSpec,
    arch: str,
    *,
    a: int,
    bb: int,
    iters: int,
    reps: int = 3,
    grid: int | None = None,
    backend: str | None = None,
    base0: int = 1 << 20,
    box: int | None = None,
    art=None,
) -> list[PingPongResult]:
    """``reps`` launches; returns one result per launch (median is the
    caller's business). ``box``: an existing >= 64 B device buffer to use as
    the mailbox (its address decides the memory channel, which moves the
    round trip); default a fresh allocation. ``art``: a precompiled
    ``(artifact, backend)``."""
    art, used = art or compile_for(K.build_pingpong(spec), arch, backend)
    fn = dev.function(art)
    grid = grid or max(a, bb) + 1
    if box is None:
        raw = dev.alloc(32768)
        box = (raw + 16383) & ~16383
    out = dev.alloc(K.PP_COLS * 8 * grid)
    res = []
    base = base0
    for _ in range(reps):
        dev.rt.memset(out, 0, K.PP_COLS * 8 * grid)
        dev.launch(fn, grid, 64, struct.pack("<QQiiii", box, out, a, bb, base, iters))
        dev.sync()
        o = dev.get(out, K.PP_COLS * 8 * grid, np.int64).reshape(grid, K.PP_COLS)
        last_tag = base + iters
        ok = int(o[a, 3]) == iters + 1000 and int(o[bb, 3]) == iters
        # stamps: A's t0 <= B's first-seen <= A's t1 (one clock, all XCDs)
        mono = int(o[a, 0]) <= int(o[bb, 2]) <= int(o[a, 1])
        res.append(
            PingPongResult(
                spec.name(),
                used,
                a,
                bb,
                int(o[a, 4]),
                int(o[bb, 4]),
                iters,
                (int(o[a, 1]) - int(o[a, 0])) / iters,
                ok,
                mono,
            )
        )
        base = last_tag + 1
    return res


# ------------------------------------------------------------ all-to-all


@dataclass
class AllToAllResult:
    name: str
    backend: str
    grid: int
    phases: int
    round_ticks: float
    ok: bool
    xcc_hist: dict[int, int]

    @property
    def round_ns(self) -> float:
        return self.round_ticks * NS_PER_TICK


def run_all_to_all(
    dev: Dev,
    spec: K.AllToAllSpec,
    arch: str,
    *,
    grid: int = 256,
    phases: int = 200,
    reps: int = 3,
    backend: str | None = None,
    base0: int = 1 << 22,
    art=None,
    calibrate: bool = True,
) -> list[AllToAllResult]:
    art, used = art or compile_for(K.build_all_to_all(spec), arch, backend)
    fn = dev.function(art)
    box_bytes = phases * spec.round_pairs * 8
    raw = dev.alloc(box_bytes + 16384)
    box = (raw + 16383) & ~16383  # numa_copies relies on 16 KiB alignment
    out = dev.alloc(K.A2A_COLS * 8 * grid)
    chk_n = phases * grid * spec.threads if spec.check else 64
    chk = dev.alloc(chk_n * 4)
    if spec.numa_copies == 2:
        flips = np.zeros(phases, dtype=np.int32)
        if calibrate:
            page_flag = calibrate_iod_home(dev, arch, box, box_bytes)
            for ph in range(phases):
                flips[ph] = page_flag[(ph * spec.round_pairs * 8) >> 21]
        flip_p = dev.put(flips)
        head = struct.pack("<QQQQ", box, out, chk, flip_p)
    else:
        head = struct.pack("<QQQ", box, out, chk)
    res = []
    base = base0
    for _ in range(reps):
        dev.launch(
            fn, grid, spec.threads, head + struct.pack("<iii", base, phases, grid)
        )
        dev.sync()
        o = dev.get(out, K.A2A_COLS * 8 * grid, np.int64).reshape(grid, K.A2A_COLS)
        span = int(o[:, 1].max()) - int(o[:, 0].min())
        ok = bool((o[:, 1] > 0).all())
        if spec.check:
            c = dev.get(chk, chk_n * 4, np.int32).reshape(phases, grid, spec.threads)
            want = np.array(
                [
                    [
                        K.a2a_expected_thread_sum(spec, ph, t)
                        for t in range(spec.threads)
                    ]
                    for ph in range(phases)
                ],
                dtype=np.int64,
            ).astype(np.int32)
            ok = ok and bool((c == want[:, None, :]).all())
            dev.rt.memset(chk, 0, chk_n * 4)
        hist: dict[int, int] = {}
        for x in o[:, 2]:
            hist[int(x)] = hist.get(int(x), 0) + 1
        res.append(
            AllToAllResult(spec.name(), used, grid, phases, span / phases, ok, hist)
        )
        base += phases + 1
    return res


# ------------------------------------------------- IOD home calibration


def calibrate_iod_home(
    dev: Dev,
    arch: str,
    buf: int,
    nbytes: int,
    *,
    iters: int = 1500,
    knobs: M.MailboxKnobs | None = None,
) -> list[int]:
    """Per 2 MiB page of ``[buf, buf + nbytes)`` (``buf`` 2 MiB-relative
    offsets counted from ``buf``): 0 if the 8 KiB block at page offset 0 is
    homed at the IOD of XCD 0 (a same-XCD ping-pong on XCD 0 is faster there
    than at +8 KiB), else 1. On gfx950 the home IOD of an address is address
    bit 13 XOR a per-page constant, so this one bit per page names the home of
    every 8 KiB block of the page. Uses CTAs 0 and 8 (same XCD under
    round-robin dispatch; checked). ``knobs`` picks the ping-pong's mailbox
    implementation: the ``ir`` one when registered (its loop has the least
    timing noise), else the default ``asm``."""
    if knobs is None:
        knobs = M.MailboxKnobs(impl="ir" if "ir" in M.registered_impls() else "asm")
    spec = K.PingPongSpec(knobs=knobs)
    art = compile_for(K.build_pingpong(spec), arch)
    flags = []
    base = 1 << 24
    for page in range((nbytes + (1 << 21) - 1) >> 21):
        t = []
        for off in (0, 8192):
            addr = buf + (page << 21) + off
            if addr + 64 > buf + nbytes + (1 << 21):
                break
            rs = run_pingpong(
                dev,
                spec,
                arch,
                a=0,
                bb=8,
                iters=iters,
                reps=3,
                base0=base,
                box=addr,
                art=art,
            )
            base += 3 * (iters + 1)
            if not all(r.ok and r.monotone for r in rs) or rs[0].xcc_a != rs[0].xcc_b:
                raise RuntimeError("IOD calibration ping-pong failed or not same-XCD")
            t.append(sorted(r.rtt_ticks for r in rs)[1])
        flags.append(0 if len(t) < 2 or t[0] <= t[1] else 1)
    return flags


# ------------------------------------------------------------ launch gap


class _Graph:
    """Minimal hipGraph capture of launches on a private stream (ctypes)."""

    def __init__(self, dev: Dev):
        from rocke.runtime import hip_module as H

        self.lib = H._resolve_hip()
        self.dev = dev
        self.stream = ctypes.c_void_p()
        self._chk(
            self.lib.hipStreamCreate(ctypes.byref(self.stream)), "hipStreamCreate"
        )
        self.graph = ctypes.c_void_p()
        self.exec = ctypes.c_void_p()

    def _chk(self, rc, where):
        if rc != 0:
            raise RuntimeError(f"{where} failed: {rc}")

    def capture(self, fn_launches) -> None:
        self._chk(
            self.lib.hipStreamBeginCapture(self.stream, 0), "hipStreamBeginCapture"
        )
        fn_launches(self.stream.value)
        self._chk(
            self.lib.hipStreamEndCapture(self.stream, ctypes.byref(self.graph)),
            "hipStreamEndCapture",
        )
        self._chk(
            self.lib.hipGraphInstantiate(
                ctypes.byref(self.exec), self.graph, None, None, ctypes.c_size_t(0)
            ),
            "hipGraphInstantiate",
        )

    def replay(self) -> None:
        self._chk(self.lib.hipGraphLaunch(self.exec, self.stream), "hipGraphLaunch")
        self._chk(self.lib.hipStreamSynchronize(self.stream), "hipStreamSynchronize")

    def close(self) -> None:
        if self.exec:
            self.lib.hipGraphExecDestroy(self.exec)
        if self.graph:
            self.lib.hipGraphDestroy(self.graph)
        self.lib.hipStreamDestroy(self.stream)


@dataclass
class LaunchGapResult:
    mode: str
    launches: int
    grid: int
    gap_ticks: float  # median over launch pairs: min start(i+1) - max end(i)
    period_ticks: float  # median over pairs: min start(i+1) - min start(i)
    span_ticks: float  # median single-launch span: max end - min start

    @property
    def gap_ns(self) -> float:
        return self.gap_ticks * NS_PER_TICK


def run_launch_gap(
    dev: Dev,
    arch: str,
    *,
    grid: int = 256,
    threads: int = 512,
    launches: int = 50,
    mode: str = "stream",
    backend: str | None = None,
) -> LaunchGapResult:
    art, _ = compile_for(K.build_stamp_kernel(threads), arch, backend)
    fn = dev.function(art)
    out = dev.alloc(launches * grid * 16)

    def issue(stream: int) -> None:
        for i in range(launches):
            dev.launch(
                fn, grid, threads, struct.pack("<Qii", out, i, grid), stream=stream
            )

    if mode == "stream":
        issue(0)
        dev.sync()
        dev.rt.memset(out, 0, launches * grid * 16)
        issue(0)
        dev.sync()
    elif mode == "graph":
        g = _Graph(dev)
        try:
            g.capture(issue)
            g.replay()
            dev.rt.memset(out, 0, launches * grid * 16)
            g.replay()
        finally:
            g.close()
    else:
        raise ValueError(mode)
    o = dev.get(out, launches * grid * 16, np.int64).reshape(launches, grid, 2)
    starts, ends = o[:, :, 0].min(axis=1), o[:, :, 1].max(axis=1)
    gaps = starts[1:] - ends[:-1]
    periods = starts[1:] - starts[:-1]
    spans = ends - starts
    return LaunchGapResult(
        mode,
        launches,
        grid,
        float(np.median(gaps)),
        float(np.median(periods)),
        float(np.median(spans)),
    )


# -------------------------------------------------------------- tag wrap


def run_tag_wrap(
    dev: Dev,
    arch: str,
    steps: Sequence[int],
    layers: Sequence[int],
    *,
    delay: int = 4,
    backend: str | None = None,
    knobs=None,
    check_layers: bool = True,
) -> list[dict]:
    """Back-to-back launches over (step, layer) in order on one never-cleared
    mailbox. Each record: step, layer, tag, payload, expected, ok.

    The kernel takes ``layer`` at run time, which ``launch_tag`` cannot check,
    so this launcher calls ``mailbox.check_layer`` on every layer (as a real
    layer-kernel launcher must); ``check_layers=False`` only for the
    wrap-hazard mutation test."""
    kn = knobs or M.MailboxKnobs()
    if check_layers:
        for layer in layers:
            M.check_layer(layer)
    art, used = compile_for(K.build_tag_wrap_kernel(kn), arch, backend)
    fn = dev.function(art)
    step_p = dev.alloc(4)
    box = dev.alloc(64)
    out = dev.alloc(16)
    recs = []
    for s in steps:
        for layer in layers:
            dev.write(step_p, np.array([s & 0xFFFFFFFF], dtype=np.uint32))
            dev.launch(fn, 2, 64, struct.pack("<QQQii", step_p, box, out, layer, delay))
            dev.sync()
            o = dev.get(out, 16, np.uint32)
            want_tag = (s * M.LAYER_SLOTS + layer + 1) & 0xFFFFFFFF
            want = (s * 3 + layer) & 0xFFFFFFFF
            recs.append(
                {
                    "step": s,
                    "layer": layer,
                    "tag": int(o[1]),
                    "payload": int(o[0]),
                    "expected": want,
                    "backend": used,
                    "ok": int(o[0]) == want and int(o[1]) == want_tag,
                }
            )
    return recs


# ------------------------------------------------- bulk hand-off + bf16


def _bf16_rne(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def run_bulk(
    dev: Dev,
    spec: K.BulkSpec,
    arch: str,
    *,
    shifts: Sequence[int] = (1, 37, 128),
    steps: int = 20,
    backend: str | None = None,
) -> dict:
    """``steps`` launches per shift (step 1..steps: the rows change every
    step, so a stale read is a mismatch). Checks, per launch: no row word
    mismatched, the bf16 values bit-exact against a numpy RNE reference, the
    agent counters (``cnt[1] == grid * threads``) and the returning system
    counter's tickets (a permutation of ``0 .. grid * threads - 1``)."""
    art, used = compile_for(K.build_bulk_kernel(spec), arch, backend)
    fn = dev.function(art)
    G, T = spec.grid, spec.threads
    data = dev.alloc(G * K.BULK_ROW * 4)
    box = dev.alloc(G * 8)
    bfb = dev.alloc(G * 16)
    cnt = dev.alloc(16)
    tickets = dev.alloc(G * T * 4)
    out = dev.alloc(G * 16)
    rec = {
        "name": spec.name(),
        "backend": used,
        "row_mismatch": 0,
        "bf_mismatch": 0,
        "counter_bad": 0,
        "ticket_bad": 0,
        "launches": 0,
    }
    for shift in shifts:
        for step in range(1, steps + 1):
            dev.write(cnt, np.zeros(4, np.int32))
            dev.launch(
                fn,
                G,
                T,
                struct.pack(
                    "<QQQQQQii", data, box, bfb, cnt, tickets, out, step, shift
                ),
            )
            dev.sync()
            c = dev.get(cnt, 16, np.int32)
            rec["row_mismatch"] += int(c[0])
            rec["counter_bad"] += int(c[1] != G * T) + int(c[2] != G * T)
            tk = np.sort(dev.get(tickets, G * T * 4, np.int32))
            rec["ticket_bad"] += int(not np.array_equal(tk, np.arange(G * T)))
            got = dev.get(out, G * 16, np.float32).reshape(G, 4)
            q = (np.arange(G) + shift) % G
            want = _bf16_rne(
                (step * 13 + q[:, None] * 4 + np.arange(4)[None, :]).astype(np.float32)
                * np.float32(0.3701)
            )
            rec["bf_mismatch"] += int(
                (got.view(np.uint32) != want.view(np.uint32)).sum()
            )
            rec["launches"] += 1
    rec["ok"] = not (
        rec["row_mismatch"]
        or rec["bf_mismatch"]
        or rec["counter_bad"]
        or rec["ticket_bad"]
    )
    return rec


# ------------------------------------------------------- device-op probe


def run_device_ops(
    dev: Dev, arch: str, *, grid: int = 64, backend: str | None = None, seed: int = 0
) -> dict:
    """``build_device_ops_kernel`` against numpy: wave max, fresh copies,
    clocks advancing across ``s_sleep``. Returns per-check pass flags."""
    art, used = compile_for(K.build_device_ops_kernel(), arch, backend)
    fn = dev.function(art)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2**32, size=grid * 64, dtype=np.uint64).astype(np.uint32)
    x[:64] = 7  # an all-equal wave
    x[64:128] = np.arange(64, dtype=np.uint32)  # max in lane 63
    x[128:192] = np.arange(64, dtype=np.uint32)[::-1]  # max in lane 0
    inp = dev.put(x)
    echo = dev.alloc(grid * 64 * 8)
    out = dev.alloc(grid * K.DOPS_COLS * 8)
    dev.launch(fn, grid, 64, struct.pack("<QQQ", inp, echo, out))
    dev.sync()
    e = dev.get(echo, grid * 64 * 8, np.uint32).reshape(grid, 64, 2)
    o = dev.get(out, grid * K.DOPS_COLS * 8, np.int64).reshape(grid, K.DOPS_COLS)
    xs = x.reshape(grid, 64)
    umax = o[:, 0].astype(np.uint64).astype(np.uint32)
    res = {
        "backend": used,
        "umax": bool((umax == xs.max(axis=1)).all()),
        "fresh_v": bool((e[:, :, 0] == xs).all()),
        "fresh_s": bool((e[:, :, 1] == xs[:, :1]).all()),
        # s_sleep N sleeps about 64 N clocks; allow half for clock-rate slack
        "memtime": bool((o[:, 1] >= 32 * K.DOPS_SLEEP).all()),
        "memrealtime": bool((o[:, 2] > 0).all()),
        "xcc": sorted({int(v) for v in o[:, 3]}),
    }
    res["ok"] = all(
        res[k] for k in ("umax", "fresh_v", "fresh_s", "memtime", "memrealtime")
    )
    return res
