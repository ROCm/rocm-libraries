# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""In-kernel clock and device health check for the gfx1250 fp8 GEMM.

Builds one ``fp8_trait_ab`` arm, then a *diagnostic* copy of it whose lowered
LLVM IR has two counter reads added per wave -- shader cycles
(``llvm.readcyclecounter``) and the fixed-rate realtime counter
(``llvm.readsteadycounter``) -- at kernel entry and just before ``ret``. The
four values per wave are stored past the end of C, which the host
over-allocates. The emitter is not involved: the stamps are spliced into the
``.ll`` text and compiled with the same comgr call the sweep uses, so this
works on any arm and changes nothing anyone else builds.

From the stamps it reports, over peak (and optionally sustained) launches:

* **dispatch and finish spread**: when workgroups start and end relative to
  the first; with one round of workgroups they should all start together.
  This compares realtime counters across workgroups, so it is only shown
  when their bases agree: on one boot of the test device a fixed half of the
  workgroups read realtime about half a millisecond ahead of the other half,
  with identical lifetimes. The probe detects that by comparing the stamped
  span of a launch with the event-timed launch, and says so;
* **straggler**: the slowest workgroup, by its own lifetime (a difference on
  one counter, so free of any base offset), and its shader cycles against
  the rest. One workgroup, not a few, so a single degraded unit is not
  averaged away.
  Equal cycles mean a slow workgroup only started late or ran at a lower
  clock; *more* cycles for identical work, at an equal or higher clock, is the
  degraded-device signature the case study describes (§17);
* **clock**: each wave's average shader clock over its lifetime, from cycles
  over realtime with the realtime rate calibrated against the host clock.

The instrumented kernel is verified exactly against the integer reference
before anything is reported, and its time is compared with the plain build's
(interleaved), so the cost of the stamps is measured rather than assumed. The
figures are a diagnostic, not a benchmark: keep them out of the repository.

  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.clock_probe
  python3 -m rocke.examples.gfx1250.gemm_perf_fp8.clock_probe --arm alternate_issuers --sustained
"""

from __future__ import annotations

import argparse
import re
import statistics
import struct
import time
from typing import Any

from .fp8_trait_ab import ARCH, ARMS, ISA, arm_spec, interleaved_order

#: Per wave: [cycles at start, realtime at start, cycles at end, realtime at end].
STAMPS_PER_WAVE = 4


def instrument_ll(ll: str, *, waves_per_wg: int, tile_n: int) -> str:
    """Splice per-wave clock stamps into a universal-GEMM kernel's LLVM IR.

    Slot of wave ``w`` in workgroup ``(x, y)``: ``(y * N/tile_n + x) *
    waves_per_wg + w``; its four i64 stamps start at i64 index
    ``M*N/4 + 4*slot``, i.e. just past C's ``M*N`` bf16 elements.
    """
    sig = re.search(r"^define amdgpu_kernel void @[\w.]+\((.*)\)", ll, re.MULTILINE)
    if sig is None:
        raise ValueError("no amdgpu_kernel definition in the IR")
    for arg in ("%C", "%M", "%N"):
        if not re.search(rf"{re.escape(arg)}\b", sig.group(1)):
            raise ValueError(f"kernel signature has no {arg} argument")
    if (
        ll.count("\nentry:\n") != 1
        or len(re.findall(r"^\s*ret void$", ll, re.MULTILINE)) != 1
    ):
        raise ValueError("expected exactly one entry block and one 'ret void'")

    start = f"""entry:
  %clk.tid = call i32 @llvm.amdgcn.workitem.id.x()
  %clk.wave = lshr i32 %clk.tid, 5
  %clk.wgx = call i32 @llvm.amdgcn.workgroup.id.x()
  %clk.wgy = call i32 @llvm.amdgcn.workgroup.id.y()
  %clk.gx = udiv i32 %N, {tile_n}
  %clk.row = mul i32 %clk.wgy, %clk.gx
  %clk.wg = add i32 %clk.row, %clk.wgx
  %clk.wgw = mul i32 %clk.wg, {waves_per_wg}
  %clk.slot = add i32 %clk.wgw, %clk.wave
  %clk.slot64 = zext i32 %clk.slot to i64
  %clk.off = mul i64 %clk.slot64, {STAMPS_PER_WAVE}
  %clk.m64 = zext i32 %M to i64
  %clk.n64 = zext i32 %N to i64
  %clk.mn = mul i64 %clk.m64, %clk.n64
  %clk.base = lshr i64 %clk.mn, 2
  %clk.i0 = add i64 %clk.base, %clk.off
  %clk.cy0 = call i64 @llvm.readcyclecounter()
  %clk.rt0 = call i64 @llvm.readsteadycounter()
  %clk.p0 = getelementptr inbounds i64, ptr addrspace(1) %C, i64 %clk.i0
  store volatile i64 %clk.cy0, ptr addrspace(1) %clk.p0, align 8
  %clk.p1 = getelementptr inbounds i64, ptr addrspace(1) %clk.p0, i64 1
  store volatile i64 %clk.rt0, ptr addrspace(1) %clk.p1, align 8
"""
    end = """  %clk.cy1 = call i64 @llvm.readcyclecounter()
  %clk.rt1 = call i64 @llvm.readsteadycounter()
  %clk.p2 = getelementptr inbounds i64, ptr addrspace(1) %clk.p0, i64 2
  store volatile i64 %clk.cy1, ptr addrspace(1) %clk.p2, align 8
  %clk.p3 = getelementptr inbounds i64, ptr addrspace(1) %clk.p0, i64 3
  store volatile i64 %clk.rt1, ptr addrspace(1) %clk.p3, align 8
  ret void
"""
    out = ll.replace("\nentry:\n", "\n" + start, 1)
    out = re.sub(r"^\s*ret void\n", end, out, count=1, flags=re.MULTILINE)
    decls = {
        "@llvm.readcyclecounter": "declare i64 @llvm.readcyclecounter()",
        "@llvm.readsteadycounter": "declare i64 @llvm.readsteadycounter()",
        "@llvm.amdgcn.workitem.id.x": "declare i32 @llvm.amdgcn.workitem.id.x()",
        "@llvm.amdgcn.workgroup.id.x": "declare i32 @llvm.amdgcn.workgroup.id.x()",
        "@llvm.amdgcn.workgroup.id.y": "declare i32 @llvm.amdgcn.workgroup.id.y()",
    }
    for name, decl in decls.items():
        if not re.search(rf"^declare [^\n]*{re.escape(name)}\(", out, re.MULTILINE):
            out += f"\n{decl}\n"
    return out


def summarize(np, stamps, *, waves_per_wg: int, rt_hz: float) -> dict[str, float]:
    """One launch's stamps ``[n_waves, 4]`` -> spreads (ms), cycles, clock (MHz).

    ``span_ms`` (first start to last end) compares counters across
    workgroups; everything chosen or compared per wave or per workgroup uses
    differences on one wave's counters and is immune to a base offset.
    """
    st = stamps.astype(np.int64)
    cycles = st[:, 2] - st[:, 0]
    real = st[:, 3] - st[:, 1]
    clock_mhz = cycles / (real / rt_hz) / 1e6
    t0 = st[:, 1].min()
    start = (st[:, 1] - t0) / rt_hz * 1e3
    end = (st[:, 3] - t0) / rt_hz * 1e3
    life = real / rt_hz * 1e3
    wg_life = life.reshape(-1, waves_per_wg).max(1)
    slow = np.zeros(wg_life.size, dtype=bool)
    slow[int(np.argmax(wg_life))] = True
    slow_w = np.repeat(slow, waves_per_wg)
    return {
        "start_p99_ms": float(np.percentile(start, 99)),
        "start_max_ms": float(start.max()),
        "end_med_ms": float(np.median(end)),
        "end_max_ms": float(end.max()),
        "span_ms": float(end.max()),
        "life_med_ms": float(np.median(life)),
        "life_max_ms": float(life.max()),
        "clock_med_mhz": float(np.median(clock_mhz)),
        "clock_slow_mhz": float(np.median(clock_mhz[slow_w])),
        "clock_rest_mhz": float(np.median(clock_mhz[~slow_w])),
        "cycles_slow": float(np.median(cycles[slow_w])),
        "cycles_rest": float(np.median(cycles[~slow_w])),
    }


def realtime_bases_agree(s: dict[str, float], launch_ms: float) -> bool:
    """Whether the stamped span of a launch fits its event-timed duration.

    A span well beyond the launch's own time cannot be real: some
    workgroups' realtime counters are offset from the others'.
    """
    return s["span_ms"] <= 1.15 * launch_ms


def health_verdict(
    s: dict[str, float], *, bases_agree: bool, cycle_tol: float = 0.10
) -> str:
    """'ok', or what looks wrong, from the median summary of the peak launches."""
    issues = []
    if s["cycles_slow"] > (1.0 + cycle_tol) * s["cycles_rest"]:
        issues.append(
            "the slowest workgroup needs "
            f"{s['cycles_slow'] / s['cycles_rest']:.2f}x the cycles of the rest "
            "(degraded compute units?)"
        )
    if bases_agree and s["start_p99_ms"] > 0.25 * s["life_med_ms"]:
        issues.append(
            "workgroups start late "
            f"(p99 {s['start_p99_ms'] / s['life_med_ms']:.2f} of a lifetime)"
        )
    return "ok" if not issues else "; ".join(issues)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arm", default="early_refill", choices=list(ARMS))
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=65536)
    p.add_argument("--rounds", type=int, default=9, help="peak launches analysed")
    p.add_argument("--gap-ms", type=float, default=25.0)
    p.add_argument("--sustained", action="store_true", help="also 3 sustained launches")
    args = p.parse_args(argv)

    import numpy as np

    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    from .fp8_data import int_operands
    from .fp8_gemm_sweep import _as_u8_buffer, _launch_geometry, _time_function

    m, n, k = args.m, args.n, args.k
    if 25 * k >= (1 << 24):
        raise SystemExit(f"K={k} breaks the exact integer check (25*K must be < 2^24)")
    spec = arm_spec(args.arm)
    waves = spec.block_size // spec.wave_size
    grid, block = _launch_geometry(spec, (m, n, k))
    n_waves = grid[0] * grid[1] * waves
    flop = 2.0 * m * n * k

    ll = lower_kernel_to_llvm(build_universal_gemm(spec, arch=ARCH), arch=ARCH)
    ll_clk = instrument_ll(ll, waves_per_wg=waves, tile_n=spec.tile.tile_n)
    blobs = {
        "plain": build_hsaco_from_llvm_ir(ll, isa=ISA)[0],
        "clock": build_hsaco_from_llvm_ir(ll_clk, isa=ISA)[0],
    }
    name = re.search(r"^define amdgpu_kernel void @([\w.]+)\(", ll, re.MULTILINE).group(
        1
    )

    a8, b8, a32, b32 = int_operands(np, m, n, k)
    ref = _f32_from_bf16(np, _bf16_from_f32(np, a32 @ b32.T))
    del a32, b32
    rt = Runtime()
    c_bytes = m * n * 2
    stamp_bytes = n_waves * STAMPS_PER_WAVE * 8
    a_dev, b_dev = rt.alloc(a8.nbytes), rt.alloc(b8.nbytes)
    c_dev = rt.alloc(c_bytes + stamp_bytes)
    rt.memcpy_h2d(a_dev, _as_u8_buffer(a8), a8.nbytes)
    rt.memcpy_h2d(b_dev, _as_u8_buffer(b8), b8.nbytes)
    kargs = struct.pack("<QQQiii", a_dev, b_dev, c_dev, m, n, k)
    mods = {key: rt.load_module(blob) for key, blob in blobs.items()}
    fns = {key: mod.get_function(name) for key, mod in mods.items()}
    c_host = np.empty((m, n), dtype=np.uint16)
    st_host = np.empty((n_waves, STAMPS_PER_WAVE), dtype=np.uint64)

    def launch(key: str, gap_s: float, warmup: int, iters: int) -> float:
        rt.wait_stream(0)
        time.sleep(gap_s)
        ms = _time_function(
            rt, fns[key], grid, block, kargs, warmup=warmup, iters=iters
        )
        rt.wait_stream(0)
        return ms

    def read_stamps():
        rt.memcpy_d2h(_as_u8_buffer(st_host), c_dev + c_bytes, stamp_bytes)
        return st_host.copy()

    # ---- verify both builds; the stamps must not touch C ----------------------
    for key in ("plain", "clock"):
        rt.memset(c_dev, 0, c_bytes + stamp_bytes)
        launch(key, 0.0, 0, 1)
        rt.memcpy_d2h(_as_u8_buffer(c_host), c_dev, c_bytes)
        exact = np.array_equal(_f32_from_bf16(np, c_host), ref)
        print(f"[clock] verify {key:<5} int exact: {exact}")
        if not exact:
            raise SystemExit("[clock] verification failed; nothing reported")
    if not (read_stamps()[:, 2] > read_stamps()[:, 0]).all():
        raise SystemExit("[clock] some wave wrote no stamps")

    # ---- realtime counter rate, against the host clock ------------------------
    launch("clock", 0.0, 0, 1)
    s0, h0 = read_stamps(), time.perf_counter()
    time.sleep(2.0)
    launch("clock", 0.0, 0, 1)
    s1, h1 = read_stamps(), time.perf_counter()
    rt_hz = float(np.median(s1[:, 1].astype(np.int64) - s0[:, 1].astype(np.int64)))
    rt_hz /= h1 - h0
    print(f"[clock] realtime counter ~{rt_hz / 1e6:.2f} MHz (host-calibrated)")

    # ---- peak (and sustained) launches: overhead and stamps -------------------
    times: dict[str, list[float]] = {"plain": [], "clock": []}
    peak: list[dict[str, float]] = []
    for rnd in range(args.rounds):
        for key in interleaved_order(["plain", "clock"], rnd):
            times[key].append(launch(key, args.gap_ms / 1000.0, 1, 3))
            if key == "clock":
                peak.append(
                    summarize(np, read_stamps(), waves_per_wg=waves, rt_hz=rt_hz)
                )
    sus: list[dict[str, float]] = []
    if args.sustained:
        for _ in range(3):
            launch("clock", 2.0, 20, 400)
            sus.append(summarize(np, read_stamps(), waves_per_wg=waves, rt_hz=rt_hz))

    for mod in mods.values():
        mod.unload()
    for ptr in (a_dev, b_dev, c_dev):
        rt.free(ptr)

    def med(rows: list[dict[str, float]]) -> dict[str, Any]:
        return {key: statistics.median(r[key] for r in rows) for key in rows[0]}

    overhead = statistics.median(times["clock"]) / statistics.median(times["plain"])
    print(
        f"\n{args.arm} at {m}x{n}x{k}, {grid[0] * grid[1]} workgroups x {waves} "
        f"waves; plain build {flop / (statistics.median(times['plain']) * 1e9):.0f} "
        f"TFLOP/s, stamps cost {overhead:.3f}x its time"
    )
    launch_ms = statistics.median(times["clock"])
    agree = realtime_bases_agree(med(peak), launch_ms)
    if not agree:
        print(
            f"realtime bases disagree across workgroups: the stamped span "
            f"({med(peak)['span_ms']:.3f} ms) exceeds the timed launch "
            f"({launch_ms:.3f} ms), so start/end columns are not comparable "
            "between workgroups; lifetimes, cycles and clocks are per wave "
            "and unaffected"
        )
    print(
        f"{'protocol':<10} {'start p99':>9} {'start max':>9} {'end med':>8} "
        f"{'end max':>8} {'life med':>8} {'clk MHz':>8} "
        f"{'Mcyc slow':>9} {'Mcyc rest':>9}  (ms from the first wave's start)"
    )
    for label, rows in (("peak", peak), ("sustained", sus)):
        if not rows:
            continue
        s = med(rows)
        print(
            f"{label:<10} {s['start_p99_ms']:9.3f} {s['start_max_ms']:9.3f} "
            f"{s['end_med_ms']:8.3f} {s['end_max_ms']:8.3f} {s['life_med_ms']:8.3f} "
            f"{s['clock_med_mhz']:8.0f} {s['cycles_slow'] / 1e6:9.3f} "
            f"{s['cycles_rest'] / 1e6:9.3f}"
        )
    print(f"\nhealth: {health_verdict(med(peak), bases_agree=agree)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
