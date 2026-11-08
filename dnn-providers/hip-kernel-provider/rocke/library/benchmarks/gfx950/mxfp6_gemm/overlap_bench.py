# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify-then-time harness for overlapping the per-call A pack with the gfx950 MXFP6 GEMM (multi-stream).

``python -m benchmarks.gfx950.mxfp6_gemm.overlap_bench run --arms <json or file> [...]``

Arm kinds (``"kind"``):

* ``gemm``     -- one GEMM launch on pre-packed operands (``"gemm"``: a ``gemm.PRESETS`` name, ``"over"``: overrides);
  ``"a_layout": "canonical"`` presets (best_mixed) read the canonical A codes with pre-packed tile scales;
* ``serial``   -- pack(A) then the GEMM on one stream (``"pack"``: pack config, see :func:`pack_launches`);
* ``mixed_as`` -- the A scale repack (raw -> tile) then a canonical-A GEMM, one stream (true live-activation mixed);
* ``chunked``  -- :func:`overlap.schedule_chunked` (``"chunks"``, ``"xcd"``, ``"mode"``, ``"gemm_only"``);
* ``nextcall`` -- :func:`overlap.schedule_next_call` (serving scenario: the next call's pack during this GEMM);
* ``pack_only``-- the pack alone (timing only).

``"prio"``: ``{"P": "hi"}`` etc. maps a schedule stream to a high-priority HIP stream. ``"mutant"``: a schedule
mutant (``nowait`` / ``nowar``), ``"pack_mutant"`` / ``"gemm_mutant"``: kernel mutants -- such arms must FAIL.

Verification (every arm, every dataset, before any timing): the pack / image buffers are poisoned (0xAB), then
four logical calls run with ALTERNATING inputs (A, A, A rotated by 256 rows, A rotated) into four C buffers; every C must be
bitwise equal to the pre-packed GEMM's C for that input (the base GEMM is checked against the float64 reference:
exact on the integer set, scaled-MFMA tolerance on Gaussian / uniform), and calls with the same input identical
(determinism). A schedule without its event waits fails this check (stale or poisoned data).

Timing: shuffled rounds; cold-start (post-idle: 25 ms idle, 1 warm-up, 3 timed calls) and warm (steady-state:
1.5 s idle, 20 warm-up calls, ~``--warm-s`` s of back-to-back calls with sysfs SCLK / board power -> mJ per call).
Every call ends with all its streams joined into the first stream (the timed interval covers every launch).
Paired-bootstrap 95 % CIs of t_ref / t_arm per round against the ``--refs`` arms. One traced call per multi-stream
arm records event times around every launch (evidence of overlap). Absolute numbers stay in the internal record.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.gfx950.mxfp6_gemm import bench as B
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import overlap as O
from kernels.gfx950.mxfp6_gemm import pack as P
from kernels.gfx950.mxfp6_gemm.gemm import PRESET_LEVERS, PRESETS, grid_of, operand_layout_of

ROT = 256  # rows by which the alternate input is rotated (one CTA tile: C rotates exactly)
NCALLS = 4  # logical calls per verification run (inputs A, A, A_rot, A_rot into 4 C buffers)


def variant_of(call: int) -> int:
    """Input of logical call ``call`` in verification: A, A, A_rot, A_rot -- two consecutive calls share an input
    and a buffer slot of a double-buffered schedule sees both inputs, so a pack that overwrites a buffer an earlier
    call still reads (a missing WAR wait) changes that call's C."""
    return (call // 2) % 2


# ----------------------------------------------------------------------------------------------- arm parsing


def gemm_over(arm: dict, m: int, n: int, k: int) -> dict:
    name = arm.get("gemm", "best_v6")
    over = {**PRESETS[name], "m": m, "n": n, "k": k, **arm.get("over", {})}
    if arm.get("gemm_mutant"):
        over["mutant"] = arm["gemm_mutant"]
    return over


def gemm_levers(arm: dict) -> str:
    return arm.get("levers") or PRESET_LEVERS.get(arm.get("gemm", "best_v6"), B.RUST_LEVERS)


def gemm_builds(arm: dict, m: int, n: int, k: int) -> dict:
    """{chunk index or "all": (build key, spec overrides)} of an arm's GEMM launches."""
    base = gemm_over(arm, m, n, k)
    out = {}
    if arm["kind"] == "chunked":
        for ch in O.chunk_plan(m, arm["chunks"], xcd_block=arm.get("xcd")):
            over = O.chunk_spec_overrides(base, ch)
            out[ch.index] = (_key(over, gemm_levers(arm)), over)
    elif arm["kind"] == "kflag":
        kf = {"flag_tiles": arm.get("flag_tiles", 16), "mutant": arm.get("kflag_mutant", "none"),
              "done_signal": arm.get("dep", "event") in ("value", "spin")}
        out["all"] = (_key({**base, "_kflag": kf}, gemm_levers(arm)), {**base, "_kflag": kf})
    elif arm["kind"] != "pack_only":
        out["all"] = (_key(base, gemm_levers(arm)), base)
    return out


def _key(over: dict, levers: str) -> str:
    h = hashlib.sha1(json.dumps({**over, "_levers": levers}, sort_keys=True).encode()).hexdigest()[:10]
    return f"g{over['m']}_{over.get('xcd_block', '4x8')}_{h}"


def pack_cfg(arm: dict) -> dict:
    cfg = {"impl": "new", **arm.get("pack", {}), **({"mutant": arm["pack_mutant"]} if arm.get("pack_mutant") else {})}
    if arm["kind"] == "kflag":  # the flag-mode co-resident pack
        cfg = {**cfg, "impl": "lite", "flag_tiles": arm.get("flag_tiles", 16)}
        if arm.get("dep") == "spin":  # the WAR order lives in the pack kernel (the nowar mutant drops it)
            cfg["wait_prev"] = arm.get("mutant") != "nowar"
    return cfg


def _pack_key(cfg: dict, rows: int) -> str:
    return "p" + hashlib.sha1(json.dumps({**cfg, "rows": rows}, sort_keys=True).encode()).hexdigest()[:10]


def build_pack_kernels(cfg: dict, rows: int, k: int):
    """[(kernel, grid, signature)] of one pack config ("new" | "lite" | "legacy"); signature "4" = (codes_src,
    scales_src, codes_dst, scales_dst), "codes" / "scales" = (src, dst)."""
    kw = {x: v for x, v in cfg.items() if x != "impl"}
    impl = cfg["impl"]
    if impl == "new":
        wr, wt = kw.get("wg_rows", 128), kw.get("wg_tiles", 4)
        return [(P.build_pack_operand(rows, k, **kw), P.pack_operand_grid(rows, k, wr, wt), "4", 256)]
    if impl == "lite":  # (kernel, grid, signature, block)
        return [(O.build_pack_lite(rows, k, **kw), O.pack_lite_grid(kw.get("n_wg", 256)),
                 "flag" if kw.get("flag_tiles") else "4", O.pack_lite_block(kw.get("stage", "none")))]
    if impl == "legacy":
        g = P.pack_grids(rows, k)
        return [(P.build_pack_c0c1(rows, k), g[0], "codes", 256), (P.build_pack_scales_tile(rows, k), g[1], "scales",
                                                                    256)]
    raise ValueError(f"pack impl {impl}")


def pack_rows(arm: dict, m: int) -> list:
    if arm["kind"] == "chunked":
        return [] if arm.get("gemm_only") else [m // arm["chunks"]]
    if arm["kind"] == "kflag" and arm.get("mode") == "gemmonly":
        return []
    if arm["kind"] in ("serial", "nextcall", "pack_only", "kflag"):
        return [m]
    return []


def is_mutant(arm: dict) -> bool:
    """A must-fail arm (schedule, pack, GEMM, chunk or K-flag mutant)."""
    return any(arm.get(x) and arm.get(x) != "none" for x in ("mutant", "pack_mutant", "gemm_mutant", "chunk_mutant",
                                                            "kflag_mutant"))


def schedule_of(arm: dict) -> O.Schedule:
    kind = arm["kind"]
    mut = arm.get("mutant", "none")
    if kind == "chunked":
        return O.schedule_chunked(arm["chunks"], mode=arm.get("mode", "strict"), gemm_only=bool(arm.get("gemm_only")),
                                  mutant=mut)
    if kind == "nextcall":
        return O.schedule_next_call(gemm_first=arm.get("gemm_first", True), mutant=mut)
    if kind in ("serial", "mixed_as"):
        return O.schedule_serial()
    if kind == "kflag":
        return O.schedule_kflag(mode=arm.get("mode", "strict"), dep=arm.get("dep", "event"), mutant=mut)
    if kind == "pack_only":
        return O.Schedule([], [O.StreamOp("launch", "G", "pack", ("all",))], ("G",), "pack_only")
    return O.Schedule([], [O.StreamOp("launch", "G", "gemm", ("all",))], ("G",), "gemm")


# ----------------------------------------------------------------------------------------------- HIP extras


def _hip():
    from rocke.runtime.hip_module import _HipEventHandle, _resolve_hip

    h = _resolve_hip()
    vp = ctypes.c_void_p
    for name, args in (("hipStreamWaitEvent", [vp, _HipEventHandle, ctypes.c_uint]),
                       ("hipStreamCreateWithPriority", [ctypes.POINTER(vp), ctypes.c_uint, ctypes.c_int]),
                       ("hipDeviceGetStreamPriorityRange", [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]),
                       ("hipStreamQuery", [vp]),
                       ("hipEventCreateWithFlags", [ctypes.POINTER(_HipEventHandle), ctypes.c_uint]),
                       ("hipExtMallocWithFlags", [ctypes.POINTER(vp), ctypes.c_size_t, ctypes.c_uint]),
                       ("hipStreamWaitValue32", [vp, vp, ctypes.c_uint32, ctypes.c_uint, ctypes.c_uint32]),
                       ("hipExtStreamCreateWithCUMask", [ctypes.POINTER(vp), ctypes.c_uint32,
                                                         ctypes.POINTER(ctypes.c_uint32)])):
        f = getattr(h, name)
        f.argtypes, f.restype = args, ctypes.c_int
    return h


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: hipError {rc}")


class Streams:
    """Named non-blocking HIP streams, created once (stream name + optional "^hi" priority suffix)."""

    def __init__(self, hip):
        self.hip = hip
        lo, hi = ctypes.c_int(), ctypes.c_int()
        _ck(hip.hipDeviceGetStreamPriorityRange(ctypes.byref(lo), ctypes.byref(hi)), "priority range")
        self.prio = {"lo": lo.value, "hi": hi.value}
        self.s = {}

    def get(self, name: str) -> int:
        """``base`` | ``base^hi`` (highest priority) | ``base^cmN`` (CU mask: the lowest N bits of the device's CU
        mask, ``hipExtStreamCreateWithCUMask``)."""
        if name not in self.s:
            base, _, pr = name.partition("^")
            h = ctypes.c_void_p()
            if pr.startswith("cm"):
                ncu = int(pr[2:])
                words = (ctypes.c_uint32 * 8)(*[(0xFFFFFFFF if 32 * (i + 1) <= ncu else
                                                 (1 << max(0, ncu - 32 * i)) - 1) & 0xFFFFFFFF for i in range(8)])
                _ck(self.hip.hipExtStreamCreateWithCUMask(ctypes.byref(h), 8, words), f"stream {name}")
            else:
                _ck(self.hip.hipStreamCreateWithPriority(ctypes.byref(h), 1, self.prio["hi" if pr == "hi" else "lo"]),
                    f"stream {name}")
            self.s[name] = int(h.value)
        return self.s[name]


# ----------------------------------------------------------------------------------------------- bootstrap


def paired_ci(ref: list, arm: list, n_boot: int = 4000, seed: int = 7):
    """Median of the per-round ratios ref / arm (> 1: the arm is faster) and its bootstrap 95 % CI."""
    r = np.asarray(ref, float) / np.asarray(arm, float)
    if r.size == 0:
        return None
    rng = np.random.default_rng(seed)
    meds = np.median(r[rng.integers(0, r.size, (n_boot, r.size))], axis=1)
    return [float(np.median(r)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


# ----------------------------------------------------------------------------------------------- run


def cmd_run(args) -> int:
    m, n, k = args.m, args.n, args.k
    cache, work = Path(args.cache), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    arms = json.loads(Path(args.arms).read_text()) if os.path.exists(args.arms) else json.loads(args.arms)
    for nm, a in arms.items():
        a.setdefault("kind", "gemm")
        if nm.endswith("~tw") and "twin_of" in a:
            arms[nm] = {**arms[a["twin_of"]], "twin_of": a["twin_of"]}
    for nm, a in arms.items():
        a["_name"] = nm
    for nm, a in arms.items():
        bad = O.check_schedule(schedule_of(a)) if a["kind"] in ("chunked", "nextcall", "kflag") else []
        a["_hazards"] = bad
        if bad and not is_mutant(a):
            raise SystemExit(f"arm {nm}: schedule hazards {bad[:3]}")
        if a.get("dep") == "value" and (a.get("kflag_mutant", "none") != "none" or a.get("pack_mutant")):
            # a GEMM that does not wait for its flags can finish (and overwrite the parity word) before the pack
            # stream checks the previous parity: the CP wait would never end. The value dependency relies on the
            # flag protocol; kernel mutants are run with the event dependency.
            raise SystemExit(f"arm {nm}: kernel mutants need dep=event (a value wait could block forever)")
    report = {"shape": [m, n, k], "arms": {nm: {"cfg": {x: v for x, v in a.items() if not x.startswith("_")},
                                                  "hazards": len(a["_hazards"])} for nm, a in arms.items()},
              "env": {x: os.environ.get(x) for x in ("SLURM_JOB_ID", "SLURMD_NODENAME", "ROCKE_RUST_COMPILE",
                                                     "GPU_MAX_HW_QUEUES", "HIP_VISIBLE_DEVICES")}}
    t_start = time.time()
    # ---------------- builds (before GPU init): GEMMs through the bench's compile subprocesses, packs in-process
    gb, gk = {}, {}
    for a in arms.values():
        for key, over in gemm_builds(a, m, n, k).values():
            if "_kflag" in over:
                gk[key] = (over, gemm_levers(a))
            else:
                gb[key] = {**over, "_compiler": "rust", "_levers": gemm_levers(a)}
    built = B.compile_arms(gb, work / "gemm", args.jobs, "rust")
    built.update(compile_kflag(gk, work / "gemm_kf", args.jobs))
    for key, meta in built.items():
        if not meta.get("ok"):
            raise SystemExit(f"GEMM build {key} failed: {meta.get('error')}")
    report["gemm_builds"] = {key: {x: built[key].get(x) for x in ("name", "notes", "loop", "rust_levers", "ir_sha256")}
                             | {"spec": {x: v for x, v in (gb.get(key) or gk[key][0]).items()
                                         if x in ("m", "xcd_block", "mutant", "_kflag")}}
                             for key in built}
    from rocke.helpers.compile import compile_kernel

    pk = {}
    for a in arms.values():
        if a["kind"] == "mixed_as":
            key = "scA"
            if key not in pk:
                kf = P.build_pack_scales(m, k)
                pk[key] = [(compile_kernel(kf, arch="gfx950", backend="python").hsaco, kf.name,
                            P.pack_scales_grid(m, k), "scA", 256)]
        for rows in pack_rows(a, m):
            cfg = pack_cfg(a)
            key = _pack_key(cfg, rows)
            if key not in pk:
                pk[key] = []
                for kf, grid, sig, blk in build_pack_kernels(cfg, rows, k):
                    pk[key].append((compile_kernel(kf, arch="gfx950", backend="python").hsaco, kf.name, grid, sig,
                                    blk))
    report["pack_builds"] = {key: [{"name": x[1], "grid": x[2], "block": x[4], "notes": B.code_object_notes(x[0])}
                                   for x in v]
                             for key, v in pk.items()}
    for key, v in report["pack_builds"].items():
        print(f"[build] pack {key}: {v}", flush=True)
    for key, v in report["gemm_builds"].items():
        print(f"[build] gemm {key}: {v['name']} {v['notes']} {v['spec']}", flush=True)
    print(f"[compile] {len(built)} GEMMs + {len(pk)} pack configs in {time.time() - t_start:.0f} s", flush=True)

    # ---------------- data (host)
    dists = [d for d in args.dists.replace(":", ",").split(",") if d]
    data = {}
    for d in dists:
        a_op, b_op = B.cached_operands(cache, m, n, k, d, args.seed, args.gen_workers)
        ref = B.cached_reference(cache, a_op, b_op, f"{d}_{m}x{n}x{k}_s{args.seed}")
        data[d] = (a_op, b_op, ref)
    print(f"[data] {dists} ready ({time.time() - t_start:.0f} s)", flush=True)

    # ---------------- GPU
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    hip = _hip()
    bdf = B.device_bdf()
    report["bdf"] = bdf
    print(f"[gpu] bdf {bdf}", flush=True)
    clk = B.Clock(bdf)
    streams = Streams(hip)
    report["stream_priority_range"] = streams.prio

    def put(arr):
        arr = np.ascontiguousarray(arr)
        p_ = rt.alloc(arr.nbytes)
        rt.memcpy_h2d(p_, B._u8(arr), arr.nbytes)
        return p_

    img_bytes, sc_bytes = m * k * 3 // 4, m * k // 32
    ws = {"img": [rt.alloc(img_bytes) for _ in range(2)], "sc": [rt.alloc(sc_bytes) for _ in range(2)],
          "C": [rt.alloc(m * n * 2) for _ in range(NCALLS)]}
    dev = {}
    for d in dists:
        a_op, b_op, _ = data[d]
        a_rot = L.Operand(m, k, np.roll(a_op.codes, -ROT, axis=0), np.roll(a_op.scales, -ROT, axis=0), a_op.fmt)
        dev[d] = {"can": [put(a_op.codes), put(a_rot.codes)], "raw": [put(a_op.scales), put(a_rot.scales)],
                  "pre_img": [put(L.pack_operand(a_op, "c0c1", 256)), put(L.pack_operand(a_rot, "c0c1", 256))],
                  "pre_sc": [put(L.pack_scales_tile(a_op.scales, "16x16x128")),
                             put(L.pack_scales_tile(a_rot.scales, "16x16x128"))],
                  "b_img": put(L.pack_operand(b_op, "c0c1", 256)),
                  "b_sc": put(L.pack_scales_tile(b_op.scales, "16x16x128"))}
    # K-flag arms: per-slot ready flags + arrival counters in uncached memory (fresh to scalar glc loads on every
    # XCD); deadlock guard: one pack wave must fit next to a resident GEMM CTA on every SIMD (registers, LDS)
    kfb = {}
    for nm, a in arms.items():
        if a["kind"] != "kflag" or nm.endswith("~tw") and a.get("twin_of") in kfb:
            continue
        gkey = gemm_builds(a, m, n, k)["all"][0]
        gnotes = built[gkey].get("notes") or {}
        pnotes = (B.code_object_notes(pk[_pack_key(pack_cfg(a), m)][0][0]) if _pack_key(pack_cfg(a), m) in pk
                  else {"vgpr_count": 0})
        regs = -(-gnotes.get("vgpr_count", 512) // 8) * 8 + -(-pnotes.get("vgpr_count", 512) // 8) * 8
        lds = gnotes.get("group_segment_fixed_size", 0) + pnotes.get("group_segment_fixed_size", 0)
        fits = regs <= 512 and lds <= 160 * 1024
        report["arms"][nm]["coresidency"] = {"vgprs": regs, "lds": lds, "fits": fits}
        if not fits:
            raise SystemExit(f"arm {nm}: pack wave does not fit next to the GEMM CTA ({regs} VGPRs, {lds} B LDS)")
        nsl = schedule_of(a).meta["slots"]
        nq = (k // 128) // a.get("flag_tiles", 16)
        bufs = {}
        for sl in range(nsl):
            for what in ("cnt", "flg"):
                h = ctypes.c_void_p()
                _ck(hip.hipExtMallocWithFlags(ctypes.byref(h), 4 * nq, 3), "hipExtMallocWithFlags uncached")
                bufs[(what, sl)] = int(h.value)
        # GEMM done counter + done word (dep value: HSA signal memory a CP wait can poll; else uncached device
        # memory the pack polls) + hang-guard timeout counters of the GEMM and the pack
        for what, nb, flag in (("dcnt", 64, 3), ("dsig", 8, 2 if a.get("dep") == "value" else 3), ("gtmo", 64, 3),
                               ("ptmo", 64, 3)):
            h = ctypes.c_void_p()
            _ck(hip.hipExtMallocWithFlags(ctypes.byref(h), nb, flag), f"hipExtMallocWithFlags {what}")
            bufs[(what, 0)] = int(h.value)
        kfb[nm] = (bufs, 4 * nq)
    for nm, a in arms.items():
        if nm.endswith("~tw") and a.get("twin_of") in kfb:
            kfb[nm] = kfb[a["twin_of"]]

    def reset_kf(nm):
        if nm in kfb:
            rt.sync()
            for (what, _), p_ in kfb[nm][0].items():
                if what == "dsig":  # the value call 0's pack waits for (parity of call -1)
                    v = np.full(2, O.kflag_parity(-1, 1), np.int32)
                    if arms[nm].get("dep") == "value":  # host-mapped HSA signal memory
                        ctypes.c_uint32.from_address(p_).value = int(v[0])
                    else:
                        rt.memcpy_h2d(p_, B._u8(v), 8)
                    continue
                if what in ("dcnt", "gtmo", "ptmo"):
                    rt.memset(p_, 0, 64)
                    continue
                if what == "flg" and schedule_of(arms[nm]).meta.get("flags_preset"):
                    ones = np.ones(kfb[nm][1] // 4, np.int32)
                    rt.memcpy_h2d(p_, B._u8(ones), ones.nbytes)
                else:
                    rt.memset(p_, 0, kfb[nm][1])
            rt.sync()

    def kf_timeouts(nm):
        """(GEMM, pack) hang-guard counts of a K-flag arm since its last reset (must stay 0)."""
        if nm not in kfb:
            return (0, 0)
        rt.sync()
        out = []
        for what in ("gtmo", "ptmo"):
            v = np.zeros(1, np.int32)
            rt.memcpy_d2h(B._u8(v), kfb[nm][0][(what, 0)], 4)
            out.append(int(v[0]))
        return tuple(out)

    mods = {}
    gfn = {}
    for key, meta in built.items():
        mods[key] = rt.load_module(Path(meta["hsaco"]).read_bytes())
        gfn[key] = mods[key].get_function(meta["name"])
    pfn = {}
    for key, lst in pk.items():
        pfn[key] = []
        for hs, name, grid, sig, blk in lst:
            mod = rt.load_module(hs)
            mods[f"{key}:{name}"] = mod
            pfn[key].append((mod.get_function(name), grid, sig, blk))
    from kernels.gfx950.mxfp6_gemm.gemm import Mxfp6GemmSpec
    from dataclasses import replace

    def spec_of(over):
        return replace(Mxfp6GemmSpec(), **over)

    def expand(arm: dict, ncalls: int):
        """Concrete ops of setup + ``ncalls`` logical calls: (kind, stream, payload, call index, variant)."""
        s = schedule_of(arm)
        per = s.meta.get("calls_per_list", 1)
        ops = [(op, 0) for op in s.setup]
        for j in range((ncalls + per - 1) // per):
            ops += [(op, j * per) for op in s.call]
        return s, ops

    def launches_for(arm: dict, op, call: int, d: str, variant_of_call, cbuf_of_call):
        """[(fn, grid, block, kargs)] of one schedule launch op."""
        dv = dev[d]
        kind = arm["kind"]
        if kind == "kflag":  # (slot, call offset); the pack and the GEMM of a call share its flag parity
            slot = op.arg[0]
            bufs = kfb[arm["_name"]][0]
            par = O.kflag_parity(call, schedule_of(arm).meta["slots"])
            if op.what == "pack":
                var = variant_of_call(call)
                (fn, grid, sig, blk), = pfn[_pack_key(pack_cfg(arm), m)]
                return [(fn, (grid, 1, 1), (blk, 1, 1),
                         O.pack_lite_flag_args(dv["can"][var], dv["raw"][var], ws["img"][slot], ws["sc"][slot],
                                               bufs[("cnt", slot)], bufs[("flg", slot)], par, bufs[("dsig", 0)],
                                               bufs[("ptmo", 0)]))]
            key, over = gemm_builds(arm, m, n, k)["all"]
            grid, block = grid_of(spec_of({x: v for x, v in over.items() if not x.startswith("_")}))
            a_ptr, sa_ptr = ws["img"][slot], ws["sc"][slot]
            if arm.get("mode") == "gemmonly":  # flags preset to 1: the polls run, never wait
                var = variant_of_call(call)
                a_ptr, sa_ptr, par = dv["pre_img"][var], dv["pre_sc"][var], 1
            return [(gfn[key], grid, block,
                     O.gemm_kflag_args(a_ptr, dv["b_img"], sa_ptr, dv["b_sc"],
                                       ws["C"][cbuf_of_call(call)], bufs[("flg", slot)], par,
                                       bufs[("dcnt", 0)], bufs[("dsig", 0)], bufs[("gtmo", 0)]))]
        if op.what == "pack":
            if kind == "nextcall":  # buffer slot b; the pack prepares logical call ``call`` (see build_program)
                slot = op.arg[0]
                var = variant_of_call(call)
                rows, row0, sc_off, img = m, 0, 0, ws["img"][slot]
                scb = ws["sc"][slot]
            elif kind == "chunked":
                c = op.arg[0]
                rows = m // arm["chunks"]
                row0, sc_off = c * rows, c * rows * k // 32
                var = variant_of_call(call)
                img, scb = ws["img"][0], ws["sc"][0]
            else:
                rows, row0, sc_off, var = m, 0, 0, variant_of_call(call)
                img, scb = ws["img"][0], ws["sc"][0]
            codes = dv["can"][var] + row0 * k * 3 // 4
            raw = dv["raw"][var] + row0 * k // 32
            out = []
            for fn, grid, sig, blk in pfn[_pack_key(pack_cfg(arm), rows)]:
                if sig == "4":
                    ka = struct.pack("<QQQQ", codes, raw, img + row0 * k * 3 // 4, scb + sc_off)
                elif sig == "codes":
                    ka = struct.pack("<QQ", codes, img + row0 * k * 3 // 4)
                else:
                    ka = struct.pack("<QQ", raw, scb + sc_off)
                out.append((fn, (grid, 1, 1), (blk, 1, 1), ka))
            return out
        # gemm
        builds = gemm_builds(arm, m, n, k)
        cb = ws["C"][cbuf_of_call(call)]
        if kind == "chunked":
            c = op.arg[0]
            key, over = builds[c]
            rows = m // arm["chunks"]
            row0 = c * rows
            var = variant_of_call(call)
            if arm.get("gemm_only"):
                a_ptr, sa_ptr = dv["pre_img"][var] + row0 * k * 3 // 4, None
                # pre-packed tile scales of the chunk rows: the full-M tile layout is not row-separable
                sa_ptr = chunk_sc[(d, var, arm["chunks"], c)]
            else:
                a_ptr, sa_ptr = ws["img"][0] + row0 * k * 3 // 4, ws["sc"][0] + c * rows * k // 32
            if arm.get("chunk_mutant") == "offset":
                a_ptr = ws["img"][0] + ((c + 1) % arm["chunks"]) * rows * k * 3 // 4
            c_ptr = cb + row0 * n * 2
        else:
            key, over = builds["all"]
            if kind == "nextcall":
                slot = op.arg[0]
                a_ptr, sa_ptr = ws["img"][slot], ws["sc"][slot]
            elif kind in ("serial",):
                a_ptr, sa_ptr = ws["img"][0], ws["sc"][0]
            elif kind == "mixed_as":
                a_ptr, sa_ptr = dv["can"][variant_of_call(call)], ws["sc"][0]
            else:  # pre-packed
                var = variant_of_call(call)
                if operand_layout_of(spec_of(over), "A") == "canonical":
                    a_ptr = dv["can"][var]
                else:
                    a_ptr = dv["pre_img"][var]
                sa_ptr = dv["pre_sc"][var]
            c_ptr = cb
        sp = spec_of(over)
        grid, block = grid_of(sp)
        return [(gfn[key], grid, block, struct.pack("<QQQQQ", a_ptr, dv["b_img"], sa_ptr, dv["b_sc"], c_ptr))]

    # chunk tile scales of the pre-packed operands (gemm_only chunk arms)
    chunk_sc = {}
    for a in arms.values():
        if a["kind"] == "chunked" and a.get("gemm_only"):
            rows = m // a["chunks"]
            for d in dists:
                a_op = data[d][0]
                for var in (0, 1):
                    sc_all = np.roll(a_op.scales, -ROT * var, axis=0)
                    for c in range(a["chunks"]):
                        if (d, var, a["chunks"], c) not in chunk_sc:
                            chunk_sc[(d, var, a["chunks"], c)] = put(
                                L.pack_scales_tile(sc_all[c * rows:(c + 1) * rows], "16x16x128"))
    for a in arms.values():  # mixed_as: the per-call scale repack writes ws sc[0]
        a["_scA"] = pfn.get("scA")

    def build_program(arm: dict, ncalls: int, d: str, variant_of_call, cbuf_of_call, trace: bool = False):
        """Prepared callables of setup + calls; returns (setup list, list of per-call lists, main stream)."""
        s, ops = expand(arm, ncalls)
        prio = arm.get("prio", {})

        def sname(x):  # "prio": {"P": "hi"} or {"P": "cm128"} (CU mask)
            v = prio.get(x.rstrip("0123456789")) or prio.get(x)
            return x + (f"^{v}" if v else "")

        main = streams.get(sname(s.streams[0]))
        evs = {}

        def ev(name):
            if name not in evs:
                if arm.get("evflags"):  # e.g. 0x2 disable timing, 0x20000000 no system-scope fence
                    from rocke.runtime.hip_module import Event, _HipEventHandle

                    h = _HipEventHandle()
                    _ck(hip.hipEventCreateWithFlags(ctypes.byref(h), int(arm["evflags"])), "event flags")
                    evs[name] = Event(h)
                else:
                    evs[name] = rt.event()
            return evs[name]

        prog_setup, prog_calls = [], {}
        trace_marks = []
        n_setup = len(s.setup)
        for idx, (op, call0) in enumerate(ops):
            st = streams.get(sname(op.stream))
            if op.kind == "record":
                f = (lambda e=ev(op.what), st=st: e.record(stream=st))
            elif op.kind == "wait":
                f = (lambda e=ev(op.what), st=st: _ck(hip.hipStreamWaitEvent(st, e.handle, 0), "wait"))
            elif op.kind in ("signal", "kwait"):  # written by the GEMM / waited for by the pack kernel itself
                continue
            elif op.kind == "waitvalue":  # CP wait: the previous call's GEMM published its parity
                if idx < len(s.setup):
                    continue
                sig = kfb[arm["_name"]][0][("dsig", 0)]
                prev = O.kflag_parity(call0 - 1, s.meta["slots"])
                f = (lambda st=st, sig=sig, v=prev: _ck(hip.hipStreamWaitValue32(st, sig, v, 1, 0xFFFFFFFF),
                                                        "hipStreamWaitValue32"))
            else:
                call = call0
                if s.meta.get("calls_per_list", 1) > 1:
                    call = call0 + op.arg[1]  # nextcall: (slot, logical call offset in the two-call list)
                if arm["kind"] == "mixed_as" and op.what == "pack":
                    fn, grid, sig, _ = arm["_scA"][0]
                    dv = dev[d]
                    ls = [(fn, (grid, 1, 1), (256, 1, 1),
                           struct.pack("<QQQQ", dv["raw"][variant_of_call(call)], 0, ws["sc"][0], 0))]
                else:
                    ls = launches_for(arm, op, call, d, variant_of_call, cbuf_of_call)
                fs = [rt.prepare_launch(fn_, g_, b_, ka_, stream=st) for fn_, g_, b_, ka_ in ls]
                if trace:
                    e_a, e_b = rt.event(), rt.event()
                    trace_marks.append((f"{op.stream}:{op.what}{op.arg}", e_a, e_b))
                    f = (lambda fs=fs, st=st, e_a=e_a, e_b=e_b: (e_a.record(stream=st), [x() for x in fs],
                                                                e_b.record(stream=st)))
                else:
                    f = (lambda fs=fs: [x() for x in fs])
            if idx < n_setup:
                prog_setup.append(f)
            else:
                prog_calls.setdefault(call0, []).append(f)
        others = [streams.get(sname(x)) for x in s.streams[1:]]
        lists = [prog_calls[c] for c in sorted(prog_calls)]
        return prog_setup, lists, main, others, trace_marks, len(lists) * s.meta.get("calls_per_list", 1)

    def run_calls(prog, main, others, e_start=None):
        """Enqueue program calls with the other streams fenced to start after e_start and joined at the end."""
        if e_start is not None:
            for o in others:
                _ck(hip.hipStreamWaitEvent(o, e_start.handle, 0), "wait start")
        for call in prog:
            for f in call:
                f()

    def join(main, others):
        for o in others:
            e = rt.event()
            e.record(stream=o)
            _ck(hip.hipStreamWaitEvent(main, e.handle, 0), "join")
            e.destroy()

    # ---------------- verify
    c_host = np.empty((m, n), dtype=np.uint16)

    def get_c(i):
        rt.memcpy_d2h(B._u8(c_host), ws["C"][i], c_host.nbytes)
        return c_host.copy()

    verdict = {}
    base_c = {}
    base_arm = args.base
    order = [base_arm] + [x for x in arms if x != base_arm]
    for nm in order:
        a = arms[nm]
        if nm.endswith("~tw"):
            continue
        if a["kind"] == "pack_only":
            verdict[nm] = True
            report["arms"][nm]["verify"] = "timing only (pack alone)"
            continue
        must_fail = is_mutant(a)
        res = {}
        for d in dists:
            for p_ in ws["img"]:
                rt.memset(p_, 0xAB, img_bytes)
            for p_ in ws["sc"]:
                rt.memset(p_, 0xAB, sc_bytes)
            for p_ in ws["C"]:
                rt.memset(p_, 0xFF, m * n * 2)
            rt.sync()
            reset_kf(nm)
            ncalls = NCALLS
            pset, pcalls, main, others, _, _ = build_program(a, ncalls, d, variant_of, lambda c: c)
            run_calls([pset], main, others)
            run_calls(pcalls, main, others)
            join(main, others)
            rt.sync()
            tmo = kf_timeouts(nm)
            outs = [get_c(i) for i in range(ncalls)]
            if nm == base_arm:
                base_c[d] = [outs[0], outs[2]]
                ref_c, ref_s = data[d][2]
                chk = L.check_against_reference(outs[0], ref_c, ref_s, exact=(d == "int"))
                rot_ok = np.array_equal(np.roll(outs[0], -ROT, axis=0), outs[2])
                det = np.array_equal(outs[0], outs[1]) and np.array_equal(outs[2], outs[3])
                ok = chk["pass"] and rot_ok and det
                res[d] = {**chk, "rot_consistent": bool(rot_ok), "deterministic": bool(det),
                          "sha1": hashlib.sha1(outs[0].tobytes()).hexdigest()[:16], "pass_all": bool(ok)}
            else:
                want = [base_c[d][variant_of(i)] for i in range(ncalls)]
                diff = [int((o != w).sum()) for o, w in zip(outs, want)]
                ref_c, ref_s = data[d][2]
                chk = L.check_against_reference(outs[0], ref_c, ref_s, exact=(d == "int"))
                ok = all(x == 0 for x in diff) and chk["pass"] and not any(tmo)
                res[d] = {"bad_vs_base_per_call": diff, "ref_check_pass": chk["pass"], "bad": chk["bad"],
                          "worst_err_over_tol": chk.get("worst_err_over_tol"), "timeouts": tmo, "pass_all": bool(ok)}
        ok_all = all(r["pass_all"] for r in res.values())
        verdict[nm] = (not ok_all) if must_fail else ok_all
        report["arms"][nm]["verify"] = res
        report["arms"][nm]["must_fail"] = must_fail
        print(f"[verify] {nm}: {'PASS' if verdict[nm] else 'FAIL'}{' (must-fail mutant)' if must_fail else ''} "
              + " ".join(f"{d}:{'ok' if r['pass_all'] else 'BAD'}" + (f"{r.get('bad_vs_base_per_call')}"
                                                                     if 'bad_vs_base_per_call' in r else
                                                                     f"(bad={r['bad']},{r.get('worst_err_over_tol', 0):.3g})")
                         for d, r in res.items()), flush=True)
    for nm in arms:
        if nm.endswith("~tw"):
            verdict[nm] = verdict.get(arms[nm]["twin_of"], False)
    timed = [nm for nm in arms if verdict.get(nm) and not is_mutant(arms[nm])]
    print(f"[verify] timed: {timed}", flush=True)
    failed = [nm for nm in arms if not verdict.get(nm)]
    if failed:
        print(f"[verify] FAILED (not timed / mutant passed): {failed}", flush=True)

    # ---------------- timing programs (gauss, fixed input / output)
    td = args.time_dist
    progs = {}
    for nm in timed:
        pset, pcalls, main, others, _, nlog = build_program(arms[nm], 2, td, lambda c: 0, lambda c: 0)
        progs[nm] = (pset, pcalls, main, others, nlog)
        reset_kf(nm)
        run_calls([pset], main, others)
        join(main, others)
        rt.sync()
    flop = 2.0 * m * n * k

    def sample(nm, gap_s, warmup, iters, warm):
        pset, pcalls, main, others, per = progs[nm]  # per: logical calls per pass over pcalls
        rt.sync()
        time.sleep(gap_s)
        for _ in range(max(1, warmup // per)):
            run_calls(pcalls, main, others)
        if warm:
            clk.start()
        e0, e1 = rt.event(), rt.event()
        e0.record(stream=main)
        reps = max(1, iters // per)
        for o in others:
            _ck(hip.hipStreamWaitEvent(o, e0.handle, 0), "wait e0")
        for _ in range(reps):
            run_calls(pcalls, main, others)
        join(main, others)
        e1.record(stream=main)
        e1.synchronize()
        ms = e0.elapsed_to(e1) / (reps * per)
        e0.destroy()
        e1.destroy()
        rt.sync()
        return ms, (clk.stop() if warm else None)

    rng = np.random.default_rng(args.seed)
    rounds = {nm: {"cold": [], "warm": []} for nm in timed}
    for _ in range(args.rounds):
        o_ = list(timed)
        rng.shuffle(o_)
        for nm in o_:
            ms, _ = sample(nm, args.gap_ms / 1000, 1, 3, False)
            rounds[nm]["cold"].append(ms)
    for _ in range(args.warm_rounds):
        o_ = list(timed)
        rng.shuffle(o_)
        for nm in o_:
            est = statistics.median(rounds[nm]["cold"]) if rounds[nm]["cold"] else 0.6
            iters = max(50, int(args.warm_s * 1000 / max(0.01, est)))
            ms, cs = sample(nm, 1.5, 20, iters, True)
            rounds[nm]["warm"].append({"ms": ms, **(cs or {})})
    for nm in timed:  # the hang guards must not have fired during the timed calls (the timed data is not checked)
        if nm in kfb:
            report["arms"][nm]["timing_timeouts"] = kf_timeouts(nm)
            if any(report["arms"][nm]["timing_timeouts"]):
                print(f"[time] {nm}: HANG GUARD FIRED during timing {report['arms'][nm]['timing_timeouts']}: "
                      "its numbers are invalid", flush=True)
                verdict[nm] = False
    # ---------------- traces (one call, event marks around every launch)
    for nm in timed:
        s = schedule_of(arms[nm])
        if len(s.streams) < 2:
            continue
        pset, pcalls, main, others, marks, _ = build_program(arms[nm], 2, td, lambda c: 0, lambda c: 0, trace=True)
        reset_kf(nm)
        run_calls([pset], main, others)
        for _ in range(3):  # warm
            run_calls(progs[nm][1], main, others)
        join(main, others)
        e0 = rt.event()
        e0.record(stream=main)
        run_calls(pcalls, main, others, e_start=e0)
        join(main, others)
        rt.sync()
        report["arms"][nm]["trace_ms"] = [(lab, round(e0.elapsed_to(a_), 4), round(e0.elapsed_to(b_), 4))
                                          for lab, a_, b_ in marks]
    # ---------------- summary
    print("\n[summary] internal record (warm = steady state, cold = post-idle start); ratios = t_ref / t_arm")
    for nm in timed:
        row = report["arms"][nm]
        w = rounds[nm]["warm"]
        wms = [x["ms"] for x in w]
        watts = [x["watts_med"] for x in w if x.get("watts_med")]
        sclk = [x["sclk_med"] for x in w if x.get("sclk_med")]
        row["cold_ms"] = rounds[nm]["cold"]
        row["warm"] = w
        row["warm_ms_median"] = statistics.median(wms)
        row["cold_ms_median"] = statistics.median(rounds[nm]["cold"])
        row["warm_tfs"] = flop / (row["warm_ms_median"] * 1e-3) / 1e12
        row["cold_tfs"] = flop / (row["cold_ms_median"] * 1e-3) / 1e12
        row["watts"] = statistics.median(watts) if watts else None
        row["sclk"] = statistics.median(sclk) if sclk else None
        row["mj"] = statistics.median([x["ms"] * x["watts_med"] for x in w if x.get("watts_med")]) if watts else None
        for ref in [r for r in args.refs.replace(":", ",").split(",") if r in timed]:
            row[f"warm_vs_{ref}"] = paired_ci([x["ms"] for x in rounds[ref]["warm"]], wms)
            row[f"cold_vs_{ref}"] = paired_ci(rounds[ref]["cold"], rounds[nm]["cold"])
        cis = "  ".join(f"vs {ref}: warm {row[f'warm_vs_{ref}'][0]:.4f} [{row[f'warm_vs_{ref}'][1]:.4f}, "
                        f"{row[f'warm_vs_{ref}'][2]:.4f}] cold {row[f'cold_vs_{ref}'][0]:.4f}"
                        for ref in [r for r in args.refs.replace(":", ",").split(",") if r in timed])
        print(f"  {nm:<30} warm {row['warm_tfs']:7.1f} TF/s ({row['warm_ms_median']:.4f} ms, "
              f"{row['mj'] or 0:.1f} mJ, {row['watts'] or 0:.0f} W, {row['sclk']} MHz)  cold {row['cold_tfs']:7.1f}  "
              + cis, flush=True)
    for nm in timed:
        if "trace_ms" in report["arms"][nm]:
            print(f"  trace {nm}: " + "; ".join(f"{lab} {a_:.3f}-{b_:.3f}" for lab, a_, b_ in
                                               report["arms"][nm]["trace_ms"]), flush=True)
    Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    print(f"[done] {time.time() - t_start:.0f} s; wrote {args.json}", flush=True)
    for mod in mods.values():
        mod.unload()
    return 0 if all(verdict.get(nm) for nm in arms) else 1


def cmd_compile(args) -> int:
    """Build + compile one K-flag GEMM in this fresh process (direct compiler; levers from ROCKE_RUST_LEVERS)."""
    from dataclasses import replace

    from kernels.gfx950.mxfp6_gemm.gemm import Mxfp6GemmSpec, check_compiler

    over = json.loads(args.spec)
    kf = over.pop("_kflag")
    spec = replace(Mxfp6GemmSpec(), **over)
    check_compiler(spec, "rust")
    t0 = time.time()
    kernel = O.build_gemm_kflag(spec, kf["flag_tiles"], mutant=kf.get("mutant", "none"),
                                done_signal=bool(kf.get("done_signal")))
    out = Path(args.out)
    meta = B._compile_rust(kernel, out)
    hs = out.read_bytes()
    meta.update({"ok": True, "name": kernel.name, "compile_s": round(time.time() - t0, 2),
                 "loop": B.isa_loop_stats(hs, out), "notes": B.code_object_notes(hs)})
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({x: meta.get(x) for x in ("ok", "name", "compile_s", "notes", "loop")}))
    return 0


def compile_kflag(builds: dict, work: Path, jobs: int) -> dict:
    """{key: (spec overrides incl. "_kflag", levers)} -> {key: meta}: one fresh compile process per build."""
    import subprocess

    work.mkdir(parents=True, exist_ok=True)
    procs, res, pending = {}, {}, list(builds.items())
    while pending or procs:
        while pending and len(procs) < jobs:
            key, (over, levers) = pending.pop(0)
            out = work / f"{key}.hsaco"
            spec = {x: v for x, v in over.items() if not x.startswith("_") or x == "_kflag"}
            cmd = [sys.executable, "-m", "benchmarks.gfx950.mxfp6_gemm.overlap_bench", "compile", "--spec",
                   json.dumps(spec), "--out", str(out)]
            procs[key] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                           env={**os.environ, "ROCKE_RUST_LEVERS": levers}), out)
        for key in list(procs):
            p_, out = procs[key]
            if p_.poll() is not None:
                so, se = p_.communicate()
                if p_.returncode != 0 or not out.exists():
                    res[key] = {"ok": False, "error": (se or so)[-2000:]}
                else:
                    res[key] = {**json.loads(Path(str(out) + ".json").read_text()), "hsaco": str(out)}
                del procs[key]
        time.sleep(0.05)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--m", type=int, default=4096)
    r.add_argument("--n", type=int, default=4096)
    r.add_argument("--k", type=int, default=65536)
    r.add_argument("--arms", required=True, help="JSON {name: arm} or a file")
    r.add_argument("--base", default="v6", help="pre-packed GEMM arm every other arm's C is compared with")
    r.add_argument("--refs", default="packA+v6,best_mixed,v6", help="arms the paired CIs are taken against")
    r.add_argument("--dists", default="int,gauss,uniform")
    r.add_argument("--time-dist", default="gauss")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--gap-ms", type=float, default=25.0)
    r.add_argument("--warm-rounds", type=int, default=6)
    r.add_argument("--warm-s", type=float, default=2.0)
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--gen-workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--cache", default="mxfp6_cache")
    r.add_argument("--work", default="mxfp6_work")
    r.add_argument("--json", default="overlap_bench.json")
    cc = sub.add_parser("compile")
    cc.add_argument("--spec", required=True)
    cc.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"run": cmd_run, "compile": cmd_compile}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
