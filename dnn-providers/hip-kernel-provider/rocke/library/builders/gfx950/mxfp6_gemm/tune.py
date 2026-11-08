#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""In-situ model-guided search for the gfx950 MXFP6 GEMM (one GPU, one job).

``python -m builders.gfx950.mxfp6_gemm.tune <cmd>``:

* ``space``   -- print the knob space read from ``Mxfp6GemmSpec`` / ``KNOBS``;
* ``compile`` -- build + compile one spec in this (fresh) process: code
  object, disassembly, resource notes, K-loop ISA metrics and an ISA digest;
* ``worker``  -- the GPU side (JSON lines on stdin / stdout): load modules,
  verify, time interleaved rounds, sustained windows, SCLK / power sampling;
* ``run``     -- the search supervisor (never touches the GPU itself);
* ``confirm`` -- re-time the top configs of a run in a fresh job, in blocks
  with an external command (the aiter binary) between them, plus the
  kernel's measurement probes (MFMA ceiling, fill floor);
* ``ledger``  -- rebuild the lever ledger and top-k from a run's JSONL.

The supervisor loop, per batch of trials the strategy asks:

1. legality pre-filter (:mod:`.space`: ``validate_spec``, LDS, register
   estimate) -- the strategies never see an illegal config;
2. compile every new config in its own fresh process (parallel; the
   supervisor holds no GPU context, so the pool is safe at any time, and the
   job pins ``ROCKE_COMGR_LIB`` to torch's comgr), then the compile gate
   (:func:`compile_gate`: spills / scratch, MFMAs in the K loop, ``vmcnt(0)``
   beyond one acquire per tile); configs whose ISA digest equals a measured
   one reuse its measurement (no-op knobs cost one compile);
3. on the GPU worker: verify every candidate before it is timed (integer
   set bitwise, Gaussian set within the scaled-MFMA tolerance, launches
   bitwise identical), then time the candidates interleaved and shuffled with
   the anchor ``ref`` (the warm start), the current best and a recompiled
   copy of it (A-vs-A controls), peak protocol (idle gap, 1 warmup + 3 timed
   launches per sample) on Gaussian data, SCLK / power sampled per round;
4. tell the strategy (value = median over rounds of ``t / t_ref``), record
   every evaluation to JSONL, checkpoint.

A hung or faulting launch stalls the worker's heartbeat (one line per
launch); the supervisor kills it, blocklists the config in flight, restarts
the worker and continues. A run resumes from its checkpoint (same kernel
fingerprint and space) with ``--resume``.

Search policy (``--members``): the engine's discounted-Thompson
:class:`~rocke.dispatch.tuning.search.portfolio.Portfolio` over the signed
lookback climber (one lever at a time over every knob, losers retained,
re-tests after each accepted move, order restarts, then a pairwise covering
screen + interaction-aware GP over combinations; loaded from
``--lookback-module`` when given), hill climbing with ``lookback=3`` and
combination re-sweeps, GP + EI on a stratified (space-filling) seed, SHADE
differential evolution and Hyperband over a two-rung round ladder; warm
started from the kernel's current best. Stops on ``--stall-batches`` batches
without an improvement beyond the noise floor, the budget or the clock.

Absolute figures are for the operator's internal record only; repository
documents carry ratios.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import importlib.util
import json
import math
import os
import queue
import random
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .space import Mxfp6Space, default_config, kernel_fingerprint, kernel_module, label, legal, overrides_of, reg_estimate

MODULE = "builders.gfx950.mxfp6_gemm.tune"
FAKE_ENV = "MXFP6_TUNE_FAKE"  # test only: synthetic compile + GPU (CPU tests of the supervisor)
STATUS_OK, STATUS_INVALID, STATUS_COMPILE, STATUS_NUMERIC, STATUS_SPILL, STATUS_TIMEOUT, STATUS_RUN = (
    "ok", "invalid", "compile_error", "numeric_fail", "spill", "timeout", "run_error")


def say(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _fake() -> bool:
    return os.environ.get(FAKE_ENV, "") not in ("", "0")


def _sha(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def _jd(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return str(x)


def median(v: Sequence[float]) -> float:
    return float(statistics.median(v)) if v else math.inf


# ===================================================================== data


DISTS = ("gauss", "uniform", "int")


def _dist_id(L, d: str) -> int:
    return {"gauss": L.DIST_GAUSS, "uniform": L.DIST_UNIFORM_CODES, "int": L.DIST_INT}[d]


def _gen_shard(a):
    from kernels.gfx950.mxfp6_gemm import layout as L

    rows, k, dist, seed, tag, r0, r1 = a
    codes, scales = L._gen_rows(r0, r1, k, dist, seed, tag)
    return r0, codes, scales


def generate_parallel(rows: int, k: int, dist: int, seed: int, tag: int, workers: int):
    """The canonical operand (same generator and file format as ``bench gen``)."""
    from concurrent.futures import ProcessPoolExecutor

    from kernels.gfx950.mxfp6_gemm import layout as L

    if dist == L.DIST_INT:
        return L.int_operand(rows, k, seed=seed, tag=tag)
    codes = np.empty((rows, k * 6 // 8), dtype=np.uint8)
    scales = np.empty((rows, k // 32), dtype=np.uint8)
    jobs = [(rows, k, dist, seed, tag, r0, min(rows, r0 + 32)) for r0 in range(0, rows, 32)]
    if workers <= 1:
        for j in jobs:
            r0, c, s = _gen_shard(j)
            codes[r0 : r0 + c.shape[0]], scales[r0 : r0 + s.shape[0]] = c, s
    else:
        with ProcessPoolExecutor(workers) as ex:
            for r0, c, s in ex.map(_gen_shard, jobs, chunksize=1):
                codes[r0 : r0 + c.shape[0]], scales[r0 : r0 + s.shape[0]] = c, s
    return L.Operand(rows, k, codes, scales)


def _op_path(cache: Path, d: str, rows: int, k: int, seed: int, tag: int) -> Path:
    return cache / f"op_{d}_{rows}x{k}_s{seed}_t{tag}.bin"


def _ref_paths(cache: Path, d: str, m: int, n: int, k: int, seed: int) -> Tuple[Path, Path]:
    tag = f"{d}_{m}x{n}x{k}_s{seed}"
    return cache / f"ref_{tag}_c.npy", cache / f"ref_{tag}_s.npy"


def ensure_data(cache: Path, m: int, n: int, k: int, dists: Sequence[str], seed: int, workers: int) -> None:
    """Operands + float64 references for every dist in ``cache`` (``bench``'s file names: caches are shared)."""
    from kernels.gfx950.mxfp6_gemm import layout as L

    cache.mkdir(parents=True, exist_ok=True)
    if _fake():
        return
    for d in dists:
        ops = []
        for rows, tag in ((m, 1), (n, 2)):
            p = _op_path(cache, d, rows, k, seed, tag)
            if p.exists():
                ops.append(None)
                continue
            t0 = time.time()
            op = generate_parallel(rows, k, _dist_id(L, d), seed, tag, workers)
            L.save_operand(op, p)
            say(f"[data] {p.name} in {time.time() - t0:.1f} s")
            ops.append(op)
        pc, ps = _ref_paths(cache, d, m, n, k, seed)
        if not (pc.exists() and ps.exists()):
            t0 = time.time()
            a = ops[0] or L.load_operand(_op_path(cache, d, m, k, seed, 1))
            b = ops[1] or L.load_operand(_op_path(cache, d, n, k, seed, 2))
            c, s = L.reference_f64(a, b)
            np.save(pc, c)
            np.save(ps, s)
            say(f"[data] {d} float64 reference in {time.time() - t0:.1f} s")


# ===================================================================== compile


NOTE_KEYS = (".vgpr_count", ".agpr_count", ".sgpr_count", ".vgpr_spill_count", ".sgpr_spill_count",
             ".private_segment_fixed_size", ".group_segment_fixed_size")


def code_object_notes(hs: bytes) -> dict:
    """Resource fields from the code object's msgpack metadata (no readelf)."""

    def rd(i):
        t = hs[i]
        if t <= 0x7F:
            return t
        if t == 0xCC:
            return hs[i + 1]
        if t == 0xCD:
            return struct.unpack(">H", hs[i + 1 : i + 3])[0]
        if t == 0xCE:
            return struct.unpack(">I", hs[i + 1 : i + 5])[0]
        return None

    out = {}
    for k in NOTE_KEYS:
        kb = k.encode()
        p = hs.find(bytes([0xA0 | len(kb)]) + kb)
        if p >= 0:
            out[k[1:]] = rd(p + 1 + len(kb))
    return out


def find_objdump() -> Optional[str]:
    cand = [os.environ.get("OBJDUMP", "")] + sorted(glob.glob("/opt/rocm*/llvm/bin/llvm-objdump"), reverse=True)
    cand.append(shutil.which("llvm-objdump") or "")
    for c in cand:
        if c and os.path.exists(c):
            return c
    return None


_INSTR = re.compile(r"^\s+([a-z_][a-z0-9_]*)(?:\s+([^/]*?))?\s*(?://.*)?$")


def _instructions(lines: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for line in lines:
        mm = _INSTR.match(line)
        if mm and not line.lstrip().startswith(("//", ";")):
            out.append((mm.group(1), " ".join((mm.group(2) or "").split())))
    return out


def loop_stats(dis: str) -> dict:
    """Instruction mix of the longest backward-branch loop of a disassembly (the K loop) and kernel totals."""
    lines = dis.splitlines()
    addr, base = {}, None
    for i, line in enumerate(lines):
        mm = re.match(r"^([0-9a-f]{16}) <", line)
        if mm and base is None:
            base = int(mm.group(1), 16)
        mm = re.search(r"//\s*([0-9A-Fa-f]{12,16}):", line)
        if mm:
            addr[int(mm.group(1), 16)] = i
    best = None
    for i, line in enumerate(lines):
        mm = re.search(r"\bs_cbranch_\w+\b.*<[^>+]*\+0x([0-9a-f]+)>", line)
        if mm and base is not None:
            tgt = base + int(mm.group(1), 16)
            if tgt in addr and addr[tgt] < i and (best is None or i - addr[tgt] > best[1] - best[0]):
                best = (addr[tgt], i)
    allins = _instructions(lines)
    total = {"kernel_instr": len(allins), "kernel_scratch": sum(1 for m, _ in allins if m.startswith("scratch_"))}
    if not best:
        return {**total, "error": "no backward branch (no K loop found)"}
    body = lines[best[0] : best[1] + 1]
    ins = _instructions(body)
    c: Dict[str, int] = {}
    for m, _ in ins:
        c[m] = c.get(m, 0) + 1
    mf = sum(v for kk, v in c.items() if kk.startswith("v_mfma"))
    return {
        **total,
        "instr": len(ins),
        "mfma": mf,
        "non_mfma_per_mfma": round((len(ins) - mf) / mf, 3) if mf else None,
        "s_nop": c.get("s_nop", 0),
        "s_waitcnt": c.get("s_waitcnt", 0),
        "vmcnt0": sum(1 for m, ops in ins if m == "s_waitcnt" and "vmcnt(0)" in ops),
        "lgkmcnt0": sum(1 for m, ops in ins if m == "s_waitcnt" and "lgkmcnt(0)" in ops),
        "s_barrier": c.get("s_barrier", 0),
        "accvgpr": sum(v for kk, v in c.items() if kk.startswith("v_accvgpr")),
        "scratch": sum(v for kk, v in c.items() if kk.startswith("scratch_")),
        "ds_read": sum(v for kk, v in c.items() if kk.startswith("ds_read")),
        "ds_write": sum(v for kk, v in c.items() if kk.startswith("ds_write")),
        "dtl": sum(1 for m, ops in ins if m.startswith("buffer_load") and re.search(r"\blds\b", ops)),
        "vmem": sum(v for kk, v in c.items() if kk.startswith(("buffer_", "global_"))),
        "valu": sum(v for kk, v in c.items() if kk.startswith("v_") and not kk.startswith(("v_mfma", "v_accvgpr"))),
        "salu": sum(v for kk, v in c.items() if kk.startswith("s_") and kk not in ("s_nop", "s_waitcnt", "s_barrier")),
    }


def isa_digest(dis: str, notes: Mapping, extra: Mapping, kernel: str = "") -> str:
    """Digest of the instruction stream (addresses, encodings and the kernel name stripped) + resources + launch."""
    h = hashlib.sha256()
    text = dis.replace(kernel, "K") if kernel else dis
    for m, ops in _instructions(text.splitlines()):
        h.update(f"{m} {ops}\n".encode())
    h.update(json.dumps(dict(notes), sort_keys=True).encode())
    h.update(json.dumps(dict(extra), sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def host_layout(spec) -> dict:
    """Spec fields that select the host data layout + launch geometry (part of the ISA identity)."""
    out = {}
    for f in ("operand_layout", "a_layout", "scale_layout", "atom", "tile_m", "tile_n", "waves_m", "waves_n", "elem_fmt",
              "out_exp2"):
        if hasattr(spec, f):
            out[f] = getattr(spec, f)
    return out


def _fake_compile(spec, mod) -> dict:
    """Synthetic compile result (test only): spills for ``stages=4``, no-op ``read_lead`` under ``prefetch``."""
    over = overrides_of(mod, spec)
    sig = dict(over)
    sig.pop("scale_slot", None)  # a no-op knob of the synthetic compiler (the no-op prepass must drop it)
    if getattr(spec, "m_outer", True):
        sig.pop("serpentine", None)  # gated: live only under m_outer=False (the prepass must keep it)
    if getattr(spec, "frag_mode", "prefetch") == "prefetch":
        sig.pop("read_lead", None)
    spills = 8 if getattr(spec, "stages", 2) == 4 else 0
    notes = {"vgpr_count": 256, "agpr_count": 256, "sgpr_count": 64, "vgpr_spill_count": spills, "sgpr_spill_count": 0,
             "private_segment_fixed_size": 4 * spills, "group_segment_fixed_size": 98304}
    loop = {"kernel_instr": 2600, "kernel_scratch": spills, "instr": 600, "mfma": 256, "non_mfma_per_mfma": 1.3,
            "vmcnt0": 4, "scratch": 0, "s_waitcnt": 4}
    return {"ok": True, "name": mod.kernel_name(spec), "compile_s": 0.01, "notes": notes, "loop": loop,
            "isa": isa_digest("", notes, {"sig": sig, **host_layout(spec)}), "fake": True}


def cmd_compile(args) -> int:
    """Build + compile one spec in this process; write ``<out>`` (hsaco), ``<out>.s`` and ``<out>.json``."""
    full = json.loads(Path(args.spec).read_text()) if os.path.exists(args.spec) else json.loads(args.spec)
    meta = compile_to(full, Path(args.out))
    print(json.dumps({k: meta.get(k) for k in ("ok", "name", "compile_s", "notes", "loop", "isa", "error")}, default=_jd))
    return 0 if meta.get("ok") else 2


CLOCK_PROBE = "mx6tune_clock_probe"
CLOCK_PROBE_FIELDS = 5  # per CTA: memtime start / end, memrealtime start / end, sink
REALTIME_MHZ = 100.0  # s_memrealtime counts a constant 100 MHz clock


def build_clock_probe(iters: int):
    """A compute-only probe: every wave runs ``iters`` x 8 dependent ``v_fma_f32``; lane 0 of wave 0 of each CTA
    stores ``s_memtime`` (shader clock counter) and ``s_memrealtime`` (100 MHz) before and after, so
    ``(t1 - t0) / (r1 - r0) * 100`` is the CTA's effective shader clock in MHz over the launch."""
    from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType

    b = IRBuilder(CLOCK_PROBE)
    out = b.param("out", PtrType(I64, "global"))
    b.kernel.attrs["max_workgroup_size"] = 256
    tid = b.thread_id_x()
    cta = b.block_id_x()
    r0 = b.inline_asm("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", result_type=I64, result_name_hint="r0")
    c0 = b.inline_asm("s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", result_type=I64, result_name_hint="c0")
    loop = b.scf_for_iter(b.const_i32(0), b.const_i32(int(iters)), b.const_i32(1), [("acc", b.const_f32(1.0))],
                          iv_name="it")
    with loop as (_it, (acc,)):
        body = "\n\t".join(["v_fma_f32 $0, $0, $0, $0"] * 8)
        b.scf_yield(b.inline_asm(body, "=v,0", [acc], result_type=F32, result_name_hint="fma"))
    c1 = b.inline_asm("s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", result_type=I64, result_name_hint="c1")
    r1 = b.inline_asm("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", result_type=I64, result_name_hint="r1")
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        base = b.mul(cta, b.const_i32(CLOCK_PROBE_FIELDS))
        for i, v in enumerate((c0, c1, r0, r1)):
            b.global_store(out, b.add(base, b.const_i32(i)), v)
        b.global_store(out, b.add(base, b.const_i32(4)), b.zext(b.bitcast(loop.results[0], I32), I64))
    b.ret()
    return b.kernel


def clock_mhz(words: np.ndarray, n_cta: int) -> dict:
    """Effective shader clock of one probe launch from its ``[n_cta, 5]`` int64 record (median / min over CTAs)."""
    w = np.asarray(words, dtype=np.int64).reshape(n_cta, CLOCK_PROBE_FIELDS)
    dc, dr = (w[:, 1] - w[:, 0]).astype(np.float64), (w[:, 3] - w[:, 2]).astype(np.float64)
    ok = dr > 0
    mhz = dc[ok] / dr[ok] * REALTIME_MHZ
    return {"mhz_med": float(np.median(mhz)) if mhz.size else None, "mhz_min": float(mhz.min()) if mhz.size else None,
            "us_med": float(np.median(dr[ok]) / REALTIME_MHZ) if mhz.size else None, "n": int(mhz.size)}


def compile_to(full: Mapping, out: Path) -> dict:
    """Build + compile one full spec dict in this process; write the code object, disassembly and meta JSON."""
    if full.get("__kind__") == "clock_probe":
        meta = {"ok": False, "error": "fake"}
        if not _fake():
            from rocke.helpers.compile import compile_kernel

            try:
                art = compile_kernel(build_clock_probe(int(full["iters"])), arch="gfx950", backend="python")
                out.write_bytes(art.hsaco)
                meta = {"ok": True, "name": CLOCK_PROBE, "notes": code_object_notes(art.hsaco)}
            except Exception as e:  # noqa: BLE001
                meta = {"ok": False, "error": f"{type(e).__name__}: {e}"[-2000:]}
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, default=_jd))
        return meta
    mod = kernel_module()
    backend = str(full.get("__backend__", "llvm"))
    levers = str(full.get("__levers__") or os.environ.get("MXFP6_RUST_LEVERS", RUST_LEVERS_DEFAULT)).replace("+", ",")
    spec = replace(mod.Mxfp6GemmSpec(), **{k: v for k, v in dict(full).items() if not k.startswith("__")})
    ok, why = mod.validate_spec(spec)
    if not ok:
        meta = {"ok": False, "error": f"invalid: {why}"}
    elif _fake():
        meta = _fake_compile(spec, mod)
        if backend != "llvm":
            meta["isa"] = isa_digest("", meta["notes"], {"backend": backend, "isa": meta["isa"]})
        meta["backend"] = backend
        out.write_bytes(b"FAKE" + json.dumps(asdict(spec), sort_keys=True).encode())
    else:
        t0 = time.time()
        try:
            hsaco, extra_meta = _build_object(mod, spec, backend, out, levers)
        except Exception as e:  # noqa: BLE001 - every build failure is a compile error of this config
            meta = {"ok": False, "error": f"{backend}: {type(e).__name__}: {e}"[-3000:], "backend": backend}
        else:
            art_hsaco = hsaco
            out.write_bytes(art_hsaco)
            name = mod.kernel_name(spec)
            notes = code_object_notes(art_hsaco)
            od = find_objdump()
            dis, loop = "", {"error": "no llvm-objdump"}
            if od:
                try:
                    dis = subprocess.run([od, "-d", "--mcpu=gfx950", str(out)], capture_output=True, text=True,
                                         timeout=120).stdout
                    Path(str(out) + ".s").write_text(dis)
                    loop = loop_stats(dis)
                except (OSError, subprocess.SubprocessError) as e:
                    loop = {"error": f"objdump: {e}"}
            grid, block = mod.grid_of(spec)
            extra = {**host_layout(spec), "grid": list(grid), "block": list(block)}
            extra["backend"] = backend
            if dis:
                isa = isa_digest(dis, notes, extra, name)
            else:
                isa = hashlib.sha256(art_hsaco.replace(name.encode(), b"K") + backend.encode()).hexdigest()[:20]
            comgr = None
            try:
                from rocke.runtime import comgr as _c

                comgr = str(getattr(_c, "resolved_lib_path", lambda: None)())
            except Exception:  # noqa: BLE001
                pass
            meta = {"ok": True, "name": name, "compile_s": round(time.time() - t0, 3), "notes": notes, "loop": loop,
                    "isa": isa, "comgr": comgr if backend == "llvm" else None, "grid": list(grid), "block": list(block),
                    "params": mod.descriptor(spec).get("params"), "backend": backend, **extra_meta}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1, default=_jd))
    return meta


RUST_LEVERS_DEFAULT = "scalesep,dsrebase,dshoist,m0fill"
#: Search axes that are not ``Mxfp6GemmSpec`` fields (plus every ``rl_<lever>`` per-lever switch).
PSEUDO_KNOBS = ("backend", "rust_levers")
LEVER_PREFIX = "rl_"


def is_pseudo(name: str) -> bool:
    return name in PSEUDO_KNOBS or name.startswith(LEVER_PREFIX)


def spec_over(over: Mapping[str, object]) -> Dict[str, object]:
    """The spec-field part of a config dict (pseudo knobs dropped)."""
    return {k: v for k, v in over.items() if not is_pseudo(k)}


def levers_of(over: Mapping[str, object], default: str) -> str:
    """The Rust lever set of a config: the base set (``rust_levers`` or ``default``) with every ``rl_<lever>``
    switch applied on top (True adds the lever, False removes it). Levers that are not axes keep their base state
    (a switch list is NOT the whole set: that dropped scalesep / dsrebase in rounds 14-19)."""
    base = [x for x in str(over.get("rust_levers", default)).replace("+", ",").split(",") if x]
    for k, v in over.items():
        if k.startswith(LEVER_PREFIX):
            name = k[len(LEVER_PREFIX):]
            if v and name not in base:
                base.append(name)
            elif not v and name in base:
                base.remove(name)
    return ",".join(base)
#: Knobs that lost only to LLVM register allocation (spills / accumulator shuffles) and compile clean with the
#: direct compiler (notes/rust-gemm.md): re-tested under ``backend=rust`` as warm starts of the lookback.
RUST_RETEST = ({}, {"unroll": 2}, {"tail_mode": "clamp"}, {"reg_stages": 1}, {"reg_stages": 2}, {"skew_dtl": 2, "skew_read": 1},
               {"operand_layout": "c0c1p"}, {"unroll": 2, "tail_mode": "clamp"}, {"unroll": 2, "skew_dtl": 2, "skew_read": 1})


_EXE_LEVERS: Dict[str, Optional[frozenset]] = {}
_EXE_BYTES: Dict[str, bytes] = {}


def compiler_levers(exe: str) -> Optional[frozenset]:
    """The exact lever names of a ``target_compile`` build (``--list-levers``, out-68317511 and later); None for
    older builds without the option."""
    if exe not in _EXE_LEVERS:
        try:
            p = subprocess.run([exe, "--list-levers"], capture_output=True, text=True, timeout=60)
            names = frozenset(x.strip() for x in p.stdout.splitlines() if x.strip())
            _EXE_LEVERS[exe] = names if p.returncode == 0 and names else None
        except (OSError, subprocess.SubprocessError):
            _EXE_LEVERS[exe] = None
    return _EXE_LEVERS[exe]


def unknown_levers(exe: str, levers: str) -> List[str]:
    """Lever names of ``levers`` the compiler does not know (it would silently ignore them in builds before
    out-68317511): exact via ``--list-levers``; for older builds a byte-substring check of the binary (a short name
    can pass it by accident -- prefer a build with --list-levers)."""
    wanted = [lv for lv in (x.strip() for x in levers.replace("+", ",").split(",")) if lv]
    known = compiler_levers(exe)
    if known is not None:
        return [lv for lv in wanted if lv not in known]
    if exe not in _EXE_BYTES:
        _EXE_BYTES[exe] = Path(exe).read_bytes()
    return [lv for lv in wanted if lv.encode() not in _EXE_BYTES[exe]]


def _build_object(mod, spec, backend: str, out: Path, levers: str = RUST_LEVERS_DEFAULT) -> Tuple[bytes, dict]:
    """The code object of ``spec`` with the given compiler: ``llvm`` (rocKE lowering + COMGR) or ``rust`` (the
    direct compiler ``target_compile`` -- ``MXFP6_RUST_COMPILER`` -- on the same rocKE IR serialized with
    ``ir_serialize``, lever set ``MXFP6_RUST_LEVERS``)."""
    if hasattr(mod, "check_compiler"):
        mod.check_compiler(spec, backend)  # e.g. a Rust-only preset refuses the LLVM build
    kernel = mod.build_mxfp6_gemm(spec, arch="gfx950")
    if backend == "llvm":
        from rocke.helpers.compile import compile_kernel

        return compile_kernel(kernel, arch="gfx950", backend="python").hsaco, {}
    if backend != "rust":
        raise ValueError(f"unknown backend {backend!r}")
    from rocke.core.ir_serialize import serialize

    exe = os.environ.get("MXFP6_RUST_COMPILER", "")
    if not exe or not os.path.exists(exe):
        raise RuntimeError(f"no Rust direct compiler (MXFP6_RUST_COMPILER={exe!r})")
    unknown = unknown_levers(exe, levers)
    if unknown:  # older target_compile builds silently ignore lever names they do not know
        raise RuntimeError(f"levers {unknown} are unknown to {exe} (lever names absent from the binary)")
    text = serialize(kernel)
    stem = out.stem
    ir = out.with_name(stem + ".ir")
    ir.write_text(text)
    odir = out.with_name(stem + ".rust")
    env = {**os.environ, "ROCKE_DIRECT_LEVERS": levers}
    p = subprocess.run([exe, "gfx950", str(odir), str(ir)], capture_output=True, text=True, timeout=300, env=env)
    line = (p.stdout.strip().splitlines() or [""])[-1].split("\t")
    if len(line) < 2 or line[1] != "ok":
        raise RuntimeError(f"target_compile: {' '.join(line)[:500]} {p.stderr[-500:]}")
    stats = dict(zip(("instructions", "code_bytes", "vgprs", "sgprs", "lds_bytes"), (int(x) for x in line[2:7])))
    return (odir / f"{stem}.hsaco").read_bytes(), {"rust": {"levers": levers, "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                                          "exe": exe, **stats}}


class CompilePool:
    """One fresh process per compile (no LLVM state shared between builds), ``jobs`` at a time."""

    def __init__(self, work: Path, jobs: int, timeout_s: float = 300.0, env: Optional[dict] = None):
        self.work = work
        self.jobs = max(1, int(jobs))
        self.timeout_s = timeout_s
        self.env = dict(env) if env else None  # None: the environment at compile time (flags set later apply)
        work.mkdir(parents=True, exist_ok=True)

    def path_of(self, tag: str) -> Path:
        return self.work / f"{tag}.hsaco"

    def compile_many(self, items: Mapping[str, Mapping]) -> Dict[str, dict]:
        """``{tag: full spec dict}`` -> ``{tag: meta}`` (meta["hsaco"] = path)."""
        pending = list(items.items())
        procs: Dict[str, Tuple[subprocess.Popen, Path, float]] = {}
        res: Dict[str, dict] = {}
        while pending or procs:
            while pending and len(procs) < self.jobs:
                tag, full = pending.pop(0)
                out = self.path_of(tag)
                cached = Path(str(out) + ".json")
                if cached.exists() and out.exists():
                    meta = json.loads(cached.read_text())
                    meta["hsaco"] = str(out)
                    res[tag] = meta
                    continue
                if _fake():  # the synthetic compiler is trivial: no process per config in CPU tests
                    meta = compile_to(full, out)
                    meta["hsaco"] = str(out)
                    res[tag] = meta
                    continue
                spec_file = self.work / f"{tag}.spec.json"
                spec_file.write_text(json.dumps(dict(full), default=_jd))
                cmd = [sys.executable, "-m", MODULE, "compile", "--spec", str(spec_file), "--out", str(out)]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     env=self.env if self.env is not None else dict(os.environ))
                procs[tag] = (p, out, time.time())
            for tag in list(procs):
                p, out, t0 = procs[tag]
                if p.poll() is None:
                    if time.time() - t0 > self.timeout_s:
                        p.kill()
                        p.communicate()
                        res[tag] = {"ok": False, "error": f"compile timeout {self.timeout_s:.0f} s"}
                        del procs[tag]
                    continue
                so, se = p.communicate()
                mj = Path(str(out) + ".json")
                if mj.exists():
                    meta = json.loads(mj.read_text())
                else:
                    meta = {"ok": False, "error": (se or so or "")[-3000:]}
                if not meta.get("ok") and not meta.get("error"):
                    meta["error"] = (se or so or "")[-3000:]
                meta["hsaco"] = str(out)
                res[tag] = meta
                del procs[tag]
            if procs:
                time.sleep(0.02)
        return res


def compile_gate(meta: Mapping, spec, mod, mode: str) -> Tuple[str, str]:
    """``(status, reason)`` before any GPU time.

    ``strict``: 0 VGPR / SGPR spills and no scratch anywhere; ``loop``: no
    scratch access inside the K loop (spills outside it are tolerated);
    ``off``: compile success only. Every mode requires MFMAs in the K loop and
    at most one ``s_waitcnt vmcnt(0)`` per K tile of the loop body (the
    acquire; any other is an LLVM-inserted full drain).
    """
    if not meta.get("ok"):
        return STATUS_COMPILE, str(meta.get("error", "compile failed"))[:300]
    notes = meta.get("notes") or {}
    loop = meta.get("loop") or {}
    vs, ss = notes.get("vgpr_spill_count", 0) or 0, notes.get("sgpr_spill_count", 0) or 0
    scr = notes.get("private_segment_fixed_size", 0) or 0
    if mode == "strict" and (vs or ss or scr):
        return STATUS_SPILL, f"spills vgpr {vs} sgpr {ss} scratch {scr} B"
    if "error" in loop:
        if mode == "off":
            return STATUS_OK, ""
        return STATUS_OK, f"no ISA metrics ({loop['error']})"
    if mode in ("strict", "loop") and loop.get("scratch", 0):
        return STATUS_SPILL, f"{loop['scratch']} scratch instructions in the K loop"
    mf = loop.get("mfma", 0)
    if not mf:
        return STATUS_COMPILE, "no MFMA in the K loop"
    try:
        per_tile = mod.mfmas_per_tile(spec)
    except Exception:  # noqa: BLE001
        per_tile = 0
    if per_tile and mode != "off":
        tiles = max(1, mf // per_tile)
        if loop.get("vmcnt0", 0) > tiles:
            return STATUS_SPILL, f"{loop['vmcnt0']} vmcnt(0) in a {tiles}-tile K loop (LLVM full drains)"
    return STATUS_OK, ""


# ===================================================================== numeric tightness (ulps)


def f64_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """bf16 bits of float64 ``x`` rounded to nearest even in ONE step (no fp32 double rounding); normal range."""
    u = np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)
    drop = np.uint64(45)  # 52 - 7 fraction bits
    lsb = (u >> drop) & np.uint64(1)
    r = (u + (np.uint64(1) << np.uint64(44)) - np.uint64(1) + lsb) & ~((np.uint64(1) << drop) - np.uint64(1))
    f32 = r.view(np.float64).astype(np.float32)  # exact: r has a bf16 significand
    return (f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _bf16_ordered(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint16).astype(np.int64)
    mag = b & 0x7FFF
    return np.where(b & 0x8000, -mag, mag)


def ulp_hist(d: np.ndarray) -> dict:
    n = int(d.size)
    return {"n": n, "exact_frac": float(np.count_nonzero(d == 0)) / n, "le1_frac": float(np.count_nonzero(d <= 1)) / n,
            "max_ulp": int(d.max()) if n else 0,
            "hist": {"0": int(np.count_nonzero(d == 0)), "1": int(np.count_nonzero(d == 1)), "2": int(np.count_nonzero(d == 2)),
                     "3-7": int(np.count_nonzero((d >= 3) & (d <= 7))), ">=8": int(np.count_nonzero(d >= 8))}}


def ulp_stats(out_bits: np.ndarray, ref_f64: np.ndarray) -> dict:
    """bf16 ulp distance of an output to the correctly rounded answer RNE(bf16(exact fp64 dot))."""
    want = f64_to_bf16_rne(ref_f64)
    d = np.abs(_bf16_ordered(out_bits) - _bf16_ordered(want))
    return ulp_hist(d)


def bitwise_compare(a_bits: np.ndarray, b_bits: np.ndarray) -> dict:
    """Outputs of two kernels on identical inputs: count differing and their ulp distances."""
    d = np.abs(_bf16_ordered(a_bits) - _bf16_ordered(b_bits))
    h = ulp_hist(d)
    return {"differ": int(np.count_nonzero(d)), "identical": bool(not np.any(d)), **h}


# ===================================================================== aiter asm arm (same harness)

#: aiter's a6w6 asm GEMM (``hsa/gfx950/f6gemm``): 256x256 tiles, 128-K steps, two zero guard K tiles per row tile.
AITER_TILE, AITER_KSTEP, AITER_PADK = 256, 128, 2
AITER_TILE_BYTES, AITER_SCALE_TILE_BYTES = 24576, 1024
AITER_KARG_BYTES = 0x180
AITER_DEFAULT_SYMBOL = "f6gemm_dmabig_kernel_func"


def aiter_pack_operand(codes: np.ndarray, rows: int, k: int) -> np.ndarray:
    """aiter ``pack_big_torch`` (padK = 2): ``[rows/256][k/128 + 2][24576]``.

    Slot ``o`` of a (row tile, K step) = (16-row block ``o // 64``, K group
    ``(o % 64) // 16``, row ``o % 16``); its 24 canonical bytes split into
    16 (``c0``, at ``16 o``) + 8 (``c1``, at ``16384 + 8 o``). Guard tiles are zero.
    """
    nt, nk = rows // AITER_TILE, k // AITER_KSTEP
    x = np.ascontiguousarray(codes).reshape(nt, 16, 16, nk, 4, 24)  # t rb r16 s kg byte
    out = np.zeros((nt, nk + AITER_PADK, AITER_TILE_BYTES), dtype=np.uint8)
    out[:, :nk, :16384] = x[..., :16].transpose(0, 3, 1, 4, 2, 5).reshape(nt, nk, 16384)
    out[:, :nk, 16384:] = x[..., 16:].transpose(0, 3, 1, 4, 2, 5).reshape(nt, nk, 8192)
    return out.reshape(-1)


def aiter_pack_scales(scales: np.ndarray, rows: int, k: int) -> np.ndarray:
    """aiter ``pack_scale_torch``: ``[rows/256][k/128 + 2][1024]``; byte ``off`` = (half ``off // 512``,
    K group ``(off % 512) // 128``, row ``(off % 128) // 8``, 16-row sub-tile ``off % 8``); guards 127."""
    nt, nk = rows // AITER_TILE, k // AITER_KSTEP
    s = np.ascontiguousarray(scales).reshape(nt, 2, 8, 16, nk, 4)  # t su sub r16 s kg
    out = np.full((nt, nk + AITER_PADK, AITER_SCALE_TILE_BYTES), 127, dtype=np.uint8)
    out[:, :nk, :] = s.transpose(0, 4, 1, 5, 3, 2).reshape(nt, nk, 1024)
    return out.reshape(-1)


def aiter_kernargs(c: int, a: int, b: int, sa: int, sb: int, m: int, n: int, k: int) -> bytes:
    """The 0x180-byte a6w6 ``KernelArgs`` (every field on a 16-byte slot)."""
    buf = bytearray(AITER_KARG_BYTES)
    struct.pack_into("<Q", buf, 0x00, c)  # ptr_D
    struct.pack_into("<Q", buf, 0x20, a)
    struct.pack_into("<Q", buf, 0x30, b)
    struct.pack_into("<f", buf, 0x40, 1.0)  # alpha
    struct.pack_into("<f", buf, 0x50, 0.0)  # beta
    struct.pack_into("<I", buf, 0x60, n)  # stride_D0 (elements)
    struct.pack_into("<I", buf, 0xE0, m)
    struct.pack_into("<I", buf, 0xF0, n)
    struct.pack_into("<I", buf, 0x100, k)
    struct.pack_into("<Q", buf, 0x110, sa)
    struct.pack_into("<Q", buf, 0x120, sb)
    struct.pack_into("<I", buf, 0x130, k // 32)
    struct.pack_into("<I", buf, 0x150, k // 32)
    return bytes(buf)


def bootstrap_ci(num: Sequence[float], den: Sequence[float], n_boot: int = 4000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Median of the per-round ratios ``num[r] / den[r]`` with a paired (by round) bootstrap CI."""
    r = np.asarray([x / y for x, y in zip(num, den)], dtype=np.float64)
    if r.size == 0:
        return {"median": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_boot, r.size))
    meds = np.median(r[idx], axis=1)
    lo, hi = np.quantile(meds, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"median": float(np.median(r)), "lo": float(lo), "hi": float(hi), "n": int(r.size)}


# ===================================================================== GPU worker


class Clock:
    """sysfs SCLK / board power sampler for the HIP device in use (5 ms period)."""

    def __init__(self, bdf: str):
        base = Path("/sys/bus/pci/devices") / bdf.lower()
        self.sclk = base / "pp_dpm_sclk"
        self.power = None
        hw = base / "hwmon"
        for h in sorted(hw.glob("hwmon*")) if hw.exists() else []:
            for nm in ("power1_average", "power1_input"):
                if (h / nm).exists():
                    self.power = h / nm
                    break
            if self.power:
                break
        self._stop = threading.Event()
        self.mhz: list = []
        self.watts: list = []
        self.trace: list = []  # (monotonic t, MHz, W)
        self._th = None

    def window(self, t0: float, t1: float, lag_s: float = 0.002) -> dict:
        """Median SCLK / board power of the samples taken inside ``[t0, t1 + lag_s]`` (one arm's timed launches)."""
        return window_median(self.trace, t0, t1 + lag_s)

    def _read_sclk(self):
        try:
            for line in self.sclk.read_text().splitlines():
                if "*" in line:
                    return float(line.split(":")[1].strip().split("M")[0])
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _read_power(self):
        if not self.power:
            return None
        try:
            return float(self.power.read_text().strip()) / 1e6
        except (OSError, ValueError):
            return None

    def start(self, period_s: float = 0.005):
        self.mhz, self.watts, self.trace = [], [], []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                v = self._read_sclk()
                if v:
                    self.mhz.append(v)
                w = self._read_power()
                if w:
                    self.watts.append(w)
                self.trace.append((time.monotonic(), v, w))
                time.sleep(period_s)

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._th:
            self._th.join()
        return {"sclk_med": median(self.mhz) if self.mhz else None, "sclk_min": min(self.mhz) if self.mhz else None,
                "sclk_max": max(self.mhz) if self.mhz else None,
                "watts_med": median(self.watts) if self.watts else None, "n": len(self.mhz)}


def window_median(trace: Sequence[Tuple[float, Optional[float], Optional[float]]], t0: float, t1: float) -> dict:
    """Median MHz / W of ``(t, MHz, W)`` samples with ``t0 <= t <= t1``."""
    mhz = [m for t, m, _ in trace if t0 <= t <= t1 and m]
    w = [x for t, _, x in trace if t0 <= t <= t1 and x]
    return {"sclk": median(mhz) if mhz else None, "watts": median(w) if w else None, "n": len(w)}


class _Proto:
    """JSON-lines protocol on the process's original stdout; everything else printed goes to stderr."""

    def __init__(self):
        sys.stdout.flush()
        fd = os.dup(1)
        os.dup2(2, 1)
        self.f = os.fdopen(fd, "w", buffering=1)
        sys.stdout = sys.stderr

    def send(self, **kw) -> None:
        self.f.write(json.dumps(kw, default=_jd) + "\n")
        self.f.flush()


def _fake_time(over: Mapping[str, object], rng: random.Random) -> float:
    """Synthetic landscape (test only): per-(knob, level) factors, one pair interaction, 0.3 % noise."""
    t = 1.0
    for k, v in sorted(over.items()):
        h = int(hashlib.sha1(f"{k}={v}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        t *= 1.0 + 0.06 * (h - 0.6)
    if over.get("setprio") == 1 and over.get("dtl_delay") == 2:
        t *= 0.95
    return t * (1.0 + rng.gauss(0.0, 0.003))


def _fake_match(env: str, over: Mapping[str, object]) -> bool:
    rule = os.environ.get(env, "")
    if not rule:
        return False
    k, _, v = rule.partition("=")
    return k in over and str(over[k]) == v


class GpuWorker:
    """The GPU side: modules, device data per host layout, verify, time, sustained."""

    def __init__(self, proto: _Proto):
        self.proto = proto
        self.mod = kernel_module()
        from kernels.gfx950.mxfp6_gemm import layout as L

        self.L = L
        self.loaded: Dict[str, dict] = {}
        self.ref_out: Dict[str, np.ndarray] = {}
        self.fake = _fake()
        self.rng = random.Random(0)
        self.dead = False
        self.outs: Dict[Tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------ setup
    def init(self, cache: str, m: int, n: int, k: int, seed: int, dists: Sequence[str]) -> dict:
        self.m, self.n, self.k = m, n, k
        self.flop = 2.0 * m * n * k
        if self.fake:
            self.dists = list(dists)
            return {"bdf": "fake", "fake": True}
        L = self.L
        cache_p = Path(cache)
        self.data = {}
        for d in dists:
            a = L.load_operand(_op_path(cache_p, d, m, k, seed, 1))
            b = L.load_operand(_op_path(cache_p, d, n, k, seed, 2))
            pc, ps = _ref_paths(cache_p, d, m, n, k, seed)
            self.data[d] = (a, b, (np.load(pc), np.load(ps)))
        self.dists = list(dists)
        from rocke.runtime.hip_module import Runtime, _resolve_hip

        self.rt = Runtime()
        hip = _resolve_hip()
        buf = ctypes.create_string_buffer(64)
        hip.hipDeviceGetPCIBusId(buf, 64, 0)
        self.bdf = buf.value.decode()
        self.clk = Clock(self.bdf)
        self.c_dev = self.rt.alloc(m * n * 2)
        self.c_host = np.empty((m, n), dtype=np.uint16)
        self.op_dev: Dict[tuple, Tuple[int, int]] = {}
        self.sc_dev: Dict[tuple, Tuple[int, int]] = {}
        return {"bdf": self.bdf, "sysfs_sclk": self.clk.sclk.exists(), "sysfs_power": bool(self.clk.power)}

    def _put(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        ptr = self.rt.alloc(arr.nbytes)
        self.rt.memcpy_h2d(ptr, (ctypes.c_ubyte * int(arr.nbytes)).from_buffer(arr), arr.nbytes)
        return ptr

    def _inputs(self, spec, d: str) -> Tuple[int, int, int, int]:
        L = self.L
        a, b, _ = self.data[d]
        lay = getattr(spec, "operand_layout", "canonical")
        la = lb = lay
        if hasattr(self.mod, "operand_layout_of"):  # a_layout: A's own layout (e.g. canonical activations)
            la, lb = self.mod.operand_layout_of(spec, "A"), self.mod.operand_layout_of(spec, "B")
        ko = (d, la, lb, spec.tile_m, spec.tile_n)
        if ko not in self.op_dev:
            if hasattr(L, "pack_operand"):
                self.op_dev[ko] = (self._put(L.pack_operand(a, la, spec.tile_m)), self._put(L.pack_operand(b, lb, spec.tile_n)))
            else:
                self.op_dev[ko] = (self._put(a.codes), self._put(b.codes))
        sl = getattr(spec, "scale_layout", "kstep")
        ks = (d, spec.atom, sl)
        if ks not in self.sc_dev:
            if hasattr(L, "pack_scales_for"):
                self.sc_dev[ks] = (self._put(L.pack_scales_for(a.scales, spec.atom, sl)),
                                   self._put(L.pack_scales_for(b.scales, spec.atom, sl)))
            else:
                self.sc_dev[ks] = (self._put(L.pack_scales(a.scales, spec.atom)), self._put(L.pack_scales(b.scales, spec.atom)))
        return self.op_dev[ko] + self.sc_dev[ks]

    def _kargs(self, name: str, d: str) -> bytes:
        x = self.loaded[name]
        key = ("kargs", d)
        if key not in x:
            if x.get("kind") == "aiter":
                if ("aiter", d) not in self.op_dev:
                    a, b, _ = self.data[d]
                    self.op_dev[("aiter", d)] = (
                        self._put(aiter_pack_operand(a.codes, self.m, self.k)), self._put(aiter_pack_operand(b.codes, self.n, self.k)),
                        self._put(aiter_pack_scales(a.scales, self.m, self.k)), self._put(aiter_pack_scales(b.scales, self.n, self.k)))
                pa, pb, sa, sb = self.op_dev[("aiter", d)]
                x[key] = aiter_kernargs(self.c_dev, pa, pb, sa, sb, self.m, self.n, self.k)
            else:
                pa, pb, sa, sb = self._inputs(x["spec"], d)
                x[key] = struct.pack("<QQQQQ", pa, pb, sa, sb, self.c_dev)
        return x[key]

    def load_aiter(self, name: str, co: str, symbol: str = AITER_DEFAULT_SYMBOL) -> dict:
        """aiter's asm code object as an arm: grid (N/256, M/256), 256 threads, its own packed inputs."""
        sp = self.mod.Mxfp6GemmSpec()
        if self.fake:
            self.loaded[name] = {"spec": sp, "over": {"aiter": symbol}, "kind": "aiter"}
            return {"ok": True}
        if self.m % AITER_TILE or self.n % AITER_TILE or self.k % AITER_KSTEP:
            raise RuntimeError("aiter a6w6 needs M, N % 256 and K % 128")
        m = self.rt.load_module(Path(co).read_bytes())
        self.loaded[name] = {"mod": m, "fn": m.get_function(symbol), "grid": (self.n // AITER_TILE, self.m // AITER_TILE, 1),
                             "block": (256, 1, 1), "spec": sp, "over": {"aiter": symbol}, "kind": "aiter"}
        return {"ok": True, "md5": hashlib.md5(Path(co).read_bytes()).hexdigest()}

    # ------------------------------------------------------------ modules
    def load(self, name: str, hsaco: str, spec: Mapping, kernel: str) -> dict:
        sp = replace(self.mod.Mxfp6GemmSpec(), **dict(spec))
        over = overrides_of(self.mod, sp)
        if self.fake:
            self.loaded[name] = {"spec": sp, "over": over}
            return {"ok": True}
        params = self.mod.descriptor(sp).get("params")
        if list(params) != ["a", "b", "sa", "sb", "c"]:
            raise RuntimeError(f"kernel params {params} changed: update GpuWorker._kargs")
        m = self.rt.load_module(Path(hsaco).read_bytes())
        grid, block = self.mod.grid_of(sp)
        self.loaded[name] = {"mod": m, "fn": m.get_function(kernel), "grid": grid, "block": block, "spec": sp, "over": over}
        return {"ok": True}

    def unload(self, names: Sequence[str]) -> dict:
        for nm in names:
            x = self.loaded.pop(nm, None)
            if x and "mod" in x:
                x["mod"].unload()
        return {"ok": True}

    def _hb(self, name: str, phase: str) -> None:
        self.proto.send(ev="hb", arm=name, phase=phase, t=time.time())
        if self.fake and _fake_match("MXFP6_TUNE_FAKE_HANG", self.loaded[name]["over"]):
            time.sleep(3600)
        if self.fake and _fake_match("MXFP6_TUNE_FAKE_CRASH", self.loaded[name]["over"]):
            os._exit(134)
        if self.fake and _fake_match("MXFP6_TUNE_FAKE_HIPERR", self.loaded[name]["over"]):
            self.dead = True  # a sticky HIP error: every later call of this process fails too
            raise RuntimeError("HipError: hipStreamSynchronize: hipError(700) an illegal memory access was encountered")

    # ------------------------------------------------------------ verify
    def run_once(self, name: str, d: str) -> np.ndarray:
        x = self.loaded[name]
        kargs = self._kargs(name, d)
        self.rt.memset(self.c_dev, 0xFF, self.c_host.nbytes)  # NaN fill
        self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
        self.rt.wait_stream(0)
        self.rt.memcpy_d2h((ctypes.c_ubyte * int(self.c_host.nbytes)).from_buffer(self.c_host), self.c_dev, self.c_host.nbytes)
        return self.c_host.copy()

    def verify(self, name: str, dists: Sequence[str], launches: int, keep: bool = False, compare: str = "",
               ulp: bool = False, keep_out: bool = False) -> dict:
        """Integer set bitwise, other dists within tolerance, every launch bitwise identical; ``ulp``: the ulp
        histogram vs the correctly rounded answer; ``keep_out``: hold the output for :meth:`compare`."""
        res = {}
        x = self.loaded[name]
        for d in dists:
            if self.fake:
                self._hb(name, f"verify:{d}")
                bad = 0 if not _fake_match("MXFP6_TUNE_FAKE_BAD", x["over"]) else 17
                if x["spec"].mutant != "none":
                    bad = 1000
                res[d] = {"pass": bad == 0, "bad": bad, "deterministic": True, "all_pass": bad == 0}
                continue
            ref_c, ref_s = self.data[d][2]
            outs, chk = [], []
            for i in range(1 if x["spec"].mutant != "none" else max(1, launches)):
                self._hb(name, f"verify:{d}:{i}")
                o = self.run_once(name, d)
                outs.append(o)
                chk.append(self.L.check_against_reference(o, ref_c, ref_s, exact=(d == "int")))
            same = all(np.array_equal(outs[0], o) for o in outs[1:])
            r = {**chk[0], "all_pass": all(c["pass"] for c in chk), "deterministic": same}
            if keep:
                self.ref_out[d] = outs[0]
            if compare and d in self.ref_out:
                r["bitwise_vs_ref"] = bool(np.array_equal(outs[0], self.ref_out[d]))
            if ulp:
                r["ulp"] = ulp_stats(outs[0], ref_c)
            if keep_out:
                self.outs[(name, d)] = outs[0]
            res[d] = r
        ok = all(r["all_pass"] and r["deterministic"] for r in res.values())
        return {"ok": ok, "dists": res}

    def load_clock_probe(self, name: str, hsaco: str, n_cta: int = 256, slots: int = 64) -> dict:
        if self.fake:
            self.loaded[name] = {"spec": self.mod.Mxfp6GemmSpec(), "over": {"clock_probe": 1}, "kind": "clock"}
            return {"ok": True}
        m = self.rt.load_module(Path(hsaco).read_bytes())
        buf = self.rt.alloc(n_cta * CLOCK_PROBE_FIELDS * 8 * slots)
        self.loaded[name] = {"mod": m, "fn": m.get_function(CLOCK_PROBE), "grid": (n_cta, 1, 1), "block": (256, 1, 1),
                             "spec": self.mod.Mxfp6GemmSpec(), "over": {"clock_probe": 1}, "kind": "clock", "buf": buf,
                             "n_cta": n_cta, "slots": slots}
        return {"ok": True}

    def coldstart(self, arms: Sequence[str], dist: str, gap_ms: float, launches: int, steady: int, reps: int,
                  seed: int) -> dict:
        """Per-launch times of ``launches`` back-to-back launches right after a ``gap_ms`` idle gap, then ``steady``
        more (steady state), ``reps`` times per arm, arms shuffled per rep; a clock-probe arm also returns its
        in-kernel effective shader clock per launch."""
        rng = random.Random(seed)
        out: Dict[str, dict] = {a: {"ms": [], "mhz": []} for a in arms}
        total = launches + steady
        for _ in range(reps):
            order = list(arms)
            rng.shuffle(order)
            for a in order:
                self._hb(a, "coldstart")
                if self.fake:
                    base = _fake_time(self.loaded[a]["over"], self.rng)
                    out[a]["ms"].append([base * (1.1 if i < 3 else 1.0) for i in range(total)])
                    continue
                x = self.loaded[a]
                self.rt.wait_stream(0)
                time.sleep(gap_ms / 1000.0)
                ev = [self.rt.event() for _ in range(total + 1)]
                ev[0].record()
                for i in range(total):
                    if x.get("kind") == "clock":
                        slot = min(i, x["slots"] - 1)
                        kargs = struct.pack("<Q", x["buf"] + slot * x["n_cta"] * CLOCK_PROBE_FIELDS * 8)
                    else:
                        kargs = self._kargs(a, dist)
                    self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
                    ev[i + 1].record()
                ev[-1].synchronize()
                out[a]["ms"].append([ev[i].elapsed_to(ev[i + 1]) for i in range(total)])
                for e in ev:
                    e.destroy()
                if x.get("kind") == "clock":
                    n = min(total, x["slots"])
                    host = np.empty(n * x["n_cta"] * CLOCK_PROBE_FIELDS, dtype=np.int64)
                    self.rt.memcpy_d2h((ctypes.c_ubyte * int(host.nbytes)).from_buffer(host), x["buf"], host.nbytes)
                    per = host.reshape(n, x["n_cta"] * CLOCK_PROBE_FIELDS)
                    out[a]["mhz"].append([clock_mhz(per[i], x["n_cta"])["mhz_med"] for i in range(n)])
        return {"coldstart": out, "launches": launches, "steady": steady, "gap_ms": gap_ms}

    def compare(self, a: str, b: str, dist: str) -> dict:
        """Bitwise comparison of two kept outputs (identical inputs)."""
        if self.fake:
            return {"differ": 0, "identical": True, "n": 0, "fake": True}
        return bitwise_compare(self.outs[(a, dist)], self.outs[(b, dist)])

    # ------------------------------------------------------------ timing
    def _sample(self, name: str, d: str, gap_s: float, warmup: int, iters: int) -> float:
        """One peak-protocol sample: idle gap, warmup launches, ``iters`` timed back to back; ms per launch."""
        self._last_window = None
        if self.fake:
            time.sleep(min(gap_s, 0.001))
            return _fake_time(self.loaded[name]["over"], self.rng)
        x = self.loaded[name]
        kargs = self._kargs(name, d)
        self.rt.wait_stream(0)
        time.sleep(gap_s)
        for _ in range(warmup):
            self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
        e0, e1 = self.rt.event(), self.rt.event()
        t0 = time.monotonic()
        e0.record()
        for _ in range(iters):
            self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
        e1.record()
        e1.synchronize()
        self._last_window = (t0, time.monotonic())
        ms = e0.elapsed_to(e1) / iters
        e0.destroy()
        e1.destroy()
        return ms

    def time_rounds(self, arms: Mapping[str, int], dist: str, gap_ms: float, warmup: int, iters: int, seed: int,
                    discard: int = 0) -> dict:
        """Interleaved shuffled rounds; arm ``a`` takes part in its first ``arms[a]`` rounds. ``discard`` leading
        rounds (every arm) are run and thrown away (the first samples after a verify / load pause run cold)."""
        rng = random.Random(seed)
        ms: Dict[str, List[float]] = {a: [] for a in arms}
        watts: Dict[str, List[Optional[float]]] = {a: [] for a in arms}
        clocks = []
        n = max(arms.values()) if arms else 0
        for r in range(-int(discard), n):
            order = [a for a in arms if r < 0 or arms[a] > r]
            rng.shuffle(order)
            if not self.fake:
                self.clk.start()
            wins = {}
            for a in order:
                self._hb(a, f"time:{r}")
                v = self._sample(a, dist, gap_ms / 1000.0, warmup, iters)
                wins[a] = self._last_window
                if r >= 0:
                    ms[a].append(v)
            c = {} if self.fake else self.clk.stop()
            if r >= 0:
                clocks.append({"round": r, "order": order, **c})
                for a in order:
                    w = None if self.fake or wins[a] is None else self.clk.window(*wins[a])["watts"]
                    watts[a].append(w)
        return {"ms": ms, "watts": watts, "clocks": clocks, "flop": self.flop}

    def sustained(self, names: Sequence[str], dist: str, seconds: float, rounds: int, seed: int, idle_s: float = 2.0) -> dict:
        rng = random.Random(seed)
        out: Dict[str, list] = {a: [] for a in names}
        for _ in range(rounds):
            order = list(names)
            rng.shuffle(order)
            for a in order:
                self._hb(a, "sustained")
                if self.fake:
                    ms = _fake_time(self.loaded[a]["over"], self.rng) * 1.05
                    out[a].append({"ms": ms, "tflops": self.flop / (ms * 1e9)})
                    continue
                est = self._sample(a, dist, 0.0, 3, 5)
                iters = max(50, int(seconds * 1000.0 / max(0.05, est)))
                x = self.loaded[a]
                kargs = self._kargs(a, dist)
                self.rt.wait_stream(0)
                time.sleep(idle_s)
                for _ in range(20):
                    self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
                self.clk.start()
                e0, e1 = self.rt.event(), self.rt.event()
                e0.record()
                for _ in range(iters):
                    self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
                e1.record()
                e1.synchronize()
                ms = e0.elapsed_to(e1) / iters
                e0.destroy()
                e1.destroy()
                cs = self.clk.stop()
                mj = cs["watts_med"] * ms if cs.get("watts_med") else None  # W x ms = mJ per GEMM
                out[a].append({"ms": ms, "tflops": self.flop / (ms * 1e9), "iters": iters, "mj": mj, **cs})
        return {"sustained": out}


def cmd_worker(args) -> int:
    proto = _Proto()
    w = GpuWorker(proto)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cid, cmd = req.get("id"), req.get("cmd")
        try:
            if cmd == "quit":
                proto.send(ev="done", id=cid, result={"ok": True})
                break
            if w.dead:
                raise RuntimeError("HipError: hipModuleUnload: hipError(700) an illegal memory access was encountered")
            fn = {"init": w.init, "load": w.load, "load_aiter": w.load_aiter, "unload": w.unload, "compare": w.compare, "coldstart": w.coldstart,
                  "load_clock_probe": w.load_clock_probe, "verify": w.verify, "time": w.time_rounds,
                  "sustained": w.sustained}[cmd]
            res = fn(**req.get("args", {}))
            proto.send(ev="done", id=cid, result=res)
        except Exception as e:  # noqa: BLE001 - reported to the supervisor, which decides
            import traceback

            proto.send(ev="error", id=cid, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:])
    return 0


# ===================================================================== supervisor side of the worker


class WorkerFault(RuntimeError):
    """The worker hung (no heartbeat) or died; ``arm`` / ``phase`` = the last heartbeat."""

    def __init__(self, kind: str, arm: Optional[str], phase: Optional[str], detail: str = ""):
        super().__init__(f"worker {kind} at {arm} {phase} {detail}")
        self.kind, self.arm, self.phase, self.detail = kind, arm, phase, detail


class WorkerError(RuntimeError):
    """A Python exception inside the worker; ``arm`` / ``phase`` = its last heartbeat, ``hip`` = a HIP error
    (the context is then dead: the worker must be restarted; an asynchronous fault surfaces at a later call)."""

    def __init__(self, text: str, arm: Optional[str], phase: Optional[str]):
        super().__init__(text)
        self.arm, self.phase = arm, phase
        self.hip = "hipError" in text or "HipError" in text


class GiveUp(RuntimeError):
    """Too many worker faults: the search stops (summary still written)."""


class WorkerClient:
    def __init__(self, log: Path, hang_s: float = 60.0, env: Optional[dict] = None):
        self.log = log
        self.hang_s = hang_s
        self.env = dict(env) if env else None  # None: the environment at compile time (flags set later apply)
        self.p: Optional[subprocess.Popen] = None
        self.q: "queue.Queue" = queue.Queue()
        self._id = 0
        self.restarts = 0

    def start(self) -> None:
        kw = {"start_new_session": True} if os.name == "posix" else {}
        self.p = subprocess.Popen([sys.executable, "-m", MODULE, "worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(self.log, "ab"), text=True, bufsize=1, env=self.env, **kw)
        self.q = queue.Queue()
        p, q = self.p, self.q

        def rd():
            for line in p.stdout:
                q.put(line)
            q.put(None)

        threading.Thread(target=rd, daemon=True).start()

    def kill(self) -> None:
        if self.p and self.p.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.p.pid, signal.SIGKILL)
                else:
                    self.p.kill()
            except OSError:
                pass
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def call(self, cmd: str, hang_s: Optional[float] = None, **kw):
        if self.p is None or self.p.poll() is not None:
            raise WorkerFault("dead", None, None, "worker not running")
        self._id += 1
        cid = self._id
        try:
            self.p.stdin.write(json.dumps({"id": cid, "cmd": cmd, "args": kw}, default=_jd) + "\n")
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise WorkerFault("dead", None, None, str(e)) from e
        last = (None, None)
        limit = hang_s or self.hang_s
        while True:
            try:
                line = self.q.get(timeout=limit)
            except queue.Empty:
                self.kill()
                raise WorkerFault("hang", last[0], last[1], f"no heartbeat for {limit:.0f} s") from None
            if line is None:
                rc = self.p.wait(timeout=30)
                raise WorkerFault("died", last[0], last[1], f"exit {rc}")
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") == "hb":
                last = (ev.get("arm"), ev.get("phase"))
                continue
            if ev.get("id") != cid:
                continue
            if ev.get("ev") == "error":
                raise WorkerError(f"worker error in {cmd}: {ev.get('error')}\n{ev.get('tb', '')}", last[0], last[1])
            return ev.get("result")

    def close(self) -> None:
        if self.p and self.p.poll() is None:
            try:
                self.call("quit", hang_s=30)
            except (WorkerFault, RuntimeError):
                pass
            self.kill()


# ===================================================================== statistics


def win_rule(ratios: Sequence[float], controls: Sequence[Sequence[float]], margin: float = 0.002) -> str:
    """WIN: median below every control's minimum by ``margin`` and below 1 in >= 3 of 4 rounds; LOSS mirrored."""
    if not ratios:
        return "NA"
    med = median(ratios)
    ctl = [c for c in controls if c]
    cmin = min(min(c) for c in ctl) if ctl else 1.0
    cmax = max(max(c) for c in ctl) if ctl else 1.0
    below = sum(1 for r in ratios if r < 1.0) / len(ratios)
    above = sum(1 for r in ratios if r > 1.0) / len(ratios)
    if med < cmin - margin and below >= 0.75:
        return "WIN"
    if med > cmax + margin and above >= 0.75:
        return "LOSS"
    return "TIE"


def noise_sigma(log_ratios: Sequence[float]) -> float:
    """Robust per-round sigma (1.4826 MAD) of control log-ratios."""
    if len(log_ratios) < 4:
        return 0.01
    m = statistics.median(log_ratios)
    return 1.4826 * statistics.median(abs(x - m) for x in log_ratios)


# ===================================================================== ledger


def _cfg(defaults: Mapping, over: Mapping) -> dict:
    return {**defaults, **dict(over)}


def config_diff(defaults: Mapping, a: Mapping, b: Mapping) -> Dict[str, object]:
    """Knobs of ``b`` (overrides) that differ from ``a`` (overrides), as ``{knob: b's value}``."""
    ca, cb = _cfg(defaults, a), _cfg(defaults, b)
    return {k: cb[k] for k in sorted(cb) if ca.get(k) != cb[k]}


def build_ledger(records: Sequence[Mapping], defaults: Mapping, thr: float) -> dict:
    """Lever ledger from evaluation records (``status``, ``over``, ``value`` = ratio vs ref, ``ref_over``,
    ``vs_best`` = ratio vs the incumbent of its batch, ``best_over``).

    ``solo_vs_ref``: configs one knob away from the warm start (gain = 1 - value);
    ``solo_vs_base``: one knob away from the incumbent they were measured
    against (gain = 1 - vs_best), per base; ``pairs``: two knobs away from a
    base where both solo moves were measured on the same base (interaction =
    pair gain - sum of solo gains). Verdicts: win / loss beyond ``thr``, else
    neutral. Failures are listed per lever with their status.
    """
    solo_ref: Dict[str, list] = {}
    solo_base: Dict[Tuple[str, str], list] = {}
    pair_base: Dict[Tuple[str, str], list] = {}
    fails: Dict[str, Dict[str, int]] = {}
    noop: Dict[str, str] = {}
    for r in records:
        if r.get("type") != "eval":
            continue
        over = r.get("over") or {}
        ref_over = r.get("ref_over") or {}
        dref = config_diff(defaults, ref_over, over)
        if r.get("isa_dup"):  # same instruction stream as a measured config: a no-op lever there, not a measurement
            if len(dref) == 1:
                (k, v), = dref.items()
                noop[f"{k}={v}"] = "same ISA as the warm start" if r["isa_dup"] == r.get("ref_key") else "same ISA as " + str(r["isa_dup"])[:12]
            continue
        if r.get("status") != STATUS_OK:
            if len(dref) == 1:
                (k, v), = dref.items()
                f = fails.setdefault(f"{k}={v}", {})
                f[r.get("status")] = f.get(r.get("status"), 0) + 1
            continue
        if r.get("value") is None:
            continue
        if len(dref) == 1:
            (k, v), = dref.items()
            solo_ref.setdefault(f"{k}={v}", []).append(1.0 - float(r["value"]))
        if r.get("vs_best") is None or r.get("best_over") is None:
            continue
        dbase = config_diff(defaults, r["best_over"], over)
        blab = label(r["best_over"])
        if len(dbase) == 1:
            (k, v), = dbase.items()
            solo_base.setdefault((f"{k}={v}", blab), []).append(1.0 - float(r["vs_best"]))
        elif len(dbase) == 2:
            levers = sorted(f"{k}={v}" for k, v in dbase.items())
            pair_base.setdefault(("|".join(levers), blab), []).append(1.0 - float(r["vs_best"]))

    def verdict(g: float) -> str:
        return "win" if g > thr else ("loss" if g < -thr else "neutral")

    rows_ref = sorted(({"lever": k, "n": len(v), "gain": median(v), "verdict": verdict(median(v))} for k, v in solo_ref.items()),
                      key=lambda x: -x["gain"])
    rows_base = sorted(({"lever": k, "base": b, "n": len(v), "gain": median(v), "verdict": verdict(median(v))}
                        for (k, b), v in solo_base.items()), key=lambda x: (x["base"], -x["gain"]))
    pairs = []
    for (pk, b), v in pair_base.items():
        a1, a2 = pk.split("|")
        s1, s2 = solo_base.get((a1, b)), solo_base.get((a2, b))
        g = median(v)
        row = {"pair": pk, "base": b, "n": len(v), "gain": g}
        if s1 and s2:
            row["interaction"] = g - median(s1) - median(s2)
        pairs.append(row)
    pairs.sort(key=lambda x: -abs(x.get("interaction", 0.0)))
    return {"threshold": thr, "solo_vs_ref": rows_ref, "solo_vs_base": rows_base, "pairs": pairs,
            "failures": {k: v for k, v in sorted(fails.items())}, "noop": dict(sorted(noop.items()))}


def ledger_text(led: Mapping, top: int = 12) -> str:
    out = [f"lever ledger (gain = 1 - t/t_base; verdict threshold {led['threshold']:.4f})", "", "solo vs ref (warm start):"]
    for r in led["solo_vs_ref"]:
        out.append(f"  {r['verdict']:<7} {r['gain']:+.4f}  n={r['n']:<2} {r['lever']}")
    out += ["", "solo vs the incumbent of the batch (per base):"]
    for r in led["solo_vs_base"]:
        out.append(f"  {r['verdict']:<7} {r['gain']:+.4f}  n={r['n']:<2} {r['lever']:<28} base={r['base']}")
    out += ["", f"pairs (top {top} by |interaction|):"]
    for r in led["pairs"][:top]:
        it = f"{r['interaction']:+.4f}" if "interaction" in r else "   n/a "
        out.append(f"  gain {r['gain']:+.4f} interaction {it}  {r['pair']}  base={r['base']}")
    if led.get("noop"):
        out += ["", "no-op levers (ISA identical to a measured config, not re-timed):"]
        out += [f"  {k}: {v}" for k, v in led["noop"].items()]
    if led["failures"]:
        out += ["", "solo failures (status counts):"]
        for k, v in led["failures"].items():
            out.append(f"  {k}: {v}")
    return "\n".join(out) + "\n"


# ===================================================================== strategy


def load_lookback_module(path: str):
    """The signed-lookback climber module (vendored read-only copy of the qwen38 tuner's ``strategies.py``)."""
    spec = importlib.util.spec_from_file_location("mxfp6_vendored_lookback", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_strategy(sp: Mxfp6Space, args, warm: Sequence[Tuple[Mapping, str]]):
    """The portfolio (``args.members``) with the warm starts first."""
    from rocke.dispatch.tuning.search import (
        GPEI,
        DifferentialEvolution,
        HillClimb,
        Hyperband,
        Portfolio,
        WarmStart,
    )

    space = sp.space
    start = sp.base_genome
    built = []
    names = [m.strip() for m in args.members.split(",") if m.strip()]
    lb_mod = None
    if "lookback" in names:
        if args.lookback_module and os.path.exists(args.lookback_module):
            lb_mod = load_lookback_module(args.lookback_module)
        else:
            say(f"[strategy] lookback module not found ({args.lookback_module!r}): member dropped")
    for i, m in enumerate(names):
        s = args.seed * 101 + i + 1
        if m == "lookback":
            if lb_mod is None:
                continue
            built.append(lb_mod.SignedLookback(space, s, start=start, min_gain=args.min_gain, lookback_batch=args.batch,
                                               budget=args.budget, order_restarts=1))
        elif m == "hill":
            built.append(HillClimb(space, s, start=start, lookback=3, combo_knobs=4, min_gain=args.min_gain, n_init=0))
        elif m == "gp":
            built.append(GPEI(space, s, n_startup=8, init=[start]))
        elif m == "de":
            built.append(DifferentialEvolution(space, s, init=[start], variant="current_to_pbest1bin", adapt="shade"))
        elif m == "hyperband":
            built.append(Hyperband(space, s, fidelities=(0, -1), max_configs=27, init=[start]))
        else:
            raise ValueError(f"unknown member {m!r}")
    if not built:
        raise ValueError("no strategy member left")
    ws = [WarmStart(config=dict(space.decode(sp.genome_of(over))), source=src) for over, src in warm]
    return Portfolio(space, args.seed, members=built, warm=ws, min_gain=0.001)


# ===================================================================== the run


class Run:
    """The supervisor's state: every evaluated key, controls, incumbent, noise, records."""

    def __init__(self, args, sp: Mxfp6Space):
        self.args = args
        self.sp = sp
        self.mod = sp.mod
        self.defaults = default_config(self.mod)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.mirror = Path(args.mirror) if args.mirror else None
        self.pool = CompilePool(self.out / "work", args.jobs)
        self.E: Dict[str, dict] = {}  # key -> entry
        self.isa_index: Dict[str, str] = {}  # isa digest -> measured key
        self.block: Dict[str, str] = {}  # key -> status (hang / fault)
        self.ctl_logr: List[float] = []
        self.batch_no = 0
        self.n_gpu = 0
        self.n_told = 0
        self.stall = 0
        self.idle_batches = 0
        self.best_hist: List[dict] = []
        self.gate_mode = args.gate
        self.t0 = time.time()
        self.jsonl = open(self.out / "evals.jsonl", "a")
        self.worker: Optional[WorkerClient] = None
        self.ref_key = sp.key(sp.base_genome)
        self.ref_over = sp.overrides(sp.base_genome)
        self.inc_key = self.ref_key
        self.loaded_ctl: Dict[str, str] = {}  # arm -> key it holds
        self.flop = 2.0 * args.m * args.n * args.k
        self.ref_tf: List[float] = []

    # ------------------------------------------------------------ io
    def rec(self, **kw) -> None:
        kw.setdefault("t", round(time.time() - self.t0, 2))
        self.jsonl.write(json.dumps(kw, default=_jd) + "\n")
        self.jsonl.flush()

    def sync(self) -> None:
        if not self.mirror:
            return
        self.mirror.mkdir(parents=True, exist_ok=True)
        for f in ("evals.jsonl", "ckpt.json", "summary.json", "ledger.txt", "space.json", "worker.log"):
            p = self.out / f
            if p.exists():
                try:
                    shutil.copy2(p, self.mirror / f)
                except OSError as e:
                    say(f"[sync] {f}: {e}")

    def rebind_space(self, sp: Mxfp6Space) -> None:
        """Swap in a narrower space with the same base (keys of the base stay valid)."""
        old = self.E.pop(self.ref_key)
        self.sp = sp
        self.ref_key = sp.key(sp.base_genome)
        self.inc_key = self.ref_key
        self.E[self.ref_key] = old
        self.isa_index[old.get("isa")] = self.ref_key

    # ------------------------------------------------------------ helpers
    def full_spec(self, key_or_over) -> dict:
        """Every spec field of a config (pseudo knobs such as ``backend`` dropped): what the worker loads."""
        over = self.E[key_or_over]["over"] if isinstance(key_or_over, str) else key_or_over
        over = spec_over(over)
        return asdict(replace(self.mod.Mxfp6GemmSpec(), **{**over, "m": self.args.m, "n": self.args.n, "k": self.args.k}))

    def compile_item(self, key_or_over) -> dict:
        """What the compile pool builds: the full spec + the compiler backend."""
        over = self.E[key_or_over]["over"] if isinstance(key_or_over, str) else key_or_over
        return {**self.full_spec(over), "__backend__": over.get("backend", "llvm"),
                "__levers__": levers_of(over, self.args.rust_levers)}

    def tag_of(self, key: str, suffix: str = "") -> str:
        return _sha(key + suffix)

    def pooled(self, key: str) -> float:
        e = self.E.get(key)
        if not e or e.get("status") != STATUS_OK:
            return math.inf
        if key == self.ref_key:
            return 1.0
        return median(e["ratios"]) if e["ratios"] else math.inf

    def incumbent(self) -> str:
        cands = [(self.pooled(k), e["first_batch"], k) for k, e in self.E.items() if e.get("status") == STATUS_OK]
        return min(cands)[2] if cands else self.ref_key

    def min_gain(self) -> float:
        sig = noise_sigma(self.ctl_logr) if len(self.ctl_logr) >= 8 else 0.01
        return max(self.args.min_gain, self.args.gain_sigmas * 1.2533 * sig / math.sqrt(max(1, self.args.rounds)))

    # ------------------------------------------------------------ worker
    def worker_start(self) -> None:
        for attempt in range(3):
            self.worker = WorkerClient(self.out / "worker.log", hang_s=self.args.hang_s)
            self.worker.start()
            try:
                info = self.worker.call("init", hang_s=900, cache=str(self.args.cache), m=self.args.m, n=self.args.n,
                                        k=self.args.k, seed=self.args.seed_data, dists=self.dists())
                gpu_guard(info)
                say(f"[worker] up: {info}")
                self.gpu_info = info
                self.loaded_ctl = {}
                return
            except (WorkerFault, RuntimeError) as e:
                say(f"[worker] init failed ({e}); retry {attempt + 1}")
                self.worker.kill()
        raise SystemExit("worker does not start: giving up")

    def worker_restart(self, why: str) -> None:
        """Kill + restart the worker; reload the anchor and re-take its reference output (bitwise comparisons)."""
        say(f"[worker] restart: {why}")
        self.worker.restarts += 1
        n = self.worker.restarts
        if self.worker:
            self.worker.kill()
        self.worker_start()
        self.worker.restarts = n
        if n > self.args.max_restarts:
            raise GiveUp(f"{n} worker restarts")
        e = self.E.get(self.ref_key)
        if e and "meta" in e:
            self.worker.call("load", name="ref", hsaco=e["meta"]["hsaco"], spec=self.full_spec(self.ref_key),
                             kernel=e["meta"]["name"])
            self.loaded_ctl["ref"] = self.ref_key
            self.worker.call("verify", name="ref", dists=self.dists(), launches=1, keep=True)

    def dists(self) -> List[str]:
        return [d for d in self.args.verify_dists.split(",") if d]

    def ensure_loaded(self, arm: str, key: str, hsaco_key: Optional[str] = None) -> None:
        """Load ``key``'s code object (or its recompiled copy) under ``arm``."""
        if self.loaded_ctl.get(arm) == (hsaco_key or key):
            return
        if arm in self.loaded_ctl:
            self.worker.call("unload", names=[arm])
        e = self.E[key]
        meta = e["rc_meta"] if hsaco_key and hsaco_key.endswith("#rc") else e["meta"]
        self.worker.call("load", name=arm, hsaco=meta["hsaco"], spec=self.full_spec(key), kernel=meta["name"])
        self.loaded_ctl[arm] = hsaco_key or key

    def ensure_rc(self, key: str) -> None:
        e = self.E[key]
        if "rc_meta" in e:
            return
        res = self.pool.compile_many({self.tag_of(key, "#rc"): self.compile_item(key)})
        e["rc_meta"] = next(iter(res.values()))
        try:
            e["rc_identical"] = Path(e["rc_meta"]["hsaco"]).read_bytes() == Path(e["meta"]["hsaco"]).read_bytes()
        except OSError:
            e["rc_identical"] = None

    def contenders(self, n: int) -> List[str]:
        """The ``n`` best configs by pooled ratio other than the incumbent and ref: re-timed every batch (racing),
        so one lucky batch cannot keep a config on top."""
        if n <= 0:
            return []
        ranked = sorted((self.pooled(k), k) for k, e in self.E.items()
                        if e.get("status") == STATUS_OK and not e.get("isa_dup") and k not in (self.inc_key, self.ref_key)
                        and "meta" in e)
        return [k for v, k in ranked[:n] if math.isfinite(v)]

    def sync_thresholds(self, strat) -> None:
        """Climbers accept a move only beyond the measured noise of a batch median."""
        g = self.min_gain()
        for m in getattr(strat, "members", []):
            if hasattr(m, "min_gain") and m.name in ("hill", "lookback"):
                m.min_gain = g

    def controls(self) -> Dict[str, Tuple[str, str]]:
        """``{arm: (key, hsaco id)}``: ref, the incumbent (twin of ref when it is ref) and its recompiled copy."""
        inc = self.inc_key
        self.ensure_rc(inc)
        return {"ref": (self.ref_key, self.ref_key), "ctl_a": (inc, inc), "ctl_b": (inc, inc + "#rc")}

    # ------------------------------------------------------------ session gate
    def session_gate(self) -> None:
        """Compile + verify the warm start, require its must-fail mutants to fail, pick the gate mode."""
        base_over = self.ref_over
        items = {self.tag_of(self.ref_key): self.compile_item(base_over)}
        muts = [m for m in getattr(self.mod, "MUTANTS", ()) if m != "none"]
        for mu in muts:
            items[self.tag_of(self.ref_key, "#mut" + mu)] = self.compile_item({**base_over, "mutant": mu})
        res = self.pool.compile_many(items)
        meta = res[self.tag_of(self.ref_key)]
        spec = self.sp.spec_of(self.sp.base_genome)
        st, why = compile_gate(meta, spec, self.mod, self.gate_mode)
        if st != STATUS_OK and self.gate_mode == "strict" and st == STATUS_SPILL:
            say(f"[gate] the warm start fails the strict gate ({why}): gate mode -> 'loop' for this run")
            self.gate_mode = "loop"
            st, why = compile_gate(meta, spec, self.mod, self.gate_mode)
        if st != STATUS_OK:
            raise SystemExit(f"warm start fails the compile gate: {st} {why}")
        self.E[self.ref_key] = {"over": base_over, "status": STATUS_OK, "meta": meta, "ratios": [], "tf": [], "batches": [],
                                "first_batch": -1, "value": 1.0, "isa": meta.get("isa"),
                                "full": True, "value_full": True}
        self.isa_index[meta.get("isa")] = self.ref_key
        self.worker.call("load", name="ref", hsaco=meta["hsaco"], spec=self.full_spec(base_over), kernel=meta["name"])
        self.loaded_ctl["ref"] = self.ref_key
        v = self.worker.call("verify", name="ref", dists=self.dists(), launches=self.args.verify_launches, keep=True)
        say(f"[gate] warm start verify: {'PASS' if v['ok'] else 'FAIL'} {json.dumps(v['dists'], default=_jd)[:400]}")
        if not v["ok"]:
            raise SystemExit("the warm start fails verification: stop")
        self.E[self.ref_key]["verify"] = v
        mres = {}
        for mu in muts:
            mm = res[self.tag_of(self.ref_key, "#mut" + mu)]
            if not mm.get("ok"):
                mres[mu] = "compile failed"
                continue
            nm = f"mut_{mu}"
            self.worker.call("load", name=nm, hsaco=mm["hsaco"], spec=self.full_spec({**base_over, "mutant": mu}),
                             kernel=mm["name"])
            dd = [d for d in self.dists() if not (mu == "scale_sel" and d == "int")]
            vm = self.worker.call("verify", name=nm, dists=dd, launches=1)
            self.worker.call("unload", names=[nm])
            mres[mu] = "caught" if not vm["ok"] else "PASSED"
        say(f"[gate] must-fail mutants: {mres}")
        self.rec(type="session", gate_mode=self.gate_mode, ref_over=base_over, ref_meta=_meta_summary(meta), verify=v,
                 mutants=mres, gpu=getattr(self, "gpu_info", None), space=self.sp.describe()["knobs"])
        if any(r == "PASSED" for r in mres.values()):
            raise SystemExit("a must-fail mutant passed verification: the gate is broken, stop")

    # ------------------------------------------------------------ one batch
    def run_batch(self, strat) -> bool:
        a = self.args
        trials = strat.ask(a.batch)
        if not trials:
            return False
        self.batch_no += 1
        bno = self.batch_no
        obs: Dict[str, object] = {}
        todo: Dict[str, Tuple[object, str]] = {}  # uid -> (trial, key)
        pending_keys: Dict[str, List[str]] = {}  # key -> uids waiting on it
        for tr in trials:
            key = self.sp.key(tr.genome)
            if key is None:
                obs[tr.uid] = self._obs(tr, None, STATUS_INVALID, detail="illegal")
                continue
            e = self.E.get(key)
            full = tr.fidelity == -1
            if key in self.block:
                obs[tr.uid] = self._obs(tr, key, self.block[key], detail="blocklisted", cached=True)
            elif e and e.get("status") != STATUS_OK and e.get("status") is not None:
                obs[tr.uid] = self._obs(tr, key, e["status"], detail=e.get("why", ""), cached=True)
            elif e and e.get("status") == STATUS_OK and (e.get("full") or not full) and e.get("value") is not None:
                obs[tr.uid] = self._obs(tr, key, STATUS_OK, value=e["value"], samples=e["ratios"][: a.rounds], cached=True)
            else:
                pending_keys.setdefault(key, []).append(tr.uid)
                todo[tr.uid] = (tr, key)
        # ---------------- compile + gate + ISA dedup
        new_keys = [k for k in pending_keys if k not in self.E or "meta" not in self.E[k]]
        items = {self.tag_of(k): self.compile_item(self.sp.overrides(todo[pending_keys[k][0]][0].genome)) for k in new_keys}
        tc = time.time()
        metas = self.pool.compile_many(items) if items else {}
        compile_s = time.time() - tc
        measure: List[str] = []
        dup_of: Dict[str, str] = {}
        for k in pending_keys:
            tr = todo[pending_keys[k][0]][0]
            over = self.sp.overrides(tr.genome)
            e = self.E.setdefault(k, {"over": over, "ratios": [], "tf": [], "batches": [], "first_batch": bno,
                                      "origin": tr.origin})
            if "meta" not in e:
                e["meta"] = metas[self.tag_of(k)]
                spec = self.sp.spec_of(tr.genome)
                e["reg"] = reg_estimate(self.mod, spec)
                st, why = compile_gate(e["meta"], spec, self.mod, self.gate_mode)
                if st != STATUS_OK:
                    e.update(status=st, why=why)
                    self._finish_key(k, pending_keys[k], todo, obs, st, why, bno)
                    continue
                e["gate_note"] = why
            isa = e["meta"].get("isa")
            e["isa"] = isa
            twin = self.isa_index.get(isa)
            if twin and twin != k and self.E.get(twin, {}).get("status") == STATUS_OK and self.E[twin].get("value") is not None:
                e.update(status=STATUS_OK, value=self.E[twin]["value"], full=self.E[twin].get("full", True), isa_dup=twin)
                e["ratios"] = list(self.E[twin]["ratios"])
                for uid in pending_keys[k]:
                    tr2 = todo[uid][0]
                    obs[uid] = self._obs(tr2, k, STATUS_OK, value=e["value"], samples=e["ratios"][: a.rounds], cached=True,
                                         detail=f"isa-dup of {twin[:12]}")
                self._record_eval(k, bno, cached=True, detail=f"isa-dup of {twin[:16]}")
                continue
            if isa in [self.E[m].get("isa") for m in measure]:
                dup_of[k] = next(m for m in measure if self.E[m].get("isa") == isa)
                continue
            measure.append(k)
        if measure:
            self._gpu_batch(measure, pending_keys, todo, obs, bno, compile_s)
        for k, twin in dup_of.items():
            e, et = self.E[k], self.E[twin]
            e.update(status=et.get("status"), value=et.get("value"), why=et.get("why", ""), full=et.get("full"), isa_dup=twin,
                     ratios=list(et.get("ratios", [])))
            for uid in pending_keys[k]:
                tr2 = todo[uid][0]
                obs[uid] = self._obs(tr2, k, e["status"], value=e.get("value"), samples=e["ratios"][: a.rounds],
                                     cached=True, detail=f"isa-dup of {twin[:12]}")
            self._record_eval(k, bno, cached=True, detail=f"isa-dup of {twin[:16]}")
        ordered = [obs[tr.uid] for tr in trials if tr.uid in obs]
        missing = [tr for tr in trials if tr.uid not in obs]
        for tr in missing:  # never leave a trial untold
            ordered.append(self._obs(tr, self.sp.key(tr.genome), STATUS_RUN, detail="not evaluated"))
        strat.tell(ordered)
        self.n_told += len(ordered)
        self.sync_thresholds(strat)
        # ---------------- incumbent / stall / checkpoint
        before = self.pooled(self.inc_key) if self.best_hist else 1.0
        new_inc = self.incumbent()
        prev_best = min((h["pooled"] for h in self.best_hist), default=1.0)
        timed_now = self.n_gpu > (self.best_hist[-1]["n_gpu"] if self.best_hist else 0)
        self.idle_batches = 0 if timed_now else self.idle_batches + 1
        if self.pooled(new_inc) < prev_best * (1.0 - self.min_gain()):
            self.stall = 0
        elif timed_now:
            self.stall += 1
        if new_inc != self.inc_key:
            say(f"[best] incumbent {label(self.E[self.inc_key]['over'])} -> {label(self.E[new_inc]['over'])} "
                f"({before:.4f} -> {self.pooled(new_inc):.4f} of ref)")
        self.inc_key = new_inc
        self.best_hist.append({"batch": bno, "key": new_inc, "pooled": self.pooled(new_inc), "n_gpu": self.n_gpu})
        self.rec(type="batch_end", batch=bno, incumbent=label(self.E[new_inc]["over"]), pooled=self.pooled(new_inc),
                 n_gpu=self.n_gpu, stall=self.stall, min_gain=self.min_gain(), noise_sigma=noise_sigma(self.ctl_logr))
        self.checkpoint(strat)
        return True

    def _obs(self, tr, key, status, value=None, samples=(), cached=False, detail="", measure_s=0.0, compile_s=0.0):
        from rocke.dispatch.tuning.search import Observation

        full = tr.fidelity == -1
        v = float(value) if (status == STATUS_OK and value is not None) else math.inf
        return Observation(trial=tr, key=key, status=status, value=v, samples=tuple(float(x) for x in samples),
                           fidelity=tr.fidelity, full=full, cached=cached, detail=detail, measure_s=measure_s,
                           compile_s=compile_s)

    def _finish_key(self, k, uids, todo, obs, status, why, bno) -> None:
        for uid in uids:
            obs[uid] = self._obs(todo[uid][0], k, status, detail=why)
        self._record_eval(k, bno, detail=why)

    def _record_eval(self, k: str, bno: int, cached: bool = False, detail: str = "", extra: Optional[dict] = None) -> None:
        e = self.E[k]
        best_over = self.E[self.inc_key]["over"]
        self.rec(type="eval", batch=bno, key=k, ref_key=self.ref_key, origin=e.get("origin"), over=e["over"], label=label(e["over"]),
                 ref_over=self.ref_over, best_over=best_over, status=e.get("status"), why=e.get("why", ""), value=e.get("value"),
                 vs_best=(extra or {}).get("vs_best"), verdict=(extra or {}).get("verdict"), cached=cached, detail=detail,
                 compile=_meta_summary(e.get("meta") or {}), reg=e.get("reg"), verify=(e.get("verify") or {}).get("dists"),
                 timing=(extra or {}).get("timing"), isa_dup=e.get("isa_dup"))

    def _gpu_batch(self, measure: List[str], pending_keys, todo, obs, bno: int, compile_s: float) -> None:
        a = self.args
        # verify every new candidate (a fault blocklists the culprit and restarts the worker)
        ok_keys: List[str] = []
        names = {k: f"c{i}" for i, k in enumerate(measure)}

        def restart(why: str) -> None:
            self.worker_restart(why)
            for k2 in ok_keys:
                self.worker.call("load", name=names[k2], hsaco=self.E[k2]["meta"]["hsaco"], spec=self.full_spec(k2),
                                 kernel=self.E[k2]["meta"]["name"])

        for k in measure:
            e = self.E[k]
            for attempt in range(2):
                try:
                    self.worker.call("load", name=names[k], hsaco=e["meta"]["hsaco"], spec=self.full_spec(k),
                                     kernel=e["meta"]["name"])
                    v = self.worker.call("verify", name=names[k], dists=self.dists(), launches=a.verify_launches,
                                         compare="ref")
                    break
                except WorkerFault as f:
                    culprit = names[k] if f.arm in (None, names[k]) else f.arm
                    if culprit == names[k]:
                        st = STATUS_TIMEOUT if f.kind == "hang" else STATUS_RUN
                        self.block[k] = st
                        e.update(status=st, why=f"verify {f}")
                        restart(str(f))
                        v = None
                        break
                    restart(str(f))
                except WorkerError as err:
                    if not err.hip:
                        e.update(status=STATUS_RUN, why=str(err)[:300])
                        v = None
                        break
                    # a fault while loading / verifying this candidate (its launches are synchronized): the culprit
                    culprit = names[k] if err.arm in (None, names[k]) else err.arm
                    restart(f"HIP error at {culprit}: {str(err)[:160]}")
                    if culprit == names[k]:
                        self.block[k] = STATUS_RUN
                        e.update(status=STATUS_RUN, why=f"verify: {str(err)[:240]}")
                        v = None
                        break
            else:
                v = None
            if v is None:
                if e.get("status") is None:
                    e.update(status=STATUS_RUN, why="verify did not complete")
                self._finish_key(k, pending_keys[k], todo, obs, e["status"], e["why"], bno)
                continue
            e["verify"] = v
            if not v["ok"]:
                e.update(status=STATUS_NUMERIC, why=_verify_why(v))
                self.worker.call("unload", names=[names[k]])
                self._finish_key(k, pending_keys[k], todo, obs, STATUS_NUMERIC, e["why"], bno)
                continue
            ok_keys.append(k)
        if not ok_keys:
            return
        # timing: candidates + controls, interleaved, shuffled
        for attempt in range(3):
            ctl = self.controls()
            try:
                for arm, (key, hid) in ctl.items():
                    self.ensure_loaded(arm, key, hid)
                fid = {k: max(todo[u][0].fidelity for u in pending_keys[k]) for k in ok_keys}
                rounds = {names[k]: (a.rounds if fid[k] == -1 else a.screen_rounds) for k in ok_keys}
                rounds.update({arm: a.rounds for arm in ctl})
                race = self.contenders(a.race)
                for i, rk in enumerate(race):
                    self.ensure_loaded(f"race{i}", rk, rk)
                    rounds[f"race{i}"] = a.rounds
                tm = time.time()
                res = self.worker.call("time", arms=rounds, dist=a.time_dist, gap_ms=a.gap_ms, warmup=a.search_warmup,
                                       iters=a.search_iters, seed=a.seed * 7919 + bno, discard=a.discard_rounds)
                measure_s = time.time() - tm
                break
            except WorkerFault as f:
                culprit = next((k for k in ok_keys if names[k] == f.arm), None)
                if culprit:
                    st = STATUS_TIMEOUT if f.kind == "hang" else STATUS_RUN
                    self.block[culprit] = st
                    self.E[culprit].update(status=st, why=f"time {f}")
                    self._finish_key(culprit, pending_keys[culprit], todo, obs, st, self.E[culprit]["why"], bno)
                    ok_keys.remove(culprit)
                self.worker_restart(str(f))
                for k in ok_keys:
                    self.worker.call("load", name=names[k], hsaco=self.E[k]["meta"]["hsaco"], spec=self.full_spec(k),
                                     kernel=self.E[k]["meta"]["name"])
                if not ok_keys:
                    return
            except WorkerError as err:
                if err.hip:  # an asynchronous fault: blocklist the arm of the last heartbeat if it is a candidate
                    culprit = next((k for k in ok_keys if names[k] == err.arm), None)
                    say(f"[time] HIP error at {err.arm} ({err.phase}): {str(err)[:200]}")
                    if culprit:
                        self.block[culprit] = STATUS_RUN
                        self.E[culprit].update(status=STATUS_RUN, why=f"time: {str(err)[:240]}")
                        self._finish_key(culprit, pending_keys[culprit], todo, obs, STATUS_RUN, self.E[culprit]["why"], bno)
                        ok_keys.remove(culprit)
                    self.worker_restart("HIP error during timing")
                    for k in ok_keys:
                        self.worker.call("load", name=names[k], hsaco=self.E[k]["meta"]["hsaco"], spec=self.full_spec(k),
                                         kernel=self.E[k]["meta"]["name"])
                    if not ok_keys:
                        return
                    continue
                say(f"[time] worker error, batch {bno} not timed: {str(err)[:400]}")
                for k in ok_keys:
                    self.E[k].update(status=STATUS_RUN, why=f"timing error: {str(err)[:200]}")
                    self._finish_key(k, pending_keys[k], todo, obs, STATUS_RUN, self.E[k]["why"], bno)
                self.worker_restart("worker error during timing")
                return
        else:
            for k in ok_keys:
                self.E[k].update(status=STATUS_RUN, why="timing did not complete")
                self._finish_key(k, pending_keys[k], todo, obs, STATUS_RUN, "timing did not complete", bno)
            return
        ms = res["ms"]
        t_ref = ms["ref"]
        t_inc = ms["ctl_a"]
        ctl_series = [[x / y for x, y in zip(ms["ctl_b"], t_inc)]]
        if self.inc_key == self.ref_key:
            ctl_series.append([x / y for x, y in zip(t_inc, t_ref)])
        for s in ctl_series:
            self.ctl_logr.extend(math.log(x) for x in s if x > 0)
        self.ref_tf.extend(self.flop / (x * 1e9) for x in t_ref)
        clocks = res.get("clocks", [])
        self.rec(type="batch", batch=bno, incumbent=label(self.E[self.inc_key]["over"]),
                 ctl_vs_inc=[median(s) for s in ctl_series], ref_tf=median([self.flop / (x * 1e9) for x in t_ref]),
                 inc_ratio=median([x / y for x, y in zip(t_inc, t_ref)]), clocks=clocks, n_cand=len(ok_keys),
                 rc_identical=self.E[self.inc_key].get("rc_identical"))
        wt = res.get("watts") or {}

        def energy(arm: str) -> List[Optional[float]]:
            return [(w * x if w else None) for w, x in zip(wt.get(arm, []), ms[arm])]

        e_ref = energy("ref")
        for k in ok_keys:
            e = self.E[k]
            t = ms[names[k]]
            n_r = len(t)
            r_ref = [x / y for x, y in zip(t, t_ref[:n_r])]
            e_c = energy(names[k])
            r_energy = [x / y for x, y in zip(e_c, e_ref[:n_r]) if x and y]
            if a.objective == "energy" and len(r_energy) >= max(3, n_r // 2):
                r_ref = r_energy
            r_inc = [x / y for x, y in zip(t, t_inc[:n_r])]
            full = n_r >= a.rounds
            verdict = win_rule(r_inc, [s[:n_r] for s in ctl_series])
            e.setdefault("ratios", []).extend(r_ref)
            e.setdefault("tf", []).extend(self.flop / (x * 1e9) for x in t)
            e.setdefault("batches", []).append(bno)
            e.update(status=STATUS_OK, full=bool(e.get("full")) or full)
            if e.get("value") is None or (full and not e.get("value_full")):
                e["value"] = median(r_ref)
                e["value_full"] = full
            self.n_gpu += 1
            for uid in pending_keys[k]:
                tr = todo[uid][0]
                obs[uid] = self._obs(tr, k, STATUS_OK, value=median(r_ref), samples=r_ref, measure_s=measure_s / len(ok_keys),
                                     compile_s=compile_s / max(1, len(measure)))
            if e.get("isa"):
                self.isa_index.setdefault(e["isa"], k)
            self._record_eval(k, bno, extra={
                "vs_best": median(r_inc), "verdict": verdict,
                "timing": {"rounds": n_r, "ratio_ref": r_ref, "ratio_inc": r_inc, "tf_med": median([self.flop / (x * 1e9) for x in t]),
                           "mj_est_med": median([x for x in e_c if x]) if any(e_c) else None,
                           "energy_ratio_ref": median(r_energy) if r_energy else None,
                           "w_med": median([w for w in wt.get(names[k], []) if w]) if any(wt.get(names[k], [])) else None,
                           "ms_med": median(t), "ref_tf_med": median([self.flop / (x * 1e9) for x in t_ref])}})
            say(f"  #{self.n_gpu:<4} {median(r_ref):.4f} of ref  vs-best {median(r_inc):.4f} {verdict:<4} "
                f"{median([self.flop / (x * 1e9) for x in t]):7.1f} TF/s  [{e.get('origin', '')}] {label(e['over'])}")
        # the incumbent (a control) and the racing contenders re-measured: pool their samples
        pooled_arms = [("ctl_a", self.inc_key)] if self.inc_key != self.ref_key else []
        pooled_arms += [(f"race{i}", rk) for i, rk in enumerate(race)]
        for arm, pk in pooled_arms:
            ei = self.E[pk]
            ei["ratios"].extend(x / y for x, y in zip(ms[arm], t_ref))
            ei["tf"].extend(self.flop / (x * 1e9) for x in ms[arm])
            ei["batches"].append(bno)
        if race:
            self.rec(type="race", batch=bno, arms={
                label(self.E[rk]["over"]): {"batch": median([x / y for x, y in zip(ms[f"race{i}"], t_ref)]),
                                            "pooled": self.pooled(rk), "n": len(self.E[rk]["ratios"])}
                for i, rk in enumerate(race)})
        self.worker.call("unload", names=[names[k] for k in ok_keys])

    # ------------------------------------------------------------ checkpoint
    def checkpoint(self, strat) -> None:
        st = {
            "version": 1,
            "kernel_fingerprint": kernel_fingerprint(self.mod),
            "space_fingerprint": self.sp.space.fingerprint(),
            "strategy": strat.state_dict(),
            "E": {k: {kk: vv for kk, vv in e.items()} for k, e in self.E.items()},
            "isa_index": self.isa_index,
            "block": self.block,
            "ctl_logr": self.ctl_logr,
            "batch_no": self.batch_no,
            "n_gpu": self.n_gpu,
            "stall": self.stall,
            "best_hist": self.best_hist,
            "gate_mode": self.gate_mode,
            "inc_key": self.inc_key,
            "ref_key": self.ref_key,
            "elapsed": time.time() - self.t0,
        }
        tmp = self.out / "ckpt.json.tmp"
        tmp.write_text(json.dumps(st, default=_jd))
        os.replace(tmp, self.out / "ckpt.json")
        self.sync()

    def restore(self, path: Path, strat) -> bool:
        st = json.loads(path.read_text())
        if st.get("kernel_fingerprint") != kernel_fingerprint(self.mod) or st.get("space_fingerprint") != self.sp.space.fingerprint():
            say("[resume] kernel or space changed since the checkpoint: its results are used as warm starts only")
            return False
        strat.load_state_dict(st["strategy"])
        for k, e in st["E"].items():
            if k in self.E and k == self.ref_key:
                continue
            self.E[k] = e
        self.isa_index.update(st["isa_index"])
        self.block = st["block"]
        self.ctl_logr = st["ctl_logr"]
        self.batch_no, self.n_gpu, self.stall = st["batch_no"], st["n_gpu"], st["stall"]
        self.best_hist = st["best_hist"]
        self.inc_key = st["inc_key"] if st["inc_key"] in self.E else self.ref_key
        say(f"[resume] {len(self.E)} keys, {self.n_gpu} GPU evaluations, batch {self.batch_no}")
        return True

    # ------------------------------------------------------------ finalists
    def finalists(self, k_top: int) -> dict:
        """Top configs (pooled ratio) re-timed together with ref: peak rounds + sustained windows, SCLK / W."""
        a = self.args
        self.worker_restart("fresh context for the finalists")
        ranked = sorted((self.pooled(k), k) for k, e in self.E.items() if e.get("status") == STATUS_OK and not e.get("isa_dup"))
        top = [k for _, k in ranked[:k_top]]
        if self.ref_key not in top:
            top.append(self.ref_key)
        arms = {}
        for i, k in enumerate(top):
            nm = "ref" if k == self.ref_key else f"f{i}"
            if nm != "ref":
                self.worker.call("load", name=nm, hsaco=self.E[k]["meta"]["hsaco"], spec=self.full_spec(k),
                                 kernel=self.E[k]["meta"]["name"])
            arms[nm] = k
        self.ensure_loaded("ref", self.ref_key, self.ref_key)
        timed = list(arms)
        aiter_ok = False
        if a.aiter_co:
            try:
                self.worker.call("load_aiter", name="aiter", co=a.aiter_co, symbol=a.aiter_symbol)
                v = self.worker.call("verify", name="aiter", dists=self.dists(), launches=a.verify_launches, compare="ref")
                aiter_ok = bool(v["ok"])
                say(f"[final] aiter verify: {'PASS' if aiter_ok else 'FAIL ' + _verify_why(v)}")
            except RuntimeError as e:
                say(f"[final] aiter arm not loaded: {e}")
            if aiter_ok:
                timed.append("aiter")
        res = self.worker.call("time", arms={nm: a.final_rounds for nm in timed}, dist=a.time_dist, gap_ms=a.gap_ms,
                               warmup=1, iters=3, seed=a.seed * 31 + 7, hang_s=max(a.hang_s, 120), discard=a.discard_rounds)
        sus = self.worker.call("sustained", names=timed, dist=a.time_dist, seconds=a.sustain_s,
                               rounds=a.sustained_rounds, seed=a.seed, hang_s=max(a.hang_s, 120))["sustained"]
        t_ref = res["ms"]["ref"]
        rows = []
        for nm, k in arms.items():
            t = res["ms"][nm]
            sv = sus.get(nm, [])
            rows.append({
                "arm": nm, "key": k, "over": self.E[k]["over"], "label": label(self.E[k]["over"]),
                "peak_tf": median([self.flop / (x * 1e9) for x in t]), "ratio_ref": median([x / y for x, y in zip(t, t_ref)]),
                "peak_tf_min_max": [self.flop / (max(t) * 1e9), self.flop / (min(t) * 1e9)],
                "sustained_tf": median([s["tflops"] for s in sv]) if sv else None,
                "sustained_sclk": median([s["sclk_med"] for s in sv if s.get("sclk_med")]) if any(s.get("sclk_med") for s in sv) else None,
                "sustained_w": median([s["watts_med"] for s in sv if s.get("watts_med")]) if any(s.get("watts_med") for s in sv) else None,
                "sustained_mj": median([s["mj"] for s in sv if s.get("mj")]) if any(s.get("mj") for s in sv) else None,
                "pooled_search_ratio": self.pooled(k), "search_batches": len(self.E[k].get("batches", [])),
                "compile": _meta_summary(self.E[k].get("meta") or {}),
            })
        # warm (steady-state) is the headline: rank by it when the windows ran, else by the cold-start ratio
        rows.sort(key=lambda r: (-(r["sustained_tf"] or 0.0), r["ratio_ref"]))
        out = {"rows": rows, "clocks": res.get("clocks"), "rounds": a.final_rounds}
        if aiter_ok:
            t = res["ms"]["aiter"]
            sv = sus.get("aiter", [])
            best_arm = next(nm for nm, k in arms.items() if k == rows[0]["key"])
            out["aiter"] = {"peak_tf": median([self.flop / (x * 1e9) for x in t]),
                            "sustained_tf": median([s["tflops"] for s in sv]) if sv else None,
                            "best_vs_aiter_peak": bootstrap_ci(t, res["ms"][best_arm]),
                            "best_vs_aiter_sustained": bootstrap_ci([s["ms"] for s in sv], [s["ms"] for s in sus.get(best_arm, [])])}
        return out


def noop_prepass(run: "Run") -> List[str]:
    """Knobs that are no-ops for this run: every level compiles to the base's ISA digest, on the base AND under
    every legal single change of every other knob (a knob gated by another -- e.g. ``b_slot`` read only when
    ``b_direct`` is set -- changes the ISA under its gate's change and stays searchable). Fresh compile processes,
    parallel; every solo / pair digest is recorded."""
    sp, base = run.sp, dict(run.ref_over)
    ref_isa = run.E[run.ref_key].get("isa")

    def legal_over(over):
        try:
            return sp.legal(sp.spec_of(over))[0]
        except TypeError:
            return False

    singles = {}
    for knob, levels in sp.knobs.items():
        for v in levels:
            if sp.base_value(knob) != v and legal_over({knob: v}):
                singles[(knob, v)] = {**base, knob: v}

    def compile_all(cfgs):
        items = {_sha(json.dumps(o, sort_keys=True, default=_jd) + "#pre"): o for o in cfgs.values()}
        res = run.pool.compile_many({t: run.compile_item(o) for t, o in items.items()}) if items else {}
        return {key: res[_sha(json.dumps(o, sort_keys=True, default=_jd) + "#pre")] for key, o in cfgs.items()}

    sres = compile_all(singles)
    solo_noop: Dict[str, bool] = {}
    for (knob, v), meta in sres.items():
        same = bool(meta.get("ok")) and meta.get("isa") == ref_isa
        solo_noop[knob] = solo_noop.get(knob, True) and same
        run.rec(type="solo_isa", knob=knob, value=v, same_isa=same, ok=meta.get("ok"), loop=(meta.get("loop") or {}),
                notes=meta.get("notes"))
    cands = sorted(k for k, flag in solo_noop.items() if flag)
    if not cands:
        return []
    pairs = {}
    for (m, mv), mover in singles.items():
        if m in cands or not sres[(m, mv)].get("ok"):
            continue
        for (n, nv) in singles:
            if n in cands and legal_over({m: mv, n: nv}):
                pairs[(m, mv, n, nv)] = {**mover, n: nv}
    pres = compile_all(pairs)
    gated: Dict[str, List[str]] = {}
    for (m, mv, n, nv), meta in pres.items():
        if not meta.get("ok") or meta.get("isa") != sres[(m, mv)].get("isa"):
            gated.setdefault(n, []).append(f"{m}={mv}")
    run.rec(type="noop_prepass", candidates=cands, gated={k: sorted(set(v)) for k, v in gated.items()}, n_pairs=len(pairs))
    if gated:
        say("[noop] gated (no-op on the base, live under another knob): "
            + "; ".join(f"{k} <- {sorted(set(v))[:4]}" for k, v in sorted(gated.items())))
    return [k for k in cands if k not in gated]


def _meta_summary(meta: Mapping) -> dict:
    return {k: meta.get(k) for k in ("ok", "name", "compile_s", "notes", "loop", "isa", "error") if k in meta}


def _verify_why(v: Mapping) -> str:
    parts = []
    for d, r in (v.get("dists") or {}).items():
        if not (r.get("all_pass") and r.get("deterministic")):
            parts.append(f"{d}: bad={r.get('bad')} det={r.get('deterministic')} worst={r.get('worst_err_over_tol')}")
    return "; ".join(parts) or "verify failed"


def _warm_from(path: str, k: int) -> List[Tuple[dict, str]]:
    p = Path(path)
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    out = []
    rows = (d.get("finalists") or {}).get("rows") or d.get("top") or []
    for r in rows[:k]:
        if r.get("over") is not None:
            out.append((dict(r["over"]), f"prev:{r.get('label', '')}"[:60]))
    return out


def cmd_run(args) -> int:
    t_start = time.time()
    mod = kernel_module()
    if args.warm and os.path.exists(args.warm):
        warm_over = json.loads(Path(args.warm).read_text())
    else:
        warm_over = json.loads(args.warm) if args.warm else {}
    if args.preset:
        warm_over = {**dict(getattr(mod, "PRESETS")[args.preset]), **warm_over}
        say(f"[warm] preset {args.preset}: {label(warm_over)}")
        plev = (getattr(mod, "PRESET_LEVERS", None) or {}).get(args.preset)
        if plev and args.preset_levers:
            args.rust_levers = plev.replace(",", "+")
            say(f"[warm] preset levers {args.preset}: {plev}")
    best_attr = getattr(mod, "BEST", None)
    if not warm_over and isinstance(best_attr, Mapping):
        warm_over = dict(best_attr)
        say(f"[warm] kernel module BEST: {label(warm_over)}")
    extra = json.loads(args.extra_knobs) if args.extra_knobs else None
    backends = [b for b in re.split(r"[,:+]", args.backends) if b]  # ':' / '+' survive sbatch --export
    if args.rust_compiler:
        os.environ["MXFP6_RUST_COMPILER"] = args.rust_compiler
    os.environ.setdefault("MXFP6_RUST_LEVERS", args.rust_levers)
    lever_sets = [x for x in args.rust_lever_sets.split(";") if x]
    pseudo = {}
    if len(backends) > 1 or backends[0] != "llvm":
        pseudo["backend"] = tuple(backends)
    if "rust" in backends and len(lever_sets) > 1:
        pseudo["rust_levers"] = tuple(lever_sets)
    base_levers = set(args.rust_levers.replace("+", ",").split(","))
    for lv in [x for x in re.split(r"[+,:]", args.rust_lever_axes) if x] if "rust" in backends else []:
        pseudo[LEVER_PREFIX + lv] = (True, False) if lv in base_levers else (False, True)
    pseudo = pseudo or None
    if "rust" in backends and not (os.environ.get("MXFP6_RUST_COMPILER") and os.path.exists(os.environ["MXFP6_RUST_COMPILER"])) \
            and not _fake():
        raise SystemExit("--backends includes rust but --rust-compiler / MXFP6_RUST_COMPILER is not an executable")
    sp = Mxfp6Space(mod, base=warm_over, exclude=[x for x in re.split(r"[,:+]", args.exclude) if x], extra=extra,
                    shape=(args.m, args.n, args.k), pseudo=pseudo)
    run = Run(args, sp)
    desc = sp.describe()
    (run.out / "space.json").write_text(json.dumps(desc, indent=1, default=_jd))
    say(f"[space] {len(sp.knobs)} knobs, size bound {desc['size_bound']:.3g}, kernel {desc['kernel_fingerprint']}, "
        f"base {label(desc['base'])}; held: {sorted(k for k in sp.fixed if k not in ('m', 'n', 'k'))}")
    dists = sorted(set(run.dists()) | {args.time_dist})
    ensure_data(Path(args.cache), args.m, args.n, args.k, dists, args.seed_data, args.gen_workers)
    run.worker_start()
    run.session_gate()
    if args.noop_prepass:
        noop = noop_prepass(run)
        if noop:
            say(f"[noop] held at the base (every level compiles to the base ISA): {sorted(noop)}")
            sp = Mxfp6Space(mod, base=warm_over, exclude=[x for x in re.split(r"[,:+]", args.exclude) if x] + sorted(noop),
                            extra=extra, shape=(args.m, args.n, args.k), pseudo=pseudo)
            run.rebind_space(sp)
            (run.out / "space.json").write_text(json.dumps(sp.describe(), indent=1, default=_jd))
    warm = [(sp.ref_over if hasattr(sp, "ref_over") else run.ref_over, "kernel-agent")]
    if args.warm_from:
        warm += _warm_from(args.warm_from, args.top)
    if "rust" in backends:
        for over in RUST_RETEST:
            cfg = {**run.ref_over, **over, "backend": "rust"}
            if all(k in sp.defaults or is_pseudo(k) for k in cfg) and sp.legal(sp.spec_of(spec_over(cfg)))[0]:
                warm.append((cfg, "rust-retest:" + (label(over) if over else "base")))
    strat = build_strategy(sp, args, warm)
    if args.resume and os.path.exists(args.resume):
        if not run.restore(Path(args.resume), strat):
            strat = build_strategy(sp, args, warm + _warm_from(str(Path(args.resume).with_name("summary.json")), args.top))
    deadline = t_start + args.minutes * 60.0 - args.reserve_s
    asks_cap = args.budget * 6
    stop = "?"
    while True:
        if time.time() > deadline:
            stop = "clock"
            break
        if run.n_gpu >= args.budget:
            stop = "budget"
            break
        if run.stall >= args.stall_batches and run.n_gpu >= args.min_evals:
            stop = f"stall ({run.stall} batches without a gain beyond {run.min_gain():.4f})"
            break
        if run.n_told >= asks_cap:
            stop = "ask cap"
            break
        if run.idle_batches >= args.idle_batches:
            stop = f"converged ({run.idle_batches} batches proposed only measured or ISA-identical configs)"
            break
        try:
            more = run.run_batch(strat)
        except GiveUp as e:
            stop = f"gave up: {e}"
            break
        if not more:
            stop = "strategy exhausted" if strat.done else "no proposals"
            break
    say(f"[stop] {stop}: {run.n_gpu} GPU evaluations, {run.batch_no} batches, {time.time() - t_start:.0f} s")
    fin = None
    try:
        if stop.startswith("gave up"):
            raise RuntimeError("no finalists after giving up")
        fin = run.finalists(args.top)
        for r in fin["rows"]:
            say(f"[final] warm {r['sustained_tf'] or float('nan'):7.1f} TF/s @ {r['sustained_sclk']} MHz {r['sustained_w']} W "
                f"{r.get('sustained_mj')} mJ  cold-start {r['peak_tf']:7.1f} ({r['ratio_ref']:.4f} of ref)  {r['label']}")
    except (WorkerFault, RuntimeError) as e:
        say(f"[final] finalists failed: {e}")
    summarize(run, strat, stop, fin)
    run.worker.close()
    run.sync()
    return 0


def summarize(run: Run, strat, stop: str, fin: Optional[dict]) -> dict:
    recs = [json.loads(x) for x in (run.out / "evals.jsonl").read_text().splitlines() if x.strip()]
    thr = run.min_gain()
    led = build_ledger(recs, run.defaults, thr)
    (run.out / "ledger.txt").write_text(ledger_text(led))
    ranked = sorted((run.pooled(k), k) for k, e in run.E.items() if e.get("status") == STATUS_OK and not e.get("isa_dup"))
    ref_tf = median(run.ref_tf) if run.ref_tf else None
    top = [{"key": k, "over": run.E[k]["over"], "label": label(run.E[k]["over"]), "pooled_ratio": v,
            "tf_est": (ref_tf / v) if ref_tf and v and math.isfinite(v) else None, "batches": len(run.E[k].get("batches", [])),
            "compile": _meta_summary(run.E[k].get("meta") or {})} for v, k in ranked[: max(10, run.args.top)]]
    status_counts: Dict[str, int] = {}
    for e in run.E.values():
        status_counts[e.get("status") or "?"] = status_counts.get(e.get("status") or "?", 0) + 1
    rep = {}
    try:
        rep["allocation"] = strat.allocation()
    except Exception:  # noqa: BLE001
        pass
    for m in getattr(strat, "members", []):
        if hasattr(m, "report"):
            try:
                rep[m.name] = m.report()
            except Exception as e:  # noqa: BLE001
                rep[m.name] = repr(e)
    summ = {
        "stop": stop, "n_gpu": run.n_gpu, "batches": run.batch_no, "keys": len(run.E), "status_counts": status_counts,
        "gate_mode": run.gate_mode, "noise_sigma": noise_sigma(run.ctl_logr), "min_gain": thr, "ref_over": run.ref_over,
        "ref_tf_median": ref_tf, "top": top, "finalists": fin, "ledger": led, "strategy": rep,
        "space": run.sp.describe(), "rejects": run.sp.rejects, "gpu": getattr(run, "gpu_info", None),
        "env": {k: os.environ.get(k) for k in ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "SLURM_JOB_ID",
                                                "SLURMD_NODENAME", "OBJDUMP")},
        "best_hist": run.best_hist, "wall_s": time.time() - run.t0,
    }
    (run.out / "summary.json").write_text(json.dumps(summ, indent=1, default=_jd))
    say(f"[summary] status {status_counts}; noise sigma {summ['noise_sigma']:.4f}; top: "
        + "; ".join(f"{t['pooled_ratio']:.4f} {t['label']}" for t in top[:5]))
    return summ


# ===================================================================== confirm


def legal_probe(mod, best: Mapping[str, object], probe: str) -> Tuple[Optional[dict], str]:
    """``(overrides, note)`` of the probe build nearest the best config: the best config itself when the probe is
    legal there, else the first single knob change (``KNOBS`` order) that makes it legal (e.g. a probe that needs
    VGPR scales on an LDS-scale body)."""
    over = {**dict(best), "probe": probe}

    def spec_of(o):
        return replace(mod.Mxfp6GemmSpec(), **spec_over(o))

    ok, why = legal(mod, spec_of(over))
    if ok:
        return over, ""
    for knob, levels in (getattr(mod, "KNOBS", None) or {}).items():
        for v in levels:
            if over.get(knob, default_config(mod).get(knob)) == v:
                continue
            o2 = {**over, knob: v}
            if legal(mod, spec_of(o2))[0]:
                return o2, f"not legal on the best config ({why}); timed with {knob}={v}"
    return None, why


def gpu_guard(info: Mapping) -> None:
    """Refuse the known-bad GPU (n07-17, bus 0x05: ECC storm)."""
    host = os.environ.get("SLURMD_NODENAME", "") or (os.uname().nodename if hasattr(os, "uname") else "")
    bdf = str((info or {}).get("bdf", "")).lower()
    if "n07-17" in host and bdf.startswith("0000:05:"):
        raise SystemExit(f"refusing GPU {bdf} on {host} (n07-17 bus 0x05)")
    # MXFP6_AVOID_GPUS="n14-05@0000:85+...": boards excluded from timing (e.g. a module-placement offset of the twins)
    for item in [x for x in re.split(r"[+;]", os.environ.get("MXFP6_AVOID_GPUS", "")) if x]:
        node, _, bus = item.partition("@")
        if node in host and bdf.startswith(bus.lower()):
            raise SystemExit(f"refusing GPU {bdf} on {host} (MXFP6_AVOID_GPUS)")


#: Paired comparisons of a confirmation: ``name: (numerator arm, denominator arm)``; the ratio is
#: ``t_num / t_den`` per round, i.e. the TF/s of the denominator arm over the numerator's (> 1: den faster).
CONFIRM_PAIRS = {
    "best_vs_aiter": ("aiter", "best"),
    "best_vs_ref": ("ref", "best"),
    "noise_best_twin": ("best_tw", "best"),
    "noise_aiter_twin": ("aiter_tw", "aiter"),
    "best_of_mfma_ceiling": ("probe_mfma_only", "best"),
    "best_vs_fill_floor": ("probe_fill_only", "best"),
}


def confirm_pairs(ms: Mapping[str, Sequence[float]], seed: int = 0) -> Dict[str, dict]:
    out = {}
    for name, (num, den) in CONFIRM_PAIRS.items():
        if num in ms and den in ms and ms[num] and ms[den]:
            out[name] = bootstrap_ci(ms[num], ms[den], seed=seed)
    return out


def cmd_confirm(args) -> int:
    """Re-time a run's top configs on one GPU: every arm (rocKE configs, a twin of the best, aiter's asm
    code object + its twin, the kernel's MFMA-ceiling / fill-floor probes) verified, then timed in the same
    shuffled rounds (peak protocol) and the same sustained windows; paired bootstrap CIs per comparison."""
    mod = kernel_module()
    summ = json.loads(Path(args.summary).read_text()) if args.summary else {}
    rows = ((summ.get("finalists") or {}).get("rows") or summ.get("top") or [])
    def _json_or_file(x):
        return json.loads(Path(x).read_text()) if os.path.exists(x) else json.loads(x)

    cfgs: Dict[str, dict] = {"ref": dict(_json_or_file(args.ref) if args.ref else (summ.get("ref_over") or {}))}
    seen = {json.dumps(cfgs["ref"], sort_keys=True, default=_jd)}
    ranked = [dict(r["over"]) for r in rows if r.get("over") is not None]
    if args.best:
        ranked.insert(0, _json_or_file(args.best))
    for over in ranked:
        s = json.dumps(over, sort_keys=True, default=_jd)
        if s in seen:
            continue
        seen.add(s)
        cfgs["best" if "best" not in cfgs else f"top{len(cfgs) - 1}"] = over
        if len(cfgs) > args.top:
            break
    if "best" not in cfgs:
        cfgs["best"] = dict(cfgs["ref"])
    extra_arms = {}
    if args.extra_arms:
        extra_arms = json.loads(Path(args.extra_arms).read_text()) if os.path.exists(args.extra_arms) else json.loads(args.extra_arms)
    for k, v in extra_arms.items():
        cfgs[k] = v
    probes = {}
    if "probe" in default_config(mod):
        for pr in getattr(mod, "PROBES", ("mfma_only", "fill_only")):
            if pr == "none":
                continue
            over, why = legal_probe(mod, cfgs["best"], pr)
            if over is not None:
                probes[f"probe_{pr}"] = over
                if why:
                    say(f"[confirm] probe {pr}: {why}")
            else:
                say(f"[confirm] probe {pr} not legal on the best config: {why}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = CompilePool(out / "work", args.jobs)
    allc = {**cfgs, **probes}
    if args.rust_compiler:
        os.environ["MXFP6_RUST_COMPILER"] = args.rust_compiler
    full = {nm: asdict(replace(mod.Mxfp6GemmSpec(), **{**spec_over(ov), "m": args.m, "n": args.n, "k": args.k}))
            for nm, ov in allc.items()}
    metas = pool.compile_many({nm: {**full[nm], "__backend__": allc[nm].get("backend", "llvm"),
                                    "__levers__": levers_of(allc[nm], RUST_LEVERS_DEFAULT)} for nm in full})
    dists = sorted(set(d for d in args.verify_dists.split(",") if d) | {args.time_dist})
    ensure_data(Path(args.cache), args.m, args.n, args.k, dists, args.seed_data, args.gen_workers)
    w = WorkerClient(out / "worker.log", hang_s=args.hang_s)
    w.start()
    info = w.call("init", hang_s=900, cache=args.cache, m=args.m, n=args.n, k=args.k, seed=args.seed_data, dists=dists)
    gpu_guard(info)
    say(f"[confirm] gpu {info}")
    host = os.environ.get("SLURMD_NODENAME", "")
    report = {"gpu": info, "host": host, "gpu_tag": args.gpu_tag, "arms": {}, "blocks": [], "between": [],
              "kernel_fingerprint": kernel_fingerprint(mod), "protocol": {
                  "rounds_per_block": args.rounds, "blocks": args.blocks, "gap_ms": args.gap_ms, "warmup": 1, "iters": 3,
                  "sustained_rounds": args.sustained_rounds, "sustain_s": args.sustain_s, "time_dist": args.time_dist},
              "env": {k: os.environ.get(k) for k in ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "SLURM_JOB_ID", "SLURMD_NODENAME",
                                                      "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}}
    timed: List[str] = []

    def verify(nm: str, keep: bool = False) -> bool:
        v = w.call("verify", name=nm, dists=[d for d in args.verify_dists.split(",") if d], launches=args.verify_launches,
                   keep=keep, compare="" if keep else "ref", ulp=True, keep_out=True, hang_s=max(args.hang_s, 180))
        report["arms"][nm]["verify"] = v
        say(f"[confirm] verify {nm}: {'PASS' if v['ok'] else 'FAIL'} {_verify_why(v) if not v['ok'] else ''}")
        return bool(v["ok"])

    order = ["ref"] + [n for n in allc if n != "ref"]
    for nm in order:
        meta = metas[nm]
        report["arms"][nm] = {"over": allc[nm], "label": label(allc[nm]), "compile": _meta_summary(meta)}
        if not meta.get("ok"):
            say(f"[confirm] {nm}: compile failed: {str(meta.get('error'))[:300]}")
            continue
        w.call("load", name=nm, hsaco=meta["hsaco"], spec=full[nm], kernel=meta["name"])
        if nm.startswith("probe_"):
            timed.append(nm)  # a measurement probe, not a GEMM: timed, never verified
            continue
        if verify(nm, keep=(nm == "ref")):
            timed.append(nm)
        if nm == "best" and nm in timed:
            w.call("load", name="best_tw", hsaco=meta["hsaco"], spec=full[nm], kernel=meta["name"])
            report["arms"]["best_tw"] = {"over": allc[nm], "label": "twin of best (second module load)"}
            timed.append("best_tw")
    if args.aiter_co:
        r = w.call("load_aiter", name="aiter", co=args.aiter_co, symbol=args.aiter_symbol)
        report["arms"]["aiter"] = {"over": {}, "label": f"aiter {args.aiter_symbol}", "co": args.aiter_co, "md5": r.get("md5")}
        if verify("aiter"):
            timed.append("aiter")
            w.call("load_aiter", name="aiter_tw", co=args.aiter_co, symbol=args.aiter_symbol)
            report["arms"]["aiter_tw"] = {"over": {}, "label": "twin of aiter (second module load)"}
            timed.append("aiter_tw")
    # numeric tightness: every rocKE GEMM arm vs aiter (and vs ref) bitwise on identical inputs, per dist
    report["compare"] = {}
    gemm_arms = [nm for nm in timed if not nm.startswith("probe_") and not nm.endswith("_tw")]
    for nm in gemm_arms:
        for other in ("aiter", "ref"):
            if other == nm or other not in gemm_arms:
                continue
            for d in [x for x in args.verify_dists.split(",") if x]:
                report["compare"][f"{nm}|{other}|{d}"] = w.call("compare", a=nm, b=other, dist=d)
    flop = 2.0 * args.m * args.n * args.k
    for b in range(args.blocks):
        res = w.call("time", arms={nm: args.rounds for nm in timed}, dist=args.time_dist, gap_ms=args.gap_ms, warmup=1,
                     iters=3, seed=1000 * (args.seed + 1) + b, hang_s=max(args.hang_s, 120), discard=args.discard_rounds)
        report["blocks"].append(res)
        say(f"[confirm] block {b}: " + "  ".join(f"{nm}={median([flop / (x * 1e9) for x in res['ms'][nm]]):.0f}" for nm in timed))
        if args.between and (b < args.blocks - 1 or args.between_after_last):
            cmd = args.between.replace("{block}", str(b))
            t0 = time.time()
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=args.between_timeout)
            report["between"].append({"block": b, "cmd": cmd, "rc": p.returncode, "s": round(time.time() - t0, 1),
                                      "tail": (p.stdout or "")[-2000:] + (p.stderr or "")[-1000:]})
    sus = w.call("sustained", names=timed, dist=args.time_dist, seconds=args.sustain_s, rounds=args.sustained_rounds,
                 seed=7 + args.seed, hang_s=max(args.hang_s, 120))["sustained"]
    if args.coldstart_reps > 0:
        report["coldstart"] = coldstart_diag(w, pool, args, [nm for nm in ("best", "aiter", "probe_mfma_only",
                                                                            "probe_fill_only") if nm in timed])
    w.close()
    peak_ms = {nm: [x for blk in report["blocks"] for x in blk["ms"][nm]] for nm in timed}
    sus_ms = {nm: [s["ms"] for s in sus.get(nm, [])] for nm in timed}
    report["sustained"] = sus
    report["peak_pairs"] = confirm_pairs(peak_ms, seed=args.seed)
    report["sustained_pairs"] = confirm_pairs(sus_ms, seed=args.seed + 1)
    report["cold_pairs"], report["warm_pairs"] = report["peak_pairs"], report["sustained_pairs"]  # new names
    rows_out = []
    for nm in timed:
        sv = sus.get(nm, [])
        rows_out.append({
            "arm": nm, "label": report["arms"][nm]["label"],
            "peak_tf": median([flop / (x * 1e9) for x in peak_ms[nm]]),
            "peak_tf_per_block": [median([flop / (x * 1e9) for x in blk["ms"][nm]]) for blk in report["blocks"]],
            "peak_cv_pct": 100.0 * float(np.std(peak_ms[nm]) / np.mean(peak_ms[nm])) if peak_ms[nm] else None,
            "sustained_tf": median([s["tflops"] for s in sv]) if sv else None,
            "sustained_sclk": median([s["sclk_med"] for s in sv if s.get("sclk_med")]) if any(s.get("sclk_med") for s in sv) else None,
            "sustained_w": median([s["watts_med"] for s in sv if s.get("watts_med")]) if any(s.get("watts_med") for s in sv) else None,
            "sustained_mj": median([s["mj"] for s in sv if s.get("mj")]) if any(s.get("mj") for s in sv) else None,
        })
    for r in rows_out:  # cold-start (post-idle) = old "peak"; warm (steady-state, the headline) = old "sustained"
        r["cold_tf"], r["warm_tf"], r["warm_mj"] = r["peak_tf"], r["sustained_tf"], r.get("sustained_mj")
    report["rows"] = rows_out
    (out / "confirm.json").write_text(json.dumps(report, indent=1, default=_jd))
    print(confirm_text([report]))
    return 0


def coldstart_diag(w: "WorkerClient", pool: CompilePool, args, arms: List[str]) -> dict:
    """Why cold-start (post-idle) runs below warm: per-launch times of the first launches after the idle gap vs
    steady state for the GEMM / aiter / MFMA-only / fill-only arms, and the in-kernel effective shader clock of a
    compute-only probe over the same pattern (a clock / voltage ramp shows in the probe's MHz; a fabric / memory
    clock ramp shows only in the memory-bound arms)."""
    meta = pool.compile_many({"clock_probe": {"__kind__": "clock_probe", "iters": args.clock_probe_iters}})["clock_probe"]
    names = list(arms)
    if meta.get("ok"):
        w.call("load_clock_probe", name="clock_probe", hsaco=meta["hsaco"])
        names.append("clock_probe")
    else:
        say(f"[cold] clock probe not built: {str(meta.get('error'))[:300]}")
    res = w.call("coldstart", arms=names, dist=args.time_dist, gap_ms=args.gap_ms, launches=args.coldstart_launches,
                 steady=args.coldstart_steady, reps=args.coldstart_reps, seed=11 + args.seed, hang_s=max(args.hang_s, 180))
    L = args.coldstart_launches
    summ = {}
    for a, d in res["coldstart"].items():
        mat = np.asarray(d["ms"], dtype=np.float64)  # reps x launches
        steady = float(np.median(mat[:, L:])) if mat.shape[1] > L else float(np.median(mat))
        row = {"steady_ms": steady, "launch_ratio": [float(np.median(mat[:, i])) / steady for i in range(L)]}
        if d.get("mhz"):
            mh = np.asarray([[np.nan if v is None else v for v in r] for r in d["mhz"]], dtype=np.float64)
            k = mh.shape[1]
            row["mhz_per_launch"] = [float(np.nanmedian(mh[:, i])) for i in range(min(L, k))]
            row["mhz_steady"] = float(np.nanmedian(mh[:, L:])) if k > L else None
        summ[a] = row
        say(f"[cold] {a:<16} launch 1..{L} / steady: " + " ".join(f"{x:.3f}" for x in row["launch_ratio"])
            + (f"  | MHz {[round(x) for x in row.get('mhz_per_launch', [])]} steady {row.get('mhz_steady')}"
               if "mhz_per_launch" in row else ""))
    return {"raw": res, "summary": summ, "clock_probe": _meta_summary(meta)}


def _ci(c: Optional[Mapping]) -> str:
    if not c or c.get("median") is None:
        return "n/a"
    return f"{c['median']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] n={c['n']}"


def confirm_text(reports: Sequence[Mapping]) -> str:
    """Per-GPU table: TF/s per arm and the paired ratios with bootstrap 95 % CIs (twins = noise floor)."""
    out = []
    for rep in reports:
        g = rep.get("gpu") or {}
        out.append(f"== GPU {g.get('bdf')} on {rep.get('host')} {rep.get('gpu_tag') or ''}")
        for r in sorted(rep["rows"], key=lambda r: -(r["sustained_tf"] or 0)):
            cv = r["peak_cv_pct"] if r["peak_cv_pct"] is not None else float("nan")
            sus = r["sustained_tf"] if r["sustained_tf"] is not None else float("nan")
            mj = r.get("sustained_mj")
            out.append(f"  {r['arm']:<18} warm {sus:7.1f} @ {r['sustained_sclk']} MHz {r['sustained_w']} W "
                       f"{mj if mj is None else round(mj)} mJ/GEMM  cold-start {r['peak_tf']:7.1f} (cv {cv:.1f} %)  {r['label']}")
        for kind, nm in (("sustained_pairs", "warm"), ("peak_pairs", "cold-start")):
            for name, c in (rep.get(kind) or {}).items():
                out.append(f"  {nm:<10} {name:<22} {_ci(c)}")
        cs = (rep.get("coldstart") or {}).get("summary") or {}
        for a, row in cs.items():
            out.append(f"  cold-start diag {a:<16} launch/steady " + " ".join(f"{x:.3f}" for x in row["launch_ratio"])
                       + (f" | MHz per launch {[round(x) for x in row['mhz_per_launch']]} steady "
                          f"{round(row['mhz_steady']) if row.get('mhz_steady') else None}" if row.get("mhz_per_launch") else ""))
        out.append("  numerics (full output): ulps vs RNE(bf16(exact fp64)); tolerance 2^-11 sum|t| + 2^-8 |ref|; int bitwise")
        for nm, arm in rep.get("arms", {}).items():
            for d, r in ((arm.get("verify") or {}).get("dists") or {}).items():
                u = r.get("ulp") or {}
                if not u:
                    continue
                out.append(f"    {nm:<10} {d:<8} pass={r.get('all_pass')} bad={r.get('bad')} worst/tol="
                           f"{(r.get('worst_err_over_tol') or 0):.3f} exact {100 * u['exact_frac']:.3f} %  <=1ulp "
                           f"{100 * u['le1_frac']:.4f} %  max {u['max_ulp']} ulp")
        for key, c in (rep.get("compare") or {}).items():
            a, b, d = key.split("|")
            out.append(f"    bitwise {a} vs {b} ({d}): {'IDENTICAL' if c.get('identical') else str(c.get('differ')) + ' differ'}"
                       + (f", max {c.get('max_ulp')} ulp, <=1ulp {100 * c.get('le1_frac', 0):.4f} %" if not c.get("identical") else ""))
    return "\n".join(out)


def cmd_report(args) -> int:
    reps = [json.loads(Path(p).read_text()) for p in args.confirm]
    print(confirm_text(reps))
    return 0


# ===================================================================== offline ledger / space


def cmd_ledger(args) -> int:
    mod = kernel_module()
    recs = [json.loads(x) for x in Path(args.jsonl).read_text().splitlines() if x.strip()]
    led = build_ledger(recs, default_config(mod), args.thr)
    print(ledger_text(led, top=args.top))
    return 0


def cmd_space(args) -> int:
    sp = Mxfp6Space(kernel_module(), base=json.loads(args.warm) if args.warm else None)
    d = sp.describe()
    print(json.dumps({k: d[k] for k in ("kernel_fingerprint", "space_fingerprint", "base", "knobs", "fixed", "size_bound")},
                     indent=1, default=_jd))
    rng = random.Random(0)
    n = 4000
    ok = sum(sp.space.valid(sp.space.random_genome(rng)) for _ in range(n))
    print(f"legal fraction of uniform genomes: {ok / n:.3f}; reject reasons: {json.dumps(sp.rejects, indent=1)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("space")
    s.add_argument("--warm", default="")
    c = sub.add_parser("compile")
    c.add_argument("--spec", required=True)
    c.add_argument("--out", required=True)
    sub.add_parser("worker")
    for name in ("run", "confirm"):
        r = sub.add_parser(name)
        r.add_argument("--out", required=True)
        r.add_argument("--cache", default="mxfp6_cache")
        r.add_argument("--m", type=int, default=4096)
        r.add_argument("--n", type=int, default=4096)
        r.add_argument("--k", type=int, default=65536)
        r.add_argument("--seed-data", type=int, default=1, help="canonical data seed (bench / C++ generator)")
        r.add_argument("--verify-dists", default="int,gauss")
        r.add_argument("--verify-launches", type=int, default=2)
        r.add_argument("--time-dist", default="gauss")
        r.add_argument("--gap-ms", type=float, default=25.0)
        r.add_argument("--rounds", type=int, default=32)
        r.add_argument("--discard-rounds", type=int, default=1, help="leading rounds run and thrown away")
        r.add_argument("--jobs", type=int, default=16)
        r.add_argument("--gen-workers", type=int, default=os.cpu_count() or 1)
        r.add_argument("--hang-s", type=float, default=60.0)
        r.add_argument("--sustain-s", type=float, default=3.0)
        r.add_argument("--sustained-rounds", type=int, default=2)
        r.add_argument("--top", type=int, default=5)
    r = sub.choices["run"]
    r.add_argument("--mirror", default="", help="results dir copied at every checkpoint (shared home)")
    r.add_argument("--warm", default="", help="JSON overrides (or a file) of the warm start; default: kernel BEST or defaults")
    r.add_argument("--warm-from", default="", help="a previous summary.json: its top configs are warm starts")
    r.add_argument("--resume", default="", help="a previous ckpt.json (same kernel + space: full resume)")
    r.add_argument("--exclude", default="", help="knobs held at the base (comma list)")
    r.add_argument("--extra-knobs", default="", help='JSON {field: [levels]} added to the space')
    r.add_argument("--members", default="lookback,hill,gp,de,hyperband")
    r.add_argument("--backends", default="llvm", help="compiler backends searched as a knob: llvm | llvm:rust")
    r.add_argument("--rust-compiler", default="", help="the direct compiler's target_compile binary")
    r.add_argument("--rust-levers", default=RUST_LEVERS_DEFAULT)
    r.add_argument("--rust-lever-axes", default="",
                   help="'+'-separated Rust levers, each searched as an on / off axis (base = in --rust-levers)")
    r.add_argument("--preset", default="", help="warm start = the kernel module's PRESETS[name]")
    r.add_argument("--preset-levers", type=int, default=1, help="1: base Rust levers = PRESET_LEVERS[--preset] when listed")
    r.add_argument("--rust-lever-sets", default="",
                   help="';'-separated Rust lever sets searched as an axis ('+' or ',' inside a set; first = base)")
    r.add_argument("--aiter-co", default="", help="aiter asm code object: timed with the finalists (same rounds)")
    r.add_argument("--aiter-symbol", default=AITER_DEFAULT_SYMBOL)
    r.add_argument("--lookback-module", default="", help="path of the vendored signed-lookback strategies.py")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--budget", type=int, default=400, help="GPU evaluations (verified + timed configs)")
    r.add_argument("--minutes", type=float, default=50.0)
    r.add_argument("--reserve-s", type=float, default=240.0, help="seconds kept for finalists + summary")
    r.add_argument("--stall-batches", type=int, default=15)
    r.add_argument("--min-evals", type=int, default=150)
    r.add_argument("--race", type=int, default=3, help="top contenders re-timed every batch (pooled)")
    r.add_argument("--objective", choices=("time", "energy"), default="time",
                   help="value told to the strategies: t / t_ref, or board energy per GEMM / ref's (window-attributed "
                        "power x time; both are recorded either way)")
    r.add_argument("--idle-batches", type=int, default=25, help="stop after this many batches with no new GPU evaluation")
    r.add_argument("--noop-prepass", type=int, default=1,
                   help="1: compile every single-knob change of the base first; knobs whose every level has the base's ISA "
                        "are held at the base for this run (reported; re-checked every run)")
    r.add_argument("--search-warmup", type=int, default=1, help="search batches: warmup launches after the idle gap")
    r.add_argument("--search-iters", type=int, default=3,
                   help="search batches: timed launches per sample (3 = peak protocol; finalists always 1 + 3)")
    r.add_argument("--gain-sigmas", type=float, default=1.0, help="noise floor = this many sigmas of a batch median")
    r.add_argument("--min-gain", type=float, default=0.003)
    r.add_argument("--screen-rounds", type=int, default=8, help="rounds of a Hyperband screen rung")
    r.add_argument("--final-rounds", type=int, default=40)
    r.add_argument("--gate", choices=("strict", "loop", "off"), default="strict")
    r.add_argument("--max-restarts", type=int, default=12, help="worker restarts (hangs / faults) before the run stops")
    q = sub.choices["confirm"]
    q.add_argument("--summary", default="", help="a run's summary.json (finalists / top configs)")
    q.add_argument("--best", default="", help="JSON overrides of the config to confirm as 'best' (instead of the summary's)")
    q.add_argument("--ref", default="", help="JSON overrides of the reference arm (default: the summary's warm start)")
    q.add_argument("--aiter-co", default="", help="aiter asm code object timed as an arm in the same rounds")
    q.add_argument("--aiter-symbol", default=AITER_DEFAULT_SYMBOL)
    q.add_argument("--rust-compiler", default="", help="target_compile binary (arms with backend=rust)")
    q.add_argument("--gpu-tag", default="")
    q.add_argument("--seed", type=int, default=1)
    q.add_argument("--blocks", type=int, default=3, help="blocks of --rounds cold-start (post-idle) rounds")
    q.add_argument("--coldstart-reps", type=int, default=5, help="cold-start diagnostic repetitions (0 = off)")
    q.add_argument("--coldstart-launches", type=int, default=10)
    q.add_argument("--coldstart-steady", type=int, default=20)
    q.add_argument("--clock-probe-iters", type=int, default=60000)
    q.set_defaults(verify_dists="int,gauss,uniform", rounds=20)
    q.add_argument("--between", default="", help="shell command run between timing blocks ({block} = index)")
    q.add_argument("--between-after-last", action="store_true")
    q.add_argument("--between-timeout", type=float, default=900.0)
    q.add_argument("--extra-arms", default="", help='JSON {name: overrides} timed as well')
    rp = sub.add_parser("report")
    rp.add_argument("confirm", nargs="+", help="confirm.json files (one per GPU)")
    lg = sub.add_parser("ledger")
    lg.add_argument("jsonl")
    lg.add_argument("--thr", type=float, default=0.004)
    lg.add_argument("--top", type=int, default=12)
    args = p.parse_args(argv)
    return {"space": cmd_space, "compile": cmd_compile, "worker": cmd_worker, "run": cmd_run, "confirm": cmd_confirm,
            "ledger": cmd_ledger, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
