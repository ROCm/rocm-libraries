# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify-then-time harness for the gfx950 MXFP6 x MXFP6 GEMM (numpy only, no torch).

Subcommands (``python -m benchmarks.gfx950.mxfp6_gemm.bench <cmd> -h``):

* ``gen``      -- write canonical operands (the C++ test generator's ``save``
  format) for one distribution, generated in parallel row shards;
* ``compare``  -- byte-compare two operand files (generator identity proof);
* ``compile``  -- build + compile one spec in this (fresh) process, write the
  code object and its resource notes (the driver runs one such process per arm,
  before it initializes the GPU, so no build inherits another's LLVM state);
* ``run``      -- load data, compile every arm (subprocess pool), verify every
  arm on the exact-integer set (bitwise) and the Gaussian / uniform-code sets
  (float64 reference, scaled-MFMA tolerance), require each must-fail mutant to
  fail, then time all arms interleaved: the *cold-start (post-idle)* protocol
  (idle gap, 1 warmup + 3 timed launches, median of N rounds, arm order
  alternating) and a *warm (steady-state)* window per arm with sysfs SCLK /
  power sampling (the headline metric). Writes JSON: ``cold_<dist>`` /
  ``warm_<dist>`` keys; the older names ``peak_<dist>`` / ``sustained_<dist>``
  are kept as aliases of the same data.

Absolute figures are for the operator's internal record only (see
``platform/AGENTS.md`` Compliance); repository documents carry ratios.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from kernels.gfx950.mxfp6_gemm import edge_cases as E
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm.gemm import (
    Mxfp6GemmSpec,
    build_mxfp6_gemm,
    check_compiler,
    descriptor,
    grid_of,
    kernel_name,
    operand_layout_of,
    validate_spec,
)

DISTS = {"gauss": L.DIST_GAUSS, "uniform": L.DIST_UNIFORM_CODES, "int": L.DIST_INT}
#: + the full-shape edge datasets of edge_cases.DATASETS (zero, maxconst, nan_all, nan_1pct, subnormal, sparse50,
#: extreme), verified with the OCP / IEEE convention (edge_cases.check_ieee)


# ----------------------------------------------------------------- data


def _gen_shard(rows_k_dist_seed_tag_r0_r1_fmt):
    rows, k, dist, seed, tag, r0, r1, fmt = rows_k_dist_seed_tag_r0_r1_fmt
    codes, scales = L._gen_rows(r0, r1, k, dist, seed, tag, fmt)
    return r0, codes, scales


def generate_parallel(rows: int, k: int, dist: int, seed: int, tag: int, workers: int,
                      fmt: str = "e2m3") -> L.Operand:
    if dist == L.DIST_INT:
        return L.int_operand(rows, k, seed=seed, tag=tag, fmt=fmt)
    codes = np.empty((rows, k * 6 // 8), dtype=np.uint8)
    scales = np.empty((rows, k // 32), dtype=np.uint8)
    step = 32
    jobs = [(rows, k, dist, seed, tag, r0, min(rows, r0 + step), fmt) for r0 in range(0, rows, step)]
    if workers <= 1:
        for j in jobs:
            r0, c, s = _gen_shard(j)
            codes[r0 : r0 + c.shape[0]], scales[r0 : r0 + s.shape[0]] = c, s
    else:
        with ProcessPoolExecutor(workers) as ex:
            for r0, c, s in ex.map(_gen_shard, jobs, chunksize=1):
                codes[r0 : r0 + c.shape[0]], scales[r0 : r0 + s.shape[0]] = c, s
    return L.Operand(rows, k, codes, scales, fmt)


def cached_operands(cache: Path, m: int, n: int, k: int, dist: str, seed: int, workers: int, fmt: str = "e2m3"):
    cache.mkdir(parents=True, exist_ok=True)
    if dist in E.DATASETS:  # full-shape edge datasets, derived from the Gaussian operands
        g = cached_operands(cache, m, n, k, "gauss", seed, workers, fmt)
        return [E.dataset(dist, rows, k, seed, tag, fmt, base=gb) for (rows, tag), gb in zip(((m, 1), (n, 2)), g)]
    out = []
    sfx = "" if fmt == "e2m3" else f"_{fmt}"  # E2M3 file names unchanged
    for rows, tag in ((m, 1), (n, 2)):
        p = cache / f"op_{dist}_{rows}x{k}_s{seed}_t{tag}{sfx}.bin"
        if p.exists():
            out.append(L.load_operand(p, fmt))
        else:
            t0 = time.time()
            op = generate_parallel(rows, k, DISTS[dist], seed, tag, workers, fmt)
            L.save_operand(op, p)
            print(f"[gen] {p.name} in {time.time() - t0:.1f} s", flush=True)
            out.append(op)
    return out


def cached_reference(cache: Path, a: L.Operand, b: L.Operand, tag: str, ieee: bool = False):
    pc, ps = cache / f"ref_{tag}_c.npy", cache / f"ref_{tag}_s.npy"
    if pc.exists() and ps.exists():
        return np.load(pc), np.load(ps)
    t0 = time.time()
    c, s = E.reference_ieee(a, b) if ieee else L.reference_f64(a, b)
    np.save(pc, c)
    np.save(ps, s)
    print(f"[ref] {tag} float64 reference in {time.time() - t0:.1f} s", flush=True)
    return c, s


# ----------------------------------------------------------------- compile


def aiter_pack_big(op: L.Operand, pad: int = 2) -> np.ndarray:
    """aiter pack_big_torch: per 256-row tile and 128-K tile the c0c1 image (16 B + 8 B per slot), 2 zero pad tiles
    after each row tile's K tiles."""
    nt, nk = op.rows // 256, op.k // 128
    img = L.pack_operand_c0c1(op, 256).reshape(nt, nk, 256 * 96)
    out = np.zeros((nt, nk + pad, 256 * 96), np.uint8)
    out[:, :nk] = img
    return out.reshape(-1)


def aiter_pack_scale(op: L.Operand, pad: int = 2) -> np.ndarray:
    """aiter pack_scale_torch: per (256-row tile t, 128-K tile s) 1024 bytes, byte off: su = off / 512,
    kg = (off % 512) / 128, r16 = (off % 128) / 8, sub = off % 8 -> row 256 t + 128 su + 16 sub + r16, block 4 s + kg;
    guard tiles 127."""
    nt, nb = op.rows // 256, op.k // 32
    nk = nb // 4
    off = np.arange(1024)
    su, kg, r16, sub = off // 512, (off % 512) // 128, (off % 128) // 8, off % 8
    rows = 128 * su + 16 * sub + r16
    sc = op.scales.reshape(nt, 256, nk, 4)  # [t][row][s][kg]
    out = np.full((nt, nk + pad, 1024), 127, np.uint8)
    out[:, :nk] = sc[:, rows, :, kg].transpose(1, 2, 0)  # advanced indices first: [off][t][s] -> [t][s][off]
    return out.reshape(-1)


def aiter_kargs(pa, pb, sa, sb, pc, m, n, k) -> bytes:
    """aiter a6w6 KernelArgs (0x180 bytes; offsets of asm_gemm_a6w6.cu)."""
    buf = bytearray(0x180)
    struct.pack_into("<Q", buf, 0x00, pc)
    struct.pack_into("<Q", buf, 0x20, pa)
    struct.pack_into("<Q", buf, 0x30, pb)
    struct.pack_into("<f", buf, 0x40, 1.0)
    struct.pack_into("<f", buf, 0x50, 0.0)
    struct.pack_into("<I", buf, 0x60, n)
    struct.pack_into("<I", buf, 0xE0, m)
    struct.pack_into("<I", buf, 0xF0, n)
    struct.pack_into("<I", buf, 0x100, k)
    struct.pack_into("<Q", buf, 0x110, sa)
    struct.pack_into("<Q", buf, 0x120, sb)
    struct.pack_into("<I", buf, 0x130, k // 32)
    struct.pack_into("<I", buf, 0x150, k // 32)
    return bytes(buf)


def _spec_from(over: dict) -> Mxfp6GemmSpec:
    return replace(Mxfp6GemmSpec(), **over)


#: Lever set of the direct (Rust) compiler for this kernel (scale operands in place, hoisted LDS bases, M0 fill,
#: no repeated M0 moves -- ``m0cse`` needs a direct compiler built with that lever; older builds ignore it).
RUST_LEVERS = "scalesep,dsrebase,dshoist,m0fill,m0cse"


_TOOL_LEVERS: Dict[str, Optional[frozenset]] = {}


def _tool_levers(tool: str) -> Optional[frozenset]:
    """The lever names of direct compiler ``tool`` (``target_compile --list-levers``; None: a build that predates
    the list -- it ignores unknown names silently)."""
    if tool not in _TOOL_LEVERS:
        r = subprocess.run([tool, "--list-levers"], capture_output=True, text=True, timeout=60)
        names = r.stdout.split() if r.returncode == 0 else []
        _TOOL_LEVERS[tool] = frozenset(names) if names else None
    return _TOOL_LEVERS[tool]


def check_rust_levers(tool: str, levers: str) -> bool:
    """Raise unless every name of ``levers`` is a lever of ``tool``: an unknown name builds the lever-off code
    silently. A compiler without ``--list-levers`` is refused unless ``ROCKE_RUST_LEVERS_UNCHECKED=1`` (then False)."""
    known = _tool_levers(tool)
    names = [n.strip().lstrip("-") for n in levers.replace("+", ",").split(",") if n.strip()]
    if known is None:
        if os.environ.get("ROCKE_RUST_LEVERS_UNCHECKED") == "1":
            return False
        raise RuntimeError(f"direct compiler {tool} cannot list its levers (it predates --list-levers and ignores "
                           "unknown names): use a newer build, or set ROCKE_RUST_LEVERS_UNCHECKED=1")
    unknown = [n for n in names if n not in known]
    if unknown:
        raise RuntimeError(f"unknown direct-compiler lever(s) {','.join(unknown)} for {tool}")
    return True


def _compile_rust(kernel, out: Path) -> dict:
    """Build with the rocKE direct (Rust) compiler: serialized IR -> ``$ROCKE_RUST_COMPILE`` (``target_compile``)."""
    import hashlib
    import tempfile

    from rocke.core.ir_serialize import serialize

    tool = os.environ.get("ROCKE_RUST_COMPILE")
    if not tool or not os.path.exists(tool):
        raise RuntimeError("compiler 'rust' needs ROCKE_RUST_COMPILE = path of the direct compiler's target_compile")
    text = serialize(kernel)
    levers = os.environ.get("ROCKE_RUST_LEVERS", RUST_LEVERS)
    checked = check_rust_levers(tool, levers)
    with tempfile.TemporaryDirectory() as td:
        ir = Path(td) / f"{out.stem}.ir"
        ir.write_text(text, newline="\n")
        r = subprocess.run([tool, "gfx950", td, str(ir)], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "ROCKE_DIRECT_LEVERS": levers})
        hs = Path(td) / f"{out.stem}.hsaco"
        if r.returncode != 0 or not hs.exists():
            raise RuntimeError(f"direct compiler failed: {(r.stderr or r.stdout)[-1500:]}")
        out.write_bytes(hs.read_bytes())
    return {"compiler": "rust", "rust_levers": levers, "rust_tool": tool, "rust_levers_checked": checked,
            "rust_tool_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
            "ir_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _spills(meta: dict) -> bool:
    lp, nt = meta.get("loop") or {}, meta.get("notes") or {}
    return bool(nt.get("vgpr_spill_count") or nt.get("private_segment_fixed_size") or lp.get("scratch")
                or lp.get("accvgpr") or lp.get("vmcnt0"))


def cmd_compile(args) -> int:
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    over = json.loads(args.spec)
    spec = _spec_from(over)
    ok, why = validate_spec(spec)
    if not ok:
        print(json.dumps({"ok": False, "error": why}))
        return 2
    t0 = time.time()
    out = Path(args.out)
    kernel = build_mxfp6_gemm(spec, arch="gfx950")
    which = args.compiler
    if which == "auto" and spec.required_compiler == "rust":
        which = "rust"  # a preset that needs the direct compiler: never fall back to LLVM silently
    try:
        check_compiler(spec, "rust" if which == "rust" else "llvm")
    except ValueError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    extra = {}
    if which in ("llvm", "auto"):
        art = compile_kernel(kernel, arch="gfx950", backend="python")
        out.write_bytes(art.hsaco)  # (written before the ISA statistics read it back)
        extra = {"compiler": "llvm", "comgr": str(getattr(comgr, "resolved_lib_path", lambda: "?")())}
    if which == "auto":
        # the direct compiler replaces the LLVM build only when LLVM spills / shuffles and the direct build is clean
        probe = {"loop": isa_loop_stats(out.read_bytes(), out), "notes": code_object_notes(out.read_bytes())}
        if _spills(probe) and os.environ.get("ROCKE_RUST_COMPILE"):
            rust_out = out.with_suffix(".rust.hsaco")
            rmeta = _compile_rust(kernel, rust_out)
            rprobe = {"loop": isa_loop_stats(rust_out.read_bytes(), rust_out),
                      "notes": code_object_notes(rust_out.read_bytes())}
            if not _spills(rprobe):
                out.write_bytes(rust_out.read_bytes())
                extra = {**rmeta, "auto_reason": "llvm build spills; direct build clean", "llvm_loop": probe["loop"]}
    elif which == "rust":
        extra = _compile_rust(kernel, out)
    hs = out.read_bytes()
    meta = {
        "ok": True,
        "loop": isa_loop_stats(hs, out),
        "name": kernel_name(spec),
        "compile_s": round(time.time() - t0, 2),
        "required_compiler": spec.required_compiler,
        **extra,
        "notes": code_object_notes(hs),
        "descriptor": descriptor(spec),
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta.get(k) for k in ("ok", "name", "compiler", "compile_s", "notes", "loop")}))
    return 0


def _objdump():
    import glob
    import shutil

    cand = [os.environ.get("OBJDUMP", "")] + sorted(glob.glob("/opt/rocm*/llvm/bin/llvm-objdump"), reverse=True)
    cand.append(shutil.which("llvm-objdump") or "")
    for c in cand:
        if c and os.path.exists(c):
            return c
    return None


def loop_stats(dis: str) -> dict:
    """Instruction mix of the longest backward-branch loop of a disassembly (the K loop)."""
    import re
    from collections import Counter

    lines = dis.splitlines()
    addr, base = {}, None
    for i, line in enumerate(lines):
        mm = re.match(r"^([0-9a-f]{16}) <", line)
        if mm and base is None:
            base = int(mm.group(1), 16)
        mm = re.search(r"// ([0-9A-F]{12}):", line)
        if mm:
            addr[int(mm.group(1), 16)] = i
    best = None
    for i, line in enumerate(lines):
        mm = re.search(r"s_cbranch_\w+ .*<.*\+0x([0-9a-f]+)>", line)
        if mm and base is not None:
            tgt = base + int(mm.group(1), 16)
            if tgt in addr and addr[tgt] < i and (best is None or i - addr[tgt] > best[1] - best[0]):
                best = (addr[tgt], i)
    if not best:
        return {}
    body = lines[best[0] : best[1] + 1]
    ins = [line.split()[0] for line in body if re.match(r"^\s+[a-z_0-9]+", line)]
    c = Counter(ins)
    mf = sum(v for kk, v in c.items() if kk.startswith("v_mfma"))
    return {
        "instr": len(ins),
        "mfma": mf,
        "non_mfma_per_mfma": round((len(ins) - mf) / mf, 3) if mf else None,
        "s_nop": c.get("s_nop", 0),
        "s_waitcnt": c.get("s_waitcnt", 0),
        "vmcnt0": sum(1 for line in body if "vmcnt(0)" in line),
        "accvgpr": sum(v for kk, v in c.items() if kk.startswith("v_accvgpr")),
        "scratch": sum(v for kk, v in c.items() if kk.startswith("scratch_")),
        "ds_read": sum(v for kk, v in c.items() if kk.startswith("ds_read")),
        "dtl": sum(1 for line in body if "buffer_load" in line and " lds" in line),
        "valu": sum(v for kk, v in c.items() if kk.startswith("v_") and not kk.startswith(("v_mfma", "v_accvgpr"))),
        "salu": sum(v for kk, v in c.items() if kk.startswith("s_") and kk not in ("s_nop", "s_waitcnt", "s_barrier")),
    }


def isa_loop_stats(hsaco: bytes, out: Path) -> dict:
    od = _objdump()
    if not od:
        return {"error": "no llvm-objdump"}
    try:
        dis = subprocess.run([od, "-d", "--mcpu=gfx950", str(out)], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    Path(str(out) + ".s").write_text(dis)
    return loop_stats(dis)


_NOTE_KEYS = (
    ".vgpr_count",
    ".agpr_count",
    ".sgpr_count",
    ".vgpr_spill_count",
    ".sgpr_spill_count",
    ".private_segment_fixed_size",
    ".group_segment_fixed_size",
)


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
    for k in _NOTE_KEYS:
        kb = k.encode()
        p = hs.find(bytes([0xA0 | len(kb)]) + kb)
        if p >= 0:
            out[k[1:]] = rd(p + 1 + len(kb))
    return out


def _patch_probe(out: Path, classes: str, branch: bool) -> dict:
    """Patch instruction classes out of a code object (``MXFP6_PATCH_CO`` = the patcher script)."""
    tool = os.environ.get("MXFP6_PATCH_CO")
    if not tool:
        return {"ok": False, "error": "MXFP6_PATCH_CO not set"}
    dst = Path(str(out) + ".patched")
    cmd = [sys.executable, tool, str(out), str(dst), classes] + (["--branch"] if branch else [])
    od = _objdump()
    if od:
        cmd += ["--objdump", od]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists():
        return {"ok": False, "error": (r.stderr or r.stdout)[-1000:]}
    return {"hsaco": str(dst), "patched": classes + (" branch" if branch else "")}


def compile_arms(arms: dict, work: Path, jobs: int, compiler: str = "auto") -> dict:
    """Compile every arm in its own subprocess (parallel, before GPU init).

    ``compiler`` (llvm | rust | auto) is the default; an arm's ``_compiler`` key overrides it.
    """
    work.mkdir(parents=True, exist_ok=True)
    procs = {}
    pending = list(arms.items())
    results = {}
    env = dict(os.environ)
    while pending or procs:
        while pending and len(procs) < jobs:
            name, over = pending.pop(0)
            out = work / f"{name}.hsaco"
            if over.get("_aiter"):  # the aiter asm kernel (code object shipped with the harness)
                co = os.environ.get("AITER_CO", "")
                results[name] = ({"ok": True, "hsaco": co, "name": "f6gemm_dmabig_kernel_func", "aiter": True,
                                  "compiler": "aiter-asm"} if co and os.path.exists(co)
                                 else {"ok": False, "error": "AITER_CO not set / missing"})
                continue
            which = over.get("_compiler", compiler)
            arm_env = dict(env)
            if over.get("_levers"):  # direct-compiler lever set of this arm (default RUST_LEVERS)
                arm_env["ROCKE_RUST_LEVERS"] = over["_levers"]
            over = {k: v for k, v in over.items() if not k.startswith("_")}
            cmd = [sys.executable, "-m", "benchmarks.gfx950.mxfp6_gemm.bench", "compile", "--spec", json.dumps(over),
                   "--out", str(out), "--compiler", which]
            procs[name] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                            env=arm_env), out)
        for name in list(procs):
            p, out = procs[name]
            if p.poll() is not None:
                so, se = p.communicate()
                if p.returncode != 0 or not out.exists():
                    results[name] = {"ok": False, "error": (se or so)[-2000:]}
                else:
                    meta = json.loads(Path(str(out) + ".json").read_text())
                    meta["hsaco"] = str(out)
                    patch = arms[name].get("_patch")
                    if patch:  # timing probe: instruction classes patched out of the real code object
                        meta.update(_patch_probe(out, patch, bool(arms[name].get("_branch"))))
                    results[name] = meta
                del procs[name]
        time.sleep(0.05)
    return results


# ----------------------------------------------------------------- GPU side


class Clock:
    """sysfs SCLK / board power sampler for the HIP device in use."""

    def __init__(self, bdf: str):
        self.bdf = bdf.lower()
        base = Path("/sys/bus/pci/devices") / self.bdf
        self.sclk = base / "pp_dpm_sclk"
        self.power = None
        for h in sorted((base / "hwmon").glob("hwmon*")) if (base / "hwmon").exists() else []:
            for nm in ("power1_average", "power1_input"):
                if (h / nm).exists():
                    self.power = h / nm
                    break
            if self.power:
                break
        self._stop = threading.Event()
        self.mhz: list = []
        self.watts: list = []
        self._th = None

    def _read_sclk(self):
        try:
            for line in self.sclk.read_text().splitlines():
                if "*" in line:
                    return float(line.split(":")[1].strip().split("M")[0])
        except OSError:
            return None
        return None

    def _read_power(self):
        if not self.power:
            return None
        try:
            return float(self.power.read_text().strip()) / 1e6
        except (OSError, ValueError):
            return None

    def start(self):
        self.mhz, self.watts = [], []
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                v = self._read_sclk()
                if v:
                    self.mhz.append(v)
                w = self._read_power()
                if w:
                    self.watts.append(w)
                time.sleep(0.005)

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join()
        med = lambda v: statistics.median(v) if v else None  # noqa: E731
        return {"sclk_med": med(self.mhz), "sclk_min": min(self.mhz) if self.mhz else None,
                "watts_med": med(self.watts), "n": len(self.mhz)}


def device_bdf() -> str:
    from rocke.runtime.hip_module import _resolve_hip

    hip = _resolve_hip()
    buf = ctypes.create_string_buffer(64)
    hip.hipDeviceGetPCIBusId(buf, 64, 0)
    return buf.value.decode()


def _u8(a: np.ndarray):
    a = np.ascontiguousarray(a)
    return (ctypes.c_ubyte * int(a.nbytes)).from_buffer(a)


def cmd_run(args) -> int:
    m, n, k = args.m, args.n, args.k
    cache = Path(args.cache)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    arms = json.loads(Path(args.arms).read_text()) if os.path.exists(args.arms) else json.loads(args.arms)
    # always include shape in every arm
    arms = {nm: {**{"m": m, "n": n, "k": k}, **over} for nm, over in arms.items()}
    base_probe = arms.get(args.base, {}).get("probe", "none") != "none"
    mutants = {} if (args.no_mutants or base_probe) else {
        f"mut_{mu}": {**arms[args.base], "mutant": mu} for mu in ("scale_sel", "k_order", "drop_kstep", "wrong_fmt")
    }
    t0 = time.time()
    built = compile_arms({**arms, **mutants}, work, args.jobs, args.compiler)
    print(f"[compile] {len(built)} arms in {time.time() - t0:.1f} s", flush=True)
    report = {"shape": [m, n, k], "arms": {}, "env": {k_: os.environ.get(k_) for k_ in
              ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "ROCKE_HIP_LIB", "SLURM_JOB_ID", "SLURMD_NODENAME")}}
    for nm, meta in built.items():
        report["arms"][nm] = {"build": {k_: meta.get(k_) for k_ in ("ok", "name", "compiler", "compile_s", "notes", "loop",
                                                                    "error", "comgr", "rust_levers", "ir_sha256",
                                                                    "auto_reason", "rust_tool", "rust_tool_sha256",
                                                                    "rust_levers_checked")}}
        print(f"[build] {nm}: {report['arms'][nm]['build']}", flush=True)
    # ---------------- data
    dists = [d for d in args.dists.split(",") if d]
    fmts = sorted({_spec_from({k_: v for k_, v in over.items() if not k_.startswith("_")}).elem_fmt
                   for over in list(arms.values()) + list(mutants.values())})
    data = {}  # (dist, element format) -> (a, b, (ref, abs_sum))
    for f in fmts:
        sfx = "" if f == "e2m3" else f"_{f}"
        for d in dists:
            a, b = cached_operands(cache, m, n, k, d, args.seed, args.gen_workers, f)
            ref = cached_reference(cache, a, b, f"{d}_{m}x{n}x{k}_s{args.seed}{sfx}", ieee=d in E.DATASETS)
            data[(d, f)] = (a, b, ref)
    # ---------------- GPU init (after all compiles)
    from rocke.runtime.hip_module import Runtime

    pack_hs = {}
    def quant_key(over: dict) -> tuple:
        """("quant", out layout, impl, fmt, stage, work-group rows) of a "_quant" arm. impl "mx" (default):
        pack.build_quant_mx (hardware conversion, its measured-best defaults); "legacy": pack.build_quant_e2m3 with
        "_quant_stage" / "_quant_wg"."""
        impl = over.get("_quant_impl", "mx")
        if impl == "mx":
            return ("quant", over["_quant"], "mx", over.get("elem_fmt", "e2m3"), "", 0)
        return ("quant", over["_quant"], "legacy", "e2m3", over.get("_quant_stage", "lds"), int(over.get("_quant_wg", 256)))

    for qk in sorted({quant_key(over) for over in arms.values() if over.get("_quant")}):  # bf16 A quantizers
        from rocke.helpers.compile import compile_kernel

        from kernels.gfx950.mxfp6_gemm.pack import build_quant_e2m3, build_quant_mx, quant_grid, quant_mx_grid

        if qk[2] == "mx":
            kq = build_quant_mx(m, k, fmt=qk[3], src="bf16", out=qk[1], scale_out="tile")
            grid_q = quant_mx_grid(m, k)
        else:
            kq = build_quant_e2m3(m, k, out=qk[1], scale_out="tile", stage=qk[4], wg_rows=qk[5])
            grid_q = quant_grid(m, k, stage=qk[4], wg_rows=qk[5])
        pack_hs[qk] = (compile_kernel(kq, arch="gfx950", backend="python").hsaco, kq.name, grid_q)
    pack_impls = {over.get("_pack_impl", "new") for over in arms.values() if over.get("_pack")}
    if pack_impls:  # per-call GPU packing (pack.py), LLVM-built
        from rocke.helpers.compile import compile_kernel

        from kernels.gfx950.mxfp6_gemm.pack import (build_pack_c0c1, build_pack_operand, build_pack_scales,
                                                    build_pack_scales_tile, pack_grids, pack_operand_grid,
                                                    pack_scales_grid)

        for rows in sorted({m, n}):
            if "legacy" in pack_impls:  # two launches per operand: codes, scales
                for kind, kfn in (("ops", build_pack_c0c1(rows, k)), ("scales", build_pack_scales_tile(rows, k))):
                    art = compile_kernel(kfn, arch="gfx950", backend="python")
                    pack_hs[(kind, rows)] = (art.hsaco, kfn.name, pack_grids(rows, k)[0 if kind == "ops" else 1])
            if "new" in pack_impls:  # one launch per operand: codes + scales
                kfn = build_pack_operand(rows, k)
                pack_hs[("opnew", rows)] = (compile_kernel(kfn, arch="gfx950", backend="python").hsaco, kfn.name,
                                            pack_operand_grid(rows, k))
        if "new" in pack_impls:  # scales only, A (+ B) in one launch
            for tag, rb in (("A", 0), ("AB", n)):
                kfn = build_pack_scales(m, k, rb)
                pack_hs[("scnew", tag)] = (compile_kernel(kfn, arch="gfx950", backend="python").hsaco, kfn.name,
                                           pack_scales_grid(m, k, rb))
        print(f"[pack] built {sorted(pack_hs)}", flush=True)
    rt = Runtime()
    pack_fn = {key: (rt.load_module(hs).get_function(name), grid) for key, (hs, name, grid) in pack_hs.items()}
    bdf = device_bdf()
    report["bdf"] = bdf
    print(f"[gpu] bdf {bdf}", flush=True)
    clk = Clock(bdf)
    dev = {"c": rt.alloc(m * n * 2)}
    op_dev, scale_dev = {}, {}

    def _put(arr: np.ndarray) -> int:
        ptr = rt.alloc(arr.nbytes)
        rt.memcpy_h2d(ptr, _u8(arr), arr.nbytes)
        return ptr

    def upload_aiter(d: str, spec) -> tuple:
        """aiter's operand / scale layout (pack_big_torch / pack_scale_torch ports) of dist ``d``."""
        a, b, _ = data[(d, spec.elem_fmt)]
        key = ("aiter", d, spec.elem_fmt)
        if key not in op_dev:
            op_dev[key] = (_put(aiter_pack_big(a)), _put(aiter_pack_big(b)), _put(aiter_pack_scale(a)),
                           _put(aiter_pack_scale(b)))
        return op_dev[key]

    def kargs_of(nm: str, d: str) -> bytes:
        x = loaded[nm]
        if x.get("aiter"):
            pa, pb, sa, sb = upload_aiter(d, x["spec"])
            return aiter_kargs(pa, pb, sa, sb, dev["c"], x["spec"].m, x["spec"].n, x["spec"].k)
        pa, pb, sa, sb = upload(d, x["spec"], x["bias"])
        return struct.pack("<QQQQQ", pa, pb, sa, sb, dev["c"])

    def upload(d: str, spec, bias=(0, 0)) -> tuple:
        """Device pointers (a, b, sa, sb) of dist ``d`` in ``spec``'s operand / scale layouts (cached).

        ``bias`` = (dA, dB): input-encoding transform, every A / B E8M0 scale shifted (layout.apply_scale_bias);
        exact when dA + dB == -spec.out_exp2.
        """
        a, b, _ = data[(d, spec.elem_fmt)]
        a, b = L.apply_scale_bias(a, bias[0]), L.apply_scale_bias(b, bias[1])
        la, lb = operand_layout_of(spec, "A"), operand_layout_of(spec, "B")
        ko = (d, spec.elem_fmt, la, lb, spec.tile_m, spec.tile_n)
        if ko not in op_dev:
            op_dev[ko] = (_put(L.pack_operand(a, la, spec.tile_m)), _put(L.pack_operand(b, lb, spec.tile_n)))
        ks = (d, spec.elem_fmt, spec.atom, spec.scale_layout, tuple(bias))
        if ks not in scale_dev:
            scale_dev[ks] = (_put(L.pack_scales_for(a.scales, spec.atom, spec.scale_layout)),
                             _put(L.pack_scales_for(b.scales, spec.atom, spec.scale_layout)))
        return op_dev[ko] + scale_dev[ks]

    loaded = {}
    for nm, meta in built.items():
        if not meta.get("ok"):
            continue
        mod = rt.load_module(Path(meta["hsaco"]).read_bytes())
        spec = _spec_from({k: v for k, v in (arms.get(nm) or mutants.get(nm)).items() if not k.startswith("_")})
        grid, block = grid_of(spec)
        if meta.get("aiter"):
            spec = _spec_from({k_: v for k_, v in (arms.get(nm) or {}).items() if not k_.startswith("_")})
            loaded[nm] = {"mod": mod, "fn": mod.get_function(meta["name"]), "grid": (spec.n // 256, spec.m // 256, 1),
                          "block": (256, 1, 1), "spec": spec, "bias": (0, 0), "aiter": True}
            continue
        bias = tuple(int(x) for x in str((arms.get(nm) or mutants.get(nm)).get("_bias", "0,0")).split(","))
        if sum(bias) != -spec.out_exp2:
            raise SystemExit(f"arm {nm}: scale bias {bias} needs out_exp2 = {-sum(bias)}")
        loaded[nm] = {"mod": mod, "fn": mod.get_function(meta["name"]), "grid": grid, "block": block, "spec": spec,
                      "bias": bias}
    c_host = np.empty((m, n), dtype=np.uint16)

    raw_dev = {}

    def launches_of(nm: str, d: str) -> list:
        """[(fn, grid, block, kargs)]: for "_pack" arms the per-call GPU packs (canonical codes + raw scales ->
        c0c1 + tile scales, into this arm's own buffers) followed by the GEMM on them."""
        x = loaded[nm]
        mode = (arms.get(nm) or mutants.get(nm) or {}).get("_pack")
        qmode = (arms.get(nm) or mutants.get(nm) or {}).get("_quant")
        if qmode:  # bf16 activations -> E2M3 (+ layout) on the GPU, then the GEMM on them (B pre-packed)
            spec = x["spec"]
            a, b, _ = data[(d, spec.elem_fmt)]
            pa, pb, sa, sb = upload(d, spec, x["bias"])
            key = ("act", d, spec.elem_fmt)
            if key not in raw_dev:  # bf16 of the decoded A (exactly representable): requantizes to the same values
                av = L.decode_operand(a).astype(np.float32)
                xb = (av.view(np.uint32) >> 16).astype(np.uint16)
                raw_dev[key] = {"x": _put(xb), "qa": rt.alloc(a.codes.nbytes), "qsa": rt.alloc(a.scales.nbytes)}
            r = raw_dev[key]
            fq, gq = pack_fn[quant_key(arms.get(nm) or mutants.get(nm))]
            qlaunch = (fq, (gq, 1, 1), (256, 1, 1), struct.pack("<QQQ", r["x"], r["qa"], r["qsa"]))
            if (arms.get(nm) or {}).get("_only") == "quant":  # the quantizer alone (timing only, not a GEMM)
                return [qlaunch]
            return [qlaunch,
                    (x["fn"], x["grid"], x["block"], struct.pack("<QQQQQ", r["qa"], pb, r["qsa"], sb, dev["c"]))]
        if not mode:
            return [(x["fn"], x["grid"], x["block"], kargs_of(nm, d))]
        spec = x["spec"]
        a, b, _ = data[(d, spec.elem_fmt)]
        pa, pb, sa, sb = upload(d, spec, x["bias"])
        key = ("raw", d, spec.elem_fmt)
        if key not in raw_dev:  # canonical inputs + per-arm-kind packed outputs
            raw_dev[key] = {"a": _put(np.ascontiguousarray(a.codes)), "b": _put(np.ascontiguousarray(b.codes)),
                            "sa": _put(np.ascontiguousarray(a.scales)), "sb": _put(np.ascontiguousarray(b.scales)),
                            "pa": rt.alloc(a.codes.nbytes), "pb": rt.alloc(b.codes.nbytes),
                            "psa": rt.alloc(a.scales.nbytes), "psb": rt.alloc(b.scales.nbytes)}
        r = raw_dev[key]
        pre = []
        impl = (arms.get(nm) or {}).get("_pack_impl", "new")
        if mode in ("As", "ABs") and impl == "new":  # scales only, one launch (pack.build_pack_scales)
            fs, gs = pack_fn[("scnew", "AB" if mode == "ABs" else "A")]
            pre.append((fs, (gs, 1, 1), (256, 1, 1), struct.pack("<QQQQ", r["sa"], r["sb"], r["psa"], r["psb"])))
        elif mode in ("As", "ABs"):  # scales only (raw -> tile, per call); the codes as the spec reads them
            for X, rows in (("a", spec.m), ("b", spec.n)):
                if X == "b" and mode != "ABs":
                    continue
                fs, gs = pack_fn[("scales", rows)]
                pre.append((fs, (gs, 1, 1), (256, 1, 1), struct.pack("<QQ", r["s" + X], r["ps" + X])))
        if mode in ("As", "ABs"):
            if (arms.get(nm) or {}).get("_only") == "pack":
                return pre
            gsb = r["psb"] if mode == "ABs" else sb
            return pre + [(x["fn"], x["grid"], x["block"], struct.pack("<QQQQQ", pa, pb, r["psa"], gsb, dev["c"]))]
        for X, rows in (("a", spec.m), ("b", spec.n)):
            if X == "b" and mode != "AB":
                continue
            if impl == "new":  # codes + scales in one launch (pack.build_pack_operand)
                fo, go = pack_fn[("opnew", rows)]
                pre.append((fo, (go, 1, 1), (256, 1, 1),
                            struct.pack("<QQQQ", r[X], r["s" + X], r["p" + X], r["ps" + X])))
                continue
            fo, go = pack_fn[("ops", rows)]
            fs, gs = pack_fn[("scales", rows)]
            pre.append((fo, (go, 1, 1), (256, 1, 1), struct.pack("<QQ", r[X], r["p" + X])))
            pre.append((fs, (gs, 1, 1), (256, 1, 1), struct.pack("<QQ", r["s" + X], r["ps" + X])))
        if (arms.get(nm) or {}).get("_only") == "pack":  # the packs alone (timing only, not a GEMM)
            return pre
        ga, gsa = r["pa"], r["psa"]
        gb, gsb = (r["pb"], r["psb"]) if mode == "AB" else (pb, sb)
        return pre + [(x["fn"], x["grid"], x["block"], struct.pack("<QQQQQ", ga, gb, gsa, gsb, dev["c"]))]

    def run_once(nm: str, d: str) -> np.ndarray:
        rt.memset(dev["c"], 0xFF, c_host.nbytes)  # NaN-fill
        for fn_, g_, bl_, ka_ in launches_of(nm, d):
            rt.launch(fn_, g_, bl_, ka_)
        rt.wait_stream(0)
        rt.memcpy_d2h(_u8(c_host), dev["c"], c_host.nbytes)
        return c_host.copy()

    # ---------------- verify
    verdict = {}
    for nm in loaded:
        is_mut = nm.startswith("mut_")
        if loaded[nm]["spec"].probe != "none" or built[nm].get("patched") or (arms.get(nm) or {}).get("_only"):
            verdict[nm] = True  # a measurement probe, not a GEMM: timed, never verified
            report["arms"].setdefault(nm, {})["verify"] = "probe (not a GEMM)"
            print(f"[verify] {nm}: probe {loaded[nm]['spec'].probe} (not verified)", flush=True)
            continue
        res = {}
        for d in dists:
            if is_mut and d == "int" and "scale_sel" in nm:
                continue  # all scales are 2^0 on the integer set: op_sel cannot show
            ref_c, ref_s = data[(d, loaded[nm]["spec"].elem_fmt)][2]
            outs = [run_once(nm, d) for _ in range(1 if is_mut else args.verify_launches)]
            if d in E.DATASETS:  # documented OCP / IEEE convention (edge_cases.py)
                chk = [{**E.check_ieee(o, ref_c, ref_s), "exact": False, "finite": True} for o in outs]
                chk = [{**c, "pass": c["bad"] == 0} for c in chk]
            else:
                chk = [L.check_against_reference(o, ref_c, ref_s, exact=(d == "int")) for o in outs]
            same = all(np.array_equal(outs[0], o) for o in outs[1:])
            res[d] = {**chk[0], "all_pass": all(c["pass"] for c in chk), "deterministic": same,
                      "sha1": hashlib.sha1(outs[0].tobytes()).hexdigest()[:16]}
        ok = all(r["all_pass"] and r["deterministic"] for r in res.values())
        if is_mut:
            ok = not all(r["all_pass"] for r in res.values())  # a mutant must fail somewhere
        verdict[nm] = ok
        report["arms"].setdefault(nm, {})["verify"] = res
        print(f"[verify] {nm}: {'PASS' if ok else 'FAIL'} " +
              " ".join(f"{d}:{'ok' if r['all_pass'] else 'BAD'}(bad={r['bad']},{r.get('worst_err_over_tol', 0):.3g})"
                       for d, r in res.items()) + " sha " + ",".join(r["sha1"] for r in res.values()), flush=True)
    # "_force_time": time the arm even when it fails verification (competitor envelopes on edge data); its verdict
    # stays recorded in the report
    timed = [nm for nm in arms if verdict.get(nm) or (arms[nm].get("_force_time") and nm in loaded)]
    if len(timed) < len(arms):
        print(f"[verify] arms failing verification are not timed: {[a for a in arms if a not in timed]}")
    if not all(verdict.get(nm) for nm in mutants):
        print("[verify] WARNING: a must-fail mutant passed")
    # ---------------- time
    flop = 2.0 * m * n * k

    def sample(nm: str, d: str, gap_s: float, warmup: int, iters: int, sustained: bool):
        seq = launches_of(nm, d)
        rt.wait_stream(0)
        time.sleep(gap_s)
        for _ in range(warmup):
            for fn_, g_, bl_, ka_ in seq:
                rt.launch(fn_, g_, bl_, ka_)
        if sustained:
            clk.start()
        e0, e1 = rt.event(), rt.event()
        e0.record()
        for _ in range(iters):
            for fn_, g_, bl_, ka_ in seq:
                rt.launch(fn_, g_, bl_, ka_)
        e1.record()
        e1.synchronize()
        ms = e0.elapsed_to(e1) / iters
        e0.destroy()
        e1.destroy()
        cs = clk.stop() if sustained else None
        return flop / (ms * 1e9), ms, cs

    for d in [x for x in args.time_dists.split(",") if x]:
        rng = np.random.default_rng(args.seed)
        for rnd in range(args.rounds):
            order = list(timed)
            rng.shuffle(order)
            for nm in order:
                tf, ms, _ = sample(nm, d, args.gap_ms / 1000.0, 1, 3, False)
                report["arms"][nm].setdefault(f"peak_{d}", []).append(tf)
        for rnd in range(args.sustained_rounds):
            order = list(timed)
            rng.shuffle(order)
            for nm in order:
                # ~sustain_s of back-to-back launches
                tf, ms, cs = sample(nm, d, 2.0, 20, max(50, int(args.sustain_s * 1000 / max(0.05, report['arms'][nm].get('_ms', 0.4)))), True)
                report["arms"][nm]["_ms"] = ms
                report["arms"][nm].setdefault(f"sustained_{d}", []).append({"tflops": tf, "ms": ms, **(cs or {})})
    # ---------------- summary
    print("\n[summary] TF/s (internal record only; warm = steady-state headline, cold = post-idle start)")
    base = args.base if args.base in timed else (timed[0] if timed else None)
    for nm in timed:
        row = report["arms"][nm]
        cells = []
        for key in sorted(kk for kk in row if kk.startswith("sustained_") and not kk.endswith("_median")):
            v = row[key]
            med = statistics.median(s["tflops"] for s in v)
            row[key + "_median"] = med
            sc = statistics.median(s["sclk_med"] for s in v if s.get("sclk_med")) if any(s.get("sclk_med") for s in v) else None
            pw = statistics.median(s["watts_med"] for s in v if s.get("watts_med")) if any(s.get("watts_med") for s in v) else None
            util = med * 1e12 / (256 * sc * 1e6 * 16384) if sc else None  # MFMA utilization per clock (256 CUs)
            row[key + "_util"] = util
            ms = statistics.median(s["ms"] for s in v)
            mj = pw * ms if pw else None  # board energy per GEMM (mJ): at the power cap TF/s ~ cap / energy
            row[key + "_mj"] = mj
            lab = "warm_" + key[len("sustained_"):]
            cells.append(f"{lab}={med:8.1f} sclk={sc} W={pw} util={util:.3f} mJ={mj:.0f}" if util and mj
                         else f"{lab}={med:8.1f}")
        for key in sorted(kk for kk in row if kk.startswith("peak_") and not kk.endswith("_median")):
            med = statistics.median(row[key])
            ref = statistics.median(report["arms"][base][key]) if base else med
            row[key + "_median"] = med
            cells.append(f"cold_{key[len('peak_'):]}={med:8.1f} ({med / ref:.3f}x)")
        # new names (cold-start = the post-idle "peak" protocol, warm = the steady-state "sustained" window);
        # the old keys stay as aliases of the same data
        for key in [kk for kk in row if kk.startswith(("peak_", "sustained_"))]:
            new = ("cold_" + key[len("peak_"):]) if key.startswith("peak_") else ("warm_" + key[len("sustained_"):])
            row[new] = row[key]
        print(f"  {nm:<24} " + "  ".join(cells))
    out = Path(args.json)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[run] wrote {out}")
    for x in loaded.values():
        x["mod"].unload()
    return 0 if all(verdict.get(nm) for nm in arms) else 1


def cmd_gen(args) -> int:
    t0 = time.time()
    op = generate_parallel(args.rows, args.k, DISTS[args.dist], args.seed, args.tag, args.workers, args.fmt)
    L.save_operand(op, args.out)
    print(f"[gen] {args.out}: {time.time() - t0:.1f} s")
    return 0


def cmd_compare(args) -> int:
    a, b = Path(args.a).read_bytes(), Path(args.b).read_bytes()
    if len(a) != len(b):
        print(f"[compare] SIZE MISMATCH {len(a)} vs {len(b)}")
        return 1
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    diff = np.flatnonzero(x != y)
    print(f"[compare] {len(a)} bytes, {diff.size} differ" + (f", first at {diff[:8].tolist()}" if diff.size else ""))
    return 0 if diff.size == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--rows", type=int, required=True)
    g.add_argument("--k", type=int, required=True)
    g.add_argument("--dist", choices=list(DISTS), default="gauss")
    g.add_argument("--fmt", choices=list(L.FMTS), default="e2m3")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--tag", type=int, default=1)
    g.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    g.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    cc = sub.add_parser("compile")
    cc.add_argument("--spec", default="{}")
    cc.add_argument("--out", required=True)
    cc.add_argument("--compiler", choices=("llvm", "rust", "auto"), default=os.environ.get("MXFP6_COMPILER", "auto"))
    r = sub.add_parser("run")
    r.add_argument("--m", type=int, default=4096)
    r.add_argument("--n", type=int, default=4096)
    r.add_argument("--k", type=int, default=65536)
    r.add_argument("--arms", default='{"base": {}, "twin": {}}', help="JSON {name: spec overrides} or a file")
    r.add_argument("--base", default="base")
    r.add_argument("--dists", default="int,gauss,uniform")
    r.add_argument("--time-dists", default="gauss,uniform")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--rounds", type=int, default=15)
    r.add_argument("--gap-ms", type=float, default=25.0)
    r.add_argument("--sustained-rounds", "--warm-rounds", dest="sustained_rounds", type=int, default=2)
    r.add_argument("--sustain-s", "--warm-s", dest="sustain_s", type=float, default=3.0)
    r.add_argument("--verify-launches", type=int, default=3)
    r.add_argument("--no-mutants", action="store_true")
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--compiler", choices=("llvm", "rust", "auto"), default=os.environ.get("MXFP6_COMPILER", "auto"),
                   help="default build path of every arm (an arm's _compiler key overrides); rust / auto need "
                        "ROCKE_RUST_COMPILE (the direct compiler's target_compile); auto = LLVM unless it spills and "
                        "the direct build is clean")
    r.add_argument("--gen-workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--cache", default="mxfp6_cache")
    r.add_argument("--work", default="mxfp6_work")
    r.add_argument("--json", default="mxfp6_bench.json")
    args = p.parse_args(argv)
    return {"gen": cmd_gen, "compare": cmd_compare, "compile": cmd_compile, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
