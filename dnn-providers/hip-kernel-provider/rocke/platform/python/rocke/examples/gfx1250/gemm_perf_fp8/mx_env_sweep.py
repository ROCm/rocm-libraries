"""EXPERIMENT: sweep the env-gated MX FP8 path of ``gemm_universal`` on gfx1250.

The MX path and its options are switched on by ``ROCKE_EXP_*`` environment
flags read at build time, so every point of the sweep is one compiled kernel:
``variants`` (flag sets: scale swizzle, K-atom fence, ...) times ``arms``
(cluster shape and multicast flags), all on the 256x256x256, 2x2-wave,
16x16x128 TDM spec of the B0 fat-wave champion. Kernels compile in parallel
worker processes; one that spills (VGPR or SGPR) is never run.

Timing goes first: ``rounds`` interleaved rounds (order reversed every other
round), each kernel 1 s idle, ``warmup`` launches, then ``repeat`` launches in
one event pair, on rand data (normal(0, 0.1) -> e4m3, E8M0 scales 2^-3..2^3), or
with ``--data mxgemm`` on the mx_gemm prototype bench's timed inputs.
Correctness comes last, for the ``--verify-top`` best kernels only, at
K = 8192: exact on the integer set (scales 2^0..2^1), max relative error under
the rand bound on the rand set.

``--gpu auto`` (two partitions per physical GPU): samples ``amd-smi process``,
takes the first pair of ``--pairs`` whose partitions both stayed idle and runs
on its first partition; a timed batch with another process on the pair (checked
before and after the batch) is tagged and left out of the ranking. ``--allow-single`` falls back to
one idle partition (quick checks only).

    python -m rocke.examples.gfx1250.gemm_perf_fp8.mx_env_sweep \
        --config mx_env_sweep.json --gpu auto --out <dir> [--jobs 8] [--dry-run]

Writes ``<out>/results.csv`` and prints a ranked summary per shape.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import ctypes
import ctypes.util
import json
import multiprocessing as mp
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------- the spec


def _spec():
    """The swept spec; variant envs ROCKE_EXP_SWEEP_LDS_PAD (default 48) and
    ROCKE_EXP_SWEEP_EPILOGUE (default cshuffle) set lds_k_pad and the epilogue."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="mxsweep_k256_w2x2",
        tile=TileSpec(tile_m=256, tile_n=256, tile_k=256, warp_m=2, warp_n=2, warp_k=1,
                      warp_tile_m=16, warp_tile_n=16, warp_tile_k=128),
        trait=TraitSpec(pipeline="mem", scheduler="intrawave", epilogue=os.environ.get("ROCKE_EXP_SWEEP_EPILOGUE", "cshuffle"), tdm=True,
                        tdm_depth=2, lds_k_pad=int(os.environ.get("ROCKE_EXP_SWEEP_LDS_PAD", "48")), pad_m=False, pad_n=False, pad_k=False,
                        split_barrier=True, tdm_alternate_issuers=True, tdm_early_refill=True),
        data=DataSpec(dtype_a="fp8e4m3", dtype_b="fp8e4m3", dtype_c="bf16",
                      dtype_acc="fp32", layout="RCR"),
        wave_size=32,
    )


# ------------------------------------------------------------------ building


def _build_one(job):
    """Worker: compile one kernel with its env; return (tag, hsaco path, name, stats, error)."""
    tag, env, out = job
    for k in [k for k in os.environ if k.startswith("ROCKE_EXP_")]:
        os.environ.pop(k)
    os.environ.update(env)
    try:
        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.gemm_universal import build_universal_gemm

        kdef = build_universal_gemm(_spec(), arch="gfx1250")
        art = compile_kernel(kdef, arch="gfx1250", backend="python", capture_ir_text=False)
        path = Path(out) / "hsaco" / f"{tag}.hsaco"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(art.hsaco)
        return tag, str(path), kdef.name, _notes(path), None
    except Exception as e:  # report and keep sweeping
        return tag, None, None, None, f"{type(e).__name__}: {e}"


def _notes(path: Path):
    readelf = shutil.which("llvm-readelf")
    if not readelf and os.environ.get("ROCM_PATH"):
        readelf = str(Path(os.environ["ROCM_PATH"]) / "llvm" / "bin" / "llvm-readelf")
    if not readelf:
        raise RuntimeError("llvm-readelf not found: put ROCm's llvm/bin on PATH or set ROCM_PATH")
    txt = subprocess.run([readelf, "--notes", str(path)], capture_output=True, text=True).stdout
    get = lambda key: int((re.search(rf"\.{key}:\s+(\d+)", txt) or [0, 0])[1])
    return {"vgpr": get("vgpr_count"), "sgpr": get("sgpr_count"),
            "vspill": get("vgpr_spill_count"), "sspill": get("sgpr_spill_count")}


# ------------------------------------------------------------- GPU picking


def _amdsmi(args):
    exe = shutil.which("amd-smi")
    if not exe:
        raise RuntimeError("amd-smi not found")
    r = subprocess.run(["timeout", "30", exe] + args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"amd-smi {' '.join(args)} failed or timed out")
    return r.stdout


def _hip_index_map():
    """amd-smi GPU id -> HIP index by PCI bus id; the HIP query runs in its own process."""
    bdf_of = {}
    for line in _amdsmi(["list", "--csv"]).splitlines()[1:]:
        f = line.split(",")
        if len(f) >= 2 and f[0].strip().isdigit():
            bdf_of[f[0].strip()] = f[1].strip().lower()
    code = (
        "import ctypes,ctypes.util,os,sys\n"
        "c=[os.path.join(os.environ[v],'lib','libamdhip64.so') for v in ('ROCM_PATH','ROCM_HOME') if v in os.environ]\n"
        "c+=[ctypes.util.find_library('amdhip64') or 'libamdhip64.so']\n"
        "h=None\n"
        "for p in c:\n"
        "  try:\n    h=ctypes.CDLL(p); break\n  except OSError: pass\n"
        "for i in range(int(sys.argv[1])):\n"
        " b=ctypes.create_string_buffer(64); h.hipDeviceGetPCIBusId(b,64,i); print(i,b.value.decode().lower())\n"
    )
    r = subprocess.run(["timeout", "60", sys.executable, "-c", code, str(len(bdf_of))],
                       capture_output=True, text=True)
    hip_of = {b: int(i) for i, b in (l.split() for l in r.stdout.splitlines())}
    return {g: hip_of[b] for g, b in bdf_of.items() if b in hip_of}


def _busy(gmap, own_pid=None, detail=None):
    """HIP indices where another process is computing right now (one amd-smi process sample).

    A process counts only with cu_occupancy > 0: one that merely holds memory does not slow
    our kernels. Columns are found by header name. detail (a list) collects the culprits."""
    lines = _amdsmi(["process", "--csv"]).splitlines()
    if not lines:
        return set()
    head = [h.strip() for h in lines[0].split(",")]
    col = {h: i for i, h in enumerate(head)}
    busy = set()
    if "pid" not in col or "gpu" not in col:
        return busy  # no process running: amd-smi then prints just "gpu,process_info"
    for line in lines[1:]:
        f = [x.strip() for x in line.split(",")]
        if len(f) < len(head) or not f[col["pid"]].isdigit() or f[col["gpu"]] not in gmap:
            continue
        if own_pid is not None and int(f[col["pid"]]) == own_pid:
            continue
        occ = f[col["cu_occupancy"]] if "cu_occupancy" in col else "1"
        if not occ.isdigit() or int(occ) > 0:
            busy.add(gmap[f[col["gpu"]]])
            if detail is not None:
                detail.append(f"HIP {gmap[f[col['gpu']]]} pid {f[col['pid']]} "
                              f"{f[col['name']].split('/')[-1] if 'name' in col else '?'} cu_occ {occ}")
    return busy


def _pick_gpu(o):
    gmap = _hip_index_map()
    if not gmap:
        sys.exit("--gpu auto: could not map amd-smi GPUs to HIP indices; pass --gpu N")
    deadline = time.time() + o.wait_free * 60
    while True:
        got = _pick_once(o, gmap)
        if got is not None:
            return got
        if time.time() >= deadline:
            sys.exit("--gpu auto: no free pair" + ("" if o.allow_single else " (--allow-single takes one)")
                     + (f" after waiting {o.wait_free} min" if o.wait_free else " (--wait-free N waits)"))
        _log(f"--gpu auto: no free pair; checking again in {o.wait_interval} s "
             f"(until {time.strftime('%H:%M', time.localtime(deadline))})")
        time.sleep(o.wait_interval)


def _pick_once(o, gmap):
    """One sampling pass: (gpu, pair, gmap, single), or None when nothing is free."""
    counts = {}
    for s in range(o.free_samples):
        for i in _busy(gmap):
            counts[i] = counts.get(i, 0) + 1
        if s + 1 < o.free_samples:
            time.sleep(o.free_interval)
    pairs = [tuple(int(x) for x in p.split(",")) for p in o.pairs.split()]
    for p in pairs:
        _log(f"  pair {p}: " + ", ".join(
            f"HIP {i} " + (f"busy ({counts[i]} samples)" if i in counts else "free") for i in p))
    for p in pairs:
        if all(i in gmap.values() and i not in counts for i in p):
            _log(f"--gpu auto: pair {p} free, running on HIP {p[0]}")
            return p[0], p, gmap, False
    if o.allow_single:
        for p in pairs:
            for i in p:
                if i in gmap.values() and i not in counts:
                    _log(f"--gpu auto: no free pair; SINGLE partition HIP {i} (quick check only)")
                    return i, (i,), gmap, True
    return None


# ------------------------------------------------------------ running on GPU


def _swz(np, x, rows, reps=8):
    """Swizzled scale rows (ROCKE_EXP_MX_SWZ=1): in each 16*reps-row block, row 16*r+l -> reps*l+r."""
    nb = rows // (16 * reps)
    return x.reshape(nb, reps, 16, -1).transpose(0, 2, 1, 3).reshape(rows, -1)


class _Problem:
    """A/B/C and scale buffers for one shape; rand data and scales loaded."""

    def __init__(self, rt, shape, data="rand"):
        import numpy as np

        from rocke.examples.gfx1250.gemm_perf_fp8.fp8_data import rand_operands

        self.np, self.rt, self.shape = np, rt, shape
        m, n, k = shape
        self.a = rt.alloc(m * k)
        self.b = rt.alloc(n * k)
        self.c = rt.alloc(m * n * 2)
        self.sa = rt.alloc(m * k // 32)
        self.sb = rt.alloc(n * k // 32)
        if data == "mxgemm":
            # The timed inputs of the mx_gemm prototype's bench: uniform e4m3
            # codes with the exponent MSB clear (finite, |x| < 2), E8M0 scales
            # 2^-3..2^3, drawn in its order (A, B, A scales, B scales).
            rng = np.random.default_rng(0x32F4)
            a8, b8 = (np.bitwise_and(rng.integers(0, 256, (r, k), dtype=np.uint8), 0xBF) for r in (m, n))
            self._h2d(self.a, a8)
            self._h2d(self.b, b8)
            sa = rng.integers(124, 131, (m, k // 32), dtype=np.uint8)
            sb = rng.integers(124, 131, (k // 32, n), dtype=np.uint8).T.copy()
            self.scales = (sa, sb)
            self.swz = None
            return
        a8, b8, _, _ = rand_operands(np, m, n, k)
        self._h2d(self.a, a8)
        self._h2d(self.b, b8)
        rng = np.random.default_rng(7)
        self.scales = (rng.integers(124, 131, size=(m, k // 32), dtype=np.uint8),
                       rng.integers(124, 131, size=(n, k // 32), dtype=np.uint8))
        self.swz = None

    def _h2d(self, dev, arr):
        arr = self.np.ascontiguousarray(arr)
        self.rt.memcpy_h2d(dev, (ctypes.c_uint8 * arr.nbytes).from_buffer(arr), arr.nbytes)

    def put_scales(self, sa, sb, swz):
        np = self.np
        m, n, k = self.shape
        a, b = (_swz(np, sa, m), _swz(np, sb, n)) if swz else (sa, sb)
        self._h2d(self.sa, a.reshape(m, k // 128, 4).transpose(1, 0, 2).copy())
        self._h2d(self.sb, b.reshape(n, k // 128, 4).transpose(1, 0, 2).copy())
        self.swz = swz

    def args(self):
        m, n, k = self.shape
        return (struct.pack("<QQQiii", self.a, self.b, self.c, m, n, k)
                + struct.pack("<iQQ", 0, self.sa, self.sb))

    def free(self):
        for p in (self.a, self.b, self.c, self.sa, self.sb):
            self.rt.free(p)


def _verify(rt, fn, grid, block, cluster, swz, mn):
    """Exact (integer set) and rand-bound checks at K = 8192."""
    import numpy as np

    from rocke.examples.gfx1250.gemm_perf_fp8.fp8_data import (
        int_operands,
        rand_operands,
        rand_rel_error_bound,
    )
    from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

    m, n = mn
    k = 8192
    p = _Problem.__new__(_Problem)
    p.np, p.rt, p.shape = np, rt, (m, n, k)
    p.a, p.b, p.c = rt.alloc(m * k), rt.alloc(n * k), rt.alloc(m * n * 2)
    p.sa, p.sb = rt.alloc(m * k // 32), rt.alloc(n * k // 32)

    def run():
        rt.memset(p.c, 0, m * n * 2)
        rt.launch(fn, grid, block, p.args(), cluster=cluster)
        rt.wait_stream(0)
        buf = (ctypes.c_uint8 * (m * n * 2))()
        rt.memcpy_d2h(buf, p.c, m * n * 2)
        return (np.frombuffer(bytes(buf), dtype=np.uint16).astype(np.uint32) << 16).view(
            np.float32).reshape(m, n)

    def deq(x, s):
        return x * np.repeat(np.exp2(s.astype(np.float64) - 127.0), 32, axis=1)

    rng = np.random.default_rng(7)
    a8, b8, af, bf = int_operands(np, m, n, k)
    p._h2d(p.a, a8)
    p._h2d(p.b, b8)
    sa = rng.integers(127, 129, size=(m, k // 32), dtype=np.uint8)
    sb = rng.integers(127, 129, size=(n, k // 32), dtype=np.uint8)
    p.put_scales(sa, sb, swz)
    ref = bf16_to_f32(f32_to_bf16((deq(af.astype(np.float64), sa)
                                   @ deq(bf.astype(np.float64), sb).T).astype(np.float32)))
    exact = bool(np.array_equal(run(), ref))
    a8, b8, af, bf = rand_operands(np, m, n, k)
    p._h2d(p.a, a8)
    p._h2d(p.b, b8)
    sa = rng.integers(124, 131, size=(m, k // 32), dtype=np.uint8)
    sb = rng.integers(124, 131, size=(n, k // 32), dtype=np.uint8)
    p.put_scales(sa, sb, swz)
    ref = deq(af.astype(np.float64), sa) @ deq(bf.astype(np.float64), sb).T
    rel = float(np.max(np.abs(run() - ref)) / np.max(np.abs(ref)))
    p.free()
    return exact, rel, rel < rand_rel_error_bound()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--gpu", required=True, help="HIP device index, or auto")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 4)))
    ap.add_argument("--verify-top", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="build and list; no GPU runs")
    ap.add_argument("--no-build", action="store_true",
                    help="reuse the kernels an earlier run built into this --out (stale if the emitter changed)")
    ap.add_argument("--pairs", default="6,7 0,1 2,3 4,5")
    ap.add_argument("--free-samples", type=int, default=6)
    ap.add_argument("--free-interval", type=float, default=10.0)
    ap.add_argument("--allow-single", action="store_true")
    ap.add_argument("--wait-free", type=float, default=0,
                    help="--gpu auto: keep checking up to this many minutes for a free pair (0: no wait)")
    ap.add_argument("--wait-interval", type=float, default=60, help="--wait-free: seconds between checks")
    ap.add_argument("--no-pair-check", action="store_true", help="do not tag busy batches")
    ap.add_argument("--data", choices=("rand", "mxgemm"), default="rand",
                    help="timed inputs: rand (normal(0, 0.1) -> e4m3) or the mx_gemm bench's (uniform codes, |x| < 2)")
    o = ap.parse_args()

    cfg = json.loads(o.config.read_text())
    out = o.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = cfg.get("base_env", {})
    jobs, launch_cluster = [], {}
    for v in cfg["variants"]:
        for a in cfg["arms"]:
            tag = f"{v['name']}|{a['name']}"
            env = dict(base, **v.get("env", {}), **a.get("env", {}))
            if a.get("cluster"):
                env["ROCKE_EXP_CLUSTER"] = a["cluster"]
                cx, cy = (int(x) for x in a["cluster"].split(","))
                launch_cluster[tag] = (cx, cy, 1)
            else:
                launch_cluster[tag] = None
            jobs.append((tag.replace("|", "__"), env, str(out)))

    built = {}
    manifest = out / "built.json"
    if o.no_build:
        # Reuse an earlier build of this --out: only the kernels this config names.
        if not manifest.exists():
            sys.exit(f"--no-build: {manifest} not found (build once with this --out first)")
        saved = json.loads(manifest.read_text())
        want = {tag.replace("__", "|") for tag, _, _ in jobs}
        missing = sorted(want - set(saved))
        for t in missing:  # failed or spilled when built: skipped again
            _log(f"  SKIP {t}: not in {manifest.name} (failed or spilled when built)")
        for t in sorted(want - set(missing)):
            path, name, st = saved[t]
            if not Path(path).exists():
                sys.exit(f"--no-build: {path} is gone")
            built[t] = (path, name, st)
            _log(f"  reuse {t}: vgpr={st['vgpr']} sgpr={st['sgpr']}")
    else:
        _log(f"building {len(jobs)} kernel(s), {o.jobs} at a time")
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(o.jobs, mp_context=ctx) as ex:
            for tag, path, name, st, err in ex.map(_build_one, jobs):
                t = tag.replace("__", "|")
                if err:
                    _log(f"  FAIL {t}: {err}")
                elif st["vspill"] or st["sspill"]:
                    _log(f"  SKIP {t}: spills (vgpr {st['vspill']}, sgpr {st['sspill']})")
                else:
                    built[t] = (path, name, st)
                    _log(f"  {t}: vgpr={st['vgpr']} sgpr={st['sgpr']}")
        saved = json.loads(manifest.read_text()) if manifest.exists() else {}
        saved.update(built)
        manifest.write_text(json.dumps(saved, indent=1) + "\n")
    swz_of = {t: (dict(base, **v.get("env", {})).get("ROCKE_EXP_MX_SWZ") == "1")
              for v in cfg["variants"] for t in [f"{v['name']}|{a['name']}" for a in cfg["arms"]]}
    if o.dry_run or not built:
        return

    gpu, pair, gmap, single = o.gpu, None, None, False
    if o.gpu == "auto":
        gpu, pair, gmap, single = _pick_gpu(o)
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)  # before the runtime starts

    from rocke.examples.gfx1250.gemm_perf_fp8.fp8_gemm_sweep import _launch_geometry
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    mods = {t: rt.load_module(Path(p).read_bytes()) for t, (p, _, _) in built.items()}
    fns = {t: mods[t].get_function(built[t][1]) for t in built}
    spec = _spec()
    rounds, warm, rep = cfg.get("rounds", 5), cfg.get("warmup", 50), cfg.get("repeat", 100)
    tags = list(built)
    res = {}
    with open(out / "results.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["kernel", "M", "N", "K", "round", "tflops", "pair_busy"])
        for shape in cfg["shapes"]:
            shape = tuple(shape)
            prob = _Problem(rt, shape, o.data)
            grid, block = _launch_geometry(spec, shape)
            fl = 2.0 * shape[0] * shape[1] * shape[2]
            for r in range(rounds):
                for t in (tags if r % 2 == 0 else tags[::-1]):
                    if prob.swz != swz_of[t]:
                        prob.put_scales(*prob.scales, swz_of[t])
                    rt.wait_stream(0)
                    time.sleep(1.0)
                    tainted = False
                    if pair and not o.no_pair_check:
                        why = []
                        try:
                            tainted = bool(_busy(gmap, os.getpid(), why) & set(pair))
                        except RuntimeError:
                            tainted, why = True, ["amd-smi failed"]
                        if tainted:
                            _log("  pair busy: " + "; ".join(w for w in why if any(f"HIP {i} " in w for i in pair) or w == "amd-smi failed"))
                    args = prob.args()
                    for _ in range(warm):
                        rt.launch(fns[t], grid, block, args, cluster=launch_cluster[t])
                    rt.wait_stream(0)
                    e0, e1 = rt.event(), rt.event()
                    e0.record()
                    for _ in range(rep):
                        rt.launch(fns[t], grid, block, args, cluster=launch_cluster[t])
                    e1.record()
                    e1.synchronize()
                    tf = fl / (e0.elapsed_to(e1) / rep * 1e9)
                    if pair and not o.no_pair_check and not tainted:
                        # A neighbour that started during the batch: check again after it.
                        why = []
                        try:
                            tainted = bool(_busy(gmap, os.getpid(), why) & set(pair))
                        except RuntimeError:
                            tainted, why = True, ["amd-smi failed"]
                        if tainted:
                            _log("  pair busy after the batch: " + "; ".join(
                                w for w in why if any(f"HIP {i} " in w for i in pair) or w == "amd-smi failed"))
                    res.setdefault((t, shape), []).append(None if tainted else tf)
                    w.writerow([t, *shape, r + 1, f"{tf:.1f}", int(tainted)])
                    fcsv.flush()
                    _log(f"r{r + 1} {shape} {t}: {tf:.0f}" + ("  [pair busy: not ranked]" if tainted else ""))
            prob.free()

    best = []
    for shape in cfg["shapes"]:
        shape = tuple(shape)
        rows = []
        for t in tags:
            v = [x for x in res.get((t, shape), []) if x is not None]
            if v:
                rows.append((statistics.median(v), min(v), max(v), len(v), t))
        rows.sort(key=lambda x: -x[0])
        _log(f"\n== {shape[0]}x{shape[1]}x{shape[2]}: median TF over clean rounds of {rounds} on HIP {gpu}"
             + ("  [SINGLE partition: quick check only]" if single else ""))
        top = rows[0][0] if rows else 1.0
        for med, lo, hi, n_ok, t in rows:
            _log(f"  {med:8.1f}  ({lo:.0f}-{hi:.0f})  {med / top:.3f}x  {t}  [{n_ok}/{rounds} clean]")
        best += [(shape, t) for *_, t in rows[: o.verify_top]]
    if o.verify_top:
        _log("\n== verify (K = 8192) of the best kernels")
        for shape, t in best:
            grid, block = _launch_geometry(spec, (shape[0], shape[1], 8192))
            exact, rel, ok = _verify(rt, fns[t], grid, block, launch_cluster[t], swz_of[t], shape[:2])
            _log(f"  {t} {shape[0]}x{shape[1]}: int_exact={exact} rand_rel={rel:.1e} "
                 + ("PASS" if exact and ok else "FAIL"))
    for m in mods.values():
        m.unload()


if __name__ == "__main__":
    main()
