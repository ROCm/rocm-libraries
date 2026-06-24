#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
r"""Comprehensive gemm_decode shape-catalog benchmark + head-to-head.

For every (model, layer, dtype, TP) decode GEMM in docs/gemm_decode_shape_catalog.md
this driver runs, on the SAME (M,N,K) inputs:

  * gemm_decode  - the autotuned CK-Tile best config (us + the winning cfg), via the
                   built sweep exes (bench_gemm_decode_msweep / *_fp8) or the blockscale
                   single-config exe.
  * skinny       - the AITER hand kernel it replaces: wvSpltK (bf16) / wvSplitKQ (fp8 pt).
                   (block-scale has no skinny peer -> N/A.)
  * aiter_tuned  - AITER's production library GEMM (tgemm.mm / gemm_a8w8 /
                   gemm_a8w8_blockscale), called directly. The aiter logger is captured;
                   if the shape is NOT in the tuned CSV ("not found tuned config ... will
                   use default config") the cell is flagged UNTUNED but still reported.

Emits a unified CSV (one row per (model,layer,tp,M,N,K,dtype)) and a grouped Markdown
report with the best time + winner per shape.

The MFMA M=16 ceiling (FlyDSL / gemm_quant) is intentionally NOT auto-run here -- it is a
different "what is the M>=5 roofline" question and is already produced per-(N,K) by the
sibling flydsl_msweep.py / gemm_quant_tensor_msweep.py; join their CSVs on (M,N,K) if you
want that column. This driver focuses on the two production baselines (skinny + aiter).

Note: the five catalogued models are BF16 or FP8 block-scale only -- none use per-tensor
FP8 -- so the fp8_pt engines (wvSplitKQ / gemm_a8w8 / bench_gemm_decode_msweep_fp8) are
wired and selectable but not exercised by the default catalog.

Design notes:
  * "aiter_direct": AITER is called in-process (module_custom.so for skinny; the aiter
    package for the tuned path, with the mx_types import shim from wvsplitk_msweep.py).
  * Resumable: rows are appended to the CSV as they complete; rerun skips shapes already
    present (unless --no-resume).
  * The autotuners are expensive; default M set is {1,2,4} and the FP8 fat-WG search is
    OFF (GD_BENCH_PERSIST=0). Use --mmax / --full to widen.

Example:
  /opt/venv/bin/python3 bench_catalog.py --build-dir build \
      --models qwen3_next,minimax_m2,kimi_k2 --tp 1,8 --out /tmp/catalog
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Catalog                                                                      #
# --------------------------------------------------------------------------- #

# dtype tags: 'bf16' (-> wvSplitK + tgemm), 'fp8_pt' (-> wvSplitKQ + gemm_a8w8),
#             'fp8_block' (-> none-skinny + gemm_a8w8_blockscale).
BF16, FP8_PT, FP8_BLOCK = "bf16", "fp8_pt", "fp8_block"


@dataclass(frozen=True)
class Layer:
    """A base (TP1) projection. `shard` decides how N/K shrink with TP."""
    name: str
    N: int
    K: int
    dtype: str
    shard: str  # 'col' | 'row' | 'rep' | 'vocab' | 'qkv'
    # for shard == 'qkv': (q_heads, kv_heads, head_dim)
    heads: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True)
class Shape:
    model: str
    layer: str
    dtype: str
    N: int
    K: int
    tp: int


def _shard(layer: Layer, tp: int) -> Tuple[int, int]:
    """Return per-rank (N, K) for the given TP degree."""
    N, K = layer.N, layer.K
    if layer.shard == "col" or layer.shard == "vocab":
        return _ceil_div(N, tp), K
    if layer.shard == "row":
        return N, _ceil_div(K, tp)
    if layer.shard == "rep":
        return N, K
    if layer.shard == "qkv":
        qh, kvh, hd = layer.heads  # type: ignore[misc]
        q = _ceil_div(qh, tp) * hd
        kv = max(1, _ceil_div(kvh, tp)) * hd  # kv replicated when kvh < tp
        return q + 2 * kv, K
    raise ValueError(layer.shard)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Per-model layer lists. Routed MoE experts are intentionally omitted (grouped
# fused_moe, not the skinny/decode path). Provenance: docs/gemm_decode_shape_catalog.md.
_MODELS: Dict[str, List[Layer]] = {
    # hidden 2048, 16 q / 2 kv heads, head_dim 256; BF16 unquantized.
    "qwen3_next": [
        Layer("full.qkv", 5120, 2048, BF16, "qkv", (16, 2, 256)),
        Layer("full.o", 2048, 4096, BF16, "row"),
        Layer("lin.in_qkvz", 12288, 2048, BF16, "col"),
        Layer("lin.in_ba", 64, 2048, BF16, "col"),
        Layer("lin.out", 2048, 4096, BF16, "row"),
        Layer("moe.router", 512, 2048, BF16, "rep"),
        Layer("moe.shared_gate_up", 1024, 2048, BF16, "col"),
        Layer("moe.shared_down", 2048, 512, BF16, "row"),
        Layer("lm_head", 151936, 2048, BF16, "vocab"),
    ],
    # hidden 7168, MLA 128 heads; FP8 block-scale (router/indexer/lm_head BF16).
    "deepseek_v32": [
        Layer("q_a", 1536, 7168, FP8_BLOCK, "rep"),
        Layer("q_b", 24576, 1536, FP8_BLOCK, "col"),
        Layer("kv_a", 576, 7168, FP8_BLOCK, "rep"),
        Layer("kv_b", 32768, 512, FP8_BLOCK, "col"),
        Layer("o", 7168, 16384, FP8_BLOCK, "row"),
        Layer("dense.gate_up", 36864, 7168, FP8_BLOCK, "col"),
        Layer("dense.down", 7168, 18432, FP8_BLOCK, "row"),
        Layer("shared.gate_up", 4096, 7168, FP8_BLOCK, "col"),
        Layer("shared.down", 7168, 2048, FP8_BLOCK, "row"),
        Layer("indexer.wq_b", 8192, 1536, FP8_BLOCK, "col"),
        Layer("router", 256, 7168, BF16, "rep"),
        Layer("lm_head", 129280, 7168, BF16, "vocab"),
    ],
    # hidden 7168, head_dim 512, o_lora 1024; FP8 block-scale dense, FP4 experts.
    "deepseek_v4": [
        Layer("fused_wqa_wkv", 2048, 7168, FP8_BLOCK, "rep"),
        Layer("wq_b", 65536, 1536, FP8_BLOCK, "col"),
        Layer("wo_b", 7168, 16384, FP8_BLOCK, "row"),
        Layer("indexer.wq_b", 8192, 1536, FP8_BLOCK, "rep"),
        Layer("shared.gate_up", 6144, 7168, FP8_BLOCK, "col"),
        Layer("shared.down", 7168, 3072, FP8_BLOCK, "row"),
        Layer("router", 384, 7168, BF16, "rep"),
        Layer("lm_head", 129280, 7168, BF16, "vocab"),
    ],
    # hidden 3072, 48 q / 8 kv heads; FP8 block-scale (gate/lm_head BF16).
    "minimax_m2": [
        Layer("qkv", 8192, 3072, FP8_BLOCK, "qkv", (48, 8, 128)),
        Layer("o", 3072, 6144, FP8_BLOCK, "row"),
        Layer("router", 256, 3072, BF16, "rep"),
        Layer("lm_head", 200064, 3072, BF16, "vocab"),
    ],
    # hidden 7168, MLA 64 heads; BF16 everywhere except INT4 experts (omitted).
    "kimi_k2": [
        Layer("q_a", 1536, 7168, BF16, "rep"),
        Layer("q_b", 12288, 1536, BF16, "col"),
        Layer("kv_a", 576, 7168, BF16, "rep"),
        Layer("kv_b", 16384, 512, BF16, "col"),
        Layer("o", 7168, 8192, BF16, "row"),
        Layer("dense.gate_up", 36864, 7168, BF16, "col"),
        Layer("dense.down", 7168, 18432, BF16, "row"),
        Layer("shared.gate_up", 4096, 7168, BF16, "col"),
        Layer("shared.down", 7168, 2048, BF16, "row"),
        Layer("router", 384, 7168, BF16, "rep"),
        Layer("lm_head", 163840, 7168, BF16, "vocab"),
    ],
}


def build_catalog(models: List[str], tps: List[int]) -> List[Shape]:
    out: List[Shape] = []
    seen = set()
    for model in models:
        for layer in _MODELS[model]:
            for tp in tps:
                N, K = _shard(layer, tp)
                key = (model, layer.name, tp)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Shape(model, layer.name, layer.dtype, N, K, tp))
    return out


# --------------------------------------------------------------------------- #
# gemm_decode side: drive the built sweep / single exes                        #
# --------------------------------------------------------------------------- #

# cfg column layout emitted by each sweep exe (after impl,M,N,K,time_us,tflops,gbytes_s).
_BF16_CFG = ["mp", "np", "kv", "kb", "swizzle", "chunk"]
_FP8_CFG = ["mp", "np", "kv", "kb", "swizzle", "chunk", "wpb", "a_lds", "streamb", "persist"]


@dataclass
class Cell:
    us: float = math.nan
    cfg: str = ""
    flag: str = ""  # e.g. UNTUNED, ERR, N/A


def _parse_best(stdout: str, best_impl: str, cfg_cols: List[str]) -> Dict[int, Cell]:
    """Pull the per-M `*_best` rows out of a sweep exe's stdout CSV."""
    out: Dict[int, Cell] = {}
    header: List[str] = []
    for line in stdout.splitlines():
        if line.startswith("impl,"):
            header = line.split(",")
            continue
        if not line.startswith(best_impl + ","):
            continue
        f = line.split(",")
        rec = dict(zip(header, f)) if header else {}
        m = int(rec.get("M", f[1]))
        us = float(rec.get("time_us", f[4]))
        cfg = "/".join(f"{c}{rec[c]}" for c in cfg_cols if c in rec)
        # The sweep prints a ~1e30 sentinel when no config passes IsSupported.
        if us >= 1e9:
            out[m] = Cell(flag="N/A:unsupported")
        else:
            out[m] = Cell(us=us, cfg=cfg or "?")
    return out


def run_gd_sweep(exe: Path, N: int, K: int, mmax: int, warmup: int, repeat: int,
                 fp8: bool, env: Dict[str, str]) -> Dict[int, Cell]:
    if not exe.is_file():
        return {}
    cmd = [str(exe), str(warmup), str(repeat), str(N), str(K), str(mmax)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False,
                             env={**os.environ, **env}, timeout=1800)
    except subprocess.TimeoutExpired:
        return {m: Cell(flag="TIMEOUT") for m in range(1, mmax + 1)}
    if res.returncode != 0:
        return {1: Cell(flag="ERR:" + (res.stderr.strip().splitlines() or ["?"])[-1][:60])}
    impl = "gemm_decode_fp8_best" if fp8 else "gemm_decode_best"
    return _parse_best(res.stdout, impl, _FP8_CFG if fp8 else _BF16_CFG)


_BLOCK_RE = re.compile(r"ms\s*=\s*([0-9.]+)")


def run_gd_blockscale(exe: Path, N: int, K: int, m: int, split_ks: List[int],
                      warmup: int, repeat: int) -> Cell:
    """Block-scale has a single compile-time config; only split_k is a runtime knob."""
    if not exe.is_file():
        return Cell(flag="N/A:exe")
    # Kernel requires K%128==0 and N%128==0 (X=Block2D<1,128>, W=Block2D<128,128>).
    if K % 128 != 0 or N % 128 != 0:
        return Cell(flag="N/A:block%128")
    best: Cell = Cell(flag="ERR")
    for sk in split_ks:
        cmd = [str(exe), f"-m={m}", f"-n={N}", f"-k={K}", f"-split_k={sk}",
               f"-warmup={warmup}", f"-repeat={repeat}", "-metric=0"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
        if res.returncode != 0:
            # Only record the rejection if no split_k has succeeded yet.
            if math.isnan(best.us):
                msg = (res.stdout + res.stderr).strip().splitlines()
                if msg:
                    best = Cell(flag="N/A:" + msg[-1][:48])
            continue
        mobj = _BLOCK_RE.search(res.stdout)
        if not mobj:
            continue
        us = float(mobj.group(1)) * 1000.0
        if math.isnan(best.us) or us < best.us:
            best = Cell(us=us, cfg=f"dsv3/split_k{sk}")
    return best


# --------------------------------------------------------------------------- #
# AITER side: skinny + tuned/library, called directly, with untuned capture    #
# --------------------------------------------------------------------------- #

_AITER_UNTUNED_RE = re.compile(r"not found tuned config|will use default config|non-tuned default")


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.buf: List[str] = []

    def emit(self, record):
        self.buf.append(record.getMessage())

    def untuned(self) -> bool:
        return any(_AITER_UNTUNED_RE.search(m) for m in self.buf)

    def reset(self):
        self.buf.clear()


class Aiter:
    """Lazy AITER access. module_custom.so for skinny; aiter pkg for the tuned path."""

    def __init__(self, so_path: str, aiter_dir: str, cu: int):
        self.so_path, self.aiter_dir, self.cu = so_path, aiter_dir, cu
        self._custom = None
        self._pkg = None
        self._cap = _LogCapture()
        logging.getLogger("aiter").addHandler(self._cap)
        logging.getLogger("aiter").setLevel(logging.INFO)

    # -- module loaders ------------------------------------------------------ #
    def custom(self):
        if self._custom is None:
            import torch  # noqa: F401  (resolve libtorch symbols first)
            spec = importlib.util.spec_from_file_location("module_custom", self.so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._custom = mod
        return self._custom

    def pkg(self):
        if self._pkg is None:
            self._pkg = _import_aiter_pkg(self.aiter_dir)
        return self._pkg

    # -- timing -------------------------------------------------------------- #
    def _time(self, run, warmup, repeat) -> float:
        import torch
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(repeat):
            run()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000.0 / repeat

    # -- skinny -------------------------------------------------------------- #
    def skinny(self, dtype, N, K, m, A, B, Aq, Bq, xs, ws, warmup, repeat) -> Cell:
        import torch
        try:
            mod = self.custom()
            if dtype == BF16:
                out = torch.empty((m, N), dtype=A.dtype, device=A.device)
                run = lambda: mod.wvSpltK(B, A[:m].contiguous(), out, m, self.cu)
                tag = "wvSpltK"
            elif dtype == FP8_PT:
                out = torch.empty((m, N), dtype=torch.bfloat16, device=A.device)
                run = lambda: self.pkg().wvSplitKQ(Bq, Aq[:m].contiguous(), out, ws, xs, self.cu)
                tag = "wvSplitKQ"
            else:
                return Cell(flag="N/A")
            run()
            torch.cuda.synchronize()
            return Cell(us=self._time(run, warmup, repeat), cfg=f"{tag}/cu{self.cu}")
        except Exception as ex:  # noqa: BLE001
            return Cell(flag="ERR:" + str(ex)[:50])

    # -- tuned / library ----------------------------------------------------- #
    def tuned(self, dtype, N, K, m, A, B, Aq, Bq, xs, ws, xsb, wsb, warmup, repeat) -> Cell:
        import torch
        try:
            pkg = self.pkg()
            self._cap.reset()
            if dtype == BF16:
                from aiter.tuned_gemm import tgemm
                run = lambda: tgemm.mm(A[:m].contiguous(), B, bias=None)
                tag = "tgemm"
            elif dtype == FP8_PT:
                run = lambda: pkg.gemm_a8w8(Aq[:m].contiguous(), Bq, xs, ws, None, torch.bfloat16)
                tag = "gemm_a8w8"
            else:
                run = lambda: pkg.gemm_a8w8_blockscale(Aq[:m].contiguous(), Bq, xsb, wsb,
                                                       torch.bfloat16)
                tag = "gemm_a8w8_blockscale"
            run()
            torch.cuda.synchronize()
            untuned = self._cap.untuned()
            us = self._time(run, warmup, repeat)
            return Cell(us=us, cfg=tag, flag="UNTUNED" if untuned else "")
        except Exception as ex:  # noqa: BLE001
            return Cell(flag="ERR:" + str(ex)[:50])


def _import_aiter_pkg(aiter_dir: str):
    """Import `aiter`, splicing the int-mirror Mx enums so a stale mx_types .so
    cannot block the per-tensor / blockscale FP8 entry points (see wvsplitk_msweep.py)."""
    import importlib.abc
    if aiter_dir and aiter_dir not in sys.path:
        sys.path.insert(0, aiter_dir)

    class _Wrap(importlib.abc.Loader):
        def __init__(self, real):
            self.real = real

        def create_module(self, spec):
            return self.real.create_module(spec)

        def exec_module(self, module):
            self.real.exec_module(module)
            for name in ("MxScaleRoundMode", "MxDtype"):
                if not hasattr(module, name) and hasattr(module, name + "Int"):
                    setattr(module, name, getattr(module, name + "Int"))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "aiter.utility.mx_types":
                return None
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    spec.loader = _Wrap(spec.loader)
                    return spec
            return None

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    import aiter  # noqa: E402
    return aiter


# --------------------------------------------------------------------------- #
# Inputs (built once per (dtype,N,K), sliced per M)                            #
# --------------------------------------------------------------------------- #

@dataclass
class Inputs:
    A: object = None
    B: object = None
    Aq: object = None
    Bq: object = None
    xs: object = None   # per-tensor x_scale
    ws: object = None   # per-tensor w_scale
    xsb: object = None  # block x_scale (M, K/128)
    wsb: object = None  # block w_scale (N/128, K/128)


def _blockscale_quant(t, bn: int, bk: int):
    """Quantize t (rows, cols) to fp8 with per-(bn x bk) block amax scales.
    Returns (q_fp8, scale_fp32[rows/bn, cols/bk])."""
    import torch
    from aiter import dtypes
    r, c = t.shape
    rb, cb = _ceil_div(r, bn), _ceil_div(c, bk)
    pad = torch.zeros((rb * bn, cb * bk), dtype=torch.float32, device=t.device)
    pad[:r, :c] = t.float()
    blk = pad.view(rb, bn, cb, bk)
    amax = blk.abs().amax(dim=(1, 3)).clamp_min(1e-6)  # (rb, cb)
    finfo = torch.finfo(dtypes.fp8)
    scale = amax / finfo.max
    q = (blk / scale[:, None, :, None]).clamp(finfo.min, finfo.max)
    q = q.view(rb * bn, cb * bk)[:r, :c].to(dtypes.fp8)
    return q.contiguous(), scale.contiguous()


def make_inputs(dtype: str, N: int, K: int, mmax: int, aiter: "Aiter") -> Inputs:
    import torch
    dev = torch.device("cuda")
    inp = Inputs()
    inp.A = torch.randn((mmax, K), dtype=torch.bfloat16, device=dev)
    inp.B = (torch.randn((N, K), dtype=torch.bfloat16, device=dev) * 0.1)
    if dtype == FP8_PT:
        pkg = aiter.pkg()
        from aiter import dtypes
        inp.Aq, inp.xs = pkg.per_tensor_quant(inp.A, quant_dtype=dtypes.fp8)
        inp.Bq, inp.ws = pkg.per_tensor_quant(inp.B, quant_dtype=dtypes.fp8)
    elif dtype == FP8_BLOCK:
        aiter.pkg()  # ensure dtypes import works
        inp.Aq, inp.xsb = _blockscale_quant(inp.A, 1, 128)
        inp.Bq, inp.wsb = _blockscale_quant(inp.B, 128, 128)
    return inp


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

_FIELDS = ["model", "layer", "tp", "M", "N", "K", "dtype",
           "gd_us", "gd_cfg", "skinny_us", "aiter_us", "aiter_cfg", "aiter_flag",
           "best_engine", "best_us", "gd_vs_skinny", "gd_vs_aiter"]


def _exe(build: Path, name: str) -> Path:
    return build / "bin" / name


def eval_nk(dtype: str, N: int, K: int, m_list: List[int], args, aiter: "Aiter",
            build: Path) -> Dict[int, Dict[str, Cell]]:
    """Run every engine for one (dtype,N,K) across the M list. Returns {M: {engine: Cell}}."""
    import torch
    mmax = max(m_list)
    res: Dict[int, Dict[str, Cell]] = {m: {} for m in m_list}

    # gemm_decode (autotuned best per M)
    gd: Dict[int, Cell] = {}
    if dtype == BF16:
        gd = run_gd_sweep(_exe(build, "bench_gemm_decode_msweep"), N, K, mmax,
                          args.warmup, args.repeat, fp8=False, env={})
    elif dtype == FP8_PT:
        gd = run_gd_sweep(_exe(build, "bench_gemm_decode_msweep_fp8"), N, K, mmax,
                          args.warmup, args.repeat, fp8=True,
                          env={"GD_BENCH_PERSIST": "1" if args.full else "0"})
    elif dtype == FP8_BLOCK:
        exe = _exe(build, "benchmark_gemm_decode_blockscale_fp8_smallm_dsv3")
        for m in m_list:
            gd[m] = run_gd_blockscale(exe, N, K, m, args.split_k, args.warmup, args.repeat)

    # AITER inputs (skip if AITER disabled or gd-only)
    inp = None
    if not args.gd_only:
        try:
            inp = make_inputs(dtype, N, K, mmax, aiter)
        except Exception as ex:  # noqa: BLE001
            inp = None
            print(f"  ! input build failed ({dtype} N={N} K={K}): {str(ex)[:80]}",
                  file=sys.stderr)

    for m in m_list:
        res[m]["gd"] = gd.get(m, Cell(flag="N/A"))
        if inp is None:
            res[m]["skinny"] = Cell(flag="N/A")
            res[m]["aiter"] = Cell(flag="N/A")
            continue
        res[m]["skinny"] = aiter.skinny(dtype, N, K, m, inp.A, inp.B, inp.Aq, inp.Bq,
                                        inp.xs, inp.ws, args.warmup, args.repeat)
        res[m]["aiter"] = aiter.tuned(dtype, N, K, m, inp.A, inp.B, inp.Aq, inp.Bq,
                                      inp.xs, inp.ws, inp.xsb, inp.wsb,
                                      args.warmup, args.repeat)
    return res


def _row(shape: Shape, m: int, cells: Dict[str, Cell]) -> Dict[str, object]:
    gd, sk, ai = cells["gd"], cells["skinny"], cells["aiter"]
    cands = [("gemm_decode", gd.us), ("skinny", sk.us), ("aiter", ai.us)]
    cands = [(k, v) for k, v in cands if not math.isnan(v)]
    best_engine, best_us = min(cands, key=lambda kv: kv[1]) if cands else ("none", math.nan)

    def _spd(other: float) -> str:
        if math.isnan(gd.us) or math.isnan(other) or gd.us == 0:
            return ""
        return f"{other / gd.us:.2f}x"

    def _u(c: Cell) -> str:
        return f"{c.us:.2f}" if not math.isnan(c.us) else (c.flag or "")

    return {
        "model": shape.model, "layer": shape.layer, "tp": shape.tp,
        "M": m, "N": shape.N, "K": shape.K, "dtype": shape.dtype,
        "gd_us": _u(gd), "gd_cfg": gd.cfg or gd.flag,
        "skinny_us": _u(sk),
        "aiter_us": _u(ai), "aiter_cfg": ai.cfg or "", "aiter_flag": ai.flag,
        "best_engine": best_engine, "best_us": f"{best_us:.2f}" if not math.isnan(best_us) else "",
        "gd_vs_skinny": _spd(sk.us), "gd_vs_aiter": _spd(ai.us),
    }


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_model.setdefault(str(r["model"]), []).append(r)
    cols = ["layer", "tp", "M", "N", "K", "dtype", "gd_us", "gd_cfg",
            "skinny_us", "aiter_us", "aiter_flag", "best_engine",
            "gd_vs_skinny", "gd_vs_aiter"]
    lines = ["# gemm_decode catalog benchmark", "",
             "`gd_us` = gemm_decode autotuned best. `gd_vs_*` > 1.0x means gemm_decode is faster. "
             "`aiter_flag=UNTUNED` = shape missing from AITER's tuned CSV (ran default).", ""]
    for model in sorted(by_model):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in sorted(by_model[model], key=lambda x: (str(x["layer"]), x["tp"], x["M"])):
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/gemm_decode_catalog"),
                    help="output prefix; writes <out>.csv and <out>.md")
    ap.add_argument("--models", default=",".join(_MODELS),
                    help="comma list: " + ",".join(_MODELS))
    ap.add_argument("--tp", default="1,2,4,8", help="comma list of TP degrees")
    ap.add_argument("--dtypes", default=f"{BF16},{FP8_PT},{FP8_BLOCK}")
    ap.add_argument("--layers", default="",
                    help="comma list of layer-name substrings to keep (default: all)")
    ap.add_argument("--m-list", default="1,2,4", help="comma list of decode M values")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--split-k", type=int, nargs="+", default=[1, 2, 4],
                    help="block-scale split_k values to try")
    ap.add_argument("--full", action="store_true",
                    help="enable the FP8 fat-WG search (GD_BENCH_PERSIST=1); much slower")
    ap.add_argument("--gd-only", action="store_true", help="skip AITER (gemm_decode only)")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing CSV rows")
    ap.add_argument("--dry-run", action="store_true", help="print the catalog and exit")
    ap.add_argument("--so", default="/home/AMD/samremes/dev/aiter/aiter/jit/module_custom.so")
    ap.add_argument("--aiter-dir", default="/home/AMD/samremes/dev/aiter")
    ap.add_argument("--cu", type=int, default=0, help="0 = device multi_processor_count")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    tps = [int(x) for x in args.tp.split(",") if x]
    dtypes = set(args.dtypes.split(","))
    m_list = sorted({int(x) for x in args.m_list.split(",") if x})
    layer_subs = [x for x in args.layers.split(",") if x]
    catalog = [s for s in build_catalog(models, tps) if s.dtype in dtypes]
    if layer_subs:
        catalog = [s for s in catalog if any(sub in s.layer for sub in layer_subs)]

    if args.dry_run:
        for s in catalog:
            print(f"{s.model:14s} {s.layer:18s} tp{s.tp} {s.dtype:9s} N={s.N} K={s.K}")
        print(f"# {len(catalog)} shapes x {len(m_list)} M = {len(catalog) * len(m_list)} cells",
              file=sys.stderr)
        return 0

    csv_path = Path(str(args.out) + ".csv")
    md_path = Path(str(args.out) + ".md")
    rows: List[Dict[str, object]] = []
    done = set()
    if csv_path.exists() and not args.no_resume:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["model"], r["layer"], int(r["tp"]), int(r["M"])))
        print(f"# resume: {len(done)} cells already in {csv_path}", file=sys.stderr)

    import torch
    cu = args.cu or torch.cuda.get_device_properties(0).multi_processor_count
    aiter = Aiter(args.so, args.aiter_dir, cu)

    nk_cache: Dict[Tuple[str, int, int], Dict[int, Dict[str, Cell]]] = {}
    fh = csv_path.open("a" if (done and not args.no_resume) else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_FIELDS)
    if not (done and not args.no_resume):
        writer.writeheader()

    t0 = time.time()
    for i, s in enumerate(catalog):
        need = [m for m in m_list if (s.model, s.layer, s.tp, m) not in done]
        if not need:
            continue
        key = (s.dtype, s.N, s.K)
        if key not in nk_cache:
            print(f"[{i + 1}/{len(catalog)}] {s.model}/{s.layer} tp{s.tp} "
                  f"{s.dtype} N={s.N} K={s.K} ...", file=sys.stderr)
            nk_cache[key] = eval_nk(s.dtype, s.N, s.K, m_list, args, aiter, args.build_dir)
        for m in need:
            row = _row(s, m, nk_cache[key][m])
            writer.writerow(row)
            rows.append(row)
        fh.flush()
    fh.close()

    write_markdown(rows, md_path)
    print(f"# wrote {csv_path} and {md_path} ({len(rows)} rows, "
          f"{time.time() - t0:.0f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
