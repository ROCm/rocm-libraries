# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L3: the large-batch tiled dense decode GEMM core (DESIGN-LB E1), gfx950.

``experiments/qwen38-true-mono/DESIGN-LB.md`` 4.3, 6 (E1) and 12 (P-L3). ``Y[token, row] = sum_k X[token, k] W[row, k]``
for the dense projections of a large-batch step (rows = live tokens 1..bucket, bucket <= 128): the weight rows are
the MFMA A operand (straight from global memory to registers), the batch is the N dimension in tiles of 16 tokens
(``v_mfma_*_16x16x*``: C element ``i`` of lane ``l`` is weight row ``4 (l / 16) + i`` of token ``l % 16``), so every
weight byte is read once per step whatever the batch. Two precision regimes:

* ``w="bf16"``: bf16 W [M, K] x bf16 X [rows, K] on ``v_mfma_f32_16x16x32_bf16`` (FlyTape's tiled tape,
  ``flytape/blocks/tiled.py``: a lane holds 64 contiguous weight bytes of its row per 128-K chunk, slice ``m`` of the
  chunk takes K ``32 g + 8 m .. + 8`` of lane group ``g`` for both operands).
* ``w="mx"``: MXFP4 W (E2M1 codes [M, K/2], the even k in the low nibble; E8M0 scales [M, KSP] with one byte per
  32 K, row stride ``KSP`` = the scale count rounded up to 16 bytes, pad bytes 127) x MXFP8 X (E4M3 codes
  [rows, K], E8M0 scales [rows, KSP]) on ``v_mfma_scale_f32_16x16x128_f8f6f4`` (fp4 A, fp8 B): ATOM's W4A8.
  The producer of X writes it already quantized (DESIGN-LB S2 / S7 epilogues: MXFP8 pack).

A *unit* is (layer, row group of ``R = 16 waves mt`` weight rows, token block of ``tb`` tiles, K split ``s``): its
``kc = ceil(K / 128) / splits`` chunks run fully unrolled. The ``grid`` persistent CTAs walk their units (``map``):
``rr`` deals unit ``o`` to CTA ``o mod grid`` (row group fastest, then token block, then split: FlyTape's order);
``xcd`` gives XCD ``x`` (CTA ``c`` runs on XCD ``c mod 8``) the contiguous unit range ``[x U / 8, (x + 1) U / 8)``, so
the units that read the same activation slice (split, token block) and the token blocks that share a weight row
group stay on one XCD's L2. Weights (and, ``xsrc="global"``, the X operands) of chunk ``c + depth`` are issued
before chunk ``c``'s MFMAs, across unit boundaries (``xu``: the next unit's first chunks are issued during this
unit's last ones, so a CTA never drains its weight stream between units). With ``xsrc="lds"`` the activation chunk
is staged once per CTA in LDS (double buffered; loads ``xa`` chunks ahead in registers, one LDS-only barrier per
chunk: no ``vmcnt`` drain).

``splits > 1``: every unit stores f32 partials ``[split, token, row]`` (``sc1`` stores, P5's ``bulk=sc1``), waits for
them, and arrives on its (layer, row group, token block) counter (agent-scope atomic); the last arriver sums the
splits in split order (``sc1`` loads, groups of 8 loads in flight) and writes bf16 Y, then resets the counter.
Determinism: fixed-order MFMA chains (chunk order, slice order) and fixed-order split sums, no float atomics: every
knob that keeps ``splits`` gives the same bits (the bench checks it). ``stamp`` builds write per-CTA
``s_memrealtime`` start / end and the XCD id (informational). Mutants (must fail): ``kskip`` (the last chunk of every
unit adds nothing), ``scale`` (mx: every lane group uses scale byte 0), ``fold`` (the fold drops the last split).

Kernel arguments: i64 ``w, ws, x, xs, y, part, cnt, gs`` (+ ``stamps``), then i32 ``m`` (weight rows), ``rows`` (live
tokens <= bucket), ``layers`` (consecutive layers a launch walks: weights, Y, partials and counters are per layer;
X is shared), ``gsync`` (non-zero: an 8-stripe grid barrier between layers, ``gs`` its never-reset counters, and
activation loads of a later layer wait for it: phase semantics; zero: layers overlap). Y is ``[layers][bucket16][Mpad]``
bf16 with ``Mpad = ceil(m / R) R``; ``fold="none"`` leaves the f32 partials for the consumer instead.

Layouts (host-side, bytes unchanged): ``wlay="tile"`` stores weights as ``[M / 16][nq][16 rows x 128 K]`` in MFMA operand
order (mx: 1 KiB codes and 64 B scales per block); ``sop`` (mx) packs the block scales of four consecutive K chunks into
one dword per lane (weights ``[M / 16][nq / 4][16][4 block][4 chunk]``, activations ``[nq / 4][bucket16][4][4]``) and
selects the byte with MFMA ``op_sel``: a scaled MFMA whose scale register is rewritten for every MFMA runs about 8x slower
on gfx950 (P-L3), so the scale registers are written once per four chunks. Same bits as the default path.
Python engine only (``buffer_*_cp``, ``int_to_ptr``, ``mfma_scale_mx``); kernel names start with ``q38lb_``. Nothing
served imports it. Measurements: ``experiments/qwen38-true-mono/large-batch/P-L3.md``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields

from rocke.core.ir import (
    BF16,
    CPOL_NT,
    CPOL_SC1,
    F32,
    I32,
    I64,
    I8,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "CHUNK",
    "TILE",
    "GemmKnobs",
    "build_gemm_kernel",
    "build_stream_kernel",
    "gemm_params",
    "kernel_name",
    "ksp",
    "mpad",
    "StreamKnobs",
    "stream_name",
    "stream_params",
]

TILE = 16
CHUNK = 128
XCDS = 8
OOB = 0x7FFFFFF0  # buffer offset past any record a layer buffer holds (layers < 2 GiB): loads 0, stores dropped
FOLD_GROUP = 8  # partial loads in flight per fold step
MUTANTS = ("none", "kskip", "scale", "fold", "sopsel")  # sopsel: the op_sel path selects the next chunk's scale byte
WCPS = {"default": 0, "nt": CPOL_NT}


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def ksp(k: int) -> int:
    """Scale row stride in bytes (MXFP4 weights and MXFP8 activations): one byte per 32 K, rounded up to 16."""
    return -(-(-(-k // 32)) // 16) * 16


def mpad(m: int, knobs: "GemmKnobs") -> int:
    """Y / partial row stride: ``m`` rounded up to the row group."""
    r = knobs.rows_per_unit
    return -(-m // r) * r


@dataclass(frozen=True)
class GemmKnobs:
    """One build. Defaults = the P-L3 default arm; the kernel name lists only deviations from them."""

    w: str = "mx"  # "mx" (MXFP4 W x MXFP8 X) | "bf16" (bf16 W x bf16 X)
    k: int = 2560  # K
    bucket: int = 128  # rows bucket (token tiles = ceil(bucket / 16))
    waves: int = 8  # waves per CTA (4 | 8)
    mt: int = 1  # 16-row weight tiles per wave (1 | 2 | 4)
    splits: int = 1  # K splits (divides ceil(K / 128))
    tb: int = 0  # token tiles per unit (0 = the whole bucket)
    depth: int = 2  # chunks of weight loads in flight
    xa: int = 1  # activation chunks in flight in registers (xsrc="lds")
    xsrc: str = "lds"  # "lds" (staged per CTA) | "global" (every wave loads its B operands, no LDS, no barrier)
    xpad: int = 0  # LDS row pad in bytes (0 = 16)
    wcp: str = "default"  # weight cache policy: "default" | "nt"
    xslay: str = "row"  # mx X scale layout: "row" ([rows][KSP] bytes) | "chunk" ([nq][bucket16] dwords: a chunk's scales of
    #   every token contiguous, staged with one 16-byte load per 4 tokens)
    xscale: str = "lds"  # mx X scales: "lds" (staged with the codes) | "glb" (every wave loads its tiles' scale dwords from
    #   global memory with the weights, depth chunks ahead; needs xslay="chunk")
    xsb: str = "shift"  # mx X scale operand: "shift" (LDS dword of 4 block scales, shifted per lane group before the MFMA)
    #   | "byte" (staging unpacks the 4 bytes into 4 LDS dwords; a lane reads its block's dword: no VALU op feeds the MFMA)
    wsb: str = "shift"  # mx weight scale operand: "shift" (dword of 4 block scales shifted per lane group) | "byte" (each
    #   lane loads its block's scale byte zero-extended (global_load_ubyte): no VALU op writes the MFMA's scale register)
    sop: bool = False  # mx op_sel scales: weight and X scale dwords hold 4 consecutive chunks' block scales of the lane's
    #   block (layouts [M/16][nq/4][16][4 g][4 j] and [nq/4][bucket16][4 g][4 j]); loaded once per 4 chunks with the
    #   weights, the MFMA selects byte j = chunk mod 4 (needs wlay=tile, xscale=glb, kc % 4 == 0 or splits == 1)
    wlay: str = "row"  # weight layout: "row" (row-major, as stored) | "tile" (16-row x 128-K tiles in MFMA operand order)
    map: str = "rr"  # unit -> CTA map: "rr" | "xcd"
    xu: bool = True  # prefetch across unit boundaries
    grid: int = 256  # persistent CTAs
    occ1: bool = True  # pad LDS so one CTA fits a CU
    sb: bool = True  # sched_barrier fences around each chunk's issue block
    bpre: int = 0  # token tiles whose B operands are read ahead of the MFMAs (0 = the compiler's order)
    fold: str = "last"  # splits > 1: "last" (last arriver writes bf16 Y) | "none" (f32 partials out: the consumer folds)
    tstamp: bool = False  # diagnosis: s_memtime at the steps of this CTA's first 4 units (stamps[(cta * 4 + n) * 64 + p])
    diag: str = "none"  # diagnosis only: "nomfma" (VALU instead of MFMA) | "nob" (constant B operand, no LDS reads) |
    #   "noxs" (mx: X scales neither loaded nor staged, constant 1.0) | "nows" (mx: weight scales not loaded, constant 1.0) |
    #   "xs0" (mx: tile 0's X scale for every tile) | "xsu" (mx: X scale made wave-uniform, readfirstlane)
    stamp: bool = False
    mutant: str = "none"
    tag: str = ""  # recompiled-control name suffix

    # ---------------------------------------------------------------- derived
    @property
    def tiles(self) -> int:
        return -(-self.bucket // TILE)

    @property
    def tbt(self) -> int:
        return self.tb or self.tiles

    @property
    def ntb(self) -> int:
        return -(-self.tiles // self.tbt)

    @property
    def nq(self) -> int:
        return -(-self.k // CHUNK)

    @property
    def kc(self) -> int:
        return self.nq // self.splits

    @property
    def ragged(self) -> bool:
        return self.k % CHUNK != 0

    @property
    def rows_per_unit(self) -> int:
        return TILE * self.waves * self.mt

    @property
    def threads(self) -> int:
        return 64 * self.waves

    @property
    def pad(self) -> int:
        return self.xpad or 16

    def check(self) -> None:
        if self.w not in ("mx", "bf16"):
            raise ValueError("w mx | bf16")
        if self.k <= 0 or self.k % 32:
            raise ValueError("k must be a positive multiple of 32")
        if not 1 <= self.bucket <= 128:
            raise ValueError("bucket 1..128")
        if self.waves not in (4, 8):
            raise ValueError("waves 4 | 8")
        if self.mt not in (1, 2, 4):
            raise ValueError("mt 1 | 2 | 4")
        if self.splits < 1 or self.nq % self.splits:
            raise ValueError(f"splits {self.splits} must divide the {self.nq} K chunks")
        if self.tb < 0 or self.tb > self.tiles:
            raise ValueError("tb 0..tiles")
        if not 1 <= self.depth <= 8:
            raise ValueError("depth 1..8")
        if not 1 <= self.xa <= 4:
            raise ValueError("xa 1..4")
        if self.xsrc not in ("lds", "global"):
            raise ValueError("xsrc lds | global")
        if self.xpad % 16 or self.xpad < 0:
            raise ValueError("xpad a multiple of 16 bytes")
        if self.xslay not in ("row", "chunk"):
            raise ValueError("xslay row | chunk")
        if self.xslay == "chunk" and (self.w != "mx" or self.xsrc != "lds"):
            raise ValueError("xslay chunk: w=mx, xsrc=lds")
        if self.xsb not in ("shift", "byte", "dword"):
            raise ValueError("xsb shift | byte | dword")
        if self.xsb == "dword" and (self.xscale != "glb"):
            raise ValueError("xsb dword: xscale glb (X scales pre-unpacked to one dword per (chunk, token, block))")
        if self.xsb == "byte" and (self.w != "mx" or self.xsrc != "lds"):
            raise ValueError("xsb byte: w=mx, xsrc=lds")
        if self.wsb not in ("shift", "byte") or (self.wsb == "byte" and (self.w != "mx" or self.mutant == "scale")):
            raise ValueError("wsb shift | byte (mx, not with the scale mutant)")
        if self.xscale not in ("lds", "glb"):
            raise ValueError("xscale lds | glb")
        if self.xscale == "glb" and (self.xslay != "chunk" or self.diag == "noxs"):
            raise ValueError("xscale glb needs xslay chunk")
        if self.sop and not (self.w == "mx" and self.wlay == "tile" and self.xscale == "glb" and self.xsb == "shift"
                             and (self.kc % 4 == 0 or self.splits == 1) and self.diag == "none" and self.mutant != "scale"
                             and self.wsb == "shift"):
            raise ValueError("sop: w=mx, wlay=tile, xscale=glb, xsb/wsb shift, kc % 4 == 0 or splits == 1")
        if self.wlay not in ("row", "tile"):
            raise ValueError("wlay row | tile")
        if self.wcp not in WCPS:
            raise ValueError(f"wcp in {sorted(WCPS)}")
        if self.map not in ("rr", "xcd"):
            raise ValueError("map rr | xcd")
        if self.grid < XCDS or self.grid % XCDS:
            raise ValueError("grid a multiple of 8")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "scale" and self.w != "mx":
            raise ValueError("the scale mutant needs w=mx")
        if self.mutant == "fold" and self.splits < 2:
            raise ValueError("the fold mutant needs splits > 1")
        if self.fold not in ("last", "none"):
            raise ValueError("fold last | none")
        if self.diag not in ("none", "nomfma", "nob", "noxs", "nows", "xs0", "xsu", "xsk"):
            raise ValueError("diag none | nomfma | nob | noxs | nows | xs0 | xsu | xsk")
        if self.mutant == "fold" and self.fold != "last":
            raise ValueError("the fold mutant needs fold=last")
        if self.mutant == "sopsel" and not self.sop:
            raise ValueError("the sopsel mutant needs sop")
        if not 0 <= self.bpre <= self.tbt:
            raise ValueError("bpre 0..tb")
        if self.tstamp and (self.stamp or 3 + 2 * self.kc > 64):
            raise ValueError("tstamp: not with stamp, kc <= 30")
        if self.kc > 80:
            raise ValueError("kc > 80 chunks per unit (unrolled): raise splits")
        if self.lds_bytes() > 160 * 1024 - 256:
            raise ValueError(f"LDS {self.lds_bytes()} bytes too large")

    def lds_bytes(self) -> int:
        if self.xsrc == "global":
            return 0
        rows = self.tbt * TILE
        if self.w == "bf16":
            return 2 * rows * (CHUNK * 2 + self.pad)
        return 2 * rows * (CHUNK + self.pad) + 2 * rows * 4

    def suffix(self) -> str:
        parts = []
        d = GemmKnobs()
        for f in fields(self):
            if f.name in ("w", "k", "bucket", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kernel_name(k: GemmKnobs) -> str:
    s = k.suffix()
    return f"q38lb_gemm_{k.w}_k{k.k}_b{k.tiles * TILE}" + (f"_{s}" if s else "")


def gemm_params(k: GemmKnobs) -> list[str]:
    """Parameter order: i64 addresses, then i32 ``m, rows, layers, gsync`` (host packs ``<Q...iiii``). ``gs``: the
    grid-sync words (word 0 = error count, then 8 stripe counters per layer boundary; never reset, zero once)."""
    p = ["w", "ws", "x", "xs", "y", "part", "cnt", "gs"]
    if k.stamp or k.tstamp:
        p.append("stamps")
    return p + ["m", "rows", "layers", "gsync"]


def _maybe_if(b: IRBuilder, cond):
    return contextlib.nullcontext() if cond is None else b.scf_if(cond)


def _ptr(b: IRBuilder, addr: Value, t=I32) -> Value:
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b: IRBuilder, v: Value) -> Value:
    return b.zext(v, I64)


def _cdiv(b: IRBuilder, a: Value, d: Value) -> Value:
    return b.div(b.add(a, b.sub(d, _c(b, 1))), d)


def _udiv(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.lshr(x, _c(b, d.bit_length() - 1))
    return b.div(x, _c(b, d))


def _umod(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.land(x, _c(b, d - 1))
    return b.mod(x, _c(b, d))


class _Unit:
    """Uniform scalars of this CTA's unit ``n`` (any ``n``; past the end it is invalid and every load is OOB)."""

    def __init__(self, g: "_Gemm", n: Value):
        b, k = g.b, g.k
        self.valid = b.cmp_lt(n, g.ncta)
        nn = b.select(self.valid, n, _c(b, 0))
        layer = b.div(nn, g.cnt1)
        self.i = b.readfirstlane(b.sub(nn, b.mul(layer, g.cnt1)))
        o = b.add(g.obase, b.mul(self.i, g.ostride))
        self.layer = b.readfirstlane(layer)
        o = b.readfirstlane(o)
        self.tbk = b.mod(o, g.NTBl)
        r2 = b.div(o, g.NTBl)
        self.rg = b.mod(r2, g.RG)
        self.s = b.div(r2, g.RG)
        self.q0 = b.mul(self.s, _c(b, k.kc))  # first global chunk
        self.tok0 = b.mul(self.tbk, _c(b, k.tbt * TILE))
        lay = _i64(b, self.layer)
        self.w_rs = _rsrc(b, b.add(g.a["w"], b.mul(lay, g.wl)), g.wl32)
        if k.w == "mx":
            self.s_rs = _rsrc(b, b.add(g.a["ws"], b.mul(lay, g.sl)), g.sl32)
            if k.wsb == "byte":
                self.s_ptr = b.int_to_ptr(b.readfirstlane(b.add(g.a["ws"], b.mul(lay, g.sl))), PtrType(I8, "global"))
        self.y_a = b.add(g.a["y"], b.mul(lay, g.yl))
        if k.splits > 1:
            self.p_a = b.add(g.a["part"], b.mul(lay, g.pl))
            self.c_a = b.add(g.a["cnt"], b.mul(lay, g.cl))


class _Gemm:
    def __init__(self, k: GemmKnobs):
        k.check()
        self.k = k
        self.b = b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = k.threads
        names = gemm_params(k)
        self.a = {n: b.param(n, I64) for n in names[:-4]}
        self.M = b.readfirstlane(b.param("m", I32))
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.L = b.readfirstlane(b.param("layers", I32))
        gs_ = b.readfirstlane(b.param("gsync", I32))
        self.gsync = b.cmp_ne(gs_, _c(b, 0))
        self.nogs = b.cmp_eq(gs_, _c(b, 0))
        self.cta = b.readfirstlane(b.block_id_x())
        if k.stamp:
            self._stamp(0, None)
            self._stamp(2, D.xcc_id(b))
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))  # lane group 0..3
        self.l16 = b.land(self.lane, _c(b, 15))
        R, T = k.rows_per_unit, k.tiles
        TP = T * TILE
        # runtime geometry
        self.RG = b.readfirstlane(_cdiv(b, self.M, _c(b, R)))
        self.Mpad = b.mul(self.RG, _c(b, R))
        Tl = _cdiv(b, self.rows, _c(b, TILE))
        self.NTBl = b.readfirstlane(_cdiv(b, Tl, _c(b, k.tbt)))
        self.U = b.readfirstlane(b.mul(b.mul(self.RG, self.NTBl), _c(b, k.splits)))
        # units of one layer: CTA c takes o = obase + i ostride, i < cnt (the same count in every layer)
        if k.map == "rr":
            self.obase = self.cta
            self.ostride = _c(b, k.grid)
            cnt = b.smax(_cdiv(b, b.sub(self.U, self.cta), _c(b, k.grid)), _c(b, 0))
        else:
            GX = k.grid // XCDS
            x = _umod(b, self.cta, XCDS)
            xj = _udiv(b, self.cta, XCDS)
            lo = b.div(b.mul(x, self.U), _c(b, XCDS))
            hi = b.div(b.mul(b.add(x, _c(b, 1)), self.U), _c(b, XCDS))
            self.obase = b.add(lo, xj)
            self.ostride = _c(b, GX)
            cnt = b.smax(_cdiv(b, b.sub(b.sub(hi, lo), xj), _c(b, GX)), _c(b, 0))
        self.cnt = b.readfirstlane(cnt)
        self.cnt1 = b.readfirstlane(b.smax(cnt, _c(b, 1)))
        self.ncta = b.readfirstlane(b.mul(cnt, self.L))
        # per-layer strides (bytes)
        Mi = _i64(b, self.M)
        MT = _i64(b, _cdiv(b, self.M, _c(b, TILE)))
        if k.wlay == "tile":  # [M / 16][nq][16 rows x 128 K] (mx: 1 KiB codes + 64 B scales per tile-chunk)
            if k.w == "mx":
                self.wl = b.mul(MT, b.const_i64(k.nq * 1024))
                self.sl = b.mul(MT, b.const_i64((-(-k.nq // 4) * 256) if k.sop else (k.nq * 64)))
                self.sl32 = b.trunc(self.sl, I32)
            else:
                self.wl = b.mul(MT, b.const_i64(k.nq * 4096))
        elif k.w == "mx":
            self.wl = b.mul(Mi, b.const_i64(k.k // 2))
            self.sl = b.mul(Mi, b.const_i64(ksp(k.k)))
            self.sl32 = b.trunc(self.sl, I32)
        else:
            self.wl = b.mul(Mi, b.const_i64(k.k * 2))
        self.wl32 = b.trunc(self.wl, I32)
        Mp = _i64(b, self.Mpad)
        self.yl = b.mul(Mp, b.const_i64(TP * 2))
        self.yl32 = b.trunc(self.yl, I32)
        if k.splits > 1:
            self.pl = b.mul(Mp, b.const_i64(TP * 4 * k.splits))
            self.pl32 = b.trunc(self.pl, I32)
            self.cl = _i64(b, b.mul(b.mul(self.RG, _c(b, k.ntb)), _c(b, 4)))
        # activations (shared by every layer)
        rl = b.smax(self.rows, _c(b, 1))
        self.rlast = b.sub(rl, _c(b, 1))
        if k.w == "mx":
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, k.k)))
            if k.sop:
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, -(-k.nq // 4) * k.tiles * TILE * 16))
            elif k.xsb == "dword":
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, k.nq * k.tiles * TILE * 16))
            elif k.xslay == "chunk":
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, k.nq * k.tiles * TILE * 4))
                self.xs_ptr = b.int_to_ptr(b.readfirstlane(self.a["xs"]), PtrType(I8, "global"))
            else:
                self.xs_rs = _rsrc(b, self.a["xs"], b.mul(rl, _c(b, ksp(k.k))))
        else:
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, k.k * 2)))
        self.aux = WCPS[k.wcp]
        # LDS
        if k.occ1:
            pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
            a0 = b.trunc(b.smem_addr_of(pad), I32)
            with b.scf_if(D.is_thread0(b)):
                b.inline_asm(
                    "ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
                    "v,v,~{memory}",
                    [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)],
                    None,
                )
        rows = k.tbt * TILE
        if k.xsrc == "lds":
            if k.w == "bf16":
                self.xs = b.smem_alloc(BF16, [2, rows, CHUNK + k.pad // 2], name_hint="xs")
            else:
                self.xs = b.smem_alloc(I8, [2, rows, CHUNK + k.pad], name_hint="xs8")
                # f32-typed LDS words (loaded with smem_load_vN_f32): an ``av1`` name hint of a 1-element smem_load_vN
                # collides with ``av16`` + counter in kernels this large (duplicate SSA names, invalid IR)
                self.xss = b.smem_alloc(F32, [2, rows * (4 if k.xsb == "byte" else 1)], name_hint="xss")
        if k.splits > 1:
            self.flag = b.smem_alloc(F32, [1], name_hint="fold")
        self.gsw = b.smem_alloc(F32, [1], name_hint="gsw")
        # this lane's weight row within a unit, per weight tile
        self.wrow = [b.add(b.mul(b.add(_c(b, mi * k.waves), self.wave), _c(b, TILE)), self.l16) for mi in range(k.mt)]
        self.sh8g = b.mul(self.g, _c(b, 8))
        if k.diag == "xsk":  # one loop-invariant (opaque) X scale VGPR per token tile: distinct registers, no producer
            self.xsk = [D.fresh(b, _c(b, 120 + (t % 4))) for t in range(k.tbt)]

    # ------------------------------------------------------------------ helpers
    def _stamp(self, col: int, val) -> None:
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            p = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            v = D.memrealtime(b) if val is None else b.zext(val, I64)
            b.store(v, p, b.add(b.mul(self.cta, _c(b, 4)), _c(b, col)))

    def tst(self, n: Value, p: int) -> None:
        if not self.k.tstamp:
            return
        b = self.b
        with b.scf_if(b.land(D.is_thread0(b), b.cmp_lt(n, _c(b, 4)))):
            ptr = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            b.store(D.memtime(b), ptr, b.add(b.mul(b.add(b.mul(self.cta, _c(b, 4)), n), _c(b, 64)), _c(b, p)))

    # ------------------------------------------------------------------ loads
    def wissue(self, u: _Unit, c: int, xonly_gate=None, xonly: bool = False) -> list:
        """Weight loads of chunk ``c`` of unit ``u`` (and with xsrc=global its X operands)."""
        b, k = self.b, self.k
        q = b.add(u.q0, _c(b, c))
        out = []
        kl = b.add(b.mul(q, _c(b, CHUNK)), b.mul(self.g, _c(b, 32)))  # first K of the lane's 32
        kok = b.cmp_lt(kl, _c(b, k.k)) if k.ragged else None
        for mi in range(0 if xonly else k.mt):
            row = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), self.wrow[mi])
            ok = b.land(u.valid, b.cmp_lt(row, self.M))
            if k.wlay == "tile":
                tq = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, k.nq)), q)  # (row tile, chunk) block
                r16 = b.land(row, _c(b, 15))
            if k.w == "mx":
                okc = ok if kok is None else b.land(ok, kok)
                if k.wlay == "tile":
                    off = b.add(b.add(b.mul(tq, _c(b, 1024)), b.mul(r16, _c(b, 64))), b.mul(self.g, _c(b, 16)))
                else:
                    off = b.add(b.mul(row, _c(b, k.k // 2)), b.lshr(kl, _c(b, 1)))
                out.append(b.buffer_load_cp(u.w_rs, b.select(okc, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.aux))
                if k.sop:
                    continue  # the scales of 4 chunks come after the codes (below)
                if k.wlay == "tile":
                    soff = b.add(b.mul(tq, _c(b, 64)), b.mul(r16, _c(b, 4)))
                else:
                    soff = b.add(b.mul(row, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                if k.diag == "nows":
                    out.append(_c(b, 0x7F7F7F7F))
                elif k.wsb == "byte":  # the lane's own block byte; invalid rows read byte 0 (their codes are 0)
                    out.append(b.zext(b.global_load_i8(u.s_ptr, b.select(ok, b.add(soff, self.g), _c(b, 0))), I32))
                else:
                    out.append(b.buffer_load_cp(u.s_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.aux))
            else:
                base = b.add(b.mul(row, _c(b, k.k)), kl)
                if k.wlay == "tile":
                    tbase = b.add(b.add(b.mul(tq, _c(b, 4096)), b.mul(r16, _c(b, 256))), b.mul(self.g, _c(b, 64)))
                for m in range(4):
                    okm = ok
                    if k.ragged:
                        okm = b.land(ok, b.cmp_lt(b.add(kl, _c(b, 8 * m)), _c(b, k.k)))
                    if k.wlay == "tile":
                        off = b.add(tbase, _c(b, 16 * m))
                    else:
                        off = b.mul(b.add(base, _c(b, 8 * m)), _c(b, 2))
                    out.append(b.buffer_load_cp(u.w_rs, b.select(okm, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.aux))
        if k.xsrc == "global":
            out += self.xglobal(u, q, self.xgate(u) if xonly_gate is None else xonly_gate)
        if k.sop:
            if c % 4:
                return out
            nq4 = -(-k.nq // 4)
            grp = b.lshr(q, _c(b, 2))
            for mi in range(0 if xonly else k.mt):
                row = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), self.wrow[mi])
                ok = b.land(u.valid, b.cmp_lt(row, self.M))
                blk = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, nq4)), grp)
                soff = b.add(b.add(b.mul(blk, _c(b, 256)), b.mul(b.land(row, _c(b, 15)), _c(b, 16))), b.mul(self.g, _c(b, 4)))
                out.append(b.buffer_load_cp(u.s_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.aux))
            gate = self.xgate(u) if xonly_gate is None else xonly_gate
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                xoff = b.mul(b.add(b.mul(b.add(b.mul(grp, _c(b, k.tiles * TILE)), tok), _c(b, 4)), self.g), _c(b, 4))
                out.append(b.buffer_load_cp(self.xs_rs, b.select(gate, xoff, _c(b, OOB)), _c(b, 0), dwords=1))
            return out
        if k.xscale == "glb":
            gate = self.xgate(u) if xonly_gate is None else xonly_gate
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                soff = b.mul(b.add(b.mul(q, _c(b, k.tiles * TILE)), tok), _c(b, 4))
                if k.xsb == "dword":  # [nq][bucket16][4] dwords: the lane's block scale, no VALU op before the MFMA
                    doff = b.mul(b.add(b.mul(b.add(b.mul(q, _c(b, k.tiles * TILE)), tok), _c(b, 4)), self.g), _c(b, 4))
                    out.append(b.buffer_load_cp(self.xs_rs, b.select(gate, doff, _c(b, OOB)), _c(b, 0), dwords=1))
                elif k.xsb == "byte":  # gated loads read byte 0 (a valid scale; re-issued at the crossing)
                    out.append(b.zext(b.global_load_i8(self.xs_ptr, b.select(gate, b.add(soff, self.g), _c(b, 0))), I32))
                else:
                    out.append(b.buffer_load_cp(self.xs_rs, b.select(gate, soff, _c(b, OOB)), _c(b, 0), dwords=1))
        return out

    def xgate(self, u: _Unit) -> Value:
        """X of unit ``u`` may be loaded now: no grid sync, or ``u`` is in the current unit's layer (an activation of a
        later layer exists only after the barrier; those loads go OOB and are re-issued at the crossing)."""
        b = self.b
        return b.lor(self.nogs, b.cmp_eq(u.layer, self.cur_layer))

    def xglobal(self, u: _Unit, q: Value, gate=None) -> list:
        """xsrc=global: this lane's B operands of every tile of chunk ``q`` (the LDS layout's values)."""
        b, k = self.b, self.k
        out = []
        for t in range(k.tbt):
            tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
            tok = b.smin(tok, self.rlast)
            if k.w == "mx":
                for h in range(2):
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(b.mul(self.g, _c(b, 16)), _c(b, 64 * h)))
                    ok = b.cmp_lt(kk, _c(b, k.k))
                    if gate is not None:
                        ok = b.land(ok, gate)
                    off = b.select(ok, b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, OOB))
                    out.append(b.buffer_load_cp(self.x_rs, off, _c(b, 0), dwords=4))
                soff = b.add(b.mul(tok, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                if gate is not None:
                    soff = b.select(gate, soff, _c(b, OOB))
                out.append(b.buffer_load_cp(self.xs_rs, soff, _c(b, 0), dwords=1))
            else:
                for m in range(4):
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(b.mul(self.g, _c(b, 32)), _c(b, 8 * m)))
                    ok = b.cmp_lt(kk, _c(b, k.k))
                    if gate is not None:
                        ok = b.land(ok, gate)
                    off = b.select(ok, b.mul(b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, 2)), _c(b, OOB))
                    out.append(b.buffer_load_cp(self.x_rs, off, _c(b, 0), dwords=4))
        return out

    def xissue(self, u: _Unit, c: int, gate=None) -> list:
        """xsrc=lds: this thread's share of the activation chunk ``c`` of unit ``u`` (codes, then scales)."""
        b, k = self.b, self.k
        q = b.add(u.q0, _c(b, c))
        rows = k.tbt * TILE
        per = 16 if k.w == "bf16" else 8  # 16-byte pieces per token
        total = rows * per
        out = []
        for it in range(-(-total // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            live = b.cmp_lt(v, _c(b, total))
            v = b.select(live, v, _c(b, total - 1))
            tl = _udiv(b, v, per)
            tok = b.smin(b.add(u.tok0, tl), self.rlast)
            kk = b.add(b.mul(q, _c(b, CHUNK)), b.mul(_umod(b, v, per), _c(b, 16 // (2 if k.w == "bf16" else 1))))
            ok = b.cmp_lt(kk, _c(b, k.k)) if k.ragged else None
            g_ = self.xgate(u) if gate is None else gate
            ok = g_ if ok is None else b.land(ok, g_)
            eb = 2 if k.w == "bf16" else 1
            off = b.mul(b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, eb)) if eb > 1 else b.add(b.mul(tok, _c(b, k.k)), kk)
            if ok is not None:
                off = b.select(ok, off, _c(b, OOB))
            out.append(b.buffer_load_cp(self.x_rs, off, _c(b, 0), dwords=4))
        if k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xslay == "chunk" and k.xscale == "lds":
            n4 = rows // 4
            for it in range(-(-n4 // k.threads)):
                v = b.add(self.tid, _c(b, it * k.threads))
                ok = b.land(b.cmp_lt(v, _c(b, n4)), self.xgate(u) if gate is None else gate)
                soff = b.mul(b.add(b.add(b.mul(q, _c(b, k.tiles * TILE)), u.tok0), b.mul(v, _c(b, 4))), _c(b, 4))
                out.append(b.buffer_load_cp(self.xs_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=4))
        elif k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xscale == "lds":
            for it in range(-(-rows // k.threads)):
                v = b.add(self.tid, _c(b, it * k.threads))
                vv = b.smin(v, _c(b, rows - 1))
                tok = b.smin(b.add(u.tok0, vv), self.rlast)
                soff = b.add(b.mul(tok, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                soff = b.select(self.xgate(u) if gate is None else gate, soff, _c(b, OOB))
                out.append(b.buffer_load_cp(self.xs_rs, soff, _c(b, 0), dwords=1))
        return out

    def xput(self, regs: list, buf: Value) -> None:
        """Write a staged activation chunk (``xissue``'s registers) to LDS buffer ``buf``."""
        b, k = self.b, self.k
        rows = k.tbt * TILE
        per = 16 if k.w == "bf16" else 8
        total = rows * per
        n_it = -(-total // k.threads)
        for it in range(n_it):
            v = b.add(self.tid, _c(b, it * k.threads))
            live = b.cmp_lt(v, _c(b, total))
            tl = _udiv(b, v, per)
            col = b.mul(_umod(b, v, per), _c(b, 8 if k.w == "bf16" else 16))
            r = regs[it]
            with _maybe_if(b, live if total % k.threads else None):
                if k.w == "bf16":
                    b.smem_store_vN(self.xs, [buf, tl, col], b.vec_bitcast(r, VectorType(BF16, 8)), 8)
                else:
                    b.smem_store_vN(self.xs, [buf, tl, col], b.vec_bitcast(r, VectorType(I8, 16)), 16)
        if k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xslay == "chunk" and k.xscale == "lds":
            n4 = rows // 4
            for j, it in enumerate(range(-(-n4 // k.threads))):
                v = b.add(self.tid, _c(b, it * k.threads))
                r = b.vec_bitcast(regs[n_it + j], VectorType(F32, 4))
                with _maybe_if(b, b.cmp_lt(v, _c(b, n4)) if n4 % k.threads else None):
                    if k.xsb == "byte":
                        for e in range(4):
                            self._xs_bytes(buf, b.add(b.mul(v, _c(b, 4)), _c(b, e)), b.bitcast(b.vec_extract(r, e), I32))
                    else:
                        b.smem_store_vN(self.xss, [buf, b.mul(v, _c(b, 4))], r, 4)
        elif k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xscale == "lds":
            for j, it in enumerate(range(-(-rows // k.threads))):
                v = b.add(self.tid, _c(b, it * k.threads))
                r = regs[n_it + j]
                with _maybe_if(b, b.cmp_lt(v, _c(b, rows)) if rows % k.threads else None):
                    if k.xsb == "byte":
                        self._xs_bytes(buf, v, r)
                    else:
                        b.smem_store_vN(self.xss, [buf, v], b.bitcast(r, F32), 1)

    def _xs_bytes(self, buf: Value, tok: Value, w: Value) -> None:
        """xsb=byte: token ``tok``'s 4 block scales (bytes of ``w``) as 4 LDS dwords (16 bytes, one store)."""
        b = self.b
        parts = [b.land(b.lshr(w, _c(b, 8 * e)), _c(b, 255)) for e in range(4)]
        vec = b.vec_bitcast(b.vec_pack(parts, I32), VectorType(F32, 4))
        b.smem_store_vN(self.xss, [buf, b.mul(tok, _c(b, 4))], vec, 4)

    # ------------------------------------------------------------------ math
    def bop(self, t: int, buf: Value | None, xg: list | None, m: int = 0):
        """B operand of token tile ``t`` (mx: (codes <8 x i32>, scale dword shifted); bf16: slice ``m``)."""
        b, k = self.b, self.k
        if k.xsrc == "global":
            if k.w == "mx":
                base = 3 * t
                v = b.vec_concat(xg[base], xg[base + 1])
                return v, b.lshr(xg[base + 2], self.sh8g)
            return b.vec_bitcast(xg[4 * t + m], VectorType(BF16, 8))
        row = b.add(_c(b, t * TILE), self.l16)
        if k.w == "mx":
            halves = []
            for h in range(2):
                col = b.add(b.mul(self.g, _c(b, 16)), _c(b, 64 * h))
                hv = b.smem_load_vN(self.xs, buf, row, col, dtype=I8, n=16)
                halves.append(b.vec_bitcast(hv, VectorType(I32, 4)))
            if k.diag == "noxs":
                return b.vec_concat(halves[0], halves[1]), _c(b, 127)
            if k.diag == "xsk":
                return b.vec_concat(halves[0], halves[1]), self.xsk[t]
            if k.sop:
                return b.vec_concat(halves[0], halves[1]), None
            if k.xscale == "glb":
                return b.vec_concat(halves[0], halves[1]), (xg[t] if k.xsb in ("byte", "dword") else b.lshr(xg[t], self.sh8g))
            if k.diag == "xs0":
                row = self.l16
            if k.xsb == "byte":
                sw = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.xss, buf, b.add(b.mul(row, _c(b, 4)), self.g), n=1), 0), I32)
                return b.vec_concat(halves[0], halves[1]), sw
            sw = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.xss, buf, row, n=1), 0), I32)
            if k.diag == "xsu":
                return b.vec_concat(halves[0], halves[1]), b.readfirstlane(b.lshr(sw, self.sh8g))
            return b.vec_concat(halves[0], halves[1]), b.lshr(sw, self.sh8g)
        col = b.add(b.mul(self.g, _c(b, 32)), _c(b, 8 * m))
        return b.smem_load_vN(self.xs, buf, row, col, dtype=BF16, n=8)

    def chunk_math(self, acc, wl: list, buf: Value | None, skip: bool, j: int = 0) -> list:
        b, k = self.b, self.k
        if k.sop:
            return self._sop_math(acc, wl, buf, skip, j)
        xg = None
        if k.xsrc == "global" or k.xscale == "glb":
            nw = k.mt * (2 if k.w == "mx" else 4)
            xg = wl[nw:]
        acc = [list(r) for r in acc]

        def ops_of(t):
            if k.diag == "nob":
                return None
            if k.w == "mx":
                return self.bop(t, buf, xg)
            return [self.bop(t, buf, xg, m) for m in range(4)]

        ahead = {}
        if k.bpre:
            for t in range(k.bpre):
                ahead[t] = ops_of(t)
            if k.sb:
                b.sched_barrier(0)
        if k.w == "mx":
            A, SA = [], []
            for mi in range(k.mt):
                A.append(wl[2 * mi])
                sw = wl[2 * mi + 1]
                SA.append(sw if (k.mutant == "scale" or k.wsb == "byte") else b.lshr(sw, self.sh8g))
        for t in range(k.tbt):
            if k.bpre:
                if t + k.bpre < k.tbt:
                    ahead[t + k.bpre] = ops_of(t + k.bpre)
                cur = ahead.pop(t)
            else:
                cur = ops_of(t)
            # mutant kskip: the chunk's weights are replaced by zeros (every value stays live)
            zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
            if k.diag in ("nomfma", "nob"):
                acc = self._diag_math(acc, t, cur, wl, SA if k.w == "mx" else None)
                continue
            if k.w == "mx":
                bv, bs = cur
                for mi in range(k.mt):
                    acc[mi][t] = b.mfma_scale_mx(
                        zw if skip else A[mi], bv, acc[mi][t], SA[mi], bs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8"
                    )
            else:
                for m in range(4):
                    for mi in range(k.mt):
                        a = b.vec_bitcast(zw if skip else wl[4 * mi + m], VectorType(BF16, 8))
                        acc[mi][t] = b.mfma_f32_16x16x32_bf16(a, cur[m], acc[mi][t])
            if k.bpre and k.sb:
                b.sched_barrier(0)
        return acc

    def _sop_math(self, acc, wl, buf, skip, j):
        """sop: the group's codes (one per weight tile), the current 4-chunk scale dwords (``self.sop_sc``: weight
        tiles then token tiles), byte ``j`` selected by op_sel."""
        b, k = self.b, self.k
        acc = [list(r) for r in acc]
        if len(wl) > k.mt:  # first chunk of a 4-chunk group: new scale dwords
            self.sop_sc = wl[k.mt :]
        SA = self.sop_sc[: k.mt]
        SB = self.sop_sc[k.mt :]
        zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
        js = (j + 1) % 4 if k.mutant == "sopsel" else j  # mutant sopsel: the next chunk's scale byte
        ahead = {}
        nb = k.bpre or 0
        for t in range(nb):
            ahead[t] = self.bop(t, buf, None)
        for t in range(k.tbt):
            if nb:
                if t + nb < k.tbt:
                    ahead[t + nb] = self.bop(t + nb, buf, None)
                bv = ahead.pop(t)[0]
            else:
                bv = self.bop(t, buf, None)[0]
            for mi in range(k.mt):
                acc[mi][t] = b.mfma_scale_mx(
                    zw if skip else wl[mi], bv, acc[mi][t], SA[mi], SB[t], shape="16x16x128", fmt_a="fp4", fmt_b="fp8",
                    opsel_a=js, opsel_b=js,
                )
        return acc

    def _diag_math(self, acc, t, cur, A, SA):
        """diag=nomfma: fold the operands into the accumulators with VALU ops (no MFMA); diag=nob: MFMAs on a
        constant B operand (the LDS reads are not issued). Wrong results by design (timing diagnosis only)."""
        b, k = self.b, self.k
        if k.diag == "nob":
            if k.w == "mx":
                bv = b.vec_pack([_c(b, 0x38383838)] * 8, I32)
                bs = _c(b, 127)
                for mi in range(k.mt):
                    acc[mi][t] = b.mfma_scale_mx(A[2 * mi], bv, acc[mi][t], SA[mi], bs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
            else:
                bv = b.vec_bitcast(b.vec_pack([_c(b, 0x3F803F80)] * 4, I32), VectorType(BF16, 8))
                for m in range(4):
                    for mi in range(k.mt):
                        a = b.vec_bitcast(A[4 * mi + m], VectorType(BF16, 8))
                        acc[mi][t] = b.mfma_f32_16x16x32_bf16(a, bv, acc[mi][t])
            return acc
        # nomfma
        if k.w == "mx":
            bv, bs = cur
            x = b.xor(b.xor(b.vec_extract(bv, 0), b.vec_extract(bv, 7)), bs)
        else:
            x = b.vec_extract(b.vec_bitcast(cur[0], VectorType(I32, 4)), 0)
            for m in range(1, 4):
                x = b.xor(x, b.vec_extract(b.vec_bitcast(cur[m], VectorType(I32, 4)), 3))
        for mi in range(k.mt):
            w0 = b.vec_extract(A[2 * mi] if k.w == "mx" else A[4 * mi], 0)
            if k.w == "mx":
                w0 = b.xor(w0, SA[mi])
            f = b.bitcast(b.land(b.xor(x, w0), _c(b, 0x3FFFFFFF)), F32)
            acc[mi][t] = b.vec_insert(acc[mi][t], b.fadd(b.vec_extract(acc[mi][t], 0), f), 0)
        return acc

    # ------------------------------------------------------------------ grid sync (phase metric)
    def gbar(self, bnd: Value) -> None:
        """Grid barrier at layer boundary ``bnd`` (CTA-uniform call): every wave's stores done, one agent-scope add
        per CTA on stripe ``cta mod 8`` of the boundary's 8 counters, thread 0 polls the 8 stripes (system-scope loads,
        2 s deadline; a timeout adds 1 to word 0). Counters are never reset: launch ``e`` waits for ``(e + 1) G / 8``."""
        b, k = self.b, self.k
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        per = k.grid // XCDS
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(D.is_thread0(b)):
            mine = b.mul(b.add(row, _umod(b, self.cta, XCDS)), _c(b, 4))
            old = D.atomic_add_scoped(b, gsa, mine, _c(b, 1), scope="agent", returns=True)
            target = b.mul(b.add(b.div(old, _c(b, per)), _c(b, 1)), _c(b, per))
            b.smem_store_vN(self.gsw, [_c(b, 0)], b.bitcast(target, F32), 1)
        # lanes 0..7 of wave 0 poll the 8 stripes together (one load round per poll iteration)
        with b.scf_if(b.cmp_lt(self.tid, _c(b, XCDS))):
            b.s_waitcnt(lgkmcnt=0)
            target = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.gsw, _c(b, 0), n=1), 0), I32)
            dl = b.add(D.memrealtime(b), b.const_i64(200_000_000))
            st = W.poll_count_ge(b, gsa, b.mul(b.add(row, self.tid), _c(b, 4)), target, dl, pause="s_sleep1")
            with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), D.is_thread0(b))):
                D.atomic_add_scoped(b, gsa, _c(b, 0), _c(b, 1), scope="agent")
        b.sync_lds_only()

    # ------------------------------------------------------------------ epilogue
    def epilogue(self, u: _Unit, acc) -> None:
        b, k = self.b, self.k
        TP = k.tiles * TILE
        if k.splits == 1:
            y_rs = _rsrc(b, u.y_a, self.yl32)
            for mi in range(k.mt):
                row = b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
                for t in range(k.tbt):
                    tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                    ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                    off = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
                    v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    b.buffer_store_cp(v, y_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0))
            return
        p_rs = _rsrc(b, u.p_a, self.pl32)
        for mi in range(k.mt):
            row = b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                off = b.mul(b.add(b.mul(b.add(b.mul(u.s, _c(b, TP)), tok), self.Mpad), row), _c(b, 4))
                v = b.vec_bitcast(acc[mi][t], VectorType(I32, 4))
                b.buffer_store_cp(v, p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if k.fold == "none":
            return
        # arrive: every wave's partial stores complete (sc1 write-through), then one agent-scope add
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        idx = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        cbase = b.readfirstlane(u.c_a)
        with b.scf_if(D.is_thread0(b)):
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(_c(b, -1), F32), 1)
            with b.scf_if(u.valid):
                old = D.atomic_add_scoped(b, cbase, b.mul(idx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flag, _c(b, 0), n=1), 0), I32))
        with b.scf_if(b.cmp_eq(seen, _c(b, k.splits - 1))):
            D.acquire_l2(b, "agent")
            self.fold(u, p_rs)
            with b.scf_if(D.is_thread0(b)):
                cp = _ptr(b, cbase)
                b.store(_c(b, 0), cp, idx)

    def fold(self, u: _Unit, p_rs: Value) -> None:
        """Last arriver: Y = sum over splits (split order) of the partials of (row group, token block)."""
        b, k = self.b, self.k
        TP = k.tiles * TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * TILE * q4
        y_rs = _rsrc(b, u.y_a, self.yl32)
        S = k.splits - (1 if k.mutant == "fold" else 0)
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = _udiv(b, v, q4)
            row = b.add(b.mul(u.rg, _c(b, R)), b.mul(_umod(b, v, q4), _c(b, 4)))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tot = None
            for s0 in range(0, S, FOLD_GROUP):
                loads = []
                for s in range(s0, min(S, s0 + FOLD_GROUP)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.Mpad), row), _c(b, 4))
                    loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1))
                for ld in loads:
                    f = b.vec_bitcast(ld, VectorType(F32, 4))
                    if tot is None:
                        tot = [b.vec_extract(f, e) for e in range(4)]
                    else:
                        tot = [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            yoff = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
            b.buffer_store_cp(yv, y_rs, b.select(ok, yoff, _c(b, OOB)), _c(b, 0))

    # ------------------------------------------------------------------ main loop
    def build(self) -> KernelDef:
        b, k = self.b, self.k
        D_, XA, KC = k.depth, k.xa, k.kc
        lds = k.xsrc == "lds"
        units = {}

        def unit(n: Value, d: int) -> _Unit:
            key = d
            if key not in units:
                units[key] = _Unit(self, b.add(n, _c(b, d)) if d else n)
            return units[key]

        def wpos(n: Value, p: int, xg=None) -> list:
            d, c = divmod(p, KC)
            return self.wissue(unit(n, d), c, xg)

        def xpos(n: Value, p: int, gate=None) -> list:
            d, c = divmod(p, KC)
            return self.xissue(unit(n, d), c, gate)

        def sel(cond, new, old):
            return [b.select(cond, x, y) for x, y in zip(new, old)]

        def parity(n: Value, c: int) -> Value:
            if KC % 2 == 0 or not k.xu:
                return _c(b, c % 2)
            return b.land(b.add(n, _c(b, c)), _c(b, 1))

        zero = _c(b, 0)
        # prologue: unit 0's first chunks
        n0 = zero
        units.clear()
        self.cur_layer = unit(n0, 0).layer
        wq = [wpos(n0, p) for p in range(D_)] if k.xu else []
        xq = []
        if lds and k.xu:
            first = xpos(n0, 0)
            self.xput(first, zero)
            xq = [xpos(n0, p) for p in range(1, XA + 1)]
            b.sync_lds_only()
        lens_w = [len(g) for g in wq]
        lens_x = [len(g) for g in xq]
        carry = [v for g in wq for v in g] + [v for g in xq for v in g]
        iters = [(f"c{i}", v) for i, v in enumerate(carry)]
        loop = b.scf_for_iter(zero, self.ncta, _c(b, 1), iters, iv_name="un")
        with loop as lv:
            n, vals = lv if iters else (lv, [])
            units.clear()
            vals = list(vals)
            u = unit(n, 0)
            self.cur_layer = u.layer
            self.tst(n, 0)
            # grid sync at a layer crossing (phase metric), then the activations of this layer (re-)issued
            cross = b.land(self.gsync, b.land(b.cmp_eq(u.i, zero), b.cmp_gt(u.layer, zero)))
            with b.scf_if(cross):
                self.gbar(b.sub(u.layer, _c(b, 1)))
            pos = 0
            if k.xu:
                wq = []
                for ln in lens_w:
                    wq.append(vals[pos : pos + ln])
                    pos += ln
                xq = []
                for ln in lens_x:
                    xq.append(vals[pos : pos + ln])
                    pos += ln
                if lds:
                    f0 = xpos(n, 0, cross)
                    with b.scf_if(cross):
                        self.xput(f0, parity(n, 0))
                        b.sync_lds_only()
                    xq = [sel(cross, xpos(n, p, b.land(cross, self.xgate(unit(n, p // KC)))), xq[p - 1]) for p in range(1, XA + 1)]
                if k.sop:
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        if c_ % 4:
                            continue
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        nx = len(fresh)
                        wq[p] = wq[p][: len(wq[p]) - nx] + sel(cross, fresh, wq[p][len(wq[p]) - nx :])
                elif k.xsrc == "global" or k.xscale == "glb":
                    nw = k.mt * (2 if k.w == "mx" else 4)
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        wq[p] = wq[p][:nw] + sel(cross, fresh, wq[p][nw:])
            else:
                wq = [wpos(n, p) for p in range(min(D_, KC))]
                if lds:
                    first = xpos(n, 0)
                    b.sync_lds_only()  # every wave is done with the previous unit's last LDS buffer
                    self.xput(first, zero)
                    xq = [xpos(n, p) for p in range(1, min(XA, KC - 1) + 1)]
                    b.sync_lds_only()
            acc = [[b.zero_vec_f32(4) for _ in range(k.tbt)] for _ in range(k.mt)]
            self.tst(n, 1)
            for c in range(KC):
                if k.sb:
                    b.sched_barrier(0)
                if k.xu or c + D_ < KC:
                    wq.append(wpos(n, c + D_))
                if lds and (k.xu or c + 1 + XA < KC):
                    xq.append(xpos(n, c + 1 + XA))
                if k.sb:
                    b.sched_barrier(0)
                cur = wq.pop(0)
                skip = k.mutant == "kskip" and c == KC - 1
                acc = self.chunk_math(acc, cur, parity(n, c) if lds else None, skip, c % 4)
                self.tst(n, 2 + 2 * c)
                if lds and (k.xu or c + 1 < KC):
                    nxt = xq.pop(0)
                    self.xput(nxt, parity(n, c + 1))
                    b.sync_lds_only()
                self.tst(n, 3 + 2 * c)
            if k.sb:
                b.sched_barrier(0)
            self.epilogue(u, acc)
            self.tst(n, 2 + 2 * KC)
            carry = [v for g in wq for v in g] + [v for g in xq for v in g]
            if not k.xu:
                carry = []
            b.scf_yield(*carry)
        # a CTA without units in a layer still takes part in every boundary's grid sync
        with b.scf_if(b.land(self.gsync, b.cmp_eq(self.cnt, zero))):
            with b.scf_for(zero, b.sub(self.L, _c(b, 1)), _c(b, 1), iv_name="gb") as bnd:
                self.gbar(bnd)
        if k.stamp:
            b.sync_lds_only()
            self._stamp(1, None)
        return b.kernel


def build_gemm_kernel(k: GemmKnobs) -> KernelDef:
    return _Gemm(k).build()


# ------------------------------------------------------------------ the bytes floor


@dataclass(frozen=True)
class StreamKnobs:
    """``q38lb_gst``: read ``nbytes`` per layer (16 bytes a lane, ``depth`` loads in flight), no math: the floor a
    GEMM of the same weight bytes is compared with. ``grid`` CTAs of 512 threads, one per CU."""

    depth: int = 4
    grid: int = 256
    wcp: str = "default"
    vec: int = 1  # contiguous 16-byte loads per lane per step (1 | 2 | 4)
    tag: str = ""


def stream_params() -> list[str]:
    return ["src", "sink", "nbytes", "layers"]


def stream_name(k: StreamKnobs) -> str:
    return (f"q38lb_gst_d{k.depth}_g{k.grid}" + ("" if k.wcp == "default" else f"_{k.wcp}") + ("" if k.vec == 1 else f"_v{k.vec}")
            + (f"_{k.tag}" if k.tag else ""))


def build_stream_kernel(k: StreamKnobs) -> KernelDef:
    b = IRBuilder(stream_name(k))
    b.kernel.attrs["max_workgroup_size"] = 512
    src = b.param("src", I64)
    sink = b.param("sink", I64)
    nbytes = b.readfirstlane(b.param("nbytes", I32))  # per layer, a multiple of 16
    L = b.readfirstlane(b.param("layers", I32))
    cta = b.readfirstlane(b.block_id_x())
    tid = b.thread_id_x()
    pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
    a0 = b.trunc(b.smem_addr_of(pad), I32)
    with b.scf_if(D.is_thread0(b)):
        b.inline_asm(
            "ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
            "v,v,~{memory}",
            [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)],
            None,
        )
    V = k.vec
    per = 512 * 16 * V  # bytes a CTA reads per step
    steps_l = _cdiv(b, nbytes, _c(b, per))  # steps per layer
    total = b.mul(steps_l, L)
    my = b.smax(_cdiv(b, b.sub(total, cta), _c(b, k.grid)), _c(b, 0))
    aux = WCPS[k.wcp]
    zero = _c(b, 0)

    def issue(i: Value):
        idx = b.add(cta, b.mul(i, _c(b, k.grid)))
        ok = b.cmp_lt(i, my)
        idx = b.select(ok, idx, zero)
        layer = b.readfirstlane(b.div(idx, steps_l))
        st = b.sub(idx, b.mul(layer, steps_l))
        base = b.add(src, b.mul(_i64(b, layer), _i64(b, nbytes)))
        rs = _rsrc(b, base, nbytes)
        off = b.add(b.mul(st, _c(b, per)), b.mul(tid, _c(b, 16 * V)))
        off = b.select(ok, off, _c(b, OOB))
        return [b.buffer_load_cp(rs, b.add(off, _c(b, 16 * j)), zero, dwords=4, aux=aux) for j in range(V)]

    q = [v for i in range(k.depth) for v in issue(_c(b, i))]
    acc0 = _c(b, 0)
    iters = [(f"s{i}", v) for i, v in enumerate(q)] + [("acc", acc0)]
    loop = b.scf_for_iter(zero, my, _c(b, 1), iters, iv_name="st")
    with loop as (i, vals):
        vals = list(vals)
        ring, acc = vals[:-1], vals[-1]
        nxt = issue(b.add(i, _c(b, k.depth)))
        for v in ring[:V]:
            acc = b.xor(acc, b.xor(b.vec_extract(v, 0), b.vec_extract(v, 3)))
        b.scf_yield(*(ring[V:] + nxt), acc)
    acc = loop.results[-1]
    # keep the loads: a data-dependent store nobody reads in practice (acc == 0x5eed never holds for random data)
    with b.scf_if(b.cmp_eq(acc, _c(b, 0x5EED1234))):
        b.store(acc, _ptr(b, b.readfirstlane(sink)), tid)
    return b.kernel
