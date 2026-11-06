# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""W17 K1: the FP8 32x32-blockscale GEMM ATOM needs at TP8 (DeepSeek-V4.1-Flash, gfx950): ``Y = X W^T`` with a short K.

At TP8 the shared expert's down projection (``ffn.shared_experts.w2``, row-parallel) has ``K = 2304 / 8 = 288`` per
rank (``N = 5120``). ATOM routes it through ``atom/model_ops/blockscale.py native_quant_linear`` -> aiter's
preshuffled group32 GEMM, which has no K = 288 config (``experiments/dsv41-mono/w17/K1-k288-gemm.md`` section 1). This
module supplies that GEMM for any ``K`` that is a multiple of 32 up to 512 (288 = 2 full 128-K chunks + one 32-K tail
block), any ``N`` that is a multiple of 32, and any number of rows ``M`` (decode 1-48 rows, prefill thousands).

It is the W13 ``wide_gemm`` core (itself ``stage_wqkv``'s FP8 GEMV core: ATOM's mono K1 ``gemv_fp8_loads`` lane map,
``op_sel``-packed E8M0 scales, ``v_mfma_scale_f32_16x16x128_f8f6f4``) generalised in three ways:

* **K tail.** ``K / 32 = KB`` blocks; chunk ``c`` (128 K) holds blocks ``4c .. 4c+3``. In the 16x16x128 fp8 lane map
  half ``h`` of lane ``l`` holds K ``128 c + 64 h + 16 (l / 16) ..+15``, i.e. block ``4c + 2h + l / 32``. A weight half
  whose blocks are all past ``KB`` is the constant zero (no load); a half with one valid block loads lanes 0-31 only
  (lanes 32-63 read an out-of-range offset = 0: they would otherwise read the next 16-row tile's bytes). The activation
  staging writes zero codes (scale 127 = 1.0) for the pad blocks, so every product past K is an exact ``0 x 0``.
* **Any M.** One CTA = (row group of ``R = 16 waves mt`` weight rows) x (an M block of ``16 tiles`` tokens); the grid is
  ``ceil(N / R) * ceil(M / (16 tiles))`` (host: :func:`grid_of`), so one build serves every M. Rows past ``M`` are never
  read or written.
* **Whole-K staging.** K is short, so the M block's activations are staged once (one LDS slot ``[16 tiles][128 nq]``),
  in ``xb``-iteration batches (bounded registers for wide blocks). ``xin="bf16"`` quantizes in the staging with ATOM's
  ``quantize_fp8`` rule bit for bit (E8M0 ceil scale of ``max(amax, 1e-4) * fl(1/448)``, E4M3 RNE);
  ``xin="fp8"`` reads ATOM's codes ``[M, K]`` and E8M0 scales ``[M, K/32]``.

Weights: ATOM's (16, 16)-preshuffled E4M3 ``[N, K]`` (``aiter.ops.shuffle.shuffle_weight``: 16 rows x 32 K blocks of
512 bytes, two 16-byte K halves), read in place: row tile ``rt``, block ``b`` at ``16 K rt + 512 b``. Weight scales:
ATOM's E8M0 ``[N/32, K/32]``, repacked once on the host (:func:`pack_wscale`) to ``[N/32, 4]`` dwords: byte ``j`` of
dword ``[r, g]`` = the scale of block ``4 j + g`` (127 past ``KB``), so one dword per (32-row group, lane group) serves
every chunk through ``op_sel``.

Epilogue ``ep="direct"``: a lane stores its 4 consecutive weight rows of one token (8 bytes bf16). ``ep="lds"``: the
wave's C tiles go through LDS and a lane stores 16 contiguous bytes (8 rows of one token) per instruction.

Mutants (must fail the numerics gate): ``tail`` (the K-tail block is dropped: its weight half is zero), ``tailx`` (the
tail half's lanes 32-63 are not masked: they read the next row tile's bytes), ``scale`` (chunk 1 picks the next weight
scale byte), ``row`` (token rows 0 / 1 of every tile swap their activations), ``mblk`` (M block ``b > 0`` reads block
``b - 1``'s activations), ``xq`` (one quantization block in four stores its scale one binade high).

Kernel arguments (``PARAMS``): i64 ``w, wsp, x, xs, y``, i32 ``n, rows``. ``xs`` is read only with ``xin="fp8"``. Block
``64 waves`` threads. Python engine only (``mfma_scale_mx``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_NT, F32, I32, I64, I8, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D

__all__ = [
    "PARAMS",
    "MUTANTS",
    "K288Knobs",
    "build_k288_kernel",
    "grid_of",
    "kernel_name",
    "pack_wscale",
    "reference",
    "e4m3_to_float",
]

TILE = 16
CHUNK = 128
BLK = 32
XPAD = 16  # LDS row pad (bytes) of the staged activation codes
OOB = 0x7FFFFFF0
RATIO_BITS = 0x3B124925  # fl(1 / 448): ATOM quantize_fp8 ``amax * (1.0 / FP8_MAX)``
AMAX_FLOOR_BITS = struct.unpack("<I", struct.pack("<f", 1e-4))[0]  # ATOM ``tl.maximum(amax, 1e-4)`` (f32)
MUTANTS = ("none", "tail", "tailx", "scale", "row", "mblk", "xq")
WCPS = {"default": 0, "nt": CPOL_NT}
PARAMS = ("w", "wsp", "x", "xs", "y", "n", "rows")
LDS_MAX = 160 * 1024 - 1024


@dataclass(frozen=True)
class K288Knobs:
    """One build. The kernel name lists the deviations from the defaults."""

    k: int = 288  # K (a multiple of 32, <= 512)
    tiles: int = 1  # token tiles per CTA (M block = 16 tiles rows)
    xin: str = "bf16"  # "bf16" (fused MXFP8 quantization) | "fp8" (ATOM's MXFP8 codes + scales)
    waves: int = 4  # waves per CTA (1 | 2 | 4 | 8)
    mt: int = 1  # 16-row weight tiles per wave (1 | 2 | 4)
    depth: int = 3  # weight chunks in flight (1..4, capped at the chunk count)
    wcp: str = "default"  # weight cache policy ("default" | "nt")
    order: str = "nfast"  # CTA -> (M block, row group): "nfast" (row group = cta mod RG) | "mfast" (M block = cta mod MB)
    wfirst: bool = True  # issue the weight loads before the activation staging
    xb: int = 2  # staging iterations per load batch (bounds the registers of wide M blocks)
    bra: int = 1  # B (activation) LDS reads: 0 next to each MFMA | 1 a chunk's tiles before its MFMAs
    sb: bool = True  # sched_barrier fences around each chunk's weight issue
    ep: str = "direct"  # epilogue: "direct" (8-byte stores) | "lds" (through LDS, 16-byte stores)
    mutant: str = "none"
    proto: str = "none"  # bound prototypes (never served, not numerics-gated): "nomath" = every load, store and the
    #   staging as built, the MFMAs replaced by one f32 add of the operands (the memory / latency floor of this
    #   structure); "noquant" = the bf16 staging without the quantization arithmetic (raw bytes stored); "storeonly" =
    #   the grid and the epilogue only (zeros stored: the output-write floor); "empty" = the grid, nothing done (the
    #   dispatch floor)
    tag: str = ""  # recompiled-control suffix

    @property
    def kb(self) -> int:
        return self.k // BLK

    @property
    def nq(self) -> int:
        return -(-self.k // CHUNK)

    @property
    def rows_per_cta(self) -> int:
        return TILE * self.waves * self.mt

    @property
    def threads(self) -> int:
        return 64 * self.waves

    @property
    def bucket(self) -> int:
        return TILE * self.tiles

    def xrow(self) -> int:
        return self.nq * CHUNK + XPAD

    def lds_bytes(self) -> int:
        b = self.bucket * self.xrow() + self.bucket * 16
        if self.ep == "lds":
            b += self.waves * 64 * 4 * 4 * 2  # per wave: 16 rows x 16 tokens f32 ... staged as bf16 pairs
        return b + 64

    def check(self) -> None:
        if self.k <= 0 or self.k % BLK or self.k > 4 * CHUNK:
            raise ValueError("k must be a positive multiple of 32, <= 512")
        if not 1 <= self.tiles <= 24:
            raise ValueError("tiles 1..24")
        if self.xin not in ("bf16", "fp8"):
            raise ValueError("xin bf16 | fp8")
        if self.waves not in (1, 2, 4, 8) or self.mt not in (1, 2, 4):
            raise ValueError("waves 1 | 2 | 4 | 8, mt 1 | 2 | 4")
        if not 1 <= self.depth <= 4:
            raise ValueError("depth 1..4 (chunks in flight; capped at the chunk count)")
        if not 1 <= self.xb <= 64:
            raise ValueError("xb 1..64")
        if self.wcp not in WCPS:
            raise ValueError(f"wcp in {sorted(WCPS)}")
        if self.order not in ("nfast", "mfast"):
            raise ValueError("order nfast | mfast")
        if self.bra not in (0, 1):
            raise ValueError("bra 0 | 1")
        if self.ep not in ("direct", "lds"):
            raise ValueError("ep direct | lds")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.proto not in ("none", "nomath", "noquant", "storeonly", "empty"):
            raise ValueError("proto none | nomath | noquant | storeonly | empty")
        if self.proto == "noquant" and self.xin != "bf16":
            raise ValueError("proto noquant needs xin bf16")
        if self.mutant in ("tail", "tailx") and self.k % CHUNK == 0:
            raise ValueError("the tail mutants need a K tail (k % 128 != 0)")
        if self.mutant == "tailx" and (self.kb % 4) not in (1, 3):
            raise ValueError("tailx needs a half with one valid block (kb % 4 in 1, 3)")
        if self.mutant == "scale" and self.nq < 2:
            raise ValueError("the scale mutant needs 2+ chunks")
        if self.lds_bytes() > LDS_MAX:
            raise ValueError(f"LDS {self.lds_bytes()} bytes too large: lower tiles")
        if self.mt * self.tiles > 32:
            raise ValueError("mt x tiles > 32 accumulators")

    def suffix(self) -> str:
        parts = []
        d = K288Knobs()
        for f in fields(self):
            if f.name in ("k", "tiles", "xin", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kernel_name(k: K288Knobs) -> str:
    s = k.suffix()
    return f"rk41k_k{k.k}_t{k.tiles}_{k.xin}" + (f"_{s}" if s else "")


def grid_of(k: K288Knobs, n: int, rows: int) -> int:
    return -(-n // k.rows_per_cta) * max(1, -(-rows // k.bucket))


def pack_wscale(ws, k: int):
    """ATOM's E8M0 weight scales ``[N/32, K/32]`` (uint8 / e8m0, numpy or torch) -> ``[N/32, 4]`` int32: byte ``j`` of
    dword ``[r, g]`` = the scale of 32-K block ``4 j + g`` (127 = 1.0 past ``K / 32``)."""
    kb = k // BLK
    try:
        import torch

        is_torch = isinstance(ws, torch.Tensor)
    except ImportError:  # pragma: no cover
        is_torch = False
    if is_torch:
        import torch

        w = ws.view(torch.uint8).to(torch.int64)
        if w.shape[1] != kb:
            raise ValueError("scale grid does not match K")
        out = torch.full((w.shape[0], 4, 4), 127, dtype=torch.int64, device=w.device)  # [r, g, j]
        for b in range(kb):
            out[:, b % 4, b // 4] = w[:, b]
        sh = torch.tensor([0, 8, 16, 24], dtype=torch.int64, device=w.device)
        v = (out << sh).sum(-1)
        v = torch.where(v >= 2**31, v - 2**32, v)
        return v.to(torch.int32).contiguous()
    import numpy as np

    w = np.asarray(ws).view(np.uint8).astype(np.int64)
    if w.shape[1] != kb:
        raise ValueError("scale grid does not match K")
    out = np.full((w.shape[0], 4, 4), 127, dtype=np.int64)
    for b in range(kb):
        out[:, b % 4, b // 4] = w[:, b]
    packed = (out << np.array([0, 8, 16, 24], dtype=np.int64)).sum(-1)
    return packed.astype(np.uint32).view(np.int32)


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global")), nbytes)


def _fbits(b: IRBuilder, bits: int) -> Value:
    return b.bitcast(_c(b, bits), F32)


def quantize32(b: IRBuilder, f: list):
    """32 f32 values -> (8 code dwords, E8M0 code): ATOM ``quantize_fp8`` bit for bit."""
    amax = b.fabs(f[0])
    for i in range(1, 32):
        amax = b.fmax(amax, b.fabs(f[i]))
    amax = b.fmax(amax, _fbits(b, AMAX_FLOOR_BITS))
    bits = b.bitcast(b.fmul(amax, _fbits(b, RATIO_BITS)), I32)
    code = b.add(b.lshr(bits, _c(b, 23)), b.select(b.cmp_ne(b.land(bits, _c(b, 0x7FFFFF)), _c(b, 0)), _c(b, 1), _c(b, 0)))
    scale = b.bitcast(b.shl(code, _c(b, 23)), F32)
    cw = []
    for d in range(8):
        wd = b.cvt_scalef32_pk_fp8_f32(_c(b, 0), f[4 * d], f[4 * d + 1], scale, hi=False)
        wd = b.cvt_scalef32_pk_fp8_f32(wd, f[4 * d + 2], f[4 * d + 3], scale, hi=True)
        cw.append(wd)
    return cw, code


class _K288:
    def __init__(self, k: K288Knobs):
        k.check()
        self.k = k
        self.b = b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = k.threads
        self.a = {n: b.param(n, I64) for n in PARAMS[:5]}
        self.N = b.readfirstlane(b.param("n", I32))
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.cta = b.readfirstlane(b.block_id_x())
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))
        self.l16 = b.land(self.lane, _c(b, 15))
        R, BKT = k.rows_per_cta, k.bucket
        RG = b.readfirstlane(b.div(b.add(self.N, _c(b, R - 1)), _c(b, R)))
        if k.order == "nfast":
            self.rg = b.readfirstlane(b.mod(self.cta, RG))
            self.mblk = b.readfirstlane(b.div(self.cta, RG))
        else:
            MB = b.readfirstlane(b.div(b.add(b.smax(self.rows, _c(b, 1)), _c(b, BKT - 1)), _c(b, BKT)))
            self.mblk = b.readfirstlane(b.mod(self.cta, MB))
            self.rg = b.readfirstlane(b.div(self.cta, MB))
        self.m0 = b.readfirstlane(b.mul(self.mblk, _c(b, BKT)))  # first token of this M block
        # activations of the M block this CTA stages (the mblk mutant: block b > 0 stages block b - 1)
        self.xm0 = self.m0
        if k.mutant == "mblk":
            self.xm0 = b.readfirstlane(b.select(b.cmp_gt(self.mblk, _c(b, 0)), b.sub(self.m0, _c(b, BKT)), self.m0))
        K = k.k
        self.w_rs = _rsrc(b, self.a["w"], b.mul(self.N, _c(b, K)))
        self.ws_rs = _rsrc(b, self.a["wsp"], b.mul(b.lshr(self.N, _c(b, 5)), _c(b, 16)))
        rl = b.smax(self.rows, _c(b, 1))
        if k.xin == "bf16":
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, 2 * K)))
        else:
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, K)))
            self.xs_ptr = b.int_to_ptr(b.readfirstlane(self.a["xs"]), PtrType(I8, "global"))
        self.y_rs = _rsrc(b, self.a["y"], b.mul(b.mul(rl, self.N), _c(b, 2)))
        self.aux = WCPS[k.wcp]
        self.xs = b.smem_alloc(I8, [BKT, k.xrow()], name_hint="kxs")
        self.xss = b.smem_alloc(I8, [BKT, 4, 4], name_hint="kxss")
        if k.ep == "lds":
            self.ys = b.smem_alloc(I32, [k.waves, 2, 16, 8], name_hint="kys")  # [wave][patch][token][row pair] bf16x2
        self.rowbase = []
        for mi in range(k.mt):
            self.rowbase.append(b.readfirstlane(b.add(b.mul(self.rg, _c(b, R)), b.mul(b.add(_c(b, mi * k.waves), self.wave), _c(b, TILE)))))
        self.zero8 = b.vec_pack([_c(b, 0)] * 4, I32)

    # ------------------------------------------------------------------ weights
    def half_blocks(self, c: int, h: int) -> int:
        """Valid 32-K blocks (0, 1 or 2) in half ``h`` of chunk ``c``."""
        return max(0, min(2, self.k.kb - (4 * c + 2 * h)))

    def wissue(self, c: int) -> list:
        """Chunk ``c``'s A operand halves per weight tile (a load, or the constant zero for a half past K)."""
        b, k = self.b, self.k
        K = k.k
        out = []
        for mi in range(k.mt):
            rt = b.lshr(self.rowbase[mi], _c(b, 4))
            ok = b.cmp_lt(self.rowbase[mi], self.N)
            base = b.add(b.add(b.mul(rt, _c(b, 16 * K)), _c(b, c * 2048)), b.mul(self.lane, _c(b, 16)))
            for h in range(2):
                nb = self.half_blocks(c, h)
                if k.mutant == "tail" and c == k.nq - 1:
                    nb = 0  # the K-tail chunk dropped
                if nb == 0:
                    out.append(self.zero8)
                    continue
                okh = ok
                if nb == 1 and k.mutant != "tailx":
                    okh = b.land(ok, b.cmp_lt(self.lane, _c(b, 32)))
                off = b.select(okh, b.add(base, _c(b, 1024 * h)), _c(b, OOB))
                out.append(b.buffer_load_cp(self.w_rs, off, _c(b, 0), dwords=4, aux=self.aux))
        return out

    def wscale(self) -> list:
        b, k = self.b, self.k
        out = []
        for mi in range(k.mt):
            r32 = b.lshr(b.add(self.rowbase[mi], self.l16), _c(b, 5))
            ok = b.cmp_lt(self.rowbase[mi], self.N)
            off = b.mul(b.add(b.mul(r32, _c(b, 4)), self.g), _c(b, 4))
            out.append(b.buffer_load_cp(self.ws_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=1))
        return out

    # ------------------------------------------------------------------ activations
    def _items(self) -> int:
        return self.k.bucket * 4 * self.k.nq  # (token, 32-K block incl. pad blocks)

    def xissue(self, it0: int, it1: int) -> list:
        b, k = self.b, self.k
        K, per = k.k, 4 * k.nq
        out = []
        for it in range(it0, it1):
            v = b.add(self.tid, _c(b, it * k.threads))
            tok = b.div(v, _c(b, per))
            blk = b.sub(v, b.mul(tok, _c(b, per)))  # 32-K block (pad blocks >= kb)
            xtok = tok
            if k.mutant == "row":
                xtok = b.select(b.cmp_eq(b.land(tok, _c(b, 15)), _c(b, 0)), b.add(tok, _c(b, 1)),
                                b.select(b.cmp_eq(b.land(tok, _c(b, 15)), _c(b, 1)), b.sub(tok, _c(b, 1)), tok))
            gtok = b.add(self.xm0, xtok)
            live = b.land(b.cmp_lt(v, _c(b, self._items())), b.cmp_lt(b.add(self.m0, tok), self.rows))
            ok = b.land(live, b.land(b.cmp_lt(blk, _c(b, k.kb)), b.cmp_lt(gtok, self.rows)))
            kel = b.add(b.mul(blk, _c(b, 32)), b.mul(gtok, _c(b, K)))  # element index
            if k.xin == "bf16":
                base = b.mul(kel, _c(b, 2))
                regs = [b.buffer_load_cp(self.x_rs, b.select(ok, b.add(base, _c(b, 16 * i)), _c(b, OOB)), _c(b, 0), dwords=4) for i in range(4)]
            else:
                regs = [b.buffer_load_cp(self.x_rs, b.select(ok, b.add(kel, _c(b, 16 * i)), _c(b, OOB)), _c(b, 0), dwords=4) for i in range(2)]
                # the E8M0 code: a byte load (a dword load at so & ~3 can straddle the end of the [M, K/32] tensor when
                # K / 32 is not a multiple of 4, and an out-of-range dword reads 0); element 0 when the item is dead
                so = b.select(ok, b.add(b.mul(gtok, _c(b, K // 32)), blk), _c(b, 0))
                regs.append(b.zext(b.global_load_i8(self.xs_ptr, so), I32))
            out.append((tok, blk, live, ok, regs))
        return out

    def xput(self, items: list) -> None:
        b, k = self.b, self.k
        for tok, blk, live, ok, regs in items:
            if k.xin == "bf16" and k.proto == "noquant":
                halves = [regs[0], regs[1]]
                code = b.land(b.vec_extract(regs[2], 0), _c(b, 127))
            elif k.xin == "bf16":
                f = []
                for r in regs:
                    for e in range(4):
                        w = b.vec_extract(r, e)
                        f.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
                        f.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
                cw, code = quantize32(b, f)
                halves = [b.vec_pack(cw[0:4], I32), b.vec_pack(cw[4:8], I32)]
            else:
                halves = [regs[0], regs[1]]
                code = b.land(regs[2], _c(b, 255))
            if k.mutant == "xq":
                code = b.select(b.cmp_eq(b.land(blk, _c(b, 3)), _c(b, 1)), b.add(code, _c(b, 1)), code)
            # pad blocks (blk >= kb) loaded nothing (out-of-range offsets read 0): zero codes; scale 127 = 1.0, so every
            # product past K is an exact 0 x 0
            code = b.select(b.cmp_ge(blk, _c(b, k.kb)), _c(b, 127), code)
            g = b.land(blk, _c(b, 3))
            j = b.lshr(blk, _c(b, 2))
            col = b.mul(blk, _c(b, 32))
            with b.scf_if(live):
                for h in range(2):
                    b.smem_store_vN(self.xs, [tok, b.add(col, _c(b, 16 * h))], b.vec_bitcast(halves[h], VectorType(I8, 16)), 16)
                b.smem_store_vN(self.xss, [tok, g, j], b.trunc(code, I8), 1)

    def stage(self) -> None:
        k = self.k
        n_it = -(-self._items() // k.threads)
        for i0 in range(0, n_it, k.xb):
            self.xput(self.xissue(i0, min(n_it, i0 + k.xb)))

    # ------------------------------------------------------------------ math
    def bop(self, t: int, c: int):
        b = self.b
        row = b.add(_c(b, t * TILE), self.l16)
        halves = []
        for h in range(2):
            col = b.add(_c(b, c * CHUNK + 64 * h), b.mul(self.g, _c(b, 16)))
            hv = b.smem_load_vN(self.xs, row, col, dtype=I8, n=16)
            halves.append(b.vec_bitcast(hv, VectorType(I32, 4)))
        return b.vec_concat(halves[0], halves[1])

    def bscale(self, t: int):
        b = self.b
        row = b.add(_c(b, t * TILE), self.l16)
        v = b.smem_load_vN(self.xss, row, self.g, _c(b, 0), dtype=I8, n=4)
        return b.vec_extract(b.vec_bitcast(v, VectorType(I32, 1)), 0)

    def run(self) -> KernelDef:
        b, k = self.b, self.k
        if k.proto == "empty":
            return b.kernel
        if k.proto == "storeonly":
            self.epilogue([[b.zero_vec_f32(4) for _ in range(k.tiles)] for _ in range(k.mt)])
            return b.kernel
        NQ, mt = k.nq, k.mt
        Dp = min(k.depth, NQ)
        wq = []
        SA = None
        if k.wfirst:
            SA = self.wscale()
            for c in range(min(Dp, NQ)):
                wq.append(self.wissue(c))
        self.stage()
        if not k.wfirst:
            SA = self.wscale()
            for c in range(min(Dp, NQ)):
                wq.append(self.wissue(c))
        b.sync_lds_only()
        acc = [[b.zero_vec_f32(4) for _ in range(k.tiles)] for _ in range(mt)]
        SB = [self.bscale(t) for t in range(k.tiles)]
        for c in range(NQ):
            if k.sb:
                b.sched_barrier(0)
            if c + Dp < NQ:
                wq.append(self.wissue(c + Dp))
            if k.sb:
                b.sched_barrier(0)
            cur = wq.pop(0)
            A = [b.vec_concat(cur[2 * mi], cur[2 * mi + 1]) for mi in range(mt)]
            osa = (c + 1) % 4 if (k.mutant == "scale" and c == 1) else c
            if k.bra == 1:
                Bs = [self.bop(t, c) for t in range(k.tiles)]
                if k.sb:
                    b.sched_barrier(0)
            for t in range(k.tiles):
                bv = self.bop(t, c) if k.bra == 0 else Bs[t]
                for mi in range(mt):
                    if k.proto == "nomath":
                        lo = b.vec_bitcast(b.vec_pack([b.lshr(b.vec_extract(A[mi], e), _c(b, 9)) for e in range(4)], I32),
                                           VectorType(F32, 4))
                        hi = b.vec_bitcast(b.vec_pack([b.lshr(b.vec_extract(bv, e), _c(b, 9)) for e in range(4)], I32), VectorType(F32, 4))
                        acc[mi][t] = b.fadd(acc[mi][t], b.fadd(lo, hi))
                        continue
                    acc[mi][t] = b.mfma_scale_mx(A[mi], bv, acc[mi][t], SA[mi], SB[t], shape="16x16x128", fmt_a="fp8",
                                                 fmt_b="fp8", opsel_a=osa, opsel_b=c)
        if k.sb:
            b.sched_barrier(0)
        self.epilogue(acc)
        return b.kernel

    # ------------------------------------------------------------------ epilogue
    def epilogue(self, acc) -> None:
        b, k = self.b, self.k
        if k.ep == "direct":
            for mi in range(k.mt):
                row = b.add(self.rowbase[mi], b.mul(self.g, _c(b, 4)))
                for t in range(k.tiles):
                    tok = b.add(self.m0, b.add(_c(b, t * TILE), self.l16))
                    ok = b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(row, self.N))
                    off = b.mul(b.add(b.mul(tok, self.N), row), _c(b, 2))
                    v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    b.buffer_store_cp(v, self.y_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0))
            return
        # ep == "lds": token tiles in pairs. Lane l writes its C fragment (rows 4g..4g+3 of token l % 16) as two bf16 pairs
        # into the wave's patch [token][row pair]; then lane l reads patch l / 32, token (l / 2) % 16, rows 8 (l % 2)
        # .. +7 (four dwords) and stores 16 contiguous bytes. One wave's own LDS: in-order, no barrier.
        which = b.lshr(self.lane, _c(b, 5))
        tq = b.land(b.lshr(self.lane, _c(b, 1)), _c(b, 15))
        part = b.land(self.lane, _c(b, 1))
        for mi in range(k.mt):
            for t0 in range(0, k.tiles, 2):
                pair = [t for t in (t0, t0 + 1) if t < k.tiles]
                for p, t in enumerate(pair):
                    v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    for e in range(2):
                        b.smem_store_vN(self.ys, [self.wave, _c(b, p), self.l16, b.add(b.mul(self.g, _c(b, 2)), _c(b, e))],
                                        b.vec_extract(v, e), 1)
                w4 = b.smem_load_vN(self.ys, self.wave, which, tq, b.mul(part, _c(b, 4)), dtype=I32, n=4)
                tok = b.add(self.m0, b.add(b.add(_c(b, t0 * TILE), b.mul(which, _c(b, TILE))), tq))
                row = b.add(self.rowbase[mi], b.mul(part, _c(b, 8)))
                ok = b.land(b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(row, self.N)), b.cmp_lt(which, _c(b, len(pair))))
                off = b.mul(b.add(b.mul(tok, self.N), row), _c(b, 2))
                b.buffer_store_cp(w4, self.y_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0))


def build_k288_kernel(k: K288Knobs) -> KernelDef:
    return _K288(k).run()


# ------------------------------------------------------------------ host reference (numpy)


def e4m3_to_float(codes):
    """E4M3FN byte codes -> float64 (numpy)."""
    import numpy as np

    c = np.asarray(codes).astype(np.int64) & 0xFF
    s = np.where(c & 0x80, -1.0, 1.0)
    e = (c >> 3) & 0xF
    m = c & 7
    val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0 ** (e - 7))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return s * val


def reference(w_rowmajor_codes, ws, x_q_codes, x_scale_codes):
    """float64 ``Y [rows, N]`` of the MXFP8 GEMM (row-major E4M3 weight codes [N, K], E8M0 [N/32, K/32]; activation
    codes [rows, K], E8M0 [rows, K/32])."""
    import numpy as np

    W = e4m3_to_float(w_rowmajor_codes)
    n, _kk = W.shape
    sw = np.exp2(np.asarray(ws).astype(np.float64) - 127.0)
    W = W * np.repeat(np.repeat(sw, 32, 0)[:n], 32, 1)
    X = e4m3_to_float(x_q_codes) * np.repeat(np.exp2(np.asarray(x_scale_codes).astype(np.float64) - 127.0), 32, 1)
    return X @ W.T
