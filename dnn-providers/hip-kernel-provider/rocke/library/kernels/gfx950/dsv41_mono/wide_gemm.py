# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP6 wide-row FP8 dense GEMM for DeepSeek-V4.1-Flash decode (gfx950): ``Y = X W^T`` at 48-384 rows.

``experiments/dsv41-mono/D0.md`` WP6 ("E1-style persistent FP8 dense GEMMs"), probe ``wide/P4.md`` section 7: ATOM's
native group32 FP8 linear (``atom/model_ops/blockscale.py native_quant_linear``: a ``quantize_fp8`` launch, then aiter's
preshuffled group32 GEMM) runs at 0.08-0.40 of its weight-bytes floor at these widths. This kernel replaces both
launches with one:

* **Weights** as ATOM stores them: E4M3 ``[N, K]`` in aiter's (16, 16) preshuffle (``shuffle_weight``: per 16 rows and
  32 K, two 16-byte halves of 16 rows), read straight into the MFMA A operand exactly as ATOM's own mono K1 GEMV does
  (``stage_wqkv.py``): lane ``l`` of a 128-K chunk takes the 16 bytes at ``(rt K/64 + 2 q) 1024 + 16 l`` (K
  ``16 (l / 16) ..``) and the 16 bytes 1024 further (K ``64 + 16 (l / 16) ..``). Weight scales: ATOM's E8M0 ``[N/32,
  K/32]`` grid, repacked on the host (:func:`pack_wscale`) so one dword holds the four chunks of a group (``op_sel``
  picks the byte: a scale VGPR rewritten for every MFMA runs about 8x slower on gfx950, qwen38 P-L3).
* **Activations**: ``xin="bf16"`` (the fused seam): the CTA quantizes its K slice of X in the prologue, ATOM's
  ``quantize_fp8`` rule (E8M0 ceil scale of ``max(amax, 1e-4) / 448``, E4M3 round-to-nearest-even), straight into LDS.
  ``xin="fp8"``: X is already MXFP8 (ATOM's ``quantize_fp8`` output: codes ``[rows, K]`` + codes ``[rows, K/32]``).
* **Work division**: one CTA per (row group of ``R = 16 waves mt`` weight rows, K split ``s``), every token tile in the
  CTA (each weight byte is read once). With ``order="sfast"`` CTA ``c`` takes split ``c mod S``: for ``S | 8`` every XCD
  reads one X slice only. ``splits > 1``: f32 partials (``sc1`` stores), an agent-scope arrival counter per row group,
  the last arriver sums the splits in split order (deterministic: the same bits whoever arrives last) and resets the
  counter.
* MFMA ``v_mfma_scale_f32_16x16x128_f8f6f4`` (fp8 x fp8), weights as A, the token tile as B, f32 accumulators, bf16
  out (``Y [rows, N]`` row-major: a lane stores 4 consecutive rows of one token).

Mutants (must fail the numerics gate): ``kskip`` (the last chunk adds nothing), ``scale`` (chunk 1 picks the next
weight-scale byte), ``fold`` (the fold drops the last split), ``xq`` (one quantization block in four stores its scale
one binade high), ``tok`` (every token tile reads tile 0's activations).

Kernel arguments (``PARAMS``): i64 ``w, wsp, x, xs, y, part, cnt``, then i32 ``n, rows``. ``xs`` is read only with
``xin="fp8"``; ``part`` / ``cnt`` only with ``splits > 1`` (``part`` holds ``splits x 16 tiles x n`` f32, ``cnt`` one
zeroed word per row group, left zero). Grid ``ceil(n / R) * splits`` CTAs of ``64 waves`` threads. Python engine only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields

from rocke.core.ir import CPOL_NT, CPOL_SC1, F32, I32, I64, I8, IRBuilder, KernelDef, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

__all__ = [
    "PARAMS",
    "WideKnobs",
    "build_wide_kernel",
    "grid_of",
    "kernel_name",
    "pack_wscale",
    "part_floats",
]

TILE = 16
CHUNK = 128
XPAD = 16  # LDS row pad (bytes) of the staged activation codes
OOB = 0x7FFFFFF0
RATIO_BITS = 0x3B124925  # fl(1 / 448): ATOM quantize_fp8 ``amax * (1.0 / FP8_MAX)``
AMAX_FLOOR_BITS = struct.unpack("<I", struct.pack("<f", 1e-4))[0]  # ATOM ``tl.maximum(amax, 1e-4)`` (f32)
MUTANTS = ("none", "kskip", "scale", "fold", "xq", "tok")
WCPS = {"default": 0, "nt": CPOL_NT}
PARAMS = ("w", "wsp", "x", "xs", "y", "part", "cnt", "n", "rows")


@dataclass(frozen=True)
class WideKnobs:
    """One build. The kernel name lists the deviations from the defaults."""

    k: int = 5120  # K (a multiple of 128)
    tiles: int = 6  # token tiles (bucket = 16 tiles rows)
    xin: str = "bf16"  # "bf16" (fused MXFP8 quantization) | "fp8" (pre-quantized ATOM MXFP8)
    waves: int = 8  # waves per CTA (1 | 2 | 4 | 8)
    mt: int = 1  # 16-row weight tiles per wave (1 | 2 | 4)
    splits: int = 1  # K splits (divides K / 128)
    depth: int = 3  # weight chunks in flight
    wcp: str = "nt"  # weight cache policy
    xb: int = 16  # staging items per thread per load batch
    order: str = "sfast"  # CTA -> (row group, split): "sfast" (split = cta mod S) | "rgfast" (row group = cta mod RG) |
    #   "xcd" (every split of a row group on one XCD: CTA c on XCD c mod 8 takes row group 8 (c / 8 / S) + c mod 8)
    fold: str = "sc1"  # splits > 1: "sc1" (partials written through, sc1 loads) | "l2" (plain partials in the XCD's L2:
    #   needs order="xcd")
    wfirst: bool = False  # issue the first weight chunks before the activation loads
    gch: int = 4  # chunks per activation staging group (1 | 2 | 4): the X scale dword of a (token, block) holds a
    #   group's chunks, picked with op_sel
    pipe: int = 0  # 0: every staging group staged in the prologue | 1: two LDS slots, group g + 1 loaded during group
    #   g's MFMAs (written after them)
    xsrc: str = "lds"  # "lds": activations staged per CTA in LDS (xin bf16: fused quantization) | "glb": no LDS, every
    #   wave loads its B operands from the operand-order MXFP8 buffer of :func:`build_quant_kernel` (xin must be "fp8")
    stamp: bool = False  # thread 0 writes s_memrealtime at the phase points to ``st`` (extra i64 argument, 8 per CTA)
    sb: bool = True  # sched_barrier fences around each chunk's weight issue
    bra: int = 1  # B (activation) LDS reads: 0 next to each MFMA | 1 a chunk's tiles before its MFMAs | 2 one chunk ahead
    occ1: bool = False  # pad LDS so one CTA fits a CU
    mutant: str = "none"
    tag: str = ""  # recompiled-control suffix

    @property
    def nq(self) -> int:
        return self.k // CHUNK

    @property
    def kc(self) -> int:
        return self.nq // self.splits

    @property
    def groups(self) -> int:
        return -(-self.kc // 4)

    @property
    def sgroups(self) -> int:
        """Activation staging groups."""
        return -(-self.kc // self.gch)

    @property
    def nslots(self) -> int:
        return self.sgroups if self.pipe == 0 else min(2, self.sgroups)

    @property
    def rows_per_cta(self) -> int:
        return TILE * self.waves * self.mt

    @property
    def threads(self) -> int:
        return 64 * self.waves

    @property
    def bucket(self) -> int:
        return TILE * self.tiles

    def lds_bytes(self) -> int:
        if self.xsrc == "glb":
            return 64
        return self.nslots * self.bucket * (self.gch * CHUNK + XPAD + 16) + 64

    def check(self) -> None:
        if self.k <= 0 or self.k % CHUNK:
            raise ValueError("k must be a positive multiple of 128")
        if not 1 <= self.tiles <= 24:
            raise ValueError("tiles 1..24")
        if self.xin not in ("bf16", "fp8"):
            raise ValueError("xin bf16 | fp8")
        if self.waves not in (1, 2, 4, 8) or self.mt not in (1, 2, 4):
            raise ValueError("waves 1 | 2 | 4 | 8, mt 1 | 2 | 4")
        if self.splits < 1 or self.nq % self.splits:
            raise ValueError(f"splits {self.splits} must divide the {self.nq} K chunks")
        if not 1 <= self.depth <= 20:
            raise ValueError("depth 1..20")
        if not 1 <= self.xb <= 64:
            raise ValueError("xb 1..64")
        if self.wcp not in WCPS:
            raise ValueError(f"wcp in {sorted(WCPS)}")
        if self.order not in ("sfast", "rgfast", "xcd"):
            raise ValueError("order sfast | rgfast | xcd")
        if self.fold not in ("sc1", "l2") or (self.fold == "l2" and self.order != "xcd"):
            raise ValueError("fold sc1 | l2 (l2 needs order=xcd)")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.bra not in (0, 1, 2):
            raise ValueError("bra 0 | 1 | 2")
        if self.gch not in (1, 2, 4) or self.pipe not in (0, 1):
            raise ValueError("gch 1 | 2 | 4, pipe 0 | 1")
        if self.xsrc not in ("lds", "glb") or (self.xsrc == "glb" and self.xin != "fp8"):
            raise ValueError("xsrc lds | glb (glb reads build_quant_kernel's buffer: xin fp8)")
        if self.pipe == 1 and self.bra == 2:
            raise ValueError("bra 2 reads the next chunk ahead: not with pipe 1")
        if self.mutant == "fold" and self.splits < 2:
            raise ValueError("the fold mutant needs splits > 1")
        if self.mutant == "tok" and self.tiles < 2:
            raise ValueError("the tok mutant needs tiles > 1")
        if self.mutant == "scale" and self.kc < 2:
            raise ValueError("the scale mutant needs 2+ chunks per split")
        if self.lds_bytes() > 160 * 1024 - 1024:
            raise ValueError(f"LDS {self.lds_bytes()} bytes too large: raise splits or lower tiles")
        if self.mt * self.tiles > 48:
            raise ValueError("mt x tiles > 48 accumulators")

    def suffix(self) -> str:
        parts = []
        d = WideKnobs()
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


def kernel_name(k: WideKnobs) -> str:
    s = k.suffix()
    return f"rk41w_k{k.k}_t{k.tiles}_{k.xin}" + (f"_{s}" if s else "")


def grid_of(k: WideKnobs, n: int) -> int:
    rg = -(-n // k.rows_per_cta)
    if k.order == "xcd" and k.splits > 1:
        return 8 * (-(-rg // 8)) * k.splits
    return rg * k.splits


def part_floats(k: WideKnobs, n: int) -> int:
    """f32 elements of the ``part`` scratch a launch of ``n`` columns needs (0 without splits)."""
    return 0 if k.splits == 1 else k.splits * k.bucket * n


def pack_wscale(ws, k: WideKnobs):
    """ATOM's E8M0 weight scales ``[N/32, K/32]`` (uint8, numpy or torch) -> ``[N/32, S, G, 4]`` int32: dword
    ``[r, s, p, g]`` holds in byte ``j`` the scale of 32-K block ``4 (s kc + 4 p + j) + g`` (127 past the split)."""
    S, kc, G = k.splits, k.kc, k.groups
    try:
        import torch

        is_torch = isinstance(ws, torch.Tensor)
    except ImportError:  # pragma: no cover
        is_torch = False
    if is_torch:
        import torch

        w = ws.view(torch.uint8).to(torch.int32)
        n32 = w.shape[0]
        out = torch.full((n32, S, G, 4, 4), 127, dtype=torch.int32, device=w.device)  # [r, s, p, g, j]
        for s in range(S):
            for c in range(kc):
                p, j = divmod(c, 4)
                q = s * kc + c
                out[:, s, p, :, j] = w[:, 4 * q : 4 * q + 4]
        sh = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=w.device)
        return (out << sh).sum(-1).to(torch.int32).contiguous()
    import numpy as np

    w = np.asarray(ws).view(np.uint8).astype(np.int64)
    n32 = w.shape[0]
    out = np.full((n32, S, G, 4, 4), 127, dtype=np.int64)
    for s in range(S):
        for c in range(kc):
            p, j = divmod(c, 4)
            q = s * kc + c
            out[:, s, p, :, j] = w[:, 4 * q : 4 * q + 4]
    packed = (out << np.array([0, 8, 16, 24], dtype=np.int64)).sum(-1)
    return packed.astype(np.uint32).view(np.int32)


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(b.int_to_ptr(b.readfirstlane(addr), PtrType(I32, "global")), nbytes)


def _fbits(b: IRBuilder, bits: int) -> Value:
    return b.bitcast(_c(b, bits), F32)


class _Wide:
    def __init__(self, k: WideKnobs):
        k.check()
        self.k = k
        self.b = b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = k.threads
        self.a = {n: b.param(n, I64) for n in PARAMS[:7]}
        if k.stamp:
            self.a["st"] = b.param("st", I64)
        self.N = b.readfirstlane(b.param("n", I32))
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.cta = b.readfirstlane(b.block_id_x())
        self.stamp(0)
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))
        self.l16 = b.land(self.lane, _c(b, 15))
        S, R = k.splits, k.rows_per_cta
        if k.order == "xcd" and S > 1:
            x = b.land(self.cta, _c(b, 7))
            j = b.lshr(self.cta, _c(b, 3))
            self.s = b.readfirstlane(b.mod(j, _c(b, S)))
            self.rg = b.readfirstlane(b.add(b.mul(b.div(j, _c(b, S)), _c(b, 8)), x))
        elif k.order == "sfast" or S == 1:
            self.s = b.readfirstlane(b.mod(self.cta, _c(b, S))) if S > 1 else _c(b, 0)
            self.rg = b.readfirstlane(b.div(self.cta, _c(b, S))) if S > 1 else self.cta
        else:
            RG = b.readfirstlane(b.div(b.add(self.N, _c(b, R - 1)), _c(b, R)))
            self.rg = b.readfirstlane(b.mod(self.cta, RG))
            self.s = b.readfirstlane(b.div(self.cta, RG))
        self.q0 = b.mul(self.s, _c(b, k.kc))  # first global chunk of this split
        K = k.k
        self.w_rs = _rsrc(b, self.a["w"], b.mul(self.N, _c(b, K)))
        self.ws_rs = _rsrc(b, self.a["wsp"], b.mul(b.lshr(self.N, _c(b, 5)), _c(b, S * k.groups * 16)))
        rl = b.smax(self.rows, _c(b, 1))
        self.rlast = b.sub(rl, _c(b, 1))
        if k.xsrc == "glb":
            self.x_rs = _rsrc(b, self.a["x"], _c(b, k.nq * k.tiles * 64 * 32))
            self.xs_rs = _rsrc(b, self.a["xs"], _c(b, S * k.groups * k.tiles * 64 * 4))
        elif k.xin == "bf16":
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, 2 * K)))
        else:
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, K)))
            self.xs_rs = _rsrc(b, self.a["xs"], b.mul(rl, _c(b, K // 32)))
        self.y_rs = _rsrc(b, self.a["y"], b.mul(b.mul(rl, self.N), _c(b, 2)))
        self.aux = WCPS[k.wcp]
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
        TP = k.bucket
        if k.xsrc == "lds":
            self.xs = b.smem_alloc(I8, [k.nslots, TP, k.gch * CHUNK + XPAD], name_hint="wxs")
            self.xss = b.smem_alloc(I8, [k.nslots, TP, 4, 4], name_hint="wxss")
        if S > 1:
            self.flag = b.smem_alloc(F32, [1], name_hint="wfold")
        self.rowbase = []  # first weight row of each of this wave's tiles (uniform)
        for mi in range(k.mt):
            self.rowbase.append(b.readfirstlane(b.add(b.mul(self.rg, _c(b, R)), b.mul(b.add(_c(b, mi * k.waves), self.wave), _c(b, TILE)))))

    def stamp(self, p: int) -> None:
        if not self.k.stamp:
            return
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            ptr = b.int_to_ptr(b.readfirstlane(self.a["st"]), PtrType(I64, "global"))
            b.store(D.memrealtime(b), ptr, b.add(b.mul(self.cta, _c(b, 8)), _c(b, p)))

    # ------------------------------------------------------------------ weights
    def wissue(self, c: int) -> list:
        """Chunk ``c``'s A operands (two dwordx4 per weight tile), plus at a group start its packed scale dwords."""
        b, k = self.b, self.k
        K = k.k
        q = b.add(self.q0, _c(b, c))
        out = []
        for mi in range(k.mt):
            rt = b.lshr(self.rowbase[mi], _c(b, 4))
            ok = b.cmp_lt(self.rowbase[mi], self.N)
            base = b.add(b.mul(b.add(b.mul(rt, _c(b, K // 64)), b.mul(q, _c(b, 2))), _c(b, 1024)), b.mul(self.lane, _c(b, 16)))
            for h in range(2):
                off = b.select(ok, b.add(base, _c(b, 1024 * h)), _c(b, OOB))
                out.append(b.buffer_load_cp(self.w_rs, off, _c(b, 0), dwords=4, aux=self.aux))
        if c % 4 == 0:
            p = c // 4
            for mi in range(k.mt):
                r32 = b.lshr(b.add(self.rowbase[mi], self.l16), _c(b, 5))
                ok = b.cmp_lt(self.rowbase[mi], self.N)
                off = b.mul(b.add(b.mul(b.add(b.mul(r32, _c(b, k.splits)), self.s), _c(b, k.groups * 4)), b.add(_c(b, 4 * p), self.g)), _c(b, 4))
                out.append(b.buffer_load_cp(self.ws_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=1))
        if k.xsrc == "glb":
            for t in range(k.tiles):
                tt = 0 if k.mutant == "tok" else t
                base = b.mul(b.add(b.mul(b.add(b.mul(q, _c(b, k.tiles)), _c(b, tt)), _c(b, 64)), self.lane), _c(b, 32))
                for h in range(2):
                    out.append(b.buffer_load_cp(self.x_rs, b.add(base, _c(b, 16 * h)), _c(b, 0), dwords=4))
            if c % 4 == 0:
                p = c // 4
                for t in range(k.tiles):
                    tt = 0 if k.mutant == "tok" else t
                    off = b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(self.s, _c(b, k.groups)), _c(b, p)), _c(b, k.tiles)), _c(b, tt)), _c(b, 64)), self.lane), _c(b, 4))
                    out.append(b.buffer_load_cp(self.xs_rs, off, _c(b, 0), dwords=1))
        return out

    # ------------------------------------------------------------------ activations
    def _items(self):
        k = self.k
        return k.bucket * k.gch * 4

    def xissue(self, sg: int) -> list:
        """This thread's loads of staging group ``sg`` (items = (token, 32-K block of the group's chunks))."""
        b, k = self.b, self.k
        K = k.k
        per = k.gch * 4
        out = []
        for it in range(-(-self._items() // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tok = b.div(v, _c(b, per))
            bl = b.sub(v, b.mul(tok, _c(b, per)))  # block within the group: chunk bl / 4, lane group bl % 4
            c = b.add(_c(b, sg * k.gch), b.lshr(bl, _c(b, 2)))  # chunk within the split
            ok = b.land(b.land(b.cmp_lt(v, _c(b, self._items())), b.cmp_lt(tok, self.rows)), b.cmp_lt(c, _c(b, k.kc)))
            blk = b.add(b.mul(b.add(self.q0, c), _c(b, 4)), b.land(bl, _c(b, 3)))  # global 32-K block
            kel = b.add(b.mul(blk, _c(b, 32)), b.mul(tok, _c(b, K)))  # element index
            if k.xin == "bf16":
                base = b.mul(kel, _c(b, 2))
                regs = [b.buffer_load_cp(self.x_rs, b.select(ok, b.add(base, _c(b, 16 * i)), _c(b, OOB)), _c(b, 0), dwords=4) for i in range(4)]
            else:
                regs = [b.buffer_load_cp(self.x_rs, b.select(ok, b.add(kel, _c(b, 16 * i)), _c(b, OOB)), _c(b, 0), dwords=4) for i in range(2)]
                so = b.add(b.mul(tok, _c(b, K // 32)), blk)  # byte offset of the code
                w = b.buffer_load_cp(self.xs_rs, b.select(ok, b.land(so, _c(b, ~3)), _c(b, OOB)), _c(b, 0), dwords=1)
                regs.append((w, b.land(so, _c(b, 3))))
            out.append((tok, bl, ok, regs))
        return out

    def xput(self, sg: int, items: list) -> None:
        b, k = self.b, self.k
        slot = _c(b, sg % k.nslots)
        for tok, bl, ok, regs in items:
            if k.xin == "bf16":
                words = []
                for r in regs:
                    for e in range(4):
                        words.append(b.vec_extract(r, e))
                f = []
                for w in words:
                    f.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
                    f.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
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
                halves = [b.vec_pack(cw[0:4], I32), b.vec_pack(cw[4:8], I32)]
            else:
                halves = [regs[0], regs[1]]
                w, sh = regs[2]
                code = b.land(b.lshr(w, b.mul(sh, _c(b, 8))), _c(b, 255))
            if k.mutant == "xq":
                code = b.select(b.cmp_eq(b.land(bl, _c(b, 3)), _c(b, 1)), b.add(code, _c(b, 1)), code)
            g = b.land(bl, _c(b, 3))
            j = b.lshr(bl, _c(b, 2))
            col = b.mul(bl, _c(b, 32))
            with b.scf_if(ok):
                for h in range(2):
                    b.smem_store_vN(self.xs, [slot, tok, b.add(col, _c(b, 16 * h))], b.vec_bitcast(halves[h], VectorType(I8, 16)), 16)
                b.smem_store_vN(self.xss, [slot, tok, g, j], b.trunc(code, I8), 1)

    # ------------------------------------------------------------------ math
    def bop(self, t: int, c: int):
        b, k = self.b, self.k
        tt = 0 if k.mutant == "tok" else t
        row = b.add(_c(b, tt * TILE), self.l16)
        slot = _c(b, (c // k.gch) % k.nslots)
        halves = []
        for h in range(2):
            col = b.add(_c(b, (c % k.gch) * CHUNK + 64 * h), b.mul(self.g, _c(b, 16)))
            hv = self.b.smem_load_vN(self.xs, slot, row, col, dtype=I8, n=16)
            halves.append(b.vec_bitcast(hv, VectorType(I32, 4)))
        return b.vec_concat(halves[0], halves[1])

    def bscale(self, t: int, sg: int):
        b, k = self.b, self.k
        tt = 0 if k.mutant == "tok" else t
        row = b.add(_c(b, tt * TILE), self.l16)
        v = b.smem_load_vN(self.xss, _c(b, sg % k.nslots), row, self.g, _c(b, 0), dtype=I8, n=4)
        return b.vec_extract(b.vec_bitcast(v, VectorType(I32, 1)), 0)

    def run(self) -> KernelDef:
        b, k = self.b, self.k
        KC, Dp, mt = k.kc, k.depth, k.mt
        wq = []

        SGn, GC = k.sgroups, k.gch
        if k.xsrc == "glb":
            GC = 4
        pre = SGn if k.pipe == 0 else 1  # groups staged in the prologue
        if k.xsrc == "glb":
            for c in range(min(Dp, KC)):
                wq.append(self.wissue(c))
        elif k.wfirst:
            for c in range(min(Dp, KC)):
                wq.append(self.wissue(c))
        if k.xsrc == "lds":
            first = self.xissue(0)
            if not k.wfirst:
                for c in range(min(Dp, KC)):
                    wq.append(self.wissue(c))
            self.xput(0, first)
            for sg in range(1, pre):
                self.xput(sg, self.xissue(sg))
            b.sync_lds_only()
        self.stamp(1)
        pending = None
        acc = [[b.zero_vec_f32(4) for _ in range(k.tiles)] for _ in range(mt)]
        SA = SB = None
        zw = b.vec_pack([_c(b, 0)] * 8, I32) if k.mutant == "kskip" else None
        nxt = [self.bop(t, 0) for t in range(k.tiles)] if (k.bra == 2 and k.xsrc == "lds") else None
        for c in range(KC):
            sg, jb = divmod(c, GC)
            if k.xsrc == "lds" and k.pipe == 1 and jb == 0 and sg + 1 < SGn:
                pending = self.xissue(sg + 1)  # the next group's activations, written after this group's MFMAs
            if k.sb:
                b.sched_barrier(0)
            if c + Dp < KC:
                wq.append(self.wissue(c + Dp))
            if k.sb:
                b.sched_barrier(0)
            cur = wq.pop(0)
            p, j = divmod(c, 4)
            if j == 0:
                SA = cur[2 * mt : 3 * mt]
            if k.xsrc == "glb":
                o = 2 * mt + (mt if j == 0 else 0)
                gB = cur[o : o + 2 * k.tiles]
                if j == 0:
                    SB = cur[o + 2 * k.tiles : o + 3 * k.tiles]
            elif jb == 0:
                SB = [self.bscale(t, sg) for t in range(k.tiles)]
            A = [b.vec_concat(cur[2 * mi], cur[2 * mi + 1]) for mi in range(mt)]
            if k.mutant == "kskip" and c == KC - 1:
                A = [zw] * mt
            osa = (j + 1) % 4 if (k.mutant == "scale" and c == 1) else j
            if k.xsrc == "glb":
                Bs = [b.vec_concat(gB[2 * t], gB[2 * t + 1]) for t in range(k.tiles)]
            elif k.bra == 1:
                Bs = [self.bop(t, c) for t in range(k.tiles)]
                if k.sb:
                    b.sched_barrier(0)
            elif k.bra == 2 and k.xsrc == "lds":
                Bs = nxt
                nxt = [self.bop(t, c + 1) for t in range(k.tiles)] if c + 1 < KC else None
                if k.sb:
                    b.sched_barrier(0)
            for t in range(k.tiles):
                bv = self.bop(t, c) if (k.bra == 0 and k.xsrc == "lds") else Bs[t]
                for mi in range(mt):
                    acc[mi][t] = b.mfma_scale_mx(A[mi], bv, acc[mi][t], SA[mi], SB[t], shape="16x16x128", fmt_a="fp8",
                                                 fmt_b="fp8", opsel_a=osa, opsel_b=jb)
            if k.xsrc == "lds" and k.pipe == 1 and (jb == GC - 1 or c == KC - 1) and sg + 1 < SGn:
                if k.sb:
                    b.sched_barrier(0)
                self.xput(sg + 1, pending)
                pending = None
                b.sync_lds_only()
        if k.sb:
            b.sched_barrier(0)
        self.stamp(2)
        self.epilogue(acc)
        self.stamp(6)
        return b.kernel

    # ------------------------------------------------------------------ epilogue
    def epilogue(self, acc) -> None:
        b, k = self.b, self.k
        S = k.splits
        if S == 1:
            for mi in range(k.mt):
                row = b.add(self.rowbase[mi], b.mul(self.g, _c(b, 4)))
                for t in range(k.tiles):
                    tok = b.add(_c(b, t * TILE), self.l16)
                    ok = b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(row, self.N))
                    off = b.mul(b.add(b.mul(tok, self.N), row), _c(b, 2))
                    v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    b.buffer_store_cp(v, self.y_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0))
            return
        TP = k.bucket
        p_rs = _rsrc(b, self.a["part"], b.mul(self.N, _c(b, S * TP * 4)))
        pc = 0 if k.fold == "l2" else CPOL_SC1
        for mi in range(k.mt):
            row = b.add(self.rowbase[mi], b.mul(self.g, _c(b, 4)))
            for t in range(k.tiles):
                tok = b.add(_c(b, t * TILE), self.l16)
                ok = b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(row, self.N))
                off = b.mul(b.add(b.mul(b.add(b.mul(self.s, _c(b, TP)), tok), self.N), row), _c(b, 4))
                b.buffer_store_cp(b.vec_bitcast(acc[mi][t], VectorType(I32, 4)), p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), aux=pc)
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        self.stamp(3)
        live = b.cmp_lt(b.mul(self.rg, _c(b, k.rows_per_cta)), self.N)  # xcd order: CTAs past the last row group
        cbase = b.readfirstlane(self.a["cnt"])
        with b.scf_if(b.land(D.is_thread0(b), live)):
            # padded asm atomic: an SGPR base restored by a VALU right before an asm VMEM needs 5 wait states
            old = b.inline_asm(
                "s_nop 4\nglobal_atomic_add $0, $1, $2, $3 sc0\ns_waitcnt vmcnt(0)",
                "=&v,v,v,s,~{memory}",
                [b.mul(self.rg, _c(b, 4)), _c(b, 1), cbase],
                I32,
                result_name_hint="aold",
            )
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        self.stamp(4)
        seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN(self.flag, _c(b, 0), dtype=F32, n=1), 0), I32))
        with b.scf_if(b.land(live, b.cmp_eq(seen, _c(b, S - 1)))):
            if k.fold == "sc1":
                D.acquire_l2(b, "agent")
            self.fold(p_rs)
            with b.scf_if(D.is_thread0(b)):
                b.store(_c(b, 0), b.int_to_ptr(cbase, PtrType(I32, "global")), self.rg)

    def fold(self, p_rs: Value) -> None:
        b, k = self.b, self.k
        TP, R, S = k.bucket, k.rows_per_cta, k.splits
        q4 = R // 4
        pieces = TP * q4
        S_ = S - (1 if k.mutant == "fold" else 0)
        rg0 = b.mul(self.rg, _c(b, R))
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tok = b.div(v, _c(b, q4))
            row = b.add(rg0, b.mul(b.sub(v, b.mul(tok, _c(b, q4))), _c(b, 4)))
            ok = b.land(b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows)), b.cmp_lt(row, self.N))
            tot = None
            loads = []
            for s in range(S_):
                off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.N), row), _c(b, 4))
                loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4,
                                              aux=0 if k.fold == "l2" else CPOL_SC1))
            for ld in loads:
                f = b.vec_bitcast(ld, VectorType(F32, 4))
                if tot is None:
                    tot = [b.vec_extract(f, e) for e in range(4)]
                else:
                    tot = [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            yoff = b.mul(b.add(b.mul(tok, self.N), row), _c(b, 2))
            b.buffer_store_cp(yv, self.y_rs, b.select(ok, yoff, _c(b, OOB)), _c(b, 0))


def build_wide_kernel(k: WideKnobs) -> KernelDef:
    return _Wide(k).run()


QUANT_THREADS = 256


def quant_name(k: WideKnobs, mutant: str = "none") -> str:
    return f"rk41q_k{k.k}_t{k.tiles}_s{k.splits}" + ("" if mutant == "none" else f"_{mutant}")


def quant_grid(k: WideKnobs) -> int:
    return -(-(k.bucket * (k.k // 32)) // QUANT_THREADS)


def build_quant_kernel(k: WideKnobs, mutant: str = "none") -> KernelDef:
    """bf16 X ``[rows, K]`` -> the ``xsrc="glb"`` GEMM's operands: E4M3 codes ``[K/128][tiles][64 lanes][32 B]`` in MFMA
    B-operand order (lane ``l`` of tile ``t``, chunk ``q``: token ``16 t + l mod 16``, K ``128 q + 16 (l / 16) ..`` then
    ``+ 64``) and E8M0 scales ``[S][G][tiles][64]`` dwords (byte ``j`` = local chunk ``4 p + j`` of split ``s``, the lane's
    32-K block ``l / 16``). ATOM's ``quantize_fp8`` rule (bitwise). One thread per (token, 32-K block); tokens >= rows are
    not written. ``mutant="xq"``: block 1 of every chunk stores its scale one binade high. Params: i64 ``x, xq, xs``,
    i32 ``rows``."""
    b = IRBuilder(quant_name(k, mutant))
    b.kernel.attrs["max_workgroup_size"] = QUANT_THREADS
    x = b.param("x", I64)
    xq = b.param("xq", I64)
    xs = b.param("xs", I64)
    rows = b.readfirstlane(b.param("rows", I32))
    K, T, S, kc, G = k.k, k.tiles, k.splits, k.kc, k.groups
    KG = K // 32
    v = b.add(b.mul(b.readfirstlane(b.block_id_x()), _c(b, QUANT_THREADS)), b.thread_id_x())
    tok = b.div(v, _c(b, KG))
    blk = b.sub(v, b.mul(tok, _c(b, KG)))
    ok = b.land(b.cmp_lt(tok, rows), b.cmp_lt(v, _c(b, k.bucket * KG)))
    rl = b.smax(rows, _c(b, 1))
    x_rs = _rsrc(b, x, b.mul(rl, _c(b, 2 * K)))
    base = b.mul(b.add(b.mul(tok, _c(b, K)), b.mul(blk, _c(b, 32))), _c(b, 2))
    regs = [b.buffer_load_cp(x_rs, b.select(ok, b.add(base, _c(b, 16 * i)), _c(b, OOB)), _c(b, 0), dwords=4) for i in range(4)]
    f = []
    for r in regs:
        for e in range(4):
            w = b.vec_extract(r, e)
            f.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
            f.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
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
    q = b.lshr(blk, _c(b, 2))
    g = b.land(blk, _c(b, 3))
    if mutant == "xq":
        code = b.select(b.cmp_eq(g, _c(b, 1)), b.add(code, _c(b, 1)), code)
    t = b.lshr(tok, _c(b, 4))
    l16 = b.land(tok, _c(b, 15))
    h = b.lshr(g, _c(b, 1))
    q_rs = _rsrc(b, xq, _c(b, k.nq * T * 64 * 32))
    with b.scf_if(ok):
        for piece in range(2):
            lg = b.land(b.add(b.mul(g, _c(b, 2)), _c(b, piece)), _c(b, 3))
            off = b.add(b.mul(b.add(b.mul(b.add(b.mul(q, _c(b, T)), t), _c(b, 64)), b.add(b.mul(lg, _c(b, 16)), l16)), _c(b, 32)), b.mul(h, _c(b, 16)))
            b.buffer_store_cp(b.vec_pack(cw[4 * piece : 4 * piece + 4], I32), q_rs, off, _c(b, 0))
        s_ = b.div(q, _c(b, kc))
        cc = b.sub(q, b.mul(s_, _c(b, kc)))
        p_ = b.lshr(cc, _c(b, 2))
        j = b.land(cc, _c(b, 3))
        soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(b.add(b.mul(s_, _c(b, G)), p_), _c(b, T)), t), _c(b, 64)), b.add(b.mul(g, _c(b, 16)), l16)), _c(b, 4)), j)
        b.store(b.trunc(code, I8), b.int_to_ptr(b.readfirstlane(xs), PtrType(I8, "global")), soff)
    return b.kernel


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
    n, kk = W.shape
    sw = np.exp2(np.asarray(ws).astype(np.float64) - 127.0)
    W = W * np.repeat(np.repeat(sw, 32, 0)[:n], 32, 1)
    X = e4m3_to_float(x_q_codes) * np.repeat(np.exp2(np.asarray(x_scale_codes).astype(np.float64) - 127.0), 32, 1)
    return X @ W.T


# ------------------------------------------------------------------ the quantization seam (ATOM layout)

QA_MUTANTS = ("none", "xq", "skip")


def quant_atom_name(gpt: int = 1, threads: int = 256, mutant: str = "none") -> str:
    return f"rk41qa_g{gpt}_b{threads}" + ("" if mutant == "none" else f"_{mutant}")


def build_quant_atom_kernel(gpt: int = 1, threads: int = 256, mutant: str = "none") -> KernelDef:
    """ATOM ``quantize_fp8`` (``atom/model_ops/blockscale_kernels/quantization.py``) as one rocKE launch, bit for bit:
    bf16 ``x`` (any shape, last dim a multiple of 32, contiguous) -> E4M3 ``y`` (same shape) + E8M0 ``s``
    (``[..., K/32]``), the 32-element groups of the flat tensor (a group never straddles a row). Per group: amax
    floored at 1e-4, code = ceil-to-power-of-two exponent of ``amax * fl(1/448)``, codes = RNE E4M3 of ``x / 2^(code -
    127)`` (``v_cvt_scalef32_pk_fp8_f32``). ``gpt`` groups per thread (adjacent threads take adjacent groups).
    Mutants: ``xq`` (every 4th group's scale one binade high), ``skip`` (the last group of each thread not written).
    Params: i64 ``x, y, s``, i32 ``groups``. Grid ``ceil(groups / (threads gpt))``."""
    if gpt not in (1, 2, 4) or threads not in (64, 128, 256, 512) or mutant not in QA_MUTANTS:
        raise ValueError("gpt 1 | 2 | 4, threads 64..512, mutant in QA_MUTANTS")
    b = IRBuilder(quant_atom_name(gpt, threads, mutant))
    b.kernel.attrs["max_workgroup_size"] = threads
    x = b.param("x", I64)
    y = b.param("y", I64)
    sp = b.param("s", I64)
    ng = b.readfirstlane(b.param("groups", I32))
    base_v = b.add(b.mul(b.readfirstlane(b.block_id_x()), _c(b, threads * gpt)), b.thread_id_x())
    x_rs = _rsrc(b, x, b.mul(ng, _c(b, 64)))
    y_rs = _rsrc(b, y, b.mul(ng, _c(b, 32)))
    s_ptr = b.int_to_ptr(b.readfirstlane(sp), PtrType(I8, "global"))
    items = []
    for i in range(gpt):
        v = b.add(base_v, _c(b, i * threads))
        ok = b.cmp_lt(v, ng)
        off = b.select(ok, b.mul(v, _c(b, 64)), _c(b, OOB))
        items.append((v, ok, [b.buffer_load_cp(x_rs, b.add(off, _c(b, 16 * j)), _c(b, 0), dwords=4) for j in range(4)]))
    for i, (v, ok, regs) in enumerate(items):
        f = []
        for r in regs:
            for e in range(4):
                w = b.vec_extract(r, e)
                f.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
                f.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
        amax = b.fabs(f[0])
        for j in range(1, 32):
            amax = b.fmax(amax, b.fabs(f[j]))
        amax = b.fmax(amax, _fbits(b, AMAX_FLOOR_BITS))
        bits = b.bitcast(b.fmul(amax, _fbits(b, RATIO_BITS)), I32)
        code = b.add(b.lshr(bits, _c(b, 23)), b.select(b.cmp_ne(b.land(bits, _c(b, 0x7FFFFF)), _c(b, 0)), _c(b, 1), _c(b, 0)))
        scale = b.bitcast(b.shl(code, _c(b, 23)), F32)
        cw = []
        for d in range(8):
            wd = b.cvt_scalef32_pk_fp8_f32(_c(b, 0), f[4 * d], f[4 * d + 1], scale, hi=False)
            wd = b.cvt_scalef32_pk_fp8_f32(wd, f[4 * d + 2], f[4 * d + 3], scale, hi=True)
            cw.append(wd)
        if mutant == "xq":
            code = b.select(b.cmp_eq(b.land(v, _c(b, 3)), _c(b, 1)), b.add(code, _c(b, 1)), code)
        wr = ok
        if mutant == "skip" and i == gpt - 1:
            wr = b.land(ok, b.cmp_ne(b.land(v, _c(b, 7)), _c(b, 5)))
        yo = b.select(wr, b.mul(v, _c(b, 32)), _c(b, OOB))
        for h in range(2):
            b.buffer_store_cp(b.vec_pack(cw[4 * h : 4 * h + 4], I32), y_rs, b.add(yo, _c(b, 16 * h)), _c(b, 0))
        with b.scf_if(wr):
            b.store(b.trunc(code, I8), s_ptr, v)
    return b.kernel
