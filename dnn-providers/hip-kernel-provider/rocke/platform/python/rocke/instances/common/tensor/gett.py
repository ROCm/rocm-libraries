# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor contraction (GETT) family: native strided operand loads and a
permuted alpha / beta epilogue on the gfx950 direct-to-LDS GEMM body.

One launch computes a hipTensor-style contraction
``D[modes_d] = alpha * sum_k A[modes_a] B[modes_b] (+ beta * C[modes_d])``
with arbitrary mode orders straight from the user's tensors, with no permute
or pack pass. Python engine only: the GEMM body is ``gemm_universal`` with the
``spec._gett`` side channel (:class:`GettHooks`; absent -> every universal
GEMM kernel is byte-identical), and the side channel has no C++ mirror.

* **Operand loads (DTL, per-lane mode offsets).** Both operands are

* **Operand loads (DTL, per-lane mode offsets).** Both operands are
  MN-contiguous (hipTensor's column-major default puts a free mode fastest).
  Each DTL lane loads ``vec`` consecutive GEMM rows (16 bytes, or 4 bytes when
  the fastest run is not a multiple of 8) of one K index into a K-outer LDS
  stage ``[tile_k][tile_mn]``. The per-lane byte offset is
  ``rowoff(row) + klocal(k)`` (mixed-radix over the modes, compile-time
  digits, computed once per output tile), the K tile's offset ``ktile(kt)``
  is the buffer ``soffset`` (scalar mixed-radix per K tile). Rows past the
  operand get an out-of-range offset (the DMA writes zeros).
* **LDS layout.** K row ``k = 32kk + 16c + 8a + 4r + g`` of a stage lives in
  1 KiB block ``g + 4a + 8c + 16kk`` (row ``r`` of it), blocks ``1024 + pad``
  bytes apart: the 8 K rows a half-wave of ``ds_read_b64_tr_b16`` touches
  sit in 8 blocks = 8 distinct bank groups (pad 32), and every read address
  is one lane VGPR per (operand, stage) plus an immediate.
* **Fragments.** ``ds_read_b64_tr_b16`` (two per 16x16x32 fragment) returns
  the MFMA operand layout from the K-outer tile; the lane part of every read
  address is one loop-invariant VGPR per (operand, atom) and the K atom /
  stage / read index are immediates.
* **Epilogue.** ``D[drow(m) + dcol(n)] = alpha * acc (+ beta * C[same])`` in
  f32 with a single rounding, ``vec_d`` consecutive rows per store (GEMM A
  owns D's fastest mode), rows / columns past the tensor masked.

Family API: :class:`TensorGettSpec` (problem + knobs), :func:`is_valid_spec`,
:func:`build_tensor_gett`, :func:`tensor_gett_grid`,
:func:`tensor_gett_signature`. ``ds_read_b64_tr_b16`` exists on gfx950 only,
so the family serves gfx950 (:data:`TENSOR_GETT_ARCHS`). Operands whose
fastest free run is not a multiple of 2 elements (no 4-byte DTL lane) and K
mode sets that do not split evenly into ``tile_k`` are rejected.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ....core.ir import BF16, F16, F32, IRBuilder, KernelDef, PtrType, Value
from ....helpers.spec import SignatureBuilder
from ..gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from ..gemm_universal import is_valid_spec as _gemm_is_valid_spec
from .gett_plan import GettPlan, plan_gett

OOB = 0x7FFFFFF0  # > DTL_BUFFER_NUM_RECORDS (0x7FFF0000): the DMA writes zeros


@dataclass(frozen=True)
class GettKnobs:
    """Kernel knobs (the default is the first value of each). ``epi_pair``,
    ``acc_pin``, ``frag_pin`` and ``llvm_opts`` are retained default-off
    alternatives; ``pad=0`` and ``lane_split=False`` are the unoptimised
    references of their levers."""

    pad: int = (
        32  # bytes after every 1 KiB LDS block (32: conflict-free tr reads; 0: none)
    )
    bilinear: bool = False  # beta * C
    epi_order: str = "mi"  # epilogue store order: "mi" (row-atom major) | "ni"
    epi_pair: bool = (
        False  # 16-byte D stores: lane pairs (q, q^1) swap half an atom pair (ds_swizzle) when vec_d8
    )
    lane_split: bool = (
        True  # one per-lane DTL offset per operand + per-pass soffset constants (when additive)
    )
    acc_pin: int = (
        0  # 1: tie the accumulators to AGPRs after every main-loop K tile (2: as a side-effect asm)
    )
    frag_pin: bool = (
        False  # empty tied-VGPR asm on every fragment (keeps the accumulators in place)
    )
    llvm_opts: str = (
        ""  # '+'-separated extra -mllvm codegen options (kernel attr ``llvm_options``)
    )

    def tag(self) -> str:
        t = ""
        if self.pad != 32:
            t += f"_p{self.pad}"
        if self.bilinear:
            t += "_bil"
        if self.epi_order != "mi":
            t += f"_e{self.epi_order}"
        return t


def _ty(dtype: str):
    return {"f16": F16, "fp16": F16, "bf16": BF16}[dtype]


class GettHooks:
    """The ``spec._gett`` side channel consumed by gemm_universal."""

    def __init__(self, plan: GettPlan, knobs: GettKnobs, spec, dtype: str = "f16"):
        self.plan, self.k, self.dt = plan, knobs, _ty(dtype)
        t = spec.tile
        self.bs = spec.block_size
        self.ws = spec.wave_size
        self.T = t.tile_k
        self.bmn = {"A": t.tile_m, "B": t.tile_n}
        self.wt = {"A": t.warp_tile_m, "B": t.warp_tile_n}
        self.mf = {
            "A": t.tile_m // (t.warp_m * t.warp_tile_m),
            "B": t.tile_n // (t.warp_n * t.warp_tile_n),
        }
        self.wmn = {"A": t.tile_m // t.warp_m, "B": t.tile_n // t.warp_n}
        self.stages = int(spec.trait.dtl_stages)
        self._split_cache: Dict[str, bool] = {}
        bad = body_trait_errors(spec.trait)
        if bad:
            raise ValueError("GETT body: " + "; ".join(bad))
        if (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k) != (16, 16, 32):
            raise ValueError("GETT fragments are built for the 16x16x32 atom")
        if self.T != plan.tile_k:
            raise ValueError(f"spec tile_k {self.T} != plan tile_k {plan.tile_k}")
        for nm in ("A", "B"):
            tot = self.bmn[nm] * self.T * 2
            if tot % (self.bs * self.lane_bytes(nm)):
                raise ValueError(
                    f"{nm}: tile bytes {tot} not a multiple of one DTL pass"
                )
            if self.bmn[nm] != 256 or self.bs != 256:
                raise ValueError(
                    "GETT LDS block layout is built for tile_mn 256 and 4 waves"
                )

    # ------------------------------------------------------------------ geometry
    # LDS stage layout (per operand, tile_mn = 256): K row k of the tile is
    # stored in 1 KiB block b(k) = g + 4a + 8c + 16kk at row r (k = 32kk + 16c
    # + 8a + 4r + g); blocks are ``1024 + pad`` bytes apart. The 8 K rows one
    # half-wave of ds_read_b64_tr_b16 touches are 8 different blocks, i.e. 8
    # different 32-byte bank groups with pad = 32, and every read address is
    # one lane VGPR + an immediate (atom, read, K atom).
    def lane_bytes(self, nm: str) -> int:
        return 16 if self.plan.ops[nm].vec == 8 else 4

    def passes(self, nm: str) -> int:
        return (self.bmn[nm] * self.T * 2) // (self.bs * self.lane_bytes(nm))

    def block_bytes(self) -> int:
        return 1024 + self.k.pad

    def lds_shape(self, nm: str):
        return [self.T // 2, self.block_bytes() // 2]

    def wave_bytes(self, nm: str) -> int:
        ib = 64 * self.lane_bytes(nm)
        return self.block_bytes() if ib == 1024 else ib

    def pass_bytes(self, nm: str) -> int:
        ib = 64 * self.lane_bytes(nm)
        return 4 * self.block_bytes() if ib == 1024 else self.block_bytes()

    @staticmethod
    def _k_of(blk: int, r: int) -> int:
        return (
            32 * (blk // 16)
            + 16 * ((blk // 8) % 2)
            + 8 * ((blk // 4) % 2)
            + 4 * r
            + (blk % 4)
        )

    def _lane_geom(self, nm: str, p: int, tid: int) -> Tuple[int, int]:
        """(k row, logical row in tile) of DTL lane ``tid`` of pass ``p`` (python ints)."""
        lin = tid * self.lane_bytes(nm) + p * 4 * 64 * self.lane_bytes(nm)
        blk, inb = lin // 1024, lin % 1024
        return self._k_of(blk, inb // 512), (inb % 512) // 2

    # ------------------------------------------------------------------ params
    def add_params(self, b: IRBuilder) -> None:
        self.p: Dict[str, Value] = {}
        self.p["alpha"] = b.param("alpha", F32)
        if self.k.bilinear:
            self.p["Cin"] = b.param(
                "Cin", PtrType(self.dt, "global"), readonly=True, align=8
            )
            self.p["beta"] = b.param("beta", F32)

    # ------------------------------------------------------------------ int helpers
    def _digits(
        self, b: IRBuilder, x: Value, digits: Sequence[Tuple[int, int]]
    ) -> Value:
        """sum_j ((x // P_j) % e_j) * s_j (the last digit needs no modulo)."""
        off = None
        q = x
        for j, (e, s) in enumerate(digits):
            last = j == len(digits) - 1
            d = q if last else b.mod(q, b.const_i32(e))
            term = d if s == 1 else b.mul(d, b.const_i32(s))
            off = term if off is None else b.add(off, term)
            if not last:
                q = b.div(q, b.const_i32(e))
        return off

    # ------------------------------------------------------------------ bind (once per kernel)
    def bind(
        self,
        b: IRBuilder,
        *,
        tid: Value,
        lane: Value,
        warp_m_idx: Value,
        warp_n_idx: Value,
    ) -> None:
        self.tid, self.lane = tid, lane
        self.widx = {"A": warp_m_idx, "B": warp_n_idx}
        c16, c4 = b.const_i32(16), b.const_i32(4)
        kq = b.div(lane, c16)
        g = b.div(b.mod(lane, c16), c4)
        blk = b.add(
            g,
            b.add(
                b.mul(b.mod(kq, b.const_i32(2)), c4),
                b.mul(b.div(kq, b.const_i32(2)), b.const_i32(8)),
            ),
        )
        HB = self.block_bytes() // 2
        lane_part = b.add(b.mul(blk, b.const_i32(HB)), b.mul(b.mod(lane, c4), c4))
        # LDS placement (smem pool: A stages then B stages, back to back)
        self._sh = {}
        o = 0
        for nm in ("A", "B"):
            for j in range(self.stages):
                self._sh[(nm, j)] = o // 2  # elements
                o += (self.T // 2) * self.block_bytes()
        self._fbase: Dict[Tuple[str, int], Value] = {}
        for nm in ("A", "B"):
            woff = b.mul(self.widx[nm], b.const_i32(self.wmn[nm]))
            lp = b.add(lane_part, woff)
            for j in range(self.stages):
                sh = self._sh[(nm, j)]
                self._fbase[(nm, j)] = b.optimization_barrier(
                    b.add(lp, b.const_i32(sh)) if sh else lp
                )

    def tile_end(self, b: IRBuilder, acc):
        """After every K tile of the main loop: ``acc_pin`` ties each
        accumulator to itself in an AGPR (empty asm, no instruction), so the
        register allocator keeps every chain in one AGPR tuple across the
        unrolled stages."""
        if not self.k.acc_pin:
            return acc
        return [
            b.inline_asm("", "=a,0", [v], v.type, sideeffect=self.k.acc_pin == 2)
            for v in acc
        ]

    def frag(
        self, b: IRBuilder, smem: Value, nm: str, stage: int, idx: int, kk: int
    ) -> Value:
        sh = self._sh[(nm, stage)]
        base = self._fbase[(nm, stage)]
        HB = self.block_bytes() // 2
        parts = [
            b.ds_read_tr16_b64(
                smem,
                b.const_i32(0),
                b.add(base, b.const_i32(16 * kk * HB + r * 256 + 16 * idx - sh)),
                dtype=self.dt,
            )
            for r in range(2)
        ]
        fr = b.vec_concat(parts[0], parts[1])
        if self.k.frag_pin:
            # hide the transpose-read origin from the MFMA: an empty tied-VGPR
            # asm (no instruction) so the MFMA selects like a ds_read_b128 operand
            fr = b.inline_asm("", "=v,0", [fr], fr.type, sideeffect=False)
        return fr

    # ------------------------------------------------------------------ DTL offsets (per output tile)
    def _lane_expr(self, b: IRBuilder, nm: str, p: int) -> Tuple[Value, Value]:
        """(k row, logical row) of this lane for pass p as IR."""
        lb = self.lane_bytes(nm)
        lin = b.add(b.mul(self.tid, b.const_i32(lb)), b.const_i32(p * 4 * 64 * lb))
        blk = b.div(lin, b.const_i32(1024))
        inb = b.mod(lin, b.const_i32(1024))
        r = b.div(inb, b.const_i32(512))
        col = b.div(b.mod(inb, b.const_i32(512)), b.const_i32(2))
        c2 = b.const_i32(2)
        k = b.add(
            b.add(
                b.mul(b.div(blk, c16 := b.const_i32(16)), b.const_i32(32)),
                b.mul(b.mod(b.div(blk, b.const_i32(8)), c2), c16),
            ),
            b.add(
                b.add(
                    b.mul(b.mod(b.div(blk, b.const_i32(4)), c2), b.const_i32(8)),
                    b.mul(r, b.const_i32(4)),
                ),
                b.mod(blk, b.const_i32(4)),
            ),
        )
        return k, col

    def lane_split(self, nm: str) -> bool:
        """True when every pass's K row is ``pass_k(p) + lane_k(tid)`` with a
        pass-independent lane part and the in-tile K offset is additive over
        that split (no mixed-radix carry): then ONE per-lane VGPR offset per
        operand serves every pass and the pass part rides in soffset."""
        if not self.k.lane_split:
            return False
        if nm in self._split_cache:
            return self._split_cache[nm]
        ok = True
        np_ = self.passes(nm)
        lane_k = [self._lane_geom(nm, 0, t)[0] for t in range(self.bs)]
        col0 = [self._lane_geom(nm, 0, t)[1] for t in range(self.bs)]
        for p in range(np_):
            ks = [self._lane_geom(nm, p, t)[0] for t in range(self.bs)]
            cs = [self._lane_geom(nm, p, t)[1] for t in range(self.bs)]
            pk = ks[0] - lane_k[0]
            if cs != col0 or any(k - lk != pk for k, lk in zip(ks, lane_k)):
                ok = False
                break
            for lk in set(lane_k):
                if self.plan.klocal(pk + lk, nm) != self.plan.klocal(
                    pk, nm
                ) + self.plan.klocal(lk, nm):
                    ok = False
                    break
            if not ok:
                break
        self._split_cache[nm] = ok
        return ok

    def pass_soff_bytes(self, nm: str, p: int) -> int:
        """Constant soffset add of pass p (lane_split mode), else 0."""
        if not self.lane_split(nm):
            return 0
        pk = self._lane_geom(nm, p, 0)[0] - self._lane_geom(nm, 0, 0)[0]
        return 2 * self.plan.klocal(pk, nm)

    def tile_voffs(self, b: IRBuilder, nm: str, mn0: Value) -> List[Value]:
        op = self.plan.ops[nm]
        np_ = self.passes(nm)
        rag = op.rows % self.bmn[nm] != 0
        if self.lane_split(nm):
            k_v, r_v = self._lane_expr(b, nm, 0)
            row = b.add(mn0, r_v)
            ro = self._digits(b, row, op.row_digits)
            if rag:
                ro = b.select(
                    b.cmp_lt(row, b.const_i32(op.rows)), ro, b.const_i32(-(1 << 30))
                )
            v = b.mul(b.add(ro, self._klocal_expr(b, nm, k_v)), b.const_i32(2))
            if rag:
                v = b.select(b.cmp_lt(ro, b.const_i32(0)), b.const_i32(OOB), v)
            return [v] * np_
        groups: Dict[Tuple[int, ...], int] = {}
        rep = []
        for p in range(np_):
            key = tuple(self._lane_geom(nm, p, t)[1] for t in range(self.bs))
            if key not in groups:
                groups[key] = p
            rep.append(groups[key])
        rowoff: Dict[int, Value] = {}
        out = []
        for p in range(np_):
            k_v, r_v = self._lane_expr(b, nm, p)
            gp = rep[p]
            if gp not in rowoff:
                if gp != p:
                    _k, r_v = self._lane_expr(b, nm, gp)
                row = b.add(mn0, r_v)
                ro = self._digits(b, row, op.row_digits)
                if rag:
                    ro = b.select(
                        b.cmp_lt(row, b.const_i32(op.rows)), ro, b.const_i32(-(1 << 30))
                    )
                rowoff[gp] = ro
            ro = rowoff[gp]
            kl = self._klocal_expr(b, nm, k_v)
            v = b.mul(b.add(ro, kl), b.const_i32(2))
            if rag:
                v = b.select(b.cmp_lt(ro, b.const_i32(0)), b.const_i32(OOB), v)
            out.append(v)
        return out

    def _klocal_expr(self, b: IRBuilder, nm: str, k: Value) -> Value:
        st = self.plan.ops[nm].strides
        dig = [(r, mult * st[m]) for (r, m, mult) in self.plan.k_local]
        return self._digits(b, k, dig)

    def ktile_off_bytes(self, b: IRBuilder, nm: str, k_ld: Value) -> Value:
        st = self.plan.ops[nm].strides
        kt = b.div(k_ld, b.const_i32(self.T))
        dig = [(r, mult * st[m] * 2) for (r, m, mult) in self.plan.k_tile]
        if not dig:
            return b.const_i32(0)
        return b.to_sgpr_u32(self._digits(b, kt, dig))

    # ------------------------------------------------------------------ epilogue
    def epilogue(
        self, b: IRBuilder, vals, bm, bn, *, C, warp_m_idx, warp_n_idx, lane, acc_out
    ) -> None:
        """Permuted alpha (/ beta) epilogue. Lane (q, c) of atom (mi, ni) holds
        GEMM rows ``m0 + 16 mi + 4q + v`` of column ``n0 + 16 ni + c``; GEMM A
        owns D's fastest mode, so ``vec_d`` consecutive rows are one vector.
        Buffer stores (C / D resource sized to the tensor): rows or columns past
        the tensor get an out-of-range offset and are dropped, no branches."""
        pl = self.plan
        vd = pl.vec_d
        c16 = b.const_i32(16)
        q = b.div(lane, c16)
        cn = b.mod(lane, c16)
        mfA, mfB = self.mf["A"], self.mf["B"]
        m0 = b.add(
            b.add(bm, b.mul(warp_m_idx, b.const_i32(self.wmn["A"]))),
            b.mul(q, b.const_i32(4)),
        )
        n0 = b.add(b.add(bn, b.mul(warp_n_idx, b.const_i32(self.wmn["B"]))), cn)
        rag_m = pl.M % self.bmn["A"] != 0
        rag_n = pl.N % self.bmn["B"] != 0
        d_bytes = 2 * (1 + sum((pl.ext[m] - 1) * pl.strides_d[m] for m in pl.modes_d))
        rs_d = b.buffer_rsrc(C, b.const_i32(d_bytes))
        rs_c = (
            b.buffer_rsrc(self.p["Cin"], b.const_i32(d_bytes))
            if self.k.bilinear
            else None
        )
        c0 = b.const_i32(0)
        big = b.const_i32(OOB)
        cols = []
        for ni in range(mfB):
            n = b.add(n0, b.const_i32(ni * 16))
            off = b.mul(self._digits(b, n, pl.d_col_digits), b.const_i32(2))
            if rag_n:
                off = b.select(b.cmp_lt(n, b.const_i32(pl.N)), off, big)
            cols.append(off)
        alpha = self.p["alpha"]
        bf = self.dt is BF16
        if self.k.epi_pair and pl.vec_d8 and mfB % 2 == 0:
            self._epilogue_pair(
                b,
                vals,
                bm,
                bn,
                warp_m_idx=warp_m_idx,
                warp_n_idx=warp_n_idx,
                lane=lane,
                acc_out=acc_out,
                rs_d=rs_d,
                rs_c=rs_c,
                big=big,
                bf=bf,
            )
            return
        for mi in range(mfA):
            rows = []
            for v0 in range(0, 4, vd):
                m = b.add(m0, b.const_i32(mi * 16 + v0))
                off = b.mul(self._digits(b, m, pl.d_row_digits), b.const_i32(2))
                if rag_m:
                    off = b.select(b.cmp_lt(m, b.const_i32(pl.M)), off, big)
                rows.append(off)
            # bilinear: every C load of this atom row first (latency overlaps), then math + stores
            cl = {}
            if self.k.bilinear:
                for ni in range(mfB):
                    for gi, roff in enumerate(rows):
                        off = b.add(roff, cols[ni])
                        if vd > 1:
                            cv = b.buffer_load_vN(rs_c, off, c0, self.dt, vd)
                            cl[(ni, gi)] = [b.vec_extract(cv, e) for e in range(vd)]
                        else:
                            cl[(ni, gi)] = [b.buffer_load(rs_c, off, c0, self.dt)]
            for ni in range(mfB):
                acc = acc_out(vals[mi * mfB + ni])
                for gi, roff in enumerate(rows):
                    # an OOB row / column (>= 0x7FFFFFF0) plus an in-range
                    # offset (< 2^31) stays out of range (no 32-bit wrap)
                    off = b.add(roff, cols[ni])
                    xs = [
                        b.fmul(alpha, b.vec_extract(acc, gi * vd + e))
                        for e in range(vd)
                    ]
                    if self.k.bilinear:
                        xs = [
                            b.fma(self.p["beta"], b.cast_to_f32(cc), x)
                            for cc, x in zip(cl[(ni, gi)], xs)
                        ]
                    if vd == 1:
                        h = b.cast_f32_to(xs[0], self.dt)
                        (b.buffer_store_bf16 if bf else b.buffer_store_f16)(
                            rs_d, off, c0, h
                        )
                    else:
                        vec = b.vec_pack(
                            [b.cast_f32_to(x, self.dt) for x in xs], self.dt
                        )
                        (b.buffer_store_vN_bf16 if bf else b.buffer_store_vN_f16)(
                            rs_d, off, c0, vec, vd // 2
                        )

    def _epilogue_pair(
        self,
        b,
        vals,
        bm,
        bn,
        *,
        warp_m_idx,
        warp_n_idx,
        lane,
        acc_out,
        rs_d,
        rs_c,
        big,
        bf,
    ):
        """16-byte stores. Atoms X = (mi, 2j), Y = (mi, 2j+1); lane (q, c) holds rows 4q..4q+3 of
        column c of both. Even q keeps X and receives X rows 4q+4..4q+7 from lane q+1 (xor 16);
        odd q keeps Y and receives Y rows 4q-4..4q-1 from lane q-1: every lane then owns 8
        consecutive GEMM rows (8 contiguous D elements) of one column."""
        pl = self.plan
        mfA, mfB = self.mf["A"], self.mf["B"]
        c16 = b.const_i32(16)
        q = b.div(lane, c16)
        cn = b.mod(lane, c16)
        odd = b.cmp_eq(b.mod(q, b.const_i32(2)), b.const_i32(1))
        oddi = b.mod(q, b.const_i32(2))
        c0 = b.const_i32(0)
        m0 = b.add(
            b.add(bm, b.mul(warp_m_idx, b.const_i32(self.wmn["A"]))),
            b.mul(b.div(q, b.const_i32(2)), b.const_i32(8)),
        )
        n0 = b.add(
            b.add(bn, b.mul(warp_n_idx, b.const_i32(self.wmn["B"]))),
            b.add(cn, b.mul(oddi, c16)),
        )
        rag_m = pl.M % self.bmn["A"] != 0
        rag_n = pl.N % self.bmn["B"] != 0
        cols = []
        for j in range(mfB // 2):
            n = b.add(n0, b.const_i32(32 * j))
            off = b.mul(self._digits(b, n, pl.d_col_digits), b.const_i32(2))
            if rag_n:
                off = b.select(b.cmp_lt(n, b.const_i32(pl.N)), off, big)
            cols.append(off)
        alpha = self.p["alpha"]
        for mi in range(mfA):
            m = b.add(m0, b.const_i32(mi * 16))
            roff = b.mul(self._digits(b, m, pl.d_row_digits), b.const_i32(2))
            if rag_m:
                roff = b.select(b.cmp_lt(m, b.const_i32(pl.M)), roff, big)
            cl = {}
            if self.k.bilinear:
                for j in range(mfB // 2):
                    cv = b.buffer_load_vN(rs_c, b.add(roff, cols[j]), c0, self.dt, 8)
                    cl[j] = [b.vec_extract(cv, e) for e in range(8)]
            for j in range(mfB // 2):
                ax = acc_out(vals[mi * mfB + 2 * j])
                ay = acc_out(vals[mi * mfB + 2 * j + 1])
                xs8 = []
                own, rcv = [], []
                for e in range(4):
                    xe, ye = b.vec_extract(ax, e), b.vec_extract(ay, e)
                    send = b.select(odd, xe, ye)
                    rcv.append(b.warp_shuffle_xor(send, 16))
                    own.append(b.select(odd, ye, xe))
                first = [b.select(odd, r, o) for r, o in zip(rcv, own)]
                last = [b.select(odd, o, r) for r, o in zip(rcv, own)]
                xs8 = [b.fmul(alpha, x) for x in first + last]
                if self.k.bilinear:
                    xs8 = [
                        b.fma(self.p["beta"], b.cast_to_f32(cc), x)
                        for cc, x in zip(cl[j], xs8)
                    ]
                vec = b.vec_pack([b.cast_f32_to(x, self.dt) for x in xs8], self.dt)
                (b.buffer_store_vN_bf16 if bf else b.buffer_store_vN_f16)(
                    rs_d, b.add(roff, cols[j]), c0, vec, 4
                )


# ---------------------------------------------------------------------------------------------
# GEMM body
# ---------------------------------------------------------------------------------------------

#: The GETT body: 256x256 tiles of 2x2 waves, 16x16x32 atoms, a direct-to-LDS
#: ring of ``dtl_stages`` K-outer stages, AGPR accumulators, fragment
#: prefetch, merged MFMA interleave and hoisted DTL addressing.
BODY_TRAIT = dict(
    pipeline="compv4",
    scheduler="intrawave",
    epilogue="default",
    chiplet_swizzle=True,
    chiplet_wgm=8,
    chiplet_num_xcds=8,
    chiplet_chunk_size=64,
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    acc_agpr=True,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    dtl_hoist=True,
    dtl_stages=3,
)
#: Body trait fields a GETT spec may set; every other TraitSpec field must keep
#: the :data:`BODY_TRAIT` / TraitSpec default (the GETT hooks replace the
#: operand addressing, the LDS layout, the fragment reads and the epilogue,
#: and were built and checked on this body only).
BODY_FREE_FIELDS = (
    "dtl_stages",
    "persistent_ctas",
    "chiplet_wgm",
    "chiplet_chunk_size",
    "chiplet_num_xcds",
    "dtl_cache_a",
    "dtl_cache_b",
)


def body_trait_errors(trait: TraitSpec) -> List[str]:
    """Why ``trait`` is not a GETT body trait (empty list: it is one)."""
    ref = TraitSpec(**BODY_TRAIT)
    errs = []
    for f in dataclasses.fields(TraitSpec):
        if f.name in BODY_FREE_FIELDS:
            continue
        got, want = getattr(trait, f.name), getattr(ref, f.name)
        if got != want:
            errs.append(f"{f.name}={got!r} (the GETT body needs {want!r})")
    if int(trait.dtl_stages) not in (2, 3, 4):
        errs.append(f"dtl_stages={trait.dtl_stages} (2..4)")
    if int(trait.persistent_ctas) < 0:
        errs.append(f"persistent_ctas={trait.persistent_ctas} (>= 0)")
    return errs


def body_spec(dtype: str = "fp16", tile_k: int = 32, **trait_over) -> UniversalGemmSpec:
    """The GETT body (:data:`BODY_TRAIT`) as a universal GEMM spec, plus trait
    overrides."""
    tile = dict(
        tile_m=256,
        tile_n=256,
        tile_k=tile_k,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    trait = dict(BODY_TRAIT)
    trait.update(trait_over)
    dt = "fp16" if dtype in ("f16", "fp16") else dtype
    return UniversalGemmSpec(
        name="tensor_gett",
        tile=TileSpec(**tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
    )


# ---------------------------------------------------------------------------------------------
# family spec
# ---------------------------------------------------------------------------------------------

TENSOR_GETT_ARCHS = ("gfx950",)
TENSOR_GETT_DTYPES = ("fp16", "bf16")
#: Largest operand / output byte span the 32-bit buffer offsets address (row
#: and column offsets past the tensor are pushed to :data:`OOB`).
MAX_TENSOR_BYTES = 0x7FFF0000

ModeMap = Tuple[Tuple[str, int], ...]


def _mode_map(d: Optional[Dict[str, int]]) -> Optional[ModeMap]:
    if d is None:
        return None
    return tuple(sorted((str(k), int(v)) for k, v in d.items()))


def _as_dict(m: Optional[ModeMap]) -> Optional[Dict[str, int]]:
    return None if m is None else dict(m)


@dataclass(frozen=True)
class TensorGettSpec:
    """One GETT launch: ``D[modes_d] = alpha * sum_k A[modes_a] B[modes_b]
    (+ beta * C[modes_d])`` (``knobs.bilinear``).

    Mode lists are ROW-major: the last mode is the fastest (a hipTensor
    column-major descriptor's mode list reversed). ``strides_*`` are element
    strides per mode (``None``: packed row-major). Every mode of D is a free
    mode of exactly one operand, every K mode is in both operands and not in
    D (no batch / Hadamard modes). ``tile_k``, ``dtl_stages``,
    ``persistent_ctas`` and the chiplet raster are GEMM body levers.
    """

    modes_a: Tuple[str, ...]
    modes_b: Tuple[str, ...]
    modes_d: Tuple[str, ...]
    extents: ModeMap
    strides_a: Optional[ModeMap] = None
    strides_b: Optional[ModeMap] = None
    strides_d: Optional[ModeMap] = None
    dtype: str = "fp16"
    knobs: GettKnobs = GettKnobs()
    tile_k: int = 32
    dtl_stages: int = 3
    persistent_ctas: int = 0
    chiplet_wgm: int = 8
    chiplet_chunk_size: int = 64

    @staticmethod
    def make(
        modes_a: Sequence[str],
        modes_b: Sequence[str],
        modes_d: Sequence[str],
        extents: Dict[str, int],
        *,
        strides_a: Optional[Dict[str, int]] = None,
        strides_b: Optional[Dict[str, int]] = None,
        strides_d: Optional[Dict[str, int]] = None,
        **kw,
    ) -> "TensorGettSpec":
        """Spec from plain mode lists and ``{mode: extent / stride}`` dicts."""
        return TensorGettSpec(
            modes_a=tuple(modes_a),
            modes_b=tuple(modes_b),
            modes_d=tuple(modes_d),
            extents=_mode_map(extents),
            strides_a=_mode_map(strides_a),
            strides_b=_mode_map(strides_b),
            strides_d=_mode_map(strides_d),
            **kw,
        )

    @property
    def bilinear(self) -> bool:
        return bool(self.knobs.bilinear)

    def plan(self) -> GettPlan:
        """The host-side plan (raises ValueError when the problem has no GETT
        plan: an operand that is not MN-contiguous, K modes that do not split
        into ``tile_k``, batch / Hadamard modes)."""
        return plan_gett(
            self.modes_a,
            self.modes_b,
            self.modes_d,
            dict(self.extents),
            strides_a=_as_dict(self.strides_a),
            strides_b=_as_dict(self.strides_b),
            strides_d=_as_dict(self.strides_d),
            tile_k=self.tile_k,
        )

    def body(self) -> UniversalGemmSpec:
        """The universal GEMM body spec this launch builds on."""
        trait = dict(
            dtl_stages=self.dtl_stages,
            persistent_ctas=self.persistent_ctas,
            chiplet_wgm=self.chiplet_wgm,
            chiplet_chunk_size=self.chiplet_chunk_size,
        )
        sp = body_spec(self.dtype, tile_k=self.tile_k, **trait)
        return dataclasses.replace(sp, name=self.kernel_name())

    def kernel_name(self) -> str:
        key = json.dumps(dataclasses.asdict(self), sort_keys=True)
        dt = "f16" if self.dtype in ("f16", "fp16") else self.dtype
        return f"tensor_gett_{dt}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _span_bytes(ext: Dict[str, int], strides: Dict[str, int], modes) -> int:
    return 2 * (1 + sum((ext[m] - 1) * strides[m] for m in modes))


def is_valid_spec(spec: TensorGettSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """``(ok, reason)``: whether ``spec`` builds and is addressable on ``arch``."""
    if arch not in TENSOR_GETT_ARCHS:
        return False, f"arch {arch!r} not in {TENSOR_GETT_ARCHS} (ds_read_b64_tr_b16)"
    if spec.dtype not in TENSOR_GETT_DTYPES + ("f16",):
        return False, f"dtype {spec.dtype!r} not in {TENSOR_GETT_DTYPES}"
    if spec.tile_k not in (32, 64):
        return False, f"tile_k {spec.tile_k} not in (32, 64)"
    k = spec.knobs
    if k.pad not in (0, 32):
        return False, f"knobs.pad {k.pad} not in (0, 32)"
    if k.epi_order not in ("mi", "ni"):
        return False, f"knobs.epi_order {k.epi_order!r} not in ('mi', 'ni')"
    if k.acc_pin not in (0, 1, 2):
        return False, f"knobs.acc_pin {k.acc_pin} not in (0, 1, 2)"
    try:
        plan = spec.plan()
    except ValueError as e:
        return False, f"no GETT plan: {e}"
    for nm in ("A", "B"):
        if plan.ops[nm].nbytes > MAX_TENSOR_BYTES:
            return False, f"operand {plan.ops[nm].name} spans more than 2 GiB"
    if _span_bytes(plan.ext, plan.strides_d, plan.modes_d) > MAX_TENSOR_BYTES:
        return False, "D spans more than 2 GiB"
    body = spec.body()
    try:
        GettHooks(plan, k, body, dtype=spec.dtype)
    except ValueError as e:
        return False, str(e)
    return _gemm_is_valid_spec(body, arch=arch)


def build_tensor_gett(spec: TensorGettSpec, *, arch: str = "gfx950") -> KernelDef:
    """KernelDef of the GETT launch. Params: A, B, C (= D), M, N, K, alpha
    [, Cin, beta] (:func:`tensor_gett_signature`); grid
    :func:`tensor_gett_grid`, block 256. The kernel's A / B params take the
    user operands :func:`launch_operands` names."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid tensor GETT spec: {why}")
    plan = spec.plan()
    body = spec.body()
    hooks = GettHooks(plan, spec.knobs, body, dtype=spec.dtype)
    object.__setattr__(body, "_gett", hooks)
    kd = build_universal_gemm(body, arch=arch)
    if spec.knobs.llvm_opts:
        kd.attrs["llvm_options"] = tuple(spec.knobs.llvm_opts.split("+"))
    return kd


def tensor_gett_grid(spec: TensorGettSpec) -> Tuple[int, int, int]:
    if spec.persistent_ctas:
        return (int(spec.persistent_ctas), 1, 1)
    plan = spec.plan()
    return (-(-plan.N // 256), -(-plan.M // 256), 1)


def tensor_gett_signature(spec: TensorGettSpec):
    dt = "f16" if spec.dtype in ("f16", "fp16") else spec.dtype
    sb = SignatureBuilder().ptr("A", dt).ptr("B", dt).ptr("C", dt)
    sb = sb.scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
    sb = sb.scalar("alpha", "f32")
    if spec.bilinear:
        sb = sb.ptr("Cin", dt).scalar("beta", "f32")
    return sb.build()


def launch_operands(spec: TensorGettSpec) -> Tuple[str, str]:
    """The user operands (``"A"`` / ``"B"``) bound to the kernel's A and B
    params: the GEMM A is the operand that owns D's fastest mode."""
    plan = spec.plan()
    return plan.ops["A"].name, plan.ops["B"].name


__all__ = [
    "BODY_FREE_FIELDS",
    "BODY_TRAIT",
    "GettHooks",
    "GettKnobs",
    "MAX_TENSOR_BYTES",
    "TENSOR_GETT_ARCHS",
    "TENSOR_GETT_DTYPES",
    "TensorGettSpec",
    "body_spec",
    "body_trait_errors",
    "build_tensor_gett",
    "is_valid_spec",
    "launch_operands",
    "tensor_gett_grid",
    "tensor_gett_signature",
]
