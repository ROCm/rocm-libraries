# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor contraction (GETT) family: native strided operand loads and a
permuted alpha / beta epilogue on the gfx950 direct-to-LDS GEMM bodies.

One launch computes a hipTensor-style contraction
``D[modes_d] = alpha * sum_k A[modes_a] B[modes_b] (+ beta * C[modes_d])``
with arbitrary mode orders straight from the user's tensors. Python engine
only: the GEMM body is ``gemm_universal`` with the ``spec._gett`` side channel
(:class:`GettHooks`; absent -> every universal GEMM kernel is byte-identical),
and the side channel has no C++ mirror. Three body kinds:

* the ring body (:data:`BODY_TRAIT`, the default ``TensorGettSpec``): the hook
  owns the operand addressing, the K-outer LDS layout below, the fragment
  reads and the epilogue (4-byte DTL lanes for short operand runs);
* MN-layout bodies (``TensorGettSpec.mn_tile`` / ``mn_trait``): any
  both-MN-contiguous (``CRR``) ring GEMM, e.g. the GEMM dispatcher's pick for
  the same M x N x K. gemm_universal's MN operand path owns the LDS stages
  and the transpose reads; the hook replaces the operand addressing
  (per-lane mode offsets over the body's DTL lane geometry) and the epilogue
  (``epi_v16``: A's transpose reads in ``QUAD_PERM`` order so lanes L and
  L + 32 hold 8 consecutive rows, one ``v_permlane32_swap`` per dword, 16-byte
  stores);
* K-contiguous bodies (``kc_tile`` / ``kc_trait``): any RCR GEMM (e.g. the
  dispatcher's row-interleave / hb_sched picks) reading K-contiguous packed
  copies of the operands (:func:`kc_operand_modes`, made by the
  ``tensor.pack`` kernel); the hook owns only the epilogue, including the
  row-interleave vector epilogue's 8-columns-per-lane layout
  (:meth:`GettHooks.epilogue_rve`).

K modes that do not split into the K tile are planned with one in-tile K mode
padded (its padded rows read zeros). D spans up to :data:`MAX_D_BYTES`; above
2 GiB the epilogue masks the summed 32-bit offset (no row / column sentinel).

The ring body in detail:

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
:func:`build_tensor_gett`, :func:`tensor_gett_grid`, :func:`tensor_gett_block`,
:func:`tensor_gett_signature`. ``ds_read_b64_tr_b16`` exists on gfx950 only,
so the family serves gfx950 (:data:`TENSOR_GETT_ARCHS`). On the ring and MN
bodies, operands whose fastest free run is not a multiple of 2 elements (no
4-byte DTL lane) are rejected; the MN bodies need runs that are multiples of
8 (the dispatcher's plan packs other operands first).
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
#: D / C byte span up to which the row / column sentinel :data:`OOB` works: an
#: OOB row plus an in-range column (or the reverse) must stay >= the buffer
#: range without wrapping 2^32. Larger D (up to :data:`MAX_D_BYTES`) masks the
#: summed offset with one select instead (:data:`WIDE_OOB`).
NARROW_D_BYTES = OOB
#: Sentinel of the wide-D epilogue: >= every D byte span it serves, and a
#: 16-byte store at it does not wrap 2^32.
WIDE_OOB = 0xFFFFFF00
#: Largest D / C byte span the 32-bit buffer offsets of the epilogue address.
MAX_D_BYTES = WIDE_OOB


def _i32_bits(v: int) -> int:
    """The i32 constant with the bit pattern of the u32 ``v``."""
    return v - (1 << 32) if v >= (1 << 31) else v


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
    # MN-layout bodies (``TensorGettSpec.mn_tile``) only:
    epi_v16: bool = (
        False  # 16-byte D (and C) accesses: A rows read in QUAD_PERM order + v_permlane32_swap per atom pair
    )
    c_prefetch: int = (
        0  # bilinear: C loads issued this many atom rows ahead of their use (0: per row)
    )
    store_nt: bool = (
        False  # epi_v16: streaming (non-temporal) global D stores, bounds as a per-lane branch
    )

    def tag(self) -> str:
        t = ""
        if self.pad != 32:
            t += f"_p{self.pad}"
        if self.bilinear:
            t += "_bil"
        if self.epi_order != "mi":
            t += f"_e{self.epi_order}"
        if self.epi_v16:
            t += "_v16"
        if self.c_prefetch:
            t += f"_cp{self.c_prefetch}"
        if self.store_nt:
            t += "_nt"
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
        # MN-layout mode: the body is a native both-MN-contiguous (CRR) ring
        # GEMM; gemm_universal's MN operand path owns the LDS stages and the
        # transpose fragment reads, the hook only replaces the global operand
        # addressing (per-lane mode offsets) and the epilogue
        self.mn = spec.data.layout == "CRR"
        #: K-contiguous packed operands (``TensorGettSpec.kc_tile``): the
        #: body reads plain [M, K] / [N, K] copies with its own addressing;
        #: the hook owns only the epilogue
        self.kc = bool(plan.kc)
        #: the hook replaces the operand addressing (ring and MN bodies)
        self.addr = not self.kc
        #: read by gemm_mn_operand.spec_mn_layouts (A's transpose reads in
        #: QUAD_PERM order for the 16-byte epilogue)
        self.a_sub_perm = bool(self.mn and knobs.epi_v16)
        self.lay: Dict[str, object] = {}
        if self.kc:
            bad = kc_body_errors(spec)
            if bad:
                raise ValueError("GETT K-contiguous body: " + "; ".join(bad))
            if knobs.epi_v16 or knobs.store_nt or knobs.epi_pair:
                raise ValueError(
                    "epi_v16 / store_nt / epi_pair are MN-body knobs (the "
                    "K-contiguous bodies use their own epilogue layout)"
                )
            if spec.trait.rilv_vec_epilogue and not plan.vec_dc8:
                raise ValueError(
                    "the row-interleave vector epilogue stores 8 consecutive "
                    "columns: D needs 8-element runs on the column side"
                )
            return
        if self.mn:
            bad = mn_body_errors(spec)
            if bad:
                raise ValueError("GETT MN body: " + "; ".join(bad))
            if self.T != plan.tile_k:
                raise ValueError(f"spec tile_k {self.T} != plan tile_k {plan.tile_k}")
            for nm in ("A", "B"):
                if plan.ops[nm].vec != 8:
                    raise ValueError(
                        f"{nm}: the MN body loads 16-byte runs (operand run vec "
                        f"{plan.ops[nm].vec}; pack it or use the ring body)"
                    )
            if knobs.store_nt and not knobs.epi_v16:
                raise ValueError("store_nt needs epi_v16")
            if knobs.epi_v16 and not self.v16_ok():
                raise ValueError(
                    "epi_v16 needs 8-element D runs (vec_d8) and an even number "
                    "of M atoms per wave"
                )
            from ..gemm_mn_operand import spec_mn_layouts

            la, lb = spec_mn_layouts(spec)
            if knobs.epi_v16:
                # accumulator rows of lane groups q and q + 2 = 8 consecutive M
                la = dataclasses.replace(la, sub_perm=True)
            self.lay = {"A": la, "B": lb}
            return
        if knobs.epi_v16 or knobs.store_nt:
            raise ValueError(
                "epi_v16 / store_nt are MN-body knobs (TensorGettSpec.mn_tile)"
            )
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
    def v16_ok(self) -> bool:
        """The 16-byte permlane epilogue applies: 8-element D runs and M atoms
        in pairs."""
        return bool(self.plan.vec_d8) and self.mf["A"] % 2 == 0

    def lane_bytes(self, nm: str) -> int:
        if self.mn:
            return 16
        return 16 if self.plan.ops[nm].vec == 8 else 4

    def passes(self, nm: str) -> int:
        if self.mn:
            return self.lay[nm].n_pass
        return (self.bmn[nm] * self.T * 2) // (self.bs * self.lane_bytes(nm))

    def block_bytes(self) -> int:
        return 1024 + self.k.pad

    def lds_shape(self, nm: str):
        return [self.T // 2, self.block_bytes() // 2]

    def wave_bytes(self, nm: str) -> int:
        ib = 64 * self.lane_bytes(nm)
        return self.block_bytes() if ib == 1024 else ib

    def pass_bytes(self, nm: str) -> int:
        if self.mn:
            return self.lay[nm].pass_bytes
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
        if self.mn:
            self._bind_mn(b)
            return
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

    def _bind_mn(self, b: IRBuilder) -> None:
        """MN mode: the per-pass DTL lane geometry (k, mn) of each operand
        and the loop-invariant in-tile K offsets (elements) with the K-pad
        lane mask; the per-tile row offsets come in :meth:`tile_voffs`."""
        from ..gemm_mn_operand import emit_dtl_lane_k_mn

        self._mn_geo: Dict[str, List[Tuple[Value, Value, Value, Optional[Value]]]] = {}
        for nm in ("A", "B"):
            geo = []
            for k, mn in emit_dtl_lane_k_mn(b, self.lay[nm], self.tid):
                kl = self._klocal_expr(b, nm, k)
                geo.append((k, mn, kl, self._k_valid(b, k)))
            self._mn_geo[nm] = geo

    def _k_valid(self, b: IRBuilder, k: Value) -> Optional[Value]:
        """Predicate: in-tile K row ``k`` is a real K index (``None`` when the
        plan pads no K mode). A padded K mode is entirely in-tile, so the
        test only looks at its local digit."""
        kp = self.plan.k_pad
        if kp is None:
            return None
        mode, real, _padded = kp
        div = 1
        for r, m, _mult in self.plan.k_local:
            if m == mode:
                d = b.div(k, b.const_i32(div)) if div > 1 else k
                d = b.mod(d, b.const_i32(r))
                return b.cmp_lt(d, b.const_i32(real))
            div *= r
        raise ValueError(f"padded K mode {mode} is not an in-tile digit")

    def _tile_voffs_mn(self, b: IRBuilder, nm: str, mn0: Value) -> List[Value]:
        op = self.plan.ops[nm]
        rag = op.rows % self.bmn[nm] != 0
        c2 = b.const_i32(2)
        big = b.const_i32(OOB)
        out = []
        for _k, mn, kl, kok in self._mn_geo[nm]:
            row = b.add(mn0, mn)
            ro = self._digits(b, row, op.row_digits)
            v = b.mul(b.add(ro, kl), c2)
            ok = b.cmp_lt(row, b.const_i32(op.rows)) if rag else None
            if kok is not None:
                ok = kok if ok is None else b.land(ok, kok)
            if ok is not None:
                v = b.select(ok, v, big)
            out.append(v)
        return out

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
        if not self.k.lane_split or self.mn or self.plan.k_pad is not None:
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
        if self.mn:
            return self._tile_voffs_mn(b, nm, mn0)
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
            kok = self._k_valid(b, k_v)
            if kok is not None:
                # padded K rows read zeros
                v = b.select(kok, v, b.const_i32(OOB))
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
    def d_bytes(self) -> int:
        """Byte span of D (and of C under ``bilinear``)."""
        pl = self.plan
        return 2 * (1 + sum((pl.ext[m] - 1) * pl.strides_d[m] for m in pl.modes_d))

    def wide_d(self) -> bool:
        """True when D spans more than :data:`NARROW_D_BYTES`: the row / column
        sentinel :data:`OOB` would then land inside D (wrong values on ragged
        tiles), so the epilogue masks the summed offset instead."""
        return self.d_bytes() > NARROW_D_BYTES

    def _d_part(self, b: IRBuilder, idx: Value, digits, lim: int, big: Value):
        """Row (column) part of a D offset for GEMM row (column) ``idx``;
        ``lim`` > 0: indices >= ``lim`` are past the tensor.

        Narrow D: the byte offset, with :data:`OOB` past the tensor. Wide D:
        ``(element offset, in-range predicate or None)``; :meth:`_d_off`
        combines two parts (element offsets stay below 2^31, so the i32
        arithmetic never overflows)."""
        e = self._digits(b, idx, digits)
        if self.wide_d():
            return (e, b.cmp_lt(idx, b.const_i32(lim)) if lim else None)
        off = b.mul(e, b.const_i32(2))
        if lim:
            off = b.select(b.cmp_lt(idx, b.const_i32(lim)), off, big)
        return off

    def _d_off(self, b: IRBuilder, r, c) -> Value:
        """Byte offset of D element (row part ``r``, column part ``c``)."""
        if not self.wide_d():
            return b.add(r, c)
        (re, rok), (ce, cok) = r, c
        e = b.add(re, ce)
        # bytes = 2 e < 2^32: an unsigned add (a signed i32 multiply would
        # overflow past 2^31)
        off = b.add_nuw(e, e)
        ok = rok if cok is None else (cok if rok is None else b.land(rok, cok))
        if ok is not None:
            off = b.select(ok, off, b.const_i32(_i32_bits(WIDE_OOB)))
        return off

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
        d_bytes = self.d_bytes()
        rs_d = b.buffer_rsrc(C, b.const_i32(_i32_bits(d_bytes)))
        rs_c = (
            b.buffer_rsrc(self.p["Cin"], b.const_i32(_i32_bits(d_bytes)))
            if self.k.bilinear
            else None
        )
        c0 = b.const_i32(0)
        big = b.const_i32(OOB)
        cols = []
        for ni in range(mfB):
            n = b.add(n0, b.const_i32(ni * 16))
            cols.append(self._d_part(b, n, pl.d_col_digits, pl.N if rag_n else 0, big))
        alpha = self.p["alpha"]
        bf = self.dt is BF16
        if self.k.epi_v16:
            self._epilogue_v16(
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
                cols16=cols,
                D_ptr=C,
            )
            return
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

        def _rows(mi):
            rows = []
            for v0 in range(0, 4, vd):
                m = b.add(m0, b.const_i32(mi * 16 + v0))
                rows.append(
                    self._d_part(b, m, pl.d_row_digits, pl.M if rag_m else 0, big)
                )
            return rows

        def _cload(rows):
            cl = {}
            for ni in range(mfB):
                for gi, roff in enumerate(rows):
                    off = self._d_off(b, roff, cols[ni])
                    if vd > 1:
                        cv = b.buffer_load_vN(rs_c, off, c0, self.dt, vd)
                        cl[(ni, gi)] = [b.vec_extract(cv, e) for e in range(vd)]
                    else:
                        cl[(ni, gi)] = [b.buffer_load(rs_c, off, c0, self.dt)]
            return cl

        # c_prefetch d: the C loads of atom row mi + d are issued before row
        # mi's math and stores (rows 0 .. d-1 up front)
        cpf = int(self.k.c_prefetch) if self.k.bilinear else 0
        pend: Dict[int, Tuple[list, dict]] = {}
        for mi in range(min(cpf, mfA)):
            r_ = _rows(mi)
            pend[mi] = (r_, _cload(r_))
        for mi in range(mfA):
            if cpf:
                rows, cl = pend.pop(mi)
                if mi + cpf < mfA:
                    r_ = _rows(mi + cpf)
                    pend[mi + cpf] = (r_, _cload(r_))
            else:
                rows = _rows(mi)
                # bilinear: every C load of this atom row first (latency
                # overlaps), then math + stores
                cl = _cload(rows) if self.k.bilinear else {}
            for ni in range(mfB):
                acc = acc_out(vals[mi * mfB + ni])
                for gi, roff in enumerate(rows):
                    # an OOB row / column (>= 0x7FFFFFF0) plus an in-range
                    # offset (< 2^31) stays out of range (no 32-bit wrap);
                    # a wide D masks the summed offset (_d_off)
                    off = self._d_off(b, roff, cols[ni])
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

    def epilogue_rve(
        self,
        b,
        *,
        accf,
        row0,
        col0,
        mfmas_m,
        mfmas_n,
        c_per_lane,
        C,
        M,
        N,
        block_m_off,
        block_n_off,
        warp_n_idx,
        lane,
    ) -> None:
        """Permuted alpha (/ beta) epilogue on the row-interleave vector
        epilogue's lane layout (the ``_mono`` hook interface of
        gemm_universal): lane (q, cn) holds rows ``row0 + v * mfmas_m + mi``
        x columns ``col0 + ni`` (``ni < mfmas_n = 8``) as f32 accumulators
        ``accf[mi * mfmas_n + ni][v]``. GEMM B owns D's fastest mode
        (``plan.vec_dc8``), so the 8 columns are 8 contiguous D elements: one
        16-byte store (and C load) per (mi, v); rows / columns past the
        tensor fall outside the buffer range."""
        pl = self.plan
        if mfmas_n != 8:
            raise ValueError(
                "the GETT row-interleave epilogue needs 8 columns per lane"
            )
        c0 = b.const_i32(0)
        big = b.const_i32(OOB)
        d_bytes = self.d_bytes()
        rs_d = b.buffer_rsrc(C, b.const_i32(_i32_bits(d_bytes)))
        rs_c = (
            b.buffer_rsrc(self.p["Cin"], b.const_i32(_i32_bits(d_bytes)))
            if self.k.bilinear
            else None
        )
        bf = self.dt is BF16
        st = b.buffer_store_vN_bf16 if bf else b.buffer_store_vN_f16
        rag_m = pl.M % self.bmn["A"] != 0
        rag_n = pl.N % self.bmn["B"] != 0
        cpart = self._d_part(b, col0, pl.d_col_digits, pl.N if rag_n else 0, big)
        alpha = self.p["alpha"]
        rows = [(mi, v) for mi in range(mfmas_m) for v in range(c_per_lane)]

        def _off(mi, v):
            m = b.add(row0, b.const_i32(v * mfmas_m + mi))
            return self._d_off(
                b,
                self._d_part(b, m, pl.d_row_digits, pl.M if rag_m else 0, big),
                cpart,
            )

        cpf = int(self.k.c_prefetch) if self.k.bilinear else 0
        pend = {}

        def _cload(i):
            mi, v = rows[i]
            off = _off(mi, v)
            cv = b.buffer_load_vN(rs_c, off, c0, self.dt, 8)
            return off, [b.vec_extract(cv, e) for e in range(8)]

        for i in range(min(cpf, len(rows))):
            pend[i] = _cload(i)
        for i, (mi, v) in enumerate(rows):
            if self.k.bilinear:
                if cpf:
                    off, cl = pend.pop(i)
                    if i + cpf < len(rows):
                        pend[i + cpf] = _cload(i + cpf)
                else:
                    off, cl = _cload(i)
            else:
                off, cl = _off(mi, v), None
            xs = [
                b.fmul(alpha, b.vec_extract(accf[mi * mfmas_n + ni], v))
                for ni in range(8)
            ]
            if cl is not None:
                xs = [
                    b.fma(self.p["beta"], b.cast_to_f32(cc), x) for cc, x in zip(cl, xs)
                ]
            vec = b.vec_pack([b.cast_f32_to(x, self.dt) for x in xs], self.dt)
            st(rs_d, off, c0, vec, 4)

    # mono-hook interface: no producer tasks (gemm_universal's persistent loop)
    writes_operands = False

    def pre_tasks(self, b, **kw) -> None:
        return None

    def wait_tile(self, b, *a, **kw) -> None:
        return None

    def post_tile(self, b, *a, **kw) -> None:
        return None

    def _epilogue_v16(
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
        cols16,
        D_ptr=None,
    ):
        """16-byte D (and C) accesses on the MN body. A's fragments are read
        with the quad permutation (``MnLayout.sub_perm``), so accumulator lane
        group q holds M rows ``4 QUAD_PERM[q] + 0..3`` of its atom: lanes L and
        L + 32 hold the two halves of 8 consecutive rows. For each M atom pair
        (2j, 2j+1) and column atom ni, one ``v_permlane32_swap`` per dword
        gives the low lanes atom 2j's 8 rows and the high lanes atom 2j+1's
        (rows ``8 (q % 2) + 0..7``, same column): one 16-byte store per lane.
        Scale-only swaps the packed 16-bit pairs (2 swaps); bilinear swaps the
        f32 accumulators (4 swaps) and rounds once after ``+ beta * C``."""
        from ....core.ir import I32, VectorType

        pl = self.plan
        mfA, mfB = self.mf["A"], self.mf["B"]
        c0 = b.const_i32(0)
        q = b.div(lane, b.const_i32(16))
        m_lane = b.add(
            b.add(bm, b.mul(warp_m_idx, b.const_i32(self.wmn["A"]))),
            b.add(
                b.mul(b.div(lane, b.const_i32(32)), b.const_i32(16)),
                b.mul(b.mod(q, b.const_i32(2)), b.const_i32(8)),
            ),
        )
        rag_m = pl.M % self.bmn["A"] != 0
        alpha = self.p["alpha"]
        i32x2 = VectorType(I32, 2)
        vec8 = VectorType(self.dt, 8)
        st = b.buffer_store_vN_bf16 if bf else b.buffer_store_vN_f16

        nt = bool(self.k.store_nt)
        if nt:
            from ....core.ir import TemporalHint

            rag_n = pl.N % self.bmn["B"] != 0
            c16 = b.const_i32(16)
            n_lane = b.add(
                b.add(bn, b.mul(warp_n_idx, b.const_i32(self.wmn["B"]))),
                b.mod(lane, c16),
            )
            cols_e = []
            for ni in range(mfB):
                n = b.add(n_lane, b.const_i32(16 * ni))
                cols_e.append(
                    (
                        self._digits(b, n, pl.d_col_digits),
                        b.cmp_lt(n, b.const_i32(pl.N)) if rag_n else None,
                    )
                )

        def _row(j):
            m = b.add(m_lane, b.const_i32(32 * j))
            if nt:
                return (
                    self._digits(b, m, pl.d_row_digits),
                    b.cmp_lt(m, b.const_i32(pl.M)) if rag_m else None,
                )
            return self._d_part(b, m, pl.d_row_digits, pl.M if rag_m else 0, big)

        def _elem(roff, ni):
            """store_nt: (element offset, in-range predicate or None)."""
            (re, rok), (ce, cok) = roff, cols_e[ni]
            ok = rok if cok is None else (cok if rok is None else b.land(rok, cok))
            return b.add(re, ce), ok

        def _boff(roff, ni):
            """Byte offset of the lane's 8 D (C) elements; past the tensor:
            out of the buffer range."""
            if not nt:
                return self._d_off(b, roff, cols16[ni])
            e, ok = _elem(roff, ni)
            off = b.add_nuw(e, e)
            if ok is not None:
                sent = WIDE_OOB if self.wide_d() else OOB
                off = b.select(ok, off, b.const_i32(_i32_bits(sent)))
            return off

        def _cload(roff):
            out = {}
            for ni in range(mfB):
                cv = b.buffer_load_vN(rs_c, _boff(roff, ni), c0, self.dt, 8)
                out[ni] = [b.vec_extract(cv, e) for e in range(8)]
            return out

        npair = mfA // 2
        cpf = int(self.k.c_prefetch) if self.k.bilinear else 0
        pend = {}
        for j in range(min(cpf, npair)):
            r_ = _row(j)
            pend[j] = (r_, _cload(r_))
        for j in range(npair):
            if cpf:
                roff, cl = pend.pop(j)
                if j + cpf < npair:
                    r_ = _row(j + cpf)
                    pend[j + cpf] = (r_, _cload(r_))
            else:
                roff = _row(j)
                cl = _cload(roff) if self.k.bilinear else {}
            for ni in range(mfB):
                ax = acc_out(vals[(2 * j) * mfB + ni])
                ay = acc_out(vals[(2 * j + 1) * mfB + ni])
                off = None if nt else self._d_off(b, roff, cols16[ni])
                if self.k.bilinear:
                    lo, hi_ = [], []
                    for e in range(4):
                        x2, y2 = b.permlane32_swap(
                            b.bitcast(b.vec_extract(ax, e), I32),
                            b.bitcast(b.vec_extract(ay, e), I32),
                        )
                        lo.append(b.bitcast(x2, F32))
                        hi_.append(b.bitcast(y2, F32))
                    xs = [
                        b.fma(self.p["beta"], b.cast_to_f32(cc), b.fmul(alpha, x))
                        for cc, x in zip(cl[ni], lo + hi_)
                    ]
                    vec = b.vec_pack([b.cast_f32_to(x, self.dt) for x in xs], self.dt)
                else:
                    da = b.vec_bitcast(
                        b.vec_pack(
                            [
                                b.cast_f32_to(
                                    b.fmul(alpha, b.vec_extract(ax, e)), self.dt
                                )
                                for e in range(4)
                            ],
                            self.dt,
                        ),
                        i32x2,
                    )
                    db = b.vec_bitcast(
                        b.vec_pack(
                            [
                                b.cast_f32_to(
                                    b.fmul(alpha, b.vec_extract(ay, e)), self.dt
                                )
                                for e in range(4)
                            ],
                            self.dt,
                        ),
                        i32x2,
                    )
                    x0, y0 = b.permlane32_swap(
                        b.vec_extract(da, 0), b.vec_extract(db, 0)
                    )
                    x1, y1 = b.permlane32_swap(
                        b.vec_extract(da, 1), b.vec_extract(db, 1)
                    )
                    vec = b.vec_bitcast(b.vec_pack([x0, x1, y0, y1], I32), vec8)
                if not nt:
                    st(rs_d, off, c0, vec, 4)
                    continue
                e, ok = _elem(roff, ni)

                def _st_nt(e=e, vec=vec):
                    b.global_store_vN(
                        D_ptr, e, vec, 8, align=16, temporal_hint=TemporalHint.STREAMING
                    )

                if ok is None:
                    _st_nt()
                else:
                    with b.scf_if(ok):
                        _st_nt()

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
            cols.append(self._d_part(b, n, pl.d_col_digits, pl.N if rag_n else 0, big))
        alpha = self.p["alpha"]

        def _row(mi):
            m = b.add(m0, b.const_i32(mi * 16))
            return self._d_part(b, m, pl.d_row_digits, pl.M if rag_m else 0, big)

        def _cload(roff):
            cl = {}
            for j in range(mfB // 2):
                cv = b.buffer_load_vN(
                    rs_c, self._d_off(b, roff, cols[j]), c0, self.dt, 8
                )
                cl[j] = [b.vec_extract(cv, e) for e in range(8)]
            return cl

        # c_prefetch d: as in :meth:`epilogue`, the C loads of atom row mi + d are
        # issued before row mi's math and stores (rows 0 .. d-1 up front)
        cpf = int(self.k.c_prefetch) if self.k.bilinear else 0
        pend: Dict[int, Tuple[Value, dict]] = {}
        for mi in range(min(cpf, mfA)):
            r_ = _row(mi)
            pend[mi] = (r_, _cload(r_))
        for mi in range(mfA):
            if cpf:
                roff, cl = pend.pop(mi)
                if mi + cpf < mfA:
                    r_ = _row(mi + cpf)
                    pend[mi + cpf] = (r_, _cload(r_))
            else:
                roff = _row(mi)
                cl = _cload(roff) if self.k.bilinear else {}
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
                    rs_d, self._d_off(b, roff, cols[j]), c0, vec, 4
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


#: Trait fields of an MN-layout GETT body that must keep the TraitSpec
#: default: they change the operand addressing, the K range, the LDS layout of
#: a K-contiguous operand, the epilogue or the work split, all of which the
#: GETT hook replaces or does not model. Bounds come from the hook's own
#: per-lane masks, so the padded / tight-resource levers of a ragged GEMM pick
#: are dropped by :func:`mn_body_trait` instead.
MN_BODY_FIXED_FIELDS = (
    "pad_k",
    "persistent",
    "preshuffle_b",
    "active_tile_skip",
    "split_k",
    "mfma_swap_ab",
    "sab_vec16",
    "sab_store_nt",
    "streamk",
    "lds_row_interleave",
    "rilv_vec_epilogue",
    "rve_store_nt",
    "hb_sched",
    "sia_sched",
    "k_stagger",
    "k_stagger_rc",
    "stagger_u",
    "dtl_clamp_k",
    "dtl_k_tail",
    "dtl_base_fold",
    "pingpong",
    "asm_loop",
    "asm_loop_rawrd",
    "vgpr_prefetch",
    "vp_mn_kv",
    "mn_vec",
    "kc_row_guard",
    "pad_row_clamp",
    "pad_k_fast",
    "split_k_aligned",
    "split_k_serial",
    "dtl_addr_spread",
    "dtl_rsrc_word3",
)
#: Levers of a GEMM pick that the MN GETT body drops (bounds and operand
#: origins are the hook's own).
MN_BODY_DROPPED = dict(
    epilogue="default",
    cshuffle_no_alias=False,
    pad_m=False,
    pad_n=False,
    dtl_tight_rsrc=False,
    dtl_tight_ragged_only=False,
    pad_fast_tile=False,
    dtl_soff0=False,
)


#: Trait fields of a K-contiguous GETT body that must keep the TraitSpec
#: default (they change the epilogue's layout or write f32 partial sums).
KC_BODY_FIXED_FIELDS = (
    "persistent",
    "split_k",
    "mfma_swap_ab",
    "sab_vec16",
    "sab_store_nt",
    "streamk",
    "pad_k",
    "dtl_k_tail",
    "active_tile_skip",
)
#: Levers of an RCR GEMM pick that the K-contiguous GETT body drops: the
#: hook's epilogue owns the output bounds (its stores honour the D buffer
#: range), so the epilogue-side levers go; the operand-side bounds (pad_m /
#: pad_n with dtl_tight_rsrc) and rve_pad (the vector epilogue's validity
#: rule for padded bodies; its store path is the hook's) stay.
KC_BODY_DROPPED = dict(
    rve_store_nt=False,
    pad_fast_tile=False,
    cshuffle_no_alias=False,
)


def kc_body_trait(trait: TraitSpec) -> TraitSpec:
    """An RCR GEMM pick's trait as a K-contiguous GETT body trait (a cshuffle
    epilogue becomes the direct one: the hook writes D)."""
    t = dataclasses.replace(trait, **KC_BODY_DROPPED)
    if t.epilogue != "default":
        t = dataclasses.replace(t, epilogue="default")
    return t


def kc_body_errors(spec: UniversalGemmSpec) -> List[str]:
    """Why ``spec`` is not a K-contiguous GETT body (empty list: it is one)."""
    t, tr = spec.tile, spec.trait
    ref = TraitSpec()
    errs = []
    if spec.data.layout != "RCR":
        errs.append(f"layout {spec.data.layout} (the K-contiguous body is RCR)")
    if (t.warp_tile_m, t.warp_tile_n) != (16, 16):
        errs.append("atoms must be 16x16 (the epilogue's lane layout)")
    if tr.epilogue != "default":
        errs.append(
            f"epilogue={tr.epilogue!r} (the GETT epilogue replaces the direct one)"
        )
    for f in KC_BODY_FIXED_FIELDS:
        got, want = getattr(tr, f), getattr(ref, f)
        if got != want:
            errs.append(f"{f}={got!r} (the K-contiguous GETT body needs {want!r})")
    for f, want in KC_BODY_DROPPED.items():
        if getattr(tr, f) != want:
            errs.append(
                f"{f}={getattr(tr, f)!r} (dropped on the K-contiguous GETT body)"
            )
    if getattr(spec, "_mono", None) is not None:
        errs.append("the _mono epilogue hook")
    return errs


def mn_body_trait(trait: TraitSpec) -> TraitSpec:
    """A both-MN-contiguous GEMM pick's trait as an MN GETT body trait."""
    return dataclasses.replace(trait, **MN_BODY_DROPPED)


def mn_body_errors(spec: UniversalGemmSpec) -> List[str]:
    """Why ``spec`` is not an MN-layout GETT body (empty list: it is one)."""
    t, tr = spec.tile, spec.trait
    ref = TraitSpec()
    errs = []
    if spec.data.layout != "CRR":
        errs.append(f"layout {spec.data.layout} (the MN body is CRR)")
    if (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k) != (16, 16, 32):
        errs.append("atoms must be 16x16x32")
    if t.warp_k != 1:
        errs.append("warp_k must be 1")
    if not (tr.direct_to_lds and tr.dtl_prefetch and int(tr.dtl_stages) >= 2):
        errs.append(
            "needs the alias-scoped DTL ring (direct_to_lds, dtl_prefetch, dtl_stages >= 2)"
        )
    if tr.epilogue != "default":
        errs.append(
            f"epilogue={tr.epilogue!r} (the GETT epilogue replaces the direct one)"
        )
    for f in MN_BODY_FIXED_FIELDS:
        got, want = getattr(tr, f), getattr(ref, f)
        if got != want:
            errs.append(f"{f}={got!r} (the MN GETT body needs {want!r})")
    for f, want in MN_BODY_DROPPED.items():
        if getattr(tr, f) != want:
            errs.append(f"{f}={getattr(tr, f)!r} (dropped on the MN GETT body)")
    if int(tr.persistent_ctas) < 0:
        errs.append(f"persistent_ctas={tr.persistent_ctas} (>= 0)")
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
#: Largest operand byte span the DTL buffer offsets address (row offsets past
#: the tensor are pushed to :data:`OOB`). D / C: :data:`MAX_D_BYTES`.
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
    #: MN-layout body: a both-MN-contiguous (CRR) ring GEMM tile / trait (e.g.
    #: the GEMM dispatcher's pick for the same M, N, K). ``None``: the ring
    #: body above (:data:`BODY_TRAIT`; tile_k / dtl_stages / persistent /
    #: chiplet fields). With an MN body those fields keep their defaults and
    #: ``tile_k`` equals the body's.
    mn_tile: Optional[TileSpec] = None
    mn_trait: Optional[TraitSpec] = None
    #: K-contiguous body: an RCR GEMM tile / trait (e.g. the GEMM dispatcher's
    #: RCR pick for the plan's M x N x K). The kernel's A / B params then take
    #: K-contiguous packed copies ([rows, K] row-major in the plan's row and K
    #: orders: :func:`kc_operand_modes`); the hook owns only the epilogue.
    #: Excludes ``mn_tile``.
    kc_tile: Optional[TileSpec] = None
    kc_trait: Optional[TraitSpec] = None

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

    @property
    def kc_body(self) -> bool:
        return self.kc_tile is not None

    def plan(self) -> GettPlan:
        """The host-side plan (raises ValueError when the problem has no GETT
        plan: an operand that is not MN-contiguous, K modes that do not split
        into ``tile_k``, batch / Hadamard modes)."""
        if self.kc_body:
            tr = self.kc_trait or TraitSpec()
            return plan_gett(
                self.modes_a,
                self.modes_b,
                self.modes_d,
                dict(self.extents),
                strides_a=_as_dict(self.strides_a),
                strides_b=_as_dict(self.strides_b),
                strides_d=_as_dict(self.strides_d),
                tile_k=self.kc_tile.tile_k,
                kc=True,
                d_fast_role="B" if tr.rilv_vec_epilogue else "A",
            )
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

    @property
    def mn_body(self) -> bool:
        return self.mn_tile is not None

    def body(self) -> UniversalGemmSpec:
        """The universal GEMM body spec this launch builds on."""
        if self.kc_body:
            dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
            return UniversalGemmSpec(
                name=self.kernel_name(),
                tile=self.kc_tile,
                trait=kc_body_trait(self.kc_trait or TraitSpec()),
                data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="RCR"),
            )
        if self.mn_body:
            dt = "fp16" if self.dtype in ("f16", "fp16") else self.dtype
            return UniversalGemmSpec(
                name=self.kernel_name(),
                tile=self.mn_tile,
                trait=mn_body_trait(self.mn_trait or TraitSpec()),
                data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="CRR"),
            )
        trait = dict(
            dtl_stages=self.dtl_stages,
            persistent_ctas=self.persistent_ctas,
            chiplet_wgm=self.chiplet_wgm,
            chiplet_chunk_size=self.chiplet_chunk_size,
        )
        sp = body_spec(self.dtype, tile_k=self.tile_k, **trait)
        return dataclasses.replace(sp, name=self.kernel_name())

    def kernel_name(self) -> str:
        d = dataclasses.asdict(self)
        # fields added after the first release are hashed only when set, so
        # every earlier spec keeps its kernel name
        for f in ("mn_tile", "mn_trait", "kc_tile", "kc_trait"):
            if d[f] is None:
                del d[f]
        for f, dflt in (("epi_v16", False), ("c_prefetch", 0), ("store_nt", False)):
            if d["knobs"][f] == dflt:
                del d["knobs"][f]
        key = json.dumps(d, sort_keys=True)
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
    if not 0 <= int(k.c_prefetch) <= 8:
        return False, f"knobs.c_prefetch {k.c_prefetch} not in 0..8"
    if spec.kc_body:
        if spec.mn_body or spec.mn_trait is not None:
            return False, "kc_tile excludes mn_tile"
        legacy = dict(
            tile_k=32,
            dtl_stages=3,
            persistent_ctas=0,
            chiplet_wgm=8,
            chiplet_chunk_size=64,
        )
        for f, v in legacy.items():
            if getattr(spec, f) != v:
                return False, f"{f} is a ring-body field; set it on kc_trait"
    elif spec.kc_trait is not None:
        return False, "kc_trait without kc_tile"
    elif spec.mn_body:
        if spec.mn_tile.tile_k != spec.tile_k:
            return False, (
                f"tile_k {spec.tile_k} != the MN body's tile_k {spec.mn_tile.tile_k}"
            )
        legacy = dict(
            dtl_stages=3, persistent_ctas=0, chiplet_wgm=8, chiplet_chunk_size=64
        )
        for f, v in legacy.items():
            if getattr(spec, f) != v:
                return False, f"{f} is a ring-body field; set it on mn_trait"
    elif spec.mn_trait is not None:
        return False, "mn_trait without mn_tile"
    try:
        plan = spec.plan()
    except ValueError as e:
        return False, f"no GETT plan: {e}"
    for nm in ("A", "B"):
        if plan.ops[nm].nbytes > MAX_TENSOR_BYTES:
            return False, f"operand {plan.ops[nm].name} spans more than 2 GiB"
    if plan.kc:
        from ..gemm_universal import launch_shape_error

        why = launch_shape_error(spec.body(), plan.M, plan.N, plan.K)
        if why is not None:
            return False, f"K-contiguous body: {why}"
    if _span_bytes(plan.ext, plan.strides_d, plan.modes_d) > MAX_D_BYTES:
        return False, f"D spans more than {MAX_D_BYTES} bytes (32-bit offsets)"
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
    plan = spec.plan()
    if spec.mn_body or spec.kc_body:
        t, pc = spec.body().tile, int(spec.body().trait.persistent_ctas)
        tiles_n, tiles_m = -(-plan.N // t.tile_n), -(-plan.M // t.tile_m)
        if pc:
            return (max(1, min(pc, tiles_n * tiles_m)), 1, 1)
        return (tiles_n, tiles_m, 1)
    if spec.persistent_ctas:
        return (int(spec.persistent_ctas), 1, 1)
    return (-(-plan.N // 256), -(-plan.M // 256), 1)


def tensor_gett_block(spec: TensorGettSpec) -> Tuple[int, int, int]:
    """Workgroup size of the launch (the body's ``block_size``)."""
    return (int(spec.body().block_size), 1, 1)


def tensor_gett_signature(spec: TensorGettSpec):
    dt = "f16" if spec.dtype in ("f16", "fp16") else spec.dtype
    sb = SignatureBuilder().ptr("A", dt).ptr("B", dt).ptr("C", dt)
    sb = sb.scalar("M", "i32").scalar("N", "i32").scalar("K", "i32")
    sb = sb.scalar("alpha", "f32")
    if spec.bilinear:
        sb = sb.ptr("Cin", dt).scalar("beta", "f32")
    return sb.build()


def kc_operand_modes(spec: TensorGettSpec) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Row-major mode orders of the K-contiguous packed copies the kernel's A
    and B params take (``kc_body``): [GEMM rows (outer -> inner), K modes
    (outer -> inner)]."""
    plan = spec.plan()
    k_outer = tuple(reversed(plan.k_order))
    return (
        tuple(plan.ops["A"].rows_order) + k_outer,
        tuple(plan.ops["B"].rows_order) + k_outer,
    )


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
    "MAX_D_BYTES",
    "MAX_TENSOR_BYTES",
    "TENSOR_GETT_ARCHS",
    "TENSOR_GETT_DTYPES",
    "TensorGettSpec",
    "body_spec",
    "body_trait_errors",
    "build_tensor_gett",
    "is_valid_spec",
    "launch_operands",
    "tensor_gett_block",
    "tensor_gett_grid",
    "tensor_gett_signature",
    "kc_body_errors",
    "kc_body_trait",
    "kc_operand_modes",
    "mn_body_errors",
    "mn_body_trait",
]
