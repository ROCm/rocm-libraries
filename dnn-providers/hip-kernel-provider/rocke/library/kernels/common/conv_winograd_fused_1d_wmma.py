# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused 1-D Winograd forward conv (gfx1250, wave32 WMMA).

One kernel computes a ``1 x r`` conv (``Y = 1``, stride 1, any pad and
dilation) as ``F(m, r)`` or segmented ``S x F(m, m)`` Winograd
(:mod:`kernels.common.conv_winograd_gen`), from raw x to y:

* **Weight-stationary in registers.** A workgroup owns one block of ``Kb``
  output channels. Wave ``w`` owns bins ``w, w + 8, ...``; the transformed
  weights of its bins (``S x C/32 x Kb/16`` A fragments each) are loaded once
  from ``U`` (written by the generalized K1) and stay in VGPRs.
* **Persistent** over blocks of ``Tb = 16 Q`` consecutive tiles of one
  dilation sub-grid row (grid-stride loop).
* Per tile block (three barriers):

  1. the raw input segment (``(Tb + S - 1) m + n - m`` positions x all C) is
     loaded with buffer loads (pad / dilation holes / image end read 0) and
     stored to LDS;
  2. input transform (fp32 VALU): ``V[b][j][c] = sum_u B^T[b][u] x[j m + u][c]``
     for the ``Tb + S - 1`` tiles (the last ``S - 1`` are the segment halo),
     stored to LDS as fp16;
  3. per-bin GEMMs: ``M_b[j] = sum_s V_b[j + s] U_{b,s}`` with
     ``wmma_f32_16x16x32_f16`` (A = U fragment, B = V fragment). Lane column
     ``l`` of fragment ``q`` is tile ``j = Q l + q`` (strided), so the
     segment-shifted fragment ``V_b[j + s]`` is fragment ``q + s``: a wave
     reads ``Q + S - 1`` V fragments per 32-channel step instead of ``Q S``.
     Accumulators go to LDS as fp32 ``M[b][j][k]``;
  4. output transform (fp32 VALU): ``y[(t0 + j) m + i] = sum_b A^T[i][b] M_b[j]``,
     8 channels per lane, 16-byte stores (positions past ``Wo`` dropped).

All LDS regions are exclusive allocations (no liveness aliasing across the
persistent loop). Every barrier is outside any wave-dependent branch and the
loop trip count is uniform per workgroup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import _OOB, _load_raw, _raw_to_f32
from kernels.common.conv_winograd_gen import WinoAxis

WAVES = 8
WAVE = 32
VEC = 8  # channels per lane in the transforms (16-byte LDS / global accesses)
PAD_H = 8  # fp16 row skew (16 bytes)
PAD_F = 4  # fp32 row skew (16 bytes)
VGPR_BUDGET = 440


@dataclass(frozen=True)
class FusedWino1DSpec:
    problem: ConvProblem
    ax: WinoAxis  # W-axis transform (F(m, r) or S x F(m, m))
    q: int = 4  # 16-tile fragments per tile block (Tb = 16 q)
    fk: int = 2  # 16-channel K fragments per block (Kb = 16 fk)
    grid_blocks: int = 256  # persistent workgroups (rounded to a K-block multiple)
    # Async buffer->LDS prefetch of the next tile block's raw input into a
    # second (unpadded) raw buffer, overlapping the GEMM / output phases.
    prefetch: bool = False
    m_f16: bool = False  # stage M in LDS as fp16 (frees LDS for prefetch)
    # Input transform: "valu" (fp32 FMAs) or "wmma" (V^T = X^T B, with X^T
    # transpose-read from the raw rows and a constant block-Toeplitz B^T;
    # needs every B^T entry exact in fp16).
    xf: str = "valu"
    xf_batch: int = 4  # WMMA transform items per wave issued together
    # Merge the output transform of block i and the input transform of
    # block i + 1 into one phase (2 barriers per block; needs prefetch).
    merge: bool = False
    # Waves per workgroup: 8 (2 per SIMD, ~480 VGPRs each) or 4 (1 per SIMD,
    # ~960 VGPRs: room for twice the register-resident weights).
    waves: int = 8
    # amdgpu-waves-per-eu hint (0 = none): 2 with 4 waves caps VGPRs at 512
    # so two workgroups can share a CU (LDS permitting).
    waves_per_eu: int = 0
    name: str = "conv_wino_fused1d"

    def __post_init__(self) -> None:
        why = self.reason()
        if why:
            raise ValueError(f"FusedWino1DSpec: {why}")

    def reason(self, arch: str = "gfx1250"):
        p = self.problem
        if p.is_3d or p.Y != 1 or p.Hi != 1:
            return "1-D convs only (Y = 1, H = 1)"
        if (p.sH, p.sW) != (1, 1) or p.pH != 0 or p.groups != 1:
            return "stride 1, no H pad, groups 1"
        if self.ax.r != p.X:
            return "axis taps must match the filter"
        try:
            self.ax.check()
        except ValueError as e:
            return str(e)
        if p.C % 32 or p.K % 16:
            return "C must be a multiple of 32 and K of 16"
        if p.K % self.kb:
            return f"K {p.K} not a multiple of Kb {self.kb}"
        if self.ax.n > 16:
            return "n > 16"
        if self.merge and not self.prefetch:
            return "merge needs prefetch"
        if self.xf not in ("valu", "wmma"):
            return "xf must be valu / wmma"
        if self.xf == "wmma":
            import numpy as np

            _, _, bt = self.ax.matrices()
            for row in bt:
                for v in row:
                    if float(np.float16(float(v))) != float(v):
                        return f"B^T entry {v} is not exact in fp16"
            if self.n + 0 > 16 or self.n > 32:
                return "n too large for the WMMA transform"
        if self.q < 1 or self.fk < 1:
            return "q, fk >= 1"
        t = ArchTarget.from_gfx(arch)
        if t.wave_size != 32:
            return "wave32 only"
        if not t.fits_lds(self.lds_bytes()):
            return f"LDS {self.lds_bytes()} B too large"
        if self.waves not in (4, 8):
            return "waves must be 4 or 8"
        if self.vgpr_estimate() > (VGPR_BUDGET if self.waves == 8 else 980):
            return f"~{self.vgpr_estimate()} VGPRs"
        if self.grid() <= 0:
            return "empty grid"
        for nb in self.buffer_bytes().values():
            if nb > _OOB:
                return "a buffer exceeds 32-bit offsets"
        return None

    # geometry
    @property
    def m(self) -> int:
        return self.ax.m

    @property
    def n(self) -> int:
        return self.ax.n

    @property
    def S(self) -> int:
        return self.ax.seg

    @property
    def tb(self) -> int:
        return 16 * self.q

    @property
    def kb(self) -> int:
        return 16 * self.fk

    @property
    def bpw(self) -> int:
        """Bins per wave."""
        return ceil_div(self.n, self.waves)

    @property
    def sub_w(self) -> int:
        return ceil_div(self.problem.Wo, self.problem.dW)

    @property
    def tiles_w(self) -> int:
        return ceil_div(self.sub_w, self.m)

    @property
    def rows(self) -> int:
        return self.problem.N * self.problem.dW

    @property
    def blocks_per_row(self) -> int:
        return ceil_div(self.tiles_w, self.tb)

    @property
    def tile_blocks(self) -> int:
        return self.rows * self.blocks_per_row

    @property
    def k_blocks(self) -> int:
        return self.problem.K // self.kb

    @property
    def vt(self) -> int:
        """V tiles per block (with the segment halo)."""
        return self.tb + self.S - 1

    @property
    def raw_rows(self) -> int:
        return self.vt * self.m + self.n - self.m

    def grid(self) -> int:
        kbn = self.k_blocks
        per_k = max(1, self.grid_blocks // kbn)
        per_k = min(per_k, self.tile_blocks)
        return per_k * kbn

    @property
    def xf_tiles(self) -> int:
        """Tiles per transform WMMA (columns = tiles x bins <= 16, the 32
        position window covers them)."""
        t = 16 // self.n
        while t > 1 and (t - 1) * self.m + self.n > 32:
            t -= 1
        return t

    @property
    def xf_win(self) -> int:
        """Positions the transpose read covers: 16 when the tile group fits
        (the upper k half then re-reads rows 0..15, which meet zero B rows)."""
        return 16 if (self.xf_tiles - 1) * self.m + self.n <= 16 else 32

    @property
    def xf_groups(self) -> int:
        return ceil_div(self.vt, self.xf_tiles)

    @property
    def raw_alloc_rows(self) -> int:
        """Rows of one raw buffer (+ a dummy row). The WMMA transform reads
        32-row windows, so rows past the block's raw rows exist and are
        zero-filled once."""
        rows = self.raw_rows
        if self.xf == "wmma":
            rows = max(
                rows, (self.xf_groups - 1) * self.xf_tiles * self.m + self.xf_win
            )
        return rows + 1

    def lds_bytes(self) -> int:
        c = self.problem.C
        raw = (2 if self.prefetch else 1) * self.raw_alloc_rows * (c + PAD_H) * 2
        v = self.n * (self.vt + 1) * (c + PAD_H) * 2
        if self.m_f16:
            mm = (self.n + 1) * self.tb * (self.kb + PAD_H) * 2
        else:
            mm = self.n * self.tb * (self.kb + PAD_F) * 4
        return raw + v + mm

    def vgpr_estimate(self) -> int:
        c32 = self.problem.C // 32
        u = self.bpw * self.S * c32 * self.fk * 8
        acc = self.q * self.fk * 8
        vf = (self.q + self.S - 1) * 8
        return u + acc + 2 * vf + 40

    def buffer_bytes(self) -> dict:
        p = self.problem
        return {
            "x": p.N * p.Wi * p.C * 2,
            "u": self.n * self.S * p.K * p.C * 2,
            "y": p.N * p.Wo * p.K * 2,
        }

    def gemm_flops(self) -> int:
        """WMMA flops actually issued for the GEMMs (incl. halo/tail tiles)."""
        p = self.problem
        return 2 * self.n * self.S * self.tile_blocks * self.tb * p.K * p.C

    def short(self) -> str:
        tag = f"{self.problem.short()}_{self.ax.label()}_q{self.q}_fk{self.fk}_g{self.grid()}"
        if self.xf != "valu":
            tag += f"_x{self.xf}{self.xf_batch}"
        if self.prefetch:
            tag += "_pf"
        if self.merge:
            tag += "_mg"
        if self.waves != 8:
            tag += f"_w{self.waves}"
        if self.waves_per_eu:
            tag += f"_o{self.waves_per_eu}"
        if self.m_f16:
            tag += "_mh"
        return tag

    def kernel_name(self) -> str:
        return kernel_name_join(self.name, self.short())


def fused_signature(spec: FusedWino1DSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", "fp16")
        .ptr("u", "fp16")
        .ptr("y", "fp16")
        .scalar("x_bytes", "i32")
        .scalar("u_bytes", "i32")
        .scalar("y_bytes", "i32")
        .build()
    )


def fused_grid(spec: FusedWino1DSpec) -> Tuple[int, int, int]:
    return (spec.grid(), 1, 1)


def build_fused_wino_1d(spec: FusedWino1DSpec, *, arch: str = "gfx1250") -> KernelDef:
    p = spec.problem
    C, K = p.C, p.K
    m, n, S, Q, FK = spec.m, spec.n, spec.S, spec.q, spec.fk
    TB, KB, VT, RR = spec.tb, spec.kb, spec.vt, spec.raw_rows
    d, pw = p.dW, p.pW
    at, _, bt = spec.ax.matrices()
    at = [[float(v) for v in r] for r in at]
    bt = [[float(v) for v in r] for r in bt]
    CP = C + PAD_H
    MP = KB + PAD_F
    C32 = C // 32
    CV = C // VEC
    WAVES = spec.waves
    NTH = WAVES * WAVE
    KBN = spec.k_blocks
    GRID = spec.grid()
    GPK = GRID // KBN  # workgroups per K block
    wmma = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype="f16", b_dtype="f16", c_dtype="fp32", m=16, n=16, k=32
    )

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = NTH
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    x = b.param("x", PtrType(F16, "global"), noalias=True, readonly=True)
    u = b.param("u", PtrType(F16, "global"), noalias=True, readonly=True)
    y = b.param("y", PtrType(F16, "global"), noalias=True, writeonly=True)
    x_rsrc = b.buffer_rsrc(x, b.param("x_bytes", I32))
    u_rsrc = b.buffer_rsrc(u, b.param("u_bytes", I32))
    y_rsrc = b.buffer_rsrc(y, b.param("y_bytes", I32))
    ci = b.const_i32
    c0 = ci(0)
    oob = ci(_OOB)

    PF = spec.prefetch
    RA = spec.raw_alloc_rows
    RPITCH = CP
    raw_lds = b.smem_alloc(
        F16, [(2 if PF else 1) * RA, RPITCH], name_hint="raw", exclusive=True
    )
    MDUMMY = n * TB  # dummy M rows (written by the waves without a bin)
    v_lds = b.smem_alloc(F16, [n * (VT + 1), CP], name_hint="vx", exclusive=True)
    MH = spec.m_f16
    MPH = KB + PAD_H
    if MH:
        m_lds = b.smem_alloc(F16, [MDUMMY + TB, MPH], name_hint="mm", exclusive=True)
    else:
        m_lds = b.smem_alloc(F32, [n * TB, MP], name_hint="mm", exclusive=True)

    tid = b.thread_id_x()
    wave = b.div(tid, ci(WAVE))
    lane = b.mod(tid, ci(WAVE))
    l16 = b.mod(lane, ci(16))
    lhalf = b.div(lane, ci(16))
    wg = b.block_id_x()
    kbi = b.mod(wg, ci(KBN))
    first = b.div(wg, ci(KBN))
    k0 = b.mul(kbi, ci(KB))

    # ---- prologue: this wave's transformed weights (A fragments) -> VGPRs.
    # A fragment rows = output channel k0 + fk*16 + l16, 16 channels
    # c = ks*32 + lhalf*16 .. +15 (same split as the B fragments below).
    ufr = []  # [slot][s][ks][fk]
    for slot in range(spec.bpw):
        binv = b.add(wave, ci(WAVES * slot))
        okb = b.cmp_lt(binv, ci(n))
        per_s = []
        for s in range(S):
            per_ks = []
            for ks in range(C32):
                per_fk = []
                for f in range(FK):
                    row = b.add(
                        b.mul(b.add(b.mul(binv, ci(S)), ci(s)), ci(K)),
                        b.add(k0, ci(f * 16)),
                    )
                    row = b.add(row, l16)
                    off = b.add(
                        b.mul(row, ci(C)), b.add(ci(ks * 32), b.mul(lhalf, ci(16)))
                    )
                    off = b.select(okb, b.mul(off, ci(2)), oob)
                    lo = _load_raw(b, u_rsrc, off, "fp16", 8)
                    hi = _load_raw(b, u_rsrc, b.add(off, ci(16)), "fp16", 8)
                    per_fk.append(b.vec_concat(lo, hi))
                per_ks.append(per_fk)
            per_s.append(per_ks)
        ufr.append(per_s)

    n_iter = b.div(b.add(b.sub(ci(spec.tile_blocks), first), ci(GPK - 1)), ci(GPK))
    if spec.xf == "wmma":
        # Transpose-read lane map of a 32 (position) x 16 (channel) block
        # (as conv_direct_dgrad_wmma's W^T read): k order matches the B
        # convention "lane half h, element e -> k = 16 h + e".
        tr_col = b.mul(b.div(l16, ci(8)), ci(8))
        if spec.xf_win == 16:
            tr_row = b.mod(lane, ci(8))
        else:
            tr_row = b.add(b.mul(lhalf, ci(16)), b.mod(lane, ci(8)))
        TPW = spec.xf_tiles
        xf_tg = b.div(l16, ci(n))
        xf_bin = b.mod(l16, ci(n))
        xf_colok = b.cmp_lt(l16, ci(TPW * n))
        # Bt[p][(tg, bin)] = B^T[bin][p - tg * m] (0 outside the tile window)
        elems = []
        for e in range(16):
            pos = b.add(b.mul(lhalf, ci(16)), ci(e))
            rel = b.sub(pos, b.mul(xf_tg, ci(m)))
            val = b.const_f32(0.0)
            for bb in range(n):
                for uu in range(n):
                    cf = bt[bb][uu]
                    if cf == 0.0:
                        continue
                    hit = b.land(
                        b.land(b.cmp_eq(xf_bin, ci(bb)), b.cmp_eq(rel, ci(uu))),
                        xf_colok,
                    )
                    val = b.select(hit, b.const_f32(cf), val)
            elems.append(b.cast_f32_to(val, F16))
        bt_frag = b.vec_pack(elems, F16)
        # Rows past the raw block are read by the 32-row windows with zero
        # weight: make them finite (0) once, in every raw buffer.
        tail0 = RR
        n_tail = (RA - 1 - tail0) * CV
        for buf in range(2 if PF else 1):
            for rep in range(ceil_div(n_tail, NTH)):
                item = b.add(tid, ci(rep * NTH))
                ok = b.cmp_lt(item, ci(n_tail))
                rr_ = b.select(
                    ok,
                    b.add(ci(buf * RA + tail0), b.div(item, ci(CV))),
                    ci(buf * RA + RA - 1),
                )
                b.smem_store_vN(
                    raw_lds,
                    [rr_, b.mul(b.mod(item, ci(CV)), ci(VEC))],
                    b.zero_vec(F16, VEC),
                    n=VEC,
                )
        b.sync_lds_only()

    def decode(blk):
        row = b.div(blk, ci(spec.blocks_per_row))
        tbi = b.mod(blk, ci(spec.blocks_per_row))
        img = b.div(row, ci(d)) if d > 1 else row
        a = b.mod(row, ci(d)) if d > 1 else c0
        return img, a, b.mul(tbi, ci(TB))

    def raw_offsets(img, a, t0, item, n_raw):
        """x byte offset of raw item (row pr, channel vector cv), OOB -> 0."""
        wi0 = b.sub(b.add(a, b.mul(b.mul(t0, ci(m)), ci(d))), ci(pw))
        pr = b.div(item, ci(CV))
        cv = b.mod(item, ci(CV))
        wi = b.add(wi0, b.mul(pr, ci(d)))
        okit = b.cmp_lt(item, ci(n_raw))
        ok = b.land(okit, b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, ci(p.Wi))))
        elem = b.add(b.mul(b.add(b.mul(img, ci(p.Wi)), wi), ci(C)), b.mul(cv, ci(VEC)))
        return b.select(ok, b.mul(elem, ci(2)), oob), okit, pr, cv, ok, elem

    n_raw = RR * CV
    zero8 = b.zero_vec(F16, VEC)

    def issue_raw(blk, buf):
        """Async global->LDS copy of block ``blk``'s raw input into raw
        buffer ``buf``; positions outside the image (pad, dilation tails)
        get zeros through a plain LDS store instead."""
        img, a, t0 = decode(blk)
        rb = b.mul(buf, ci(RA))
        for rep in range(ceil_div(n_raw, NTH)):
            item = b.add(tid, ci(rep * NTH))
            _, okit, pr, cv, ok, elem = raw_offsets(img, a, t0, item, n_raw)
            idx = [b.add(rb, pr), b.mul(cv, ci(VEC))]
            with b.scf_if_else(ok) as (then_ctx, else_ctx):
                with then_ctx:
                    b.global_load_async_to_lds(x, elem, raw_lds, idx, width_bytes=16)
                with else_ctx:
                    with b.scf_if(okit):
                        b.smem_store_vN(raw_lds, idx, zero8, n=VEC)

    def start_raw(blk, buf, guard=None):
        """Begin moving block ``blk``'s raw input into raw slot ``buf``
        (``guard``: wave-uniform i1, False = no such block)."""
        if guard is None:
            issue_raw(blk, buf)
            return
        with b.scf_if(guard):
            issue_raw(blk, buf)

    def land_raw():
        """Complete :func:`start_raw` (before the barrier that publishes it):
        every wave waits for its own copies."""
        b.s_wait_asynccnt(0)

    def do_xf(rrow0):
        vb = c0  # V buffer base row
        # ---- 2. input transform -> V[b][j][c] (fp16)
        n_v = VT * CV
        for rep in range(0 if spec.xf != "valu" else ceil_div(n_v, NTH)):
            item = b.add(tid, ci(rep * NTH))
            okit = b.cmp_lt(item, ci(n_v))
            j = b.div(item, ci(CV))
            j = b.select(okit, j, ci(VT))  # dummy row
            cc = b.mul(b.mod(item, ci(CV)), ci(VEC))
            rbase = b.add(rrow0, b.select(okit, b.mul(j, ci(m)), c0))
            xs = [
                _raw_to_f32(
                    b,
                    b.smem_load_vN(raw_lds, b.add(rbase, ci(uu)), cc, dtype=F16, n=VEC),
                    VEC,
                )
                for uu in range(n)
            ]
            for bb in range(n):
                vals = []
                for ch in range(VEC):
                    acc = None
                    for uu in range(n):
                        cf = bt[bb][uu]
                        if cf == 0.0:
                            continue
                        xv = xs[uu][ch]
                        if acc is None:
                            acc = (
                                xv
                                if cf == 1.0
                                else (
                                    b.fneg(xv)
                                    if cf == -1.0
                                    else b.fmul(b.const_f32(cf), xv)
                                )
                            )
                        elif cf == 1.0:
                            acc = b.fadd(acc, xv)
                        elif cf == -1.0:
                            acc = b.fsub(acc, xv)
                        else:
                            acc = b.fma(b.const_f32(cf), xv, acc)
                    vals.append(acc if acc is not None else b.const_f32(0.0))
                packed = b.vec_pack([b.cast_f32_to(v_, F16) for v_ in vals], F16)
                b.smem_store_vN(
                    v_lds, [b.add(vb, b.add(j, ci(bb * (VT + 1)))), cc], packed, n=VEC
                )
        if spec.xf == "wmma":
            # V^T[c][(tg, bin)] = sum_p X^T[c][p] * Bt[p][(tg, bin)] for the
            # TPW tiles j0 + tg of a group; p = position - j0 * m.
            TPW = spec.xf_tiles
            NG = spec.xf_groups * (C // 16)
            NR = ceil_div(NG, WAVES)
            XB = spec.xf_batch
            for r0 in range(0, NR, XB):
                rs = list(range(r0, min(NR, r0 + XB)))
                meta, frags = [], []
                for r in rs:  # all transpose reads of the batch first
                    wi = b.add(wave, ci(r * WAVES))
                    okg = b.cmp_lt(wi, ci(NG))
                    wi = b.select(okg, wi, c0)
                    grp = b.div(wi, ci(C // 16))
                    cg = b.mod(wi, ci(C // 16))
                    j0 = b.mul(grp, ci(TPW))
                    prow = b.add(rrow0, b.add(b.mul(j0, ci(m)), tr_row))
                    pcol = b.add(b.mul(cg, ci(16)), tr_col)
                    lo = b.ds_read_tr16_b128(raw_lds, prow, pcol, dtype=F16)
                    hi = b.ds_read_tr16_b128(
                        raw_lds, b.add(prow, ci(8)), pcol, dtype=F16
                    )
                    frags.append(b.vec_concat(lo, hi))
                    meta.append((okg, j0, cg))
                outs = [b.mma(wmma, f_, bt_frag, b.zero_vec_f32(8)) for f_ in frags]
                for (okg, j0, cg), vt_ in zip(meta, outs):
                    # lane: column l16 = (tg, bin), channels cg*16 + lhalf*8 .. +7
                    jj = b.add(j0, xf_tg)
                    okw = b.land(okg, b.land(xf_colok, b.cmp_lt(jj, ci(VT))))
                    vrow = b.select(okw, b.add(b.mul(xf_bin, ci(VT + 1)), jj), ci(VT))
                    vrow = b.add(vrow, vb)
                    b.smem_store_vN(
                        v_lds,
                        [vrow, b.add(b.mul(cg, ci(16)), b.mul(lhalf, ci(8)))],
                        b.vec_cast_f32_to(vt_, F16),
                        n=8,
                    )

    def do_gemm():
        # ---- 3. per-bin GEMMs -> M[b][j][k]
        vb = c0  # V buffer base row
        mb = c0  # M buffer base row
        for slot in range(spec.bpw):
            binv = b.add(wave, ci(WAVES * slot))
            okb = b.cmp_lt(binv, ci(n))
            with b.scf_if(okb):
                vrow0 = b.add(vb, b.mul(b.select(okb, binv, c0), ci(VT + 1)))
                mrow0 = b.select(okb, b.add(mb, b.mul(binv, ci(TB))), ci(MDUMMY))
                accs = [b.zero_vec_f32(8) for _ in range(Q * FK)]
                col = b.mul(lhalf, ci(16))
                for ks in range(C32):
                    vf = []
                    for g in range(Q + S - 1):
                        r_ = b.add(vrow0, b.add(b.mul(l16, ci(Q)), ci(g)))
                        lo = b.smem_load_vN(
                            v_lds, r_, b.add(col, ci(ks * 32)), dtype=F16, n=8
                        )
                        hi = b.smem_load_vN(
                            v_lds, r_, b.add(col, ci(ks * 32 + 8)), dtype=F16, n=8
                        )
                        vf.append(b.vec_concat(lo, hi))
                    for s in range(S):
                        for qq in range(Q):
                            for f in range(FK):
                                i_ = qq * FK + f
                                accs[i_] = b.mma(
                                    wmma, ufr[slot][s][ks][f], vf[qq + s], accs[i_]
                                )
                # lane: tile j = Q*l16 + qq, channels f*16 + lhalf*8 .. +7
                for qq in range(Q):
                    jr = b.add(mrow0, b.add(b.mul(l16, ci(Q)), ci(qq)))
                    for f in range(FK):
                        acc = accs[qq * FK + f]
                        kc = b.add(ci(f * 16), b.mul(lhalf, ci(8)))
                        if MH:
                            b.smem_store_vN(
                                m_lds, [jr, kc], b.vec_cast_f32_to(acc, F16), n=8
                            )
                            continue
                        lo = b.vec_pack([b.vec_extract(acc, e) for e in range(4)], F32)
                        hi = b.vec_pack(
                            [b.vec_extract(acc, e) for e in range(4, 8)], F32
                        )
                        b.smem_store_vN_f32(m_lds, [jr, kc], lo, n=4)
                        b.smem_store_vN_f32(m_lds, [jr, b.add(kc, ci(4))], hi, n=4)

    def do_out(img, a, t0):
        mb = c0  # M buffer base row
        # ---- 4. output transform -> y
        KV = KB // VEC
        n_o = TB * KV
        yimg = b.mul(img, ci(p.Wo))
        for rep in range(ceil_div(n_o, NTH)):
            item = b.add(tid, ci(rep * NTH))
            okit = b.cmp_lt(item, ci(n_o))
            j = b.select(okit, b.div(item, ci(KV)), c0)
            kc = b.mul(b.mod(item, ci(KV)), ci(VEC))
            ms = []
            for bb in range(n):
                r_ = b.add(mb, b.add(ci(bb * TB), j))
                if MH:
                    ms.append(
                        _raw_to_f32(b, b.smem_load_vN(m_lds, r_, kc, dtype=F16, n=8), 8)
                    )
                    continue
                lo = b.smem_load_vN_f32(m_lds, r_, kc, n=4)
                hi = b.smem_load_vN_f32(m_lds, r_, b.add(kc, ci(4)), n=4)
                ms.append(
                    [b.vec_extract(lo, e) for e in range(4)]
                    + [b.vec_extract(hi, e) for e in range(4)]
                )
            tile = b.add(t0, j)
            for i in range(m):
                vals = []
                for ch in range(VEC):
                    acc = None
                    for bb in range(n):
                        cf = at[i][bb]
                        if cf == 0.0:
                            continue
                        mv = ms[bb][ch]
                        if acc is None:
                            acc = mv if cf == 1.0 else b.fmul(b.const_f32(cf), mv)
                        elif cf == 1.0:
                            acc = b.fadd(acc, mv)
                        elif cf == -1.0:
                            acc = b.fsub(acc, mv)
                        else:
                            acc = b.fma(b.const_f32(cf), mv, acc)
                    vals.append(acc if acc is not None else b.const_f32(0.0))
                wo = b.add(a, b.mul(b.add(b.mul(tile, ci(m)), ci(i)), ci(d)))
                ok = b.land(okit, b.cmp_lt(wo, ci(p.Wo)))
                off = b.mul(b.add(b.mul(b.add(yimg, wo), ci(K)), b.add(k0, kc)), ci(2))
                packed = b.vec_pack([b.cast_f32_to(v_, F16) for v_ in vals], F16)
                b.buffer_store_vN_f16(
                    y_rsrc, b.select(ok, off, oob), c0, packed, VEC // 2
                )

    if PF:
        start_raw(first, c0)
        land_raw()
        b.sync_lds_only()

    if spec.merge:
        # Block i: GEMM | wait + barrier | out(i) + xf(i + 1) | barrier.
        do_xf(c0)
        b.sync_lds_only()
        with b.scf_for(c0, n_iter, ci(1), iv_name="it") as it:
            blk = b.add(first, b.mul(it, ci(GPK)))
            img, a, t0 = decode(blk)
            cur = b.mod(it, ci(2))
            more = b.cmp_lt(b.add(it, ci(1)), n_iter)
            start_raw(b.add(blk, ci(GPK)), b.sub(ci(1), cur), more)
            do_gemm()
            land_raw()
            b.sync_lds_only()
            do_out(img, a, t0)
            with b.scf_if(more):
                do_xf(b.mul(b.sub(ci(1), cur), ci(RA)))
            b.sync_lds_only()
        return b.kernel

    with b.scf_for(c0, n_iter, ci(1), iv_name="it") as it:
        blk = b.add(first, b.mul(it, ci(GPK)))
        img, a, t0 = decode(blk)
        cur = b.mod(it, ci(2)) if PF else c0
        rrow0 = b.mul(cur, ci(RA)) if PF else c0

        # ---- 1. raw input -> LDS (positions p = t0*m + pr, pr < RR)
        if not PF:
            for rep in range(ceil_div(n_raw, NTH)):
                item = b.add(tid, ci(rep * NTH))
                off, okit, pr, cv, _, _ = raw_offsets(img, a, t0, item, n_raw)
                val = _load_raw(b, x_rsrc, off, "fp16", VEC)
                prs = b.select(okit, pr, ci(RR))  # out-of-range items -> dummy row
                b.smem_store_vN(raw_lds, [prs, b.mul(cv, ci(VEC))], val, n=VEC)
            b.sync_lds_only()

        do_xf(rrow0)
        b.sync_lds_only()
        if PF:
            start_raw(
                b.add(blk, ci(GPK)),
                b.sub(ci(1), cur),
                b.cmp_lt(b.add(it, ci(1)), n_iter),
            )

        do_gemm()
        b.sync_lds_only()

        do_out(img, a, t0)
        if PF:
            # The next block's raw input has landed (every wave waits for its
            # own copies) and is visible to all waves after the barrier.
            land_raw()
            b.sync_lds_only()
    return b.kernel


__all__ = [
    "FusedWino1DSpec",
    "build_fused_wino_1d",
    "fused_grid",
    "fused_signature",
]
